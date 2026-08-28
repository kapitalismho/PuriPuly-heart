from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from experiments.psem_training_strategy_gate.data.label_contract import LabelResult

FRAME_SAMPLES = 1280
WINDOW_SAMPLES = 480000
FRAME_COUNT = WINDOW_SAMPLES // FRAME_SAMPLES
WARMUP_FRAME_COUNT = 32000 // FRAME_SAMPLES
SLOT_COUNT = 4


class SupervisionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FrameSupervision:
    anchor_targets: torch.Tensor
    replacement_targets: torch.Tensor
    psem_mask: torch.Tensor
    arrival_order_targets: torch.Tensor
    native_mask: torch.Tensor
    arrival_order_speakers: tuple[str, ...]
    mapping_anchor_active: torch.Tensor
    anchor_episode_ids: tuple[str | None, ...]


def build_frame_supervision(
    labels: LabelResult,
    window_start_sample: int,
    anchor_episode_ids: tuple[str | None, ...],
    anchor_speakers: tuple[str | None, ...],
) -> FrameSupervision:
    if (
        window_start_sample % FRAME_SAMPLES
        or len(anchor_episode_ids) != FRAME_COUNT
        or len(anchor_speakers) != FRAME_COUNT
    ):
        raise SupervisionError("frame supervision geometry differs from the frozen recipe")
    window_end = window_start_sample + WINDOW_SAMPLES
    if (
        window_start_sample < labels.intervals[0].start_sample
        or window_end > labels.intervals[-1].end_sample
    ):
        raise SupervisionError("training window lies outside the normalized source timeline")
    interval_index = 0
    while labels.intervals[interval_index].end_sample <= window_start_sample:
        interval_index += 1
    active_by_frame: list[tuple[str, ...]] = []
    valid_by_frame: list[bool] = []
    arrival_order: list[str] = []
    for frame_index in range(FRAME_COUNT):
        center = window_start_sample + frame_index * FRAME_SAMPLES + FRAME_SAMPLES // 2
        while labels.intervals[interval_index].end_sample <= center:
            interval_index += 1
        interval = labels.intervals[interval_index]
        activity = labels.activity_labels[interval_index]
        if not interval.start_sample <= center < interval.end_sample:
            raise SupervisionError("normalized intervals do not cover the native frame grid")
        valid = bool(
            activity.get("mask_state") == "valid"
            and not interval.ambiguous
            and interval.speaker_identity_known
        )
        active = interval.active_speakers if valid else ()
        active_by_frame.append(active)
        valid_by_frame.append(valid)
        for speaker in active:
            if speaker not in arrival_order:
                arrival_order.append(speaker)
    if len(arrival_order) > SLOT_COUNT:
        raise SupervisionError(
            "a 30-second sequence contains more than four arrival-order speakers"
        )
    slot_by_speaker = {speaker: index for index, speaker in enumerate(arrival_order)}
    anchor_targets = torch.zeros(FRAME_COUNT, dtype=torch.float32)
    replacement_targets = torch.zeros(FRAME_COUNT, dtype=torch.float32)
    psem_mask = torch.zeros(FRAME_COUNT, dtype=torch.float32)
    native_targets = torch.zeros((FRAME_COUNT, SLOT_COUNT), dtype=torch.float32)
    native_mask = torch.zeros(FRAME_COUNT, dtype=torch.bool)
    mapping_anchor_active = torch.zeros(FRAME_COUNT, dtype=torch.bool)
    for frame_index, (active, valid, episode_id, anchor_speaker) in enumerate(
        zip(
            active_by_frame,
            valid_by_frame,
            anchor_episode_ids,
            anchor_speakers,
            strict=True,
        )
    ):
        if valid and frame_index >= WARMUP_FRAME_COUNT:
            native_mask[frame_index] = True
            for speaker in active:
                native_targets[frame_index, slot_by_speaker[speaker]] = 1
        if valid and episode_id is not None and anchor_speaker is not None:
            anchor_active = anchor_speaker in active
            mapping_anchor_active[frame_index] = anchor_active
        if (
            valid
            and frame_index >= WARMUP_FRAME_COUNT
            and episode_id is not None
            and anchor_speaker is not None
        ):
            psem_mask[frame_index] = 1
            anchor_active = anchor_speaker in active
            anchor_targets[frame_index] = float(anchor_active)
            replacement_targets[frame_index] = float(bool(active) and not anchor_active)
    return FrameSupervision(
        anchor_targets=anchor_targets,
        replacement_targets=replacement_targets,
        psem_mask=psem_mask,
        arrival_order_targets=native_targets,
        native_mask=native_mask,
        arrival_order_speakers=tuple(arrival_order),
        mapping_anchor_active=mapping_anchor_active,
        anchor_episode_ids=anchor_episode_ids,
    )


def anchor_timeline(
    source_id: str, labels: LabelResult, window_start_sample: int
) -> tuple[tuple[str | None, ...], tuple[str | None, ...]]:
    from experiments.psem_frozen_ceiling_gate.experiment_support import simulate_gt_session

    window_end = window_start_sample + WINDOW_SAMPLES
    intervals = []
    for interval, activity in zip(labels.intervals, labels.activity_labels, strict=True):
        start = max(interval.start_sample, window_start_sample)
        end = min(interval.end_sample, window_end)
        if end <= start:
            continue
        intervals.append(
            {
                "start_sample": start,
                "end_sample": end,
                "active_speakers": list(interval.active_speakers),
                "masked": bool(
                    activity.get("mask_state") != "valid"
                    or interval.ambiguous
                    or not interval.speaker_identity_known
                ),
            }
        )
    if (
        not intervals
        or intervals[0]["start_sample"] != window_start_sample
        or intervals[-1]["end_sample"] != window_end
    ):
        raise SupervisionError("window anchor timeline is not fully covered")
    reference = simulate_gt_session(
        {"source_id": source_id, "intervals": intervals},
        replacement_confirmation_samples=8000,
        enrollment_samples=3200,
        silence_reset_samples=19200,
    )
    episode_ids: list[str | None] = []
    speakers: list[str | None] = []
    episode_index = 0
    for frame_index in range(FRAME_COUNT):
        center = window_start_sample + frame_index * FRAME_SAMPLES + FRAME_SAMPLES // 2
        while (
            episode_index < len(reference.episodes)
            and reference.episodes[episode_index].end_emit_sample <= center
        ):
            episode_index += 1
        if episode_index < len(reference.episodes):
            episode = reference.episodes[episode_index]
            if episode.anchor_emit_sample <= center < episode.end_emit_sample:
                episode_ids.append(episode.episode_id)
                speakers.append(episode.anchor_speaker)
                continue
        episode_ids.append(None)
        speakers.append(None)
    return tuple(episode_ids), tuple(speakers)


def oracle_mapping_from_frames(
    probabilities: torch.Tensor,
    slot_alive: torch.Tensor,
    supervision: FrameSupervision,
) -> dict[str, int]:
    if probabilities.shape != (FRAME_COUNT, SLOT_COUNT) or slot_alive.shape != probabilities.shape:
        raise SupervisionError("oracle mapping posterior geometry is invalid")
    result = {}
    for episode_id in sorted(
        {value for value in supervision.anchor_episode_ids if value is not None}
    ):
        episode_mask = torch.tensor(
            [value == episode_id for value in supervision.anchor_episode_ids],
            dtype=torch.bool,
            device=probabilities.device,
        )
        support = episode_mask & supervision.mapping_anchor_active.to(probabilities.device)
        if not bool(support.any()):
            raise SupervisionError(
                f"oracle episode has no valid anchor-active support: {episode_id}"
            )
        scores = (probabilities[support] * slot_alive[support].to(probabilities.dtype)).mean(dim=0)
        result[episode_id] = int(torch.argmax(scores))
    if not result:
        raise SupervisionError("window has no mappable oracle anchor episode")
    return result


def oracle_anchor_one_hot(
    anchor_episode_ids: tuple[str, ...],
    slot_by_episode: dict[str, int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    slots = []
    for episode_id in anchor_episode_ids:
        slot = slot_by_episode.get(episode_id)
        if not isinstance(slot, int) or not 0 <= slot < SLOT_COUNT:
            raise SupervisionError(f"oracle mapping is absent or invalid: {episode_id}")
        slots.append(slot)
    result = torch.zeros((len(slots), SLOT_COUNT), dtype=dtype, device=device)
    result[
        torch.arange(len(slots), device=result.device), torch.tensor(slots, device=result.device)
    ] = 1
    return result


def supervision_identity(supervision: FrameSupervision) -> dict[str, Any]:
    return {
        "frame_count": FRAME_COUNT,
        "warmup_frame_count": WARMUP_FRAME_COUNT,
        "arrival_order_speakers": list(supervision.arrival_order_speakers),
        "psem_valid_frame_count": int(supervision.psem_mask.sum()),
        "native_valid_frame_count": int(supervision.native_mask.sum()),
        "oracle_episode_count": len(
            {value for value in supervision.anchor_episode_ids if value is not None}
        ),
    }
