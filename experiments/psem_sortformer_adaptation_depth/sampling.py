from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio

from experiments.psem_sortformer_adaptation_depth.preflight import (
    SOURCE_MANIFEST_PATH,
    canonical_sha256,
)
from experiments.psem_sortformer_adaptation_depth.receipts import build_data_split_receipt
from experiments.psem_training_strategy_gate.sampling import (
    EVAL_ROLE,
    TRAIN_ROLE,
    RuntimeSession,
    load_runtime_sessions,
)

SAMPLE_RATE_HZ = 16000
FRAME_SAMPLES = 1280
WINDOW_SAMPLES = 480000
WARMUP_SAMPLES = 32000
MAXIMUM_EPOCHS = 1
WINDOWS_PER_EPOCH = 4096
SEEDS = (7301,)
ARMS = ("H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL")
POSITIVE_FAMILIES = (
    "clean_direct_different_speaker_handoff",
    "silence_gap_different_speaker_handoff",
    "overlap_takeover",
)
HARD_NEGATIVE_FAMILIES = (
    "stable_anchor_continuation",
    "same_speaker_silence_gap_resume",
    "overlap_return",
    "overlap_continuation",
)
ROLE_COUNTS = {
    "source_time_uniform": WINDOWS_PER_EPOCH // 2,
    "replacement_positive": WINDOWS_PER_EPOCH // 4,
    "hard_negative": WINDOWS_PER_EPOCH // 4,
}
POSITIVE_TOPOLOGY = {
    "clean_direct_different_speaker_handoff": "clean_direct_different_speaker_handoff",
    "silence_gap_different_speaker_handoff": "silence_gap_different_speaker_handoff",
    "micro_gap_different_speaker_handoff": "silence_gap_different_speaker_handoff",
    "overlap_takeover": "overlap_takeover",
    "micro_overlap_takeover": "overlap_takeover",
    "overlap_gap_takeover": "overlap_takeover",
}
HARD_NEGATIVE_TOPOLOGY = {
    "same_speaker_silence_gap_resume": "same_speaker_silence_gap_resume",
    "micro_gap_same_speaker_resume": "same_speaker_silence_gap_resume",
    "overlap_return": "overlap_return",
    "short_backchannel_return": "overlap_return",
    "micro_overlap_return": "overlap_return",
    "overlap_gap_return": "overlap_return",
}
AUGMENTATION_VERSION = "issue-107-waveform-augmentation-v1"
TARGET_RECIPE_VERSION = "issue-107-oracle-anchor-frame-targets-v1"
OVERFIT_SOURCE_RULE_VERSION = "issue-107-overfit-source-v1"
OVERFIT_WINDOW_RULE_VERSION = "issue-107-overfit-window-v1"


class SamplingContractError(RuntimeError):
    pass


def _source_rows() -> dict[str, dict[str, Any]]:
    return {
        row["source_id"]: row
        for row in (
            json.loads(line)
            for line in SOURCE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        )
    }


def _train_split_binding(sessions: Mapping[str, RuntimeSession]) -> dict[str, Any]:
    receipt = build_data_split_receipt()
    expected = set(receipt["source_ids_by_role"][TRAIN_ROLE])
    observed = set(sessions)
    source_rows = _source_rows()
    if (
        not sessions
        or observed != expected
        or any(key != session.source_id for key, session in sessions.items())
        or any(session.role != TRAIN_ROLE for session in sessions.values())
        or any(
            session.audio_ref != source_rows[key]["audio_ref"]
            or session.waveform_sha256 != source_rows[key]["waveform_sha256"]
            for key, session in sessions.items()
        )
    ):
        raise SamplingContractError("training sessions differ from the exact frozen TRAIN split")
    return {
        "data_split_receipt_sha256": canonical_sha256(receipt),
        "split_manifest_sha256": receipt["artifact_hashes"]["split_manifest"],
        "source_manifest_sha256": receipt["artifact_hashes"]["source_manifest"],
        "train_source_count": len(expected),
    }


def _source_split_binding(session: RuntimeSession) -> dict[str, Any]:
    receipt = build_data_split_receipt()
    expected = set(receipt["source_ids_by_role"][TRAIN_ROLE])
    source_row = _source_rows().get(session.source_id)
    if (
        session.role != TRAIN_ROLE
        or session.source_id not in expected
        or not isinstance(source_row, dict)
        or session.audio_ref != source_row.get("audio_ref")
        or session.waveform_sha256 != source_row.get("waveform_sha256")
    ):
        raise SamplingContractError("waveform source is outside the exact frozen TRAIN split")
    return {
        "data_split_receipt_sha256": canonical_sha256(receipt),
        "split_manifest_sha256": receipt["artifact_hashes"]["split_manifest"],
        "source_manifest_sha256": receipt["artifact_hashes"]["source_manifest"],
        "train_source_count": len(expected),
    }


@dataclass(frozen=True, slots=True)
class WindowCandidate:
    source_id: str
    window_start_sample: int
    family: str
    boundary_sample: int | None

    @property
    def window_end_sample(self) -> int:
        return self.window_start_sample + WINDOW_SAMPLES

    @property
    def identity(self) -> str:
        return f"{self.source_id}:{self.window_start_sample}:{self.family}"


@dataclass(frozen=True, slots=True)
class UniformRange:
    source_id: str
    first_start_sample: int
    count: int


def _digest(key: str, field: str) -> bytes:
    return hashlib.sha256(f"{AUGMENTATION_VERSION}\0{key}\0{field}".encode()).digest()


def _unit(key: str, field: str) -> float:
    return int.from_bytes(_digest(key, field)[:8], "big") / float(2**64)


def _uniform(key: str, field: str, low: float, high: float) -> float:
    return low + (high - low) * _unit(key, field)


def augmentation_decision(row_id: str) -> dict[str, Any]:
    if not row_id:
        raise SamplingContractError("augmentation row identity is required")
    return {
        "recipe_version": AUGMENTATION_VERSION,
        "decision_key": row_id,
        "global_gain": {
            "enabled": True,
            "gain_db": round(_uniform(row_id, "gain_db", -6.0, 6.0), 6),
        },
        "additive_non_speech_noise": {
            "enabled": _unit(row_id, "noise_enabled") < 0.5,
            "snr_db": round(_uniform(row_id, "snr_db", 10.0, 30.0), 6),
            "noise_seed": int.from_bytes(_digest(row_id, "noise_seed")[:8], "big"),
        },
        "light_reverberation": {
            "enabled": _unit(row_id, "reverb_enabled") < 0.3,
            "decay_seconds": round(_uniform(row_id, "reverb_decay_seconds", 0.02, 0.12), 6),
            "reverb_seed": int.from_bytes(_digest(row_id, "reverb_seed")[:8], "big"),
        },
        "band_limitation": {
            "enabled": _unit(row_id, "band_enabled") < 0.3,
            "cutoff_hz": round(_uniform(row_id, "cutoff_hz", 3000.0, 7000.0), 3),
        },
    }


def apply_augmentation(waveform: torch.Tensor, decision: Mapping[str, Any]) -> torch.Tensor:
    key = decision.get("decision_key")
    if not isinstance(key, str) or decision != augmentation_decision(key):
        raise SamplingContractError("augmentation decision is not canonical")
    if waveform.shape != (WINDOW_SAMPLES,) or not torch.is_floating_point(waveform):
        raise SamplingContractError("augmentation requires one complete floating-point sequence")
    result = waveform * (10.0 ** (float(decision["global_gain"]["gain_db"]) / 20.0))
    noise = decision["additive_non_speech_noise"]
    if noise["enabled"]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(noise["noise_seed"]) % (2**63 - 1))
        values = torch.randn(waveform.shape, generator=generator, dtype=torch.float32).to(
            waveform.device, waveform.dtype
        )
        signal_rms = result.square().mean().sqrt().clamp_min(1e-6)
        noise_rms = values.square().mean().sqrt().clamp_min(1e-6)
        target_rms = signal_rms / (10.0 ** (float(noise["snr_db"]) / 20.0))
        result = result + values * (target_rms / noise_rms)
    reverb = decision["light_reverberation"]
    if reverb["enabled"]:
        length = SAMPLE_RATE_HZ // 10
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(reverb["reverb_seed"]) % (2**63 - 1))
        values = torch.randn(length, generator=generator, dtype=torch.float32)
        time = torch.arange(length, dtype=torch.float32) / SAMPLE_RATE_HZ
        impulse = values * torch.exp(-time / float(reverb["decay_seconds"]))
        impulse[0] += 4
        impulse = impulse / impulse.abs().sum().clamp_min(1e-6)
        result = torch.nn.functional.conv1d(
            result[None, None],
            impulse.flip(0).to(result.device, result.dtype)[None, None],
            padding=length - 1,
        )[0, 0, :WINDOW_SAMPLES]
    band = decision["band_limitation"]
    if band["enabled"]:
        result = torchaudio.functional.lowpass_biquad(
            result, SAMPLE_RATE_HZ, float(band["cutoff_hz"])
        )
    if not bool(torch.isfinite(result).all()):
        raise SamplingContractError("augmentation produced non-finite waveform samples")
    return result.clamp(-1, 1)


def _align_up(value: int) -> int:
    return ((value + FRAME_SAMPLES - 1) // FRAME_SAMPLES) * FRAME_SAMPLES


def _align_down(value: int) -> int:
    return (value // FRAME_SAMPLES) * FRAME_SAMPLES


def _window_around(session: RuntimeSession, center: int) -> int | None:
    lower = _align_up(session.labels.intervals[0].start_sample)
    upper = _align_down(session.labels.intervals[-1].end_sample - WINDOW_SAMPLES)
    start = _align_down(center - WINDOW_SAMPLES // 2)
    if lower <= start <= upper:
        return start
    return None


def _transition_center(transition: Mapping[str, Any], session: RuntimeSession) -> int | None:
    sample = transition.get("handoff_source_sample")
    if isinstance(sample, int):
        return sample
    index = transition.get("to_interval_index")
    if isinstance(index, int) and 0 <= index < len(session.labels.intervals):
        return session.labels.intervals[index].start_sample
    return None


def _window_fits_slots(session: RuntimeSession, window_start_sample: int) -> bool:
    from experiments.psem_sortformer_adaptation_depth.supervision import (
        window_fits_arrival_order_slots,
    )

    return window_fits_arrival_order_slots(session.labels, window_start_sample)


def candidate_pools(
    sessions: Mapping[str, RuntimeSession],
) -> dict[str, tuple[WindowCandidate, ...]]:
    pools: dict[str, dict[str, WindowCandidate]] = {
        family: {} for family in (*POSITIVE_FAMILIES, *HARD_NEGATIVE_FAMILIES)
    }
    eligibility: dict[tuple[str, int], bool] = {}

    def eligible(source_id: str, session: RuntimeSession, start: int) -> bool:
        key = (source_id, start)
        if key not in eligibility:
            eligibility[key] = _window_fits_slots(session, start)
        return eligibility[key]

    for source_id, session in sorted(sessions.items()):
        if session.role != TRAIN_ROLE:
            raise SamplingContractError("sampling accepts TRAIN sessions only")
        for transition in session.labels.transitions:
            if transition.get("mask_state") != "valid":
                continue
            topology = str(transition.get("primary_topology"))
            target = transition.get("handoff_confirmed")
            family = (
                POSITIVE_TOPOLOGY.get(topology)
                if target == 1
                else HARD_NEGATIVE_TOPOLOGY.get(topology)
                if target == 0
                else None
            )
            center = _transition_center(transition, session)
            if family is None or center is None:
                continue
            start = _window_around(session, center)
            if start is None or not eligible(source_id, session, start):
                continue
            candidate = WindowCandidate(source_id, start, family, center)
            pools[family][candidate.identity] = candidate
        for interval, activity in zip(
            session.labels.intervals, session.labels.activity_labels, strict=True
        ):
            if (
                activity.get("mask_state") != "valid"
                or interval.ambiguous
                or not interval.speaker_identity_known
            ):
                continue
            state = activity.get("state")
            family = {
                "singleton": "stable_anchor_continuation",
                "overlap": "overlap_continuation",
            }.get(state)
            if family is None:
                continue
            center = (interval.start_sample + interval.end_sample) // 2
            start = _window_around(session, center)
            if start is None or not eligible(source_id, session, start):
                continue
            candidate = WindowCandidate(source_id, start, family, None)
            pools[family][candidate.identity] = candidate
    result = {
        family: tuple(values[key] for key in sorted(values)) for family, values in pools.items()
    }
    missing = [family for family, values in result.items() if not values]
    if missing:
        raise SamplingContractError(f"mandatory sampling pools are empty: {missing}")
    return result


def uniform_ranges(sessions: Mapping[str, RuntimeSession]) -> tuple[UniformRange, ...]:
    ranges = []
    for source_id, session in sorted(sessions.items()):
        if session.role != TRAIN_ROLE:
            raise SamplingContractError("uniform sampling accepts TRAIN sessions only")
        first = _align_up(session.labels.intervals[0].start_sample)
        last = _align_down(session.labels.intervals[-1].end_sample - WINDOW_SAMPLES)
        if last < first:
            continue
        ranges.append(UniformRange(source_id, first, (last - first) // FRAME_SAMPLES + 1))
    if not ranges:
        raise SamplingContractError("no TRAIN source contains a complete 30-second window")
    return tuple(ranges)


def _quota(total: int, families: Sequence[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(families))
    return {family: base + int(index < remainder) for index, family in enumerate(families)}


def _ordered(values: Sequence[WindowCandidate], family: str) -> tuple[WindowCandidate, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: hashlib.sha256(
                f"issue-107-sampling-v1\0{family}\0{value.identity}".encode()
            ).digest(),
        )
    )


def _uniform_candidate(
    ranges: Sequence[UniformRange], ordinal: int, total_ordinals: int
) -> WindowCandidate:
    total = sum(value.count for value in ranges)
    if total <= 0 or total_ordinals > total:
        raise SamplingContractError("uniform grid cannot provide unique shared windows")
    offset = (
        int.from_bytes(hashlib.sha256(b"issue-107-uniform-offset-v1").digest()[:8], "big") % total
    )
    step = int.from_bytes(hashlib.sha256(b"issue-107-uniform-step-v1").digest()[:8], "big") % total
    step = max(step, 1)
    while math.gcd(step, total) != 1:
        step += 1
        if step == total:
            step = 1
    index = (offset + ordinal * step) % total
    for value in ranges:
        if index < value.count:
            start = value.first_start_sample + index * FRAME_SAMPLES
            return WindowCandidate(value.source_id, start, "source_time_uniform", None)
        index -= value.count
    raise SamplingContractError("uniform grid index was not resolved")


def _uniform_candidates(
    ranges: Sequence[UniformRange],
    base: int,
    count: int,
    total_ordinals: int,
    sessions: Mapping[str, RuntimeSession] | None,
) -> tuple[WindowCandidate, ...]:
    if sessions is None:
        return tuple(
            _uniform_candidate(ranges, base + index, total_ordinals) for index in range(count)
        )
    total = sum(value.count for value in ranges)
    selected: list[WindowCandidate] = []
    ordinal = base
    while ordinal < total and len(selected) < count:
        candidate = _uniform_candidate(ranges, ordinal, total_ordinals)
        session = sessions.get(candidate.source_id)
        if (
            session is not None
            and session.role == TRAIN_ROLE
            and _window_fits_slots(session, candidate.window_start_sample)
        ):
            selected.append(candidate)
        ordinal += 1
    if len(selected) != count:
        raise SamplingContractError("uniform grid cannot provide enough four-slot training windows")
    return tuple(selected)


def epoch_plan(
    pools: Mapping[str, Sequence[WindowCandidate]],
    ranges: Sequence[UniformRange],
    epoch: int,
    sessions: Mapping[str, RuntimeSession] | None = None,
) -> tuple[tuple[str, WindowCandidate], ...]:
    if not 1 <= epoch <= MAXIMUM_EPOCHS:
        raise SamplingContractError("epoch lies outside the frozen training recipe")
    selected: list[tuple[str, WindowCandidate]] = []
    for role, families in (
        ("replacement_positive", POSITIVE_FAMILIES),
        ("hard_negative", HARD_NEGATIVE_FAMILIES),
    ):
        quotas = _quota(ROLE_COUNTS[role], families)
        for family in families:
            ordered = _ordered(pools[family], family)
            count = quotas[family]
            offset = (epoch - 1) * count
            selected.extend(
                (role, ordered[(offset + index) % len(ordered)]) for index in range(count)
            )
    uniform_count = ROLE_COUNTS["source_time_uniform"]
    base = (epoch - 1) * uniform_count
    selected.extend(
        ("source_time_uniform", candidate)
        for candidate in _uniform_candidates(
            ranges,
            base,
            uniform_count,
            MAXIMUM_EPOCHS * uniform_count,
            sessions,
        )
    )
    selected.sort(
        key=lambda item: hashlib.sha256(
            f"issue-107-epoch-order-v1\0{epoch}\0{item[0]}\0{item[1].identity}".encode()
        ).digest()
    )
    if len(selected) != WINDOWS_PER_EPOCH:
        raise SamplingContractError("epoch mixture differs from the frozen manifest recipe")
    return tuple(selected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_sampling_manifest(
    sessions: Mapping[str, RuntimeSession], output_path: Path
) -> dict[str, Any]:
    split_binding = _train_split_binding(sessions)
    source_rows = _source_rows()
    pools = candidate_pools(sessions)
    ranges = uniform_ranges(sessions)
    label_hashes = {
        source_id: canonical_sha256(session.labels.to_dict())
        for source_id, session in sessions.items()
    }
    plans = {
        epoch: epoch_plan(pools, ranges, epoch, sessions) for epoch in range(1, MAXIMUM_EPOCHS + 1)
    }
    role_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for epoch in range(1, MAXIMUM_EPOCHS + 1):
            for epoch_index, (role, candidate) in enumerate(plans[epoch]):
                row_id = f"epoch-{epoch:02d}-window-{epoch_index:04d}"
                target_identity = {
                    "recipe_version": TARGET_RECIPE_VERSION,
                    "label_result_sha256": label_hashes[candidate.source_id],
                    "window_start_sample": candidate.window_start_sample,
                    "window_end_sample": candidate.window_end_sample,
                    "frame_samples": FRAME_SAMPLES,
                    "warmup_samples": WARMUP_SAMPLES,
                }
                row = {
                    "schema_version": 1,
                    "artifact_role": "psem_sortformer_shared_training_window",
                    "row_id": row_id,
                    "epoch": epoch,
                    "epoch_index": epoch_index,
                    "split_role": TRAIN_ROLE,
                    "source_id": candidate.source_id,
                    "corpus": source_rows[candidate.source_id]["corpus"],
                    "source_waveform_sha256": sessions[candidate.source_id].waveform_sha256,
                    **split_binding,
                    "window_start_sample": candidate.window_start_sample,
                    "window_end_sample": candidate.window_end_sample,
                    "loss_start_sample": candidate.window_start_sample + WARMUP_SAMPLES,
                    "state_reset_at_window_start": True,
                    "sampling_role": role,
                    "topology_family": candidate.family,
                    "boundary_sample": candidate.boundary_sample,
                    "target_identity": target_identity,
                    "target_identity_sha256": canonical_sha256(target_identity),
                    "augmentation": augmentation_decision(row_id),
                }
                row["augmentation_identity_sha256"] = canonical_sha256(row["augmentation"])
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                role_counts[role] += 1
                family_counts[candidate.family] += 1
    temporary.replace(output_path)
    expected_roles = Counter({role: count * MAXIMUM_EPOCHS for role, count in ROLE_COUNTS.items()})
    if role_counts != expected_roles:
        raise SamplingContractError("materialized mixture differs from the frozen recipe")
    return {
        "schema_version": 1,
        "artifact_role": "sampling_manifest_receipt",
        "manifest_path": str(output_path.resolve()),
        "manifest_sha256": _sha256_file(output_path),
        "row_count": sum(role_counts.values()),
        "epoch_count": MAXIMUM_EPOCHS,
        "windows_per_epoch": WINDOWS_PER_EPOCH,
        "window_samples": WINDOW_SAMPLES,
        "warmup_samples": WARMUP_SAMPLES,
        "frame_samples": FRAME_SAMPLES,
        "sampling_role_counts": dict(sorted(role_counts.items())),
        "topology_family_counts": dict(sorted(family_counts.items())),
        "pool_counts": {key: len(value) for key, value in sorted(pools.items())},
        "source_count": len(sessions),
        "split_roles": [TRAIN_ROLE],
        "eval_source_count": 0,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "shared_window_target_and_augmentation_manifest": True,
        **split_binding,
        "augmentation_families": [
            "global_gain",
            "additive_non_speech_noise",
            "light_reverberation",
            "band_limitation",
        ],
        "target_recipe_version": TARGET_RECIPE_VERSION,
        "overfit_source_rule_version": OVERFIT_SOURCE_RULE_VERSION,
        "overfit_window_rule_version": OVERFIT_WINDOW_RULE_VERSION,
    }


def load_sampling_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SamplingContractError("sampling manifest rows must be JSON objects")
        rows.append(value)
    return rows


def select_overfit_rows(
    rows: Sequence[Mapping[str, Any]], corpus_by_source: Mapping[str, str]
) -> tuple[dict[str, Any], ...]:
    raise SamplingContractError("legacy overfit selection is not supported")


def _legacy_select_overfit_rows(
    rows: Sequence[Mapping[str, Any]], corpus_by_source: Mapping[str, str]
) -> tuple[dict[str, Any], ...]:
    sources_by_corpus = {
        corpus: sorted(
            {
                str(row["source_id"])
                for row in rows
                if row.get("split_role") == TRAIN_ROLE
                and corpus_by_source.get(str(row.get("source_id"))) == corpus
            },
            key=lambda source_id: hashlib.sha256(
                f"{OVERFIT_SOURCE_RULE_VERSION}\0{corpus}\0{source_id}".encode()
            ).digest(),
        )
        for corpus in ("AMI", "AliMeeting")
    }
    if any(len(values) < 2 for values in sources_by_corpus.values()):
        raise SamplingContractError("overfit selection requires two TRAIN sources per corpus")
    selected_sources = {
        source_id for values in sources_by_corpus.values() for source_id in values[:2]
    }
    selected = []
    for source_id in sorted(selected_sources):
        unique = {}
        for row in rows:
            if row.get("source_id") != source_id or row.get("split_role") != TRAIN_ROLE:
                continue
            identity = (row.get("window_start_sample"), row.get("window_end_sample"))
            unique.setdefault(identity, dict(row))
        ordered = sorted(
            unique.values(),
            key=lambda row: hashlib.sha256(
                f"{OVERFIT_WINDOW_RULE_VERSION}\0{source_id}\0{row['window_start_sample']}".encode()
            ).digest(),
        )
        if len(ordered) < 15:
            raise SamplingContractError(
                f"overfit source has fewer than 15 unique windows: {source_id}"
            )
        selected.extend(ordered[:15])
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{OVERFIT_WINDOW_RULE_VERSION}\0final\0{row['source_id']}\0{row['window_start_sample']}".encode()
        ).digest()
    )
    if len(selected) != 60:
        raise SamplingContractError("overfit selection differs from the 30-minute budget")
    return tuple(selected)


def validate_sampling_manifest(
    path: Path, sessions: Mapping[str, RuntimeSession]
) -> dict[str, Any]:
    split_binding = _train_split_binding(sessions)
    source_rows = _source_rows()
    rows = load_sampling_rows(path)
    if len(rows) != WINDOWS_PER_EPOCH or any(row.get("epoch") != 1 for row in rows):
        raise SamplingContractError(
            "sampling manifest must contain exactly one epoch-1 TRAIN manifest"
        )
    pools = candidate_pools(sessions)
    ranges = uniform_ranges(sessions)
    label_hashes = {
        source_id: canonical_sha256(session.labels.to_dict())
        for source_id, session in sessions.items()
    }
    plans = {
        epoch: epoch_plan(pools, ranges, epoch, sessions) for epoch in range(1, MAXIMUM_EPOCHS + 1)
    }
    role_counts: Counter[str] = Counter()
    for absolute_index, row in enumerate(rows):
        epoch = absolute_index // WINDOWS_PER_EPOCH + 1
        epoch_index = absolute_index % WINDOWS_PER_EPOCH
        role, candidate = plans[epoch][epoch_index]
        row_id = f"epoch-{epoch:02d}-window-{epoch_index:04d}"
        target_identity = {
            "recipe_version": TARGET_RECIPE_VERSION,
            "label_result_sha256": label_hashes[candidate.source_id],
            "window_start_sample": candidate.window_start_sample,
            "window_end_sample": candidate.window_end_sample,
            "frame_samples": FRAME_SAMPLES,
            "warmup_samples": WARMUP_SAMPLES,
        }
        expected = {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_shared_training_window",
            "row_id": row_id,
            "epoch": epoch,
            "epoch_index": epoch_index,
            "split_role": TRAIN_ROLE,
            "source_id": candidate.source_id,
            "corpus": source_rows[candidate.source_id]["corpus"],
            "source_waveform_sha256": sessions[candidate.source_id].waveform_sha256,
            **split_binding,
            "window_start_sample": candidate.window_start_sample,
            "window_end_sample": candidate.window_end_sample,
            "loss_start_sample": candidate.window_start_sample + WARMUP_SAMPLES,
            "state_reset_at_window_start": True,
            "sampling_role": role,
            "topology_family": candidate.family,
            "boundary_sample": candidate.boundary_sample,
            "target_identity": target_identity,
            "target_identity_sha256": canonical_sha256(target_identity),
            "augmentation": augmentation_decision(row_id),
        }
        expected["augmentation_identity_sha256"] = canonical_sha256(expected["augmentation"])
        if row != expected:
            raise SamplingContractError(f"sampling manifest row is not canonical: {row_id}")
        role_counts[role] += 1
    return {
        "passed": True,
        "manifest_sha256": _sha256_file(path),
        "row_count": len(rows),
        "sampling_role_counts": dict(sorted(role_counts.items())),
        "eval_source_count": 0,
        "shared_window_target_and_augmentation_manifest": True,
        **split_binding,
    }


def load_window_waveform(
    row: Mapping[str, Any],
    session: RuntimeSession,
    corpus_root: Path,
) -> torch.Tensor:
    split_binding = _source_split_binding(session)
    target_identity = row.get("target_identity")
    augmentation = row.get("augmentation")
    if (
        row.get("split_role") != TRAIN_ROLE
        or row.get("source_id") != session.source_id
        or session.role != TRAIN_ROLE
        or row.get("source_waveform_sha256") != session.waveform_sha256
        or any(row.get(key) != value for key, value in split_binding.items())
        or not isinstance(target_identity, Mapping)
        or row.get("target_identity_sha256") != canonical_sha256(target_identity)
        or not isinstance(augmentation, Mapping)
        or row.get("augmentation_identity_sha256") != canonical_sha256(augmentation)
    ):
        raise SamplingContractError("waveform request differs from the TRAIN manifest identity")
    start = row.get("window_start_sample")
    end = row.get("window_end_sample")
    if not isinstance(start, int) or end != start + WINDOW_SAMPLES:
        raise SamplingContractError("waveform request is not one complete 30-second sequence")
    relative = Path(session.audio_ref)
    root = corpus_root.resolve()
    path = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.is_relative_to(root)
        or not path.is_file()
        or _sha256_file(path) != session.waveform_sha256
    ):
        raise SamplingContractError("waveform path escapes the bound corpus root")
    waveform, sample_rate = torchaudio.load(
        path,
        frame_offset=start,
        num_frames=WINDOW_SAMPLES,
    )
    if sample_rate != SAMPLE_RATE_HZ or waveform.shape != (1, WINDOW_SAMPLES):
        raise SamplingContractError("waveform geometry differs from the frozen mono 16 kHz input")
    expected_target = {
        "recipe_version": TARGET_RECIPE_VERSION,
        "label_result_sha256": canonical_sha256(session.labels.to_dict()),
        "window_start_sample": start,
        "window_end_sample": end,
        "frame_samples": FRAME_SAMPLES,
        "warmup_samples": WARMUP_SAMPLES,
    }
    if target_identity != expected_target or augmentation != augmentation_decision(
        str(row.get("row_id"))
    ):
        raise SamplingContractError("window target or augmentation identity is not canonical")
    return apply_augmentation(waveform[0], augmentation)


def load_training_sessions(corpus_root: Path, reference_root: Path) -> dict[str, RuntimeSession]:
    sessions = load_runtime_sessions(corpus_root, reference_root, roles=(TRAIN_ROLE,))
    _train_split_binding(sessions)
    if any(session.role == EVAL_ROLE for session in sessions.values()):
        raise SamplingContractError("EVAL entered the training session loader")
    return sessions
