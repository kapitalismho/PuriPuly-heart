from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from experiments.psem_sortformer_adaptation_depth.nemo_adapter import SortformerEvidence

FRAME_SAMPLES = 1280
EVIDENCE_DELAY_SAMPLES = 16640


class PredictionContractError(RuntimeError):
    pass


def prediction_rows(
    source_id: str,
    source_start_sample: int,
    evidence: SortformerEvidence,
    psem_outputs: dict[str, torch.Tensor],
    anchor_episode_ids: Sequence[str],
    oracle_anchor_slots: Sequence[int],
) -> list[dict[str, Any]]:
    if evidence.probabilities.shape[0] != 1:
        raise PredictionContractError("prediction serialization requires one source at a time")
    frame_count = evidence.probabilities.shape[1]
    expected_shapes = {
        "probabilities": (1, frame_count, 4),
        "activity_logits": (1, frame_count, 4),
        "final_temporal_hidden": (1, frame_count, 192),
        "slot_alive": (1, frame_count, 4),
        "state_reset": (1, frame_count, 1),
        "evidence_delay_seconds": (1, frame_count, 1),
    }
    observed_shapes = {field: tuple(getattr(evidence, field).shape) for field in expected_shapes}
    if observed_shapes != expected_shapes:
        raise PredictionContractError(f"evidence tensor geometry is invalid: {observed_shapes}")
    evidence_tensors = (
        evidence.probabilities,
        evidence.activity_logits,
        evidence.final_temporal_hidden,
        evidence.slot_alive,
        evidence.state_reset,
        evidence.evidence_delay_seconds,
    )
    if not all(bool(torch.isfinite(value).all()) for value in evidence_tensors):
        raise PredictionContractError("prediction evidence contains non-finite values")
    expected_reset = torch.zeros_like(evidence.state_reset, dtype=torch.bool)
    if frame_count > 0:
        expected_reset[:, 0, 0] = True
    if (
        not bool(((evidence.probabilities >= 0) & (evidence.probabilities <= 1)).all())
        or not bool(((evidence.slot_alive == 0) | (evidence.slot_alive == 1)).all())
        or not bool((evidence.slot_alive == 1).all())
        or not bool(((evidence.state_reset == 0) | (evidence.state_reset == 1)).all())
        or not torch.equal(evidence.state_reset.to(torch.bool), expected_reset)
        or not bool((evidence.evidence_delay_seconds == 1.04).all())
    ):
        raise PredictionContractError(
            "posterior, lifecycle, or evidence-delay semantics are invalid"
        )
    if (
        psem_outputs.get("anchor_present") is None
        or psem_outputs.get("replacement_evidence") is None
        or tuple(psem_outputs["anchor_present"].shape) != (1, frame_count)
        or tuple(psem_outputs["replacement_evidence"].shape) != (1, frame_count)
        or len(anchor_episode_ids) != frame_count
        or len(oracle_anchor_slots) != frame_count
    ):
        raise PredictionContractError("PSEM output or oracle episode geometry is invalid")
    if not all(
        bool(torch.isfinite(psem_outputs[key]).all())
        for key in ("anchor_present", "replacement_evidence")
    ):
        raise PredictionContractError("PSEM output contains non-finite logits")
    if source_start_sample < 0 or source_start_sample % FRAME_SAMPLES:
        raise PredictionContractError("source frame origin is invalid")
    if not source_id or any(
        value is not None and (not isinstance(value, str) or not value)
        for value in anchor_episode_ids
    ):
        raise PredictionContractError("source or anchor episode identity is invalid")
    rows = []
    for index in range(frame_count):
        slot = oracle_anchor_slots[index]
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < 4:
            raise PredictionContractError("oracle anchor slot lies outside the four-slot graph")
        start = source_start_sample + index * FRAME_SAMPLES
        delay_samples = round(float(evidence.evidence_delay_seconds[0, index, 0]) * 16000)
        if delay_samples != EVIDENCE_DELAY_SAMPLES or not math.isfinite(float(delay_samples)):
            raise PredictionContractError("saved prediction carries an altered evidence delay")
        rows.append(
            {
                "schema_version": 1,
                "artifact_role": "psem_sortformer_frame_prediction",
                "source_id": source_id,
                "source_frame_start_sample": start,
                "source_frame_end_sample": start + FRAME_SAMPLES,
                "model_evidence_frontier_source_sample": start + delay_samples,
                "anchor_episode_id": anchor_episode_ids[index],
                "oracle_anchor_slot": slot,
                "slot_alive": [bool(value) for value in evidence.slot_alive[0, index].tolist()],
                "state_reset": bool(evidence.state_reset[0, index, 0]),
                "raw_sortformer_activity_logits": [
                    float(value) for value in evidence.activity_logits[0, index].detach().cpu()
                ],
                "raw_anchor_present_logit": float(
                    psem_outputs["anchor_present"][0, index].detach().cpu()
                ),
                "raw_replacement_evidence_logit": float(
                    psem_outputs["replacement_evidence"][0, index].detach().cpu()
                ),
            }
        )
    return rows
