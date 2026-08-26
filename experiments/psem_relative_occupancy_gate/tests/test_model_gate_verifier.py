from __future__ import annotations

import pytest

from experiments.psem_relative_occupancy_gate.contracts import ActivityInterval
from experiments.psem_relative_occupancy_gate.io_utils import canonical_sha256
from experiments.psem_relative_occupancy_gate.verify_model_gates import (
    ModelGateVerificationError,
    _causal_opportunity_replay,
    _validate_replacement_event,
    _validate_row_hash,
)


def test_model_gate_verifier_rejects_event_ledger_tamper() -> None:
    row = {"gate": "gate1_oracle_anchor", "source_id": "source"}
    row["row_sha256"] = canonical_sha256(row)
    _validate_row_hash(row)
    row["source_id"] = "forged"
    with pytest.raises(ModelGateVerificationError, match="row hash"):
        _validate_row_hash(row)


def test_model_gate_verifier_rejects_noncausal_event_timing() -> None:
    event = {
        "source_id": "source",
        "anchor_episode_id": "E1",
        "boundary_source_sample": 2000,
        "model_evidence_frontier_sample": 1500,
        "decoder_emit_sample": 3000,
        "confirmation_samples": 1600,
    }
    with pytest.raises(ModelGateVerificationError, match="timing contract"):
        _validate_replacement_event(
            event,
            source_id="source",
            source_end=4000,
            confirmation_samples=1600,
        )


def test_model_gate_verifier_replays_every_lifecycle_and_latest_alignment() -> None:
    manifest = {
        "source_id": "source",
        "scored_start_sample": 0,
        "scored_end_sample": 16000,
        "intervals": [
            value.to_dict()
            for value in (
                ActivityInterval(0, 3200, ("A",), False),
                ActivityInterval(3200, 6400, (), False),
                ActivityInterval(6400, 9600, ("B",), False),
                ActivityInterval(9600, 12800, (), False),
                ActivityInterval(12800, 16000, ("C",), False),
            )
        ],
    }
    row = {
        "annotated_episodes": [
            {
                "episode_id": "E1",
                "anchor_emit_sample": 16000,
                "end_emit_sample": 16000,
                "expected_anchor_speaker": "C",
                "opportunity_start_sample": 12800,
            }
        ]
    }
    cfg = {
        "gate0_enrollment_confirm_ms": 100,
        "lifecycle_proxy_silence_reset_ms": 100,
    }
    replayed = _causal_opportunity_replay(row, manifest, cfg)
    assert len(replayed) == 3
    assert replayed[-1]["matched_anchor_episode_id"] == "E1"
    row["annotated_episodes"][0]["expected_anchor_speaker"] = "A"
    with pytest.raises(ModelGateVerificationError, match="latest lifecycle"):
        _causal_opportunity_replay(row, manifest, cfg)
