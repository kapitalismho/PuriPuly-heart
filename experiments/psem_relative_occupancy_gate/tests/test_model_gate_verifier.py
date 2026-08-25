from __future__ import annotations

import pytest

from experiments.psem_relative_occupancy_gate.io_utils import canonical_sha256
from experiments.psem_relative_occupancy_gate.verify_model_gates import (
    ModelGateVerificationError,
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
