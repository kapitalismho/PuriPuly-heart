from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies import (
    SimplificationError,
    _global_overlap_records,
    _unique_index,
)
from experiments.psem_relative_occupancy_gate.contracts import EvaluationCell
from experiments.psem_relative_occupancy_gate.model_decode import PosteriorCell

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_trace_reuse_receipt_forbids_new_inference() -> None:
    receipt = json.loads((PACKAGE_ROOT / "trace_reuse_receipt.json").read_text(encoding="utf-8"))
    assert receipt["missing_required_neutral_fields"] == []
    assert receipt["new_model_inference_required"] is False
    assert receipt["new_model_inference_performed"] is False
    for role in ("dev", "eval"):
        for family in ("streaming_sortformer", "ls_eend"):
            coverage = receipt["roles"][role]["families"][family]["field_coverage"]
            assert coverage["native_frame_start_source_samples"] is True
            assert coverage["native_frame_end_source_samples"] is True
            assert coverage["model_evidence_frontier_source_samples"] is True
            assert coverage["speaker_activity_probabilities"] is True


def test_causal_dependency_audit_requires_counterfactual_label() -> None:
    audit = json.loads(
        (PACKAGE_ROOT / "causal_lifecycle_dependency_audit.json").read_text(encoding="utf-8")
    )
    assert audit["conclusion"] == "material_dependency_present"
    assert audit["native_simplified_ontology_runtime_claim_allowed"] is False
    assert audit["s2_label"] == "fixed-issue-97-lifecycle-counterfactual-ablation"
    assert {Path(value["path"]).name for value in audit["audited_sources"]} == {
        "model_decode.py",
        "decoder.py",
    }


def test_duplicate_frozen_input_keys_fail_closed() -> None:
    with pytest.raises(SimplificationError, match="duplicate test row"):
        _unique_index([{"id": "a"}, {"id": "a"}], lambda value: value["id"], "test row")


def test_global_overlap_invalid_cells_are_counted_as_unknown() -> None:
    cells = (
        PosteriorCell(
            cell=EvaluationCell(0, 0, 1600, 800, (), False),
            probabilities=(0.1, 0.0),
            slot_alive=(True, True),
            evidence_frontier_sample=1600,
            state_reset=True,
            trace_valid=False,
        ),
        PosteriorCell(
            cell=EvaluationCell(1, 1600, 3200, 2400, ("A", "B"), False),
            probabilities=(0.9, 0.8),
            slot_alive=(True, True),
            evidence_frontier_sample=3200,
            state_reset=False,
            trace_valid=True,
        ),
    )
    records, coverage = _global_overlap_records({"source_id": "source"}, cells)
    assert len(records) == 1
    assert coverage["total_unmasked_cell_count"] == 2
    assert coverage["scored_cell_count"] == 1
    assert coverage["invalid_cell_count"] == 1
    assert coverage["invalid_support_seconds"] == 0.1
