from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data.split_assignment import (
    OFFICIAL_ROLES,
    SAMPLE_RATE_HZ,
    SplitAssignmentError,
    _aggregate,
    _build_leakage_audit,
    _train_dev_combined_minimum_passes,
    build_split_manifest,
)
from experiments.psem_training_strategy_gate.data.split_feasibility import (
    CORPUS_BALANCE_REQUIREMENTS,
    build_split_feasibility,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(__file__).resolve().parents[1] / "v2"
REGISTRY_PATH = REPO_ROOT / "experiments/speaker_representation_scd/models/registry.json"
SOURCE_REGISTRY_PATH = (
    REPO_ROOT / "experiments/speaker_representation_scd/models/source_registry.json"
)


def test_v2_raw_feasibility_exposes_exact_corpus_balance_contract() -> None:
    feasibility = build_split_feasibility(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH)
    assert feasibility["authority_ref"].endswith("/issues/86")
    assert feasibility["contract_version"] == "psem-handoff-v1"
    assert feasibility["requirements"]["corpus_balance"] == CORPUS_BALANCE_REQUIREMENTS
    assert feasibility["blocking_lower_bounds"] == []


def test_v2_checked_split_is_current_leakage_safe_and_corpus_balanced() -> None:
    checked = json.loads((DATA_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    assert checked == build_split_manifest(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH)
    assert checked["search"]["algorithm_version"] == "2"
    assert checked["search"]["upper_bound_achieved"] is True
    assert checked["search"]["search_complete_for_chosen_primary_optimum"] is True
    assert checked["hard_gate_status"] == "pass"
    assert len(checked["hard_gate_results"]) == 37
    assert all(result["passed"] is True for result in checked["hard_gate_results"])
    assert len(checked["assignments"]["sources"]) == 93
    assert len(checked["assignments"]["components"]) == 57
    assert all(len(row["reference_sha256"]) == 64 for row in checked["assignments"]["sources"])
    assert checked["leakage_audit"] == {
        "exact_source_coverage": True,
        "component_may_span_roles": False,
        "meeting_session_may_span_roles": False,
        "waveform_may_span_roles": False,
        "known_speaker_may_span_roles": False,
        "prior_selection_exposed_component_in_eval": False,
        "exact_known_wavlm_pretraining_overlap_in_eval": False,
    }
    for role in OFFICIAL_ROLES:
        summary = checked["role_summaries"][role]
        requirement = CORPUS_BALANCE_REQUIREMENTS[role]
        assert summary["corpora"] == ["AMI", "AliMeeting"]
        for corpus, required in requirement["minimum_corpus_source_counts"].items():
            assert summary["corpus_source_counts"][corpus] >= required
        for corpus, required in requirement["minimum_corpus_scored_samples"].items():
            assert summary["corpus_scored_samples"][corpus] >= required
        maximum = requirement["maximum_corpus_scored_share"]
        assert (
            max(summary["corpus_scored_samples"].values()) * maximum["denominator"]
            <= summary["scored_samples"] * maximum["numerator"]
        )


def test_eval_frontier_requires_combined_train_dev_minima() -> None:
    empty = _aggregate([], [])
    assert _train_dev_combined_minimum_passes(empty) is False
    passing = replace(
        empty,
        scored_samples=25 * 3600 * SAMPLE_RATE_HZ,
        source_ids=tuple(f"source-{index}" for index in range(16)),
    )
    assert _train_dev_combined_minimum_passes(passing) is True
    assert (
        _train_dev_combined_minimum_passes(
            replace(passing, scored_samples=passing.scored_samples - 1)
        )
        is False
    )
    assert (
        _train_dev_combined_minimum_passes(
            replace(passing, source_ids=passing.source_ids[:-1])
        )
        is False
    )


def test_v2_census_describes_current_split_without_claiming_a_finished_freeze() -> None:
    markdown = (DATA_DIR / "DATA_CENSUS.md").read_text(encoding="utf-8")
    assert "93 sources in 57 components" in markdown
    assert "passes all 37 role-specific hard gates" in markdown
    assert "PSEM-STRATEGY-DATA-v2" in markdown
    assert "will bind these artifacts at the final freeze checkpoint" in markdown
    assert "PSEM-STRATEGY-DATA-v1" not in markdown


def test_v2_leakage_audit_is_computed_from_assignments() -> None:
    split = json.loads((DATA_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    graph = json.loads(
        (DATA_DIR / "identity_components.json").read_text(encoding="utf-8")
    )
    overlap = json.loads(
        (DATA_DIR / "wavlm_pretraining_overlap.json").read_text(encoding="utf-8")
    )
    sources = {
        row["source_id"]: row
        for row in (
            json.loads(line)
            for line in (DATA_DIR / "source_manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    assignments = {
        row["source_id"]: row["role"] for row in split["assignments"]["sources"]
    }
    assert (
        _build_leakage_audit(graph, overlap["sources"], sources, assignments)
        == split["leakage_audit"]
    )
    component = next(item for item in graph["components"] if len(item["source_ids"]) > 1)
    changed = dict(assignments)
    source_id = component["source_ids"][0]
    changed[source_id] = next(role for role in OFFICIAL_ROLES if role != changed[source_id])
    with pytest.raises(SplitAssignmentError, match="computed split leakage audit failed"):
        _build_leakage_audit(graph, overlap["sources"], sources, changed)
