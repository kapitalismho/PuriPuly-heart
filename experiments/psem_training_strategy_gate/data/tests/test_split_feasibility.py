from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data import split_assignment
from experiments.psem_training_strategy_gate.data.provenance import sha256_file
from experiments.psem_training_strategy_gate.data.split_assignment import (
    EVAL_ROLE,
    OFFICIAL_ROLES,
    SEARCH_ALGORITHM,
    SEARCH_SEED,
    SplitAssignmentError,
    _input_fingerprint,
    build_resolved_split_feasibility,
    build_split_manifest,
    validate_checked_split_package,
    write_split_assignment,
)
from experiments.psem_training_strategy_gate.data.split_feasibility import (
    ROLE_REQUIREMENTS,
    TOPOLOGY_REQUIREMENTS,
    SplitFeasibilityError,
    _eval_eligible_component_ids,
    _validate_natural_sources,
    build_split_feasibility,
    write_split_feasibility,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "experiments/speaker_representation_scd/models/registry.json"
SOURCE_REGISTRY_PATH = (
    REPO_ROOT / "experiments/speaker_representation_scd/models/source_registry.json"
)


def test_checked_in_feasibility_proves_valid_component_assignment() -> None:
    checked = json.loads((DATA_DIR / "split_feasibility.json").read_text(encoding="utf-8"))
    assert checked == build_resolved_split_feasibility(
        DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH
    )
    assert checked["search_status"] == "valid_component_assignment_found"
    assert checked["valid_assignment_exists"] is True
    assert checked["assignment_manifest_emitted"] is True
    assert len(checked["assignments"]) == 42
    assert checked["blocking_lower_bounds"] == []
    assert checked["deficits"]["scored_hours"] == 0.0
    assert checked["acquisition_handoff"]["minimum_additional_natural_scored_hours"] == 0.0
    assert all(deficit == 0 for deficit in checked["deficits"]["primary_topology_counts"].values())


def test_raw_lower_bound_audit_remains_assignment_agnostic() -> None:
    raw = build_split_feasibility(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH)
    assert raw["search_status"] == "component_assignment_search_required"
    assert raw["valid_assignment_exists"] is None
    assert raw["assignments"] == {}


def test_checked_split_manifest_is_current_and_passes_every_hard_gate() -> None:
    checked = json.loads((DATA_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    assert checked == build_split_manifest(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH)
    assert checked["official_roles"] == list(OFFICIAL_ROLES)
    assert checked["selection_order"] == [
        "PSEM-STRATEGY-EVAL",
        "PSEM-STRATEGY-DEV",
        "PSEM-STRATEGY-TRAIN",
    ]
    assert checked["search"]["algorithm"] == SEARCH_ALGORITHM
    assert checked["search"]["seed"] == SEARCH_SEED
    assert checked["search"]["input_fingerprint_sha256"] == _input_fingerprint(
        DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH
    )
    assert checked["search"]["eval_enumeration_complete"] is True
    assert checked["search"]["dev_search_exhaustive"] is False
    assert checked["search"]["secondary_search_bounded"] is True
    assert checked["search"]["upper_bound_achieved"] is True
    assert checked["objective_result"]["integer_global_upper_bound_achieved"] is True
    assert checked["hard_gate_status"] == "pass"
    assert all(result["passed"] is True for result in checked["hard_gate_results"])
    assert all(
        summary["corpora"] == ["AMI", "AliMeeting"]
        for summary in checked["role_summaries"].values()
    )
    assert len(checked["assignments"]["sources"]) == 76
    assert len(checked["assignments"]["components"]) == 42
    assert {row["role"] for row in checked["assignments"]["sources"]} == set(
        OFFICIAL_ROLES
    )
    eval_components = {
        row["component_id"]
        for row in checked["assignments"]["components"]
        if row["role"] == EVAL_ROLE
    }
    assert eval_components
    assert all(
        row["eval_eligible"] is True
        for row in checked["assignments"]["components"]
        if row["component_id"] in eval_components
    )
    assert all(value is False for value in checked["model_policy"].values())
    assert checked["search"]["model_derived_quantities_allowed"] is False


def test_split_manifest_has_exact_component_and_source_role_consistency() -> None:
    checked = json.loads((DATA_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    component_roles = {
        row["component_id"]: row["role"] for row in checked["assignments"]["components"]
    }
    source_rows = checked["assignments"]["sources"]
    assert len({row["source_id"] for row in source_rows}) == len(source_rows)
    assert all(row["role"] == component_roles[row["component_id"]] for row in source_rows)
    assert checked["leakage_audit"] == {
        "exact_source_coverage": True,
        "component_may_span_roles": False,
        "meeting_session_may_span_roles": False,
        "waveform_may_span_roles": False,
        "known_speaker_may_span_roles": False,
        "prior_selection_exposed_component_in_eval": False,
        "exact_known_wavlm_pretraining_overlap_in_eval": False,
    }


def test_data_census_is_bound_to_selected_split_manifest() -> None:
    census = (DATA_DIR / "DATA_CENSUS.md").read_text(encoding="utf-8")
    assert "role-specific allocation remains unproven" not in census
    assert "Split roles remain unassigned" not in census
    assert (
        f"Split manifest SHA-256: `{sha256_file(DATA_DIR / 'split_manifest.json')}`"
        in census
    )
    for role in ("TRAIN", "DEV", "EVAL"):
        assert f"| {role} |" in census


def test_historical_prior_config_identity_participates_in_search_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _input_fingerprint(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH)
    original_sha256_file = split_assignment.sha256_file
    historical_path = (
        DATA_DIR.parents[2]
        / next(iter(split_assignment.HISTORICAL_CONFIGS.values()))
    ).resolve()

    def changed_sha256(path: Path) -> str:
        if path.resolve() == historical_path:
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(split_assignment, "sha256_file", changed_sha256)
    changed = _input_fingerprint(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH)
    assert changed != original


def test_truncated_eval_enumeration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_assignment._build_split_manifest_cached.cache_clear()
    monkeypatch.setattr(split_assignment, "MAX_EVAL_SUBSETS", 1000)
    try:
        with pytest.raises(SplitAssignmentError, match="completing the optimal frontier"):
            build_split_manifest(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH)
    finally:
        split_assignment._build_split_manifest_cached.cache_clear()


def test_checked_split_package_rejects_tampered_resolved_feasibility(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "split_manifest.json"
    feasibility_path = tmp_path / "split_feasibility.json"
    manifest_path.write_bytes((DATA_DIR / "split_manifest.json").read_bytes())
    feasibility = json.loads(
        (DATA_DIR / "split_feasibility.json").read_text(encoding="utf-8")
    )
    feasibility["valid_assignment_exists"] = False
    feasibility_path.write_text(json.dumps(feasibility), encoding="utf-8")
    with pytest.raises(SplitAssignmentError, match="feasibility is not current"):
        validate_checked_split_package(
            DATA_DIR,
            REGISTRY_PATH,
            SOURCE_REGISTRY_PATH,
            manifest_path,
            feasibility_path,
        )


def test_authority_role_and_topology_minima_are_exact() -> None:
    assert ROLE_REQUIREMENTS == {
        "PSEM-STRATEGY-TRAIN": {"scored_hours": 20, "independent_meetings": 12},
        "PSEM-STRATEGY-DEV": {"scored_hours": 5, "independent_meetings": 4},
        "PSEM-STRATEGY-EVAL": {"scored_hours": 8, "independent_meetings": 6},
    }


def test_eval_eligibility_rejects_forbidden_or_unresolved_whole_components() -> None:
    graph = {
        "components": [
            {
                "component_id": "component-mixed",
                "source_ids": ["a", "b"],
                "split_assignment_eligible": True,
                "eval_forbidden": False,
            },
            {
                "component_id": "component-unresolved",
                "source_ids": ["c"],
                "split_assignment_eligible": False,
                "eval_forbidden": False,
            },
        ]
    }
    overlap_rows = [
        {
            "source_id": "a",
            "identity_component_id": "component-mixed",
            "eval_forbidden_by_pretraining_overlap": False,
        },
        {
            "source_id": "b",
            "identity_component_id": "component-mixed",
            "eval_forbidden_by_pretraining_overlap": True,
        },
        {
            "source_id": "c",
            "identity_component_id": "component-unresolved",
            "eval_forbidden_by_pretraining_overlap": False,
        },
    ]
    assert _eval_eligible_component_ids(graph, overlap_rows) == set()


@pytest.mark.parametrize(
    "changes",
    [
        {"corpus": "UnclassifiedCorpus"},
        {"synthetic_parent_id": "synthetic-parent"},
        {"synthetic_transformation_seed": "seed"},
    ],
)
def test_natural_only_gate_rejects_unclassified_or_synthetic_sources(
    changes: dict[str, str],
) -> None:
    row = {
        "corpus": "AMI",
        "source_id": "ami_test",
        "session_id": "test",
        "meeting_type": "scenario",
        "audio_source_url": "https://example.invalid/test.wav",
        "license_id": "CC-BY-4.0",
        "use_authorization": "public_research_under_source_license",
        **changes,
    }
    with pytest.raises(SplitFeasibilityError, match="natural-only"):
        _validate_natural_sources([row])
    assert TOPOLOGY_REQUIREMENTS == {
        "clean_direct_different_speaker_handoff": {"train_dev": 100, "eval": 20},
        "silence_gap_different_speaker_handoff": {"train_dev": 200, "eval": 40},
        "same_speaker_silence_gap_resume": {"train_dev": 200, "eval": 40},
        "overlap_return": {"train_dev": 100, "eval": 20},
        "overlap_takeover": {"train_dev": 100, "eval": 20},
        "short_backchannel_return": {"train_dev": 80, "eval": 20},
    }


def test_split_feasibility_output_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "split_feasibility.json"
    write_split_feasibility(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH, output)
    first = output.read_bytes()
    write_split_feasibility(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH, output)
    assert output.read_bytes() == first


def test_split_assignment_output_is_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "split_manifest.json"
    feasibility = tmp_path / "split_feasibility.json"
    markdown = tmp_path / "DATA_CENSUS.md"
    write_split_assignment(
        DATA_DIR,
        REGISTRY_PATH,
        SOURCE_REGISTRY_PATH,
        manifest,
        feasibility,
        markdown,
    )
    first_manifest = manifest.read_bytes()
    first_feasibility = feasibility.read_bytes()
    first_markdown = markdown.read_bytes()
    write_split_assignment(
        DATA_DIR,
        REGISTRY_PATH,
        SOURCE_REGISTRY_PATH,
        manifest,
        feasibility,
        markdown,
    )
    assert manifest.read_bytes() == first_manifest
    assert feasibility.read_bytes() == first_feasibility
    assert markdown.read_bytes() == first_markdown
