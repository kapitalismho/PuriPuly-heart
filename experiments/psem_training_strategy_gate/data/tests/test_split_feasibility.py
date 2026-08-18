from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_checked_in_feasibility_requires_component_assignment_search() -> None:
    checked = json.loads((DATA_DIR / "split_feasibility.json").read_text(encoding="utf-8"))
    assert checked == build_split_feasibility(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH)
    assert checked["search_status"] == "component_assignment_search_required"
    assert checked["valid_assignment_exists"] is None
    assert checked["assignments"] == {}
    assert checked["blocking_lower_bounds"] == []
    assert checked["deficits"]["scored_hours"] == 0.0
    assert checked["acquisition_handoff"]["minimum_additional_natural_scored_hours"] == 0.0
    assert all(deficit == 0 for deficit in checked["deficits"]["primary_topology_counts"].values())


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
