from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data.pretraining_overlap import (
    CLASSIFICATIONS,
    MODEL_ARTIFACT,
    MODEL_ID,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PretrainingOverlapError,
    _classify_overlap,
    build_pretraining_overlap_audit,
    write_pretraining_overlap_audit,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    EXPECTED_ALIMEETING_MEETINGS,
    EXPECTED_AMI_MEETINGS,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "experiments/speaker_representation_scd/models/registry.json"
SOURCE_REGISTRY_PATH = (
    REPO_ROOT / "experiments/speaker_representation_scd/models/source_registry.json"
)
EXPECTED_SOURCE_COUNT = len(EXPECTED_AMI_MEETINGS) + len(EXPECTED_ALIMEETING_MEETINGS)


def test_checked_in_audit_binds_the_exact_checkpoint_and_all_sources() -> None:
    audit_path = DATA_DIR / "wavlm_pretraining_overlap.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit == build_pretraining_overlap_audit(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH)
    assert audit["checkpoint"] == {
        "model_id": MODEL_ID,
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "artifact": MODEL_ARTIFACT,
    }
    identity_graph = json.loads(
        (DATA_DIR / "identity_components.json").read_text(encoding="utf-8")
    )
    assert audit["summary"] == {
        "source_count": EXPECTED_SOURCE_COUNT,
        "identity_component_count": identity_graph["summary"]["component_count"],
        "classification_counts": {
            "exact_session_overlap_known": 0,
            "corpus_level_overlap_known": 0,
            "no_known_overlap": EXPECTED_SOURCE_COUNT,
            "unknown": 0,
        },
        "eval_forbidden_source_count": 0,
        "pretraining_clean_claim_count": 0,
    }
    assert all(
        row["classification"] == "no_known_overlap"
        and row["pretraining_clean_claimed"] is False
        and row["identity_component_id"].startswith("component-")
        and len(row["identity_node_sha256"]) == 64
        for row in audit["sources"]
    )
    assert all(
        len(evidence["sha256"]) == 64
        for evidence in audit["official_pretraining_provenance"]["evidence"]
    )
    assert audit["input_artifacts"]["identity_components_sha256"] == sha256_file(
        DATA_DIR / "identity_components.json"
    )


def test_checkpoint_registry_drift_fails_closed(tmp_path: Path) -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    next(row for row in registry["models"] if row["model_id"] == MODEL_ID)["revision"] = "0" * 40
    changed_registry = tmp_path / "registry.json"
    changed_registry.write_text(
        json.dumps(registry, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(PretrainingOverlapError, match="checkpoint binding mismatch"):
        build_pretraining_overlap_audit(DATA_DIR, changed_registry, SOURCE_REGISTRY_PATH)


def test_checked_in_identity_graph_drift_fails_closed(tmp_path: Path) -> None:
    graph = json.loads((DATA_DIR / "identity_components.json").read_text(encoding="utf-8"))
    graph["nodes"][0]["component_id"] = "component-tampered"
    changed_graph = tmp_path / "identity_components.json"
    changed_graph.write_text(
        json.dumps(graph, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(PretrainingOverlapError, match="identity graph is not current"):
        build_pretraining_overlap_audit(
            DATA_DIR,
            REGISTRY_PATH,
            SOURCE_REGISTRY_PATH,
            identity_graph_path=changed_graph,
        )


@pytest.mark.parametrize(
    ("corpus", "session_id", "exact", "complete", "expected", "forbidden"),
    [
        (
            "AMI",
            "IS1009a",
            {("AMI", "IS1009a")},
            True,
            "exact_session_overlap_known",
            True,
        ),
        (
            "VoxPopuli",
            "session",
            set(),
            True,
            "corpus_level_overlap_known",
            False,
        ),
        ("AMI", "IS1009a", set(), True, "no_known_overlap", False),
        ("AMI", "IS1009a", set(), False, "unknown", False),
    ],
)
def test_overlap_classification_policy(
    corpus: str,
    session_id: str,
    exact: set[tuple[str, str]],
    complete: bool,
    expected: str,
    forbidden: bool,
) -> None:
    result = _classify_overlap(
        corpus,
        session_id,
        exact_session_overlaps=exact,
        provenance_complete=complete,
    )
    assert result["classification"] == expected
    assert result["classification"] in CLASSIFICATIONS
    assert result["eval_forbidden_by_pretraining_overlap"] is forbidden
    assert result["pretraining_clean_claimed"] is False


def test_audit_output_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    write_pretraining_overlap_audit(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH, output)
    first = output.read_bytes()
    write_pretraining_overlap_audit(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH, output)
    assert output.read_bytes() == first
