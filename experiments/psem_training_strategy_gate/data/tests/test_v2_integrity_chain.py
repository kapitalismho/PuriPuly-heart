from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data import identity_components
from experiments.psem_training_strategy_gate.data.identity_components import (
    EXPECTED_V2_SOURCE_IDS,
    IdentityGraphError,
    _load_json_object,
    _load_jsonl_objects,
    _validate_topology_census,
    _validate_v2_reference_package,
    build_identity_graph,
)
from experiments.psem_training_strategy_gate.data.label_contract import load_contract
from experiments.psem_training_strategy_gate.data.provenance import (
    REQUIRED_PRIOR_SOURCE_IDS,
)

DATA_DIR = Path(__file__).resolve().parents[1]
V2_DIR = DATA_DIR / "v2"


def test_v2_identity_graph_binds_exact_scope_and_reference_package() -> None:
    graph = build_identity_graph(V2_DIR)
    assert graph["summary"]["source_count"] == len(EXPECTED_V2_SOURCE_IDS) == 93
    assert set(graph["input_artifacts"]) >= {
        "reference_artifact_receipt_sha256",
        "reference_integrity_report_sha256",
        "reference_provenance_sha256",
    }
    sources = _load_jsonl_objects(V2_DIR / "source_manifest.jsonl")
    exposed = {row["source_id"] for row in sources if row["selection_exposed"]}
    assert exposed == REQUIRED_PRIOR_SOURCE_IDS


def test_v2_reference_package_rejects_incomplete_source_scope() -> None:
    sources = _load_jsonl_objects(V2_DIR / "source_manifest.jsonl")
    normalized = _load_jsonl_objects(V2_DIR / "normalization_manifest.jsonl")
    _validate_v2_reference_package(V2_DIR, sources, normalized)
    with pytest.raises(IdentityGraphError, match="source coverage"):
        _validate_v2_reference_package(V2_DIR, sources[:-1], normalized[:-1])


def test_v2_identity_graph_rejects_incomplete_prior_exposure_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstructed = {
        source_id: {} for source_id in sorted(REQUIRED_PRIOR_SOURCE_IDS)[:-1]
    }
    monkeypatch.setattr(
        identity_components,
        "collect_prior_exposure",
        lambda _: reconstructed,
    )
    with pytest.raises(IdentityGraphError, match="scope mismatch"):
        build_identity_graph(V2_DIR)


def test_topology_consumer_rejects_row_not_bound_to_normalization() -> None:
    sources = _load_jsonl_objects(V2_DIR / "source_manifest.jsonl")
    normalized = _load_jsonl_objects(V2_DIR / "normalization_manifest.jsonl")
    topology = _load_jsonl_objects(V2_DIR / "topology_manifest.jsonl")
    census = _load_json_object(V2_DIR / "topology_census.json")
    contract = load_contract(version="psem-handoff-v1")
    changed = copy.deepcopy(topology)
    changed[0]["label_result_sha256"] = "0" * 64
    with pytest.raises(IdentityGraphError, match="inventory identity mismatch"):
        _validate_topology_census(
            V2_DIR,
            sources,
            normalized,
            changed,
            census,
            contract.contract_version,
            contract.document_sha256,
            contract.status,
        )


def test_v2_reference_receipt_rejects_artifact_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _load_jsonl_objects(V2_DIR / "source_manifest.jsonl")
    normalized = _load_jsonl_objects(V2_DIR / "normalization_manifest.jsonl")
    original = identity_components.sha256_file

    def changed(path: Path) -> str:
        if path.name == "reference_provenance.json":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(identity_components, "sha256_file", changed)
    with pytest.raises(IdentityGraphError, match="package binding mismatch"):
        _validate_v2_reference_package(V2_DIR, sources, normalized)
