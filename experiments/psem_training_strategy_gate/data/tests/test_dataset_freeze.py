from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data import dataset_freeze
from experiments.psem_training_strategy_gate.data.dataset_freeze import (
    DATASET_FREEZE_ID,
    FROZEN_ARTIFACTS,
    REPOSITORY_INPUTS,
    DatasetFreezeError,
    build_dataset_freeze,
    validate_checked_dataset_freeze,
    write_dataset_freeze,
)
from experiments.psem_training_strategy_gate.data.provenance import canonical_sha256

DATA_DIR = Path(__file__).resolve().parents[1]


def test_checked_dataset_freeze_is_current_and_complete() -> None:
    checked = json.loads((DATA_DIR / "dataset_freeze.json").read_text(encoding="utf-8"))
    assert checked == build_dataset_freeze(DATA_DIR)
    assert checked == validate_checked_dataset_freeze(DATA_DIR)
    assert checked["dataset_freeze_id"] == DATASET_FREEZE_ID
    assert checked["freeze_status"] == "frozen"
    assert set(checked["artifact_sha256"]) == set(FROZEN_ARTIFACTS)
    assert set(checked["repository_input_sha256"]) == set(REPOSITORY_INPUTS)
    assert checked["source_identity_binding"]["source_count"] == 76
    assert checked["source_identity_binding"]["eval_source_count"] == 19
    assert checked["source_identity_binding"]["eval_sources_finally_eligible"] is True
    assert (
        checked["source_identity_binding"]["final_role_and_eval_eligibility_authority"]
        == "split_manifest.json"
    )
    assert checked["split_binding"]["hard_gate_count"] == 22
    assert checked["split_binding"]["hard_gate_status"] == "pass"
    payload = copy.deepcopy(checked)
    observed_digest = payload.pop("freeze_payload_sha256")
    assert observed_digest == canonical_sha256(payload)


def test_dataset_freeze_binds_all_roles_and_model_exclusions() -> None:
    checked = validate_checked_dataset_freeze(DATA_DIR)
    assert set(checked["role_summaries"]) == {
        "PSEM-STRATEGY-TRAIN",
        "PSEM-STRATEGY-DEV",
        "PSEM-STRATEGY-EVAL",
    }
    assert checked["role_summaries"]["PSEM-STRATEGY-TRAIN"]["scored_hours"] == 23.842971
    assert checked["role_summaries"]["PSEM-STRATEGY-DEV"]["scored_hours"] == 5.958004
    assert checked["role_summaries"]["PSEM-STRATEGY-EVAL"]["scored_hours"] == 9.989041
    assert all(value is False for value in checked["model_policy"].values())
    assert checked["creation_provenance"]["preflight_required"] is True


def test_checked_dataset_freeze_rejects_tampering(tmp_path: Path) -> None:
    checked = validate_checked_dataset_freeze(DATA_DIR)
    tampered = copy.deepcopy(checked)
    tampered["dataset_freeze_id"] = "PSEM-STRATEGY-DATA-v2"
    manifest_path = tmp_path / "dataset_freeze.json"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DatasetFreezeError, match="not current"):
        validate_checked_dataset_freeze(DATA_DIR, manifest_path)


def test_checked_dataset_freeze_rejects_bound_artifact_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sha256_file = dataset_freeze.sha256_file
    changed_path = (DATA_DIR / "DATASET_PLAN.md").resolve()

    def changed_sha256(path: Path) -> str:
        if path.resolve() == changed_path:
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(dataset_freeze, "sha256_file", changed_sha256)
    with pytest.raises(DatasetFreezeError, match="not current"):
        validate_checked_dataset_freeze(DATA_DIR)


def test_dataset_freeze_rejects_stale_upstream_hash_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sha256_file = dataset_freeze.sha256_file
    changed_path = (DATA_DIR / "topology_census.json").resolve()

    def changed_sha256(path: Path) -> str:
        if path.resolve() == changed_path:
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(dataset_freeze, "sha256_file", changed_sha256)
    with pytest.raises(DatasetFreezeError, match="artifact hash binding is stale"):
        build_dataset_freeze(DATA_DIR)


def test_dataset_freeze_rejects_stale_historical_search_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sha256_file = dataset_freeze.sha256_file
    historical_path = (
        DATA_DIR.parents[2] / next(iter(dataset_freeze.HISTORICAL_CONFIGS.values()))
    ).resolve()

    def changed_sha256(path: Path) -> str:
        if path.resolve() == historical_path:
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(dataset_freeze, "sha256_file", changed_sha256)
    with pytest.raises(DatasetFreezeError, match="split search input fingerprint is stale"):
        build_dataset_freeze(DATA_DIR)


def test_dataset_freeze_rejects_duplicate_manifest_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load_jsonl = dataset_freeze._load_jsonl

    def duplicate_source(path: Path) -> list[dict[str, object]]:
        rows = original_load_jsonl(path)
        if path.name == "source_manifest.jsonl":
            return [*rows, copy.deepcopy(rows[0])]
        return rows

    monkeypatch.setattr(dataset_freeze, "_load_jsonl", duplicate_source)
    with pytest.raises(DatasetFreezeError, match="coverage is not exact"):
        build_dataset_freeze(DATA_DIR)


def test_dataset_freeze_writer_refuses_changed_existing_v1(tmp_path: Path) -> None:
    output = tmp_path / "dataset_freeze.json"
    write_dataset_freeze(DATA_DIR, output)
    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["freeze_payload_sha256"] = "0" * 64
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DatasetFreezeError, match="existing dataset freeze is immutable"):
        write_dataset_freeze(DATA_DIR, output)


def test_dataset_freeze_output_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "dataset_freeze.json"
    write_dataset_freeze(DATA_DIR, output)
    first = output.read_bytes()
    write_dataset_freeze(DATA_DIR, output)
    assert output.read_bytes() == first
    assert json.loads(first)["dataset_freeze_id"] == DATASET_FREEZE_ID
