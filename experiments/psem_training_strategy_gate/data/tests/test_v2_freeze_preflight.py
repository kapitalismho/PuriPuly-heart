from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data import (
    dataset_context,
    dataset_freeze,
    dataset_preflight,
)
from experiments.psem_training_strategy_gate.data.dataset_context import DatasetContextError
from experiments.psem_training_strategy_gate.data.dataset_freeze import (
    V2_FROZEN_ARTIFACTS,
    V2_INHERITED_ARTIFACTS,
    V2_REPOSITORY_INPUTS,
    DatasetFreezeError,
)
from experiments.psem_training_strategy_gate.data.dataset_preflight import (
    V2_REQUIRED_CHECK_IDS,
    DatasetPreflightError,
    validate_checked_dataset_preflight,
)
from experiments.psem_training_strategy_gate.data.label_contract import LabelContractError
from experiments.psem_training_strategy_gate.data.provenance import (
    canonical_sha256,
    sha256_file,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "v2"


def _load_json(name: str) -> dict[str, object]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def test_checked_v2_freeze_and_preflight_are_current_and_complete() -> None:
    freeze = _load_json("dataset_freeze.json")
    report = validate_checked_dataset_preflight(DATA_DIR)
    assert freeze["dataset_freeze_id"] == "PSEM-STRATEGY-DATA-v2"
    assert freeze["source_identity_binding"]["source_count"] == 93
    assert freeze["source_identity_binding"]["eval_source_count"] == 19
    assert freeze["source_identity_binding"]["eval_sources_finally_eligible"] is True
    assert freeze["split_binding"]["hard_gate_count"] == 37
    assert freeze["split_binding"]["hard_gate_status"] == "pass"
    assert set(freeze["artifact_sha256"]) == set(V2_FROZEN_ARTIFACTS)
    assert set(freeze["inherited_artifact_sha256"]) == set(V2_INHERITED_ARTIFACTS)
    assert set(freeze["repository_input_sha256"]) == set(V2_REPOSITORY_INPUTS)
    assert {"pyproject.toml", "uv.lock"} <= set(freeze["repository_input_sha256"])
    freeze_core = copy.deepcopy(freeze)
    freeze_core.pop("freeze_payload_sha256")
    preflight_binding = freeze_core.pop("preflight_binding")
    freeze_core_digest = freeze_core.pop("freeze_core_payload_sha256")
    assert freeze_core_digest == canonical_sha256(freeze_core)
    freeze_payload = copy.deepcopy(freeze)
    assert freeze_payload.pop("freeze_payload_sha256") == canonical_sha256(freeze_payload)
    assert report["ready_for_issue_76"] is True
    assert report["failed_checks"] == []
    assert tuple(check["id"] for check in report["checks"]) == V2_REQUIRED_CHECK_IDS
    assert len(report["checks"]) == len(set(V2_REQUIRED_CHECK_IDS)) == 59
    assert all(check["passed"] is True for check in report["checks"])
    report_payload = copy.deepcopy(report)
    assert report_payload.pop("preflight_payload_sha256") == canonical_sha256(report_payload)
    assert preflight_binding == {
        "preflight_report_sha256": sha256_file(DATA_DIR / "preflight_report.json"),
        "preflight_report_canonical_sha256": canonical_sha256(report),
        "preflight_payload_sha256": report["preflight_payload_sha256"],
        "freeze_core_payload_sha256": freeze_core_digest,
        "check_count": 59,
        "ready_for_issue_76": True,
    }
    assert report["freeze_binding"]["freeze_core_payload_sha256"] == freeze_core_digest


def test_v2_freeze_rejects_forged_preflight_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = dataset_context.resolve_dataset_context(DATA_DIR)
    freeze_core = dataset_freeze.build_v2_dataset_freeze_core(DATA_DIR)
    checked = _load_json("preflight_report.json")
    forged_checks = copy.deepcopy(checked)
    for index, check in enumerate(forged_checks["checks"]):
        check["id"] = f"forged.check.{index}"
    forged_generator = {**copy.deepcopy(checked), "generator": "forged.generator"}
    forged_version = {**copy.deepcopy(checked), "generator_version": "2"}
    for report in (forged_checks, forged_generator, forged_version):
        payload = copy.deepcopy(report)
        payload.pop("preflight_payload_sha256")
        report["preflight_payload_sha256"] = canonical_sha256(payload)
        monkeypatch.setattr(dataset_freeze, "_load_json", lambda _path: report)
        with pytest.raises(DatasetFreezeError, match="not current and passing"):
            dataset_freeze._validate_v2_preflight_result(context, freeze_core)


@pytest.mark.parametrize("schema_version", (1.0, True))
def test_v2_freeze_validation_rejects_numeric_type_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    schema_version: object,
) -> None:
    expected = _load_json("dataset_freeze.json")
    tampered = copy.deepcopy(expected)
    tampered["schema_version"] = schema_version
    manifest_path = tmp_path / "dataset_freeze.json"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(
        dataset_freeze,
        "build_dataset_freeze",
        lambda data_dir: copy.deepcopy(expected),
    )
    with pytest.raises(DatasetFreezeError, match="not current"):
        dataset_freeze.validate_checked_dataset_freeze(DATA_DIR, manifest_path)


@pytest.mark.parametrize("schema_version", (1.0, True))
def test_v2_preflight_validation_rejects_numeric_type_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    schema_version: object,
) -> None:
    expected = _load_json("preflight_report.json")
    tampered = copy.deepcopy(expected)
    tampered["schema_version"] = schema_version
    report_path = tmp_path / "preflight_report.json"
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(
        dataset_preflight,
        "build_dataset_preflight",
        lambda data_dir: copy.deepcopy(expected),
    )
    with pytest.raises(DatasetPreflightError, match="not current"):
        validate_checked_dataset_preflight(DATA_DIR, report_path)


def test_v2_reference_check_rejects_integrity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = _load_json("dataset_freeze.json")
    original_load_json = dataset_preflight._load_json

    def changed_load_json(path: Path) -> dict[str, object]:
        value = original_load_json(path)
        if path.name == "reference_integrity_report.json":
            value["checks"].pop("exact_upstream_revision")
        return value

    monkeypatch.setattr(dataset_preflight, "_load_json", changed_load_json)
    check = dataset_preflight._v2_reference_integrity_check(DATA_DIR, freeze)
    assert check["passed"] is False
    assert check["observed"]["all_integrity_checks_pass"] is False


def test_v2_reference_check_rejects_nonlexical_inventory_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = _load_json("dataset_freeze.json")
    original_load_json = dataset_preflight._load_json

    def changed_load_json(path: Path) -> dict[str, object]:
        value = original_load_json(path)
        if path.name == "nonlexical_risk_inventory.json":
            value["inventory_version"] = "drifted"
        return value

    monkeypatch.setattr(dataset_preflight, "_load_json", changed_load_json)
    check = dataset_preflight._v2_reference_integrity_check(DATA_DIR, freeze)
    assert check["passed"] is False
    assert check["observed"]["nonlexical_inventory_exact"] is False


def test_v2_gate_check_rejects_fraction_receipt_drift() -> None:
    split = _load_json("split_manifest.json")
    gate = next(
        row
        for row in split["hard_gate_results"]
        if row["id"] == "PSEM-STRATEGY-EVAL.maximum_corpus_scored_share"
    )
    gate["observed"]["decimal"] = 0.0
    with pytest.raises(DatasetPreflightError, match="corpus-share gate is invalid"):
        dataset_preflight._v2_gate_checks(split)


def test_v2_gate_check_rejects_negative_fraction_numerator() -> None:
    split = _load_json("split_manifest.json")
    gate = next(
        row
        for row in split["hard_gate_results"]
        if row["id"] == "PSEM-STRATEGY-TRAIN.maximum_corpus_scored_share"
    )
    gate["observed"]["numerator"] = -1
    gate["observed"]["decimal"] = round(-1 / gate["observed"]["denominator"], 8)
    with pytest.raises(DatasetPreflightError, match="corpus-share gate is invalid"):
        dataset_preflight._v2_gate_checks(split)


def test_v2_annotation_check_rejects_float_sample_rate() -> None:
    context = dataset_context.resolve_dataset_context(DATA_DIR)
    source_rows = dataset_preflight._load_jsonl(DATA_DIR / "source_manifest.jsonl")
    annotation_rows = dataset_preflight._load_jsonl(DATA_DIR / "annotation_manifest.jsonl")
    normalization_rows = dataset_preflight._load_jsonl(DATA_DIR / "normalization_manifest.jsonl")
    split = _load_json("split_manifest.json")
    source_rows[0]["sample_rate_hz"] = 16000.0
    check = dataset_preflight._v2_annotation_coverage_check(
        context,
        source_rows,
        annotation_rows,
        normalization_rows,
        {row["source_id"] for row in split["assignments"]["sources"]},
    )
    assert check["passed"] is False


def test_v2_masking_check_rejects_unknown_reason_key() -> None:
    contract = _load_json("operational_label_contract.json")
    census = _load_json("topology_census.json")
    topology_rows = dataset_preflight._load_jsonl(DATA_DIR / "topology_manifest.jsonl")
    row = next(
        item for item in topology_rows if item["mask_diagnostics"]["masked_transition_reasons"]
    )
    reasons = row["mask_diagnostics"]["masked_transition_reasons"]
    original = next(iter(reasons))
    count = reasons.pop(original)
    reasons["bogus_unknown_class"] = count
    overall = census["overall"]["mask_diagnostics"]["masked_transition_reasons"]
    overall[original] -= count
    if overall[original] == 0:
        overall.pop(original)
    overall["bogus_unknown_class"] = count
    check = dataset_preflight._masking_check(
        contract,
        census,
        topology_rows,
        dataset_preflight.V2_EXPECTED_MASK_RULES,
    )
    assert check["passed"] is False


def test_v2_exclusive_counting_rejects_float_episode_count() -> None:
    contract = _load_json("operational_label_contract.json")
    census = _load_json("topology_census.json")
    topology_rows = dataset_preflight._load_jsonl(DATA_DIR / "topology_manifest.jsonl")
    topology_rows[0]["exclusive_primary_episode_count"] = float(
        topology_rows[0]["exclusive_primary_episode_count"]
    )
    check = dataset_preflight._exclusive_counting_check(contract, census, topology_rows)
    assert check["passed"] is False


def test_v2_identity_check_rejects_float_scored_coordinate() -> None:
    context = dataset_context.resolve_dataset_context(DATA_DIR)
    freeze = _load_json("dataset_freeze.json")
    source_rows = dataset_preflight._load_jsonl(DATA_DIR / "source_manifest.jsonl")
    annotation_rows = dataset_preflight._load_jsonl(DATA_DIR / "annotation_manifest.jsonl")
    normalization_rows = dataset_preflight._load_jsonl(DATA_DIR / "normalization_manifest.jsonl")
    topology_rows = dataset_preflight._load_jsonl(DATA_DIR / "topology_manifest.jsonl")
    split = _load_json("split_manifest.json")
    normalization_rows[0]["scored_start_sample"] = float(
        normalization_rows[0]["scored_start_sample"]
    )
    topology_rows[0]["scored_start_sample"] = normalization_rows[0]["scored_start_sample"]
    topology_rows[0]["normalization_row_sha256"] = canonical_sha256(normalization_rows[0])
    check = dataset_preflight._hash_checks(
        DATA_DIR,
        freeze,
        source_rows,
        annotation_rows,
        normalization_rows,
        topology_rows,
        split["assignments"]["sources"],
        context,
    )[1]
    assert check["passed"] is False
    assert normalization_rows[0]["source_id"] in check["violations"]


def test_dataset_context_wraps_unsupported_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported_contract(*args: object, **kwargs: object) -> object:
        raise LabelContractError("unsupported")

    monkeypatch.setattr(dataset_context, "load_contract", unsupported_contract)
    with pytest.raises(DatasetContextError, match="unsupported"):
        dataset_context.resolve_dataset_context(DATA_DIR)
