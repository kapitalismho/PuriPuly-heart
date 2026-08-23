from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data import dataset_preflight
from experiments.psem_training_strategy_gate.data.dataset_preflight import (
    EXPECTED_GATE_SPECS,
    DatasetPreflightError,
    build_dataset_preflight,
    validate_checked_dataset_preflight,
    write_dataset_preflight,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    canonical_sha256,
    sha256_file,
)

DATA_DIR = Path(__file__).resolve().parents[1]
AUTHORITY_REQUIRED_CHECK_IDS = (
    "natural_hours.train",
    "natural_hours.dev",
    "natural_hours.eval",
    "independent_meetings.train",
    "independent_meetings.dev",
    "independent_meetings.eval",
    "topology.train_dev.clean_direct_different_speaker_handoff",
    "topology.eval.clean_direct_different_speaker_handoff",
    "topology.train_dev.silence_gap_different_speaker_handoff",
    "topology.eval.silence_gap_different_speaker_handoff",
    "topology.train_dev.same_speaker_silence_gap_resume",
    "topology.eval.same_speaker_silence_gap_resume",
    "topology.train_dev.overlap_return",
    "topology.eval.overlap_return",
    "topology.train_dev.overlap_takeover",
    "topology.eval.overlap_takeover",
    "topology.train_dev.short_backchannel_return",
    "topology.eval.short_backchannel_return",
    "negative_exposure.train_dev.stable_singleton",
    "negative_exposure.eval.stable_singleton",
    "negative_exposure.train_dev.ongoing_overlap",
    "negative_exposure.eval.ongoing_overlap",
    "leakage.meeting_session",
    "leakage.waveform",
    "leakage.known_speaker",
    "leakage.connected_component",
    "leakage.prior_selection_eval",
    "leakage.exact_wavlm_pretraining_session_eval",
    "annotations.cover_every_scored_range",
    "annotations.unresolved_and_ambiguous_regions_masked",
    "topology.primary_gate_counts_exclusive_and_reproducible",
    "hashes.frozen_artifacts_and_repository_inputs_resolve",
    "hashes.source_annotation_split_identities_resolve",
    "contract.operational_version_frozen",
    "freeze.dataset_freeze_id_present_and_consistent",
    "freeze.current_and_internally_consistent",
    "data.natural_only",
    "split.model_derived_quantities_forbidden",
    "model_boundary.model_predictions_consulted",
    "model_boundary.model_scores_consulted",
    "model_boundary.official_model_results_inspected",
    "model_boundary.official_model_training_performed",
)


def _load_json(name: str) -> dict[str, object]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _load_jsonl(name: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (DATA_DIR / name).read_text(encoding="utf-8").splitlines()
    ]


def test_checked_dataset_preflight_is_current_and_passes() -> None:
    checked = json.loads((DATA_DIR / "preflight_report.json").read_text(encoding="utf-8"))
    assert checked == build_dataset_preflight(DATA_DIR)
    assert checked == validate_checked_dataset_preflight(DATA_DIR)
    assert checked["dataset_freeze_id"] == "PSEM-STRATEGY-DATA-v1"
    assert checked["ready_for_issue_76"] is True
    assert checked["failed_checks"] == []
    assert all(check["passed"] is True for check in checked["checks"])
    assert all(
        {"observed", "required", "deficit"} <= set(check)
        for check in checked["checks"]
    )
    assert (
        checked["freeze_binding"]["dataset_freeze_manifest_sha256"]
        == sha256_file(DATA_DIR / "dataset_freeze.json")
    )
    payload = copy.deepcopy(checked)
    observed_digest = payload.pop("preflight_payload_sha256")
    assert observed_digest == canonical_sha256(payload)


def test_dataset_preflight_enumerates_every_required_boundary() -> None:
    checked = json.loads((DATA_DIR / "preflight_report.json").read_text(encoding="utf-8"))
    check_ids = tuple(check["id"] for check in checked["checks"])
    assert len(AUTHORITY_REQUIRED_CHECK_IDS) == 42
    assert len(set(check_ids)) == 42
    assert check_ids == AUTHORITY_REQUIRED_CHECK_IDS


def test_dataset_preflight_evaluates_complete_frozen_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = json.loads((DATA_DIR / "dataset_freeze.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(
        dataset_preflight,
        "validate_checked_dataset_freeze",
        lambda data_dir: copy.deepcopy(freeze),
    )
    report = build_dataset_preflight(DATA_DIR)
    assert report["ready_for_issue_76"] is True, report["failed_checks"]
    assert report["failed_checks"] == []
    assert all(check["passed"] is True for check in report["checks"])


def test_minimum_check_reports_exact_deficit() -> None:
    check = dataset_preflight._minimum_check(
        "topology.eval.overlap_takeover",
        17,
        20,
        "count",
    )
    assert check == {
        "id": "topology.eval.overlap_takeover",
        "observed": 17,
        "required": 20,
        "deficit": 3,
        "unit": "count",
        "passed": False,
    }


def test_gate_inventory_rejects_weakened_requirement() -> None:
    split = {
        "hard_gate_results": [
            {"id": gate_id, "observed": required, "required": required, "passed": True}
            for gate_id, required, _ in EXPECTED_GATE_SPECS
        ]
    }
    split["hard_gate_results"][0]["required"] = 1
    with pytest.raises(DatasetPreflightError, match="internally inconsistent"):
        dataset_preflight._gate_checks(split)


def test_fixed_check_inventory_rejects_missing_no_model_boundary() -> None:
    checked = _load_json("preflight_report.json")
    checks = copy.deepcopy(checked["checks"])
    checks.pop()
    with pytest.raises(DatasetPreflightError, match="fixed 42-check contract"):
        dataset_preflight._require_check_inventory(checks)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("primary_event.overlap_onset_is_handoff", True),
        ("constants_ms.local_continuity_max_gap", 1),
    ),
)
def test_contract_check_rejects_semantic_drift(field: str, value: object) -> None:
    contract = _load_json("operational_label_contract.json")
    section, name = field.split(".")
    contract[section][name] = value
    check = dataset_preflight._contract_check(contract)
    assert check["passed"] is False


def test_masking_check_rejects_diagnostic_map_drift() -> None:
    contract = _load_json("operational_label_contract.json")
    census = _load_json("topology_census.json")
    topology_rows = _load_jsonl("topology_manifest.jsonl")
    census["overall"]["mask_diagnostics"]["diagnostic_masked_region_counts"][
        "complex_overlap_region"
    ] += 1
    check = dataset_preflight._masking_check(contract, census, topology_rows)
    assert check["passed"] is False
    assert "census.diagnostic_masked_region_counts" in check["violations"]


def test_exclusive_counting_rejects_legacy_event_counts() -> None:
    contract = _load_json("operational_label_contract.json")
    census = _load_json("topology_census.json")
    topology_rows = _load_jsonl("topology_manifest.jsonl")
    census["counting_policy"]["old_r7_or_r7b_event_counts_used"] = True
    check = dataset_preflight._exclusive_counting_check(
        contract,
        census,
        topology_rows,
    )
    assert check["passed"] is False
    assert "census.old_r7_or_r7b_event_counts_forbidden" in check["violations"]


@pytest.mark.parametrize("mutation", ("sample_rate", "negative_coordinate"))
def test_annotation_coverage_rejects_unit_or_coordinate_drift(mutation: str) -> None:
    source_rows = _load_jsonl("source_manifest.jsonl")
    annotation_rows = _load_jsonl("annotation_manifest.jsonl")
    normalization_rows = _load_jsonl("normalization_manifest.jsonl")
    split = _load_json("split_manifest.json")
    selected_source_ids = {
        row["source_id"] for row in split["assignments"]["sources"]
    }
    if mutation == "sample_rate":
        source_rows[0]["sample_rate_hz"] = 8000
    else:
        normalization_rows[0]["scored_start_sample"] = -1
    check = dataset_preflight._annotation_coverage_check(
        source_rows,
        annotation_rows,
        normalization_rows,
        selected_source_ids,
    )
    assert check["passed"] is False
    assert check["deficit"] == 1


def test_hash_check_rejects_topology_identity_drift() -> None:
    freeze = _load_json("dataset_freeze.json")
    source_rows = _load_jsonl("source_manifest.jsonl")
    annotation_rows = _load_jsonl("annotation_manifest.jsonl")
    normalization_rows = _load_jsonl("normalization_manifest.jsonl")
    topology_rows = _load_jsonl("topology_manifest.jsonl")
    split = _load_json("split_manifest.json")
    topology_rows[0]["label_result_sha256"] = "0" * 64
    check = dataset_preflight._hash_checks(
        DATA_DIR,
        freeze,
        source_rows,
        annotation_rows,
        normalization_rows,
        topology_rows,
        split["assignments"]["sources"],
    )[1]
    assert check["passed"] is False
    assert topology_rows[0]["source_id"] in check["violations"]


def test_accepted_freeze_check_rejects_same_id_payload_drift() -> None:
    freeze = _load_json("dataset_freeze.json")
    freeze["freeze_payload_sha256"] = "0" * 64
    check = dataset_preflight._accepted_freeze_check(DATA_DIR, freeze)
    assert check["passed"] is False


def test_writer_replaces_stale_pass_with_failure_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight_report.json"
    output.write_text(
        json.dumps({"ready_for_issue_76": True, "failed_checks": []}),
        encoding="utf-8",
    )

    def fail_build(data_dir: Path) -> dict[str, object]:
        raise DatasetPreflightError(f"stale freeze: {data_dir}")

    monkeypatch.setattr(dataset_preflight, "build_dataset_preflight", fail_build)
    with pytest.raises(DatasetPreflightError, match="failed closed"):
        write_dataset_preflight(DATA_DIR, output)
    failed = json.loads(output.read_text(encoding="utf-8"))
    assert failed["ready_for_issue_76"] is False
    assert failed["failed_checks"][0]["observed"] is False
    assert failed["failed_checks"][0]["required"] is True
    assert failed["failed_checks"][0]["deficit"] is None
    assert "stale freeze" in failed["failed_checks"][0]["reason"]


def test_dataset_preflight_writer_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    value = {
        "dataset_freeze_id": "PSEM-STRATEGY-DATA-v1",
        "ready_for_issue_76": True,
        "failed_checks": [],
    }
    monkeypatch.setattr(
        dataset_preflight,
        "build_dataset_preflight",
        lambda data_dir: copy.deepcopy(value),
    )
    output = tmp_path / "preflight_report.json"
    write_dataset_preflight(DATA_DIR, output)
    first = output.read_bytes()
    write_dataset_preflight(DATA_DIR, output)
    assert output.read_bytes() == first


def test_checked_dataset_preflight_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = json.loads((DATA_DIR / "preflight_report.json").read_text(encoding="utf-8"))
    tampered = copy.deepcopy(expected)
    tampered["ready_for_issue_76"] = False
    report_path = tmp_path / "preflight_report.json"
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(
        dataset_preflight,
        "build_dataset_preflight",
        lambda data_dir: copy.deepcopy(expected),
    )
    with pytest.raises(DatasetPreflightError, match="not current"):
        validate_checked_dataset_preflight(DATA_DIR, report_path)
