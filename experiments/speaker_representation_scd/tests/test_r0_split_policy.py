from __future__ import annotations

import copy
from pathlib import Path

from experiments.speaker_representation_scd.provenance import load_json, with_self_sha256
from experiments.speaker_representation_scd.schemas import component_leakage, validate_document

ROOT = Path(__file__).resolve().parents[1]


def _document(relative: str) -> dict:
    return load_json(ROOT / relative)


def test_frozen_split_has_no_connected_component_leakage() -> None:
    split = _document("data/split_contract.json")
    assert not component_leakage(split["component_assignments"])
    assert not validate_document(split, "split_contract")


def test_ami_is_development_only_and_voxconverse_has_no_legacy_reference() -> None:
    split = _document("data/split_contract.json")
    ami = [
        row
        for row in split["component_assignments"]
        if row["namespace"] == "ami_corpus"
    ]
    assert ami == [
        {
            "namespace": "ami_corpus",
            "component_id": "all_meetings_gt_exposed",
            "tier": "development_known",
            "source_id": "legacy-common-gt-v1",
        }
    ]
    legacy_root = ROOT.parent / "speaker_turn_boundary"
    suffixes = {".json", ".jsonl", ".md", ".py"}
    assert not any(
        "voxconverse" in path.read_text(encoding="utf-8", errors="ignore").lower()
        for path in legacy_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def test_speaker_component_cross_tier_mutation_is_rejected() -> None:
    split = _document("data/split_contract.json")
    changed = copy.deepcopy(split)
    changed["component_assignments"].append(
        {
            "namespace": "jvs_speaker",
            "component_id": "jvs050",
            "tier": "development_known",
            "source_id": "jvs-development",
        }
    )
    errors = validate_document(with_self_sha256(changed), "split_contract")
    assert any("cross-tier leakage" in error and "jvs050" in error for error in errors)


def test_legacy_manifest_cannot_be_reclassified_confirmatory() -> None:
    split = _document("data/split_contract.json")
    changed = copy.deepcopy(split)
    changed["legacy_manifest"]["tier"] = "confirmatory_test"
    errors = validate_document(with_self_sha256(changed), "split_contract")
    assert any("legacy data must be development-known" in error for error in errors)


def test_valid_but_different_legacy_hash_is_rejected() -> None:
    source = _document("data/source_ledger.json")
    changed = copy.deepcopy(source)
    changed["sources"][0]["artifact_sha256"] = "4" * 64
    errors = validate_document(with_self_sha256(changed), "source_ledger")
    assert any("legacy byte identity changed" in error for error in errors)


def test_rehashed_voxconverse_and_jvs_identity_mutations_are_rejected() -> None:
    source = _document("data/source_ledger.json")
    split = _document("data/split_contract.json")
    changed_source = copy.deepcopy(source)
    vox = next(
        item
        for item in changed_source["sources"]
        if item["source_id"] == "voxconverse-v03-confirmatory-natural"
    )
    vox["selection"]["annotation_repository_revision"] = "1" * 40
    source_errors = validate_document(with_self_sha256(changed_source), "source_ledger")
    changed_split = copy.deepcopy(split)
    jvs = next(
        item
        for item in changed_split["component_assignments"]
        if item["namespace"] == "jvs_speaker" and item["tier"] == "confirmatory_test"
    )
    jvs["component_id"] = "jvs099"
    split_errors = validate_document(with_self_sha256(changed_split), "split_contract")
    assert any("frozen R0 contract" in error for error in source_errors)
    assert any("frozen R0 contract" in error for error in split_errors)


def test_confirmatory_access_entry_is_rejected_while_sealed() -> None:
    access = _document("data/confirmatory_access_policy.json")
    changed = copy.deepcopy(access)
    changed["access_ledger"].append(
        {"at_utc": "2026-08-10T00:00:00Z", "operation": "audio_content"}
    )
    errors = validate_document(with_self_sha256(changed), "confirmatory_access_policy")
    assert any("must be empty while sealed" in error for error in errors)


def test_opened_confirmatory_state_requires_complete_lock() -> None:
    access = _document("data/confirmatory_access_policy.json")
    changed = copy.deepcopy(access)
    changed["state"] = "opened"
    errors = validate_document(with_self_sha256(changed), "confirmatory_access_policy")
    assert any("seal-only" in error for error in errors)


def test_hash_shaped_placeholder_unlock_is_rejected() -> None:
    access = _document("data/confirmatory_access_policy.json")
    changed = copy.deepcopy(access)
    changed["state"] = "opened"
    for key in (
        "run_contract_sha256",
        "code_sha256",
        "model_registry_sha256",
        "split_contract_sha256",
        "analysis_contract_sha256",
        "evaluation_environment_sha256",
        "development_promotion_ledger_sha256",
        "verifier_sha256",
    ):
        changed["lock"][key] = "0" * 64
    changed["lock"]["locked_encoder_configuration_ids"] = ["a", "b", "c", "d"]
    changed["lock"]["locked_at_utc"] = "x"
    errors = validate_document(with_self_sha256(changed), "confirmatory_access_policy")
    assert any("seal-only" in error for error in errors)
    assert any("must remain empty" in error for error in errors)


def test_prelock_allowlist_cannot_include_forbidden_content() -> None:
    access = _document("data/confirmatory_access_policy.json")
    changed = copy.deepcopy(access)
    changed["allowed_prelock_operations"].append("scores")
    errors = validate_document(with_self_sha256(changed), "confirmatory_access_policy")
    assert any("overlaps forbidden operations" in error for error in errors)
