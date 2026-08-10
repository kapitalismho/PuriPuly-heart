from __future__ import annotations

import json
from pathlib import Path

from experiments.speaker_representation_scd.provenance import (
    canonical_json_bytes,
    document_sha256,
    load_json,
    self_sha256_valid,
    verify_file_identity,
    with_self_sha256,
)
from experiments.speaker_representation_scd.schemas import AUTHORITY_SHA256, validate_document
from experiments.speaker_representation_scd.validate_r0 import DEFAULT_PATHS

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]


def test_canonical_json_is_order_independent() -> None:
    left = {"b": [2, 1], "a": {"z": True, "x": None}}
    right = {"a": {"x": None, "z": True}, "b": [2, 1]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)


def test_all_contract_self_hashes_are_valid() -> None:
    for role, relative in DEFAULT_PATHS.items():
        document = load_json(ROOT / relative)
        assert document["artifact_role"] == role
        assert self_sha256_valid(document)
        assert document_sha256(document) == document["self_sha256"]


def test_self_hash_detects_tampering() -> None:
    document = load_json(ROOT / "configs/protocol/analysis_contract.json")
    document["uncertainty"]["bootstrap_seed"] += 1
    assert not self_sha256_valid(document)
    assert any("canonical content" in error for error in validate_document(document, "analysis_contract"))


def test_rehash_does_not_hide_authority_identity_mutation() -> None:
    document = load_json(ROOT / "configs/protocol/r0_protocol.json")
    document["authority"]["sha256"] = "3" * 64
    document = with_self_sha256(document)
    errors = validate_document(document, "r0_protocol")
    assert any(AUTHORITY_SHA256 in error for error in errors)


def test_existing_legacy_manifest_byte_identity() -> None:
    source = load_json(ROOT / "data/source_ledger.json")["sources"][0]
    path = REPO_ROOT / source["source_url"]
    assert not verify_file_identity(path, source["artifact_sha256"], source["artifact_size_bytes"])


def test_file_identity_rejects_size_and_hash_mutation(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"known")
    assert verify_file_identity(path, "0" * 64, 6)


def test_contract_json_round_trips_without_nonfinite_values() -> None:
    for relative in DEFAULT_PATHS.values():
        document = load_json(ROOT / relative)
        encoded = canonical_json_bytes(document)
        assert json.loads(encoded) == document
