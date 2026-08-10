from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def document_sha256(document: dict[str, Any]) -> str:
    payload = copy.deepcopy(document)
    payload.pop("self_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def with_self_sha256(document: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(document)
    payload["self_sha256"] = document_sha256(payload)
    return payload


def self_sha256_valid(document: dict[str, Any]) -> bool:
    expected = document.get("self_sha256")
    return isinstance(expected, str) and expected == document_sha256(document)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def verify_file_identity(path: Path, expected_sha256: str, expected_size: int | None) -> list[str]:
    if not path.is_file():
        return [f"missing file: {path}"]
    errors: list[str] = []
    if expected_size is not None and path.stat().st_size != expected_size:
        errors.append(
            f"size mismatch for {path}: {path.stat().st_size} != {expected_size}"
        )
    actual = sha256_file(path)
    if actual != expected_sha256:
        errors.append(f"sha256 mismatch for {path}: {actual} != {expected_sha256}")
    return errors
