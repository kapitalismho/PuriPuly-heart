from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.psem_sortformer_adaptation_depth.preflight import (
    REPOSITORY_ROOT,
    canonical_sha256,
)

AUTHORITY_PIN = "eba82c5a39421b7c8d619cfd971720d8b35b19c8d198605e6e5c0dd09fcd0a97"


class AuthorityRegistryError(RuntimeError):
    pass


def authority_registry_root() -> Path:
    raw = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common = Path(raw)
    if not common.is_absolute():
        common = (REPOSITORY_ROOT / common).resolve()
    else:
        common = common.resolve()
    if not common.is_dir():
        raise AuthorityRegistryError("Git common directory is unavailable")
    return common / "psem-sortformer-adaptation-depth" / AUTHORITY_PIN


def _record_path(kind: str, payload_sha256: str) -> Path:
    if (
        not kind
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in kind)
        or len(payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in payload_sha256)
    ):
        raise AuthorityRegistryError("execution record identity is invalid")
    return authority_registry_root() / "executions" / kind / f"{payload_sha256}.json"


def register_execution(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    payload_sha256 = payload.get("payload_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if not isinstance(payload_sha256, str) or payload_sha256 != canonical_sha256(unsigned):
        raise AuthorityRegistryError("execution payload is not content-bound")
    path = _record_path(kind, payload_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_authority_execution_record",
        "authority_pin": AUTHORITY_PIN,
        "kind": kind,
        "payload_sha256": payload_sha256,
        "payload": dict(payload),
    }
    encoded = (
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise AuthorityRegistryError("execution registry contains a digest collision")
    return {
        "authority_registry_record": str(path),
        "authority_registry_record_sha256": canonical_sha256(record),
    }


def require_registered_execution(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    payload_sha256 = payload.get("payload_sha256")
    if not isinstance(payload_sha256, str):
        raise AuthorityRegistryError("execution payload digest is absent")
    path = _record_path(kind, payload_sha256)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorityRegistryError("execution is absent from the authority registry") from exc
    expected = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_authority_execution_record",
        "authority_pin": AUTHORITY_PIN,
        "kind": kind,
        "payload_sha256": payload_sha256,
        "payload": dict(payload),
    }
    if record != expected:
        raise AuthorityRegistryError("registered execution payload differs")
    return record
