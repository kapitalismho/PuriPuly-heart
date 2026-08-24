from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from experiments.psem_training_strategy_gate.preflight import (
    _binding,
    _safe_git_state,
    canonical_sha256,
    sha256_file,
)
from experiments.psem_training_strategy_gate.runtime_contract import (
    RUNTIME_ARTIFACT_PATHS,
    RUNTIME_CHECK_ARTIFACT_PATHS,
    RUNTIME_RECEIPT_ROLES,
    RuntimeEvidenceError,
    runtime_artifact_checks,
)


class ReceiptContractError(RuntimeError):
    pass


def current_binding() -> dict[str, Any]:
    git = _safe_git_state()
    if git.get("dirty") is not False:
        raise ReceiptContractError("runtime receipts require a clean Git candidate")
    binding = _binding(git)
    if any(value is None for value in binding.values()):
        raise ReceiptContractError("current experiment binding is incomplete")
    return binding


def check(
    check_id: str,
    passed: bool,
    *,
    expected: Any,
    observed: Any,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "expected": expected,
        "observed": observed,
    }


def runtime_receipt(
    receipt_name: str,
    artifact_role: str,
    *,
    details: Mapping[str, Any],
    validation_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        receipt_name not in RUNTIME_RECEIPT_ROLES
        or artifact_role != RUNTIME_RECEIPT_ROLES[receipt_name]
        or not isinstance(details, Mapping)
        or not details
    ):
        raise ReceiptContractError("runtime receipt identity and details are required")
    rows = _checks_from_details(
        receipt_name,
        details,
        validation_context=validation_context,
    )
    payload = {
        "schema_version": 1,
        "artifact_role": artifact_role,
        "status": "pass" if all(row["passed"] is True for row in rows) else "fail",
        "generated_at": datetime.now(UTC).isoformat(),
        "binding": current_binding(),
        "checks": rows,
        "details": dict(details),
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def write_runtime_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    validation_context: Mapping[str, Any] | None = None,
) -> None:
    receipt_name = path.stem
    if receipt_name not in RUNTIME_RECEIPT_ROLES:
        raise ReceiptContractError("runtime receipt path has an unknown identity")
    payload = dict(receipt)
    digest = payload.pop("payload_sha256", None)
    rows = receipt.get("checks")
    try:
        generated_at = datetime.fromisoformat(str(receipt.get("generated_at")))
    except ValueError as error:
        raise ReceiptContractError("runtime receipt timestamp is invalid") from error
    expected_rows = _checks_from_details(
        receipt_name,
        receipt.get("details"),
        validation_context=validation_context,
    )
    if (
        set(receipt)
        != {
            "schema_version",
            "artifact_role",
            "status",
            "generated_at",
            "binding",
            "checks",
            "details",
            "payload_sha256",
        }
        or receipt.get("schema_version") != 1
        or receipt.get("artifact_role") != RUNTIME_RECEIPT_ROLES[receipt_name]
        or receipt.get("binding") != current_binding()
        or rows != expected_rows
        or receipt.get("status")
        != ("pass" if all(row["passed"] is True for row in expected_rows) else "fail")
        or generated_at.tzinfo is None
        or digest != canonical_sha256(payload)
    ):
        raise ReceiptContractError("runtime receipt is not authoritative or internally consistent")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checks_from_details(
    receipt_name: str,
    details: Mapping[str, Any] | Any,
    *,
    validation_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(details, Mapping) or set(details) != {"artifacts"}:
        raise ReceiptContractError("runtime receipt details are missing")
    descriptors = details.get("artifacts")
    expected_relatives = tuple(Path(value) for value in RUNTIME_ARTIFACT_PATHS[receipt_name])
    if not isinstance(descriptors, list) or len(descriptors) != len(expected_relatives):
        raise ReceiptContractError("runtime receipt artifact inventory differs from its contract")
    check_relative = Path(RUNTIME_CHECK_ARTIFACT_PATHS[receipt_name])
    check_artifact: Mapping[str, Any] | None = None
    roots = []
    for descriptor, relative in zip(descriptors, expected_relatives, strict=True):
        try:
            if not isinstance(descriptor, Mapping):
                raise ReceiptContractError("runtime check artifact descriptor is invalid")
            path = Path(str(descriptor["path"])).resolve()
            if tuple(path.parts[-len(relative.parts) :]) != relative.parts:
                raise ReceiptContractError("runtime check artifact path differs from its contract")
            expected_keys = {"path", "sha256", "size_bytes"}
            if relative.suffix == ".json":
                expected_keys.add("canonical_sha256")
            if (
                set(descriptor) != expected_keys
                or str(path) != str(descriptor["path"])
                or not path.is_file()
                or not isinstance(descriptor["size_bytes"], int)
                or isinstance(descriptor["size_bytes"], bool)
                or descriptor["size_bytes"] < 0
                or path.stat().st_size != descriptor["size_bytes"]
                or re.fullmatch(r"[0-9a-f]{64}", str(descriptor["sha256"])) is None
                or sha256_file(path) != descriptor["sha256"]
            ):
                raise ReceiptContractError("runtime check artifact descriptor is invalid")
            root = path
            for _ in relative.parts:
                root = root.parent
            roots.append(root)
            if relative.suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                if (
                    not isinstance(value, Mapping)
                    or re.fullmatch(r"[0-9a-f]{64}", str(descriptor["canonical_sha256"])) is None
                    or canonical_sha256(value) != descriptor["canonical_sha256"]
                ):
                    raise ReceiptContractError("runtime check artifact content is invalid")
                if relative == check_relative:
                    check_artifact = value
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ReceiptContractError("runtime check artifact cannot be validated") from error
    if len(set(roots)) != 1 or check_artifact is None:
        raise ReceiptContractError("runtime receipt must bind one authoritative check artifact")
    try:
        return runtime_artifact_checks(
            receipt_name,
            check_artifact,
            validation_context=validation_context,
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        RuntimeEvidenceError,
    ) as error:
        raise ReceiptContractError(str(error)) from error
