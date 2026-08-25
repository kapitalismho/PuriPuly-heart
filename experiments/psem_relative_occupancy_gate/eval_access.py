from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    PACKAGE_ROOT,
    ExperimentError,
    canonical_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    strict_regular_file,
)
from experiments.psem_relative_occupancy_gate.provenance import load_frozen_dataset


class EvalAccessError(RuntimeError):
    pass


def _strict_file(path: Path, field: str) -> Path:
    try:
        return strict_regular_file(path, field)
    except ExperimentError as exc:
        raise EvalAccessError(str(exc)) from exc


def load_frozen_selection(path: Path) -> dict[str, Any]:
    path = _strict_file(path, "DEV selection receipt")
    value = load_json(path)
    if not isinstance(value, dict):
        raise EvalAccessError("DEV selection receipt must be an object")
    payload = dict(value)
    observed_hash = payload.pop("selection_sha256", None)
    if (
        value.get("schema_version") != "psem.relative_occupancy.dev_selection.v1"
        or value.get("role") != "PSEM-STRATEGY-DEV"
        or value.get("eval_status") != "sealed"
        or value.get("eval_open_authorized") is not False
        or value.get("eval_open_count") != 0
        or observed_hash != canonical_sha256(payload)
    ):
        raise EvalAccessError("DEV selection receipt is invalid or not sealed")
    return value


def access_receipt_path(manifest_output: Path) -> Path:
    return manifest_output.with_name(f"{manifest_output.stem}_access_receipt.json")


def consumption_receipt_path(authorization_path: Path) -> Path:
    return authorization_path.with_name(f"{authorization_path.stem}_consumed.json")


def _current_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_frozen_selection_bindings(
    selection_path: Path, selection: dict[str, Any]
) -> None:
    root = selection_path.resolve().parent
    bindings = selection.get("artifact_bindings")
    if not isinstance(bindings, dict):
        raise EvalAccessError("DEV selection artifact bindings are missing")
    contracts = bindings.get("contract_files")
    if not isinstance(contracts, dict) or not contracts:
        raise EvalAccessError("DEV selection contract bindings are missing")
    for name, expected_hash in contracts.items():
        path = PACKAGE_ROOT / str(name)
        if (
            Path(str(name)).name != str(name)
            or _strict_file(path, f"DEV contract {name}") != path.resolve()
            or sha256_file(path) != expected_hash
        ):
            raise EvalAccessError(f"DEV selection contract changed: {name}")
    expected = {
        "gate0_metrics_sha256": root / "gate0_oracle_metrics.json",
        "gate0_verification_sha256": root / "gate0_verification.json",
        "gate1_metrics_sha256": root / "gate1_metrics.json",
        "gate1_product_frontier_sha256": root / "gate1_product_frontier.json",
        "gate1_topology_slices_sha256": root / "gate1_topology_slices.json",
        "gate1_latency_breakdown_sha256": root / "gate1_latency_breakdown.json",
        "gate1_event_ledger_sha256": root / "gate1_event_ledger.jsonl",
        "gate2_metrics_sha256": root / "gate2_metrics.json",
        "gate2_event_ledger_sha256": root / "gate2_event_ledger.jsonl",
        "product_frontiers_sha256": root / "product_frontiers.json",
        "topology_slices_sha256": root / "topology_slices.json",
        "latency_breakdown_sha256": root / "latency_breakdown.json",
    }
    for key, path in expected.items():
        if _strict_file(path, f"DEV artifact {path.name}") != path.resolve() or sha256_file(
            path
        ) != bindings.get(key):
            raise EvalAccessError(f"DEV selection artifact changed: {path.name}")
    receipts = bindings.get("model_receipts")
    if not isinstance(receipts, dict):
        raise EvalAccessError("DEV selection model bindings are missing")
    for family, name in (
        ("sortformer", "sortformer_model_receipt.json"),
        ("lseend", "lseend_model_receipt.json"),
    ):
        path = root / name
        if _strict_file(path, f"DEV model receipt {family}") != path.resolve() or sha256_file(
            path
        ) != receipts.get(family):
            raise EvalAccessError(f"DEV selection model receipt changed: {family}")
    manifest = Path(str(selection.get("manifest_path", ""))).resolve()
    if (
        _strict_file(manifest, "DEV manifest") != manifest
        or sha256_file(manifest) != selection.get("manifest_sha256")
        or sha256_file(CONFIG_PATH) != selection.get("config_sha256")
    ):
        raise EvalAccessError("DEV selection manifest or configuration changed")


def load_eval_authorization(
    path: Path,
    *,
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, Any]:
    path = _strict_file(path, "EVAL authorization")
    value = load_json(path)
    if not isinstance(value, dict):
        raise EvalAccessError("EVAL authorization must be an object")
    payload = dict(value)
    observed_hash = payload.pop("authorization_sha256", None)
    expected = {
        "schema_version": "psem.relative_occupancy.eval_authorization.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "authorization_state": "authorized_for_one_manifest_derivation",
        "use_limit": 1,
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": sha256_file(selection_path),
        "manifest_output_path": str(manifest_output.resolve()),
    }
    if any(
        value.get(field) != expected_value for field, expected_value in expected.items()
    ) or observed_hash != canonical_sha256(payload):
        raise EvalAccessError("EVAL authorization binding mismatch")
    candidate_head = value.get("accepted_c2_head")
    verification_path = Path(str(value.get("model_gate_verification_path", ""))).resolve()
    if (
        not isinstance(candidate_head, str)
        or len(candidate_head) != 40
        or candidate_head != _current_head()
        or verification_path
        != selection_path.resolve().parent / "model_gate_verification.json"
        or _strict_file(verification_path, "model-gate verification")
        != verification_path
        or sha256_file(verification_path) != value.get("model_gate_verification_sha256")
    ):
        raise EvalAccessError("EVAL authorization lacks the accepted C2 head")
    validate_frozen_selection_bindings(selection_path, selection)
    return value


def validate_unused_eval_authorization(
    *,
    selection_path: Path,
    authorization_path: Path,
    manifest_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path = selection_path.resolve()
    authorization_path = authorization_path.resolve()
    manifest_output = manifest_output.resolve()
    selection = load_frozen_selection(selection_path)
    authorization = load_eval_authorization(
        authorization_path,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_output,
    )
    if (
        manifest_output.exists()
        or access_receipt_path(manifest_output).exists()
        or consumption_receipt_path(authorization_path).exists()
    ):
        raise EvalAccessError("EVAL authorization has already been consumed")
    return selection, authorization


def claim_eval_authorization(
    *,
    selection_path: Path,
    authorization_path: Path,
    manifest_output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection, authorization = validate_unused_eval_authorization(
        selection_path=selection_path,
        authorization_path=authorization_path,
        manifest_output=manifest_output,
    )
    claim_path = consumption_receipt_path(authorization_path.resolve())
    payload = {
        "schema_version": "psem.relative_occupancy.eval_consumption.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": sha256_file(authorization_path.resolve()),
        "accepted_c2_head": authorization["accepted_c2_head"],
        "manifest_output_path": str(manifest_output.resolve()),
    }
    payload["claim_sha256"] = canonical_sha256(payload)
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise EvalAccessError("EVAL authorization has already been consumed") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return selection, authorization, payload


def validate_opened_eval_manifest(
    *,
    manifest_path: Path,
    access_path: Path,
    selection_path: Path,
    authorization_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    access_path = access_path.resolve()
    selection_path = selection_path.resolve()
    authorization_path = authorization_path.resolve()
    selection = load_frozen_selection(selection_path)
    authorization = load_eval_authorization(
        authorization_path,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_path,
    )
    claim_path = consumption_receipt_path(authorization_path)
    _strict_file(manifest_path, "opened EVAL manifest")
    _strict_file(access_path, "EVAL access receipt")
    _strict_file(claim_path, "EVAL authorization consumption receipt")
    claim = load_json(claim_path)
    if not isinstance(claim, dict):
        raise EvalAccessError("EVAL authorization consumption receipt is missing")
    claim_payload = dict(claim)
    claim_hash = claim_payload.pop("claim_sha256", None)
    if (
        claim_hash != canonical_sha256(claim_payload)
        or claim.get("authorization_sha256") != authorization["authorization_sha256"]
        or claim.get("accepted_c2_head") != authorization["accepted_c2_head"]
        or claim.get("manifest_output_path") != str(manifest_path)
    ):
        raise EvalAccessError("EVAL authorization consumption binding mismatch")
    rows = load_jsonl(manifest_path)
    expected_ids = sorted(load_frozen_dataset().source_ids("PSEM-STRATEGY-EVAL"))
    observed_ids = sorted(str(value.get("source_id", "")) for value in rows)
    if observed_ids != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise EvalAccessError("EVAL manifest does not cover the exact frozen EVAL role")
    for row in rows:
        payload = dict(row)
        row_hash = payload.pop("row_sha256", None)
        if (
            row_hash != canonical_sha256(payload)
            or row.get("role") != "PSEM-STRATEGY-EVAL"
            or row.get("eval_status") != "opened_once"
            or row.get("eval_selection_sha256") != selection["selection_sha256"]
            or row.get("eval_authorization_sha256")
            != authorization["authorization_sha256"]
            or row.get("config_sha256") != sha256_file(CONFIG_PATH)
        ):
            raise EvalAccessError(f"EVAL manifest binding mismatch: {row.get('source_id')}")
    access = load_json(access_path)
    access_payload = dict(access) if isinstance(access, dict) else {}
    access_hash = access_payload.pop("access_sha256", None)
    if (
        not isinstance(access, dict)
        or access_hash != canonical_sha256(access_payload)
        or access.get("schema_version")
        != "psem.relative_occupancy.eval_access_receipt.v1"
        or access.get("role") != "PSEM-STRATEGY-EVAL"
        or access.get("open_count") != 1
        or access.get("selection_sha256") != selection["selection_sha256"]
        or access.get("accepted_c2_head") != authorization["accepted_c2_head"]
        or access.get("authorization_path") != str(authorization_path)
        or access.get("authorization_file_sha256") != sha256_file(authorization_path)
        or access.get("authorization_sha256") != authorization["authorization_sha256"]
        or access.get("consumption_receipt_path") != str(claim_path)
        or access.get("consumption_receipt_sha256") != sha256_file(claim_path)
        or access.get("model_gate_verification_sha256")
        != authorization["model_gate_verification_sha256"]
        or access.get("manifest_path") != str(manifest_path)
        or access.get("manifest_sha256") != sha256_file(manifest_path)
        or access.get("source_count") != len(expected_ids)
    ):
        raise EvalAccessError("EVAL access receipt binding mismatch")
    return rows, selection
