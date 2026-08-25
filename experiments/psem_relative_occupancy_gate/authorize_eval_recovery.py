from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.psem_relative_occupancy_gate.eval_access import (
    EVAL_RECOVERY_ACCEPTED_C3_HEAD,
    EVAL_RECOVERY_ALLOWED_PATHS,
    EVAL_RECOVERY_EXTENSION_REASON,
    EVAL_RECOVERY_EXTENSION_SCHEMA_VERSION,
    EVAL_RECOVERY_REQUIRED_PATHS,
    EvalAccessError,
    _current_head,
    _git_changed_paths,
    _git_file_sha256,
    _git_is_ancestor,
    _git_is_direct_child,
    _load_eval_recovery_history,
    _tracked_worktree_is_clean,
    access_receipt_path,
    consumption_receipt_path,
    load_frozen_selection,
    recovery_extension_receipt_path,
    recovery_receipt_path,
    validate_frozen_selection_bindings,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    PACKAGE_ROOT,
    canonical_sha256,
    load_json,
    load_jsonl,
    safe_output_path,
    sha256_file,
)
from experiments.psem_relative_occupancy_gate.provenance import load_frozen_dataset


class EvalRecoveryAuthorizationError(RuntimeError):
    pass


def _load_canonical(path: Path, hash_field: str, label: str) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise EvalRecoveryAuthorizationError(f"{label} must be an object")
    payload = dict(value)
    observed = payload.pop(hash_field, None)
    if observed != canonical_sha256(payload):
        raise EvalRecoveryAuthorizationError(f"{label} hash is invalid")
    return value


def _validate_ancestor(base: str, head: str) -> None:
    if not _git_is_ancestor(base, head):
        raise EvalRecoveryAuthorizationError("recovery head does not descend from accepted C2")


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise EvalRecoveryAuthorizationError(
            "EVAL recovery authorization already exists"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _validate_manifest(
    manifest_path: Path,
    *,
    selection: dict[str, Any],
    authorization: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = load_jsonl(manifest_path)
    expected_ids = sorted(load_frozen_dataset().source_ids("PSEM-STRATEGY-EVAL"))
    observed_ids = sorted(str(row.get("source_id", "")) for row in rows)
    if observed_ids != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise EvalRecoveryAuthorizationError("opened EVAL manifest membership changed")
    for row in rows:
        payload = dict(row)
        observed = payload.pop("row_sha256", None)
        if (
            observed != canonical_sha256(payload)
            or row.get("role") != "PSEM-STRATEGY-EVAL"
            or row.get("eval_status") != "opened_once"
            or row.get("eval_selection_sha256") != selection["selection_sha256"]
            or row.get("eval_authorization_sha256")
            != authorization["authorization_sha256"]
            or row.get("config_sha256") != sha256_file(CONFIG_PATH)
        ):
            raise EvalRecoveryAuthorizationError("opened EVAL manifest binding changed")
    return rows


def _validate_claim_and_access(
    *,
    manifest_path: Path,
    access_path: Path,
    authorization_path: Path,
    authorization: dict[str, Any],
    selection: dict[str, Any],
    source_count: int,
) -> Path:
    claim_path = consumption_receipt_path(authorization_path)
    claim = _load_canonical(claim_path, "claim_sha256", "EVAL consumption receipt")
    if (
        claim.get("authorization_sha256") != authorization["authorization_sha256"]
        or claim.get("authorization_file_sha256") != sha256_file(authorization_path)
        or claim.get("accepted_c2_head") != authorization["accepted_c2_head"]
        or claim.get("manifest_output_path") != str(manifest_path)
    ):
        raise EvalRecoveryAuthorizationError("EVAL consumption binding changed")
    access = _load_canonical(access_path, "access_sha256", "EVAL access receipt")
    expected = {
        "schema_version": "psem.relative_occupancy.eval_access_receipt.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "open_count": 1,
        "selection_sha256": selection["selection_sha256"],
        "accepted_c2_head": authorization["accepted_c2_head"],
        "authorization_path": str(authorization_path),
        "authorization_file_sha256": sha256_file(authorization_path),
        "authorization_sha256": authorization["authorization_sha256"],
        "consumption_receipt_path": str(claim_path),
        "consumption_receipt_sha256": sha256_file(claim_path),
        "model_gate_verification_sha256": authorization[
            "model_gate_verification_sha256"
        ],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_count": source_count,
    }
    if any(access.get(field) != expected_value for field, expected_value in expected.items()):
        raise EvalRecoveryAuthorizationError("EVAL access binding changed")
    return claim_path


def _contract_overrides(
    selection: dict[str, Any], accepted_c2_head: str
) -> dict[str, dict[str, str]]:
    contracts = selection.get("artifact_bindings", {}).get("contract_files", {})
    if not isinstance(contracts, dict) or not contracts:
        raise EvalRecoveryAuthorizationError("DEV contract bindings are missing")
    result: dict[str, dict[str, str]] = {}
    for name, before_sha256 in contracts.items():
        relative_path = f"experiments/psem_relative_occupancy_gate/{name}"
        old_hash = _git_file_sha256(accepted_c2_head, relative_path)
        current_path = PACKAGE_ROOT / str(name)
        if old_hash != before_sha256 or not current_path.is_file():
            raise EvalRecoveryAuthorizationError("accepted DEV contract binding changed")
        after_sha256 = sha256_file(current_path)
        if after_sha256 != before_sha256:
            result[str(name)] = {
                "before_sha256": str(before_sha256),
                "after_sha256": after_sha256,
            }
    if set(result) != {
        "eval_access.py",
        "model_run_io.py",
        "run_sortformer_trace.py",
    }:
        raise EvalRecoveryAuthorizationError("recovery contract override scope is invalid")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    manifest_path = Path(args.manifest).resolve()
    access_path = Path(args.access_receipt).resolve()
    authorization_path = Path(args.eval_authorization).resolve()
    output = safe_output_path(Path(args.output))
    if access_path != access_receipt_path(manifest_path):
        raise EvalRecoveryAuthorizationError("EVAL access receipt path is not canonical")
    prior_recovery_path = recovery_receipt_path(authorization_path)
    expected_output = recovery_extension_receipt_path(authorization_path)
    if output != expected_output:
        raise EvalRecoveryAuthorizationError("EVAL recovery output path is not canonical")
    if not prior_recovery_path.is_file():
        raise EvalRecoveryAuthorizationError("prior EVAL recovery receipt is missing")
    selection = load_frozen_selection(selection_path)
    authorization = _load_canonical(
        authorization_path, "authorization_sha256", "EVAL authorization"
    )
    expected_authorization = {
        "schema_version": "psem.relative_occupancy.eval_authorization.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "authorization_state": "authorized_for_one_manifest_derivation",
        "use_limit": 1,
        "accepted_c2_head": args.accepted_c2_head,
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": sha256_file(selection_path),
        "manifest_output_path": str(manifest_path),
    }
    if any(
        authorization.get(field) != expected_value
        for field, expected_value in expected_authorization.items()
    ):
        raise EvalRecoveryAuthorizationError("original EVAL authorization binding changed")
    try:
        prior_recovery = _load_eval_recovery_history(
            authorization_path,
            authorization=authorization,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_path,
        )
    except EvalAccessError as exc:
        raise EvalRecoveryAuthorizationError(
            "prior EVAL recovery binding changed"
        ) from exc
    recovery_head = _current_head()
    if recovery_head == args.accepted_c2_head or not _tracked_worktree_is_clean(
        manifest_path.parent
    ):
        raise EvalRecoveryAuthorizationError("recovery candidate is not an exact clean commit")
    _validate_ancestor(args.accepted_c2_head, recovery_head)
    if not _git_is_direct_child(EVAL_RECOVERY_ACCEPTED_C3_HEAD, recovery_head):
        raise EvalRecoveryAuthorizationError(
            "recovery head is not the direct C4 child of accepted C3"
        )
    changed_paths = _git_changed_paths(args.accepted_c2_head, recovery_head)
    if (
        not EVAL_RECOVERY_REQUIRED_PATHS <= changed_paths
        or not changed_paths <= EVAL_RECOVERY_ALLOWED_PATHS
    ):
        raise EvalRecoveryAuthorizationError("recovery candidate changed disallowed paths")
    contract_overrides = _contract_overrides(selection, args.accepted_c2_head)
    try:
        validate_frozen_selection_bindings(
            selection_path, selection, contract_overrides=contract_overrides
        )
    except EvalAccessError as exc:
        raise EvalRecoveryAuthorizationError(
            "accepted DEV evidence binding changed"
        ) from exc
    rows = _validate_manifest(
        manifest_path, selection=selection, authorization=authorization
    )
    claim_path = _validate_claim_and_access(
        manifest_path=manifest_path,
        access_path=access_path,
        authorization_path=authorization_path,
        authorization=authorization,
        selection=selection,
        source_count=len(rows),
    )
    aggregate_names = {
        "sortformer_model_receipt.json",
        "lseend_model_receipt.json",
        "eval_metrics.json",
        "eval_verification.json",
        "gate1_metrics.json",
        "gate2_metrics.json",
        "product_frontiers.json",
        "topology_slices.json",
        "latency_breakdown.json",
        "FINAL_DECISION.md",
    }
    existing_aggregates = sorted(
        name for name in aggregate_names if (manifest_path.parent / name).exists()
    )
    if existing_aggregates:
        raise EvalRecoveryAuthorizationError("EVAL aggregate already exists before recovery")
    repo_root = PACKAGE_ROOT.parent.parent
    changed_files: dict[str, dict[str, str | None]] = {}
    for relative_path in sorted(changed_paths):
        current_path = (repo_root / relative_path).resolve()
        changed_files[relative_path] = {
            "before_sha256": _git_file_sha256(args.accepted_c2_head, relative_path),
            "after_sha256": sha256_file(current_path),
        }
    payload: dict[str, Any] = {
        "schema_version": EVAL_RECOVERY_EXTENSION_SCHEMA_VERSION,
        "role": "PSEM-STRATEGY-EVAL",
        "recovery_state": "authorized_for_same_opened_manifest_resume",
        "recovery_reason": EVAL_RECOVERY_EXTENSION_REASON,
        "accepted_c2_head": args.accepted_c2_head,
        "recovery_head": recovery_head,
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": sha256_file(authorization_path),
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": sha256_file(selection_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "access_receipt_path": str(access_path),
        "access_receipt_sha256": sha256_file(access_path),
        "consumption_receipt_path": str(claim_path),
        "consumption_receipt_sha256": sha256_file(claim_path),
        "manifest_open_count": 1,
        "additional_manifest_derivations": 0,
        "prior_eval_aggregate_count": 0,
        "changed_files": changed_files,
        "contract_overrides": contract_overrides,
        "recovery_sequence": 2,
        "prior_recovery_path": str(prior_recovery_path),
        "prior_recovery_file_sha256": sha256_file(prior_recovery_path),
        "prior_recovery_sha256": prior_recovery["recovery_sha256"],
        "prior_recovery_head": prior_recovery["recovery_head"],
        "prior_recovery_result": "failed_before_eval_aggregate",
        "failed_family": "streaming_sortformer",
        "failed_source_id": "alimeeting_R1021_M1940",
        "completed_sortformer_source_count": 2,
        "lseend_started": False,
    }
    payload["recovery_sha256"] = canonical_sha256(payload)
    _write_json_exclusive(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--access-receipt", required=True)
    parser.add_argument("--eval-authorization", required=True)
    parser.add_argument("--accepted-c2-head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    print({"output": str(Path(args.output).resolve()), "recovery_head": result["recovery_head"]})


if __name__ == "__main__":
    main()
