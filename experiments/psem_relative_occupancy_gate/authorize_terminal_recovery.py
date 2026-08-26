from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.psem_relative_occupancy_gate.authorize_eval_recovery import (
    _load_canonical,
    _validate_claim_and_access,
    _validate_manifest,
    _validate_model_aggregates,
)
from experiments.psem_relative_occupancy_gate.eval_access import (
    EVAL_PRE_REPAIR_TERMINAL_HEAD,
    EVAL_TERMINAL_AUTHORITY_PIN,
    EVAL_TERMINAL_MUTABLE_OUTPUT_NAMES,
    EVAL_TERMINAL_RECOVERY_ALLOWED_PATHS,
    EVAL_TERMINAL_RECOVERY_REASON,
    EVAL_TERMINAL_RECOVERY_REQUIRED_PATHS,
    EVAL_TERMINAL_RECOVERY_SCHEMA_VERSION,
    _current_head,
    _git_changed_paths,
    _git_file_sha256,
    _git_is_direct_child,
    _load_eval_recovery_finalization_history,
    _recovery_expected,
    _terminal_prior_artifact_hashes,
    _tracked_worktree_is_clean,
    access_receipt_path,
    consumption_receipt_path,
    load_frozen_selection,
    recovery_finalization_receipt_path,
    terminal_recovery_consumption_path,
    terminal_recovery_receipt_path,
    validate_eval_result_directory,
    validate_frozen_selection_bindings,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    canonical_sha256,
    safe_output_path,
    sha256_file,
)


class TerminalRecoveryAuthorizationError(RuntimeError):
    pass


def _contract_overrides(
    selection: dict[str, Any], recovery_head: str
) -> dict[str, dict[str, str]]:
    contracts = selection.get("artifact_bindings", {}).get("contract_files", {})
    if not isinstance(contracts, dict) or not contracts:
        raise TerminalRecoveryAuthorizationError("DEV contract bindings are missing")
    result: dict[str, dict[str, str]] = {}
    for name, before_sha256 in contracts.items():
        relative_path = f"experiments/psem_relative_occupancy_gate/{name}"
        after_sha256 = _git_file_sha256(recovery_head, relative_path)
        if after_sha256 is None:
            raise TerminalRecoveryAuthorizationError("repair contract file is missing")
        if after_sha256 != before_sha256:
            result[str(name)] = {
                "before_sha256": str(before_sha256),
                "after_sha256": after_sha256,
            }
    return result


def _repair_changed_files(recovery_head: str) -> dict[str, dict[str, str | None]]:
    paths = _git_changed_paths(EVAL_PRE_REPAIR_TERMINAL_HEAD, recovery_head)
    if (
        not EVAL_TERMINAL_RECOVERY_REQUIRED_PATHS <= paths
        or not paths <= EVAL_TERMINAL_RECOVERY_ALLOWED_PATHS
    ):
        raise TerminalRecoveryAuthorizationError("terminal repair path scope is invalid")
    return {
        path: {
            "before_sha256": _git_file_sha256(EVAL_PRE_REPAIR_TERMINAL_HEAD, path),
            "after_sha256": _git_file_sha256(recovery_head, path),
        }
        for path in sorted(paths)
    }


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise TerminalRecoveryAuthorizationError(
            "terminal recovery authorization already exists"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    manifest_path = Path(args.manifest).resolve()
    access_path = Path(args.access_receipt).resolve()
    authorization_path = Path(args.eval_authorization).resolve()
    output = safe_output_path(Path(args.output))
    if output != terminal_recovery_receipt_path(authorization_path):
        raise TerminalRecoveryAuthorizationError("terminal recovery path is not canonical")
    if access_path != access_receipt_path(manifest_path):
        raise TerminalRecoveryAuthorizationError("EVAL access path is not canonical")
    if args.authority_pin != EVAL_TERMINAL_AUTHORITY_PIN:
        raise TerminalRecoveryAuthorizationError("owner amendment authority pin mismatch")
    if args.pre_repair_head != EVAL_PRE_REPAIR_TERMINAL_HEAD:
        raise TerminalRecoveryAuthorizationError("pre-repair terminal head mismatch")
    recovery_head = _current_head()
    if (
        not _git_is_direct_child(EVAL_PRE_REPAIR_TERMINAL_HEAD, recovery_head)
        or not _tracked_worktree_is_clean(manifest_path.parent)
    ):
        raise TerminalRecoveryAuthorizationError(
            "terminal recovery requires the exact clean direct repair child"
        )
    if terminal_recovery_consumption_path(authorization_path).exists():
        raise TerminalRecoveryAuthorizationError("terminal recovery was already consumed")
    for name in ("gate1_event_ledger.jsonl", "gate2_event_ledger.jsonl"):
        if (manifest_path.parent / name).exists():
            raise TerminalRecoveryAuthorizationError("EVAL source ledger already exists")
    selection = load_frozen_selection(selection_path)
    authorization = _load_canonical(
        authorization_path, "authorization_sha256", "EVAL authorization"
    )
    manifest = _validate_manifest(
        manifest_path, selection=selection, authorization=authorization
    )
    claim_path = _validate_claim_and_access(
        manifest_path=manifest_path,
        access_path=access_path,
        authorization_path=authorization_path,
        authorization=authorization,
        selection=selection,
        source_count=len(manifest),
    )
    prior = _load_eval_recovery_finalization_history(
        authorization_path,
        authorization=authorization,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_path,
    )
    overrides = _contract_overrides(selection, recovery_head)
    validate_frozen_selection_bindings(
        selection_path, selection, contract_overrides=overrides
    )
    validate_eval_result_directory(
        manifest_path=manifest_path,
        authorization_path=authorization_path,
        require_finalization=True,
        allow_final_outputs=True,
    )
    prior_artifacts = _terminal_prior_artifact_hashes(EVAL_PRE_REPAIR_TERMINAL_HEAD)
    for name, expected_hash in prior_artifacts.items():
        if sha256_file(manifest_path.parent / name) != expected_hash:
            raise TerminalRecoveryAuthorizationError(
                "pre-repair terminal artifact binding changed"
            )
    model_aggregates = _validate_model_aggregates(
        manifest_path=manifest_path,
        manifest=manifest,
        access_path=access_path,
    )
    prior_path = recovery_finalization_receipt_path(authorization_path)
    receipt = {
        **_recovery_expected(
            authorization_path=authorization_path,
            authorization=authorization,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_path,
        ),
        "schema_version": EVAL_TERMINAL_RECOVERY_SCHEMA_VERSION,
        "recovery_reason": EVAL_TERMINAL_RECOVERY_REASON,
        "recovery_sequence": 4,
        "authority_pin": EVAL_TERMINAL_AUTHORITY_PIN,
        "prior_recovery_path": str(prior_path),
        "prior_recovery_file_sha256": sha256_file(prior_path),
        "prior_recovery_sha256": prior["recovery_sha256"],
        "prior_recovery_head": prior["recovery_head"],
        "prior_recovery_result": "terminal_review_found_structural_correctness_defects",
        "pre_repair_terminal_head": EVAL_PRE_REPAIR_TERMINAL_HEAD,
        "prior_final_artifacts": prior_artifacts,
        "prior_model_aggregates": model_aggregates,
        "manifest_open_count": 1,
        "additional_manifest_derivations": 0,
        "model_inference_authorized": False,
        "dev_selection_mutation_authorized": False,
        "derived_regeneration_use_limit": 1,
        "derived_output_names": sorted(EVAL_TERMINAL_MUTABLE_OUTPUT_NAMES),
        "repair_changed_files": _repair_changed_files(recovery_head),
        "contract_overrides": overrides,
        "recovery_head": recovery_head,
        "consumption_receipt_path": str(claim_path),
        "consumption_receipt_sha256": sha256_file(consumption_receipt_path(authorization_path)),
    }
    receipt["recovery_sha256"] = canonical_sha256(receipt)
    _write_exclusive(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--access-receipt", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--eval-authorization", required=True)
    parser.add_argument("--authority-pin", required=True)
    parser.add_argument("--pre-repair-head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)
    print({"output": str(Path(args.output).resolve())})


if __name__ == "__main__":
    main()
