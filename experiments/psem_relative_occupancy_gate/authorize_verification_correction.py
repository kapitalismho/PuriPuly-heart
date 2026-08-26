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
from experiments.psem_relative_occupancy_gate.authorize_terminal_recovery import (
    _contract_overrides,
)
from experiments.psem_relative_occupancy_gate.eval_access import (
    EVAL_PRE_REPAIR_TERMINAL_HEAD,
    EVAL_TERMINAL_AUTHORITY_PIN,
    EVAL_TERMINAL_RECOVERY_HEAD,
    EVAL_TERMINAL_RECOVERY_SHA256,
    EVAL_VERIFICATION_CORRECTION_ALLOWED_PATHS,
    EVAL_VERIFICATION_CORRECTION_REASON,
    EVAL_VERIFICATION_CORRECTION_REQUIRED_PATHS,
    EVAL_VERIFICATION_CORRECTION_SCHEMA_VERSION,
    _current_head,
    _git_changed_paths,
    _git_file_sha256,
    _git_is_direct_child,
    _load_eval_terminal_recovery_history,
    _load_terminal_recovery_consumption,
    _tracked_worktree_matches_terminal_recovery_outputs,
    _verification_canonical_artifact_hashes,
    access_receipt_path,
    load_frozen_selection,
    terminal_recovery_consumption_path,
    terminal_recovery_receipt_path,
    validate_eval_result_directory,
    validate_frozen_selection_bindings,
    verification_correction_receipt_path,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    canonical_sha256,
    safe_output_path,
    sha256_file,
)


class VerificationCorrectionAuthorizationError(RuntimeError):
    pass


def _changed_files(correction_head: str) -> dict[str, dict[str, str | None]]:
    paths = _git_changed_paths(EVAL_TERMINAL_RECOVERY_HEAD, correction_head)
    if (
        not EVAL_VERIFICATION_CORRECTION_REQUIRED_PATHS <= paths
        or not paths <= EVAL_VERIFICATION_CORRECTION_ALLOWED_PATHS
    ):
        raise VerificationCorrectionAuthorizationError(
            "verification correction path scope is invalid"
        )
    return {
        path: {
            "before_sha256": _git_file_sha256(EVAL_TERMINAL_RECOVERY_HEAD, path),
            "after_sha256": _git_file_sha256(correction_head, path),
        }
        for path in sorted(paths)
    }


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise VerificationCorrectionAuthorizationError(
            "verification correction authorization already exists"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    manifest_path = Path(args.manifest).resolve()
    access_path = Path(args.access_receipt).resolve()
    authorization_path = Path(args.eval_authorization).resolve()
    output = safe_output_path(Path(args.output))
    if output != verification_correction_receipt_path(authorization_path):
        raise VerificationCorrectionAuthorizationError(
            "verification correction path is not canonical"
        )
    if access_path != access_receipt_path(manifest_path):
        raise VerificationCorrectionAuthorizationError(
            "EVAL access path is not canonical"
        )
    if args.authority_pin != EVAL_TERMINAL_AUTHORITY_PIN:
        raise VerificationCorrectionAuthorizationError(
            "owner amendment authority pin mismatch"
        )
    if args.parent_recovery_head != EVAL_TERMINAL_RECOVERY_HEAD:
        raise VerificationCorrectionAuthorizationError(
            "terminal recovery head mismatch"
        )
    correction_head = _current_head()
    if (
        not _git_is_direct_child(EVAL_TERMINAL_RECOVERY_HEAD, correction_head)
        or not _tracked_worktree_matches_terminal_recovery_outputs(
            manifest_path.parent
        )
    ):
        raise VerificationCorrectionAuthorizationError(
            "verification correction requires the exact direct correction child"
        )
    selection = load_frozen_selection(selection_path)
    authorization = _load_canonical(
        authorization_path, "authorization_sha256", "EVAL authorization"
    )
    manifest = _validate_manifest(
        manifest_path, selection=selection, authorization=authorization
    )
    _validate_claim_and_access(
        manifest_path=manifest_path,
        access_path=access_path,
        authorization_path=authorization_path,
        authorization=authorization,
        selection=selection,
        source_count=len(manifest),
    )
    _load_eval_terminal_recovery_history(
        authorization_path,
        authorization=authorization,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_path,
    )
    consumption = _load_terminal_recovery_consumption(
        authorization_path=authorization_path,
        manifest_path=manifest_path,
        selection=selection,
    )
    validate_eval_result_directory(
        manifest_path=manifest_path,
        authorization_path=authorization_path,
        require_finalization=True,
        allow_final_outputs=True,
        require_terminal_recovery=True,
    )
    overrides = _contract_overrides(selection, correction_head)
    validate_frozen_selection_bindings(
        selection_path, selection, contract_overrides=overrides
    )
    prior_verification_sha256 = _git_file_sha256(
        EVAL_PRE_REPAIR_TERMINAL_HEAD,
        "experiments/psem_relative_occupancy_gate/results/eval/eval_verification.json",
    )
    verification_path = manifest_path.parent / "eval_verification.json"
    if (
        prior_verification_sha256 is None
        or sha256_file(verification_path) != prior_verification_sha256
    ):
        raise VerificationCorrectionAuthorizationError(
            "pre-correction verification artifact changed"
        )
    recovery_path = terminal_recovery_receipt_path(authorization_path)
    consumption_path = terminal_recovery_consumption_path(authorization_path)
    receipt = {
        "schema_version": EVAL_VERIFICATION_CORRECTION_SCHEMA_VERSION,
        "role": "PSEM-STRATEGY-EVAL",
        "correction_state": "authorized_for_verification_only",
        "correction_reason": EVAL_VERIFICATION_CORRECTION_REASON,
        "authority_pin": EVAL_TERMINAL_AUTHORITY_PIN,
        "prior_recovery_path": str(recovery_path),
        "prior_recovery_file_sha256": sha256_file(recovery_path),
        "prior_recovery_sha256": EVAL_TERMINAL_RECOVERY_SHA256,
        "prior_recovery_head": EVAL_TERMINAL_RECOVERY_HEAD,
        "recovery_consumption_path": str(consumption_path),
        "recovery_consumption_file_sha256": sha256_file(consumption_path),
        "recovery_consumption_sha256": consumption["consumption_sha256"],
        "canonical_artifacts": _verification_canonical_artifact_hashes(
            manifest_path.parent
        ),
        "model_aggregates": _validate_model_aggregates(
            manifest_path=manifest_path,
            manifest=manifest,
            access_path=access_path,
        ),
        "pre_correction_verification_sha256": prior_verification_sha256,
        "canonical_regeneration_authorized": False,
        "model_inference_authorized": False,
        "dev_selection_mutation_authorized": False,
        "independent_temporary_regeneration_authorized": True,
        "verification_output_name": "eval_verification.json",
        "changed_files": _changed_files(correction_head),
        "contract_overrides": overrides,
        "correction_head": correction_head,
    }
    receipt["correction_sha256"] = canonical_sha256(receipt)
    _write_exclusive(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--access-receipt", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--eval-authorization", required=True)
    parser.add_argument("--authority-pin", required=True)
    parser.add_argument("--parent-recovery-head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)
    print({"output": str(Path(args.output).resolve())})


if __name__ == "__main__":
    main()
