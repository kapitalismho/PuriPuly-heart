from __future__ import annotations

import hashlib
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


EVAL_RECOVERY_SCHEMA_VERSION = "psem.relative_occupancy.eval_recovery.v1"
EVAL_RECOVERY_REASON = "sortformer_exact_frozen_source_window_materialization"
EVAL_RECOVERY_EXTENSION_SCHEMA_VERSION = "psem.relative_occupancy.eval_recovery.v2"
EVAL_RECOVERY_EXTENSION_REASON = "sortformer_zero_flush_for_native_partial_final_chunk"
EVAL_RECOVERY_ACCEPTED_C3_HEAD = "07f78c391f980132f97d43f3b6d4280f7898d741"
EVAL_RECOVERY_ACCEPTED_C3_SHA256 = (
    "5965b4a839f3ed114253e9e14899460f380629512c9d337d6301a9cb6887f7f3"
)
EVAL_RECOVERY_ACCEPTED_C3_FILE_SHA256 = (
    "b7ce28976cafb50abf2eeb7d947eb93864f47e3944f510284bae3eb71b9c31e5"
)
EVAL_RECOVERY_FINALIZATION_SCHEMA_VERSION = "psem.relative_occupancy.eval_recovery.v3"
EVAL_RECOVERY_FINALIZATION_REASON = "lseend_unemitted_partial_terminal_audio"
EVAL_RECOVERY_ACCEPTED_C4_HEAD = "cc757522886048bc4ab1d40072ce4fd3c0521899"
EVAL_RECOVERY_ACCEPTED_C4_SHA256 = (
    "3d708dfc0dc9b2a84d8558686bd81eb6be8ffdb37a75ed2dc8b10fd53da9b36d"
)
EVAL_RECOVERY_ACCEPTED_C4_FILE_SHA256 = (
    "75a506fa64fc36419b239be13797f30de37427465bf98f9cc9587e614ed86524"
)
EVAL_TERMINAL_RECOVERY_SCHEMA_VERSION = "psem.relative_occupancy.eval_recovery.v4"
EVAL_TERMINAL_RECOVERY_REASON = "terminal_structural_correctness_recalculation"
EVAL_TERMINAL_RECOVERY_HEAD = "ba776b70f85c5bb3222c20adb4b2f7a51861a794"
EVAL_TERMINAL_RECOVERY_SHA256 = (
    "95e19f1ee22f210660ee289ec5192c4fd4ba00229792b0a9b434f79fe786bc57"
)
EVAL_TERMINAL_RECOVERY_FILE_SHA256 = (
    "a7b921148f809534f0bb4b3fc94859dad4f8554b67426db980434a0fdece3c7f"
)
EVAL_TERMINAL_RECOVERY_CONSUMPTION_SHA256 = (
    "e53490aa66c4de460f7608ec598487cbdf704e75be69367958c56d1a84355d59"
)
EVAL_TERMINAL_RECOVERY_CONSUMPTION_FILE_SHA256 = (
    "c3b24f3e81fd33931c6fdd7789701bb179b1b26d11f47d915a5bad14755bed5f"
)
EVAL_VERIFICATION_CORRECTION_SCHEMA_VERSION = (
    "psem.relative_occupancy.eval_verification_correction.v1"
)
EVAL_VERIFICATION_CORRECTION_REASON = "canonical_evaluation_grid_terminal_coverage"
EVAL_RECOVERY_ACCEPTED_C5_HEAD = "1edac159a8f7dfb4f7014d5cdec1292f0cfbbe83"
EVAL_RECOVERY_ACCEPTED_C5_SHA256 = (
    "9c98dc57892f24f5b6b1ba7d32536856bef013152ca2d68f161369cd2d61d29f"
)
EVAL_RECOVERY_ACCEPTED_C5_FILE_SHA256 = (
    "a606fe9e8151473aded4bd20c46d0a6ca7e135251ea8358fed3c6eb818670916"
)
EVAL_PRE_REPAIR_TERMINAL_HEAD = "a9b8229ede486b7e090d50b763c0aa1d2c47af8d"
EVAL_TERMINAL_AUTHORITY_PIN = (
    "08cd431c7cf92050c60e28d3a94f57bf1035dc909eaeb94a219412c2b604ddb8"
)
EVAL_RECOVERY_ALLOWED_PATHS = {
    "experiments/psem_relative_occupancy_gate/authorize_eval_recovery.py",
    "experiments/psem_relative_occupancy_gate/eval_access.py",
    "experiments/psem_relative_occupancy_gate/model_run_io.py",
    "experiments/psem_relative_occupancy_gate/run_sortformer_trace.py",
    "experiments/psem_relative_occupancy_gate/tests/test_provenance_sealing.py",
    "experiments/psem_relative_occupancy_gate/tests/test_trace_adapters.py",
}
EVAL_RECOVERY_REQUIRED_PATHS = {
    "experiments/psem_relative_occupancy_gate/authorize_eval_recovery.py",
    "experiments/psem_relative_occupancy_gate/eval_access.py",
    "experiments/psem_relative_occupancy_gate/model_run_io.py",
    "experiments/psem_relative_occupancy_gate/run_sortformer_trace.py",
}
EVAL_RECOVERY_FINALIZATION_ALLOWED_PATHS = EVAL_RECOVERY_ALLOWED_PATHS | {
    "experiments/psem_relative_occupancy_gate/run_eval.py",
    "experiments/psem_relative_occupancy_gate/run_eval_traces.py",
    "experiments/psem_relative_occupancy_gate/trace_runtime.py",
    "experiments/psem_relative_occupancy_gate/verify_eval.py",
}
EVAL_RECOVERY_FINALIZATION_REQUIRED_PATHS = EVAL_RECOVERY_REQUIRED_PATHS | {
    "experiments/psem_relative_occupancy_gate/run_eval.py",
    "experiments/psem_relative_occupancy_gate/run_eval_traces.py",
    "experiments/psem_relative_occupancy_gate/trace_runtime.py",
    "experiments/psem_relative_occupancy_gate/verify_eval.py",
}
EVAL_TERMINAL_RECOVERY_ALLOWED_PATHS = {
    "experiments/psem_relative_occupancy_gate/authorize_terminal_recovery.py",
    "experiments/psem_relative_occupancy_gate/eval_access.py",
    "experiments/psem_relative_occupancy_gate/model_evaluate.py",
    "experiments/psem_relative_occupancy_gate/run_eval.py",
    "experiments/psem_relative_occupancy_gate/verify_eval.py",
    "experiments/psem_relative_occupancy_gate/verify_model_gates.py",
    "experiments/psem_relative_occupancy_gate/tests/test_model_evaluate.py",
    "experiments/psem_relative_occupancy_gate/tests/test_model_gate_verifier.py",
    "experiments/psem_relative_occupancy_gate/tests/test_provenance_sealing.py",
}
EVAL_TERMINAL_RECOVERY_REQUIRED_PATHS = {
    "experiments/psem_relative_occupancy_gate/authorize_terminal_recovery.py",
    "experiments/psem_relative_occupancy_gate/eval_access.py",
    "experiments/psem_relative_occupancy_gate/model_evaluate.py",
    "experiments/psem_relative_occupancy_gate/run_eval.py",
    "experiments/psem_relative_occupancy_gate/verify_eval.py",
    "experiments/psem_relative_occupancy_gate/verify_model_gates.py",
    "experiments/psem_relative_occupancy_gate/tests/test_model_evaluate.py",
    "experiments/psem_relative_occupancy_gate/tests/test_model_gate_verifier.py",
}
EVAL_VERIFICATION_CORRECTION_ALLOWED_PATHS = {
    "experiments/psem_relative_occupancy_gate/authorize_verification_correction.py",
    "experiments/psem_relative_occupancy_gate/eval_access.py",
    "experiments/psem_relative_occupancy_gate/verify_model_gates.py",
    "experiments/psem_relative_occupancy_gate/tests/test_model_gate_verifier.py",
    "experiments/psem_relative_occupancy_gate/tests/test_provenance_sealing.py",
}
EVAL_VERIFICATION_CORRECTION_REQUIRED_PATHS = (
    EVAL_VERIFICATION_CORRECTION_ALLOWED_PATHS
)
EVAL_TERMINAL_MUTABLE_OUTPUT_NAMES = {
    "eval_metrics.json",
    "product_frontiers.json",
    "topology_slices.json",
    "latency_breakdown.json",
    "gate1_event_ledger.jsonl",
    "gate2_event_ledger.jsonl",
    "eval_verification.json",
    "FINAL_DECISION.md",
}


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


def recovery_receipt_path(authorization_path: Path) -> Path:
    return authorization_path.with_name(f"{authorization_path.stem}_recovery.json")


def recovery_extension_receipt_path(authorization_path: Path) -> Path:
    return authorization_path.with_name(f"{authorization_path.stem}_recovery_2.json")


def recovery_finalization_receipt_path(authorization_path: Path) -> Path:
    return authorization_path.with_name(f"{authorization_path.stem}_recovery_3.json")


def terminal_recovery_receipt_path(authorization_path: Path) -> Path:
    return authorization_path.with_name(f"{authorization_path.stem}_recovery_4.json")


def terminal_recovery_consumption_path(authorization_path: Path) -> Path:
    return authorization_path.with_name(
        f"{authorization_path.stem}_recovery_4_consumed.json"
    )


def verification_correction_receipt_path(authorization_path: Path) -> Path:
    return authorization_path.with_name("eval_verification_correction.json")


def validate_eval_result_directory(
    *,
    manifest_path: Path,
    authorization_path: Path,
    require_finalization: bool,
    allow_final_outputs: bool,
    require_terminal_recovery: bool = False,
    require_verification_correction: bool = False,
) -> None:
    manifest_path = manifest_path.resolve()
    authorization_path = authorization_path.resolve()
    root = manifest_path.parent
    if authorization_path.parent != root:
        raise EvalAccessError("EVAL authorization and manifest directories differ")
    required = {
        manifest_path.name,
        access_receipt_path(manifest_path).name,
        authorization_path.name,
        consumption_receipt_path(authorization_path).name,
        recovery_receipt_path(authorization_path).name,
        recovery_extension_receipt_path(authorization_path).name,
        "sortformer_model_receipt.json",
        "lseend_model_receipt.json",
    }
    if require_finalization:
        required.add(recovery_finalization_receipt_path(authorization_path).name)
    if require_terminal_recovery:
        required.add(terminal_recovery_receipt_path(authorization_path).name)
    if require_verification_correction:
        required.add(verification_correction_receipt_path(authorization_path).name)
    allowed = set(required)
    if allow_final_outputs:
        allowed.update(
            {
                "eval_metrics.json",
                "product_frontiers.json",
                "topology_slices.json",
                "latency_breakdown.json",
                "eval_verification.json",
                "FINAL_DECISION.md",
                "gate1_event_ledger.jsonl",
                "gate2_event_ledger.jsonl",
                terminal_recovery_receipt_path(authorization_path).name,
                terminal_recovery_consumption_path(authorization_path).name,
                verification_correction_receipt_path(authorization_path).name,
            }
        )
    observed: set[str] = set()
    for entry in root.iterdir():
        if entry.name not in allowed or _strict_file(entry, "EVAL result artifact") != entry.resolve():
            raise EvalAccessError(f"unexpected EVAL result artifact: {entry.name}")
        observed.add(entry.name)
    if not required <= observed:
        raise EvalAccessError("required EVAL result artifact is missing")


def _current_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_worktree_is_clean(allowed_untracked_root: Path) -> bool:
    repo_root = PACKAGE_ROOT.parent.parent.resolve()
    allowed_untracked_root = allowed_untracked_root.resolve()
    if repo_root not in allowed_untracked_root.parents:
        return False
    output = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        check=True,
        capture_output=True,
    ).stdout
    for record in output.split(b"\0"):
        if not record:
            continue
        if not record.startswith(b"?? "):
            return False
        relative_path = record[3:].decode("utf-8", errors="surrogateescape")
        path = (repo_root / relative_path).resolve()
        if path != allowed_untracked_root and allowed_untracked_root not in path.parents:
            return False
    return True


def _tracked_worktree_matches_terminal_recovery_outputs(eval_root: Path) -> bool:
    repo_root = PACKAGE_ROOT.parent.parent.resolve()
    eval_root = eval_root.resolve()
    if repo_root not in eval_root.parents:
        return False
    output = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        check=True,
        capture_output=True,
    ).stdout
    allowed_names = EVAL_TERMINAL_MUTABLE_OUTPUT_NAMES | {
        terminal_recovery_receipt_path(eval_root / "eval_authorization.json").name,
        terminal_recovery_consumption_path(eval_root / "eval_authorization.json").name,
        verification_correction_receipt_path(eval_root / "eval_authorization.json").name,
    }
    for record in output.split(b"\0"):
        if not record:
            continue
        if len(record) < 4 or record[:2] not in {
            b"??",
            b" M",
            b"M ",
            b"MM",
            b" A",
            b"A ",
        }:
            return False
        relative_path = record[3:].decode("utf-8", errors="surrogateescape")
        path = (repo_root / relative_path).resolve()
        if path.parent != eval_root or path.name not in allowed_names:
            return False
    return True


def _git_is_ancestor(base: str, head: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, head],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _git_is_direct_child(parent: str, child: str) -> bool:
    output = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", child],
        check=False,
        capture_output=True,
        text=True,
    )
    return output.returncode == 0 and output.stdout.split() == [child, parent]


def _git_changed_paths(base: str, head: str) -> set[str]:
    output = subprocess.run(
        ["git", "diff", "--name-only", f"{base}..{head}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()}


def _git_file_sha256(head: str, relative_path: str) -> str | None:
    completed = subprocess.run(
        ["git", "show", f"{head}:{relative_path}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


def validate_frozen_selection_bindings(
    selection_path: Path,
    selection: dict[str, Any],
    *,
    contract_overrides: dict[str, dict[str, str]] | None = None,
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
        observed_hash = sha256_file(path)
        override = (contract_overrides or {}).get(str(name))
        if (
            Path(str(name)).name != str(name)
            or _strict_file(path, f"DEV contract {name}") != path.resolve()
            or (
                observed_hash != expected_hash
                and (
                    not isinstance(override, dict)
                    or override.get("before_sha256") != expected_hash
                    or override.get("after_sha256") != observed_hash
                )
            )
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


def _recovery_expected(
    *,
    authorization_path: Path,
    authorization: dict[str, Any],
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, Any]:
    access_path = access_receipt_path(manifest_output)
    claim_path = consumption_receipt_path(authorization_path)
    return {
        "role": "PSEM-STRATEGY-EVAL",
        "recovery_state": "authorized_for_same_opened_manifest_resume",
        "accepted_c2_head": authorization["accepted_c2_head"],
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": sha256_file(authorization_path),
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": sha256_file(selection_path),
        "manifest_path": str(manifest_output),
        "manifest_sha256": sha256_file(manifest_output),
        "access_receipt_path": str(access_path),
        "access_receipt_sha256": sha256_file(access_path),
        "consumption_receipt_path": str(claim_path),
        "consumption_receipt_sha256": sha256_file(claim_path),
        "manifest_open_count": 1,
        "additional_manifest_derivations": 0,
        "prior_eval_aggregate_count": 0,
    }


def _load_recovery_value(path: Path, label: str) -> dict[str, Any]:
    path = _strict_file(path, label)
    value = load_json(path)
    if not isinstance(value, dict):
        raise EvalAccessError(f"{label} must be an object")
    payload = dict(value)
    observed_hash = payload.pop("recovery_sha256", None)
    if observed_hash != canonical_sha256(payload):
        raise EvalAccessError(f"{label} hash is invalid")
    return value


def _validate_recovery_file_set(
    value: dict[str, Any],
    *,
    authorization: dict[str, Any],
    selection: dict[str, Any],
    active: bool,
    allowed_paths: set[str] | None = None,
    required_paths: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    recovery_head = str(value.get("recovery_head", ""))
    accepted_c2_head = str(authorization["accepted_c2_head"])
    if not _git_is_ancestor(accepted_c2_head, recovery_head):
        raise EvalAccessError("EVAL recovery ancestry mismatch")
    changed_files = value.get("changed_files")
    if not isinstance(changed_files, dict) or not changed_files:
        raise EvalAccessError("EVAL recovery changed-file binding is missing")
    allowed = EVAL_RECOVERY_ALLOWED_PATHS if allowed_paths is None else allowed_paths
    required = EVAL_RECOVERY_REQUIRED_PATHS if required_paths is None else required_paths
    observed_paths = _git_changed_paths(accepted_c2_head, recovery_head)
    if (
        observed_paths != set(changed_files)
        or not required <= observed_paths
        or not observed_paths <= allowed
    ):
        raise EvalAccessError("EVAL recovery changed-file set mismatch")
    root = PACKAGE_ROOT.parent.parent
    for relative_path, binding in changed_files.items():
        if (
            not isinstance(relative_path, str)
            or not isinstance(binding, dict)
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise EvalAccessError("EVAL recovery changed-file path is invalid")
        current_path = (root / relative_path).resolve()
        before_sha256 = _git_file_sha256(accepted_c2_head, relative_path)
        after_sha256 = _git_file_sha256(recovery_head, relative_path)
        if root not in current_path.parents or after_sha256 is None:
            raise EvalAccessError("EVAL recovery changed-file location is invalid")
        if (
            binding.get("before_sha256") != before_sha256
            or binding.get("after_sha256") != after_sha256
            or (
                active
                and (
                    not current_path.is_file()
                    or sha256_file(current_path) != after_sha256
                )
            )
        ):
            raise EvalAccessError("EVAL recovery changed-file hash mismatch")
    overrides = value.get("contract_overrides")
    if not isinstance(overrides, dict):
        raise EvalAccessError("EVAL recovery contract overrides are invalid")
    contracts = selection.get("artifact_bindings", {}).get("contract_files", {})
    expected_overrides: dict[str, dict[str, str]] = {}
    for name, before_sha256 in contracts.items():
        relative_path = f"experiments/psem_relative_occupancy_gate/{name}"
        after_sha256 = _git_file_sha256(recovery_head, relative_path)
        if after_sha256 is None:
            raise EvalAccessError("EVAL recovery contract file is missing")
        if after_sha256 != before_sha256:
            expected_overrides[str(name)] = {
                "before_sha256": str(before_sha256),
                "after_sha256": after_sha256,
            }
    if overrides != expected_overrides:
        raise EvalAccessError("EVAL recovery contract override set mismatch")
    return overrides


def _load_eval_recovery_history(
    authorization_path: Path,
    *,
    authorization: dict[str, Any],
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, Any]:
    path = recovery_receipt_path(authorization_path)
    value = _load_recovery_value(path, "EVAL recovery receipt")
    expected = {
        **_recovery_expected(
            authorization_path=authorization_path,
            authorization=authorization,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_output,
        ),
        "schema_version": EVAL_RECOVERY_SCHEMA_VERSION,
        "recovery_reason": EVAL_RECOVERY_REASON,
    }
    if (
        sha256_file(path) != EVAL_RECOVERY_ACCEPTED_C3_FILE_SHA256
        or value.get("recovery_head") != EVAL_RECOVERY_ACCEPTED_C3_HEAD
        or value.get("recovery_sha256") != EVAL_RECOVERY_ACCEPTED_C3_SHA256
        or any(value.get(field) != expected_value for field, expected_value in expected.items())
    ):
        raise EvalAccessError("EVAL recovery history binding mismatch")
    _validate_recovery_file_set(
        value, authorization=authorization, selection=selection, active=False
    )
    return value


def _load_eval_recovery(
    authorization_path: Path,
    *,
    authorization: dict[str, Any],
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, dict[str, str]]:
    value = _load_eval_recovery_history(
        authorization_path,
        authorization=authorization,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_output,
    )
    if (
        value.get("recovery_head") != _current_head()
        or not _tracked_worktree_is_clean(manifest_output.parent)
    ):
        raise EvalAccessError("EVAL recovery binding mismatch")
    return _validate_recovery_file_set(
        value, authorization=authorization, selection=selection, active=True
    )


def _load_eval_recovery_extension_history(
    authorization_path: Path,
    *,
    authorization: dict[str, Any],
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, dict[str, str]]:
    prior_path = recovery_receipt_path(authorization_path)
    prior = _load_eval_recovery_history(
        authorization_path,
        authorization=authorization,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_output,
    )
    value = _load_recovery_value(
        recovery_extension_receipt_path(authorization_path),
        "EVAL recovery extension receipt",
    )
    expected = {
        **_recovery_expected(
            authorization_path=authorization_path,
            authorization=authorization,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_output,
        ),
        "schema_version": EVAL_RECOVERY_EXTENSION_SCHEMA_VERSION,
        "recovery_reason": EVAL_RECOVERY_EXTENSION_REASON,
        "recovery_sequence": 2,
        "prior_recovery_path": str(prior_path),
        "prior_recovery_file_sha256": sha256_file(prior_path),
        "prior_recovery_sha256": prior["recovery_sha256"],
        "prior_recovery_head": prior["recovery_head"],
        "prior_recovery_result": "failed_before_eval_aggregate",
        "failed_family": "streaming_sortformer",
        "failed_source_id": "alimeeting_R1021_M1940",
        "completed_sortformer_source_count": 2,
        "lseend_started": False,
    }
    if (
        sha256_file(recovery_extension_receipt_path(authorization_path))
        != EVAL_RECOVERY_ACCEPTED_C4_FILE_SHA256
        or value.get("recovery_head") != EVAL_RECOVERY_ACCEPTED_C4_HEAD
        or value.get("recovery_sha256") != EVAL_RECOVERY_ACCEPTED_C4_SHA256
        or any(value.get(field) != expected_value for field, expected_value in expected.items())
        or not _git_is_direct_child(
            EVAL_RECOVERY_ACCEPTED_C3_HEAD, str(value.get("recovery_head", ""))
        )
    ):
        raise EvalAccessError("EVAL recovery extension binding mismatch")
    _validate_recovery_file_set(
        value, authorization=authorization, selection=selection, active=False
    )
    return value


def _load_eval_recovery_extension(
    authorization_path: Path,
    *,
    authorization: dict[str, Any],
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, dict[str, str]]:
    value = _load_eval_recovery_extension_history(
        authorization_path,
        authorization=authorization,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_output,
    )
    if (
        value.get("recovery_head") != _current_head()
        or not _tracked_worktree_is_clean(manifest_output.parent)
    ):
        raise EvalAccessError("EVAL recovery extension binding mismatch")
    return _validate_recovery_file_set(
        value, authorization=authorization, selection=selection, active=True
    )


def _load_eval_recovery_finalization_history(
    authorization_path: Path,
    *,
    authorization: dict[str, Any],
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, dict[str, str]]:
    prior_path = recovery_extension_receipt_path(authorization_path)
    prior = _load_eval_recovery_extension_history(
        authorization_path,
        authorization=authorization,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_output,
    )
    value = _load_recovery_value(
        recovery_finalization_receipt_path(authorization_path),
        "EVAL recovery finalization receipt",
    )
    expected = {
        **_recovery_expected(
            authorization_path=authorization_path,
            authorization=authorization,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_output,
        ),
        "schema_version": EVAL_RECOVERY_FINALIZATION_SCHEMA_VERSION,
        "recovery_reason": EVAL_RECOVERY_FINALIZATION_REASON,
        "recovery_sequence": 3,
        "prior_recovery_path": str(prior_path),
        "prior_recovery_file_sha256": sha256_file(prior_path),
        "prior_recovery_sha256": prior["recovery_sha256"],
        "prior_recovery_head": prior["recovery_head"],
        "prior_recovery_result": "failed_before_eval_metrics",
        "failed_stage": "eval_trace_validation",
        "failed_family": "ls_eend",
        "failed_source_id": "alimeeting_R1021_M1940",
        "completed_sortformer_source_count": 19,
        "completed_lseend_source_count": 19,
        "prior_eval_aggregate_count": 2,
        "prior_model_aggregates": {
            name: sha256_file(manifest_output.parent / name)
            for name in ("lseend_model_receipt.json", "sortformer_model_receipt.json")
        },
    }
    if (
        sha256_file(recovery_finalization_receipt_path(authorization_path))
        != EVAL_RECOVERY_ACCEPTED_C5_FILE_SHA256
        or value.get("recovery_head") != EVAL_RECOVERY_ACCEPTED_C5_HEAD
        or value.get("recovery_sha256") != EVAL_RECOVERY_ACCEPTED_C5_SHA256
        or any(value.get(field) != expected_value for field, expected_value in expected.items())
        or not _git_is_direct_child(
            EVAL_RECOVERY_ACCEPTED_C4_HEAD, str(value.get("recovery_head", ""))
        )
    ):
        raise EvalAccessError("EVAL recovery finalization binding mismatch")
    _validate_recovery_file_set(
        value,
        authorization=authorization,
        selection=selection,
        active=False,
        allowed_paths=EVAL_RECOVERY_FINALIZATION_ALLOWED_PATHS,
        required_paths=EVAL_RECOVERY_FINALIZATION_REQUIRED_PATHS,
    )
    return value


def _load_eval_recovery_finalization(
    authorization_path: Path,
    *,
    authorization: dict[str, Any],
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, dict[str, str]]:
    value = _load_eval_recovery_finalization_history(
        authorization_path,
        authorization=authorization,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_output,
    )
    if (
        value.get("recovery_head") != _current_head()
        or not _tracked_worktree_is_clean(manifest_output.parent)
    ):
        raise EvalAccessError("EVAL recovery finalization binding mismatch")
    validate_eval_result_directory(
        manifest_path=manifest_output,
        authorization_path=authorization_path,
        require_finalization=True,
        allow_final_outputs=True,
    )
    return _validate_recovery_file_set(
        value,
        authorization=authorization,
        selection=selection,
        active=True,
        allowed_paths=EVAL_RECOVERY_FINALIZATION_ALLOWED_PATHS,
        required_paths=EVAL_RECOVERY_FINALIZATION_REQUIRED_PATHS,
    )


def _terminal_prior_artifact_hashes(head: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "eval_metrics.json",
        "product_frontiers.json",
        "topology_slices.json",
        "latency_breakdown.json",
        "eval_verification.json",
        "FINAL_DECISION.md",
    ):
        relative_path = f"experiments/psem_relative_occupancy_gate/results/eval/{name}"
        value = _git_file_sha256(head, relative_path)
        if value is None:
            raise EvalAccessError("pre-repair terminal artifact is missing")
        result[name] = value
    return result


def _validate_terminal_repair_file_set(
    value: dict[str, Any], selection: dict[str, Any], *, active: bool
) -> dict[str, dict[str, str]]:
    recovery_head = str(value.get("recovery_head", ""))
    changed_files = value.get("repair_changed_files")
    observed_paths = _git_changed_paths(EVAL_PRE_REPAIR_TERMINAL_HEAD, recovery_head)
    if (
        not isinstance(changed_files, dict)
        or set(changed_files) != observed_paths
        or not EVAL_TERMINAL_RECOVERY_REQUIRED_PATHS <= observed_paths
        or not observed_paths <= EVAL_TERMINAL_RECOVERY_ALLOWED_PATHS
    ):
        raise EvalAccessError("terminal recovery changed-file set mismatch")
    repo_root = PACKAGE_ROOT.parent.parent
    for relative_path, binding in changed_files.items():
        before_sha256 = _git_file_sha256(EVAL_PRE_REPAIR_TERMINAL_HEAD, relative_path)
        after_sha256 = _git_file_sha256(recovery_head, relative_path)
        current_path = (repo_root / relative_path).resolve()
        if (
            not isinstance(binding, dict)
            or binding.get("before_sha256") != before_sha256
            or binding.get("after_sha256") != after_sha256
            or after_sha256 is None
            or (
                active
                and (
                    not current_path.is_file()
                    or sha256_file(current_path) != after_sha256
                )
            )
        ):
            raise EvalAccessError("terminal recovery changed-file hash mismatch")
    contracts = selection.get("artifact_bindings", {}).get("contract_files", {})
    expected_overrides: dict[str, dict[str, str]] = {}
    for name, before_sha256 in contracts.items():
        relative_path = f"experiments/psem_relative_occupancy_gate/{name}"
        after_sha256 = _git_file_sha256(recovery_head, relative_path)
        if after_sha256 is None:
            raise EvalAccessError("terminal recovery contract file is missing")
        if after_sha256 != before_sha256:
            expected_overrides[str(name)] = {
                "before_sha256": str(before_sha256),
                "after_sha256": after_sha256,
            }
    if value.get("contract_overrides") != expected_overrides:
        raise EvalAccessError("terminal recovery contract override set mismatch")
    return expected_overrides


def _load_eval_terminal_recovery_history(
    authorization_path: Path,
    *,
    authorization: dict[str, Any],
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, dict[str, str]]:
    prior_path = recovery_finalization_receipt_path(authorization_path)
    prior = _load_eval_recovery_finalization_history(
        authorization_path,
        authorization=authorization,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_output,
    )
    value = _load_recovery_value(
        terminal_recovery_receipt_path(authorization_path),
        "EVAL terminal recovery receipt",
    )
    prior_model_aggregates = {
        name: _git_file_sha256(
            EVAL_PRE_REPAIR_TERMINAL_HEAD,
            f"experiments/psem_relative_occupancy_gate/results/eval/{name}",
        )
        for name in ("lseend_model_receipt.json", "sortformer_model_receipt.json")
    }
    expected = {
        **_recovery_expected(
            authorization_path=authorization_path,
            authorization=authorization,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_output,
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
        "prior_final_artifacts": _terminal_prior_artifact_hashes(
            EVAL_PRE_REPAIR_TERMINAL_HEAD
        ),
        "prior_model_aggregates": prior_model_aggregates,
        "manifest_open_count": 1,
        "additional_manifest_derivations": 0,
        "model_inference_authorized": False,
        "dev_selection_mutation_authorized": False,
        "derived_regeneration_use_limit": 1,
        "derived_output_names": sorted(EVAL_TERMINAL_MUTABLE_OUTPUT_NAMES),
    }
    if (
        any(value.get(field) != expected_value for field, expected_value in expected.items())
        or sha256_file(terminal_recovery_receipt_path(authorization_path))
        != EVAL_TERMINAL_RECOVERY_FILE_SHA256
        or value.get("recovery_sha256") != EVAL_TERMINAL_RECOVERY_SHA256
        or value.get("recovery_head") != EVAL_TERMINAL_RECOVERY_HEAD
        or not _git_is_direct_child(
            EVAL_PRE_REPAIR_TERMINAL_HEAD, str(value.get("recovery_head", ""))
        )
    ):
        raise EvalAccessError("EVAL terminal recovery binding mismatch")
    return _validate_terminal_repair_file_set(value, selection, active=False)


def _load_eval_terminal_recovery(
    authorization_path: Path,
    *,
    authorization: dict[str, Any],
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, dict[str, str]]:
    _load_eval_terminal_recovery_history(
        authorization_path,
        authorization=authorization,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_output,
    )
    value = _load_recovery_value(
        terminal_recovery_receipt_path(authorization_path),
        "EVAL terminal recovery receipt",
    )
    if (
        value.get("recovery_head") != _current_head()
        or not _tracked_worktree_matches_terminal_recovery_outputs(
            manifest_output.parent
        )
    ):
        raise EvalAccessError("EVAL terminal recovery binding mismatch")
    validate_eval_result_directory(
        manifest_path=manifest_output,
        authorization_path=authorization_path,
        require_finalization=True,
        allow_final_outputs=True,
        require_terminal_recovery=True,
    )
    return _validate_terminal_repair_file_set(value, selection, active=True)


def _load_verification_correction_value(path: Path) -> dict[str, Any]:
    path = _strict_file(path, "EVAL verification correction receipt")
    value = load_json(path)
    if not isinstance(value, dict):
        raise EvalAccessError("EVAL verification correction receipt must be an object")
    payload = dict(value)
    observed_hash = payload.pop("correction_sha256", None)
    if observed_hash != canonical_sha256(payload):
        raise EvalAccessError("EVAL verification correction receipt hash is invalid")
    return value


def _verification_canonical_artifact_hashes(eval_root: Path) -> dict[str, str]:
    return {
        name: sha256_file(_strict_file(eval_root / name, f"canonical EVAL {name}"))
        for name in (
            "eval_metrics.json",
            "product_frontiers.json",
            "topology_slices.json",
            "latency_breakdown.json",
            "gate1_event_ledger.jsonl",
            "gate2_event_ledger.jsonl",
        )
    }


def _validate_verification_correction_file_set(
    value: dict[str, Any], selection: dict[str, Any], *, active: bool
) -> dict[str, dict[str, str]]:
    correction_head = str(value.get("correction_head", ""))
    changed_files = value.get("changed_files")
    observed_paths = _git_changed_paths(EVAL_TERMINAL_RECOVERY_HEAD, correction_head)
    if (
        not isinstance(changed_files, dict)
        or set(changed_files) != observed_paths
        or not EVAL_VERIFICATION_CORRECTION_REQUIRED_PATHS <= observed_paths
        or not observed_paths <= EVAL_VERIFICATION_CORRECTION_ALLOWED_PATHS
    ):
        raise EvalAccessError("verification correction changed-file set mismatch")
    repo_root = PACKAGE_ROOT.parent.parent
    for relative_path, binding in changed_files.items():
        before_sha256 = _git_file_sha256(EVAL_TERMINAL_RECOVERY_HEAD, relative_path)
        after_sha256 = _git_file_sha256(correction_head, relative_path)
        current_path = (repo_root / relative_path).resolve()
        if (
            not isinstance(binding, dict)
            or binding.get("before_sha256") != before_sha256
            or binding.get("after_sha256") != after_sha256
            or after_sha256 is None
            or (
                active
                and (
                    not current_path.is_file()
                    or sha256_file(current_path) != after_sha256
                )
            )
        ):
            raise EvalAccessError("verification correction changed-file hash mismatch")
    contracts = selection.get("artifact_bindings", {}).get("contract_files", {})
    expected_overrides: dict[str, dict[str, str]] = {}
    for name, before_sha256 in contracts.items():
        relative_path = f"experiments/psem_relative_occupancy_gate/{name}"
        after_sha256 = _git_file_sha256(correction_head, relative_path)
        if after_sha256 is None:
            raise EvalAccessError("verification correction contract file is missing")
        if after_sha256 != before_sha256:
            expected_overrides[str(name)] = {
                "before_sha256": str(before_sha256),
                "after_sha256": after_sha256,
            }
    if value.get("contract_overrides") != expected_overrides:
        raise EvalAccessError("verification correction contract override set mismatch")
    return expected_overrides


def _load_eval_verification_correction(
    authorization_path: Path,
    *,
    authorization: dict[str, Any],
    selection_path: Path,
    selection: dict[str, Any],
    manifest_output: Path,
) -> dict[str, dict[str, str]]:
    _load_eval_terminal_recovery_history(
        authorization_path,
        authorization=authorization,
        selection_path=selection_path,
        selection=selection,
        manifest_output=manifest_output,
    )
    consumption = _load_terminal_recovery_consumption(
        authorization_path=authorization_path,
        manifest_path=manifest_output,
        selection=selection,
    )
    recovery_path = terminal_recovery_receipt_path(authorization_path)
    consumption_path = terminal_recovery_consumption_path(authorization_path)
    value = _load_verification_correction_value(
        verification_correction_receipt_path(authorization_path)
    )
    expected = {
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
            manifest_output.parent
        ),
        "model_aggregates": {
            name: sha256_file(
                _strict_file(
                    manifest_output.parent / name,
                    f"EVAL model aggregate {name}",
                )
            )
            for name in ("lseend_model_receipt.json", "sortformer_model_receipt.json")
        },
        "pre_correction_verification_sha256": _git_file_sha256(
            EVAL_PRE_REPAIR_TERMINAL_HEAD,
            "experiments/psem_relative_occupancy_gate/results/eval/eval_verification.json",
        ),
        "canonical_regeneration_authorized": False,
        "model_inference_authorized": False,
        "dev_selection_mutation_authorized": False,
        "independent_temporary_regeneration_authorized": True,
        "verification_output_name": "eval_verification.json",
    }
    if (
        any(value.get(field) != expected_value for field, expected_value in expected.items())
        or value.get("correction_head") != _current_head()
        or not _git_is_direct_child(
            EVAL_TERMINAL_RECOVERY_HEAD, str(value.get("correction_head", ""))
        )
        or not _tracked_worktree_matches_terminal_recovery_outputs(
            manifest_output.parent
        )
    ):
        raise EvalAccessError("EVAL verification correction binding mismatch")
    validate_eval_result_directory(
        manifest_path=manifest_output,
        authorization_path=authorization_path,
        require_finalization=True,
        allow_final_outputs=True,
        require_terminal_recovery=True,
        require_verification_correction=True,
    )
    return _validate_verification_correction_file_set(value, selection, active=True)


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
        or verification_path
        != selection_path.resolve().parent / "model_gate_verification.json"
        or _strict_file(verification_path, "model-gate verification")
        != verification_path
        or sha256_file(verification_path) != value.get("model_gate_verification_sha256")
    ):
        raise EvalAccessError("EVAL authorization lacks the accepted C2 head")
    recovery_path = recovery_receipt_path(path)
    recovery_extension_path = recovery_extension_receipt_path(path)
    recovery_finalization_path = recovery_finalization_receipt_path(path)
    terminal_recovery_path = terminal_recovery_receipt_path(path)
    verification_correction_path = verification_correction_receipt_path(path)
    current_head = _current_head()
    if verification_correction_path.exists():
        overrides = _load_eval_verification_correction(
            path,
            authorization=value,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_output.resolve(),
        )
        validate_frozen_selection_bindings(
            selection_path, selection, contract_overrides=overrides
        )
    elif terminal_recovery_path.exists():
        overrides = _load_eval_terminal_recovery(
            path,
            authorization=value,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_output.resolve(),
        )
        validate_frozen_selection_bindings(
            selection_path, selection, contract_overrides=overrides
        )
    elif recovery_finalization_path.exists():
        overrides = _load_eval_recovery_finalization(
            path,
            authorization=value,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_output.resolve(),
        )
        validate_frozen_selection_bindings(
            selection_path, selection, contract_overrides=overrides
        )
    elif recovery_extension_path.exists():
        overrides = _load_eval_recovery_extension(
            path,
            authorization=value,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_output.resolve(),
        )
        validate_frozen_selection_bindings(
            selection_path, selection, contract_overrides=overrides
        )
    elif recovery_path.exists():
        overrides = _load_eval_recovery(
            path,
            authorization=value,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_output.resolve(),
        )
        validate_frozen_selection_bindings(
            selection_path, selection, contract_overrides=overrides
        )
    elif candidate_head == current_head:
        validate_frozen_selection_bindings(selection_path, selection)
    else:
        overrides = _load_eval_recovery(
            path,
            authorization=value,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_output.resolve(),
        )
        validate_frozen_selection_bindings(
            selection_path, selection, contract_overrides=overrides
        )
    return value


def _terminal_recovery_consumption_expected(
    *, authorization_path: Path, manifest_path: Path, selection: dict[str, Any]
) -> dict[str, Any]:
    recovery_path = terminal_recovery_receipt_path(authorization_path)
    recovery = _load_recovery_value(recovery_path, "EVAL terminal recovery receipt")
    return {
        "schema_version": "psem.relative_occupancy.eval_recovery_consumption.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "consumption_state": "claimed_for_one_derived_regeneration",
        "use_count": 1,
        "recovery_path": str(recovery_path),
        "recovery_file_sha256": sha256_file(recovery_path),
        "recovery_sha256": recovery["recovery_sha256"],
        "recovery_head": recovery["recovery_head"],
        "authority_pin": EVAL_TERMINAL_AUTHORITY_PIN,
        "manifest_sha256": sha256_file(manifest_path),
        "selection_sha256": selection["selection_sha256"],
        "prior_final_artifacts": recovery["prior_final_artifacts"],
        "derived_output_names": recovery["derived_output_names"],
    }


def _load_terminal_recovery_consumption(
    *, authorization_path: Path, manifest_path: Path, selection: dict[str, Any]
) -> dict[str, Any]:
    path = _strict_file(
        terminal_recovery_consumption_path(authorization_path),
        "EVAL terminal recovery consumption receipt",
    )
    value = load_json(path)
    if not isinstance(value, dict):
        raise EvalAccessError("EVAL terminal recovery consumption receipt must be an object")
    payload = dict(value)
    observed_hash = payload.pop("consumption_sha256", None)
    expected = _terminal_recovery_consumption_expected(
        authorization_path=authorization_path,
        manifest_path=manifest_path,
        selection=selection,
    )
    if (
        observed_hash != canonical_sha256(payload)
        or observed_hash != EVAL_TERMINAL_RECOVERY_CONSUMPTION_SHA256
        or sha256_file(path) != EVAL_TERMINAL_RECOVERY_CONSUMPTION_FILE_SHA256
        or any(value.get(field) != expected_value for field, expected_value in expected.items())
    ):
        raise EvalAccessError("EVAL terminal recovery consumption binding mismatch")
    return value


def claim_terminal_recovery_derivation(
    *, authorization_path: Path, manifest_path: Path, selection: dict[str, Any]
) -> dict[str, Any]:
    authorization_path = authorization_path.resolve()
    manifest_path = manifest_path.resolve()
    recovery = _load_recovery_value(
        terminal_recovery_receipt_path(authorization_path),
        "EVAL terminal recovery receipt",
    )
    root = manifest_path.parent
    if recovery.get("recovery_head") != _current_head():
        raise EvalAccessError("terminal recovery is not active at the current head")
    for name, expected_hash in recovery["prior_final_artifacts"].items():
        path = root / str(name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise EvalAccessError("pre-repair EVAL artifact changed before regeneration")
    if any((root / name).exists() for name in ("gate1_event_ledger.jsonl", "gate2_event_ledger.jsonl")):
        raise EvalAccessError("terminal recovery event ledgers already exist")
    payload = _terminal_recovery_consumption_expected(
        authorization_path=authorization_path,
        manifest_path=manifest_path,
        selection=selection,
    )
    payload["consumption_sha256"] = canonical_sha256(payload)
    path = terminal_recovery_consumption_path(authorization_path)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise EvalAccessError("terminal recovery regeneration was already claimed") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return payload


def validate_terminal_recovery_consumption(
    *, authorization_path: Path, manifest_path: Path, selection: dict[str, Any]
) -> dict[str, Any]:
    path = _strict_file(
        terminal_recovery_consumption_path(authorization_path.resolve()),
        "EVAL terminal recovery consumption receipt",
    )
    value = load_json(path)
    payload = dict(value) if isinstance(value, dict) else {}
    observed_hash = payload.pop("consumption_sha256", None)
    expected = _terminal_recovery_consumption_expected(
        authorization_path=authorization_path.resolve(),
        manifest_path=manifest_path.resolve(),
        selection=selection,
    )
    if observed_hash != canonical_sha256(payload) or payload != expected:
        raise EvalAccessError("terminal recovery consumption binding mismatch")
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
    if access_path != access_receipt_path(manifest_path):
        raise EvalAccessError("EVAL access receipt path is not canonical")
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
        or claim.get("schema_version") != "psem.relative_occupancy.eval_consumption.v1"
        or claim.get("role") != "PSEM-STRATEGY-EVAL"
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
