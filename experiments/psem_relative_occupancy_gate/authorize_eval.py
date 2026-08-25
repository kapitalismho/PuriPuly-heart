from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from experiments.psem_relative_occupancy_gate.eval_access import (
    EvalAccessError,
    access_receipt_path,
    consumption_receipt_path,
    load_frozen_selection,
    validate_frozen_selection_bindings,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    PACKAGE_ROOT,
    canonical_sha256,
    load_json,
    safe_output_path,
    sha256_file,
    write_json,
)


class EvalAuthorizationError(RuntimeError):
    pass


def _validate_dev_evidence(
    *,
    selection_path: Path,
    selection: dict[str, object],
    verification: dict[str, object],
) -> None:
    root = selection_path.parent
    try:
        validate_frozen_selection_bindings(selection_path, selection)
    except EvalAccessError as exc:
        raise EvalAuthorizationError(str(exc)) from exc
    contract_files = selection.get("artifact_bindings", {})
    if not isinstance(contract_files, dict):
        raise EvalAuthorizationError("DEV selection artifact bindings are invalid")
    contracts = contract_files.get("contract_files")
    if not isinstance(contracts, dict) or not contracts:
        raise EvalAuthorizationError("DEV selection contract bindings are missing")
    for name, expected_hash in contracts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise EvalAuthorizationError("DEV selection contract path is invalid")
        path = PACKAGE_ROOT / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise EvalAuthorizationError(f"DEV selection contract changed: {name}")
    expected_artifacts = {
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
    for key, path in expected_artifacts.items():
        if not path.is_file() or sha256_file(path) != contract_files.get(key):
            raise EvalAuthorizationError(f"accepted DEV artifact changed: {path.name}")
    model_receipts = contract_files.get("model_receipts")
    expected_receipts = {
        "sortformer": root / "sortformer_model_receipt.json",
        "lseend": root / "lseend_model_receipt.json",
    }
    if not isinstance(model_receipts, dict):
        raise EvalAuthorizationError("DEV model receipt bindings are missing")
    for family, path in expected_receipts.items():
        if not path.is_file() or sha256_file(path) != model_receipts.get(family):
            raise EvalAuthorizationError(f"accepted DEV model receipt changed: {family}")
    fixed = {
        "gate0": root / "gate0_oracle_metrics.json",
        "gate0_verification": root / "gate0_verification.json",
        "gate1": root / "gate1_metrics.json",
        "gate2": root / "gate2_metrics.json",
        "product": root / "product_frontiers.json",
        "topology": root / "topology_slices.json",
        "latency": root / "latency_breakdown.json",
        "selection": selection_path,
    }
    verification_hashes = verification.get("artifact_sha256")
    if not isinstance(verification_hashes, dict):
        raise EvalAuthorizationError("DEV verification artifact bindings are missing")
    for name, path in fixed.items():
        expected_hash = (
            selection.get("gate0_sha256")
            if name == "gate0"
            else (
                selection.get("gate0_verification_sha256")
                if name == "gate0_verification"
                else verification_hashes.get(name)
            )
        )
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise EvalAuthorizationError(f"DEV verification input changed: {path.name}")
    gate0 = load_json(fixed["gate0"])
    gate0_contracts = gate0.get("contract_artifacts") if isinstance(gate0, dict) else None
    if not isinstance(gate0_contracts, dict) or not gate0_contracts:
        raise EvalAuthorizationError("Gate 0 contract bindings are missing")
    for name, expected_hash in gate0_contracts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise EvalAuthorizationError("Gate 0 contract path is invalid")
        path = PACKAGE_ROOT / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise EvalAuthorizationError(f"Gate 0 contract changed: {name}")
    manifest_path = Path(str(selection.get("manifest_path", ""))).resolve()
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != selection.get("manifest_sha256")
        or sha256_file(CONFIG_PATH) != selection.get("config_sha256")
    ):
        raise EvalAuthorizationError("DEV manifest or configuration changed")


def run(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection).resolve()
    verification_path = Path(args.verification).resolve()
    manifest_output = Path(args.manifest_output).resolve()
    output = safe_output_path(Path(args.output))
    selection = load_frozen_selection(selection_path)
    if verification_path != selection_path.parent / "model_gate_verification.json":
        raise EvalAuthorizationError("model-gate verification path is not canonical")
    verification = load_json(verification_path)
    if (
        not isinstance(verification, dict)
        or verification.get("schema_version")
        != "psem.relative_occupancy.model_gate_verification.v1"
        or verification.get("role") != "PSEM-STRATEGY-DEV"
        or verification.get("eval_status") != "sealed"
        or verification.get("passed") is not True
        or verification.get("selection_sha256") != selection["selection_sha256"]
        or verification.get("artifact_sha256", {}).get("selection") != sha256_file(selection_path)
    ):
        raise EvalAuthorizationError("accepted DEV model-gate verification is missing")
    _validate_dev_evidence(
        selection_path=selection_path,
        selection=selection,
        verification=verification,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != args.accepted_c2_head:
        raise EvalAuthorizationError("current head differs from the accepted C2 head")
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise EvalAuthorizationError("worktree differs from the accepted C2 head")
    if (
        manifest_output.exists()
        or access_receipt_path(manifest_output).exists()
        or consumption_receipt_path(output).exists()
    ):
        raise EvalAuthorizationError("the one-use EVAL manifest target already exists")
    if output.exists():
        raise EvalAuthorizationError("EVAL authorization output already exists")
    payload = {
        "schema_version": "psem.relative_occupancy.eval_authorization.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "authorization_state": "authorized_for_one_manifest_derivation",
        "use_limit": 1,
        "accepted_c2_head": head,
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": sha256_file(selection_path),
        "model_gate_verification_sha256": sha256_file(verification_path),
        "model_gate_verification_path": str(verification_path),
        "manifest_output_path": str(manifest_output),
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    write_json(output, payload)
    print({"output": str(output), "accepted_c2_head": head})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--accepted-c2-head", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
