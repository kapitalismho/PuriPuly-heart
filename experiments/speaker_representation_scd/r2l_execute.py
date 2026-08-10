from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.speaker_representation_scd.execution_guard import (
    ExecutionGuardError,
    ExecutionLease,
    action_receipt_is_authoritative,
    load_completed_action_receipt,
    quarantine_orphan_action_receipt,
    run_supervised,
)
from experiments.speaker_representation_scd.r1_gate import REPOSITORY_ROOT
from experiments.speaker_representation_scd.r2l_gate import validated_r2l_cache_root


def _worker_python() -> Path:
    path = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"
    if not path.is_file():
        raise ExecutionGuardError("the repository virtual environment is not synchronized")
    return path.resolve()


def _receipt(cache_root: Path) -> Path:
    return cache_root / "manifests" / "r2" / "legacy_common_gt" / "validation_receipt.json"


def execute(action: str, requested_argv: tuple[str, ...]) -> Path:
    if action != "materialize":
        raise ExecutionGuardError(f"unknown R2-L action: {action}")
    gate_action = "coordinate_materialization"
    cache_root = validated_r2l_cache_root(gate_action)
    receipt = _receipt(cache_root)
    execution_action = "r2l-materialize"
    with ExecutionLease(
        cache_root,
        execution_action,
        requested_argv,
        expected_receipt=receipt,
    ) as lease:
        if receipt.exists():
            if action_receipt_is_authoritative(cache_root, receipt, execution_action):
                raise ExecutionGuardError(
                    f"refusing to rerun an action with completed evidence: {receipt}"
                )
            quarantine_orphan_action_receipt(cache_root, receipt)
        command = [
            str(_worker_python()),
            "-m",
            "experiments.speaker_representation_scd.r2l_materialize",
            "--worker",
            "materialize",
        ]
        environment = lease.worker_environment()
        environment["OMP_NUM_THREADS"] = "8"
        environment["MKL_NUM_THREADS"] = "8"
        environment["OPENBLAS_NUM_THREADS"] = "8"
        environment["NUMEXPR_NUM_THREADS"] = "8"
        environment["PYTHONHASHSEED"] = "0"
        run_supervised(
            lease,
            command,
            cwd=REPOSITORY_ROOT,
            environment=environment,
        )
        if not receipt.is_file():
            raise ExecutionGuardError(f"supervised worker did not produce its receipt: {receipt}")
        lease.bind_action_receipt()
        lease.complete()
    load_completed_action_receipt(cache_root, receipt, execution_action)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("materialize",))
    args = parser.parse_args(argv)
    requested = tuple([sys.executable, "-m", __package__ + ".r2l_execute", *(argv or sys.argv[1:])])
    print(execute(args.action, requested))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutionGuardError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
