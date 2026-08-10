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
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT, REPOSITORY_ROOT
from experiments.speaker_representation_scd.r2_gate import validated_r2_cache_root


def _research_python() -> Path:
    path = EXPERIMENT_ROOT / "environment" / ".venv" / "Scripts" / "python.exe"
    if not path.is_file():
        raise ExecutionGuardError("the locked R1 research environment is not synchronized")
    return path.resolve()


def _receipt(cache_root: Path, action: str) -> Path:
    name = {
        "archives": "development_archive_receipt.json",
        "materialize": "development_materialization_receipt.json",
    }[action]
    return cache_root / "manifests" / "r2" / "development" / name


def execute(action: str, requested_argv: tuple[str, ...]) -> Path:
    gate_action = {
        "archives": "development_archive_download",
        "materialize": "development_waveform_materialization",
    }[action]
    cache_root = validated_r2_cache_root(gate_action)
    receipt = _receipt(cache_root, action)
    execution_action = f"r2-{action}"
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
            str(_research_python()),
            "-m",
            "experiments.speaker_representation_scd.r2_materialize",
            "--worker",
            action,
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
    parser.add_argument("action", choices=("archives", "materialize"))
    args = parser.parse_args(argv)
    requested = tuple([sys.executable, "-m", __package__ + ".r2_execute", *(argv or sys.argv[1:])])
    print(execute(args.action, requested))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutionGuardError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
