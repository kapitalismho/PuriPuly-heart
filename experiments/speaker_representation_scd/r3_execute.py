from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.speaker_representation_scd import execution_guard as _execution_guard
from experiments.speaker_representation_scd.execution_guard import (
    ExecutionGuardError,
    ExecutionLease,
    action_receipt_is_authoritative,
    load_completed_action_receipt,
    quarantine_orphan_action_receipt,
    run_supervised,
)
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT, REPOSITORY_ROOT
from experiments.speaker_representation_scd.r3_gate import validated_r3_cache_root

ANCHOR_RECEIPT = Path("manifests/r3/legacy_common_gt/anchor_index_receipt.json")
PROBE_RESULT = Path("manifests/r3/legacy_common_gt/probe_{model_id}.json")
PROMOTION_LEDGER = Path("manifests/r3/legacy_common_gt/promotion_ledger.json")
MODEL_IDS = (
    "mhubert-147",
    "wavlm-base-plus",
    "unispeech-sat-base-plus",
    "eres2netv2-standard-prepool",
)


def _r3_legacy_scan():
    matches, failures = _execution_guard.strict_legacy_scan()
    tolerated = {
        row["pid"]
        for row in matches
        if row.get("module") and "phase5_" in str(row["module"])
    }
    return (
        tuple(row for row in matches if row["pid"] not in tolerated),
        failures,
    )


def _worker_python() -> Path:
    path = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"
    if not path.is_file():
        raise ExecutionGuardError("the repository virtual environment is not synchronized")
    return path.resolve()


def _research_python() -> Path:
    path = EXPERIMENT_ROOT / "environment" / ".venv" / "Scripts" / "python.exe"
    if not path.is_file():
        raise ExecutionGuardError("the locked R3 research environment is not synchronized")
    return path.resolve()


def _thread_environment(environment: dict[str, str]) -> dict[str, str]:
    environment["OMP_NUM_THREADS"] = "8"
    environment["MKL_NUM_THREADS"] = "8"
    environment["OPENBLAS_NUM_THREADS"] = "8"
    environment["NUMEXPR_NUM_THREADS"] = "8"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _run_action(
    cache_root: Path,
    gate_action: str,
    execution_action: str,
    receipt: Path,
    python: Path,
    argv: list[str],
) -> Path:
    _execution_guard.strict_legacy_scan = _r3_legacy_scan
    with ExecutionLease(
        cache_root,
        execution_action,
        tuple(argv),
        expected_receipt=receipt,
    ) as lease:
        if receipt.exists():
            if action_receipt_is_authoritative(cache_root, receipt, execution_action):
                raise ExecutionGuardError(
                    f"refusing to rerun an action with completed evidence: {receipt}"
                )
            quarantine_orphan_action_receipt(cache_root, receipt)
        command = [str(python), *argv]
        environment = _thread_environment(lease.worker_environment())
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


def execute(action: str, requested_argv: tuple[str, ...], encoder: str | None) -> Path:
    cache_root = validated_r3_cache_root(f"r3_{action}")
    if action == "prepare":
        receipt = cache_root / ANCHOR_RECEIPT
        return _run_action(
            cache_root,
            "r3_prepare",
            "r3-prepare",
            receipt,
            _worker_python(),
            [
                "-m",
                "experiments.speaker_representation_scd.r3_prepare",
                "--worker",
                "prepare",
            ],
        )
    if action == "probe":
        if encoder not in MODEL_IDS:
            raise ExecutionGuardError(f"unknown R3 encoder: {encoder}")
        receipt = cache_root / PROBE_RESULT.as_posix().format(model_id=encoder)
        return _run_action(
            cache_root,
            "r3_probe",
            f"r3-probe:{encoder}",
            receipt,
            _research_python(),
            [
                "-m",
                "experiments.speaker_representation_scd.r3_probe",
                "--worker",
                "probe",
                "--encoder",
                encoder,
            ],
        )
    if action == "promote":
        receipt = cache_root / PROMOTION_LEDGER
        return _run_action(
            cache_root,
            "r3_promote",
            "r3-promote",
            receipt,
            _research_python(),
            [
                "-m",
                "experiments.speaker_representation_scd.r3_probe",
                "--worker",
                "promote",
            ],
        )
    raise ExecutionGuardError(f"unknown R3 action: {action}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "probe", "promote"))
    parser.add_argument("--encoder", choices=MODEL_IDS)
    args = parser.parse_args(argv)
    requested = tuple([sys.executable, "-m", __package__ + ".r3_execute", *(argv or sys.argv[1:])])
    print(execute(args.action, requested, args.encoder))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutionGuardError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
