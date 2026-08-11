from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.speaker_representation_scd import execution_guard as _execution_guard
from experiments.speaker_representation_scd import r4_gate as _r4_gate_module
from experiments.speaker_representation_scd.execution_guard import (
    ExecutionGuardError,
    ExecutionLease,
    action_receipt_is_authoritative,
    load_completed_action_receipt,
    quarantine_orphan_action_receipt,
    run_supervised,
)
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT, REPOSITORY_ROOT
from experiments.speaker_representation_scd.r4_gate import validated_r4_cache_root

CONTINUOUS_RESULT = Path("manifests/r4/legacy_common_gt/continuous_{model_id}.json")
SENSITIVITY_RESULT = Path("manifests/r4/legacy_common_gt/sensitivity_{model_id}.json")
SELECTION_LEDGER = Path("manifests/r4/legacy_common_gt/candidate_selection_ledger.json")
MODEL_IDS = (
    "mhubert-147",
    "wavlm-base-plus",
    "unispeech-sat-base-plus",
    "eres2netv2-standard-prepool",
)

_ORIGINAL_LEGACY_SCAN = _execution_guard.strict_legacy_scan


def _r4_legacy_scan():
    matches, failures = _ORIGINAL_LEGACY_SCAN()
    tolerated = {
        row["pid"]
        for row in matches
        if row.get("name") not in ("python.exe", "pythonw.exe")
        or (row.get("module") and "phase5_" in str(row["module"]))
    }
    return (
        tuple(row for row in matches if row["pid"] not in tolerated),
        failures,
    )


def _research_python() -> Path:
    path = EXPERIMENT_ROOT / "environment" / ".venv" / "Scripts" / "python.exe"
    if not path.is_file():
        raise ExecutionGuardError("the locked R4 research environment is not synchronized")
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
    execution_action: str,
    receipt: Path,
    argv: list[str],
) -> Path:
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
        command = [str(_research_python()), *argv]
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
    _execution_guard.strict_legacy_scan = _r4_legacy_scan
    _r4_gate_module.strict_legacy_scan = _r4_legacy_scan
    cache_root = validated_r4_cache_root(f"r4_{action}")
    if action == "continuous":
        if encoder not in MODEL_IDS:
            raise ExecutionGuardError(f"unknown R4 encoder: {encoder}")
        receipt = cache_root / CONTINUOUS_RESULT.as_posix().format(model_id=encoder)
        return _run_action(
            cache_root,
            f"r4-continuous:{encoder}",
            receipt,
            [
                "-m",
                "experiments.speaker_representation_scd.r4_continuous",
                "--worker",
                "continuous",
                "--encoder",
                encoder,
            ],
        )
    if action == "sensitivity":
        if encoder not in MODEL_IDS:
            raise ExecutionGuardError(f"unknown R4 encoder: {encoder}")
        receipt = cache_root / SENSITIVITY_RESULT.as_posix().format(model_id=encoder)
        return _run_action(
            cache_root,
            f"r4-sensitivity:{encoder}",
            receipt,
            [
                "-m",
                "experiments.speaker_representation_scd.r4_continuous",
                "--worker",
                "sensitivity",
                "--encoder",
                encoder,
            ],
        )
    if action == "report":
        receipt = cache_root / SELECTION_LEDGER
        return _run_action(
            cache_root,
            "r4-report",
            receipt,
            [
                "-m",
                "experiments.speaker_representation_scd.r4_continuous",
                "--worker",
                "report",
            ],
        )
    raise ExecutionGuardError(f"unknown R4 action: {action}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("continuous", "sensitivity", "report"))
    parser.add_argument("--encoder", choices=MODEL_IDS)
    args = parser.parse_args(argv)
    requested = tuple([sys.executable, "-m", __package__ + ".r4_execute", *(argv or sys.argv[1:])])
    print(execute(args.action, requested, args.encoder))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutionGuardError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
