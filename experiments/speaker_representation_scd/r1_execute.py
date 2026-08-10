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
from experiments.speaker_representation_scd.r1_gate import (
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
    validated_cache_root,
)

MODEL_IDS = (
    "mhubert-147",
    "wavlm-base-plus",
    "unispeech-sat-base-plus",
    "eres2netv2-standard-prepool",
)


def _research_python() -> Path:
    path = EXPERIMENT_ROOT / "environment" / ".venv" / "Scripts" / "python.exe"
    if not path.is_file():
        raise ExecutionGuardError("the locked R1 research environment is not synchronized")
    return path.resolve()


def _expected_receipt(cache_root: Path, action: str, model_id: str | None) -> Path:
    if action == "sync-environment":
        return cache_root / "manifests" / "r1_environment_sync.json"
    if action == "models":
        return cache_root / "manifests" / "r1_model_acquisition.json"
    if model_id is None:
        raise ExecutionGuardError("smoke requires a model ID")
    return cache_root / "results" / "r1" / "smoke" / f"{model_id}.json"


def _worker_command(action: str, model_id: str | None) -> list[str]:
    if action == "sync-environment":
        return [
            sys.executable,
            "-m",
            "experiments.speaker_representation_scd.acquire_r1",
            "--worker",
            "sync-environment",
        ]
    python = str(_research_python())
    if action == "models":
        return [
            python,
            "-m",
            "experiments.speaker_representation_scd.acquire_r1",
            "--worker",
            "models",
        ]
    if model_id not in MODEL_IDS:
        raise ExecutionGuardError(f"unsupported smoke model: {model_id}")
    return [
        python,
        "-m",
        "experiments.speaker_representation_scd.r1_smoke",
        "--worker",
        "--model",
        str(model_id),
    ]


def execute(action: str, model_id: str | None, requested_argv: tuple[str, ...]) -> Path:
    gate_action = {
        "sync-environment": "environment_sync",
        "models": "model_artifact_download",
        "smoke": "neural_smoke",
    }[action]
    cache_root = validated_cache_root(gate_action)
    receipt = _expected_receipt(cache_root, action, model_id)
    with ExecutionLease(
        cache_root,
        action,
        requested_argv,
        expected_receipt=receipt,
    ) as lease:
        if receipt.exists():
            if action_receipt_is_authoritative(cache_root, receipt, action):
                raise ExecutionGuardError(
                    f"refusing to rerun an action with completed evidence: {receipt}"
                )
            quarantine_orphan_action_receipt(cache_root, receipt)
        command = _worker_command(action, model_id)
        environment = lease.worker_environment()
        environment["OMP_NUM_THREADS"] = "8"
        environment["MKL_NUM_THREADS"] = "8"
        environment["OPENBLAS_NUM_THREADS"] = "8"
        environment["NUMEXPR_NUM_THREADS"] = "8"
        environment["TOKENIZERS_PARALLELISM"] = "false"
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
    load_completed_action_receipt(cache_root, receipt, action)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("sync-environment")
    subparsers.add_parser("models")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--model", required=True, choices=MODEL_IDS)
    args = parser.parse_args(argv)
    requested = tuple([sys.executable, "-m", __package__ + ".r1_execute", *(argv or sys.argv[1:])])
    receipt = execute(args.action, getattr(args, "model", None), requested)
    print(receipt)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutionGuardError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
