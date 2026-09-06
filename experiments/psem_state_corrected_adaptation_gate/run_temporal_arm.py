from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Issue #121 temporal arm runner")
    parser.add_argument("--arm", required=True, choices=("R-T2-SC", "R-TA-SC"))
    parser.add_argument("--seed", required=True, type=int, choices=(7301, 7302))
    parser.add_argument("--root", default="/workspace/issue-121/arms")
    parser.add_argument("--store", default="/workspace/issue-121/authorizations")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-hash", default=None)
    parser.add_argument("--checkpoint-hash", default=None)
    parser.add_argument("--partition-hash", default=None)
    parser.add_argument("--weights-hash", default=None)
    parser.add_argument("--code-hash", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--nemo-checkout", default=None)
    parser.add_argument("--dependency-lock", default=None)
    parser.add_argument("--corpus-root", default=None)
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--sampling-manifest", default=None)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=None, type=int)
    parser.add_argument("--baseline-frontier", default=None)
    parser.add_argument("--gpu-price-usd", default=None, type=float)
    parser.add_argument("--profile-only", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace):
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime
    from experiments.psem_state_corrected_adaptation_gate import temporal_train

    temporal_train.require_temporal_arm(args.arm)
    if args.config is not None:
        config = arm_runtime.load_config(Path(args.config))
        if config.arm != args.arm or config.seed != args.seed:
            print("config arm/seed differs from CLI selection", file=sys.stderr)
            raise SystemExit(2)
        return config
    missing = [
        name
        for name in (
            "input_hash",
            "checkpoint_hash",
            "partition_hash",
            "weights_hash",
            "code_hash",
        )
        if getattr(args, name) is None
    ]
    if missing:
        print(f"missing binding hashes: {','.join(missing)}", file=sys.stderr)
        raise SystemExit(2)
    return arm_runtime.config_from_dict(
        {
            "arm": args.arm,
            "seed": args.seed,
            "root": args.root,
            "input_hash": args.input_hash,
            "checkpoint_hash": args.checkpoint_hash,
            "partition_hash": args.partition_hash,
            "weights_hash": args.weights_hash,
            "code_hash": args.code_hash,
        }
    )


def _require_backend_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "checkpoint",
            "bundle",
            "nemo_checkout",
            "dependency_lock",
            "corpus_root",
            "reference_root",
            "sampling_manifest",
        )
        if getattr(args, name) is None
    ]
    if missing:
        print(f"missing backend paths: {','.join(missing)}", file=sys.stderr)
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    try:
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime
        from experiments.psem_state_corrected_adaptation_gate import temporal_train
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime
        from experiments.psem_state_corrected_adaptation_gate import temporal_train

    args = build_parser().parse_args(argv)
    arm_runtime.enforce_thread_caps()
    try:
        config = _config_from_args(args)
    except arm_runtime.AuthorizationError as exc:
        print(f"authorization blocked: {exc}", file=sys.stderr)
        return 3
    _require_backend_args(args)
    try:
        if args.profile_only:
            outcome = temporal_train.run_profile_command(config, Path(args.store), args)
            print(json.dumps(outcome["profile"], sort_keys=True))
            return 0
        if args.baseline_frontier is None:
            print("full mode requires --baseline-frontier", file=sys.stderr)
            return 2
        outcome = temporal_train.run_full_command(config, Path(args.store), args)
        print(json.dumps({"final_manifest": outcome["final_manifest"]}, sort_keys=True))
        return 0
    except arm_runtime.AuthorizationError as exc:
        print(f"authorization blocked: {exc}", file=sys.stderr)
        return 3
    except temporal_train.TemporalArmError as exc:
        print(f"temporal arm blocked: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
