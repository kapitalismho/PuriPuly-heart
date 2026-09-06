from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Issue #121 common arm dispatcher")
    parser.add_argument("--arm", required=True, choices=("R-H-SC", "R-T2-SC", "R-TA-SC"))
    parser.add_argument("--seed", required=True, type=int, choices=(7301, 7302))
    parser.add_argument("--root", default="/workspace/issue-121/arms")
    parser.add_argument("--store", default="/workspace/issue-121/authorizations")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-hash", default=None)
    parser.add_argument("--checkpoint-hash", default=None)
    parser.add_argument("--partition-hash", default=None)
    parser.add_argument("--weights-hash", default=None)
    parser.add_argument("--code-hash", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    arm_runtime.enforce_thread_caps()
    args = build_parser().parse_args(argv)
    if args.config is not None:
        config = arm_runtime.load_config(Path(args.config))
        if config.arm != args.arm or config.seed != args.seed:
            print("config arm/seed differs from CLI selection", file=sys.stderr)
            return 2
    else:
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
            return 2
        config = arm_runtime.config_from_dict(
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
    try:
        receipt = arm_runtime.check_authorization(config, Path(args.store))
    except arm_runtime.AuthorizationError as exc:
        print(f"authorization blocked: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"authorized": True, **receipt, "run_dir": str(config.run_dir())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
