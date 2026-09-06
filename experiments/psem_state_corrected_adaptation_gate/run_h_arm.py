from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


GPU_REQUIRED = (
    "config",
    "store",
    "bundle_dir",
    "checkpoint",
    "nemo_checkout",
    "dependency_lock",
    "corpus_root",
    "reference_root",
    "sampling_manifest",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Issue #121 R-H-SC arm runner")
    parser.add_argument("--command", required=True, choices=("profile", "run", "postprocess"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--store", default=None)
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--nemo-checkout", default=None)
    parser.add_argument("--dependency-lock", default=None)
    parser.add_argument("--corpus-root", default=None)
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--sampling-manifest", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=None, type=int)
    parser.add_argument("--hourly-cost-usd", default=0.0, type=float)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser


def _missing(args: argparse.Namespace, names: tuple[str, ...]) -> list[str]:
    flagged: list[str] = []
    for name in names:
        if getattr(args, name) in (None, ""):
            flagged.append("--" + name.replace("_", "-"))
    return flagged


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "postprocess":
        missing = _missing(args, ("export_dir", "out_dir"))
        if missing:
            print(f"postprocess requires {', '.join(missing)}", file=sys.stderr)
            return 2
        from experiments.psem_state_corrected_adaptation_gate import h_arm

        try:
            out = h_arm.run_postprocess_command(
                Path(args.export_dir), Path(args.out_dir), workers=args.workers
            )
        except Exception as exc:
            print(f"h_arm postprocess blocked: {exc}", file=sys.stderr)
            return 3
        print(json.dumps({"command": "postprocess", "result": out}))
        return 0

    missing = _missing(args, GPU_REQUIRED)
    if missing:
        print(f"{args.command} requires {', '.join(missing)}", file=sys.stderr)
        return 2

    from experiments.psem_state_corrected_adaptation_gate import arm_runtime, h_arm

    arm_runtime.enforce_thread_caps()

    config = arm_runtime.load_config(Path(args.config))
    inputs = h_arm.HArmPodInputs(
        bundle_dir=Path(args.bundle_dir),
        checkpoint=Path(args.checkpoint),
        nemo_checkout=Path(args.nemo_checkout),
        dependency_lock=Path(args.dependency_lock),
        corpus_root=Path(args.corpus_root),
        reference_root=Path(args.reference_root),
        sampling_manifest=Path(args.sampling_manifest),
        device=str(args.device),
        workers=args.workers,
        hourly_cost_usd=float(args.hourly_cost_usd),
    )
    if args.command == "profile":
        try:
            out = h_arm.run_profile_pod(config, Path(args.store), inputs)
        except (arm_runtime.AuthorizationError, h_arm.HArmError) as exc:
            print(f"h_arm profile blocked: {exc}", file=sys.stderr)
            return 3
        print(json.dumps({"command": "profile", "profile": out["profile"]}))
        return 0
    try:
        out = h_arm.run_h_arm_pod(config, Path(args.store), inputs)
    except (arm_runtime.AuthorizationError, h_arm.HArmError) as exc:
        print(f"h_arm blocked: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"command": "run", "final_manifest": out["final_manifest"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
