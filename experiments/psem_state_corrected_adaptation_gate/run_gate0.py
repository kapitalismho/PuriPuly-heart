from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.psem_state_corrected_adaptation_gate.material import (
    MaterialBlockedError,
    MaterialError,
    resolve_material_inputs,
    run_material_slice,
)
from experiments.psem_state_corrected_adaptation_gate.receipts import NEMO_SHA256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Issue #121 Gate-0 material slice")
    parser.add_argument(
        "--mode",
        choices=("material", "precompute", "cuda", "postprocess"),
        default="material",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--nemo-checkout")
    parser.add_argument("--dependency-lock")
    parser.add_argument("--corpus-root")
    parser.add_argument("--reference-root")
    parser.add_argument("--sampling-manifest")
    parser.add_argument("--bundle")
    parser.add_argument("--stage-b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU worker processes for target-build/frontier (default: min(24, CPUs); 1 forces sequential)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "precompute":
            from experiments.psem_state_corrected_adaptation_gate.stages import (
                run_stage_a,
            )

            for flag in ("corpus-root", "reference-root", "sampling-manifest"):
                if getattr(args, flag.replace("-", "_")) is None:
                    print(f"precompute requires --{flag}", file=sys.stderr)
                    return 2
            manifest = run_stage_a(
                Path(args.corpus_root),
                Path(args.reference_root),
                Path(args.sampling_manifest),
                Path(args.out),
                args.workers,
            )
            print(json.dumps({"mode": "precompute", "manifest": str(manifest)}))
            return 0
        if args.mode == "cuda":
            from experiments.psem_state_corrected_adaptation_gate.stages import (
                run_stage_b,
            )

            for flag in (
                "bundle",
                "checkpoint",
                "nemo-checkout",
                "dependency-lock",
                "corpus-root",
                "reference-root",
            ):
                if getattr(args, flag.replace("-", "_")) is None:
                    print(f"cuda requires --{flag}", file=sys.stderr)
                    return 2
            manifest = run_stage_b(
                Path(args.bundle),
                Path(args.checkpoint),
                Path(args.nemo_checkout),
                Path(args.dependency_lock),
                Path(args.corpus_root),
                Path(args.reference_root),
                args.device,
                Path(args.out),
                args.workers,
            )
            print(json.dumps({"mode": "cuda", "manifest": str(manifest)}))
            return 0
        if args.mode == "postprocess":
            from experiments.psem_state_corrected_adaptation_gate.stages import (
                run_stage_c,
            )

            for flag in ("bundle", "stage-b"):
                if getattr(args, flag.replace("-", "_")) is None:
                    print(f"postprocess requires --{flag}", file=sys.stderr)
                    return 2
            record = run_stage_c(
                Path(args.bundle), Path(args.stage_b), Path(args.out), args.workers
            )
            print(json.dumps({"mode": "postprocess", "record": str(record)}))
            return 0
        for flag in (
            "checkpoint",
            "nemo-checkout",
            "dependency-lock",
            "corpus-root",
            "reference-root",
            "sampling-manifest",
        ):
            if getattr(args, flag.replace("-", "_")) is None:
                print(f"material requires --{flag}", file=sys.stderr)
                return 2
        resolved = resolve_material_inputs(
            Path(args.checkpoint),
            Path(args.nemo_checkout),
            Path(args.dependency_lock),
            Path(args.corpus_root),
            Path(args.reference_root),
            Path(args.sampling_manifest),
            NEMO_SHA256,
            args.device,
        )
        record = run_material_slice(resolved, Path(args.out), workers=args.workers)
    except MaterialBlockedError as exc:
        print(f"material blocked: {exc}", file=sys.stderr)
        return 3
    except MaterialError as exc:
        print(f"material failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"mode": "material", "verdict": record["verdict"]}))
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
