from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.psem_sortformer_adaptation_depth.preflight import build_preflight, resolve_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--checkpoint", type=Path)
    preflight.add_argument("--corpus-root", type=Path)
    preflight.add_argument("--reference-root", type=Path)
    preflight.add_argument("--output-root", type=Path)
    preflight.add_argument("--static-only", action="store_true")
    args = parser.parse_args(argv)
    paths = resolve_paths(
        checkpoint=args.checkpoint,
        corpus_root=args.corpus_root,
        reference_root=args.reference_root,
        output_root=args.output_root,
    )
    receipt = build_preflight(paths, static_only=args.static_only)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    passed = (
        receipt["static_contract_valid"] if args.static_only else receipt["ready_for_runtime_audit"]
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
