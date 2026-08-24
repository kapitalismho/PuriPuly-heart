from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.psem_training_strategy_gate.preflight import (
    DEFAULT_OUTPUT_ROOT,
    PreflightPaths,
    build_preflight,
    resolve_paths,
    write_preflight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--cache-root", type=Path)
    preflight.add_argument("--corpus-root", type=Path)
    preflight.add_argument("--reference-root", type=Path)
    preflight.add_argument("--output-root", type=Path)
    preflight.add_argument("--skip-source-byte-hashes", action="store_true")
    args = parser.parse_args(argv)
    paths = resolve_paths(
        cache_root=args.cache_root,
        corpus_root=args.corpus_root,
        reference_root=args.reference_root,
        output_root=args.output_root,
    )
    receipt = build_preflight(
        paths,
        verify_source_bytes=not args.skip_source_byte_hashes,
    )
    receipt_path = (
        (paths.output_root or DEFAULT_OUTPUT_ROOT) / "preflight" / "experiment_receipt.json"
    )
    try:
        write_preflight(receipt_path, receipt)
    except OSError as exc:
        paths = PreflightPaths(
            paths.cache_root,
            paths.corpus_root,
            paths.reference_root,
            DEFAULT_OUTPUT_ROOT,
            (*paths.errors, f"output receipt persistence failed: {type(exc).__name__}: {exc}"),
        )
        receipt = build_preflight(
            paths,
            verify_source_bytes=not args.skip_source_byte_hashes,
        )
        receipt_path = DEFAULT_OUTPUT_ROOT / "preflight" / "experiment_receipt.json"
        write_preflight(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt["ready_for_material_run"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
