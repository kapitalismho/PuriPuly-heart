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


def _add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-root", type=Path)


def _runtime_path_error(paths: PreflightPaths, *, require_cache: bool) -> list[str]:
    errors = list(paths.errors)
    required = {
        "corpus_root": paths.corpus_root,
        "reference_root": paths.reference_root,
        "output_root": paths.output_root,
    }
    if require_cache:
        required["cache_root"] = paths.cache_root
    errors.extend(f"{name} is required" for name, value in required.items() if value is None)
    return errors


def _receipt_summary(receipts: dict[str, object]) -> dict[str, object]:
    rows = {
        name: {
            "status": value.get("status"),
            "payload_sha256": value.get("payload_sha256"),
        }
        for name, value in receipts.items()
        if isinstance(value, dict)
    }
    return {
        "status": "pass"
        if rows and all(row["status"] == "pass" for row in rows.values())
        else "fail",
        "receipts": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _add_path_arguments(preflight)
    preflight.add_argument("--skip-source-byte-hashes", action="store_true")
    prepare = commands.add_parser("prepare")
    _add_path_arguments(prepare)
    audit = commands.add_parser("audit")
    _add_path_arguments(audit)
    train = commands.add_parser("train")
    _add_path_arguments(train)
    train.add_argument(
        "--arm",
        required=True,
        choices=("FROZEN-WAVLM", "FINETUNE-WAVLM", "SCRATCH-PSEM"),
    )
    train.add_argument("--seed", required=True, type=int, choices=(7301, 7302))
    train_all = commands.add_parser("train-all")
    _add_path_arguments(train_all)
    status = commands.add_parser("training-status")
    _add_path_arguments(status)
    args = parser.parse_args(argv)
    paths = resolve_paths(
        cache_root=args.cache_root,
        corpus_root=args.corpus_root,
        reference_root=args.reference_root,
        output_root=args.output_root,
    )
    if args.command in {"prepare", "audit", "train", "train-all"}:
        errors = _runtime_path_error(
            paths,
            require_cache=args.command in {"audit", "train", "train-all"},
        )
        if errors:
            print(json.dumps({"status": "fail", "errors": errors}, sort_keys=True))
            return 2
    if args.command in {"prepare", "audit"}:
        if args.command == "prepare":
            from experiments.psem_training_strategy_gate.audit import prepare_runtime_manifests

            receipts = prepare_runtime_manifests(
                corpus_root=paths.corpus_root,
                reference_root=paths.reference_root,
                output_root=paths.output_root,
            )
        else:
            from experiments.psem_training_strategy_gate.audit import run_runtime_audits

            receipts = run_runtime_audits(
                cache_root=paths.cache_root,
                corpus_root=paths.corpus_root,
                reference_root=paths.reference_root,
                output_root=paths.output_root,
            )
        summary = _receipt_summary(receipts)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["status"] == "pass" else 2
    if args.command in {"train", "train-all"}:
        from experiments.psem_training_strategy_gate.training import (
            TrainingContractError,
            train_all_official_runs,
            train_official_run,
        )

        try:
            if args.command == "train":
                result = [train_official_run(paths, args.arm, args.seed)]
            else:
                result = train_all_official_runs(paths)
        except TrainingContractError as error:
            print(json.dumps({"status": "fail", "error": str(error)}, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": "pass",
                    "completed_runs": [value["run_id"] for value in result],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "training-status":
        from experiments.psem_training_strategy_gate.training import (
            TrainingContractError,
            training_status,
        )

        try:
            result = training_status(paths)
        except TrainingContractError as error:
            print(json.dumps({"status": "fail", "error": str(error)}, sort_keys=True))
            return 2
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
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
