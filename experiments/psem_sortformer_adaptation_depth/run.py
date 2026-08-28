from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
    load_pinned_sortformer,
    write_dependency_lock,
)
from experiments.psem_sortformer_adaptation_depth.preflight import build_preflight, resolve_paths
from experiments.psem_sortformer_adaptation_depth.receipts import (
    SOURCE_MANIFEST,
    build_data_split_receipt,
    evaluator_reconstruction_contract,
    validate_material_training_gate,
    validate_overfit_canary,
    validate_trainable_checkpoint_lineage,
)
from experiments.psem_sortformer_adaptation_depth.sampling import (
    load_sampling_rows,
    load_training_sessions,
    materialize_sampling_manifest,
    validate_sampling_manifest,
)
from experiments.psem_sortformer_adaptation_depth.training import (
    build_manifest_class_weight_receipt,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("receipt must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--checkpoint", type=Path)
    preflight.add_argument("--corpus-root", type=Path)
    preflight.add_argument("--reference-root", type=Path)
    preflight.add_argument("--output-root", type=Path)
    preflight.add_argument("--static-only", action="store_true")
    commands.add_parser("data-split-receipt")
    commands.add_parser("evaluator-contract")
    dependency_lock = commands.add_parser("dependency-lock")
    dependency_lock.add_argument("--output", type=Path, required=True)
    sampling = commands.add_parser("sampling-manifest")
    sampling.add_argument("--corpus-root", type=Path, required=True)
    sampling.add_argument("--reference-root", type=Path, required=True)
    sampling.add_argument("--output", type=Path, required=True)
    lineage = commands.add_parser("validate-lineage")
    lineage.add_argument("receipt", type=Path)
    lineage.add_argument("--runtime-identity", type=Path, required=True)
    overfit = commands.add_parser("validate-overfit")
    overfit.add_argument("receipt", type=Path)
    overfit.add_argument("--manifest", type=Path, required=True)
    overfit.add_argument("--canaries", type=Path, required=True)
    material = commands.add_parser("validate-material-gate")
    material.add_argument("bundle", type=Path)
    material.add_argument("--manifest", type=Path, required=True)
    material.add_argument("--corpus-root", type=Path, required=True)
    material.add_argument("--reference-root", type=Path, required=True)
    weights = commands.add_parser("class-weights")
    weights.add_argument("--manifest", type=Path, required=True)
    weights.add_argument("--corpus-root", type=Path, required=True)
    weights.add_argument("--reference-root", type=Path, required=True)
    graph = commands.add_parser("model-graph")
    graph.add_argument("--checkpoint", type=Path, required=True)
    graph.add_argument("--nemo-checkout", type=Path, required=True)
    graph.add_argument("--dependency-lock", type=Path, required=True)
    graph.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "data-split-receipt":
        print(json.dumps(build_data_split_receipt(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "evaluator-contract":
        print(json.dumps(evaluator_reconstruction_contract(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "dependency-lock":
        print(json.dumps(write_dependency_lock(args.output), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "sampling-manifest":
        sessions = load_training_sessions(args.corpus_root, args.reference_root)
        receipt = materialize_sampling_manifest(sessions, args.output)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "validate-lineage":
        receipt = validate_trainable_checkpoint_lineage(
            _load_json(args.receipt),
            runtime_identity=_load_json(args.runtime_identity),
            evaluator_contract=evaluator_reconstruction_contract(),
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "validate-overfit":
        source_rows = [
            json.loads(line) for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines()
        ]
        receipt = validate_overfit_canary(
            _load_json(args.receipt),
            sampling_rows=load_sampling_rows(args.manifest),
            sampling_manifest_path=args.manifest,
            corpus_by_source={row["source_id"]: row["corpus"] for row in source_rows},
            canary_receipts=_load_json(args.canaries),
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "validate-material-gate":
        bundle = _load_json(args.bundle)
        receipt = validate_material_training_gate(
            arm=bundle["arm"],
            seed=bundle["seed"],
            preflight_receipt=bundle["preflight_receipt"],
            sampling_validation=bundle["sampling_validation"],
            sampling_manifest_path=args.manifest,
            sampling_rows=load_sampling_rows(args.manifest),
            training_sessions=load_training_sessions(args.corpus_root, args.reference_root),
            class_weight_receipt=bundle["class_weight_receipt"],
            lineage_receipt=bundle["lineage_receipt"],
            runtime_identity=bundle["runtime_identity"],
            evaluator_contract=evaluator_reconstruction_contract(),
            parameter_inventory=bundle["parameter_inventory"],
            gradient_receipt=bundle["gradient_receipt"],
            update_receipt=bundle["update_receipt"],
            timing_receipt=bundle["timing_receipt"],
            overfit_receipt=bundle["overfit_receipt"],
            overfit_canary_receipts=bundle["overfit_canary_receipts"],
            staged_execution_receipt=bundle["staged_execution_receipt"],
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "class-weights":
        sessions = load_training_sessions(args.corpus_root, args.reference_root)
        validate_sampling_manifest(args.manifest, sessions)
        receipt = build_manifest_class_weight_receipt(
            load_sampling_rows(args.manifest), sessions, args.manifest
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "model-graph":
        _, receipt = load_pinned_sortformer(
            args.checkpoint,
            args.nemo_checkout,
            args.dependency_lock,
            args.device,
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
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
