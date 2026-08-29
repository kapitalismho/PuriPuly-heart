from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.psem_sortformer_adaptation_depth.evaluation import evaluate_prediction_set
from experiments.psem_sortformer_adaptation_depth.execution import (
    candidate_code_identity,
    infer_prediction_set,
    run_canary_arm,
    run_overfit_arm_result,
    run_training_arm,
    validate_current_candidate_identity,
    write_json,
    write_jsonl,
)
from experiments.psem_sortformer_adaptation_depth.lineage import (
    build_lineage_receipt,
    lineage_authorization,
)
from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
    load_pinned_sortformer,
    write_dependency_lock,
)
from experiments.psem_sortformer_adaptation_depth.preflight import build_preflight, resolve_paths
from experiments.psem_sortformer_adaptation_depth.protocol import (
    append_dev_result,
    freeze_candidate_set,
    initial_staged_state,
    open_eval_once,
)
from experiments.psem_sortformer_adaptation_depth.receipts import (
    SOURCE_MANIFEST,
    build_data_split_receipt,
    evaluator_reconstruction_contract,
    validate_material_training_gate,
    validate_overfit_canary,
    validate_trainable_checkpoint_lineage,
)
from experiments.psem_sortformer_adaptation_depth.reporting import build_final_artifacts
from experiments.psem_sortformer_adaptation_depth.sampling import (
    load_sampling_rows,
    load_training_sessions,
    materialize_sampling_manifest,
    validate_sampling_manifest,
)
from experiments.psem_sortformer_adaptation_depth.training import (
    build_manifest_class_weight_receipt,
    build_overfit_receipt,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("receipt must be a JSON object")
    return value


def _load_path_list(value: object, field: str) -> list[Path]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of paths")
    return [Path(item) for item in value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--checkpoint", type=Path)
    preflight.add_argument("--corpus-root", type=Path)
    preflight.add_argument("--reference-root", type=Path)
    preflight.add_argument("--output-root", type=Path)
    preflight.add_argument("--protocol-registry-root", type=Path)
    preflight.add_argument("--static-only", action="store_true")
    preflight.add_argument("--receipt-output", type=Path)
    data_split = commands.add_parser("data-split-receipt")
    data_split.add_argument("--output", type=Path)
    evaluator = commands.add_parser("evaluator-contract")
    evaluator.add_argument("--output", type=Path)
    dependency_lock = commands.add_parser("dependency-lock")
    dependency_lock.add_argument("--output", type=Path, required=True)
    sampling = commands.add_parser("sampling-manifest")
    sampling.add_argument("--corpus-root", type=Path, required=True)
    sampling.add_argument("--reference-root", type=Path, required=True)
    sampling.add_argument("--output", type=Path, required=True)
    lineage = commands.add_parser("validate-lineage")
    lineage.add_argument("receipt", type=Path)
    lineage.add_argument("--runtime-identity", type=Path, required=True)
    lineage.add_argument("--output", type=Path)
    overfit = commands.add_parser("validate-overfit")
    overfit.add_argument("receipt", type=Path)
    overfit.add_argument("--manifest", type=Path, required=True)
    overfit.add_argument("--canaries", type=Path, required=True)
    overfit.add_argument("--output", type=Path)
    material = commands.add_parser("validate-material-gate")
    material.add_argument("bundle", type=Path)
    material.add_argument("--manifest", type=Path, required=True)
    material.add_argument("--corpus-root", type=Path, required=True)
    material.add_argument("--reference-root", type=Path, required=True)
    material.add_argument("--output", type=Path, required=True)
    weights = commands.add_parser("class-weights")
    weights.add_argument("--manifest", type=Path, required=True)
    weights.add_argument("--corpus-root", type=Path, required=True)
    weights.add_argument("--reference-root", type=Path, required=True)
    weights.add_argument("--output", type=Path)
    graph = commands.add_parser("model-graph")
    graph.add_argument("--checkpoint", type=Path, required=True)
    graph.add_argument("--nemo-checkout", type=Path, required=True)
    graph.add_argument("--dependency-lock", type=Path, required=True)
    graph.add_argument("--device", default="cpu")
    graph.add_argument("--output", type=Path)
    sampling_validation = commands.add_parser("validate-sampling-manifest")
    sampling_validation.add_argument("--manifest", type=Path, required=True)
    sampling_validation.add_argument("--corpus-root", type=Path, required=True)
    sampling_validation.add_argument("--reference-root", type=Path, required=True)
    sampling_validation.add_argument("--output", type=Path, required=True)
    canary = commands.add_parser("canary-arm")
    canary.add_argument("--checkpoint", type=Path, required=True)
    canary.add_argument("--nemo-checkout", type=Path, required=True)
    canary.add_argument("--dependency-lock", type=Path, required=True)
    canary.add_argument("--corpus-root", type=Path, required=True)
    canary.add_argument("--reference-root", type=Path, required=True)
    canary.add_argument("--manifest", type=Path, required=True)
    canary.add_argument("--arm", choices=("H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"), required=True)
    canary.add_argument("--device", default="cuda")
    canary.add_argument("--staged-state", type=Path)
    canary.add_argument("--staged-dev-result", type=Path, action="append", default=[])
    canary.add_argument("--output", type=Path, required=True)
    overfit_run = commands.add_parser("overfit-arm")
    overfit_run.add_argument("--checkpoint", type=Path, required=True)
    overfit_run.add_argument("--nemo-checkout", type=Path, required=True)
    overfit_run.add_argument("--dependency-lock", type=Path, required=True)
    overfit_run.add_argument("--corpus-root", type=Path, required=True)
    overfit_run.add_argument("--reference-root", type=Path, required=True)
    overfit_run.add_argument("--manifest", type=Path, required=True)
    overfit_run.add_argument("--class-weights", type=Path, required=True)
    overfit_run.add_argument(
        "--arm", choices=("H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"), required=True
    )
    overfit_run.add_argument("--device", default="cuda")
    overfit_run.add_argument("--staged-state", type=Path)
    overfit_run.add_argument("--staged-dev-result", type=Path, action="append", default=[])
    overfit_run.add_argument("--output", type=Path, required=True)
    overfit_run.add_argument("--selected-rows-output", type=Path, required=True)
    overfit_build = commands.add_parser("build-overfit-receipt")
    overfit_build.add_argument("bundle", type=Path)
    overfit_build.add_argument("--manifest", type=Path, required=True)
    overfit_build.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train-arm")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--nemo-checkout", type=Path, required=True)
    train.add_argument("--dependency-lock", type=Path, required=True)
    train.add_argument("--corpus-root", type=Path, required=True)
    train.add_argument("--reference-root", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--class-weights", type=Path, required=True)
    train.add_argument("--material-gate", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--training-output", type=Path, required=True)
    train.add_argument("--checkpoint-receipt-output", type=Path, required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--nemo-checkout", type=Path, required=True)
    infer.add_argument("--dependency-lock", type=Path, required=True)
    infer.add_argument("--corpus-root", type=Path, required=True)
    infer.add_argument("--reference-root", type=Path, required=True)
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--protocol-registry-root", type=Path, required=True)
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--role", choices=("PSEM-STRATEGY-DEV", "PSEM-STRATEGY-EVAL"), required=True)
    infer.add_argument(
        "--arm", choices=("F0-FROZEN-FLOAT", "H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"), required=True
    )
    infer.add_argument("--seed", type=int)
    infer.add_argument("--trained-checkpoint", type=Path)
    infer.add_argument("--trained-checkpoint-receipt", type=Path)
    infer.add_argument("--eval-authorization", type=Path)
    infer.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("prediction_set", type=Path)
    evaluate.add_argument("--eval-authorization", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    stage_init = commands.add_parser("stage-init")
    stage_init.add_argument("f0_dev_result", type=Path)
    stage_init.add_argument("--output", type=Path, required=True)
    stage_append = commands.add_parser("stage-append")
    stage_append.add_argument("state", type=Path)
    stage_append.add_argument("result", type=Path)
    stage_append.add_argument("--prior-result", type=Path, action="append", required=True)
    stage_append.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze-candidates")
    freeze.add_argument("bundle", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    eval_open = commands.add_parser("open-eval")
    eval_open.add_argument("candidate_freeze", type=Path)
    eval_open.add_argument("--output-root", type=Path, required=True)
    final_report = commands.add_parser("final-report")
    final_report.add_argument("bundle", type=Path)
    final_report.add_argument("--output-root", type=Path, required=True)
    lineage_auth = commands.add_parser("lineage-authorization")
    lineage_auth.add_argument("--output", type=Path, required=True)
    lineage_build = commands.add_parser("build-lineage")
    lineage_build.add_argument("--checkpoint", type=Path, required=True)
    lineage_build.add_argument("--nemo-checkout", type=Path, required=True)
    lineage_build.add_argument("--dependency-lock", type=Path, required=True)
    lineage_build.add_argument("--corpus-root", type=Path, required=True)
    lineage_build.add_argument("--reference-root", type=Path, required=True)
    lineage_build.add_argument("--output-root", type=Path, required=True)
    lineage_build.add_argument("--authorization", type=Path, required=True)
    lineage_build.add_argument("--device", default="cuda")
    lineage_build.add_argument("--output", type=Path, required=True)
    lineage_build.add_argument("--runtime-identity-output", type=Path, required=True)
    material_assemble = commands.add_parser("assemble-material-bundle")
    material_assemble.add_argument("inputs", type=Path)
    material_assemble.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "data-split-receipt":
        receipt = build_data_split_receipt()
        if args.output:
            write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "evaluator-contract":
        receipt = evaluator_reconstruction_contract()
        if args.output:
            write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
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
        if args.output:
            write_json(args.output, receipt)
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
        if args.output:
            write_json(args.output, receipt)
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
            staged_dev_results=bundle["staged_dev_results"],
        )
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "class-weights":
        sessions = load_training_sessions(args.corpus_root, args.reference_root)
        validate_sampling_manifest(args.manifest, sessions)
        receipt = build_manifest_class_weight_receipt(
            load_sampling_rows(args.manifest), sessions, args.manifest
        )
        if args.output:
            write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "model-graph":
        _, receipt = load_pinned_sortformer(
            args.checkpoint,
            args.nemo_checkout,
            args.dependency_lock,
            args.device,
        )
        if args.output:
            write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "validate-sampling-manifest":
        receipt = validate_sampling_manifest(
            args.manifest,
            load_training_sessions(args.corpus_root, args.reference_root),
        )
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "canary-arm":
        receipt = run_canary_arm(
            checkpoint_path=args.checkpoint,
            nemo_checkout=args.nemo_checkout,
            dependency_lock=args.dependency_lock,
            corpus_root=args.corpus_root,
            reference_root=args.reference_root,
            sampling_manifest=args.manifest,
            arm=args.arm,
            device=args.device,
            staged_execution_receipt=(_load_json(args.staged_state) if args.staged_state else None),
            staged_dev_results=[_load_json(path) for path in args.staged_dev_result],
        )
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "overfit-arm":
        result, selected = run_overfit_arm_result(
            checkpoint_path=args.checkpoint,
            nemo_checkout=args.nemo_checkout,
            dependency_lock=args.dependency_lock,
            corpus_root=args.corpus_root,
            reference_root=args.reference_root,
            sampling_manifest=args.manifest,
            class_weight_receipt=_load_json(args.class_weights),
            arm=args.arm,
            device=args.device,
            staged_execution_receipt=(_load_json(args.staged_state) if args.staged_state else None),
            staged_dev_results=[_load_json(path) for path in args.staged_dev_result],
        )
        write_json(args.output, result)
        write_jsonl(args.selected_rows_output, selected)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-overfit-receipt":
        bundle = _load_json(args.bundle)
        sampling_rows = load_sampling_rows(args.manifest)
        source_rows = [
            json.loads(line) for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines()
        ]
        receipt = build_overfit_receipt(
            bundle["arm_results"],
            bundle["selected_rows"],
            {row["source_id"]: row["corpus"] for row in source_rows},
            sampling_rows,
            args.manifest,
            bundle["canary_receipts"],
        )
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "train-arm":
        result, checkpoint_receipt = run_training_arm(
            checkpoint_path=args.checkpoint,
            nemo_checkout=args.nemo_checkout,
            dependency_lock=args.dependency_lock,
            corpus_root=args.corpus_root,
            reference_root=args.reference_root,
            sampling_manifest=args.manifest,
            class_weight_receipt=_load_json(args.class_weights),
            material_gate=_load_json(args.material_gate),
            output_root=args.output_root,
            device=args.device,
        )
        write_json(args.training_output, result)
        write_json(args.checkpoint_receipt_output, checkpoint_receipt)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "infer":
        receipt = infer_prediction_set(
            checkpoint_path=args.checkpoint,
            nemo_checkout=args.nemo_checkout,
            dependency_lock=args.dependency_lock,
            corpus_root=args.corpus_root,
            reference_root=args.reference_root,
            output_root=args.output_root,
            protocol_registry_root=args.protocol_registry_root,
            device=args.device,
            role=args.role,
            arm=args.arm,
            seed=args.seed,
            trained_checkpoint_path=args.trained_checkpoint,
            trained_checkpoint_receipt=(
                _load_json(args.trained_checkpoint_receipt)
                if args.trained_checkpoint_receipt
                else None
            ),
            eval_authorization=(
                _load_json(args.eval_authorization) if args.eval_authorization else None
            ),
        )
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "evaluate":
        receipt = evaluate_prediction_set(
            _load_json(args.prediction_set),
            _load_json(args.eval_authorization) if args.eval_authorization else None,
        )
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "stage-init":
        receipt = initial_staged_state(_load_json(args.f0_dev_result))
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "stage-append":
        receipt = append_dev_result(
            _load_json(args.state),
            _load_json(args.result),
            [_load_json(path) for path in args.prior_result],
        )
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "freeze-candidates":
        bundle = _load_json(args.bundle)
        results = [_load_json(path) for path in _load_path_list(bundle["results"], "results")]
        checkpoints = {
            key: _load_json(Path(path)) for key, path in bundle["checkpoint_receipts"].items()
        }
        predictions = {
            key: _load_json(Path(path)) for key, path in bundle["prediction_sets"].items()
        }
        receipt = freeze_candidate_set(
            _load_json(Path(bundle["state"])),
            results,
            checkpoints,
            predictions,
            candidate_code_identity(),
        )
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "open-eval":
        receipt = open_eval_once(_load_json(args.candidate_freeze), str(args.output_root.resolve()))
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "final-report":
        bundle = _load_json(args.bundle)
        eval_authorization = _load_json(Path(bundle["eval_authorization"]))
        if eval_authorization.get("experiment_output_root") != str(args.output_root.resolve()):
            raise ValueError("final report output root differs from EVAL authorization")
        validate_current_candidate_identity(
            {
                "schema_version": 1,
                "artifact_role": "psem_sortformer_candidate_code_identity",
                "git_head": eval_authorization["candidate_git_head"],
                "worktree_clean": True,
                "artifact_sha256s": eval_authorization["candidate_artifact_sha256s"],
                "payload_sha256": eval_authorization["candidate_code_identity_sha256"],
            }
        )
        artifacts, markdown = build_final_artifacts(
            eval_authorization=eval_authorization,
            eval_results=[
                _load_json(path) for path in _load_path_list(bundle["eval_results"], "eval_results")
            ],
            eval_prediction_sets=[
                _load_json(path)
                for path in _load_path_list(bundle["eval_prediction_sets"], "eval_prediction_sets")
            ],
            training_results=[
                _load_json(path)
                for path in _load_path_list(bundle["training_results"], "training_results")
            ],
        )
        for name, value in artifacts.items():
            path = args.output_root / name
            if name.endswith(".jsonl"):
                write_jsonl(path, value)
            else:
                write_json(path, value)
        decision_path = args.output_root / "ADAPTATION_DECISION.md"
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(markdown, encoding="utf-8", newline="\n")
        print(json.dumps({"output_root": str(args.output_root.resolve())}, sort_keys=True))
        return 0
    if args.command == "lineage-authorization":
        receipt = lineage_authorization()
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-lineage":
        receipt, runtime_identity = build_lineage_receipt(
            checkpoint_path=args.checkpoint,
            nemo_checkout=args.nemo_checkout,
            dependency_lock=args.dependency_lock,
            corpus_root=args.corpus_root,
            reference_root=args.reference_root,
            output_root=args.output_root,
            device=args.device,
            authorization=_load_json(args.authorization),
        )
        write_json(args.output, receipt)
        write_json(args.runtime_identity_output, runtime_identity)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "assemble-material-bundle":
        inputs = _load_json(args.inputs)
        canary = _load_json(Path(inputs["canary_bundle"]))
        all_canaries = {
            arm: _load_json(Path(path)) for arm, path in inputs["overfit_canary_bundles"].items()
        }
        bundle = {
            "arm": inputs["arm"],
            "seed": inputs["seed"],
            "preflight_receipt": _load_json(Path(inputs["preflight_receipt"])),
            "sampling_validation": _load_json(Path(inputs["sampling_validation"])),
            "class_weight_receipt": _load_json(Path(inputs["class_weight_receipt"])),
            "lineage_receipt": _load_json(Path(inputs["lineage_receipt"])),
            "runtime_identity": _load_json(Path(inputs["runtime_identity"])),
            "parameter_inventory": canary["parameter_inventory"],
            "gradient_receipt": canary["gradient_canary_receipt"],
            "update_receipt": canary["update_canary_receipt"],
            "timing_receipt": canary["timing_receipt"],
            "overfit_receipt": _load_json(Path(inputs["overfit_receipt"])),
            "overfit_canary_receipts": all_canaries,
            "staged_execution_receipt": _load_json(Path(inputs["staged_execution_receipt"])),
            "staged_dev_results": [
                _load_json(path)
                for path in _load_path_list(inputs["staged_dev_results"], "staged_dev_results")
            ],
        }
        write_json(args.output, bundle)
        print(json.dumps({"output": str(args.output.resolve())}, sort_keys=True))
        return 0
    paths = resolve_paths(
        checkpoint=args.checkpoint,
        corpus_root=args.corpus_root,
        reference_root=args.reference_root,
        output_root=args.output_root,
        protocol_registry_root=args.protocol_registry_root,
    )
    receipt = build_preflight(paths, static_only=args.static_only)
    if args.receipt_output:
        write_json(args.receipt_output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    passed = (
        receipt["static_contract_valid"] if args.static_only else receipt["ready_for_runtime_audit"]
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
