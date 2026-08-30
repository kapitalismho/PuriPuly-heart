import json

import pytest

from experiments.psem_sortformer_adaptation_depth import execution as execution_module
from experiments.psem_sortformer_adaptation_depth import run as run_module
from experiments.psem_sortformer_adaptation_depth import training as training_module
from experiments.psem_sortformer_adaptation_depth.execution import (
    CHECKPOINT_SHA256,
    ExecutionError,
    build_cost_receipt,
    run_overfit_arm_result,
)
from experiments.psem_sortformer_adaptation_depth.preflight import canonical_sha256
from experiments.psem_sortformer_adaptation_depth.protocol import bind_payload
from experiments.psem_sortformer_adaptation_depth.receipts import validate_cost_receipt
from experiments.psem_sortformer_adaptation_depth.run import main


def test_cost_receipt_uses_exact_arithmetic_and_allows_the_hard_stop_boundary() -> None:
    receipt = build_cost_receipt(
        hourly_price_usd=36.0,
        hourly_price_source="trusted operator quote",
        actual_gpu_seconds=1800.0,
        projected_remaining_gpu_seconds=1200.0,
        command="train-arm H-HEAD",
    )
    assert receipt["actual_cost_usd"] == 18.0
    assert receipt["projected_remaining_cost_usd"] == 12.0
    assert receipt["projected_total_cost_usd"] == 30.0
    assert receipt["target_is_informational"] is True
    assert validate_cost_receipt(receipt) == receipt


def test_cost_receipt_rejects_a_projection_above_usd_30() -> None:
    with pytest.raises(ExecutionError, match="USD-30 hard stop"):
        build_cost_receipt(
            hourly_price_usd=36.0,
            hourly_price_source="trusted operator quote",
            actual_gpu_seconds=1800.0,
            projected_remaining_gpu_seconds=1201.0,
            command="train-arm H-HEAD",
        )


def test_cost_receipt_cli_preserves_the_executed_command(tmp_path) -> None:
    output = tmp_path / "cost.json"
    assert (
        main(
            [
                "cost-receipt",
                "--hourly-price-usd",
                "1",
                "--hourly-price-source",
                "operator quote",
                "--actual-gpu-seconds",
                "3600",
                "--projected-remaining-gpu-seconds",
                "3600",
                "--command",
                "train-arm H-HEAD",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["command"] == "train-arm H-HEAD"


def test_legacy_overfit_runner_is_a_fail_closed_tombstone() -> None:
    with pytest.raises(ExecutionError, match="legacy overfit arm is not supported; use smoke-arm"):
        run_overfit_arm_result()


def test_smoke_receipt_binds_official_weights_and_runtime_identity(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "sampling.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    rows = [
        {
            "row_id": f"epoch-01-window-{index:04d}",
            "epoch_index": index,
            "source_id": "source",
            "corpus": "AMI",
        }
        for index in range(512)
    ]
    sessions = {"source": object()}
    official_weights = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "train_class_weight_receipt",
            "replacement_positive_weight": 2.0,
            "anchor_positive_weight": 3.0,
        }
    )
    runtime_identity = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dependency_lock": {"sha256": "d" * 64},
        "dependency_lock_sha256": "d" * 64,
    }
    registered = []
    monkeypatch.setattr(execution_module, "require_material_execution_ready", lambda: None)
    monkeypatch.setattr(execution_module, "_eval_registry_marker", lambda: tmp_path / "sealed")
    monkeypatch.setattr(execution_module, "seed_runtime", lambda _seed: None)
    monkeypatch.setattr(execution_module, "load_sampling_rows", lambda _path: rows)
    monkeypatch.setattr(execution_module, "load_training_sessions", lambda *_args: sessions)
    monkeypatch.setattr(execution_module, "validate_sampling_manifest", lambda *_args: {})
    monkeypatch.setattr(
        execution_module, "prepare_training_example", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        training_module,
        "build_manifest_class_weight_receipt",
        lambda *_args: official_weights,
    )
    monkeypatch.setattr(
        execution_module,
        "load_pinned_sortformer",
        lambda *_args: (object(), runtime_identity),
    )
    monkeypatch.setattr(
        execution_module,
        "apply_parameter_policy",
        lambda _model, arm: {"arm": arm},
    )
    monkeypatch.setattr(
        execution_module,
        "run_short_smoke",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "artifact_role": "short_smoke_metrics",
            "arm": "H-HEAD",
            "seed": 7301,
            "optimizer_steps": 32,
            "consumed_row_count": 512,
        },
    )
    monkeypatch.setattr(
        execution_module,
        "register_execution",
        lambda kind, value: registered.append((kind, value)),
    )

    receipt = execution_module.run_smoke_arm(
        checkpoint_path=tmp_path / "base.nemo",
        nemo_checkout=tmp_path / "nemo",
        dependency_lock=tmp_path / "lock.json",
        corpus_root=tmp_path / "corpus",
        reference_root=tmp_path / "reference",
        sampling_manifest=manifest,
        class_weight_receipt=official_weights,
        arm="H-HEAD",
        device="cuda",
    )

    assert receipt["class_weight_receipt_sha256"] == official_weights["payload_sha256"]
    assert receipt["runtime_identity"] == runtime_identity
    assert receipt["runtime_identity_sha256"] == canonical_sha256(runtime_identity)
    assert receipt["base_checkpoint_sha256"] == CHECKPOINT_SHA256
    assert receipt["dependency_lock_sha256"] == "d" * 64
    assert registered == [("short-smoke", receipt)]

    forged_weights = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "train_class_weight_receipt",
            "replacement_positive_weight": 4.0,
            "anchor_positive_weight": 3.0,
        }
    )
    with pytest.raises(ExecutionError, match="class weights differ"):
        execution_module.run_smoke_arm(
            checkpoint_path=tmp_path / "base.nemo",
            nemo_checkout=tmp_path / "nemo",
            dependency_lock=tmp_path / "lock.json",
            corpus_root=tmp_path / "corpus",
            reference_root=tmp_path / "reference",
            sampling_manifest=manifest,
            class_weight_receipt=forged_weights,
            arm="H-HEAD",
            device="cuda",
        )


def test_final_report_cli_writes_the_documented_decision_name(tmp_path, monkeypatch) -> None:
    output_root = (tmp_path / "output").resolve()
    output_root.mkdir()
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "experiment_output_root": str(output_root),
                "candidate_git_head": "a" * 40,
                "candidate_artifact_sha256s": {},
                "candidate_code_identity_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "eval_authorization": str(authorization_path),
                "eval_results": [],
                "eval_prediction_sets": [],
                "training_results": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_module, "validate_current_candidate_identity", lambda _value: None)
    monkeypatch.setattr(
        run_module,
        "build_final_artifacts",
        lambda **_kwargs: ({}, "decision\n"),
    )

    assert (
        run_module.main(["final-report", str(bundle_path), "--output-root", str(output_root)]) == 0
    )
    assert (output_root / "ADAPTATION_DECISION.md").read_text(encoding="utf-8") == "decision\n"
    assert not (output_root / "LEAN_ADAPTATION_DECISION.md").exists()


def test_final_report_cli_writes_stop_decision_without_opening_eval(tmp_path, monkeypatch) -> None:
    output_root = (tmp_path / "output").resolve()
    output_root.mkdir()
    operator_decision = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_operator_dev_decision",
            "decision": "stop",
            "selected_arm": None,
            "rationale": "The trusted operator found no supported adaptation path.",
            "eval_open_count": 0,
        }
    )
    candidate_freeze = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_candidate_freeze",
            "candidate_set": [],
            "operator_dev_decision": operator_decision,
            "operator_dev_decision_sha256": operator_decision["payload_sha256"],
            "eval_open_count": 0,
            "eval_used_for_development": False,
            "experiment_output_root": str(output_root),
            "candidate_git_head": "a" * 40,
            "candidate_artifact_sha256s": {},
            "candidate_code_identity_sha256": "b" * 64,
        }
    )
    freeze_path = tmp_path / "candidate-freeze.json"
    freeze_path.write_text(json.dumps(candidate_freeze), encoding="utf-8")
    monkeypatch.setattr(run_module, "validate_current_candidate_identity", lambda _value: None)
    monkeypatch.setattr(
        "experiments.psem_sortformer_adaptation_depth.protocol.validate_candidate_freeze",
        lambda value: value,
    )
    monkeypatch.setattr(
        run_module,
        "open_eval_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("EVAL opened")),
    )

    assert (
        run_module.main(["final-report", str(freeze_path), "--output-root", str(output_root)]) == 0
    )
    markdown = (output_root / "ADAPTATION_DECISION.md").read_text(encoding="utf-8")
    receipt = json.loads((output_root / "decision_receipt.json").read_text(encoding="utf-8"))
    assert receipt["outcome"] == "D"
    assert receipt["eval_open_count"] == 0
    assert "Outcome: **D**" in markdown
    assert "EVAL was not opened or used" in markdown
    assert "No student KD was performed or authorized" in markdown
    assert "does not authorize acoustic/NEST unfreezing" in markdown
    assert not (output_root / "LEAN_ADAPTATION_DECISION.md").exists()
