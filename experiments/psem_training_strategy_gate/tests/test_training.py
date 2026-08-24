from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import experiments.psem_training_strategy_gate.training as training_module
from experiments.psem_training_strategy_gate.losses import (
    LossAccumulator,
    LossWeights,
    TargetBatch,
    compute_losses,
    loss_statistics,
)
from experiments.psem_training_strategy_gate.models import build_optimizer
from experiments.psem_training_strategy_gate.preflight import PreflightPaths, load_json
from experiments.psem_training_strategy_gate.sampling import WINDOWS_PER_EPOCH
from experiments.psem_training_strategy_gate.targets import (
    FUTURE_SAMPLES,
    WINDOW_SAMPLES,
    valid_center_samples,
)
from experiments.psem_training_strategy_gate.training import (
    OfficialTrainingSettings,
    TrainingAccumulator,
    TrainingContractError,
    _best_checkpoint,
    _build_scheduler,
    _checkpoint_payload,
    _checkpoint_valid,
    _descriptor,
    _improves,
    _initial_plan,
    _load_checkpoint,
    _model_state_schema,
    _next_run,
    _restore_checkpoint,
    _save_progress,
    _schedule_factor,
    _validate_plan,
    official_training_settings,
    train_official_run,
    training_status,
)


class _Head(torch.nn.Module):
    def relation_logits(
        self,
        hidden: torch.Tensor,
        batch_indices: torch.Tensor,
        left_cells: torch.Tensor,
        right_cells: torch.Tensor,
    ) -> torch.Tensor:
        return hidden[batch_indices, left_cells, 0] - hidden[batch_indices, right_cells, 0]


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.head = _Head()


class _TinyPSEM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.arm = "SCRATCH-PSEM"
        self.encoder = torch.nn.Linear(2, 2)
        self.projection = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 1)


def _receipt() -> dict[str, object]:
    return {
        "binding": {"git_commit": "a" * 40},
        "payload_sha256": "b" * 64,
    }


def _paths(root: Path) -> PreflightPaths:
    return PreflightPaths(root / "cache", root / "corpus", root / "reference", root / "output")


def test_official_training_settings_and_cosine_schedule_are_frozen() -> None:
    settings = official_training_settings()
    assert settings == OfficialTrainingSettings(20, 4, 4, 128, 250, 0.05, 5.0)
    assert _schedule_factor(0, 100, 5) == 0.2
    assert _schedule_factor(4, 100, 5) == 1.0
    assert _schedule_factor(100, 100, 5) == 0.0


def test_dev_loss_statistics_match_the_common_batch_objective() -> None:
    model = _TinyModel()
    outputs = {
        "handoff_logits": torch.tensor([0.2, -0.4]),
        "state_logits": torch.randn(2, 30, 3, generator=torch.Generator().manual_seed(4)),
        "hidden": torch.randn(2, 30, 2, generator=torch.Generator().manual_seed(5)),
    }
    targets = TargetBatch(
        handoff_targets=torch.tensor([1.0, 0.0]),
        handoff_mask=torch.tensor([True, True]),
        state_targets=torch.tensor([[0, 1, 2] * 10, [2, 1, 0] * 10]),
        state_mask=torch.ones((2, 30), dtype=torch.bool),
        relation_batch_indices=torch.tensor([0, 1]),
        relation_left_cells=torch.tensor([0, 1]),
        relation_right_cells=torch.tensor([1, 2]),
        relation_targets=torch.tensor([0.0, 1.0]),
    )
    weights = LossWeights(2.0, (0.5, 1.0, 1.5), (0.75, 1.25))
    expected = compute_losses(model, outputs, targets, weights)
    accumulator = LossAccumulator()
    accumulator.update(loss_statistics(model, outputs, targets, weights))
    actual = accumulator.result()
    assert actual["total"] == pytest.approx(float(expected["total"]))
    assert actual["handoff"] == pytest.approx(float(expected["handoff"]))
    assert actual["state"] == pytest.approx(float(expected["state"]))
    assert actual["relation"] == pytest.approx(float(expected["relation"]))


def test_plan_is_eval_sealed_and_requires_the_exact_run_order() -> None:
    plan = _initial_plan(_receipt())
    validated = _validate_plan(plan, _receipt())
    assert validated["eval_status"] == "sealed"
    assert (_next_run(validated)["arm"], _next_run(validated)["seed"]) == (
        "FROZEN-WAVLM",
        7301,
    )
    forged = deepcopy(plan)
    forged.pop("payload_sha256")
    forged["runs"][1]["status"] = "completed"
    forged["runs"][1]["completion_receipt"] = {
        "path": "x",
        "size_bytes": 1,
        "sha256": "c" * 64,
    }
    forged = training_module._payload(forged)
    with pytest.raises(TrainingContractError, match="completion order"):
        _validate_plan(forged)
    forged = deepcopy(plan)
    forged.pop("payload_sha256")
    forged["runs"][1]["status"] = "running"
    forged = training_module._payload(forged)
    with pytest.raises(TrainingContractError, match="active row"):
        _validate_plan(forged)


def test_material_guard_failure_precedes_any_run_state_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)

    def reject(_: PreflightPaths) -> dict[str, object]:
        raise TrainingContractError("guard rejected")

    monkeypatch.setattr(training_module, "_guard", reject)
    with pytest.raises(TrainingContractError, match="guard rejected"):
        train_official_run(paths, "FROZEN-WAVLM", 7301)
    assert not (paths.output_root / training_module.RUNS_DIRECTORY).exists()


def test_public_runner_rejects_skips_and_records_a_completed_first_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(training_module, "_guard", lambda _: _receipt())
    monkeypatch.setattr(
        training_module,
        "_run_identity",
        lambda _paths, _receipt_value, arm, seed, _settings, _execution: {
            "run_id": training_module._run_id(arm, seed),
            "payload_sha256": "d" * 64,
        },
    )

    def execute(
        _paths_value: PreflightPaths,
        identity: dict[str, object],
        _settings: OfficialTrainingSettings,
    ) -> dict[str, object]:
        run_root = paths.output_root / training_module.RUNS_DIRECTORY / str(identity["run_id"])
        value = training_module._payload(
            {
                "artifact_role": "psem_official_training_completion",
                "run_id": identity["run_id"],
            }
        )
        training_module._write_json(run_root / "completion_receipt.json", value)
        return value

    monkeypatch.setattr(training_module, "_execute_run", execute)
    with pytest.raises(TrainingContractError, match="requires FROZEN-WAVLM seed 7301"):
        train_official_run(paths, "FINETUNE-WAVLM", 7301)
    result = train_official_run(paths, "FROZEN-WAVLM", 7301)
    assert result["run_id"] == "frozen-wavlm-seed-7301"
    plan = _validate_plan(
        load_json(
            paths.output_root / training_module.RUNS_DIRECTORY / training_module.PLAN_FILENAME
        ),
        _receipt(),
    )
    assert plan["eval_status"] == "sealed"
    assert plan["runs"][0]["status"] == "completed"
    assert plan["runs"][1]["status"] == "pending"


def test_checkpoint_selection_uses_ap_then_dev_loss() -> None:
    best = {"event_average_precision": 0.5, "total_loss": 1.0}
    assert _improves(0.6, 2.0, best)
    assert _improves(0.5, 0.9, best)
    assert not _improves(0.5, 1.1, best)
    assert not _improves(0.4, 0.1, best)


def test_best_checkpoint_two_slots_preserve_the_previous_valid_artifact(tmp_path: Path) -> None:
    model = _TinyModel()
    identity = {
        "payload_sha256": "f" * 64,
        "arm": "FROZEN-WAVLM",
        "seed": 7301,
    }
    first_descriptor = _best_checkpoint(
        tmp_path,
        identity,
        model,
        1,
        {"event_average_precision": 0.5, "total_loss": 1.0, "collar_ms": 250},
        None,
    )
    first = {"checkpoint": first_descriptor}
    second_descriptor = _best_checkpoint(
        tmp_path,
        identity,
        model,
        2,
        {"event_average_precision": 0.6, "total_loss": 0.9, "collar_ms": 250},
        first,
    )
    assert Path(first_descriptor["path"]).name == "best-a.pt"
    assert Path(second_descriptor["path"]).name == "best-b.pt"
    assert training_module._descriptor_valid(first_descriptor)
    assert training_module._descriptor_valid(second_descriptor)


def test_two_slot_checkpoint_resume_selects_the_newest_valid_state(tmp_path: Path) -> None:
    identity = {"run_id": "run", "payload_sha256": "e" * 64}
    run_root = tmp_path / "run"
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    settings = OfficialTrainingSettings(20, 4, 4, 128, 250, 0.05, 5.0)
    scheduler = _build_scheduler(optimizer, settings)
    first = _checkpoint_payload(
        run_identity_sha256=identity["payload_sha256"],
        sequence=0,
        phase="train",
        epoch=1,
        next_batch_index=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulator=TrainingAccumulator(),
        history=[],
        best=None,
        stale_epochs=0,
    )
    _save_progress(run_root, identity, first)
    with torch.no_grad():
        model.linear.weight.add_(1.0)
    second = _checkpoint_payload(
        run_identity_sha256=identity["payload_sha256"],
        sequence=1,
        phase="train",
        epoch=1,
        next_batch_index=4,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulator=TrainingAccumulator(
            batch_count=1,
            window_count=4,
            total_loss_sum=1.0,
            handoff_loss_sum=1.0,
            state_loss_sum=1.0,
            relation_loss_sum=1.0,
            handoff_valid_count=4,
            state_valid_count=120,
            relation_valid_count=9,
            gradient_norm_max=1.0,
            elapsed_seconds=1.0,
            peak_rss_bytes=1,
        ),
        history=[],
        best=None,
        stale_epochs=0,
    )
    _save_progress(run_root, identity, second)
    restored = _load_checkpoint(
        run_root,
        identity["payload_sha256"],
        _model_state_schema(model),
    )
    assert restored is not None
    assert restored["sequence"] == 1
    assert restored["next_batch_index"] == 4
    fresh = _TinyModel()
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_scheduler = _build_scheduler(fresh_optimizer, settings)
    _restore_checkpoint(
        restored,
        fresh,
        fresh_optimizer,
        fresh_scheduler,
        torch.device("cpu"),
    )
    assert torch.equal(fresh.linear.weight, model.linear.weight)


def test_missing_checkpoint_files_fail_when_run_state_records_progress(tmp_path: Path) -> None:
    identity = {"run_id": "run", "payload_sha256": "1" * 64}
    run_root = tmp_path / "run"
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = _build_scheduler(optimizer, official_training_settings())
    checkpoint = _checkpoint_payload(
        run_identity_sha256=identity["payload_sha256"],
        sequence=0,
        phase="train",
        epoch=1,
        next_batch_index=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulator=TrainingAccumulator(),
        history=[],
        best=None,
        stale_epochs=0,
    )
    _save_progress(run_root, identity, checkpoint)
    (run_root / "checkpoints" / "latest-a.pt").unlink()
    with pytest.raises(TrainingContractError, match="checkpoints are absent"):
        _load_checkpoint(
            run_root,
            identity["payload_sha256"],
            _model_state_schema(model),
        )


def test_deeply_malformed_newest_checkpoint_falls_back_to_previous_slot(tmp_path: Path) -> None:
    identity = {"run_id": "run", "payload_sha256": "2" * 64}
    run_root = tmp_path / "run"
    model = _TinyPSEM()
    optimizer = build_optimizer(model)
    settings = official_training_settings()
    scheduler = _build_scheduler(optimizer, settings)
    first = _checkpoint_payload(
        run_identity_sha256=identity["payload_sha256"],
        sequence=0,
        phase="train",
        epoch=1,
        next_batch_index=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulator=TrainingAccumulator(),
        history=[],
        best=None,
        stale_epochs=0,
    )
    _save_progress(run_root, identity, first)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()
    scheduler.step()
    second = _checkpoint_payload(
        run_identity_sha256=identity["payload_sha256"],
        sequence=1,
        phase="train",
        epoch=1,
        next_batch_index=4,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulator=TrainingAccumulator(
            batch_count=1,
            window_count=4,
            total_loss_sum=1.0,
            handoff_loss_sum=1.0,
            state_loss_sum=1.0,
            relation_loss_sum=1.0,
            handoff_valid_count=4,
            state_valid_count=120,
            relation_valid_count=8,
            gradient_norm_max=1.0,
            elapsed_seconds=1.0,
            peak_rss_bytes=1,
        ),
        history=[],
        best=None,
        stale_epochs=0,
    )
    _save_progress(run_root, identity, second)
    valid_model = _TinyPSEM()
    valid_optimizer = build_optimizer(valid_model)
    valid_scheduler = _build_scheduler(valid_optimizer, settings)
    valid = _load_checkpoint(
        run_root,
        identity["payload_sha256"],
        _model_state_schema(valid_model),
        model=valid_model,
        optimizer=valid_optimizer,
        scheduler=valid_scheduler,
        device=torch.device("cpu"),
    )
    assert valid is not None
    assert valid["sequence"] == 1
    newest_path = run_root / "checkpoints" / "latest-b.pt"
    newest = torch.load(newest_path, map_location="cpu", weights_only=False)
    swapped = deepcopy(newest)
    saved_parameters = swapped["optimizer_state"]["param_groups"][0]["params"]
    saved_parameters[0], saved_parameters[1] = saved_parameters[1], saved_parameters[0]
    torch.save(swapped, newest_path)
    swapped_model = _TinyPSEM()
    swapped_optimizer = build_optimizer(swapped_model)
    swapped_scheduler = _build_scheduler(swapped_optimizer, settings)
    swapped_result = _load_checkpoint(
        run_root,
        identity["payload_sha256"],
        _model_state_schema(swapped_model),
        model=swapped_model,
        optimizer=swapped_optimizer,
        scheduler=swapped_scheduler,
        device=torch.device("cpu"),
    )
    assert swapped_result is not None
    assert swapped_result["sequence"] == 0
    torch.save(newest, newest_path)
    state = next(iter(newest["optimizer_state"]["state"].values()))
    state["exp_avg"] = torch.zeros(3)
    torch.save(newest, newest_path)
    fresh = _TinyPSEM()
    fresh_optimizer = build_optimizer(fresh)
    fresh_scheduler = _build_scheduler(fresh_optimizer, settings)
    restored = _load_checkpoint(
        run_root,
        identity["payload_sha256"],
        _model_state_schema(fresh),
        model=fresh,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        device=torch.device("cpu"),
    )
    assert restored is not None
    assert restored["sequence"] == 0


def test_file_snapshot_rejects_replacement_with_preserved_size_and_mtime(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"original")
    descriptor = _descriptor(path)
    inputs = {"input": descriptor}
    identity = {
        "inputs": inputs,
        "file_snapshots": training_module._file_snapshots(inputs),
    }
    observed = path.stat()
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replaced")
    os.utime(replacement, ns=(observed.st_atime_ns, observed.st_mtime_ns))
    replacement.replace(path)
    with pytest.raises(TrainingContractError, match="changed during execution"):
        training_module._assert_file_snapshots(identity)


def test_dev_metric_timing_and_timestamp_semantics_fail_closed() -> None:
    valid_timing = {
        "window_count": 2,
        "per_window_seconds_p50": 0.1,
        "per_window_seconds_p95": 0.2,
    }
    assert training_module._timing_valid(valid_timing, 2)
    assert not training_module._timing_valid({**valid_timing, "window_count": "2"}, 2)
    assert not training_module._timing_valid({**valid_timing, "window_count": True}, 1)
    assert not training_module._timing_valid({**valid_timing, "window_count": 1.0}, 1)
    assert not training_module._timing_valid(
        {**valid_timing, "per_window_seconds_p95": 0.05}, 2
    )
    assert training_module._timestamp_valid("2026-08-25T01:02:03+00:00")
    assert not training_module._timestamp_valid("not-a-timestamp")


def test_running_progress_requires_run_local_semantic_dev_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    dev_root = run_root / "dev"
    dev_root.mkdir(parents=True)
    raw_path = dev_root / "epoch-01-predictions.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    contract = {
        "payload_sha256": "a" * 64,
        "dev_sources": [
            {
                "source_id": "dev",
                "scored_start_sample": 0,
                "scored_end_sample": WINDOW_SAMPLES,
                "prediction_count": 1,
                "scored_prediction_count": 1,
                "excluded_prediction_count": 0,
                "scored_mask_sha256": "b" * 64,
            }
        ],
    }
    metrics = training_module._payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_dev_checkpoint_metrics",
            "run_identity_sha256": contract["payload_sha256"],
            "epoch": 1,
            "data_role": training_module.DEV_ROLE,
            "eval_opened": False,
            "source_ids": ["dev"],
            "scored_source_samples": WINDOW_SAMPLES,
            "prediction_count": 1,
            "scored_prediction_count": 1,
            "excluded_prediction_count": 0,
            "candidate_count": 1,
            "reference_count": 1,
            "checkpoint_metric": {
                "event_average_precision": 0.5,
                "maximum_f1": 0.5,
                "candidate_count": 1,
                "reference_count": 1,
                "threshold_count": 1,
                "collar_ms": 250,
            },
            "losses": {"total": 1.0, "handoff": 1.0, "state": 1.0, "relation": 1.0},
            "timing": {
                "window_count": 1,
                "per_window_seconds_p50": 0.1,
                "per_window_seconds_p95": 0.1,
            },
            "peak_rss_bytes": 1,
            "raw_predictions": _descriptor(raw_path),
            "generated_at": "2026-08-25T01:02:03+00:00",
        }
    )
    metrics_path = dev_root / "epoch-01-metrics.json"
    training_module._write_json(metrics_path, metrics)
    row = {
        "epoch": 1,
        "dev_event_average_precision": 0.5,
        "dev_total_loss": 1.0,
        "dev_metrics": _descriptor(metrics_path),
    }
    assert training_module._progress_dev_metric_valid(run_root, row, contract)
    outside = tmp_path / "outside.json"
    training_module._write_json(outside, metrics)
    assert not training_module._progress_dev_metric_valid(
        run_root,
        {**row, "dev_metrics": _descriptor(outside)},
        contract,
    )


def test_run_state_is_verified_before_checkpoint_resume(tmp_path: Path) -> None:
    identity = {"run_id": "run", "payload_sha256": "3" * 64}
    run_root = tmp_path / "run"
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = _build_scheduler(optimizer, official_training_settings())
    checkpoint = _checkpoint_payload(
        run_identity_sha256=identity["payload_sha256"],
        sequence=0,
        phase="train",
        epoch=1,
        next_batch_index=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulator=TrainingAccumulator(),
        history=[],
        best=None,
        stale_epochs=0,
    )
    _save_progress(run_root, identity, checkpoint)
    state = dict(load_json(run_root / "run_state.json"))
    state.pop("payload_sha256")
    state["eval_opened"] = True
    training_module._write_json(run_root / "run_state.json", training_module._payload(state))
    with pytest.raises(TrainingContractError, match="run state is invalid"):
        _load_checkpoint(
            run_root,
            identity["payload_sha256"],
            _model_state_schema(model),
        )


def test_premature_complete_checkpoint_is_invalid(tmp_path: Path) -> None:
    identity_sha = "4" * 64
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = _build_scheduler(optimizer, official_training_settings())
    metrics_path = tmp_path / "dev.json"
    metrics_path.write_text("{}\n", encoding="utf-8")
    best_path = tmp_path / "best.pt"
    best_path.write_bytes(b"best")
    history = [
        {
            "epoch": 1,
            "train": {
                "batch_count": WINDOWS_PER_EPOCH // 4,
                "window_count": WINDOWS_PER_EPOCH,
                "mean_losses": {
                    "total": 1.0,
                    "handoff": 1.0,
                    "state": 1.0,
                    "relation": 1.0,
                },
                "valid_counts": {"handoff": 1, "state": 1, "relation": 1},
                "gradient_norm_max": 1.0,
                "elapsed_seconds": 1.0,
                "peak_rss_bytes": 1,
            },
            "dev_event_average_precision": 0.5,
            "dev_total_loss": 1.0,
            "checkpoint_matching_collar_ms": 250,
            "improved": True,
            "dev_metrics": _descriptor(metrics_path),
        }
    ]
    checkpoint = _checkpoint_payload(
        run_identity_sha256=identity_sha,
        sequence=1,
        phase="complete",
        epoch=1,
        next_batch_index=WINDOWS_PER_EPOCH,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulator=TrainingAccumulator(),
        history=history,
        best={
            "epoch": 1,
            "event_average_precision": 0.5,
            "total_loss": 1.0,
            "collar_ms": 250,
            "checkpoint": _descriptor(best_path),
        },
        stale_epochs=0,
    )
    assert not _checkpoint_valid(
        checkpoint,
        identity_sha,
        _model_state_schema(model),
        slot="b",
    )


def test_dev_predictions_are_bound_to_path_role_and_inventory(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    path = run_root / "dev" / "epoch-01-predictions.jsonl"
    path.parent.mkdir(parents=True)
    centers = list(valid_center_samples(0, WINDOW_SAMPLES))
    contract = {
        "dev_sources": [
            {
                "source_id": "dev-source",
                "scored_start_sample": 0,
                "scored_end_sample": WINDOW_SAMPLES,
                "prediction_count": len(centers),
                "scored_prediction_count": len(centers),
                "excluded_prediction_count": 0,
                "scored_mask_sha256": training_module.canonical_sha256([True] * len(centers)),
            }
        ]
    }

    def write_predictions(artifact_role: str) -> dict[str, object]:
        rows = [
            {
                "schema_version": 1,
                "artifact_role": artifact_role,
                "source_id": "dev-source",
                "boundary_sample": center,
                "observed_frontier_sample": center + FUTURE_SAMPLES,
                "score": 0.5,
                "scored": True,
            }
            for center in centers
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return {
            "raw_predictions": _descriptor(path),
            "prediction_count": len(rows),
            "scored_prediction_count": len(rows),
            "excluded_prediction_count": 0,
        }

    training_module._validate_dev_predictions(
        run_root, 1, write_predictions("psem_dev_prediction_score"), contract
    )
    with pytest.raises(TrainingContractError, match="prediction row is invalid"):
        training_module._validate_dev_predictions(
            run_root,
            1,
            write_predictions("psem_eval_prediction_score"),
            contract,
        )
    outside = tmp_path / "outside.jsonl"
    outside.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(TrainingContractError, match="escape their run binding"):
        training_module._validate_dev_predictions(
            run_root,
            1,
            {
                "raw_predictions": _descriptor(outside),
                "prediction_count": len(centers),
                "scored_prediction_count": len(centers),
                "excluded_prediction_count": 0,
            },
            contract,
        )


def test_training_status_revalidates_the_plan_and_completed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    plan_path = paths.output_root / training_module.RUNS_DIRECTORY / training_module.PLAN_FILENAME
    training_module._write_json(plan_path, _initial_plan(_receipt()))
    monkeypatch.setattr(training_module, "_guard", lambda _: _receipt())

    def reject(*_: object) -> None:
        raise TrainingContractError("completed prefix rejected")

    monkeypatch.setattr(training_module, "_verify_completed_prefix", reject)
    with pytest.raises(TrainingContractError, match="completed prefix rejected"):
        training_status(paths)
