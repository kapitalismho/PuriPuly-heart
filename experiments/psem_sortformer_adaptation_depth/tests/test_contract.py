import json

from experiments.psem_sortformer_adaptation_depth.preflight import PACKAGE_ROOT, static_checks
from experiments.psem_sortformer_adaptation_depth.runtime_audit import LOW_LATENCY_STREAMING


def _load(name: str) -> dict:
    return json.loads((PACKAGE_ROOT / name).read_text(encoding="utf-8"))


def test_static_contract_is_bound_to_authoritative_artifacts() -> None:
    checks = static_checks()
    assert checks
    assert all(row["passed"] for row in checks), checks


def test_runtime_contract_records_ready_lean_authority() -> None:
    contract = _load("contract.json")
    config = _load("config.json")
    runtime = _load("runtime_contract.json")
    environment = _load("runtime_environment.json")
    streaming = runtime["streaming"]
    assert LOW_LATENCY_STREAMING == {
        "chunk_len": streaming["chunk_len_frames"],
        "chunk_right_context": streaming["chunk_right_context_frames"],
        "fifo_len": streaming["fifo_len_frames"],
        "spkcache_update_period": streaming["speaker_cache_update_period_frames"],
        "spkcache_len": streaming["speaker_cache_len_frames"],
        "chunk_left_context": streaming["chunk_left_context_frames"],
    }
    assert contract["authority"]["mode"] == "cost_bounded_hobby_engineering_probe"
    assert contract["material_execution"]["status"] == "ready"
    assert config["material_execution_status"] == "ready"
    assert runtime["material_execution"]["status"] == "ready"
    assert runtime["runtime_contract_version"] == "issue-107-lean-runtime-v5"
    assert runtime["material_execution"]["required_status_for_material_execution"] == "ready"
    assert runtime["material_execution"]["known_legacy_blockers"] == []
    assert config["optimization"]["seed"] == 7301
    assert config["optimization"]["confirmation_seed_allowed"] is False
    assert config["optimization"]["short_smoke_maximum_optimizer_steps"] == 32
    assert config["optimization"]["official_maximum_optimizer_steps"] == 256
    assert runtime["sampling"]["manifest_epochs"] == 1
    assert runtime["sampling"]["windows_per_manifest"] == 4096
    assert runtime["optimization_execution"]["seed"] == 7301
    assert config["optimization"]["micro_batch_size"] == 2
    assert config["optimization"]["gradient_accumulation_steps"] == 8
    assert environment["compute"]["micro_batch_size"] == 2
    assert environment["compute"]["gradient_accumulation_steps"] == 8
    assert runtime["optimization_execution"]["precision_mode"] == "float32"
    assert runtime["optimization_execution"]["mixed_precision"] is False
    assert runtime["optimization_execution"]["micro_batch_size"] == 2
    assert runtime["optimization_execution"]["gradient_accumulation_steps"] == 8
    assert runtime["optimization_execution"]["effective_windows_per_optimizer_step"] == 16
    assert runtime["optimization_execution"]["short_smoke_maximum_optimizer_steps"] == 32
    assert runtime["optimization_execution"]["official_maximum_optimizer_steps"] == 256
    assert runtime["evaluation"]["replacement_thresholds"] == [0.5]
    assert runtime["evaluation"]["replacement_confirmation_ms"] == [500]
    assert runtime["evaluation"]["bootstrap_required"] is False
    assert runtime["evaluation"]["eval_candidates"] == "F0_plus_dev_selected_candidate"
    assert environment["cost_budget"] == {
        "currency": "USD",
        "target_total": 15.0,
        "hard_stop_total": 30.0,
        "hourly_price_and_source_required": True,
        "actual_gpu_seconds_required": True,
        "projected_remaining_cost_required_before_material_run": True,
        "amendment_required_to_exceed_hard_stop": True,
    }
