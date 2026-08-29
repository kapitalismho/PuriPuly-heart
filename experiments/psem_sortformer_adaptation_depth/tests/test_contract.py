import json

from experiments.psem_sortformer_adaptation_depth.preflight import PACKAGE_ROOT, static_checks
from experiments.psem_sortformer_adaptation_depth.runtime_audit import (
    LOW_LATENCY_STREAMING,
)
from experiments.psem_sortformer_adaptation_depth.sampling import (
    MAXIMUM_EPOCHS,
    ROLE_COUNTS,
    WINDOWS_PER_EPOCH,
)
from experiments.psem_sortformer_adaptation_depth.training import (
    GRADIENT_ACCUMULATION_STEPS,
    MICRO_BATCH_SIZE,
    OPTIMIZER_STEPS_PER_EPOCH,
)


def test_static_contract_is_bound_to_authoritative_artifacts() -> None:
    checks = static_checks()
    assert checks
    assert all(row["passed"] for row in checks), checks


def test_runtime_contract_matches_the_executable_recipe() -> None:
    runtime = json.loads((PACKAGE_ROOT / "runtime_contract.json").read_text(encoding="utf-8"))
    streaming = runtime["streaming"]
    assert LOW_LATENCY_STREAMING == {
        "chunk_len": streaming["chunk_len_frames"],
        "chunk_right_context": streaming["chunk_right_context_frames"],
        "fifo_len": streaming["fifo_len_frames"],
        "spkcache_update_period": streaming["speaker_cache_update_period_frames"],
        "spkcache_len": streaming["speaker_cache_len_frames"],
        "chunk_left_context": streaming["chunk_left_context_frames"],
    }
    sampling = runtime["sampling"]
    assert sampling["manifest_epochs"] == MAXIMUM_EPOCHS
    assert sampling["windows_per_epoch"] == WINDOWS_PER_EPOCH
    assert sampling["source_time_uniform_per_epoch"] == ROLE_COUNTS["source_time_uniform"]
    assert sampling["replacement_positive_per_epoch"] == ROLE_COUNTS["replacement_positive"]
    assert sampling["hard_negative_per_epoch"] == ROLE_COUNTS["hard_negative"]
    optimization = runtime["optimization_execution"]
    assert optimization["micro_batch_size"] == MICRO_BATCH_SIZE
    assert optimization["gradient_accumulation_steps"] == GRADIENT_ACCUMULATION_STEPS
    assert optimization["effective_windows_per_optimizer_step"] == (
        MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    )
    assert optimization["optimizer_steps_per_epoch"] == OPTIMIZER_STEPS_PER_EPOCH
