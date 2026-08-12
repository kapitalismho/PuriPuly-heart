from __future__ import annotations

import copy

import pytest

from experiments.speaker_turn_boundary.turn_episode.phase5_design import (
    Phase5DesignError,
    policy_space,
    runtime_forecast,
    validate_interstage_word_timing_receipts,
)
from experiments.speaker_turn_boundary.turn_episode.phase5_policy_benchmark import (
    historical_scoring_benchmark,
)
from experiments.speaker_turn_boundary.turn_episode.phase5_storage_benchmark import (
    BLOCK_COUNT,
    HISTORICAL_AGGREGATE_FIELDS,
    HISTORICAL_SYSTEM_AGGREGATE_ROWS,
    POOL_ORDER,
    PROPOSAL_EXECUTION_RECEIPT_FIELDS,
    SYSTEM_METRIC_FIELDS,
    current_system_aggregate_row,
    historical_system_aggregate_row,
    proposal_execution_row,
)


def test_compact_policy_space_logical_and_physical_counts() -> None:
    result = policy_space(
        4,
        {
            "diagnostic_dev": 695,
            "frontier_dev": 109,
            "natural_exposure_validation": 74,
        },
    )
    assert result["physical_execution_system_count"] == 2503
    assert result["logical_system_count"] == 4611
    assert result["logical_episode_identity_count"] == 4048458
    assert result["logical_ladder_alias_edge_count"] == 2108
    dag = result["content_addressed_execution_dag"]
    assert dag["proposal_profile_episode_batch_count"] == 3512
    assert dag["cluster_execution_count"] == 126432
    assert dag["fusion_execution_count"] == 505728
    assert dag["maximum_frequency_control_episode_count"] == 1517184
    assert dag["physical_execution_episode_node_count"] == 2197634
    assert dag["logical_ladder_alias_episode_edge_count"] == 1850824


def test_runtime_forecast_uses_batched_policy_benchmark() -> None:
    policy = policy_space(
        4,
        {
            "diagnostic_dev": 695,
            "frontier_dev": 109,
            "natural_exposure_validation": 74,
        },
    )
    result = runtime_forecast(
        {
            "new_inference_window_count": 207764,
            "reusable_checkpoint_window_jobs": 219802,
        },
        policy,
        {
            "content_sha256": "benchmark",
            "conservative_parallel_batches_per_second": 100.0,
            "selected_policy_workers": 8,
            "historical_worst_policy_grid_parallel": {
                "conservative_proposal_positions_per_second": 100.0
            },
            "source_prefix_state": {"conservative_probe_steps_per_second_floor": 100.0},
            "frequency_controls": {
                "parallel": {"conservative_action_placements_per_second": 100.0}
            },
            "scoring": {
                "source_manifest_byte_sha256": "manifest",
                "historical_exact_shapes": [
                    {
                        "case_id": "ami_ES2003a",
                        "word_interval_count": 2038,
                        "word_timing_observable": True,
                        "raw_word_record_count": 2387,
                        "word_annotation_files": [
                            {
                                "filename": "ES2003a.A.words.xml",
                                "byte_sha256": "es-digest",
                            }
                        ],
                        "word_record_sha256": "es-word-record-digest",
                    },
                    {
                        "case_id": "ami_IS1008a",
                        "word_interval_count": 2504,
                        "word_timing_observable": True,
                        "raw_word_record_count": 2890,
                        "word_annotation_files": [
                            {
                                "filename": "IS1008a.A.words.xml",
                                "byte_sha256": "is-digest",
                            }
                        ],
                        "word_record_sha256": "is-word-record-digest",
                    },
                ],
                "joint_forecast_envelope": {
                    "shape_id": "shape",
                    "action_count": 11392,
                    "reference_count": 132,
                    "region_count": 338,
                    "singleton_interval_count": 169,
                    "overlap_interval_count": 64,
                    "pause_interval_count": 10,
                    "unscored_interval_count": 0,
                    "word_interval_count": 2504,
                    "word_timing_observable": True,
                    "raw_word_record_count": 2890,
                    "word_annotation_files": [
                        {"filename": "IS1008a.A.words.xml", "byte_sha256": "digest"}
                    ],
                    "word_record_sha256": "word-record-digest",
                },
                "parallel": {"conservative_actions_per_second": 100.0},
            },
            "independent_verifier_recompute": {
                "algorithm": ["a", "b", "c", "d"],
                "representative_shapes": [0, 1],
                "worst_shapes": {"proposal_positions": 1},
                "serial": {"elapsed_seconds": 1.0},
                "parallel": {
                    "workers": 8,
                    "wall_seconds": 1.0,
                    "trace_sha256s": ["digest"],
                },
            },
            "logical_identity_digest": {"parallel": {"conservative_rows_per_second": 1000000.0}},
        },
        {"representation": {}},
        {
            "total_logical_emittable_position_count": 0,
            "total_physical_probe_step_count": 0,
        },
        {
            "proposal_profile_case_count": 0,
            "logical_emittable_position_count": 0,
            "proposal_probe_step_count": 0,
            "logical_policy_case_identity_count": 0,
            "logical_case_identity_count_including_baselines": 0,
        },
    )
    assert result["policy_replay_batch_count"] == 3512
    assert result["policy_replay_seconds"] == 35.12
    assert result["execution_ceiling_hours"] == 3.0
    assert result["within_execution_ceiling"] is True
    assert result["new_inference_workers"] == 10
    assert result["policy_replay_workers"] == 8
    assert result["independent_audit_contract"]["sample_size"] == 2048
    assert result["independent_verification_audit_parallel_batch_count"] == 256
    assert result["scoring_joint_forecast_envelope"]["reference_count"] == 132
    assert result["scoring_joint_forecast_envelope"]["word_interval_count"] == 2504
    assert result["scoring_joint_forecast_envelope"]["word_timing_observable"] is True
    assert (
        "joint_scoring_shape_count_by_action_reference_and_timeline_cardinalities"
        in result["interstage_exact_cardinality_gate"]["required_counts"]
    )


def test_interstage_gate_rejects_word_observability_and_count_drift() -> None:
    policy = policy_space(
        4,
        {
            "diagnostic_dev": 695,
            "frontier_dev": 109,
            "natural_exposure_validation": 74,
        },
    )
    benchmark = {
        "content_sha256": "benchmark",
        "conservative_parallel_batches_per_second": 100.0,
        "selected_policy_workers": 8,
        "historical_worst_policy_grid_parallel": {
            "conservative_proposal_positions_per_second": 100.0
        },
        "source_prefix_state": {"conservative_probe_steps_per_second_floor": 100.0},
        "frequency_controls": {"parallel": {"conservative_action_placements_per_second": 100.0}},
        "scoring": {
            "source_manifest_byte_sha256": "manifest",
            "historical_exact_shapes": [
                {
                    "case_id": "ami_IS1008a",
                    "word_interval_count": 2504,
                    "word_timing_observable": True,
                    "raw_word_record_count": 2890,
                    "word_annotation_files": [
                        {"filename": "IS1008a.A.words.xml", "byte_sha256": "digest"}
                    ],
                    "word_record_sha256": "word-record-digest",
                }
            ],
            "joint_forecast_envelope": {
                "shape_id": "shape",
                "action_count": 11392,
                "reference_count": 132,
                "region_count": 338,
                "singleton_interval_count": 169,
                "overlap_interval_count": 64,
                "pause_interval_count": 36,
                "unscored_interval_count": 0,
                "word_interval_count": 2504,
                "word_timing_observable": True,
                "raw_word_record_count": 2890,
                "word_annotation_files": [
                    {"filename": "IS1008a.A.words.xml", "byte_sha256": "digest"}
                ],
                "word_record_sha256": "word-record-digest",
            },
            "parallel": {"conservative_actions_per_second": 100.0},
        },
        "independent_verifier_recompute": {
            "algorithm": ["a", "b", "c", "d"],
            "representative_shapes": [0, 1],
            "worst_shapes": {"proposal_positions": 1},
            "serial": {"elapsed_seconds": 1.0},
            "parallel": {
                "workers": 8,
                "wall_seconds": 1.0,
                "trace_sha256s": ["digest"],
            },
        },
        "logical_identity_digest": {"parallel": {"conservative_rows_per_second": 1000000.0}},
    }
    forecast = runtime_forecast(
        {"new_inference_window_count": 0, "reusable_checkpoint_window_jobs": 0},
        policy,
        benchmark,
        {"representation": {}},
        {"total_logical_emittable_position_count": 0, "total_physical_probe_step_count": 0},
        {
            "proposal_profile_case_count": 0,
            "logical_emittable_position_count": 0,
            "proposal_probe_step_count": 0,
            "logical_policy_case_identity_count": 0,
            "logical_case_identity_count_including_baselines": 0,
        },
    )
    gate = forecast["interstage_exact_cardinality_gate"]
    expected = copy.deepcopy(gate["word_timing_receipt_contract"]["historical_sentinels"])
    receipt = {
        "expected_word_timing_receipts": expected,
        "observed_word_timing_receipts": copy.deepcopy(expected),
    }
    assert validate_interstage_word_timing_receipts(gate, receipt)["stage_b_allowed"] is True
    receipt["observed_word_timing_receipts"][0]["word_timing_observable"] = False
    with pytest.raises(Phase5DesignError, match="exact-value drift"):
        validate_interstage_word_timing_receipts(gate, receipt)
    receipt["observed_word_timing_receipts"] = copy.deepcopy(expected)
    receipt["observed_word_timing_receipts"][0]["word_interval_count"] = 0
    with pytest.raises(Phase5DesignError, match="exact-value drift"):
        validate_interstage_word_timing_receipts(gate, receipt)


def test_storage_rows_preserve_pool_and_reconstruction_dimensions() -> None:
    aggregate = current_system_aggregate_row(3)
    pool_rows = aggregate[2]
    block_rows = aggregate[3]
    assert [row[0] for row in pool_rows] == list(POOL_ORDER)
    assert len(block_rows) == BLOCK_COUNT
    assert {row[0] for row in block_rows} == set(POOL_ORDER)
    natural_indexes = [
        index for index, field in enumerate(SYSTEM_METRIC_FIELDS) if field.startswith("natural_")
    ]
    for row in pool_rows[:2]:
        metrics = row[3:]
        assert all(metrics[index] == 0 for index in natural_indexes)
    natural_metrics = pool_rows[2][3:]
    assert all(natural_metrics[index] > 0 for index in natural_indexes)
    receipt = proposal_execution_row(3)
    assert set(receipt) == set(PROPOSAL_EXECUTION_RECEIPT_FIELDS)
    assert receipt["input_cache_receipts_sha256"]
    assert receipt["lifecycle_trace_sha256"]
    assert receipt["state_snapshot_index_sha256"]
    assert receipt["tail_evidence_sha256"]


def test_historical_storage_includes_b0_b1_identities_and_metrics() -> None:
    assert HISTORICAL_SYSTEM_AGGREGATE_ROWS == 4610
    b0 = dict(zip(HISTORICAL_AGGREGATE_FIELDS, historical_system_aggregate_row(0)))
    b1 = dict(zip(HISTORICAL_AGGREGATE_FIELDS, historical_system_aggregate_row(1)))
    neural = dict(zip(HISTORICAL_AGGREGATE_FIELDS, historical_system_aggregate_row(2)))
    assert b0["logical_system_id"] == "baseline:B0"
    assert b1["logical_system_id"] == "baseline:B1"
    assert b0["system_kind"] == b1["system_kind"] == "baseline"
    assert neural["system_kind"] == "neural_policy"
    assert b0["logical_case_identity_count"] == b1["logical_case_identity_count"] == 204
    assert b0["b0_b1_equivalence_receipt_sha256"]
    assert b0["b0_b1_equivalence_receipt_sha256"] == b1["b0_b1_equivalence_receipt_sha256"]
    assert set(SYSTEM_METRIC_FIELDS).issubset(b0)


def test_historical_joint_scoring_benchmark_uses_frozen_ami_shape() -> None:
    row = historical_scoring_benchmark("ami_IS1008a", 11392)
    assert row["action_count"] == 11392
    assert row["reference_count"] == 132
    assert row["region_count"] == 338
    assert row["singleton_interval_count"] == 169
    assert row["overlap_interval_count"] == 64
    assert row["word_interval_count"] == 2504
    assert row["word_timing_observable"] is True
    assert row["raw_word_record_count"] == 2890
    assert len(row["word_annotation_files"]) == 4
    assert row["word_record_sha256"]
    assert row["elapsed_seconds"] > 0
    assert row["score_trace_sha256"]
