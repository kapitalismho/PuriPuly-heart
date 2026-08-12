from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

from .phase4_signal import atomic_write_json

AUTHORITY_SHA256 = "e3efdd9410a84bd343da5ba41d634ceec2d54626e1b512f41e410c0668329e36"
CURRENT_PROPOSAL_ROUTE_ROWS = 3512
HISTORICAL_PROPOSAL_ROUTE_ROWS = 816
PHYSICAL_PROPOSAL_EXECUTION_ROWS = 3824
PHYSICAL_SYSTEM_DEFINITION_ROWS = 2503
LOGICAL_SYSTEM_DEFINITION_ROWS = 4611
LOGICAL_ALIAS_EDGE_ROWS = 2108
CURRENT_SYSTEM_AGGREGATE_ROWS = 4611
HISTORICAL_NEURAL_SYSTEM_AGGREGATE_ROWS = 4608
HISTORICAL_BASELINE_SYSTEM_AGGREGATE_ROWS = 2
HISTORICAL_SYSTEM_AGGREGATE_ROWS = (
    HISTORICAL_NEURAL_SYSTEM_AGGREGATE_ROWS + HISTORICAL_BASELINE_SYSTEM_AGGREGATE_ROWS
)
HISTORICAL_CASE_COUNT = 204
FAILURE_EXAMPLE_ROWS = 420
INDEPENDENT_AUDIT_UNIT_ROWS = 2048
BLOCK_COUNT = 43
POOL_ORDER = ("diagnostic_dev", "frontier_dev", "natural_exposure_validation")
POOL_BLOCK_COUNTS = {
    "diagnostic_dev": 13,
    "frontier_dev": 10,
    "natural_exposure_validation": 20,
}
SAMPLE_ROWS = 4096
DETAIL_SHARD_LIMIT_BYTES = 20 * 1024**2
AGGREGATE_JSON_LIMIT_BYTES = 10 * 1024**2
RESULT_CEILING_BYTES = 8 * 1024**3
SYSTEM_METRIC_FIELDS = (
    "episode_count",
    "clean_gap_episode_count",
    "clean_gap_singleton_denominator_samples",
    "clean_gap_contaminated_samples",
    "mixed_turn_100ms_count",
    "mixed_turn_250ms_count",
    "mixed_turn_500ms_count",
    "clean_gap_hard_target_count",
    "hard_match_count",
    "hard_miss_count",
    "retained_b0_success_count",
    "recovered_b0_hard_miss_count",
    "accelerated_b0_success_count",
    "late_target_action_count",
    "detector_created_hard_action_count",
    "harmful_active_split_count_100ms",
    "harmful_active_split_count_200ms",
    "harmful_active_split_count_300ms",
    "lexical_split_count",
    "lexical_not_observable_count",
    "duplicate_hard_boundary_count",
    "same_speaker_pause_split_count",
    "same_speaker_extra_turn_count",
    "overlap_hard_action_count",
    "unscored_action_count",
    "fragments_lt_250ms_count",
    "fragments_lt_500ms_count",
    "fragments_lt_1000ms_count",
    "segment_duration_p10_samples",
    "segment_duration_p50_samples",
    "segment_duration_p90_samples",
    "active_speech_duration_p10_samples",
    "active_speech_duration_p50_samples",
    "active_speech_duration_p90_samples",
    "availability_delay_sum_samples",
    "availability_delay_count",
    "localization_error_sum_samples",
    "localization_error_count",
    "control_infeasible_count",
    "overlap_counterfactual_actual_samples_50ms",
    "overlap_counterfactual_suppressed_samples_50ms",
    "overlap_counterfactual_actual_samples_100ms",
    "overlap_counterfactual_suppressed_samples_100ms",
    "overlap_counterfactual_actual_samples_200ms",
    "overlap_counterfactual_suppressed_samples_200ms",
    "b0_b1_mismatch_count",
    "natural_contamination_numerator_samples",
    "natural_contamination_denominator_samples",
    "natural_harmful_active_split_count",
    "natural_same_speaker_extra_turn_count",
    "natural_sampled_source_samples",
    "natural_sampled_active_speech_samples",
    "natural_eligible_source_samples",
    "natural_session_count",
)
HISTORICAL_AGGREGATE_FIELDS = (
    "logical_system_id",
    "system_kind",
    "population_role",
    "case_count",
    "logical_case_identity_count",
    "ordered_identity_digest_sha256",
    "ordered_action_digest_sha256",
    "ordered_score_digest_sha256",
    "b0_b1_equivalence_receipt_sha256",
    *SYSTEM_METRIC_FIELDS,
)
BLOCK_METRIC_FIELDS = (
    "episode_count",
    "clean_gap_singleton_denominator_samples",
    "candidate_clean_gap_contaminated_samples",
    "b0_clean_gap_contaminated_samples",
    "b1_clean_gap_contaminated_samples",
    "candidate_harmful_active_split_count",
    "b0_harmful_active_split_count",
    "b1_harmful_active_split_count",
    "detector_created_hard_action_count",
    "same_speaker_extra_turn_count",
    "lexical_split_count",
    "lexical_observable_action_count",
    "duplicate_hard_boundary_count",
    "hard_target_count",
    "hard_match_250ms_count",
    "hard_match_500ms_count",
    "hard_match_1000ms_count",
    "hard_match_1500ms_count",
    "hard_match_2000ms_count",
    "availability_delay_sum_samples",
    "availability_delay_count",
    "overlap_hard_action_count",
    "overlap_contribution_samples",
    "sampled_source_samples",
    "sampled_active_speech_samples",
    "natural_exposure_eligible_source_samples",
)
FAILURE_CATEGORIES = (
    "contamination_regression",
    "contamination_improvement",
    "harmful_active_split",
    "duplicate_cluster",
    "late_accurate_target",
    "clean_gap_miss_strong_evidence",
    "overlap_hard_action",
)
PROPOSAL_EXECUTION_RECEIPT_FIELDS = (
    "proposal_execution_id",
    "proposal_profile_id",
    "source_or_case_id",
    "execution_mode",
    "probe_step_count",
    "input_window_universe_sha256",
    "input_cache_receipts_sha256",
    "lifecycle_trace_sha256",
    "proposal_code_sha256",
    "proposal_count",
    "proposal_trace_sha256",
    "progress_count",
    "progress_trace_sha256",
    "state_snapshot_index_sha256",
    "final_state_sha256",
    "tail_evidence_sha256",
    "status",
)
PROPOSAL_EVENT_FIELDS = (
    "proposal_id",
    "family",
    "checkpoint",
    "profile_id",
    "audio_epoch",
    "source_session_id",
    "proposal_kind",
    "boundary_source_sample",
    "observed_source_sample_at_emit",
    "emitted_monotonic_ns",
    "confidence",
    "confidence_semantics_id",
    "state_provenance",
    "debug_evidence",
)
PROGRESS_EVENT_FIELDS = (
    "audio_epoch",
    "observed_source_sample",
    "safe_boundary_frontier_sample",
)
CLUSTER_TRACE_FIELDS = (
    "cluster_id",
    "member_proposal_ids",
    "output_kind",
    "compatible_representative_subset",
    "representative_proposal_id",
    "representative_reason",
    "confidence_semantics_id",
    "cluster_open_frontier_sample",
    "cluster_close_frontier_sample",
    "boundary_source_sample",
    "observed_source_sample_at_emit",
    "tail_closed",
    "refractory_owner",
)
FUSION_ACTION_FIELDS = (
    "final_action_id",
    "origin",
    "action_kind",
    "boundary_source_sample",
    "observed_source_sample_at_emit",
    "source_session_id",
    "audio_epoch",
    "cluster_id",
    "associated_vad_action_id",
    "silence_interval_id",
    "association_forbidden_reason",
)
SCORE_TRACE_FIELDS = (
    "logical_system_id",
    "episode_or_case_id",
    "pool",
    "ordered_match_digest_sha256",
    "benefit_attribution_digest_sha256",
    "harm_or_structure_digest_sha256",
    "contamination_digest_sha256",
    "timing_digest_sha256",
    "metric_vector_sha256",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def digest(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def number(seed: str, modulus: int) -> int:
    return int(digest(seed)[:16], 16) % modulus


def metric_values(seed: str, fields: tuple[str, ...]) -> list[int]:
    return [number(f"{seed}|{field}", 100_000_000) for field in fields]


def proposal_execution_row(index: int) -> dict[str, Any]:
    return {
        "proposal_execution_id": f"proposal-execution:{digest(f'pe|{index}')}",
        "proposal_profile_id": f"profile:{digest(f'p|{index}')}",
        "source_or_case_id": f"source:{digest(f'source|{index}')[:24]}",
        "execution_mode": "source_prefix" if index % 3 else "episode_reset",
        "probe_step_count": number(f"steps|{index}", 32697),
        "input_window_universe_sha256": digest(f"windows|{index}"),
        "input_cache_receipts_sha256": digest(f"cache-receipts|{index}"),
        "lifecycle_trace_sha256": digest(f"lifecycle|{index}"),
        "proposal_code_sha256": digest("phase5-proposal-code"),
        "proposal_count": number(f"proposal-count|{index}", 11393),
        "proposal_trace_sha256": digest(f"proposal|{index}"),
        "progress_count": number(f"progress-count|{index}", 32698),
        "progress_trace_sha256": digest(f"progress|{index}"),
        "state_snapshot_index_sha256": digest(f"snapshots|{index}"),
        "final_state_sha256": digest(f"state|{index}"),
        "tail_evidence_sha256": digest(f"tail|{index}"),
        "status": "complete",
    }


def proposal_route_row(index: int) -> list[Any]:
    return [
        f"profile:{digest(f'p|{index}')}",
        f"episode:{digest(f'e|{index}')}",
        f"proposal-execution:{digest(f'pe|{index}')}",
        number(f"pc|{index}", 11393),
        digest(f"pt|{index}"),
        number(f"prc|{index}", 11394),
        digest(f"prt|{index}"),
        bool(number(f"tail|{index}", 2)),
    ]


def physical_system_definition_row(index: int) -> list[Any]:
    return [
        f"node:{digest(f'node|{index}')}",
        f"profile:{digest(f'p|{index}')}",
        number(f"stage|{index}", 6),
        number(f"d|{index}", 3),
        number(f"w|{index}", 2),
        number(f"r|{index}", 3),
        number(f"rep|{index}", 2),
        number(f"v|{index}", 2),
        number(f"silence|{index}", 2),
        number(f"control|{index}", 4),
    ]


def logical_system_definition_row(index: int) -> list[Any]:
    return [
        f"system:{digest(f'system|{index}')}",
        f"node:{digest(f'node|{index}')}",
        f"profile:{digest(f'p|{index}')}",
        number(f"stage|{index}", 8),
        digest(f"logical-key|{index}"),
    ]


def alias_edge_row(index: int) -> list[str]:
    return [
        f"system:{digest(f'system|{index}')}",
        f"node:{digest(f'node|{index}')}",
        digest(f"alias-reason|{index}"),
    ]


def pool_values(index: int, pool: str) -> list[Any]:
    values = metric_values(f"pool|{index}|{pool}", SYSTEM_METRIC_FIELDS)
    if pool != "natural_exposure_validation":
        for field_index, field in enumerate(SYSTEM_METRIC_FIELDS):
            if field.startswith("natural_"):
                values[field_index] = 0
    return [
        pool,
        digest(f"pool-action|{index}|{pool}"),
        digest(f"pool-score|{index}|{pool}"),
        *values,
    ]


def block_values(index: int, block: int) -> list[Any]:
    cursor = 0
    for pool in POOL_ORDER:
        count = POOL_BLOCK_COUNTS[pool]
        if block < cursor + count:
            ordinal = block - cursor
            return [
                pool,
                f"{pool}:block:{ordinal:02d}",
                *metric_values(f"block|{index}|{pool}|{ordinal}", BLOCK_METRIC_FIELDS),
            ]
        cursor += count
    raise ValueError("pool block index exceeds frozen universe")


def current_system_aggregate_row(index: int) -> list[Any]:
    return [
        f"system:{digest(f'system|{index}')}",
        f"node:{digest(f'node|{index}')}",
        [pool_values(index, pool) for pool in POOL_ORDER],
        [block_values(index, block) for block in range(BLOCK_COUNT)],
    ]


def historical_system_aggregate_row(index: int) -> list[Any]:
    baseline_id = ("B0", "B1")[index] if index < HISTORICAL_BASELINE_SYSTEM_AGGREGATE_ROWS else None
    system_key = baseline_id or f"neural|{index - HISTORICAL_BASELINE_SYSTEM_AGGREGATE_ROWS}"
    return [
        f"baseline:{baseline_id}" if baseline_id is not None else f"system:{digest(system_key)}",
        "baseline" if baseline_id is not None else "neural_policy",
        "historical_validation_corrected_rescore_only",
        HISTORICAL_CASE_COUNT,
        HISTORICAL_CASE_COUNT,
        digest(f"historical-identity|{system_key}"),
        digest(f"historical-action|{system_key}"),
        digest(f"historical-score|{system_key}"),
        digest("historical-b0-b1-equivalence|204-cases"),
        *metric_values(f"historical-metric|{system_key}", SYSTEM_METRIC_FIELDS),
    ]


def failure_example_row(index: int) -> dict[str, Any]:
    return {
        "proposal_profile_id": f"profile:{digest(f'p|{index}')}",
        "corpus": ("ami", "alimeeting", "synthetic")[index % 3],
        "category": FAILURE_CATEGORIES[index % len(FAILURE_CATEGORIES)],
        "rank": index % 5,
        "source_session_id": f"source:{digest(f'source|{index}')[:24]}",
        "episode_id": f"episode:{digest(f'episode|{index}')[:24]}",
        "boundary_source_sample": number(f"boundary|{index}", 60_000_000),
        "action_trace_sha256": digest(f"action|{index}"),
        "score_evidence_sha256": digest(f"score-evidence|{index}"),
    }


def independent_audit_unit_row(index: int) -> dict[str, Any]:
    return {
        "audit_unit_id": f"audit:{digest(f'audit|{index}')}",
        "canonical_unit_id": f"unit:{digest(f'unit|{index}')}",
        "selection_sha256": digest(
            f"turn-episode-v1-phase5-audit-v1|unit:{digest(f'unit|{index}')}"
        ),
        "selection_reason": "stratified_hash_fill",
        "population": "current" if index % 5 else "historical_validation",
        "proposal_profile_id": f"profile:{index % 4}",
        "policy_class": "adjacent" if index % 2 else "prototype_memory_4",
        "pool": POOL_ORDER[index % len(POOL_ORDER)],
        "corpus": ("ami", "alimeeting", "synthetic")[index % 3],
        "ladder_stage": ("naive", "cluster", "refractory", "vad", "full")[index % 5],
        "fusion_mode": index % 4,
        "control_kind": (
            None
            if index % 4
            else (
                "uniform_vad_active",
                "causal_energy_change_peak",
                "within_vad_active_position_shuffle",
            )[index % 3]
        ),
        "raw_trace_sha256": digest(f"raw-audit|{index}"),
        "derived_trace_sha256": digest(f"derived-audit|{index}"),
        "status": "pending_recompute",
    }


def encoded_sample(factory: Callable[[int], Any]) -> bytes:
    return b"".join(
        (canonical_json(factory(index)) + "\n").encode("utf-8") for index in range(SAMPLE_ROWS)
    )


def projection(
    factory: Callable[[int], Any], expected_rows: int, field_order: list[str] | None = None
) -> dict[str, Any]:
    encode_started = time.perf_counter()
    plain = encoded_sample(factory)
    encode_seconds = time.perf_counter() - encode_started
    compress_started = time.perf_counter()
    compressed = gzip.compress(plain, compresslevel=9, mtime=0)
    compress_seconds = time.perf_counter() - compress_started
    hash_started = time.perf_counter()
    compressed_sha256 = hashlib.sha256(compressed).hexdigest()
    hash_seconds = time.perf_counter() - hash_started
    verify_started = time.perf_counter()
    restored = gzip.decompress(compressed)
    verification_passed = (
        restored == plain and hashlib.sha256(compressed).hexdigest() == compressed_sha256
    )
    verify_seconds = time.perf_counter() - verify_started
    bytes_per_row = len(compressed) / SAMPLE_ROWS
    rows_per_shard = max(1, int(DETAIL_SHARD_LIMIT_BYTES // bytes_per_row))
    shard_count = math.ceil(expected_rows / rows_per_shard)
    projected_compressed = bytes_per_row * expected_rows + shard_count * 64
    return {
        "sample_row_count": SAMPLE_ROWS,
        "sample_plain_bytes": len(plain),
        "sample_gzip_bytes": len(compressed),
        "expected_row_count": expected_rows,
        "projected_plain_bytes": len(plain) / SAMPLE_ROWS * expected_rows,
        "projected_gzip_bytes": projected_compressed,
        "compression_ratio": len(compressed) / len(plain),
        "rows_per_detail_shard": rows_per_shard,
        "projected_detail_shard_count": shard_count,
        "field_order": field_order,
        "serialization_benchmark": {
            "encode_seconds": encode_seconds,
            "compress_seconds": compress_seconds,
            "sha256_seconds": hash_seconds,
            "decompress_and_hash_verify_seconds": verify_seconds,
            "verification_passed": verification_passed,
            "plain_bytes_per_second": len(plain)
            / max(encode_seconds + compress_seconds + hash_seconds + verify_seconds, 1e-9),
        },
    }


def run() -> dict[str, Any]:
    representations = {
        "physical_proposal_execution_receipt": projection(
            proposal_execution_row,
            PHYSICAL_PROPOSAL_EXECUTION_ROWS,
            list(PROPOSAL_EXECUTION_RECEIPT_FIELDS),
        ),
        "logical_proposal_route_index": projection(
            proposal_route_row,
            CURRENT_PROPOSAL_ROUTE_ROWS + HISTORICAL_PROPOSAL_ROUTE_ROWS,
            [
                "proposal_profile_id",
                "episode_or_case_id",
                "proposal_execution_id",
                "proposal_count",
                "proposal_trace_sha256",
                "progress_count",
                "progress_trace_sha256",
                "tail_pending",
            ],
        ),
        "physical_system_definition": projection(
            physical_system_definition_row, PHYSICAL_SYSTEM_DEFINITION_ROWS
        ),
        "logical_system_definition": projection(
            logical_system_definition_row, LOGICAL_SYSTEM_DEFINITION_ROWS
        ),
        "logical_alias_edge": projection(alias_edge_row, LOGICAL_ALIAS_EDGE_ROWS),
        "current_system_block_aggregate": projection(
            current_system_aggregate_row,
            CURRENT_SYSTEM_AGGREGATE_ROWS,
            [
                "logical_system_id",
                "physical_node_id",
                "pool_metric_rows",
                "pool_block_metric_rows",
            ],
        ),
        "historical_corrected_system_aggregate": projection(
            historical_system_aggregate_row,
            HISTORICAL_SYSTEM_AGGREGATE_ROWS,
            list(HISTORICAL_AGGREGATE_FIELDS),
        ),
        "deterministic_failure_example": projection(failure_example_row, FAILURE_EXAMPLE_ROWS),
        "independent_audit_unit": projection(
            independent_audit_unit_row,
            INDEPENDENT_AUDIT_UNIT_ROWS,
            [
                "audit_unit_id",
                "canonical_unit_id",
                "selection_sha256",
                "selection_reason",
                "population",
                "proposal_profile_id",
                "policy_class",
                "pool",
                "corpus",
                "ladder_stage",
                "fusion_mode",
                "control_kind",
                "raw_trace_sha256",
                "derived_trace_sha256",
                "status",
            ],
        ),
    }
    projected = sum(float(row["projected_gzip_bytes"]) for row in representations.values())
    return {
        "schema_version": "turn_episode_phase5_storage_benchmark.v5",
        "authority_sha256": AUTHORITY_SHA256,
        "representation": representations,
        "shared_schema": {
            "system_metric_field_order": list(SYSTEM_METRIC_FIELDS),
            "historical_aggregate_field_order": list(HISTORICAL_AGGREGATE_FIELDS),
            "historical_baseline_contract": {
                "baseline_system_ids": ["B0", "B1"],
                "baseline_system_count": HISTORICAL_BASELINE_SYSTEM_AGGREGATE_ROWS,
                "case_count_per_baseline": HISTORICAL_CASE_COUNT,
                "baseline_case_identity_count": HISTORICAL_BASELINE_SYSTEM_AGGREGATE_ROWS
                * HISTORICAL_CASE_COUNT,
                "neural_system_count": HISTORICAL_NEURAL_SYSTEM_AGGREGATE_ROWS,
                "neural_case_identity_count": HISTORICAL_NEURAL_SYSTEM_AGGREGATE_ROWS
                * HISTORICAL_CASE_COUNT,
                "total_system_count": HISTORICAL_SYSTEM_AGGREGATE_ROWS,
                "total_case_identity_count": HISTORICAL_SYSTEM_AGGREGATE_ROWS
                * HISTORICAL_CASE_COUNT,
                "b0_b1_equivalence_receipt_required": True,
                "metric_fields": list(SYSTEM_METRIC_FIELDS),
            },
            "block_metric_field_order": list(BLOCK_METRIC_FIELDS),
            "block_count": BLOCK_COUNT,
            "pool_order": list(POOL_ORDER),
            "pool_block_counts": POOL_BLOCK_COUNTS,
            "pool_metric_row_field_order": [
                "pool",
                "ordered_action_digest_sha256",
                "ordered_score_digest_sha256",
                *SYSTEM_METRIC_FIELDS,
            ],
            "pool_block_metric_row_field_order": [
                "pool",
                "pool_block_id",
                *BLOCK_METRIC_FIELDS,
            ],
            "failure_categories": list(FAILURE_CATEGORIES),
            "failure_example_count_rule": "4 proposal profiles times 3 current-population corpora times 7 categories times 5 deterministic ranks",
            "independent_audit_sample_size": INDEPENDENT_AUDIT_UNIT_ROWS,
            "independent_audit_selection": "include mandatory sentinels and deterministic failure examples, cover every observed profile class/pool/corpus/ladder/fusion/control stratum, then fill to 2048 distinct physical units by ascending sha256('turn-episode-v1-phase5-audit-v1' || canonical_unit_id)",
            "audio_conservation_violation_rule": "accepted Phase 5 output count must be zero; on first violation execution stops and every violation row is retained outside accepted aggregates",
            "logical_episode_identity_rows": "not materialized; verifier expands canonical system definitions and proposal routes in the frozen sort order and checks rolling framed digests",
            "proposal_event_reconstruction_field_order": list(PROPOSAL_EVENT_FIELDS),
            "progress_event_reconstruction_field_order": list(PROGRESS_EVENT_FIELDS),
            "cluster_trace_reconstruction_field_order": list(CLUSTER_TRACE_FIELDS),
            "fusion_action_reconstruction_field_order": list(FUSION_ACTION_FIELDS),
            "score_trace_reconstruction_field_order": list(SCORE_TRACE_FIELDS),
            "trace_framing": "sha256 over repeated uint64be(canonical_row_byte_count) followed by canonical UTF-8 JSON row bytes in frozen order",
            "reconstruction_authority": {
                "persisted_evidence": "typed physical proposal execution receipts, logical proposal routes, physical and logical system definitions, alias edges, per-pool and per-pool-block aggregates, and ordered trace digests",
                "accepted_inputs": "exact Phase 4 embedding cache receipts and payloads, episode and historical case manifests, B0 lifecycle traces, profile definitions, and the reviewed code hashes bound by the Phase 5 design ledger",
                "proposal_and_progress_rule": "recompute every physical proposal state pass from accepted embeddings and coordinates; compare event count, progress count, state snapshots, final state, tail evidence, and framed trace digests",
                "derived_trace_rule": "cluster, refractory, fusion, control, match, harm, contamination, and timing rows are not persisted individually; recompute the frozen 2048-unit stratified sample plus mandatory sentinels and deterministic failure examples from accepted proposal/B0 evidence and reviewed pure functions",
                "acceptance_rule": "file and self hashes, identities, completeness, B0/B1 equivalence, causal/schema guards, pool/block aggregates, and summary arithmetic are exhaustive; sampled raw/derived reconstruction must have zero mismatch",
            },
        },
        "projected_result_bytes": projected,
        "result_ceiling_bytes": RESULT_CEILING_BYTES,
        "within_result_ceiling": projected <= RESULT_CEILING_BYTES,
        "detail_shard_limit_bytes": DETAIL_SHARD_LIMIT_BYTES,
        "aggregate_json_limit_bytes": AGGREGATE_JSON_LIMIT_BYTES,
        "single_large_json_forbidden": True,
        "untyped_fixed_reserve_bytes": 0,
        "hardware": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "generated_from": {"phase5_storage_benchmark.py": sha256_file(Path(__file__).resolve())},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or (
        Path(__file__).resolve().parents[1]
        / "results"
        / "turn_episode_v1"
        / "phase_5_storage_benchmark.json"
    )
    written = atomic_write_json(output, run())
    print(
        canonical_json(
            {
                "path": str(output),
                "content_sha256": written["content_sha256"],
                "projected_result_bytes": written["projected_result_bytes"],
                "within_result_ceiling": written["within_result_ceiling"],
            }
        )
    )


if __name__ == "__main__":
    main()
