from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies import (
    AUTHORITY_SNAPSHOT_PATH,
    CONFIG_PATH,
    FAMILY_KEYS,
    PACKAGE_ROOT,
    RECEIPT_NAMES,
    _aggregate_product,
    _anchor_metrics,
    _causal_product_session,
    _config,
    _episode_anchor_records,
    _load_role_inputs,
    _posterior_inputs,
    _predecessor_result,
    _primary_candidate,
    _r0_causal_session,
    _result_dir,
    _role_name,
    _session_row,
    _session_topology,
    _sustained_dropout,
    _trace_by_source,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    canonical_sha256,
    load_json,
    sha256_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.model_decode import PosteriorCell
from experiments.psem_relative_occupancy_gate.model_evaluate import (
    gt_reference_session,
    intervals_from_manifest,
)


def _causal_records_with_coverage(
    row: dict[str, Any], cells: Sequence[PosteriorCell], gate2_row: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    records = []
    mapped = []
    unmapped = []
    continuity_invalid_episode_count = 0
    continuity_invalid_samples = 0
    continuity_invalid_unmasked_active_samples = 0
    for annotation in gate2_row["annotated_episodes"]:
        anchor_speaker = annotation.get("expected_anchor_speaker")
        slot_index = annotation.get("anchor_slot_index")
        if anchor_speaker is None or slot_index is None:
            unmapped.append(annotation)
            continue
        mapped.append(annotation)
        start = int(annotation["anchor_emit_sample"])
        end = int(annotation["end_emit_sample"])
        episode_records, invalid_samples, invalid_active_samples = _episode_anchor_records(
            source_id=str(row["source_id"]),
            episode_id=str(annotation["episode_id"]),
            anchor_speaker=str(anchor_speaker),
            anchor_slot_index=int(slot_index),
            episode_start=start,
            episode_end=end,
            cells=cells,
        )
        records.extend(episode_records)
        if invalid_samples:
            continuity_invalid_episode_count += 1
            continuity_invalid_samples += invalid_samples
            continuity_invalid_unmasked_active_samples += invalid_active_samples
    intervals = intervals_from_manifest(row)
    annotations = list(gate2_row["annotated_episodes"])
    total_samples = sum(
        int(value["end_emit_sample"]) - int(value["anchor_emit_sample"])
        for value in annotations
    )
    unmapped_samples = sum(
        int(value["end_emit_sample"]) - int(value["anchor_emit_sample"])
        for value in unmapped
    )
    unmapped_active_samples = sum(
        max(
            0,
            min(interval.end_sample, int(annotation["end_emit_sample"]))
            - max(interval.start_sample, int(annotation["anchor_emit_sample"])),
        )
        for annotation in unmapped
        for interval in intervals
        if not interval.masked and interval.active_speakers
    )
    valid_samples = sum(int(value.weight_samples) for value in records)
    coverage = {
        "source_id": str(row["source_id"]),
        "episode_count": len(annotations),
        "mapped_episode_count": len(mapped),
        "unmapped_episode_count": len(unmapped),
        "mapped_episode_fraction": len(mapped) / len(annotations) if annotations else None,
        "episode_support_seconds": total_samples / 16000.0,
        "valid_diagnostic_support_seconds": valid_samples / 16000.0,
        "unmapped_episode_support_seconds": unmapped_samples / 16000.0,
        "unmapped_unmasked_active_speech_seconds": unmapped_active_samples / 16000.0,
        "continuity_invalid_episode_count": continuity_invalid_episode_count,
        "continuity_invalid_support_seconds": continuity_invalid_samples / 16000.0,
        "continuity_invalid_unmasked_active_speech_seconds": (
            continuity_invalid_unmasked_active_samples / 16000.0
        ),
    }
    return records, coverage


def _aggregate_coverage(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    episodes = sum(int(value["episode_count"]) for value in rows)
    mapped = sum(int(value["mapped_episode_count"]) for value in rows)
    return {
        "source_count": len(rows),
        "episode_count": episodes,
        "mapped_episode_count": mapped,
        "unmapped_episode_count": sum(int(value["unmapped_episode_count"]) for value in rows),
        "mapped_episode_fraction": mapped / episodes if episodes else None,
        "episode_support_seconds": sum(float(value["episode_support_seconds"]) for value in rows),
        "valid_diagnostic_support_seconds": sum(
            float(value["valid_diagnostic_support_seconds"]) for value in rows
        ),
        "unmapped_episode_support_seconds": sum(
            float(value["unmapped_episode_support_seconds"]) for value in rows
        ),
        "unmapped_unmasked_active_speech_seconds": sum(
            float(value["unmapped_unmasked_active_speech_seconds"]) for value in rows
        ),
        "continuity_invalid_episode_count": sum(
            int(value["continuity_invalid_episode_count"]) for value in rows
        ),
        "continuity_invalid_support_seconds": sum(
            float(value["continuity_invalid_support_seconds"]) for value in rows
        ),
        "continuity_invalid_unmasked_active_speech_seconds": sum(
            float(value["continuity_invalid_unmasked_active_speech_seconds"]) for value in rows
        ),
        "per_source": list(rows),
    }


def _canonical_match(left: Any, right: Any) -> bool:
    return canonical_sha256(left) == canonical_sha256(right)


def _candidate_cells(cfg: dict[str, Any], family: str) -> list[tuple[str, str, float, float | None, bool]]:
    anchor_threshold = float(cfg["candidate_a"]["anchor_thresholds"][family][0])
    grid = cfg["candidate_b"]["threshold_grids"][family]
    cells: list[tuple[str, str, float, float | None, bool]] = [
        ("simple_anchor", "primary", anchor_threshold, None, False)
    ]
    cells.extend(
        (
            "anchor_overlap",
            "primary",
            float(candidate_anchor),
            float(candidate_overlap),
            False,
        )
        for candidate_anchor in grid["anchor"]
        for candidate_overlap in grid["anchor_overlap"]
    )
    cells.append(
        (
            "anchor_overlap",
            "strict_non_anchor",
            float(grid["primary"][0]),
            float(grid["primary"][1]),
            True,
        )
    )
    return cells


def run_s2_role(role: str) -> Path:
    cfg = _config()
    manifest, receipts, _, gate2_rows = _load_role_inputs(role)
    source_rows = {str(value["source_id"]): value for value in manifest}
    enrollment_samples = int(cfg["gate0_enrollment_confirm_ms"]) * 16
    silence_samples = int(cfg["lifecycle_proxy_silence_reset_ms"]) * 16
    tolerance_samples = int(cfg["product_event_alignment_tolerance_ms"]) * 16
    diagnostic_thresholds = [value / 100.0 for value in range(5, 100, 5)]
    diagnostics: dict[str, Any] = {}
    dropout: dict[str, Any] = {}
    session_rows: list[dict[str, Any]] = []
    for family in FAMILY_KEYS:
        receipt_by_source = _trace_by_source(receipts[family])
        prepared: dict[str, dict[str, Any]] = {}
        records_by_persistence: dict[int, list[Any]] = {
            int(value): [] for value in cfg["replacement_confirm_ms"]
        }
        coverage_by_persistence: dict[int, list[dict[str, Any]]] = {
            int(value): [] for value in cfg["replacement_confirm_ms"]
        }
        for source_id in sorted(source_rows):
            row = source_rows[source_id]
            cells, observations, _ = _posterior_inputs(row, receipt_by_source[source_id])
            for persistence in cfg["replacement_confirm_ms"]:
                persistence = int(persistence)
                records, coverage = _causal_records_with_coverage(
                    row,
                    cells,
                    gate2_rows[(source_id, family, persistence)],
                )
                records_by_persistence[persistence].extend(records)
                coverage_by_persistence[persistence].append(coverage)
            prepared[source_id] = {
                "row": row,
                "observations": observations,
            }
        anchor_threshold = float(cfg["candidate_a"]["anchor_thresholds"][family][0])
        diagnostics[family] = {}
        dropout[family] = {}
        for persistence, records in records_by_persistence.items():
            coverage = _aggregate_coverage(coverage_by_persistence[persistence])
            metrics = _anchor_metrics(records, anchor_threshold, diagnostic_thresholds)
            metrics["causal_anchor_lifecycle_coverage"] = coverage
            diagnostics[family][str(persistence)] = metrics
            dropout_metrics = _sustained_dropout(records, anchor_threshold)
            dropout_metrics["causal_anchor_lifecycle_coverage"] = coverage
            dropout[family][str(persistence)] = dropout_metrics
        for persistence_value in cfg["replacement_confirm_ms"]:
            persistence = int(persistence_value)
            confirmation_samples = persistence * 16
            for source_id in sorted(prepared):
                row = prepared[source_id]["row"]
                observations = prepared[source_id]["observations"]
                reference = gt_reference_session(
                    row,
                    replacement_confirmation_samples=confirmation_samples,
                    enrollment_samples=enrollment_samples,
                    silence_reset_samples=silence_samples,
                )
                gate2_row = gate2_rows[(source_id, family, persistence)]
                r0_metrics, r0_events = _r0_causal_session(
                    row=row,
                    reference=reference,
                    gate2_row=gate2_row,
                    tolerance_samples=tolerance_samples,
                )
                session_rows.append(
                    _session_row(
                        family=family,
                        arm="s2_fixed_issue97_lifecycle",
                        candidate="r0_relative_occupancy",
                        variant="primary",
                        anchor_threshold=float(gate2_row["anchor_threshold"]),
                        overlap_threshold=float(gate2_row["other_threshold"]),
                        persistence=persistence,
                        source_id=source_id,
                        metrics=r0_metrics,
                        topology=_session_topology(row, r0_events, reference, tolerance_samples),
                    )
                )
                for candidate, variant, candidate_anchor, candidate_overlap, strict in _candidate_cells(
                    cfg, family
                ):
                    metrics, events = _causal_product_session(
                        row=row,
                        reference=reference,
                        observations=observations,
                        gate2_row=gate2_row,
                        candidate=candidate,
                        anchor_threshold=candidate_anchor,
                        overlap_threshold=candidate_overlap,
                        strict_inconsistent=strict,
                        confirmation_samples=confirmation_samples,
                        tolerance_samples=tolerance_samples,
                    )
                    session_rows.append(
                        _session_row(
                            family=family,
                            arm="s2_fixed_issue97_lifecycle",
                            candidate=candidate,
                            variant=variant,
                            anchor_threshold=candidate_anchor,
                            overlap_threshold=candidate_overlap,
                            persistence=persistence,
                            source_id=source_id,
                            metrics=metrics,
                            topology=_session_topology(row, events, reference, tolerance_samples),
                        )
                    )
    keys = sorted(
        {
            (
                value["family"],
                value["arm"],
                value["candidate"],
                value["variant"],
                value["anchor_threshold"],
                value["overlap_threshold"],
                value["replacement_confirm_ms"],
            )
            for value in session_rows
        },
        key=str,
    )
    frontier = []
    for key in keys:
        rows = [
            value
            for value in session_rows
            if (
                value["family"],
                value["arm"],
                value["candidate"],
                value["variant"],
                value["anchor_threshold"],
                value["overlap_threshold"],
                value["replacement_confirm_ms"],
            )
            == key
        ]
        frontier.append(
            {
                "family": key[0],
                "arm": key[1],
                "candidate": key[2],
                "variant": key[3],
                "anchor_threshold": key[4],
                "overlap_threshold": key[5],
                "replacement_confirm_ms": key[6],
                **_aggregate_product(rows),
            }
        )
    result_dir = _result_dir(role)
    existing = [
        value
        for value in load_json(result_dir / "product_frontiers.json")["rows"]
        if value["arm"] == "s2_fixed_issue97_lifecycle"
    ]
    primary_frontier = [value for value in frontier if _primary_candidate(value, cfg)]
    output_path = result_dir / "s2_replay.json"
    existing_frontier_sha256 = canonical_sha256(existing)
    recomputed_frontier_sha256 = canonical_sha256(frontier)
    write_json(
        output_path,
        {
            "schema_version": "psem.ontology_simplification.s2_replay.v1",
            "role": _role_name(role),
            "arm": "s2_fixed_issue97_lifecycle",
            "interpretation": cfg["causal_arm"]["interpretation"],
            "authority_snapshot_sha256": sha256_file(AUTHORITY_SNAPSHOT_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "trace_reuse_receipt_sha256": sha256_file(PACKAGE_ROOT / "trace_reuse_receipt.json"),
            "causal_dependency_audit_sha256": sha256_file(
                PACKAGE_ROOT / "causal_lifecycle_dependency_audit.json"
            ),
            "predecessor_manifest_sha256": sha256_file(
                _predecessor_result(role, "relative_occupancy_manifest.jsonl")
            ),
            "predecessor_gate2_ledger_sha256": sha256_file(
                _predecessor_result(role, "gate2_event_ledger.jsonl")
            ),
            "predecessor_model_receipt_sha256": {
                family: sha256_file(_predecessor_result(role, RECEIPT_NAMES[family]))
                for family in FAMILY_KEYS
            },
            "s2_runner_source_sha256": sha256_file(Path(__file__)),
            "new_model_inference_performed": False,
            "s0_recomputed": False,
            "s1_recomputed": False,
            "production_vad_recomputed": False,
            "source_count": len(manifest),
            "continuity_coverage": {
                family: {
                    persistence: diagnostics[family][persistence][
                        "causal_anchor_lifecycle_coverage"
                    ]
                    for persistence in diagnostics[family]
                }
                for family in FAMILY_KEYS
            },
            "anchor_diagnostics": diagnostics,
            "anchor_dropout_slices": dropout,
            "product_frontier": frontier,
            "primary_product_frontier": primary_frontier,
            "existing_s2_frontier_sha256": existing_frontier_sha256,
            "recomputed_s2_frontier_sha256": recomputed_frontier_sha256,
            "exact_existing_frontier_match": _canonical_match(existing, frontier),
        },
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("dev", "eval"))
    args = parser.parse_args()
    run_s2_role(args.role)


if __name__ == "__main__":
    main()
