from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    canonical_sha256,
    config,
    load_jsonl,
    percentile,
    safe_output_path,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.psem_relative_occupancy_gate.model_decode import (
    model_observations,
    posterior_cells,
)
from experiments.psem_relative_occupancy_gate.model_evaluate import (
    gate1_primitive_records,
    gate1_product_session,
    gt_reference_session,
    intervals_from_manifest,
    primitive_metrics,
    transition_timing,
)
from experiments.psem_relative_occupancy_gate.model_run_io import load_model_traces
from experiments.psem_relative_occupancy_gate.provenance import load_frozen_dataset

FAMILIES = (
    ("streaming_sortformer", "sortformer"),
    ("ls_eend", "lseend"),
)


class Gate1Error(RuntimeError):
    pass


def _validate_manifest(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    dataset = load_frozen_dataset()
    expected = sorted(dataset.source_ids("PSEM-STRATEGY-DEV"))
    observed = sorted(str(value.get("source_id", "")) for value in rows)
    if observed != expected or len(observed) != len(set(observed)):
        raise Gate1Error("Gate 1 requires the exact frozen V2 DEV source set")
    for row in rows:
        payload = dict(row)
        row_hash = payload.pop("row_sha256", None)
        if (
            row_hash != canonical_sha256(payload)
            or row.get("role") != "PSEM-STRATEGY-DEV"
            or row.get("eval_status") != "sealed"
            or row.get("eval_selection_sha256") is not None
            or row.get("config_sha256") != sha256_file(CONFIG_PATH)
        ):
            raise Gate1Error(f"Gate 1 manifest binding mismatch: {row.get('source_id')}")
    return rows


def _aggregate_product(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    active_seconds = sum(float(value["active_speech_seconds"]) for value in rows)
    active_hours = active_seconds / 3600.0
    contamination = sum(float(value["exclusive_other_contamination_seconds"]) for value in rows)
    contamination_upper = sum(
        float(value["exclusive_other_contamination_upper_bound_seconds"]) for value in rows
    )
    masked_seconds = sum(float(value["masked_seconds"]) for value in rows)
    masked_active = sum(float(value["masked_active_speech_seconds"]) for value in rows)
    unanchored_active = sum(float(value["unanchored_active_speech_seconds"]) for value in rows)
    uncertain_active = sum(
        float(value["anchor_uncertain_active_speech_seconds"]) for value in rows
    )
    fail_closed_unknown = sum(
        float(value["fail_closed_unknown_active_speech_seconds"]) for value in rows
    )
    logical_episode_contamination = sum(
        float(value["logical_episode_exclusive_other_contamination_seconds"]) for value in rows
    )
    cuts = sum(int(value["predicted_cut_count"]) for value in rows)
    references = sum(int(value["reference_replacement_count"]) for value in rows)
    matched = sum(int(value["matched_replacement_count"]) for value in rows)
    false_cuts = sum(int(value["false_cut_count"]) for value in rows)
    missed = sum(int(value["missed_replacement_count"]) for value in rows)
    emit = [float(item) for value in rows for item in value["replacement_emit_delay_values_ms"]]
    evidence = [float(item) for value in rows for item in value["model_evidence_delay_values_ms"]]
    boundaries = [
        float(item) for value in rows for item in value["backdated_boundary_error_values_ms"]
    ]
    contamination_per_replacement = [
        float(item)
        for value in rows
        for item in value["contamination_values_seconds_per_true_replacement"]
    ]
    return {
        "source_count": len(rows),
        "active_speech_hours": active_hours,
        "predicted_cut_count": cuts,
        "reference_replacement_count": references,
        "matched_replacement_count": matched,
        "false_cut_count": false_cuts,
        "missed_replacement_count": missed,
        "speaker_induced_cut_count_per_active_speech_hour": (
            cuts / active_hours if active_hours else None
        ),
        "exclusive_other_contamination_seconds": contamination,
        "exclusive_other_contamination_upper_bound_seconds": contamination_upper,
        "masked_seconds": masked_seconds,
        "masked_active_speech_seconds": masked_active,
        "unanchored_active_speech_seconds": unanchored_active,
        "anchor_uncertain_active_speech_seconds": uncertain_active,
        "fail_closed_unknown_active_speech_seconds": fail_closed_unknown,
        "logical_episode_exclusive_other_contamination_seconds": (logical_episode_contamination),
        "exclusive_other_contamination_seconds_per_active_speech_hour": (
            contamination / active_hours if active_hours else None
        ),
        "contamination_seconds_per_true_replacement": {
            "p50": percentile(contamination_per_replacement, 50),
            "p90": percentile(contamination_per_replacement, 90),
        },
        "replacement_emit_delay_ms": {
            "p50": percentile(emit, 50),
            "p90": percentile(emit, 90),
        },
        "model_evidence_delay_ms": {
            "p50": percentile(evidence, 50),
            "p90": percentile(evidence, 90),
        },
        "backdated_boundary_error_ms": {
            "p50": percentile(boundaries, 50),
            "p90": percentile(boundaries, 90),
        },
    }


def _topology_window(row: dict[str, Any], episode: dict[str, Any]) -> tuple[int, int] | None:
    transitions = {value["transition_id"]: value for value in row["transitions"]}
    episode_transitions = [
        transitions[value] for value in episode["transition_ids"] if value in transitions
    ]
    indices = [
        int(index)
        for transition in episode_transitions
        for index in (
            transition.get("from_interval_index"),
            transition.get("to_interval_index"),
        )
        if index is not None
    ]
    if not indices:
        return None
    intervals = row["intervals"]
    return (
        int(intervals[min(indices)]["start_sample"]),
        int(intervals[max(indices)]["end_sample"]),
    )


def _topology_slices(
    manifest: Sequence[dict[str, Any]],
    events: dict[str, Sequence[Any]],
    references: dict[str, Any],
    tolerance_samples: int,
) -> dict[str, Any]:
    counters: dict[str, dict[str, int]] = {}
    for row in manifest:
        source_id = str(row["source_id"])
        predicted = events[source_id]
        reference = references[source_id]
        for episode in row["topology_episodes"]:
            if not episode.get("coverage_gate_eligible", False):
                continue
            window = _topology_window(row, episode)
            if window is None:
                continue
            start, end = window
            topology = str(episode["primary_topology"])
            values = counters.setdefault(
                topology,
                {
                    "eligible_episode_count": 0,
                    "episodes_with_predicted_cut": 0,
                    "episodes_with_reference_replacement": 0,
                    "episodes_with_aligned_cut": 0,
                    "episodes_with_early_cut": 0,
                },
            )
            values["eligible_episode_count"] += 1
            predicted_in_window = [
                value for value in predicted if start <= value.boundary_source_sample < end
            ]
            references_in_window = [
                value for value in reference.events if start <= value.boundary_source_sample < end
            ]
            if predicted_in_window:
                values["episodes_with_predicted_cut"] += 1
            if references_in_window:
                values["episodes_with_reference_replacement"] += 1
            if any(
                0
                <= left.boundary_source_sample - right.boundary_source_sample
                <= tolerance_samples
                for left in predicted_in_window
                for right in references_in_window
            ):
                values["episodes_with_aligned_cut"] += 1
            if any(
                -tolerance_samples
                <= left.boundary_source_sample - right.boundary_source_sample
                < 0
                for left in predicted_in_window
                for right in references_in_window
            ):
                values["episodes_with_early_cut"] += 1
    result: dict[str, Any] = {}
    for topology, values in sorted(counters.items()):
        count = values["eligible_episode_count"]
        result[topology] = {
            **values,
            "cut_episode_rate": values["episodes_with_predicted_cut"] / count if count else None,
            "aligned_cut_episode_rate": values["episodes_with_aligned_cut"] / count
            if count
            else None,
            "overlap_return_preservation_rate": (
                1.0 - values["episodes_with_predicted_cut"] / count
                if topology == "overlap_return" and count
                else None
            ),
            "overlap_takeover_success_rate": (
                values["episodes_with_aligned_cut"] / count
                if topology == "overlap_takeover" and count
                else None
            ),
        }
    return result


def _runtime_summary(receipt_path: Path, family_key: str) -> dict[str, Any]:
    receipt = __import__("json").loads(receipt_path.read_text(encoding="utf-8"))
    wall_values: list[float] = []
    cpu_values: list[float] = []
    origin_counts: dict[str, int] = {}
    for source in receipt["source_receipts"]:
        if family_key == "sortformer":
            inference = source.get("inference", {})
            wall = inference.get("process_wall_seconds")
            if wall is not None:
                wall_values.append(float(wall))
            origin = str(inference.get("origin", "unknown"))
        else:
            runtime = source.get("runtime", {})
            wall = runtime.get("wall_seconds")
            cpu = runtime.get("cpu_seconds")
            if wall is not None:
                wall_values.append(float(wall))
            if cpu is not None:
                cpu_values.append(float(cpu))
            origin = str(source.get("inference_origin", "fresh_single_frozen_inference_pass"))
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
    audio_seconds = sum(
        (int(value["source_end_sample"]) - int(value["source_start_sample"])) / 16000.0
        for value in receipt["source_receipts"]
    )
    return {
        "source_count": receipt["source_count"],
        "audio_seconds": audio_seconds,
        "measured_process_wall_seconds": sum(wall_values),
        "measured_process_wall_rtf": sum(wall_values) / audio_seconds
        if wall_values and audio_seconds
        else None,
        "measured_cpu_seconds": sum(cpu_values) if cpu_values else None,
        "inference_origin_counts": dict(sorted(origin_counts.items())),
    }


def run(args: argparse.Namespace) -> None:
    cfg = config()
    manifest_path = Path(args.manifest).resolve()
    manifest = _validate_manifest(manifest_path)
    rows_by_id = {str(value["source_id"]): value for value in manifest}
    enrollment_samples = int(cfg["gate0_enrollment_confirm_ms"]) * 16
    silence_samples = int(cfg["lifecycle_proxy_silence_reset_ms"]) * 16
    reference_ms = int(cfg.get("oracle_mapping_reference_replacement_confirm_ms", 200))
    reference_samples = reference_ms * 16
    thresholds = [float(value) for value in cfg["activity_thresholds"]]
    tolerance_samples = int(cfg["product_event_alignment_tolerance_ms"]) * 16
    receipt_paths = {
        "sortformer": Path(args.sortformer_receipt).resolve(),
        "lseend": Path(args.lseend_receipt).resolve(),
    }
    gate1_families: dict[str, Any] = {}
    product_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    latency: dict[str, Any] = {}
    event_rows: list[dict[str, Any]] = []
    shared_trace_root: str | None = None
    for family, family_key in FAMILIES:
        receipt_identity, traces = load_model_traces(
            receipt_paths[family_key],
            manifest_path=manifest_path,
            manifest=manifest,
            family=family,
            role="PSEM-STRATEGY-DEV",
        )
        if shared_trace_root is None:
            shared_trace_root = str(receipt_identity["trace_root"])
        elif receipt_identity["trace_root"] != shared_trace_root:
            raise Gate1Error("model families do not share one frozen trace root")
        all_records = []
        mapping_receipts: list[dict[str, Any]] = []
        cells_by_source = {}
        observations_by_source = {}
        canonical_references = {}
        for source_id in sorted(rows_by_id):
            row = rows_by_id[source_id]
            intervals = intervals_from_manifest(row)
            cells = posterior_cells(
                traces[source_id],
                intervals,
                int(row["scored_start_sample"]),
                int(row["scored_end_sample"]),
            )
            observations = model_observations(cells, intervals)
            reference = gt_reference_session(
                row,
                replacement_confirmation_samples=reference_samples,
                enrollment_samples=enrollment_samples,
                silence_reset_samples=silence_samples,
            )
            records, mappings = gate1_primitive_records(
                source_id=source_id,
                episodes=reference.episodes,
                cells=cells,
                slot_ids=traces[source_id].slot_ids,
            )
            all_records.extend(records)
            mapping_receipts.extend(value.to_dict() for value in mappings)
            cells_by_source[source_id] = cells
            observations_by_source[source_id] = observations
            canonical_references[source_id] = reference
        primitive = primitive_metrics(all_records, thresholds)
        selected = primitive["selected_operating_point"]
        anchor_threshold = float(selected["anchor_threshold"])
        other_threshold = float(selected["other_threshold"])
        timing = transition_timing(all_records, anchor_threshold, other_threshold)
        family_products = []
        for replacement_ms in cfg["replacement_confirm_ms"]:
            replacement_samples = int(replacement_ms) * 16
            session_rows = []
            events_by_source = {}
            references_by_source = {}
            duration_mapping_count = 0
            for source_id in sorted(rows_by_id):
                row = rows_by_id[source_id]
                reference = gt_reference_session(
                    row,
                    replacement_confirmation_samples=replacement_samples,
                    enrollment_samples=enrollment_samples,
                    silence_reset_samples=silence_samples,
                )
                metrics, mappings, events = gate1_product_session(
                    source_id=source_id,
                    reference=reference,
                    cells=cells_by_source[source_id],
                    slot_ids=traces[source_id].slot_ids,
                    observations=observations_by_source[source_id],
                    intervals=intervals_from_manifest(row),
                    anchor_threshold=anchor_threshold,
                    other_threshold=other_threshold,
                    replacement_confirmation_samples=replacement_samples,
                    tolerance_samples=tolerance_samples,
                )
                session_rows.append(metrics)
                duration_mapping_count += len(mappings)
                events_by_source[source_id] = events
                references_by_source[source_id] = reference
                event_row = {
                    "schema_version": "psem.relative_occupancy.gate_event_session.v1",
                    "gate": "gate1_oracle_anchor",
                    "family": family,
                    "source_id": source_id,
                    "anchor_threshold": anchor_threshold,
                    "other_threshold": other_threshold,
                    "replacement_confirm_ms": int(replacement_ms),
                    "oracle_mappings": [value.to_dict() for value in mappings],
                    "events": [value.to_dict() for value in events],
                    "reference_events": [value.to_dict() for value in reference.events],
                    "fail_closed_exposure": {
                        key: metrics[key]
                        for key in (
                            "masked_seconds",
                            "masked_active_speech_seconds",
                            "unanchored_active_speech_seconds",
                            "anchor_uncertain_active_speech_seconds",
                            "fail_closed_unknown_active_speech_seconds",
                            "exclusive_other_contamination_upper_bound_seconds",
                        )
                    },
                }
                event_row["row_sha256"] = canonical_sha256(event_row)
                event_rows.append(event_row)
            aggregate = _aggregate_product(session_rows)
            product = {
                "family": family,
                "gate": "gate1_oracle_anchor",
                "anchor_threshold": anchor_threshold,
                "other_threshold": other_threshold,
                "replacement_confirm_ms": int(replacement_ms),
                "oracle_mapping_count": duration_mapping_count,
                **aggregate,
            }
            family_products.append(product)
            product_rows.append(product)
            topology_rows.append(
                {
                    "family": family,
                    "gate": "gate1_oracle_anchor",
                    "replacement_confirm_ms": int(replacement_ms),
                    "slices": _topology_slices(
                        manifest,
                        events_by_source,
                        references_by_source,
                        tolerance_samples,
                    ),
                }
            )
        gate1_families[family] = {
            "model_receipt": receipt_identity,
            "oracle_mapping_reference_replacement_confirm_ms": reference_ms,
            "oracle_mapping_count": len(mapping_receipts),
            "oracle_mappings_sha256": canonical_sha256(mapping_receipts),
            "primitive": primitive,
            "transition_timing": timing,
            "product_frontier": family_products,
        }
        latency[family] = {
            "native_frame_ms": cfg[family_key]["native_frame_ms"],
            "algorithmic_buffering_ms": (
                cfg[family_key].get("algorithmic_lookahead_ms") if family_key == "sortformer" else 0
            ),
            "replacement_confirmation_ms": list(cfg["replacement_confirm_ms"]),
            "runtime": _runtime_summary(receipt_paths[family_key], family_key),
        }
    event_output = safe_output_path(Path(args.event_output))
    outputs = {
        "gate1": safe_output_path(Path(args.output)),
        "product": safe_output_path(Path(args.product_output)),
        "topology": safe_output_path(Path(args.topology_output)),
        "latency": safe_output_path(Path(args.latency_output)),
        "events": event_output,
    }
    input_paths = {
        manifest_path,
        *(path.resolve() for path in receipt_paths.values()),
    }
    if len(set(outputs.values())) != len(outputs) or set(outputs.values()) & input_paths:
        raise Gate1Error("Gate 1 outputs must be distinct from all inputs")
    write_jsonl(event_output, event_rows)
    common = {
        "role": "PSEM-STRATEGY-DEV",
        "eval_status": "sealed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(CONFIG_PATH),
        "event_ledger_sha256": sha256_file(event_output),
    }
    gate1 = {
        "schema_version": "psem.relative_occupancy.gate1_metrics.v1",
        **common,
        "families": gate1_families,
    }
    product = {
        "schema_version": "psem.relative_occupancy.product_frontier.v1",
        **common,
        "gate": "gate1_oracle_anchor",
        "rows": product_rows,
    }
    topology = {
        "schema_version": "psem.relative_occupancy.topology_slices.v1",
        **common,
        "gate": "gate1_oracle_anchor",
        "rows": topology_rows,
    }
    latency_result = {
        "schema_version": "psem.relative_occupancy.latency_breakdown.v1",
        **common,
        "families": latency,
    }
    write_json(outputs["gate1"], gate1)
    write_json(outputs["product"], product)
    write_json(outputs["topology"], topology)
    write_json(outputs["latency"], latency_result)
    print(
        {
            "gate1": str(outputs["gate1"]),
            "product": str(outputs["product"]),
            "topology": str(outputs["topology"]),
            "latency": str(outputs["latency"]),
            "events": str(outputs["events"]),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sortformer-receipt", required=True)
    parser.add_argument("--lseend-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--product-output", required=True)
    parser.add_argument("--topology-output", required=True)
    parser.add_argument("--latency-output", required=True)
    parser.add_argument("--event-output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
