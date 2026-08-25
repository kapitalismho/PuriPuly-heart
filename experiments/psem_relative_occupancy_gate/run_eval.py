from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.psem_relative_occupancy_gate.eval_access import validate_opened_eval_manifest
from experiments.psem_relative_occupancy_gate.evaluate import (
    aggregate_gate0_metrics,
    gate0_session_metrics,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    canonical_sha256,
    config,
    safe_output_path,
    sha256_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.model_decode import (
    CausalEnrollmentConfig,
    model_observations,
    oracle_anchor_mapping,
    posterior_cells,
    simulate_causal_session,
)
from experiments.psem_relative_occupancy_gate.model_evaluate import (
    annotate_causal_episodes,
    causal_primitive_records,
    causal_product_metrics,
    count_causal_opportunities,
    gate1_primitive_records,
    gate1_product_session,
    gt_reference_session,
    intervals_from_manifest,
    primitive_metrics,
    transition_timing,
)
from experiments.psem_relative_occupancy_gate.model_run_io import load_model_traces
from experiments.psem_relative_occupancy_gate.run_gate1 import (
    FAMILIES,
    _aggregate_product,
    _runtime_summary,
    _topology_slices,
)
from experiments.psem_relative_occupancy_gate.run_gate2 import _aggregate_causal_product


class EvalRunError(RuntimeError):
    pass


def _canonical_oracle_slots(
    reference: Any, cells: Any, slot_ids: tuple[str, ...]
) -> dict[str, int]:
    result = {}
    for episode in reference.episodes:
        try:
            result[episode.episode_id] = oracle_anchor_mapping(episode, cells, slot_ids).slot_index
        except ValueError:
            continue
    return result


def run(args: argparse.Namespace) -> None:
    cfg = config()
    manifest_path = Path(args.manifest).resolve()
    selection_path = Path(args.selection).resolve()
    access_path = Path(args.access_receipt).resolve()
    authorization_path = Path(args.eval_authorization).resolve()
    manifest, selection = validate_opened_eval_manifest(
        manifest_path=manifest_path,
        access_path=access_path,
        selection_path=selection_path,
        authorization_path=authorization_path,
    )
    rows_by_id = {str(value["source_id"]): value for value in manifest}
    receipt_paths = {
        "sortformer": Path(args.sortformer_receipt).resolve(),
        "lseend": Path(args.lseend_receipt).resolve(),
    }
    gt_enrollment_samples = int(cfg["gate0_enrollment_confirm_ms"]) * 16
    silence_samples = int(cfg["lifecycle_proxy_silence_reset_ms"]) * 16
    canonical_ms = int(cfg["gate0_enrollment_confirm_ms"])
    tolerance_samples = int(cfg["product_event_alignment_tolerance_ms"]) * 16
    thresholds = [float(value) for value in cfg["activity_thresholds"]]
    active_seconds = sum(
        sum(
            value.end_sample - value.start_sample
            for value in intervals_from_manifest(row)
            if value.active_speakers
        )
        / 16000.0
        for row in manifest
    )
    baseline_sessions = [
        gt_reference_session(
            row,
            replacement_confirmation_samples=int(row["scored_end_sample"]) + 1,
            enrollment_samples=gt_enrollment_samples,
            silence_reset_samples=silence_samples,
        )
        for row in manifest
    ]
    baseline_metrics = [gate0_session_metrics(value) for value in baseline_sessions]
    baseline = {
        "family": "none",
        "gate": "vad_only_no_speaker_cut",
        "replacement_confirm_ms": None,
        **aggregate_gate0_metrics(baseline_metrics, active_seconds),
    }
    gate0_rows = []
    for replacement_ms in cfg["replacement_confirm_ms"]:
        sessions = [
            gt_reference_session(
                row,
                replacement_confirmation_samples=int(replacement_ms) * 16,
                enrollment_samples=gt_enrollment_samples,
                silence_reset_samples=silence_samples,
            )
            for row in manifest
        ]
        metrics = [gate0_session_metrics(value) for value in sessions]
        gate0_rows.append(
            {
                "family": "perfect_gt_relative_occupancy",
                "gate": "gate0_oracle",
                "replacement_confirm_ms": int(replacement_ms),
                **aggregate_gate0_metrics(metrics, active_seconds),
            }
        )
    family_results: dict[str, Any] = {}
    model_product_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    latency: dict[str, Any] = {}
    shared_trace_root: str | None = None
    for family, family_key in FAMILIES:
        selected = selection["selected_settings"][family]
        anchor_threshold = float(selected["anchor_threshold"])
        other_threshold = float(selected["other_threshold"])
        enrollment = selected["causal_enrollment"]
        enrollment_config = CausalEnrollmentConfig(
            float(enrollment["active_threshold"]),
            float(enrollment["other_low_threshold"]),
            int(enrollment["confirmation_samples"]),
        )
        receipt_identity, traces = load_model_traces(
            receipt_paths[family_key],
            manifest_path=manifest_path,
            manifest=manifest,
            family=family,
            role="PSEM-STRATEGY-EVAL",
            eval_access_path=access_path,
        )
        if shared_trace_root is None:
            shared_trace_root = str(receipt_identity["trace_root"])
        elif receipt_identity["trace_root"] != shared_trace_root:
            raise EvalRunError("model families do not share one frozen trace root")
        cells_by_source = {}
        observations_by_source = {}
        canonical_references = {}
        canonical_oracle_slots = {}
        gate1_records = []
        mapping_rows = []
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
                replacement_confirmation_samples=canonical_ms * 16,
                enrollment_samples=gt_enrollment_samples,
                silence_reset_samples=silence_samples,
            )
            records, mappings = gate1_primitive_records(
                source_id=source_id,
                episodes=reference.episodes,
                cells=cells,
                slot_ids=traces[source_id].slot_ids,
            )
            gate1_records.extend(records)
            mapping_rows.extend(value.to_dict() for value in mappings)
            cells_by_source[source_id] = cells
            observations_by_source[source_id] = observations
            canonical_references[source_id] = reference
            canonical_oracle_slots[source_id] = _canonical_oracle_slots(
                reference, cells, traces[source_id].slot_ids
            )
        gate1_primitive = primitive_metrics(
            gate1_records,
            thresholds,
            (anchor_threshold, other_threshold),
        )
        gate1_products = []
        gate2_products = []
        gate2_primitive_records = []
        for replacement_ms in cfg["replacement_confirm_ms"]:
            replacement_samples = int(replacement_ms) * 16
            gate1_session_metrics = []
            gate2_session_metrics = []
            gate1_events = {}
            gate2_events = {}
            references = {}
            for source_id in sorted(rows_by_id):
                row = rows_by_id[source_id]
                intervals = intervals_from_manifest(row)
                reference = gt_reference_session(
                    row,
                    replacement_confirmation_samples=replacement_samples,
                    enrollment_samples=gt_enrollment_samples,
                    silence_reset_samples=silence_samples,
                )
                gate1_metrics, _, oracle_events = gate1_product_session(
                    source_id=source_id,
                    reference=reference,
                    cells=cells_by_source[source_id],
                    slot_ids=traces[source_id].slot_ids,
                    observations=observations_by_source[source_id],
                    intervals=intervals,
                    anchor_threshold=anchor_threshold,
                    other_threshold=other_threshold,
                    replacement_confirmation_samples=replacement_samples,
                    tolerance_samples=tolerance_samples,
                )
                causal_session = simulate_causal_session(
                    source_id=source_id,
                    slot_ids=traces[source_id].slot_ids,
                    observations=observations_by_source[source_id],
                    enrollment_config=enrollment_config,
                    replacement_confirmation_samples=replacement_samples,
                    anchor_threshold=anchor_threshold,
                    other_threshold=other_threshold,
                    silence_reset_samples=silence_samples,
                    scored_end_sample=int(row["scored_end_sample"]),
                )
                annotations = annotate_causal_episodes(
                    session=causal_session,
                    intervals=intervals,
                    cells=cells_by_source[source_id],
                    slot_ids=traces[source_id].slot_ids,
                    scored_start_sample=int(row["scored_start_sample"]),
                    gt_enrollment_samples=gt_enrollment_samples,
                    silence_reset_samples=silence_samples,
                    oracle_reference=canonical_references[source_id],
                    oracle_slots=canonical_oracle_slots[source_id],
                )
                expected_count = count_causal_opportunities(
                    session=causal_session,
                    intervals=intervals,
                    scored_start_sample=int(row["scored_start_sample"]),
                    scored_end_sample=int(row["scored_end_sample"]),
                    enrollment_samples=gt_enrollment_samples,
                    silence_reset_samples=silence_samples,
                )
                causal_metrics = causal_product_metrics(
                    session=causal_session,
                    annotated=annotations,
                    reference=reference,
                    intervals=intervals,
                    tolerance_samples=tolerance_samples,
                    expected_opportunity_count=expected_count,
                )
                if int(replacement_ms) == canonical_ms:
                    gate2_primitive_records.extend(
                        causal_primitive_records(
                            source_id=source_id,
                            annotated=annotations,
                            cells=cells_by_source[source_id],
                        )
                    )
                gate1_session_metrics.append(gate1_metrics)
                gate2_session_metrics.append(causal_metrics)
                gate1_events[source_id] = oracle_events
                gate2_events[source_id] = causal_session.replacement_events
                references[source_id] = reference
            gate1_product = {
                "family": family,
                "gate": "gate1_oracle_anchor",
                "replacement_confirm_ms": int(replacement_ms),
                "anchor_threshold": anchor_threshold,
                "other_threshold": other_threshold,
                **_aggregate_product(gate1_session_metrics),
            }
            gate2_product = {
                "family": family,
                "gate": "gate2_causal_anchor",
                "replacement_confirm_ms": int(replacement_ms),
                "anchor_threshold": anchor_threshold,
                "other_threshold": other_threshold,
                "causal_enrollment": enrollment,
                **_aggregate_causal_product(gate2_session_metrics),
            }
            gate1_products.append(gate1_product)
            gate2_products.append(gate2_product)
            model_product_rows.extend((gate1_product, gate2_product))
            topology_rows.extend(
                (
                    {
                        "family": family,
                        "gate": "gate1_oracle_anchor",
                        "replacement_confirm_ms": int(replacement_ms),
                        "slices": _topology_slices(
                            manifest, gate1_events, references, tolerance_samples
                        ),
                    },
                    {
                        "family": family,
                        "gate": "gate2_causal_anchor",
                        "replacement_confirm_ms": int(replacement_ms),
                        "slices": _topology_slices(
                            manifest, gate2_events, references, tolerance_samples
                        ),
                    },
                )
            )
        gate2_primitive = primitive_metrics(
            gate2_primitive_records,
            thresholds,
            (anchor_threshold, other_threshold),
        )
        family_results[family] = {
            "model_receipt": receipt_identity,
            "selected_settings": selected,
            "oracle_mapping_count": len(mapping_rows),
            "oracle_mappings_sha256": canonical_sha256(mapping_rows),
            "gate1": {
                "primitive": gate1_primitive,
                "transition_timing": transition_timing(
                    gate1_records, anchor_threshold, other_threshold
                ),
                "product_frontier": gate1_products,
            },
            "gate2": {
                "primitive": gate2_primitive,
                "transition_timing": transition_timing(
                    gate2_primitive_records, anchor_threshold, other_threshold
                ),
                "product_frontier": gate2_products,
            },
        }
        latency[family] = {
            "native_frame_ms": cfg[family_key]["native_frame_ms"],
            "algorithmic_buffering_ms": (
                cfg[family_key].get("algorithmic_lookahead_ms") if family_key == "sortformer" else 0
            ),
            "selected_enrollment_confirmation_ms": enrollment["confirmation_samples"] / 16,
            "replacement_confirmation_ms": list(cfg["replacement_confirm_ms"]),
            "runtime": _runtime_summary(receipt_paths[family_key], family_key),
            "causal_enrollment_delay_ms": gate2_products[0]["anchor_enrollment_delay_ms"],
            "fraction_enrolled_within_1000ms": gate2_products[0]["fraction_enrolled_within_1000ms"],
            "fraction_enrolled_within_1500ms": gate2_products[0]["fraction_enrolled_within_1500ms"],
        }
    common = {
        "role": "PSEM-STRATEGY-EVAL",
        "eval_status": "opened_once",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_path": str(selection_path),
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": sha256_file(selection_path),
        "eval_access_receipt_sha256": sha256_file(Path(args.access_receipt)),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    metrics = {
        "schema_version": "psem.relative_occupancy.eval_metrics.v1",
        **common,
        "families": family_results,
    }
    product = {
        "schema_version": "psem.relative_occupancy.product_frontiers.v1",
        **common,
        "rows": [baseline, *gate0_rows, *model_product_rows],
    }
    topology = {
        "schema_version": "psem.relative_occupancy.topology_slices.v1",
        **common,
        "rows": topology_rows,
    }
    latency_result = {
        "schema_version": "psem.relative_occupancy.latency_breakdown.v2",
        **common,
        "families": latency,
    }
    outputs = {
        "metrics": safe_output_path(Path(args.output)),
        "product": safe_output_path(Path(args.product_output)),
        "topology": safe_output_path(Path(args.topology_output)),
        "latency": safe_output_path(Path(args.latency_output)),
    }
    write_json(outputs["metrics"], metrics)
    write_json(outputs["product"], product)
    write_json(outputs["topology"], topology)
    write_json(outputs["latency"], latency_result)
    print({key: str(value) for key, value in outputs.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--access-receipt", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--eval-authorization", required=True)
    parser.add_argument("--sortformer-receipt", required=True)
    parser.add_argument("--lseend-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--product-output", required=True)
    parser.add_argument("--topology-output", required=True)
    parser.add_argument("--latency-output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
