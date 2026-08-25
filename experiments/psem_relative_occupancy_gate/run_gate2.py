from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    PACKAGE_ROOT,
    canonical_sha256,
    config,
    load_json,
    percentile,
    safe_output_path,
    sha256_file,
    write_json,
    write_jsonl,
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
    annotate_causal_with_gt_reference,
    causal_anchor_metrics,
    causal_primitive_records,
    causal_product_metrics,
    count_causal_opportunities,
    expected_outer_opportunity_count,
    gt_reference_session,
    intervals_from_manifest,
    primitive_metrics,
    transition_timing,
)
from experiments.psem_relative_occupancy_gate.model_run_io import load_model_traces
from experiments.psem_relative_occupancy_gate.run_gate1 import (
    FAMILIES,
    _aggregate_product,
    _topology_slices,
    _validate_manifest,
)

SELECTION_BINDINGS = (
    "BASELINE.md",
    "ONTOLOGY.md",
    "EVALUATOR.md",
    "authorize_eval.py",
    "config.json",
    "contracts.py",
    "decoder.py",
    "derive_relative_occupancy.py",
    "eval_access.py",
    "evaluate.py",
    "io_utils.py",
    "model_decode.py",
    "model_evaluate.py",
    "model_run_io.py",
    "preflight.py",
    "provenance.py",
    "trace_io.py",
    "trace_runtime.py",
    "run_sortformer_trace.py",
    "run_lseend_trace.py",
    "run_gate1.py",
    "run_gate2.py",
    "run_gate0.py",
    "run_eval.py",
    "run_eval_traces.py",
    "verify_model_gates.py",
    "verify_gate0.py",
    "verify_eval.py",
)


class Gate2Error(RuntimeError):
    pass


def _aggregate_anchor_selection(annotations: Sequence[Any], expected_count: int) -> dict[str, Any]:
    return causal_anchor_metrics(annotations, expected_count)


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    p90 = row["metrics"]["enrollment_delay_ms"]["p90"]
    return (
        float(row["metrics"]["wrong_anchor_rate"]),
        -float(row["metrics"]["fraction_enrolled_within_1500ms"]),
        float(row["metrics"]["enrollment_failure_rate"]),
        float(p90) if p90 is not None else float("inf"),
        -float(row["config"]["active_threshold"]),
        float(row["config"]["other_low_threshold"]),
        float(row["config"]["confirmation_samples"]),
    )


def _aggregate_causal_product(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    product = _aggregate_product(rows)
    expected = sum(int(value["expected_opportunity_count"]) for value in rows)
    enrollments = sum(int(value["enrollment_count"]) for value in rows)
    total_enrollments = sum(int(value["total_enrollment_count"]) for value in rows)
    unmatched_enrollments = sum(int(value["unmatched_enrollment_count"]) for value in rows)
    wrong = sum(int(value["wrong_anchor_count"]) for value in rows)
    failures = sum(int(value["enrollment_failure_count"]) for value in rows)
    delays = [float(item) for value in rows for item in value["enrollment_delay_values_ms"]]
    uncertain_seconds = sum(float(value["anchor_uncertain_seconds"]) for value in rows)
    scored_seconds = sum(float(value["scored_seconds"]) for value in rows)
    slot_losses = sum(int(value["slot_loss_count"]) for value in rows)
    wrong_false_cuts = sum(int(value["false_cuts_after_wrong_anchor"]) for value in rows)
    cascade_values = [
        int(length)
        for value in rows
        for length, count in value["anchor_error_cascade_length"]["distribution"].items()
        for _ in range(int(count))
    ]
    return {
        **product,
        "expected_opportunity_count": expected,
        "enrollment_count": enrollments,
        "total_enrollment_count": total_enrollments,
        "unmatched_enrollment_count": unmatched_enrollments,
        "wrong_anchor_count": wrong,
        "wrong_anchor_rate": wrong / total_enrollments if total_enrollments else 1.0,
        "enrollment_failure_count": failures,
        "enrollment_failure_rate": failures / expected if expected else 0.0,
        "no_anchor_timeout_rate": failures / expected if expected else 0.0,
        "anchor_enrollment_delay_ms": {
            "p50": percentile(delays, 50),
            "p90": percentile(delays, 90),
        },
        "fraction_enrolled_within_1000ms": (
            sum(value <= 1000.0 for value in delays) / expected if expected else 0.0
        ),
        "fraction_enrolled_within_1500ms": (
            sum(value <= 1500.0 for value in delays) / expected if expected else 0.0
        ),
        "slot_loss_count": slot_losses,
        "slot_loss_rate": slot_losses / total_enrollments if total_enrollments else 0.0,
        "anchor_uncertain_seconds": uncertain_seconds,
        "anchor_uncertain_time_fraction": (
            uncertain_seconds / scored_seconds if scored_seconds else 0.0
        ),
        "false_cuts_after_wrong_anchor": wrong_false_cuts,
        "anchor_error_cascade_length": {
            "maximum": max(cascade_values, default=0),
            "p50": percentile(cascade_values, 50),
            "p90": percentile(cascade_values, 90),
            "distribution": {
                str(length): cascade_values.count(length) for length in sorted(set(cascade_values))
            },
        },
    }


def _canonical_oracle_slots(
    reference: Any, cells: Sequence[Any], slot_ids: Sequence[str]
) -> dict[str, int]:
    result = {}
    for episode in reference.episodes:
        try:
            result[episode.episode_id] = oracle_anchor_mapping(episode, cells, slot_ids).slot_index
        except ValueError:
            continue
    return result


def _gate1_family(gate1: dict[str, Any], family: str) -> dict[str, Any]:
    families = gate1.get("families")
    if not isinstance(families, dict) or family not in families:
        raise Gate2Error(f"Gate 1 family result is missing: {family}")
    value = families[family]
    if not isinstance(value, dict):
        raise Gate2Error(f"Gate 1 family result is invalid: {family}")
    return value


def _optional_delta(current: float | None, reference: float | None) -> float | None:
    if current is None or reference is None:
        return None
    return float(current) - float(reference)


def run(args: argparse.Namespace) -> None:
    cfg = config()
    manifest_path = Path(args.manifest).resolve()
    manifest = _validate_manifest(manifest_path)
    rows_by_id = {str(value["source_id"]): value for value in manifest}
    gate1_path = Path(args.gate1).resolve()
    gate1 = load_json(gate1_path)
    gate1_event_path = Path(args.gate1_events).resolve()
    gate0_path = Path(args.gate0).resolve()
    gate0 = load_json(gate0_path)
    gate0_verification_path = Path(args.gate0_verification).resolve()
    gate0_verification = load_json(gate0_verification_path)
    if (
        not isinstance(gate1, dict)
        or gate1.get("schema_version") != "psem.relative_occupancy.gate1_metrics.v1"
        or gate1.get("role") != "PSEM-STRATEGY-DEV"
        or gate1.get("eval_status") != "sealed"
        or gate1.get("manifest_sha256") != sha256_file(manifest_path)
        or gate1.get("config_sha256") != sha256_file(CONFIG_PATH)
        or not gate1_event_path.is_file()
        or gate1.get("event_ledger_sha256") != sha256_file(gate1_event_path)
    ):
        raise Gate2Error("Gate 1 result binding mismatch")
    if (
        not isinstance(gate0, dict)
        or gate0.get("schema_version") != "psem.relative_occupancy.gate0_metrics.v1"
        or gate0.get("passed") is not True
        or gate0.get("role") != "PSEM-STRATEGY-DEV"
        or gate0.get("eval_status") != "sealed"
        or gate0.get("manifest_sha256") != sha256_file(manifest_path)
        or gate0.get("config_sha256") != sha256_file(CONFIG_PATH)
    ):
        raise Gate2Error("Gate 0 result binding mismatch")
    if (
        not isinstance(gate0_verification, dict)
        or gate0_verification.get("schema_version")
        != "psem.relative_occupancy.gate0_verification.v1"
        or gate0_verification.get("passed") is not True
        or gate0_verification.get("eval_status") != "sealed"
        or gate0_verification.get("manifest_sha256") != sha256_file(manifest_path)
        or gate0_verification.get("metrics_sha256") != sha256_file(gate0_path)
        or gate0_verification.get("config_sha256") != sha256_file(CONFIG_PATH)
    ):
        raise Gate2Error("Gate 0 verification binding mismatch")
    receipt_paths = {
        "sortformer": Path(args.sortformer_receipt).resolve(),
        "lseend": Path(args.lseend_receipt).resolve(),
    }
    enrollment_grid = cfg["causal_enrollment"]
    activity_thresholds = [float(value) for value in cfg["activity_thresholds"]]
    gt_enrollment_samples = int(cfg["gate0_enrollment_confirm_ms"]) * 16
    silence_samples = int(cfg["lifecycle_proxy_silence_reset_ms"]) * 16
    tolerance_samples = int(cfg["product_event_alignment_tolerance_ms"]) * 16
    canonical_ms = int(
        cfg.get(
            "oracle_mapping_reference_replacement_confirm_ms",
            cfg["gate0_enrollment_confirm_ms"],
        )
    )
    family_results: dict[str, Any] = {}
    gate2_product_rows: list[dict[str, Any]] = []
    gate2_topology_rows: list[dict[str, Any]] = []
    selected_settings: dict[str, Any] = {}
    latency_updates: dict[str, Any] = {}
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
            raise Gate2Error("model families do not share one frozen trace root")
        gate1_family = _gate1_family(gate1, family)
        if gate1_family["model_receipt"] != receipt_identity:
            raise Gate2Error(f"Gate 1 and Gate 2 trace receipts differ: {family}")
        operating = gate1_family["primitive"]["selected_operating_point"]
        anchor_threshold = float(operating["anchor_threshold"])
        other_threshold = float(operating["other_threshold"])
        cells_by_source = {}
        observations_by_source = {}
        selection_reference_by_source = {}
        canonical_reference_by_source = {}
        selection_oracle_slots = {}
        canonical_oracle_slots = {}
        expected_outer_count = {}
        for source_id in sorted(rows_by_id):
            row = rows_by_id[source_id]
            intervals = intervals_from_manifest(row)
            cells = posterior_cells(
                traces[source_id],
                intervals,
                int(row["scored_start_sample"]),
                int(row["scored_end_sample"]),
            )
            cells_by_source[source_id] = cells
            observations_by_source[source_id] = model_observations(cells, intervals)
            selection_reference = gt_reference_session(
                row,
                replacement_confirmation_samples=int(row["scored_end_sample"]) + 1,
                enrollment_samples=gt_enrollment_samples,
                silence_reset_samples=silence_samples,
            )
            canonical_reference = gt_reference_session(
                row,
                replacement_confirmation_samples=canonical_ms * 16,
                enrollment_samples=gt_enrollment_samples,
                silence_reset_samples=silence_samples,
            )
            selection_reference_by_source[source_id] = selection_reference
            canonical_reference_by_source[source_id] = canonical_reference
            selection_oracle_slots[source_id] = _canonical_oracle_slots(
                selection_reference, cells, traces[source_id].slot_ids
            )
            canonical_oracle_slots[source_id] = _canonical_oracle_slots(
                canonical_reference, cells, traces[source_id].slot_ids
            )
            expected_outer_count[source_id] = expected_outer_opportunity_count(
                row,
                enrollment_samples=gt_enrollment_samples,
                silence_reset_samples=silence_samples,
            )
        selection_rows = []
        if (
            enrollment_grid.get("validity_rule")
            != "other_low_threshold < active_threshold"
        ):
            raise Gate2Error("causal enrollment grid validity rule is not frozen")
        for active_threshold in enrollment_grid["active_thresholds"]:
            for other_low_threshold in enrollment_grid["other_low_thresholds"]:
                if float(other_low_threshold) >= float(active_threshold):
                    continue
                for confirm_ms in enrollment_grid["confirm_ms"]:
                    enrollment_config = CausalEnrollmentConfig(
                        float(active_threshold),
                        float(other_low_threshold),
                        int(confirm_ms) * 16,
                    )
                    annotations = []
                    expected_count = 0
                    for source_id in sorted(rows_by_id):
                        row = rows_by_id[source_id]
                        session = simulate_causal_session(
                            source_id=source_id,
                            slot_ids=traces[source_id].slot_ids,
                            observations=observations_by_source[source_id],
                            enrollment_config=enrollment_config,
                            replacement_confirmation_samples=int(row["scored_end_sample"]) + 1,
                            anchor_threshold=anchor_threshold,
                            other_threshold=other_threshold,
                            silence_reset_samples=silence_samples,
                            scored_end_sample=int(row["scored_end_sample"]),
                        )
                        annotations.extend(
                            annotate_causal_with_gt_reference(
                                session=session,
                                gt_reference=selection_reference_by_source[source_id],
                                cells=cells_by_source[source_id],
                                slot_ids=traces[source_id].slot_ids,
                                gt_oracle_slots=selection_oracle_slots[source_id],
                            )
                        )
                        expected_count += expected_outer_count[source_id]
                    metrics = _aggregate_anchor_selection(annotations, expected_count)
                    selection_rows.append(
                        {
                            "config": enrollment_config.to_dict(),
                            "confirm_ms": int(confirm_ms),
                            "metrics": metrics,
                        }
                    )
        if len(selection_rows) != int(enrollment_grid["valid_candidate_count"]):
            raise Gate2Error("causal enrollment grid candidate count mismatch")
        selected = min(selection_rows, key=_selection_key)
        selected_config = CausalEnrollmentConfig(
            float(selected["config"]["active_threshold"]),
            float(selected["config"]["other_low_threshold"]),
            int(selected["config"]["confirmation_samples"]),
        )
        selected_settings[family] = {
            "anchor_threshold": anchor_threshold,
            "other_threshold": other_threshold,
            "causal_enrollment": selected["config"],
            "selection_order": enrollment_grid["selection_order"],
        }
        family_products = []
        causal_primitive = []
        canonical_annotations = []
        for replacement_ms in cfg["replacement_confirm_ms"]:
            replacement_samples = int(replacement_ms) * 16
            session_rows = []
            events_by_source = {}
            references_by_source = {}
            for source_id in sorted(rows_by_id):
                row = rows_by_id[source_id]
                intervals = intervals_from_manifest(row)
                session = simulate_causal_session(
                    source_id=source_id,
                    slot_ids=traces[source_id].slot_ids,
                    observations=observations_by_source[source_id],
                    enrollment_config=selected_config,
                    replacement_confirmation_samples=replacement_samples,
                    anchor_threshold=anchor_threshold,
                    other_threshold=other_threshold,
                    silence_reset_samples=silence_samples,
                    scored_end_sample=int(row["scored_end_sample"]),
                )
                reference = gt_reference_session(
                    row,
                    replacement_confirmation_samples=replacement_samples,
                    enrollment_samples=gt_enrollment_samples,
                    silence_reset_samples=silence_samples,
                )
                annotations = annotate_causal_episodes(
                    session=session,
                    intervals=intervals,
                    cells=cells_by_source[source_id],
                    slot_ids=traces[source_id].slot_ids,
                    scored_start_sample=int(row["scored_start_sample"]),
                    gt_enrollment_samples=gt_enrollment_samples,
                    silence_reset_samples=silence_samples,
                    oracle_reference=canonical_reference_by_source[source_id],
                    oracle_slots=canonical_oracle_slots[source_id],
                )
                expected_count = count_causal_opportunities(
                    session=session,
                    intervals=intervals,
                    scored_start_sample=int(row["scored_start_sample"]),
                    scored_end_sample=int(row["scored_end_sample"]),
                    enrollment_samples=gt_enrollment_samples,
                    silence_reset_samples=silence_samples,
                )
                metrics = causal_product_metrics(
                    session=session,
                    annotated=annotations,
                    reference=reference,
                    intervals=intervals,
                    tolerance_samples=tolerance_samples,
                    expected_opportunity_count=expected_count,
                )
                session_rows.append(metrics)
                event_row = {
                    "schema_version": "psem.relative_occupancy.gate_event_session.v1",
                    "gate": "gate2_causal_anchor",
                    "family": family,
                    "source_id": source_id,
                    "anchor_threshold": anchor_threshold,
                    "other_threshold": other_threshold,
                    "causal_enrollment": selected["config"],
                    "replacement_confirm_ms": int(replacement_ms),
                    "expected_opportunity_count": expected_count,
                    "enrollments": [value.to_dict() for value in session.enrollments],
                    "annotated_episodes": [value.to_dict() for value in annotations],
                    "timeline": [value.to_dict() for value in session.timeline],
                    "uncertain_entry_count": session.uncertain_entry_count,
                    "final_reset_count": session.final_reset_count,
                    "events": [value.to_dict() for value in session.replacement_events],
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
                events_by_source[source_id] = session.replacement_events
                references_by_source[source_id] = reference
                if int(replacement_ms) == canonical_ms:
                    canonical_annotations.extend(annotations)
                    causal_primitive.extend(
                        causal_primitive_records(
                            source_id=source_id,
                            annotated=annotations,
                            cells=cells_by_source[source_id],
                        )
                    )
            aggregate = _aggregate_causal_product(session_rows)
            product = {
                "family": family,
                "gate": "gate2_causal_anchor",
                "anchor_threshold": anchor_threshold,
                "other_threshold": other_threshold,
                "causal_enrollment": selected["config"],
                "replacement_confirm_ms": int(replacement_ms),
                **aggregate,
            }
            family_products.append(product)
            gate2_product_rows.append(product)
            gate2_topology_rows.append(
                {
                    "family": family,
                    "gate": "gate2_causal_anchor",
                    "replacement_confirm_ms": int(replacement_ms),
                    "slices": _topology_slices(
                        manifest,
                        events_by_source,
                        references_by_source,
                        tolerance_samples,
                    ),
                }
            )
        causal_primitive_metrics = primitive_metrics(
            causal_primitive,
            activity_thresholds,
            (anchor_threshold, other_threshold),
        )
        causal_timing = transition_timing(causal_primitive, anchor_threshold, other_threshold)
        gate1_primitive = gate1_family["primitive"]
        family_results[family] = {
            "model_receipt": receipt_identity,
            "selection_grid": selection_rows,
            "selected_configuration": selected,
            "primitive": causal_primitive_metrics,
            "transition_timing": causal_timing,
            "product_frontier": family_products,
            "oracle_to_causal_degradation": {
                "anchor_present_average_precision_delta": _optional_delta(
                    causal_primitive_metrics["anchor_present"]["average_precision"],
                    gate1_primitive["anchor_present"]["average_precision"],
                ),
                "other_present_average_precision_delta": _optional_delta(
                    causal_primitive_metrics["other_present"]["average_precision"],
                    gate1_primitive["other_present"]["average_precision"],
                ),
                "four_state_macro_f1_delta": (
                    causal_primitive_metrics["selected_operating_point"]["macro_f1"]
                    - gate1_primitive["selected_operating_point"]["macro_f1"]
                ),
            },
            "canonical_annotated_episode_count": len(canonical_annotations),
        }
        latency_updates[family] = {
            "selected_enrollment_confirmation_ms": selected["confirm_ms"],
            "selected_enrollment_active_threshold": selected["config"]["active_threshold"],
            "selected_enrollment_other_low_threshold": selected["config"]["other_low_threshold"],
            "causal_enrollment_delay_ms": family_products[0]["anchor_enrollment_delay_ms"],
            "fraction_enrolled_within_1000ms": family_products[0][
                "fraction_enrolled_within_1000ms"
            ],
            "fraction_enrolled_within_1500ms": family_products[0][
                "fraction_enrolled_within_1500ms"
            ],
        }
    event_output = safe_output_path(Path(args.event_output))
    outputs = {
        "gate2": safe_output_path(Path(args.output)),
        "product": safe_output_path(Path(args.product_output)),
        "topology": safe_output_path(Path(args.topology_output)),
        "latency": safe_output_path(Path(args.latency_output)),
        "selection": safe_output_path(Path(args.selection_output)),
        "events": event_output,
    }
    input_paths = {
        manifest_path,
        gate0_path,
        gate0_verification_path,
        gate1_path,
        gate1_event_path,
        Path(args.gate1_product).resolve(),
        Path(args.gate1_topology).resolve(),
        Path(args.latency).resolve(),
        *(path.resolve() for path in receipt_paths.values()),
    }
    if len(set(outputs.values())) != len(outputs) or set(outputs.values()) & input_paths:
        raise Gate2Error("Gate 2 outputs must be distinct from all inputs")
    write_jsonl(event_output, event_rows)
    common = {
        "role": "PSEM-STRATEGY-DEV",
        "eval_status": "sealed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(CONFIG_PATH),
        "gate0_sha256": sha256_file(gate0_path),
        "gate0_verification_sha256": sha256_file(gate0_verification_path),
        "gate1_sha256": sha256_file(gate1_path),
        "gate1_event_ledger_sha256": sha256_file(gate1_event_path),
        "event_ledger_sha256": sha256_file(event_output),
    }
    gate2 = {
        "schema_version": "psem.relative_occupancy.gate2_metrics.v1",
        **common,
        "families": family_results,
    }
    gate1_product = load_json(Path(args.gate1_product))
    gate1_topology = load_json(Path(args.gate1_topology))
    latency = load_json(Path(args.latency))
    supporting = (
        (gate1_product, "psem.relative_occupancy.product_frontier.v1", True),
        (gate1_topology, "psem.relative_occupancy.topology_slices.v1", True),
        (latency, "psem.relative_occupancy.latency_breakdown.v1", False),
    )
    if any(
        not isinstance(value, dict)
        or value.get("schema_version") != schema
        or value.get("role") != "PSEM-STRATEGY-DEV"
        or value.get("eval_status") != "sealed"
        or value.get("manifest_sha256") != common["manifest_sha256"]
        or value.get("config_sha256") != common["config_sha256"]
        or value.get("event_ledger_sha256") != common["gate1_event_ledger_sha256"]
        or (requires_gate and value.get("gate") != "gate1_oracle_anchor")
        for value, schema, requires_gate in supporting
    ):
        raise Gate2Error("Gate 1 supporting artifact binding mismatch")
    baseline = {
        "family": "none",
        "gate": "vad_only_no_speaker_cut",
        "replacement_confirm_ms": None,
        **gate0["no_speaker_cut_baseline"],
    }
    gate0_rows = [
        {
            "family": "perfect_gt_relative_occupancy",
            "gate": "gate0_oracle",
            "replacement_confirm_ms": int(value["confirmation_ms"]),
            **value["aggregate"],
        }
        for value in gate0["settings"]
    ]
    product = {
        "schema_version": "psem.relative_occupancy.product_frontiers.v1",
        **common,
        "rows": [baseline, *gate0_rows, *gate1_product["rows"], *gate2_product_rows],
    }
    topology = {
        "schema_version": "psem.relative_occupancy.topology_slices.v1",
        **common,
        "rows": [*gate1_topology["rows"], *gate2_topology_rows],
    }
    for family, update in latency_updates.items():
        latency["families"][family]["causal"] = update
    latency["schema_version"] = "psem.relative_occupancy.latency_breakdown.v2"
    latency["gate1_sha256"] = common["gate1_sha256"]
    write_json(outputs["gate2"], gate2)
    write_json(outputs["product"], product)
    write_json(outputs["topology"], topology)
    write_json(outputs["latency"], latency)
    artifact_bindings = {
        "gate0_metrics_sha256": sha256_file(gate0_path),
        "gate1_metrics_sha256": sha256_file(gate1_path),
        "gate1_product_frontier_sha256": sha256_file(Path(args.gate1_product)),
        "gate1_topology_slices_sha256": sha256_file(Path(args.gate1_topology)),
        "gate1_latency_breakdown_sha256": sha256_file(Path(args.latency)),
        "gate1_event_ledger_sha256": sha256_file(gate1_event_path),
        "gate2_event_ledger_sha256": sha256_file(event_output),
        "gate0_verification_sha256": sha256_file(gate0_verification_path),
        "gate2_metrics_sha256": sha256_file(outputs["gate2"]),
        "product_frontiers_sha256": sha256_file(outputs["product"]),
        "topology_slices_sha256": sha256_file(outputs["topology"]),
        "latency_breakdown_sha256": sha256_file(outputs["latency"]),
        "model_receipts": {
            family_key: sha256_file(receipt_paths[family_key]) for _, family_key in FAMILIES
        },
        "contract_files": {name: sha256_file(PACKAGE_ROOT / name) for name in SELECTION_BINDINGS},
    }
    selection_payload = {
        "schema_version": "psem.relative_occupancy.dev_selection.v1",
        "authority": cfg["authority"],
        **common,
        "eval_open_authorized": False,
        "eval_open_count": 0,
        "selected_settings": selected_settings,
        "selection_order": enrollment_grid["selection_order"],
        "causal_enrollment_grid": enrollment_grid,
        "same_cached_trace_for_gate1_and_gate2": True,
        "artifact_bindings": artifact_bindings,
    }
    selection_payload["selection_sha256"] = canonical_sha256(selection_payload)
    write_json(outputs["selection"], selection_payload)
    print(
        {
            "gate2": str(outputs["gate2"]),
            "product": str(outputs["product"]),
            "topology": str(outputs["topology"]),
            "latency": str(outputs["latency"]),
            "selection": str(outputs["selection"]),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sortformer-receipt", required=True)
    parser.add_argument("--lseend-receipt", required=True)
    parser.add_argument("--gate1", required=True)
    parser.add_argument("--gate1-events", required=True)
    parser.add_argument("--gate0", required=True)
    parser.add_argument("--gate0-verification", required=True)
    parser.add_argument("--gate1-product", required=True)
    parser.add_argument("--gate1-topology", required=True)
    parser.add_argument("--latency", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--product-output", required=True)
    parser.add_argument("--topology-output", required=True)
    parser.add_argument("--latency-output", required=True)
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--event-output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
