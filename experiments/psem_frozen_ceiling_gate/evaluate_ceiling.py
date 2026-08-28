from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
    PACKAGE_ROOT,
    SessionExamples,
    config,
)
from experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies import (
    _aggregate_topology,
    _session_topology,
)
from experiments.psem_relative_occupancy_gate.decoder import ReplacementEvent
from experiments.psem_relative_occupancy_gate.io_utils import (
    canonical_sha256,
    load_json,
    percentile,
    sha256_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.model_evaluate import (
    intervals_from_manifest,
    product_event_metrics,
)

RESULTS_ROOT = PACKAGE_ROOT / "results" / "frozen_ceiling_1"
HIDDEN_CONFIG_PATH = PACKAGE_ROOT / "hidden_config.json"
REPRESENTATION_RECEIPT_PATH = PACKAGE_ROOT / "hidden_representation_receipt.json"
SPLIT_PATH = PACKAGE_ROOT / "split_manifest.json"
CONFIG_PATH = PACKAGE_ROOT / "config.json"
MAPPING_PATH = PACKAGE_ROOT / "oracle_mapping_ledger.jsonl"
ACTION_REFERENCE_PATH = PACKAGE_ROOT / "action_reference_ledger.jsonl"
EXTRACTION_RECEIPT_PATH = RESULTS_ROOT / "hidden_extraction_receipt.json"
HIDDEN_SOURCE_RESULTS_PATH = RESULTS_ROOT / "hidden_source_family_results.json"
EXTRACTOR_PATH = PACKAGE_ROOT / "extract_hidden_features.py"


def _expected_hidden_provenance() -> dict[str, str]:
    return {
        "config_sha256": sha256_file(CONFIG_PATH),
        "hidden_config_sha256": sha256_file(HIDDEN_CONFIG_PATH),
        "split_manifest_sha256": sha256_file(SPLIT_PATH),
        "hidden_representation_receipt_sha256": sha256_file(REPRESENTATION_RECEIPT_PATH),
        "hidden_extraction_receipt_sha256": sha256_file(EXTRACTION_RECEIPT_PATH),
        "oracle_mapping_ledger_sha256": sha256_file(MAPPING_PATH),
        "action_reference_ledger_sha256": sha256_file(ACTION_REFERENCE_PATH),
    }


def _validate_hidden_evidence(hidden_paths: dict[str, Path]) -> bool:
    artifact_paths = [*hidden_paths.values(), HIDDEN_SOURCE_RESULTS_PATH, EXTRACTION_RECEIPT_PATH]
    present = [path.is_file() for path in artifact_paths]
    if not any(present):
        return False
    if not all(present):
        raise ValueError("hidden evidence set is partial")
    representation = load_json(REPRESENTATION_RECEIPT_PATH)
    trigger_paths = {
        "gt_result_sha256": RESULTS_ROOT / "gt_causal_action_frontier.json",
        "posterior_causal_result_sha256": RESULTS_ROOT / "fullslot_causal_metrics.json",
        "posterior_noncausal_result_sha256": RESULTS_ROOT / "fullslot_noncausal_metrics.json",
        "source_family_result_sha256": RESULTS_ROOT / "source_family_results.json",
    }
    if representation.get("opened") is not True or any(
        sha256_file(path) != representation["trigger"].get(field)
        for field, path in trigger_paths.items()
    ):
        raise ValueError("hidden trigger evidence differs")
    split = load_json(SPLIT_PATH)
    expected_sources = {str(value["source_id"]) for value in split["sources"]}
    expected_eval_sources = {
        str(value["source_id"])
        for value in split["sources"]
        if value["old_v2_role"] == config()["split"]["scoring_role"]
    }
    expected_families = {str(value["held_out_family"]) for value in split["folds"]}
    extraction = load_json(EXTRACTION_RECEIPT_PATH)
    extraction_contract = {
        "schema_version": "psem.hidden_ceiling.extraction_receipt.v1",
        "status": "complete",
        "representation_receipt_sha256": sha256_file(REPRESENTATION_RECEIPT_PATH),
        "hidden_config_sha256": sha256_file(HIDDEN_CONFIG_PATH),
        "split_manifest_sha256": sha256_file(SPLIT_PATH),
        "extractor_sha256": sha256_file(EXTRACTOR_PATH),
        "instrumented_bench_sha256": representation["runtime"]["instrumented_bench_sha256"],
        "model_sha256": representation["runtime"]["model_sha256"],
        "hidden_export_patch_sha256": representation["runtime"]["hidden_export_patch_sha256"],
        "source_count": len(expected_sources),
    }
    if any(extraction.get(key) != value for key, value in extraction_contract.items()):
        raise ValueError("hidden extraction receipt differs from the current contract")
    source_ids = []
    for value in extraction.get("source_receipts", []):
        source_id = str(value.get("source_id"))
        source_ids.append(source_id)
        receipt_path = Path(str(value.get("receipt_path", "")))
        if sha256_file(receipt_path) != value.get("receipt_sha256"):
            raise ValueError(f"hidden source receipt identity differs: {source_id}")
        receipt = load_json(receipt_path)
        contract = receipt.get("extraction_contract", {})
        feature_path = Path(str(receipt.get("hidden_features_path", "")))
        if (
            receipt.get("status") != "complete"
            or receipt.get("source_id") != source_id
            or receipt.get("extraction_contract_sha256") != value.get("extraction_contract_sha256")
            or canonical_sha256(contract) != receipt.get("extraction_contract_sha256")
            or contract.get("source_id") != source_id
            or contract.get("representation_receipt_sha256")
            != extraction_contract["representation_receipt_sha256"]
            or contract.get("hidden_config_sha256") != extraction_contract["hidden_config_sha256"]
            or contract.get("split_manifest_sha256") != extraction_contract["split_manifest_sha256"]
            or contract.get("extractor_sha256") != extraction_contract["extractor_sha256"]
            or contract.get("instrumented_bench_sha256")
            != extraction_contract["instrumented_bench_sha256"]
            or contract.get("model_sha256") != extraction_contract["model_sha256"]
            or contract.get("hidden_export_patch_sha256")
            != extraction_contract["hidden_export_patch_sha256"]
            or receipt.get("posterior_equivalence", {}).get("status") != "equivalent"
            or sha256_file(feature_path) != receipt.get("hidden_features_sha256")
        ):
            raise ValueError(f"hidden source evidence differs: {source_id}")
    if len(source_ids) != len(set(source_ids)) or set(source_ids) != expected_sources:
        raise ValueError("hidden extraction source coverage differs")
    expected_provenance = _expected_hidden_provenance()
    for condition, path in hidden_paths.items():
        value = load_json(path)
        if (
            value.get("schema_version") != "psem.frozen_ceiling.hidden_metrics.v1"
            or value.get("condition") != condition
            or value.get("provenance") != expected_provenance
            or not value.get("rows")
        ):
            raise ValueError(f"hidden metric contract differs: {condition}")
        if any(
            row.get("condition") != condition
            or set(row.get("per_source_family", {})) != expected_families
            for row in value["rows"]
        ):
            raise ValueError(f"hidden metric family coverage differs: {condition}")
        if {
            str(receipt.get("held_out_family")) for receipt in value.get("training_receipts", [])
        } != expected_families:
            raise ValueError(f"hidden train-fit fold coverage differs: {condition}")
    source_results = load_json(HIDDEN_SOURCE_RESULTS_PATH)
    if (
        source_results.get("schema_version") != "psem.frozen_ceiling.hidden_source_results.v1"
        or source_results.get("provenance") != expected_provenance
        or {str(row.get("source_id")) for row in source_results.get("rows", [])}
        != expected_eval_sources
    ):
        raise ValueError("hidden per-source result coverage differs")
    return True


def decode_scores(
    session: SessionExamples,
    scores: np.ndarray,
    *,
    threshold: float,
    confirmation_ms: int,
    future_context_frames: int = 0,
    confirmation_support: Sequence[tuple[int, int]] | None = None,
) -> tuple[ReplacementEvent, ...]:
    confirmation = confirmation_ms * 16
    events = []
    index = 0
    support_index = 0
    while index < len(session.starts):
        episode_id = str(session.episode_ids[index])
        speaker = str(session.episode_speakers[index])
        pending_boundary: int | None = None
        pending_samples = 0
        previous_support_end: int | None = None
        while index < len(session.starts) and session.episode_ids[index] == episode_id:
            if not session.valid[index]:
                pending_boundary = None
                pending_samples = 0
                index += 1
                continue
            if session.masked[index]:
                index += 1
                continue
            start = int(session.starts[index])
            end = int(session.ends[index])
            if confirmation_support is None:
                segments = [(start, end)] if session.speech_present[index] else []
            else:
                while (
                    support_index < len(confirmation_support)
                    and confirmation_support[support_index][1] <= start
                ):
                    support_index += 1
                segments = []
                cursor = support_index
                while cursor < len(confirmation_support) and confirmation_support[cursor][0] < end:
                    segment_start = max(start, int(confirmation_support[cursor][0]))
                    segment_end = min(end, int(confirmation_support[cursor][1]))
                    if segment_end > segment_start:
                        segments.append((segment_start, segment_end))
                    cursor += 1
                if not segments or segments[0][0] != previous_support_end:
                    pending_boundary = None
                    pending_samples = 0
            if not segments:
                index += 1
                continue
            if float(scores[index]) < threshold:
                pending_boundary = None
                pending_samples = 0
                index += 1
                continue
            emitted = False
            for segment_start, segment_end in segments:
                if previous_support_end is not None and segment_start != previous_support_end:
                    pending_boundary = None
                    pending_samples = 0
                if pending_boundary is None:
                    pending_boundary = segment_start
                duration = segment_end - segment_start
                needed = confirmation - pending_samples
                if duration >= needed:
                    qualifying = segment_start + needed
                    future_index = min(index + future_context_frames, len(session.starts) - 1)
                    while future_index > index and session.episode_ids[future_index] != episode_id:
                        future_index -= 1
                    frontier = int(session.frontiers[future_index])
                    events.append(
                        ReplacementEvent(
                            source_id=session.source_id,
                            anchor_episode_id=episode_id,
                            anchor_id=speaker,
                            boundary_source_sample=pending_boundary,
                            model_evidence_frontier_sample=frontier,
                            decoder_emit_sample=max(qualifying, frontier),
                            compute_lag_ms=None,
                            confirmation_samples=confirmation,
                        )
                    )
                    emitted = True
                    break
                pending_samples += duration
                previous_support_end = segment_end
            if emitted:
                while index < len(session.starts) and session.episode_ids[index] == episode_id:
                    index += 1
                break
            index += 1
    return tuple(events)


def session_metrics(
    session: SessionExamples,
    events: Sequence[ReplacementEvent],
) -> dict[str, Any]:
    event_by_episode = {value.anchor_episode_id: value for value in events}
    contamination_episodes = [
        (
            episode.anchor_speaker,
            episode.anchor_emit_sample,
            min(
                event_by_episode[episode.episode_id].decoder_emit_sample,
                episode.end_emit_sample,
            )
            if episode.episode_id in event_by_episode
            else episode.end_emit_sample,
        )
        for episode in session.reference.episodes
    ]
    intervals = intervals_from_manifest(session.manifest)
    cfg = config()
    metrics = product_event_metrics(
        predicted_events=events,
        reference=session.reference,
        intervals=intervals,
        contamination_episodes=contamination_episodes,
        tolerance_samples=int(cfg["product_event_alignment_tolerance_ms"] * 16),
    )
    metrics["topology"] = _session_topology(
        session.manifest,
        events,
        session.reference,
        int(cfg["product_event_alignment_tolerance_ms"] * 16),
    )
    return metrics


def session_row(
    session: SessionExamples,
    scores: np.ndarray,
    *,
    condition: str,
    probe_class: str,
    threshold: float,
    confirmation_ms: int,
    time_condition: str,
    future_context_frames: int = 0,
    confirmation_support: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    events = decode_scores(
        session,
        scores,
        threshold=threshold,
        confirmation_ms=confirmation_ms,
        future_context_frames=future_context_frames,
        confirmation_support=confirmation_support,
    )
    metrics = session_metrics(session, events)
    usable = np.logical_and(session.valid, np.logical_not(session.masked))
    selectors = {
        "anchor_only": np.logical_and(session.anchor_present, np.logical_not(session.overlap)),
        "anchor_overlap": session.overlap,
        "anchor_absent_live": session.target,
    }
    diagnostics = {}
    for name, selector in selectors.items():
        chosen = np.logical_and(usable, selector)
        weight = float(session.weights[chosen].sum())
        diagnostics[name] = {
            "support_seconds": weight / 16000.0,
            "mean_hazard": (
                float(np.average(scores[chosen], weights=session.weights[chosen]))
                if weight
                else None
            ),
        }
    topology = metrics["topology"]
    return {
        "condition": condition,
        "probe_class": probe_class,
        "time_condition": time_condition,
        "threshold": threshold,
        "confirmation_ms": confirmation_ms,
        "source_id": session.source_id,
        "source_family": session.source_family,
        "old_v2_role": session.manifest["role"],
        "metrics": metrics,
        "topology": topology,
        "diagnostics": diagnostics,
    }


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = [value["metrics"] for value in rows]
    active_seconds = sum(float(value["active_speech_seconds"]) for value in metrics)
    active_hours = active_seconds / 3600.0
    emit = [float(item) for value in metrics for item in value["replacement_emit_delay_values_ms"]]
    boundary = [
        float(item) for value in metrics for item in value["backdated_boundary_error_values_ms"]
    ]
    topology = _aggregate_topology(rows)
    diagnostics = {}
    if rows and all("diagnostics" in value for value in rows):
        for name in rows[0]["diagnostics"]:
            support = sum(float(value["diagnostics"][name]["support_seconds"]) for value in rows)
            diagnostics[name] = {
                "support_seconds": support,
                "mean_hazard": (
                    sum(
                        float(value["diagnostics"][name]["mean_hazard"] or 0.0)
                        * float(value["diagnostics"][name]["support_seconds"])
                        for value in rows
                    )
                    / support
                    if support
                    else None
                ),
            }
    return {
        "source_count": len(rows),
        "source_families": sorted({value["source_family"] for value in rows}),
        "active_speech_hours": active_hours,
        "predicted_cut_count": sum(int(value["predicted_cut_count"]) for value in metrics),
        "reference_replacement_count": sum(
            int(value["reference_replacement_count"]) for value in metrics
        ),
        "matched_replacement_count": sum(
            int(value["matched_replacement_count"]) for value in metrics
        ),
        "false_cut_count": sum(int(value["false_cut_count"]) for value in metrics),
        "missed_replacement_count": sum(
            int(value["missed_replacement_count"]) for value in metrics
        ),
        "speaker_induced_cut_count_per_active_speech_hour": (
            sum(int(value["predicted_cut_count"]) for value in metrics) / active_hours
            if active_hours
            else None
        ),
        "exclusive_other_contamination_seconds_per_active_speech_hour": (
            sum(float(value["exclusive_other_contamination_seconds"]) for value in metrics)
            / active_hours
            if active_hours
            else None
        ),
        "replacement_emit_delay_ms": {
            "p50": percentile(emit, 50),
            "p90": percentile(emit, 90),
        },
        "backdated_boundary_error_ms": {
            "p50": percentile(boundary, 50),
            "p90": percentile(boundary, 90),
        },
        "overlap_return_preservation_rate": topology.get("overlap_return", {}).get(
            "overlap_return_preservation_rate"
        ),
        "overlap_takeover_success_rate": topology.get("overlap_takeover", {}).get(
            "overlap_takeover_success_rate"
        ),
        "topology": topology,
        "diagnostic_slices": diagnostics,
    }


def aggregate_conditions(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["condition"],
            row["probe_class"],
            row["time_condition"],
            float(row["threshold"]),
            int(row["confirmation_ms"]),
        )
        groups[key].append(row)
    result = []
    for key, chosen in sorted(groups.items()):
        result.append(
            {
                "condition": key[0],
                "probe_class": key[1],
                "time_condition": key[2],
                "threshold": key[3],
                "confirmation_ms": key[4],
                "metrics": aggregate_rows(chosen),
                "per_source_family": {
                    family: aggregate_rows(
                        [value for value in chosen if value["source_family"] == family]
                    )
                    for family in sorted({value["source_family"] for value in chosen})
                },
            }
        )
    return result


def _point(
    path: Path, condition: str, probe: str, threshold: float, persistence: int
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return next(
        row
        for row in value["rows"]
        if row["condition"] == condition
        and row["probe_class"] == probe
        and row["threshold"] == threshold
        and row["confirmation_ms"] == persistence
    )


def _per_source_family_point(
    value: dict[str, Any],
    condition: str,
    probe: str,
    threshold: float,
    persistence: int,
) -> dict[str, Any]:
    point = next(
        row
        for row in aggregate_conditions(value["per_source"])
        if row["condition"] == condition
        and row["probe_class"] == probe
        and row["threshold"] == threshold
        and row["confirmation_ms"] == persistence
    )
    return point["per_source_family"]


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "contamination_seconds_per_active_speech_hour": (
            left["exclusive_other_contamination_seconds_per_active_speech_hour"]
            - right["exclusive_other_contamination_seconds_per_active_speech_hour"]
        ),
        "false_cut_count": left["false_cut_count"] - right["false_cut_count"],
        "missed_replacement_count": (
            left["missed_replacement_count"] - right["missed_replacement_count"]
        ),
        "replacement_emit_delay_p90_ms": (
            (left["replacement_emit_delay_ms"]["p90"] or 0.0)
            - (right["replacement_emit_delay_ms"]["p90"] or 0.0)
        ),
        "overlap_return_preservation_rate": (
            (left["overlap_return_preservation_rate"] or 0.0)
            - (right["overlap_return_preservation_rate"] or 0.0)
        ),
        "overlap_takeover_success_rate": (
            (left["overlap_takeover_success_rate"] or 0.0)
            - (right["overlap_takeover_success_rate"] or 0.0)
        ),
    }


def _pareto_relation(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_values = (
        left["exclusive_other_contamination_seconds_per_active_speech_hour"],
        left["false_cut_count"],
        left["missed_replacement_count"],
        left["replacement_emit_delay_ms"]["p90"] or float("inf"),
        -(left["overlap_return_preservation_rate"] or 0.0),
        -(left["overlap_takeover_success_rate"] or 0.0),
    )
    right_values = (
        right["exclusive_other_contamination_seconds_per_active_speech_hour"],
        right["false_cut_count"],
        right["missed_replacement_count"],
        right["replacement_emit_delay_ms"]["p90"] or float("inf"),
        -(right["overlap_return_preservation_rate"] or 0.0),
        -(right["overlap_takeover_success_rate"] or 0.0),
    )
    if all(left <= right for left, right in zip(left_values, right_values, strict=True)) and any(
        left < right for left, right in zip(left_values, right_values, strict=True)
    ):
        return 1
    if all(right <= left for left, right in zip(left_values, right_values, strict=True)) and any(
        right < left for left, right in zip(left_values, right_values, strict=True)
    ):
        return -1
    return 0


def _frontier_comparison(left_path: Path, right_path: Path) -> dict[str, int]:
    left_rows = json.loads(left_path.read_text(encoding="utf-8"))["rows"]
    right_rows = json.loads(right_path.read_text(encoding="utf-8"))["rows"]
    right_index = {
        (value["probe_class"], value["threshold"], value["confirmation_ms"]): value
        for value in right_rows
    }
    counts = {"left_dominates": 0, "right_dominates": 0, "tradeoff": 0}
    for value in left_rows:
        key = (value["probe_class"], value["threshold"], value["confirmation_ms"])
        if key not in right_index:
            continue
        relation = _pareto_relation(value["metrics"], right_index[key]["metrics"])
        counts[
            "left_dominates"
            if relation == 1
            else "right_dominates"
            if relation == -1
            else "tradeoff"
        ] += 1
    return counts


def _hidden_train_fit(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    reference_probe = str(cfg["reference_probe_class"])
    receipts = [
        receipt
        for receipt in value["training_receipts"]
        if receipt["probe_class"] == reference_probe
    ]
    checks = [
        receipt["train_fit_sanity"]["duration_weighted_average_precision"]
        >= float(cfg["train_fit_min_average_precision"])
        and receipt["train_fit_sanity"]["duration_weighted_accuracy"]
        >= float(cfg["train_fit_min_accuracy"])
        for receipt in receipts
    ]
    return {
        "status": "passed" if receipts and all(checks) else "failed",
        "reference_probe_class": reference_probe,
        "fold_count": len(receipts),
        "minimum_average_precision": min(
            (
                receipt["train_fit_sanity"]["duration_weighted_average_precision"]
                for receipt in receipts
            ),
            default=None,
        ),
        "minimum_accuracy": min(
            (receipt["train_fit_sanity"]["duration_weighted_accuracy"] for receipt in receipts),
            default=None,
        ),
    }


def _hidden_success(
    left_path: Path,
    right_path: Path,
    left: dict[str, Any],
    right: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    comparison = _frontier_comparison(left_path, right_path)
    fields = tuple(map(str, cfg["success_reference_improvements"]))
    pooled = {field: left["metrics"][field] < right["metrics"][field] for field in fields}
    family_counts = {
        family: sum(
            left["per_source_family"][family][field] < right["per_source_family"][family][field]
            for field in fields
        )
        for family in sorted(left["per_source_family"])
    }
    passed = (
        comparison["left_dominates"] > comparison["right_dominates"]
        and all(pooled.values())
        and all(
            count >= int(cfg["source_family_min_improved_metrics"])
            for count in family_counts.values()
        )
    )
    return {
        "status": "passed" if passed else "failed",
        "frontier_comparison": comparison,
        "reference_cell_improvements": pooled,
        "source_family_improved_metric_counts": family_counts,
    }


def render_final_decision() -> str:
    required = {
        "G": RESULTS_ROOT / "gt_causal_action_frontier.json",
        "S-current": RESULTS_ROOT / "scalar_current_metrics.json",
        "S-probe": RESULTS_ROOT / "scalar_probe_metrics.json",
        "P-C": RESULTS_ROOT / "fullslot_causal_metrics.json",
        "P-NC": RESULTS_ROOT / "fullslot_noncausal_metrics.json",
        "VAD": RESULTS_ROOT / "vad_support_hygiene.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing result artifacts: " + ", ".join(missing))
    cfg = config()
    hidden_cfg = load_json(HIDDEN_CONFIG_PATH)
    persistence = int(cfg["gap_reference_confirmation_ms"])
    g_result = json.loads(required["G"].read_text(encoding="utf-8"))
    g_cell = next(
        value
        for value in g_result["rows"]
        if value["confirmation_ms"] == persistence
    )
    g_per_source_family = _per_source_family_point(
        g_result,
        "G",
        "gt_fixed_confirmation",
        1.0,
        persistence,
    )
    g = g_cell["metrics"]
    current = _point(required["S-current"], "S-current", "current", 0.5, persistence)["metrics"]
    scalar = _point(required["S-probe"], "S-probe", "tiny_mlp", 0.5, persistence)["metrics"]
    causal = _point(required["P-C"], "P-C", "tiny_mlp", 0.5, persistence)["metrics"]
    noncausal = _point(required["P-NC"], "P-NC", "tiny_mlp", 0.5, persistence)["metrics"]
    vad = json.loads(required["VAD"].read_text(encoding="utf-8"))
    pc_vs_scalar = _frontier_comparison(required["P-C"], required["S-probe"])
    pnc_vs_pc = _frontier_comparison(required["P-NC"], required["P-C"])
    hidden_paths = {
        "H-C": RESULTS_ROOT / "hidden_causal_metrics.json",
        "H-NC": RESULTS_ROOT / "hidden_noncausal_metrics.json",
    }
    hidden_available = _validate_hidden_evidence(hidden_paths)
    hidden_cells: dict[str, Any] = {}
    hidden_diagnostics: dict[str, Any] = {}
    terminal_status = "hidden_ceiling_not_evaluated"
    if hidden_available:
        hidden_cells = {
            "H-C": _point(
                hidden_paths["H-C"],
                "H-C",
                str(hidden_cfg["reference_probe_class"]),
                float(hidden_cfg["reference_threshold"]),
                persistence,
            ),
            "H-NC": _point(
                hidden_paths["H-NC"],
                "H-NC",
                str(hidden_cfg["reference_probe_class"]),
                float(hidden_cfg["reference_threshold"]),
                persistence,
            ),
        }
        fit = {
            condition: _hidden_train_fit(path, hidden_cfg)
            for condition, path in hidden_paths.items()
        }
        causal_success = _hidden_success(
            hidden_paths["H-C"],
            required["P-C"],
            hidden_cells["H-C"],
            _point(required["P-C"], "P-C", "tiny_mlp", 0.5, persistence),
            hidden_cfg,
        )
        noncausal_success = _hidden_success(
            hidden_paths["H-NC"],
            hidden_paths["H-C"],
            hidden_cells["H-NC"],
            hidden_cells["H-C"],
            hidden_cfg,
        )
        hidden_diagnostics = {
            "train_fit_sanity": fit,
            "hidden_causal_over_posterior_causal": causal_success,
            "hidden_noncausal_over_hidden_causal": noncausal_success,
        }
        family_concentration = {}
        for family, hidden_family in hidden_cells["H-C"]["per_source_family"].items():
            hidden_slices = hidden_family["diagnostic_slices"]
            g_family = g_per_source_family[family]
            overlap_hazard = hidden_slices["anchor_overlap"]["mean_hazard"]
            anchor_hazard = hidden_slices["anchor_only"]["mean_hazard"]
            absent_hazard = hidden_slices["anchor_absent_live"]["mean_hazard"]
            hidden_takeover = hidden_family["overlap_takeover_success_rate"]
            g_takeover = g_family["overlap_takeover_success_rate"]
            family_concentration[family] = {
                "overlap_masking": (
                    None not in (overlap_hazard, anchor_hazard, hidden_takeover, g_takeover)
                    and overlap_hazard > anchor_hazard
                    and hidden_takeover < g_takeover
                ),
                "competitor_separation": (
                    None not in (absent_hazard, overlap_hazard) and absent_hazard <= overlap_hazard
                ),
            }
        residual_concentration = {
            "per_source_family": family_concentration,
            "status": (
                "passed"
                if family_concentration
                and all(any(checks.values()) for checks in family_concentration.values())
                else "failed"
            ),
        }
        hidden_diagnostics["neural_acoustic_failure_concentration"] = residual_concentration
        if any(value["status"] != "passed" for value in fit.values()):
            path = "unresolved; hidden probe train-fit sanity failed"
            next_issue = "hidden probe train-fit repair within this issue"
            hidden = (
                "opened and executed, but failure cannot be interpreted because the predeclared "
                f"train-fit sanity gate failed: {fit}"
            )
            rejected = "Full Sortformer adaptation remains unauthorized until hidden train-fit sanity is proven."
            terminal_status = "hidden_probe_train_fit_inconclusive"
        elif causal_success["status"] == "passed":
            path = "B. frozen backbone + PSEM task head/adapter -> native S2 -> KD"
            next_issue = "frozen-head/adapter experiment"
            hidden = (
                "opened and the hidden causal ceiling passed the predeclared pooled, frontier, "
                f"and source-family consistency rule: {causal_success}"
            )
            rejected = "Direct scalar-posterior distillation and immediate full Sortformer fine-tuning are rejected."
            terminal_status = "hidden_ceiling_interpreted"
        elif noncausal_success["status"] == "passed":
            path = "C. streaming/context reformulation or partial adaptation"
            next_issue = "streaming reformulation experiment"
            hidden = (
                "opened; hidden causal failed while bounded hidden non-causal context passed the "
                f"predeclared rule: {noncausal_success}"
            )
            rejected = "Direct scalar-posterior distillation and immediate full fine-tuning are rejected before streaming reformulation."
            terminal_status = "hidden_ceiling_interpreted"
        elif residual_concentration["status"] == "passed":
            path = "D. Sortformer task adaptation / full FT -> KD"
            next_issue = "Sortformer task adaptation"
            hidden = (
                "opened; causal and bounded non-causal hidden ceilings both failed despite passing "
                f"train-fit sanity: {hidden_diagnostics}"
            )
            rejected = "Further decoder proliferation and direct scalar-posterior distillation are rejected; task adaptation is now justified."
            terminal_status = "hidden_ceiling_interpreted"
        else:
            path = (
                "unresolved; hidden failure is not localized to the required neural/acoustic slices"
            )
            next_issue = "hidden failure attribution repair within this issue"
            hidden = (
                "opened; causal and non-causal hidden ceilings failed with train-fit sanity, but "
                f"the predeclared neural/acoustic concentration check failed: {residual_concentration}"
            )
            rejected = "Full Sortformer adaptation remains unauthorized until the residual failure is localized."
            terminal_status = "hidden_failure_attribution_inconclusive"
    elif pnc_vs_pc["left_dominates"] > pnc_vs_pc["right_dominates"]:
        path = "C. streaming/context reformulation or partial adaptation"
        next_issue = "streaming reformulation experiment"
        hidden = "not opened; bounded future context is the first diagnosed bottleneck"
        rejected = (
            "Direct scalar-posterior distillation and immediate full fine-tuning are rejected."
        )
        terminal_status = "posterior_ceiling_interpreted"
    elif pc_vs_scalar["left_dominates"] > pc_vs_scalar["right_dominates"]:
        path = "A. frozen posterior/readout -> native S2 -> compact student GT/KD"
        next_issue = "NATIVE-S2-1"
        hidden = "not opened because the causal posterior frontier improves on the scalar frontier"
        rejected = (
            "Direct scalar-posterior distillation and immediate full fine-tuning are rejected."
        )
        terminal_status = "posterior_ceiling_interpreted"
    else:
        path = "posterior ceiling did not establish a path; HIDDEN-CEILING-1 is required before selecting B or D"
        next_issue = "conditional HIDDEN-CEILING-1 within this issue"
        hidden = (
            "opened by the stop rule; hidden results are required before this decision is terminal"
        )
        rejected = "Direct scalar-posterior distillation and immediate full fine-tuning remain rejected until hidden evidence exists."
        terminal_status = "hidden_ceiling_required"
    gap = {
        "G_to_S_current": _delta(current, g),
        "S_current_to_S_probe": _delta(scalar, current),
        "S_probe_to_P_C": _delta(causal, scalar),
        "P_C_to_P_NC": _delta(noncausal, causal),
        "G_to_P_C_residual": _delta(causal, g),
        "pareto_counts": {
            "P_C_vs_S_probe": pc_vs_scalar,
            "P_NC_vs_P_C": pnc_vs_pc,
        },
    }
    if hidden_available:
        gap.update(
            {
                "P_C_to_H_C": _delta(hidden_cells["H-C"]["metrics"], causal),
                "H_C_to_H_NC": _delta(
                    hidden_cells["H-NC"]["metrics"], hidden_cells["H-C"]["metrics"]
                ),
                "G_to_H_C_residual": _delta(hidden_cells["H-C"]["metrics"], g),
                "hidden_decision_diagnostics": hidden_diagnostics,
            }
        )
    vad_old = vad["issue98_mixed_support_reference"]
    vad_new = vad["corrected_cached_posterior_replay"]
    text = "# FROZEN-CEILING-1 final decision\n\n"
    text += "This report is generated only after the scored artifacts exist. It is development-known path-selection evidence, not production readiness or a fresh holdout.\n\n"
    text += "## Ordered answers\n\n"
    answers = [
        "Yes as the bounded evaluator floor: G reproduces the shared GT Simple Anchor event ledger across the predeclared confirmation frontier without neural evidence. This does not declare one scalar utility or production readiness.",
        f"At the predeclared {persistence} ms / 0.5 diagnostic cell, the action-policy plus neural gap is recorded as G→S-current below; the full frontier remains authoritative.",
        "S-probe versus S-current is quantified below at the fixed diagnostic cell; no architecture or threshold was selected on the held-out families.",
        f"P-C versus S-probe has Pareto counts {pc_vs_scalar}; the fixed-cell delta is recorded below.",
        f"P-NC versus P-C has Pareto counts {pnc_vs_pc}; future context remains diagnostic and its frontier delay includes the future evidence availability.",
        f"The fixed-cell residual diagnostics are {json.dumps((hidden_cells.get('H-C') or {'metrics': causal})['metrics']['diagnostic_slices'], sort_keys=True)} and the topology metrics below retain direct handoff, silence-gap handoff, pause/resume, overlap return, overlap takeover, and short-backchannel slices.",
        (
            "Yes; proceed to native causal S2."
            if next_issue == "NATIVE-S2-1"
            else "No; the posterior result does not yet authorize native S2."
        ),
        hidden,
        f"Selected path: {path}.",
        rejected,
        f"The corrected committed-live-only replay changed contamination seconds per active speech hour from {vad_old['exclusive_other_contamination_seconds_per_active_speech_hour']:.3f} to {vad_new['exclusive_other_contamination_seconds_per_active_speech_hour']:.3f}; this is integration hygiene, not teacher selection.",
        "Actual persistent-state VAD gating remains deferred until a viable oracle-binding path and native S2 pass. The recorded duty-cycle split supports a later A/B but cannot validate the gated model trajectory.",
        f"Single next issue: {next_issue}.",
    ]
    for index, answer in enumerate(answers, 1):
        text += f"{index}. {answer}\n\n"
    text += f"## Nested gap attribution at {persistence} ms / 0.5\n\n```json\n"
    text += json.dumps(gap, indent=2, sort_keys=True)
    text += "\n```\n\n## Reference product cells\n\n```json\n"
    reference_cells = {
        "G": g,
        "S-current": current,
        "S-probe": scalar,
        "P-C": causal,
        "P-NC": noncausal,
    }
    reference_cells.update(
        {condition: value["metrics"] for condition, value in hidden_cells.items()}
    )
    text += json.dumps(reference_cells, indent=2, sort_keys=True)
    text += "\n```\n"
    (RESULTS_ROOT / "FINAL_DECISION.md").write_text(text, encoding="utf-8")
    return terminal_status


def main() -> None:
    status = render_final_decision()
    hidden_artifacts = {
        name: sha256_file(RESULTS_ROOT / filename)
        for name, filename in (
            ("H-C", "hidden_causal_metrics.json"),
            ("H-NC", "hidden_noncausal_metrics.json"),
            ("source_families", "hidden_source_family_results.json"),
            ("extraction", "hidden_extraction_receipt.json"),
        )
        if (RESULTS_ROOT / filename).is_file()
    }
    write_json(
        RESULTS_ROOT / "evaluation_receipt.json",
        {
            "schema_version": "psem.frozen_ceiling.evaluation_receipt.v1",
            "status": status,
            "final_decision_sha256": sha256_file(RESULTS_ROOT / "FINAL_DECISION.md"),
            "hidden_artifacts": hidden_artifacts,
        },
    )


if __name__ == "__main__":
    main()
