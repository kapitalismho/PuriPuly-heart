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
from experiments.psem_relative_occupancy_gate.io_utils import percentile, write_json
from experiments.psem_relative_occupancy_gate.model_evaluate import (
    intervals_from_manifest,
    product_event_metrics,
)

RESULTS_ROOT = PACKAGE_ROOT / "results" / "frozen_ceiling_1"


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


def render_final_decision() -> None:
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
    persistence = int(cfg["gap_reference_confirmation_ms"])
    g = next(
        value
        for value in json.loads(required["G"].read_text(encoding="utf-8"))["rows"]
        if value["confirmation_ms"] == persistence
    )["metrics"]
    current = _point(required["S-current"], "S-current", "current", 0.5, persistence)["metrics"]
    scalar = _point(required["S-probe"], "S-probe", "tiny_mlp", 0.5, persistence)["metrics"]
    causal = _point(required["P-C"], "P-C", "tiny_mlp", 0.5, persistence)["metrics"]
    noncausal = _point(required["P-NC"], "P-NC", "tiny_mlp", 0.5, persistence)["metrics"]
    vad = json.loads(required["VAD"].read_text(encoding="utf-8"))
    pc_vs_scalar = _frontier_comparison(required["P-C"], required["S-probe"])
    pnc_vs_pc = _frontier_comparison(required["P-NC"], required["P-C"])
    if pnc_vs_pc["left_dominates"] > pnc_vs_pc["right_dominates"]:
        path = "C. streaming/context reformulation or partial adaptation"
        next_issue = "streaming reformulation experiment"
        hidden = "not opened; bounded future context is the first diagnosed bottleneck"
    elif pc_vs_scalar["left_dominates"] > pc_vs_scalar["right_dominates"]:
        path = "A. frozen posterior/readout -> native S2 -> compact student GT/KD"
        next_issue = "NATIVE-S2-1"
        hidden = "not opened because the causal posterior frontier improves on the scalar frontier"
    else:
        path = "posterior ceiling did not establish a path; HIDDEN-CEILING-1 is required before selecting B or D"
        next_issue = "conditional HIDDEN-CEILING-1 within this issue"
        hidden = (
            "opened by the stop rule; hidden results are required before this decision is terminal"
        )
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
        f"The fixed-cell residual diagnostics are {json.dumps(causal['diagnostic_slices'], sort_keys=True)} and the topology metrics below retain direct handoff, silence-gap handoff, pause/resume, overlap return, overlap takeover, and short-backchannel slices.",
        (
            "Yes; proceed to native causal S2."
            if next_issue == "NATIVE-S2-1"
            else "No; the posterior result does not yet authorize native S2."
        ),
        hidden,
        f"Selected path: {path}.",
        "Direct distillation of the current scalar p_anchor decoder is rejected. Immediate full Sortformer fine-tuning is also rejected unless the conditional hidden ceiling later satisfies its explicit trigger.",
        f"The corrected committed-live-only replay changed contamination seconds per active speech hour from {vad_old['exclusive_other_contamination_seconds_per_active_speech_hour']:.3f} to {vad_new['exclusive_other_contamination_seconds_per_active_speech_hour']:.3f}; this is integration hygiene, not teacher selection.",
        "Actual persistent-state VAD gating remains deferred until a viable oracle-binding path and native S2 pass. The recorded duty-cycle split supports a later A/B but cannot validate the gated model trajectory.",
        f"Single next issue: {next_issue}.",
    ]
    for index, answer in enumerate(answers, 1):
        text += f"{index}. {answer}\n\n"
    text += f"## Nested gap attribution at {persistence} ms / 0.5\n\n```json\n"
    text += json.dumps(gap, indent=2, sort_keys=True)
    text += "\n```\n\n## Reference product cells\n\n```json\n"
    text += json.dumps(
        {"G": g, "S-current": current, "S-probe": scalar, "P-C": causal, "P-NC": noncausal},
        indent=2,
        sort_keys=True,
    )
    text += "\n```\n"
    (RESULTS_ROOT / "FINAL_DECISION.md").write_text(text, encoding="utf-8")


def main() -> None:
    render_final_decision()
    write_json(
        RESULTS_ROOT / "evaluation_receipt.json",
        {
            "schema_version": "psem.frozen_ceiling.evaluation_receipt.v1",
            "status": "posterior_ceiling_interpreted",
        },
    )


if __name__ == "__main__":
    main()
