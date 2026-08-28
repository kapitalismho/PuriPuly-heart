from __future__ import annotations

from typing import Any

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
    ACTION_REFERENCE_COVERAGE_PATH,
    ACTION_REFERENCE_LEDGER_PATH,
    config,
    load_sessions,
)
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import RESULTS_ROOT, aggregate_rows
from experiments.psem_frozen_ceiling_gate.experiment_support import (
    intervals_from_manifest,
    product_event_metrics,
    session_topology,
    sha256_file,
    simulate_gt_session,
    write_json,
)


def _signature(value: Any) -> tuple[Any, ...]:
    return (
        value.source_id,
        value.anchor_episode_id,
        value.anchor_id,
        value.boundary_source_sample,
        value.model_evidence_frontier_sample,
        value.decoder_emit_sample,
        value.confirmation_samples,
    )


def run() -> dict[str, Any]:
    cfg = config()
    sessions = load_sessions((int(cfg["action_reference_confirmation_ms"]),))
    rows = []
    for persistence in map(int, cfg["gt_confirmation_ms"]):
        for session in sessions:
            candidate = simulate_gt_session(
                session.manifest,
                replacement_confirmation_samples=persistence * 16,
                enrollment_samples=int(cfg["enrollment_ms"]) * 16,
                silence_reset_samples=int(cfg["silence_reset_ms"]) * 16,
            )
            if persistence == int(cfg["action_reference_confirmation_ms"]) and tuple(
                map(_signature, candidate.events)
            ) != tuple(map(_signature, session.reference.events)):
                raise ValueError(f"500 ms GT reference mismatch: {session.source_id}")
            event_by_episode = {value.anchor_episode_id: value for value in candidate.events}
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
                for episode in candidate.episodes
            ]
            metrics = product_event_metrics(
                predicted_events=candidate.events,
                reference=session.reference,
                intervals=intervals_from_manifest(session.manifest),
                contamination_episodes=contamination_episodes,
                tolerance_samples=int(cfg["product_event_alignment_tolerance_ms"]) * 16,
            )
            metrics["topology"] = session_topology(
                session.manifest,
                candidate.events,
                session.reference,
                int(cfg["product_event_alignment_tolerance_ms"]) * 16,
            )
            rows.append(
                {
                    "condition": "G",
                    "probe_class": "gt_fixed_confirmation",
                    "time_condition": "causal",
                    "threshold": 1.0,
                    "confirmation_ms": persistence,
                    "source_id": session.source_id,
                    "source_family": session.source_family,
                    "partition": session.role,
                    "old_v2_role": session.manifest["role"],
                    "candidate_episode_count": len(candidate.episodes),
                    "metrics": metrics,
                    "topology": metrics["topology"],
                }
            )
    scoring = [value for value in rows if value["partition"] == "eval"]
    aggregates = []
    for persistence in map(int, cfg["gt_confirmation_ms"]):
        chosen = [value for value in scoring if value["confirmation_ms"] == persistence]
        aggregates.append(
            {
                "condition": "G",
                "probe_class": "gt_fixed_confirmation",
                "time_condition": "causal",
                "threshold": 1.0,
                "confirmation_ms": persistence,
                "required_issue98_reference_point": persistence
                == int(cfg["action_reference_confirmation_ms"]),
                "metrics": aggregate_rows(chosen),
            }
        )
    result = {
        "schema_version": "psem.frozen_ceiling.gt_frontier.v2",
        "decision_latency_budget_ms": cfg["maximum_decision_latency_ms"],
        "frontier_rule": "each confirmation arm is simulated directly from the exact-source-time GT activity timeline; the sealed issue98 500 ms ledger remains the shared scoring reference",
        "action_reference": {
            "confirmation_ms": int(cfg["action_reference_confirmation_ms"]),
            "ledger_sha256": sha256_file(ACTION_REFERENCE_LEDGER_PATH),
            "coverage_sha256": sha256_file(ACTION_REFERENCE_COVERAGE_PATH),
        },
        "rows": aggregates,
        "per_source": scoring,
    }
    write_json(RESULTS_ROOT / "gt_causal_action_frontier.json", result)
    return result


if __name__ == "__main__":
    run()
