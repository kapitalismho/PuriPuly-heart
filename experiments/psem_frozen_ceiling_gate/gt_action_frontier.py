from __future__ import annotations

from typing import Any

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
    ACTION_REFERENCE_COVERAGE_PATH,
    ACTION_REFERENCE_LEDGER_PATH,
    config,
    load_sessions,
)
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
    RESULTS_ROOT,
    aggregate_rows,
    session_metrics,
)
from experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies import (
    _gt_challenger_event,
)
from experiments.psem_relative_occupancy_gate.io_utils import sha256_file, write_json


def run() -> dict[str, Any]:
    cfg = config()
    rows = []
    sessions = load_sessions((int(cfg["action_reference_confirmation_ms"]),))
    for persistence in map(int, cfg["gt_confirmation_ms"]):
        for session in sessions:
            events = tuple(
                event
                for episode in session.reference.episodes
                if (
                    event := _gt_challenger_event(
                        row=session.manifest,
                        episode=episode,
                        candidate="simple_anchor",
                        confirmation_samples=persistence * 16,
                    )
                )
                is not None
            )
            metrics = session_metrics(session, events)
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
                    "metrics": metrics,
                    "topology": metrics["topology"],
                }
            )
    scoring = [value for value in rows if value["partition"] == "eval"]
    aggregates = []
    for persistence in cfg["gt_confirmation_ms"]:
        chosen = [value for value in scoring if value["confirmation_ms"] == persistence]
        aggregates.append(
            {
                "condition": "G",
                "probe_class": "gt_fixed_confirmation",
                "time_condition": "causal",
                "threshold": 1.0,
                "confirmation_ms": persistence,
                "required_issue98_reference_point": persistence == 500,
                "metrics": aggregate_rows(chosen),
            }
        )
    result = {
        "schema_version": "psem.frozen_ceiling.gt_frontier.v1",
        "decision_latency_budget_ms": cfg["maximum_decision_latency_ms"],
        "shared_reference_rule": "the issue98 500ms fixed-persistence GT Simple Anchor episode and event ledger is the single action authority for all evidence conditions",
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
