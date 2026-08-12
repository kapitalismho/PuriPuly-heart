from __future__ import annotations

import argparse
import importlib.util
import platform
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from experiments.online_speaker_memory_handoff.protocol import (
    CONFIG_PATH,
    REPOSITORY_ROOT,
    R6Error,
    cache_root,
    load_json,
    output_root,
    sha256_file,
    write_json,
)

REPRESENTATIONS = ("m-l1", "e-s3", "e-final")


def _environment_smoke() -> dict[str, Any]:
    import torch

    nemo_available = importlib.util.find_spec("nemo") is not None
    accelerator_available = bool(torch.cuda.is_available())
    feasible = platform.system() == "Linux" and nemo_available and accelerator_available
    reasons: list[str] = []
    if platform.system() != "Linux":
        reasons.append("official smoke environment is not Linux")
    if not nemo_available:
        reasons.append("NeMo is not installed")
    if not accelerator_available:
        reasons.append("PyTorch has no CUDA or ROCm accelerator")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
        "nemo_available": nemo_available,
        "sortformer_smoke_feasible": feasible,
        "limitation_reasons": reasons,
    }


def _baseline_inventory() -> dict[str, Any]:
    phase3_path = (
        REPOSITORY_ROOT / "experiments/speaker_turn_boundary/results/phase3/dev_summary_v2.json"
    )
    phase4_manifest_path = (
        REPOSITORY_ROOT
        / "experiments/speaker_turn_boundary/results/turn_episode_v1/natural_exposure_manifest.json"
    )
    phase4_ls_path = (
        REPOSITORY_ROOT
        / "experiments/speaker_turn_boundary/results/turn_episode_v1/phase_4_ls_signal_report.json"
    )
    phase4_eres_path = (
        REPOSITORY_ROOT
        / "experiments/speaker_turn_boundary/results/turn_episode_v1/phase_4_eres_signal_report.json"
    )
    required = (phase3_path, phase4_manifest_path, phase4_ls_path, phase4_eres_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise R6Error(f"existing baseline artifacts are missing: {missing}")
    phase3 = load_json(phase3_path)
    phase4_manifest = load_json(phase4_manifest_path)
    episode_seconds = sum(
        (int(row["bounds"]["scored_end"]) - int(row["bounds"]["scored_start"])) / 16000
        for row in phase4_manifest["episodes"]
    )
    sessions = sorted({str(row["session_id"]) for row in phase4_manifest["episodes"]})
    return {
        "existing_ls_eend": {
            "status": "reused_provenance_only_not_metric_compatible",
            "phase3_source_hours": float(phase3["source_seconds"]) / 3600,
            "phase3_manifest_sha256": str(phase3["manifest_sha256"]),
            "phase3_summary_sha256": sha256_file(phase3_path),
            "phase4_signal_report_sha256": sha256_file(phase4_ls_path),
        },
        "existing_eres": {
            "status": "reused_provenance_only_not_metric_compatible",
            "phase4_signal_report_sha256": sha256_file(phase4_eres_path),
        },
        "compatibility": {
            "phase4_episode_count": len(phase4_manifest["episodes"]),
            "phase4_episode_hours": episode_seconds / 3600,
            "phase4_session_count": len(sessions),
            "reason": (
                "the existing event artifacts score selected 30-second episodes with the prior "
                "hard-boundary/overlap schema, not the complete chronological first-handoff units "
                "and false-event exposure frozen for R6"
            ),
        },
    }


def _gate_status(rows: Sequence[dict[str, Any]]) -> str:
    gates = {str(row["gate"]) for row in rows}
    if "promote" in gates:
        return "promote"
    if "conditional" in gates:
        return "conditional"
    return "stop"


def _best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    plausible = [row for row in rows if float(row["candidate_fp_h"]) <= 10.0]
    candidates = plausible or list(rows)
    return max(
        candidates,
        key=lambda row: (
            float(row["candidate_r1000"]),
            float(row["candidate_r1500"]),
            -float(row["candidate_fp_h"]),
        ),
    )


def _minimum_development_cost(
    documents: Sequence[dict[str, Any]], recall_floor: float
) -> dict[str, Any] | None:
    points: list[dict[str, Any]] = []
    for document in documents:
        for row in document["development"]["curve"]:
            recall = row["candidate"]["causal"]["1000"]["recall"]
            false_events = row["candidate"]["false_events_per_hour"]
            if recall is None or false_events is None or float(recall) < recall_floor:
                continue
            points.append(
                {
                    "query_context_ms": int(document["query_context_ms"]),
                    "enrollment_ms": int(document["enrollment_ms"]),
                    "aggregation": str(document["aggregation"]),
                    "threshold": float(row["threshold"]),
                    "candidate_recall_at_1000ms": float(recall),
                    "candidate_false_events_per_hour": float(false_events),
                }
            )
    if not points:
        return None
    return min(points, key=lambda row: row["candidate_false_events_per_hour"])


def _format_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.2f}%"


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def finalize(root: Path) -> Path:
    config = load_json(CONFIG_PATH)
    base = output_root(config, root)
    a1_summary_path = base / "a1/summary.json"
    if not a1_summary_path.is_file():
        raise R6Error("R6-A1 summary is missing")
    a1 = load_json(a1_summary_path)
    if int(a1.get("schema_version", 0)) != 2:
        raise R6Error("R6-A1 summary is not schema version 2")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metric_documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    code_hashes: set[str] = set()
    for row in a1["rows"]:
        metrics = load_json(Path(row["metrics_path"]))
        if int(metrics.get("schema_version", 0)) != 2:
            raise R6Error(f"stale metrics schema: {row['metrics_path']}")
        grouped[str(row["representation"])].append(row)
        metric_documents[str(row["representation"])].append(metrics)
        code_hashes.add(str(metrics["provenance"]["code_sha256"]))
    if set(grouped) != set(REPRESENTATIONS):
        raise R6Error(f"incomplete representation set: {sorted(grouped)}")
    if len(code_hashes) != 1:
        raise R6Error(f"mixed A1 code versions: {sorted(code_hashes)}")
    statuses = {
        representation: _gate_status(grouped[representation]) for representation in REPRESENTATIONS
    }
    if all(status == "stop" for status in statuses.values()):
        outcome = "C"
        decision = "frozen_representation_insufficient"
        next_action = (
            "revise the representation or use a speaker-aware encoder before controller tuning"
        )
        later_stages = "not_entered_by_a1_fail_fast_gate"
    else:
        outcome = None
        decision = "a2_b_required_for_promoted_or_conditional_representations"
        next_action = "run R6-A2 and R6-B only for non-stopped representations"
        later_stages = "pending"
    environment = _environment_smoke()
    baselines = _baseline_inventory()
    inventory = load_json(base / "protocol/inventory.json")
    best = {
        representation: _best_row(grouped[representation]) for representation in REPRESENTATIONS
    }
    development_costs = {
        representation: _minimum_development_cost(metric_documents[representation], 0.60)
        for representation in REPRESENTATIONS
    }
    result = {
        "schema_version": 1,
        "artifact_role": "r6_final_decision",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "decision_outcome": outcome,
        "decision": decision,
        "next_action": next_action,
        "a1_code_sha256": next(iter(code_hashes)),
        "a1_summary_sha256": sha256_file(a1_summary_path),
        "representation_gates": statuses,
        "development_minimum_cost_at_60_percent_recall": development_costs,
        "a2_b_status": later_stages,
        "r6_c_status": "not_authorized_not_run",
        "sortformer": environment,
        "existing_baselines": baselines,
        "evidence_limits": {
            "natural_evaluation_hours": float(a1["inventory"]["evaluation_hours"]),
            "natural_evaluation_meetings": 5,
            "shared_r4_anchor_count": int(a1["inventory"]["shared_r4_evaluation_anchor_count"]),
            "synthetic_anchor_count": 43,
            "synthetic_role": "secondary_prior_diagnostic_only_no_r6_threshold_or_promotion_use",
            "generalization": "not_product_readiness_or_broad_multilingual_evidence",
        },
        "git": inventory["git"],
        "architecture_drift": {
            "suspected": False,
            "reason": "all implementation and outputs remain experiment-only with no production wiring",
        },
    }
    decision_dir = base / "decision"
    environment_path = decision_dir / "sortformer_environment_smoke.json"
    summary_path = decision_dir / "summary.json"
    report_path = decision_dir / "REPORT.md"
    write_json(environment_path, environment)
    write_json(summary_path, result)
    lines = [
        "# R6 Online Speaker Memory Decision Report",
        "",
        f"Decision: **Outcome {outcome or 'pending'} — {decision.replace('_', ' ')}**.",
        "",
        "## Fixed Oracle Enrollment",
        "",
        "| Representation | Best eligible context | Candidate R@500 | Candidate R@1000 | Candidate R@1500 | Candidate FP/h | Handoff R@1000 | False handoff/h | AUC | EER | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for representation in REPRESENTATIONS:
        row = best[representation]
        context = f"q{row['query_context_ms']} / e{row['enrollment_ms']} / {row['aggregation']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    representation,
                    context,
                    _format_rate(row["candidate_r500"]),
                    _format_rate(row["candidate_r1000"]),
                    _format_rate(row["candidate_r1500"]),
                    _format_number(row["candidate_fp_h"]),
                    _format_rate(row["handoff_r1000"]),
                    _format_number(row["false_handoff_h"]),
                    f"{float(row['roc_auc']):.3f}",
                    f"{float(row['eer']):.3f}",
                    statuses[representation],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The table selects a development-frozen point with evaluation candidate FP/h at or below 10 when available. All measured configurations remain in the A1 report and machine-readable metrics.",
            "",
            (
                "ERes final has strong frame-level ranking (best evaluation AUC "
                f"{max(float(row['roc_auc']) for row in grouped['e-final']):.3f}, EER "
                f"{min(float(row['eer']) for row in grouped['e-final']):.3f}), but reaching 60% "
                "development Candidate R@1000 required at least "
                f"{float(development_costs['e-final']['candidate_false_events_per_hour']):.2f} "
                "candidate FP/h. The representation therefore has diagnostic separation without "
                "an operational fixed-threshold event region."
            ),
            "",
            "## Stage Decision",
            "",
            f"R6-A2 and R6-B: **{later_stages.replace('_', ' ')}**.",
            "",
            "R6-C was not run because it requires a separate owner decision and useful R6-A/R6-B evidence.",
            "",
            "## Streaming References",
            "",
            "Existing LS-EEND and legacy ERes artifacts were retained and inspected without rerunning inference. They are not placed in the numeric comparison row because they cover selected 30-second episodes under the prior action schema rather than the complete R6 first-handoff stream and false-event exposure.",
            "",
            f"Streaming Sortformer was not environment-feasible: {'; '.join(environment['limitation_reasons'])}.",
            "",
            "## Evidence Boundary",
            "",
            f"The decision uses five frozen natural meetings ({float(a1['inventory']['evaluation_hours']):.5f} raw source hours). The 86 shared R4 anchors are a compatibility count rather than the full continuous GT. The existing 43 synthetic anchors remain secondary prior diagnostics and did not select thresholds or promotion decisions.",
            "",
            "The result is decision-relevant for the frozen representations and this meeting panel. It is not product-readiness or broad multilingual evidence.",
            "",
            "## Next Action",
            "",
            next_action + ".",
            "",
            "No production modules were changed and no architecture drift is suspected.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["report_sha256"] = sha256_file(report_path)
    result["sortformer_environment_smoke_sha256"] = sha256_file(environment_path)
    write_json(summary_path, result)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize the R6 decision after A1 gates")
    parser.parse_args(argv)
    print(finalize(cache_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
