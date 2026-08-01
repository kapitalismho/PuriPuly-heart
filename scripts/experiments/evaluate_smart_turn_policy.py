from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_PROBES_MS = (224, 416, 608)
DEFAULT_HARD_END_MS = 800
BASELINE_VAD_END_MS = 512


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ValueError("prediction artifact must be an object, record list, or records object")
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError("prediction artifact must contain at least one object record")
    return records


def _score_map(record: dict[str, Any], probes_ms: tuple[int, ...]) -> dict[int, float]:
    raw_scores = record.get("scores")
    if isinstance(raw_scores, dict):
        scores = {int(key): float(value) for key, value in raw_scores.items()}
    elif isinstance(raw_scores, list):
        scores = {}
        for item in raw_scores:
            if not isinstance(item, dict) or "silence_ms" not in item or "score" not in item:
                raise ValueError("score list entries require silence_ms and score")
            scores[int(item["silence_ms"])] = float(item["score"])
    else:
        raise ValueError("each record requires scores as an object or list")
    missing = [probe_ms for probe_ms in probes_ms if probe_ms not in scores]
    if missing:
        raise ValueError(f"record is missing probe scores: {missing}")
    if any(not math.isfinite(scores[probe_ms]) for probe_ms in probes_ms):
        raise ValueError("probe scores must be finite")
    return scores


def _latency_map(record: dict[str, Any], probes_ms: tuple[int, ...]) -> dict[int, float]:
    raw_latencies = record.get("inference_ms")
    if raw_latencies is None:
        return {}
    if isinstance(raw_latencies, dict):
        latencies = {int(key): float(value) for key, value in raw_latencies.items()}
    elif isinstance(raw_latencies, list):
        latencies = {}
        for item in raw_latencies:
            if not isinstance(item, dict) or "silence_ms" not in item or "latency_ms" not in item:
                raise ValueError("latency list entries require silence_ms and latency_ms")
            latencies[int(item["silence_ms"])] = float(item["latency_ms"])
    else:
        raise ValueError("inference_ms must be an object or list")
    if any(
        probe_ms in latencies
        and (not math.isfinite(latencies[probe_ms]) or latencies[probe_ms] < 0.0)
        for probe_ms in probes_ms
    ):
        raise ValueError("inference latencies must be finite and non-negative")
    return latencies


def summarize_policy(
    records: list[dict[str, Any]],
    *,
    threshold: float = 0.5,
    probes_ms: tuple[int, ...] = DEFAULT_PROBES_MS,
    hard_end_ms: int = DEFAULT_HARD_END_MS,
    baseline_vad_end_ms: int = BASELINE_VAD_END_MS,
) -> dict[str, Any]:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in 0.0..1.0")
    if not probes_ms or tuple(sorted(probes_ms)) != probes_ms:
        raise ValueError("probes_ms must be a non-empty ordered tuple")
    if hard_end_ms <= probes_ms[-1]:
        raise ValueError("hard_end_ms must follow the final probe")

    decisions: list[int] = []
    early_decisions: list[int] = []
    early_true = 0
    early_false = 0
    endpoint_true = 0
    hard_boundary_count = 0
    first_accept_counts = {probe_ms: 0 for probe_ms in probes_ms}
    inference_latencies: list[float] = []
    missing_inference_latency_count = 0
    decision_latencies: list[float] = []
    missing_decision_latency_count = 0

    for record in records:
        if not isinstance(record.get("endpoint_bool"), bool):
            raise ValueError("each record requires boolean endpoint_bool")
        is_endpoint = record["endpoint_bool"]
        endpoint_true += int(is_endpoint)
        scores = _score_map(record, probes_ms)
        latencies = _latency_map(record, probes_ms)
        for probe_ms in probes_ms:
            if probe_ms in latencies:
                inference_latencies.append(latencies[probe_ms])
            else:
                missing_inference_latency_count += 1
        accepted_probe = next(
            (probe_ms for probe_ms in probes_ms if scores[probe_ms] >= threshold),
            None,
        )
        if accepted_probe is None:
            decision_ms = hard_end_ms
            hard_boundary_count += 1
            decision_latencies.append(float(hard_end_ms))
        else:
            decision_ms = accepted_probe
            early_decisions.append(decision_ms)
            first_accept_counts[accepted_probe] += 1
            inference_latency = latencies.get(accepted_probe)
            if inference_latency is None:
                missing_decision_latency_count += 1
            else:
                decision_latencies.append(accepted_probe + inference_latency)
            if is_endpoint:
                early_true += 1
            else:
                early_false += 1
        decisions.append(decision_ms)

    total = len(records)
    endpoint_false = total - endpoint_true
    early_accept_count = len(early_decisions)
    decision_latency_metrics = {
        "available": missing_decision_latency_count == 0,
        "observed_count": len(decision_latencies),
        "missing_count": missing_decision_latency_count,
        "mean": (sum(decision_latencies) / len(decision_latencies) if decision_latencies else None),
        "p50": _percentile(decision_latencies, 0.50) if decision_latencies else None,
        "p95": _percentile(decision_latencies, 0.95) if decision_latencies else None,
        "mean_delta_vs_512ms_vad": (
            sum(value - baseline_vad_end_ms for value in decision_latencies)
            / len(decision_latencies)
            if decision_latencies
            else None
        ),
    }
    return {
        "records": total,
        "threshold": threshold,
        "probes_ms": list(probes_ms),
        "hard_end_ms": hard_end_ms,
        "baseline_vad_end_ms": baseline_vad_end_ms,
        "first_accept_counts": {str(key): value for key, value in first_accept_counts.items()},
        "hard_boundary_count": hard_boundary_count,
        "early_accept_count": early_accept_count,
        "early_accept_rate": early_accept_count / total,
        "early_true_count": early_true,
        "early_false_complete_count": early_false,
        "early_precision": early_true / early_accept_count if early_accept_count else None,
        "early_recall_for_endpoint_records": early_true / endpoint_true if endpoint_true else None,
        "early_false_complete_rate": early_false / endpoint_false if endpoint_false else None,
        "inference_latency_ms": {
            "observed_count": len(inference_latencies),
            "missing_count": missing_inference_latency_count,
            "mean": (
                sum(inference_latencies) / len(inference_latencies) if inference_latencies else None
            ),
            "p50": _percentile(inference_latencies, 0.50) if inference_latencies else None,
            "p95": _percentile(inference_latencies, 0.95) if inference_latencies else None,
        },
        "end_to_end_decision_latency_ms": decision_latency_metrics,
        "decision_ms": {
            "mean": sum(decisions) / total,
            "p50": _percentile(decisions, 0.50),
            "p95": _percentile(decisions, 0.95),
            "mean_delta_vs_512ms_vad": sum(decision - baseline_vad_end_ms for decision in decisions)
            / total,
        },
    }


def _percentile(values: list[float] | list[int], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hard-end-ms", type=int, default=DEFAULT_HARD_END_MS)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize_policy(
                _load_records(args.artifact),
                threshold=args.threshold,
                hard_end_ms=args.hard_end_ms,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
