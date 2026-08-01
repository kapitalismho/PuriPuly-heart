from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "livekit/eot-bench-data"
LANGUAGES = ("ko", "ja", "en", "zh")
PROBES_MS = (224, 512)
TIMEOUT_MS = 800.0
N_FOLDS = 5
REPEAT_COUNT = 50
CV_SEEDS = tuple(20260802 + index * 7919 for index in range(REPEAT_COUNT))
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_RESAMPLES = 10_000
MIN_OUTER_EOT = 20
MIN_OUTER_ELIGIBLE_HOLD = 20
MATCH_TOLERANCE = 0.005
MATCH_LATENCY_TOLERANCE_MS = 1.0
OUTER_ATTEMPTS_PER_REPEAT = 500


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        empty_fields = {
            "rejected_split_manifest.csv": (
                "language",
                "repeat",
                "seed",
                "fold",
                "status",
                "reason",
                "folds_requested",
                "folds_used",
                "attempt",
            ),
        }.get(path.name)
        path.write_text(",".join(empty_fields or ()) + "\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _map_language(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"cmn", "zh-cn", "zh_cn", "chinese"}:
        return "zh"
    return raw


def _group_key(row: dict[str, Any]) -> str:
    for key in (
        "conversation_id",
        "recording_id",
        "session_id",
        "recording_session_id",
        "user_turn_id",
        "turn_id",
        "id",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"span:{row.get('span_id', '')}"


def _turn_key(row: dict[str, Any]) -> str:
    for key in ("turn_id", "user_turn_id", "conversation_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return _group_key(row)


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return None
    return float(np.percentile(array, percentile))


def _validate_prediction_rows(rows: list[dict[str, Any]], *, language: str) -> None:
    seen_span_ids: set[str] = set()
    providers: set[str] = set()
    intra_threads: set[int] = set()
    inter_threads: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        row_language = _map_language(row.get("language"))
        if row_language != language:
            raise ValueError(
                f"prediction row {row_number}: language {row_language!r} != {language!r}"
            )
        label = str(row.get("label", "")).strip().lower()
        if label not in {"hold", "eot"}:
            raise ValueError(f"prediction row {row_number}: invalid label {label!r}")
        duration = row.get("span_duration_ms")
        if not _finite(duration) or float(duration) < 0.0:
            raise ValueError(f"prediction row {row_number}: invalid span_duration_ms")
        span_id = str(row.get("span_id", row.get("id", row_number)))
        if span_id in seen_span_ids:
            raise ValueError(f"duplicate prediction row for span {span_id!r}")
        seen_span_ids.add(span_id)
        if not _group_key(row):
            raise ValueError(f"prediction row {row_number}: no grouping key")
        provider = row.get("execution_provider")
        if provider is not None:
            providers.add(str(provider))
        if row.get("intra_op_threads") is not None:
            intra_threads.add(int(row["intra_op_threads"]))
        if row.get("inter_op_threads") is not None:
            inter_threads.add(int(row["inter_op_threads"]))
        for probe_ms in PROBES_MS:
            score_key = f"score_{probe_ms}"
            latency_key = f"inference_latency_{probe_ms}_ms"
            score_present = _finite(row.get(score_key))
            latency_present = _finite(row.get(latency_key))
            survives = float(duration) >= probe_ms - 1e-6
            if survives and (not score_present or not latency_present):
                raise ValueError(
                    f"prediction row {row_number}: {score_key} and {latency_key} are required"
                )
            if not survives and (score_present or latency_present):
                raise ValueError(
                    f"prediction row {row_number}: {score_key} is present before its probe"
                )
            if latency_present and float(row[latency_key]) < 0.0:
                raise ValueError(f"prediction row {row_number}: negative {latency_key}")
    if providers - {"CPUExecutionProvider"}:
        raise ValueError(f"{language}: non-CPU prediction provider present: {sorted(providers)}")
    if intra_threads - {2}:
        raise ValueError(f"{language}: expected two intra-op threads, got {sorted(intra_threads)}")
    if inter_threads - {1}:
        raise ValueError(f"{language}: expected one inter-op thread, got {sorted(inter_threads)}")


def _load_prediction_rows(predictions_dir: Path, language: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = predictions_dir / f"cpu_predictions_{language}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = pq.read_table(path).to_pylist()
    _validate_prediction_rows(rows, language=language)
    return rows


def validate_input_artifacts(
    predictions_dir: Path,
    languages: tuple[str, ...],
    *,
    output_dir: Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_language: dict[str, list[dict[str, Any]]] = {}
    artifacts: list[dict[str, Any]] = []
    for language in languages:
        path = predictions_dir / f"cpu_predictions_{language}.parquet"
        rows = _load_prediction_rows(predictions_dir, language)
        providers = sorted(
            {
                str(row["execution_provider"])
                for row in rows
                if row.get("execution_provider") is not None
            }
        )
        artifacts.append(
            {
                "language": language,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "rows": len(rows),
                "eot_spans": sum(row["label"] == "eot" for row in rows),
                "hold_spans": sum(row["label"] == "hold" for row in rows),
                "groups": len({_group_key(row) for row in rows}),
                "score_224": sum(_finite(row.get("score_224")) for row in rows),
                "score_512": sum(_finite(row.get("score_512")) for row in rows),
                "execution_providers": providers,
                "intra_op_threads": sorted(
                    {
                        int(row["intra_op_threads"])
                        for row in rows
                        if row.get("intra_op_threads") is not None
                    }
                ),
                "inter_op_threads": sorted(
                    {
                        int(row["inter_op_threads"])
                        for row in rows
                        if row.get("inter_op_threads") is not None
                    }
                ),
            }
        )
        rows_by_language[language] = rows
    validation = {
        "valid": True,
        "mode": "policy_only",
        "reused_existing_cpu_prediction_artifacts": True,
        "regenerated_predictions": False,
        "prediction_directory": str(predictions_dir.resolve()),
        "languages": list(languages),
        "artifacts": artifacts,
        "required_fields": [
            "language",
            "conversation_id",
            "turn_id",
            "span_id",
            "label",
            "span_duration_ms",
            "score_224",
            "inference_latency_224_ms",
            "score_512",
            "inference_latency_512_ms",
        ],
        "runtime_contract": {
            "model": "Smart Turn v3.2 CPU int8",
            "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": 2,
            "inter_op_threads": 1,
            "execution_mode": "sequential",
        },
    }
    if output_dir is not None:
        _write_json(output_dir / "input_validation.json", validation)
    return rows_by_language, validation


def _array_data(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "duration": np.asarray([float(row["span_duration_ms"]) for row in rows], dtype=np.float64),
        "hold": np.asarray([row["label"] == "hold" for row in rows], dtype=bool),
        "eot": np.asarray([row["label"] == "eot" for row in rows], dtype=bool),
        "score224": np.asarray(
            [float(row["score_224"]) if _finite(row.get("score_224")) else np.nan for row in rows],
            dtype=np.float64,
        ),
        "score512": np.asarray(
            [float(row["score_512"]) if _finite(row.get("score_512")) else np.nan for row in rows],
            dtype=np.float64,
        ),
        "lat224": np.asarray(
            [
                (
                    float(row["inference_latency_224_ms"])
                    if _finite(row.get("inference_latency_224_ms"))
                    else np.nan
                )
                for row in rows
            ],
            dtype=np.float64,
        ),
        "lat512": np.asarray(
            [
                (
                    float(row["inference_latency_512_ms"])
                    if _finite(row.get("inference_latency_512_ms"))
                    else np.nan
                )
                for row in rows
            ],
            dtype=np.float64,
        ),
        "groups": np.asarray([_group_key(row) for row in rows], dtype=object),
        "turns": np.asarray([_turn_key(row) for row in rows], dtype=object),
    }


def _policy_trace(
    rows: list[dict[str, Any]],
    policy: str,
    threshold224: float | None = None,
    threshold512: float | None = None,
    *,
    array_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    data = _array_data(rows) if array_data is None else array_data
    duration = data["duration"]
    score224 = data["score224"]
    score512 = data["score512"]
    lat224 = data["lat224"]
    lat512 = data["lat512"]
    has224 = np.isfinite(score224) & np.isfinite(lat224)
    has512 = np.isfinite(score512) & np.isfinite(lat512)
    arrival224 = 224.0 + np.where(has224, lat224, np.inf)
    first_valid = np.zeros(len(rows), dtype=bool)
    second_scheduled = np.zeros(len(rows), dtype=bool)
    second_valid = np.zeros(len(rows), dtype=bool)
    stale224 = np.zeros(len(rows), dtype=bool)
    stale512 = np.zeros(len(rows), dtype=bool)
    overlap = np.zeros(len(rows), dtype=bool)
    first_arrival = arrival224
    second_start = np.maximum(512.0, arrival224)
    second_arrival = second_start + np.where(has512, lat512, np.inf)
    if policy == "B0":
        decision = np.full(len(rows), 512.0, dtype=np.float64)
        probe = np.full(len(rows), "512ms_fixed", dtype=object)
        eligible = duration > 512.0
        accepted_any = np.zeros(len(rows), dtype=bool)
    elif policy == "B1":
        decision = np.full(len(rows), TIMEOUT_MS, dtype=np.float64)
        probe = np.full(len(rows), "timeout", dtype=object)
        eligible = duration > TIMEOUT_MS
        accepted_any = np.zeros(len(rows), dtype=bool)
    elif policy in {"P1", "P2", "P3"}:
        if threshold224 is None:
            raise ValueError(f"{policy} requires threshold224")
        first_valid = (
            has224
            & (duration > first_arrival)
            & (first_arrival < TIMEOUT_MS)
            & (score224 > float(threshold224))
        )
        stale224 = has224 & ((duration <= first_arrival) | (first_arrival >= TIMEOUT_MS))
        if policy in {"P2", "P3"}:
            if threshold512 is None:
                raise ValueError(f"{policy} requires threshold512")
            second_scheduled = (
                has512 & (duration > 512.0) & ~(first_valid & (first_arrival <= 512.0))
            )
            overlap = second_scheduled & (first_arrival > 512.0)
            second_valid = (
                second_scheduled
                & (duration > second_arrival)
                & (second_arrival < TIMEOUT_MS)
                & (score512 > float(threshold512))
            )
            stale512 = second_scheduled & (
                (duration <= second_arrival) | (second_arrival >= TIMEOUT_MS)
            )
        winner224 = first_valid & ~(second_valid & (second_arrival < first_arrival))
        winner512 = second_valid & ~(first_valid & (first_arrival <= second_arrival))
        decision = np.full(len(rows), TIMEOUT_MS, dtype=np.float64)
        probe = np.full(len(rows), "timeout", dtype=object)
        decision[winner224] = first_arrival[winner224]
        probe[winner224] = "224ms"
        decision[winner512] = second_arrival[winner512]
        probe[winner512] = "512ms"
        accepted_any = first_valid | second_valid
        possible_arrivals = []
        if np.any(has224):
            possible_arrivals.append(first_arrival)
        if policy in {"P2", "P3"} and np.any(second_scheduled):
            possible_arrivals.append(second_arrival)
        if possible_arrivals:
            earliest = np.minimum.reduce(possible_arrivals)
            eligible = (
                data["hold"]
                & np.isfinite(earliest)
                & (duration > earliest)
                & (earliest < TIMEOUT_MS)
            )
        else:
            eligible = np.zeros(len(rows), dtype=bool)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    if policy in {"B0", "B1"}:
        winner224 = np.zeros(len(rows), dtype=bool)
        winner512 = np.zeros(len(rows), dtype=bool)
    return {
        "decision_ms": decision,
        "probe": probe,
        "first_valid": first_valid,
        "second_scheduled": second_scheduled,
        "second_valid": second_valid,
        "winner224": winner224,
        "winner512": winner512,
        "accepted_any": accepted_any,
        "stale224": stale224,
        "stale512": stale512,
        "overlap": overlap,
        "eligible_hold": eligible,
        "first_arrival": first_arrival,
        "second_arrival": second_arrival,
    }


def _trace_metrics(
    rows: list[dict[str, Any]],
    trace: dict[str, Any],
    *,
    array_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    data = _array_data(rows) if array_data is None else array_data
    decision = trace["decision_ms"]
    hold = data["hold"]
    eot = data["eot"]
    false_cut = hold & (data["duration"] > decision)
    eligible_hold = hold & trace["eligible_hold"]
    eot_latencies = decision[eot]
    hard_timeout = decision >= TIMEOUT_MS - 1e-6
    unresolved = ~trace["accepted_any"]
    attempts = int(np.isfinite(data["score224"]).sum() + trace["second_scheduled"].sum())
    false_by_turn: dict[str, int] = defaultdict(int)
    turns: set[str] = set()
    for turn, is_false in zip(data["turns"], false_cut, strict=True):
        turn_key = str(turn)
        turns.add(turn_key)
        if is_false:
            false_by_turn[turn_key] += 1
    affected = [turn for turn, count in false_by_turn.items() if count]
    false_count = int(false_cut.sum())
    hold_count = int(hold.sum())
    eligible_count = int(eligible_hold.sum())
    eot_count = int(eot.sum())
    stale_count = int(trace["stale224"].sum() + trace["stale512"].sum())
    return {
        "n_spans": len(rows),
        "eot_spans": eot_count,
        "hold_spans": hold_count,
        "false_cutoffs": false_count,
        "false_cutoff_rate": false_count / hold_count if hold_count else 0.0,
        "eligible_hold_spans": eligible_count,
        "eligible_false_cutoffs": int((false_cut & eligible_hold).sum()),
        "eligible_false_cutoff_rate": (
            int((false_cut & eligible_hold).sum()) / eligible_count if eligible_count else 0.0
        ),
        "mean_endpoint_latency_ms": float(np.mean(eot_latencies)) if eot_latencies.size else None,
        "p50_endpoint_latency_ms": _percentile(eot_latencies, 50),
        "p90_endpoint_latency_ms": _percentile(eot_latencies, 90),
        "p95_endpoint_latency_ms": _percentile(eot_latencies, 95),
        "p99_endpoint_latency_ms": _percentile(eot_latencies, 99),
        "acceptance_224_rate": float(trace["first_valid"].mean()) if len(rows) else 0.0,
        "acceptance_512_rate": float(trace["second_valid"].mean()) if len(rows) else 0.0,
        "hard_timeout_rate": float(hard_timeout.mean()) if len(rows) else 0.0,
        "eot_timeout_rate": (float((eot & hard_timeout).sum() / eot_count) if eot_count else 0.0),
        "unresolved_span_rate": float(unresolved.mean()) if len(rows) else 0.0,
        "eot_early_detection_rate": (
            float((eot & ~hard_timeout).sum() / eot_count) if eot_count else 0.0
        ),
        "probe_overlap_rate": (
            float(trace["overlap"].sum() / trace["second_scheduled"].sum())
            if trace["second_scheduled"].sum()
            else 0.0
        ),
        "probe_overlap_count": int(trace["overlap"].sum()),
        "second_scheduled_count": int(trace["second_scheduled"].sum()),
        "stale_result_rate": float(stale_count / attempts) if attempts else 0.0,
        "stale_result_count": stale_count,
        "turns": len(turns),
        "turns_with_false_cutoff": len(affected),
        "turn_fragmentation_rate": len(affected) / len(turns) if turns else 0.0,
        "false_splits_per_100_turns": false_count / len(turns) * 100.0 if turns else 0.0,
        "mean_false_cutoffs_per_affected_turn": false_count / len(affected) if affected else 0.0,
        "unresolved_spans": int(unresolved.sum()),
    }


def simulate_policy(
    rows: list[dict[str, Any]],
    policy: str,
    threshold224: float | None = None,
    threshold512: float | None = None,
) -> dict[str, Any]:
    trace = _policy_trace(rows, policy, threshold224, threshold512)
    return _trace_metrics(rows, trace)


def _threshold_grid() -> list[float]:
    values = [i / 100.0 for i in range(91)]
    values.extend(round(0.90 + i * 0.002, 6) for i in range(1, 46))
    values.extend(round(0.99 + i * 0.0005, 6) for i in range(1, 21))
    return sorted(set(value for value in values if value <= 1.0))


def _candidate_thresholds(
    rows: list[dict[str, Any]], key: str, *, exact_limit: int = 2500
) -> tuple[list[float], str]:
    del exact_limit
    values = sorted({round(float(row[key]), 9) for row in rows if _finite(row.get(key))})
    return sorted(set([0.0, 1.0, *values])), "observed_inner_training_scores"


def _relative_reduction(baseline: float, value: float) -> float:
    return (baseline - value) / baseline if baseline else 0.0


def _candidate_is_valid(
    metrics: dict[str, Any], baseline: dict[str, Any], target_reduction: float
) -> bool:
    return (
        _relative_reduction(
            float(baseline["false_cutoff_rate"]), float(metrics["false_cutoff_rate"])
        )
        >= target_reduction - 1e-12
        and float(metrics["false_cutoff_rate"]) <= float(baseline["false_cutoff_rate"]) + 1e-12
        and float(metrics["mean_endpoint_latency_ms"] or math.inf) <= 600.0
        and float(metrics["p50_endpoint_latency_ms"] or math.inf) <= 600.0
        and float(metrics["eot_timeout_rate"]) <= 0.25 + 1e-12
    )


def _candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = item["metrics"]
    return (
        float(metrics["mean_endpoint_latency_ms"] or math.inf),
        float(metrics["false_cutoff_rate"]),
        float(metrics["eot_timeout_rate"]),
        -float(item.get("threshold224", 0.0)),
        -float(item.get("threshold512") or 0.0),
    )


def _candidate_metrics(
    rows: list[dict[str, Any]],
    policy: str,
    threshold224: float,
    threshold512: float | None = None,
    *,
    array_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    data = _array_data(rows) if array_data is None else array_data
    trace = _policy_trace(rows, policy, threshold224, threshold512, array_data=data)
    return _trace_metrics(rows, trace, array_data=data)


def _candidate_metrics_for_selection(
    rows: list[dict[str, Any]],
    policy: str,
    threshold224: float,
    threshold512: float | None,
    *,
    array_data: dict[str, np.ndarray],
) -> dict[str, Any]:
    trace = _policy_trace(rows, policy, threshold224, threshold512, array_data=array_data)
    hold = array_data["hold"]
    eot = array_data["eot"]
    decision = trace["decision_ms"]
    false_cutoffs = int((hold & (array_data["duration"] > decision)).sum())
    eot_count = int(eot.sum())
    return {
        "false_cutoff_rate": false_cutoffs / int(hold.sum()) if hold.sum() else 0.0,
        "mean_endpoint_latency_ms": (float(np.mean(decision[eot])) if eot_count else None),
        "p50_endpoint_latency_ms": (float(np.percentile(decision[eot], 50)) if eot_count else None),
        "eot_timeout_rate": (
            float((eot & (decision >= TIMEOUT_MS - 1e-6)).sum() / eot_count) if eot_count else 0.0
        ),
    }


def _enumerate_candidates(
    rows: list[dict[str, Any]], policy: str, *, evaluate: bool = True
) -> tuple[list[dict[str, Any]], str]:
    array_data = _array_data(rows)
    if evaluate:

        def empty_or_metrics(threshold224, threshold512=None):
            return _candidate_metrics(
                rows,
                policy,
                threshold224,
                threshold512,
                array_data=array_data,
            )

    else:

        def empty_or_metrics(threshold224, threshold512=None):
            return {}

    thresholds224, source224 = _candidate_thresholds(rows, "score_224")
    if policy == "P1":
        return [
            {
                "threshold224": threshold,
                "threshold512": None,
                "metrics": empty_or_metrics(threshold),
            }
            for threshold in thresholds224
        ], source224
    if policy not in {"P2", "P3"}:
        raise ValueError(policy)
    thresholds512, source512 = _candidate_thresholds(rows, "score_512")
    values = sorted(set(thresholds224 + thresholds512))
    if policy == "P2":
        return [
            {
                "threshold224": threshold,
                "threshold512": threshold,
                "metrics": empty_or_metrics(threshold, threshold),
            }
            for threshold in values
        ], f"shared_{source224}_{source512}"
    candidates = []
    for threshold224 in values:
        for threshold512 in values:
            if threshold512 + 1e-12 < threshold224:
                continue
            candidates.append(
                {
                    "threshold224": threshold224,
                    "threshold512": threshold512,
                    "metrics": empty_or_metrics(threshold224, threshold512),
                }
            )
    return candidates, f"independent_{source224}_{source512}"


def _selection_tie_metadata(
    candidates: list[dict[str, Any]], selected: dict[str, Any]
) -> dict[str, Any]:
    if len(candidates) == 1:
        return {
            "selection_tie_count": 1,
            "selection_tie_reason": "unique valid candidate",
        }
    metrics = [candidate["metrics"] for candidate in candidates]
    best_latency = min(
        float(metric.get("mean_endpoint_latency_ms") or math.inf) for metric in metrics
    )
    latency_ties = [
        candidate
        for candidate in candidates
        if abs(
            float(candidate["metrics"].get("mean_endpoint_latency_ms") or math.inf) - best_latency
        )
        <= 1e-9
    ]
    if len(latency_ties) == 1:
        reason = "lower runtime-aware mean endpoint latency"
        tied = latency_ties
    else:
        best_false = min(
            float(candidate["metrics"]["false_cutoff_rate"]) for candidate in latency_ties
        )
        false_ties = [
            candidate
            for candidate in latency_ties
            if abs(float(candidate["metrics"]["false_cutoff_rate"]) - best_false) <= 1e-12
        ]
        if len(false_ties) == 1:
            reason = "lower runtime-aware mean endpoint latency, then lower false-cutoff rate"
            tied = false_ties
        else:
            best_timeout = min(
                float(candidate["metrics"]["eot_timeout_rate"]) for candidate in false_ties
            )
            timeout_ties = [
                candidate
                for candidate in false_ties
                if abs(float(candidate["metrics"]["eot_timeout_rate"]) - best_timeout) <= 1e-12
            ]
            if len(timeout_ties) == 1:
                reason = (
                    "lower runtime-aware mean endpoint latency, then lower false-cutoff rate, "
                    "then lower EOT timeout rate"
                )
                tied = timeout_ties
            else:
                highest_threshold = max(
                    float(candidate.get("threshold224") or 0.0) for candidate in timeout_ties
                )
                threshold_ties = [
                    candidate
                    for candidate in timeout_ties
                    if abs(float(candidate.get("threshold224") or 0.0) - highest_threshold) <= 1e-12
                ]
                if len(threshold_ties) == 1:
                    reason = (
                        "lower runtime-aware mean endpoint latency, then lower false-cutoff rate, "
                        "then lower EOT timeout rate, then higher threshold"
                    )
                    tied = threshold_ties
                else:
                    reason = (
                        "lower runtime-aware mean endpoint latency, then lower false-cutoff rate, "
                        "then lower EOT timeout rate, then deterministic threshold ordering"
                    )
                    tied = threshold_ties
    return {
        "selection_tie_count": len(tied),
        "selection_tie_reason": reason,
    }


def _select_candidate(
    candidates: list[dict[str, Any]], baseline: dict[str, Any], target_reduction: float
) -> dict[str, Any] | None:
    valid = [
        item
        for item in candidates
        if _candidate_is_valid(item["metrics"], baseline, target_reduction)
    ]
    if not valid:
        return None
    selected = min(valid, key=_candidate_sort_key)
    return selected | {
        "valid_candidate_count": len(valid),
        **_selection_tie_metadata(valid, selected),
    }


def _split_groups(
    rows: list[dict[str, Any]], seed: int, n_folds: int
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]]:
    groups = sorted({_group_key(row) for row in rows})
    if not groups or n_folds < 2 or len(groups) < n_folds:
        return []
    shuffled = groups[:]
    random.Random(seed).shuffle(shuffled)
    assignments = {group: index % n_folds for index, group in enumerate(shuffled)}
    return [
        (
            fold,
            [
                row
                for row in rows
                if _group_key(row) not in {g for g, f in assignments.items() if f == fold}
            ],
            [
                row
                for row in rows
                if _group_key(row) in {g for g, f in assignments.items() if f == fold}
            ],
        )
        for fold in range(n_folds)
    ]


def _group_splits(
    rows: list[dict[str, Any]], seed: int
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]]:
    groups = len({_group_key(row) for row in rows})
    return _split_groups(rows, seed, min(N_FOLDS, groups))


def _eligible_hold_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if row["label"] != "hold":
            continue
        duration = float(row["span_duration_ms"])
        arrivals = []
        if _finite(row.get("inference_latency_224_ms")):
            arrivals.append(224.0 + float(row["inference_latency_224_ms"]))
        if _finite(row.get("inference_latency_512_ms")):
            first_arrival = (
                224.0 + float(row["inference_latency_224_ms"])
                if _finite(row.get("inference_latency_224_ms"))
                else 224.0
            )
            arrivals.append(max(512.0, first_arrival) + float(row["inference_latency_512_ms"]))
        if any(arrival < TIMEOUT_MS and duration > arrival for arrival in arrivals):
            count += 1
    return count


def _split_reason(test_rows: list[dict[str, Any]]) -> str | None:
    eot_count = sum(row["label"] == "eot" for row in test_rows)
    eligible_hold = _eligible_hold_count(test_rows)
    if eot_count < MIN_OUTER_EOT:
        return f"test fold has {eot_count} true EOT spans; minimum is {MIN_OUTER_EOT}"
    if eligible_hold < MIN_OUTER_ELIGIBLE_HOLD:
        return (
            f"test fold has {eligible_hold} eligible hold spans; "
            f"minimum is {MIN_OUTER_ELIGIBLE_HOLD}"
        )
    return None


def _outer_splits(
    rows: list[dict[str, Any]],
    language: str,
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    group_count = len({_group_key(row) for row in rows})
    n_folds = next((folds for folds in (5, 4, 3) if group_count >= folds), 0)
    accepted: list[dict[str, Any]] = []
    if not n_folds:
        rejected.append(
            {
                "language": language,
                "repeat": None,
                "seed": None,
                "fold": None,
                "status": "rejected",
                "reason": f"only {group_count} grouping units; at least 3 are required",
                "folds_requested": 5,
                "folds_used": 0,
            }
        )
        return accepted
    total_eot = sum(row["label"] == "eot" for row in rows)
    total_eligible_hold = _eligible_hold_count(rows)
    if (
        total_eot < MIN_OUTER_EOT * n_folds
        or total_eligible_hold < MIN_OUTER_ELIGIBLE_HOLD * n_folds
    ):
        rejected.append(
            {
                "language": language,
                "repeat": None,
                "seed": None,
                "fold": None,
                "status": "unavailable",
                "reason": (
                    f"dataset has {total_eot} EOT and {total_eligible_hold} eligible hold spans; "
                    f"{n_folds} folds require at least "
                    f"{MIN_OUTER_EOT * n_folds} EOT and {MIN_OUTER_ELIGIBLE_HOLD * n_folds} eligible hold spans"
                ),
                "folds_requested": 5,
                "folds_used": n_folds,
            }
        )
        return accepted
    for repeat_index, base_seed in enumerate(CV_SEEDS):
        found = False
        for attempt in range(OUTER_ATTEMPTS_PER_REPEAT):
            seed = int(base_seed + attempt * 1_000_003)
            splits = _split_groups(rows, seed, n_folds)
            invalid = [(fold, _split_reason(test_rows)) for fold, _, test_rows in splits]
            invalid = [(fold, reason) for fold, reason in invalid if reason]
            if invalid:
                rejected.extend(
                    {
                        "language": language,
                        "repeat": repeat_index,
                        "seed": seed,
                        "fold": fold,
                        "status": "rejected",
                        "reason": reason,
                        "folds_requested": 5,
                        "folds_used": n_folds,
                        "attempt": attempt,
                    }
                    for fold, reason in invalid
                )
                continue
            for fold, train_rows, test_rows in splits:
                accepted.append(
                    {
                        "language": language,
                        "repeat": repeat_index,
                        "seed": seed,
                        "fold": fold,
                        "train_rows": train_rows,
                        "test_rows": test_rows,
                        "train_groups": len({_group_key(row) for row in train_rows}),
                        "test_groups": len({_group_key(row) for row in test_rows}),
                        "test_eot": sum(row["label"] == "eot" for row in test_rows),
                        "test_eligible_hold": _eligible_hold_count(test_rows),
                        "folds_requested": 5,
                        "folds_used": n_folds,
                        "fold_reduced": n_folds != 5,
                    }
                )
            found = True
            break
        if not found:
            rejected.append(
                {
                    "language": language,
                    "repeat": repeat_index,
                    "seed": base_seed,
                    "fold": None,
                    "status": "unavailable",
                    "reason": "no deterministic valid split found within attempt limit",
                    "folds_requested": 5,
                    "folds_used": n_folds,
                }
            )
    return accepted


def _inner_select(
    rows: list[dict[str, Any]],
    policy: str,
    target: str,
    seed: int,
) -> dict[str, Any] | None:
    prepared = _prepare_inner_selection(rows, policy, seed)
    return _select_prepared_inner(prepared, target)


def _prepare_inner_selection(
    rows: list[dict[str, Any]],
    policy: str,
    seed: int,
) -> dict[str, Any] | None:
    groups = len({_group_key(row) for row in rows})
    n_folds = min(N_FOLDS, groups)
    if n_folds < 2:
        return None
    inner_splits = _split_groups(rows, seed + 104_729, n_folds)
    if not inner_splits:
        return None
    inner_training_rows = [row for _, train, _ in inner_splits for row in train]
    validation_rows = [row for _, _, validation in inner_splits for row in validation]
    validation_data = _array_data(validation_rows)
    candidates, source = _enumerate_candidates(
        inner_training_rows,
        policy,
        evaluate=False,
    )
    baseline = simulate_policy(validation_rows, "B0")
    evaluated = [
        item
        | {
            "metrics": _candidate_metrics_for_selection(
                validation_rows,
                policy,
                item["threshold224"],
                item["threshold512"],
                array_data=validation_data,
            )
        }
        for item in candidates
    ]
    return {
        "candidates": evaluated,
        "baseline": baseline,
        "threshold_source": source,
        "inner_folds": n_folds,
        "inner_validation_rows": len(validation_rows),
        "inner_training_rows": len(inner_training_rows),
        "candidate_count": len(candidates),
    }


def _select_prepared_inner(
    prepared: dict[str, Any] | None,
    target: str,
) -> dict[str, Any] | None:
    if prepared is None:
        return None
    target_reduction = 0.20 if target == "low_latency" else 0.35
    selected = _select_candidate(
        prepared["candidates"],
        prepared["baseline"],
        target_reduction,
    )
    if selected is None:
        return None
    return selected | {
        "threshold_source": prepared["threshold_source"],
        "inner_folds": prepared["inner_folds"],
        "inner_validation_rows": prepared["inner_validation_rows"],
        "inner_training_rows": prepared["inner_training_rows"],
        "candidate_count": prepared["candidate_count"],
        "inner_false_cutoff_rate": selected["metrics"]["false_cutoff_rate"],
        "inner_mean_endpoint_latency_ms": selected["metrics"]["mean_endpoint_latency_ms"],
        "inner_eot_timeout_rate": selected["metrics"]["eot_timeout_rate"],
        "target": target,
        "selection_kind": "selected",
    }


METRIC_FIELDS = (
    "n_spans",
    "eot_spans",
    "hold_spans",
    "eligible_hold_spans",
    "false_cutoffs",
    "false_cutoff_rate",
    "eligible_false_cutoffs",
    "eligible_false_cutoff_rate",
    "relative_false_cutoff_reduction",
    "mean_endpoint_latency_ms",
    "p50_endpoint_latency_ms",
    "p90_endpoint_latency_ms",
    "p95_endpoint_latency_ms",
    "p99_endpoint_latency_ms",
    "acceptance_224_rate",
    "acceptance_512_rate",
    "eot_timeout_rate",
    "unresolved_span_rate",
    "stale_result_rate",
    "probe_overlap_rate",
    "turns_with_false_cutoff",
    "turn_fragmentation_rate",
    "false_splits_per_100_turns",
    "mean_false_cutoffs_per_affected_turn",
    "hard_timeout_rate",
)


def _metric_columns(
    metrics: dict[str, Any], baseline: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = {field: metrics.get(field) for field in METRIC_FIELDS}
    if baseline is not None:
        result["relative_false_cutoff_reduction"] = _relative_reduction(
            float(baseline["false_cutoff_rate"]), float(metrics["false_cutoff_rate"])
        )
    return result


def _cv_row(
    *,
    split: dict[str, Any],
    target: str,
    policy: str,
    selection: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    baseline: dict[str, Any],
    status: str,
    selection_kind: str,
) -> dict[str, Any]:
    row = {
        "language": split["language"],
        "repeat": split["repeat"],
        "outer_seed": split["seed"],
        "outer_fold": split["fold"],
        "target": target,
        "policy": policy,
        "selection_kind": selection_kind,
        "status": status,
        "train_rows": len(split["train_rows"]),
        "test_rows": len(split["test_rows"]),
        "train_groups": split["train_groups"],
        "test_groups": split["test_groups"],
        "test_eot_spans": split["test_eot"],
        "test_eligible_hold_spans": split["test_eligible_hold"],
        "threshold224": selection.get("threshold224") if selection else None,
        "threshold512": selection.get("threshold512") if selection else None,
        "threshold_source": selection.get("threshold_source") if selection else None,
        "inner_folds": selection.get("inner_folds") if selection else None,
        "inner_training_rows": selection.get("inner_training_rows") if selection else None,
        "inner_validation_rows": selection.get("inner_validation_rows") if selection else None,
        "candidate_count": selection.get("candidate_count") if selection else None,
        "inner_false_cutoff_rate": (
            selection.get("inner_false_cutoff_rate") if selection else None
        ),
        "inner_mean_endpoint_latency_ms": (
            selection.get("inner_mean_endpoint_latency_ms") if selection else None
        ),
        "inner_eot_timeout_rate": (selection.get("inner_eot_timeout_rate") if selection else None),
        "valid_candidate_count": selection.get("valid_candidate_count") if selection else None,
        "selection_tie_count": selection.get("selection_tie_count") if selection else None,
        "selection_tie_reason": selection.get("selection_tie_reason") if selection else None,
        "selection_mode": selection.get("selection_mode") if selection else None,
        "matched_target_training_false_cutoff": (
            selection.get("matched_target_training_false_cutoff") if selection else None
        ),
        "matched_training_false_cutoff_difference": (
            selection.get("matched_training_false_cutoff_difference") if selection else None
        ),
        "matched_target_training_mean_latency_ms": (
            selection.get("matched_target_training_mean_latency_ms") if selection else None
        ),
        "matched_training_mean_latency_difference_ms": (
            selection.get("matched_training_mean_latency_difference_ms") if selection else None
        ),
        "target_available": bool(selection),
    }
    row.update(
        _metric_columns(metrics, baseline)
        if metrics is not None
        else {field: None for field in METRIC_FIELDS}
    )
    return row


def _paired_row(
    *,
    p1: dict[str, Any] | None,
    p2: dict[str, Any] | None,
    split: dict[str, Any],
    target: str,
) -> dict[str, Any]:
    result = {
        "language": split["language"],
        "repeat": split["repeat"],
        "outer_seed": split["seed"],
        "outer_fold": split["fold"],
        "target": target,
        "status": "available" if p1 and p2 else "unavailable_candidate",
        "p1_threshold224": p1.get("threshold224") if p1 else None,
        "p2_threshold224": p2.get("threshold224") if p2 else None,
        "p2_threshold512": p2.get("threshold512") if p2 else None,
    }
    if not p1 or not p2:
        result.update(
            {
                "p2_mean_endpoint_minus_p1_ms": None,
                "p2_false_cutoff_minus_p1": None,
                "p2_eot_timeout_minus_p1": None,
                "p2_turn_fragmentation_minus_p1": None,
                "value_condition": None,
                "false_cutoff_regression_ok": None,
            }
        )
        return result
    latency_delta = float(p2["mean_endpoint_latency_ms"]) - float(p1["mean_endpoint_latency_ms"])
    false_delta = float(p2["false_cutoff_rate"]) - float(p1["false_cutoff_rate"])
    timeout_delta = float(p2["eot_timeout_rate"]) - float(p1["eot_timeout_rate"])
    fragmentation_delta = float(p2["turn_fragmentation_rate"]) - float(
        p1["turn_fragmentation_rate"]
    )
    result.update(
        {
            "p2_mean_endpoint_minus_p1_ms": latency_delta,
            "p2_false_cutoff_minus_p1": false_delta,
            "p2_eot_timeout_minus_p1": timeout_delta,
            "p2_turn_fragmentation_minus_p1": fragmentation_delta,
            "value_condition": latency_delta <= -20.0 or timeout_delta <= -0.05,
            "false_cutoff_regression_ok": false_delta <= MATCH_TOLERANCE,
        }
    )
    return result


def _matched_candidate(
    candidates: list[dict[str, Any]], target_training_false_cutoff: float
) -> dict[str, Any] | None:
    target_training_false_cutoff = round(float(target_training_false_cutoff), 12)
    available = [
        item for item in candidates if _finite(item.get("metrics", {}).get("false_cutoff_rate"))
    ]
    if not available:
        return None
    selected = min(
        available,
        key=lambda item: (
            abs(float(item["metrics"]["false_cutoff_rate"]) - target_training_false_cutoff),
            float(item["metrics"].get("mean_endpoint_latency_ms") or math.inf),
        ),
    )
    selected_training_false_cutoff = round(
        float(selected["metrics"]["false_cutoff_rate"]),
        12,
    )
    difference = round(
        abs(selected_training_false_cutoff - target_training_false_cutoff),
        12,
    )
    if difference > MATCH_TOLERANCE:
        return None
    assert (
        abs(round(selected_training_false_cutoff - target_training_false_cutoff, 12))
        <= MATCH_TOLERANCE
    )
    return selected | {
        "matched_target_training_false_cutoff": target_training_false_cutoff,
        "matched_training_false_cutoff_difference": difference,
        "matched_status": "matched",
    }


def _select_p3_prepared_inner(
    prepared: dict[str, Any] | None,
    target: str,
    p2_selection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if prepared is None or p2_selection is None:
        return None
    target_reduction = 0.20 if target == "low_latency" else 0.35
    valid = [
        item
        for item in prepared["candidates"]
        if _candidate_is_valid(item["metrics"], prepared["baseline"], target_reduction)
    ]
    if not valid:
        return None
    target_false = float(p2_selection["inner_false_cutoff_rate"])
    target_latency = float(p2_selection["inner_mean_endpoint_latency_ms"] or math.inf)
    false_matched = [
        item
        for item in valid
        if abs(float(item["metrics"]["false_cutoff_rate"]) - target_false) <= MATCH_TOLERANCE
    ]
    latency_matched = [
        item
        for item in valid
        if abs(float(item["metrics"].get("mean_endpoint_latency_ms") or math.inf) - target_latency)
        <= MATCH_LATENCY_TOLERANCE_MS
    ]
    false_gain = [
        item
        for item in false_matched
        if float(item["metrics"].get("mean_endpoint_latency_ms") or math.inf)
        <= target_latency - 15.0
    ]
    latency_gain = [
        item
        for item in latency_matched
        if float(item["metrics"]["false_cutoff_rate"]) <= target_false - 0.01
    ]
    if false_gain:
        selected = min(false_gain, key=_candidate_sort_key)
        matched = _matched_candidate(false_matched, target_false)
        if matched is None:
            return None
        mode = "false_cutoff_matched"
        selected = (
            matched
            if selected == matched
            else selected
            | {
                "matched_target_training_false_cutoff": target_false,
                "matched_training_false_cutoff_difference": abs(
                    float(selected["metrics"]["false_cutoff_rate"]) - target_false
                ),
                "matched_status": "matched",
            }
        )
        matched_latency_difference = (
            float(selected["metrics"].get("mean_endpoint_latency_ms") or math.inf) - target_latency
        )
    elif latency_gain:
        selected = min(
            latency_gain,
            key=lambda item: (
                float(item["metrics"]["false_cutoff_rate"]),
                *_candidate_sort_key(item),
            ),
        )
        matched_latency_difference = abs(
            float(selected["metrics"].get("mean_endpoint_latency_ms") or math.inf) - target_latency
        )
        assert matched_latency_difference <= MATCH_LATENCY_TOLERANCE_MS
        selected = selected | {
            "matched_target_training_mean_latency_ms": target_latency,
            "matched_training_mean_latency_difference_ms": matched_latency_difference,
            "matched_status": "matched_latency",
        }
        mode = "mean_latency_matched"
    else:
        return None
    return selected | {
        "threshold_source": prepared["threshold_source"],
        "inner_folds": prepared["inner_folds"],
        "inner_validation_rows": prepared["inner_validation_rows"],
        "inner_training_rows": prepared["inner_training_rows"],
        "candidate_count": prepared["candidate_count"],
        "valid_candidate_count": len(valid),
        "inner_false_cutoff_rate": selected["metrics"]["false_cutoff_rate"],
        "inner_mean_endpoint_latency_ms": selected["metrics"]["mean_endpoint_latency_ms"],
        "inner_eot_timeout_rate": selected["metrics"]["eot_timeout_rate"],
        "target": target,
        "selection_kind": "selected",
        "selection_mode": mode,
        **_selection_tie_metadata(
            false_gain if mode == "false_cutoff_matched" else latency_gain,
            selected,
        ),
    }


def _increment_row(
    *,
    language: str,
    seed: int,
    fold: int,
    target: str,
    comparison: str,
    train_reference: dict[str, Any],
    train_candidate: dict[str, Any],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_policy = "P1" if comparison.endswith("vs_P1") else "P2"
    candidate_policy = "P2" if comparison == "P2_vs_P1" else "P3"
    reference = simulate_policy(
        test_rows,
        reference_policy,
        train_reference.get("threshold224"),
        train_reference.get("threshold512"),
    )
    candidate = simulate_policy(
        test_rows,
        candidate_policy,
        train_candidate.get("threshold224"),
        train_candidate.get("threshold512"),
    )
    return {
        "language": language,
        "repeat": None,
        "outer_seed": seed,
        "outer_fold": fold,
        "target": target,
        "comparison": comparison,
        "reference_threshold224": train_reference.get("threshold224"),
        "reference_threshold512": train_reference.get("threshold512"),
        "candidate_threshold224": train_candidate.get("threshold224"),
        "candidate_threshold512": train_candidate.get("threshold512"),
        "false_cutoff_delta": candidate["false_cutoff_rate"] - reference["false_cutoff_rate"],
        "mean_latency_change_ms": (
            float(candidate["mean_endpoint_latency_ms"] or 0.0)
            - float(reference["mean_endpoint_latency_ms"] or 0.0)
        ),
        "eot_timeout_change": candidate["eot_timeout_rate"] - reference["eot_timeout_rate"],
        "timeout_rate_change_pp": (candidate["eot_timeout_rate"] - reference["eot_timeout_rate"])
        * 100.0,
        "turn_fragmentation_change_pp": (
            candidate["turn_fragmentation_rate"] - reference["turn_fragmentation_rate"]
        )
        * 100.0,
        "reference_false_cutoff_rate": reference["false_cutoff_rate"],
        "candidate_false_cutoff_rate": candidate["false_cutoff_rate"],
        "reference_mean_latency_ms": reference["mean_endpoint_latency_ms"],
        "candidate_mean_latency_ms": candidate["mean_endpoint_latency_ms"],
        "reference_eot_timeout_rate": reference["eot_timeout_rate"],
        "candidate_eot_timeout_rate": candidate["eot_timeout_rate"],
    }


def _availability_rows(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cv_rows:
        if row["policy"] not in {"P1", "P2", "P3"}:
            continue
        grouped[(row["language"], row["policy"], row["target"])].append(row)
    result = []
    for (language, policy, target), rows in sorted(grouped.items()):
        selected = sum(row["status"] == "available" for row in rows)
        expected = len(rows)
        result.append(
            {
                "language": language,
                "policy": policy,
                "target": target,
                "available_outer_evaluations": selected,
                "expected_outer_evaluations": expected,
                "availability_rate": selected / expected if expected else 0.0,
            }
        )
    return result


def _threshold_stability(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cv_rows:
        if row["status"] == "available" and row["policy"] in {"P1", "P2", "P3"}:
            grouped[(row["language"], row["policy"], row["target"])].append(row)
    result = []
    for (language, policy, target), rows in sorted(grouped.items()):
        expected = sum(
            1
            for item in cv_rows
            if item["language"] == language
            and item["policy"] == policy
            and item["target"] == target
        )
        for key, label in (("threshold224", "T224"), ("threshold512", "T512")):
            values = [float(row[key]) for row in rows if _finite(row.get(key))]
            if values:
                p10, p25, median, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
                result.append(
                    {
                        "language": language,
                        "policy": policy,
                        "target": target,
                        "threshold": label,
                        "count": len(values),
                        "availability_rate": len(values) / expected if expected else 0.0,
                        "p10": float(p10),
                        "p25": float(p25),
                        "median": float(median),
                        "p75": float(p75),
                        "p90": float(p90),
                        "minimum": float(min(values)),
                        "maximum": float(max(values)),
                        "iqr": float(p75 - p25),
                    }
                )
            else:
                result.append(
                    {
                        "language": language,
                        "policy": policy,
                        "target": target,
                        "threshold": label,
                        "count": 0,
                        "availability_rate": 0.0,
                        "p10": None,
                        "p25": None,
                        "median": None,
                        "p75": None,
                        "p90": None,
                        "minimum": None,
                        "maximum": None,
                        "iqr": None,
                    }
                )
    return result


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for field in METRIC_FIELDS:
        values = [float(row[field]) for row in rows if _finite(row.get(field))]
        result[field] = float(np.mean(values)) if values else None
    return result


def _aggregate_cv(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cv_rows:
        if row["status"] == "available":
            grouped[(row["language"], row["policy"], row["target"])].append(row)
    result = []
    for (language, policy, target), rows in sorted(grouped.items()):
        thresholds224 = [row["threshold224"] for row in rows if _finite(row.get("threshold224"))]
        thresholds512 = [row["threshold512"] for row in rows if _finite(row.get("threshold512"))]
        result.append(
            {
                "language": language,
                "policy": policy,
                "target": target,
                "evaluations": len(rows),
                **_aggregate_metrics(rows),
                "threshold224_median": float(np.median(thresholds224)) if thresholds224 else None,
                "threshold512_median": float(np.median(thresholds512)) if thresholds512 else None,
            }
        )
    return result


def _final_operating_points(
    cv_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    expected: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in cv_rows:
        if row["policy"] not in {"P1", "P2", "P3"}:
            continue
        key = (row["language"], row["policy"], row["target"])
        expected[key] += 1
        if row["status"] == "available":
            grouped[key].append(row)
    points = {}
    for key, total in sorted(expected.items()):
        language, policy, target = key
        rows = grouped.get(key, [])
        t224 = [float(row["threshold224"]) for row in rows if _finite(row.get("threshold224"))]
        t512 = [float(row["threshold512"]) for row in rows if _finite(row.get("threshold512"))]
        points[key] = {
            "language": language,
            "policy": policy,
            "target": target,
            "selection_kind": "selected" if rows else "unavailable",
            "folds_available": len(rows),
            "folds_expected": total,
            "availability_rate": len(rows) / total if total else 0.0,
            "threshold224": float(np.median(t224)) if t224 else None,
            "threshold512": float(np.median(t512)) if t512 else None,
            "threshold224_iqr": (
                float(np.percentile(t224, 75) - np.percentile(t224, 25)) if t224 else None
            ),
            "threshold512_iqr": (
                float(np.percentile(t512, 75) - np.percentile(t512, 25)) if t512 else None
            ),
            "selection_tie_reasons": sorted(
                {
                    str(row["selection_tie_reason"])
                    for row in rows
                    if row.get("selection_tie_reason")
                }
            ),
            "selection_tie_count_max": max(
                [int(row["selection_tie_count"]) for row in rows if row.get("selection_tie_count")]
                or [0]
            ),
            "heldout_mean": _aggregate_metrics(rows) if rows else None,
        }
    return points


def _preferred_point(
    points: dict[tuple[str, str, str], dict[str, Any]],
    language: str,
    policy: str,
) -> dict[str, Any] | None:
    for target in ("stability", "low_latency"):
        point = points.get((language, policy, target))
        if point and point["selection_kind"] == "selected":
            return point
    return None


def _aggregate_turn_fragmentation(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in _aggregate_cv(cv_rows):
        result.append(
            {
                "language": row["language"],
                "policy": row["policy"],
                "target": row["target"],
                "evaluations": row["evaluations"],
                "turns_with_false_cutoff_rate": row["turn_fragmentation_rate"],
                "false_splits_per_100_turns": row["false_splits_per_100_turns"],
                "mean_false_cutoffs_per_affected_turn": row["mean_false_cutoffs_per_affected_turn"],
            }
        )
    return result


def _policy_gate_by_language(
    paired_rows: list[dict[str, Any]],
    languages: Iterable[str],
    bootstrap_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    result = {}
    bootstrap_evidence = bootstrap_evidence or []
    for language in languages:
        candidates = [
            row
            for row in paired_rows
            if row["language"] == language and row["status"] == "available"
        ]
        selected = None
        for target in ("stability", "low_latency"):
            rows = [row for row in candidates if row["target"] == target]
            if not rows:
                continue
            value_conditions = [bool(row["value_condition"]) for row in rows]
            false_deltas = [float(row["p2_false_cutoff_minus_p1"]) for row in rows]
            robust_rate = sum(value_conditions) / len(value_conditions)
            false_ci_rows = [
                evidence
                for evidence in bootstrap_evidence
                if evidence["language"] == language
                and evidence["target"] == target
                and evidence["comparison"] == "P2_vs_P1"
                and evidence["metric"] == "false_cutoff_delta"
            ]
            false_ci = false_ci_rows[0] if false_ci_rows else None
            false_ci_low = false_ci.get("ci_low") if false_ci else None
            false_ci_high = false_ci.get("ci_high") if false_ci else None
            regression_rate = sum(bool(row["false_cutoff_regression_ok"]) for row in rows) / len(
                rows
            )
            gate = {
                "language": language,
                "target": target,
                "valid_outer_evaluations": len(rows),
                "value_condition_count": sum(value_conditions),
                "value_condition_rate": robust_rate,
                "false_cutoff_regression_rate": regression_rate,
                "false_cutoff_delta_point_estimate": float(np.mean(false_deltas)),
                "false_cutoff_delta_ci_low": false_ci_low,
                "false_cutoff_delta_ci_high": false_ci_high,
                "false_cutoff_regression_limit": MATCH_TOLERANCE,
                "passes_robustness": robust_rate >= 0.80,
                "passes_false_cutoff_regression": regression_rate >= 1.0,
                "passes_false_cutoff_ci": (
                    false_ci_high is not None and false_ci_high <= MATCH_TOLERANCE
                ),
                "bootstrap_resamples": false_ci.get("resamples") if false_ci else None,
                "passes": (
                    robust_rate >= 0.80
                    and regression_rate >= 1.0
                    and false_ci_high is not None
                    and false_ci_high <= MATCH_TOLERANCE
                ),
            }
            if gate["passes"]:
                selected = gate
                break
            if selected is None:
                selected = gate
        result[language] = selected or {
            "language": language,
            "target": None,
            "valid_outer_evaluations": 0,
            "value_condition_count": 0,
            "value_condition_rate": 0.0,
            "false_cutoff_regression_rate": 0.0,
            "false_cutoff_delta_point_estimate": None,
            "false_cutoff_delta_ci_low": None,
            "false_cutoff_delta_ci_high": None,
            "false_cutoff_regression_limit": MATCH_TOLERANCE,
            "passes_robustness": False,
            "passes_false_cutoff_regression": False,
            "passes_false_cutoff_ci": False,
            "bootstrap_resamples": None,
            "passes": False,
        }
    return result


def _p3_gate(
    p3_rows: list[dict[str, Any]],
    language: str,
    target: str | None,
    bootstrap_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if target is None:
        return {"language": language, "target": None, "evaluations": 0, "passes": False}
    bootstrap_evidence = bootstrap_evidence or []
    rows = [
        row
        for row in p3_rows
        if row.get("language") == language
        and row.get("target") == target
        and row.get("status") == "available"
    ]
    if not rows:
        return {"language": language, "target": target, "evaluations": 0, "passes": False}
    mode_results = []
    for mode, metric, limit, direction in (
        ("false_cutoff_matched", "mean_endpoint_delta_ms", -15.0, "latency"),
        ("mean_latency_matched", "false_cutoff_delta", -0.01, "false_cutoff"),
    ):
        mode_rows = [row for row in rows if row.get("matching_mode") == mode]
        if not mode_rows:
            continue
        values = [
            float(
                row["mean_latency_change_ms"]
                if direction == "latency"
                else row["false_cutoff_delta"]
            )
            for row in mode_rows
        ]
        condition_rate = sum(value <= limit for value in values) / len(values)
        evidence_rows = [
            evidence
            for evidence in bootstrap_evidence
            if evidence["language"] == language
            and evidence["target"] == target
            and evidence["comparison"] == "P3_vs_P2"
            and evidence["metric"] == metric
        ]
        evidence = evidence_rows[0] if evidence_rows else None
        ci_high = evidence.get("ci_high") if evidence else None
        ci_low = evidence.get("ci_low") if evidence else None
        regression_rate = sum(
            float(row["false_cutoff_delta"]) <= MATCH_TOLERANCE for row in mode_rows
        ) / len(mode_rows)
        mode_results.append(
            {
                "matching_mode": mode,
                "evaluations": len(mode_rows),
                "condition_rate": condition_rate,
                "point_estimate": float(np.mean(values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "required_limit": limit,
                "passes_condition": condition_rate >= 0.80,
                "passes_ci": ci_high is not None and ci_high <= limit,
                "false_cutoff_regression_rate": regression_rate,
                "passes_false_cutoff_regression": regression_rate >= 1.0,
                "passes": (
                    condition_rate >= 0.80
                    and ci_high is not None
                    and ci_high <= limit
                    and regression_rate >= 1.0
                ),
                "ci_metric": metric,
                "bootstrap_resamples": evidence.get("resamples") if evidence else None,
            }
        )
    passing = [result for result in mode_results if result["passes"]]
    return {
        "language": language,
        "target": target,
        "evaluations": len(rows),
        "matched_evaluations": len(rows),
        "modes": mode_results,
        "passes": bool(passing),
        "selected_mode": passing[0]["matching_mode"] if passing else None,
    }


def _language_decisions(
    rows_by_language: dict[str, list[dict[str, Any]]],
    cv_rows: list[dict[str, Any]],
    p2_gates: dict[str, dict[str, Any]],
    p3_gates: dict[str, dict[str, Any]],
    bootstrap_evidence: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    points = _final_operating_points(cv_rows)
    bootstrap_evidence = bootstrap_evidence or []
    decisions: dict[str, Any] = {}
    for language in rows_by_language:
        p1_low = points.get((language, "P1", "low_latency"))
        p1_stability = points.get((language, "P1", "stability"))
        p1 = (
            p1_low
            if p1_low and p1_low["selection_kind"] == "selected"
            else (
                p1_stability
                if p1_stability and p1_stability["selection_kind"] == "selected"
                else None
            )
        )
        p1_constraints_pass = bool(
            p1
            and p1["availability_rate"] >= 0.80
            and p1.get("heldout_mean")
            and float(p1["heldout_mean"]["mean_endpoint_latency_ms"] or math.inf) <= 600.0
            and float(p1["heldout_mean"]["p50_endpoint_latency_ms"] or math.inf) <= 600.0
            and float(p1["heldout_mean"]["eot_timeout_rate"] or math.inf) <= 0.25
        )
        p1_reliability = _p1_reliability(language, cv_rows, bootstrap_evidence)
        selected_reliability = p1_reliability["targets"].get(p1["target"]) if p1 else None
        p1_available = bool(p1_constraints_pass and p1_reliability["reliable"])
        p2_gate = p2_gates[language]
        p3_gate = p3_gates[language]
        if not p1:
            decision = "BASELINE_ONLY"
            reason = "no valid P1 operating target"
        elif not p1_reliability["reliable"]:
            point_reduction = float(
                (p1.get("heldout_mean") or {}).get("relative_false_cutoff_reduction") or 0.0
            )
            target_reduction = 0.20 if p1["target"] == "low_latency" else 0.35
            if p1_constraints_pass and point_reduction >= target_reduction:
                decision = "NEEDS_MORE_DATA"
                reason = "P1 point estimate is promising but reliability, CI, or target availability is insufficient"
            else:
                decision = "BASELINE_ONLY"
                reason = "P1 does not reliably improve B0 under held-out constraints and bootstrap evidence"
        elif not p2_gate["passes"]:
            decision = "P1_SHADOW"
            reason = "P2 did not pass the paired second-probe value gate"
        elif p3_gate["passes"]:
            decision = "P3_SHADOW"
            reason = "P3 passed its conditional matched-complexity gate"
        else:
            decision = "P2_SHADOW"
            reason = "P2 passed and P3 was unavailable or not materially better"
        decisions[language] = {
            "language": language,
            "decision": decision,
            "reason": reason,
            "p1_target": p1["target"] if p1 else None,
            "p1_available": p1_available,
            "p1_constraints_pass": p1_constraints_pass,
            "p1_reliability": selected_reliability,
            "p2_gate": p2_gate,
            "p3_gate": p3_gate,
        }
    values = [item["decision"] for item in decisions.values()]
    if any(value == "NEEDS_MORE_DATA" for value in values):
        global_decision = "NEEDS_MORE_POLICY_DATA"
    elif any(value in {"P1_SHADOW", "P2_SHADOW", "P3_SHADOW"} for value in values):
        global_decision = "PROCEED_TO_LANGUAGE_SPECIFIC_SHADOW"
    elif values and all(value == "BASELINE_ONLY" for value in values):
        global_decision = "STOP_SMART_TURN_INTEGRATION"
    else:
        global_decision = "NEEDS_MORE_POLICY_DATA"
    return decisions, global_decision


def cross_validate(
    rows_by_language: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected: list[dict[str, Any]] = []
    outer_by_language = {
        language: _outer_splits(rows, language, rejected)
        for language, rows in rows_by_language.items()
    }
    cv_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    stage1_selected: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for language, splits in outer_by_language.items():
        for split in splits:
            split["language"] = language
            baseline_test = {
                "B0": simulate_policy(split["test_rows"], "B0"),
                "B1": simulate_policy(split["test_rows"], "B1"),
            }
            prepared_selection = {
                policy: _prepare_inner_selection(
                    split["train_rows"],
                    policy,
                    int(split["seed"]) + int(split["fold"]) * 1_003,
                )
                for policy in ("P1", "P2")
            }
            for target in ("low_latency", "stability"):
                for policy in ("B0", "B1"):
                    cv_rows.append(
                        _cv_row(
                            split=split,
                            target=target,
                            policy=policy,
                            selection=None,
                            metrics=baseline_test[policy],
                            baseline=baseline_test["B0"],
                            status="available",
                            selection_kind="baseline",
                        )
                    )
                selected_for_pair: dict[str, dict[str, Any] | None] = {}
                for policy in ("P1", "P2"):
                    selection = _select_prepared_inner(
                        prepared_selection[policy],
                        target,
                    )
                    if selection is None:
                        cv_rows.append(
                            _cv_row(
                                split=split,
                                target=target,
                                policy=policy,
                                selection=None,
                                metrics=None,
                                baseline=baseline_test["B0"],
                                status="unavailable",
                                selection_kind="selected",
                            )
                        )
                        selected_for_pair[policy] = None
                        continue
                    metrics = simulate_policy(
                        split["test_rows"],
                        policy,
                        selection["threshold224"],
                        selection["threshold512"],
                    )
                    row = _cv_row(
                        split=split,
                        target=target,
                        policy=policy,
                        selection=selection,
                        metrics=metrics,
                        baseline=baseline_test["B0"],
                        status="available",
                        selection_kind="selected",
                    )
                    cv_rows.append(row)
                    selected_for_pair[policy] = row
                    stage1_selected[(language, split["repeat"], split["fold"], target, policy)] = (
                        row
                    )
                paired_rows.append(
                    _paired_row(
                        p1=selected_for_pair["P1"],
                        p2=selected_for_pair["P2"],
                        split=split,
                        target=target,
                    )
                )
    stage1_cv = {
        "cv_rows": cv_rows,
        "paired_rows": paired_rows,
        "outer_splits": outer_by_language,
    }
    stage1_bootstrap = _outer_bootstrap_evidence(
        stage1_cv,
        rows_by_language,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    p2_gates = _policy_gate_by_language(
        paired_rows,
        rows_by_language,
        stage1_bootstrap,
    )
    p3_rows: list[dict[str, Any]] = []
    p3_conditional_rows: list[dict[str, Any]] = []
    increment_rows: list[dict[str, Any]] = []
    for language, splits in outer_by_language.items():
        if not p2_gates[language]["passes"]:
            for target in ("low_latency", "stability"):
                p3_conditional_rows.append(
                    {
                        "language": language,
                        "target": target,
                        "status": "not evaluated: second probe did not pass complexity gate",
                        "reason": "P2 value gate failed or had insufficient valid outer evaluations",
                    }
                )
            continue
        for split in splits:
            prepared_p3 = _prepare_inner_selection(
                split["train_rows"],
                "P3",
                int(split["seed"]) + int(split["fold"]) * 1_003,
            )
            for target in ("low_latency", "stability"):
                p2_row = stage1_selected.get(
                    (language, split["repeat"], split["fold"], target, "P2")
                )
                p2_selection = (
                    {
                        "inner_false_cutoff_rate": p2_row["inner_false_cutoff_rate"],
                        "inner_mean_endpoint_latency_ms": p2_row["inner_mean_endpoint_latency_ms"],
                    }
                    if p2_row
                    else None
                )
                selection = _select_p3_prepared_inner(prepared_p3, target, p2_selection)
                baseline = simulate_policy(split["test_rows"], "B0")
                if selection is None:
                    p3_rows.append(
                        _cv_row(
                            split=split,
                            target=target,
                            policy="P3",
                            selection=None,
                            metrics=None,
                            baseline=baseline,
                            status="unavailable",
                            selection_kind="selected",
                        )
                    )
                    p3_conditional_rows.append(
                        {
                            "language": language,
                            "repeat": split["repeat"],
                            "outer_seed": split["seed"],
                            "outer_fold": split["fold"],
                            "target": target,
                            "status": "unavailable",
                            "reason": (
                                "P2 outer candidate unavailable"
                                if p2_row is None
                                else "no P3 candidate passed nested inner-CV operating constraints and matched comparison gates"
                            ),
                        }
                    )
                    continue
                p3_metrics = simulate_policy(
                    split["test_rows"],
                    "P3",
                    selection["threshold224"],
                    selection["threshold512"],
                )
                p3_row = _cv_row(
                    split=split,
                    target=target,
                    policy="P3",
                    selection=selection,
                    metrics=p3_metrics,
                    baseline=baseline,
                    status="available",
                    selection_kind="selected",
                )
                p3_rows.append(p3_row)
                increment = _increment_row(
                    language=language,
                    seed=int(split["seed"]),
                    fold=int(split["fold"]),
                    target=target,
                    comparison="P3_vs_P2",
                    train_reference={
                        "threshold224": p2_row["threshold224"],
                        "threshold512": p2_row["threshold512"],
                    },
                    train_candidate=selection,
                    test_rows=split["test_rows"],
                )
                increment["repeat"] = split["repeat"]
                increment["matched_training_false_cutoff_difference"] = selection.get(
                    "matched_training_false_cutoff_difference"
                )
                increment["matched_training_mean_latency_difference_ms"] = selection.get(
                    "matched_training_mean_latency_difference_ms"
                )
                increment["matching_mode"] = selection.get("selection_mode")
                increment_rows.append(increment)
                p3_conditional_rows.append(
                    {
                        "language": language,
                        "repeat": split["repeat"],
                        "outer_seed": split["seed"],
                        "outer_fold": split["fold"],
                        "target": target,
                        "status": "available",
                        "reason": "nested inner-CV matched P3 candidate evaluated",
                        "matching_mode": selection.get("selection_mode"),
                        "training_false_cutoff_difference": selection.get(
                            "matched_training_false_cutoff_difference"
                        ),
                        "training_mean_latency_difference_ms": selection.get(
                            "matched_training_mean_latency_difference_ms"
                        ),
                        **{
                            key: value
                            for key, value in increment.items()
                            if key
                            in {
                                "false_cutoff_delta",
                                "mean_latency_change_ms",
                                "eot_timeout_change",
                                "turn_fragmentation_change_pp",
                            }
                        },
                    }
                )
    cv_rows.extend(p3_rows)
    for row in paired_rows:
        if row["status"] != "available":
            continue
        increment_rows.append(
            {
                "language": row["language"],
                "repeat": row["repeat"],
                "outer_seed": row["outer_seed"],
                "outer_fold": row["outer_fold"],
                "target": row["target"],
                "comparison": "P2_vs_P1",
                "false_cutoff_delta": row["p2_false_cutoff_minus_p1"],
                "mean_latency_change_ms": row["p2_mean_endpoint_minus_p1_ms"],
                "eot_timeout_change": row["p2_eot_timeout_minus_p1"],
                "timeout_rate_change_pp": row["p2_eot_timeout_minus_p1"] * 100.0,
                "turn_fragmentation_change_pp": row["p2_turn_fragmentation_minus_p1"] * 100.0,
            }
        )
    final_cv = {
        "cv_rows": cv_rows,
        "paired_rows": paired_rows,
        "outer_splits": outer_by_language,
    }
    p3_bootstrap = _outer_bootstrap_evidence(
        final_cv,
        rows_by_language,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        policies=("P3",),
        comparisons=(("P2", "P3"),),
    )
    bootstrap_evidence = stage1_bootstrap + p3_bootstrap
    split_manifest = [
        {key: value for key, value in split.items() if key not in {"train_rows", "test_rows"}}
        for splits in outer_by_language.values()
        for split in splits
    ]
    _write_csv(output_dir / "outer_split_manifest.csv", split_manifest)
    _write_csv(output_dir / "rejected_split_manifest.csv", rejected)
    _write_csv(output_dir / "nested_cv_all.csv", cv_rows)
    for language in rows_by_language:
        _write_csv(
            output_dir / f"nested_cv_{language}.csv",
            [row for row in cv_rows if row["language"] == language],
        )
    _write_csv(output_dir / "candidate_availability.csv", _availability_rows(cv_rows))
    _write_csv(output_dir / "threshold_stability.csv", _threshold_stability(cv_rows))
    _write_csv(output_dir / "span_metrics.csv", _aggregate_cv(cv_rows))
    _write_csv(output_dir / "turn_fragmentation.csv", _aggregate_turn_fragmentation(cv_rows))
    _write_csv(output_dir / "p1_vs_p2_paired.csv", paired_rows)
    _write_csv(output_dir / "p3_conditional_results.csv", p3_conditional_rows)
    return {
        "cv_rows": cv_rows,
        "paired_rows": paired_rows,
        "increment_rows": increment_rows,
        "p3_conditional_rows": p3_conditional_rows,
        "outer_splits": outer_by_language,
        "rejected_splits": rejected,
        "p2_gates": p2_gates,
        "threshold_stability": _threshold_stability(cv_rows),
        "bootstrap_evidence": bootstrap_evidence,
    }


def _bootstrap_group_sample(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    groups = sorted({_group_key(row) for row in rows})
    group_rows = {group: [row for row in rows if _group_key(row) == group] for group in groups}
    sampled_groups = [rng.choice(groups) for _ in groups]
    return [row for group in sampled_groups for row in group_rows[group]]


def _bootstrap_data_factory(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    data = _array_data(rows)
    group_names = list(dict.fromkeys(str(group) for group in data["groups"]))
    group_indices = [np.flatnonzero(data["groups"] == group) for group in group_names]
    return data, group_indices


def _sample_bootstrap_data(
    data: dict[str, np.ndarray],
    group_indices: list[np.ndarray],
    rng: random.Random,
) -> dict[str, np.ndarray]:
    selected = [group_indices[rng.randrange(len(group_indices))] for _ in range(len(group_indices))]
    indices = np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)
    return {key: values[indices] for key, values in data.items()}


def _fast_bootstrap_metrics(
    data: dict[str, np.ndarray],
    policy: str,
    threshold224: float | None = None,
    threshold512: float | None = None,
) -> dict[str, Any]:
    trace = _policy_trace(
        [None] * len(data["duration"]),
        policy,
        threshold224,
        threshold512,
        array_data=data,
    )
    decision = trace["decision_ms"]
    hold = data["hold"]
    eot = data["eot"]
    false_cutoffs = hold & (data["duration"] > decision)
    turns, turn_codes = np.unique(data["turns"], return_inverse=True)
    false_by_turn = np.bincount(
        turn_codes,
        weights=false_cutoffs.astype(np.int64),
        minlength=len(turns),
    )
    eot_count = int(eot.sum())
    return {
        "false_cutoff_rate": (float(false_cutoffs.sum() / hold.sum()) if hold.sum() else 0.0),
        "relative_false_cutoff_reduction": None,
        "mean_endpoint_latency_ms": float(np.mean(decision[eot])) if eot_count else 0.0,
        "eot_timeout_rate": (
            float((eot & (decision >= TIMEOUT_MS - 1e-6)).sum() / eot_count) if eot_count else 0.0
        ),
        "turn_fragmentation_rate": (
            float(np.count_nonzero(false_by_turn) / len(turns)) if len(turns) else 0.0
        ),
    }


def _group_metric_stats(
    rows: list[dict[str, Any]],
    policy: str,
    threshold224: float | None = None,
    threshold512: float | None = None,
) -> dict[str, np.ndarray]:
    data = _array_data(rows)
    trace = _policy_trace(
        [None] * len(rows),
        policy,
        threshold224,
        threshold512,
        array_data=data,
    )
    decision = trace["decision_ms"]
    false_cutoffs = data["hold"] & (data["duration"] > decision)
    groups = list(dict.fromkeys(str(group) for group in data["groups"]))
    group_indices = [np.flatnonzero(data["groups"] == group) for group in groups]
    result = {
        "hold_count": np.zeros(len(groups), dtype=np.float64),
        "false_count": np.zeros(len(groups), dtype=np.float64),
        "eot_count": np.zeros(len(groups), dtype=np.float64),
        "eot_endpoint_sum": np.zeros(len(groups), dtype=np.float64),
        "eot_timeout_count": np.zeros(len(groups), dtype=np.float64),
        "turn_count": np.zeros(len(groups), dtype=np.float64),
        "affected_turn_count": np.zeros(len(groups), dtype=np.float64),
    }
    hard_timeout = decision >= TIMEOUT_MS - 1e-6
    for index, row_indices in enumerate(group_indices):
        hold_indices = row_indices[data["hold"][row_indices]]
        eot_indices = row_indices[data["eot"][row_indices]]
        result["hold_count"][index] = len(hold_indices)
        result["false_count"][index] = float(false_cutoffs[row_indices].sum())
        result["eot_count"][index] = len(eot_indices)
        result["eot_endpoint_sum"][index] = float(decision[eot_indices].sum())
        result["eot_timeout_count"][index] = float(hard_timeout[eot_indices].sum())
        turns = {str(value) for value in data["turns"][row_indices]}
        affected = {str(value) for value in data["turns"][row_indices][false_cutoffs[row_indices]]}
        result["turn_count"][index] = len(turns)
        result["affected_turn_count"][index] = len(affected)
    return result


def _bootstrap_counts(
    group_count: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if group_count < 1:
        return np.empty((resamples, 0), dtype=np.float64)
    probabilities = np.full(group_count, 1.0 / group_count, dtype=np.float64)
    return rng.multinomial(group_count, probabilities, size=resamples).astype(np.float64)


def _metric_samples_from_totals(totals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    false_rate = np.divide(
        totals["false_count"],
        totals["hold_count"],
        out=np.zeros_like(totals["false_count"]),
        where=totals["hold_count"] != 0,
    )
    mean_latency = np.divide(
        totals["eot_endpoint_sum"],
        totals["eot_count"],
        out=np.zeros_like(totals["eot_endpoint_sum"]),
        where=totals["eot_count"] != 0,
    )
    timeout_rate = np.divide(
        totals["eot_timeout_count"],
        totals["eot_count"],
        out=np.zeros_like(totals["eot_timeout_count"]),
        where=totals["eot_count"] != 0,
    )
    fragmentation_rate = np.divide(
        totals["affected_turn_count"],
        totals["turn_count"],
        out=np.zeros_like(totals["affected_turn_count"]),
        where=totals["turn_count"] != 0,
    )
    return {
        "false_cutoff_rate": false_rate,
        "mean_endpoint_latency_ms": mean_latency,
        "eot_timeout_rate": timeout_rate,
        "turn_fragmentation_rate": fragmentation_rate,
    }


def _zero_metric_totals(resamples: int) -> dict[str, np.ndarray]:
    return {
        key: np.zeros(resamples, dtype=np.float64)
        for key in (
            "hold_count",
            "false_count",
            "eot_count",
            "eot_endpoint_sum",
            "eot_timeout_count",
            "turn_count",
            "affected_turn_count",
        )
    }


def _add_group_sample(
    totals: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    counts: np.ndarray,
    weights: np.ndarray | None = None,
) -> None:
    multiplier = counts if weights is None else counts * weights[:, None]
    for key in totals:
        totals[key] += multiplier @ stats[key]


def _record_metric_samples(
    records: list[dict[str, Any]],
    policy: str,
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    policy_totals = _zero_metric_totals(resamples)
    baseline_totals = _zero_metric_totals(resamples)
    rng = np.random.default_rng(seed)
    for record in records:
        rows = record["rows"]
        policy_stats = _group_metric_stats(
            rows,
            policy,
            record.get("threshold224"),
            record.get("threshold512"),
        )
        baseline_stats = _group_metric_stats(rows, "B0")
        counts = _bootstrap_counts(len(policy_stats["hold_count"]), resamples, rng)
        _add_group_sample(policy_totals, policy_stats, counts)
        _add_group_sample(baseline_totals, baseline_stats, counts)
    return _metric_samples_from_totals(policy_totals), _metric_samples_from_totals(baseline_totals)


def _record_pair_samples(
    records: list[dict[str, Any]],
    reference_policy: str,
    candidate_policy: str,
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    reference_totals = _zero_metric_totals(resamples)
    candidate_totals = _zero_metric_totals(resamples)
    rng = np.random.default_rng(seed)
    for record in records:
        rows = record["rows"]
        reference_stats = _group_metric_stats(
            rows,
            reference_policy,
            record.get("reference_threshold224"),
            record.get("reference_threshold512"),
        )
        candidate_stats = _group_metric_stats(
            rows,
            candidate_policy,
            record.get("candidate_threshold224"),
            record.get("candidate_threshold512"),
        )
        counts = _bootstrap_counts(len(reference_stats["hold_count"]), resamples, rng)
        _add_group_sample(reference_totals, reference_stats, counts)
        _add_group_sample(candidate_totals, candidate_stats, counts)
    return _metric_samples_from_totals(reference_totals), _metric_samples_from_totals(
        candidate_totals
    )


def _split_lookup(cv: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (str(language), int(split["repeat"]), int(split["fold"])): split
        for language, splits in cv["outer_splits"].items()
        for split in splits
    }


def _policy_bootstrap_records(
    cv: dict[str, Any],
    language: str,
    policy: str,
    target: str,
) -> list[dict[str, Any]]:
    lookup = _split_lookup(cv)
    records = []
    for row in cv["cv_rows"]:
        if (
            row["language"] != language
            or row["policy"] != policy
            or row["target"] != target
            or row["status"] != "available"
        ):
            continue
        split = lookup[(language, int(row["repeat"]), int(row["outer_fold"]))]
        records.append(
            {
                "rows": split["test_rows"],
                "repeat": int(row["repeat"]),
                "outer_fold": int(row["outer_fold"]),
                "threshold224": row.get("threshold224"),
                "threshold512": row.get("threshold512"),
            }
        )
    return records


def _pair_bootstrap_records(
    cv: dict[str, Any],
    language: str,
    reference_policy: str,
    candidate_policy: str,
    target: str,
) -> list[dict[str, Any]]:
    lookup = _split_lookup(cv)
    records = []
    if reference_policy == "P1" and candidate_policy == "P2":
        for row in cv["paired_rows"]:
            if (
                row["language"] != language
                or row["target"] != target
                or row["status"] != "available"
            ):
                continue
            split = lookup[(language, int(row["repeat"]), int(row["outer_fold"]))]
            records.append(
                {
                    "rows": split["test_rows"],
                    "repeat": int(row["repeat"]),
                    "outer_fold": int(row["outer_fold"]),
                    "reference_threshold224": row.get("p1_threshold224"),
                    "reference_threshold512": None,
                    "candidate_threshold224": row.get("p2_threshold224"),
                    "candidate_threshold512": row.get("p2_threshold512"),
                }
            )
        return records
    for row in cv["cv_rows"]:
        if (
            row["language"] != language
            or row["policy"] != candidate_policy
            or row["target"] != target
            or row["status"] != "available"
        ):
            continue
        reference_rows = [
            candidate
            for candidate in cv["cv_rows"]
            if candidate["language"] == language
            and candidate["policy"] == reference_policy
            and candidate["target"] == target
            and candidate["repeat"] == row["repeat"]
            and candidate["outer_fold"] == row["outer_fold"]
            and candidate["status"] == "available"
        ]
        if not reference_rows:
            continue
        reference = reference_rows[0]
        split = lookup[(language, int(row["repeat"]), int(row["outer_fold"]))]
        records.append(
            {
                "rows": split["test_rows"],
                "repeat": int(row["repeat"]),
                "outer_fold": int(row["outer_fold"]),
                "reference_threshold224": reference.get("threshold224"),
                "reference_threshold512": reference.get("threshold512"),
                "candidate_threshold224": row.get("threshold224"),
                "candidate_threshold512": row.get("threshold512"),
            }
        )
    return records


def _bootstrap_ci_rows(
    *,
    language: str,
    policy: str,
    target: str,
    comparison: str,
    samples: dict[str, np.ndarray],
    resamples: int,
    seed: int,
    outer_partitions: int,
    outer_repeats: int,
    bootstrap_unit: str = "conversation_within_outer_test_partition",
) -> list[dict[str, Any]]:
    rows = []
    for metric, values in samples.items():
        rows.append(
            {
                "language": language,
                "policy": policy,
                "target": target,
                "comparison": comparison,
                "metric": metric,
                "estimate": float(np.mean(values)),
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "resamples": resamples,
                "seed": seed,
                "bootstrap_unit": bootstrap_unit,
                "outer_partitions": outer_partitions,
                "outer_repeats": outer_repeats,
                "ci_method": "percentile",
            }
        )
    return rows


def _outer_bootstrap_evidence(
    cv: dict[str, Any],
    languages: Iterable[str],
    *,
    resamples: int,
    seed: int,
    policies: tuple[str, ...] = ("P1", "P2", "P3"),
    comparisons: tuple[tuple[str, str], ...] = (("P1", "P2"), ("P2", "P3")),
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for language in languages:
        for target in ("low_latency", "stability"):
            for policy in policies:
                records = _policy_bootstrap_records(cv, language, policy, target)
                if not records:
                    continue
                policy_samples, baseline_samples = _record_metric_samples(
                    records,
                    policy,
                    resamples=resamples,
                    seed=seed + sum(ord(char) for char in f"{language}:{policy}:{target}"),
                )
                relative = np.divide(
                    baseline_samples["false_cutoff_rate"] - policy_samples["false_cutoff_rate"],
                    baseline_samples["false_cutoff_rate"],
                    out=np.zeros(resamples, dtype=np.float64),
                    where=baseline_samples["false_cutoff_rate"] != 0,
                )
                samples = {
                    "false_cutoff_rate": policy_samples["false_cutoff_rate"],
                    "relative_false_cutoff_reduction": relative,
                    "mean_endpoint_latency_ms": policy_samples["mean_endpoint_latency_ms"],
                    "eot_timeout_rate": policy_samples["eot_timeout_rate"],
                    "turn_fragmentation_rate": policy_samples["turn_fragmentation_rate"],
                }
                output.extend(
                    _bootstrap_ci_rows(
                        language=language,
                        policy=policy,
                        target=target,
                        comparison=f"{policy}_vs_B0",
                        samples=samples,
                        resamples=resamples,
                        seed=seed,
                        outer_partitions=len(records),
                        outer_repeats=len({record["repeat"] for record in records}),
                    )
                )
            for reference_policy, candidate_policy in comparisons:
                records = _pair_bootstrap_records(
                    cv,
                    language,
                    reference_policy,
                    candidate_policy,
                    target,
                )
                if not records:
                    continue
                reference_samples, candidate_samples = _record_pair_samples(
                    records,
                    reference_policy,
                    candidate_policy,
                    resamples=resamples,
                    seed=seed
                    + 100_003
                    + sum(ord(char) for char in f"{language}:{target}:{candidate_policy}"),
                )
                samples = {
                    "mean_endpoint_delta_ms": candidate_samples["mean_endpoint_latency_ms"]
                    - reference_samples["mean_endpoint_latency_ms"],
                    "false_cutoff_delta": candidate_samples["false_cutoff_rate"]
                    - reference_samples["false_cutoff_rate"],
                    "eot_timeout_delta": candidate_samples["eot_timeout_rate"]
                    - reference_samples["eot_timeout_rate"],
                    "turn_fragmentation_delta": candidate_samples["turn_fragmentation_rate"]
                    - reference_samples["turn_fragmentation_rate"],
                }
                output.extend(
                    _bootstrap_ci_rows(
                        language=language,
                        policy=candidate_policy,
                        target=target,
                        comparison=f"{candidate_policy}_vs_{reference_policy}",
                        samples=samples,
                        resamples=resamples,
                        seed=seed,
                        outer_partitions=len(records),
                        outer_repeats=len({record["repeat"] for record in records}),
                    )
                )
    return output


def _hierarchical_pair_samples(
    records: list[dict[str, Any]],
    reference_policy: str,
    candidate_policy: str,
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["repeat"])].append(record)
    repeat_values = sorted(grouped)
    reference_totals = _zero_metric_totals(resamples)
    candidate_totals = _zero_metric_totals(resamples)
    if not repeat_values:
        return (
            _metric_samples_from_totals(reference_totals),
            _metric_samples_from_totals(candidate_totals),
        )
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(repeat_values), 1.0 / len(repeat_values), dtype=np.float64)
    repeat_counts = rng.multinomial(len(repeat_values), probabilities, size=resamples).astype(
        np.float64
    )
    for repeat_index, repeat in enumerate(repeat_values):
        weights = repeat_counts[:, repeat_index]
        for record in grouped[repeat]:
            rows = record["rows"]
            reference_stats = _group_metric_stats(
                rows,
                reference_policy,
                record.get("reference_threshold224"),
                record.get("reference_threshold512"),
            )
            candidate_stats = _group_metric_stats(
                rows,
                candidate_policy,
                record.get("candidate_threshold224"),
                record.get("candidate_threshold512"),
            )
            counts = _bootstrap_counts(len(reference_stats["hold_count"]), resamples, rng)
            _add_group_sample(reference_totals, reference_stats, counts, weights)
            _add_group_sample(candidate_totals, candidate_stats, counts, weights)
    return _metric_samples_from_totals(reference_totals), _metric_samples_from_totals(
        candidate_totals
    )


def bootstrap_confidence_intervals(
    rows_by_language: dict[str, list[dict[str, Any]]],
    points: dict[tuple[str, str, str], dict[str, Any]],
    *,
    output_dir: Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    cv: dict[str, Any] | None = None,
    precomputed: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if cv is not None:
        del points
        output = precomputed or _outer_bootstrap_evidence(
            cv,
            rows_by_language,
            resamples=resamples,
            seed=seed,
        )
        _write_csv(output_dir / "bootstrap_confidence_intervals.csv", output)
        return output
    output: list[dict[str, Any]] = []
    for (language, policy, target), point in sorted(points.items()):
        if point.get("selection_kind") == "unavailable" or point.get("threshold224") is None:
            continue
        rows = rows_by_language[language]
        base_data, group_indices = _bootstrap_data_factory(rows)
        rng = random.Random(seed + sum(ord(char) for char in f"{language}:{policy}:{target}"))
        values: dict[str, list[float]] = {
            "false_cutoff_rate": [],
            "relative_false_cutoff_reduction": [],
            "mean_endpoint_latency_ms": [],
            "eot_timeout_rate": [],
        }
        for _ in range(resamples):
            sampled_data = _sample_bootstrap_data(base_data, group_indices, rng)
            metrics = _fast_bootstrap_metrics(
                sampled_data,
                policy,
                point.get("threshold224"),
                point.get("threshold512"),
            )
            baseline = _fast_bootstrap_metrics(sampled_data, "B0")
            values["false_cutoff_rate"].append(float(metrics["false_cutoff_rate"]))
            values["relative_false_cutoff_reduction"].append(
                _relative_reduction(
                    float(baseline["false_cutoff_rate"]),
                    float(metrics["false_cutoff_rate"]),
                )
            )
            values["mean_endpoint_latency_ms"].append(
                float(metrics["mean_endpoint_latency_ms"] or 0.0)
            )
            values["eot_timeout_rate"].append(float(metrics["eot_timeout_rate"]))
        for metric, samples in values.items():
            output.append(
                {
                    "language": language,
                    "policy": policy,
                    "target": target,
                    "comparison": f"{policy}_vs_B0",
                    "metric": metric,
                    "estimate": float(np.mean(samples)),
                    "ci_low": float(np.percentile(samples, 2.5)),
                    "ci_high": float(np.percentile(samples, 97.5)),
                    "resamples": resamples,
                    "seed": seed,
                    "bootstrap_unit": "conversation",
                }
            )
    for language in rows_by_language:
        for target in ("low_latency", "stability"):
            p1 = points.get((language, "P1", target))
            p2 = points.get((language, "P2", target))
            if not p1 or not p2 or p1.get("threshold224") is None or p2.get("threshold224") is None:
                continue
            rows = rows_by_language[language]
            base_data, group_indices = _bootstrap_data_factory(rows)
            rng = random.Random(seed + 100_003 + sum(ord(char) for char in f"{language}:{target}"))
            values = {
                "mean_endpoint_delta_ms": [],
                "false_cutoff_delta": [],
                "eot_timeout_delta": [],
                "turn_fragmentation_delta": [],
            }
            for _ in range(resamples):
                sampled_data = _sample_bootstrap_data(base_data, group_indices, rng)
                p1_metrics = _fast_bootstrap_metrics(
                    sampled_data,
                    "P1",
                    p1["threshold224"],
                )
                p2_metrics = _fast_bootstrap_metrics(
                    sampled_data,
                    "P2",
                    p2["threshold224"],
                    p2["threshold512"],
                )
                values["mean_endpoint_delta_ms"].append(
                    float(p2_metrics["mean_endpoint_latency_ms"] or 0.0)
                    - float(p1_metrics["mean_endpoint_latency_ms"] or 0.0)
                )
                values["false_cutoff_delta"].append(
                    float(p2_metrics["false_cutoff_rate"] - p1_metrics["false_cutoff_rate"])
                )
                values["eot_timeout_delta"].append(
                    float(p2_metrics["eot_timeout_rate"] - p1_metrics["eot_timeout_rate"])
                )
                values["turn_fragmentation_delta"].append(
                    float(
                        p2_metrics["turn_fragmentation_rate"]
                        - p1_metrics["turn_fragmentation_rate"]
                    )
                )
            for metric, samples in values.items():
                output.append(
                    {
                        "language": language,
                        "policy": "P2",
                        "target": target,
                        "comparison": "P2_vs_P1",
                        "metric": metric,
                        "estimate": float(np.mean(samples)),
                        "ci_low": float(np.percentile(samples, 2.5)),
                        "ci_high": float(np.percentile(samples, 97.5)),
                        "resamples": resamples,
                        "seed": seed,
                        "bootstrap_unit": "conversation",
                    }
                )
    _write_csv(output_dir / "bootstrap_confidence_intervals.csv", output)
    return output


def hierarchical_bootstrap(
    rows_by_language: dict[str, list[dict[str, Any]]],
    points: dict[tuple[str, str, str], dict[str, Any]],
    *,
    output_dir: Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    cv: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if cv is not None:
        del points
        output: list[dict[str, Any]] = []
        for language in rows_by_language:
            for target in ("low_latency", "stability"):
                records = _pair_bootstrap_records(cv, language, "P1", "P2", target)
                if not records:
                    continue
                reference_samples, candidate_samples = _hierarchical_pair_samples(
                    records,
                    "P1",
                    "P2",
                    resamples=resamples,
                    seed=seed + 300_007 + sum(ord(char) for char in f"{language}:{target}"),
                )
                samples = {
                    "mean_endpoint_delta_ms": candidate_samples["mean_endpoint_latency_ms"]
                    - reference_samples["mean_endpoint_latency_ms"],
                    "false_cutoff_delta": candidate_samples["false_cutoff_rate"]
                    - reference_samples["false_cutoff_rate"],
                    "eot_timeout_delta": candidate_samples["eot_timeout_rate"]
                    - reference_samples["eot_timeout_rate"],
                    "turn_fragmentation_delta": candidate_samples["turn_fragmentation_rate"]
                    - reference_samples["turn_fragmentation_rate"],
                }
                output.extend(
                    _bootstrap_ci_rows(
                        language=language,
                        policy="P2",
                        target=target,
                        comparison="P2_vs_P1",
                        samples=samples,
                        resamples=resamples,
                        seed=seed,
                        outer_partitions=len(records),
                        outer_repeats=len({record["repeat"] for record in records}),
                        bootstrap_unit="outer_repeat_then_conversation_within_partition",
                    )
                )
        _write_csv(output_dir / "hierarchical_bootstrap.csv", output)
        return output
    output = []
    for language in rows_by_language:
        p1 = _preferred_point(points, language, "P1")
        p2 = _preferred_point(points, language, "P2")
        if not p1 or not p2 or p1.get("threshold224") is None or p2.get("threshold224") is None:
            continue
        rows = rows_by_language[language]
        base_data, group_indices = _bootstrap_data_factory(rows)
        groups = list(range(len(group_indices)))
        rng = random.Random(seed + 300_007 + sum(ord(char) for char in language))
        values = {
            "mean_endpoint_delta_ms": [],
            "false_cutoff_delta": [],
            "eot_timeout_delta": [],
            "turn_fragmentation_delta": [],
        }
        for _ in range(resamples):
            first_level = [rng.choice(groups) for _ in groups]
            second_level = [rng.choice(first_level) for _ in groups]
            indices = np.concatenate([group_indices[group] for group in second_level])
            sampled_data = {key: values[indices] for key, values in base_data.items()}
            p1_metrics = _fast_bootstrap_metrics(
                sampled_data,
                "P1",
                p1["threshold224"],
            )
            p2_metrics = _fast_bootstrap_metrics(
                sampled_data,
                "P2",
                p2["threshold224"],
                p2["threshold512"],
            )
            values["mean_endpoint_delta_ms"].append(
                float(p2_metrics["mean_endpoint_latency_ms"] or 0.0)
                - float(p1_metrics["mean_endpoint_latency_ms"] or 0.0)
            )
            values["false_cutoff_delta"].append(
                float(p2_metrics["false_cutoff_rate"] - p1_metrics["false_cutoff_rate"])
            )
            values["eot_timeout_delta"].append(
                float(p2_metrics["eot_timeout_rate"] - p1_metrics["eot_timeout_rate"])
            )
            values["turn_fragmentation_delta"].append(
                float(p2_metrics["turn_fragmentation_rate"] - p1_metrics["turn_fragmentation_rate"])
            )
        for metric, samples in values.items():
            output.append(
                {
                    "language": language,
                    "target": p2["target"],
                    "comparison": "P2_vs_P1",
                    "metric": metric,
                    "estimate": float(np.mean(samples)),
                    "ci_low": float(np.percentile(samples, 2.5)),
                    "ci_high": float(np.percentile(samples, 97.5)),
                    "resamples": resamples,
                    "seed": seed,
                    "bootstrap_unit": "hierarchical_repeat_then_conversation",
                }
            )
    _write_csv(output_dir / "hierarchical_bootstrap.csv", output)
    return output


def audit_providers(*, output_dir: Path, model_path: Path) -> dict[str, Any]:
    audit = {
        "previous_policy_predictions": {
            "model_variant": "external GPU prediction artifact",
            "execution_provider": "unknown",
        },
        "previous_local_benchmark": {
            "model_variant": "CPU int8",
            "execution_provider": "unknown",
        },
        "new_cpu_prediction_run": {
            "model_variant": "CPU int8",
            "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": 2,
            "inter_op_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "model_sha256": _sha256(model_path) if model_path.is_file() else None,
        },
    }
    _write_json(output_dir / "provider_audit.json", audit)
    return audit


def _write_corrections(output_dir: Path) -> None:
    lines = [
        "# Evaluator corrections",
        "",
        "This run reuses the existing CPU-int8 prediction parquet files and performs policy analysis only.",
        "",
        "- EOT timeout rate counts only true EOT spans whose authoritative decision is the 800 ms timeout.",
        "- Unresolved span rate counts all spans without an accepted Smart Turn result and is diagnostic only.",
        "- P3 searches all inner-CV candidates, enforces T512 >= T224, records false-cutoff or latency matching, and returns unavailable when no matched candidate passes.",
        "- Thresholds come from observed inner-training scores; the outer test fold is not used for selection.",
        "- Outer evaluation uses 50 deterministic repeated group splits per language and records rejected folds.",
        "- Probe decisions use measured inference latency, one-worker scheduling, stale-result rejection, and 800 ms precedence.",
        "- Bootstrap intervals resample conversations inside each outer test partition rather than individual spans and use 10,000 resamples by default; hierarchical intervals resample outer repeats first.",
        "",
        "Smart Turn inference, CPU latency benchmarks, cold-start benchmarks, provider benchmarks, production behavior, and shadow-mode wiring are outside this run.",
        "The worktree contains pre-existing production Smart Turn thread/default changes outside this policy-analysis candidate; they are preserved, excluded from the candidate scope, and reported as architecture/scope drift rather than changed by this experiment.",
    ]
    (output_dir / "evaluator_corrections.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _p1_reliability(
    language: str,
    cv_rows: list[dict[str, Any]],
    bootstrap_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    target_rows = {}
    for target in ("low_latency", "stability"):
        rows = [
            row
            for row in cv_rows
            if row["language"] == language and row["policy"] == "P1" and row["target"] == target
        ]
        available = [row for row in rows if row["status"] == "available"]
        expected = len(rows)
        aggregate = _aggregate_metrics(available) if available else {}
        evidence_rows = [
            evidence
            for evidence in bootstrap_evidence
            if evidence["language"] == language
            and evidence["policy"] == "P1"
            and evidence["target"] == target
            and evidence["comparison"] == "P1_vs_B0"
            and evidence["metric"] == "relative_false_cutoff_reduction"
        ]
        evidence = evidence_rows[0] if evidence_rows else None
        target_reduction = 0.20 if target == "low_latency" else 0.35
        reliable = bool(
            available
            and expected
            and len(available) / expected >= 0.80
            and float(aggregate.get("relative_false_cutoff_reduction") or 0.0) >= target_reduction
            and evidence
            and float(evidence["ci_low"]) >= target_reduction
            and float(aggregate.get("eot_timeout_rate") or math.inf) <= 0.25
        )
        target_rows[target] = {
            "target": target,
            "available_outer_evaluations": len(available),
            "expected_outer_evaluations": expected,
            "availability_rate": len(available) / expected if expected else 0.0,
            "heldout_mean": aggregate,
            "relative_reduction_ci_low": evidence.get("ci_low") if evidence else None,
            "relative_reduction_ci_high": evidence.get("ci_high") if evidence else None,
            "bootstrap_resamples": evidence.get("resamples") if evidence else None,
            "reliable": reliable,
            "answer": "reliably beats B0" if reliable else "not reliably established",
        }
    selected = next(
        (
            target_rows[target]
            for target in ("stability", "low_latency")
            if target_rows[target]["reliable"]
        ),
        None,
    )
    return {
        "language": language,
        "selected_target": selected["target"] if selected else None,
        "reliable": bool(selected),
        "answer": selected["answer"] if selected else "not reliably established",
        "targets": target_rows,
    }


def build_summary(
    *,
    output_dir: Path,
    rows_by_language: dict[str, list[dict[str, Any]]],
    cv: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cv_rows = cv["cv_rows"]
    points = _final_operating_points(cv_rows)
    p3_gates = {
        language: _p3_gate(
            [row for row in cv["increment_rows"] if row["comparison"] == "P3_vs_P2"],
            language,
            cv["p2_gates"][language].get("target"),
            cv.get("bootstrap_evidence"),
        )
        for language in rows_by_language
    }
    decisions, global_decision = _language_decisions(
        rows_by_language,
        cv_rows,
        cv["p2_gates"],
        p3_gates,
        cv.get("bootstrap_evidence"),
    )
    p1_reliability = {
        language: _p1_reliability(
            language,
            cv_rows,
            cv.get("bootstrap_evidence", []),
        )
        for language in rows_by_language
    }
    selected_points = {}
    for language in rows_by_language:
        selected_points[language] = {}
        for policy in ("P1", "P2", "P3"):
            selected_points[language][policy] = _preferred_point(points, language, policy)
    selected_payload = {
        "selection_rule": "stability target preferred, then low-latency target; unavailable targets remain unavailable",
        "points": selected_points,
    }
    _write_json(output_dir / "selected_operating_points.json", selected_payload)
    _write_json(output_dir / "language_decisions.json", {"languages": decisions})
    summary = {
        "mode": "policy_only",
        "languages": list(rows_by_language),
        "decision": global_decision,
        "language_decisions": decisions,
        "input_validation": validation,
        "cross_validation": {
            "outer_folds": N_FOLDS,
            "accepted_repeats_requested": REPEAT_COUNT,
            "seeds": list(CV_SEEDS),
            "evaluations_per_language": {
                language: len(cv["outer_splits"].get(language, [])) for language in rows_by_language
            },
            "aggregates": _aggregate_cv(cv_rows),
            "p2_gates": cv["p2_gates"],
            "p3_gates": p3_gates,
            "threshold_stability": cv["threshold_stability"],
        },
        "p1_reliability": p1_reliability,
        "span_counts": {
            language: {
                "spans": len(rows),
                "eot": sum(row["label"] == "eot" for row in rows),
                "hold": sum(row["label"] == "hold" for row in rows),
                "groups": len({_group_key(row) for row in rows}),
            }
            for language, rows in rows_by_language.items()
        },
        "scope": {
            "prediction_artifacts_reused": True,
            "inference_rerun": False,
            "latency_benchmark_rerun": False,
            "production_behavior_changed": False,
            "shadow_mode_added": False,
            "candidate_scope": [
                "tools/eot_experiment/policy_analysis.py",
                "tools/eot_experiment/calibrate_cpu.py",
                "tests/scripts/test_policy_analysis.py",
            ],
            "preexisting_worktree_drift_preserved_outside_candidate": {
                "files": [
                    "src/puripuly_heart/core/vad/smart_turn.py",
                    "tests/core/test_smart_turn.py",
                    "tests/scripts/test_calibrate_cpu.py",
                    "tests/scripts/test_evaluate_repeated_probes.py",
                ],
                "architecture_note": "The existing worktree changes alter the production Smart Turn CPU thread default and related endpoint wiring; this policy-only candidate does not own or modify them.",
            },
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _format(value: Any, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}{suffix}"
    return f"{value}{suffix}"


def write_report(summary: dict[str, Any], *, output_dir: Path) -> None:
    lines = [
        "# Smart Turn policy-only experiment",
        "",
        f"Global decision: **{summary['decision']}**",
        "",
        "The evaluator reused CPU-int8 prediction artifacts. It did not run Smart Turn inference or CPU/provider benchmarks.",
        "",
        "## Language decisions",
        "",
        "| Language | Decision | P1 target | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for language, decision in summary["language_decisions"].items():
        lines.append(
            f"| {language} | {decision['decision']} | {decision['p1_target'] or '—'} | {decision['reason']} |"
        )
    lines.extend(
        [
            "",
            "## P1 reliability answer",
            "",
            "P1 is considered reliable only when target availability, held-out constraints, and the conversation-bootstrap reduction interval all pass the target gate.",
            "",
            "| Language | Selected target | Available / expected | Reduction 95% CI | Reliable answer |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for language, reliability in summary["p1_reliability"].items():
        selected_target = reliability["selected_target"]
        selected = reliability["targets"].get(selected_target) if selected_target else None
        availability = (
            f"{selected['available_outer_evaluations']} / {selected['expected_outer_evaluations']}"
            if selected
            else "—"
        )
        reduction_ci = (
            f"{_format(selected['relative_reduction_ci_low'])}–{_format(selected['relative_reduction_ci_high'])}"
            if selected and selected["relative_reduction_ci_low"] is not None
            else "—"
        )
        lines.append(
            f"| {language} | {selected_target or '—'} | {availability} | {reduction_ci} | {reliability['answer']} |"
        )
    lines.extend(
        [
            "",
            "## Threshold stability answer",
            "",
            "Threshold stability is reported over accepted outer evaluations; unavailable targets remain unavailable and are not replaced by a baseline-matched point.",
            "",
            "| Language | Policy | Target | Threshold | Availability | Median | P10–P90 | IQR |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["cross_validation"]["threshold_stability"]:
        lines.append(
            f"| {row['language']} | {row['policy']} | {row['target']} | {row['threshold']} | "
            f"{_format(row['availability_rate'])} | {_format(row['median'])} | "
            f"{_format(row['p10'])}–{_format(row['p90'])} | {_format(row['iqr'])} |"
        )
    lines.extend(
        [
            "",
            "## Pinned gates",
            "",
            "EOT timeout rate is the product timeout metric. Unresolved span rate is diagnostic only.",
            "P3 is evaluated only for languages whose paired P2 result passes the value gate.",
            "",
            "| Language | P2 target | Valid paired evaluations | Value condition rate | False-cutoff CI high | P2 gate |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for language, gate in summary["cross_validation"]["p2_gates"].items():
        lines.append(
            f"| {language} | {gate.get('target') or '—'} | {gate.get('valid_outer_evaluations', 0)} | "
            f"{_format(gate.get('value_condition_rate'))} | {_format(gate.get('false_cutoff_delta_ci_high'))} | "
            f"{gate.get('passes', False)} |"
        )
    lines.extend(
        [
            "",
            "## Central answers",
            "",
            "1. P1 reliability is answered by the P1 reliability table and its conversation-level bootstrap interval, not by point estimates alone.",
            "2. The 512 ms probe is retained only when the paired value rate, false-cutoff regression, and bootstrap CI pass the P2 gate.",
            "3. Language-specific shadow decisions are listed above; languages without a reliable target remain baseline-only or need more policy data.",
            "4. Threshold stability is answered by repeated outer-evaluation availability, percentiles, and IQR.",
            "",
            "## Scope and architecture note",
            "",
            "This candidate is policy-analysis tooling and tests only. Pre-existing Smart Turn production thread/default changes remain outside the candidate, are preserved rather than silently overwritten, and are reported as architecture/scope drift for separate owner review.",
            "",
            "## Evidence interpretation",
            "",
            "Repeated nested CV measures sensitivity to conversation grouping; it does not create new independent audio samples.",
            "Bootstrap intervals resample conversations inside each outer test partition and remain separate from the repeated-CV evaluation rows.",
            "",
            "Selection tie counts and reasons are recorded in nested_cv_all.csv. See nested_cv_all.csv, p1_vs_p2_paired.csv, p3_conditional_results.csv, threshold_stability.csv, and the bootstrap CSVs for detailed evidence.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_languages(values: list[str]) -> tuple[str, ...]:
    languages = tuple(_map_language(value) for value in values)
    if not languages or any(language not in LANGUAGES for language in languages):
        raise SystemExit(f"languages must be selected from {LANGUAGES}")
    return languages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive")
    languages = _parse_languages(args.languages)
    output_dir = args.output_dir.resolve()
    predictions_dir = (args.predictions_dir or output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_language, validation = validate_input_artifacts(
        predictions_dir,
        languages,
        output_dir=output_dir,
    )
    _write_corrections(output_dir)
    cv = cross_validate(
        rows_by_language,
        output_dir=output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    points = _final_operating_points(cv["cv_rows"])
    bootstrap_confidence_intervals(
        rows_by_language,
        points,
        output_dir=output_dir,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        cv=cv,
        precomputed=cv.get("bootstrap_evidence"),
    )
    hierarchical_bootstrap(
        rows_by_language,
        points,
        output_dir=output_dir,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        cv=cv,
    )
    summary = build_summary(
        output_dir=output_dir,
        rows_by_language=rows_by_language,
        cv=cv,
        validation=validation,
    )
    write_report(summary, output_dir=output_dir)
    print(json.dumps({"output_dir": str(output_dir), "decision": summary["decision"]}, indent=2))


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "CV_SEEDS",
    "LANGUAGES",
    "N_FOLDS",
    "REPEAT_COUNT",
    "_candidate_thresholds",
    "_candidate_is_valid",
    "_group_splits",
    "_matched_candidate",
    "_policy_trace",
    "_threshold_grid",
    "_trace_metrics",
    "_validate_prediction_rows",
    "audit_providers",
    "bootstrap_confidence_intervals",
    "build_summary",
    "cross_validate",
    "hierarchical_bootstrap",
    "main",
    "simulate_policy",
    "validate_input_artifacts",
    "write_report",
]


if __name__ == "__main__":
    main()
