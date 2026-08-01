from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DATASET_ID = "livekit/eot-bench-data"
VALID_LABELS = {"hold", "eot"}
DEFAULT_LANGUAGES = ("ko", "ja", "en", "zh")
DEFAULT_PROBES_MS = (200, 500)
DEFAULT_TIMEOUT_MS = 800
DEFAULT_BASELINE_MS = 512
DEFAULT_THRESHOLD_STEP = 0.01
BALANCED_LATENCY_LIMIT_MS = 560.0
EPS = 1e-7
TIME_EPS_MS = 1e-6
CPU_REQUIRED_LENGTHS_S = (0.5, 1.0, 2.0, 4.0, 8.0)
CPU_REQUIRED_LANGUAGES = {"ko", "ja", "en", "zh"}
CPU_MIN_WARMUPS = 10
CPU_MIN_MEASURED = 100
CPU_PREFERRED_P95_MS = 150.0
CPU_ACCEPTABLE_P95_MS = 200.0


@dataclass(frozen=True)
class Span:
    span_id: str
    language: str
    span_index: int
    label: str
    span_duration_ms: float
    scores: dict[int, float]


@dataclass(frozen=True)
class Artifact:
    language: str
    predictions_path: Path
    manifest_path: Path
    span_set_path: Path | None
    manifest: dict[str, Any]
    spans: tuple[Span, ...]
    validation: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _read_parquet(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read eot-bench predictions.parquet") from exc
    table = parquet.read_table(path)
    return table.to_pylist(), set(table.column_names)


def _as_float(value: Any, *, field: str, row_number: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"row {row_number}: {field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return result


def _as_span_index(value: Any, *, row_number: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"row {row_number}: span_index must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: span_index must be a non-negative integer") from exc
    if result < 0 or float(value) != result:
        raise ValueError(f"row {row_number}: span_index must be a non-negative integer")
    return result


def _normalize_language(value: Any, *, row_number: int) -> str:
    if value is None:
        raise ValueError(f"row {row_number}: language is required")
    language = str(value).strip().lower()
    if not language:
        raise ValueError(f"row {row_number}: language must not be empty")
    return language


def _dataset_path_from_manifest(manifest: dict[str, Any]) -> str | None:
    dataset = manifest.get("dataset")
    if isinstance(dataset, dict) and dataset.get("path") is not None:
        return str(dataset["path"])
    for key in ("path", "repo_id", "dataset_path"):
        if manifest.get(key) is not None:
            return str(manifest[key])
    return None


def _find_prediction_path(language_root: Path) -> Path:
    candidates = sorted(language_root.rglob("predictions.parquet"))
    if not candidates:
        raise ValueError(f"requested language has no predictions.parquet: {language_root.name}")
    smart_turn_candidates = [
        path
        for path in candidates
        if "smart_turn" in str(path).lower() or "smartturn" in str(path).lower()
    ]
    if len(smart_turn_candidates) == 1:
        return smart_turn_candidates[0]
    if len(candidates) != 1:
        paths = ", ".join(str(path) for path in candidates[:5])
        raise ValueError(f"ambiguous prediction artifacts for {language_root.name}: {paths}")
    return candidates[0]


def _find_span_set(language_root: Path, predictions_path: Path) -> Path | None:
    direct = language_root / "span_set.parquet"
    if direct.is_file():
        return direct
    candidates = sorted(language_root.rglob("span_set.parquet"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        paths = ", ".join(str(path) for path in candidates[:5])
        raise ValueError(f"ambiguous span-set artifacts for {language_root.name}: {paths}")
    if predictions_path.parent.parent == language_root:
        return None
    return None


def _load_span_set(
    path: Path,
    *,
    language: str,
) -> tuple[dict[tuple[str, int], tuple[str, float]], dict[str, Any]]:
    rows, columns = _read_parquet(path)
    required = {"id", "language", "span_index", "label", "duration"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"span-set artifact is missing duration or required fields: {missing}")
    spans: dict[tuple[str, int], tuple[str, float]] = {}
    duplicate_count = 0
    for row_number, row in enumerate(rows, start=1):
        row_language = _normalize_language(row.get("language"), row_number=row_number)
        if row_language != language:
            raise ValueError(
                f"span-set row {row_number}: language {row_language!r} != {language!r}"
            )
        span_id = str(row.get("id", "")).strip()
        if not span_id:
            raise ValueError(f"span-set row {row_number}: id is required")
        span_index = _as_span_index(row.get("span_index"), row_number=row_number)
        label = str(row.get("label", "")).strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(f"span-set row {row_number}: invalid label {label!r}")
        duration = _as_float(row.get("duration"), field="duration", row_number=row_number)
        if duration <= 0:
            raise ValueError(f"span-set row {row_number}: duration must be positive")
        key = (span_id, span_index)
        if key in spans:
            duplicate_count += 1
            raise ValueError(f"duplicate span-set row for {key!r}")
        spans[key] = (label, duration * 1000.0)
    if not spans:
        raise ValueError(f"span-set artifact is empty: {path}")
    labels = {label for label, _ in spans.values()}
    if labels != VALID_LABELS:
        raise ValueError("span-set must contain both hold and eot labels")
    return spans, {
        "source": "span_set.parquet",
        "span_count": len(spans),
        "duplicate_count": duplicate_count,
        "duration_field": "duration",
    }


def _load_prediction_spans(
    path: Path,
    *,
    language: str,
    span_set: dict[tuple[str, int], tuple[str, float]] | None,
    probes_ms: tuple[int, ...],
) -> tuple[tuple[Span, ...], dict[str, Any]]:
    rows, columns = _read_parquet(path)
    required = {"id", "language", "span_index", "timestamp", "silence_dur", "p_eot", "label"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"prediction artifact is missing required fields: {missing}")
    if span_set is None and not ({"span_dur", "span_duration", "duration"} & columns):
        raise ValueError(
            "span duration is unavailable: provide span_set.parquet or a span_dur/span_duration/duration field",
        )

    groups: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_probe_rows = 0
    for row_number, row in enumerate(rows, start=1):
        row_language = _normalize_language(row.get("language"), row_number=row_number)
        if row_language != language:
            raise ValueError(
                f"prediction row {row_number}: language {row_language!r} != {language!r}"
            )
        span_id = str(row.get("id", "")).strip()
        if not span_id:
            raise ValueError(f"prediction row {row_number}: id is required")
        span_index = _as_span_index(row.get("span_index"), row_number=row_number)
        label = str(row.get("label", "")).strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(f"prediction row {row_number}: invalid label {label!r}")
        _as_float(row.get("timestamp"), field="timestamp", row_number=row_number)
        silence_dur = _as_float(row.get("silence_dur"), field="silence_dur", row_number=row_number)
        if silence_dur < -EPS:
            raise ValueError(f"prediction row {row_number}: silence_dur must be non-negative")
        score = _as_float(row.get("p_eot"), field="p_eot", row_number=row_number)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"prediction row {row_number}: p_eot must be in 0.0..1.0")
        key = (span_id, span_index)
        group = groups.setdefault(
            key,
            {
                "language": row_language,
                "label": label,
                "rows": [],
                "duration_ms": None,
                "explicit_duration_ms": None,
            },
        )
        if group["label"] != label:
            raise ValueError(f"conflicting labels for span {key!r}")
        for field in ("span_dur", "span_duration", "duration"):
            if field in row and row[field] is not None:
                explicit = _as_float(row[field], field=field, row_number=row_number)
                if explicit <= 0:
                    raise ValueError(f"prediction row {row_number}: {field} must be positive")
                explicit_ms = explicit * 1000.0 if explicit < 20.0 else explicit
                if (
                    group["explicit_duration_ms"] is not None
                    and abs(group["explicit_duration_ms"] - explicit_ms) > 0.01
                ):
                    raise ValueError(f"conflicting span durations for span {key!r}")
                group["explicit_duration_ms"] = explicit_ms
        normalized_silence = round(silence_dur, 6)
        if any(abs(existing - normalized_silence) <= EPS for existing, _ in group["rows"]):
            duplicate_probe_rows += 1
            raise ValueError(f"duplicate score row for span {key!r} at silence_dur={silence_dur}")
        group["rows"].append((normalized_silence, score))

    if not groups:
        raise ValueError(f"prediction artifact is empty: {path}")
    prediction_keys = set(groups)
    if span_set is not None:
        span_keys = set(span_set)
        missing_predictions = sorted(span_keys - prediction_keys)[:5]
        extra_predictions = sorted(prediction_keys - span_keys)[:5]
        if missing_predictions or extra_predictions or span_keys != prediction_keys:
            raise ValueError(
                "prediction and span-set keys differ; "
                f"missing={missing_predictions}, extra={extra_predictions}",
            )

    spans: list[Span] = []
    missing_score_counts = {str(probe_ms): 0 for probe_ms in probes_ms}
    permitted_missing_score_counts = {str(probe_ms): 0 for probe_ms in probes_ms}
    for key, group in groups.items():
        rows_for_span = group["rows"]
        predicted_duration_ms = max(silence for silence, _ in rows_for_span) * 1000.0
        if span_set is not None:
            label, duration_ms = span_set[key]
        else:
            label = group["label"]
            duration_ms = group["explicit_duration_ms"] or predicted_duration_ms
        if label != group["label"]:
            raise ValueError(f"conflicting labels between prediction and span set for {key!r}")
        if predicted_duration_ms > duration_ms + 0.1:
            raise ValueError(f"prediction extends beyond span duration for {key!r}")
        scores: dict[int, float] = {}
        for probe_ms in probes_ms:
            candidates = [
                (abs(silence_ms * 1000.0 - probe_ms), score)
                for silence_ms, score in rows_for_span
                if abs(silence_ms * 1000.0 - probe_ms) <= 0.1
            ]
            if candidates:
                candidates.sort(key=lambda item: item[0])
                if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) <= EPS:
                    raise ValueError(f"duplicate score row for span {key!r} at probe {probe_ms}ms")
                scores[probe_ms] = candidates[0][1]
        for probe_ms in probes_ms:
            if duration_ms + TIME_EPS_MS >= probe_ms and probe_ms not in scores:
                missing_score_counts[str(probe_ms)] += 1
            elif duration_ms + TIME_EPS_MS < probe_ms and probe_ms not in scores:
                permitted_missing_score_counts[str(probe_ms)] += 1
        spans.append(
            Span(
                span_id=key[0],
                language=language,
                span_index=key[1],
                label=label,
                span_duration_ms=duration_ms,
                scores=scores,
            ),
        )

    missing_expected = {probe: count for probe, count in missing_score_counts.items() if count}
    if missing_expected:
        raise ValueError(
            "missing required probe scores for spans that survive to the probe: "
            f"{missing_expected}",
        )
    spans.sort(key=lambda span: (span.span_id, span.span_index))
    if not any(span.label == "hold" for span in spans) or not any(
        span.label == "eot" for span in spans
    ):
        raise ValueError("prediction artifact must contain both hold and eot spans")
    return tuple(spans), {
        "prediction_row_count": len(rows),
        "span_count": len(spans),
        "duplicate_probe_row_count": duplicate_probe_rows,
        "missing_required_scores": missing_score_counts,
        "permitted_missing_scores": permitted_missing_score_counts,
        "duration_field": (
            "span_set.parquet" if span_set is not None else "explicit_prediction_duration"
        ),
    }


def load_artifact(root: Path, language: str, probes_ms: tuple[int, ...]) -> Artifact:
    language_root = root / language
    if not language_root.is_dir():
        raise ValueError(f"requested language is missing from predictions root: {language}")
    predictions_path = _find_prediction_path(language_root)
    manifest_path = predictions_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"prediction manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    dataset_path = _dataset_path_from_manifest(manifest)
    if dataset_path != DATASET_ID:
        raise ValueError(
            f"dataset validation failed for {language}: expected {DATASET_ID!r}, got {dataset_path!r}",
        )
    manifest_language = str(
        manifest.get("language") or manifest.get("dataset", {}).get("language", "")
    ).lower()
    if manifest_language and manifest_language != language:
        raise ValueError(
            f"manifest language {manifest_language!r} does not match requested {language!r}"
        )
    span_set_path = _find_span_set(language_root, predictions_path)
    span_set: dict[tuple[str, int], tuple[str, float]] | None = None
    span_validation: dict[str, Any]
    if span_set_path is not None:
        span_set, span_validation = _load_span_set(span_set_path, language=language)
    else:
        span_validation = {
            "source": "prediction artifact",
            "span_count": None,
            "duration_field": "explicit_prediction_duration",
        }
    spans, prediction_validation = _load_prediction_spans(
        predictions_path,
        language=language,
        span_set=span_set,
        probes_ms=probes_ms,
    )
    if span_set is not None and len(span_set) != len(spans):
        raise ValueError(f"span-set count does not match predictions for {language}")
    validation = {
        "language": language,
        "dataset": DATASET_ID,
        "predictions_path": str(predictions_path),
        "manifest_path": str(manifest_path),
        "span_set_path": str(span_set_path) if span_set_path else None,
        "span_set": span_validation,
        "predictions": prediction_validation,
        "eot_spans": sum(span.label == "eot" for span in spans),
        "hold_spans": sum(span.label == "hold" for span in spans),
        "probe_score_counts": {
            str(probe_ms): sum(probe_ms in span.scores for span in spans) for probe_ms in probes_ms
        },
        "missing_score_counts": {
            str(probe_ms): sum(probe_ms not in span.scores for span in spans)
            for probe_ms in probes_ms
        },
        "permitted_missing_score_counts": {
            str(probe_ms): sum(
                span.span_duration_ms + TIME_EPS_MS < probe_ms and probe_ms not in span.scores
                for span in spans
            )
            for probe_ms in probes_ms
        },
    }
    return Artifact(
        language=language,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        span_set_path=span_set_path,
        manifest=manifest,
        spans=spans,
        validation=validation,
    )


def thresholds_from_step(step: float) -> tuple[float, ...]:
    if not math.isfinite(step) or step <= 0 or step > 1:
        raise ValueError("threshold-step must be in (0.0, 1.0]")
    count = round(1.0 / step)
    if abs(count * step - 1.0) > 1e-8:
        raise ValueError("threshold-step must divide 1.00 exactly")
    return tuple(round(index * step, 10) for index in range(count + 1))


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _accepted(span: Span, probe_ms: int, threshold: float) -> bool:
    score = span.scores.get(probe_ms)
    return (
        span.span_duration_ms + TIME_EPS_MS >= probe_ms and score is not None and score > threshold
    )


def _survives(span: Span, probe_ms: int) -> bool:
    return span.span_duration_ms + TIME_EPS_MS >= probe_ms


def _decision(
    span: Span,
    policy: str,
    threshold: float | None,
    probes_ms: tuple[int, ...],
    timeout_ms: int,
    baseline_ms: int,
) -> tuple[float, str]:
    if policy == "B0":
        return float(baseline_ms), f"fixed_{baseline_ms}ms"
    if policy == "B1":
        return float(timeout_ms), "timeout"
    if threshold is None:
        raise ValueError(f"{policy} requires a threshold")
    if policy == "S1":
        if _accepted(span, probes_ms[0], threshold):
            return float(probes_ms[0]), f"{probes_ms[0]}ms"
        return float(timeout_ms), "timeout"
    if policy == "S2":
        for probe_ms in probes_ms:
            if _accepted(span, probe_ms, threshold):
                return float(probe_ms), f"{probe_ms}ms"
        return float(timeout_ms), "timeout"
    raise ValueError(f"unknown policy: {policy}")


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_policy(
    spans: Sequence[Span],
    *,
    language: str,
    policy: str,
    threshold: float | None,
    probes_ms: tuple[int, ...],
    timeout_ms: int,
    baseline_ms: int,
) -> dict[str, Any]:
    if policy in {"S1", "S2"} and threshold is None:
        raise ValueError(f"{policy} needs a threshold")
    decisions = [
        _decision(span, policy, threshold, probes_ms, timeout_ms, baseline_ms) for span in spans
    ]
    hold_spans = [span for span in spans if span.label == "hold"]
    eot_spans = [span for span in spans if span.label == "eot"]
    false_cutoff_flags = [
        span.label == "hold" and span.span_duration_ms > decision_ms + TIME_EPS_MS
        for span, (decision_ms, _) in zip(spans, decisions)
    ]
    eot_latencies = [
        decision_ms for span, (decision_ms, _) in zip(spans, decisions) if span.label == "eot"
    ]
    decision_probes = [probe for _, probe in decisions]
    accepts_200 = (
        sum(_accepted(span, probes_ms[0], threshold) for span in spans)
        if threshold is not None
        else 0
    )
    eligible_200 = sum(_survives(span, probes_ms[0]) for span in spans)
    eligible_500 = sum(_survives(span, probes_ms[-1]) for span in spans)
    accepts_500 = (
        sum(_accepted(span, probes_ms[-1], threshold) for span in spans)
        if threshold is not None
        else 0
    )
    survivor_500 = [
        span
        for span in spans
        if _survives(span, probes_ms[-1])
        and threshold is not None
        and not _accepted(span, probes_ms[0], threshold)
    ]
    accepts_500_on_survivors = (
        sum(_accepted(span, probes_ms[-1], threshold) for span in survivor_500)
        if threshold is not None
        else 0
    )
    first_probe_false = {
        f"{probes_ms[0]}": sum(
            span.label == "hold" and probe == f"{probes_ms[0]}ms" and is_false
            for span, probe, is_false in zip(spans, decision_probes, false_cutoff_flags)
        ),
        f"{probes_ms[-1]}": sum(
            span.label == "hold" and probe == f"{probes_ms[-1]}ms" and is_false
            for span, probe, is_false in zip(spans, decision_probes, false_cutoff_flags)
        ),
    }
    first_probe_true = {
        f"{probes_ms[0]}": sum(
            span.label == "eot" and probe == f"{probes_ms[0]}ms"
            for span, probe in zip(spans, decision_probes)
        ),
        f"{probes_ms[-1]}": sum(
            span.label == "eot" and probe == f"{probes_ms[-1]}ms"
            for span, probe in zip(spans, decision_probes)
        ),
    }
    timeout_flags = [
        probe == "timeout"
        and (span.label == "eot" or span.span_duration_ms > timeout_ms + TIME_EPS_MS)
        for span, probe in zip(spans, decision_probes)
    ]
    timeout_count = sum(timeout_flags)
    false_count = sum(false_cutoff_flags)
    eot_detect_count = sum(
        span.label == "eot" and probe in {f"{probe_ms}ms" for probe_ms in probes_ms}
        for span, probe in zip(spans, decision_probes)
    )
    row: dict[str, Any] = {
        "language": language,
        "aggregate": "language",
        "policy": policy,
        "threshold": threshold,
        "probe_ms": (
            "/".join(str(probe_ms) for probe_ms in probes_ms) if policy in {"S1", "S2"} else None
        ),
        "timeout_ms": timeout_ms,
        "baseline_ms": baseline_ms,
        "n_spans": len(spans),
        "n_hold_spans": len(hold_spans),
        "n_eot_spans": len(eot_spans),
        "false_cutoff_count": false_count,
        "false_cutoff_rate": _safe_rate(false_count, len(hold_spans)),
        "mean_latency_ms": sum(eot_latencies) / len(eot_latencies) if eot_latencies else None,
        "p50_latency_ms": _percentile(eot_latencies, 0.50),
        "p90_latency_ms": _percentile(eot_latencies, 0.90),
        "p95_latency_ms": _percentile(eot_latencies, 0.95),
        "probe_200_acceptance_rate": (
            _safe_rate(accepts_200, eligible_200) if threshold is not None else None
        ),
        "probe_500_acceptance_rate": (
            _safe_rate(accepts_500, eligible_500) if threshold is not None else None
        ),
        "probe_500_survivor_acceptance_rate": (
            _safe_rate(accepts_500_on_survivors, len(survivor_500))
            if threshold is not None
            else None
        ),
        "n_200_eligible": eligible_200,
        "n_500_eligible": eligible_500,
        "hard_timeout_count": timeout_count,
        "hard_timeout_rate": _safe_rate(timeout_count, len(spans)),
        "eot_detection_count": eot_detect_count,
        "eot_detection_rate": _safe_rate(eot_detect_count, len(eot_spans)),
        "false_cutoffs_first_introduced_200ms": first_probe_false[str(probes_ms[0])],
        "true_eot_first_detected_200ms": first_probe_true[str(probes_ms[0])],
        "false_cutoffs_first_introduced_500ms": first_probe_false[str(probes_ms[-1])],
        "true_eot_recovered_500ms": first_probe_true[str(probes_ms[-1])],
        "decision_count_200ms": decision_probes.count(f"{probes_ms[0]}ms"),
        "decision_count_500ms": decision_probes.count(f"{probes_ms[-1]}ms"),
        "decision_count_timeout": timeout_count,
        "decision_count_fixed": sum(probe.startswith("fixed_") for probe in decision_probes),
        "mean_latency_delta_vs_baseline_ms": (
            (sum(eot_latencies) / len(eot_latencies)) - baseline_ms if eot_latencies else None
        ),
        "_eot_latencies": eot_latencies,
        "_false_cutoff_count": false_count,
        "_hold_count": len(hold_spans),
        "_eot_count": len(eot_spans),
        "_timeout_count": timeout_count,
    }
    return row


def remap_product_time_spans(spans: Sequence[Span]) -> tuple[Span, ...]:
    remapped: list[Span] = []
    for span in spans:
        scores: dict[int, float] = {}
        if 200 in span.scores:
            scores[224] = span.scores[200]
        if 500 in span.scores:
            scores[512] = span.scores[500]
        remapped.append(
            Span(
                span_id=span.span_id,
                language=span.language,
                span_index=span.span_index,
                label=span.label,
                span_duration_ms=span.span_duration_ms,
                scores=scores,
            ),
        )
    return tuple(remapped)


METRIC_FIELDS = [
    "n_spans",
    "n_hold_spans",
    "n_eot_spans",
    "false_cutoff_count",
    "false_cutoff_rate",
    "mean_latency_ms",
    "p50_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "probe_200_acceptance_rate",
    "probe_500_acceptance_rate",
    "probe_500_survivor_acceptance_rate",
    "n_200_eligible",
    "n_500_eligible",
    "hard_timeout_count",
    "hard_timeout_rate",
    "eot_detection_count",
    "eot_detection_rate",
    "false_cutoffs_first_introduced_200ms",
    "true_eot_first_detected_200ms",
    "false_cutoffs_first_introduced_500ms",
    "true_eot_recovered_500ms",
    "decision_count_200ms",
    "decision_count_500ms",
    "decision_count_timeout",
    "decision_count_fixed",
    "mean_latency_delta_vs_baseline_ms",
]


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _aggregate_micro(
    spans: Sequence[Span],
    *,
    policy: str,
    threshold: float | None,
    probes_ms: tuple[int, ...],
    timeout_ms: int,
    baseline_ms: int,
) -> dict[str, Any]:
    row = evaluate_policy(
        spans,
        language="all",
        policy=policy,
        threshold=threshold,
        probes_ms=probes_ms,
        timeout_ms=timeout_ms,
        baseline_ms=baseline_ms,
    )
    row["aggregate"] = "micro"
    return row


def _aggregate_macro(
    rows: Sequence[dict[str, Any]],
    *,
    policy: str,
    threshold: float | None,
    probes_ms: tuple[int, ...],
    timeout_ms: int,
    baseline_ms: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "language": "macro",
        "aggregate": "macro",
        "policy": policy,
        "threshold": threshold,
        "probe_ms": (
            "/".join(str(probe_ms) for probe_ms in probes_ms) if policy in {"S1", "S2"} else None
        ),
        "timeout_ms": timeout_ms,
        "baseline_ms": baseline_ms,
    }
    for field in METRIC_FIELDS:
        values = [row_value[field] for row_value in rows if row_value.get(field) is not None]
        if field in {
            "n_spans",
            "n_hold_spans",
            "n_eot_spans",
            "false_cutoff_count",
            "n_200_eligible",
            "n_500_eligible",
            "hard_timeout_count",
            "eot_detection_count",
            "false_cutoffs_first_introduced_200ms",
            "true_eot_first_detected_200ms",
            "false_cutoffs_first_introduced_500ms",
            "true_eot_recovered_500ms",
            "decision_count_200ms",
            "decision_count_500ms",
            "decision_count_timeout",
            "decision_count_fixed",
        }:
            row[field] = sum(int(value) for value in values)
        else:
            row[field] = sum(float(value) for value in values) / len(values) if values else None
    row["_eot_latencies"] = []
    row["_false_cutoff_count"] = sum(int(value.get("_false_cutoff_count", 0)) for value in rows)
    row["_hold_count"] = sum(int(value.get("_hold_count", 0)) for value in rows)
    row["_eot_count"] = sum(int(value.get("_eot_count", 0)) for value in rows)
    row["_timeout_count"] = sum(int(value.get("_timeout_count", 0)) for value in rows)
    return row


def _pareto(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("mean_latency_ms") is not None and row.get("false_cutoff_rate") is not None
    ]
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            no_worse = (
                other["false_cutoff_rate"] <= candidate["false_cutoff_rate"] + EPS
                and other["mean_latency_ms"] <= candidate["mean_latency_ms"] + EPS
            )
            strictly_better = (
                other["false_cutoff_rate"] < candidate["false_cutoff_rate"] - EPS
                or other["mean_latency_ms"] < candidate["mean_latency_ms"] - EPS
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (
            row["false_cutoff_rate"],
            row["mean_latency_ms"],
            row["policy"],
            row.get("threshold") is None,
            row.get("threshold") or 0.0,
        ),
    )


def _select_conservative(
    rows: Sequence[dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, Any] | None:
    feasible = [
        row
        for row in rows
        if row["false_cutoff_rate"] <= baseline["false_cutoff_rate"] + EPS
        and row["mean_latency_ms"] is not None
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda row: (
            row["mean_latency_ms"],
            row["false_cutoff_rate"],
            -(row["threshold"] or 0.0),
        ),
    )


def _select_balanced(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [
        row
        for row in rows
        if row["mean_latency_ms"] is not None
        and row["mean_latency_ms"] <= BALANCED_LATENCY_LIMIT_MS + EPS
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda row: (
            row["false_cutoff_rate"],
            row["mean_latency_ms"],
            -(row["threshold"] or 0.0),
        ),
    )


def _is_adjacent_to_frontier(
    row: dict[str, Any], frontier: Sequence[dict[str, Any]], threshold_step: float
) -> bool:
    if any(
        row["policy"] == frontier_row["policy"]
        and row.get("threshold") is not None
        and frontier_row.get("threshold") is not None
        and abs(row["threshold"] - frontier_row["threshold"]) <= threshold_step + EPS
        for frontier_row in frontier
    ):
        return True
    return False


def _matched_row(
    row: dict[str, Any], candidates: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda other: (
            abs(other["false_cutoff_rate"] - row["false_cutoff_rate"]),
            other["mean_latency_ms"],
            -(other["threshold"] or 0.0),
        ),
    )


def _increment_row(
    s1: dict[str, Any],
    s2: dict[str, Any],
    *,
    language: str,
    threshold: float,
    matched_candidates: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    matched = _matched_row(s2, matched_candidates or [s1])
    return {
        "language": language,
        "threshold": threshold,
        "s1_false_cutoff_rate": s1["false_cutoff_rate"],
        "s2_false_cutoff_rate": s2["false_cutoff_rate"],
        "false_cutoff_rate_change": s2["false_cutoff_rate"] - s1["false_cutoff_rate"],
        "additional_true_eot_detected_at_500ms": s2["true_eot_recovered_500ms"],
        "additional_false_cutoffs_introduced_at_500ms": s2["false_cutoffs_first_introduced_500ms"],
        "s1_mean_latency_ms": s1["mean_latency_ms"],
        "s2_mean_latency_ms": s2["mean_latency_ms"],
        "mean_latency_change_ms": s2["mean_latency_ms"] - s1["mean_latency_ms"],
        "s1_timeout_rate": s1["hard_timeout_rate"],
        "s2_timeout_rate": s2["hard_timeout_rate"],
        "timeout_rate_change": s2["hard_timeout_rate"] - s1["hard_timeout_rate"],
        "matched_s1_threshold": matched["threshold"] if matched else None,
        "matched_s1_false_cutoff_rate": matched["false_cutoff_rate"] if matched else None,
        "matched_false_cutoff_delta": (
            s2["false_cutoff_rate"] - matched["false_cutoff_rate"] if matched else None
        ),
        "matched_mean_latency_delta_ms": (
            s2["mean_latency_ms"] - matched["mean_latency_ms"] if matched else None
        ),
        "matched_timeout_rate_delta": (
            s2["hard_timeout_rate"] - matched["hard_timeout_rate"] if matched else None
        ),
        "matched_cutoff_within_0_5pp": (
            s2["false_cutoff_rate"] <= matched["false_cutoff_rate"] + 0.005 + EPS
            if matched
            else False
        ),
        "useful_500ms_at_matched_point": (
            bool(matched)
            and s2["false_cutoff_rate"] <= matched["false_cutoff_rate"] + 0.005 + EPS
            and (
                matched["mean_latency_ms"] - s2["mean_latency_ms"] >= 25.0 - EPS
                or matched["hard_timeout_rate"] - s2["hard_timeout_rate"] >= 0.05 - EPS
            )
        ),
    }


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _finite_cpu_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _cpu_result(output_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    for name in ("cpu_benchmark.json", "smart_turn_cpu_benchmark.json"):
        path = output_dir / name
        if path.is_file():
            try:
                result = _read_json(path)
            except (OSError, ValueError) as exc:
                return None, {
                    "present": True,
                    "valid": False,
                    "preferred": False,
                    "acceptable": False,
                    "one_thread_8s_p95_ms": None,
                    "errors": [f"could not parse CPU benchmark artifact: {exc}"],
                }
            errors: list[str] = []
            warmups = result.get("warmup_calls")
            measured = result.get("measured_calls")
            if not isinstance(warmups, int) or warmups < CPU_MIN_WARMUPS:
                errors.append(f"warmup_calls must be an integer >= {CPU_MIN_WARMUPS}")
            if not isinstance(measured, int) or measured < CPU_MIN_MEASURED:
                errors.append(f"measured_calls must be an integer >= {CPU_MIN_MEASURED}")
            if result.get("sample_rate_hz") != 16000:
                errors.append("sample_rate_hz must be 16000")
            model_sha = str(result.get("model_sha256") or "")
            if len(model_sha) != 64 or any(
                character not in "0123456789abcdef" for character in model_sha.lower()
            ):
                errors.append("model_sha256 must be a 64-character hexadecimal digest")
            lengths = result.get("input_lengths_s")
            numeric_lengths = (
                [float(value) for value in lengths if _finite_cpu_number(value)]
                if isinstance(lengths, list)
                else []
            )
            if not isinstance(lengths, list) or any(
                not any(abs(value - expected) <= EPS for value in numeric_lengths)
                for expected in CPU_REQUIRED_LENGTHS_S
            ):
                errors.append(f"input_lengths_s must include {list(CPU_REQUIRED_LENGTHS_S)}")
            settings = result.get("settings")
            settings_by_name = {}
            if isinstance(settings, list):
                for setting in settings:
                    if isinstance(setting, dict) and setting.get("thread_setting") in {
                        "default",
                        "one",
                    }:
                        settings_by_name[setting["thread_setting"]] = setting
            if set(settings_by_name) != {"default", "one"}:
                errors.append("settings must contain both default and one thread results")
            for thread_setting in ("default", "one"):
                setting = settings_by_name.get(thread_setting)
                if setting is None:
                    continue
                synthetic = setting.get("synthetic")
                if not isinstance(synthetic, list):
                    errors.append(f"{thread_setting} setting is missing synthetic rows")
                    continue
                for expected in CPU_REQUIRED_LENGTHS_S:
                    matching = [
                        row
                        for row in synthetic
                        if isinstance(row, dict)
                        and _finite_cpu_number(row.get("duration_s"))
                        and abs(float(row["duration_s"]) - expected) <= EPS
                    ]
                    if len(matching) != 1:
                        errors.append(
                            f"{thread_setting} setting must have one synthetic row for {expected}s"
                        )
                    elif not _finite_cpu_number(matching[0].get("p95_ms")):
                        errors.append(f"{thread_setting} {expected}s row is missing finite p95_ms")
                real_audio = setting.get("real_audio")
                real_languages = (
                    {
                        row.get("language")
                        for row in real_audio
                        if isinstance(row, dict) and row.get("language")
                    }
                    if isinstance(real_audio, list)
                    else set()
                )
                missing_real = sorted(CPU_REQUIRED_LANGUAGES - real_languages)
                if missing_real:
                    errors.append(
                        f"{thread_setting} setting is missing real-audio languages: {missing_real}"
                    )
            one_setting = settings_by_name.get("one")
            one_row = None
            if one_setting is not None:
                one_row = next(
                    (
                        row
                        for row in one_setting.get("synthetic", [])
                        if isinstance(row, dict)
                        and _finite_cpu_number(row.get("duration_s"))
                        and abs(float(row["duration_s"]) - 8.0) <= EPS
                    ),
                    None,
                )
            one_p95 = (
                float(one_row.get("p95_ms"))
                if one_row is not None and _finite_cpu_number(one_row.get("p95_ms"))
                else None
            )
            if one_p95 is None:
                errors.append("one-thread 8-second p95 is unavailable")
            validation = {
                "present": True,
                "valid": not errors,
                "preferred": not errors and one_p95 <= CPU_PREFERRED_P95_MS,
                "acceptable": not errors and one_p95 <= CPU_ACCEPTABLE_P95_MS,
                "one_thread_8s_p95_ms": one_p95,
                "errors": errors,
            }
            return (result if not errors else None), validation
    return None, {
        "present": False,
        "valid": False,
        "preferred": False,
        "acceptable": False,
        "one_thread_8s_p95_ms": None,
        "errors": ["CPU benchmark artifact is missing"],
    }


def _report(
    output_dir: Path,
    *,
    artifacts: dict[str, Artifact],
    selected: dict[str, Any],
    increment_rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    cpu_result: dict[str, Any] | None,
) -> None:
    lines = [
        "# LiveKit EOT repeated-probe experiment",
        "",
        "## Decision",
        "",
        f"**{summary['decision']}**",
        "",
        summary["decision_explanation"],
        "",
        "This is an offline policy evaluation. It does not change PuriPuly production endpoint behavior or remove speculative translation.",
        "",
        "## Data integrity",
        "",
        "| Language | EOT spans | Hold spans | 200 ms scores | 500 ms scores | Missing required | Permitted missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for language, artifact in artifacts.items():
        validation = artifact.validation
        missing = sum(validation["predictions"]["missing_required_scores"].values())
        permitted = sum(validation["predictions"]["permitted_missing_scores"].values())
        lines.append(
            f"| {language} | {validation['eot_spans']} | {validation['hold_spans']} | "
            f"{validation['probe_score_counts']['200']} | {validation['probe_score_counts']['500']} | {missing} | {permitted} |",
        )
    lines.extend(
        [
            "",
            "## Dataset and model provenance",
            "",
            f"- Dataset: `{summary['dataset']['id']}` revision `{summary['dataset']['revision']}`; split `{summary['dataset']['split']}`.",
            f"- eot-bench revision: `{summary['eot_bench_revision'] or 'not available'}`.",
            f"- Smart Turn artifact: `{summary['smart_turn']['adapter_id']}`; model revision `{summary['smart_turn']['revision'] or 'not pinned in prediction manifest'}`.",
            f"- Prediction source: `{summary['smart_turn']['prediction_source']}`.",
            "",
            "## Selected operating points",
            "",
            "The gate point is the balanced S2 point: the lowest S2 false-cutoff rate with mean EOT latency no greater than 560 ms. Conservative points are also reported.",
            "",
            "| Language | Policy | Point | Threshold | False cutoff | Mean | P50 | P95 | Timeout | Pareto |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ],
    )
    for language in artifacts:
        language_selection = selected["languages"][language]
        for policy in ("B0", "B1", "S1", "S2"):
            if policy == "B0":
                points = [("baseline", language_selection.get("baseline"))]
            elif policy == "B1":
                points = [("timeout", language_selection.get("B1", {}).get("timeout"))]
            else:
                points = [
                    ("conservative", language_selection.get(policy, {}).get("conservative")),
                    ("balanced", language_selection.get(policy, {}).get("balanced")),
                ]
            for point_name, point in points:
                if point is None:
                    lines.append(
                        f"| {language} | {policy} | {point_name} | — | — | — | — | — | — | — |"
                    )
                    continue
                lines.append(
                    f"| {language} | {policy} | {point_name} | {_format_number(point.get('threshold'))} | "
                    f"{_format_number(point.get('false_cutoff_rate') * 100 if point.get('false_cutoff_rate') is not None else None)}% | "
                    f"{_format_number(point.get('mean_latency_ms'))} ms | {_format_number(point.get('p50_latency_ms'))} ms | "
                    f"{_format_number(point.get('p95_latency_ms'))} ms | {_format_number(point.get('hard_timeout_rate') * 100 if point.get('hard_timeout_rate') is not None else None)}% | "
                    f"{_format_number(point.get('on_or_adjacent_pareto'))} |",
                )
    lines.extend(
        [
            "",
            "## Estimated PuriPuly chunk-time view",
            "",
            "The benchmark scores remain the 200/500 ms rows. This view only remaps action times to the nearest PuriPuly schedule, 224/512/800 ms; it is not a new audio inference run.",
            "",
            "| Language | Threshold | False cutoff | Mean | P50 | P95 | Timeout |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ],
    )
    for language in artifacts:
        product_point = selected["languages"][language].get("product_time_gate")
        if product_point is None:
            lines.append(f"| {language} | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {language} | {_format_number(product_point.get('threshold'))} | "
            f"{_format_number(product_point.get('false_cutoff_rate') * 100)}% | "
            f"{_format_number(product_point.get('mean_latency_ms'))} ms | "
            f"{_format_number(product_point.get('p50_latency_ms'))} ms | "
            f"{_format_number(product_point.get('p95_latency_ms'))} ms | "
            f"{_format_number(product_point.get('hard_timeout_rate') * 100)}% |",
        )
    lines.extend(
        [
            "",
            "## Value of the 500 ms probe",
            "",
            "The full threshold-level comparison is in `probe_500_increment.csv`; the rows below use each language's balanced S2 gate point and its closest S1 false-cutoff operating point.",
            "",
            "| Language | S2 threshold | Matched S1 threshold | True EOT recovered | New false cutoffs | Same-threshold mean change | Same-threshold timeout change | Matched mean change | Matched cutoff delta | Useful |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ],
    )
    for language in artifacts:
        increment = selected["languages"][language].get("increment_at_gate")
        if increment is None:
            lines.append(f"| {language} | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {language} | {_format_number(increment.get('threshold'))} | {_format_number(increment.get('matched_s1_threshold'))} | "
            f"{increment.get('additional_true_eot_detected_at_500ms')} | {increment.get('additional_false_cutoffs_introduced_at_500ms')} | "
            f"{_format_number(increment.get('mean_latency_change_ms'))} ms | "
            f"{_format_number(increment.get('timeout_rate_change') * 100 if increment.get('timeout_rate_change') is not None else None)} pp | "
            f"{_format_number(increment.get('matched_mean_latency_delta_ms'))} ms | "
            f"{_format_number(increment.get('matched_false_cutoff_delta') * 100 if increment.get('matched_false_cutoff_delta') is not None else None)} pp | "
            f"{_format_number(increment.get('useful_500ms_at_matched_point'))} |",
        )
    lines.extend(
        [
            "",
            "## Aggregate gate results",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Languages passing S2 false-cutoff comparison | {summary['gate_checks']['s2_better_than_b0_count']} / 4 |",
            f"| Languages within 1 percentage point of B0 | {summary['gate_checks']['no_language_regression_count']} / 4 |",
            f"| Macro mean EOT latency | {_format_number(summary['gate_checks']['macro_mean_latency_ms'])} ms |",
            f"| Micro false-cutoff rate | {_format_number(summary['gate_checks']['micro_false_cutoff_rate'] * 100)}% |",
            f"| Micro mean EOT latency | {_format_number(summary['gate_checks']['micro_mean_latency_ms'])} ms |",
            f"| Micro P50 / P95 EOT latency | {_format_number(summary['gate_checks']['micro_p50_latency_ms'])} / {_format_number(summary['gate_checks']['micro_p95_latency_ms'])} ms |",
            f"| Micro timeout rate | {_format_number(summary['gate_checks']['micro_timeout_rate'] * 100)}% |",
            f"| Languages S2 <= matched S1 latency | {summary['gate_checks']['s2_better_or_equal_s1_count']} / 4 |",
            f"| Gate points on/adjacent to Pareto | {summary['gate_checks']['pareto_count']} / 4 |",
            f"| CPU acceptable gate (1-thread 8s p95 <= 200 ms) | {summary['gate_checks']['criterion_6_cpu_gate']} |",
            "",
            "## CPU benchmark",
            "",
        ],
    )
    if cpu_result is None:
        if summary["cpu_validation"]["present"]:
            lines.append("CPU benchmark artifact failed validation.")
        else:
            lines.append("CPU benchmark artifact was not present when this report was generated.")
        lines.append(
            f"CPU gate acceptable: `no`; validation errors: {', '.join(summary['cpu_validation']['errors'])}."
        )
    else:
        lines.extend(
            [
                f"The CPU artifact reports `{cpu_result.get('warmup_calls')}` warm-up calls and `{cpu_result.get('measured_calls')}` measured calls per input. Results include preprocessing and one ONNX inference with a reused session.",
                "",
                "| Audio length | Thread setting | p50 | p95 | p99 |",
                "| ---: | --- | ---: | ---: | ---: |",
            ],
        )
        for setting in cpu_result.get("settings", []):
            for row in setting.get("synthetic", []):
                lines.append(
                    f"| {row.get('duration_s')} s | {setting.get('thread_setting')} | {_format_number(row.get('p50_ms'))} ms | {_format_number(row.get('p95_ms'))} ms | {_format_number(row.get('p99_ms'))} ms |",
                )
        lines.extend(
            [
                "",
                "| Thread setting | Cold init | Session RSS increase | 8s p95 |",
                "| --- | ---: | ---: | ---: |",
            ],
        )
        for setting in cpu_result.get("settings", []):
            eight_second = next(
                (row for row in setting.get("synthetic", []) if row.get("duration_s") == 8.0), None
            )
            lines.append(
                f"| {setting.get('thread_setting')} | {_format_number(setting.get('cold_initialization_ms'))} ms | "
                f"{_format_number(setting.get('memory_increase_mb'))} MB | "
                f"{_format_number(eight_second.get('p95_ms') if eight_second else None)} ms |",
            )
        lines.extend(
            [
                "",
                "Real-audio sample timing (each language uses one validation row; effective input is capped at 8 seconds):",
                "",
                "| Language | Sample ID | Thread setting | P50 | P95 | P99 |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ],
        )
        for setting in cpu_result.get("settings", []):
            for row in setting.get("real_audio", []):
                lines.append(
                    f"| {row.get('language')} | {row.get('sample_id') or '—'} | {setting.get('thread_setting')} | "
                    f"{_format_number(row.get('p50_ms'))} ms | {_format_number(row.get('p95_ms'))} ms | {_format_number(row.get('p99_ms'))} ms |",
                )
        cpu_validation = summary["cpu_validation"]
        lines.extend(
            [
                "",
                f"The 1-thread 8-second p95 is `{_format_number(cpu_validation.get('one_thread_8s_p95_ms'))} ms`; preferred gate (<=150 ms): `{_format_number(cpu_validation.get('preferred'))}`, acceptable gate (<=200 ms): `{_format_number(cpu_validation.get('acceptable'))}`.",
            ],
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Scores at 200 ms and 500 ms are from fresh causal prediction rows in the official Smart Turn artifact; this evaluator does not treat the two scores as independent votes.",
            "- A hold span is a false cutoff only when its real pause duration is strictly longer than the policy decision time.",
            "- Hard-timeout rate counts actual 800 ms boundary events: unresolved EOT spans plus unresolved hold spans whose pauses last beyond 800 ms; a hold that resumes before 800 ms is not counted as a timeout.",
            "- Raw Smart Turn scores are used only for threshold sweep decisions and are not presented as calibrated probabilities.",
            "- The policy scores use the official committed Smart Turn v3.2 GPU prediction artifact; the CPU int8 model is benchmarked separately as required by the CPU gate.",
            "",
            "## Artifacts",
            "",
            "- `data_validation.json`",
            "- `policy_sweep_all.csv` and `policy_sweep_{ko,ja,en,zh}.csv`",
            "- `pareto_{ko,ja,en,zh}.csv`",
            "- `probe_500_increment.csv`",
            "- `product_time_view.csv`",
            "- `selected_operating_points.json`",
            "- `summary.json`",
        ],
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    *,
    predictions_root: Path,
    languages: Sequence[str],
    probes_ms: Sequence[int],
    timeout_ms: int,
    baseline_ms: int,
    threshold_step: float,
    output_dir: Path,
    eot_bench_revision: str | None = None,
    dataset_revision: str | None = None,
    smart_turn_revision: str | None = None,
    smart_turn_adapter_id: str | None = None,
) -> dict[str, Any]:
    root = predictions_root.resolve()
    language_values = tuple(dict.fromkeys(str(language).strip().lower() for language in languages))
    if not language_values or any(not language for language in language_values):
        raise ValueError("languages must be non-empty")
    probe_values = tuple(int(value) for value in probes_ms)
    if (
        len(probe_values) != 2
        or tuple(sorted(probe_values)) != probe_values
        or any(value <= 0 for value in probe_values)
    ):
        raise ValueError("probe-ms must contain two positive, strictly increasing values")
    if timeout_ms <= probe_values[-1]:
        raise ValueError("timeout-ms must be greater than the final probe")
    if baseline_ms <= 0:
        raise ValueError("baseline-ms must be positive")
    thresholds = thresholds_from_step(threshold_step)
    artifacts = {
        language: load_artifact(root, language, probe_values) for language in language_values
    }
    spans_by_language = {language: artifact.spans for language, artifact in artifacts.items()}
    all_spans = tuple(span for language in language_values for span in spans_by_language[language])

    language_sweeps: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for language in language_values:
        spans = spans_by_language[language]
        rows = [
            evaluate_policy(
                spans,
                language=language,
                policy="B0",
                threshold=None,
                probes_ms=probe_values,
                timeout_ms=timeout_ms,
                baseline_ms=baseline_ms,
            ),
            evaluate_policy(
                spans,
                language=language,
                policy="B1",
                threshold=None,
                probes_ms=probe_values,
                timeout_ms=timeout_ms,
                baseline_ms=baseline_ms,
            ),
        ]
        for policy in ("S1", "S2"):
            rows.extend(
                evaluate_policy(
                    spans,
                    language=language,
                    policy=policy,
                    threshold=threshold,
                    probes_ms=probe_values,
                    timeout_ms=timeout_ms,
                    baseline_ms=baseline_ms,
                )
                for threshold in thresholds
            )
        language_sweeps[language] = rows
        all_rows.extend(_public_row(row) for row in rows)

    aggregate_rows: list[dict[str, Any]] = []
    for policy in ("B0", "B1", "S1", "S2"):
        policy_rows = [
            row
            for language in language_values
            for row in language_sweeps[language]
            if row["policy"] == policy
        ]
        thresholds_for_policy = (None,) if policy in {"B0", "B1"} else thresholds
        for threshold in thresholds_for_policy:
            matching = [row for row in policy_rows if row.get("threshold") == threshold]
            aggregate_rows.append(
                _public_row(
                    _aggregate_macro(
                        matching,
                        policy=policy,
                        threshold=threshold,
                        probes_ms=probe_values,
                        timeout_ms=timeout_ms,
                        baseline_ms=baseline_ms,
                    )
                )
            )
            aggregate_rows.append(
                _public_row(
                    _aggregate_micro(
                        all_spans,
                        policy=policy,
                        threshold=threshold,
                        probes_ms=probe_values,
                        timeout_ms=timeout_ms,
                        baseline_ms=baseline_ms,
                    )
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "language",
        "aggregate",
        "policy",
        "threshold",
        "probe_ms",
        "timeout_ms",
        "baseline_ms",
    ] + METRIC_FIELDS
    for language, rows in language_sweeps.items():
        _write_csv(
            output_dir / f"policy_sweep_{language}.csv",
            [_public_row(row) for row in rows],
            fieldnames,
        )
    _write_csv(output_dir / "policy_sweep_all.csv", all_rows + aggregate_rows, fieldnames)

    product_probe_values = (224, 512)
    product_rows: list[dict[str, Any]] = []
    product_sweeps: dict[str, list[dict[str, Any]]] = {}
    product_fieldnames = ["timing_view"] + fieldnames
    for language in language_values:
        product_spans = remap_product_time_spans(spans_by_language[language])
        rows = [
            evaluate_policy(
                product_spans,
                language=language,
                policy="B0",
                threshold=None,
                probes_ms=product_probe_values,
                timeout_ms=timeout_ms,
                baseline_ms=baseline_ms,
            ),
            evaluate_policy(
                product_spans,
                language=language,
                policy="B1",
                threshold=None,
                probes_ms=product_probe_values,
                timeout_ms=timeout_ms,
                baseline_ms=baseline_ms,
            ),
        ]
        for policy in ("S1", "S2"):
            rows.extend(
                evaluate_policy(
                    product_spans,
                    language=language,
                    policy=policy,
                    threshold=threshold,
                    probes_ms=product_probe_values,
                    timeout_ms=timeout_ms,
                    baseline_ms=baseline_ms,
                )
                for threshold in thresholds
            )
        for row in rows:
            row["timing_view"] = "224/512/800ms_estimate"
        product_sweeps[language] = rows
        product_rows.extend(_public_row(row) for row in rows)
    _write_csv(output_dir / "product_time_view.csv", product_rows, product_fieldnames)

    selected_languages: dict[str, Any] = {}
    increment_rows: list[dict[str, Any]] = []
    pareto_rows_by_language: dict[str, list[dict[str, Any]]] = {}
    for language in language_values:
        rows = language_sweeps[language]
        baseline = next(row for row in rows if row["policy"] == "B0")
        timeout_baseline = next(row for row in rows if row["policy"] == "B1")
        model_rows = [row for row in rows if row["policy"] in {"S1", "S2"}]
        frontier = _pareto(model_rows + [baseline, timeout_baseline])
        pareto_rows_by_language[language] = [_public_row(row) for row in frontier]
        _write_csv(
            output_dir / f"pareto_{language}.csv", pareto_rows_by_language[language], fieldnames
        )
        points: dict[str, Any] = {
            "baseline": _public_row(baseline),
            "B0": {"baseline": _public_row(baseline)},
            "B1": {"timeout": _public_row(timeout_baseline)},
        }
        for policy in ("S1", "S2"):
            policy_rows = [row for row in rows if row["policy"] == policy]
            conservative = _select_conservative(policy_rows, baseline)
            balanced = _select_balanced(policy_rows)
            for name, point in (("conservative", conservative), ("balanced", balanced)):
                if point is not None:
                    point["on_or_adjacent_pareto"] = point in frontier or _is_adjacent_to_frontier(
                        point, frontier, threshold_step
                    )
            points[policy] = {
                "conservative": _public_row(conservative) if conservative else None,
                "balanced": _public_row(balanced) if balanced else None,
            }
        gate = points["S2"].get("balanced")
        gate_increment = None
        if gate is not None:
            s1_rows = [row for row in rows if row["policy"] == "S1"]
            s2_rows = [
                row
                for row in rows
                if row["policy"] == "S2" and row["threshold"] == gate["threshold"]
            ]
            if s2_rows:
                gate_increment = _increment_row(
                    s1_rows[[row["threshold"] for row in s1_rows].index(gate["threshold"])],
                    s2_rows[0],
                    language=language,
                    threshold=gate["threshold"],
                    matched_candidates=s1_rows,
                )
                increment_rows.append(gate_increment)
        points["increment_at_gate"] = gate_increment
        product_gate = None
        if gate is not None:
            product_gate = next(
                (
                    row
                    for row in product_sweeps[language]
                    if row["policy"] == "S2" and row["threshold"] == gate["threshold"]
                ),
                None,
            )
        points["product_time_gate"] = _public_row(product_gate) if product_gate else None
        selected_languages[language] = points

    threshold_increment_rows: list[dict[str, Any]] = []
    for language in language_values:
        s1_rows = {
            row["threshold"]: row for row in language_sweeps[language] if row["policy"] == "S1"
        }
        s2_rows = {
            row["threshold"]: row for row in language_sweeps[language] if row["policy"] == "S2"
        }
        for threshold in thresholds:
            threshold_increment_rows.append(
                _increment_row(
                    s1_rows[threshold],
                    s2_rows[threshold],
                    language=language,
                    threshold=threshold,
                    matched_candidates=list(s1_rows.values()),
                )
            )
    increment_fields = (
        list(threshold_increment_rows[0]) if threshold_increment_rows else ["language", "threshold"]
    )
    _write_csv(output_dir / "probe_500_increment.csv", threshold_increment_rows, increment_fields)

    gate_points = [
        selected_languages[language]["S2"].get("balanced") for language in language_values
    ]
    valid_gate_points = [point for point in gate_points if point is not None]
    b0_points = [selected_languages[language]["baseline"] for language in language_values]
    matched_s1_points: list[dict[str, Any] | None] = []
    for language, gate in zip(language_values, gate_points):
        if gate is None:
            matched_s1_points.append(None)
            continue
        s1_rows = [row for row in language_sweeps[language] if row["policy"] == "S1"]
        matched_s1_points.append(_matched_row(gate, s1_rows))
    s2_better_b0 = sum(
        gate is not None and gate["false_cutoff_rate"] <= b0["false_cutoff_rate"] + EPS
        for gate, b0 in zip(gate_points, b0_points)
    )
    no_regression = sum(
        gate is not None and gate["false_cutoff_rate"] - b0["false_cutoff_rate"] <= 0.01 + EPS
        for gate, b0 in zip(gate_points, b0_points)
    )
    macro_mean_latency = (
        sum(point["mean_latency_ms"] for point in valid_gate_points) / len(valid_gate_points)
        if valid_gate_points
        else None
    )
    gate_original_rows = [
        (
            next(
                (
                    row
                    for row in language_sweeps[language]
                    if row["policy"] == "S2" and row["threshold"] == gate["threshold"]
                ),
                None,
            )
            if gate is not None
            else None
        )
        for language, gate in zip(language_values, gate_points)
    ]
    gate_original_rows = [row for row in gate_original_rows if row is not None]
    gate_eot_latencies = [
        latency for row in gate_original_rows for latency in row["_eot_latencies"]
    ]
    micro_false_count = sum(row["_false_cutoff_count"] for row in gate_original_rows)
    micro_hold_count = sum(row["_hold_count"] for row in gate_original_rows)
    micro_timeout_count = sum(row["_timeout_count"] for row in gate_original_rows)
    micro_eot_count = sum(row["_eot_count"] for row in gate_original_rows)
    micro_false_cutoff_rate = _safe_rate(micro_false_count, micro_hold_count)
    micro_timeout_rate = _safe_rate(
        micro_timeout_count, sum(row["n_spans"] for row in gate_original_rows)
    )
    micro_mean_latency = (
        sum(gate_eot_latencies) / len(gate_eot_latencies) if gate_eot_latencies else None
    )
    s2_better_s1 = sum(
        gate is not None
        and matched is not None
        and gate["false_cutoff_rate"] <= matched["false_cutoff_rate"] + 0.005 + EPS
        and gate["mean_latency_ms"] <= matched["mean_latency_ms"] + EPS
        for gate, matched in zip(gate_points, matched_s1_points)
    )
    cpu_result, cpu_validation = _cpu_result(output_dir)
    pareto_count = sum(
        bool(selected_languages[language]["S2"].get("balanced", {}).get("on_or_adjacent_pareto"))
        for language in language_values
    )
    gate_checks = {
        "valid_gate_point_count": len(valid_gate_points),
        "s2_better_than_b0_count": s2_better_b0,
        "no_language_regression_count": no_regression,
        "macro_mean_latency_ms": macro_mean_latency,
        "micro_false_cutoff_rate": micro_false_cutoff_rate,
        "micro_mean_latency_ms": micro_mean_latency,
        "micro_p50_latency_ms": _percentile(gate_eot_latencies, 0.50),
        "micro_p95_latency_ms": _percentile(gate_eot_latencies, 0.95),
        "micro_timeout_rate": micro_timeout_rate,
        "micro_eot_count": micro_eot_count,
        "s2_better_or_equal_s1_count": s2_better_s1,
        "pareto_count": pareto_count,
        "criterion_1": s2_better_b0 >= 3,
        "criterion_2": no_regression == len(language_values),
        "criterion_3": macro_mean_latency is not None
        and macro_mean_latency <= BALANCED_LATENCY_LIMIT_MS + EPS,
        "criterion_4": s2_better_s1 >= 3,
        "criterion_5": pareto_count == len(language_values),
        "criterion_6_cpu_gate": bool(cpu_validation["acceptable"]),
    }
    policy_criteria = [gate_checks[f"criterion_{index}"] for index in range(1, 6)]
    all_criteria = all(policy_criteria) and gate_checks["criterion_6_cpu_gate"]
    passed_criteria = sum(policy_criteria) + int(gate_checks["criterion_6_cpu_gate"])
    if all_criteria:
        decision = "PASS"
    elif len(valid_gate_points) == len(language_values) and passed_criteria >= 3:
        decision = "PARTIAL"
    else:
        decision = "FAIL"
    decision_explanation = (
        f"At the balanced S2 gate point, {s2_better_b0}/4 languages improve or match B0 false-cutoff rate, "
        f"{no_regression}/4 stay within the 1 percentage-point regression allowance, "
        f"the macro mean EOT latency is {_format_number(macro_mean_latency)} ms, and "
        f"{s2_better_s1}/4 are no slower than S1 at a matched false-cutoff point. "
        f"{pareto_count}/4 gate points are on or adjacent to a Pareto frontier. "
        f"CPU acceptable gate: {str(bool(cpu_validation['acceptable'])).lower()}."
    )
    dataset_revision_value = dataset_revision or "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
    model_manifest = next(iter(artifacts.values())).manifest
    model_data = (
        model_manifest.get("model") if isinstance(model_manifest.get("model"), dict) else {}
    )
    adapter_id = smart_turn_adapter_id or str(
        model_data.get("adapter_id") or model_manifest.get("adapter_id") or "SmartTurn v3.2"
    )
    model_revision_value = (
        smart_turn_revision or model_data.get("revision") or model_manifest.get("model_revision")
    )
    validation_payload = {
        "dataset": {
            "id": DATASET_ID,
            "revision": dataset_revision_value,
            "split": str(
                next(iter(artifacts.values())).manifest.get("split")
                or next(iter(artifacts.values())).manifest.get("dataset", {}).get("split")
                or "validation"
            ),
        },
        "eot_bench_revision": eot_bench_revision,
        "languages": list(language_values),
        "probes_ms": list(probe_values),
        "timeout_ms": timeout_ms,
        "baseline_ms": baseline_ms,
        "threshold_step": threshold_step,
        "threshold_count": len(thresholds),
        "product_time_view": {
            "probe_ms": [224, 512],
            "timeout_ms": timeout_ms,
            "score_mapping": {"224": 200, "512": 500},
            "artifact": "product_time_view.csv",
        },
        "language_validation": {
            language: artifact.validation for language, artifact in artifacts.items()
        },
    }
    _write_json(output_dir / "data_validation.json", validation_payload)
    selected_payload = {
        "selection_definition": {
            "conservative": "lowest mean latency subject to false-cutoff rate <= fixed 512ms B0",
            "balanced": f"lowest false-cutoff rate subject to mean EOT latency <= {BALANCED_LATENCY_LIMIT_MS:g}ms",
            "gate": "balanced S2 point",
        },
        "languages": selected_languages,
        "gate_checks": gate_checks,
        "cpu_validation": cpu_validation,
        "product_time_view": {
            language: selected_languages[language].get("product_time_gate")
            for language in language_values
        },
        "aggregate_views": {
            "macro": {
                "mean_latency_ms": macro_mean_latency,
                "false_cutoff_rate": (
                    sum(point["false_cutoff_rate"] for point in valid_gate_points)
                    / len(valid_gate_points)
                    if valid_gate_points
                    else None
                ),
            },
            "micro": {
                "mean_latency_ms": micro_mean_latency,
                "p50_latency_ms": _percentile(gate_eot_latencies, 0.50),
                "p95_latency_ms": _percentile(gate_eot_latencies, 0.95),
                "false_cutoff_rate": micro_false_cutoff_rate,
                "timeout_rate": micro_timeout_rate,
            },
        },
    }
    _write_json(output_dir / "selected_operating_points.json", selected_payload)
    summary = {
        "decision": decision,
        "decision_explanation": decision_explanation,
        "dataset": validation_payload["dataset"],
        "eot_bench_revision": eot_bench_revision,
        "smart_turn": {
            "adapter_id": adapter_id,
            "revision": model_revision_value,
            "prediction_source": "official eot-bench committed predictions.parquet",
        },
        "languages": list(language_values),
        "span_counts": {
            language: {
                "eot": artifacts[language].validation["eot_spans"],
                "hold": artifacts[language].validation["hold_spans"],
                "total": len(artifacts[language].spans),
            }
            for language in language_values
        },
        "gate_checks": gate_checks,
        "product_time_view": {
            language: selected_languages[language].get("product_time_gate")
            for language in language_values
        },
        "aggregate_views": {
            "macro": {
                "mean_latency_ms": macro_mean_latency,
                "false_cutoff_rate": (
                    sum(point["false_cutoff_rate"] for point in valid_gate_points)
                    / len(valid_gate_points)
                    if valid_gate_points
                    else None
                ),
            },
            "micro": {
                "mean_latency_ms": micro_mean_latency,
                "p50_latency_ms": _percentile(gate_eot_latencies, 0.50),
                "p95_latency_ms": _percentile(gate_eot_latencies, 0.95),
                "false_cutoff_rate": micro_false_cutoff_rate,
                "timeout_rate": micro_timeout_rate,
            },
        },
        "incremental_500ms": increment_rows,
        "cpu_benchmark_present": cpu_validation["present"],
        "cpu_validation": cpu_validation,
    }
    _write_json(output_dir / "summary.json", summary)
    _report(
        output_dir,
        artifacts=artifacts,
        selected=selected_payload,
        increment_rows=increment_rows,
        summary=summary,
        cpu_result=cpu_result,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LiveKit EoT repeated Smart Turn probes.")
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--probe-ms", nargs=2, type=int, default=list(DEFAULT_PROBES_MS))
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--baseline-ms", type=int, default=DEFAULT_BASELINE_MS)
    parser.add_argument("--threshold-step", type=float, default=DEFAULT_THRESHOLD_STEP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eot-bench-revision", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--smart-turn-revision", default=None)
    parser.add_argument("--smart-turn-adapter-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = run_evaluation(
            predictions_root=args.predictions_root,
            languages=args.languages,
            probes_ms=args.probe_ms,
            timeout_ms=args.timeout_ms,
            baseline_ms=args.baseline_ms,
            threshold_step=args.threshold_step,
            output_dir=args.output_dir,
            eot_bench_revision=args.eot_bench_revision,
            dataset_revision=args.dataset_revision,
            smart_turn_revision=args.smart_turn_revision,
            smart_turn_adapter_id=args.smart_turn_adapter_id,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"eot experiment validation/evaluation failed: {exc}") from exc
    print(
        json.dumps(
            {"decision": summary["decision"], "output_dir": str(args.output_dir.resolve())},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
