from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.eot_experiment.evaluate_repeated_probes import (
    Span,
    _aggregate_macro,
    _aggregate_micro,
    _cpu_result,
    _increment_row,
    evaluate_policy,
    load_artifact,
    thresholds_from_step,
)

PROBES = (200, 500)


def make_span(
    label: str,
    duration_ms: float,
    *,
    score_200: float | None = None,
    score_500: float | None = None,
    language: str = "en",
    span_id: str = "turn",
) -> Span:
    scores = {}
    if score_200 is not None:
        scores[200] = score_200
    if score_500 is not None:
        scores[500] = score_500
    return Span(span_id, language, 0, label, duration_ms, scores)


def evaluate(spans: list[Span], policy: str, threshold: float = 0.5) -> dict:
    return evaluate_policy(
        spans,
        language="en",
        policy=policy,
        threshold=threshold if policy in {"S1", "S2"} else None,
        probes_ms=PROBES,
        timeout_ms=800,
        baseline_ms=512,
    )


def test_hold_shorter_than_200ms_cannot_be_cut_by_probe() -> None:
    row = evaluate([make_span("hold", 150)], "S2")
    assert row["decision_count_200ms"] == 0
    assert row["decision_count_timeout"] == 0
    assert row["hard_timeout_count"] == 0
    assert row["false_cutoff_count"] == 0


def test_policy_duration_boundaries_use_strict_probe_times() -> None:
    exact = evaluate([make_span("hold", 200, score_200=0.9)], "S2")
    beyond = evaluate([make_span("hold", 200.063, score_200=0.9)], "S2")
    assert exact["false_cutoff_count"] == 0
    assert beyond["false_cutoff_count"] == 1


def test_hold_between_200_and_500ms_can_only_be_cut_by_first_probe() -> None:
    row = evaluate([make_span("hold", 300, score_200=0.9)], "S2")
    assert row["decision_count_200ms"] == 1
    assert row["decision_count_500ms"] == 0
    assert row["false_cutoff_count"] == 1


def test_hold_between_500_and_800ms_can_be_cut_by_either_probe() -> None:
    first = evaluate([make_span("hold", 700, score_200=0.9, score_500=0.9)], "S2")
    second = evaluate([make_span("hold", 700, score_200=0.1, score_500=0.9)], "S2")
    assert first["false_cutoff_count"] == 1
    assert second["false_cutoffs_first_introduced_500ms"] == 1


def test_hold_longer_than_800ms_is_cut_by_timeout() -> None:
    row = evaluate([make_span("hold", 900, score_200=0.1, score_500=0.1)], "S2")
    assert row["decision_count_timeout"] == 1
    assert row["false_cutoff_count"] == 1


def test_eot_accepted_at_200ms_has_200ms_latency() -> None:
    row = evaluate([make_span("eot", 300, score_200=0.9)], "S2")
    assert row["mean_latency_ms"] == 200.0
    assert row["eot_detection_count"] == 1


def test_eot_rejected_at_200ms_and_accepted_at_500ms_has_500ms_latency() -> None:
    row = evaluate([make_span("eot", 600, score_200=0.1, score_500=0.9)], "S2")
    assert row["mean_latency_ms"] == 500.0
    assert row["true_eot_recovered_500ms"] == 1


def test_eot_rejected_at_both_probes_has_800ms_latency() -> None:
    row = evaluate([make_span("eot", 600, score_200=0.1, score_500=0.1)], "S2")
    assert row["mean_latency_ms"] == 800.0
    assert row["eot_detection_count"] == 0


def test_first_threshold_crossing_wins() -> None:
    row = evaluate([make_span("eot", 600, score_200=0.8, score_500=0.9)], "S2")
    assert row["decision_count_200ms"] == 1
    assert row["decision_count_500ms"] == 0


def test_s1_and_s2_differ_only_through_500ms_probe() -> None:
    span = [make_span("eot", 600, score_200=0.1, score_500=0.9)]
    s1 = evaluate(span, "S1")
    s2 = evaluate(span, "S2")
    assert s1["mean_latency_ms"] == 800.0
    assert s2["mean_latency_ms"] == 500.0
    assert s1["probe_200_acceptance_rate"] == s2["probe_200_acceptance_rate"]


def test_fixed_512_and_fixed_800_baselines() -> None:
    spans = [make_span("hold", 600), make_span("eot", 600, span_id="eot")]
    b0 = evaluate(spans, "B0")
    b1 = evaluate(spans, "B1")
    assert b0["false_cutoff_count"] == 1
    assert b0["mean_latency_ms"] == 512.0
    assert b1["false_cutoff_count"] == 0
    assert b1["mean_latency_ms"] == 800.0


def test_per_language_aggregation_is_separate_and_macro_differs_from_micro() -> None:
    ko_spans = [make_span("eot", 600, score_200=0.1, score_500=0.9, language="ko", span_id="ko")]
    en_spans = [make_span("eot", 300, score_200=0.9, language="en", span_id="en") for _ in range(3)]
    ko = evaluate_policy(
        ko_spans,
        language="ko",
        policy="S2",
        threshold=0.5,
        probes_ms=PROBES,
        timeout_ms=800,
        baseline_ms=512,
    )
    en = evaluate_policy(
        en_spans,
        language="en",
        policy="S2",
        threshold=0.5,
        probes_ms=PROBES,
        timeout_ms=800,
        baseline_ms=512,
    )
    macro = _aggregate_macro(
        [ko, en], policy="S2", threshold=0.5, probes_ms=PROBES, timeout_ms=800, baseline_ms=512
    )
    micro = _aggregate_micro(
        ko_spans + en_spans,
        policy="S2",
        threshold=0.5,
        probes_ms=PROBES,
        timeout_ms=800,
        baseline_ms=512,
    )
    assert ko["mean_latency_ms"] == 500.0
    assert en["mean_latency_ms"] == 200.0
    assert macro["mean_latency_ms"] == 350.0
    assert micro["mean_latency_ms"] == 275.0


def test_threshold_grid_includes_zero_and_one() -> None:
    values = thresholds_from_step(0.01)
    assert len(values) == 101
    assert values[0] == 0.0
    assert values[-1] == 1.0


def test_500ms_usefulness_uses_matched_operating_point() -> None:
    same_threshold_s1 = {
        "false_cutoff_rate": 0.10,
        "mean_latency_ms": 600.0,
        "hard_timeout_rate": 0.80,
    }
    s2 = {
        "false_cutoff_rate": 0.102,
        "mean_latency_ms": 550.0,
        "hard_timeout_rate": 0.70,
        "true_eot_recovered_500ms": 4,
        "false_cutoffs_first_introduced_500ms": 1,
    }
    matched_s1 = {
        "threshold": 0.47,
        "false_cutoff_rate": 0.101,
        "mean_latency_ms": 545.0,
        "hard_timeout_rate": 0.72,
    }
    row = _increment_row(
        same_threshold_s1,
        s2,
        language="en",
        threshold=0.48,
        matched_candidates=[matched_s1],
    )
    assert row["matched_s1_threshold"] == 0.47
    assert row["matched_mean_latency_delta_ms"] == 5.0
    assert row["useful_500ms_at_matched_point"] is False


def test_cpu_gate_requires_valid_one_thread_eight_second_result(tmp_path: Path) -> None:
    output_dir = tmp_path / "cpu"
    output_dir.mkdir()
    (output_dir / "cpu_benchmark.json").write_text(
        json.dumps(
            {
                "model_sha256": "0" * 64,
                "sample_rate_hz": 16000,
                "warmup_calls": 10,
                "measured_calls": 100,
                "input_lengths_s": [0.5, 1.0, 2.0, 4.0, 8.0],
                "settings": [
                    {
                        "thread_setting": thread_setting,
                        "synthetic": [
                            {"duration_s": duration, "p95_ms": 100.0}
                            for duration in (0.5, 1.0, 2.0, 4.0, 8.0)
                        ],
                        "real_audio": [
                            {"language": language} for language in ("ko", "ja", "en", "zh")
                        ],
                    }
                    for thread_setting in ("default", "one")
                ],
            }
        ),
        encoding="utf-8",
    )
    _, validation = _cpu_result(output_dir)
    assert validation["valid"] is True
    assert validation["acceptable"] is True

    malformed = output_dir / "cpu_benchmark.json"
    malformed.write_text("{}", encoding="utf-8")
    _, validation = _cpu_result(output_dir)
    assert validation["valid"] is False
    assert validation["acceptable"] is False


def write_artifact(
    root: Path,
    *,
    duration_s: float,
    prediction_rows: list[dict],
    dataset: str = "livekit/eot-bench-data",
    include_span_set: bool = True,
) -> Path:
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    language_root = root / "ko"
    model_root = language_root / "smart_turn_audio_adapter__test"
    model_root.mkdir(parents=True)
    hold_row = {
        "id": "hold",
        "language": "ko",
        "span_index": 0,
        "timestamp": 0.0,
        "silence_dur": 0.0,
        "p_eot": 0.1,
        "label": "hold",
    }
    parquet.write_table(
        arrow.Table.from_pylist([hold_row, *prediction_rows]), model_root / "predictions.parquet"
    )
    manifest = {
        "dataset": {"path": dataset, "split": "validation", "language": "ko"},
        "language": "ko",
    }
    (model_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if include_span_set:
        parquet.write_table(
            arrow.Table.from_pylist(
                [
                    {
                        "id": "hold",
                        "language": "ko",
                        "span_index": 0,
                        "label": "hold",
                        "duration": 0.1,
                    },
                    {
                        "id": "turn",
                        "language": "ko",
                        "span_index": 0,
                        "label": "eot",
                        "duration": duration_s,
                    },
                ],
            ),
            language_root / "span_set.parquet",
        )
    return root


def prediction_row(silence_dur: float, score: float = 0.1, **extra: object) -> dict:
    return {
        "id": "turn",
        "language": "ko",
        "span_index": 0,
        "timestamp": silence_dur,
        "silence_dur": silence_dur,
        "p_eot": score,
        "label": "eot",
        **extra,
    }


def test_missing_500ms_score_is_allowed_only_when_pause_ended_before_500ms(tmp_path: Path) -> None:
    root = tmp_path / "short"
    write_artifact(
        root, duration_s=0.3, prediction_rows=[prediction_row(0.0), prediction_row(0.2, 0.8)]
    )
    artifact = load_artifact(root, "ko", PROBES)
    eot = next(span for span in artifact.spans if span.span_id == "turn")
    assert eot.scores == {200: 0.8}

    long_root = tmp_path / "long"
    write_artifact(
        long_root, duration_s=0.6, prediction_rows=[prediction_row(0.0), prediction_row(0.2, 0.8)]
    )
    with pytest.raises(ValueError, match="missing required probe scores"):
        load_artifact(long_root, "ko", PROBES)


def test_invalid_dataset_duplicate_rows_and_missing_duration_fail(tmp_path: Path) -> None:
    invalid_dataset = tmp_path / "invalid-dataset"
    write_artifact(
        invalid_dataset, duration_s=0.3, prediction_rows=[prediction_row(0.0)], dataset="other/data"
    )
    with pytest.raises(ValueError, match="dataset validation failed"):
        load_artifact(invalid_dataset, "ko", PROBES)

    duplicate = tmp_path / "duplicate"
    write_artifact(
        duplicate,
        duration_s=0.3,
        prediction_rows=[prediction_row(0.0), prediction_row(0.2), prediction_row(0.2)],
    )
    with pytest.raises(ValueError, match="duplicate score row"):
        load_artifact(duplicate, "ko", PROBES)

    isolated = tmp_path / "isolated"
    write_artifact(
        isolated, duration_s=0.3, prediction_rows=[prediction_row(0.0)], include_span_set=False
    )
    with pytest.raises(ValueError, match="span duration is unavailable"):
        load_artifact(isolated, "ko", PROBES)
