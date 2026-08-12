from __future__ import annotations

import math

import numpy as np

from experiments.speaker_representation_scd.r9_sortformer_change_verification_upper_bound import (
    _balance_weights,
    _curve_row,
    _features_for_candidate,
    _fit_linear,
    _grouped_events,
    _label_candidate,
    _linear_predict,
    _one_to_one,
    _segment_start_candidates,
    _select_row,
    _standardize,
    _standardize_fit,
    config,
    smoke,
)


def test_smoke_action_is_deterministic_and_consistent() -> None:
    first = smoke()
    second = smoke()
    assert first == second
    assert first["segment_start_frames"] == [0, 2, 6]
    assert first["grouped_events"] == [(150, 0.8), (4000, 0.9)]


def test_segment_starts_mirror_fixed_threshold_runs() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    starts = _segment_start_candidates(probabilities, 80)
    assert [(row["speaker_slot"], row["start_frame"]) for row in starts] == [
        (1, 0),
        (2, 2),
        (1, 6),
    ]


def test_label_windows() -> None:
    events = [{"sample": 10_000, "stratum": "clean_change"}]
    positive = _label_candidate(10_000 + 200 * 16, events)
    assert positive["label"] == "positive"
    ambiguous = _label_candidate(10_000 + 300 * 16, events)
    assert ambiguous["label"] == "ambiguous"
    negative = _label_candidate(10_000 + 600 * 16, events)
    assert negative["label"] == "negative"
    assert _label_candidate(5, [])["label"] == "negative"


def test_features_for_overlap_onset() -> None:
    cfg = config()
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    features = _features_for_candidate(
        probabilities, 2, 1, int(cfg["confirmation"]["base_frames"]), cfg["feature_windows"], 80
    )
    assert features["gap_ms"] == 0.0
    assert features["same_slot_resume"] == 0.0
    assert features["persistence_ms"] == 240.0
    assert features["co_activity_max"] > 0.5
    assert features["argmax_switch"] == 1.0
    assert math.isclose(features["peak_probability"], 0.9, rel_tol=1e-6)
    assert math.isclose(features["cross_probability"], 0.9, rel_tol=1e-6)


def test_features_for_pause_resume() -> None:
    cfg = config()
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    features = _features_for_candidate(
        probabilities, 4, 0, int(cfg["confirmation"]["base_frames"]), cfg["feature_windows"], 80
    )
    assert features["gap_ms"] == 160.0
    assert features["same_slot_resume"] == 1.0
    assert features["argmax_switch"] == 0.0
    assert math.isclose(features["pre_depth_mean"], 0.9, rel_tol=1e-6)


def test_recording_start_gap_is_imputed_to_cap() -> None:
    cfg = config()
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    features = _features_for_candidate(
        probabilities, 0, 0, int(cfg["confirmation"]["base_frames"]), cfg["feature_windows"], 80
    )
    assert features["gap_ms"] == float(cfg["feature_windows"]["gap_cap_frames"] * 80)
    assert features["pre_depth_min"] == 0.5
    assert features["same_slot_resume"] == 0.0


def test_co_onset_after_silence_has_zero_gap() -> None:
    cfg = config()
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    features = _features_for_candidate(
        probabilities, 2, 1, int(cfg["confirmation"]["base_frames"]), cfg["feature_windows"], 80
    )
    assert features["gap_ms"] == 0.0


def test_confirmation_window_never_reads_beyond_three_frames() -> None:
    cfg = config()
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.8, 0.1, 0.1],
            [0.1, 0.99, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    features = _features_for_candidate(
        probabilities, 2, 1, int(cfg["confirmation"]["base_frames"]), cfg["feature_windows"], 80
    )
    assert math.isclose(features["peak_probability"], 0.8, rel_tol=1e-6)
    assert math.isclose(features["rise_slope"], 0.2, rel_tol=1e-6)
    assert features["persistence_ms"] == 240.0


def test_grouping_keeps_highest_score_within_radius() -> None:
    rows = [
        {"sample": 100, "score": 0.4},
        {"sample": 150, "score": 0.8},
        {"sample": 4000, "score": 0.9},
    ]
    assert _grouped_events(rows, 3200) == [(150, 0.8), (4000, 0.9)]


def test_matcher_is_one_to_one() -> None:
    matched, false, misses = _one_to_one([100, 120, 900], [110, 910], 30)
    assert matched == [(100, 110), (900, 910)]
    assert false == [120]
    assert misses == []


def test_balance_weights_normalize_to_length() -> None:
    labels = np.asarray([1, 1, 1, 0, 0], dtype=np.int8)
    weights = _balance_weights(labels)
    assert math.isclose(float(weights.sum()), 5.0)
    assert math.isclose(float(weights[labels == 1].sum()), 2.5)
    assert math.isclose(float(weights[labels == 0].sum()), 2.5)


def test_linear_verifier_is_deterministic() -> None:
    cfg = config()
    matrix = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int8)
    weights = np.ones(len(y), dtype=np.float32)
    coef_a, intercept_a = _fit_linear(matrix, y, weights, cfg["verifier"])
    coef_b, intercept_b = _fit_linear(matrix, y, weights, cfg["verifier"])
    assert np.array_equal(coef_a, coef_b)
    assert intercept_a == intercept_b
    scores = _linear_predict(matrix, coef_a, intercept_a)
    assert np.isfinite(scores).all()
    assert bool((scores[y == 1] > scores[y == 0]).mean() > 0.5)


def test_standardization_matches_scale_guard() -> None:
    values = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    mean, scale = _standardize_fit(values)
    assert scale[1] == 1.0
    standardized = _standardize(values, mean, scale)
    assert abs(float(standardized.mean(axis=0)[0])) < 1e-6
    assert float(standardized[0, 0]) < float(standardized[2, 0])


def test_select_row_prefers_recall_within_budget() -> None:
    rows = [
        {"threshold": 0.9, "false_events_per_hour": 5.0, "recall_250": 0.3},
        {"threshold": 0.8, "false_events_per_hour": 12.0, "recall_250": 0.6},
        {"threshold": 0.7, "false_events_per_hour": 9.5, "recall_250": 0.55},
    ]
    selected = _select_row(rows, 10.0)
    assert selected["threshold"] == 0.7
    selected = _select_row(rows, 1.0)
    assert selected["threshold"] == 0.9


def test_curve_row_uses_score_cache() -> None:
    from pathlib import Path

    from experiments.speaker_representation_scd.r9_sortformer_change_verification_upper_bound import (
        ScoreEvaluationCache,
        Session,
    )

    session = Session(
        session_id="s1",
        fold=0,
        waveform_path=Path("unused.wav"),
        waveform_sha256="unused",
        first_boundary=0,
        last_boundary=100_000,
        scored_hours=1.0,
        events=[{"sample": 10_000, "stratum": "clean_change"}],
    )
    cache = ScoreEvaluationCache({"s1": session}, {"s1": [(9_600, 0.9), (50_000, 0.4)]})
    row = _curve_row(cache, 0.3)
    assert row["prediction_count"] == 2
    assert row["true_positive_count"] == 1
    assert row["false_event_count"] == 1
    assert row["false_events_per_hour"] == 1.0
    assert row["recall_250"] == 1.0
    row_high = _curve_row(cache, 0.85)
    assert row_high["prediction_count"] == 1
    assert row_high["recall_250"] == 1.0
