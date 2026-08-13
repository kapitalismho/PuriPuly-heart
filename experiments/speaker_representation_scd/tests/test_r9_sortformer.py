from __future__ import annotations

import math
import struct

import numpy as np

from experiments.speaker_representation_scd.r9_sortformer_change_verification_upper_bound import (
    EMBEDDING_DIM,
    EMBEDDING_SPEAKERS,
    _b1_fold_transform,
    _b2_detect,
    _balance_weights,
    _curve_row,
    _dense_embedding_frames,
    _embedding_features_for_candidate,
    _features_for_candidate,
    _fit_linear,
    _grouped_events,
    _label_candidate,
    _linear_predict,
    _one_to_one,
    _parse_dump_records,
    _segment_start_candidates,
    _select_row,
    _standardize,
    _standardize_fit,
    _validate_dump_records,
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


def _row(frame_idx: int, slot: int, emb_seed: float) -> tuple[int, list[float], list[float]]:
    emb = np.full(EMBEDDING_DIM, float(emb_seed), dtype=np.float32)
    preds = np.zeros(EMBEDDING_SPEAKERS, dtype=np.float32)
    preds[slot] = 0.9
    return frame_idx, [float(value) for value in emb], [float(value) for value in preds]


def _write_dump(tmp_path, records: list[dict]) -> object:
    blob = bytearray()
    for record in records:
        cache_rows = record.get("cache", [])
        fifo_rows = record.get("fifo", [])
        blob += struct.pack(
            "<qqiiii",
            int(record["chunk"]),
            int(record["total_n"]),
            len(cache_rows),
            len(fifo_rows),
            int(record.get("compression", 0)),
            int(record.get("compress_count", 0)),
        )
        for frame_idx, emb, preds in cache_rows + fifo_rows:
            blob += struct.pack("<i", int(frame_idx))
            blob += np.asarray(emb, dtype="<f4").tobytes()
            blob += np.asarray(preds, dtype="<f4").tobytes()
    path = tmp_path / "dump.bin"
    path.write_bytes(bytes(blob))
    return path


def test_parse_dump_records_round_trip(tmp_path) -> None:
    rows_a = [_row(4, 0, 1.0), _row(5, 0, 2.0)]
    rows_b = [_row(6, 1, 3.0)]
    path = _write_dump(
        tmp_path,
        [
            {"chunk": 0, "total_n": 6, "fifo": rows_a, "compression": 0, "compress_count": 0},
            {"chunk": 2, "total_n": 12, "cache": rows_b, "compression": 1, "compress_count": 1},
        ],
    )
    records = _parse_dump_records(path)
    assert len(records) == 2
    assert [int(value) for value in records[0]["frame_idx"]] == [4, 5]
    assert records[0]["emb"].shape == (2, EMBEDDING_DIM)
    assert records[0]["preds"].shape == (2, EMBEDDING_SPEAKERS)
    assert int(records[1]["compression"]) == 1
    assert int(records[1]["compress_count"]) == 1


def test_validate_dump_records_catches_violations(tmp_path) -> None:
    path = _write_dump(
        tmp_path,
        [
            {"chunk": 0, "total_n": 6, "fifo": [_row(0, 0, 0.0)], "compression": 0},
            {"chunk": 3, "total_n": 12, "fifo": [_row(11, 0, 0.0)], "compression": 0},
        ],
    )
    records = _parse_dump_records(path)
    result = _validate_dump_records(records, cadence=2, horizon=188)
    assert not result["valid"]
    assert any("chunk" in violation for violation in result["violations"])
    valid_path = _write_dump(
        tmp_path,
        [{"chunk": 0, "total_n": 6, "fifo": [_row(0, 0, 0.0)], "compression": 0}],
    )
    valid = _validate_dump_records(_parse_dump_records(valid_path), cadence=2, horizon=188)
    assert valid["valid"]
    horizon_path = _write_dump(
        tmp_path,
        [{"chunk": 0, "total_n": 100, "fifo": [_row(0, 0, 0.0)], "compression": 0}],
    )
    horizon_result = _validate_dump_records(
        _parse_dump_records(horizon_path), cadence=2, horizon=10
    )
    assert not horizon_result["valid"]


def test_embedding_features_for_candidate_values() -> None:
    record = {
        "chunk": 0,
        "total_n": 40,
        "frame_idx": np.asarray([18, 19, 20, 21, 22], dtype=np.int32),
        "emb": np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "preds": np.asarray(
            [
                [0.9, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1],
            ],
            dtype=np.float32,
        ),
        "compression": 0,
        "compress_count": 0,
        "n_cache": 5,
        "n_fifo": 0,
    }
    features = config()["embedding"]["features"]
    values, excluded = _embedding_features_for_candidate(record, 20, 1, features)
    assert excluded is False
    assert math.isclose(values["same_slot_similarity"], 1.0, abs_tol=1e-6)
    assert math.isclose(values["embedding_jump"], 0.0, abs_tol=1e-6)
    assert math.isclose(values["best_other_similarity"], 1.0, abs_tol=1e-6)
    empty_record = {
        **record,
        "frame_idx": np.asarray([40, 41], dtype=np.int32),
        "emb": np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
        "preds": np.asarray([[0.9, 0.1, 0.1, 0.1]] * 2, dtype=np.float32),
        "n_cache": 2,
        "n_fifo": 0,
    }
    empty_values, excluded = _embedding_features_for_candidate(empty_record, 20, 1, features)
    assert excluded is True
    assert math.isnan(empty_values["same_slot_similarity"])
    assert math.isnan(empty_values["best_other_similarity"])
    assert math.isnan(empty_values["embedding_jump"])


def test_b1_fold_transform_imputes_nan_with_train_median() -> None:
    cfg = config()
    embedding_names = list(cfg["embedding"]["features"]["feature_names"])
    assert embedding_names.index("compression_boundary") == 3
    base_count = len(list(cfg["feature_names"]))
    matrix = np.zeros((4, base_count + 4), dtype=np.float32)
    matrix[:, base_count] = [1.0, 2.0, np.nan, 3.0]
    matrix[:, base_count + 1] = [10.0, np.nan, 20.0, 30.0]
    train_mask = np.asarray([True, True, False, False])
    transformed = _b1_fold_transform(matrix, train_mask)
    assert math.isclose(transformed[2, base_count], 1.5)
    assert math.isclose(transformed[1, base_count + 1], 10.0)
    assert math.isclose(transformed[0, base_count], 1.0)


def test_dense_embedding_frames_fills_from_records(tmp_path) -> None:
    rows_a = [_row(0, 0, 5.0), _row(1, 0, 6.0)]
    rows_b = [_row(2, 0, 7.0)]
    path = _write_dump(
        tmp_path,
        [
            {"chunk": 0, "total_n": 2, "fifo": rows_a, "compression": 0},
            {"chunk": 2, "total_n": 4, "fifo": rows_b, "compression": 0},
        ],
    )
    dense = _dense_embedding_frames(path, 3)
    assert dense.shape == (3, EMBEDDING_DIM)
    assert math.isclose(float(dense[0, 0]), 5.0)
    assert math.isclose(float(dense[2, 0]), 7.0)


def test_b2_detect_flags_intra_slot_drop_and_deduplicates() -> None:
    frames = 61
    probabilities = np.zeros((frames, 4), dtype=np.float32)
    probabilities[:, 0] = 0.9
    dense = np.zeros((frames, EMBEDDING_DIM), dtype=np.float32)
    dense[:30, 0] = 1.0
    dense[30:, 1] = 1.0
    detected = _b2_detect(
        probabilities,
        dense,
        {"window_frames": 15, "stride_frames": 3, "similarity_threshold": 0.75},
    )
    assert detected[0][0] == 24
    assert all(slot == 0 for _, slot in detected)
    assert all(later[0] - earlier[0] >= 2 for earlier, later in zip(detected, detected[1:]))
    dense_stride = _b2_detect(
        probabilities,
        dense,
        {"window_frames": 15, "stride_frames": 1, "similarity_threshold": 0.75},
    )
    assert len(dense_stride) == 8
    assert [frame for frame, _ in dense_stride] == [23, 25, 27, 29, 31, 33, 35, 37]
