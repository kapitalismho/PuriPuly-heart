from __future__ import annotations

import json

import pytest

from experiments.speaker_turn_boundary.coalescing import LogicalCut
from experiments.speaker_turn_boundary.ground_truth import SpeakerChangeGT
from experiments.speaker_turn_boundary.metrics import (
    PRODUCT_FALSE_CUT_TOLERANCE_MS,
    CaseBoundaryMetrics,
    aggregate_cases,
    evaluate_case,
    incremental_over_b0,
    match_gt_to_cuts,
)


def _change(sample: int, kind: str = "clean_handoff", epoch: int = 0) -> SpeakerChangeGT:
    return SpeakerChangeGT(
        audio_epoch=epoch,
        change_sample=sample,
        kind=kind,
        prev_speakers=frozenset({"A"}),
        next_speakers=frozenset({"B"}),
    )


def _cut(sample: int, kind: str = "detector", epoch: int = 0) -> LogicalCut:
    return LogicalCut(audio_epoch=epoch, sample=sample, kind=kind, ref_event_index=0)


def _metrics(
    case_id: str = "case",
    epoch: int = 0,
    gt: list[SpeakerChangeGT] | None = None,
    cuts: list[LogicalCut] | None = None,
    detector_events: list[LogicalCut] | None = None,
    active_speech_samples: int = 16000 * 30,
    vad_cut_count: int | None = None,
    detector_cut_count: int | None = None,
) -> CaseBoundaryMetrics:
    gt = gt or []
    cuts = cuts or []
    if detector_events is None:
        detector_events = [cut for cut in cuts if cut.kind == "detector"]
    if vad_cut_count is None:
        vad_cut_count = sum(1 for cut in cuts if cut.kind == "vad")
    if detector_cut_count is None:
        detector_cut_count = sum(1 for cut in cuts if cut.kind == "detector")
    return evaluate_case(
        case_id=case_id,
        audio_epoch=epoch,
        gt_changes=gt,
        cuts=cuts,
        detector_events=detector_events,
        vad_cut_count=vad_cut_count,
        detector_cut_count=detector_cut_count,
        active_speech_samples=active_speech_samples,
    )


def test_gt_matching_one_to_one():
    changes = [_change(16000), _change(80000)]
    cuts = [_cut(16200), _cut(80100)]
    matched = match_gt_to_cuts(changes, cuts, window_samples=4000)
    assert len(matched) == 2


def test_gt_matching_nearest_winner():
    changes = [_change(16000)]
    cuts = [_cut(15800), _cut(17500)]
    matched = match_gt_to_cuts(changes, cuts, window_samples=2000)
    assert len(matched) == 1


def test_gt_matching_respects_window():
    changes = [_change(16000)]
    cuts = [_cut(16000 + 3000)]
    matched = match_gt_to_cuts(changes, cuts, window_samples=2000)
    assert matched == []


def test_gt_matching_one_prediction_one_gt():
    changes = [_change(16000), _change(16500)]
    cuts = [_cut(16200)]
    matched = match_gt_to_cuts(changes, cuts, window_samples=4000)
    assert len(matched) == 1


def test_gt_matching_maximum_cardinality_counterexample():
    changes = [_change(0), _change(50)]
    cuts = [_cut(-50), _cut(40)]
    matched = match_gt_to_cuts(changes, cuts, window_samples=50)
    assert len(matched) == 2


def test_gt_matching_deterministic_across_input_order():
    window = 50
    changes = [_change(0), _change(50)]
    cuts = [_cut(-50), _cut(40)]
    first = match_gt_to_cuts(list(reversed(changes)), list(reversed(cuts)), window_samples=window)
    second = match_gt_to_cuts(changes, cuts, window_samples=window)
    assert sorted(first) == sorted(second)
    assert len(first) == 2


def test_detector_only_metrics_use_raw_pre_coalescing_events():
    metrics = _metrics(
        gt=[_change(1000)],
        cuts=[_cut(1000, kind="vad")],
        detector_events=[_cut(1000, kind="detector")],
    )
    assert metrics.recall_at_ms[500] == 1.0
    assert metrics.detector_only_recall_at_ms[500] == 1.0
    assert metrics.product_false_cuts == 0
    assert metrics.detector_only_false_cuts == 0


def test_detector_only_metrics_independent_of_product_coalescing():
    metrics = _metrics(
        gt=[_change(1000)],
        cuts=[_cut(1000, kind="vad")],
        detector_events=[_cut(1000, kind="detector"), _cut(60000, kind="detector")],
    )
    assert metrics.detector_only_recall_at_ms[500] == 1.0
    assert metrics.detector_only_false_cuts == 1
    assert metrics.product_false_cuts == 0


def test_evaluate_case_metrics():
    metrics = _metrics(
        gt=[_change(16000)],
        cuts=[_cut(16000 + 200), _cut(40000)],
    )
    assert metrics.gt_change_count == 1
    assert metrics.recall_at_ms[500] == 1.0
    assert metrics.recall_at_ms[250] == 1.0
    assert metrics.product_false_cuts == 1
    assert metrics.detector_only_false_cuts == 1
    assert metrics.detector_cut_count == 2
    assert metrics.active_speech_samples == 16000 * 30


def test_matched_true_cut_not_false():
    metrics = _metrics(gt=[_change(16000)], cuts=[_cut(16000)])
    assert metrics.recall_at_ms[500] == 1.0
    assert metrics.product_false_cuts == 0
    assert metrics.detector_only_false_cuts == 0


def test_unmatched_cut_false():
    metrics = _metrics(gt=[_change(16000)], cuts=[_cut(40000)])
    assert metrics.recall_at_ms[500] == 0.0
    assert metrics.product_false_cuts == 1
    assert metrics.detector_only_false_cuts == 1


def test_false_cut_tolerance_is_pinned_500ms():
    assert PRODUCT_FALSE_CUT_TOLERANCE_MS == 500
    assert PRODUCT_FALSE_CUT_TOLERANCE_MS * 16 == 8000


def test_product_vs_detector_only_metrics():
    metrics = _metrics(
        gt=[_change(16000)],
        cuts=[_cut(16000, kind="vad"), _cut(40000, kind="detector")],
    )
    assert metrics.recall_at_ms[500] == 1.0
    assert metrics.detector_only_recall_at_ms[500] == 0.0
    assert metrics.product_false_cuts == 1
    assert metrics.detector_only_false_cuts == 1
    assert metrics.b0_cut_count == 1


def test_detector_recovers_change_vad_misses():
    metrics = _metrics(
        gt=[_change(16000)],
        cuts=[_cut(16000, kind="detector")],
    )
    assert metrics.recall_at_ms[500] == 1.0
    assert metrics.detector_only_recall_at_ms[500] == 1.0
    assert metrics.product_false_cuts == 0
    assert metrics.detector_only_false_cuts == 0


def test_aggregate_and_incremental():
    baseline_metrics = [
        _metrics(
            case_id=f"c{index}",
            epoch=index,
            gt=[_change(16000, epoch=index)],
            cuts=[_cut(16000 + 100, kind="vad", epoch=index)],
            active_speech_samples=16000 * 30,
        )
        for index in range(2)
    ]
    baseline = aggregate_cases(baseline_metrics, profile_id="b0")
    assert baseline.case_count == 2
    assert baseline.gt_change_count == 2
    assert baseline.recall_at_ms[500] == 1.0
    assert baseline.product_false_cuts_total == 0
    candidate = aggregate_cases(
        [
            _metrics(
                case_id=f"c{index}",
                epoch=index,
                gt=[_change(16000, epoch=index)],
                cuts=[
                    _cut(16000 + 100, kind="vad", epoch=index),
                    _cut(40000 + index, kind="detector", epoch=index),
                ],
                active_speech_samples=16000 * 30,
            )
            for index in range(2)
        ],
        profile_id="det",
    )
    assert candidate.product_false_cuts_total == 2
    delta = incremental_over_b0(baseline, candidate)
    assert delta["incremental_recall_at_500ms"] == 0.0
    assert delta["incremental_false_cuts"] == 2


def test_case_metrics_serialization():
    metrics = _metrics(gt=[_change(16000)], cuts=[_cut(16000)])
    data = metrics.to_dict()
    assert json.dumps(data, sort_keys=True) == json.dumps(data, sort_keys=True)
    assert "product_false_cuts" in data
    assert "detector_only_false_cuts" in data
    assert "active_speech_samples" in data


def test_sweep_result_canonical_json(tmp_dir):
    from experiments.speaker_turn_boundary.schemas import canonical_json

    payload = {"a": [1, 2], "b": {"x": "y"}}
    first = canonical_json(payload)
    second = canonical_json(dict(payload))
    assert first == second
    assert first == json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


def test_aggregate_requires_all_deadlines():
    metrics = [
        _metrics(
            case_id=f"c{index}",
            epoch=index,
            gt=[_change(16000, epoch=index)],
            cuts=[_cut(16000, kind="vad", epoch=index)],
            active_speech_samples=16000 * 3600,
        )
        for index in range(2)
    ]
    aggregate = aggregate_cases(metrics, profile_id="p")
    assert aggregate.gt_change_count == 2
    assert aggregate.recall_at_ms[500] == 1.0
    assert aggregate.product_false_cuts_total == 0
    assert aggregate.false_cuts_per_speech_hour == 0.0
    assert aggregate.active_speech_samples == 2 * 16000 * 3600


def test_aggregate_speech_hour_uses_active_speech_denominator():
    metrics = [
        _metrics(
            case_id="c",
            epoch=0,
            gt=[_change(16000)],
            cuts=[_cut(40000)],
            active_speech_samples=16000 * 1800,
        )
    ]
    aggregate = aggregate_cases(metrics, profile_id="p")
    assert aggregate.product_false_cuts_total == 1
    assert aggregate.false_cuts_per_speech_hour == pytest.approx(2.0)
