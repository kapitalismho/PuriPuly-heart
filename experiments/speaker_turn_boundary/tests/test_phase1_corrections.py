from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.config import (
    CANONICAL_SAMPLE_RATE_HZ,
    EXPERIMENT_DATA_DIR,
)
from experiments.speaker_turn_boundary.events import SpeakerBoundaryEvent
from experiments.speaker_turn_boundary.ground_truth import (
    SpeakerRegion,
    active_speech_sample_count,
    rebase_regions_to_epoch,
)
from experiments.speaker_turn_boundary.metrics import (
    aggregate_cases,
    incremental_over_b0,
)
from experiments.speaker_turn_boundary.schemas import DatasetManifest
from experiments.speaker_turn_boundary.sweep_common import (
    SWEEP_SCHEMA_VERSION,
    b0_aggregate_for_manifest,
    build_sweep_result,
    new_summary,
    profile_summary_dict,
    run_b0_and_detector,
)


def _silero(silero_model_path: Path):
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    return lambda: SileroVadOnnx(silero_model_path)


def _phase1_manifest() -> DatasetManifest:
    return DatasetManifest.load(EXPERIMENT_DATA_DIR / "manifests" / "phase1_dev.json")


class _ExactChangeDetector:
    def __init__(self, change_samples: dict[int, int]) -> None:
        self._change_samples = change_samples

    def start_epoch(self, audio_epoch: int) -> None:
        self.epoch = audio_epoch

    def run_case(self, samples: np.ndarray) -> tuple[list[SpeakerBoundaryEvent], list]:
        change_sample = self._change_samples.get(self.epoch)
        if change_sample is None:
            return [], []
        return (
            [
                SpeakerBoundaryEvent(
                    audio_epoch=self.epoch,
                    boundary_source_sample=change_sample,
                    observed_source_sample_at_emit=change_sample,
                    emitted_monotonic_ns=0,
                    confidence=1.0,
                    source="exact_test",
                    debug={},
                )
            ],
            [],
        )


def _write_smoke_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = np.round(pcm * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(CANONICAL_SAMPLE_RATE_HZ)
        wav_file.writeframes(pcm16)


def test_smoke_wavs_excluded_from_benchmark_aggregates(
    silero_model_path: Path, tmp_dir: Path
) -> None:
    smoke_one = tmp_dir / "smoke_one.wav"
    smoke_two = tmp_dir / "smoke_two.wav"
    _write_smoke_wav(smoke_one, np.zeros(16000, dtype=np.float32))
    _write_smoke_wav(smoke_two, np.zeros(16000, dtype=np.float32))
    manifest = _phase1_manifest()
    records, smoke_records = run_b0_and_detector(
        manifest,
        EXPERIMENT_DATA_DIR,
        engine_factory=_silero(silero_model_path),
        detector_run=_ExactChangeDetector({2: 100, 3: 100}),
        case_ids=None,
        coalesce_window_samples=8000,
        smoke_wavs=[smoke_one, smoke_two],
    )
    assert [record.case_id for record in records] == [
        "zero_gap_handoff_ab",
        "overlap_300ms_ab",
    ]
    assert [record.case_id for record in smoke_records] == [
        "smoke:smoke_one",
        "smoke:smoke_two",
    ]
    assert [record.audio_epoch for record in smoke_records] == [2, 3]
    aggregate = aggregate_cases(
        [record.metrics for record in records],
        profile_id="test",
        coalescing_reports=[record.coalescing.report for record in records],
        smoke_metrics=[record.metrics for record in smoke_records],
    )
    assert aggregate.case_count == 2
    assert aggregate.smoke_case_count == 2
    assert aggregate.gt_change_count == 2
    assert aggregate.product_false_cuts_total == 0
    assert aggregate.detector_only_false_cuts_total == 0
    assert aggregate.smoke_detector_cut_count_total == 2
    assert aggregate.recall_at_ms[500] == 0.5
    assert aggregate.active_speech_samples == 116800
    smoke_aggregate = aggregate_cases(
        [record.metrics for record in smoke_records], profile_id="smoke"
    )
    assert smoke_aggregate.gt_change_count == 0


def test_active_speech_denominator_union_no_double_count() -> None:
    overlapping = [
        SpeakerRegion(
            audio_epoch=0,
            start_sample=0,
            end_sample=16000,
            speakers=frozenset({"A"}),
        ),
        SpeakerRegion(
            audio_epoch=0,
            start_sample=8000,
            end_sample=24000,
            speakers=frozenset({"B"}),
        ),
        SpeakerRegion(
            audio_epoch=0,
            start_sample=24000,
            end_sample=40000,
            speakers=frozenset(),
        ),
        SpeakerRegion(
            audio_epoch=0,
            start_sample=40000,
            end_sample=48000,
            speakers=frozenset({"A"}),
            ambiguous=True,
        ),
        SpeakerRegion(
            audio_epoch=0,
            start_sample=48000,
            end_sample=56000,
            speakers=frozenset({"B"}),
        ),
    ]
    assert active_speech_sample_count(overlapping) == 24000 + 8000


def test_phase1_active_speech_denominator_matches_regions() -> None:
    manifest = _phase1_manifest()
    totals = []
    for case in manifest.cases:
        region_count = active_speech_sample_count(case.regions)
        assert region_count > 0
        totals.append(region_count)
    assert sum(totals) == 116800


def test_b0_incremental_metrics(silero_model_path: Path) -> None:
    manifest = _phase1_manifest()
    b0 = b0_aggregate_for_manifest(
        manifest,
        EXPERIMENT_DATA_DIR,
        engine_factory=_silero(silero_model_path),
        case_ids=None,
        coalesce_window_samples=8000,
    )
    assert b0.gt_change_count == 2
    assert b0.recall_at_ms[500] == 0.5
    assert b0.product_false_cuts_total == 0
    assert b0.active_speech_samples == 116800
    records, _ = run_b0_and_detector(
        manifest,
        EXPERIMENT_DATA_DIR,
        engine_factory=_silero(silero_model_path),
        detector_run=_ExactChangeDetector({1: 40000}),
        case_ids=None,
        coalesce_window_samples=8000,
    )
    candidate = aggregate_cases(
        [record.metrics for record in records],
        profile_id="cand",
        coalescing_reports=[record.coalescing.report for record in records],
    )
    assert candidate.recall_at_ms[500] == 1.0
    assert candidate.product_false_cuts_total == 0
    delta = incremental_over_b0(b0, candidate)
    assert delta["incremental_recall_at_500ms"] == 0.5
    assert delta["incremental_false_cuts"] == 0
    assert delta["b0_product_false_cuts_total"] == 0
    assert delta["candidate_product_false_cuts_total"] == 0


def test_unmatched_candidate_adds_incremental_false_cuts(silero_model_path: Path) -> None:
    manifest = _phase1_manifest()
    b0 = b0_aggregate_for_manifest(
        manifest,
        EXPERIMENT_DATA_DIR,
        engine_factory=_silero(silero_model_path),
        case_ids=None,
        coalesce_window_samples=8000,
    )
    records, _ = run_b0_and_detector(
        manifest,
        EXPERIMENT_DATA_DIR,
        engine_factory=_silero(silero_model_path),
        detector_run=_ExactChangeDetector({1: 64000}),
        case_ids=None,
        coalesce_window_samples=8000,
    )
    candidate = aggregate_cases(
        [record.metrics for record in records],
        profile_id="cand",
        coalescing_reports=[record.coalescing.report for record in records],
    )
    assert candidate.product_false_cuts_total == 1
    delta = incremental_over_b0(b0, candidate)
    assert delta["incremental_false_cuts"] == 1


def test_summary_and_result_structure(silero_model_path: Path, tmp_dir: Path) -> None:
    manifest = _phase1_manifest()
    smoke_path = tmp_dir / "smoke_struct.wav"
    _write_smoke_wav(smoke_path, np.zeros(16000, dtype=np.float32))
    records, smoke_records = run_b0_and_detector(
        manifest,
        EXPERIMENT_DATA_DIR,
        engine_factory=_silero(silero_model_path),
        detector_run=_ExactChangeDetector({0: 40000, 1: 40000, 2: 100}),
        case_ids=None,
        coalesce_window_samples=8000,
        smoke_wavs=[smoke_path],
    )
    aggregate = aggregate_cases(
        [record.metrics for record in records],
        profile_id="p",
        coalescing_reports=[record.coalescing.report for record in records],
        smoke_metrics=[record.metrics for record in smoke_records],
    )
    b0 = b0_aggregate_for_manifest(
        manifest,
        EXPERIMENT_DATA_DIR,
        engine_factory=_silero(silero_model_path),
        case_ids=None,
        coalesce_window_samples=8000,
    )
    incremental = incremental_over_b0(b0, aggregate)
    result_hash = build_sweep_result(
        manifest=manifest,
        detector={"family": "fake", "profile_id": "p"},
        runtime_metadata={},
        epochs=records,
        aggregate=aggregate,
        b0_aggregate=b0,
        incremental=incremental,
        smoke_epochs=smoke_records,
        out_dir=tmp_dir,
        out_name="result.json",
    )
    payload = json.loads((tmp_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == SWEEP_SCHEMA_VERSION
    assert payload["result_sha256"] == result_hash
    assert payload["aggregate"]["case_count"] == 2
    assert payload["aggregate"]["smoke_case_count"] == 1
    assert payload["b0_aggregate"]["case_count"] == 2
    assert "incremental_over_b0" in payload
    assert len(payload["smoke_epochs"]) == 1
    assert payload["smoke_epochs"][0]["case_id"] == "smoke:smoke_struct"
    summary = new_summary(manifest_id="phase1_dev", detector_family="fake")
    summary["variants"]["V"] = {"p": profile_summary_dict(aggregate, b0, incremental)}
    assert summary["schema_version"]
    assert summary["manifest_id"] == "phase1_dev"
    assert summary["detector_family"] == "fake"
    entry = summary["variants"]["V"]["p"]
    assert entry["case_count"] == 2
    assert entry["smoke_case_count"] == 1
    assert "b0" in entry
    assert "incremental_over_b0" in entry
    assert (
        entry["incremental_over_b0"]["incremental_false_cuts"]
        == incremental["incremental_false_cuts"]
    )


def test_rebase_regions_to_epoch() -> None:
    regions = [
        SpeakerRegion(
            audio_epoch=0,
            start_sample=0,
            end_sample=100,
            speakers=frozenset({"A"}),
        )
    ]
    rebased = rebase_regions_to_epoch(regions, 3)
    assert rebased[0].audio_epoch == 3
    assert rebased[0].start_sample == 0
    assert rebased[0].end_sample == 100
    assert regions[0].audio_epoch == 0


def test_eres_confidence_clamped_at_adapter_boundary() -> None:
    import experiments.speaker_turn_boundary.adapters.eres2netv2 as adapter
    from experiments.speaker_turn_boundary.adapters.eres2netv2 import (
        AdjacentWindowDetector,
        EresAdjacentProfile,
        EresStableAnchorProfile,
        StableAnchorDetector,
        clamp_confidence,
    )

    assert clamp_confidence(1.2) == 0.0
    assert clamp_confidence(-0.3) == 1.0
    assert clamp_confidence(0.4) == 0.6

    class _ZeroRuntime:
        def embed(self, samples: np.ndarray) -> np.ndarray:
            return np.zeros(4, dtype=np.float32)

    original = adapter.cosine_similarity
    adapter.cosine_similarity = lambda left, right: -1e-7
    try:
        samples = np.zeros(16000 * 3, dtype=np.float32)
        adjacent = AdjacentWindowDetector(
            _ZeroRuntime(),
            EresAdjacentProfile(
                window_seconds=0.5,
                step_seconds=0.25,
                threshold=1.0,
                confirmation=1,
            ),
        )
        boundaries = adjacent.run_utterance(samples, (0, 48000))
        assert boundaries
        assert all(boundary.confidence == 1.0 for boundary in boundaries)
        anchor = StableAnchorDetector(
            _ZeroRuntime(),
            EresStableAnchorProfile(
                window_seconds=0.5,
                step_seconds=0.25,
                threshold=1.0,
                confirmation=1,
                mutual_similarity_threshold=0.5,
                anchor_update="none",
            ),
        )
        anchor_boundaries = anchor.run_utterance(samples, (0, 48000))
        assert anchor_boundaries
        assert all(boundary.confidence == 1.0 for boundary in anchor_boundaries)
    finally:
        adapter.cosine_similarity = original


def test_eres_runner_clamps_confidence_at_event_boundary() -> None:
    from experiments.speaker_turn_boundary.adapters.eres2netv2 import EresBoundary
    from experiments.speaker_turn_boundary.run_eres_sweep import EresDetectorRunner

    class _Runtime:
        def embed(self, samples: np.ndarray) -> np.ndarray:
            return np.zeros(192, dtype=np.float32)

    runner = EresDetectorRunner(_Runtime(), "E-standard")
    runner.start_epoch(0)
    events = runner._to_events(
        [
            EresBoundary(
                audio_epoch=0,
                boundary_sample=100,
                observed_sample=200,
                confidence=1.5,
                debug={"profile": {"profile_id": "adjacent-W0p50-s0p25-thr0p50-c1"}},
            )
        ]
    )
    assert events[0].confidence == 0.0


def test_eres_sweep_epoch_identity_second_case(silero_model_path: Path) -> None:
    from experiments.speaker_turn_boundary.run_eres_sweep import (
        _run_epochs,
        build_epoch_inputs,
    )

    manifest = _phase1_manifest()
    annotated, smoke = build_epoch_inputs(
        manifest,
        EXPERIMENT_DATA_DIR,
        _silero(silero_model_path),
        None,
        None,
    )
    assert smoke == []

    class _ExactRunner:
        def __init__(self) -> None:
            self._audio_epoch = -1

        def start_epoch(self, audio_epoch: int) -> None:
            self._audio_epoch = audio_epoch

        def run_case(self, samples, utterances, builder):
            return (
                [
                    SpeakerBoundaryEvent(
                        audio_epoch=self._audio_epoch,
                        boundary_source_sample=40000,
                        observed_source_sample_at_emit=40000,
                        emitted_monotonic_ns=0,
                        confidence=1.0,
                        source="eres_test",
                        debug={},
                    )
                ],
                [],
            )

    def builder(samples, utterance):
        return [], []

    records = _run_epochs(annotated, _ExactRunner(), builder, 8000)
    assert [record.audio_epoch for record in records] == [0, 1]
    assert [change.audio_epoch for record in records for change in record.gt_changes] == [
        0,
        1,
    ]
    assert records[1].case_id == "overlap_300ms_ab"
    assert records[1].metrics.recall_at_ms[500] == 1.0
    assert records[1].metrics.product_false_cuts == 0
