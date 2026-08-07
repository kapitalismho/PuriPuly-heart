from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.config import EXPERIMENT_DATA_DIR
from experiments.speaker_turn_boundary.events import SpeakerBoundaryEvent
from experiments.speaker_turn_boundary.metrics import aggregate_cases
from experiments.speaker_turn_boundary.schemas import DatasetManifest
from experiments.speaker_turn_boundary.sweep_common import run_b0_and_detector


class _NoopDetector:
    def __init__(self) -> None:
        self.epoch = -1

    def start_epoch(self, audio_epoch: int) -> None:
        self.epoch = audio_epoch

    def run_case(self, samples: np.ndarray) -> tuple[list, list]:
        return [], []


class _ExactChangeDetector:
    def __init__(self, change_samples: dict[int, int]) -> None:
        self._change_samples = change_samples
        self.epoch = -1

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


def _silero(silero_model_path: Path):
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    return lambda: SileroVadOnnx(silero_model_path)


def test_multi_case_epoch_gt_attribution(silero_model_path: Path) -> None:
    manifest = DatasetManifest.load(EXPERIMENT_DATA_DIR / "manifests" / "phase1_dev.json")
    records, smoke_records = run_b0_and_detector(
        manifest,
        EXPERIMENT_DATA_DIR,
        engine_factory=_silero(silero_model_path),
        detector_run=_NoopDetector(),
        case_ids=None,
        coalesce_window_samples=8000,
    )
    assert smoke_records == []
    assert [record.case_id for record in records] == [
        "zero_gap_handoff_ab",
        "overlap_300ms_ab",
    ]
    assert [record.audio_epoch for record in records] == [0, 1]
    assert [len(record.gt_changes) for record in records] == [1, 1]
    assert [change.audio_epoch for record in records for change in record.gt_changes] == [
        0,
        1,
    ]
    aggregate = aggregate_cases(
        [record.metrics for record in records],
        profile_id="test",
        coalescing_reports=[record.coalescing.report for record in records],
    )
    assert aggregate.gt_change_count == 2


def test_exact_detector_cut_matches_second_case_gt(silero_model_path: Path) -> None:
    manifest = DatasetManifest.load(EXPERIMENT_DATA_DIR / "manifests" / "phase1_dev.json")
    detector = _ExactChangeDetector(
        {
            0: 40000,
            1: 40000,
        }
    )
    records, _ = run_b0_and_detector(
        manifest,
        EXPERIMENT_DATA_DIR,
        engine_factory=_silero(silero_model_path),
        detector_run=detector,
        case_ids=None,
        coalesce_window_samples=8000,
    )
    first, second = records
    assert second.case_id == "overlap_300ms_ab"
    assert second.audio_epoch == 1
    assert [change.change_sample for change in second.gt_changes] == [40000]
    assert second.metrics.gt_change_count == 1
    assert second.metrics.recall_matched_counts[500] == 1
    assert second.metrics.recall_at_ms[500] == 1.0
    assert second.metrics.product_false_cuts == 0
    assert first.metrics.recall_at_ms[500] == 1.0
