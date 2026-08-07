from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.speaker_turn_boundary.coalescing import (
    CoalesceConfig,
    CoalescingOutcome,
    coalesce_vad_and_detector,
)
from experiments.speaker_turn_boundary.config import (
    BASELINE_SHA,
)
from experiments.speaker_turn_boundary.events import (
    DetectorProgress,
    SpeakerBoundaryEvent,
)
from experiments.speaker_turn_boundary.ground_truth import (
    SpeakerChangeGT,
    active_speech_sample_count,
    classify_active_speaker_transitions,
    rebase_regions_to_epoch,
)
from experiments.speaker_turn_boundary.metrics import (
    CaseBoundaryMetrics,
    SweepAggregate,
    aggregate_cases,
    detector_events_to_cuts,
    evaluate_case,
)
from experiments.speaker_turn_boundary.schemas import (
    DatasetManifest,
    canonical_json,
    sha256_hex,
)
from experiments.speaker_turn_boundary.vad_baseline import (
    load_canonical_wav,
    replay_wav_epoch,
)

SWEEP_SCHEMA_VERSION = "experiments.speaker_turn_boundary.sweep.v2"
SWEEP_SUMMARY_SCHEMA_VERSION = "experiments.speaker_turn_boundary.sweep_summary.v1"
B0_AGGREGATE_PROFILE_ID = "b0_vad_only"


@dataclass(frozen=True, slots=True)
class EpochSweepRecord:
    audio_epoch: int
    case_id: str
    length_samples: int
    gt_changes: list[SpeakerChangeGT]
    vad_boundaries: list[SpeakerBoundaryEvent]
    detector_boundaries: list[SpeakerBoundaryEvent]
    progress: list[DetectorProgress]
    coalescing: CoalescingOutcome
    metrics: CaseBoundaryMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "audio_epoch": self.audio_epoch,
            "case_id": self.case_id,
            "length_samples": self.length_samples,
            "gt_changes": [
                {
                    "audio_epoch": change.audio_epoch,
                    "change_sample": change.change_sample,
                    "kind": change.kind,
                    "prev_speakers": sorted(change.prev_speakers),
                    "next_speakers": sorted(change.next_speakers),
                }
                for change in self.gt_changes
            ],
            "vad_boundaries": [boundary.to_dict() for boundary in self.vad_boundaries],
            "detector_boundaries": [boundary.to_dict() for boundary in self.detector_boundaries],
            "progress": [snapshot.to_dict() for snapshot in self.progress],
            "coalescing": self.coalescing.to_dict(),
            "metrics": self.metrics.to_dict(),
        }


class _NoopDetector:
    def start_epoch(self, audio_epoch: int) -> None:
        pass

    def run_case(self, samples: np.ndarray) -> tuple[list, list]:
        return [], []


def run_b0_and_detector(
    manifest: DatasetManifest,
    wav_root: Path,
    *,
    engine_factory: Any,
    detector_run: Any,
    case_ids: list[str] | None,
    coalesce_window_samples: int,
    smoke_wavs: list[Path] | None = None,
) -> tuple[list[EpochSweepRecord], list[EpochSweepRecord]]:
    selected = [case for case in manifest.cases if case_ids is None or case.case_id in case_ids]
    records: list[EpochSweepRecord] = []
    for epoch_index, case in enumerate(selected):
        records.append(
            _run_one_epoch(
                epoch_index,
                case.case_id,
                (wav_root / case.wav_relative_path).resolve(),
                case.regions,
                engine_factory,
                detector_run,
                coalesce_window_samples,
            )
        )
    smoke_records: list[EpochSweepRecord] = []
    for index, wav_path in enumerate(smoke_wavs or []):
        smoke_records.append(
            _run_one_epoch(
                len(records) + index,
                f"smoke:{wav_path.stem}",
                wav_path,
                [],
                engine_factory,
                detector_run,
                coalesce_window_samples,
            )
        )
    return records, smoke_records


def _run_one_epoch(
    epoch_index: int,
    case_id: str,
    wav_path: Path,
    regions: list[Any],
    engine_factory: Any,
    detector_run: Any,
    coalesce_window_samples: int,
) -> EpochSweepRecord:
    samples = load_canonical_wav(wav_path)
    vad_result = replay_wav_epoch(wav_path, audio_epoch=epoch_index, engine_factory=engine_factory)
    detector_run.start_epoch(epoch_index)
    detector_boundaries, progress = detector_run.run_case(samples)
    changes, _ = classify_active_speaker_transitions(
        rebase_regions_to_epoch(list(regions), epoch_index)
    )
    vad_boundaries = vad_result.boundaries
    outcome = coalesce_vad_and_detector(
        vad_boundaries,
        detector_boundaries,
        config=CoalesceConfig(window_samples=coalesce_window_samples),
    )
    vad_count = sum(1 for cut in outcome.cuts if cut.kind == "vad")
    detector_cut_count = sum(1 for cut in outcome.cuts if cut.kind == "detector")
    metrics = evaluate_case(
        case_id=case_id,
        audio_epoch=epoch_index,
        gt_changes=changes,
        cuts=outcome.cuts,
        detector_events=detector_events_to_cuts(detector_boundaries),
        vad_cut_count=vad_count,
        detector_cut_count=detector_cut_count,
        active_speech_samples=active_speech_sample_count(list(regions)),
    )
    return EpochSweepRecord(
        audio_epoch=epoch_index,
        case_id=case_id,
        length_samples=vad_result.length_samples,
        gt_changes=changes,
        vad_boundaries=vad_boundaries,
        detector_boundaries=detector_boundaries,
        progress=progress,
        coalescing=outcome,
        metrics=metrics,
    )


def b0_aggregate_for_manifest(
    manifest: DatasetManifest,
    wav_root: Path,
    *,
    engine_factory: Any,
    case_ids: list[str] | None,
    coalesce_window_samples: int,
) -> SweepAggregate:
    records, _ = run_b0_and_detector(
        manifest,
        wav_root,
        engine_factory=engine_factory,
        detector_run=_NoopDetector(),
        case_ids=case_ids,
        coalesce_window_samples=coalesce_window_samples,
    )
    return aggregate_cases(
        [record.metrics for record in records],
        profile_id=B0_AGGREGATE_PROFILE_ID,
        coalescing_reports=[record.coalescing.report for record in records],
    )


def profile_summary_dict(
    aggregate: SweepAggregate,
    b0_aggregate: SweepAggregate,
    incremental: dict[str, object],
) -> dict[str, object]:
    return {
        **aggregate.to_dict(),
        "b0": b0_aggregate.to_dict(),
        "incremental_over_b0": incremental,
    }


def new_summary(
    *,
    manifest_id: str,
    detector_family: str,
) -> dict[str, object]:
    return {
        "schema_version": SWEEP_SUMMARY_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "detector_family": detector_family,
        "variants": {},
    }


def build_sweep_result(
    *,
    manifest: DatasetManifest,
    detector: dict[str, object],
    runtime_metadata: dict[str, object],
    epochs: list[EpochSweepRecord],
    aggregate: SweepAggregate,
    b0_aggregate: SweepAggregate,
    incremental: dict[str, object],
    smoke_epochs: list[EpochSweepRecord] | None = None,
    extra: dict[str, object] | None = None,
    out_dir: Path,
    out_name: str,
) -> str:
    started_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "result_id": str(uuid.uuid4()),
        "schema_version": SWEEP_SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.hash,
        "detector": detector,
        "runtime_metadata": runtime_metadata,
        "started_at_utc": started_at,
        "epochs": [epoch.to_dict() for epoch in epochs],
        "aggregate": aggregate.to_dict(),
        "b0_aggregate": b0_aggregate.to_dict(),
        "incremental_over_b0": incremental,
    }
    if smoke_epochs:
        payload["smoke_epochs"] = [epoch.to_dict() for epoch in smoke_epochs]
    if extra:
        payload["extra"] = extra
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / out_name
    result_hash = sha256_hex(payload)
    payload["result_sha256"] = result_hash
    path.write_text(canonical_json(payload), encoding="utf-8")
    return result_hash


def load_manifest(args_manifest: Path) -> DatasetManifest:
    return DatasetManifest.load(args_manifest)


def default_coalesce_window_samples() -> int:
    from experiments.speaker_turn_boundary.config import VAD_COALESCE_WINDOW_SAMPLES

    return VAD_COALESCE_WINDOW_SAMPLES
