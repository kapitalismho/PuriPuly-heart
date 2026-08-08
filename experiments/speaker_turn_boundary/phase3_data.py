from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.speaker_turn_boundary.config import EXPERIMENT_DATA_DIR
from experiments.speaker_turn_boundary.corpus.external import corpus_root
from experiments.speaker_turn_boundary.corpus.phase2_schemas import (
    PHASE2_MANIFEST_SCHEMA,
    Phase2Case,
    Phase2Manifest,
)
from experiments.speaker_turn_boundary.events import SpeakerBoundaryEvent
from experiments.speaker_turn_boundary.ground_truth import (
    SpeakerChangeGT,
    SpeakerRegion,
    active_speech_sample_count,
    classify_active_speaker_transitions,
    rebase_regions_to_epoch,
)
from experiments.speaker_turn_boundary.vad_baseline import load_canonical_wav

PHASE3_SCHEMA = "experiments.speaker_turn_boundary.phase3.v2"

GAP_BUCKET_LARGE = "gap_ge_300ms"
GAP_BUCKET_SMALL = "gap_1_299ms"
GAP_BUCKET_ZERO = "zero_gap"
OVERLAP_BUCKET = "overlap_onset"
TURN_BUCKET_SHORT = "turn_le_0p75s"
TURN_BUCKET_MID = "turn_0p75_1p5s"
TURN_BUCKET_LONG = "turn_gt_1p5s"


class Phase3DataError(RuntimeError):
    pass


def phase3_wav_roots(
    data_dir: Path | None = None,
    corpus_root_override: Path | None = None,
) -> list[Path]:
    data = Path(data_dir) if data_dir is not None else EXPERIMENT_DATA_DIR
    root = Path(corpus_root_override) if corpus_root_override is not None else corpus_root()
    roots = [data.resolve(), (root / "phase2_build").resolve(), root.resolve()]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for item in roots:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def resolve_case_wav_strict(case: Phase2Case, roots: list[Path]) -> Path:
    for root in roots:
        candidate = (root / case.wav_relative_path).resolve()
        if candidate.is_file():
            return candidate
    raise Phase3DataError(
        f"case {case.case_id}: wav {case.wav_relative_path} not found under roots "
        f"{[str(root) for root in roots]}"
    )


def verify_wav_sha256(path: Path, expected_sha256: str, case_id: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise Phase3DataError(
            f"case {case_id}: wav sha256 mismatch at {path} " f"({actual} != {expected_sha256})"
        )


def load_phase2_manifest(path: Path) -> Phase2Manifest:
    manifest = Phase2Manifest.load(path)
    if manifest.schema_version != PHASE2_MANIFEST_SCHEMA:
        raise Phase3DataError(f"manifest {path.name}: unsupported schema {manifest.schema_version}")
    return manifest


def dataset_label(manifest_id: str) -> tuple[str, str, str]:
    if manifest_id.startswith("ls_") or manifest_id == "mixed_dev_pool":
        return "librispeech_synthetic", "en", "synthetic_splice"
    if manifest_id.startswith("ami_"):
        return "ami", "en", "meeting_mix_headset"
    if manifest_id.startswith("alimeeting_"):
        return "alimeeting", "zh", "meeting_far_field"
    return manifest_id, "unknown", "unknown"


def case_dataset_label(case: Phase2Case, manifest_id: str) -> tuple[str, str, str]:
    if manifest_id == "mixed_dev_pool":
        if case.kind == "real_meeting":
            return dataset_label(f"ami_{case.case_id}")
        return "librispeech_synthetic", "en", "synthetic_splice"
    return dataset_label(manifest_id)


def gap_bucket_for_change(
    change: SpeakerChangeGT,
    regions: list[SpeakerRegion],
    condition: dict[str, Any],
) -> str:
    if change.kind == "interruption_onset":
        return OVERLAP_BUCKET
    if change.kind == "clean_handoff":
        return GAP_BUCKET_ZERO
    gap_ms = condition.get("gap_ms")
    if gap_ms is not None:
        if gap_ms == 0:
            return GAP_BUCKET_ZERO
        if gap_ms >= 300:
            return GAP_BUCKET_LARGE
        return GAP_BUCKET_SMALL
    prev_region = None
    for region in regions:
        if region.end_sample == change.change_sample:
            prev_region = region
            break
    if prev_region is not None and not prev_region.speakers:
        gap_samples = prev_region.end_sample - prev_region.start_sample
        if gap_samples <= 0:
            return GAP_BUCKET_ZERO
        if gap_samples >= 4800:
            return GAP_BUCKET_LARGE
        return GAP_BUCKET_SMALL
    return GAP_BUCKET_ZERO


def turn_bucket_for_change(
    change: SpeakerChangeGT,
    regions: list[SpeakerRegion],
    condition: dict[str, Any],
) -> str:
    duration_target = condition.get("duration_target_s")
    next_duration_s: float | None = None
    if duration_target is not None:
        next_duration_s = float(duration_target)
    else:
        for region in regions:
            if region.start_sample == change.change_sample and region.speakers:
                next_duration_s = (region.end_sample - region.start_sample) / 16000.0
                break
    if next_duration_s is None:
        return TURN_BUCKET_MID
    if next_duration_s <= 0.75:
        return TURN_BUCKET_SHORT
    if next_duration_s <= 1.5:
        return TURN_BUCKET_MID
    return TURN_BUCKET_LONG


def stress_label(case: Phase2Case) -> str:
    return "codec_noise" if case.transforms else "clean"


@dataclass(frozen=True, slots=True)
class ChangeLabels:
    change_sample: int
    gap_bucket: str
    turn_bucket: str
    gt_kind: str


@dataclass(slots=True)
class CaseInputs:
    case: Phase2Case
    audio_epoch: int
    wav_path: Path
    samples: np.ndarray
    vad_boundaries: list[SpeakerBoundaryEvent]
    vad_utterances: list[tuple[int, int]]
    gt_changes: list[SpeakerChangeGT]
    active_speech_samples: int
    dataset: str
    language: str
    domain: str
    stress: str
    change_labels: list[ChangeLabels] = field(default_factory=list)
    length_samples: int = 0
    b0_chunk_wall_seconds: list[float] = field(default_factory=list)
    b0_cpu_seconds: float = 0.0
    b0_wall_seconds: float = 0.0


def timed_b0_replay(
    samples: np.ndarray,
    *,
    audio_epoch: int,
    engine_factory: Any,
    chunk_samples: int = 512,
) -> tuple[list[SpeakerBoundaryEvent], list[float], float, float]:
    import time

    from experiments.speaker_turn_boundary.vad_baseline import VadBoundaryReplay

    replay = VadBoundaryReplay(engine_factory=engine_factory, chunk_samples=chunk_samples)
    replay.start_epoch(audio_epoch)
    chunk_walls: list[float] = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    offset = 0
    while offset < samples.size:
        chunk = samples[offset : offset + chunk_samples]
        if chunk.size < chunk_samples:
            break
        started = time.perf_counter()
        replay.process_chunk(chunk)
        chunk_walls.append(time.perf_counter() - started)
        offset += chunk_samples
    return (
        replay.boundaries,
        chunk_walls,
        time.process_time() - cpu_start,
        time.perf_counter() - wall_start,
    )


def build_case_inputs(
    manifest: Phase2Manifest,
    *,
    roots: list[Path],
    engine_factory: Any,
    case_ids: list[str] | None = None,
    verify_hashes: bool = True,
    wav_hash_cache: dict[str, bool] | None = None,
) -> list[CaseInputs]:
    from experiments.speaker_turn_boundary.run_eres_sweep import extract_vad_utterances

    hash_cache = wav_hash_cache if wav_hash_cache is not None else {}
    inputs: list[CaseInputs] = []
    selected = [case for case in manifest.cases if case_ids is None or case.case_id in case_ids]
    for epoch_index, case in enumerate(selected):
        wav_path = resolve_case_wav_strict(case, roots)
        cache_key = f"{wav_path}:{case.wav_sha256}"
        if verify_hashes and cache_key not in hash_cache:
            verify_wav_sha256(wav_path, case.wav_sha256, case.case_id)
            hash_cache[cache_key] = True
        samples = load_canonical_wav(wav_path)
        if samples.size != case.duration_samples:
            raise Phase3DataError(
                f"case {case.case_id}: duration mismatch "
                f"({samples.size} != {case.duration_samples})"
            )
        vad_boundaries, chunk_walls, b0_cpu, b0_wall = timed_b0_replay(
            samples, audio_epoch=epoch_index, engine_factory=engine_factory
        )
        utterances = extract_vad_utterances(samples, engine_factory)
        regions = rebase_regions_to_epoch(list(case.regions), epoch_index)
        changes, _ = classify_active_speaker_transitions(regions)
        dataset, language, domain = case_dataset_label(case, manifest.manifest_id)
        labels = [
            ChangeLabels(
                change_sample=change.change_sample,
                gap_bucket=gap_bucket_for_change(change, regions, case.condition),
                turn_bucket=turn_bucket_for_change(change, regions, case.condition),
                gt_kind=change.kind,
            )
            for change in changes
        ]
        inputs.append(
            CaseInputs(
                case=case,
                audio_epoch=epoch_index,
                wav_path=wav_path,
                samples=samples,
                vad_boundaries=vad_boundaries,
                vad_utterances=utterances,
                gt_changes=changes,
                active_speech_samples=active_speech_sample_count(list(case.regions)),
                dataset=dataset,
                language=language,
                domain=domain,
                stress=stress_label(case),
                change_labels=labels,
                length_samples=samples.size,
                b0_chunk_wall_seconds=chunk_walls,
                b0_cpu_seconds=b0_cpu,
                b0_wall_seconds=b0_wall,
            )
        )
    return inputs
