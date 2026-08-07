from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.adapters.eres2netv2 import (
    ERES_SAMPLE_RATE_HZ,
    EresAdjacentProfile,
    EresBoundary,
    EresEmbeddingRuntime,
    EresStableAnchorProfile,
    clamp_confidence,
    eres_adjacent_profiles,
    eres_anchor_profiles,
    eres_embedding_profile_dict,
)
from experiments.speaker_turn_boundary.config import (
    CANONICAL_CHUNK_SAMPLES,
    EXPERIMENT_DATA_DIR,
    EXPERIMENT_RESULTS_DIR,
    VAD_COALESCE_WINDOW_SAMPLES,
)
from experiments.speaker_turn_boundary.events import (
    DetectorProgress,
    SpeakerBoundaryEvent,
)
from experiments.speaker_turn_boundary.ground_truth import (
    active_speech_sample_count,
    classify_active_speaker_transitions,
    rebase_regions_to_epoch,
)
from experiments.speaker_turn_boundary.metadata import collect_runtime_metadata
from experiments.speaker_turn_boundary.metrics import (
    aggregate_cases,
    incremental_over_b0,
)
from experiments.speaker_turn_boundary.sweep_common import (
    EpochSweepRecord,
    b0_aggregate_for_manifest,
    build_sweep_result,
    load_manifest,
    new_summary,
    profile_summary_dict,
)
from experiments.speaker_turn_boundary.vad_baseline import load_canonical_wav

ERES_CHECKPOINTS = {
    "E-standard": {
        "model_id": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "ckpt": "pretrained_eres2netv2.ckpt",
        "onnx": "eres2netv2.onnx",
        "sha256": "0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c",
    },
    "E-w24s4ep4": {
        "model_id": "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common",
        "ckpt": "pretrained_eres2netv2w24s4ep4.ckpt",
        "onnx": "eres2netv2_w24s4ep4.onnx",
        "sha256": "740bb6584a99ee4cf910101536acba38c15a8017ea6a3a2813ec668fb62981f1",
    },
}


def eres_artifact_filename(
    manifest_id: str,
    checkpoint_tag: str,
    profile_kind: str,
    profile_id: str,
) -> str:
    stem = profile_id.replace(".", "p")
    return f"sweep_eres_{manifest_id}_{checkpoint_tag}_{profile_kind}_{stem}.json"


def extract_vad_utterances(samples: np.ndarray, engine_factory) -> list[tuple[int, int]]:
    from puripuly_heart.core.vad.gating import create_peer_vad_gating

    gating = create_peer_vad_gating(
        engine_factory(), sample_rate_hz=16000, ring_buffer_ms=500, hangover_ms=500
    )
    utterances: list[tuple[int, int]] = []
    current_start: int | None = None
    offset = 0
    while offset < samples.size:
        chunk = samples[offset : offset + CANONICAL_CHUNK_SAMPLES]
        if chunk.size < CANONICAL_CHUNK_SAMPLES:
            break
        for event in gating.process_chunk(chunk):
            kind = type(event).__name__
            if kind == "SpeechStart":
                current_start = offset
            elif kind == "SpeechEnd":
                trailing_silence_ms = int(getattr(event, "trailing_silence_ms", 0))
                chunk_ms = CANONICAL_CHUNK_SAMPLES / 16000 * 1000.0
                silence_run = int(round(trailing_silence_ms / chunk_ms))
                speech_end = (
                    offset // CANONICAL_CHUNK_SAMPLES + 1 - silence_run
                ) * CANONICAL_CHUNK_SAMPLES
                if current_start is not None and speech_end > current_start:
                    utterances.append((current_start, min(speech_end, samples.size)))
                current_start = None
        offset += CANONICAL_CHUNK_SAMPLES
    if current_start is not None:
        utterances.append((current_start, samples.size))
    return utterances


class EresDetectorRunner:
    def __init__(self, runtime: EresEmbeddingRuntime, checkpoint_tag: str) -> None:
        self._runtime = runtime
        self._checkpoint_tag = checkpoint_tag
        self._embedding_cache: dict[tuple[int, int], np.ndarray] = {}
        self._embedding_times: list[float] = []

    def start_epoch(self, audio_epoch: int) -> None:
        self._audio_epoch = audio_epoch

    def run_case(
        self, samples: np.ndarray, utterances: list[tuple[int, int]], builder
    ) -> tuple[list[SpeakerBoundaryEvent], list[DetectorProgress]]:
        boundaries: list[SpeakerBoundaryEvent] = []
        progress: list[DetectorProgress] = []
        for start_sample, end_sample in utterances:
            emitted, snapshots = builder(samples, (start_sample, end_sample))
            boundaries.extend(self._to_events(emitted))
            progress.extend(
                DetectorProgress(
                    audio_epoch=self._audio_epoch,
                    observed_source_sample=snapshot.observed_source_sample,
                    safe_boundary_frontier_sample=snapshot.safe_boundary_frontier_sample,
                )
                for snapshot in snapshots
            )
        progress.append(
            DetectorProgress(
                audio_epoch=self._audio_epoch,
                observed_source_sample=samples.size,
                safe_boundary_frontier_sample=samples.size,
            )
        )
        return boundaries, progress

    def _to_events(self, emitted: list[EresBoundary]) -> list[SpeakerBoundaryEvent]:
        events: list[SpeakerBoundaryEvent] = []
        for boundary in emitted:
            events.append(
                SpeakerBoundaryEvent(
                    audio_epoch=self._audio_epoch,
                    boundary_source_sample=boundary.boundary_sample,
                    observed_source_sample_at_emit=boundary.observed_sample,
                    emitted_monotonic_ns=0,
                    confidence=clamp_confidence(boundary.confidence),
                    source=f"eres:{self._checkpoint_tag}:{boundary.debug['profile']['profile_id']}",
                    debug=boundary.debug,
                )
            )
        return events

    def embed_cached(self, samples: np.ndarray, start: int, end: int) -> np.ndarray:
        key = (int(start), int(end))
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        begin = time.perf_counter()
        embedding = self._runtime.embed(samples[start:end])
        self._embedding_times.append(time.perf_counter() - begin)
        self._embedding_cache[key] = embedding
        return embedding


def _adjacent_builder(runner: EresDetectorRunner, profile: EresAdjacentProfile, audio_epoch: int):
    from experiments.speaker_turn_boundary.adapters.eres2netv2 import cosine_similarity

    window = int(round(profile.window_seconds * ERES_SAMPLE_RATE_HZ))
    step = int(round(profile.step_seconds * ERES_SAMPLE_RATE_HZ))

    def build(
        samples: np.ndarray, utterance: tuple[int, int]
    ) -> tuple[list[EresBoundary], list[DetectorProgress]]:
        start_sample, end_sample = utterance
        boundaries: list[EresBoundary] = []
        progress: list[DetectorProgress] = []
        candidate: tuple[float, int] | None = None
        position = start_sample + window
        while position + window <= end_sample:
            left = runner.embed_cached(samples, position - window, position)
            right = runner.embed_cached(samples, position, position + window)
            score = cosine_similarity(left, right)
            is_candidate = score < profile.threshold
            if is_candidate and candidate is None:
                candidate = (score, position)
            elif is_candidate and candidate is not None:
                if profile.confirmation == 2:
                    first_score, first_position = candidate
                    boundaries.append(
                        EresBoundary(
                            audio_epoch=audio_epoch,
                            boundary_sample=first_position,
                            observed_sample=first_position + window,
                            confidence=clamp_confidence(first_score),
                            debug={
                                "profile": {**profile.to_dict(), "profile_id": profile.profile_id},
                                "score_first": first_score,
                                "score_second": score,
                            },
                        )
                    )
                    candidate = None
            else:
                candidate = None
            progress.append(
                DetectorProgress(
                    audio_epoch=audio_epoch,
                    observed_source_sample=position + window,
                    safe_boundary_frontier_sample=position - step + 1,
                )
            )
            position += step
        if candidate is not None and profile.confirmation == 1:
            first_score, first_position = candidate
            boundaries.append(
                EresBoundary(
                    audio_epoch=audio_epoch,
                    boundary_sample=first_position,
                    observed_sample=first_position + window,
                    confidence=clamp_confidence(first_score),
                    debug={
                        "profile": {**profile.to_dict(), "profile_id": profile.profile_id},
                        "score_first": first_score,
                    },
                )
            )
        return boundaries, progress

    return build


def _anchor_builder(runner: EresDetectorRunner, profile: EresStableAnchorProfile, audio_epoch: int):
    window = int(round(profile.window_seconds * ERES_SAMPLE_RATE_HZ))
    step = int(round(profile.step_seconds * ERES_SAMPLE_RATE_HZ))

    def build(
        samples: np.ndarray, utterance: tuple[int, int]
    ) -> tuple[list[EresBoundary], list[DetectorProgress]]:
        from experiments.speaker_turn_boundary.adapters.eres2netv2 import cosine_similarity

        start_sample, end_sample = utterance
        boundaries: list[EresBoundary] = []
        progress: list[DetectorProgress] = []
        if start_sample + window > end_sample:
            return boundaries, progress
        anchor = runner.embed_cached(samples, start_sample, start_sample + window)
        anchor = anchor / (np.linalg.norm(anchor) + 1e-12)
        candidate: tuple[float, int, np.ndarray] | None = None
        position = start_sample + window
        while position + window <= end_sample:
            probe = runner.embed_cached(samples, position, position + window)
            score = cosine_similarity(anchor, probe)
            is_candidate = score < profile.threshold
            if is_candidate and candidate is None:
                candidate = (score, position, probe)
            elif is_candidate and candidate is not None:
                first_score, first_position, first_embedding = candidate
                confirmed = profile.confirmation == 1
                mutual = None
                if profile.confirmation == 2:
                    mutual = cosine_similarity(first_embedding, probe)
                    confirmed = mutual >= profile.mutual_similarity_threshold
                if confirmed:
                    boundaries.append(
                        EresBoundary(
                            audio_epoch=audio_epoch,
                            boundary_sample=first_position,
                            observed_sample=first_position + window,
                            confidence=clamp_confidence(first_score),
                            debug={
                                "profile": {**profile.to_dict(), "profile_id": profile.profile_id},
                                "score_first": first_score,
                                "score_second": score,
                                "mutual_similarity": mutual,
                            },
                        )
                    )
                    anchor = probe / (np.linalg.norm(probe) + 1e-12)
                    candidate = None
                else:
                    candidate = (score, position, probe)
            else:
                if (
                    profile.anchor_update == "ema"
                    and candidate is None
                    and np.linalg.norm(probe) > 0
                ):
                    alpha = profile.anchor_ema_alpha
                    anchor = (1.0 - alpha) * anchor + alpha * probe / (
                        np.linalg.norm(probe) + 1e-12
                    )
                    anchor = anchor / (np.linalg.norm(anchor) + 1e-12)
            progress.append(
                DetectorProgress(
                    audio_epoch=audio_epoch,
                    observed_source_sample=position + window,
                    safe_boundary_frontier_sample=position - step + 1,
                )
            )
            position += step
        if candidate is not None and profile.confirmation == 1:
            first_score, first_position, _ = candidate
            boundaries.append(
                EresBoundary(
                    audio_epoch=audio_epoch,
                    boundary_sample=first_position,
                    observed_sample=first_position + window,
                    confidence=clamp_confidence(first_score),
                    debug={
                        "profile": {**profile.to_dict(), "profile_id": profile.profile_id},
                        "score_first": first_score,
                    },
                )
            )
        return boundaries, progress

    return build


def run_sweep(
    *,
    manifest_path: Path,
    data_dir: Path,
    eres_onnx_root: Path,
    model_path: Path,
    out_dir: Path,
    checkpoints: list[str],
    run_adjacent: bool,
    run_anchor: bool,
    case_ids: list[str] | None,
    coalesce_window_samples: int,
    anchor_windows: tuple[float, ...],
    smoke_dir: Path | None,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    engine_factory = _silero_engine_factory(model_path)
    smoke_wavs = sorted(smoke_dir.glob("*.wav")) if smoke_dir is not None else None
    b0_aggregate = b0_aggregate_for_manifest(
        manifest,
        data_dir,
        engine_factory=engine_factory,
        case_ids=case_ids,
        coalesce_window_samples=coalesce_window_samples,
    )
    summary = new_summary(manifest_id=manifest.manifest_id, detector_family="eres2netv2")
    runtime_metadata = collect_runtime_metadata()
    for checkpoint_tag in checkpoints:
        info = ERES_CHECKPOINTS[checkpoint_tag]
        onnx_path = eres_onnx_root / info["onnx"]
        runtime = EresEmbeddingRuntime(str(onnx_path))
        runner = EresDetectorRunner(runtime, checkpoint_tag)
        checkpoint_summary: dict[str, object] = {}
        annotated_inputs, smoke_inputs = build_epoch_inputs(
            manifest, data_dir, engine_factory, case_ids, smoke_wavs
        )
        profiles = []
        if run_adjacent:
            profiles.extend(eres_adjacent_profiles())
        if run_anchor:
            profiles.extend(eres_anchor_profiles(windows=anchor_windows))
        for profile in profiles:
            builder = (
                _adjacent_builder(runner, profile, 0)
                if isinstance(profile, EresAdjacentProfile)
                else _anchor_builder(runner, profile, 0)
            )
            profile_id = profile.profile_id
            started = time.perf_counter()
            epochs = _run_epochs(annotated_inputs, runner, builder, coalesce_window_samples)
            smoke_epochs = _run_epochs(smoke_inputs, runner, builder, coalesce_window_samples)
            aggregate = aggregate_cases(
                [record.metrics for record in epochs],
                profile_id=f"{checkpoint_tag}:{profile_id}",
                coalescing_reports=[record.coalescing.report for record in epochs],
                smoke_metrics=[record.metrics for record in smoke_epochs],
            )
            incremental = incremental_over_b0(b0_aggregate, aggregate)
            profile_kind = "adjacent" if isinstance(profile, EresAdjacentProfile) else "anchor"
            extra = {
                "frontend_profile": eres_embedding_profile_dict(192),
                "checkpoint_sha256": info["sha256"],
                "model_id": info["model_id"],
                "embedding_cache_hits": len(runner._embedding_cache),
                "embedding_compute_seconds_mean": (
                    float(np.mean(runner._embedding_times)) if runner._embedding_times else None
                ),
                "embedding_compute_seconds_p95": (
                    float(np.percentile(runner._embedding_times, 95))
                    if runner._embedding_times
                    else None
                ),
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
            build_sweep_result(
                manifest=manifest,
                detector={
                    "family": "eres2netv2",
                    "checkpoint": checkpoint_tag,
                    "checkpoint_sha256": info["sha256"],
                    "profile_kind": profile_kind,
                    "profile": {**profile.to_dict(), "profile_id": profile.profile_id},
                    "profile_id": profile_id,
                    "coalesce_window_samples": coalesce_window_samples,
                },
                runtime_metadata=runtime_metadata,
                epochs=epochs,
                aggregate=aggregate,
                b0_aggregate=b0_aggregate,
                incremental=incremental,
                smoke_epochs=smoke_epochs,
                extra=extra,
                out_dir=out_dir,
                out_name=eres_artifact_filename(
                    manifest.manifest_id,
                    checkpoint_tag,
                    profile_kind,
                    profile_id,
                ),
            )
            checkpoint_summary[profile_id] = profile_summary_dict(
                aggregate, b0_aggregate, incremental
            )
        summary["variants"][checkpoint_tag] = checkpoint_summary
    summary_path = out_dir / f"sweep_eres_summary_{manifest.manifest_id}.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {summary_path}")
    return summary


def build_epoch_inputs(manifest, data_dir, engine_factory, case_ids, smoke_wavs):
    from experiments.speaker_turn_boundary.vad_baseline import replay_wav_epoch

    selected = [case for case in manifest.cases if case_ids is None or case.case_id in case_ids]
    cases = [
        (case.case_id, (data_dir / case.wav_relative_path).resolve(), case.regions)
        for case in selected
    ]
    smoke_cases = [(f"smoke:{wav_path.stem}", wav_path, []) for wav_path in smoke_wavs or []]

    def build_one(case_id, wav_path, regions, audio_epoch):
        samples = load_canonical_wav(wav_path)
        vad_result = replay_wav_epoch(
            wav_path, audio_epoch=audio_epoch, engine_factory=engine_factory
        )
        utterances = extract_vad_utterances(samples, engine_factory)
        changes, _ = classify_active_speaker_transitions(
            rebase_regions_to_epoch(list(regions), audio_epoch)
        )
        return {
            "case_id": case_id,
            "audio_epoch": audio_epoch,
            "samples": samples,
            "vad_boundaries": vad_result.boundaries,
            "utterances": utterances,
            "changes": changes,
            "length_samples": vad_result.length_samples,
            "active_speech_samples": active_speech_sample_count(list(regions)),
        }

    annotated = [
        build_one(case_id, wav_path, regions, epoch_index)
        for epoch_index, (case_id, wav_path, regions) in enumerate(cases)
    ]
    smoke = [
        build_one(case_id, wav_path, regions, len(annotated) + index)
        for index, (case_id, wav_path, regions) in enumerate(smoke_cases)
    ]
    return annotated, smoke


def _run_epochs(epoch_inputs, runner, builder, coalesce_window_samples):
    from experiments.speaker_turn_boundary.coalescing import (
        CoalesceConfig,
        coalesce_vad_and_detector,
    )
    from experiments.speaker_turn_boundary.metrics import (
        detector_events_to_cuts,
        evaluate_case,
    )

    records: list[EpochSweepRecord] = []
    for epoch_input in epoch_inputs:
        samples = epoch_input["samples"]
        audio_epoch = epoch_input["audio_epoch"]
        runner.start_epoch(audio_epoch)
        boundaries, progress = runner.run_case(samples, epoch_input["utterances"], builder)
        outcome = coalesce_vad_and_detector(
            epoch_input["vad_boundaries"],
            boundaries,
            config=CoalesceConfig(window_samples=coalesce_window_samples),
        )
        vad_count = sum(1 for cut in outcome.cuts if cut.kind == "vad")
        detector_cut_count = sum(1 for cut in outcome.cuts if cut.kind == "detector")
        metrics = evaluate_case(
            case_id=epoch_input["case_id"],
            audio_epoch=audio_epoch,
            gt_changes=epoch_input["changes"],
            cuts=outcome.cuts,
            detector_events=detector_events_to_cuts(boundaries),
            vad_cut_count=vad_count,
            detector_cut_count=detector_cut_count,
            active_speech_samples=epoch_input["active_speech_samples"],
        )
        records.append(
            EpochSweepRecord(
                audio_epoch=audio_epoch,
                case_id=epoch_input["case_id"],
                length_samples=epoch_input["length_samples"],
                gt_changes=epoch_input["changes"],
                vad_boundaries=epoch_input["vad_boundaries"],
                detector_boundaries=boundaries,
                progress=progress,
                coalescing=outcome,
                metrics=metrics,
            )
        )
    return records


def _silero_engine_factory(model_path: Path):
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    return lambda: SileroVadOnnx(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 ERes2NetV2 adjacent/anchor sweep on the Phase 0 corpus"
    )
    parser.add_argument(
        "--manifest", type=Path, default=EXPERIMENT_DATA_DIR / "manifests" / "b0_phase0.json"
    )
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT_DATA_DIR)
    parser.add_argument("--eres-onnx-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=EXPERIMENT_RESULTS_DIR)
    parser.add_argument("--checkpoint", action="append", default=None)
    parser.add_argument("--adjacent", action="store_true")
    parser.add_argument("--anchor", action="store_true")
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--coalesce-window-samples", type=int, default=VAD_COALESCE_WINDOW_SAMPLES)
    parser.add_argument("--anchor-window", action="append", type=float, default=None)
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=None,
        help="directory of 16k mono wavs run without GT (smoke inputs)",
    )
    args = parser.parse_args()
    checkpoints = args.checkpoint or list(ERES_CHECKPOINTS)
    anchor_windows = tuple(args.anchor_window or (0.50, 0.75, 1.00, 1.50))
    if not args.adjacent and not args.anchor:
        args.adjacent = True
        args.anchor = True
    model_path = args.model
    if model_path is None:
        from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path

        model_path = Path(str(bundled_silero_vad_onnx_path()))
    run_sweep(
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        eres_onnx_root=args.eres_onnx_root,
        model_path=model_path,
        out_dir=args.out,
        checkpoints=checkpoints,
        run_adjacent=args.adjacent,
        run_anchor=args.anchor,
        case_ids=args.case,
        coalesce_window_samples=args.coalesce_window_samples,
        anchor_windows=anchor_windows,
        smoke_dir=args.smoke_dir,
    )


if __name__ == "__main__":
    main()
