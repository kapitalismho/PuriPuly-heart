from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.speaker_turn_boundary.adapters.eres2netv2 import (
    EresAdjacentProfile,
    EresEmbeddingRuntime,
    EresStableAnchorProfile,
    cosine_similarity,
    eres_adjacent_profiles,
    eres_embedding_profile_dict,
    threshold_range,
)
from experiments.speaker_turn_boundary.config import EXPERIMENT_DATA_DIR
from experiments.speaker_turn_boundary.corpus.phase2_schemas import Phase2Manifest
from experiments.speaker_turn_boundary.frontend import frontend_profile
from experiments.speaker_turn_boundary.phase3_data import (
    CaseInputs,
    build_case_inputs,
    load_phase2_manifest,
    phase3_wav_roots,
)
from experiments.speaker_turn_boundary.phase3_eres import (
    EresEmbeddingStore,
    cached_adjacent_events,
    cached_anchor_events,
    enumerate_adjacent_windows,
    enumerate_anchor_windows,
    load_legacy_case,
    sanitize_case_id,
)
from experiments.speaker_turn_boundary.phase3_eval import (
    ProfileEvaluation,
    evaluate_profile,
    scheduling_backlog_ms,
)
from experiments.speaker_turn_boundary.phase3_ls import (
    LSCaptureEpoch,
    LSEENDCapture,
    load_capture_cache,
    load_sidecar_metadata,
    replay_profile,
    save_capture_cache,
)
from experiments.speaker_turn_boundary.provenance import LS_EEND_VARIANTS, sha256_file
from experiments.speaker_turn_boundary.reducer import ReductionProfile
from experiments.speaker_turn_boundary.run_eres_sweep import ERES_CHECKPOINTS
from experiments.speaker_turn_boundary.run_ls_eend_sweep import (
    DEFAULT_MEDIAN_WIDTHS,
    DEFAULT_PERSISTENCE,
    DEFAULT_POLICIES,
    DEFAULT_THRESHOLDS,
)
from experiments.speaker_turn_boundary.schemas import sha256_hex

ERES_ONNX_SHA256 = {
    "E-standard": "7a6d4f89dcb92a554806bdf6bfb13c7fae0a63e8f992a49b3a503b9a03c705cf",
    "E-w24s4ep4": "3761572a872a29f36af66065075cc9a48adc23c8b26fb0c68488aa3ed8f35f26",
}
ANCHOR_WINDOWS = (0.50, 0.75, 1.00, 1.50)
ANCHOR_STEPS = (0.10, 0.25)


class Phase3StageError(RuntimeError):
    pass


@dataclass(slots=True)
class StageContext:
    data_dir: Path = EXPERIMENT_DATA_DIR
    corpus_root: Path | None = None
    hf_root: Path | None = None
    eres_onnx_root: Path | None = None
    scratch: Path | None = None
    legacy_scratch: Path | None = None
    engine_factory: Any = None
    wav_hash_cache: dict[str, bool] = field(default_factory=dict)

    def roots(self) -> list[Path]:
        return phase3_wav_roots(self.data_dir, self.corpus_root)


@dataclass(slots=True)
class LSCheckpointData:
    checkpoint: str
    captures: list[LSCaptureEpoch]
    track_count: int
    stats: dict[str, Any]


@dataclass(slots=True)
class EresCheckpointData:
    checkpoint: str
    embeddings_by_epoch: dict[int, dict[tuple[int, int], np.ndarray]]
    stats: dict[str, Any]


def default_engine_factory():
    from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    model_path = Path(str(bundled_silero_vad_onnx_path()))
    return lambda: SileroVadOnnx(model_path)


def build_inputs(
    ctx: StageContext,
    manifest_path: Path,
    *,
    case_ids: list[str] | None = None,
) -> tuple[Phase2Manifest, list[CaseInputs]]:
    manifest = load_phase2_manifest(manifest_path)
    if ctx.engine_factory is None:
        ctx.engine_factory = default_engine_factory()
    inputs = build_case_inputs(
        manifest,
        roots=ctx.roots(),
        engine_factory=ctx.engine_factory,
        case_ids=case_ids,
        wav_hash_cache=ctx.wav_hash_cache,
    )
    return manifest, inputs


def ls_frontend_contract() -> str:
    return sha256_hex(
        {
            "schema": "experiments.speaker_turn_boundary.ls_streaming_frontend.v2",
            "profile": frontend_profile().to_dict(),
            "state_scope": "continuous_within_audio_epoch",
            "tail_policy": "flush_real_frontend_then_decode_pending_without_padding_audio",
        }
    )


def eres_frontend_contract() -> str:
    return sha256_hex(
        {
            "schema": "experiments.speaker_turn_boundary.eres_frontend.v2",
            "profile": eres_embedding_profile_dict(192),
            "window_padding": "none",
            "state_scope": "reset_each_vad_utterance",
        }
    )


def expected_model_hashes() -> dict[str, str]:
    hashes = {
        f"ls_eend:{checkpoint}": str(info["onnx_sha256"])
        for checkpoint, info in LS_EEND_VARIANTS.items()
    }
    hashes.update(
        {f"eres2netv2:{checkpoint}": expected for checkpoint, expected in ERES_ONNX_SHA256.items()}
    )
    return hashes


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise Phase3StageError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise Phase3StageError(f"{label} sha256 mismatch: {actual} != {expected_sha256}")


def load_or_capture_ls(
    ctx: StageContext,
    manifest: Phase2Manifest,
    inputs: list[CaseInputs],
    checkpoint: str,
) -> LSCheckpointData:
    if ctx.hf_root is None or ctx.scratch is None:
        raise Phase3StageError("hf_root and scratch are required for LS-EEND")
    info = LS_EEND_VARIANTS[checkpoint]
    onnx_path = ctx.hf_root / str(info["dir"]) / str(info["onnx"])
    sidecar_path = ctx.hf_root / str(info["dir"]) / str(info["sidecar"])
    _verify_file(onnx_path, str(info["onnx_sha256"]), f"LS {checkpoint} ONNX")
    _verify_file(sidecar_path, str(info["sidecar_sha256"]), f"LS {checkpoint} sidecar")
    wav_hashes = {case.case.case_id: case.case.wav_sha256 for case in inputs}
    cache_root = ctx.scratch / "cache" / "ls_capture_v2"
    cached = load_capture_cache(
        cache_root,
        checkpoint=checkpoint,
        checkpoint_sha256=str(info["onnx_sha256"]),
        sidecar_sha256=str(info["sidecar_sha256"]),
        frontend_contract_sha256=ls_frontend_contract(),
        manifest_sha256=manifest.hash,
        case_wav_sha256=wav_hashes,
    )
    if cached is not None:
        captures, index = cached
        return LSCheckpointData(
            checkpoint=checkpoint,
            captures=captures,
            track_count=int(index["track_count"]),
            stats=_ls_stats(captures, cache_hit=True, load_seconds=None),
        )
    metadata = load_sidecar_metadata(sidecar_path)
    runtime = LSEENDCapture(onnx_path, metadata, checkpoint_variant=checkpoint)
    captures: list[LSCaptureEpoch] = []
    started = time.perf_counter()
    for case in inputs:
        captures.append(
            runtime.run_case(
                case.samples,
                case_id=case.case.case_id,
                audio_epoch=case.audio_epoch,
            )
        )
    total_seconds = time.perf_counter() - started
    save_capture_cache(
        cache_root,
        checkpoint=checkpoint,
        checkpoint_sha256=str(info["onnx_sha256"]),
        sidecar_sha256=str(info["sidecar_sha256"]),
        frontend_contract_sha256=ls_frontend_contract(),
        manifest_sha256=manifest.hash,
        case_wav_sha256=wav_hashes,
        captures=captures,
        track_count=runtime.real_output_dim,
    )
    stats = _ls_stats(captures, cache_hit=False, load_seconds=runtime.load_seconds)
    stats["capture_elapsed_seconds"] = total_seconds
    return LSCheckpointData(
        checkpoint=checkpoint,
        captures=captures,
        track_count=runtime.real_output_dim,
        stats=stats,
    )


def _ls_stats(
    captures: list[LSCaptureEpoch],
    *,
    cache_hit: bool,
    load_seconds: float | None,
) -> dict[str, Any]:
    chunk_times = [value for capture in captures for value in capture.chunk_wall_seconds]
    audio_seconds = sum(capture.length_samples for capture in captures) / 16000.0
    compute_seconds = sum(capture.wall_seconds for capture in captures)
    return {
        "cache_hit": cache_hit,
        "model_load_seconds": load_seconds,
        "case_count": len(captures),
        "audio_seconds": audio_seconds,
        "captured_compute_seconds": compute_seconds,
        "captured_realtime_factor": compute_seconds / audio_seconds if audio_seconds else None,
        "chunk_service_seconds": {
            "count": len(chunk_times),
            "mean": float(np.mean(chunk_times)) if chunk_times else None,
            "p50": float(np.percentile(chunk_times, 50)) if chunk_times else None,
            "p95": float(np.percentile(chunk_times, 95)) if chunk_times else None,
        },
        "single_stream_scheduling": scheduling_backlog_ms(chunk_times),
    }


def ls_profiles() -> list[ReductionProfile]:
    return [
        ReductionProfile(
            threshold=threshold,
            persistence=persistence,
            policy=policy,
            median_width=median_width,
        )
        for policy in DEFAULT_POLICIES
        for median_width in DEFAULT_MEDIAN_WIDTHS
        for persistence in DEFAULT_PERSISTENCE
        for threshold in DEFAULT_THRESHOLDS
    ]


def anchor_profiles() -> list[EresStableAnchorProfile]:
    return [
        EresStableAnchorProfile(
            window_seconds=window,
            step_seconds=step,
            threshold=threshold,
            confirmation=confirmation,
            mutual_similarity_threshold=0.5,
            anchor_update=update,
            anchor_ema_alpha=0.9,
        )
        for window in ANCHOR_WINDOWS
        for step in ANCHOR_STEPS
        for threshold in threshold_range()
        for confirmation in (1, 2)
        for update in ("none", "ema")
    ]


def all_eres_profiles() -> tuple[list[EresAdjacentProfile], list[EresStableAnchorProfile]]:
    return eres_adjacent_profiles(), anchor_profiles()


def prepare_eres_embeddings(
    ctx: StageContext,
    manifest: Phase2Manifest,
    inputs: list[CaseInputs],
    checkpoint: str,
    *,
    adjacent_profiles: list[EresAdjacentProfile],
    stable_profiles: list[EresStableAnchorProfile],
) -> EresCheckpointData:
    if ctx.eres_onnx_root is None or ctx.scratch is None:
        raise Phase3StageError("eres_onnx_root and scratch are required")
    info = ERES_CHECKPOINTS[checkpoint]
    onnx_path = ctx.eres_onnx_root / str(info["onnx"])
    _verify_file(onnx_path, ERES_ONNX_SHA256[checkpoint], f"ERes {checkpoint} ONNX")
    runtime = EresEmbeddingRuntime(str(onnx_path))
    store = EresEmbeddingStore(
        checkpoint_tag=checkpoint,
        checkpoint_sha256=ERES_ONNX_SHA256[checkpoint],
        frontend_contract_sha256=eres_frontend_contract(),
        manifest_sha256=manifest.hash,
        cache_dir=ctx.scratch / "cache" / "eres_embedding_v2",
    )
    adjacent_grid = sorted(
        {(profile.window_seconds, profile.step_seconds) for profile in adjacent_profiles}
    )
    anchor_grid = sorted(
        {(profile.window_seconds, profile.step_seconds) for profile in stable_profiles}
    )
    needed_by_epoch: dict[int, set[tuple[int, int]]] = {}
    legacy_by_epoch: dict[int, dict[tuple[int, int], np.ndarray]] = {}
    legacy_records: list[tuple[CaseInputs, tuple[int, int], np.ndarray]] = []
    for case in inputs:
        needed = enumerate_adjacent_windows(case.vad_utterances, adjacent_grid)
        needed |= enumerate_anchor_windows(case.vad_utterances, anchor_grid)
        needed_by_epoch[case.audio_epoch] = needed
        legacy = _load_legacy_for_case(ctx, manifest.manifest_id, checkpoint, case)
        imported = {window: legacy[window] for window in needed if window in legacy}
        legacy_by_epoch[case.audio_epoch] = imported
        legacy_records.extend((case, window, vector) for window, vector in imported.items())
    evidence = _verify_legacy_records(runtime, legacy_records)
    embeddings_by_epoch: dict[int, dict[tuple[int, int], np.ndarray]] = {}
    started = time.perf_counter()
    for case in inputs:
        embeddings_by_epoch[case.audio_epoch] = store.ensure_case(
            runtime,
            case.samples,
            case.case.case_id,
            case.case.wav_sha256,
            needed_by_epoch[case.audio_epoch],
            imported=legacy_by_epoch[case.audio_epoch],
            import_evidence=evidence if legacy_by_epoch[case.audio_epoch] else None,
        )
    elapsed = time.perf_counter() - started
    stats = {
        "cache_schema": "v2",
        "legacy_import": evidence,
        "windows_loaded_v2": store.window_count_loaded,
        "windows_imported_legacy": len(legacy_records),
        "windows_computed": store.window_count_computed,
        "embedding_elapsed_seconds": elapsed,
        "embedding_service_seconds": {
            "count": len(store.embed_seconds),
            "mean": float(np.mean(store.embed_seconds)) if store.embed_seconds else None,
            "p50": float(np.percentile(store.embed_seconds, 50)) if store.embed_seconds else None,
            "p95": float(np.percentile(store.embed_seconds, 95)) if store.embed_seconds else None,
        },
    }
    return EresCheckpointData(
        checkpoint=checkpoint,
        embeddings_by_epoch=embeddings_by_epoch,
        stats=stats,
    )


def _load_legacy_for_case(
    ctx: StageContext,
    manifest_id: str,
    checkpoint: str,
    case: CaseInputs,
) -> dict[tuple[int, int], np.ndarray]:
    if ctx.legacy_scratch is None:
        return {}
    tags = [f"stage1_{checkpoint}"]
    if manifest_id != "mixed_dev_pool":
        tags = [f"stage3_{manifest_id}_{checkpoint}"]
    for tag in tags:
        path = (
            ctx.legacy_scratch
            / "cache"
            / "eres_emb"
            / tag
            / f"emb_{sanitize_case_id(case.case.case_id)}.npz"
        )
        if path.is_file():
            return load_legacy_case(path)
    return {}


def _verify_legacy_records(
    runtime: EresEmbeddingRuntime,
    records: list[tuple[CaseInputs, tuple[int, int], np.ndarray]],
) -> dict[str, Any] | None:
    if not records:
        return None
    ordered = sorted(
        records,
        key=lambda item: (item[0].case.case_id, item[1][0], item[1][1]),
    )
    sample_count = min(16, len(ordered))
    positions = sorted(
        {
            int(round(index * (len(ordered) - 1) / max(1, sample_count - 1)))
            for index in range(sample_count)
        }
    )
    max_abs = 0.0
    min_cosine = 1.0
    sampled: list[dict[str, Any]] = []
    for position in positions:
        case, (start, end), expected = ordered[position]
        actual = np.asarray(runtime.embed(case.samples[start:end]), dtype=np.float32).reshape(-1)
        absolute = float(np.max(np.abs(expected - actual)))
        cosine = cosine_similarity(expected, actual)
        max_abs = max(max_abs, absolute)
        min_cosine = min(min_cosine, cosine)
        sampled.append(
            {
                "case_id": case.case.case_id,
                "start": start,
                "end": end,
                "max_abs_error": absolute,
                "cosine_similarity": cosine,
            }
        )
    passed = max_abs <= 1e-5 and min_cosine >= 0.99999
    evidence = {
        "available_window_count": len(records),
        "sample_count": len(sampled),
        "max_abs_error": max_abs,
        "min_cosine_similarity": min_cosine,
        "passed": passed,
        "samples": sampled,
    }
    if not passed:
        raise Phase3StageError(f"legacy ERes cache verification failed: {evidence}")
    return evidence


def evaluate_ls_profile(
    inputs: list[CaseInputs],
    checkpoint_data: LSCheckpointData,
    profile: ReductionProfile,
) -> ProfileEvaluation:
    profile_id = f"{checkpoint_data.checkpoint}:{profile.profile_id}"
    events_by_epoch: dict[int, list[Any]] = {}
    for capture in checkpoint_data.captures:
        events, _ = replay_profile(
            capture,
            profile,
            track_count=checkpoint_data.track_count,
            source_label=f"ls_eend:{profile_id}",
        )
        events_by_epoch[capture.audio_epoch] = events
    return evaluate_profile(
        inputs,
        lambda case: events_by_epoch[case.audio_epoch],
        profile_id=profile_id,
        family="ls_eend",
        checkpoint=checkpoint_data.checkpoint,
        profile_kind="reducer",
        params=profile.to_dict(),
        compute=checkpoint_data.stats,
    )


def evaluate_eres_profile(
    inputs: list[CaseInputs],
    checkpoint_data: EresCheckpointData,
    profile: EresAdjacentProfile | EresStableAnchorProfile,
) -> ProfileEvaluation:
    profile_kind = "adjacent" if isinstance(profile, EresAdjacentProfile) else "stable_anchor"
    profile_id = f"{checkpoint_data.checkpoint}:{profile.profile_id}"

    def provider(case: CaseInputs):
        embeddings = checkpoint_data.embeddings_by_epoch[case.audio_epoch]
        if isinstance(profile, EresAdjacentProfile):
            events, _ = cached_adjacent_events(
                utterances=case.vad_utterances,
                embeddings=embeddings,
                profile=profile,
                audio_epoch=case.audio_epoch,
            )
        else:
            events, _ = cached_anchor_events(
                utterances=case.vad_utterances,
                embeddings=embeddings,
                profile=profile,
                audio_epoch=case.audio_epoch,
            )
        return events

    return evaluate_profile(
        inputs,
        provider,
        profile_id=profile_id,
        family="eres2netv2",
        checkpoint=checkpoint_data.checkpoint,
        profile_kind=profile_kind,
        params=profile.to_dict(),
        compute=checkpoint_data.stats,
    )


def reconstruct_profile(row: dict[str, Any]):
    params = row["params"]
    if row["family"] == "ls_eend":
        return ReductionProfile(
            threshold=float(params["threshold"]),
            persistence=int(params["persistence"]),
            policy=str(params["policy"]),
            median_width=int(params["median_width"]),
        )
    if row["profile_kind"] == "adjacent":
        return EresAdjacentProfile(
            window_seconds=float(params["window_seconds"]),
            step_seconds=float(params["step_seconds"]),
            threshold=float(params["threshold"]),
            confirmation=int(params["confirmation"]),
        )
    return EresStableAnchorProfile(
        window_seconds=float(params["window_seconds"]),
        step_seconds=float(params["step_seconds"]),
        threshold=float(params["threshold"]),
        confirmation=int(params["confirmation"]),
        mutual_similarity_threshold=float(params["mutual_similarity_threshold"]),
        anchor_update=str(params["anchor_update"]),
        anchor_ema_alpha=float(params["anchor_ema_alpha"]),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
