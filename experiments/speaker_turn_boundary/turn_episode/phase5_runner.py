from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from experiments.speaker_turn_boundary.adapters.eres2netv2 import (
    EresEmbeddingRuntime,
    kaldi_fbank_numpy,
)
from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.phase2_schemas import Phase2Manifest
from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path
from puripuly_heart.core.vad.gating import create_peer_vad_gating
from puripuly_heart.core.vad.silero import SileroVadOnnx
from experiments.speaker_turn_boundary.vad_baseline import VadBoundaryReplay
from experiments.speaker_turn_boundary.frontend import output_frame_available_16k_count
from experiments.speaker_turn_boundary.run_eres_sweep import ERES_CHECKPOINTS

from .build_coverage_inventory import SILERO_MODEL_SHA256, canonical_json as b0_canonical_json
from .build_episodes import load_ami_raw_words
from .phase4_design import (
    ceil_grid,
    corpus_for,
    load_public_regions,
    load_synthetic_cases,
    synthetic_case_id,
    synthetic_manifest_name,
)
from .phase4_signal import (
    ERES_MODEL_SHA256,
    AudioSource,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    content_hash,
    load_eres_embeddings,
    read_json,
    read_wav,
    save_eres_embeddings,
    sha256_bytes,
    sha256_file,
    source_by_wav,
    tensor_contract,
)
from .phase5_controls import active_intervals_from_lifecycle, causal_energy_candidates
from .phase5_design import (
    HISTORICAL_MANIFEST_BYTE_SHA256,
    INDEPENDENT_AUDIT_SAMPLE_SIZE,
    historical_development_contract,
    historical_input,
    load_populations,
    merge_window_sets,
    pool_block_index,
    proposal_workload,
    proposal_profiles,
    required_windows,
    rows_sha256,
    validate_interstage_word_timing_receipts,
    verify_self_hash,
    window_ledger,
)
from .phase5_inputs import annotation_views, episode_references
from .phase5_policy import (
    actionize_clusters,
    cluster_proposals,
    derive_fusion_context,
    detector_created_hard_actions,
    frequency_matched_control,
    full_fusion_replay,
    fuse_actions,
)
from .phase5_proposals import (
    adjacent_trace,
    anchor_trace,
    generate_proposal_trace,
    source_prefix_routes,
)
from .phase5_scoring import score_policy_episode
from .phase5_storage import (
    RepresentationWriter,
    canonical_json as storage_canonical_json,
    deterministic_gzip,
    framed_digest,
    phase5_cache_root,
    read_representation,
    rows_sha256 as storage_rows_sha256,
    sha256_bytes as storage_sha256_bytes,
    sha256_file as storage_sha256_file,
    verify_shard_receipts,
)
from .schemas import ReferenceAction

AUTHORITY_SHA256 = "e3efdd9410a84bd343da5ba41d634ceec2d54626e1b512f41e410c0668329e36"
OWNER_OVERRIDE_MARKER = "### Owner override for Phase 5 execution (2026-08-11)"
PHASE5_CACHE_SCHEMA = "turn_episode_phase5_cache_contract.v1"
PHASE5_B0_SCHEMA = "turn_episode_phase5_b0_evidence.v1"
STAGE_A_RECEIPT_NAME = "phase_5_stage_a_receipt.json"
INTERSTAGE_GATE_NAME = "phase_5_interstage_gate.json"
STAGE_B_RECEIPT_NAME = "phase_5_stage_b_receipt.json"
PHYSICAL_SYSTEMS = 2503
LOGICAL_SYSTEMS = 4611
LOGICAL_ALIAS_EDGES = 2108
CURRENT_AGGREGATES = 4611
HISTORICAL_AGGREGATES = 4610
FAILURE_EXAMPLES = 420
HISTORICAL_CASES = 204
HISTORICAL_NEURAL_SYSTEMS = 4608
HISTORICAL_BASELINES = 2
POOL_ORDER = ("diagnostic_dev", "frontier_dev", "natural_exposure_validation")
CONTROL_KINDS = (
    "uniform_vad_active",
    "causal_energy_change_peak",
    "within_vad_active_position_shuffle",
)
LADDER_STAGES = (
    "naive_proposal_as_cut",
    "clustering_only",
    "clustering_plus_refractory",
    "plus_vad_association",
    "full_hard_soft_fusion",
)
CLUSTER_DEBOUNCE_MS = (0, 100, 250)
CLUSTER_RADIUS_MS = (250, 500)
REFRACTORY_MS = (0, 250, 500)
REPRESENTATIVES = ("first", "max_confidence")
VAD_RADIUS_MS = (250, 500)
SILENCE_ASSOCIATION = (False, True)
SYSTEM_METRIC_FIELDS = (
    "episode_count",
    "clean_gap_episode_count",
    "clean_gap_singleton_denominator_samples",
    "clean_gap_contaminated_samples",
    "mixed_turn_100ms_count",
    "mixed_turn_250ms_count",
    "mixed_turn_500ms_count",
    "clean_gap_hard_target_count",
    "hard_match_count",
    "hard_miss_count",
    "retained_b0_success_count",
    "recovered_b0_hard_miss_count",
    "accelerated_b0_success_count",
    "late_target_action_count",
    "detector_created_hard_action_count",
    "harmful_active_split_count_100ms",
    "harmful_active_split_count_200ms",
    "harmful_active_split_count_300ms",
    "lexical_split_count",
    "lexical_not_observable_count",
    "duplicate_hard_boundary_count",
    "same_speaker_pause_split_count",
    "same_speaker_extra_turn_count",
    "overlap_hard_action_count",
    "unscored_action_count",
    "fragments_lt_250ms_count",
    "fragments_lt_500ms_count",
    "fragments_lt_1000ms_count",
    "segment_duration_p10_samples",
    "segment_duration_p50_samples",
    "segment_duration_p90_samples",
    "active_speech_duration_p10_samples",
    "active_speech_duration_p50_samples",
    "active_speech_duration_p90_samples",
    "availability_delay_sum_samples",
    "availability_delay_count",
    "localization_error_sum_samples",
    "localization_error_count",
    "control_infeasible_count",
    "overlap_counterfactual_actual_samples_50ms",
    "overlap_counterfactual_suppressed_samples_50ms",
    "overlap_counterfactual_actual_samples_100ms",
    "overlap_counterfactual_suppressed_samples_100ms",
    "overlap_counterfactual_actual_samples_200ms",
    "overlap_counterfactual_suppressed_samples_200ms",
    "b0_b1_mismatch_count",
    "natural_contamination_numerator_samples",
    "natural_contamination_denominator_samples",
    "natural_harmful_active_split_count",
    "natural_same_speaker_extra_turn_count",
    "natural_sampled_source_samples",
    "natural_sampled_active_speech_samples",
    "natural_eligible_source_samples",
    "natural_session_count",
)
BLOCK_METRIC_FIELDS = (
    "episode_count",
    "clean_gap_singleton_denominator_samples",
    "candidate_clean_gap_contaminated_samples",
    "b0_clean_gap_contaminated_samples",
    "b1_clean_gap_contaminated_samples",
    "candidate_harmful_active_split_count",
    "b0_harmful_active_split_count",
    "b1_harmful_active_split_count",
    "detector_created_hard_action_count",
    "same_speaker_extra_turn_count",
    "lexical_split_count",
    "lexical_observable_action_count",
    "duplicate_hard_boundary_count",
    "hard_target_count",
    "hard_match_250ms_count",
    "hard_match_500ms_count",
    "hard_match_1000ms_count",
    "hard_match_1500ms_count",
    "hard_match_2000ms_count",
    "availability_delay_sum_samples",
    "availability_delay_count",
    "overlap_hard_action_count",
    "overlap_contribution_samples",
    "sampled_source_samples",
    "sampled_active_speech_samples",
    "natural_exposure_eligible_source_samples",
)
FAILURE_CATEGORIES = (
    "contamination_regression",
    "contamination_improvement",
    "harmful_active_split",
    "duplicate_cluster",
    "late_accurate_target",
    "clean_gap_miss_strong_evidence",
    "overlap_hard_action",
)
HISTORICAL_WORD_FILE_SHA256 = {
    "ami_ES2003a": {
        "ES2003a.A.words.xml": "8ee1e2ba5ab4421e16ebf42d0e221247a83e174fe34aa06e3aadd7a08540e470",
        "ES2003a.B.words.xml": "d9af4424014f9e2ae55e1a33fc7fb95a2352903453e3afb51118845cea4c304d",
        "ES2003a.C.words.xml": "9393f7af2a0cd95da35d306e4861e2513544c54059b11515ff311cd9b051cff4",
        "ES2003a.D.words.xml": "bdcf987020238618bed2941bc906d31aa61aec6732e3104d16f11b7f1a104384",
    },
    "ami_IS1008a": {
        "IS1008a.A.words.xml": "db744e8ca1f45f794e288c5521ab5dfbc2b2f8e5e9f814dd5300053cdab09a2d",
        "IS1008a.B.words.xml": "4a92c5b967c6a32fdfc4a288b7a61064d7259710b6afb32c27e49e4bd146f322",
        "IS1008a.C.words.xml": "d61e501c37877f731058889d799e7ff5a3d19494b8a22f12c7f0f50a87c62758",
        "IS1008a.D.words.xml": "e7f475b606a950de682d275b912821f3d3d612136ce62bd41bc685146767f40b",
    },
}


class Phase5RunError(RuntimeError):
    pass


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_eres_root() -> Path:
    return Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "opencode" / "eres_onnx"


def prd_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "specs"
        / "prd"
        / "bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md"
    )


def result_dir(experiment_dir: Path) -> Path:
    return experiment_dir / "results" / "turn_episode_v1"


def temp_stage_dir() -> Path:
    return phase5_cache_root() / "stage"


def proposal_trace_dir() -> Path:
    return phase5_cache_root() / "proposal_traces"


def b0_trace_dir() -> Path:
    return phase5_cache_root() / "b0_traces"


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Start guard
# ---------------------------------------------------------------------------


def verify_start_guard(experiment_dir: Path) -> dict[str, Any]:
    ledger_path = result_dir(experiment_dir) / "phase_5_design_ledger.json"
    if not ledger_path.is_file():
        raise Phase5RunError("phase 5 design ledger missing")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    verify_self_hash(ledger)
    if str(ledger["authority_sha256"]) != AUTHORITY_SHA256:
        raise Phase5RunError("design ledger authority pin drift")
    actual_prd = sha256_file(prd_path())
    if actual_prd != AUTHORITY_SHA256:
        raise Phase5RunError(
            f"PRD authority pin drift: expected {AUTHORITY_SHA256[:16]}... got {actual_prd[:16]}..."
        )
    text = prd_path().read_text(encoding="utf-8")
    if OWNER_OVERRIDE_MARKER not in text:
        raise Phase5RunError("PRD owner override record missing")
    if ledger["population"]["confirmatory_heldout_episode_count"] != 0:
        raise Phase5RunError("held-out episodes resolved")
    if ledger["population"]["heldout_paths_resolved"]:
        raise Phase5RunError("held-out paths resolved")
    profiles = ledger["family_compute_envelopes"]["eres2netv2"]["proposal_profiles"]
    if [str(row["proposal_profile_id"]) for row in profiles] != [
        "phase4_native:adjacent_direct:E-standard:W8000:S1600:T500",
        "phase4_native:adjacent_direct:E-standard:W8000:S4000:T500",
        "phase4_native:prototype_memory_4:E-standard:W8000:S1600:T500",
        "phase4_native:prototype_memory_4:E-standard:W8000:S4000:T500",
    ]:
        raise Phase5RunError("compact four-profile allowlist drift")
    if ledger["family_compute_envelopes"]["ls_eend"]["new_neural_inference"]:
        raise Phase5RunError("LS neural inference not allowed")
    if ledger["family_compute_envelopes"]["eres2netv2"]["w24_phase5_inference_or_replay"]:
        raise Phase5RunError("W24 inference/replay not allowed")
    if ledger["historical_correction"]["legacy_eres_profile_replay_count"] != 0:
        raise Phase5RunError("legacy ERes profile replay not allowed")
    guard = {
        "prd_sha256": actual_prd,
        "design_ledger_content_sha256": ledger["content_sha256"],
        "design_ledger_byte_sha256": sha256_file(ledger_path),
        "authority_consistent": True,
        "owner_override_recorded": True,
        "heldout_paths_resolved": False,
        "compact_profile_ids": [
            str(row["proposal_profile_id"]) for row in profiles
        ],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "network": "forbidden",
        "credentials": "forbidden",
        "provider_cost_usd": 0,
    }
    return guard


def verify_benchmark_inputs(experiment_dir: Path) -> dict[str, Any]:
    from .phase5_design import policy_benchmark, storage_benchmark

    result = result_dir(experiment_dir)
    policy = policy_benchmark(result)
    storage = storage_benchmark(result)
    return {
        "policy_benchmark_content_sha256": policy["content_sha256"],
        "storage_benchmark_content_sha256": storage["content_sha256"],
        "policy_workers": int(policy["selected_policy_workers"]),
        "projected_result_bytes": storage["projected_result_bytes"],
        "within_result_ceiling": bool(storage["within_result_ceiling"]),
    }


# ---------------------------------------------------------------------------
# B0 replay with lifecycle capture
# ---------------------------------------------------------------------------


def _capture_lifecycle(
    replay: Any,
    events: Sequence[Any],
    lifecycle: list[dict[str, Any]],
    chunk_index: int,
    chunk_samples: int,
    sample_rate_hz: int,
    current_utterance_id: str | None,
) -> None:
    start_sample = chunk_index * chunk_samples
    for event in events:
        kind = type(event).__name__
        if kind == "SpeechStart":
            utterance_id = str(getattr(event, "utterance_id"))
            lifecycle.append(
                {
                    "event_kind": "speech_start",
                    "normalized_utterance_id": utterance_id,
                    "event_source_sample": start_sample,
                    "observed_source_sample_at_emit": start_sample + chunk_samples,
                }
            )
            current_utterance_id = utterance_id
        elif kind == "SpeechEnd":
            trailing_silence_ms = int(getattr(event, "trailing_silence_ms", 0))
            reason = str(getattr(event, "reason", "silence"))
            chunk_ms = chunk_samples / sample_rate_hz * 1000.0
            silence_run = int(round(trailing_silence_ms / chunk_ms))
            end_sample = (chunk_index + 1 - silence_run) * chunk_samples
            lifecycle.append(
                {
                    "event_kind": "speech_end",
                    "normalized_utterance_id": str(getattr(event, "utterance_id")),
                    "event_source_sample": max(end_sample, 0),
                    "observed_source_sample_at_emit": start_sample + chunk_samples,
                    "reason": reason,
                }
            )
            current_utterance_id = None
    replay._current_utterance_id = current_utterance_id


class _CapturingReplay(VadBoundaryReplay):
    def __init__(self, model_path: Path) -> None:
        super().__init__(
            engine_factory=lambda: SileroVadOnnx(model_path),
        )
        self.lifecycle: list[dict[str, Any]] = []
        self._current_utterance_id: str | None = None

    def process_chunk(self, chunk: np.ndarray) -> None:
        if self._audio_epoch is None or self._gating is None:
            raise Phase5RunError("start_epoch must be called before process_chunk")
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size != self.chunk_samples:
            raise Phase5RunError("B0 replay chunk contract drift")
        events = self._gating.process_chunk(chunk)
        _capture_lifecycle(
            self,
            events,
            self.lifecycle,
            self._chunk_index,
            self.chunk_samples,
            self.sample_rate_hz,
            self._current_utterance_id,
        )
        boundaries = self._translate_events(events)
        self._boundaries.extend(boundaries)
        frontier_start = self._chunk_index * self.chunk_samples
        from experiments.speaker_turn_boundary.events import DetectorProgress

        self._progress.append(
            DetectorProgress(
                audio_epoch=self._audio_epoch,
                observed_source_sample=frontier_start + self.chunk_samples,
                safe_boundary_frontier_sample=frontier_start,
            )
        )
        self._chunk_index += 1
        return boundaries

    @property
    def boundaries(self) -> list[Any]:
        return self._boundaries

    @property
    def progress(self) -> list[Any]:
        return self._progress


def replay_b0_capture(wav_path: Path, model_path: Path) -> dict[str, Any]:
    from experiments.speaker_turn_boundary.vad_baseline import load_canonical_wav

    samples = load_canonical_wav(wav_path)
    replay = _CapturingReplay(model_path)
    replay.start_epoch(0)
    offset = 0
    while offset < samples.size:
        chunk = samples[offset : offset + 512]
        if chunk.size < 512:
            break
        replay.process_chunk(chunk)
        offset += 512
    active_id = replay._current_utterance_id
    if active_id is not None:
        replay.lifecycle.append(
            {
                "event_kind": "terminal",
                "normalized_utterance_id": active_id,
                "event_source_sample": samples.size,
                "observed_source_sample_at_emit": samples.size,
                "active_state_remained": True,
            }
        )
    return {
        "length_samples": samples.size,
        "boundaries": [row.to_dict() for row in replay.boundaries],
        "progress": [row.to_dict() for row in replay.progress],
        "lifecycle": replay.lifecycle,
    }


def _b0_projection(boundary: dict[str, Any]) -> dict[str, Any]:
    return {
        "audio_epoch": int(boundary["audio_epoch"]),
        "boundary_source_sample": int(boundary["boundary_source_sample"]),
        "observed_source_sample_at_emit": int(boundary["observed_source_sample_at_emit"]),
        "confidence": boundary.get("confidence"),
        "source": str(boundary["source"]),
        "debug": dict(sorted((boundary.get("debug") or {}).items(), key=lambda item: item[0])),
    }


def run_b0_traces(
    experiment_dir: Path,
    sources: dict[str, AudioSource],
    source_by_episode: dict[str, str],
) -> dict[str, Any]:
    model_path = Path(str(bundled_silero_vad_onnx_path()))
    actual = sha256_file(model_path)
    if actual != SILERO_MODEL_SHA256:
        raise Phase5RunError(
            f"Silero model hash mismatch: expected {SILERO_MODEL_SHA256}, got {actual}"
        )
    inventory = json.loads(
        (result_dir(experiment_dir) / "phase_4_cache_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    b0_replay_dir = result_dir(experiment_dir) / "b0_inventory_replay"
    traces: dict[str, dict[str, Any]] = {}
    accepted_hits = 0
    synthetic_runs = 0
    started = time.perf_counter()
    for source_id, source in sorted(sources.items()):
        accepted_path = b0_replay_dir / f"{source_id}.json"
        if accepted_path.is_file():
            accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
            trace = replay_b0_capture(source.path, model_path)
            projected = [_b0_projection(row) for row in trace["boundaries"]]
            accepted_projected = list(accepted["trace_projection"])
            if sha256_bytes(b0_canonical_json(projected).encode("utf-8")) != accepted["trace_hash"]:
                raise Phase5RunError(f"B0 replay mismatch against accepted trace: {source_id}")
            if projected != accepted_projected:
                raise Phase5RunError(f"B0 replay identity drift: {source_id}")
            accepted_hits += 1
            traces[source_id] = {
                "session_id": source_id,
                "length_samples": accepted["length_samples"],
                "trace_hash": accepted["trace_hash"],
                "trace_projection": projected,
                "lifecycle": trace["lifecycle"],
                "b0_receipt_source": "accepted_b0_inventory_replay",
            }
        else:
            trace = replay_b0_capture(source.path, model_path)
            projected = [_b0_projection(row) for row in trace["boundaries"]]
            traces[source_id] = {
                "session_id": source_id,
                "length_samples": trace["length_samples"],
                "trace_hash": sha256_bytes(b0_canonical_json(projected).encode("utf-8")),
                "trace_projection": projected,
                "lifecycle": trace["lifecycle"],
                "b0_receipt_source": "phase5_synthetic_b0_replay",
            }
            synthetic_runs += 1
    elapsed = time.perf_counter() - started
    return {
        "source_count": len(traces),
        "accepted_public_hit_count": accepted_hits,
        "synthetic_replay_count": synthetic_runs,
        "elapsed_seconds": round(elapsed, 3),
        "traces": traces,
    }


def project_b0_episode(
    trace: dict[str, Any],
    episode: dict[str, Any],
    *,
    case_mode: bool = False,
) -> list[dict[str, Any]]:
    bounds = episode["bounds"]
    if case_mode:
        warm_start = 0
        tail_end = int(episode["duration_samples"])
    else:
        warm_start = int(bounds["warm_start"])
        tail_end = int(bounds["tail_end"])
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(trace["trace_projection"]):
        boundary = int(row["boundary_source_sample"])
        if boundary < warm_start or boundary > tail_end:
            continue
        session_id = str(episode["session_id"]) if not case_mode else str(episode["case_id"])
        action_id = (
            f"vad:{session_id}:{index}" if not case_mode else f"historical-b0:{session_id}:{index}"
        )
        rows.append(
            {
                "action_id": action_id,
                "event_id": action_id,
                "audio_epoch": int(episode["audio_epoch"]) if not case_mode else 0,
                "source_session_id": session_id,
                "boundary_source_sample": boundary,
                "observed_source_sample_at_emit": int(
                    row["observed_source_sample_at_emit"]
                ),
                "action_kind": "retain_vad",
                "origin": "vad",
                "owner": "vad",
                "debug": row.get("debug") or {},
            }
        )
    return rows


def b0_lifecycle_rows(
    trace: dict[str, Any],
    episode: dict[str, Any],
    *,
    case_mode: bool = False,
) -> list[dict[str, Any]]:
    if case_mode:
        return []
    bounds = episode["bounds"]
    warm_start = int(bounds["warm_start"])
    tail_end = int(bounds["tail_end"])
    rows: list[dict[str, Any]] = []
    for row in trace["lifecycle"]:
        observed = int(row["observed_source_sample_at_emit"])
        if observed < warm_start or observed > tail_end:
            continue
        rows.append(
            {
                **row,
                "audio_epoch": int(episode["audio_epoch"]),
                "source_session_id": str(episode["session_id"]),
            }
        )
    return rows

# ---------------------------------------------------------------------------
# Phase 5 cache contract, phase 4 validation, inference
# ---------------------------------------------------------------------------


def phase5_cache_contract(
    experiment_dir: Path,
    ledger: dict[str, Any],
    eres_root: Path,
) -> tuple[dict[str, Any], Path, EresEmbeddingRuntime]:
    model = eres_root / str(ERES_CHECKPOINTS["E-standard"]["onnx"])
    if sha256_file(model) != ERES_MODEL_SHA256["E-standard"]:
        raise Phase5RunError("E-standard ONNX model hash drift")
    runtime = EresEmbeddingRuntime(str(model))
    frontend_path = experiment_dir / "frontend.py"
    adapter_path = experiment_dir / "adapters" / "eres2netv2.py"
    body = {
        "schema_version": PHASE5_CACHE_SCHEMA,
        "authority_sha256": AUTHORITY_SHA256,
        "phase_5_design_ledger_content_sha256": ledger["content_sha256"],
        "checkpoint": "E-standard",
        "checkpoint_sha256": ERES_MODEL_SHA256["E-standard"],
        "frontend_sha256": sha256_file(frontend_path),
        "eres_adapter_sha256": sha256_file(adapter_path),
        "state_mode": "absolute_exact_window_source_prefix_state_replay",
        "source_origin": "absolute_16khz_sample_zero",
        "tensor_contract": tensor_contract(runtime._session),
    }
    body["tensor_contract_sha256"] = content_hash(body["tensor_contract"])
    body["contract_sha256"] = content_hash(body)
    return body, model, runtime


def _validate_phase4_cache_inventory(
    inventory: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = inventory["eres"]["E-standard"]["sources"]
    checked: list[dict[str, Any]] = []
    for row in rows:
        metadata_path = Path(str(row["metadata_path"]))
        if not metadata_path.is_file():
            raise Phase5RunError(f"phase4 cache metadata missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        shards = metadata.get("shards")
        if not isinstance(shards, list) or not shards:
            raise Phase5RunError(f"phase4 cache shard inventory missing: {row['source_id']}")
        if content_hash(
            {
                "shards": [
                    {key: value for key, value in item.items() if key != "path"}
                    for item in shards
                ]
            }
        ) != metadata.get("payload_sha256"):
            raise Phase5RunError(f"phase4 cache payload hash mismatch: {row['source_id']}")
        for shard in shards:
            path = metadata_path.parent / str(shard["path"])
            if not path.is_file():
                raise Phase5RunError(f"phase4 cache shard missing: {path}")
            if path.stat().st_size != int(shard["size_bytes"]):
                raise Phase5RunError(f"phase4 cache shard size drift: {path}")
            if sha256_file(path) != shard["byte_sha256"]:
                raise Phase5RunError(f"phase4 cache shard byte hash drift: {path}")
        checked.append(row)
    return checked


def load_phase4_embeddings(
    inventory: dict[str, Any],
    needed_by_wav: dict[str, set[tuple[int, int]]],
    historical_development: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[tuple[int, int], np.ndarray]], dict[str, Any]]:
    from .phase4_signal import _decode_eres_shard

    sources = inventory["eres"]["E-standard"]["sources"]
    by_wav: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    reusable_windows = 0
    total_rows = 0
    for row in sources:
        wav = str(row["wav_sha256"])
        needed = needed_by_wav.get(wav)
        if not needed:
            continue
        metadata_path = Path(str(row["metadata_path"]))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        found: dict[tuple[int, int], np.ndarray] = {}
        for shard in metadata["shards"]:
            path = metadata_path.parent / str(shard["path"])
            compressed = path.read_bytes()
            if sha256_bytes(compressed) != shard["byte_sha256"]:
                raise Phase5RunError(f"phase4 cache shard hash drift: {path}")
            plain = gzip.decompress(compressed)
            if sha256_bytes(plain) != shard["content_sha256"]:
                raise Phase5RunError(f"phase4 cache shard content drift: {path}")
            windows, embeddings, _, _ = _decode_eres_shard(plain)
            for window, embedding in zip(windows.tolist(), embeddings):
                key = (int(window[0]), int(window[1]))
                if key in needed:
                    found[key] = np.asarray(embedding, dtype=np.float32)
        by_wav[wav] = found
        reusable_windows += len(found)
        total_rows += len(found)
    if historical_development is not None:
        cache_root = Path(str(historical_development["cache_root"]))
        for receipt in historical_development["cache_receipt_rows"]:
            metadata_path = cache_root / str(receipt["metadata_relative_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            wav = str(metadata["wav_sha256"])
            needed = needed_by_wav.get(wav)
            if not needed:
                continue
            existing = by_wav.get(wav)
            missing = needed - set(existing) if existing else needed
            if not missing:
                continue
            npz_path = cache_root / str(receipt["npz_relative_path"])
            if sha256_file(npz_path) != receipt["npz_byte_sha256"]:
                raise Phase5RunError(f"historical cache npz hash drift: {npz_path}")
            with np.load(npz_path) as data:
                merged = dict(existing) if existing else {}
                for start, end in missing:
                    key = f"{start}-{end}"
                    if key in data:
                        merged[(start, end)] = np.asarray(
                            data[key], dtype=np.float32
                        ).reshape(-1)
            added = len(merged) - (len(existing) if existing else 0)
            by_wav[wav] = merged
            reusable_windows += added
            total_rows += added
    return by_wav, {
        "reusable_window_count": reusable_windows,
        "imported_window_row_count": total_rows,
    }


def infer_missing_windows(
    experiment_dir: Path,
    ledger: dict[str, Any],
    missing_by_wav: dict[str, set[tuple[int, int]]],
    sources_by_wav: dict[str, AudioSource],
    eres_root: Path,
) -> dict[str, Any]:
    contract, model, runtime = phase5_cache_contract(experiment_dir, ledger, eres_root)
    cache_root = phase5_cache_root()
    rows: list[dict[str, Any]] = []
    pending: list[tuple[AudioSource, list[tuple[int, int]]]] = []
    for wav, windows in sorted(missing_by_wav.items()):
        source = sources_by_wav.get(wav)
        if source is None:
            raise Phase5RunError(f"missing window wav has no source: {wav}")
        ordered = sorted(windows)
        cached = load_eres_embeddings(
            cache_root,
            contract,
            "E-standard",
            source,
            ordered,
        )
        if cached is not None:
            _, _, evidence = cached
            rows.append({**evidence, "cache_hit": True})
            continue
        pending.append((source, ordered))

    def execute_partition(
        worker_index: int,
        partition: list[tuple[AudioSource, list[tuple[int, int]]]],
    ) -> list[dict[str, Any]]:
        worker_runtime = EresEmbeddingRuntime(str(model))
        completed: list[dict[str, Any]] = []
        processed = 0
        for source, windows in partition:
            samples = read_wav(source)
            embeddings: list[np.ndarray] = []
            acoustic_shadows: list[np.ndarray] = []
            acoustic_log_rms: list[float] = []
            service: list[float] = []
            for start, end in windows:
                begin = time.perf_counter()
                window_samples = samples[start:end]
                fbank = kaldi_fbank_numpy(window_samples)
                shadow = None
                if fbank.size:
                    flat = fbank.mean(axis=0)
                    norm = float(np.linalg.norm(flat))
                    shadow = flat / norm if norm > 0 else None
                if shadow is None:
                    raise Phase5RunError(
                        f"invalid ERes acoustic shadow E-standard:{source.source_id}:{start}-{end}"
                    )
                centered = fbank - fbank.mean(axis=0, keepdims=True)
                output = worker_runtime._session.run(
                    worker_runtime._output_names,
                    {"fbank": centered[None, :, :].astype(np.float32)},
                )[0]
                vector = np.asarray(output[0], dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vector))
                if vector.size != 192 or not np.isfinite(vector).all() or norm == 0.0:
                    raise Phase5RunError(
                        f"invalid ERes embedding E-standard:{source.source_id}:{start}-{end}"
                    )
                vector = vector / norm
                service.append(time.perf_counter() - begin)
                embeddings.append(vector)
                acoustic_shadows.append(shadow)
                rms = float(np.sqrt(np.mean(np.square(window_samples, dtype=np.float64))))
                acoustic_log_rms.append(math.log(max(rms, 1e-8)))
                processed += 1
                if processed % 5000 == 0:
                    print(
                        f"phase5 eres inference worker={worker_index} windows={processed}",
                        flush=True,
                    )
            matrix = (
                np.stack(embeddings).astype(np.float32, copy=False)
                if embeddings
                else np.zeros((0, 192), dtype=np.float32)
            )
            shadow_matrix = (
                np.stack(acoustic_shadows).astype(np.float32, copy=False)
                if acoustic_shadows
                else np.zeros((0, 80), dtype=np.float32)
            )
            evidence = save_eres_embeddings(
                cache_root,
                contract,
                "E-standard",
                source,
                windows,
                matrix,
                shadow_matrix,
                np.asarray(acoustic_log_rms, dtype=np.float32),
                service,
            )
            completed.append({**evidence, "cache_hit": False})
        return completed

    if pending:
        gc.collect()
        partition_count = min(10, len(pending))
        partitions: list[list[tuple[AudioSource, list[tuple[int, int]]]]] = [
            [] for _ in range(partition_count)
        ]
        partition_sizes = [0] * partition_count
        for item in sorted(pending, key=lambda value: (-len(value[1]), value[0].source_id)):
            index = min(range(partition_count), key=lambda value: (partition_sizes[value], value))
            partitions[index].append(item)
            partition_sizes[index] += len(item[1])
        with ThreadPoolExecutor(max_workers=partition_count) as executor:
            futures = {
                executor.submit(execute_partition, index, partition): index
                for index, partition in enumerate(partitions)
            }
            for future in as_completed(futures):
                rows.extend(future.result())
    rows.sort(key=lambda row: str(row["source_id"]))
    return {
        "contract": contract,
        "contract_sha256": contract["contract_sha256"],
        "cache_root": str(cache_root),
        "source_count": len(rows),
        "cache_hit_count": sum(bool(row["cache_hit"]) for row in rows),
        "new_inference_window_count": sum(int(row["window_count"]) for row in rows),
        "payload_bytes": sum(int(row["payload_size_bytes"]) for row in rows),
        "source_receipts": rows,
    }


def load_phase5_embeddings(
    contract: dict[str, Any],
    missing_by_wav: dict[str, set[tuple[int, int]]],
    sources_by_wav: dict[str, AudioSource],
) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    cache_root = phase5_cache_root()
    result: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for wav, windows in sorted(missing_by_wav.items()):
        source = sources_by_wav[wav]
        ordered = sorted(windows)
        if not ordered:
            result[wav] = {}
            continue
        cached = load_eres_embeddings(cache_root, contract, "E-standard", source, ordered)
        if cached is None:
            raise Phase5RunError(f"phase5 cache window missing: {source.source_id}")
        embeddings, _, _ = cached
        result[wav] = embeddings
    return result

# ---------------------------------------------------------------------------
# Annotation views and word timing receipts
# ---------------------------------------------------------------------------


def load_raw_words(corpus_root: Path, session_id: str) -> tuple[list[Any] | None, list[dict[str, Any]], str]:
    if not session_id.startswith("ami_"):
        return None, [], ""
    meeting_id = session_id.removeprefix("ami_")
    words_dir = corpus_root / "ami" / "annotations" / "words"
    paths = sorted(words_dir.glob(f"{meeting_id}.*.words.xml"))
    if not paths:
        return None, [], ""
    receipts: list[dict[str, Any]] = []
    for path in paths:
        receipts.append({"filename": path.name, "byte_sha256": sha256_file(path)})
    raw_words = load_ami_raw_words(words_dir, meeting_id)
    record_rows = [
        [
            row.speaker,
            row.start_time_s,
            row.end_time_s,
            row.text,
            row.ambiguous,
            row.path_index,
        ]
        for row in raw_words
    ]
    record_sha256 = hashlib.sha256(canonical_json(record_rows).encode("utf-8")).hexdigest()
    return raw_words, receipts, record_sha256


def annotation_views_for_episode(
    episode: dict[str, Any],
    regions_by_session: dict[str, list[Any]],
    cases: dict[tuple[str, str], dict[str, Any]],
    raw_words_by_session: dict[str, list[Any] | None],
) -> dict[str, Any]:
    session_id = str(episode["session_id"])
    manifest_name = synthetic_manifest_name(session_id)
    if manifest_name is None:
        regions = regions_by_session.get(session_id) or []
    else:
        case = cases.get((manifest_name, synthetic_case_id(session_id)))
        regions = list(case["regions"]) if case is not None else []
    references = episode_references(episode)
    raw_words = raw_words_by_session.get(session_id)
    return annotation_views(
        regions,
        references,
        raw_words,
        scored_start=int(episode["bounds"]["scored_start"]),
        scored_end=int(episode["bounds"]["scored_end"]),
    )


def annotation_views_for_case(
    case: dict[str, Any],
    raw_words: list[Any] | None,
) -> dict[str, Any]:
    from .build_episodes import build_reference_specs, references_for_episode
    from ..ground_truth import SpeakerRegion

    regions = [SpeakerRegion.from_dict(row) for row in case["regions"]]
    specs = build_reference_specs(
        regions, int(case["duration_samples"]), raw_words
    )
    references = references_for_episode(
        specs,
        str(case["case_id"]),
        "full_session_scoring_benchmark",
        0,
        int(case["duration_samples"]),
        "overlap_present",
        True,
    )
    views = annotation_views(
        regions,
        references,
        raw_words,
        scored_start=0,
        scored_end=int(case["duration_samples"]),
    )
    views["_references"] = [
        ReferenceAction.to_dict(reference)
        for reference in references
    ]
    return views


def word_timing_receipt(
    unit_id: str,
    annotation_source_identity: dict[str, Any],
    views: dict[str, Any],
    raw_words: list[Any] | None,
    word_record_sha256: str,
) -> dict[str, Any]:
    observable = bool(views["word_timing_observable"])
    return {
        "unit_id": unit_id,
        "annotation_source_identity": annotation_source_identity,
        "word_record_sha256": word_record_sha256,
        "raw_word_record_count": len(raw_words) if raw_words is not None else 0,
        "word_interval_count": len(views["word_intervals"]) if observable else 0,
        "word_timing_observable": observable,
        "lexical_scoring_disposition": (
            "scored_with_trusted_word_timing"
            if observable
            else "unscored_missing_word_timing"
        ),
    }


def expected_word_timing_receipts(
    experiment_dir: Path,
    episodes: list[dict[str, Any]],
    historical_case_rows: list[dict[str, Any]],
    details: dict[str, dict[str, Any]],
    cases: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    manifests_dir = experiment_dir / "data" / "manifests"
    corpus_root = external.corpus_root()
    raw_words_by_session: dict[str, list[Any] | None] = {}
    word_receipts_by_session: dict[str, list[dict[str, Any]]] = {}
    session_ids = sorted({str(row["session_id"]) for row in episodes})
    for session_id in session_ids:
        raw_words, receipts, _ = load_raw_words(corpus_root, session_id)
        raw_words_by_session[session_id] = raw_words
        word_receipts_by_session[session_id] = receipts
    inventory = json.loads(
        (result_dir(experiment_dir) / "coverage_inventory.json").read_text(encoding="utf-8")
    )
    public_sessions = [
        session_id
        for session_id in session_ids
        if synthetic_manifest_name(session_id) is None
    ]
    regions_by_session = load_public_regions(
        inventory, details, public_sessions, manifests_dir
    )
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        session_id = str(episode["session_id"])
        raw_words = raw_words_by_session[session_id]
        views = annotation_views_for_episode(
            episode, regions_by_session, cases, raw_words_by_session
        )
        identity = {
            "session_id": session_id,
            "annotation_sha256": str(
                details.get(session_id, {}).get("annotation_sha256", episode.get("annotation_sha256") or "")
            ),
            "word_annotation_files": word_receipts_by_session[session_id],
        }
        record_sha256 = ""
        if raw_words is not None:
            _, _, record_sha256 = load_raw_words(corpus_root, session_id)
        rows.append(
            word_timing_receipt(
                str(episode["episode_id"]),
                identity,
                views,
                raw_words,
                record_sha256,
            )
        )
    case_rows: list[dict[str, Any]] = []
    manifest = json.loads(
        (experiment_dir / "data" / "manifests" / "mixed_dev_pool.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_cases = {str(row["case_id"]): row for row in manifest["cases"]}
    for case in historical_case_rows:
        case_id = str(case["case_id"])
        raw_words = None
        receipts: list[dict[str, Any]] = []
        record_sha256 = ""
        if case_id in HISTORICAL_WORD_FILE_SHA256:
            raw_words, receipts, record_sha256 = load_raw_words(corpus_root, case_id)
        manifest_case = manifest_cases.get(case_id)
        if manifest_case is None:
            raise Phase5RunError(f"historical word timing case missing: {case_id}")
        views = annotation_views_for_case(manifest_case, raw_words)
        case_rows.append(
            word_timing_receipt(
                case_id,
                {
                    "manifest_byte_sha256": HISTORICAL_MANIFEST_BYTE_SHA256,
                    "word_annotation_files": receipts,
                },
                views,
                raw_words,
                record_sha256,
            )
        )
    receipts = sorted(rows + case_rows, key=lambda row: str(row["unit_id"]))
    return {
        "current_receipt_count": len(rows),
        "historical_receipt_count": len(case_rows),
        "receipts": receipts,
    }


# ---------------------------------------------------------------------------
# Proposal traces
# ---------------------------------------------------------------------------


def proposal_trace_store_path() -> Path:
    return proposal_trace_dir() / "traces.jsonl.gz"


def save_proposal_traces(rows: Iterable[dict[str, Any]]) -> None:
    payload = b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in rows)
    atomic_write_bytes(proposal_trace_store_path(), deterministic_gzip(payload))


def load_proposal_traces() -> dict[str, dict[str, Any]]:
    path = proposal_trace_store_path()
    if not path.is_file():
        raise Phase5RunError("proposal trace store missing; run stage_a first")
    plain = __import__("gzip").decompress(path.read_bytes())
    rows: dict[str, dict[str, Any]] = {}
    for line in plain.splitlines():
        if not line:
            continue
        row = json.loads(line)
        rows[str(row["execution_id"])] = row
    return rows


def build_current_proposal_traces(
    embeddings_by_wav: dict[str, dict[tuple[int, int], np.ndarray]],
    profiles: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    sources_by_wav: dict[str, AudioSource],
    cache_receipt_hash: str,
    lifecycle_hash: str,
) -> tuple[list[dict[str, Any]], list[list[Any]], list[dict[str, Any]]]:
    proposal_code = sha256_file(Path(__file__).with_name("phase5_proposals.py"))
    receipts: list[dict[str, Any]] = []
    routes: list[list[Any]] = []
    traces: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_source[str(episode["session_id"])].append(episode)
    for profile in profiles:
        profile_id = str(profile["proposal_profile_id"])
        if profile["scored_state_mode"] == "source_prefix":
            for source_id in sorted(by_source):
                group = by_source[source_id]
                wav = str(group[0]["wav_sha256"])
                embeddings = embeddings_by_wav[wav]
                result = source_prefix_routes(embeddings, profile, group)
                execution_id = str(result["source_prefix_execution_id"])
                window_rows = [
                    {"start": start, "end": end}
                    for start, end in sorted(embeddings)
                ]
                proposal_count = sum(int(row["proposal_count"]) for row in result["routes"])
                all_proposals = [
                    row
                    for route in result["routes"]
                    for row in route["proposals"]
                ]
                trace_sha = content_sha256(all_proposals)
                progress_sha = content_sha256(
                    [
                        row
                        for route in result["routes"]
                        for row in route["progress"]
                    ]
                )
                receipts.append(
                    {
                        "proposal_execution_id": execution_id,
                        "proposal_profile_id": profile_id,
                        "source_or_case_id": source_id,
                        "execution_mode": "source_prefix",
                        "probe_step_count": int(result["probe_step_count"]),
                        "input_window_universe_sha256": rows_sha256(window_rows),
                        "input_cache_receipts_sha256": cache_receipt_hash,
                        "lifecycle_trace_sha256": lifecycle_hash,
                        "proposal_code_sha256": proposal_code,
                        "proposal_count": proposal_count,
                        "proposal_trace_sha256": trace_sha,
                        "progress_count": sum(
                            len(row["progress"]) for row in result["routes"]
                        ),
                        "progress_trace_sha256": progress_sha,
                        "state_snapshot_index_sha256": str(result["route_index_sha256"]),
                        "final_state_sha256": "",
                        "tail_evidence_sha256": content_sha256(
                            [row["tail_evidence"] for row in result["routes"]]
                        ),
                        "status": "complete",
                    }
                )
                traces.append(
                    {
                        "execution_id": execution_id,
                        "mode": "source_prefix",
                        "proposal_profile_id": profile_id,
                        "source_or_case_id": source_id,
                        "routes": {
                            str(row["episode_id"]): {
                                key: row[key]
                                for key in (
                                    "episode_id",
                                    "audio_epoch",
                                    "proposals",
                                    "progress",
                                    "proposal_count",
                                    "proposal_trace_sha256",
                                    "progress_trace_sha256",
                                    "final_state_sha256",
                                    "tail_evidence",
                                )
                            }
                            for row in result["routes"]
                        },
                    }
                )
                for row in result["routes"]:
                    routes.append(
                        [
                            profile_id,
                            str(row["episode_id"]),
                            execution_id,
                            int(row["proposal_count"]),
                            str(row["proposal_trace_sha256"]),
                            len(row["progress"]),
                            str(row["progress_trace_sha256"]),
                            bool(row["tail_evidence"]["pending_confirmation_suppressed"]),
                        ]
                    )
        else:
            for episode in episodes:
                wav = str(episode["wav_sha256"])
                embeddings = embeddings_by_wav[wav]
                trace = generate_proposal_trace(embeddings, profile, episode)
                episode_id = str(episode["episode_id"])
                execution_id = f"episode-reset:{profile_id}:{episode_id}"
                window_rows = [
                    {"start": start, "end": end} for start, end in sorted(embeddings)
                ]
                receipts.append(
                    {
                        "proposal_execution_id": execution_id,
                        "proposal_profile_id": profile_id,
                        "source_or_case_id": str(episode["session_id"]),
                        "execution_mode": "episode_reset",
                        "probe_step_count": int(
                            len(
                                range(
                                    ceil_grid(
                                        int(episode["bounds"]["warm_start"])
                                        + int(profile["window_samples"]),
                                        int(profile["step_samples"]),
                                    ),
                                    int(episode["bounds"]["tail_end"])
                                    - int(profile["window_samples"])
                                    + 1,
                                    int(profile["step_samples"]),
                                )
                            )
                        ),
                        "input_window_universe_sha256": rows_sha256(window_rows),
                        "input_cache_receipts_sha256": cache_receipt_hash,
                        "lifecycle_trace_sha256": lifecycle_hash,
                        "proposal_code_sha256": proposal_code,
                        "proposal_count": int(trace["proposal_count"]),
                        "proposal_trace_sha256": str(trace["proposal_trace_sha256"]),
                        "progress_count": len(trace["progress"]),
                        "progress_trace_sha256": str(trace["progress_trace_sha256"]),
                        "state_snapshot_index_sha256": "",
                        "final_state_sha256": "",
                        "tail_evidence_sha256": content_sha256(trace["tail_evidence"]),
                        "status": "complete",
                    }
                )
                traces.append(
                    {
                        "execution_id": execution_id,
                        "mode": "episode_reset",
                        "proposal_profile_id": profile_id,
                        "source_or_case_id": str(episode["session_id"]),
                        "routes": {
                            episode_id: {
                                "episode_id": episode_id,
                                "audio_epoch": int(episode["audio_epoch"]),
                                "proposals": trace["proposals"],
                                "progress": trace["progress"],
                                "proposal_count": int(trace["proposal_count"]),
                                "proposal_trace_sha256": trace["proposal_trace_sha256"],
                                "progress_trace_sha256": trace["progress_trace_sha256"],
                                "final_state_sha256": "",
                                "tail_evidence": trace["tail_evidence"],
                            }
                        },
                    }
                )
                routes.append(
                    [
                        profile_id,
                        episode_id,
                        execution_id,
                        int(trace["proposal_count"]),
                        str(trace["proposal_trace_sha256"]),
                        len(trace["progress"]),
                        str(trace["progress_trace_sha256"]),
                        bool(trace["tail_evidence"]["pending_confirmation_suppressed"]),
                    ]
                )
    return receipts, routes, traces


def build_historical_proposal_traces(
    embeddings_by_wav: dict[str, dict[tuple[int, int], np.ndarray]],
    profiles: list[dict[str, Any]],
    historical_cases: list[dict[str, Any]],
    cache_receipt_hash: str,
) -> tuple[list[dict[str, Any]], list[list[Any]], list[dict[str, Any]]]:
    proposal_code = sha256_file(Path(__file__).with_name("phase5_proposals.py"))
    receipts: list[dict[str, Any]] = []
    routes: list[list[Any]] = []
    traces: list[dict[str, Any]] = []
    for profile in profiles:
        profile_id = str(profile["proposal_profile_id"])
        for case in historical_cases:
            case_id = str(case["case_id"])
            wav = str(case["wav_sha256"])
            embeddings = embeddings_by_wav.get(wav, {})
            if not embeddings:
                raise Phase5RunError(f"historical case embeddings missing: {case_id}")
            case_episode = {
                "session_id": case_id,
                "case_id": case_id,
                "audio_epoch": 0,
                "duration_samples": int(case["duration_samples"]),
                "bounds": {
                    "warm_start": 0,
                    "scored_start": 0,
                    "scored_end": int(case["duration_samples"]),
                    "tail_end": int(case["duration_samples"]),
                },
            }
            if profile["profile_class"] == "adjacent":
                trace = adjacent_trace(
                    embeddings,
                    profile,
                    source_session_id=case_id,
                    audio_epoch=0,
                    warm_start=0,
                    tail_end=int(case["duration_samples"]),
                )
            else:
                trace = anchor_trace(
                    embeddings,
                    profile,
                    source_session_id=case_id,
                    audio_epoch=0,
                    replay_start=0,
                    warm_start=0,
                    tail_end=int(case["duration_samples"]),
                )
            ordered = sorted(
                trace["proposals"],
                key=lambda row: (
                    int(row["observed_source_sample_at_emit"]),
                    int(row["boundary_source_sample"]),
                    str(row["profile_id"]),
                    str(row["proposal_id"]),
                ),
            )
            proposal_trace_sha = content_sha256(ordered)
            progress_sha = content_sha256(trace["progress"])
            execution_id = f"historical:{profile_id}:{case_id}"
            window_rows = [{"start": start, "end": end} for start, end in sorted(embeddings)]
            receipts.append(
                {
                    "proposal_execution_id": execution_id,
                    "proposal_profile_id": profile_id,
                    "source_or_case_id": case_id,
                    "execution_mode": "episode_reset",
                    "probe_step_count": int(
                        len(
                            range(
                                ceil_grid(
                                    int(profile["window_samples"]),
                                    int(profile["step_samples"]),
                                ),
                                int(case["duration_samples"])
                                - (
                                    int(profile["window_samples"])
                                    if profile["profile_class"] == "adjacent"
                                    else 0
                                )
                                + 1,
                                int(profile["step_samples"]),
                            )
                        )
                    ),
                    "input_window_universe_sha256": rows_sha256(window_rows),
                    "input_cache_receipts_sha256": cache_receipt_hash,
                    "lifecycle_trace_sha256": "",
                    "proposal_code_sha256": proposal_code,
                    "proposal_count": len(ordered),
                    "proposal_trace_sha256": proposal_trace_sha,
                    "progress_count": len(trace["progress"]),
                    "progress_trace_sha256": progress_sha,
                    "state_snapshot_index_sha256": "",
                    "final_state_sha256": str(trace.get("final_state_sha256") or ""),
                    "tail_evidence_sha256": content_sha256(trace["tail_evidence"]),
                    "status": "complete",
                }
            )
            traces.append(
                {
                    "execution_id": execution_id,
                    "mode": "episode_reset",
                    "proposal_profile_id": profile_id,
                    "source_or_case_id": case_id,
                    "routes": {
                        case_id: {
                            "episode_id": case_id,
                            "audio_epoch": 0,
                            "proposals": ordered,
                            "progress": trace["progress"],
                            "proposal_count": len(ordered),
                            "proposal_trace_sha256": proposal_trace_sha,
                            "progress_trace_sha256": progress_sha,
                            "final_state_sha256": str(trace.get("final_state_sha256") or ""),
                            "tail_evidence": trace["tail_evidence"],
                        }
                    },
                }
            )
            routes.append(
                [
                    profile_id,
                    case_id,
                    execution_id,
                    len(ordered),
                    proposal_trace_sha,
                    len(trace["progress"]),
                    progress_sha,
                    bool(trace["tail_evidence"]["pending_confirmation_suppressed"]),
                ]
            )
    return receipts, routes, traces

# ---------------------------------------------------------------------------
# B0/B1 episode evidence
# ---------------------------------------------------------------------------


def b1_equivalence_check(
    b0_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    fused = fuse_actions(
        b0_actions,
        [],
        detector_vad_radius_ms=250,
        same_silence_interval_association=False,
    )
    b1_actions = list(fused["final_actions"])
    b0_sig = [
        (
            str(row["action_kind"]),
            int(row["boundary_source_sample"]),
            int(row["observed_source_sample_at_emit"]),
        )
        for row in b0_actions
    ]
    b1_sig = [
        (
            str(row["action_kind"]),
            int(row["boundary_source_sample"]),
            int(row["observed_source_sample_at_emit"]),
        )
        for row in b1_actions
    ]
    receipt = {
        "kind_identical": [row[0] for row in b0_sig] == [row[0] for row in b1_sig],
        "boundary_identical": [row[1] for row in b0_sig] == [row[1] for row in b1_sig],
        "observed_identical": [row[2] for row in b0_sig] == [row[2] for row in b1_sig],
        "b0_action_count": len(b0_sig),
        "b1_action_count": len(b1_sig),
        "final_segmentation_identical": b0_sig == b1_sig,
    }
    receipt["passed"] = all(
        (
            receipt["kind_identical"],
            receipt["boundary_identical"],
            receipt["observed_identical"],
            receipt["final_segmentation_identical"],
        )
    )
    receipt["receipt_sha256"] = content_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    return {"b1_actions": b1_actions, "receipt": receipt}


def normalize_episode_lifecycle(
    lifecycle: Sequence[dict[str, Any]],
    warm_start: int,
    tail_end: int,
) -> list[dict[str, Any]]:
    starts: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in lifecycle:
        kind = str(row["event_kind"])
        if kind == "speech_start":
            utterance_id = str(row["normalized_utterance_id"])
            starts[utterance_id] = row
    for row in lifecycle:
        kind = str(row["event_kind"])
        utterance_id = str(row["normalized_utterance_id"])
        if kind == "speech_start":
            rows.append(dict(row))
        elif kind == "speech_end":
            if utterance_id in starts:
                rows.append(dict(row))
                del starts[utterance_id]
        else:
            continue
    started_ids = {
        str(row["normalized_utterance_id"])
        for row in rows
        if str(row["event_kind"]) == "speech_start"
    }
    ended_ids = {
        str(row["normalized_utterance_id"])
        for row in rows
        if str(row["event_kind"]) == "speech_end"
    }
    for utterance_id in sorted(ended_ids - started_ids):
        rows.append(
            {
                "event_kind": "speech_start",
                "normalized_utterance_id": utterance_id,
                "event_source_sample": warm_start,
                "observed_source_sample_at_emit": warm_start,
            }
        )
    for utterance_id in sorted(starts):
        rows.append(
            {
                "event_kind": "terminal",
                "normalized_utterance_id": utterance_id,
                "event_source_sample": tail_end,
                "observed_source_sample_at_emit": tail_end,
                "active_state_remained": True,
            }
        )
    kind_order = {"speech_start": 0, "speech_end": 1, "terminal": 2}
    rows.sort(
        key=lambda row: (
            int(row["event_source_sample"]),
            kind_order.get(str(row["event_kind"]), 3),
            str(row["normalized_utterance_id"]),
        )
    )
    return rows


def lifecycle_from_b0_projection(
    trace_projection: Sequence[dict[str, Any]],
    duration_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    maximum_seq = 0
    for row in trace_projection:
        debug = row.get("debug") or {}
        prev_seq = debug.get("prev_utterance_seq")
        if prev_seq is not None:
            maximum_seq = max(maximum_seq, int(prev_seq))
    fallback_id = f"u{maximum_seq + 1}"
    for index, row in enumerate(trace_projection):
        debug = row.get("debug") or {}
        prev_seq = debug.get("prev_utterance_seq")
        prev_start = debug.get("prev_utterance_start_sample")
        prev_end = debug.get("prev_speech_end_sample")
        boundary = int(row["boundary_source_sample"])
        observed = int(row["observed_source_sample_at_emit"])
        if prev_seq is not None and prev_start is not None and prev_end is not None:
            utterance_id = f"u{int(prev_seq)}"
            rows.append(
                {
                    "event_kind": "speech_end",
                    "normalized_utterance_id": utterance_id,
                    "event_source_sample": int(prev_end),
                    "observed_source_sample_at_emit": observed,
                    "reason": str(debug.get("prev_end_reason") or "silence"),
                }
            )
            if index == 0:
                rows.append(
                    {
                        "event_kind": "speech_start",
                        "normalized_utterance_id": utterance_id,
                        "event_source_sample": int(prev_start),
                        "observed_source_sample_at_emit": observed,
                    }
                )
        next_seq = None
        if index + 1 < len(trace_projection):
            next_seq = (trace_projection[index + 1].get("debug") or {}).get(
                "prev_utterance_seq"
            )
        if next_seq is not None:
            utterance_id = f"u{int(next_seq)}"
        else:
            utterance_id = fallback_id
        rows.append(
            {
                "event_kind": "speech_start",
                "normalized_utterance_id": utterance_id,
                "event_source_sample": boundary,
                "observed_source_sample_at_emit": observed,
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["observed_source_sample_at_emit"]),
            str(row["event_kind"]),
            str(row["normalized_utterance_id"]),
        )
    )
    return rows


def build_b0_evidence(
    b0_run: dict[str, Any],
    episodes: list[dict[str, Any]],
    historical_case_rows: list[dict[str, Any]],
    historical_b0: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    traces = b0_run["traces"]
    mismatch_count = 0
    for episode in episodes:
        session_id = str(episode["session_id"])
        trace = traces[session_id]
        b0_actions = project_b0_episode(trace, episode)
        lifecycle = normalize_episode_lifecycle(
            b0_lifecycle_rows(trace, episode),
            int(episode["bounds"]["warm_start"]),
            int(episode["bounds"]["tail_end"]),
        )
        equivalence = b1_equivalence_check(b0_actions)
        if not equivalence["receipt"]["passed"]:
            mismatch_count += 1
        rows.append(
            {
                "unit_id": str(episode["episode_id"]),
                "unit_kind": "current",
                "session_id": session_id,
                "pool": str(episode["pool"]),
                "b0_receipt_source": trace["b0_receipt_source"],
                "b0_trace_sha256": trace["trace_hash"],
                "b0_actions": b0_actions,
                "b1_actions": equivalence["b1_actions"],
                "lifecycle_events": lifecycle,
                "b1_equivalence": equivalence["receipt"],
            }
        )
    historical_by_case = {str(row["case_id"]): row for row in historical_b0["cases"]}
    for case in historical_case_rows:
        case_id = str(case["case_id"])
        baseline = historical_by_case[case_id]
        b0_actions = [
            {
                "action_id": f"historical-b0:{case_id}:{index}",
                "event_id": f"historical-b0:{case_id}:{index}",
                "audio_epoch": 0,
                "source_session_id": case_id,
                "boundary_source_sample": int(row["boundary_source_sample"]),
                "observed_source_sample_at_emit": int(
                    row["observed_source_sample_at_emit"]
                ),
                "action_kind": "retain_vad",
                "origin": "vad",
                "owner": "vad",
                "debug": row.get("debug") or {},
            }
            for index, row in enumerate(baseline["vad_boundaries"])
        ]
        lifecycle = normalize_episode_lifecycle(
            lifecycle_from_b0_projection(
                baseline["vad_boundaries"], int(case["duration_samples"])
            ),
            0,
            int(case["duration_samples"]),
        )
        equivalence = b1_equivalence_check(b0_actions)
        if not equivalence["receipt"]["passed"]:
            mismatch_count += 1
        rows.append(
            {
                "unit_id": case_id,
                "unit_kind": "historical",
                "session_id": case_id,
                "pool": "historical_validation_corrected_rescore_only",
                "b0_receipt_source": "pinned_phase3_b0_vad_only",
                "b0_trace_sha256": str(case["b0_actions_sha256"]),
                "b0_actions": b0_actions,
                "b1_actions": equivalence["b1_actions"],
                "lifecycle_events": lifecycle,
                "b1_equivalence": equivalence["receipt"],
            }
        )
    summary = {
        "unit_count": len(rows),
        "current_count": len(episodes),
        "historical_count": len(historical_case_rows),
        "b0_b1_mismatch_count": mismatch_count,
        "equivalence_receipts_sha256": rows_sha256(
            [row["b1_equivalence"] for row in rows]
        ),
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------


def stage_a(experiment_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    result = result_dir(experiment_dir)
    guard = verify_start_guard(experiment_dir)
    benchmarks = verify_benchmark_inputs(experiment_dir)
    ledger = json.loads((result / "phase_5_design_ledger.json").read_text(encoding="utf-8"))
    receipt_path = result / STAGE_A_RECEIPT_NAME
    if receipt_path.is_file():
        raise Phase5RunError("stage A already executed; refusing to restart the full stage")
    inputs = __import__(
        "experiments.speaker_turn_boundary.turn_episode.phase5_design",
        fromlist=["phase4_inputs"],
    ).phase4_inputs(result)
    episodes, _, source_by_episode = load_populations(experiment_dir, inputs)
    inventory = json.loads((result / "phase_4_cache_inventory.json").read_text(encoding="utf-8"))
    coverage = json.loads((result / "coverage_inventory.json").read_text(encoding="utf-8"))
    details = {
        str(row["session_id"]): row
        for row in map(
            json.loads,
            (result / "coverage_inventory_details.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    cases = load_synthetic_cases(experiment_dir / "data" / "manifests")
    from .phase4_signal import _source_maps

    sources, rebuilt_source_by_episode = _source_maps(
        experiment_dir, episodes, cases, coverage, details
    )
    if rebuilt_source_by_episode != source_by_episode:
        raise Phase5RunError("source map is not reproducible")
    profiles = proposal_profiles(experiment_dir, inputs["phase_4_signal_disposition.json"], inputs["phase_4_state_equivalence.json"])
    historical_development, historical_required, historical_available = (
        historical_development_contract(experiment_dir, profiles)
    )
    case_rows = historical_development["case_rows"]
    current_required = required_windows(episodes, source_by_episode, sources, profiles)
    merged_required = merge_window_sets(current_required, historical_required)
    phase4 = __import__(
        "experiments.speaker_turn_boundary.turn_episode.phase4_signal",
        fromlist=["load_inputs"],
    ).load_inputs(experiment_dir)
    window_info = window_ledger(
        merged_required,
        merge_window_sets(phase4.embedding_windows, historical_available),
    )
    expected_cache = ledger["cache_reuse"]
    for key in ("unique_window_count", "reusable_window_count", "new_inference_window_count"):
        if int(window_info[key]) != int(expected_cache[key]):
            raise Phase5RunError(f"window universe drift: {key}")
    if window_info["window_rows_sha256"] != expected_cache["window_rows_sha256"]:
        raise Phase5RunError("window universe rows digest drift")
    _validate_phase4_cache_inventory(inventory)
    sources_by_wav = source_by_wav(phase4)
    for source in sources.values():
        existing = sources_by_wav.get(source.wav_sha256)
        if existing is None or source.source_id < existing.source_id:
            sources_by_wav[source.wav_sha256] = source
    phase4_embeddings, import_summary = load_phase4_embeddings(
        inventory, merged_required, historical_development
    )
    if int(import_summary["reusable_window_count"]) != int(
        expected_cache["reusable_window_count"]
    ):
        raise Phase5RunError("phase4 reusable window import count drift")
    missing_by_wav: dict[str, set[tuple[int, int]]] = {}
    for wav, windows in merged_required.items():
        cached = set(phase4_embeddings.get(wav, {}))
        missing = windows - cached
        if missing:
            missing_by_wav[wav] = missing
    missing_total = sum(len(value) for value in missing_by_wav.values())
    if missing_total != int(expected_cache["new_inference_window_count"]):
        raise Phase5RunError("missing window count drift")
    inference = infer_missing_windows(
        experiment_dir, ledger, missing_by_wav, sources_by_wav, Path(args.eres_onnx_root)
    )
    if int(inference["new_inference_window_count"]) != missing_total:
        raise Phase5RunError("phase5 inference window count drift")
    phase5_embeddings = load_phase5_embeddings(
        inference["contract"],
        missing_by_wav,
        sources_by_wav,
    )
    embeddings_by_wav: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for wav in merged_required:
        merged = dict(phase4_embeddings.get(wav, {}))
        merged.update(phase5_embeddings.get(wav, {}))
        if len(merged) != len(merged_required[wav]):
            raise Phase5RunError(f"embedding coverage incomplete: {wav[:16]}")
        embeddings_by_wav[wav] = merged
    b0_run = run_b0_traces(experiment_dir, sources, source_by_episode)
    historical_b0 = json.loads(
        (experiment_dir / "results" / "phase3" / "dev_evidence" / "b0_vad_only.json").read_text(
            encoding="utf-8"
        )
    )
    b0_rows, b0_summary = build_b0_evidence(
        b0_run, episodes, case_rows, historical_b0
    )
    if b0_summary["b0_b1_mismatch_count"]:
        raise Phase5RunError("B0/B1 equivalence mismatch; stopping before Stage B")
    cache_receipt_hash = content_sha256(
        {
            "phase4": import_summary,
            "phase5": {
                "contract_sha256": inference["contract_sha256"],
                "new_inference_window_count": inference["new_inference_window_count"],
            },
        }
    )
    lifecycle_hash = content_sha256(
        [
            {"session_id": row["session_id"], "lifecycle": row["lifecycle"]}
            for row in b0_run["traces"].values()
        ]
    )
    current_receipts, current_routes, current_traces = build_current_proposal_traces(
        embeddings_by_wav,
        profiles,
        episodes,
        sources_by_wav,
        cache_receipt_hash,
        lifecycle_hash,
    )
    historical_receipts, historical_routes, historical_traces = (
        build_historical_proposal_traces(
            embeddings_by_wav,
            profiles,
            case_rows,
            cache_receipt_hash,
        )
    )
    all_receipts = current_receipts + historical_receipts
    all_routes = current_routes + historical_routes
    all_traces = current_traces + historical_traces
    if len(all_receipts) != 3824 or len(all_routes) != 4328:
        raise Phase5RunError(
            f"proposal receipt cardinality drift: {len(all_receipts)}/{len(all_routes)}"
        )
    word_receipts = expected_word_timing_receipts(
        experiment_dir, episodes, case_rows, details, cases
    )
    if word_receipts["current_receipt_count"] != 878 or word_receipts[
        "historical_receipt_count"
    ] != 204:
        raise Phase5RunError("word timing receipt unit count drift")
    save_proposal_traces(all_traces)
    execution_dir = result / "phase_5_proposal_executions"
    route_dir = result / "phase_5_proposal_routes"
    b0_dir = result / "phase_5_b0_evidence"
    execution_writer = RepresentationWriter(execution_dir, "phase_5_proposal_executions")
    route_writer = RepresentationWriter(route_dir, "phase_5_proposal_routes")
    b0_writer = RepresentationWriter(b0_dir, "phase_5_b0_evidence")
    execution_writer.add_rows(
        (str(row["proposal_execution_id"]), row)
        for row in sorted(all_receipts, key=lambda row: str(row["proposal_execution_id"]))
    )
    route_writer.add_rows(
        (f"{row[0]}|{row[1]}", row)
        for row in sorted(all_routes, key=lambda row: f"{row[0]}|{row[1]}")
    )
    b0_writer.add_rows(
        (str(row["unit_id"]), row) for row in sorted(b0_rows, key=lambda row: str(row["unit_id"]))
    )
    execution_receipt = execution_writer.write()
    route_receipt = route_writer.write()
    b0_receipt = b0_writer.write()
    verify_shard_receipts(execution_dir, execution_receipt["shards"])
    verify_shard_receipts(route_dir, route_receipt["shards"])
    verify_shard_receipts(b0_dir, b0_receipt["shards"])
    payload = {
        "schema_version": "turn_episode_phase5_stage_a.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "design_ledger_content_sha256": ledger["content_sha256"],
        "start_guard": guard,
        "benchmark_inputs": benchmarks,
        "window_universe": window_info,
        "phase4_cache_import": import_summary,
        "phase5_inference": {
            key: inference[key]
            for key in (
                "contract_sha256",
                "cache_root",
                "source_count",
                "cache_hit_count",
                "new_inference_window_count",
                "payload_bytes",
            )
        },
        "b0_evidence": b0_summary,
        "proposal_execution_receipts": execution_receipt,
        "logical_proposal_routes": route_receipt,
        "b0_evidence_shards": b0_receipt,
        "expected_word_timing_receipts": word_receipts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    written = atomic_write_json(receipt_path, payload)
    print(
        canonical_json(
            {
                "path": str(receipt_path),
                "content_sha256": written["content_sha256"],
                "proposal_execution_count": len(all_receipts),
                "logical_route_count": len(all_routes),
                "b0_unit_count": len(b0_rows),
                "expected_word_receipt_count": len(word_receipts["receipts"]),
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )
    )
    return written

# ---------------------------------------------------------------------------
# Logical/physical system universe
# ---------------------------------------------------------------------------


FULL_GRID = [
    (d, w, r, rep, v, s)
    for d in CLUSTER_DEBOUNCE_MS
    for w in CLUSTER_RADIUS_MS
    for r in REFRACTORY_MS
    for rep in REPRESENTATIVES
    for v in VAD_RADIUS_MS
    for s in SILENCE_ASSOCIATION
]
CLUSTER_WITHOUT_REFRACTORY = [
    (d, w, rep)
    for d in CLUSTER_DEBOUNCE_MS
    for w in CLUSTER_RADIUS_MS
    for rep in REPRESENTATIVES
]
CLUSTER_WITH_REFRACTORY = [
    (d, w, r, rep)
    for d in CLUSTER_DEBOUNCE_MS
    for w in CLUSTER_RADIUS_MS
    for r in REFRACTORY_MS
    for rep in REPRESENTATIVES
]
BASELINE_IDS = ("B0", "B1", "no_neural_policy_control")
CHAIN_FIELD_ORDER = (
    "cluster_debounce_ms",
    "cluster_boundary_radius_ms",
    "refractory_ms",
    "representative",
    "detector_vad_radius_ms",
    "same_silence_interval_association",
)


def chain_dict(chain: tuple[int, int, int, str, int, bool]) -> dict[str, Any]:
    return {
        "cluster_debounce_ms": chain[0],
        "cluster_boundary_radius_ms": chain[1],
        "refractory_ms": chain[2],
        "representative": chain[3],
        "detector_vad_radius_ms": chain[4],
        "same_silence_interval_association": chain[5],
    }


def physical_node_key(
    kind: str,
    profile_id: str | None,
    chain: tuple[int, int, int, str, int, bool] | None = None,
    control_kind: str | None = None,
    cluster: tuple[int, int, str] | tuple[int, int, int, str] | None = None,
) -> dict[str, Any]:
    if kind == "baseline":
        return {"kind": "baseline", "baseline_id": profile_id}
    if kind == "naive":
        return {"kind": "ladder", "proposal_profile_id": profile_id, "stage": "naive_proposal_as_cut"}
    if kind == "clustering_only":
        d, w, rep = cluster
        return {
            "kind": "ladder",
            "proposal_profile_id": profile_id,
            "stage": "clustering_only",
            "cluster_debounce_ms": d,
            "cluster_boundary_radius_ms": w,
            "representative": rep,
        }
    if kind == "clustering_plus_refractory":
        d, w, r, rep = cluster
        return {
            "kind": "ladder",
            "proposal_profile_id": profile_id,
            "stage": "clustering_plus_refractory",
            "cluster_debounce_ms": d,
            "cluster_boundary_radius_ms": w,
            "refractory_ms": r,
            "representative": rep,
        }
    if kind == "vad":
        return {
            "kind": "ladder",
            "proposal_profile_id": profile_id,
            "stage": "plus_vad_association",
            **chain_dict(chain),
        }
    if kind == "control":
        return {
            "kind": "frequency_control",
            "proposal_profile_id": profile_id,
            **chain_dict(chain),
            "control_kind": control_kind,
        }
    raise Phase5RunError(f"unknown physical node kind: {kind}")


def node_id(key: dict[str, Any]) -> str:
    return "node:" + content_sha256(key)


def logical_ladder_key(
    profile_id: str,
    chain: tuple[int, int, int, str, int, bool],
    stage: str,
) -> dict[str, Any]:
    return {
        "kind": "ladder",
        "proposal_profile_id": profile_id,
        **chain_dict(chain),
        "stage": stage,
    }


def logical_control_key(
    profile_id: str,
    chain: tuple[int, int, int, str, int, bool],
    control_kind: str,
) -> dict[str, Any]:
    return {
        "kind": "frequency_control",
        "proposal_profile_id": profile_id,
        **chain_dict(chain),
        "control_kind": control_kind,
    }


def system_id(key: dict[str, Any]) -> str:
    return "system:" + content_sha256(key)


def stage_index(stage: str) -> int:
    return LADDER_STAGES.index(stage)


def build_system_universe(
    profiles: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    physical_rows: list[list[Any]] = []
    logical_rows: list[list[Any]] = []
    alias_rows: list[list[str]] = []
    logical_to_physical: dict[str, str] = {}
    for baseline_id in BASELINE_IDS:
        key = {"kind": "baseline", "baseline_id": baseline_id}
        node = node_id(key)
        logical = system_id(key)
        physical_rows.append([node, baseline_id, -1, -1, -1, -1, -1, -1, -1, -1])
        logical_rows.append([logical, node, "", -1, content_sha256(key)])
        logical_to_physical[logical] = node
    representative_chain = FULL_GRID[0]
    for profile in profiles:
        profile_id = str(profile["proposal_profile_id"])
        representative_by_cluster: dict[str, tuple] = {}
        for chain in FULL_GRID:
            d, w, r, rep, _, _ = chain
            cluster = (d, w, r, rep)
            representative_by_cluster.setdefault(str(cluster), chain)
        for stage in LADDER_STAGES:
            for chain in FULL_GRID:
                key = logical_ladder_key(profile_id, chain, stage)
                logical = system_id(key)
                if stage == "full_hard_soft_fusion":
                    physical_key = physical_node_key("vad", profile_id, chain=chain)
                else:
                    physical_key = physical_node_for_logical(profile_id, chain, stage)
                node = node_id(physical_key)
                logical_rows.append(
                    [logical, node, profile_id, stage_index(stage), content_sha256(key)]
                )
                logical_to_physical[logical] = node
        for chain in FULL_GRID:
            for kind in CONTROL_KINDS:
                key = logical_control_key(profile_id, chain, kind)
                logical = system_id(key)
                physical_key = physical_node_key(
                    "control", profile_id, chain=chain, control_kind=kind
                )
                node = node_id(physical_key)
                logical_rows.append([logical, node, profile_id, 5, content_sha256(key)])
                logical_to_physical[logical] = node
        alias_count = 0
        for chain in FULL_GRID:
            d, w, r, rep, _, _ = chain
            if chain == representative_chain:
                continue
            logical = system_id(logical_ladder_key(profile_id, chain, "naive_proposal_as_cut"))
            node = node_id(physical_node_key("naive", profile_id))
            alias_rows.append([logical, node, "inactive_later_parameters_are_execution_aliases"])
            alias_count += 1
        for cluster in CLUSTER_WITHOUT_REFRACTORY:
            d, w, rep = cluster
            representative = representative_by_cluster[str((d, w, 0, rep))]
            for chain in FULL_GRID:
                cd, cw, cr, crep, _, _ = chain
                if (cd, cw, crep) != cluster:
                    continue
                if chain == representative:
                    continue
                logical = system_id(logical_ladder_key(profile_id, chain, "clustering_only"))
                node = node_id(physical_node_key("clustering_only", profile_id, cluster=cluster))
                alias_rows.append([logical, node, "inactive_later_parameters_are_execution_aliases"])
                alias_count += 1
        for cluster in CLUSTER_WITH_REFRACTORY:
            d, w, r, rep = cluster
            representative = representative_by_cluster[str(cluster)]
            for chain in FULL_GRID:
                cd, cw, cr, crep, _, _ = chain
                if (cd, cw, cr, crep) != cluster:
                    continue
                if chain == representative:
                    continue
                logical = system_id(
                    logical_ladder_key(profile_id, chain, "clustering_plus_refractory")
                )
                node = node_id(
                    physical_node_key("clustering_plus_refractory", profile_id, cluster=cluster)
                )
                alias_rows.append([logical, node, "inactive_later_parameters_are_execution_aliases"])
                alias_count += 1
        for chain in FULL_GRID:
            logical = system_id(logical_ladder_key(profile_id, chain, "full_hard_soft_fusion"))
            node = node_id(physical_node_key("vad", profile_id, chain=chain))
            alias_rows.append([logical, node, "eres_full_hard_soft_equals_vad_association"])
            alias_count += 1
        if alias_count != 527:
            raise Phase5RunError(f"per-profile alias count drift: {alias_count}")
    for profile in profiles:
        profile_id = str(profile["proposal_profile_id"])
        seen: set[str] = set()

        def add_physical(
            key: dict[str, Any], stage_label: int, params: tuple[int, ...]
        ) -> None:
            node = node_id(key)
            if node in seen:
                return
            seen.add(node)
            d, w, r, rep, v, s, c = params
            physical_rows.append([node, profile_id, stage_label, d, w, r, rep, v, s, c])

        for chain in FULL_GRID:
            d, w, r, rep, v, s = chain
            add_physical(
                physical_node_key("vad", profile_id, chain=chain),
                3,
                (d, w, r, rep, v, s, -1),
            )
        for cluster in CLUSTER_WITH_REFRACTORY:
            d, w, r, rep = cluster
            add_physical(
                physical_node_key("clustering_plus_refractory", profile_id, cluster=cluster),
                2,
                (d, w, r, rep, -1, -1, -1),
            )
        for cluster in CLUSTER_WITHOUT_REFRACTORY:
            d, w, rep = cluster
            add_physical(
                physical_node_key("clustering_only", profile_id, cluster=cluster),
                1,
                (d, w, 0, rep, -1, -1, -1),
            )
        add_physical(physical_node_key("naive", profile_id), 0, (0, 0, 0, 0, 0, 0, 0))
        for chain in FULL_GRID:
            d, w, r, rep, v, s = chain
            for index, kind in enumerate(CONTROL_KINDS):
                add_physical(
                    physical_node_key(
                        "control", profile_id, chain=chain, control_kind=kind
                    ),
                    5,
                    (d, w, r, rep, v, s, index),
                )
    if len(physical_rows) != PHYSICAL_SYSTEMS:
        raise Phase5RunError(f"physical system count drift: {len(physical_rows)}")
    if len(logical_rows) != LOGICAL_SYSTEMS:
        raise Phase5RunError(f"logical system count drift: {len(logical_rows)}")
    if len(alias_rows) != LOGICAL_ALIAS_EDGES:
        raise Phase5RunError(f"logical alias edge count drift: {len(alias_rows)}")
    physical_rows.sort(key=lambda row: str(row[0]))
    logical_rows.sort(key=lambda row: str(row[0]))
    alias_rows.sort(key=lambda row: str(row[0]))
    node_group: dict[str, str] = {}
    system_info: dict[str, dict[str, Any]] = {}
    for baseline_id in BASELINE_IDS:
        logical = system_id({"kind": "baseline", "baseline_id": baseline_id})
        system_info[logical] = {
            "baseline_id": baseline_id,
            "profile_id": "",
            "stage": "baseline",
            "chain": None,
            "control_kind": None,
        }
    for profile in profiles:
        profile_id = str(profile["proposal_profile_id"])
        policy_class = str(profile["profile_class"])
        for chain in FULL_GRID:
            for stage in LADDER_STAGES:
                logical = system_id(logical_ladder_key(profile_id, chain, stage))
                system_info[logical] = {
                    "baseline_id": None,
                    "profile_id": profile_id,
                    "policy_class": policy_class,
                    "stage": stage,
                    "chain": chain_dict(chain),
                    "control_kind": None,
                }
            for kind in CONTROL_KINDS:
                logical = system_id(logical_control_key(profile_id, chain, kind))
                system_info[logical] = {
                    "baseline_id": None,
                    "profile_id": profile_id,
                    "policy_class": policy_class,
                    "stage": "frequency_control",
                    "chain": chain_dict(chain),
                    "control_kind": kind,
                }
        for chain in FULL_GRID:
            node_group[node_id(physical_node_key("vad", profile_id, chain=chain))] = (
                physical_group_id("vad", chain=chain)
            )
        for cluster in CLUSTER_WITH_REFRACTORY:
            node_group[
                node_id(
                    physical_node_key(
                        "clustering_plus_refractory", profile_id, cluster=cluster
                    )
                )
            ] = physical_group_id("clustering_plus_refractory", cluster=cluster)
        for cluster in CLUSTER_WITHOUT_REFRACTORY:
            node_group[
                node_id(physical_node_key("clustering_only", profile_id, cluster=cluster))
            ] = physical_group_id("clustering_only", cluster=cluster)
        node_group[node_id(physical_node_key("naive", profile_id))] = "naive"
        for chain in FULL_GRID:
            for kind in CONTROL_KINDS:
                node_group[
                    node_id(
                        physical_node_key(
                            "control", profile_id, chain=chain, control_kind=kind
                        )
                    )
                ] = physical_group_id("control", chain=chain, control_kind=kind)
    return {
        "physical_rows": physical_rows,
        "logical_rows": logical_rows,
        "alias_rows": alias_rows,
        "logical_to_physical": logical_to_physical,
        "node_group": node_group,
        "system_info": system_info,
    }


def physical_group_id(
    kind: str,
    chain: tuple[int, int, int, str, int, bool] | None = None,
    cluster: tuple[int, int, str] | tuple[int, int, int, str] | None = None,
    control_kind: str | None = None,
) -> str:
    if kind == "naive":
        return "naive"
    if kind == "clustering_only":
        return f"clustering_only|{'|'.join(str(v) for v in cluster)}"
    if kind == "clustering_plus_refractory":
        return f"clustering_plus_refractory|{'|'.join(str(v) for v in cluster)}"
    if kind == "vad":
        return "vad|" + "|".join(str(v) for v in chain)
    if kind == "control":
        return "control|" + "|".join(str(v) for v in chain) + f"|{control_kind}"
    raise Phase5RunError(f"unknown physical group: {kind}")


def physical_node_for_logical(
    profile_id: str,
    chain: tuple[int, int, int, str, int, bool],
    stage: str,
) -> dict[str, Any]:
    if stage == "naive_proposal_as_cut":
        return physical_node_key("naive", profile_id)
    if stage == "clustering_only":
        d, w, _, r, _, _ = chain
        return physical_node_key("clustering_only", profile_id, cluster=(d, w, r))
    if stage == "clustering_plus_refractory":
        d, w, r, rep, _, _ = chain
        return physical_node_key("clustering_plus_refractory", profile_id, cluster=(d, w, r, rep))
    return physical_node_key("vad", profile_id, chain=chain)


def seed_material(profile_id: str, unit_id: str) -> str:
    return f"turn-episode-v1-phase5|{profile_id}|{unit_id}"


def naive_cut(proposals: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proposal in sorted(
        proposals,
        key=lambda row: (
            int(row["observed_source_sample_at_emit"]),
            int(row["boundary_source_sample"]),
            str(row["profile_id"]),
            str(row["proposal_id"]),
        ),
    ):
        rows.append(
            {
                "final_action_id": f"final:naive:{proposal['proposal_id']}",
                "event_id": f"naive:{proposal['proposal_id']}",
                "detector_action_id": f"naive:{proposal['proposal_id']}",
                "origin": "detector",
                "owner": "detector",
                "action_kind": "add_hard_boundary",
                "boundary_source_sample": int(proposal["boundary_source_sample"]),
                "observed_source_sample_at_emit": int(
                    proposal["observed_source_sample_at_emit"]
                ),
                "audio_epoch": int(proposal["audio_epoch"]),
                "source_session_id": str(proposal["source_session_id"]),
                "cluster_id": "",
                "proposal_kind": str(proposal["proposal_kind"]),
                "confidence": float(proposal["confidence"]),
                "confidence_semantics_id": str(proposal["confidence_semantics_id"]),
            }
        )
    return rows


def actionize_detector_actions(
    clusters: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in actionize_clusters(clusters):
        requested = str(action["requested_action"])
        if requested == "hard_candidate":
            action_kind = "add_hard_boundary"
        elif requested == "soft_marker":
            action_kind = "emit_soft_marker"
        else:
            continue
        rows.append(
            {
                **action,
                "action_kind": action_kind,
                "final_action_id": f"final:{action['detector_action_id']}",
                "event_id": action["detector_action_id"],
                "origin": "detector",
                "owner": "detector",
            }
        )
    return rows


def execute_physical_episode(
    proposals: Sequence[dict[str, Any]],
    vad_actions: Sequence[dict[str, Any]],
    lifecycle_events: Sequence[dict[str, Any]],
    *,
    episode_observed_end: int,
    waveform: np.ndarray | None,
    profile_id: str,
    unit_id: str,
    wave_start: int = 0,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    naive = naive_cut(proposals)
    results["naive"] = {"final_actions": naive}
    clustered_by: dict[tuple, dict[str, Any]] = {}
    for cluster in CLUSTER_WITH_REFRACTORY:
        d, w, r, rep = cluster
        clustered = cluster_proposals(
            proposals,
            cluster_debounce_ms=d,
            cluster_boundary_radius_ms=w,
            refractory_ms=r,
            representative=rep,
            episode_observed_end=episode_observed_end,
        )
        clustered_by[cluster] = clustered
        results[physical_group_id("clustering_plus_refractory", cluster=cluster)] = {
            "final_actions": actionize_detector_actions(clustered["clusters"])
        }
    for cluster in CLUSTER_WITHOUT_REFRACTORY:
        d, w, rep = cluster
        key = (d, w, 0, rep)
        if key in clustered_by:
            clustered = clustered_by[key]
        else:
            clustered = cluster_proposals(
                proposals,
                cluster_debounce_ms=d,
                cluster_boundary_radius_ms=w,
                refractory_ms=0,
                representative=rep,
                episode_observed_end=episode_observed_end,
            )
        results[physical_group_id("clustering_only", cluster=cluster)] = {
            "final_actions": actionize_detector_actions(clustered["clusters"])
        }
    detector_by_chain: dict[tuple, list[dict[str, Any]]] = {}
    active_intervals: list[dict[str, Any]] = []
    energy_candidates: list[dict[str, Any]] = []
    if lifecycle_events:
        active_intervals = active_intervals_from_lifecycle(
            lifecycle_events, episode_observed_end
        )
    if active_intervals and waveform is not None:
        local_intervals: list[dict[str, Any]] = []
        for interval in active_intervals:
            local = {
                **interval,
                "start": max(0, int(interval["start"]) - wave_start),
                "end": min(waveform.size, int(interval["end"]) - wave_start),
            }
            if local["end"] > local["start"]:
                local_intervals.append(local)
        if local_intervals:
            energy_candidates = causal_energy_candidates(
                waveform, local_intervals, offset_samples=wave_start
            )
    for chain in FULL_GRID:
        d, w, r, rep, v, s = chain
        cluster = (d, w, r, rep)
        detector = detector_by_chain.get(chain[:4])
        if detector is None:
            clustered = clustered_by[cluster]
            detector = actionize_clusters(clustered["clusters"])
            detector_by_chain[chain[:4]] = detector
        fused = fuse_actions(
            vad_actions,
            detector,
            detector_vad_radius_ms=v,
            same_silence_interval_association=s,
        )
        results[physical_group_id("vad", chain=chain)] = {
            "final_actions": fused["final_actions"]
        }
    for chain in FULL_GRID:
        group = results[physical_group_id("vad", chain=chain)]
        neural = group["final_actions"]
        for kind in CONTROL_KINDS:
            control = frequency_matched_control(
                kind,
                neural,
                active_intervals,
                energy_candidates=energy_candidates,
                forbidden_boundaries=[],
                seed_material=seed_material(profile_id, unit_id),
            )
            actions = list(control["actions"])
            results[physical_group_id("control", chain=chain, control_kind=kind)] = {
                "final_actions": actions,
                "control_status": control["status"],
                "infeasible_count": len(control["infeasible_placements"]),
                "required_count": int(control["required_hard_action_count"]),
                "placed_count": int(control["placed_hard_action_count"]),
            }
    return results

# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(round(percent / 100.0 * (len(ordered) - 1)))))
    return float(ordered[index])


PERCENTILE_FIELDS = (
    ("segment_duration_p10_samples", 10),
    ("segment_duration_p50_samples", 50),
    ("segment_duration_p90_samples", 90),
    ("active_speech_duration_p10_samples", 10),
    ("active_speech_duration_p50_samples", 50),
    ("active_speech_duration_p90_samples", 90),
)


def _apply_percentiles_from_durations(
    metrics: list[int],
    durations: Sequence[int],
    active_durations: Sequence[int],
) -> list[int]:
    for field, percent in PERCENTILE_FIELDS:
        index = SYSTEM_METRIC_FIELDS.index(field)
        values = durations if field.startswith("segment_duration_") else active_durations
        metrics[index] = int(percentile(values, percent))
    return metrics


def score_metric_vector(
    score: dict[str, Any],
    episode: dict[str, Any],
    references: Sequence[dict[str, Any]],
    *,
    control_infeasible: int,
    pool: str,
    hard_reference_kind_map: dict[str, str],
    deadline_views: dict[str, int],
) -> dict[str, Any]:
    tag = str(episode["tag"])
    headline = tag == "hard_only"
    hard_matches = [
        match
        for match in score["matches"]
        if hard_reference_kind_map.get(str(match["reference_id"])) == "hard_boundary"
    ]
    hard_miss_count = int(score["hard_miss_count"])
    attribution = score["benefit_attribution_counts"]
    segments = [int(value) for value in score["segment_source_duration_samples"]]
    active_segments = [int(value) for value in score["segment_active_duration_samples"]]
    delays = [int(match["availability_delay_ms"]) for match in score["matches"]]
    localization = [int(match["localization_error_ms"]) for match in score["matches"]]
    harm = score["harm_or_structure_counts"]
    counterfactual = score["overlap_hard_action_counterfactual_by_owner_threshold"]
    clean_gap_contaminated = score.get("clean_gap_contaminated_samples")
    clean_gap_denominator = score.get("clean_gap_singleton_denominator_samples")
    natural = pool == "natural_exposure_validation"
    contamination_100 = score["contamination_by_owner_threshold"]["100"]
    vector: dict[str, Any] = {
        "episode_count": 1,
        "clean_gap_episode_count": 1 if headline else 0,
        "clean_gap_singleton_denominator_samples": int(clean_gap_denominator or 0),
        "clean_gap_contaminated_samples": int(clean_gap_contaminated or 0),
        "mixed_turn_100ms_count": int(score["second_speaker_turn_counts"]["100"]),
        "mixed_turn_250ms_count": int(score["second_speaker_turn_counts"]["250"]),
        "mixed_turn_500ms_count": int(score["second_speaker_turn_counts"]["500"]),
        "clean_gap_hard_target_count": int(score["clean_gap_hard_target_count"]),
        "hard_match_count": len(hard_matches),
        "hard_miss_count": hard_miss_count,
        "retained_b0_success_count": int(attribution.get("retained_b0_success", 0)),
        "recovered_b0_hard_miss_count": int(attribution.get("recovered_b0_hard_miss", 0)),
        "accelerated_b0_success_count": int(attribution.get("accelerated_b0_success", 0)),
        "late_target_action_count": int(attribution.get("late_target_action", 0)),
        "detector_created_hard_action_count": int(
            score["detector_created_hard_action_count"]
        ),
        "harmful_active_split_count_100ms": int(
            score["harmful_active_split_sensitivity"]["100"]
        ),
        "harmful_active_split_count_200ms": int(
            score["harmful_active_split_sensitivity"]["200"]
        ),
        "harmful_active_split_count_300ms": int(
            score["harmful_active_split_sensitivity"]["300"]
        ),
        "lexical_split_count": int(harm.get("lexical_split", 0)),
        "lexical_not_observable_count": int(harm.get("lexical_not_observable", 0)),
        "duplicate_hard_boundary_count": int(harm.get("duplicate_hard_boundary", 0)),
        "same_speaker_pause_split_count": int(harm.get("same_speaker_pause_split", 0)),
        "same_speaker_extra_turn_count": int(score["same_speaker_extra_turn_count"]),
        "overlap_hard_action_count": int(harm.get("overlap_hard_action", 0)),
        "unscored_action_count": int(harm.get("unscored_action", 0)),
        "fragments_lt_250ms_count": int(score["short_active_fragment_counts"]["250"]),
        "fragments_lt_500ms_count": int(score["short_active_fragment_counts"]["500"]),
        "fragments_lt_1000ms_count": int(score["short_active_fragment_counts"]["1000"]),
        "segment_duration_p10_samples": percentile(segments, 10),
        "segment_duration_p50_samples": percentile(segments, 50),
        "segment_duration_p90_samples": percentile(segments, 90),
        "active_speech_duration_p10_samples": percentile(active_segments, 10),
        "active_speech_duration_p50_samples": percentile(active_segments, 50),
        "active_speech_duration_p90_samples": percentile(active_segments, 90),
        "availability_delay_sum_samples": sum(delays) * 16,
        "availability_delay_count": len(delays),
        "localization_error_sum_samples": sum(localization) * 16,
        "localization_error_count": len(localization),
        "control_infeasible_count": control_infeasible,
        "overlap_counterfactual_actual_samples_50ms": int(
            counterfactual["50"]["actual_contaminated_samples"]
        ),
        "overlap_counterfactual_suppressed_samples_50ms": int(
            counterfactual["50"]["suppressed_contaminated_samples"]
        ),
        "overlap_counterfactual_actual_samples_100ms": int(
            counterfactual["100"]["actual_contaminated_samples"]
        ),
        "overlap_counterfactual_suppressed_samples_100ms": int(
            counterfactual["100"]["suppressed_contaminated_samples"]
        ),
        "overlap_counterfactual_actual_samples_200ms": int(
            counterfactual["200"]["actual_contaminated_samples"]
        ),
        "overlap_counterfactual_suppressed_samples_200ms": int(
            counterfactual["200"]["suppressed_contaminated_samples"]
        ),
        "b0_b1_mismatch_count": 0,
    }
    if natural:
        vector.update(
            {
                "natural_contamination_numerator_samples": int(
                    contamination_100["contaminated_samples"]
                ),
                "natural_contamination_denominator_samples": int(
                    contamination_100["denominator_samples"]
                ),
                "natural_harmful_active_split_count": int(
                    score["harmful_active_split_sensitivity"]["200"]
                ),
                "natural_same_speaker_extra_turn_count": int(
                    score["same_speaker_extra_turn_count"]
                ),
                "natural_sampled_source_samples": int(
                    episode["bounds"]["scored_end"] - episode["bounds"]["scored_start"]
                ),
                "natural_sampled_active_speech_samples": int(
                    score["sampled_singleton_exposure_samples"]
                ),
                "natural_eligible_source_samples": int(
                    episode["bounds"]["scored_end"] - episode["bounds"]["scored_start"]
                ),
                "natural_session_count": 1,
            }
        )
    else:
        vector.update(
            {
                "natural_contamination_numerator_samples": 0,
                "natural_contamination_denominator_samples": 0,
                "natural_harmful_active_split_count": 0,
                "natural_same_speaker_extra_turn_count": 0,
                "natural_sampled_source_samples": 0,
                "natural_sampled_active_speech_samples": 0,
                "natural_eligible_source_samples": 0,
                "natural_session_count": 0,
            }
        )
    return vector


def score_block_vector(
    episode_metrics: Sequence[dict[str, Any]],
    b0_metrics: dict[str, Any] | None,
    b1_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    def total(key: str) -> int:
        return sum(int(row.get(key, 0)) for row in episode_metrics)

    def b0_value(key: str) -> int:
        return int(b0_metrics.get(key, 0)) if b0_metrics is not None else 0

    def b1_value(key: str) -> int:
        return int(b1_metrics.get(key, 0)) if b1_metrics is not None else 0

    deadline_counts = Counter()
    for row in episode_metrics:
        for deadline, count in (row.get("deadline_views") or {}).items():
            deadline_counts[str(deadline)] += int(count)
    return {
        "episode_count": total("episode_count"),
        "clean_gap_singleton_denominator_samples": total(
            "clean_gap_singleton_denominator_samples"
        ),
        "candidate_clean_gap_contaminated_samples": total(
            "clean_gap_contaminated_samples"
        ),
        "b0_clean_gap_contaminated_samples": b0_value("clean_gap_contaminated_samples"),
        "b1_clean_gap_contaminated_samples": b1_value("clean_gap_contaminated_samples"),
        "candidate_harmful_active_split_count": total(
            "harmful_active_split_count_200ms"
        ),
        "b0_harmful_active_split_count": b0_value("harmful_active_split_count_200ms"),
        "b1_harmful_active_split_count": b1_value("harmful_active_split_count_200ms"),
        "detector_created_hard_action_count": total(
            "detector_created_hard_action_count"
        ),
        "same_speaker_extra_turn_count": total("same_speaker_extra_turn_count"),
        "lexical_split_count": total("lexical_split_count"),
        "lexical_observable_action_count": total(
            "lexical_split_count"
        )
        + total("lexical_not_observable_count"),
        "duplicate_hard_boundary_count": total("duplicate_hard_boundary_count"),
        "hard_target_count": total("clean_gap_hard_target_count"),
        "hard_match_250ms_count": deadline_counts.get("250", 0),
        "hard_match_500ms_count": deadline_counts.get("500", 0),
        "hard_match_1000ms_count": deadline_counts.get("1000", 0),
        "hard_match_1500ms_count": deadline_counts.get("1500", 0),
        "hard_match_2000ms_count": deadline_counts.get("2000", 0),
        "availability_delay_sum_samples": total("availability_delay_sum_samples"),
        "availability_delay_count": total("availability_delay_count"),
        "overlap_hard_action_count": total("overlap_hard_action_count"),
        "overlap_contribution_samples": total(
            "overlap_counterfactual_actual_samples_100ms"
        )
        - total("overlap_counterfactual_suppressed_samples_100ms"),
        "sampled_source_samples": total("natural_sampled_source_samples"),
        "sampled_active_speech_samples": total("natural_sampled_active_speech_samples"),
        "natural_exposure_eligible_source_samples": total(
            "natural_eligible_source_samples"
        ),
    }

# ---------------------------------------------------------------------------
# Stage B workers
# ---------------------------------------------------------------------------


def _score_unit(
    final_actions: Sequence[dict[str, Any]],
    b0_actions: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
    views: dict[str, Any],
    *,
    scored_start: int,
    scored_end: int,
    episode_tag: str,
) -> dict[str, Any]:
    reference_rows = [ReferenceAction.from_dict(row) for row in references]
    score = score_policy_episode(
        final_actions,
        b0_actions,
        reference_rows,
        views["singleton_intervals"],
        views["pause_intervals"],
        views["overlap_intervals"],
        views["word_intervals"],
        views["unscored_intervals"],
        scored_start=scored_start,
        scored_end=scored_end,
        episode_tag=episode_tag,
    )
    clean_gap_hard_target_count = sum(
        1
        for row in references
        if str(row["action_kind"]) == "hard_boundary"
        and bool(row.get("scorable"))
        and episode_tag == "hard_only"
    )
    score["clean_gap_hard_target_count"] = clean_gap_hard_target_count
    return score


def _failure_signals(
    vad_groups: dict[str, dict[str, Any]],
    b0_clean_gap: int | None,
    proposals: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    signals: dict[str, dict[str, Any]] = {}
    max_proposal_confidence = (
        max((float(row["confidence"]) for row in proposals), default=0.0)
        if proposals
        else 0.0
    )
    for group in vad_groups.values():
        score = group["score"]
        harm = score["harm_or_structure_counts"]
        candidate_clean = int(score.get("clean_gap_contaminated_samples") or 0)
        b0_clean = int(b0_clean_gap or 0)
        candidates = {
            "contamination_regression": (
                candidate_clean - b0_clean,
                candidate_clean,
                b0_clean,
            ),
            "contamination_improvement": (
                b0_clean - candidate_clean,
                b0_clean,
                candidate_clean,
            ),
            "harmful_active_split": (
                int(score["harmful_active_split_sensitivity"]["200"]),
                None,
                None,
            ),
            "duplicate_cluster": (int(harm.get("duplicate_hard_boundary", 0)), None, None),
            "overlap_hard_action": (int(harm.get("overlap_hard_action", 0)), None, None),
        }
        late = 0
        late_boundary = 0
        for match in score["matches"]:
            delay = int(match["availability_delay_ms"])
            localization = int(match["localization_error_ms"])
            if delay > late and localization <= 500:
                late = delay
        candidates["late_accurate_target"] = (late, None, None)
        miss_signal = (
            int(score["hard_miss_count"]),
            max_proposal_confidence,
            None,
        )
        for category, (value, first, second) in candidates.items():
            boundary = _category_boundary(score, category)
            if category == "clean_gap_miss_strong_evidence":
                continue
            current = signals.get(category)
            if current is None or value > float(current["value"]):
                signals[category] = {
                    "value": float(value),
                    "boundary": boundary,
                    "action_trace_sha256": group["action_trace_sha256"],
                    "score_evidence_sha256": group["score_trace_sha256"],
                }
        current = signals.get("clean_gap_miss_strong_evidence")
        if miss_signal[0] > 0 and (
            current is None
            or (
                miss_signal[0] > 0
                and float(miss_signal[1]) > float(current["value"])
            )
        ):
            signals["clean_gap_miss_strong_evidence"] = {
                "value": float(miss_signal[1]),
                "boundary": _category_boundary(score, "clean_gap_miss_strong_evidence"),
                "action_trace_sha256": group["action_trace_sha256"],
                "score_evidence_sha256": group["score_trace_sha256"],
            }
    return signals


def _category_boundary(score: dict[str, Any], category: str) -> int:
    if category == "overlap_hard_action" and score["matches"]:
        return int(score["matches"][-1].get("boundary_source_sample", 0))
    return 0


def _worker_group_summary(
    result: dict[str, Any],
    references: Sequence[dict[str, Any]],
    pool: str,
    episode: dict[str, Any],
) -> dict[str, Any]:
    hard_kind_map = {
        str(row["reference_id"]): str(row["action_kind"]) for row in references
    }
    summary: dict[str, Any] = {}
    for group_id, group in result.items():
        score = group["score"]
        metrics = score_metric_vector(
            score,
            episode,
            references,
            control_infeasible=int(group.get("control_infeasible", 0)),
            pool=pool,
            hard_reference_kind_map=hard_kind_map,
            deadline_views=score["deadline_views"],
        )
        summary[group_id] = {
            "metrics": metrics,
            "control_infeasible": int(group.get("control_infeasible", 0)),
            "action_trace_sha256": str(group.get("action_trace_sha256", "")),
            "score_trace_sha256": str(group.get("score_trace_sha256", "")),
            "final_action_count": int(group.get("final_action_count", 0)),
            "segment_durations": [int(v) for v in score["segment_source_duration_samples"]],
            "active_segment_durations": [
                int(v) for v in score["segment_active_duration_samples"]
            ],
            "deadline_views": dict(score["deadline_views"]),
        }
    return summary


def current_batch_worker(batch: dict[str, Any]) -> dict[str, Any]:
    profile = batch["profile"]
    episode = batch["episode"]
    profile_id = str(profile["proposal_profile_id"])
    episode_id = str(episode["episode_id"])
    proposals = batch["proposals"]
    b0_actions = batch["b0_actions"]
    lifecycle = batch["lifecycle"]
    references = batch["references"]
    views = batch["views"]
    bounds = episode["bounds"]
    scored_start = int(bounds["scored_start"])
    scored_end = int(bounds["scored_end"])
    observed_end = int(bounds["tail_end"])
    waveform = None
    if batch.get("wav_path"):
        waveform = read_wav_slice(
            Path(batch["wav_path"]), int(batch["wave_start"]), int(batch["wave_end"])
        )
    physical = execute_physical_episode(
        proposals,
        b0_actions,
        lifecycle,
        episode_observed_end=observed_end,
        waveform=waveform,
        profile_id=profile_id,
        unit_id=episode_id,
        wave_start=int(batch.get("wave_start", 0)),
    )
    groups: dict[str, dict[str, Any]] = {}
    vad_groups: dict[str, dict[str, Any]] = {}
    for group_id, result in physical.items():
        score = _score_unit(
            result["final_actions"],
            b0_actions,
            references,
            views,
            scored_start=scored_start,
            scored_end=scored_end,
            episode_tag=str(episode["tag"]),
        )
        entry = {
            "score": score,
            "control_infeasible": int(result.get("infeasible_count", 0)),
            "action_trace_sha256": content_sha256(result["final_actions"]),
            "score_trace_sha256": content_sha256(score),
            "final_action_count": len(result["final_actions"]),
        }
        groups[group_id] = entry
        if group_id.startswith("vad|"):
            vad_groups[group_id] = entry
    b0_clean_gap = batch.get("b0_clean_gap")
    failure_signals = _failure_signals(vad_groups, b0_clean_gap, proposals)
    observed_receipt = word_timing_receipt(
        episode_id,
        batch["annotation_source_identity"],
        views,
        batch.get("raw_words"),
        batch.get("word_record_sha256", ""),
    )
    return {
        "unit_id": episode_id,
        "profile_id": profile_id,
        "session_id": str(episode["session_id"]),
        "pool": str(episode["pool"]),
        "block_id": batch["block_id"],
        "corpus": corpus_for(str(episode["session_id"])),
        "episode_tag": str(episode["tag"]),
        "proposal_count": len(proposals),
        "b0_action_count": len(b0_actions),
        "reference_count": len(references),
        "observed_word_receipt": observed_receipt,
        "failure_signals": failure_signals,
        "groups": _worker_group_summary(groups, references, str(episode["pool"]), episode),
    }


def historical_batch_worker(batch: dict[str, Any]) -> dict[str, Any]:
    profile = batch["profile"]
    case = batch["case"]
    profile_id = str(profile["proposal_profile_id"])
    case_id = str(case["case_id"])
    proposals = batch["proposals"]
    b0_actions = batch["b0_actions"]
    lifecycle = batch["lifecycle"]
    references = batch["references"]
    views = batch["views"]
    duration = int(case["duration_samples"])
    waveform = None
    if batch.get("wav_path"):
        waveform = read_wav_slice(
            Path(batch["wav_path"]), int(batch["wave_start"]), int(batch["wave_end"])
        )
    physical = execute_physical_episode(
        proposals,
        b0_actions,
        lifecycle,
        episode_observed_end=duration,
        waveform=waveform,
        profile_id=profile_id,
        unit_id=case_id,
    )
    groups: dict[str, dict[str, Any]] = {}
    vad_groups: dict[str, dict[str, Any]] = {}
    for group_id, result in physical.items():
        score = _score_unit(
            result["final_actions"],
            b0_actions,
            references,
            views,
            scored_start=0,
            scored_end=duration,
            episode_tag="overlap_present",
        )
        entry = {
            "score": score,
            "control_infeasible": int(result.get("infeasible_count", 0)),
            "action_trace_sha256": content_sha256(result["final_actions"]),
            "score_trace_sha256": content_sha256(score),
            "final_action_count": len(result["final_actions"]),
        }
        groups[group_id] = entry
        if group_id.startswith("vad|"):
            vad_groups[group_id] = entry
    failure_signals = _failure_signals(vad_groups, batch.get("b0_clean_gap"), proposals)
    observed_receipt = word_timing_receipt(
        case_id,
        batch["annotation_source_identity"],
        views,
        batch.get("raw_words"),
        batch.get("word_record_sha256", ""),
    )
    case_episode = {
        "case_id": case_id,
        "episode_id": case_id,
        "duration_samples": int(case["duration_samples"]),
        "tag": "overlap_present",
    }
    return {
        "unit_id": case_id,
        "profile_id": profile_id,
        "pool": "historical_validation_corrected_rescore_only",
        "block_id": batch["block_id"],
        "corpus": batch["corpus"],
        "episode_tag": "overlap_present",
        "proposal_count": len(proposals),
        "b0_action_count": len(b0_actions),
        "reference_count": len(references),
        "observed_word_receipt": observed_receipt,
        "failure_signals": failure_signals,
        "groups": _worker_group_summary(
            groups,
            references,
            "historical_validation_corrected_rescore_only",
            case_episode,
        ),
    }


def baseline_batch_worker(batch: dict[str, Any]) -> dict[str, Any]:
    episode = batch["episode"]
    unit_id = str(episode["episode_id"]) if not batch["case_mode"] else str(episode["case_id"])
    b0_actions = batch["b0_actions"]
    b1_actions = batch["b1_actions"]
    references = batch["references"]
    views = batch["views"]
    if batch["case_mode"]:
        scored_start, scored_end, tag = 0, int(episode["duration_samples"]), "overlap_present"
    else:
        scored_start = int(episode["bounds"]["scored_start"])
        scored_end = int(episode["bounds"]["scored_end"])
        tag = str(episode["tag"])
    b0_score = _score_unit(
        b0_actions,
        b0_actions,
        references,
        views,
        scored_start=scored_start,
        scored_end=scored_end,
        episode_tag=tag,
    )
    b1_score = _score_unit(
        b1_actions,
        b0_actions,
        references,
        views,
        scored_start=scored_start,
        scored_end=scored_end,
        episode_tag=tag,
    )
    return {
        "unit_id": unit_id,
        "episode": episode,
        "pool": batch["pool"],
        "block_id": batch["block_id"],
        "case_mode": batch["case_mode"],
        "references": references,
        "b0_actions": b0_actions,
        "b1_actions": b1_actions,
        "b0_score": b0_score,
        "b1_score": b1_score,
        "no_neural_score": b1_score,
    }


# ---------------------------------------------------------------------------
# Interstage gate
# ---------------------------------------------------------------------------


def interstage_gate(experiment_dir: Path) -> dict[str, Any]:
    result = result_dir(experiment_dir)
    stage_a_path = result / STAGE_A_RECEIPT_NAME
    if not stage_a_path.is_file():
        raise Phase5RunError("stage A receipt missing; run stage_a first")
    stage_a = read_json(stage_a_path)
    ledger = json.loads((result / "phase_5_design_ledger.json").read_text(encoding="utf-8"))
    gate_path = result / INTERSTAGE_GATE_NAME
    if gate_path.is_file():
        raise Phase5RunError("interstage gate already evaluated; refusing to re-evaluate")
    gate_contract = ledger["runtime_forecast"]["interstage_exact_cardinality_gate"]
    expected = stage_a["expected_word_timing_receipts"]["receipts"]
    from .phase5_design import validate_interstage_word_timing_receipts

    observed = _observed_word_receipts(experiment_dir)
    receipt = {
        "expected_word_timing_receipts": expected,
        "observed_word_timing_receipts": observed,
    }
    word_check = validate_interstage_word_timing_receipts(gate_contract, receipt)
    storage = ledger["storage_benchmark"]
    projected_bytes = float(storage["projected_result_bytes"])
    forecast = ledger["runtime_forecast"]
    stage_a_elapsed = float(stage_a["elapsed_seconds"])
    forecast_stage_a = float(forecast["stage_a_inference_cache_and_proposal_seconds"])
    total_forecast = float(forecast["total_forecast_seconds"])
    observed_checks = {
        "proposal_execution_count": int(
            stage_a["proposal_execution_receipts"]["row_count"]
        ),
        "logical_route_count": int(stage_a["logical_proposal_routes"]["row_count"]),
        "b0_unit_count": int(stage_a["b0_evidence"]["unit_count"]),
        "current_unit_count": int(stage_a["b0_evidence"]["current_count"]),
        "historical_unit_count": int(stage_a["b0_evidence"]["historical_count"]),
        "b0_b1_mismatch_count": int(stage_a["b0_evidence"]["b0_b1_mismatch_count"]),
        "cache_hit_window_count": int(
            stage_a["phase4_cache_import"]["reusable_window_count"]
        ),
        "cache_miss_window_count": int(
            stage_a["phase5_inference"]["new_inference_window_count"]
        ),
        "stage_a_actual_seconds": stage_a_elapsed,
        "stage_a_forecast_seconds": forecast_stage_a,
        "remaining_total_forecast_seconds": total_forecast - stage_a_elapsed,
        "projected_result_bytes": projected_bytes,
        "result_ceiling_bytes": int(forecast["result_limit_bytes"]),
        "aggregate_json_limit_bytes": int(forecast["aggregate_json_limit_bytes"]),
        "detail_shard_limit_bytes": int(forecast["detail_shard_limit_bytes"]),
        "peak_rss_limit_bytes": int(forecast["peak_rss_limit_bytes"]),
    }
    stop_conditions = {
        "total_forecast_exceeds_3h": total_forecast > 3 * 3600,
        "projected_result_exceeds_8GiB": projected_bytes > forecast["result_limit_bytes"],
        "stage_a_actual_materially_exceeds_forecast": stage_a_elapsed
        > forecast_stage_a * 3,
        "word_timing_receipt_mismatch": not bool(word_check["stage_b_allowed"]),
        "b0_b1_mismatch": int(stage_a["b0_evidence"]["b0_b1_mismatch_count"]) != 0,
        "cardinality_drift": any(
            observed_checks[key] != expected_count
            for key, expected_count in {
                "proposal_execution_count": 3824,
                "logical_route_count": 4328,
                "b0_unit_count": 1082,
                "current_unit_count": 878,
                "historical_unit_count": 204,
            }.items()
        ),
    }
    stage_b_allowed = not any(stop_conditions.values())
    payload = {
        "schema_version": "turn_episode_phase5_interstage_gate.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "design_ledger_content_sha256": ledger["content_sha256"],
        "stage_a_receipt_content_sha256": stage_a["content_sha256"],
        "word_timing_verification": word_check,
        "observed_checks": observed_checks,
        "stop_conditions": stop_conditions,
        "stage_b_allowed": stage_b_allowed,
        "elapsed_seconds": 0,
    }
    written = atomic_write_json(gate_path, payload)
    print(
        canonical_json(
            {
                "path": str(gate_path),
                "content_sha256": written["content_sha256"],
                "stage_b_allowed": stage_b_allowed,
                "word_timing_receipt_count": word_check["word_timing_receipt_count"],
            }
        )
    )
    return written

def _observed_word_receipts(experiment_dir: Path) -> list[dict[str, Any]]:
    result = result_dir(experiment_dir)
    ledger = json.loads((result / "phase_5_design_ledger.json").read_text(encoding="utf-8"))
    inputs = __import__(
        "experiments.speaker_turn_boundary.turn_episode.phase5_design",
        fromlist=["phase4_inputs"],
    ).phase4_inputs(result)
    episodes, _, _ = load_populations(experiment_dir, inputs)
    profiles = proposal_profiles(
        experiment_dir,
        inputs["phase_4_signal_disposition.json"],
        inputs["phase_4_state_equivalence.json"],
    )
    historical_development, _, _ = historical_development_contract(experiment_dir, profiles)
    case_rows = historical_development["case_rows"]
    details = {
        str(row["session_id"]): row
        for row in map(
            json.loads,
            (result / "coverage_inventory_details.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    cases = load_synthetic_cases(experiment_dir / "data" / "manifests")
    receipts = expected_word_timing_receipts(
        experiment_dir, episodes, case_rows, details, cases
    )
    return receipts["receipts"]


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------


def read_wav_slice(path: Path, start: int, end: int) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != 16000 or handle.getsampwidth() != 2:
            raise Phase5RunError(f"unsupported WAV contract: {path}")
        length = handle.getnframes()
        handle.setpos(max(0, min(start, length)))
        frames = handle.readframes(max(0, min(end, length) - max(0, min(start, length))))
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def stage_b(experiment_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    result = result_dir(experiment_dir)
    gate_path = result / INTERSTAGE_GATE_NAME
    if not gate_path.is_file():
        raise Phase5RunError("interstage gate missing; run gate first")
    gate = read_json(gate_path)
    if not gate["stage_b_allowed"]:
        raise Phase5RunError("interstage gate did not allow Stage B")
    stage_a = read_json(result / STAGE_A_RECEIPT_NAME)
    if (result / STAGE_B_RECEIPT_NAME).is_file():
        raise Phase5RunError("stage B receipt already written; refusing to overwrite")
    guard = verify_start_guard(experiment_dir)
    if (
        str(guard["runner_sha256"])
        != str(stage_a["start_guard"]["runner_sha256"])
    ):
        raise Phase5RunError(
            "runner hash drift between stage A and stage B: "
            f"stage_a={stage_a['start_guard']['runner_sha256']} "
            f"current={guard['runner_sha256']}"
        )
    ledger = json.loads((result / "phase_5_design_ledger.json").read_text(encoding="utf-8"))
    inputs = __import__(
        "experiments.speaker_turn_boundary.turn_episode.phase5_design",
        fromlist=["phase4_inputs"],
    ).phase4_inputs(result)
    episodes, _, source_by_episode = load_populations(experiment_dir, inputs)
    profiles = proposal_profiles(
        experiment_dir,
        inputs["phase_4_signal_disposition.json"],
        inputs["phase_4_state_equivalence.json"],
    )
    historical_development, _, _ = historical_development_contract(experiment_dir, profiles)
    case_rows = historical_development["case_rows"]
    manifest = json.loads(
        (experiment_dir / "data" / "manifests" / "mixed_dev_pool.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_cases = {str(row["case_id"]): row for row in manifest["cases"]}
    details = {
        str(row["session_id"]): row
        for row in map(
            json.loads,
            (result / "coverage_inventory_details.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    cases = load_synthetic_cases(experiment_dir / "data" / "manifests")
    inventory = json.loads((result / "coverage_inventory.json").read_text(encoding="utf-8"))
    from .phase4_signal import _source_maps

    sources, _ = _source_maps(experiment_dir, episodes, cases, inventory, details)
    traces = load_proposal_traces()
    b0_receipt = stage_a["b0_evidence_shards"]
    b0_rows = read_representation(result / "phase_5_b0_evidence", b0_receipt)
    b0_by_unit = {str(row["unit_id"]): row for row in b0_rows}
    raw_words_by_session: dict[str, list[Any] | None] = {}
    word_receipts_by_session: dict[str, list[dict[str, Any]]] = {}
    record_sha_by_session: dict[str, str] = {}
    corpus_root = external.corpus_root()
    for session_id in sorted({str(row["session_id"]) for row in episodes}):
        raw_words, receipts, record_sha = load_raw_words(corpus_root, session_id)
        raw_words_by_session[session_id] = raw_words
        word_receipts_by_session[session_id] = receipts
        record_sha_by_session[session_id] = record_sha
    raw_words_by_case: dict[str, list[Any] | None] = {}
    word_receipts_by_case: dict[str, list[dict[str, Any]]] = {}
    record_sha_by_case: dict[str, str] = {}
    for case in case_rows:
        case_id = str(case["case_id"])
        raw_words = None
        receipts: list[dict[str, Any]] = []
        record_sha = ""
        if case_id in HISTORICAL_WORD_FILE_SHA256:
            raw_words, receipts, record_sha = load_raw_words(corpus_root, case_id)
        raw_words_by_case[case_id] = raw_words
        word_receipts_by_case[case_id] = receipts
        record_sha_by_case[case_id] = record_sha
    public_sessions = [
        session_id
        for session_id in raw_words_by_session
        if synthetic_manifest_name(session_id) is None
    ]
    regions_by_session = load_public_regions(
        inventory,
        details,
        public_sessions,
        experiment_dir / "data" / "manifests",
    )
    block_rows = pool_block_index(episodes, inventory)
    block_id_by_pool: dict[tuple[str, str], str] = {}
    session_to_pool_block: dict[tuple[str, str], str] = {}
    session_to_block: dict[tuple[str, str], str] = {}
    for row in block_rows:
        block_id_by_pool[(str(row["pool"]), str(row["statistical_block_id"]))] = str(
            row["pool_block_id"]
        )
        for session_id in row["source_session_ids"]:
            session_to_pool_block[(str(row["pool"]), str(session_id))] = str(
                row["pool_block_id"]
            )
            session_to_block[(str(row["pool"]), str(session_id))] = str(
                row["statistical_block_id"]
            )
    universe = build_system_universe(profiles)
    logical_rows = universe["logical_rows"]
    group_by_node = universe["node_group"]
    current_logical_ids = [str(row[0]) for row in logical_rows]
    logical_groups: dict[tuple[str, str], list[str]] = {}
    for row in logical_rows:
        logical, node = str(row[0]), str(row[1])
        profile_id = str(row[2])
        group = group_by_node.get(node)
        if group is None:
            continue
        logical_groups.setdefault((group, profile_id), []).append(logical)
    historical_logical_ids = []
    for row in logical_rows:
        profile_id = str(row[2])
        if profile_id != "":
            historical_logical_ids.append(str(row[0]))
    baseline_logical_ids = {
        "B0": system_id({"kind": "baseline", "baseline_id": "B0"}),
        "B1": system_id({"kind": "baseline", "baseline_id": "B1"}),
        "no_neural_policy_control": system_id(
            {"kind": "baseline", "baseline_id": "no_neural_policy_control"}
        ),
    }
    workers = int(args.workers or 8)
    batch_list = _current_batches(
        profiles,
        episodes,
        traces,
        b0_by_unit,
        raw_words_by_session,
        word_receipts_by_session,
        record_sha_by_session,
        regions_by_session,
        cases,
        sources,
        session_to_pool_block,
        session_to_block,
    )
    agg_pool: dict[tuple[str, str], dict[str, Any]] = {}
    agg_block: dict[tuple[str, str, str], dict[str, Any]] = {}
    baseline_batches = _baseline_batches(
        episodes,
        case_rows,
        b0_by_unit,
        raw_words_by_session,
        raw_words_by_case,
        regions_by_session,
        cases,
        manifest_cases,
        session_to_pool_block,
        session_to_block,
    )
    baseline_scores = _run_baseline_scores(baseline_batches, workers)
    _accumulate_current_baselines(
        baseline_scores,
        agg_pool,
        agg_block,
        baseline_logical_ids,
    )
    failure_heaps: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    observed_receipts: dict[str, dict[str, Any]] = {}
    processed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(current_batch_worker, batch): index
            for index, batch in enumerate(batch_list)
        }
        for future in as_completed(futures):
            outcome = future.result()
            _accumulate_current(
                outcome,
                agg_pool,
                agg_block,
                logical_groups,
                failure_heaps,
                workers,
            )
            observed_receipts[str(outcome["observed_word_receipt"]["unit_id"])] = outcome[
                "observed_word_receipt"
            ]
            processed += 1
            if processed % 200 == 0:
                print(f"phase5 stage_b current batches={processed}/{len(batch_list)}", flush=True)
    historical_batches = _historical_batches(
        profiles,
        case_rows,
        traces,
        b0_by_unit,
        raw_words_by_case,
        cases,
        manifest_cases,
        word_receipts_by_case,
        record_sha_by_case,
        sources,
    )
    hist_agg: dict[str, dict[str, Any]] = {}
    hist_processed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(historical_batch_worker, batch): index
            for index, batch in enumerate(historical_batches)
        }
        for future in as_completed(futures):
            outcome = future.result()
            _accumulate_historical(
                outcome,
                hist_agg,
                logical_groups,
                failure_heaps,
            )
            observed_receipts[str(outcome["observed_word_receipt"]["unit_id"])] = outcome[
                "observed_word_receipt"
            ]
            hist_processed += 1
    expected_ids = [str(row["unit_id"]) for row in stage_a["expected_word_timing_receipts"]["receipts"]]
    observed_ids = sorted(str(row["unit_id"]) for row in observed_receipts.values())
    if observed_ids != expected_ids:
        raise Phase5RunError("observed word timing unit order drift")
    observed_receipt_rows = [
        observed_receipts[str(row["unit_id"])]
        for row in stage_a["expected_word_timing_receipts"]["receipts"]
    ]
    if canonical_json(observed_receipt_rows) != canonical_json(
        stage_a["expected_word_timing_receipts"]["receipts"]
    ):
        for e, o in zip(
            stage_a["expected_word_timing_receipts"]["receipts"], observed_receipt_rows
        ):
            if e != o:
                print("DRIFT", e["unit_id"], flush=True)
                for k in sorted(set(e) | set(o)):
                    if e.get(k) != o.get(k):
                        print(
                            "  field:", k,
                            "expected:", json.dumps(e.get(k))[:300],
                            "observed:", json.dumps(o.get(k))[:300],
                            flush=True,
                        )
        raise Phase5RunError("observed word timing receipts drifted from expected")
    block_order = [
        (str(row["pool"]), str(row["pool_block_id"])) for row in block_rows
    ]
    natural_manifest = json.loads(
        (result / "natural_exposure_manifest.json").read_text(encoding="utf-8")
    )
    natural_eligible_source_samples = (
        int(natural_manifest["window_frame"]["eligible_duration_ms"]) * 16
    )
    block_eligible_by_key: dict[tuple[str, str], int] = {}
    if sources:
        eligible_ms_by_session: dict[str, int] = {}
        for window in inventory["natural_exposure"]["windows"]:
            session_id = str(window["session_id"])
            eligible_ms_by_session[session_id] = (
                eligible_ms_by_session.get(session_id, 0)
                + int(window["eligible_duration_ms"])
            )
        for row in block_rows:
            if str(row["pool"]) != "natural_exposure_validation":
                continue
            eligible = sum(
                eligible_ms_by_session.get(str(session_id), 0) * 16
                for session_id in row["source_session_ids"]
            )
            block_eligible_by_key[(str(row["pool"]), str(row["pool_block_id"]))] = (
                eligible
            )
    current_agg_rows = _finalize_current_aggregates(
        logical_rows,
        agg_pool,
        agg_block,
        universe,
        block_order,
        natural_eligible_source_samples,
        block_eligible_by_key,
    )
    historical_baseline = _historical_baseline_aggregates(baseline_scores)
    historical_agg_rows = _finalize_historical_aggregates(
        historical_logical_ids,
        baseline_logical_ids,
        hist_agg,
        b0_by_unit,
        historical_baseline,
        str(stage_a["b0_evidence"]["equivalence_receipts_sha256"]),
    )
    failure_rows = _select_failure_examples(failure_heaps, profiles)
    audit_rows = _build_audit_index(
        profiles,
        episodes,
        case_rows,
        current_logical_ids,
        historical_logical_ids,
        failure_rows,
        universe["system_info"],
        gate,
    )
    result_dirs = {
        "phase_5_physical_systems": ("physical_system_definition", universe["physical_rows"]),
        "phase_5_logical_systems": ("logical_system_definition", logical_rows),
        "phase_5_alias_edges": ("logical_alias_edge", universe["alias_rows"]),
        "phase_5_current_aggregates": ("current_system_block_aggregate", current_agg_rows),
        "phase_5_historical_aggregates": (
            "historical_corrected_system_aggregate",
            historical_agg_rows,
        ),
        "phase_5_failure_examples": ("deterministic_failure_example", failure_rows),
        "phase_5_audit_units": ("independent_audit_unit", audit_rows),
    }
    receipts = {}
    for directory_name, (representation, rows) in result_dirs.items():
        writer = RepresentationWriter(result / directory_name, representation)
        if rows and isinstance(rows[0], dict):
            keyed_rows = (
                (
                    f"{str(row['proposal_profile_id'])}|{str(row['corpus'])}|{str(row['category'])}|{int(row['rank']):02d}",
                    row,
                )
                for row in rows
            )
        else:
            keyed_rows = ((str(row[0]), row) for row in rows)
        writer.add_rows(keyed_rows)
        receipt = writer.write()
        verify_shard_receipts(result / directory_name, receipt["shards"])
        receipts[representation] = receipt
    payload = {
        "schema_version": "turn_episode_phase5_stage_b.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "design_ledger_content_sha256": ledger["content_sha256"],
        "interstage_gate_content_sha256": gate["content_sha256"],
        "start_guard": guard,
        "workers": workers,
        "current_batch_count": len(batch_list),
        "historical_batch_count": len(historical_batches),
        "observed_word_timing_receipt_count": len(observed_receipts),
        "aggregate_shard_receipts": receipts,
        "failure_example_count": len(failure_rows),
        "audit_unit_count": len(audit_rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    written = atomic_write_json(result / STAGE_B_RECEIPT_NAME, payload)
    print(
        canonical_json(
            {
                "path": str(result / STAGE_B_RECEIPT_NAME),
                "content_sha256": written["content_sha256"],
                "current_batches": len(batch_list),
                "historical_batches": len(historical_batches),
                "failure_examples": len(failure_rows),
                "audit_units": len(audit_rows),
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )
    )
    return written

# ---------------------------------------------------------------------------
# Stage B batch construction and aggregation helpers
# ---------------------------------------------------------------------------


def _find_route(
    traces: dict[str, dict[str, Any]],
    profile_id: str,
    unit_id: str,
    *,
    historical: bool,
) -> dict[str, Any] | None:
    for execution in traces.values():
        if str(execution.get("proposal_profile_id")) != profile_id:
            continue
        routes = execution.get("routes") or {}
        if unit_id in routes:
            return routes[unit_id]
    return None


def _current_batches(
    profiles: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    b0_by_unit: dict[str, dict[str, Any]],
    raw_words_by_session: dict[str, list[Any] | None],
    word_receipts_by_session: dict[str, list[dict[str, Any]]],
    record_sha_by_session: dict[str, str],
    regions_by_session: dict[str, list[Any]],
    cases: dict[tuple[str, str], dict[str, Any]],
    sources: dict[str, AudioSource],
    session_to_pool_block: dict[str, str],
    session_to_block: dict[str, str],
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for profile in profiles:
        profile_id = str(profile["proposal_profile_id"])
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            route = _find_route(traces, profile_id, episode_id, historical=False)
            if route is None:
                raise Phase5RunError(f"proposal route missing: {profile_id}:{episode_id}")
            b0 = b0_by_unit[episode_id]
            session_id = str(episode["session_id"])
            raw_words = raw_words_by_session.get(session_id)
            views = annotation_views_for_episode(
                episode, regions_by_session, cases, raw_words_by_session
            )
            source = sources[session_id]
            bounds = episode["bounds"]
            wave_start = max(0, int(bounds["warm_start"]) - 8000)
            wave_end = min(int(source.duration_samples), int(bounds["tail_end"]) + 8000)
            batches.append(
                {
                    "profile": profile,
                    "episode": episode,
                    "proposals": route["proposals"],
                    "b0_actions": b0["b0_actions"],
                    "lifecycle": b0["lifecycle_events"],
                    "references": episode["references"],
                    "views": views,
                    "raw_words": raw_words,
                    "word_record_sha256": record_sha_by_session.get(session_id, ""),
                    "annotation_source_identity": {
                        "session_id": session_id,
                        "annotation_sha256": str(
                            episode.get("annotation_sha256") or ""
                        ),
                        "word_annotation_files": word_receipts_by_session.get(session_id, []),
                    },
                    "waveform": None,
                    "wav_path": str(source.path),
                    "wave_start": wave_start,
                    "wave_end": wave_end,
                    "block_id": session_to_pool_block.get(
                        (str(episode["pool"]), session_id), ""
                    ),
                    "b0_clean_gap": None,
                }
            )
    return batches


def _historical_batches(
    profiles: Sequence[dict[str, Any]],
    case_rows: Sequence[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    b0_by_unit: dict[str, dict[str, Any]],
    raw_words_by_case: dict[str, list[Any] | None],
    cases: dict[tuple[str, str], dict[str, Any]],
    manifest_cases: dict[str, dict[str, Any]],
    word_receipts_by_case: dict[str, list[dict[str, Any]]],
    record_sha_by_case: dict[str, str],
    sources: dict[str, AudioSource],
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    source_by_wav_sha: dict[str, AudioSource] = {}
    for source in sources.values():
        source_by_wav_sha.setdefault(str(source.wav_sha256), source)
    for profile in profiles:
        profile_id = str(profile["proposal_profile_id"])
        for case in case_rows:
            case_id = str(case["case_id"])
            route = _find_route(traces, profile_id, case_id, historical=True)
            if route is None:
                raise Phase5RunError(f"historical proposal route missing: {profile_id}:{case_id}")
            b0 = b0_by_unit[case_id]
            raw_words = raw_words_by_case.get(case_id)
            manifest_case = manifest_cases.get(case_id)
            if manifest_case is None:
                raise Phase5RunError(f"historical batch manifest case missing: {case_id}")
            views = annotation_views_for_case(manifest_case, raw_words)
            source = source_by_wav_sha.get(str(case["wav_sha256"]))
            wav_path = ""
            wave_start = 0
            wave_end = int(case["duration_samples"])
            if source is not None:
                wav_path = str(source.path)
                wave_end = min(int(source.duration_samples), int(case["duration_samples"]))
            batches.append(
                {
                    "profile": profile,
                    "case": case,
                    "proposals": route["proposals"],
                    "b0_actions": b0["b0_actions"],
                    "lifecycle": b0["lifecycle_events"],
                    "references": views["_references"],
                    "views": views,
                    "raw_words": raw_words,
                    "word_record_sha256": record_sha_by_case.get(case_id, ""),
                    "annotation_source_identity": {
                        "manifest_byte_sha256": HISTORICAL_MANIFEST_BYTE_SHA256,
                        "word_annotation_files": word_receipts_by_case.get(case_id, []),
                    },
                    "waveform": None,
                    "wav_path": wav_path,
                    "wave_start": wave_start,
                    "wave_end": wave_end,
                    "block_id": "historical",
                    "corpus": _case_corpus(case_id),
                    "b0_clean_gap": None,
                }
            )
    return batches


def _case_corpus(case_id: str) -> str:
    if case_id.startswith("ami_"):
        return "ami"
    return "synthetic"


def _block_b0_field(field: str) -> str:
    mapping = {
        "clean_gap_contaminated_samples": "b0_clean_gap_contaminated_samples",
        "harmful_active_split_count_200ms": "b0_harmful_active_split_count",
    }
    return mapping.get(field, field)


def _accumulate_current_baselines(
    baseline_scores: dict[str, dict[str, Any]],
    agg_pool: dict[tuple[str, str], dict[str, Any]],
    agg_block: dict[tuple[str, str, str], dict[str, Any]],
    baseline_logical_ids: dict[str, str],
) -> None:
    for unit_id, score in baseline_scores.items():
        if score["case_mode"]:
            continue
        pool = str(score["pool"])
        block_id = str(score["block_id"])
        session_id = str(score.get("session_id", ""))
        for baseline_id in ("B0", "B1"):
            system = baseline_logical_ids[baseline_id]
            if baseline_id == "B0":
                metrics = score["b0_metrics"]
                durations = score["b0_durations"]
                active_durations = score["b0_active_durations"]
                deadline_views = score["b0_deadline_views"]
                action_trace = score["b0_action_trace_sha256"]
                score_trace = score["b0_score_trace_sha256"]
            else:
                metrics = score["b1_metrics"]
                durations = score["b1_durations"]
                active_durations = score["b1_active_durations"]
                deadline_views = score["b1_deadline_views"]
                action_trace = score["b1_action_trace_sha256"]
                score_trace = score["b1_score_trace_sha256"]
            key = (system, pool)
            entry = agg_pool.setdefault(
                key,
                {
                    "metrics": {field: 0 for field in SYSTEM_METRIC_FIELDS},
                    "durations": [],
                    "active_durations": [],
                    "action_digest": hashlib.sha256(),
                    "score_digest": hashlib.sha256(),
                    "sessions": set(),
                    "deadline_views": Counter(),
                },
            )
            for field in SYSTEM_METRIC_FIELDS:
                entry["metrics"][field] += int(metrics[field])
            entry["durations"].extend(durations)
            entry["active_durations"].extend(active_durations)
            _digest_row(entry["action_digest"], action_trace)
            _digest_row(entry["score_digest"], score_trace)
            entry["sessions"].add(session_id)
            block_key = (system, pool, block_id)
            block_entry = agg_block.setdefault(
                block_key,
                {
                    "fields": {field: 0 for field in BLOCK_METRIC_FIELDS},
                    "deadline_views": Counter(),
                },
            )
            for field in BLOCK_METRIC_FIELDS:
                if field.startswith("hard_match_") and field.endswith("ms_count"):
                    continue
                if field in (
                    "b0_clean_gap_contaminated_samples",
                    "b1_clean_gap_contaminated_samples",
                    "b0_harmful_active_split_count",
                    "b1_harmful_active_split_count",
                ):
                    continue
                block_entry["fields"][field] += int(
                    metrics.get(_metric_for_block(field), 0)
                )
            for deadline, count in (deadline_views or {}).items():
                block_entry["deadline_views"][str(deadline)] += int(count)


def _accumulate_current(
    outcome: dict[str, Any],
    agg_pool: dict[tuple[str, str], dict[str, Any]],
    agg_block: dict[tuple[str, str, str], dict[str, Any]],
    logical_groups: dict[tuple[str, str], list[str]],
    failure_heaps: dict[tuple[str, str, str], list[dict[str, Any]]],
    workers: int,
) -> None:
    pool = str(outcome["pool"])
    block_id = str(outcome["block_id"])
    episode_id = str(outcome["unit_id"])
    profile_id = str(outcome["profile_id"])
    corpus = str(outcome["corpus"])
    for group_id, group in outcome["groups"].items():
        for system_id in logical_groups.get((group_id, profile_id), []):
            key = (system_id, pool)
            entry = agg_pool.setdefault(
                key,
                {
                    "metrics": {field: 0 for field in SYSTEM_METRIC_FIELDS},
                    "durations": [],
                    "active_durations": [],
                    "action_digest": hashlib.sha256(),
                    "score_digest": hashlib.sha256(),
                    "sessions": set(),
                    "deadline_views": Counter(),
                },
            )
            metrics = group["metrics"]
            for field in SYSTEM_METRIC_FIELDS:
                entry["metrics"][field] += int(metrics[field])
            entry["durations"].extend(group.get("segment_durations") or [])
            entry["active_durations"].extend(group.get("active_segment_durations") or [])
            _digest_row(entry["action_digest"], group["action_trace_sha256"])
            _digest_row(entry["score_digest"], group["score_trace_sha256"])
            entry["sessions"].add(str(outcome.get("session_id", "")))
            block_key = (system_id, pool, block_id)
            block_entry = agg_block.setdefault(
                block_key,
                {
                    "fields": {field: 0 for field in BLOCK_METRIC_FIELDS},
                    "deadline_views": Counter(),
                },
            )
            for field in BLOCK_METRIC_FIELDS:
                if field.startswith("hard_match_") and field.endswith("ms_count"):
                    continue
                if field in (
                    "b0_clean_gap_contaminated_samples",
                    "b1_clean_gap_contaminated_samples",
                    "b0_harmful_active_split_count",
                    "b1_harmful_active_split_count",
                ):
                    continue
                block_entry["fields"][field] += int(
                    metrics.get(_metric_for_block(field), 0)
                )
            for deadline, count in (group.get("deadline_views") or {}).items():
                block_entry["deadline_views"][str(deadline)] += int(count)
    for category, signal in (outcome.get("failure_signals") or {}).items():
        heap_key = (profile_id, corpus, category)
        heap = failure_heaps.setdefault(heap_key, [])
        heap.append(
            {
                "value": float(signal["value"]),
                "episode_id": episode_id,
                "source_session_id": str(outcome.get("session_id", "")),
                "boundary": int(signal["boundary"]),
                "action_trace_sha256": str(signal["action_trace_sha256"]),
                "score_evidence_sha256": str(signal["score_evidence_sha256"]),
            }
        )
        heap.sort(
            key=lambda row: (
                -float(row["value"]),
                str(row["source_session_id"]),
                str(row["episode_id"]),
                int(row["boundary"]),
            )
        )
        del heap[5:]


def _metric_for_block(field: str) -> str:
    mapping = {
        "episode_count": "episode_count",
        "clean_gap_singleton_denominator_samples": "clean_gap_singleton_denominator_samples",
        "candidate_clean_gap_contaminated_samples": "clean_gap_contaminated_samples",
        "candidate_harmful_active_split_count": "harmful_active_split_count_200ms",
        "detector_created_hard_action_count": "detector_created_hard_action_count",
        "same_speaker_extra_turn_count": "same_speaker_extra_turn_count",
        "lexical_split_count": "lexical_split_count",
        "lexical_observable_action_count": "lexical_observable_action_count",
        "duplicate_hard_boundary_count": "duplicate_hard_boundary_count",
        "hard_target_count": "clean_gap_hard_target_count",
        "availability_delay_sum_samples": "availability_delay_sum_samples",
        "availability_delay_count": "availability_delay_count",
        "overlap_hard_action_count": "overlap_hard_action_count",
        "overlap_contribution_samples": "overlap_contribution_samples",
        "sampled_source_samples": "natural_sampled_source_samples",
        "sampled_active_speech_samples": "natural_sampled_active_speech_samples",
        "natural_exposure_eligible_source_samples": "natural_eligible_source_samples",
    }
    return mapping.get(field, field)


def _digest_row(digest: Any, value: str) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _accumulate_historical(
    outcome: dict[str, Any],
    hist_agg: dict[str, dict[str, Any]],
    logical_groups: dict[tuple[str, str], list[str]],
    failure_heaps: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> None:
    profile_id = str(outcome["profile_id"])
    case_id = str(outcome["unit_id"])
    corpus = str(outcome["corpus"])
    for group_id, group in outcome["groups"].items():
        for system_id in logical_groups.get((group_id, profile_id), []):
            entry = hist_agg.setdefault(
                system_id,
                {
                    "metrics": {field: 0 for field in SYSTEM_METRIC_FIELDS},
                    "case_ids": [],
                    "action_digest": hashlib.sha256(),
                    "score_digest": hashlib.sha256(),
                    "identity_digest": hashlib.sha256(),
                    "durations": [],
                    "active_durations": [],
                },
            )
            metrics = group["metrics"]
            for field in SYSTEM_METRIC_FIELDS:
                entry["metrics"][field] += int(metrics[field])
            entry["case_ids"].append(case_id)
            entry["durations"].extend(group.get("segment_durations") or [])
            entry["active_durations"].extend(group.get("active_segment_durations") or [])
            _digest_row(entry["identity_digest"], f"{system_id}|{case_id}")
            _digest_row(entry["action_digest"], group["action_trace_sha256"])
            _digest_row(entry["score_digest"], group["score_trace_sha256"])
    for category, signal in (outcome.get("failure_signals") or {}).items():
        heap_key = (profile_id, corpus, category)
        heap = failure_heaps.setdefault(heap_key, [])
        heap.append(
            {
                "value": float(signal["value"]),
                "episode_id": case_id,
                "source_session_id": case_id,
                "boundary": int(signal["boundary"]),
                "action_trace_sha256": str(signal["action_trace_sha256"]),
                "score_evidence_sha256": str(signal["score_evidence_sha256"]),
            }
        )
        heap.sort(
            key=lambda row: (
                -float(row["value"]),
                str(row["source_session_id"]),
                str(row["episode_id"]),
                int(row["boundary"]),
            )
        )
        del heap[5:]


def _finalize_current_aggregates(
    logical_rows: Sequence[Sequence[Any]],
    agg_pool: dict[tuple[str, str], dict[str, Any]],
    agg_block: dict[tuple[str, str, str], dict[str, Any]],
    universe: dict[str, Any],
    block_order: Sequence[tuple[str, str]],
    natural_eligible_source_samples: int,
    block_eligible_by_key: dict[tuple[str, str], int],
) -> list[list[Any]]:
    node_by_system = {str(row[0]): str(row[1]) for row in logical_rows}
    rows: list[list[Any]] = []
    for system_id, node in sorted(node_by_system.items()):
        pool_rows: list[list[Any]] = []
        for pool in POOL_ORDER:
            entry = agg_pool.get((system_id, pool))
            if entry is None:
                metrics = [0 for _ in SYSTEM_METRIC_FIELDS]
                action_digest = ""
                score_digest = ""
            else:
                metrics = [
                    int(entry["metrics"][field]) for field in SYSTEM_METRIC_FIELDS
                ]
                _apply_percentiles_from_durations(
                    metrics, entry["durations"], entry["active_durations"]
                )
                if pool == "natural_exposure_validation":
                    metrics[SYSTEM_METRIC_FIELDS.index("natural_session_count")] = len(
                        {str(session_id) for session_id in entry["sessions"] if session_id}
                    )
                    metrics[
                        SYSTEM_METRIC_FIELDS.index("natural_eligible_source_samples")
                    ] = natural_eligible_source_samples
                action_digest = entry["action_digest"].hexdigest()
                score_digest = entry["score_digest"].hexdigest()
            pool_rows.append([pool, action_digest, score_digest, *metrics])
        block_rows: list[list[Any]] = []
        for pool, pool_block_id in block_order:
            entry = agg_block.get((system_id, pool, pool_block_id))
            if entry is None:
                values = [0 for _ in BLOCK_METRIC_FIELDS]
            else:
                deadline = entry["deadline_views"]
                values = []
                for field in BLOCK_METRIC_FIELDS:
                    if field.startswith("hard_match_") and field.endswith("ms_count"):
                        deadline_ms = field.removeprefix("hard_match_").removesuffix(
                            "ms_count"
                        )
                        values.append(int(deadline.get(deadline_ms, 0)))
                    elif field == "natural_exposure_eligible_source_samples":
                        values.append(
                            block_eligible_by_key.get((pool, pool_block_id), 0)
                        )
                    else:
                        values.append(int(entry["fields"][field]))
            block_rows.append([pool, pool_block_id, *values])
        rows.append([system_id, node, pool_rows, block_rows])
    return rows


def _finalize_historical_aggregates(
    historical_logical_ids: Sequence[str],
    baseline_logical_ids: dict[str, str],
    hist_agg: dict[str, dict[str, Any]],
    b0_by_unit: dict[str, dict[str, Any]],
    historical_baseline: dict[str, dict[str, Any]],
    b0_b1_equivalence_receipt_sha256: str,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for system_id in sorted(historical_logical_ids):
        entry = hist_agg.get(system_id)
        if entry is None:
            metrics = [0 for _ in SYSTEM_METRIC_FIELDS]
            identity_digest = ""
            action_digest = ""
            score_digest = ""
            case_count = 0
        else:
            metrics = [int(entry["metrics"][field]) for field in SYSTEM_METRIC_FIELDS]
            _apply_percentiles_from_durations(
                metrics, entry["durations"], entry["active_durations"]
            )
            identity_digest = entry["identity_digest"].hexdigest()
            action_digest = entry["action_digest"].hexdigest()
            score_digest = entry["score_digest"].hexdigest()
            case_count = len(entry["case_ids"])
        rows.append(
            [
                system_id,
                "neural_policy",
                "historical_validation_corrected_rescore_only",
                case_count,
                case_count,
                identity_digest,
                action_digest,
                score_digest,
                b0_b1_equivalence_receipt_sha256,
                *metrics,
            ]
        )
    for baseline_id in ("B0", "B1"):
        system = baseline_logical_ids[baseline_id]
        entry = historical_baseline.get(baseline_id)
        if entry is None:
            metrics = [0 for _ in SYSTEM_METRIC_FIELDS]
            action_digest = ""
            score_digest = ""
            case_count = 0
        else:
            metrics = [int(entry["metrics"][field]) for field in SYSTEM_METRIC_FIELDS]
            _apply_percentiles_from_durations(
                metrics, entry["durations"], entry["active_durations"]
            )
            action_digest = entry["action_digest"]
            score_digest = entry["score_digest"]
            case_count = len(entry["case_ids"])
        identity_rows = [
            f"{baseline_id}|{case_id}" for case_id in entry["case_ids"]
        ] if entry is not None else []
        identity_digest = framed_digest(identity_rows)
        rows.append(
            [
                system,
                "baseline",
                "historical_validation_corrected_rescore_only",
                case_count,
                case_count,
                identity_digest,
                action_digest,
                score_digest,
                b0_b1_equivalence_receipt_sha256,
                *metrics,
            ]
        )
    rows.sort(key=lambda row: str(row[0]))
    return rows


def _select_failure_examples(
    failure_heaps: dict[tuple[str, str, str], list[dict[str, Any]]],
    profiles: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profile_ids = [str(row["proposal_profile_id"]) for row in profiles]
    for profile_id in profile_ids:
        for corpus in ("ami", "alimeeting", "synthetic"):
            for category in FAILURE_CATEGORIES:
                heap = failure_heaps.get((profile_id, corpus, category), [])
                for rank in range(5):
                    if rank < len(heap):
                        candidate = heap[rank]
                        rows.append(
                            {
                                "proposal_profile_id": profile_id,
                                "corpus": corpus,
                                "category": category,
                                "rank": rank,
                                "source_session_id": str(candidate["source_session_id"]),
                                "episode_id": str(candidate["episode_id"]),
                                "boundary_source_sample": int(candidate["boundary"]),
                                "action_trace_sha256": str(
                                    candidate["action_trace_sha256"]
                                ),
                                "score_evidence_sha256": str(
                                    candidate["score_evidence_sha256"]
                                ),
                            }
                        )
                    else:
                        rows.append(
                            {
                                "proposal_profile_id": profile_id,
                                "corpus": corpus,
                                "category": category,
                                "rank": rank,
                                "source_session_id": "",
                                "episode_id": "",
                                "boundary_source_sample": 0,
                                "action_trace_sha256": "",
                                "score_evidence_sha256": "",
                            }
                        )
    if len(rows) != FAILURE_EXAMPLES:
        raise Phase5RunError(f"failure example count drift: {len(rows)}")
    rows.sort(
        key=lambda row: (
            str(row["proposal_profile_id"]),
            str(row["corpus"]),
            str(row["category"]),
            int(row["rank"]),
        )
    )
    return rows


def _build_audit_index(
    profiles: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    case_rows: Sequence[dict[str, Any]],
    current_logical_ids: Sequence[str],
    historical_logical_ids: Sequence[str],
    failure_rows: Sequence[dict[str, Any]],
    system_info: dict[str, dict[str, Any]],
    gate: dict[str, Any],
) -> list[list[Any]]:
    current_units = {str(row["episode_id"]) for row in episodes}
    historical_units = {str(row["case_id"]) for row in case_rows}
    unit_by_id: dict[str, dict[str, Any]] = {}
    for sid in current_logical_ids:
        info = system_info[sid]
        for episode in episodes:
            unit_id = str(episode["episode_id"])
            key = f"{sid}:{unit_id}"
            unit_by_id[key] = {
                "canonical_unit_id": key,
                "population": "current",
                "proposal_profile_id": info["profile_id"],
                "policy_class": info.get("policy_class", ""),
                "pool": str(episode["pool"]),
                "corpus": corpus_for(str(episode["session_id"])),
                "ladder_stage": info["stage"],
                "fusion_mode": info.get("chain"),
                "control_kind": info.get("control_kind"),
                "selection_reason": "hash_fill_candidate",
            }
    for sid in historical_logical_ids:
        info = system_info[sid]
        for case in case_rows:
            case_id = str(case["case_id"])
            key = f"{sid}:{case_id}"
            unit_by_id[key] = {
                "canonical_unit_id": key,
                "population": "historical_validation",
                "proposal_profile_id": info["profile_id"],
                "policy_class": info.get("policy_class", ""),
                "pool": "historical_validation_corrected_rescore_only",
                "corpus": _case_corpus(case_id),
                "ladder_stage": info["stage"],
                "fusion_mode": info.get("chain"),
                "control_kind": info.get("control_kind"),
                "selection_reason": "hash_fill_candidate",
            }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_unit(unit: dict[str, Any], reason: str) -> None:
        key = str(unit["canonical_unit_id"])
        if key in seen:
            return
        seen.add(key)
        row = dict(unit)
        row["selection_reason"] = reason
        selected.append(row)

    for failure in failure_rows:
        episode_id = str(failure["episode_id"])
        if not episode_id:
            continue
        unit_id = (
            episode_id
            if episode_id in current_units
            else (episode_id if episode_id in historical_units else None)
        )
        if unit_id is None:
            continue
        profile_id = str(failure["proposal_profile_id"])
        representative = system_id(
            logical_ladder_key(profile_id, FULL_GRID[0], "full_hard_soft_fusion")
        )
        key = f"{representative}:{unit_id}"
        if key in unit_by_id:
            add_unit(unit_by_id[key], "deterministic_failure_example")
    first_episode = sorted(episodes, key=lambda row: str(row["episode_id"]))[0][
        "episode_id"
    ]
    first_case = sorted(case_rows, key=lambda row: str(row["case_id"]))[0]["case_id"]
    for baseline_id in ("B0", "B1", "no_neural_policy_control"):
        system = system_id({"kind": "baseline", "baseline_id": baseline_id})
        key = f"{system}:{first_episode}"
        if key in unit_by_id:
            add_unit(unit_by_id[key], "mandatory_baseline_sentinel")
        if baseline_id != "no_neural_policy_control":
            key = f"{system}:{first_case}"
            if key in unit_by_id:
                add_unit(unit_by_id[key], "mandatory_baseline_sentinel")
    for info in system_info.values():
        profile_id = str(info["profile_id"])
        if not profile_id:
            continue
        stage = str(info["stage"])
        if stage in LADDER_STAGES:
            key = f"{system_id(logical_ladder_key(profile_id, FULL_GRID[0], stage))}:{first_episode}"
            if key in unit_by_id:
                add_unit(unit_by_id[key], "mandatory_ladder_stage_sentinel")
        if info.get("control_kind"):
            key = f"{system_id(logical_control_key(profile_id, FULL_GRID[0], info['control_kind']))}:{first_episode}"
            if key in unit_by_id:
                add_unit(unit_by_id[key], "mandatory_control_kind_sentinel")
    ordered = sorted(
        unit_by_id.values(),
        key=lambda unit: hashlib.sha256(
            (f"turn-episode-v1-phase5-audit-v1|{unit['canonical_unit_id']}").encode(
                "utf-8"
            )
        ).hexdigest(),
    )
    for unit in ordered:
        if len(selected) >= INDEPENDENT_AUDIT_SAMPLE_SIZE:
            break
        add_unit(unit, "stratified_hash_fill")
    if len(selected) != INDEPENDENT_AUDIT_SAMPLE_SIZE:
        raise Phase5RunError(
            f"audit unit count drift: {len(selected)} != {INDEPENDENT_AUDIT_SAMPLE_SIZE}"
        )
    rows: list[list[Any]] = []
    for unit in selected:
        canonical = str(unit["canonical_unit_id"])
        selection_sha = hashlib.sha256(
            f"turn-episode-v1-phase5-audit-v1|{canonical}".encode("utf-8")
        ).hexdigest()
        rows.append(
            [
                "audit:" + hashlib.sha256((canonical + "|" + selection_sha).encode("utf-8")).hexdigest()[:32],
                canonical,
                selection_sha,
                str(unit["selection_reason"]),
                str(unit["population"]),
                str(unit["proposal_profile_id"]),
                str(unit["policy_class"]),
                str(unit["pool"]),
                str(unit["corpus"]),
                str(unit["ladder_stage"]),
                _fusion_mode_label(unit["fusion_mode"]),
                str(unit["control_kind"] or ""),
                "",
                "",
                "pending_recompute",
            ]
        )
    rows.sort(key=lambda row: str(row[0]))
    return rows


def _fusion_mode_label(chain: dict[str, Any] | None) -> str:
    if chain is None:
        return ""
    return (
        f"v{chain['detector_vad_radius_ms']}:s{int(chain['same_silence_interval_association'])}"
    )


def _sentinel_units(system_info: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    sentinels: dict[str, list[str]] = {}
    baseline_ids = {
        "B0": system_id({"kind": "baseline", "baseline_id": "B0"}),
        "B1": system_id({"kind": "baseline", "baseline_id": "B1"}),
        "no_neural_policy_control": system_id(
            {"kind": "baseline", "baseline_id": "no_neural_policy_control"}
        ),
    }
    for info in system_info.values():
        profile_id = str(info["profile_id"])
        if not profile_id:
            continue
        policy_class = str(info.get("policy_class", ""))
        stage = str(info["stage"])
        for unit_marker in ("",):
            if stage in LADDER_STAGES:
                sentinels.setdefault(f"mandatory_ladder_sentinel|{profile_id}|{stage}", [])
            if info.get("control_kind"):
                sentinels.setdefault(
                    f"mandatory_control_sentinel|{profile_id}|{info['control_kind']}", []
                )
            break
    return sentinels

def _baseline_batches(
    episodes: Sequence[dict[str, Any]],
    case_rows: Sequence[dict[str, Any]],
    b0_by_unit: dict[str, dict[str, Any]],
    raw_words_by_session: dict[str, list[Any] | None],
    raw_words_by_case: dict[str, list[Any] | None],
    regions_by_session: dict[str, list[Any]],
    cases: dict[tuple[str, str], dict[str, Any]],
    manifest_cases: dict[str, dict[str, Any]],
    session_to_pool_block: dict[str, str],
    session_to_block: dict[str, str],
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        b0 = b0_by_unit[episode_id]
        session_id = str(episode["session_id"])
        raw_words = raw_words_by_session.get(session_id)
        views = annotation_views_for_episode(
            episode, regions_by_session, cases, raw_words_by_session
        )
        batches.append(
            {
                "episode": episode,
                "case_mode": False,
                "b0_actions": b0["b0_actions"],
                "b1_actions": b0["b1_actions"],
                "references": episode["references"],
                "views": views,
                "pool": str(episode["pool"]),
                "block_id": session_to_pool_block.get(
                    (str(episode["pool"]), session_id), ""
                ),
            }
        )
    for case in case_rows:
        case_id = str(case["case_id"])
        b0 = b0_by_unit[case_id]
        raw_words = raw_words_by_case.get(case_id)
        manifest_case = manifest_cases.get(case_id)
        if manifest_case is None:
            raise Phase5RunError(f"baseline manifest case missing: {case_id}")
        views = annotation_views_for_case(manifest_case, raw_words)
        references = views.pop("_references", [])
        case_episode = {
            "case_id": case_id,
            "episode_id": case_id,
            "duration_samples": int(case["duration_samples"]),
            "bounds": {
                "scored_start": 0,
                "scored_end": int(case["duration_samples"]),
                "warm_start": 0,
                "tail_end": int(case["duration_samples"]),
            },
            "tag": "overlap_present",
        }
        batches.append(
            {
                "episode": case_episode,
                "case_mode": True,
                "b0_actions": b0["b0_actions"],
                "b1_actions": b0["b1_actions"],
                "references": references,
                "views": views,
                "pool": "historical_validation_corrected_rescore_only",
                "block_id": "historical",
            }
        )
    return batches


def _run_baseline_scores(
    batches: Sequence[dict[str, Any]],
    workers: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(baseline_batch_worker, batch): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            outcome = future.result()
            unit_id = str(outcome["unit_id"])
            episode = outcome["episode"]
            pool = str(outcome["pool"])
            case_mode = bool(outcome["case_mode"])
            references = outcome["references"]
            hard_kind_map = {
                str(row["reference_id"]): str(row["action_kind"]) for row in references
            }
            results[unit_id] = {
                "pool": pool,
                "block_id": str(outcome["block_id"]),
                "case_mode": case_mode,
                "session_id": str(episode.get("session_id", "")),
                "b0_metrics": score_metric_vector(
                    outcome["b0_score"],
                    episode,
                    references,
                    control_infeasible=0,
                    pool=pool,
                    hard_reference_kind_map=hard_kind_map,
                    deadline_views=outcome["b0_score"]["deadline_views"],
                ),
                "b1_metrics": score_metric_vector(
                    outcome["b1_score"],
                    episode,
                    references,
                    control_infeasible=0,
                    pool=pool,
                    hard_reference_kind_map=hard_kind_map,
                    deadline_views=outcome["b1_score"]["deadline_views"],
                ),
                "b0_durations": [
                    int(v)
                    for v in outcome["b0_score"]["segment_source_duration_samples"]
                ],
                "b1_durations": [
                    int(v)
                    for v in outcome["b1_score"]["segment_source_duration_samples"]
                ],
                "b0_active_durations": [
                    int(v)
                    for v in outcome["b0_score"]["segment_active_duration_samples"]
                ],
                "b1_active_durations": [
                    int(v)
                    for v in outcome["b1_score"]["segment_active_duration_samples"]
                ],
                "b0_deadline_views": dict(outcome["b0_score"]["deadline_views"]),
                "b1_deadline_views": dict(outcome["b1_score"]["deadline_views"]),
                "b0_action_trace_sha256": content_sha256(outcome["b0_actions"]),
                "b1_action_trace_sha256": content_sha256(outcome["b1_actions"]),
                "b0_score_trace_sha256": content_sha256(outcome["b0_score"]),
                "b1_score_trace_sha256": content_sha256(outcome["b1_score"]),
            }
    return results


def _historical_baseline_aggregates(
    baseline_scores: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for baseline_id in ("B0", "B1"):
        entry: dict[str, Any] = {
            "metrics": {field: 0 for field in SYSTEM_METRIC_FIELDS},
            "case_ids": [],
            "durations": [],
            "active_durations": [],
            "action_digest": hashlib.sha256(),
            "score_digest": hashlib.sha256(),
        }
        for unit_id, score in baseline_scores.items():
            if not score["case_mode"]:
                continue
            entry["case_ids"].append(str(unit_id))
            if baseline_id == "B0":
                metrics = score["b0_metrics"]
                durations = score["b0_durations"]
                active_durations = score["b0_active_durations"]
                action_trace = score["b0_action_trace_sha256"]
                score_trace = score["b0_score_trace_sha256"]
            else:
                metrics = score["b1_metrics"]
                durations = score["b1_durations"]
                active_durations = score["b1_active_durations"]
                action_trace = score["b1_action_trace_sha256"]
                score_trace = score["b1_score_trace_sha256"]
            for field in SYSTEM_METRIC_FIELDS:
                entry["metrics"][field] += int(metrics[field])
            entry["durations"].extend(durations)
            entry["active_durations"].extend(active_durations)
            _digest_row(entry["action_digest"], action_trace)
            _digest_row(entry["score_digest"], score_trace)
        entry["case_ids"].sort()
        entry["action_digest"] = entry["action_digest"].hexdigest()
        entry["score_digest"] = entry["score_digest"].hexdigest()
        result[baseline_id] = entry
    return result


def _sentinel_units(
    system_info: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    return {}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("stage_a", "gate", "stage_b", "verify"),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="experiments/speaker_turn_boundary directory",
    )
    parser.add_argument(
        "--eres-onnx-root",
        type=Path,
        default=None,
        help="ERes ONNX model root (default TEMP/opencode/eres_onnx)",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = (
        Path(args.experiment_dir).resolve()
        if args.experiment_dir is not None
        else experiment_root()
    )
    if args.eres_onnx_root is None:
        args.eres_onnx_root = default_eres_root()
    if args.command == "stage_a":
        stage_a(experiment_dir, args)
    elif args.command == "gate":
        interstage_gate(experiment_dir)
    elif args.command == "stage_b":
        stage_b(experiment_dir, args)
    elif args.command == "verify":
        from .phase5_verify import run_verification

        run_verification(experiment_dir, args)


if __name__ == "__main__":
    main()

def _load_embedding_universe(
    phase4_inventory: dict[str, Any],
    phase4_contract: dict[str, Any],
    phase4_cache_root: Path,
    phase5_contract: dict[str, Any],
    phase5_cache: Path,
    sources_by_wav: dict[str, AudioSource],
    episodes: Sequence[dict[str, Any]],
    case_rows: Sequence[dict[str, Any]],
    profiles: Sequence[dict[str, Any]],
    historical_development: dict[str, Any] | None = None,
) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    from .phase4_signal import _decode_eres_shard

    needed_by_wav: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for episode in episodes:
        wav = str(episode["wav_sha256"])
        bounds = episode["bounds"]
        for profile in profiles:
            window = int(profile["window_samples"])
            step = int(profile["step_samples"])
            if profile["scored_state_mode"] == "source_prefix":
                first = ceil_grid(window, step)
                for end in range(first, int(bounds["tail_end"]) + 1, step):
                    needed_by_wav[wav].add((end - window, end))
            else:
                low = int(bounds["warm_start"]) + window
                high = int(bounds["tail_end"]) - window
                for boundary in range(ceil_grid(low, step), high + 1, step):
                    needed_by_wav[wav].add((boundary - window, boundary))
                    needed_by_wav[wav].add((boundary, boundary + window))
    for case in case_rows:
        wav = str(case["wav_sha256"])
        duration = int(case["duration_samples"])
        for profile in profiles:
            window = int(profile["window_samples"])
            step = int(profile["step_samples"])
            if profile["profile_class"] == "adjacent":
                for boundary in range(
                    ceil_grid(window, step), duration - window + 1, step
                ):
                    needed_by_wav[wav].add((boundary - window, boundary))
                    needed_by_wav[wav].add((boundary, boundary + window))
            else:
                for end in range(ceil_grid(window, step), duration + 1, step):
                    needed_by_wav[wav].add((end - window, end))
    phase4_rows = phase4_inventory["eres"]["E-standard"]["sources"]
    phase4_by_wav: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for row in phase4_rows:
        wav = str(row["wav_sha256"])
        needed = needed_by_wav.get(wav)
        if not needed:
            continue
        metadata_path = Path(str(row["metadata_path"]))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        found: dict[tuple[int, int], np.ndarray] = {}
        for shard in metadata["shards"]:
            path = metadata_path.parent / str(shard["path"])
            compressed = path.read_bytes()
            if sha256_bytes(compressed) != shard["byte_sha256"]:
                raise Phase5RunError(f"phase4 cache shard hash drift: {path}")
            plain = gzip.decompress(compressed)
            windows, embeddings, _, _ = _decode_eres_shard(plain)
            for window, embedding in zip(windows.tolist(), embeddings):
                key = (int(window[0]), int(window[1]))
                if key in needed:
                    found[key] = np.asarray(embedding, dtype=np.float32)
        phase4_by_wav[wav] = found
    missing_by_wav: dict[str, set[tuple[int, int]]] = {}
    for wav, windows in needed_by_wav.items():
        cached = set(phase4_by_wav.get(wav, {}))
        missing = windows - cached
        if missing:
            missing_by_wav[wav] = missing
    if historical_development is not None:
        cache_root = Path(str(historical_development["cache_root"]))
        for receipt in historical_development["cache_receipt_rows"]:
            metadata_path = cache_root / str(receipt["metadata_relative_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            wav = str(metadata["wav_sha256"])
            missing = missing_by_wav.get(wav)
            if not missing:
                continue
            npz_path = cache_root / str(receipt["npz_relative_path"])
            if sha256_file(npz_path) != receipt["npz_byte_sha256"]:
                raise Phase5RunError(f"historical cache npz hash drift: {npz_path}")
            with np.load(npz_path) as data:
                merged = dict(phase4_by_wav.get(wav, {}))
                for start, end in missing:
                    key = f"{start}-{end}"
                    if key in data:
                        merged[(start, end)] = np.asarray(
                            data[key], dtype=np.float32
                        ).reshape(-1)
            phase4_by_wav[wav] = merged
            remaining = missing - set(merged)
            if remaining:
                missing_by_wav[wav] = remaining
            else:
                missing_by_wav.pop(wav, None)
    phase5_by_wav: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for wav, windows in missing_by_wav.items():
        source = sources_by_wav[wav]
        ordered = sorted(windows)
        cached = load_eres_embeddings(
            phase5_cache, phase5_contract, "E-standard", source, ordered
        )
        if cached is None:
            raise Phase5RunError(f"phase5 cache window missing: {source.source_id}")
        embeddings, _, _ = cached
        phase5_by_wav[wav] = embeddings
    result: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for wav in needed_by_wav:
        merged = dict(phase4_by_wav.get(wav, {}))
        merged.update(phase5_by_wav.get(wav, {}))
        if len(merged) != len(needed_by_wav[wav]):
            raise Phase5RunError(f"embedding universe incomplete: {wav[:16]}")
        result[wav] = merged
    return result
