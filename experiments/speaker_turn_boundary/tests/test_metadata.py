from __future__ import annotations

import os

from experiments.speaker_turn_boundary.config import (
    B0_VAD_HANGOVER_MS,
    B0_VAD_MAX_SEGMENT_MS,
    B0_VAD_PRE_ROLL_MS,
    B0_VAD_PROFILE,
    B0_VAD_SPEECH_THRESHOLD,
    B0_VAD_START_COMMIT_CHUNKS,
    B0_VAD_START_DEBOUNCE_CHUNKS,
    CANONICAL_SAMPLE_RATE_HZ,
)
from experiments.speaker_turn_boundary.metadata import (
    collect_runtime_metadata,
    resolve_baseline_sha,
)
from puripuly_heart.core.vad.gating import (
    PEER_MAX_SEGMENT_MS,
    PEER_VAD_SPEECH_THRESHOLD,
    PEER_VAD_START_COMMIT_CHUNKS,
    PEER_VAD_START_DEBOUNCE_CHUNKS,
)

REQUIRED_KEYS = {
    "baseline_sha",
    "baseline_label",
    "canonical_sample_rate_hz",
    "python_version",
    "python_full",
    "python_implementation",
    "platform",
    "platform_release",
    "platform_version",
    "machine",
    "processor",
    "cpu_count",
    "ram_total_bytes",
    "onnxruntime_version",
    "ort_provider",
    "ort_intra_op_threads",
    "ort_inter_op_threads",
    "ort_graph_optimization",
    "b0_vad_profile",
    "b0_vad_speech_threshold",
    "b0_vad_start_debounce_chunks",
    "b0_vad_start_commit_chunks",
    "b0_vad_max_segment_ms",
    "b0_vad_hangover_ms",
    "b0_vad_pre_roll_ms",
    "vad_coalesce_window_ms",
}


def test_metadata_contains_all_required_keys():
    metadata = collect_runtime_metadata()
    assert REQUIRED_KEYS <= set(metadata)


def test_metadata_records_baseline_sha():
    metadata = collect_runtime_metadata()
    assert metadata["baseline_sha"] == resolve_baseline_sha()
    assert len(metadata["baseline_sha"]) == 40


def test_metadata_records_canonical_rate_and_b0_profile():
    metadata = collect_runtime_metadata()
    assert metadata["canonical_sample_rate_hz"] == CANONICAL_SAMPLE_RATE_HZ
    assert metadata["b0_vad_profile"] == B0_VAD_PROFILE
    assert metadata["b0_vad_speech_threshold"] == B0_VAD_SPEECH_THRESHOLD
    assert metadata["b0_vad_hangover_ms"] == B0_VAD_HANGOVER_MS
    assert metadata["b0_vad_pre_roll_ms"] == B0_VAD_PRE_ROLL_MS
    assert metadata["b0_vad_max_segment_ms"] == B0_VAD_MAX_SEGMENT_MS


def test_metadata_records_ort_thread_configuration():
    metadata = collect_runtime_metadata()
    assert metadata["ort_provider"] == "CPUExecutionProvider"
    assert metadata["ort_intra_op_threads"] == 1
    assert metadata["ort_inter_op_threads"] == 1
    assert metadata["ort_graph_optimization"] == "ORT_ENABLE_ALL"


def test_metadata_platform_fields_present():
    metadata = collect_runtime_metadata()
    assert metadata["platform_release"]
    assert metadata["machine"]
    assert metadata["cpu_count"] == os.cpu_count()
    assert isinstance(metadata["ram_total_bytes"], int) or metadata["ram_total_bytes"] is None


def test_b0_config_matches_actual_dev_vad_constants():
    assert B0_VAD_SPEECH_THRESHOLD == PEER_VAD_SPEECH_THRESHOLD
    assert B0_VAD_START_DEBOUNCE_CHUNKS == PEER_VAD_START_DEBOUNCE_CHUNKS
    assert B0_VAD_START_COMMIT_CHUNKS == PEER_VAD_START_COMMIT_CHUNKS
    assert B0_VAD_MAX_SEGMENT_MS == PEER_MAX_SEGMENT_MS
    assert B0_VAD_HANGOVER_MS == 500
    assert B0_VAD_PRE_ROLL_MS == 500


def test_resolve_baseline_sha_is_hex_sha():
    resolved = resolve_baseline_sha()
    assert len(resolved) == 40
    assert all(character in "0123456789abcdef" for character in resolved)
