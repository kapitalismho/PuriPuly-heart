from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

from experiments.speaker_turn_boundary.config import (
    B0_VAD_HANGOVER_MS,
    B0_VAD_MAX_SEGMENT_MS,
    B0_VAD_PRE_ROLL_MS,
    B0_VAD_PROFILE,
    B0_VAD_SPEECH_THRESHOLD,
    B0_VAD_START_COMMIT_CHUNKS,
    B0_VAD_START_DEBOUNCE_CHUNKS,
    BASELINE_LABEL,
    BASELINE_SHA,
    CANONICAL_SAMPLE_RATE_HZ,
    VAD_COALESCE_WINDOW_MS,
)

B0_ORT_INTRA_OP_THREADS = 1
B0_ORT_INTER_OP_THREADS = 1
B0_ORT_GRAPH_OPTIMIZATION = "ORT_ENABLE_ALL"
B0_ORT_PROVIDER = "CPUExecutionProvider"


def resolve_baseline_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return BASELINE_SHA


def collect_runtime_metadata() -> dict[str, Any]:
    ram_total_bytes: int | None = None
    try:
        import psutil

        ram_total_bytes = int(psutil.virtual_memory().total)
    except (ImportError, OSError):
        pass
    ort_version: str | None = None
    try:
        import onnxruntime

        ort_version = str(onnxruntime.__version__)
    except (ImportError, OSError):
        pass
    return {
        "baseline_sha": resolve_baseline_sha(),
        "baseline_label": BASELINE_LABEL,
        "canonical_sample_rate_hz": CANONICAL_SAMPLE_RATE_HZ,
        "python_version": platform.python_version(),
        "python_full": sys.version.splitlines()[0] if sys.version else None,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": ram_total_bytes,
        "onnxruntime_version": ort_version,
        "ort_provider": B0_ORT_PROVIDER,
        "ort_intra_op_threads": B0_ORT_INTRA_OP_THREADS,
        "ort_inter_op_threads": B0_ORT_INTER_OP_THREADS,
        "ort_graph_optimization": B0_ORT_GRAPH_OPTIMIZATION,
        "b0_vad_profile": B0_VAD_PROFILE,
        "b0_vad_speech_threshold": B0_VAD_SPEECH_THRESHOLD,
        "b0_vad_start_debounce_chunks": B0_VAD_START_DEBOUNCE_CHUNKS,
        "b0_vad_start_commit_chunks": B0_VAD_START_COMMIT_CHUNKS,
        "b0_vad_max_segment_ms": B0_VAD_MAX_SEGMENT_MS,
        "b0_vad_hangover_ms": B0_VAD_HANGOVER_MS,
        "b0_vad_pre_roll_ms": B0_VAD_PRE_ROLL_MS,
        "vad_coalesce_window_ms": VAD_COALESCE_WINDOW_MS,
    }
