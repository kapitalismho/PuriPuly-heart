from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.psem_relative_occupancy_gate.contracts import Trace
from experiments.psem_relative_occupancy_gate.io_utils import (
    PACKAGE_ROOT,
    ExperimentError,
    config,
    load_json,
    load_jsonl,
    percentile,
    research_root,
    safe_child,
    safe_output_path,
    sha256_file,
    strict_regular_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.trace_io import (
    TRACE_SCHEMA_VERSION,
    TraceIOError,
    validate_trace_receipt,
    write_trace,
)
from experiments.psem_relative_occupancy_gate.trace_runtime import (
    TraceRuntimeError,
    backend_resolution_matches,
    external_trace_root,
    load_trace_manifest,
    source_window,
    trace_run_key,
    validate_full_trace_geometry,
    validate_trace_location,
    waveform_slice,
)

SORTFORMER_FAMILY = "streaming_sortformer"
SORTFORMER_FRAME_SAMPLES = 1280
SORTFORMER_CHUNK_FRAMES = 6
SORTFORMER_RIGHT_CONTEXT_FRAMES = 7
SORTFORMER_MEL_HOP_SAMPLES = 160
SORTFORMER_SUBSAMPLING = 8
SORTFORMER_WINDOW_SAMPLES = 400
SORTFORMER_STFT_RIGHT_SAMPLES = SORTFORMER_WINDOW_SAMPLES // 2
SLOT_IDS = ("slot-0", "slot-1", "slot-2", "slot-3")
INFERENCE_AUDIO_MATERIALIZATION = (
    "exact_pcm16_mono_16khz_frozen_source_with_zero_flush_to_full_context_chunk"
)


class SortformerTraceError(RuntimeError):
    pass


def _load_probability_dump(directory: Path) -> np.ndarray:
    metadata_path = directory / "diar.probs.json"
    data_path = directory / "diar.probs.f32"
    metadata = load_json(metadata_path)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("shape"), list):
        raise SortformerTraceError("Sortformer probability metadata is invalid")
    shape = tuple(int(value) for value in metadata["shape"])
    values = np.fromfile(data_path, dtype="<f4")
    if values.size != int(np.prod(shape)):
        raise SortformerTraceError("Sortformer probability dump size is invalid")
    probabilities = values.reshape(shape).astype(np.float32, copy=False)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(SLOT_IDS):
        raise SortformerTraceError("Sortformer probability shape is invalid")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise SortformerTraceError("Sortformer probabilities are invalid")
    return probabilities


def _load_probability_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        probabilities = np.asarray(data["probabilities"], dtype=np.float32)
        model_sha256 = str(data["model_sha256"].item())
        frame_ms = int(data["frame_ms"].item())
        backend = str(data["backend"].item())
    cfg = config()["sortformer"]
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(SLOT_IDS)
        or model_sha256 != cfg["model_sha256"]
        or frame_ms != cfg["native_frame_ms"]
        or backend != cfg["backend"]
    ):
        raise SortformerTraceError("R8 probability cache contract mismatch")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise SortformerTraceError("R8 probability cache values are invalid")
    return probabilities


def _telemetry_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows:
        raise SortformerTraceError("Sortformer telemetry is empty")
    required = {
        "chunk_index",
        "pre_encode_us",
        "infer_us",
        "update_us",
        "compression_called",
        "compression_us",
        "total_us",
        "new_audio_frames",
    }
    if any(not required <= set(row) for row in rows):
        raise SortformerTraceError("Sortformer telemetry schema is incomplete")
    return rows


def _telemetry_runs(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if int(row["chunk_index"]) == 0 and current:
            result.append(current)
            current = []
        current.append(row)
    if current:
        result.append(current)
    return result


def sortformer_frame_geometry(
    frame_count: int,
    sample_count: int,
    source_start_sample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_frames = math.ceil(sample_count / SORTFORMER_FRAME_SAMPLES)
    if frame_count != expected_frames:
        raise SortformerTraceError(
            f"Sortformer frame count mismatch: {frame_count} != {expected_frames}"
        )
    local_starts = np.arange(frame_count, dtype=np.int64) * SORTFORMER_FRAME_SAMPLES
    local_ends = np.minimum(local_starts + SORTFORMER_FRAME_SAMPLES, sample_count)
    mel_frames = math.ceil(sample_count / SORTFORMER_MEL_HOP_SAMPLES)
    local_frontiers = np.empty(frame_count, dtype=np.int64)
    for index in range(frame_count):
        chunk = index // SORTFORMER_CHUNK_FRAMES
        chunk_mel_start = (
            chunk * SORTFORMER_CHUNK_FRAMES * SORTFORMER_SUBSAMPLING
        )
        chunk_mel_end = min(
            chunk_mel_start
            + SORTFORMER_CHUNK_FRAMES * SORTFORMER_SUBSAMPLING,
            mel_frames,
        )
        window_mel_end = min(
            chunk_mel_end
            + SORTFORMER_RIGHT_CONTEXT_FRAMES * SORTFORMER_SUBSAMPLING,
            mel_frames,
        )
        raw_frontier = min(
            sample_count,
            (window_mel_end - 1) * SORTFORMER_MEL_HOP_SAMPLES
            + SORTFORMER_STFT_RIGHT_SAMPLES,
        )
        local_frontiers[index] = max(int(local_ends[index]), raw_frontier)
    return (
        local_starts + source_start_sample,
        local_ends + source_start_sample,
        local_frontiers + source_start_sample,
    )


def _trace(
    *,
    row: dict[str, Any],
    probabilities: np.ndarray,
    source_start_sample: int,
    source_end_sample: int,
    inference_audio: dict[str, Any],
) -> Trace:
    sample_count = source_end_sample - source_start_sample
    starts, ends, frontiers = sortformer_frame_geometry(
        probabilities.shape[0], sample_count, source_start_sample
    )
    state_reset = np.zeros(probabilities.shape[0], dtype=np.bool_)
    state_reset[0] = True
    cfg = config()
    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "source_id": row["source_id"],
        "family": SORTFORMER_FAMILY,
        "sample_rate_hz": 16000,
        "speaker_slot_ids": list(SLOT_IDS),
        "role": row["role"],
        "manifest_row_sha256": row["row_sha256"],
        "waveform_sha256": row["waveform_sha256"],
        "inference_audio_path": inference_audio["path"],
        "inference_audio_sha256": inference_audio["sha256"],
        "inference_audio_sample_count": inference_audio["sample_count"],
        "inference_audio_source_sample_count": inference_audio[
            "source_sample_count"
        ],
        "inference_audio_trailing_zero_sample_count": inference_audio[
            "trailing_zero_sample_count"
        ],
        "inference_audio_native_frame_count": inference_audio[
            "native_frame_count"
        ],
        "inference_audio_retained_frame_count": inference_audio[
            "retained_frame_count"
        ],
        "inference_audio_materialization": inference_audio["materialization"],
        "source_start_sample": source_start_sample,
        "source_end_sample": source_end_sample,
        "model_sha256": cfg["sortformer"]["model_sha256"],
        "model_revision": cfg["sortformer"]["model_revision"],
        "source_commit": cfg["sortformer"]["source_commit"],
        "telemetry_patch_sha256": cfg["sortformer"]["telemetry_patch_sha256"],
        "bench_relative_path": cfg["sortformer"]["bench_relative_path"],
        "bench_sha256": cfg["sortformer"]["bench_sha256"],
        "backend": cfg["sortformer"]["backend"],
        "threads": cfg["sortformer"]["threads"],
        "preset": cfg["sortformer"]["preset"],
        "native_frame_samples": SORTFORMER_FRAME_SAMPLES,
        "native_frame_support_rule": "frame_index_times_1280_half_open_clipped",
        "chunk_frames": SORTFORMER_CHUNK_FRAMES,
        "right_context_frames": SORTFORMER_RIGHT_CONTEXT_FRAMES,
        "recorded_algorithmic_lookahead_samples": 16640,
        "frontend_mel_hop_samples": SORTFORMER_MEL_HOP_SAMPLES,
        "frontend_subsampling": SORTFORMER_SUBSAMPLING,
        "frontend_stft_window_samples": SORTFORMER_WINDOW_SAMPLES,
        "evidence_frontier_rule": "chunk_window_last_mel_nonzero_raw_support_end_clipped",
        "slot_validity_metadata_exposed": False,
        "slot_continuity": "stable_columns_within_one_uninterrupted_model_epoch",
        "state_reset_rule": "first_frame_only",
        "whole_file_api_limitation": "internally_streaming_core_without_live_push_audio_entrypoint",
        "adapter_code_sha256": sha256_file(Path(__file__)),
        "trace_io_code_sha256": sha256_file(PACKAGE_ROOT / "trace_io.py"),
        "contracts_code_sha256": sha256_file(PACKAGE_ROOT / "contracts.py"),
    }
    return Trace(
        source_id=str(row["source_id"]),
        family=SORTFORMER_FAMILY,
        slot_ids=SLOT_IDS,
        probabilities=probabilities,
        frame_start_samples=starts,
        frame_end_samples=ends,
        evidence_frontier_samples=frontiers,
        slot_alive=np.ones(probabilities.shape, dtype=np.bool_),
        state_reset=state_reset,
        metadata=metadata,
    )


def _telemetry_summary(
    path: Path, rows: Sequence[dict[str, Any]], expected_chunks: int
) -> dict[str, Any]:
    if len(rows) != expected_chunks:
        raise SortformerTraceError(
            f"Sortformer telemetry chunk count mismatch: {len(rows)} != {expected_chunks}"
        )
    if any(
        int(row["chunk_index"]) != index
        or int(row["new_audio_frames"]) != SORTFORMER_CHUNK_FRAMES
        for index, row in enumerate(rows)
    ):
        raise SortformerTraceError("Sortformer telemetry chunk geometry is invalid")
    totals = [float(row["total_us"]) / 1000.0 for row in rows]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "chunk_count": len(rows),
        "chunk_compute_ms": {
            "p50": percentile(totals, 50),
            "p90": percentile(totals, 90),
            "p99": percentile(totals, 99),
            "maximum": max(totals),
            "sum": sum(totals),
        },
        "compression_call_count": sum(bool(row["compression_called"]) for row in rows),
    }


def validate_sortformer_telemetry_receipt(
    path: Path,
    receipt: Any,
    *,
    expected_chunks: int,
) -> dict[str, Any]:
    try:
        path = strict_regular_file(path, "Sortformer telemetry")
        runs = _telemetry_runs(_telemetry_rows(path))
    except ExperimentError as exc:
        raise SortformerTraceError("Sortformer telemetry receipt is invalid") from exc
    if len(runs) != 1:
        raise SortformerTraceError("Sortformer telemetry does not contain exactly one run")
    observed = _telemetry_summary(path, runs[0], expected_chunks)
    if not isinstance(receipt, dict) or receipt != observed:
        raise SortformerTraceError("Sortformer telemetry receipt binding mismatch")
    return observed


def _materialize_inference_audio(
    source_root: Path,
    row: dict[str, Any],
    source_start_sample: int,
    source_end_sample: int,
) -> tuple[Path, dict[str, Any]]:
    path = source_root / "input.wav"
    geometry = sortformer_inference_audio_geometry(
        source_end_sample - source_start_sample
    )
    payload = waveform_slice(row, source_start_sample, source_end_sample)
    if len(payload) != geometry["source_sample_count"] * 2:
        raise SortformerTraceError("Sortformer source audio geometry is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframesraw(payload)
        writer.writeframes(bytes(geometry["trailing_zero_sample_count"] * 2))
    return path, _inspect_inference_audio(path, source_start_sample, source_end_sample)


def sortformer_inference_audio_geometry(
    source_sample_count: int,
) -> dict[str, int | str]:
    if source_sample_count <= 0:
        raise SortformerTraceError("Sortformer source audio is empty")
    source_mel_frame_count = math.ceil(
        source_sample_count / SORTFORMER_MEL_HOP_SAMPLES
    )
    native_mel_chunk = SORTFORMER_CHUNK_FRAMES * SORTFORMER_SUBSAMPLING
    minimum_native_mel_frames = (
        source_mel_frame_count
        + SORTFORMER_RIGHT_CONTEXT_FRAMES * SORTFORMER_SUBSAMPLING
    )
    native_mel_frame_count = (
        math.ceil(minimum_native_mel_frames / native_mel_chunk) * native_mel_chunk
    )
    sample_count = native_mel_frame_count * SORTFORMER_MEL_HOP_SAMPLES
    return {
        "sample_count": sample_count,
        "source_sample_count": source_sample_count,
        "trailing_zero_sample_count": sample_count - source_sample_count,
        "source_mel_frame_count": source_mel_frame_count,
        "native_mel_frame_count": native_mel_frame_count,
        "retained_frame_count": math.ceil(
            source_sample_count / SORTFORMER_FRAME_SAMPLES
        ),
        "native_frame_count": native_mel_frame_count // SORTFORMER_SUBSAMPLING,
        "materialization": INFERENCE_AUDIO_MATERIALIZATION,
    }


def _inspect_inference_audio(
    path: Path, source_start_sample: int, source_end_sample: int
) -> dict[str, Any]:
    geometry = sortformer_inference_audio_geometry(
        source_end_sample - source_start_sample
    )
    try:
        path = strict_regular_file(path, "Sortformer inference audio")
        with wave.open(str(path), "rb") as reader:
            valid = (
                reader.getnchannels() == 1
                and reader.getsampwidth() == 2
                and reader.getframerate() == 16000
                and reader.getnframes() == geometry["sample_count"]
            )
            if valid:
                reader.setpos(int(geometry["source_sample_count"]))
                trailing = reader.readframes(
                    int(geometry["trailing_zero_sample_count"])
                )
                valid = trailing == bytes(len(trailing)) and len(trailing) == int(
                    geometry["trailing_zero_sample_count"]
                ) * 2
    except (ExperimentError, EOFError, OSError, wave.Error) as exc:
        raise SortformerTraceError("Sortformer inference audio is invalid") from exc
    if not valid:
        raise SortformerTraceError("Sortformer inference audio geometry is invalid")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        **geometry,
        "source_start_sample": source_start_sample,
        "source_end_sample": source_end_sample,
    }


def _load_resume_inference_audio(
    receipt_path: Path,
    audio_path: Path,
    source_start_sample: int,
    source_end_sample: int,
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    receipt = load_json(receipt_path)
    if not isinstance(receipt, dict):
        raise SortformerTraceError("Sortformer resume receipt is invalid")
    expected = receipt.get("inference_audio")
    if expected is None:
        return None
    if (
        not isinstance(expected, dict)
        or expected.get("materialization") != INFERENCE_AUDIO_MATERIALIZATION
    ):
        return None
    observed = _inspect_inference_audio(
        audio_path, source_start_sample, source_end_sample
    )
    if observed != expected:
        raise SortformerTraceError("Sortformer resume inference audio binding mismatch")
    return observed


def _retain_source_probabilities(
    raw_probabilities: np.ndarray, inference_audio: dict[str, Any]
) -> np.ndarray:
    if raw_probabilities.shape[0] != inference_audio["native_frame_count"]:
        raise SortformerTraceError("Sortformer native probability frame count mismatch")
    retained = raw_probabilities[
        : int(inference_audio["retained_frame_count"])
    ].copy()
    if retained.shape[0] != inference_audio["retained_frame_count"]:
        raise SortformerTraceError("Sortformer retained probability frame count mismatch")
    return retained


def _run_new_inference(
    *,
    bench: Path,
    model: Path,
    audio: Path,
    run_directory: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], Path]:
    dump_directory = run_directory / "dump"
    dump_directory.mkdir(parents=True, exist_ok=True)
    for name in ("diar.probs.json", "diar.probs.f32"):
        stale = dump_directory / name
        if stale.exists():
            stale.unlink()
    telemetry_path = run_directory / "telemetry.jsonl"
    bench_path = run_directory / "bench.json"
    log_path = run_directory / "run.log"
    for stale in (telemetry_path, bench_path):
        if stale.exists():
            stale.unlink()
    cfg = config()["sortformer"]
    command = [
        str(bench),
        "--model",
        str(model),
        "--sample",
        str(audio),
        "--backend",
        str(cfg["backend"]),
        "--threads",
        str(cfg["threads"]),
        "--warmup",
        "0",
        "--iters",
        "1",
        "--json-out",
        str(bench_path),
    ]
    environment = os.environ.copy()
    environment["TRANSCRIBE_DUMP_DIR"] = str(dump_directory)
    environment["TRANSCRIBE_SORTFORMER_STREAM_PRESET"] = str(cfg["preset"])
    environment["TRANSCRIBE_SORTFORMER_TELEMETRY_PATH"] = str(telemetry_path)
    started = time.perf_counter()
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command,
            cwd=run_directory,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    process_wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise SortformerTraceError(f"Sortformer inference failed; see {log_path}")
    bench_result = load_json(bench_path)
    if (
        not isinstance(bench_result, dict)
        or not backend_resolution_matches(cfg["backend"], bench_result.get("backend"))
        or bench_result.get("iters") != 1
        or bench_result.get("warmup") != 0
    ):
        raise SortformerTraceError("Sortformer bench receipt is invalid")
    probabilities = _load_probability_dump(dump_directory)
    runs = _telemetry_runs(_telemetry_rows(telemetry_path))
    if len(runs) != 1:
        raise SortformerTraceError("Sortformer inference did not produce exactly one telemetry run")
    return (
        probabilities,
        runs[0],
        {
            "origin": "fresh_single_frozen_inference_pass",
            "backend_resolved": str(bench_result["backend"]),
            "command": command,
            "process_wall_seconds": process_wall_seconds,
            "bench_path": str(bench_path.resolve()),
            "bench_sha256": sha256_file(bench_path),
            "bench": bench_result,
            "log_path": str(log_path.resolve()),
            "log_sha256": sha256_file(log_path),
        },
        telemetry_path,
    )


def _reuse_r8(
    *, row: dict[str, Any], research: Path
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], Path] | None:
    source_id = str(row["source_id"])
    base = safe_child(
        research,
        "results/r8/streaming_sortformer_feasibility_v1",
        "R8 cache root",
    )
    receipt_path = safe_child(
        base,
        Path("runs/full/cpu/diar_streaming_sortformer_4spk-v2.1-Q8_0")
        / f"{source_id}.receipt.json",
        "R8 source receipt",
    )
    if not receipt_path.is_file():
        return None
    receipt = load_json(receipt_path)
    if not isinstance(receipt, dict):
        raise SortformerTraceError("R8 source receipt is invalid")
    probability_path = Path(str(receipt.get("probability_path", ""))).resolve()
    telemetry_path = Path(str(receipt.get("telemetry_path", ""))).resolve()
    audio_path = Path(str(receipt.get("audio_path", ""))).resolve()
    cfg = config()["sortformer"]
    if (
        receipt.get("item_id") != source_id
        or receipt.get("backend_requested") != cfg["backend"]
        or receipt.get("backend_resolved") != cfg["backend"]
        or receipt.get("model_sha256") != cfg["model_sha256"]
        or receipt.get("bench_sha256") != cfg["bench_sha256"]
        or not bool(receipt.get("full_waveform_duration"))
        or not probability_path.is_file()
        or sha256_file(probability_path) != receipt.get("probability_sha256")
        or not telemetry_path.is_file()
        or not audio_path.is_file()
        or audio_path.stat().st_size != int(row["waveform_size_bytes"])
        or sha256_file(audio_path) != row["waveform_sha256"]
    ):
        raise SortformerTraceError(f"R8 cache identity mismatch: {source_id}")
    probabilities = _load_probability_npz(probability_path)
    runs = _telemetry_runs(_telemetry_rows(telemetry_path))
    if len(runs) != int(receipt.get("telemetry_run_count", -1)) or not runs:
        raise SortformerTraceError("R8 telemetry run count mismatch")
    return (
        probabilities,
        runs[-1],
        {
            "origin": "verified_existing_r8_frozen_inference_pass",
            "backend_resolved": str(receipt["backend_resolved"]),
            "r8_receipt_path": str(receipt_path.resolve()),
            "r8_receipt_sha256": sha256_file(receipt_path),
            "probability_cache_path": str(probability_path),
            "probability_cache_sha256": sha256_file(probability_path),
            "bench": receipt.get("bench"),
            "process": receipt.get("process"),
        },
        telemetry_path,
    )


def _resume_receipt(
    receipt_path: Path,
    *,
    row: dict[str, Any],
    source_start_sample: int,
    source_end_sample: int,
    source_root: Path,
    bench: Path,
    model: Path,
    audio: Path,
    inference_audio: dict[str, Any],
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    receipt = load_json(receipt_path)
    if not isinstance(receipt, dict):
        return None
    trace_path = Path(str(receipt.get("trace", {}).get("trace_path", "")))
    model_cfg = config()["sortformer"]
    inference_receipt = receipt.get("inference")
    telemetry_receipt = receipt.get("telemetry")
    bench_receipt = (
        inference_receipt.get("bench")
        if isinstance(inference_receipt, dict)
        else None
    )
    expected_usage = (
        "full_frozen_source"
        if source_start_sample == 0
        and source_end_sample == int(row["source_duration_samples"])
        else "non_authoritative_train_adapter_smoke"
    )
    stable = (
        receipt.get("schema_version")
        == "psem.relative_occupancy.model_source_receipt.v1"
        and receipt.get("trace_schema_version") == TRACE_SCHEMA_VERSION
        and receipt.get("family") == SORTFORMER_FAMILY
        and receipt.get("source_id") == row["source_id"]
        and receipt.get("role") == row["role"]
        and receipt.get("usage") == expected_usage
        and receipt.get("manifest_row_sha256") == row["row_sha256"]
        and receipt.get("waveform_sha256") == row["waveform_sha256"]
        and receipt.get("waveform_size_bytes") == row["waveform_size_bytes"]
        and receipt.get("inference_audio") == inference_audio
        and receipt.get("source_start_sample") == source_start_sample
        and receipt.get("source_end_sample") == source_end_sample
        and receipt.get("model_sha256") == model_cfg["model_sha256"]
        and receipt.get("bench_relative_path") == model_cfg["bench_relative_path"]
        and receipt.get("bench_sha256") == model_cfg["bench_sha256"]
        and receipt.get("source_repository") == model_cfg["source_repository"]
        and receipt.get("source_commit") == model_cfg["source_commit"]
        and receipt.get("telemetry_patch_sha256")
        == model_cfg["telemetry_patch_sha256"]
        and receipt.get("backend") == model_cfg["backend"]
        and backend_resolution_matches(
            model_cfg["backend"], receipt.get("backend_resolved")
        )
        and isinstance(inference_receipt, dict)
        and inference_receipt.get("backend_resolved") == receipt.get("backend_resolved")
        and inference_receipt.get("audio") == inference_audio
        and inference_receipt.get("raw_probability_frame_count")
        == inference_audio["native_frame_count"]
        and inference_receipt.get("retained_probability_frame_count")
        == inference_audio["retained_frame_count"]
        and isinstance(bench_receipt, dict)
        and bench_receipt.get("backend") == receipt.get("backend_resolved")
        and bench_receipt.get("iters") == 1
        and bench_receipt.get("warmup") == 0
        and receipt.get("threads") == model_cfg["threads"]
        and receipt.get("preset") == model_cfg["preset"]
        and Path(str(receipt.get("bench_path", ""))).resolve() == bench
        and Path(str(receipt.get("model_path", ""))).resolve() == model
        and Path(str(receipt.get("waveform_path", ""))).resolve()
        == Path(str(row["audio_path"])).resolve()
    )
    if not stable:
        return None
    try:
        trace_path = validate_trace_location(
            trace_path,
            family=SORTFORMER_FAMILY,
            backend=str(model_cfg["backend"]),
            role=str(row["role"]),
            source_id=str(row["source_id"]),
        )
        if trace_path != source_root / "posterior_trace.npz":
            return None
        trace = validate_trace_receipt(trace_path, receipt["trace"])
        validate_full_trace_geometry(
            trace,
            family=SORTFORMER_FAMILY,
            source_start_sample=source_start_sample,
            source_end_sample=source_end_sample,
        )
        raw_bench_path = strict_regular_file(
            Path(str(inference_receipt.get("bench_path", ""))), "Sortformer raw bench receipt"
        )
        telemetry_path = strict_regular_file(
            Path(str(telemetry_receipt.get("path", "")))
            if isinstance(telemetry_receipt, dict)
            else Path(),
            "Sortformer telemetry",
        )
        if (
            raw_bench_path != source_root / "run" / "bench.json"
            or telemetry_path != source_root / "run" / "telemetry.jsonl"
            or inference_receipt.get("bench_sha256") != sha256_file(raw_bench_path)
            or load_json(raw_bench_path) != bench_receipt
            or bench_receipt.get("model_path") != str(model)
            or bench_receipt.get("sample_path") != str(audio)
            or inference_receipt.get("command")
            != [
                str(bench),
                "--model",
                str(model),
                "--sample",
                str(audio),
                "--backend",
                str(model_cfg["backend"]),
                "--threads",
                str(model_cfg["threads"]),
                "--warmup",
                "0",
                "--iters",
                "1",
                "--json-out",
                str(raw_bench_path),
            ]
        ):
            return None
        validate_sortformer_telemetry_receipt(
            telemetry_path,
            telemetry_receipt,
            expected_chunks=int(inference_audio["native_frame_count"])
            // SORTFORMER_CHUNK_FRAMES,
        )
    except (
        ExperimentError,
        KeyError,
        SortformerTraceError,
        TraceIOError,
        TraceRuntimeError,
        OSError,
    ):
        return None
    expected_metadata = {
        "source_id": row["source_id"],
        "family": SORTFORMER_FAMILY,
        "role": row["role"],
        "manifest_row_sha256": row["row_sha256"],
        "waveform_sha256": row["waveform_sha256"],
        "inference_audio_path": inference_audio["path"],
        "inference_audio_sha256": inference_audio["sha256"],
        "inference_audio_sample_count": inference_audio["sample_count"],
        "inference_audio_source_sample_count": inference_audio[
            "source_sample_count"
        ],
        "inference_audio_trailing_zero_sample_count": inference_audio[
            "trailing_zero_sample_count"
        ],
        "inference_audio_native_frame_count": inference_audio[
            "native_frame_count"
        ],
        "inference_audio_retained_frame_count": inference_audio[
            "retained_frame_count"
        ],
        "inference_audio_materialization": inference_audio["materialization"],
        "source_start_sample": source_start_sample,
        "source_end_sample": source_end_sample,
        "model_sha256": model_cfg["model_sha256"],
        "model_revision": model_cfg["model_revision"],
        "source_commit": model_cfg["source_commit"],
        "telemetry_patch_sha256": model_cfg["telemetry_patch_sha256"],
        "bench_relative_path": model_cfg["bench_relative_path"],
        "bench_sha256": model_cfg["bench_sha256"],
        "backend": model_cfg["backend"],
        "threads": model_cfg["threads"],
        "preset": model_cfg["preset"],
        "adapter_code_sha256": sha256_file(Path(__file__)),
        "trace_io_code_sha256": sha256_file(PACKAGE_ROOT / "trace_io.py"),
        "contracts_code_sha256": sha256_file(PACKAGE_ROOT / "contracts.py"),
    }
    if any(trace.metadata.get(field) != value for field, value in expected_metadata.items()):
        return None
    return receipt


def run_traces(
    *,
    manifest: Path,
    role: str,
    research: Path,
    trace_root: Path | None,
    output: Path,
    source_ids: Sequence[str] | None,
    smoke_samples: int | None,
    resume: bool,
    reuse_r8_cache: bool,
) -> dict[str, Any]:
    cfg = config()
    model_cfg = cfg["sortformer"]
    research = research_root(research)
    root = external_trace_root(research, trace_root)
    aggregate_output = safe_output_path(output)
    model = safe_child(
        research,
        Path("models/r8") / model_cfg["model_filename"],
        "Sortformer model",
    )
    bench = safe_child(
        research,
        model_cfg["bench_relative_path"],
        "Sortformer bench",
    )
    if sha256_file(model) != model_cfg["model_sha256"]:
        raise SortformerTraceError("Sortformer model hash mismatch")
    if sha256_file(bench) != model_cfg["bench_sha256"]:
        raise SortformerTraceError("Sortformer bench hash mismatch")
    if (
        aggregate_output in {manifest.resolve(), model, bench}
        or aggregate_output == root
        or root in aggregate_output.parents
    ):
        raise SortformerTraceError("Sortformer aggregate output aliases an inference input")
    if reuse_r8_cache and model_cfg["backend"] != "cpu":
        raise SortformerTraceError("R8 cache reuse is limited to the CPU backend")
    rows = load_trace_manifest(manifest, role=role, source_ids=source_ids)
    if role == "PSEM-STRATEGY-DEV" and smoke_samples is not None:
        raise SortformerTraceError("DEV inference must cover complete frozen sources")
    if role == "PSEM-STRATEGY-TRAIN" and smoke_samples is None:
        raise SortformerTraceError("TRAIN adapter inference must be explicitly bounded as smoke")
    source_receipts: list[dict[str, Any]] = []
    for row in rows:
        source_start, source_end, usage = source_window(row, smoke_samples=smoke_samples)
        run_key = trace_run_key(row, source_start, source_end)
        source_root = safe_child(
            root,
            Path(SORTFORMER_FAMILY) / str(model_cfg["backend"]) / role / run_key,
            "Sortformer source trace root",
        )
        source_root.mkdir(parents=True, exist_ok=True)
        receipt_path = source_root / "receipt.json"
        audio = source_root / "input.wav"
        inference_audio = (
            _load_resume_inference_audio(
                receipt_path, audio, source_start, source_end
            )
            if resume
            else None
        )
        if resume:
            if inference_audio is not None:
                previous = _resume_receipt(
                    receipt_path,
                    row=row,
                    source_start_sample=source_start,
                    source_end_sample=source_end,
                    source_root=source_root,
                    bench=bench,
                    model=model,
                    audio=audio,
                    inference_audio=inference_audio,
                )
                if previous is None:
                    raise SortformerTraceError(
                        f"Sortformer resume receipt is invalid: {row['source_id']}"
                    )
                source_receipts.append(previous)
                print(f"{row['source_id']}: resumed verified Sortformer trace", flush=True)
                continue
        audio, inference_audio = _materialize_inference_audio(
            source_root, row, source_start, source_end
        )
        reused = None
        if reuse_r8_cache and usage == "full_frozen_source":
            reused = _reuse_r8(row=row, research=research)
        if reused is None:
            raw_probabilities, telemetry, inference, telemetry_path = _run_new_inference(
                bench=bench,
                model=model,
                audio=audio,
                run_directory=source_root / "run",
            )
        else:
            raw_probabilities, telemetry, inference, telemetry_path = reused
        probabilities = _retain_source_probabilities(
            raw_probabilities, inference_audio
        )
        inference = dict(inference)
        inference["audio"] = inference_audio
        inference["raw_probability_frame_count"] = raw_probabilities.shape[0]
        inference["retained_probability_frame_count"] = probabilities.shape[0]
        trace = _trace(
            row=row,
            probabilities=probabilities,
            source_start_sample=source_start,
            source_end_sample=source_end,
            inference_audio=inference_audio,
        )
        trace_path = source_root / "posterior_trace.npz"
        trace_info = write_trace(trace_path, trace)
        expected_chunks = (
            int(inference_audio["native_frame_count"]) // SORTFORMER_CHUNK_FRAMES
        )
        receipt = {
            "schema_version": "psem.relative_occupancy.model_source_receipt.v1",
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "family": SORTFORMER_FAMILY,
            "source_id": row["source_id"],
            "role": role,
            "usage": usage,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "manifest_row_sha256": row["row_sha256"],
            "waveform_path": str(Path(str(row["audio_path"])).resolve()),
            "waveform_sha256": row["waveform_sha256"],
            "waveform_size_bytes": row["waveform_size_bytes"],
            "inference_audio": inference_audio,
            "source_start_sample": source_start,
            "source_end_sample": source_end,
            "model_path": str(model),
            "model_sha256": sha256_file(model),
            "bench_path": str(bench),
            "bench_relative_path": model_cfg["bench_relative_path"],
            "bench_sha256": sha256_file(bench),
            "source_repository": model_cfg["source_repository"],
            "source_commit": model_cfg["source_commit"],
            "telemetry_patch_sha256": model_cfg["telemetry_patch_sha256"],
            "backend": model_cfg["backend"],
            "backend_resolved": inference["backend_resolved"],
            "threads": model_cfg["threads"],
            "preset": model_cfg["preset"],
            "inference": inference,
            "telemetry": _telemetry_summary(
                telemetry_path, telemetry, expected_chunks
            ),
            "trace": trace_info,
        }
        write_json(receipt_path, receipt)
        source_receipts.append(receipt)
        print(
            f"{row['source_id']}: wrote {probabilities.shape[0]} Sortformer frames",
            flush=True,
        )
    aggregate = {
        "schema_version": "psem.relative_occupancy.model_receipt.v1",
        "family": SORTFORMER_FAMILY,
        "role": role,
        "usage": source_receipts[0]["usage"],
        "model_repository": model_cfg["model_repository"],
        "model_revision": model_cfg["model_revision"],
        "model_filename": model_cfg["model_filename"],
        "model_sha256": model_cfg["model_sha256"],
        "source_repository": model_cfg["source_repository"],
        "source_commit": model_cfg["source_commit"],
        "telemetry_patch_sha256": model_cfg["telemetry_patch_sha256"],
        "bench_relative_path": model_cfg["bench_relative_path"],
        "bench_sha256": model_cfg["bench_sha256"],
        "backend": model_cfg["backend"],
        "threads": model_cfg["threads"],
        "preset": model_cfg["preset"],
        "native_frame_ms": model_cfg["native_frame_ms"],
        "chunk_audio_ms": model_cfg["chunk_audio_ms"],
        "recorded_algorithmic_lookahead_ms": model_cfg["algorithmic_lookahead_ms"],
        "slot_count": model_cfg["slot_count"],
        "slot_validity_metadata": model_cfg["slot_validity_metadata"],
        "source_count": len(source_receipts),
        "source_ids": [receipt["source_id"] for receipt in source_receipts],
        "source_receipts": source_receipts,
        "eval_status": "sealed",
    }
    write_json(aggregate_output, aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("PSEM-STRATEGY-TRAIN", "PSEM-STRATEGY-DEV"),
        required=True,
    )
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--smoke-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-r8-cache", action="store_true")
    args = parser.parse_args()
    try:
        result = run_traces(
            manifest=args.manifest.resolve(),
            role=args.role,
            research=research_root(args.research_root),
            trace_root=args.trace_root.resolve() if args.trace_root else None,
            output=args.output.resolve(),
            source_ids=args.source_id,
            smoke_samples=args.smoke_samples,
            resume=args.resume,
            reuse_r8_cache=args.reuse_r8_cache,
        )
    except (SortformerTraceError, TraceRuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_count": result["source_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
