from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import experiments.psem_relative_occupancy_gate.run_lseend_trace as lseend_module
import experiments.psem_relative_occupancy_gate.run_sortformer_trace as sortformer_module
from experiments.psem_relative_occupancy_gate import model_run_io
from experiments.psem_relative_occupancy_gate.contracts import Trace
from experiments.psem_relative_occupancy_gate.io_utils import (
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.psem_relative_occupancy_gate.run_lseend_trace import (
    LSEENDTraceError,
    lseend_frame_geometry,
)
from experiments.psem_relative_occupancy_gate.run_sortformer_trace import (
    sortformer_frame_geometry,
)
from experiments.psem_relative_occupancy_gate.trace_io import (
    TRACE_SCHEMA_VERSION,
    TraceIOError,
    load_trace,
    write_trace,
)
from experiments.psem_relative_occupancy_gate.trace_runtime import (
    TraceRuntimeError,
    backend_resolution_matches,
    validate_full_trace_geometry,
    validate_trace_location,
)


def _fixture_trace() -> Trace:
    slot_ids = ("slot-0", "slot-1")
    return Trace(
        source_id="fixture",
        family="fixture_family",
        slot_ids=slot_ids,
        probabilities=np.asarray([[0.25, 0.75], [0.5, 0.5]], dtype=np.float32),
        frame_start_samples=np.asarray([0, 1600], dtype=np.int64),
        frame_end_samples=np.asarray([1600, 3200], dtype=np.int64),
        evidence_frontier_samples=np.asarray([2000, 3600], dtype=np.int64),
        slot_alive=np.ones((2, 2), dtype=np.bool_),
        state_reset=np.asarray([True, False], dtype=np.bool_),
        metadata={
            "schema_version": TRACE_SCHEMA_VERSION,
            "source_id": "fixture",
            "family": "fixture_family",
            "sample_rate_hz": 16000,
            "speaker_slot_ids": list(slot_ids),
        },
    )


def test_trace_archive_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_receipt = write_trace(first, _fixture_trace())
    second_receipt = write_trace(second, _fixture_trace())
    assert first.read_bytes() == second.read_bytes()
    assert sha256_file(first) == sha256_file(second)
    assert first_receipt["trace_sha256"] == second_receipt["trace_sha256"]
    loaded = load_trace(first)
    assert loaded.slot_ids == ("slot-0", "slot-1")
    np.testing.assert_array_equal(loaded.probabilities, _fixture_trace().probabilities)


def test_trace_writer_rejects_metadata_identity_drift(tmp_path: Path) -> None:
    trace = _fixture_trace()
    trace.metadata["source_id"] = "different"
    with pytest.raises(TraceIOError, match="source_id"):
        write_trace(tmp_path / "trace.npz", trace)


def test_full_trace_geometry_rejects_truncation_and_gaps() -> None:
    trace = Trace(
        source_id="source",
        family="streaming_sortformer",
        slot_ids=("slot-0", "slot-1"),
        probabilities=np.ones((3, 2), dtype=np.float32) * 0.5,
        frame_start_samples=np.asarray([0, 1280, 2560], dtype=np.int64),
        frame_end_samples=np.asarray([1280, 2560, 3200], dtype=np.int64),
        evidence_frontier_samples=np.asarray([1280, 2560, 3200], dtype=np.int64),
        slot_alive=np.ones((3, 2), dtype=np.bool_),
        state_reset=np.asarray([True, False, False], dtype=np.bool_),
        metadata={"source_start_sample": 0, "source_end_sample": 3200},
    )
    validate_full_trace_geometry(
        trace,
        family="streaming_sortformer",
        source_start_sample=0,
        source_end_sample=3200,
    )
    truncated = Trace(
        source_id=trace.source_id,
        family=trace.family,
        slot_ids=trace.slot_ids,
        probabilities=trace.probabilities[:2],
        frame_start_samples=trace.frame_start_samples[:2],
        frame_end_samples=trace.frame_end_samples[:2],
        evidence_frontier_samples=trace.evidence_frontier_samples[:2],
        slot_alive=trace.slot_alive[:2],
        state_reset=trace.state_reset[:2],
        metadata=trace.metadata,
    )
    with pytest.raises(TraceRuntimeError, match="incomplete"):
        validate_full_trace_geometry(
            truncated,
            family="streaming_sortformer",
            source_start_sample=0,
            source_end_sample=3200,
        )
    gapped = Trace(
        source_id=trace.source_id,
        family=trace.family,
        slot_ids=trace.slot_ids,
        probabilities=trace.probabilities,
        frame_start_samples=np.asarray([0, 1290, 2560], dtype=np.int64),
        frame_end_samples=trace.frame_end_samples,
        evidence_frontier_samples=trace.evidence_frontier_samples,
        slot_alive=trace.slot_alive,
        state_reset=trace.state_reset,
        metadata=trace.metadata,
    )
    with pytest.raises(TraceRuntimeError, match="not contiguous"):
        validate_full_trace_geometry(
            gapped,
            family="streaming_sortformer",
            source_start_sample=0,
            source_end_sample=3200,
        )


def test_trace_location_is_bound_to_family_backend_role(tmp_path: Path) -> None:
    valid = (
        tmp_path
        / "streaming_sortformer"
        / "vulkan"
        / "PSEM-STRATEGY-DEV"
        / "source"
        / "posterior_trace.npz"
    )
    valid.parent.mkdir(parents=True)
    valid.write_bytes(b"trace")
    assert validate_trace_location(
        valid,
        family="streaming_sortformer",
        backend="vulkan",
        role="PSEM-STRATEGY-DEV",
        source_id="source",
    ) == valid.resolve()
    with pytest.raises(TraceRuntimeError, match="family/backend role"):
        validate_trace_location(
            valid,
            family="streaming_sortformer",
            backend="cpu",
            role="PSEM-STRATEGY-DEV",
            source_id="source",
        )


def test_sortformer_geometry_uses_scalar_chunk_frontier() -> None:
    starts, ends, frontiers = sortformer_frame_geometry(32, 40000, 1000)
    assert starts[:7].tolist() == [1000, 2280, 3560, 4840, 6120, 7400, 8680]
    assert ends[-1] == 41000
    assert frontiers[:6].tolist() == [17680] * 6
    assert frontiers[6] == 25360
    assert frontiers[-1] == 41000


def test_sortformer_materializes_frozen_source_with_zero_flush(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    samples = np.arange(4000, dtype=np.int16)
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.tobytes())
    row = {
        "audio_path": str(source),
        "source_duration_samples": 3200,
    }
    path, receipt = sortformer_module._materialize_inference_audio(
        tmp_path / "trace", row, 0, 3200
    )
    with wave.open(str(path), "rb") as reader:
        observed = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16)
        assert reader.getnframes() == 15360
    np.testing.assert_array_equal(observed[:3200], samples[:3200])
    np.testing.assert_array_equal(observed[3200:], np.zeros(12160, dtype=np.int16))
    assert receipt["sample_count"] == 15360
    assert receipt["source_sample_count"] == 3200
    assert receipt["trailing_zero_sample_count"] == 12160
    assert receipt["native_mel_frame_count"] == 96
    assert receipt["native_frame_count"] == 12
    assert receipt["retained_frame_count"] == 3
    assert receipt["source_end_sample"] == 3200
    assert receipt["materialization"] == sortformer_module.INFERENCE_AUDIO_MATERIALIZATION


def test_sortformer_failed_eval_source_zero_flush_geometry() -> None:
    geometry = sortformer_module.sortformer_inference_audio_geometry(28_329_539)
    assert geometry == {
        "sample_count": 28_339_200,
        "source_sample_count": 28_329_539,
        "trailing_zero_sample_count": 9_661,
        "source_mel_frame_count": 177_060,
        "native_mel_frame_count": 177_120,
        "retained_frame_count": 22_133,
        "native_frame_count": 22_140,
        "materialization": sortformer_module.INFERENCE_AUDIO_MATERIALIZATION,
    }


def test_sortformer_retains_only_frozen_source_probability_frames() -> None:
    geometry = sortformer_module.sortformer_inference_audio_geometry(3200)
    raw = np.arange(48, dtype=np.float32).reshape(12, 4)
    retained = sortformer_module._retain_source_probabilities(raw, geometry)
    np.testing.assert_array_equal(retained, raw[:3])
    assert not np.shares_memory(retained, raw)
    with pytest.raises(
        sortformer_module.SortformerTraceError,
        match="native probability frame count mismatch",
    ):
        sortformer_module._retain_source_probabilities(raw[:-1], geometry)


def test_sortformer_resume_rejects_missing_or_changed_materialized_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    samples = np.arange(4000, dtype=np.int16)
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.tobytes())
    source_root = tmp_path / "trace"
    audio_path, inference_audio = sortformer_module._materialize_inference_audio(
        source_root,
        {"audio_path": str(source), "source_duration_samples": 3200},
        0,
        3200,
    )
    receipt_path = source_root / "receipt.json"
    write_json(receipt_path, {"inference_audio": inference_audio})
    audio_path.unlink()
    with pytest.raises(sortformer_module.SortformerTraceError, match="audio is invalid"):
        sortformer_module._load_resume_inference_audio(receipt_path, audio_path, 0, 3200)
    audio_path, _ = sortformer_module._materialize_inference_audio(
        source_root,
        {"audio_path": str(source), "source_duration_samples": 3200},
        0,
        3200,
    )
    with wave.open(str(audio_path), "rb") as reader:
        changed = np.frombuffer(
            reader.readframes(reader.getnframes()), dtype=np.int16
        ).copy()
    changed[0] += 1
    with wave.open(str(audio_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(changed.tobytes())
    with pytest.raises(sortformer_module.SortformerTraceError, match="binding mismatch"):
        sortformer_module._load_resume_inference_audio(receipt_path, audio_path, 0, 3200)


def test_sortformer_resume_reruns_legacy_materialization(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    write_json(
        receipt_path,
        {
            "inference_audio": {
                "materialization": "exact_pcm16_mono_16khz_frozen_source_window"
            }
        },
    )
    assert (
        sortformer_module._load_resume_inference_audio(
            receipt_path, tmp_path / "missing.wav", 0, 3200
        )
        is None
    )


def test_model_run_io_binds_sortformer_materialized_input(tmp_path: Path) -> None:
    source_path = tmp_path / "source.wav"
    samples = np.arange(3200, dtype=np.int16)
    with wave.open(str(source_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.tobytes())
    audio_path, binding = sortformer_module._materialize_inference_audio(
        tmp_path / "trace",
        {"audio_path": str(source_path), "source_duration_samples": 3200},
        0,
        3200,
    )
    observed_path, observed_binding = model_run_io._sortformer_inference_audio(
        {"inference_audio": binding}, {"audio": binding}, 3200
    )
    assert observed_path == audio_path.resolve()
    assert observed_binding == binding
    with wave.open(str(audio_path), "rb") as reader:
        changed = np.frombuffer(
            reader.readframes(reader.getnframes()), dtype=np.int16
        ).copy()
    changed[0] += 1
    with wave.open(str(audio_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(changed.tobytes())
    with pytest.raises(model_run_io.ModelRunError, match="binding mismatch"):
        model_run_io._sortformer_inference_audio(
            {"inference_audio": binding}, {"audio": binding}, 3200
        )


def test_lseend_geometry_preserves_native_centers_and_actual_frontiers() -> None:
    starts, ends, frontiers = lseend_frame_geometry(
        2,
        20000,
        1000,
        np.asarray([15872, 17408], dtype=np.int64),
    )
    assert starts.tolist() == [14631, 16231]
    assert ends.tolist() == [16231, 17831]
    assert frontiers.tolist() == [16872, 18408]


def test_lseend_geometry_rejects_future_input_hidden_by_early_frontier() -> None:
    with pytest.raises(LSEENDTraceError, match="required model input"):
        lseend_frame_geometry(1, 20000, 0, np.asarray([15805], dtype=np.int64))


@pytest.mark.parametrize(
    ("requested", "resolved", "expected"),
    [
        ("cpu", "CPU", True),
        ("vulkan", "Vulkan0", True),
        ("vulkan", "vulkan", True),
        ("cuda", "CUDA12", True),
        ("vulkan", "Vulkan-CPU", False),
        ("cpu", "CPU0", False),
    ],
)
def test_backend_resolution_matching(
    requested: str, resolved: str, expected: bool
) -> None:
    assert backend_resolution_matches(requested, resolved) is expected


def test_sortformer_resume_rejects_backend_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = {
        "model_sha256": "model",
        "model_revision": "model-revision",
        "source_repository": "source-repository",
        "source_commit": "source-commit",
        "telemetry_patch_sha256": "telemetry-patch",
        "bench_relative_path": "external/build/transcribe-bench.exe",
        "bench_sha256": "bench",
        "backend": "vulkan",
        "threads": 8,
        "preset": "low_latency",
    }
    row = {
        "source_id": "source",
        "role": "PSEM-STRATEGY-DEV",
        "row_sha256": "row",
        "waveform_sha256": "waveform",
        "waveform_size_bytes": 7,
        "source_duration_samples": 3200,
    }
    source_root = (
        tmp_path
        / "streaming_sortformer"
        / "vulkan"
        / "PSEM-STRATEGY-DEV"
        / "source"
    )
    (source_root / "run").mkdir(parents=True)
    trace_path = source_root / "posterior_trace.npz"
    trace_path.write_bytes(b"trace")
    bench_path = tmp_path / "transcribe-bench.exe"
    model_path = tmp_path / "model.gguf"
    waveform_path = tmp_path / "audio.wav"
    audio_path = source_root / "input.wav"
    for path in (bench_path, model_path, waveform_path, audio_path):
        path.write_bytes(b"fixture")
    raw_bench_path = source_root / "run" / "bench.json"
    bench_receipt = {
        "backend": "Vulkan0",
        "iters": 1,
        "model_path": str(model_path.resolve()),
        "sample_path": str(audio_path.resolve()),
        "warmup": 0,
    }
    write_json(raw_bench_path, bench_receipt)
    telemetry_path = source_root / "run" / "telemetry.jsonl"
    telemetry_rows = [
        {
            "chunk_index": index,
            "pre_encode_us": 0,
            "infer_us": 0,
            "update_us": 0,
            "compression_called": False,
            "compression_us": 0,
            "total_us": 1000,
            "new_audio_frames": 6,
        }
        for index in range(2)
    ]
    write_jsonl(telemetry_path, telemetry_rows)
    telemetry_receipt = {
        "path": str(telemetry_path.resolve()),
        "sha256": "code",
        "chunk_count": 2,
        "chunk_compute_ms": {
            "p50": 1.0,
            "p90": 1.0,
            "p99": 1.0,
            "maximum": 1.0,
            "sum": 2.0,
        },
        "compression_call_count": 0,
    }
    row["audio_path"] = str(waveform_path.resolve())
    inference_audio = {
        "path": str(audio_path.resolve()),
        "sha256": "code",
        "size_bytes": 7,
        **sortformer_module.sortformer_inference_audio_geometry(3200),
        "source_start_sample": 0,
        "source_end_sample": 3200,
    }
    metadata = {
        "source_id": "source",
        "family": "streaming_sortformer",
        "role": "PSEM-STRATEGY-DEV",
        "manifest_row_sha256": "row",
        "waveform_sha256": "waveform",
        "inference_audio_path": str(audio_path.resolve()),
        "inference_audio_sha256": "code",
        "inference_audio_sample_count": inference_audio["sample_count"],
        "inference_audio_source_sample_count": 3200,
        "inference_audio_trailing_zero_sample_count": inference_audio[
            "trailing_zero_sample_count"
        ],
        "inference_audio_native_frame_count": inference_audio["native_frame_count"],
        "inference_audio_retained_frame_count": inference_audio[
            "retained_frame_count"
        ],
        "inference_audio_materialization": sortformer_module.INFERENCE_AUDIO_MATERIALIZATION,
        "source_start_sample": 0,
        "source_end_sample": 3200,
        "model_sha256": "model",
        "model_revision": "model-revision",
        "source_commit": "source-commit",
        "telemetry_patch_sha256": "telemetry-patch",
        "bench_relative_path": "external/build/transcribe-bench.exe",
        "bench_sha256": "bench",
        "backend": "vulkan",
        "threads": 8,
        "preset": "low_latency",
        "adapter_code_sha256": "code",
        "trace_io_code_sha256": "code",
        "contracts_code_sha256": "code",
    }
    receipt = {
        "schema_version": "psem.relative_occupancy.model_source_receipt.v1",
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "family": "streaming_sortformer",
        "source_id": "source",
        "role": "PSEM-STRATEGY-DEV",
        "usage": "full_frozen_source",
        "manifest_row_sha256": "row",
        "waveform_sha256": "waveform",
        "waveform_size_bytes": 7,
        "inference_audio": inference_audio,
        "source_start_sample": 0,
        "source_end_sample": 3200,
        "model_sha256": "model",
        "bench_relative_path": "external/build/transcribe-bench.exe",
        "bench_sha256": "bench",
        "source_repository": "source-repository",
        "source_commit": "source-commit",
        "telemetry_patch_sha256": "telemetry-patch",
        "backend": "vulkan",
        "backend_resolved": "Vulkan0",
        "threads": 8,
        "preset": "low_latency",
        "bench_path": str(bench_path.resolve()),
        "model_path": str(model_path.resolve()),
        "waveform_path": str(waveform_path.resolve()),
        "inference": {
            "backend_resolved": "Vulkan0",
            "audio": inference_audio,
            "raw_probability_frame_count": inference_audio["native_frame_count"],
            "retained_probability_frame_count": inference_audio[
                "retained_frame_count"
            ],
            "bench": bench_receipt,
            "bench_path": str(raw_bench_path.resolve()),
            "bench_sha256": "code",
            "command": [
                str(bench_path.resolve()),
                "--model",
                str(model_path.resolve()),
                "--sample",
                str(audio_path.resolve()),
                "--backend",
                "vulkan",
                "--threads",
                "8",
                "--warmup",
                "0",
                "--iters",
                "1",
                "--json-out",
                str(raw_bench_path.resolve()),
            ],
        },
        "telemetry": telemetry_receipt,
        "trace": {"trace_path": str(trace_path)},
    }
    receipt_path = tmp_path / "sortformer-receipt.json"
    write_json(receipt_path, receipt)
    monkeypatch.setattr(sortformer_module, "config", lambda: {"sortformer": model})
    monkeypatch.setattr(sortformer_module, "sha256_file", lambda _: "code")
    monkeypatch.setattr(
        sortformer_module,
        "validate_trace_receipt",
        lambda *_: SimpleNamespace(metadata=metadata),
    )
    monkeypatch.setattr(sortformer_module, "validate_trace_location", lambda path, **_: path)
    monkeypatch.setattr(sortformer_module, "validate_full_trace_geometry", lambda *_args, **_kwargs: None)
    assert (
        sortformer_module._resume_receipt(
            receipt_path,
            row=row,
            source_start_sample=0,
            source_end_sample=3200,
            source_root=source_root,
            bench=bench_path.resolve(),
            model=model_path.resolve(),
            audio=audio_path.resolve(),
            inference_audio=inference_audio,
        )
        == receipt
    )
    raw_bench_path.unlink()
    assert (
        sortformer_module._resume_receipt(
            receipt_path,
            row=row,
            source_start_sample=0,
            source_end_sample=3200,
            source_root=source_root,
            bench=bench_path.resolve(),
            model=model_path.resolve(),
            audio=audio_path.resolve(),
            inference_audio=inference_audio,
        )
        is None
    )
    write_json(raw_bench_path, bench_receipt)
    telemetry_path.unlink()
    assert (
        sortformer_module._resume_receipt(
            receipt_path,
            row=row,
            source_start_sample=0,
            source_end_sample=3200,
            source_root=source_root,
            bench=bench_path.resolve(),
            model=model_path.resolve(),
            audio=audio_path.resolve(),
            inference_audio=inference_audio,
        )
        is None
    )
    write_jsonl(telemetry_path, telemetry_rows)
    receipt["backend_resolved"] = "CPU"
    receipt["inference"]["backend_resolved"] = "CPU"
    receipt["inference"]["bench"]["backend"] = "CPU"
    write_json(receipt_path, receipt)
    assert (
        sortformer_module._resume_receipt(
            receipt_path,
            row=row,
            source_start_sample=0,
            source_end_sample=3200,
            source_root=source_root,
            bench=bench_path.resolve(),
            model=model_path.resolve(),
            audio=audio_path.resolve(),
            inference_audio=inference_audio,
        )
        is None
    )
    receipt["backend_resolved"] = "Vulkan0"
    receipt["inference"]["backend_resolved"] = "Vulkan0"
    receipt["inference"]["bench"]["backend"] = "Vulkan0"
    receipt["backend"] = "cpu"
    write_json(receipt_path, receipt)
    assert (
        sortformer_module._resume_receipt(
            receipt_path,
            row=row,
            source_start_sample=0,
            source_end_sample=3200,
            source_root=source_root,
            bench=bench_path.resolve(),
            model=model_path.resolve(),
            audio=audio_path.resolve(),
            inference_audio=inference_audio,
        )
        is None
    )


def test_lseend_resume_rejects_sidecar_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = {
        "model_sha256": "model",
        "sidecar_sha256": "sidecar",
        "repository": "repository",
        "revision": "revision",
        "backend": "CPUExecutionProvider",
        "intra_op_threads": 1,
        "inter_op_threads": 1,
    }
    row = {
        "source_id": "source",
        "role": "PSEM-STRATEGY-DEV",
        "row_sha256": "row",
        "waveform_sha256": "waveform",
        "source_duration_samples": 3200,
    }
    source_root = (
        tmp_path
        / "ls_eend"
        / "CPUExecutionProvider"
        / "PSEM-STRATEGY-DEV"
        / "source"
    )
    source_root.mkdir(parents=True)
    trace_path = source_root / "posterior_trace.npz"
    trace_path.write_bytes(b"trace")
    model_path = tmp_path / "model.onnx"
    sidecar_path = tmp_path / "model.json"
    model_path.write_bytes(b"model")
    sidecar_path.write_bytes(b"sidecar")
    metadata = {
        "source_id": "source",
        "family": "ls_eend",
        "role": "PSEM-STRATEGY-DEV",
        "manifest_row_sha256": "row",
        "waveform_sha256": "waveform",
        "source_start_sample": 0,
        "source_end_sample": 3200,
        "model_sha256": "model",
        "model_revision": "revision",
        "sidecar_sha256": "sidecar",
        "backend": "CPUExecutionProvider",
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "adapter_code_sha256": "code",
        "trace_io_code_sha256": "code",
        "contracts_code_sha256": "code",
        "neutral_capture_code_sha256": "code",
        "frontend_code_sha256": "code",
    }
    receipt = {
        "schema_version": "psem.relative_occupancy.model_source_receipt.v1",
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "family": "ls_eend",
        "source_id": "source",
        "role": "PSEM-STRATEGY-DEV",
        "usage": "full_frozen_source",
        "manifest_row_sha256": "row",
        "waveform_sha256": "waveform",
        "source_start_sample": 0,
        "source_end_sample": 3200,
        "model_sha256": "model",
        "sidecar_sha256": "sidecar",
        "repository": "repository",
        "revision": "revision",
        "backend": "CPUExecutionProvider",
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "model_path": str(model_path.resolve()),
        "sidecar_path": str(sidecar_path.resolve()),
        "runtime": {"providers": ["CPUExecutionProvider"]},
        "trace": {"trace_path": str(trace_path)},
    }
    receipt_path = tmp_path / "lseend-receipt.json"
    write_json(receipt_path, receipt)
    monkeypatch.setattr(lseend_module, "config", lambda: {"lseend": model})
    monkeypatch.setattr(lseend_module, "sha256_file", lambda _: "code")
    monkeypatch.setattr(
        lseend_module,
        "validate_trace_receipt",
        lambda *_: SimpleNamespace(metadata=metadata),
    )
    monkeypatch.setattr(lseend_module, "validate_trace_location", lambda path, **_: path)
    monkeypatch.setattr(lseend_module, "validate_full_trace_geometry", lambda *_args, **_kwargs: None)
    assert (
        lseend_module._resume_receipt(
            receipt_path,
            row=row,
            source_start_sample=0,
            source_end_sample=3200,
            source_root=source_root,
            model=model_path.resolve(),
            sidecar_path=sidecar_path.resolve(),
        )
        == receipt
    )
    receipt["runtime"]["providers"] = ["CUDAExecutionProvider"]
    write_json(receipt_path, receipt)
    assert (
        lseend_module._resume_receipt(
            receipt_path,
            row=row,
            source_start_sample=0,
            source_end_sample=3200,
            source_root=source_root,
            model=model_path.resolve(),
            sidecar_path=sidecar_path.resolve(),
        )
        is None
    )
    receipt["runtime"]["providers"] = ["CPUExecutionProvider"]
    receipt["sidecar_sha256"] = "different"
    write_json(receipt_path, receipt)
    assert (
        lseend_module._resume_receipt(
            receipt_path,
            row=row,
            source_start_sample=0,
            source_end_sample=3200,
            source_root=source_root,
            model=model_path.resolve(),
            sidecar_path=sidecar_path.resolve(),
        )
        is None
    )
