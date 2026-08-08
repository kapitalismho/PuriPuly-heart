from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from experiments.speaker_turn_boundary.adapters.ls_eend import sigmoid
from experiments.speaker_turn_boundary.events import (
    DetectorProgress,
    SpeakerBoundaryEvent,
)
from experiments.speaker_turn_boundary.frontend import (
    LS_EEND_CONV_DELAY,
    LS_EEND_MODEL_INPUT_DIM,
    Resampler16k8k,
    StreamingLSEENDFrontend,
    model_input_frame_center_8k,
)
from experiments.speaker_turn_boundary.reducer import (
    ReducedBoundary,
    ReductionProfile,
    StreamingReducer,
)
from experiments.speaker_turn_boundary.schemas import canonical_json, sha256_hex


class LSCaptureError(RuntimeError):
    pass


LS_CAPTURE_CACHE_SCHEMA = "experiments.speaker_turn_boundary.phase3.ls_capture_cache.v2"


@dataclass(slots=True)
class LSCaptureEpoch:
    case_id: str
    audio_epoch: int
    normal_probs: list[np.ndarray] = field(default_factory=list)
    normal_frontiers: list[int] = field(default_factory=list)
    frame_wall_ns: list[int] = field(default_factory=list)
    tail_probs: list[np.ndarray] = field(default_factory=list)
    epoch_end_count: int = 0
    finalize_wall_ns: int = 0
    chunk_observed_counts: list[int] = field(default_factory=list)
    chunk_wall_seconds: list[float] = field(default_factory=list)
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    length_samples: int = 0

    @property
    def track_count(self) -> int:
        if self.normal_probs:
            return int(self.normal_probs[0].size)
        if self.tail_probs:
            return int(self.tail_probs[0].size)
        return 0


class LSEENDCapture:
    def __init__(
        self,
        onnx_path: Path,
        metadata: dict[str, Any],
        *,
        checkpoint_variant: str = "",
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
    ) -> None:
        self._onnx_path = Path(onnx_path)
        self._metadata = metadata
        self._checkpoint_variant = checkpoint_variant
        self._full_output_dim = int(metadata["full_output_dim"])
        self._real_output_dim = int(metadata["real_output_dim"])
        if self._real_output_dim != self._full_output_dim - 2:
            raise LSCaptureError(
                f"real_output_dim {self._real_output_dim} must equal "
                f"full_output_dim - 2 ({self._full_output_dim - 2})"
            )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        load_start = time.perf_counter()
        self._session = ort.InferenceSession(
            str(self._onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.load_seconds = time.perf_counter() - load_start
        self._output_names = [output.name for output in self._session.get_outputs()]
        self._state_shapes = {name: list(shape) for name, shape in metadata["state_shapes"].items()}
        self._capture: LSCaptureEpoch | None = None
        self._resampler: Resampler16k8k | None = None
        self._frontend: StreamingLSEENDFrontend | None = None
        self._state: dict[str, np.ndarray] | None = None
        self._decoded_frames = 0
        self._audio_epoch: int | None = None

    @property
    def real_output_dim(self) -> int:
        return self._real_output_dim

    @property
    def checkpoint_variant(self) -> str:
        return self._checkpoint_variant

    def start_epoch(self, audio_epoch: int, case_id: str) -> None:
        self._audio_epoch = audio_epoch
        self._resampler = Resampler16k8k()
        self._frontend = StreamingLSEENDFrontend()
        self._state = {
            name: np.zeros(shape, dtype=np.float32) for name, shape in self._state_shapes.items()
        }
        self._decoded_frames = 0
        self._capture = LSCaptureEpoch(case_id=case_id, audio_epoch=audio_epoch)

    def _run_session(
        self, feature: np.ndarray, ingest: float, decode: float
    ) -> dict[str, np.ndarray]:
        if self._state is None:
            raise LSCaptureError("start_epoch must be called before processing")
        outputs = self._session.run(
            self._output_names,
            {
                "frame": feature.reshape(1, 1, -1).astype(np.float32, copy=False),
                "enc_ret_kv": self._state["enc_ret_kv"],
                "enc_ret_scale": self._state["enc_ret_scale"],
                "enc_conv_cache": self._state["enc_conv_cache"],
                "dec_ret_kv": self._state["dec_ret_kv"],
                "dec_ret_scale": self._state["dec_ret_scale"],
                "top_buffer": self._state["top_buffer"],
                "ingest": np.array([ingest], dtype=np.float32),
                "decode": np.array([decode], dtype=np.float32),
            },
        )
        named = dict(zip(self._output_names, outputs))
        self._state = {
            "enc_ret_kv": named["enc_ret_kv_out"],
            "enc_ret_scale": named["enc_ret_scale_out"],
            "enc_conv_cache": named["enc_conv_cache_out"],
            "dec_ret_kv": named["dec_ret_kv_out"],
            "dec_ret_scale": named["dec_ret_scale_out"],
            "top_buffer": named["top_buffer_out"],
        }
        return named

    def _ingest_feature(self, feature: np.ndarray) -> None:
        if self._resampler is None or self._capture is None:
            raise LSCaptureError("start_epoch must be called before processing")
        capture = self._capture
        ingested_index = self._decoded_frames
        should_decode = 1.0 if ingested_index >= LS_EEND_CONV_DELAY else 0.0
        named = self._run_session(feature, ingest=1.0, decode=should_decode)
        self._decoded_frames += 1
        if should_decode == 0.0:
            return
        capture.normal_frontiers.append(self._resampler.input_count)
        capture.frame_wall_ns.append(time.perf_counter_ns())
        probabilities = sigmoid(named["full_logits"])[0, 0, 1:-1]
        capture.normal_probs.append(probabilities)

    def process_chunk(self, chunk: np.ndarray) -> None:
        if self._resampler is None or self._frontend is None or self._capture is None:
            raise LSCaptureError("start_epoch must be called before processing")
        capture = self._capture
        started = time.perf_counter()
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        resampled = self._resampler.push(chunk)
        features = self._frontend.push_audio(resampled)
        for feature in features:
            self._ingest_feature(feature)
        capture.chunk_observed_counts.append(self._resampler.input_count)
        capture.chunk_wall_seconds.append(time.perf_counter() - started)

    def finalize(self) -> None:
        if self._resampler is None or self._frontend is None or self._capture is None:
            raise LSCaptureError("start_epoch must be called before processing")
        capture = self._capture
        features = self._frontend.finalize()
        for feature in features:
            self._ingest_feature(feature)
        epoch_end_count = self._resampler.input_count
        capture.epoch_end_count = epoch_end_count
        pending = self._decoded_frames - len(capture.normal_frontiers)
        for _ in range(pending):
            named = self._run_session(
                np.zeros(LS_EEND_MODEL_INPUT_DIM, dtype=np.float32), ingest=0.0, decode=1.0
            )
            ingested_index = self._decoded_frames
            self._decoded_frames += 1
            if self._frontend.total_8k_count > 0:
                center_8k = model_input_frame_center_8k(ingested_index)
                if center_8k >= self._frontend.total_8k_count:
                    continue
            capture.tail_probs.append(sigmoid(named["full_logits"])[0, 0, 1:-1])
        capture.finalize_wall_ns = time.perf_counter_ns()

    def run_case(
        self, samples_16k: np.ndarray, *, case_id: str, audio_epoch: int, chunk_samples: int = 512
    ) -> LSCaptureEpoch:
        self.start_epoch(audio_epoch, case_id)
        capture = self._capture
        if capture is None:
            raise LSCaptureError("capture missing")
        capture.length_samples = int(samples_16k.size)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        offset = 0
        while offset < samples_16k.size:
            chunk = samples_16k[offset : offset + chunk_samples]
            self.process_chunk(chunk)
            offset += chunk_samples
        self.finalize()
        capture.wall_seconds = time.perf_counter() - wall_start
        capture.cpu_seconds = time.process_time() - cpu_start
        return capture


def replay_profile(
    capture: LSCaptureEpoch,
    profile: ReductionProfile,
    *,
    track_count: int,
    source_label: str,
) -> tuple[list[SpeakerBoundaryEvent], list[DetectorProgress]]:
    reducer = StreamingReducer(
        profile,
        track_count=track_count,
        audio_epoch=capture.audio_epoch,
        sample_count_at_epoch_end=0,
    )
    events: list[SpeakerBoundaryEvent] = []
    progress: list[DetectorProgress] = []
    emitted_before = 0

    def make_event(boundary: ReducedBoundary, observed: int, wall_ns: int) -> SpeakerBoundaryEvent:
        return SpeakerBoundaryEvent(
            audio_epoch=capture.audio_epoch,
            boundary_source_sample=boundary.boundary_source_sample(),
            observed_source_sample_at_emit=observed,
            emitted_monotonic_ns=wall_ns,
            confidence=boundary.confidence,
            source=source_label,
            debug={
                "checkpoint": source_label,
                "profile": profile.to_dict(),
                "track_index": boundary.track_index,
                "onset_output_frame": boundary.onset_output_frame,
                "confirmed_output_frame": boundary.confirmed_output_frame,
                "debug": boundary.debug,
            },
        )

    def emit_frame(index: int) -> None:
        nonlocal emitted_before
        reducer.emit(index, capture.normal_probs[index])
        boundaries = reducer.boundaries
        for boundary in boundaries[emitted_before:]:
            events.append(
                make_event(
                    boundary,
                    capture.normal_frontiers[index],
                    capture.frame_wall_ns[index],
                )
            )
        emitted_before = len(boundaries)

    frontiers = np.asarray(capture.normal_frontiers, dtype=np.int64)
    frame_index = 0
    for observed_count in capture.chunk_observed_counts:
        available = int(np.searchsorted(frontiers, observed_count, side="right"))
        while frame_index < available:
            emit_frame(frame_index)
            frame_index += 1
        progress.append(
            DetectorProgress(
                audio_epoch=capture.audio_epoch,
                observed_source_sample=observed_count,
                safe_boundary_frontier_sample=min(
                    reducer.safe_boundary_frontier_sample(), observed_count
                ),
            )
        )
    while frame_index < len(capture.normal_probs):
        emit_frame(frame_index)
        frame_index += 1
    if capture.tail_probs:
        reducer.emit_final_tail(
            np.stack(capture.tail_probs, axis=0),
            epoch_end_count=capture.epoch_end_count,
        )
    else:
        reducer.finalize(epoch_end_count=capture.epoch_end_count)
    boundaries = reducer.boundaries
    for boundary in boundaries[emitted_before:]:
        events.append(make_event(boundary, capture.epoch_end_count, capture.finalize_wall_ns))
    progress.append(
        DetectorProgress(
            audio_epoch=capture.audio_epoch,
            observed_source_sample=capture.epoch_end_count,
            safe_boundary_frontier_sample=capture.epoch_end_count,
        )
    )
    return events, progress


def load_sidecar_metadata(sidecar_path: Path) -> dict[str, Any]:
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def _stack_probabilities(values: list[np.ndarray], track_count: int) -> np.ndarray:
    if not values:
        return np.zeros((0, track_count), dtype=np.float32)
    return np.stack(values, axis=0).astype(np.float32, copy=False)


def _capture_bytes(capture: LSCaptureEpoch, track_count: int) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        normal_probs=_stack_probabilities(capture.normal_probs, track_count),
        normal_frontiers=np.asarray(capture.normal_frontiers, dtype=np.int64),
        frame_wall_ns=np.asarray(capture.frame_wall_ns, dtype=np.int64),
        tail_probs=_stack_probabilities(capture.tail_probs, track_count),
        chunk_observed_counts=np.asarray(capture.chunk_observed_counts, dtype=np.int64),
        chunk_wall_seconds=np.asarray(capture.chunk_wall_seconds, dtype=np.float64),
        scalar_ints=np.asarray(
            [
                capture.audio_epoch,
                capture.epoch_end_count,
                capture.finalize_wall_ns,
                capture.length_samples,
            ],
            dtype=np.int64,
        ),
        scalar_floats=np.asarray(
            [capture.cpu_seconds, capture.wall_seconds],
            dtype=np.float64,
        ),
    )
    return buffer.getvalue()


def save_capture_cache(
    cache_dir: Path,
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    sidecar_sha256: str,
    frontend_contract_sha256: str,
    manifest_sha256: str,
    case_wav_sha256: dict[str, str],
    captures: list[LSCaptureEpoch],
    track_count: int,
) -> dict[str, Any]:
    root = cache_dir / checkpoint / manifest_sha256[:16]
    root.mkdir(parents=True, exist_ok=True)
    cases: dict[str, Any] = {}
    for capture in captures:
        wav_sha256 = case_wav_sha256[capture.case_id]
        file_name = f"{sanitize_capture_case_id(capture.case_id)}_{wav_sha256[:16]}.npz"
        payload = _capture_bytes(capture, track_count)
        (root / file_name).write_bytes(payload)
        cases[capture.case_id] = {
            "audio_epoch": capture.audio_epoch,
            "wav_sha256": wav_sha256,
            "file": file_name,
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        }
    index: dict[str, Any] = {
        "schema_version": LS_CAPTURE_CACHE_SCHEMA,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "sidecar_sha256": sidecar_sha256,
        "frontend_contract_sha256": frontend_contract_sha256,
        "manifest_sha256": manifest_sha256,
        "track_count": track_count,
        "cases": cases,
    }
    index["contract_sha256"] = sha256_hex(index)
    (root / "index.json").write_text(canonical_json(index), encoding="utf-8")
    return index


def load_capture_cache(
    cache_dir: Path,
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    sidecar_sha256: str,
    frontend_contract_sha256: str,
    manifest_sha256: str,
    case_wav_sha256: dict[str, str],
) -> tuple[list[LSCaptureEpoch], dict[str, Any]] | None:
    root = cache_dir / checkpoint / manifest_sha256[:16]
    index_path = root / "index.json"
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": LS_CAPTURE_CACHE_SCHEMA,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "sidecar_sha256": sidecar_sha256,
        "frontend_contract_sha256": frontend_contract_sha256,
        "manifest_sha256": manifest_sha256,
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise LSCaptureError(f"LS capture cache contract mismatch: {key}")
    stored_hash = index.get("contract_sha256")
    actual_hash = sha256_hex(
        {key: value for key, value in index.items() if key != "contract_sha256"}
    )
    if stored_hash != actual_hash:
        raise LSCaptureError("LS capture cache index hash mismatch")
    if set(index.get("cases") or {}) != set(case_wav_sha256):
        raise LSCaptureError("LS capture cache case set mismatch")
    track_count = int(index["track_count"])
    captures: list[LSCaptureEpoch] = []
    for case_id, case_info in sorted(
        index["cases"].items(), key=lambda item: int(item[1]["audio_epoch"])
    ):
        if case_info.get("wav_sha256") != case_wav_sha256[case_id]:
            raise LSCaptureError(f"LS capture cache WAV hash mismatch: {case_id}")
        path = root / str(case_info["file"])
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != case_info.get("file_sha256"):
            raise LSCaptureError(f"LS capture cache byte hash mismatch: {case_id}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            normal_probs = np.asarray(data["normal_probs"], dtype=np.float32)
            tail_probs = np.asarray(data["tail_probs"], dtype=np.float32)
            if normal_probs.ndim != 2 or normal_probs.shape[1] != track_count:
                raise LSCaptureError(f"LS normal probability shape mismatch: {case_id}")
            if tail_probs.ndim != 2 or tail_probs.shape[1] != track_count:
                raise LSCaptureError(f"LS tail probability shape mismatch: {case_id}")
            scalar_ints = np.asarray(data["scalar_ints"], dtype=np.int64)
            scalar_floats = np.asarray(data["scalar_floats"], dtype=np.float64)
            capture = LSCaptureEpoch(
                case_id=case_id,
                audio_epoch=int(scalar_ints[0]),
                normal_probs=[row.copy() for row in normal_probs],
                normal_frontiers=[int(value) for value in data["normal_frontiers"]],
                frame_wall_ns=[int(value) for value in data["frame_wall_ns"]],
                tail_probs=[row.copy() for row in tail_probs],
                epoch_end_count=int(scalar_ints[1]),
                finalize_wall_ns=int(scalar_ints[2]),
                chunk_observed_counts=[int(value) for value in data["chunk_observed_counts"]],
                chunk_wall_seconds=[float(value) for value in data["chunk_wall_seconds"]],
                cpu_seconds=float(scalar_floats[0]),
                wall_seconds=float(scalar_floats[1]),
                length_samples=int(scalar_ints[3]),
            )
        if len(capture.normal_probs) != len(capture.normal_frontiers):
            raise LSCaptureError(f"LS capture frame/frontier mismatch: {case_id}")
        if len(capture.normal_probs) != len(capture.frame_wall_ns):
            raise LSCaptureError(f"LS capture frame/time mismatch: {case_id}")
        captures.append(capture)
    return captures, index


def sanitize_capture_case_id(case_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case_id)
