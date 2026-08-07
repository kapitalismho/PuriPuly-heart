from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from experiments.speaker_turn_boundary.events import (
    DetectorProgress,
    SpeakerBoundaryEvent,
)
from experiments.speaker_turn_boundary.frontend import (
    LS_EEND_CONV_DELAY,
    LS_EEND_MODEL_INPUT_DIM,
    Resampler16k8k,
    StreamingLSEENDFrontend,
    frontend_profile,
)
from experiments.speaker_turn_boundary.reducer import (
    ReducedBoundary,
    ReductionProfile,
    StreamingReducer,
)


class LSEENDError(RuntimeError):
    pass


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float32)))


def load_sidecar(metadata_path: Path) -> dict[str, Any]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class LSEENDOnnxDetector:
    def __init__(
        self,
        onnx_path: Path,
        metadata: dict[str, Any],
        profile: ReductionProfile,
        *,
        checkpoint_variant: str = "",
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
    ) -> None:
        self._onnx_path = Path(onnx_path)
        self._metadata = metadata
        self._profile = profile
        self._checkpoint_variant = checkpoint_variant
        self._full_output_dim = int(metadata["full_output_dim"])
        self._real_output_dim = int(metadata["real_output_dim"])
        if self._real_output_dim != self._full_output_dim - 2:
            raise LSEENDError(
                f"real_output_dim {self._real_output_dim} must equal "
                f"full_output_dim - 2 ({self._full_output_dim - 2})"
            )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        self._session = ort.InferenceSession(
            str(self._onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._output_names = [output.name for output in self._session.get_outputs()]
        self._state_shapes = {name: list(shape) for name, shape in metadata["state_shapes"].items()}
        self._reducer: StreamingReducer | None = None
        self._resampler: Resampler16k8k | None = None
        self._frontend: StreamingLSEENDFrontend | None = None
        self._state: dict[str, np.ndarray] | None = None
        self._audio_epoch: int | None = None
        self._decoded_frames = 0
        self._decoded_frame_frontier: list[int] = []
        self._boundary_count_before = 0
        self._finalized = False

    @property
    def profile(self) -> ReductionProfile:
        return self._profile

    def start_epoch(self, audio_epoch: int) -> None:
        self._audio_epoch = audio_epoch
        self._resampler = Resampler16k8k()
        self._frontend = StreamingLSEENDFrontend()
        self._state = {
            name: np.zeros(shape, dtype=np.float32) for name, shape in self._state_shapes.items()
        }
        self._decoded_frames = 0
        self._decoded_frame_frontier = []
        self._boundary_count_before = 0
        self._finalized = False
        self._reducer = None

    def _ensure_reducer(self) -> StreamingReducer:
        if self._reducer is None:
            raise LSEENDError("start_epoch must be called before processing")
        return self._reducer

    def _ingest_feature(self, feature: np.ndarray) -> list[SpeakerBoundaryEvent]:
        if self._resampler is None or self._frontend is None or self._state is None:
            raise LSEENDError("start_epoch must be called before processing")
        ingested_index = self._decoded_frames
        should_decode = 1.0 if ingested_index >= LS_EEND_CONV_DELAY else 0.0
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
                "ingest": np.array([1.0], dtype=np.float32),
                "decode": np.array([should_decode], dtype=np.float32),
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
        self._decoded_frames += 1
        if should_decode == 0.0:
            return []
        decoded_frame = ingested_index - LS_EEND_CONV_DELAY
        if self._resampler is None:
            raise LSEENDError("resampler missing")
        self._decoded_frame_frontier.append(self._resampler.input_count)
        probabilities = sigmoid(named["full_logits"])[0, 0, 1:-1]
        return self._emit_decoded_frame(decoded_frame, probabilities)

    def _emit_decoded_frame(
        self, decoded_frame: int, probabilities: np.ndarray
    ) -> list[SpeakerBoundaryEvent]:
        reducer = self._ensure_reducer()
        reducer.emit(decoded_frame, probabilities)
        events: list[SpeakerBoundaryEvent] = []
        boundaries = reducer.boundaries
        for boundary in boundaries[self._boundary_count_before :]:
            events.append(self._event_for_boundary(boundary, self._decoded_frame_frontier[-1]))
        self._boundary_count_before = len(boundaries)
        return events

    def _event_for_boundary(
        self, boundary: ReducedBoundary, observed_frontier: int
    ) -> SpeakerBoundaryEvent:
        return SpeakerBoundaryEvent(
            audio_epoch=self._audio_epoch,
            boundary_source_sample=boundary.boundary_source_sample(),
            observed_source_sample_at_emit=observed_frontier,
            emitted_monotonic_ns=0,
            confidence=boundary.confidence,
            source=f"ls_eend:{self._profile.profile_id}",
            debug={
                "checkpoint": self._checkpoint_variant or self._metadata.get("checkpoint_variant"),
                "profile": self._profile.to_dict(),
                "track_index": boundary.track_index,
                "onset_output_frame": boundary.onset_output_frame,
                "confirmed_output_frame": boundary.confirmed_output_frame,
                "debug": boundary.debug,
            },
        )

    def process_chunk(
        self, chunk: np.ndarray
    ) -> tuple[list[SpeakerBoundaryEvent], DetectorProgress]:
        if self._resampler is None or self._frontend is None:
            raise LSEENDError("start_epoch must be called before processing")
        if self._finalized:
            raise LSEENDError("epoch already finalized")
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self._reducer is None:
            self._reducer = StreamingReducer(
                self._profile,
                track_count=self._real_output_dim,
                audio_epoch=self._audio_epoch,
                sample_count_at_epoch_end=0,
            )
        events: list[SpeakerBoundaryEvent] = []
        resampled = self._resampler.push(chunk)
        features = self._frontend.push_audio(resampled)
        for feature in features:
            events.extend(self._ingest_feature(feature))
        progress = DetectorProgress(
            audio_epoch=self._audio_epoch,
            observed_source_sample=self._resampler.input_count,
            safe_boundary_frontier_sample=min(
                self._ensure_reducer().safe_boundary_frontier_sample(),
                self._resampler.input_count,
            ),
        )
        return events, progress

    def finalize(self) -> tuple[list[SpeakerBoundaryEvent], DetectorProgress]:
        if self._resampler is None or self._frontend is None:
            raise LSEENDError("start_epoch must be called before processing")
        if self._finalized:
            raise LSEENDError("epoch already finalized")
        self._finalized = True
        events: list[SpeakerBoundaryEvent] = []
        features = self._frontend.finalize()
        for feature in features:
            events.extend(self._ingest_feature(feature))
        epoch_end_count = self._resampler.input_count
        pending = self._decoded_frames - len(self._decoded_frame_frontier)
        tail: list[np.ndarray] = []
        reducer = self._ensure_reducer()
        for _ in range(pending):
            outputs = self._session.run(
                self._output_names,
                {
                    "frame": np.zeros((1, 1, LS_EEND_MODEL_INPUT_DIM), dtype=np.float32),
                    "enc_ret_kv": self._state["enc_ret_kv"],
                    "enc_ret_scale": self._state["enc_ret_scale"],
                    "enc_conv_cache": self._state["enc_conv_cache"],
                    "dec_ret_kv": self._state["dec_ret_kv"],
                    "dec_ret_scale": self._state["dec_ret_scale"],
                    "top_buffer": self._state["top_buffer"],
                    "ingest": np.array([0.0], dtype=np.float32),
                    "decode": np.array([1.0], dtype=np.float32),
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
            self._decoded_frame_frontier.append(epoch_end_count)
            ingested_index = self._decoded_frames
            self._decoded_frames += 1
            if self._frontend.total_8k_count > 0:
                from experiments.speaker_turn_boundary.frontend import (
                    model_input_frame_center_8k,
                )

                center_8k = model_input_frame_center_8k(ingested_index)
                if center_8k >= self._frontend.total_8k_count:
                    continue
            tail.append(sigmoid(named["full_logits"])[0, 0, 1:-1])
        if tail:
            reducer.emit_final_tail(np.stack(tail, axis=0), epoch_end_count=epoch_end_count)
        else:
            reducer.finalize(epoch_end_count=epoch_end_count)
        boundaries = reducer.boundaries
        for boundary in boundaries[self._boundary_count_before :]:
            events.append(self._event_for_boundary(boundary, epoch_end_count))
        self._boundary_count_before = len(boundaries)
        progress = DetectorProgress(
            audio_epoch=self._audio_epoch,
            observed_source_sample=epoch_end_count,
            safe_boundary_frontier_sample=reducer.safe_boundary_frontier_sample(),
        )
        return events, progress

    def run_case(
        self, samples_16k: np.ndarray, *, chunk_samples: int = 512
    ) -> tuple[list[SpeakerBoundaryEvent], list[DetectorProgress]]:
        boundaries: list[SpeakerBoundaryEvent] = []
        progress: list[DetectorProgress] = []
        offset = 0
        while offset < samples_16k.size:
            chunk = samples_16k[offset : offset + chunk_samples]
            emitted, snapshot = self.process_chunk(chunk)
            boundaries.extend(emitted)
            progress.append(snapshot)
            offset += chunk_samples
        emitted, snapshot = self.finalize()
        boundaries.extend(emitted)
        progress.append(snapshot)
        return boundaries, progress

    @property
    def frontend_profile_dict(self) -> dict[str, object]:
        return frontend_profile().to_dict()
