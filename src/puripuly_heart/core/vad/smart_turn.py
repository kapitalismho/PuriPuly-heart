from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast
from uuid import UUID

import numpy as np

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadEvent
from puripuly_heart.core.vad.smart_turn_features import compute_whisper_log_mel_features

SmartTurnStage = Literal["off", "shadow", "active"]
SmartTurnLog = Callable[[str], object]

SMART_TURN_SAMPLE_RATE_HZ = 16000
SMART_TURN_MAX_AUDIO_MS = 8000
SMART_TURN_PROBE_SILENCE_MS = (224, 416, 608)
SMART_TURN_HARD_END_MS = 800


@dataclass(frozen=True, slots=True)
class SmartTurnExperimentConfig:
    stage: SmartTurnStage = "off"
    threshold: float = 0.5
    probe_silence_ms: tuple[int, ...] = SMART_TURN_PROBE_SILENCE_MS
    hard_end_ms: int = SMART_TURN_HARD_END_MS
    model_path: Path | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"off", "shadow", "active"}:
            raise ValueError("Smart Turn stage must be off, shadow, or active")
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Smart Turn threshold must be in 0.0..1.0")
        if not self.probe_silence_ms:
            raise ValueError("Smart Turn requires at least one probe")
        if any(value <= 0 for value in self.probe_silence_ms):
            raise ValueError("Smart Turn probe times must be positive")
        if tuple(sorted(self.probe_silence_ms)) != self.probe_silence_ms:
            raise ValueError("Smart Turn probe times must be ordered")
        if self.hard_end_ms <= self.probe_silence_ms[-1]:
            raise ValueError("Smart Turn hard end must follow the final probe")

    @classmethod
    def from_environment(cls) -> SmartTurnExperimentConfig:
        stage_value = os.environ.get("PURIPULY_SMART_TURN_STAGE", "off").strip().lower()
        stage: SmartTurnStage = cast(
            SmartTurnStage,
            stage_value if stage_value in {"off", "shadow", "active"} else "off",
        )
        try:
            threshold = float(os.environ.get("PURIPULY_SMART_TURN_THRESHOLD", "0.5"))
        except ValueError:
            threshold = 0.5
        model_value = os.environ.get("PURIPULY_SMART_TURN_MODEL_PATH", "").strip()
        return cls(
            stage=stage,
            threshold=threshold,
            model_path=Path(model_value) if model_value else None,
        )


@dataclass(frozen=True, slots=True)
class SmartTurnPrediction:
    score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("Smart Turn score must be finite")


class SmartTurnInferencePort(Protocol):
    async def predict(
        self,
        audio: np.ndarray,
        *,
        sample_rate_hz: int,
    ) -> SmartTurnPrediction: ...


class SmartTurnVadControlPort(Protocol):
    def reset(self) -> None: ...


class VadEventSink(Protocol):
    async def handle_vad_event(self, event: VadEvent) -> None: ...


def prepare_smart_turn_audio(audio: np.ndarray, *, sample_rate_hz: int) -> np.ndarray:
    if sample_rate_hz != SMART_TURN_SAMPLE_RATE_HZ:
        raise ValueError("Smart Turn audio must use 16 kHz sampling")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Smart Turn audio must be one-dimensional, got {audio.shape}")
    max_samples = SMART_TURN_SAMPLE_RATE_HZ * SMART_TURN_MAX_AUDIO_MS // 1000
    if audio.size > max_samples:
        return audio[-max_samples:].copy()
    if audio.size < max_samples:
        return np.pad(audio, (max_samples - audio.size, 0), mode="constant")
    return audio.copy()


class SmartTurnOnnxInference:
    def __init__(self, model_path: Path, *, cpu_count: int = 1) -> None:
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Smart Turn model does not exist: {model_path}")
        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = max(1, int(cpu_count))
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(model_path), sess_options=session_options)
        self._session_lock = threading.Lock()

    async def predict(
        self,
        audio: np.ndarray,
        *,
        sample_rate_hz: int,
    ) -> SmartTurnPrediction:
        return await asyncio.to_thread(self._predict_sync, audio, sample_rate_hz)

    def _predict_sync(self, audio: np.ndarray, sample_rate_hz: int) -> SmartTurnPrediction:
        prepared = prepare_smart_turn_audio(audio, sample_rate_hz=sample_rate_hz)
        features = compute_whisper_log_mel_features(prepared)
        with self._session_lock:
            outputs = self._session.run(None, {"input_features": np.expand_dims(features, axis=0)})
        if not outputs:
            raise RuntimeError("Smart Turn ONNX model returned no outputs")
        score = float(np.asarray(outputs[0]).reshape(-1)[0])
        return SmartTurnPrediction(score=score)


class _LazySmartTurnInference:
    def __init__(self, model_path: Path | None) -> None:
        self._model_path = model_path
        self._model: SmartTurnOnnxInference | None = None
        self._initialization_lock = asyncio.Lock()

    async def predict(
        self,
        audio: np.ndarray,
        *,
        sample_rate_hz: int,
    ) -> SmartTurnPrediction:
        model = self._model
        if model is None:
            async with self._initialization_lock:
                model = self._model
                if model is None:
                    if self._model_path is None:
                        raise RuntimeError(
                            "Smart Turn is enabled but PURIPULY_SMART_TURN_MODEL_PATH is unset"
                        )
                    model = await asyncio.to_thread(SmartTurnOnnxInference, self._model_path)
                    self._model = model
        return await model.predict(audio, sample_rate_hz=sample_rate_hz)


class _NoopSmartTurnVadControl:
    def reset(self) -> None:
        return None


class SmartTurnEndpointPolicy:
    def __init__(
        self,
        *,
        downstream: VadEventSink,
        vad_control: SmartTurnVadControlPort,
        inference: SmartTurnInferencePort,
        config: SmartTurnExperimentConfig,
        log_detailed: SmartTurnLog | None = None,
        channel_label: str = "self",
        sample_rate_hz: int = SMART_TURN_SAMPLE_RATE_HZ,
    ) -> None:
        if config.stage == "off":
            raise ValueError("Smart Turn endpoint policy requires shadow or active stage")
        if sample_rate_hz != SMART_TURN_SAMPLE_RATE_HZ:
            raise ValueError("Smart Turn endpoint policy requires 16 kHz audio")
        self._downstream = downstream
        self._vad_control = vad_control
        self._inference = inference
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        self._log_detailed = log_detailed or (lambda _message: None)
        self._channel_label = channel_label
        self._lock = asyncio.Lock()
        self._probe_scope = LifecycleScope(f"smart-turn-{channel_label}")
        self._utterance_id: UUID | None = None
        self._audio_parts: list[np.ndarray] = []
        self._silence_ms = 0
        self._next_probe_index = 0
        self._generation = 0
        self._probe_tasks: set[asyncio.Task[None]] = set()
        self._hard_end_task: asyncio.Task[None] | None = None
        self._ended_ids: set[UUID] = set()

    async def handle_vad_event(self, event: VadEvent) -> None:
        async with self._lock:
            if isinstance(event, SpeechStart):
                await self._handle_speech_start(event)
                return
            if isinstance(event, SpeechChunk):
                await self._handle_speech_chunk(event)
                return
            if isinstance(event, SpeechEnd):
                await self._handle_speech_end(event)
                return
            raise TypeError(f"Unknown VAD event: {type(event)}")

    async def close(self) -> None:
        async with self._lock:
            self._generation += 1
            self._cancel_probe_tasks()
            self._utterance_id = None
            self._audio_parts.clear()
            self._silence_ms = 0
            self._next_probe_index = 0
        await self._probe_scope.close()

    async def _handle_speech_start(self, event: SpeechStart) -> None:
        self._generation += 1
        self._cancel_probe_tasks()
        self._utterance_id = event.utterance_id
        self._audio_parts = [
            np.asarray(event.pre_roll, dtype=np.float32).reshape(-1).copy(),
            np.asarray(event.chunk, dtype=np.float32).reshape(-1).copy(),
        ]
        self._silence_ms = 0
        self._next_probe_index = 0
        self._ended_ids.discard(event.utterance_id)
        await self._downstream.handle_vad_event(event)

    async def _handle_speech_chunk(self, event: SpeechChunk) -> None:
        if self._utterance_id != event.utterance_id:
            await self._downstream.handle_vad_event(event)
            return
        self._audio_parts.append(np.asarray(event.chunk, dtype=np.float32).reshape(-1).copy())
        if event.is_speech:
            if self._silence_ms:
                self._generation += 1
                self._cancel_probe_tasks()
                self._next_probe_index = 0
            self._silence_ms = 0
        else:
            self._silence_ms += int(round(event.chunk.size * 1000.0 / self._sample_rate_hz))
            self._schedule_hard_end()
            self._schedule_due_probes()
        await self._downstream.handle_vad_event(event)

    async def _handle_speech_end(self, event: SpeechEnd) -> None:
        if event.utterance_id in self._ended_ids:
            return
        if self._utterance_id != event.utterance_id:
            await self._downstream.handle_vad_event(event)
            return
        self._generation += 1
        self._cancel_probe_tasks()
        self._utterance_id = None
        self._audio_parts.clear()
        self._silence_ms = 0
        self._next_probe_index = 0
        self._ended_ids.add(event.utterance_id)
        self._trim_ended_ids()
        await self._downstream.handle_vad_event(event)

    def _schedule_due_probes(self) -> None:
        while self._next_probe_index < len(self._config.probe_silence_ms):
            probe_ms = self._config.probe_silence_ms[self._next_probe_index]
            if self._silence_ms < probe_ms:
                return
            self._next_probe_index += 1
            utterance_id = self._utterance_id
            if utterance_id is None:
                return
            audio = np.concatenate(self._audio_parts) if self._audio_parts else np.empty(0)
            task = start_lifecycle_task(
                self._probe_scope,
                self._run_probe(
                    utterance_id=utterance_id,
                    generation=self._generation,
                    probe_ms=probe_ms,
                    audio=audio,
                ),
                name=f"probe-{self._generation}-{probe_ms}",
            )
            self._track_probe_task(task)

    def _schedule_hard_end(self) -> None:
        if self._config.stage != "active" or self._hard_end_task is not None:
            return
        utterance_id = self._utterance_id
        if utterance_id is None:
            return
        remaining_ms = max(0, self._config.hard_end_ms - self._silence_ms)
        task = start_lifecycle_task(
            self._probe_scope,
            self._run_hard_end(
                utterance_id=utterance_id,
                generation=self._generation,
                delay_ms=remaining_ms,
            ),
            name=f"hard-end-{self._generation}",
        )
        self._track_probe_task(task, hard_end=True)

    def _track_probe_task(
        self,
        task: asyncio.Task[None],
        *,
        hard_end: bool = False,
    ) -> None:
        self._probe_tasks.add(task)
        if hard_end:
            self._hard_end_task = task
        task.add_done_callback(self._on_probe_task_done)

    def _on_probe_task_done(self, task: asyncio.Task[None]) -> None:
        self._probe_tasks.discard(task)
        if self._hard_end_task is task:
            self._hard_end_task = None

    async def _run_probe(
        self,
        *,
        utterance_id: UUID,
        generation: int,
        probe_ms: int,
        audio: np.ndarray,
    ) -> None:
        started_at = time.perf_counter()
        try:
            prediction = await self._inference.predict(
                audio,
                sample_rate_hz=self._sample_rate_hz,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._log(
                "inference_error"
                f" channel={self._channel_label}"
                f" probe_ms={probe_ms}"
                f" error={type(exc).__name__}"
            )
            return
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        async with self._lock:
            if generation != self._generation or utterance_id != self._utterance_id:
                self._log(
                    "probe_stale"
                    f" channel={self._channel_label}"
                    f" probe_ms={probe_ms}"
                    f" inference_ms={elapsed_ms}"
                )
                return
            accepted = prediction.score >= self._config.threshold
            self._log(
                "probe"
                f" channel={self._channel_label}"
                f" probe_ms={probe_ms}"
                f" score={prediction.score:.6f}"
                f" threshold={self._config.threshold:.6f}"
                f" accepted={accepted}"
                f" inference_ms={elapsed_ms}"
                f" stage={self._config.stage}"
            )
            if self._config.stage == "active" and accepted:
                await self._emit_early_end(utterance_id, probe_ms)

    async def _run_hard_end(
        self,
        *,
        utterance_id: UUID,
        generation: int,
        delay_ms: int,
    ) -> None:
        try:
            await asyncio.sleep(delay_ms / 1000.0)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if generation != self._generation or utterance_id != self._utterance_id:
                return
            await self._emit_boundary(
                utterance_id,
                trailing_silence_ms=self._config.hard_end_ms,
                boundary_label="hard_end",
            )

    async def _emit_early_end(self, utterance_id: UUID, probe_ms: int) -> None:
        if self._utterance_id != utterance_id or utterance_id in self._ended_ids:
            return
        await self._emit_boundary(
            utterance_id,
            trailing_silence_ms=probe_ms,
            boundary_label="early_end",
        )

    async def _emit_boundary(
        self,
        utterance_id: UUID,
        *,
        trailing_silence_ms: int,
        boundary_label: str,
    ) -> None:
        if self._utterance_id != utterance_id or utterance_id in self._ended_ids:
            return
        self._generation += 1
        self._cancel_probe_tasks()
        self._utterance_id = None
        self._audio_parts.clear()
        self._silence_ms = 0
        self._next_probe_index = 0
        self._ended_ids.add(utterance_id)
        self._trim_ended_ids()
        try:
            self._vad_control.reset()
        except Exception as exc:
            self._log(
                "vad_reset_error" f" channel={self._channel_label}" f" error={type(exc).__name__}"
            )
        self._log(
            f"{boundary_label}"
            f" channel={self._channel_label}"
            f" trailing_silence_ms={trailing_silence_ms}"
        )
        await self._downstream.handle_vad_event(
            SpeechEnd(
                utterance_id=utterance_id,
                trailing_silence_ms=trailing_silence_ms,
                reason="silence",
            )
        )

    def _cancel_probe_tasks(self) -> None:
        current = asyncio.current_task()
        for task in tuple(self._probe_tasks):
            if task is not current and not task.done():
                task.cancel()
        self._probe_tasks.clear()
        self._hard_end_task = None

    def _trim_ended_ids(self) -> None:
        while len(self._ended_ids) > 32:
            self._ended_ids.pop()

    def _log(self, message: str) -> None:
        try:
            self._log_detailed(f"[SmartTurn] {message}")
        except Exception:
            return


SmartTurnEventSinkFactory = Callable[..., VadEventSink]


def create_smart_turn_event_sink_factory(
    *,
    log_detailed: SmartTurnLog | None = None,
) -> SmartTurnEventSinkFactory:
    cached_inference: dict[str, SmartTurnInferencePort] = {}

    def create(
        *,
        sink: VadEventSink,
        vad: object,
        config: SmartTurnExperimentConfig,
        channel_label: str,
        sample_rate_hz: int,
    ) -> VadEventSink:
        if config.stage == "off":
            return sink
        cache_key = str(config.model_path) if config.model_path is not None else "<unset>"
        inference = cached_inference.get(cache_key)
        if inference is None:
            inference = _LazySmartTurnInference(config.model_path)
            cached_inference[cache_key] = inference
        vad_control = (
            cast(SmartTurnVadControlPort, vad)
            if callable(getattr(vad, "reset", None))
            else _NoopSmartTurnVadControl()
        )
        return SmartTurnEndpointPolicy(
            downstream=sink,
            vad_control=vad_control,
            inference=inference,
            config=config,
            log_detailed=log_detailed,
            channel_label=channel_label,
            sample_rate_hz=sample_rate_hz,
        )

    return create


__all__ = [
    "SMART_TURN_HARD_END_MS",
    "SMART_TURN_MAX_AUDIO_MS",
    "SMART_TURN_PROBE_SILENCE_MS",
    "SMART_TURN_SAMPLE_RATE_HZ",
    "SmartTurnEndpointPolicy",
    "SmartTurnExperimentConfig",
    "SmartTurnInferencePort",
    "SmartTurnLog",
    "SmartTurnOnnxInference",
    "SmartTurnPrediction",
    "SmartTurnStage",
    "SmartTurnVadControlPort",
    "SmartTurnEventSinkFactory",
    "create_smart_turn_event_sink_factory",
    "prepare_smart_turn_audio",
]
