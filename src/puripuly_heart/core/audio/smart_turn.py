from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

import numpy as np

from puripuly_heart.core.audio.smart_turn_features import compute_whisper_log_mel_features

SMART_TURN_MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_RESOURCE_RELATIVE_PATH = f"data/models/{SMART_TURN_MODEL_FILENAME}"
SMART_TURN_INPUT_REVISION = "8dd248b8f73556ac32d24c00223b4b413d4aca98"
SMART_TURN_SAMPLE_RATE_HZ = 16000
SMART_TURN_WINDOW_SAMPLES = 8 * SMART_TURN_SAMPLE_RATE_HZ
SMART_TURN_PREPARE_TIMEOUT_S = 60.0
SMART_TURN_COMPLETE_THRESHOLD = 0.75
SMART_TURN_RESOURCE_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
SMART_TURN_SUPPORTED_LANGUAGES = frozenset(
    {
        "ar",
        "zh",
        "da",
        "nl",
        "de",
        "en",
        "fi",
        "fr",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "no",
        "pl",
        "pt",
        "ru",
        "es",
        "tr",
        "uk",
        "vi",
    }
)
SmartTurnAvailability = Literal[
    "disabled",
    "unloaded",
    "missing",
    "loading",
    "ready",
    "error",
    "closed",
]


logger = logging.getLogger(__name__)


def smart_turn_language_profile(source_mode: str, language: str) -> tuple[str, float | None]:
    if source_mode == "auto":
        return ("on", SMART_TURN_COMPLETE_THRESHOLD)
    normalized = language.strip().lower().replace("_", "-")
    base = normalized.split("-", 1)[0]
    if base not in SMART_TURN_SUPPORTED_LANGUAGES:
        return "unsupported_language", None
    return ("on", SMART_TURN_COMPLETE_THRESHOLD)


def prepare_smart_turn_audio(audio: np.ndarray, *, sample_rate_hz: int) -> np.ndarray:
    if sample_rate_hz != SMART_TURN_SAMPLE_RATE_HZ:
        raise ValueError("Smart Turn audio must use 16 kHz sampling")
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1:
        raise ValueError(f"Smart Turn audio must be one-dimensional, got {value.shape}")
    if value.size > SMART_TURN_WINDOW_SAMPLES:
        return value[-SMART_TURN_WINDOW_SAMPLES:].copy()
    if value.size < SMART_TURN_WINDOW_SAMPLES:
        prepared = np.empty(SMART_TURN_WINDOW_SAMPLES, dtype=np.float32)
        padding = SMART_TURN_WINDOW_SAMPLES - value.size
        prepared[:padding] = 0.0
        prepared[padding:] = value
        return prepared
    return value.copy()


@dataclass(frozen=True, slots=True)
class SmartTurnRequestIdentity:
    activation_generation: int
    segment_id: UUID
    pause_id: int
    context_revision: int
    input_revision: str
    source_frontier: int
    probe_frontier_monotonic_s: float
    complete_deadline_monotonic_s: float


@dataclass(frozen=True, slots=True)
class SmartTurnCompletion:
    identity: SmartTurnRequestIdentity
    score: float | None
    completed_at_monotonic_s: float
    duration_s: float
    outcome: Literal["complete", "error", "nonfinite"]


@dataclass(frozen=True, slots=True)
class SmartTurnRuntimeSnapshot:
    availability: SmartTurnAvailability
    inference_count: int
    busy_skip_count: int
    late_count: int
    active_request: SmartTurnRequestIdentity | None
    last_error: str | None


class SmartTurnInferencePort(Protocol):
    async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float: ...


async def _await_owned_operation(awaitable):
    operation = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        await operation
        raise


class SmartTurnOnnxInference:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def _predict_blocking(self, audio: np.ndarray) -> float:
        prepared = prepare_smart_turn_audio(audio, sample_rate_hz=SMART_TURN_SAMPLE_RATE_HZ)
        features = compute_whisper_log_mel_features(prepared)
        session = self._session
        if session is None:
            raise RuntimeError("Smart Turn ONNX session is closed")
        outputs = session.run(None, {"input_features": np.expand_dims(features, axis=0)})
        if not outputs:
            raise RuntimeError("Smart Turn ONNX model returned no outputs")
        return float(np.asarray(outputs[0]).reshape(-1)[0])

    async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
        if sample_rate_hz != SMART_TURN_SAMPLE_RATE_HZ:
            raise ValueError("Smart Turn audio must use 16 kHz sampling")
        return await _await_owned_operation(asyncio.to_thread(self._predict_blocking, audio))

    def close(self) -> None:
        self._session = None


def bundled_smart_turn_onnx_path() -> Path:
    model_path = Path(__file__).resolve().parents[2] / SMART_TURN_RESOURCE_RELATIVE_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Bundled Smart Turn model missing: {SMART_TURN_RESOURCE_RELATIVE_PATH}"
        )
    return model_path


class SmartTurnInferenceOwner:
    def __init__(
        self,
        *,
        model_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        inference_factory: Callable[[Path], SmartTurnInferencePort] = SmartTurnOnnxInference,
        prepare_timeout_s: float = SMART_TURN_PREPARE_TIMEOUT_S,
    ) -> None:
        if prepare_timeout_s <= 0:
            raise ValueError("Smart Turn preparation timeout must be positive")
        self._model_path = model_path
        self._clock = clock
        self._inference_factory = inference_factory
        self._prepare_timeout_s = prepare_timeout_s
        self._availability: SmartTurnAvailability = "unloaded"
        self._inference: SmartTurnInferencePort | None = None
        self._prepare_task: asyncio.Task[None] | None = None
        self._execution_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._prepare_attempted = False
        self._active_request: SmartTurnRequestIdentity | None = None
        self._closed = False
        self._inference_count = 0
        self._busy_skip_count = 0
        self._late_count = 0
        self._last_error: str | None = None

    @property
    def snapshot(self) -> SmartTurnRuntimeSnapshot:
        return SmartTurnRuntimeSnapshot(
            availability=self._availability,
            inference_count=self._inference_count,
            busy_skip_count=self._busy_skip_count,
            late_count=self._late_count,
            active_request=self._active_request,
            last_error=self._last_error,
        )

    def request_prepare(self) -> None:
        if (
            self._closed
            or self._inference is not None
            or self._prepare_attempted
            or self._prepare_task is not None
        ):
            return
        self._prepare_attempted = True
        self._availability = "loading"
        self._prepare_task = asyncio.create_task(self._prepare(), name="SmartTurn:prepare")

    def submit(
        self,
        identity: SmartTurnRequestIdentity,
        audio: np.ndarray,
        completion: Callable[[SmartTurnCompletion], Awaitable[None]],
    ) -> Literal["started", "busy", "unavailable"]:
        if self._closed:
            return "unavailable"
        if self._execution_task is not None and not self._execution_task.done():
            self._busy_skip_count += 1
            return "busy"
        if self._inference is None:
            self.request_prepare()
            return "unavailable"
        owned_audio = np.array(audio, dtype=np.float32, order="C", copy=True).reshape(-1)
        self._active_request = identity
        self._execution_task = asyncio.create_task(
            self._execute(identity, owned_audio, completion),
            name=f"SmartTurn:infer:{identity.segment_id}:{identity.pause_id}",
        )
        return "started"

    def record_late(self) -> None:
        self._late_count += 1

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(
                self._close_owned_resources(),
                name="SmartTurn:close",
            )
        await asyncio.shield(self._close_task)

    async def _close_owned_resources(self) -> None:
        try:
            prepare = self._prepare_task
            if prepare is not None:
                await _await_owned_operation(asyncio.gather(prepare, return_exceptions=True))
            execution = self._execution_task
            if execution is not None:
                await _await_owned_operation(asyncio.gather(execution, return_exceptions=True))
        finally:
            self._availability = "closed"
            inference = self._inference
            self._inference = None
            close = getattr(inference, "close", None)
            if callable(close):
                close()

    async def _prepare(self) -> None:
        if self._closed:
            return
        operation = asyncio.create_task(
            self._construct_inference(),
            name="SmartTurn:prepare-resource",
        )
        timed_out = False
        try:
            done, _pending = await asyncio.wait(
                {operation},
                timeout=self._prepare_timeout_s,
                return_when=asyncio.ALL_COMPLETED,
            )
            if operation not in done:
                timed_out = True
                self._availability = "error"
                self._last_error = "TimeoutError"
                logger.warning("[STT][Runtime] smart-turn error error=TimeoutError")
            try:
                inference = await _await_owned_operation(operation)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not timed_out:
                    self._availability = "error"
                    self._last_error = type(exc).__name__
                    logger.warning("[STT][Runtime] smart-turn error error=%s", type(exc).__name__)
                return
            if inference is None:
                return
            if timed_out or self._closed:
                close = getattr(inference, "close", None)
                if callable(close):
                    close()
                return
            self._inference = inference
            self._availability = "ready"
            self._last_error = None
        finally:
            self._prepare_task = None

    async def _construct_inference(self) -> SmartTurnInferencePort | None:
        if self._closed:
            return None
        return await _await_owned_operation(asyncio.to_thread(self._load_inference))

    def _load_inference(self) -> SmartTurnInferencePort:
        model_path = self._model_path or bundled_smart_turn_onnx_path()
        return self._inference_factory(model_path)

    async def _execute(
        self,
        identity: SmartTurnRequestIdentity,
        audio: np.ndarray,
        completion: Callable[[SmartTurnCompletion], Awaitable[None]],
    ) -> None:
        started = self._clock()
        outcome: Literal["complete", "error", "nonfinite"] = "error"
        score: float | None = None
        error: str | None = None
        try:
            assert self._inference is not None
            score = await _await_owned_operation(
                self._inference.predict(
                    audio,
                    sample_rate_hz=SMART_TURN_SAMPLE_RATE_HZ,
                )
            )
            outcome = "complete" if math.isfinite(score) else "nonfinite"
        except asyncio.CancelledError:
            self._active_request = None
            self._execution_task = None
            raise
        except Exception as exc:
            self._last_error = type(exc).__name__
            error = self._last_error
        completed = self._clock()
        self._inference_count += 1
        result = SmartTurnCompletion(
            identity=identity,
            score=score,
            completed_at_monotonic_s=completed,
            duration_s=max(0.0, completed - started),
            outcome=outcome,
        )
        if outcome != "complete":
            logger.warning(
                "[STT][Runtime] smart-turn inference failed cause=%s",
                error or outcome,
            )
        self._active_request = None
        try:
            await completion(result)
        finally:
            self._execution_task = None


__all__ = [
    "SMART_TURN_INPUT_REVISION",
    "SMART_TURN_MODEL_FILENAME",
    "SMART_TURN_RESOURCE_RELATIVE_PATH",
    "SMART_TURN_RESOURCE_SHA256",
    "SMART_TURN_COMPLETE_THRESHOLD",
    "SMART_TURN_SUPPORTED_LANGUAGES",
    "SmartTurnCompletion",
    "SmartTurnInferenceOwner",
    "SmartTurnOnnxInference",
    "SmartTurnRequestIdentity",
    "SmartTurnRuntimeSnapshot",
    "bundled_smart_turn_onnx_path",
    "prepare_smart_turn_audio",
    "smart_turn_language_profile",
    "SMART_TURN_PREPARE_TIMEOUT_S",
]
