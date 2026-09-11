from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

import httpx
import numpy as np

from puripuly_heart.config.paths import default_models_dir
from puripuly_heart.core.audio.smart_turn_features import compute_whisper_log_mel_features

SMART_TURN_MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_MODEL_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
SMART_TURN_MODEL_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
)
SMART_TURN_INPUT_REVISION = "8dd248b8f73556ac32d24c00223b4b413d4aca98"
SMART_TURN_SAMPLE_RATE_HZ = 16000
SMART_TURN_WINDOW_SAMPLES = 8 * SMART_TURN_SAMPLE_RATE_HZ
SMART_TURN_THRESHOLDS = {
    "ko": 0.967305183,
    "ja": 0.844703436,
    "en": 0.772239923,
    "zh": 0.925585747,
}
SmartTurnAvailability = Literal[
    "disabled", "missing", "loading", "ready", "artifact_mismatch", "error", "closed"
]


def smart_turn_language_profile(source_mode: str, language: str) -> tuple[str, float | None]:
    if source_mode != "manual":
        return "unsupported_auto", None
    normalized = language.strip().lower().replace("_", "-")
    base = normalized.split("-", 1)[0]
    threshold = SMART_TURN_THRESHOLDS.get(base)
    return ("on", threshold) if threshold is not None else ("unsupported_language", None)


def prepare_smart_turn_audio(audio: np.ndarray, *, sample_rate_hz: int) -> np.ndarray:
    if sample_rate_hz != SMART_TURN_SAMPLE_RATE_HZ:
        raise ValueError("Smart Turn audio must use 16 kHz sampling")
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1:
        raise ValueError(f"Smart Turn audio must be one-dimensional, got {value.shape}")
    if value.size > SMART_TURN_WINDOW_SAMPLES:
        return value[-SMART_TURN_WINDOW_SAMPLES:].copy()
    if value.size < SMART_TURN_WINDOW_SAMPLES:
        return np.pad(value, (SMART_TURN_WINDOW_SAMPLES - value.size, 0), mode="constant")
    return value.copy()


def default_smart_turn_model_path() -> Path:
    configured = os.environ.get("PURIPULY_SMART_TURN_MODEL_PATH", "").strip()
    return Path(configured) if configured else default_models_dir() / SMART_TURN_MODEL_FILENAME


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

        if _sha256_file(model_path) != SMART_TURN_MODEL_SHA256:
            raise ValueError("Smart Turn artifact checksum mismatch")
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
        prepared = prepare_smart_turn_audio(audio, sample_rate_hz=sample_rate_hz)
        features = await _await_owned_operation(
            asyncio.to_thread(compute_whisper_log_mel_features, prepared)
        )
        session = self._session
        if session is None:
            raise RuntimeError("Smart Turn ONNX session is closed")
        outputs = await _await_owned_operation(
            asyncio.to_thread(
                session.run,
                None,
                {"input_features": np.expand_dims(features, axis=0)},
            )
        )
        if not outputs:
            raise RuntimeError("Smart Turn ONNX model returned no outputs")
        return float(np.asarray(outputs[0]).reshape(-1)[0])

    def close(self) -> None:
        self._session = None


class SmartTurnInferenceOwner:
    def __init__(
        self,
        *,
        model_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        inference_factory: Callable[[Path], SmartTurnInferencePort] = SmartTurnOnnxInference,
        downloader: Callable[[Path], Awaitable[None]] | None = None,
    ) -> None:
        self._model_path = model_path or default_smart_turn_model_path()
        self._clock = clock
        self._inference_factory = inference_factory
        self._downloader = downloader or self._download
        self._availability: SmartTurnAvailability = "missing"
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
        if self._model_path.is_file():
            self._availability = "loading"

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
        owned_audio = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
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
            inference = self._inference
            self._inference = None
            close = getattr(inference, "close", None)
            if callable(close):
                close()
            self._availability = "closed"

    async def _prepare(self) -> None:
        try:
            if self._closed:
                return
            if not self._model_path.is_file():
                await self._downloader(self._model_path)
            if self._closed:
                return
            digest = await _await_owned_operation(asyncio.to_thread(_sha256_file, self._model_path))
            if digest != SMART_TURN_MODEL_SHA256:
                self._availability = "artifact_mismatch"
                self._last_error = "artifact_mismatch"
                return
            if self._closed:
                return
            inference = await _await_owned_operation(
                asyncio.to_thread(self._inference_factory, self._model_path)
            )
            if self._closed:
                close = getattr(inference, "close", None)
                if callable(close):
                    close()
                return
            self._inference = inference
            self._availability = "ready"
            self._last_error = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._availability = "error"
            self._last_error = type(exc).__name__
        finally:
            self._prepare_task = None

    async def _execute(
        self,
        identity: SmartTurnRequestIdentity,
        audio: np.ndarray,
        completion: Callable[[SmartTurnCompletion], Awaitable[None]],
    ) -> None:
        started = self._clock()
        outcome: Literal["complete", "error", "nonfinite"] = "error"
        score: float | None = None
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
        completed = self._clock()
        self._inference_count += 1
        result = SmartTurnCompletion(
            identity=identity,
            score=score,
            completed_at_monotonic_s=completed,
            duration_s=max(0.0, completed - started),
            outcome=outcome,
        )
        self._active_request = None
        try:
            await completion(result)
        finally:
            self._execution_task = None

    async def _download(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_suffix(destination.suffix + ".part")
        digest = hashlib.sha256()
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                async with client.stream("GET", SMART_TURN_MODEL_URL) as response:
                    response.raise_for_status()
                    with staging.open("wb") as handle:
                        async for chunk in response.aiter_bytes():
                            handle.write(chunk)
                            digest.update(chunk)
            if digest.hexdigest() != SMART_TURN_MODEL_SHA256:
                raise ValueError("Smart Turn artifact checksum mismatch")
            staging.replace(destination)
        finally:
            if staging.exists():
                staging.unlink()


__all__ = [
    "SMART_TURN_INPUT_REVISION",
    "SMART_TURN_MODEL_FILENAME",
    "SMART_TURN_MODEL_SHA256",
    "SMART_TURN_MODEL_URL",
    "SMART_TURN_THRESHOLDS",
    "SmartTurnCompletion",
    "SmartTurnInferenceOwner",
    "SmartTurnOnnxInference",
    "SmartTurnRequestIdentity",
    "SmartTurnRuntimeSnapshot",
    "default_smart_turn_model_path",
    "prepare_smart_turn_audio",
    "smart_turn_language_profile",
]
