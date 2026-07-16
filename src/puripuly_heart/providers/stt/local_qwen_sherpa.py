from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable

import numpy as np

from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
from puripuly_heart.core.audio.format import AudioFrameF32, pcm16le_bytes_to_float32
from puripuly_heart.core.local_qwen_runtime import (
    LocalQwenRuntimeBootstrapError,
    ensure_local_qwen_windows_runtime,
)
from puripuly_heart.core.local_stt_assets import (
    LocalQwenSherpaLoadError,
    validate_local_stt_runtime_ready,
)
from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)
from puripuly_heart.core.stt.local_qwen_hallucination import (
    is_known_local_qwen_hallucination,
)

DEFAULT_SHERPA_NUM_THREADS = 3
LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ = 16000
_KNOWN_HALLUCINATION_LOG_REDACTION = "<known-local-qwen-hallucination>"
logger = logging.getLogger(__name__)


class LocalQwenSherpaInferenceError(RuntimeError):
    """Raised when local sherpa inference fails for an utterance."""


class _LocalQwenSherpaImportError(ImportError):
    """Internal sentinel for sherpa_onnx import failures."""


def _log_prefix(stream_label: str | None) -> str:
    prefix = "[STT][local_qwen]"
    if stream_label:
        return f"{prefix}[{stream_label}]"
    return prefix


def _audio_diag_prefix(stream_label: str | None) -> str:
    prefix = "[AudioDiag][local_qwen]"
    if stream_label:
        return f"{prefix}[{stream_label}]"
    return prefix


def _looks_repetitive(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 6:
        return False
    for unit_len in range(1, (len(stripped) // 2) + 1):
        if len(stripped) % unit_len == 0 and stripped == stripped[:unit_len] * (
            len(stripped) // unit_len
        ):
            return len(stripped) // unit_len >= 3
    if len(stripped) < 12:
        return False
    return len(set(stripped)) <= max(4, len(stripped) // 8)


def _looks_script_mismatched(text: str, language_hint: str | None) -> bool:
    if not text or language_hint != "Korean":
        return False
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    latin = sum("a" <= ch.lower() <= "z" for ch in text)
    return cjk >= 3 or latin >= max(5, len(text) // 2)


def _pcm16le_duration_ms(pcm16le_size_bytes: int, sample_rate_hz: int) -> float:
    if pcm16le_size_bytes <= 0:
        return 0.0
    return _sample_count_duration_ms(pcm16le_size_bytes // 2, sample_rate_hz)


def _sample_count_duration_ms(sample_count: int, sample_rate_hz: int) -> float:
    if sample_count <= 0 or sample_rate_hz <= 0:
        return 0.0
    return sample_count * 1000.0 / float(sample_rate_hz)


def create_local_qwen_sherpa_recognizer(
    *,
    model_dir: Path,
    num_threads: int,
    sample_rate_hz: int = LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ,
    feature_dim: int = 128,
    provider: str = "cpu",
) -> object:
    if sample_rate_hz != LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ:
        raise ValueError(f"sample_rate_hz must be {LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ}")
    ensure_local_qwen_windows_runtime()
    try:
        import sherpa_onnx

        recognizer_module = importlib.import_module("sherpa_onnx.offline_recognizer")
    except ImportError as exc:
        raise _LocalQwenSherpaImportError from exc

    qwen3_config = sherpa_onnx.OfflineQwen3ASRModelConfig(
        conv_frontend=str(model_dir / "conv_frontend.onnx"),
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        tokenizer=str(model_dir / "tokenizer"),
        max_total_len=512,
        max_new_tokens=128,
        temperature=1e-6,
        top_p=0.8,
        seed=42,
    )
    model_config = sherpa_onnx.OfflineModelConfig(
        qwen3_asr=qwen3_config,
        num_threads=num_threads,
        debug=False,
        provider=provider,
    )
    feat_config = sherpa_onnx.FeatureExtractorConfig(
        sampling_rate=sample_rate_hz,
        feature_dim=feature_dim,
    )
    recognizer_config = sherpa_onnx.OfflineRecognizerConfig(
        feat_config=feat_config,
        model_config=model_config,
        decoding_method="greedy_search",
    )
    recognizer_cls = getattr(recognizer_module, "_Recognizer")
    return recognizer_cls(recognizer_config)


@dataclass(slots=True)
class LocalQwenSherpaSTTBackend(STTBackend):
    model_dir: Path
    sample_rate_hz: int = 16000
    num_threads: int = DEFAULT_SHERPA_NUM_THREADS
    feature_dim: int = 128
    provider: str = "cpu"
    stream_label: str | None = None
    language_hint: str | None = None
    hotwords: tuple[str, ...] = ()
    diagnostics_enabled: Callable[[], bool] | None = None
    _recognizer: object | None = field(init=False, default=None, repr=False)
    _load_lock: asyncio.Lock = field(init=False, repr=False)
    _decode_lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sample_rate_hz != LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ:
            raise ValueError(f"sample_rate_hz must be {LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ}")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be > 0")
        self._load_lock = asyncio.Lock()
        self._decode_lock = asyncio.Lock()

    async def open_session(self) -> STTBackendSession:
        await self._ensure_recognizer()
        return _LocalQwenSherpaSession(backend=self)

    async def close(self) -> None:
        self._recognizer = None

    async def _ensure_recognizer(self) -> object:
        if self._recognizer is not None:
            return self._recognizer

        async with self._load_lock:
            if self._recognizer is not None:
                return self._recognizer
            await asyncio.to_thread(validate_local_stt_runtime_ready, self.model_dir)
            self._recognizer = await asyncio.to_thread(self._create_recognizer)
            return self._recognizer

    def _create_recognizer(self) -> object:
        try:
            return create_local_qwen_sherpa_recognizer(
                model_dir=self.model_dir,
                num_threads=self.num_threads,
                sample_rate_hz=LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ,
                feature_dim=self.feature_dim,
                provider=self.provider,
            )
        except LocalQwenRuntimeBootstrapError as exc:
            raise LocalQwenSherpaLoadError(str(exc)) from exc
        except _LocalQwenSherpaImportError as exc:
            raise LocalQwenSherpaLoadError("failed to import sherpa_onnx") from exc.__cause__
        except Exception as exc:
            raise LocalQwenSherpaLoadError(str(exc)) from exc

    async def decode_pcm16le(self, pcm16le: bytes) -> str:
        return await self.decode_f32(pcm16le_bytes_to_float32(pcm16le))

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        recognizer = await self._ensure_recognizer()
        async with self._decode_lock:
            try:
                return await asyncio.to_thread(
                    self._decode_f32_sync,
                    recognizer,
                    samples_f32,
                )
            except Exception as exc:
                raise LocalQwenSherpaInferenceError(str(exc)) from exc

    def _decode_f32_sync(self, recognizer: object, samples_f32: np.ndarray) -> str:
        samples = np.asarray(samples_f32, dtype=np.float32).reshape(-1).copy()
        stream = recognizer.create_stream()
        set_option = getattr(stream, "set_option", None)
        if callable(set_option):
            if self.language_hint:
                set_option("language", self.language_hint)
            if self.hotwords:
                set_option("hotwords", ",".join(self.hotwords))
        np.clip(samples, -1.0, 1.0, out=samples)
        stream.accept_waveform(LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ, samples)
        recognizer.decode_stream(stream)
        result = getattr(stream, "result", None)
        text = getattr(result, "text", "")
        return str(text).strip()


@dataclass(slots=True)
class _LocalQwenSherpaSession(STTBackendSession):
    backend: LocalQwenSherpaSTTBackend
    _buffer_f32: list[np.ndarray] = field(init=False, repr=False)
    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False,
        repr=False,
    )
    _closed: bool = field(init=False, default=False, repr=False)
    _closed_event_enqueued: bool = field(init=False, default=False, repr=False)
    _utterances: int = field(init=False, default=0, repr=False)
    _total_audio_ms: float = field(init=False, default=0.0, repr=False)
    _total_inference_ms: float = field(init=False, default=0.0, repr=False)
    _total_rtf: float = field(init=False, default=0.0, repr=False)
    _summary_logged: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._buffer_f32 = []
        self._events = asyncio.Queue()

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._closed:
            return
        await self.send_audio_f32(pcm16le_bytes_to_float32(pcm16le))

    async def send_audio_f32(self, samples_f32: np.ndarray) -> None:
        if self._closed:
            return
        samples = np.asarray(samples_f32, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        self._buffer_f32.append(samples.copy())

    def drain_buffer_f32(self) -> np.ndarray | None:
        if not self._buffer_f32:
            return None
        snapshot = list(self._buffer_f32)
        self._buffer_f32.clear()
        return np.concatenate(snapshot)

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        audio_f32: np.ndarray | None = None,
    ) -> None:
        _ = trailing_silence_ms
        if self._closed:
            return

        if audio_f32 is None:
            audio_f32 = self.drain_buffer_f32()
        if audio_f32 is None or audio_f32.size == 0:
            return

        samples_f32 = audio_f32
        audio_ms = _sample_count_duration_ms(samples_f32.size, self.backend.sample_rate_hz)
        diag_enabled = self._diagnostics_enabled()
        if diag_enabled:
            self._log_decode_start_diagnostics(samples_f32)

        try:
            started_at = time.perf_counter()
            text = await self.backend.decode_f32(samples_f32)
            inference_ms = (time.perf_counter() - started_at) * 1000.0
        except Exception as exc:
            await self._events.put(exc)
            return

        rtf = inference_ms / audio_ms if audio_ms > 0 else 0.0
        self._utterances += 1
        self._total_audio_ms += audio_ms
        self._total_inference_ms += inference_ms
        self._total_rtf += rtf

        if diag_enabled:
            self._log_decode_done_diagnostics(
                audio_ms=audio_ms,
                inference_ms=inference_ms,
                rtf=rtf,
                text=text,
            )

        if text:
            logger.info(
                "%s Transcript final text_len=%s known_hallucination=%s audio_ms=%.1f inference_ms=%.1f rtf=%.3f",
                _log_prefix(self.backend.stream_label),
                len(text),
                is_known_local_qwen_hallucination(text),
                audio_ms,
                inference_ms,
                rtf,
            )
            await self._events.put(STTBackendTranscriptEvent(text=text, is_final=True))

    async def stop(self) -> None:
        self._log_summary_once()
        await self.close()

    async def close(self) -> None:
        self._log_summary_once()
        self._closed = True
        self._buffer_f32.clear()
        if self._closed_event_enqueued:
            return
        self._closed_event_enqueued = True
        await self._events.put(None)

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                break
            if isinstance(event, BaseException):
                raise event
            yield event

    def _diagnostics_enabled(self) -> bool:
        diagnostics_enabled = self.backend.diagnostics_enabled
        if diagnostics_enabled is None:
            return False
        with contextlib.suppress(Exception):
            return bool(diagnostics_enabled())
        return False

    def _log_decode_start_diagnostics(self, samples_f32: np.ndarray) -> None:
        with contextlib.suppress(Exception):
            metrics = compute_audio_frame_metrics(
                AudioFrameF32(
                    samples=samples_f32,
                    sample_rate_hz=self.backend.sample_rate_hz,
                    channels=1,
                )
            )
            logger.info(
                "%s decode_start audio_ms=%.1f rms_db=%.1f peak_db=%.1f zero_ratio=%.3f language_hint=%r",
                _audio_diag_prefix(self.backend.stream_label),
                metrics.audio_ms,
                metrics.rms_db,
                metrics.peak_db,
                metrics.zero_ratio,
                self.backend.language_hint,
            )

    def _log_decode_done_diagnostics(
        self,
        *,
        audio_ms: float,
        inference_ms: float,
        rtf: float,
        text: str,
    ) -> None:
        with contextlib.suppress(Exception):
            logger.info(
                "%s decode_done audio_ms=%.1f inference_ms=%.1f rtf=%.3f text_len=%s empty_result=%s suspicious_repetition=%s suspicious_script=%s",
                _audio_diag_prefix(self.backend.stream_label),
                audio_ms,
                inference_ms,
                rtf,
                len(text),
                not bool(text),
                _looks_repetitive(text),
                _looks_script_mismatched(text, self.backend.language_hint),
            )

    def _log_summary_once(self) -> None:
        if self._summary_logged or self._utterances == 0:
            return
        self._summary_logged = True
        weighted_total_rtf = (
            self._total_inference_ms / self._total_audio_ms if self._total_audio_ms > 0 else 0.0
        )
        mean_rtf = self._total_rtf / self._utterances if self._utterances > 0 else 0.0
        logger.info(
            "%s Session summary: utterances=%s total_audio_ms=%.1f total_inference_ms=%.1f weighted_total_rtf=%.3f mean_rtf=%.3f",
            _log_prefix(self.backend.stream_label),
            self._utterances,
            self._total_audio_ms,
            self._total_inference_ms,
            weighted_total_rtf,
            mean_rtf,
        )
