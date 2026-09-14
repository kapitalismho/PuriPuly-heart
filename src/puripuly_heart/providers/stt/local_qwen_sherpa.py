from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import AsyncIterator, Callable

import numpy as np

from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
from puripuly_heart.core.audio.format import (
    AudioCaptureSpan,
    AudioFrameF32,
    pcm16le_bytes_to_float32,
)
from puripuly_heart.core.audio.ownership import SegmentTerminalOutcome
from puripuly_heart.core.local_qwen_runtime import (
    LocalQwenRuntimeBootstrapError,
    ensure_local_qwen_windows_runtime,
)
from puripuly_heart.core.local_stt_assets import (
    LOCAL_STT_MODEL_ID,
    LocalQwenSherpaLoadError,
    load_local_stt_asset_manifest,
    validate_local_stt_runtime_ready,
)
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.local_qwen_hallucination import (
    is_known_local_qwen_hallucination,
)
from puripuly_heart.core.stt.session_projection import STTSessionEventProjection
from puripuly_heart.domain.models import FinalLanguageRun
from puripuly_heart.providers.stt.local_decode import (
    LocalDecodeBacklog,
    LocalDecodeCompletion,
    LocalDecodeCoordinator,
    LocalDecodeExpired,
    LocalDecodeFailure,
)

DEFAULT_SHERPA_NUM_THREADS = 3
LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ = 16000
LOCAL_ASR_PENDING_TTL_S = 12.0
_KNOWN_HALLUCINATION_LOG_REDACTION = "<known-local-qwen-hallucination>"
logger = logging.getLogger(__name__)


class LocalQwenSherpaInferenceError(RuntimeError):
    """Raised when local sherpa inference fails for an utterance."""


class _LocalQwenSherpaImportError(ImportError):
    """Internal sentinel for sherpa_onnx import failures."""


def _log_prefix(provider_id: str, stream_label: str | None) -> str:
    prefix = f"[STT][{provider_id}]"
    if stream_label:
        return f"{prefix}[{stream_label}]"
    return prefix


def _audio_diag_prefix(provider_id: str, stream_label: str | None) -> str:
    prefix = f"[AudioDiag][{provider_id}]"
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
    model_id: str = field(default=LOCAL_STT_MODEL_ID, init=False)
    active_decode_timeout_s: float = 30.0
    provider_id: str = field(default="local_qwen", init=False)
    pending_ttl_s: float = LOCAL_ASR_PENDING_TTL_S
    decode_clock: Callable[[], float] = field(default_factory=lambda: time.perf_counter)
    queue_clock: Callable[[], float] = field(default_factory=lambda: time.monotonic)
    _recognizer: object | None = field(init=False, default=None, repr=False)
    _load_lock: asyncio.Lock = field(init=False, repr=False)
    _decode_lock: asyncio.Lock = field(init=False, repr=False)
    _session_handoff_tail: asyncio.Event | None = field(init=False, default=None, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _close_started: bool = field(init=False, default=False, repr=False)
    _close_complete: asyncio.Event = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sample_rate_hz != LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ:
            raise ValueError(f"sample_rate_hz must be {LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ}")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be > 0")
        if self.active_decode_timeout_s <= 0:
            raise ValueError("active_decode_timeout_s must be > 0")
        if self.pending_ttl_s <= 0:
            raise ValueError("pending_ttl_s must be > 0")
        self._load_lock = asyncio.Lock()
        self._decode_lock = asyncio.Lock()
        self._close_complete = asyncio.Event()

    @property
    def is_loaded(self) -> bool:
        return self._recognizer is not None

    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession:
        await self._ensure_recognizer()
        if self._closed:
            raise RuntimeError("Local STT backend is closed")
        session = _LocalQwenSherpaSession(
            backend=self,
            decode_start_after=self._session_handoff_tail,
            projection=projection,
        )
        self._session_handoff_tail = session.handoff_complete_event
        return session

    async def reconfigure_session_options(self, options: LocalASRSessionOptions) -> None:
        self.language_hint = options.language_hint

    async def close(self) -> None:
        if self._close_started:
            await asyncio.shield(self._close_complete.wait())
            return
        self._close_started = True
        self._closed = True
        cleanup_cancelled = False
        try:
            while True:
                try:
                    async with self._load_lock:
                        self._recognizer = None
                        self._session_handoff_tail = None
                    async with self._decode_lock:
                        pass
                    break
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is None or not current_task.cancelling():
                        raise
                    cleanup_cancelled = True
        finally:
            self._close_complete.set()
        if cleanup_cancelled:
            raise asyncio.CancelledError

    async def _ensure_recognizer(self) -> object:
        if self._closed:
            raise RuntimeError("Local STT backend is closed")
        if self._recognizer is not None:
            return self._recognizer

        async with self._load_lock:
            if self._closed:
                raise RuntimeError("Local STT backend is closed")
            if self._recognizer is not None:
                return self._recognizer
            await run_owned_thread_call(self._validate_runtime_assets)
            if self._closed:
                raise RuntimeError("Local STT backend is closed")
            recognizer = await run_owned_thread_call(self._create_recognizer)
            if self._closed:
                raise RuntimeError("Local STT backend is closed")
            self._recognizer = recognizer
            return recognizer

    def _validate_runtime_assets(self) -> None:
        if self.model_id == LOCAL_STT_MODEL_ID:
            validate_local_stt_runtime_ready(self.model_dir)
            return
        validate_local_stt_runtime_ready(
            self.model_dir,
            manifest=load_local_stt_asset_manifest(self.model_id),
        )

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
            if self._closed:
                raise RuntimeError("Local STT backend is closed")
            try:
                return await run_owned_thread_call(
                    partial(
                        self._decode_f32_sync,
                        recognizer,
                        samples_f32,
                    )
                )
            except Exception as exc:
                raise self._inference_error(exc) from exc

    def _inference_error(self, exc: Exception) -> RuntimeError:
        return LocalQwenSherpaInferenceError(str(exc))

    def is_known_hallucination(self, text: str) -> bool:
        return is_known_local_qwen_hallucination(text)

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
    decode_start_after: asyncio.Event | None = field(default=None, repr=False)
    projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION
    _buffer_f32: list[np.ndarray] = field(init=False, repr=False)
    _event_projection: STTSessionEventProjection = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _stopping: bool = field(init=False, default=False, repr=False)
    _closed_event_enqueued: bool = field(init=False, default=False, repr=False)
    _utterances: int = field(init=False, default=0, repr=False)
    _total_audio_ms: float = field(init=False, default=0.0, repr=False)
    _total_inference_ms: float = field(init=False, default=0.0, repr=False)
    _total_rtf: float = field(init=False, default=0.0, repr=False)
    _summary_logged: bool = field(init=False, default=False, repr=False)
    _decode_coordinator: LocalDecodeCoordinator = field(init=False, repr=False)
    _events_started: bool = field(init=False, default=False, repr=False)
    _failure_handoff_safe: bool = field(init=False, default=False, repr=False)
    _handoff_complete: asyncio.Event = field(init=False, repr=False)
    _close_complete: asyncio.Event = field(init=False, repr=False)
    _scoped_job_identities: dict[int, STTProviderTurnIdentity] = field(
        init=False, default_factory=dict, repr=False
    )
    _scoped_watchdogs: dict[int, asyncio.Task[None]] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self._buffer_f32 = []
        self._event_projection = STTSessionEventProjection(self.projection)
        self._handoff_complete = asyncio.Event()
        self._close_complete = asyncio.Event()
        self._decode_coordinator = LocalDecodeCoordinator(
            owner_name=f"{self.backend.provider_id}-session",
            sample_rate_hz=self.backend.sample_rate_hz,
            decode=self._decode_samples,
            on_completion=self._handle_decode_completion,
            on_failure=self._handle_decode_failure,
            on_expired=self._handle_decode_expired,
            on_backlog_warning=self._log_decode_backlog_warning,
            start_after=self.decode_start_after,
            pending_ttl_s=self.backend.pending_ttl_s,
            clock=self.backend.decode_clock,
            queue_clock=self.backend.queue_clock,
        )

    @property
    def handoff_complete_event(self) -> asyncio.Event:
        return self._handoff_complete

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._closed or self._stopping or not self._decode_coordinator.accepting:
            return
        await self.send_audio_f32(pcm16le_bytes_to_float32(pcm16le))

    async def send_audio_f32(self, samples_f32: np.ndarray) -> None:
        if self._closed or self._stopping or not self._decode_coordinator.accepting:
            return
        samples = np.asarray(samples_f32, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        self._buffer_f32.append(samples.copy())

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self._closed or self._stopping or not self._decode_coordinator.accepting:
            raise RuntimeError("local STT session is unavailable")
        self._event_projection.begin(request)
        self._buffer_f32.clear()

    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None:
        _ = source_ranges, context_only
        self._event_projection.validate_payload(identity, payload_sequence)
        await self.send_audio(pcm16le)
        self._event_projection.payload_written(identity, payload_sequence)

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        _ = sealed_content_ranges, seal_reason, observed_trailing_silence_ms
        self._event_projection.seal(identity)
        samples_f32 = (
            np.concatenate(self._buffer_f32)
            if self._buffer_f32
            else np.empty((0,), dtype=np.float32)
        )
        self._buffer_f32.clear()
        job = self._decode_coordinator.enqueue_job(samples_f32, copy_samples=False)
        if job is None:
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason="local_decode_unavailable",
                retire=True,
            )
            return
        self._scoped_job_identities[job.sequence] = identity
        watchdog = asyncio.create_task(
            self._watch_scoped_decode(job.sequence, identity),
            name=f"{self.backend.provider_id}-scoped-decode-timeout",
        )
        self._scoped_watchdogs[job.sequence] = watchdog

    async def _watch_scoped_decode(
        self,
        sequence: int,
        identity: STTProviderTurnIdentity,
    ) -> None:
        try:
            await asyncio.sleep(self.backend.active_decode_timeout_s)
        except asyncio.CancelledError:
            return
        self._scoped_watchdogs.pop(sequence, None)
        owner = self._scoped_job_identities.pop(sequence, None)
        if owner == identity:
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason="local_decode_timeout",
                retire=True,
            )

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        self._event_projection.require_open(identity)
        self._buffer_f32.clear()
        self._scoped_job_identities = {
            sequence: owner
            for sequence, owner in self._scoped_job_identities.items()
            if owner != identity
        }
        self._terminalize_scoped(
            identity,
            outcome="cancelled",
            failure_reason=reason,
        )

    async def turn_events(self):
        async for event in self._event_projection.turn_events():
            yield event

    def _terminalize_scoped(
        self,
        identity: STTProviderTurnIdentity,
        *,
        outcome: SegmentTerminalOutcome,
        text: str = "",
        final_language_runs: tuple[FinalLanguageRun, ...] = (),
        failure_reason: str | None = None,
        retire: bool = False,
    ) -> None:
        if not self._event_projection.is_current(identity):
            return
        authority = "authoritative" if outcome in ("final", "empty") else "none"
        if outcome == "degraded":
            authority = "degraded"
        should_retire = retire or outcome in ("failed", "cancelled")
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome=outcome,
                text=text,
                final_language_runs=final_language_runs,
                text_authority=authority,
                failure_reason=failure_reason,
                epoch_disposition="retire" if should_retire else "reuse",
            )
        )

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = (trailing_silence_ms, reason)
        if self._closed or self._stopping or not self._decode_coordinator.accepting:
            return

        samples_f32 = (
            np.concatenate(self._buffer_f32)
            if self._buffer_f32
            else np.empty((0,), dtype=np.float32)
        )
        self._buffer_f32.clear()
        self._decode_coordinator.enqueue(samples_f32)

    async def _decode_samples(self, samples_f32: np.ndarray) -> str:
        if self._diagnostics_enabled():
            self._log_decode_start_diagnostics(samples_f32)
        return await self.backend.decode_f32(samples_f32)

    async def _handle_decode_completion(self, completion: LocalDecodeCompletion) -> None:
        text = completion.text
        audio_ms = completion.job.audio_ms
        inference_ms = completion.inference_ms
        rtf = inference_ms / audio_ms if audio_ms > 0 else 0.0
        if audio_ms > 0:
            self._utterances += 1
            self._total_audio_ms += audio_ms
            self._total_inference_ms += inference_ms
            self._total_rtf += rtf

        if audio_ms > 0 and self._diagnostics_enabled():
            self._log_decode_done_diagnostics(
                audio_ms=audio_ms,
                inference_ms=inference_ms,
                rtf=rtf,
                text=text,
            )

        if text:
            logger.info(
                "%s Transcript final text_len=%s known_hallucination=%s audio_ms=%.1f inference_ms=%.1f rtf=%.3f",
                _log_prefix(self.backend.provider_id, self.backend.stream_label),
                len(text),
                self.backend.is_known_hallucination(text),
                audio_ms,
                inference_ms,
                rtf,
            )
        if audio_ms > 0 and self._diagnostics_enabled():
            self._log_attempt_diagnostic(
                audio_ms=audio_ms,
                inference_ms=inference_ms,
                queue_wait_ms=completion.queue_wait_ms,
                result="success",
            )
        identity = self._scoped_job_identities.pop(completion.job.sequence, None)
        if identity is None:
            self._event_projection.put_legacy(STTBackendTranscriptEvent(text=text, is_final=True))
        watchdog = self._scoped_watchdogs.pop(completion.job.sequence, None)
        if watchdog is not None:
            watchdog.cancel()
        if identity is not None:
            if text and self.backend.is_known_hallucination(text):
                self._terminalize_scoped(identity, outcome="suppressed")
            elif text:
                self._terminalize_scoped(identity, outcome="final", text=text)
            else:
                self._terminalize_scoped(identity, outcome="empty")

    async def _handle_decode_failure(self, failure: LocalDecodeFailure) -> None:
        if failure.job.audio_ms > 0 and self._diagnostics_enabled():
            self._log_attempt_diagnostic(
                audio_ms=failure.job.audio_ms,
                inference_ms=failure.inference_ms,
                queue_wait_ms=failure.queue_wait_ms,
                result="failure",
            )
        retired_jobs = (failure.job, *failure.discarded_jobs)
        legacy_failure = False
        for job in retired_jobs:
            identity = self._scoped_job_identities.pop(job.sequence, None)
            watchdog = self._scoped_watchdogs.pop(job.sequence, None)
            if watchdog is not None:
                watchdog.cancel()
            if identity is None:
                legacy_failure = True
                self._event_projection.put_legacy(STTBackendTranscriptEvent(text="", is_final=True))
            else:
                self._terminalize_scoped(
                    identity,
                    outcome="failed",
                    failure_reason=type(failure.error).__name__,
                    retire=True,
                )
        if legacy_failure:
            self._failure_handoff_safe = True
            self._event_projection.put_legacy(failure.error)

    async def _handle_decode_expired(self, expired: LocalDecodeExpired) -> None:
        if self._diagnostics_enabled():
            logger.info(
                "[LocalASR][Expiry] channel=%s model=%s intended_provider=%s reason=%s queue_wait_seconds=%.3f",
                self.backend.stream_label or "unknown",
                self.backend.model_id,
                self.backend.provider_id,
                expired.reason,
                expired.queue_wait_ms / 1000.0,
            )
        identity = self._scoped_job_identities.pop(expired.job.sequence, None)
        if identity is None:
            self._event_projection.put_legacy(STTBackendTranscriptEvent(text="", is_final=True))
        watchdog = self._scoped_watchdogs.pop(expired.job.sequence, None)
        if watchdog is not None:
            watchdog.cancel()
        if identity is not None:
            self._terminalize_scoped(
                identity,
                outcome="expired",
                failure_reason=expired.reason,
            )

    def _log_decode_backlog_warning(self, backlog: LocalDecodeBacklog) -> None:
        logger.warning(
            "%s Decode backlog is unexpectedly high: pending_jobs=%s buffered_audio_ms=%.1f threshold=%s",
            _log_prefix(self.backend.provider_id, self.backend.stream_label),
            backlog.pending_jobs,
            backlog.buffered_audio_ms,
            backlog.warning_threshold,
        )

    async def stop(self) -> None:
        self._stopping = True
        await self._decode_coordinator.stop()
        self._log_summary_once()
        await self.close()

    async def abort_for_toggle_off(self) -> None:
        self._stopping = True
        self._buffer_f32.clear()
        await self.close()

    async def close(self) -> None:
        if self._closed:
            await asyncio.shield(self._close_complete.wait())
            return
        self._closed = True
        self._buffer_f32.clear()
        identity = self._event_projection.active_identity
        if identity is not None:
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason="session_closed",
                retire=True,
            )
        try:
            if self._decode_coordinator.pending_jobs:
                logger.info(
                    "%s Decode cancellation requested: pending_jobs=%s buffered_audio_ms=%.1f",
                    _log_prefix(self.backend.provider_id, self.backend.stream_label),
                    self._decode_coordinator.pending_jobs,
                    self._decode_coordinator.buffered_audio_ms,
                )
            await self._decode_coordinator.close()
        finally:
            self._log_summary_once()
            if not self._closed_event_enqueued:
                self._closed_event_enqueued = True
                self._event_projection.close()
            if not self._events_started:
                self._handoff_complete.set()
            self._close_complete.set()

            for watchdog in self._scoped_watchdogs.values():
                watchdog.cancel()
            self._scoped_watchdogs.clear()

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        self._events_started = True
        try:
            async for event in self._event_projection.events():
                yield event
        except BaseException:
            if self._failure_handoff_safe:
                self._handoff_complete.set()
            raise
        else:
            self._handoff_complete.set()

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
                _audio_diag_prefix(self.backend.provider_id, self.backend.stream_label),
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
                _audio_diag_prefix(self.backend.provider_id, self.backend.stream_label),
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
            _log_prefix(self.backend.provider_id, self.backend.stream_label),
            self._utterances,
            self._total_audio_ms,
            self._total_inference_ms,
            weighted_total_rtf,
            mean_rtf,
        )

    def _log_attempt_diagnostic(
        self,
        *,
        audio_ms: float,
        inference_ms: float,
        queue_wait_ms: float,
        result: str,
    ) -> None:
        rtf = inference_ms / audio_ms if audio_ms > 0 else 0.0
        logger.info(
            "[LocalASR][Attempt] channel=%s model=%s backend=CPU audio_seconds=%.3f decode_seconds=%.3f rtf=%.6f result=%s queue_wait_seconds=%.3f",
            self.backend.stream_label or "unknown",
            self.backend.model_id,
            audio_ms / 1000.0,
            inference_ms / 1000.0,
            rtf,
            result,
            queue_wait_ms / 1000.0,
        )
