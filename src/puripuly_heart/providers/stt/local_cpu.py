from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from puripuly_heart.core.language import get_local_qwen_language_hint
from puripuly_heart.core.local_stt_assets import (
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    LocalSTTAssetManifest,
    default_local_stt_model_root,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_stt_catalog import (
    LocalCPUAutoUnavailableError,
    inspect_required_cpu_model_installs,
    resolve_cpu_auto_model,
)
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    STTBackend,
    STTBackendSession,
    STTSessionProjection,
)
from puripuly_heart.providers.stt.local_parakeet_sherpa import (
    LocalParakeetJapaneseSherpaSTTBackend,
    LocalParakeetV3SherpaSTTBackend,
)
from puripuly_heart.providers.stt.local_qwen_sherpa import (
    LOCAL_ASR_PENDING_TTL_S,
    LocalQwenSherpaSTTBackend,
)

LocalCPUBackendFactory = Callable[..., STTBackend]


def create_local_cpu_backend(
    model_id: str,
    *,
    model_root: Path | None = None,
    source_language: str,
    sample_rate_hz: int,
    stream_label: str | None,
    hotwords: tuple[str, ...] = (),
    pending_ttl_s: float = LOCAL_ASR_PENDING_TTL_S,
    decode_clock: Callable[[], float] = time.perf_counter,
    queue_clock: Callable[[], float] = time.monotonic,
    attempt_log_sink: Callable[[str, int], None] | None = None,
) -> STTBackend:
    if model_id not in {
        LOCAL_STT_MODEL_ID,
        PARAKEET_V3_MODEL_ID,
        PARAKEET_JAPANESE_MODEL_ID,
    }:
        raise ValueError(f"Unsupported local CPU model: {model_id}")
    manifest = load_local_stt_asset_manifest(model_id)
    resolved_root = model_root or default_local_stt_model_root()
    common = {
        "model_dir": resolved_root / manifest.install_dirname,
        "sample_rate_hz": sample_rate_hz,
        "stream_label": stream_label,
        "pending_ttl_s": pending_ttl_s,
        "decode_clock": decode_clock,
        "queue_clock": queue_clock,
    }
    if model_id == LOCAL_STT_MODEL_ID:
        return LocalQwenSherpaSTTBackend(
            **common,
            language_hint=get_local_qwen_language_hint(source_language),
            hotwords=hotwords,
            attempt_log_sink=attempt_log_sink,
        )
    if model_id == PARAKEET_V3_MODEL_ID:
        return LocalParakeetV3SherpaSTTBackend(**common)
    if model_id == PARAKEET_JAPANESE_MODEL_ID:
        return LocalParakeetJapaneseSherpaSTTBackend(**common)
    raise AssertionError(model_id)


@dataclass(slots=True)
class LocalCPUAutoSTTBackend(STTBackend):
    source_language: str
    sample_rate_hz: int = 16000
    stream_label: str | None = None
    model_root: Path | None = None
    hotwords: tuple[str, ...] = ()
    pending_ttl_s: float = LOCAL_ASR_PENDING_TTL_S
    decode_clock: Callable[[], float] = field(default_factory=lambda: time.perf_counter)
    queue_clock: Callable[[], float] = field(default_factory=lambda: time.monotonic)
    attempt_log_sink: Callable[[str, int], None] | None = field(default=None, repr=False)
    manifests: Mapping[str, LocalSTTAssetManifest] | None = None
    backend_factory: LocalCPUBackendFactory = create_local_cpu_backend
    _delegate: STTBackend | None = field(init=False, default=None, repr=False)
    _load_lock: asyncio.Lock = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _close_started: bool = field(init=False, default=False, repr=False)
    _close_complete: asyncio.Event = field(init=False, repr=False)
    _resolved_model_id: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 16000:
            raise ValueError("sample_rate_hz must be 16000")
        if self.pending_ttl_s <= 0:
            raise ValueError("pending_ttl_s must be > 0")
        self._load_lock = asyncio.Lock()
        self._close_complete = asyncio.Event()

    @property
    def resolved_model_id(self) -> str | None:
        return self._resolved_model_id

    @property
    def is_loaded(self) -> bool:
        return bool(self._delegate is not None and getattr(self._delegate, "is_loaded", False))

    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession:
        if self._closed:
            raise RuntimeError("CPU Auto backend is closed")
        async with self._load_lock:
            delegate = await self._ensure_delegate_locked()
            session = await delegate.open_session(projection=projection)
            if self._closed:
                await session.close()
                raise RuntimeError("CPU Auto backend is closed")
            return session

    async def reconfigure_session_options(self, options: LocalASRSessionOptions) -> None:
        async with self._load_lock:
            self.source_language = options.source_language
            delegate = self._delegate
            if delegate is None:
                return
            reconfigure = getattr(delegate, "reconfigure_session_options", None)
            if callable(reconfigure):
                result = reconfigure(options)
                if inspect.isawaitable(result):
                    await result

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
                        delegate = self._delegate
                        if delegate is not None:
                            await delegate.close()
                        self._delegate = None
                        self._resolved_model_id = None
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

    async def _ensure_delegate_locked(self) -> STTBackend:
        if self._closed:
            raise RuntimeError("CPU Auto backend is closed")
        if self._delegate is not None:
            return self._delegate
        snapshot = await run_owned_thread_call(
            partial(
                inspect_required_cpu_model_installs,
                self.model_root,
                manifests=self.manifests,
                verify_checksums=False,
            )
        )
        if self._closed:
            raise RuntimeError("CPU Auto backend is closed")
        if not snapshot.cpu_auto_available:
            raise LocalCPUAutoUnavailableError(snapshot)
        model_id = resolve_cpu_auto_model(self.source_language)
        delegate = self.backend_factory(
            model_id,
            model_root=self.model_root,
            source_language=self.source_language,
            sample_rate_hz=self.sample_rate_hz,
            stream_label=self.stream_label,
            hotwords=self.hotwords,
            pending_ttl_s=self.pending_ttl_s,
            decode_clock=self.decode_clock,
            queue_clock=self.queue_clock,
            attempt_log_sink=self.attempt_log_sink,
        )
        self._resolved_model_id = model_id
        self._delegate = delegate
        return delegate
