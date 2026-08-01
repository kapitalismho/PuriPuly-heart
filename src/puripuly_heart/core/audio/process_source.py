from __future__ import annotations

import contextlib
import importlib
import queue
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Protocol

import janus
import numpy as np

from puripuly_heart.config.process_capture_platform import (
    ProcessCapturePlatformAvailability,
    get_process_capture_platform_availability,
)
from puripuly_heart.config.process_capture_resolution import ResolvedProcessCaptureIdentity
from puripuly_heart.core.audio.format import AudioFrameF32

PROCESS_CAPTURE_SAMPLE_RATE_HZ = 48000
PROCESS_CAPTURE_CHANNELS = 2


class ProcessAudioCaptureSetupError(RuntimeError):
    pass


class ProcessAudioCaptureUnavailableError(RuntimeError):
    pass


class ProcessAudioCapturePort(Protocol):
    def start(self) -> None: ...
    def close(self) -> None: ...


class ProcessAudioCaptureFactory(Protocol):
    def create(
        self,
        *,
        pid: int,
        on_data: Callable[[bytes, int], None],
    ) -> ProcessAudioCapturePort: ...


class ProcessIdentityWatchPort(Protocol):
    @property
    def identity_verified(self) -> bool: ...

    def close(self) -> None: ...


class ProcessIdentityWatcher(Protocol):
    def watch(
        self,
        identity: ResolvedProcessCaptureIdentity,
        on_terminal: Callable[[], None],
    ) -> ProcessIdentityWatchPort: ...


@dataclass(frozen=True, slots=True)
class ProcTapProcessAudioCaptureFactory:
    platform_availability: Callable[[], ProcessCapturePlatformAvailability] = (
        get_process_capture_platform_availability
    )

    def create(
        self,
        *,
        pid: int,
        on_data: Callable[[bytes, int], None],
    ) -> ProcessAudioCapturePort:
        if not self.platform_availability().available:
            raise ProcessAudioCaptureUnavailableError("process capture platform is unavailable")
        module = importlib.import_module("proctap")
        capture_type = getattr(module, "ProcessAudioCapture")
        capture = capture_type(pid, on_data=on_data)
        try:
            verify_proctap_process_specific(capture)
        except Exception as exc:
            with contextlib.suppress(Exception):
                capture.close()
            if isinstance(exc, ProcessAudioCaptureSetupError):
                raise
            raise ProcessAudioCaptureSetupError(
                "process capture mode could not be verified"
            ) from None
        return capture


def verify_proctap_process_specific(capture: object) -> bool:
    backend = getattr(capture, "_backend", None)
    native = getattr(backend, "_native", None)
    verifier = getattr(native, "is_process_specific", None)
    if not callable(verifier):
        raise ProcessAudioCaptureSetupError("process capture mode could not be verified")
    try:
        verified = verifier()
    except Exception:
        raise ProcessAudioCaptureSetupError("process capture mode could not be verified") from None
    if verified is not True:
        raise ProcessAudioCaptureSetupError("process capture mode could not be verified")
    return True


@dataclass(slots=True)
class ProcessAudioCaptureSource:
    identity: ResolvedProcessCaptureIdentity
    watcher: ProcessIdentityWatcher
    max_queue_frames: int = 64
    capture_factory: ProcessAudioCaptureFactory = field(
        default_factory=ProcTapProcessAudioCaptureFactory
    )
    platform_availability: Callable[[], ProcessCapturePlatformAvailability] = (
        get_process_capture_platform_availability
    )

    _queue: janus.Queue[np.ndarray | None] = field(init=False, repr=False)
    _capture: ProcessAudioCapturePort | None = field(init=False, default=None, repr=False)
    _watch: ProcessIdentityWatchPort | None = field(init=False, default=None, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _terminal_reason: str | None = field(init=False, default=None, repr=False)
    _queue_drop_count: int = field(init=False, default=0, repr=False)
    _lock: threading.RLock = field(init=False, default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if self.max_queue_frames <= 0:
            raise ValueError("max_queue_frames must be > 0")
        self._queue = janus.Queue(maxsize=self.max_queue_frames)
        if not self.platform_availability().available:
            self._queue.close()
            raise ProcessAudioCaptureUnavailableError("process capture platform is unavailable")
        try:
            capture = self.capture_factory.create(pid=self.identity.pid, on_data=self._on_data)
            self._capture = capture
            watch = self.watcher.watch(self.identity, self._on_process_terminal)
            with self._lock:
                if self._terminal_reason is None and watch.identity_verified:
                    self._watch = watch
                    capture.start()
                    if self._terminal_reason is not None:
                        self._release_native_resources()
                else:
                    with contextlib.suppress(Exception):
                        watch.close()
                    if self._terminal_reason is None:
                        self._terminal_reason = "target_identity_mismatch"
                        self._signal_terminal()
                        raise ProcessAudioCaptureSetupError(
                            "resolved process identity could not be verified"
                        )
                    self._release_native_resources()
        except Exception as exc:
            self._release_native_resources()
            self._queue.close()
            if isinstance(exc, ProcessAudioCaptureUnavailableError):
                raise
            raise ProcessAudioCaptureSetupError("process audio capture setup failed") from exc

    @property
    def terminal_reason(self) -> str | None:
        return self._terminal_reason

    @property
    def queue_drop_count(self) -> int:
        return self._queue_drop_count

    async def frames(self) -> AsyncIterator[AudioFrameF32]:
        while True:
            samples = await self._queue.async_q.get()
            if samples is None:
                await self.close()
                return
            yield AudioFrameF32(
                samples=samples,
                sample_rate_hz=PROCESS_CAPTURE_SAMPLE_RATE_HZ,
                channels=PROCESS_CAPTURE_CHANNELS,
            )

    async def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._terminal_reason is None:
                self._terminal_reason = "closed"
            self._release_native_resources()
            self._signal_terminal()
        self._queue.close()
        with contextlib.suppress(Exception):
            await self._queue.wait_closed()

    def _on_data(self, data: bytes, frames: int) -> None:
        if self._closed or self._terminal_reason is not None:
            return
        samples = _decode_process_capture_frame(data, frames)
        if samples is None:
            self._signal_terminal_failure("source_failure")
            return
        try:
            self._queue.sync_q.put_nowait(samples)
        except queue.Full:
            self._queue_drop_count += 1
        except Exception:
            self._signal_terminal_failure("source_failure")

    def _on_process_terminal(self) -> None:
        self._signal_terminal_failure("target_exited")

    def _signal_terminal_failure(self, reason: str) -> None:
        with self._lock:
            if self._terminal_reason is not None:
                return
            self._terminal_reason = reason
            self._signal_terminal()

    def _release_native_resources(self) -> None:
        watch, self._watch = self._watch, None
        capture, self._capture = self._capture, None
        if watch is not None:
            with contextlib.suppress(Exception):
                watch.close()
        if capture is not None:
            with contextlib.suppress(Exception):
                capture.close()

    def _signal_terminal(self) -> None:
        try:
            self._queue.sync_q.put_nowait(None)
            return
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._queue.sync_q.get_nowait()
            with contextlib.suppress(Exception):
                self._queue.sync_q.put_nowait(None)
        except Exception:
            return


def _decode_process_capture_frame(data: bytes, frames: int) -> np.ndarray | None:
    if isinstance(frames, bool) or not isinstance(frames, int) or frames == 0 or frames < -1:
        return None
    if not isinstance(data, bytes) or len(data) % (PROCESS_CAPTURE_CHANNELS * 4) != 0:
        return None
    derived_frames = len(data) // (PROCESS_CAPTURE_CHANNELS * 4)
    if derived_frames <= 0 or (frames != -1 and frames != derived_frames):
        return None
    return np.frombuffer(data, dtype="<f4").reshape((derived_frames, PROCESS_CAPTURE_CHANNELS))
