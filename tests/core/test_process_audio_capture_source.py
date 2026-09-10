from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from puripuly_heart.config.process_capture_platform import ProcessCapturePlatformAvailability
from puripuly_heart.config.settings_vnext.schema import ProcessCaptureTargetIntent
from puripuly_heart.core.audio.process_identity import PsutilProcessIdentityWatcher
from puripuly_heart.core.audio.process_source import (
    PROCESS_CAPTURE_CHANNELS,
    PROCESS_CAPTURE_SAMPLE_RATE_HZ,
    ProcessAudioCaptureSetupError,
    ProcessAudioCaptureSource,
    ProcessAudioCaptureUnavailableError,
    ProcTapProcessAudioCaptureFactory,
    ResolvedProcessCaptureIdentity,
)


@dataclass
class FakeCapture:
    on_data: object
    fail_start: bool = False
    started: bool = field(init=False, default=False)
    closed: bool = field(init=False, default=False)
    close_thread_ids: list[int] = field(init=False, default_factory=list)

    def start(self) -> None:
        self.started = True
        if self.fail_start:
            raise RuntimeError("start failed")

    def close(self) -> None:
        self.closed = True
        self.close_thread_ids.append(threading.get_ident())


@dataclass
class FakeFactory:
    fail_create: bool = False
    fail_start: bool = False
    captures: list[tuple[int, FakeCapture]] = field(init=False, default_factory=list)

    def create(self, *, pid: int, on_data):
        if self.fail_create:
            raise RuntimeError("create failed")
        capture = FakeCapture(on_data=on_data, fail_start=self.fail_start)
        self.captures.append((pid, capture))
        return capture


@dataclass
class FakeWatch:
    identity_verified: bool = True
    closed: bool = field(init=False, default=False)

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeWatcher:
    fail_attach: bool = False
    terminal_during_watch: bool = False
    identity_verified: bool = True
    identities: list[ResolvedProcessCaptureIdentity] = field(init=False, default_factory=list)
    callbacks: list[object] = field(init=False, default_factory=list)
    watches: list[FakeWatch] = field(init=False, default_factory=list)

    def watch(self, identity: ResolvedProcessCaptureIdentity, on_terminal):
        if self.fail_attach:
            raise RuntimeError("watch failed")
        self.identities.append(identity)
        self.callbacks.append(on_terminal)
        watch = FakeWatch(identity_verified=self.identity_verified)
        self.watches.append(watch)
        if self.terminal_during_watch:
            on_terminal()
        return watch

    def emit_terminal(self) -> None:
        self.callbacks[-1]()


def _supported() -> ProcessCapturePlatformAvailability:
    return ProcessCapturePlatformAvailability(available=True)


def _identity(pid: int = 100) -> ResolvedProcessCaptureIdentity:
    return ResolvedProcessCaptureIdentity(
        pid=pid,
        target=ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Game\Game.exe"),
        instance_id=f"instance-{pid}",
    )


def _frame_bytes(samples: np.ndarray) -> bytes:
    return np.asarray(samples, dtype="<f4").tobytes()


@pytest.mark.asyncio
async def test_process_source_bridges_fixed_stereo_float32_frames_and_drops_without_blocking() -> (
    None
):
    factory = FakeFactory()
    watcher = FakeWatcher()
    source = ProcessAudioCaptureSource(
        identity=_identity(),
        watcher=watcher,
        capture_factory=factory,
        platform_availability=_supported,
        max_queue_frames=1,
    )
    capture = factory.captures[0][1]
    first = np.array([[0.1, -0.1]], dtype=np.float32)
    second = np.array([[0.2, -0.2]], dtype=np.float32)

    capture.on_data(_frame_bytes(first), -1)
    capture.on_data(_frame_bytes(second), 1)
    frame = await source.frames().__anext__()

    assert factory.captures[0][0] == 100
    assert capture.started is True
    np.testing.assert_array_equal(frame.samples, first)
    assert frame.sample_rate_hz == PROCESS_CAPTURE_SAMPLE_RATE_HZ
    assert frame.channels == PROCESS_CAPTURE_CHANNELS
    assert frame.capture is not None
    assert frame.capture.source_start_sample == 0
    assert frame.capture.source_end_sample == 1
    capture.on_data(_frame_bytes(second), 1)
    successor = await source.frames().__anext__()
    assert successor.capture is not None
    assert successor.capture.source_start_sample == 2
    assert successor.capture.source_end_sample == 3
    assert successor.capture.discontinuity_before is not None
    assert successor.capture.discontinuity_before.kind == "known_loss"
    assert successor.capture.discontinuity_before.lost_source_samples == 1
    assert source.queue_drop_count == 1

    await source.close()

@pytest.mark.asyncio
async def test_process_terminal_preserves_admitted_pcm_before_end_of_stream() -> None:
    factory = FakeFactory()
    watcher = FakeWatcher()
    source = ProcessAudioCaptureSource(
        identity=_identity(),
        watcher=watcher,
        capture_factory=factory,
        platform_availability=_supported,
        max_queue_frames=1,
    )
    capture = factory.captures[0][1]
    samples = np.array([[0.1, -0.1]], dtype=np.float32)

    capture.on_data(_frame_bytes(samples), 1)
    watcher.emit_terminal()

    frame = await source.frames().__anext__()
    np.testing.assert_array_equal(frame.samples, samples)
    assert frame.capture is not None
    assert frame.capture.source_start_sample == 0
    assert frame.capture.source_end_sample == 1
    assert source.capture_progression_snapshot.admitted_source_end_sample == 1
    assert source.queue_drop_count == 0
    with pytest.raises(StopAsyncIteration):
        await source.frames().__anext__()


@pytest.mark.asyncio
async def test_process_source_close_releases_capture_and_identity_watch() -> None:
    factory = FakeFactory()
    watcher = FakeWatcher()
    source = ProcessAudioCaptureSource(
        identity=_identity(),
        watcher=watcher,
        capture_factory=factory,
        platform_availability=_supported,
    )

    await source.close()
    await source.close()

    assert factory.captures[0][1].close_thread_ids == [threading.get_ident()]
    assert watcher.watches[0].closed is True
    assert source.terminal_reason == "closed"


def test_process_source_closes_partial_setup_capture_when_start_fails() -> None:
    factory = FakeFactory(fail_start=True)
    watcher = FakeWatcher()

    with pytest.raises(ProcessAudioCaptureSetupError):
        ProcessAudioCaptureSource(
            identity=_identity(),
            watcher=watcher,
            capture_factory=factory,
            platform_availability=_supported,
        )

    assert factory.captures[0][1].closed is True
    assert watcher.watches[0].closed is True


@pytest.mark.parametrize(
    ("factory", "watcher", "captures_created", "watches_created"),
    [
        (FakeFactory(fail_create=True), FakeWatcher(), 0, 0),
        (FakeFactory(), FakeWatcher(fail_attach=True), 1, 0),
    ],
)
def test_process_source_setup_failures_release_created_native_resources(
    factory: FakeFactory,
    watcher: FakeWatcher,
    captures_created: int,
    watches_created: int,
) -> None:
    with pytest.raises(ProcessAudioCaptureSetupError):
        ProcessAudioCaptureSource(
            identity=_identity(),
            watcher=watcher,
            capture_factory=factory,
            platform_availability=_supported,
        )

    assert len(factory.captures) == captures_created
    assert len(watcher.watches) == watches_created
    if factory.captures:
        assert factory.captures[0][1].closed is True


@pytest.mark.asyncio
async def test_process_source_terminal_exit_closes_original_identity_after_pid_reuse_without_replacement_attachment() -> (
    None
):
    factory = FakeFactory()
    watcher = FakeWatcher()
    identity = _identity(pid=101)
    source = ProcessAudioCaptureSource(
        identity=identity,
        watcher=watcher,
        capture_factory=factory,
        platform_availability=_supported,
    )

    watcher.emit_terminal()

    reused_pid_identity = ResolvedProcessCaptureIdentity(
        pid=101,
        target=ProcessCaptureTargetIntent.vrchat(r"C:\VRChat\VRChat.exe"),
        instance_id="reused-instance",
    )
    assert watcher.identities == [identity]
    assert watcher.identities != [reused_pid_identity]
    assert source.terminal_reason == "target_exited"
    with pytest.raises(StopAsyncIteration):
        await source.frames().__anext__()
    assert factory.captures[0][1].closed is True
    assert watcher.watches[0].closed is True

    await source.close()


@pytest.mark.asyncio
async def test_process_source_terminal_frame_failure_defers_native_close_from_callback_thread() -> (
    None
):
    factory = FakeFactory()
    watcher = FakeWatcher()
    source = ProcessAudioCaptureSource(
        identity=_identity(),
        watcher=watcher,
        capture_factory=factory,
        platform_availability=_supported,
    )

    callback_thread_ids: list[int] = []

    def invoke_callback() -> None:
        callback_thread_ids.append(threading.get_ident())
        factory.captures[0][1].on_data(b"\x00", 1)

    callback_thread = threading.Thread(target=invoke_callback)
    callback_thread.start()
    callback_thread.join()

    assert source.terminal_reason == "source_failure"
    assert factory.captures[0][1].closed is False
    assert watcher.watches[0].closed is False

    await source.close()

    assert factory.captures[0][1].closed is True
    assert watcher.watches[0].closed is True
    assert factory.captures[0][1].close_thread_ids != callback_thread_ids


def test_unsupported_platform_never_constructs_or_imports_proctap(monkeypatch) -> None:
    factory = FakeFactory()
    watcher = FakeWatcher()
    monkeypatch.delitem(__import__("sys").modules, "proctap", raising=False)

    with pytest.raises(ProcessAudioCaptureUnavailableError):
        ProcessAudioCaptureSource(
            identity=_identity(),
            watcher=watcher,
            capture_factory=factory,
            platform_availability=lambda: ProcessCapturePlatformAvailability(
                available=False,
                reason="unsupported_windows_build",
            ),
        )

    assert factory.captures == []
    assert "proctap" not in __import__("sys").modules


def test_synchronous_terminal_during_watch_attach_closes_returned_watch() -> None:
    factory = FakeFactory()
    watcher = FakeWatcher(terminal_during_watch=True)

    source = ProcessAudioCaptureSource(
        identity=_identity(),
        watcher=watcher,
        capture_factory=factory,
        platform_availability=_supported,
    )

    assert source.terminal_reason == "target_exited"
    assert factory.captures[0][1].closed is True
    assert watcher.watches[0].closed is True


def test_identity_mismatch_during_watch_attach_fails_closed_before_capture_start() -> None:
    factory = FakeFactory()
    watcher = FakeWatcher(identity_verified=False)

    with pytest.raises(ProcessAudioCaptureSetupError):
        ProcessAudioCaptureSource(
            identity=_identity(),
            watcher=watcher,
            capture_factory=factory,
            platform_availability=_supported,
        )

    assert factory.captures[0][1].started is False
    assert factory.captures[0][1].closed is True
    assert watcher.watches[0].closed is True


def test_proctap_factory_lazily_uses_public_capture_constructor(monkeypatch) -> None:
    created: dict[str, object] = {}
    imported: list[str] = []

    class PublicCapture:
        def __init__(self, pid: int, on_data) -> None:
            created["pid"] = pid
            created["on_data"] = on_data
            self._backend = SimpleNamespace(
                _native=SimpleNamespace(is_process_specific=lambda: True)
            )

        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    import puripuly_heart.core.audio.process_source as source_module

    def import_module(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(ProcessAudioCapture=PublicCapture)

    monkeypatch.setattr(
        source_module.importlib,
        "import_module",
        import_module,
    )

    def callback(_data: bytes, _frames: int) -> None:
        return None

    capture = ProcTapProcessAudioCaptureFactory(platform_availability=_supported).create(
        pid=102,
        on_data=callback,
    )

    assert isinstance(capture, PublicCapture)
    assert imported == ["proctap"]
    assert created == {"pid": 102, "on_data": callback}
    assert "DesktopLoopback" not in source_module.__dict__


@pytest.mark.parametrize("native_state", [False, "missing", "throws"])
def test_proctap_factory_rejects_unverified_native_mode_and_closes(
    monkeypatch, native_state: object
) -> None:
    captures = []

    class PublicCapture:
        def __init__(self, _pid: int, on_data) -> None:  # noqa: ANN001
            _ = on_data
            self.closed = False
            self.started = False
            if native_state == "missing":
                self._backend = SimpleNamespace(_native=SimpleNamespace())
            elif native_state == "throws":

                def fail_verification() -> bool:
                    raise RuntimeError("native detail")

                self._backend = SimpleNamespace(
                    _native=SimpleNamespace(is_process_specific=fail_verification)
                )
            else:
                self._backend = SimpleNamespace(
                    _native=SimpleNamespace(is_process_specific=lambda: native_state)
                )
            captures.append(self)

        def start(self) -> None:
            self.started = True

        def close(self) -> None:
            self.closed = True

    import puripuly_heart.core.audio.process_source as source_module

    monkeypatch.setattr(
        source_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(ProcessAudioCapture=PublicCapture),
    )

    with pytest.raises(
        ProcessAudioCaptureSetupError, match="mode could not be verified"
    ) as exc_info:
        ProcTapProcessAudioCaptureFactory(platform_availability=_supported).create(
            pid=102,
            on_data=lambda _data, _frames: None,
        )

    assert captures[0].closed is True
    assert captures[0].started is False
    assert "native detail" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_silence_does_not_signal_terminal_state() -> None:
    factory = FakeFactory()
    watcher = FakeWatcher()
    source = ProcessAudioCaptureSource(
        identity=_identity(),
        watcher=watcher,
        capture_factory=factory,
        platform_availability=_supported,
    )

    await asyncio.sleep(0)

    assert source.terminal_reason is None
    assert factory.captures[0][1].closed is False
    await source.close()


def _spawn_live_process_identity() -> (
    tuple[subprocess.Popen[bytes], ResolvedProcessCaptureIdentity]
):
    psutil = pytest.importorskip("psutil")
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        process = psutil.Process(child.pid)
        identity = ResolvedProcessCaptureIdentity(
            pid=child.pid,
            target=ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Game\Game.exe"),
            instance_id=f"{child.pid}:{process.create_time()}",
        )
    except Exception:
        child.terminate()
        child.wait(timeout=5)
        raise
    return child, identity


@pytest.mark.asyncio
async def test_process_source_close_stops_real_identity_watch_thread_while_target_alive() -> None:
    child, identity = _spawn_live_process_identity()
    factory = FakeFactory()
    try:
        source = ProcessAudioCaptureSource(
            identity=identity,
            watcher=PsutilProcessIdentityWatcher(),
            capture_factory=factory,
            platform_availability=_supported,
        )
        watch = source._watch
        assert watch is not None
        assert watch.watch_thread_alive is True

        started = time.monotonic()
        await source.close()
        elapsed = time.monotonic() - started

        assert watch.watch_thread_alive is False
        assert elapsed < 1.0
        assert child.poll() is None
        assert factory.captures[0][1].closed is True
        assert source.terminal_reason == "closed"
    finally:
        child.terminate()
        child.wait(timeout=5)
