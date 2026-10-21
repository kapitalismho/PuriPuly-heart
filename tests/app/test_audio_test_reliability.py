from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import janus
import pytest

from puripuly_heart.app.adapters.audio_test_sources import LoopbackAudioTestSourceOpener
from puripuly_heart.app.services.capture.peer_capture_target_application import (
    PeerCaptureTargetApplicationOwner,
)
from puripuly_heart.core.audio.desktop_source import DesktopLoopbackAudioSource
from puripuly_heart.core.audio.process_source import ProcessAudioCaptureSource
from puripuly_heart.core.audio.source import SoundDeviceAudioSource
from puripuly_heart.core.peer_capture import (
    PeerCaptureResolvedTarget,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.core.runtime.audio_test import AudioTestCaptureRuntime
from tests.app.test_audio_test_session_owner import Harness
from tests.app.test_audio_test_sources import Resolver, _microphone, _route


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ("self", "peer"))
@pytest.mark.parametrize("close_fails", (False, True))
async def test_blocking_source_open_keeps_loop_responsive_and_owns_cancelled_late_source(
    channel: str,
    close_fails: bool,
) -> None:
    harness = Harness(self_enabled=False, peer_enabled=False)
    harness.owner.runtime_factory = lambda: AudioTestCaptureRuntime(cancel_timeout_s=0.001)
    entered = threading.Event()
    release = threading.Event()
    factory_calls: list[None] = []
    source = harness.sources[channel]
    original_close = source.close
    failures = [RuntimeError("late close failed")] if close_fails else []

    async def close():
        if failures:
            raise failures.pop()
        await original_close()

    source.close = close

    def factory(*_args, **_kwargs):
        factory_calls.append(None)
        entered.set()
        assert release.wait(5)
        return source

    if channel == "self":
        harness.openers[channel] = _microphone(_route(), factory)
    else:
        target = PeerCaptureTargetIntent(kind="default_output_device")
        harness.openers[channel] = LoopbackAudioTestSourceOpener(
            target=target,
            output_device="",
            target_resolver=Resolver(
                PeerCaptureTargetResolution(
                    status=PeerCaptureTargetStatus.RESOLVED,
                    target=PeerCaptureResolvedTarget(intent=target),
                )
            ),
            source_factory=factory,
        )
    try:
        assert await harness.start()
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.2)
        with pytest.raises(RuntimeError, match="resource release is incomplete"):
            await harness.owner.stop()
        runtime = harness.owner.runtime_for(channel)
        task = runtime.session_task
        assert task is not None and not task.done()
        assert harness.channel(channel)["failure_reason"] == "resource_release_incomplete"
        assert await harness.start() is False
        assert not source.closed
        assert len(factory_calls) == 1
        release.set()
        if close_fails:
            await asyncio.wait_for(asyncio.shield(task), timeout=2)
            assert runtime.source is source
            assert not source.closed
            assert harness.owner.snapshot["effective_active"] is True
        else:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=2)
            assert source.closed
            assert runtime.source is None
        await harness.owner.stop()
        assert source.closed
        assert harness.owner.snapshot["state"] == "off"
    finally:
        release.set()
        tasks = [harness.owner.runtime_for(item).session_task for item in ("self", "peer")]
        await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
        await harness.owner.close()


@pytest.mark.asyncio
async def test_blocking_vad_factory_drains_before_cancelled_session_retires() -> None:
    harness = Harness(self_enabled=False, peer_enabled=False)
    harness.owner.runtime_factory = lambda: AudioTestCaptureRuntime(cancel_timeout_s=0.001)
    entered = threading.Event()
    release = threading.Event()
    factory_calls: list[None] = []
    original = harness.plan

    def plan(channel):
        value = original(channel)
        if channel != "self":
            return value

        def probe(settings):
            factory_calls.append(None)
            entered.set()
            assert release.wait(5)
            return value.probe_factory(settings)

        return replace(value, probe_factory=probe)

    harness.owner.plan_factory = plan
    try:
        assert await harness.start()
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.2)
        with pytest.raises(RuntimeError, match="resource release is incomplete"):
            await harness.owner.stop()
        task = harness.owner.runtime_for("self").session_task
        assert task is not None and not task.done()
        assert await harness.start() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=2)
        assert len(factory_calls) == 1
        assert harness.owner.runtime_for("self").source is None
        await harness.owner.stop()
        assert not harness.owner.active
    finally:
        release.set()
        tasks = [harness.owner.runtime_for(item).session_task for item in ("self", "peer")]
        await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
        await harness.owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_type",
    (
        SoundDeviceAudioSource,
        DesktopLoopbackAudioSource,
        ProcessAudioCaptureSource,
    ),
)
async def test_native_close_uses_owned_worker_and_keeps_janus_on_application_loop(
    source_type,
) -> None:
    source = object.__new__(source_type)
    queue = janus.Queue()
    await queue.async_q.put(None)
    source._queue = queue
    source._closed = False
    entered = threading.Event()
    release = threading.Event()
    close_calls: list[None] = []

    def close_native():
        close_calls.append(None)
        entered.set()
        assert release.wait(5)

    if source_type is SoundDeviceAudioSource:
        source._stream = SimpleNamespace(stop=close_native, close=lambda: None)
    elif source_type is DesktopLoopbackAudioSource:
        source._stream = SimpleNamespace(stop_stream=close_native, close=lambda: None)
        source._manager = SimpleNamespace(terminate=lambda: None)
    else:
        source._lock = threading.RLock()
        source._terminal_reason = None
        source._terminal_diagnostics = None
        source._watch = None
        source._capture = SimpleNamespace(close=close_native)
    task = asyncio.create_task(source.close())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not queue.closed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert queue.closed
        assert len(close_calls) == 1
        await source.close()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        queue.close()
        await queue.wait_closed()


@pytest.mark.asyncio
async def test_specialized_target_cutover_stops_all_test_sources_before_persist_and_refresh() -> (
    None
):
    from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, with_capture_target

    harness = Harness()
    assert await harness.start()
    harness.sources["self"].feed(0.0)
    assert await harness.owner.wait_ready()
    events: list[str] = []
    settings = SimpleNamespace(canonical=AppSettingsVNext())

    def persist(intent):
        assert all(source.closed for source in harness.sources.values())
        events.append("persist")
        return with_capture_target(settings.canonical, intent)

    settings.apply_capture_target = persist
    settings.remember_projection = lambda _settings: None

    async def refresh(_settings):
        assert not harness.owner.active
        events.append("refresh")

    owner = PeerCaptureTargetApplicationOwner(
        settings=settings,
        localize=lambda value: value,
        processes=SimpleNamespace(candidates=lambda: ()),
        devices=SimpleNamespace(names=lambda: ("New Speakers",)),
        runtime_effects=SimpleNamespace(apply_capture_target=refresh),
        settings_presentation_sink=lambda _settings: None,
        warning_reset=lambda: None,
        stop_audio_test=harness.owner.stop,
    )
    try:
        await owner.apply("device:New Speakers")
        assert events == ["persist", "refresh"]
        assert harness.owner.snapshot["state"] == "off"
        assert harness.control.enabled == {"self": True, "peer": True}
        assert owner.current_value() == "device:New Speakers"
    finally:
        await harness.owner.close()
