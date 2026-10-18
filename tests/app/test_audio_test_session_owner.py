from __future__ import annotations

import asyncio

import numpy as np
import pytest

from puripuly_heart.app.adapters.audio_test_capture import AudioTestCaptureAdapter
from puripuly_heart.app.ports.audio_test import (
    AudioTestCaptureState,
    AudioTestOpenedSource,
    AudioTestSourceUnavailable,
)
from puripuly_heart.app.services.audio_test import AudioTestChannelPlan, AudioTestSessionOwner
from puripuly_heart.core.audio.audio_test_vad import AudioTestVadProbe, AudioTestVadSettings
from puripuly_heart.core.audio.format import AudioFrameF32


class LoudnessEngine:
    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float:
        return 0.9 if float(np.max(np.abs(samples))) > 0.1 else 0.05

    def reset(self) -> None:
        return None


class FeedSource:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[AudioFrameF32 | None] = asyncio.Queue()
        self.closed = False

    def feed(self, amplitude: float, chunks: int = 1) -> None:
        samples = np.full(512 * chunks, amplitude, dtype=np.float32)
        self.queue.put_nowait(AudioFrameF32(samples=samples, sample_rate_hz=16000))

    async def frames(self):
        while True:
            frame = await self.queue.get()
            if frame is None:
                return
            yield frame

    async def close(self) -> None:
        self.closed = True
        self.queue.put_nowait(None)


class Opener:
    def __init__(self, source: FeedSource | BaseException, device: dict[str, object]) -> None:
        self.source = source
        self.device = device

    async def open(self, *, on_acquired) -> AudioTestOpenedSource:
        if isinstance(self.source, BaseException):
            raise self.source
        on_acquired(self.source)
        return AudioTestOpenedSource(source=self.source, device=self.device)


class CaptureControl:
    def __init__(self, *, self_enabled: bool, peer_enabled: bool) -> None:
        self.enabled = {"self": self_enabled, "peer": peer_enabled}
        self.generation = {"self": 1, "peer": 1}
        self.pending = {"self": False, "peer": False}
        self.stuck_open: set[str] = set()
        self.calls: list[tuple[str, bool]] = []

    def state(self, channel: str) -> AudioTestCaptureState:
        enabled = self.enabled[channel]
        source_open = channel in self.stuck_open
        return AudioTestCaptureState(
            enabled=enabled,
            busy=enabled or self.pending[channel] or source_open,
            source_open=source_open,
            token=("owner", self.generation[channel]),
        )

    async def set_enabled(self, channel: str, enabled: bool) -> None:
        self.calls.append((channel, enabled))
        self.user_set(channel, enabled)
        self.pending[channel] = False

    def user_set(self, channel: str, enabled: bool) -> None:
        self.enabled[channel] = enabled
        self.generation[channel] += 1


class Harness:
    def __init__(self, *, self_enabled: bool = True, peer_enabled: bool = True) -> None:
        self.sources = {"self": FeedSource(), "peer": FeedSource()}
        self.openers: dict[str, object] = {
            "self": Opener(self.sources["self"], {"name": "Desk Mic", "host_api": "WASAPI"}),
            "peer": Opener(self.sources["peer"], {"name": "Speakers", "default_fallback": True}),
        }
        self.thresholds = {"self": 0.5, "peer": 0.5}
        self.control = CaptureControl(self_enabled=self_enabled, peer_enabled=peer_enabled)
        self.observed: list[tuple[str, int]] = []
        self.logs: list[str] = []
        self.probe_error: Exception | None = None
        self.owner = AudioTestSessionOwner(
            capture_port=AudioTestCaptureAdapter(log_sink=self.logs.append),
            plan_factory=self.plan,
            vad_settings=self.vad_settings,
            capture_control=self.control,
            log_sink=self.logs.append,
        )

    def vad_settings(self, channel: str) -> AudioTestVadSettings:
        return AudioTestVadSettings(
            speech_threshold=self.thresholds[channel],
            hangover_ms=64,
            ring_buffer_ms=300,
        )

    def plan(self, channel: str) -> AudioTestChannelPlan:
        def probe(settings: AudioTestVadSettings) -> AudioTestVadProbe:
            if self.probe_error is not None:
                raise self.probe_error
            return AudioTestVadProbe(engine=LoudnessEngine(), settings=settings)

        return AudioTestChannelPlan(
            opener=self.openers[channel],
            probe_factory=probe,
            ready_on_open=channel == "peer",
            frame_timeout_s=5.0 if channel == "self" else None,
        )

    async def start(self) -> bool:
        return await self.owner.start(
            observation_callback=lambda channel, samples: self.observed.append(
                (channel, len(samples))
            )
        )

    def channel(self, name: str) -> dict[str, object]:
        return self.owner.snapshot["channels"][name]


async def _until(predicate) -> None:
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached")


@pytest.mark.asyncio
async def test_test_pauses_translation_capture_observes_both_channels_and_restores() -> None:
    harness = Harness()

    assert await harness.start() is True
    harness.sources["self"].feed(0.0)
    assert await harness.owner.wait_ready() is True

    snapshot = harness.owner.snapshot
    assert snapshot["state"] == "ready"
    assert snapshot["effective_active"] is True
    assert snapshot["restore_capture"] == ["self", "peer"]
    assert harness.control.calls == [("self", False), ("peer", False)]
    assert harness.channel("self")["device"] == {"name": "Desk Mic", "host_api": "WASAPI"}
    assert harness.channel("peer")["device"]["default_fallback"] is True

    harness.sources["peer"].feed(0.5, chunks=4)
    await _until(lambda: harness.channel("peer")["speech_detected"])
    peer = harness.channel("peer")
    assert peer["speech_probability"] == pytest.approx(0.9)
    assert peer["meter_level"] == pytest.approx(0.5)
    assert peer["speech_threshold"] == pytest.approx(0.5)
    assert ("peer", 4) in harness.observed

    await harness.owner.stop()

    assert harness.sources["self"].closed and harness.sources["peer"].closed
    assert harness.owner.snapshot["state"] == "off"
    assert harness.owner.snapshot["restore_capture"] == []
    assert harness.control.calls[-2:] == [("self", True), ("peer", True)]
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
async def test_user_capture_choices_during_the_test_are_not_overridden() -> None:
    harness = Harness()
    assert await harness.start() is True
    harness.sources["self"].feed(0.0)
    await harness.owner.wait_ready()

    harness.control.user_set("self", True)
    harness.control.user_set("self", False)
    harness.owner.release_capture_restore("peer")
    await harness.owner.stop()

    assert harness.control.calls == [("self", False), ("peer", False)]
    assert harness.control.enabled == {"self": False, "peer": False}


@pytest.mark.asyncio
async def test_pending_capture_is_released_but_only_enabled_capture_is_restored() -> None:
    harness = Harness(self_enabled=False, peer_enabled=False)
    harness.control.pending["self"] = True

    assert await harness.start() is True
    assert harness.control.calls == [("self", False)]
    assert harness.owner.snapshot["restore_capture"] == []

    await harness.owner.stop()

    assert harness.control.calls == [("self", False)]


@pytest.mark.asyncio
async def test_capture_that_stays_open_blocks_the_test_and_restores_paused_channels() -> None:
    harness = Harness()
    harness.control.stuck_open.add("peer")

    assert await harness.start() is False

    assert harness.control.calls == [("self", False), ("peer", False), ("self", True)]
    assert harness.owner.active is False
    assert harness.owner.snapshot["state"] == "failed"
    assert any("failed to pause translation capture" in line for line in harness.logs)


@pytest.mark.asyncio
async def test_one_unavailable_channel_degrades_the_test_without_stopping_the_other() -> None:
    harness = Harness()
    harness.openers["peer"] = Opener(
        AudioTestSourceUnavailable("target_unavailable", "process_not_running"),
        {},
    )

    assert await harness.start() is True
    harness.sources["self"].feed(0.0)

    assert await harness.owner.wait_ready() is False
    snapshot = harness.owner.snapshot
    assert snapshot["state"] == "degraded"
    assert snapshot["effective_active"] is True
    assert harness.channel("self")["state"] == "ready"
    assert harness.channel("peer")["state"] == "failed"
    assert harness.channel("peer")["failure_reason"] == "target_unavailable"
    assert harness.channel("peer")["failure_type"] == "process_not_running"

    await harness.owner.stop()


@pytest.mark.asyncio
async def test_vad_unavailable_fails_channels_without_opening_sources() -> None:
    harness = Harness(self_enabled=False, peer_enabled=False)
    harness.probe_error = RuntimeError("onnx missing")

    assert await harness.start() is True
    assert await harness.owner.wait_ready() is False

    assert harness.owner.snapshot["state"] == "failed"
    assert harness.channel("self")["failure_reason"] == "vad_unavailable"
    assert harness.sources["self"].closed is False
    await harness.owner.stop()


@pytest.mark.asyncio
async def test_threshold_setting_changes_reach_the_running_probe() -> None:
    harness = Harness(self_enabled=False, peer_enabled=False)
    assert await harness.start() is True
    harness.sources["self"].feed(0.0)
    await harness.owner.wait_ready()

    harness.thresholds["self"] = 0.95
    harness.sources["self"].feed(0.5, chunks=2)
    await _until(lambda: harness.channel("self")["meter_level"] > 0.4)

    assert harness.channel("self")["speech_threshold"] == pytest.approx(0.95)
    assert harness.channel("self")["speech_detected"] is False
    await harness.owner.stop()


@pytest.mark.asyncio
async def test_frames_after_stop_and_restart_after_close_are_rejected() -> None:
    harness = Harness(self_enabled=False, peer_enabled=False)
    assert await harness.start() is True
    harness.sources["self"].feed(0.0)
    await harness.owner.wait_ready()
    generation = harness.owner.runtime_for("self").generation

    await harness.owner.stop()
    harness.owner.observe_frame(
        "self",
        generation,
        AudioFrameF32(samples=np.full(512, 0.5, dtype=np.float32), sample_rate_hz=16000),
    )
    assert harness.channel("self")["meter_level"] == 0.0

    await harness.owner.close()
    assert await harness.start() is False
    assert harness.control.calls == []


@pytest.mark.asyncio
async def test_active_start_attaches_detaches_and_reattaches_without_restarting() -> None:
    harness = Harness()
    assert await harness.owner.start()
    harness.sources["self"].feed(0.0)
    assert await harness.owner.wait_ready()
    runtimes = {channel: harness.owner.runtime_for(channel) for channel in ("self", "peer")}
    generations = {channel: runtime.generation for channel, runtime in runtimes.items()}
    first: list[str] = []
    second: list[str] = []

    assert await harness.owner.start(
        observation_callback=lambda channel, _samples: first.append(channel)
    )
    harness.sources["peer"].feed(0.5)
    await _until(lambda: first == ["peer"])
    assert await harness.owner.start(observation_callback=None)
    harness.sources["peer"].feed(0.2)
    await _until(lambda: harness.channel("peer")["meter_level"] == pytest.approx(0.2))
    assert first == ["peer"]
    assert await harness.owner.start(
        observation_callback=lambda channel, _samples: second.append(channel)
    )
    harness.sources["self"].feed(0.4)
    await _until(lambda: second == ["self"])
    assert harness.owner.snapshot["restore_capture"] == ["self", "peer"]
    assert {channel: runtime.generation for channel, runtime in runtimes.items()} == generations
    assert not any(source.closed for source in harness.sources.values())
    await harness.owner.stop()
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_succeeds", (False, True))
async def test_failed_retry_retains_original_capture_restoration(retry_succeeds: bool) -> None:
    harness = Harness()
    for channel in ("self", "peer"):
        harness.openers[channel] = Opener(AudioTestSourceUnavailable("unavailable"), {})
    assert await harness.start()
    assert await harness.owner.wait_ready() is False
    await _until(lambda: not harness.owner.active)
    assert harness.owner.snapshot["restore_capture"] == ["self", "peer"]
    if retry_succeeds:
        for channel in ("self", "peer"):
            harness.openers[channel] = Opener(harness.sources[channel], {})
    assert await harness.start()
    if retry_succeeds:
        harness.sources["self"].feed(0.0)
        assert await harness.owner.wait_ready()
    else:
        assert await harness.owner.wait_ready() is False
    assert harness.owner.snapshot["restore_capture"] == ["self", "peer"]
    await harness.owner.stop()
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
async def test_failed_retry_preserves_explicit_override_and_owner_replacement() -> None:
    harness = Harness()
    harness.probe_error = RuntimeError("unavailable")
    assert await harness.start()
    assert await harness.owner.wait_ready() is False
    await _until(lambda: not harness.owner.active)
    harness.owner.release_capture_restore("self")
    harness.control.user_set("peer", False)
    assert await harness.start()
    assert await harness.owner.wait_ready() is False
    await harness.owner.stop()
    assert harness.control.enabled == {"self": False, "peer": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", (False, True))
async def test_first_channel_cleanup_failure_does_not_skip_peer_and_retries(
    closing: bool, monkeypatch
) -> None:
    harness = Harness()
    assert await harness.start()
    harness.sources["self"].feed(0.0)
    assert await harness.owner.wait_ready()
    runtime = harness.owner.runtime_for("self")
    operation = "close" if closing else "stop"
    original = getattr(runtime, operation)

    async def fail() -> None:
        raise RuntimeError("synthetic close failure")

    monkeypatch.setattr(runtime, operation, fail)
    with pytest.raises(ExceptionGroup, match="Audio test cleanup failed"):
        await getattr(harness.owner, operation)()
    assert harness.sources["peer"].closed
    assert harness.owner.runtime_for("peer").session_task is None
    assert not harness.sources["self"].closed
    assert harness.channel("self")["state"] == "failed"
    assert harness.channel("self")["failure_reason"] == "resource_release_incomplete"
    assert harness.channel("peer")["state"] == "off"
    assert harness.owner.snapshot["effective_active"] is True
    assert harness.control.enabled == {"self": False, "peer": False}
    monkeypatch.setattr(runtime, operation, original)
    await getattr(harness.owner, operation)()
    assert harness.sources["self"].closed
    assert harness.owner.snapshot["state"] == "off"
    assert not harness.owner.active
    assert harness.control.enabled == {"self": not closing, "peer": not closing}
