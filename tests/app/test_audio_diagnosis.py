from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.audio_test import AudioDiagnosticCandidate
from puripuly_heart.app.services import audio_test
from puripuly_heart.app.services.application_audio_devices import (
    enumerate_microphone_diagnostic_candidates,
    safe_audio_device_label,
)
from puripuly_heart.core.audio.audio_test_vad import AudioTestVadProbe
from puripuly_heart.core.audio.source import SoundDeviceAudioSource
from puripuly_heart.core.runtime.audio_test import AudioTestCaptureRuntime
from tests.app.test_audio_test_session_owner import FeedSource, Harness, Opener, _until
from tests.app.test_audio_test_sources import _microphone, _route


def candidate(index: int, *, current: bool = False) -> AudioDiagnosticCandidate:
    return AudioDiagnosticCandidate(
        input_host_api="Windows WASAPI",
        input_device=f"Mic {index}",
        device_index=index,
        max_input_channels=2,
        current_route=current,
    )


@pytest.fixture(autouse=True)
def bounded_diagnosis(monkeypatch):
    monkeypatch.setattr(audio_test, "AUDIO_DIAGNOSE_CANDIDATE_TIMEOUT_S", 0.025)
    monkeypatch.setattr(audio_test, "DIAGNOSTIC_CLEANUP_TIMEOUT_S", 0.1)


def diagnose(harness, candidates, openers, *, planned=None):
    def plan(route):
        if planned is not None:
            planned.append(route.device_index)
        return replace(harness.plan("self"), opener=openers[route.device_index])

    return harness.owner.diagnose(
        candidates=tuple(candidates),
        plan_factory=plan,
        input_host_api="Windows WASAPI",
        input_device="Mic 0",
    )


@pytest.mark.asyncio
async def test_sequential_routes_preserve_open_frame_silence_speech_and_failure_evidence():
    harness = Harness()
    sources = {index: FeedSource() for index in (1, 2, 3, 4)}
    sources[1].feed(0.0, chunks=2)
    sources[2].feed(0.5, chunks=4)
    opened = []

    class SequentialOpener(Opener):
        async def open(self, *, on_acquired):
            assert all(source.closed for source in opened)
            result = await super().open(on_acquired=on_acquired)
            opened.append(result.source)
            return result

    openers = {
        0: Opener(OSError("private device path and credential"), {}),
        **{index: SequentialOpener(source, {}) for index, source in sources.items()},
    }
    planned = []
    result = await diagnose(
        harness,
        [candidate(index, current=index == 0) for index in range(5)],
        openers,
        planned=planned,
    )

    assert result["status"] == "applied"
    assert result["recognition"] == "not_exercised"
    assert result["settings_changed"] is False
    diagnosis = result["diagnosis"]
    assert diagnosis["state"] == "completed"
    assert planned == [0, 1, 2, 3]
    failed, silent, speech, empty = diagnosis["candidates"]
    assert not failed["opened"] and not failed["frames_received"]
    assert failed["failure_code"] == "source_open_failed"
    assert silent["opened"] and silent["frames_received"] and silent["silence_observed"]
    assert silent["vad_sample_count"] == 2 and not silent["speech_detected"]
    assert silent["failure_code"] is None
    assert speech["speech_detected"] and speech["frame_count"] == 1
    assert speech["speech_probability_max"] == pytest.approx(0.9)
    assert speech["failure_code"] is None
    assert empty["opened"] and not empty["frames_received"]
    assert empty["failure_code"] == "candidate_timeout"
    assert diagnosis["selected_candidate_id"] == "input-2"
    assert diagnosis["selection_basis"] == "speech"
    assert diagnosis["cleanup_complete"]
    assert diagnosis["restoration"] == {"self": "restored", "peer": "restored"}
    assert harness.control.calls == [
        ("self", False),
        ("peer", False),
        ("self", True),
        ("peer", True),
    ]
    assert all(sources[index].closed for index in (1, 2, 3))
    assert not sources[4].closed
    assert not harness.owner.active
    assert "private device" not in json.dumps(result)


@pytest.mark.asyncio
async def test_probabilities_below_configured_threshold_remain_measured_not_speech_labels():
    harness = Harness()
    harness.thresholds["self"] = 0.95
    source = FeedSource()
    source.feed(0.4, chunks=3)

    class MeasuredEngine:
        def __init__(self):
            self.values = iter((0.1, 0.55, 0.72))

        def speech_probability(self, _samples, *, sample_rate_hz):
            return next(self.values)

        def reset(self):
            pass

    plan = replace(
        harness.plan("self"),
        opener=Opener(source, {}),
        probe_factory=lambda settings: AudioTestVadProbe(
            engine=MeasuredEngine(), settings=settings
        ),
    )
    result = await harness.owner.diagnose(
        candidates=(candidate(0, current=True),),
        plan_factory=lambda _candidate: plan,
        input_host_api="Windows WASAPI",
        input_device="Mic 0",
    )

    observed = result["diagnosis"]["candidates"][0]
    assert observed["speech_probability_min"] == pytest.approx(0.1)
    assert observed["speech_probability_max"] == pytest.approx(0.72)
    assert observed["observed_speech_probabilities"] == [0.1, 0.55, 0.72]
    assert observed["speech_threshold"] == pytest.approx(0.95)
    assert observed["speech_detected"] is False
    assert result["diagnosis"]["selection_basis"] == "frames"
    assert "recommended" not in json.dumps(result)
    assert result["recognition"] == "not_exercised"


@pytest.mark.asyncio
async def test_received_speech_does_not_erase_later_capture_failure():
    harness = Harness()

    class FailingSource(FeedSource):
        async def frames(self):
            yield await self.queue.get()
            raise OSError("private failure prose")

    source = FailingSource()
    source.feed(0.5, chunks=4)
    result = await diagnose(harness, (candidate(0, current=True),), {0: Opener(source, {})})
    observed = result["diagnosis"]["candidates"][0]
    assert observed["speech_detected"] and observed["frame_count"] == 1
    assert observed["failure_code"] == "capture_failed"
    assert observed["cleanup_complete"] and source.closed
    assert result["status"] == "action_required"
    assert result["diagnosis"]["selected_candidate_id"] is None
    assert result["diagnosis"]["selection_basis"] is None
    assert "private failure prose" not in json.dumps(result)


@pytest.mark.asyncio
async def test_cancel_drains_source_before_capture_restoration():
    harness = Harness()
    source = FeedSource()
    original = harness.control.set_enabled

    async def set_enabled(channel, enabled):
        if enabled:
            assert source.closed
        await original(channel, enabled)

    harness.control.set_enabled = set_enabled
    task = asyncio.create_task(
        diagnose(harness, (candidate(0, current=True),), {0: Opener(source, {})})
    )
    await _until(lambda: harness.owner.runtime_for("self") is not None)
    await _until(lambda: harness.owner.runtime_for("self").source is source)
    task.cancel()
    result = await task

    assert result["status"] == "cancelled"
    assert result["diagnosis"]["state"] == "cancelled"
    assert result["diagnosis"]["cleanup_complete"]
    assert result["diagnosis"]["restoration"] == {"self": "restored", "peer": "restored"}
    assert source.closed and not harness.owner.active


@pytest.mark.asyncio
async def test_repeated_cancel_cannot_interrupt_close_or_restore_early(monkeypatch):
    monkeypatch.setattr(audio_test, "DIAGNOSTIC_CLEANUP_TIMEOUT_S", 1.0)
    harness = Harness()
    entered = asyncio.Event()
    release = asyncio.Event()

    class ClosingSource(FeedSource):
        async def close(self):
            entered.set()
            await release.wait()
            await super().close()

    source = ClosingSource()
    task = asyncio.create_task(
        diagnose(harness, (candidate(0, current=True),), {0: Opener(source, {})})
    )
    await _until(lambda: harness.owner.runtime_for("self") is not None)
    await _until(lambda: harness.owner.runtime_for("self").source is source)
    task.cancel()
    await entered.wait()
    task.cancel()
    assert harness.control.enabled == {"self": False, "peer": False}
    release.set()
    result = await task

    assert result["status"] == "cancelled"
    assert source.closed and result["diagnosis"]["cleanup_complete"]
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
async def test_close_failure_retains_source_restore_debt_and_measured_evidence():
    harness = Harness()

    class RetainedSource(FeedSource):
        release_allowed = False

        async def close(self):
            if not self.release_allowed:
                raise OSError("private close path")
            await super().close()

    source = RetainedSource()
    source.feed(0.5, chunks=4)
    result = await diagnose(harness, (candidate(0, current=True),), {0: Opener(source, {})})
    diagnosis = result["diagnosis"]

    assert result["status"] == "failed"
    assert diagnosis["failure_code"] == "resource_release_incomplete"
    assert diagnosis["candidates"][0]["speech_detected"]
    assert diagnosis["candidates"][0]["cleanup_complete"] is False
    assert diagnosis["selected_candidate_id"] is None
    assert diagnosis["selection_basis"] is None
    assert diagnosis["restoration"] == {"self": "cleanup_pending", "peer": "cleanup_pending"}
    assert harness.owner.runtime_for("self").source is source
    assert harness.owner.active and not source.closed
    assert harness.control.enabled == {"self": False, "peer": False}
    source.release_allowed = True
    await harness.owner.stop()
    assert source.closed and not harness.owner.active
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
async def test_uncancellable_open_keeps_cleanup_ownership_until_late_acquisition_is_released():
    harness = Harness()
    harness.owner.runtime_factory = lambda: AudioTestCaptureRuntime(cancel_timeout_s=0.005)
    source = FeedSource()
    opening = asyncio.Event()
    release = asyncio.Event()

    class DelayedOpener(Opener):
        async def open(self, *, on_acquired):
            opening.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return await super().open(on_acquired=on_acquired)

    task = asyncio.create_task(
        diagnose(harness, (candidate(0, current=True),), {0: DelayedOpener(source, {})})
    )
    await opening.wait()
    task.cancel()
    result = await task
    assert result["status"] == "cancelled"
    assert not result["diagnosis"]["cleanup_complete"]
    assert harness.owner.active
    assert harness.control.enabled == {"self": False, "peer": False}
    release.set()
    await _until(lambda: source.closed)
    await harness.owner.stop()
    assert not harness.owner.active
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("release_authority", (False, True))
async def test_user_capture_changes_and_owner_replacement_are_not_revived(release_authority):
    harness = Harness()
    source = FeedSource()
    source.feed(0.0)
    task = asyncio.create_task(
        diagnose(harness, (candidate(0, current=True),), {0: Opener(source, {})})
    )
    await _until(lambda: harness.owner.runtime_for("self") is not None)
    await _until(lambda: harness.owner.runtime_for("self").source is source)
    if release_authority:
        harness.owner.release_capture_restore("self")
    else:
        harness.control.user_set("self", False)
    result = await task
    assert result["diagnosis"]["restoration"]["self"] == "skipped"
    assert harness.control.enabled == {"self": False, "peer": True}
    assert ("self", True) not in harness.control.calls


@pytest.mark.asyncio
async def test_user_off_during_pause_revokes_not_yet_recorded_restore_authority():
    harness = Harness()
    entered = asyncio.Event()
    release = asyncio.Event()
    original = harness.control.set_enabled

    async def set_enabled(channel, enabled):
        if channel == "self" and not enabled:
            entered.set()
            await release.wait()
        await original(channel, enabled)

    harness.control.set_enabled = set_enabled
    source = FeedSource()
    task = asyncio.create_task(
        diagnose(harness, (candidate(0, current=True),), {0: Opener(source, {})})
    )
    await entered.wait()
    harness.owner.release_capture_restore("self")
    release.set()
    result = await task
    assert result["diagnosis"]["restoration"]["self"] == "skipped"
    assert harness.control.enabled == {"self": False, "peer": True}
    assert ("self", True) not in harness.control.calls


@pytest.mark.asyncio
async def test_restore_failure_is_reported_and_owned_for_retry():
    harness = Harness()
    source = FeedSource()
    source.feed(0.0)
    original = harness.control.set_enabled

    async def set_enabled(channel, enabled):
        if channel == "self" and enabled:
            raise OSError("private restore detail")
        await original(channel, enabled)

    harness.control.set_enabled = set_enabled
    result = await diagnose(harness, (candidate(0, current=True),), {0: Opener(source, {})})
    assert result["status"] == "failed"
    assert result["diagnosis"]["failure_code"] == "restore_failed"
    assert result["diagnosis"]["restoration"] == {"self": "failed", "peer": "restored"}
    assert harness.owner.snapshot["restore_capture"] == ["self"]
    assert source.closed and not harness.owner.active
    harness.control.set_enabled = original
    await harness.owner.stop()
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
async def test_diagnosis_never_takes_over_an_existing_manual_test():
    harness = Harness()
    assert await harness.start()
    harness.sources["self"].feed(0.0)
    assert await harness.owner.wait_ready()
    before = list(harness.control.calls)
    result = await diagnose(harness, (candidate(0, current=True),), {})
    assert result["status"] == "action_required"
    assert result["diagnosis"]["failure_code"] == "audio_test_busy"
    assert result["diagnosis"]["candidates"] == []
    assert harness.control.calls == before
    assert not any(source.closed for source in harness.sources.values())
    await harness.owner.stop()


def test_enumeration_uses_actual_input_routes_current_first_and_no_invented_profiles(monkeypatch):
    devices = [
        {"name": "Other", "hostapi": 0, "max_input_channels": 1},
        {"name": "Desk Mic", "hostapi": 1, "max_input_channels": 2},
        {"name": "Output only", "hostapi": 1, "max_input_channels": 0},
        {"name": "Desk Mic", "hostapi": 0, "max_input_channels": 1},
    ]
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_hostapis=lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}],
            query_devices=lambda: devices,
            default=SimpleNamespace(device=(1, 2)),
        ),
    )
    routes = enumerate_microphone_diagnostic_candidates(
        input_host_api="Windows WASAPI (Compatibility Mode)",
        input_device="Desk Mic",
        max_candidates=4,
    )
    assert [route.device_index for route in routes] == [1, 3, 0]
    assert routes[0].current_route
    assert routes[0].input_host_api == "Windows WASAPI (Compatibility Mode)"
    assert [route.input_host_api for route in routes[1:]] == ["MME", "MME"]
    missing = enumerate_microphone_diagnostic_candidates(
        input_host_api="missing API", input_device="missing mic", max_candidates=2
    )
    assert len(missing) == 2 and not any(route.current_route for route in missing)
    automatic = enumerate_microphone_diagnostic_candidates(
        input_host_api="", input_device="", max_candidates=4
    )
    assert automatic[0].device_index == 1 and automatic[0].current_route


@pytest.mark.parametrize("current_index", (1, 5))
def test_present_vr_microphone_drivers_precede_unrelated_and_duplicate_host_routes(
    monkeypatch, current_index
):
    devices = [
        {"name": "Unrelated microphone", "hostapi": 0, "max_input_channels": 1},
        {"name": "Desk Mic", "hostapi": 1, "max_input_channels": 1},
        {"name": "Speakers (Virtual Desktop Audio)", "hostapi": 1, "max_input_channels": 0},
        {"name": "Steam Streaming Speakers", "hostapi": 1, "max_input_channels": 1},
        {"name": "Microphone (Virtual Desktop Audio)", "hostapi": 0, "max_input_channels": 1},
        {"name": "Microphone (Virtual Desktop Audio)", "hostapi": 1, "max_input_channels": 1},
        {"name": "Steam Streaming Microphone", "hostapi": 1, "max_input_channels": 1},
        {
            "name": "Headset Microphone (Oculus Virtual Audio Device)",
            "hostapi": 1,
            "max_input_channels": 1,
        },
        {"name": "Microphone (PicoStreamingMicrophone)", "hostapi": 1, "max_input_channels": 1},
        {"name": "PICO Camera Audio", "hostapi": 1, "max_input_channels": 1},
    ]
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_hostapis=lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}],
            query_devices=lambda: devices,
            default=SimpleNamespace(device=(current_index, -1)),
        ),
    )
    routes = enumerate_microphone_diagnostic_candidates(
        input_host_api="Windows WASAPI",
        input_device=devices[current_index]["name"],
        max_candidates=4,
    )
    assert [route.device_index for route in routes] == (
        [1, 5, 6, 7] if current_index == 1 else [5, 6, 7, 8]
    )
    assert routes[0].current_route and sum(route.current_route for route in routes) == 1
    assert all(route.input_host_api == "Windows WASAPI" for route in routes)
    unavailable = enumerate_microphone_diagnostic_candidates(
        input_host_api="missing", input_device="missing", max_candidates=4
    )
    assert [route.device_index for route in unavailable] == [4, 6, 7, 8]
    assert not any(route.current_route for route in unavailable)


@pytest.mark.parametrize(
    "driver",
    (
        "PicoStreamingMicrophone",
        "Pico Streaming Virtual Audio",
        "Pico Streaming Audio Microphone Array - Front",
        "Pico Streaming Audio Wave Microphone Array - Front",
    ),
)
def test_enumerated_pico_recording_driver_names_are_prioritized_without_brand_matching(
    monkeypatch, driver
):
    devices = [
        {"name": "PICO Camera Audio", "hostapi": 0, "max_input_channels": 1},
        {"name": f"Microphone ({driver})", "hostapi": 0, "max_input_channels": 1},
    ]
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_hostapis=lambda: [{"name": "Windows WASAPI"}],
            query_devices=lambda: devices,
        ),
    )
    routes = enumerate_microphone_diagnostic_candidates(
        input_host_api="missing", input_device="missing", max_candidates=1
    )
    assert len(routes) == 1 and routes[0].device_index == 1


@pytest.mark.asyncio
async def test_silence_ties_preserve_current_route_without_claiming_speech():
    harness = Harness()
    sources = {index: FeedSource() for index in (0, 1)}
    for source in sources.values():
        source.feed(0.0, chunks=4)
    result = await diagnose(
        harness,
        (candidate(0, current=True), candidate(1)),
        {index: Opener(source, {}) for index, source in sources.items()},
    )
    assert result["diagnosis"]["selected_candidate_id"] == "input-0"
    assert result["diagnosis"]["selection_basis"] == "frames"
    assert all(item["silence_observed"] for item in result["diagnosis"]["candidates"])
    assert not any(item["speech_detected"] for item in result["diagnosis"]["candidates"])
    assert result["settings_changed"] is False


@pytest.mark.asyncio
async def test_failed_speech_candidate_cannot_outvote_a_healthy_silent_route():
    harness = Harness()

    class FailingSource(FeedSource):
        async def frames(self):
            yield await self.queue.get()
            raise OSError("controlled read failure")

    failed = FailingSource()
    failed.feed(0.5, chunks=4)
    silent = FeedSource()
    silent.feed(0.0, chunks=4)
    result = await diagnose(
        harness,
        (candidate(0, current=True), candidate(1)),
        {0: Opener(failed, {}), 1: Opener(silent, {})},
    )
    measured = result["diagnosis"]["candidates"]
    assert measured[0]["speech_detected"] and measured[0]["failure_code"] == "capture_failed"
    assert measured[1]["silence_observed"] and measured[1]["failure_code"] is None
    assert result["diagnosis"]["selected_candidate_id"] == "input-1"
    assert result["diagnosis"]["selection_basis"] == "frames"
    assert result["diagnosis"]["cleanup_complete"] and failed.closed and silent.closed


def test_device_labels_are_bounded_redacted_single_line_metadata():
    label = safe_audio_device_label("Desk Mic\n\x1b token=private-credential-value " + "x" * 300)
    assert "\n" not in label and "\x1b" not in label
    assert "private-credential-value" not in label
    assert len(label) <= 160


@pytest.mark.asyncio
async def test_cancel_during_capture_pause_restores_only_already_paused_intent():
    harness = Harness()
    paused = asyncio.Event()
    release = asyncio.Event()
    original = harness.control.set_enabled

    async def set_enabled(channel, enabled):
        await original(channel, enabled)
        if channel == "self" and not enabled:
            paused.set()
            await release.wait()

    harness.control.set_enabled = set_enabled
    task = asyncio.create_task(diagnose(harness, (candidate(0, current=True),), {}))
    await paused.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert harness.control.enabled["self"] is False
    release.set()
    result = await task
    assert result["status"] == "cancelled"
    assert result["diagnosis"]["cleanup_complete"]
    assert result["diagnosis"]["candidates"] == []
    assert result["diagnosis"]["restoration"] == {"self": "restored", "peer": "not_needed"}
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
async def test_shutdown_releases_diagnostic_source_without_restoring_capture():
    harness = Harness()
    source = FeedSource()
    task = asyncio.create_task(
        diagnose(harness, (candidate(0, current=True),), {0: Opener(source, {})})
    )
    await _until(lambda: harness.owner.runtime_for("self") is not None)
    await _until(lambda: harness.owner.runtime_for("self").source is source)
    await harness.owner.close()
    result = await task
    assert result["status"] == "cancelled"
    assert result["diagnosis"]["cleanup_complete"]
    assert result["diagnosis"]["failure_code"] is None
    assert result["diagnosis"]["restoration"] == {"self": "skipped", "peer": "skipped"}
    assert harness.control.enabled == {"self": False, "peer": False}
    assert source.closed and not harness.owner.active


@pytest.mark.asyncio
async def test_no_actual_candidates_never_pauses_capture_or_opens_a_source():
    harness = Harness()
    result = await diagnose(harness, (), {})
    assert result["status"] == "action_required"
    assert result["diagnosis"]["failure_code"] == "no_input_candidates"
    assert result["diagnosis"]["candidates"] == []
    assert harness.control.calls == []
    assert not harness.owner.active


@pytest.mark.asyncio
async def test_vad_failure_remains_local_and_restores_capture_without_opening_microphone():
    harness = Harness()
    harness.probe_error = RuntimeError("private model path")
    result = await diagnose(
        harness,
        (candidate(0, current=True),),
        {0: Opener(AssertionError("Microphone must not open without local VAD"), {})},
    )
    assert result["status"] == "action_required"
    assert result["diagnosis"]["candidates"][0]["failure_code"] == "vad_unavailable"
    assert result["diagnosis"]["candidates"][0]["opened"] is False
    assert result["diagnosis"]["restoration"] == {"self": "restored", "peer": "restored"}
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
async def test_pause_failure_keeps_original_failure_and_capture_cleanup_debt():
    harness = Harness()
    harness.control.stuck_open.add("peer")
    result = await diagnose(harness, (candidate(0, current=True),), {})
    assert result["status"] == "failed"
    assert result["diagnosis"]["failure_code"] == "capture_pause_failed"
    assert result["diagnosis"]["cleanup_complete"] is False
    assert result["diagnosis"]["candidates"] == []
    assert result["diagnosis"]["restoration"] == {"self": "restored", "peer": "cleanup_pending"}
    assert harness.owner.snapshot["restore_capture"] == ["peer"]
    harness.control.stuck_open.clear()
    await harness.owner.stop()
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
async def test_real_native_partial_open_failure_cannot_hide_cleanup_or_restore_capture(
    monkeypatch,
):
    streams = []

    class Stream:
        samplerate = 48000

        def __init__(self, **_kwargs):
            self.release_allowed = False
            streams.append(self)

        def start(self):
            raise RuntimeError("private primary native open failure")

        def stop(self):
            pass

        def close(self):
            if not self.release_allowed:
                raise OSError("private secondary native close failure")

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=Stream))
    harness = Harness()
    opener = replace(
        _microphone(_route(), SoundDeviceAudioSource),
        expected_device_name="Desk Mic",
        allow_channel_fallback=False,
    )
    route = replace(candidate(7, current=True), input_device="Desk Mic")
    result = await harness.owner.diagnose(
        candidates=(route,),
        plan_factory=lambda _route: replace(harness.plan("self"), opener=opener),
        input_host_api="Windows WASAPI",
        input_device="Desk Mic",
    )
    diagnosis = result["diagnosis"]
    assert result["status"] == "failed" and not diagnosis["cleanup_complete"]
    assert diagnosis["candidates"][0]["failure_code"] == "source_open_failed"
    assert diagnosis["candidates"][0]["cleanup_complete"] is False
    assert len(streams) == 1
    assert harness.owner.runtime_for("self").source is not None
    assert harness.control.enabled == {"self": False, "peer": False}
    assert diagnosis["restoration"] == {"self": "cleanup_pending", "peer": "cleanup_pending"}
    assert "private" not in json.dumps(result)
    streams[0].release_allowed = True
    await harness.owner.stop()
    assert not harness.owner.active
    assert harness.control.enabled == {"self": True, "peer": True}


@pytest.mark.asyncio
async def test_preexisting_native_capture_debt_is_not_reported_as_completed_cleanup():
    harness = Harness(self_enabled=False, peer_enabled=False)
    original = harness.control.state
    failure = OSError("retained native capture release failure")

    def state(channel):
        current = original(channel)
        if channel == "self":
            return replace(current, busy=True, source_open=True, close_exception=failure)
        return current

    harness.control.state = state
    result = await diagnose(harness, (candidate(0, current=True),), {})
    assert result["status"] == "failed"
    assert result["diagnosis"]["failure_code"] == "resource_release_incomplete"
    assert result["diagnosis"]["cleanup_complete"] is False
    assert result["diagnosis"]["restoration"] == {"self": "cleanup_pending", "peer": "not_needed"}
    assert result["diagnosis"]["candidates"] == []
    assert harness.control.calls == []
    assert harness.control.state("self").close_exception is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
async def test_blocked_native_capture_pause_is_bounded_owned_and_cannot_revive_later_user_off(
    monkeypatch, cancelled
):
    monkeypatch.setattr(audio_test, "DIAGNOSTIC_CLEANUP_TIMEOUT_S", 0.025)
    harness = Harness()
    entered = asyncio.Event()
    release = asyncio.Event()
    original = harness.control.set_enabled

    async def set_enabled(channel, enabled):
        await original(channel, enabled)
        if channel == "self" and not enabled:
            harness.control.stuck_open.add("self")
            entered.set()
            await release.wait()
            harness.control.stuck_open.discard("self")

    harness.control.set_enabled = set_enabled
    task = asyncio.create_task(diagnose(harness, (candidate(0, current=True),), {}))
    await entered.wait()
    if cancelled:
        task.cancel()
    result = await task
    assert result["status"] == ("cancelled" if cancelled else "failed")
    assert result["diagnosis"]["cleanup_complete"] is False
    assert result["diagnosis"]["candidates"] == []
    assert result["diagnosis"]["restoration"]["self"] == "cleanup_pending"
    assert harness.owner.active
    assert await harness.start() is False
    assert harness.control.calls == [("self", False)]
    for channel in ("self", "peer"):
        harness.owner.release_capture_restore(channel)
        harness.control.user_set(channel, False)
    release.set()
    await _until(
        lambda: harness.owner._diagnosis_cleanup_task is None
        or harness.owner._diagnosis_cleanup_task.done()
    )
    await harness.owner.stop()
    assert not harness.owner.active
    assert harness.control.enabled == {"self": False, "peer": False}
    assert all(not enabled for _channel, enabled in harness.control.calls)


def test_auto_route_resolves_native_default_sentinel_to_an_actual_enumerated_candidate(monkeypatch):
    devices = [
        {"name": "Other Mic", "hostapi": 0, "max_input_channels": 1},
        {"name": "Native Default", "hostapi": 1, "max_input_channels": 2},
    ]

    def query_devices(*, kind=None):
        return devices[1] if kind == "input" else devices

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_hostapis=lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}],
            query_devices=query_devices,
            default=SimpleNamespace(device=(-1, -1)),
        ),
    )
    routes = enumerate_microphone_diagnostic_candidates(
        input_host_api="", input_device="", max_candidates=4
    )
    assert routes[0].device_index == 1
    assert routes[0].input_device == "Native Default" and routes[0].current_route
    assert all(route.device_index >= 0 for route in routes)


@pytest.mark.asyncio
async def test_shutdown_does_not_discard_real_native_capture_cleanup_debt():
    harness = Harness(peer_enabled=False)
    entered = asyncio.Event()
    release = asyncio.Event()
    original = harness.control.set_enabled

    async def set_enabled(channel, enabled):
        await original(channel, enabled)
        if channel == "self" and not enabled:
            entered.set()
            await release.wait()
            harness.control.stuck_open.add("self")
            raise OSError("retained native close failure")

    harness.control.set_enabled = set_enabled
    task = asyncio.create_task(diagnose(harness, (candidate(0, current=True),), {}))
    await entered.wait()
    closing = asyncio.create_task(harness.owner.close())
    try:
        await asyncio.sleep(0)
        release.set()
        await closing
        result = await task
        assert result["status"] == "cancelled"
        assert result["diagnosis"]["cleanup_complete"] is False
        assert result["diagnosis"]["failure_code"] == "resource_release_incomplete"
        assert result["diagnosis"]["candidates"] == []
        assert harness.control.state("self").source_open
        assert harness.control.calls == [("self", False)]
        assert harness.control.enabled == {"self": False, "peer": False}
    finally:
        release.set()
        await asyncio.gather(task, closing, return_exceptions=True)


@pytest.mark.asyncio
async def test_optional_opened_metadata_failure_stays_secondary_to_successful_capture():
    from puripuly_heart.app.services.agent_chat.tools import project_operation_receipt

    class Source(FeedSource):
        def opened_input_metadata(self):
            raise OSError(5, "PRIVATE-METADATA-PATH")

    harness = Harness()
    source = Source()
    source.feed(0.5, chunks=4)
    result = await diagnose(
        harness,
        (candidate(0, current=True),),
        {0: _microphone(_route(), lambda **kwargs: source)},
    )
    assert result["status"] == "applied"
    observed = result["diagnosis"]["candidates"][0]
    assert observed["speech_detected"] and observed["cleanup_complete"]
    assert observed["failure_code"] is None and observed["diagnostics"] is None
    assert observed["device_metadata_availability"] == "unavailable"
    (diagnostic,) = observed["secondary_diagnostics"]
    assert diagnostic["code"] == "capture.self.device_metadata_failed"
    assert diagnostic["operation"] == "device_metadata"
    assert diagnostic["fields"]["errno"] == 5
    projected = project_operation_receipt(result, "diagnose_microphone")
    assert (
        projected["diagnosis"]["candidates"][0]["secondary_diagnostics"][0]["code"]
        == diagnostic["code"]
    )
    assert "PRIVATE" not in json.dumps([result, projected])
