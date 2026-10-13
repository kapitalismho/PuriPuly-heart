import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from puripuly_heart.app.services.microphone_test import MicrophoneTestSessionOwner
from puripuly_heart.app.services.ui_application import UiApplicationBoundary
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.test_recording import TestRecording
from tests.helpers.ui_application import compose_test_ui_application_boundary


def test_raw_pcm_and_normal_processing_are_bounded_and_unamplified():
    recording = TestRecording(seconds=100)
    recording.device = dict(name="actual", host_api="WASAPI")
    samples = np.column_stack(
        (np.full(16000, 0.2, dtype=np.float32), np.full(16000, -0.1, dtype=np.float32))
    )
    frame = AudioFrameF32(samples=samples, channels=2, sample_rate_hz=16000)
    for _ in range(30):
        keep = recording.append(frame)
    assert not keep
    assert recording.duration == 30
    raw, rate = recording.audio()
    processed, target = recording.audio(processed=True)
    assert (rate, target) == (16000, 16000)
    assert raw.shape == (480000, 2)
    assert np.allclose(raw[0], [0.2, -0.1])
    assert np.allclose(processed, 0.05)
    assert recording.snapshot()["actual_device"]["name"] == "actual"
    recording.clear()
    assert not recording.raw_chunks and not recording.processed_chunks
    with pytest.raises(ValueError):
        recording.audio()


def test_format_change_and_no_audio_are_explicit():
    recording = TestRecording()
    recording.append(AudioFrameF32(samples=np.zeros(100, dtype=np.float32), sample_rate_hz=16000))
    with pytest.raises(ValueError, match="capture_format_changed"):
        recording.append(
            AudioFrameF32(samples=np.zeros(100, dtype=np.float32), sample_rate_hz=48000)
        )
    empty = TestRecording()
    empty.finish()
    with pytest.raises(ValueError, match="no_recording"):
        empty.audio()


async def test_playback_failure_logged_and_close_releases_pcm(caplog):
    owner = MicrophoneTestSessionOwner(capture_port=None, capture_request_factory=None)
    owner._recording = TestRecording()
    owner._recording.append(
        AudioFrameF32(samples=np.zeros(160, dtype=np.float32), sample_rate_hz=16000)
    )
    owner._recording.finish()

    async def fail(samples, rate, cancellation):
        assert rate == 16000 and samples.size == 160
        raise OSError(5, "secret-input api-key-must-not-leak")

    with pytest.raises(OSError):
        await owner.play_recording(playback=fail)
    assert "stage=playback code=playback_failed" in caplog.text
    assert "secret-input" not in caplog.text
    assert owner.snapshot["failure_reason"] == "playback_failed"
    await owner.close()
    assert owner._recording is None and owner._playback_task is None


def test_playback_failure_is_not_reported_as_capture_start_failure():
    from puripuly_heart.ui.components.microphone_test_dialog import MicrophoneTestDialog
    from puripuly_heart.ui.i18n import t

    dialog = MicrophoneTestDialog(SimpleNamespace(), on_close=lambda: None)
    dialog.show_failure("playback_failed")
    assert dialog._text_value() == t("audio_test.error.playback_failed")
    assert dialog._text_value() != t("settings.microphone_test.start_failed")
    dialog.set_snapshot(dict(playing=True, recording_complete=True))
    assert dialog._text_value() == "0%"


async def test_stop_playback_waits_for_worker_cleanup():
    owner = MicrophoneTestSessionOwner(capture_port=None, capture_request_factory=None)
    owner._recording = TestRecording()
    owner._recording.append(
        AudioFrameF32(samples=np.zeros(160, dtype=np.float32), sample_rate_hz=16000)
    )
    owner._recording.finish()
    started = asyncio.Event()
    done = asyncio.Event()

    async def controlled(samples, rate, cancellation):
        started.set()
        while not cancellation.is_set():
            await asyncio.sleep(0)
        done.set()

    task = asyncio.create_task(owner.play_recording(playback=controlled))
    await started.wait()
    await owner.stop_playback()
    await task
    assert done.is_set() and owner._playback_task is None


@pytest.mark.parametrize("user_changes", [False, True])
async def test_playback_pauses_capture_and_respects_latest_user_off(user_changes):
    flags = {"self": True, "peer": True}
    generations = {"self": 0, "peer": 0}

    def state(channel):
        return flags[channel], generations[channel]

    async def set_flag(channel, enabled):
        flags[channel] = enabled
        generations[channel] += 1

    async def noop():
        return None

    async def play(*, processed):
        assert not any(flags.values())
        if user_changes:
            await boundary.set_stt_enabled(False)
            await boundary.set_peer_translation_enabled(False)

    backend = SimpleNamespace(
        stop_microphone_test=noop,
        stop_microphone_playback=noop,
        self_capture_test_state=lambda: state("self"),
        peer_capture_test_state=lambda: state("peer"),
        play_microphone_recording=play,
        set_stt_enabled=lambda enabled: set_flag("self", enabled),
        set_peer_translation_enabled=lambda enabled: set_flag("peer", enabled),
    )
    boundary = compose_test_ui_application_boundary(backend)
    await UiApplicationBoundary.play_microphone_recording(boundary)
    assert flags == {"self": not user_changes, "peer": not user_changes}


async def test_rapid_playback_requests_do_not_drop_capture_guard():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def stop():
        await asyncio.sleep(0)

    async def play(*, processed):
        calls.append(processed)
        started.set()
        await release.wait()

    backend = SimpleNamespace(
        stop_microphone_test=stop,
        self_capture_test_state=lambda: (False, 0),
        peer_capture_test_state=lambda: (False, 0),
        play_microphone_recording=play,
    )
    boundary = compose_test_ui_application_boundary(backend)
    first = asyncio.create_task(boundary.play_microphone_recording())
    await started.wait()
    await boundary.play_microphone_recording(processed=True)
    assert boundary._audio_playback_active
    assert calls == [False]
    release.set()
    await first
    assert not boundary._audio_playback_active
