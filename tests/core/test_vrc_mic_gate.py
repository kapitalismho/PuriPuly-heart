from __future__ import annotations

import numpy as np
import pytest

from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.osc import receiver as receiver_module
from puripuly_heart.core.osc.receiver import VrcMicState, VrcOscReceiver


def _samples(value: float) -> np.ndarray:
    return np.full((8,), value, dtype=np.float32)


def test_vrc_mic_audio_gate_passes_audio_when_unmuted() -> None:
    state = VrcMicState(muted=False)
    gate = VrcMicAudioGate(state=state, enabled=True)
    gate.set_receiver_active(True)

    out = gate.process_chunk(_samples(1.0))

    assert np.array_equal(out, _samples(1.0))


def test_vrc_mic_audio_gate_mutes_audio_when_muted() -> None:
    state = VrcMicState(muted=True)
    gate = VrcMicAudioGate(state=state, enabled=True)
    gate.set_receiver_active(True)

    out = gate.process_chunk(_samples(1.0))

    assert np.array_equal(out, np.zeros((8,), dtype=np.float32))


def test_vrc_mic_audio_gate_holds_closed_during_initial_sync_grace() -> None:
    now = 100.0

    def monotonic() -> float:
        return now

    state = VrcMicState()
    gate = VrcMicAudioGate(
        state=state,
        enabled=True,
        initial_sync_grace_s=1.0,
        monotonic=monotonic,
    )
    gate.set_receiver_active(True)

    assert np.array_equal(gate.process_chunk(_samples(1.0)), np.zeros((8,), dtype=np.float32))

    now = 101.5
    assert np.array_equal(gate.process_chunk(_samples(1.0)), _samples(1.0))


def test_vrc_mic_audio_gate_resumes_immediately_after_unmute() -> None:
    state = VrcMicState(muted=True)
    gate = VrcMicAudioGate(state=state, enabled=True)
    gate.set_receiver_active(True)

    assert np.array_equal(gate.process_chunk(_samples(1.0)), np.zeros((8,), dtype=np.float32))

    state.update(False)
    assert np.array_equal(gate.process_chunk(_samples(1.0)), _samples(1.0))


def test_vrc_mic_audio_gate_enters_sync_grace_after_receiver_restart() -> None:
    now = 100.0

    def monotonic() -> float:
        return now

    state = VrcMicState(muted=True)
    gate = VrcMicAudioGate(
        state=state,
        enabled=True,
        initial_sync_grace_s=1.0,
        monotonic=monotonic,
    )

    gate.set_receiver_active(True)
    assert np.array_equal(gate.process_chunk(_samples(1.0)), np.zeros((8,), dtype=np.float32))

    gate.set_receiver_active(False)
    state.reset()
    assert np.array_equal(gate.process_chunk(_samples(1.0)), _samples(1.0))

    gate.set_receiver_active(True)
    assert np.array_equal(gate.process_chunk(_samples(1.0)), np.zeros((8,), dtype=np.float32))

    now = 101.5
    assert np.array_equal(gate.process_chunk(_samples(1.0)), _samples(1.0))


def test_vrc_osc_receiver_stop_preserves_last_known_state() -> None:
    state = VrcMicState(muted=True)
    receiver = VrcOscReceiver(state=state)

    receiver.stop()

    assert state.muted is True


@pytest.mark.asyncio
async def test_vrc_osc_receiver_start_resets_last_known_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VrcMicState(muted=True)
    transport_closed: list[bool] = []

    class FakeTransport:
        def close(self) -> None:
            transport_closed.append(True)

    class FakeServer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def create_serve_endpoint(self):
            return FakeTransport(), object()

    monkeypatch.setattr(receiver_module, "AsyncIOOSCUDPServer", FakeServer)

    receiver = VrcOscReceiver(state=state)
    await receiver.start()

    assert state.muted is None

    receiver.stop()
    assert transport_closed == [True]


def test_vrc_mic_state_update_returns_false_when_value_is_unchanged() -> None:
    state = VrcMicState()

    assert state.update(True) is True
    assert state.update(True) is False
    assert state.muted is True


def test_vrc_osc_receiver_mute_handler_ignores_empty_args() -> None:
    state = VrcMicState(muted=False)
    receiver = VrcOscReceiver(state=state)

    receiver.mute_handler(receiver_module.VRC_OSC_MUTE_ADDRESS)

    assert receiver._mute_task is None
    assert state.muted is False


def test_vrc_osc_receiver_stop_cancels_pending_task_and_closes_transport() -> None:
    state = VrcMicState()
    receiver = VrcOscReceiver(state=state)
    events: list[str] = []

    class FakeTask:
        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            events.append("task_cancelled")

    class FakeTransport:
        def close(self) -> None:
            events.append("transport_closed")

    receiver._mute_task = FakeTask()
    receiver.transport = FakeTransport()

    receiver.stop()

    assert events == ["task_cancelled", "transport_closed"]
    assert receiver._mute_task is None
    assert receiver.transport is None


@pytest.mark.asyncio
async def test_vrc_osc_receiver_mute_handler_cancels_previous_pending_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VrcMicState()
    receiver = VrcOscReceiver(state=state)
    events: list[str] = []

    class FakePendingTask:
        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            events.append("previous_cancelled")

    new_task = object()

    class FakeLoop:
        def create_task(self, coro):
            events.append("new_task_created")
            coro.close()
            return new_task

    monkeypatch.setattr(receiver_module.asyncio, "get_running_loop", lambda: FakeLoop())
    receiver._mute_task = FakePendingTask()

    receiver.mute_handler(receiver_module.VRC_OSC_MUTE_ADDRESS, 1)

    assert events == ["previous_cancelled", "new_task_created"]
    assert receiver._mute_task is new_task


@pytest.mark.asyncio
async def test_vrc_osc_receiver_apply_mute_state_delays_only_for_muted_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VrcMicState()
    receiver = VrcOscReceiver(state=state, mute_delay_s=0.4)
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(receiver_module.asyncio, "sleep", fake_sleep)

    await receiver._apply_mute_state(True)
    await receiver._apply_mute_state(False)

    assert sleep_calls == [0.4]
    assert state.muted is False


@pytest.mark.asyncio
async def test_vrc_osc_receiver_start_is_idempotent_when_transport_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VrcMicState(muted=True)
    receiver = VrcOscReceiver(state=state)
    transport = object()
    receiver.transport = transport

    class UnexpectedServer:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("server should not be created")

    monkeypatch.setattr(receiver_module, "AsyncIOOSCUDPServer", UnexpectedServer)

    await receiver.start()

    assert receiver.transport is transport
    assert state.muted is True


@pytest.mark.asyncio
async def test_vrc_osc_receiver_start_reraises_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VrcMicState(muted=True)

    class FailingServer:
        def __init__(self, *_args, **_kwargs) -> None:
            raise OSError("port already in use")

    monkeypatch.setattr(receiver_module, "AsyncIOOSCUDPServer", FailingServer)
    receiver = VrcOscReceiver(state=state)

    with pytest.raises(OSError, match="port already in use"):
        await receiver.start()

    assert state.muted is None
    assert receiver.transport is None
