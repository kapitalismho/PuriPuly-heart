from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from puripuly_heart.core.osc.receiver import VrcMicState
from puripuly_heart.core.runtime import OscReceiverRuntime, VrcMicReceiverRuntime
from tests.helpers.lifecycle import assert_lifecycle_structure


async def _flush_loop() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


class FakeTransport:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1


class FakeOscReceiver:
    def __init__(self) -> None:
        self.transport = FakeTransport()
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1
        self.transport.close()


def test_receiver_runtimes_expose_lifecycle_inventory() -> None:
    osc_runtime = OscReceiverRuntime(receiver_factory=FakeOscReceiver)
    vrc_runtime = VrcMicReceiverRuntime(state=VrcMicState())

    osc_snapshot = osc_runtime.lifecycle_owner_snapshot()
    vrc_snapshot = vrc_runtime.lifecycle_owner_snapshot()

    assert_lifecycle_structure(osc_snapshot)
    assert osc_snapshot["owner"] == "OscReceiverRuntime"
    assert "receiver" in osc_snapshot["resource_fields"]

    assert_lifecycle_structure(vrc_snapshot)
    assert vrc_snapshot["owner"] == "VrcMicReceiverRuntime"


@pytest.mark.asyncio
async def test_osc_receiver_runtime_start_stop_close_socket_and_is_idempotent() -> None:
    receiver = FakeOscReceiver()
    runtime = OscReceiverRuntime(receiver_factory=lambda: receiver)

    started = await runtime.start()
    await runtime.stop()
    await runtime.close()

    assert started is receiver
    assert receiver.start_calls == 1
    assert receiver.stop_calls == 1
    assert receiver.transport.closed is True
    assert runtime.receiver is None
    assert runtime.is_closed is True

    await runtime.close()
    assert receiver.stop_calls == 1


@pytest.mark.asyncio
async def test_vrc_mic_receiver_runtime_cancels_and_gathers_delayed_mute_on_stop() -> None:
    state = VrcMicState()
    receiver = FakeOscReceiver()
    runtime = VrcMicReceiverRuntime(
        state=state,
        receiver_factory=lambda **_kwargs: receiver,
        mute_delay_s=999.0,
        cancel_timeout_s=0.01,
    )
    await runtime.start()

    assert runtime.handle_mute_packet(True) is True
    mute_task = runtime.mute_task
    assert mute_task is not None
    await _flush_loop()

    await runtime.stop()

    assert receiver.stop_calls == 1
    assert runtime.receiver is None
    assert runtime.mute_task is None
    assert mute_task.done() is True
    assert state.muted is None


@pytest.mark.asyncio
async def test_vrc_mic_receiver_runtime_terminal_close_timeout_keeps_mute_task_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VrcMicState()
    receiver = FakeOscReceiver()
    runtime = VrcMicReceiverRuntime(
        state=state,
        receiver_factory=lambda **_kwargs: receiver,
        mute_delay_s=0.0,
        cancel_timeout_s=0.01,
    )
    await runtime.start()
    started = asyncio.Event()
    release = asyncio.Event()
    cancellations: list[str] = []

    async def suppress_cancellation(_is_muted: bool, _generation: int) -> None:
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellations.append("cancelled")

    monkeypatch.setattr(runtime, "_apply_mute_state", suppress_cancellation)

    assert runtime.handle_mute_packet(True) is True
    task = runtime.mute_task
    assert task is not None
    await started.wait()

    try:
        with pytest.raises(TimeoutError, match="mute"):
            await asyncio.wait_for(runtime.close(), timeout=0.2)

        assert receiver.stop_calls == 1
        assert cancellations == ["cancelled"]
        assert task.done() is False
        assert task in runtime._mute_tasks

        with pytest.raises(TimeoutError, match="mute"):
            await asyncio.wait_for(runtime.close(), timeout=0.2)

        assert task in runtime._mute_tasks
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=0.2)
        await _flush_loop()

    assert task not in runtime._mute_tasks


@pytest.mark.asyncio
async def test_vrc_mic_receiver_runtime_state_callback_error_does_not_abort_close_cleanup() -> None:
    receiver = FakeOscReceiver()
    callback_calls = 0

    def state_changed(_runtime: object) -> None:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls > 2:
            raise RuntimeError("state callback failed")

    runtime = VrcMicReceiverRuntime(
        state=VrcMicState(),
        receiver_factory=lambda **_kwargs: receiver,
        state_changed=state_changed,
    )
    await runtime.start()

    try:
        await runtime.close()
    finally:
        if receiver.stop_calls == 0:
            receiver.stop()

    assert receiver.stop_calls == 1
    assert runtime.receiver is None


@pytest.mark.asyncio
async def test_vrc_mic_receiver_runtime_drops_late_packets_after_stop_and_generation_change() -> (
    None
):
    state = VrcMicState()
    callbacks: list[Callable[[bool], bool]] = []

    def receiver_factory(**kwargs):  # noqa: ANN003, ANN202
        callbacks.append(kwargs["mute_packet_handler"])
        return FakeOscReceiver()

    runtime = VrcMicReceiverRuntime(
        state=state,
        receiver_factory=receiver_factory,
        mute_delay_s=0.0,
    )
    await runtime.start()
    stale_callback = callbacks[0]
    stale_generation = runtime.generation
    await runtime.stop()

    assert stale_callback(False) is False
    assert runtime.handle_mute_packet(False, generation=stale_generation) is False
    await _flush_loop()
    assert state.muted is None

    await runtime.start()
    assert callbacks[1](False) is True
    await _flush_loop()
    assert state.muted is False


@pytest.mark.asyncio
async def test_vrc_mic_receiver_runtime_reports_start_failure_without_leaking_receiver() -> None:
    diagnostics: list[tuple[str, dict[str, object]]] = []

    class FailingReceiver(FakeOscReceiver):
        async def start(self) -> None:
            raise OSError("port busy")

    runtime = VrcMicReceiverRuntime(
        state=VrcMicState(),
        receiver_factory=lambda **_kwargs: FailingReceiver(),
        diagnostics_sink=lambda event, metadata: diagnostics.append((event, dict(metadata))),
    )

    with pytest.raises(OSError, match="port busy"):
        await runtime.start()

    assert runtime.receiver is None
    assert diagnostics == [
        (
            "vrc_mic_receiver_start_failed",
            {"host": "127.0.0.1", "port": 9001, "error_type": "OSError"},
        )
    ]


@pytest.mark.asyncio
async def test_vrc_mic_receiver_runtime_composes_osc_receiver_owner_for_start_stop_close() -> None:
    events: list[object] = []
    receivers: list[FakeOscReceiver] = []
    osc_owners: list[object] = []

    class SpyOscReceiverRuntime:
        def __init__(
            self,
            *,
            receiver_factory,
            diagnostics_sink=None,
            state_changed=None,
        ) -> None:  # noqa: ANN001
            self._receiver_factory = receiver_factory
            self._state_changed = state_changed
            self.receiver = None
            self.close_calls = 0
            self.stop_strict_flags: list[bool] = []
            osc_owners.append(self)

        async def start(self):  # noqa: ANN202
            events.append("osc-start")
            receiver = self._receiver_factory()
            receivers.append(receiver)
            await receiver.start()
            self.receiver = receiver
            if self._state_changed is not None:
                self._state_changed(self)
            return receiver

        async def stop(self, *, strict_runtime_errors: bool = False) -> None:
            events.append(("osc-stop", strict_runtime_errors))
            self.stop_strict_flags.append(strict_runtime_errors)
            if self.receiver is not None:
                self.receiver.stop()
                self.receiver = None
            if self._state_changed is not None:
                self._state_changed(self)

        async def close(self) -> None:
            events.append("osc-close")
            self.close_calls += 1
            if self.receiver is not None:
                self.receiver.stop()
                self.receiver = None
            if self._state_changed is not None:
                self._state_changed(self)

    runtime = VrcMicReceiverRuntime(
        state=VrcMicState(),
        receiver_factory=lambda **_kwargs: FakeOscReceiver(),
        osc_runtime_factory=SpyOscReceiverRuntime,
    )

    first_receiver = await runtime.start()
    await runtime.stop()
    second_receiver = await runtime.start()
    await runtime.close()

    assert first_receiver is receivers[0]
    assert second_receiver is receivers[1]
    assert len(osc_owners) == 1
    assert events == ["osc-start", ("osc-stop", False), "osc-start", "osc-close"]
    assert runtime.receiver is None
