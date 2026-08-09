from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from puripuly_heart.app.services.vrc_mic_sync import VrcMicSyncOwner
from puripuly_heart.core.osc.receiver import VrcMicState


class RecordingGate:
    def __init__(self) -> None:
        self.enabled: list[bool] = []
        self.active: list[bool] = []
        self.reset_calls = 0

    def set_enabled(self, enabled: bool) -> None:
        self.enabled.append(enabled)

    def set_receiver_active(self, active: bool) -> None:
        self.active.append(active)

    def reset(self) -> None:
        self.reset_calls += 1


class RecordingReceiver:
    def __init__(self, **_kwargs: object) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def _owner(
    *,
    state: VrcMicState | None,
    gate: RecordingGate,
    receiver_factory=RecordingReceiver,
) -> tuple[VrcMicSyncOwner, list[str], list[tuple[str, dict[str, object]]]]:
    errors: list[str] = []
    diagnostics: list[tuple[str, dict[str, object]]] = []

    def diagnostic_sink(event: str, metadata: Mapping[str, object]) -> None:
        diagnostics.append((event, dict(metadata)))

    return (
        VrcMicSyncOwner(
            state_provider=lambda: state,
            gate_provider=lambda: gate,
            receiver_factory=receiver_factory,
            diagnostics_sink=diagnostic_sink,
            error_sink=errors.append,
            host="127.0.0.1",
            port=9001,
        ),
        errors,
        diagnostics,
    )


@pytest.mark.asyncio
async def test_owner_configures_starts_and_stops_receiver_with_gate() -> None:
    gate = RecordingGate()
    owner, errors, _ = _owner(state=VrcMicState(), gate=gate)

    await owner.configure(enabled=True)
    receiver = owner.receiver
    await owner.configure(enabled=False)

    assert isinstance(receiver, RecordingReceiver)
    assert receiver.started is True
    assert receiver.stopped is True
    assert owner.receiver is None
    assert owner.last_enabled is False
    assert gate.enabled == [True, False]
    assert gate.active == [True, False]
    assert gate.reset_calls == 1
    assert errors == []


@pytest.mark.asyncio
async def test_owner_keeps_legacy_mute_receiver_when_control_mode_is_off() -> None:
    gate = RecordingGate()
    owner, errors, _ = _owner(state=VrcMicState(), gate=gate)

    await owner.configure_control(
        active=False,
        host="127.0.0.1",
        port=9001,
        force_restart=True,
    )
    await owner.configure(enabled=True)

    assert isinstance(owner.receiver, RecordingReceiver)
    assert owner.receiver.started is True
    assert owner.control_active is False
    assert errors == []

    await owner.close()


@pytest.mark.asyncio
async def test_owner_skips_receiver_without_state_and_marks_gate_inactive() -> None:
    gate = RecordingGate()
    factory_calls = 0

    def receiver_factory(**kwargs: object) -> RecordingReceiver:
        nonlocal factory_calls
        factory_calls += 1
        return RecordingReceiver(**kwargs)

    owner, _, _ = _owner(
        state=None,
        gate=gate,
        receiver_factory=receiver_factory,
    )

    await owner.configure(enabled=True)

    assert factory_calls == 0
    assert owner.runtime is None
    assert gate.enabled == [True]
    assert gate.active == [False]


@pytest.mark.asyncio
async def test_owner_contains_receiver_start_oserror_and_reports_endpoint() -> None:
    gate = RecordingGate()

    class FailingReceiver(RecordingReceiver):
        async def start(self) -> None:
            raise OSError("busy")

    owner, errors, diagnostics = _owner(
        state=VrcMicState(),
        gate=gate,
        receiver_factory=FailingReceiver,
    )

    await owner.configure(enabled=True)

    assert owner.receiver is None
    assert gate.active == [False]
    assert len(errors) == 1
    assert "127.0.0.1:9001" in errors[0]
    assert "busy" in errors[0]
    assert diagnostics[0][0] == "vrc_mic_receiver_start_failed"


@pytest.mark.asyncio
async def test_owner_close_preserves_failing_legacy_receiver_for_retry() -> None:
    gate = RecordingGate()
    owner, _, _ = _owner(state=None, gate=gate)

    class FailingReceiver:
        def stop(self) -> None:
            raise RuntimeError("stop failed")

    receiver = FailingReceiver()
    owner.receiver = receiver

    with pytest.raises(RuntimeError, match="stop failed"):
        await owner.close()

    assert owner.receiver is receiver
    assert owner.last_enabled is False
    assert gate.enabled == [False]
    assert gate.active == [False]


@pytest.mark.asyncio
async def test_owner_rejects_runtime_resurrection_after_close() -> None:
    gate = RecordingGate()
    factory_calls = 0

    def receiver_factory(**kwargs: object) -> RecordingReceiver:
        nonlocal factory_calls
        factory_calls += 1
        return RecordingReceiver(**kwargs)

    owner, _, _ = _owner(
        state=VrcMicState(),
        gate=gate,
        receiver_factory=receiver_factory,
    )
    await owner.configure(enabled=True)

    await owner.close()
    await owner.configure(enabled=True)

    assert owner.accepting_ingress is False
    assert owner.runtime is None
    assert owner.receiver is None
    assert factory_calls == 1
    assert gate.enabled == [True, False]
    assert gate.active == [True, False]


@pytest.mark.asyncio
async def test_owner_disposes_inflight_receiver_when_ingress_freezes() -> None:
    gate = RecordingGate()
    start_entered = asyncio.Event()
    finish_start = asyncio.Event()
    receivers: list[RecordingReceiver] = []

    class BlockingReceiver(RecordingReceiver):
        async def start(self) -> None:
            start_entered.set()
            await finish_start.wait()
            self.started = True

    def receiver_factory(**kwargs: object) -> RecordingReceiver:
        receiver = BlockingReceiver(**kwargs)
        receivers.append(receiver)
        return receiver

    owner, _, _ = _owner(
        state=VrcMicState(),
        gate=gate,
        receiver_factory=receiver_factory,
    )
    configure_task = asyncio.create_task(owner.configure(enabled=True))
    await start_entered.wait()

    owner.stop_ingress()
    close_task = asyncio.create_task(owner.close())
    finish_start.set()
    await asyncio.gather(configure_task, close_task)

    assert owner.accepting_ingress is False
    assert owner.runtime is None
    assert owner.receiver is None
    assert len(receivers) == 1
    assert receivers[0].started is True
    assert receivers[0].stopped is True
    assert gate.enabled == [True, False, False]
    assert gate.active == [False, False]
    assert gate.reset_calls == 0
