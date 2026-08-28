from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.osc.noop_query import NoopOscQueryService
from puripuly_heart.app.services.osc.state_publisher import OscCanonicalState
from puripuly_heart.app.wiring import wiring_application_runtime_logging, wiring_vrc_mic_sync
from puripuly_heart.core.observability import SessionLoggingMode
from puripuly_heart.core.osc.receiver_contract import VrcMicState


class RecordingLoggingAdapter:
    def __init__(self) -> None:
        self.mode = SessionLoggingMode.BASIC
        self.log_file = Path("runtime.log")
        self.basic_messages: list[tuple[int, str]] = []
        self.close_calls = 0

    def set_mode(self, mode: SessionLoggingMode | str) -> None:
        self.mode = SessionLoggingMode(mode)

    def attach_realtime_sink(self, _sink: object) -> None:
        return None

    def detach_realtime_sink(self) -> None:
        return None

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self.basic_messages.append((level, message))

    def emit_detailed(self, _message: str, *, level: int = logging.INFO) -> bool:
        _ = level
        return self.mode is SessionLoggingMode.DETAILED

    def emit_detailed_lazy(self, _build_message, *, level: int = logging.INFO) -> bool:
        _ = level
        return self.mode is SessionLoggingMode.DETAILED

    def emit_persisted(self, _message: str, *, level: int = logging.INFO) -> None:
        _ = level

    def close_terminal_owner(self) -> None:
        self.close_calls += 1


class RecordingGate:
    def set_enabled(self, _enabled: bool) -> None:
        return None

    def set_receiver_active(self, _active: bool) -> None:
        return None

    def reset(self) -> None:
        return None


class RecordingOscReceiver:
    instances: list[RecordingOscReceiver] = []

    def __init__(self, **kwargs: object) -> None:
        self.effective_port = int(kwargs["port"])
        self.start_calls = 0
        self.stop_calls = 0
        type(self).instances.append(self)

    async def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def test_production_logging_composition_constructs_adapter_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters: list[RecordingLoggingAdapter] = []

    def adapter_factory(**_kwargs: object) -> RecordingLoggingAdapter:
        adapter = RecordingLoggingAdapter()
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(
        wiring_application_runtime_logging,
        "SessionRuntimeLoggingService",
        adapter_factory,
    )
    attached: list[object] = []
    owner = wiring_application_runtime_logging.compose_application_runtime_logging(
        presentation=SimpleNamespace(
            attach_runtime_log_sink=lambda service: attached.append(service)
        ),
        sinks=None,
        overlay_logging_mode_update=lambda: _noop(),
        overlay_logging_mode_update_available=lambda: False,
    )

    first = owner.service
    second = owner.service
    first.emit_basic("wired", level=logging.WARNING)
    first.close()

    assert first is second
    assert len(adapters) == 1
    assert adapters[0].basic_messages == [(logging.WARNING, "wired")]
    assert adapters[0].close_calls == 1
    assert attached == [first, first]


@pytest.mark.asyncio
async def test_production_receiver_composition_constructs_adapter_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingOscReceiver.instances.clear()
    state = VrcMicState()
    monkeypatch.setattr(
        wiring_vrc_mic_sync,
        "VrcOscReceiver",
        RecordingOscReceiver,
    )
    integration = wiring_vrc_mic_sync.compose_vrc_mic_sync(
        state_provider=lambda: state,
        gate_provider=RecordingGate,
        log_detailed=lambda _message, _level: None,
        error_sink=lambda _message: None,
        settings_provider=lambda: None,
        apply_settings=_apply_settings,
        application_provider=lambda: None,
        sender_provider=lambda: None,
        osc_state_provider=OscCanonicalState,
        language_state_provider=lambda: ("ko", "en", "en", "ko"),
        translation_model_normalizer=lambda model: model,
        query_service=NoopOscQueryService(),
    )

    await integration.configure(enabled=True)
    await integration.configure(enabled=True)
    await integration.close()

    assert len(RecordingOscReceiver.instances) == 1
    receiver = RecordingOscReceiver.instances[0]
    assert receiver.start_calls == 1
    assert receiver.stop_calls == 1


async def _noop() -> None:
    return None


async def _apply_settings(settings: object) -> object:
    return settings
