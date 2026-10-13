from __future__ import annotations

import pytest

from puripuly_heart.app.services.osc.noop_query import NoopOscQueryService
from puripuly_heart.app.services.osc.state_publisher import OscCanonicalState
from puripuly_heart.app.wiring import wiring_vrc_mic_sync
from puripuly_heart.core.osc.receiver_contract import VrcMicState


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
        log_diagnostic=lambda _message, _level: None,
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
