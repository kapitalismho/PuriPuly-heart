from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from puripuly_heart.app.services.manual_local_asr_fallback import ManualLocalASRFallbackOwner
from puripuly_heart.composition.application_startup import ApplicationStartupAdapter
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


class _FailingPersistOwner:
    def __init__(self, settings: AppSettingsVNext) -> None:
        self.canonical = settings
        self.authoritative = False

    def remember_projection(self, _settings: AppSettingsVNext) -> None:
        return None

    def save_current(self, failure_sink=None) -> bool:
        if failure_sink is not None:
            failure_sink(RuntimeError("persist failed"))
        return False

    def set_overlay_enabled(self, _enabled: bool) -> None:
        return None

    def set_peer_translation_enabled(self, _enabled: bool) -> None:
        return None


@pytest.mark.asyncio
async def test_startup_restores_pre_fallback_canonical_when_persist_fails() -> None:
    original = replace(
        AppSettingsVNext(),
        intent=replace(
            AppSettingsVNext().intent,
            stt=replace(AppSettingsVNext().intent.stt, provider="local_parakeet_v3"),
            peer_stt=replace(AppSettingsVNext().intent.peer_stt, provider="local_parakeet_ja"),
        ),
    )
    owner = _FailingPersistOwner(original)
    provisioning = Mock()
    provisioning.snapshot = SimpleNamespace(cpu_auto_available=True)
    provisioning.inspect_cpu = AsyncMock()
    provisioning.inspect_gpu = AsyncMock()
    adapter = ApplicationStartupAdapter(
        settings=owner,
        settings_loader=lambda: original,
        provisioning=provisioning,
        gpu_state=lambda: SimpleNamespace(selected_provider_requires_model=False),
        manual_fallback=ManualLocalASRFallbackOwner(),
        save_failure_sink=lambda _exc: None,
        calibration=Mock(),
        presentation=Mock(),
        sync_presentation=lambda: None,
        notify_fallback=lambda *_args: None,
        runtime_logging=Mock(),
        sync_runtime_signatures=lambda _settings: None,
        pipeline_launcher=Mock(),
        pipeline=Mock(),
        sync_local_asr_notice=lambda: None,
        stt_requires_secret=lambda _provider: False,
        llm_requires_secret=lambda _provider: False,
        alibaba_verified_key=lambda: "",
        managed_translation_available=lambda: False,
        receiver_active=lambda: False,
        create_event_bridge=lambda _logging: Mock(),
        start_event_bridge=lambda _bridge: None,
        wait_for_event_bridge=AsyncMock(),
        sync_clipboard=AsyncMock(),
    )

    state = await adapter.prepare_startup_settings()

    assert state.settings is original
    assert state.settings.intent.stt.provider == "local_parakeet_v3"
    assert state.fallback_channels == ()
