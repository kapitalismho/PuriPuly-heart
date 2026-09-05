from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from puripuly_heart.app.adapters.ui_runtime import UiProviderRuntimeAdapter
from puripuly_heart.app.ports.ui_models import GpuDeviceOption
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel
from puripuly_heart.core.local_translation.devices import LlamaVulkanDevice


def _custom_http_settings() -> AppSettingsVNext:
    settings = AppSettingsVNext()
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model=TranslationModel.CUSTOM_HTTP.value,
                connection=TranslationConnection.CUSTOM_HTTP.value,
                http_extension_id="demo",
            ),
        ),
    )


def _adapter(
    settings: AppSettingsVNext,
    *,
    change_secret: AsyncMock,
    apply: AsyncMock,
    managed_gemma: object | None = None,
) -> UiProviderRuntimeAdapter:
    return UiProviderRuntimeAdapter(
        settings=SimpleNamespace(canonical=settings),
        provider_application=SimpleNamespace(apply=apply),
        gpu=object(),
        managed=object(),
        credential_verification=object(),
        provider_settings=SimpleNamespace(change_secret=change_secret),
        build_byok_target_settings=lambda _settings: None,
        managed_gemma=managed_gemma,
    )


@pytest.mark.asyncio
async def test_active_custom_http_secret_change_rebuilds_runtime_backend() -> None:
    settings = _custom_http_settings()
    change_secret = AsyncMock(return_value=True)
    apply = AsyncMock(return_value=True)
    adapter = _adapter(settings, change_secret=change_secret, apply=apply)

    assert await adapter.persist_provider_secret_change(
        "http_extension.demo.api_key",
        "new-secret",
    )

    change_secret.assert_awaited_once_with(
        "http_extension.demo.api_key",
        "new-secret",
    )
    apply.assert_awaited_once_with(
        None,
        force_rebuild_llm=True,
        persist_settings=False,
        refresh_ui=True,
    )


@pytest.mark.asyncio
async def test_rolling_member_secret_change_rebinds_without_provider_apply() -> None:
    settings = AppSettingsVNext()
    change_secret = AsyncMock(return_value=True)
    apply = AsyncMock(return_value=True)
    rebound: list[tuple[str, str]] = []
    adapter = _adapter(settings, change_secret=change_secret, apply=apply)
    adapter.provider_application.rebind_rolling_stt_secret = (
        lambda key, value: rebound.append((key, value))
    )

    assert await adapter.persist_provider_secret_change(
        "gemini_transcribe_api_key",
        "rotated-key",
    )

    assert rebound == [("gemini_transcribe_api_key", "rotated-key")]
    apply.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_secret_change_does_not_rebind_rolling() -> None:
    settings = AppSettingsVNext()
    change_secret = AsyncMock(return_value=False)
    apply = AsyncMock(return_value=True)
    rebound: list[tuple[str, str]] = []
    adapter = _adapter(settings, change_secret=change_secret, apply=apply)
    adapter.provider_application.rebind_rolling_stt_secret = (
        lambda key, value: rebound.append((key, value))
    )

    assert await adapter.persist_provider_secret_change(
        "gemini_transcribe_api_key",
        "rotated-key",
    ) is False

    assert rebound == []
    apply.assert_not_awaited()


@pytest.mark.asyncio
async def test_inactive_custom_http_secret_change_does_not_rebuild_active_runtime() -> None:
    settings = _custom_http_settings()
    change_secret = AsyncMock(return_value=True)
    apply = AsyncMock(return_value=True)
    adapter = _adapter(settings, change_secret=change_secret, apply=apply)

    assert await adapter.persist_provider_secret_change(
        "http_extension.other.api_key",
        "new-secret",
    )

    apply.assert_not_awaited()


@pytest.mark.asyncio
async def test_managed_gemma_notice_cancel_targets_owned_prepare() -> None:
    settings = AppSettingsVNext()
    cancel_calls: list[bool] = []
    adapter = _adapter(
        settings,
        change_secret=AsyncMock(),
        apply=AsyncMock(),
        managed_gemma=SimpleNamespace(cancel=lambda: cancel_calls.append(True) or True),
    )

    assert await adapter.handle_managed_gemma_notice_action("cancel") is True
    assert cancel_calls == [True]


@pytest.mark.asyncio
async def test_gpu_discovery_publishes_llama_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[tuple[GpuDeviceOption, ...]] = []
    adapter = _adapter(
        AppSettingsVNext(),
        change_secret=AsyncMock(),
        apply=AsyncMock(),
    )
    adapter.gpu = SimpleNamespace(ensure_device_discovery=AsyncMock())
    adapter.llm_devices_sink = published.append
    monkeypatch.setattr(
        "puripuly_heart.app.adapters.ui_runtime.list_llama_vulkan_devices",
        lambda: (LlamaVulkanDevice("Vulkan1", "AMD Radeon Graphics"),),
    )

    await adapter.ensure_gpu_device_discovery()

    adapter.gpu.ensure_device_discovery.assert_awaited_once_with(
        force=False,
        origin="settings",
    )
    assert published == [(GpuDeviceOption("Vulkan1", "AMD Radeon Graphics", "Vulkan1"),)]
