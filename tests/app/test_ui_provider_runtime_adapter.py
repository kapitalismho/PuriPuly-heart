from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from puripuly_heart.app.adapters.ui_runtime import UiProviderRuntimeAdapter
from puripuly_heart.config.settings import (
    AppSettings,
    TranslationConnection,
    TranslationModel,
    TranslationSettings,
)


def _adapter(
    settings: AppSettings,
    *,
    change_secret: AsyncMock,
    apply: AsyncMock,
) -> UiProviderRuntimeAdapter:
    return UiProviderRuntimeAdapter(
        settings=SimpleNamespace(current=settings),
        provider_application=SimpleNamespace(apply=apply),
        gpu=object(),
        managed=object(),
        credential_verification=object(),
        provider_settings=SimpleNamespace(change_secret=change_secret),
        build_byok_target_settings=lambda _settings: None,
    )


@pytest.mark.asyncio
async def test_active_custom_http_secret_change_rebuilds_runtime_backend() -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.CUSTOM_HTTP,
        connection=TranslationConnection.CUSTOM_HTTP,
        extension_id="demo",
    )
    change_secret = AsyncMock(return_value=True)
    apply = AsyncMock(return_value=True)
    adapter = _adapter(settings, change_secret=change_secret, apply=apply)

    assert await adapter.persist_provider_secret_change(
        "translation_extension.demo.api_key",
        "new-secret",
    )

    change_secret.assert_awaited_once_with(
        "translation_extension.demo.api_key",
        "new-secret",
    )
    apply.assert_awaited_once_with(
        None,
        force_rebuild_llm=True,
        persist_settings=False,
    )


@pytest.mark.asyncio
async def test_inactive_custom_http_secret_change_does_not_rebuild_active_runtime() -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.CUSTOM_HTTP,
        connection=TranslationConnection.CUSTOM_HTTP,
        extension_id="demo",
    )
    change_secret = AsyncMock(return_value=True)
    apply = AsyncMock(return_value=True)
    adapter = _adapter(settings, change_secret=change_secret, apply=apply)

    assert await adapter.persist_provider_secret_change(
        "translation_extension.other.api_key",
        "new-secret",
    )

    apply.assert_not_awaited()
