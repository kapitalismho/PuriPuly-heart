from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.app.wiring.wiring_managed_gemma import (
    managed_gemma_selection,
    managed_gemma_translation_desired,
    sync_managed_gemma_demand,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
)


def _managed_gemma_settings() -> AppSettingsVNext:
    settings = AppSettingsVNext()
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model=TranslationModel.MANAGED_GEMMA.value,
                connection=TranslationConnection.CPU.value,
            ),
        ),
    )


def test_demand_is_true_when_either_channel_is_on() -> None:
    assert managed_gemma_translation_desired(
        translation_enabled=True,
        peer_translation_enabled=False,
    )
    assert managed_gemma_translation_desired(
        translation_enabled=False,
        peer_translation_enabled=True,
    )
    assert not managed_gemma_translation_desired(
        translation_enabled=False,
        peer_translation_enabled=False,
    )


@pytest.mark.asyncio
async def test_sync_prepares_when_demand_is_on() -> None:
    events: list[object] = []
    settings = _managed_gemma_settings()

    class Owner:
        async def prepare(self, selection: ManagedGemmaTranslationSelection) -> object:
            events.append(("prepare", selection))
            return object()

        async def deactivate(self, *, linger: bool = False) -> None:
            events.append(("deactivate", linger))

    await sync_managed_gemma_demand(
        managed_gemma=Owner(),
        settings=settings,
        desired=True,
    )

    assert events == [("prepare", managed_gemma_selection(settings))]


@pytest.mark.asyncio
async def test_sync_lingers_when_demand_is_off() -> None:
    events: list[object] = []

    class Owner:
        async def prepare(self, _selection: ManagedGemmaTranslationSelection) -> object:
            events.append("prepare")
            return object()

        async def deactivate(self, *, linger: bool = False) -> None:
            events.append(("deactivate", linger))

    await sync_managed_gemma_demand(
        managed_gemma=Owner(),
        settings=SimpleNamespace(),
        desired=False,
    )

    assert events == [("deactivate", True)]
