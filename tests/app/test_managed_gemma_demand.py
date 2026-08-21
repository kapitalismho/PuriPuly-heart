from __future__ import annotations

from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.app.wiring.wiring_managed_gemma import (
    managed_gemma_translation_desired,
    sync_managed_gemma_demand,
)
from puripuly_heart.config.prompts import render_translation_prompt_template
from puripuly_heart.config.settings import (
    AppSettings,
    TranslationConnection,
    TranslationModel,
    materialize_translation_settings,
)
from puripuly_heart.core.language import get_llm_language_name


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
    settings = materialize_translation_settings(AppSettings())
    settings.translation.model = TranslationModel.MANAGED_GEMMA
    settings.translation.connection = TranslationConnection.CPU

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

    assert events == [
        (
            "prepare",
            ManagedGemmaTranslationSelection(
                backend="cpu",
                source_language="ko",
                target_language="en",
                system_prompt=render_translation_prompt_template(
                    settings.system_prompt,
                    source_name=get_llm_language_name("ko"),
                    target_name=get_llm_language_name("en"),
                ),
            ),
        )
    ]


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
