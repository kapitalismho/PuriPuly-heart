from __future__ import annotations

import copy

import pytest

from puripuly_heart.app.ports.settings_runtime_effects import SettingsRuntimeTransition
from puripuly_heart.app.services.settings.settings_runtime_effects import (
    managed_gemma_prefix_refresh_required,
    refresh_managed_gemma_prefix,
)
from puripuly_heart.config.settings import AppSettings, TranslationModel


def _transition(
    previous: AppSettings | None,
    current: AppSettings,
    *,
    source_changed: bool = False,
    target_changed: bool = False,
) -> SettingsRuntimeTransition[AppSettings]:
    return SettingsRuntimeTransition(
        settings=current,
        previous_settings=previous,
        previous_locale="en",
        previous_overlay_enabled=False,
        previous_self_signature=None,
        previous_peer_signature=None,
        previous_peer_translation_enabled=False,
        previous_peer_activation_requested=False,
        source_language_changed=source_changed,
        target_language_changed=target_changed,
        effective_peer_source_changed=False,
        effective_peer_target_changed=False,
        peer_source_language_changed=False,
        peer_target_language_changed=False,
        peer_source_mode_changed=False,
        desktop_runtime_controls=(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source", "target", "prompt"])
async def test_managed_gemma_language_or_prompt_change_rebuilds_prefix(change: str) -> None:
    previous = AppSettings()
    previous.translation.model = TranslationModel.MANAGED_GEMMA
    current = copy.deepcopy(previous)
    if change == "source":
        current.languages.source_language = "ja"
    elif change == "target":
        current.languages.target_language = "ko"
    else:
        current.system_prompt = "changed prompt"
    transition = _transition(
        previous,
        current,
        source_changed=change == "source",
        target_changed=change == "target",
    )
    rebuilds = 0

    async def rebuild() -> bool:
        nonlocal rebuilds
        rebuilds += 1
        return True

    assert managed_gemma_prefix_refresh_required(transition) is True
    await refresh_managed_gemma_prefix(transition, rebuild=rebuild)
    assert rebuilds == 1


@pytest.mark.asyncio
async def test_non_managed_prompt_change_does_not_rebuild_managed_prefix() -> None:
    previous = AppSettings()
    current = copy.deepcopy(previous)
    current.system_prompt = "changed prompt"
    transition = _transition(previous, current)
    rebuilds = 0

    async def rebuild() -> bool:
        nonlocal rebuilds
        rebuilds += 1
        return True

    assert managed_gemma_prefix_refresh_required(transition) is False
    await refresh_managed_gemma_prefix(transition, rebuild=rebuild)
    assert rebuilds == 0


@pytest.mark.asyncio
async def test_failed_managed_prefix_rebuild_reports_runtime_apply_failure() -> None:
    previous = AppSettings()
    previous.translation.model = TranslationModel.MANAGED_GEMMA
    current = copy.deepcopy(previous)
    current.languages.target_language = "ja"
    transition = _transition(previous, current, target_changed=True)

    async def rebuild() -> bool:
        return False

    with pytest.raises(RuntimeError, match="prefix rebuild failed"):
        await refresh_managed_gemma_prefix(transition, rebuild=rebuild)
