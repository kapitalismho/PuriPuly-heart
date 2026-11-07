from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from puripuly_heart.app.adapters.ui_runtime import UiProviderRuntimeAdapter
from puripuly_heart.app.ports.settings_view import OpenRouterPkceTarget, TranslationSelectionEdit
from puripuly_heart.config.provider_values import OpenRouterSelectionAlias
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel


def _adapter(canonical, projected):
    projection = Mock(return_value=projected)
    return (
        UiProviderRuntimeAdapter(
            settings=SimpleNamespace(canonical=canonical),
            provider_application=None,
            gpu=None,
            managed=None,
            credential_verification=None,
            provider_settings=None,
            build_byok_target_settings=projection,
        ),
        projection,
    )


def test_ui_byok_target_projection_returns_only_focused_pkce_values() -> None:
    canonical = AppSettingsVNext()
    projected = replace(
        canonical,
        intent=replace(
            canonical.intent,
            translation=replace(
                canonical.intent.translation,
                model="gemma4_26b_31b",
                connection="openrouter",
                openrouter_selected_source="byok",
                openrouter_selection_alias=OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK.value,
            ),
            prompts=replace(
                canonical.intent.prompts,
                system_prompt_override="  Synthetic prompt ${targetName}.\n",
            ),
        ),
    )
    adapter, projection = _adapter(canonical, projected)

    target = adapter.build_managed_openrouter_byok_target()

    projection.assert_called_once_with(canonical)
    assert isinstance(target, OpenRouterPkceTarget)
    assert target.selection_alias is OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK
    assert target.system_prompt == "  Synthetic prompt ${targetName}.\n"
    (edit,) = target.provider_intent.edits
    assert isinstance(edit, TranslationSelectionEdit)
    assert edit.selection.model is TranslationModel.GEMMA4_26B_31B
    assert edit.selection.connection is TranslationConnection.OPENROUTER
    assert edit.history_updates == (
        (TranslationModel.GEMMA4_26B_31B, TranslationConnection.OPENROUTER),
    )
    assert set(asdict(target)) == {"selection_alias", "provider_intent", "system_prompt"}
    assert set(asdict(edit)) == {"selection", "history_updates"}
    assert canonical.intent.translation.openrouter_selected_source != "byok"


@pytest.mark.parametrize("missing", ["settings", "alias"])
def test_ui_byok_target_projection_does_not_invent_missing_authority(missing) -> None:
    canonical = AppSettingsVNext()
    projected = None
    if missing == "alias":
        projected = replace(
            canonical,
            intent=replace(
                canonical.intent,
                translation=replace(canonical.intent.translation, openrouter_selection_alias=None),
            ),
        )
    adapter, projection = _adapter(canonical, projected)

    assert adapter.build_managed_openrouter_byok_target() is None
    projection.assert_called_once_with(canonical)
