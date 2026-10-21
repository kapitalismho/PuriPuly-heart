from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

from puripuly_heart.app.ports.settings_view import (
    CustomSttEndpointEdit,
    CustomSttModelEdit,
    SystemPromptEdit,
    TranslationSelectionEdit,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.services.settings.settings_application import (
    materialize_provider_apply_intent,
    settings_view_surface_snapshots,
)
from puripuly_heart.config.provider_values import LLMProviderName, OpenRouterCredentialSource
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel
from puripuly_heart.ui.app import TranslatorApp
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.views import settings as settings_module


def _settings(model: TranslationModel, connection: TranslationConnection) -> AppSettingsVNext:
    settings = AppSettingsVNext()
    return materialize_canonical_translation_settings(
        replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(
                    settings.intent.translation, model=model.value, connection=connection.value
                ),
            ),
        )
    )


def _load(view, settings):
    provider, general, prompt, overlay = settings_view_surface_snapshots(settings)
    view.load_from_settings(
        provider=provider,
        general=general,
        prompt=prompt,
        overlay=overlay,
        config_path=Path("synthetic-settings.json"),
    )


@pytest.fixture
def view(monkeypatch):
    monkeypatch.setattr(settings_module.SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(settings_module.SettingsView, "_refresh_microphones", lambda self: None)
    return settings_module.SettingsView()


def test_index_picker_is_enabled_and_public_connection_needs_no_auth(view, monkeypatch):
    _load(view, _settings(TranslationModel.INDEX_TRANSLATE_35B, TranslationConnection.PUBLIC_API))
    assert view._llm_text.content.value == t("provider.index_translate_35b")
    assert view._translation_connection_text.content.value == t(
        "settings.translation_connection.public_api"
    )
    assert view._translation_connection_row.visible
    assert not view._managed_key_card.visible
    assert not view._managed_trial_usage_bar.visible
    assert not view._openrouter_key.visible
    assert not view._openrouter_pkce_button_row.visible
    assert not view._google_key.visible
    assert not view._openai_key.visible
    assert not view._deepseek_key.visible
    assert not view._chatgpt_account_card.visible
    assert not view._local_llm_connection_card.visible
    assert view._active_prompt_key() == "index_translate"
    captured = []

    class Modal:
        def __init__(self, page, title, options, on_select, **kwargs):
            captured.extend(options)

        def open(self, current):
            assert current == "index_translate_35b"

    monkeypatch.setattr(settings_module, "is_control_mounted", lambda control: True)
    monkeypatch.setattr(settings_module.SettingsView, "page", property(lambda self: None))
    monkeypatch.setattr(settings_module, "SettingsModal", Modal)
    view._on_llm_click(None)
    index = [option for option in captured if option.value == "index_translate_35b"]
    assert len(index) == 1
    assert index[0].label == t("provider.index_translate_35b")
    assert not index[0].disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_reload_pending", [False, True])
@pytest.mark.parametrize("runtime_applied", [True, False])
async def test_index_immediate_apply_commits_only_selection_and_keeps_drafts(
    view, runtime_reload_pending, runtime_applied
):
    canonical = _settings(TranslationModel.GEMMA4_26B_31B, TranslationConnection.MANAGED)
    original_prompt_intent = canonical.intent.prompts
    _load(view, canonical)
    view._http_extension_runtime_reload_pending = runtime_reload_pending
    view._provider_draft = replace(
        view._provider_snapshot, custom_stt_endpoint="https://unapplied.invalid/stt"
    )
    view._record_provider_edit(CustomSttEndpointEdit("https://unapplied.invalid/stt"))
    view._prompt_editor.value = "unapplied prompt"
    view._stage_prompt_draft("unapplied prompt")
    queued = []
    observed = []

    async def apply(intent):
        nonlocal canonical
        assert tuple(type(edit) for edit in intent.edits) == (TranslationSelectionEdit,)
        canonical = materialize_provider_apply_intent(
            canonical, intent, materialize_translation=materialize_canonical_translation_settings
        )
        observed.append(canonical)
        _load(view, canonical)
        return runtime_applied

    app = TranslatorApp.__new__(TranslatorApp)
    app.view_settings = view
    app._queue_settings_mutation_task = queued.append
    app._log_basic = lambda message: None
    app._ui_application = SimpleNamespace(
        apply_provider_intent=apply,
        refresh_settings_projection=lambda: _load(view, canonical),
        compatibility_settings=lambda: canonical,
    )
    view.on_providers_changed = app._on_providers_changed
    view._on_llm_selected("index_translate_35b")
    assert len(queued) == 1
    await queued.pop()()
    assert canonical.intent.translation.model == "index_translate_35b"
    assert canonical.intent.translation.connection == "public_api"
    assert canonical.intent.prompts == original_prompt_intent
    assert view._provider_snapshot.llm_provider == LLMProviderName.INDEX_TRANSLATE
    assert view._provider_snapshot.openrouter_selected_source == OpenRouterCredentialSource.NONE
    assert view._provider_snapshot.openrouter_selection_alias is None
    assert set(view._provider_edits) == {CustomSttEndpointEdit, SystemPromptEdit}
    assert view._prompt_editor.value == "unapplied prompt"
    assert view._provider_draft.custom_stt_endpoint == "https://unapplied.invalid/stt"
    assert view.has_provider_changes
    view._on_llm_selected("gemini_flash")
    assert len(queued) == 1
    await queued.pop()()
    assert canonical.intent.translation.model == "gemini_flash"
    assert len(observed) == 2
    canonical = _settings(TranslationModel.GEMMA4_26B_31B, TranslationConnection.MANAGED)
    _load(view, canonical)
    assert view._llm_text.content.value == t("provider.gemma4_26b_31b")
    assert view._translation_connection_text.content.value == t(
        "settings.translation_connection.managed"
    )
    assert view._provider_draft.translation.model == TranslationModel.GEMMA4_26B_31B
    assert view._prompt_editor.value == "unapplied prompt"
    assert view._provider_draft.custom_stt_endpoint == "https://unapplied.invalid/stt"


@pytest.mark.asyncio
async def test_failed_index_immediate_apply_keeps_committed_selection_and_pending_draft(view):
    _load(view, _settings(TranslationModel.GEMMA4_26B_31B, TranslationConnection.MANAGED))
    queued = []
    applied = []

    async def apply(intent):
        applied.append(intent)
        return False

    app = TranslatorApp.__new__(TranslatorApp)
    app.view_settings = view
    app._queue_settings_mutation_task = queued.append
    app._log_basic = lambda message: None
    app._ui_application = SimpleNamespace(
        apply_provider_intent=apply,
        refresh_settings_projection=lambda: pytest.fail("failed write cannot acknowledge"),
        compatibility_settings=lambda: _settings(
            TranslationModel.GEMMA4_26B_31B, TranslationConnection.MANAGED
        ),
    )
    view.on_providers_changed = app._on_providers_changed
    view._on_llm_selected("index_translate_35b")
    assert len(queued) == 1
    await queued.pop()()
    assert len(applied) == 1
    assert view._provider_snapshot.translation.model == TranslationModel.GEMMA4_26B_31B
    assert view._provider_draft.translation.model == TranslationModel.INDEX_TRANSLATE_35B
    assert TranslationSelectionEdit in view._provider_edits
    assert view.has_provider_changes


@pytest.mark.parametrize(
    "connection", [TranslationConnection.MANAGED, TranslationConnection.OPENROUTER]
)
def test_background_fallback_updates_selection_without_discarding_unapplied_drafts(
    view, connection
):
    _load(view, _settings(TranslationModel.INDEX_TRANSLATE_35B, TranslationConnection.PUBLIC_API))
    view._provider_draft = replace(view._provider_snapshot, custom_stt_model="unapplied-model")
    view._record_provider_edit(CustomSttModelEdit("unapplied-model"))
    view.has_provider_changes = True
    view._prompt_editor.value = "unapplied prompt"
    view._stage_prompt_draft("unapplied prompt")
    view._custom_stt_endpoint.value = "unfinished endpoint"
    view._custom_vocab_tag_editor._input_field.value = "unsubmitted vocabulary"
    _load(view, _settings(TranslationModel.GEMMA4_26B_31B, connection))
    assert view._llm_text.content.value == t("provider.gemma4_26b_31b")
    assert view._translation_connection_text.content.value == t(
        f"settings.translation_connection.{connection.value}"
    )
    assert view._openrouter_key.visible == (connection == TranslationConnection.OPENROUTER)
    assert view._openrouter_pkce_button_row.visible == (
        connection == TranslationConnection.OPENROUTER
    )
    assert view._managed_key_card.visible == (connection == TranslationConnection.MANAGED)
    assert view._provider_draft.custom_stt_model == "unapplied-model"
    assert view._custom_stt_endpoint.value == "unfinished endpoint"
    assert view._custom_vocab_tag_editor._input_field.value == "unsubmitted vocabulary"
    assert view._prompt_editor.value == "unapplied prompt"
    assert view.has_provider_changes
    assert view.has_pending_prompt_changes
    assert view.consume_immediate_provider_apply_settings() is None
