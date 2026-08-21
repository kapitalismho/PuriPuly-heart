from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import flet as ft
import pytest

pytest.importorskip("flet")

from puripuly_heart.core.managed_openrouter_release import TalkTogetherPassStatus

from puripuly_heart.config.audio_host_api import WINDOWS_WASAPI_COMPATIBILITY_HOST_API
from puripuly_heart.config.settings import (
    LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS,
    LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS,
    AppSettings,
    DeepSeekLLMModel,
    GeminiLLMModel,
    LLMProviderName,
    LocalLLMBackend,
    LocalLLMSettings,
    OpenRouterCredentialSource,
    OpenRouterFallbackSelectionAlias,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterSelectionAlias,
    ProviderSettings,
    QwenLLMModel,
    QwenRegion,
    STTProviderName,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
    TranslationSettings,
    to_dict,
)
from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.components import subtab_shell as subtab_shell_module
from puripuly_heart.ui.components.bottom_nav import BottomNavBar
from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.gpu_device import GpuDeviceOption
from puripuly_heart.ui.i18n import language_name, provider_label, t
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from puripuly_heart.ui.theme import COLOR_NEUTRAL_DARK
from puripuly_heart.ui.views import settings as settings_view
from tests.helpers.flet_page import attach_dummy_page


class DummySecretStore:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.set_calls: list[tuple[str, str]] = []
        self.delete_calls: list[str] = []

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value
        self.set_calls.append((key, value))

    def delete(self, key: str) -> None:
        self.values.pop(key, None)
        self.delete_calls.append(key)


def _make_settings_view(
    monkeypatch: pytest.MonkeyPatch,
    store: DummySecretStore | None = None,
    settings: AppSettings | None = None,
):
    monkeypatch.setattr(settings_view.SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "_refresh_microphones", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "update", lambda self: None)
    store = store or DummySecretStore()
    monkeypatch.setattr(settings_view, "create_secret_store", lambda *_args, **_kwargs: store)
    view = settings_view.SettingsView()
    if settings is not None:
        view._settings = settings
    return view, store


def test_cpu_auto_option_is_disabled_until_all_models_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _store = _make_settings_view(monkeypatch)

    unavailable = view._stt_option_item(STTProviderName.LOCAL_CPU_AUTO)
    assert unavailable.disabled is True
    assert unavailable.description == t("provider.local_cpu_auto.description")

    view.set_local_cpu_auto_available(True)
    available = view._stt_option_item(STTProviderName.LOCAL_CPU_AUTO)
    assert available.disabled is False
    assert available.description == t("provider.local_cpu_auto.description")
    assert view._peer_stt_option_item(STTProviderName.LOCAL_CPU_AUTO).disabled is False


def test_telemetry_card_loads_state_and_modal_allow_decline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _store = _make_settings_view(monkeypatch)
    page = attach_dummy_page(monkeypatch, view)
    settings = AppSettings()
    calls: list[str] = []
    opened: list[tuple[str, list[object], str]] = []

    class CapturingModal:
        def __init__(self, page_arg, title, options, on_select, *, show_description=False):
            assert page_arg is page
            assert title == t("settings.telemetry.modal.title")
            assert show_description is True
            self.options = options
            self.on_select = on_select

        def open(self, current: str) -> None:
            opened.append((current, list(self.options), current))

    monkeypatch.setattr(settings_view, "SettingsModal", CapturingModal)
    view.on_telemetry_consent_change = calls.append
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._telemetry_consent_text.content.value == t("settings.telemetry.state.on")
    view._on_telemetry_consent_click(None)
    assert opened[0][0] == "allow"
    assert getattr(view, "_telemetry_consent_card") is not None

    captured_select = []

    class SelectingModal(CapturingModal):
        def __init__(self, page_arg, title, options, on_select, *, show_description=False):
            super().__init__(page_arg, title, options, on_select, show_description=show_description)
            captured_select.append(on_select)

    monkeypatch.setattr(settings_view, "SettingsModal", SelectingModal)
    view._on_telemetry_consent_click(None)
    captured_select[-1]("allow")
    assert calls[-1] == "allow"
    assert view._settings.telemetry.consent == "allow"
    assert view._settings.telemetry_state.anonymous_id
    assert view._telemetry_consent_text.content.value == t("settings.telemetry.state.on")
    view._settings.telemetry_state.sent_translation_success_dates_utc = ["2026-07-03"]
    captured_select[-1]("decline")
    assert calls[-1] == "decline"
    assert view._settings.telemetry.consent == "decline"
    assert view._settings.telemetry_state.anonymous_id is None
    assert view._settings.telemetry_state.sent_translation_success_dates_utc == []


def test_telemetry_card_uses_callback_instead_of_send(monkeypatch: pytest.MonkeyPatch) -> None:
    view, _store = _make_settings_view(monkeypatch)
    attach_dummy_page(monkeypatch, view)
    view.load_from_settings(AppSettings(), config_path=Path("settings.json"))
    calls: list[str] = []
    view.on_telemetry_consent_change = calls.append

    monkeypatch.setattr(
        settings_view,
        "SettingsModal",
        lambda _page, _title, _options, on_select, **_kwargs: SimpleNamespace(
            open=lambda _current: on_select("allow")
        ),
    )

    view._on_telemetry_consent_click(None)

    assert calls == ["allow"]
    assert not hasattr(view, "telemetry_client")


def _enabled_fallback(
    model: TranslationModel,
    connection: TranslationConnection,
) -> TranslationFallbackSettings:
    return TranslationFallbackSettings(enabled=True, model=model, connection=connection)


def _make_llm_selection_view(
    monkeypatch: pytest.MonkeyPatch,
    settings: AppSettings,
) -> settings_view.SettingsView:
    monkeypatch.setattr(settings_view.SettingsView, "page", property(lambda self: None))
    view = settings_view.SettingsView.__new__(settings_view.SettingsView)
    view._settings = settings
    view._provider_settings_draft = None
    view._config_path = Path("settings.json")
    view.has_provider_changes = False
    view.has_pending_prompt_changes = False
    view._managed_trial_usage_visible = False
    view._managed_trial_usage_remaining_percent = None
    view._llm_text = SimpleNamespace(content=SimpleNamespace(value=""), update=lambda: None)
    view._translation_connection_text = SimpleNamespace(
        content=SimpleNamespace(value="", size=None),
        update=lambda: None,
    )
    view._openrouter_fallback_text = SimpleNamespace(
        content=SimpleNamespace(value="", size=None),
        update=lambda: None,
    )
    view._openrouter_fallback_helper_text = SimpleNamespace(value="", update=lambda: None)
    view._translation_connection_row = SimpleNamespace(visible=False, update=lambda: None)
    view._openrouter_routing_row = view._translation_connection_row
    view._local_llm_base_url = SimpleNamespace(
        value=settings.local_llm.base_url,
        label="",
        error_text=None,
        update=lambda: None,
    )
    view._local_llm_model = SimpleNamespace(
        value=settings.local_llm.model,
        label="",
        error_text=None,
        update=lambda: None,
    )
    view._local_llm_api_key = SimpleNamespace(
        value="",
        visible=True,
        apply_locale=lambda: None,
        update=lambda: None,
    )
    view._local_llm_api_key_helper = SimpleNamespace(value="", update=lambda: None)
    view._local_llm_extra_body = SimpleNamespace(
        value=json.dumps(settings.local_llm.extra_body, ensure_ascii=False),
        label="",
        error_text=None,
        update=lambda: None,
    )
    view._local_llm_extra_body_helper = SimpleNamespace(value="", update=lambda: None)
    view._local_llm_extra_body_error = SimpleNamespace(
        value="",
        visible=False,
        update=lambda: None,
    )
    view._local_llm_connection_card = SimpleNamespace(visible=False, update=lambda: None)
    view._custom_stt_connection_card = SimpleNamespace(visible=False, update=lambda: None)
    view._managed_trial_usage_bar = SimpleNamespace(
        visible=False,
        percent=None,
        update=lambda: None,
        _remaining_text=SimpleNamespace(update=lambda: None),
        _fill_segments=SimpleNamespace(update=lambda: None),
    )
    view._managed_trial_usage_bar.set_percent = lambda percent: setattr(
        view._managed_trial_usage_bar, "percent", percent
    )
    view._managed_key_card = SimpleNamespace(visible=False, update=lambda: None)
    view._managed_key_title = SimpleNamespace(value="")
    view._managed_key_free_usage_label = SimpleNamespace(value="")
    view._managed_key_referral_id = None
    view._managed_key_referral_id_label = SimpleNamespace(value="")
    view._managed_key_referral_id_value = SimpleNamespace(value="")
    view._managed_key_referral_helper_text = SimpleNamespace(value="")
    view._managed_key_pass_status = None
    view._managed_key_invite_progress_label = SimpleNamespace(value="", update=lambda: None)
    view._managed_key_invite_progress_value = SimpleNamespace(value="", update=lambda: None)
    view._managed_key_invite_progress_row = SimpleNamespace(visible=False, update=lambda: None)
    view._qwen_region_btn = SimpleNamespace(visible=False, update=lambda: None)
    view._api_keys_column = SimpleNamespace(update=lambda: None)
    view._deepgram_key = SimpleNamespace(visible=False)
    view._soniox_key = SimpleNamespace(visible=False)
    view._google_key = SimpleNamespace(visible=False)
    view._openrouter_key = SimpleNamespace(visible=False)
    view._deepseek_key = SimpleNamespace(visible=False)
    view._cerebras_key = SimpleNamespace(visible=False)
    view._openrouter_pkce_button_row = SimpleNamespace(visible=False, update=lambda: None)
    view._openrouter_pkce_button = SimpleNamespace(text="", style=None, update=lambda: None)
    view._alibaba_key_beijing = SimpleNamespace(visible=False)
    view._alibaba_key_singapore = SimpleNamespace(visible=False)
    view._prompt_editor = SimpleNamespace(
        value=settings.system_prompts.get("gemini", settings.system_prompt),
        provider=None,
    )
    view._prompt_for_text = SimpleNamespace(value="")
    view._custom_vocab_description_text = SimpleNamespace(value="", update=lambda: None)
    view._custom_vocab_tag_editor = SimpleNamespace(
        set_placeholder=lambda _text: None,
        set_add_label=lambda _text: None,
        set_empty_text=lambda _text: None,
        set_remove_label_template=lambda _template: None,
    )
    view.on_request_openrouter_pkce = None
    view._prompt_editor.set_provider = lambda provider: setattr(
        view._prompt_editor, "provider", provider
    )
    view._prompt_editor.load_default_prompt = lambda emit_change=False: setattr(
        view._prompt_editor,
        "value",
        "DEFAULT PROMPT",
    )
    view._update_peer_provider_visibility = lambda: None
    return view


def _row_cards(container: ft.Container) -> list[ft.Control]:
    return list(container.content.controls)


def _subtab_controls(view: settings_view.SettingsView, key: str) -> list[ft.Control]:
    return list(view._settings_subtab_shell.body_by_key[key].controls)


def _layout_cards(control: ft.Control) -> list[ft.Control]:
    content = getattr(control, "content", None)
    if isinstance(content, ft.Row):
        return list(content.controls)
    if _card_title(control) is not None:
        return [control]
    return []


def _prompt_tab_cards(view: settings_view.SettingsView) -> list[ft.Control]:
    return list(_subtab_controls(view, "prompt"))


def _overlay_tab_cards(view: settings_view.SettingsView) -> list[ft.Control]:
    cards: list[ft.Control] = []
    for control in _subtab_controls(view, "overlay"):
        if getattr(control, "visible", True) is False:
            continue
        for card in _layout_cards(control):
            try:
                title = _card_title(card)
            except Exception:
                continue
            if title is not None:
                cards.append(card)
    return cards


def _wrapped_card_column(card: ft.Control) -> ft.Control:
    content = card.content.content
    if isinstance(content, ft.Stack):
        for control in content.controls:
            if isinstance(control, ft.TransparentPointer):
                return control.content
    return content


def _wrapped_card_stack(card: ft.Control) -> ft.Stack:
    content = card.content.content
    assert isinstance(content, ft.Stack)
    return content


def _card_title(card: ft.Control) -> str | None:
    column = _wrapped_card_column(card)
    controls = getattr(column, "controls", None)
    if not controls:
        return None
    title = column.controls[0]
    if isinstance(title, ft.Text):
        return title.value
    if isinstance(title, ft.Row):
        for child in title.controls:
            if isinstance(child, ft.Text) and child.value:
                return child.value
    return None


def _general_tab_card_titles(view: settings_view.SettingsView) -> list[str]:
    titles: list[str] = []
    for row in _subtab_controls(view, "general"):
        titles.extend(
            title for card in _layout_cards(row) if (title := _card_title(card)) is not None
        )
    return titles


def _api_tab_card_titles(view: settings_view.SettingsView) -> list[str]:
    titles: list[str] = []
    for row in _subtab_controls(view, "api"):
        titles.extend(
            title for card in _layout_cards(row) if (title := _card_title(card)) is not None
        )
    return titles


def _general_tab_card(view: settings_view.SettingsView, title: str) -> ft.Control:
    for row in _subtab_controls(view, "general"):
        for card in _layout_cards(row):
            if _card_title(card) == title:
                return card
    raise AssertionError(f"General tab card not found: {title}")


def _api_tab_card(view: settings_view.SettingsView, title: str) -> ft.Control:
    for row in _subtab_controls(view, "api"):
        for card in _layout_cards(row):
            if _card_title(card) == title:
                return card
    raise AssertionError(f"API tab card not found: {title}")


def _row_card_titles(control: ft.Control) -> list[str]:
    return [title for card in _layout_cards(control) if (title := _card_title(card)) is not None]


def _prompt_tab_card(view: settings_view.SettingsView, title: str) -> ft.Control:
    for card in _prompt_tab_cards(view):
        if _card_title(card) == title:
            return card
    raise AssertionError(f"prompt tab card not found: {title}")


def _overlay_tab_card_titles(view: settings_view.SettingsView) -> list[str]:
    titles: list[str] = []
    for card in _overlay_tab_cards(view):
        if (title := _card_title(card)) is not None:
            titles.append(title)
    return titles


def _overlay_tab_card(view: settings_view.SettingsView, title: str) -> ft.Control:
    for card in _overlay_tab_cards(view):
        if _card_title(card) == title:
            return card
    raise AssertionError(f"overlay tab card not found: {title}")


def _iter_control_tree(control: ft.Control):
    yield control
    content = getattr(control, "content", None)
    if content is not None:
        yield from _iter_control_tree(content)
    controls = getattr(control, "controls", None) or []
    for child in controls:
        yield from _iter_control_tree(child)


def _control_labels(control: ft.Control) -> list[str]:
    labels: list[str] = []
    for node in _iter_control_tree(control):
        if isinstance(node, ft.Text) and node.value:
            labels.append(node.value)
        elif isinstance(node, ft.TextField) and node.label:
            labels.append(node.label)
        elif isinstance(node, ft.TextButton) and node.content:
            labels.append(node.content)
    return labels


def _tree_contains_control(root: ft.Control, target: ft.Control) -> bool:
    return any(node is target for node in _iter_control_tree(root))


def _control_tooltips(control: ft.Control) -> list[str]:
    return [
        tooltip
        for node in _iter_control_tree(control)
        if (tooltip := getattr(node, "tooltip", None))
    ]


def _custom_vocab_chip_terms(view: settings_view.SettingsView) -> list[str]:
    return [
        str(getattr(chip, "data", ""))
        for chip in view._custom_vocab_tag_editor._chips_wrap.controls  # noqa: SLF001
    ]


def _button_style_value(
    button: ft.TextButton,
    attribute: str,
    state: ft.ControlState = ft.ControlState.DEFAULT,
):
    return getattr(button.style, attribute)[state]


def _subtab_label(button: ft.Control) -> ft.Text:
    if isinstance(button, ft.Container) and isinstance(button.content, ft.Text):
        return button.content
    raise AssertionError(f"Expected subtab container label, got {type(button)!r}")


def _subtab_text_color(button: ft.Control) -> str | None:
    if isinstance(button, ft.TextButton):
        return _button_style_value(button, "color")
    return _subtab_label(button).color


def _container_text_size(control: ft.Container) -> int | None:
    if not isinstance(control.content, ft.Text):
        raise AssertionError(
            f"Expected container-backed text control, got {type(control.content)!r}"
        )
    return control.content.size


def test_load_secret_value_prefers_existing_value() -> None:
    store = DummySecretStore({"new_key": "new", "old_key": "old"})

    value = settings_view._load_secret_value(store, "new_key", legacy_keys=("old_key",))

    assert value == "new"
    assert store.set_calls == []


def test_load_secret_value_migrates_legacy_value() -> None:
    store = DummySecretStore({"old_key": "legacy"})

    value = settings_view._load_secret_value(store, "new_key", legacy_keys=("old_key",))

    assert value == "legacy"
    assert store.set_calls == [("new_key", "legacy")]


def test_peer_language_card_removed_from_general_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    general_titles = _general_tab_card_titles(view)
    api_titles = _api_tab_card_titles(view)
    general_labels: list[str] = []
    api_labels: list[str] = []
    for row in _subtab_controls(view, "general"):
        general_labels.extend(_control_labels(row))
    for row in _subtab_controls(view, "api"):
        api_labels.extend(_control_labels(row))

    assert t("settings.peer_language") not in general_titles
    assert t("settings.section.peer_stt") not in general_titles
    assert t("settings.section.peer_stt") in api_titles
    assert t("settings.peer_language.source") not in general_labels
    assert t("settings.peer_language.target") not in general_labels
    assert t("settings.dashboard_language_redirect") not in general_labels
    assert t("settings.dashboard_language_redirect") not in api_labels
    assert not hasattr(view, "_peer_source_text")
    assert not hasattr(view, "_peer_target_text")


def test_load_from_settings_peer_stt_card_has_no_peer_subsetting_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.QWEN_ASR

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    peer_card = _api_tab_card(view, t("settings.section.peer_stt"))
    labels = _control_labels(peer_card)

    assert t("settings.peer_qwen_region") not in labels
    assert t("settings.peer_qwen_model") not in labels
    assert t("settings.peer_soniox_model") not in labels
    assert not hasattr(view, "_peer_qwen_region_text")
    assert not hasattr(view, "_peer_qwen_model_text")
    assert not hasattr(view, "_peer_soniox_model_text")


def test_clipboard_auto_translate_selection_updates_settings_and_emits_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _store = _make_settings_view(monkeypatch)
    settings = AppSettings()
    emitted: list[AppSettings] = []
    view.on_settings_changed = emitted.append

    view.load_from_settings(settings, config_path=Path("settings.json"))
    view._on_clipboard_auto_translate_selected("on")

    assert settings.ui.clipboard_auto_translate_enabled is True
    assert view._clipboard_auto_translate_text.content.value == t(
        "settings.clipboard_auto_translate.on"
    )
    assert emitted[-1].ui.clipboard_auto_translate_enabled is True


def test_clipboard_auto_translate_click_toggles_immediately_without_modal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _store = _make_settings_view(monkeypatch)
    settings = AppSettings()
    emitted: list[AppSettings] = []
    view.on_settings_changed = emitted.append

    view.load_from_settings(settings, config_path=Path("settings.json"))
    view._on_clipboard_auto_translate_click(None)

    assert settings.ui.clipboard_auto_translate_enabled is True
    assert view._clipboard_auto_translate_text.content.value == t(
        "settings.clipboard_auto_translate.on"
    )
    assert emitted[-1].ui.clipboard_auto_translate_enabled is True

    view._on_clipboard_auto_translate_click(None)

    assert settings.ui.clipboard_auto_translate_enabled is False
    assert view._clipboard_auto_translate_text.content.value == t(
        "settings.clipboard_auto_translate.off"
    )
    assert emitted[-1].ui.clipboard_auto_translate_enabled is False


def test_load_from_settings_uses_system_prompt_when_provider_prompt_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompt = "LEGACY PROMPT"
    settings.system_prompts = {}

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._prompt_editor.value == "LEGACY PROMPT"
    assert settings.system_prompts == {}


def test_load_from_settings_uses_default_prompt_when_all_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.system_prompt = ""
    settings.system_prompts = {}

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert bool(view._prompt_editor.value.strip())
    assert settings.system_prompt == view._prompt_editor.value
    assert settings.system_prompts == {}


def test_load_secrets_failure_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    basic_messages: list[str] = []

    monkeypatch.setattr(settings_view.SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "_refresh_microphones", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "update", lambda self: None)

    def raise_store(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(settings_view, "create_secret_store", raise_store)
    view = settings_view.SettingsView()
    view.runtime_log_basic = lambda message, *, level=logging.INFO: basic_messages.append(message)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._google_key.value == ""
    assert view._deepgram_key.value == ""
    assert view._soniox_key.value == ""
    assert basic_messages == ["Failed to load secrets: boom"]


def test_restore_api_key_icons_sets_idle_success_error(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    settings.api_key_verified.deepgram = True
    settings.api_key_verified.google = False

    view, _ = _make_settings_view(monkeypatch)
    view._deepgram_key.value = "deepgram-secret"
    view._google_key.value = "google-secret"
    view._soniox_key.value = ""
    view._alibaba_key_beijing.value = ""
    view._alibaba_key_singapore.value = ""

    view._restore_api_key_icons(settings)

    assert view._deepgram_key._current_status == "success"
    assert view._deepgram_key._last_verified_hash
    assert view._google_key._current_status == "error"
    assert view._soniox_key._current_status == "idle"


def test_update_api_visibility_tracks_provider_and_region(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.QWEN_ASR
    settings.provider.llm = LLMProviderName.GEMINI
    settings.qwen.region = QwenRegion.BEIJING

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._qwen_region_btn.visible is True
    assert view._google_key.visible is True
    assert view._alibaba_key_beijing.visible is True
    assert view._alibaba_key_singapore.visible is False

    settings.qwen.region = QwenRegion.SINGAPORE
    settings.provider.llm = LLMProviderName.QWEN
    view._update_api_visibility()

    assert view._google_key.visible is False
    assert view._alibaba_key_beijing.visible is False
    assert view._alibaba_key_singapore.visible is True


def test_update_api_visibility_shows_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PURIPULY_HEART_OPENROUTER_LEGACY_CONNECT", raising=False)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._google_key.visible is False
    assert view._alibaba_key_beijing.visible is False
    assert view._alibaba_key_singapore.visible is False
    assert view._openrouter_key.visible is True
    assert view._openrouter_pkce_button_row.visible is True
    assert view._translation_connection_row.visible is True


def test_update_api_visibility_shows_deepseek_key(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.DEEPSEEK

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._google_key.visible is False
    assert view._openrouter_key.visible is False
    assert view._deepseek_key.visible is True
    assert view._alibaba_key_beijing.visible is False
    assert view._alibaba_key_singapore.visible is False


def test_update_api_visibility_hides_openrouter_key_for_managed_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._openrouter_key.visible is False
    assert view._openrouter_pkce_button_row.visible is False
    assert view._managed_trial_usage_bar.visible is True
    assert view._translation_connection_row.visible is True


def test_update_api_visibility_shows_managed_key_card_for_managed_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.translation.fallback = _enabled_fallback(
        TranslationModel.DEEPSEEK_V4_FLASH,
        TranslationConnection.MANAGED_CHINA,
    )

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._managed_key_card.visible is True
    assert view._managed_trial_usage_bar.visible is True
    assert view._openrouter_key.visible is False


def test_load_from_settings_shows_managed_usage_bar_in_managed_key_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._managed_key_card not in view._api_keys_column.controls
    assert view._managed_key_card in _subtab_controls(view, "api")
    assert view._managed_trial_usage_bar not in view._api_keys_column.controls
    assert _tree_contains_control(view._managed_key_card, view._managed_trial_usage_bar)
    assert view._managed_key_card.visible is True
    assert view._managed_trial_usage_bar.visible is True
    assert view._openrouter_key.visible is False
    assert view._openrouter_pkce_button_row.visible is False


def test_load_from_settings_places_managed_key_card_above_provider_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    api_controls = _subtab_controls(view, "api")
    api_card = _api_tab_card(view, t("settings.section.api_keys"))
    assert view._managed_key_card in api_controls
    assert api_controls.index(view._managed_key_card) < api_controls.index(api_card)
    assert view._managed_key_card not in view._api_keys_column.controls
    assert view._managed_trial_usage_bar not in view._api_keys_column.controls
    assert _card_title(view._managed_key_card) == t("settings.managed_key.title")

    labels = _control_labels(view._managed_key_card)
    assert t("settings.managed_key.referral_id.label") in labels
    assert "무료 사용량" not in labels
    assert "Free usage" not in labels
    assert _tree_contains_control(view._managed_key_card, view._managed_trial_usage_bar)


@pytest.mark.parametrize(
    (
        "model",
        "connection",
        "selected_source",
        "active_ref",
        "referral_id",
        "expected_visible",
    ),
    [
        (
            TranslationModel.GEMMA4,
            TranslationConnection.MANAGED,
            OpenRouterCredentialSource.MANAGED,
            None,
            None,
            True,
        ),
        (
            TranslationModel.DEEPSEEK_V4_FLASH,
            TranslationConnection.MANAGED_CHINA,
            OpenRouterCredentialSource.MANAGED,
            None,
            None,
            True,
        ),
        (
            TranslationModel.GEMMA4,
            TranslationConnection.OPENROUTER,
            OpenRouterCredentialSource.BYOK,
            "managed-ref",
            None,
            False,
        ),
        (
            TranslationModel.GEMMA4,
            TranslationConnection.OPENROUTER,
            OpenRouterCredentialSource.BYOK,
            None,
            "7KQ9M2",
            False,
        ),
    ],
)
def test_managed_key_card_visibility_follows_translation_connection(
    monkeypatch: pytest.MonkeyPatch,
    model: TranslationModel,
    connection: TranslationConnection,
    selected_source: OpenRouterCredentialSource,
    active_ref: str | None,
    referral_id: str | None,
    expected_visible: bool,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation.model = model
    settings.translation.connection = connection
    settings.translation.connection_history[model.value] = connection
    settings.openrouter.selected_source = selected_source
    settings.managed_identity.active_managed_credential_ref = active_ref
    settings.managed_identity.referral_id = referral_id
    settings.validate()

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._managed_key_card.visible is expected_visible


def test_managed_key_referral_row_shows_empty_state_without_copy_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._managed_key_referral_id_value.value == t("settings.managed_key.referral_id.empty")
    referral_row = _wrapped_card_column(view._managed_key_card).controls[4].controls[0]
    assert view._managed_key_referral_id_value in referral_row.controls
    assert not any(isinstance(control, ft.IconButton) for control in referral_row.controls)
    assert view._managed_key_referral_helper_text.value == t(
        "settings.managed_key.referral_id.pending_helper"
    )


def test_managed_key_referral_row_shows_owned_id_without_copy_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.referral_id = "7kq9m2"
    settings.validate()

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._managed_key_card.visible is True
    assert view._managed_key_referral_id_value.value == "7KQ9M2"
    referral_row = _wrapped_card_column(view._managed_key_card).controls[4].controls[0]
    assert view._managed_key_referral_id_value in referral_row.controls
    assert not any(isinstance(control, ft.IconButton) for control in referral_row.controls)
    assert view._managed_key_referral_helper_text.value == t(
        "settings.managed_key.referral_id.helper"
    )


def test_set_managed_trial_usage_state_tracks_visible_and_remaining_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view.set_managed_trial_usage_state(visible=True, remaining_percent=71)

    assert view.managed_trial_usage_state == {
        "visible": True,
        "remaining_percent": 71,
    }
    assert view._managed_trial_usage_bar.visible is True
    assert view._managed_trial_usage_bar.percent == 71

    view.set_managed_trial_usage_state(visible=False, remaining_percent=12)

    assert view.managed_trial_usage_state == {
        "visible": False,
        "remaining_percent": None,
    }
    assert view._managed_trial_usage_bar.visible is True
    assert view._managed_trial_usage_bar.percent is None


def test_set_managed_key_state_updates_card_controls_and_api_section_repaint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    attach_dummy_page(monkeypatch, view)
    updates: list[str] = []
    mounted_page = object()
    view._managed_key_card = SimpleNamespace(
        visible=False,
        page=mounted_page,
        update=lambda: updates.append("managed_key_card"),
    )
    view._api_keys_column = SimpleNamespace(
        page=mounted_page,
        update=lambda: updates.append("api_keys_column"),
    )

    view.set_managed_key_state(
        visible=True,
        remaining_percent=64,
        referral_id="7kq9m2",
    )

    assert view._managed_key_card.visible is True
    assert view._managed_trial_usage_bar.visible is True
    assert view._managed_trial_usage_bar.percent == 64
    assert view._managed_key_referral_id == "7KQ9M2"
    assert view._managed_key_referral_id_value.value == "7KQ9M2"
    assert "api_keys_column" in updates


def test_set_managed_key_state_hides_card_for_openrouter_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation.connection = TranslationConnection.OPENROUTER
    settings.translation.connection_history[TranslationModel.GEMMA4.value] = (
        TranslationConnection.OPENROUTER
    )
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    attach_dummy_page(monkeypatch, view)

    view.set_managed_key_state(
        visible=True,
        remaining_percent=64,
        referral_id="7kq9m2",
    )

    assert view._managed_key_card.visible is False
    assert view._managed_trial_usage_bar.visible is False
    assert view._managed_trial_usage_bar.percent is None
    assert view._managed_key_referral_id == "7KQ9M2"


def test_set_managed_key_state_keeps_card_visible_for_managed_connection_when_usage_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation.connection = TranslationConnection.MANAGED
    settings.translation.connection_history[TranslationModel.GEMMA4.value] = (
        TranslationConnection.MANAGED
    )
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view.set_managed_key_state(
        visible=False,
        remaining_percent=64,
        referral_id="7kq9m2",
    )

    assert view._managed_key_card.visible is True
    assert view._managed_trial_usage_bar.visible is True
    assert view._managed_trial_usage_bar.percent is None
    assert view._managed_key_referral_id == "7KQ9M2"


@pytest.mark.parametrize(
    (
        "initial_connection",
        "initial_source",
        "initial_alias",
        "selected_connection",
        "expected_visible",
    ),
    [
        (
            TranslationConnection.OPENROUTER,
            OpenRouterCredentialSource.BYOK,
            OpenRouterSelectionAlias.GEMMA4_BYOK,
            TranslationConnection.MANAGED,
            True,
        ),
        (
            TranslationConnection.MANAGED,
            OpenRouterCredentialSource.MANAGED,
            OpenRouterSelectionAlias.GEMMA4_MANAGED,
            TranslationConnection.OPENROUTER,
            False,
        ),
    ],
)
def test_translation_connection_selected_repaints_managed_key_card_immediately(
    monkeypatch: pytest.MonkeyPatch,
    initial_connection: TranslationConnection,
    initial_source: OpenRouterCredentialSource,
    initial_alias: OpenRouterSelectionAlias,
    selected_connection: TranslationConnection,
    expected_visible: bool,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMMA4,
        connection=initial_connection,
        connection_history={TranslationModel.GEMMA4.value: initial_connection},
    )
    settings.openrouter.selected_source = initial_source
    settings.openrouter.selection_alias = initial_alias
    view = _make_llm_selection_view(monkeypatch, settings)
    updates: list[str] = []
    mounted_page = object()
    view._managed_key_card = SimpleNamespace(
        visible=initial_connection == TranslationConnection.MANAGED,
        page=mounted_page,
        update=lambda: updates.append("managed_key_card"),
    )
    view._api_keys_column = SimpleNamespace(
        page=mounted_page,
        update=lambda: updates.append("api_keys_column"),
    )
    view._settings_subtab_shell = SimpleNamespace(
        body_by_key={
            "api": SimpleNamespace(
                page=mounted_page,
                update=lambda: updates.append("api_body"),
            )
        }
    )
    attach_dummy_page(monkeypatch, view)

    view._on_translation_connection_selected(selected_connection.value)

    assert view._managed_key_card.visible is expected_visible
    assert "managed_key_card" in updates
    assert "api_body" in updates


def test_set_managed_key_state_empty_referral_disables_copy_and_repaints_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    attach_dummy_page(monkeypatch, view)
    updates: list[str] = []
    mounted_page = object()
    view._managed_key_card = SimpleNamespace(
        visible=True,
        page=mounted_page,
        update=lambda: updates.append("managed_key_card"),
    )
    view._api_keys_column = SimpleNamespace(
        page=mounted_page,
        update=lambda: updates.append("api_keys_column"),
    )

    view.set_managed_key_state(
        visible=True,
        remaining_percent=None,
        referral_id=None,
    )

    assert view._managed_key_card.visible is True
    assert view._managed_trial_usage_bar.visible is True
    assert view._managed_trial_usage_bar.percent is None
    assert view._managed_key_referral_id is None
    assert view._managed_key_referral_id_value.value == t("settings.managed_key.referral_id.empty")
    assert "api_keys_column" in updates


def test_set_managed_key_state_shows_talk_together_pass_invite_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view.set_managed_key_state(
        visible=True,
        remaining_percent=64,
        referral_id="7KQ9M2",
        pass_status=TalkTogetherPassStatus(
            pass_id="7KQ9M2",
            invite_count=1,
            invite_limit=5,
            bonus_translations_per_friend=200,
        ),
    )

    assert view._managed_key_invite_progress_row.visible is True
    assert view._managed_key_invite_progress_label.value == t(
        "settings.managed_key.invite_progress.label"
    )
    assert view._managed_key_invite_progress_value.value == "1 / 5"


def test_set_managed_key_state_repaints_mounted_pass_id_usage_and_invite_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    view = _make_llm_selection_view(monkeypatch, settings)
    updates: list[str] = []
    mounted_page = object()

    view._managed_trial_usage_bar.page = mounted_page
    view._managed_trial_usage_bar.update = lambda: updates.append("usage_bar")
    view._managed_trial_usage_bar._remaining_text.page = mounted_page  # noqa: SLF001
    view._managed_trial_usage_bar._remaining_text.update = lambda: updates.append(  # noqa: SLF001
        "usage_remaining_text"
    )
    view._managed_trial_usage_bar._fill_segments.page = mounted_page  # noqa: SLF001
    view._managed_trial_usage_bar._fill_segments.update = lambda: updates.append(  # noqa: SLF001
        "usage_fill_segments"
    )
    view._managed_key_referral_id_value.page = mounted_page
    view._managed_key_referral_id_value.update = lambda: updates.append("referral_value")
    view._managed_key_referral_helper_text.page = mounted_page
    view._managed_key_referral_helper_text.update = lambda: updates.append("referral_helper")
    view._managed_key_invite_progress_label.page = mounted_page
    view._managed_key_invite_progress_label.update = lambda: updates.append("invite_label")
    view._managed_key_invite_progress_value.page = mounted_page
    view._managed_key_invite_progress_value.update = lambda: updates.append("invite_value")
    view._managed_key_invite_progress_row.page = mounted_page
    view._managed_key_invite_progress_row.update = lambda: updates.append("invite_row")

    view.set_managed_key_state(
        visible=True,
        remaining_percent=64,
        referral_id="7KQ9M2",
        pass_status=TalkTogetherPassStatus(
            pass_id="7KQ9M2",
            invite_count=1,
            invite_limit=5,
            bonus_translations_per_friend=200,
        ),
    )

    assert view._managed_trial_usage_bar.percent == 64
    assert view._managed_key_referral_id_value.value == "7KQ9M2"
    assert view._managed_key_invite_progress_row.visible is True
    assert view._managed_key_invite_progress_value.value == "1 / 5"
    assert {
        "usage_bar",
        "usage_remaining_text",
        "usage_fill_segments",
        "referral_value",
        "referral_helper",
        "invite_label",
        "invite_value",
        "invite_row",
    }.issubset(updates)


def test_managed_key_invite_progress_survives_managed_china_round_trip_before_settings_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED,
            TranslationModel.GEMMA4.value: TranslationConnection.MANAGED,
        },
    )
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view.set_managed_key_state(
        visible=True,
        remaining_percent=64,
        referral_id="7KQ9M2",
        pass_status=TalkTogetherPassStatus(
            pass_id="7KQ9M2",
            invite_count=1,
            invite_limit=5,
            bonus_translations_per_friend=200,
        ),
    )

    view._on_translation_connection_selected(TranslationConnection.MANAGED_CHINA.value)
    assert view._managed_key_referral_id == "7KQ9M2"
    assert view._managed_key_invite_progress_row.visible is True
    assert view._managed_key_invite_progress_value.value == "1 / 5"

    view._on_translation_connection_selected(TranslationConnection.MANAGED.value)
    assert view._managed_key_referral_id == "7KQ9M2"
    assert view._managed_key_invite_progress_row.visible is True
    assert view._managed_key_invite_progress_value.value == "1 / 5"

    view._on_llm_selected(TranslationModel.GEMMA4.value)
    assert view._managed_key_referral_id == "7KQ9M2"
    assert view._managed_key_invite_progress_row.visible is True
    assert view._managed_key_invite_progress_value.value == "1 / 5"


def test_set_managed_key_state_can_display_preview_referral_without_remembering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view.set_managed_key_state(
        visible=True,
        remaining_percent=100,
        referral_id="7KQ9M2",
        pass_status=TalkTogetherPassStatus(
            pass_id="7KQ9M2",
            invite_count=1,
            invite_limit=5,
            bonus_translations_per_friend=200,
        ),
        remember_referral_id=False,
    )

    assert settings.managed_identity.referral_id is None
    assert view._managed_key_referral_id == "7KQ9M2"
    assert view._managed_key_referral_id_value.value == "7KQ9M2"
    assert view._managed_key_invite_progress_row.visible is True


def test_managed_key_invite_progress_row_appears_between_talk_together_pass_id_and_helper_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    managed_key_content = _wrapped_card_column(view._managed_key_card)
    details_column = managed_key_content.controls[4]
    referral_id_row = details_column.controls[0]

    assert view._managed_key_referral_id_label in referral_id_row.controls
    assert details_column.controls[1] is view._managed_key_invite_progress_row
    assert details_column.controls[2] is view._managed_key_referral_helper_text


def test_set_managed_key_state_shows_invite_progress_placeholder_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view.set_managed_key_state(
        visible=True,
        remaining_percent=None,
        referral_id="7KQ9M2",
        pass_status=TalkTogetherPassStatus(
            pass_id="7KQ9M2",
            invite_count=1,
            invite_limit=5,
            bonus_translations_per_friend=200,
        ),
    )
    assert view._managed_key_invite_progress_row.visible is True
    assert view._managed_key_invite_progress_value.value == "1 / 5"

    view.set_managed_key_state(
        visible=True,
        remaining_percent=None,
        referral_id="7KQ9M2",
        pass_status=None,
    )

    assert view._managed_key_invite_progress_row.visible is True
    assert view._managed_key_pass_status is None
    assert view._managed_key_invite_progress_value.value == "- / -"


def test_set_managed_key_state_clamps_over_limit_invite_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view.set_managed_key_state(
        visible=True,
        referral_id="7KQ9M2",
        pass_status=TalkTogetherPassStatus(
            pass_id="7KQ9M2",
            invite_count=7,
            invite_limit=5,
            bonus_translations_per_friend=200,
        ),
    )

    assert view._managed_key_invite_progress_value.value == "5 / 5"


def test_update_api_visibility_treats_peer_local_qwen_as_local_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    settings.provider.peer_stt = STTProviderName.LOCAL_QWEN
    settings.provider.llm = LLMProviderName.GEMINI

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._deepgram_key.visible is False
    assert view._soniox_key.visible is False
    assert view._qwen_region_btn.visible is False
    assert view._alibaba_key_beijing.visible is False
    assert view._alibaba_key_singapore.visible is False
    assert view._google_key.visible is True


def test_deepseek_connection_selection_controls_api_key_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PURIPULY_HEART_OPENROUTER_LEGACY_CONNECT", raising=False)
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.openrouter.fallback_selection_alias = OpenRouterFallbackSelectionAlias.NONE

    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view._on_llm_selected(TranslationModel.DEEPSEEK_V4_FLASH.value)
    assert view._managed_trial_usage_bar.visible is True
    assert view._openrouter_key.visible is False
    assert view._deepseek_key.visible is False

    view._on_translation_connection_selected(TranslationConnection.OPENROUTER.value)
    assert view._managed_trial_usage_bar.visible is False
    assert view._openrouter_key.visible is True
    assert view._deepseek_key.visible is False

    view._on_translation_connection_selected(TranslationConnection.MANAGED_CHINA.value)
    assert view._managed_trial_usage_bar.visible is True
    assert view._openrouter_key.visible is False
    assert view._deepseek_key.visible is False

    view._on_translation_connection_selected(TranslationConnection.OFFICIAL_BYOK.value)
    assert view._managed_trial_usage_bar.visible is False
    assert view._openrouter_key.visible is False
    assert view._deepseek_key.visible is True


def test_on_llm_selected_updates_to_local_llms_with_ollama_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view._on_llm_selected(TranslationModel.LOCAL_LLM.value)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.translation.model == TranslationModel.LOCAL_LLM
    assert pending.translation.connection == TranslationConnection.OLLAMA
    assert pending.provider.llm == LLMProviderName.LOCAL_LLM
    assert view._llm_text.content.value == t("provider.local_llms")
    assert view._translation_connection_text.content.value == t(
        "settings.translation_connection.ollama"
    )
    assert view.has_provider_changes is True


def test_managed_gemma_selection_auto_applies_and_exposes_only_cpu_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
        fallback=_enabled_fallback(
            TranslationModel.DEEPSEEK_V4_FLASH,
            TranslationConnection.OPENROUTER,
        ),
    )
    settings.provider.llm = LLMProviderName.GEMINI
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    applies: list[bool] = []
    view.on_providers_changed = lambda: applies.append(True)

    view._on_llm_selected("managed_gemma_cpu")

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.translation.model == TranslationModel.MANAGED_GEMMA
    assert pending.translation.connection == TranslationConnection.CPU
    assert pending.provider.llm == LLMProviderName.MANAGED_GEMMA
    assert applies == [True]
    assert view._openrouter_fallback_card.visible is False
    assert view._translation_connection_row.visible is False
    assert view._translation_connection_title.value == t("settings.translation_connection")
    assert view._get_llm_display_label(pending) == t("provider.managed_gemma_cpu")

    view._on_llm_selected("managed_gemma_gpu")

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.translation.model == TranslationModel.MANAGED_GEMMA
    assert pending.translation.connection == TranslationConnection.GPU
    assert applies == [True, True]
    assert view._get_llm_display_label(pending) == t("provider.managed_gemma_gpu")

    view._on_llm_selected("managed_gemma_gpu")

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.translation.connection == TranslationConnection.GPU
    assert applies == [True, True]


def test_local_llm_visibility_shows_connection_card_with_server_api_key_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view._update_api_visibility()

    assert view._local_llm_connection_card.visible is True
    assert view._local_llm_api_key.visible is True
    assert view._google_key.visible is False
    assert view._openrouter_key.visible is False
    assert view._deepseek_key.visible is False
    assert view._alibaba_key_beijing.visible is False
    assert view._alibaba_key_singapore.visible is False


def test_local_llm_connection_card_matches_api_field_scale_and_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_locale = i18n_module.get_locale()
    i18n_module.set_locale("ko")
    try:
        view, _ = _make_settings_view(monkeypatch)
        column = _wrapped_card_column(view._local_llm_connection_card)
        controls = list(column.controls)
        api_field = view._google_key._text_field

        assert view._local_llm_base_url.label == "Base URL"
        assert view._local_llm_model.label == "모델 ID"
        assert view._local_llm_extra_body.label == "JSON extra body"
        assert view._local_llm_extra_body_helper.value == (
            "낮은 지연시간을 위해 추론을 끄고 사용하는 것을 권장해요. "
            "JSON extra body에 알맞은 파라미터를 입력해서 추론 레벨을 제어하세요."
        )
        assert view._local_llm_extra_body.value == json.dumps(
            {"reasoning_effort": "none"}, ensure_ascii=False, indent=2
        )
        assert controls[2] is view._local_llm_extra_body_helper
        assert controls[3] is view._local_llm_base_url
        assert controls[4] is view._local_llm_model
        assert controls[5] is view._local_llm_api_key
        assert controls[6] is view._local_llm_api_key_helper
        assert controls[7] is view._local_llm_extra_body

        for field in (
            view._local_llm_base_url,
            view._local_llm_model,
            view._local_llm_extra_body,
        ):
            assert field.border_radius == api_field.border_radius
            assert field.color == api_field.color
            assert field.label_style.weight == api_field.label_style.weight
            assert field.label_style.color == api_field.label_style.color
            assert field.expand is True
            assert field.dense is not True
    finally:
        i18n_module.set_locale(old_locale)


def test_load_from_settings_loads_local_llm_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    store = DummySecretStore({"local_llm_api_key": "server-secret"})
    view, _ = _make_settings_view(monkeypatch, store)

    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._local_llm_api_key.value == "server-secret"


def test_order22_live_settings_view_audio_change_emits_copied_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    settings = AppSettings()
    settings.provider.stt = STTProviderName.DEEPGRAM
    settings.provider.peer_stt = STTProviderName.DEEPGRAM
    settings.audio.input_device = "Built-in Mic"
    view.load_from_settings(settings, config_path=Path("settings.json"))

    emitted: list[AppSettings] = []
    view.on_settings_changed = emitted.append
    view._audio_settings.microphone = "Headset Mic"

    view._on_audio_change()

    assert settings.audio.input_device == "Headset Mic"
    assert view._settings is not None
    assert view._settings is settings
    assert view._settings.audio.input_device == "Headset Mic"
    assert len(emitted) == 1
    assert emitted[0] is not settings
    assert emitted[0] is not view._settings
    assert emitted[0].audio.input_device == "Headset Mic"


def test_load_from_settings_without_local_llm_api_key_shows_empty_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    store = DummySecretStore()
    view, _ = _make_settings_view(monkeypatch, store)

    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._local_llm_api_key.value == ""


def test_load_from_settings_does_not_touch_existing_local_llm_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.GEMINI))
    store = DummySecretStore({"local_llm_api_key": "server-secret"})
    view, _ = _make_settings_view(monkeypatch, store)

    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert store.set_calls == []
    assert store.delete_calls == []


def test_local_llm_secret_change_trims_saves_and_requests_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    store = DummySecretStore()
    view, _ = _make_settings_view(monkeypatch, store)
    view._settings = settings
    view._config_path = Path("settings.json")
    callbacks: list[str] = []
    view.on_local_llm_secret_changed = lambda: callbacks.append("changed")

    view._on_local_llm_secret_change("local_llm_api_key", "  server-secret  ")

    assert store.set_calls == [("local_llm_api_key", "server-secret")]
    assert store.delete_calls == []
    assert callbacks == ["changed"]


def test_local_llm_secret_change_whitespace_deletes_and_requests_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    store = DummySecretStore({"local_llm_api_key": "server-secret"})
    view, _ = _make_settings_view(monkeypatch, store)
    view._settings = settings
    view._config_path = Path("settings.json")
    callbacks: list[str] = []
    view.on_local_llm_secret_changed = lambda: callbacks.append("changed")

    view._on_local_llm_secret_change("local_llm_api_key", "   ")

    assert store.set_calls == []
    assert store.delete_calls == ["local_llm_api_key"]
    assert callbacks == ["changed"]


def test_local_llm_secret_change_failure_does_not_request_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingSecretStore(DummySecretStore):
        def set(self, key: str, value: str) -> None:
            raise RuntimeError("keyring unavailable")

    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    store = FailingSecretStore()
    view, _ = _make_settings_view(monkeypatch, store)
    view._settings = settings
    view._config_path = Path("settings.json")
    snackbars: list[str] = []
    callbacks: list[str] = []
    view.show_snackbar = lambda message, _bg: snackbars.append(message)
    view.on_local_llm_secret_changed = lambda: callbacks.append("changed")

    view._on_local_llm_secret_change("local_llm_api_key", "server-secret")

    assert callbacks == []
    assert snackbars == [t("settings.local_llm.api_key.save_failed")]


def test_local_llm_secret_delete_failure_does_not_request_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingDeleteSecretStore(DummySecretStore):
        def delete(self, key: str) -> None:
            raise RuntimeError("delete failed for server-secret")

    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    store = FailingDeleteSecretStore({"local_llm_api_key": "server-secret"})
    view, _ = _make_settings_view(monkeypatch, store)
    view._settings = settings
    view._config_path = Path("settings.json")
    snackbars: list[str] = []
    callbacks: list[str] = []
    view.show_snackbar = lambda message, _bg: snackbars.append(message)
    view.on_local_llm_secret_changed = lambda: callbacks.append("changed")

    view._on_local_llm_secret_change("local_llm_api_key", "   ")

    assert callbacks == []
    assert snackbars == [t("settings.local_llm.api_key.save_failed")]


def test_local_llm_secret_failure_logs_do_not_include_secret(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class LeakyFailureSecretStore(DummySecretStore):
        def set(self, key: str, value: str) -> None:
            raise RuntimeError(f"backend echoed {value}")

    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    store = LeakyFailureSecretStore()
    view, _ = _make_settings_view(monkeypatch, store)
    view._settings = settings
    view._config_path = Path("settings.json")

    with caplog.at_level(logging.WARNING):
        view._on_local_llm_secret_change("local_llm_api_key", "server-secret")

    assert "server-secret" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_local_llm_fields_update_provider_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    settings.translation = TranslationSettings(
        model=TranslationModel.LOCAL_LLM,
        connection=TranslationConnection.OLLAMA,
    )
    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view._local_llm_base_url.value = "http://127.0.0.1:11434/v1/"
    view._on_local_llm_base_url_change_end(None)
    view._local_llm_model.value = "qwen2.5:7b"
    view._on_local_llm_model_change_end(None)
    view._local_llm_extra_body.value = '{"enable_thinking": false}'
    view._on_local_llm_extra_body_change_end(None)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.local_llm.base_url == "http://127.0.0.1:11434/v1"
    assert pending.local_llm.model == "qwen2.5:7b"
    assert pending.local_llm.extra_body == {"enable_thinking": False}
    assert view.has_provider_changes is True


def test_local_llm_unblurred_fields_commit_when_building_provider_apply_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._local_llm_base_url.value = "http://mac-studio.local:11434/v1"
    view._local_llm_model.value = "gemma3:4b"
    view._local_llm_extra_body.value = '{"think": false}'

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.local_llm.base_url == "http://mac-studio.local:11434/v1"
    assert pending.local_llm.model == "gemma3:4b"
    assert pending.local_llm.extra_body == {"think": False}


def test_local_llm_field_on_change_marks_dirty_and_consume_commits_unblurred_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    settings.translation = TranslationSettings(
        model=TranslationModel.LOCAL_LLM,
        connection=TranslationConnection.OLLAMA,
    )
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._local_llm_base_url.value = "http://mac-studio.local:11434/v1"
    view._local_llm_model.value = "gemma3:4b"
    view._local_llm_extra_body.value = '{"enable_thinking": false}'

    assert view._local_llm_base_url.on_change is not None
    assert view._local_llm_model.on_change is not None
    assert view._local_llm_extra_body.on_change is not None
    view._local_llm_model.on_change(None)

    assert view.has_provider_changes is True

    pending = view.consume_provider_apply_settings()

    assert pending is not None
    assert pending.local_llm.base_url == "http://mac-studio.local:11434/v1"
    assert pending.local_llm.model == "gemma3:4b"
    assert pending.local_llm.extra_body == {"enable_thinking": False}
    assert view.has_provider_changes is False


def test_local_llm_invalid_base_url_shows_error_without_saving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._local_llm_base_url.value = "ftp://127.0.0.1:11434/v1"

    view._on_local_llm_base_url_change_end(None)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.local_llm.base_url == "http://127.0.0.1:11434/v1"
    assert view.has_provider_changes is False
    assert view._local_llm_base_url.value == "ftp://127.0.0.1:11434/v1"
    assert view._local_llm_base_url.error == t("settings.local_llm.base_url.invalid")


def test_local_llm_empty_model_shows_error_without_saving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._local_llm_model.value = "   "

    view._on_local_llm_model_change_end(None)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.local_llm.model == "llama3.1:8b"
    assert view.has_provider_changes is False
    assert view._local_llm_model.value == "   "
    assert view._local_llm_model.error == t("settings.local_llm.model.required")


def test_local_llm_settings_survive_provider_draft_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    settings.translation = TranslationSettings(
        model=TranslationModel.LOCAL_LLM,
        connection=TranslationConnection.OLLAMA,
    )
    settings.local_llm = LocalLLMSettings(
        backend=LocalLLMBackend.OLLAMA,
        base_url="http://127.0.0.1:11434/v1",
        model="llama3.1:8b",
        extra_body={"think": False},
    )
    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view._local_llm_base_url.value = "http://127.0.0.1:11434/v1"
    view._on_local_llm_base_url_change_end(None)
    view._local_llm_model.value = "qwen2.5:7b"
    view._on_local_llm_model_change_end(None)
    view._local_llm_extra_body.value = '{"enable_thinking": false}'
    view._on_local_llm_extra_body_change_end(None)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.local_llm.backend == LocalLLMBackend.OLLAMA
    assert pending.local_llm.base_url == "http://127.0.0.1:11434/v1"
    assert pending.local_llm.model == "qwen2.5:7b"
    assert pending.local_llm.extra_body == {"enable_thinking": False}


def test_local_llm_extra_body_invalid_json_does_not_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._local_llm_extra_body.value = "{invalid-json"

    view._on_local_llm_extra_body_change_end(None)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.local_llm.extra_body == {"reasoning_effort": "none"}
    assert view.has_provider_changes is False
    assert view._local_llm_extra_body.value == "{invalid-json"
    assert view._local_llm_extra_body_error.visible is True
    assert view._local_llm_extra_body_error.value == t("settings.local_llm.extra_body.invalid_json")


def test_local_llm_blank_extra_body_uses_current_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._local_llm_extra_body.value = "  "

    view._on_local_llm_extra_body_change_end(None)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.local_llm.extra_body == {"reasoning_effort": "none"}


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_local_llm_extra_body_rejects_non_standard_json_constants(
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._local_llm_extra_body.value = f'{{"temperature": {constant}}}'

    view._on_local_llm_extra_body_change_end(None)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.local_llm.extra_body == {"reasoning_effort": "none"}
    assert view.has_provider_changes is False
    assert view._local_llm_extra_body.value == f'{{"temperature": {constant}}}'
    assert view._local_llm_extra_body_error.visible is True
    assert view._local_llm_extra_body_error.value == t("settings.local_llm.extra_body.invalid_json")


def test_local_llm_extra_body_non_object_json_does_not_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._local_llm_extra_body.value = '["not", "an", "object"]'

    view._on_local_llm_extra_body_change_end(None)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.local_llm.extra_body == {"reasoning_effort": "none"}
    assert view.has_provider_changes is False
    assert view._local_llm_extra_body.value == '["not", "an", "object"]'
    assert view._local_llm_extra_body_error.visible is True
    assert view._local_llm_extra_body_error.value == t(
        "settings.local_llm.extra_body.must_be_object"
    )


def test_local_llm_extra_body_non_serializable_value_does_not_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._local_llm_extra_body.value = '{"callback": "not actually serializable"}'
    monkeypatch.setattr(settings_view.json, "loads", lambda _raw, **_kwargs: {"callback": object()})

    view._on_local_llm_extra_body_change_end(None)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.local_llm.extra_body == {"reasoning_effort": "none"}
    assert view.has_provider_changes is False
    assert view._local_llm_extra_body_error.visible is True
    assert view._local_llm_extra_body_error.value == t(
        "settings.local_llm.extra_body.not_serializable"
    )


@pytest.mark.parametrize("key", sorted(LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS))
def test_local_llm_extra_body_reserved_key_does_not_save(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._local_llm_extra_body.value = json.dumps({key: True})

    view._on_local_llm_extra_body_change_end(None)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.local_llm.extra_body == {"reasoning_effort": "none"}
    assert view.has_provider_changes is False
    assert view._local_llm_extra_body_error.visible is True
    assert view._local_llm_extra_body_error.value == t(
        "settings.local_llm.extra_body.reserved_key"
    ).format(key=key)


@pytest.mark.parametrize("key", sorted(LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS))
def test_local_llm_extra_body_sensitive_key_does_not_save(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._local_llm_extra_body.value = json.dumps({key: "do-not-save"})

    view._on_local_llm_extra_body_change_end(None)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.local_llm.extra_body == {"reasoning_effort": "none"}
    assert view.has_provider_changes is False
    assert view._local_llm_extra_body_error.visible is True
    assert view._local_llm_extra_body_error.value == t(
        "settings.local_llm.extra_body.sensitive_key"
    ).format(key=key)


def test_official_api_connection_hides_openrouter_key_even_with_saved_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    settings.translation.fallback = TranslationFallbackSettings(enabled=False)

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()
    assert view._openrouter_key.visible is True

    view._on_translation_connection_selected(TranslationConnection.OFFICIAL_BYOK.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.provider.llm == LLMProviderName.DEEPSEEK
    assert pending.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert pending.translation.fallback.enabled is False
    assert view._openrouter_key.visible is False
    assert view._deepseek_key.visible is True
    assert view._openrouter_fallback_helper_text.value == t("settings.fallback.none.description")


def test_on_stt_selected_updates_provider_and_pipeline_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_stt_selected(STTProviderName.SONIOX.value)

    pending = view.build_provider_apply_settings()

    assert settings.provider.stt == STTProviderName.LOCAL_CPU_AUTO
    assert pending is not None
    assert pending.provider.stt == STTProviderName.SONIOX
    assert view.has_provider_changes is True
    assert changed == []


def test_on_stt_selected_routes_compatibility_warning_through_snackbar_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    snackbars: list[tuple[str, object]] = []
    view, _ = _make_settings_view(monkeypatch)
    page = SimpleNamespace(opened=[])
    page.open = lambda control: page.opened.append(control)
    attach_dummy_page(monkeypatch, view, page)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    monkeypatch.setattr(view._qwen_region_btn, "update", lambda: None)
    monkeypatch.setattr(view._api_keys_column, "update", lambda: None)
    monkeypatch.setattr(view._stt_text, "update", lambda: None)
    view.show_snackbar = lambda message, color: snackbars.append((message, color))
    warning = SimpleNamespace(key="warning.deepgram_not_supported", language_code="xx")
    monkeypatch.setattr(
        settings_view,
        "get_stt_compatibility_warning",
        lambda *_args, **_kwargs: warning,
    )

    view._on_stt_selected(STTProviderName.SONIOX.value)

    assert snackbars == [
        (
            t(warning.key, language=language_name(warning.language_code)),
            settings_view.ft.Colors.ORANGE_700,
        )
    ]
    assert page.opened == []


def test_on_peer_stt_selected_updates_provider_and_pipeline_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_peer_stt_selected(STTProviderName.SONIOX.value)

    pending = view.build_provider_apply_settings()

    assert settings.provider.peer_stt == STTProviderName.LOCAL_CPU_AUTO
    assert pending is not None
    assert pending.provider.peer_stt == STTProviderName.SONIOX
    assert view.has_provider_changes is True
    assert changed == []


def test_peer_stt_local_qwen_option_is_selectable_with_provider_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    attach_dummy_page(monkeypatch, view)

    captured: dict[str, object] = {}

    class DummyModal:
        def __init__(
            self,
            _page,
            title,
            options,
            _on_select,
            *,
            show_description=False,
            two_column=False,
            left_column_sections=1,
        ):
            captured["title"] = title
            captured["options"] = options
            captured["show_description"] = show_description
            captured["two_column"] = two_column
            captured["left_column_sections"] = left_column_sections

        def open(self, current: str) -> None:
            captured["current"] = current

    monkeypatch.setattr(settings_view, "SettingsModal", DummyModal)

    view._on_peer_stt_click(None)

    options = captured["options"]
    local_qwen_option = next(
        option for option in options if option.value == STTProviderName.LOCAL_QWEN.value
    )

    assert captured["title"] == t("settings.section.peer_stt")
    assert captured["show_description"] is True
    assert captured["two_column"] is True
    assert captured["left_column_sections"] == 2
    assert local_qwen_option.label == "Qwen3 ASR 0.6B"
    assert local_qwen_option.disabled is False
    assert local_qwen_option.description == t("provider.local_qwen.description")
    assert all(
        option.disabled == (option.value == STTProviderName.LOCAL_CPU_AUTO.value)
        for option in options
    )
    assert {option.value for option in options} == {
        STTProviderName.LOCAL_CPU_AUTO.value,
        STTProviderName.LOCAL_PARAKEET_V3.value,
        STTProviderName.LOCAL_PARAKEET_JAPANESE.value,
        STTProviderName.LOCAL_QWEN.value,
        STTProviderName.LOCAL_QWEN_GPU.value,
        STTProviderName.DEEPGRAM.value,
        STTProviderName.QWEN_ASR.value,
        STTProviderName.SONIOX.value,
        STTProviderName.CUSTOM_OFFLINE.value,
        STTProviderName.CUSTOM_REALTIME.value,
    }
    assert STTProviderName.LOCAL_QWEN_GPU.value in {option.value for option in options}


def test_peer_stt_local_qwen_choice_can_be_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.DEEPGRAM
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_peer_stt_selected(STTProviderName.LOCAL_QWEN.value)

    pending = view.build_provider_apply_settings()

    assert settings.provider.peer_stt == STTProviderName.DEEPGRAM
    assert pending is not None
    assert pending.provider.peer_stt == STTProviderName.LOCAL_QWEN
    assert view.has_provider_changes is True

    settings.provider.peer_stt = STTProviderName.LOCAL_QWEN
    view.load_from_settings(settings, config_path=Path("settings.json"))

    normalized_pending = view.build_provider_apply_settings()

    assert normalized_pending is not None
    assert normalized_pending.provider.peer_stt == STTProviderName.LOCAL_QWEN


def test_gpu_selection_shows_one_shared_device_card_and_retains_unavailable_saved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.LOCAL_QWEN_GPU
    settings.stt.gpu_device_id = "vk:saved"
    view, _ = _make_settings_view(monkeypatch)

    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.set_gpu_devices(devices=(GpuDeviceOption("vk:0", "NVIDIA GeForce RTX 4070", "Vulkan0"),))

    assert view._gpu_device_card.visible is True
    assert view._gpu_device_row.visible is True
    assert view._gpu_device_text.content.value == t(
        "settings.gpu_device.unavailable",
        device="vk:saved",
    )

    view._on_gpu_device_selected("vk:0")

    assert view._gpu_device_text.content.value == "NVIDIA GeForce RTX 4070"
    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.stt.gpu_device_id == "vk:0"


def test_gpu_card_is_standard_one_by_one_and_contains_only_device_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    column = _wrapped_card_column(view._gpu_device_card)

    assert len(column.controls) == 2
    assert column.controls[0] is view._gpu_device_title
    assert column.controls[1].content is view._gpu_device_text
    assert view._gpu_device_text.content.size == view._stt_text.content.size
    assert view._gpu_device_text.content.color == view._stt_text.content.color
    assert view._gpu_device_text.content.text_align == view._stt_text.content.text_align
    assert len(view._gpu_device_row.content.controls) == 3
    assert view._gpu_device_row.content.controls[1].opacity == 1.0
    assert view._gpu_device_row.content.controls[2].opacity == 1.0


def test_gpu_row_repaints_and_collapses_immediately_with_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    view, _ = _make_settings_view(monkeypatch)
    repainted: list[ft.Control] = []
    monkeypatch.setattr(settings_view, "_update_control_if_mounted", repainted.append)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._gpu_device_card.visible is False
    assert view._gpu_device_row.visible is False

    repainted.clear()
    view._on_stt_selected(STTProviderName.LOCAL_QWEN_GPU.value)

    assert view._gpu_device_card.visible is True
    assert view._gpu_device_row.visible is True
    assert view._gpu_device_row in repainted

    repainted.clear()
    view._on_stt_selected(STTProviderName.LOCAL_QWEN.value)

    assert view._gpu_device_card.visible is False
    assert view._gpu_device_row.visible is False
    assert view._gpu_device_row in repainted


def test_selecting_gpu_for_self_or_peer_requests_discovery_once_per_new_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.DEEPGRAM
    settings.provider.peer_stt = STTProviderName.DEEPGRAM
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    requests: list[str] = []
    view.on_gpu_discovery_requested = lambda: requests.append("discover")

    view._on_stt_selected(STTProviderName.LOCAL_QWEN_GPU.value)
    view._on_stt_selected(STTProviderName.LOCAL_QWEN_GPU.value)
    view._on_peer_stt_selected(STTProviderName.LOCAL_QWEN_GPU.value)
    view._on_peer_stt_selected(STTProviderName.LOCAL_QWEN_GPU.value)

    assert requests == ["discover", "discover"]


def test_settings_view_omits_legacy_overlay_peer_toggle_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    assert not hasattr(view, "on_overlay_toggle")
    assert not hasattr(view, "on_peer_translation_toggle")
    assert not hasattr(view, "_overlay_enabled_label")
    assert not hasattr(view, "_overlay_enabled_button")
    assert not hasattr(view, "_peer_translation_label")
    assert not hasattr(view, "_peer_translation_button")
    assert not hasattr(view, "_peer_translation_status_text")
    assert not hasattr(view, "_peer_translation_hint")
    assert not hasattr(view, "_overlay_status_text")
    assert not hasattr(settings_view.SettingsView, "_on_overlay_click")
    assert not hasattr(settings_view.SettingsView, "_on_overlay_selected")
    assert not hasattr(settings_view.SettingsView, "_on_peer_translation_click")
    assert not hasattr(settings_view.SettingsView, "_on_peer_translation_selected")


def test_on_llm_selected_updates_model_and_prompt_state(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {"gemini": "G", "qwen": "Q"}
    settings.system_prompt = "G"

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)

    pending = view.build_provider_apply_settings()

    assert settings.provider.llm == LLMProviderName.GEMINI
    assert pending is not None
    assert pending.translation.model == TranslationModel.QWEN_35_PLUS
    assert pending.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert pending.provider.llm == LLMProviderName.QWEN
    assert pending.qwen.llm_model == QwenLLMModel.QWEN_35_PLUS
    assert view._prompt_editor.value == "G"
    assert settings.system_prompt == "G"

    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)
    assert view.has_provider_changes is True


def test_on_translation_connection_selected_updates_openrouter_model_and_prompt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {
        "gemini": "G",
        "openrouter": "O",
        "qwen": "Q",
    }
    settings.system_prompt = "G"

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view.on_request_openrouter_pkce = lambda _settings: (_ for _ in ()).throw(
        AssertionError("BYOK selection should not launch PKCE immediately")
    )

    view._on_llm_selected(TranslationModel.GEMMA4.value)
    view._on_translation_connection_selected(TranslationConnection.OPENROUTER.value)

    pending = view.build_provider_apply_settings()

    assert settings.provider.llm == LLMProviderName.GEMINI
    assert pending is not None
    assert pending.translation.model == TranslationModel.GEMMA4
    assert pending.translation.connection == TranslationConnection.OPENROUTER
    assert pending.provider.llm == LLMProviderName.OPENROUTER
    assert pending.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_BYOK
    assert pending.system_prompt == "G"
    assert view._prompt_editor.value == "G"
    assert settings.system_prompt == "G"
    assert view.has_provider_changes is True


def test_on_llm_selected_updates_deepseek_model_with_default_managed_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {
        "gemini": "G",
        "openrouter": "O",
        "deepseek": "D",
    }
    settings.system_prompt = "G"

    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view._on_llm_selected(TranslationModel.DEEPSEEK_V4_FLASH.value)

    pending = view.build_provider_apply_settings()

    assert settings.provider.llm == LLMProviderName.GEMINI
    assert pending is not None
    assert pending.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert pending.translation.connection == TranslationConnection.MANAGED
    assert pending.provider.llm == LLMProviderName.OPENROUTER
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    assert pending.system_prompt == "G"
    assert view._prompt_editor.value == "G"
    assert view._llm_text.content.value == t("provider.deepseek_v4_flash")
    assert view._translation_connection_text.content.value == t(
        "settings.translation_connection.managed"
    )
    assert view._managed_trial_usage_bar.visible is True
    assert settings.system_prompt == "G"
    assert view.has_provider_changes is True


def test_on_llm_selected_restores_saved_connection_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMMA4,
        connection=TranslationConnection.MANAGED,
        connection_history={
            TranslationModel.GEMMA4.value: TranslationConnection.MANAGED,
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.OFFICIAL_BYOK,
        },
    )
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_MANAGED
    settings.system_prompts = {"openrouter": "O", "deepseek": "D"}
    settings.system_prompt = "O"

    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view._on_llm_selected(TranslationModel.DEEPSEEK_V4_FLASH.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert pending.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert pending.provider.llm == LLMProviderName.DEEPSEEK
    assert pending.deepseek.llm_model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH
    assert view._llm_text.content.value == t("provider.deepseek_v4_flash")
    assert view._translation_connection_text.content.value == t(
        "settings.translation_connection.official_byok"
    )
    assert view._prompt_editor.value == "O"


def test_on_llm_selected_invalid_value_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {"gemini": "G", "openrouter": "O"}
    settings.system_prompt = "G"

    view = _make_llm_selection_view(monkeypatch, settings)
    view._llm_text.content.value = "Gemini 3 Flash"
    view._translation_connection_text.content.value = "Official BYOK"

    view._on_llm_selected(LLMProviderName.DEEPSEEK.value)

    pending = view.build_provider_apply_settings()

    assert pending is settings
    assert view._provider_settings_draft is None
    assert pending.translation.model == TranslationModel.GEMINI_37_FLASH
    assert pending.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert pending.provider.llm == LLMProviderName.GEMINI
    assert view._llm_text.content.value == "Gemini 3 Flash"
    assert view._translation_connection_text.content.value == "Official BYOK"
    assert view._prompt_editor.value == "G"
    assert view.has_provider_changes is False


def test_on_llm_selected_stages_openrouter_byok_alias_without_pkce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.system_prompts = {"gemini": "G", "openrouter": "O", "qwen": "Q"}
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view.on_request_openrouter_pkce = lambda _settings: (_ for _ in ()).throw(
        AssertionError("BYOK selection should not launch PKCE immediately")
    )

    view._on_translation_connection_selected(TranslationConnection.OPENROUTER.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.translation.model == TranslationModel.GEMMA4_26B_31B
    assert pending.translation.connection == TranslationConnection.OPENROUTER
    assert pending.provider.llm == LLMProviderName.OPENROUTER
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert view.has_provider_changes is True


def test_on_llm_selected_stages_byok_with_default_openrouter_prompt_when_unsaved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings_view, "load_prompt_for_provider", lambda _provider: "DEFAULT PROMPT"
    )
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {"gemini": "G", "qwen": "Q"}
    settings.system_prompt = ""
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view.on_request_openrouter_pkce = lambda _settings: (_ for _ in ()).throw(
        AssertionError("BYOK selection should not launch PKCE immediately")
    )

    view._on_llm_selected(TranslationModel.GEMMA4.value)
    view._on_translation_connection_selected(TranslationConnection.OPENROUTER.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.provider.llm == LLMProviderName.OPENROUTER
    assert pending.system_prompt == "DEFAULT PROMPT"
    assert pending.system_prompts == {}
    assert view._prompt_editor.value == "DEFAULT PROMPT"


def test_on_llm_selected_updates_managed_openrouter_label_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {
        "gemini": "G",
        "openrouter": "O",
        "qwen": "Q",
    }
    settings.system_prompt = "G"

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._on_llm_selected(TranslationModel.GEMMA4.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.provider.llm == LLMProviderName.OPENROUTER
    assert pending.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_MANAGED
    assert view._llm_text.content.value == t("provider.gemma4_26b_a4b_it")
    assert view._translation_connection_text.content.value == t(
        "settings.translation_connection.managed"
    )
    assert view._openrouter_key.visible is False
    assert view._prompt_editor.value == "G"


def test_on_llm_selected_openrouter_provider_value_defaults_to_gemma_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._on_llm_selected(LLMProviderName.OPENROUTER.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.provider.llm == LLMProviderName.OPENROUTER
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_MANAGED
    assert view._llm_text.content.value == t("provider.gemma4_26b_a4b_it")


def test_on_llm_selected_sets_deepseek_managed_connection_and_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {
        "gemini": "G",
        "openrouter": "O",
        "qwen": "Q",
    }
    settings.system_prompt = "G"

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._on_llm_selected(TranslationModel.DEEPSEEK_V4_FLASH.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert pending.translation.connection == TranslationConnection.MANAGED
    assert pending.provider.llm == LLMProviderName.OPENROUTER
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    assert pending.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert view._llm_text.content.value == t("provider.deepseek_v4_flash")
    assert view._translation_connection_text.content.value == t(
        "settings.translation_connection.managed"
    )
    assert view._prompt_editor.value == "G"


def test_on_llm_selected_updates_prompt_helper_copy_live_when_mounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {"gemini": "G", "qwen": "Q"}
    settings.system_prompt = "G"

    view = _make_llm_selection_view(monkeypatch, settings)
    monkeypatch.setattr(settings_view.SettingsView, "page", property(lambda self: object()))
    prompt_copy_updates: list[str] = []
    view._prompt_for_text = SimpleNamespace(
        value="stale",
        update=lambda: prompt_copy_updates.append(view._prompt_for_text.value),
    )

    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)

    assert view._prompt_for_text.value == t(
        "settings.prompt_for",
        provider=provider_label(LLMProviderName.QWEN.value),
    )
    assert prompt_copy_updates == [
        t(
            "settings.prompt_for",
            provider=provider_label(LLMProviderName.QWEN.value),
        )
    ]


def test_on_llm_selected_stages_byok_without_mutating_managed_identity_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.verified_hardware_hash = "hardware-hash"
    settings.managed_identity.verified_hardware_hash_salt_version = 7

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view.on_request_openrouter_pkce = lambda _settings: (_ for _ in ()).throw(
        AssertionError("BYOK selection should not launch PKCE immediately")
    )

    view._on_translation_connection_selected(TranslationConnection.OPENROUTER.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert pending.managed_identity.verified_hardware_hash == "hardware-hash"
    assert pending.managed_identity.verified_hardware_hash_salt_version == 7


def test_on_llm_selected_round_trips_back_to_managed_without_dropping_verified_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.verified_hardware_hash = "hardware-hash"
    settings.managed_identity.verified_hardware_hash_salt_version = 7

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_translation_connection_selected(TranslationConnection.OPENROUTER.value)
    view._on_translation_connection_selected(TranslationConnection.MANAGED.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert pending.managed_identity.verified_hardware_hash == "hardware-hash"
    assert pending.managed_identity.verified_hardware_hash_salt_version == 7


def test_on_llm_selected_switching_away_from_openrouter_preserves_saved_selection_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMMA4,
        connection=TranslationConnection.OPENROUTER,
    )
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    settings.system_prompts = {"gemini": "G", "openrouter": "O", "qwen": "Q"}
    settings.system_prompt = "O"

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.provider.llm == LLMProviderName.QWEN
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_BYOK


def test_on_llm_selected_preserves_default_openrouter_managed_selection_during_gemini_and_qwen_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_llm_selected(TranslationModel.GEMINI_31_FLASH_LITE.value)
    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.provider.llm == LLMProviderName.GEMINI
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_26B_31B_MANAGED

    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)
    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.provider.llm == LLMProviderName.QWEN
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_26B_31B_MANAGED


def test_on_translation_connection_selected_updates_settings_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED,
        },
    )
    changed: list[AppSettings] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_translation_connection_selected(TranslationConnection.OFFICIAL_BYOK.value)

    pending = view.build_provider_apply_settings()

    assert settings.translation.connection == TranslationConnection.MANAGED
    assert pending is not None
    assert pending.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert (
        pending.translation.connection_history[TranslationModel.DEEPSEEK_V4_FLASH.value]
        == TranslationConnection.OFFICIAL_BYOK
    )
    assert pending.provider.llm == LLMProviderName.DEEPSEEK
    assert view._translation_connection_text.content.value == t(
        "settings.translation_connection.official_byok"
    )
    assert view.has_provider_changes is True
    assert changed == []


def test_on_translation_connection_selected_auto_applies_managed_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMMA4,
        connection=TranslationConnection.OPENROUTER,
        connection_history={TranslationModel.GEMMA4.value: TranslationConnection.OPENROUTER},
    )
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    provider_changes: list[str] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_providers_changed = lambda: provider_changes.append("apply")

    view._on_translation_connection_selected(TranslationConnection.MANAGED.value)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.translation.connection == TranslationConnection.MANAGED
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert provider_changes == ["apply"]


def test_on_translation_connection_selected_stages_deepseek_managed_china_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED,
        },
    )

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_translation_connection_selected(TranslationConnection.MANAGED_CHINA.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.translation.connection == TranslationConnection.MANAGED_CHINA
    assert (
        pending.translation.connection_history[TranslationModel.DEEPSEEK_V4_FLASH.value]
        == TranslationConnection.MANAGED_CHINA
    )
    assert pending.provider.llm == LLMProviderName.OPENROUTER
    assert pending.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    assert pending.openrouter.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY
    assert view._translation_connection_text.content.value == t(
        "settings.translation_connection.managed_china"
    )
    assert view._managed_trial_usage_bar.visible is True


def test_on_openrouter_fallback_selected_updates_draft_and_helper_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_openrouter_fallback_selected("openrouter_deepseek_v4_flash")

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.translation.fallback == TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    assert view._openrouter_fallback_text.content.value == t(
        "settings.fallback.openrouter_deepseek_v4_flash"
    )
    assert view._openrouter_fallback_helper_text.value == t("settings.fallback.active_helper")

    view._on_llm_selected(TranslationModel.GEMMA4.value)

    assert view._openrouter_fallback_helper_text.value == t("settings.fallback.active_helper")


def test_on_openrouter_fallback_selected_requests_immediate_provider_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    applied: list[AppSettings] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    def apply_provider_changes() -> None:
        pending = view.consume_provider_apply_settings()
        assert pending is not None
        applied.append(pending)

    view.on_providers_changed = apply_provider_changes

    view._on_openrouter_fallback_selected("openrouter_deepseek_v4_flash")

    assert len(applied) == 1
    assert applied[0].translation.fallback == TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    assert view.has_provider_changes is False


def test_on_openrouter_fallback_selected_defaults_invalid_value_to_deepseek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.openrouter.fallback_selection_alias = OpenRouterFallbackSelectionAlias.NONE

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_openrouter_fallback_selected("broken-fallback")
    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.translation.fallback.enabled is False
    assert view._openrouter_fallback_text.content.value == t("settings.fallback.none")


def test_update_api_visibility_keeps_openrouter_key_for_openrouter_deepseek_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PURIPULY_HEART_OPENROUTER_LEGACY_CONNECT", raising=False)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    settings.translation.fallback = _enabled_fallback(
        TranslationModel.DEEPSEEK_V4_FLASH,
        TranslationConnection.OPENROUTER,
    )

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._google_key.visible is False
    assert view._openrouter_key.visible is True
    assert view._alibaba_key_beijing.visible is False
    assert view._alibaba_key_singapore.visible is False


def test_update_api_visibility_hides_openrouter_key_for_inactive_byok_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PURIPULY_HEART_OPENROUTER_LEGACY_CONNECT", raising=False)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    settings.translation.fallback = TranslationFallbackSettings(enabled=False)

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._google_key.visible is True
    assert view._openrouter_key.visible is False


def test_update_api_visibility_shows_openrouter_key_for_openrouter_fallback_when_main_provider_is_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PURIPULY_HEART_OPENROUTER_LEGACY_CONNECT", raising=False)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    settings.translation.fallback = _enabled_fallback(
        TranslationModel.DEEPSEEK_V4_FLASH,
        TranslationConnection.OPENROUTER,
    )

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._google_key.visible is True
    assert view._openrouter_key.visible is True


def test_update_api_visibility_shows_openrouter_key_for_openrouter_gemma_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PURIPULY_HEART_OPENROUTER_LEGACY_CONNECT", raising=False)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.translation.fallback = _enabled_fallback(
        TranslationModel.GEMMA4,
        TranslationConnection.OPENROUTER,
    )

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._google_key.visible is True
    assert view._openrouter_key.visible is True
    assert view._cerebras_key.visible is False


def test_update_api_visibility_shows_cerebras_key_for_cerebras_fallback_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PURIPULY_HEART_OPENROUTER_LEGACY_CONNECT", raising=False)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.translation.fallback = _enabled_fallback(
        TranslationModel.GEMMA4_31B,
        TranslationConnection.CEREBRAS,
    )

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._google_key.visible is True
    assert view._openrouter_key.visible is False
    assert view._cerebras_key.visible is True


def test_openrouter_key_field_and_pkce_button_are_visible_for_byok_without_break_glass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PURIPULY_HEART_OPENROUTER_LEGACY_CONNECT", raising=False)
    view, _store = _make_settings_view(monkeypatch)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK

    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._openrouter_key.visible is True
    assert view._openrouter_pkce_button_row.visible is True
    assert view._api_keys_column.controls.index(
        view._openrouter_key
    ) < view._api_keys_column.controls.index(view._openrouter_pkce_button_row)
    assert view._openrouter_pkce_button.content == t("settings.openrouter_authenticate")
    assert view._openrouter_pkce_button.disabled is False
    assert view._openrouter_pkce_button.style.color[ft.ControlState.DEFAULT] == COLOR_NEUTRAL_DARK
    assert view._openrouter_pkce_button.style.color[ft.ControlState.DISABLED] == COLOR_NEUTRAL_DARK
    assert (
        view._openrouter_pkce_button.style.color[ft.ControlState.HOVERED]
        == settings_view.COLOR_PRIMARY
    )


def test_openrouter_pkce_button_shows_authenticated_state_after_verified_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DummySecretStore({"openrouter_api_key": "sk-or-v1-pkce"})
    view, _ = _make_settings_view(monkeypatch, store)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    settings.api_key_verified.openrouter = True

    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._openrouter_pkce_button.content == t("settings.openrouter_authenticated")
    assert view._openrouter_pkce_button.disabled is True
    assert view._openrouter_pkce_button.style.color[ft.ControlState.DEFAULT] == COLOR_NEUTRAL_DARK


def test_openrouter_pkce_button_returns_to_authenticate_when_key_is_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DummySecretStore({"openrouter_api_key": "sk-or-v1-pkce"})
    view, _ = _make_settings_view(monkeypatch, store)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    settings.api_key_verified.openrouter = True
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_secret_cleared = lambda key: setattr(settings.api_key_verified, "openrouter", False)

    view._openrouter_key.value = ""
    view._on_secret_change("openrouter_api_key", "")

    assert view._openrouter_pkce_button.content == t("settings.openrouter_authenticate")
    assert view._openrouter_pkce_button.disabled is False
    assert view._openrouter_pkce_button.style.color[ft.ControlState.DEFAULT] == COLOR_NEUTRAL_DARK


@pytest.mark.asyncio
async def test_openrouter_pkce_button_disables_after_manual_key_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view._openrouter_key.value = "sk-or-v1-manual"

    async def fake_verify(provider: str, key: str) -> tuple[bool, str]:
        assert provider == "openrouter"
        assert key == "sk-or-v1-manual"
        settings.api_key_verified.openrouter = True
        return True, "ok"

    view.on_verify_api_key = fake_verify

    await view._openrouter_key._verify_async(
        "sk-or-v1-manual",
        view._openrouter_key._get_key_hash("sk-or-v1-manual"),
    )

    assert view._openrouter_pkce_button.content == t("settings.openrouter_authenticated")
    assert view._openrouter_pkce_button.disabled is True


@pytest.mark.asyncio
async def test_openrouter_pkce_button_reenables_after_manual_key_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DummySecretStore({"openrouter_api_key": "sk-or-v1-old"})
    view, _ = _make_settings_view(monkeypatch, store)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    settings.api_key_verified.openrouter = True
    view.load_from_settings(settings, config_path=Path("settings.json"))
    assert view._openrouter_pkce_button.disabled is True
    view._openrouter_key.value = "invalid-openrouter-key"

    async def fake_verify(provider: str, key: str) -> tuple[bool, str]:
        assert provider == "openrouter"
        assert key == "invalid-openrouter-key"
        settings.api_key_verified.openrouter = False
        return False, "401 unauthorized"

    view.on_verify_api_key = fake_verify

    await view._openrouter_key._verify_async(
        "invalid-openrouter-key",
        view._openrouter_key._get_key_hash("invalid-openrouter-key"),
    )

    assert view._openrouter_pkce_button.content == t("settings.openrouter_authenticate")
    assert view._openrouter_pkce_button.disabled is False


def test_on_llm_selected_stages_byok_even_when_legacy_openrouter_key_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {"gemini": "G", "openrouter": "O", "qwen": "Q"}
    store = DummySecretStore({"openrouter_api_key": "legacy-openrouter-key"})
    view, _ = _make_settings_view(monkeypatch, store)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_request_openrouter_pkce = lambda _settings: (_ for _ in ()).throw(
        AssertionError("BYOK selection should not launch PKCE immediately")
    )

    assert view._openrouter_key.value == "legacy-openrouter-key"

    view._on_translation_connection_selected(TranslationConnection.OPENROUTER.value)

    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.translation.model == TranslationModel.GEMMA4_26B_31B
    assert pending.translation.connection == TranslationConnection.OPENROUTER
    assert pending.provider.llm == LLMProviderName.OPENROUTER
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK


def test_openrouter_pkce_button_requests_auth_for_current_byok_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {"gemini": "G", "openrouter": "O", "qwen": "Q"}
    settings.system_prompt = "G"
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    requested: list[AppSettings] = []
    view.on_request_openrouter_pkce = requested.append

    view._on_translation_connection_selected(TranslationConnection.OPENROUTER.value)
    view._on_openrouter_pkce_click(None)

    assert requested[0].provider.llm == LLMProviderName.OPENROUTER
    assert requested[0].openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert requested[0].openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK
    assert requested[0].system_prompt == "G"


def test_refresh_after_openrouter_pkce_success_preserves_unrelated_drafts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DummySecretStore({"openrouter_api_key": "pkce-openrouter-key"})
    initial = AppSettings()
    initial.provider.llm = LLMProviderName.GEMINI
    initial.languages.source_language = "ko"
    initial.system_prompts = {"gemini": "G", "openrouter": "O", "qwen": "Q"}
    initial.system_prompt = "G"

    view, _ = _make_settings_view(monkeypatch, store)
    view.load_from_settings(initial, config_path=Path("settings.json"))
    view._custom_vocab_tag_editor._input_field.value = "VRChat"  # noqa: SLF001
    view._google_key.value = "typed-google-draft"

    updated = AppSettings()
    updated.provider.llm = LLMProviderName.OPENROUTER
    updated.languages.source_language = "ko"
    updated.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    updated.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    updated.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    updated.api_key_verified.openrouter = True
    updated.system_prompts = {"gemini": "G", "openrouter": "O", "qwen": "Q"}
    updated.system_prompt = "O"

    view.refresh_after_openrouter_pkce_success(updated, config_path=Path("settings.json"))

    assert view._custom_vocab_tag_editor._input_field.value == ""  # noqa: SLF001
    assert _custom_vocab_chip_terms(view) == []
    assert view._google_key.value == "typed-google-draft"
    assert view._openrouter_key.value == "pkce-openrouter-key"
    assert view._openrouter_key._current_status == "success"
    assert view._llm_text.content.value == view._get_llm_display_label(updated)
    assert view.has_provider_changes is False
    assert view.has_pending_prompt_changes is False


def test_hidden_legacy_deepseek_china_fallback_displays_safe_current_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation.fallback = _enabled_fallback(
        TranslationModel.DEEPSEEK_V4_FLASH,
        TranslationConnection.MANAGED_CHINA,
    )

    view, _ = _make_settings_view(monkeypatch, settings=settings)

    assert view._translation_fallback_preset_value(settings.translation.fallback) == "none"
    assert view._get_openrouter_fallback_display_label(settings) == t("settings.fallback.none")


def test_on_llm_selected_updates_gemini_model(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.gemini.llm_model = GeminiLLMModel.GEMINI_37_FLASH
    settings.system_prompts = {"gemini": "G", "qwen": "Q"}
    settings.system_prompt = "G"

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view._on_llm_selected(TranslationModel.GEMINI_31_FLASH_LITE.value)

    pending = view.build_provider_apply_settings()

    assert settings.provider.llm == LLMProviderName.GEMINI
    assert settings.gemini.llm_model == GeminiLLMModel.GEMINI_37_FLASH
    assert pending is not None
    assert pending.translation.model == TranslationModel.GEMINI_31_FLASH_LITE
    assert pending.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert pending.gemini.llm_model == GeminiLLMModel.GEMINI_31_FLASH_LITE
    assert view._prompt_editor.value == "G"
    assert settings.system_prompt == "G"
    assert view.has_provider_changes is True


def test_on_llm_selected_logs_only_changed_fields_for_provider_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_31_FLASH_LITE,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.gemini.llm_model = GeminiLLMModel.GEMINI_31_FLASH_LITE
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_PLUS
    basic_messages: list[str] = []
    detailed_messages: list[str] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.runtime_log_basic = lambda message, *, level=logging.INFO: basic_messages.append(message)
    view.runtime_log_detailed = lambda message, *, level=logging.INFO: detailed_messages.append(
        message
    )

    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)

    assert basic_messages == ["[Settings] LLM provider changed: gemini -> qwen"]
    assert detailed_messages == [
        "[Settings] Translation selection changed: "
        "model=gemini31_flash_lite->qwen35_plus, provider=gemini->qwen"
    ]


def test_on_llm_selected_skips_log_when_selection_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.QWEN_35_PLUS,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.QWEN
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_PLUS
    basic_messages: list[str] = []
    detailed_messages: list[str] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.runtime_log_basic = lambda message, *, level=logging.INFO: basic_messages.append(message)
    view.runtime_log_detailed = lambda message, *, level=logging.INFO: detailed_messages.append(
        message
    )

    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)

    assert basic_messages == []
    assert detailed_messages == []


def test_on_ui_and_region_selection_emit_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_ui_selected("ko")
    view._on_qwen_region_selected(QwenRegion.SINGAPORE.value)

    assert settings.ui.locale == "ko"
    assert settings.qwen.region == QwenRegion.BEIJING
    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.qwen.region == QwenRegion.SINGAPORE
    assert view.has_provider_changes is True
    assert len(changed) == 1
    assert changed[0].qwen.region == QwenRegion.BEIJING


def test_provider_draft_does_not_leak_into_immediate_settings_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_stt_selected(STTProviderName.SONIOX.value)
    view._on_ui_selected("ko")

    pending = view.build_provider_apply_settings()

    assert len(changed) == 1
    assert changed[0].ui.locale == "ko"
    assert changed[0].provider.stt == STTProviderName.LOCAL_CPU_AUTO
    assert pending is not None
    assert pending.provider.stt == STTProviderName.SONIOX


def test_provider_selection_equality_guards_skip_noop_draft_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_stt_selected(STTProviderName.LOCAL_CPU_AUTO.value)
    view._on_peer_stt_selected(STTProviderName.LOCAL_CPU_AUTO.value)
    view._on_translation_connection_selected(TranslationConnection.MANAGED.value)
    view._on_qwen_region_selected(QwenRegion.BEIJING.value)

    assert view.has_provider_changes is False
    assert changed == []


def test_on_secret_change_saves_and_clears_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    store = DummySecretStore()
    cleared: list[str] = []
    view, _ = _make_settings_view(monkeypatch, store)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_secret_cleared = lambda key: cleared.append(key)

    view._on_secret_change("google_api_key", "abc")
    view._on_secret_change("google_api_key", "")

    assert store.values.get("google_api_key") is None
    assert store.set_calls == [("google_api_key", "abc")]
    assert store.delete_calls == ["google_api_key"]
    assert cleared == ["google_api_key"]


def test_audio_and_vad_handlers_update_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._audio_settings.host_api = "MME"
    view._audio_settings.microphone = "Mic 2"
    view._on_audio_change()

    visual_event = SimpleNamespace(control=SimpleNamespace(value=0.72))
    monkeypatch.setattr(type(view._vad_slider), "update", lambda self: None)
    view._handle_vad_visual_change(visual_event)
    view._handle_vad_change(visual_event)
    view._peer_vad_field.value = "0.61"
    view._on_peer_vad_threshold_change(SimpleNamespace(control=view._peer_vad_field))

    assert view._settings.audio.input_host_api == "MME"
    assert view._settings.audio.input_device == "Mic 2"
    assert view._settings.stt.vad_speech_threshold == 0.72
    assert view._settings.desktop_audio.vad_speech_threshold == 0.61
    assert settings.audio.input_host_api == "MME"
    assert settings.audio.input_device == "Mic 2"
    assert settings.stt.vad_speech_threshold == 0.72
    assert settings.desktop_audio.vad_speech_threshold == 0.61


def test_peer_vad_slider_change_skips_hidden_field_update_when_view_is_mounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UpdateRecorder:
        def __init__(self) -> None:
            self.updated: list[object] = []

        def update(self, control: object) -> None:
            self.updated.append(control)

    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    attach_dummy_page(monkeypatch, view)
    slider_page = UpdateRecorder()
    attach_dummy_page(monkeypatch, view._peer_vad_slider, slider_page)

    view._handle_peer_vad_change(SimpleNamespace(control=SimpleNamespace(value=0.77)))

    assert view._settings.desktop_audio.vad_speech_threshold == 0.77
    assert view._peer_vad_field.value == "0.77"
    assert view._peer_vad_slider.label == "0.77"
    assert slider_page.updated == [view._peer_vad_slider]
    assert len(changed) == 1
    assert changed[0] is not settings
    assert changed[0].desktop_audio.vad_speech_threshold == 0.77


def test_audio_change_messages_use_basic_runtime_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    settings.audio.input_device = "Old Mic"
    settings.desktop_audio.output_device = "Old Speakers"
    basic_messages: list[str] = []
    detailed_messages: list[str] = []
    changed: list[AppSettings] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.runtime_log_basic = lambda message, *, level=logging.INFO: basic_messages.append(message)
    view.runtime_log_detailed = lambda message, *, level=logging.INFO: detailed_messages.append(
        message
    )
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._audio_settings.host_api = "MME"
    view._audio_settings.microphone = "New Mic"
    view._audio_settings.desktop_output_device = "New Speakers"
    view._on_audio_change()

    expected_messages = [
        f"[Settings] Audio Host changed: {WINDOWS_WASAPI_COMPATIBILITY_HOST_API} -> MME",
        "[Settings] Microphone changed: Old Mic -> New Mic",
        "[Settings] Desktop loopback output changed: Old Speakers -> New Speakers",
    ]
    audio_change_prefixes = (
        "[Settings] Audio Host changed:",
        "[Settings] Microphone changed:",
        "[Settings] Desktop loopback output changed:",
    )

    assert all(message in basic_messages for message in expected_messages)
    assert not any(message.startswith(audio_change_prefixes) for message in detailed_messages)
    assert view._settings.audio.input_host_api == "MME"
    assert view._settings.audio.input_device == "New Mic"
    assert view._settings.desktop_audio.output_device == "New Speakers"
    assert settings.audio.input_host_api == "MME"
    assert settings.audio.input_device == "New Mic"
    assert settings.desktop_audio.output_device == "New Speakers"
    assert len(changed) == 1
    assert changed[0] is not settings
    assert changed[0].audio.input_host_api == "MME"
    assert changed[0].audio.input_device == "New Mic"
    assert changed[0].desktop_audio.output_device == "New Speakers"


def test_retired_translation_policy_controls_are_not_constructed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.stt.low_latency_mode = False
    settings.ui.integrated_context_enabled = False
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert not hasattr(view, "_low_latency_card")
    assert not hasattr(view, "_low_latency_text")
    assert not hasattr(view, "_on_low_latency_click")
    assert not hasattr(view, "_integrated_context_card")
    assert not hasattr(view, "_integrated_context_button")
    assert not hasattr(view, "_on_integrated_context_click")


def test_peer_qwen_region_control_is_removed_before_peer_translation_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.QWEN_ASR
    settings.ui.peer_translation_enabled = False

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert not hasattr(view, "_peer_qwen_region_label")
    assert not hasattr(view, "_peer_qwen_region_text")


def test_update_api_visibility_keeps_peer_auth_controls_visible_when_peer_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    settings.provider.peer_stt = STTProviderName.DEEPGRAM
    settings.provider.llm = LLMProviderName.GEMINI
    settings.ui.peer_translation_enabled = False

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._deepgram_key.visible is True
    assert view._soniox_key.visible is False
    assert view._google_key.visible is True


def test_update_api_visibility_keeps_peer_qwen_credentials_visible_when_peer_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    settings.provider.peer_stt = STTProviderName.QWEN_ASR
    settings.provider.llm = LLMProviderName.GEMINI
    settings.ui.peer_translation_enabled = False
    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._alibaba_key_beijing.visible is True
    assert view._alibaba_key_singapore.visible is False


def test_peer_qwen_region_override_controls_are_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.QWEN_ASR
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    assert not hasattr(view, "_on_peer_qwen_region_selected")
    assert not hasattr(view, "_peer_qwen_region_text")


def test_peer_soniox_model_override_controls_are_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.SONIOX
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    assert not hasattr(view, "_on_peer_soniox_model_selected")
    assert not hasattr(view, "_peer_soniox_model_text")


def test_update_api_visibility_includes_enabled_peer_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    settings.provider.peer_stt = STTProviderName.DEEPGRAM
    settings.provider.llm = LLMProviderName.GEMINI
    settings.ui.peer_translation_enabled = True

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._deepgram_key.visible is True
    assert view._google_key.visible is True


def test_update_api_visibility_uses_shared_qwen_region_for_peer_and_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.QWEN_ASR
    settings.provider.peer_stt = STTProviderName.QWEN_ASR
    settings.ui.peer_translation_enabled = True
    settings.qwen.region = QwenRegion.BEIJING

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._alibaba_key_beijing.visible is True
    assert view._alibaba_key_singapore.visible is False


def test_update_api_visibility_shows_shared_qwen_region_for_peer_qwen_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.SONIOX
    settings.provider.peer_stt = STTProviderName.QWEN_ASR
    settings.provider.llm = LLMProviderName.GEMINI
    settings.ui.peer_translation_enabled = True
    settings.qwen.region = QwenRegion.BEIJING

    view, _ = _make_settings_view(monkeypatch, settings=settings)
    view._update_api_visibility()

    assert view._soniox_key.visible is True
    assert view._qwen_region_btn.visible is True
    assert view._alibaba_key_beijing.visible is True
    assert view._alibaba_key_singapore.visible is False


def test_on_peer_stt_selected_refreshes_api_visibility_and_redraws_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    settings.provider.peer_stt = STTProviderName.DEEPGRAM
    settings.ui.peer_translation_enabled = True

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    attach_dummy_page(monkeypatch, view)

    api_key_updates: list[str] = []
    monkeypatch.setattr(
        type(view._peer_stt_text),
        "update",
        lambda self: api_key_updates.append("peer_stt_text"),
    )
    monkeypatch.setattr(
        type(view._api_keys_column),
        "update",
        lambda self: api_key_updates.append("api_keys_column"),
    )
    monkeypatch.setattr(
        view._qwen_region_btn,
        "update",
        lambda: api_key_updates.append("qwen_region_btn"),
    )

    view._on_peer_stt_selected(STTProviderName.SONIOX.value)

    pending = view.build_provider_apply_settings()

    assert settings.provider.peer_stt == STTProviderName.DEEPGRAM
    assert pending is not None
    assert pending.provider.peer_stt == STTProviderName.SONIOX
    assert view._peer_stt_text.content.value == t("provider.soniox")
    assert "peer_stt_text" in api_key_updates
    assert "qwen_region_btn" in api_key_updates
    assert "api_keys_column" in api_key_updates
    assert view._peer_auto_languages_card.visible is True


def test_peer_auto_detection_languages_card_is_visible_only_for_peer_soniox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    view, _ = _make_settings_view(monkeypatch, settings=settings)

    view._update_api_visibility()
    assert view._peer_auto_languages_card.visible is False

    settings.provider.peer_stt = STTProviderName.SONIOX
    settings.languages.peer_expected_languages = ["ja"]
    view._update_api_visibility()

    assert view._peer_auto_languages_card.visible is True
    assert view._peer_auto_languages_editor._terms == ["ja"]


def test_overlay_display_toggles_update_persistent_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings_calls: list[AppSettings] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: settings_calls.append(incoming)
    attach_dummy_page(monkeypatch, view)

    view._on_overlay_translation_click(None)
    view._on_overlay_peer_original_click(None)

    assert settings.overlay.show_translation is False
    assert settings.overlay.show_peer_original is False
    assert len(settings_calls) == 2
    assert all(incoming is not settings for incoming in settings_calls)
    assert settings_calls[-1].overlay.show_translation is False
    assert settings_calls[-1].overlay.show_peer_original is False


def test_overlay_single_action_cards_use_broad_value_slot_click_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    desktop_settings = AppSettings()
    desktop_settings.overlay.target = "desktop"
    desktop_view, _ = _make_settings_view(monkeypatch)
    desktop_view.load_from_settings(desktop_settings, config_path=Path("settings.json"))

    cases = [
        (
            _overlay_tab_card(view, t("settings.overlay.caption_location")),
            view._overlay_target_button,
            t("settings.overlay.target.steamvr"),
        ),
        (
            _overlay_tab_card(view, t("settings.overlay.show_translation")),
            view._overlay_translation_button,
            t("settings.option.on"),
        ),
        (
            _overlay_tab_card(view, t("settings.overlay.show_peer_original")),
            view._overlay_peer_original_button,
            t("settings.option.on"),
        ),
        (
            _overlay_tab_card(view, t("settings.overlay.calibration.anchor")),
            view._overlay_anchor_button,
            t("settings.overlay.calibration.anchor.head_locked"),
        ),
        (
            _overlay_tab_card(desktop_view, t("settings.overlay.desktop.size.title")),
            desktop_view._desktop_overlay_size_button,
            t("settings.overlay.desktop.size.option.medium"),
        ),
        (
            _overlay_tab_card(desktop_view, t("settings.overlay.desktop.lock.title")),
            desktop_view._desktop_overlay_lock_button,
            t("settings.overlay.desktop.lock.value.move"),
        ),
    ]

    for card, control, expected in cases:
        value_control = _wrapped_card_column(card).controls[1].content

        assert value_control is control
        assert isinstance(control, ft.Container)
        assert control.expand is True
        assert control.content.value == expected
        assert control.content.color == settings_view.COLOR_ON_BACKGROUND


def test_overlay_text_scale_modal_selection_updates_settings_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_overlay_text_scale_selected("large")

    assert settings.overlay.calibration.text_scale == 1.2
    assert view._overlay_text_scale_text.content.value == t(
        "settings.overlay.calibration.text_scale.large"
    )
    assert changed == [settings]


def test_desktop_gui_caption_location_selector_updates_settings_with_localized_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "steamvr"
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    attach_dummy_page(monkeypatch, view)

    captured: dict[str, object] = {}

    class DummyModal:
        def __init__(self, _page, title, options, on_select, *, show_description=False):
            captured["title"] = title
            captured["options"] = options
            captured["on_select"] = on_select
            captured["show_description"] = show_description

        def open(self, current: str) -> None:
            captured["current"] = current

    monkeypatch.setattr(settings_view, "SettingsModal", DummyModal)

    view._on_overlay_target_click(None)

    assert captured["title"] == t("settings.overlay.caption_location")
    assert captured["show_description"] is True
    assert [option.value for option in captured["options"]] == ["steamvr", "desktop"]
    assert [option.label for option in captured["options"]] == [
        t("settings.overlay.target.steamvr"),
        t("settings.overlay.target.desktop"),
    ]
    assert captured["current"] == "steamvr"

    captured_on_select = captured["on_select"]
    assert callable(captured_on_select)
    captured_on_select("desktop")

    assert settings.overlay.target == "desktop"
    assert view._overlay_target_button.content.value == t("settings.overlay.target.desktop")
    assert changed == [settings]


def test_desktop_gui_product_standard_cards_show_current_values_and_desktop_only_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale("ko")
        settings = AppSettings()
        settings.overlay.target = "desktop"
        settings.overlay.desktop_flet.size_preset = "large"
        settings.overlay.desktop_flet.locked = False
        view, _ = _make_settings_view(monkeypatch)
        view.load_from_settings(settings, config_path=Path("settings.json"))

        overlay_titles = _overlay_tab_card_titles(view)
        reset_card = _overlay_tab_card(view, t("settings.overlay.position_reset.desktop.title"))
        reset_actions = _wrapped_card_column(reset_card).controls[1].content

        assert t("settings.overlay.desktop.size.title") in overlay_titles
        assert t("settings.overlay.desktop.background_alpha.title") in overlay_titles
        assert t("settings.overlay.desktop.lock.title") in overlay_titles
        assert t("settings.overlay.position_reset.desktop.title") in overlay_titles
        assert t("settings.overlay.calibration.anchor") not in overlay_titles
        assert t("settings.overlay.calibration.distance") not in overlay_titles
        assert t("settings.overlay.calibration.offset_x") not in overlay_titles
        assert t("settings.overlay.calibration.offset_y") not in overlay_titles
        assert t("settings.overlay.calibration.text_scale") not in overlay_titles
        assert t("settings.overlay.status.off") not in overlay_titles
        assert all(row.visible is False for row in view._overlay_vr_rows)
        assert all(row.visible is True for row in view._overlay_desktop_rows)
        assert view._desktop_overlay_size_button.content.value == t(
            "settings.overlay.desktop.size.option.large"
        )
        assert view._desktop_overlay_background_alpha_value_text.value == "40%"
        assert view._desktop_overlay_lock_button.content.value == t(
            "settings.overlay.desktop.lock.value.move"
        )
        assert view._desktop_overlay_status_card.visible is False
        assert view._desktop_overlay_recovery_row.visible is False

        assert reset_actions is view._overlay_desktop_reset_button
        assert view._overlay_desktop_reset_button.on_click is not None
        assert view._overlay_desktop_reset_button.content.value == t(
            "settings.overlay.position_reset.action.desktop"
        )

        visible_overlay_labels: list[str] = []
        for row in _subtab_controls(view, "overlay"):
            if getattr(row, "visible", True) is not False:
                visible_overlay_labels.extend(_control_labels(row))
        normal_technical_fragments = ("bridge", "renderer", "runtime", "logs")
        assert not [
            (label, fragment)
            for label in visible_overlay_labels
            for fragment in normal_technical_fragments
            if fragment in label.lower()
        ]
    finally:
        i18n_module.set_locale(previous_locale)


def test_overlay_tab_switches_visible_cards_when_caption_location_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "steamvr"
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_overlay_target_selected("desktop")

    overlay_titles = _overlay_tab_card_titles(view)
    assert t("settings.overlay.desktop.size.title") in overlay_titles
    assert t("settings.overlay.desktop.background_alpha.title") in overlay_titles
    assert t("settings.overlay.desktop.lock.title") in overlay_titles
    assert t("settings.overlay.position_reset.desktop.title") in overlay_titles
    assert t("settings.overlay.calibration.anchor") not in overlay_titles
    assert all(row.visible is False for row in view._overlay_vr_rows)
    assert all(row.visible is True for row in view._overlay_desktop_rows)


def test_desktop_gui_background_transparency_card_adjusts_in_ten_percent_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "desktop"
    settings.overlay.desktop_flet.visual.background_alpha = 0.5
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    assert view._desktop_overlay_background_alpha_value_text.value == "50%"

    view._on_desktop_overlay_background_alpha_step(0.1)

    assert view._settings is not None
    assert view._settings.overlay.desktop_flet.visual.background_alpha == pytest.approx(0.4)
    assert view._desktop_overlay_background_alpha_value_text.value == "60%"

    view._on_desktop_overlay_background_alpha_step(-0.1)
    view._on_desktop_overlay_background_alpha_step(-0.1)

    assert view._settings.overlay.desktop_flet.visual.background_alpha == pytest.approx(0.6)
    assert view._desktop_overlay_background_alpha_value_text.value == "40%"
    assert [
        incoming.overlay.desktop_flet.visual.background_alpha for incoming in changed
    ] == pytest.approx([0.4, 0.5, 0.6])


def test_desktop_gui_background_alpha_emits_copy_without_mutating_loaded_settings_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "desktop"
    settings.overlay.desktop_flet.visual.background_alpha = 0.5
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_desktop_overlay_background_alpha_step(0.1)

    assert settings.overlay.desktop_flet.visual.background_alpha == pytest.approx(0.5)
    assert changed
    assert changed[-1] is not settings
    assert changed[-1].overlay.desktop_flet.visual.background_alpha == pytest.approx(0.4)
    assert view._settings is not changed[-1]
    assert view._settings.overlay.desktop_flet.visual.background_alpha == pytest.approx(0.4)


def test_desktop_gui_background_transparency_card_clamps_to_zero_and_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "desktop"
    settings.overlay.desktop_flet.visual.background_alpha = 0.05
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_desktop_overlay_background_alpha_step(0.1)
    view._on_desktop_overlay_background_alpha_step(0.1)

    assert view._settings is not None
    assert view._settings.overlay.desktop_flet.visual.background_alpha == pytest.approx(0.0)
    assert view._desktop_overlay_background_alpha_value_text.value == "100%"

    view._settings.overlay.desktop_flet.visual.background_alpha = 0.95
    view._sync_desktop_overlay_main_controls()
    view._on_desktop_overlay_background_alpha_step(-0.1)
    view._on_desktop_overlay_background_alpha_step(-0.1)

    assert view._settings.overlay.desktop_flet.visual.background_alpha == pytest.approx(1.0)
    assert view._desktop_overlay_background_alpha_value_text.value == "0%"


def test_desktop_gui_size_selection_persists_and_emits_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "desktop"
    settings.overlay.desktop_flet.size_preset = "medium"
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_desktop_overlay_size_selected("xlarge")

    assert settings.overlay.desktop_flet.size_preset == "xlarge"
    assert view._desktop_overlay_size_button.content.value == t(
        "settings.overlay.desktop.size.option.xlarge"
    )
    assert changed == [settings]


def test_desktop_gui_size_selection_uses_runtime_callback_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.desktop_flet.size_preset = "medium"
    changed: list[AppSettings] = []
    runtime_size_requests: list[str] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    view.on_desktop_overlay_size_change = runtime_size_requests.append

    view._on_desktop_overlay_size_selected("xlarge")

    assert runtime_size_requests == ["xlarge"]
    assert changed == []
    assert settings.overlay.desktop_flet.size_preset == "medium"
    assert view._desktop_overlay_size_button.content.value == t(
        "settings.overlay.desktop.size.option.xlarge"
    )


def test_desktop_gui_runtime_size_selection_is_preserved_on_next_settings_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.desktop_flet.size_preset = "medium"
    changed: list[AppSettings] = []
    runtime_size_requests: list[str] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    view.on_desktop_overlay_size_change = runtime_size_requests.append

    view._on_desktop_overlay_size_selected("xlarge")
    view._on_overlay_translation_selected("off")

    assert runtime_size_requests == ["xlarge"]
    assert settings.overlay.desktop_flet.size_preset == "medium"
    assert changed
    assert changed[-1] is not settings
    assert changed[-1].overlay.desktop_flet.size_preset == "xlarge"
    assert changed[-1].overlay.show_translation is False


def test_desktop_gui_size_runtime_callback_can_return_to_previous_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.desktop_flet.size_preset = "medium"
    runtime_size_requests: list[str] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_desktop_overlay_size_change = runtime_size_requests.append

    view._on_desktop_overlay_size_selected("xlarge")
    view._on_desktop_overlay_size_selected("medium")

    assert runtime_size_requests == ["xlarge", "medium"]
    assert view._desktop_overlay_size_button.content.value == t(
        "settings.overlay.desktop.size.option.medium"
    )


def test_desktop_gui_lock_card_displays_move_for_legacy_saved_lock_when_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "desktop"
    settings.overlay.desktop_flet.locked = True
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.move"
    )
    assert view._desktop_overlay_lock_button.content.value != t(
        "settings.overlay.desktop.action.lock_captions"
    )
    assert settings.overlay.desktop_flet.locked is True
    assert changed == []


def test_desktop_gui_runtime_lock_callback_is_authoritative_without_settings_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "desktop"
    settings.overlay.desktop_flet.locked = False
    changed: list[AppSettings] = []
    runtime_lock_requests: list[bool] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    view.on_desktop_overlay_lock_change = runtime_lock_requests.append
    view.set_overlay_runtime_state(
        "connected",
        overlay_target="desktop",
        desktop_captions_locked=False,
    )

    view._on_desktop_overlay_lock_selected("locked")

    assert runtime_lock_requests == [True]
    assert changed == []
    assert settings.overlay.desktop_flet.locked is False
    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.locked"
    )

    view._on_overlay_peer_original_selected("off")

    assert changed
    assert changed[-1].overlay.desktop_flet.locked is False
    assert changed[-1].overlay.show_peer_original is False
    assert "locked" not in to_dict(changed[-1])["overlay"]["desktop_flet"]


def test_desktop_gui_non_desktop_runtime_lock_sync_displays_move_for_legacy_saved_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "steamvr"
    settings.overlay.desktop_flet.locked = True
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    view.on_desktop_overlay_lock_change = lambda _locked: None

    view.set_overlay_runtime_state(
        "connected",
        overlay_target="steamvr",
        desktop_captions_locked=False,
    )
    view._on_overlay_translation_selected("off")

    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.move"
    )
    assert changed
    assert changed[-1].overlay.show_translation is False
    assert "locked" not in to_dict(changed[-1])["overlay"]["desktop_flet"]


def test_desktop_gui_runtime_lock_notification_controls_next_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "desktop"
    settings.overlay.desktop_flet.locked = False
    runtime_lock_requests: list[bool] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_desktop_overlay_lock_change = runtime_lock_requests.append

    view.set_overlay_runtime_state(
        "running",
        overlay_target="desktop",
        desktop_captions_locked=True,
    )
    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.locked"
    )

    view._on_desktop_overlay_lock_click(None)

    assert runtime_lock_requests == [False]
    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.move"
    )


def test_desktop_gui_clears_pending_runtime_lock_when_runtime_becomes_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "desktop"
    runtime_lock_requests: list[bool] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_desktop_overlay_lock_change = runtime_lock_requests.append
    view.set_overlay_runtime_state(
        "connected",
        overlay_target="desktop",
        desktop_captions_locked=False,
    )

    view._on_desktop_overlay_lock_click(None)

    assert runtime_lock_requests == [True]
    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.locked"
    )

    view.set_overlay_runtime_state(
        "off",
        overlay_target="desktop",
        desktop_captions_locked=True,
    )

    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.move"
    )

    view._on_desktop_overlay_lock_click(None)

    assert runtime_lock_requests == [True]
    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.move"
    )


@pytest.mark.parametrize("state", ["starting", "failed", "stopping", "off"])
def test_desktop_gui_ignores_stale_runtime_lock_when_desktop_runtime_is_not_active(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    settings = AppSettings()
    settings.overlay.target = "desktop"
    runtime_lock_requests: list[bool] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_desktop_overlay_lock_change = runtime_lock_requests.append

    view.set_overlay_runtime_state(
        state,
        overlay_target="desktop",
        desktop_captions_locked=True,
    )

    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.move"
    )

    view._on_desktop_overlay_lock_click(None)

    assert runtime_lock_requests == []
    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.move"
    )


def test_overlay_position_reset_card_separates_vr_and_desktop_reset_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.calibration.distance = 1.2
    settings.overlay.calibration.offset_y = 0.5
    settings.overlay.desktop_flet.position.x = 80
    settings.overlay.desktop_flet.position.y = 90
    settings.overlay.desktop_flet.size_preset = "large"
    settings.overlay.desktop_flet.locked = True
    settings.overlay.desktop_flet.visual.background_alpha = 0.44
    changed: list[AppSettings] = []
    desktop_resets: list[bool] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.set_overlay_calibration(settings.overlay.calibration)
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_overlay_position_reset(None)

    assert settings.overlay.calibration.distance == OverlayCalibration().distance
    assert settings.overlay.desktop_flet.position.x == 80
    assert settings.overlay.desktop_flet.position.y == 90
    assert settings.overlay.desktop_flet.size_preset == "large"
    assert settings.overlay.desktop_flet.locked is True
    assert desktop_resets == []

    view._on_desktop_overlay_position_reset(None)

    assert settings.overlay.calibration.offset_y == OverlayCalibration().offset_y
    assert settings.overlay.desktop_flet.position.x is None
    assert settings.overlay.desktop_flet.position.y is None
    assert settings.overlay.desktop_flet.size_preset == "large"
    assert settings.overlay.desktop_flet.locked is False
    assert settings.overlay.desktop_flet.visual.background_alpha == 0.44
    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.move"
    )
    assert desktop_resets == []
    assert len(changed) == 2
    assert all(incoming is not settings for incoming in changed)
    assert changed[-1].overlay.desktop_flet.position.x is None
    assert changed[-1].overlay.desktop_flet.position.y is None
    assert changed[-1].overlay.desktop_flet.locked is False


def test_desktop_gui_runtime_position_reset_defers_to_callback_without_stale_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.desktop_flet.position.x = 80
    settings.overlay.desktop_flet.position.y = 90
    settings.overlay.desktop_flet.size_preset = "large"
    settings.overlay.desktop_flet.locked = True
    settings.overlay.desktop_flet.visual.background_alpha = 0.44
    changed: list[AppSettings] = []
    desktop_resets: list[bool] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    view.on_desktop_overlay_position_reset = lambda: desktop_resets.append(True)

    view._on_desktop_overlay_position_reset(None)

    assert desktop_resets == [True]
    assert changed == []
    assert settings.overlay.desktop_flet.position.x == 80
    assert settings.overlay.desktop_flet.position.y == 90
    assert settings.overlay.desktop_flet.locked is True
    assert view._desktop_overlay_lock_button.content.value == t(
        "settings.overlay.desktop.lock.value.move"
    )

    view._on_overlay_translation_selected("off")

    assert changed
    assert changed[-1] is not settings
    assert changed[-1].overlay.desktop_flet.position.x is None
    assert changed[-1].overlay.desktop_flet.position.y is None
    assert changed[-1].overlay.desktop_flet.size_preset == "large"
    assert changed[-1].overlay.desktop_flet.locked is False
    assert changed[-1].overlay.desktop_flet.visual.background_alpha == 0.44


def test_overlay_failure_i18n_desktop_gui_recovery_actions_are_user_facing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale("en")
        settings = AppSettings()
        settings.overlay.target = "desktop"
        recovery_actions: list[str] = []
        details_opened: list[bool] = []
        view, _ = _make_settings_view(monkeypatch)
        view.load_from_settings(settings, config_path=Path("settings.json"))
        view.on_desktop_overlay_recovery_action = recovery_actions.append
        view.on_view_logs = lambda: details_opened.append(True)

        view.set_overlay_runtime_state(
            "failed",
            failure_reason="window_configuration_failed",
            overlay_target="desktop",
            desktop_captions_locked=False,
        )

        failure_labels = _control_labels(view._desktop_overlay_status_card)
        assert view._desktop_overlay_status_card.visible is True
        assert t("settings.overlay.desktop.status.failed") in failure_labels
        assert t("settings.overlay.desktop.recovery.message.reopen") in failure_labels
        assert t("settings.overlay.desktop.recovery.action.reopen") in failure_labels
        assert t("settings.overlay.desktop.recovery.action.view_details") in failure_labels

        raw_key_labels = [label for label in failure_labels if "settings." in label]
        assert raw_key_labels == []
        technical_fragments = ("executable", "bridge", "renderer", "runtime", "logs")
        assert not [
            (label, fragment)
            for label in failure_labels
            for fragment in technical_fragments
            if fragment in label.lower()
        ]

        view._on_desktop_overlay_primary_action(None)
        view._on_desktop_overlay_view_logs(None)

        assert recovery_actions == ["reopen"]
        assert details_opened == [True]

        view.set_overlay_runtime_state(
            "failed",
            failure_reason="bridge_auth_failed",
            overlay_target="desktop",
            desktop_captions_locked=False,
        )
        retry_labels = _control_labels(view._desktop_overlay_status_card)
        view._on_desktop_overlay_primary_action(None)

        assert t("settings.overlay.desktop.recovery.message.retry") in retry_labels
        assert t("settings.overlay.desktop.recovery.action.retry") in retry_labels
        assert not [label for label in retry_labels if "settings." in label]
        assert not [
            (label, fragment)
            for label in retry_labels
            for fragment in technical_fragments
            if fragment in label.lower()
        ]
        assert recovery_actions == ["reopen", "retry"]
    finally:
        i18n_module.set_locale(previous_locale)


def test_audio_change_updates_desktop_loopback_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._audio_settings.desktop_output_device = "Speakers (Loopback)"
    view._on_audio_change()
    view._peer_vad_field.value = "0.72"
    view._on_peer_vad_threshold_change(SimpleNamespace(control=view._peer_vad_field))
    view._peer_hangover_field.value = "950"
    view._on_peer_hangover_change(SimpleNamespace(control=view._peer_hangover_field))
    view._peer_pre_roll_field.value = "420"
    view._on_peer_pre_roll_change(SimpleNamespace(control=view._peer_pre_roll_field))

    assert view._settings.desktop_audio.output_device == "Speakers (Loopback)"
    assert view._settings.desktop_audio.vad_speech_threshold == 0.72
    assert view._settings.desktop_audio.vad_hangover_ms == 950
    assert view._settings.desktop_audio.vad_pre_roll_ms == 420
    assert settings.desktop_audio.output_device == "Speakers (Loopback)"
    assert settings.desktop_audio.vad_speech_threshold == 0.72
    assert settings.desktop_audio.vad_hangover_ms == 950
    assert settings.desktop_audio.vad_pre_roll_ms == 420
    assert len(changed) == 4
    assert all(incoming is not settings for incoming in changed)
    assert changed[-1].desktop_audio.output_device == "Speakers (Loopback)"
    assert changed[-1].desktop_audio.vad_speech_threshold == 0.72
    assert changed[-1].desktop_audio.vad_hangover_ms == 950
    assert changed[-1].desktop_audio.vad_pre_roll_ms == 420


def test_general_tab_keeps_fixed_three_slot_rows_with_vrchat_osc_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.ui.components.settings.settings_unit_card import SettingsUnitCard

    view, _ = _make_settings_view(monkeypatch)
    general_controls = _subtab_controls(view, "general")

    assert len(general_controls) == 4
    assert {len(control.content.controls) for control in general_controls} == {3}
    assert _row_card_titles(general_controls[0]) == [
        t("settings.section.ui"),
        t("settings.chatbox_include_source"),
        t("settings.osc.connection.title"),
    ]
    assert _row_card_titles(general_controls[1]) == [
        t("settings.audio_host_api"),
        t("settings.section.microphone_audio"),
        t("settings.section.loopback_audio"),
    ]
    assert _row_card_titles(general_controls[2]) == [
        t("settings.microphone_test"),
        t("settings.section.self_vad_sensitivity"),
        t("settings.section.peer_vad_sensitivity"),
    ]
    assert _row_card_titles(general_controls[3]) == [
        t("settings.clipboard_auto_translate"),
        t("settings.vrc_mic_intercept"),
        t("settings.telemetry.title"),
    ]

    osc_card = general_controls[0].content.controls[2]
    assert osc_card is view._vrchat_osc_card
    assert isinstance(osc_card, SettingsUnitCard)
    assert osc_card.height == SettingsUnitCard.DEFAULT_HEIGHT
    assert osc_card.expand is True
    osc_column = _wrapped_card_column(osc_card)
    value_slot = osc_column.controls[1]
    assert isinstance(value_slot, ft.Container)
    assert value_slot.content is view._osc_connection_text
    assert view._osc_connection_text.content.value == t("settings.osc.mode.automatic")


def test_api_translation_connection_row_keeps_empty_fast_translation_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.ui.components.settings.settings_unit_card import SettingsUnitCard
    from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper

    view, _ = _make_settings_view(monkeypatch)
    cards = _row_cards(view._translation_connection_row)

    assert len(cards) == 3
    assert _row_card_titles(view._translation_connection_row) == [
        t("settings.translation_connection"),
        t("settings.fallback"),
    ]
    assert {card.height for card in cards} == {SettingsUnitCard.DEFAULT_HEIGHT}
    assert all(card.expand is True for card in cards)

    empty_card = cards[0]
    empty_content = _wrapped_card_column(empty_card)

    assert isinstance(empty_card, SharedCardWrapper)
    assert not isinstance(empty_card, SettingsUnitCard)
    assert empty_card.ignore_interactions is True
    assert empty_card.on_click is None
    assert empty_card.on_hover is None
    assert isinstance(empty_content, ft.Container)
    assert empty_content.content is None


@pytest.mark.parametrize("locale", ["en", "ko", "ja", "ru", "zh-CN"])
def test_general_osc_card_is_locale_independent(
    monkeypatch: pytest.MonkeyPatch,
    locale: str,
) -> None:
    old_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale(locale)
        view, _ = _make_settings_view(monkeypatch)
        general_row = _subtab_controls(view, "general")[0]
        api_row = view._translation_connection_row

        assert len(_row_cards(general_row)) == 3
        assert len(_row_cards(api_row)) == 3
        assert _row_card_titles(general_row) == [
            t("settings.section.ui"),
            t("settings.chatbox_include_source"),
            t("settings.osc.connection.title"),
        ]
        assert _row_card_titles(api_row) == [
            t("settings.translation_connection"),
            t("settings.fallback"),
        ]
        assert _control_labels(_row_cards(api_row)[0]) == []
    finally:
        i18n_module.set_locale(old_locale)


def test_microphone_test_card_click_invokes_start_callback_without_modal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    calls: list[str] = []
    view.on_start_microphone_test = lambda: calls.append("start")
    modal_calls: list[str] = []

    class DummyModal:
        def __init__(self, *_args, **_kwargs) -> None:
            modal_calls.append("created")

    monkeypatch.setattr(settings_view, "SettingsModal", DummyModal)

    view._on_microphone_test_click(None)

    assert calls == ["start"]
    assert modal_calls == []


def test_microphone_test_card_clicks_always_request_start_without_inline_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    card = _general_tab_card(view, t("settings.microphone_test"))
    calls: list[str] = []
    view.on_start_microphone_test = lambda: calls.append("start")

    assert not any(isinstance(node, ft.Slider) for node in _iter_control_tree(card))
    assert view._microphone_test_text.content.value == t("settings.microphone_test.action")

    view._on_microphone_test_click(None)
    assert calls == ["start"]

    view._on_microphone_test_click(None)
    assert calls == ["start", "start"]
    assert view._microphone_test_text.content.value == t("settings.microphone_test.action")


def test_api_tab_places_independent_managed_key_card_above_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    api_controls = _subtab_controls(view, "api")

    assert len(api_controls) == 9
    assert _row_card_titles(api_controls[0]) == [
        t("settings.section.stt"),
        t("settings.section.peer_stt"),
        t("settings.section.translation"),
    ]
    assert _row_card_titles(api_controls[1]) == [
        t("settings.translation_connection"),
        t("settings.fallback"),
    ]
    assert api_controls[2].content is view._http_extension_row
    assert _row_card_titles(api_controls[2]) == [
        t("settings.http_extension.title"),
        t("settings.http_extension.path"),
        t("settings.http_extension.refresh"),
    ]
    assert _row_card_titles(api_controls[3]) == [t("settings.gpu_device.title")]
    assert view._gpu_device_card.visible is False
    assert _row_card_titles(api_controls[4]) == [t("settings.local_llm.connection")]
    assert api_controls[4] is view._local_llm_connection_card
    assert api_controls[5] is view._custom_stt_connection_card
    assert _row_card_titles(api_controls[5]) == [t("settings.custom_stt.title")]
    assert api_controls[6] is view._managed_key_card
    assert _row_card_titles(api_controls[6]) == [t("settings.managed_key.title")]
    assert api_controls[7] is view._peer_auto_languages_card
    assert api_controls[8] is not view._api_keys_column
    assert _row_card_titles(api_controls[8]) == [t("settings.section.api_keys")]


def test_api_tab_primary_value_typography_is_consistent_across_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert {
        _container_text_size(view._stt_text),
        _container_text_size(view._peer_stt_text),
        _container_text_size(view._llm_text),
        _container_text_size(view._translation_connection_text),
        _container_text_size(view._openrouter_fallback_text),
    } == {28}


def test_general_tab_host_api_card_exposes_host_api_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    host_api_card = _general_tab_card(view, t("settings.audio_host_api"))
    host_api_labels = _control_labels(host_api_card)

    assert t("settings.desktop_audio.output_device") not in host_api_labels
    assert _tree_contains_control(host_api_card, view._audio_host_api_text)


def test_general_tab_microphone_audio_card_exposes_microphone_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    mic_audio_card = _general_tab_card(view, t("settings.section.microphone_audio"))
    mic_audio_labels = _control_labels(mic_audio_card)

    assert t("settings.audio_host_api") not in mic_audio_labels
    assert t("settings.desktop_audio.output_device") not in mic_audio_labels
    assert _tree_contains_control(mic_audio_card, view._mic_audio_text)
    assert not _tree_contains_control(mic_audio_card, view._audio_host_api_text)


def test_general_tab_loopback_audio_card_exposes_loopback_device_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    loopback_audio_card = _general_tab_card(view, t("settings.section.loopback_audio"))
    loopback_audio_labels = _control_labels(loopback_audio_card)

    assert t("settings.audio_host_api") not in loopback_audio_labels
    assert _tree_contains_control(loopback_audio_card, view._loopback_audio_text)


def test_general_tab_self_vad_card_contains_only_self_vad_slider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    self_vad_card = _general_tab_card(view, t("settings.section.self_vad_sensitivity"))

    assert _tree_contains_control(self_vad_card, view._vad_slider)
    assert not _tree_contains_control(self_vad_card, view._peer_vad_field)
    assert not _tree_contains_control(self_vad_card, view._peer_hangover_field)
    assert not _tree_contains_control(self_vad_card, view._peer_pre_roll_field)


def test_general_tab_peer_vad_card_contains_peer_fields_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    peer_vad_card = _general_tab_card(view, t("settings.section.peer_vad_sensitivity"))
    peer_vad_labels = _control_labels(peer_vad_card)

    assert t("settings.vad.peer") not in peer_vad_labels
    assert t("settings.vad.peer_hangover_ms") not in peer_vad_labels
    assert t("settings.vad.peer_pre_roll_ms") not in peer_vad_labels
    assert _tree_contains_control(peer_vad_card, view._peer_vad_slider)
    assert not _tree_contains_control(peer_vad_card, view._peer_vad_field)
    assert not _tree_contains_control(peer_vad_card, view._peer_hangover_field)
    assert not _tree_contains_control(peer_vad_card, view._peer_pre_roll_field)
    assert not _tree_contains_control(peer_vad_card, view._vad_slider)


def test_overlay_tab_uses_target_specific_unit_card_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    overlay_titles = _overlay_tab_card_titles(view)
    overlay_controls = _subtab_controls(view, "overlay")

    assert overlay_titles == [
        t("settings.overlay.caption_location"),
        t("settings.overlay.show_translation"),
        t("settings.overlay.show_peer_original"),
        t("settings.overlay.calibration.anchor"),
        t("settings.overlay.calibration.distance"),
        t("settings.overlay.calibration.offset_x"),
        t("settings.overlay.calibration.offset_y"),
        t("settings.overlay.calibration.text_scale"),
        t("settings.overlay.position_reset.vr.title"),
    ]
    assert len(overlay_controls) == 6
    assert _row_card_titles(overlay_controls[0]) == [
        t("settings.overlay.caption_location"),
        t("settings.overlay.show_translation"),
        t("settings.overlay.show_peer_original"),
    ]
    assert _row_card_titles(overlay_controls[1]) == [
        t("settings.overlay.calibration.anchor"),
        t("settings.overlay.calibration.distance"),
        t("settings.overlay.calibration.offset_x"),
    ]
    assert _row_card_titles(overlay_controls[2]) == [
        t("settings.overlay.calibration.offset_y"),
        t("settings.overlay.calibration.text_scale"),
        t("settings.overlay.position_reset.vr.title"),
    ]
    assert _row_card_titles(overlay_controls[3]) == [
        t("settings.overlay.desktop.size.title"),
        t("settings.overlay.desktop.lock.title"),
        t("settings.overlay.desktop.background_alpha.title"),
    ]
    assert _row_card_titles(overlay_controls[4]) == [
        t("settings.overlay.position_reset.desktop.title"),
    ]
    assert len(_layout_cards(overlay_controls[4])) == 3
    assert [getattr(card, "visible", True) for card in _layout_cards(overlay_controls[4])] == [
        True,
        True,
        True,
    ]
    assert overlay_controls[1].visible is True
    assert overlay_controls[2].visible is True
    assert overlay_controls[3].visible is False
    assert overlay_controls[4].visible is False
    assert overlay_controls[5].visible is False


def test_apply_locale_updates_general_clickable_value_fonts_to_zh_cn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.ui.locale = "en"
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    zh_font = font_for_language("zh-CN")
    assert zh_font is not None

    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale("zh-CN")
        settings.ui.locale = "zh-CN"
        view.apply_locale()

        for control in (
            view._ui_text,
            view._chatbox_source_text,
            view._vrc_mic_text,
            view._mic_audio_text,
            view._audio_host_api_text,
            view._loopback_audio_text,
        ):
            assert control.content.font_family == zh_font
    finally:
        i18n_module.set_locale(previous_locale)


def test_apply_locale_updates_all_settings_clickable_value_fonts_to_zh_cn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.ui.locale = "ko"
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    zh_font = font_for_language("zh-CN")
    assert zh_font is not None

    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale("zh-CN")
        settings.ui.locale = "zh-CN"
        view.apply_locale()

        for control in (
            view._stt_text,
            view._peer_stt_text,
            view._llm_text,
            view._translation_connection_text,
            view._openrouter_fallback_text,
            view._overlay_target_button,
            view._overlay_text_scale_text,
            view._desktop_overlay_size_button,
            view._desktop_overlay_lock_button,
            view._overlay_vr_reset_button,
            view._overlay_desktop_reset_button,
            view._desktop_overlay_primary_action,
            view._desktop_overlay_view_logs_action,
        ):
            assert control.content.font_family == zh_font
    finally:
        i18n_module.set_locale(previous_locale)


def test_overlay_distance_step_buttons_apply_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_overlay_distance_step(0.05)
    view._on_overlay_distance_step(0.20)

    assert settings.overlay.calibration.distance == 1.35
    assert view._overlay_distance_value_text.value == "1.35"


def test_overlay_offset_step_buttons_apply_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_overlay_offset_x_step(0.05)
    view._on_overlay_offset_y_step(-0.05)

    assert settings.overlay.calibration.offset_x == 0.05
    assert settings.overlay.calibration.offset_y == -0.50
    assert view._overlay_offset_x_value_text.value == "0.05"
    assert view._overlay_offset_y_value_text.value == "-0.50"


def test_overlay_calibration_hides_background_alpha_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(AppSettings(), config_path=Path("settings.json"))

    assert not hasattr(view, "_overlay_background_alpha_field")
    assert not hasattr(view, "_overlay_background_alpha_label")


def test_overlay_reset_card_restores_defaults_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.calibration.distance = 1.2
    settings.overlay.calibration.offset_y = 0.5
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.set_overlay_calibration(settings.overlay.calibration)

    defaults = OverlayCalibration()

    view._on_overlay_position_reset(None)

    assert settings.overlay.calibration.distance == defaults.distance
    assert settings.overlay.calibration.offset_y == -0.45
    assert view._overlay_distance_value_text.value == view._format_overlay_calibration_number(
        defaults.distance
    )
    assert view._overlay_offset_y_value_text.value == view._format_overlay_calibration_number(
        defaults.offset_y
    )


def test_overlay_tab_cards_use_settings_unit_card_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.ui.components.settings.settings_unit_card import SettingsUnitCard
    from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper

    view, _ = _make_settings_view(monkeypatch)

    overlay_cards = _overlay_tab_cards(view)
    assert all(isinstance(card, SharedCardWrapper) for card in overlay_cards)
    assert {card.height for card in overlay_cards} == {SettingsUnitCard.DEFAULT_HEIGHT}
    assert all(card.expand is True for card in overlay_cards)
    assert _tree_contains_control(
        _overlay_tab_card(view, t("settings.overlay.calibration.distance")),
        view._overlay_distance_decrease_button,
    )
    assert _tree_contains_control(
        _overlay_tab_card(view, t("settings.overlay.calibration.offset_x")),
        view._overlay_offset_x_decrease_button,
    )
    assert not hasattr(view, "_overlay_calibration_apply_button")
    assert not hasattr(view, "_overlay_calibration_cancel_button")


def test_overlay_distance_card_uses_inline_minus_value_plus_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    distance_card = _overlay_tab_card(view, t("settings.overlay.calibration.distance"))
    distance_card_stack = _wrapped_card_stack(distance_card)
    distance_column = _wrapped_card_column(distance_card)
    distance_value_row = distance_column.controls[1].content

    assert isinstance(distance_card_stack, ft.Stack)
    assert distance_card_stack.fit == ft.StackFit.EXPAND
    click_row = distance_card_stack.controls[0]
    visual_layer = distance_card_stack.controls[1]
    distance_value_container = distance_value_row.controls[1]

    assert isinstance(click_row, ft.Row)
    assert click_row.expand == 1
    assert click_row.spacing == 0
    assert click_row.vertical_alignment == ft.CrossAxisAlignment.STRETCH
    assert click_row.controls[0] is view._overlay_distance_decrease_button
    assert click_row.controls[1] is view._overlay_distance_increase_button
    assert isinstance(visual_layer, ft.TransparentPointer)
    assert visual_layer.content is distance_column
    assert isinstance(distance_value_row, ft.Row)
    assert distance_value_row.controls[1].content is view._overlay_distance_value_text
    assert distance_value_row.spacing == 4
    assert isinstance(view._overlay_distance_decrease_button, ft.Container)
    assert isinstance(view._overlay_distance_increase_button, ft.Container)
    assert view._overlay_distance_decrease_button.expand == 1
    assert view._overlay_distance_decrease_button.width is None
    assert view._overlay_distance_decrease_button.height is None
    assert view._overlay_distance_increase_button.expand == 1
    assert view._overlay_distance_increase_button.width is None
    assert view._overlay_distance_increase_button.height is None
    assert distance_value_row.controls[0].content.value == "－"
    assert distance_value_row.controls[0].alignment == ft.Alignment.CENTER_RIGHT
    assert distance_value_row.controls[2].content.value == "＋"
    assert distance_value_row.controls[2].alignment == ft.Alignment.CENTER_LEFT
    assert distance_value_container.width == 84


def test_overlay_offset_cards_use_inline_arrow_value_arrow_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    offset_x_card = _overlay_tab_card(view, t("settings.overlay.calibration.offset_x"))
    offset_y_card = _overlay_tab_card(view, t("settings.overlay.calibration.offset_y"))
    offset_x_card_stack = _wrapped_card_stack(offset_x_card)
    offset_y_card_stack = _wrapped_card_stack(offset_y_card)
    offset_x_column = _wrapped_card_column(offset_x_card)
    offset_y_column = _wrapped_card_column(offset_y_card)
    offset_x_value_row = offset_x_column.controls[1].content
    offset_y_value_row = offset_y_column.controls[1].content

    assert isinstance(offset_x_card_stack, ft.Stack)
    assert offset_x_card_stack.fit == ft.StackFit.EXPAND
    offset_x_click_row = offset_x_card_stack.controls[0]
    offset_x_visual_layer = offset_x_card_stack.controls[1]
    offset_x_value_container = offset_x_value_row.controls[1]

    assert isinstance(offset_x_click_row, ft.Row)
    assert offset_x_click_row.expand == 1
    assert offset_x_click_row.spacing == 0
    assert offset_x_click_row.vertical_alignment == ft.CrossAxisAlignment.STRETCH
    assert offset_x_click_row.controls[0] is view._overlay_offset_x_decrease_button
    assert offset_x_click_row.controls[1] is view._overlay_offset_x_increase_button
    assert isinstance(offset_x_visual_layer, ft.TransparentPointer)
    assert offset_x_visual_layer.content is offset_x_column
    assert isinstance(offset_x_value_row, ft.Row)
    assert offset_x_value_row.controls[1].content is view._overlay_offset_x_value_text
    assert offset_x_value_row.spacing == 4
    assert offset_x_value_container.width == 84
    assert isinstance(view._overlay_offset_x_decrease_button, ft.Container)
    assert isinstance(view._overlay_offset_x_increase_button, ft.Container)
    assert view._overlay_offset_x_decrease_button.expand == 1
    assert view._overlay_offset_x_decrease_button.width is None
    assert view._overlay_offset_x_decrease_button.height is None
    assert view._overlay_offset_x_increase_button.expand == 1
    assert view._overlay_offset_x_increase_button.width is None
    assert view._overlay_offset_x_increase_button.height is None
    assert offset_x_value_row.controls[0].content.value == "◀"
    assert offset_x_value_row.controls[0].alignment == ft.Alignment.CENTER_RIGHT
    assert offset_x_value_row.controls[2].content.value == "▶"
    assert offset_x_value_row.controls[2].alignment == ft.Alignment.CENTER_LEFT

    assert isinstance(offset_y_card_stack, ft.Stack)
    assert offset_y_card_stack.fit == ft.StackFit.EXPAND
    offset_y_click_row = offset_y_card_stack.controls[0]
    offset_y_visual_layer = offset_y_card_stack.controls[1]
    offset_y_value_container = offset_y_value_row.controls[1]

    assert isinstance(offset_y_click_row, ft.Row)
    assert offset_y_click_row.expand == 1
    assert offset_y_click_row.spacing == 0
    assert offset_y_click_row.vertical_alignment == ft.CrossAxisAlignment.STRETCH
    assert offset_y_click_row.controls[0] is view._overlay_offset_y_decrease_button
    assert offset_y_click_row.controls[1] is view._overlay_offset_y_increase_button
    assert isinstance(offset_y_visual_layer, ft.TransparentPointer)
    assert offset_y_visual_layer.content is offset_y_column
    assert isinstance(offset_y_value_row, ft.Row)
    assert offset_y_value_row.controls[1].content is view._overlay_offset_y_value_text
    assert offset_y_value_row.spacing == 4
    assert offset_y_value_container.width == 84
    assert isinstance(view._overlay_offset_y_decrease_button, ft.Container)
    assert isinstance(view._overlay_offset_y_increase_button, ft.Container)
    assert view._overlay_offset_y_decrease_button.expand == 1
    assert view._overlay_offset_y_decrease_button.width is None
    assert view._overlay_offset_y_decrease_button.height is None
    assert view._overlay_offset_y_increase_button.expand == 1
    assert view._overlay_offset_y_increase_button.width is None
    assert view._overlay_offset_y_increase_button.height is None
    assert offset_y_value_row.controls[0].content.value == "▲"
    assert offset_y_value_row.controls[0].alignment == ft.Alignment.CENTER_RIGHT
    assert offset_y_value_row.controls[2].content.value == "▼"
    assert offset_y_value_row.controls[2].alignment == ft.Alignment.CENTER_LEFT


def test_overlay_step_buttons_use_large_vr_hit_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    for button in (
        view._overlay_distance_decrease_button,
        view._overlay_distance_increase_button,
        view._overlay_offset_x_decrease_button,
        view._overlay_offset_x_increase_button,
        view._overlay_offset_y_decrease_button,
        view._overlay_offset_y_increase_button,
    ):
        assert isinstance(button, ft.Container)
        assert button.expand == 1
        assert button.width is None
        assert button.height is None


def test_translation_card_no_longer_contains_translation_connection_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    translation_card = _api_tab_card(view, t("settings.section.translation"))
    translation_column = translation_card.content.content

    assert view._translation_connection_row not in translation_column.controls


@pytest.mark.asyncio
async def test_prompt_verify_and_emit_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_prompt_change("custom prompt")
    assert settings.system_prompt != "custom prompt"
    assert view.has_pending_prompt_changes is True

    view._on_prompt_commit("custom prompt")
    assert changed[-1].system_prompt == "custom prompt"

    view._on_reset_prompt(None)
    assert settings.system_prompt == view._prompt_editor.value
    assert changed

    unavailable = await view._verify_key("google", "abc")
    assert unavailable == (False, "Verification not available")

    async def fake_verify(provider: str, key: str) -> tuple[bool, str]:
        return provider == "google", key

    view.on_verify_api_key = fake_verify
    available = await view._verify_key("google", "abc")
    assert available == (True, "abc")


def test_prompt_change_only_updates_draft_until_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    original_prompt = settings.system_prompt
    view._on_prompt_change("custom prompt")

    pending = view.build_provider_apply_settings()

    assert settings.system_prompt == original_prompt
    assert settings.system_prompts == {}
    assert view.has_pending_prompt_changes is True
    assert pending is not None
    assert pending.system_prompt == "custom prompt"
    assert pending.system_prompts == {}
    assert changed == []


def test_prompt_commit_emits_once_when_no_provider_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_prompt_change("custom prompt")
    view._on_prompt_commit("custom prompt")

    assert view.has_pending_prompt_changes is False
    assert changed
    assert changed[-1].system_prompt == "custom prompt"
    assert changed[-1].system_prompts == {}


def test_prompt_commit_preserves_peer_local_qwen_before_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.LOCAL_QWEN
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_prompt_change("custom prompt")
    view._on_prompt_commit("custom prompt")

    assert changed
    assert changed[-1] is not settings
    assert changed[-1].provider.peer_stt == STTProviderName.LOCAL_QWEN
    assert settings.provider.peer_stt == STTProviderName.LOCAL_QWEN
    assert changed[-1].system_prompt == "custom prompt"


def test_prompt_commit_noops_when_value_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    current_prompt = view._prompt_editor.value
    view._on_prompt_commit(current_prompt)

    assert changed == []
    assert view.has_pending_prompt_changes is False


def test_prompt_reverting_to_committed_value_clears_pending_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    original_prompt = view._prompt_editor.value

    view._on_prompt_change("temporary prompt")
    assert view.has_pending_prompt_changes is True

    view._on_prompt_change(original_prompt)
    view._on_prompt_commit(original_prompt)

    assert view.has_pending_prompt_changes is False
    assert changed == []


def test_refresh_prompt_if_empty_stages_default_for_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._settings.system_prompt = ""
    view._settings.system_prompts = {}
    view._provider_settings_draft = None
    view.has_provider_changes = False
    view.has_pending_prompt_changes = False
    view._prompt_editor.value = ""

    view.refresh_prompt_if_empty()
    pending = view.build_provider_apply_settings()

    assert bool(view._prompt_editor.value.strip())
    assert view.has_pending_prompt_changes is True
    assert pending is not None
    assert pending.system_prompt == view._prompt_editor.value
    assert pending.system_prompts == {}


def test_on_text_hover_updates_container_once(monkeypatch: pytest.MonkeyPatch) -> None:
    view, _ = _make_settings_view(monkeypatch)
    updates: list[str] = []
    text_control = SimpleNamespace(color=settings_view.COLOR_ON_BACKGROUND)
    container = SimpleNamespace(
        content=text_control,
        update=lambda: updates.append(text_control.color),
    )

    view._on_text_hover(SimpleNamespace(control=container, data="true"))

    assert text_control.color == settings_view.COLOR_PRIMARY
    assert len(updates) == 1


def test_apply_locale_refreshes_deepseek_api_key_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    calls: list[str] = []
    view._deepseek_key.apply_locale = lambda: calls.append("deepseek")

    view.apply_locale()

    assert calls == ["deepseek"]


def test_settings_view_does_not_create_peer_deepgram_model_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    assert not hasattr(view, "_peer_deepgram_model_label")
    assert not hasattr(view, "_peer_deepgram_model_text")


def test_on_vrc_mic_click_toggles_without_page(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    settings.osc.vrc_mic_intercept = False
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    modal_calls: list[str] = []

    class DummyModal:
        def __init__(self, *_args, **_kwargs) -> None:
            modal_calls.append("created")

    monkeypatch.setattr(settings_view, "SettingsModal", DummyModal)

    view._on_vrc_mic_click(None)

    assert settings.osc.vrc_mic_intercept is True
    assert view._vrc_mic_text.content.value == t("settings.vrc_mic.on")
    assert changed == [settings]
    assert modal_calls == []


def test_on_vrc_mic_click_toggles_immediately_without_modal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.osc.vrc_mic_intercept = True
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    attach_dummy_page(monkeypatch, view)

    modal_calls: list[str] = []

    class DummyModal:
        def __init__(self, *_args, **_kwargs) -> None:
            modal_calls.append("created")

        def open(self, current: str) -> None:
            modal_calls.append(f"opened:{current}")

    monkeypatch.setattr(settings_view, "SettingsModal", DummyModal)
    monkeypatch.setattr(type(view._vrc_mic_text), "update", lambda self: None)

    view._on_vrc_mic_click(None)

    assert settings.osc.vrc_mic_intercept is False
    assert view._vrc_mic_text.content.value == t("settings.vrc_mic.off")
    assert changed == [settings]
    assert modal_calls == []


def test_on_vrc_mic_selected_updates_setting_label_and_emits_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    attach_dummy_page(monkeypatch, view)
    monkeypatch.setattr(type(view._vrc_mic_text), "update", lambda self: None)

    view._on_vrc_mic_selected("on")

    assert settings.osc.vrc_mic_intercept is True
    assert view._vrc_mic_text.content.value == t("settings.vrc_mic.on")
    assert changed == [settings]


def test_on_vrc_mic_selected_without_settings_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed: list[AppSettings] = []
    view, _ = _make_settings_view(monkeypatch)
    view.on_settings_changed = lambda incoming: changed.append(incoming)

    view._on_vrc_mic_selected("on")

    assert changed == []


def test_custom_vocabulary_loads_current_source_language_bucket_as_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.SONIOX
    settings.languages.source_language = "ko"
    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {"ko": ["Puripuly", "VRChat"], "en": ["Avatar"]}

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    custom_vocab_card = _prompt_tab_card(view, t("settings.section.custom_vocabulary"))

    assert _tree_contains_control(custom_vocab_card, view._custom_vocab_tag_editor)
    assert not hasattr(view, "_custom_vocab_terms")
    assert _custom_vocab_chip_terms(view) == ["Puripuly", "VRChat"]
    assert view._custom_vocab_tag_editor._input_field.hint_text == ""  # noqa: SLF001
    text_fields = [
        node for node in _iter_control_tree(custom_vocab_card) if isinstance(node, ft.TextField)
    ]
    assert text_fields == [view._custom_vocab_tag_editor._input_field]  # noqa: SLF001
    assert view._custom_vocab_tag_editor._input_field.multiline is False  # noqa: SLF001
    assert view._custom_vocab_tag_editor._input_field.max_lines == 1  # noqa: SLF001


@pytest.mark.parametrize(
    "source_language",
    ["ko", "en", "zh-CN", "ja"],
)
def test_custom_vocabulary_loads_empty_tags_for_fresh_settings(
    monkeypatch: pytest.MonkeyPatch,
    source_language: str,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = source_language

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert _custom_vocab_chip_terms(view) == []
    assert view._custom_vocab_tag_editor._empty_text.visible is False  # noqa: SLF001


def test_custom_vocabulary_card_shows_inline_helper_without_info_icon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    custom_vocab_card = _prompt_tab_card(view, t("settings.section.custom_vocabulary"))
    custom_vocab_column = _wrapped_card_column(custom_vocab_card)

    assert custom_vocab_column.controls[0] is view._custom_vocab_title
    assert view._custom_vocab_description_text in custom_vocab_column.controls
    assert view._custom_vocab_description_text.value == t("settings.custom_vocabulary.description")
    assert t("settings.custom_vocabulary.description") in _control_labels(custom_vocab_card)
    assert not hasattr(view, "_custom_vocab_info_icon")
    assert t("settings.custom_vocabulary_tooltip") not in _control_tooltips(custom_vocab_card)


def test_prompt_tab_uses_shared_full_width_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper

    view, _ = _make_settings_view(monkeypatch)

    prompt_cards = _subtab_controls(view, "prompt")

    assert len(prompt_cards) == 2
    assert all(isinstance(card, SharedCardWrapper) for card in prompt_cards)
    assert all(card.height is None for card in prompt_cards)
    assert all(card.expand is False for card in prompt_cards)


def test_prompt_tab_hides_prompt_provider_copy_and_old_language_helper_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.ui.locale = "ko"
    settings.languages.source_language = "zh-CN"
    settings.provider.llm = LLMProviderName.GEMINI

    old_locale = i18n_module.get_locale()
    try:
        view, _ = _make_settings_view(monkeypatch)
        view.load_from_settings(settings, config_path=Path("settings.json"))

        i18n_module.set_locale("ko")
        view.apply_locale()

        prompt_card = _prompt_tab_card(view, t("settings.section.persona"))
        custom_vocab_card = _prompt_tab_card(view, t("settings.section.custom_vocabulary"))

        assert t(
            "settings.prompt_for",
            provider=provider_label(LLMProviderName.GEMINI.value),
        ) not in _control_labels(prompt_card)
        assert not hasattr(view, "_custom_vocab_helper_text")
        assert t("settings.custom_vocabulary.description") in _control_labels(custom_vocab_card)
    finally:
        i18n_module.set_locale(old_locale)


def test_settings_api_unit_cards_use_settings_unit_card_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.ui.components.settings.settings_unit_card import SettingsUnitCard

    view, _ = _make_settings_view(monkeypatch)

    unit_cards = [
        _api_tab_card(view, t("settings.section.stt")),
        _api_tab_card(view, t("settings.section.peer_stt")),
        _api_tab_card(view, t("settings.section.translation")),
        view._translation_connection_card,
        view._openrouter_fallback_card,
    ]

    assert all(isinstance(card, SettingsUnitCard) for card in unit_cards)
    assert {card.height for card in unit_cards} == {SettingsUnitCard.DEFAULT_HEIGHT}
    assert all(card.expand is True for card in unit_cards)


def test_general_cards_use_settings_unit_card_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.ui.components.settings.settings_unit_card import SettingsUnitCard

    view, _ = _make_settings_view(monkeypatch)

    general_cards = [
        _general_tab_card(view, t("settings.section.ui")),
        _general_tab_card(view, t("settings.chatbox_include_source")),
        _general_tab_card(view, t("settings.clipboard_auto_translate")),
        _general_tab_card(view, t("settings.vrc_mic_intercept")),
        _general_tab_card(view, t("settings.osc.connection.title")),
        _general_tab_card(view, t("settings.audio_host_api")),
        _general_tab_card(view, t("settings.section.microphone_audio")),
        _general_tab_card(view, t("settings.section.loopback_audio")),
        _general_tab_card(view, t("settings.section.self_vad_sensitivity")),
        _general_tab_card(view, t("settings.section.peer_vad_sensitivity")),
    ]

    assert all(isinstance(card, SettingsUnitCard) for card in general_cards)
    assert {card.height for card in general_cards} == {SettingsUnitCard.DEFAULT_HEIGHT}
    assert all(card.expand is True for card in general_cards)
    assert all(getattr(row, "height", None) is None for row in _subtab_controls(view, "general"))
    assert view._translation_connection_row.height is None


def test_api_keys_card_uses_shared_full_width_auto_height(monkeypatch: pytest.MonkeyPatch) -> None:
    from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper

    view, _ = _make_settings_view(monkeypatch)

    api_card = _api_tab_card(view, t("settings.section.api_keys"))

    assert isinstance(api_card, SharedCardWrapper)
    assert api_card.height is None
    assert api_card.expand is False


def test_api_keys_card_omits_helper_copy_and_keeps_qwen_region_button_in_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.qwen.region = QwenRegion.SINGAPORE

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    api_card = _api_tab_card(view, t("settings.section.api_keys"))
    api_column = _wrapped_card_column(api_card)
    api_header = api_column.controls[0]

    assert view._qwen_region_btn.visible is True
    assert isinstance(api_header, ft.Row)
    assert api_header.controls[0] is view._api_title
    assert api_header.controls[2] is view._qwen_region_btn
    assert view._api_credentials_helper_text not in api_column.controls
    assert t("settings.api_credentials_helper") not in _control_labels(api_card)


def test_api_provider_row_does_not_override_shared_card_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    api_provider_row = _subtab_controls(view, "api")[0]

    assert api_provider_row.height is None


def test_integrated_context_controls_are_removed_from_overlay_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)

    overlay_labels: list[str] = []
    for control in _subtab_controls(view, "overlay"):
        overlay_labels.extend(_control_labels(control))

    assert t("settings.integrated_context") not in overlay_labels
    assert not hasattr(view, "_integrated_context_button")
    assert not hasattr(view, "_integrated_context_hint")


def test_custom_vocabulary_switching_source_language_updates_tags_and_clears_add_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly"], "en": ["Avatar", "OSC"]}

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    assert _custom_vocab_chip_terms(view) == ["Puripuly"]
    view._custom_vocab_tag_editor._input_field.value = "unsubmitted hint"  # noqa: SLF001

    settings.languages.source_language = "en"
    view.load_from_settings(
        settings,
        config_path=Path("settings.json"),
        preserve_custom_vocab_draft=True,
    )

    assert _custom_vocab_chip_terms(view) == ["Avatar", "OSC"]
    assert view._custom_vocab_tag_editor._input_field.value == ""  # noqa: SLF001
    assert view._custom_vocab_description_text.value == t("settings.custom_vocabulary.description")


def test_custom_vocabulary_preserve_reload_discards_unsubmitted_add_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly"], "en": ["Avatar"]}

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view._custom_vocab_tag_editor._input_field.value = "VRChat"  # noqa: SLF001

    settings.languages.source_language = "en"
    view.load_from_settings(
        settings,
        config_path=Path("settings.json"),
        preserve_custom_vocab_draft=True,
    )
    assert _custom_vocab_chip_terms(view) == ["Avatar"]
    assert view._custom_vocab_tag_editor._input_field.value == ""  # noqa: SLF001

    settings.languages.source_language = "ko"
    view._custom_vocab_tag_editor._input_field.value = "Puripuly draft"  # noqa: SLF001
    view.load_from_settings(
        settings,
        config_path=Path("settings.json"),
        preserve_custom_vocab_draft=True,
    )

    assert _custom_vocab_chip_terms(view) == ["Puripuly"]
    assert view._custom_vocab_tag_editor._input_field.value == ""  # noqa: SLF001


def test_custom_vocabulary_default_load_refreshes_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly"]}

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._custom_vocab_tag_editor.set_terms(["Puripuly", "VRChat"])
    view._custom_vocab_tag_editor._input_field.value = "unsubmitted"  # noqa: SLF001

    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert _custom_vocab_chip_terms(view) == ["Puripuly"]
    assert view._custom_vocab_tag_editor._input_field.value == ""  # noqa: SLF001


def test_custom_vocabulary_token_input_persists_unique_space_terms_and_emits_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly"], "en": ["Avatar"]}
    settings.stt.custom_vocabulary_enabled = False
    changed: list[AppSettings] = []
    detailed_messages: list[str] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = lambda incoming: changed.append(incoming)
    view.runtime_log_detailed = lambda message, *, level=logging.INFO: detailed_messages.append(
        message
    )
    view._custom_vocab_tag_editor._input_field.value = " VRChat Soniox\nPuripuly "  # noqa: SLF001

    view._custom_vocab_tag_editor._input_field.on_change(None)  # noqa: SLF001

    assert view._custom_vocab_tag_editor.on_add_terms is not None
    assert view._custom_vocab_tag_editor._input_field.value == ""  # noqa: SLF001
    assert settings.stt.custom_vocabulary_enabled is True
    assert settings.stt.custom_terms == {
        "ko": ["Puripuly", "VRChat", "Soniox"],
        "en": ["Avatar"],
    }
    assert _custom_vocab_chip_terms(view) == ["Puripuly", "VRChat", "Soniox"]
    assert len(changed) == 1
    assert changed[-1].stt.custom_terms == settings.stt.custom_terms
    assert changed[-1].stt.custom_vocabulary_enabled is True
    assert detailed_messages == ["[Settings] Custom vocabulary applied: language=ko, terms=3"]


def test_custom_vocabulary_add_normalizes_direct_raw_terms_exact_case_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly"], "en": ["Avatar"]}
    changed: list[AppSettings] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = changed.append
    view._custom_vocab_tag_editor._input_field.value = "draft"  # noqa: SLF001

    view._on_custom_vocabulary_add_terms([" puripuly Puripuly\nPURIPULY ", " "])

    assert view._custom_vocab_tag_editor._input_field.value == ""  # noqa: SLF001
    assert settings.stt.custom_terms == {
        "ko": ["Puripuly", "puripuly", "PURIPULY"],
        "en": ["Avatar"],
    }
    assert _custom_vocab_chip_terms(view) == ["Puripuly", "puripuly", "PURIPULY"]
    assert len(changed) == 1


def test_custom_vocabulary_empty_and_duplicate_adds_clear_input_without_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly"], "en": ["Avatar"]}
    settings.stt.custom_vocabulary_enabled = True
    changed: list[AppSettings] = []
    detailed_messages: list[str] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = changed.append
    view.runtime_log_detailed = lambda message, *, level=logging.INFO: detailed_messages.append(
        message
    )
    view._custom_vocab_tag_editor._input_field.value = " Puripuly  Puripuly  \n "  # noqa: SLF001

    view._custom_vocab_tag_editor._input_field.on_change(None)  # noqa: SLF001

    assert view._custom_vocab_tag_editor._input_field.value == ""  # noqa: SLF001
    assert settings.stt.custom_terms == {"ko": ["Puripuly"], "en": ["Avatar"]}
    assert _custom_vocab_chip_terms(view) == ["Puripuly"]
    assert changed == []
    assert detailed_messages == []


def test_custom_vocabulary_typing_add_input_does_not_emit_or_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly"], "en": ["Avatar"]}
    changed: list[AppSettings] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = changed.append

    view._custom_vocab_tag_editor._input_field.value = "Puripuly\nVRChat"  # noqa: SLF001

    assert changed == []
    assert settings.stt.custom_terms == {"ko": ["Puripuly"], "en": ["Avatar"]}
    assert view._custom_vocab_tag_editor._input_field.value == "Puripuly\nVRChat"  # noqa: SLF001


def test_custom_vocabulary_remove_control_persists_current_bucket_and_emits_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly", "VRChat"], "en": ["Avatar"]}
    settings.stt.custom_vocabulary_enabled = True
    changed: list[AppSettings] = []
    detailed_messages: list[str] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = changed.append
    view.runtime_log_detailed = lambda message, *, level=logging.INFO: detailed_messages.append(
        message
    )

    chip = view._custom_vocab_tag_editor._chips_wrap.controls[0]  # noqa: SLF001
    chip.on_click(None)

    assert settings.stt.custom_vocabulary_enabled is True
    assert settings.stt.custom_terms == {"ko": ["VRChat"], "en": ["Avatar"]}
    assert _custom_vocab_chip_terms(view) == ["VRChat"]
    assert len(changed) == 1
    assert changed[-1].stt.custom_terms == settings.stt.custom_terms
    assert detailed_messages == ["[Settings] Custom vocabulary applied: language=ko, terms=1"]


def test_custom_vocabulary_remove_last_term_derives_disabled_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly"], "en": []}
    settings.stt.custom_vocabulary_enabled = True
    changed: list[AppSettings] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = changed.append

    view._on_custom_vocabulary_remove_term("Puripuly")

    assert settings.stt.custom_terms == {"ko": [], "en": []}
    assert settings.stt.custom_vocabulary_enabled is False
    assert _custom_vocab_chip_terms(view) == []
    assert len(changed) == 1
    assert changed[-1].stt.custom_vocabulary_enabled is False


def test_custom_vocabulary_add_caps_partial_terms_and_shows_snackbar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    existing_terms = [f"term-{i:03d}" for i in range(99)]
    settings.stt.custom_terms = {"ko": existing_terms, "en": ["Avatar"]}
    settings.stt.custom_vocabulary_enabled = True
    changed: list[AppSettings] = []
    snackbars: list[tuple[str, str]] = []
    detailed_messages: list[str] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = changed.append
    view.show_snackbar = lambda msg, bg: snackbars.append((msg, bg))
    view.runtime_log_detailed = lambda message, *, level=logging.INFO: detailed_messages.append(
        message
    )

    view._on_custom_vocabulary_add_terms(["fits overflow"])

    assert settings.stt.custom_terms == {
        "ko": [*existing_terms, "fits"],
        "en": ["Avatar"],
    }
    assert settings.stt.custom_vocabulary_enabled is True
    assert _custom_vocab_chip_terms(view)[-1] == "fits"
    assert len(changed) == 1
    assert snackbars == [
        (t("snackbar.custom_vocabulary_limit", max_terms=100), settings_view.ft.Colors.ORANGE_700)
    ]
    assert detailed_messages == [
        "[Settings] Custom vocabulary capped: language=ko, requested=101, applied=100",
        "[Settings] Custom vocabulary applied: language=ko, terms=100",
    ]


def test_custom_vocabulary_add_when_bucket_full_shows_limit_without_emit_or_runtime_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    existing_terms = [f"term-{i:03d}" for i in range(100)]
    settings.stt.custom_terms = {"ko": existing_terms, "en": []}
    settings.stt.custom_vocabulary_enabled = True
    changed: list[AppSettings] = []
    snackbars: list[tuple[str, str]] = []
    detailed_messages: list[str] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_settings_changed = changed.append
    view.show_snackbar = lambda msg, bg: snackbars.append((msg, bg))
    view.runtime_log_detailed = lambda message, *, level=logging.INFO: detailed_messages.append(
        message
    )

    view._on_custom_vocabulary_add_terms(["overflow"])

    assert settings.stt.custom_terms == {"ko": existing_terms, "en": []}
    assert settings.stt.custom_vocabulary_enabled is True
    assert _custom_vocab_chip_terms(view) == existing_terms
    assert changed == []
    assert snackbars == [
        (t("snackbar.custom_vocabulary_limit", max_terms=100), settings_view.ft.Colors.ORANGE_700)
    ]
    assert detailed_messages == []


def test_on_qwen_region_selected_uses_detailed_runtime_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    detailed_messages: list[str] = []

    view, _ = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.runtime_log_detailed = lambda message, *, level=logging.INFO: detailed_messages.append(
        message
    )

    view._on_qwen_region_selected(QwenRegion.SINGAPORE.value)

    assert detailed_messages == ["[Settings] Qwen region changed: beijing -> singapore"]


def test_settings_view_uses_generic_subtab_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    from puripuly_heart.ui.components.subtab_shell import TextSubtabShell

    view, _ = _make_settings_view(monkeypatch)

    assert view.scroll is None
    assert view.controls == [view._settings_subtab_shell]
    assert isinstance(view._settings_subtab_shell, TextSubtabShell)
    assert isinstance(view._settings_subtab_shell.body_host, ft.Stack)
    assert view._settings_subtab_shell.title_region is None
    assert isinstance(view._settings_subtab_shell.body_region, ft.Container)
    assert view._settings_subtab_shell.body_region.content is view._settings_subtab_shell.body_host
    assert view._settings_subtab_shell.controls == [
        view._settings_subtab_shell.body_region,
        view._settings_subtab_shell.subtab_bar,
    ]


def test_settings_subtab_shell_preserves_per_tab_scroll_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    shell = view._settings_subtab_shell

    api_body = shell.body_by_key["api"]
    general_body = shell.body_by_key["general"]

    shell.record_scroll("api", SimpleNamespace(pixels=144.0))
    shell.select_tab("general")
    shell.record_scroll("general", SimpleNamespace(pixels=320.0))
    shell.select_tab("api")

    assert shell.active_key == "api"
    assert api_body.scroll == ft.ScrollMode.AUTO
    assert general_body.scroll == ft.ScrollMode.AUTO
    assert shell.scroll_offsets["api"] == 144.0
    assert shell.scroll_offsets["general"] == 320.0
    assert api_body.visible is True
    assert general_body.visible is False


def test_settings_subtab_shell_restores_scroll_on_tab_switch_for_mounted_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    shell = view._settings_subtab_shell
    api_body = shell.body_by_key["api"]
    scroll_calls: list[tuple[float, int]] = []

    monkeypatch.setattr(type(api_body), "page", property(lambda self: object()))
    monkeypatch.setattr(
        api_body,
        "scroll_to",
        lambda **kwargs: scroll_calls.append((kwargs["offset"], kwargs["duration"])),
    )

    shell.record_scroll("api", SimpleNamespace(pixels=144.0))
    shell.select_tab("general")
    shell.select_tab("api")

    assert scroll_calls == [(144.0, 0)]


def test_settings_subtab_bar_matches_bottom_nav_family_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view, _ = _make_settings_view(monkeypatch)
    shell = view._settings_subtab_shell
    main_nav = BottomNavBar(on_change=lambda _idx: None)

    buttons = [shell.button_by_key[key] for key in settings_view._SETTINGS_SUBTAB_ORDER]
    dividers = [
        control for control in shell.subtab_row.controls if isinstance(control, ft.VerticalDivider)
    ]
    border = shell.subtab_bar.border
    nav_row = main_nav.content
    nav_dividers = [
        control for control in nav_row.controls if isinstance(control, ft.VerticalDivider)
    ]

    assert isinstance(shell.subtab_bar.content, ft.Row)
    assert shell.controls[-1] is shell.subtab_bar
    assert shell.spacing == 0
    assert shell.subtab_row.expand is True
    assert shell.subtab_row.wrap is False
    assert shell.subtab_row.scroll is None
    assert shell.subtab_row.spacing == 0
    assert all(isinstance(button, ft.Container) for button in buttons)
    assert all(button.expand is True for button in buttons)
    assert all(button.alignment == ft.Alignment.CENTER for button in buttons)
    assert all(callable(button.on_click) for button in buttons)
    assert all(callable(button.on_hover) for button in buttons)
    assert all(isinstance(button.content, ft.Text) for button in buttons)
    assert len(dividers) == len(settings_view._SETTINGS_SUBTAB_ORDER) - 1
    assert shell.subtab_bar.bgcolor == main_nav.bgcolor
    assert shell.subtab_bar.height == int(main_nav.height * 0.8)
    assert shell.subtab_bar.border_radius is None
    assert border.top.width == main_nav.border.top.width
    assert border.top.color == main_nav.border.top.color
    assert border.left.style == ft.BorderStyle.NONE
    assert border.right.style == ft.BorderStyle.NONE
    assert border.bottom.style == ft.BorderStyle.NONE
    assert shell.subtab_bar.padding is None
    assert len(nav_dividers) == len(dividers)
    assert all(divider.width == nav_dividers[0].width for divider in dividers)
    assert all(divider.thickness == nav_dividers[0].thickness for divider in dividers)
    assert all(divider.color == nav_dividers[0].color for divider in dividers)
    assert _subtab_text_color(buttons[0]) == subtab_shell_module.COLOR_PRIMARY
    assert _subtab_text_color(buttons[1]) == subtab_shell_module.COLOR_SECONDARY

    buttons[1].on_hover(SimpleNamespace(data="true"))
    assert _subtab_text_color(buttons[1]) == subtab_shell_module.COLOR_PRIMARY

    buttons[1].on_hover(SimpleNamespace(data="false"))
    assert _subtab_text_color(buttons[1]) == subtab_shell_module.COLOR_SECONDARY

    buttons[1].on_click(SimpleNamespace())
    assert shell.active_key == "general"
    assert _subtab_text_color(buttons[0]) == subtab_shell_module.COLOR_SECONDARY
    assert _subtab_text_color(buttons[1]) == subtab_shell_module.COLOR_PRIMARY


def test_text_subtab_shell_keeps_floating_treatment_when_bar_is_top() -> None:
    from puripuly_heart.ui.components.subtab_shell import TextSubtab, TextSubtabShell

    shell = TextSubtabShell(
        title=ft.Text("Settings"),
        tabs=[
            TextSubtab("api", "API", (ft.Text("One"),)),
            TextSubtab("general", "General", (ft.Text("Two"),)),
        ],
    )

    assert shell.controls == [shell.title_region, shell.subtab_bar, shell.body_host]
    assert shell.spacing == 16
    assert shell.subtab_row.scroll == ft.ScrollMode.AUTO
    assert shell.subtab_row.spacing == 8
    assert shell.subtab_bar.bgcolor == subtab_shell_module.COLOR_SURFACE
    assert _button_style_value(shell.button_by_key["api"], "bgcolor") == (
        subtab_shell_module.COLOR_PRIMARY_CONTAINER
    )
    assert _button_style_value(shell.button_by_key["general"], "bgcolor") == (ft.Colors.TRANSPARENT)


def test_text_subtab_shell_rejects_duplicate_keys() -> None:
    from puripuly_heart.ui.components.subtab_shell import TextSubtab, TextSubtabShell

    with pytest.raises(ValueError, match="unique tab keys"):
        TextSubtabShell(
            title=ft.Text("Settings"),
            tabs=[
                TextSubtab("api", "API", (ft.Text("One"),)),
                TextSubtab("api", "Again", (ft.Text("Two"),)),
            ],
        )


def test_text_subtab_shell_rejects_unknown_initial_key() -> None:
    from puripuly_heart.ui.components.subtab_shell import TextSubtab, TextSubtabShell

    with pytest.raises(ValueError, match="Unknown initial tab key"):
        TextSubtabShell(
            title=ft.Text("Settings"),
            tabs=[
                TextSubtab("api", "API", (ft.Text("One"),)),
                TextSubtab("general", "General", (ft.Text("Two"),)),
            ],
            initial_key="overlay",
        )


def test_text_subtab_shell_can_render_without_title_and_pin_subtab_bar_to_bottom() -> None:
    from puripuly_heart.ui.components.subtab_shell import TextSubtab, TextSubtabShell

    shell = TextSubtabShell(
        title=None,
        tabs=[
            TextSubtab("api", "API", (ft.Text("One"),)),
            TextSubtab("general", "General", (ft.Text("Two"),)),
        ],
        subtab_bar_position="bottom",
    )

    assert shell.title_region is None
    assert isinstance(shell.body_region, ft.Container)
    assert shell.body_region.content is shell.body_host
    assert shell.controls == [shell.body_region, shell.subtab_bar]
