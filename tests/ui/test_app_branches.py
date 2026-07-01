from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")
import flet as ft

import puripuly_heart.ui.app as app_module
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterSelectionAlias,
    ProviderSettings,
    STTProviderName,
    TranslationConnection,
    TranslationModel,
    TranslationSettings,
)
from puripuly_heart.core.managed_openrouter_release import TalkTogetherPassStatus
from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.app import TranslatorApp, _check_and_notify_update

MISSING = object()


class DummyPage:
    def __init__(self) -> None:
        self.opened: list[object] = []
        self.closed: list[object] = []
        self.tasks: list[object] = []
        self.title: str = ""
        self.theme = None
        self.updated = 0
        self.theme_mode = None
        self.bgcolor = None
        self.padding = None
        self.added: list[object] = []
        self.window = SimpleNamespace(
            frameless=False,
            resizable=False,
            width=0,
            height=0,
            min_width=0,
            min_height=0,
            icon="",
        )
        self.dialog = None

    def open(self, control) -> None:
        self.opened.append(control)

    def close(self, control) -> None:
        self.closed.append(control)
        if self.dialog is control:
            self.dialog = None

    def run_task(self, coro_fn) -> None:
        self.tasks.append(coro_fn)

    def update(self) -> None:
        self.updated += 1

    def add(self, control) -> None:
        self.added.append(control)


class DummyContent:
    def __init__(self, content=None) -> None:
        self.content = content
        self.update_calls = 0

    def update(self) -> None:
        self.update_calls += 1


class InlineMicrophoneTestSettingsView:
    def __init__(self) -> None:
        self.levels: list[float] = []
        self.active_states: list[bool] = []

    def set_microphone_test_level(self, value: float) -> None:
        self.levels.append(value)

    def set_microphone_test_active(self, active: bool) -> None:
        self.active_states.append(active)


class RuntimeLoggingController:
    def __init__(self) -> None:
        self.basic_messages: list[str] = []
        self.detailed_messages: list[str] = []

    def log_basic(self, message: str, *, level: int = app_module.logging.INFO) -> None:
        _ = level
        self.basic_messages.append(message)

    def log_detailed(self, message: str, *, level: int = app_module.logging.INFO) -> None:
        _ = level
        self.detailed_messages.append(message)


def _iter_control_tree(control):
    if control is None:
        return
    yield control
    for attr in ("title", "content", "icon"):
        child = getattr(control, attr, None)
        if child is not None and not isinstance(child, str):
            yield from _iter_control_tree(child)
    for attr in ("controls", "actions"):
        for child in getattr(control, attr, None) or []:
            yield from _iter_control_tree(child)


def _dialog_text_values(dialog) -> list[str]:
    return [
        node.value
        for node in _iter_control_tree(dialog)
        if isinstance(node, ft.Text) and node.value
    ]


def _dialog_containers(dialog) -> list[ft.Container]:
    return [node for node in _iter_control_tree(dialog) if isinstance(node, ft.Container)]


class ConstructionDummyController:
    def __init__(self, page, app, config_path):
        self.page = page
        self.app = app
        self.config_path = config_path
        self.settings = None
        self.runtime_logging_mode = "detailed"
        self.basic_messages: list[str] = []
        self.detailed_messages: list[str] = []

    def set_runtime_logging_mode(self, mode: str) -> None:
        self.runtime_logging_mode = mode

    def log_basic(self, message: str, *, level: int = app_module.logging.INFO) -> None:
        _ = level
        self.basic_messages.append(message)

    def log_detailed(self, message: str, *, level: int = app_module.logging.INFO) -> None:
        _ = level
        self.detailed_messages.append(message)

    def cycle_debug_capture_fault_profile(self) -> str:
        self.capture_fault_cycled = True
        return "capture_attenuate_40db"

    def cycle_debug_stt_fault_profile(self) -> str:
        self.stt_fault_cycled = True
        return "stt_input_low_snr_vad_pass"

    def clear_debug_audio_fault_profiles(self) -> None:
        self.audio_faults_cleared = True


class ConstructionDummyDashboardView(ft.Container):
    def __init__(self) -> None:
        super().__init__()
        self.on_send_message = None
        self.on_toggle_translation = None
        self.on_toggle_stt = None
        self.on_toggle_overlay = None
        self.on_toggle_peer_translation = None
        self.on_language_change = None
        self.overlay_peer_contract = None
        self.runtime_log_detailed = None

    def set_overlay_peer_contract(self, contract) -> None:
        self.overlay_peer_contract = contract

    def apply_locale(self) -> None:
        return None


class ConstructionDummySettingsView(ft.Container):
    def __init__(self) -> None:
        super().__init__()
        self.on_settings_changed = None
        self.on_prompt_apply_settings = None
        self.on_providers_changed = None
        self.on_start_microphone_test = None
        self.on_request_openrouter_pkce = None
        self.on_verify_api_key = None
        self.on_secret_cleared = None
        self.on_local_llm_secret_changed = None
        self.on_desktop_overlay_lock_change = None
        self.on_desktop_overlay_size_change = None
        self.on_stop_microphone_test = None
        self.on_desktop_overlay_recovery_action = None
        self.on_desktop_overlay_position_reset = None
        self.show_snackbar = None
        self.overlay_peer_contract = None
        self.has_provider_changes = False
        self.has_pending_prompt_changes = False
        self.synced_desktop_settings: list[AppSettings] = []

    def set_overlay_runtime_state(self, *_args, **_kwargs) -> None:
        return None

    def sync_desktop_overlay_settings(self, settings: AppSettings) -> None:
        self.synced_desktop_settings.append(settings)

    def set_overlay_peer_contract(self, contract) -> None:
        self.overlay_peer_contract = contract

    def apply_locale(self) -> None:
        return None

    def refresh_prompt_if_empty(self) -> None:
        return None


class ConstructionDummyLogsView(ft.Container):
    def __init__(self) -> None:
        super().__init__()
        self.on_mode_change = None
        self.runtime_logging_mode = "basic"

    def set_runtime_logging_mode(self, mode: str) -> None:
        self.runtime_logging_mode = mode

    def apply_locale(self) -> None:
        return None

    async def scroll_to_bottom(self) -> None:
        return None

    def log_basic(self, message: str, *, level: int = app_module.logging.INFO) -> None:
        _ = (message, level)

    def log_detailed(self, message: str, *, level: int = app_module.logging.INFO) -> None:
        _ = (message, level)


def _patch_app_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "GuiController", ConstructionDummyController)
    monkeypatch.setattr(app_module, "DashboardView", ConstructionDummyDashboardView)
    monkeypatch.setattr(app_module, "SettingsView", ConstructionDummySettingsView)
    monkeypatch.setattr(app_module, "LogsView", ConstructionDummyLogsView)
    monkeypatch.setattr(app_module, "AboutView", lambda: ft.Container())
    monkeypatch.setattr(app_module, "TitleBar", lambda _page: ft.Container())
    monkeypatch.setattr(app_module, "BottomNavBar", lambda on_change: ft.Container(data=on_change))
    monkeypatch.setattr(app_module, "register_fonts", lambda _page: None)
    monkeypatch.setattr(app_module, "get_app_theme", lambda **_kwargs: "theme")
    monkeypatch.setattr(app_module, "font_for_language", lambda _code: "font")
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")


def test_translator_app_init_builds_layout_and_wires_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    page = DummyPage()
    app = TranslatorApp(page, config_path=Path("settings.json"))

    assert app.controller.config_path == Path("settings.json")
    assert page.title == app_module.t("app.title")
    assert page.window.frameless is True
    assert page.window.resizable is True
    assert page.window.width == app_module.DEFAULT_WINDOW_WIDTH
    assert page.window.height == app_module.DEFAULT_WINDOW_HEIGHT
    assert page.window.min_width == app_module.MIN_WINDOW_WIDTH
    assert page.window.min_height == app_module.MIN_WINDOW_HEIGHT
    assert page.window.width >= page.window.min_width
    assert page.window.height >= page.window.min_height
    assert page.added
    assert app.view_dashboard.on_send_message == app._on_manual_submit
    assert app.view_dashboard.on_toggle_overlay == app._on_overlay_toggle
    assert app.view_dashboard.on_toggle_peer_translation == app._on_peer_translation_toggle
    assert app.view_settings.on_verify_api_key == app._on_verify_api_key
    assert app.view_settings.on_prompt_apply_settings == app._on_prompt_apply_settings
    assert app.view_settings.on_start_microphone_test == app._on_start_microphone_test
    assert app.view_settings.on_desktop_overlay_lock_change == (app._on_desktop_overlay_lock_change)
    assert app.view_settings.on_desktop_overlay_size_change == (app._on_desktop_overlay_size_change)
    assert app.view_settings.on_desktop_overlay_recovery_action == (
        app._on_desktop_overlay_recovery_action
    )
    assert app.view_settings.on_desktop_overlay_position_reset == (
        app._on_desktop_overlay_position_reset
    )
    assert app.view_settings.on_view_logs == app._open_logs_tab
    assert not hasattr(app.view_settings, "on_overlay_toggle")
    assert not hasattr(app.view_settings, "on_peer_translation_toggle")
    assert app.view_settings.runtime_log_basic == app.controller.log_basic
    assert app.view_settings.runtime_log_detailed == app.controller.log_detailed
    assert app.view_logs.on_mode_change == app._on_runtime_logging_mode_change
    assert app.view_logs.runtime_logging_mode == "detailed"


@pytest.mark.asyncio
async def test_desktop_gui_state_actions_are_dispatched_through_translator_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    page = DummyPage()
    app = TranslatorApp(page, config_path=Path("settings.json"))
    locked_requests: list[bool] = []
    size_requests: list[str] = []
    retry_requests: list[bool] = []
    desktop_reset_requests: list[bool] = []

    async def fake_set_locked(locked: bool) -> None:
        locked_requests.append(locked)

    async def fake_set_overlay_enabled(enabled: bool) -> None:
        retry_requests.append(enabled)

    async def fake_set_desktop_overlay_size_preset(size_preset: str) -> None:
        size_requests.append(size_preset)

    async def fake_reset_desktop_overlay_position() -> None:
        desktop_reset_requests.append(True)

    app.controller.set_desktop_overlay_captions_locked = fake_set_locked
    app.controller.set_desktop_overlay_size_preset = fake_set_desktop_overlay_size_preset
    app.controller.set_overlay_enabled = fake_set_overlay_enabled
    app.controller.reset_desktop_overlay_position = fake_reset_desktop_overlay_position

    app._on_desktop_overlay_lock_change(False)
    app._on_desktop_overlay_size_change("xlarge")
    app._on_desktop_overlay_recovery_action("retry")
    app._on_desktop_overlay_position_reset()

    assert len(page.tasks) == 4
    await page.tasks[0]()
    await page.tasks[1]()
    await page.tasks[2]()
    await page.tasks[3]()

    assert locked_requests == [False]
    assert size_requests == ["xlarge"]
    assert retry_requests == [True]
    assert desktop_reset_requests == [True]


@pytest.mark.asyncio
async def test_desktop_gui_state_actions_refresh_settings_view_after_runtime_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    page = DummyPage()
    app = TranslatorApp(page, config_path=Path("settings.json"))
    app.controller.settings = AppSettings()
    app.controller.settings.overlay.desktop_flet.position.x = 80
    app.controller.settings.overlay.desktop_flet.position.y = 90

    async def fake_set_locked(locked: bool) -> None:
        app.controller.settings.overlay.desktop_flet.locked = locked

    async def fake_set_desktop_overlay_size_preset(size_preset: str) -> None:
        updated = copy.deepcopy(app.controller.settings)
        updated.overlay.desktop_flet.size_preset = size_preset
        app.controller.settings = updated

    async def fake_reset_desktop_overlay_position() -> None:
        app.controller.settings.overlay.desktop_flet.position.x = None
        app.controller.settings.overlay.desktop_flet.position.y = None
        app.controller.settings.overlay.desktop_flet.locked = False

    app.controller.set_desktop_overlay_captions_locked = fake_set_locked
    app.controller.set_desktop_overlay_size_preset = fake_set_desktop_overlay_size_preset
    app.controller.reset_desktop_overlay_position = fake_reset_desktop_overlay_position

    app._on_desktop_overlay_lock_change(True)
    app._on_desktop_overlay_size_change("xlarge")
    app._on_desktop_overlay_position_reset()

    assert len(page.tasks) == 3
    await page.tasks[0]()
    await page.tasks[1]()
    await page.tasks[2]()

    synced_settings = app.view_settings.synced_desktop_settings
    assert len(synced_settings) == 3
    assert synced_settings[0].overlay.desktop_flet.locked is True
    assert synced_settings[1].overlay.desktop_flet.size_preset == "xlarge"
    assert synced_settings[2].overlay.desktop_flet.position.x is None
    assert synced_settings[2].overlay.desktop_flet.position.y is None
    assert synced_settings[2].overlay.desktop_flet.locked is False


def test_translator_app_does_not_mount_debug_preview_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    page = DummyPage()
    app = TranslatorApp(page, config_path=Path("settings.json"))

    assert app.debug_ui_preview is False
    assert app.debug_preview_panel is None
    root = page.added[0]
    assert not isinstance(root.content, ft.Stack)


def test_translator_app_mounts_debug_preview_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)
    seen: dict[str, object] = {}

    class FakeDebugPreviewPanel(ft.Container):
        def __init__(self, **callbacks):
            seen["callbacks"] = callbacks
            super().__init__(data="fake-debug-preview-panel")

    monkeypatch.setattr(app_module, "DebugPreviewPanel", FakeDebugPreviewPanel)

    page = DummyPage()
    app = TranslatorApp(
        page,
        config_path=Path("settings.json"),
        debug_ui_preview=True,
    )

    assert app.debug_ui_preview is True
    assert app.debug_preview_panel is not None
    assert set(seen["callbacks"]) == {
        "on_brake_notice",
        "on_revoked_notice",
        "on_github_star_snackbar",
        "on_founder_letter",
        "on_pkce_failure",
        "on_discord_auth",
        "on_discord_callback_page",
        "on_peer_translation_eula",
        "on_local_qwen_hallucination_modal",
        "on_talk_together_pass_invite_progress",
        "on_capture_fault_cycle",
        "on_stt_fault_cycle",
        "on_audio_fault_clear",
    }
    discord_callback = seen["callbacks"]["on_discord_auth"]
    assert getattr(discord_callback, "__self__", None) is app
    assert getattr(discord_callback, "__func__", None) is TranslatorApp._preview_discord_auth
    callback_page = seen["callbacks"]["on_discord_callback_page"]
    assert getattr(callback_page, "__self__", None) is app
    assert getattr(callback_page, "__func__", None) is TranslatorApp._preview_discord_callback_page
    github_star = seen["callbacks"]["on_github_star_snackbar"]
    assert getattr(github_star, "__self__", None) is app
    assert getattr(github_star, "__func__", None) is TranslatorApp._preview_github_star_snackbar
    local_qwen_modal = seen["callbacks"]["on_local_qwen_hallucination_modal"]
    assert getattr(local_qwen_modal, "__self__", None) is app
    assert (
        getattr(local_qwen_modal, "__func__", None)
        is TranslatorApp._preview_local_qwen_hallucination_modal
    )
    pass_progress = seen["callbacks"]["on_talk_together_pass_invite_progress"]
    assert getattr(pass_progress, "__self__", None) is app
    assert (
        getattr(pass_progress, "__func__", None)
        is TranslatorApp._preview_talk_together_pass_invite_progress
    )
    preview_calls: list[bool] = []
    monkeypatch.setattr(
        app,
        "show_discord_managed_auth_dialog",
        lambda *, preview=False: preview_calls.append(preview),
    )
    discord_callback()
    assert preview_calls == [True]
    root = page.added[0]
    assert isinstance(root.content, ft.Stack)
    assert root.content.controls[-1] is app.debug_preview_panel


def test_debug_preview_local_qwen_modal_opens_production_dialog_without_state_or_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = AppSettings()
    settings.provider.stt = STTProviderName.DEEPGRAM
    app.controller = SimpleNamespace(
        settings=settings,
        _local_qwen_hallucination_detection_count=1,
        _local_qwen_hallucination_modal_shown=False,
        apply_settings=lambda *_args, **_kwargs: pytest.fail(
            "debug modal preview must not apply settings"
        ),
        apply_providers=lambda *_args, **_kwargs: pytest.fail(
            "debug modal preview must not apply providers"
        ),
    )
    captured: dict[str, object] = {}

    class FakeLocalQwenHallucinationDialog:
        def __init__(self, page, *, on_open_guide):
            captured["page"] = page
            captured["on_open_guide"] = on_open_guide

        def open(self) -> None:
            captured["opened"] = True

    monkeypatch.setattr(
        app_module,
        "LocalQwenHallucinationDialog",
        FakeLocalQwenHallucinationDialog,
    )
    monkeypatch.setattr(
        app_module,
        "save_settings",
        lambda *_args, **_kwargs: pytest.fail("debug modal preview must not save settings"),
    )
    monkeypatch.setattr(
        app_module.webbrowser,
        "open",
        lambda *_args, **_kwargs: pytest.fail("debug modal preview must not open external URLs"),
    )

    app._preview_local_qwen_hallucination_modal()

    assert captured["page"] is app.page
    assert captured["opened"] is True
    assert getattr(captured["on_open_guide"], "__self__", None) is app
    assert (
        getattr(captured["on_open_guide"], "__func__", None) is TranslatorApp._open_local_qwen_guide
    )
    assert app._local_qwen_hallucination_dialog.__class__ is FakeLocalQwenHallucinationDialog
    assert app.controller._local_qwen_hallucination_detection_count == 1
    assert app.controller._local_qwen_hallucination_modal_shown is False
    assert app.page.tasks == []


def test_debug_preview_panel_wires_audio_fault_actions(monkeypatch) -> None:
    _patch_app_construction(monkeypatch)
    captured_kwargs: dict[str, object] = {}
    snackbars: list[tuple[str, object]] = []

    class FakeDebugPreviewPanel(ft.Container):
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            super().__init__()

    monkeypatch.setattr(app_module, "DebugPreviewPanel", FakeDebugPreviewPanel)
    app = app_module.TranslatorApp(
        DummyPage(), config_path=Path("settings.json"), debug_ui_preview=True
    )
    monkeypatch.setattr(
        app, "_show_snackbar", lambda message, color=None: snackbars.append((message, color))
    )

    assert callable(captured_kwargs["on_capture_fault_cycle"])
    assert callable(captured_kwargs["on_stt_fault_cycle"])
    assert callable(captured_kwargs["on_audio_fault_clear"])
    captured_kwargs["on_capture_fault_cycle"]()
    captured_kwargs["on_stt_fault_cycle"]()
    captured_kwargs["on_audio_fault_clear"]()
    assert app.controller.capture_fault_cycled is True
    assert app.controller.stt_fault_cycled is True
    assert app.controller.audio_faults_cleared is True
    assert snackbars[0][0] == app_module.t(
        "debug_preview.capture_fault_snackbar", profile="capture_attenuate_40db"
    )
    assert snackbars[1][0] == app_module.t(
        "debug_preview.stt_fault_snackbar", profile="stt_input_low_snr_vad_pass"
    )
    assert snackbars[2][0] == app_module.t("debug_preview.audio_fault_clear")


def test_debug_audio_fault_actions_do_not_call_persistence_or_providers(monkeypatch) -> None:
    _patch_app_construction(monkeypatch)
    forbidden_calls: list[str] = []
    monkeypatch.setattr(
        app_module,
        "save_settings",
        lambda *args, **kwargs: forbidden_calls.append("save_settings"),
    )
    monkeypatch.setattr(
        app_module,
        "webbrowser",
        SimpleNamespace(open=lambda *args, **kwargs: forbidden_calls.append("webbrowser.open")),
    )

    app = app_module.TranslatorApp(
        DummyPage(), config_path=Path("settings.json"), debug_ui_preview=True
    )
    monkeypatch.setattr(app, "_show_snackbar", lambda *_args, **_kwargs: None)

    app._preview_capture_fault_cycle()
    app._preview_stt_fault_cycle()
    app._preview_audio_fault_clear()

    assert forbidden_calls == []


def test_local_qwen_guidance_modal_open_guide_opens_github_api_key_guide_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    class FakeBottomNavBar(ft.Container):
        def __init__(self, on_change):
            super().__init__()
            self._on_change = on_change
            self._selected = 0
            self.visual_updates = 0

        def _update_visuals(self) -> None:
            self.visual_updates += 1

    monkeypatch.setattr(app_module, "BottomNavBar", FakeBottomNavBar)
    forbidden_calls: list[str] = []
    opened_urls: list[str] = []
    monkeypatch.setattr(
        app_module,
        "save_settings",
        lambda *args, **kwargs: forbidden_calls.append("save_settings"),
    )
    monkeypatch.setattr(
        app_module.webbrowser,
        "open",
        lambda url, *args, **kwargs: opened_urls.append(url),
    )
    monkeypatch.setattr(app_module, "get_locale", lambda: "ko")

    page = DummyPage()
    app = TranslatorApp(page, config_path=Path("settings.json"))
    monkeypatch.setattr(app.content_area, "update", lambda: None)
    app.controller.apply_settings = lambda *args, **kwargs: forbidden_calls.append("apply_settings")
    app.controller.apply_providers = lambda *args, **kwargs: forbidden_calls.append(
        "apply_providers"
    )
    app.view_settings.has_provider_changes = True

    app.show_local_qwen_hallucination_dialog()

    dialog = app._local_qwen_hallucination_dialog
    opened_dialog = page.opened[-1]
    assert opened_dialog is dialog._dialog

    dialog._dialog_result.primary_button.on_click(None)

    assert page.closed == [opened_dialog]
    assert opened_urls == [app_module.founder_readme_url_for_locale("ko")]
    assert app.content_area.content is not app.view_settings
    assert app.bottom_nav._selected == 0
    assert app.bottom_nav.visual_updates == 0
    assert page.tasks == []
    assert forbidden_calls == []


def test_debug_preview_talk_together_pass_invite_progress_sets_settings_state_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    managed_key_calls: list[dict[str, object]] = []
    app.view_settings = SimpleNamespace(
        set_managed_key_state=lambda **kwargs: managed_key_calls.append(kwargs)
    )
    forbidden_calls: list[str] = []
    monkeypatch.setattr(
        app_module,
        "save_settings",
        lambda *args, **kwargs: forbidden_calls.append("save_settings"),
    )
    monkeypatch.setattr(
        app_module.webbrowser,
        "open",
        lambda *args, **kwargs: forbidden_calls.append("webbrowser.open"),
    )

    app._preview_talk_together_pass_invite_progress()

    assert forbidden_calls == []
    assert managed_key_calls == [
        {
            "visible": True,
            "remaining_percent": 100,
            "referral_id": "7KQ9M2",
            "remember_referral_id": False,
            "pass_status": TalkTogetherPassStatus(
                pass_id="7KQ9M2",
                invite_count=1,
                invite_limit=5,
                bonus_translations_per_friend=200,
            ),
        }
    ]


def test_debug_preview_discord_callback_page_opens_local_preview_without_oauth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    seen_locales: list[str | None] = []
    opened_urls: list[str] = []

    monkeypatch.setattr(app_module, "get_locale", lambda: "ko")
    monkeypatch.setattr(
        app_module,
        "_write_discord_callback_preview_page",
        lambda locale: seen_locales.append(locale) or "file:///tmp/puripuly-callback.html",
        raising=False,
    )
    monkeypatch.setattr(
        app_module.webbrowser,
        "open",
        lambda url: opened_urls.append(url) or True,
    )
    monkeypatch.setattr(
        app,
        "_start_discord_managed_auth",
        lambda *_args, **_kwargs: pytest.fail("callback page preview must not start OAuth"),
        raising=False,
    )

    app._preview_discord_callback_page()

    assert seen_locales == ["ko"]
    assert opened_urls == ["file:///tmp/puripuly-callback.html"]


def test_mark_discord_managed_auth_callback_received_updates_open_dialog() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    calls: list[str] = []
    app._discord_managed_auth_generation = 7
    app._discord_managed_auth_cancelled = False
    app._discord_managed_auth_dialog = SimpleNamespace(
        set_callback_received=lambda: calls.append("received")
    )

    app.mark_discord_managed_auth_callback_received(7)

    assert calls == ["received"]


def test_mark_discord_managed_auth_callback_received_ignores_stale_generation() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    calls: list[str] = []
    app._discord_managed_auth_generation = 2
    app._discord_managed_auth_cancelled = False
    app._discord_managed_auth_dialog = SimpleNamespace(
        set_callback_received=lambda: calls.append("received")
    )

    app.mark_discord_managed_auth_callback_received(1)

    assert calls == []


def test_translator_app_keeps_debug_ui_preview_out_of_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)
    seen: dict[str, object] = {}

    class RecordingController(ConstructionDummyController):
        def __init__(self, page, app, config_path):
            super().__init__(page, app, config_path)
            seen["controller_args"] = (page, app, config_path)

    monkeypatch.setattr(app_module, "GuiController", RecordingController)

    app = TranslatorApp(
        DummyPage(),
        config_path=Path("settings.json"),
        debug_ui_preview=True,
    )

    assert app.debug_ui_preview is True
    assert seen["controller_args"] == (app.page, app, Path("settings.json"))


def test_translator_app_wires_runtime_log_detailed_into_dashboard_visual_commit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyController:
        def __init__(self, page, app, config_path, debug_ui_preview: bool = False):
            self.page = page
            self.app = app
            self.config_path = config_path
            self.debug_ui_preview = debug_ui_preview
            self.settings = None
            self.runtime_logging_mode = "basic"

        def set_runtime_logging_mode(self, mode: str) -> None:
            self.runtime_logging_mode = mode

        def log_basic(self, message: str, *, level: int = app_module.logging.INFO) -> None:
            _ = (message, level)

        def log_detailed(self, message: str, *, level: int = app_module.logging.INFO) -> bool:
            _ = (message, level)
            return True

    class DummyDashboardView(ft.Container):
        def __init__(self) -> None:
            super().__init__()
            self.on_send_message = None
            self.on_toggle_translation = None
            self.on_toggle_stt = None
            self.on_toggle_overlay = None
            self.on_toggle_peer_translation = None
            self.on_language_change = None
            self.runtime_log_detailed = None

        def apply_locale(self) -> None:
            return None

    class DummySettingsView(ft.Container):
        def __init__(self) -> None:
            super().__init__()
            self.on_settings_changed = None
            self.on_prompt_apply_settings = None
            self.on_providers_changed = None
            self.on_verify_api_key = None
            self.on_secret_cleared = None
            self.show_snackbar = None

        def set_overlay_runtime_state(self, *_args, **_kwargs) -> None:
            return None

        def apply_locale(self) -> None:
            return None

    class DummyLogsView(ft.Container):
        def __init__(self) -> None:
            super().__init__()
            self.on_mode_change = None

        def set_runtime_logging_mode(self, mode: str) -> None:
            _ = mode

        def apply_locale(self) -> None:
            return None

        async def scroll_to_bottom(self) -> None:
            return None

    monkeypatch.setattr(app_module, "GuiController", DummyController)
    monkeypatch.setattr(app_module, "DashboardView", DummyDashboardView)
    monkeypatch.setattr(app_module, "SettingsView", DummySettingsView)
    monkeypatch.setattr(app_module, "LogsView", DummyLogsView)
    monkeypatch.setattr(app_module, "AboutView", lambda: ft.Container())
    monkeypatch.setattr(app_module, "TitleBar", lambda _page: ft.Container())
    monkeypatch.setattr(app_module, "BottomNavBar", lambda on_change: ft.Container(data=on_change))
    monkeypatch.setattr(app_module, "register_fonts", lambda _page: None)
    monkeypatch.setattr(app_module, "get_app_theme", lambda **_kwargs: "theme")
    monkeypatch.setattr(app_module, "font_for_language", lambda _code: "font")
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")

    app = TranslatorApp(DummyPage(), config_path=Path("settings.json"))

    assert app.view_dashboard.runtime_log_detailed == app._log_detailed


def test_settings_view_pkce_callback_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_app_construction(monkeypatch)

    app = TranslatorApp(DummyPage(), config_path=Path("settings.json"))

    assert app.view_settings.on_request_openrouter_pkce == app._on_request_openrouter_pkce


def test_translator_app_4x3_window_keeps_shell_navigation_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    page = DummyPage()
    app = TranslatorApp(page, config_path=Path("settings.json"))
    monkeypatch.setattr(app.content_area, "update", lambda: None)

    assert app.content_area.padding == app_module.APP_CONTENT_PADDING
    assert app.layout.controls == [app.title_bar, app.content_area, app.bottom_nav]
    assert app.content_area.content is app.view_dashboard

    app._on_nav_change(1)
    assert app.content_area.content is app.view_settings
    assert app.content_area.padding == 0

    app._on_nav_change(2)
    assert app.content_area.content is app.view_logs
    assert app.content_area.padding == app_module.APP_CONTENT_PADDING

    app._on_nav_change(3)
    assert app.content_area.content is app.view_about
    assert app.content_area.padding == app_module.APP_CONTENT_PADDING

    app._on_nav_change(0)
    assert app.content_area.content is app.view_dashboard
    assert app.content_area.padding == app_module.APP_CONTENT_PADDING
    assert len(page.tasks) == 1


def test_app_tab_key_reverses_message_input_languages_only_on_dashboard() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    calls: list[str] = []
    dashboard = SimpleNamespace(handle_message_input_tab_key=lambda: calls.append("tab") or True)
    app.view_dashboard = dashboard
    app.content_area = DummyContent(content=dashboard)

    app._on_keyboard_event(
        SimpleNamespace(key="Tab", shift=False, ctrl=False, alt=False, meta=False)
    )
    app._on_keyboard_event(
        SimpleNamespace(key="Tab", shift=True, ctrl=False, alt=False, meta=False)
    )

    app.content_area.content = SimpleNamespace()
    app._on_keyboard_event(
        SimpleNamespace(key="Tab", shift=False, ctrl=False, alt=False, meta=False)
    )
    app._on_keyboard_event(SimpleNamespace(key="A", shift=False, ctrl=False, alt=False, meta=False))

    assert calls == ["tab"]


def test_on_runtime_logging_mode_change_updates_controller_and_logs_view() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    seen: list[str] = []

    def fake_set_mode(mode: str) -> None:
        seen.append(mode)
        app.controller.runtime_logging_mode = mode

    app.controller = SimpleNamespace(
        runtime_logging_mode="basic",
        set_runtime_logging_mode=fake_set_mode,
    )
    app.view_logs = SimpleNamespace(
        set_runtime_logging_mode=lambda mode: seen.append(f"view:{mode}")
    )

    app._on_runtime_logging_mode_change("detailed")

    assert seen == ["detailed", "view:detailed"]


@pytest.mark.asyncio
async def test_main_gui_routes_update_check_through_app_log_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = DummyPage()
    seen: dict[str, object] = {}

    class FakeController:
        async def start(self) -> None:
            seen["started"] = True

    class FakeApp:
        def __init__(self, incoming_page, *, config_path, debug_ui_preview=False):
            seen["init"] = (incoming_page, config_path, debug_ui_preview)
            seen["app"] = self
            self.page = incoming_page
            self.controller = FakeController()

        def _log_detailed(self, message: str, *, level: int = app_module.logging.INFO) -> None:
            _ = (message, level)

    async def fake_check_and_notify_update(incoming_page, *, log_detailed=None) -> None:
        seen["check"] = (incoming_page, log_detailed)

    monkeypatch.setattr(app_module, "TranslatorApp", FakeApp)
    monkeypatch.setattr(app_module, "_check_and_notify_update", fake_check_and_notify_update)

    await app_module.main_gui(page, config_path=Path("settings.json"))

    assert seen["started"] is True
    assert seen["check"][0] is page
    assert getattr(seen["check"][1], "__self__", None) is seen["app"]
    assert getattr(seen["check"][1], "__func__", None) is FakeApp._log_detailed
    assert seen["init"] == (page, Path("settings.json"), False)


@pytest.mark.asyncio
async def test_main_gui_forwards_debug_ui_preview_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = DummyPage()
    seen: dict[str, object] = {}

    class FakeController:
        async def start(self) -> None:
            seen["started"] = True

    class FakeApp:
        def __init__(self, incoming_page, *, config_path, debug_ui_preview=False):
            seen["init"] = (incoming_page, config_path, debug_ui_preview)
            self.page = incoming_page
            self.controller = FakeController()

        def _log_detailed(self, message: str, *, level: int = app_module.logging.INFO) -> None:
            _ = (message, level)

    async def fake_check_and_notify_update(incoming_page, *, log_detailed=None) -> None:
        seen["check"] = (incoming_page, log_detailed)

    monkeypatch.setattr(app_module, "TranslatorApp", FakeApp)
    monkeypatch.setattr(app_module, "_check_and_notify_update", fake_check_and_notify_update)

    await app_module.main_gui(
        page,
        config_path=Path("settings.json"),
        debug_ui_preview=True,
    )

    assert seen["started"] is True
    assert seen["init"] == (page, Path("settings.json"), True)
    assert seen["check"][0] is page


class PreviewDashboard:
    def __init__(self) -> None:
        self.managed_trial_calls: list[dict[str, object]] = []

    def set_managed_trial_state(self, **kwargs) -> None:
        self.managed_trial_calls.append(kwargs)
        raise AssertionError("debug preview must not write removed Dashboard trial-card state")


def test_debug_preview_surviving_managed_actions_are_snackbar_only() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.view_dashboard = PreviewDashboard()
    snackbar_calls: list[tuple[str, object]] = []
    app._show_snackbar = lambda message, bgcolor: snackbar_calls.append((message, bgcolor))

    assert not hasattr(app, "_set_debug_managed_trial_preview")
    assert not hasattr(app, "_preview_managed_normal")
    assert not hasattr(app, "_preview_managed_exhausted")
    assert not hasattr(app, "_preview_clear")

    app._preview_brake_notice()
    app._preview_revoked_notice()

    assert snackbar_calls == [
        (app_module.t("managed_release.brake"), ft.Colors.ORANGE_700),
        (app_module.t("managed_release.revoked_contact"), ft.Colors.ORANGE_700),
    ]
    assert app.view_dashboard.managed_trial_calls == []


def test_managed_release_ko_snackbar_copy_matches_requested_wording() -> None:
    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale("ko")

        assert (
            i18n_module.t("managed_release.brake")
            == "신규 인증이 잠시 중지된 상태에요. BYOK 방식으로 이용해주세요."
        )
        assert (
            i18n_module.t("managed_release.revoked_contact")
            == "엑세스 키가 손상되었어요. 저에게 연락해서 새 키를 받아가세요."
        )
    finally:
        i18n_module.set_locale(previous_locale)


def test_debug_preview_founder_letter_opens_dialog_with_readme_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    captured: dict[str, object] = {}
    opened_urls: list[str] = []
    previous_locale = i18n_module.get_locale()

    def fail_save(*_args, **_kwargs):
        pytest.fail("debug founder-letter preview must not save settings")

    app.controller = SimpleNamespace(settings=AppSettings(), _save_settings=fail_save)
    monkeypatch.setattr(
        app,
        "_on_request_openrouter_pkce",
        lambda *_args, **_kwargs: pytest.fail("debug founder-letter preview must not launch PKCE"),
        raising=False,
    )
    monkeypatch.setattr(
        app_module.webbrowser,
        "open",
        lambda url: opened_urls.append(url),
    )

    class FakeFounderLetterDialog:
        def __init__(self, page, *, on_readme=None, on_connect=None, on_contact=None):
            captured["page"] = page
            captured["on_readme"] = on_readme
            captured["on_connect"] = on_connect
            captured["on_contact"] = on_contact

        def open(self) -> None:
            captured["opened"] = True

    monkeypatch.setattr(app_module, "FounderLetterDialog", FakeFounderLetterDialog)

    try:
        i18n_module.set_locale("ko")

        app._preview_founder_letter()

        assert captured["page"] is app.page
        assert captured["opened"] is True
        assert callable(captured["on_readme"])
        assert captured["on_connect"] is None
        assert captured["on_contact"] is None
        assert opened_urls == []
        captured["on_readme"]()
    finally:
        i18n_module.set_locale(previous_locale)

    assert app._founder_letter_dialog is not None
    assert opened_urls == [
        "https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.ko.md#자신의-api-키-사용하기"
    ]


def test_founder_readme_url_for_locale_uses_origin_readme_pages() -> None:
    resolver = getattr(app_module, "founder_readme_url_for_locale", None)

    assert callable(resolver)
    assert resolver("ko") == (
        "https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.ko.md#자신의-api-키-사용하기"
    )
    assert resolver("zh-CN") == (
        "https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.zh-CN.md#使用您自己的-api-密钥"
    )
    assert resolver("ja") == (
        "https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.ja.md#自分のapiキーを使う"
    )
    assert resolver("en") == (
        "https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.md#using-your-own-api-keys"
    )
    assert resolver("fr") == (
        "https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.md#using-your-own-api-keys"
    )
    assert resolver(None) == (
        "https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.md#using-your-own-api-keys"
    )


def test_debug_preview_pkce_failure_only_shows_failure_snackbar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    seen: list[tuple[str, object]] = []

    monkeypatch.setattr(
        app,
        "_on_request_openrouter_pkce",
        lambda *_args, **_kwargs: pytest.fail("debug preview must not launch PKCE"),
        raising=False,
    )
    monkeypatch.setattr(
        app,
        "_show_snackbar",
        lambda message, bgcolor: seen.append((message, bgcolor)),
    )

    app._preview_pkce_failure()

    assert seen == [(app_module.t("openrouter.pkce.failed"), ft.Colors.ORANGE_700)]


def test_debug_preview_peer_translation_eula_opens_preview_safe_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    seen: dict[str, object] = {}

    class FakeDialog:
        def __init__(self, page, *, on_accept, on_cancel=None):
            seen["page"] = page
            seen["on_accept"] = on_accept
            seen["on_cancel"] = on_cancel

        def open(self):
            seen["opened"] = True

    monkeypatch.setattr(app_module, "PeerTranslationEulaDialog", FakeDialog)

    app._preview_peer_translation_eula()

    assert seen["page"] is app.page
    assert seen["opened"] is True
    seen["on_accept"]()
    if seen["on_cancel"] is not None:
        seen["on_cancel"]()
    assert not hasattr(app, "controller")


def test_debug_preview_discord_auth_opens_dialog_with_close_only_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
    app.controller = SimpleNamespace(
        settings=settings,
        config_path=Path("settings.json"),
        start_discord_managed_auth=lambda *_args, **_kwargs: pytest.fail(
            "debug Discord-auth preview must not start broker OAuth"
        ),
        reopen_discord_managed_auth_browser=lambda *_args, **_kwargs: pytest.fail(
            "debug Discord-auth preview must not reopen broker OAuth"
        ),
        cancel_discord_managed_auth=lambda *_args, **_kwargs: pytest.fail(
            "debug Discord-auth preview must not cancel broker OAuth"
        ),
    )
    monkeypatch.setattr(
        app,
        "_start_discord_managed_auth",
        lambda *_args, **_kwargs: pytest.fail(
            "debug Discord-auth preview must not call OAuth start hook"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        app,
        "_on_discord_managed_auth_byok",
        lambda *_args, **_kwargs: pytest.fail(
            "debug Discord-auth preview must not launch BYOK PKCE"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        app,
        "_on_request_openrouter_pkce",
        lambda *_args, **_kwargs: pytest.fail(
            "debug Discord-auth preview must not launch OpenRouter PKCE"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "save_settings",
        lambda *_args, **_kwargs: pytest.fail("debug Discord-auth preview must not save settings"),
    )
    monkeypatch.setattr(
        app_module.webbrowser,
        "open",
        lambda *_args, **_kwargs: pytest.fail(
            "debug Discord-auth preview must not open external URLs"
        ),
    )

    def open_preview_dialog():
        app._preview_discord_auth()
        dialog = app._discord_managed_auth_dialog
        assert dialog._dialog is app.page.opened[-1]
        return dialog, dialog._dialog

    for button_attr in ("_continue_button", "_close_button"):
        dialog, opened_dialog = open_preview_dialog()
        getattr(dialog, button_attr).on_click(None)
        assert app.page.closed[-1] is opened_dialog

    for button_attr in ("_reopen_browser_button", "_cancel_button"):
        dialog, opened_dialog = open_preview_dialog()
        dialog.set_waiting()
        getattr(dialog, button_attr).on_click(None)
        assert app.page.closed[-1] is opened_dialog

    assert app.page.tasks == []
    assert settings.openrouter.selected_source is OpenRouterCredentialSource.MANAGED


def test_discord_managed_auth_byok_launches_openrouter_pkce_with_byok_target() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
    settings.openrouter.llm_model = OpenRouterLLMModel.QWEN_35_FLASH_02_23
    app.controller = SimpleNamespace(settings=settings)
    pkce_calls: list[tuple[AppSettings, str]] = []
    app._on_request_openrouter_pkce = (
        lambda target_settings, *, launch_source="settings": pkce_calls.append(
            (target_settings, launch_source)
        )
    )
    app._show_snackbar = lambda *_args, **_kwargs: pytest.fail(
        "managed Discord auth BYOK should build a valid OpenRouter target"
    )

    app._on_discord_managed_auth_byok()

    assert len(pkce_calls) == 1
    target_settings, launch_source = pkce_calls[0]
    assert launch_source == "discord_auth"
    assert target_settings is not settings
    assert target_settings.provider.llm is LLMProviderName.OPENROUTER
    assert target_settings.openrouter.selected_source is OpenRouterCredentialSource.BYOK
    assert target_settings.openrouter.selection_alias is OpenRouterSelectionAlias.QWEN35_FLASH_BYOK
    assert target_settings.openrouter.llm_model is OpenRouterLLMModel.QWEN_35_FLASH_02_23
    assert settings.openrouter.selected_source is OpenRouterCredentialSource.MANAGED


def test_discord_managed_auth_byok_clears_managed_china_translation_state() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED_CHINA,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA,
        },
    )
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    settings.openrouter.llm_model = OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    settings.openrouter.provider_routing = OpenRouterProviderRouting.DEEPSEEK_ONLY
    app.controller = SimpleNamespace(settings=settings)

    target_settings = app._build_managed_openrouter_byok_target_settings()

    assert target_settings is not None
    assert target_settings.openrouter.selected_source is OpenRouterCredentialSource.BYOK
    assert (
        target_settings.openrouter.selection_alias
        is OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK
    )
    assert target_settings.openrouter.provider_routing is OpenRouterProviderRouting.DEFAULT
    assert target_settings.translation.connection is TranslationConnection.OPENROUTER
    assert (
        target_settings.translation.connection_history[TranslationModel.DEEPSEEK_V4_FLASH.value]
        is TranslationConnection.OPENROUTER
    )


@pytest.mark.asyncio
async def test_start_discord_managed_auth_uses_run_task_and_success_enables_translation() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dialog = SimpleNamespace(set_waiting_calls=0, close_calls=0)
    dialog.set_waiting = lambda: setattr(dialog, "set_waiting_calls", dialog.set_waiting_calls + 1)
    dialog.close = lambda: setattr(dialog, "close_calls", dialog.close_calls + 1)
    app._discord_managed_auth_dialog = dialog
    snackbar_calls: list[tuple[str, object]] = []
    enable_calls: list[bool] = []
    start_calls: list[str] = []
    dashboard_translation_calls: list[bool] = []
    hub = SimpleNamespace(llm=object(), translation_enabled=False)
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))

    async def fake_start_discord_managed_auth_from_dialog(**_kwargs) -> bool:
        start_calls.append("start")
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        hub.translation_enabled = enabled
        return True

    app.controller = SimpleNamespace(
        hub=hub,
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_discord_managed_auth()

    assert dialog.set_waiting_calls == 1
    assert start_calls == []
    assert len(app.page.tasks) == 1

    await app.page.tasks[0]()

    assert start_calls == ["start"]
    assert dialog.close_calls == 1
    assert snackbar_calls == [(app_module.t("discord_auth.success"), app_module.COLOR_SUCCESS)]
    assert enable_calls == [True]
    assert dashboard_translation_calls == [True]


@pytest.mark.asyncio
async def test_start_discord_managed_auth_passes_dialog_referral_id_to_controller() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dialog = SimpleNamespace(referral_id="not a referral id", set_waiting=lambda: None)
    app._discord_managed_auth_dialog = dialog
    start_kwargs: list[dict[str, object]] = []
    enable_calls: list[bool] = []

    async def fake_start_discord_managed_auth_from_dialog(**kwargs) -> bool:
        start_kwargs.append(kwargs)
        return False

    async def fake_set_translation_enabled(enabled: bool) -> None:
        enable_calls.append(enabled)

    app.controller = SimpleNamespace(
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_discord_managed_auth()
    await app.page.tasks[0]()

    assert len(start_kwargs) == 1
    assert start_kwargs[0]["referral_id"] == "not a referral id"
    assert callable(start_kwargs[0]["on_callback_received"])
    assert enable_calls == []


@pytest.mark.asyncio
async def test_start_discord_managed_auth_shows_referral_reward_snackbar_when_bonus_applied() -> (
    None
):
    previous_locale = i18n_module.get_locale()
    i18n_module.set_locale("ko")
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dialog = SimpleNamespace(referral_id="7KQ9M2", set_waiting=lambda: None, close=lambda: None)
    app._discord_managed_auth_dialog = dialog
    snackbar_calls: list[tuple[str, object]] = []
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))
    app.view_dashboard = SimpleNamespace(set_translation_enabled=lambda _enabled: None)
    hub = SimpleNamespace(llm=object(), translation_enabled=False)

    controller = SimpleNamespace(hub=hub)

    async def fake_start_discord_managed_auth_from_dialog(**_kwargs) -> bool:
        controller.last_discord_managed_auth_referral_bonus_applied = True
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        hub.translation_enabled = enabled
        return True

    controller.start_discord_managed_auth_from_dialog = fake_start_discord_managed_auth_from_dialog
    controller.set_translation_enabled = fake_set_translation_enabled
    app.controller = controller

    try:
        app._start_discord_managed_auth()
        await app.page.tasks[0]()

        assert snackbar_calls == [
            (app_module.t("discord_auth.success"), app_module.COLOR_SUCCESS),
            (app_module.t("discord_auth.referral_reward_applied"), app_module.COLOR_SUCCESS),
        ]
    finally:
        i18n_module.set_locale(previous_locale)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "referral_bonus_applied",
    [MISSING, False, None, "true", 1],
)
async def test_start_discord_managed_auth_omits_referral_snackbar_without_boolean_true(
    referral_bonus_applied: object,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dialog = SimpleNamespace(referral_id="7KQ9M2", set_waiting=lambda: None, close=lambda: None)
    app._discord_managed_auth_dialog = dialog
    snackbar_calls: list[tuple[str, object]] = []
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))
    app.view_dashboard = SimpleNamespace(set_translation_enabled=lambda _enabled: None)
    hub = SimpleNamespace(llm=object(), translation_enabled=False)
    controller = SimpleNamespace(hub=hub)

    async def fake_start_discord_managed_auth_from_dialog(**_kwargs) -> bool:
        if referral_bonus_applied is not MISSING:
            controller.last_discord_managed_auth_referral_bonus_applied = referral_bonus_applied
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        hub.translation_enabled = enabled
        return True

    controller.start_discord_managed_auth_from_dialog = fake_start_discord_managed_auth_from_dialog
    controller.set_translation_enabled = fake_set_translation_enabled
    app.controller = controller

    app._start_discord_managed_auth()
    await app.page.tasks[0]()

    assert snackbar_calls == [(app_module.t("discord_auth.success"), app_module.COLOR_SUCCESS)]


@pytest.mark.asyncio
async def test_start_discord_managed_auth_no_success_when_enable_fails() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dialog = SimpleNamespace(set_waiting=lambda: None, close=lambda: None)
    app._discord_managed_auth_dialog = dialog
    snackbar_calls: list[tuple[str, object]] = []
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))

    async def fake_start_discord_managed_auth_from_dialog(**_kwargs) -> bool:
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        return False

    app.controller = SimpleNamespace(
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_discord_managed_auth()
    await app.page.tasks[0]()

    assert enable_calls == [True]
    assert snackbar_calls == []
    assert dashboard_translation_calls == []


@pytest.mark.asyncio
async def test_start_discord_managed_auth_no_success_when_llm_unavailable_after_enable() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dialog = SimpleNamespace(set_waiting=lambda: None, close=lambda: None)
    app._discord_managed_auth_dialog = dialog
    snackbar_calls: list[tuple[str, object]] = []
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))

    async def fake_start_discord_managed_auth_from_dialog(**_kwargs) -> bool:
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        return True

    app.controller = SimpleNamespace(
        hub=SimpleNamespace(llm=None, translation_enabled=False),
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_discord_managed_auth()
    await app.page.tasks[0]()

    assert enable_calls == [True]
    assert snackbar_calls == []
    assert dashboard_translation_calls == []


@pytest.mark.asyncio
async def test_start_discord_managed_auth_failure_does_not_show_success_snackbar() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._discord_managed_auth_dialog = SimpleNamespace(set_waiting=lambda: None, close=lambda: None)
    snackbar_calls: list[tuple[str, object]] = []
    enable_calls: list[bool] = []
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))

    async def fake_start_discord_managed_auth_from_dialog(**_kwargs) -> bool:
        return False

    async def fake_set_translation_enabled(enabled: bool) -> None:
        enable_calls.append(enabled)

    app.controller = SimpleNamespace(
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_discord_managed_auth()
    await app.page.tasks[0]()

    assert snackbar_calls == []
    assert enable_calls == []


@pytest.mark.asyncio
async def test_start_qq_managed_auth_completes_success_only_after_translation_enable() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    completion_generations: list[int | None] = []
    close_calls: list[str] = []
    set_waiting_calls: list[str] = []
    dialog = SimpleNamespace(
        qq_identity="qq-user-123",
        credential="a" * 64,
        auth_generation=17,
        set_waiting=lambda: set_waiting_calls.append("waiting"),
        complete_success=lambda *, generation=None: (
            completion_generations.append(generation) or True
        ),
        close=lambda: close_calls.append("close"),
    )
    app._qq_managed_auth_dialog = dialog
    start_kwargs: list[dict[str, str]] = []
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    hub = SimpleNamespace(llm=object(), translation_enabled=False)
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda _message, _color: pytest.fail(
        "QQ success should be emitted through the dialog completion callback"
    )

    async def fake_start_qq_managed_auth_from_dialog(**kwargs) -> bool:
        start_kwargs.append(kwargs)
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        hub.translation_enabled = enabled
        return True

    app.controller = SimpleNamespace(
        hub=hub,
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_qq_managed_auth()

    assert set_waiting_calls == ["waiting"]
    assert start_kwargs == []
    assert len(app.page.tasks) == 1

    await app.page.tasks[0]()

    assert start_kwargs == [{"qq_identity": "qq-user-123", "credential": "a" * 64}]
    assert enable_calls == [True]
    assert dashboard_translation_calls == [True]
    assert completion_generations == [17]
    assert close_calls == []


@pytest.mark.asyncio
async def test_start_qq_managed_auth_does_not_complete_success_when_enable_fails() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    close_calls: list[str] = []
    dialog = SimpleNamespace(
        qq_identity="qq-user-123",
        credential="a" * 64,
        auth_generation=17,
        set_waiting=lambda: None,
        complete_success=lambda *, generation=None: pytest.fail(
            f"unexpected QQ success generation={generation}"
        ),
        close=lambda: close_calls.append("close"),
    )
    app._qq_managed_auth_dialog = dialog
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda _message, _color: None

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs) -> bool:
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        return False

    app.controller = SimpleNamespace(
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_qq_managed_auth()
    await app.page.tasks[0]()

    assert enable_calls == [True]
    assert dashboard_translation_calls == []
    assert close_calls == ["close"]


@pytest.mark.asyncio
async def test_start_qq_managed_auth_enable_exception_completes_translation_failure() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    recoverable_calls: list[dict[str, object]] = []
    translation_failure_generations: list[int | None] = []
    success_generations: list[int | None] = []
    close_calls: list[str] = []
    dialog = SimpleNamespace(
        qq_identity="qq-user-123",
        credential="a" * 64,
        auth_generation=17,
        set_waiting=lambda: None,
        set_recoverable_error=lambda message_key, **kwargs: recoverable_calls.append(
            {"message_key": message_key, **kwargs}
        )
        or True,
        complete_translation_enable_failed=lambda *, generation=None: (
            translation_failure_generations.append(generation) or True
        ),
        complete_success=lambda *, generation=None: success_generations.append(generation) or True,
        close=lambda: close_calls.append("close"),
    )
    app._qq_managed_auth_dialog = dialog
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda _message, _color: pytest.fail(
        "translation-enable exception should complete through dialog failure path"
    )

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs) -> bool:
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        raise RuntimeError("UNSAFE_QQ_ENABLE_EXCEPTION_DETAIL")

    app.controller = SimpleNamespace(
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_qq_managed_auth()
    await app.page.tasks[0]()

    assert enable_calls == [True]
    assert translation_failure_generations == [17]
    assert recoverable_calls == []
    assert success_generations == []
    assert dashboard_translation_calls == []
    assert close_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["auth", "enable"])
async def test_start_qq_managed_auth_exception_logs_do_not_include_raw_message(
    caplog: pytest.LogCaptureFixture,
    failure_stage: str,
) -> None:
    caplog.set_level(app_module.logging.ERROR, logger="puripuly_heart.ui.app")
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    raw_message = f"UNSAFE_QQ_{failure_stage.upper()}_EXCEPTION_DETAIL"
    dialog = SimpleNamespace(
        qq_identity="qq-user-123",
        credential="a" * 64,
        auth_generation=17,
        set_waiting=lambda: None,
        set_recoverable_error=lambda *_args, **_kwargs: True,
        complete_translation_enable_failed=lambda *, generation=None: True,
        close=lambda: None,
    )
    app._qq_managed_auth_dialog = dialog
    app.view_dashboard = SimpleNamespace(set_translation_enabled=lambda _enabled: None)
    app._show_snackbar = lambda _message, _color: None

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs) -> bool:
        if failure_stage == "auth":
            raise RuntimeError(raw_message)
        return True

    async def fake_set_translation_enabled(_enabled: bool) -> bool:
        if failure_stage == "enable":
            raise RuntimeError(raw_message)
        return True

    app.controller = SimpleNamespace(
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_qq_managed_auth()
    await app.page.tasks[0]()

    assert raw_message not in caplog.text


@pytest.mark.asyncio
async def test_start_qq_managed_auth_late_success_after_close_does_not_enable_translation() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    completion_generations: list[int | None] = []
    close_calls: list[str] = []
    recoverable_calls: list[str] = []
    translation_failure_generations: list[int | None] = []
    dialog = SimpleNamespace(
        qq_identity="qq-user-123",
        credential="a" * 64,
        auth_generation=17,
        is_open=True,
        set_waiting=lambda: None,
        set_recoverable_error=lambda message_key, **_kwargs: recoverable_calls.append(message_key)
        or True,
        complete_translation_enable_failed=lambda *, generation=None: (
            translation_failure_generations.append(generation) or True
        ),
        complete_success=lambda *, generation=None: (
            completion_generations.append(generation) and False
        ),
        close=lambda: close_calls.append("close"),
    )
    app._qq_managed_auth_dialog = dialog
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    hub = SimpleNamespace(llm=object(), translation_enabled=False)
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda _message, _color: pytest.fail(
        "stale QQ success should not show success"
    )
    start_entered = asyncio.Event()
    release_start = asyncio.Event()

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs) -> bool:
        start_entered.set()
        await release_start.wait()
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        hub.translation_enabled = enabled
        return True

    app.controller = SimpleNamespace(
        hub=hub,
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_qq_managed_auth()
    task = asyncio.create_task(app.page.tasks[0]())
    await start_entered.wait()
    dialog.auth_generation = 18
    dialog.is_open = False
    release_start.set()
    await task

    assert enable_calls == []
    assert dashboard_translation_calls == []
    assert completion_generations == []
    assert translation_failure_generations == []
    assert recoverable_calls == []
    assert close_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["false", "exception"])
async def test_start_qq_managed_auth_recoverable_failure_keeps_dialog_open(
    failure_mode: str,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    recoverable_calls: list[dict[str, object]] = []
    close_calls: list[str] = []
    dialog = SimpleNamespace(
        qq_identity="qq-user-123",
        credential="a" * 64,
        auth_generation=17,
        set_waiting=lambda: None,
        set_recoverable_error=lambda message_key, **kwargs: recoverable_calls.append(
            {"message_key": message_key, **kwargs}
        )
        or True,
        close=lambda: close_calls.append("close"),
    )
    app._qq_managed_auth_dialog = dialog
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda _message, _color: pytest.fail(
        "recoverable QQ auth failure should stay in the dialog"
    )

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs) -> bool:
        if failure_mode == "exception":
            raise RuntimeError("network unavailable")
        return False

    async def fake_set_translation_enabled(enabled: bool) -> None:
        enable_calls.append(enabled)

    app.controller = SimpleNamespace(
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_qq_managed_auth()
    await app.page.tasks[0]()

    assert recoverable_calls == [
        {
            "message_key": "qq_auth.error.retry",
            "clear_credential": False,
            "message_kwargs": {},
            "generation": 17,
        }
    ]
    assert enable_calls == []
    assert dashboard_translation_calls == []
    assert close_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("subcode", "message_key", "message_kwargs", "expected_clear", "expected_kwargs"),
    [
        (
            "qq_credential_invalid",
            "qq_auth.error.credential_mismatch",
            {},
            True,
            {},
        ),
        (
            "qq_credential_mismatch",
            "qq_auth.error.credential_mismatch",
            {},
            True,
            {},
        ),
        (
            "ip_rate_limited",
            "qq_auth.error.retry",
            {"retry_after_ms": 999_999_999, "details": "raw broker detail"},
            False,
            {"retry_after_ms": 86_400_000},
        ),
    ],
)
async def test_start_qq_managed_auth_result_like_recoverable_errors_update_dialog(
    subcode: str,
    message_key: str,
    message_kwargs: dict[str, object],
    expected_clear: bool,
    expected_kwargs: dict[str, object],
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    recoverable_calls: list[dict[str, object]] = []
    close_calls: list[str] = []
    dialog = SimpleNamespace(
        qq_identity="qq-user-123",
        credential="a" * 64,
        auth_generation=17,
        set_waiting=lambda: None,
        set_recoverable_error=lambda message_key, **kwargs: recoverable_calls.append(
            {"message_key": message_key, **kwargs}
        )
        or True,
        close=lambda: close_calls.append("close"),
    )
    app._qq_managed_auth_dialog = dialog
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda _message, _color: pytest.fail(
        "recoverable QQ auth failure should stay in the dialog"
    )
    result = SimpleNamespace(
        message_key=message_key,
        message_kwargs=message_kwargs,
        diagnostics=SimpleNamespace(subcode=subcode),
    )

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs):
        return result

    async def fake_set_translation_enabled(enabled: bool) -> None:
        enable_calls.append(enabled)

    app.controller = SimpleNamespace(
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_qq_managed_auth()
    await app.page.tasks[0]()

    assert recoverable_calls == [
        {
            "message_key": message_key,
            "clear_credential": expected_clear,
            "message_kwargs": expected_kwargs,
            "generation": 17,
        }
    ]
    assert enable_calls == []
    assert dashboard_translation_calls == []
    assert close_calls == []


@pytest.mark.asyncio
async def test_start_qq_managed_auth_ignored_success_completion_does_not_mutate_ui() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    completion_generations: list[int | None] = []
    close_calls: list[str] = []
    dialog = SimpleNamespace(
        qq_identity="qq-user-123",
        credential="a" * 64,
        auth_generation=17,
        set_waiting=lambda: None,
        complete_success=lambda *, generation=None: (
            completion_generations.append(generation) and False
        ),
        close=lambda: close_calls.append("close"),
    )
    app._qq_managed_auth_dialog = dialog
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    hub = SimpleNamespace(llm=object(), translation_enabled=False)
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda _message, _color: pytest.fail(
        "stale QQ completion should not show success"
    )

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs) -> bool:
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        hub.translation_enabled = enabled
        return True

    app.controller = SimpleNamespace(
        hub=hub,
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_qq_managed_auth()
    await app.page.tasks[0]()

    assert completion_generations == [17]
    assert enable_calls == [True]
    assert dashboard_translation_calls == []
    assert close_calls == []


@pytest.mark.asyncio
async def test_cancel_qq_managed_auth_prevents_late_failure_side_effects() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    close_calls: list[str] = []
    recoverable_calls: list[dict[str, object]] = []
    key_unavailable_generations: list[int | None] = []
    dialog = SimpleNamespace(
        qq_identity="qq-synthetic-user",
        credential="a" * 64,
        auth_generation=17,
        is_open=True,
        set_waiting=lambda: None,
        set_recoverable_error=lambda message_key, **kwargs: recoverable_calls.append(
            {"message_key": message_key, **kwargs}
        )
        or True,
        complete_key_unavailable=lambda *, generation=None: (
            key_unavailable_generations.append(generation) or True
        ),
        close=lambda: close_calls.append("close"),
    )
    app._qq_managed_auth_dialog = dialog
    snackbar_calls: list[tuple[str, object]] = []
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    cancel_calls: list[str] = []
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))
    start_entered = asyncio.Event()
    release_start = asyncio.Event()
    late_result = SimpleNamespace(
        message_key="qq_auth.error.credential_mismatch",
        message_kwargs={},
        diagnostics=SimpleNamespace(subcode="qq_credential_invalid"),
    )

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs):
        start_entered.set()
        await release_start.wait()
        return late_result

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        return True

    app.controller = SimpleNamespace(
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
        cancel_qq_managed_auth=lambda: cancel_calls.append("cancel"),
    )

    app._start_qq_managed_auth()
    task = asyncio.create_task(app.page.tasks[0]())
    await start_entered.wait()

    app._cancel_qq_managed_auth()
    release_start.set()
    await task

    assert close_calls == ["close"]
    assert cancel_calls == ["cancel"]
    assert recoverable_calls == []
    assert key_unavailable_generations == []
    assert snackbar_calls == []
    assert enable_calls == []
    assert dashboard_translation_calls == []


@pytest.mark.asyncio
async def test_start_qq_managed_auth_stale_failure_does_not_mutate_ui() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    recoverable_calls: list[dict[str, object]] = []
    close_calls: list[str] = []
    dialog = SimpleNamespace(
        qq_identity="qq-synthetic-user",
        credential="a" * 64,
        auth_generation=17,
        is_open=True,
        set_waiting=lambda: None,
        set_recoverable_error=lambda message_key, **kwargs: recoverable_calls.append(
            {"message_key": message_key, **kwargs}
        )
        or True,
        close=lambda: close_calls.append("close"),
    )
    app._qq_managed_auth_dialog = dialog
    snackbar_calls: list[tuple[str, object]] = []
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))
    start_entered = asyncio.Event()
    release_start = asyncio.Event()
    stale_result = SimpleNamespace(
        message_key="qq_auth.error.credential_mismatch",
        message_kwargs={},
        diagnostics=SimpleNamespace(subcode="qq_credential_invalid"),
    )

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs):
        start_entered.set()
        await release_start.wait()
        return stale_result

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        return True

    app.controller = SimpleNamespace(
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_qq_managed_auth()
    task = asyncio.create_task(app.page.tasks[0]())
    await start_entered.wait()
    dialog.auth_generation = 18
    release_start.set()
    await task

    assert recoverable_calls == []
    assert snackbar_calls == []
    assert enable_calls == []
    assert dashboard_translation_calls == []
    assert close_calls == []


@pytest.mark.asyncio
async def test_cancel_discord_managed_auth_prevents_late_success_and_enable() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dialog = SimpleNamespace(set_waiting=lambda: None, close_calls=0)
    dialog.close = lambda: setattr(dialog, "close_calls", dialog.close_calls + 1)
    app._discord_managed_auth_dialog = dialog
    snackbar_calls: list[tuple[str, object]] = []
    enable_calls: list[bool] = []
    dashboard_translation_calls: list[bool] = []
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))
    start_entered = asyncio.Event()
    release_start = asyncio.Event()

    async def fake_start_discord_managed_auth_from_dialog(**_kwargs) -> bool:
        start_entered.set()
        await release_start.wait()
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        return True

    app.controller = SimpleNamespace(
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )

    app._start_discord_managed_auth()
    task = asyncio.create_task(app.page.tasks[0]())
    await start_entered.wait()

    app._cancel_discord_managed_auth()
    release_start.set()
    await task

    assert dialog.close_calls == 1
    assert snackbar_calls == []
    assert enable_calls == []
    assert dashboard_translation_calls == []


def test_discord_managed_auth_waiting_hides_reopen_when_controller_cannot_reopen() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.controller = SimpleNamespace()

    app.show_discord_managed_auth_dialog(preview=False)
    dialog = app._discord_managed_auth_dialog
    dialog.set_waiting()

    assert dialog._reopen_browser_button is None
    assert [control.text for control in dialog._actions.controls] == [
        app_module.t("discord_auth.cancel")
    ]


def test_peer_translation_toggle_first_enable_opens_eula_without_running_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = AppSettings()
    settings.ui.peer_translation_eula_accepted = False
    app.controller = SimpleNamespace(settings=settings)
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        app,
        "_show_peer_translation_eula",
        lambda on_accept: seen.setdefault("on_accept", on_accept),
    )

    app._on_peer_translation_toggle(True)

    assert "on_accept" in seen
    assert app.page.tasks == []
    assert settings.ui.peer_translation_eula_accepted is False


@pytest.mark.asyncio
async def test_peer_translation_toggle_after_eula_acceptance_saves_and_enables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = AppSettings()
    settings.ui.peer_translation_eula_accepted = False
    calls: list[bool] = []

    async def fake_enable(enabled: bool):
        calls.append(enabled)

    app.controller = SimpleNamespace(
        settings=settings,
        config_path="settings.json",
        set_peer_translation_enabled=fake_enable,
    )
    saves: list[tuple[str, AppSettings]] = []
    monkeypatch.setattr(app_module, "save_settings", lambda path, cfg: saves.append((path, cfg)))

    app._accept_peer_translation_eula_and_enable()
    await app.page.tasks[0]()

    assert settings.ui.peer_translation_eula_accepted is True
    assert calls == [True]
    assert saves == [("settings.json", settings)]


@pytest.mark.asyncio
async def test_peer_translation_toggle_with_existing_acceptance_enables_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = AppSettings()
    settings.ui.peer_translation_eula_accepted = True
    calls: list[bool] = []
    monkeypatch.setattr(
        app,
        "_show_peer_translation_eula",
        lambda _on_accept: pytest.fail("accepted peer translation should not reopen EULA"),
    )

    async def fake_enable(enabled: bool):
        calls.append(enabled)

    app.controller = SimpleNamespace(settings=settings, set_peer_translation_enabled=fake_enable)

    app._on_peer_translation_toggle(True)
    await app.page.tasks[0]()

    assert calls == [True]


@pytest.mark.asyncio
async def test_peer_translation_disable_does_not_open_eula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = AppSettings()
    settings.ui.peer_translation_eula_accepted = False
    calls: list[bool] = []
    monkeypatch.setattr(
        app,
        "_show_peer_translation_eula",
        lambda _on_accept: pytest.fail("disabling peer translation should not show EULA"),
    )

    async def fake_enable(enabled: bool):
        calls.append(enabled)

    app.controller = SimpleNamespace(settings=settings, set_peer_translation_enabled=fake_enable)

    app._on_peer_translation_toggle(False)
    await app.page.tasks[0]()

    assert calls == [False]


@pytest.mark.asyncio
async def test_on_nav_change_merges_current_languages_into_prompt_only_apply() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._current_tab = 1
    app.view_dashboard = object()
    app.view_logs = SimpleNamespace(scroll_to_bottom=lambda: asyncio.sleep(0))
    app.view_about = object()
    pending_settings = object()
    merged_settings = object()
    app.view_settings = SimpleNamespace(
        has_provider_changes=False,
        has_pending_prompt_changes=True,
        consume_prompt_apply_settings=lambda: pending_settings,
        refresh_prompt_if_empty=lambda: None,
    )
    app.content_area = DummyContent()
    events: list[tuple[str, object]] = []

    def fake_merge_settings(settings) -> object:
        events.append(("merge", settings))
        return merged_settings

    async def fake_apply_settings(settings) -> None:
        events.append(("apply", settings))

    app.controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        apply_settings=fake_apply_settings,
        apply_providers=lambda _settings=None: asyncio.sleep(0),
    )

    app._on_nav_change(0)

    assert app.content_area.content is app.view_dashboard
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert events == [("merge", pending_settings), ("apply", merged_settings)]


@pytest.mark.asyncio
async def test_on_nav_change_applies_provider_changes_when_leaving_settings() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._current_tab = 1
    app.view_dashboard = object()
    app.view_logs = SimpleNamespace(scroll_to_bottom=lambda: asyncio.sleep(0))
    app.view_about = object()
    app.view_settings = SimpleNamespace(
        has_provider_changes=True,
        consume_provider_apply_settings=lambda: "merged-settings",
        refresh_prompt_if_empty=lambda: None,
    )
    app.content_area = DummyContent()
    seen: list[object] = []

    async def fake_apply_providers(settings) -> None:
        seen.append(settings)

    app.controller = SimpleNamespace(apply_providers=fake_apply_providers)

    app._on_nav_change(0)
    assert app.content_area.content is app.view_dashboard
    assert app.view_settings.has_provider_changes is False
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert seen == ["merged-settings"]


@pytest.mark.asyncio
async def test_on_providers_changed_applies_consumed_provider_draft() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._settings_mutation_queue = []
    app._settings_mutation_worker_active = False
    app.view_settings = SimpleNamespace(
        has_provider_changes=True,
        consume_provider_apply_settings=lambda: "managed-settings",
    )
    seen: list[object] = []

    async def fake_apply_providers(settings=None) -> None:
        seen.append(settings)

    app.controller = SimpleNamespace(apply_providers=fake_apply_providers)

    app._on_providers_changed()

    assert app.view_settings.has_provider_changes is False
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert seen == ["managed-settings"]


@pytest.mark.asyncio
async def test_on_nav_change_refreshes_prompt_and_schedules_log_scroll() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._current_tab = 0
    refreshed = {"count": 0}
    scrolled = {"count": 0}

    async def fake_scroll_to_bottom():
        scrolled["count"] += 1

    app.view_dashboard = object()
    app.view_settings = SimpleNamespace(
        has_provider_changes=False,
        refresh_prompt_if_empty=lambda: refreshed.__setitem__("count", refreshed["count"] + 1),
    )
    app.view_logs = SimpleNamespace(scroll_to_bottom=fake_scroll_to_bottom)
    app.view_about = object()
    app.content_area = DummyContent()
    app.controller = SimpleNamespace(apply_providers=lambda _settings=None: asyncio.sleep(0))

    app._on_nav_change(1)
    assert app.content_area.content is app.view_settings
    assert refreshed["count"] == 1

    app._on_nav_change(2)
    assert app.content_area.content is app.view_logs
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert scrolled["count"] == 1


@pytest.mark.asyncio
async def test_on_nav_change_applies_pending_prompt_changes_when_leaving_settings() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._current_tab = 1
    app.view_dashboard = object()
    app.view_logs = SimpleNamespace(scroll_to_bottom=lambda: asyncio.sleep(0))
    app.view_about = object()
    pending_settings = object()
    merged_settings = object()
    merge_calls: list[object] = []
    app.view_settings = SimpleNamespace(
        has_provider_changes=False,
        has_pending_prompt_changes=True,
        consume_prompt_apply_settings=lambda: pending_settings,
        refresh_prompt_if_empty=lambda: None,
    )
    app.content_area = DummyContent()
    seen: list[object] = []

    def fake_merge_settings(settings) -> object:
        merge_calls.append(settings)
        return merged_settings

    async def fake_apply_settings(settings) -> None:
        seen.append(settings)

    app.controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        apply_settings=fake_apply_settings,
        apply_providers=lambda _settings=None: asyncio.sleep(0),
    )

    app._on_nav_change(0)

    assert app.content_area.content is app.view_dashboard
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert merge_calls == [pending_settings]
    assert seen == [merged_settings]


@pytest.mark.asyncio
async def test_on_prompt_apply_settings_merges_current_languages_before_apply_settings() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    pending_settings = object()
    merged_settings = object()
    events: list[tuple[str, object]] = []

    def fake_merge_settings(settings) -> object:
        events.append(("merge", settings))
        return merged_settings

    async def fake_apply_settings(settings) -> None:
        events.append(("apply", settings))

    app.controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        apply_settings=fake_apply_settings,
    )

    app._on_prompt_apply_settings(pending_settings)

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert events == [("merge", pending_settings), ("apply", merged_settings)]


@pytest.mark.asyncio
async def test_prompt_apply_keeps_dashboard_target_for_next_request() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    pending_settings = AppSettings()
    pending_settings.languages.target_language = "en"
    merged_settings = AppSettings()
    merged_settings.languages.target_language = "ja"
    applied_targets: list[str] = []

    def fake_merge_settings(settings: AppSettings) -> AppSettings:
        assert settings is pending_settings
        return merged_settings

    async def fake_apply_settings(settings: AppSettings) -> None:
        applied_targets.append(settings.languages.target_language)

    app.controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        apply_settings=fake_apply_settings,
    )

    app._on_prompt_apply_settings(pending_settings)

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert pending_settings.languages.target_language == "en"
    assert applied_targets == ["ja"]


@pytest.mark.asyncio
async def test_on_settings_changed_applies_raw_settings_without_prompt_merge() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    raw_settings = object()
    seen: list[object] = []

    def fake_merge_settings(_settings) -> object:
        raise AssertionError("prompt merge should not run for generic settings changes")

    async def fake_apply_settings(settings) -> None:
        seen.append(settings)

    app.controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        apply_settings=fake_apply_settings,
    )

    app._on_settings_changed(raw_settings)

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert seen == [raw_settings]


@pytest.mark.asyncio
async def test_start_microphone_test_success_opens_percentage_modal() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()
    start_kwargs: list[dict[str, object]] = []

    async def fake_start_microphone_test(**kwargs) -> bool:
        start_kwargs.append(dict(kwargs))
        kwargs["meter_callback"](0.37)
        return True

    app.controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=lambda: None,
        microphone_test_active=True,
    )

    app._on_start_microphone_test()
    await app.page.tasks[0]()

    assert len(app.page.opened) == 1
    dialog = app.page.opened[0]
    assert "37%" in _dialog_text_values(dialog)
    percent_text = next(
        node
        for node in _iter_control_tree(dialog)
        if isinstance(node, ft.Text) and node.value == "37%"
    )
    assert percent_text.color == app_module.COLOR_PRIMARY
    assert percent_text.size == 96
    assert i18n_module.t("settings.microphone_test.host_api_hint") in _dialog_text_values(dialog)
    modal_panel = next(
        node
        for node in _dialog_containers(dialog)
        if getattr(node, "width", None) == 450 and getattr(node, "height", None) == 500
    )
    assert modal_panel.width == 450
    assert modal_panel.height == 500
    hint_text = next(
        node
        for node in _iter_control_tree(dialog)
        if isinstance(node, ft.Text)
        and node.value == i18n_module.t("settings.microphone_test.host_api_hint")
    )
    assert hint_text.size == 28
    number_container = next(
        node
        for node in _dialog_containers(dialog)
        if getattr(node, "content", None) is percent_text
    )
    assert number_container.bgcolor == ft.Colors.TRANSPARENT
    assert dialog.modal is False
    assert not any(isinstance(node, ft.IconButton) for node in _iter_control_tree(dialog))
    assert "meter_callback" in start_kwargs[0]


@pytest.mark.asyncio
async def test_start_microphone_test_callback_uses_page_run_task() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()
    calls: list[str] = []

    async def fake_start_microphone_test() -> bool:
        calls.append("start")
        return True

    app.controller = SimpleNamespace(start_microphone_test=fake_start_microphone_test)

    app._on_start_microphone_test()

    assert len(app.page.tasks) == 1
    assert calls == []
    await app.page.tasks[0]()
    assert calls == ["start"]
    assert len(app.page.opened) == 1


@pytest.mark.asyncio
async def test_start_microphone_test_updates_modal_meter_on_success() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()
    start_kwargs: list[dict[str, object]] = []

    async def fake_start_microphone_test(**kwargs) -> bool:
        start_kwargs.append(dict(kwargs))
        callback = kwargs["meter_callback"]
        callback(0.37)
        return True

    app.controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=lambda: None,
        microphone_test_active=True,
    )

    app._on_start_microphone_test()
    await app.page.tasks[0]()

    assert len(app.page.opened) == 1
    assert "37%" in _dialog_text_values(app.page.opened[0])
    assert "meter_callback" in start_kwargs[0]


@pytest.mark.asyncio
async def test_start_microphone_test_false_opens_failure_modal() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()

    async def fake_start_microphone_test(**_kwargs) -> bool:
        return False

    app.controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
    )

    app._on_start_microphone_test()
    await app.page.tasks[0]()

    assert len(app.page.opened) == 1
    assert i18n_module.t("settings.microphone_test.start_failed") in _dialog_text_values(
        app.page.opened[0]
    )
    assert i18n_module.t("settings.microphone_test.host_api_hint") in _dialog_text_values(
        app.page.opened[0]
    )


@pytest.mark.asyncio
async def test_start_microphone_test_false_modal_has_no_close_button() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()

    async def fake_start_microphone_test(**_kwargs) -> bool:
        return False

    app.controller = SimpleNamespace(start_microphone_test=fake_start_microphone_test)

    app._on_start_microphone_test()
    await app.page.tasks[0]()

    assert len(app.page.opened) == 1
    assert not any(
        isinstance(node, ft.IconButton) for node in _iter_control_tree(app.page.opened[0])
    )


@pytest.mark.asyncio
async def test_microphone_test_meter_callback_updates_modal_percentage() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()
    callbacks: list[object] = []

    async def fake_start_microphone_test(**kwargs) -> bool:
        callbacks.append(kwargs["meter_callback"])
        return True

    app.controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=lambda: None,
        microphone_test_active=True,
    )

    app._on_start_microphone_test()
    await app.page.tasks[0]()
    callbacks[0](0.82)

    assert "82%" in _dialog_text_values(app.page.opened[0])


@pytest.mark.asyncio
async def test_stop_microphone_test_stops_runtime_through_settings_queue() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()
    stop_calls: list[str] = []

    async def fake_start_microphone_test(**_kwargs) -> bool:
        return True

    async def fake_stop_microphone_test() -> None:
        stop_calls.append("stop")

    app.controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=fake_stop_microphone_test,
        microphone_test_active=True,
    )

    app._on_stop_microphone_test()

    assert stop_calls == []
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert stop_calls == ["stop"]


@pytest.mark.asyncio
async def test_microphone_test_backdrop_dismiss_stops_runtime() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()
    stop_calls: list[str] = []

    async def fake_start_microphone_test(**_kwargs) -> bool:
        return True

    async def fake_stop_microphone_test() -> None:
        stop_calls.append("stop")

    app.controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=fake_stop_microphone_test,
        microphone_test_active=True,
    )

    app._on_start_microphone_test()
    await app.page.tasks[0]()

    assert len(app.page.opened) == 1
    app._microphone_test_dialog._handle_dismiss(None)

    assert len(app.page.tasks) == 2
    await app.page.tasks[1]()
    assert stop_calls == ["stop"]


@pytest.mark.asyncio
async def test_navigation_cleanup_closes_microphone_test_modal_and_stops_runtime() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()
    stop_calls: list[str] = []

    async def fake_start_microphone_test(**_kwargs) -> bool:
        return True

    async def fake_stop_microphone_test() -> None:
        stop_calls.append("stop")

    app.controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=fake_stop_microphone_test,
        microphone_test_active=True,
    )

    app._on_start_microphone_test()
    await app.page.tasks[0]()
    opened_dialog = app.page.opened[0]

    app._close_open_dialog_for_navigation()

    assert app.page.closed == [opened_dialog]
    assert len(app.page.tasks) == 2
    await app.page.tasks[1]()
    assert stop_calls == ["stop"]


@pytest.mark.asyncio
async def test_settings_apply_closes_microphone_test_modal_after_audio_cleanup() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()
    events: list[str] = []
    controller = SimpleNamespace(microphone_test_active=True)

    async def fake_start_microphone_test(**_kwargs) -> bool:
        events.append("start")
        return True

    async def fake_apply_settings(settings) -> None:
        events.append(f"apply:{settings}")
        controller.microphone_test_active = False

    async def fake_stop_microphone_test() -> None:
        events.append("stop")

    controller.start_microphone_test = fake_start_microphone_test
    controller.apply_settings = fake_apply_settings
    controller.stop_microphone_test = fake_stop_microphone_test
    app.controller = controller

    app._on_start_microphone_test()
    await app.page.tasks[0]()

    app._on_settings_changed("audio")
    await app.page.tasks[1]()

    assert events == ["start", "apply:audio"]
    assert len(app.page.opened) == 1
    assert app.page.closed == [app.page.opened[0]]


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ("en", "Microphone level"),
        ("ko", "마이크 입력 레벨"),
        ("ja", "マイク入力レベル"),
        ("zh-CN", "麦克风输入电平"),
    ],
)
def test_microphone_test_level_accessibility_label_is_localized(
    locale: str,
    expected: str,
) -> None:
    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale(locale)
        assert i18n_module.t("settings.microphone_test.level_label") == expected
    finally:
        i18n_module.set_locale(previous_locale)


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ("en", "Couldn’t start microphone test"),
        ("ko", "마이크 테스트를 시작하지 못했어요"),
        ("ja", "マイクテストを開始できませんでした"),
        ("zh-CN", "无法开始麦克风测试"),
    ],
)
def test_microphone_test_start_failed_label_is_localized(
    locale: str,
    expected: str,
) -> None:
    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale(locale)
        assert i18n_module.t("settings.microphone_test.start_failed") == expected
    finally:
        i18n_module.set_locale(previous_locale)


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        (
            "en",
            "If audio isn’t being captured, change Host API to Auto or MME, then restart the app.",
        ),
        (
            "ko",
            "오디오 캡쳐가 되지 않으면 호스트 API를 자동선택 혹은 MME로 변경 후 앱을 재시작해주세요",
        ),
        (
            "ja",
            "音声がキャプチャされない場合は、ホストAPIを自動選択またはMMEに変更してからアプリを再起動してください",
        ),
        ("zh-CN", "如果无法捕获音频，请将主机 API 改为自动选择或 MME，然后重启应用"),
    ],
)
def test_microphone_test_host_api_hint_is_localized(
    locale: str,
    expected: str,
) -> None:
    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale(locale)
        assert i18n_module.t("settings.microphone_test.host_api_hint") == expected
    finally:
        i18n_module.set_locale(previous_locale)


@pytest.mark.asyncio
async def test_start_microphone_test_waits_for_pending_settings_queue() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()
    events: list[str] = []

    async def fake_apply_settings(settings) -> None:
        events.append(f"apply:{settings}")

    async def fake_start_microphone_test() -> bool:
        events.append("start")
        return True

    app.controller = SimpleNamespace(
        apply_settings=fake_apply_settings,
        start_microphone_test=fake_start_microphone_test,
    )

    app._on_settings_changed("audio")
    app._on_start_microphone_test()

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert events == ["apply:audio", "start"]


def test_on_request_openrouter_pkce_uses_settings_mutation_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)
    app = TranslatorApp(DummyPage(), config_path=Path("settings.json"))
    target_settings = AppSettings()
    queued: list[object] = []
    monkeypatch.setattr(app, "_queue_settings_mutation_task", queued.append)

    app._on_request_openrouter_pkce(target_settings, launch_source="settings")

    assert len(queued) == 1


def test_on_request_openrouter_pkce_reopens_existing_auth_url_while_flow_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    target_settings = AppSettings()
    reopen_calls: list[str] = []

    async def fake_connect_openrouter_via_pkce(
        *, target_settings: AppSettings, launch_source: str
    ) -> bool:
        _ = (target_settings, launch_source)
        return False

    app.controller = SimpleNamespace(
        connect_openrouter_via_pkce=fake_connect_openrouter_via_pkce,
        reopen_openrouter_pkce_authorization_url=lambda: reopen_calls.append("reopen") or True,
        settings=AppSettings(),
        config_path=Path("settings.json"),
    )
    app.view_settings = SimpleNamespace(
        refresh_after_openrouter_pkce_success=lambda *_args, **_kwargs: None,
        load_from_settings=lambda *_args, **_kwargs: None,
    )
    queued: list[object] = []
    monkeypatch.setattr(app, "_queue_settings_mutation_task", queued.append)

    app._on_request_openrouter_pkce(target_settings, launch_source="settings")
    app._on_request_openrouter_pkce(target_settings, launch_source="settings")

    assert len(queued) == 1
    assert reopen_calls == ["reopen"]


@pytest.mark.asyncio
async def test_on_request_openrouter_pkce_ignores_duplicate_while_flow_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    target_settings = AppSettings()
    pkce_calls: list[str] = []

    async def fake_connect_openrouter_via_pkce(
        *, target_settings: AppSettings, launch_source: str
    ) -> bool:
        _ = target_settings
        pkce_calls.append(launch_source)
        return False

    app.controller = SimpleNamespace(
        connect_openrouter_via_pkce=fake_connect_openrouter_via_pkce,
        reopen_openrouter_pkce_authorization_url=lambda: False,
        settings=AppSettings(),
        config_path=Path("settings.json"),
    )
    app.view_settings = SimpleNamespace(
        refresh_after_openrouter_pkce_success=lambda *_args, **_kwargs: None,
        load_from_settings=lambda *_args, **_kwargs: None,
    )
    queued: list[object] = []
    monkeypatch.setattr(app, "_queue_settings_mutation_task", queued.append)

    app._on_request_openrouter_pkce(target_settings, launch_source="settings")
    app._on_request_openrouter_pkce(target_settings, launch_source="settings")

    assert len(queued) == 1
    await queued[0]()
    assert pkce_calls == ["settings"]

    app._on_request_openrouter_pkce(target_settings, launch_source="settings")

    assert len(queued) == 2


@pytest.mark.asyncio
async def test_on_request_openrouter_pkce_uses_draft_preserving_refresh_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    target_settings = AppSettings()
    updated_settings = AppSettings()
    pkce_calls: list[tuple[AppSettings, str]] = []
    refresh_calls: list[tuple[AppSettings, Path]] = []
    snackbar_calls: list[tuple[str, str]] = []

    async def fake_connect_openrouter_via_pkce(
        *, target_settings: AppSettings, launch_source: str
    ) -> bool:
        pkce_calls.append((target_settings, launch_source))
        return True

    app.controller = SimpleNamespace(
        connect_openrouter_via_pkce=fake_connect_openrouter_via_pkce,
        settings=updated_settings,
        config_path=Path("settings.json"),
    )
    app.view_settings = SimpleNamespace(
        refresh_after_openrouter_pkce_success=lambda settings, *, config_path: refresh_calls.append(
            (settings, config_path)
        ),
        load_from_settings=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full load_from_settings refresh should not run on PKCE success")
        ),
    )
    app._show_snackbar = lambda message, bgcolor: snackbar_calls.append((message, bgcolor))
    queued: list[object] = []
    monkeypatch.setattr(app, "_queue_settings_mutation_task", queued.append)

    app._on_request_openrouter_pkce(target_settings, launch_source="settings")

    assert len(queued) == 1
    await queued[0]()
    assert pkce_calls == [(target_settings, "settings")]
    assert refresh_calls == [(updated_settings, Path("settings.json"))]
    assert snackbar_calls == [(app_module.t("openrouter.pkce.connected"), app_module.COLOR_SUCCESS)]
    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale("ko")
        assert i18n_module.t("openrouter.pkce.connected") == "OpenRouter 인증이 완료되었어요."
    finally:
        i18n_module.set_locale(previous_locale)


@pytest.mark.asyncio
async def test_on_request_openrouter_pkce_does_not_refresh_settings_view_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    target_settings = AppSettings()
    refresh_calls: list[tuple[AppSettings, Path]] = []

    async def fake_connect_openrouter_via_pkce(
        *, target_settings: AppSettings, launch_source: str
    ) -> bool:
        _ = (target_settings, launch_source)
        return False

    app.controller = SimpleNamespace(
        connect_openrouter_via_pkce=fake_connect_openrouter_via_pkce,
        settings=AppSettings(),
        config_path=Path("settings.json"),
    )
    app.view_settings = SimpleNamespace(
        refresh_after_openrouter_pkce_success=lambda settings, *, config_path: refresh_calls.append(
            (settings, config_path)
        ),
        load_from_settings=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full load_from_settings refresh should not run on PKCE failure")
        ),
    )
    queued: list[object] = []
    monkeypatch.setattr(app, "_queue_settings_mutation_task", queued.append)

    app._on_request_openrouter_pkce(target_settings, launch_source="settings")

    assert len(queued) == 1
    await queued[0]()
    assert refresh_calls == []


@pytest.mark.asyncio
async def test_queue_orders_generic_settings_change_before_prompt_apply() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    raw_settings = object()
    pending_settings = object()
    merged_settings = object()
    events: list[tuple[str, object]] = []

    def fake_merge_settings(settings) -> object:
        events.append(("merge", settings))
        return merged_settings

    async def fake_apply_settings(settings) -> None:
        events.append(("apply", settings))

    app.controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        apply_settings=fake_apply_settings,
    )

    app._on_settings_changed(raw_settings)
    app._on_prompt_apply_settings(pending_settings)

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()

    assert events == [
        ("apply", raw_settings),
        ("merge", pending_settings),
        ("apply", merged_settings),
    ]


@pytest.mark.asyncio
async def test_queue_orders_generic_settings_change_before_provider_apply_on_settings_exit() -> (
    None
):
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._current_tab = 1
    raw_settings = object()
    provider_settings = object()
    app.view_dashboard = object()
    app.view_logs = SimpleNamespace(scroll_to_bottom=lambda: asyncio.sleep(0))
    app.view_about = object()
    app.view_settings = SimpleNamespace(
        has_provider_changes=True,
        consume_provider_apply_settings=lambda: provider_settings,
        refresh_prompt_if_empty=lambda: None,
    )
    app.content_area = DummyContent()
    events: list[tuple[str, object]] = []

    async def fake_apply_settings(settings) -> None:
        events.append(("settings", settings))

    async def fake_apply_providers(settings) -> None:
        events.append(("providers", settings))

    app.controller = SimpleNamespace(
        apply_settings=fake_apply_settings,
        apply_providers=fake_apply_providers,
    )

    app._on_settings_changed(raw_settings)
    app._on_nav_change(0)

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()

    assert events == [("settings", raw_settings), ("providers", provider_settings)]


def test_on_nav_change_closes_open_dialog_before_switching_tabs() -> None:
    events: list[tuple[str, object]] = []

    class RecordingContent:
        def __init__(self, initial) -> None:
            self._content = initial

        @property
        def content(self):
            return self._content

        @content.setter
        def content(self, value) -> None:
            events.append(("content", value))
            self._content = value

        def update(self) -> None:
            events.append(("update", self._content))

    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dialog = object()
    app.page.dialog = dialog

    def fake_close(control) -> None:
        events.append(("close", control))
        app.page.closed.append(control)
        if app.page.dialog is control:
            app.page.dialog = None

    app.page.close = fake_close
    app._current_tab = 0
    app.view_dashboard = object()
    app.view_settings = SimpleNamespace(
        has_provider_changes=False,
        refresh_prompt_if_empty=lambda: None,
    )
    app.view_logs = SimpleNamespace(scroll_to_bottom=lambda: asyncio.sleep(0))
    app.view_about = object()
    app.content_area = RecordingContent(app.view_dashboard)
    app.controller = SimpleNamespace(apply_providers=lambda _settings=None: asyncio.sleep(0))

    app._on_nav_change(1)

    assert events[:3] == [
        ("close", dialog),
        ("content", app.view_settings),
        ("update", app.view_settings),
    ]
    assert app.page.closed == [dialog]


def test_apply_locale_updates_views_and_page(monkeypatch: pytest.MonkeyPatch) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.title_bar = SimpleNamespace(set_title=lambda value: setattr(app, "_title", value))
    view_calls: list[str] = []
    app.view_dashboard = SimpleNamespace(apply_locale=lambda: view_calls.append("dash"))
    app.view_settings = SimpleNamespace(apply_locale=lambda: view_calls.append("settings"))
    app.view_logs = SimpleNamespace(apply_locale=lambda: view_calls.append("logs"))
    monkeypatch.setattr(app_module, "get_app_theme", lambda **_kwargs: "theme")
    monkeypatch.setattr(app_module, "font_for_language", lambda _code: "font")
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")

    app.apply_locale()

    assert app.page.title == app_module.t("app.title")
    assert view_calls == ["dash", "settings", "logs"]
    assert app.page.updated == 1


def test_refresh_overlay_peer_contract_ignores_missing_controller() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.view_dashboard = SimpleNamespace(
        set_overlay_peer_contract=lambda contract: (_ for _ in ()).throw(
            AssertionError(f"unexpected dashboard contract: {contract}")
        )
    )
    app.view_settings = SimpleNamespace(
        set_overlay_peer_contract=lambda contract: (_ for _ in ()).throw(
            AssertionError(f"unexpected settings contract: {contract}")
        )
    )

    app.refresh_overlay_peer_contract()

    assert getattr(app, "overlay_peer_contract", None) is None


def test_on_overlay_state_changed_updates_settings_view_runtime_state() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    contract = object()
    seen: list[tuple[str, str | None, str | None, bool | None]] = []
    refreshed: list[object] = []
    app.controller = SimpleNamespace(build_overlay_peer_consumer_contract=lambda: contract)
    app.view_dashboard = SimpleNamespace(
        set_overlay_peer_contract=lambda incoming: refreshed.append(("dashboard", incoming))
    )
    app.view_settings = SimpleNamespace(
        set_overlay_runtime_state=lambda state, failure_reason=None, **kwargs: seen.append(
            (
                state,
                failure_reason,
                kwargs.get("overlay_target"),
                kwargs.get("desktop_captions_locked"),
            )
        ),
        set_overlay_peer_contract=lambda incoming: refreshed.append(("settings", incoming)),
    )

    app.on_overlay_state_changed(state="failed", failure_reason="runtime_crashed")

    assert app.overlay_state == "failed"
    assert app.overlay_failure_reason == "runtime_crashed"
    assert seen == [("failed", "runtime_crashed", None, False)]
    assert refreshed == [("settings", contract), ("dashboard", contract)]


@pytest.mark.asyncio
async def test_submit_toggle_and_settings_wrappers_schedule_controller_tasks() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    seen: list[tuple[str, object]] = []

    async def fake_submit(text: str) -> None:
        seen.append(("submit", text))

    async def fake_translation(enabled: bool) -> None:
        seen.append(("translation", enabled))

    async def fake_stt(enabled: bool) -> None:
        seen.append(("stt", enabled))

    async def fake_overlay(enabled: bool) -> None:
        seen.append(("overlay", enabled))

    async def fake_peer(enabled: bool) -> None:
        seen.append(("peer", enabled))

    async def fake_apply_settings(settings) -> None:
        seen.append(("apply_settings", settings))

    async def fake_apply_providers() -> None:
        seen.append(("apply_providers", True))

    app.controller = SimpleNamespace(
        submit_text=fake_submit,
        set_translation_enabled=fake_translation,
        set_stt_enabled=fake_stt,
        set_overlay_enabled=fake_overlay,
        set_peer_translation_enabled=fake_peer,
        apply_settings=fake_apply_settings,
        apply_providers=fake_apply_providers,
    )

    app._on_manual_submit("You", "hello")
    app._on_translation_toggle(True)
    app._on_stt_toggle(False)
    app._on_overlay_toggle(True)
    app._on_peer_translation_toggle(True)
    app._on_settings_changed("settings")
    app._on_providers_changed()

    assert len(app.page.tasks) == 6
    for task_fn in app.page.tasks:
        await task_fn()

    assert seen == [
        ("submit", "hello"),
        ("translation", True),
        ("stt", False),
        ("overlay", True),
        ("peer", True),
        ("apply_settings", "settings"),
        ("apply_providers", True),
    ]


@pytest.mark.asyncio
async def test_providers_changed_managed_china_uses_normalized_key_gate_for_qq_prompt() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = SimpleNamespace(has_provider_changes=False)
    app.view_dashboard = SimpleNamespace()
    app.view_logs = SimpleNamespace(scroll_to_bottom=lambda: asyncio.sleep(0))
    app.content_area = DummyContent()
    prompts: list[bool] = []

    async def fake_apply_providers(_settings=None) -> None:
        return None

    app.controller = SimpleNamespace(
        apply_providers=fake_apply_providers,
        _is_managed_china_connection=lambda: True,
        _managed_openrouter_local_key_available=lambda: False,
        _managed_qq_key_available=lambda: pytest.fail(
            "settings apply should use normalized managed key resolution"
        ),
    )
    app.show_qq_managed_auth_dialog = lambda preview=False: prompts.append(preview)

    app._on_providers_changed()

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert prompts == [False]


@pytest.mark.asyncio
async def test_local_llm_secret_changed_forces_local_llm_rebuild() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    calls: list[bool] = []

    async def fake_apply_providers(
        _settings=None,
        *,
        force_rebuild_llm: bool = False,
    ) -> None:
        calls.append(force_rebuild_llm)

    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    app.controller = SimpleNamespace(settings=settings, apply_providers=fake_apply_providers)

    app._on_local_llm_secret_changed()

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert calls == [True]


@pytest.mark.asyncio
async def test_local_llm_secret_changed_ignores_non_local_llm_provider() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    calls: list[bool] = []

    async def fake_apply_providers(
        _settings=None,
        *,
        force_rebuild_llm: bool = False,
    ) -> None:
        calls.append(force_rebuild_llm)

    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.GEMINI))
    app.controller = SimpleNamespace(settings=settings, apply_providers=fake_apply_providers)

    app._on_local_llm_secret_changed()

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert calls == []


def test_toggle_handlers_route_basic_and_detailed_runtime_logs() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.overlay_state = "connected"
    app.overlay_failure_reason = "runtime_crashed"
    app.view_dashboard = SimpleNamespace(is_translation_on=False, is_stt_on=True)

    controller = RuntimeLoggingController()

    async def fake_translation(enabled: bool) -> None:
        _ = enabled

    async def fake_stt(enabled: bool) -> None:
        _ = enabled

    async def fake_overlay(enabled: bool) -> None:
        _ = enabled

    controller.set_translation_enabled = fake_translation
    controller.set_stt_enabled = fake_stt
    controller.set_overlay_enabled = fake_overlay
    app.controller = controller

    app._on_translation_toggle(True)
    app._on_stt_toggle(False)
    app._on_overlay_toggle(True)

    assert app.controller.basic_messages == [
        "[Dashboard] Translation toggle requested: enabled=True",
        "[Dashboard] STT toggle requested: enabled=False",
        "[Dashboard] Overlay toggle requested: enabled=True",
    ]
    assert app.controller.detailed_messages == [
        "[Dashboard] Translation toggle detail: dashboard_state=False overlay_state=connected",
        "[Dashboard] STT toggle detail: dashboard_state=True overlay_state=connected",
        "[Dashboard] Overlay toggle detail: overlay_state=connected failure_reason=runtime_crashed",
    ]


def test_on_overlay_state_changed_routes_runtime_logs() -> None:
    controller = RuntimeLoggingController()
    app = TranslatorApp.__new__(TranslatorApp)
    app.controller = controller
    seen: list[tuple[str, str | None, str | None, bool | None]] = []
    app.overlay_state = "off"
    app.view_settings = SimpleNamespace(
        set_overlay_runtime_state=lambda state, failure_reason=None, **kwargs: seen.append(
            (
                state,
                failure_reason,
                kwargs.get("overlay_target"),
                kwargs.get("desktop_captions_locked"),
            )
        )
    )

    app.on_overlay_state_changed(state="failed", failure_reason="runtime_crashed")

    assert controller.basic_messages == ["[Overlay] State changed: off -> failed"]
    assert controller.detailed_messages == [
        "[Overlay] State detail: overlay_state=failed failure_reason=runtime_crashed"
    ]
    assert seen == [("failed", "runtime_crashed", None, False)]


@pytest.mark.asyncio
async def test_on_language_change_updates_settings_and_shows_warning(monkeypatch) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = SimpleNamespace(
        languages=SimpleNamespace(source_language="ko", target_language="en"),
        provider=SimpleNamespace(stt=SimpleNamespace(value="deepgram")),
    )
    seen: list[tuple[str, str, str, str]] = []

    async def fake_on_dashboard_language_change(
        *, source_code: str, target_code: str, peer_source_code: str, peer_target_code: str
    ) -> None:
        seen.append((source_code, target_code, peer_source_code, peer_target_code))

    warning = SimpleNamespace(key="dashboard.warn_stt_key", language_code="ko")
    monkeypatch.setattr(
        app_module, "get_stt_compatibility_warning", lambda *_args, **_kwargs: warning
    )
    app.controller = SimpleNamespace(
        settings=settings,
        on_dashboard_language_change=fake_on_dashboard_language_change,
    )

    app._on_language_change("ja", "fr", "", "it")

    assert settings.languages.source_language == "ko"
    assert settings.languages.target_language == "en"
    assert len(app.page.opened) == 1
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert seen == [("ja", "fr", "", "it")]


@pytest.mark.asyncio
async def test_on_verify_api_key_persists_and_updates_dashboard_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.view_dashboard = SimpleNamespace(
        stt_calls=[],
        trans_calls=[],
        set_stt_needs_key=lambda value, update_ui=False: app.view_dashboard.stt_calls.append(
            (value, update_ui)
        ),
        set_translation_needs_key=lambda value, update_ui=False: app.view_dashboard.trans_calls.append(
            (value, update_ui)
        ),
    )

    async def fake_verify(provider: str, key: str):
        _ = key
        return provider in {"deepgram", "deepseek", "cerebras"}, "ok"

    settings = SimpleNamespace(
        api_key_verified=SimpleNamespace(
            deepgram=False,
            soniox=False,
            google=False,
            openrouter=False,
            deepseek=False,
            cerebras=False,
            alibaba_beijing=False,
            alibaba_singapore=False,
        )
    )
    app.controller = SimpleNamespace(
        verify_api_key=fake_verify,
        settings=settings,
        config_path="settings.json",
    )

    saves: list[tuple[object, object]] = []
    monkeypatch.setattr(app_module, "save_settings", lambda path, cfg: saves.append((path, cfg)))

    deepgram_result = await app._on_verify_api_key("deepgram", "k")
    google_result = await app._on_verify_api_key("google", "k")
    openrouter_result = await app._on_verify_api_key("openrouter", "k")
    deepseek_result = await app._on_verify_api_key("deepseek", "k")
    cerebras_result = await app._on_verify_api_key("cerebras", "k")

    assert deepgram_result == (True, "ok")
    assert google_result == (False, "ok")
    assert openrouter_result == (False, "ok")
    assert deepseek_result == (True, "ok")
    assert cerebras_result == (True, "ok")
    assert settings.api_key_verified.deepgram is True
    assert settings.api_key_verified.google is False
    assert settings.api_key_verified.openrouter is False
    assert settings.api_key_verified.deepseek is True
    assert settings.api_key_verified.cerebras is True
    assert app.view_dashboard.stt_calls[-1] == (False, False)
    assert app.view_dashboard.trans_calls[-1] == (False, False)
    assert len(saves) == 5


@pytest.mark.asyncio
async def test_on_verify_api_key_skips_persistence_for_stale_field_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.view_settings = SimpleNamespace(_google_key=SimpleNamespace(value="new-key"))
    app.view_dashboard = SimpleNamespace(
        stt_calls=[],
        trans_calls=[],
        set_stt_needs_key=lambda value, update_ui=False: app.view_dashboard.stt_calls.append(
            (value, update_ui)
        ),
        set_translation_needs_key=lambda value, update_ui=False: app.view_dashboard.trans_calls.append(
            (value, update_ui)
        ),
    )

    async def fake_verify(provider: str, key: str):
        assert (provider, key) == ("google", "old-key")
        return True, "ok"

    settings = SimpleNamespace(
        api_key_verified=SimpleNamespace(
            google=False,
        )
    )
    app.controller = SimpleNamespace(
        verify_api_key=fake_verify,
        settings=settings,
        config_path="settings.json",
    )

    saves: list[tuple[object, object]] = []
    monkeypatch.setattr(app_module, "save_settings", lambda path, cfg: saves.append((path, cfg)))

    result = await app._on_verify_api_key("google", "old-key")

    assert result == (True, "ok")
    assert settings.api_key_verified.google is False
    assert saves == []
    assert app.view_dashboard.stt_calls == []
    assert app.view_dashboard.trans_calls == []


@pytest.mark.asyncio
async def test_on_verify_api_key_skips_persistence_for_stale_cerebras_field_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.view_settings = SimpleNamespace(_cerebras_key=SimpleNamespace(value="new-key"))
    app.view_dashboard = SimpleNamespace(
        stt_calls=[],
        trans_calls=[],
        set_stt_needs_key=lambda value, update_ui=False: app.view_dashboard.stt_calls.append(
            (value, update_ui)
        ),
        set_translation_needs_key=lambda value, update_ui=False: app.view_dashboard.trans_calls.append(
            (value, update_ui)
        ),
    )

    async def fake_verify(provider: str, key: str):
        assert (provider, key) == ("cerebras", "old-key")
        return True, "ok"

    settings = SimpleNamespace(
        api_key_verified=SimpleNamespace(
            cerebras=False,
        )
    )
    app.controller = SimpleNamespace(
        verify_api_key=fake_verify,
        settings=settings,
        config_path="settings.json",
    )

    saves: list[tuple[object, object]] = []
    monkeypatch.setattr(app_module, "save_settings", lambda path, cfg: saves.append((path, cfg)))

    result = await app._on_verify_api_key("cerebras", "old-key")

    assert result == (True, "ok")
    assert settings.api_key_verified.cerebras is False
    assert saves == []
    assert app.view_dashboard.stt_calls == []
    assert app.view_dashboard.trans_calls == []


def test_show_snackbar_opens_page_snackbar() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()

    app._show_snackbar("hello", "green", duration=1234)

    assert len(app.page.opened) == 1
    snackbar = app.page.opened[0]
    assert snackbar.duration == 1234


def test_show_founder_letter_dialog_opens_with_locale_readme_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
    settings.openrouter.llm_model = OpenRouterLLMModel.QWEN_35_FLASH_02_23
    app.controller = SimpleNamespace(settings=settings)

    captured: dict[str, object] = {}
    pkce_calls: list[tuple[AppSettings, str]] = []
    opened_urls: list[str] = []
    previous_locale = i18n_module.get_locale()

    class FakeFounderLetterDialog:
        def __init__(self, page, *, on_readme=None, on_connect=None, on_contact=None):
            captured["page"] = page
            captured["on_readme"] = on_readme
            captured["on_connect"] = on_connect
            captured["on_contact"] = on_contact

        def open(self) -> None:
            captured["opened"] = True

    monkeypatch.setattr(app_module, "FounderLetterDialog", FakeFounderLetterDialog)
    monkeypatch.setattr(app_module.webbrowser, "open", lambda url: opened_urls.append(url))
    monkeypatch.setattr(
        app,
        "_on_request_openrouter_pkce",
        lambda target_settings, *, launch_source="settings": pkce_calls.append(
            (target_settings, launch_source)
        ),
    )

    try:
        i18n_module.set_locale("ko")

        app.show_founder_letter_dialog()

        assert captured["page"] is app.page
        assert captured["opened"] is True
        assert callable(captured["on_readme"])
        assert captured["on_connect"] is None
        assert captured["on_contact"] is None
        captured["on_readme"]()
    finally:
        i18n_module.set_locale(previous_locale)

    assert pkce_calls == []
    assert opened_urls == [
        "https://github.com/kapitalismho/PuriPuly-heart/blob/main/README.ko.md#자신의-api-키-사용하기"
    ]


def test_show_founder_letter_dialog_does_not_prepare_byok_alias_when_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.openrouter.selection_alias = None
    settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    app.controller = SimpleNamespace(settings=settings)

    captured: dict[str, object] = {}
    pkce_calls: list[tuple[AppSettings, str]] = []

    class FakeFounderLetterDialog:
        def __init__(self, _page, *, on_readme=None, on_connect=None, on_contact=None):
            captured["page"] = _page
            captured["on_readme"] = on_readme
            captured["on_connect"] = on_connect
            captured["on_contact"] = on_contact

        def open(self) -> None:
            captured["opened"] = True

    monkeypatch.setattr(app_module, "FounderLetterDialog", FakeFounderLetterDialog)
    monkeypatch.setattr(
        app,
        "_on_request_openrouter_pkce",
        lambda target_settings, *, launch_source="settings": pkce_calls.append(
            (target_settings, launch_source)
        ),
    )

    app.show_founder_letter_dialog()

    assert captured["opened"] is True
    assert captured["page"] is app.page
    assert callable(captured["on_readme"])
    assert captured["on_connect"] is None
    assert captured["on_contact"] is None
    assert pkce_calls == []


@pytest.mark.asyncio
async def test_check_and_notify_update_handles_none_and_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = DummyPage()

    async def no_update():
        return None

    monkeypatch.setattr(app_module, "check_for_update", no_update)
    await _check_and_notify_update(page)
    assert page.opened == []

    update_info = SimpleNamespace(version="9.9.9", download_url="https://example.com")

    async def has_update():
        return update_info

    monkeypatch.setattr(app_module, "check_for_update", has_update)
    opened_urls: list[str] = []
    monkeypatch.setattr(app_module.webbrowser, "open", lambda url: opened_urls.append(url))
    monkeypatch.setattr(
        app_module.ft, "Icon", lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs)
    )
    monkeypatch.setattr(
        app_module.ft,
        "TextButton",
        lambda *args, **kwargs: SimpleNamespace(on_click=kwargs.get("on_click")),
    )
    await _check_and_notify_update(page)

    assert len(page.opened) == 1
    snackbar = page.opened[0]
    download_btn = snackbar.content.controls[2]
    download_btn.on_click(None)
    assert opened_urls == ["https://example.com"]
    assert page.updated == 1


@pytest.mark.asyncio
async def test_check_and_notify_update_swallows_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    page = DummyPage()
    app = TranslatorApp.__new__(TranslatorApp)
    app.controller = RuntimeLoggingController()

    async def raise_error():
        raise RuntimeError("network down")

    monkeypatch.setattr(app_module, "check_for_update", raise_error)
    await _check_and_notify_update(page, log_detailed=app._log_detailed)
    assert page.opened == []
    assert app.controller.basic_messages == []
    assert app.controller.detailed_messages == ["[Update] Check notification failed: network down"]
