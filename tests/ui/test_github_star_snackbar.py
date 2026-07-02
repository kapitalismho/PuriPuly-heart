from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")
import flet as ft

from puripuly_heart.config.settings import (
    AppSettings,
    TranslationConnection,
    TranslationModel,
    materialize_translation_settings,
    to_dict,
)
from puripuly_heart.providers.llm.openrouter import OpenRouterKeyMetadata
from puripuly_heart.ui import app as app_module
from puripuly_heart.ui import controller as controller_module
from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.app import TranslatorApp, _check_and_notify_update
from puripuly_heart.ui.components.debug_preview_panel import DebugPreviewPanel
from puripuly_heart.ui.controller import GuiController


class DummyPage:
    def __init__(self) -> None:
        self.opened: list[object] = []
        self.closed: list[object] = []
        self.tasks: list[object] = []
        self.updated = 0

    def open(self, control) -> None:  # noqa: ANN001
        if hasattr(control, "open"):
            control.open = True
        self.opened.append(control)

    def close(self, control) -> None:  # noqa: ANN001
        if hasattr(control, "open"):
            control.open = False
        self.closed.append(control)

    def run_task(self, task_factory) -> None:  # noqa: ANN001
        self.tasks.append(task_factory)

    def update(self) -> None:
        self.updated += 1


class Flet028SnackBarPage(DummyPage):
    """Model Flet 0.28.3's state-only SnackBar close behavior.

    Flet 0.28.3 sets the Python-side ``open`` flag to false on ``page.close()``,
    but the visible Flutter SnackBar is not removed until another SnackBar is
    opened or the duration expires.
    """

    def __init__(self) -> None:
        super().__init__()
        self.visible_snackbar: ft.SnackBar | None = None

    def open(self, control) -> None:  # noqa: ANN001
        super().open(control)
        if isinstance(control, ft.SnackBar):
            self.visible_snackbar = control

    def close(self, control) -> None:  # noqa: ANN001
        if hasattr(control, "open"):
            control.open = False
        self.closed.append(control)


def _settings_for_connection(connection: TranslationConnection) -> AppSettings:
    settings = AppSettings()
    settings.translation.connection = connection
    if connection == TranslationConnection.MANAGED_CHINA:
        settings.translation.model = TranslationModel.DEEPSEEK_V4_FLASH
    elif connection == TranslationConnection.OFFICIAL_BYOK:
        settings.translation.model = TranslationModel.DEEPSEEK_V4_FLASH
    elif connection == TranslationConnection.OLLAMA:
        settings.translation.model = TranslationModel.LOCAL_LLM
    settings.translation.connection_history[settings.translation.model.value] = connection
    materialize_translation_settings(settings)
    return settings


def _controller_for(settings: AppSettings) -> GuiController:
    controller = GuiController(
        page=SimpleNamespace(),
        app=SimpleNamespace(),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    return controller


def _eligible_managed_controller() -> GuiController:
    controller = _controller_for(_settings_for_connection(TranslationConnection.MANAGED))
    controller._managed_trial_usage_metadata = OpenRouterKeyMetadata(  # noqa: SLF001
        limit_usd=100.0,
        remaining_usd=50.0,
        usage_usd=50.0,
    )
    return controller


def _utc_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _eligible_app(
    page: DummyPage | None = None,
    settings: AppSettings | None = None,
) -> tuple[TranslatorApp, DummyPage, GuiController]:
    page = page or DummyPage()
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = page
    controller = GuiController(
        page=page,
        app=app,
        config_path=Path("settings.json"),
    )
    controller.settings = settings or _settings_for_connection(TranslationConnection.MANAGED)
    controller._managed_trial_usage_metadata = OpenRouterKeyMetadata(  # noqa: SLF001
        limit_usd=100.0,
        remaining_usd=50.0,
        usage_usd=50.0,
    )
    app.controller = controller
    return app, page, controller


async def _async_noop(*_args: object, **_kwargs: object) -> None:
    return None


def test_github_star_prompt_state_blocks_clicked_and_recent_shows() -> None:
    controller = _eligible_managed_controller()
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)

    assert controller.should_show_github_star_prompt(now=now) is False

    controller.settings.ui.github_star_prompt_eligible_launch_count = 3
    assert controller.should_show_github_star_prompt(now=now) is True

    controller.settings.ui.github_star_prompt_clicked = True
    assert controller.should_show_github_star_prompt(now=now) is False

    controller.settings.ui.github_star_prompt_clicked = False
    controller.settings.ui.github_star_prompt_last_shown_at = _utc_z(now - timedelta(days=13))
    assert controller.should_show_github_star_prompt(now=now) is False

    controller.settings.ui.github_star_prompt_last_shown_at = _utc_z(now - timedelta(days=14))
    assert controller.should_show_github_star_prompt(now=now) is True


@pytest.mark.asyncio
async def test_launch_github_star_snackbar_counts_eligible_launches_before_first_show(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings_for_connection(TranslationConnection.MANAGED)
    saved_payloads: list[dict[str, object]] = []
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )

    first_app, first_page, first_controller = _eligible_app(settings=settings)
    first_shown = await first_app.maybe_show_github_star_prompt_after_launch()

    assert first_shown is False
    assert first_page.opened == []
    assert sleeps == []
    assert first_controller.settings.ui.github_star_prompt_eligible_launch_count == 1
    assert saved_payloads[-1]["ui"]["github_star_prompt_eligible_launch_count"] == 1

    second_app, second_page, second_controller = _eligible_app(settings=settings)
    second_shown = await second_app.maybe_show_github_star_prompt_after_launch()

    assert second_shown is False
    assert second_page.opened == []
    assert sleeps == []
    assert second_controller.settings.ui.github_star_prompt_eligible_launch_count == 2
    assert saved_payloads[-1]["ui"]["github_star_prompt_eligible_launch_count"] == 2

    third_app, third_page, third_controller = _eligible_app(settings=settings)
    third_shown = await third_app.maybe_show_github_star_prompt_after_launch()

    assert third_shown is True
    assert len(sleeps) == 1
    assert len(third_page.opened) == 1
    assert third_controller.settings.ui.github_star_prompt_eligible_launch_count == 3
    assert third_controller.settings.ui.github_star_prompt_show_count == 1
    assert saved_payloads[-2]["ui"]["github_star_prompt_eligible_launch_count"] == 3
    assert saved_payloads[-2]["ui"]["github_star_prompt_show_count"] == 0
    assert saved_payloads[-1]["ui"]["github_star_prompt_show_count"] == 1


@pytest.mark.asyncio
async def test_launch_github_star_snackbar_does_not_count_ineligible_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, page, controller = _eligible_app()
    controller._managed_trial_usage_metadata = None  # noqa: SLF001
    saved_payloads: list[dict[str, object]] = []

    async def fail_sleep(_seconds: float) -> None:
        pytest.fail("ineligible launch should skip before sleeping")

    monkeypatch.setattr(app_module.asyncio, "sleep", fail_sleep)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )

    shown = await app.maybe_show_github_star_prompt_after_launch()

    assert shown is False
    assert page.opened == []
    assert controller.settings.ui.github_star_prompt_eligible_launch_count == 0
    assert saved_payloads == []


def test_record_github_star_prompt_opened_persists_timestamp_count_and_not_clicked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _eligible_managed_controller()
    controller.settings.ui.github_star_prompt_show_count = 2
    saved_payloads: list[dict[str, object]] = []
    opened_at = datetime(2026, 5, 24, 12, 34, 56, tzinfo=timezone.utc)

    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )

    assert controller.record_github_star_prompt_opened(opened_at=opened_at) is True

    assert controller.settings.ui.github_star_prompt_last_shown_at == "2026-05-24T12:34:56Z"
    assert controller.settings.ui.github_star_prompt_show_count == 3
    assert controller.settings.ui.github_star_prompt_clicked is False
    assert saved_payloads[-1]["ui"]["github_star_prompt_last_shown_at"] == ("2026-05-24T12:34:56Z")
    assert saved_payloads[-1]["ui"]["github_star_prompt_show_count"] == 3
    assert saved_payloads[-1]["ui"]["github_star_prompt_clicked"] is False


def test_record_github_star_prompt_clicked_persists_permanent_suppression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _eligible_managed_controller()
    saved_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )

    assert controller.record_github_star_prompt_clicked() is True

    assert controller.settings.ui.github_star_prompt_clicked is True
    assert saved_payloads[-1]["ui"]["github_star_prompt_clicked"] is True
    assert (
        controller.should_show_github_star_prompt(now=datetime(2026, 5, 24, tzinfo=timezone.utc))
        is False
    )


@pytest.mark.asyncio
async def test_prompt_open_persistence_uses_async_save_before_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, page, controller = _eligible_app()
    controller.settings.ui.github_star_prompt_eligible_launch_count = 3
    saved_payloads: list[dict[str, object]] = []
    to_thread_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def fake_to_thread(func, /, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        nonlocal to_thread_calls
        to_thread_calls += 1
        return func(*args, **kwargs)

    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(controller_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )

    shown = await app.maybe_show_github_star_prompt_after_launch()

    assert shown is True
    assert to_thread_calls >= 1
    assert len(page.opened) == 1
    assert saved_payloads[0]["ui"]["github_star_prompt_show_count"] == 1


@pytest.mark.asyncio
async def test_prompt_open_refuses_display_and_restores_state_when_persistence_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, page, controller = _eligible_app()
    controller.settings.ui.github_star_prompt_eligible_launch_count = 3

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)

    def fail_save_settings(*_args: object, **_kwargs: object) -> None:
        raise OSError("settings write failed")

    monkeypatch.setattr(controller_module, "save_settings", fail_save_settings)

    shown = await app.maybe_show_github_star_prompt_after_launch()

    assert shown is False
    assert page.opened == []
    assert controller.settings.ui.github_star_prompt_last_shown_at is None
    assert controller.settings.ui.github_star_prompt_show_count == 0
    assert controller.settings.ui.github_star_prompt_clicked is False


@pytest.mark.asyncio
async def test_apply_settings_preserves_current_github_star_prompt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _settings_for_connection(TranslationConnection.MANAGED)
    current.ui.github_star_prompt_clicked = True
    current.ui.github_star_prompt_last_shown_at = "2026-05-24T12:34:56Z"
    current.ui.github_star_prompt_show_count = 4
    current.ui.github_star_prompt_translation_success_observed = True
    current.ui.github_star_prompt_eligible_launch_count = 2
    controller = _controller_for(current)
    replacement = _settings_for_connection(TranslationConnection.OPENROUTER)
    replacement.ui.github_star_prompt_eligible_launch_count = 1
    saved_payloads: list[dict[str, object]] = []

    async def noop(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", noop)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )

    controller.page = SimpleNamespace()
    controller.app = SimpleNamespace(view_settings=None)
    controller.hub = None

    await controller.apply_settings(replacement)

    assert replacement.ui.github_star_prompt_clicked is True
    assert replacement.ui.github_star_prompt_last_shown_at == "2026-05-24T12:34:56Z"
    assert replacement.ui.github_star_prompt_show_count == 4
    assert replacement.ui.github_star_prompt_translation_success_observed is True
    assert replacement.ui.github_star_prompt_eligible_launch_count == 2
    assert saved_payloads[-1]["ui"]["github_star_prompt_clicked"] is True
    assert saved_payloads[-1]["ui"]["github_star_prompt_eligible_launch_count"] == 2


@pytest.mark.asyncio
async def test_prompt_open_persistence_does_not_stale_overwrite_replaced_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _settings_for_connection(TranslationConnection.MANAGED)
    initial.languages.target_language = "en"
    controller = _controller_for(initial)
    controller.app = SimpleNamespace(view_settings=None)
    controller.hub = None
    replacement = _settings_for_connection(TranslationConnection.MANAGED)
    replacement.languages.target_language = "ja"
    saved_payloads: list[dict[str, object]] = []
    first_to_thread_started = asyncio.Event()
    release_first_to_thread = asyncio.Event()
    to_thread_calls = 0
    opened_at = datetime(2026, 5, 24, 12, 34, 56, tzinfo=timezone.utc)

    async def delayed_first_to_thread(func, /, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        nonlocal to_thread_calls
        to_thread_calls += 1
        if to_thread_calls == 1:
            first_to_thread_started.set()
            await release_first_to_thread.wait()
        return func(*args, **kwargs)

    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", _async_noop)
    monkeypatch.setattr(controller_module.asyncio, "to_thread", delayed_first_to_thread)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )

    open_task = asyncio.create_task(
        controller.persist_github_star_prompt_opened(opened_at=opened_at)
    )
    await asyncio.wait_for(first_to_thread_started.wait(), timeout=1.0)

    apply_task = asyncio.create_task(controller.apply_settings(replacement))
    await asyncio.sleep(0)

    assert not apply_task.done()

    release_first_to_thread.set()
    assert await asyncio.wait_for(open_task, timeout=1.0) is True
    await asyncio.wait_for(apply_task, timeout=1.0)

    assert controller.settings is replacement
    assert saved_payloads[-1]["languages"]["target_language"] == "ja"
    assert saved_payloads[-1]["ui"]["github_star_prompt_last_shown_at"] == ("2026-05-24T12:34:56Z")
    assert saved_payloads[-1]["ui"]["github_star_prompt_show_count"] == 1


@pytest.mark.asyncio
async def test_prompt_click_persistence_does_not_stale_overwrite_replaced_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _settings_for_connection(TranslationConnection.MANAGED)
    initial.languages.target_language = "en"
    controller = _controller_for(initial)
    controller.app = SimpleNamespace(view_settings=None)
    controller.hub = None
    replacement = _settings_for_connection(TranslationConnection.MANAGED)
    replacement.languages.target_language = "ja"
    saved_payloads: list[dict[str, object]] = []
    first_to_thread_started = asyncio.Event()
    release_first_to_thread = asyncio.Event()
    to_thread_calls = 0

    async def delayed_first_to_thread(func, /, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        nonlocal to_thread_calls
        to_thread_calls += 1
        if to_thread_calls == 1:
            first_to_thread_started.set()
            await release_first_to_thread.wait()
        return func(*args, **kwargs)

    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", _async_noop)
    monkeypatch.setattr(controller_module.asyncio, "to_thread", delayed_first_to_thread)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )

    click_task = asyncio.create_task(controller.persist_github_star_prompt_clicked())
    await asyncio.wait_for(first_to_thread_started.wait(), timeout=1.0)

    apply_task = asyncio.create_task(controller.apply_settings(replacement))
    await asyncio.sleep(0)

    assert not apply_task.done()

    release_first_to_thread.set()
    assert await asyncio.wait_for(click_task, timeout=1.0) is True
    await asyncio.wait_for(apply_task, timeout=1.0)

    assert controller.settings is replacement
    assert saved_payloads[-1]["languages"]["target_language"] == "ja"
    assert saved_payloads[-1]["ui"]["github_star_prompt_clicked"] is True


@pytest.mark.asyncio
async def test_launch_github_star_snackbar_waits_opens_records_and_action_click_suppresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, page, controller = _eligible_app()
    controller.settings.ui.github_star_prompt_eligible_launch_count = 3
    saved_payloads: list[dict[str, object]] = []
    sleeps: list[float] = []
    opened_urls: list[str] = []
    previous_locale = i18n_module.get_locale()

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )
    monkeypatch.setattr(app_module.webbrowser, "open", lambda url: opened_urls.append(url) or True)

    try:
        i18n_module.set_locale("ko")
        shown = await app.maybe_show_github_star_prompt_after_launch()
    finally:
        i18n_module.set_locale(previous_locale)

    assert shown is True
    assert len(sleeps) == 1
    assert 2.0 <= sleeps[0] <= 3.0
    assert len(page.opened) == 1
    assert controller.settings.ui.github_star_prompt_last_shown_at is not None
    assert controller.settings.ui.github_star_prompt_show_count == 1
    assert saved_payloads[0]["ui"]["github_star_prompt_show_count"] == 1

    snackbar = page.opened[0]
    assert snackbar.bgcolor == app_module.COLOR_SUCCESS
    assert snackbar.duration == 8000
    assert snackbar.behavior == ft.SnackBarBehavior.FLOATING
    assert getattr(snackbar, "show_close_icon", False) is not True
    assert isinstance(snackbar.content, ft.Row)
    message = snackbar.content.controls[0]
    action = snackbar.content.controls[1]
    assert message.value == "PuriPuly가 도움이 됐다면 GitHub에서 Star를 눌러주세요! 큰 힘이 되어요!"
    assert action.text == "이동"

    in_click_callback = True
    click_callback_save_calls: list[dict[str, object]] = []

    def capture_click_save(_path: Path, updated: AppSettings) -> None:
        if in_click_callback:
            click_callback_save_calls.append(to_dict(updated))
            return
        saved_payloads.append(to_dict(updated))

    monkeypatch.setattr(controller_module, "save_settings", capture_click_save)

    action.on_click(None)
    in_click_callback = False

    assert opened_urls == ["https://github.com/kapitalismho/PuriPuly-heart"]
    assert page.closed == [snackbar]
    assert snackbar.open is False
    assert click_callback_save_calls == []
    assert controller.settings.ui.github_star_prompt_clicked is False
    assert len(page.tasks) == 1
    await page.tasks.pop(0)()

    assert controller.settings.ui.github_star_prompt_clicked is True
    assert saved_payloads[-1]["ui"]["github_star_prompt_clicked"] is True


@pytest.mark.asyncio
async def test_github_star_action_displaces_visible_snackbar_when_page_close_is_state_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flet028_page = Flet028SnackBarPage()
    app, page, _controller = _eligible_app(flet028_page)
    opened_urls: list[str] = []

    monkeypatch.setattr(controller_module, "save_settings", lambda _path, _updated: None)
    monkeypatch.setattr(app_module.webbrowser, "open", lambda url: opened_urls.append(url) or True)

    shown = await app._open_github_star_prompt_snackbar()  # noqa: SLF001

    assert shown is True
    snackbar = page.opened[0]
    assert flet028_page.visible_snackbar is snackbar

    action = snackbar.content.controls[1]
    action.on_click(None)

    assert opened_urls == ["https://github.com/kapitalismho/PuriPuly-heart"]
    assert flet028_page.visible_snackbar is not snackbar


@pytest.mark.asyncio
async def test_launch_github_star_snackbar_skips_if_higher_priority_feedback_was_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, page, controller = _eligible_app()
    saved_payloads: list[dict[str, object]] = []
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )
    app._mark_launch_high_priority_feedback_shown("update")

    shown = await app.maybe_show_github_star_prompt_after_launch()

    assert shown is False
    assert sleeps == []
    assert page.opened == []
    assert controller.settings.ui.github_star_prompt_eligible_launch_count == 1
    assert saved_payloads[-1]["ui"]["github_star_prompt_eligible_launch_count"] == 1


@pytest.mark.asyncio
async def test_launch_github_star_snackbar_skips_if_feedback_appears_during_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, page, controller = _eligible_app()
    controller.settings.ui.github_star_prompt_eligible_launch_count = 2
    saved_payloads: list[dict[str, object]] = []

    async def fake_sleep(_seconds: float) -> None:
        app._mark_launch_high_priority_feedback_shown("connection_failure")

    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )

    shown = await app.maybe_show_github_star_prompt_after_launch()

    assert shown is False
    assert page.opened == []
    assert controller.settings.ui.github_star_prompt_eligible_launch_count == 3
    assert controller.settings.ui.github_star_prompt_show_count == 0
    assert saved_payloads[-1]["ui"]["github_star_prompt_eligible_launch_count"] == 3
    assert saved_payloads[-1]["ui"]["github_star_prompt_show_count"] == 0


@pytest.mark.asyncio
async def test_launch_github_star_snackbar_skips_and_restores_if_feedback_appears_during_open_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, page, controller = _eligible_app()
    controller.settings.ui.github_star_prompt_eligible_launch_count = 3
    saved_payloads: list[dict[str, object]] = []
    to_thread_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def feedback_during_first_to_thread(
        func, /, *args, **kwargs
    ):  # noqa: ANN001, ANN002, ANN003
        nonlocal to_thread_calls
        to_thread_calls += 1
        if to_thread_calls == 1:
            app._mark_launch_high_priority_feedback_shown("connection_failure")
        return func(*args, **kwargs)

    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(controller_module.asyncio, "to_thread", feedback_during_first_to_thread)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )

    shown = await app.maybe_show_github_star_prompt_after_launch()

    assert shown is False
    assert page.opened == []
    assert controller.settings.ui.github_star_prompt_last_shown_at is None
    assert controller.settings.ui.github_star_prompt_show_count == 0
    assert saved_payloads[0]["ui"]["github_star_prompt_show_count"] == 1
    assert saved_payloads[-1]["ui"]["github_star_prompt_show_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "error",
        "auth_required",
        "update",
        "usage_exhaustion",
        "managed_release_warning",
        "connection_failure",
    ],
)
async def test_named_higher_priority_launch_feedback_categories_suppress_prompt(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    app, page, controller = _eligible_app()
    saved_payloads: list[dict[str, object]] = []

    async def fail_sleep(_seconds: float) -> None:
        pytest.fail("conflicted launch prompts must skip before sleeping")

    monkeypatch.setattr(app_module.asyncio, "sleep", fail_sleep)
    monkeypatch.setattr(
        controller_module,
        "save_settings",
        lambda _path, updated: saved_payloads.append(to_dict(updated)),
    )
    app._mark_launch_high_priority_feedback_shown(reason)

    assert await app.maybe_show_github_star_prompt_after_launch() is False
    assert page.opened == []
    assert controller.settings.ui.github_star_prompt_eligible_launch_count == 1
    assert saved_payloads[-1]["ui"]["github_star_prompt_eligible_launch_count"] == 1


@pytest.mark.asyncio
async def test_main_gui_runs_github_star_prompt_after_update_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    page = DummyPage()

    class FakeController:
        async def start(self) -> None:
            events.append("start")

    class FakeApp:
        def __init__(self, incoming_page, *, config_path, debug_ui_preview=False):  # noqa: ANN001
            _ = (incoming_page, config_path, debug_ui_preview)
            self.controller = FakeController()

        def _log_detailed(self, message: str, *, level: int = app_module.logging.INFO) -> None:
            _ = (message, level)

        def _mark_launch_high_priority_feedback_shown(self, reason: str, snackbar=None) -> None:
            _ = (reason, snackbar)

        async def maybe_show_github_star_prompt_after_launch(self) -> bool:
            events.append("github-star")
            return True

    async def fake_check_and_notify_update(incoming_page, **kwargs) -> None:  # noqa: ANN001
        _ = (incoming_page, kwargs)
        events.append("update")

    monkeypatch.setattr(app_module, "TranslatorApp", FakeApp)
    monkeypatch.setattr(app_module, "_check_and_notify_update", fake_check_and_notify_update)

    await app_module.main_gui(page, config_path=Path("settings.json"))

    assert events == ["start", "update", "github-star"]


@pytest.mark.asyncio
async def test_update_notification_marks_launch_high_priority_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = DummyPage()
    marked: list[object] = []
    update_info = SimpleNamespace(version="9.9.9", download_url="https://example.com")

    async def has_update():
        return update_info

    monkeypatch.setattr(app_module, "check_for_update", has_update)
    monkeypatch.setattr(
        app_module.ft, "Icon", lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs)
    )
    monkeypatch.setattr(
        app_module.ft,
        "TextButton",
        lambda *args, **kwargs: SimpleNamespace(on_click=kwargs.get("on_click")),
    )

    await _check_and_notify_update(
        page,
        on_launch_snackbar_shown=lambda snackbar: marked.append(snackbar),
    )

    assert marked == [page.opened[0]]


def test_stt_compatibility_snackbar_marks_launch_high_priority_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._github_star_prompt_launch_pending = True
    app._launch_high_priority_feedback_shown = False
    app._launch_high_priority_feedback_reason = None
    app._launch_high_priority_snackbar = None
    settings = SimpleNamespace(
        languages=SimpleNamespace(source_language="ko", target_language="en"),
        provider=SimpleNamespace(stt=SimpleNamespace(value="deepgram")),
    )

    async def fake_on_dashboard_language_change(
        *,
        source_code: str,
        target_code: str,
        peer_source_code: str,
        peer_target_code: str,
    ) -> None:
        _ = (source_code, target_code, peer_source_code, peer_target_code)

    warning = SimpleNamespace(key="dashboard.warn_stt_key", language_code="ko")
    monkeypatch.setattr(
        app_module, "get_stt_compatibility_warning", lambda *_args, **_kwargs: warning
    )
    app.controller = SimpleNamespace(
        settings=settings,
        on_dashboard_language_change=fake_on_dashboard_language_change,
    )

    app._on_language_change("ja", "fr", "", "it")

    assert len(app.page.opened) == 1
    assert app._launch_feedback_conflicts_with_github_star_prompt() is True


def test_debug_preview_panel_includes_github_star_snackbar_action() -> None:
    invoked: list[str] = []

    def noop() -> None:
        return None

    panel = DebugPreviewPanel(
        on_brake_notice=noop,
        on_revoked_notice=noop,
        on_founder_letter=noop,
        on_pkce_failure=noop,
        on_discord_auth=noop,
        on_qq_auth=noop,
        on_discord_callback_page=noop,
        on_peer_translation_eula=noop,
        on_local_qwen_hallucination_modal=noop,
        on_talk_together_pass_invite_progress=noop,
        on_capture_fault_cycle=noop,
        on_stt_fault_cycle=noop,
        on_audio_fault_clear=noop,
        on_github_star_snackbar=lambda: invoked.append("github-star"),
    )

    action = panel._action_buttons["github_star_snackbar"]  # noqa: SLF001
    assert action.text == "GitHub Star"

    action.on_click(None)

    assert invoked == ["github-star"]


def test_debug_preview_github_star_snackbar_opens_without_mutating_prompt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    controller = _eligible_managed_controller()
    initial_prompt_state = to_dict(controller.settings)["ui"]
    app.controller = controller
    opened_urls: list[str] = []
    previous_locale = i18n_module.get_locale()

    monkeypatch.setattr(app_module.webbrowser, "open", lambda url: opened_urls.append(url) or True)

    try:
        i18n_module.set_locale("ko")
        app._preview_github_star_snackbar()  # noqa: SLF001
    finally:
        i18n_module.set_locale(previous_locale)

    assert len(app.page.opened) == 1
    assert to_dict(controller.settings)["ui"] == initial_prompt_state

    snackbar = app.page.opened[0]
    assert snackbar.bgcolor == app_module.COLOR_SUCCESS
    assert snackbar.duration == 8000
    assert snackbar.behavior == ft.SnackBarBehavior.FLOATING
    assert getattr(snackbar, "show_close_icon", False) is not True
    assert isinstance(snackbar.content, ft.Row)
    message = snackbar.content.controls[0]
    action = snackbar.content.controls[1]
    assert message.value == "PuriPuly가 도움이 됐다면 GitHub에서 Star를 눌러주세요! 큰 힘이 되어요!"
    assert action.text == "이동"

    action.on_click(None)

    assert opened_urls == ["https://github.com/kapitalismho/PuriPuly-heart"]
    assert app.page.closed == [snackbar]
    assert snackbar.open is False
    assert to_dict(controller.settings)["ui"] == initial_prompt_state
