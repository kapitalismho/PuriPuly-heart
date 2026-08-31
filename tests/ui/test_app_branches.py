from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("flet")
from dataclasses import replace

import flet as ft
from puripuly_heart.core.managed_openrouter_release import TalkTogetherPassStatus

import puripuly_heart.ui.app as app_module
from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.settings_view import (
    OpenRouterPkceTarget,
    OverlaySettingsSnapshot,
    PromptApplyIntent,
)
from puripuly_heart.app.ports.ui_models import OverlayPeerPresentationState
from puripuly_heart.app.services.application_shutdown import (
    application_shutdown_callback,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    SettingsOwner,
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.services.http_extension_registry import HttpExtensionRegistryService
from puripuly_heart.app.services.telemetry_reporting import APP_ACTIVE_DAY_RETRY_DELAY_S
from puripuly_heart.app.services.ui_application import UiApplicationBoundary
from puripuly_heart.composition.ui_application import compose_ui_application
from puripuly_heart.config.provider_values import (
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.lifecycle import SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS
from puripuly_heart.core.openrouter_routing import OpenRouterProviderRouting
from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.app import TranslatorApp, _check_and_notify_update
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from tests.helpers.ui_application import compose_test_ui_application_boundary

MISSING = object()


def _vnext(
    *,
    llm: str | None = None,
    model: str | None = None,
    connection: str | None = None,
    openrouter_source: str | None = None,
    openrouter_alias: str | None = None,
    openrouter_model: str | None = None,
    openrouter_routing: str | None = None,
    connection_history: dict[str, str] | None = None,
    telemetry_last_sent: str | None = None,
    overlay_pos_x: int | float | None = None,
    overlay_pos_y: int | float | None = None,
    target_language: str | None = None,
) -> AppSettingsVNext:
    current = AppSettingsVNext()
    translation = current.intent.translation
    if llm == "gemini":
        model = model or "gemini37_flash"
        connection = connection or "official_byok"
    elif llm == "local_llm":
        model = model or "local_llm"
        connection = connection or "ollama"
    elif llm == "openrouter":
        if openrouter_source == "managed":
            model = model or "gemma4"
            connection = connection or "managed"
        else:
            model = model or "gemma4"
            connection = connection or "openrouter"
    if model is not None or connection is not None:
        translation = replace(
            translation,
            model=model or translation.model,
            connection=connection or translation.connection,
        )
    if openrouter_source is not None:
        translation = replace(translation, openrouter_selected_source=openrouter_source)
    if openrouter_alias is not None:
        translation = replace(translation, openrouter_selection_alias=openrouter_alias)
    if openrouter_model is not None:
        translation = replace(translation, openrouter_model=openrouter_model)
    if openrouter_routing is not None:
        translation = replace(translation, openrouter_provider_routing=openrouter_routing)
    if connection_history is not None:
        translation = replace(translation, connection_history=connection_history)
    languages = current.intent.languages
    if target_language is not None:
        languages = replace(languages, target_language=target_language)
    overlay = current.intent.overlay
    if overlay_pos_x is not None or overlay_pos_y is not None:
        overlay = replace(
            overlay,
            desktop_flet=replace(
                overlay.desktop_flet,
                position=replace(
                    overlay.desktop_flet.position,
                    x=overlay.desktop_flet.position.x if overlay_pos_x is None else overlay_pos_x,
                    y=overlay.desktop_flet.position.y if overlay_pos_y is None else overlay_pos_y,
                ),
            ),
        )
    state = current.state
    if telemetry_last_sent is not None:
        state = replace(
            state,
            telemetry=replace(state.telemetry, last_sent_date_utc=telemetry_last_sent),
        )
    current = replace(
        current,
        intent=replace(
            current.intent,
            translation=translation,
            languages=languages,
            overlay=overlay,
        ),
        state=state,
    )
    return materialize_canonical_translation_settings(current)


def _byok_pkce_target(settings: AppSettingsVNext) -> OpenRouterPkceTarget | None:
    """Mirror the provider adapter's projection-to-PKCE-target wrapping."""
    owner = SettingsOwner(
        path=Path("settings.json"),
        persistence=SettingsVNextCanonicalPersistenceAdapter(),
        canonical=settings,
    )
    target_settings = owner.build_managed_openrouter_byok_target(settings)
    if target_settings is None:
        return None
    alias = target_settings.intent.translation.openrouter_selection_alias
    if alias is None:
        return None
    return OpenRouterPkceTarget(OpenRouterSelectionAlias(alias))


def _application_boundary_with_stop(backend: object) -> UiApplicationBoundary:
    boundary = compose_test_ui_application_boundary(backend)
    boundary.register_application_shutdown_callbacks(
        (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS,
                owner_name="ApplicationTestRuntime",
                callback_name="stop",
                callback=backend.stop,
            ),
        )
    )
    return boundary


class DummyPage:
    def __init__(self) -> None:
        async def destroy_window() -> None:
            self.window.destroy_calls += 1

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
        self.visibility_updates: list[bool] = []
        self.window = SimpleNamespace(
            frameless=False,
            resizable=False,
            maximizable=True,
            width=0,
            height=0,
            min_width=0,
            max_width=0,
            min_height=0,
            max_height=0,
            prevent_close=False,
            on_event=None,
            destroy_calls=0,
            destroy=destroy_window,
            icon="",
            center_calls=0,
            visible=False,
            center=lambda: None,
        )
        self.window.center = lambda: setattr(
            self.window, "center_calls", self.window.center_calls + 1
        )
        self.dialog = None

    def show_dialog(self, control) -> None:
        if hasattr(control, "open"):
            control.open = True
        self.opened.append(control)
        self.dialog = control

    def pop_dialog(self):
        control = self.dialog
        if control is None:
            return None
        if hasattr(control, "open"):
            control.open = False
        self.closed.append(control)
        self.dialog = None
        return control

    def run_task(self, coro_fn) -> None:
        self.tasks.append(coro_fn)

    def update(self) -> None:
        self.updated += 1
        self.visibility_updates.append(self.window.visible)

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


class FakeTelemetryApplication:
    def __init__(self, settings: AppSettingsVNext) -> None:
        self.settings = settings
        self.applied: list[bool] = []
        self.active_day_results: list[object] = []
        self.active_day_dates: list[str] = []

    async def apply_telemetry_enabled(self, enabled: bool) -> None:
        self.applied.append(enabled)
        if enabled is False:
            self.settings = replace(
                self.settings,
                intent=replace(
                    self.settings.intent,
                    telemetry=replace(self.settings.intent.telemetry, enabled=False),
                ),
                state=replace(
                    self.settings.state,
                    telemetry=replace(
                        self.settings.state.telemetry,
                        anonymous_id=None,
                        last_sent_date_utc=None,
                    ),
                ),
            )

    def settings_general_snapshot(self):
        return SimpleNamespace(telemetry_enabled=self.settings.intent.telemetry.enabled)

    async def record_app_active_day(self, active_date_utc: str) -> object:
        self.active_day_dates.append(active_date_utc)
        if self.active_day_results:
            return self.active_day_results.pop(0)
        return SimpleNamespace(status="sent")


def _make_telemetry_owner(
    application: FakeTelemetryApplication,
    *,
    queued: list | None = None,
    background: list | None = None,
    clock=None,
    sleep=None,
    shutting_down: bool = False,
    synced: list | None = None,
):
    from puripuly_heart.app.services.telemetry_reporting import TelemetryReportingOwner

    return TelemetryReportingOwner(
        application,
        run_background=(
            background.append if background is not None else (lambda coroutine, *args: None)
        ),
        queue_settings_mutation=(
            queued.append if queued is not None else (lambda task_factory: None)
        ),
        is_shutting_down=lambda: shutting_down,
        sync_telemetry=synced.append if synced is not None else None,
        clock=clock,
        sleep=sleep,
    )


def test_telemetry_enabled_choice_persists_and_syncs_settings_view() -> None:
    settings = _vnext(telemetry_last_sent="2026-07-01")
    application = FakeTelemetryApplication(settings)
    synced: list[object] = []
    tasks: list[object] = []
    owner = _make_telemetry_owner(application, background=tasks, synced=synced)

    owner.apply_telemetry_enabled(False)
    assert len(tasks) == 1
    asyncio.run(tasks.pop(0)())
    assert application.applied == [False]
    assert application.settings.intent.telemetry.enabled is False
    assert application.settings.state.telemetry.anonymous_id is None
    assert application.settings.state.telemetry.last_sent_date_utc is None
    assert synced[-1].telemetry_enabled is False

    owner.apply_telemetry_enabled(True)
    asyncio.run(tasks.pop(0)())
    assert application.applied == [False, True]


def test_telemetry_setting_action_does_not_send() -> None:
    application = FakeTelemetryApplication(AppSettingsVNext())
    tasks: list[object] = []
    owner = _make_telemetry_owner(application, background=tasks)

    owner.apply_telemetry_enabled(False)

    assert len(tasks) == 1
    assert application.active_day_dates == []


def test_app_active_day_report_captures_utc_date_before_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FakeTelemetryApplication(AppSettingsVNext())
    queued: list[object] = []

    class CapturedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.fromisoformat("2026-07-03T23:59:59+00:00")

    owner = _make_telemetry_owner(application, queued=queued, clock=CapturedDateTime.now)
    owner.schedule_app_active_day_report()

    assert len(queued) == 1
    asyncio.run(queued.pop(0)())
    assert application.active_day_dates == ["2026-07-03"]


def test_app_active_day_report_retries_once_after_60_seconds_on_same_utc_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FakeTelemetryApplication(AppSettingsVNext())
    application.active_day_results = [
        SimpleNamespace(status="send_failed"),
        SimpleNamespace(status="sent"),
    ]
    queued: list[object] = []
    background: list[object] = []
    sleeps: list[float] = []

    class CapturedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.fromisoformat("2026-07-03T12:00:00+00:00")

    async def fake_sleep(delay_s: float) -> None:
        sleeps.append(delay_s)

    owner = _make_telemetry_owner(
        application,
        queued=queued,
        background=background,
        clock=CapturedDateTime.now,
        sleep=fake_sleep,
    )
    owner.schedule_app_active_day_report()
    asyncio.run(queued.pop(0)())

    assert application.active_day_dates == ["2026-07-03"]
    assert len(background) == 1
    asyncio.run(background.pop(0)())
    assert len(queued) == 1
    asyncio.run(queued.pop(0)())

    assert application.active_day_dates == ["2026-07-03", "2026-07-03"]
    assert sleeps == [APP_ACTIVE_DAY_RETRY_DELAY_S]


def test_app_active_day_report_does_not_retry_after_utc_date_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FakeTelemetryApplication(AppSettingsVNext())
    application.active_day_results = [SimpleNamespace(status="send_failed")]
    queued: list[object] = []
    background: list[object] = []
    now_values = iter(
        (
            real_datetime.fromisoformat("2026-07-03T23:59:59+00:00"),
            real_datetime.fromisoformat("2026-07-04T00:00:01+00:00"),
        )
    )

    class CapturedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return next(now_values)

    async def fake_sleep(delay_s: float) -> None:
        assert delay_s == APP_ACTIVE_DAY_RETRY_DELAY_S

    owner = _make_telemetry_owner(
        application,
        queued=queued,
        background=background,
        clock=CapturedDateTime.now,
        sleep=fake_sleep,
    )
    owner.schedule_app_active_day_report()
    asyncio.run(queued.pop(0)())
    asyncio.run(background.pop(0)())

    assert application.active_day_dates == ["2026-07-03"]
    assert queued == []


def test_app_active_day_report_does_not_retry_when_queue_crosses_utc_midnight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FakeTelemetryApplication(AppSettingsVNext())
    application.active_day_results = [SimpleNamespace(status="send_failed")]
    queued: list[object] = []
    background: list[object] = []
    now_values = iter(
        (
            real_datetime.fromisoformat("2026-07-03T23:59:58+00:00"),
            real_datetime.fromisoformat("2026-07-03T23:59:59+00:00"),
            real_datetime.fromisoformat("2026-07-04T00:00:01+00:00"),
        )
    )

    class CapturedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return next(now_values)

    async def fake_sleep(delay_s: float) -> None:
        assert delay_s == APP_ACTIVE_DAY_RETRY_DELAY_S

    owner = _make_telemetry_owner(
        application,
        queued=queued,
        background=background,
        clock=CapturedDateTime.now,
        sleep=fake_sleep,
    )
    owner.schedule_app_active_day_report()
    asyncio.run(queued.pop(0)())
    asyncio.run(background.pop(0)())

    assert len(queued) == 1
    asyncio.run(queued.pop(0)())
    assert application.active_day_dates == ["2026-07-03"]


@pytest.mark.asyncio
async def test_app_active_day_retry_does_not_block_settings_mutation_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FakeTelemetryApplication(AppSettingsVNext())
    application.active_day_results = [
        SimpleNamespace(status="send_failed"),
        SimpleNamespace(status="sent"),
    ]
    retry_recorded = asyncio.Event()
    original_record = application.record_app_active_day

    async def record_app_active_day(active_date_utc: str) -> object:
        result = await original_record(active_date_utc)
        if len(application.active_day_dates) == 2:
            retry_recorded.set()
        return result

    application.record_app_active_day = record_app_active_day

    from puripuly_heart.app.services.telemetry_reporting import TelemetryReportingOwner

    queue: list[Callable[[], Awaitable[None]]] = []
    worker_active = False
    retry_sleep_started = asyncio.Event()
    release_retry_sleep = asyncio.Event()
    later_mutation_ran = asyncio.Event()
    tasks: list[asyncio.Task[None]] = []

    def run_background(coroutine_factory, *args):
        task = asyncio.create_task(coroutine_factory(*args))
        tasks.append(task)
        return task

    async def queue_worker() -> None:
        nonlocal worker_active
        try:
            while queue:
                next_task = queue.pop(0)
                await next_task()
        finally:
            worker_active = False

    def queue_settings_mutation(task_factory) -> None:
        nonlocal worker_active
        queue.append(task_factory)
        if worker_active:
            return
        worker_active = True
        tasks.append(asyncio.create_task(queue_worker()))

    async def fake_sleep(delay_s: float) -> None:
        assert delay_s == APP_ACTIVE_DAY_RETRY_DELAY_S
        retry_sleep_started.set()
        await release_retry_sleep.wait()

    async def later_settings_mutation() -> None:
        later_mutation_ran.set()

    now_values = iter(
        (
            real_datetime.fromisoformat("2026-07-03T12:00:00+00:00"),
            real_datetime.fromisoformat("2026-07-03T12:00:00+00:00"),
            real_datetime.fromisoformat("2026-07-03T12:00:00+00:00"),
        )
    )

    class CapturedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return next(now_values)

    owner = TelemetryReportingOwner(
        application,
        run_background=run_background,
        queue_settings_mutation=queue_settings_mutation,
        sleep=fake_sleep,
        clock=CapturedDateTime.now,
    )
    owner.schedule_app_active_day_report()
    queue_settings_mutation(later_settings_mutation)

    await asyncio.wait_for(retry_sleep_started.wait(), timeout=1)
    await asyncio.wait_for(later_mutation_ran.wait(), timeout=1)
    assert application.active_day_dates == ["2026-07-03"]

    release_retry_sleep.set()
    await asyncio.wait_for(retry_recorded.wait(), timeout=1)
    await asyncio.gather(*tasks)

    assert application.active_day_dates == ["2026-07-03", "2026-07-03"]
    assert owner._retry_task_handle is None


@pytest.mark.asyncio
async def test_close_after_launch_tasks_cancels_app_active_day_retry() -> None:
    from puripuly_heart.app.services.telemetry_reporting import TelemetryReportingOwner

    application = FakeTelemetryApplication(AppSettingsVNext())
    retry_started = asyncio.Event()
    retry_cancelled = asyncio.Event()

    async def retry_task() -> None:
        retry_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            retry_cancelled.set()

    owner = TelemetryReportingOwner(
        application,
        run_background=lambda coroutine, *args: asyncio.create_task(coroutine()),
        queue_settings_mutation=lambda task_factory: None,
    )
    handle = asyncio.create_task(retry_task())
    owner._retry_task_handle = handle
    await retry_started.wait()
    await owner.cancel_retry()

    assert handle.cancelled()
    assert retry_cancelled.is_set()
    assert owner._retry_task_handle is None


def test_translator_app_telemetry_binding_delegates_to_owner() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    calls: list[object] = []

    class _Owner:
        def apply_telemetry_enabled(self, enabled: bool) -> None:
            calls.append(("apply", enabled))

        def schedule_app_active_day_report(self) -> None:
            calls.append(("schedule",))

    app._telemetry_reporting = _Owner()
    app._on_telemetry_enabled_change(True)
    app._schedule_app_active_day_report()
    assert calls == [("apply", True), ("schedule",)]


@pytest.mark.asyncio
async def test_close_after_launch_cancels_existing_telemetry_retry_only() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    cancelled: list[bool] = []

    class _Owner:
        async def cancel_retry(self) -> None:
            cancelled.append(True)

    app._telemetry_reporting = _Owner()
    app._github_star_prompt_launch_pending = True

    async def _close_handle(_name: str) -> None:
        return None

    app._close_ui_task_handle = _close_handle  # type: ignore[method-assign]
    await app._close_after_launch_ui_tasks()
    assert cancelled == [True]
    assert app._github_star_prompt_launch_pending is False


@pytest.mark.asyncio
async def test_close_after_launch_skips_telemetry_cancel_when_owner_absent() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app._github_star_prompt_launch_pending = True

    async def _close_handle(_name: str) -> None:
        return None

    app._close_ui_task_handle = _close_handle  # type: ignore[method-assign]
    await app._close_after_launch_ui_tasks()
    assert getattr(app, "_telemetry_reporting", None) is None
    assert app._github_star_prompt_launch_pending is False


class ConstructionDummyController:
    def __init__(self, page, app, config_path):
        self.page = page
        self.app = app
        self.config_path = config_path
        self.settings = None
        self.runtime_logging_mode = "detailed"
        self.basic_messages: list[str] = []
        self.detailed_messages: list[str] = []
        self.start_calls = 0
        self.window_visible_at_start: bool | None = None

    async def start(self) -> None:
        self.window_visible_at_start = self.app._app.page.window.visible
        self.start_calls += 1

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

    def _save_settings(self) -> None:
        return None

    def persist_settings(self) -> None:
        return None

    async def install_selected_gpu_model_if_needed(self) -> bool:
        return False


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

    def bind_dashboard_intents(self, *, translation, capture) -> None:
        self.on_send_message = translation.submit_message
        self.on_toggle_translation = translation.toggle_translation
        self.on_language_change = translation.change_language
        self.on_message_input_activity = translation.report_input_activity
        self.on_toggle_stt = capture.toggle_self_capture
        self.on_toggle_peer_translation = capture.toggle_peer_capture
        self.on_toggle_overlay = capture.toggle_overlay
        self.on_retry_peer_process_capture = capture.retry_peer_process_capture
        self.on_gpu_notice_action = capture.run_gpu_notice_action

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
        self.synced_desktop_settings: list[OverlaySettingsSnapshot] = []

    def bind_settings_intents(self, *, surface, provider, general, prompt, overlay) -> None:
        self.on_settings_changed = surface.settings_changed
        self.show_snackbar = surface.show_snackbar
        if surface.runtime_log_basic is not None:
            self.runtime_log_basic = surface.runtime_log_basic
        if surface.runtime_log_detailed is not None:
            self.runtime_log_detailed = surface.runtime_log_detailed
        self.on_providers_changed = provider.providers_changed
        self.on_request_openrouter_pkce = provider.request_openrouter_pkce
        self.on_verify_api_key = provider.verify_api_key
        self.on_provider_secret_change = provider.provider_secret_change
        self.on_secret_cleared = provider.secret_cleared
        self.on_local_llm_secret_changed = provider.local_llm_secret_changed
        self.on_gpu_discovery_requested = provider.gpu_discovery_requested
        self.on_start_microphone_test = general.start_microphone_test
        self.on_telemetry_enabled_change = general.telemetry_enabled_change
        self.on_list_loopback_capture_options = general.list_loopback_capture_options
        self.on_list_loopback_process_options = general.list_loopback_process_options
        self.on_list_loopback_device_options = general.list_loopback_device_options
        self.on_current_loopback_capture_option = general.current_loopback_capture_option
        self.on_apply_loopback_capture_option = general.apply_loopback_capture_option
        self.on_loopback_capture_summary = general.loopback_capture_summary
        self.on_prompt_apply_settings = prompt.prompt_apply_settings
        self.on_desktop_overlay_lock_change = overlay.desktop_overlay_lock_change
        self.on_desktop_overlay_size_change = overlay.desktop_overlay_size_change
        self.on_desktop_overlay_recovery_action = overlay.desktop_overlay_recovery_action
        self.on_desktop_overlay_position_reset = overlay.desktop_overlay_position_reset
        self.on_view_logs = overlay.view_logs
        self.on_overlay_calibration_begin = overlay.calibration_begin
        self.on_overlay_calibration_change = overlay.calibration_change
        self.on_overlay_calibration_apply = overlay.calibration_apply
        self.on_overlay_calibration_cancel = overlay.calibration_cancel

    def set_overlay_runtime_state(self, *_args, **_kwargs) -> None:
        return None

    def sync_desktop_overlay_settings(self, settings: OverlaySettingsSnapshot) -> None:
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
        self.bound_logs_intents: list[object] = []
        self.runtime_logging_mode = "basic"

    def bind_logs_intents(self, intents: object) -> None:
        self.bound_logs_intents.append(intents)
        self.on_mode_change = getattr(intents, "runtime_logging_mode_change", None)

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


_construction_backends: list[ConstructionDummyController] = []


def _construction_application_factory(
    *,
    presentation,
    config_path,
    runtime_logging_sinks=None,
    vrchat_osc_presence=None,
):
    _ = (
        runtime_logging_sinks,
        vrchat_osc_presence,
    )
    backend = ConstructionDummyController(
        page=None,
        app=presentation,
        config_path=config_path,
    )
    _construction_backends.append(backend)
    return compose_test_ui_application_boundary(backend)


def _patch_app_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "DashboardView", ConstructionDummyDashboardView)
    monkeypatch.setattr(app_module, "SettingsView", ConstructionDummySettingsView)
    monkeypatch.setattr(app_module, "LogsView", ConstructionDummyLogsView)
    monkeypatch.setattr(app_module, "AboutView", lambda: ft.Container())
    monkeypatch.setattr(
        app_module,
        "TitleBar",
        lambda _page, *, on_close: ft.Container(data=on_close),
    )
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
    app = TranslatorApp(
        page,
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )
    backend = _construction_backends[-1]

    assert backend.config_path == Path("settings.json")
    assert page.title == app_module.t("app.title")
    assert page.window.frameless is True
    assert page.window.resizable is False
    assert page.window.maximizable is False
    assert page.window.width == app_module.DEFAULT_WINDOW_WIDTH
    assert page.window.height == app_module.DEFAULT_WINDOW_HEIGHT
    assert page.window.min_width == app_module.DEFAULT_WINDOW_WIDTH
    assert page.window.max_width == app_module.DEFAULT_WINDOW_WIDTH
    assert page.window.min_height == app_module.DEFAULT_WINDOW_HEIGHT
    assert page.window.max_height == app_module.DEFAULT_WINDOW_HEIGHT
    assert page.window.prevent_close is True
    assert page.window.on_event == app._on_window_event
    assert page.window.center_calls == 0
    assert page.added
    assert app.view_dashboard.on_send_message == app._on_manual_submit
    assert app.view_dashboard.on_message_input_activity == app._on_message_input_activity
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
    assert app.view_settings.runtime_log_basic == app.application.log_basic
    assert app.view_settings.runtime_log_detailed == app.application.log_detailed
    assert app.view_logs.on_mode_change == app._on_runtime_logging_mode_change
    assert app.view_logs.runtime_logging_mode == "detailed"
    assert isinstance(app.application, UiApplicationBoundary)
    assert app.application is app._ui_application
    assert isinstance(backend.app, FletUiPresentationAdapter)
    assert backend.app is app._presentation_adapter


def test_application_property_preserves_factory_port_without_controller_compatibility() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    application = object()
    app._ui_application = application

    assert app.application is application


def test_translator_app_consumes_production_application_factory_without_controller_reach_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    app = TranslatorApp(
        DummyPage(),
        config_path=Path("settings.json"),
        application_factory=compose_ui_application,
    )

    assert app.application is app._ui_application
    assert app.application.state().config_path == Path("settings.json")
    assert not hasattr(app, "controller")


def test_translator_app_rejects_an_empty_application_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="Application factory did not compose an application boundary",
    ):
        TranslatorApp(
            DummyPage(),
            config_path=Path("settings.json"),
            application_factory=lambda **_kwargs: None,
        )


@pytest.mark.asyncio
async def test_window_close_awaits_application_shutdown_before_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)
    page = DummyPage()
    app = TranslatorApp(
        page,
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )
    transitions: list[str] = []

    async def shutdown() -> None:
        transitions.append("shutdown")

    async def destroy() -> None:
        transitions.append("destroy")

    app.shutdown = shutdown
    page.window.destroy = destroy

    app._on_window_event(SimpleNamespace(type=ft.WindowEventType.RESIZE))
    assert page.tasks == []

    app.title_bar.data()
    assert len(page.tasks) == 1
    app._on_window_event(SimpleNamespace(type=ft.WindowEventType.CLOSE))
    assert len(page.tasks) == 1

    await page.tasks[0]()

    assert transitions == ["shutdown", "destroy"]


@pytest.mark.asyncio
async def test_main_gui_constructs_the_real_application_and_presentation_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)
    constructed: dict[str, object] = {}

    def application_factory(
        *,
        presentation,
        config_path,
        runtime_logging_sinks=None,
        vrchat_osc_presence=None,
    ):
        _ = (
            runtime_logging_sinks,
            vrchat_osc_presence,
        )
        controller = ConstructionDummyController(
            page=None,
            app=presentation,
            config_path=config_path,
        )
        constructed["controller"] = controller
        return compose_test_ui_application_boundary(controller)

    monkeypatch.setattr(TranslatorApp, "schedule_after_launch_tasks", lambda self: None)

    page = DummyPage()
    await app_module.main_gui(
        page,
        config_path=Path("settings.json"),
        application_factory=application_factory,
    )

    controller = constructed["controller"]
    assert isinstance(controller, ConstructionDummyController)
    assert isinstance(controller.app, FletUiPresentationAdapter)
    translator_app = controller.app._app
    assert isinstance(translator_app, TranslatorApp)
    assert isinstance(translator_app.application, UiApplicationBoundary)
    assert translator_app.application is translator_app._ui_application
    assert controller.start_calls == 1
    assert controller.window_visible_at_start is True
    assert page.window.center_calls == 1
    assert page.window.visible is True
    assert page.visibility_updates[-2:] == [False, True]


@pytest.mark.asyncio
async def test_desktop_gui_state_actions_are_dispatched_through_translator_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    page = DummyPage()
    app = TranslatorApp(
        page,
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )
    backend = _construction_backends[-1]
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

    backend.set_desktop_overlay_captions_locked = fake_set_locked
    backend.set_desktop_overlay_size_preset = fake_set_desktop_overlay_size_preset
    backend.set_overlay_enabled = fake_set_overlay_enabled
    backend.reset_desktop_overlay_position = fake_reset_desktop_overlay_position

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
    app = TranslatorApp(
        page,
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )
    backend = _construction_backends[-1]
    backend.settings = _vnext(overlay_pos_x=80, overlay_pos_y=90)

    async def fake_set_locked(locked: bool) -> None:
        backend.desktop_overlay_captions_locked = locked

    async def fake_set_desktop_overlay_size_preset(size_preset: str) -> None:
        backend.settings = replace(
            backend.settings,
            intent=replace(
                backend.settings.intent,
                overlay=replace(
                    backend.settings.intent.overlay,
                    desktop_flet=replace(
                        backend.settings.intent.overlay.desktop_flet,
                        size_preset=size_preset,
                    ),
                ),
            ),
        )

    async def fake_reset_desktop_overlay_position() -> None:
        overlay = backend.settings.intent.overlay
        backend.settings = replace(
            backend.settings,
            intent=replace(
                backend.settings.intent,
                overlay=replace(
                    overlay,
                    desktop_flet=replace(
                        overlay.desktop_flet,
                        position=replace(overlay.desktop_flet.position, x=None, y=None),
                    ),
                ),
            ),
        )
        backend.desktop_overlay_captions_locked = False

    backend.set_desktop_overlay_captions_locked = fake_set_locked
    backend.set_desktop_overlay_size_preset = fake_set_desktop_overlay_size_preset
    backend.reset_desktop_overlay_position = fake_reset_desktop_overlay_position

    app._on_desktop_overlay_lock_change(True)
    app._on_desktop_overlay_size_change("xlarge")
    app._on_desktop_overlay_position_reset()

    assert len(page.tasks) == 3
    await page.tasks[0]()
    await page.tasks[1]()
    await page.tasks[2]()

    synced_settings = app.view_settings.synced_desktop_settings
    assert len(synced_settings) == 3
    assert synced_settings[1].desktop_size_preset == "xlarge"
    assert synced_settings[2].desktop_size_preset == "xlarge"
    assert backend.settings.intent.overlay.desktop_flet.position.x is None
    assert backend.settings.intent.overlay.desktop_flet.position.y is None
    assert getattr(backend, "desktop_overlay_captions_locked", False) is False


def test_translator_app_does_not_mount_debug_preview_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    page = DummyPage()
    app = TranslatorApp(
        page,
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )

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
        application_factory=_construction_application_factory,
        debug_ui_preview=True,
    )

    assert app.debug_ui_preview is True
    assert app.debug_preview_panel is not None
    assert set(seen["callbacks"]) == {
        "on_display_turn_cycle",
        "on_peer_translation_eula",
        "on_discord_auth",
        "on_qq_auth",
        "on_qq_auth_recoverable_error",
        "on_qq_auth_translation_gated",
        "on_discord_callback_page",
        "on_founder_letter",
        "on_local_qwen_hallucination_modal",
        "on_github_star_snackbar",
        "on_talk_together_pass_invite_progress",
        "on_foundation_primitives",
        "on_http_extension_form",
        "on_brake_notice",
        "on_revoked_notice",
        "on_pkce_failure",
        "on_capture_fault_cycle",
        "on_stt_fault_cycle",
        "on_audio_fault_clear",
        "on_gpu_state_cycle",
        "on_stt_loading_button_cycle",
    }
    display_turn = seen["callbacks"]["on_display_turn_cycle"]
    assert getattr(display_turn, "__self__", None) is app
    assert (
        getattr(display_turn, "__func__", None) is TranslatorApp._cycle_debug_preview_display_turn
    )
    discord_callback = seen["callbacks"]["on_discord_auth"]
    assert getattr(discord_callback, "__self__", None) is app
    assert getattr(discord_callback, "__func__", None) is TranslatorApp._preview_discord_auth
    qq_callback = seen["callbacks"]["on_qq_auth"]
    assert getattr(qq_callback, "__self__", None) is app
    assert getattr(qq_callback, "__func__", None) is TranslatorApp._preview_qq_auth
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


def test_debug_preview_display_turn_cycles_source_and_translation_examples() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    displayed: list[tuple[str, str, str]] = []
    app.view_dashboard = SimpleNamespace(
        set_display_text=lambda text, *, language_code: displayed.append(
            ("source", language_code, text)
        ),
        set_display_translation_text=lambda text, *, language_code: displayed.append(
            ("translation", language_code, text)
        ),
    )

    for _ in range(4):
        app._cycle_debug_preview_display_turn()

    assert displayed == [
        ("source", "ko", "오늘 같이 뭐 하고 놀까?"),
        ("translation", "en", "What should we do together today?"),
        (
            "source",
            "ko",
            "아까 말한 그 월드 이름이 뭐였지? 다시 가보고 싶은데 검색해도 안 나오더라.",
        ),
        (
            "translation",
            "en",
            "What was the name of that world you mentioned earlier? "
            "I want to visit again but searching turns up nothing.",
        ),
    ]


def test_debug_preview_http_extension_form_writes_demo_extension_and_selects_it(
    tmp_path: Path,
) -> None:
    registry = HttpExtensionRegistry(tmp_path)
    registry.reload()
    service = HttpExtensionRegistryService(registry)
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._ui_application = SimpleNamespace(http_extension_registry=lambda: service)
    llm_values: list[str] = []
    extension_values: list[str] = []
    app.view_settings = SimpleNamespace(
        set_http_extension_registry=lambda _service: None,
        _on_llm_selected=lambda value: llm_values.append(value),
        _on_http_extension_selected=lambda value: extension_values.append(value),
    )
    app._current_tab = 1

    app._preview_http_extension_form()

    demo_path = tmp_path / "debug_demo.json"
    assert demo_path.exists()
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    assert demo["secrets"] == [{"id": "api_key", "label": "API Key"}]
    assert llm_values == [TranslationModel.CUSTOM_HTTP.value]
    assert extension_values == ["debug_demo"]
    assert registry.snapshot.get("debug_demo") is not None

    app._preview_http_extension_form()
    assert llm_values == [TranslationModel.CUSTOM_HTTP.value] * 2
    assert extension_values == ["debug_demo", "debug_demo"]


def test_debug_preview_local_qwen_modal_opens_production_dialog_without_state_or_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    controller = SimpleNamespace(
        _local_qwen_hallucination_detection_count=1,
        _local_qwen_hallucination_modal_shown=False,
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
    assert controller._local_qwen_hallucination_detection_count == 1
    assert controller._local_qwen_hallucination_modal_shown is False
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
        DummyPage(),
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
        debug_ui_preview=True,
    )
    backend = _construction_backends[-1]
    monkeypatch.setattr(
        app, "_show_snackbar", lambda message, color=None: snackbars.append((message, color))
    )

    assert callable(captured_kwargs["on_capture_fault_cycle"])
    assert callable(captured_kwargs["on_stt_fault_cycle"])
    assert callable(captured_kwargs["on_audio_fault_clear"])
    captured_kwargs["on_capture_fault_cycle"]()
    captured_kwargs["on_stt_fault_cycle"]()
    captured_kwargs["on_audio_fault_clear"]()
    assert backend.capture_fault_cycled is True
    assert backend.stt_fault_cycled is True
    assert backend.audio_faults_cleared is True
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
        ConstructionDummyController,
        "_save_settings",
        lambda self: forbidden_calls.append("save_settings"),
    )
    monkeypatch.setattr(
        ConstructionDummyController,
        "persist_settings",
        lambda self: forbidden_calls.append("persist_settings"),
    )
    monkeypatch.setattr(
        app_module,
        "webbrowser",
        SimpleNamespace(open=lambda *args, **kwargs: forbidden_calls.append("webbrowser.open")),
    )

    app = app_module.TranslatorApp(
        DummyPage(),
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
        debug_ui_preview=True,
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
        ConstructionDummyController,
        "_save_settings",
        lambda self: forbidden_calls.append("save_settings"),
    )
    monkeypatch.setattr(
        ConstructionDummyController,
        "persist_settings",
        lambda self: forbidden_calls.append("persist_settings"),
    )
    monkeypatch.setattr(
        app_module.webbrowser,
        "open",
        lambda url, *args, **kwargs: opened_urls.append(url),
    )
    monkeypatch.setattr(app_module, "get_locale", lambda: "ko")

    page = DummyPage()
    app = TranslatorApp(
        page,
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )
    monkeypatch.setattr(app.content_area, "update", lambda: None)
    backend = _construction_backends[-1]
    backend.apply_settings = lambda *args, **kwargs: forbidden_calls.append("apply_settings")
    backend.apply_providers = lambda *args, **kwargs: forbidden_calls.append("apply_providers")
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
                invite_limit=3,
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
    app._ui_application = compose_test_ui_application_boundary(None)
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

    def application_factory(
        *,
        presentation,
        config_path,
        runtime_logging_sinks=None,
        vrchat_osc_presence=None,
    ):
        _ = (
            runtime_logging_sinks,
            vrchat_osc_presence,
        )
        controller = RecordingController(None, presentation, config_path)
        return compose_test_ui_application_boundary(controller)

    app = TranslatorApp(
        DummyPage(),
        config_path=Path("settings.json"),
        application_factory=application_factory,
        debug_ui_preview=True,
    )

    assert app.debug_ui_preview is True
    assert seen["controller_args"] == (
        None,
        app._presentation_adapter,
        Path("settings.json"),
    )
    assert not hasattr(app._presentation_adapter, "app")


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

        def bind_dashboard_intents(self, *, translation, capture) -> None:
            self.on_send_message = translation.submit_message
            self.on_toggle_translation = translation.toggle_translation
            self.on_language_change = translation.change_language
            self.on_message_input_activity = translation.report_input_activity
            self.on_toggle_stt = capture.toggle_self_capture
            self.on_toggle_peer_translation = capture.toggle_peer_capture
            self.on_toggle_overlay = capture.toggle_overlay
            self.on_retry_peer_process_capture = capture.retry_peer_process_capture
            self.on_gpu_notice_action = capture.run_gpu_notice_action

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

        def bind_settings_intents(self, *, surface, provider, general, prompt, overlay) -> None:
            self.on_settings_changed = surface.settings_changed
            self.show_snackbar = surface.show_snackbar
            self.on_providers_changed = provider.providers_changed
            self.on_request_openrouter_pkce = provider.request_openrouter_pkce
            self.on_verify_api_key = provider.verify_api_key
            self.on_provider_secret_change = provider.provider_secret_change
            self.on_secret_cleared = provider.secret_cleared
            self.on_local_llm_secret_changed = provider.local_llm_secret_changed
            self.on_gpu_discovery_requested = provider.gpu_discovery_requested
            self.on_start_microphone_test = general.start_microphone_test
            self.on_prompt_apply_settings = prompt.prompt_apply_settings
            self.on_view_logs = overlay.view_logs

        def set_overlay_runtime_state(self, *_args, **_kwargs) -> None:
            return None

        def apply_locale(self) -> None:
            return None

    class DummyLogsView(ft.Container):
        def __init__(self) -> None:
            super().__init__()
            self.on_mode_change = None

        def bind_logs_intents(self, intents: object) -> None:
            self.on_mode_change = getattr(intents, "runtime_logging_mode_change", None)

        def set_runtime_logging_mode(self, mode: str) -> None:
            _ = mode

        def apply_locale(self) -> None:
            return None

        async def scroll_to_bottom(self) -> None:
            return None

    monkeypatch.setattr(app_module, "DashboardView", DummyDashboardView)
    monkeypatch.setattr(app_module, "SettingsView", DummySettingsView)
    monkeypatch.setattr(app_module, "LogsView", DummyLogsView)
    monkeypatch.setattr(app_module, "AboutView", lambda: ft.Container())
    monkeypatch.setattr(
        app_module,
        "TitleBar",
        lambda _page, *, on_close: ft.Container(data=on_close),
    )
    monkeypatch.setattr(app_module, "BottomNavBar", lambda on_change: ft.Container(data=on_change))
    monkeypatch.setattr(app_module, "register_fonts", lambda _page: None)
    monkeypatch.setattr(app_module, "get_app_theme", lambda **_kwargs: "theme")
    monkeypatch.setattr(app_module, "font_for_language", lambda _code: "font")
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")

    def application_factory(
        *,
        presentation,
        config_path,
        runtime_logging_sinks=None,
        vrchat_osc_presence=None,
    ):
        _ = (
            runtime_logging_sinks,
            vrchat_osc_presence,
        )
        controller = DummyController(None, presentation, config_path)
        return compose_test_ui_application_boundary(controller)

    app = TranslatorApp(
        DummyPage(),
        config_path=Path("settings.json"),
        application_factory=application_factory,
    )

    assert app.view_dashboard.runtime_log_detailed == app._log_detailed


def test_settings_view_pkce_callback_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_app_construction(monkeypatch)

    app = TranslatorApp(
        DummyPage(),
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )

    assert app.view_settings.on_request_openrouter_pkce == app._on_request_openrouter_pkce


def test_translator_app_4x3_window_keeps_shell_navigation_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)

    page = DummyPage()
    app = TranslatorApp(
        page,
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )
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
        controller.runtime_logging_mode = mode

    controller = SimpleNamespace(
        runtime_logging_mode="basic",
        set_runtime_logging_mode=fake_set_mode,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)
    app.view_logs = SimpleNamespace(
        set_runtime_logging_mode=lambda mode: seen.append(f"view:{mode}")
    )

    app._on_runtime_logging_mode_change("detailed")

    assert seen == ["detailed", "view:detailed"]


def _make_fake_main_gui_app(seen: dict[str, object]) -> type:
    class FakeController:
        async def start(self) -> None:
            seen["started"] = True

    class FakeApp:
        def __init__(
            self,
            incoming_page,
            *,
            config_path,
            application_factory,
            debug_ui_preview=False,
            runtime_logging_sinks=None,
            vrchat_osc_presence=None,
        ):
            seen["init"] = (incoming_page, config_path, debug_ui_preview)
            seen["app"] = self
            self.page = incoming_page
            _ = (
                application_factory,
                runtime_logging_sinks,
                vrchat_osc_presence,
            )
            backend = FakeController()
            self.application = compose_test_ui_application_boundary(backend)

        async def _on_page_lifecycle_end(self, _event=None) -> None:
            return None

        async def shutdown(self) -> None:
            await self.application.stop()

        def _log_detailed(self, message: str, *, level: int = app_module.logging.INFO) -> None:
            _ = (message, level)

        def schedule_after_launch_tasks(self) -> None:
            async def run() -> None:
                await app_module._check_and_notify_update(
                    self.page,
                    log_detailed=self._log_detailed,
                )

            seen["after_launch_task"] = asyncio.create_task(run())

    return FakeApp


@pytest.mark.asyncio
async def test_main_gui_routes_update_check_through_app_log_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = DummyPage()
    seen: dict[str, object] = {}
    FakeApp = _make_fake_main_gui_app(seen)

    async def fake_check_and_notify_update(incoming_page, *, log_detailed=None) -> None:
        seen["check"] = (incoming_page, log_detailed)

    monkeypatch.setattr(app_module, "TranslatorApp", FakeApp)
    monkeypatch.setattr(app_module, "_check_and_notify_update", fake_check_and_notify_update)

    await app_module.main_gui(
        page,
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )
    await seen["after_launch_task"]

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
    FakeApp = _make_fake_main_gui_app(seen)

    async def fake_check_and_notify_update(incoming_page, *, log_detailed=None) -> None:
        seen["check"] = (incoming_page, log_detailed)

    monkeypatch.setattr(app_module, "TranslatorApp", FakeApp)
    monkeypatch.setattr(app_module, "_check_and_notify_update", fake_check_and_notify_update)

    await app_module.main_gui(
        page,
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
        debug_ui_preview=True,
    )
    await seen["after_launch_task"]

    assert seen["started"] is True
    assert seen["init"] == (page, Path("settings.json"), True)
    assert seen["check"][0] is page


def test_debug_preview_gpu_states_cycle_in_memory() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.debug_ui_preview = True
    app.page = DummyPage()
    app.view_settings = SimpleNamespace(
        devices=[],
        set_gpu_devices=lambda **kwargs: app.view_settings.devices.append(kwargs["devices"]),
    )
    app.view_dashboard = SimpleNamespace(
        notices=[],
        set_gpu_notice=lambda notice: app.view_dashboard.notices.append(notice),
    )
    for _ in range(8):
        app._cycle_debug_preview_gpu_state()

    assert [notice.status for notice in app.view_dashboard.notices] == [
        "discovery_failed",
        "not_installed",
        "invalid",
        "installing",
        "install_failed",
        "unsupported",
        "unavailable_device",
        "activation_failed",
    ]
    assert [notice.action for notice in app.view_dashboard.notices] == [
        "rediscover",
        None,
        None,
        None,
        None,
        None,
        None,
        "restart",
    ]
    assert len(app.view_settings.devices) == 8


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


def test_debug_preview_founder_letter_opens_dialog_with_readme_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    captured: dict[str, object] = {}
    opened_urls: list[str] = []
    previous_locale = i18n_module.get_locale()

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
    settings = _vnext(
        llm="openrouter",
        openrouter_source="managed",
        openrouter_alias=OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED.value,
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
    assert (
        settings.intent.translation.openrouter_selected_source
        == OpenRouterCredentialSource.MANAGED.value
    )


def test_debug_preview_qq_auth_states_are_pure_ui_without_tasks_or_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = _vnext(llm="openrouter", openrouter_source="managed")
    monkeypatch.setattr(
        app,
        "_start_qq_managed_auth",
        lambda *_args, **_kwargs: pytest.fail("debug QQ-auth preview must not start auth"),
        raising=False,
    )

    app._preview_qq_auth()
    initial_dialog = app._qq_managed_auth_dialog
    assert initial_dialog._dialog is app.page.opened[-1]
    initial_dialog._continue_button.on_click(None)
    assert app.page.tasks == []
    assert initial_dialog._error_text.value == app_module.t("qq_auth.error.invalid_input")
    initial_dialog._close_button.on_click(None)
    assert app.page.closed[-1] is initial_dialog._dialog

    app._preview_qq_auth_recoverable_error()
    recoverable_dialog = app._qq_managed_auth_dialog
    assert recoverable_dialog._error_text.value == app_module.t("qq_auth.error.credential_mismatch")

    app._preview_qq_auth_translation_gated()
    gated_dialog = app._qq_managed_auth_dialog
    assert gated_dialog._error_text.value == app_module.t("qq_auth.error.key_unavailable")
    assert app.page.tasks == []
    assert (
        settings.intent.translation.openrouter_selected_source
        == OpenRouterCredentialSource.MANAGED.value
    )


def test_discord_managed_auth_byok_launches_openrouter_pkce_with_byok_target() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    settings = _vnext(
        model="openrouter_qwen35_flash",
        connection="managed",
        openrouter_source="managed",
        openrouter_alias=OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED.value,
        openrouter_model=OpenRouterLLMModel.QWEN_35_FLASH_02_23.value,
    )
    app._ui_application = SimpleNamespace(
        build_managed_openrouter_byok_target=lambda: _byok_pkce_target(settings),
    )
    pkce_calls: list[tuple[object, str]] = []
    app._on_request_openrouter_pkce = lambda target, *, launch_source="settings": pkce_calls.append(
        (target, launch_source)
    )
    app._show_snackbar = lambda *_args, **_kwargs: pytest.fail(
        "managed Discord auth BYOK should build a valid OpenRouter target"
    )

    app._on_discord_managed_auth_byok()

    assert len(pkce_calls) == 1
    target, launch_source = pkce_calls[0]
    assert launch_source == "discord_auth"
    assert target.selection_alias is OpenRouterSelectionAlias.QWEN35_FLASH_BYOK
    assert (
        settings.intent.translation.openrouter_selected_source
        == OpenRouterCredentialSource.MANAGED.value
    )


def test_discord_managed_auth_byok_clears_managed_china_translation_state() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    settings = _vnext(
        llm="openrouter",
        model=TranslationModel.DEEPSEEK_V4_FLASH.value,
        connection=TranslationConnection.MANAGED_CHINA.value,
        openrouter_source="managed",
        openrouter_alias=OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED.value,
        openrouter_model=OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value,
        openrouter_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY.value,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA.value,
        },
    )
    app._ui_application = SimpleNamespace(
        build_managed_openrouter_byok_target=lambda: _byok_pkce_target(settings),
    )

    target = app._build_managed_openrouter_byok_target()

    assert target is not None
    assert target.selection_alias is OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK


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

    controller = SimpleNamespace(
        hub=hub,
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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
async def test_start_qq_managed_auth_uses_run_task_and_success_closes_dialog() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dialog = SimpleNamespace(
        qq_identity="qq-user",
        credential="credential",
        set_waiting_calls=0,
        close_calls=0,
    )
    dialog.set_waiting = lambda: setattr(dialog, "set_waiting_calls", dialog.set_waiting_calls + 1)
    dialog.close = lambda: setattr(dialog, "close_calls", dialog.close_calls + 1)
    app._qq_managed_auth_dialog = dialog
    app._qq_managed_auth_generation = 0
    app._qq_managed_auth_cancelled = False
    snackbar_calls: list[tuple[str, object]] = []
    dashboard_translation_calls: list[bool] = []
    start_calls: list[tuple[str, str]] = []
    enable_calls: list[bool] = []
    hub = SimpleNamespace(llm=object(), translation_enabled=False)
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )

    async def fake_start_qq_managed_auth_from_dialog(**kwargs) -> bool:
        start_calls.append((kwargs["qq_identity"], kwargs["credential"]))
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        enable_calls.append(enabled)
        hub.translation_enabled = enabled
        return True

    controller = SimpleNamespace(
        hub=hub,
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._start_qq_managed_auth()

    assert dialog.set_waiting_calls == 1
    assert start_calls == []
    assert len(app.page.tasks) == 1

    await app.page.tasks[0]()

    assert start_calls == [("qq-user", "credential")]
    assert enable_calls == [True]
    assert hub.translation_enabled is True
    assert dialog.close_calls == 1
    assert snackbar_calls == [(app_module.t("qq_auth.success"), app_module.COLOR_SUCCESS)]
    assert dashboard_translation_calls == [True]


def test_managed_china_dashboard_prompt_opens_qq_auth_not_discord() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dashboard_translation_calls: list[bool] = []
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    discord_calls: list[bool] = []
    qq_calls: list[str] = []
    app.show_discord_managed_auth_dialog = lambda *, preview=False: discord_calls.append(preview)
    app.show_qq_managed_auth_dialog = lambda: qq_calls.append("qq")
    controller = SimpleNamespace(
        dashboard_managed_auth_action=lambda: "prompt",
        dashboard_managed_auth_prompt_kind=lambda: "qq",
        set_translation_enabled=lambda _enabled: pytest.fail(
            "dashboard QQ prompt should not bypass auth dialog"
        ),
    )
    app._ui_application = compose_test_ui_application_boundary(controller)
    app._log_basic = lambda *_args, **_kwargs: None
    app._log_detailed = lambda *_args, **_kwargs: None

    handled = app._on_translation_toggle(True)

    assert handled is False
    assert qq_calls == ["qq"]
    assert discord_calls == []
    assert dashboard_translation_calls == [False]
    assert app.page.tasks == []


@pytest.mark.asyncio
async def test_start_qq_managed_auth_renders_recoverable_error_without_closing() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    error_calls: list[tuple[str, dict[str, object]]] = []
    dialog = SimpleNamespace(
        qq_identity="qq-user",
        credential="credential",
        set_waiting=lambda: None,
        close=lambda: pytest.fail("recoverable QQ error must keep the dialog open"),
        set_error=lambda key, **kwargs: error_calls.append((key, kwargs)),
    )
    app._qq_managed_auth_dialog = dialog
    app._qq_managed_auth_generation = 0
    app._qq_managed_auth_cancelled = False
    app._show_snackbar = lambda *_args, **_kwargs: pytest.fail(
        "recoverable QQ error must render in dialog"
    )

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs):
        return "qq_auth.error.rate_limited", {"retry_after_ms": 3000}

    controller = SimpleNamespace(
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._start_qq_managed_auth()
    await app.page.tasks[0]()

    assert error_calls == [("qq_auth.error.rate_limited", {"retry_after_ms": 3000})]


@pytest.mark.asyncio
async def test_start_qq_managed_auth_key_unavailable_stays_recoverable_and_translation_off() -> (
    None
):
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    error_calls: list[tuple[str, dict[str, object]]] = []
    dialog = SimpleNamespace(
        qq_identity="qq-user",
        credential="credential",
        set_waiting=lambda: None,
        close=lambda: pytest.fail("key unavailable must keep the QQ auth dialog open"),
        set_error=lambda key, **kwargs: error_calls.append((key, kwargs)),
    )
    app._qq_managed_auth_dialog = dialog
    app._qq_managed_auth_generation = 0
    app._qq_managed_auth_cancelled = False
    app._show_snackbar = lambda *_args, **_kwargs: pytest.fail(
        "key unavailable must not render as success"
    )
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda _enabled: pytest.fail(
            "key unavailable must not visually enable translation"
        )
    )

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs):
        return "qq_auth.error.key_unavailable", {}

    async def fake_set_translation_enabled(_enabled: bool):
        pytest.fail("key unavailable must not enable translation")

    controller = SimpleNamespace(
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._start_qq_managed_auth()
    await app.page.tasks[0]()

    assert error_calls == [("qq_auth.error.key_unavailable", {})]


@pytest.mark.asyncio
async def test_start_qq_managed_auth_exception_uses_safe_bounded_log() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    error_calls: list[tuple[str, dict[str, object]]] = []
    dialog = SimpleNamespace(
        qq_identity="raw-qq-user",
        credential="raw-credential",
        set_waiting=lambda: None,
        set_error=lambda key, **kwargs: error_calls.append((key, kwargs)),
    )
    app._qq_managed_auth_dialog = dialog
    app._qq_managed_auth_generation = 0
    app._qq_managed_auth_cancelled = False
    logs: list[tuple[str, object]] = []
    app._log_basic = lambda message, *, level=app_module.logging.INFO: logs.append((message, level))

    async def fake_start_qq_managed_auth_from_dialog(**_kwargs):
        raise RuntimeError("raw broker payload and raw-credential must not leak")

    controller = SimpleNamespace(
        start_qq_managed_auth_from_dialog=fake_start_qq_managed_auth_from_dialog,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._start_qq_managed_auth()
    await app.page.tasks[0]()

    assert error_calls == [("qq_auth.error.retry", {})]
    rendered_logs = repr(logs)
    assert "raw broker payload" not in rendered_logs
    assert "raw-credential" not in rendered_logs
    assert "Traceback" not in rendered_logs


def test_start_discord_managed_auth_registers_page_task_with_oauth_runtime() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._discord_managed_auth_dialog = SimpleNamespace(referral_id="", set_waiting=lambda: None)
    app._show_snackbar = lambda *_args, **_kwargs: None
    app.view_dashboard = SimpleNamespace(set_translation_enabled=lambda _enabled: None)
    controller = SimpleNamespace(
        hub=SimpleNamespace(llm=object(), translation_enabled=False),
        start_discord_managed_auth_from_dialog=lambda **_kwargs: False,
        set_translation_enabled=lambda _enabled: None,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)
    app._start_discord_managed_auth()

    assert app.application.managed_auth_task_names() == ("discord-managed-auth-dialog",)
    assert len(app.page.tasks) == 1


@pytest.mark.asyncio
async def test_start_discord_managed_auth_skips_controller_start_for_stale_generation() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._discord_managed_auth_dialog = SimpleNamespace(referral_id="", set_waiting=lambda: None)
    start_calls: list[str] = []

    async def fake_start_discord_managed_auth_from_dialog(**_kwargs) -> bool:
        start_calls.append("start")
        return False

    controller = SimpleNamespace(
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._start_discord_managed_auth()
    app._discord_managed_auth_generation += 1
    await app.page.tasks[0]()

    assert start_calls == []


@pytest.mark.asyncio
async def test_close_oauth_runtime_blocks_late_discord_auth_ui_mutation() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    dialog = SimpleNamespace(referral_id="", close_calls=0, set_waiting=lambda: None)
    dialog.close = lambda: setattr(dialog, "close_calls", dialog.close_calls + 1)
    app._discord_managed_auth_dialog = dialog
    app._discord_managed_auth_generation = 0
    app._discord_managed_auth_cancelled = False
    app._discord_managed_auth_task_handle = None
    snackbar_calls: list[tuple[str, object]] = []
    dashboard_translation_calls: list[bool] = []
    start_entered = asyncio.Event()
    release_start = asyncio.Event()
    hub = SimpleNamespace(llm=object(), translation_enabled=True)
    app.view_dashboard = SimpleNamespace(
        set_translation_enabled=lambda enabled: dashboard_translation_calls.append(enabled)
    )
    app._show_snackbar = lambda message, color: snackbar_calls.append((message, color))

    async def fake_start_discord_managed_auth_from_dialog(**_kwargs) -> bool:
        start_entered.set()
        await release_start.wait()
        return True

    async def fake_set_translation_enabled(enabled: bool) -> bool:
        hub.translation_enabled = enabled
        return True

    controller = SimpleNamespace(
        hub=hub,
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._start_discord_managed_auth()
    auth_task = asyncio.create_task(app.page.tasks[0]())
    await start_entered.wait()

    await app.close_oauth_runtime()
    release_start.set()
    await auth_task

    assert app.application.managed_auth_tasks_open() is False
    assert app._discord_managed_auth_cancelled is True
    assert app._discord_managed_auth_task_handle is None
    assert snackbar_calls == []
    assert dashboard_translation_calls == []
    assert dialog.close_calls == 0


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

    controller = SimpleNamespace(
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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
    app._ui_application = compose_test_ui_application_boundary(controller)

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
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    controller = SimpleNamespace(
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    controller = SimpleNamespace(
        hub=SimpleNamespace(llm=None, translation_enabled=False),
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    controller = SimpleNamespace(
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._start_discord_managed_auth()
    await app.page.tasks[0]()

    assert snackbar_calls == []
    assert enable_calls == []


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

    controller = SimpleNamespace(
        start_discord_managed_auth_from_dialog=fake_start_discord_managed_auth_from_dialog,
        set_translation_enabled=fake_set_translation_enabled,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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
    controller = SimpleNamespace()
    app._ui_application = compose_test_ui_application_boundary(controller)

    app.show_discord_managed_auth_dialog(preview=False)
    dialog = app._discord_managed_auth_dialog
    dialog.set_waiting()

    assert dialog._reopen_browser_button is None
    assert [control.content for control in dialog._actions.controls] == [
        app_module.t("discord_auth.cancel")
    ]


def _peer_translation_binding_app(
    *,
    eula_accepted: bool,
):
    app = TranslatorApp.__new__(TranslatorApp)
    app._log_basic = lambda *_args, **_kwargs: None
    app._log_detailed = lambda *_args, **_kwargs: None
    tasks: list[object] = []
    shown: list[object] = []
    enabled: list[bool] = []
    accepted: list[bool] = []

    async def set_peer_translation_enabled(value: bool) -> None:
        enabled.append(value)

    async def accept_peer_translation_eula_and_enable() -> None:
        accepted.append(True)

    app._run_page_task = lambda coroutine, *args: tasks.append(coroutine)
    app._show_peer_translation_eula = shown.append
    app._ui_application = SimpleNamespace(
        state=lambda: SimpleNamespace(peer_translation_eula_accepted=eula_accepted),
        set_peer_translation_enabled=set_peer_translation_enabled,
        accept_peer_translation_eula_and_enable=accept_peer_translation_eula_and_enable,
    )
    return app, tasks, shown, enabled, accepted


def test_peer_translation_toggle_first_enable_opens_eula_without_running_task() -> None:
    app, tasks, shown, enabled, _accepted = _peer_translation_binding_app(eula_accepted=False)

    app._on_peer_translation_toggle(True)

    assert shown == [app._accept_peer_translation_eula_and_enable]
    assert tasks == []
    assert enabled == []


def test_peer_translation_accept_binding_runs_application_port() -> None:
    app, tasks, _shown, _enabled, accepted = _peer_translation_binding_app(eula_accepted=False)

    app._accept_peer_translation_eula_and_enable()
    assert len(tasks) == 1
    asyncio.run(tasks.pop(0)())

    assert accepted == [True]


def test_peer_translation_toggle_with_existing_acceptance_enables_directly() -> None:
    app, tasks, shown, enabled, _accepted = _peer_translation_binding_app(eula_accepted=True)

    app._on_peer_translation_toggle(True)
    assert shown == []
    assert len(tasks) == 1
    asyncio.run(tasks.pop(0)())

    assert enabled == [True]


def test_peer_translation_disable_does_not_open_eula() -> None:
    app, tasks, shown, enabled, _accepted = _peer_translation_binding_app(eula_accepted=False)

    app._on_peer_translation_toggle(False)
    assert shown == []
    assert len(tasks) == 1
    asyncio.run(tasks.pop(0)())

    assert enabled == [False]


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

    controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        apply_settings=fake_apply_settings,
        apply_providers=lambda _settings=None: asyncio.sleep(0),
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_nav_change(0)

    assert app.content_area.content is app.view_dashboard
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert events == [("apply", pending_settings)]


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
    auto_installs: list[str] = []

    async def fake_apply_providers(settings) -> bool:
        seen.append(settings)
        return True

    async def fake_auto_install() -> None:
        auto_installs.append("started")

    controller = SimpleNamespace(
        apply_providers=fake_apply_providers,
        install_selected_gpu_model_if_needed=fake_auto_install,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_nav_change(0)
    assert app.content_area.content is app.view_dashboard
    assert app.view_settings.has_provider_changes is False
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert seen == ["merged-settings"]
    assert len(app.page.tasks) == 1
    assert auto_installs == []


@pytest.mark.asyncio
async def test_on_nav_change_does_not_auto_install_when_provider_apply_fails() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._current_tab = 1
    app.view_dashboard = object()
    app.view_logs = SimpleNamespace(scroll_to_bottom=lambda: asyncio.sleep(0))
    app.view_about = object()
    app.view_settings = SimpleNamespace(
        has_provider_changes=True,
        consume_provider_apply_settings=lambda: "rejected-settings",
        refresh_prompt_if_empty=lambda: None,
    )
    app.content_area = DummyContent()
    auto_install = AsyncMock()

    async def fake_apply_providers(_settings) -> bool:
        return False

    controller = SimpleNamespace(
        apply_providers=fake_apply_providers,
        install_selected_gpu_model_if_needed=auto_install,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_nav_change(0)
    await app.page.tasks[0]()

    assert len(app.page.tasks) == 1
    auto_install.assert_not_awaited()


def test_gpu_provider_selection_alone_does_not_start_install() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._current_tab = 1
    app.view_dashboard = object()
    app.view_logs = SimpleNamespace(scroll_to_bottom=lambda: asyncio.sleep(0))
    app.view_about = object()
    app.view_settings = SimpleNamespace(
        has_provider_changes=True,
        refresh_prompt_if_empty=lambda: None,
    )
    app.content_area = DummyContent()
    controller = SimpleNamespace(
        apply_providers=AsyncMock(),
        install_selected_gpu_model_if_needed=AsyncMock(),
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_nav_change(1)

    assert app.page.tasks == []
    assert app.view_settings.has_provider_changes is True


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

    controller = SimpleNamespace(apply_providers=fake_apply_providers)
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_providers_changed()

    assert app.view_settings.has_provider_changes is False
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert seen == ["managed-settings"]


@pytest.mark.asyncio
async def test_on_providers_changed_runtime_only_reload_does_not_consume_provider_draft() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app._settings_mutation_queue = []
    app._settings_mutation_worker_active = False
    consumed = {"draft": False, "reload": True}
    app.view_settings = SimpleNamespace(
        has_provider_changes=True,
        consume_http_extension_runtime_reload=lambda: consumed.__setitem__("reload", False) or True,
        consume_provider_apply_settings=lambda: consumed.__setitem__("draft", True),
    )
    seen: list[tuple[bool, bool]] = []

    async def fake_apply_providers(
        settings=None,
        *,
        force_rebuild_llm: bool = False,
        persist_settings: bool = True,
        refresh_ui: bool = True,
    ) -> None:
        assert settings is None
        assert force_rebuild_llm is False
        seen.append((persist_settings, refresh_ui))

    controller = SimpleNamespace(apply_providers=fake_apply_providers)
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_providers_changed()
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()

    assert seen == [(False, False)]
    assert consumed == {"draft": False, "reload": False}


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
    auto_installs = {"count": 0}

    async def fake_auto_install() -> None:
        auto_installs["count"] += 1

    controller = SimpleNamespace(
        apply_providers=lambda _settings=None: asyncio.sleep(0),
        install_selected_gpu_model_if_needed=fake_auto_install,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_nav_change(1)
    assert app.content_area.content is app.view_settings
    assert refreshed["count"] == 1

    app._on_nav_change(2)
    assert app.content_area.content is app.view_logs
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert auto_installs["count"] == 0
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

    controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        apply_settings=fake_apply_settings,
        apply_providers=lambda _settings=None: asyncio.sleep(0),
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_nav_change(0)

    assert app.content_area.content is app.view_dashboard
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert merge_calls == []
    assert seen == [pending_settings]


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

    controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        apply_settings=fake_apply_settings,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_prompt_apply_settings(pending_settings)

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert events == [("apply", pending_settings)]


@pytest.mark.asyncio
async def test_prompt_apply_keeps_dashboard_target_for_next_request() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    pending_settings = PromptApplyIntent("new prompt")
    current_settings = _vnext(target_language="ja")
    applied_targets: list[str] = []

    async def fake_apply_settings(settings) -> None:
        applied_targets.append(settings.intent.languages.target_language)

    controller = SimpleNamespace(
        settings=current_settings,
        apply_settings=fake_apply_settings,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_prompt_apply_settings(pending_settings)

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert applied_targets == ["ja"]


@pytest.mark.asyncio
async def test_on_settings_changed_captures_patch_before_queued_apply() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    raw_settings = object()
    captured_change = object()
    merged_settings = object()
    seen: list[object] = []

    def fake_merge_settings(_settings) -> object:
        raise AssertionError("prompt merge should not run for generic settings changes")

    def fake_capture_settings_view_change(settings) -> object:
        assert settings is raw_settings
        return captured_change

    def fake_merge_settings_view_change(change) -> object:
        assert change is captured_change
        return merged_settings

    async def fake_apply_settings(settings) -> None:
        seen.append(settings)

    controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        capture_settings_view_change=fake_capture_settings_view_change,
        merge_settings_view_change_with_current=fake_merge_settings_view_change,
        apply_settings=fake_apply_settings,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=lambda: None,
        microphone_test_active=True,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    controller = SimpleNamespace(start_microphone_test=fake_start_microphone_test)
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_start_microphone_test()

    assert len(app.page.tasks) == 1
    assert calls == []
    await app.page.tasks[0]()
    assert calls == ["start"]
    assert len(app.page.opened) == 1


@pytest.mark.asyncio
async def test_start_microphone_test_false_opens_failure_modal() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    app.view_settings = InlineMicrophoneTestSettingsView()

    async def fake_start_microphone_test(**_kwargs) -> bool:
        return False

    controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    controller = SimpleNamespace(start_microphone_test=fake_start_microphone_test)
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=lambda: None,
        microphone_test_active=True,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=fake_stop_microphone_test,
        microphone_test_active=True,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=fake_stop_microphone_test,
        microphone_test_active=True,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    controller = SimpleNamespace(
        start_microphone_test=fake_start_microphone_test,
        stop_microphone_test=fake_stop_microphone_test,
        microphone_test_active=True,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_start_microphone_test()
    await app.page.tasks[0]()

    app._on_settings_changed("audio")
    await app.page.tasks[1]()

    assert events == ["start", "apply:audio"]
    assert len(app.page.opened) == 1
    assert app.page.closed == [app.page.opened[0]]


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

    controller = SimpleNamespace(
        apply_settings=fake_apply_settings,
        start_microphone_test=fake_start_microphone_test,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_settings_changed("audio")
    app._on_start_microphone_test()

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert events == ["apply:audio", "start"]


def test_on_request_openrouter_pkce_uses_settings_mutation_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_app_construction(monkeypatch)
    app = TranslatorApp(
        DummyPage(),
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )
    target_settings = AppSettingsVNext()
    queued: list[object] = []
    monkeypatch.setattr(app, "_queue_settings_mutation_task", queued.append)

    app._on_request_openrouter_pkce(target_settings, launch_source="settings")

    assert len(queued) == 1


def test_on_request_openrouter_pkce_reopens_existing_auth_url_while_flow_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    target_settings = AppSettingsVNext()
    reopen_calls: list[str] = []

    async def fake_connect_openrouter_via_pkce(
        *, target: AppSettingsVNext, launch_source: str
    ) -> bool:
        _ = (target, launch_source)
        return False

    controller = SimpleNamespace(
        connect_openrouter_via_pkce=fake_connect_openrouter_via_pkce,
        reopen_openrouter_pkce_authorization_url=lambda: reopen_calls.append("reopen") or True,
        settings=AppSettingsVNext(),
        config_path=Path("settings.json"),
    )
    app._ui_application = compose_test_ui_application_boundary(controller)
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
    target_settings = AppSettingsVNext()
    pkce_calls: list[str] = []

    async def fake_connect_openrouter_via_pkce(
        *, target: AppSettingsVNext, launch_source: str
    ) -> bool:
        _ = target
        pkce_calls.append(launch_source)
        return False

    controller = SimpleNamespace(
        connect_openrouter_via_pkce=fake_connect_openrouter_via_pkce,
        reopen_openrouter_pkce_authorization_url=lambda: False,
        settings=AppSettingsVNext(),
        config_path=Path("settings.json"),
    )
    app._ui_application = compose_test_ui_application_boundary(controller)
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
    target_settings = AppSettingsVNext()
    updated_settings = AppSettingsVNext()
    pkce_calls: list[tuple[AppSettingsVNext, str]] = []
    refresh_calls: list[tuple[AppSettingsVNext, Path]] = []
    snackbar_calls: list[tuple[str, str]] = []

    async def fake_connect_openrouter_via_pkce(
        *, target: AppSettingsVNext, launch_source: str
    ) -> bool:
        pkce_calls.append((target, launch_source))
        return True

    controller = SimpleNamespace(
        connect_openrouter_via_pkce=fake_connect_openrouter_via_pkce,
        settings=updated_settings,
        config_path=Path("settings.json"),
        refresh_settings_after_openrouter_pkce_success=lambda: (
            refresh_calls.append((updated_settings, Path("settings.json"))) or True
        ),
    )
    app._ui_application = compose_test_ui_application_boundary(controller)
    app._show_snackbar = lambda message, bgcolor: snackbar_calls.append((message, bgcolor))
    queued: list[object] = []
    monkeypatch.setattr(app, "_queue_settings_mutation_task", queued.append)

    app._on_request_openrouter_pkce(target_settings, launch_source="settings")

    assert len(queued) == 1
    await queued[0]()
    assert pkce_calls == [(target_settings, "settings")]
    assert refresh_calls == [(updated_settings, Path("settings.json"))]
    assert snackbar_calls == [(app_module.t("openrouter.pkce.connected"), app_module.COLOR_SUCCESS)]


@pytest.mark.asyncio
async def test_on_request_openrouter_pkce_does_not_refresh_settings_view_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    target_settings = AppSettingsVNext()
    refresh_calls: list[tuple[AppSettingsVNext, Path]] = []

    async def fake_connect_openrouter_via_pkce(
        *, target: AppSettingsVNext, launch_source: str
    ) -> bool:
        _ = (target, launch_source)
        return False

    controller = SimpleNamespace(
        connect_openrouter_via_pkce=fake_connect_openrouter_via_pkce,
        settings=AppSettingsVNext(),
        config_path=Path("settings.json"),
        refresh_settings_after_openrouter_pkce_success=lambda: (
            refresh_calls.append((AppSettingsVNext(), Path("settings.json"))) or True
        ),
    )
    app._ui_application = compose_test_ui_application_boundary(controller)
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

    controller = SimpleNamespace(
        merge_settings_tab_apply_with_current_languages=fake_merge_settings,
        apply_settings=fake_apply_settings,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_settings_changed(raw_settings)
    app._on_prompt_apply_settings(pending_settings)

    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()

    assert events == [
        ("apply", raw_settings),
        ("apply", pending_settings),
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

    async def fake_apply_providers(settings) -> bool:
        events.append(("providers", settings))
        return True

    controller = SimpleNamespace(
        apply_settings=fake_apply_settings,
        apply_providers=fake_apply_providers,
        install_selected_gpu_model_if_needed=AsyncMock(),
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    def fake_pop_dialog():
        control = app.page.dialog
        events.append(("close", control))
        app.page.closed.append(control)
        app.page.dialog = None
        return control

    app.page.pop_dialog = fake_pop_dialog
    app._current_tab = 0
    app.view_dashboard = object()
    app.view_settings = SimpleNamespace(
        has_provider_changes=False,
        refresh_prompt_if_empty=lambda: None,
    )
    app.view_logs = SimpleNamespace(scroll_to_bottom=lambda: asyncio.sleep(0))
    app.view_about = object()
    app.content_area = RecordingContent(app.view_dashboard)
    controller = SimpleNamespace(apply_providers=lambda _settings=None: asyncio.sleep(0))
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_nav_change(1)

    assert events[:3] == [
        ("close", dialog),
        ("content", app.view_settings),
        ("update", app.view_settings),
    ]
    assert app.page.closed == [dialog]


def test_apply_locale_updates_views_and_page(monkeypatch: pytest.MonkeyPatch) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app._ui_application = compose_test_ui_application_boundary(None)
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
    app._ui_application = compose_test_ui_application_boundary(None)
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
    state = OverlayPeerPresentationState(
        overlay_intent_enabled=True,
        overlay_state="failed",
        overlay_failure_reason="runtime_crashed",
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="overlay_failed",
        peer_activation_starting=False,
    )
    seen: list[tuple[str, str | None, str | None, bool | None]] = []
    refreshed: list[object] = []
    controller = SimpleNamespace(overlay_peer_presentation_state=lambda: state)
    app._ui_application = compose_test_ui_application_boundary(controller)
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
    assert refreshed == [
        ("settings", app.overlay_peer_contract),
        ("dashboard", app.overlay_peer_contract),
    ]


@pytest.mark.asyncio
async def test_debug_preview_submit_toggle_and_settings_wrappers_schedule_controller_tasks() -> (
    None
):
    app = TranslatorApp.__new__(TranslatorApp)
    app.debug_ui_preview = True
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

    def fake_manual_activity(has_text: bool) -> None:
        seen.append(("manual_activity", has_text))

    controller = SimpleNamespace(
        submit_text=fake_submit,
        set_manual_input_activity=fake_manual_activity,
        set_translation_enabled=fake_translation,
        set_stt_enabled=fake_stt,
        set_overlay_enabled=fake_overlay,
        set_peer_translation_enabled=fake_peer,
        apply_settings=fake_apply_settings,
        apply_providers=fake_apply_providers,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_manual_submit("You", "hello")
    app._on_message_input_activity(True)
    app._on_translation_toggle(True)
    app._on_stt_toggle(False)
    app._on_overlay_toggle(True)
    app._on_peer_translation_toggle(True)
    app._on_settings_changed("settings")
    app._on_providers_changed()

    assert len(app.page.tasks) == 7
    for task_fn in app.page.tasks:
        await task_fn()

    assert seen == [
        ("submit", "hello"),
        ("manual_activity", True),
        ("translation", True),
        ("stt", False),
        ("overlay", True),
        ("peer", True),
        ("apply_settings", "settings"),
        ("apply_providers", True),
    ]


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

    settings = _vnext(llm="local_llm")
    controller = SimpleNamespace(settings=settings, apply_providers=fake_apply_providers)
    app._ui_application = compose_test_ui_application_boundary(controller)

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

    settings = _vnext(llm="gemini")
    controller = SimpleNamespace(settings=settings, apply_providers=fake_apply_providers)
    app._ui_application = compose_test_ui_application_boundary(controller)

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
    app._ui_application = compose_test_ui_application_boundary(controller)

    app._on_translation_toggle(True)
    app._on_stt_toggle(False)
    app._on_overlay_toggle(True)

    assert controller.basic_messages == [
        "[Dashboard] Translation toggle requested: enabled=True",
        "[Dashboard] STT toggle requested: enabled=False",
        "[Dashboard] Overlay toggle requested: enabled=True",
    ]
    assert controller.detailed_messages == [
        "[Dashboard] Translation toggle detail: dashboard_state=False overlay_state=connected",
        "[Dashboard] STT toggle detail: dashboard_state=True overlay_state=connected",
        "[Dashboard] Overlay toggle detail: overlay_state=connected failure_reason=runtime_crashed",
    ]


def test_on_overlay_state_changed_routes_runtime_logs() -> None:
    controller = RuntimeLoggingController()
    app = TranslatorApp.__new__(TranslatorApp)
    app._ui_application = compose_test_ui_application_boundary(controller)
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
    settings = AppSettingsVNext()
    seen = []

    async def fake_on_dashboard_language_change(change) -> None:
        seen.append(change)

    warning = SimpleNamespace(key="dashboard.warn_stt_key", language_code="ko")
    monkeypatch.setattr(
        app_module, "get_stt_compatibility_warning", lambda *_args, **_kwargs: warning
    )
    controller = SimpleNamespace(
        settings=settings,
        on_dashboard_language_change=fake_on_dashboard_language_change,
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    change = app_module.LanguageSelectionChange(
        source_code="ja",
        target_code="fr",
        peer_source_code="",
        peer_target_code="it",
        peer_source_mode="manual",
        recent_source_codes=("ja",),
        recent_target_codes=("fr",),
    )
    app._on_language_change(change)

    assert settings.intent.languages.source_language == "ko"
    assert settings.intent.languages.target_language == "en"
    assert len(app.page.opened) == 1
    assert len(app.page.tasks) == 1
    await app.page.tasks[0]()
    assert seen == [change]


@pytest.mark.asyncio
async def test_on_language_change_forwards_automatic_peer_mode(
    monkeypatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    settings = AppSettingsVNext()
    seen = []

    async def callback(change) -> None:
        seen.append(change)

    monkeypatch.setattr(app_module, "get_stt_compatibility_warning", lambda *_args: None)
    controller = SimpleNamespace(settings=settings, on_dashboard_language_change=callback)
    app._ui_application = compose_test_ui_application_boundary(controller)

    change = app_module.LanguageSelectionChange(
        source_code="ko",
        target_code="en",
        peer_source_code="ja",
        peer_target_code="fr",
        peer_source_mode="auto",
        recent_source_codes=(),
        recent_target_codes=(),
    )
    app._on_language_change(change)
    await app.page.tasks[0]()

    assert seen == [change]


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
        set_translation_needs_key=lambda value, update_ui=False: (
            app.view_dashboard.trans_calls.append((value, update_ui))
        ),
    )

    async def fake_verify(provider: str, key: str):
        _ = key
        return provider in {"deepgram", "deepseek"}, "ok"

    settings = SimpleNamespace(
        api_key_verified=SimpleNamespace(
            deepgram=False,
            soniox=False,
            google=False,
            openrouter=False,
            deepseek=False,
            alibaba_beijing=False,
            alibaba_singapore=False,
        )
    )
    saves: list[tuple[str, str, bool]] = []

    def persist_verification(provider: str, key: str, success: bool) -> None:
        setattr(settings.api_key_verified, provider, success)
        saves.append((provider, key, success))

    controller = SimpleNamespace(
        verify_api_key=fake_verify,
        persist_api_key_verification=persist_verification,
        settings=settings,
        config_path="settings.json",
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    deepgram_result = await app._on_verify_api_key("deepgram", "k")
    google_result = await app._on_verify_api_key("google", "k")
    openrouter_result = await app._on_verify_api_key("openrouter", "k")
    deepseek_result = await app._on_verify_api_key("deepseek", "k")

    assert deepgram_result == (True, "ok")
    assert google_result == (False, "ok")
    assert openrouter_result == (False, "ok")
    assert deepseek_result == (True, "ok")
    assert settings.api_key_verified.deepgram is True
    assert settings.api_key_verified.google is False
    assert settings.api_key_verified.openrouter is False
    assert settings.api_key_verified.deepseek is True
    assert app.view_dashboard.stt_calls[-1] == (False, False)
    assert app.view_dashboard.trans_calls[-1] == (False, False)
    assert len(saves) == 4


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
        set_translation_needs_key=lambda value, update_ui=False: (
            app.view_dashboard.trans_calls.append((value, update_ui))
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
    saves: list[tuple[str, str, bool]] = []
    controller = SimpleNamespace(
        verify_api_key=fake_verify,
        persist_api_key_verification=lambda provider, key, success: saves.append(
            (provider, key, success)
        ),
        settings=settings,
        config_path="settings.json",
    )
    app._ui_application = compose_test_ui_application_boundary(controller)

    result = await app._on_verify_api_key("google", "old-key")

    assert result == (True, "ok")
    assert settings.api_key_verified.google is False
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
    captured: dict[str, object] = {}
    pkce_calls: list[tuple[AppSettingsVNext, str]] = []
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
    captured: dict[str, object] = {}
    pkce_calls: list[tuple[AppSettingsVNext, str]] = []

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

    await _check_and_notify_update(page, load_update_info=no_update)
    assert page.opened == []

    update_info = SimpleNamespace(version="9.9.9", download_url="https://example.com")

    async def has_update():
        return update_info

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
    await _check_and_notify_update(page, load_update_info=has_update)

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
    controller = RuntimeLoggingController()
    app._ui_application = compose_test_ui_application_boundary(controller)

    async def raise_error():
        raise RuntimeError("network down")

    await _check_and_notify_update(
        page,
        log_detailed=app._log_detailed,
        load_update_info=raise_error,
    )
    assert page.opened == []
    assert controller.basic_messages == []
    assert controller.detailed_messages == ["[Update] Check notification failed: network down"]


@pytest.mark.asyncio
async def test_shutdown_is_idempotent_and_cancels_tracked_page_tasks() -> None:
    stop_calls = 0
    background_cancelled = asyncio.Event()

    class Controller:
        async def stop(self) -> None:
            nonlocal stop_calls
            stop_calls += 1
            await asyncio.sleep(0)

    async def background() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            background_cancelled.set()

    app = TranslatorApp.__new__(TranslatorApp)
    app.page = DummyPage()
    controller = Controller()
    app._ui_application = _application_boundary_with_stop(controller)
    app._shutdown_lock = None
    app._shutdown_complete = False
    app._shutting_down = False
    app._settings_mutation_queue = [object()]
    foundation_runtime = app._ensure_foundation_runtime()
    foundation_runtime._tracked_tasks = {asyncio.create_task(background())}

    await asyncio.gather(app.shutdown(), app.shutdown())

    assert stop_calls == 1
    assert background_cancelled.is_set()
    assert foundation_runtime.snapshot.tracked_task_count == 0
    assert app._settings_mutation_queue == []


@pytest.mark.asyncio
async def test_main_gui_registers_awaited_idempotent_page_lifecycle_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = DummyPage()
    lifecycle_events: list[str] = []

    class Controller:
        async def start(self) -> None:
            lifecycle_events.append("started")

        async def stop(self) -> None:
            lifecycle_events.append("stop_started")
            await asyncio.sleep(0)
            lifecycle_events.append("stop_completed")

    def fake_init(
        self,
        incoming_page,
        *,
        config_path,
        application_factory,
        debug_ui_preview=False,
        runtime_logging_sinks=None,
        vrchat_osc_presence=None,
    ) -> None:
        _ = (
            config_path,
            application_factory,
            debug_ui_preview,
            runtime_logging_sinks,
            vrchat_osc_presence,
        )
        self.page = incoming_page
        backend = Controller()
        self._ui_application = _application_boundary_with_stop(backend)
        self._tracked_page_tasks = set()
        self._shutdown_lock = None
        self._shutdown_complete = False
        self._shutting_down = False
        self._settings_mutation_queue = []
        self.schedule_after_launch_tasks = lambda: None

    monkeypatch.setattr(TranslatorApp, "__init__", fake_init)

    await app_module.main_gui(
        page,
        config_path=Path("settings.json"),
        application_factory=_construction_application_factory,
    )

    assert asyncio.iscoroutinefunction(page.on_close)
    assert page.on_close == page.on_disconnect

    await asyncio.gather(page.on_close(None), page.on_disconnect(None))

    assert lifecycle_events == ["started", "stop_started", "stop_completed"]
    assert page.tasks == []
