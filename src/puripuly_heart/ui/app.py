import asyncio
import contextlib
import inspect
import json
import logging
import tempfile
import webbrowser
from pathlib import Path

import flet as ft
from puripuly_heart.core.discord_oauth_loopback import (
    render_discord_oauth_callback_completion_page,
)
from puripuly_heart.core.managed_openrouter_release import TalkTogetherPassStatus

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.ui_application import (
    UiApplicationFactoryPort,
    UiApplicationPort,
)
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownCallback,
    ApplicationShutdownCoordinator,
    application_shutdown_callback,
)
from puripuly_heart.core.language import get_stt_compatibility_warning
from puripuly_heart.core.lifecycle import (
    SHUTDOWN_PHASE_FREEZE_INGRESS,
    SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
)
from puripuly_heart.ui.components.bottom_nav import BottomNavBar
from puripuly_heart.ui.components.debug_preview_panel import DebugPreviewPanel
from puripuly_heart.ui.components.discord_managed_auth_dialog import DiscordManagedAuthDialog
from puripuly_heart.ui.components.founder_letter_dialog import FounderLetterDialog
from puripuly_heart.ui.components.local_qwen_hallucination_dialog import (
    LocalQwenHallucinationDialog,
)
from puripuly_heart.ui.components.microphone_test_dialog import MicrophoneTestDialog
from puripuly_heart.ui.components.peer_translation_eula_dialog import PeerTranslationEulaDialog
from puripuly_heart.ui.components.qq_managed_auth_dialog import QqManagedAuthDialog
from puripuly_heart.ui.components.telemetry_consent_dialog import TelemetryConsentDialog
from puripuly_heart.ui.components.title_bar import TitleBar
from puripuly_heart.ui.dashboard.contract import (
    DashboardCaptureIntents,
    DashboardTranslationIntents,
)
from puripuly_heart.ui.fonts import font_for_language, register_fonts
from puripuly_heart.ui.foundation.adapter import FletFoundationAdapter
from puripuly_heart.ui.foundation.preview import FoundationPreviewSurface
from puripuly_heart.ui.foundation.resources import DEFAULT_FOUNDATION_RESOURCES
from puripuly_heart.ui.foundation.runtime import FletFoundationRuntime
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS
from puripuly_heart.ui.gpu_device import GpuDeviceOption
from puripuly_heart.ui.gpu_notice import GpuDashboardNotice
from puripuly_heart.ui.i18n import (
    get_locale,
    language_name,
    t,
)
from puripuly_heart.ui.logs.contract import LogsIntents
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from puripuly_heart.ui.settings.contract import (
    SettingsGeneralIntents,
    SettingsOverlayIntents,
    SettingsPromptIntents,
    SettingsProviderIntents,
    SettingsSurfaceIntents,
)
from puripuly_heart.ui.shell.contract import AppShellSlots
from puripuly_heart.ui.shell.renderer import compose_app_shell
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    get_app_theme,
)
from puripuly_heart.ui.views.about import AboutView
from puripuly_heart.ui.views.dashboard import DashboardView
from puripuly_heart.ui.views.logs import LogsView
from puripuly_heart.ui.views.settings import SettingsView

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_WIDTH = FOUNDATION_DESIGN_TOKENS.window.width
DEFAULT_WINDOW_HEIGHT = FOUNDATION_DESIGN_TOKENS.window.height
APP_CONTENT_PADDING = FOUNDATION_DESIGN_TOKENS.spacing.page
FOUNDER_CONTACT_URL = "https://x.com/kapitalismho"
FOUNDER_README_BASE_URL = "https://github.com/kapitalismho/PuriPuly-heart/blob/main"
FOUNDER_README_PATH_BY_LOCALE = {
    "ko": "README.ko.md",
    "zh-CN": "README.zh-CN.md",
    "ja": "README.ja.md",
}
FOUNDER_README_API_KEYS_ANCHOR_BY_LOCALE = {
    "ko": "자신의-api-키-사용하기",
    "zh-CN": "使用您自己的-api-密钥",
    "ja": "自分のapiキーを使う",
}
FOUNDER_README_DEFAULT_API_KEYS_ANCHOR = "using-your-own-api-keys"
DEBUG_PREVIEW_TALK_TOGETHER_PASS_ID = "7KQ9M2"
GITHUB_STAR_REPOSITORY_URL = "https://github.com/kapitalismho/PuriPuly-heart"
GITHUB_STAR_PROMPT_DELAY_S = 2.5
GITHUB_STAR_PROMPT_DURATION_MS = 8000


async def _prepare_and_show_main_window(page: ft.Page) -> None:
    try:
        page.update()

        wait_until_ready = getattr(page.window, "wait_until_ready_to_show", None)
        if callable(wait_until_ready):
            ready_result = wait_until_ready()
            if inspect.isawaitable(ready_result):
                await ready_result

        center_result = page.window.center()
        if inspect.isawaitable(center_result):
            await center_result
    except Exception:
        logger.warning(
            "Failed to center the main window before showing it",
            exc_info=True,
        )
    finally:
        page.window.visible = True
        page.update()


def _callable_accepts_keyword(callable_obj: object, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return True
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def founder_readme_url_for_locale(locale: str | None) -> str:
    readme_path = FOUNDER_README_PATH_BY_LOCALE.get(locale or "", "README.md")
    anchor = FOUNDER_README_API_KEYS_ANCHOR_BY_LOCALE.get(
        locale or "", FOUNDER_README_DEFAULT_API_KEYS_ANCHOR
    )
    return f"{FOUNDER_README_BASE_URL}/{readme_path}#{anchor}"


def _write_discord_callback_preview_page(locale: str | None) -> str:
    html = render_discord_oauth_callback_completion_page(locale)
    with tempfile.NamedTemporaryFile(
        "wb",
        prefix="puripuly-discord-callback-",
        suffix=".html",
        delete=False,
    ) as handle:
        handle.write(html)
        path = Path(handle.name)
    return path.as_uri()


class TranslatorApp:
    def __init__(
        self,
        page: ft.Page,
        *,
        config_path,
        application_factory: UiApplicationFactoryPort,
        debug_ui_preview: bool = False,
        allow_stable_settings_import: bool = False,
        runtime_logging_sinks=None,
        vrchat_osc_presence=None,
    ):
        self.page = page
        self._presentation_adapter = FletUiPresentationAdapter(self)
        application = application_factory(
            presentation=self._presentation_adapter,
            config_path=config_path,
            allow_stable_settings_import=allow_stable_settings_import,
            runtime_logging_sinks=runtime_logging_sinks,
            vrchat_osc_presence=vrchat_osc_presence,
        )
        if application is None:
            raise RuntimeError("Application factory did not compose an application boundary")
        self._ui_application = application
        self._shutdown_lock: asyncio.Lock | None = None
        self._shutdown_complete = False
        self._shutting_down = False
        self._window_close_requested = False
        self.overlay_state = "off"
        self.overlay_failure_reason: str | None = None
        self.overlay_peer_contract = None
        self.debug_ui_preview = bool(debug_ui_preview)
        self.debug_preview_panel: DebugPreviewPanel | None = None
        self._openrouter_pkce_request_active = False
        self._discord_managed_auth_generation = 0
        self._discord_managed_auth_cancelled = False
        self._discord_managed_auth_task_handle = None
        self._qq_managed_auth_generation = 0
        self._qq_managed_auth_cancelled = False
        self._qq_managed_auth_task_handle = None
        self._github_star_prompt_launch_pending = True
        self._after_launch_task_handle = None
        self._launch_high_priority_feedback_shown = False
        self._launch_high_priority_feedback_reason: str | None = None
        self._launch_high_priority_snackbar = None
        self._github_star_prompt_shown_this_launch = False
        self._microphone_test_dialog: MicrophoneTestDialog | None = None
        self._telemetry_consent_dialog: TelemetryConsentDialog | None = None
        self._foundation_preview_dialog: ft.AlertDialog | None = None
        self._foundation_adapter = FletFoundationAdapter(
            self.application,
            self._presentation_adapter,
        )
        self._foundation_runtime = FletFoundationRuntime(
            self.page,
            self._foundation_adapter,
        )
        self._application_lifecycle = self._compose_application_lifecycle()
        self._setup_page()
        self._build_layout()

        # Link Dashboard callbacks
        self.view_dashboard.bind_dashboard_intents(
            translation=DashboardTranslationIntents(
                submit_message=self._on_manual_submit,
                toggle_translation=self._on_translation_toggle,
                change_language=self._on_language_change,
                report_input_activity=self._on_message_input_activity,
            ),
            capture=DashboardCaptureIntents(
                toggle_self_capture=self._on_stt_toggle,
                toggle_peer_capture=self._on_peer_translation_toggle,
                toggle_overlay=self._on_overlay_toggle,
                retry_peer_process_capture=self._on_retry_peer_process_capture,
                run_gpu_notice_action=self.application.handle_gpu_notice_action,
            ),
        )

        runtime_log_basic = self.application.log_basic
        runtime_log_detailed = self.application.log_detailed
        calibration_begin = self.application.begin_overlay_calibration
        calibration_change = self.application.set_overlay_calibration_field
        calibration_apply = self.application.apply_overlay_calibration
        calibration_cancel = self.application.cancel_overlay_calibration
        self.view_settings.bind_settings_intents(
            surface=SettingsSurfaceIntents(
                settings_changed=self._on_settings_changed,
                show_snackbar=self._show_snackbar,
                runtime_log_basic=(runtime_log_basic if callable(runtime_log_basic) else None),
                runtime_log_detailed=(
                    runtime_log_detailed if callable(runtime_log_detailed) else None
                ),
            ),
            provider=SettingsProviderIntents(
                providers_changed=self._on_providers_changed,
                request_openrouter_pkce=self._on_request_openrouter_pkce,
                verify_api_key=self._on_verify_api_key,
                provider_secret_change=self._on_provider_secret_change,
                secret_cleared=self._on_secret_cleared,
                local_llm_secret_changed=self._on_local_llm_secret_changed,
                custom_stt_secret_changed=self._on_custom_stt_secret_changed,
                gpu_discovery_requested=self._on_gpu_discovery_requested,
            ),
            general=SettingsGeneralIntents(
                start_microphone_test=self._on_start_microphone_test,
                telemetry_consent_change=self._on_telemetry_consent_change,
                list_loopback_capture_options=(
                    lambda: self.application.list_loopback_capture_options()
                ),
                list_loopback_process_options=(
                    lambda: self.application.list_loopback_process_options()
                ),
                list_loopback_device_options=(
                    lambda: self.application.list_loopback_device_options()
                ),
                current_loopback_capture_option=(
                    lambda: self.application.current_loopback_capture_option_value()
                ),
                apply_loopback_capture_option=self._on_apply_loopback_capture_option,
                loopback_capture_summary=(lambda: self.application.loopback_capture_summary()),
                osc_effective_ports=self._effective_osc_ports,
            ),
            prompt=SettingsPromptIntents(
                prompt_apply_settings=self._on_prompt_apply_settings,
            ),
            overlay=SettingsOverlayIntents(
                desktop_overlay_lock_change=self._on_desktop_overlay_lock_change,
                desktop_overlay_size_change=self._on_desktop_overlay_size_change,
                desktop_overlay_recovery_action=self._on_desktop_overlay_recovery_action,
                desktop_overlay_position_reset=self._on_desktop_overlay_position_reset,
                view_logs=self._open_logs_tab,
                calibration_begin=(calibration_begin if callable(calibration_begin) else None),
                calibration_change=(calibration_change if callable(calibration_change) else None),
                calibration_apply=(calibration_apply if callable(calibration_apply) else None),
                calibration_cancel=(calibration_cancel if callable(calibration_cancel) else None),
            ),
        )
        self.view_logs.bind_logs_intents(
            LogsIntents(runtime_logging_mode_change=self._on_runtime_logging_mode_change)
        )
        self.view_logs.set_runtime_logging_mode(self.application.state().runtime_logging_mode)
        self.view_dashboard.runtime_log_detailed = self._log_detailed

        set_overlay_calibration = getattr(self.view_settings, "set_overlay_calibration", None)
        overlay_calibration = self.application.overlay_calibration
        if callable(set_overlay_calibration) and overlay_calibration is not None:
            set_overlay_calibration(overlay_calibration)

    @property
    def application(self) -> UiApplicationPort:
        return self._ui_application

    def _effective_osc_ports(self) -> tuple[int | None, int | None]:
        try:
            value = self.application.effective_osc_ports()
        except Exception:
            return (None, None)
        if not isinstance(value, tuple) or len(value) != 2:
            return (None, None)
        return (
            value[0] if isinstance(value[0], int) and value[0] > 0 else None,
            value[1] if isinstance(value[1], int) and value[1] > 0 else None,
        )

    def _run_page_task(self, coroutine, *args):
        if getattr(self, "_shutting_down", False):
            close = getattr(coroutine, "close", None)
            if callable(close):
                close()
            return None
        runtime = self._ensure_foundation_runtime()
        if not runtime.snapshot.lifecycle_bound:
            runtime.bind_application_lifecycle(self._get_application_lifecycle())
        return runtime.run_page_task(coroutine, *args)

    def _ensure_foundation_runtime(self) -> FletFoundationRuntime:
        runtime = getattr(self, "_foundation_runtime", None)
        if runtime is not None:
            return runtime
        presentation = getattr(self, "_presentation_adapter", None)
        if presentation is None:
            presentation = FletUiPresentationAdapter(self)
            self._presentation_adapter = presentation
        adapter = FletFoundationAdapter(self.application, presentation)
        runtime = FletFoundationRuntime(self.page, adapter)
        self._foundation_adapter = adapter
        self._foundation_runtime = runtime
        return runtime

    async def shutdown(self) -> None:
        lifecycle = self._get_application_lifecycle()
        try:
            await lifecycle.shutdown()
        finally:
            self._shutdown_complete = lifecycle.is_terminal

    def _compose_application_lifecycle(self) -> ApplicationShutdownCoordinator:
        foundation_runtime = self._ensure_foundation_runtime()
        self.application.register_application_shutdown_callbacks(
            self._application_shutdown_callbacks()
        )
        lifecycle = self.application.application_lifecycle()
        foundation_runtime.bind_application_lifecycle(lifecycle)
        return lifecycle

    def _get_application_lifecycle(self) -> ApplicationShutdownCoordinator:
        lifecycle = getattr(self, "_application_lifecycle", None)
        if lifecycle is None:
            lifecycle = self._compose_application_lifecycle()
            self._application_lifecycle = lifecycle
        return lifecycle

    def _application_shutdown_callbacks(self) -> tuple[ApplicationShutdownCallback, ...]:
        foundation_runtime = self._ensure_foundation_runtime()
        return (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
                owner_name="TranslatorApp",
                callback_name="freeze_ui_ingress",
                callback=self._freeze_ui_ingress,
            ),
            *foundation_runtime.application_shutdown_callbacks(),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="TranslatorApp",
                callback_name="close_after_launch_tasks",
                callback=self._close_after_launch_ui_tasks,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="TranslatorApp",
                callback_name="close_managed_auth_tasks",
                callback=self._close_managed_auth_ui_tasks,
            ),
        )

    def _freeze_ui_ingress(self) -> None:
        self._shutting_down = True
        self._settings_mutation_queue = []

    async def _on_page_lifecycle_end(self, _event=None) -> None:
        await self.shutdown()

    def _on_window_event(self, event) -> None:
        event_type = getattr(event, "type", getattr(event, "data", None))
        if event_type not in {ft.WindowEventType.CLOSE, ft.WindowEventType.CLOSE.value}:
            return
        self._request_window_close()

    def _request_window_close(self) -> None:
        if self._window_close_requested:
            return
        self._window_close_requested = True
        self._run_page_task(self._close_after_window_request)

    async def _close_after_window_request(self) -> None:
        try:
            await self.shutdown()
        finally:
            destroy_result = self.page.window.destroy()
            if inspect.isawaitable(destroy_result):
                await destroy_result

    def _setup_page(self):
        self.page.title = t("app.title")
        self.page.theme_mode = ft.ThemeMode.LIGHT
        register_fonts(self.page)
        self.page.theme = get_app_theme(font_family=font_for_language(get_locale()))
        self.page.bgcolor = COLOR_BACKGROUND
        self.page.padding = 0
        self.page.window.frameless = FOUNDATION_DESIGN_TOKENS.window.frameless
        self.page.window.resizable = FOUNDATION_DESIGN_TOKENS.window.resizable
        self.page.window.maximizable = FOUNDATION_DESIGN_TOKENS.window.maximizable
        self.page.window.width = DEFAULT_WINDOW_WIDTH
        self.page.window.height = DEFAULT_WINDOW_HEIGHT
        self.page.window.min_width = DEFAULT_WINDOW_WIDTH
        self.page.window.max_width = DEFAULT_WINDOW_WIDTH
        self.page.window.min_height = DEFAULT_WINDOW_HEIGHT
        self.page.window.max_height = DEFAULT_WINDOW_HEIGHT
        self.page.window.prevent_close = True
        self.page.window.on_event = self._on_window_event
        self.page.window.icon = DEFAULT_FOUNDATION_RESOURCES.asset_url(
            FOUNDATION_DESIGN_TOKENS.icon_asset
        )
        self.page.on_keyboard_event = self._on_keyboard_event

    def _build_layout(self):
        self.view_dashboard = DashboardView()
        self.view_settings = SettingsView()
        set_http_extension_registry = getattr(
            self.view_settings,
            "set_http_extension_registry",
            None,
        )
        http_extension_registry = self.application.http_extension_registry()
        if callable(set_http_extension_registry):
            set_http_extension_registry(http_extension_registry)
        self.view_logs = LogsView()
        self.view_about = AboutView()
        self.view_settings.set_overlay_runtime_state(self.overlay_state)

        # Custom title bar
        self.title_bar = TitleBar(self.page, on_close=self._request_window_close)

        # Bottom navigation (order: Home, Settings, Logs, About)
        self.bottom_nav = BottomNavBar(on_change=self._on_nav_change)

        self.debug_preview_panel = (
            self._build_debug_preview_panel() if self.debug_ui_preview else None
        )
        shell = compose_app_shell(
            AppShellSlots(
                title_bar=self.title_bar,
                content=self.view_dashboard,
                bottom_nav=self.bottom_nav,
                content_padding=APP_CONTENT_PADDING,
                debug_panel=self.debug_preview_panel,
            )
        )
        self.content_area = shell.content_area
        self.layout = shell.layout
        self.page.add(shell.root)

    def _build_debug_preview_panel(self) -> DebugPreviewPanel:
        return DebugPreviewPanel(
            on_brake_notice=self._preview_brake_notice,
            on_revoked_notice=self._preview_revoked_notice,
            on_founder_letter=self._preview_founder_letter,
            on_pkce_failure=self._preview_pkce_failure,
            on_discord_auth=self._preview_discord_auth,
            on_qq_auth=self._preview_qq_auth,
            on_qq_auth_recoverable_error=self._preview_qq_auth_recoverable_error,
            on_qq_auth_translation_gated=self._preview_qq_auth_translation_gated,
            on_discord_callback_page=self._preview_discord_callback_page,
            on_peer_translation_eula=self._preview_peer_translation_eula,
            on_local_qwen_hallucination_modal=self._preview_local_qwen_hallucination_modal,
            on_talk_together_pass_invite_progress=(
                self._preview_talk_together_pass_invite_progress
            ),
            on_capture_fault_cycle=self._preview_capture_fault_cycle,
            on_stt_fault_cycle=self._preview_stt_fault_cycle,
            on_audio_fault_clear=self._preview_audio_fault_clear,
            on_gpu_state_cycle=self._cycle_debug_preview_gpu_state,
            on_github_star_snackbar=self._preview_github_star_snackbar,
            on_telemetry_consent=self._preview_telemetry_consent,
            on_stt_loading_button_cycle=self._cycle_debug_preview_stt_loading_button,
            on_foundation_primitives=self._preview_foundation_primitives,
            on_http_extension_form=self._preview_http_extension_form,
        )

    def _mark_launch_high_priority_feedback_shown(
        self,
        reason: str,
        snackbar: object | None = None,
    ) -> None:
        if not getattr(self, "_github_star_prompt_launch_pending", True):
            return
        self._launch_high_priority_feedback_shown = True
        self._launch_high_priority_feedback_reason = reason
        if snackbar is not None:
            self._launch_high_priority_snackbar = snackbar

    def _launch_feedback_conflicts_with_github_star_prompt(self) -> bool:
        if getattr(self, "_launch_high_priority_feedback_shown", False):
            return True
        snackbar = getattr(self, "_launch_high_priority_snackbar", None)
        return bool(getattr(snackbar, "open", False))

    async def maybe_show_github_star_prompt_after_launch(
        self,
        *,
        delay_s: float = GITHUB_STAR_PROMPT_DELAY_S,
    ) -> bool:
        try:
            task = self.application.start_github_star_prompt(
                lambda generation: self._run_github_star_prompt_after_launch(
                    delay_s=delay_s,
                    generation=generation,
                )
            )
        except RuntimeError:
            return False
        return await task

    async def _run_github_star_prompt_after_launch(
        self,
        *,
        delay_s: float,
        generation: int,
    ) -> bool:
        try:
            launch_gate_satisfied = (
                await self.application.persist_github_star_prompt_eligible_launch()
            )
            if self._launch_feedback_conflicts_with_github_star_prompt():
                return False
            if not launch_gate_satisfied:
                return False
            if not self.application.should_show_github_star_prompt():
                return False
            if not self._is_current_github_star_prompt_generation(generation):
                return False

            await asyncio.sleep(delay_s)

            if not self._is_current_github_star_prompt_generation(generation):
                return False
            if self._launch_feedback_conflicts_with_github_star_prompt():
                return False
            if not self.application.should_show_github_star_prompt():
                return False
            return await self._open_github_star_prompt_snackbar(
                should_open=lambda: not self._launch_feedback_conflicts_with_github_star_prompt()
            )
        finally:
            self._github_star_prompt_launch_pending = False

    def _is_current_github_star_prompt_generation(self, generation: int) -> bool:
        return self.application.is_current_github_star_prompt_generation(generation)

    async def close_github_star_prompt_runtime(self) -> None:
        await self.application.close_github_star_prompt_runtime()
        self._github_star_prompt_launch_pending = False

    def schedule_after_launch_tasks(self) -> None:
        handle = getattr(self, "_after_launch_task_handle", None)
        if handle is not None and not handle.done():
            return
        self._after_launch_task_handle = self._run_page_task(self._run_after_launch_tasks)

    async def _run_after_launch_tasks(self) -> None:
        await asyncio.gather(
            self._run_launch_notification_flow(),
            self._run_after_launch_runtime_preparation(),
        )

    async def _run_launch_notification_flow(self) -> None:
        exhausted, _ = await asyncio.gather(
            self._refresh_openrouter_usage_after_launch(),
            self._check_for_update_after_launch(),
        )
        if exhausted:
            self._mark_launch_high_priority_feedback_shown("usage_exhaustion")

        try:
            await self.maybe_show_github_star_prompt_after_launch()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_detailed(
                f"[Startup] GitHub star prompt failed: {exc!r}",
                level=logging.WARNING,
            )

    async def _refresh_openrouter_usage_after_launch(self) -> bool:
        try:
            return await self.application.refresh_openrouter_usage_after_launch()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_detailed(
                f"[Startup] OpenRouter usage refresh failed: {exc!r}",
                level=logging.WARNING,
            )
            return False

    async def _check_for_update_after_launch(self) -> None:
        update_kwargs = {"log_detailed": self._log_detailed}
        try:
            update_parameters = inspect.signature(_check_and_notify_update).parameters
        except (TypeError, ValueError):
            update_parameters = {}
        if "load_update_info" in update_parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in update_parameters.values()
        ):
            update_kwargs["load_update_info"] = self.application.check_for_update
        if "on_launch_snackbar_shown" in update_parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in update_parameters.values()
        ):
            update_kwargs["on_launch_snackbar_shown"] = (
                lambda snackbar: self._mark_launch_high_priority_feedback_shown("update", snackbar)
            )
        try:
            await _check_and_notify_update(self.page, **update_kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_detailed(
                f"[Startup] Update check failed: {exc!r}",
                level=logging.WARNING,
            )

    async def _run_after_launch_runtime_preparation(self) -> None:
        try:
            await self.application.prepare_runtime_after_launch()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_detailed(
                f"[Startup] Runtime preparation failed: {exc!r}",
                level=logging.WARNING,
            )

    async def _close_after_launch_ui_tasks(self) -> None:
        handle = getattr(self, "_after_launch_task_handle", None)
        if handle is not None:
            if not handle.done():
                handle.cancel()
            try:
                if isinstance(handle, asyncio.Future):
                    await asyncio.gather(handle, return_exceptions=True)
                elif inspect.isawaitable(handle):
                    await handle
                else:
                    await asyncio.wrap_future(handle)
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            finally:
                if self._after_launch_task_handle is handle:
                    self._after_launch_task_handle = None
        self._github_star_prompt_launch_pending = False

    async def close_after_launch_tasks(self) -> None:
        await self._close_after_launch_ui_tasks()
        await self.close_github_star_prompt_runtime()

    async def _open_github_star_prompt_snackbar(self, *, should_open=None) -> bool:  # noqa: ANN001
        if getattr(self, "_github_star_prompt_shown_this_launch", False):
            return False
        if not await self.application.persist_github_star_prompt_opened(should_open=should_open):
            return False

        snackbar = None

        def _open_repository(_event) -> None:  # noqa: ANN001
            async def _persist_click() -> None:
                await self.application.persist_github_star_prompt_clicked()

            self._queue_settings_mutation_task(_persist_click)
            webbrowser.open(GITHUB_STAR_REPOSITORY_URL)
            if snackbar is not None:
                self._close_github_star_prompt_snackbar(snackbar)

        snackbar = self._build_github_star_prompt_snackbar(_open_repository)
        self._github_star_prompt_shown_this_launch = True
        self.page.show_dialog(snackbar)
        return True

    def _build_github_star_prompt_snackbar(self, on_click) -> ft.SnackBar:  # noqa: ANN001
        return ft.SnackBar(
            content=ft.Row(
                controls=[
                    ft.Text(
                        t("github_star.snackbar.message"),
                        size=18,
                        color=ft.Colors.WHITE,
                        font_family=font_for_language(get_locale()),
                        expand=True,
                    ),
                    ft.TextButton(
                        content=t("github_star.snackbar.action"),
                        on_click=on_click,
                        style=ft.ButtonStyle(
                            color=ft.Colors.WHITE,
                            text_style=ft.TextStyle(
                                size=18,
                                font_family=font_for_language(get_locale()),
                            ),
                            overlay_color=COLOR_PRIMARY,
                        ),
                    ),
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=12,
            ),
            bgcolor=COLOR_SUCCESS,
            duration=GITHUB_STAR_PROMPT_DURATION_MS,
            behavior=ft.SnackBarBehavior.FLOATING,
            elevation=0,
            margin=ft.Margin.only(bottom=90),
            padding=20,
        )

    def _close_github_star_prompt_snackbar(self, snackbar: ft.SnackBar) -> None:
        pop_dialog = getattr(self.page, "pop_dialog", None)
        if callable(pop_dialog):
            with contextlib.suppress(Exception):
                pop_dialog()
        else:
            snackbar.open = False
            with contextlib.suppress(Exception):
                self.page.update()

    def _preview_github_star_snackbar(self) -> None:
        snackbar = None

        def _open_repository(_event) -> None:  # noqa: ANN001
            webbrowser.open(GITHUB_STAR_REPOSITORY_URL)
            if snackbar is not None:
                self._close_github_star_prompt_snackbar(snackbar)

        snackbar = self._build_github_star_prompt_snackbar(_open_repository)
        self.page.show_dialog(snackbar)

    def _preview_telemetry_consent(self) -> None:
        dialog = TelemetryConsentDialog(
            self.page,
            on_allow=self._debug_preview_noop,
            on_decline=self._debug_preview_noop,
        )
        self._telemetry_consent_dialog = dialog
        dialog.open()

    def _preview_brake_notice(self) -> None:
        self._show_snackbar(t("managed_release.brake"), ft.Colors.ORANGE_700)

    def _preview_revoked_notice(self) -> None:
        self._show_snackbar(t("managed_release.revoked_contact"), ft.Colors.ORANGE_700)

    def _debug_preview_noop(self) -> None:
        return None

    def _preview_founder_letter(self) -> None:
        dialog = FounderLetterDialog(self.page, on_readme=self._on_founder_letter_readme)
        self._founder_letter_dialog = dialog
        dialog.open()

    def _preview_pkce_failure(self) -> None:
        self._show_snackbar(t("openrouter.pkce.failed"), ft.Colors.ORANGE_700)

    def _preview_discord_auth(self) -> None:
        self.show_discord_managed_auth_dialog(preview=True)

    def _open_qq_auth_preview_dialog(self) -> QqManagedAuthDialog:
        dialog = QqManagedAuthDialog(
            self.page,
            on_continue=self._close_qq_managed_auth_dialog,
            on_close=self._close_qq_managed_auth_dialog,
            on_cancel=self._close_qq_managed_auth_dialog,
        )
        self._qq_managed_auth_dialog = dialog
        dialog.open()
        return dialog

    def _preview_qq_auth(self) -> None:
        self._open_qq_auth_preview_dialog()

    def _preview_qq_auth_recoverable_error(self) -> None:
        dialog = self._open_qq_auth_preview_dialog()
        dialog.set_error("qq_auth.error.credential_mismatch")

    def _preview_qq_auth_translation_gated(self) -> None:
        dialog = self._open_qq_auth_preview_dialog()
        dialog.set_error("qq_auth.error.key_unavailable")

    def _preview_discord_callback_page(self) -> None:
        webbrowser.open(_write_discord_callback_preview_page(get_locale()))

    def _preview_peer_translation_eula(self) -> None:
        self._show_peer_translation_eula(self._debug_preview_noop)

    def _preview_local_qwen_hallucination_modal(self) -> None:
        self.show_local_qwen_hallucination_dialog()

    def _preview_talk_together_pass_invite_progress(self) -> None:
        set_managed_key_state = getattr(self.view_settings, "set_managed_key_state", None)
        if not callable(set_managed_key_state):
            return
        set_managed_key_state(
            visible=True,
            remaining_percent=100,
            referral_id=DEBUG_PREVIEW_TALK_TOGETHER_PASS_ID,
            remember_referral_id=False,
            pass_status=TalkTogetherPassStatus(
                pass_id=DEBUG_PREVIEW_TALK_TOGETHER_PASS_ID,
                invite_count=1,
                invite_limit=5,
                bonus_translations_per_friend=200,
            ),
        )

    def _preview_capture_fault_cycle(self) -> None:
        profile = self.application.cycle_debug_capture_fault_profile()
        self._show_snackbar(
            t("debug_preview.capture_fault_snackbar", profile=profile), ft.Colors.ORANGE_700
        )

    def _preview_stt_fault_cycle(self) -> None:
        profile = self.application.cycle_debug_stt_fault_profile()
        self._show_snackbar(
            t("debug_preview.stt_fault_snackbar", profile=profile), ft.Colors.ORANGE_700
        )

    def _preview_audio_fault_clear(self) -> None:
        self.application.clear_debug_audio_fault_profiles()
        self._show_snackbar(t("debug_preview.audio_fault_clear"), ft.Colors.GREEN_700)

    def _preview_foundation_primitives(self) -> None:
        if not self._foundation_adapter.debug_preview_enabled:
            return
        dialog = ft.AlertDialog(
            modal=False,
            content=FoundationPreviewSurface(get_locale()),
            bgcolor=COLOR_BACKGROUND,
        )
        self._foundation_preview_dialog = dialog
        self.page.show_dialog(dialog)

    def _preview_http_extension_form(self) -> None:
        registry_service = self.application.http_extension_registry()
        if registry_service is None:
            return
        directory = registry_service.directory
        directory.mkdir(parents=True, exist_ok=True)
        demo_path = directory / "debug_demo.json"
        if not demo_path.exists():
            demo_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "id": "debug_demo",
                        "name": "Debug Demo Translator",
                        "description": "Debug preview: HTTP extension credential form",
                        "url": "https://example.com/translate",
                        "request": {
                            "body": {
                                "type": "json",
                                "value": {
                                    "q": "{{text}}",
                                    "api_key": "{{secret:api_key}}",
                                },
                            }
                        },
                        "response": {"type": "text"},
                        "secrets": [{"id": "api_key", "label": "API Key"}],
                    }
                ),
                encoding="utf-8",
            )
        registry_service.reload()
        view = self.view_settings
        set_registry = getattr(view, "set_http_extension_registry", None)
        if callable(set_registry):
            set_registry(registry_service)
        select_llm = getattr(view, "_on_llm_selected", None)
        if callable(select_llm):
            select_llm("custom_http")
        select_extension = getattr(view, "_on_http_extension_selected", None)
        if callable(select_extension):
            select_extension("debug_demo")
        if self._current_tab != 1:
            self._open_settings_tab()

    def _show_peer_translation_eula(self, on_accept) -> None:
        dialog = PeerTranslationEulaDialog(
            self.page,
            on_accept=on_accept,
            on_cancel=self._debug_preview_noop,
        )
        self._peer_translation_eula_dialog = dialog
        dialog.open()

    def maybe_show_telemetry_consent_dialog(self) -> bool:
        return False

    def _on_telemetry_consent_change(self, consent: str) -> None:
        if consent not in {"allow", "decline"}:
            return

        async def _task() -> None:
            settings = await self.application.apply_telemetry_consent(consent)
            sync_telemetry = getattr(self.view_settings, "sync_telemetry_settings", None)
            if callable(sync_telemetry) and settings is not None:
                sync_telemetry(settings)

        self._run_page_task(_task)

    def show_local_qwen_hallucination_dialog(self) -> None:
        dialog = LocalQwenHallucinationDialog(
            self.page,
            on_open_guide=self._open_local_qwen_guide,
        )
        self._local_qwen_hallucination_dialog = dialog
        dialog.open()

    def _open_local_qwen_guide(self) -> None:
        webbrowser.open(founder_readme_url_for_locale(get_locale()))

    def _accept_peer_translation_eula_and_enable(self) -> None:
        async def _task():
            await self.application.accept_peer_translation_eula_and_enable()

        self._run_page_task(_task)

    def _close_open_dialog_for_navigation(self) -> None:
        microphone_test_dialog = getattr(self, "_microphone_test_dialog", None)
        if microphone_test_dialog is not None and getattr(
            microphone_test_dialog,
            "is_open",
            False,
        ):
            microphone_test_dialog.close(notify=True)
            return

        pop_dialog = getattr(self.page, "pop_dialog", None)
        if not callable(pop_dialog):
            return
        try:
            pop_dialog()
        except Exception:
            logger.exception("Failed to close dialog during navigation")

    def _queue_settings_mutation_task(self, task_factory) -> None:
        queue = getattr(self, "_settings_mutation_queue", None)
        if queue is None:
            queue = []
            self._settings_mutation_queue = queue
        queue.append(task_factory)
        if getattr(self, "_settings_mutation_worker_active", False):
            return
        self._settings_mutation_worker_active = True

        async def _worker():
            try:
                while self._settings_mutation_queue:
                    next_task = self._settings_mutation_queue.pop(0)
                    try:
                        await next_task()
                    except Exception:
                        logger.exception("Settings mutation task failed")
            finally:
                self._settings_mutation_worker_active = False

        self._run_page_task(_worker)

    def _content_padding_for_index(self, index: int) -> int:
        return 0 if index == 1 else APP_CONTENT_PADDING

    def _on_nav_change(self, index: int):
        # Track previous tab for Settings auto-apply
        previous_tab = getattr(self, "_current_tab", 0)
        if previous_tab != index:
            self._close_open_dialog_for_navigation()
        self._current_tab = index

        # Auto-apply Settings changes when leaving Settings (tab 1)
        if previous_tab == 1 and index != 1:
            if self.view_settings.has_provider_changes:
                pending_settings = self.view_settings.consume_provider_apply_settings()
                if pending_settings is not None:
                    self.view_settings.has_provider_changes = False

                    async def _task():
                        applied = await self.application.apply_providers(pending_settings)
                        if applied:
                            self._run_page_task(
                                self.application.install_selected_gpu_model_if_needed
                            )

                    self._queue_settings_mutation_task(_task)
            elif getattr(self.view_settings, "has_pending_prompt_changes", False):
                pending_settings = self.view_settings.consume_prompt_apply_settings()
                if pending_settings is not None:

                    async def _task():
                        merged_settings = (
                            self.application.merge_settings_tab_apply_with_current_languages(
                                pending_settings
                            )
                        )
                        await self.application.apply_settings(merged_settings)

                    self._queue_settings_mutation_task(_task)
            else:
                self._run_page_task(self.application.install_selected_gpu_model_if_needed)

        if index == 0:
            self.content_area.content = self.view_dashboard
        elif index == 1:
            self.content_area.content = self.view_settings
        elif index == 2:
            self.content_area.content = self.view_logs
        elif index == 3:
            self.content_area.content = self.view_about

        self.content_area.padding = self._content_padding_for_index(index)
        self.content_area.update()
        if index == 1:
            self.view_settings.refresh_prompt_if_empty()
        elif index == 2:
            # Async scroll after rendering completes
            async def _scroll():
                import asyncio

                await asyncio.sleep(0.05)
                await self.view_logs.scroll_to_bottom()

            self._run_page_task(_scroll)

    def _open_logs_tab(self) -> None:
        self._on_nav_change(2)
        self._set_bottom_nav_selected(2)

    def _open_settings_tab(self) -> None:
        self._on_nav_change(1)
        self._set_bottom_nav_selected(1)

    def _set_bottom_nav_selected(self, index: int) -> None:
        selected_attr = getattr(self.bottom_nav, "_selected", None)
        if selected_attr != index and hasattr(self.bottom_nav, "_selected"):
            self.bottom_nav._selected = index
        update_visuals = getattr(self.bottom_nav, "_update_visuals", None)
        if callable(update_visuals):
            with contextlib.suppress(Exception):
                update_visuals()

    def apply_locale(self) -> None:
        self.page.title = t("app.title")
        self.page.theme = get_app_theme(font_family=font_for_language(get_locale()))
        self.title_bar.set_title(t("app.title"))
        self.view_dashboard.apply_locale()
        self.view_settings.apply_locale()
        self.refresh_overlay_peer_contract()
        self.view_logs.apply_locale()
        debug_preview_panel = getattr(self, "debug_preview_panel", None)
        apply_debug_locale = getattr(debug_preview_panel, "apply_locale", None)
        if callable(apply_debug_locale):
            apply_debug_locale()
        foundation_preview_dialog = getattr(self, "_foundation_preview_dialog", None)
        if foundation_preview_dialog is not None:
            foundation_preview_dialog.content = FoundationPreviewSurface(get_locale())
        self.page.update()

    def _cycle_debug_preview_stt_loading_button(self) -> None:
        states = ("off", "starting", "on")
        index = int(getattr(self, "_debug_preview_stt_loading_button_index", -1)) + 1
        index %= len(states)
        self._debug_preview_stt_loading_button_index = index
        state = states[index]
        self.view_dashboard.stt_button.set_state(
            state == "on",
            is_starting=state == "starting",
        )

    def _cycle_debug_preview_gpu_state(self) -> None:
        states = (
            "discovery_failed",
            "not_installed",
            "invalid",
            "installing",
            "install_failed",
            "unsupported",
            "unavailable_device",
            "activation_failed",
        )
        index = int(getattr(self, "_debug_preview_gpu_state_index", -1)) + 1
        index %= len(states)
        self._debug_preview_gpu_state_index = index
        devices = (
            GpuDeviceOption("vulkan-index-0", "Debug GPU 0", "Vulkan0"),
            GpuDeviceOption("vulkan-index-1", "Debug GPU 1", "Vulkan1"),
        )
        set_devices = getattr(self.view_settings, "set_gpu_devices", None)
        if callable(set_devices):
            set_devices(devices=devices)
        action_by_state = {
            "discovery_failed": "rediscover",
            "activation_failed": "restart",
        }
        notice = GpuDashboardNotice(
            status=states[index],
            progress_percent=42 if states[index] == "installing" else None,
            action=action_by_state.get(states[index]),
        )
        set_notice = getattr(getattr(self, "view_dashboard", None), "set_gpu_notice", None)
        if callable(set_notice):
            set_notice(notice)
        else:
            legacy_setter = getattr(self.view_settings, "set_gpu_runtime_state", None)
            if callable(legacy_setter):
                legacy_setter(
                    states[index],
                    devices=devices,
                    progress_percent=notice.progress_percent,
                )

    def refresh_overlay_peer_contract(self) -> None:
        presentation = getattr(self, "_presentation_adapter", None)
        if presentation is None:
            presentation = FletUiPresentationAdapter(self)
            self._presentation_adapter = presentation
        presentation.refresh_overlay_peer_contract(
            self.application.overlay_peer_presentation_state()
        )

    def _sync_settings_overlay_runtime_state(self) -> None:
        view_settings = getattr(self, "view_settings", None)
        set_state = getattr(view_settings, "set_overlay_runtime_state", None)
        if not callable(set_state):
            return
        state = self.application.state()
        set_state(
            self.overlay_state,
            failure_reason=self.overlay_failure_reason,
            overlay_target=state.overlay_target,
            desktop_captions_locked=state.desktop_overlay_captions_locked,
        )

    def _on_desktop_overlay_lock_change(self, locked: bool) -> None:
        async def _task():
            await self.application.set_desktop_overlay_captions_locked(bool(locked))
            self._refresh_settings_desktop_overlay_state()

        self._run_page_task(_task)

    def _on_desktop_overlay_size_change(self, size_preset: str) -> None:
        async def _task():
            await self.application.set_desktop_overlay_size_preset(size_preset)
            self._refresh_settings_desktop_overlay_state()

        self._run_page_task(_task)

    def _on_desktop_overlay_recovery_action(self, action: str) -> None:
        if action not in {"retry", "reopen"}:
            return

        async def _task():
            await self.application.set_overlay_enabled(True)

        self._run_page_task(_task)

    def _on_desktop_overlay_position_reset(self) -> None:
        async def _task():
            await self.application.reset_desktop_overlay_position()
            self._refresh_settings_desktop_overlay_state()

        self._run_page_task(_task)

    def _refresh_settings_desktop_overlay_state(self) -> None:
        settings = self.application.compatibility_settings()
        view_settings = getattr(self, "view_settings", None)
        sync_settings = getattr(view_settings, "sync_desktop_overlay_settings", None)
        if settings is not None and callable(sync_settings):
            sync_settings(settings)
        self._sync_settings_overlay_runtime_state()

    def on_desktop_overlay_state_changed(
        self,
        *,
        interaction_mode: str | None = None,
        captions_locked: bool | None = None,
    ) -> None:
        _ = (interaction_mode, captions_locked)
        self._sync_settings_overlay_runtime_state()

    def _on_manual_submit(self, _source: str, text: str) -> None:
        async def _task():
            await self.application.submit_text(text)

        self._run_page_task(_task)

    def _on_message_input_activity(self, has_text: bool) -> None:
        async def _task():
            self.application.set_manual_input_activity(has_text)

        self._run_page_task(_task)

    def _on_keyboard_event(self, event) -> None:
        if getattr(event, "key", None) != "Tab":
            return
        if any(
            bool(getattr(event, modifier, False)) for modifier in ("shift", "ctrl", "alt", "meta")
        ):
            return

        dashboard = getattr(self, "view_dashboard", None)
        content_area = getattr(self, "content_area", None)
        if dashboard is None or getattr(content_area, "content", None) is not dashboard:
            return

        handler = getattr(dashboard, "handle_message_input_tab_key", None)
        if callable(handler):
            handler()

    def _log_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self.application.log_basic(message, level=level)

    def _log_detailed(self, message: str, *, level: int = logging.INFO) -> None:
        self.application.log_detailed(message, level=level)

    def _revert_dashboard_translation_toggle(self) -> None:
        self._set_dashboard_translation_visual_state(False)

    def _set_dashboard_translation_visual_state(self, enabled: bool) -> None:
        dash = getattr(self, "view_dashboard", None)
        set_translation_enabled = getattr(dash, "set_translation_enabled", None)
        if callable(set_translation_enabled):
            try:
                set_translation_enabled(enabled)
            except Exception:
                logger.exception("Failed to update dashboard translation toggle")

    def _dashboard_managed_auth_action(self) -> str:
        try:
            return self.application.dashboard_managed_auth_action()
        except Exception:
            logger.exception("Failed to evaluate managed auth dashboard gate")
            return "prompt"

    def _dashboard_managed_auth_prompt_kind(self) -> str:
        try:
            resolved = self.application.dashboard_managed_auth_prompt_kind()
        except Exception:
            logger.warning("Failed to evaluate managed auth prompt kind")
            return "discord"
        return "qq" if resolved == "qq" else "discord"

    def _on_translation_toggle(self, enabled: bool) -> bool:
        self._log_basic(f"[Dashboard] Translation toggle requested: enabled={enabled}")
        self._log_detailed(
            "[Dashboard] Translation toggle detail: "
            f"dashboard_state={getattr(getattr(self, 'view_dashboard', None), 'is_translation_on', None)} "
            f"overlay_state={getattr(self, 'overlay_state', 'unknown')}"
        )
        if enabled:
            managed_auth_action = self._dashboard_managed_auth_action()
            if managed_auth_action in {"prompt", "in_progress"}:
                self._revert_dashboard_translation_toggle()
                if managed_auth_action == "prompt":
                    if self._dashboard_managed_auth_prompt_kind() == "qq":
                        self.show_qq_managed_auth_dialog()
                    else:
                        self.show_discord_managed_auth_dialog(preview=False)
                return False

        async def _task():
            await self.application.set_translation_enabled(enabled)

        self._run_page_task(_task)
        return True

    def _on_stt_toggle(self, enabled: bool) -> None:
        self._log_basic(f"[Dashboard] STT toggle requested: enabled={enabled}")
        self._log_detailed(
            "[Dashboard] STT toggle detail: "
            f"dashboard_state={getattr(getattr(self, 'view_dashboard', None), 'is_stt_on', None)} "
            f"overlay_state={getattr(self, 'overlay_state', 'unknown')}"
        )

        async def _task():
            await self.application.set_stt_enabled(enabled)

        self._run_page_task(_task)

    def _on_overlay_toggle(self, enabled: bool) -> None:
        self._log_basic(f"[Dashboard] Overlay toggle requested: enabled={enabled}")
        self._log_detailed(
            "[Dashboard] Overlay toggle detail: "
            f"overlay_state={getattr(self, 'overlay_state', 'unknown')} "
            f"failure_reason={getattr(self, 'overlay_failure_reason', None)}"
        )

        async def _task():
            await self.application.set_overlay_enabled(enabled)

        self._run_page_task(_task)

    def _on_peer_translation_toggle(self, enabled: bool) -> None:
        self._log_basic(f"[Dashboard] Peer toggle requested: enabled={enabled}")
        self._log_detailed(
            "[Dashboard] Peer toggle detail: "
            f"overlay_state={getattr(self, 'overlay_state', 'unknown')} "
            f"failure_reason={getattr(self, 'overlay_failure_reason', None)}"
        )
        if enabled and self.application.state().peer_translation_eula_accepted is False:
            self._show_peer_translation_eula(self._accept_peer_translation_eula_and_enable)
            return

        async def _task():
            await self.application.set_peer_translation_enabled(enabled)

        self._run_page_task(_task)

    def _on_retry_peer_process_capture(self) -> None:
        self._log_basic("[Dashboard] Peer process capture retry requested")

        async def _task():
            await self.application.retry_peer_process_capture()

        self._queue_settings_mutation_task(_task)

    def _on_apply_loopback_capture_option(self, value: str) -> None:
        async def _task():
            await self.application.apply_loopback_capture_option(value)

        self._queue_settings_mutation_task(_task)

    def _on_gpu_discovery_requested(self) -> None:
        self._run_page_task(self.application.ensure_gpu_device_discovery)

    def _on_language_change(
        self,
        change: LanguageSelectionChange,
    ) -> None:
        settings = self.application.compatibility_settings()
        if settings is None:
            return
        previous_source_code = settings.languages.source_language
        previous_target_code = settings.languages.target_language
        previous_peer_source_code = getattr(settings.languages, "peer_source_language", "")
        previous_peer_target_code = getattr(settings.languages, "peer_target_language", "")
        self._log_basic(
            "[Dashboard] Language change requested: "
            f"source={previous_source_code}->{change.source_code} "
            f"target={previous_target_code}->{change.target_code} "
            f"peer_source={previous_peer_source_code}->{change.peer_source_code} "
            f"peer_target={previous_peer_target_code}->{change.peer_target_code}"
        )
        self._log_detailed(
            f"[Dashboard] Language change detail: overlay_state={getattr(self, 'overlay_state', 'unknown')}"
        )

        # Check STT provider compatibility and show warning if needed
        warning = None
        if change.source_code != previous_source_code:
            stt_provider = settings.provider.stt.value
            warning = get_stt_compatibility_warning(change.source_code, stt_provider)
        if warning:
            snackbar = ft.SnackBar(
                ft.Text(t(warning.key, language=language_name(warning.language_code))),
                bgcolor=ft.Colors.ORANGE_700,
                duration=4000,
                behavior=ft.SnackBarBehavior.FLOATING,
                elevation=0,
                margin=ft.Margin.only(bottom=90),
                padding=20,
            )
            self._mark_launch_high_priority_feedback_shown("stt_compatibility", snackbar)
            self.page.show_dialog(snackbar)

        async def _task():
            await self.application.on_dashboard_language_change(change)

        self._queue_settings_mutation_task(_task)

    def _on_settings_changed(self, settings) -> None:
        captured_change = self.application.capture_settings_view_change(settings)

        async def _task():
            next_settings = self.application.merge_settings_view_change_with_current(
                captured_change
            )
            await self.application.apply_settings(next_settings)
            self._sync_microphone_test_dialog_if_inactive()

        self._queue_settings_mutation_task(_task)

    def _on_start_microphone_test(self) -> None:
        async def _task():
            dialog = self._get_microphone_test_dialog()
            dialog.reset()
            dialog.open()
            started = await self.application.start_microphone_test(meter_callback=dialog.set_level)
            if not started:
                dialog.show_failure()
                return

        self._queue_settings_mutation_task(_task)

    def _on_stop_microphone_test(self) -> None:
        async def _task() -> None:
            await self.application.stop_microphone_test()
            self._close_microphone_test_dialog()

        self._queue_settings_mutation_task(_task)

    def _get_microphone_test_dialog(self) -> MicrophoneTestDialog:
        dialog = getattr(self, "_microphone_test_dialog", None)
        if dialog is None:
            dialog = MicrophoneTestDialog(
                self.page,
                on_close=self._on_microphone_test_dialog_dismiss,
            )
            self._microphone_test_dialog = dialog
        return dialog

    def _close_microphone_test_dialog(self) -> None:
        dialog = getattr(self, "_microphone_test_dialog", None)
        if dialog is None:
            return
        dialog.close(notify=False)
        dialog.reset()

    def _on_microphone_test_dialog_dismiss(self) -> None:
        self._on_stop_microphone_test()

    def _sync_microphone_test_dialog_if_inactive(self) -> None:
        if self.application.state().microphone_test_active:
            return
        self._close_microphone_test_dialog()

    def _on_prompt_apply_settings(self, settings) -> None:
        async def _task():
            merged_settings = self.application.merge_settings_tab_apply_with_current_languages(
                settings
            )
            await self.application.apply_settings(merged_settings)

        self._queue_settings_mutation_task(_task)

    def _on_runtime_logging_mode_change(self, mode: str) -> None:
        resolved_mode = self.application.set_runtime_logging_mode(mode)
        self.view_logs.set_runtime_logging_mode(resolved_mode)

    def _on_providers_changed(self) -> None:
        view_settings = getattr(self, "view_settings", None)
        consume_http_extension_runtime_reload = getattr(
            view_settings,
            "consume_http_extension_runtime_reload",
            None,
        )
        if callable(consume_http_extension_runtime_reload) and (
            consume_http_extension_runtime_reload()
        ):

            async def _runtime_only_task():
                await self.application.apply_providers(
                    persist_settings=False,
                    refresh_ui=False,
                )

            self._queue_settings_mutation_task(_runtime_only_task)
            return

        pending_settings = None
        consume_provider_apply_settings = getattr(
            view_settings,
            "consume_provider_apply_settings",
            None,
        )
        if callable(consume_provider_apply_settings) and getattr(
            view_settings,
            "has_provider_changes",
            False,
        ):
            pending_settings = consume_provider_apply_settings()
            view_settings.has_provider_changes = False

        async def _task():
            if pending_settings is None:
                await self.application.apply_providers()
            else:
                await self.application.apply_providers(pending_settings)

        self._queue_settings_mutation_task(_task)

    def _on_local_llm_secret_changed(self) -> None:
        async def _task():
            if not self.application.local_llm_selected():
                return
            await self.application.apply_providers(force_rebuild_llm=True)

        self._queue_settings_mutation_task(_task)

    def _on_custom_stt_secret_changed(self) -> None:
        async def _task():
            await self.application.apply_providers(
                persist_settings=False,
                refresh_ui=False,
            )

        self._queue_settings_mutation_task(_task)

    def _on_request_openrouter_pkce(
        self,
        target_settings: object,
        *,
        launch_source: str = "settings",
    ) -> None:
        if getattr(self, "_openrouter_pkce_request_active", False):
            reopen_authorization_url = getattr(
                self.application,
                "reopen_openrouter_pkce_authorization_url",
                None,
            )
            if callable(reopen_authorization_url):
                reopen_authorization_url()
            return
        self._openrouter_pkce_request_active = True

        async def _task() -> None:
            try:
                ok = await self.application.connect_openrouter_via_pkce(
                    target_settings=target_settings,
                    launch_source=launch_source,
                )
                if ok:
                    self.application.refresh_settings_after_openrouter_pkce_success()
                    self._show_snackbar(t("openrouter.pkce.connected"), COLOR_SUCCESS)
            finally:
                self._openrouter_pkce_request_active = False

        self._queue_settings_mutation_task(_task)

    def _close_discord_managed_auth_dialog(self) -> None:
        dialog = getattr(self, "_discord_managed_auth_dialog", None)
        close = getattr(dialog, "close", None)
        if callable(close):
            close()

    def show_discord_managed_auth_dialog(self, preview: bool = False) -> None:
        if not preview:
            self._mark_launch_high_priority_feedback_shown("auth_required")
        if preview:
            on_continue = self._close_discord_managed_auth_dialog
            on_byok = self._close_discord_managed_auth_dialog
            on_close = self._close_discord_managed_auth_dialog
            on_reopen_browser = self._close_discord_managed_auth_dialog
            on_cancel = self._close_discord_managed_auth_dialog
        else:
            on_continue = self._start_discord_managed_auth
            on_byok = self._on_discord_managed_auth_byok
            on_close = self._close_discord_managed_auth_dialog
            on_reopen_browser = (
                self._reopen_discord_managed_auth_browser
                if self._supports_discord_managed_auth_reopen()
                else None
            )
            on_cancel = self._cancel_discord_managed_auth

        dialog = DiscordManagedAuthDialog(
            self.page,
            on_continue=on_continue,
            on_byok=on_byok,
            on_close=on_close,
            on_reopen_browser=on_reopen_browser,
            on_cancel=on_cancel,
        )
        self._discord_managed_auth_dialog = dialog
        dialog.open()

    def show_qq_managed_auth_dialog(self) -> None:
        self._mark_launch_high_priority_feedback_shown("auth_required")
        dialog = QqManagedAuthDialog(
            self.page,
            on_continue=self._start_qq_managed_auth,
            on_close=self._close_qq_managed_auth_dialog,
            on_cancel=self._cancel_qq_managed_auth,
        )
        self._qq_managed_auth_dialog = dialog
        dialog.open()

    def _close_qq_managed_auth_dialog(self) -> None:
        dialog = getattr(self, "_qq_managed_auth_dialog", None)
        if dialog is not None:
            close = getattr(dialog, "close", None)
            if callable(close):
                close()

    def _next_qq_managed_auth_generation(self) -> int:
        generation = int(getattr(self, "_qq_managed_auth_generation", 0)) + 1
        self._qq_managed_auth_generation = generation
        self._qq_managed_auth_cancelled = False
        return generation

    def _is_current_qq_managed_auth_generation(self, generation: int) -> bool:
        return bool(
            generation == getattr(self, "_qq_managed_auth_generation", None)
            and not getattr(self, "_qq_managed_auth_cancelled", False)
        )

    def _start_qq_managed_auth(self) -> None:
        dialog = getattr(self, "_qq_managed_auth_dialog", None)
        qq_identity = getattr(dialog, "qq_identity", "")
        credential = getattr(dialog, "credential", "")
        set_waiting = getattr(dialog, "set_waiting", None)
        if callable(set_waiting):
            set_waiting()
        generation = self._next_qq_managed_auth_generation()

        async def _task() -> None:
            application = self.application
            if not self._is_current_qq_managed_auth_generation(generation):
                return
            try:
                result = await application.start_qq_managed_auth_from_dialog(
                    qq_identity=qq_identity,
                    credential=credential,
                )
            except asyncio.CancelledError:
                return
            except Exception:
                self._log_basic("[ManagedAuth] QQ auth task failed", level=logging.ERROR)
                result = None
            if not self._is_current_qq_managed_auth_generation(generation):
                return
            if result is True:
                enable_result = await application.set_translation_enabled(True)
                if not self._is_current_qq_managed_auth_generation(generation):
                    return
                if not application.translation_enable_succeeded(enable_result):
                    set_error = getattr(dialog, "set_error", None)
                    if callable(set_error):
                        set_error("qq_auth.error.retry")
                    self._qq_managed_auth_task_handle = None
                    return
                self._close_qq_managed_auth_dialog()
                self._show_snackbar(t("qq_auth.success"), COLOR_SUCCESS)
                self._set_dashboard_translation_visual_state(True)
                self._qq_managed_auth_task_handle = None
                self.application.clear_managed_auth_task(
                    "qq-managed-auth-dialog",
                )
                return
            message_key = "qq_auth.error.retry"
            message_kwargs: dict[str, object] = {}
            if isinstance(result, tuple) and result:
                message_key = str(result[0])
                if len(result) > 1 and isinstance(result[1], dict):
                    message_kwargs = dict(result[1])
            set_error = getattr(dialog, "set_error", None)
            if callable(set_error):
                set_error(message_key, **message_kwargs)
            if self._is_current_qq_managed_auth_generation(generation):
                self._qq_managed_auth_task_handle = None
                self.application.clear_managed_auth_task(
                    "qq-managed-auth-dialog",
                )

        self._qq_managed_auth_task_handle = self.application.start_managed_auth_task(
            task_runner=self._run_page_task,
            task_factory=_task,
            task_name="qq-managed-auth-dialog",
            generation=generation,
        )

    def _cancel_qq_managed_auth(self) -> None:
        self._qq_managed_auth_cancelled = True
        task_handle = getattr(self, "_qq_managed_auth_task_handle", None)
        self.application.cancel_managed_auth_task(
            task_handle,
            task_name="qq-managed-auth-dialog",
        )
        cancel = getattr(task_handle, "cancel", None)
        if callable(cancel):
            with contextlib.suppress(Exception):
                cancel()
        self._qq_managed_auth_task_handle = None
        self._close_qq_managed_auth_dialog()

    def _supports_discord_managed_auth_reopen(self) -> bool:
        return self.application.supports_discord_managed_auth_reopen()

    def _next_discord_managed_auth_generation(self) -> int:
        generation = int(getattr(self, "_discord_managed_auth_generation", 0)) + 1
        self._discord_managed_auth_generation = generation
        self._discord_managed_auth_cancelled = False
        return generation

    def _is_current_discord_managed_auth_generation(self, generation: int) -> bool:
        return bool(
            generation == getattr(self, "_discord_managed_auth_generation", None)
            and not getattr(self, "_discord_managed_auth_cancelled", False)
            and self.application.managed_auth_tasks_open()
        )

    def _start_discord_managed_auth(self) -> None:
        dialog = getattr(self, "_discord_managed_auth_dialog", None)
        raw_referral_id = getattr(dialog, "referral_id", "")
        referral_id = (
            raw_referral_id if isinstance(raw_referral_id, str) and raw_referral_id else None
        )
        set_waiting = getattr(dialog, "set_waiting", None)
        if callable(set_waiting):
            set_waiting()
        generation = self._next_discord_managed_auth_generation()

        async def _task() -> None:
            application = self.application
            if not self._is_current_discord_managed_auth_generation(generation):
                return

            def _mark_callback_received() -> None:
                self.mark_discord_managed_auth_callback_received(generation)

            try:
                ok = await application.start_discord_managed_auth_from_dialog(
                    on_callback_received=_mark_callback_received,
                    referral_id=referral_id,
                )
                if not ok or not self._is_current_discord_managed_auth_generation(generation):
                    return
                enable_result = await application.set_translation_enabled(True)
                if not self._is_current_discord_managed_auth_generation(generation):
                    return
                if not application.translation_enable_succeeded(enable_result):
                    return
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Discord managed auth task failed")
                return
            self._close_discord_managed_auth_dialog()
            self._show_snackbar(t("discord_auth.success"), COLOR_SUCCESS)
            if application.state().managed_auth_referral_bonus_applied:
                self._show_snackbar(t("discord_auth.referral_reward_applied"), COLOR_SUCCESS)
            self._set_dashboard_translation_visual_state(True)
            if self._is_current_discord_managed_auth_generation(generation):
                self._discord_managed_auth_task_handle = None
                self.application.clear_managed_auth_task(
                    "discord-managed-auth-dialog",
                )

        self._discord_managed_auth_task_handle = self.application.start_managed_auth_task(
            task_runner=self._run_page_task,
            task_factory=_task,
            task_name="discord-managed-auth-dialog",
            generation=generation,
        )

    async def _close_managed_auth_ui_tasks(self) -> None:
        self._discord_managed_auth_cancelled = True
        self._qq_managed_auth_cancelled = True
        self._discord_managed_auth_generation = (
            int(getattr(self, "_discord_managed_auth_generation", 0)) + 1
        )
        self._qq_managed_auth_generation = int(getattr(self, "_qq_managed_auth_generation", 0)) + 1
        task_handle = getattr(self, "_discord_managed_auth_task_handle", None)
        qq_task_handle = getattr(self, "_qq_managed_auth_task_handle", None)
        self._discord_managed_auth_task_handle = None
        self._qq_managed_auth_task_handle = None
        self.application.cancel_managed_auth_task(
            qq_task_handle,
            task_name="qq-managed-auth-dialog",
        )
        self.application.cancel_managed_auth_task(
            task_handle,
            task_name="discord-managed-auth-dialog",
        )

    async def close_oauth_runtime(self) -> None:
        await self._close_managed_auth_ui_tasks()
        await self.application.close_managed_auth_tasks()

    def mark_discord_managed_auth_callback_received(self, generation: int | None = None) -> None:
        if generation is not None and not self._is_current_discord_managed_auth_generation(
            generation
        ):
            return
        dialog = getattr(self, "_discord_managed_auth_dialog", None)
        if getattr(dialog, "is_open", True) is False:
            return
        if getattr(dialog, "is_waiting", True) is False:
            return
        set_callback_received = getattr(dialog, "set_callback_received", None)
        if callable(set_callback_received):
            set_callback_received()

    def _reopen_discord_managed_auth_browser(self) -> None:
        result = self.application.reopen_discord_managed_auth_browser()
        if inspect.isawaitable(result):

            async def _task() -> None:
                await result

            self._run_page_task(_task)

    def _cancel_discord_managed_auth(self) -> None:
        self._discord_managed_auth_cancelled = True
        task_handle = getattr(self, "_discord_managed_auth_task_handle", None)
        self.application.cancel_managed_auth_task(
            task_handle,
            task_name="discord-managed-auth-dialog",
        )
        cancel = getattr(task_handle, "cancel", None)
        if callable(cancel):
            with contextlib.suppress(Exception):
                cancel()
        self._discord_managed_auth_task_handle = None
        self._close_discord_managed_auth_dialog()

    def _build_managed_openrouter_byok_target_settings(self) -> object | None:
        return self.application.build_managed_openrouter_byok_target_settings()

    def _build_founder_letter_target_settings(self) -> object | None:
        return self._build_managed_openrouter_byok_target_settings()

    def _on_discord_managed_auth_byok(self) -> None:
        target_settings = self._build_managed_openrouter_byok_target_settings()
        if target_settings is None:
            self._show_snackbar(t("openrouter.pkce.failed"), ft.Colors.ORANGE_700)
            return
        self._on_request_openrouter_pkce(target_settings, launch_source="discord_auth")

    def _on_founder_letter_connect(self) -> None:
        target_settings = self._build_founder_letter_target_settings()
        if target_settings is None:
            self._show_snackbar(t("openrouter.pkce.failed"), ft.Colors.ORANGE_700)
            return
        self._on_request_openrouter_pkce(target_settings, launch_source="letter")

    def _on_founder_letter_contact(self) -> None:
        webbrowser.open(FOUNDER_CONTACT_URL)

    def _on_founder_letter_readme(self) -> None:
        webbrowser.open(founder_readme_url_for_locale(get_locale()))

    def show_founder_letter_dialog(self) -> None:
        self._mark_launch_high_priority_feedback_shown("usage_exhaustion")
        dialog = FounderLetterDialog(self.page, on_readme=self._on_founder_letter_readme)
        self._founder_letter_dialog = dialog
        dialog.open()

    def _api_key_verification_matches_current_field(self, provider: str, key: str) -> bool:
        field_by_provider = {
            "deepgram": "_deepgram_key",
            "soniox": "_soniox_key",
            "google": "_google_key",
            "openrouter": "_openrouter_key",
            "deepseek": "_deepseek_key",
            "cerebras": "_cerebras_key",
            "alibaba_beijing": "_alibaba_key_beijing",
            "alibaba_singapore": "_alibaba_key_singapore",
        }
        field_name = field_by_provider.get(provider)
        if field_name is None:
            return True

        field = getattr(getattr(self, "view_settings", None), field_name, None)
        if field is None:
            return True

        current_key = getattr(field, "value", None)
        if current_key is None:
            return True

        return current_key == key

    async def _on_verify_api_key(self, provider: str, key: str) -> tuple[bool, str]:
        success, msg = await self.application.verify_api_key(provider, key)

        if not self._api_key_verification_matches_current_field(provider, key):
            return success, msg

        self.application.persist_api_key_verification(provider, key, success)

        # Sync verification result with dashboard needs_key flags (UI update on user click)
        if provider in ("deepgram", "soniox", "qwen_asr"):
            self.view_dashboard.set_stt_needs_key(not success, update_ui=False)
        elif provider in (
            "google",
            "openrouter",
            "deepseek",
            "cerebras",
            "alibaba_beijing",
            "alibaba_singapore",
        ):
            self.view_dashboard.set_translation_needs_key(not success, update_ui=False)

        return success, msg

    async def _on_provider_secret_change(self, key: str, value: str) -> bool:
        succeeded = await self.application.persist_provider_secret_change(key, value)
        if not succeeded:
            return False
        provider = {
            "deepgram_api_key": "deepgram",
            "soniox_api_key": "soniox",
            "google_api_key": "google",
            "openrouter_api_key": "openrouter",
            "deepseek_api_key": "deepseek",
            "cerebras_api_key": "cerebras",
            "alibaba_api_key_beijing": "alibaba_beijing",
            "alibaba_api_key_singapore": "alibaba_singapore",
        }.get(key)
        if provider in {"deepgram", "soniox"}:
            self.view_dashboard.set_stt_needs_key(True, update_ui=False)
        elif provider is not None:
            self.view_dashboard.set_translation_needs_key(True, update_ui=False)
        return True

    def _on_secret_cleared(self, key: str) -> None:
        """Reset verification status when API key is cleared."""
        # Map secret key name to provider name
        key_to_provider = {
            "deepgram_api_key": "deepgram",
            "soniox_api_key": "soniox",
            "google_api_key": "google",
            "openrouter_api_key": "openrouter",
            "deepseek_api_key": "deepseek",
            "cerebras_api_key": "cerebras",
            "alibaba_api_key": "alibaba_beijing",  # Use beijing as default
            "alibaba_api_key_beijing": "alibaba_beijing",
            "alibaba_api_key_singapore": "alibaba_singapore",
        }
        provider = key_to_provider.get(key)
        if provider:
            self.application.clear_provider_verification(provider)

            # Update dashboard needs_key flag
            if provider in ("deepgram", "soniox"):
                self.view_dashboard.set_stt_needs_key(True, update_ui=False)
            elif provider in (
                "google",
                "openrouter",
                "deepseek",
                "cerebras",
                "alibaba_beijing",
                "alibaba_singapore",
            ):
                self.view_dashboard.set_translation_needs_key(True, update_ui=False)

    def _show_snackbar(self, message: str, bgcolor, duration: int = 4000) -> None:
        """Show a snackbar above the bottom nav."""
        snackbar = ft.SnackBar(
            ft.Text(message, size=18, color=ft.Colors.WHITE),
            bgcolor=bgcolor,
            duration=duration,
            behavior=ft.SnackBarBehavior.FLOATING,
            elevation=0,
            margin=ft.Margin.only(bottom=90),
            padding=20,
        )
        self._mark_launch_high_priority_feedback_shown("snackbar", snackbar)
        self.page.show_dialog(snackbar)

    def show_snackbar(self, message: str, bgcolor) -> None:
        self._show_snackbar(message, bgcolor)

    def clear_managed_auth_pending_state(self) -> None:
        self.application.clear_managed_auth_pending_state()

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        return self.application.get_event_language_codes()

    def is_event_translation_enabled(self) -> bool:
        return self.application.state().translation_enabled

    def get_event_stt_state(self) -> object | None:
        return self.application.state().stt_state

    def on_github_star_translation_success(self) -> None:
        self.application.schedule_github_star_prompt_translation_success_observed()

    def on_telemetry_translation_success(self) -> None:
        async def _task() -> None:
            await self.application.record_telemetry_translation_success_day()

        self._queue_settings_mutation_task(_task)

    def on_overlay_state_changed(
        self,
        *,
        state: str,
        failure_reason: str | None = None,
    ) -> None:
        previous_state = getattr(self, "overlay_state", "unknown")
        self._log_basic(f"[Overlay] State changed: {previous_state} -> {state}")
        self.overlay_state = state
        self.overlay_failure_reason = failure_reason
        self._log_detailed(
            f"[Overlay] State detail: overlay_state={state} failure_reason={failure_reason}"
        )
        self._sync_settings_overlay_runtime_state()
        self.refresh_overlay_peer_contract()


async def main_gui(
    page: ft.Page,
    *,
    config_path,
    application_factory: UiApplicationFactoryPort,
    debug_ui_preview: bool = False,
    allow_stable_settings_import: bool = False,
    runtime_logging_sinks=None,
    vrchat_osc_presence=None,
):
    app = TranslatorApp(
        page,
        config_path=config_path,
        application_factory=application_factory,
        debug_ui_preview=debug_ui_preview,
        allow_stable_settings_import=allow_stable_settings_import,
        runtime_logging_sinks=runtime_logging_sinks,
        vrchat_osc_presence=vrchat_osc_presence,
    )
    page.on_disconnect = app._on_page_lifecycle_end
    page.on_close = app._on_page_lifecycle_end
    try:
        await _prepare_and_show_main_window(page)
        await app.application.start()
    except BaseException:
        with contextlib.suppress(BaseException):
            await app.shutdown()
        raise
    app.schedule_after_launch_tasks()


async def _check_and_notify_update(
    page: ft.Page,
    log_detailed=None,
    on_launch_snackbar_shown=None,
    load_update_info=None,
) -> None:
    """Check for updates and show notification as a toast."""
    try:
        if not callable(load_update_info):
            return
        update_info = await load_update_info()
        if update_info is None:
            return

        def _open_download(_e):
            webbrowser.open(update_info.download_url)
            snackbar.open = False
            page.update()

        snackbar = ft.SnackBar(
            content=ft.Row(
                controls=[
                    ft.Icon(
                        icon=ft.Icons.SYSTEM_UPDATE,
                        color=ft.Colors.WHITE,
                        size=28,
                    ),
                    ft.Text(
                        t("update.available", version=update_info.version),
                        color=ft.Colors.WHITE,
                        size=18,
                        font_family=font_for_language(get_locale()),
                        expand=True,
                    ),
                    ft.TextButton(
                        content=t("update.download"),
                        on_click=_open_download,
                        style=ft.ButtonStyle(
                            color=ft.Colors.WHITE,
                            text_style=ft.TextStyle(
                                size=18,
                                font_family=font_for_language(get_locale()),
                            ),
                            overlay_color=COLOR_PRIMARY,
                        ),
                    ),
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=12,
            ),
            bgcolor=COLOR_SUCCESS,
            behavior=ft.SnackBarBehavior.FLOATING,
            elevation=0,
            margin=ft.Margin.only(bottom=90),
            padding=20,
            duration=30000,  # 30초
            show_close_icon=True,
            close_icon_color=ft.Colors.WHITE,
        )
        page.show_dialog(snackbar)
        if callable(on_launch_snackbar_shown):
            on_launch_snackbar_shown(snackbar)

    except Exception as exc:
        message = f"[Update] Check notification failed: {exc}"
        if callable(log_detailed):
            log_detailed(message)
            return
        logger.debug(message)
