import inspect
from typing import Callable

import flet as ft

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.ui_models import (
    ManagedGemmaDashboardNotice,
    ManagedGemmaNoticeAction,
)
from puripuly_heart.core.language import get_all_language_options
from puripuly_heart.ui.components.display_card import DisplayCard
from puripuly_heart.ui.components.language_card import LanguageCard
from puripuly_heart.ui.components.language_modal import LanguageModal
from puripuly_heart.ui.components.power_button import PowerButton
from puripuly_heart.ui.dashboard.capture import (
    DashboardCaptureControls,
    capture_presentation_from_contract,
)
from puripuly_heart.ui.dashboard.capture_notices import (
    gpu_capture_action_label,
    gpu_capture_notice,
    local_asr_capture_notice,
    managed_gemma_action_label,
    managed_gemma_capture_notice,
)
from puripuly_heart.ui.dashboard.contract import (
    DashboardCaptureIntents,
    DashboardSurfaceSlots,
    DashboardTranslationIntents,
)
from puripuly_heart.ui.dashboard.renderer import (
    DASHBOARD_CONTROL_REGION_EXPAND,
    DASHBOARD_DISPLAY_CARD_EXPAND,
    DASHBOARD_INFO_REGION_EXPAND,
    DASHBOARD_LANGUAGE_CARD_EXPAND,
    DASHBOARD_LAYOUT_GAP,
    DASHBOARD_POWER_BUTTON_ICON_SIZE,
    DASHBOARD_POWER_BUTTON_LABEL_SIZE,
    DASHBOARD_SHELL_SPACING,
    compose_dashboard_surface,
)
from puripuly_heart.ui.flet_runtime import control_page
from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.gpu_notice import GpuDashboardNotice, GpuNoticeAction
from puripuly_heart.ui.i18n import get_locale, language_name, t
from puripuly_heart.ui.overlay_peer_contract import OverlayPeerConsumerContract

OVERLAY_FAILURE_REASON_ONLY_NOTICE_REASONS = {"steamvr_not_running"}

__all__ = [
    "DASHBOARD_CONTROL_REGION_EXPAND",
    "DASHBOARD_DISPLAY_CARD_EXPAND",
    "DASHBOARD_INFO_REGION_EXPAND",
    "DASHBOARD_LANGUAGE_CARD_EXPAND",
    "DASHBOARD_LAYOUT_GAP",
    "DASHBOARD_POWER_BUTTON_ICON_SIZE",
    "DASHBOARD_POWER_BUTTON_LABEL_SIZE",
    "DASHBOARD_SHELL_SPACING",
    "DashboardView",
]
OVERLAY_YIELDING_NOTICE_SOURCES = {"overlay_fallback", "overlay_failure"}
PEER_SOURCE_MODE_AUTO = "auto"


class DashboardView(ft.Column):
    """Main dashboard tuned for the 4:3 VR-friendly shell layout."""

    _LANG_OPTIONS = get_all_language_options()

    def __init__(self):
        super().__init__(expand=True, spacing=DASHBOARD_SHELL_SPACING)

        # State
        self.is_connected = False
        self._connection_status = "disconnected"
        self.is_power_on = False
        self.is_translation_on = False
        self.is_stt_on = False
        self._stt_is_starting = False
        self._translation_is_starting = False
        self.translation_needs_key = False
        self.stt_needs_key = False
        self.last_sent_text = t("dashboard.ready")
        self.history_items = []

        # Warning state for UI feedback
        self._translation_showing_warning = False
        self._stt_showing_warning = False
        self._managed_auth_pending = False
        self._local_stt_notice_status: str | None = None
        self._local_stt_notice_percent: int | None = None
        self._local_stt_notice_model_id: str | None = None
        self._gpu_notice: GpuDashboardNotice | None = None
        self._managed_gemma_notice: ManagedGemmaDashboardNotice | None = None
        self._notice_sequence = 0
        self._notice_started: dict[str, int] = {}
        self._visible_notice_source: str | None = None
        self._vrchat_osc_notice_active = False
        self._overlay_session_fallback_notice_active = False
        self._overlay_peer_contract: OverlayPeerConsumerContract | None = None
        self._process_capture_warning_active = False
        self._process_capture_warning_reason: str | None = None
        self._process_capture_warning_locale: str | None = None
        self._process_capture_warning_text = ""
        self._current_display_text: str | None = None
        self._primary_display_revision = 0
        self._process_capture_warning_display_revision: int | None = None

        # Current language settings
        self._source_lang_code = "ko"
        self._target_lang_code = "en"
        self._peer_source_lang_code = "en"
        self._peer_target_lang_code = "ko"
        self._peer_source_mode = "manual"
        self._peer_auto_detect_available = True
        self._message_input_focused = False

        # Recent languages (max 3 each)
        self._recent_source_langs: list[str] = []
        self._recent_target_langs: list[str] = []

        # Callbacks (assigned by App)
        self.on_send_message = None
        self.on_toggle_translation = None
        self.on_toggle_stt = None
        self.on_toggle_overlay = None
        self.on_toggle_peer_translation = None
        self.on_retry_peer_process_capture = None
        self.on_gpu_notice_action: Callable[[GpuNoticeAction], object] | None = None
        self.on_managed_gemma_notice_action: Callable[[ManagedGemmaNoticeAction], object] | None = (
            None
        )
        self.on_language_change: Callable[[LanguageSelectionChange], None] | None = None
        self.on_message_input_activity = None
        self.runtime_log_detailed: Callable[..., bool | None] | None = None

        self._build_ui()

    def _build_ui(self):
        # Left-side control grid
        self._capture_controls = DashboardCaptureControls(
            on_self_capture_click=self._toggle_stt,
            on_peer_capture_click=self._toggle_peer_translation,
            on_overlay_click=self._toggle_overlay,
        )
        self.trans_button = PowerButton(
            label=t("dashboard.trans_label"),
            icon=ft.Icons.TRANSLATE,
            on_click=self._toggle_translation,
            icon_size=DASHBOARD_POWER_BUTTON_ICON_SIZE,
            label_size=DASHBOARD_POWER_BUTTON_LABEL_SIZE,
        )
        self._sync_stt_button_state()
        self._sync_translation_button_state()
        self._sync_overlay_peer_buttons()

        # Right-side information stack
        self.display_card = DisplayCard(
            on_submit=self._on_submit,
            on_input_focus_change=self._set_message_input_focused,
            on_input_activity=self._on_message_input_activity,
        )
        self.language_card = LanguageCard(
            on_self_source_click=self._open_source_dialog,
            on_self_target_click=self._open_target_dialog,
            on_self_swap_click=self._swap_languages,
            on_peer_source_click=self._open_peer_source_dialog,
            on_peer_target_click=self._open_peer_target_dialog,
            on_peer_swap_click=self._swap_peer_languages,
        )
        self.language_card.set_row_labels(
            t("dashboard.language.self"),
            t("dashboard.language.peer"),
        )
        self._refresh_language_card()
        self._update_input_font()

        surface = compose_dashboard_surface(
            DashboardSurfaceSlots.from_capture_provider(
                self._capture_controls,
                translation=self.trans_button,
                display=self.display_card,
                language=self.language_card,
            )
        )
        self.top_controls = surface.top_controls
        self.bottom_controls = surface.bottom_controls
        self.control_grid = surface.control_grid
        self.display_card_slot = surface.display_card_slot
        self.language_card_slot = surface.language_card_slot
        self.info_stack = surface.info_stack
        self.control_region = surface.control_region
        self.info_region = surface.info_region
        self.main_surface = surface.main_surface
        self.shell_content = surface.shell_content
        self.controls = [surface.root]

    @property
    def stt_button(self) -> ft.Control:
        return self._capture_controls.self_capture_control()

    @property
    def peer_button(self) -> ft.Control:
        return self._capture_controls.peer_capture_control()

    @property
    def overlay_button(self) -> ft.Control:
        return self._capture_controls.overlay_control()

    def self_capture_control(self) -> ft.Control:
        return self._capture_controls.self_capture_control()

    def peer_capture_control(self) -> ft.Control:
        return self._capture_controls.peer_capture_control()

    def overlay_control(self) -> ft.Control:
        return self._capture_controls.overlay_control()

    def bind_dashboard_intents(
        self,
        *,
        translation: DashboardTranslationIntents,
        capture: DashboardCaptureIntents,
    ) -> None:
        self.on_send_message = translation.submit_message
        self.on_toggle_translation = translation.toggle_translation
        self.on_language_change = translation.change_language
        self.on_message_input_activity = translation.report_input_activity
        self.on_toggle_stt = capture.toggle_self_capture
        self.on_toggle_peer_translation = capture.toggle_peer_capture
        self.on_toggle_overlay = capture.toggle_overlay
        self.on_retry_peer_process_capture = capture.retry_peer_process_capture
        self.on_gpu_notice_action = capture.run_gpu_notice_action
        self.on_managed_gemma_notice_action = capture.run_managed_gemma_notice_action

    def _toggle_overlay(self) -> None:
        enabled = True
        if self._overlay_peer_contract is not None:
            enabled = not self._overlay_peer_contract.overlay.intent_enabled
        if self.on_toggle_overlay:
            self.on_toggle_overlay(enabled)

    def _toggle_peer_translation(self) -> None:
        contract = self._overlay_peer_contract
        enabled = True
        if contract is not None:
            enabled = not contract.peer.intent_enabled
        if self.on_toggle_peer_translation:
            self.on_toggle_peer_translation(enabled)

    def _sync_stt_button_state(self) -> None:
        self._capture_controls.apply_self_capture_state(
            enabled=self.is_stt_on,
            starting=self._stt_is_starting,
            warning=self._stt_showing_warning,
        )

    def _sync_translation_button_state(self) -> None:
        self.trans_button.set_state(
            self.is_translation_on,
            needs_key=self._translation_showing_warning,
            is_starting=self._translation_is_starting,
        )

    def _sync_overlay_peer_buttons(self) -> None:
        self._capture_controls.apply_presentation(
            capture_presentation_from_contract(self._overlay_peer_contract)
        )
        self._sync_notice()

    def _restore_status_display(self) -> None:
        status = getattr(self, "_connection_status", "disconnected") or "disconnected"
        if not hasattr(self, "display_card"):
            self._connection_status = status
            self.is_connected = status == "connected"
            self.set_display_text("")
            return
        self.set_status(status)

    def _dismiss_api_key_warning_display(self) -> None:
        if self._stt_showing_warning:
            self.set_display_text(t("dashboard.warn_stt_key"))
            return
        if self._translation_showing_warning:
            self.set_display_text(t("dashboard.warn_llm_key"))
            return
        if self._process_capture_warning_active and self._process_capture_warning_text:
            self.set_display_text(self._process_capture_warning_text)
            return
        self._restore_status_display()

    def _toggle_stt(self):
        if self._stt_is_starting or self.is_stt_on:
            self.is_stt_on = False
            self._stt_is_starting = False
            self._stt_showing_warning = False
        elif self._stt_showing_warning:
            self._stt_showing_warning = False
            self._dismiss_api_key_warning_display()
        elif self.stt_needs_key:
            self._stt_showing_warning = True
            self.set_display_text(t("dashboard.warn_stt_key"))
        else:
            self.is_stt_on = True
            self._stt_is_starting = True
            self._stt_showing_warning = False

        self._sync_stt_button_state()

        if self.on_toggle_stt:
            self.on_toggle_stt(self.is_stt_on)

    def _toggle_translation(self):
        if self.is_translation_on:
            self.is_translation_on = False
            self._translation_showing_warning = False
        elif self._translation_showing_warning:
            self._translation_showing_warning = False
            self._dismiss_api_key_warning_display()
        elif self.translation_needs_key:
            self._translation_showing_warning = True
            self.set_display_text(t("dashboard.warn_llm_key"))
        else:
            self.is_translation_on = True
            self._translation_showing_warning = False

        self._sync_translation_button_state()

        self.is_power_on = self.is_translation_on
        if self.on_toggle_translation:
            self.on_toggle_translation(self.is_translation_on)

    def _on_submit(self, text: str):
        self.set_display_text(text, language_code=self._source_lang_code)
        if self.on_send_message:
            self.on_send_message("You", text)

    def _set_message_input_focused(self, focused: bool) -> None:
        self._message_input_focused = bool(focused)

    def _on_message_input_activity(self, has_text: bool) -> None:
        if self.on_message_input_activity:
            self.on_message_input_activity(bool(has_text))

    def handle_message_input_tab_key(self) -> bool:
        if not self._message_input_focused:
            return False

        self._swap_languages()
        self.display_card.focus_input()
        return True

    def _open_source_dialog(self):
        modal = LanguageModal(
            page=self.page,
            languages=self._LANG_OPTIONS,
            on_select=self._on_source_select,
        )
        modal.open(current=self._source_lang_code, recent=self._recent_source_langs)

    def _open_target_dialog(self):
        modal = LanguageModal(
            page=self.page,
            languages=self._LANG_OPTIONS,
            on_select=self._on_target_select,
        )
        modal.open(current=self._target_lang_code, recent=self._recent_target_langs)

    def _open_peer_source_dialog(self):
        modal = LanguageModal(
            page=self.page,
            languages=((PEER_SOURCE_MODE_AUTO, ""), *self._LANG_OPTIONS),
            on_select=self._on_peer_source_select,
            label_for_code=lambda code: (
                t("dashboard.peer_source.automatic")
                if code == PEER_SOURCE_MODE_AUTO
                else language_name(code)
            ),
            description_for_code=lambda code: (
                t("dashboard.peer_source.automatic.description")
                if code == PEER_SOURCE_MODE_AUTO
                else ""
            ),
            disabled_codes=(set() if self._peer_auto_detect_available else {PEER_SOURCE_MODE_AUTO}),
        )
        modal.open(
            current=(
                PEER_SOURCE_MODE_AUTO
                if self._peer_source_mode == PEER_SOURCE_MODE_AUTO
                else self._effective_peer_source_lang_code()
            ),
            recent=self._recent_source_langs,
        )

    def _open_peer_target_dialog(self):
        modal = LanguageModal(
            page=self.page,
            languages=self._LANG_OPTIONS,
            on_select=self._on_peer_target_select,
        )
        modal.open(
            current=self._effective_peer_target_lang_code(), recent=self._recent_target_langs
        )

    def _on_source_select(self, lang_code: str):
        """Handle source language selection."""
        self._source_lang_code = lang_code
        self._add_to_recent(lang_code, is_source=True)
        self._update_input_font()
        self._refresh_language_card()
        self._notify_language_change()

    def _on_target_select(self, lang_code: str):
        """Handle target language selection."""
        self._target_lang_code = lang_code
        self._add_to_recent(lang_code, is_source=False)
        self._refresh_language_card()
        self._notify_language_change()

    def _on_peer_source_select(self, lang_code: str):
        if lang_code == PEER_SOURCE_MODE_AUTO:
            self._peer_source_mode = PEER_SOURCE_MODE_AUTO
            self._refresh_language_card()
            self._notify_language_change()
            return
        self._peer_source_mode = "manual"
        self._peer_source_lang_code = "" if lang_code == self._source_lang_code else lang_code
        self._add_to_recent(lang_code, is_source=True)
        self._refresh_language_card()
        self._notify_language_change()

    def _on_peer_target_select(self, lang_code: str):
        self._peer_target_lang_code = "" if lang_code == self._target_lang_code else lang_code
        self._add_to_recent(lang_code, is_source=False)
        self._refresh_language_card()
        self._notify_language_change()

    def _swap_languages(self):
        """Swap source and target languages."""
        self._source_lang_code, self._target_lang_code = (
            self._target_lang_code,
            self._source_lang_code,
        )
        self._update_input_font()
        self._refresh_language_card()
        self._notify_language_change()

    def _swap_peer_languages(self):
        if self._peer_source_mode == PEER_SOURCE_MODE_AUTO:
            manual_peer_source = self._peer_source_lang_code or self._source_lang_code
            self._peer_source_lang_code = self._effective_peer_target_lang_code()
            self._peer_target_lang_code = manual_peer_source
            self._refresh_language_card()
            self._notify_language_change()
            return
        current_peer_source = self._effective_peer_source_lang_code()
        current_peer_target = self._effective_peer_target_lang_code()
        self._peer_source_lang_code = current_peer_target
        self._peer_target_lang_code = current_peer_source
        self._refresh_language_card()
        self._notify_language_change()

    def _add_to_recent(self, lang_code: str, is_source: bool) -> None:
        """Add language to recent list, maintaining max 6 unique entries."""
        recent = self._recent_source_langs if is_source else self._recent_target_langs
        if lang_code in recent:
            recent.remove(lang_code)
        recent.insert(0, lang_code)
        if len(recent) > 6:
            recent.pop()

    def _notify_language_change(self):
        if self.on_language_change:
            self.on_language_change(
                LanguageSelectionChange(
                    source_code=self._source_lang_code,
                    target_code=self._target_lang_code,
                    peer_source_code=self._peer_source_lang_code,
                    peer_target_code=self._peer_target_lang_code,
                    peer_source_mode=self._peer_source_mode,
                    recent_source_codes=tuple(self._recent_source_langs),
                    recent_target_codes=tuple(self._recent_target_langs),
                )
            )

    def _effective_peer_source_lang_code(self) -> str:
        if self._peer_source_mode == PEER_SOURCE_MODE_AUTO:
            return PEER_SOURCE_MODE_AUTO
        return self._peer_source_lang_code or self._source_lang_code

    def _effective_peer_target_lang_code(self) -> str:
        return self._peer_target_lang_code or self._target_lang_code

    def _refresh_language_card(self) -> None:
        self.language_card.set_languages(
            language_name(self._source_lang_code),
            language_name(self._target_lang_code),
            (
                t("dashboard.peer_source.automatic")
                if self._peer_source_mode == PEER_SOURCE_MODE_AUTO
                else language_name(self._effective_peer_source_lang_code())
            ),
            language_name(self._effective_peer_target_lang_code()),
        )

    def set_status(self, status: str) -> None:
        self._connection_status = status
        self.is_connected = status == "connected"
        self._primary_display_revision += 1
        self._current_display_text = None
        self.display_card.set_status(status, font_family=self._ui_font())

    def set_languages_from_codes(
        self,
        source_code: str,
        target_code: str,
        peer_source_code: str = "",
        peer_target_code: str = "",
        peer_source_mode: str = "manual",
    ) -> None:
        self._source_lang_code = source_code
        self._target_lang_code = target_code
        self._peer_source_lang_code = peer_source_code
        self._peer_target_lang_code = peer_target_code
        self._peer_source_mode = peer_source_mode
        self._update_input_font()
        self._refresh_language_card()

    def set_peer_auto_detect_available(self, available: bool) -> None:
        self._peer_auto_detect_available = bool(available)

    def set_translation_enabled(self, enabled: bool) -> None:
        self.is_translation_on = bool(enabled)
        if self.is_translation_on:
            self._translation_showing_warning = False
        self._sync_translation_button_state()

    def set_stt_enabled(self, enabled: bool) -> None:
        self.is_stt_on = bool(enabled)
        self._stt_is_starting = False
        if self.is_stt_on:
            self._stt_showing_warning = False
        self._sync_stt_button_state()

    def set_stt_starting(self, starting: bool) -> None:
        self._stt_is_starting = bool(starting)
        self._sync_stt_button_state()

    def set_translation_starting(self, starting: bool) -> None:
        self._translation_is_starting = bool(starting)
        self._sync_translation_button_state()

    def set_overlay_peer_contract(self, contract: OverlayPeerConsumerContract) -> None:
        self._overlay_peer_contract = contract
        self._sync_overlay_peer_buttons()
        presentation = capture_presentation_from_contract(contract)
        if presentation.process_capture_warning_active:
            warning_changed = (
                not self._process_capture_warning_active
                or self._process_capture_warning_reason
                != presentation.process_capture_warning_reason
                or (
                    self._process_capture_warning_text != presentation.process_capture_warning_text
                    and self._process_capture_warning_locale == get_locale()
                )
            )
            self._process_capture_warning_active = True
            self._process_capture_warning_reason = presentation.process_capture_warning_reason
            if warning_changed:
                self._process_capture_warning_text = presentation.process_capture_warning_text
                self._process_capture_warning_locale = get_locale()
                self.set_display_text(presentation.process_capture_warning_text)
                self._process_capture_warning_display_revision = self._primary_display_revision
            return
        if self._process_capture_warning_active:
            warning_text = self._process_capture_warning_text
            self._process_capture_warning_active = False
            self._process_capture_warning_reason = None
            self._process_capture_warning_locale = None
            self._process_capture_warning_text = ""
            warning_display_revision = self._process_capture_warning_display_revision
            self._process_capture_warning_display_revision = None
            if (
                self._current_display_text == warning_text
                and self._primary_display_revision == warning_display_revision
            ):
                self._restore_status_display()

    def set_translation_needs_key(self, needs_key: bool, *, update_ui: bool = True) -> None:
        self.translation_needs_key = bool(needs_key)
        if update_ui and not self.is_translation_on:
            self._translation_showing_warning = bool(needs_key)
            self._sync_translation_button_state()

    def set_stt_needs_key(self, needs_key: bool, *, update_ui: bool = True) -> None:
        self.stt_needs_key = bool(needs_key)
        if update_ui and not self.is_stt_on:
            self._stt_showing_warning = bool(needs_key)
            self._sync_stt_button_state()

    def set_display_text(
        self,
        text: str,
        *,
        language_code: str | None = None,
        is_error: bool = False,
        update_id: str | None = None,
        origin_wall_clock_ms: int | None = None,
        utterance_id: object | None = None,
        channel: str | None = None,
        source_text_len: int | None = None,
        transcript_kind: str | None = None,
        should_log: bool = False,
        debug_prefix: str | None = None,
    ) -> None:
        """Update the display card primary line with new text."""
        self._primary_display_revision += 1
        self._current_display_text = text
        font_family = font_for_language(language_code) if language_code else self._ui_font()
        self.display_card.set_display(
            text,
            is_error=is_error,
            font_family=font_family,
            runtime_log_detailed=self.runtime_log_detailed,
            update_id=update_id,
            origin_wall_clock_ms=origin_wall_clock_ms,
            utterance_id=utterance_id,
            channel=channel,
            source_text_len=source_text_len,
            transcript_kind=transcript_kind,
            should_log=should_log,
            debug_prefix=debug_prefix,
        )

    def set_display_translation_text(
        self,
        text: str | None,
        *,
        language_code: str | None = None,
        update_id: str | None = None,
        origin_wall_clock_ms: int | None = None,
        utterance_id: object | None = None,
        channel: str | None = None,
        session_scope: str | None = None,
        source_text_hash: str | None = None,
        source_text_len: int | None = None,
        logical_turn_key: str | None = None,
        debug_prefix: str | None = None,
    ) -> None:
        """Update the display card translation line."""
        font_family = font_for_language(language_code) if language_code else self._ui_font()
        self.display_card.set_display_translation(
            text,
            font_family=font_family,
            runtime_log_detailed=self.runtime_log_detailed,
            update_id=update_id,
            origin_wall_clock_ms=origin_wall_clock_ms,
            utterance_id=utterance_id,
            channel=channel,
            session_scope=session_scope,
            source_text_hash=source_text_hash,
            source_text_len=source_text_len,
            logical_turn_key=logical_turn_key,
            debug_prefix=debug_prefix,
        )

    def set_managed_auth_pending(self, pending: bool) -> None:
        self._managed_auth_pending = bool(pending)
        self._sync_notice()

    def set_local_stt_notice(self, status: str | None, percent: int | None = None) -> None:
        self._local_stt_notice_status = status
        self._local_stt_notice_percent = percent if status == "downloading" else None

        self._sync_notice()

    def set_local_stt_notice_model(self, model_id: str | None) -> None:
        self._local_stt_notice_model_id = model_id
        self._sync_notice()

    def set_gpu_notice(self, notice: GpuDashboardNotice | None) -> None:
        self._gpu_notice = notice
        self._sync_notice()

    def set_managed_gemma_notice(
        self,
        notice: ManagedGemmaDashboardNotice | None,
    ) -> None:
        self._managed_gemma_notice = notice
        self._sync_notice()

    def _run_gpu_notice_action(self, action: GpuNoticeAction) -> None:
        self._run_notice_action(self.on_gpu_notice_action, action)

    def _run_notice_action(
        self,
        callback: Callable[..., object] | None,
        action: str,
    ) -> None:
        page = control_page(self)
        run_task = getattr(page, "run_task", None)
        if callback is None or not callable(run_task):
            return

        async def invoke() -> None:
            result = callback(action)
            if inspect.isawaitable(result):
                await result

        run_task(invoke)

    def _run_managed_gemma_notice_action(
        self,
        action: ManagedGemmaNoticeAction,
    ) -> None:
        self._run_notice_action(self.on_managed_gemma_notice_action, action)

    def set_vrchat_osc_notice(self, active: bool) -> None:
        self._vrchat_osc_notice_active = bool(active)
        self._sync_notice()

    def set_overlay_session_fallback_notice(self, active: bool) -> None:
        self._overlay_session_fallback_notice_active = bool(active)
        self._sync_notice()

    def _current_local_stt_notice(self) -> tuple[str | None, str | None]:
        notice = local_asr_capture_notice(
            status=self._local_stt_notice_status,
            percent=self._local_stt_notice_percent,
            model_id=self._local_stt_notice_model_id,
        )
        if notice is None:
            return None, None
        return notice.text, notice.tone

    def _current_overlay_failure_notice(self) -> tuple[str | None, str | None]:
        contract = self._overlay_peer_contract
        if contract is None:
            return None, None

        overlay = contract.overlay
        if overlay.state != "warning" or not overlay.failure_reason:
            return None, None

        status_text = t("settings.overlay.status.failed", default="failed")
        reason_text = t(
            f"settings.overlay.failure.{overlay.failure_reason}",
            default=overlay.failure_reason,
        )
        if overlay.failure_reason in OVERLAY_FAILURE_REASON_ONLY_NOTICE_REASONS:
            return reason_text, "error"
        return (
            t(
                "settings.overlay.status.failed_with_reason",
                status=status_text,
                reason=reason_text,
                default=f"{status_text}: {reason_text}",
            ),
            "error",
        )

    def _notice_candidates(self) -> dict[str, tuple[str, str | None, str | None]]:
        candidates: dict[str, tuple[str, str | None, str | None]] = {}
        if self._managed_auth_pending:
            candidates["managed_auth"] = (t("dashboard.managed_auth_pending"), "info", None)
        notice_text, tone = self._current_local_stt_notice()
        if notice_text is not None:
            candidates["local_stt"] = (notice_text, tone, None)
        gpu_notice = gpu_capture_notice(self._gpu_notice)
        if gpu_notice is not None:
            candidates["gpu"] = (gpu_notice.text, gpu_notice.tone, gpu_notice.action)
        managed_gemma_notice = managed_gemma_capture_notice(self._managed_gemma_notice)
        if managed_gemma_notice is not None:
            candidates["managed_gemma"] = (
                managed_gemma_notice.text,
                managed_gemma_notice.tone,
                self._managed_gemma_notice.action,
            )
        if self._overlay_session_fallback_notice_active:
            candidates["overlay_fallback"] = (
                t("dashboard.overlay_session_fallback_desktop"),
                "info",
                None,
            )
        if self._vrchat_osc_notice_active:
            candidates["vrchat_osc"] = (t("dashboard.vrchat_osc_disabled"), "warning", None)
        overlay_text, overlay_tone = self._current_overlay_failure_notice()
        if overlay_text is not None:
            candidates["overlay_failure"] = (overlay_text, overlay_tone, None)
        return candidates

    def _sync_notice(self) -> None:
        if not hasattr(self, "display_card"):
            return
        candidates = self._notice_candidates()
        for source in tuple(self._notice_started):
            if source not in candidates:
                self._notice_started.pop(source, None)
        for source in candidates:
            if source not in self._notice_started:
                self._notice_sequence += 1
                self._notice_started[source] = self._notice_sequence

        download_source = None
        if self._managed_gemma_notice is not None and self._managed_gemma_notice.status in {
            "checking",
            "downloading",
            "preparing",
        }:
            download_source = "managed_gemma"
        elif self._gpu_notice is not None and self._gpu_notice.status == "installing":
            download_source = "gpu"
        elif self._local_stt_notice_status == "downloading":
            download_source = "local_stt"
        if download_source is not None:
            selected = download_source
        elif self._visible_notice_source in candidates:
            selected = self._visible_notice_source
        else:
            selected = min(candidates, key=self._notice_started.__getitem__) if candidates else None
        self._visible_notice_source = selected
        if selected is None:
            self.display_card.set_notice(None, None)
            return
        text, tone, _action = candidates[selected]
        if selected == "managed_gemma":
            managed_action = (
                None if self._managed_gemma_notice is None else self._managed_gemma_notice.action
            )
            action_label = managed_gemma_action_label(managed_action)
            on_action = (
                None
                if managed_action is None
                else lambda: self._run_managed_gemma_notice_action(managed_action)
            )
        else:
            gpu_action = None if self._gpu_notice is None else self._gpu_notice.action
            action_label = gpu_capture_action_label(gpu_action if selected == "gpu" else None)
            on_action = (
                None
                if selected != "gpu" or gpu_action is None
                else lambda: self._run_gpu_notice_action(gpu_action)
            )
        yields_to_content = selected in OVERLAY_YIELDING_NOTICE_SOURCES
        try:
            self.display_card.set_notice(
                text,
                tone,
                action_label=action_label,
                on_action=on_action,
                yields_to_content=yields_to_content,
            )
        except TypeError:
            self.display_card.set_notice(text, tone)

    def apply_locale(self) -> None:
        self._capture_controls.apply_locale()
        self.trans_button.set_label(t("dashboard.trans_label"))
        self._sync_stt_button_state()
        self._sync_translation_button_state()
        self._sync_overlay_peer_buttons()
        self.display_card.apply_locale(
            display_font_family=self._ui_font(),
            input_font_family=font_for_language(self._source_lang_code),
        )
        self.language_card.set_row_labels(
            t("dashboard.language.self"),
            t("dashboard.language.peer"),
        )
        self._refresh_language_card()
        if not self._process_capture_warning_active and self._stt_showing_warning:
            self.set_display_text(t("dashboard.warn_stt_key"))
        elif not self._process_capture_warning_active and self._translation_showing_warning:
            self.set_display_text(t("dashboard.warn_llm_key"))

    def set_recent_languages(self, source: list[str], target: list[str]) -> None:
        """Set recent languages from settings (for persistence)."""
        self._recent_source_langs = list(source)
        self._recent_target_langs = list(target)
        self._recent_source_langs = self._recent_source_langs[:6]
        self._recent_target_langs = self._recent_target_langs[:6]

    def _update_input_font(self) -> None:
        self.display_card.set_input_font(font_for_language(self._source_lang_code))

    def _ui_font(self) -> str | None:
        return font_for_language(get_locale())
