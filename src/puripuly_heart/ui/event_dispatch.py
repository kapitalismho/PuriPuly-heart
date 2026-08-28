from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol

import flet as ft
from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterUserFacingError

from puripuly_heart.core.diagnostic_validation import (
    DIAGNOSTIC_REDACTION_MARKER,
    DIAGNOSTIC_SINK_DASHBOARD,
    DIAGNOSTIC_SINK_SNACKBAR,
    DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
    redact_diagnostics_for_sink,
    redact_message_params_for_sink,
    redact_text_for_sink,
    redact_user_message_ref_for_sink,
)
from puripuly_heart.core.error_messages import sanitize_legacy_raw_user_visible_error_text
from puripuly_heart.core.messages import UserErrorReport, UserMessageRef
from puripuly_heart.domain.events import STTSessionState, UIEvent
from puripuly_heart.domain.models import OSCMessage, Translation
from puripuly_heart.ui.event_mapping import map_ui_event
from puripuly_heart.ui.event_projection import (
    EventProjectionContext,
    EventProjectionService,
    TranslationAppliedDiagnostic,
)
from puripuly_heart.ui.i18n import localize_user_message_ref, t

logger = logging.getLogger(__name__)

_FINAL_TRANSCRIPT_CACHE_LIMIT = 500
_RAW_STRING_ERROR_DEPRECATION_DIAGNOSTIC = (
    "[UIEventBridge] Deprecated raw string error payload sanitized for user-visible sinks"
)


class DashboardEventDestination(Protocol):
    def publish_status(self, status: str) -> None: ...

    def publish_transcript(
        self,
        text: str,
        *,
        language_code: str | None = None,
        utterance_id: object | None = None,
        channel: str | None = None,
        source_text_len: int | None = None,
        transcript_kind: str | None = None,
        should_log: bool = False,
        debug_prefix: str | None = None,
    ) -> bool | None: ...

    def publish_translation(
        self,
        text: str,
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
    ) -> bool | None: ...

    def publish_error(self, text: str) -> None: ...


class HistoryEventDestination(Protocol):
    def append_entry(
        self,
        source: str,
        text: str,
        *,
        translated: bool = False,
        language_code: str | None = None,
    ) -> None: ...


class ConversationEventDestination(Protocol):
    def append_record(
        self,
        *,
        source: str,
        channel: str,
        source_text: str,
        translated_text: str,
        origin_wall_clock_ms: int | None = None,
    ) -> None: ...


class ErrorEventDestination(Protocol):
    def publish_error(
        self,
        text: str,
        *,
        payload: object | None,
        event: UIEvent,
    ) -> bool: ...


class RuntimeLoggingPort(Protocol):
    mode: object

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None: ...

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool: ...


class AppDashboardEventDestination:
    def __init__(self, dashboard: object | None) -> None:
        self._dashboard = dashboard

    def publish_status(self, status: str) -> None:
        dashboard = self._dashboard
        if dashboard is not None:
            dashboard.set_status(status)

    def publish_transcript(
        self,
        text: str,
        *,
        language_code: str | None = None,
        utterance_id: object | None = None,
        channel: str | None = None,
        source_text_len: int | None = None,
        transcript_kind: str | None = None,
        should_log: bool = False,
        debug_prefix: str | None = None,
    ) -> bool:
        dashboard = self._dashboard
        if dashboard is None:
            return False
        dashboard.set_display_text(
            text,
            language_code=language_code,
            utterance_id=utterance_id,
            channel=channel,
            source_text_len=source_text_len,
            transcript_kind=transcript_kind,
            should_log=should_log,
            debug_prefix=debug_prefix,
        )
        return True

    def publish_translation(
        self,
        text: str,
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
    ) -> bool:
        dashboard = self._dashboard
        if dashboard is None:
            return False
        dashboard.set_display_translation_text(
            text,
            language_code=language_code,
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
        return True

    def publish_error(self, text: str) -> None:
        dashboard = self._dashboard
        if dashboard is not None:
            dashboard.set_display_text(text, is_error=True)


class AppHistoryEventDestination:
    def __init__(self, append_history_entry: object | None) -> None:
        self._append_history_entry = append_history_entry

    def append_entry(
        self,
        source: str,
        text: str,
        *,
        translated: bool = False,
        language_code: str | None = None,
    ) -> None:
        if callable(self._append_history_entry):
            self._append_history_entry(
                source, text, translated=translated, language_code=language_code
            )


class AppConversationEventDestination:
    def __init__(self, append_conversation_record: object | None) -> None:
        self._append_conversation_record = append_conversation_record

    def append_record(
        self,
        *,
        source: str,
        channel: str,
        source_text: str,
        translated_text: str,
        origin_wall_clock_ms: int | None = None,
    ) -> None:
        if callable(self._append_conversation_record):
            self._append_conversation_record(
                source=source,
                channel=channel,
                source_text=source_text,
                translated_text=translated_text,
                origin_wall_clock_ms=origin_wall_clock_ms,
            )


class AppErrorEventDestination:
    def __init__(
        self,
        *,
        runtime_logging: RuntimeLoggingPort | None,
        clear_managed_auth_pending: object | None = None,
        show_snackbar: object | None = None,
        get_stt_state: object | None = None,
    ) -> None:
        self._runtime_logging = runtime_logging
        self._clear_managed_auth_pending = clear_managed_auth_pending
        self._show_snackbar_callback = show_snackbar
        self._get_stt_state = get_stt_state

    def publish_error(
        self,
        text: str,
        *,
        payload: object | None,
        event: UIEvent,
    ) -> bool:
        self._emit_runtime_error_log(text, runtime_log_handled=event.runtime_log_handled)
        if _is_legacy_raw_error_payload(payload):
            self._emit_legacy_raw_payload_deprecation_diagnostic()
        if _is_managed_openrouter_error_payload(payload):
            self._clear_managed_auth_pending_state()
            if self._show_managed_auth_snackbar(text):
                return False
        return self._should_display_dashboard_error(text)

    def _emit_runtime_error_log(self, text: str, *, runtime_log_handled: bool) -> None:
        try:
            if self._runtime_logging is not None:
                if not runtime_log_handled:
                    self._runtime_logging.emit_basic(text, level=logging.ERROR)
            else:
                logger.error(text)
        except Exception:
            logger.error(text)

    def _emit_legacy_raw_payload_deprecation_diagnostic(self) -> None:
        emit_detailed = getattr(self._runtime_logging, "emit_detailed", None)
        if not callable(emit_detailed):
            return
        with contextlib.suppress(Exception):
            emit_detailed(_RAW_STRING_ERROR_DEPRECATION_DIAGNOSTIC, level=logging.WARNING)

    def _clear_managed_auth_pending_state(self) -> None:
        if callable(self._clear_managed_auth_pending):
            with contextlib.suppress(Exception):
                self._clear_managed_auth_pending()

    def _show_managed_auth_snackbar(self, text: str) -> bool:
        if not callable(self._show_snackbar_callback):
            return False
        with contextlib.suppress(Exception):
            self._show_snackbar_callback(text, ft.Colors.ORANGE_700)
            return True
        return False

    def _should_display_dashboard_error(self, text: str) -> bool:
        stt_state = self._get_stt_state() if callable(self._get_stt_state) else None
        msg_lower = text.lower()
        return not (
            "soniox" in msg_lower
            and "400" in msg_lower
            and stt_state in (STTSessionState.DRAINING, STTSessionState.DISCONNECTED)
        )


def _localized_error_event_text(payload: object | None) -> str:
    sink = (
        DIAGNOSTIC_SINK_SNACKBAR
        if _is_managed_openrouter_error_payload(payload)
        else DIAGNOSTIC_SINK_DASHBOARD
    )
    if isinstance(payload, ManagedOpenRouterUserFacingError):
        return t(
            payload.message_key,
            **redact_message_params_for_sink(_safe_i18n_params(payload.message_kwargs), sink),
        )
    if isinstance(payload, UserErrorReport):
        with contextlib.suppress(Exception):
            redact_diagnostics_for_sink(payload.diagnostics, sink)
        return localize_user_message_ref(redact_user_message_ref_for_sink(payload.message, sink))
    if isinstance(payload, UserMessageRef):
        return localize_user_message_ref(redact_user_message_ref_for_sink(payload, sink))
    if payload is None:
        return t("error.unknown")
    redaction = redact_text_for_sink(str(payload), sink)
    if redaction.status == DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED and redaction.text is not None:
        legacy_text = redaction.text
    else:
        legacy_text = DIAGNOSTIC_REDACTION_MARKER
    return sanitize_legacy_raw_user_visible_error_text(legacy_text) or t("error.unknown")


def _safe_i18n_params(params: Mapping[str, object]) -> dict[str, object]:
    safe_params: dict[str, object] = {}
    for key, value in params.items():
        if not isinstance(key, str) or len(key) > 64:
            continue
        if value is None or isinstance(value, str | int | float | bool):
            safe_params[key] = value
    return safe_params


def _is_managed_openrouter_error_payload(payload: object | None) -> bool:
    if isinstance(payload, ManagedOpenRouterUserFacingError):
        return True
    if isinstance(payload, UserErrorReport):
        return _is_managed_openrouter_message(payload.message)
    if isinstance(payload, UserMessageRef):
        return _is_managed_openrouter_message(payload)
    return False


def _is_legacy_raw_error_payload(payload: object | None) -> bool:
    return payload is not None and not isinstance(
        payload,
        ManagedOpenRouterUserFacingError | UserErrorReport | UserMessageRef,
    )


def _is_managed_openrouter_message(message: UserMessageRef) -> bool:
    return message.key.startswith("managed_release.")


class UIEventBridge:
    def __init__(
        self,
        *,
        event_queue: asyncio.Queue[UIEvent],
        runtime_logging: RuntimeLoggingPort | None = None,
        dashboard_destination: DashboardEventDestination,
        history_destination: HistoryEventDestination,
        conversation_destination: ConversationEventDestination,
        error_destination: ErrorEventDestination | None = None,
        get_language_codes: object | None = None,
        is_translation_enabled: object | None = None,
        get_stt_state: object | None = None,
        clear_managed_auth_pending: object | None = None,
        show_snackbar: object | None = None,
        on_github_star_translation_success: object | None = None,
        on_overlay_state_changed: object | None = None,
    ):
        self.event_queue = event_queue
        self.runtime_logging = runtime_logging
        self._get_language_codes_callback = get_language_codes
        self._is_translation_enabled_callback = is_translation_enabled
        self._get_stt_state_callback = get_stt_state
        self._github_star_translation_success_callback = on_github_star_translation_success
        self._overlay_state_changed_callback = on_overlay_state_changed
        self.dashboard_destination = dashboard_destination
        self.history_destination = history_destination
        self.conversation_destination = conversation_destination
        self.error_destination = error_destination or AppErrorEventDestination(
            runtime_logging=runtime_logging,
            clear_managed_auth_pending=clear_managed_auth_pending,
            show_snackbar=show_snackbar,
            get_stt_state=self._get_stt_state_callback,
        )
        self.projection_service = EventProjectionService(
            final_transcript_cache_limit=_FINAL_TRANSCRIPT_CACHE_LIMIT
        )
        self._running = False
        self._closed = False
        self._started = asyncio.Event()

    def _get_language_codes(self) -> tuple[str | None, str | None]:
        if callable(self._get_language_codes_callback):
            source_language, target_language = self._get_language_codes_callback()
            return source_language, target_language
        return None, None

    def _translation_enabled(self) -> bool:
        if callable(self._is_translation_enabled_callback):
            return bool(self._is_translation_enabled_callback())
        return False

    @property
    def _final_self_transcripts(self) -> object:
        return self.projection_service.final_self_transcripts

    def _projection_context(self) -> EventProjectionContext:
        source_lang, target_lang = self._get_language_codes()
        stt_state = (
            self._get_stt_state_callback() if callable(self._get_stt_state_callback) else None
        )
        return EventProjectionContext(
            source_language=source_lang,
            target_language=target_lang,
            translation_enabled=self._translation_enabled(),
            runtime_logging_mode=getattr(self.runtime_logging, "mode", None),
            stt_state=stt_state,
        )

    def _append_conversation_record_projection(self, projection: object | None) -> None:
        if self._closed or projection is None:
            return
        try:
            self.conversation_destination.append_record(
                source=projection.source,
                channel=projection.channel,
                source_text=projection.source_text,
                translated_text=projection.translated_text,
                origin_wall_clock_ms=projection.origin_wall_clock_ms,
            )
        except Exception:
            logger.error("Failed to append conversation record")

    def _emit_dashboard_translation_applied_detailed(
        self,
        *,
        diagnostic: TranslationAppliedDiagnostic,
    ) -> None:
        if self.runtime_logging is None:
            return
        message = (
            "[Detailed][UIEventBridge] dashboard_translation_applied "
            f"utterance_id={diagnostic.utterance_id} "
            f"channel={diagnostic.channel} "
            f"source_label={json.dumps(diagnostic.source_label, ensure_ascii=False)} "
            f"dashboard_target_language={diagnostic.dashboard_target_language} "
            f"translation_target_language={diagnostic.translation_target_language} "
            f"text_len={diagnostic.text_len}"
        )
        with contextlib.suppress(Exception):
            self.runtime_logging.emit_detailed(message)

    def _schedule_github_star_prompt_translation_success(self, translation: Translation) -> None:
        if not translation.text.strip():
            return
        if not callable(self._github_star_translation_success_callback):
            return
        with contextlib.suppress(Exception):
            self._github_star_translation_success_callback()

    def report_overlay_state(
        self,
        state: str,
        *,
        failure_reason: str | None = None,
    ) -> None:
        if callable(self._overlay_state_changed_callback):
            self._overlay_state_changed_callback(state=state, failure_reason=failure_reason)

    async def run(self) -> None:
        self._running = True
        self._started.set()
        logger.info("UI Event Bridge started")
        try:
            while self._running and not self._closed:
                event = await self.event_queue.get()
                try:
                    await self._handle_event(event)
                except Exception as exc:
                    logger.error(
                        "Error handling UI event: event_type=%s channel=%s exception_type=%s",
                        event.type.value,
                        event.channel,
                        type(exc).__name__,
                    )
                finally:
                    self.event_queue.task_done()
        except asyncio.CancelledError:
            logger.info("UI Event Bridge cancelled")
            raise
        finally:
            self._running = False

    async def wait_started(self) -> None:
        await self._started.wait()

    def close(self) -> None:
        self._closed = True
        self._running = False
        self.projection_service.close()

    async def _handle_event(self, event: UIEvent) -> None:
        if self._closed:
            return
        mapped = map_ui_event(event)
        if mapped is None:
            return
        if mapped.kind == "status" and mapped.status is not None:
            self.dashboard_destination.publish_status(mapped.status)
            return

        if mapped.kind == "transcript":
            projection = self.projection_service.project(mapped, self._projection_context())
            if projection.transcript is None:
                return
            transcript_projection = projection.transcript
            self.dashboard_destination.publish_transcript(
                transcript_projection.text,
                language_code=transcript_projection.language_code,
                utterance_id=transcript_projection.utterance_id,
                channel=transcript_projection.channel,
                source_text_len=transcript_projection.source_text_len,
                transcript_kind=transcript_projection.transcript_kind,
                should_log=transcript_projection.should_log,
                debug_prefix=transcript_projection.debug_prefix,
            )
            for history in projection.history:
                self.history_destination.append_entry(
                    history.source,
                    history.text,
                    translated=history.translated,
                    language_code=history.language_code,
                )
            return

        if mapped.kind == "translation":
            projection = self.projection_service.project(mapped, self._projection_context())
            if projection.translation is None:
                return
            translation_projection = projection.translation
            dashboard_published = self.dashboard_destination.publish_translation(
                translation_projection.text,
                language_code=translation_projection.language_code,
                update_id=translation_projection.update_id,
                origin_wall_clock_ms=translation_projection.origin_wall_clock_ms,
                utterance_id=translation_projection.utterance_id,
                channel=translation_projection.channel,
                session_scope=translation_projection.session_scope,
                source_text_hash=translation_projection.source_text_hash,
                source_text_len=translation_projection.source_text_len,
                logical_turn_key=translation_projection.logical_turn_key,
                debug_prefix=translation_projection.debug_prefix,
            )
            if dashboard_published is not False and projection.translation_diagnostic is not None:
                self._emit_dashboard_translation_applied_detailed(
                    diagnostic=projection.translation_diagnostic,
                )
            self._append_conversation_record_projection(projection.conversation)
            for history in projection.history:
                self.history_destination.append_entry(
                    history.source,
                    history.text,
                    translated=history.translated,
                    language_code=history.language_code,
                )
            if isinstance(mapped.payload, Translation):
                self._schedule_github_star_prompt_translation_success(mapped.payload)
            return

        if mapped.kind == "osc" and isinstance(mapped.payload, OSCMessage):
            projection = self.projection_service.project(mapped, self._projection_context())
            self.history_destination.append_entry(
                "VRChat",
                mapped.payload.text,
                language_code=projection.osc_history_language_code,
            )
            return

        if mapped.kind == "error":
            self._handle_error_event(event)

    def _handle_error_event(self, event: UIEvent) -> None:
        payload = event.payload
        text = _localized_error_event_text(payload)
        destination_payload = payload
        destination_event = event
        if _is_legacy_raw_error_payload(payload):
            destination_payload = text
            destination_event = replace(event, payload=destination_payload)
        if self.error_destination.publish_error(
            text,
            payload=destination_payload,
            event=destination_event,
        ):
            self.dashboard_destination.publish_error(text)
