from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import OrderedDict

import flet as ft

from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterUserFacingError
from puripuly_heart.core.runtime_logging import SessionLoggingMode, SessionRuntimeLoggingService
from puripuly_heart.domain.events import STTSessionState, UIEvent, UIEventType
from puripuly_heart.domain.models import OSCMessage, Transcript, Translation
from puripuly_heart.ui.i18n import t

logger = logging.getLogger(__name__)

_FINAL_TRANSCRIPT_CACHE_LIMIT = 500


def _short_visual_debug_token(value: object | None) -> str:
    text = "" if value is None else str(value).strip()
    normalized = "".join(char for char in text if char.isalnum())
    return (normalized[:4] or "none").lower()


class UIEventBridge:
    def __init__(
        self,
        *,
        app: object,
        event_queue: asyncio.Queue[UIEvent],
        runtime_logging: SessionRuntimeLoggingService | None = None,
    ):
        self.app = app
        self.event_queue = event_queue
        self.runtime_logging = runtime_logging
        self._running = False
        self._primary_first_partial_emitted: set[str] = set()
        self._final_self_transcripts: OrderedDict[str, Transcript] = OrderedDict()

    def _get_language_codes(self) -> tuple[str | None, str | None]:
        controller = getattr(self.app, "controller", None)
        settings = getattr(controller, "settings", None)
        if settings is None:
            return None, None
        return settings.languages.source_language, settings.languages.target_language

    def _translation_enabled(self) -> bool:
        controller = getattr(self.app, "controller", None)
        hub = getattr(controller, "hub", None)
        return bool(getattr(hub, "translation_enabled", False))

    def _remember_final_self_transcript(self, transcript: Transcript) -> None:
        if transcript.channel != "self" or not transcript.is_final:
            return
        key = str(transcript.utterance_id)
        self._final_self_transcripts[key] = transcript
        self._final_self_transcripts.move_to_end(key)
        while len(self._final_self_transcripts) > _FINAL_TRANSCRIPT_CACHE_LIMIT:
            self._final_self_transcripts.popitem(last=False)

    def _source_text_for_translation(self, translation: Translation) -> str:
        source_text = translation.source_text.strip()
        if source_text:
            return source_text
        transcript = self._final_self_transcripts.get(str(translation.utterance_id))
        if transcript is None:
            return ""
        return transcript.text.strip()

    def _append_conversation_record(self, translation: Translation, *, source: str) -> None:
        if translation.channel != "self":
            return
        translated_text = translation.text.strip()
        if not translated_text:
            return
        source_text = self._source_text_for_translation(translation)
        if not source_text:
            return
        append_record = getattr(
            getattr(self.app, "view_logs", None), "append_conversation_record", None
        )
        if not callable(append_record):
            return
        try:
            append_record(
                source=source,
                channel=translation.channel,
                source_text=source_text,
                translated_text=translated_text,
                origin_wall_clock_ms=translation.origin_wall_clock_ms,
            )
        except Exception:
            logger.exception("Failed to append conversation record")

    def _visual_debug_prefix(
        self,
        *,
        channel: str | None,
        utterance_id: object | None,
        update_id: str | None = None,
    ) -> str | None:
        if channel != "peer" or utterance_id is None:
            return None
        mode = getattr(self.runtime_logging, "mode", None)
        mode_value = getattr(mode, "value", mode)
        if mode_value != SessionLoggingMode.DETAILED.value:
            return None
        turn_token = _short_visual_debug_token(utterance_id)
        stage_token = _short_visual_debug_token(update_id) if update_id else "src"
        return f"[P {turn_token}/{stage_token}]"

    def _emit_dashboard_translation_applied_detailed(
        self,
        *,
        translation: Translation,
        source_label: str,
        dashboard_target_language: str | None,
    ) -> None:
        if self.runtime_logging is None:
            return
        message = (
            "[Detailed][UIEventBridge] dashboard_translation_applied "
            f"utterance_id={translation.utterance_id} "
            f"channel={translation.channel} "
            f"source_label={json.dumps(source_label, ensure_ascii=False)} "
            f"dashboard_target_language={dashboard_target_language} "
            f"translation_target_language={translation.target_language} "
            f"text_len={len(translation.text)}"
        )
        with contextlib.suppress(Exception):
            self.runtime_logging.emit_detailed(message)

    def _schedule_github_star_prompt_translation_success(self, translation: Translation) -> None:
        if not translation.text.strip():
            return
        controller = getattr(self.app, "controller", None)
        scheduler = getattr(
            controller,
            "schedule_github_star_prompt_translation_success_observed",
            None,
        )
        if not callable(scheduler):
            return
        with contextlib.suppress(Exception):
            scheduler()

    def _schedule_translation_success_telemetry(self, translation: Translation) -> None:
        if not translation.text.strip():
            return
        controller = getattr(self.app, "controller", None)
        scheduler = getattr(controller, "schedule_translation_success_telemetry", None)
        if not callable(scheduler):
            return
        with contextlib.suppress(Exception):
            scheduler()

    def report_overlay_state(
        self,
        state: str,
        *,
        failure_reason: str | None = None,
    ) -> None:
        state_handler = getattr(self.app, "on_overlay_state_changed", None)
        if callable(state_handler):
            state_handler(state=state, failure_reason=failure_reason)

    async def run(self) -> None:
        self._running = True
        logger.info("UI Event Bridge started")
        try:
            while self._running:
                event = await self.event_queue.get()
                try:
                    await self._handle_event(event)
                except Exception:
                    logger.exception("Error handling UI event")
                finally:
                    self.event_queue.task_done()
        except asyncio.CancelledError:
            logger.info("UI Event Bridge cancelled")
            raise

    async def _handle_event(self, event: UIEvent) -> None:
        if event.type == UIEventType.SESSION_STATE_CHANGED:
            state = event.payload
            state_name = getattr(state, "name", "")
            if state_name == "CONNECTING":
                status = "connecting"
            elif state_name == "STREAMING":
                status = "connected"
            elif state_name == "DRAINING":
                status = "stopping"
            else:
                status = "disconnected"
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_status(status)
            return

        if event.type in (UIEventType.TRANSCRIPT_PARTIAL, UIEventType.TRANSCRIPT_FINAL):
            transcript = event.payload
            if not isinstance(transcript, Transcript):
                return
            source = event.source or "Mic"
            source_lang, _ = self._get_language_codes()

            is_final = event.type == UIEventType.TRANSCRIPT_FINAL
            utterance_key = str(transcript.utterance_id)
            if is_final:
                self._primary_first_partial_emitted.discard(utterance_key)
                should_log = True
                transcript_kind = "final"
            else:
                should_log = utterance_key not in self._primary_first_partial_emitted
                if should_log:
                    self._primary_first_partial_emitted.add(utterance_key)
                transcript_kind = "partial"

            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_display_text(
                    transcript.text,
                    language_code=source_lang,
                    utterance_id=transcript.utterance_id,
                    channel=transcript.channel,
                    source_text_len=len(transcript.text),
                    transcript_kind=transcript_kind,
                    should_log=should_log,
                    debug_prefix=self._visual_debug_prefix(
                        channel=transcript.channel,
                        utterance_id=transcript.utterance_id,
                    ),
                )

            if is_final:
                self._remember_final_self_transcript(transcript)
                add_history = getattr(self.app, "add_history_entry", None)
                if add_history is not None:
                    add_history(source, transcript.text, language_code=source_lang)
            return

        if event.type == UIEventType.TRANSLATION_DONE:
            translation = event.payload
            if not isinstance(translation, Translation):
                return
            source = event.source or "Mic"
            _, target_lang = self._get_language_codes()
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_display_translation_text(
                    translation.text,
                    language_code=target_lang,
                    update_id=translation.update_id,
                    origin_wall_clock_ms=translation.origin_wall_clock_ms,
                    utterance_id=translation.utterance_id,
                    channel=translation.channel,
                    session_scope=translation.session_scope,
                    source_text_hash=translation.source_text_hash,
                    source_text_len=translation.source_text_len,
                    logical_turn_key=translation.logical_turn_key,
                    debug_prefix=self._visual_debug_prefix(
                        channel=translation.channel,
                        utterance_id=translation.utterance_id,
                        update_id=translation.update_id,
                    ),
                )
                self._emit_dashboard_translation_applied_detailed(
                    translation=translation,
                    source_label=source,
                    dashboard_target_language=target_lang,
                )
            self._append_conversation_record(translation, source=source)
            add_history = getattr(self.app, "add_history_entry", None)
            if add_history is not None:
                add_history(source, translation.text, translated=True, language_code=target_lang)
            self._schedule_github_star_prompt_translation_success(translation)
            self._schedule_translation_success_telemetry(translation)
            return

        if event.type == UIEventType.OSC_SENT:
            msg = event.payload
            if not isinstance(msg, OSCMessage):
                return
            source_lang, target_lang = self._get_language_codes()
            lang_code = target_lang if self._translation_enabled() else source_lang
            add_history = getattr(self.app, "add_history_entry", None)
            if add_history is not None:
                add_history("VRChat", msg.text, language_code=lang_code)
            return

        if event.type == UIEventType.ERROR:
            payload = event.payload
            text = str(payload) if payload is not None else t("error.unknown")
            controller = getattr(self.app, "controller", None)
            try:
                if self.runtime_logging is not None:
                    if not event.runtime_log_handled:
                        self.runtime_logging.emit_basic(text, level=logging.ERROR)
                else:
                    logger.error(text)
            except Exception:
                logger.error(text)
            if isinstance(payload, ManagedOpenRouterUserFacingError):
                clear_pending = (
                    getattr(controller, "clear_managed_auth_pending_state", None)
                    if controller is not None
                    else None
                )
                if callable(clear_pending):
                    with contextlib.suppress(Exception):
                        clear_pending()
                show_snackbar = getattr(self.app, "_show_snackbar", None)
                if callable(show_snackbar):
                    with contextlib.suppress(Exception):
                        show_snackbar(text, ft.Colors.ORANGE_700)
                        return
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                msg_lower = text.lower()
                controller = getattr(self.app, "controller", None)
                hub = getattr(controller, "hub", None)
                stt = getattr(hub, "stt", None)
                stt_state = getattr(stt, "state", None)
                if (
                    "soniox" in msg_lower
                    and "400" in msg_lower
                    and stt_state in (STTSessionState.DRAINING, STTSessionState.DISCONNECTED)
                ):
                    return
                dash.set_display_text(text, is_error=True)
            return
