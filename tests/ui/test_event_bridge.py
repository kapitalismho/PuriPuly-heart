from __future__ import annotations

import asyncio
import inspect
import io
import json
import logging
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch
from uuid import uuid4

import flet as ft
import pytest

pytest.importorskip("flet")

from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseDiagnostics,
    ManagedOpenRouterUserFacingError,
)

from puripuly_heart.core import messages
from puripuly_heart.core.runtime_logging import SessionLoggingMode, SessionRuntimeLoggingService
from puripuly_heart.domain.events import STTSessionState, UIEvent, UIEventType
from puripuly_heart.domain.models import OSCMessage, Transcript, Translation
from puripuly_heart.ui import event_dispatch as event_dispatch_module
from puripuly_heart.ui.event_bridge import (
    AppConversationEventDestination,
    AppDashboardEventDestination,
    AppHistoryEventDestination,
    UIEventBridge,
)
from puripuly_heart.ui.event_mapping import map_ui_event
from puripuly_heart.ui.event_projection import EventProjectionContext, EventProjectionService
from puripuly_heart.ui.i18n import get_locale, set_locale, t
from puripuly_heart.ui.views import logs as logs_view_module
from puripuly_heart.ui.views.logs import FletLogHandler, LogsView
from tests.helpers.ui_application import compose_test_ui_application_boundary

assert logs_view_module.LogsView is LogsView


class DummyDashboard:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.display_calls: list[tuple[str, str | None, bool]] = []
        self.display_debug_prefixes: list[str | None] = []
        self.translation_calls: list[tuple[str, str | None]] = []
        self.translation_metadata_calls: list[dict[str, object]] = []
        self.notice_calls: list[str | None] = []

    def set_status(self, status: str) -> None:
        self.statuses.append(status)

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
        self.display_calls.append((text, language_code, is_error))
        self.display_debug_prefixes.append(debug_prefix)

    def set_display_translation_text(
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
    ) -> None:
        self.translation_calls.append((text, language_code))
        self.translation_metadata_calls.append(
            {
                "update_id": update_id,
                "origin_wall_clock_ms": origin_wall_clock_ms,
                "utterance_id": utterance_id,
                "channel": channel,
                "session_scope": session_scope,
                "source_text_hash": source_text_hash,
                "source_text_len": source_text_len,
                "logical_turn_key": logical_turn_key,
                "debug_prefix": debug_prefix,
            }
        )

    def set_local_stt_notice(self, status: str | None) -> None:
        self.notice_calls.append(status)


class FailingTranslationDashboard(DummyDashboard):
    def set_display_translation_text(
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
    ) -> None:
        _ = (
            text,
            language_code,
            update_id,
            origin_wall_clock_ms,
            utterance_id,
            channel,
            session_scope,
            source_text_hash,
            source_text_len,
            logical_turn_key,
            debug_prefix,
        )
        raise RuntimeError("dashboard setter failed")


class DummyLogs:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.conversation_records: list[dict[str, object]] = []

    def append_log(self, line: str) -> None:
        self.lines.append(line)

    def append_conversation_record(
        self,
        *,
        source: str,
        channel: str,
        source_text: str,
        translated_text: str,
        origin_wall_clock_ms: int | None = None,
    ) -> None:
        self.conversation_records.append(
            {
                "source": source,
                "channel": channel,
                "source_text": source_text,
                "translated_text": translated_text,
                "origin_wall_clock_ms": origin_wall_clock_ms,
            }
        )


class FailingConversationLogs(DummyLogs):
    def append_conversation_record(
        self,
        *,
        source: str,
        channel: str,
        source_text: str,
        translated_text: str,
        origin_wall_clock_ms: int | None = None,
    ) -> None:
        _ = (source, channel, source_text, translated_text, origin_wall_clock_ms)
        raise RuntimeError("conversation append failed")


class DummyApp:
    def __init__(self) -> None:
        self.view_dashboard = DummyDashboard()
        self.view_logs = DummyLogs()
        self.snackbar_calls: list[tuple[str, object]] = []
        self.clear_managed_auth_pending_calls = 0
        self.history: list[tuple[str, str, bool, str | None]] = []
        self.overlay_state = "off"
        self.overlay_failure_reason: str | None = None
        self.controller = SimpleNamespace(
            settings=SimpleNamespace(
                languages=SimpleNamespace(source_language="ko", target_language="en")
            ),
            hub=SimpleNamespace(
                translation_enabled=False,
                stt=SimpleNamespace(state=STTSessionState.STREAMING),
            ),
            get_event_language_codes=lambda: ("ko", "en"),
            managed_auth_pending=False,
            clear_managed_auth_pending_state=lambda: self._record_clear_managed_auth_pending(),
        )

    def _record_clear_managed_auth_pending(self) -> None:
        self.clear_managed_auth_pending_calls += 1
        self.controller.managed_auth_pending = False

    def clear_managed_auth_pending_state(self) -> None:
        self._record_clear_managed_auth_pending()

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        return self.controller.get_event_language_codes()

    def is_event_translation_enabled(self) -> bool:
        return bool(self.controller.hub.translation_enabled)

    def get_event_stt_state(self) -> STTSessionState | None:
        return self.controller.hub.stt.state

    def on_github_star_translation_success(self) -> None:
        scheduler = getattr(
            self.controller,
            "schedule_github_star_prompt_translation_success_observed",
            None,
        )
        if callable(scheduler):
            scheduler()

    def _show_snackbar(self, message: str, bgcolor, duration: int = 4000) -> None:
        _ = duration
        self.snackbar_calls.append((message, bgcolor))

    def show_snackbar(self, message: str, bgcolor) -> None:
        self._show_snackbar(message, bgcolor)

    def add_history_entry(
        self,
        source: str,
        text: str,
        *,
        translated: bool = False,
        language_code: str | None = None,
    ) -> None:
        self.history.append((source, text, translated, language_code))

    def on_overlay_state_changed(
        self,
        *,
        state: str,
        failure_reason: str | None = None,
    ) -> None:
        self.overlay_state = state
        self.overlay_failure_reason = failure_reason


class RuntimeLoggingCapture:
    def __init__(
        self,
        *,
        detailed_enabled: bool = True,
        detailed_error: Exception | None = None,
    ) -> None:
        self.detailed_enabled = detailed_enabled
        self.detailed_error = detailed_error
        self.basic_messages: list[tuple[int, str]] = []
        self.detailed_calls: list[tuple[int, str]] = []
        self.detailed_messages: list[tuple[int, str]] = []

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self.basic_messages.append((level, message))

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool:
        self.detailed_calls.append((level, message))
        if self.detailed_error is not None:
            raise self.detailed_error
        if not self.detailed_enabled:
            return False
        self.detailed_messages.append((level, message))
        return True


def make_bridge(app: object, **kwargs: object) -> UIEventBridge:
    event_queue = kwargs.pop("event_queue", asyncio.Queue())
    return UIEventBridge(
        event_queue=event_queue,
        dashboard_destination=kwargs.pop(
            "dashboard_destination",
            AppDashboardEventDestination(getattr(app, "view_dashboard", None)),
        ),
        history_destination=kwargs.pop(
            "history_destination",
            AppHistoryEventDestination(getattr(app, "add_history_entry", None)),
        ),
        conversation_destination=kwargs.pop(
            "conversation_destination",
            AppConversationEventDestination(
                getattr(getattr(app, "view_logs", None), "append_conversation_record", None)
            ),
        ),
        error_destination=kwargs.pop("error_destination", None),
        runtime_logging=kwargs.pop("runtime_logging", None),
        get_language_codes=kwargs.pop(
            "get_language_codes", getattr(app, "get_event_language_codes", None)
        ),
        is_translation_enabled=kwargs.pop(
            "is_translation_enabled", getattr(app, "is_event_translation_enabled", None)
        ),
        get_stt_state=kwargs.pop("get_stt_state", getattr(app, "get_event_stt_state", None)),
        clear_managed_auth_pending=kwargs.pop(
            "clear_managed_auth_pending", getattr(app, "clear_managed_auth_pending_state", None)
        ),
        show_snackbar=kwargs.pop("show_snackbar", getattr(app, "show_snackbar", None)),
        on_github_star_translation_success=kwargs.pop(
            "on_github_star_translation_success",
            getattr(app, "on_github_star_translation_success", None),
        ),
        on_overlay_state_changed=kwargs.pop(
            "on_overlay_state_changed", getattr(app, "on_overlay_state_changed", None)
        ),
    )


def test_event_bridge_reads_language_codes_from_controller_contract_without_settings_shape() -> (
    None
):
    app = DummyApp()
    app.controller = SimpleNamespace(
        get_event_language_codes=lambda: ("ja", "de"),
        hub=SimpleNamespace(translation_enabled=False),
    )
    app._ui_application = compose_test_ui_application_boundary(app.controller)
    bridge = make_bridge(app, event_queue=asyncio.Queue())

    assert bridge._get_language_codes() == ("ja", "de")


def test_event_bridge_constructor_requires_explicit_dispatch_ports() -> None:
    signature = inspect.signature(UIEventBridge)

    assert "app" not in signature.parameters
    for name in (
        "dashboard_destination",
        "history_destination",
        "conversation_destination",
    ):
        assert signature.parameters[name].default is inspect.Parameter.empty


@pytest.mark.asyncio
async def test_event_bridge_reports_started_only_after_run_loop_entry() -> None:
    bridge = make_bridge(DummyApp(), event_queue=asyncio.Queue())
    wait_started = asyncio.create_task(bridge.wait_started())
    await asyncio.sleep(0)
    assert not wait_started.done()

    run_task = asyncio.create_task(bridge.run())
    await wait_started

    assert bridge._running is True
    bridge.close()
    run_task.cancel()
    await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_event_bridge_failure_log_identifies_event_without_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = DummyApp()
    queue: asyncio.Queue[UIEvent] = asyncio.Queue()
    bridge = make_bridge(app, event_queue=queue)

    def fail_history(*_args: object, **_kwargs: object) -> None:
        raise AttributeError("private transcript text")

    bridge.history_destination = AppHistoryEventDestination(fail_history)
    event = UIEvent(
        type=UIEventType.TRANSCRIPT_FINAL,
        payload=Transcript(utterance_id=uuid4(), text="secret", is_final=True),
        channel="self",
    )
    task = asyncio.create_task(bridge.run())
    await bridge.wait_started()
    await queue.put(event)
    await queue.join()
    bridge.close()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    message = next(
        record.getMessage() for record in caplog.records if record.levelno == logging.ERROR
    )
    assert message == (
        "Error handling UI event: event_type=TRANSCRIPT_FINAL "
        "channel=self exception_type=AttributeError"
    )
    assert "secret" not in message
    assert "private transcript text" not in message


def test_event_mapping_is_testable_without_view_mutation() -> None:
    transcript = Transcript(utterance_id=uuid4(), text="partial", is_final=False)
    mapped = map_ui_event(
        UIEvent(type=UIEventType.TRANSCRIPT_PARTIAL, payload=transcript, source="Mic")
    )

    assert mapped is not None
    assert mapped.kind == "transcript"
    assert mapped.payload is transcript
    assert mapped.source == "Mic"
    assert mapped.transcript_kind == "partial"
    assert map_ui_event(UIEvent(type=UIEventType.TRANSCRIPT_FINAL, payload="bad")) is None


def test_event_projection_builds_dtos_without_runtime_subscription() -> None:
    service = EventProjectionService(final_transcript_cache_limit=2)
    utterance_id = uuid4()
    context = EventProjectionContext(
        source_language="ko",
        target_language="en",
        translation_enabled=True,
        runtime_logging_mode="detailed",
    )
    mapped_transcript = map_ui_event(
        UIEvent(
            type=UIEventType.TRANSCRIPT_FINAL,
            payload=Transcript(
                utterance_id=utterance_id,
                text="source",
                is_final=True,
                channel="self",
            ),
            source="Mic",
        )
    )
    assert mapped_transcript is not None
    transcript_projection = service.project(mapped_transcript, context)

    mapped_translation = map_ui_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=utterance_id,
                text="translated",
                channel="self",
                target_language="en",
            ),
            source="Mic",
        )
    )
    assert mapped_translation is not None
    translation_projection = service.project(mapped_translation, context)

    assert transcript_projection.transcript is not None
    assert transcript_projection.transcript.channel == "self"
    assert transcript_projection.history[0].language_code == "ko"
    assert translation_projection.translation is not None
    assert translation_projection.translation.channel == "self"
    assert translation_projection.conversation is not None
    assert translation_projection.conversation.source_text == "source"
    assert translation_projection.translation_diagnostic is not None
    assert translation_projection.translation_diagnostic.text_len == len("translated")


class RecordingDashboardDestination:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.transcripts: list[dict[str, object]] = []
        self.translations: list[dict[str, object]] = []
        self.errors: list[str] = []

    def publish_status(self, status: str) -> None:
        self.statuses.append(status)

    def publish_transcript(self, text: str, **metadata: object) -> None:
        self.transcripts.append({"text": text, **metadata})

    def publish_translation(self, text: str, **metadata: object) -> None:
        self.translations.append({"text": text, **metadata})

    def publish_error(self, text: str) -> None:
        self.errors.append(text)


class RecordingHistoryDestination:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def append_entry(
        self,
        source: str,
        text: str,
        *,
        translated: bool = False,
        language_code: str | None = None,
    ) -> None:
        self.entries.append(
            {
                "source": source,
                "text": text,
                "translated": translated,
                "language_code": language_code,
            }
        )


class RecordingConversationDestination:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def append_record(
        self,
        *,
        source: str,
        channel: str,
        source_text: str,
        translated_text: str,
        origin_wall_clock_ms: int | None = None,
    ) -> None:
        self.records.append(
            {
                "source": source,
                "channel": channel,
                "source_text": source_text,
                "translated_text": translated_text,
                "origin_wall_clock_ms": origin_wall_clock_ms,
            }
        )


class RecordingErrorDestination:
    def __init__(self, *, show_dashboard_error: bool = True) -> None:
        self.show_dashboard_error = show_dashboard_error
        self.errors: list[dict[str, object]] = []
        self.event_payloads: list[object | None] = []
        self.events: list[UIEvent] = []

    def publish_error(
        self,
        text: str,
        *,
        payload: object | None,
        event: UIEvent,
    ) -> bool:
        self.event_payloads.append(event.payload)
        self.events.append(event)
        self.errors.append(
            {
                "text": text,
                "payload": payload,
                "runtime_log_handled": event.runtime_log_handled,
            }
        )
        return self.show_dashboard_error


def assert_dashboard_translation_applied_marker(
    message: str,
    *,
    utterance_id: str,
    channel: str,
    source_label: str,
    dashboard_target_language: str | None,
    translation_target_language: str | None,
    text_len: int,
) -> None:
    assert "dashboard_translation_applied" in message
    assert f"utterance_id={utterance_id}" in message
    assert f"channel={channel}" in message
    assert f"source_label={json.dumps(source_label, ensure_ascii=False)}" in message
    assert f"dashboard_target_language={dashboard_target_language}" in message
    assert f"translation_target_language={translation_target_language}" in message
    assert f"text_len={text_len}" in message


@pytest.mark.asyncio
async def test_event_bridge_maps_session_and_transcript_events() -> None:
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    utterance_id = uuid4()

    await bridge._handle_event(
        UIEvent(type=UIEventType.SESSION_STATE_CHANGED, payload=STTSessionState.CONNECTING)
    )
    await bridge._handle_event(
        UIEvent(type=UIEventType.SESSION_STATE_CHANGED, payload=STTSessionState.STREAMING)
    )
    await bridge._handle_event(
        UIEvent(type=UIEventType.SESSION_STATE_CHANGED, payload=STTSessionState.DRAINING)
    )
    await bridge._handle_event(
        UIEvent(type=UIEventType.SESSION_STATE_CHANGED, payload=STTSessionState.DISCONNECTED)
    )

    partial = Transcript(utterance_id=utterance_id, text="partial", is_final=False)
    final = Transcript(utterance_id=utterance_id, text="final", is_final=True)
    await bridge._handle_event(
        UIEvent(type=UIEventType.TRANSCRIPT_PARTIAL, payload=partial, source="Mic")
    )
    await bridge._handle_event(
        UIEvent(type=UIEventType.TRANSCRIPT_FINAL, payload=final, source="Mic")
    )
    await bridge._handle_event(
        UIEvent(type=UIEventType.TRANSCRIPT_PARTIAL, payload="not-transcript")
    )

    assert app.view_dashboard.statuses == ["connecting", "connected", "stopping", "disconnected"]
    assert app.view_dashboard.display_calls[:2] == [
        ("partial", "ko", False),
        ("final", "ko", False),
    ]
    assert app.view_dashboard.notice_calls == []
    assert app.history == [("Mic", "final", False, "ko")]


@pytest.mark.asyncio
async def test_event_bridge_routes_translation_and_osc_history_by_language_mode() -> None:
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    utterance_id = uuid4()

    translation = Translation(utterance_id=utterance_id, text="translated")
    await bridge._handle_event(
        UIEvent(type=UIEventType.TRANSLATION_DONE, payload=translation, source="Mic")
    )
    await bridge._handle_event(
        UIEvent(type=UIEventType.TRANSLATION_DONE, payload="not-translation")
    )

    app.controller.hub.translation_enabled = True
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.OSC_SENT,
            payload=OSCMessage(utterance_id=utterance_id, text="hello", created_at=0.0),
        )
    )

    app.controller.hub.translation_enabled = False
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.OSC_SENT,
            payload=OSCMessage(utterance_id=utterance_id, text="bye", created_at=0.0),
        )
    )

    assert app.view_dashboard.translation_calls == [("translated", "en")]
    assert ("Mic", "translated", True, "en") in app.history
    assert ("VRChat", "hello", False, "en") in app.history
    assert ("VRChat", "bye", False, "ko") in app.history


@pytest.mark.asyncio
async def test_event_bridge_appends_self_conversation_record_from_translation_source_text() -> None:
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    utterance_id = uuid4()

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=utterance_id,
                text="고마워",
                source_text="ありがとう",
                channel="self",
                origin_wall_clock_ms=1712345678901,
            ),
            source="Mic",
        )
    )

    assert app.view_logs.conversation_records == [
        {
            "source": "Mic",
            "channel": "self",
            "source_text": "ありがとう",
            "translated_text": "고마워",
            "origin_wall_clock_ms": 1712345678901,
        }
    ]


@pytest.mark.asyncio
async def test_event_bridge_uses_cached_final_self_transcript_as_source_fallback() -> None:
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    utterance_id = uuid4()

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSCRIPT_FINAL,
            payload=Transcript(utterance_id=utterance_id, text="あああ", is_final=True),
            source="Mic",
        )
    )
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(utterance_id=utterance_id, text="아아아", channel="self"),
            source="Mic",
        )
    )

    assert app.view_logs.conversation_records[-1]["source_text"] == "あああ"
    assert app.view_logs.conversation_records[-1]["translated_text"] == "아아아"


@pytest.mark.asyncio
async def test_event_bridge_translation_source_text_takes_precedence_over_cached_transcript() -> (
    None
):
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    utterance_id = uuid4()

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSCRIPT_FINAL,
            payload=Transcript(utterance_id=utterance_id, text="cached source", is_final=True),
            source="Mic",
        )
    )
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=utterance_id,
                text="translated",
                source_text="explicit source",
                channel="self",
            ),
            source="Mic",
        )
    )

    assert app.view_logs.conversation_records[-1]["source_text"] == "explicit source"


@pytest.mark.asyncio
async def test_event_bridge_conversation_append_failure_does_not_skip_translation_history() -> None:
    app = DummyApp()
    app.view_logs = FailingConversationLogs()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    utterance_id = uuid4()

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=utterance_id,
                text="translated",
                source_text="source",
                channel="self",
            ),
            source="Mic",
        )
    )

    assert app.view_dashboard.translation_calls == [("translated", "en")]
    assert app.history == [("Mic", "translated", True, "en")]


@pytest.mark.asyncio
async def test_event_bridge_missing_logs_sink_does_not_skip_translation_history() -> None:
    app_without_logs = DummyApp()
    delattr(app_without_logs, "view_logs")
    bridge_without_logs = make_bridge(app_without_logs, event_queue=asyncio.Queue())

    await bridge_without_logs._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=uuid4(),
                text="translated without logs view",
                source_text="source",
                channel="self",
            ),
            source="Mic",
        )
    )

    app_without_append = DummyApp()
    app_without_append.view_logs = SimpleNamespace(lines=[])
    bridge_without_append = make_bridge(app_without_append, event_queue=asyncio.Queue())

    await bridge_without_append._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=uuid4(),
                text="translated without append method",
                source_text="source",
                channel="self",
            ),
            source="Mic",
        )
    )

    assert app_without_logs.history == [("Mic", "translated without logs view", True, "en")]
    assert app_without_append.history == [("Mic", "translated without append method", True, "en")]


@pytest.mark.asyncio
async def test_event_bridge_skips_invalid_incomplete_peer_and_partial_conversation_records() -> (
    None
):
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    partial_id = uuid4()
    peer_id = uuid4()

    await bridge._handle_event(UIEvent(type=UIEventType.TRANSLATION_DONE, payload="bad"))
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSCRIPT_PARTIAL,
            payload=Transcript(utterance_id=partial_id, text="partial", is_final=False),
            source="Mic",
        )
    )
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=partial_id, text="partial translation", channel="self"
            ),
            source="Mic",
        )
    )
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=uuid4(),
                text="translated without source",
                channel="self",
            ),
            source="Mic",
        )
    )
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=uuid4(),
                text="   ",
                source_text="has source",
                channel="self",
            ),
            source="Mic",
        )
    )
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSCRIPT_FINAL,
            payload=Transcript(
                utterance_id=peer_id,
                text="peer final",
                is_final=True,
                channel="peer",
            ),
            source="Peer Mic",
        )
    )
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=peer_id,
                text="peer translation",
                channel="self",
            ),
            source="Mic",
        )
    )
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=uuid4(),
                text="peer translation",
                source_text="peer source",
                channel="peer",
            ),
            source="Peer Mic",
        )
    )

    assert app.view_logs.conversation_records == []


@pytest.mark.asyncio
async def test_event_bridge_final_self_transcript_cache_is_bounded(monkeypatch) -> None:
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    bridge.projection_service.final_transcript_cache_limit = 2
    first_id = uuid4()
    second_id = uuid4()
    third_id = uuid4()

    for utterance_id, text in ((first_id, "first"), (second_id, "second"), (third_id, "third")):
        await bridge._handle_event(
            UIEvent(
                type=UIEventType.TRANSCRIPT_FINAL,
                payload=Transcript(utterance_id=utterance_id, text=text, is_final=True),
                source="Mic",
            )
        )

    assert list(bridge._final_self_transcripts) == [str(second_id), str(third_id)]


@pytest.mark.asyncio
async def test_event_bridge_close_clears_conversation_cache_and_rejects_late_records() -> None:
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    utterance_id = uuid4()

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSCRIPT_FINAL,
            payload=Transcript(utterance_id=utterance_id, text="cached secret", is_final=True),
            source="Mic",
        )
    )
    assert list(bridge._final_self_transcripts) == [str(utterance_id)]

    bridge.close()
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=utterance_id,
                text="late translation",
                source_text="late secret source",
                channel="self",
            ),
            source="Mic",
        )
    )

    assert bridge._final_self_transcripts == {}
    assert app.view_logs.conversation_records == []


@pytest.mark.asyncio
async def test_event_bridge_appends_to_real_logs_view_conversation_text() -> None:
    app = DummyApp()
    app.view_logs = LogsView()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    utterance_id = uuid4()

    with (
        patch.object(type(app.view_logs), "page", new_callable=PropertyMock, return_value=None),
        patch.object(
            logs_view_module,
            "_format_conversation_timestamp",
            return_value="18:06:12",
        ),
    ):
        await bridge._handle_event(
            UIEvent(
                type=UIEventType.TRANSLATION_DONE,
                payload=Translation(
                    utterance_id=utterance_id,
                    text="고마워",
                    source_text="ありがとう",
                    channel="self",
                    origin_wall_clock_ms=1712345678901,
                ),
                source="Mic",
            )
        )
        app.view_logs._on_conversation_button_click(SimpleNamespace())

    assert app.view_logs._log_text.value == (
        "[18:06:12] " f"{logs_view_module.source_label('Mic')}\n" "ありがとう\n" "고마워"
    )


@pytest.mark.asyncio
async def test_event_bridge_logs_self_dashboard_translation_applied_detail_only() -> None:
    app = DummyApp()
    runtime_logging = RuntimeLoggingCapture()
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )
    utterance_id = uuid4()
    translation = Translation(
        utterance_id=utterance_id,
        text="translated self",
        channel="self",
        target_language="en",
    )

    await bridge._handle_event(
        UIEvent(type=UIEventType.TRANSLATION_DONE, payload=translation, source="Mic")
    )

    assert app.view_dashboard.translation_calls == [("translated self", "en")]
    assert app.history == [("Mic", "translated self", True, "en")]
    assert runtime_logging.basic_messages == []
    assert len(runtime_logging.detailed_messages) == 1
    level, message = runtime_logging.detailed_messages[0]
    assert level == logging.INFO
    assert_dashboard_translation_applied_marker(
        message,
        utterance_id=str(utterance_id),
        channel="self",
        source_label="Mic",
        dashboard_target_language="en",
        translation_target_language="en",
        text_len=len("translated self"),
    )


@pytest.mark.asyncio
async def test_event_bridge_accepts_separable_dashboard_history_conversation_destinations() -> None:
    app = DummyApp()
    dashboard = RecordingDashboardDestination()
    history = RecordingHistoryDestination()
    conversation = RecordingConversationDestination()
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        dashboard_destination=dashboard,
        history_destination=history,
        conversation_destination=conversation,
    )
    utterance_id = uuid4()

    await bridge._handle_event(
        UIEvent(type=UIEventType.SESSION_STATE_CHANGED, payload=STTSessionState.STREAMING)
    )
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSCRIPT_FINAL,
            payload=Transcript(utterance_id=utterance_id, text="source text", is_final=True),
            source="Mic",
        )
    )
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=utterance_id,
                text="translated text",
                source_text="source text",
                channel="self",
                origin_wall_clock_ms=1712345678901,
            ),
            source="Mic",
        )
    )

    assert dashboard.statuses == ["connected"]
    assert dashboard.transcripts[-1]["text"] == "source text"
    assert dashboard.translations[-1]["text"] == "translated text"
    assert history.entries == [
        {
            "source": "Mic",
            "text": "source text",
            "translated": False,
            "language_code": "ko",
        },
        {
            "source": "Mic",
            "text": "translated text",
            "translated": True,
            "language_code": "en",
        },
    ]
    assert conversation.records == [
        {
            "source": "Mic",
            "channel": "self",
            "source_text": "source text",
            "translated_text": "translated text",
            "origin_wall_clock_ms": 1712345678901,
        }
    ]
    assert app.view_dashboard.display_calls == []
    assert app.history == []
    assert app.view_logs.conversation_records == []


@pytest.mark.asyncio
async def test_event_bridge_error_destination_is_separable_from_dashboard_display() -> None:
    app = DummyApp()
    dashboard = RecordingDashboardDestination()
    error_destination = RecordingErrorDestination(show_dashboard_error=False)
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        dashboard_destination=dashboard,
        error_destination=error_destination,
    )

    await bridge._handle_event(
        UIEvent(type=UIEventType.ERROR, payload="separable failure", runtime_log_handled=True)
    )

    assert error_destination.errors == [
        {
            "text": "separable failure",
            "payload": "separable failure",
            "runtime_log_handled": True,
        }
    ]
    assert dashboard.errors == []
    assert app.view_dashboard.display_calls == []


@pytest.mark.asyncio
async def test_event_bridge_sanitizes_legacy_raw_string_payload_for_error_destination() -> None:
    app = DummyApp()
    dashboard = RecordingDashboardDestination()
    error_destination = RecordingErrorDestination(show_dashboard_error=False)
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        dashboard_destination=dashboard,
        error_destination=error_destination,
    )
    utterance_id = uuid4()
    raw = (
        "Provider failed provider_response_body="
        "{'error': {'message': 'bad'}, 'token': 'provider-secret-custom-destination'}"
        "\nTraceback (most recent call last):\n"
        '  File "provider.py", line 42, in translate\n'
        "RuntimeError: authorization=Bearer provider-token-custom-destination"
    )
    event = UIEvent(
        type=UIEventType.ERROR,
        utterance_id=utterance_id,
        payload=raw,
        source="Provider",
        channel="peer",
        runtime_log_handled=True,
    )

    await bridge._handle_event(event)

    assert len(error_destination.errors) == 1
    published = error_destination.errors[0]
    sanitized = published["text"]
    assert isinstance(sanitized, str)
    assert published["payload"] == sanitized
    assert error_destination.event_payloads == [sanitized]
    sanitized_event = error_destination.events[0]
    assert sanitized_event is not event
    assert sanitized_event.type is UIEventType.ERROR
    assert sanitized_event.utterance_id == utterance_id
    assert sanitized_event.source == "Provider"
    assert sanitized_event.channel == "peer"
    assert sanitized_event.runtime_log_handled is True
    assert sanitized_event.payload == sanitized
    combined = repr(error_destination.errors) + repr(error_destination.event_payloads)
    assert "provider-secret-custom-destination" not in combined
    assert "provider-token-custom-destination" not in combined
    assert "provider_response_body" not in combined
    assert "Traceback" not in combined
    assert dashboard.errors == []
    assert app.view_dashboard.display_calls == []


@pytest.mark.asyncio
async def test_event_bridge_preserves_typed_error_payload_identity_for_error_destination() -> None:
    app = DummyApp()
    dashboard = RecordingDashboardDestination()
    error_destination = RecordingErrorDestination(show_dashboard_error=False)
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        dashboard_destination=dashboard,
        error_destination=error_destination,
    )
    message = messages.UserMessageRef(
        key="stt.failure",
        params={
            "category": messages.DIAGNOSTIC_CATEGORY_TIMEOUT,
            "operation": "open_session",
            "provider": "soniox",
        },
        severity=messages.SEVERITY_ERROR,
    )
    report = messages.UserErrorReport(
        message=message,
        diagnostics=messages.ErrorDiagnostics(
            component="provider.stt",
            operation="open_session",
            code="stt.timeout",
            category=messages.DIAGNOSTIC_CATEGORY_TIMEOUT,
            visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields={"provider": "soniox"},
        ),
    )
    managed_error = ManagedOpenRouterUserFacingError(
        message_key="managed_release.openrouter_not_ready",
        message_kwargs={},
        diagnostics=ManagedOpenRouterReleaseDiagnostics(operation="issue"),
    )
    payloads = [message, report, managed_error]
    events = [
        UIEvent(type=UIEventType.ERROR, payload=payload, runtime_log_handled=True)
        for payload in payloads
    ]

    for event in events:
        await bridge._handle_event(event)

    assert [entry["payload"] for entry in error_destination.errors] == payloads
    for index, payload in enumerate(payloads):
        assert error_destination.errors[index]["payload"] is payload
        assert error_destination.event_payloads[index] is payload
        assert error_destination.events[index] is events[index]
    assert dashboard.errors == []
    assert app.view_dashboard.display_calls == []


@pytest.mark.asyncio
async def test_event_bridge_localizes_user_error_report_without_diagnostic_leak() -> None:
    previous_locale = get_locale()
    set_locale("en")
    try:
        raw_detail = "upstream response body token=provider-secret-123"
        app = DummyApp()
        runtime_logging = RuntimeLoggingCapture()
        bridge = make_bridge(
            app,
            event_queue=asyncio.Queue(),
            runtime_logging=runtime_logging,
        )
        report = messages.UserErrorReport(
            message=messages.UserMessageRef(
                key="provider.failure",
                params={
                    "category": messages.DIAGNOSTIC_CATEGORY_NETWORK,
                    "operation": "translate",
                    "provider": "openrouter",
                },
                severity=messages.SEVERITY_ERROR,
            ),
            diagnostics=messages.ErrorDiagnostics(
                component="provider.llm",
                operation="translate",
                code="provider.network",
                category=messages.DIAGNOSTIC_CATEGORY_NETWORK,
                visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
                content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
                status_code=None,
                retry_after_ms=None,
                fields={"raw_exception": raw_detail, "provider": "openrouter"},
            ),
        )

        await bridge._handle_event(UIEvent(type=UIEventType.ERROR, payload=report))

        expected = t(
            "provider.failure",
            category=messages.DIAGNOSTIC_CATEGORY_NETWORK,
            operation="translate",
            provider="openrouter",
        )
        assert app.view_dashboard.display_calls[-1] == (expected, None, True)
        assert runtime_logging.basic_messages == [(logging.ERROR, expected)]
        rendered = repr(app.view_dashboard.display_calls) + repr(runtime_logging.basic_messages)
        assert raw_detail not in rendered
        assert "provider-secret-123" not in rendered
        assert "provider.network" not in rendered
        assert "provider.failure (category=" not in rendered
    finally:
        set_locale(previous_locale)


@pytest.mark.asyncio
async def test_event_bridge_localizes_direct_message_ref_payload() -> None:
    previous_locale = get_locale()
    set_locale("en")
    try:
        app = DummyApp()
        runtime_logging = RuntimeLoggingCapture()
        bridge = make_bridge(
            app,
            event_queue=asyncio.Queue(),
            runtime_logging=runtime_logging,
        )
        message = messages.UserMessageRef(
            key="stt.failure",
            params={
                "category": messages.DIAGNOSTIC_CATEGORY_TIMEOUT,
                "operation": "open_session",
                "provider": "soniox",
            },
            severity=messages.SEVERITY_ERROR,
        )

        await bridge._handle_event(UIEvent(type=UIEventType.ERROR, payload=message))

        expected = t(
            "stt.failure",
            category=messages.DIAGNOSTIC_CATEGORY_TIMEOUT,
            operation="open_session",
            provider="soniox",
        )
        assert app.view_dashboard.display_calls[-1] == (expected, None, True)
        assert runtime_logging.basic_messages == [(logging.ERROR, expected)]
        rendered = repr(app.view_dashboard.display_calls) + repr(runtime_logging.basic_messages)
        assert "UserMessageRef(" not in rendered
        assert "stt.failure" not in rendered
    finally:
        set_locale(previous_locale)


@pytest.mark.asyncio
async def test_event_bridge_redacts_unsafe_message_ref_params_before_user_visible_sinks() -> None:
    previous_locale = get_locale()
    set_locale("en")
    try:
        app = DummyApp()
        runtime_logging = RuntimeLoggingCapture()
        bridge = make_bridge(
            app,
            event_queue=asyncio.Queue(),
            runtime_logging=runtime_logging,
        )
        message = messages.UserMessageRef(
            key="provider.failure",
            params={
                "category": "network token=provider-secret-param",
                "operation": "translate",
                "provider": "openrouter",
            },
            severity=messages.SEVERITY_ERROR,
        )

        await bridge._handle_event(UIEvent(type=UIEventType.ERROR, payload=message))

        combined = repr(app.view_dashboard.display_calls) + repr(runtime_logging.basic_messages)
        assert "provider-secret-param" not in combined
        assert "network token=" not in combined
        assert "[redacted]" in combined
        assert app.view_dashboard.display_calls[-1][0] == runtime_logging.basic_messages[-1][1]
    finally:
        set_locale(previous_locale)


@pytest.mark.asyncio
async def test_event_bridge_sanitizes_legacy_raw_string_error_before_user_visible_sinks() -> None:
    app = DummyApp()
    runtime_logging = RuntimeLoggingCapture()
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )
    raw = (
        "Provider failure provider_response_body="
        "{'error':'bad','token':'provider-secret-123'}\n"
        "Traceback (most recent call last):\n"
        '  File "provider.py", line 42, in translate\n'
        "RuntimeError: authorization=Bearer provider-token-456"
    )

    await bridge._handle_event(UIEvent(type=UIEventType.ERROR, payload=raw))

    assert app.view_dashboard.display_calls
    displayed = app.view_dashboard.display_calls[-1][0]
    assert displayed == runtime_logging.basic_messages[-1][1]
    combined_user_visible = repr(app.view_dashboard.display_calls) + repr(
        runtime_logging.basic_messages
    )
    assert "Provider failure" in displayed
    assert "[redacted]" in displayed
    assert "provider-secret-123" not in combined_user_visible
    assert "provider-token-456" not in combined_user_visible
    assert "provider_response_body" not in combined_user_visible
    assert "Traceback" not in combined_user_visible
    assert 'File "provider.py"' not in combined_user_visible
    assert runtime_logging.detailed_messages == [
        (
            logging.WARNING,
            "[UIEventBridge] Deprecated raw string error payload sanitized for user-visible sinks",
        )
    ]


@pytest.mark.asyncio
async def test_event_bridge_routes_legacy_raw_fallback_through_central_redactor() -> None:
    app = DummyApp()
    runtime_logging = RuntimeLoggingCapture()
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )
    raw = (
        "raw_exception=ValueError('raw provider exception text')\n"
        "file_contents=user document text must not display"
    )

    await bridge._handle_event(UIEvent(type=UIEventType.ERROR, payload=raw))

    combined = repr(app.view_dashboard.display_calls) + repr(runtime_logging.basic_messages)
    assert "raw_exception" not in combined
    assert "raw provider exception text" not in combined
    assert "file_contents" not in combined
    assert "user document text" not in combined
    assert "[redacted]" in combined


@pytest.mark.asyncio
async def test_event_bridge_passes_dashboard_translation_visual_commit_metadata_to_dashboard() -> (
    None
):
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    utterance_id = uuid4()
    translation = Translation(
        utterance_id=utterance_id,
        text="translated peer",
        channel="peer",
        target_language="ja",
        update_id="upd-dashboard-1",
        origin_wall_clock_ms=1712345678901,
        session_scope="session-42",
        source_text_hash="src-hash-42",
        source_text_len=17,
        logical_turn_key="peer:turn-42",
    )

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=translation,
            source="Peer Mic",
        )
    )

    assert app.view_dashboard.translation_calls == [("translated peer", "en")]
    assert app.view_dashboard.translation_metadata_calls == [
        {
            "update_id": "upd-dashboard-1",
            "origin_wall_clock_ms": 1712345678901,
            "utterance_id": utterance_id,
            "channel": "peer",
            "session_scope": "session-42",
            "source_text_hash": "src-hash-42",
            "source_text_len": 17,
            "logical_turn_key": "peer:turn-42",
            "debug_prefix": None,
        }
    ]


@pytest.mark.asyncio
async def test_event_bridge_passes_peer_debug_prefix_when_runtime_logging_is_detailed() -> None:
    app = DummyApp()
    runtime_logging = RuntimeLoggingCapture()
    runtime_logging.mode = SessionLoggingMode.DETAILED
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )
    utterance_id = uuid4()

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSCRIPT_FINAL,
            payload=Transcript(
                utterance_id=utterance_id,
                text="peer source",
                is_final=True,
                channel="peer",
            ),
            source="Peer Mic",
        )
    )

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(
                utterance_id=utterance_id,
                text="peer translation",
                channel="peer",
                target_language="en",
                update_id="3bd7ffff-1111-2222-3333-444455556666",
            ),
            source="Peer Mic",
        )
    )

    turn_tail = str(utterance_id).replace("-", "")[:4]
    assert app.view_dashboard.display_debug_prefixes[-1] == f"[P {turn_tail}/src]"
    assert app.view_dashboard.translation_metadata_calls[-1]["debug_prefix"] == (
        f"[P {turn_tail}/3bd7]"
    )


@pytest.mark.asyncio
async def test_event_bridge_logs_peer_dashboard_translation_applied_detail_only() -> None:
    app = DummyApp()
    runtime_logging = RuntimeLoggingCapture()
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )
    utterance_id = uuid4()
    translation = Translation(
        utterance_id=utterance_id,
        text="translated peer",
        channel="peer",
        target_language="ja",
    )

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=translation,
            source="Peer Mic",
        )
    )

    assert app.view_dashboard.translation_calls == [("translated peer", "en")]
    assert app.history == [("Peer Mic", "translated peer", True, "en")]
    assert runtime_logging.basic_messages == []
    assert len(runtime_logging.detailed_messages) == 1
    level, message = runtime_logging.detailed_messages[0]
    assert level == logging.INFO
    assert_dashboard_translation_applied_marker(
        message,
        utterance_id=str(utterance_id),
        channel="peer",
        source_label="Peer Mic",
        dashboard_target_language="en",
        translation_target_language="ja",
        text_len=len("translated peer"),
    )


@pytest.mark.asyncio
async def test_event_bridge_does_not_log_dashboard_translation_applied_for_invalid_payload() -> (
    None
):
    app = DummyApp()
    runtime_logging = RuntimeLoggingCapture()
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )

    await bridge._handle_event(
        UIEvent(type=UIEventType.TRANSLATION_DONE, payload="not-translation")
    )

    assert app.view_dashboard.translation_calls == []
    assert app.history == []
    assert runtime_logging.detailed_calls == []
    assert runtime_logging.basic_messages == []


@pytest.mark.asyncio
async def test_event_bridge_does_not_log_dashboard_translation_applied_without_dashboard() -> None:
    app = DummyApp()
    app.view_dashboard = None
    runtime_logging = RuntimeLoggingCapture()
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )
    translation = Translation(utterance_id=uuid4(), text="translated", channel="self")

    await bridge._handle_event(
        UIEvent(type=UIEventType.TRANSLATION_DONE, payload=translation, source="Mic")
    )

    assert app.history == [("Mic", "translated", True, "en")]
    assert runtime_logging.detailed_calls == []
    assert runtime_logging.basic_messages == []


@pytest.mark.asyncio
async def test_event_bridge_best_effort_translation_apply_logging_does_not_block_history() -> None:
    app = DummyApp()
    runtime_logging = RuntimeLoggingCapture(detailed_error=RuntimeError("detail emit failed"))
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )
    translation = Translation(utterance_id=uuid4(), text="translated", channel="self")

    await bridge._handle_event(
        UIEvent(type=UIEventType.TRANSLATION_DONE, payload=translation, source="Mic")
    )

    assert app.view_dashboard.translation_calls == [("translated", "en")]
    assert app.history == [("Mic", "translated", True, "en")]
    assert len(runtime_logging.detailed_calls) == 1
    assert runtime_logging.detailed_messages == []
    assert runtime_logging.basic_messages == []


@pytest.mark.asyncio
async def test_event_bridge_dashboard_translation_applied_detail_disabled_keeps_dashboard_and_history() -> (
    None
):
    app = DummyApp()
    runtime_logging = RuntimeLoggingCapture(detailed_enabled=False)
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )
    translation = Translation(utterance_id=uuid4(), text="translated", channel="peer")

    await bridge._handle_event(
        UIEvent(type=UIEventType.TRANSLATION_DONE, payload=translation, source="Peer Mic")
    )

    assert app.view_dashboard.translation_calls == [("translated", "en")]
    assert app.history == [("Peer Mic", "translated", True, "en")]
    assert len(runtime_logging.detailed_calls) == 1
    assert runtime_logging.detailed_messages == []
    assert runtime_logging.basic_messages == []


@pytest.mark.asyncio
async def test_event_bridge_does_not_log_dashboard_translation_applied_when_setter_fails() -> None:
    app = DummyApp()
    app.view_dashboard = FailingTranslationDashboard()
    runtime_logging = RuntimeLoggingCapture()
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )
    translation = Translation(utterance_id=uuid4(), text="translated", channel="self")

    with pytest.raises(RuntimeError, match="dashboard setter failed"):
        await bridge._handle_event(
            UIEvent(type=UIEventType.TRANSLATION_DONE, payload=translation, source="Mic")
        )

    assert app.history == []
    assert runtime_logging.detailed_calls == []
    assert runtime_logging.basic_messages == []


@pytest.mark.asyncio
async def test_event_bridge_handles_error_and_soniox_shutdown_suppression(tmp_path) -> None:
    app = DummyApp()
    root_logger = logging.getLogger(f"test.event_bridge.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    session_logger = logging.getLogger(f"test.event_bridge.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False
    log_file = tmp_path / "event-bridge.log"
    stream_handler = logging.StreamHandler(io.StringIO())
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=SimpleNamespace(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
        ui_handler_factory=FletLogHandler,
    )
    runtime_logging.attach_realtime_sink(app.view_logs)
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )

    try:
        app.controller.hub.stt.state = STTSessionState.DRAINING
        await bridge._handle_event(
            UIEvent(type=UIEventType.ERROR, payload="Soniox 400 bad request")
        )

        app.controller.hub.stt.state = STTSessionState.STREAMING
        await bridge._handle_event(UIEvent(type=UIEventType.ERROR, payload="General failure"))
        await bridge._handle_event(UIEvent(type=UIEventType.ERROR, payload=None))

        assert len(app.view_logs.lines) == 3
        assert all("[ERROR]" in line for line in app.view_logs.lines)
        assert app.view_dashboard.display_calls[-2:] == [
            ("General failure", None, True),
            (t("error.unknown"), None, True),
        ]
    finally:
        runtime_logging.close()
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()


@pytest.mark.asyncio
async def test_event_bridge_skips_duplicate_runtime_log_for_already_logged_errors(tmp_path) -> None:
    app = DummyApp()
    root_logger = logging.getLogger(f"test.event_bridge.runtime.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    session_logger = logging.getLogger(f"test.event_bridge.runtime.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False
    log_file = tmp_path / "event-bridge-duplicate.log"
    stream_handler = logging.StreamHandler(io.StringIO())
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=SimpleNamespace(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
        ui_handler_factory=FletLogHandler,
    )
    runtime_logging.attach_realtime_sink(app.view_logs)
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=runtime_logging,
    )

    try:
        runtime_logging.emit_basic("already logged failure", level=logging.ERROR)

        await bridge._handle_event(
            UIEvent(
                type=UIEventType.ERROR,
                payload="already logged failure",
                runtime_log_handled=True,
            )
        )

        assert len(app.view_logs.lines) == 1
        assert "already logged failure" in app.view_logs.lines[0]
        assert app.view_dashboard.display_calls[-1] == ("already logged failure", None, True)
    finally:
        runtime_logging.close()
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()


@pytest.mark.asyncio
async def test_event_bridge_ignores_unknown_event_and_keeps_queue_alive() -> None:
    app = DummyApp()
    queue: asyncio.Queue = asyncio.Queue()
    bridge = make_bridge(app, event_queue=queue)

    task = asyncio.create_task(bridge.run())
    await queue.put(SimpleNamespace(type="UNKNOWN", payload="x", source=None))
    await queue.put(UIEvent(type=UIEventType.ERROR, payload="after unknown"))
    await queue.join()

    assert app.view_dashboard.display_calls[-1] == ("after unknown", None, True)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_event_bridge_error_without_runtime_logging_uses_standard_logger_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())
    seen: list[str] = []
    monkeypatch.setattr(event_dispatch_module.logger, "error", lambda message: seen.append(message))

    await bridge._handle_event(UIEvent(type=UIEventType.ERROR, payload="plain failure"))

    assert seen == ["plain failure"]
    assert app.view_logs.lines == []
    assert app.view_dashboard.display_calls[-1] == ("plain failure", None, True)


@pytest.mark.asyncio
async def test_event_bridge_error_with_broken_runtime_logging_uses_standard_logger_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DummyApp()
    seen: list[str] = []

    class BrokenRuntimeLogging:
        def emit_basic(self, _message: str, *, level: int = logging.INFO) -> None:
            _ = level
            raise RuntimeError("emit failed")

    monkeypatch.setattr(event_dispatch_module.logger, "error", lambda message: seen.append(message))
    bridge = make_bridge(
        app,
        event_queue=asyncio.Queue(),
        runtime_logging=BrokenRuntimeLogging(),
    )

    await bridge._handle_event(UIEvent(type=UIEventType.ERROR, payload="broken runtime"))

    assert seen == ["broken runtime"]
    assert app.view_logs.lines == []
    assert app.view_dashboard.display_calls[-1] == ("broken runtime", None, True)


@pytest.mark.asyncio
async def test_event_bridge_routes_managed_message_report_to_snackbar_without_dashboard_clobber() -> (
    None
):
    previous_locale = get_locale()
    set_locale("en")
    try:
        app = DummyApp()
        app.controller.managed_auth_pending = True
        bridge = make_bridge(app, event_queue=asyncio.Queue())
        payload = messages.UserErrorReport(
            message=messages.UserMessageRef(
                key="managed_release.retry_after_ms",
                params={"retry_after_ms": 9000},
                severity=messages.SEVERITY_ERROR,
            ),
            diagnostics=messages.ErrorDiagnostics(
                component="provider.llm",
                operation="translate",
                code="provider.unknown",
                category=messages.DIAGNOSTIC_CATEGORY_UNKNOWN,
                visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
                content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
                status_code=None,
                retry_after_ms=9000,
                fields={
                    "exception_type": "ManagedOpenRouterUserFacingError",
                    "managed_operation": "issue",
                    "managed_code": "trial_unavailable",
                    "managed_error_class": "retryable",
                    "managed_subcode": "broker_backoff",
                },
            ),
        )

        await bridge._handle_event(
            UIEvent(type=UIEventType.ERROR, payload=payload, runtime_log_handled=True)
        )

        expected = t("managed_release.retry_after_ms", retry_after_ms=9000)
        assert app.snackbar_calls == [(expected, ft.Colors.ORANGE_700)]
        assert app.clear_managed_auth_pending_calls == 1
        assert app.view_dashboard.display_calls == []
    finally:
        set_locale(previous_locale)


@pytest.mark.asyncio
async def test_event_bridge_routes_managed_auth_error_to_snackbar_without_dashboard_clobber() -> (
    None
):
    previous_locale = get_locale()
    set_locale("en")
    try:
        app = DummyApp()
        app.controller.managed_auth_pending = True
        bridge = make_bridge(app, event_queue=asyncio.Queue())
        payload = ManagedOpenRouterUserFacingError(
            message_key="managed_release.retry_after_ms",
            message_kwargs={"retry_after_ms": 9000},
            diagnostics=ManagedOpenRouterReleaseDiagnostics(
                operation="issue",
                code="trial_unavailable",
                error_class="retryable",
                subcode="broker_backoff",
                retry_after_ms=9000,
                message="broker is temporarily unavailable",
            ),
        )

        await bridge._handle_event(
            UIEvent(type=UIEventType.ERROR, payload=payload, runtime_log_handled=True)
        )

        expected = t("managed_release.retry_after_ms", retry_after_ms=9000)
        assert app.snackbar_calls == [(expected, ft.Colors.ORANGE_700)]
        assert app.clear_managed_auth_pending_calls == 1
        assert app.view_dashboard.display_calls == []
    finally:
        set_locale(previous_locale)


@pytest.mark.asyncio
async def test_event_bridge_keeps_general_error_display_when_managed_auth_is_pending() -> None:
    app = DummyApp()
    app.controller.managed_auth_pending = True
    bridge = make_bridge(app, event_queue=asyncio.Queue())

    await bridge._handle_event(
        UIEvent(type=UIEventType.ERROR, payload="managed auth boom", runtime_log_handled=True)
    )

    assert app.snackbar_calls == []
    assert app.clear_managed_auth_pending_calls == 0
    assert app.view_dashboard.display_calls == [("managed auth boom", None, True)]


def test_event_bridge_reports_overlay_state_to_app() -> None:
    app = DummyApp()
    bridge = make_bridge(app, event_queue=asyncio.Queue())

    bridge.report_overlay_state("starting")
    bridge.report_overlay_state("failed", failure_reason="runtime_crashed")

    assert app.overlay_state == "failed"
    assert app.overlay_failure_reason == "runtime_crashed"
