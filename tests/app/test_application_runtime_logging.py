from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.application_runtime_logging import (
    ApplicationRuntimeLoggingOwner,
)
from puripuly_heart.app.services.application_shutdown import ApplicationShutdownDiagnostic
from puripuly_heart.app.wiring.wiring_application_runtime_logging import (
    compose_application_runtime_logging,
)
from puripuly_heart.core.lifecycle import SHUTDOWN_PHASE_FINAL_DIAGNOSTICS
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks, SessionLoggingMode


class RecordingRuntimeLogging:
    def __init__(self) -> None:
        self.mode = SessionLoggingMode.BASIC
        self.basic: list[tuple[int, str]] = []
        self.detailed: list[tuple[int, str]] = []
        self.persisted: list[tuple[int, str]] = []
        self.close_failures: tuple[BaseException, ...] | None = None

    def set_mode(self, mode: SessionLoggingMode | str) -> None:
        self.mode = SessionLoggingMode(mode)

    def emit_basic(self, message: str, *, level: int) -> None:
        self.basic.append((level, message))

    def emit_detailed(self, message: str, *, level: int) -> bool:
        self.detailed.append((level, message))
        return True

    def emit_detailed_lazy(self, build_message, *, level: int) -> bool:
        self.detailed.append((level, build_message()))
        return True

    def emit_persisted(self, message: str, *, level: int) -> None:
        self.persisted.append((level, message))

    def close_after_producers_stop(self, *, cleanup_failures=()) -> None:
        self.close_failures = tuple(cleanup_failures)


def _owner() -> tuple[ApplicationRuntimeLoggingOwner, list[object]]:
    attached: list[object] = []
    owner = ApplicationRuntimeLoggingOwner(
        presentation=SimpleNamespace(
            attach_runtime_log_sink=lambda service: attached.append(service),
        ),
        service_factory=RecordingRuntimeLogging,
        fallback_logger=logging.getLogger("test.application-runtime-logging"),
    )
    return owner, attached


def test_owner_controls_mode_transition_and_presentation_attachment() -> None:
    owner, attached = _owner()
    service = RecordingRuntimeLogging()
    owner.install_service(service)
    detailed_enabled: list[str] = []
    modes: list[str] = []

    owner.set_mode(
        "detailed",
        detailed_enabled=lambda: detailed_enabled.append("enabled"),
        mode_changed=modes.append,
    )

    assert owner.mode == "detailed"
    assert detailed_enabled == ["enabled"]
    assert modes == ["detailed"]
    assert attached == [service, service]


def test_owner_initializes_mode_without_ui_transition_effects() -> None:
    owner, attached = _owner()
    service = RecordingRuntimeLogging()
    owner.install_service(service)

    owner.initialize_mode("detailed")

    assert service.mode is SessionLoggingMode.DETAILED
    assert attached == [service]


def test_owner_exposes_provider_observation_capability() -> None:
    owner, _ = _owner()
    service = RecordingRuntimeLogging()
    owner.install_service(service)

    observation: ProviderObservationPort = owner
    observation.emit_basic("provider observation", level=logging.WARNING)

    assert service.basic == [(logging.WARNING, "provider observation")]


def test_owner_formats_exception_detail_and_preserves_lazy_evaluation() -> None:
    owner, _ = _owner()
    service = RecordingRuntimeLogging()
    owner.install_service(service)

    try:
        raise RuntimeError("sensitive detail")
    except RuntimeError as error:
        assert owner.emit_detailed("failed", level=logging.WARNING, exception=error) is True
    assert owner.emit_detailed_lazy(lambda: "lazy", level=logging.INFO) is True

    assert service.detailed[0][0] == logging.WARNING
    assert service.detailed[0][1].startswith("failed\nTraceback")
    assert "RuntimeError: sensitive detail" in service.detailed[0][1]
    assert service.detailed[1] == (logging.INFO, "lazy")


def test_owner_keeps_shutdown_diagnostics_and_close_on_the_logging_boundary() -> None:
    owner, _ = _owner()
    service = RecordingRuntimeLogging()
    owner.install_service(service)
    cleanup_error = RuntimeError("cleanup")
    context = SimpleNamespace(
        failures=(
            SimpleNamespace(
                owner_name="CaptureOwner",
                callback_name="close",
                exception_class="TimeoutError",
                timed_out=True,
            ),
        ),
        cleanup_exceptions=(cleanup_error,),
    )
    diagnostic = ApplicationShutdownDiagnostic(
        phase=SHUTDOWN_PHASE_FINAL_DIAGNOSTICS,
        owner_name="Owner",
        callback_name="close",
        exception_class="RuntimeError",
        timed_out=False,
    )

    owner.emit_shutdown_diagnostic(diagnostic)
    owner.emit_terminal_summary(context)
    owner.close_after_producers_stop(context)

    assert service.persisted[0][0] == logging.ERROR
    assert "owner=Owner" in service.persisted[0][1]
    assert service.persisted[1][0] == logging.INFO
    assert "failure_count=1" in service.persisted[1][1]
    assert "first_failure=CaptureOwner/close/TimeoutError/timed_out=true" in (
        service.persisted[1][1]
    )
    assert "additional_failure_count=0" in service.persisted[1][1]
    assert service.close_failures == (cleanup_error,)


@pytest.mark.asyncio
async def test_owner_owns_and_cancels_fallback_overlay_update_task() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def update_overlay() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    owner = ApplicationRuntimeLoggingOwner(
        presentation=SimpleNamespace(
            attach_runtime_log_sink=lambda _service: None,
            schedule_task=lambda _callback: False,
        ),
        service_factory=RecordingRuntimeLogging,
        fallback_logger=logging.getLogger("test.application-runtime-logging.tasks"),
        overlay_logging_mode_update=update_overlay,
        overlay_logging_mode_update_available=lambda: True,
    )

    owner.schedule_overlay_logging_mode_update()
    await entered.wait()

    assert owner.active_task_names

    await owner.close_background_tasks()

    assert cancelled.is_set()
    assert owner.active_task_names == ()


def test_owner_stops_new_background_ingress() -> None:
    scheduled: list[object] = []
    owner = ApplicationRuntimeLoggingOwner(
        presentation=SimpleNamespace(
            attach_runtime_log_sink=lambda _service: None,
            schedule_task=lambda callback: scheduled.append(callback) or True,
        ),
        service_factory=RecordingRuntimeLogging,
        fallback_logger=logging.getLogger("test.application-runtime-logging.ingress"),
        overlay_logging_mode_update=lambda: asyncio.sleep(0),
        overlay_logging_mode_update_available=lambda: True,
    )

    owner.stop_ingress()
    owner.schedule_audio_environment_snapshot()
    owner.schedule_overlay_logging_mode_update()

    assert scheduled == []


@pytest.fixture
def composed_logging(tmp_path: Path, monkeypatch):
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(root, "level", root.level)
    log_file = tmp_path / "conversation.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    conversation = []
    realtime = SimpleNamespace(
        append_log=lambda line: None,
        append_conversation_record=lambda **record: conversation.append(record),
    )
    owner = compose_application_runtime_logging(
        presentation=SimpleNamespace(
            attach_runtime_log_sink=lambda service: service.attach_realtime_sink(realtime),
        ),
        sinks=RuntimeLoggingSinks(logging.NullHandler(), file_handler, log_file),
        overlay_logging_mode_update=lambda: asyncio.sleep(0),
        overlay_logging_mode_update_available=lambda: False,
    )
    try:
        yield owner, log_file, conversation
    finally:
        owner.service.close()
        file_handler.close()


def test_composed_logging_persists_source_and_translation(composed_logging, caplog) -> None:
    owner, log_file, conversation = composed_logging
    for source, translation in (("안녕하세요", None), (None, "Hello")):
        owner.record_conversation_observation(
            utterance_id="manual-turn",
            speaker_channel="self",
            transcript_text=source,
            translation_text=translation,
            source_language="ko",
            target_language="en" if translation else None,
            metadata=(
                {"turn_kind": "manual", "target_index": 0}
                if translation
                else {"turn_kind": "manual"}
            ),
        )

    persisted = log_file.read_text(encoding="utf-8")
    assert 'source="안녕하세요"' in persisted
    assert 'translation="Hello"' in persisted
    assert [(record["source_text"], record["translated_text"]) for record in conversation] == [
        ("안녕하세요", None),
        (None, "Hello"),
    ]
    assert "record_rejected" not in caplog.text


def test_composed_logging_ignores_conversation_after_close(composed_logging, caplog) -> None:
    owner, log_file, conversation = composed_logging
    owner.service.close()
    before = log_file.read_text(encoding="utf-8")

    owner.record_conversation_observation(
        utterance_id="late-turn",
        speaker_channel="peer",
        transcript_text="private late source",
        translation_text="private late translation",
        source_language="ko",
        target_language="en",
    )

    assert log_file.read_text(encoding="utf-8") == before
    assert conversation == []
    assert "record_rejected" not in caplog.text
    assert "private late" not in caplog.text


def test_conversation_failure_reports_type_without_private_payload(caplog) -> None:
    owner, _ = _owner()

    def reject(**kwargs) -> None:
        raise ValueError("private source and provider secret")

    owner.install_service(SimpleNamespace(record_conversation_observation=reject))
    owner.record_conversation_observation(
        utterance_id="failed-turn",
        speaker_channel="self",
        transcript_text="private source",
        translation_text="private translation",
        source_language="ko",
        target_language="en",
    )

    assert "exception_class=ValueError" in caplog.text
    assert "private source" not in caplog.text
    assert "private translation" not in caplog.text
    assert "provider secret" not in caplog.text
