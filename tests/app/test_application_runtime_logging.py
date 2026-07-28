from __future__ import annotations

import logging
from types import SimpleNamespace

from puripuly_heart.app.services.application_runtime_logging import (
    ApplicationRuntimeLoggingOwner,
)
from puripuly_heart.app.services.application_shutdown import ApplicationShutdownDiagnostic
from puripuly_heart.core.lifecycle import SHUTDOWN_PHASE_FINAL_DIAGNOSTICS
from puripuly_heart.core.runtime_logging import SessionLoggingMode


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
        failures=(object(),),
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
    assert service.close_failures == (cleanup_error,)
