from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from puripuly_heart.app.ports.application_startup import ApplicationStartupDiagnostic
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownContext,
    ApplicationShutdownDiagnostic,
    ApplicationShutdownStallDiagnostic,
)
from puripuly_heart.core.lifecycle import LifecycleScope
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.runtime.logging import emit_safe_fallback_log


@dataclass(slots=True)
class ApplicationRuntimeLoggingOwner:
    presentation: UiPresentationPort
    service_factory: Callable[[], Any]
    fallback_logger: logging.Logger
    _service: Any | None = field(init=False, default=None, repr=False)
    control_event_sink: Callable[[dict[str, object]], None] | None = None
    _instance_id: str | None = field(init=False, default=None, repr=False)
    _task_scope: LifecycleScope = field(
        init=False,
        default_factory=lambda: LifecycleScope("ApplicationRuntimeLoggingOwner"),
        repr=False,
    )
    _ingress_stopped: bool = field(init=False, default=False, repr=False)

    def _metadata_service(self) -> Any:
        if self._service is None:
            self._service = self.service_factory()
        self._service.configure_diagnostics(
            instance_id=self._instance_id, event_sink=self.control_event_sink
        )
        return self._service

    @property
    def service(self) -> Any:
        service = self._metadata_service()
        self.presentation.attach_runtime_log_sink(service)
        return service

    @property
    def installed_service(self) -> Any | None:
        return self._service

    def install_service(self, service: Any | None) -> None:
        self._service = service

    @property
    def active_task_names(self) -> tuple[str, ...]:
        return self._task_scope.active_task_names

    def stop_ingress(self) -> None:
        self._ingress_stopped = True

    async def close_background_tasks(self) -> None:
        self.stop_ingress()
        await self._task_scope.close()

    def bind_instance(self, instance_id: str) -> None:
        self._instance_id = instance_id
        self._metadata_service().configure_diagnostics(instance_id=instance_id)

    def diagnostic_history(self, arguments: dict) -> dict:
        return self._metadata_service().diagnostic_history(arguments)

    def diagnostic_health(self) -> dict:
        return self._metadata_service().diagnostic_health()

    def emit_failure(
        self,
        diagnostics: ErrorDiagnostics,
        *,
        severity: str = "error",
        disposition: str = "failed",
        stage: str | None = None,
        exception: BaseException | None = None,
    ) -> dict:
        return self._metadata_service().emit_failure(
            diagnostics,
            severity=severity,
            disposition=disposition,
            stage=stage,
            exception=exception,
        )

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        try:
            self.service.emit_basic(message, level=level)
        except Exception:
            emit_safe_fallback_log(
                self.fallback_logger,
                message,
                level=level,
                live=True,
            )

    def emit_translation_latency(self, message: str) -> bool:
        service = self._service
        if service is None:
            return False
        try:
            return bool(service.emit_translation_latency(message))
        except Exception:
            return False

    def _emit_exception(self, exception: BaseException, *, level: int) -> bool:
        diagnostics = ErrorDiagnostics(
            component="application_runtime_logging",
            operation="emit",
            code="unclassified_failure",
            category="unknown",
            visibility="diagnostic_only",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={},
        )
        try:
            self.emit_failure(
                diagnostics,
                severity=(
                    "error"
                    if level >= logging.ERROR
                    else "warning" if level >= logging.WARNING else "info"
                ),
                disposition="unknown",
                stage="diagnostic",
                exception=exception,
            )
            return True
        except Exception:
            return emit_safe_fallback_log(
                self.fallback_logger,
                "[Logging] diagnostic_unavailable",
                level=level,
                live=False,
            )

    def emit_diagnostic(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        if exception is not None:
            return self._emit_exception(exception, level=level)
        try:
            return bool(self.service.emit_diagnostic(message, level=level))
        except Exception:
            return emit_safe_fallback_log(
                self.fallback_logger,
                message,
                level=level,
                live=False,
            )

    def emit_diagnostic_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        if exception is not None:
            return self._emit_exception(exception, level=level)
        rendered_message: str | None = None

        def render_message() -> str:
            nonlocal rendered_message
            if rendered_message is None:
                rendered_message = build_message()
            return rendered_message

        try:
            return bool(self.service.emit_diagnostic_lazy(render_message, level=level))
        except Exception:
            return emit_safe_fallback_log(
                self.fallback_logger,
                render_message(),
                level=level,
                live=False,
            )

    def record_output_routing_decision(self, decision: object) -> None:
        service = self._service
        if service is None:
            return
        record = getattr(service, "record_output_routing_decision", None)
        if callable(record):
            record(decision)

    def record_conversation_observation(
        self,
        *,
        utterance_id: str,
        speaker_channel: str,
        transcript_text: str | None,
        translation_text: str | None,
        source_language: str | None,
        target_language: str | None,
        metadata: Mapping[str, str | int | float | bool | None] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        try:
            self.service.record_conversation_observation(
                utterance_id=utterance_id,
                speaker_channel=speaker_channel,
                transcript_text=transcript_text,
                translation_text=translation_text,
                source_language=source_language,
                target_language=target_language,
                metadata=metadata,
                correlation_id=correlation_id,
            )
        except Exception as exc:
            self.fallback_logger.error(
                "[Conversation] record_rejected turn=%s channel=%s "
                "reason=logging_failure exception_class=%s",
                utterance_id,
                speaker_channel,
                type(exc).__name__,
            )

    def record_request_context(
        self,
        *,
        utterance_id: str,
        context_texts: Sequence[str],
        segment_index: int | None = None,
    ) -> None:
        try:
            self.service.record_request_context(
                utterance_id=utterance_id,
                context_texts=context_texts,
                segment_index=segment_index,
            )
        except Exception as exc:
            self.fallback_logger.error(
                "[Context] record_rejected turn=%s reason=logging_failure exception_class=%s",
                utterance_id,
                type(exc).__name__,
            )

    def emit_startup_diagnostic(self, diagnostic: ApplicationStartupDiagnostic) -> None:
        if diagnostic.diagnostics is not None:
            owning = replace(
                diagnostic.diagnostics,
                fields={
                    **diagnostic.diagnostics.fields,
                    "attempt_id": diagnostic.attempt_id,
                    "process_id": diagnostic.process_id,
                    "monotonic_ns": diagnostic.monotonic_ns,
                },
            )
            self.emit_failure(
                owning,
                severity="info" if diagnostic.outcome == "cancelled" else "error",
                disposition="cancelled" if diagnostic.outcome == "cancelled" else "failed",
            )
            return
        service = self.service
        emit_boundary = getattr(service, "emit_startup_boundary", None)
        if callable(emit_boundary):
            try:
                emit_boundary(
                    outcome=diagnostic.outcome,
                    attempt_id=diagnostic.attempt_id,
                    process_id=diagnostic.process_id,
                    monotonic_ns=diagnostic.monotonic_ns,
                    exception_class=diagnostic.exception_class,
                )
            except Exception:
                emit_safe_fallback_log(
                    self.fallback_logger,
                    _format_startup_diagnostic(diagnostic),
                    level=_startup_diagnostic_level(diagnostic),
                    live=False,
                )
            return
        message = _format_startup_diagnostic(diagnostic)
        level = _startup_diagnostic_level(diagnostic)
        try:
            service.emit_persisted(message, level=level)
        except Exception:
            emit_safe_fallback_log(
                self.fallback_logger,
                message,
                level=level,
                live=False,
            )

    def emit_terminal_summary(self, context: ApplicationShutdownContext) -> None:
        service = self._service
        if service is None:
            return
        emit_persisted = getattr(service, "emit_persisted", None)
        if not callable(emit_persisted):
            return
        first = context.failures[0] if context.failures else None
        first_failure = (
            "none"
            if first is None
            else (
                f"{first.owner_name}/{first.callback_name}/"
                f"{first.exception_class}/timed_out={str(first.timed_out).lower()}"
            )
        )
        emit_persisted(
            "[Lifecycle][Shutdown] coordinator_terminal "
            "owner=ApplicationShutdownCoordinator "
            f"failure_count={len(context.failures)} "
            f"first_failure={first_failure} "
            f"additional_failure_count={max(0, len(context.failures) - 1)}",
            level=logging.INFO,
        )

    def close_after_producers_stop(self, context: ApplicationShutdownContext) -> None:
        service = self._service
        if service is None:
            return
        service.close_after_producers_stop(
            cleanup_failures=context.cleanup_exceptions,
        )

    def emit_shutdown_stall_diagnostic(
        self,
        diagnostic: ApplicationShutdownStallDiagnostic,
    ) -> None:
        service = self._service
        emit_persisted = getattr(service, "emit_persisted", None) if service is not None else None
        messages = [
            "[Lifecycle][Shutdown] stall "
            f"state={diagnostic.coordinator_state} "
            f"terminal={str(diagnostic.coordinator_terminal).lower()} "
            f"failure_count={diagnostic.coordinator_failure_count} "
            f"phase={diagnostic.phase or 'none'} "
            f"owner={diagnostic.active_owner_name or 'none'} "
            f"callback={diagnostic.active_callback_name or 'none'} "
            f"native_stack_available={str(diagnostic.native_stack_available).lower()}"
        ]
        messages.extend(
            "[Lifecycle][Shutdown] runtime_state "
            f"owner={_safe_diagnostic_token(state.owner_name)} "
            f"generation={state.generation if state.generation is not None else 'none'} "
            "native_operations="
            f"{','.join(_safe_diagnostic_token(item) for item in state.active_native_operations) or 'none'} "
            "children="
            f"{','.join(_safe_diagnostic_token(item) for item in state.child_states) or 'none'}"
            for state in diagnostic.runtime_states
        )
        messages.extend(
            "[Lifecycle][Shutdown] await_graph "
            f"task={_safe_diagnostic_token(task_name)} "
            f"graph={_safe_diagnostic_graph(graph)}"
            for task_name, graph in sorted(diagnostic.task_await_graphs.items())
        )
        for message in messages:
            if callable(emit_persisted):
                emit_persisted(message, level=logging.ERROR)
            else:
                self.fallback_logger.error(message)

    def emit_shutdown_diagnostic(
        self,
        diagnostic: ApplicationShutdownDiagnostic,
    ) -> None:
        if diagnostic.diagnostics is not None:
            self.emit_failure(diagnostic.diagnostics, disposition="failed", stage="cleanup")
            if diagnostic.stall_diagnostic is not None:
                self.emit_shutdown_stall_diagnostic(diagnostic.stall_diagnostic)
            return
        message = (
            "[Lifecycle][Shutdown] callback_failed "
            f"phase={diagnostic.phase} "
            f"owner={diagnostic.owner_name} "
            f"callback={diagnostic.callback_name} "
            f"exception_class={diagnostic.exception_class} "
            f"timed_out={str(diagnostic.timed_out).lower()}"
        )
        service = self._service
        emit_persisted = getattr(service, "emit_persisted", None) if service is not None else None
        if callable(emit_persisted):
            emit_persisted(message, level=logging.ERROR)
        else:
            self.fallback_logger.error(message)
        if diagnostic.stall_diagnostic is not None:
            self.emit_shutdown_stall_diagnostic(diagnostic.stall_diagnostic)


def _startup_diagnostic_level(diagnostic: ApplicationStartupDiagnostic) -> int:
    return logging.ERROR if diagnostic.outcome == "failed" else logging.INFO


def _format_startup_diagnostic(diagnostic: ApplicationStartupDiagnostic) -> str:
    return (
        "[Lifecycle][Startup] boundary "
        f"outcome={diagnostic.outcome} "
        f"attempt_id={diagnostic.attempt_id} "
        f"process_id={diagnostic.process_id} "
        f"monotonic_ns={diagnostic.monotonic_ns} "
        f"exception_class={diagnostic.exception_class or 'none'}"
    )


def _safe_diagnostic_token(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._:-=," else "_"
        for character in value[:256]
    )


def _safe_diagnostic_graph(value: str) -> str:
    return " ".join(value.split())[:4096]
