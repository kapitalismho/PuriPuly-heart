from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import pytest

from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
    SEVERITY_ERROR,
)
from puripuly_heart.core.observability import DiagnosticEvent


class RecordingDiagnosticsSink:
    def __init__(self, *, on_emit: Callable[[DiagnosticEvent], None] | None = None) -> None:
        self.events: list[DiagnosticEvent] = []
        self._on_emit = on_emit

    async def emit_diagnostic(self, event: DiagnosticEvent) -> None:
        self.events.append(event)
        if self._on_emit is not None:
            self._on_emit(event)


def _lifecycle_module() -> Any:
    try:
        return importlib.import_module("puripuly_heart.core.lifecycle")
    except ModuleNotFoundError as exc:
        pytest.fail(f"lifecycle module missing: {exc.name}")


def _required_attr(module: Any, name: str) -> Any:
    if not hasattr(module, name):
        pytest.fail(f"puripuly_heart.core.lifecycle must export {name}")
    return getattr(module, name)


def _all_diagnostic_field_values(event: DiagnosticEvent) -> list[str]:
    values: list[str] = [str(value) for value in event.fields.values()]
    if event.diagnostics is not None:
        values.extend(str(value) for value in event.diagnostics.fields.values())
    return values


def test_shutdown_phase_order_exports_spec_dag_and_keeps_logging_last() -> None:
    lifecycle = _lifecycle_module()

    phase_order = _required_attr(lifecycle, "LIFECYCLE_SHUTDOWN_PHASE_ORDER")
    freeze_ingress = _required_attr(lifecycle, "SHUTDOWN_PHASE_FREEZE_INGRESS")
    stop_external_producers = _required_attr(
        lifecycle,
        "SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS",
    )
    owner_drain_cancel = _required_attr(
        lifecycle,
        "SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL",
    )
    close_providers_outputs = _required_attr(
        lifecycle,
        "SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS",
    )
    final_diagnostics = _required_attr(
        lifecycle,
        "SHUTDOWN_PHASE_FINAL_DIAGNOSTICS",
    )
    close_logging_diagnostics = _required_attr(
        lifecycle,
        "SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS",
    )

    assert phase_order == (
        freeze_ingress,
        stop_external_producers,
        owner_drain_cancel,
        close_providers_outputs,
        final_diagnostics,
        close_logging_diagnostics,
    )
    assert phase_order == (
        "freeze_ingress",
        "stop_external_producers",
        "run_owner_specific_drain_cancel_policies",
        "close_providers_and_output_adapters",
        "emit_final_shutdown_diagnostics",
        "flush_and_close_logging_diagnostics",
    )
    assert phase_order[-1] == close_logging_diagnostics


@pytest.mark.asyncio
async def test_shutdown_runner_executes_owner_policy_before_provider_output_close() -> None:
    lifecycle = _lifecycle_module()
    LifecycleShutdownCoordinator = _required_attr(lifecycle, "LifecycleShutdownCoordinator")
    phase_order = _required_attr(lifecycle, "LIFECYCLE_SHUTDOWN_PHASE_ORDER")
    calls: list[str] = []
    sink = RecordingDiagnosticsSink()
    coordinator = LifecycleShutdownCoordinator(diagnostics_sink=sink)

    def callback(label: str) -> Callable[[], None]:
        def record() -> None:
            calls.append(label)

        return record

    registrations = [
        (phase_order[0], "app", "freeze", "freeze"),
        (phase_order[1], "self_audio", "stop_source", "stop-source"),
        (phase_order[2], "self_audio", "drain_cancel", "owner-drain-cancel"),
        (phase_order[3], "provider_runtime", "close_provider", "provider-close"),
        (phase_order[3], "output_runtime", "close_output", "output-close"),
        (phase_order[4], "runtime_logging", "final_summary", "final-diagnostics"),
        (phase_order[5], "runtime_logging", "close", "logging-close"),
    ]
    for phase, owner_name, callback_name, label in registrations:
        coordinator.register_callback(
            phase=phase,
            owner_name=owner_name,
            callback_name=callback_name,
            callback=callback(label),
        )

    await coordinator.run()

    assert calls == [
        "freeze",
        "stop-source",
        "owner-drain-cancel",
        "provider-close",
        "output-close",
        "final-diagnostics",
        "logging-close",
    ]
    assert sink.events == []


@pytest.mark.asyncio
async def test_shutdown_callback_exceptions_emit_bounded_diagnostics_and_later_phases_continue() -> (
    None
):
    lifecycle = _lifecycle_module()
    LifecycleShutdownCoordinator = _required_attr(lifecycle, "LifecycleShutdownCoordinator")
    phase_order = _required_attr(lifecycle, "LIFECYCLE_SHUTDOWN_PHASE_ORDER")
    sink = RecordingDiagnosticsSink()
    calls: list[str] = []
    coordinator = LifecycleShutdownCoordinator(diagnostics_sink=sink)

    async def failing_owner_policy() -> None:
        calls.append("owner-policy")
        raise RuntimeError("secret-token raw provider payload")

    async def close_provider() -> None:
        calls.append("provider-close")

    async def final_diagnostics() -> None:
        calls.append("final-diagnostics")

    async def close_logging() -> None:
        calls.append("logging-close")

    coordinator.register_callback(
        phase=phase_order[2],
        owner_name="self_audio",
        callback_name="drain_cancel",
        callback=failing_owner_policy,
    )
    coordinator.register_callback(
        phase=phase_order[3],
        owner_name="provider_runtime",
        callback_name="close_provider",
        callback=close_provider,
    )
    coordinator.register_callback(
        phase=phase_order[4],
        owner_name="runtime_logging",
        callback_name="final_summary",
        callback=final_diagnostics,
    )
    coordinator.register_callback(
        phase=phase_order[5],
        owner_name="runtime_logging",
        callback_name="close",
        callback=close_logging,
    )

    await coordinator.run()

    assert calls == ["owner-policy", "provider-close", "final-diagnostics", "logging-close"]
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.category == DIAGNOSTIC_CATEGORY_LIFECYCLE
    assert event.severity == SEVERITY_ERROR
    assert event.visibility == DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY
    assert event.content_policy == CONTENT_POLICY_METADATA_ONLY
    assert event.fields == {
        "phase": phase_order[2],
        "owner_name": "self_audio",
        "callback_name": "drain_cancel",
        "exception_class": "RuntimeError",
    }
    assert event.diagnostics is not None
    assert event.diagnostics.component == "lifecycle.shutdown"
    assert event.diagnostics.operation == phase_order[2]
    assert event.diagnostics.code == "lifecycle_shutdown_exception"
    assert event.diagnostics.fields == event.fields


@pytest.mark.asyncio
async def test_shutdown_diagnostics_exclude_raw_exception_payload_before_logging_closes() -> None:
    lifecycle = _lifecycle_module()
    LifecycleShutdownCoordinator = _required_attr(lifecycle, "LifecycleShutdownCoordinator")
    phase_order = _required_attr(lifecycle, "LIFECYCLE_SHUTDOWN_PHASE_ORDER")
    calls: list[str] = []
    sink = RecordingDiagnosticsSink(
        on_emit=lambda event: calls.append(f"diagnostic:{event.fields['phase']}")
    )
    coordinator = LifecycleShutdownCoordinator(diagnostics_sink=sink)

    async def failing_final_diagnostic() -> None:
        calls.append("final-diagnostic-callback")
        raise ValueError("secret-token provider payload from close path")

    async def close_logging() -> None:
        calls.append("logging-close")

    coordinator.register_callback(
        phase=phase_order[4],
        owner_name="runtime_logging",
        callback_name="final_summary",
        callback=failing_final_diagnostic,
    )
    coordinator.register_callback(
        phase=phase_order[5],
        owner_name="runtime_logging",
        callback_name="close",
        callback=close_logging,
    )

    await coordinator.run()

    assert calls == [
        "final-diagnostic-callback",
        f"diagnostic:{phase_order[4]}",
        "logging-close",
    ]
    assert len(sink.events) == 1
    values = _all_diagnostic_field_values(sink.events[0])
    assert "ValueError" in values
    assert not any("secret-token" in value for value in values)
    assert not any("provider payload" in value for value in values)
    assert not any("close path" in value for value in values)
