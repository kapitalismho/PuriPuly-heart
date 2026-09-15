from __future__ import annotations

import asyncio
import importlib
from collections.abc import Coroutine
from typing import Any

import pytest

from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
    SEVERITY_ERROR,
)
from puripuly_heart.core.observability import DiagnosticEvent


def _lifecycle_types() -> tuple[type[Any], type[Exception]]:
    lifecycle = _lifecycle_module()
    return lifecycle.LifecycleScope, lifecycle.LifecycleScopeClosedError


def _lifecycle_module() -> Any:
    try:
        return importlib.import_module("puripuly_heart.core.lifecycle")
    except ModuleNotFoundError as exc:
        pytest.fail(f"lifecycle module missing: {exc.name}")


class RecordingDiagnosticsSink:
    def __init__(self) -> None:
        self.events: list[DiagnosticEvent] = []

    async def emit_diagnostic(self, event: DiagnosticEvent) -> None:
        self.events.append(event)


async def _wait_for_event_then_return(
    ready: asyncio.Event,
    value: str,
) -> str:
    await ready.wait()
    return value


def _all_diagnostic_field_values(event: DiagnosticEvent) -> list[str]:
    values: list[str] = [str(value) for value in event.fields.values()]
    if event.diagnostics is not None:
        values.extend(str(value) for value in event.diagnostics.fields.values())
    return values


@pytest.mark.asyncio
async def test_create_task_tracks_by_name_and_removes_when_done() -> None:
    LifecycleScope, _ = _lifecycle_types()
    scope = LifecycleScope("runtime")
    ready = asyncio.Event()

    task = scope.create_task(
        _wait_for_event_then_return(ready, "completed"),
        name="worker",
    )

    assert scope.active_task_names == ("worker",)

    ready.set()
    assert await task == "completed"
    await asyncio.sleep(0)

    assert scope.active_task_names == ()
    await scope.close()


@pytest.mark.asyncio
async def test_close_cancels_pending_tasks_invokes_callbacks_and_is_idempotent() -> None:
    LifecycleScope, _ = _lifecycle_types()
    scope = LifecycleScope("runtime")
    cancellation_seen = asyncio.Event()
    callback_calls: list[str] = []

    async def pending_forever() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancellation_seen.set()
            raise

    async def close_resource() -> None:
        callback_calls.append("resource")

    task = scope.create_task(pending_forever(), name="pending")
    scope.register_close_callback("resource", close_resource)
    await asyncio.sleep(0)

    await scope.close()
    await scope.close()

    assert task.cancelled()
    assert cancellation_seen.is_set()
    assert callback_calls == ["resource"]
    assert scope.active_task_names == ()


@pytest.mark.asyncio
async def test_create_task_rejects_duplicate_active_name_and_closes_supplied_coroutine() -> None:
    lifecycle = _lifecycle_module()
    LifecycleScope = lifecycle.LifecycleScope
    LifecycleTaskNameInUseError = lifecycle.LifecycleTaskNameInUseError
    scope = LifecycleScope("runtime")
    first_started = asyncio.Event()

    async def pending_worker() -> None:
        first_started.set()
        await asyncio.Future()

    async def duplicate_worker() -> None:
        await asyncio.sleep(0)

    scope.create_task(pending_worker(), name="worker")
    await first_started.wait()
    duplicate_coroutine: Coroutine[Any, Any, None] = duplicate_worker()

    with pytest.raises(LifecycleTaskNameInUseError):
        scope.create_task(duplicate_coroutine, name="worker")

    assert duplicate_coroutine.cr_frame is None
    assert scope.active_task_names == ("worker",)
    await scope.close()


@pytest.mark.asyncio
async def test_task_that_raises_during_close_cancellation_emits_safe_diagnostic() -> None:
    LifecycleScope, _ = _lifecycle_types()
    sink = RecordingDiagnosticsSink()
    scope = LifecycleScope("runtime", diagnostics_sink=sink)
    task_started = asyncio.Event()

    async def raises_when_cancelled() -> None:
        task_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            raise RuntimeError("secret-token cancellation payload") from None

    scope.create_task(raises_when_cancelled(), name="worker")
    await task_started.wait()

    await scope.close()

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.fields["scope_name"] == "runtime"
    assert event.fields["task_name"] == "worker"
    assert event.fields["exception_class"] == "RuntimeError"
    assert event.category == DIAGNOSTIC_CATEGORY_LIFECYCLE
    assert event.severity == SEVERITY_ERROR
    assert event.visibility == DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY
    assert event.content_policy == CONTENT_POLICY_METADATA_ONLY
    assert not any("secret-token" in value for value in _all_diagnostic_field_values(event))


@pytest.mark.asyncio
async def test_task_and_close_callback_failures_emit_metadata_only_diagnostics() -> None:
    LifecycleScope, _ = _lifecycle_types()
    sink = RecordingDiagnosticsSink()
    scope = LifecycleScope("runtime", diagnostics_sink=sink)

    async def failing_task() -> None:
        raise RuntimeError("secret-token raw provider payload")

    async def failing_callback() -> None:
        raise ValueError("secret-token file contents")

    scope.create_task(failing_task(), name="worker")
    scope.register_close_callback("resource", failing_callback)
    await asyncio.sleep(0)

    await scope.close()

    assert len(sink.events) == 2
    assert {event.fields["phase"] for event in sink.events} == {
        "task_done",
        "close_callback",
    }
    assert {event.fields["scope_name"] for event in sink.events} == {"runtime"}
    assert {event.category for event in sink.events} == {DIAGNOSTIC_CATEGORY_LIFECYCLE}
    assert {event.severity for event in sink.events} == {SEVERITY_ERROR}
    assert {event.visibility for event in sink.events} == {DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY}
    assert {event.content_policy for event in sink.events} == {CONTENT_POLICY_METADATA_ONLY}

    diagnostic_values = [
        value for event in sink.events for value in _all_diagnostic_field_values(event)
    ]
    assert "RuntimeError" in diagnostic_values
    assert "ValueError" in diagnostic_values
    assert not any("secret-token" in value for value in diagnostic_values)
    assert not any("raw provider payload" in value for value in diagnostic_values)
    assert not any("file contents" in value for value in diagnostic_values)


@pytest.mark.asyncio
async def test_close_without_diagnostics_sink_raises_safe_unavailable_diagnostics() -> None:
    lifecycle = _lifecycle_module()
    LifecycleScope = lifecycle.LifecycleScope
    LifecycleDiagnosticsUnavailableError = lifecycle.LifecycleDiagnosticsUnavailableError
    scope = LifecycleScope("runtime")

    async def failing_task() -> None:
        raise RuntimeError("secret-token raw provider payload")

    async def failing_callback() -> None:
        raise ValueError("secret-token file contents")

    scope.create_task(failing_task(), name="worker")
    scope.register_close_callback("resource", failing_callback)
    await asyncio.sleep(0)

    with pytest.raises(LifecycleDiagnosticsUnavailableError) as exc_info:
        await scope.close()

    events = exc_info.value.diagnostics
    assert isinstance(events, tuple)
    assert len(events) == 2
    assert {event.fields["phase"] for event in events} == {
        "task_done",
        "close_callback",
    }
    assert {event.fields["scope_name"] for event in events} == {"runtime"}

    unsafe_strings = [
        str(exc_info.value),
        *[value for event in events for value in _all_diagnostic_field_values(event)],
    ]
    assert not any("secret-token" in value for value in unsafe_strings)
    assert not any("raw provider payload" in value for value in unsafe_strings)
    assert not any("file contents" in value for value in unsafe_strings)


@pytest.mark.asyncio
async def test_cancelled_close_keeps_unfinished_callback_for_later_retry() -> None:
    LifecycleScope, _ = _lifecycle_types()
    scope = LifecycleScope("runtime")
    callback_started = asyncio.Event()
    callback_calls: list[str] = []

    async def close_resource() -> None:
        callback_calls.append("call")
        if len(callback_calls) == 1:
            callback_started.set()
            await asyncio.Future()

    scope.register_close_callback("resource", close_resource)

    close_task = asyncio.create_task(scope.close())
    await callback_started.wait()
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    await scope.close()

    assert callback_calls == ["call", "call"]


@pytest.mark.asyncio
async def test_failed_diagnostic_sink_keeps_pending_event_for_later_retry() -> None:
    LifecycleScope, _ = _lifecycle_types()

    class SinkTransportError(RuntimeError):
        pass

    class FlakyDiagnosticsSink:
        def __init__(self) -> None:
            self.fail_next_emit = True
            self.events: list[DiagnosticEvent] = []

        async def emit_diagnostic(self, event: DiagnosticEvent) -> None:
            if self.fail_next_emit:
                self.fail_next_emit = False
                raise SinkTransportError("sink transport secret-token")
            self.events.append(event)

    sink = FlakyDiagnosticsSink()
    scope = LifecycleScope("runtime", diagnostics_sink=sink)

    async def failing_task() -> None:
        raise RuntimeError("provider secret-token payload")

    scope.create_task(failing_task(), name="worker")
    await asyncio.sleep(0)

    with pytest.raises(SinkTransportError):
        await scope.close()

    await scope.close()

    assert len(sink.events) == 1
    assert sink.events[0].fields["task_name"] == "worker"
    assert not any(
        "provider secret-token" in value for value in _all_diagnostic_field_values(sink.events[0])
    )


@pytest.mark.asyncio
async def test_create_task_after_close_rejects_and_closes_supplied_coroutine() -> None:
    LifecycleScope, LifecycleScopeClosedError = _lifecycle_types()
    scope = LifecycleScope("runtime")
    await scope.close()

    async def late_worker() -> None:
        await asyncio.sleep(0)

    coroutine: Coroutine[Any, Any, None] = late_worker()

    with pytest.raises(LifecycleScopeClosedError):
        scope.create_task(coroutine, name="late")

    assert coroutine.cr_frame is None
