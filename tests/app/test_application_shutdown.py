from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.app.services.application_shutdown import (
    ApplicationIntentRejectedError,
    ApplicationShutdownCallback,
    ApplicationShutdownCoordinator,
    ApplicationShutdownRegistrationError,
    application_shutdown_callback,
)
from puripuly_heart.core.lifecycle import (
    LIFECYCLE_SHUTDOWN_PHASE_ORDER,
    SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS,
    SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS,
    SHUTDOWN_PHASE_FINAL_DIAGNOSTICS,
    SHUTDOWN_PHASE_FREEZE_INGRESS,
    SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
    SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
)


@pytest.mark.asyncio
async def test_application_shutdown_owns_admission_order_and_terminal_completion() -> None:
    calls: list[str] = []
    coordinator = ApplicationShutdownCoordinator(
        tuple(
            application_shutdown_callback(
                phase=phase,
                owner_name=f"owner-{index}",
                callback_name="close",
                callback=lambda phase=phase: calls.append(phase),
            )
            for index, phase in enumerate(LIFECYCLE_SHUTDOWN_PHASE_ORDER)
        )
    )

    coordinator.admit_intent("manual_translation")
    snapshot = await coordinator.shutdown()

    assert calls == list(LIFECYCLE_SHUTDOWN_PHASE_ORDER)
    assert snapshot.state == "completed"
    assert snapshot.accepting_intents is False
    assert snapshot.terminal is True
    assert snapshot.phase_history == LIFECYCLE_SHUTDOWN_PHASE_ORDER
    with pytest.raises(ApplicationIntentRejectedError):
        coordinator.admit_intent("late_provider_transition")


@pytest.mark.asyncio
async def test_application_shutdown_continues_after_failure_and_closes_logging_last() -> None:
    calls: list[str] = []
    diagnostics = []
    raw_failure = RuntimeError("secret-token raw provider payload")

    async def fail_owner() -> None:
        calls.append("owner-failed")
        raise raw_failure

    async def close_provider() -> None:
        calls.append("provider-closed")

    def close_logging(context) -> None:
        calls.append(
            "logging-closed:"
            + ",".join(type(exception).__name__ for exception in context.cleanup_exceptions)
        )

    coordinator = ApplicationShutdownCoordinator(
        (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="SelfCaptureSessionOwner",
                callback_name="close",
                callback=fail_owner,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS,
                owner_name="ProviderRuntime",
                callback_name="close",
                callback=close_provider,
            ),
            ApplicationShutdownCallback(
                phase=SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS,
                owner_name="RuntimeLoggingService",
                callback_name="close",
                callback=close_logging,
            ),
        ),
        diagnostics_sink=lambda diagnostic: diagnostics.append(diagnostic),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await coordinator.shutdown()

    assert exc_info.value is raw_failure
    assert calls == ["owner-failed", "provider-closed", "logging-closed:RuntimeError"]
    assert coordinator.snapshot.state == "completed_with_failures"
    assert coordinator.snapshot.terminal is True
    assert len(diagnostics) == 1
    assert diagnostics[0].exception_class == "RuntimeError"
    assert "secret-token" not in repr(diagnostics[0])
    assert "provider payload" not in repr(diagnostics[0])


@pytest.mark.asyncio
async def test_application_shutdown_times_out_one_callback_and_continues() -> None:
    calls: list[str] = []

    async def blocked() -> None:
        await asyncio.Event().wait()

    coordinator = ApplicationShutdownCoordinator(
        (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="ExternalProducer",
                callback_name="stop",
                callback=blocked,
                timeout_seconds=0.01,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_FINAL_DIAGNOSTICS,
                owner_name="Diagnostics",
                callback_name="terminal",
                callback=lambda: calls.append("continued"),
            ),
        )
    )

    with pytest.raises(TimeoutError):
        await coordinator.shutdown()

    assert calls == ["continued"]
    assert coordinator.snapshot.failures[0].timed_out is True


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_abandon_owned_shutdown_task() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def slow_close() -> None:
        entered.set()
        await release.wait()
        calls.append("closed")

    coordinator = ApplicationShutdownCoordinator(
        (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="ProviderRuntime",
                callback_name="close",
                callback=slow_close,
            ),
        )
    )
    caller = asyncio.create_task(coordinator.shutdown())
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert coordinator.snapshot.state == "shutting_down"
    release.set()
    first, second = await asyncio.gather(coordinator.shutdown(), coordinator.shutdown())

    assert first == second
    assert calls == ["closed"]
    assert coordinator.snapshot.state == "completed"


@pytest.mark.asyncio
async def test_shutdown_registration_closes_when_shutdown_starts() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def freeze() -> None:
        entered.set()
        await release.wait()

    coordinator = ApplicationShutdownCoordinator(
        (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
                owner_name="Application",
                callback_name="freeze",
                callback=freeze,
            ),
        )
    )
    task = asyncio.create_task(coordinator.shutdown())
    await entered.wait()

    with pytest.raises(ApplicationShutdownRegistrationError):
        coordinator.register_callback(
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="LateOwner",
                callback_name="close",
                callback=lambda: None,
            )
        )

    release.set()
    await task
