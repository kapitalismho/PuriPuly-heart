from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.app.services.managed_status_refresh import ManagedStatusRefreshOwner


@pytest.mark.asyncio
async def test_owner_names_both_refresh_kinds_and_cancels_them_on_close() -> None:
    entered = 0
    both_entered = asyncio.Event()
    cancelled: list[str] = []

    async def blocked(kind: str) -> None:
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(kind)
            raise

    owner = ManagedStatusRefreshOwner()

    assert owner.schedule_status_refresh(lambda: blocked("status")) is True
    assert owner.schedule_trial_usage_refresh(lambda: blocked("trial_usage")) is True
    await both_entered.wait()

    assert owner.active_task_names == ("status-1", "trial_usage-2")

    await owner.close()

    assert sorted(cancelled) == ["status", "trial_usage"]
    assert owner.active_task_names == ()


@pytest.mark.asyncio
async def test_owner_contains_refresh_failure_and_reports_diagnostics() -> None:
    error = RuntimeError("boom")
    diagnostics: list[tuple[str, dict[str, object], BaseException | None]] = []
    failed = asyncio.Event()

    async def fail() -> None:
        failed.set()
        raise error

    owner = ManagedStatusRefreshOwner(
        diagnostics_sink=lambda event, metadata, exception: diagnostics.append(
            (event, dict(metadata), exception)
        )
    )

    assert owner.schedule_status_refresh(fail) is True
    await failed.wait()
    for _ in range(20):
        if diagnostics and not owner.active_task_names:
            break
        await asyncio.sleep(0)

    assert diagnostics == [
        (
            "managed_status_refresh_failed",
            {"kind": "status", "error_type": "RuntimeError"},
            error,
        )
    ]
    assert owner.active_task_names == ()

    await owner.close()


@pytest.mark.asyncio
async def test_owner_rejects_new_refresh_work_after_ingress_stops() -> None:
    invoked = False

    async def work() -> None:
        nonlocal invoked
        invoked = True

    owner = ManagedStatusRefreshOwner()
    owner.stop_ingress()

    assert owner.schedule_trial_usage_refresh(work) is False
    await asyncio.sleep(0)
    assert invoked is False

    await owner.close()


@pytest.mark.asyncio
async def test_refresh_keeps_admission_identity_and_secondary_notification_cause(caplog):
    from uuid import uuid4

    from puripuly_heart.core.diagnostics import diagnostic_operation

    instance_id, operation_id, request_id = (uuid4().hex for _ in range(3))
    failure = RuntimeError("synthetic_secret_provider_body")
    finished = asyncio.Event()

    async def fail():
        finished.set()
        raise failure

    def fail_notification(event, metadata, exception):
        raise TypeError("synthetic_secret_provider_body")

    owner = ManagedStatusRefreshOwner(diagnostics_sink=fail_notification)
    with diagnostic_operation(instance_id, operation_id, request_id):
        assert owner.schedule_status_refresh(fail)
    await finished.wait()
    await asyncio.sleep(0)
    await owner.close()
    diagnostic = failure.diagnostics
    assert diagnostic.operation_id == operation_id
    assert diagnostic.request_id == request_id
    assert diagnostic.generation == 1
    assert diagnostic.fields["exception_class"] == "RuntimeError"
    causes = [
        record._puripuly_heart_failure[0]
        for record in caplog.records
        if hasattr(record, "_puripuly_heart_failure")
    ]
    assert causes[-1].cause_id == diagnostic.diagnostic_id
    assert causes[-1].fields["exception_class"] == "TypeError"
    assert "synthetic_secret_provider_body" not in caplog.text
