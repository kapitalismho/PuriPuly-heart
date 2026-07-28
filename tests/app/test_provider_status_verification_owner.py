from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.app.services.provider_status_verification import (
    ProviderStatusVerificationOwner,
)


@pytest.mark.asyncio
async def test_owner_names_verification_and_cancels_it_on_close() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    owner = ProviderStatusVerificationOwner()

    assert owner.schedule(blocked) is True
    await entered.wait()

    assert owner.active_task_names == ("verification-1",)

    await owner.close()

    assert cancelled.is_set()
    assert owner.active_task_names == ()


@pytest.mark.asyncio
async def test_owner_contains_verification_failure_and_reports_diagnostics() -> None:
    error = RuntimeError("boom")
    diagnostics: list[tuple[str, dict[str, object], BaseException | None]] = []

    async def fail() -> None:
        raise error

    owner = ProviderStatusVerificationOwner(
        diagnostics_sink=lambda event, metadata, exception: diagnostics.append(
            (event, dict(metadata), exception)
        )
    )

    assert owner.schedule(fail) is True
    for _ in range(20):
        if diagnostics and not owner.active_task_names:
            break
        await asyncio.sleep(0)

    assert diagnostics == [
        (
            "provider_status_verification_failed",
            {"error_type": "RuntimeError"},
            error,
        )
    ]
    assert owner.active_task_names == ()

    await owner.close()


@pytest.mark.asyncio
async def test_owner_rejects_verification_after_ingress_stops() -> None:
    invoked = False

    async def work() -> None:
        nonlocal invoked
        invoked = True

    owner = ProviderStatusVerificationOwner()
    owner.stop_ingress()

    assert owner.schedule(work) is False
    await asyncio.sleep(0)
    assert invoked is False

    await owner.close()
