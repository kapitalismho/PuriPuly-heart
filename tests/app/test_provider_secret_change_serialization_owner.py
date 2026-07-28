from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.app.services.provider_secret_change_serialization import (
    ProviderSecretChangeSerializationOwner,
)


@pytest.mark.asyncio
async def test_owner_serializes_awaited_operations_in_admission_order() -> None:
    owner = ProviderSecretChangeSerializationOwner()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    events: list[str] = []

    async def first() -> str:
        events.append("first-start")
        first_started.set()
        await release_first.wait()
        events.append("first-end")
        return "first"

    async def second() -> str:
        events.append("second-start")
        return "second"

    first_task = asyncio.create_task(owner.run(first))
    await first_started.wait()
    second_task = asyncio.create_task(owner.run(second))
    await asyncio.sleep(0)

    assert events == ["first-start"]

    release_first.set()
    assert await first_task == "first"
    assert await second_task == "second"
    assert events == ["first-start", "first-end", "second-start"]


@pytest.mark.asyncio
async def test_owner_releases_serialization_after_operation_failure() -> None:
    owner = ProviderSecretChangeSerializationOwner()

    async def fail() -> bool:
        raise RuntimeError("operation detail")

    with pytest.raises(RuntimeError, match="operation detail"):
        await owner.run(fail)

    assert await owner.run(lambda: asyncio.sleep(0, result=True)) is True
    assert owner.lifecycle_owner_snapshot() == {
        "owner": "ProviderSecretChangeSerializationOwner",
        "resource_fields": ("_lock",),
        "operation_policy": "serialize awaited provider-secret changes in admission order",
        "cancellation_policy": "the active operation owns its completion semantics",
        "shutdown_policy": "no background task or external resource is retained",
    }
