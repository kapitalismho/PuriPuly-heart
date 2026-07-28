from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from puripuly_heart.app.services.provider_secret_change import (
    ProviderSecretChangeExecution,
    ProviderSecretChangeOwner,
    ProviderSecretChangeRequest,
)
from puripuly_heart.app.services.secret_settings_transaction import (
    SecretClearRequest,
    SecretSetRequest,
)
from puripuly_heart.core.lifecycle import LifecycleDiagnosticsUnavailableError
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TransactionResult,
)


def _result(*, succeeded: bool = True) -> TransactionResult:
    return TransactionResult(
        status=(
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
            if succeeded
            else TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
        ),
        message=None,
        diagnostics=None,
    )


@dataclass
class TransactionStub:
    result: TransactionResult = field(default_factory=_result)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    set_requests: list[SecretSetRequest] = field(default_factory=list)
    clear_requests: list[SecretClearRequest] = field(default_factory=list)
    failure: Exception | None = None

    async def set_provider_secret(self, request: SecretSetRequest) -> TransactionResult:
        self.set_requests.append(request)
        self.started.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        return self.result

    async def clear_provider_secret(self, request: SecretClearRequest) -> TransactionResult:
        self.clear_requests.append(request)
        self.started.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        return self.result


def _execution(
    transaction: TransactionStub,
    *,
    value: str,
    results: list[tuple[TransactionResult, bool]],
) -> ProviderSecretChangeExecution:
    return ProviderSecretChangeExecution(
        transaction=transaction,
        request=ProviderSecretChangeRequest(
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_value=value,
            settings_values={"state": {"provider_verification": {"openrouter": "unknown"}}},
        ),
        result_handler=lambda result, succeeded: results.append((result, succeeded)),
    )


@pytest.mark.asyncio
async def test_owner_invokes_execution_factory_under_ordered_admission_lock() -> None:
    owner = ProviderSecretChangeOwner()
    first = TransactionStub()
    second = TransactionStub()
    events: list[str] = []
    results: list[tuple[TransactionResult, bool]] = []

    def first_factory() -> ProviderSecretChangeExecution:
        events.append("first-factory")
        return _execution(first, value="first-secret", results=results)

    def second_factory() -> ProviderSecretChangeExecution:
        events.append("second-factory")
        return _execution(second, value="second-secret", results=results)

    first_task = asyncio.create_task(owner.change(first_factory))
    await first.started.wait()
    second_task = asyncio.create_task(owner.change(second_factory))
    await asyncio.sleep(0)

    assert events == ["first-factory"]

    first.release.set()
    assert await first_task is True
    await second.started.wait()
    assert events == ["first-factory", "second-factory"]
    second.release.set()
    assert await second_task is True
    assert [succeeded for _, succeeded in results] == [True, True]


@pytest.mark.asyncio
async def test_owner_maps_set_and_clear_requests_and_delivers_result() -> None:
    owner = ProviderSecretChangeOwner()
    set_transaction = TransactionStub()
    clear_transaction = TransactionStub(result=_result(succeeded=False))
    results: list[tuple[TransactionResult, bool]] = []
    raw_secret = "raw-provider-value-123"
    set_transaction.release.set()
    clear_transaction.release.set()

    assert await owner.change(
        lambda: _execution(set_transaction, value=raw_secret, results=results)
    )
    assert not await owner.change(lambda: _execution(clear_transaction, value="", results=results))

    assert len(set_transaction.set_requests) == 1
    assert set_transaction.set_requests[0].secret_value == raw_secret
    assert len(clear_transaction.clear_requests) == 1
    assert [succeeded for _, succeeded in results] == [True, False]
    assert raw_secret not in repr(
        _execution(set_transaction, value=raw_secret, results=results).request
    )


@pytest.mark.asyncio
async def test_owner_finishes_transaction_and_delivers_result_before_replaying_cancellation() -> (
    None
):
    owner = ProviderSecretChangeOwner()
    transaction = TransactionStub()
    results: list[tuple[TransactionResult, bool]] = []
    task = asyncio.create_task(
        owner.change(lambda: _execution(transaction, value="secret", results=results))
    )
    await transaction.started.wait()

    task.cancel()
    transaction.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert results == [(transaction.result, True)]

    next_transaction = TransactionStub()
    next_transaction.release.set()
    assert await owner.change(
        lambda: _execution(next_transaction, value="next-secret", results=results)
    )


@pytest.mark.asyncio
async def test_owner_releases_serialization_after_transaction_failure() -> None:
    owner = ProviderSecretChangeOwner()
    failed = TransactionStub(failure=RuntimeError("operation detail"))
    failed.release.set()
    results: list[tuple[TransactionResult, bool]] = []

    with pytest.raises(LifecycleDiagnosticsUnavailableError):
        await owner.change(lambda: _execution(failed, value="secret", results=results))

    succeeded = TransactionStub()
    succeeded.release.set()
    assert await owner.change(lambda: _execution(succeeded, value="next-secret", results=results))


def test_owner_declares_transaction_lifecycle_policy() -> None:
    owner = ProviderSecretChangeOwner()

    assert owner.lifecycle_owner_snapshot() == {
        "owner": "ProviderSecretChangeOwner",
        "resource_fields": ("_lock", "per-call LifecycleScope"),
        "operation_policy": (
            "serialize provider-secret transactions and deliver results in admission order"
        ),
        "cancellation_policy": (
            "finish the admitted atomic transaction, deliver its result, then replay cancellation"
        ),
        "shutdown_policy": "no background task or external resource is retained",
    }
