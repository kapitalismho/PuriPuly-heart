from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from puripuly_heart.app.ports._settings_values import freeze_settings_values
from puripuly_heart.app.services.secret_settings_transaction import (
    SecretClearRequest,
    SecretSetRequest,
)
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TransactionResult,
)


class ProviderSecretChangeTransactionPort(Protocol):
    async def set_provider_secret(self, request: SecretSetRequest) -> TransactionResult: ...

    async def clear_provider_secret(self, request: SecretClearRequest) -> TransactionResult: ...


@dataclass(frozen=True, slots=True)
class ProviderSecretChangeRequest:
    provider: str
    secret_key: str
    secret_value: str = field(repr=False)
    settings_values: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "settings_values",
            freeze_settings_values(self.settings_values),
        )


ProviderSecretChangeResultHandler = Callable[[TransactionResult, bool], None]


@dataclass(frozen=True, slots=True)
class ProviderSecretChangeExecution:
    transaction: ProviderSecretChangeTransactionPort
    request: ProviderSecretChangeRequest
    result_handler: ProviderSecretChangeResultHandler


ProviderSecretChangeExecutionFactory = Callable[[], ProviderSecretChangeExecution]


@dataclass(slots=True)
class ProviderSecretChangeOwner:
    _lock: asyncio.Lock | None = field(init=False, default=None, repr=False)

    @property
    def owner_name(self) -> str:
        return "ProviderSecretChangeOwner"

    async def change(self, execution_factory: ProviderSecretChangeExecutionFactory) -> bool:
        async with self._serialization_lock():
            return await self._execute(execution_factory())

    async def _execute(self, execution: ProviderSecretChangeExecution) -> bool:
        request = execution.request
        scope = LifecycleScope(f"provider-secret-change:{request.provider}")
        if request.secret_value:
            operation = start_lifecycle_task(
                scope,
                execution.transaction.set_provider_secret(
                    SecretSetRequest(
                        secret_key=request.secret_key,
                        secret_value=request.secret_value,
                        settings_values=request.settings_values,
                        expected_settings_revision=None,
                        reason="provider_secret_change",
                        correlation_id=None,
                    )
                ),
                name="transaction",
            )
        else:
            operation = start_lifecycle_task(
                scope,
                execution.transaction.clear_provider_secret(
                    SecretClearRequest(
                        secret_key=request.secret_key,
                        settings_values=request.settings_values,
                        expected_settings_revision=None,
                        reason="provider_secret_change",
                        correlation_id=None,
                    )
                ),
                name="transaction",
            )
        cancelled = False
        try:
            result = await asyncio.shield(operation)
        except asyncio.CancelledError:
            cancelled = True
            result = await operation
        finally:
            await scope.close()
        succeeded = result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
        execution.result_handler(result, succeeded)
        if cancelled:
            raise asyncio.CancelledError
        return succeeded

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": ("_lock", "per-call LifecycleScope"),
            "operation_policy": (
                "serialize provider-secret transactions and deliver results in admission order"
            ),
            "cancellation_policy": (
                "finish the admitted atomic transaction, deliver its result, then replay cancellation"
            ),
            "shutdown_policy": "no background task or external resource is retained",
        }

    def _serialization_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


__all__ = [
    "ProviderSecretChangeExecution",
    "ProviderSecretChangeExecutionFactory",
    "ProviderSecretChangeOwner",
    "ProviderSecretChangeRequest",
    "ProviderSecretChangeResultHandler",
    "ProviderSecretChangeTransactionPort",
]
