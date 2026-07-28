from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

ProviderSecretChangeResultT = TypeVar("ProviderSecretChangeResultT")
ProviderSecretChangeOperation = Callable[
    [],
    Awaitable[ProviderSecretChangeResultT],
]


@dataclass(slots=True)
class ProviderSecretChangeSerializationOwner:
    _lock: asyncio.Lock | None = field(init=False, default=None, repr=False)

    @property
    def owner_name(self) -> str:
        return "ProviderSecretChangeSerializationOwner"

    async def run(
        self,
        operation: ProviderSecretChangeOperation[ProviderSecretChangeResultT],
    ) -> ProviderSecretChangeResultT:
        async with self._serialization_lock():
            return await operation()

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": ("_lock",),
            "operation_policy": "serialize awaited provider-secret changes in admission order",
            "cancellation_policy": "the active operation owns its completion semantics",
            "shutdown_policy": "no background task or external resource is retained",
        }

    def _serialization_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


__all__ = [
    "ProviderSecretChangeOperation",
    "ProviderSecretChangeSerializationOwner",
]
