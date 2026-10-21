from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class ApplicationControl(Protocol):
    instance_id: str | None

    def bind_instance(self, instance_id: str) -> None: ...
    def freeze_ingress(self) -> None: ...

    async def drain_operations(self) -> None: ...

    def capabilities(self) -> dict: ...

    async def query(self, name: str, arguments: dict) -> dict: ...

    async def submit(
        self,
        command: str,
        arguments: dict,
        *,
        request_id: str,
        expected_revision: int | None = None,
    ) -> dict: ...

    async def operation(self, operation_id: str) -> dict: ...

    async def wait(self, operation_id: str, timeout: float | None = None) -> dict: ...

    async def cancel(self, operation_id: str) -> dict: ...

    def subscribe(
        self,
        *,
        topics: list[str],
        channel: str | None = None,
        include_transcripts: bool = False,
        include_translations: bool = False,
        after: int | None = None,
        cursor_instance_id: str | None = None,
    ) -> AsyncIterator[dict]: ...
