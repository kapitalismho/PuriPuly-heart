from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from puripuly_heart.domain.models import Translation


class LLMProvider(Protocol):
    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        context_pairs: list[dict[str, str]] | None = None,
    ) -> Translation: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class SemaphoreLLMProvider:
    inner: LLMProvider
    semaphore: asyncio.Semaphore

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        context_pairs: list[dict[str, str]] | None = None,
    ) -> Translation:
        async with self.semaphore:
            return await self.inner.translate(
                utterance_id=utterance_id,
                text=text,
                system_prompt=system_prompt,
                source_language=source_language,
                target_language=target_language,
                context=context,
                context_pairs=context_pairs,
            )

    async def close(self) -> None:
        await self.inner.close()
