from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

from puripuly_heart.domain.models import Translation


class LLMProvider:
    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Translation:
        _ = (
            utterance_id,
            text,
            system_prompt,
            source_language,
            target_language,
            context,
            scene_participant_count,
            max_output_tokens,
        )
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class SemaphoreLLMProvider(LLMProvider):
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
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Translation:
        kwargs = {
            "utterance_id": utterance_id,
            "text": text,
            "system_prompt": system_prompt,
            "source_language": source_language,
            "target_language": target_language,
            "context": context,
            "scene_participant_count": scene_participant_count,
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        # A race can publish its winner before a started loser releases the
        # provider resource. Transfer this permit to the race's tracked cleanup.
        from puripuly_heart.core.llm.fallback_racing import FallbackRacingLLMProvider

        if isinstance(self.inner, FallbackRacingLLMProvider):
            await self.semaphore.acquire()
            return await self.inner._translate(**kwargs, release_permit=self.semaphore.release)
        async with self.semaphore:
            return await self.inner.translate(**kwargs)  # type: ignore[arg-type]

    async def close(self) -> None:
        await self.inner.close()
