from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from puripuly_heart.core.llm.latency import current_request, observe_attempt

if TYPE_CHECKING:
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


class LLMRequestExecution(Protocol):
    @property
    def attempt_count(self) -> int: ...

    async def translate_attempt(self, attempt_index: int) -> Translation: ...

    async def close(self) -> None: ...


class LLMRequestAdmissionPort(Protocol):
    async def admit_request(
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
        max_attempts: int = 2,
    ) -> LLMRequestExecution: ...


@dataclass(slots=True)
class SemaphoreLLMProvider(LLMProvider):
    inner: LLMProvider
    semaphore: asyncio.Semaphore
    provider_name: str = "unknown"
    model: str | None = None

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

        request = current_request()
        queued_at = request.clock() if request is not None else None
        try:
            await self.semaphore.acquire()
        finally:
            if request is not None and queued_at is not None:
                request.queue_ms += max(0, round((request.clock() - queued_at) * 1000))
        if isinstance(self.inner, FallbackRacingLLMProvider):
            return await self.inner._translate(**kwargs, release_permit=self.semaphore.release)
        try:
            with observe_attempt(provider=self.provider_name, model=self.model):
                result = await self.inner.translate(**kwargs)  # type: ignore[arg-type]
                if request is not None:
                    request.winner_attempt = 0
                return result
        finally:
            self.semaphore.release()

    async def close(self) -> None:
        await self.inner.close()
