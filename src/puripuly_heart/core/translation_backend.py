from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from puripuly_heart.domain.models import Translation


@dataclass(frozen=True, slots=True)
class TranslationBackendRequest:
    utterance_id: UUID
    text: str
    system_prompt: str
    source_language: str
    target_language: str
    context: str = ""


class LegacyTranslationProvider(Protocol):
    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation: ...

    async def close(self) -> None: ...


class TranslationBackend(ABC):
    @abstractmethod
    async def translate(self, request: TranslationBackendRequest) -> Translation: ...

    @abstractmethod
    async def close(self) -> None: ...


@dataclass(slots=True)
class LlmTranslationBackend(TranslationBackend):
    provider: LegacyTranslationProvider

    async def translate(self, request: TranslationBackendRequest) -> Translation:
        return await self.provider.translate(
            utterance_id=request.utterance_id,
            text=request.text,
            system_prompt=request.system_prompt,
            source_language=request.source_language,
            target_language=request.target_language,
            context=request.context,
        )

    async def close(self) -> None:
        await self.provider.close()


__all__ = [
    "LegacyTranslationProvider",
    "LlmTranslationBackend",
    "TranslationBackend",
    "TranslationBackendRequest",
]
