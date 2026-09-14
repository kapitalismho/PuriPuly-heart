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
    scene_participant_count: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_output_tokens is None:
            return
        if isinstance(self.max_output_tokens, bool) or not isinstance(self.max_output_tokens, int):
            raise TypeError("max_output_tokens must be an integer")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


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
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Translation: ...

    async def close(self) -> None: ...


class TranslationSecretResolver(Protocol):
    def get(self, key: str) -> str | None: ...


class TranslationBackend(ABC):
    @abstractmethod
    async def translate(self, request: TranslationBackendRequest) -> Translation: ...

    @abstractmethod
    async def close(self) -> None: ...


@dataclass(slots=True)
class LlmTranslationBackend(TranslationBackend):
    provider: LegacyTranslationProvider

    async def translate(self, request: TranslationBackendRequest) -> Translation:
        kwargs = {
            "utterance_id": request.utterance_id,
            "text": request.text,
            "system_prompt": request.system_prompt,
            "source_language": request.source_language,
            "target_language": request.target_language,
            "context": request.context,
            "scene_participant_count": request.scene_participant_count,
        }
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        return await self.provider.translate(**kwargs)  # type: ignore[arg-type]

    async def close(self) -> None:
        await self.provider.close()


__all__ = [
    "LegacyTranslationProvider",
    "LlmTranslationBackend",
    "TranslationBackend",
    "TranslationBackendRequest",
    "TranslationSecretResolver",
]
