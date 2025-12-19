from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from ajebal_daera_translator.domain.models import Translation


class GeminiClient(Protocol):
    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
    ) -> str: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class GeminiLLMProvider:
    api_key: str
    model: str = "gemini-3-flash-preview"
    client: GeminiClient | None = None
    _internal_client: GeminiClient | None = field(init=False, default=None, repr=False)

    def _get_client(self) -> GeminiClient:
        if self.client is not None:
            return self.client
        if self._internal_client is None:
            self._internal_client = GoogleGenaiGeminiClient(api_key=self.api_key, model=self.model)
        return self._internal_client

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
    ) -> Translation:
        client = self._get_client()
        translated = await client.translate(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
        )
        return Translation(utterance_id=utterance_id, text=translated)

    async def close(self) -> None:
        if self._internal_client is not None:
            await self._internal_client.close()
            self._internal_client = None


@dataclass(slots=True)
class GoogleGenaiGeminiClient:
    api_key: str
    model: str
    _client: Any = field(init=False, default=None, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai  # type: ignore
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
    ) -> str:
        from google.genai import types  # type: ignore

        # Apply template variables to system prompt
        formatted_system_prompt = system_prompt.format(
            source_language=source_language,
            target_language=target_language,
        ) if "{source_language}" in system_prompt else system_prompt

        client = self._get_client()
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=formatted_system_prompt,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
            ),
        )
        if getattr(response, "text", None):
            return str(response.text).strip()
        raise RuntimeError("Gemini response did not contain text")

    async def close(self) -> None:
        self._client = None
