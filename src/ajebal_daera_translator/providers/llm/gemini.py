from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from ajebal_daera_translator.domain.models import Translation

logger = logging.getLogger(__name__)


class GeminiClient(Protocol):
    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
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
        context: str = "",
    ) -> Translation:
        client = self._get_client()
        translated = await client.translate(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )
        return Translation(utterance_id=utterance_id, text=translated)

    async def close(self) -> None:
        if self._internal_client is not None:
            await self._internal_client.close()
            self._internal_client = None

    @staticmethod
    async def verify_api_key(api_key: str) -> bool:
        if not api_key:
            return False
        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=api_key)
            # Try listing models as a lightweight auth check
            async for _ in await client.aio.models.list(config={"page_size": 1}):
                break
            return True
        except Exception:
            return False


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
        context: str = "",
    ) -> str:
        from google.genai import types  # type: ignore

        # Apply template variables to system prompt
        formatted_system_prompt = (
            system_prompt.format(
                source_language=source_language,
                target_language=target_language,
            )
            if "{source_language}" in system_prompt
            else system_prompt
        )

        # Build the message with context if provided
        if context:
            user_message = f"context:\n{context}\n\nTranslate: {text}"
            logger.info(
                f"[LLM] Request with context: '{text}' -> {source_language} to {target_language}"
            )
        else:
            user_message = text
            logger.info(f"[LLM] Request: '{text}' -> {source_language} to {target_language}")

        client = self._get_client()
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=formatted_system_prompt,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
            ),
        )
        if getattr(response, "text", None):
            result = str(response.text).strip()
            logger.info(f"[LLM] Response: '{result}'")
            return result
        logger.error("[LLM] No text in response")
        raise RuntimeError("Gemini response did not contain text")

    async def close(self) -> None:
        self._client = None
