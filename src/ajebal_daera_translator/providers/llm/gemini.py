from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol
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


@dataclass(slots=True)
class GeminiLLMProvider:
    api_key: str
    model: str = "gemini-3-flash-preview"
    client: GeminiClient | None = None

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
    ) -> Translation:
        client = self.client or GoogleGenaiGeminiClient(api_key=self.api_key, model=self.model)
        translated = await client.translate(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
        )
        return Translation(utterance_id=utterance_id, text=translated)


@dataclass(slots=True)
class GoogleGenaiGeminiClient:
    api_key: str
    model: str

    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
    ) -> str:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        prompt = f"Source language: {source_language}\nTarget language: {target_language}\n\n{text}"

        def _call() -> str:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                ),
            )
            if getattr(response, "text", None):
                return str(response.text).strip()
            raise RuntimeError("Gemini response did not contain text")

        return await asyncio.to_thread(_call)
