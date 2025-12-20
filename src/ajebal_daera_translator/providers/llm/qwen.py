from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from ajebal_daera_translator.domain.models import Translation

logger = logging.getLogger(__name__)


class QwenClient(Protocol):
    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
    ) -> str: ...


@dataclass(slots=True)
class QwenLLMProvider:
    api_key: str
    model: str = "qwen-mt-flash"
    client: QwenClient | None = None

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
    ) -> Translation:
        client = self.client or DashScopeQwenClient(api_key=self.api_key, model=self.model)
        translated = await client.translate(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
        )
        return Translation(utterance_id=utterance_id, text=translated)

    async def close(self) -> None:
        pass

    @staticmethod
    async def verify_api_key(api_key: str) -> bool:
        if not api_key:
            return False
        try:
            import dashscope  # type: ignore

            def _check():
                try:
                    dashscope.api_key = api_key
                    dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
                    # Use qwen-mt-lite for a cheap/fast check
                    response = dashscope.Generation.call(
                        model="qwen-mt-lite",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    return response.status_code == 200
                except Exception:
                    return False

            return await asyncio.to_thread(_check)
        except Exception:
            return False


@dataclass(slots=True)
class DashScopeQwenClient:
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
        import dashscope  # type: ignore

        # Apply template variables to system prompt (like Gemini)
        formatted_system_prompt = system_prompt.format(
            source_language=source_language,
            target_language=target_language,
        ) if "{source_language}" in system_prompt else system_prompt

        # Qwen-MT uses Custom Prompt (system role not supported)
        full_prompt = f"{formatted_system_prompt}\n\n{text}"

        logger.info(f"[LLM] Request: '{text}' -> {source_language} to {target_language}")

        def _call() -> str:
            dashscope.api_key = self.api_key
            dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
            response = dashscope.Generation.call(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                result_format="message",
            )
            output = getattr(response, "output", None)
            if not output:
                raise RuntimeError("DashScope response did not contain output")
            choice = output.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            if not content:
                raise RuntimeError("DashScope response did not contain message content")
            result = str(content).strip()
            logger.info(f"[LLM] Response: '{result}'")
            return result

        return await asyncio.to_thread(_call)
