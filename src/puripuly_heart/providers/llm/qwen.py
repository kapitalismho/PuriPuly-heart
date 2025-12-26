from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from puripuly_heart.domain.models import Translation

logger = logging.getLogger(__name__)


class QwenClient(Protocol):
    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> str: ...


@dataclass(slots=True)
class QwenLLMProvider:
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
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
        context: str = "",
    ) -> Translation:
        client = self.client or DashScopeQwenClient(
            api_key=self.api_key, model=self.model, base_url=self.base_url
        )
        translated = await client.translate(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )
        return Translation(utterance_id=utterance_id, text=translated)

    async def close(self) -> None:
        pass

    @staticmethod
    async def verify_api_key(
        api_key: str, base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    ) -> bool:
        if not api_key:
            return False
        try:
            import dashscope  # type: ignore

            def _check():
                try:
                    dashscope.api_key = api_key
                    dashscope.base_http_api_url = base_url
                    # Use qwen-mt-lite for a cheap/fast check
                    response = dashscope.Generation.call(
                        model="qwen-mt-lite",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1,
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
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"

    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> str:
        import dashscope  # type: ignore

        # Qwen-MT uses Custom Prompt (system role not supported)
        # Template variables (${sourceName}, ${targetName}) are already substituted by hub.py
        # Use XML tags to clearly separate system prompt, context, and translation target
        if context:
            full_prompt = f"""<system_prompt>
{system_prompt}
</system_prompt>

<context>
{context}
</context>

<translate>
{text}
</translate>"""
            logger.info(
                f"[LLM] Request with context: '{text}' -> {source_language} to {target_language}"
            )
        else:
            full_prompt = f"""<system_prompt>
{system_prompt}
</system_prompt>

<translate>
{text}
</translate>"""
            logger.info(f"[LLM] Request: '{text}' -> {source_language} to {target_language}")

        def _call() -> str:
            dashscope.api_key = self.api_key
            dashscope.base_http_api_url = self.base_url
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
