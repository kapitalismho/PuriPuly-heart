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
        source_language: str,
        target_language: str,
        domain_prompt: str = "",
        context_pairs: list[dict[str, str]] | None = None,
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
        context_pairs: list[dict[str, str]] | None = None,
    ) -> Translation:
        _ = context
        domain_prompt = system_prompt
        client = self.client or DashScopeQwenClient(
            api_key=self.api_key, model=self.model, base_url=self.base_url
        )
        translated = await client.translate(
            text=text,
            source_language=source_language,
            target_language=target_language,
            domain_prompt=domain_prompt,
            context_pairs=context_pairs,
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

    @staticmethod
    def _normalize_language_code(code: str) -> str:
        if not code:
            return "auto"
        normalized = code.lower()
        if normalized in {"auto"}:
            return "auto"
        if normalized in {"zh-cn", "zh-hans", "zh"}:
            return "zh"
        if normalized in {"zh-tw", "zh-hant", "zh_tw"}:
            return "zh_tw"
        return normalized.split("-")[0]

    async def translate(
        self,
        *,
        text: str,
        source_language: str,
        target_language: str,
        domain_prompt: str = "",
        context_pairs: list[dict[str, str]] | None = None,
    ) -> str:
        import dashscope  # type: ignore

        logger.info(f"[LLM] Request: '{text}' -> {source_language} to {target_language}")

        def _call() -> str:
            dashscope.api_key = self.api_key
            dashscope.base_http_api_url = self.base_url
            translation_options = {
                "source_lang": self._normalize_language_code(source_language),
                "target_lang": self._normalize_language_code(target_language),
            }
            if domain_prompt:
                translation_options["domains"] = domain_prompt
            if context_pairs:
                translation_options["tm_list"] = context_pairs
            response = dashscope.Generation.call(
                model=self.model,
                messages=[{"role": "user", "content": text}],
                result_format="message",
                translation_options=translation_options,
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
