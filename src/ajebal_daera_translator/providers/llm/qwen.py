from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from ajebal_daera_translator.domain.models import Translation


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
    model: str = "qwen-plus"
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
                    # Use qwen-turbo for a cheap/fast check
                    response = dashscope.Generation.call(
                        model="qwen-turbo",
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

        prompt = f"Source language: {source_language}\nTarget language: {target_language}\n\n{text}"

        def _call() -> str:
            dashscope.api_key = self.api_key
            response = dashscope.Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
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
            return str(content).strip()

        return await asyncio.to_thread(_call)
