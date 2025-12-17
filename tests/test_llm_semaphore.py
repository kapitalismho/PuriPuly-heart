from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ajebal_daera_translator.core.llm.provider import SemaphoreLLMProvider
from ajebal_daera_translator.domain.models import Translation


@dataclass(slots=True)
class CountingLLM:
    active: int = 0
    peak: int = 0

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
    ) -> Translation:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            return Translation(utterance_id=utterance_id, text=f"{text}-translated")
        finally:
            self.active -= 1


def test_llm_semaphore_limits_concurrency():
    async def run():
        inner = CountingLLM()
        provider = SemaphoreLLMProvider(inner=inner, semaphore=asyncio.Semaphore(2))

        async def one(i: int):
            return await provider.translate(
                utterance_id=__import__("uuid").uuid4(),
                text=f"t{i}",
                system_prompt="",
                source_language="ko-KR",
                target_language="en",
            )

        results = await asyncio.gather(*(one(i) for i in range(8)))
        assert len(results) == 8
        assert inner.peak <= 2

    asyncio.run(run())

