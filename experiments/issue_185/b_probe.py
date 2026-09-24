"""Deterministic Outcome B smoke through the production racing + semaphore owners.

Run with the repository interpreter and PYTHONPATH=src. No network or paid API.
"""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

from puripuly_heart.core.llm.fallback_racing import (
    FallbackRacingLLMProvider,
    LLMProviderAttempt,
)
from puripuly_heart.core.llm.provider import LLMProvider, SemaphoreLLMProvider
from puripuly_heart.domain.models import Translation


@dataclass
class BarrierProvider(LLMProvider):
    name: str
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    release_cancel: asyncio.Event = field(default_factory=asyncio.Event)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    calls: int = 0
    close_calls: int = 0
    result_ready_ns: int | None = None
    finished_ns: int | None = None

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
    ) -> Translation:
        self.calls += 1
        self.started.set()
        try:
            await self.gate.wait()
            result = Translation(
                utterance_id=utterance_id,
                text=self.name,
                source_text=text,
                source_language=source_language,
                target_language=target_language,
            )
            self.result_ready_ns = time.monotonic_ns()
            return result
        except asyncio.CancelledError:
            self.cancelled.set()
            while not self.release_cancel.is_set():
                try:
                    await self.release_cancel.wait()
                except asyncio.CancelledError:
                    continue
            raise
        finally:
            self.finished_ns = time.monotonic_ns()

    async def close(self) -> None:
        self.close_calls += 1


async def main() -> None:
    loser = BarrierProvider("loser")
    winner = BarrierProvider("winner")
    scheduled = BarrierProvider("scheduled")
    inner = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(loser),
            LLMProviderAttempt(winner, start_after_ms=0),
            LLMProviderAttempt(scheduled, start_after_ms=25),
        ),
        loser_grace_ms=10,
    )
    selected_ns: list[int] = []
    inner._winner_selected = lambda: selected_ns.append(time.monotonic_ns())
    provider = SemaphoreLLMProvider(inner=inner, semaphore=asyncio.Semaphore(1))
    request = dict(
        utterance_id=uuid4(),
        text="synthetic fixture",
        system_prompt="fixture",
        source_language="en",
        target_language="ko",
    )
    operation = asyncio.create_task(provider.translate(**request))
    await asyncio.wait_for(loser.started.wait(), 1)
    await asyncio.wait_for(winner.started.wait(), 1)
    winner.gate.set()
    try:
        result = await asyncio.wait_for(operation, 1)
        consumer_ns = time.monotonic_ns()
        assert result.text == "winner" and winner.result_ready_ns is not None
        assert len(selected_ns) == 1
        assert provider.semaphore.locked()
        # Cross the scheduled hedge deadline before shutdown; no paid late call.
        await asyncio.sleep(0.04)
        assert scheduled.calls == 0
        assert loser.finished_ns is None
        closing = asyncio.create_task(provider.close())
        await asyncio.wait_for(loser.cancelled.wait(), 1)
        assert not closing.done() and loser.close_calls == 0
    finally:
        loser.release_cancel.set()
        await asyncio.wait_for(provider.close(), 1)
    assert loser.finished_ns is not None
    assert [loser.calls, winner.calls, scheduled.calls] == [1, 1, 0]
    assert [loser.close_calls, winner.close_calls, scheduled.close_calls] == [1, 1, 1]
    await asyncio.wait_for(provider.semaphore.acquire(), 1)
    provider.semaphore.release()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = {
        "revision": revision,
        "python": platform.python_version(),
        "scenario": "winner with held loser cancellation, pending scheduled attempt; semaphore limit=1",
        "samples": 1,
        "loser_grace_ms": 10,
        "provider_ready_to_selection_ms": round((selected_ns[0] - winner.result_ready_ns) / 1e6, 3),
        "winner_to_consumer_ms": round((consumer_ns - selected_ns[0]) / 1e6, 3),
        "winner_to_loser_cleanup_ms": round((loser.finished_ns - selected_ns[0]) / 1e6, 3),
        "provider_calls": [loser.calls, winner.calls, scheduled.calls],
        "provider_closes": [loser.close_calls, winner.close_calls, scheduled.close_calls],
        "status": "passed",
    }
    Path(__file__).with_name("b_after.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
