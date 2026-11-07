from __future__ import annotations

import pytest

from puripuly_heart.app.adapters.vrchat_osc_presence import (
    PsutilVrchatOscPresenceAdapter,
)
from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresence


@pytest.mark.asyncio
async def test_adapter_reports_probe_decision() -> None:
    destinations: list[tuple[str, int]] = []

    def probe(*, host: str, port: int) -> VrchatOscPresence:
        destinations.append((host, port))
        return VrchatOscPresence(vrchat_running=True, osc_listening=False)

    adapter = PsutilVrchatOscPresenceAdapter(probe=probe)

    assert await adapter.should_prompt_enable_osc(port=9012) is True
    assert destinations == [("127.0.0.1", 9012)]
