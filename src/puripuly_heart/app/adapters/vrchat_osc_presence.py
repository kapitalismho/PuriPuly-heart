from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresence
from puripuly_heart.core.osc.vrchat_osc_presence import probe_vrchat_osc_presence
from puripuly_heart.core.owned_thread import run_owned_thread_call


@dataclass(frozen=True, slots=True)
class PsutilVrchatOscPresenceAdapter:
    probe: Callable[..., VrchatOscPresence] = probe_vrchat_osc_presence

    async def observe(self, host: str, port: int) -> VrchatOscPresence:
        return await run_owned_thread_call(lambda: self.probe(host=host, port=port))

    async def should_prompt_enable_osc(self, *, port: int) -> bool | None:
        presence = await self.observe("127.0.0.1", port)
        if presence.vrchat_running is None or (
            presence.vrchat_running and presence.osc_listening is None
        ):
            return None
        return presence.should_prompt_enable_osc
