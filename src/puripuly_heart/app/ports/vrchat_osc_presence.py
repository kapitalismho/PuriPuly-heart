from __future__ import annotations

from typing import Protocol

from puripuly_heart.core.osc.vrchat_osc_presence_contract import VrchatOscPresence


class VrchatOscPresencePort(Protocol):
    async def should_prompt_enable_osc(self, *, port: int) -> bool | None: ...


__all__ = ["VrchatOscPresence", "VrchatOscPresencePort"]
