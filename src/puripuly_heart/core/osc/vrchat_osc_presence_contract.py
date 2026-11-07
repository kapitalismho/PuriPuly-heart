from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VrchatOscPresence:
    vrchat_running: bool | None
    osc_listening: bool | None
    listener_status: str = "unknown"

    @property
    def should_prompt_enable_osc(self) -> bool:
        return self.vrchat_running is True and self.osc_listening is False
