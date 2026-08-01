from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TranslationContextPolicy = Literal["integrated_preferred"]


@dataclass(frozen=True, slots=True)
class TranslationRuntimePolicy:
    fast_translation_enabled: bool = True
    context_policy: TranslationContextPolicy = "integrated_preferred"
    first_hedge_delay_ms: int = 1300
    emergency_hedge_delay_ms: int = 4500
    loser_grace_ms: int = 50

    def __post_init__(self) -> None:
        if not self.fast_translation_enabled:
            raise ValueError("Fast Translation is a fixed enabled policy")
        if self.first_hedge_delay_ms < 0:
            raise ValueError("first_hedge_delay_ms must be >= 0")
        if self.emergency_hedge_delay_ms < self.first_hedge_delay_ms:
            raise ValueError("emergency_hedge_delay_ms must be >= first_hedge_delay_ms")
        if self.loser_grace_ms < 0:
            raise ValueError("loser_grace_ms must be >= 0")


FIXED_TRANSLATION_POLICY = TranslationRuntimePolicy()


__all__ = [
    "FIXED_TRANSLATION_POLICY",
    "TranslationContextPolicy",
    "TranslationRuntimePolicy",
]
