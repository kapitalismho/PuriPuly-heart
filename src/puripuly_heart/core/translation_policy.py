from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TranslationContextPolicy = Literal["integrated_preferred"]


@dataclass(frozen=True, slots=True)
class TranslationRuntimePolicy:
    fast_translation_enabled: bool = True
    context_policy: TranslationContextPolicy = "integrated_preferred"

    def __post_init__(self) -> None:
        if not self.fast_translation_enabled:
            raise ValueError("Fast Translation is a fixed enabled policy")


FIXED_TRANSLATION_POLICY = TranslationRuntimePolicy()


__all__ = [
    "FIXED_TRANSLATION_POLICY",
    "TranslationContextPolicy",
    "TranslationRuntimePolicy",
]
