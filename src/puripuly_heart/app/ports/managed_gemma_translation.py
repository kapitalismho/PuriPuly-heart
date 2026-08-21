from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ManagedGemmaBackend = Literal["cpu", "gpu"]


@dataclass(frozen=True, slots=True)
class ManagedGemmaTranslationSelection:
    backend: ManagedGemmaBackend
    source_language: str
    target_language: str
    system_prompt: str

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "gpu"}:
            raise ValueError("managed Gemma backend must be cpu or gpu")
        if not self.source_language.strip() or not self.target_language.strip():
            raise ValueError("managed Gemma language pair must be non-empty")


__all__ = ["ManagedGemmaBackend", "ManagedGemmaTranslationSelection"]
