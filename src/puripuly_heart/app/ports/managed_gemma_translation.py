from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from puripuly_heart.core.local_translation.assets import GEMMA_MODEL_ID

ManagedGemmaBackend = Literal["cpu", "gpu"]


@dataclass(frozen=True, slots=True)
class ManagedGemmaTranslationSelection:
    backend: ManagedGemmaBackend
    source_language: str
    target_language: str
    system_prompt: str
    model_id: str = GEMMA_MODEL_ID
    vulkan_device: str = "Vulkan0"

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "gpu"}:
            raise ValueError("managed Gemma backend must be cpu or gpu")
        if not self.source_language.strip() or not self.target_language.strip():
            raise ValueError("managed Gemma language pair must be non-empty")
        if not self.model_id.strip():
            raise ValueError("managed Gemma model id must be non-empty")
        if not self.vulkan_device.strip():
            raise ValueError("managed Gemma vulkan device must be non-empty")


__all__ = ["ManagedGemmaBackend", "ManagedGemmaTranslationSelection"]
