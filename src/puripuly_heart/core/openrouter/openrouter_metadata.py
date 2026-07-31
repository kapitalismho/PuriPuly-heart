from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpenRouterKeyMetadata:
    limit_usd: float | None
    remaining_usd: float | None
    usage_usd: float | None


__all__ = ["OpenRouterKeyMetadata"]
