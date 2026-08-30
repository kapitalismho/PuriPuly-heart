from __future__ import annotations

from enum import Enum


class OpenRouterRoutingMode(str, Enum):
    LATENCY = "latency"


class OpenRouterProviderRouting(str, Enum):
    DEFAULT = "default"
    DEEPSEEK_ONLY = "deepseek_only"
    GOOGLE_GEMINI_LATENCY = "google_gemini_latency"
    GEMMA4_26B_31B_LATENCY = "gemma4_26b_31b_latency"
    GEMMA4_31B_LATENCY = "gemma4_31b_latency"
    GEMMA4_26B_LATENCY = "gemma4_26b_latency"
    DEEPSEEK_V4_FLASH_LATENCY = "deepseek_v4_flash_latency"
    GEMMA4_31B_CEREBRAS_ONLY = "gemma4_31b_cerebras_only"
