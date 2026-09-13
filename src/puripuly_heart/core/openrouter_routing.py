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
    DEEPSEEK_V4_FLASH_CHINA = "deepseek_v4_flash_china"
    DEEPSEEK_V4_FLASH_41_STRICT = "deepseek_v4_flash_41_strict"
    GEMMA4_31B_MODELRUN_ONLY = "gemma4_31b_modelrun_only"
