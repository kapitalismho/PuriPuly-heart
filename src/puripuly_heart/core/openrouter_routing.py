from __future__ import annotations

from enum import Enum


class OpenRouterRoutingMode(str, Enum):
    LATENCY = "latency"


class OpenRouterProviderRouting(str, Enum):
    DEFAULT = "default"
    GEMMA4_26B_31B_LATENCY = "gemma4_26b_31b_latency"
    GEMMA4_31B_KOREA = "gemma4_31b_korea"
    DEEPSEEK_V4_FLASH_41_STRICT = "deepseek_v4_flash_41_strict"
    GEMMA4_31B_MODELRUN_ONLY = "gemma4_31b_modelrun_only"
