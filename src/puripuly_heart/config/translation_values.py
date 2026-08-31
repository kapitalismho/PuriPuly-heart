from __future__ import annotations

from enum import Enum


class TranslationModel(str, Enum):
    GEMMA4_26B_31B = "gemma4_26b_31b"
    GEMMA4_31B = "gemma4_31b"
    GEMMA4 = "gemma4"
    DEEPSEEK_V4_FLASH = "deepseek_v4_flash"
    GEMINI_37_FLASH = "gemini37_flash"
    QWEN_38_FLASH = "qwen38_flash"
    MANAGED_GEMMA = "managed_gemma"
    MANAGED_GEMMA_12B = "managed_gemma_12b"
    LOCAL_LLM = "local_llm"
    CUSTOM_HTTP = "custom_http"


class TranslationConnection(str, Enum):
    MANAGED = "managed"
    MANAGED_CHINA = "managed_china"
    OPENROUTER = "openrouter"
    CEREBRAS = "cerebras"
    OFFICIAL_BYOK = "official_byok"
    OLLAMA = "ollama"
    CPU = "cpu"
    GPU = "gpu"
    CUSTOM_HTTP = "custom_http"


TRANSLATION_CONNECTIONS_BY_MODEL: dict[
    TranslationModel,
    tuple[TranslationConnection, ...],
] = {
    TranslationModel.GEMMA4_26B_31B: (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.GEMMA4_31B: (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
        TranslationConnection.CEREBRAS,
    ),
    TranslationModel.GEMMA4: (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.DEEPSEEK_V4_FLASH: (
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
        TranslationConnection.OPENROUTER,
        TranslationConnection.OFFICIAL_BYOK,
    ),
    TranslationModel.GEMINI_37_FLASH: (
        TranslationConnection.OFFICIAL_BYOK,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.QWEN_38_FLASH: (TranslationConnection.OFFICIAL_BYOK,),
    TranslationModel.MANAGED_GEMMA: (
        TranslationConnection.CPU,
        TranslationConnection.GPU,
    ),
    TranslationModel.MANAGED_GEMMA_12B: (TranslationConnection.GPU,),
    TranslationModel.LOCAL_LLM: (TranslationConnection.OLLAMA,),
    TranslationModel.CUSTOM_HTTP: (TranslationConnection.CUSTOM_HTTP,),
}

TRANSLATION_CONNECTION_PRIORITY: tuple[TranslationConnection, ...] = (
    TranslationConnection.MANAGED,
    TranslationConnection.OPENROUTER,
    TranslationConnection.OFFICIAL_BYOK,
)


def supported_translation_connections(
    model: TranslationModel,
) -> tuple[TranslationConnection, ...]:
    return TRANSLATION_CONNECTIONS_BY_MODEL[model]


def default_translation_connection(model: TranslationModel) -> TranslationConnection:
    if model == TranslationModel.CUSTOM_HTTP:
        return TranslationConnection.CUSTOM_HTTP
    if model == TranslationModel.GEMINI_37_FLASH:
        return TranslationConnection.OFFICIAL_BYOK
    supported_connections = supported_translation_connections(model)
    for connection in TRANSLATION_CONNECTION_PRIORITY:
        if connection in supported_connections:
            return connection
    return supported_connections[0]


_MANAGED_GEMMA_MODELS = frozenset({"managed_gemma", "managed_gemma_12b"})


def provider_llm_for_translation(model: str, connection: str) -> str:
    if model in _MANAGED_GEMMA_MODELS:
        return "managed_gemma"
    if model == "local_llm":
        return "local_llm"
    if model == "gemma4_31b_cerebras" or (model == "gemma4_31b" and connection == "cerebras"):
        return "cerebras"
    if model == "gemini37_flash":
        if connection == "openrouter":
            return "openrouter"
        return "gemini"
    if model == "qwen38_flash":
        return "qwen"
    return "openrouter"


__all__ = [
    "TRANSLATION_CONNECTIONS_BY_MODEL",
    "TRANSLATION_CONNECTION_PRIORITY",
    "TranslationConnection",
    "TranslationModel",
    "default_translation_connection",
    "provider_llm_for_translation",
    "supported_translation_connections",
]
