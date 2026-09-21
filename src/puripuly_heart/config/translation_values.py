from __future__ import annotations

from enum import Enum


class TranslationModel(str, Enum):
    GEMMA4_26B_31B = "gemma4_26b_31b"
    GEMMA4_31B = "gemma4_31b"
    GEMMA4 = "gemma4"
    DEEPSEEK_V4_FLASH = "deepseek_v4_flash"
    DEEPSEEK_V4_FLASH_41 = "deepseek_v4_flash_41"
    GEMINI_FLASH = "gemini_flash"
    QWEN_38_FLASH = "qwen38_flash"
    MANAGED_GEMMA = "managed_gemma"
    LOCAL_LLM = "local_llm"
    CUSTOM_HTTP = "custom_http"


class TranslationConnection(str, Enum):
    MANAGED = "managed"
    MANAGED_CHINA = "managed_china"
    OPENROUTER = "openrouter"
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
    ),
    TranslationModel.GEMMA4: (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.DEEPSEEK_V4_FLASH: (
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.DEEPSEEK_V4_FLASH_41: (
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
        TranslationConnection.OPENROUTER,
        TranslationConnection.OFFICIAL_BYOK,
    ),
    TranslationModel.GEMINI_FLASH: (
        TranslationConnection.OFFICIAL_BYOK,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.QWEN_38_FLASH: (TranslationConnection.OFFICIAL_BYOK,),
    TranslationModel.MANAGED_GEMMA: (
        TranslationConnection.CPU,
        TranslationConnection.GPU,
    ),
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
    if model == TranslationModel.GEMINI_FLASH:
        return TranslationConnection.OFFICIAL_BYOK
    supported_connections = supported_translation_connections(model)
    for connection in TRANSLATION_CONNECTION_PRIORITY:
        if connection in supported_connections:
            return connection
    return supported_connections[0]


def provider_llm_for_translation(model: str, connection: str) -> str:
    if model == TranslationModel.MANAGED_GEMMA:
        return "managed_gemma"
    if model == "local_llm":
        return "local_llm"
    if model == "deepseek_v4_flash_41" and connection == "official_byok":
        return "deepseek"
    if model == "gemini_flash":
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
