from __future__ import annotations

from enum import Enum


class TranslationModel(str, Enum):
    INDEX_TRANSLATE_35B = "index_translate_35b"
    GEMMA4_26B_31B = "gemma4_26b_31b"
    DEEPSEEK_V4_FLASH = "deepseek_v4_flash"
    DEEPSEEK_V4_FLASH_41 = "deepseek_v4_flash_41"
    GEMINI_FLASH = "gemini_flash"
    GPT_6_LUNA = "gpt_6_luna"
    QWEN_38_FLASH = "qwen38_flash"
    MANAGED_GEMMA = "managed_gemma"
    MANAGED_GEMMA_26B = "managed_gemma_26b"
    LOCAL_LLM = "local_llm"
    CUSTOM_HTTP = "custom_http"


MANAGED_GEMMA_MODELS = (TranslationModel.MANAGED_GEMMA, TranslationModel.MANAGED_GEMMA_26B)


class TranslationConnection(str, Enum):
    PUBLIC_API = "public_api"
    MANAGED = "managed"
    MANAGED_CHINA = "managed_china"
    CHATGPT = "chatgpt"
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
    TranslationModel.INDEX_TRANSLATE_35B: (TranslationConnection.PUBLIC_API,),
    TranslationModel.GEMMA4_26B_31B: (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
    ),
    TranslationModel.DEEPSEEK_V4_FLASH: (TranslationConnection.OPENROUTER,),
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
    TranslationModel.GPT_6_LUNA: (
        TranslationConnection.CHATGPT,
        TranslationConnection.OPENROUTER,
        TranslationConnection.OFFICIAL_BYOK,
    ),
    TranslationModel.MANAGED_GEMMA: (
        TranslationConnection.CPU,
        TranslationConnection.GPU,
    ),
    TranslationModel.MANAGED_GEMMA_26B: (TranslationConnection.GPU,),
    TranslationModel.LOCAL_LLM: (TranslationConnection.OLLAMA,),
    TranslationModel.CUSTOM_HTTP: (TranslationConnection.CUSTOM_HTTP,),
}

TRANSLATION_CONNECTION_PRIORITY: tuple[TranslationConnection, ...] = (
    TranslationConnection.MANAGED,
    TranslationConnection.CHATGPT,
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
    if model == "index_translate_35b":
        return "index_translate"
    if model in MANAGED_GEMMA_MODELS:
        return "managed_gemma"
    if model == "local_llm":
        return "local_llm"
    if model == "deepseek_v4_flash_41" and connection == "official_byok":
        return "deepseek"
    if model == "gpt_6_luna" and connection == "official_byok":
        return "openai"
    if model == "gpt_6_luna" and connection == "chatgpt":
        return "chatgpt"
    if model == "gemini_flash":
        if connection == "openrouter":
            return "openrouter"
        return "gemini"
    if model == "qwen38_flash":
        return "qwen"
    return "openrouter"


__all__ = [
    "MANAGED_GEMMA_MODELS",
    "TRANSLATION_CONNECTIONS_BY_MODEL",
    "TRANSLATION_CONNECTION_PRIORITY",
    "TranslationConnection",
    "TranslationModel",
    "default_translation_connection",
    "provider_llm_for_translation",
    "supported_translation_connections",
]
