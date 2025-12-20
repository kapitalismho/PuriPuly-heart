"""Unified language mapper for UI, STT, and LLM.

Provides consistent language codes and names across:
- Deepgram STT (Nova-3 language codes)
- LLM prompts (Gemini, Qwen)
- UI display

All supported languages are from Nova-3's supported language list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class LanguageInfo:
    """Language information for mapping."""
    code: str   # ISO 639-1 code: "ko", "en", etc.
    name: str   # English name: "Korean", "English"


# Supported languages for UI (union of Deepgram Nova-3 + Qwen ASR)
SUPPORTED_LANGUAGES: dict[str, LanguageInfo] = {
    "ar": LanguageInfo(code="ar", name="Arabic"),
    "bg": LanguageInfo(code="bg", name="Bulgarian"),
    "ca": LanguageInfo(code="ca", name="Catalan"),
    "cs": LanguageInfo(code="cs", name="Czech"),
    "da": LanguageInfo(code="da", name="Danish"),
    "de": LanguageInfo(code="de", name="German"),
    "el": LanguageInfo(code="el", name="Greek"),
    "en": LanguageInfo(code="en", name="English"),
    "es": LanguageInfo(code="es", name="Spanish"),
    "et": LanguageInfo(code="et", name="Estonian"),
    "fi": LanguageInfo(code="fi", name="Finnish"),
    "fr": LanguageInfo(code="fr", name="French"),
    "hi": LanguageInfo(code="hi", name="Hindi"),
    "hu": LanguageInfo(code="hu", name="Hungarian"),
    "id": LanguageInfo(code="id", name="Indonesian"),
    "it": LanguageInfo(code="it", name="Italian"),
    "ja": LanguageInfo(code="ja", name="Japanese"),
    "ko": LanguageInfo(code="ko", name="Korean"),
    "lt": LanguageInfo(code="lt", name="Lithuanian"),
    "lv": LanguageInfo(code="lv", name="Latvian"),
    "ms": LanguageInfo(code="ms", name="Malay"),
    "nl": LanguageInfo(code="nl", name="Dutch"),
    "no": LanguageInfo(code="no", name="Norwegian"),
    "pl": LanguageInfo(code="pl", name="Polish"),
    "pt": LanguageInfo(code="pt", name="Portuguese"),
    "ro": LanguageInfo(code="ro", name="Romanian"),
    "ru": LanguageInfo(code="ru", name="Russian"),
    "sk": LanguageInfo(code="sk", name="Slovak"),
    "sv": LanguageInfo(code="sv", name="Swedish"),
    "th": LanguageInfo(code="th", name="Thai"),
    "tr": LanguageInfo(code="tr", name="Turkish"),
    "uk": LanguageInfo(code="uk", name="Ukrainian"),
    "vi": LanguageInfo(code="vi", name="Vietnamese"),
    "zh-CN": LanguageInfo(code="zh-CN", name="Chinese (Simplified)"),
    "zh-TW": LanguageInfo(code="zh-TW", name="Chinese (Traditional)"),
}


def get_language_info(code: str) -> LanguageInfo | None:
    """Get language info by code. Returns None if not supported."""
    # 1. Try exact match (e.g. "zh-CN", "zh-TW")
    if code in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[code]

    # 2. Normalize: strip regional suffix (e.g., "ko-KR" -> "ko")
    normalized = code.split("-")[0].lower()
    return SUPPORTED_LANGUAGES.get(normalized)


def get_deepgram_language(code: str) -> str:
    """Get Deepgram-compatible language code. Falls back to 'en' if unknown."""
    info = get_language_info(code)
    return info.code if info else "en"


def get_llm_language_name(code: str) -> str:
    """Get human-readable language name for LLM prompts. Falls back to 'English'."""
    info = get_language_info(code)
    return info.name if info else "English"


# Qwen ASR language code mapping (ISO 639-1 -> Qwen ASR codes)
_QWEN_ASR_LANGUAGE_MAP: dict[str, str] = {
    "zh-CN": "zh",
    "zh-TW": "zh",  # Qwen ASR uses "zh" for both Mandarin variants
    "ko": "ko",
    "ja": "ja",
    "en": "en",
    "de": "de",
    "ru": "ru",
    "fr": "fr",
    "pt": "pt",
    "ar": "ar",
    "it": "it",
    "es": "es",
    "hi": "hi",
    "id": "id",
    "th": "th",
    "tr": "tr",
    "uk": "uk",
    "vi": "vi",
    "cs": "cs",
    "da": "da",
    "fi": "fi",
    "ms": "ms",
    "no": "no",
    "pl": "pl",
    "sv": "sv",
}


def get_qwen_asr_language(code: str) -> str:
    """Get Qwen-ASR compatible language code. Falls back to 'en' if unknown."""
    # Try exact match first (e.g., "zh-CN")
    if code in _QWEN_ASR_LANGUAGE_MAP:
        return _QWEN_ASR_LANGUAGE_MAP[code]
    # Try base code (e.g., "ko-KR" -> "ko")
    base_code = code.split("-")[0].lower()
    return _QWEN_ASR_LANGUAGE_MAP.get(base_code, "en")


def get_all_language_options() -> Sequence[tuple[str, str]]:
    """Get all supported languages as (code, name) tuples for UI dropdowns.
    
    Returns sorted list by English name.
    """
    return tuple(
        sorted(
            ((info.code, info.name) for info in SUPPORTED_LANGUAGES.values()),
            key=lambda x: x[1]
        )
    )


def is_supported_language(code: str) -> bool:
    """Check if a language code is supported."""
    return get_language_info(code) is not None


# Deepgram Nova-3 supported languages (subset of SUPPORTED_LANGUAGES)
_DEEPGRAM_SUPPORTED: set[str] = {
    "bg", "ca", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
    "hi", "hu", "id", "it", "ja", "ko", "lt", "lv", "ms", "nl", "no",
    "pl", "pt", "ro", "ru", "sk", "sv", "tr", "uk", "vi", "zh-CN", "zh-TW",
}


def is_deepgram_supported(code: str) -> bool:
    """Check if a language is supported by Deepgram Nova-3."""
    if code in _DEEPGRAM_SUPPORTED:
        return True
    base_code = code.split("-")[0].lower()
    return base_code in _DEEPGRAM_SUPPORTED


def is_qwen_asr_supported(code: str) -> bool:
    """Check if a language is supported by Qwen ASR."""
    if code in _QWEN_ASR_LANGUAGE_MAP:
        return True
    base_code = code.split("-")[0].lower()
    return base_code in _QWEN_ASR_LANGUAGE_MAP


def get_stt_compatibility_warning(code: str, stt_provider: str) -> str | None:
    """Get a warning message if the language is not fully supported by the STT provider.

    Returns None if the language is fully supported, or a warning message otherwise.
    Suggests an alternative STT provider if available.
    """
    lang_info = get_language_info(code)
    lang_name = lang_info.name if lang_info else code

    if stt_provider == "deepgram" and not is_deepgram_supported(code):
        if is_qwen_asr_supported(code):
            return f"{lang_name} is not supported by Deepgram. Use Qwen ASR instead."
        return f"{lang_name} is not supported by Deepgram."

    if stt_provider == "qwen_asr" and not is_qwen_asr_supported(code):
        if is_deepgram_supported(code):
            return f"{lang_name} is not supported by Qwen ASR. Use Deepgram instead."
        return f"{lang_name} is not supported by Qwen ASR."

    return None
