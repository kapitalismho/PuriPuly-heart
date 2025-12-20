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


# Nova-3 supported languages (simplified codes, no regional variants)
SUPPORTED_LANGUAGES: dict[str, LanguageInfo] = {
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
