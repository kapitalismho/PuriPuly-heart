"""Unified language mapper for UI, STT, and LLM.

Provides consistent language codes and names across:
- Deepgram STT (Nova-3 language codes)
- Soniox STT (language hints)
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

    code: str  # ISO 639-1 code: "ko", "en", etc.
    name: str  # English name: "Korean", "English"


@dataclass(frozen=True, slots=True)
class SttCompatibilityWarning:
    key: str
    language_code: str


@dataclass(frozen=True, slots=True)
class DetectedLanguageForLLM:
    code: str
    name: str


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


def map_detected_language_for_llm(language: str) -> DetectedLanguageForLLM | None:
    normalized = language.strip().replace("_", "-").lower()
    if normalized == "zh":
        return DetectedLanguageForLLM(code="zh", name="Chinese")
    info = get_language_info(normalized)
    if info is None:
        return None
    return DetectedLanguageForLLM(code=info.code, name=info.name)


# Qwen ASR language code mapping (ISO 639-1 -> Qwen ASR codes)
_QWEN_ASR_LANGUAGE_MAP: dict[str, str] = {
    "zh": "zh",
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
_QWEN_AUDIO_ASR_LANGUAGE_MAP: dict[str, str] = {
    "zh": "zh",
    "zh-CN": "zh",
    "zh-TW": "zh",
    "ko": "ko",
    "ja": "ja",
    "en": "en",
    "vi": "vi",
    "th": "th",
    "id": "id",
    "ms": "ms",
    "tl": "tl",
    "hi": "hi",
    "ar": "ar",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "pt": "pt",
    "ru": "ru",
    "it": "it",
    "nl": "nl",
    "sv": "sv",
    "da": "da",
    "fi": "fi",
    "no": "no",
    "el": "el",
    "pl": "pl",
    "cs": "cs",
    "hu": "hu",
    "ro": "ro",
    "bg": "bg",
    "hr": "hr",
    "sk": "sk",
}


def _language_from_map(code: str, language_map: dict[str, str]) -> str:
    if code in language_map:
        return language_map[code]
    base_code = code.split("-")[0].lower()
    return language_map.get(base_code, "en")


def get_qwen3_asr_language(code: str) -> str:
    return _language_from_map(code, _QWEN_ASR_LANGUAGE_MAP)


def get_qwen_asr_language(code: str) -> str:
    return get_qwen3_asr_language(code)


def get_qwen_audio_asr_language(code: str) -> str:
    return _language_from_map(code, _QWEN_AUDIO_ASR_LANGUAGE_MAP)


def qwen_audio_asr_language_hint(code: str) -> str | None:
    if not is_qwen_audio_asr_supported(code):
        return None
    return _language_from_map(code, _QWEN_AUDIO_ASR_LANGUAGE_MAP)


def qwen_audio_asr_language_hints(
    codes: Sequence[str],
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    hints: list[str] = []
    for code in codes:
        normalized = str(code).strip()
        if not normalized:
            continue
        if limit is not None and len(hints) >= limit:
            break
        mapped = qwen_audio_asr_language_hint(normalized)
        if mapped is None or mapped in hints:
            continue
        hints.append(mapped)
    return tuple(hints)


def is_qwen3_asr_supported(code: str) -> bool:
    if code in _QWEN_ASR_LANGUAGE_MAP:
        return True
    return code.split("-")[0].lower() in _QWEN_ASR_LANGUAGE_MAP


def is_qwen_asr_supported(code: str) -> bool:
    return is_qwen3_asr_supported(code)


def is_qwen_audio_asr_supported(code: str) -> bool:
    if code in _QWEN_AUDIO_ASR_LANGUAGE_MAP:
        return True
    return code.split("-")[0].lower() in _QWEN_AUDIO_ASR_LANGUAGE_MAP


_LOCAL_QWEN_LANGUAGE_HINT_MAP: dict[str, str] = {
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "zh": "zh",
    "zh-CN": "zh",
    "zh-TW": "zh",
}


def get_local_qwen_language_hint(code: str) -> str | None:
    """Get a conservative Qwen language code for local GPU STT."""
    normalized = code.strip()
    if normalized in _LOCAL_QWEN_LANGUAGE_HINT_MAP:
        return _LOCAL_QWEN_LANGUAGE_HINT_MAP[normalized]
    base_code = normalized.split("-")[0].lower()
    return _LOCAL_QWEN_LANGUAGE_HINT_MAP.get(base_code)


def get_soniox_language_hints(code: str) -> list[str]:
    """Get Soniox language hints from UI language code. Falls back to ['en']."""
    info = get_language_info(code)
    if not info:
        return ["en"]
    base_code = info.code.split("-")[0].lower()
    return [base_code or "en"]


def get_all_language_options() -> Sequence[tuple[str, str]]:
    """Get all supported languages as (code, name) tuples for UI dropdowns.

    Returns sorted list by English name.
    """
    return tuple(
        sorted(
            ((info.code, info.name) for info in SUPPORTED_LANGUAGES.values()), key=lambda x: x[1]
        )
    )


def is_supported_language(code: str) -> bool:
    """Check if a language code is supported."""
    return get_language_info(code) is not None


# Deepgram Nova-3 supported languages (subset of SUPPORTED_LANGUAGES)
_DEEPGRAM_SUPPORTED: set[str] = {
    "bg",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "lt",
    "lv",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sv",
    "tr",
    "uk",
    "vi",
    "zh-CN",
    "zh-TW",
}


def is_deepgram_supported(code: str) -> bool:
    """Check if a language is supported by Deepgram Nova-3."""
    if code in _DEEPGRAM_SUPPORTED:
        return True
    base_code = code.split("-")[0].lower()
    return base_code in _DEEPGRAM_SUPPORTED


def is_soniox_supported(code: str) -> bool:
    """Check if a language is supported by Soniox (UI language list)."""
    return get_language_info(code) is not None


def get_stt_compatibility_warning(
    code: str,
    stt_provider: str,
    stt_model: str | None = None,
) -> SttCompatibilityWarning | None:
    """Return a warning key if the language is not supported by the STT provider."""
    lang_info = get_language_info(code)
    lang_code = lang_info.code if lang_info else code

    if stt_provider == "deepgram" and not is_deepgram_supported(code):
        if is_qwen_asr_supported(code):
            return SttCompatibilityWarning("warning.deepgram_suggest_qwen", lang_code)
        return SttCompatibilityWarning("warning.deepgram_not_supported", lang_code)

    if stt_provider in {"qwen_asr", "qwen_audio"}:
        qwen_supported = (
            is_qwen_audio_asr_supported(code)
            if stt_provider == "qwen_audio"
            else is_qwen_asr_supported(code)
        )
        if not qwen_supported:
            if is_deepgram_supported(code):
                return SttCompatibilityWarning("warning.qwen_suggest_deepgram", lang_code)
            return SttCompatibilityWarning("warning.qwen_not_supported", lang_code)
    elif stt_provider == "local_qwen" and not is_qwen_asr_supported(code):
        if is_deepgram_supported(code):
            return SttCompatibilityWarning("warning.qwen_suggest_deepgram", lang_code)
        return SttCompatibilityWarning("warning.qwen_not_supported", lang_code)

    if stt_provider == "soniox" and not is_soniox_supported(code):
        return SttCompatibilityWarning("warning.soniox_not_supported", lang_code)

    return None
