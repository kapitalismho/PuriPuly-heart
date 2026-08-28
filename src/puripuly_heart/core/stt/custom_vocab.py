from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from puripuly_heart.config.provider_values import MAX_CUSTOM_VOCAB_TERMS

LOCAL_QWEN_MAX_HOTWORDS = 12


@dataclass(frozen=True, slots=True)
class CustomVocabularyRuntimeConfig:
    """Narrow read-only runtime DTO for custom-vocabulary effective-term resolution."""

    enabled: bool
    terms: Mapping[str, list[str]]


def _raw_terms_for_language(
    config: CustomVocabularyRuntimeConfig,
    source_language: str,
) -> list[str]:
    if source_language in config.terms:
        return config.terms[source_language]
    base_language = source_language.split("-")[0].lower()
    return config.terms.get(base_language, [])


def get_effective_custom_terms(
    config: CustomVocabularyRuntimeConfig,
    source_language: str,
) -> list[str]:
    if not config.enabled:
        return []

    raw_terms = _raw_terms_for_language(config, source_language)
    effective_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in raw_terms:
        normalized_term = term.strip()
        if not normalized_term or normalized_term in seen_terms:
            continue
        if len(effective_terms) >= MAX_CUSTOM_VOCAB_TERMS:
            break
        seen_terms.add(normalized_term)
        effective_terms.append(normalized_term)
    return effective_terms


def _normalize_local_qwen_hotword(term: str) -> str:
    return " ".join(term.replace(",", " ").split())


def get_effective_local_qwen_hotwords(
    config: CustomVocabularyRuntimeConfig,
    source_language: str,
) -> list[str]:
    if not config.enabled:
        return []

    raw_terms = _raw_terms_for_language(config, source_language)
    effective_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in raw_terms:
        normalized_term = _normalize_local_qwen_hotword(term)
        if not normalized_term or normalized_term in seen_terms:
            continue
        if len(effective_terms) >= LOCAL_QWEN_MAX_HOTWORDS:
            break
        seen_terms.add(normalized_term)
        effective_terms.append(normalized_term)
    return effective_terms


__all__ = [
    "LOCAL_QWEN_MAX_HOTWORDS",
    "CustomVocabularyRuntimeConfig",
    "get_effective_custom_terms",
    "get_effective_local_qwen_hotwords",
]
