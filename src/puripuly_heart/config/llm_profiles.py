from __future__ import annotations

from dataclasses import dataclass

LLM_PROVIDER_GEMINI = "gemini"
LLM_PROVIDER_OPENROUTER = "openrouter"
LLM_PROVIDER_QWEN = "qwen"

OPENROUTER_CREDENTIAL_SOURCE_NONE = "none"
OPENROUTER_CREDENTIAL_SOURCE_MANAGED = "managed"
OPENROUTER_CREDENTIAL_SOURCE_BYOK = "byok"

OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT = "google/gemma-4-26b-a4b-it"
OPENROUTER_MODEL_GEMMA_4_31B_IT = "google/gemma-4-31b-it"
OPENROUTER_MODEL_QWEN_35_FLASH_02_23 = "qwen/qwen3.5-flash-02-23"
OPENROUTER_MODEL_DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash-0731"
LEGACY_OPENROUTER_MODEL_DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash"
LEGACY_OPENROUTER_MODEL_GEMINI_31_FLASH_LITE = "google/gemini-3.1-flash-lite"
OPENROUTER_MODEL_GEMINI_37_FLASH = "google/gemini-3.7-flash"

OPENROUTER_SELECTION_ALIAS_GEMMA4_MANAGED = "gemma4_managed"
OPENROUTER_SELECTION_ALIAS_GEMMA4_BYOK = "gemma4_byok"
OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED = "gemma4_26b_31b_managed"
OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK = "gemma4_26b_31b_byok"
OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED = "gemma4_31b_managed"
OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK = "gemma4_31b_byok"
OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_MANAGED = "qwen35_flash_managed"
OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_BYOK = "qwen35_flash_byok"
OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_MANAGED = "deepseek_v4_flash_managed"
OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_BYOK = "deepseek_v4_flash_byok"
OPENROUTER_SELECTION_ALIAS_GEMINI37_FLASH_BYOK = "gemini37_flash_byok"
LEGACY_OPENROUTER_SELECTION_ALIAS_GEMINI31_FLASH_LITE_BYOK = "gemini31_flash_lite_byok"

OPENROUTER_FALLBACK_SELECTION_ALIAS_NONE = "none"
OPENROUTER_FALLBACK_SELECTION_ALIAS_QWEN35_FLASH = "qwen35_flash"
OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH = "deepseek_v4_flash"
OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_CHINA = "deepseek_v4_flash_china"
OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_26B_31B = "gemma4_26b_31b"
OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_31B = "gemma4_31b"

LEGACY_OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMINI25_FLASH_LITE = "gemini25_flash_lite"
LEGACY_OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMINI31_FLASH_LITE = "gemini31_flash_lite"
LEGACY_OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4 = "gemma4"
LEGACY_OPENROUTER_SELECTION_ALIAS_NONE_GEMMA_4_26B_A4B_IT = (
    "openrouter:none:google/gemma-4-26b-a4b-it"
)
LEGACY_OPENROUTER_SELECTION_ALIAS_MANAGED_GEMMA_4_26B_A4B_IT = (
    "openrouter:managed:google/gemma-4-26b-a4b-it"
)
LEGACY_OPENROUTER_SELECTION_ALIAS_BYOK_GEMMA_4_26B_A4B_IT = (
    "openrouter:byok:google/gemma-4-26b-a4b-it"
)
LEGACY_OPENROUTER_SELECTION_ALIAS_NONE_QWEN_35_FLASH_02_23 = (
    "openrouter:none:qwen/qwen3.5-flash-02-23"
)
LEGACY_OPENROUTER_SELECTION_ALIAS_BYOK_QWEN_35_FLASH_02_23 = (
    "openrouter:byok:qwen/qwen3.5-flash-02-23"
)


@dataclass(frozen=True, slots=True)
class LLMSelectionProfile:
    alias: str
    provider: str
    label_key: str
    description_key: str
    gemini_model: str | None = None
    qwen_model: str | None = None
    openrouter_model: str | None = None
    openrouter_models: tuple[str, ...] = ()
    openrouter_source: str = OPENROUTER_CREDENTIAL_SOURCE_NONE


@dataclass(frozen=True, slots=True)
class OpenRouterFallbackProfile:
    alias: str
    label_key: str
    description_key: str
    openrouter_model: str | None = None
    openrouter_models: tuple[str, ...] = ()


PROFILE_BY_ALIAS: dict[str, LLMSelectionProfile] = {
    OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemma4_26b_31b",
        description_key="provider.gemma4_26b_31b.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        openrouter_models=(
            OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
            OPENROUTER_MODEL_GEMMA_4_31B_IT,
        ),
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_MANAGED,
    ),
    OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemma4_26b_31b",
        description_key="provider.gemma4_26b_31b.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        openrouter_models=(
            OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
            OPENROUTER_MODEL_GEMMA_4_31B_IT,
        ),
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_BYOK,
    ),
    OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemma4_31b_openrouter",
        description_key="provider.gemma4_31b_openrouter.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_31B_IT,
        openrouter_models=(OPENROUTER_MODEL_GEMMA_4_31B_IT,),
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_MANAGED,
    ),
    OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemma4_31b_openrouter",
        description_key="provider.gemma4_31b_openrouter.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_31B_IT,
        openrouter_models=(OPENROUTER_MODEL_GEMMA_4_31B_IT,),
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_BYOK,
    ),
    OPENROUTER_SELECTION_ALIAS_GEMMA4_MANAGED: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_GEMMA4_MANAGED,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemma4_managed",
        description_key="provider.gemma4_managed.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        openrouter_models=(OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,),
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_MANAGED,
    ),
    OPENROUTER_SELECTION_ALIAS_GEMMA4_BYOK: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_GEMMA4_BYOK,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemma4_26b_a4b_it",
        description_key="provider.gemma4_26b_a4b_it.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        openrouter_models=(OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,),
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_BYOK,
    ),
    OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_MANAGED: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_MANAGED,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.qwen35_flash_managed",
        description_key="provider.qwen35_flash_managed.description",
        openrouter_model=OPENROUTER_MODEL_QWEN_35_FLASH_02_23,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_MANAGED,
    ),
    OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_BYOK: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_BYOK,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.qwen35_flash_openrouter",
        description_key="provider.qwen35_flash_openrouter.description",
        openrouter_model=OPENROUTER_MODEL_QWEN_35_FLASH_02_23,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_BYOK,
    ),
    OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_MANAGED: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_MANAGED,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.deepseek_v4_flash_managed",
        description_key="provider.deepseek_v4_flash_managed.description",
        openrouter_model=OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_MANAGED,
    ),
    OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_BYOK: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_BYOK,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.deepseek_v4_flash_openrouter",
        description_key="provider.deepseek_v4_flash_openrouter.description",
        openrouter_model=OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_BYOK,
    ),
    OPENROUTER_SELECTION_ALIAS_GEMINI37_FLASH_BYOK: LLMSelectionProfile(
        alias=OPENROUTER_SELECTION_ALIAS_GEMINI37_FLASH_BYOK,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemini37_flash_openrouter",
        description_key="provider.gemini37_flash_openrouter.description",
        openrouter_model=OPENROUTER_MODEL_GEMINI_37_FLASH,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_BYOK,
    ),
}

FALLBACK_PROFILE_BY_ALIAS: dict[str, OpenRouterFallbackProfile] = {
    OPENROUTER_FALLBACK_SELECTION_ALIAS_NONE: OpenRouterFallbackProfile(
        alias=OPENROUTER_FALLBACK_SELECTION_ALIAS_NONE,
        label_key="settings.openrouter_fallback.none",
        description_key="settings.openrouter_fallback.none.description",
        openrouter_model=None,
    ),
    OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH: OpenRouterFallbackProfile(
        alias=OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH,
        label_key="provider.deepseek_v4_flash_fallback",
        description_key="provider.deepseek_v4_flash_fallback.description",
        openrouter_model=OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
    ),
    OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_CHINA: OpenRouterFallbackProfile(
        alias=OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_CHINA,
        label_key="provider.deepseek_v4_flash_china_fallback",
        description_key="provider.deepseek_v4_flash_china_fallback.description",
        openrouter_model=OPENROUTER_MODEL_DEEPSEEK_V4_FLASH,
    ),
    OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_26B_31B: OpenRouterFallbackProfile(
        alias=OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_26B_31B,
        label_key="settings.fallback.openrouter_gemma4_26b_31b",
        description_key="settings.fallback.openrouter_gemma4_26b_31b.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        openrouter_models=(
            OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
            OPENROUTER_MODEL_GEMMA_4_31B_IT,
        ),
    ),
    OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_31B: OpenRouterFallbackProfile(
        alias=OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_31B,
        label_key="settings.fallback.openrouter_gemma4_31b",
        description_key="settings.fallback.openrouter_gemma4_31b.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_31B_IT,
        openrouter_models=(OPENROUTER_MODEL_GEMMA_4_31B_IT,),
    ),
}

OPENROUTER_FALLBACK_MODEL_BY_ALIAS: dict[str, str | None] = {
    alias: profile.openrouter_model for alias, profile in FALLBACK_PROFILE_BY_ALIAS.items()
}

LEGACY_PROFILE_BY_ALIAS: dict[str, LLMSelectionProfile] = {
    LEGACY_OPENROUTER_SELECTION_ALIAS_NONE_GEMMA_4_26B_A4B_IT: LLMSelectionProfile(
        alias=LEGACY_OPENROUTER_SELECTION_ALIAS_NONE_GEMMA_4_26B_A4B_IT,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemma4_26b_a4b_it",
        description_key="provider.gemma4_26b_a4b_it.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_NONE,
    ),
    LEGACY_OPENROUTER_SELECTION_ALIAS_MANAGED_GEMMA_4_26B_A4B_IT: LLMSelectionProfile(
        alias=LEGACY_OPENROUTER_SELECTION_ALIAS_MANAGED_GEMMA_4_26B_A4B_IT,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemma4_managed",
        description_key="provider.gemma4_managed.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_MANAGED,
    ),
    LEGACY_OPENROUTER_SELECTION_ALIAS_BYOK_GEMMA_4_26B_A4B_IT: LLMSelectionProfile(
        alias=LEGACY_OPENROUTER_SELECTION_ALIAS_BYOK_GEMMA_4_26B_A4B_IT,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemma4_26b_a4b_it",
        description_key="provider.gemma4_26b_a4b_it.description",
        openrouter_model=OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_BYOK,
    ),
    LEGACY_OPENROUTER_SELECTION_ALIAS_NONE_QWEN_35_FLASH_02_23: LLMSelectionProfile(
        alias=LEGACY_OPENROUTER_SELECTION_ALIAS_NONE_QWEN_35_FLASH_02_23,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.qwen35_flash_openrouter",
        description_key="provider.qwen35_flash_openrouter.description",
        openrouter_model=OPENROUTER_MODEL_QWEN_35_FLASH_02_23,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_NONE,
    ),
    LEGACY_OPENROUTER_SELECTION_ALIAS_BYOK_QWEN_35_FLASH_02_23: LLMSelectionProfile(
        alias=LEGACY_OPENROUTER_SELECTION_ALIAS_BYOK_QWEN_35_FLASH_02_23,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.qwen35_flash_openrouter",
        description_key="provider.qwen35_flash_openrouter.description",
        openrouter_model=OPENROUTER_MODEL_QWEN_35_FLASH_02_23,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_BYOK,
    ),
    LEGACY_OPENROUTER_SELECTION_ALIAS_GEMINI31_FLASH_LITE_BYOK: LLMSelectionProfile(
        alias=LEGACY_OPENROUTER_SELECTION_ALIAS_GEMINI31_FLASH_LITE_BYOK,
        provider=LLM_PROVIDER_OPENROUTER,
        label_key="provider.gemini37_flash_openrouter",
        description_key="provider.gemini37_flash_openrouter.description",
        openrouter_model=OPENROUTER_MODEL_GEMINI_37_FLASH,
        openrouter_source=OPENROUTER_CREDENTIAL_SOURCE_BYOK,
    ),
}

LEGACY_FALLBACK_ALIAS_TO_ALIAS: dict[str, str] = {}

OPENROUTER_MAIN_SELECTION_ALIASES: tuple[str, ...] = (
    OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_MANAGED,
    OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_MANAGED,
    OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_MANAGED,
    OPENROUTER_SELECTION_ALIAS_GEMINI37_FLASH_BYOK,
    OPENROUTER_SELECTION_ALIAS_GEMMA4_BYOK,
    OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_BYOK,
    OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_BYOK,
)

OPENROUTER_FALLBACK_SELECTION_ALIASES: tuple[str, ...] = (
    OPENROUTER_FALLBACK_SELECTION_ALIAS_NONE,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_CHINA,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_26B_31B,
    OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMMA4_31B,
)


def get_openrouter_llm_profile(alias: str) -> LLMSelectionProfile | None:
    profile = PROFILE_BY_ALIAS.get(alias)
    if profile is not None:
        return profile
    return LEGACY_PROFILE_BY_ALIAS.get(alias)


def profile_for_alias(alias: str) -> LLMSelectionProfile:
    profile = get_openrouter_llm_profile(alias)
    if profile is None:
        raise KeyError(alias)
    return profile


def normalize_openrouter_fallback_selection_alias(alias: str | None) -> str | None:
    if alias is None:
        return None
    normalized = alias.strip()
    if not normalized:
        return None
    if normalized in FALLBACK_PROFILE_BY_ALIAS:
        return normalized
    return LEGACY_FALLBACK_ALIAS_TO_ALIAS.get(normalized)


def fallback_profile_for_alias(alias: str) -> OpenRouterFallbackProfile:
    normalized = normalize_openrouter_fallback_selection_alias(alias)
    if normalized is None or normalized not in FALLBACK_PROFILE_BY_ALIAS:
        raise KeyError(alias)
    return FALLBACK_PROFILE_BY_ALIAS[normalized]


def resolve_openrouter_fallback_model(alias: str) -> str | None:
    return fallback_profile_for_alias(alias).openrouter_model


def openrouter_alias_for_fields(
    *,
    model: str,
    source: str,
    models: tuple[str, ...] = (),
) -> str | None:
    if source == OPENROUTER_CREDENTIAL_SOURCE_NONE:
        return None
    if models == (
        OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        OPENROUTER_MODEL_GEMMA_4_31B_IT,
    ):
        if source == OPENROUTER_CREDENTIAL_SOURCE_MANAGED:
            return OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_MANAGED
        return OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK
    if model == OPENROUTER_MODEL_GEMMA_4_31B_IT:
        if source == OPENROUTER_CREDENTIAL_SOURCE_MANAGED:
            return OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_MANAGED
        return OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK
    if model == OPENROUTER_MODEL_QWEN_35_FLASH_02_23:
        if source == OPENROUTER_CREDENTIAL_SOURCE_MANAGED:
            return OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_MANAGED
        return OPENROUTER_SELECTION_ALIAS_QWEN35_FLASH_BYOK
    if model == OPENROUTER_MODEL_DEEPSEEK_V4_FLASH:
        if source == OPENROUTER_CREDENTIAL_SOURCE_MANAGED:
            return OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_MANAGED
        return OPENROUTER_SELECTION_ALIAS_DEEPSEEK_V4_FLASH_BYOK
    if model == OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT:
        if source == OPENROUTER_CREDENTIAL_SOURCE_MANAGED:
            return OPENROUTER_SELECTION_ALIAS_GEMMA4_MANAGED
        return OPENROUTER_SELECTION_ALIAS_GEMMA4_BYOK
    if model == OPENROUTER_MODEL_GEMINI_37_FLASH:
        if source == OPENROUTER_CREDENTIAL_SOURCE_BYOK:
            return OPENROUTER_SELECTION_ALIAS_GEMINI37_FLASH_BYOK
        return None
    return None


def get_openrouter_selection_alias_for_model_and_source(
    llm_model: str,
    selected_source: str,
) -> str | None:
    return openrouter_alias_for_fields(model=llm_model, source=selected_source)
