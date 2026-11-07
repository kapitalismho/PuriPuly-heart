from __future__ import annotations

import pytest

from puripuly_heart.app.services.local_asr.local_asr_selection import (
    required_local_asr_model_ids,
    resolve_local_asr_selection,
)
from puripuly_heart.core.local_asr.local_stt_assets import (
    FUN_ASR_MLT_NANO_MODEL_ID,
    FUN_ASR_NANO_MODEL_ID,
    GIGAAM_MODEL_ID,
    GRANITE_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)


@pytest.mark.parametrize(
    ("language", "model_id"),
    [
        ("en-US", GRANITE_MODEL_ID),
        ("zh_CN", FUN_ASR_NANO_MODEL_ID),
        ("ko-KR", FUN_ASR_MLT_NANO_MODEL_ID),
        ("ja-JP", PARAKEET_JAPANESE_MODEL_ID),
        ("fr-FR", GRANITE_MODEL_ID),
        ("vi-VN", FUN_ASR_MLT_NANO_MODEL_ID),
        ("pt-BR", GRANITE_MODEL_ID),
        ("ru-RU", GIGAAM_MODEL_ID),
    ],
)
def test_cpu_auto_routes_only_the_selected_language_model(language: str, model_id: str) -> None:
    decision = resolve_local_asr_selection("local_cpu_auto", language)

    assert decision.requested_provider == decision.effective_provider == "local_cpu_auto"
    assert decision.source_language == language
    assert decision.model_id == model_id
    assert decision.supported
    assert required_local_asr_model_ids("local_cpu_auto") == REQUIRED_CPU_LOCAL_STT_MODEL_IDS


@pytest.mark.parametrize(
    ("provider", "language", "model_id", "supported"),
    [
        ("local_granite", "en", GRANITE_MODEL_ID, True),
        ("local_granite", "fr", GRANITE_MODEL_ID, True),
        ("local_granite", "de", GRANITE_MODEL_ID, True),
        ("local_granite", "es", GRANITE_MODEL_ID, True),
        ("local_granite", "pt", GRANITE_MODEL_ID, True),
        ("local_granite", "it", GRANITE_MODEL_ID, True),
        ("local_granite", "ja", GRANITE_MODEL_ID, True),
        ("local_granite", "ru", GRANITE_MODEL_ID, False),
        ("local_granite", "ko", GRANITE_MODEL_ID, False),
        ("local_granite", "uk", GRANITE_MODEL_ID, False),
        ("local_gigaam", "ru-RU", GIGAAM_MODEL_ID, True),
        ("local_gigaam", "uk", GIGAAM_MODEL_ID, False),
        ("local_gigaam", "en", GIGAAM_MODEL_ID, False),
        ("local_fun_asr_nano", "zh-TW", FUN_ASR_NANO_MODEL_ID, True),
        ("local_fun_asr_nano", "en", FUN_ASR_NANO_MODEL_ID, False),
        ("local_fun_asr_mlt_nano", "ko", FUN_ASR_MLT_NANO_MODEL_ID, True),
        ("local_fun_asr_mlt_nano", "zh", FUN_ASR_MLT_NANO_MODEL_ID, True),
        ("local_fun_asr_mlt_nano", "en", FUN_ASR_MLT_NANO_MODEL_ID, True),
        ("local_fun_asr_mlt_nano", "ja", FUN_ASR_MLT_NANO_MODEL_ID, True),
        ("local_fun_asr_mlt_nano", "vi", FUN_ASR_MLT_NANO_MODEL_ID, True),
        ("local_fun_asr_mlt_nano", "fr", FUN_ASR_MLT_NANO_MODEL_ID, False),
        ("local_fun_asr_mlt_nano", "ru", FUN_ASR_MLT_NANO_MODEL_ID, False),
        ("local_fun_asr_mlt_nano", "uk", FUN_ASR_MLT_NANO_MODEL_ID, False),
        ("local_parakeet_ja", "ja", PARAKEET_JAPANESE_MODEL_ID, True),
        ("local_parakeet_ja", "en", PARAKEET_JAPANESE_MODEL_ID, False),
        ("local_cpu_auto", "fr", GRANITE_MODEL_ID, True),
        ("local_cpu_auto", "ca", None, False),
        ("local_cpu_auto", "uk", None, False),
        ("deepgram", "fr", None, True),
    ],
)
def test_selection_preserves_intent_without_cpu_cloud_or_gpu_fallback(
    provider: str, language: str, model_id: str | None, supported: bool
) -> None:
    decision = resolve_local_asr_selection(provider, language)

    assert decision.requested_provider == decision.effective_provider == provider
    assert decision.source_language == language
    assert decision.model_id == model_id
    assert decision.supported is supported


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [
        ("local_granite", GRANITE_MODEL_ID),
        ("local_gigaam", GIGAAM_MODEL_ID),
        ("local_fun_asr_nano", FUN_ASR_NANO_MODEL_ID),
        ("local_fun_asr_mlt_nano", FUN_ASR_MLT_NANO_MODEL_ID),
        ("local_parakeet_ja", PARAKEET_JAPANESE_MODEL_ID),
    ],
)
def test_direct_cpu_selection_requires_only_its_selected_asset(
    provider: str, model_id: str
) -> None:
    assert required_local_asr_model_ids(provider) == (model_id,)
