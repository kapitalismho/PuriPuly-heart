from __future__ import annotations

import pytest

from puripuly_heart.app.services.local_asr.local_asr_selection import (
    required_local_asr_model_ids,
    resolve_local_asr_selection,
)
from puripuly_heart.core.local_asr.local_stt_assets import (
    FUN_ASR_MLT_NANO_MODEL_ID,
    FUN_ASR_NANO_MODEL_ID,
    ORUKEET_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)


@pytest.mark.parametrize(
    ("language", "model_id"),
    [
        ("en-US", ORUKEET_MODEL_ID),
        ("zh_CN", FUN_ASR_NANO_MODEL_ID),
        ("ko-KR", FUN_ASR_MLT_NANO_MODEL_ID),
        ("ja-JP", PARAKEET_JAPANESE_MODEL_ID),
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
        ("local_orukeet", "en", ORUKEET_MODEL_ID, True),
        ("local_orukeet", "fr", ORUKEET_MODEL_ID, False),
        ("local_orukeet", "ko", ORUKEET_MODEL_ID, False),
        ("local_fun_asr_nano", "zh-TW", FUN_ASR_NANO_MODEL_ID, True),
        ("local_fun_asr_nano", "en", FUN_ASR_NANO_MODEL_ID, False),
        ("local_fun_asr_mlt_nano", "ko", FUN_ASR_MLT_NANO_MODEL_ID, True),
        ("local_fun_asr_mlt_nano", "zh", FUN_ASR_MLT_NANO_MODEL_ID, False),
        ("local_parakeet_ja", "ja", PARAKEET_JAPANESE_MODEL_ID, True),
        ("local_parakeet_ja", "en", PARAKEET_JAPANESE_MODEL_ID, False),
        ("local_cpu_auto", "fr", None, False),
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
