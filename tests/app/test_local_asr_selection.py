from __future__ import annotations

import pytest
from puripuly_heart.app.services.local_asr_selection import resolve_local_asr_selection
from puripuly_heart.core.local_stt_assets import LOCAL_STT_MODEL_ID


@pytest.mark.parametrize(
    ("provider", "language", "effective", "supported", "fallback"),
    [
        ("local_parakeet_ja", "ja-JP", "local_parakeet_ja", True, False),
        ("local_parakeet_v3", "fr-FR", "local_parakeet_v3", True, False),
        ("local_parakeet_v3", "ko-KR", "local_qwen", True, True),
        ("local_parakeet_ja", "en-US", "local_qwen", True, True),
        ("local_qwen", "ca", "local_qwen", False, False),
        ("local_cpu_auto", "ko-KR", "local_cpu_auto", True, False),
        ("local_cpu_auto", "ca", "local_cpu_auto", False, False),
        ("deepgram", "ko-KR", "deepgram", True, False),
    ],
)
def test_resolve_local_asr_selection(
    provider: str,
    language: str,
    effective: str,
    supported: bool,
    fallback: bool,
) -> None:
    decision = resolve_local_asr_selection(provider, language)

    assert decision.effective_provider == effective
    assert decision.supported is supported
    assert decision.fallback_applied is fallback


@pytest.mark.parametrize("language", ["ko-KR", "ja-JP", "en-US"])
def test_cpu_auto_falls_back_to_qwen_when_model_set_is_incomplete(language: str) -> None:
    decision = resolve_local_asr_selection(
        "local_cpu_auto",
        language,
        cpu_auto_available=False,
    )

    assert decision.effective_provider == "local_qwen"
    assert decision.model_id == LOCAL_STT_MODEL_ID
    assert decision.supported is True
    assert decision.fallback_applied is True
