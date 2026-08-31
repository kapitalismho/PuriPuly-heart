from __future__ import annotations

import puripuly_heart.core.stt.custom_vocab as custom_vocab_module
from puripuly_heart.core.stt.custom_vocab import (
    CustomVocabularyRuntimeConfig,
    get_effective_custom_terms,
)


def _vocab_config(
    *,
    enabled: bool = False,
    terms: dict[str, list[str]] | None = None,
) -> CustomVocabularyRuntimeConfig:
    return CustomVocabularyRuntimeConfig(enabled=enabled, terms=terms or {})


def test_get_effective_custom_terms_returns_empty_for_fresh_settings() -> None:
    config = _vocab_config()

    assert get_effective_custom_terms(config, "ko") == []
    assert get_effective_custom_terms(config, "en") == []
    assert get_effective_custom_terms(config, "zh-CN") == []


def test_get_effective_custom_terms_reads_current_language_bucket_only() -> None:
    config = _vocab_config(
        enabled=True,
        terms={
            "ko": ["Puripuly", "VRChat"],
            "en": ["Soniox", "OSC"],
        },
    )

    assert get_effective_custom_terms(config, "ko") == ["Puripuly", "VRChat"]


def test_get_effective_custom_terms_preserves_first_occurrence_order_when_deduping() -> None:
    config = _vocab_config(
        enabled=True,
        terms={
            "ko": ["VRChat", "Puripuly", "VRChat", "OSC", "Puripuly", "Soniox"],
        },
    )

    assert get_effective_custom_terms(config, "ko") == [
        "VRChat",
        "Puripuly",
        "OSC",
        "Soniox",
    ]


def test_get_effective_custom_terms_trims_whitespace_and_drops_empty_values() -> None:
    config = _vocab_config(
        enabled=True,
        terms={
            "ko": ["  Puripuly  ", "", "   ", "\tVRChat\t", "\nSoniox\n", "OSC"],
        },
    )

    assert get_effective_custom_terms(config, "ko") == [
        "Puripuly",
        "VRChat",
        "Soniox",
        "OSC",
    ]


def test_get_effective_custom_terms_is_stable_and_respects_disabled_flag() -> None:
    terms = {
        "ko": ["  VRChat  ", "Puripuly", "VRChat", "  ", "OSC"],
        "en": ["Ignored"],
    }
    enabled = _vocab_config(enabled=True, terms=terms)

    first = get_effective_custom_terms(enabled, "ko")
    second = get_effective_custom_terms(enabled, "ko")

    assert first == ["VRChat", "Puripuly", "OSC"]
    assert second == first
    assert second is not first
    assert get_effective_custom_terms(_vocab_config(enabled=False, terms=terms), "ko") == []


def test_get_effective_custom_terms_caps_to_100_terms() -> None:
    config = _vocab_config(
        enabled=True,
        terms={"ko": [f"term-{i:03d}" for i in range(120)]},
    )

    effective_terms = get_effective_custom_terms(config, "ko")

    assert len(effective_terms) == 100
    assert effective_terms[0] == "term-000"
    assert effective_terms[-1] == "term-099"


def test_get_effective_local_qwen_hotwords_uses_smaller_cap_and_sanitizes_commas() -> None:
    config = _vocab_config(
        enabled=True,
        terms={
            "ko": [
                " Puripuly ",
                "VRChat, Japan",
                "VRChat   Japan",
                *[f"term-{i:02d}" for i in range(20)],
            ],
        },
    )

    assert hasattr(custom_vocab_module, "LOCAL_QWEN_MAX_HOTWORDS")
    assert hasattr(custom_vocab_module, "get_effective_local_qwen_hotwords")

    hotwords = custom_vocab_module.get_effective_local_qwen_hotwords(config, "ko")

    assert hotwords[:2] == ["Puripuly", "VRChat Japan"]
    assert len(hotwords) == custom_vocab_module.LOCAL_QWEN_MAX_HOTWORDS
    assert hotwords[-1] == f"term-{custom_vocab_module.LOCAL_QWEN_MAX_HOTWORDS - 3:02d}"
