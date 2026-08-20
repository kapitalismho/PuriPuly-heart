from __future__ import annotations

import pytest

from puripuly_heart.core.language import SUPPORTED_LANGUAGES
from puripuly_heart.core.osc.control_codec import (
    OscControlCodecError,
    UnknownOscControlValueError,
    decode_control_message,
    encode_control_message,
)
from puripuly_heart.core.osc.control_schema import (
    ASR_IDS,
    BOOLEAN_CONTROLS,
    FALLBACK_IDS,
    INTEGER_CONTROLS,
    LANGUAGE_IDS,
    OSC_BOOLEAN_PARAMETER_NAMES,
    OSC_INTEGER_PARAMETER_NAMES,
    OSC_PARAMETER_DEFINITIONS,
    TRANSLATION_MODEL_IDS,
)


def test_osc_abi_registries_are_explicit_and_cover_current_languages() -> None:
    assert ASR_IDS == {
        0: "local_cpu_auto",
        1: "local_parakeet_v3",
        2: "local_parakeet_ja",
        3: "local_qwen",
        4: "local_qwen_gpu",
        5: "deepgram",
        6: "qwen_asr",
        7: "soniox",
        8: "custom_offline",
        9: "custom_realtime",
    }
    assert TRANSLATION_MODEL_IDS[0] == "gemma4_26b_31b"
    assert TRANSLATION_MODEL_IDS[9] == "custom_http"
    assert FALLBACK_IDS[0] == "none"
    assert set(LANGUAGE_IDS.values()) == set(SUPPORTED_LANGUAGES)
    assert len(LANGUAGE_IDS) == len(set(LANGUAGE_IDS.values()))


def test_codec_validates_absolute_boolean_and_integer_controls() -> None:
    bool_message = decode_control_message("/avatar/parameters/PuriPuly_Talk", True)
    integer_message = decode_control_message("/avatar/parameters/PuriPuly_SelfASR", 5)

    assert bool_message.name == "PuriPuly_Talk"
    assert bool_message.value is True
    assert integer_message.value == 5
    assert encode_control_message("PuriPuly_Translator", 2) == (
        "/avatar/parameters/PuriPuly_Translator",
        2,
    )


def test_codec_rejects_wrong_types_unknown_ids_and_unknown_parameters() -> None:
    with pytest.raises(OscControlCodecError):
        decode_control_message("/avatar/parameters/PuriPuly_Talk", 1)
    with pytest.raises(UnknownOscControlValueError):
        decode_control_message("/avatar/parameters/PuriPuly_SelfASR", 99)
    with pytest.raises(OscControlCodecError):
        decode_control_message("/avatar/parameters/PuriPuly_Unknown", True)


def test_control_schema_keeps_boolean_and_integer_surfaces_separate() -> None:
    assert set(OSC_BOOLEAN_PARAMETER_NAMES).isdisjoint(OSC_INTEGER_PARAMETER_NAMES)
    assert len(OSC_BOOLEAN_PARAMETER_NAMES) == 7
    assert len(OSC_INTEGER_PARAMETER_NAMES) == 8


def test_osc_public_abi_snapshot_is_append_only_and_exact() -> None:
    assert OSC_BOOLEAN_PARAMETER_NAMES == (
        "PuriPuly_Talk",
        "PuriPuly_Listen",
        "PuriPuly_Trans",
        "PuriPuly_Captions",
        "PuriPuly_PeerAuto",
        "PuriPuly_MuteSync",
        "PuriPuly_ChatboxSource",
    )
    assert OSC_INTEGER_PARAMETER_NAMES == (
        "PuriPuly_SelfSrcLang",
        "PuriPuly_SelfDstLang",
        "PuriPuly_PeerSrcLang",
        "PuriPuly_PeerDstLang",
        "PuriPuly_SelfASR",
        "PuriPuly_PeerASR",
        "PuriPuly_Translator",
        "PuriPuly_Fallback",
    )
    assert dict(BOOLEAN_CONTROLS) == {
        "PuriPuly_Talk": "self_capture",
        "PuriPuly_Listen": "peer_capture",
        "PuriPuly_Trans": "translation",
        "PuriPuly_Captions": "captions",
        "PuriPuly_PeerAuto": "peer_source_mode",
        "PuriPuly_MuteSync": "vrc_mic_intercept",
        "PuriPuly_ChatboxSource": "chatbox_include_source",
    }
    assert dict(INTEGER_CONTROLS) == {
        "PuriPuly_SelfSrcLang": "languages.source_language",
        "PuriPuly_SelfDstLang": "languages.target_language",
        "PuriPuly_PeerSrcLang": "languages.peer_source_language",
        "PuriPuly_PeerDstLang": "languages.peer_target_language",
        "PuriPuly_SelfASR": "stt.provider",
        "PuriPuly_PeerASR": "peer_stt.provider",
        "PuriPuly_Translator": "translation.model",
        "PuriPuly_Fallback": "translation.fallback",
    }
    assert tuple(
        (name, definition.value_type, definition.target)
        for name, definition in OSC_PARAMETER_DEFINITIONS.items()
    ) == (
        ("PuriPuly_Talk", "bool", "self_capture"),
        ("PuriPuly_Listen", "bool", "peer_capture"),
        ("PuriPuly_Trans", "bool", "translation"),
        ("PuriPuly_Captions", "bool", "captions"),
        ("PuriPuly_PeerAuto", "bool", "peer_source_mode"),
        ("PuriPuly_MuteSync", "bool", "vrc_mic_intercept"),
        ("PuriPuly_ChatboxSource", "bool", "chatbox_include_source"),
        ("PuriPuly_SelfSrcLang", "int", "languages.source_language"),
        ("PuriPuly_SelfDstLang", "int", "languages.target_language"),
        ("PuriPuly_PeerSrcLang", "int", "languages.peer_source_language"),
        ("PuriPuly_PeerDstLang", "int", "languages.peer_target_language"),
        ("PuriPuly_SelfASR", "int", "stt.provider"),
        ("PuriPuly_PeerASR", "int", "peer_stt.provider"),
        ("PuriPuly_Translator", "int", "translation.model"),
        ("PuriPuly_Fallback", "int", "translation.fallback"),
    )
    assert dict(ASR_IDS) == {
        0: "local_cpu_auto",
        1: "local_parakeet_v3",
        2: "local_parakeet_ja",
        3: "local_qwen",
        4: "local_qwen_gpu",
        5: "deepgram",
        6: "qwen_asr",
        7: "soniox",
        8: "custom_offline",
        9: "custom_realtime",
    }
    assert dict(TRANSLATION_MODEL_IDS) == {
        0: "gemma4_26b_31b",
        1: "gemma4_31b",
        2: "gemma4",
        3: "deepseek_v4_flash",
        5: "gemini37_flash",
        6: "gemini31_flash_lite",
        7: "qwen35_plus",
        8: "local_llm",
        9: "custom_http",
    }
    assert dict(FALLBACK_IDS) == {
        0: "none",
        1: "deepseek_v4_flash_official",
        2: "openrouter_deepseek_v4_flash",
        3: "openrouter_gemma4_26b_a4b",
        4: "openrouter_gemma4_26b_31b",
        5: "openrouter_gemma4_31b",
        6: "managed_gemma4_26b_31b",
        7: "managed_gemma4_31b",
        8: "cerebras_gemma4_31b",
    }
    assert dict(LANGUAGE_IDS) == {
        0: "ar",
        1: "bg",
        2: "ca",
        3: "cs",
        4: "da",
        5: "de",
        6: "el",
        7: "en",
        8: "es",
        9: "et",
        10: "fi",
        11: "fr",
        12: "hi",
        13: "hu",
        14: "id",
        15: "it",
        16: "ja",
        17: "ko",
        18: "lt",
        19: "lv",
        20: "ms",
        21: "nl",
        22: "no",
        23: "pl",
        24: "pt",
        25: "ro",
        26: "ru",
        27: "sk",
        28: "sv",
        29: "th",
        30: "tr",
        31: "uk",
        32: "vi",
        33: "zh-CN",
        34: "zh-TW",
    }
