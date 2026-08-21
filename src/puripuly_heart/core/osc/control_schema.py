from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from puripuly_heart.core.language import SUPPORTED_LANGUAGES

OscParameterType = Literal["bool", "int"]

OSC_PARAMETER_PREFIX: Final = "PuriPuly_"
OSC_PARAMETER_ADDRESS_PREFIX: Final = "/avatar/parameters/"
OSC_AVATAR_CHANGE_ADDRESS: Final = "/avatar/change"
OSC_MUTE_SELF_ADDRESS: Final = f"{OSC_PARAMETER_ADDRESS_PREFIX}MuteSelf"


@dataclass(frozen=True, slots=True)
class OscParameterDefinition:
    name: str
    value_type: OscParameterType
    target: str

    @property
    def address(self) -> str:
        return f"{OSC_PARAMETER_ADDRESS_PREFIX}{self.name}"


BOOLEAN_CONTROLS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "PuriPuly_Talk": "self_capture",
        "PuriPuly_Listen": "peer_capture",
        "PuriPuly_Trans": "translation",
        "PuriPuly_Captions": "captions",
        "PuriPuly_PeerAuto": "peer_source_mode",
        "PuriPuly_MuteSync": "vrc_mic_intercept",
        "PuriPuly_ChatboxSource": "chatbox_include_source",
    }
)

INTEGER_CONTROLS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "PuriPuly_SelfSrcLang": "languages.source_language",
        "PuriPuly_SelfDstLang": "languages.target_language",
        "PuriPuly_PeerSrcLang": "languages.peer_source_language",
        "PuriPuly_PeerDstLang": "languages.peer_target_language",
        "PuriPuly_SelfASR": "stt.provider",
        "PuriPuly_PeerASR": "peer_stt.provider",
        "PuriPuly_Translator": "translation.model",
        "PuriPuly_Fallback": "translation.fallback",
    }
)

OSC_PARAMETER_DEFINITIONS: Final[Mapping[str, OscParameterDefinition]] = MappingProxyType(
    {
        **{
            name: OscParameterDefinition(name, "bool", target)
            for name, target in BOOLEAN_CONTROLS.items()
        },
        **{
            name: OscParameterDefinition(name, "int", target)
            for name, target in INTEGER_CONTROLS.items()
        },
    }
)

OSC_BOOLEAN_PARAMETER_NAMES: Final = tuple(BOOLEAN_CONTROLS)
OSC_INTEGER_PARAMETER_NAMES: Final = tuple(INTEGER_CONTROLS)

ASR_IDS: Final[Mapping[int, str]] = MappingProxyType(
    {
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
)

TRANSLATION_MODEL_IDS: Final[Mapping[int, str]] = MappingProxyType(
    {
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
)

FALLBACK_IDS: Final[Mapping[int, str]] = MappingProxyType(
    {
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
)

_LANGUAGE_ABI_ENTRIES: Final[tuple[tuple[int, str], ...]] = (
    (0, "ar"),
    (1, "bg"),
    (2, "ca"),
    (3, "cs"),
    (4, "da"),
    (5, "de"),
    (6, "el"),
    (7, "en"),
    (8, "es"),
    (9, "et"),
    (10, "fi"),
    (11, "fr"),
    (12, "hi"),
    (13, "hu"),
    (14, "id"),
    (15, "it"),
    (16, "ja"),
    (17, "ko"),
    (18, "lt"),
    (19, "lv"),
    (20, "ms"),
    (21, "nl"),
    (22, "no"),
    (23, "pl"),
    (24, "pt"),
    (25, "ro"),
    (26, "ru"),
    (27, "sk"),
    (28, "sv"),
    (29, "th"),
    (30, "tr"),
    (31, "uk"),
    (32, "vi"),
    (33, "zh-CN"),
    (34, "zh-TW"),
)

if {code for _identifier, code in _LANGUAGE_ABI_ENTRIES} != set(SUPPORTED_LANGUAGES):
    raise RuntimeError("OSC language ABI must cover SUPPORTED_LANGUAGES exactly")

LANGUAGE_IDS: Final[Mapping[int, str]] = MappingProxyType(dict(_LANGUAGE_ABI_ENTRIES))
ASR_ID_BY_PROVIDER: Final[Mapping[str, int]] = MappingProxyType(
    {
        **{value: identifier for identifier, value in ASR_IDS.items()},
        "custom": 8,
    }
)
TRANSLATION_MODEL_ID_BY_VALUE: Final[Mapping[str, int]] = MappingProxyType(
    {
        **{value: identifier for identifier, value in TRANSLATION_MODEL_IDS.items()},
        "gemma4_31b_cerebras": 1,
    }
)
FALLBACK_ID_BY_ALIAS: Final[Mapping[str, int]] = MappingProxyType(
    {value: identifier for identifier, value in FALLBACK_IDS.items()}
)
LANGUAGE_ID_BY_CODE: Final[Mapping[str, int]] = MappingProxyType(
    {value: identifier for identifier, value in LANGUAGE_IDS.items()}
)
OSC_INTEGER_REGISTRIES: Final[Mapping[str, Mapping[int, str]]] = MappingProxyType(
    {
        "PuriPuly_SelfSrcLang": LANGUAGE_IDS,
        "PuriPuly_SelfDstLang": LANGUAGE_IDS,
        "PuriPuly_PeerSrcLang": LANGUAGE_IDS,
        "PuriPuly_PeerDstLang": LANGUAGE_IDS,
        "PuriPuly_SelfASR": ASR_IDS,
        "PuriPuly_PeerASR": ASR_IDS,
        "PuriPuly_Translator": TRANSLATION_MODEL_IDS,
        "PuriPuly_Fallback": FALLBACK_IDS,
    }
)


def parameter_definition(name: str) -> OscParameterDefinition:
    try:
        return OSC_PARAMETER_DEFINITIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown PuriPuly OSC parameter: {name}") from exc


def parameter_definition_for_address(address: str) -> OscParameterDefinition:
    if not isinstance(address, str) or not address.startswith(OSC_PARAMETER_ADDRESS_PREFIX):
        raise ValueError(f"unsupported OSC control address: {address!r}")
    return parameter_definition(address[len(OSC_PARAMETER_ADDRESS_PREFIX) :])


def registry_for_parameter(name: str) -> Mapping[int, str] | None:
    return OSC_INTEGER_REGISTRIES.get(name)


def is_puripuly_parameter_address(address: str) -> bool:
    return (
        isinstance(address, str)
        and address.startswith(f"{OSC_PARAMETER_ADDRESS_PREFIX}{OSC_PARAMETER_PREFIX}")
        and address[len(OSC_PARAMETER_ADDRESS_PREFIX) :] in OSC_PARAMETER_DEFINITIONS
    )


__all__ = [
    "ASR_IDS",
    "ASR_ID_BY_PROVIDER",
    "BOOLEAN_CONTROLS",
    "FALLBACK_IDS",
    "FALLBACK_ID_BY_ALIAS",
    "INTEGER_CONTROLS",
    "LANGUAGE_IDS",
    "LANGUAGE_ID_BY_CODE",
    "OSC_AVATAR_CHANGE_ADDRESS",
    "OSC_BOOLEAN_PARAMETER_NAMES",
    "OSC_INTEGER_PARAMETER_NAMES",
    "OSC_INTEGER_REGISTRIES",
    "OSC_MUTE_SELF_ADDRESS",
    "OSC_PARAMETER_ADDRESS_PREFIX",
    "OSC_PARAMETER_DEFINITIONS",
    "OSC_PARAMETER_PREFIX",
    "OscParameterDefinition",
    "OscParameterType",
    "TRANSLATION_MODEL_IDS",
    "TRANSLATION_MODEL_ID_BY_VALUE",
    "is_puripuly_parameter_address",
    "parameter_definition",
    "parameter_definition_for_address",
    "registry_for_parameter",
]
