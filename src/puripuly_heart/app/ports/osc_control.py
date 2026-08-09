from __future__ import annotations

from typing import Final, Literal, Protocol

from puripuly_heart.core.osc.control_codec import (
    OscControlCodecError,
    OscControlMessage,
    UnknownOscControlValueError,
    decode_control_address,
    decode_control_message,
    encode_control_message,
    validate_control_value,
)
from puripuly_heart.core.osc.control_schema import (
    ASR_ID_BY_PROVIDER,
    ASR_IDS,
    BOOLEAN_CONTROLS,
    FALLBACK_ID_BY_ALIAS,
    FALLBACK_IDS,
    INTEGER_CONTROLS,
    LANGUAGE_ID_BY_CODE,
    LANGUAGE_IDS,
    OSC_AVATAR_CHANGE_ADDRESS,
    OSC_BOOLEAN_PARAMETER_NAMES,
    OSC_INTEGER_PARAMETER_NAMES,
    OSC_INTEGER_REGISTRIES,
    OSC_MUTE_SELF_ADDRESS,
    OSC_PARAMETER_ADDRESS_PREFIX,
    OSC_PARAMETER_DEFINITIONS,
    OSC_PARAMETER_PREFIX,
    TRANSLATION_MODEL_ID_BY_VALUE,
    TRANSLATION_MODEL_IDS,
    OscParameterDefinition,
    OscParameterType,
    is_puripuly_parameter_address,
    parameter_definition,
    parameter_definition_for_address,
    registry_for_parameter,
)

OscConnectionMode = Literal["automatic", "manual", "off"]
OSC_CONNECTION_MODES: Final[tuple[OscConnectionMode, ...]] = ("automatic", "manual", "off")


class OscSenderPort(Protocol):
    def send_message(self, address: str, *values: object) -> None: ...

    def send_chatbox(self, text: str) -> None: ...

    def send_typing(self, is_typing: bool) -> None: ...

    def set_destination(self, host: str, port: int) -> None: ...


class OscControlApplicationPort(Protocol):
    async def set_self_capture(self, enabled: bool) -> object: ...

    async def set_peer_capture(self, enabled: bool) -> object: ...

    async def set_translation(self, enabled: bool) -> object: ...

    async def set_captions(self, enabled: bool) -> object: ...

    async def set_languages(
        self,
        *,
        self_source: str,
        self_target: str,
        peer_source: str,
        peer_target: str,
    ) -> object: ...

    async def set_peer_auto_detect(self, enabled: bool) -> object: ...

    async def set_self_asr(self, provider: str) -> object: ...

    async def set_peer_asr(self, provider: str) -> object: ...

    async def set_translation_model(self, model: str) -> object: ...

    async def set_fallback(self, alias: str) -> object: ...

    async def set_mute_sync(self, enabled: bool) -> object: ...

    async def set_chatbox_source(self, enabled: bool) -> object: ...


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
    "OSC_CONNECTION_MODES",
    "OSC_INTEGER_PARAMETER_NAMES",
    "OSC_INTEGER_REGISTRIES",
    "OSC_MUTE_SELF_ADDRESS",
    "OSC_PARAMETER_ADDRESS_PREFIX",
    "OSC_PARAMETER_DEFINITIONS",
    "OSC_PARAMETER_PREFIX",
    "OscConnectionMode",
    "OscControlApplicationPort",
    "OscControlCodecError",
    "OscControlMessage",
    "OscParameterDefinition",
    "OscParameterType",
    "OscSenderPort",
    "TRANSLATION_MODEL_IDS",
    "TRANSLATION_MODEL_ID_BY_VALUE",
    "UnknownOscControlValueError",
    "decode_control_address",
    "decode_control_message",
    "encode_control_message",
    "is_puripuly_parameter_address",
    "parameter_definition",
    "parameter_definition_for_address",
    "registry_for_parameter",
    "validate_control_value",
]
