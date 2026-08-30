from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

from puripuly_heart.app.ports.osc_control import (
    ASR_ID_BY_PROVIDER,
    BOOLEAN_CONTROLS,
    FALLBACK_ID_BY_ALIAS,
    LANGUAGE_ID_BY_CODE,
    OSC_PARAMETER_ADDRESS_PREFIX,
    SECONDARY_LANGUAGE_ID_BY_CODE,
    OscControlCodecError,
    OscSenderPort,
    translation_model_id_for_selection,
    validate_control_value,
)


@dataclass(frozen=True, slots=True)
class OscCanonicalState:
    self_capture: bool = False
    peer_capture: bool = False
    translation: bool = False
    captions: bool = False
    peer_source_auto: bool = False
    mute_sync: bool = False
    chatbox_source: bool = False
    self_source_language: str = "ko"
    self_target_language: str = "en"
    self_secondary_target_language: str = ""
    peer_source_language: str = "en"
    peer_target_language: str = "ko"
    self_asr: str = "local_cpu_auto"
    peer_asr: str = "local_cpu_auto"
    translation_model: str = "gemma4_26b_31b"
    translation_connection: str = "managed"
    fallback: str = "none"


@dataclass(frozen=True, slots=True)
class OscPublishedValue:
    parameter: str
    value: bool | int

    @property
    def address(self) -> str:
        return f"{OSC_PARAMETER_ADDRESS_PREFIX}{self.parameter}"


class OscStatePublisher:
    def __init__(
        self,
        sender: OscSenderPort,
        *,
        state_provider: Callable[[], OscCanonicalState] | None = None,
    ) -> None:
        self._sender = sender
        self._state_provider = state_provider
        self._last_published: dict[str, bool | int] = {}
        self._started = False

    @property
    def last_published(self) -> Mapping[str, bool | int]:
        return dict(self._last_published)

    def is_echo(self, parameter: str, value: bool | int) -> bool:
        published = self._last_published.get(parameter)
        return published is not None and self.values_equal(parameter, value, published)

    @classmethod
    def value_for_state(
        cls,
        state: OscCanonicalState,
        parameter: str,
    ) -> bool | int:
        return cls._values_for_state(state)[parameter]

    @staticmethod
    def values_equal(
        parameter: str,
        first: object,
        second: object,
    ) -> bool:
        try:
            return validate_control_value(parameter, first) == validate_control_value(
                parameter,
                second,
            )
        except OscControlCodecError:
            return False

    def start(self, state: OscCanonicalState | None = None) -> tuple[OscPublishedValue, ...]:
        self._started = True
        return self.publish_full(state or self._require_state())

    def publish_full(self, state: OscCanonicalState) -> tuple[OscPublishedValue, ...]:
        values = self._values_for_state(state)
        published = tuple(OscPublishedValue(name, value) for name, value in values.items())
        for item in published:
            self._send(item)
        return published

    def publish_delta(self, state: OscCanonicalState) -> tuple[OscPublishedValue, ...]:
        values = self._values_for_state(state)
        changed = tuple(
            OscPublishedValue(name, value)
            for name, value in values.items()
            if self._last_published.get(name) != value
        )
        for item in changed:
            self._send(item)
        return changed

    def publish_state(
        self,
        state: OscCanonicalState,
        *,
        full: bool = False,
    ) -> tuple[OscPublishedValue, ...]:
        return self.publish_full(state) if full or not self._started else self.publish_delta(state)

    def on_avatar_change(self, state: OscCanonicalState) -> tuple[OscPublishedValue, ...]:
        return self.publish_full(state)

    def on_discovery(self, state: OscCanonicalState) -> tuple[OscPublishedValue, ...]:
        return self.publish_full(state)

    def close(self) -> None:
        self._started = False

    def _send(self, item: OscPublishedValue) -> None:
        self._sender.send_message(item.address, item.value)
        self._last_published[item.parameter] = item.value

    def _require_state(self) -> OscCanonicalState:
        if self._state_provider is None:
            raise RuntimeError("OSC state publisher requires a state")
        return self._state_provider()

    @staticmethod
    def _values_for_state(state: OscCanonicalState) -> dict[str, bool | int]:
        fields_by_name = {field.name: getattr(state, field.name) for field in fields(state)}
        values: dict[str, bool | int] = {}
        for parameter, target in BOOLEAN_CONTROLS.items():
            field_name = (
                target.replace("vrc_mic_intercept", "mute_sync")
                .replace("chatbox_include_source", "chatbox_source")
                .replace("peer_source_mode", "peer_source_auto")
            )
            values[parameter] = bool(fields_by_name[field_name])
        values.update(
            {
                "PuriPuly_SelfSrcLang": LANGUAGE_ID_BY_CODE[fields_by_name["self_source_language"]],
                "PuriPuly_SelfDstLang": LANGUAGE_ID_BY_CODE[fields_by_name["self_target_language"]],
                "PuriPuly_SelfDstLang2": SECONDARY_LANGUAGE_ID_BY_CODE[
                    fields_by_name["self_secondary_target_language"]
                ],
                "PuriPuly_PeerSrcLang": LANGUAGE_ID_BY_CODE[fields_by_name["peer_source_language"]],
                "PuriPuly_PeerDstLang": LANGUAGE_ID_BY_CODE[fields_by_name["peer_target_language"]],
                "PuriPuly_SelfASR": ASR_ID_BY_PROVIDER[fields_by_name["self_asr"]],
                "PuriPuly_PeerASR": ASR_ID_BY_PROVIDER[fields_by_name["peer_asr"]],
                "PuriPuly_Translator": translation_model_id_for_selection(
                    fields_by_name["translation_model"],
                    fields_by_name["translation_connection"],
                ),
                "PuriPuly_Fallback": FALLBACK_ID_BY_ALIAS[fields_by_name["fallback"]],
            }
        )
        return values


def state_from_settings(
    settings: Any,
    *,
    self_capture: bool = False,
    peer_capture: bool = False,
    translation: bool = False,
    captions: bool = False,
) -> OscCanonicalState:
    intent = settings.intent
    languages = intent.languages
    translation_intent = intent.translation
    return OscCanonicalState(
        self_capture=self_capture,
        peer_capture=peer_capture,
        translation=translation,
        captions=captions,
        peer_source_auto=languages.peer_source_mode == "auto",
        mute_sync=bool(intent.osc.vrc_mic_intercept),
        chatbox_source=bool(intent.osc.chatbox_include_source),
        self_source_language=languages.source_language,
        self_target_language=languages.target_language,
        self_secondary_target_language=languages.secondary_target_language,
        peer_source_language=languages.peer_source_language,
        peer_target_language=languages.peer_target_language,
        self_asr=_osc_asr_provider(intent.stt.provider, intent.stt.custom.mode),
        peer_asr=_osc_asr_provider(intent.peer_stt.provider, intent.stt.custom.mode),
        translation_model=translation_intent.model,
        translation_connection=translation_intent.connection,
        fallback=fallback_alias_from_settings(settings),
    )


def _osc_asr_provider(provider: object, custom_mode: object) -> str:
    value = str(getattr(provider, "value", provider))
    if value != "custom":
        return value
    if str(custom_mode) == "realtime":
        return "custom_realtime"
    return "custom_offline"


def fallback_alias_from_settings(settings: Any) -> str:
    fallback = settings.intent.translation.fallback
    if not fallback.enabled:
        return "none"
    model = fallback.model
    connection = fallback.connection
    aliases = {
        ("deepseek_v4_flash", "official_byok"): "deepseek_v4_flash_official",
        ("deepseek_v4_flash", "openrouter"): "openrouter_deepseek_v4_flash",
        ("gemma4", "openrouter"): "openrouter_gemma4_26b_a4b",
        ("gemma4_26b_31b", "openrouter"): "openrouter_gemma4_26b_31b",
        ("gemma4_31b", "openrouter"): "openrouter_gemma4_31b",
        ("gemma4_26b_31b", "managed"): "managed_gemma4_26b_31b",
        ("gemma4_31b", "managed"): "managed_gemma4_31b",
        ("gemma4_31b", "cerebras"): "cerebras_gemma4_31b",
        ("gemma4_31b_cerebras", "official_byok"): "cerebras_gemma4_31b",
    }
    return aliases.get((str(model), str(connection)), "none")


__all__ = [
    "OscCanonicalState",
    "OscPublishedValue",
    "OscStatePublisher",
    "fallback_alias_from_settings",
    "state_from_settings",
]
