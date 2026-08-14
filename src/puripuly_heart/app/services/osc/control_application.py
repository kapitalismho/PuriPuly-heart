from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.osc_control import (
    FALLBACK_IDS,
    OscControlApplicationPort,
)

from .state_publisher import fallback_alias_from_settings

SettingsProvider = Callable[[], object | None]
SettingsApply = Callable[[object], Awaitable[object]]
ApplicationCall = Callable[..., Awaitable[object]]
TranslationModelNormalizer = Callable[[object], object]


@dataclass(slots=True)
class SettingsBackedOscControlApplication(OscControlApplicationPort):
    settings_provider: SettingsProvider
    apply_settings: SettingsApply
    translation_model_normalizer: TranslationModelNormalizer
    application: object | None = None
    set_self_capture_command: ApplicationCall | None = None
    set_peer_capture_command: ApplicationCall | None = None
    set_translation_command: ApplicationCall | None = None
    set_captions_command: ApplicationCall | None = None

    async def set_self_capture(self, enabled: bool) -> object:
        return await self._call_runtime(
            self.set_self_capture_command, bool(enabled), "set_stt_enabled"
        )

    async def set_peer_capture(self, enabled: bool) -> object:
        return await self._call_runtime(
            self.set_peer_capture_command,
            bool(enabled),
            "set_peer_translation_enabled",
        )

    async def set_translation(self, enabled: bool) -> object:
        return await self._call_runtime(
            self.set_translation_command,
            bool(enabled),
            "set_translation_enabled",
        )

    async def set_captions(self, enabled: bool) -> object:
        return await self._call_runtime(
            self.set_captions_command,
            bool(enabled),
            "set_overlay_enabled",
        )

    async def set_languages(
        self,
        *,
        self_source: str,
        self_target: str,
        peer_source: str,
        peer_target: str,
    ) -> object:
        current = self.settings_provider()
        if current is not None and _language_values_match(
            current,
            self_source=self_source,
            self_target=self_target,
            peer_source=peer_source,
            peer_target=peer_target,
        ):
            return True
        return await self._apply_settings(
            lambda settings: self._set_languages(
                settings,
                self_source=self_source,
                self_target=self_target,
                peer_source=peer_source,
                peer_target=peer_target,
            )
        )

    async def set_peer_auto_detect(self, enabled: bool) -> object:
        current = self.settings_provider()
        if current is not None and current.languages.peer_source_mode == (
            "auto" if enabled else "manual"
        ):
            return True
        return await self._apply_settings(
            lambda settings: setattr(
                settings.languages, "peer_source_mode", "auto" if enabled else "manual"
            )
        )

    async def set_self_asr(self, provider: str) -> object:
        current = self.settings_provider()
        if current is not None and _enum_value_for_compare(current.provider.stt) == provider:
            return True
        return await self._apply_settings(
            lambda settings: setattr(
                settings.provider,
                "stt",
                _enum_value(settings.provider.stt, provider),
            )
        )

    async def set_peer_asr(self, provider: str) -> object:
        current = self.settings_provider()
        if current is not None and _enum_value_for_compare(current.provider.peer_stt) == provider:
            return True
        return await self._apply_settings(
            lambda settings: setattr(
                settings.provider,
                "peer_stt",
                _enum_value(settings.provider.peer_stt, provider),
            )
        )

    async def set_translation_model(self, model: str) -> object:
        current = self.settings_provider()
        if current is not None and _osc_translation_model_value(current.translation.model) == model:
            return True
        return await self._apply_settings(
            lambda settings: self._set_translation_model(settings, model)
        )

    async def set_fallback(self, alias: str) -> object:
        current = self.settings_provider()
        if current is not None and fallback_alias_from_settings(current) == alias:
            return True
        return await self._apply_settings(lambda settings: self._set_fallback(settings, alias))

    async def set_mute_sync(self, enabled: bool) -> object:
        current = self.settings_provider()
        if current is not None and bool(current.osc.vrc_mic_intercept) is bool(enabled):
            return True
        return await self._apply_settings(
            lambda settings: setattr(settings.osc, "vrc_mic_intercept", bool(enabled))
        )

    async def set_chatbox_source(self, enabled: bool) -> object:
        current = self.settings_provider()
        if current is not None and bool(current.osc.chatbox_include_source) is bool(enabled):
            return True
        return await self._apply_settings(
            lambda settings: setattr(settings.osc, "chatbox_include_source", bool(enabled))
        )

    async def _call_runtime(
        self,
        command: ApplicationCall | None,
        value: bool,
        fallback_name: str,
    ) -> object:
        if command is not None:
            return await command(value)
        application = getattr(self, "application", None)
        if application is not None:
            fallback = getattr(application, fallback_name, None)
            if callable(fallback):
                return await fallback(value)
        raise RuntimeError(f"OSC control application command is not wired: {fallback_name}")

    async def _apply_settings(self, mutator: Callable[[object], object]) -> object:
        current = self.settings_provider()
        if current is None:
            raise RuntimeError("OSC control settings are unavailable")
        updated = copy.deepcopy(current)
        mutator(updated)
        result = await self.apply_settings(updated)
        if not _settings_control_values_match(self.settings_provider(), updated):
            return False
        return result

    @staticmethod
    def _set_languages(
        settings: object,
        *,
        self_source: str,
        self_target: str,
        peer_source: str,
        peer_target: str,
    ) -> None:
        settings.languages.source_language = self_source
        settings.languages.target_language = self_target
        settings.languages.peer_source_language = peer_source
        settings.languages.peer_target_language = peer_target

    def _set_translation_model(self, settings: object, model: str) -> None:
        current_model = settings.translation.model
        current_value = getattr(current_model, "value", current_model)
        if model == "custom_http" and current_value != "custom_http":
            settings.translation.previous_llm_model = current_model
        settings.translation.model = _enum_value(settings.translation.model, model)
        self.translation_model_normalizer(settings)

    @staticmethod
    def _set_fallback(settings: object, alias: str) -> None:
        if alias not in FALLBACK_IDS.values():
            raise ValueError(f"unknown OSC fallback alias: {alias}")
        if alias == "none":
            settings.translation.fallback.enabled = False
            return
        specs: dict[str, tuple[str, str]] = {
            "deepseek_v4_flash_official": (
                "deepseek_v4_flash",
                "official_byok",
            ),
            "openrouter_deepseek_v4_flash": (
                "deepseek_v4_flash",
                "openrouter",
            ),
            "openrouter_gemma4_26b_a4b": (
                "gemma4",
                "openrouter",
            ),
            "openrouter_gemma4_26b_31b": (
                "gemma4_26b_31b",
                "openrouter",
            ),
            "openrouter_gemma4_31b": (
                "gemma4_31b",
                "openrouter",
            ),
            "managed_gemma4_26b_31b": (
                "gemma4_26b_31b",
                "managed",
            ),
            "managed_gemma4_31b": (
                "gemma4_31b",
                "managed",
            ),
            "cerebras_gemma4_31b": (
                "gemma4_31b",
                "cerebras",
            ),
        }
        model, connection = specs[alias]
        settings.translation.fallback.enabled = True
        settings.translation.fallback.model = _enum_value(
            settings.translation.fallback.model,
            model,
        )
        settings.translation.fallback.connection = _enum_value(
            settings.translation.fallback.connection,
            connection,
        )


def _enum_value(current: object, value: str) -> object:
    enum_type = type(current)
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        return value


def _language_values_match(
    settings: object,
    *,
    self_source: str,
    self_target: str,
    peer_source: str,
    peer_target: str,
) -> bool:
    languages = settings.languages
    return (
        languages.source_language == self_source
        and languages.target_language == self_target
        and languages.peer_source_language == peer_source
        and languages.peer_target_language == peer_target
    )


def _settings_control_values_match(actual: object | None, expected: object) -> bool:
    if actual is None:
        return False
    fields = (
        ("languages", "source_language"),
        ("languages", "target_language"),
        ("languages", "peer_source_language"),
        ("languages", "peer_target_language"),
        ("languages", "peer_source_mode"),
        ("provider", "stt"),
        ("provider", "peer_stt"),
        ("translation", "model"),
        ("translation", "connection"),
        ("translation", "http_extension_id"),
        ("translation", "previous_llm_model"),
        ("translation.fallback", "enabled"),
        ("translation.fallback", "model"),
        ("translation.fallback", "connection"),
        ("osc", "vrc_mic_intercept"),
        ("osc", "chatbox_include_source"),
    )
    for owner_path, field_name in fields:
        actual_owner = _nested_attribute(actual, owner_path)
        expected_owner = _nested_attribute(expected, owner_path)
        if actual_owner is None or expected_owner is None:
            return False
        actual_value = _enum_value_for_compare(getattr(actual_owner, field_name, None))
        expected_value = _enum_value_for_compare(getattr(expected_owner, field_name, None))
        if actual_value != expected_value:
            return False
    return True


def _osc_translation_model_value(model: object) -> str:
    value = str(getattr(model, "value", model))
    if value == "gemma4_31b_cerebras":
        return "gemma4_31b"
    return value


def _nested_attribute(value: object, path: str) -> object | None:
    current: object | None = value
    for part in path.split("."):
        current = getattr(current, part, None) if current is not None else None
    return current


def _enum_value_for_compare(value: object) -> object:
    return getattr(value, "value", value)


__all__ = ["SettingsBackedOscControlApplication"]
