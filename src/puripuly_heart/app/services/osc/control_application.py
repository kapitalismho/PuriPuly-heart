from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from puripuly_heart.app.ports.osc_control import (
    FALLBACK_IDS,
    OscControlApplicationPort,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext

from .state_publisher import fallback_alias_from_settings

SettingsProvider = Callable[[], AppSettingsVNext | None]
SettingsApply = Callable[[AppSettingsVNext], Awaitable[object]]
ApplicationCall = Callable[..., Awaitable[object]]
TranslationModelNormalizer = Callable[[object], object]


@dataclass(frozen=True, slots=True)
class OscControlApplyResult:
    applied: bool
    canonical_state_changed: bool


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
        if current is not None and current.intent.languages.peer_source_mode == (
            "auto" if enabled else "manual"
        ):
            return True
        return await self._apply_settings(
            lambda settings: _with_languages(
                settings,
                peer_source_mode="auto" if enabled else "manual",
            )
        )

    async def set_self_asr(self, provider: str) -> object:
        current = self.settings_provider()
        if current is not None and current.intent.stt.provider == provider:
            return True
        return await self._apply_settings(
            lambda settings: replace(
                settings,
                intent=replace(
                    settings.intent,
                    stt=replace(settings.intent.stt, provider=provider),
                ),
            )
        )

    async def set_peer_asr(self, provider: str) -> object:
        current = self.settings_provider()
        if current is not None and current.intent.peer_stt.provider == provider:
            return True
        return await self._apply_settings(
            lambda settings: replace(
                settings,
                intent=replace(
                    settings.intent,
                    peer_stt=replace(settings.intent.peer_stt, provider=provider),
                ),
            )
        )

    async def set_translation_model(
        self,
        model: str,
        connection: str | None = None,
    ) -> object:
        current = self.settings_provider()
        if (
            current is not None
            and _osc_translation_model_value(current.intent.translation.model) == model
            and (
                connection is None
                or current.intent.translation.connection == connection
            )
        ):
            return True
        return await self._apply_settings(
            lambda settings: self._set_translation_model(settings, model, connection)
        )

    async def set_fallback(self, alias: str) -> object:
        current = self.settings_provider()
        if current is not None and fallback_alias_from_settings(current) == alias:
            return True
        return await self._apply_settings(lambda settings: self._set_fallback(settings, alias))

    async def set_mute_sync(self, enabled: bool) -> object:
        current = self.settings_provider()
        if current is not None and bool(current.intent.osc.vrc_mic_intercept) is bool(enabled):
            return True
        return await self._apply_settings(
            lambda settings: replace(
                settings,
                intent=replace(
                    settings.intent,
                    osc=replace(settings.intent.osc, vrc_mic_intercept=bool(enabled)),
                ),
            )
        )

    async def set_chatbox_source(self, enabled: bool) -> object:
        current = self.settings_provider()
        if current is not None and bool(current.intent.osc.chatbox_include_source) is bool(enabled):
            return True
        return await self._apply_settings(
            lambda settings: replace(
                settings,
                intent=replace(
                    settings.intent,
                    osc=replace(settings.intent.osc, chatbox_include_source=bool(enabled)),
                ),
            )
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

    async def _apply_settings(
        self, mutator: Callable[[AppSettingsVNext], AppSettingsVNext]
    ) -> object:
        current = self.settings_provider()
        if current is None:
            raise RuntimeError("OSC control settings are unavailable")
        previous = copy.deepcopy(current)
        updated = mutator(copy.deepcopy(current))
        if updated is None:
            updated = current
        result = await self.apply_settings(updated)
        actual = self.settings_provider()
        if not _settings_control_values_match(actual, updated):
            if not _settings_control_values_match(actual, previous):
                return OscControlApplyResult(
                    applied=False,
                    canonical_state_changed=True,
                )
            return False
        return result

    @staticmethod
    def _set_languages(
        settings: AppSettingsVNext,
        *,
        self_source: str,
        self_target: str,
        peer_source: str,
        peer_target: str,
    ) -> AppSettingsVNext:
        return _with_languages(
            settings,
            source_language=self_source,
            target_language=self_target,
            peer_source_language=peer_source,
            peer_target_language=peer_target,
        )

    def _set_translation_model(
        self,
        settings: AppSettingsVNext,
        model: str,
        connection: str | None,
    ) -> AppSettingsVNext:
        translation = settings.intent.translation
        current_value = translation.model
        next_translation = replace(
            translation,
            previous_llm_model=(
                translation.model
                if model == "custom_http" and current_value != "custom_http"
                else translation.previous_llm_model
            ),
            model=model,
            connection=connection if connection is not None else translation.connection,
        )
        updated = replace(
            settings,
            intent=replace(settings.intent, translation=next_translation),
        )
        return self.translation_model_normalizer(updated)

    @staticmethod
    def _set_fallback(settings: AppSettingsVNext, alias: str) -> AppSettingsVNext:
        if alias not in FALLBACK_IDS.values():
            raise ValueError(f"unknown OSC fallback alias: {alias}")
        fallback = settings.intent.translation.fallback
        next_fallback = replace(fallback, selection_alias=alias)
        return replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(settings.intent.translation, fallback=next_fallback),
            ),
        )


def _with_languages(settings: AppSettingsVNext, **changes: object) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            languages=replace(settings.intent.languages, **changes),
        ),
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
    languages = settings.intent.languages
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
        ("intent.languages", "source_language"),
        ("intent.languages", "target_language"),
        ("intent.languages", "peer_source_language"),
        ("intent.languages", "peer_target_language"),
        ("intent.languages", "peer_source_mode"),
        ("intent.stt", "provider"),
        ("intent.peer_stt", "provider"),
        ("intent.translation", "model"),
        ("intent.translation", "connection"),
        ("intent.translation", "http_extension_id"),
        ("intent.translation", "previous_llm_model"),
        ("intent.translation.fallback", "enabled"),
        ("intent.translation.fallback", "model"),
        ("intent.translation.fallback", "connection"),
        ("intent.osc", "vrc_mic_intercept"),
        ("intent.osc", "chatbox_include_source"),
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


__all__ = ["OscControlApplyResult", "SettingsBackedOscControlApplication"]
