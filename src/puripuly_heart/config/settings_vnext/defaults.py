from __future__ import annotations

import locale
from dataclasses import replace

from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    UiIntent,
    with_telemetry_enabled,
)


def detect_system_locale() -> str | None:
    try:
        return locale.getlocale()[0]
    except ValueError, locale.Error:
        return None


def resolve_first_run_ui_locale(system_locale: str | None) -> str:
    if system_locale is None:
        return "en"
    normalized = system_locale.strip()
    if not normalized:
        return "en"
    normalized = normalized.split(".", maxsplit=1)[0]
    normalized = normalized.split("@", maxsplit=1)[0]
    normalized = normalized.replace("_", "-").casefold()
    if normalized == "ko" or normalized.startswith(("ko-", "korean")):
        return "ko"
    if normalized == "ja" or normalized.startswith(("ja-", "japanese")):
        return "ja"
    if normalized == "zh" or normalized.startswith(("zh-", "chinese")):
        return "zh-CN"
    if normalized == "ru" or normalized.startswith(("ru-", "russian")):
        return "ru"
    return "en"


def new_settings_for_reset(system_locale: str | None = None) -> AppSettingsVNext:
    locale_value = resolve_first_run_ui_locale(
        detect_system_locale() if system_locale is None else system_locale
    )
    settings = AppSettingsVNext()
    translation = settings.intent.translation
    if locale_value == "zh-CN":
        translation = replace(
            translation,
            model="deepseek_v4_flash_41",
            connection="managed_china",
            connection_history={
                **translation.connection_history,
                "deepseek_v4_flash_41": "managed_china",
            },
            openrouter_model="deepseek/deepseek-v4.1-flash",
            openrouter_selection_alias="deepseek_v4_flash_41_managed",
            openrouter_provider_routing="deepseek_v4_flash_41_strict",
        )
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            translation=translation,
            ui=UiIntent(locale=locale_value),
        ),
    )
    return with_telemetry_enabled(settings, True)


def new_settings_for_first_run(system_locale: str | None = None) -> AppSettingsVNext:
    locale_value = resolve_first_run_ui_locale(
        detect_system_locale() if system_locale is None else system_locale
    )
    settings = AppSettingsVNext()
    translation = replace(
        settings.intent.translation,
        model="index_translate_35b",
        connection="public_api",
        openrouter_selected_source="none",
        openrouter_selection_alias=None,
        openrouter_provider_routing="default",
        connection_history={
            **settings.intent.translation.connection_history,
            "index_translate_35b": "public_api",
        },
    )
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            translation=translation,
            ui=UiIntent(locale=locale_value),
        ),
    )
    return with_telemetry_enabled(settings, True)


__all__ = [
    "detect_system_locale",
    "new_settings_for_first_run",
    "resolve_first_run_ui_locale",
]
