from __future__ import annotations

import json
import logging
from importlib import resources
from typing import Any

from puripuly_heart.core.language import get_language_info
from puripuly_heart.core.messages import UserMessageRef

logger = logging.getLogger(__name__)

_I18N_DIR = "data/i18n"
_DEFAULT_LOCALE = "en"
_FALLBACK_LOCALE = "en"
_LOCALE_DISPLAY_ORDER = ("en", "ko", "zh-CN", "ja", "ru")
_LOCALE_DISPLAY_RANK = {code: index for index, code in enumerate(_LOCALE_DISPLAY_ORDER)}

_current_locale = _DEFAULT_LOCALE
_bundles: dict[str, dict[str, str]] = {}
_locale_cache: tuple[str, ...] | None = None


def _locale_display_sort_key(locale_code: str) -> tuple[int, str]:
    return (
        _LOCALE_DISPLAY_RANK.get(locale_code, len(_LOCALE_DISPLAY_ORDER)),
        locale_code.casefold(),
    )


def _load_bundle(locale: str) -> dict[str, str]:
    if locale in _bundles:
        return _bundles[locale]

    data: dict[str, str] = {}
    try:
        bundle_path = resources.files("puripuly_heart").joinpath(f"{_I18N_DIR}/{locale}.json")
        if bundle_path.is_file():
            raw = json.loads(bundle_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = {
                    str(key): value
                    for key, value in raw.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
    except Exception as exc:
        logger.warning("Failed to load i18n bundle for locale '%s': %s", locale, exc)

    _bundles[locale] = data
    return data


def available_locales() -> tuple[str, ...]:
    global _locale_cache
    if _locale_cache is not None:
        return _locale_cache

    locales: list[str] = []
    try:
        base = resources.files("puripuly_heart").joinpath(_I18N_DIR)
        for entry in base.iterdir():
            if not entry.is_file():
                continue
            name = entry.name
            if name.endswith(".json"):
                locales.append(name[:-5])
    except Exception:
        locales = []

    if not locales:
        locales = [_DEFAULT_LOCALE]

    locales = sorted(locales, key=_locale_display_sort_key)

    _locale_cache = tuple(locales)
    return _locale_cache


def resolve_locale(locale: str | None) -> str:
    if not locale:
        return _DEFAULT_LOCALE
    candidates = available_locales()
    if locale in candidates:
        return locale
    base = locale.split("-")[0]
    if base in candidates:
        return base
    return _DEFAULT_LOCALE


def set_locale(locale: str | None) -> str:
    global _current_locale
    _current_locale = resolve_locale(locale)
    _load_bundle(_current_locale)
    _load_bundle(_FALLBACK_LOCALE)
    return _current_locale


def get_locale() -> str:
    return _current_locale


def t(key: str, *, default: str | None = None, **params: Any) -> str:
    value = _load_bundle(_current_locale).get(key)
    if value is None:
        value = _load_bundle(_FALLBACK_LOCALE).get(key)
    if value is None:
        value = default if default is not None else key
    if params:
        try:
            return value.format(**params)
        except Exception:
            logger.debug("Failed to format i18n key '%s' with params %s", key, params)
    return value


def t_for_locale(
    locale: str | None,
    key: str,
    *,
    default: str | None = None,
    **params: Any,
) -> str:
    resolved_locale = resolve_locale(locale)
    value = _load_bundle(resolved_locale).get(key)
    if value is None:
        value = _load_bundle(_FALLBACK_LOCALE).get(key)
    if value is None:
        value = default if default is not None else key
    if params:
        try:
            return value.format(**params)
        except Exception:
            logger.debug(
                "Failed to format i18n key '%s' for locale '%s' with params %s",
                key,
                resolved_locale,
                params,
            )
    return value


def localize_user_message_ref(message: UserMessageRef) -> str:
    params = dict(message.params)
    category = params.get("category")
    if isinstance(category, str) and category:
        params["category"] = t(f"error.category.{category}", default=category)
    return t(message.key, **params)


def language_name(code: str) -> str:
    info = get_language_info(code)
    if not info:
        return code
    return t(f"language.{info.code}", default=info.name)


def locale_label(locale_code: str) -> str:
    return t(f"locale.{locale_code}", default=locale_code)


def provider_label(provider_code: str) -> str:
    return t(f"provider.{provider_code}", default=provider_code)


_SOURCE_KEY_MAP = {
    "Managed": "dashboard.trial.source.managed",
    "You": "source.you",
    "Mic": "source.mic",
    "VRChat": "source.vrchat",
    "Clipboard": "source.clipboard",
}


def source_label(source: str | None) -> str:
    if not source:
        return t("source.unknown", default="")
    return t(_SOURCE_KEY_MAP.get(source, ""), default=source)


def translated_source_label(source: str) -> str:
    return t("history.translated_source", source=source)
