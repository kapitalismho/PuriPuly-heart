from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass, replace
from typing import Any, Final

from puripuly_heart.config.settings_vnext.schema import (
    DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS,
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
    ensure_telemetry_default_allow,
    is_safe_compatibility_extension_key,
    with_telemetry_consent,
    with_translation_runtime_policy,
)

CANONICAL_TOP_LEVEL_KEYS: Final = frozenset({"settings_version", "intent", "state"})
_PROVIDER_VERIFICATION_FIELDS: Final = (
    "deepgram",
    "soniox",
    "google",
    "openrouter",
    "deepseek",
    "cerebras",
    "alibaba_beijing",
    "alibaba_singapore",
)
_PROVIDER_VERIFICATION_NON_UNKNOWN_STATUSES: Final = frozenset({"verified", "failed", "skipped"})
_FALLBACK_DISABLED: Final = {"enabled": False}
_FALLBACK_DEFAULT: Final = {
    "enabled": True,
    "model": "gemma4_26b_31b",
    "connection": "openrouter",
    "selection_alias": DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS,
}
_TEMPORARY_GENERIC_FALLBACK_ALIASES: Final = {
    "none": {"enabled": False},
    "deepseek_v4_flash_official": {
        "enabled": True,
        "model": "deepseek_v4_flash",
        "connection": "official_byok",
        "selection_alias": "deepseek_v4_flash_official",
    },
    "openrouter_deepseek_v4_flash": {
        "enabled": True,
        "model": "deepseek_v4_flash",
        "connection": "openrouter",
        "selection_alias": "openrouter_deepseek_v4_flash",
    },
    "openrouter_gemma4_26b_a4b": {
        "enabled": True,
        "model": "gemma4",
        "connection": "openrouter",
        "selection_alias": "openrouter_gemma4_26b_a4b",
    },
    DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS: {
        "enabled": True,
        "model": "gemma4_26b_31b",
        "connection": "openrouter",
        "selection_alias": "openrouter_gemma4_26b_31b",
    },
    "openrouter_gemma4_31b": {
        "enabled": True,
        "model": "gemma4_31b",
        "connection": "openrouter",
        "selection_alias": "openrouter_gemma4_31b",
    },
    "managed_gemma4_26b_31b": {
        "enabled": True,
        "model": "gemma4_26b_31b",
        "connection": "managed",
        "selection_alias": "managed_gemma4_26b_31b",
    },
    "managed_gemma4_31b": {
        "enabled": True,
        "model": "gemma4_31b",
        "connection": "managed",
        "selection_alias": "managed_gemma4_31b",
    },
    "cerebras_gemma4_31b": {
        "enabled": True,
        "model": "gemma4_31b",
        "connection": "cerebras",
        "selection_alias": "cerebras_gemma4_31b",
    },
}
_FALLBACK_FIELDS_ALIAS: Final = {
    (False, "deepseek_v4_flash", "official_byok"): "none",
    (True, "deepseek_v4_flash", "official_byok"): "deepseek_v4_flash_official",
    (True, "deepseek_v4_flash", "openrouter"): "openrouter_deepseek_v4_flash",
    (True, "gemma4", "openrouter"): "openrouter_gemma4_26b_a4b",
    (True, "gemma4_26b_31b", "openrouter"): DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS,
    (True, "gemma4_31b", "openrouter"): "openrouter_gemma4_31b",
    (True, "gemma4_26b_31b", "managed"): "managed_gemma4_26b_31b",
    (True, "gemma4_31b", "managed"): "managed_gemma4_31b",
    (True, "gemma4_31b", "cerebras"): "cerebras_gemma4_31b",
    (True, "gemma4_31b_cerebras", "official_byok"): "cerebras_gemma4_31b",
    (True, "deepseek_v4_flash", "managed_china"): "deepseek_v4_flash_china",
}
_OPEN_MAPPING_PATHS: Final = frozenset(
    {
        ("intent", "translation", "connection_history"),
        ("intent", "local_llm", "extra_body"),
        ("intent", "stt", "custom_terms"),
        ("intent", "stt", "custom", "extra"),
    }
)


def to_dict(settings: AppSettingsVNext) -> dict[str, Any]:
    """Serialize canonical vNext settings.

    The vNext persisted schema intentionally writes no legacy projection keys. Runtime-only
    state and raw secret values are excluded by the schema itself.
    """

    if not isinstance(settings, AppSettingsVNext):
        raise TypeError("vNext settings serializer requires AppSettingsVNext")
    normalized = with_translation_runtime_policy(settings)
    data = asdict(normalized)
    persisted = {
        "settings_version": VNEXT_SETTINGS_SCHEMA_VERSION,
        "intent": data["intent"],
        "state": data["state"],
    }
    _merge_compatible_extensions(persisted, normalized.compatibility_extensions)
    return persisted


def to_json_text(settings: AppSettingsVNext) -> str:
    return json.dumps(to_persisted_dict(settings), ensure_ascii=False, indent=2)


def to_persisted_dict(settings: AppSettingsVNext) -> dict[str, Any]:
    return normalize_persisted_dict(to_dict(settings))


def normalize_persisted_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(data))
    state = normalized.get("state")
    if not isinstance(state, dict):
        return normalized
    entries = state.get("provider_verification")
    if not isinstance(entries, dict):
        return normalized
    for provider in _PROVIDER_VERIFICATION_FIELDS:
        entry = entries.get(provider)
        if isinstance(entry, Mapping) and entry.get("status") == "unknown":
            entries[provider] = {"status": "unknown"}
    return normalized


def from_dict(data: Mapping[str, Any]) -> AppSettingsVNext:
    if not isinstance(data, Mapping):
        raise ValueError("vNext settings must be a JSON object")
    default = AppSettingsVNext(settings_version=VNEXT_SETTINGS_SCHEMA_VERSION)
    compatible_data = _with_current_settings_version(
        _project_legacy_translation_fallback_fields(
            _downgrade_unbound_provider_verification_entries(data)
        )
    )
    merged = _merge_dataclass(default, compatible_data, path="settings")
    if not isinstance(merged, AppSettingsVNext):
        raise TypeError("vNext settings merge produced unexpected type")
    merged = replace(
        merged,
        compatibility_extensions=_extract_compatible_extensions(data, default),
    )
    merged = with_telemetry_consent(
        merged,
        merged.intent.telemetry.consent,
    )
    return with_translation_runtime_policy(ensure_telemetry_default_allow(merged))


def _extract_compatible_extensions(
    raw: Mapping[str, Any],
    default: AppSettingsVNext,
) -> dict[str, object]:
    template_data = asdict(default)
    template = {
        "settings_version": VNEXT_SETTINGS_SCHEMA_VERSION,
        "intent": template_data["intent"],
        "state": template_data["state"],
    }
    return _extract_unknown_mapping(raw, template, path=())


def _extract_unknown_mapping(
    raw: Mapping[object, object],
    template: Mapping[object, object],
    *,
    path: tuple[str, ...],
) -> dict[str, object]:
    extensions: dict[str, object] = {}
    for key, value in raw.items():
        if not is_safe_compatibility_extension_key(key):
            continue
        if key not in template:
            extensions[str(key)] = copy.deepcopy(value)
            continue
        template_value = template[key]
        if isinstance(value, Mapping) and isinstance(template_value, Mapping):
            child_path = (*path, str(key))
            if child_path in _OPEN_MAPPING_PATHS or child_path[-1] in {
                "verifier_context",
                "verifier_evidence",
            }:
                continue
            nested = _extract_unknown_mapping(value, template_value, path=child_path)
            if nested:
                extensions[str(key)] = nested
    return extensions


def _merge_compatible_extensions(
    target: dict[str, Any],
    extensions: Mapping[str, object],
) -> None:
    for key, value in extensions.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
            continue
        target_value = target[key]
        if isinstance(target_value, dict) and isinstance(value, Mapping):
            _merge_compatible_extensions(target_value, value)


def _with_current_settings_version(data: Mapping[str, Any]) -> Mapping[str, Any]:
    compatible = copy.deepcopy(dict(data))
    compatible["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION
    return compatible


def _downgrade_unbound_provider_verification_entries(
    data: Mapping[str, Any],
) -> Mapping[str, Any]:
    state = data.get("state")
    if not isinstance(state, Mapping):
        return data
    provider_verification = state.get("provider_verification")
    if not isinstance(provider_verification, Mapping):
        return data

    entries_to_downgrade = {
        provider
        for provider in _PROVIDER_VERIFICATION_FIELDS
        if _is_unbound_non_unknown_provider_verification_entry(provider_verification.get(provider))
    }
    if not entries_to_downgrade:
        return data

    compatible = copy.deepcopy(dict(data))
    compatible_state = dict(compatible.get("state", {}))
    compatible_provider_verification = dict(compatible_state.get("provider_verification", {}))
    for provider in entries_to_downgrade:
        compatible_provider_verification[provider] = {"status": "unknown"}
    compatible_state["provider_verification"] = compatible_provider_verification
    compatible["state"] = compatible_state
    return compatible


def _project_legacy_translation_fallback_fields(data: Mapping[str, Any]) -> Mapping[str, Any]:
    intent = data.get("intent")
    if not isinstance(intent, Mapping):
        return data
    translation = intent.get("translation")
    if not isinstance(translation, Mapping):
        return data
    if (
        "fallback" in translation
        and "fallback_selection_alias" not in translation
        and "openrouter_fallback_selection_alias" not in translation
        and (
            not isinstance(translation.get("fallback"), Mapping)
            or "selection_alias" in translation.get("fallback", {})
        )
    ):
        return data

    compatible = copy.deepcopy(dict(data))
    compatible_intent = dict(compatible.get("intent", {}))
    compatible_translation = dict(compatible_intent.get("translation", {}))
    explicit_fallback = compatible_translation.get("fallback")
    if isinstance(explicit_fallback, Mapping):
        compatible_translation["fallback"] = _fallback_with_inferred_selection_alias(
            explicit_fallback
        )
    else:
        alias = compatible_translation.get("fallback_selection_alias")
        if isinstance(alias, str):
            compatible_translation["fallback"] = _fallback_from_temporary_alias(alias)
        else:
            compatible_translation["fallback"] = _fallback_from_legacy_openrouter_alias(
                compatible_translation.get("openrouter_fallback_selection_alias"),
                selected_source=compatible_translation.get("openrouter_selected_source"),
            )
    compatible_translation.pop("fallback_selection_alias", None)
    compatible_translation.pop("openrouter_fallback_selection_alias", None)
    compatible_intent["translation"] = compatible_translation
    compatible["intent"] = compatible_intent
    return compatible


def _fallback_with_inferred_selection_alias(value: Mapping[object, object]) -> dict[str, object]:
    fallback = copy.deepcopy(dict(value))
    if not fallback:
        return dict(_FALLBACK_DEFAULT)
    if "selection_alias" not in fallback:
        fields = (
            bool(fallback.get("enabled", False)),
            str(fallback.get("model", "deepseek_v4_flash")),
            str(fallback.get("connection", "official_byok")),
        )
        fallback["selection_alias"] = (
            DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS
            if fields == (False, "deepseek_v4_flash", "official_byok")
            else _FALLBACK_FIELDS_ALIAS.get(fields, "none")
        )
    return fallback


def _fallback_from_temporary_alias(value: str) -> dict[str, object]:
    alias = value.strip()
    fallback = dict(
        _TEMPORARY_GENERIC_FALLBACK_ALIASES.get(
            alias,
            _FALLBACK_DEFAULT if not alias else _FALLBACK_DISABLED,
        )
    )
    fallback.setdefault(
        "selection_alias",
        DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS if not alias else "none",
    )
    return fallback


def _fallback_from_legacy_openrouter_alias(
    value: object,
    *,
    selected_source: object,
) -> dict[str, object]:
    if value is None:
        return dict(_FALLBACK_DEFAULT)
    if not isinstance(value, str):
        return {**_FALLBACK_DISABLED, "selection_alias": "none"}
    alias = value.strip()
    if not alias:
        return dict(_FALLBACK_DEFAULT)
    if alias in ("none", "qwen35_flash"):
        return {**_FALLBACK_DISABLED, "selection_alias": "none"}
    if alias == "deepseek_v4_flash_china":
        return {
            "enabled": True,
            "model": "deepseek_v4_flash",
            "connection": "managed_china",
            "selection_alias": "deepseek_v4_flash_china",
        }
    if alias == "deepseek_v4_flash":
        if selected_source in {"managed", "byok"}:
            connection = "openrouter"
        else:
            return {**_FALLBACK_DISABLED, "selection_alias": "none"}
        return {
            "enabled": True,
            "model": "deepseek_v4_flash",
            "connection": connection,
            "selection_alias": "openrouter_deepseek_v4_flash",
        }
    return {**_FALLBACK_DISABLED, "selection_alias": "none"}


def _is_unbound_non_unknown_provider_verification_entry(entry: object) -> bool:
    if not isinstance(entry, Mapping):
        return False
    if entry.get("status") not in _PROVIDER_VERIFICATION_NON_UNKNOWN_STATUSES:
        return False
    return not _has_provider_verification_binding_evidence(entry)


def _has_provider_verification_binding_evidence(entry: Mapping[object, object]) -> bool:
    return (
        _is_non_empty_string(entry.get("provider"))
        and _is_non_empty_string(entry.get("secret_key"))
        and (
            _is_non_empty_string(entry.get("secret_revision"))
            or _is_non_empty_string(entry.get("secret_fingerprint"))
        )
        and isinstance(entry.get("verifier_context"), Mapping)
        and bool(entry.get("verifier_context"))
    )


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _merge_dataclass(default: object, raw: object, *, path: str) -> object:
    if not is_dataclass(default) or isinstance(default, type):
        return copy.deepcopy(raw)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must be a JSON object")

    kwargs: dict[str, object] = {}
    for field in fields(default):
        default_value = getattr(default, field.name)
        child_path = f"{path}.{field.name}"
        if field.name not in raw:
            kwargs[field.name] = copy.deepcopy(default_value)
            continue
        raw_value = raw[field.name]
        if is_dataclass(default_value) and not isinstance(default_value, type):
            kwargs[field.name] = _merge_dataclass(default_value, raw_value, path=child_path)
        elif isinstance(default_value, dict):
            if not isinstance(raw_value, Mapping):
                raise ValueError(f"{child_path} must be a JSON object")
            kwargs[field.name] = copy.deepcopy(dict(raw_value))
        elif isinstance(default_value, list):
            if not isinstance(raw_value, list):
                raise ValueError(f"{child_path} must be a JSON array")
            kwargs[field.name] = copy.deepcopy(raw_value)
        else:
            kwargs[field.name] = copy.deepcopy(raw_value)

    merged = type(default)(**kwargs)
    validate = getattr(merged, "validate", None)
    if callable(validate):
        validate()
    return merged


__all__ = [
    "CANONICAL_TOP_LEVEL_KEYS",
    "from_dict",
    "to_dict",
    "to_json_text",
]
