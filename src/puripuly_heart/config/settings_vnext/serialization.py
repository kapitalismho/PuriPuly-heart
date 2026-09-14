from __future__ import annotations

import copy
import json
import math
import types
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass, replace
from typing import Any, Final, Literal, Union, get_args, get_origin, get_type_hints

from puripuly_heart.config.llm_profiles import normalize_legacy_openrouter_model
from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
    is_safe_compatibility_extension_key,
    with_telemetry_enabled,
    with_translation_runtime_policy,
)

CANONICAL_TOP_LEVEL_KEYS: Final = frozenset({"settings_version", "intent", "state"})
_PROVIDER_VERIFICATION_FIELDS: Final = (
    "deepgram",
    "gemini_transcribe",
    "elevenlabs_scribe",
    "soniox",
    "google",
    "openrouter",
    "deepseek",
    "alibaba_beijing",
    "alibaba_singapore",
)
_PROVIDER_VERIFICATION_NON_UNKNOWN_STATUSES: Final = frozenset({"verified", "failed", "skipped"})
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
    normalized = with_translation_runtime_policy(
        with_telemetry_enabled(settings, settings.intent.telemetry.enabled)
    )
    data = asdict(normalized)
    persisted = {
        "settings_version": VNEXT_SETTINGS_SCHEMA_VERSION,
        "intent": data["intent"],
        "state": data["state"],
    }
    _merge_compatible_extensions(persisted, normalized.compatibility_extensions)
    return persisted


def to_json_text(settings: AppSettingsVNext) -> str:
    return json.dumps(
        to_persisted_dict(settings),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


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
    _validate_persisted_types(data)
    default = AppSettingsVNext(settings_version=VNEXT_SETTINGS_SCHEMA_VERSION)
    compatible_data = _with_current_settings_version(
        _normalize_legacy_openrouter_model_fields(
            _downgrade_unbound_provider_verification_entries(data)
        )
    )
    merged = _merge_dataclass(default, compatible_data, path="settings")
    if not isinstance(merged, AppSettingsVNext):
        raise TypeError("vNext settings merge produced unexpected type")
    extensions = _extract_compatible_extensions(data, default)
    extensions = _drop_removed_settings_extensions(extensions)
    merged = replace(
        merged,
        compatibility_extensions=extensions,
    )
    merged = with_telemetry_enabled(
        merged,
        merged.intent.telemetry.enabled,
    )
    return with_translation_runtime_policy(merged)


def _validate_persisted_types(data: Mapping[str, Any]) -> None:
    if not isinstance(data, Mapping):
        raise ValueError("vNext settings must be a JSON object")
    _validate_finite_json_value(data, path="settings")
    _validate_raw_value_type(data, AppSettingsVNext, path="settings")


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


def _drop_removed_settings_extensions(
    extensions: dict[str, object],
) -> dict[str, object]:
    cleaned = copy.deepcopy(extensions)
    intent = cleaned.get("intent")
    if isinstance(intent, dict):
        translation = intent.get("translation")
        if isinstance(translation, dict):
            translation.pop("cerebras", None)
            translation.pop("fallback", None)
            translation.pop("fallback_selection_alias", None)
            translation.pop("openrouter_fallback_selection_alias", None)
            if not translation:
                intent.pop("translation", None)
        for key in ("stt", "peer_stt"):
            provider = intent.get(key)
            if isinstance(provider, dict):
                provider.pop("rolling_enabled", None)
                if not provider:
                    intent.pop(key, None)
        if not intent:
            cleaned.pop("intent", None)
    state = cleaned.get("state")
    if isinstance(state, dict):
        verification = state.get("provider_verification")
        if isinstance(verification, dict):
            verification.pop("cerebras", None)
            if not verification:
                state.pop("provider_verification", None)
        if not state:
            cleaned.pop("state", None)
    return cleaned


def _with_current_settings_version(data: Mapping[str, Any]) -> Mapping[str, Any]:
    compatible = copy.deepcopy(dict(data))
    compatible["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION
    return compatible


def _normalize_legacy_openrouter_model_fields(data: Mapping[str, Any]) -> Mapping[str, Any]:
    intent = data.get("intent")
    if not isinstance(intent, Mapping):
        return data
    translation = intent.get("translation")
    if not isinstance(translation, Mapping):
        return data
    raw_model = translation.get("openrouter_model")
    normalized = normalize_legacy_openrouter_model(raw_model)
    if (
        isinstance(raw_model, str)
        and isinstance(normalized, str)
        and normalized != raw_model.strip()
    ):
        compatible = copy.deepcopy(dict(data))
        compatible_intent = dict(compatible.get("intent", {}))
        compatible_translation = dict(compatible_intent.get("translation", {}))
        compatible_translation["openrouter_model"] = normalized
        compatible_intent["translation"] = compatible_translation
        compatible["intent"] = compatible_intent
        return compatible
    return data


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
    type_hints = get_type_hints(type(default))
    for field in fields(default):
        default_value = getattr(default, field.name)
        child_path = f"{path}.{field.name}"
        if field.name not in raw:
            kwargs[field.name] = copy.deepcopy(default_value)
            continue
        raw_value = raw[field.name]
        _validate_raw_value_type(raw_value, type_hints[field.name], path=child_path)
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


def _validate_raw_value_type(value: object, expected: object, *, path: str) -> None:
    if expected is Any or expected is object:
        _validate_finite_json_value(value, path=path)
        return
    origin = get_origin(expected)
    args = get_args(expected)
    if origin in (types.UnionType, Union):
        for option in args:
            try:
                _validate_raw_value_type(value, option, path=path)
            except ValueError:
                continue
            return
        raise ValueError(f"{path} has an invalid type")
    if origin is Literal:
        if any(type(value) is type(option) and value == option for option in args):
            return
        raise ValueError(f"{path} has an unsupported value")
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a JSON array")
        item_type = args[0] if args else object
        for index, item in enumerate(value):
            _validate_raw_value_type(item, item_type, path=f"{path}[{index}]")
        return
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ValueError(f"{path} must be a JSON array")
        if args and args[-1] is Ellipsis:
            for index, item in enumerate(value):
                _validate_raw_value_type(item, args[0], path=f"{path}[{index}]")
        elif args:
            if len(value) != len(args):
                raise ValueError(f"{path} has an invalid array length")
            for index, (item, item_type) in enumerate(zip(value, args, strict=True)):
                _validate_raw_value_type(item, item_type, path=f"{path}[{index}]")
        return
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a JSON object")
        key_type, value_type = args if len(args) == 2 else (object, object)
        for key, item in value.items():
            _validate_raw_value_type(key, key_type, path=f"{path}.<key>")
            _validate_raw_value_type(item, value_type, path=f"{path}.{key}")
        return
    if isinstance(expected, type) and is_dataclass(expected):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a JSON object")
        type_hints = get_type_hints(expected)
        for field in fields(expected):
            if field.name in value:
                _validate_raw_value_type(
                    value[field.name],
                    type_hints[field.name],
                    path=f"{path}.{field.name}",
                )
        return
    if not _raw_value_matches_type(value, expected):
        raise ValueError(f"{path} has an invalid type")
    _validate_finite_json_value(value, path=path)


def _raw_value_matches_type(value: object, expected: object) -> bool:
    origin = get_origin(expected)
    if origin in (types.UnionType, Union):
        return any(_raw_value_matches_type(value, option) for option in get_args(expected))
    if origin is Literal:
        return any(type(value) is type(option) and value == option for option in get_args(expected))
    if expected is None or expected is type(None):
        return value is None
    if expected is bool:
        return type(value) is bool
    if expected is int:
        return type(value) is int
    if expected is float:
        return type(value) in (int, float) and not isinstance(value, bool)
    if expected is str:
        return isinstance(value, str)
    if expected is Any or expected is object:
        return True
    if isinstance(expected, type) and is_dataclass(expected):
        return isinstance(value, Mapping)
    if isinstance(expected, type):
        return isinstance(value, expected)
    return True


def _validate_finite_json_value(value: object, *, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must contain a finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_json_value(key, path=f"{path}.<key>")
            _validate_finite_json_value(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_finite_json_value(item, path=f"{path}[{index}]")


__all__ = [
    "CANONICAL_TOP_LEVEL_KEYS",
    "from_dict",
    "to_dict",
    "to_json_text",
]
