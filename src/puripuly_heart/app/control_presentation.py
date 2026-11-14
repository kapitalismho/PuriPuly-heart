from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from puripuly_heart.core.language import get_language_info


@dataclass(frozen=True, slots=True)
class SettingPresentation:
    label_key: str
    description_key: str
    canonical_path: str | None
    value_kind: str
    section: str
    local_only: bool


_DEFINITIONS = {
    "locale": ("settings.section.ui", "locale", "general"),
    "clipboard.enabled": ("settings.clipboard_auto_translate", "boolean", "general"),
    "photo.enabled": ("settings.photo.enabled.title", "boolean", "photo"),
    "photo.route": ("settings.photo.route.title", "photo_route", "photo"),
    "photo.delete_translated": ("settings.photo.delete_translated.title", "boolean", "photo"),
    "chatbox.source.enabled": ("settings.chatbox_include_source", "chatbox", "general"),
    "chatbox.activation_notice.enabled": ("settings.activation_notice.title", "boolean", "general"),
    "vrc_mic_intercept.enabled": ("settings.vrc_mic_intercept", "boolean", "general"),
    "overlay.target": ("settings.overlay.caption_location", "overlay", "overlay"),
    "overlay.translation.enabled": ("settings.overlay.show_translation", "boolean", "overlay"),
    "overlay.peer_original.enabled": ("settings.overlay.show_peer_original", "boolean", "overlay"),
    "overlay.background_alpha": (
        "settings.overlay.desktop.background_alpha.title",
        "transparency",
        "overlay",
    ),
    "overlay.swap_caption_languages": (
        "settings.overlay.desktop.swap_caption_languages.title",
        "boolean",
        "overlay",
    ),
    "overlay.desktop_size": ("settings.overlay.desktop.size.title", "size", "overlay"),
    "self_vad.speech_threshold": ("settings.section.self_vad_sensitivity", "threshold", "general"),
    "peer_vad.speech_threshold": ("settings.section.peer_vad_sensitivity", "threshold", "general"),
    "peer_vad.hangover_ms": (
        "settings.presentation.peer_vad.hangover_ms",
        "milliseconds",
        "general",
    ),
    "peer_vad.pre_roll_ms": (
        "settings.presentation.peer_vad.pre_roll_ms",
        "milliseconds",
        "general",
    ),
    "peer_expected_languages": ("settings.peer_auto_languages.title", "languages", "api"),
    "stt.gpu_device_id": ("settings.gpu_device.asr", "gpu", "api"),
    "translation.gpu_device_id": ("settings.gpu_device.llm", "gpu", "api"),
    "translation.qwen.region": ("settings.qwen_region", "region", "api"),
    "translation.qwen.beijing.api_host": ("settings.qwen_api_host", "private", "api"),
    "translation.qwen.singapore.api_host": ("settings.qwen_api_host", "private", "api"),
    "translation.http_extension_id": ("settings.http_extension.title", "extension", "api"),
    "local_llm.base_url": ("settings.local_llm.base_url", "private", "api"),
    "local_llm.model": ("settings.local_llm.model", "private", "api"),
    "local_llm.extra_body_json": ("settings.local_llm.extra_body", "private", "api"),
    "stt.custom.endpoint": ("settings.custom_stt.endpoint", "private", "api"),
    "stt.custom.model": ("settings.custom_stt.model", "private", "api"),
    "stt.custom.extra_json": ("settings.custom_stt.extra", "private", "api"),
    "stt.soniox.enable_speaker_diarization": (
        "settings.soniox_speaker_diarization",
        "boolean",
        "api",
    ),
    "prompts.system_prompt_override": ("settings.section.persona", "private", "general"),
    "stt.provider": ("settings.section.stt", "provider", "api"),
    "peer_stt.provider": ("settings.section.peer_stt", "provider", "api"),
    "stt.cloud_free_tier_providers": ("settings.cloud_free_tier", "providers", "api"),
    "audio.input_host_api": ("settings.audio_host_api", "host", "general"),
    "audio.input_device": ("settings.section.microphone_audio", "device", "general"),
    "audio.output_device": ("settings.section.loopback_audio", "device", "general"),
    "translation.model": ("settings.section.translation", "provider", "api"),
    "translation.connection": ("settings.translation_connection", "connection", "api"),
    "translation.connection_history": ("settings.translation_connection", "connections", "api"),
    "translation.previous_llm_model": ("settings.section.translation", "provider", "api"),
    "prompts.value": ("settings.section.persona", "private", "general"),
    "osc.connection": ("settings.osc.connection.title", "osc", "general"),
    "custom_vocabulary": ("settings.section.custom_vocabulary", "private", "api"),
    "languages": ("settings.presentation.languages", "language_group", "general"),
    "telemetry.enabled": ("settings.telemetry.title", "boolean", "general"),
    "audio": ("settings.section.microphone_audio", "device", "general"),
    "translation": ("settings.section.translation", "provider", "api"),
}
for _name in (
    "source",
    "target",
    "secondary_target",
    "peer_source",
    "peer_target",
    "peer_source_mode",
):
    _DEFINITIONS[f"languages.{_name}"] = (
        f"settings.presentation.languages.{_name}",
        "source_mode" if _name == "peer_source_mode" else "language",
        "general",
    )

_CANONICAL_PATHS = {
    "locale": "ui.locale",
    "clipboard.enabled": "clipboard.auto_translate_enabled",
    "photo.enabled": "photo.enabled",
    "photo.route": "photo.route",
    "photo.delete_translated": "photo.delete_translated",
    "chatbox.source.enabled": "osc.chatbox_include_source",
    "chatbox.activation_notice.enabled": "osc.activation_notice_enabled",
    "vrc_mic_intercept.enabled": "osc.vrc_mic_intercept",
    "overlay.target": "overlay.target",
    "overlay.translation.enabled": "overlay.show_translation",
    "overlay.peer_original.enabled": "overlay.show_peer_original",
    "overlay.background_alpha": "overlay.desktop_flet.visual.background_alpha",
    "overlay.swap_caption_languages": "overlay.desktop_flet.swap_caption_languages",
    "overlay.desktop_size": "overlay.desktop_flet.size_preset",
    "self_vad.speech_threshold": "stt.vad_speech_threshold",
    "peer_vad.speech_threshold": "desktop_audio.vad_speech_threshold",
    "peer_vad.hangover_ms": "desktop_audio.vad_hangover_ms",
    "peer_vad.pre_roll_ms": "desktop_audio.vad_pre_roll_ms",
    "peer_expected_languages": "languages.peer_expected_languages",
    "audio.output_device": "desktop_audio.output_device",
    "local_llm.extra_body_json": "local_llm.extra_body",
    "stt.custom.extra_json": "stt.custom.extra",
    "prompts.system_prompt_override": "prompts.system_prompt_override",
    "stt.soniox.enable_speaker_diarization": "stt.soniox.enable_speaker_diarization",
}
for _field in (
    "stt.gpu_device_id",
    "translation.gpu_device_id",
    "translation.qwen.region",
    "translation.qwen.beijing.api_host",
    "translation.qwen.singapore.api_host",
    "translation.http_extension_id",
    "local_llm.base_url",
    "local_llm.model",
    "stt.custom.endpoint",
    "stt.custom.model",
    "stt.provider",
    "peer_stt.provider",
    "stt.cloud_free_tier_providers",
    "audio.input_host_api",
    "audio.input_device",
    "translation.model",
    "translation.connection",
    "translation.connection_history",
    "translation.previous_llm_model",
):
    _CANONICAL_PATHS[_field] = _field
for _name in ("source", "target", "secondary_target", "peer_source", "peer_target"):
    _CANONICAL_PATHS[f"languages.{_name}"] = f"languages.{_name}_language"
_CANONICAL_PATHS["languages.peer_source_mode"] = "languages.peer_source_mode"
_CANONICAL_PATHS["telemetry.enabled"] = "telemetry.enabled"


SETTINGS_PRESENTATION: dict[str, SettingPresentation] = {
    field: SettingPresentation(
        label,
        f"settings.description.{field}",
        _CANONICAL_PATHS.get(field),
        kind,
        section,
        kind == "private",
    )
    for field, (label, kind, section) in _DEFINITIONS.items()
}


_SECRET_OWNERS = {
    "alibaba_api_key_beijing": "provider.alibaba_beijing",
    "alibaba_api_key_singapore": "provider.alibaba_singapore",
    "custom_stt_api_key": "settings.custom_stt.title",
}
_BRAND_NAMES = {"openai": "OpenAI", "qq": "QQ", "discord": "Discord", "chatgpt": "ChatGPT"}
COMMAND_CHANGES = {
    "secrets.delete": False,
    "auth.logout": False,
    "overlay.position.reset": "default",
}


def command_field(command: str, arguments: Mapping[str, object]) -> str:
    if command == "secrets.delete":
        return f"secrets.{arguments['name']}"
    if command == "auth.logout":
        return f"accounts.{arguments['provider']}"
    if command == "overlay.position.reset":
        return "overlay.desktop_position"
    raise ValueError(command)


def command_field_kind(field: str) -> str | None:
    if field.startswith("secrets."):
        return "secret"
    if field.startswith("accounts."):
        return "account"
    if field == "overlay.desktop_position":
        return "position"
    return None


def _owner_name(code: str, localize: Callable[[str], str]) -> str:
    return _translated(f"provider.{code}", _BRAND_NAMES.get(code, code), localize)


def _command_label(field: str, kind: str, localize: Callable[[str], str]) -> str:
    name = field.split(".", 1)[1]
    if kind == "secret":
        key = _SECRET_OWNERS.get(name)
        owner = (
            _translated(key, name, localize)
            if key is not None
            else (
                _owner_name(name.removesuffix("_api_key"), localize)
                if name.endswith("_api_key")
                else name
            )
        )
        return localize("settings.presentation.secret").format(owner=owner)
    if kind == "account":
        return localize("settings.presentation.account").format(
            provider=_owner_name(name, localize)
        )
    return localize("settings.presentation.desktop_position")


def _command_value(kind: str, value: object, localize: Callable[[str], str]) -> str:
    keys = {
        ("secret", True): "settings.presentation.secret.saved",
        ("secret", False): "settings.presentation.secret.none",
        ("account", True): "settings.presentation.account.signed_in",
        ("account", False): "settings.presentation.account.signed_out",
        ("position", "default"): "settings.presentation.desktop_position.default",
    }
    key = keys.get((kind, value)) if type(value) in (bool, str) else None
    return localize(key or "settings.presentation.unknown")


def setting_label(field: str, localize: Callable[[str], str]) -> str:
    kind = command_field_kind(field)
    if kind is not None:
        return _command_label(field, kind, localize)
    descriptor = SETTINGS_PRESENTATION.get(field)
    return localize(descriptor.label_key) if descriptor else field


def _translated(key: str, fallback: str, localize: Callable[[str], str]) -> str:
    text = localize(key)
    return fallback if text == key else text


def setting_value_text(
    field: str,
    value: object,
    localize: Callable[[str], str],
    *,
    choice_labels: Mapping[str, str] | None = None,
) -> str:
    command_kind = command_field_kind(field)
    if command_kind is not None:
        return _command_value(command_kind, value, localize)
    descriptor = SETTINGS_PRESENTATION.get(field)
    if descriptor is None:
        return localize("settings.presentation.unknown")
    kind = descriptor.value_kind
    if descriptor.local_only:
        return localize("settings.presentation.local_only")
    if choice_labels is not None and isinstance(value, str) and value in choice_labels:
        return choice_labels[value]
    if kind == "transparency" and isinstance(value, int | float) and not isinstance(value, bool):
        return f"{round((1 - value) * 100)}%"
    if kind == "threshold" and isinstance(value, int | float) and not isinstance(value, bool):
        return f"{value:.2f}"
    if kind == "milliseconds" and isinstance(value, int | float):
        return f"{int(value)}ms"
    if isinstance(value, bool):
        if kind == "chatbox":
            return localize(
                "settings.chatbox_source.on" if value else "settings.chatbox_source.off"
            )
        return localize("settings.option.on" if value else "settings.option.off")
    if value is None or value == "":
        return localize(
            "settings.default_option"
            if kind in {"host", "device"}
            else "dashboard.language.secondary_target.none"
        )
    if kind in {"languages", "providers"} and isinstance(value, list | tuple):
        leaf = "languages.source" if kind == "languages" else "stt.provider"
        return " · ".join(setting_value_text(leaf, item, localize) for item in value)
    if isinstance(value, dict):
        if kind == "language_group":
            return " · ".join(
                f"{setting_label(f'languages.{name}', localize)}: {setting_value_text(f'languages.{name}', item, localize)}"
                for name, item in value.items()
            )
        if kind == "connections":
            return " · ".join(
                f"{setting_value_text('translation.model', name, localize)}: {setting_value_text('translation.connection', item, localize)}"
                for name, item in value.items()
            )
        if kind == "osc":
            labels = {
                "mode": "settings.osc.mode",
                "send_port": "settings.osc.send_port",
                "receive_port": "settings.osc.receive_port",
            }
            return " · ".join(
                f"{localize(labels.get(name, name))}: {localize(f'settings.osc.mode.{item}') if name == 'mode' else item}"
                for name, item in value.items()
            )
        return localize("settings.presentation.unknown")
    text = str(value)
    if kind == "language":
        info = get_language_info(text)
        return _translated(f"language.{info.code}", info.name, localize) if info else text
    if kind == "source_mode":
        key = {"auto": "dashboard.peer_source.automatic", "manual": "settings.osc.mode.manual"}.get(
            text
        )
        return localize(key or "settings.presentation.unknown")
    prefixes = {
        "provider": "provider",
        "locale": "locale",
        "overlay": "settings.overlay.target",
        "photo_route": "settings.photo.route",
        "size": "settings.overlay.desktop.size.option",
        "connection": "settings.translation_connection",
        "region": "region",
    }
    if kind in prefixes:
        return _translated(f"{prefixes[kind]}.{text}", text, localize)
    if kind == "gpu" and text == "auto":
        return localize("settings.gpu_device.auto")
    if kind == "gpu":
        return localize("settings.gpu_device.unavailable").format(device=text)
    if kind == "host":
        host_keys = {
            "MME": "windows_mme",
            "Windows WASAPI": "windows_wasapi",
            "Windows WASAPI (Compatibility Mode)": "windows_wasapi_compatibility",
            "Windows DirectSound": "windows_directsound",
        }
        return _translated(
            f"settings.audio_host_api.option.{host_keys.get(text, text)}", text, localize
        )
    return text


def setting_catalog(catalog: dict, localize: Callable[[str], str]) -> dict:
    fields = list(catalog.get("fields", ()))
    fields.extend(
        f"languages.{name}"
        for name in (
            "source",
            "target",
            "secondary_target",
            "peer_source",
            "peer_target",
            "peer_source_mode",
        )
    )
    choices = catalog.get("choices", {})
    result = {}
    for field in fields:
        descriptor = SETTINGS_PRESENTATION.get(field)
        if descriptor is None:
            continue
        options = choices.get(field)
        schema = catalog.get("field_schemas", {}).get(field, {})
        if field.startswith("languages."):
            name = field.removeprefix("languages.")
            schema = (
                catalog.get("field_schemas", {})
                .get("languages", {})
                .get("properties", {})
                .get(name, {})
            )
            languages = choices.get("languages", {})
            options = languages.get(
                "peer_source_modes" if name == "peer_source_mode" else "codes", []
            )
            if schema.get("empty_allowed"):
                options = ["", *options]
        choice_field = (
            "translation.connection" if field == "translation.connection_history" else field
        )

        def choice_labels(items: object, label_field: str = choice_field) -> object:
            if isinstance(items, dict):
                return {
                    name: choice_labels(
                        values,
                        (
                            "languages.peer_source_mode"
                            if name == "peer_source_modes"
                            else "languages.source" if field == "languages" else label_field
                        ),
                    )
                    for name, values in items.items()
                }
            if isinstance(items, list | tuple):
                labels = {
                    str(item["value"]): str(item.get("label") or item.get("display_name"))
                    for item in items
                    if isinstance(item, dict)
                    and "value" in item
                    and (item.get("label") or item.get("display_name"))
                    and item.get("available", True)
                }
                return [
                    (
                        {
                            **item,
                            "label": setting_value_text(
                                label_field, item.get("value"), localize, choice_labels=labels
                            ),
                        }
                        if isinstance(item, dict)
                        else {
                            "value": item,
                            "label": setting_value_text(
                                label_field, item, localize, choice_labels=labels
                            ),
                        }
                    )
                    for item in items
                ]
            return items

        result[field] = {
            "label": setting_label(field, localize),
            "description": localize(descriptor.description_key),
            "section": localize(f"settings.subtab.{descriptor.section}"),
            "local_only": descriptor.local_only,
            "value_kind": descriptor.value_kind,
            "schema": schema,
            "choices": choice_labels(options),
        }
    return result
