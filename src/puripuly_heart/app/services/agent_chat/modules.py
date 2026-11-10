from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy

from puripuly_heart.app.control_presentation import SETTINGS_PRESENTATION, setting_label
from puripuly_heart.app.services.agent_chat.provider_setup import PROVIDER_SETUPS
from puripuly_heart.app.services.agent_chat.tools import LANGUAGE_FIELD_PATHS, SETTINGS_FIELD_PATHS

MODULES = (
    "core",
    "audio",
    "recognition",
    "translation",
    "accounts",
    "languages",
    "output",
    "general",
)
MODULE_DESCRIPTIONS = {
    "audio": "Microphone, loopback, audio devices, host APIs and voice activity detection.",
    "recognition": "Speech recognition providers, quality, cloud allowances, GPU choice, vocabulary and recognition key readiness.",
    "translation": "Translation models, connections, speed, GPU choice, custom routes, prompts and translation key readiness.",
    "accounts": "ChatGPT subscription/sign-in and managed QQ/Discord account readiness, usage and account actions.",
    "languages": "Self and Peer source/target languages, secondary translation and automatic language detection.",
    "output": "Captions, desktop/VR overlays, VRChat chatbox, OSC connection and microphone mute synchronization.",
    "general": "App UI language, clipboard and anonymous usage-data settings.",
}
TOPICS = frozenset({"overview", "diagnostics", *MODULES[1:]})
MODULE_FIELDS = {
    "audio": (
        "audio.input_host_api",
        "audio.input_device",
        "audio.output_device",
        "self_vad.speech_threshold",
        "peer_vad.speech_threshold",
        "peer_vad.hangover_ms",
        "peer_vad.pre_roll_ms",
    ),
    "recognition": (
        "stt.provider",
        "peer_stt.provider",
        "stt.cloud_free_tier_providers",
        "stt.soniox.enable_speaker_diarization",
        "stt.gpu_device_id",
        "stt.custom.endpoint",
        "stt.custom.model",
        "stt.custom.extra_json",
        "custom_vocabulary",
    ),
    "translation": (
        "translation.model",
        "translation.connection",
        "translation.connection_history",
        "translation.previous_llm_model",
        "translation.gpu_device_id",
        "translation.http_extension_id",
        "translation.qwen.region",
        "translation.qwen.beijing.api_host",
        "translation.qwen.singapore.api_host",
        "local_llm.base_url",
        "local_llm.model",
        "local_llm.extra_body_json",
        "prompts.system_prompt_override",
        "prompts.value",
    ),
    "languages": (
        "languages.source",
        "languages.target",
        "languages.secondary_target",
        "languages.peer_source",
        "languages.peer_target",
        "languages.peer_source_mode",
        "peer_expected_languages",
    ),
    "output": (
        "overlay.target",
        "overlay.translation.enabled",
        "overlay.peer_original.enabled",
        "overlay.background_alpha",
        "overlay.swap_caption_languages",
        "overlay.desktop_size",
        "osc.connection",
        "chatbox.source.enabled",
        "chatbox.activation_notice.enabled",
        "vrc_mic_intercept.enabled",
    ),
    "general": ("locale", "clipboard.enabled", "telemetry.enabled"),
}
MODULE_KEYS = {
    "recognition": (
        "deepgram_api_key",
        "gemini_api_key",
        "elevenlabs_scribe_api_key",
        "soniox_api_key",
        "custom_stt_api_key",
        "alibaba_api_key_beijing",
        "alibaba_api_key_singapore",
    ),
    "translation": (
        "openrouter_api_key",
        "openai_api_key",
        "deepseek_api_key",
        "local_llm_api_key",
        "alibaba_api_key_beijing",
        "alibaba_api_key_singapore",
    ),
}
_SETTINGS_QUERIES = frozenset({"settings.current", "settings.choices"})
_CORE_QUERIES = frozenset({"app.status", "settings.current", "diagnostics"})
_MODULE_QUERIES = {
    "audio": _SETTINGS_QUERIES | {"audio.devices"},
    "recognition": _SETTINGS_QUERIES | {"secrets.presence"},
    "translation": _SETTINGS_QUERIES | {"secrets.presence"},
    "accounts": frozenset({"accounts.status"}),
    "languages": _SETTINGS_QUERIES,
    "output": _SETTINGS_QUERIES | {"overlay.status", "osc.status"},
    "general": _SETTINGS_QUERIES,
}
_LABEL_SOURCES = {
    "peer_expected_languages": "languages.source",
    "stt.cloud_free_tier_providers": "stt.provider",
}
_UNITS = {"milliseconds": "ms"}
_DYNAMIC_CHOICE_KINDS = frozenset({"device", "host", "gpu", "extension"})
_DIAGNOSTIC_COVERAGE = (
    "lookup",
    "oldest_sequence",
    "latest_sequence",
    "next_after",
    "gap",
    "evicted_records",
    "truncated",
)
_HEALTH_COUNTERS = {
    "history": (
        "evicted_records",
        "rejected_records",
        "subscriber_delivery_failures",
        "file_projection_failures",
    ),
    "file": ("delivery_failures", "dropped_records"),
}


def topic_modules(topic: str) -> tuple[str, ...]:
    if topic not in TOPICS:
        raise ValueError("unsupported context topic")
    if topic == "overview":
        return MODULES
    if topic == "diagnostics":
        return ("core",)
    return ("core", topic)


def topic_queries(topic: str) -> frozenset[str]:
    queries = set(_CORE_QUERIES)
    for module in topic_modules(topic):
        queries |= _MODULE_QUERIES.get(module, frozenset())
    return frozenset(queries)


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def setting_values(settings: Mapping) -> dict:
    values = {
        field: deepcopy(settings[path])
        for field, path in SETTINGS_FIELD_PATHS.items()
        if path in settings
    }
    values.update(
        {
            f"languages.{name}": deepcopy(settings[path])
            for name, path in LANGUAGE_FIELD_PATHS.items()
            if path in settings
        }
    )
    return values


def _static_choices(choices: object) -> bool:
    if isinstance(choices, Mapping):
        return all(_static_choices(value) for value in choices.values())
    return isinstance(choices, list) and all(
        isinstance(item, Mapping)
        and "value" in item
        and set(item) <= {"value", "label"}
        and not isinstance(item["value"], (Mapping, list))
        and not isinstance(item.get("label"), (Mapping, list))
        for item in choices
    )


def _options(choices: object) -> object:
    if isinstance(choices, Mapping):
        return {key: _options(value) for key, value in choices.items()}
    if not isinstance(choices, list):
        return deepcopy(choices)
    if all(
        isinstance(item, Mapping)
        and isinstance(item.get("value"), str)
        and set(item) <= {"value", "label"}
        for item in choices
    ) and len({item["value"] for item in choices}) == len(choices):
        return {item["value"]: deepcopy(item.get("label", item["value"])) for item in choices}
    return [
        (
            {
                key: deepcopy(value)
                for key, value in item.items()
                if not (key == "label" and value == item["value"])
            }
            if isinstance(item, Mapping) and "value" in item
            else _options(item)
        )
        for item in choices
    ]


def _settings(
    snapshot: Mapping, fields: tuple[str, ...], localize: Callable[[str], str]
) -> tuple[dict, dict]:
    missing = snapshot.get("missing", ())
    catalog = (
        _mapping(snapshot.get("settings_catalog")) if "settings.choices" not in missing else {}
    )
    values = (
        setting_values(_mapping(snapshot.get("settings")))
        if "settings.current" not in missing
        else {}
    )
    private = _mapping(snapshot.get("local_settings")) if "settings.current" not in missing else {}
    result: dict = {"settings": {}}
    options: dict[str, object] = {}
    shared: dict[str, str] = {}
    groups: dict[str, dict] = {}
    for field in fields:
        descriptor = SETTINGS_PRESENTATION[field]
        entry: dict = {"label": setting_label(field, localize)}
        description = localize(descriptor.description_key)
        if description != entry["label"]:
            entry["description"] = description
        if descriptor.local_only:
            presence = _mapping(private.get(field)).get("presence")
            result.setdefault("private", {})[field] = (
                "saved" if presence is True else "empty" if presence is False else "unknown"
            )
        elif field in values:
            result["settings"][field] = values[field]
        if descriptor.value_kind in _UNITS:
            entry["unit"] = _UNITS[descriptor.value_kind]
        raw_choices = _mapping(catalog.get(field)).get("choices")
        if raw_choices is not None and not descriptor.local_only:
            choices = _options(raw_choices)
            if descriptor.value_kind in _DYNAMIC_CHOICE_KINDS or not _static_choices(raw_choices):
                result.setdefault("options", {})[field] = choices
            else:
                source = options.get(_LABEL_SOURCES.get(field))
                if isinstance(choices, dict) and isinstance(source, dict):
                    choices = {
                        value: source.get(value, label) if label == value else label
                        for value, label in choices.items()
                    }
                if isinstance(choices, dict) and "" in choices:
                    entry["empty_label"] = choices.pop("")
                options[field] = choices
                identity = json.dumps(choices, ensure_ascii=False, sort_keys=True)
                if identity in shared:
                    entry["options_from"] = shared[identity]
                else:
                    shared[identity] = field
                    entry["options"] = choices
        section = localize(f"settings.subtab.{descriptor.section}")
        groups.setdefault(section, {})[field] = entry
    return result, groups


def _keys(snapshot: Mapping, module: str) -> dict:
    presence = _mapping(snapshot.get("secret_presence"))
    names = MODULE_KEYS[module]
    verification = _mapping(snapshot.get("provider_verification"))
    result: dict = {"keys": {name: presence[name] for name in names if name in presence}}
    if "secret_presence" not in snapshot:
        result.pop("keys")
    result["verification"] = {
        provider: status
        for provider, setup in PROVIDER_SETUPS.items()
        if setup.secret_key.value in names
        for status in (_mapping(verification.get(provider)).get("status"),)
        if status not in (None, "unknown")
    }
    return result


def _diagnostics(snapshot: Mapping) -> dict:
    coverage = _mapping(snapshot.get("diagnostic_coverage"))
    result: dict = {
        "records": deepcopy(snapshot.get("recent_diagnostics", [])),
        "coverage": {key: coverage[key] for key in _DIAGNOSTIC_COVERAGE if key in coverage},
    }
    if coverage.get("filters"):
        result["coverage"]["filters"] = deepcopy(coverage["filters"])
    if "requested_diagnostic" in snapshot:
        result["requested"] = deepcopy(snapshot["requested_diagnostic"])
    availability = _mapping(snapshot.get("diagnostics_availability"))
    if any(value != "retained_current_run" for value in availability.values()):
        result["availability"] = dict(availability)
    health = _mapping(snapshot.get("diagnostics_health"))
    issues = {
        f"{part}.{key}": value
        for part, keys in _HEALTH_COUNTERS.items()
        for key in keys
        for value in (_mapping(health.get(part)).get(key),)
        if type(value) is int and value > 0
    }
    file = _mapping(health.get("file"))
    if file.get("writer_alive") is False:
        issues["file.writer_alive"] = False
    if file.get("availability") not in (None, "active"):
        issues["file.availability"] = file["availability"]
    if issues:
        result["health_issues"] = issues
    return result


def compact_evidence(value: object) -> object:
    if isinstance(value, Mapping):
        result = {
            key: compact_evidence(item)
            for key, item in value.items()
            if key != "observed_speech_probabilities"
        }
        if isinstance(result.get("candidates"), list):
            fields = {
                "id",
                "current_route",
                "input_host_api",
                "input_device",
                "opened",
                "frames_received",
                "speech_detected",
                "silence_observed",
                "peak_level",
                "speech_probability_min",
                "speech_probability_max",
                "speech_threshold",
                "failure_code",
                "cleanup_complete",
            }
            result["candidates"] = [
                {key: item for key, item in candidate.items() if key in fields}
                for candidate in result["candidates"][:4]
            ]
        if "observed_speech_probabilities" in value:
            result["samples"] = "omitted_from_model"
        return result
    if isinstance(value, list):
        return [compact_evidence(item) for item in value]
    return deepcopy(value)


def operation_header(operation: Mapping) -> dict:
    result = _mapping(operation.get("result"))
    receipt = _mapping(result.get("receipt", operation.get("receipt")))
    return {
        **{
            key: operation[key]
            for key in ("action_id", "tool", "phase", "changed")
            if key in operation
        },
        **{key: result[key] for key in ("status", "changes", "code", "error") if key in result},
        **{
            key: compact_evidence(receipt[key])
            for key in (
                "instance_id",
                "operation_id",
                "request_id",
                "status",
                "terminal",
                "revision",
                "diagnostic_id",
                "reason",
                "effect",
                "error",
                "saved",
                "runtime_ready",
                "settings_changed",
                "cleanup_complete",
                "cleanup_outcome",
                "diagnosis",
                "recognition_result",
                "recognition_scope",
                "translation_result",
                "configured_provider",
                "exercised_provider",
                "output_delivery",
                "display",
                "runtime_observation",
                "transaction",
            )
            if key in receipt
        },
    }


def _core(snapshot: Mapping) -> dict:
    activity = _mapping(snapshot.get("activity_events", snapshot.get("last_activity")))
    result: dict = {
        "revision": snapshot.get("revision"),
        "app": deepcopy(snapshot.get("app")),
        "channels": deepcopy(snapshot.get("channels")),
        "activity": {
            "availability": snapshot.get("activity_availability"),
            **deepcopy(dict(activity)),
        },
        "recent_actions": [
            operation_header(item) for item in snapshot.get("recent_agent_actions", [])
        ],
        "operations": [operation_header(item) for item in snapshot.get("agent_operations", [])],
        "pending_confirmation": deepcopy(snapshot.get("pending_confirmation")),
        "support_reports": deepcopy(snapshot.get("support_reports", [])),
        "puri_allowance": deepcopy(snapshot.get("puri_allowance")),
    }
    if "recent_diagnostics" in snapshot:
        result["diagnostics"] = _diagnostics(snapshot)
    return result


def _module(snapshot: Mapping, module: str, localize: Callable[[str], str]) -> tuple[dict, dict]:
    if module == "core":
        return _core(snapshot), {}
    if module == "accounts":
        accounts = _mapping(snapshot.get("accounts"))
        return {
            **{key: deepcopy(accounts[key]) for key in ("chatgpt", "managed") if key in accounts},
            "actions": [
                action["action_id"]
                for action in accounts.get("actions", ())
                if isinstance(action, Mapping) and "action_id" in action
            ],
        }, {}
    result, catalog = _settings(snapshot, MODULE_FIELDS[module], localize)
    if module in MODULE_KEYS:
        result.update(_keys(snapshot, module))
    if module == "audio" and isinstance(snapshot.get("audio"), Mapping):
        result["devices"] = {
            key: deepcopy(value)
            for key, value in snapshot["audio"].items()
            if key != "current_route"
        }
    if module == "audio" and snapshot.get("audio_test"):
        result["audio_test"] = deepcopy(snapshot["audio_test"])
    if module == "output" and isinstance(snapshot.get("output"), Mapping):
        result.update(deepcopy(dict(snapshot["output"])))
    return result, catalog


def ordered_modules(modules: Iterable[str]) -> tuple[str, ...]:
    selected = set(modules) | {"core"}
    if selected - set(MODULES):
        raise ValueError("unsupported context module")
    return tuple(module for module in MODULES if module in selected)


def provider_modules(provider: str) -> tuple[str, ...]:
    setup = PROVIDER_SETUPS.get(provider)
    if setup is None:
        return MODULES
    return ordered_modules(
        module for module, keys in MODULE_KEYS.items() if setup.secret_key.value in keys
    )


def confirmation_modules(card: Mapping) -> tuple[str, ...]:
    command = card.get("command")
    if command == "auth.logout":
        return ("core", "translation", "accounts")
    if command == "overlay.position.reset":
        return ("core", "output")
    if command == "secrets.delete":
        key = _mapping(card.get("arguments")).get("name")
        selected = [module for module, keys in MODULE_KEYS.items() if key in keys]
        return ordered_modules(selected) if selected else MODULES
    changes = _mapping(card.get("changes"))
    fields = {
        f"languages.{name}" if field == "languages" else field
        for field, value in changes.items()
        for name in (_mapping(value) if field == "languages" else (field,))
    }
    known = {field for names in MODULE_FIELDS.values() for field in names}
    if not fields or fields - known:
        return MODULES
    return ordered_modules(
        module for module, names in MODULE_FIELDS.items() if fields.intersection(names)
    )


def model_context(
    snapshot: Mapping,
    topic: str = "overview",
    localize: Callable[[str], str] = lambda key: key,
    *,
    modules: tuple[str, ...] | None = None,
) -> dict:
    if modules is None:
        modules = topic_modules(topic)
    elif topic != "overview":
        raise ValueError("module selection requires overview")
    else:
        modules = ordered_modules(modules)
    result: dict = {
        "observed_at": snapshot.get("observed_at"),
        **(
            {"observed_monotonic": snapshot["observed_monotonic"]}
            if "observed_monotonic" in snapshot
            else {}
        ),
        "modules": list(modules),
        "missing": list(snapshot.get("missing", [])),
    }
    if topic == "diagnostics":
        return {
            **result,
            "modules": [],
            "view": "diagnostics",
            "diagnostics": _diagnostics(snapshot),
        }
    for module in modules:
        result[module], catalog = _module(snapshot, module, localize)
        if catalog:
            result.setdefault("catalogs", {})[module] = catalog
    return result


def model_receipt(result: Mapping) -> dict:
    context = result.get("context")
    if not isinstance(context, Mapping) or not isinstance(context.get("settings"), Mapping):
        return compact_evidence(result)
    return {
        **compact_evidence(result),
        "context": {
            **context,
            "settings": setting_values(context["settings"]),
        },
    }
