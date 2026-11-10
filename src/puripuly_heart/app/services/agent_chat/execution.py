from __future__ import annotations

import math
from collections.abc import Mapping

from puripuly_heart.app.control_presentation import SETTINGS_PRESENTATION, setting_catalog
from puripuly_heart.app.services.application_audio_devices import safe_audio_device_label
from puripuly_heart.app.services.application_control import COMMANDS, QUERIES, SETTINGS_FIELDS
from puripuly_heart.core.diagnostic_validation import contains_known_secret_pattern
from puripuly_heart.core.diagnostics import sanitize_diagnostic_fields

DEDICATED_COMMANDS = frozenset(
    {
        "capture.set",
        "translation.set",
        "translation.probe",
        "recognition.probe",
        "audio.diagnose",
        "osc.discovery.retry",
        "settings.apply",
        "provider.apply",
    }
)
DIRECT_COMMANDS = frozenset({"overlay.set", "audio.retry", "models.retry"})
CARD_COMMANDS = frozenset({"secrets.delete", "auth.logout", "overlay.position.reset"})
CLI_ONLY_COMMANDS = frozenset(
    {
        "app.stop",
        "audio.test",
        "audio.target.set",
        "models.install",
        "models.prepare",
        "models.cancel",
        "models.download",
        "models.import",
        "models.delete",
        "gemma.prepare",
        "gemma.cancel",
        "gpu.discover",
        "secrets.set",
        "secrets.verify",
        "auth.login",
        "text.submit",
        "overlay.lock",
        "overlay.size",
        "overlay.calibrate",
    }
)
CLI_ONLY_QUERIES = frozenset({"audio.processes", "audio.target"})
QUERY_DOMAINS = {
    "app.status": "app",
    "settings.current": "settings",
    "settings.choices": "catalog",
    "capture.channels": "channels",
    "providers.status": "channels",
    "audio.devices": "devices",
    "gpu.status": "gpu",
    "models.status": "models",
    "secrets.presence": "secrets",
    "accounts.status": "accounts",
    "auth.status": "auth",
    "overlay.status": "overlay",
    "osc.status": "osc",
    "consent.peer_translation": "consent",
    "diagnostics.history": "diagnostics",
    "diagnostics.health": "health",
}


def check_inventory() -> None:
    classified = DEDICATED_COMMANDS | DIRECT_COMMANDS | CARD_COMMANDS | CLI_ONLY_COMMANDS
    if classified != COMMANDS.keys() or QUERY_DOMAINS.keys() | CLI_ONLY_QUERIES != QUERIES:
        raise ValueError("agent_control_inventory_changed")
    if not SETTINGS_FIELDS <= SETTINGS_PRESENTATION.keys():
        raise ValueError("agent_settings_inventory_changed")


def private_fields() -> frozenset[str]:
    return frozenset(field for field in SETTINGS_FIELDS if SETTINGS_PRESENTATION[field].local_only)


def scalar(kind: str, choices=None, **bounds) -> dict:
    return {"type": kind, **({"enum": list(choices)} if choices is not None else {}), **bounds}


def object_schema(properties: dict, optional=()) -> dict:
    return {
        "type": "object",
        "properties": {
            key: {"anyOf": [value, {"type": "null"}]} if key in optional else value
            for key, value in properties.items()
        },
        "required": list(properties),
        "additionalProperties": False,
    }


def model_safe_catalog(catalog: dict) -> dict:
    schema_keys = {
        "type",
        "free_form",
        "nullable",
        "choices",
        "items",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "min_items",
        "max_items",
        "unique",
        "empty_allowed",
        "required",
        "properties",
        "key_choices",
        "value_choices_by_key",
    }

    def schema(value):
        if not isinstance(value, Mapping):
            return value
        return {
            key: (
                {name: schema(item) for name, item in value[key].items()}
                if key == "properties"
                else schema(value[key])
            )
            for key in value
            if key in schema_keys
        }

    def options(value):
        if isinstance(value, list):
            return [options(item) for item in value[:128]]
        if isinstance(value, Mapping):
            if "value" in value:
                original = value["value"]
                label = safe_audio_device_label(original)
                result = {
                    "value": label,
                    "selectable": label == original and value.get("selectable") is not False,
                }
                for key in ("label", "display_name", "host_api"):
                    if type(value.get(key)) is str:
                        result[key] = safe_audio_device_label(value[key])
                if type(value.get("available")) is bool:
                    result["available"] = value["available"]
                return result
            return {
                key: options(item)
                for key, item in value.items()
                if type(key) is str and len(key) <= 128 and safe_audio_device_label(key) == key
            }
        if type(value) is str:
            return safe_audio_device_label(value)
        return value if value is None or type(value) in (bool, int, float) else None

    return {
        "fields": [field for field in catalog.get("fields", ()) if field in SETTINGS_FIELDS],
        "field_schemas": {
            field: schema(value)
            for field, value in catalog.get("field_schemas", {}).items()
            if field in SETTINGS_FIELDS
        },
        "choices": {
            field: options(value)
            for field, value in catalog.get("choices", {}).items()
            if field in SETTINGS_FIELDS and field not in private_fields()
        },
    }


def project_catalog(catalog: dict, localize) -> dict:
    return setting_catalog(model_safe_catalog(catalog), localize)


def request_schemas(catalog: dict) -> list[dict]:
    check_inventory()
    s = scalar("string", minLength=1, maxLength=128)
    secret = scalar("string", catalog.get("command_choices", {}).get("secrets.names", ()))
    provider = scalar("string", ("qq", "discord", "openrouter", "chatgpt"))
    specs = {
        "overlay.set": ({"enabled": scalar("boolean")}, ()),
        "audio.retry": ({}, ()),
        "models.retry": ({"backend": scalar("string", ("gpu",))}, ("backend",)),
        "secrets.delete": ({"name": secret}, ()),
        "auth.logout": ({"provider": provider}, ()),
        "overlay.position.reset": ({}, ()),
    }
    for name in QUERY_DOMAINS:
        fields = {}
        if name == "diagnostics.history":
            fields = {
                "operation_id": s,
                "diagnostic_id": s,
                "after": scalar("integer", minimum=0),
                "limit": scalar("integer", minimum=1, maximum=100),
            }
        specs["query." + name] = (fields, tuple(fields))
    return [
        object_schema({"action": scalar("string", (name,)), **fields}, optional)
        for name, (fields, optional) in sorted(specs.items())
    ]


APP_TOOLS = ("read_app", "execute_app", "propose_app")
DETAIL_TOPICS = {
    "gpu": "gpu.status",
    "models": "models.status",
    "auth": "auth.status",
    "peer_consent": "consent.peer_translation",
}


def app_tool_variants(catalog: dict, family: str) -> list[dict]:
    from puripuly_heart.app.services.agent_chat.modules import TOPICS
    from puripuly_heart.app.services.agent_chat.provider_setup import NAVIGATION_PROVIDERS
    from puripuly_heart.app.services.agent_chat.tools import settings_tool_schema

    def sparse(schema):
        if "anyOf" in schema:
            variants = [item for item in schema["anyOf"] if item != {"type": "null"}]
            if len(variants) == 1:
                return sparse(variants[0])
        result = dict(schema)
        if result.get("type") == "object":
            result["properties"] = {
                key: sparse(value) for key, value in result["properties"].items()
            }
            result["required"] = []
        return result

    def variant(key, values, fields=None, optional=()):
        schema = object_schema({key: scalar("string", values), **(fields or {})})
        schema["required"] = [name for name in schema["properties"] if name not in optional]
        return schema

    provider = scalar("string", sorted(NAVIGATION_PROVIDERS))
    identity = scalar("string", minLength=1, maxLength=64)
    if family == "read_app":
        return [
            variant("topic", sorted(TOPICS - {"diagnostics"} | DETAIL_TOPICS.keys())),
            variant(
                "topic",
                ["diagnostics"],
                {
                    "diagnostic_id": identity,
                    "operation_id": identity,
                    "after": scalar("integer", minimum=0),
                },
                ("diagnostic_id", "operation_id", "after"),
            ),
            variant("topic", ["provider_setup"], {"provider": provider}),
            variant("topic", ["operation"], {"operation_id": identity}),
        ]
    if family == "execute_app":
        return [
            variant(
                "action",
                ["capture.set"],
                {"channel": scalar("string", ("self", "peer")), "enabled": scalar("boolean")},
            ),
            variant("action", ["translation.set", "overlay.set"], {"enabled": scalar("boolean")}),
            variant(
                "action", ["recognition.probe"], {"channel": scalar("string", ("self", "peer"))}
            ),
            variant(
                "action",
                [
                    "translation.probe",
                    "audio.diagnose",
                    "audio.retry",
                    "osc.discovery.retry",
                    "ui.accounts",
                ],
            ),
            variant(
                "action", ["models.retry"], {"backend": scalar("string", ("gpu",))}, ("backend",)
            ),
            variant("action", ["ui.provider_setup"], {"provider": provider}),
            variant(
                "action",
                ["ui.settings"],
                {
                    "section": scalar("string", ("api", "general", "overlay", "prompts")),
                    "provider": provider,
                },
                ("provider",),
            ),
            variant("action", ["confirmation.withdraw"], {"card_id": identity}),
        ]
    if family == "propose_app":
        return [
            variant("action", ["settings"], {"changes": sparse(settings_tool_schema(catalog))}),
            variant(
                "action", ["report"], {"summary": scalar("string", minLength=1, maxLength=2000)}
            ),
            *[
                spec
                for spec in request_schemas(catalog)
                if spec["properties"]["action"]["enum"][0] in CARD_COMMANDS
            ],
        ]
    raise ValueError("unsupported_app_tool")


def app_tool_schemas(catalog: dict, *, family: str | None = None) -> dict[str, dict]:
    schemas = {}
    for name in APP_TOOLS if family is None else (family,):
        variants = app_tool_variants(catalog, name)
        key = next(iter(variants[0]["properties"]))
        properties = {
            key: scalar(
                "string", [value for item in variants for value in item["properties"][key]["enum"]]
            )
        }
        for item in variants:
            for field, schema in item["properties"].items():
                if field != key and properties.setdefault(field, schema) != schema:
                    raise ValueError("conflicting_app_tool_field")
        schemas[name] = {
            "type": "object",
            "properties": properties,
            "required": [key],
            "additionalProperties": False,
        }
    return schemas


def validate_app_request(family: str, arguments: dict, catalog: dict) -> dict:
    request = {key: value for key, value in arguments.items() if value is not None}
    key = "topic" if family == "read_app" else "action"
    variants = app_tool_variants(catalog, family)
    if request.keys() - {field for item in variants for field in item["properties"]}:
        raise ValueError("invalid_control_fields")
    variant = next(
        (item for item in variants if request.get(key) in item["properties"][key]["enum"]),
        None,
    )
    if variant is None:
        raise ValueError("invalid_control_choice")
    request = {field: value for field, value in request.items() if field in variant["properties"]}
    validate(request, variant)
    return request


def app_tool_target(name: str, arguments: dict) -> tuple[str, dict]:
    request = dict(arguments)
    if name == "read_app":
        topic = request.pop("topic")
        if topic == "provider_setup":
            return "get_provider_setup", request
        if topic == "operation":
            return "read_operation", request
        if topic in DETAIL_TOPICS:
            if request:
                raise ValueError("unsupported_read_filters")
            return "control_app", {"request": {"action": "query." + DETAIL_TOPICS[topic]}}
        return "get_app_context", {"topic": topic, **request}
    action = request.pop("action")
    if name == "propose_app":
        if action == "settings":
            return "change_settings", request
        if action == "report":
            return "propose_report", request
        return "control_app", {"request": {"action": action, **request}}
    targets = {
        "capture.set": "set_capture",
        "translation.set": "set_translation",
        "audio.diagnose": "diagnose_microphone",
        "osc.discovery.retry": "retry_osc_discovery",
        "translation.probe": "probe_translation",
        "ui.provider_setup": "get_provider_setup",
        "ui.settings": "open_settings",
        "confirmation.withdraw": "withdraw_confirmation",
    }
    if action == "recognition.probe":
        return f"probe_{request['channel']}_recognition", {}
    if action == "ui.accounts":
        return "get_app_context", {"topic": "accounts"}
    if action in targets:
        return targets[action], request
    return "control_app", {"request": {"action": action, **request}}


def validate(value, schema: dict, *, partial: bool = False) -> None:
    if "anyOf" in schema:
        for variant in schema["anyOf"]:
            try:
                validate(value, variant, partial=partial)
                return
            except ValueError:
                pass
        raise ValueError("invalid_control_request")
    kind = schema.get("type")
    if not {
        "null": value is None,
        "string": type(value) is str,
        "boolean": type(value) is bool,
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "object": type(value) is dict,
        "array": type(value) is list,
    }.get(kind, False):
        raise ValueError("invalid_control_type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("invalid_control_choice")
    if kind == "string" and (
        not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 4096)
        or contains_known_secret_pattern(value)
    ):
        raise ValueError("invalid_control_text")
    if kind in ("number", "integer"):
        if (
            not math.isfinite(value)
            or value < schema.get("minimum", -math.inf)
            or value > schema.get("maximum", math.inf)
            or value <= schema.get("exclusiveMinimum", -math.inf)
        ):
            raise ValueError("invalid_control_range")
    if kind == "object":
        fields = schema["properties"]
        required = {
            key
            for key in schema.get("required", ())
            if {"type": "null"} not in fields[key].get("anyOf", ())
        }
        if value.keys() - fields.keys() or (not partial and required - value.keys()):
            raise ValueError("invalid_control_fields")
        for key, item in value.items():
            validate(item, fields[key], partial=partial)
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100):
            raise ValueError("invalid_control_length")
        for item in value:
            validate(item, schema["items"])
        if schema.get("uniqueItems") and len(set(value)) != len(value):
            raise ValueError("duplicate_control_values")


def safe_fields(value: object, fields) -> dict:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key in fields
        and (
            item is None
            or type(item) in (bool, int, float)
            or type(item) is str
            and len(item) <= 128
            and safe_audio_device_label(item) == item
        )
    }


RECEIPT_FIELDS = frozenset(
    {
        "instance_id",
        "operation_id",
        "request_id",
        "status",
        "terminal",
        "revision",
        "code",
        "current_revision",
        "diagnostic_id",
        "effect",
        "reason",
        "saved",
        "runtime_ready",
        "translation_enabled",
        "settings_changed",
        "cancel_requested",
        "cleanup_complete",
        "cleanup_outcome",
        "translation_result",
        "recognition_result",
        "output_delivery",
        "turn_id",
        "admitted",
        "model_id",
        "backend",
        "verified",
        "present",
    }
)


def safe_receipt(value: object) -> dict:
    source = value if isinstance(value, Mapping) else {}

    def project(item):
        result = safe_fields(item, RECEIPT_FIELDS | {"category", "disposition", "verification"})
        for key in ("reason", "code", "effect", "status", "verification"):
            text = result.get(key)
            if type(text) is str and (
                not text.isascii()
                or not text.replace("_", "").replace(".", "").isalnum()
                or text != text.lower()
            ):
                del result[key]
        return result

    result = project(source)
    for key in ("error", "diagnostics", "transaction", "runtime_observation"):
        if isinstance(source.get(key), Mapping):
            result[key] = project(source[key])
    return result


async def project_query(name: str, source: dict, control, localize) -> dict:
    from puripuly_heart.app.services.agent_chat.context import (
        CHANNEL_FIELDS,
        OVERLAY_FIELDS,
        _audio_devices,
    )
    from puripuly_heart.app.services.agent_chat.tools import (
        project_accounts,
        project_osc_output,
        project_settings,
    )

    base = safe_fields(source, ("instance_id", "revision", "status", "availability"))
    domain = QUERY_DOMAINS[name]
    if domain == "settings":
        return {**base, **project_settings(source)}
    if domain == "catalog":
        return {**base, "settings_catalog": project_catalog(source, localize)}
    if domain == "accounts":
        return {**base, **project_accounts(source)}
    if domain == "osc":
        return {
            **base,
            "output": project_osc_output(source.get("output")),
            "ports": safe_fields(source.get("ports"), ("send_port", "receive_port")),
        }
    if domain == "overlay":
        return {
            **base,
            **safe_fields(source, ("target", "locked")),
            "output": safe_fields(source.get("output"), OVERLAY_FIELDS),
        }
    if domain == "devices":
        return {**base, **_audio_devices(source, {})}
    if domain in {"channels", "app"}:
        result = {
            **base,
            **safe_fields(
                source,
                ("translation_enabled", "translation_runtime_ready", "overlay_target", "stopping"),
            ),
        }
        result["channels"] = {
            channel: {
                **safe_fields(source.get("channels", {}).get(channel), CHANNEL_FIELDS),
                "capture": safe_fields(
                    source.get("channels", {}).get(channel, {}).get("capture"),
                    ("state", "generation", "desired_active", "effective_active", "has_source"),
                ),
            }
            for channel in ("self", "peer")
        }
        result["audio_test"] = safe_fields(
            source.get("audio_test"),
            ("state", "ready", "active", "failure_code", "cleanup_complete"),
        )
        return result
    if domain == "secrets":
        names = set(control.capabilities().get("secret_keys", ()))
        return {
            **base,
            "presence": {
                key: value
                for key, value in source.get(
                    "effective_presence", source.get("presence", {})
                ).items()
                if key in names and type(value) is bool
            },
        }
    if domain == "auth":
        return {
            **base,
            **safe_fields(source, ("action", "prompt_kind", "failure_kind", "consent_accepted")),
            "chatgpt": safe_fields(
                source.get("chatgpt"), ("available", "signed_in", "in_progress", "sign_in_required")
            ),
            "login_available_locally": True,
        }
    if domain == "consent":
        return {
            **base,
            "accepted": source.get("accepted") is True,
            "acceptance": "local_informed_click_only",
        }
    if domain in {"gpu", "models"}:
        fields = {
            "state",
            "status",
            "available",
            "ready",
            "active",
            "model_id",
            "id",
            "backend",
            "device_id",
            "label",
            "name",
            "display_name",
            "description",
            "failure_code",
            "progress_percent",
            "size_bytes",
            "purpose",
            "importable",
            "installed",
            "valid",
            "integrity",
            "operation",
            "origin",
            "generation",
            "revision",
            "closed",
            "runtime_ready",
            "provider",
            "phase",
            "in_progress",
            "cancel_requested",
            "ui_state",
            "discovery_attempted",
            "discovery_failed",
            "discovery_failure_state",
            "discovery_origin",
            "pending_count",
            "configured_device_id",
            "model_resident",
            "retry_required",
            "availability",
            "pending_startups",
            "close_complete",
            "pending_temporary_directories",
            "temporary_cleanup_diagnostic_id",
            "detail_rejected",
            "truncated",
            "gpu_model_id",
        }

        def project(item, depth=0):
            if depth > 5:
                return None
            if isinstance(item, list):
                return [project(entry, depth + 1) for entry in item[:100]]
            if isinstance(item, Mapping):
                return {
                    **safe_fields(item, fields),
                    **{
                        key: project(value, depth + 1)
                        for key, value in item.items()
                        if key
                        in {
                            "state",
                            "runtime",
                            "gemma",
                            "assets",
                            "models",
                            "devices",
                            "required_files",
                            "required_cpu_model_ids",
                            "activities",
                            "active_channels",
                            "pending_channels",
                            "startup_cleanup",
                            "cleanup_failures",
                        }
                        and isinstance(value, (Mapping, list))
                    },
                }
            if type(item) is str and len(item) <= 128 and safe_audio_device_label(item) == item:
                return item
            return None

        return {**base, **project(source)}
    if domain == "diagnostics":
        return {
            **base,
            **safe_fields(
                source,
                (
                    "lookup",
                    "oldest_sequence",
                    "latest_sequence",
                    "next_after",
                    "gap",
                    "evicted_records",
                    "truncated",
                ),
            ),
            "records": [project_event(item) for item in source.get("records", [])[:100]],
        }
    if domain == "health":
        fields = {
            "availability",
            "retained_records",
            "evicted_records",
            "rejected_records",
            "subscriber_delivery_failures",
            "file_projection_failures",
            "durable",
            "restart_resets_history",
            "delivery_failures",
            "dropped_records",
            "writer_alive",
            "active_subscribers",
            "retained_events",
            "history_capacity",
            "latest_sequence",
            "oldest_sequence",
            "evicted_events",
            "subscriber_gaps",
            "disconnected_consumers",
            "owner_history",
            "owner_history_capacity",
            "owner_history_evictions",
            "observer",
            "observer_failure_count",
            "observer_failure_diagnostic_id",
            "last_observer_failure_type",
            "subscriber_dispatch_pending",
            "pending_records",
            "written_records",
            "pending_age_ms",
            "cleanup_failures",
        }
        return {
            **base,
            **{
                key: safe_fields(source.get(key), fields)
                for key in ("history", "file", "events", "output")
            },
        }
    raise AssertionError(name)


def project_event(source: dict, *, depth: int = 0) -> dict:
    fields = {
        "topic",
        "instance_id",
        "sequence",
        "diagnostic_id",
        "diagnostic_sequence",
        "occurred_at",
        "component",
        "operation",
        "stage",
        "severity",
        "disposition",
        "code",
        "category",
        "operation_id",
        "request_id",
        "channel",
        "generation",
        "state",
        "status",
        "terminal",
        "revision",
        "level",
        "event",
        "gap",
        "dropped",
        "oldest_sequence",
        "latest_sequence",
        "next_after",
        "content_available",
    }
    fields |= {
        "kind",
        "provider",
        "utterance_id",
        "turn_id",
        "snapshot_required",
        "after",
        "reason",
    }
    result = safe_fields(source, fields)
    if isinstance(source.get("fields"), Mapping):
        result["fields"] = sanitize_diagnostic_fields(source["fields"])[0]
    if isinstance(source.get("receipt"), Mapping):
        result["receipt"] = safe_receipt(source["receipt"])
    if depth == 0 and isinstance(source.get("diagnostic"), Mapping):
        result["diagnostic"] = project_event(source["diagnostic"], depth=1)
    return result
