from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from puripuly_heart.app.control_presentation import SETTINGS_PRESENTATION, setting_catalog
from puripuly_heart.app.services.application_audio_devices import safe_audio_device_label
from puripuly_heart.app.services.application_control import COMMANDS, QUERIES, SETTINGS_FIELDS
from puripuly_heart.core.diagnostic_validation import contains_known_secret_pattern
from puripuly_heart.core.diagnostics import (
    DIAGNOSTIC_CAUSE_MAX_COUNT,
    project_diagnostic_record,
    sanitize_diagnostic_fields,
)

CHANNEL_FIELDS = (
    "selected_provider",
    "runtime_provider",
    "capture_attached_provider",
    "desired_active",
    "effective_active",
    "pending_handoff",
    "failure_reason",
    "provider_status",
)
OVERLAY_FIELDS = (
    "settings_available",
    "desired_enabled",
    "configured_target",
    "effective_target",
    "attempting_target",
    "lifecycle",
    "runtime_active",
    "process_state",
    "presentation_ready",
    "desktop_visible",
    "failure_reason",
    "auto_restart_scheduled",
    "fallback_active",
    "recovery_active",
    "recovery_reason",
    "ingress_stopped",
    "generation",
)

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
DIRECT_COMMANDS = frozenset(
    {"overlay.set", "audio.retry", "models.retry", "accounts.managed.refresh"}
)
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


APP_TOOLS = ("read_app", "set_app", "probe_app", "retry_app", "open_app", "manage_proposal")
DETAIL_TOPICS = {
    "gpu": "gpu.status",
    "models": "models.status",
    "auth": "auth.status",
    "peer_consent": "consent.peer_translation",
}
TOOL_PURPOSES = {
    "read_app": "Observe current app state without UI or runtime effects.",
    "set_app": "Set an existing runtime switch; no saved settings change.",
    "probe_app": "Run an isolated scoped check, not live pipeline repair or output delivery.",
    "retry_app": "Retry existing application recovery through its resource owner.",
    "open_app": "Stage existing app controls; staging is not display, click, verification or readiness.",
    "manage_proposal": "Create or withdraw a user card, never approve or send it.",
}


@dataclass(frozen=True, slots=True)
class AppAction:
    tool: str
    key: str
    value: str
    handler: str
    binding: dict
    fields: dict
    optional: tuple[str, ...] = ()
    guidance: str = ""
    settings: bool = False

    def schema(self, catalog: dict) -> dict:
        fields = self.fields
        if self.settings:
            from puripuly_heart.app.services.agent_chat.tools import settings_tool_schema

            fields = {"changes": settings_tool_schema(catalog)}
        return {
            "type": "object",
            "properties": {self.key: scalar("string", (self.value,)), **fields},
            "required": [self.key, *(key for key in fields if key not in self.optional)],
            "additionalProperties": False,
        }

    def arguments(self, request: dict) -> dict:
        return {**self.binding, **{key: value for key, value in request.items() if key != self.key}}

    @property
    def signature(self) -> str:
        fields = {"changes": {}} if self.settings else self.fields
        arguments = ", ".join(f"{name}{'?' if name in self.optional else ''}" for name in fields)
        return f"{self.value}{'(' + arguments + ')' if arguments else ''}"


def app_actions() -> tuple[AppAction, ...]:
    from puripuly_heart.app.ports.settings_secrets import SettingsSecretKey
    from puripuly_heart.app.services.agent_chat.modules import TOPICS
    from puripuly_heart.app.services.agent_chat.provider_setup import NAVIGATION_PROVIDERS

    provider = scalar("string", sorted(NAVIGATION_PROVIDERS))
    identity = scalar("string", minLength=1, maxLength=64)
    actions = []

    def add(
        tool, value, handler, binding=None, fields=None, optional=(), guidance="", settings=False
    ):
        key = {"read_app": "topic", "open_app": "view", "manage_proposal": "action"}.get(
            tool, "target"
        )
        actions.append(
            AppAction(
                tool, key, value, handler, binding or {}, fields or {}, optional, guidance, settings
            )
        )

    for topic in sorted(TOPICS - {"diagnostics"}):
        add("read_app", topic, "context", {"topic": topic})
    add(
        "read_app",
        "diagnostics",
        "context",
        {"topic": "diagnostics"},
        {
            "diagnostic_id": identity,
            "operation_id": identity,
            "after": scalar("integer", minimum=0),
        },
        ("diagnostic_id", "operation_id", "after"),
    )
    for topic, query in DETAIL_TOPICS.items():
        add("read_app", topic, "query", {"query": query})
    add("read_app", "provider_setup", "provider_setup", fields={"provider": provider})
    add(
        "read_app",
        "operation",
        "operation",
        fields={"operation_id": identity},
        guidance="Retained receipt; resolve pending/uncertain work by its identity, not resubmission.",
    )
    for channel in ("self", "peer"):
        add(
            "set_app",
            f"capture.{channel}",
            "command",
            {"command": "capture.set", "channel": channel},
            {"enabled": scalar("boolean")},
        )
    for target in ("translation", "overlay"):
        add(
            "set_app",
            target,
            "command",
            {"command": f"{target}.set"},
            {"enabled": scalar("boolean")},
        )
    for kind in ("recognition", "translation"):
        for channel in ("self", "peer"):
            add(
                "probe_app",
                f"{kind}.{channel}",
                "command",
                {"command": f"{kind}.probe", "channel": channel},
            )
    add(
        "probe_app",
        "microphone",
        "command",
        {"command": "audio.diagnose"},
        guidance="Local capture/VAD only; no settings change.",
    )
    add(
        "probe_app",
        "managed_access",
        "command",
        {"command": "accounts.managed.refresh"},
        guidance="Whether the selected managed credential is still usable; no translation request",
    )
    for target, command, binding in (
        ("audio", "audio.retry", {}),
        ("osc_discovery", "osc.discovery.retry", {}),
        ("gpu_activation", "models.retry", {"backend": "gpu"}),
    ):
        add("retry_app", target, "command", {"command": command, **binding})
    add("open_app", "accounts", "context", {"topic": "accounts"})
    add("open_app", "provider_setup", "provider_setup", fields={"provider": provider})
    add(
        "open_app",
        "settings",
        "navigation",
        fields={
            "section": scalar("string", ("api", "general", "overlay", "prompts")),
            "provider": provider,
        },
        optional=("provider",),
    )
    add(
        "manage_proposal",
        "settings",
        "settings",
        settings=True,
        guidance="Sparse changes only; absent fields stay unchanged, never null placeholders.",
    )
    add(
        "manage_proposal",
        "report",
        "report",
        fields={"summary": scalar("string", minLength=1, maxLength=2000)},
        guidance="Offers a send button; upload requires the user's click.",
    )
    add(
        "manage_proposal",
        "auth.logout",
        "card",
        {"command": "auth.logout"},
        {"provider": scalar("string", ("qq", "discord", "openrouter", "chatgpt"))},
    )
    add("manage_proposal", "overlay.position.reset", "card", {"command": "overlay.position.reset"})
    add(
        "manage_proposal",
        "secrets.delete",
        "card",
        {"command": "secrets.delete"},
        {"name": scalar("string", tuple(key.value for key in SettingsSecretKey))},
    )
    add(
        "manage_proposal",
        "withdraw",
        "withdraw",
        fields={"card_id": identity},
        guidance="Only the matching pending card; accepted application work is not cancelled.",
    )
    return tuple(actions)


def select_app_action(tool: str, arguments: dict) -> AppAction:
    for action in app_actions():
        if action.tool == tool and arguments.get(action.key) == action.value:
            return action
    raise ValueError("invalid_control_choice")


def app_tool_variants(catalog: dict, family: str) -> list[dict]:
    return [action.schema(catalog) for action in app_actions() if action.tool == family]


def app_tool_schemas(catalog: dict, *, family: str | None = None) -> dict[str, dict]:
    check_inventory()
    schemas = {}
    for name in APP_TOOLS if family is None else (family,):
        variants = app_tool_variants(catalog, name)
        key = next(iter(variants[0]["properties"]))
        properties = {
            key: scalar("string", [item["properties"][key]["enum"][0] for item in variants])
        }
        for item in variants:
            for field, schema in item["properties"].items():
                if field != key and properties.setdefault(field, schema) != schema:
                    raise ValueError("conflicting_app_tool_field")
        schemas[name] = {
            "type": "object",
            "properties": properties,
            "required": [
                field for field in properties if all(field in item["required"] for item in variants)
            ],
            "additionalProperties": False,
        }
    return schemas


def app_tool_descriptions() -> dict[str, str]:
    return {
        tool: TOOL_PURPOSES[tool]
        + " Arguments: "
        + "; ".join(
            action.signature + (". " + action.guidance.rstrip(".") if action.guidance else "")
            for action in app_actions()
            if action.tool == tool
        )
        + ". Supply only the selected action's fields."
        for tool in APP_TOOLS
    }


def validate_app_request(family: str, arguments: dict, catalog: dict) -> dict:
    action = select_app_action(family, arguments)
    validate(arguments, action.schema(catalog))
    return dict(arguments)


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


def project_diagnostic_evidence(value: object) -> dict:
    source = value if isinstance(value, Mapping) else {}
    result = {
        key: project_diagnostic_record(source[key])
        for key in (
            "diagnostics",
            "cause_diagnostics",
            "failure_diagnostics",
            "query_diagnostics",
            "discovery_diagnostics",
            "avatar_diagnostics",
            "source_open_diagnostics",
        )
        if isinstance(source.get(key), Mapping)
    }
    for key in ("secondary_diagnostics_truncated", "diagnostics_truncated"):
        if type(source.get(key)) is bool:
            result[key] = source[key]
    for key, bound, count_key, flag_key in (
        ("secondary_diagnostics", 4, "secondary_omitted_count", "secondary_truncated"),
        (
            "recovered_diagnostics",
            DIAGNOSTIC_CAUSE_MAX_COUNT,
            "recovered_diagnostics_omitted_count",
            "recovered_failures_truncated",
        ),
        (
            "recovery_diagnostics",
            DIAGNOSTIC_CAUSE_MAX_COUNT,
            "recovery_diagnostics_omitted_count",
            "recovery_diagnostics_truncated",
        ),
        (
            "runtime_failures",
            DIAGNOSTIC_CAUSE_MAX_COUNT,
            "runtime_failures_omitted_count",
            "runtime_failures_truncated",
        ),
        (
            "service_failures",
            DIAGNOSTIC_CAUSE_MAX_COUNT,
            "service_failures_omitted_count",
            "service_failures_truncated",
        ),
        (
            "dispatch_failures",
            DIAGNOSTIC_CAUSE_MAX_COUNT,
            "dispatch_failures_omitted_count",
            "dispatch_failures_truncated",
        ),
    ):
        if type(source.get(flag_key)) is bool:
            result[flag_key] = source[flag_key]
        omitted = source.get(count_key)
        if type(omitted) is int and 0 <= omitted < 2**31:
            result[count_key] = omitted
        items = source.get(key)
        if isinstance(items, (list, tuple)):
            result[key] = [
                project_diagnostic_record(item)
                for item in items[:bound]
                if isinstance(item, Mapping)
            ]
            if len(items) > bound:
                result[count_key] = result.get(count_key, 0) + len(items) - bound
                result[flag_key] = True
    return result


def project_capture_channel(value: object) -> dict:
    source = value if isinstance(value, Mapping) else {}
    capture = source.get("capture")
    capture = capture if isinstance(capture, Mapping) else {}
    result = {
        **safe_fields(source, CHANNEL_FIELDS),
        **project_diagnostic_evidence(source),
        "capture": {
            **safe_fields(
                capture,
                (
                    "state",
                    "generation",
                    "desired_active",
                    "effective_active",
                    "has_source",
                    "cleanup_debt",
                    "closed",
                    "source_open_recovered",
                ),
            ),
            **project_diagnostic_evidence(capture),
        },
    }
    provider = project_diagnostic_evidence(source.get("provider"))
    if provider:
        result["provider"] = provider
    return result


def project_overlay_output(value: object) -> dict:
    return {
        **safe_fields(value, OVERLAY_FIELDS),
        **project_diagnostic_evidence(value),
    }


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
    result.update(project_diagnostic_evidence(source))
    for key in ("error", "runtime_observation"):
        if isinstance(source.get(key), Mapping):
            result[key] = project(source[key])
    observation = source.get("runtime_observation")
    if isinstance(observation, Mapping) and isinstance(observation.get("channels"), Mapping):
        result["runtime_observation"]["channels"] = {
            channel: project_capture_channel(value)
            for channel, value in observation["channels"].items()
            if channel in {"self", "peer"}
        }
    if isinstance(source.get("channel"), Mapping):
        result["channel"] = project_capture_channel(source["channel"])
    if isinstance(source.get("output"), Mapping):
        result["output"] = project_diagnostic_evidence(source["output"])
    if isinstance(source.get("transaction"), Mapping):
        result["transaction"] = safe_receipt(source["transaction"])
    if isinstance(source.get("audio_test"), Mapping):
        result["audio_test"] = project_audio_test(source["audio_test"])
    return result


def project_audio_test(source: Mapping) -> dict:
    result = safe_fields(source, ("state", "desired_active", "effective_active"))
    result["channels"] = {
        channel: {
            **safe_fields(
                value, ("state", "failure_reason", "failure_type", "device_metadata_availability")
            ),
            **safe_receipt(value),
        }
        for channel, value in source.get("channels", {}).items()
        if channel in {"self", "peer"} and isinstance(value, Mapping)
    }
    return result


async def project_query(name: str, source: dict, control, localize) -> dict:
    from puripuly_heart.app.services.agent_chat.context import _audio_devices
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
            "output": project_overlay_output(source.get("output")),
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
            channel: project_capture_channel(source.get("channels", {}).get(channel))
            for channel in ("self", "peer")
        }
        result["audio_test"] = project_audio_test(
            source["audio_test"] if isinstance(source.get("audio_test"), Mapping) else {}
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
            **safe_receipt(source),
            **safe_fields(source, ("action", "prompt_kind", "failure_kind", "consent_accepted")),
            "chatgpt": {
                **safe_fields(
                    source.get("chatgpt"),
                    ("available", "signed_in", "in_progress", "sign_in_required"),
                ),
                **safe_receipt(source.get("chatgpt")),
            },
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
                    **safe_receipt(item),
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
            "records": [
                project_diagnostic_record(item)
                for item in source.get("records", [])[:100]
                if isinstance(item, Mapping)
            ],
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
        result["diagnostic"] = project_diagnostic_record(source["diagnostic"])
    if isinstance(source.get("error"), Mapping):
        result["error"] = safe_receipt({"error": source["error"]})["error"]
        if isinstance(source["error"].get("diagnostics"), Mapping):
            result["error"]["diagnostics"] = project_diagnostic_record(
                source["error"]["diagnostics"]
            )
    return result
