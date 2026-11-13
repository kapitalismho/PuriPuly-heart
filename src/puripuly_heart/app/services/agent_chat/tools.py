from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from uuid import uuid4

from puripuly_heart.app.control_presentation import (
    COMMAND_CHANGES,
    SETTINGS_PRESENTATION,
    command_field,
)
from puripuly_heart.app.ports.application_control import ApplicationControl
from puripuly_heart.app.ports.settings_secrets import SettingsSecretKey
from puripuly_heart.app.services.agent_chat.execution import (
    AppAction,
    private_fields,
    project_catalog,
    project_query,
    safe_receipt,
    select_app_action,
    validate,
)
from puripuly_heart.app.services.agent_chat.provider_setup import (
    NAVIGATION_PROVIDERS,
    PROVIDER_SETUPS,
    RECOGNITION_ALLOWANCES,
)
from puripuly_heart.app.services.application_audio_devices import safe_audio_device_label
from puripuly_heart.config.audio_host_api import normalize_input_host_api
from puripuly_heart.config.provider_values import STTProviderName, is_custom_stt_provider
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    provider_llm_for_translation,
)
from puripuly_heart.core.lifecycle import LifecycleScope

SETTINGS_FIELD_PATHS = {
    field: descriptor.canonical_path
    for field, descriptor in SETTINGS_PRESENTATION.items()
    if not descriptor.local_only
    and descriptor.canonical_path is not None
    and field not in {"audio", "translation", "languages"}
    and not field.startswith("languages.")
}
LANGUAGE_FIELD_PATHS = {
    field.removeprefix("languages."): descriptor.canonical_path
    for field, descriptor in SETTINGS_PRESENTATION.items()
    if field.startswith("languages.")
}
SETTINGS_FIELD_PATHS["osc.connection"] = "osc.connection"
SETTINGS_FIELDS = frozenset((*SETTINGS_FIELD_PATHS, "languages"))
MAX_APPROVAL_BLOCKS = 2
_COMMAND_EFFECTS = {
    "secrets.delete": "Removes only the locally stored credential; the provider-side key is not revoked.",
    "auth.logout": "Removes this app's stored authorization; only ChatGPT also requests remote revocation.",
    "overlay.position.reset": "Clears the saved desktop caption position and unlocks the window.",
}


def _approval_groups(changes: dict) -> list[dict]:
    blocks = []
    coupled = set()
    for key, value in changes.items():
        pair = next(
            (
                pair
                for pair in (
                    ("translation.model", "translation.connection"),
                    ("audio.input_host_api", "audio.input_device"),
                )
                if key in pair
            ),
            None,
        )
        if pair is not None:
            if pair not in coupled:
                coupled.add(pair)
                blocks.append({field: changes[field] for field in pair if field in changes})
        elif key == "languages":
            blocks.extend({"languages": {field: item}} for field, item in value.items())
        else:
            blocks.append({key: value})
    groups = []
    for offset in range(0, len(blocks), MAX_APPROVAL_BLOCKS):
        group = {}
        for block in blocks[offset : offset + MAX_APPROVAL_BLOCKS]:
            for key, value in block.items():
                if key == "languages":
                    group.setdefault(key, {}).update(value)
                else:
                    group[key] = value
        groups.append(group)
    return groups


NAVIGATION_SECTIONS = ("api", "general", "overlay", "prompts")
_OSC_ENUMS = {
    "configured_mode": frozenset({"automatic", "manual", "off", None}),
    "effective_mode": frozenset({"automatic", "manual", "off"}),
    "query_failure": frozenset({"oscquery_unavailable", None}),
    "local_availability": frozenset(
        {"closed", "off", "receiver_unavailable", "sender_unavailable", "starting", "available"}
    ),
    "destination_source": frozenset({"discovered", "manual", "fallback", "off"}),
}
_AUDIO_ROUTE_FIELDS = frozenset({"audio.input_host_api", "audio.input_device"})
_DYNAMIC_CHOICE_FIELDS = _AUDIO_ROUTE_FIELDS | {
    "audio.output_device",
    "stt.gpu_device_id",
    "translation.gpu_device_id",
    "translation.http_extension_id",
}
_STT_SECRET_KEYS = {
    STTProviderName.DEEPGRAM.value: SettingsSecretKey.DEEPGRAM_API_KEY.value,
    STTProviderName.GEMINI_TRANSCRIBE.value: SettingsSecretKey.GEMINI_TRANSCRIBE_API_KEY.value,
    STTProviderName.ELEVENLABS_SCRIBE.value: SettingsSecretKey.ELEVENLABS_SCRIBE_API_KEY.value,
    STTProviderName.SONIOX.value: SettingsSecretKey.SONIOX_API_KEY.value,
}
_TRANSLATION_SECRET_KEYS = {
    "gemini": SettingsSecretKey.GOOGLE_API_KEY.value,
    "openai": SettingsSecretKey.OPENAI_API_KEY.value,
    "openrouter": SettingsSecretKey.OPENROUTER_API_KEY.value,
    "deepseek": SettingsSecretKey.DEEPSEEK_API_KEY.value,
}
_TERMINAL = frozenset(
    {
        "applied",
        "degraded",
        "rejected",
        "persistence_failed",
        "failed",
        "action_required",
        "cancelled",
        "interrupted",
    }
)
Navigation = Callable[[str, str | None], dict]


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _pick(value: object, fields: tuple[str, ...]) -> dict:
    return {
        key: item
        for key, item in _mapping(value).items()
        if key in fields and (item is None or type(item) in (str, bool, int, float))
    }


def project_accounts(value: object) -> dict:
    source = _mapping(value)
    chatgpt = _mapping(source.get("chatgpt"))
    managed = _mapping(source.get("managed"))
    budget = _mapping(managed.get("budget"))
    remaining = budget.get("remaining_percent")
    availability = budget.get("availability")
    if type(availability) is not str or availability not in {"cached", "unknown", "stale"}:
        availability = "unknown"
    if type(remaining) is not int or not 0 <= remaining <= 100:
        remaining = None
        if availability == "cached":
            availability = "unknown"
    if availability != "cached":
        remaining = None
    provider = managed.get("provider")
    return {
        "chatgpt": {
            **{
                key: chatgpt[key] if type(chatgpt.get(key)) is bool else None
                for key in ("available", "signed_in", "in_progress", "sign_in_required")
            },
            "usage": {"availability": "unavailable", "remaining_percent": None},
        },
        "managed": {
            "provider": (
                provider if type(provider) is str and provider in {"qq", "discord"} else None
            ),
            **{
                key: managed[key] if type(managed.get(key)) is bool else None
                for key in ("available", "selected", "credential_present", "in_progress")
            },
            "budget": {
                "unit": "percent",
                "remaining_percent": remaining,
                "availability": availability,
                "freshness": "unknown",
            },
        },
        "puri": {
            "allowance": {
                "unit": "submitted_prompt_turns",
                "limit": 30,
                "period": "server_utc_day",
                "remaining": None,
                "availability": "unavailable",
                "exhaustion": "not_observed",
            }
        },
    }


def _credential_readiness(presence: Mapping, keys: tuple[str, ...]) -> str:
    if any(presence.get(key) is True for key in keys):
        return "present"
    if keys and all(presence.get(key) is False for key in keys):
        return "absent"
    return "unknown"


def _alibaba_secret_key(context: Mapping) -> str | None:
    region = _mapping(context.get("settings")).get("translation.qwen.region")
    return f"alibaba_api_key_{region}" if region in {"beijing", "singapore"} else None


def project_osc_output(value: object) -> dict:
    source = _mapping(value)
    result = {
        key: source[key]
        for key, allowed in _OSC_ENUMS.items()
        if key in source
        and (source[key] is None or type(source[key]) is str)
        and source[key] in allowed
    }
    result.update(
        {
            key: source[key]
            for key in (
                "receiver_available",
                "sender_available",
                "query_service_started",
                "discovered_remote_present",
            )
            if type(source.get(key)) is bool
        }
    )
    result["remote_delivery"] = "unacknowledged"
    listener = _mapping(source.get("local_listener"))
    if listener:
        status = listener.get("status")
        result["local_listener"] = {
            "status": (
                status
                if type(status) is str
                and status
                in {
                    "unknown",
                    "not_applicable",
                    "matched",
                    "other_process",
                    "not_listening",
                    "vrchat_not_running",
                }
                else "unknown"
            ),
            "vrchat_running": (
                listener["vrchat_running"] if type(listener.get("vrchat_running")) is bool else None
            ),
        }
        observed_at = listener.get("observed_at")
        if observed_at is None or (
            type(observed_at) in (int, float) and math.isfinite(observed_at) and observed_at >= 0
        ):
            result["local_listener"]["observed_at"] = observed_at
    return result


def _safe_osc_receipt(receipt: dict) -> dict:
    result = _pick(
        receipt,
        ("instance_id", "operation_id", "request_id", "status", "terminal", "revision"),
    )
    reason = receipt.get("reason")
    if type(reason) is str and reason in {
        "lifecycle_stopped",
        "connection_changed",
        "automatic_mode_required",
        "discovery_in_progress",
        "settings_unavailable",
        "target_not_found",
        "oscquery_unavailable",
        "discovery_timeout",
        "local_sender_unavailable",
    }:
        result["reason"] = reason
    effect = receipt.get("effect")
    if type(effect) is str and effect in {
        "discovery_target_applied",
        "discovery_unavailable",
    }:
        result["effect"] = effect
    if type(receipt.get("retry_started")) is bool:
        result["retry_started"] = receipt["retry_started"]
    if "output" in receipt:
        result["output"] = project_osc_output(receipt["output"])
    return result


def project_settings(current: dict) -> dict:
    intent = _mapping(_mapping(current.get("settings")).get("intent"))
    values = {}
    for path in (*SETTINGS_FIELD_PATHS.values(), *LANGUAGE_FIELD_PATHS.values()):
        value = intent
        for part in path.split("."):
            value = _mapping(value).get(part)
        if type(value) is str:
            value = safe_audio_device_label(value)
        if value is None or type(value) in (str, bool, int, float, list, dict):
            values[path] = deepcopy(value)
    osc = _mapping(intent.get("osc"))
    values["osc.connection"] = {
        key: osc.get("connection_mode" if key == "mode" else key)
        for key in ("mode", "send_port", "receive_port")
    }
    return {
        "revision": current.get("revision"),
        "settings": values,
        "private_presence": {
            field: value
            for field, value in _mapping(current.get("private_presence")).items()
            if field in private_fields() and type(value) is bool
        },
    }


def _choices(catalog: dict, pointer: str) -> object:
    value = catalog
    for part in pointer.removeprefix("/").split("/"):
        value = _mapping(value).get(part)
    if value is None:
        raise ValueError("settings choices unavailable")
    return value


def _schema(catalog: dict, source: dict) -> dict:
    kind = source.get("type")
    if kind == "optional-string":
        kind = "string"
    if kind == "object":
        members = source.get("properties")
        if members is None and "key_choices" in source:
            members = {
                key: {"type": "string", "choices": source["value_choices_by_key"] + "/" + key}
                for key in _choices(catalog, source["key_choices"])
            }
        properties = {key: _schema(catalog, item) for key, item in (members or {}).items()}
        return {
            "type": "object",
            "properties": properties,
            "required": [],
            "additionalProperties": False,
        }
    if kind == "array<string>":
        result = {"type": "array", "items": {"type": "string"}}
        if "items" in source:
            result["items"]["enum"] = list(_choices(catalog, source["items"]))
        for original, target in (
            ("min_items", "minItems"),
            ("max_items", "maxItems"),
            ("unique", "uniqueItems"),
        ):
            if original in source:
                result[target] = source[original]
        return result
    if kind not in {"string", "boolean", "integer", "number"}:
        raise ValueError("unsupported settings schema")
    result = {"type": kind}
    for bound in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if bound in source:
            result[bound] = source[bound]
    if "choices" in source:
        choices = _choices(catalog, source["choices"])
        if isinstance(choices, Mapping):
            choices = list(dict.fromkeys(value for values in choices.values() for value in values))
        elif isinstance(choices, list):
            choices = [item.get("value") if isinstance(item, Mapping) else item for item in choices]
        result["enum"] = list(choices)
        if source.get("empty_allowed") and "" not in result["enum"]:
            result["enum"].append("")
    return result


def settings_tool_schema(catalog: dict) -> dict:
    schemas = _mapping(catalog.get("field_schemas"))
    if not SETTINGS_FIELDS <= schemas.keys():
        raise ValueError("settings schemas unavailable")
    properties = {
        key: {"type": "string"} if key in _DYNAMIC_CHOICE_FIELDS else _schema(catalog, schemas[key])
        for key in sorted(SETTINGS_FIELDS)
    }
    return {
        "type": "object",
        "properties": properties,
        "required": [],
        "additionalProperties": False,
    }


def _validate(value: object, schema: dict) -> None:
    validate(value, schema, partial=True)


def project_audio_diagnosis(value: object) -> dict:
    source = _mapping(value)
    result = _pick(
        source,
        (
            "state",
            "scope",
            "selected_candidate_id",
            "selection_basis",
            "cleanup_complete",
            "failure_code",
        ),
    )
    result["current_route"] = {
        key: safe_audio_device_label(item)
        for key, item in _mapping(source.get("current_route")).items()
        if key in {"input_host_api", "input_device"} and type(item) is str
    }
    candidates = source.get("candidates")
    result["candidates"] = []
    if isinstance(candidates, list):
        for candidate in candidates[:4]:
            projected = _pick(
                candidate,
                (
                    "id",
                    "device_index",
                    "current_route",
                    "opened",
                    "frames_received",
                    "frame_count",
                    "vad_sample_count",
                    "silence_observed",
                    "speech_detected",
                    "peak_level",
                    "speech_probability_min",
                    "speech_probability_max",
                    "speech_threshold",
                    "elapsed_ms",
                    "failure_code",
                    "cleanup_complete",
                ),
            )
            samples = _mapping(candidate).get("observed_speech_probabilities")
            if isinstance(samples, list):
                projected["observed_speech_probabilities"] = [
                    item
                    for item in samples[:16]
                    if type(item) in (int, float) and math.isfinite(item) and 0 <= item <= 1
                ]
            projected.update(
                {
                    key: safe_audio_device_label(item)
                    for key, item in _mapping(candidate).items()
                    if key in {"input_host_api", "input_device"} and type(item) is str
                }
            )
            result["candidates"].append(projected)
    result["restoration"] = _pick(source.get("restoration"), ("self", "peer"))
    result["bounds"] = _pick(source.get("bounds"), ("max_candidates", "candidate_timeout_s"))
    return result


def _safe_receipt(receipt: dict) -> dict:
    result = _pick(
        receipt,
        (
            "instance_id",
            "operation_id",
            "request_id",
            "status",
            "terminal",
            "revision",
            "reason",
            "diagnostic_id",
            "action",
            "translation_enabled",
            "runtime_ready",
            "recognition",
            "settings_changed",
        ),
    )
    if isinstance(receipt.get("error"), Mapping):
        result["error"] = _pick(receipt["error"], ("code", "type", "current_revision"))
    diagnostic_fields = ("diagnostic_id", "code", "category", "disposition", "operation")
    if isinstance(receipt.get("diagnostics"), Mapping):
        result["diagnostics"] = _pick(receipt["diagnostics"], diagnostic_fields)
    if isinstance(receipt.get("transaction"), Mapping):
        transaction = receipt["transaction"]
        result["transaction"] = _pick(transaction, ("status",))
        if isinstance(transaction.get("diagnostics"), Mapping):
            result["transaction"]["diagnostics"] = _pick(
                transaction["diagnostics"], diagnostic_fields
            )
    if isinstance(receipt.get("runtime_observation"), Mapping):
        result["runtime_observation"] = _pick(receipt["runtime_observation"], ("status", "reason"))
    if isinstance(receipt.get("channel"), Mapping):
        state = receipt["channel"]
        result["channel"] = _pick(
            state,
            (
                "selected_provider",
                "runtime_provider",
                "capture_attached_provider",
                "desired_active",
                "effective_active",
                "pending_handoff",
                "failure_reason",
                "provider_status",
            ),
        )
        result["channel"]["capture"] = _pick(
            state.get("capture"),
            ("generation", "state", "desired_active", "effective_active", "has_source"),
        )
    if isinstance(receipt.get("diagnosis"), Mapping):
        result["diagnosis"] = project_audio_diagnosis(receipt["diagnosis"])
    return result


def project_operation_receipt(receipt: dict, tool: str) -> dict:
    if tool == "osc.discovery.retry":
        return _safe_osc_receipt(receipt)
    result = _safe_receipt(receipt)
    result.update(
        _pick(
            receipt,
            (
                "translation_result",
                "recognition_result",
                "channel",
                "captured_settings_revision",
                "current_settings_revision",
                "final_event",
                "configured_provider",
                "exercised_provider",
                "recognition_scope",
                "input_samples",
                "sample_rate_hz",
                "fixture_duration_seconds",
                "configured_language",
                "source_mode",
                "accuracy",
                "capture",
                "translation",
                "captured_generation",
                "current_generation",
                "source_language",
                "target_language",
                "output_delivery",
                "display",
                "failure_category",
                "cleanup_complete",
                "cleanup_outcome",
            ),
        )
    )
    if isinstance(receipt.get("target_languages"), list):
        result["target_languages"] = [
            target for target in receipt["target_languages"][:2] if isinstance(target, str)
        ]
    if isinstance(receipt.get("target_results"), list):
        result["target_results"] = [
            _pick(target, ("target_language", "translation_result"))
            for target in receipt["target_results"][:2]
            if isinstance(target, Mapping)
        ]
    return result


def _rejected(code: str, explanation: str) -> dict:
    return {"status": "rejected", "error": {"code": code, "explanation": explanation}}


class AgentTools:
    def __init__(
        self,
        control: ApplicationControl,
        *,
        changed: Callable[[], None],
        record_action: Callable[[dict], None],
        emit_event: Callable[[dict], bool],
    ) -> None:
        self._control = control
        self._changed = changed
        self._record_action = record_action
        self._emit_event = emit_event
        self._context: dict = {}
        self._navigation: Navigation | None = None
        self._navigation_observations: dict[str, dict] = {}
        self._confirmation: dict | None = None
        self._confirmation_queue: list[dict] = []
        self._proposal_id: str | None = None
        self._tasks: set[asyncio.Task] = set()
        self._command_requests: dict[str, dict] = {}
        self._scope = LifecycleScope("agent-settings-tools")
        self._closed = False
        self._localize: Callable[[str], str] = lambda key: key

    def bind_localize(self, localize: Callable[[str], str]) -> None:
        self._localize = localize

    @property
    def confirmation(self) -> dict | None:
        if self._confirmation is None:
            return None
        return deepcopy(
            {key: value for key, value in self._confirmation.items() if key != "previous"}
        )

    @property
    def confirmation_display(self) -> dict | None:
        if self._confirmation is None:
            return None
        return deepcopy(self._confirmation)

    def observe_context(self, context: dict) -> None:
        if type(context.get("revision")) is int and isinstance(context.get("settings"), Mapping):
            self._context["revision"] = context["revision"]
            self._context["settings"] = deepcopy(context["settings"])
        for key in ("app", "channels"):
            if isinstance(context.get(key), Mapping):
                self._context[key] = deepcopy(context[key])

    def reset_context(self) -> None:
        self._context.clear()

    def bind_navigation(self, callback: Navigation | None) -> None:
        self._navigation = callback
        self._navigation_observations.clear()

    def cancel_confirmation(self) -> None:
        changed = self._confirmation is not None
        self._confirmation = None
        self._confirmation_queue.clear()
        self._proposal_id = None
        if changed:
            self._changed()

    async def execute(
        self,
        name: str,
        arguments: dict,
        *,
        require_current: Callable[[], None] = lambda: None,
        request_id: str | None = None,
    ) -> dict:
        require_current()
        if self._closed:
            return _rejected("tools_closed", "Settings tools are closed.")
        if type(arguments) is not dict:
            return _rejected("invalid_arguments", "Tool arguments must be an object.")
        try:
            action = select_app_action(name, arguments)
        except ValueError:
            return _rejected("invalid_arguments", "Choose a supported action for this tool.")
        catalog = {}
        if action.settings:
            try:
                catalog = await self._control.query("settings.choices", {})
                require_current()
            except Exception:
                return _rejected("settings_unavailable", "Settings choices are unavailable.")
        try:
            validate(arguments, action.schema(catalog))
        except ValueError:
            return _rejected(
                "invalid_arguments", f"Use {action.signature}; omit other fields and nulls."
            )
        return await self.execute_action(
            action, action.arguments(arguments), catalog, require_current, request_id
        )

    async def execute_action(
        self,
        action: AppAction,
        arguments: dict,
        catalog: dict,
        require_current: Callable[[], None],
        request_id: str | None,
    ) -> dict:
        require_current()
        if action.handler == "command":
            command = arguments["command"]
            task = self._scope.create_task(
                self._run_command(
                    command,
                    command,
                    {key: value for key, value in arguments.items() if key != "command"},
                    require_current,
                    request_id,
                ),
                name=f"{command}:{uuid4().hex}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return await asyncio.shield(task)
        if action.handler == "query":
            name = arguments["query"]
            try:
                source = await self._control.query(name, {})
                require_current()
                if not isinstance(source, dict) or source.get("status") == "failed":
                    return {"status": "unavailable", "code": "query_unavailable"}
                projected = await project_query(name, source, self._control, self._localize)
                return {"status": "observed", "result": projected}
            except Exception:
                return {"status": "unavailable", "code": "query_unavailable"}
        if action.handler == "card":
            return await self._propose_command(
                arguments["command"],
                {key: value for key, value in arguments.items() if key != "command"},
                require_current,
            )
        if action.handler == "navigation":
            return self._open_settings(arguments)
        if action.handler == "provider_setup":
            return self._get_provider_setup(arguments)
        if action.handler == "withdraw":
            card = self._confirmation
            if card is None or card["id"] != arguments["card_id"]:
                return _rejected(
                    "confirmation_unavailable", "This confirmation card is no longer active."
                )
            self.cancel_confirmation()
            return {"status": "withdrawn", "card_id": card["id"]}
        if action.handler != "settings":
            return _rejected("unsupported_tool", "This action is owned by the conversation.")
        return await self._change_settings(arguments, catalog, require_current)

    async def _change_settings(
        self, arguments: dict, catalog: dict, require_current: Callable[[], None]
    ) -> dict:
        if arguments.keys() != {"changes"} or type(arguments["changes"]) is not dict:
            return _rejected("invalid_arguments", "Provide only a changes object.")
        original = arguments["changes"]
        if original.keys() - SETTINGS_FIELDS:
            return _rejected(
                "invalid_arguments", "The requested settings fields are not supported."
            )
        changes = deepcopy(original)
        if not changes or changes.get("languages") == {}:
            return _rejected("invalid_arguments", "Provide at least one supported change.")
        context = deepcopy(self._context)
        if type(context.get("revision")) is not int:
            return _rejected("context_required", "Read current settings before changing them.")
        schema = settings_tool_schema(catalog)
        try:
            if changes.keys() & {"translation.model", "translation.connection"}:
                for field in ("translation.model", "translation.connection"):
                    if field not in changes:
                        changes[field] = context["settings"][SETTINGS_FIELD_PATHS[field]]
            for key, value in changes.items():
                _validate(value, schema["properties"][key])
            if "translation.model" in changes:
                connections = _choices(catalog, "/choices/translation.connection")
                if (
                    changes["translation.connection"]
                    not in connections[changes["translation.model"]]
                ):
                    raise ValueError("unsupported translation model and connection combination")
            await self._validate_audio_route(changes, catalog, context, require_current)
            for field in changes.keys() & (_DYNAMIC_CHOICE_FIELDS - _AUDIO_ROUTE_FIELDS):
                options = catalog["choices"][field]
                if not any(
                    (
                        (
                            option.get("value") == changes[field]
                            and option.get("available") is not False
                        )
                        if isinstance(option, Mapping)
                        else option == changes[field]
                    )
                    for option in options
                ):
                    raise ValueError("Requested setting choice is not currently available.")
            previous = self._previous(changes, context)
        except (KeyError, TypeError, ValueError) as exc:
            return _rejected("invalid_arguments", str(exc))
        policy_context = {"settings": context["settings"]}
        try:
            runtime = await self._control.query("app.status", {})
            require_current()
            policy_context.update({"app": runtime, "channels": runtime.get("channels", {})})
        except Exception:
            pass
        if changes.keys() & {
            "stt.provider",
            "peer_stt.provider",
            "stt.cloud_free_tier_providers",
            "translation.model",
            "translation.connection",
        }:
            try:
                secrets = await self._control.query("secrets.presence", {})
                if secrets.get("status") != "failed":
                    policy_context["secret_presence"] = _mapping(
                        secrets.get("effective_presence", secrets.get("presence"))
                    )
            except Exception:
                pass
            if changes.get("translation.connection") in {
                TranslationConnection.CHATGPT.value,
                TranslationConnection.MANAGED.value,
                TranslationConnection.MANAGED_CHINA.value,
            }:
                try:
                    policy_context["accounts"] = project_accounts(
                        await self._control.query("accounts.status", {})
                    )
                except Exception:
                    pass
        if self._closed:
            return _rejected("tools_closed", "Settings tools are closed.")
        require_current()
        self._proposal_id = uuid4().hex
        self._confirmation_queue = []
        presentation = project_catalog(catalog, self._localize)
        for group in _approval_groups(changes):
            before = {
                key: (
                    {field: previous[key][field] for field in value}
                    if key == "languages"
                    else previous[key]
                )
                for key, value in group.items()
            }
            route_effects = self._route_effects(group, before, policy_context)
            self._confirmation_queue.append(
                {
                    "changes": group,
                    "previous": before,
                    "effects": self._effects(group, before, policy_context, route_effects),
                    "route_effects": route_effects,
                    "presentation": {
                        field: presentation[field] for field in group if field in presentation
                    },
                }
            )
            for key, value in group.items():
                if key == "languages":
                    for field, item in value.items():
                        policy_context["settings"][LANGUAGE_FIELD_PATHS[field]] = item
                else:
                    policy_context["settings"][SETTINGS_FIELD_PATHS[key]] = value
        self._stage_confirmation(context["revision"])
        self._changed()
        return {"status": "needs_confirmation", "confirmation": self.confirmation}

    def _stage_confirmation(self, revision: int) -> None:
        self._confirmation = {
            **self._confirmation_queue.pop(0),
            "id": uuid4().hex,
            "revision": revision,
        }

    @staticmethod
    def _previous(changes: dict, context: dict) -> dict:
        settings = context["settings"]
        previous = {}
        for key, value in changes.items():
            if key == "languages":
                previous[key] = {field: settings[LANGUAGE_FIELD_PATHS[field]] for field in value}
            else:
                previous[key] = deepcopy(settings.get(SETTINGS_FIELD_PATHS[key]))
        return previous

    async def _validate_audio_route(
        self,
        changes: dict,
        catalog: dict,
        context: dict,
        require_current: Callable[[], None],
    ) -> None:
        if not changes.keys() & _AUDIO_ROUTE_FIELDS:
            return
        choices = _mapping(catalog.get("choices"))
        hosts = choices.get("audio.input_host_api")
        if not isinstance(hosts, list):
            raise ValueError("Audio route enumeration unavailable.")
        host = changes.get("audio.input_host_api", context["settings"].get("audio.input_host_api"))
        if type(host) is not str or not any(
            _mapping(item).get("value") == host and _mapping(item).get("available") is True
            for item in hosts
        ):
            raise ValueError("Audio host API is not currently available.")
        actual_host = normalize_input_host_api(host).actual_host_api
        try:
            enumeration = await self._control.query("audio.devices", {})
            require_current()
        except Exception:
            raise ValueError("Audio route enumeration unavailable.") from None
        devices = _mapping(enumeration).get("microphones")
        if not isinstance(devices, list):
            raise ValueError("Audio route enumeration unavailable.")
        device = changes.get("audio.input_device", context["settings"].get("audio.input_device"))
        if type(device) is not str:
            raise ValueError("Audio input device is unavailable.")
        if device and not any(
            _mapping(item).get("name") == device
            and (not actual_host or _mapping(item).get("host_api") == actual_host)
            for item in devices
        ):
            raise ValueError("Audio input device is not available for the requested host API.")

    @staticmethod
    def _route_effects(changes: dict, previous: dict, context: dict) -> list[dict]:
        changed = {key for key in changes if changes[key] != previous[key]}
        presence = _mapping(context.get("secret_presence"))
        settings = _mapping(context.get("settings"))
        result = []
        for key, scope in (
            ("stt.provider", "self_recognition"),
            ("peer_stt.provider", "peer_recognition"),
        ):
            provider = changes.get(key, settings.get(SETTINGS_FIELD_PATHS[key]))
            if key not in changed and not (
                "stt.cloud_free_tier_providers" in changed and provider == "rolling_free"
            ):
                continue
            fact = {"scope": scope, "provider": provider}
            secret = _STT_SECRET_KEYS.get(provider)
            if provider == STTProviderName.QWEN_AUDIO.value:
                secret = _alibaba_secret_key(context)
            if provider == "rolling_free":
                members = changes.get(
                    "stt.cloud_free_tier_providers", settings.get("stt.cloud_free_tier_providers")
                )
                keys = tuple(
                    _STT_SECRET_KEYS[member]
                    for member in members or ()
                    if member in _STT_SECRET_KEYS
                )
                fact.update(
                    cost="free_tier_eligible",
                    credential_requirement="any_free_tier_key",
                    credential_readiness=_credential_readiness(presence, keys),
                )
            elif secret is not None or provider == STTProviderName.QWEN_AUDIO.value:
                fact.update(
                    cost=(
                        "free_credit_or_allowance"
                        if provider in RECOGNITION_ALLOWANCES
                        else "provider_billed"
                    ),
                    credential_requirement="api_key",
                    credential_readiness=_credential_readiness(
                        presence, (secret,) if secret is not None else ()
                    ),
                )
                if provider in RECOGNITION_ALLOWANCES:
                    fact["allowance_kind"] = RECOGNITION_ALLOWANCES[provider]
                    fact["allowance_remaining"] = None
            else:
                fact.update(
                    cost="custom" if is_custom_stt_provider(provider) else "local",
                    credential_requirement="none",
                    credential_readiness="not_required",
                )
            result.append(fact)
        if changed & {"translation.model", "translation.connection"}:
            model = changes["translation.model"]
            connection = changes["translation.connection"]
            provider = (
                "custom_http"
                if connection == TranslationConnection.CUSTOM_HTTP.value
                else provider_llm_for_translation(model, connection)
            )
            fact = {"scope": "translation", "provider": provider, "connection": connection}
            accounts = _mapping(context.get("accounts"))
            if connection in {
                TranslationConnection.MANAGED.value,
                TranslationConnection.MANAGED_CHINA.value,
            }:
                managed = _mapping(accounts.get("managed"))
                account_provider = (
                    "qq" if connection == TranslationConnection.MANAGED_CHINA.value else "discord"
                )
                credential = (
                    managed.get("credential_present")
                    if managed.get("provider") == account_provider
                    else None
                )
                fact.update(
                    cost="managed_budget",
                    credential_requirement="managed_account",
                    credential_readiness=_credential_readiness(
                        {"account": credential}, ("account",)
                    ),
                )
                if managed.get("provider") == account_provider:
                    fact["budget"] = dict(_mapping(managed.get("budget")))
            elif connection == TranslationConnection.CHATGPT.value:
                chatgpt = _mapping(accounts.get("chatgpt"))
                signed_in = chatgpt.get("signed_in")
                if chatgpt.get("sign_in_required") is True:
                    signed_in = False
                fact.update(
                    cost="subscription",
                    credential_requirement="chatgpt_account",
                    credential_readiness=_credential_readiness(
                        {"account": signed_in}, ("account",)
                    ),
                )
            elif connection in {
                TranslationConnection.OPENROUTER.value,
                TranslationConnection.OFFICIAL_BYOK.value,
            }:
                secret = (
                    _alibaba_secret_key(context)
                    if provider == "qwen"
                    else _TRANSLATION_SECRET_KEYS.get(provider)
                )
                fact.update(
                    cost="free_tier_eligible" if provider == "gemini" else "provider_billed",
                    credential_requirement="api_key",
                    credential_readiness=_credential_readiness(
                        presence, (secret,) if secret is not None else ()
                    ),
                )
            else:
                fact.update(
                    cost=(
                        "public"
                        if connection == TranslationConnection.PUBLIC_API.value
                        else (
                            "custom"
                            if connection == TranslationConnection.CUSTOM_HTTP.value
                            else "local"
                        )
                    ),
                    credential_requirement="none",
                    credential_readiness="not_required",
                )
            result.append(fact)
        return result

    @staticmethod
    def _effects(
        changes: dict, previous: dict, context: dict, route_effects: list[dict]
    ) -> list[str]:
        changed = {key for key in changes if changes[key] != previous[key]}
        effects = []
        for route in route_effects:
            label = {
                "self_recognition": "Self recognition",
                "peer_recognition": "Peer recognition",
                "translation": "Translation",
            }[route["scope"]]
            cost = route["cost"]
            if cost == "provider_billed":
                effects.append(
                    f"{label} switches to a paid service billed by the provider or prepaid balance."
                )
            elif cost == "free_credit_or_allowance" and route["provider"] != "gemini_transcribe":
                effects.append(
                    f"{label} offers free credit or allowance; usage can be billed on a paid account."
                )
            elif cost == "managed_budget":
                effects.append("Translation uses the managed account's entitlement and budget.")
            elif cost == "subscription":
                effects.append(
                    "Translation uses the ChatGPT subscription's limits, not managed budget or API balance."
                )
            readiness = route["credential_readiness"]
            if readiness == "absent":
                requirement = route["credential_requirement"]
                credential = (
                    "an eligible free-tier key"
                    if requirement == "any_free_tier_key"
                    else (
                        "account authorization"
                        if requirement in {"managed_account", "chatgpt_account"}
                        else "a provider key"
                    )
                )
                effects.append(f"{label} cannot start until {credential} is available.")
            elif readiness == "unknown":
                effects.append(
                    f"{label} credential readiness is unknown; availability has not been confirmed."
                )
        for channel, fields in (
            (
                "self",
                {
                    "audio.input_host_api",
                    "audio.input_device",
                    "stt.provider",
                    "stt.cloud_free_tier_providers",
                    "self_vad.speech_threshold",
                    "languages",
                    "translation.model",
                    "translation.connection",
                },
            ),
            (
                "peer",
                {
                    "peer_stt.provider",
                    "stt.cloud_free_tier_providers",
                    "peer_vad.speech_threshold",
                    "peer_vad.hangover_ms",
                    "peer_vad.pre_roll_ms",
                    "peer_expected_languages",
                    "languages",
                    "translation.model",
                    "translation.connection",
                },
            ),
        ):
            state = _mapping(_mapping(context.get("channels")).get(channel))
            capture = _mapping(state.get("capture"))
            active = any(
                item is True
                for item in (
                    state.get("effective_active"),
                    state.get("desired_active"),
                    capture.get("effective_active"),
                    capture.get("desired_active"),
                )
            )
            if changed & fields and (active or not state):
                effects.append(f"This change may interrupt {channel} recognition.")
        if changed & {"languages", "translation.model", "translation.connection"}:
            app = _mapping(context.get("app"))
            if app.get("translation_enabled") is True or "translation_enabled" not in app:
                effects.append("This change may interrupt active translation.")
        return list(dict.fromkeys(effects))

    async def confirm(
        self,
        card_id: str,
        approved: bool,
        *,
        require_current: Callable[[], None] = lambda: None,
    ) -> dict:
        require_current()
        card = self._confirmation
        if self._closed or type(approved) is not bool or card is None or card.get("id") != card_id:
            return _rejected(
                "confirmation_unavailable", "This confirmation card is no longer active."
            )
        self._confirmation = None
        proposal_id = self._proposal_id
        self._changed()
        if "command" in card:
            self._confirmation_queue.clear()
            self._proposal_id = None
            result = (
                await self._apply_command(card, require_current)
                if approved
                else {"status": "declined", "command": card["command"], "changes": card["changes"]}
            )
            self._changed()
            self._emit_event(
                {"kind": "confirmation", "card_id": card_id, "result": deepcopy(result)}
            )
            return result
        if not approved:
            self._confirmation_queue.clear()
            self._proposal_id = None
            result = {
                "status": "declined",
                "changes": card["changes"],
                "revision": card["revision"],
            }
            self._emit_event({"kind": "confirmation", "card_id": card_id, "result": result})
            return result
        return await self._start_apply(
            card["changes"],
            card["previous"],
            card["revision"],
            proposal_id=proposal_id,
            card_id=card_id,
            route_effects=card["route_effects"],
            require_current=require_current,
        )

    async def _start_apply(
        self,
        changes: dict,
        previous: dict,
        revision: int,
        *,
        proposal_id: str | None,
        card_id: str,
        route_effects: list[dict] | None = None,
        require_current: Callable[[], None] = lambda: None,
    ) -> dict:
        require_current()
        if proposal_id != self._proposal_id:
            return _rejected(
                "confirmation_unavailable", "This confirmation card is no longer active."
            )
        task = self._scope.create_task(
            self._apply(
                changes,
                previous,
                revision,
                proposal_id=proposal_id,
                card_id=card_id,
                route_effects=route_effects,
                require_current=require_current,
            ),
            name=f"settings.apply:{uuid4().hex}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return await asyncio.shield(task)

    async def _apply(
        self,
        changes: dict,
        previous: dict,
        revision: int,
        *,
        proposal_id: str | None,
        card_id: str,
        route_effects: list[dict] | None,
        require_current: Callable[[], None],
    ) -> dict:
        request_id = uuid4().hex
        require_current()
        if proposal_id != self._proposal_id:
            return _rejected(
                "confirmation_unavailable", "This confirmation card is no longer active."
            )
        try:
            receipt = await self._control.submit(
                "settings.apply",
                {"changes": deepcopy(changes)},
                request_id=request_id,
                expected_revision=revision,
            )
            submitted = _safe_receipt(receipt)
            self._record_action(
                {
                    "tool": "settings.apply",
                    "action_id": request_id,
                    "phase": "submitted",
                    "changes": deepcopy(changes),
                    "expected_revision": revision,
                    "receipt": submitted,
                }
            )
            if receipt.get("status") not in _TERMINAL:
                receipt = await self._control.wait(receipt["operation_id"])
            safe = _safe_receipt(receipt)
            if receipt.get("status") not in _TERMINAL:
                result = {"status": "pending", "receipt": safe, "changes": deepcopy(changes)}
            else:
                result = {"status": safe["status"], "receipt": safe, "changes": deepcopy(changes)}
        except Exception:
            result = {
                "status": "failed",
                "error": {"code": "settings_operation_unavailable"},
                "changes": deepcopy(changes),
            }
        if result["status"] != "pending":
            try:
                current = await self._control.query("settings.current", {})
                result["context"] = project_settings(current)
            except Exception:
                result["context_unavailable"] = True
        if route_effects:
            result["route_effects"] = deepcopy(route_effects)
        current_proposal = proposal_id is not None and proposal_id == self._proposal_id
        resume = current_proposal
        if current_proposal:
            revision = result.get("receipt", {}).get("revision")
            if (
                result["status"] == "applied"
                and type(revision) is int
                and self._confirmation_queue
                and not self._closed
            ):
                self._stage_confirmation(revision)
                result["confirmation"] = self.confirmation
                resume = False
            else:
                self._confirmation_queue.clear()
                self._proposal_id = None
        phase = "observed" if result["status"] == "pending" else "completed"
        self._record_action(
            {
                "action_id": request_id,
                "tool": "settings.apply",
                "phase": phase,
                "changed": changes != previous,
                "result": deepcopy(result),
            }
        )
        self._changed()
        if result["status"] != "pending":
            self._emit_event(
                {
                    "kind": "confirmation",
                    "card_id": card_id,
                    "result": deepcopy(result),
                    "resume": resume,
                }
            )
        return result

    async def _run_command(
        self,
        name: str,
        command: str,
        arguments: dict,
        require_current: Callable[[], None],
        request_id: str | None = None,
    ) -> dict:
        signature = json.dumps([command, arguments], sort_keys=True, separators=(",", ":"))
        previous = self._command_requests.get(signature)
        retained = self._command_requests.setdefault(
            signature, {"request_id": request_id or uuid4().hex}
        )
        request_id = retained["request_id"]
        receipt = retained.get("receipt")
        require_current()
        if previous is not None and receipt is None:
            return deepcopy(retained.get("result", {"status": "pending", "request_id": request_id}))
        try:
            if receipt is not None:
                try:
                    observed = await self._control.operation(receipt["operation_id"])
                    if (
                        isinstance(observed, dict)
                        and observed.get("operation_id") == receipt["operation_id"]
                        and isinstance(observed.get("status"), str)
                    ):
                        receipt = observed
                except Exception:
                    pass
            else:
                receipt = await self._control.submit(command, arguments, request_id=request_id)
            retained["receipt"] = dict(receipt)
            self._record_action(
                {
                    "tool": name,
                    "action_id": request_id,
                    "phase": "submitted",
                    "receipt": (
                        _safe_osc_receipt(receipt)
                        if command == "osc.discovery.retry"
                        else _safe_receipt(receipt)
                    ),
                }
            )
            if receipt.get("status") not in _TERMINAL:
                receipt = await self._control.wait(receipt["operation_id"])
            safe = project_operation_receipt(receipt, name)
            result = {
                "status": safe["status"] if safe.get("terminal") else "pending",
                "receipt": safe,
            }
        except Exception:
            if isinstance(receipt, dict) and receipt.get("operation_id"):
                try:
                    observed = await self._control.operation(receipt["operation_id"])
                    if (
                        isinstance(observed, dict)
                        and observed.get("operation_id") == receipt["operation_id"]
                        and isinstance(observed.get("status"), str)
                    ):
                        receipt = observed
                except Exception:
                    pass
                result = {
                    "status": receipt["status"] if receipt.get("terminal") else "pending",
                    "receipt": project_operation_receipt(receipt, name),
                }
            else:
                result = {
                    "status": "unavailable",
                    "request_id": request_id,
                    "error": {"code": f"{command.replace('.', '_')}_outcome_unknown"},
                }
        if isinstance(receipt, dict):
            retained["receipt"] = dict(receipt)
        retained["result"] = deepcopy(result)
        if result["status"] not in {"pending", "unavailable"} or (
            isinstance(receipt, dict) and receipt.get("terminal") is True
        ):
            self._command_requests.pop(signature, None)
        self._record_action(
            {
                "tool": name,
                "action_id": request_id,
                "phase": "observed" if result["status"] == "pending" else "completed",
                "result": deepcopy(result),
            }
        )
        return result

    async def _propose_command(
        self, command: str, arguments: dict, require_current: Callable[[], None]
    ) -> dict:
        try:
            current = await self._control.query("settings.current", {})
            require_current()
            before = await self._command_state(command, arguments)
            require_current()
        except asyncio.CancelledError:
            raise
        except Exception:
            return {"status": "unavailable", "code": "settings_unavailable"}
        if type(current.get("revision")) is not int:
            return {"status": "unavailable", "code": "settings_unavailable"}
        after = COMMAND_CHANGES[command]
        if before == after:
            return {"status": "rejected", "code": "already_applied"}
        field = command_field(command, arguments)
        self.cancel_confirmation()
        self._confirmation = {
            "id": uuid4().hex,
            "revision": current["revision"],
            "command": command,
            "arguments": deepcopy(arguments),
            "changes": {field: after},
            "previous": {} if before is None else {field: before},
            "effects": [_COMMAND_EFFECTS[command]],
            "presentation": {},
        }
        self._changed()
        return {"status": "needs_confirmation", "confirmation": self.confirmation}

    async def _command_state(self, command: str, arguments: dict) -> object | None:
        if command == "overlay.position.reset":
            return None
        if command == "auth.logout" and arguments["provider"] in {"chatgpt", "qq", "discord"}:
            accounts = project_accounts(await self._control.query("accounts.status", {}))
            if arguments["provider"] == "chatgpt":
                return accounts["chatgpt"]["signed_in"]
            managed = accounts["managed"]
            return (
                managed["credential_present"]
                if managed["provider"] == arguments["provider"]
                else None
            )
        name = arguments["name"] if command == "secrets.delete" else "openrouter_api_key"
        source = await self._control.query("secrets.presence", {})
        value = _mapping(source.get("effective_presence", source.get("presence"))).get(name)
        return value if type(value) is bool else None

    async def _apply_command(self, card: dict, require_current: Callable[[], None]) -> dict:
        command, arguments = card["command"], deepcopy(card["arguments"])
        request_id = uuid4().hex
        try:
            current = await self._control.query("settings.current", {})
            require_current()
            if current.get("revision") != card["revision"]:
                result = {"status": "rejected", "code": "revision_conflict"}
            else:
                receipt = await self._control.submit(
                    command, arguments, request_id=request_id, expected_revision=card["revision"]
                )
                if receipt.get("status") not in _TERMINAL:
                    receipt = await self._control.wait(receipt["operation_id"])
                safe = safe_receipt(receipt)
                result = {
                    "status": safe["status"] if safe.get("terminal") else "pending",
                    "receipt": safe,
                }
        except asyncio.CancelledError:
            raise
        except Exception:
            result = {"status": "failed", "code": "control_operation_unavailable"}
        result.update(command=command, changes=deepcopy(card["changes"]))
        self._record_action(
            {
                "tool": command,
                "action_id": request_id,
                "phase": "observed" if result["status"] == "pending" else "completed",
                "result": deepcopy(result),
            }
        )
        return result

    def _get_provider_setup(self, arguments: dict) -> dict:
        if (
            arguments.keys() != {"provider"}
            or type(arguments["provider"]) is not str
            or arguments["provider"] not in PROVIDER_SETUPS
        ):
            return _rejected("invalid_arguments", "Provide only a supported provider.")
        provider = arguments["provider"]
        return {
            "status": "available",
            **PROVIDER_SETUPS[provider].snapshot(),
            "link_rendered": False,
            "link_opened": False,
            "field_visibility": "not_observed",
            "completion_observable": False,
            "last_navigation": deepcopy(self._navigation_observations.get(provider)),
        }

    def _open_settings(self, arguments: dict) -> dict:
        if arguments.keys() - {"section", "provider"} or "section" not in arguments:
            return _rejected("invalid_arguments", "Provide only a section and optional provider.")
        section, provider = arguments["section"], arguments.get("provider")
        if type(section) is not str or section not in NAVIGATION_SECTIONS:
            return _rejected("invalid_arguments", "Unsupported settings section.")
        if provider is not None and (
            type(provider) is not str or provider not in NAVIGATION_PROVIDERS or section != "api"
        ):
            return _rejected("invalid_arguments", "Unsupported provider or settings section.")
        watching = (
            self._emit_event(
                {"kind": "navigation_requested", "section": section, "provider": provider}
            )
            is True
        )
        result = {
            "opened": False,
            "field_visible": False,
            "completion_observable": False,
            "verification_available": False,
        }
        if self._navigation is not None:
            try:
                observed = self._navigation(section, provider)
                result["opened"] = observed.get("opened") is True
                result["field_visible"] = result["opened"] and observed.get("field_visible") is True
                result["verification_available"] = (
                    result["field_visible"] and observed.get("verification_available") is True
                )
            except Exception:
                result.update(opened=False, field_visible=False, verification_available=False)
        result["status"] = "opened" if result["opened"] else "unavailable"
        result["completion_observable"] = watching
        if provider is not None:
            self._navigation_observations[provider] = {
                **_pick(
                    result,
                    (
                        "status",
                        "opened",
                        "field_visible",
                        "verification_available",
                        "completion_observable",
                    ),
                ),
                "source": "last_navigation",
            }
        return result

    async def close(self) -> None:
        self._closed = True
        self.cancel_confirmation()
        self.bind_navigation(None)
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        await self._scope.close()
