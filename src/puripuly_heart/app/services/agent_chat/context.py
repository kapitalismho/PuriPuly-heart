from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone

from puripuly_heart.app.control_presentation import SETTINGS_PRESENTATION
from puripuly_heart.app.ports.application_control import ApplicationControl
from puripuly_heart.app.ports.settings_secrets import SettingsSecretKey
from puripuly_heart.app.services.agent_chat.execution import (
    model_safe_catalog,
    project_audio_test,
    project_capture_channel,
    project_catalog,
    project_overlay_output,
)
from puripuly_heart.app.services.agent_chat.modules import TOPICS, topic_queries
from puripuly_heart.app.services.agent_chat.provider_setup import (
    ACCOUNT_SETUP_ACTIONS,
    PROVIDER_SETUPS,
)
from puripuly_heart.app.services.agent_chat.tools import (
    LANGUAGE_FIELD_PATHS,
    SETTINGS_FIELD_PATHS,
    project_accounts,
    project_osc_output,
    project_settings,
)
from puripuly_heart.app.services.application_audio_devices import safe_audio_device_label
from puripuly_heart.core.diagnostics import project_diagnostic_record

SETTINGS_PATHS = tuple(
    "intent." + path for path in (*SETTINGS_FIELD_PATHS.values(), *LANGUAGE_FIELD_PATHS.values())
)


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _pick(value: object, fields: tuple[str, ...]) -> dict:
    source = _mapping(value)
    return {
        key: source[key]
        for key in fields
        if key in source and (source[key] is None or type(source[key]) in (str, bool, int, float))
    }


def _settings(value: object) -> dict:
    return project_settings({"settings": value})["settings"]


def _audio_devices(value: dict | None, settings: Mapping) -> dict:
    result = {
        "current_route": {
            "input_host_api": settings.get("audio.input_host_api"),
            "input_device": settings.get("audio.input_device"),
        },
        "availability": "query_failed" if value is None else "available",
        "host_apis": [],
        "microphones": [],
    }
    if value is None:
        return result
    errors = _mapping(value.get("errors"))
    if "microphones" in errors:
        result["availability"] = "unavailable"
    result["unavailable_components"] = [
        key for key in ("microphones", "loopback_outputs") if key in errors
    ]
    hosts = value.get("host_apis")
    microphones = value.get("microphones")
    if "microphones" not in errors and (
        not isinstance(hosts, list) or not isinstance(microphones, list)
    ):
        result["availability"] = "unknown"
    if isinstance(hosts, list):
        result["host_apis"] = [
            safe_audio_device_label(item) for item in hosts[:32] if type(item) is str
        ]
    if isinstance(microphones, list):
        for item in microphones[:64]:
            source = _mapping(item)
            if type(source.get("name")) is not str or type(source.get("host_api")) is not str:
                continue
            name = safe_audio_device_label(source["name"])
            host = safe_audio_device_label(source["host_api"])
            result["microphones"].append(
                {
                    "input_device": name,
                    "input_host_api": host,
                    "selected_host_api": source.get("selected_host_api") is True,
                    "selectable": name == source["name"] and host == source["host_api"],
                }
            )
    result["truncated"] = (
        isinstance(hosts, list)
        and len(hosts) > 32
        or isinstance(microphones, list)
        and len(microphones) > 64
    )
    return result


def _audio_choices(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:64]:
        source = _mapping(item)
        if type(source.get("value")) is not str:
            continue
        label = safe_audio_device_label(source["value"])
        choice = {
            "value": label,
            "available": source.get("available") is True,
            "selectable": label == source["value"] and source.get("selectable") is not False,
        }
        if type(source.get("host_api")) is str:
            host = safe_audio_device_label(source["host_api"])
            choice["host_api"] = host
            choice["selectable"] = choice["selectable"] and host == source["host_api"]
        result.append(choice)
    return result


class AgentActivity:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.availability = "not_subscribed"
        self._generations: dict[str, object] = {}
        self._received: dict[str, dict[str, float]] = {"self": {}, "peer": {}}

    def clear(self) -> None:
        for values in self._received.values():
            values.clear()
        self._generations.clear()

    def record(self, event: dict) -> None:
        topic, channel = event.get("topic"), event.get("channel")
        if topic == "gap":
            self.clear()
            return
        if channel not in self._received:
            return
        if topic == "capture" and type(event.get("generation")) is int:
            generation = event["generation"]
            if self._generations.get(channel) != generation:
                self._received[channel].clear()
                self._generations[channel] = generation
        elif topic in {"transcript", "translation"}:
            self._received[channel][topic] = self._clock()

    def snapshot(self) -> dict:
        now = self._clock()
        return {
            channel: {
                f"{kind}_seconds_ago": (max(0, now - received[kind]) if kind in received else None)
                for kind in ("transcript", "translation")
            }
            for channel, received in self._received.items()
        }

    def event_facts(self) -> dict:
        return {
            channel: {
                "generation": self._generations.get(channel),
                **{
                    f"{kind}_observed_monotonic": received.get(kind)
                    for kind in ("transcript", "translation")
                },
            }
            for channel, received in self._received.items()
        }

    def observed_monotonic(self) -> float:
        return self._clock()


async def collect_agent_context(
    control: ApplicationControl,
    *,
    activity: AgentActivity,
    localize: Callable[[str], str] = lambda key: key,
    topic: str = "overview",
    diagnostic_id: str | None = None,
    operation_id: str | None = None,
    after: int | None = None,
    recent_agent_actions: tuple[dict, ...] = (),
    puri_allowance_exhausted: bool = False,
) -> dict:
    if topic not in TOPICS:
        raise ValueError("unsupported context topic")
    if (operation_id is not None or after is not None) and topic != "diagnostics":
        raise ValueError("diagnostic filters require the diagnostics topic")
    queries = topic_queries(topic)
    missing = []
    allowance = project_accounts(None)["puri"]["allowance"]
    if puri_allowance_exhausted is True:
        allowance["exhaustion"] = "observed"
    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "last_activity": activity.snapshot(),
        "activity_events": activity.event_facts(),
        "observed_monotonic": activity.observed_monotonic(),
        "activity_availability": activity.availability,
        "recent_agent_actions": list(recent_agent_actions),
        "missing": missing,
        "puri_allowance": allowance,
    }

    async def query(name: str, arguments: dict | None = None) -> dict | None:
        try:
            value = await control.query(name, arguments or {})
            if not isinstance(value, dict) or value.get("status") == "failed":
                raise ValueError("snapshot unavailable")
            return value
        except Exception:
            if name not in missing:
                missing.append(name)
            return None

    if "accounts.status" in queries:
        result["accounts"] = project_accounts(await query("accounts.status"))
        if puri_allowance_exhausted is True:
            result["accounts"]["puri"]["allowance"]["exhaustion"] = "observed"
        result["accounts"]["actions"] = [
            action.snapshot() for action in ACCOUNT_SETUP_ACTIONS.values()
        ]

    if "app.status" in queries:
        app = await query("app.status")
        if app is not None:
            result["app"] = _pick(
                app,
                (
                    "translation_enabled",
                    "translation_runtime_ready",
                    "overlay_target",
                    "stopping",
                ),
            )
            result["audio_test"] = project_audio_test(_mapping(app.get("audio_test")))
            result["channels"] = {}
            for channel in ("self", "peer"):
                source = _mapping(_mapping(app.get("channels")).get(channel))
                result["channels"][channel] = project_capture_channel(source)
    if "settings.current" in queries:
        settings = await query("settings.current")
        if settings is not None:
            result["revision"] = settings.get("revision")
            if not isinstance(settings.get("settings"), Mapping):
                missing.append("settings.current")
            else:
                result["settings"] = _settings(settings["settings"])
                result["local_settings"] = {
                    field: {
                        "presence": settings.get("private_presence", {}).get(field),
                        "model_editable": False,
                        "model_readable": False,
                    }
                    for field, descriptor in SETTINGS_PRESENTATION.items()
                    if descriptor.local_only
                }
                verification = _mapping(
                    _mapping(settings["settings"].get("state")).get("provider_verification")
                )
                result["provider_verification"] = {
                    provider: {
                        "status": (
                            status
                            if status in {"unknown", "verified", "failed", "skipped"}
                            else "unknown"
                        )
                    }
                    for provider, entry in verification.items()
                    if provider in PROVIDER_SETUPS
                    for status in (_mapping(entry).get("status"),)
                }
    if "settings.choices" in queries:
        choices = await query("settings.choices")
        if choices is not None:
            source = _mapping(model_safe_catalog(choices).get("choices"))
            allowed = set(SETTINGS_FIELD_PATHS) | {"languages", "peer_expected_languages"}
            result["choices"] = {
                key: (
                    _audio_choices(source[key])
                    if key in {"audio.input_host_api", "audio.input_device"}
                    else source[key]
                )
                for key in allowed
                if key in source and isinstance(source[key], (list, dict))
            }
            result["settings_catalog"] = project_catalog(choices, localize)
    if "secrets.presence" in queries:
        secrets = await query("secrets.presence")
        if secrets is not None:
            result["secret_presence"] = {
                key: value
                for key, value in _mapping(
                    secrets.get("effective_presence", secrets.get("presence"))
                ).items()
                if key in {item.value for item in SettingsSecretKey} and type(value) is bool
            }
    if "audio.devices" in queries:
        result["audio"] = _audio_devices(
            await query("audio.devices"), _mapping(result.get("settings"))
        )
    if "overlay.status" in queries:
        overlay = await query("overlay.status")
        osc = await query("osc.status")
        result["output"] = {
            "overlay": (
                {**project_overlay_output(overlay.get("output")), **_pick(overlay, ("locked",))}
                if overlay
                else None
            ),
            "osc": project_osc_output(osc.get("output")) if osc else None,
        }
    if "diagnostics" in queries:
        health = await query("diagnostics.health")
        filters = {
            key: value
            for key, value in (
                ("diagnostic_id", diagnostic_id),
                ("operation_id", operation_id),
                ("after", after),
            )
            if value is not None and topic == "diagnostics"
        }
        history = await query("diagnostics.history", {"limit": 10, **filters})
        if (
            not filters
            and history
            and history.get("truncated")
            and type(history.get("latest_sequence")) is int
        ):
            history = await query(
                "diagnostics.history",
                {
                    "limit": 10,
                    "after": max(0, history["latest_sequence"] - 10),
                },
            )
        records = list(reversed(history.get("records", []))) if history else []
        if diagnostic_id is not None and not filters:
            target = await query(
                "diagnostics.history", {"limit": 1, "diagnostic_id": diagnostic_id}
            )
            prioritized = list(target.get("records", [])) if target else []
            records = prioritized + [
                item for item in records if _mapping(item).get("diagnostic_id") != diagnostic_id
            ]
            result["requested_diagnostic"] = {
                "diagnostic_id": diagnostic_id,
                "lookup": target.get("lookup") if target else "query_failed",
            }
        result["recent_diagnostics"] = [
            project_diagnostic_record(item) for item in records[:10] if isinstance(item, Mapping)
        ]
        result["diagnostic_coverage"] = {
            "scope": "filtered_current_instance" if filters else "recent_current_instance",
            "filters": filters,
            "limit": 10,
            **_pick(
                history,
                (
                    "instance_id",
                    "lookup",
                    "oldest_sequence",
                    "latest_sequence",
                    "next_after",
                    "gap",
                    "evicted_records",
                    "truncated",
                ),
            ),
        }
        result["diagnostics_availability"] = {
            "history": history.get("availability", "unknown") if history else "query_failed",
            "health": (
                _mapping(health.get("history")).get(
                    "availability", health.get("availability", "unknown")
                )
                if health
                else "query_failed"
            ),
        }
        result["diagnostics_health"] = {
            "history": _pick(
                _mapping(health).get("history"),
                (
                    "retained_records",
                    "evicted_records",
                    "rejected_records",
                    "subscriber_delivery_failures",
                    "file_projection_failures",
                    "durable",
                    "restart_resets_history",
                ),
            ),
            "file": _pick(
                _mapping(health).get("file"),
                (
                    "availability",
                    "delivery_failures",
                    "dropped_records",
                    "writer_alive",
                ),
            ),
        }
    return result
