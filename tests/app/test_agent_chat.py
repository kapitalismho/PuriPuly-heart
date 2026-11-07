from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from puripuly_heart.app.adapters.agent_relay_auth import AgentRelayTokenOwner
from puripuly_heart.app.adapters.agent_relay_transport import AgentRelayTransport
from puripuly_heart.app.ports.agent_chat import AgentServiceError, AgentTransportError
from puripuly_heart.app.services.agent_chat.context import (
    TOPICS,
    AgentActivity,
    collect_agent_context,
)
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.app.services.application_control_events import ControlEvents
from puripuly_heart.app.wiring.wiring_agent_chat import compose_agent_transport
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.diagnostics import DiagnosticHistory
from puripuly_heart.core.managed_identity import (
    ManagedIdentityBundle,
    decode_base64url,
    encode_base64url,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation


class Tokens:
    async def get_token(self):
        return "synthetic-installation-token"

    def invalidate(self, token):
        raise AgentServiceError("service_authentication_failed")

    async def close(self):
        pass


class Records:
    def __init__(self):
        self.events = ControlEvents(instance_id="agent-test")
        self.queries = []
        self.subscriptions = []
        self.history = DiagnosticHistory()
        self.history.bind_instance("agent-test")
        self.fail = set()
        settings = to_dict(AppSettingsVNext())
        settings["intent"]["prompts"]["system_prompt_override"] = "PRIVATE-PROMPT"
        settings["intent"]["stt"]["custom_terms"] = {"en": ["PRIVATE-TERMS"]}
        settings["intent"]["local_llm"]["base_url"] = "http://PRIVATE-ENDPOINT"
        settings["state"]["managed_connection"]["installation_id"] = "PRIVATE-INSTALLATION"
        self.values = {
            "app.status": {
                "host_pid": 999,
                "translation_enabled": False,
                "translation_runtime_ready": True,
                "scene_observation": {"world": "PRIVATE-WORLD"},
                "channels": {
                    "self": {
                        "selected_provider": "local_cpu_auto",
                        "effective_active": True,
                        "capture": {
                            "generation": 3,
                            "has_source": True,
                            "device_name": "PRIVATE-DEVICE",
                        },
                    },
                    "peer": {"selected_provider": "local_cpu_auto", "effective_active": False},
                },
            },
            "settings.current": {"revision": 42, "settings": settings},
            "accounts.status": {},
            "audio.devices": {
                "host_apis": ["MME"],
                "microphones": [
                    {
                        "name": "Synthetic microphone",
                        "host_api": "MME",
                        "selected_host_api": True,
                        "raw_log": "PRIVATE-AUDIO-LOG",
                    }
                ],
                "errors": {},
            },
            "overlay.status": {
                "output": {
                    "lifecycle": "off",
                    "presentation_ready": False,
                    "caption": "PRIVATE-CAPTION",
                }
            },
            "osc.status": {
                "ports": [9000, 9001],
                "output": {
                    "local_availability": "off",
                    "configured_host": "PRIVATE-HOST",
                    "configured_send_port": 19000,
                    "configured_receive_port": 19001,
                    "effective_send_port": 9000,
                    "effective_receive_port": 49152,
                    "query_advertised_port": 49152,
                    "discovered_remote": {"host": "PRIVATE-REMOTE", "send_port": 19000},
                    "discovered_remote_present": True,
                    "remote_delivery": "unacknowledged",
                },
            },
            "secrets.presence": {
                "presence": {"openai_api_key": True, "custom_name": "PRIVATE-SECRET"}
            },
            "settings.choices": {
                "choices": {
                    "translation.model": ["gpt_6_luna"],
                    "translation.connection": {"gpt_6_luna": ["official_byok"]},
                    "languages": {"codes": ["ko", "en"], "peer_source_modes": ["manual", "auto"]},
                    "audio.input_device": [
                        {"value": "Synthetic microphone", "host_api": "MME", "available": True}
                    ],
                    "translation.http_extension_id": ["PRIVATE-EXTENSION"],
                }
            },
        }

    async def query(self, name, arguments):
        self.queries.append((name, arguments))
        if name in self.fail:
            raise RuntimeError("PRIVATE-EXCEPTION")
        if name == "diagnostics.history":
            return self.history.query(arguments)
        if name == "diagnostics.health":
            return self.history.health()
        return deepcopy(self.values[name])

    def subscribe(self, **kwargs):
        self.subscriptions.append(kwargs)
        return self.events.subscribe(channel=None, after=None, **kwargs)


class Responses:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.classifications = []
        self.closed = False

    async def prepare(self):
        pass

    async def classify(self, **request):
        self.classifications.append(deepcopy(request))
        return {
            "answers": [
                {"type": "choice", "name": "route", "choice": "work"},
                {"type": "choice", "name": "effort", "choice": "high"},
            ]
        }

    async def respond(self, **request):
        self.requests.append(deepcopy(request))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        yield {"type": "completed", "response": response}

    async def close(self):
        self.closed = True


def answer(text="Translation is off in the current records. Turn it on in the app."):
    return {
        "status": "completed",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


def context_of(request):
    return next(
        json.loads(item["content"])["app_context"]
        for item in reversed(request["input"])
        if isinstance(item.get("content"), str) and item["content"].startswith('{"app_context":')
    )


def tool(call_id="call_1", *, arguments='{"topic":"output"}', status="completed"):
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "get_app_context",
        "arguments": arguments,
        "status": status,
    }


@pytest.mark.asyncio
async def test_context_allowlist_recent_priority_and_initial_supported_choices():
    records = Records()
    ids = []
    for _ in range(15):
        diagnostic, _ = records.history.retain(
            ErrorDiagnostics(
                component="translation_request",
                operation="translate",
                code="provider_receive_failed",
                category="provider",
                visibility="basic",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                cause_id="a" * 32,
                fields={"status_code": 401, "raw_log": "PRIVATE-LOG", "api_key": "PRIVATE-KEY"},
            ),
            severity="error",
            disposition="failed",
        )
        ids.append(diagnostic["diagnostic_id"])
    context = await collect_agent_context(records, activity=AgentActivity(), diagnostic_id=ids[0])
    assert context["revision"] == 42
    assert context["app"]["translation_enabled"] is False
    assert context["channels"]["self"]["effective_active"] is True
    assert context["settings"]["languages.source_language"] == "ko"
    assert context["diagnostics_availability"] == {
        "history": "retained_current_run",
        "health": "retained_current_run",
    }
    assert context["recent_diagnostics"][0]["diagnostic_id"] == ids[0]
    assert context["recent_diagnostics"][1]["diagnostic_id"] == ids[14]
    assert context["recent_diagnostics"][-1]["diagnostic_id"] == ids[6]
    assert len(context["recent_diagnostics"]) == 10
    assert context["recent_diagnostics"][0]["fields"] == {"status_code": 401}
    assert context["recent_diagnostics"][0]["code"] == "provider_receive_failed"
    assert context["recent_diagnostics"][0]["cause_id"] == "a" * 32
    assert context["last_activity"]["self"]["transcript_seconds_ago"] is None
    assert context["choices"]["translation.connection"] == {"gpt_6_luna": ["official_byok"]}
    assert context["choices"]["languages"]["codes"] == ["ko", "en"]
    assert context["choices"]["audio.input_device"] == [
        {
            "value": "Synthetic microphone",
            "host_api": "MME",
            "available": True,
            "selectable": True,
        }
    ]
    assert context["secret_presence"] == {"openai_api_key": True}
    assert context["audio"]["availability"] == "available"
    assert context["audio"]["microphones"] == [
        {
            "input_device": "Synthetic microphone",
            "input_host_api": "MME",
            "selected_host_api": True,
            "selectable": True,
        }
    ]
    assert context["accounts"]["chatgpt"]["signed_in"] is None
    assert context["accounts"]["managed"]["credential_present"] is None
    assert all(action["requires_user_click"] for action in context["accounts"]["actions"])
    for topic in TOPICS:
        projected = await collect_agent_context(records, activity=AgentActivity(), topic=topic)
        assert "PRIVATE-" not in json.dumps(projected)
    output = await collect_agent_context(records, activity=AgentActivity(), topic="output")
    assert output["output"]["osc"] == {
        "local_availability": "off",
        "discovered_remote_present": True,
        "remote_delivery": "unacknowledged",
    }
    settings = await collect_agent_context(records, activity=AgentActivity(), topic="settings")
    assert settings["choices"]["translation.connection"] == {"gpt_6_luna": ["official_byok"]}
    assert settings["choices"] == context["choices"]
    assert settings["choices"]["languages"]["codes"] == ["ko", "en"]
    translation = await collect_agent_context(
        records, activity=AgentActivity(), topic="translation"
    )
    assert translation["secret_presence"] == {"openai_api_key": True}
    assert translation["provider_verification"] == context["provider_verification"]
    assert translation["secret_presence"] == context["secret_presence"]
    with pytest.raises(ValueError, match="unsupported"):
        await collect_agent_context(records, activity=AgentActivity(), topic="raw_logs")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    (
        "unknown",
        "not_applicable",
        "matched",
        "other_process",
        "not_listening",
        "vrchat_not_running",
    ),
)
async def test_output_context_projects_retained_listener_without_active_observation(status):
    records = Records()
    records.values["osc.status"]["output"].update(
        {
            "destination_source": "fallback",
            "local_listener": {
                "status": status,
                "vrchat_running": None,
                "observed_at": 13.5,
                "pid": 12345,
                "host": "PRIVATE-HOST",
                "raw_log": "PRIVATE-LOG",
                "unexpected": {"message": "PRIVATE-PROSE"},
            },
        }
    )
    context = await collect_agent_context(records, activity=AgentActivity(), topic="output")
    assert records.queries == [("overlay.status", {}), ("osc.status", {})]
    assert context["output"]["osc"]["destination_source"] == "fallback"
    assert context["output"]["osc"]["local_listener"] == {
        "status": status,
        "vrchat_running": None,
        "observed_at": 13.5,
    }
    assert context["output"]["osc"]["remote_delivery"] == "unacknowledged"
    assert "PRIVATE-" not in json.dumps(context)


@pytest.mark.asyncio
async def test_output_context_rejects_unexpected_nested_evidence_and_prose():
    records = Records()
    records.values["osc.status"]["output"].update(
        {
            "configured_mode": {"host": "PRIVATE-HOST"},
            "query_failure": "PRIVATE-EXCEPTION",
            "destination_source": "PRIVATE-DESTINATION",
            "remote_delivery": "PRIVATE-CLAIM",
            "local_listener": {
                "status": {"raw_log": "PRIVATE-LOG"},
                "vrchat_running": "PRIVATE-PROSE",
                "observed_at": {"port": 19000},
            },
        }
    )
    context = await collect_agent_context(records, activity=AgentActivity(), topic="output")
    assert context["output"]["osc"]["local_listener"] == {
        "status": "unknown",
        "vrchat_running": None,
    }
    assert context["output"]["osc"]["remote_delivery"] == "unacknowledged"
    assert "PRIVATE-" not in json.dumps(context)


@pytest.mark.asyncio
async def test_empty_unavailable_and_failed_diagnostics_remain_distinct():
    records = Records()
    empty = await collect_agent_context(records, activity=AgentActivity())
    assert empty["recent_diagnostics"] == []
    assert empty["diagnostics_availability"]["history"] == "retained_current_run"
    records.history.instance_id = None
    unavailable = await collect_agent_context(records, activity=AgentActivity())
    assert unavailable["diagnostics_availability"] == {
        "history": "unavailable",
        "health": "unbound",
    }
    records.fail = {"diagnostics.history", "diagnostics.health", "settings.current"}
    failed = await collect_agent_context(records, activity=AgentActivity())
    assert failed["diagnostics_availability"] == {
        "history": "query_failed",
        "health": "query_failed",
    }
    assert set(failed["missing"]) == records.fail
    assert failed["app"]["translation_enabled"] is False
    assert "PRIVATE-" not in json.dumps(failed)


@pytest.mark.asyncio
async def test_targeted_diagnostics_find_older_evidence_and_expose_paging_and_eviction():
    records = Records()
    operation_id = "b" * 32
    related_ids = []
    for index in range(35):
        diagnostic, _ = records.history.retain(
            ErrorDiagnostics(
                component="translation_request",
                operation="translate",
                code="provider_receive_failed",
                category="provider",
                visibility="basic",
                content_policy="metadata_only",
                operation_id=operation_id if index < 13 else "c" * 32,
                status_code=401 if index < 13 else 503,
                retry_after_ms=None,
                fields={},
            ),
            severity="error",
            disposition="failed",
        )
        if index < 13:
            related_ids.append(diagnostic["diagnostic_id"])
    recent = await collect_agent_context(records, activity=AgentActivity())
    assert not set(related_ids) & {item["diagnostic_id"] for item in recent["recent_diagnostics"]}
    transport = Responses(
        {
            "status": "completed",
            "output": [
                tool(
                    "history",
                    arguments=json.dumps(
                        {
                            "topic": "diagnostics",
                            "diagnostic_id": None,
                            "operation_id": operation_id,
                            "after": None,
                        }
                    ),
                )
            ],
        },
        answer("The requested operation has retained authentication failures."),
    )
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        outcome = await owner.send("Inspect the failed operation.")
        first = outcome.tool_results[0].result["context"]
        assert {item["diagnostic_id"] for item in first["recent_diagnostics"]} == set(
            related_ids[:10]
        )
        assert first["diagnostic_coverage"]["truncated"] is True
        second = await collect_agent_context(
            records,
            activity=AgentActivity(),
            topic="diagnostics",
            operation_id=operation_id,
            after=first["diagnostic_coverage"]["next_after"],
        )
        assert {item["diagnostic_id"] for item in second["recent_diagnostics"]} == set(
            related_ids[10:]
        )
        assert second["diagnostic_coverage"]["truncated"] is False
        for _ in range(records.history.max_records):
            records.history.retain(
                ErrorDiagnostics(
                    component="translation_request",
                    operation="translate",
                    code="provider_receive_failed",
                    category="provider",
                    visibility="basic",
                    content_policy="metadata_only",
                    status_code=None,
                    retry_after_ms=None,
                    fields={},
                ),
                severity="error",
                disposition="failed",
            )
        expired = await collect_agent_context(
            records,
            activity=AgentActivity(),
            topic="diagnostics",
            diagnostic_id=related_ids[0],
        )
        assert expired["recent_diagnostics"] == []
        assert expired["diagnostic_coverage"]["lookup"] == "not_retained"
        assert expired["diagnostic_coverage"]["gap"] is True
        assert expired["diagnostic_coverage"]["evicted_records"] > 0
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_text_free_subscription_generations_and_idle_make_no_model_calls():
    records, transport = Records(), Responses(answer())
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        assert records.subscriptions == [
            {
                "topics": ["transcript", "translation", "capture"],
                "include_transcripts": False,
                "include_translations": False,
            }
        ]
        records.events.publish({"topic": "capture", "channel": "self", "generation": 1})
        records.events.publish_ui(
            UIEvent(
                UIEventType.TRANSCRIPT_PARTIAL,
                channel="self",
                payload=Transcript(uuid4(), "PRIVATE-SPEECH", is_final=False),
            )
        )
        records.events.publish_ui(
            UIEvent(
                UIEventType.TRANSLATION_DONE,
                channel="peer",
                payload=Translation(uuid4(), "PRIVATE-TRANSLATION"),
            )
        )
        await asyncio.sleep(0)
        assert not transport.requests
        await owner.send("번역이 안 떠")
        context = context_of(transport.requests[0])
        assert context["last_activity"]["self"]["transcript_seconds_ago"] is not None
        assert context["last_activity"]["peer"]["translation_seconds_ago"] is not None
        assert "PRIVATE-" not in json.dumps(transport.requests)
        records.events.publish({"topic": "capture", "channel": "self", "generation": 2})
        await asyncio.sleep(0)
        snapshot = owner._activity.snapshot()
        assert snapshot["self"]["transcript_seconds_ago"] is None
        assert snapshot["peer"]["translation_seconds_ago"] is not None
        owner._activity.record({"topic": "gap"})
        assert owner._activity.snapshot()["peer"]["translation_seconds_ago"] is None
        assert len(transport.requests) == 1
    finally:
        await owner.close()
    assert not records.events.subscribers
    assert transport.closed


@pytest.mark.asyncio
async def test_completed_continuation_preserves_encrypted_reasoning_and_pairs():
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"}
    output = [reasoning, tool()]
    transport = Responses({"status": "completed", "output": output}, answer(), answer("Follow-up"))
    records = Records()
    owner = AgentChatOwner(records, transport, localize=str)
    phases = []
    owner.observe(lambda snapshot: phases.append((snapshot.phase, snapshot.busy)))
    await owner.start()
    phases.clear()
    try:
        first = await owner.send("번역이 안 떠")
        assert first.status == "completed"
        assert first.tool_results[0].status == "observed"
        assert phases[0] == ("thinking", True)
        continuation = transport.requests[1]["input"]
        reasoning_index = continuation.index(reasoning)
        assert continuation[reasoning_index + 1] == output[1]
        result = next(item for item in continuation if item.get("type") == "function_call_output")
        assert result["call_id"] == output[1]["call_id"]
        assert (
            json.loads(result["output"])["context"]["output"]["osc"]["remote_delivery"]
            == "unacknowledged"
        )
        await owner.send("그 다음은?")
        followup = transport.requests[2]["input"]
        assert reasoning in followup
        assert all(request["effort"] == "high" for request in transport.requests)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_history_retains_prior_evidence_when_fresh_query_fails():
    records = Records()
    diagnostic, _ = records.history.retain(
        ErrorDiagnostics(
            component="translation_request",
            operation="translate",
            code="provider_receive_failed",
            category="provider",
            visibility="basic",
            content_policy="metadata_only",
            status_code=401,
            retry_after_ms=None,
            cause_id="a" * 32,
            fields={"status_code": 401, "raw_log": "PRIVATE-LOG"},
        ),
        severity="error",
        disposition="failed",
    )
    transport = Responses(
        {
            "status": "completed",
            "output": [tool("settings", arguments='{"topic":"settings"}')],
        },
        {
            "status": "completed",
            "output": [tool("diagnostics", arguments='{"topic":"diagnostics"}')],
        },
        answer("The retained diagnostic reports a provider failure."),
        answer("The latest diagnostic query failed; the earlier record is historical."),
    )
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        first = await owner.send("Check translation and supported settings.")
        assert first.status == "completed"
        assert [item.topic for item in first.tool_results] == ["settings", "diagnostics"]
        continuation = transport.requests[2]["input"]
        outputs = [item for item in continuation if item.get("type") == "function_call_output"]
        assert [item["call_id"] for item in outputs] == ["settings", "diagnostics"]
        assert (
            first.tool_results[1].result["context"]["recent_diagnostics"][0]["diagnostic_id"]
            == diagnostic["diagnostic_id"]
        )
        records.values["app.status"]["translation_enabled"] = True
        records.values["settings.current"]["revision"] = 43
        records.fail.add("diagnostics.history")
        assert (await owner.send("I enabled translation. Check again.")).status == "completed"
        followup = transport.requests[-1]["input"]
        assert followup[: len(continuation)] == continuation
        snapshot_items = [
            item
            for item in followup
            if isinstance(item.get("content"), str)
            and item["content"].startswith('{"app_context":')
        ]
        snapshots = [json.loads(item["content"])["app_context"] for item in snapshot_items]
        assert [item["revision"] for item in snapshots] == [42, 43]
        assert [item["app"]["translation_enabled"] for item in snapshots] == [False, True]
        assert snapshots[0]["recent_diagnostics"][0]["diagnostic_id"] == diagnostic["diagnostic_id"]
        assert snapshots[1]["recent_diagnostics"] == []
        assert snapshots[1]["diagnostics_availability"]["history"] == "query_failed"
        assert snapshots[1]["missing"] == ["diagnostics.history"]
        assert owner._tools._context["revision"] == 43
        assert owner._tools._context["settings"] == snapshots[1]["settings"]
        assert [
            item["reasoning"]["effort"]
            for item in followup
            if item.get("type") == "configuration_update"
        ] == ["high", "high"]
        assert "PRIVATE-" not in json.dumps(transport.requests)
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [tool(arguments='{"topic":', status="in_progress")],
        },
        {"status": "incomplete", "output": [tool()]},
        {"status": "completed", "output": [tool(status="in_progress")]},
    ],
)
async def test_interrupted_responses_never_execute_tools_or_retain_uncommitted_output(response):
    records, transport = Records(), Responses(response, answer())
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Check output")
        assert result.status == "interrupted"
        assert "interrupted" in result.text
        assert len(records.queries) == 10
        await owner.send("Continue")
        items = transport.requests[1]["input"]
        calls = [item for item in items if item.get("type") == "function_call"]
        outputs = [item for item in items if item.get("type") == "function_call_output"]
        assert calls == outputs == []
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_bounded_serial_tools_summarize_then_finish_and_keep_facts_on_relay_failure():
    transport = Responses(
        *[{"status": "completed", "output": [tool(f"call_{index}")]} for index in range(10)],
        answer("The existing facts are enough."),
    )
    owner = AgentChatOwner(Records(), transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Check output")
        assert result.status == "completed"
        assert len(result.tool_results) == 10
        assert [request["allow_tools"] for request in transport.requests] == [True] * 10 + [False]
    finally:
        await owner.close()
    failing = Responses({"status": "completed", "output": [tool()]}, AgentTransportError())
    owner = AgentChatOwner(Records(), failing, localize=str)
    await owner.start()
    try:
        result = await owner.send("Check output")
        assert result.status == "failed"
        assert result.tool_results[0].result["context"]["output"]["overlay"]["lifecycle"] == "off"
        assert owner.snapshot.tool_results == result.tool_results
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_duplicate_call_id_does_not_execute_or_break_followup_pairing():
    records = Records()
    transport = Responses(
        {"status": "completed", "output": [tool()]},
        {"status": "completed", "output": [tool()]},
        answer(),
    )
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Check output")
        assert result.status == "interrupted"
        assert len(result.tool_results) == 1
        assert len(records.queries) == 12
        await owner.send("Continue")
        items = transport.requests[-1]["input"]
        calls = [item["call_id"] for item in items if item.get("type") == "function_call"]
        outputs = [item["call_id"] for item in items if item.get("type") == "function_call_output"]
        assert calls == outputs == ["call_1"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_invalid_singleton_arguments_are_rejected_without_reading():
    records = Records()
    transport = Responses(
        {"status": "completed", "output": [tool(arguments='{"topic":')]},
        answer(),
    )
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Check output")
        assert result.status == "completed"
        assert len(records.queries) == 10
        assert len(result.tool_results) == 1
        assert result.tool_results[0].result == {"status": "rejected", "code": "invalid_tool_call"}
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("account_topic", ["accounts", "overview"])
async def test_parallel_context_reads_wait_for_all_and_publish_in_call_order(
    monkeypatch, account_topic
):
    output_entered, accounts_completed, release = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    class HeldRecords(Records):
        async def query(self, name, arguments):
            result = await super().query(name, arguments)
            if owner.snapshot.phase == "working":
                if name == "overlay.status":
                    output_entered.set()
                    await release.wait()
                elif name == "accounts.status":
                    accounts_completed.set()
            return result

    records = HeldRecords()
    records.values["accounts.status"] = {}
    calls = [
        tool("output-read"),
        tool("accounts-read", arguments=json.dumps({"topic": account_topic})),
    ]
    transport = Responses({"status": "completed", "output": calls}, answer())
    owner = AgentChatOwner(records, transport, localize=str)
    observed = []
    observe_context = owner._tools.observe_context

    def observe(context):
        observed.append(deepcopy(context))
        observe_context(context)

    monkeypatch.setattr(owner._tools, "observe_context", observe)
    await owner.start()
    sending = asyncio.create_task(owner.send("Check output and accounts"))
    try:
        await asyncio.wait_for(output_entered.wait(), 1)
        await asyncio.wait_for(accounts_completed.wait(), 1)
        assert not sending.done()
        assert not owner.snapshot.tool_results
        assert not owner.snapshot.setup_actions
        assert len(observed) == 1
        assert observed[0]["accounts"]["actions"]
        assert all(action["requires_user_click"] for action in observed[0]["accounts"]["actions"])
        release.set()
        result = await sending
        assert result.status == "completed"
        assert [item.call_id for item in result.tool_results] == ["output-read", "accounts-read"]
        assert [item.topic for item in result.tool_results] == ["output", account_topic]
        contexts = [item.result["context"] for item in result.tool_results]
        assert contexts[0]["output"]["overlay"]["lifecycle"] == "off"
        assert contexts[1]["accounts"] == observed[0]["accounts"]
        assert observed[1:] == contexts
        assert bool(owner.snapshot.setup_actions) is (account_topic == "accounts")
        outputs = [
            item
            for item in transport.requests[1]["input"]
            if item.get("type") == "function_call_output"
        ]
        assert [item["call_id"] for item in outputs] == ["output-read", "accounts-read"]
        assert [json.loads(item["output"]) for item in outputs] == [
            item.result for item in result.tool_results
        ]
    finally:
        release.set()
        await owner.close()
        await asyncio.gather(sending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        '{"topic":',
        "[]",
        '{"topic":"raw_logs"}',
        '{"topic":"output","after":0}',
        '{"topic":"diagnostics","after":true}',
        '{"topic":"diagnostics","operation_id":""}',
        '{"topic":"output","unexpected":true}',
    ],
)
@pytest.mark.parametrize("invalid_first", [False, True])
async def test_invalid_context_batch_arguments_reject_every_member(arguments, invalid_first):
    invalid = tool("invalid", arguments=arguments)
    valid = tool("valid")
    calls = [invalid, valid] if invalid_first else [valid, invalid]
    records, transport = Records(), Responses({"status": "completed", "output": calls})
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Check output")
        assert result.status == "interrupted"
        assert len(records.queries) == 10
        assert [item.call_id for item in result.tool_results] == [call["call_id"] for call in calls]
        assert [item.result for item in result.tool_results] == [
            {"status": "rejected", "code": "invalid_tool_call"}
        ] * 2
        assert len(transport.requests) == 1
        assert not owner.snapshot.setup_actions
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("change_settings", {"changes": {"overlay.show_translation": False}}),
        ("withdraw_confirmation", {"card_id": "pending-card"}),
        ("open_settings", {"section": "api"}),
        ("probe_translation", {}),
        ("probe_self_recognition", {}),
        ("probe_peer_recognition", {}),
        ("propose_report", {"summary": "Check the app"}),
        ("set_capture", {"channel": "self", "enabled": True}),
        ("set_translation", {"enabled": True}),
        ("diagnose_microphone", {}),
        ("get_provider_setup", {"provider": "openrouter"}),
        ("retry_osc_discovery", {}),
        ("unsupported_tool", {}),
        (None, {}),
        ([], {}),
        ({}, {}),
        (123, {}),
    ],
)
@pytest.mark.parametrize("non_read_first", [False, True])
async def test_mixed_context_batches_never_execute_or_stage_actions(
    name, arguments, non_read_first
):
    non_read = {**tool("non-read"), "arguments": json.dumps(arguments)}
    if name is None:
        del non_read["name"]
    else:
        non_read["name"] = name
    read = tool("read")
    calls = [non_read, read] if non_read_first else [read, non_read]
    records, transport = Records(), Responses({"status": "completed", "output": calls})
    owner = AgentChatOwner(records, transport, localize=str)
    navigations = []
    owner.bind_navigation(lambda *args: navigations.append(args) or {"opened": True})
    await owner.start()
    try:
        result = await owner.send("Check output and configure the app")
        assert result.status == "interrupted"
        assert len(records.queries) == 10
        assert [item.status for item in result.tool_results] == ["rejected"] * 2
        assert not navigations
        assert not owner.snapshot.setup_actions
        assert not owner.snapshot.secure_entry_requests
        assert not owner.snapshot.report_offers
        assert not owner.snapshot.recent_changes
        assert owner.snapshot.confirmation is None
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_four_read_batches_share_the_ten_call_turn_budget():
    calls = [tool(f"read-{index}") for index in range(10)]
    transport = Responses(
        *[
            {"status": "completed", "output": calls[start:end]}
            for start, end in ((0, 4), (4, 8), (8, 10))
        ],
        answer("The existing facts are enough."),
    )
    records = Records()
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Read current app facts")
        assert result.status == "completed"
        assert len(records.queries) == 30
        assert [item.call_id for item in result.tool_results] == [call["call_id"] for call in calls]
        assert [item.status for item in result.tool_results] == ["observed"] * 10
        assert [request["allow_tools"] for request in transport.requests] == [
            True,
            True,
            True,
            False,
        ]
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("prior_count", "batch_size"), [(0, 5), (7, 4), (9, 2), (10, 2)])
async def test_oversized_or_overbudget_batches_execute_no_members(prior_count, batch_size):
    batch = [tool(f"batch-{index}") for index in range(batch_size)]
    transport = Responses(
        *[
            {"status": "completed", "output": [tool(f"prior-{index}")]}
            for index in range(prior_count)
        ],
        {"status": "completed", "output": batch},
    )
    records = Records()
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Read current app facts")
        assert result.status == "interrupted"
        assert len(records.queries) == 10 + prior_count * 2
        assert [item.status for item in result.tool_results] == (
            ["observed"] * prior_count + ["rejected"] * batch_size
        )
        assert [item.result["code"] for item in result.tool_results[-batch_size:]] == [
            "tool_limit"
        ] * batch_size
        assert not owner.snapshot.setup_actions
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["read", "", None, 123, []])
async def test_invalid_batch_call_ids_never_execute_or_enter_followup_history(bad_id):
    calls = [tool("read"), {**tool("invalid"), "call_id": bad_id}]
    records = Records()
    transport = Responses({"status": "completed", "output": calls}, answer())
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Check output")
        assert result.status == "interrupted"
        assert not result.tool_results
        assert len(records.queries) == 10
        await owner.send("Continue")
        assert not [
            item
            for item in transport.requests[1]["input"]
            if item.get("type") in {"function_call", "function_call_output"}
        ]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_batch_cannot_reuse_a_prior_call_id_or_execute_its_valid_sibling():
    records = Records()
    transport = Responses(
        {"status": "completed", "output": [tool("prior")]},
        {"status": "completed", "output": [tool("new"), tool("prior")]},
        answer(),
    )
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Check output")
        assert result.status == "interrupted"
        assert len(records.queries) == 12
        assert [item.call_id for item in result.tool_results] == ["prior"]
        await owner.send("Continue")
        items = transport.requests[-1]["input"]
        assert [item["call_id"] for item in items if item.get("type") == "function_call"] == [
            "prior"
        ]
        assert [
            item["call_id"] for item in items if item.get("type") == "function_call_output"
        ] == ["prior"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_one_active_turn_shutdown_rejects_late_response_and_drains():
    entered, release = asyncio.Event(), asyncio.Event()

    class LateResponse(Responses):
        async def respond(self, **request):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            yield {"type": "completed", "response": {"status": "completed", "output": [tool()]}}

    records, transport = Records(), LateResponse()
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    active = asyncio.create_task(owner.send("Check output"))
    await entered.wait()
    owner.freeze_ingress()
    release.set()
    await owner.close()
    assert active.cancelled()
    assert len(records.queries) == 10
    assert transport.closed
    assert not owner.snapshot.busy
    assert (await owner.send("Later")).status == "unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "high", "xhigh"])
async def test_transport_rejects_insecure_routes_authenticates_and_sends_only_fixed_interface(
    monkeypatch, effort
):
    monkeypatch.setenv("PURIPULY_AGENT_RELAY_URL", "")
    assert compose_agent_transport(settings=None, config_path=None) is None
    with pytest.raises(ValueError):
        AgentRelayTransport("http://127.0.0.1:8788/v1/respond", Tokens())
    secure = AgentRelayTransport("https://relay.example/v1/respond", Tokens())
    await secure.close()
    for endpoint in (
        "http://example.com/v1/respond",
        "http://localhost/v1/respond",
        "http://127.0.0.1/v1/respond?api_key=private",
        "http://user:pw@127.0.0.1/v1/respond",
    ):
        with pytest.raises(ValueError):
            AgentRelayTransport(endpoint, Tokens(), allow_loopback_http=True)
    monkeypatch.delenv("PURIPULY_AGENT_RELAY_URL")
    monkeypatch.setattr(
        "puripuly_heart.app.wiring.wiring_agent_chat.current_runtime_layout",
        lambda: SimpleNamespace(host_kind="native"),
    )
    monkeypatch.setenv("PURIPULY_AGENT_RELAY_URL", "http://127.0.0.1:8788/v1/respond")
    with pytest.raises(ValueError):
        compose_agent_transport(settings=None, config_path=None)
    transport = AgentRelayTransport(
        "http://127.0.0.1:8788/v1/respond", Tokens(), allow_loopback_http=True
    )
    await transport._client.aclose()
    seen = []
    response = answer()

    def handle(request):
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: " + json.dumps({"type": "completed", "response": response}) + "\n\n",
        )

    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    inputs = [
        {"type": "configuration_update", "reasoning": {"effort": effort}},
        {"role": "user", "content": "synthetic"},
    ]
    try:
        events = [
            event
            async for event in transport.respond(
                input=inputs,
                turn_id="0123456789abcdef0123456789abcdef",
                effort=effort,
                route="work",
                allow_tools=False,
            )
        ]
        assert events == [{"type": "completed", "response": response}]
        assert json.loads(seen[0].content) == {
            "agent_protocol": 4,
            "input": inputs,
            "turn_id": "0123456789abcdef0123456789abcdef",
            "effort": effort,
            "route": "work",
            "allow_tools": False,
        }
        assert seen[0].headers["authorization"] == "Bearer synthetic-installation-token"
        for invalid in ("medium", "extreme"):
            with pytest.raises(AgentTransportError, match="invalid_effort"):
                await anext(
                    transport.respond(
                        input=inputs,
                        turn_id="0123456789abcdef0123456789abcdef",
                        effort=invalid,
                        route="work",
                        allow_tools=True,
                    )
                )
        assert len(seen) == 1
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_composed_owner_read_only_turn_and_agent_chat_view(tmp_path, monkeypatch):
    from puripuly_heart.composition.headless_application import compose_headless_application
    from puripuly_heart.ui.views.agent_chat import AgentChatView

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "synthetic-agent-test-passphrase")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("PURIPULY_AGENT_RELAY_URL", "http://127.0.0.1:18831/v1/respond")
    monkeypatch.setenv("PURIPULY_AGENT_BROKER_URL", "http://127.0.0.1:18830")
    proofs = []
    transport = Responses(answer())
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(settings.intent.osc, connection_mode="off"),
            telemetry=replace(settings.intent.telemetry, enabled=False),
            integrated_context=replace(settings.intent.integrated_context, enabled=False),
        ),
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    app = compose_headless_application(path)
    owner = app.agent_chat()
    assert owner is not None
    actual_transport = owner._transport

    async def wire(request):
        body = json.loads(request.content)
        if request.url.path == "/v1/puri/token":
            proofs.append(body)
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic-installation-token",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "expires_at": (datetime.now(UTC) + timedelta(seconds=3600)).isoformat(),
                },
            )
        assert request.headers["authorization"] == "Bearer synthetic-installation-token"
        assert body.pop("agent_protocol") == 4
        if request.url.path == "/v1/classify":
            return httpx.Response(200, json=await transport.classify(**body))
        frames = [json.dumps(event) async for event in transport.respond(**body)]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + event + "\n\n" for event in frames),
        )

    await actual_transport._client.aclose()
    actual_transport._client = httpx.AsyncClient(transport=httpx.MockTransport(wire))
    actual_transport._token_owner._client = httpx.AsyncClient(transport=httpx.MockTransport(wire))
    try:
        app.control().bind_instance("agent-composed-smoke")
        await app.start()
        owner = app.agent_chat()
        assert owner is not None
        before = await app.control().query("settings.current", {})
        view = AgentChatView(owner)
        assert not view._input.disabled and not view._voice_tile.disabled
        view._input.value = "번역이 안 떠"
        await view._submit(None)
        assert owner.snapshot.messages[-1].role == "assistant"
        assert not owner.snapshot.busy
        assert not view._input.disabled and view._input.value == ""
        context = context_of(transport.requests[0])
        current = await app.control().query("settings.current", {})
        assert context["revision"] == current["revision"]
        assert context["app"]["translation_enabled"] is False
        assert context["channels"]["self"]["selected_provider"] == "local_cpu_auto"
        assert context["activity_availability"] == "subscribed"
        assert current["settings"]["intent"] == before["settings"]["intent"]
        assert len(proofs) == 1
        assert (
            proofs[0]["installation_id"]
            == current["settings"]["state"]["managed_connection"]["installation_id"]
        )
        assert len(transport.requests) == 1
    finally:
        await app.stop()
    assert actual_transport._client.is_closed
    assert actual_transport._token_owner._token is None
    assert owner.snapshot.phase == "closed"


@pytest.mark.asyncio
async def test_serial_context_tools_reject_eleventh_call_without_executing_it():
    transport = Responses(
        *[{"status": "completed", "output": [tool(str(index))]} for index in range(11)]
    )
    records = Records()
    owner = AgentChatOwner(records, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Read current app facts")
        assert result.status == "interrupted"
        assert len(records.queries) == 30
        assert [item.status for item in result.tool_results] == ["observed"] * 10 + ["rejected"]
        assert result.tool_results[-1].result["code"] == "tool_limit"
        assert [request["allow_tools"] for request in transport.requests] == [True] * 10 + [False]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_first_send_and_new_conversation_keep_known_keys_before_auth_context_and_archive():
    entered, release = asyncio.Event(), asyncio.Event()

    class Prepared(Responses):
        async def prepare(self):
            entered.set()
            await release.wait()

    records, transport = Records(), Prepared(answer(), answer())
    owner = AgentChatOwner(records, transport, localize=str, decision_timeout=0.01)
    await owner.start()
    try:
        secret = "sk-syntheticSecret123456789"
        assert (await owner.send(f"Please use {secret}")).status == "rejected"
        assert secret not in json.dumps(owner.report_archive)
        assert secret not in json.dumps(owner._items)
        assert not entered.is_set() and not records.queries
        assert not transport.classifications and not transport.requests
        active = asyncio.create_task(owner.send("How do API keys work?"))
        await entered.wait()
        await asyncio.sleep(0.03)
        assert not records.queries and not transport.classifications and not transport.requests
        release.set()
        assert (await active).status == "completed"
        observed = owner.observations[-1]
        assert observed.model_calls == 2
        assert observed.decision_seconds is not None
        assert observed.first_useful_seconds >= observed.decision_seconds
        assert len(transport.classifications) == 1
        owner.new_conversation()
        assert (await owner.send("Try again")).status == "completed"
        assert len(transport.classifications) == len(transport.requests) == 2
        assert owner.snapshot.phase == "idle"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_token_owner_shares_signing_initialization_and_refreshes_only_in_memory():
    key = Ed25519PrivateKey.generate()
    bundle = ManagedIdentityBundle(
        installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        device_public_key=encode_base64url(key.public_key().public_bytes_raw()),
        _private_key=key,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    nonces = []
    now = [0.0]

    async def identity():
        entered.set()
        await release.wait()
        return bundle

    def issue(request):
        body = json.loads(request.content)
        assert set(body) == {
            "installation_id",
            "device_public_key",
            "app_version",
            "signed_at",
            "nonce",
            "signature_alg",
            "signature",
        }
        assert request.url.path == "/v1/puri/token"
        assert body["signature_alg"] == "ed25519"
        assert len(decode_base64url(body["nonce"])) == 32
        canonical = "\n".join(
            [
                "puripuly-heart:puri-token:v1",
                "POST",
                "/v1/puri/token",
                "puripuly-agent-relay",
                bundle.installation_id,
                bundle.device_public_key,
                "synthetic-version",
                body["signed_at"],
                body["nonce"],
            ]
        ).encode()
        key.public_key().verify(decode_base64url(body["signature"]), canonical)
        nonces.append(body["nonce"])
        return httpx.Response(
            200,
            json={
                "access_token": f"synthetic.token.{len(nonces)}",
                "token_type": "Bearer",
                "expires_in": 3600,
                "expires_at": (datetime.now(UTC) + timedelta(seconds=3600)).isoformat(),
            },
        )

    owner = AgentRelayTokenOwner(
        lambda: "https://broker.example",
        identity,
        app_version="synthetic-version",
        clock=lambda: now[0],
    )
    owner._client = httpx.AsyncClient(transport=httpx.MockTransport(issue))
    first = asyncio.create_task(owner.get_token())
    second = asyncio.create_task(owner.get_token())
    try:
        await entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await second == await owner.get_token() == "synthetic.token.1"
        now[0] = 901
        assert await owner.get_token() == "synthetic.token.1"
        now[0] = 3569
        assert await owner.get_token() == "synthetic.token.1"
        now[0] = 3570
        assert await owner.get_token() == "synthetic.token.2"
        assert len(set(nonces)) == 2
    finally:
        await owner.close()
    assert owner._token is None
    with pytest.raises(AgentServiceError):
        await owner.get_token()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_caller", ["chat", "managed"])
@pytest.mark.parametrize("cancel_preflight", [False, True])
async def test_composed_chat_and_managed_preflight_share_durable_identity(
    tmp_path, monkeypatch, first_caller, cancel_preflight
):
    from puripuly_heart.app.ports.managed_identity import ManagedIdentityPreflightRequest
    from puripuly_heart.app.wiring import wiring_managed_auth_factory as factory
    from puripuly_heart.composition import application_runtime
    from puripuly_heart.composition.headless_application import compose_headless_application
    from puripuly_heart.core import managed_identity
    from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore

    for name in ("HOME", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "TEMP", "TMP"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setenv(name, str(directory))
    for name in tuple(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "synthetic-identity-passphrase")
    monkeypatch.setenv("PURIPULY_AGENT_RELAY_URL", "https://relay.example/v1/respond")
    monkeypatch.delenv("PURIPULY_AGENT_BROKER_URL", raising=False)
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            translation=replace(
                settings.intent.translation, openrouter_broker_base_url="https://broker.example"
            ),
            osc=replace(settings.intent.osc, connection_mode="off"),
            telemetry=replace(settings.intent.telemetry, enabled=False),
            integrated_context=replace(settings.intent.integrated_context, enabled=False),
        ),
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    captured = {}

    def compose(**kwargs):
        captured.update(kwargs)
        captured["transport"] = compose_agent_transport(**kwargs)
        return captured["transport"]

    monkeypatch.setattr(application_runtime, "compose_agent_transport", compose)
    connections = []

    def reject_network(_socket, address, *args):
        connections.append(address)
        raise AssertionError("Unexpected network activity in isolated identity smoke")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "sendto", reject_network)
    first_entered = threading.Event()
    second_entered = threading.Event()
    start_first = threading.Event()
    first_private = threading.Event()
    finish_first = threading.Event()

    class ObservedLock:
        def __init__(self):
            self.lock = managed_identity._identity_lock

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                second_entered.set()
                finish_first.set()
                self.lock.acquire()

        def __exit__(self, *_args):
            self.lock.release()

    monkeypatch.setattr(managed_identity, "_identity_lock", ObservedLock())
    replace_bundle = managed_identity._replace_managed_identity_bundle

    def scheduled_replace(state, secrets):
        if not first_entered.is_set():
            first_entered.set()
            assert start_first.wait(10)
            return replace_bundle(state, secrets)
        second_entered.set()
        assert first_private.wait(10)
        try:
            return replace_bundle(state, secrets)
        finally:
            finish_first.set()

    monkeypatch.setattr(managed_identity, "_replace_managed_identity_bundle", scheduled_replace)
    real_set = EncryptedFileSecretStore.set

    def scheduled_set(store, key, value):
        real_set(store, key, value)
        if key == managed_identity.MANAGED_DEVICE_PRIVATE_KEY_SECRET and not first_private.is_set():
            first_private.set()
            assert finish_first.wait(10)

    monkeypatch.setattr(EncryptedFileSecretStore, "set", scheduled_set)
    app = compose_headless_application(path)
    tasks = []
    issued = []
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path == "/v1/puri/token":
            body = json.loads(request.content)
            issued.append(body)
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic.identity.token",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "expires_at": (datetime.now(UTC) + timedelta(seconds=3600)).isoformat(),
                },
            )
        assert request.headers["authorization"] == "Bearer synthetic.identity.token"
        if request.url.path == "/v1/classify":
            return httpx.Response(
                200,
                json={
                    "answers": [
                        {"type": "choice", "name": "route", "choice": "social"},
                        {"type": "choice", "name": "effort", "choice": "low"},
                    ]
                },
            )
        assert request.url.path == "/v1/respond"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: "
            + json.dumps({"type": "completed", "response": answer("Synthetic completed response.")})
            + "\n\n",
        )

    transport = captured["transport"]
    await transport._client.aclose()
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    transport._token_owner._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        await app.start()
        owner = app.agent_chat()
        canonical_owner = captured["settings"]
        before = await app.control().query("settings.current", {})
        before_intent = canonical_owner.require_canonical().intent
        state = factory.managed_identity_state_port_from_vnext(
            canonical_owner, canonical_owner.require_canonical()
        )
        store = EncryptedFileSecretStore(
            tmp_path / "secrets.json", passphrase="synthetic-identity-passphrase"
        )
        store.set("openai_api_key", "synthetic-preserved-provider-key")
        preflight = factory.ManagedIdentityPreflightAdapter(state, store)
        request = ManagedIdentityPreflightRequest(
            local_secret_key="openrouter_managed_qq_api_key", correlation_id=None, metadata={}
        )
        operations = {
            "chat": lambda: owner.send("Hello"),
            "managed": lambda: preflight.preflight_managed_identity(request),
        }
        first = asyncio.create_task(operations[first_caller]())
        tasks.append(first)
        assert await asyncio.to_thread(first_entered.wait, 5)
        second = asyncio.create_task(operations["managed" if first_caller == "chat" else "chat"]())
        tasks.append(second)
        assert await asyncio.to_thread(second_entered.wait, 5)
        if cancel_preflight:
            managed_task = first if first_caller == "managed" else second
            managed_task.cancel()
            await asyncio.sleep(0)
            assert not managed_task.done()
        start_first.set()
        results = await asyncio.gather(*tasks, return_exceptions=cancel_preflight)
        chat_result, managed_result = results if first_caller == "chat" else results[::-1]
        assert chat_result.status == "completed"
        if cancel_preflight:
            assert isinstance(managed_result, asyncio.CancelledError)
        else:
            assert managed_result.succeeded
        assert len(issued) == 1
        identity = issued[0]
        if not cancel_preflight:
            assert managed_result.local_identity_revision == identity["installation_id"]
            assert managed_result.local_public_key == identity["device_public_key"]
        fresh_store = EncryptedFileSecretStore(
            tmp_path / "secrets.json", passphrase="synthetic-identity-passphrase"
        )
        loaded = managed_identity.load_existing_managed_identity_bundle(state, fresh_store)
        assert loaded is not None
        assert loaded.installation_id == identity["installation_id"]
        assert loaded.device_public_key == identity["device_public_key"]
        canonical = "\n".join(
            [
                "puripuly-heart:puri-token:v1",
                "POST",
                "/v1/puri/token",
                "puripuly-agent-relay",
                loaded.installation_id,
                loaded.device_public_key,
                identity["app_version"],
                identity["signed_at"],
                identity["nonce"],
            ]
        ).encode()
        loaded._private_key.public_key().verify(decode_base64url(identity["signature"]), canonical)
        assert fresh_store.get("openai_api_key") == "synthetic-preserved-provider-key"
        for key in ("openrouter_managed_api_key", "openrouter_managed_qq_api_key"):
            assert fresh_store.get(key) is None
        current = await app.control().query("settings.current", {})
        assert current["settings"]["intent"] == before["settings"]["intent"]
        assert canonical_owner.require_canonical().intent == before_intent
        assert (
            json.loads(path.read_text(encoding="utf-8"))["intent"]
            == to_dict(canonical_owner.require_canonical())["intent"]
        )
    finally:
        start_first.set()
        finish_first.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await app.stop()
    assert transport._token_owner._token is None
    assert calls == ["/v1/puri/token", "/v1/classify", "/v1/respond"]
    assert not connections
    code = """
import json, sys
from pathlib import Path
from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import SettingsVNextCanonicalPersistenceAdapter
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.wiring.wiring_managed_auth_factory import managed_identity_state_port_from_vnext
from puripuly_heart.core.managed_identity import load_existing_managed_identity_bundle
from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore
path = Path(sys.argv[1])
persistence = SettingsVNextCanonicalPersistenceAdapter()
settings = persistence.load_active(path).canonical_settings
owner = SettingsOwner(path=path, persistence=persistence, canonical=settings, authoritative=True)
state = managed_identity_state_port_from_vnext(owner, settings)
store = EncryptedFileSecretStore(path.parent / "secrets.json", passphrase="synthetic-identity-passphrase")
bundle = load_existing_managed_identity_bundle(state, store)
assert bundle is not None
assert store.get("openai_api_key") == "synthetic-preserved-provider-key"
print(json.dumps({"installation_id": bundle.installation_id, "device_public_key": bundle.device_public_key}))
"""
    restarted = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=30
    )
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr
    assert json.loads(restarted.stdout) == {
        "installation_id": loaded.installation_id,
        "device_public_key": loaded.device_public_key,
    }
