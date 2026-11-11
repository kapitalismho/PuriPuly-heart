from __future__ import annotations

import asyncio
import json

import pytest

from puripuly_heart.app.services.agent_chat.context import AgentActivity, collect_agent_context
from puripuly_heart.app.services.agent_chat.conversation import AgentConversation
from puripuly_heart.app.services.agent_chat.execution import validate_app_request
from puripuly_heart.app.services.agent_chat.modules import model_context
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from tests.app.test_agent_chat import Records, Responses, answer
from tests.app.test_agent_chat_control_execution import ExecutionControl
from tests.app.test_agent_chat_tools import catalog as catalog


def call(identity, family, action, **arguments):
    key = "topic" if family == "read_app" else "action"
    return {
        "type": "function_call",
        "status": "completed",
        "call_id": identity,
        "name": family,
        "arguments": json.dumps({key: action, **arguments}),
    }


def batch(*calls):
    return {"status": "completed", "output": list(calls)}


@pytest.mark.asyncio
async def test_three_topic_reads_keep_independent_results_on_partial_failure(catalog):
    control = Records()
    control.values["settings.choices"] = catalog
    control.fail.add("gpu.status")
    transport = Responses(
        batch(
            call("output", "read_app", "output"),
            call("gpu", "read_app", "gpu", provider="soniox"),
            call("diagnostics", "read_app", "diagnostics"),
        ),
        answer(),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Inspect OSC, GPU and diagnostics.")
        assert [(item.call_id, item.status) for item in result.tool_results] == [
            ("output", "unchanged"),
            ("gpu", "unavailable"),
            ("diagnostics", "observed"),
        ]
        diagnostics = result.tool_results[2].result["context"]
        assert diagnostics["view"] == "diagnostics" and "core" not in diagnostics
        assert len(transport.requests) == 2
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_failed_read_blocks_later_action_and_recovers_on_next_turn(catalog):
    class RecoveringGpu(ExecutionControl):
        gpu_available = False

        async def query(self, name, arguments):
            if name == "gpu.status":
                if not self.gpu_available:
                    raise RuntimeError("GPU query unavailable")
                return {"state": "ready"}
            return await super().query(name, arguments)

    control = RecoveringGpu(catalog)
    control.settings["intent"]["ui"]["locale"] = "ko"
    transport = Responses(
        batch(
            call("failed-gpu", "read_app", "gpu"),
            call("independent-before", "read_app", "general"),
            call("blocked-overlay", "execute_app", "overlay.set", enabled=True),
            call("independent-after", "read_app", "general"),
        ),
        answer(),
        batch(
            call("recovered-gpu", "read_app", "gpu"),
            call("allowed-overlay", "execute_app", "overlay.set", enabled=True),
        ),
        answer(),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    try:
        await asyncio.wait_for(owner.start(), 3)
        failed = await asyncio.wait_for(owner.send("Inspect GPU and enable the overlay."), 3)
        assert failed.status == "completed"
        assert control.submissions == []
        assert [(item.call_id, item.status) for item in failed.tool_results] == [
            ("failed-gpu", "unavailable"),
            ("independent-before", "observed"),
            ("blocked-overlay", "not_executed"),
            ("independent-after", "observed"),
        ]
        assert failed.tool_results[0].result["code"] == "query_unavailable"
        assert failed.tool_results[2].result["code"] == "read_failed"
        for item in (failed.tool_results[1], failed.tool_results[3]):
            assert item.result["context"]["general"]["settings"]["locale"] == "ko"

        control.gpu_available = True
        recovered = await asyncio.wait_for(owner.send("Recheck GPU and enable the overlay."), 3)
        assert recovered.status == "completed"
        assert [(item.call_id, item.status) for item in recovered.tool_results] == [
            ("recovered-gpu", "observed"),
            ("allowed-overlay", "applied"),
        ]
        assert recovered.tool_results[0].result["result"]["state"] == "ready"
        assert control.submissions == [
            {"command": "overlay.set", "arguments": {"enabled": True}, "revision": None}
        ]
        outputs = [
            (item["call_id"], json.loads(item["output"])["status"])
            for item in owner.report_archive
            if item.get("type") == "function_call_output"
        ]
        assert outputs == [
            ("failed-gpu", "unavailable"),
            ("independent-before", "observed"),
            ("blocked-overlay", "not_executed"),
            ("independent-after", "observed"),
            ("recovered-gpu", "observed"),
            ("allowed-overlay", "applied"),
        ]
        assert len(transport.requests) == 4
    finally:
        await asyncio.wait_for(owner.close(), 3)


@pytest.mark.asyncio
async def test_read_action_read_observes_order_and_proposal_stops_effects(catalog):
    class Ordered(ExecutionControl):
        async def submit(self, command, arguments, **kwargs):
            receipt = await super().submit(command, arguments, **kwargs)
            if command == "overlay.set":
                self.app["translation_enabled"] = arguments["enabled"]
            return receipt

    control = Ordered(catalog)
    transport = Responses(
        batch(
            call("before", "read_app", "general"),
            call("execute", "execute_app", "overlay.set", enabled=True, channel="self"),
            call("after", "read_app", "general"),
        ),
        batch(
            call(
                "proposal",
                "propose_app",
                "settings",
                changes={"overlay.background_alpha": 0.4},
                summary="Adjust overlay transparency",
            ),
            call("blocked", "execute_app", "overlay.set", enabled=False),
        ),
        answer(),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Inspect and change the overlay.")
        assert [item.status for item in result.tool_results] == [
            "observed",
            "applied",
            "observed",
            "needs_confirmation",
            "not_executed",
        ]
        assert len(control.submissions) == 1
        assert owner.snapshot.confirmation["changes"] == {"overlay.background_alpha": 0.4}
        assert result.tool_results[0].result != result.tool_results[2].result
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_pure_provider_read_does_not_replace_staged_secure_entry(catalog):
    control = ExecutionControl(catalog)
    transport = Responses(
        batch(call("stage", "execute_app", "ui.provider_setup", provider="openrouter")),
        answer(),
        batch(call("read", "read_app", "provider_setup", provider="deepseek")),
        answer(),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        await owner.send("Show the OpenRouter secure controls.")
        staged = owner.snapshot.secure_entry_requests
        assert [item.provider for item in staged] == ["openrouter"]
        assert staged[0].status == "ready" and not staged[0].displayed
        read = call("pure-read", "read_app", "provider_setup", provider="deepseek")
        result = await owner._execute_call(read, owner._arguments(read), owner._generation)
        assert result["status"] == "available"
        assert owner.snapshot.secure_entry_requests == staged
        assert not result.get("controls_staged")
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_flat_requests_reject_unknown_null_invalid_and_missing_values(catalog):
    for family, request in (
        ("propose_app", {"action": "settings", "changes": {"overlay.background_alpha": None}}),
        ("propose_app", {"action": "settings", "changes": {"private": "value"}}),
        ("execute_app", {"action": "capture.set", "channel": "self"}),
        ("execute_app", {"action": "recognition.probe", "channel": "both"}),
        ("execute_app", {"action": "auth.login", "provider": "chatgpt"}),
        ("read_app", {"topic": "settings.current"}),
        ("read_app", {"topic": "provider_setup"}),
        ("read_app", {"request": {"topic": "overview"}}),
    ):
        with pytest.raises(ValueError):
            validate_app_request(family, request, catalog)
    assert validate_app_request(
        "propose_app",
        {
            "action": "settings",
            "changes": {
                "translation.model": "gemma4_26b_31b",
                "translation.connection": "openrouter",
            },
        },
        catalog,
    )["changes"] == {"translation.model": "gemma4_26b_31b", "translation.connection": "openrouter"}
    assert validate_app_request(
        "execute_app", {"action": "ui.settings", "section": "api", "provider": None}, catalog
    ) == {"action": "ui.settings", "section": "api"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "arguments", "expected"),
    [
        (
            "settings",
            {"changes": {"translation.qwen.region": "singapore"}},
            {"translation.qwen.region": "singapore"},
        ),
        (
            "secrets.delete",
            {"name": "deepgram_api_key", "changes": {"clipboard.enabled": True}},
            {"secrets.deepgram_api_key": False},
        ),
        (
            "auth.logout",
            {"provider": "chatgpt", "changes": {"clipboard.enabled": True}},
            {"accounts.chatgpt": False},
        ),
        (
            "overlay.position.reset",
            {"provider": "chatgpt", "changes": {"clipboard.enabled": True}},
            {"overlay.desktop_position": "default"},
        ),
    ],
)
async def test_flat_approval_requests_ignore_other_actions_without_extra_effects(
    catalog, action, arguments, expected
):
    control = ExecutionControl(catalog)
    control.accounts = {"chatgpt": {"signed_in": True}}
    transport = Responses(
        batch(call("proposal", "propose_app", action, summary="Requested change", **arguments)),
        answer(),
        answer(),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Apply the requested change.")
        assert [item.status for item in result.tool_results] == ["needs_confirmation"]
        card = owner.snapshot.confirmation
        assert card["changes"] == expected
        assert not control.submissions
        applied = await owner.confirm(card["id"], True)
        assert applied["status"] == "applied"
        if action == "settings":
            assert control.submissions == [{"changes": expected, "expected_revision": 11}]
        else:
            assert control.submissions == [
                {
                    "command": action,
                    "arguments": {
                        key: value
                        for key, value in arguments.items()
                        if key != "changes" and action != "overlay.position.reset"
                    },
                    "revision": 11,
                }
            ]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_relative_age_alone_keeps_core_baseline_and_trim_rehydrates():
    clock = [10.0]
    activity = AgentActivity(lambda: clock[0])
    activity.record({"topic": "transcript", "channel": "self"})
    control = Records()
    conversation = AgentConversation(token_limit=1)
    first = model_context(
        await collect_agent_context(control, activity=activity), modules=("core",)
    )
    conversation.append({"role": "user", "content": "baseline"}, context=first)
    clock[0] = 20.0
    second = model_context(
        await collect_agent_context(control, activity=activity), modules=("core",)
    )
    projected = conversation.project_context(second)
    assert projected["unchanged"] == ["core"]
    assert "core" not in projected
    conversation.begin()
    conversation.append({"role": "user", "content": "new group"})
    conversation.trim()
    assert conversation.project_context(second)["core"] == second["core"]


@pytest.mark.asyncio
async def test_stop_keeps_accepted_operation_once_and_rejects_late_results(catalog):
    class Held(ExecutionControl):
        async def wait(self, identity, timeout=None):
            self.entered.set()
            await self.release.wait()
            return await super().wait(identity, timeout)

    control = Held(catalog)
    transport = Responses(
        batch(
            call("accepted", "execute_app", "overlay.set", enabled=True),
            call(
                "unapproved", "propose_app", "settings", changes={"overlay.background_alpha": 0.4}
            ),
        )
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    task = asyncio.create_task(owner.send("Enable overlay then propose transparency."))
    try:
        await asyncio.wait_for(control.entered.wait(), 3)
        owner.cancel()
        control.release.set()
        assert (await task).status == "cancelled"
        await asyncio.sleep(0)
        assert len(control.submissions) == 1
        assert owner.snapshot.confirmation is None
        outputs = [
            item for item in owner.report_archive if item.get("type") == "function_call_output"
        ]
        assert [json.loads(item["output"])["status"] for item in outputs] == [
            "cancelled",
            "cancelled",
        ]
    finally:
        control.release.set()
        await owner.close()


@pytest.mark.asyncio
async def test_uncertain_wait_reconciles_same_receipt_before_later_effects(catalog):
    class Uncertain(ExecutionControl):
        async def wait(self, identity, timeout=None):
            raise TimeoutError()

        async def operation(self, identity):
            return dict(self.operations[identity])

    control = Uncertain(catalog)
    transport = Responses(
        batch(call("original", "execute_app", "recognition.probe", channel="self")),
        answer("The accepted operation is still pending."),
        batch(
            call("receipt", "read_app", "operation", operation_id="op-1"),
            call("blocked", "execute_app", "translation.set", enabled=True),
        ),
        answer("Wait for the existing operation, without resubmitting it."),
        batch(
            call("completed", "read_app", "operation", operation_id="op-1"),
            call("later", "execute_app", "translation.set", enabled=True),
        ),
        answer("The earlier receipt is complete; the next operation was accepted."),
    )
    owner = AgentChatOwner(control, transport, localize=lambda key, **values: key)
    try:
        await owner.start()
        await owner.send("Check Self capture and recognition scope.")
        await owner.send("Read the existing receipt before enabling translation.")
        outputs = {
            item["call_id"]: json.loads(item["output"])
            for item in owner.report_archive
            if item.get("type") == "function_call_output"
        }
        assert outputs["receipt"]["status"] == "pending"
        assert outputs["receipt"]["receipt"]["request_id"] == "original"
        assert outputs["blocked"]["status"] == "not_executed"
        assert [item["command"] for item in control.submissions] == ["recognition.probe"]
        control.operations["op-1"].update(
            status="applied",
            terminal=True,
            recognition_result="not_exercised",
            recognition_scope="capture_vad_only",
            configured_provider="gemini_transcribe",
            exercised_provider=None,
            cleanup_complete=True,
        )
        await owner.send("Now enable translation.")
        assert [item["command"] for item in control.submissions] == [
            "recognition.probe",
            "translation.set",
        ]
        completed = next(
            json.loads(item["output"])
            for item in owner.report_archive
            if item.get("type") == "function_call_output" and item.get("call_id") == "completed"
        )
        assert completed["status"] == "applied"
        assert completed["receipt"]["request_id"] == "original"
        assert completed["receipt"]["recognition_scope"] == "capture_vad_only"
        assert completed["receipt"]["recognition_result"] == "not_exercised"
        assert completed["receipt"]["configured_provider"] == "gemini_transcribe"
        assert completed["receipt"]["exercised_provider"] is None
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_settings_ui_stops_dependent_actions_until_user_continuation(catalog):
    control = ExecutionControl(catalog)
    transport = Responses(
        batch(
            call("settings", "execute_app", "ui.settings", section="api", provider="deepseek"),
            call("premature", "execute_app", "translation.set", enabled=True),
        ),
        answer("Use the displayed secure field before continuing."),
    )
    owner = AgentChatOwner(control, transport, localize=lambda key, **values: key)
    opened = []
    owner.bind_navigation(
        lambda *args: opened.append(args) or {"opened": True, "field_visible": True}
    )
    try:
        await owner.start()
        await owner.send("Open the DeepSeek key field before doing anything else.")
        results = {
            item["call_id"]: json.loads(item["output"])
            for item in owner.report_archive
            if item.get("type") == "function_call_output"
        }
        assert opened == [("api", "deepseek")]
        assert results["settings"]["opened"] is True
        assert results["premature"]["status"] == "not_executed"
        assert control.submissions == []
    finally:
        await owner.close()
