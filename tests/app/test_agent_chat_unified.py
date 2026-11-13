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
from tests.app.test_agent_chat_control_execution import ExecutionControl, tools_for
from tests.app.test_agent_chat_tools import catalog as catalog


def call(identity, family, action, **arguments):
    key = {"read_app": "topic", "open_app": "view", "manage_proposal": "action"}.get(
        family, "target"
    )
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
            call("gpu", "read_app", "gpu"),
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
            call("blocked-overlay", "set_app", "overlay", enabled=True),
            call("independent-after", "read_app", "general"),
        ),
        answer(),
        batch(
            call("recovered-gpu", "read_app", "gpu"),
            call("allowed-overlay", "set_app", "overlay", enabled=True),
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
            call("execute", "set_app", "overlay", enabled=True),
            call("after", "read_app", "general"),
        ),
        batch(
            call(
                "proposal",
                "manage_proposal",
                "settings",
                changes={"overlay.background_alpha": 0.4},
            ),
            call("blocked", "set_app", "overlay", enabled=False),
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
        batch(call("stage", "open_app", "provider_setup", provider="openrouter")),
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
        ("manage_proposal", {"action": "settings", "changes": {"overlay.background_alpha": None}}),
        ("manage_proposal", {"action": "settings", "changes": {"private": "value"}}),
        ("set_app", {"target": "capture.self"}),
        ("probe_app", {"target": "recognition.both"}),
        ("open_app", {"view": "auth.login", "provider": "chatgpt"}),
        ("read_app", {"topic": "settings.current"}),
        ("read_app", {"topic": "provider_setup"}),
        ("read_app", {"request": {"topic": "overview"}}),
    ):
        with pytest.raises(ValueError):
            validate_app_request(family, request, catalog)
    assert validate_app_request(
        "manage_proposal",
        {
            "action": "settings",
            "changes": {
                "translation.model": "gemma4_26b_31b",
                "translation.connection": "openrouter",
            },
        },
        catalog,
    )["changes"] == {"translation.model": "gemma4_26b_31b", "translation.connection": "openrouter"}
    for request in (
        {"view": "settings", "section": "api", "provider": None},
        {"target": "overlay", "enabled": True, "channel": "self"},
        {"action": "report", "summary": "Failure", "changes": {}},
    ):
        family = (
            "open_app"
            if "view" in request
            else "set_app" if "target" in request else "manage_proposal"
        )
        with pytest.raises(ValueError):
            validate_app_request(family, request, catalog)


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
async def test_flat_approval_requests_reject_other_actions_without_extra_effects(
    catalog, action, arguments, expected
):
    control = ExecutionControl(catalog)
    control.accounts = {"chatgpt": {"signed_in": True}}
    transport = Responses(
        batch(call("proposal", "manage_proposal", action, summary="Requested change", **arguments)),
        answer(),
        answer(),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Apply the requested change.")
        assert [item.status for item in result.tool_results] == ["rejected"]
        assert owner.snapshot.confirmation is None
        assert not control.submissions
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
            call("accepted", "set_app", "overlay", enabled=True),
            call(
                "unapproved",
                "manage_proposal",
                "settings",
                changes={"overlay.background_alpha": 0.4},
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
        batch(call("original", "probe_app", "recognition.self")),
        answer("The accepted operation is still pending."),
        batch(
            call("receipt", "read_app", "operation", operation_id="op-1"),
            call("blocked", "set_app", "translation", enabled=True),
        ),
        batch(call("independent", "set_app", "overlay", enabled=True)),
        answer("The existing probe is pending; an unrelated switch remains available."),
        batch(
            call("completed", "read_app", "operation", operation_id="op-1"),
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
        assert outputs["independent"]["status"] == "pending"
        assert [item["command"] for item in control.submissions] == [
            "recognition.probe",
            "overlay.set",
        ]
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
            "overlay.set",
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
            call("settings", "open_app", "settings", section="api", provider="deepseek"),
            call("premature", "set_app", "translation", enabled=True),
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


@pytest.mark.asyncio
async def test_failed_peer_owner_probe_continues_to_diagnostics_and_user_click_report():
    import httpx

    from tests.app.test_agent_chat_report import ReportResponses
    from tests.app.test_translation_probe import ControlledBackend, control_for

    backend = ControlledBackend(failure=httpx.ConnectError("PRIVATE-ENDPOINT"))
    production, runtime = control_for(
        backend,
        source_language="ko",
        target_language="en",
        peer_source_language="ja",
        peer_target_language="ko",
    )

    class LinkedControl(Records):
        async def query(self, name, arguments):
            if name == "settings.choices":
                self.queries.append((name, arguments))
                raise RuntimeError("Settings catalog unavailable")
            return await super().query(name, arguments)

        async def submit(self, *args, **kwargs):
            return await production.submit(*args, **kwargs)

        async def wait(self, *args, **kwargs):
            return await production.wait(*args, **kwargs)

        async def operation(self, identity):
            return await production.operation(identity)

    control = LinkedControl()

    class LinkedResponses(ReportResponses):
        async def respond(self, **request):
            catalog_counts.append(sum(name == "settings.choices" for name, _ in control.queries))
            async for event in super().respond(**request):
                yield event

    catalog_counts = []
    transport = LinkedResponses(
        batch(
            call("peer", "probe_app", "translation.peer"),
            call("dependent", "set_app", "overlay", enabled=True),
            call("diagnostics", "read_app", "diagnostics"),
        ),
        batch(
            call(
                "report",
                "manage_proposal",
                "report",
                summary="Peer translation failed during the isolated network check.",
            )
        ),
        answer(
            "The isolated Peer check failed. Review the report card to send the retained evidence."
        ),
    )
    owner = AgentChatOwner(
        control, transport, localize=str, support_log_snapshot=lambda: b"synthetic log\n"
    )
    try:
        await owner.start()
        result = await owner.send(
            "Check Peer translation, inspect the failure and offer a developer report."
        )
        assert [item.status for item in result.tool_results] == [
            "failed",
            "not_executed",
            "observed",
            "pending",
        ]
        receipt = result.tool_results[0].result["receipt"]
        assert receipt["channel"] == "peer"
        assert receipt["source_language"] == "ja" and receipt["target_languages"] == ["ko"]
        assert receipt["translation_result"] == "failed"
        assert receipt["failure_category"] == "network"
        assert receipt["output_delivery"] == receipt["display"] == "not_exercised"
        assert backend.requests[0].source_language == "ja"
        assert backend.requests[0].target_language == "ko"
        assert len(backend.requests) == 1
        assert result.tool_results[2].result["context"]["view"] == "diagnostics"
        assert all(request["allow_tools"] for request in transport.requests)
        assert len(transport.requests) == 3
        assert catalog_counts[1] == catalog_counts[2]
        assert not transport.uploads
        (offer,) = owner.snapshot.report_offers
        assert offer.status == "offered"
        assert owner.mark_report_displayed(offer.id)["status"] == "displayed"
        uploaded = await owner.send_report(offer.id)
        assert uploaded["status"] == "sent"
        assert len(transport.uploads) == 1
        assert "PRIVATE-" not in json.dumps([item.result for item in result.tool_results])
    finally:
        await owner.close()
        production.freeze_ingress()
        await production.drain_operations()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("uncertainty", ["pending", "submit_unknown", "lookup_missing"])
async def test_repeated_unresolved_signature_keeps_original_identity_and_independent_work(
    catalog, uncertainty
):
    class Uncertain(ExecutionControl):
        async def submit(self, command, arguments, **kwargs):
            receipt = await super().submit(command, arguments, **kwargs)
            if command == "overlay.set" and uncertainty == "submit_unknown":
                raise ConnectionError("Submission acknowledgment lost")
            return receipt

        async def wait(self, identity, timeout=None):
            if self.operations[identity]["request_id"] == "original":
                raise TimeoutError()
            return await super().wait(identity, timeout)

        async def operation(self, identity):
            if uncertainty == "lookup_missing":
                return {}
            return dict(self.operations[identity])

    control = Uncertain(catalog)
    transport = Responses(
        batch(call("original", "set_app", "overlay", enabled=True)),
        batch(
            call("repeat", "set_app", "overlay", enabled=True),
            call("independent-read", "read_app", "general"),
        ),
        batch(call("independent-command", "set_app", "translation", enabled=True)),
        answer("The original work remains unresolved; the unrelated switch completed."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    try:
        await owner.start()
        result = await owner.send("Enable the overlay, then inspect and enable translation.")
        assert [item.call_id for item in result.tool_results] == [
            "original",
            "repeat",
            "independent-read",
            "independent-command",
        ]
        original, repeated, read, independent = result.tool_results
        assert (
            original.status
            == repeated.status
            == ("unavailable" if uncertainty == "submit_unknown" else "pending")
        )
        assert read.status == "observed" and independent.status == "applied"
        assert [item["command"] for item in control.submissions] == [
            "overlay.set",
            "translation.set",
        ]
        assert control.operations["op-1"]["request_id"] == "original"
        if uncertainty == "submit_unknown":
            assert repeated.result["request_id"] == "original"
        else:
            assert repeated.result["receipt"]["operation_id"] == "op-1"
            assert repeated.result["receipt"]["request_id"] == "original"
        assert all(request["allow_tools"] for request in transport.requests)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_retired_generation_before_submit_does_not_poison_next_turn_signature(catalog):
    control = ExecutionControl(catalog)
    tools, actions, _ = tools_for(control)
    generation = 0

    def require_current(expected):
        if generation != expected:
            raise asyncio.CancelledError

    retired = asyncio.create_task(
        tools.execute(
            "set_app",
            {"target": "overlay", "enabled": True},
            require_current=lambda: require_current(0),
            request_id="retired",
        )
    )
    try:
        await asyncio.sleep(0)
        assert not retired.done()
        assert not control.submissions
        generation = 1
        with pytest.raises(asyncio.CancelledError):
            await retired
        assert not control.submissions
        assert not actions

        current = await tools.execute(
            "set_app",
            {"target": "overlay", "enabled": True},
            require_current=lambda: require_current(1),
            request_id="current",
        )
        assert current["status"] == "applied"
        assert current["receipt"]["terminal"] is True
        assert current["receipt"]["request_id"] == "current"
        assert control.submissions == [
            {"command": "overlay.set", "arguments": {"enabled": True}, "revision": None}
        ]
        assert control.operations["op-1"]["request_id"] == "current"
    finally:
        await asyncio.gather(retired, return_exceptions=True)
        await tools.close()
