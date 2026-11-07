from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from puripuly_heart.app.ports.agent_chat import AgentServiceError
from puripuly_heart.app.services.agent_chat.conversation import AgentConversation
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from tests.app.test_agent_chat import (
    Records,
    Responses,
    answer,
    context_of,
    tool,
)
from tests.app.test_agent_chat_routing import decision
from tests.app.test_agent_chat_tools import Control
from tests.app.test_agent_chat_tools import catalog as catalog


def call(name, arguments, identity="change"):
    return {
        "status": "completed",
        "output": [{**tool(identity), "name": name, "arguments": json.dumps(arguments)}],
    }


class HeldResponses(Responses):
    def __init__(self, *responses):
        super().__init__(*responses)
        self.arrivals = asyncio.Queue()

    async def respond(self, **request):
        self.requests.append(deepcopy(request))
        self.arrivals.put_nowait(len(self.requests))
        response = self.responses.pop(0)
        if isinstance(response, asyncio.Future):
            try:
                response = await asyncio.shield(response)
            except asyncio.CancelledError:
                response = await asyncio.shield(response)
        if isinstance(response, Exception):
            raise response
        yield {"type": "completed", "response": response}


async def watch(owner, provider="openai"):
    return await owner._tools.execute("open_settings", {"section": "api", "provider": provider})


async def notices_done(owner):
    task = owner._notice_task
    if task is not None:
        await task


def test_configuration_updates_are_internal_even_when_recorded_directly():
    conversation = AgentConversation()
    update = {"type": "configuration_update", "reasoning": {"effort": "high"}}
    conversation.record(update)
    conversation.append(update)
    assert not conversation.archive
    assert not conversation.messages and not conversation.active_messages
    user = {"role": "user", "content": "Inspect the app."}
    conversation.append(user)
    assert conversation.archive == [user]


@pytest.mark.asyncio
async def test_watch_precedes_navigation_replacement_failure_retry_and_success_clear():
    transport = Responses(answer(), answer(), answer(), answer())
    owner = AgentChatOwner(Records(), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai", "deepgram"}))
    events = []
    owner.observe_events(lambda event: events.append(event))
    await owner.start()
    try:
        assert (
            await owner.send("Show the API key settings", diagnostic_id="synthetic-setup")
        ).status == "completed"

        def navigate(section, provider):
            assert owner._watched_provider == provider
            return {"opened": True, "field_visible": False}

        owner.bind_navigation(navigate)
        assert (await watch(owner))["completion_observable"] is True
        await watch(owner, "deepgram")
        owner.key_verification_completed("openai", True)
        assert len(transport.requests) == 1
        owner.key_verification_completed("deepgram", False, "PRIVATE displayed prose")
        await notices_done(owner)
        owner.key_verification_completed("deepgram", False, "provider_unavailable")
        await notices_done(owner)
        owner.key_verification_completed("deepgram", True)
        await notices_done(owner)
        owner.key_verification_completed("deepgram", False)
        assert len(transport.requests) == 2
        assert not transport.classifications
        notices = [message.text for message in owner.snapshot.messages if message.role == "notice"]
        assert len(notices) == 1 and "result=success" in notices[0]
        resumed = [item.get("content", "") for item in transport.requests[-1]["input"]]
        order = [
            next(index for index, text in enumerate(resumed) if marker in text)
            for marker in (
                "code=verification_failed",
                "code=provider_unavailable",
                "result=success",
            )
        ]
        assert order == sorted(order)
        assert "PRIVATE" not in json.dumps(transport.requests)
        assert len([event for event in events if event["kind"] == "app_notice"]) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_unobservable_provider_and_ordinary_events_never_resume_model():
    transport = Responses(answer())
    owner = AgentChatOwner(Records(), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    owner.observe_events(lambda event: True)
    await owner.start()
    try:
        assert (await watch(owner, "soniox"))["completion_observable"] is False
        owner.key_verification_completed("soniox", True)
        owner.key_verification_completed("openai", True)
        owner.archive_support_report("[app notice] key_verification provider=openai result=success")
        owner._tool_event({"kind": "ordinary_app_event"})
        await asyncio.sleep(0)
        assert not transport.requests and not transport.classifications
        assert not owner._items
        assert owner.report_archive[-1]["type"] == "support_report"
        assert not hasattr(owner, "send_notice")
        await owner.send("ordinary user request")
        assert "key_verification" not in json.dumps(transport.classifications)
        assert "key_verification" not in json.dumps(transport.requests)
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("absorb", [False, True])
async def test_notice_fifo_and_new_message_absorption(absorb):
    release = asyncio.get_running_loop().create_future()
    transport = HeldResponses(release, answer("notice one"), answer("notice two"))
    owner = AgentChatOwner(Records(), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    first = asyncio.create_task(owner.send("first", diagnostic_id="synthetic-error"))
    try:
        await transport.arrivals.get()
        await watch(owner)
        owner.key_verification_completed("openai", True, "first_notice")
        await watch(owner)
        owner.key_verification_completed("openai", True, "second_notice")
        assert len(transport.requests) == 1
        if absorb:
            newest = asyncio.create_task(owner.send("newest"))
            await transport.arrivals.get()
            release.set_result(answer("late old answer"))
            assert (await first).status == "cancelled"
            await newest
            await notices_done(owner)
            assert len(transport.requests) == 2
            contents = [item.get("content", "") for item in transport.requests[-1]["input"]]
            first_index = next(i for i, value in enumerate(contents) if "first_notice" in value)
            second_index = next(i for i, value in enumerate(contents) if "second_notice" in value)
            assert first_index < second_index < contents.index("newest")
            assert "late old answer" not in json.dumps(owner.report_archive)
            assert len(transport.classifications) == 1
        else:
            release.set_result(answer("first finished"))
            await first
            await notices_done(owner)
            assert len(transport.requests) == 3
            assert "first_notice" in json.dumps(transport.requests[1])
            assert "second_notice" not in json.dumps(transport.requests[1])
            assert "second_notice" in json.dumps(transport.requests[2])
            assert [request["effort"] for request in transport.requests] == ["high"] * 3
            assert not transport.classifications
    finally:
        if not release.done():
            release.set_result(answer())
        await owner.close()


@pytest.mark.asyncio
async def test_known_exhaustion_makes_eligible_notice_static_without_d_or_m():
    transport = Responses(AgentServiceError("service_allowance_exhausted"))
    owner = AgentChatOwner(Records(), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        assert (await owner.send("first")).status == "service_unavailable"
        await watch(owner)
        owner.key_verification_completed("openai", True)
        await notices_done(owner)
        assert len(transport.requests) == len(transport.classifications) == 1
        assert owner.snapshot.messages[-1].text == "agent_chat.service_allowance_exhausted"
        assert owner.snapshot.phase == "idle"
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
async def test_only_active_confirmation_result_resumes_without_reclassification(catalog, approved):
    control = Control(catalog)
    transport = Responses(
        call("change_settings", {"changes": {"stt.provider": "deepgram"}}),
        answer("Review the card."),
        answer("Actual confirmation observed."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        await owner.send("change", diagnostic_id="synthetic-error")
        assert (await owner.confirm("stale-card", approved))["status"] == "rejected"
        card = owner.snapshot.confirmation
        result = await owner.confirm(card["id"], approved)
        await notices_done(owner)
        assert result["status"] == ("applied" if approved else "declined")
        if not approved:
            assert len(transport.requests) == 2
            await owner.send("next", diagnostic_id="synthetic-error")
        assert len(transport.requests) == 3
        assert not transport.classifications
        assert transport.requests[-1]["effort"] == "high"
        assert result["status"] in json.dumps(transport.requests[-1])
        assert len(owner._actions) == int(approved)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_batch_recovery_preserves_confirmation_gate_until_user_approval(catalog):
    control = Control(catalog)
    change = call("change_settings", {"changes": {"stt.provider": "deepgram"}})
    transport = Responses(
        {"status": "completed", "output": [tool("read"), *change["output"]]},
        call("change_settings", {"changes": {"stt.provider": "deepgram"}}, "corrected"),
        call("open_settings", {"section": "api"}, "blocked"),
        answer("The approved setting is saved."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    navigations = []
    owner.bind_navigation(lambda *args: navigations.append(args) or {"opened": True})
    await owner.start()
    try:
        result = await owner.send("Check the app and change recognition to Deepgram")
        assert result.status == "needs_confirmation"
        assert not control.submissions
        assert not navigations
        assert [item.status for item in result.tool_results] == [
            "rejected",
            "rejected",
            "needs_confirmation",
            "not_executed",
        ]
        assert transport.requests[-1]["allow_tools"] is False
        card = owner.snapshot.confirmation
        assert card["changes"] == {"stt.provider": "deepgram"}
        confirmed = await owner.confirm(card["id"], True)
        await notices_done(owner)
        assert confirmed["status"] == "applied"
        assert control.context()["settings"]["stt.provider"] == "deepgram"
        assert len(control.submissions) == 1
        assert owner.snapshot.confirmation is None
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_model_withdrawal_uses_current_card_context_without_notice_continuation(catalog):
    control = Control(catalog)
    transport = Responses(
        call("change_settings", {"changes": {"stt.provider": "deepgram"}}),
        answer("Review the card."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    events = []
    owner.observe_events(lambda event: events.append(event))
    await owner.start()
    try:
        await owner.send("Propose Deepgram.")
        card = owner.snapshot.confirmation
        before = control.context()
        transport.responses.extend(
            [
                call("withdraw_confirmation", {"card_id": card["id"]}, "withdraw"),
                answer("The pending proposal is withdrawn."),
                answer("No settings were applied."),
            ]
        )
        result = await owner.send("Cancel that settings proposal.")
        await notices_done(owner)
        assert result.status == "completed"
        assert result.tool_results[0].result == {"status": "withdrawn", "card_id": card["id"]}
        assert context_of(transport.requests[2])["core"]["pending_confirmation"] == card
        assert "previous" not in context_of(transport.requests[2])["core"]["pending_confirmation"]
        assert owner.snapshot.confirmation is None
        assert control.context() == before
        assert not control.submissions and not owner.snapshot.recent_changes
        assert len(transport.requests) == 4 and len(transport.classifications) == 2
        assert not any(event["kind"] in {"confirmation", "app_notice"} for event in events)
        assert not any(message.role == "notice" for message in owner.snapshot.messages)
        await owner.send("What is pending now?")
        assert context_of(transport.requests[-1])["core"]["pending_confirmation"] is None
        assert (await owner.confirm(card["id"], True))["status"] == "rejected"
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt", ["stale_withdrawal", "model_approval"])
async def test_model_cannot_withdraw_a_replaced_card_or_approve_current_settings(catalog, attempt):
    control = Control(catalog)
    transport = Responses(
        call("change_settings", {"changes": {"stt.provider": "deepgram"}}),
        answer("Review Deepgram."),
        call("change_settings", {"changes": {"stt.provider": "soniox"}}, "replacement"),
        answer("Review Soniox."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        await owner.send("Propose Deepgram.")
        old = owner.snapshot.confirmation
        await owner.send("Propose Soniox instead.")
        current = owner.snapshot.confirmation
        transport.responses.extend(
            [
                (
                    call("withdraw_confirmation", {"card_id": old["id"]}, "stale")
                    if attempt == "stale_withdrawal"
                    else call("confirm", {"card_id": current["id"], "approved": True}, "approval")
                ),
                answer("Only the current card remains for your decision."),
            ]
        )
        before = control.context()
        result = await owner.send("Reconsider that proposal.")
        await notices_done(owner)
        rejected = result.tool_results[0].result
        assert rejected["status"] == "rejected"
        assert (
            rejected["error"]["code"] == "confirmation_unavailable"
            if attempt == "stale_withdrawal"
            else rejected["code"] == "invalid_tool_call"
        )
        assert context_of(transport.requests[4])["core"]["pending_confirmation"] == current
        assert owner.snapshot.confirmation == current
        assert control.context() == before
        assert not control.submissions and not owner.snapshot.recent_changes
        assert len(transport.requests) == 6 and len(transport.classifications) == 3
        assert not any(message.role == "notice" for message in owner.snapshot.messages)
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "message"),
    [
        ("work", "What will the pending proposal change?"),
        ("out_of_scope", "Tell me a joke about penguins."),
    ],
)
async def test_question_or_off_topic_message_preserves_pending_confirmation(
    catalog, route, message
):
    class RoutedResponses(Responses):
        async def classify(self, **request):
            self.classifications.append(deepcopy(request))
            return decision(route)

    control = Control(catalog)
    transport = RoutedResponses(
        call("change_settings", {"changes": {"stt.provider": "deepgram"}}),
        answer("Review the card."),
        answer("Your pending settings have not changed."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        await owner.send("Propose Deepgram.", diagnostic_id="synthetic-proposal")
        card = owner.snapshot.confirmation
        before = control.context()
        await owner.send(message)
        assert owner.snapshot.confirmation == card
        assert control.context() == before
        assert not control.submissions
        if route == "work":
            assert context_of(transport.requests[-1])["core"]["pending_confirmation"] == card
        else:
            assert transport.requests[-1]["input"] == [{"role": "user", "content": message}]
        assert card["id"] not in json.dumps(transport.classifications)
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("point", ["settings.choices", "app.status"])
@pytest.mark.parametrize("interrupt", ["cancel", "message"])
async def test_generation_fence_after_validation_awaits(catalog, point, interrupt):
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldControl(Control):
        async def query(self, name, arguments):
            if name == point and owner.snapshot.phase == "working":
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
            return await super().query(name, arguments)

    control = HeldControl(catalog)
    transport = Responses(
        call("change_settings", {"changes": {"overlay.background_alpha": 0.6}}),
        answer("new answer"),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    first = asyncio.create_task(owner.send("first"))
    try:
        await entered.wait()
        await watch(owner)
        if interrupt == "cancel":
            owner.cancel()
            assert owner.snapshot.phase == "idle" and not owner.snapshot.busy
            assert owner._watched_provider is None
        else:
            await owner.send("newest")
            assert owner._watched_provider == "openai"
        release.set()
        assert (await first).status == "cancelled"
        assert not control.submissions
        assert owner.snapshot.confirmation is None
        outputs = [item for item in owner._items if item.get("type") == "function_call_output"]
        assert len(outputs) == 1
        assert json.loads(outputs[0]["output"])["status"] == "cancelled"
        assert len(transport.requests) == (1 if interrupt == "cancel" else 2)
    finally:
        release.set()
        await owner.close()


@pytest.mark.asyncio
async def test_late_model_finally_cannot_clear_new_active_turn_or_navigate():
    old_release = asyncio.get_running_loop().create_future()
    new_release = asyncio.get_running_loop().create_future()
    transport = HeldResponses(old_release, new_release)
    owner = AgentChatOwner(Records(), transport, localize=str)
    navigations = []
    owner.bind_navigation(lambda *args: navigations.append(args) or {"opened": True})
    await owner.start()
    first = asyncio.create_task(owner.send("first"))
    try:
        await transport.arrivals.get()
        second = asyncio.create_task(owner.send("newest"))
        await transport.arrivals.get()
        active = owner._active
        old_release.set_result(call("open_settings", {"section": "overlay"}))
        assert (await first).status == "cancelled"
        assert owner._active is active and owner.snapshot.busy
        assert not navigations
        new_release.set_result(answer("new answer"))
        assert (await second).text == "new answer"
        assert not owner.snapshot.busy
    finally:
        if not old_release.done():
            old_release.set_result(answer())
        if not new_release.done():
            new_release.set_result(answer())
        await owner.close()


@pytest.mark.asyncio
async def test_late_decision_after_message_interruption_does_not_consume_new_turn():
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldRouter(Responses):
        async def classify(self, **request):
            self.classifications.append(request)
            if len(self.classifications) == 1:
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                return decision("out_of_scope")
            return decision("work", "high")

    transport = HeldRouter(answer("newest answer"))
    owner = AgentChatOwner(Records(), transport, localize=str, decision_timeout=10)
    await owner.start()
    first = asyncio.create_task(owner.send("first"))
    try:
        await entered.wait()
        assert (await owner.send("newest")).text == "newest answer"
        release.set()
        assert (await first).status == "cancelled"
        assert len(transport.requests) == 1
        assert owner.snapshot.messages[-1].text == "newest answer"
    finally:
        release.set()
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["degraded", "rejected"])
async def test_cancel_during_submitted_work_keeps_real_fact_without_continuation(catalog, status):
    control = Control(catalog)
    control.hold = True
    control.status = status
    control.receipt_extra = {"error": {"code": "controlled_runtime_result"}}
    transport = Responses(
        call("change_settings", {"changes": {"overlay.background_alpha": 0.4}}),
        answer("Review the card."),
        answer("new answer"),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    await owner.send("change")
    card = owner.snapshot.confirmation
    first = asyncio.create_task(owner.confirm(card["id"], True))
    try:
        await control.entered.wait()
        owner.cancel()
        assert not first.done()
        await owner.send("what is happening?")
        operations = context_of(transport.requests[-1])["core"]["operations"]
        assert len(operations) == 1 and operations[0]["phase"] == "submitted"
        assert not context_of(transport.requests[-1])["core"]["recent_actions"]
        control.release.set()
        await asyncio.gather(*tuple(owner._tools._tasks))
        assert (await first)["status"] == status
        assert len(owner._actions) == int(status == "degraded")
        if status == "degraded":
            assert owner._actions[0]["result"]["status"] == status
        assert len(transport.requests) == 3
        assert len(owner._operations) == 1
        assert next(iter(owner._operations.values()))["phase"] == "completed"
        actual = next(iter(owner._operations.values()))["result"]
        assert actual["status"] == status
        assert actual["receipt"]["error"]["code"] == "controlled_runtime_result"
    finally:
        control.release.set()
        await owner.close()


@pytest.mark.asyncio
async def test_cancel_pending_confirmation_work_does_not_resume_and_new_message_retains_card(
    catalog,
):
    control = Control(catalog)
    transport = Responses(
        call("change_settings", {"changes": {"stt.provider": "deepgram"}}),
        answer("Review card"),
        answer("newest answer"),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        await owner.send("paid provider")
        card = owner.snapshot.confirmation
        await owner.send("another question")
        assert owner.snapshot.confirmation == card
        control.hold = True
        confirmation = asyncio.create_task(owner.confirm(card["id"], True))
        await control.entered.wait()
        owner.cancel()
        control.release.set()
        assert (await confirmation)["status"] == "applied"
        await notices_done(owner)
        assert len(transport.requests) == 3
        assert owner.snapshot.confirmation is None
        assert len(owner._actions) == 1
    finally:
        control.release.set()
        await owner.close()


@pytest.mark.asyncio
async def test_last_five_actual_changes_exclude_queries_proposals_noops_and_duplicate_rows(catalog):
    control = Control(catalog)
    owner = AgentChatOwner(control, Responses(answer()), localize=str)
    await owner.start()
    try:
        for index in range(7):
            owner._tools.observe_context(control.context())
            control.status = "degraded" if index == 5 else "applied"
            proposed = await owner._tools.execute(
                "change_settings", {"changes": {"overlay.background_alpha": index / 10}}
            )
            await owner._tools.confirm(proposed["confirmation"]["id"], True)
        owner._tools.observe_context(control.context())
        proposed = await owner._tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.6}}
        )
        await owner._tools.confirm(proposed["confirmation"]["id"], True)
        await owner._tools.execute("change_settings", {"changes": {"stt.provider": "deepgram"}})
        await watch(owner)
        await owner.send("query")
        changes = context_of(owner._transport.requests[-1])["core"]["recent_actions"]
        assert len(changes) == 5
        assert [row["result"]["changes"]["overlay.background_alpha"] for row in changes] == [
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ]
        assert [row["result"]["status"] for row in changes] == [
            "applied",
            "applied",
            "applied",
            "degraded",
            "applied",
        ]
        owner.new_conversation()
        assert list(owner._actions) == changes
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", ["inactivity", "explicit"])
async def test_expiry_and_new_conversation_clear_context_reasoning_watch_and_card(catalog, reset):
    clock = [0.0]
    reasoning = {"type": "reasoning", "encrypted_content": "opaque-old-reasoning"}
    control = Control(catalog)
    transport = Responses(
        {"status": "completed", "output": [reasoning]},
        answer("new conversation answer"),
    )
    owner = AgentChatOwner(control, transport, localize=str, clock=lambda: clock[0])
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        await owner.send("old conversation", diagnostic_id="synthetic-error")
        await watch(owner)
        owner._tools.observe_context(control.context())
        await owner._tools.execute("change_settings", {"changes": {"stt.provider": "deepgram"}})
        owner.archive_support_report("Report sent")
        if reset == "explicit":
            owner.new_conversation()
        else:
            clock[0] = 1800
        await owner.send("new conversation")
        request = json.dumps(transport.requests[-1])
        assert "old conversation" not in request and "opaque-old-reasoning" not in request
        assert "Report sent" not in request
        assert "old conversation" not in json.dumps(transport.classifications[-1])
        assert owner._watched_provider is None
        assert owner.snapshot.confirmation is None
        assert owner.snapshot.messages[0].archived
        assert any(not row.archived and row.role == "notice" for row in owner.snapshot.messages)
        archive = owner.report_archive
        assert "old conversation" in json.dumps(archive)
        assert "Report sent" in json.dumps(archive)
        assert "opaque-old-reasoning" not in json.dumps(archive)
        assert [
            item["reasoning"]["effort"]
            for item in transport.requests[-1]["input"]
            if item.get("type") == "configuration_update"
        ] == ["high"]
        assert not any(item.get("type") == "configuration_update" for item in archive)
        archive[0]["content"] = "mutated copy"
        assert "mutated copy" not in json.dumps(owner.report_archive)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_notice_activity_extends_expiry_and_elapsed_time_never_resets_running_turn():
    clock = [0.0]
    release = asyncio.get_running_loop().create_future()
    transport = HeldResponses(release, answer("notice answer"), answer("follow-up"))
    owner = AgentChatOwner(Records(), transport, localize=str, clock=lambda: clock[0])
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    first = asyncio.create_task(owner.send("old task"))
    try:
        await transport.arrivals.get()
        clock[0] = 7200
        await watch(owner)
        owner.key_verification_completed("openai", False)
        assert owner.snapshot.busy and not owner.snapshot.messages[0].archived
        release.set_result(answer("old answer"))
        await first
        await notices_done(owner)
        clock[0] = 8999
        await owner.send("follow-up")
        assert not owner.snapshot.messages[0].archived
        assert "old task" in json.dumps(transport.requests[-1])
    finally:
        if not release.done():
            release.set_result(answer())
        await owner.close()


@pytest.mark.asyncio
async def test_grouped_trim_preserves_current_call_result_notices_and_full_report():
    transport = Responses(
        answer("old answer " + "x" * 900),
        {"status": "completed", "output": [tool("kept")]},
        answer("current answer"),
        answer("notice answer"),
    )
    owner = AgentChatOwner(Records(), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        await owner.send("old turn", diagnostic_id="synthetic-error")
        owner._conversation._token_limit = 100
        await owner.send("current turn")
        items = owner._items
        assert "old turn" not in json.dumps(items)
        assert [item["call_id"] for item in items if item.get("type") == "function_call"] == [
            "kept"
        ]
        assert [
            item["call_id"] for item in items if item.get("type") == "function_call_output"
        ] == ["kept"]
        assert owner.snapshot.messages[0].archived
        assert "old turn" in json.dumps(owner.report_archive)
        assert [
            item["reasoning"]["effort"]
            for item in transport.requests[-1]["input"]
            if item.get("type") == "configuration_update"
        ] == ["high"]
        await watch(owner)
        owner.key_verification_completed("openai", True)
        await notices_done(owner)
        assert "key_verification" in json.dumps(owner._items)
        assert "kept" in json.dumps(owner._items)
        assert "kept" in json.dumps(transport.requests[-1]["input"])
        assert "kept" in json.dumps(owner.report_archive)
        assert [
            item["reasoning"]["effort"]
            for item in transport.requests[-1]["input"]
            if item.get("type") == "configuration_update"
        ] == ["high", "high"]
        assert not any(item.get("type") == "configuration_update" for item in owner.report_archive)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_cancel_command_bypasses_models_and_discards_queued_notices(catalog):
    control = Control(catalog)
    release = asyncio.get_running_loop().create_future()
    transport = HeldResponses(release)
    owner = AgentChatOwner(control, transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    first = asyncio.create_task(owner.send("first"))
    try:
        await transport.arrivals.get()
        await watch(owner)
        owner._tools.observe_context(control.context())
        await owner._tools.execute("change_settings", {"changes": {"stt.provider": "deepgram"}})
        owner.key_verification_completed("openai", True)
        assert (await owner.send("취소")).status == "cancelled"
        assert not owner._pending_notices and owner._watched_provider is None
        assert owner.snapshot.confirmation is None and not owner.snapshot.busy
        release.set_result(call("open_settings", {"section": "overlay"}))
        assert (await first).status == "cancelled"
        assert len(transport.requests) == len(transport.classifications) == 1
        assert not control.submissions
        assert "key_verification" in json.dumps(owner._items)
    finally:
        if not release.done():
            release.set_result(answer())
        await owner.close()


@pytest.mark.asyncio
async def test_composed_runtime_withdraws_pending_card_without_settings_or_network(
    tmp_path, monkeypatch
):
    from puripuly_heart.composition import application_runtime
    from puripuly_heart.composition.headless_application import compose_headless_application

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "synthetic-withdrawal-passphrase")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    for name in tuple(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN")):
            monkeypatch.delenv(name)
    connections = []

    def reject_network(_socket, address, *args):
        connections.append(address)
        raise AssertionError("Unexpected network activity in isolated withdrawal smoke")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "sendto", reject_network)
    transport = Responses(
        call("change_settings", {"changes": {"stt.provider": "deepgram"}}),
        answer("Review the pending proposal."),
    )
    monkeypatch.setattr(application_runtime, "compose_agent_transport", lambda **_kwargs: transport)
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
    try:
        app.control().bind_instance("agent-withdrawal-composed-smoke")
        await app.start()
        owner = app.agent_chat()
        assert owner is not None
        before = await app.control().query("settings.current", {})
        saved = path.read_bytes()
        await owner.send("Propose Deepgram.")
        card = owner.snapshot.confirmation
        transport.responses.extend(
            [
                call("withdraw_confirmation", {"card_id": "unknown"}, "unknown"),
                answer("The current proposal has not changed."),
                call("withdraw_confirmation", {"card_id": card["id"]}, "withdraw"),
                answer("The proposal is withdrawn without applying settings."),
            ]
        )
        stale = await owner.send("Withdraw the unknown card.")
        assert stale.tool_results[0].result["error"]["code"] == "confirmation_unavailable"
        assert owner.snapshot.confirmation == card
        result = await owner.send("Withdraw the current proposal.")
        await notices_done(owner)
        assert result.tool_results[0].result == {"status": "withdrawn", "card_id": card["id"]}
        assert owner.snapshot.confirmation is None
        assert (await owner.confirm(card["id"], True))["status"] == "rejected"
        assert await app.control().query("settings.current", {}) == before
        assert path.read_bytes() == saved
        assert not owner.snapshot.recent_changes
        assert len(transport.requests) == 6 and len(transport.classifications) == 3
    finally:
        await app.stop()
    assert transport.closed and not connections


@pytest.mark.asyncio
async def test_composed_runtime_notices_mutations_reset_and_restart(tmp_path, monkeypatch):
    from puripuly_heart.composition import application_runtime
    from puripuly_heart.composition.headless_application import compose_headless_application

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "synthetic-c4-test-passphrase")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    for name in tuple(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN")):
            monkeypatch.delenv(name)
    connections = []

    def reject_network(_socket, address, *args):
        connections.append(address)
        raise AssertionError("Unexpected network activity in isolated chat smoke")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "sendto", reject_network)
    transport = Responses(
        call("change_settings", {"changes": {"overlay.background_alpha": 0.4}}),
        answer("Review the settings card."),
        call("open_settings", {"section": "api", "provider": "openai"}, "navigate"),
        answer("Enter the key in Settings; the turn is finished."),
        answer("Verification succeeded; the current screen stays in place."),
        answer("Fresh conversation."),
    )
    monkeypatch.setattr(application_runtime, "compose_agent_transport", lambda **_kwargs: transport)
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
    navigations = []
    try:
        app.control().bind_instance("agent-c4-composed-smoke")
        await app.start()
        owner = app.agent_chat()
        assert owner is not None
        owner.bind_verification_providers(frozenset({"openai"}))

        def navigate(section, provider):
            assert owner._watched_provider == provider
            navigations.append((section, provider))
            return {"opened": True, "field_visible": True}

        owner.bind_navigation(navigate)
        before = await app.control().query("settings.current", {})
        await owner.send("Make the overlay opaque and show the key input.")
        assert not owner.snapshot.busy
        card = owner.snapshot.confirmation
        assert card is not None
        assert await app.control().query("settings.current", {}) == before
        mutation = await owner.confirm(card["id"], True)
        await notices_done(owner)
        assert mutation["status"] == "applied" and mutation["receipt"]["terminal"] is True
        current = await app.control().query("settings.current", {})
        assert current["revision"] > before["revision"]
        assert (
            current["settings"]["intent"]["overlay"]["desktop_flet"]["visual"]["background_alpha"]
            == 0.4
        )
        owner.key_verification_completed("openai", True)
        await notices_done(owner)
        assert len(transport.classifications) == 1 and len(transport.requests) == 5
        assert navigations == [("api", "openai")]
        assert (
            context_of(transport.requests[-1])["core"]["recent_actions"][0]["result"]["status"]
            == "applied"
        )
        owner.new_conversation()
        await owner.send("Fresh request")
        assert "Make the overlay" not in json.dumps(transport.requests[-1])
        assert "Make the overlay" in json.dumps(owner.report_archive)
    finally:
        await app.stop()
    assert transport.closed and not connections
    assert len(owner._actions) == 1
    print(
        json.dumps(
            {
                "smoke": "c4_composed_runtime",
                "mutation": mutation["status"],
                "terminal": mutation["receipt"]["terminal"],
                "before_revision": before["revision"],
                "after_revision": current["revision"],
                "navigation_count": len(navigations),
                "classifications": len(transport.classifications),
                "responses": len(transport.requests),
                "network_attempts": len(connections),
                "owner_module": AgentChatOwner.__module__,
            }
        )
    )
    code = """
import asyncio, json, socket, sys
from pathlib import Path
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.headless_application import compose_headless_application
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from tests.app.test_agent_chat import Responses, answer
attempts = []
def reject_network(self, address, *args):
    attempts.append(address)
    raise AssertionError("Unexpected restart network activity")
transport = Responses(answer("Restarted"))
application_runtime.compose_agent_transport = lambda **_kwargs: transport
async def check():
    socket.socket.connect = reject_network
    socket.socket.sendto = reject_network
    assert Path(sys.modules[AgentChatOwner.__module__].__file__).resolve() == Path(sys.argv[2]).resolve()
    app = compose_headless_application(Path(sys.argv[1]))
    try:
        app.control().bind_instance("agent-c4-fresh-process")
        await app.start()
        owner = app.agent_chat()
        assert owner.snapshot.messages == () and owner.report_archive == () and not owner._actions
        current = await app.control().query("settings.current", {})
        alpha = current["settings"]["intent"]["overlay"]["desktop_flet"]["visual"]["background_alpha"]
        assert alpha == 0.4
        assert (await owner.send("Restarted request")).status == "completed"
        assert len(transport.classifications) == len(transport.requests) == 1
        assert "Make the overlay" not in json.dumps(transport.requests)
    finally:
        await app.stop()
    assert transport.closed and not attempts
    print(json.dumps({"smoke":"c4_fresh_process","persisted_alpha":alpha,"initial_messages":0,"initial_archive":0,"network_attempts":len(attempts)}))
asyncio.run(check())
"""
    restarted = subprocess.run(
        [sys.executable, "-c", code, str(path), sys.modules[AgentChatOwner.__module__].__file__],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr
    assert '"smoke": "c4_fresh_process"' in restarted.stdout
    print(restarted.stdout.strip())


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
async def test_confirmation_during_turn_queues_after_key_notice_with_actual_conflict(
    catalog, approved
):
    release = asyncio.get_running_loop().create_future()
    control = Control(catalog)
    transport = HeldResponses(
        call("change_settings", {"changes": {"stt.provider": "deepgram"}}),
        answer("Review the card"),
        release,
        answer("Key result received"),
        answer("Confirmation received"),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        await owner.send("prepare card", diagnostic_id="synthetic-error")
        await transport.arrivals.get()
        await transport.arrivals.get()
        card = owner.snapshot.confirmation
        current = asyncio.create_task(
            owner.send("another request", diagnostic_id="synthetic-error")
        )
        await transport.arrivals.get()
        await watch(owner)
        owner.key_verification_completed("openai", True, "queued_key_notice")
        control.revision += 1
        result = await owner.confirm(card["id"], approved)
        assert result["status"] == ("rejected" if approved else "declined")
        if approved:
            assert result["receipt"]["error"]["code"] == "revision_conflict"
        assert len(transport.requests) == 3
        release.set_result(answer("Active turn finished"))
        await current
        await notices_done(owner)
        assert not transport.classifications
        assert all(request["effort"] == "high" for request in transport.requests)
        first_notice = json.dumps(transport.requests[3])
        assert "queued_key_notice" in first_notice
        if approved:
            assert len(transport.requests) == 5
            assert "[app notice] confirmation" not in first_notice
            assert "[app notice] confirmation" in json.dumps(transport.requests[4])
        else:
            assert len(transport.requests) == 4
            resumed = [item.get("content", "") for item in transport.requests[3]["input"]]
            key = next(i for i, text in enumerate(resumed) if "queued_key_notice" in text)
            declined = next(
                i for i, text in enumerate(resumed) if "[app notice] confirmation" in text
            )
            assert key < declined
        assert not owner._actions
    finally:
        if not release.done():
            release.set_result(answer())
        await owner.close()


@pytest.mark.asyncio
async def test_replacement_card_retires_old_confirmation_continuation_but_keeps_applied_fact(
    catalog,
):
    control = Control(catalog)
    transport = Responses(
        call(
            "change_settings",
            {
                "changes": {
                    "stt.provider": "deepgram",
                    "overlay.background_alpha": 0.4,
                    "languages": {"source": "en"},
                }
            },
        ),
        answer("Review the card"),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        source_before = control.context()["settings"]["languages.source_language"]
        await owner.send("prepare card")
        card = owner.snapshot.confirmation
        control.hold = True
        confirmation = asyncio.create_task(owner.confirm(card["id"], True))
        await control.entered.wait()
        replacement = await owner._tools.execute(
            "change_settings",
            {
                "changes": {
                    "stt.provider": "soniox",
                    "locale": "en",
                    "peer_stt.provider": "deepgram",
                }
            },
        )
        control.release.set()
        assert (await confirmation)["status"] == "applied"
        await notices_done(owner)
        assert owner.snapshot.confirmation == replacement["confirmation"]
        assert len(transport.requests) == 2
        assert len(owner._actions) == 1
        assert owner._actions[0]["result"]["changes"] == {
            "stt.provider": "deepgram",
            "overlay.background_alpha": 0.4,
        }
        assert control.context()["settings"]["languages.source_language"] == source_before
        conflict = await owner._tools.confirm(replacement["confirmation"]["id"], True)
        assert conflict["status"] == "rejected"
        assert conflict["receipt"]["error"]["code"] == "revision_conflict"
        assert owner.snapshot.confirmation is None
        assert len(control.submissions) == 2
    finally:
        control.release.set()
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["key", "card"])
@pytest.mark.parametrize("absorb", [False, True])
async def test_pending_task_keeps_initiating_call_evidence_and_notice_under_trim(
    catalog, task, absorb
):
    initiating = (
        call("open_settings", {"section": "api", "provider": "openai"}, "task-call")
        if task == "key"
        else call("change_settings", {"changes": {"stt.provider": "deepgram"}}, "task-call")
    )
    transport = Responses(
        answer("old unrelated evidence " + "x" * 3000),
        initiating,
        answer("current task evidence " + "y" * 3000),
        answer("The pending task remains."),
        answer("Continuation observed."),
        answer("New task."),
    )
    owner = AgentChatOwner(Control(catalog), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        await owner.send("old unrelated task")
        await owner.send("start current task")
        owner._conversation._token_limit = 100
        await owner.send("question while the task is pending")
        pending_input = transport.requests[-1]["input"]
        assert "old unrelated evidence" not in json.dumps(pending_input)
        assert "current task evidence" in json.dumps(pending_input)
        assert [
            item["call_id"] for item in pending_input if item.get("type") == "function_call"
        ] == ["task-call"]
        assert [
            item["call_id"] for item in pending_input if item.get("type") == "function_call_output"
        ] == ["task-call"]
        if task == "key":
            owner.key_verification_completed("openai", True)
        else:
            card = owner.snapshot.confirmation
            assert (await owner.confirm(card["id"], False))["status"] == "declined"
        if absorb or task == "card":
            await owner.send("new message absorbing the notice")
        await notices_done(owner)
        resumed_input = transport.requests[-1]["input"]
        assert "current task evidence" in json.dumps(resumed_input)
        assert "[app notice]" in json.dumps(resumed_input)
        assert [
            item["call_id"] for item in resumed_input if item.get("type") == "function_call"
        ] == ["task-call"]
        assert [
            item["call_id"] for item in resumed_input if item.get("type") == "function_call_output"
        ] == ["task-call"]
        assert len(transport.requests) == 5
        assert not next(
            row for row in owner.snapshot.messages if row.text == "start current task"
        ).archived
        await owner.send("next unrelated task")
        next_input = json.dumps(transport.requests[-1]["input"])
        assert "task-call" not in next_input and "[app notice]" not in next_input
        assert "task-call" in json.dumps(owner.report_archive)
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", [False, True])
async def test_actual_verification_callback_preserves_typed_namespace_without_inferring_prose(
    typed,
):
    from puripuly_heart.core.openrouter.authentication import OpenRouterAuthenticationError
    from puripuly_heart.ui.components.settings.api_key_verification_controller import (
        ApiKeyVerificationController,
    )

    transport = Responses(
        answer("Open the secure API field."), answer("Verification result received.")
    )
    owner = AgentChatOwner(Records(), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openrouter"}))
    saved = []
    synthetic_key = "SYNTHETIC-KEY-NOT-FOR-MODEL"

    def verify(provider, value):
        assert provider == "openrouter" and value == synthetic_key
        if typed:
            raise OpenRouterAuthenticationError(
                stage="key_verification", reason="credential_rejected", status_code=401
            )
        raise RuntimeError("PRIVATE prose about 401 and truncated credentials")

    controller = ApiKeyVerificationController(
        secret_key="openrouter_api_key",
        provider="openrouter",
        on_save=lambda secret, value: saved.append(secret),
        on_verify=verify,
        on_complete=owner.key_verification_completed,
    )
    controller.set_value_getter(lambda: synthetic_key)
    await owner.start()
    try:
        assert (
            await owner.send("Help verify OpenRouter", diagnostic_id="synthetic-setup")
        ).status == "completed"
        await watch(owner, "openrouter")
        controller.force_status("success")
        assert len(transport.requests) == 1
        controller.notify_edit()
        controller.handle_blur(synthetic_key)
        await controller.run_pending()
        await notices_done(owner)
        expected = "openrouter.auth.credential_rejected" if typed else "verification_failed"
        assert not [row for row in owner.snapshot.messages if row.role == "notice"]
        assert len(transport.requests) == 1
        await owner.send("Why did it fail?", diagnostic_id="synthetic-setup")
        assert saved == ["openrouter_api_key"]
        assert len(transport.requests) == 2 and not transport.classifications
        assert f"code={expected}" in json.dumps(transport.requests)
        assert synthetic_key not in json.dumps(transport.requests)
        assert controller.key_hash(synthetic_key) not in json.dumps(transport.requests)
        assert "PRIVATE prose" not in json.dumps(transport.requests)
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["cancel", "message", "reset", "shutdown"])
async def test_context_read_late_completion_is_fenced_and_call_is_closed(interrupt):
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldRecords(Records):
        async def query(self, name, arguments):
            result = await super().query(name, arguments)
            if name == "overlay.status" and owner.snapshot.phase == "working":
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                result["output"]["lifecycle"] = "LATE-CONTEXT-MUST-NOT-ENTER"
            return result

    transport = Responses(
        call("get_app_context", {"topic": "output"}, "context"),
        answer("New turn completed."),
    )
    owner = AgentChatOwner(HeldRecords(), transport, localize=str)
    await owner.start()
    first = asyncio.create_task(owner.send("Check output"))
    try:
        await entered.wait()
        if interrupt == "cancel":
            owner.cancel()
        elif interrupt == "message":
            assert (await owner.send("New question")).status == "completed"
        elif interrupt == "reset":
            owner.new_conversation()
        else:
            owner.freeze_ingress()
        release.set()
        if interrupt == "shutdown":
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            assert (await first).status == "cancelled"
        assert not owner.snapshot.busy
        assert len(transport.requests) == (2 if interrupt == "message" else 1)
        assert "LATE-CONTEXT-MUST-NOT-ENTER" not in json.dumps(owner.report_archive)
        outputs = [
            item for item in owner.report_archive if item.get("type") == "function_call_output"
        ]
        assert len(outputs) == 1 and outputs[0]["call_id"] == "context"
        assert json.loads(outputs[0]["output"]) == {"status": "cancelled", "code": "turn_cancelled"}
        assert not owner._actions
    finally:
        release.set()
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["cancel", "message", "reset", "shutdown"])
@pytest.mark.parametrize("completed_sibling", [False, True])
async def test_parallel_context_reads_are_drained_without_stale_publication(
    interrupt, completed_sibling, monkeypatch
):
    names = ("overlay.status", "accounts.status")
    entered = {name: asyncio.Event() for name in names}
    cancelled = {name: asyncio.Event() for name in names}
    release = asyncio.Event()
    finished = set()

    class HeldRecords(Records):
        async def query(self, name, arguments):
            result = await super().query(name, arguments)
            if owner.snapshot.phase != "working" or name not in entered:
                return result
            entered[name].set()
            try:
                if name == "accounts.status" and completed_sibling:
                    return result
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        cancelled[name].set()
                return result
            finally:
                finished.add(name)

    records = HeldRecords()
    records.values["accounts.status"] = {}
    transport = Responses(
        {
            "status": "completed",
            "output": [
                tool("output"),
                tool("accounts", arguments='{"topic":"accounts"}'),
            ],
        },
        answer("New turn completed."),
    )
    owner = AgentChatOwner(records, transport, localize=str)
    observed = []
    observe_context = owner._tools.observe_context

    def observe(context):
        observed.append(deepcopy(context))
        observe_context(context)

    monkeypatch.setattr(owner._tools, "observe_context", observe)
    await owner.start()
    first = asyncio.create_task(owner.send("Check output and accounts"))
    closing = None
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 1)
        assert not owner.snapshot.tool_results
        assert not owner.snapshot.setup_actions
        assert len(observed) == 1
        if interrupt == "cancel":
            owner.cancel()
        elif interrupt == "message":
            assert (await owner.send("New question")).status == "completed"
        elif interrupt == "reset":
            owner.new_conversation()
        else:
            owner.freeze_ingress()
            closing = asyncio.create_task(owner.close())
        pending = names[:1] if completed_sibling else names
        await asyncio.wait_for(asyncio.gather(*(cancelled[name].wait() for name in pending)), 1)
        assert not first.done()
        if closing is not None:
            assert not closing.done()
        release.set()
        if interrupt == "shutdown":
            with pytest.raises(asyncio.CancelledError):
                await first
            await closing
        else:
            assert (await first).status == "cancelled"
        assert finished == set(names)
        assert len(observed) == (2 if interrupt == "message" else 1)
        assert len(transport.requests) == (2 if interrupt == "message" else 1)
        assert not owner.snapshot.setup_actions
        assert not any(
            item.status in {"observed", "unchanged"} for item in owner.snapshot.tool_results
        )
        outputs = [
            item for item in owner.report_archive if item.get("type") == "function_call_output"
        ]
        assert [item["call_id"] for item in outputs] == ["output", "accounts"]
        assert [json.loads(item["output"]) for item in outputs] == [
            {"status": "cancelled", "code": "turn_cancelled"}
        ] * 2
    finally:
        release.set()
        if closing is None:
            await owner.close()
        else:
            await closing
        await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_snapshot_exposes_running_tool_and_card_before_values_to_the_renderer(catalog):
    control = Control(catalog)
    seen = []
    transport = Responses(
        call("change_settings", {"changes": {"stt.provider": "deepgram"}}),
        answer("Review the card."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    owner.observe(lambda snapshot: seen.append(snapshot.active_tool))
    await owner.start()
    try:
        await owner.send("change", diagnostic_id="synthetic-error")
        assert ("change_settings", None) in seen
        assert owner.snapshot.active_tool is None
        card = owner.snapshot.confirmation
        previous = control.context()["settings"]["stt.provider"]
        assert owner.snapshot.confirmation_previous == {"stt.provider": previous}
        assert "previous" not in card
        outputs = [
            json.loads(item["output"])
            for item in transport.requests[-1]["input"]
            if item.get("type") == "function_call_output"
        ]
        offered = next(item for item in outputs if item.get("status") == "needs_confirmation")
        assert offered["confirmation"]["id"] == card["id"]
        assert "previous" not in offered["confirmation"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_approval_queue_advances_without_model_until_final_card(catalog):
    control = Control(catalog)
    changes = {
        "stt.provider": "deepgram",
        "peer_stt.provider": "deepgram",
        "overlay.background_alpha": 0.4,
        "locale": "en",
        "overlay.translation.enabled": False,
    }
    transport = Responses(
        call("change_settings", {"changes": changes}),
        answer("Review the cards."),
        answer("All approved settings are applied."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    seen = []
    owner.observe(seen.append)
    await owner.start()
    try:
        await owner.send("Make these changes.")
        assert owner.snapshot.confirmation is not None
        assert not control.submissions
        first = owner.snapshot.confirmation
        assert first["changes"] == {
            "stt.provider": "deepgram",
            "peer_stt.provider": "deepgram",
        }
        first_result = await owner.confirm(first["id"], True)
        second = owner.snapshot.confirmation
        assert first_result["changes"] == first["changes"]
        assert first_result["confirmation"] == second
        assert second["revision"] == first_result["receipt"]["revision"] == 12
        assert second["changes"] == {"overlay.background_alpha": 0.4, "locale": "en"}
        assert seen[-1].confirmation == second
        assert seen[-1].recent_changes[-1]["result"]["changes"] == first["changes"]
        await notices_done(owner)
        assert len(transport.requests) == 2
        assert not any(message.role == "notice" for message in owner.snapshot.messages)
        assert owner.snapshot.confirmation_previous == {
            "overlay.background_alpha": control.context()["settings"][
                "overlay.desktop_flet.visual.background_alpha"
            ],
            "locale": control.context()["settings"]["ui.locale"],
        }
        second_result = await owner.confirm(second["id"], True)
        third = owner.snapshot.confirmation
        assert second_result["confirmation"] == third
        assert third["changes"] == {"overlay.translation.enabled": False}
        assert third["revision"] == 13
        await notices_done(owner)
        assert len(transport.requests) == 2
        final = await owner.confirm(third["id"], True)
        await notices_done(owner)
        assert final["status"] == "applied" and "confirmation" not in final
        assert owner.snapshot.confirmation is None
        assert len(transport.requests) == 3 and len(transport.classifications) == 1
        assert [item["expected_revision"] for item in control.submissions] == [11, 12, 13]
        resumed = json.dumps(transport.requests[-1]["input"])
        assert all(card["id"] in resumed for card in (first, second, third))
        assert (await owner.confirm(first["id"], True))["status"] == "rejected"
        assert len(control.submissions) == 3
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["cancel", "cancel_confirmation", "reset"])
async def test_held_approved_operation_cannot_revive_cancelled_queue(catalog, cancel):
    control = Control(catalog)
    changes = {
        "overlay.background_alpha": 0.4,
        "locale": "en",
        "stt.provider": "deepgram",
    }
    transport = Responses(
        call("change_settings", {"changes": changes}),
        answer("Review the cards."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        await owner.send("Make these changes.")
        card = owner.snapshot.confirmation
        control.hold = True
        pending = asyncio.create_task(owner.confirm(card["id"], True))
        await control.entered.wait()
        if cancel == "reset":
            owner.new_conversation()
        elif cancel == "cancel":
            owner.cancel()
        else:
            owner.cancel_confirmation()
        control.release.set()
        result = await pending
        await notices_done(owner)
        assert result["status"] == "applied"
        assert owner.snapshot.confirmation is None
        assert len(control.submissions) == 1
        assert control.context()["settings"]["overlay.desktop_flet.visual.background_alpha"] == 0.4
        assert control.context()["settings"]["stt.provider"] != "deepgram"
        assert len(transport.requests) == 2
        assert len(owner.snapshot.recent_changes) == 1
        assert (await owner.confirm(card["id"], True))["status"] == "rejected"
    finally:
        control.release.set()
        await owner.close()
