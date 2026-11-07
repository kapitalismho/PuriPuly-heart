from __future__ import annotations

import json
from copy import deepcopy

import pytest

from puripuly_heart.app.ports.agent_chat import AgentServiceError
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from tests.app.test_agent_chat import Records, Responses, answer, context_of
from tests.app.test_agent_chat_lifecycle import notices_done, watch
from tests.app.test_agent_chat_tools import Control
from tests.app.test_agent_chat_tools import catalog as catalog


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostic_id", [None, "synthetic-retry-error"])
@pytest.mark.parametrize("reset", [False, True])
async def test_explicit_retry_recovers_allowance_and_reenables_notices(diagnostic_id, reset):
    transport = Responses(
        AgentServiceError("service_allowance_exhausted"),
        answer("The intentional retry recovered."),
        answer("Verification continuation recovered too."),
    )
    owner = AgentChatOwner(Records(), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        assert (await owner.send("original setting request")).status == "service_unavailable"
        await watch(owner)
        owner.key_verification_completed("openai", True)
        await notices_done(owner)
        assert len(transport.classifications) == len(transport.requests) == 1
        assert owner.snapshot.messages[-1].text == "agent_chat.service_allowance_exhausted"
        if reset:
            owner.new_conversation()
        result = await owner.send("Please retry now.", diagnostic_id=diagnostic_id)
        assert result.status == "completed"
        expected_d = 2 if diagnostic_id is None else 1
        assert len(transport.classifications) == expected_d
        assert len(transport.requests) == 2
        if not reset:
            assert "original setting request" in json.dumps(transport.requests[-1])
            assert "agent_chat.service_allowance_exhausted" in json.dumps(transport.requests[-1])
        else:
            assert "original setting request" not in json.dumps(transport.requests[-1])
            assert "original setting request" in json.dumps(owner.report_archive)
        assert not owner._allowance_exhausted
        await watch(owner)
        owner.key_verification_completed("openai", True)
        await notices_done(owner)
        assert len(transport.classifications) == expected_d
        assert len(transport.requests) == 3
        assert transport.requests[-1]["effort"] == "high"
        assert owner.snapshot.phase == "idle"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_repeated_allowance_denial_recaches_until_another_intentional_retry():
    transport = Responses(
        AgentServiceError("service_allowance_exhausted"),
        AgentServiceError("service_allowance_exhausted"),
        answer("Recovered on the next intentional retry."),
        answer("Automatic continuation is enabled again."),
    )
    owner = AgentChatOwner(Records(), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        for index in range(2):
            assert (
                await owner.send(f"intentional attempt {index}")
            ).status == "service_unavailable"
            assert owner._allowance_exhausted
            for _ in range(2):
                await watch(owner)
                owner.key_verification_completed("openai", True)
            await notices_done(owner)
            assert len(transport.classifications) == len(transport.requests) == index + 1
        assert (await owner.send("Try once more.")).status == "completed"
        assert len(transport.classifications) == len(transport.requests) == 3
        assert "intentional attempt 0" in json.dumps(transport.requests[-1])
        assert "intentional attempt 1" in json.dumps(transport.requests[-1])
        await watch(owner)
        owner.key_verification_completed("openai", True)
        await notices_done(owner)
        assert len(transport.classifications) == 3 and len(transport.requests) == 4
    finally:
        await owner.close()


class StateChangeControl(Control):
    async def wait(self, operation_id):
        before = deepcopy(self.settings)
        revision = self.revision
        receipt = await super().wait(operation_id)
        if receipt["status"] in {"applied", "degraded"} and self.settings == before:
            self.revision = revision
            receipt["revision"] = revision
        return receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["applied", "degraded"])
async def test_approved_noop_has_no_actual_change_fact(catalog, status):
    control = StateChangeControl(catalog)
    control.status = status
    owner = AgentChatOwner(control, Responses(answer("No setting changed.")), localize=str)
    await owner.start()
    try:
        before = control.context()
        owner._tools.observe_context(before)
        alpha = before["settings"]["overlay.desktop_flet.visual.background_alpha"]
        result = await owner._tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": alpha}}
        )
        assert result["status"] == "needs_confirmation"
        assert not control.submissions
        result = await owner._tools.confirm(result["confirmation"]["id"], True)
        assert result["status"] == status and result["receipt"]["terminal"] is True
        assert control.context() == before
        assert not owner._actions
        assert len(control.submissions) == 1
        await owner.send("What actually changed?")
        assert context_of(owner._transport.requests[-1])["core"]["recent_actions"] == []
        owner.new_conversation()
        assert not owner._actions
        operations = [row for row in owner.report_archive if row.get("type") == "app_operation"]
        assert len(operations) == 2 and operations[-1]["changed"] is False
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("noop_status", ["applied", "degraded"])
async def test_noop_preserves_five_real_changes_and_approved_revert(catalog, noop_status):
    control = StateChangeControl(catalog)
    owner = AgentChatOwner(control, Responses(answer("History remains accurate.")), localize=str)
    await owner.start()
    try:
        for index, alpha in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
            owner._tools.observe_context(control.context())
            before_alpha = control.context()["settings"][
                "overlay.desktop_flet.visual.background_alpha"
            ]
            assert before_alpha != alpha
            control.status = "degraded" if index == 2 else "applied"
            result = await owner._tools.execute(
                "change_settings", {"changes": {"overlay.background_alpha": alpha}}
            )
            assert result["status"] == "needs_confirmation"
            result = await owner._tools.confirm(result["confirmation"]["id"], True)
            assert result["status"] == control.status
            assert (
                control.context()["settings"]["overlay.desktop_flet.visual.background_alpha"]
                == alpha
            )
        history = deepcopy(list(owner._actions))
        assert len(history) == 5 and history[2]["result"]["status"] == "degraded"
        before = control.context()
        owner._tools.observe_context(before)
        control.status = noop_status
        noop = await owner._tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.5}}
        )
        noop = await owner._tools.confirm(noop["confirmation"]["id"], True)
        assert noop["status"] == noop_status
        assert control.context() == before
        assert list(owner._actions) == history
        owner.new_conversation()
        assert list(owner._actions) == history
        control.status = "applied"
        owner._tools.observe_context(control.context())
        revert = await owner._tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.4}}
        )
        assert revert["status"] == "needs_confirmation"
        assert control.context() == before
        reverted = await owner._tools.confirm(revert["confirmation"]["id"], True)
        assert reverted["status"] == "applied"
        assert control.context()["settings"]["overlay.desktop_flet.visual.background_alpha"] == 0.4
        retained = list(owner._actions)
        assert retained[:-1] == history[1:]
        assert retained[-1]["tool"] == "change_settings"
        assert retained[-1]["result"]["changes"] == {"overlay.background_alpha": 0.4}
        await owner.send("What actually changed?")
        assert context_of(owner._transport.requests[-1])["core"]["recent_actions"] == retained
    finally:
        await owner.close()
