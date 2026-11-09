from __future__ import annotations

import json

import pytest

from tests.app.test_agent_chat import Records, answer, context_of, setting
from tests.app.test_agent_chat_lifecycle import call
from tests.app.test_agent_chat_routing import selected_decision
from tests.app.test_agent_chat_routing_integration import RoutedResponses, owner_for


def latest_update(request):
    return next(
        json.loads(item["content"])["app_context"]
        for item in reversed(request["input"])
        if isinstance(item.get("content"), str) and item["content"].startswith('{"app_context":')
    )


@pytest.mark.asyncio
async def test_repeated_reads_reuse_retained_evidence_and_join_followup_scope():
    transport = RoutedResponses(
        selected_decision("translation"),
        call("get_app_context", {"topic": "output"}, "first-read"),
        call("get_app_context", {"topic": "output"}, "repeat-read"),
        answer(),
        answer(),
    )
    owner = owner_for(transport)
    await owner.start()
    try:
        first = await owner.send("Inspect translation and output.")
        assert first.status == "completed"
        first_read, repeat = first.tool_results
        assert first_read.status == "observed"
        assert first_read.result["context"]["output"]["osc"]["remote_delivery"] == "unacknowledged"
        assert repeat.status == "unchanged"
        assert repeat.result["context"]["unchanged"] == ["core", "output"]
        assert repeat.result["context"]["observed_at"]
        assert "output" not in repeat.result["context"]
        transport.routing = selected_decision("translation", relation="continue")
        assert (await owner.send("Check the same task again.")).status == "completed"
        update = latest_update(transport.requests[-1])
        assert update["modules"] == ["core", "translation", "output"]
        assert update["unchanged"] == update["modules"]
        assert "catalogs" not in update
        assert (
            context_of(transport.requests[-1])["output"]["osc"]["remote_delivery"]
            == "unacknowledged"
        )
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_current_values_change_without_repeating_the_setting_dictionary():
    records = Records()
    transport = RoutedResponses(selected_decision("languages"), answer(), answer())
    owner = owner_for(transport, records)
    await owner.start()
    try:
        await owner.send("Inspect the translation languages.")
        initial = context_of(transport.requests[-1])
        records.values["settings.current"]["settings"]["intent"]["languages"][
            "target_language"
        ] = "ko"
        records.values["settings.current"]["revision"] = 43
        transport.routing = selected_decision("languages", relation="continue")
        await owner.send("Inspect the changed target language.")
        update = latest_update(transport.requests[-1])
        assert "catalogs" not in update
        assert update["languages"]["settings"]["languages.target"] == "ko"
        current = context_of(transport.requests[-1])
        assert setting(current, "languages.target")["value"] == "ko"
        assert (
            setting(current, "languages.target")["label"]
            == setting(initial, "languages.target")["label"]
        )
        assert current["core"]["revision"] == 43
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_failed_observation_and_recovery_never_reuse_a_stale_presence_result():
    records = Records()
    transport = RoutedResponses(selected_decision("translation"), answer(), answer(), answer())
    owner = owner_for(transport, records)
    await owner.start()
    try:
        await owner.send("Inspect translation readiness.")
        records.fail.add("secrets.presence")
        transport.routing = selected_decision("translation", relation="continue")
        await owner.send("Check readiness again.")
        failed = latest_update(transport.requests[-1])
        assert failed["missing"] == ["secrets.presence"]
        assert not failed.get("unchanged")
        assert "keys" not in failed["translation"]
        records.fail.remove("secrets.presence")
        await owner.send("Check after the query recovered.")
        recovered = latest_update(transport.requests[-1])
        assert recovered["missing"] == []
        assert recovered["translation"]["keys"]["openai_api_key"] is True
        assert "translation" in recovered["catalogs"]
        assert "PRIVATE-" not in json.dumps(transport.requests)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_device_availability_updates_without_a_settings_revision_change():
    records = Records()
    records.values["settings.choices"]["fields"] = ["audio.input_device"]
    transport = RoutedResponses(selected_decision("audio"), answer(), answer())
    owner = owner_for(transport, records)
    await owner.start()
    try:
        await owner.send("Inspect microphone availability.")
        first = context_of(transport.requests[-1])
        assert setting(first, "audio.input_device")["options"][0]["available"] is True
        records.values["settings.choices"]["choices"]["audio.input_device"][0]["available"] = False
        transport.routing = selected_decision("audio", relation="continue")
        await owner.send("Check the microphone again.")
        update = latest_update(transport.requests[-1])
        assert "catalogs" not in update
        assert update["audio"]["options"]["audio.input_device"][0]["available"] is False
        assert context_of(transport.requests[-1])["core"]["revision"] == first["core"]["revision"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_account_read_stages_current_controls_even_when_account_facts_are_known():
    transport = RoutedResponses(
        selected_decision("accounts"),
        answer(),
        call("get_app_context", {"topic": "accounts"}, "first-controls"),
        call("get_app_context", {"topic": "accounts"}, "same-controls"),
        answer(),
        call("get_app_context", {"topic": "accounts"}, "new-controls"),
        answer(),
    )
    owner = owner_for(transport)
    await owner.start()
    try:
        await owner.send("Inspect account readiness.")
        assert not owner.snapshot.setup_actions
        transport.routing = selected_decision("accounts", relation="continue")
        second = await owner.send("Show account connection controls.")
        staged, repeated = second.tool_results
        assert staged.status == "observed"
        assert "account.chatgpt.setup" in staged.result["context"]["accounts"]["actions"]
        assert repeated.status == "not_executed"
        first_actions = {action.id for action in owner.snapshot.setup_actions if action.available}
        assert any(
            action.action_id == "account.chatgpt.setup" for action in owner.snapshot.setup_actions
        )
        third = await owner.send("Show the controls for this turn again.")
        assert third.tool_results[0].status == "observed"
        current_actions = {action.id for action in owner.snapshot.setup_actions if action.available}
        assert first_actions.isdisjoint(current_actions)
        assert all(
            owner.mark_setup_action_displayed(identity)["status"] == "displayed"
            for identity in current_actions
        )
        assert all(
            owner.mark_setup_action_displayed(identity)["code"] == "stale_action"
            for identity in first_actions
        )
    finally:
        await owner.close()
