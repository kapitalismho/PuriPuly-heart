from __future__ import annotations

import json

import pytest

from puripuly_heart.app.control_presentation import SETTINGS_PRESENTATION
from puripuly_heart.app.ports.settings_secrets import SettingsSecretKey
from puripuly_heart.app.services.agent_chat.context import AgentActivity, collect_agent_context
from puripuly_heart.app.services.agent_chat.modules import (
    MODULE_FIELDS,
    MODULE_KEYS,
    MODULES,
    TOPICS,
    model_context,
    model_receipt,
    topic_modules,
)
from puripuly_heart.app.services.agent_chat.provider_setup import PROVIDER_SETUPS
from puripuly_heart.app.services.agent_chat.tools import (
    LANGUAGE_FIELD_PATHS,
    SETTINGS_FIELD_PATHS,
)
from puripuly_heart.app.services.application_audio_devices import safe_audio_device_label
from puripuly_heart.ui.agent_chat_model import tool_label
from tests.app.test_agent_chat import Records, answer, context_of, setting
from tests.app.test_agent_chat_lifecycle import call, notices_done, watch
from tests.app.test_agent_chat_routing import selected_decision
from tests.app.test_agent_chat_routing_integration import RoutedResponses, owner_for
from tests.app.test_agent_chat_tools import Control
from tests.app.test_agent_chat_tools import catalog as catalog


def test_every_setting_key_and_provider_belongs_to_a_module():
    fields = [field for module in MODULE_FIELDS.values() for field in module]
    assert len(fields) == len(set(fields))
    assert set(fields) == set(SETTINGS_PRESENTATION) - {"languages", "audio", "translation"}
    keys = {key for module in MODULE_KEYS.values() for key in module}
    assert keys == {item.value for item in SettingsSecretKey}
    assert {setup.secret_key.value for setup in PROVIDER_SETUPS.values()} <= keys
    assert TOPICS == {"overview", "diagnostics", *MODULES} - {"core"}
    assert topic_modules("overview") == MODULES
    assert topic_modules("diagnostics") == ("core",)
    assert topic_modules("accounts") == ("core", "accounts")
    with pytest.raises(ValueError):
        topic_modules("settings")


@pytest.mark.asyncio
async def test_model_context_lists_each_setting_once_by_tool_field_without_private_values(
    catalog,
):
    records = Records()
    records.values["settings.choices"] = catalog
    records.values["settings.current"]["private_presence"] = {
        "prompts.system_prompt_override": True,
        "local_llm.base_url": False,
    }
    snapshot = await collect_agent_context(records, activity=AgentActivity())
    view = model_context(snapshot, localize=str)
    text = json.dumps(view, ensure_ascii=False)
    assert view["modules"] == list(MODULES)
    assert "PRIVATE-" not in text
    assert not {"settings_catalog", "choices", "schema", "value_kind", "local_settings"} & {
        key for module in MODULES for key in view[module]
    }
    assert '"schema"' not in text and '"value_kind"' not in text
    for path in (*SETTINGS_FIELD_PATHS.values(), *LANGUAGE_FIELD_PATHS.values()):
        if path not in SETTINGS_PRESENTATION:
            assert f'"{path}"' not in text
    assert setting(view, "languages.source")["value"] == "ko"
    assert setting(view, "languages.secondary_target")["value"] == ""
    assert setting(view, "languages.target")["options_from"] == "languages.source"
    assert setting(view, "languages.secondary_target")["empty_label"]
    assert "official_byok" in setting(view, "translation.connection")["options"]["gpt_6_luna"]
    for field, source in (
        ("peer_stt.provider", "stt.provider"),
        ("translation.previous_llm_model", "translation.model"),
        ("translation.connection_history", "translation.connection"),
        ("peer_expected_languages", "languages.source"),
    ):
        assert setting(view, field)["options_from"] == source
    assert setting(view, "stt.cloud_free_tier_providers")["options"]["deepgram"] == (
        setting(view, "stt.provider")["options"]["deepgram"]
    )
    assert setting(view, "peer_vad.hangover_ms")["unit"] == "ms"
    private = setting(view, "prompts.system_prompt_override")
    assert private["private"] == "saved"
    assert setting(view, "local_llm.base_url")["private"] == "empty"
    assert setting(view, "stt.custom.endpoint")["private"] == "unknown"
    assert view["translation"]["keys"]["openai_api_key"] is True
    assert "keys" not in view["audio"]
    assert view["accounts"]["actions"] == [
        "account.chatgpt.setup",
        "account.qq.setup",
        "account.discord.setup",
        "account.chatgpt.usage",
    ]
    assert view["core"]["revision"] == 42
    assert view["core"]["diagnostics"]["records"] == []
    assert "health_issues" not in view["core"]["diagnostics"]


@pytest.mark.asyncio
async def test_topic_read_returns_core_and_requested_module_only():
    records = Records()
    snapshot = await collect_agent_context(records, activity=AgentActivity(), topic="output")
    view = model_context(snapshot, "output", str)
    assert view["modules"] == ["core", "output"]
    assert view["output"]["osc"]["remote_delivery"] == "unacknowledged"
    assert setting(view, "overlay.background_alpha")["value"] == 0.6


def test_model_receipt_reports_settings_by_tool_field():
    receipt = model_receipt(
        {
            "status": "applied",
            "context": {
                "revision": 3,
                "settings": {
                    "overlay.desktop_flet.visual.background_alpha": 0.3,
                    "languages.source_language": "ja",
                },
                "private_presence": {},
            },
        }
    )
    assert receipt["context"]["settings"] == {
        "overlay.background_alpha": 0.3,
        "languages.source": "ja",
    }
    assert receipt["context"]["revision"] == 3


def test_empty_device_and_language_values_are_not_reported_as_redacted():
    assert safe_audio_device_label("") == ""
    assert safe_audio_device_label("   ") == ""
    assert safe_audio_device_label("Synthetic microphone") == "Synthetic microphone"


def test_every_context_topic_has_its_own_status_subject():
    labels = {topic: tool_label(("get_app_context", topic), done=True) for topic in TOPICS}
    assert len(set(labels.values())) == len(TOPICS)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["selected", "full"])
async def test_selected_context_separates_omission_from_failed_observations(mode):
    records = Records()
    records.fail.add("secrets.presence")
    transport = RoutedResponses(selected_decision("translation", "languages"), answer())
    owner = owner_for(transport, records, context_mode=mode)
    await owner.start()
    try:
        assert (await owner.send("Inspect translation languages.")).status == "completed"
        context = context_of(transport.requests[-1])
        expected = list(MODULES) if mode == "full" else ["core", "translation", "languages"]
        assert context["modules"] == expected
        assert context["missing"] == ["secrets.presence"]
        assert "keys" not in context["translation"]
        assert setting(context, "languages.source")["value"] == "ko"
        assert context["core"]["revision"] == 42
        assert "PRIVATE-" not in json.dumps(context)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_additional_reads_join_same_task_but_do_not_leak_into_new_task():
    transport = RoutedResponses(
        selected_decision("translation"),
        call("get_app_context", {"topic": "audio"}),
        answer("Audio inspected."),
        answer("Language inspected."),
        answer("General settings inspected."),
    )
    owner = owner_for(transport)
    await owner.start()
    try:
        await owner.send("Investigate translation input.")
        assert context_of(transport.requests[0])["modules"] == ["core", "translation"]
        read = next(
            item
            for item in transport.requests[1]["input"]
            if item.get("type") == "function_call_output"
        )
        assert json.loads(read["output"])["context"]["modules"] == ["core", "audio"]
        transport.routing = selected_decision("languages", relation="continue")
        await owner.send("Could the language be wrong?")
        assert context_of(transport.requests[-1])["modules"] == [
            "core",
            "audio",
            "translation",
            "languages",
        ]
        classified = json.loads(transport.classifications[-1]["input"][0]["content"])
        assert classified["previous_modules"] == ["core", "audio", "translation"]
        transport.routing = selected_decision("general")
        await owner.send("Now inspect clipboard settings.")
        assert context_of(transport.requests[-1])["modules"] == ["core", "general"]
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["reset", "expiry", "trim"])
async def test_reset_expiry_and_trim_discard_previous_module_scope(boundary):
    clock = [0.0]
    transport = RoutedResponses(selected_decision("audio"), answer(), answer())
    owner = owner_for(transport, clock=lambda: clock[0])
    await owner.start()
    try:
        await owner.send("Inspect microphone.")
        if boundary == "reset":
            owner.new_conversation()
        elif boundary == "expiry":
            clock[0] = 1800
        else:
            owner._conversation._token_limit = 1
        transport.routing = selected_decision("translation", relation="continue")
        await owner.send("Inspect translation.")
        classified = json.loads(transport.classifications[-1]["input"][0]["content"])
        assert classified["previous_modules"] == []
        assert classified["active_task"] is None
        assert context_of(transport.requests[-1])["modules"] == ["core", "translation"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_key_notice_restores_its_source_task_after_an_unrelated_message():
    transport = RoutedResponses(
        selected_decision("translation"), answer(), answer(), answer(), answer()
    )
    owner = owner_for(transport)
    owner.bind_navigation(lambda *args: {"opened": True})
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        await owner.send("Set up translation.")
        await watch(owner)
        transport.routing = selected_decision("audio")
        await owner.send("Which microphone is selected?")
        owner.key_verification_completed("openai", True)
        await notices_done(owner)
        assert len(transport.classifications) == 2
        assert context_of(transport.requests[-1])["modules"] == ["core", "translation"]
        transport.routing = selected_decision("languages", relation="continue")
        await owner.send("Use that for Japanese.")
        classified = json.loads(transport.classifications[-1]["input"][0]["content"])
        assert classified["active_task"] == "Set up translation."
        assert context_of(transport.requests[-1])["modules"] == ["core", "translation", "languages"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_queued_approval_notice_preserves_proposal_modules_across_a_detour(catalog):
    control = Control(catalog)
    transport = RoutedResponses(
        selected_decision("recognition"),
        call(
            "change_settings",
            {
                "changes": {
                    "stt.provider": "deepgram",
                    "languages": {"source": "ja"},
                    "overlay.background_alpha": 0.3,
                }
            },
        ),
        answer("Review the settings."),
        answer("Clipboard inspected."),
        answer("Approved settings applied."),
    )
    owner = owner_for(transport, control)
    await owner.start()
    try:
        await owner.send("Change recognition, language and overlay.")
        transport.routing = selected_decision("general")
        await owner.send("What does clipboard do?")
        while owner.snapshot.confirmation is not None:
            card = owner.snapshot.confirmation
            assert (await owner.confirm(card["id"], True))["status"] == "applied"
            await notices_done(owner)
        assert len(transport.classifications) == 2
        context = context_of(transport.requests[-1])
        assert context["modules"] == ["core", "recognition", "languages", "output"]
        assert setting(context, "stt.provider")["value"] == "deepgram"
        assert setting(context, "languages.source")["value"] == "ja"
        assert setting(context, "overlay.background_alpha")["value"] == 0.3
    finally:
        await owner.close()
