from __future__ import annotations

import asyncio
import json
import runpy
import socket
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import pytest_asyncio

from puripuly_heart.app.adapters.ui_runtime import UiManagedRuntimeAdapter
from puripuly_heart.app.ports.chatgpt_account import ChatGptAccountSnapshot
from puripuly_heart.app.ports.oscquery import OscQueryServiceInfo
from puripuly_heart.app.ports.settings_secrets import (
    SettingsSecretKey,
    SettingsSecretValuesSnapshot,
)
from puripuly_heart.app.services.agent_chat.context import AgentActivity, collect_agent_context
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.app.services.agent_chat.provider_setup import (
    NAVIGATION_PROVIDERS,
    PROVIDER_SETUPS,
    ProviderSetupLinkKind,
)
from puripuly_heart.app.services.agent_chat.tools import (
    LANGUAGE_FIELD_PATHS,
    SETTINGS_FIELD_PATHS,
    SETTINGS_FIELDS,
    AgentTools,
    project_audio_diagnosis,
    project_settings,
    settings_tool_schema,
)
from puripuly_heart.app.services.application_control import ApplicationControlOwner
from puripuly_heart.app.services.application_control_events import ControlEvents
from puripuly_heart.app.services.managed.managed_auth import ManagedAuthState
from puripuly_heart.app.services.managed.managed_usage import (
    ManagedUsageOwner,
    ManagedUsageState,
)
from puripuly_heart.app.services.provider.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.openrouter.openrouter_metadata import OpenRouterKeyMetadata
from puripuly_heart.release_evidence.agent_chat_simulator.app import choice_catalog
from tests.app.test_agent_chat import Records, Responses
from tests.app.test_osc_control_runtime import (
    FakeReceiverOwner,
    FakeSender,
    RetryDiscoveryService,
)


@pytest_asyncio.fixture
async def catalog(monkeypatch):
    monkeypatch.setattr(
        "puripuly_heart.app.services.application_audio_devices.enumerate_audio_devices",
        lambda _settings: {
            "host_apis": [],
            "microphones": [],
            "loopback_outputs": [],
            "errors": [],
        },
    )
    application = SimpleNamespace(
        http_extension_registry=lambda: None,
        list_loopback_capture_options=lambda: (),
        current_loopback_capture_option_value=lambda: "",
    )
    owner = SimpleNamespace(
        application=application,
        locale_choices=("ko", "en", "ja"),
        gpu=lambda: None,
        provisioning=lambda: None,
    )
    return await ApplicationControlOwner._settings_choice_catalog(owner, {}, AppSettingsVNext())


class Control:
    def __init__(self, catalog):
        self.catalog = deepcopy(catalog)
        self.settings = to_dict(AppSettingsVNext())
        self.revision = 11
        self.submissions = []
        self.waits = []
        self.status = "applied"
        self.receipt_extra = {}
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.hold = False
        self.fail_current = False
        self.audio_devices = {"host_apis": [], "microphones": [], "errors": {}}
        self.app = {"translation_enabled": False}
        self.presence = {key.value: True for key in SettingsSecretKey}
        self.accounts = {}
        self.fail_secrets = False
        self.channels = {
            "self": {"desired_active": False, "effective_active": False},
            "peer": {"desired_active": False, "effective_active": False},
        }
        self.events = ControlEvents(instance_id="agent-tools-test")

    def context(self):
        return {
            **project_settings({"revision": self.revision, "settings": self.settings}),
            "app": deepcopy(self.app),
            "channels": deepcopy(self.channels),
        }

    async def query(self, name, arguments):
        if name == "settings.choices":
            return deepcopy(self.catalog)
        if name == "app.status":
            return {**self.app, "channels": deepcopy(self.channels)}
        if name == "audio.devices":
            return deepcopy(self.audio_devices)
        if name == "secrets.presence":
            if self.fail_secrets:
                raise RuntimeError("PRIVATE-SECRET-FAILURE")
            return {"presence": deepcopy(self.presence)}
        if name == "accounts.status":
            return deepcopy(self.accounts)
        assert name == "settings.current"
        if self.fail_current:
            raise RuntimeError("PRIVATE-EXCEPTION")
        return {"revision": self.revision, "settings": deepcopy(self.settings)}

    def subscribe(self, **kwargs):
        return self.events.subscribe(channel=None, after=None, **kwargs)

    async def submit(self, command, arguments, *, request_id, expected_revision):
        assert command == "settings.apply"
        self.submissions.append(
            {"changes": deepcopy(arguments["changes"]), "expected_revision": expected_revision}
        )
        return {
            "operation_id": str(len(self.submissions)),
            "request_id": request_id,
            "status": "accepted",
            "terminal": False,
            "revision": self.revision,
        }

    async def wait(self, operation_id):
        self.waits.append(operation_id)
        self.entered.set()
        if self.hold:
            await self.release.wait()
        submission = self.submissions[int(operation_id) - 1]
        if submission["expected_revision"] != self.revision:
            return {
                "operation_id": operation_id,
                "status": "rejected",
                "terminal": True,
                "revision": self.revision,
                "error": {"code": "revision_conflict", "current_revision": self.revision},
            }
        if self.status in {"applied", "degraded"}:
            for field, value in submission["changes"].items():
                values = (
                    {LANGUAGE_FIELD_PATHS[key]: item for key, item in value.items()}
                    if field == "languages"
                    else {SETTINGS_FIELD_PATHS[field]: value}
                )
                for path, item in values.items():
                    target = self.settings["intent"]
                    parts = path.split(".")
                    for part in parts[:-1]:
                        target = target[part]
                    target[parts[-1]] = deepcopy(item)
            self.revision += 1
        return {
            "operation_id": operation_id,
            "status": self.status,
            "terminal": True,
            "revision": self.revision,
            **deepcopy(self.receipt_extra),
        }


def adapter(control, *, watching=False):
    actions, events, changes = [], [], []

    def emit(event):
        events.append(deepcopy(event))
        return watching

    tools = AgentTools(
        control,
        changed=lambda: changes.append(True),
        record_action=actions.append,
        emit_event=emit,
    )
    tools.observe_context(control.context())
    return tools, actions, events


@pytest.mark.asyncio
async def test_actual_receipt_approved_revert_and_following_navigation(catalog):
    control = Control(catalog)
    tools, actions, events = adapter(control)
    try:
        first = await tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.4}}
        )
        assert first["status"] == "needs_confirmation"
        assert not control.submissions
        first = await tools.confirm(first["confirmation"]["id"], True)
        assert first["receipt"]["terminal"] is True
        assert control.waits == ["1"]
        assert control.submissions[0]["expected_revision"] == 11
        tools.observe_context(first["context"])
        second = await tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.8}}
        )
        assert second["status"] == "needs_confirmation"
        assert len(control.submissions) == 1
        second = await tools.confirm(second["confirmation"]["id"], True)
        assert second["status"] == "applied"
        tools.observe_context(second["context"])
        revert = await tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.4}}
        )
        assert revert["status"] == "needs_confirmation"
        assert len(control.submissions) == 2
        reverted = await tools.confirm(revert["confirmation"]["id"], True)
        assert reverted["status"] == "applied"
        assert control.submissions[-1] == {
            "changes": {"overlay.background_alpha": 0.4},
            "expected_revision": second["receipt"]["revision"],
        }
        assert (
            reverted["context"]["settings"][SETTINGS_FIELD_PATHS["overlay.background_alpha"]] == 0.4
        )
        tools.bind_navigation(
            lambda section, provider: {
                "opened": section == "overlay",
                "field_visible": False,
                "completion_observable": False,
            }
        )
        opened = await tools.execute("open_settings", {"section": "overlay", "provider": None})
        assert opened["opened"] is True
        assert events[-1]["kind"] == "navigation_requested"
        assert [action["phase"] for action in actions] == [
            "submitted",
            "completed",
            "submitted",
            "completed",
            "submitted",
            "completed",
        ]
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_confirmation_replacement_decline_and_captured_revision_conflict(catalog):
    control = Control(catalog)
    tools, actions, events = adapter(control)
    try:
        request = {"changes": {"stt.provider": "deepgram"}}
        needed = await tools.execute("change_settings", request)
        first = needed["confirmation"]
        assert needed["status"] == "needs_confirmation"
        assert first["changes"] == request["changes"]
        assert first["revision"] == 11
        assert not control.submissions
        declined = await tools.confirm(first["id"], False)
        assert declined["status"] == "declined"
        assert not control.submissions
        assert control.context()["settings"]["stt.provider"] != "deepgram"
        assert events[-1]["kind"] == "confirmation"
        old = (await tools.execute("change_settings", request))["confirmation"]
        replacement = (
            await tools.execute("change_settings", {"changes": {"stt.provider": "soniox"}})
        )["confirmation"]
        assert tools.confirmation["id"] == replacement["id"]
        assert (await tools.confirm(old["id"], True))["status"] == "rejected"
        control.revision += 1
        tools.observe_context(control.context())
        conflicted = await tools.confirm(replacement["id"], True)
        assert control.submissions == [
            {"changes": {"stt.provider": "soniox"}, "expected_revision": 11}
        ]
        assert conflicted["status"] == "rejected"
        assert conflicted["receipt"]["error"]["code"] == "revision_conflict"
        assert conflicted["context"]["revision"] == 12
        assert conflicted["context"]["settings"]["stt.provider"] != "soniox"
        assert events[-1]["result"] == conflicted
        assert tools.confirmation is None
        assert actions[-1]["phase"] == "completed"
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_withdraw_exact_pending_confirmation_preserves_completed_settings(catalog):
    control = Control(catalog)
    tools, actions, events = adapter(control)
    try:
        proposed = await tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.4}}
        )
        await tools.confirm(proposed["confirmation"]["id"], True)
        before = control.context()
        card = (await tools.execute("change_settings", {"changes": {"stt.provider": "deepgram"}}))[
            "confirmation"
        ]
        prior_actions, prior_events = deepcopy(actions), deepcopy(events)
        result = await tools.execute("withdraw_confirmation", {"card_id": card["id"]})
        assert result == {"status": "withdrawn", "card_id": card["id"]}
        assert tools.confirmation is None
        assert control.context() == before
        assert len(control.submissions) == 1
        assert actions == prior_actions and events == prior_events
        assert (await tools.confirm(card["id"], True))["error"][
            "code"
        ] == "confirmation_unavailable"
        assert (
            control.context()["settings"][SETTINGS_FIELD_PATHS["overlay.background_alpha"]] == 0.4
        )
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"card_id": None},
        {"card_id": 1},
        {"card_id": ""},
        {"card_id": []},
        {"card_id": "unknown"},
        {"card_id": "unknown", "approved": True},
    ],
)
async def test_withdraw_invalid_or_unknown_card_keeps_current_proposal(catalog, arguments):
    control = Control(catalog)
    tools, actions, events = adapter(control)
    try:
        card = (await tools.execute("change_settings", {"changes": {"stt.provider": "deepgram"}}))[
            "confirmation"
        ]
        before = control.context()
        result = await tools.execute("withdraw_confirmation", arguments)
        assert result["status"] == "rejected"
        assert tools.confirmation == card
        assert control.context() == before
        assert not control.submissions and not actions and not events
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_withdraw_stale_or_already_applied_card_cannot_change_settings(catalog):
    control = Control(catalog)
    tools, actions, events = adapter(control)
    try:
        old = (await tools.execute("change_settings", {"changes": {"stt.provider": "deepgram"}}))[
            "confirmation"
        ]
        current = (await tools.execute("change_settings", {"changes": {"stt.provider": "soniox"}}))[
            "confirmation"
        ]
        before = control.context()
        rejected = await tools.execute("withdraw_confirmation", {"card_id": old["id"]})
        assert rejected["error"]["code"] == "confirmation_unavailable"
        assert tools.confirmation == current
        assert control.context() == before
        assert not control.submissions and not actions and not events
        assert (await tools.confirm(current["id"], True))["status"] == "applied"
        applied = control.context()
        prior_actions, prior_events = deepcopy(actions), deepcopy(events)
        rejected = await tools.execute("withdraw_confirmation", {"card_id": current["id"]})
        assert rejected["error"]["code"] == "confirmation_unavailable"
        assert tools.confirmation is None
        assert control.context() == applied
        assert applied["settings"]["stt.provider"] == "soniox"
        assert len(control.submissions) == 1
        assert actions == prior_actions and events == prior_events
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_active_interruption_approval_groups_exact_translation_choice(catalog):
    control = Control(catalog)
    tools, _, events = adapter(control)
    control.app["translation_enabled"] = True
    control.channels["self"]["effective_active"] = True
    try:
        vad = await tools.execute(
            "change_settings", {"changes": {"self_vad.speech_threshold": 0.7}}
        )
        assert vad["status"] == "needs_confirmation"
        assert not control.submissions
        tools.cancel_confirmation()
        changes = {"translation.model": "gpt_6_luna", "translation.connection": "official_byok"}
        needed = await tools.execute("change_settings", {"changes": changes})
        card = needed["confirmation"]
        changes["translation.model"] = "local_llm"
        assert card["changes"]["translation.model"] == "gpt_6_luna"
        assert not control.submissions
        applied = await tools.confirm(card["id"], True)
        assert applied["status"] == "applied"
        assert control.submissions[0] == {"changes": card["changes"], "expected_revision": 11}
        assert events[-1]["kind"] == "confirmation"
        control.app["translation_enabled"] = False
        control.channels["self"]["effective_active"] = False
        tools.observe_context(applied["context"])
        local = await tools.execute(
            "change_settings",
            {"changes": {"translation.model": "managed_gemma", "translation.connection": "cpu"}},
        )
        assert local["status"] == "needs_confirmation"
        assert len(control.submissions) == 1
        assert (await tools.confirm(local["confirmation"]["id"], True))["status"] == "applied"
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_strict_real_schema_excludes_private_content_and_uses_catalog_bounds(catalog):
    schema = settings_tool_schema(catalog)
    assert set(schema["properties"]) == SETTINGS_FIELDS
    assert set(schema["required"]) == SETTINGS_FIELDS
    assert schema["additionalProperties"] is False
    language_schema = schema["properties"]["languages"]["anyOf"][0]
    assert set(language_schema["required"]) == set(LANGUAGE_FIELD_PATHS)
    assert schema["properties"]["stt.cloud_free_tier_providers"]["anyOf"][0] == {
        "type": "array",
        "items": {
            "type": "string",
            "enum": catalog["choices"]["stt.cloud_free_tier_providers"],
        },
        "minItems": 1,
        "uniqueItems": True,
    }
    control = Control(catalog)
    control.catalog["field_schemas"]["overlay.background_alpha"]["maximum"] = 0.75
    tools, _, _ = adapter(control)
    try:
        for excluded in (
            "secrets.openai_api_key",
            "auth.login",
            "consent.peer_translation",
            "osc.host",
            "osc.send_port",
            "osc.receive_port",
            "osc.connection_mode",
            "local_llm.base_url",
        ):
            result = await tools.execute("change_settings", {"changes": {excluded: None}})
            assert result["status"] == "rejected"
        for changes in (
            {"overlay.background_alpha": 0.9},
            {"overlay.background_alpha": True},
            {"overlay.background_alpha": float("nan")},
            {"stt.provider": "local_parakeet_v3"},
            {"languages": {"source": "invented"}},
            {"languages": {"api_key": None}},
            {"languages": {}},
            {"languages": {"source": None}},
            {"translation.model": "gpt_6_luna"},
            {"translation.model": "gpt_6_luna", "translation.connection": "cpu"},
            {"stt.cloud_free_tier_providers": []},
            {"stt.cloud_free_tier_providers": ["deepgram", "deepgram"]},
            {"stt.cloud_free_tier_providers": ["soniox"]},
            {"stt.cloud_free_tier_providers": ["unknown"]},
            {"stt.cloud_free_tier_providers": ["deepgram", None]},
            {"stt.cloud_free_tier_providers": "deepgram"},
        ):
            assert (await tools.execute("change_settings", {"changes": changes}))[
                "status"
            ] == "rejected"
        assert not control.submissions
        values = dict.fromkeys(SETTINGS_FIELDS)
        values["languages"] = dict.fromkeys(LANGUAGE_FIELD_PATHS)
        values["languages"]["source"] = "en"
        applied = await tools.execute("change_settings", {"changes": values})
        assert applied["status"] == "needs_confirmation"
        assert not control.submissions
        applied = await tools.confirm(applied["confirmation"]["id"], True)
        assert applied["status"] == "applied"
        assert control.submissions[0]["changes"] == {"languages": {"source": "en"}}
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_free_cloud_membership_uses_current_catalog_without_selecting_paid_provider(catalog):
    control = Control(catalog)
    tools, _, _ = adapter(control)
    field = "stt.cloud_free_tier_providers"
    original_provider = control.context()["settings"]["stt.provider"]
    try:
        assert control.context()["settings"][field] == ["gemini_transcribe"]
        applied = await tools.execute("change_settings", {"changes": {field: ["deepgram"]}})
        assert applied["status"] == "needs_confirmation"
        assert not control.submissions
        applied = await tools.confirm(applied["confirmation"]["id"], True)
        assert applied["status"] == "applied"
        assert applied["context"]["settings"][field] == ["deepgram"]
        assert applied["context"]["settings"]["stt.provider"] == original_provider
        assert control.submissions == [{"changes": {field: ["deepgram"]}, "expected_revision": 11}]
        applied["context"]["settings"][field].append("soniox")
        assert control.context()["settings"][field] == ["deepgram"]
        tools.observe_context(applied["context"])
        restored = await tools.execute(
            "change_settings", {"changes": {field: ["gemini_transcribe"]}}
        )
        assert restored["status"] == "needs_confirmation"
        restored = await tools.confirm(restored["confirmation"]["id"], True)
        assert restored["status"] == "applied"
        assert restored["context"]["settings"][field] == ["gemini_transcribe"]
        tools.observe_context(restored["context"])
        changes = {"stt.provider": "rolling_free", field: ["deepgram"]}
        selected = await tools.execute("change_settings", {"changes": changes})
        assert selected["status"] == "needs_confirmation"
        selected = await tools.confirm(selected["confirmation"]["id"], True)
        assert selected["status"] == "applied"
        assert selected["context"]["settings"]["stt.provider"] == "rolling_free"
        assert selected["context"]["settings"][field] == ["deepgram"]
        tools.observe_context(selected["context"])
        control.catalog["choices"][field] = ["gemini_transcribe"]
        rejected = await tools.execute("change_settings", {"changes": {field: ["deepgram"]}})
        assert rejected["status"] == "rejected"
        assert len(control.submissions) == 3
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("active_channel", ["self", "peer", None])
async def test_free_cloud_membership_preserves_disruptive_approval_and_exact_changes(
    catalog, active_channel
):
    control = Control(catalog)
    control.settings["intent"]["stt"]["provider"] = "rolling_free"
    control.settings["intent"]["peer_stt"]["provider"] = "rolling_free"
    if active_channel is None:
        control.channels = {}
    else:
        control.channels[active_channel]["effective_active"] = True
    tools, _, events = adapter(control)
    field = "stt.cloud_free_tier_providers"
    changes = {field: ["deepgram"]}
    try:
        needed = await tools.execute("change_settings", {"changes": changes})
        assert needed["status"] == "needs_confirmation"
        card = needed["confirmation"]
        channels = ("self", "peer") if active_channel is None else (active_channel,)
        for channel in channels:
            assert f"This change may interrupt {channel} recognition." in card["effects"]
        assert not any("paid" in effect for effect in card["effects"])
        assert control.context()["settings"][field] == ["gemini_transcribe"]
        assert control.submissions == []
        declined = await tools.confirm(card["id"], False)
        assert declined["status"] == "declined"
        assert control.context()["settings"][field] == ["gemini_transcribe"]
        needed = await tools.execute("change_settings", {"changes": changes})
        changes[field].append("soniox")
        card = needed["confirmation"]
        assert card["changes"][field] == ["deepgram"]
        applied = await tools.confirm(card["id"], True)
        assert applied["status"] == "applied"
        assert control.submissions == [{"changes": {field: ["deepgram"]}, "expected_revision": 11}]
        assert applied["context"]["settings"][field] == ["deepgram"]
        assert applied["context"]["settings"]["stt.provider"] == "rolling_free"
        assert events[-1]["result"] == applied
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_free_cloud_membership_keeps_revision_conflicts_and_paid_switch_policy(catalog):
    control = Control(catalog)
    control.channels["self"]["effective_active"] = True
    tools, _, _ = adapter(control)
    field = "stt.cloud_free_tier_providers"
    try:
        needed = await tools.execute("change_settings", {"changes": {field: ["deepgram"]}})
        card = needed["confirmation"]
        control.revision += 1
        tools.observe_context(control.context())
        conflicted = await tools.confirm(card["id"], True)
        assert conflicted["status"] == "rejected"
        assert conflicted["receipt"]["error"]["code"] == "revision_conflict"
        assert conflicted["context"]["settings"][field] == ["gemini_transcribe"]
        assert control.submissions == [{"changes": {field: ["deepgram"]}, "expected_revision": 11}]
        control.channels["self"]["effective_active"] = False
        needed = await tools.execute(
            "change_settings",
            {"changes": {"stt.provider": "deepgram", field: ["deepgram"]}},
        )
        assert needed["status"] == "needs_confirmation"
        assert len(control.submissions) == 1
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_navigation_hidden_field_precedes_navigation_and_requires_real_watch(catalog):
    control = Control(catalog)
    tools, _, events = adapter(control)
    try:

        def navigate(section, provider):
            assert events[-1] == {
                "kind": "navigation_requested",
                "section": "api",
                "provider": "openai",
            }
            return {
                "opened": True,
                "field_visible": False,
                "completion_observable": False,
                "verification_available": False,
            }

        headless = await tools.execute("open_settings", {"section": "api", "provider": "openai"})
        assert headless["opened"] is False
        assert headless["verification_available"] is False
        tools.bind_navigation(navigate)
        result = await tools.execute("open_settings", {"section": "api", "provider": "openai"})
        assert result["opened"] is True
        assert result["field_visible"] is False
        assert result["completion_observable"] is False
        assert result["verification_available"] is False
        assert not control.submissions
        assert "local_llm" not in NAVIGATION_PROVIDERS
        assert (await tools.execute("open_settings", {"section": "api", "provider": "unknown"}))[
            "status"
        ] == "rejected"
        assert (
            await tools.execute(
                "open_settings", {"section": "api", "provider": "openai", "key": "PRIVATE-KEY"}
            )
        )["status"] == "rejected"
        assert (
            await tools.execute(
                "open_settings",
                {"section": "api", "provider": "openai", "url": "https://attacker.invalid"},
            )
        )["status"] == "rejected"
        assert "PRIVATE-KEY" not in json.dumps(events)
    finally:
        await tools.close()
    watched, _, _ = adapter(control, watching=True)
    watched.bind_navigation(
        lambda *_: {
            "opened": True,
            "field_visible": True,
            "completion_observable": False,
            "verification_available": True,
        }
    )
    try:
        result = await watched.execute("open_settings", {"section": "api", "provider": "openai"})
        assert result["completion_observable"] is True
        assert result["verification_available"] is True
    finally:
        await watched.close()


@pytest.mark.asyncio
async def test_interrupted_caller_retains_terminal_receipt_and_revert_conflicts(catalog):
    control = Control(catalog)
    control.hold = True
    control.status = "degraded"
    control.receipt_extra = {
        "error": {
            "code": "runtime_not_ready",
            "explanation": "The selected provider is not effective.",
            "api_key": "PRIVATE-KEY",
        },
        "transaction": {
            "status": "settings_commit_success_runtime_degraded",
            "message": "settings.runtime_degraded",
        },
    }
    tools, actions, _ = adapter(control)
    proposal = await tools.execute(
        "change_settings", {"changes": {"overlay.background_alpha": 0.6}}
    )
    caller = asyncio.create_task(tools.confirm(proposal["confirmation"]["id"], True))
    await control.entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    control.release.set()
    await tools.close()
    assert [action["phase"] for action in actions] == ["submitted", "completed"]
    result = actions[-1]["result"]
    assert result["status"] == "degraded"
    assert result["receipt"]["error"] == {"code": "runtime_not_ready"}
    assert result["receipt"]["transaction"]["status"] == "settings_commit_success_runtime_degraded"
    assert "explanation" not in result["receipt"]["error"]
    assert "message" not in result["receipt"]["transaction"]
    assert "PRIVATE-KEY" not in json.dumps(actions)
    control.hold = False
    control.status = "applied"
    control.receipt_extra = {}
    tools, _, _ = adapter(control)
    try:
        applied = await tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.7}}
        )
        applied = await tools.confirm(applied["confirmation"]["id"], True)
        assert applied["status"] == "applied"
        tools.observe_context(applied["context"])
        revert = await tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.6}}
        )
        control.revision += 1
        reverted = await tools.confirm(revert["confirmation"]["id"], True)
        assert reverted["status"] == "rejected"
        assert reverted["receipt"]["error"]["code"] == "revision_conflict"
        assert (
            reverted["context"]["settings"][SETTINGS_FIELD_PATHS["overlay.background_alpha"]] == 0.7
        )
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_visible_snapshot_revision_never_refreshes_and_projection_failure_keeps_receipt(
    catalog,
):
    control = Control(catalog)
    tools, _, _ = adapter(control)
    try:
        applied = await tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.4}}
        )
        applied = await tools.confirm(applied["confirmation"]["id"], True)
        assert applied["status"] == "applied"
        stale = await tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.8}}
        )
        stale = await tools.confirm(stale["confirmation"]["id"], True)
        assert stale["status"] == "rejected"
        assert [item["expected_revision"] for item in control.submissions] == [11, 11]
        tools.observe_context(stale["context"])
        control.status = "degraded"
        control.fail_current = True
        degraded = await tools.execute(
            "change_settings", {"changes": {"overlay.background_alpha": 0.5}}
        )
        degraded = await tools.confirm(degraded["confirmation"]["id"], True)
        assert degraded["status"] == "degraded"
        assert degraded["receipt"]["terminal"] is True
        assert degraded["context_unavailable"] is True
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("following_tool", ["change_settings", "open_settings", "context_batch"])
async def test_owner_cannot_execute_following_tool_while_confirmation_is_pending(
    catalog, following_tool
):
    def response(call_id, changes):
        return {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "change_settings",
                    "arguments": json.dumps({"changes": changes}),
                    "status": "completed",
                }
            ],
        }

    class Transport:
        def __init__(self):
            self.allowed = []
            self.responses = [
                response("paid", {"stt.provider": "deepgram"}),
                (
                    response("rogue", {"overlay.background_alpha": 0.9})
                    if following_tool == "change_settings"
                    else {
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "rogue",
                                "status": "completed",
                                "name": "open_settings",
                                "arguments": '{"section":"overlay"}',
                            }
                        ],
                    }
                ),
            ]
            if following_tool == "context_batch":
                self.responses[1] = {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": f"context-{index}",
                            "name": "get_app_context",
                            "arguments": json.dumps({"topic": topic}),
                            "status": "completed",
                        }
                        for index, topic in enumerate(("settings", "accounts"))
                    ],
                }

        async def prepare(self):
            pass

        async def classify(self, *, input, turn_id):
            return {
                "answers": [
                    {"type": "choice", "name": "route", "choice": "work"},
                    {"type": "choice", "name": "effort", "choice": "high"},
                ]
            }

        async def respond(self, *, input, turn_id, effort, route, allow_tools):
            self.allowed.append(allow_tools)
            yield {"type": "completed", "response": self.responses.pop(0)}

        async def close(self):
            self.responses.clear()

    control = Control(catalog)
    transport = Transport()
    owner = AgentChatOwner(control, transport, localize=str)
    navigations = []
    owner.bind_navigation(lambda *args: navigations.append(args) or {"opened": True})
    await owner.start()
    try:
        result = await owner.send("Switch recognition to Deepgram.")
        assert result.status == "needs_confirmation"
        assert transport.allowed == [True, False]
        following_count = 2 if following_tool == "context_batch" else 1
        assert [item.status for item in result.tool_results] == (
            ["needs_confirmation"] + ["not_executed"] * following_count
        )
        assert [item.result["code"] for item in result.tool_results[1:]] == [
            "confirmation_pending"
        ] * following_count
        assert not control.submissions
        assert not navigations
        card = owner.snapshot.confirmation
        assert card["changes"] == {"stt.provider": "deepgram"}
        approved = await owner.confirm(card["id"], True)
        assert approved["status"] == "applied"
        assert control.submissions == [{"changes": card["changes"], "expected_revision": 11}]
    finally:
        await owner.close()


class RuntimeControl(Control):
    async def submit(self, command, arguments, *, request_id):
        self.submissions.append(
            {"command": command, "arguments": deepcopy(arguments), "request_id": request_id}
        )
        return {
            "operation_id": str(len(self.submissions)),
            "request_id": request_id,
            "status": "accepted",
            "terminal": False,
        }

    async def wait(self, operation_id):
        self.waits.append(operation_id)
        self.entered.set()
        if self.hold:
            await self.release.wait()
        return {
            "operation_id": operation_id,
            "status": self.status,
            "terminal": True,
            **deepcopy(self.receipt_extra),
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ("applied", "degraded", "rejected"))
async def test_osc_discovery_retry_uses_empty_native_command_and_safe_actual_receipt(
    catalog, status
):
    control = RuntimeControl(catalog)
    control.status = status
    control.receipt_extra = {
        "retry_started": status != "rejected",
        "effect": "discovery_target_applied" if status == "applied" else "discovery_unavailable",
        "reason": "automatic_mode_required" if status == "rejected" else "target_not_found",
        "message": "PRIVATE-PROSE",
        "error": {"message": "PRIVATE-ERROR", "nested": {"host": "PRIVATE-HOST"}},
        "output": {
            "configured_mode": "automatic",
            "effective_mode": "automatic",
            "query_service_started": False,
            "discovered_remote_present": False,
            "query_failure": None,
            "local_availability": "starting",
            "remote_delivery": "unacknowledged",
            "destination_source": "discovered" if status == "applied" else "fallback",
            "local_listener": {
                "status": "matched",
                "vrchat_running": True,
                "observed_at": 10.5,
                "host": "PRIVATE-HOST",
                "port": 19000,
                "pid": 19234,
                "raw_log": "PRIVATE-LOG",
                "unexpected": {"text": "PRIVATE-PROSE"},
            },
            "configured_host": "PRIVATE-HOST",
            "configured_send_port": 19000,
            "configured_receive_port": 19001,
            "effective_send_port": 9000,
            "effective_receive_port": 49152,
            "query_advertised_port": 49152,
            "discovered_remote": {"host": "PRIVATE-REMOTE", "send_port": 19000},
        },
    }
    tools, actions, events = adapter(control)
    before = control.context()
    try:
        for arguments in (
            {"host": "PRIVATE-HOST"},
            {"send_port": 19000},
            {"connection": "automatic"},
            {"text": "PRIVATE-TEXT"},
            {"destination": "PRIVATE-REMOTE"},
        ):
            assert (await tools.execute("retry_osc_discovery", arguments))["status"] == "rejected"
            assert (
                AgentChatOwner._arguments(
                    {"name": "retry_osc_discovery", "arguments": json.dumps(arguments)}
                )
                is None
            )
        assert not control.submissions
        assert AgentChatOwner._arguments({"name": "retry_osc_discovery", "arguments": "{}"}) == {}
        result = await tools.execute("retry_osc_discovery", {})
        assert result["status"] == status
        assert result["receipt"]["terminal"] is True
        assert result["receipt"]["retry_started"] is (status != "rejected")
        assert result["receipt"]["effect"] == control.receipt_extra["effect"]
        assert result["receipt"]["reason"] == control.receipt_extra["reason"]
        assert result["receipt"]["output"] == {
            key: value
            for key, value in control.receipt_extra["output"].items()
            if key
            in {
                "configured_mode",
                "effective_mode",
                "query_service_started",
                "discovered_remote_present",
                "query_failure",
                "local_availability",
                "remote_delivery",
            }
        } | {
            "destination_source": control.receipt_extra["output"]["destination_source"],
            "local_listener": {"status": "matched", "vrchat_running": True, "observed_at": 10.5},
        }
        assert control.submissions[0]["command"] == "osc.discovery.retry"
        assert control.submissions[0]["arguments"] == {}
        assert control.waits == ["1"]
        assert control.context() == before
        assert "PRIVATE-" not in json.dumps(actions)
        assert not events
        assert [action["phase"] for action in actions] == ["submitted", "completed"]
        assert actions[-1]["result"] == result
        assert tools.confirmation is None
    finally:
        await tools.close()


@pytest_asyncio.fixture
async def composed_osc_agent(tmp_path, monkeypatch, request):
    from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresence
    from puripuly_heart.app.wiring import wiring_runtime_pipeline, wiring_vrc_mic_sync
    from puripuly_heart.composition import application_runtime
    from puripuly_heart.composition.headless_application import compose_headless_application

    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "synthetic-osc-agent-test")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("PURIPULY_AGENT_RELAY_URL", "")

    class IsolatedSender(FakeSender):
        def close(self):
            pass

        def send_chatbox(self, _text):
            pytest.fail("OSC recovery must not publish Chatbox text")

    sender = IsolatedSender()
    monkeypatch.setattr(wiring_runtime_pipeline, "VrchatOscUdpSender", lambda **_: sender)
    receiver = FakeReceiverOwner()
    monkeypatch.setattr(wiring_vrc_mic_sync, "VrcMicSyncOwner", lambda **_: receiver)
    service = RetryDiscoveryService(
        OscQueryServiceInfo(
            service_id="controlled-VRChat",
            host="127.0.0.1",
            osc_send_port=19000,
            is_vrchat=True,
        )
    )
    owners = []
    compose_osc = application_runtime.compose_vrc_mic_sync

    async def observe(_host, _port):
        return VrchatOscPresence(None, None)

    def compose(**kwargs):
        kwargs["local_listener_probe"] = observe
        owner = compose_osc(**kwargs, query_service=service)
        owners.append(owner)
        return owner

    monkeypatch.setattr(application_runtime, "compose_vrc_mic_sync", compose)
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(
                settings.intent.osc, connection_mode=getattr(request, "param", "automatic")
            ),
            telemetry=replace(settings.intent.telemetry, enabled=False),
            integrated_context=replace(settings.intent.integrated_context, enabled=False),
            stt=replace(settings.intent.stt, provider="deepgram"),
            peer_stt=replace(settings.intent.peer_stt, provider="deepgram"),
            translation=replace(
                settings.intent.translation, model="deepseek_v4_flash", connection="official_byok"
            ),
        ),
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    app = compose_headless_application(path)
    actions = []
    tools = AgentTools(
        app.control(),
        changed=lambda: None,
        record_action=actions.append,
        emit_event=lambda _: False,
    )
    try:
        app.control().bind_instance("composed-osc-agent")
        await app.start()
        assert len(owners) == 1
        await owners[0].wait_automatic_query_start()
        yield SimpleNamespace(
            app=app,
            owner=owners[0],
            service=service,
            receiver=receiver,
            tools=tools,
            actions=actions,
            path=path,
        )
    finally:
        service.discovery_gate.set()
        await tools.close()
        await app.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_failure", (True, False))
async def test_composed_agent_retry_awaits_target_application_and_records_it(
    composed_osc_agent, initial_failure
):
    fixture = composed_osc_agent
    control = fixture.app.control()
    capability = control.capabilities()["commands"]["osc.discovery.retry"]
    assert capability["arguments"] == {}
    assert capability["cancellable"] is False
    if not initial_failure:
        fixture.service.fail_discovery = False
        fixture.service.info = None
        await fixture.tools.execute("retry_osc_discovery", {})
        assert fixture.owner.output_snapshot()["query_service_started"] is True
        assert fixture.owner.output_snapshot()["discovered_remote_present"] is False
    before = await control.query("settings.current", {})
    persisted = fixture.path.read_bytes()
    fixture.service.fail_discovery = False
    fixture.service.info = OscQueryServiceInfo(
        service_id="recovered-VRChat", host="127.0.0.1", osc_send_port=19000, is_vrchat=True
    )
    fixture.service.block_discovery = True
    receiver_calls = list(fixture.receiver.control_calls)
    caller = asyncio.create_task(fixture.tools.execute("retry_osc_discovery", {}))
    try:
        await asyncio.wait_for(fixture.service.discovery_started.wait(), 5)
        assert not caller.done()
        assert fixture.actions[-1]["phase"] == "submitted"
        fixture.service.discovery_gate.set()
        result = await asyncio.wait_for(caller, 5)
        assert result["status"] == "applied"
        assert result["receipt"]["terminal"] is True
        assert result["receipt"]["effect"] == "discovery_target_applied"
        assert result["receipt"]["output"]["destination_source"] == "discovered"
        assert result["receipt"]["output"]["remote_delivery"] == "unacknowledged"
        current = await control.query("osc.status", {})
        assert current["output"]["effective_send_port"] == 19000
        assert current["output"]["discovered_remote_present"] is True
        assert (await control.query("settings.current", {})) == before
        assert fixture.path.read_bytes() == persisted
        assert fixture.receiver.control_calls == receiver_calls
        assert fixture.actions[-1]["phase"] == "completed"
        assert fixture.actions[-1]["result"] == result
    finally:
        fixture.service.discovery_gate.set()
        await caller


@pytest.mark.asyncio
async def test_composed_agent_cancellation_keeps_admitted_osc_retry_and_actual_action(
    composed_osc_agent,
):
    fixture = composed_osc_agent
    fixture.service.fail_discovery = False
    fixture.service.block_discovery = True
    transport = Responses(
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_retry",
                    "call_id": "retry",
                    "name": "retry_osc_discovery",
                    "arguments": "{}",
                    "status": "completed",
                }
            ],
        }
    )
    chat = AgentChatOwner(fixture.app.control(), transport, localize=str)
    await chat.start()
    turn = asyncio.create_task(chat.send("Recover automatic OSC discovery"))
    try:
        await asyncio.wait_for(fixture.service.discovery_started.wait(), 5)
        context = await chat._context(topic="output")
        submitted = context["agent_operations"][-1]
        assert submitted["phase"] == "submitted"
        operation_id = submitted["receipt"]["operation_id"]
        chat.cancel()
        assert (await turn).status == "cancelled"
        fixture.service.discovery_gate.set()
        receipt = await fixture.app.control().wait(operation_id)
        await chat.close()
        assert receipt["status"] == "applied"
        assert receipt["effect"] == "discovery_target_applied"
        context = await chat._context(topic="output")
        completed = context["agent_operations"][-1]
        assert completed["phase"] == "completed"
        assert completed["result"]["receipt"]["operation_id"] == operation_id
        assert completed["result"]["receipt"]["effect"] == "discovery_target_applied"
        assert fixture.owner.output_snapshot()["effective_send_port"] == 19000
        assert len(transport.requests) == 1
    finally:
        fixture.service.discovery_gate.set()
        await chat.close()
        await turn


@pytest.mark.asyncio
@pytest.mark.parametrize("composed_osc_agent", ("off", "manual"), indirect=True)
async def test_composed_agent_retry_preserves_unsupported_mode_and_destinations(composed_osc_agent):
    fixture = composed_osc_agent
    control = fixture.app.control()
    before = await control.query("settings.current", {})
    persisted = fixture.path.read_bytes()
    output = fixture.owner.output_snapshot()
    calls = list(fixture.receiver.control_calls), fixture.service.started, fixture.service.stopped
    result = await fixture.tools.execute("retry_osc_discovery", {})
    assert result["status"] == "rejected"
    assert result["receipt"]["reason"] == "automatic_mode_required"
    assert result["receipt"]["retry_started"] is False
    assert fixture.owner.output_snapshot() == output
    assert (
        fixture.receiver.control_calls,
        fixture.service.started,
        fixture.service.stopped,
    ) == calls
    assert (await control.query("settings.current", {})) == before
    assert fixture.path.read_bytes() == persisted


@pytest.mark.asyncio
async def test_composed_agent_retry_resolves_local_host_info_before_applying_udp_target(
    composed_osc_agent, monkeypatch
):
    from pythonosc.osc_message import OscMessage

    from puripuly_heart.core.osc.oscquery import ZeroconfOscQueryService
    from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender

    fixture = composed_osc_agent
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []
    handlers = set()
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.setblocking(False)
    target_port = receiver.getsockname()[1]

    async def http_request(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
            target = request.split(b" ", 2)[1]
            requests.append(target)
            if target == b"/?HOST_INFO":
                entered.set()
                await release.wait()
                payload = {"OSC_IP": "127.0.0.1", "OSC_PORT": target_port}
            else:
                assert target == b"/avatar"
                payload = {}
            body = json.dumps(payload).encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(body)).encode("ascii")
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(http_request, "127.0.0.1", 0)
    discovery = ZeroconfOscQueryService()
    discovery._services["controlled-VRChat"] = OscQueryServiceInfo(
        service_id="controlled-VRChat",
        host="127.0.0.1",
        query_port=server.sockets[0].getsockname()[1],
        is_vrchat=True,
    )
    monkeypatch.setattr(fixture.service, "discover_vrchat", discovery.discover_vrchat)
    monkeypatch.setattr(fixture.service, "query_avatar", discovery.query_avatar)

    class IsolatedUdpSender(VrchatOscUdpSender):
        def set_destination(self, host, port):
            assert (host, port) == ("127.0.0.1", target_port)
            super().set_destination(host, port)

        def send_chatbox(self, _text):
            pytest.fail("Discovery recovery must not send Chatbox text")

    sender = IsolatedUdpSender(host="127.0.0.1", port=target_port)
    monkeypatch.setattr(fixture.owner, "_sender_provider", lambda: sender)
    persisted = fixture.path.read_bytes()
    revision = (await fixture.app.control().query("settings.current", {}))["revision"]
    task = asyncio.create_task(fixture.tools.execute("retry_osc_discovery", {}))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not task.done()
        assert fixture.actions[-1]["phase"] == "submitted"
        assert fixture.owner.output_snapshot()["discovered_remote_present"] is False
        release.set()
        result = await asyncio.wait_for(task, 3)
        assert result["status"] == "applied" and result["receipt"]["terminal"]
        assert result["receipt"]["effect"] == "discovery_target_applied"
        assert result["receipt"]["output"]["destination_source"] == "discovered"
        assert result["receipt"]["output"]["remote_delivery"] == "unacknowledged"
        assert sender.host == "127.0.0.1" and sender.port == target_port
        assert fixture.owner.output_snapshot()["effective_send_port"] == target_port
        assert b"/?HOST_INFO" in requests
        sender.send_message("/diagnostics/synthetic_route", 332)
        packet, _address = await asyncio.wait_for(
            asyncio.get_running_loop().sock_recvfrom(receiver, 4096), 1
        )
        message = OscMessage(packet)
        assert message.address == "/diagnostics/synthetic_route"
        assert message.params == [332]
        assert fixture.path.read_bytes() == persisted
        assert (await fixture.app.control().query("settings.current", {}))["revision"] == revision
        assert fixture.actions[-1]["result"] == result
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*tuple(handlers))
        sender.close()
        receiver.close()


@pytest.mark.asyncio
async def test_typed_runtime_controls_preserve_user_only_peer_terms_and_actual_receipts(catalog):
    control = RuntimeControl(catalog)
    tools, actions, events = adapter(control)
    try:
        for name, arguments in (
            ("set_capture", {"channel": "peer", "enabled": True, "accept_terms": True}),
            ("set_capture", {"channel": "peer", "enabled": True, "accept_terms": False}),
            ("set_capture", {"channel": "both", "enabled": True}),
            ("set_capture", {"channel": "self", "enabled": 1}),
            ("set_translation", {"enabled": "true"}),
            ("set_translation", {"enabled": True, "provider": "openai"}),
            ("diagnose_microphone", {"device": "invented"}),
        ):
            assert (await tools.execute(name, arguments))["status"] == "rejected"
        assert not control.submissions
        control.status = "action_required"
        control.receipt_extra = {"action": "accept_peer_translation_terms"}
        peer = await tools.execute("set_capture", {"channel": "peer", "enabled": True})
        assert peer["status"] == "action_required"
        assert peer["receipt"]["action"] == "accept_peer_translation_terms"
        assert control.submissions[-1]["command"] == "capture.set"
        assert control.submissions[-1]["arguments"] == {"channel": "peer", "enabled": True}
        control.status = "applied"
        control.receipt_extra = {"translation_enabled": True}
        translation = await tools.execute("set_translation", {"enabled": True})
        assert translation["receipt"]["translation_enabled"] is True
        assert control.submissions[-1]["command"] == "translation.set"
        assert not events
        assert [action["phase"] for action in actions] == [
            "submitted",
            "completed",
            "submitted",
            "completed",
        ]
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_runtime_controls_project_safe_channel_and_diagnosis_evidence(catalog):
    control = RuntimeControl(catalog)
    tools, actions, _ = adapter(control)
    try:
        control.receipt_extra = {
            "channel": {
                "effective_active": True,
                "runtime_provider": "local_cpu_auto",
                "raw_log": "PRIVATE-LOG",
                "capture": {
                    "generation": 7,
                    "has_source": True,
                    "transcript": "PRIVATE-VOICE",
                },
            }
        }
        capture = await tools.execute("set_capture", {"channel": "self", "enabled": True})
        assert capture["receipt"]["channel"]["effective_active"] is True
        assert capture["receipt"]["channel"]["capture"] == {"generation": 7, "has_source": True}
        control.status = "action_required"
        control.receipt_extra = {
            "recognition": "not_exercised",
            "settings_changed": False,
            "diagnosis": {
                "state": "unavailable",
                "scope": "local_capture_vad",
                "current_route": {"input_host_api": "Windows WASAPI", "input_device": "Mic"},
                "candidates": [
                    {
                        "id": "candidate-1",
                        "input_host_api": "Windows WASAPI",
                        "input_device": "Mic api_key=PRIVATE-KEY",
                        "device_index": 1,
                        "current_route": True,
                        "opened": True,
                        "frames_received": False,
                        "frame_count": 0,
                        "vad_sample_count": 0,
                        "speech_detected": False,
                        "peak_level": 0.0,
                        "failure_code": "no_frames",
                        "cleanup_complete": True,
                        "pcm": "PRIVATE-AUDIO",
                        "exception": "PRIVATE-EXCEPTION",
                    }
                ],
                "selected_candidate_id": None,
                "selection_basis": None,
                "cleanup_complete": True,
                "restoration": {"self": "restored", "peer": "not_needed", "log": "PRIVATE-LOG"},
                "bounds": {"max_candidates": 4, "candidate_timeout_s": 3.0},
                "raw_log": "PRIVATE-LOG",
            },
        }
        diagnosis = await tools.execute("diagnose_microphone", {})
        assert control.submissions[-1]["command"] == "audio.diagnose"
        assert control.submissions[-1]["arguments"] == {}
        receipt = diagnosis["receipt"]
        assert receipt["recognition"] == "not_exercised"
        assert receipt["settings_changed"] is False
        assert receipt["diagnosis"]["state"] == "unavailable"
        assert receipt["diagnosis"]["candidates"][0]["frames_received"] is False
        assert receipt["diagnosis"]["restoration"] == {"self": "restored", "peer": "not_needed"}
        assert "PRIVATE-" not in json.dumps(actions)
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_runtime_action_outlives_cancelled_caller_and_retains_terminal_receipt(catalog):
    control = RuntimeControl(catalog)
    control.hold = True
    tools, actions, _ = adapter(control)
    caller = asyncio.create_task(
        tools.execute("set_capture", {"channel": "self", "enabled": False})
    )
    await control.entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    control.release.set()
    await tools.close()
    assert [action["phase"] for action in actions] == ["submitted", "completed"]
    assert actions[-1]["result"]["receipt"]["terminal"] is True


@pytest.mark.asyncio
async def test_audio_setting_routes_use_current_enumeration_pair_and_interruption_card(catalog):
    control = Control(catalog)
    control.catalog["choices"]["audio.input_host_api"] = [
        {"value": "", "available": True},
        {"value": "Windows WASAPI", "available": True},
        {"value": "MME", "available": True},
        {"value": "Gone host", "available": False},
    ]
    control.catalog["choices"]["audio.input_device"] = [
        {"value": "WASAPI Mic", "host_api": "Windows WASAPI", "available": True},
        {"value": "MME Mic", "host_api": "MME", "available": False},
        {"value": "Gone Mic", "host_api": "MME", "available": False},
    ]
    control.audio_devices = {
        "host_apis": ["Windows WASAPI", "MME"],
        "microphones": [
            {"name": "WASAPI Mic", "host_api": "Windows WASAPI"},
            {"name": "MME Mic", "host_api": "MME"},
        ],
        "errors": {},
    }
    control.channels["self"]["effective_active"] = True
    tools, _, _ = adapter(control)
    try:
        for changes in (
            {"audio.input_host_api": "invented", "audio.input_device": ""},
            {"audio.input_host_api": "Gone host", "audio.input_device": ""},
            {"audio.input_host_api": "MME", "audio.input_device": "WASAPI Mic"},
            {"audio.input_host_api": "MME", "audio.input_device": "Gone Mic"},
        ):
            assert (await tools.execute("change_settings", {"changes": changes}))[
                "status"
            ] == "rejected"
        assert not control.submissions
        changes = {"audio.input_host_api": "MME", "audio.input_device": "MME Mic"}
        needed = await tools.execute("change_settings", {"changes": changes})
        assert needed["status"] == "needs_confirmation"
        assert needed["confirmation"]["changes"] == changes
        assert not control.submissions
        applied = await tools.confirm(needed["confirmation"]["id"], True)
        assert applied["status"] == "applied"
        assert control.submissions[-1]["changes"] == changes
        tools.observe_context(applied["context"])
        control.channels["self"]["effective_active"] = False
        default = await tools.execute("change_settings", {"changes": {"audio.input_device": ""}})
        assert default["status"] == "needs_confirmation"
        default = await tools.confirm(default["confirmation"]["id"], True)
        assert default["status"] == "applied"
        schema = settings_tool_schema(control.catalog)
        for field in ("audio.input_host_api", "audio.input_device"):
            assert schema["properties"][field]["anyOf"][0] == {"type": "string"}
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_provider_setup_is_fixed_public_metadata_without_navigation_or_secret_actions(
    catalog,
):
    control = Control(catalog)
    tools, _, events = adapter(control, watching=True)
    navigations = []
    tools.bind_navigation(lambda *args: navigations.append(args))
    hosts = {
        "aistudio.google.com",
        "ai.google.dev",
        "openrouter.ai",
        "platform.openai.com",
        "platform.deepseek.com",
        "developers.deepgram.com",
        "console.deepgram.com",
        "elevenlabs.io",
        "console.soniox.com",
        "bailian.console.aliyun.com",
        "bailian.console.alibabacloud.com",
        "www.alibabacloud.com",
    }
    try:
        assert set(PROVIDER_SETUPS) == set(NAVIGATION_PROVIDERS)
        fields = set()
        for provider in NAVIGATION_PROVIDERS:
            result = await tools.execute("get_provider_setup", {"provider": provider})
            assert result["status"] == "available"
            field = SettingsSecretKey(result["secure_field"]["secret_key"])
            fields.add(field)
            assert ProviderVerificationBindingOwner.provider_for_secret_key(field.value) == provider
            assert result["secure_field"]["section"] == "api"
            assert result["secure_field"]["provider"] == provider
            assert result["links"]
            if provider in {"google", "gemini_transcribe", "openrouter", "deepgram"}:
                assert {link["kind"] for link in result["links"]} == {
                    "key_console",
                    "key_documentation",
                }
            if provider == "openrouter":
                assert result["browser_authorization"] == {
                    "method": "oauth_pkce",
                    "entry": "user_action_in_app_settings",
                    "requires_selected_connection": "openrouter_byok",
                    "applies_selection_on_success": True,
                    "manual_key_fallback": True,
                }
                result["browser_authorization"]["method"] = "untrusted"
                assert (await tools.execute("get_provider_setup", {"provider": provider}))[
                    "browser_authorization"
                ]["method"] == "oauth_pkce"
            else:
                assert "browser_authorization" not in result
            for link in result["links"]:
                assert ProviderSetupLinkKind(link["kind"]) in ProviderSetupLinkKind
                parsed = urlsplit(link["url"])
                assert parsed.scheme == "https"
                assert parsed.hostname in hosts
                assert not parsed.username and not parsed.password and not parsed.fragment
            assert result["link_rendered"] is False
            assert result["link_opened"] is False
            assert result["field_visibility"] == "not_observed"
            assert result["completion_observable"] is False
            assert result["last_navigation"] is None
            result["links"][0]["url"] = "https://attacker.invalid"
            assert (await tools.execute("get_provider_setup", {"provider": provider}))["links"][0][
                "url"
            ] != "https://attacker.invalid"
        assert len(fields) == len(NAVIGATION_PROVIDERS)
        for arguments in (
            {"provider": "unknown"},
            {"provider": "google", "url": "https://attacker.invalid"},
            {"provider": "openai", "key": "PRIVATE-KEY"},
        ):
            assert (await tools.execute("get_provider_setup", arguments))["status"] == "rejected"
        assert not navigations and not events and not control.submissions
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_provider_setup_marks_navigation_as_last_observation_without_installing_watch(
    catalog,
):
    control = Control(catalog)
    tools, _, events = adapter(control, watching=True)
    tools.bind_navigation(lambda *_: {"opened": True, "field_visible": True})
    try:
        opened = await tools.execute("open_settings", {"section": "api", "provider": "openai"})
        assert opened["completion_observable"] is True
        count = len(events)
        setup = await tools.execute("get_provider_setup", {"provider": "openai"})
        assert len(events) == count
        assert setup["completion_observable"] is False
        assert setup["field_visibility"] == "not_observed"
        assert setup["last_navigation"] == {
            "status": "opened",
            "opened": True,
            "field_visible": True,
            "verification_available": False,
            "completion_observable": True,
            "source": "last_navigation",
        }
        tools.bind_navigation(None)
        assert (await tools.execute("get_provider_setup", {"provider": "openai"}))[
            "last_navigation"
        ] is None
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_initial_context_projects_bounded_local_evidence_route_and_safe_readiness():
    control = Records()
    settings = control.values["settings.current"]["settings"]
    settings["intent"]["audio"]["input_host_api"] = "MME"
    settings["intent"]["audio"]["input_device"] = "Mic"
    settings["intent"]["stt"]["cloud_free_tier_providers"] = ["deepgram", "gemini_transcribe"]
    settings["state"]["provider_verification"]["deepgram"] = {
        "status": "verified",
        "secret_fingerprint": "PRIVATE-FINGERPRINT",
        "verifier_context": {"raw_log": "PRIVATE-LOG"},
    }
    settings["state"]["provider_verification"]["openai"] = {
        "status": "PRIVATE-PROSE",
        "secret_fingerprint": "PRIVATE-FINGERPRINT",
    }
    control.values["secrets.presence"]["effective_presence"] = {
        "openai_api_key": False,
        "deepgram_api_key": True,
        "google_api_key": None,
        "custom_name": "PRIVATE-SECRET",
    }
    control.values["audio.devices"] = {
        "host_apis": ["MME"],
        "microphones": [
            {
                "name": "Mic",
                "host_api": "MME",
                "selected_host_api": True,
                "transcript": "PRIVATE-VOICE",
            },
            {"name": "Mic api_key=PRIVATE-KEY", "host_api": "MME"},
        ],
        "errors": {"loopback_outputs": "PRIVATE-EXCEPTION"},
    }
    ids = []
    for index in range(14):
        diagnostic, _ = control.history.retain(
            ErrorDiagnostics(
                component="audio_diagnosis",
                operation="audio.diagnose",
                code="audio_diagnose_candidate",
                category="lifecycle",
                visibility="basic",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                cause_id=None,
                fields={
                    "target_index": index % 4,
                    "frame_count": index,
                    "frames_received": index > 0,
                    "raw_log": "PRIVATE-LOG",
                },
            ),
            severity="info",
            disposition="recovered",
        )
        ids.append(diagnostic["diagnostic_id"])
    context = await collect_agent_context(control, activity=AgentActivity(), diagnostic_id=ids[0])
    assert context["audio"]["current_route"] == {"input_host_api": "MME", "input_device": "Mic"}
    assert context["audio"]["availability"] == "available"
    assert context["audio"]["unavailable_components"] == ["loopback_outputs"]
    assert context["audio"]["microphones"][0]["selectable"] is True
    assert context["audio"]["microphones"][1]["selectable"] is False
    assert context["settings"]["stt.cloud_free_tier_providers"] == ["deepgram", "gemini_transcribe"]
    assert context["provider_verification"]["deepgram"] == {"status": "verified"}
    assert context["provider_verification"]["openai"] == {"status": "unknown"}
    assert context["secret_presence"] == {"openai_api_key": False, "deepgram_api_key": True}
    assert len(context["recent_diagnostics"]) == 10
    assert context["recent_diagnostics"][0]["diagnostic_id"] == ids[0]
    assert context["recent_diagnostics"][1]["diagnostic_id"] == ids[-1]
    assert context["recent_diagnostics"][1]["fields"]["frame_count"] == 13
    assert context["diagnostics_availability"]["history"] == "retained_current_run"
    audio = await collect_agent_context(
        control, activity=AgentActivity(), topic="audio", diagnostic_id=ids[0]
    )
    for key in ("audio", "provider_verification", "recent_diagnostics"):
        assert audio[key] == context[key]
    translation = await collect_agent_context(
        control, activity=AgentActivity(), topic="translation"
    )
    assert translation["secret_presence"] == context["secret_presence"]
    assert "PRIVATE-" not in json.dumps([context, audio, translation])


@pytest.mark.asyncio
async def test_initial_context_preserves_empty_unavailable_and_query_failed_states():
    control = Records()
    control.values["audio.devices"] = {"host_apis": [], "microphones": [], "errors": {}}
    empty = await collect_agent_context(control, activity=AgentActivity())
    assert empty["audio"]["availability"] == "available"
    assert empty["audio"]["microphones"] == []
    assert empty["recent_diagnostics"] == []
    control.values["audio.devices"]["errors"] = {"microphones": "PRIVATE-EXCEPTION"}
    unavailable = await collect_agent_context(control, activity=AgentActivity())
    assert unavailable["audio"]["availability"] == "unavailable"
    assert unavailable["audio"]["unavailable_components"] == ["microphones"]
    control.values["audio.devices"] = {}
    unknown = await collect_agent_context(control, activity=AgentActivity())
    assert unknown["audio"]["availability"] == "unknown"
    control.fail = {
        "audio.devices",
        "diagnostics.history",
        "diagnostics.health",
        "accounts.status",
        "settings.current",
        "settings.choices",
        "secrets.presence",
    }
    failed = await collect_agent_context(control, activity=AgentActivity())
    assert failed["audio"]["availability"] == "query_failed"
    assert failed["diagnostics_availability"] == {
        "history": "query_failed",
        "health": "query_failed",
    }
    assert set(failed["missing"]) == control.fail
    assert failed["channels"]["self"]["effective_active"] is True
    assert failed["accounts"]["chatgpt"]["signed_in"] is None
    assert failed["accounts"]["managed"]["credential_present"] is None
    assert failed["accounts"]["managed"]["budget"]["availability"] == "unknown"
    assert failed["accounts"]["managed"]["budget"]["remaining_percent"] is None
    assert failed["audio"]["current_route"] == {"input_host_api": None, "input_device": None}
    assert not {"settings", "choices", "provider_verification", "secret_presence"} & failed.keys()
    assert "PRIVATE-" not in json.dumps([empty, unavailable, failed])


@pytest.mark.asyncio
async def test_audio_setting_choice_projection_drops_private_fields_and_marks_redacted_routes():
    control = Records()
    control.values["settings.choices"]["choices"]["audio.input_device"] = [
        {
            "value": "Mic api_key=PRIVATE-KEY",
            "host_api": "MME",
            "available": True,
            "raw_log": "PRIVATE-LOG",
            "exception": "PRIVATE-EXCEPTION",
        }
    ]
    result = await collect_agent_context(control, activity=AgentActivity(), topic="audio")
    initial = await collect_agent_context(control, activity=AgentActivity())
    assert initial["choices"] == result["choices"]
    choice = result["choices"]["audio.input_device"][0]
    assert choice["host_api"] == "MME"
    assert choice["available"] is True
    assert choice["selectable"] is False
    assert "PRIVATE-" not in json.dumps([result, initial])


@pytest.mark.asyncio
async def test_generator_omits_enumerated_local_audio_values(catalog, tmp_path, monkeypatch):
    catalog["choices"]["audio.input_host_api"] = [{"value": "PRIVATE-HOST", "available": True}]
    catalog["choices"]["audio.input_device"] = [
        {"value": "PRIVATE-DEVICE", "host_api": "PRIVATE-HOST", "available": True}
    ]
    input_path = tmp_path / "choices.json"
    output_path = tmp_path / "app-tools.ts"
    input_path.write_text(json.dumps(catalog), encoding="utf-8")
    generator = Path(__file__).resolve().parents[2] / "agent-relay/generate-tools.py"
    monkeypatch.setattr(
        sys, "argv", [str(generator), str(input_path), "--output", str(output_path)]
    )
    runpy.run_path(str(generator), run_name="__main__")
    assert "PRIVATE-" not in output_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_committed_relay_tools_match_generator_output():
    root = Path(__file__).resolve().parents[2]
    generator = runpy.run_path(str(root / "agent-relay/generate-tools.py"))
    committed = (root / "agent-relay/src/app-tools.ts").read_text(encoding="utf-8")
    assert json.loads(
        committed.removeprefix("export const appTools = ").removesuffix(";\n")
    ) == generator["app_tools"](await choice_catalog())


def test_diagnosis_projection_preserves_measured_ranges_and_bounds_numeric_samples():
    result = project_audio_diagnosis(
        {
            "scope": "local_capture_vad",
            "candidates": [
                {
                    "input_host_api": "MME",
                    "input_device": "Mic",
                    "speech_probability_min": 0.1,
                    "speech_probability_max": 0.72,
                    "speech_threshold": 0.95,
                    "observed_speech_probabilities": [
                        0.1,
                        0.55,
                        0.72,
                        float("nan"),
                        float("inf"),
                        -1,
                        2,
                        "PRIVATE-VOICE",
                    ],
                    "transcript": "PRIVATE-VOICE",
                    "exception": "PRIVATE-EXCEPTION",
                }
                for _ in range(7)
            ],
        }
    )
    assert len(result["candidates"]) == 4
    candidate = result["candidates"][0]
    assert candidate["speech_probability_min"] == 0.1
    assert candidate["speech_probability_max"] == 0.72
    assert candidate["speech_threshold"] == 0.95
    assert candidate["observed_speech_probabilities"] == [0.1, 0.55, 0.72]
    assert "PRIVATE-" not in json.dumps(result, allow_nan=False)


@pytest.mark.asyncio
async def test_runtime_command_failure_retains_safe_action_without_private_exception(catalog):
    control = RuntimeControl(catalog)

    async def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE-EXCEPTION")

    control.submit = fail
    tools, actions, _ = adapter(control)
    try:
        result = await tools.execute("diagnose_microphone", {})
        assert result["status"] == "unavailable"
        assert result["error"]["code"] == "audio_diagnose_outcome_unknown"
        assert result["request_id"] == actions[-1]["action_id"]
        assert [action["phase"] for action in actions] == ["completed"]
        assert actions[-1]["result"] == result
        assert "PRIVATE-" not in json.dumps(actions)
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "presence", "cost", "readiness", "status"),
    (
        ("gemini_transcribe", True, "free_credit_or_allowance", "present", "needs_confirmation"),
        ("gemini_transcribe", False, "free_credit_or_allowance", "absent", "needs_confirmation"),
        ("gemini_transcribe", None, "free_credit_or_allowance", "unknown", "needs_confirmation"),
        ("deepgram", True, "free_credit_or_allowance", "present", "needs_confirmation"),
        ("deepgram", False, "free_credit_or_allowance", "absent", "needs_confirmation"),
        ("deepgram", None, "free_credit_or_allowance", "unknown", "needs_confirmation"),
        ("elevenlabs_scribe", True, "free_credit_or_allowance", "present", "needs_confirmation"),
        ("soniox", True, "provider_billed", "present", "needs_confirmation"),
    ),
)
async def test_recognition_cost_and_readiness_follow_observed_credential(
    catalog, provider, presence, cost, readiness, status
):
    control = Control(catalog)
    control.settings["intent"]["stt"]["provider"] = "local_cpu_auto"
    key = f"{provider}_api_key"
    control.presence = {key: presence} if presence is not None else {}
    tools, _, _ = adapter(control)
    try:
        result = await tools.execute("change_settings", {"changes": {"stt.provider": provider}})
        assert result["status"] == status
        details = result["confirmation"] if status == "needs_confirmation" else result
        expected = {
            "scope": "self_recognition",
            "provider": provider,
            "cost": cost,
            "credential_requirement": "api_key",
            "credential_readiness": readiness,
        }
        if cost == "free_credit_or_allowance":
            expected.update(
                allowance_kind=(
                    "trial_credit" if provider == "deepgram" else "recurring_free_allowance"
                ),
                allowance_remaining=None,
            )
        assert details["route_effects"] == [expected]
        assert bool(control.submissions) is (status == "applied")
        if provider == "deepgram" and presence is True:
            assert len(details["effects"]) == 1
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ("self", "peer"))
async def test_free_tier_ready_switch_still_requires_active_capture_approval(catalog, channel):
    control = Control(catalog)
    field = "stt.provider" if channel == "self" else "peer_stt.provider"
    target = control.settings["intent"]["stt" if channel == "self" else "peer_stt"]
    target["provider"] = "local_cpu_auto"
    control.channels[channel]["effective_active"] = True
    tools, _, _ = adapter(control)
    try:
        result = await tools.execute("change_settings", {"changes": {field: "gemini_transcribe"}})
        card = result["confirmation"]
        assert result["status"] == "needs_confirmation"
        assert card["route_effects"][0]["cost"] == "free_credit_or_allowance"
        assert card["route_effects"][0]["credential_readiness"] == "present"
        assert len(card["effects"]) == 1
        assert not control.submissions
        approved = await tools.confirm(card["id"], True)
        assert approved["status"] == "applied"
        assert approved["route_effects"] == card["route_effects"]
        assert control.submissions == [
            {"changes": {field: "gemini_transcribe"}, "expected_revision": 11}
        ]
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("members", "presence", "readiness", "status"),
    (
        (
            ["deepgram", "gemini_transcribe"],
            {"deepgram_api_key": False, "gemini_transcribe_api_key": True},
            "present",
            "needs_confirmation",
        ),
        (
            ["deepgram"],
            {"deepgram_api_key": False, "gemini_transcribe_api_key": True},
            "absent",
            "needs_confirmation",
        ),
        (
            ["deepgram", "gemini_transcribe"],
            {"deepgram_api_key": False},
            "unknown",
            "needs_confirmation",
        ),
    ),
)
async def test_rolling_free_requires_any_selected_member_not_an_unselected_key(
    catalog, members, presence, readiness, status
):
    control = Control(catalog)
    control.settings["intent"]["stt"]["provider"] = "local_cpu_auto"
    control.presence = presence
    tools, _, _ = adapter(control)
    try:
        result = await tools.execute(
            "change_settings",
            {
                "changes": {
                    "stt.provider": "rolling_free",
                    "stt.cloud_free_tier_providers": members,
                }
            },
        )
        assert result["status"] == status
        details = result["confirmation"] if status == "needs_confirmation" else result
        route = details["route_effects"][0]
        assert route["cost"] == "free_tier_eligible"
        assert route["credential_requirement"] == "any_free_tier_key"
        assert route["credential_readiness"] == readiness
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "connection", "cost", "readiness", "status"),
    (
        ("gemini_flash", "official_byok", "free_tier_eligible", "present", "needs_confirmation"),
        ("gemini_flash", "openrouter", "provider_billed", "present", "needs_confirmation"),
        ("gpt_6_luna", "official_byok", "provider_billed", "present", "needs_confirmation"),
        ("deepseek_v4_flash_41", "managed", "managed_budget", "present", "needs_confirmation"),
        (
            "deepseek_v4_flash_41",
            "managed_china",
            "managed_budget",
            "unknown",
            "needs_confirmation",
        ),
        ("gpt_6_luna", "chatgpt", "subscription", "present", "needs_confirmation"),
        ("index_translate_35b", "public_api", "public", "not_required", "needs_confirmation"),
        ("managed_gemma", "cpu", "local", "not_required", "needs_confirmation"),
        ("local_llm", "ollama", "local", "not_required", "needs_confirmation"),
        ("custom_http", "custom_http", "custom", "not_required", "needs_confirmation"),
    ),
)
async def test_translation_cost_follows_connection_and_keeps_account_budgets_separate(
    catalog, model, connection, cost, readiness, status
):
    control = Control(catalog)
    control.settings["intent"]["translation"]["model"] = "gemma4_26b_31b"
    control.settings["intent"]["translation"]["connection"] = "openrouter"
    control.accounts = {
        "chatgpt": {"signed_in": True, "sign_in_required": False},
        "managed": {
            "provider": "discord",
            "credential_present": True,
            "budget": {"remaining_percent": 31, "availability": "cached"},
        },
    }
    tools, _, _ = adapter(control)
    try:
        result = await tools.execute(
            "change_settings",
            {
                "changes": {
                    "translation.model": model,
                    "translation.connection": connection,
                }
            },
        )
        assert result["status"] == status
        details = result["confirmation"] if status == "needs_confirmation" else result
        route = details["route_effects"][0]
        assert route["cost"] == cost
        assert route["credential_readiness"] == readiness
        if connection == "managed":
            assert route["budget"]["remaining_percent"] == 31
        else:
            assert "budget" not in route
        if (
            connection in {"managed", "chatgpt", "openrouter", "official_byok"}
            and cost != "free_tier_eligible"
        ):
            assert len(details["effects"]) == 1
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_failed_credential_observation_is_unknown_not_missing(catalog):
    control = Control(catalog)
    control.settings["intent"]["stt"]["provider"] = "local_cpu_auto"
    control.fail_secrets = True
    tools, _, _ = adapter(control)
    try:
        result = await tools.execute(
            "change_settings", {"changes": {"stt.provider": "gemini_transcribe"}}
        )
        assert result["confirmation"]["route_effects"][0]["credential_readiness"] == "unknown"
        assert "PRIVATE-" not in json.dumps(result)
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True))
async def test_accounts_context_allowlists_coarse_state_without_auth_refresh_or_identity(failed):
    control = Records()
    control.values["accounts.status"] = {
        "chatgpt": {
            "available": True,
            "signed_in": None,
            "in_progress": False,
            "sign_in_required": None,
            "email": "PRIVATE-EMAIL",
            "token": "PRIVATE-TOKEN",
            "usage": {"remaining_percent": 99},
        },
        "managed": {
            "provider": "qq",
            "available": True,
            "selected": True,
            "credential_present": True,
            "in_progress": False,
            "qq_identity": "PRIVATE-QQ",
            "budget": {
                "remaining_percent": 42,
                "availability": "cached",
                "freshness": "PRIVATE-CLAIM",
                "remaining_uses": 800,
                "entitlement_ref": "PRIVATE-ENTITLEMENT",
            },
        },
        "challenge": {"url": "PRIVATE-URL", "credential": "PRIVATE-SECRET"},
        "puri": {"allowance": {"remaining": 50}},
    }
    if failed:
        control.fail.add("accounts.status")
    result = await collect_agent_context(control, activity=AgentActivity(), topic="accounts")
    assert [name for name, _ in control.queries if name.startswith(("accounts.", "auth."))] == [
        "accounts.status"
    ]
    initial = await collect_agent_context(control, activity=AgentActivity())
    assert initial["accounts"] == result["accounts"]
    accounts = result["accounts"]
    assert accounts["chatgpt"]["signed_in"] is None
    assert accounts["chatgpt"]["usage"]["remaining_percent"] is None
    assert accounts["managed"]["budget"]["remaining_percent"] == (None if failed else 42)
    assert accounts["managed"]["budget"]["freshness"] == "unknown"
    assert accounts["managed"]["budget"]["unit"] == "percent"
    assert accounts["puri"]["allowance"]["remaining"] is None
    assert accounts["puri"]["allowance"]["unit"] == "submitted_prompt_turns"
    assert accounts["puri"]["allowance"]["period"] == "server_utc_day"
    assert all(action["requires_user_click"] for action in accounts["actions"])
    assert "PRIVATE-" not in json.dumps([result, initial])
    assert "challenge" not in json.dumps(result)
    assert "remaining_uses" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("availability", "remaining", "expected"),
    (
        ("stale", 55, None),
        ("cached", -1, None),
        ("cached", 101, None),
        ("cached", True, None),
        ("cached", 0, 0),
        ("unknown", None, None),
    ),
)
async def test_accounts_budget_does_not_turn_stale_or_invalid_values_into_known_allowance(
    availability, remaining, expected
):
    control = Records()
    control.values["accounts.status"] = {
        "managed": {
            "budget": {
                "availability": availability,
                "remaining_percent": remaining,
            }
        }
    }
    result = await collect_agent_context(control, activity=AgentActivity(), topic="accounts")
    assert result["accounts"]["managed"]["budget"]["remaining_percent"] == expected
    assert result["accounts"]["managed"]["budget"]["availability"] == (
        "unknown" if availability == "cached" and expected is None else availability
    )
    assert result["accounts"]["chatgpt"]["signed_in"] is None


@pytest.mark.asyncio
async def test_native_accounts_query_reads_existing_usage_without_refresh_or_account_details():
    auth_state = ManagedAuthState(
        settings_available=True,
        managed_selected=True,
        managed_china=False,
        local_key_available=True,
        release_service_available=True,
        runtime_available=True,
        ingress_frozen=False,
    )
    usage_state = ManagedUsageState(
        settings_available=True,
        managed_key_visible=True,
        release_settings_available=True,
        installation_id="PRIVATE-INSTALLATION",
        entitlement_ref="PRIVATE-REF",
        ingress_frozen=False,
    )
    refreshes = []
    usage = ManagedUsageOwner(
        state_provider=lambda: usage_state,
        release_service_provider=lambda: None,
        metadata_fetcher=lambda: refreshes.append("network"),
        pending_sink=lambda _: None,
        view_sink=lambda _: None,
        disable_translation_sink=lambda _: None,
        auto_show_founder_letter_provider=lambda _: False,
        warning_sink=lambda *_: None,
        usage_metadata=OpenRouterKeyMetadata(limit_usd=10, remaining_usd=6.5, usage_usd=3.5),
        usage_metadata_entitlement_ref="PRIVATE-REF",
    )
    adapter = UiManagedRuntimeAdapter(
        SimpleNamespace(
            auth=SimpleNamespace(
                state_provider=lambda: auth_state, dashboard_action=lambda: "continue"
            ),
            usage=usage,
        )
    )
    application = SimpleNamespace(
        managed_account_snapshot=adapter.managed_account_snapshot,
        chatgpt_account_snapshot=lambda: ChatGptAccountSnapshot(
            signed_in=True, email="PRIVATE-EMAIL", in_progress=False
        ),
        chatgpt_sign_in_required=lambda: False,
    )
    owner = SimpleNamespace(
        application=application,
        instance_id="account-context",
        _current_revision=lambda: 11,
        settings=SimpleNamespace(committed_settings=None),
    )
    try:
        result = await ApplicationControlOwner.query(owner, "accounts.status", {})
        assert result["chatgpt"]["signed_in"] is True
        assert result["managed"]["credential_present"] is True
        assert result["managed"]["budget"]["remaining_percent"] == 65
        usage_state = replace(usage_state, entitlement_ref="PRIVATE-NEW-REF")
        stale = await ApplicationControlOwner.query(owner, "accounts.status", {})
        assert stale["managed"]["budget"]["availability"] == "stale"
        assert stale["managed"]["budget"]["remaining_percent"] is None
        assert usage.usage_metadata.remaining_usd == 6.5
        auth_state = replace(auth_state, managed_selected=False)
        deselected = await ApplicationControlOwner.query(owner, "accounts.status", {})
        assert deselected["managed"]["credential_present"] is None
        assert deselected["managed"]["provider"] is None
        assert not refreshes
        assert "PRIVATE-" not in json.dumps((result, stale, deselected))
    finally:
        await usage.close()


@pytest.mark.asyncio
async def test_effective_credentials_include_runtime_environment_and_legacy_sources(
    catalog, monkeypatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "PRIVATE-ENV-KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    snapshot = SettingsSecretValuesSnapshot((("alibaba_api_key", "PRIVATE-LEGACY-KEY"),))
    application = SimpleNamespace(
        http_extension_registry=lambda: None,
        settings_secrets=lambda: SimpleNamespace(
            load_values=lambda _: SimpleNamespace(snapshot=snapshot, diagnostics=None)
        ),
    )
    native = SimpleNamespace(
        application=application,
        instance_id="credential-context",
        _current_revision=lambda: 11,
        settings=SimpleNamespace(committed_settings=None),
    )
    observed = await ApplicationControlOwner.query(native, "secrets.presence", {})
    assert observed["presence"]["gemini_transcribe_api_key"] is False
    assert observed["effective_presence"]["gemini_transcribe_api_key"] is True
    assert observed["effective_presence"]["google_api_key"] is False
    assert observed["effective_presence"]["alibaba_api_key_beijing"] is True
    assert observed["effective_presence"]["alibaba_api_key_singapore"] is True
    assert "PRIVATE-" not in json.dumps(observed)
    control = Control(catalog)
    control.settings["intent"]["stt"]["provider"] = "local_cpu_auto"
    original_query = control.query

    async def query(name, arguments):
        return observed if name == "secrets.presence" else await original_query(name, arguments)

    control.query = query
    tools, _, _ = adapter(control)
    try:
        result = await tools.execute(
            "change_settings", {"changes": {"stt.provider": "gemini_transcribe"}}
        )
        assert result["status"] == "needs_confirmation"
        assert not control.submissions
        assert result["confirmation"]["route_effects"][0]["credential_readiness"] == "present"
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("signed_in", (False, None))
async def test_native_account_unknown_credentials_are_not_signed_out(signed_in):
    application = SimpleNamespace(
        chatgpt_account_snapshot=lambda: ChatGptAccountSnapshot(signed_in=signed_in),
        chatgpt_sign_in_required=lambda: True,
    )
    owner = SimpleNamespace(
        application=application,
        instance_id="unknown-account-context",
        _current_revision=lambda: 11,
        settings=SimpleNamespace(committed_settings=None),
    )
    observed = await ApplicationControlOwner.query(owner, "accounts.status", {})
    assert observed["chatgpt"]["signed_in"] is signed_in
    assert observed["chatgpt"]["sign_in_required"] is (True if signed_in is False else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", (False, True))
async def test_puri_exhaustion_is_separate_observed_evidence_not_an_invented_remaining_count(
    exhausted,
):
    control = Records()
    control.values["accounts.status"] = {
        "managed": {
            "budget": {
                "remaining_percent": 70,
                "availability": "cached",
            }
        }
    }
    context = await collect_agent_context(
        control, activity=AgentActivity(), topic="accounts", puri_allowance_exhausted=exhausted
    )
    allowance = context["accounts"]["puri"]["allowance"]
    assert allowance["exhaustion"] == ("observed" if exhausted else "not_observed")
    assert allowance["remaining"] is None
    assert context["accounts"]["managed"]["budget"]["remaining_percent"] == 70


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_available", [False, True])
async def test_queue_keeps_coupled_routes_and_each_language_block_exact_once(
    catalog, projection_available
):
    control = Control(catalog)
    control.fail_current = not projection_available
    control.catalog["choices"]["audio.input_host_api"] = [{"value": "MME", "available": True}]
    control.audio_devices["microphones"] = [{"name": "Desk Mic", "host_api": "MME"}]
    tools, actions, events = adapter(control)
    changes = {
        "translation.model": "managed_gemma",
        "languages": {"source": "en", "target": "ko", "secondary_target": "ja"},
        "audio.input_host_api": "MME",
        "overlay.background_alpha": 0.4,
        "translation.connection": "cpu",
        "audio.input_device": "Desk Mic",
        "locale": "en",
    }
    original = deepcopy(changes)
    before = control.context()
    expected = [
        {
            "translation.model": "managed_gemma",
            "translation.connection": "cpu",
            "languages": {"source": "en"},
        },
        {"languages": {"target": "ko", "secondary_target": "ja"}},
        {
            "audio.input_host_api": "MME",
            "audio.input_device": "Desk Mic",
            "overlay.background_alpha": 0.4,
        },
        {"locale": "en"},
    ]
    try:
        proposal = await tools.execute("change_settings", {"changes": changes})
        assert proposal["status"] == "needs_confirmation"
        assert not control.submissions and not actions
        assert control.context() == before
        changes["languages"]["target"] = "ja"
        changes["audio.input_device"] = "Replaced Mic"
        card = proposal["confirmation"]
        control.hold = True
        first = asyncio.create_task(tools.confirm(card["id"], True))
        await control.entered.wait()
        assert tools.confirmation is None
        assert len(control.submissions) == 1
        assert (await tools.confirm(card["id"], True))["status"] == "rejected"
        control.release.set()
        result = await first
        control.hold = False
        seen = {card["id"]}
        assert card["changes"] == expected[0]
        assert result["changes"] == expected[0]
        for index, group in enumerate(expected[1:], 1):
            card = tools.confirmation
            assert result["confirmation"] == card
            assert card["id"] not in seen
            seen.add(card["id"])
            assert card["revision"] == 11 + index
            assert card["changes"] == group
            previous = {
                key: (
                    {field: before["settings"][LANGUAGE_FIELD_PATHS[field]] for field in value}
                    if key == "languages"
                    else before["settings"][SETTINGS_FIELD_PATHS[key]]
                )
                for key, value in group.items()
            }
            assert tools.confirmation_display["previous"] == previous
            result = await tools.confirm(card["id"], True)
            assert result["status"] == "applied"
            assert result["changes"] == group
        assert tools.confirmation is None and "confirmation" not in result
        assert control.submissions == [
            {"changes": group, "expected_revision": 11 + index}
            for index, group in enumerate(expected)
        ]
        for key, value in original.items():
            if key == "languages":
                for field, item in value.items():
                    assert control.context()["settings"][LANGUAGE_FIELD_PATHS[field]] == item
            else:
                assert control.context()["settings"][SETTINGS_FIELD_PATHS[key]] == value
        assert [event["resume"] for event in events] == [False, False, False, True]
        assert [row["phase"] for row in actions] == ["submitted", "completed"] * 4
    finally:
        control.release.set()
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("resolution", ["decline", "withdraw", "cancel"])
async def test_stopping_later_card_preserves_completed_changes_and_clears_remainder(
    catalog, resolution
):
    control = Control(catalog)
    tools, _, _ = adapter(control)
    changes = {
        "overlay.background_alpha": 0.4,
        "locale": "en",
        "languages": {"source": "en", "target": "ko", "secondary_target": "ja"},
    }
    try:
        proposal = await tools.execute("change_settings", {"changes": changes})
        await tools.confirm(proposal["confirmation"]["id"], True)
        before = control.context()
        card = tools.confirmation
        if resolution == "decline":
            assert (await tools.confirm(card["id"], False))["status"] == "declined"
        elif resolution == "withdraw":
            assert (await tools.execute("withdraw_confirmation", {"card_id": card["id"]}))[
                "status"
            ] == "withdrawn"
        else:
            tools.cancel_confirmation()
        assert tools.confirmation is None
        assert control.context() == before
        assert len(control.submissions) == 1
        assert (await tools.confirm(card["id"], True))["status"] == "rejected"
        assert len(control.submissions) == 1
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["degraded", "failed", "rejected", "persistence_failed"])
async def test_non_successful_card_stops_all_remaining_cards(catalog, status):
    control = Control(catalog)
    tools, actions, events = adapter(control)
    changes = {"overlay.background_alpha": 0.4, "locale": "en", "stt.provider": "deepgram"}
    try:
        proposal = await tools.execute("change_settings", {"changes": changes})
        control.status = status
        result = await tools.confirm(proposal["confirmation"]["id"], True)
        assert result["status"] == status
        assert tools.confirmation is None
        assert "confirmation" not in result
        assert len(control.submissions) == 1
        assert events[-1]["resume"] is True
        assert actions[-1]["result"] == result
        assert control.context()["settings"]["stt.provider"] != "deepgram"
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_next_card_uses_own_receipt_revision_not_external_current_revision(catalog):
    class ExternalChangeControl(Control):
        async def query(self, name, arguments):
            if name == "settings.current" and self.revision == 12:
                self.revision += 1
            return await super().query(name, arguments)

    control = ExternalChangeControl(catalog)
    tools, _, _ = adapter(control)
    changes = {
        "overlay.background_alpha": 0.4,
        "locale": "en",
        "stt.provider": "deepgram",
        "peer_stt.provider": "deepgram",
        "overlay.translation.enabled": False,
    }
    try:
        proposal = await tools.execute("change_settings", {"changes": changes})
        first = await tools.confirm(proposal["confirmation"]["id"], True)
        assert first["receipt"]["revision"] == 12
        assert first["context"]["revision"] == 13
        card = tools.confirmation
        assert card["revision"] == 12
        tools.observe_context(first["context"])
        conflict = await tools.confirm(card["id"], True)
        assert conflict["status"] == "rejected"
        assert conflict["receipt"]["error"]["code"] == "revision_conflict"
        assert tools.confirmation is None
        assert len(control.submissions) == 2
        assert control.submissions[-1]["expected_revision"] == 12
        assert (await tools.confirm(card["id"], True))["status"] == "rejected"
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_invalid_later_block_never_stages_or_replaces_valid_queue(catalog):
    control = Control(catalog)
    tools, actions, events = adapter(control)
    try:
        proposal = await tools.execute("change_settings", {"changes": {"locale": "en"}})
        current = proposal["confirmation"]
        invalid = await tools.execute(
            "change_settings",
            {
                "changes": {
                    "overlay.background_alpha": 0.4,
                    "stt.provider": "deepgram",
                    "languages": {"source": "not-a-language"},
                }
            },
        )
        assert invalid["status"] == "rejected"
        assert tools.confirmation == current
        assert not control.submissions and not actions and not events
    finally:
        await tools.close()
