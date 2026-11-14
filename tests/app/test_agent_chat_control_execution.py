from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.agent_chat.context import AgentActivity, collect_agent_context
from puripuly_heart.app.services.agent_chat.execution import (
    CLI_ONLY_COMMANDS,
    app_actions,
    check_inventory,
    private_fields,
    project_query,
    safe_receipt,
    validate,
)
from puripuly_heart.app.services.agent_chat.modules import model_context
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.app.services.agent_chat.tools import (
    SETTINGS_FIELDS,
    AgentTools,
    project_settings,
)
from puripuly_heart.app.services.application_control import COMMANDS, QUERIES
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.settings_vnext.serialization import to_dict
from tests.app.test_agent_chat_tools import Control
from tests.app.test_agent_chat_tools import catalog as catalog


class ExecutionControl(Control):
    instance_id = "current-synthetic-instance"

    def __init__(self, catalog):
        super().__init__(catalog)
        self.operations = {}
        self.query_values = {}
        self.terminal_status = "applied"

    def capabilities(self):
        return {
            "instance_id": self.instance_id,
            "commands": COMMANDS,
            "queries": sorted(QUERIES),
            "secret_keys": self.catalog["command_choices"]["secrets.names"],
        }

    async def query(self, name, arguments):
        if name in self.query_values:
            return deepcopy(self.query_values[name])
        return await super().query(name, arguments)

    async def submit(self, command, arguments, *, request_id, expected_revision=None):
        if command == "settings.apply":
            return await super().submit(
                command, arguments, request_id=request_id, expected_revision=expected_revision
            )
        self.submissions.append(
            {"command": command, "arguments": deepcopy(arguments), "revision": expected_revision}
        )
        receipt = {
            "instance_id": self.instance_id,
            "operation_id": f"op-{len(self.submissions)}",
            "request_id": request_id,
            "status": "accepted",
            "terminal": False,
        }
        self.operations[receipt["operation_id"]] = receipt
        return deepcopy(receipt)

    async def wait(self, identity, timeout=None):
        if identity not in self.operations:
            return await super().wait(identity)
        self.waits.append(identity)
        self.operations[identity].update(status=self.terminal_status, terminal=True)
        return deepcopy(self.operations[identity])


def tools_for(control):
    actions, events = [], []
    tools = AgentTools(
        control, changed=lambda: None, record_action=actions.append, emit_event=events.append
    )
    tools.observe_context(control.context())
    return tools, actions, events


async def execute(tools, action, **arguments):
    selected = next(
        (
            item
            for item in app_actions()
            if item.binding.get("command") == action
            or item.binding.get("query") == action.removeprefix("query.")
        ),
        None,
    )
    if selected is None:
        return await tools.execute("manage_proposal", {"action": action, **arguments})
    supplied = {key: value for key, value in arguments.items() if key not in selected.binding}
    return await tools.execute(selected.tool, {selected.key: selected.value, **supplied})


def test_inventory_classifies_every_cli_capability_and_exposes_only_the_chat_subset(catalog):
    check_inventory()
    actions = app_actions()
    commands = {item.binding["command"] for item in actions if "command" in item.binding}
    assert commands == {
        "capture.set",
        "translation.set",
        "overlay.set",
        "audio.retry",
        "models.retry",
        "accounts.managed.refresh",
        "recognition.probe",
        "translation.probe",
        "audio.diagnose",
        "osc.discovery.retry",
        "secrets.delete",
        "auth.logout",
        "overlay.position.reset",
    }
    assert commands.isdisjoint(CLI_ONLY_COMMANDS)
    assert {item.value for item in actions if item.tool == "probe_app"} == {
        "recognition.self",
        "recognition.peer",
        "translation.self",
        "translation.peer",
        "microphone",
        "managed_access",
    }
    assert private_fields().isdisjoint(SETTINGS_FIELDS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_",
    [
        {"action": "app.stop"},
        {"action": "app.restart"},
        {"action": "text.submit"},
        {"action": "models.import", "model_id": "gemma"},
        {"action": "models.download", "model_id": "gemma"},
        {"action": "secrets.set", "name": "openai_api_key"},
        {"action": "auth.login", "provider": "openrouter"},
        {"action": "overlay.lock", "locked": True},
        {"action": "overlay.calibrate", "calibration_action": "begin"},
        {"action": "audio.test", "enabled": True},
        {"action": "audio.target.set", "value": "device:"},
        {"action": "operation.wait", "operation_id": "op-1", "seconds": 1},
        {"action": "query.capabilities"},
        {"action": "query.audio.processes"},
        {"action": "settings.apply", "changes": {}},
        {"action": "secrets.delete", "name": "invented-secret"},
        {"action": "overlay.set"},
    ],
)
async def test_cli_only_and_malformed_requests_are_rejected_without_side_effects(catalog, request_):
    control = ExecutionControl(catalog)
    tools, actions, events = tools_for(control)
    try:
        result = await execute(
            tools,
            request_["action"],
            **{key: value for key, value in request_.items() if key != "action"},
        )
        assert result["status"] == "rejected"
        assert tools.confirmation is None
        assert not control.submissions and not actions and not events
    finally:
        await tools.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action, arguments",
    [
        ("overlay.set", {"enabled": False}),
        ("audio.retry", {}),
        ("models.retry", {"backend": "gpu"}),
    ],
)
async def test_direct_actions_run_without_a_card_and_report_the_terminal_result(
    catalog, action, arguments
):
    control = ExecutionControl(catalog)
    tools, _, events = tools_for(control)
    try:
        result = await execute(tools, action, **arguments)
        assert result["status"] == "applied" and result["receipt"]["terminal"] is True
        assert control.submissions == [
            {"command": action, "arguments": arguments, "revision": None}
        ]
        assert tools.confirmation is None and not events
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_managed_access_probe_reports_usability_without_a_card(catalog):
    control = ExecutionControl(catalog)
    control.terminal_status = "failed"
    original_wait = control.wait

    async def wait(identity, timeout=None):
        control.operations[identity].update(managed_access="unusable")
        return await original_wait(identity, timeout)

    control.wait = wait
    tools, _, events = tools_for(control)
    try:
        result = await tools.execute("probe_app", {"target": "managed_access"})
        assert result["status"] == "failed"
        assert result["receipt"]["managed_access"] == "unusable"
        assert control.submissions == [
            {"command": "accounts.managed.refresh", "arguments": {}, "revision": None}
        ]
        assert tools.confirmation is None and not events
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_key_deletion_card_shows_before_after_and_applies_only_on_approval(catalog):
    control = ExecutionControl(catalog)
    tools, _, events = tools_for(control)
    try:
        offered = await execute(tools, "secrets.delete", name="deepgram_api_key")
        assert offered["status"] == "needs_confirmation"
        display = tools.confirmation_display
        assert display["changes"] == {"secrets.deepgram_api_key": False}
        assert display["previous"] == {"secrets.deepgram_api_key": True}
        assert "provider-side key is not revoked" in display["effects"][0]
        assert "previous" not in offered["confirmation"]
        assert (await tools.confirm("stale", True))["status"] == "rejected"
        declined = await tools.confirm(display["id"], False)
        assert declined["status"] == "declined" and not control.submissions
        assert events[-1]["result"]["status"] == "declined"

        offered = await execute(tools, "secrets.delete", name="deepgram_api_key")
        applied = await tools.confirm(offered["confirmation"]["id"], True)
        assert applied["status"] == "applied" and applied["receipt"]["terminal"] is True
        assert applied["changes"] == {"secrets.deepgram_api_key": False}
        assert control.submissions == [
            {
                "command": "secrets.delete",
                "arguments": {"name": "deepgram_api_key"},
                "revision": control.revision,
            }
        ]
        assert events[-1] == {
            "kind": "confirmation",
            "card_id": offered["confirmation"]["id"],
            "result": applied,
        }
        assert tools.confirmation is None
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_sign_out_and_position_cards_use_observed_state_and_reject_stale_revisions(catalog):
    control = ExecutionControl(catalog)
    control.accounts = {"chatgpt": {"signed_in": True}}
    tools, _, _ = tools_for(control)
    try:
        offered = await execute(tools, "auth.logout", provider="chatgpt")
        assert tools.confirmation_display["previous"] == {"accounts.chatgpt": True}
        control.revision += 1
        rejected = await tools.confirm(offered["confirmation"]["id"], True)
        assert rejected["status"] == "rejected" and rejected["code"] == "revision_conflict"
        assert not control.submissions

        await execute(tools, "overlay.position.reset")
        display = tools.confirmation_display
        assert display["changes"] == {"overlay.desktop_position": "default"}
        assert display["previous"] == {}
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_absent_key_or_signed_out_account_is_not_offered(catalog):
    control = ExecutionControl(catalog)
    control.presence["deepgram_api_key"] = False
    control.accounts = {"chatgpt": {"signed_in": False}}
    tools, _, _ = tools_for(control)
    try:
        for action, arguments in (
            ("secrets.delete", {"name": "deepgram_api_key"}),
            ("auth.logout", {"provider": "chatgpt"}),
        ):
            result = await execute(tools, action, **arguments)
            assert result == {"status": "rejected", "code": "already_applied"}
            assert tools.confirmation is None
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_private_settings_are_outside_model_authority(catalog):
    control = ExecutionControl(catalog)
    control.settings["intent"]["prompts"]["system_prompt_override"] = "PRIVATE-PROMPT-SENTINEL"
    control.settings["intent"]["stt"]["custom"][
        "endpoint"
    ] = "https://PRIVATE-ENDPOINT.invalid/token"
    owner = AgentChatOwner(control, None, localize=lambda key: key)
    try:
        for field in sorted(private_fields()):
            for action in ("settings.edit", "settings.inspect"):
                assert (await execute(owner._tools, action, field=field))["status"] == "rejected"
            rejected = await owner._tools.execute(
                "manage_proposal", {"action": "settings", "changes": {field: "PRIVATE-EDIT"}}
            )
            assert rejected["status"] == "rejected"
        context = await collect_agent_context(
            control,
            activity=AgentActivity(),
            localize=lambda key: "LOCALIZED:" + key,
        )
        assert context["settings_catalog"]["chatbox.activation_notice.enabled"][
            "description"
        ].startswith("LOCALIZED:")
        view = model_context(context, localize=lambda key: "LOCALIZED:" + key)
        private = {
            field: presence
            for module in view["modules"]
            for field, presence in view[module].get("private", {}).items()
        }
        assert set(private) == private_fields()
        assert set(private.values()) <= {"saved", "empty", "unknown"}
        assert all(
            not private_fields() & view[module].get("settings", {}).keys()
            for module in view["modules"]
        )
        current = project_settings(await control.query("settings.current", {}))
        surfaces = json.dumps(
            [
                context,
                view,
                current,
                owner.report_archive,
                safe_receipt(
                    {
                        "status": "failed",
                        "message": "PRIVATE-RECEIPT",
                        "error": {"message": "PRIVATE-ERROR"},
                    }
                ),
            ]
        )
        assert "PRIVATE-" not in surfaces
        assert all(
            item["model_editable"] is False and item["model_readable"] is False
            for item in context["local_settings"].values()
        )
        assert not control.submissions
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_models_projection_preserves_selectable_ids_and_effective_state_without_paths(
    catalog,
):
    control = ExecutionControl(catalog)
    result = await project_query(
        "models.status",
        {
            "assets": [
                {
                    "model_id": "gemma-synthetic",
                    "status": "installed",
                    "directory": "PRIVATE-PATH",
                    "required_files": ["weights.gguf"],
                }
            ],
            "gemma": {
                "state": "ready",
                "model_id": "gemma-synthetic",
                "failure_code": None,
                "raw_output": "PRIVATE-CHILD",
            },
        },
        control,
        lambda key: key,
    )
    assert result["assets"][0]["model_id"] == "gemma-synthetic"
    assert result["gemma"]["state"] == "ready"
    assert "PRIVATE-" not in json.dumps(result)


@pytest.mark.asyncio
async def test_production_tools_apply_notice_and_speaker_with_terminal_effective_state(
    tmp_path, monkeypatch
):
    from puripuly_heart.app.services import application_audio_devices
    from puripuly_heart.composition import application_runtime
    from puripuly_heart.composition.headless_application import compose_headless_application
    from puripuly_heart.config.settings_vnext.schema import SecretsIntent

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "synthetic-isolated-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(
        application_audio_devices,
        "enumerate_audio_devices",
        lambda _settings: {
            "host_apis": [],
            "microphones": [],
            "loopback_outputs": [],
            "errors": {},
        },
    )
    path = tmp_path / "settings.json"
    initial = AppSettingsVNext()
    initial = replace(
        initial,
        intent=replace(
            initial.intent,
            secrets=SecretsIntent(
                backend="encrypted_file", encrypted_file_path=str(tmp_path / "secrets.json")
            ),
            osc=replace(initial.intent.osc, connection_mode="off"),
            telemetry=replace(initial.intent.telemetry, enabled=False),
        ),
    )
    path.write_text(json.dumps(to_dict(initial)), encoding="utf-8")
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("synthetic-agent-control-production")
        owner = app.agent_chat()
        context = await collect_agent_context(control, activity=AgentActivity())
        owner._tools.observe_context(context)
        offered = await owner._tools.execute(
            "manage_proposal",
            {
                "action": "settings",
                "changes": {
                    "chatbox.activation_notice.enabled": False,
                    "stt.soniox.enable_speaker_diarization": False,
                },
            },
        )
        assert offered["status"] == "needs_confirmation"
        completed = await owner.confirm(offered["confirmation"]["id"], True)
        assert completed["status"] == "applied" and completed["receipt"]["terminal"] is True
        current = (await control.query("settings.current", {}))["settings"]["intent"]
        assert current["osc"]["activation_notice_enabled"] is False
        assert current["stt"]["soniox"]["enable_speaker_diarization"] is False
        provider = await control.query("providers.status", {})
        assert not provider["channels"]["self"]["effective_active"]
    finally:
        await app.stop()
    saved = json.loads(path.read_text(encoding="utf-8"))["intent"]
    assert saved["osc"]["activation_notice_enabled"] is False
    assert saved["stt"]["soniox"]["enable_speaker_diarization"] is False


@pytest.mark.asyncio
async def test_production_key_deletion_card_removes_the_stored_key_across_restart(
    tmp_path, monkeypatch
):
    from puripuly_heart.app.services import application_audio_devices
    from puripuly_heart.composition import application_runtime
    from puripuly_heart.composition.headless_application import compose_headless_application
    from puripuly_heart.config.settings_vnext.schema import SecretsIntent

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "synthetic-isolated-passphrase")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(
        application_audio_devices,
        "enumerate_audio_devices",
        lambda _settings: {
            "host_apis": [],
            "microphones": [],
            "loopback_outputs": [],
            "errors": {},
        },
    )
    path = tmp_path / "settings.json"
    initial = AppSettingsVNext()
    initial = replace(
        initial,
        intent=replace(
            initial.intent,
            secrets=SecretsIntent(
                backend="encrypted_file", encrypted_file_path=str(tmp_path / "secrets.json")
            ),
            osc=replace(initial.intent.osc, connection_mode="off"),
            telemetry=replace(initial.intent.telemetry, enabled=False),
        ),
    )
    path.write_text(json.dumps(to_dict(initial)), encoding="utf-8")

    async def presence():
        app = compose_headless_application(path)
        await app.start()
        try:
            app.control().bind_instance("synthetic-restarted-instance")
            return (await app.control().query("secrets.presence", {}))["presence"][
                "deepgram_api_key"
            ]
        finally:
            await app.stop()

    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("synthetic-key-deletion")
        stored = await control.submit(
            "secrets.set",
            {"name": "deepgram_api_key", "value": "synthetic-deepgram-value"},
            request_id="synthetic-store",
        )
        if not stored.get("terminal"):
            stored = await control.wait(stored["operation_id"])
        assert stored["status"] == "applied"
        owner = app.agent_chat()
        offered = await execute(owner._tools, "secrets.delete", name="deepgram_api_key")
        assert offered["status"] == "needs_confirmation"
        assert owner.snapshot.confirmation_previous == {"secrets.deepgram_api_key": True}
        assert (await control.query("secrets.presence", {}))["presence"]["deepgram_api_key"] is True
        completed = await owner.confirm(offered["confirmation"]["id"], True)
        assert completed["status"] == "applied" and completed["receipt"]["terminal"] is True
        assert (await control.query("secrets.presence", {}))["presence"][
            "deepgram_api_key"
        ] is False
        assert "synthetic-deepgram-value" not in json.dumps([completed, owner.report_archive])
    finally:
        await app.stop()
    assert await presence() is False


@pytest.mark.asyncio
async def test_private_presence_is_actual_boolean_metadata():
    from puripuly_heart.app.services.application_control import ApplicationControlOwner

    canonical = AppSettingsVNext()
    query_owner = SimpleNamespace(
        instance_id="presence-instance",
        _current_revision=lambda: 1,
        settings=SimpleNamespace(committed_settings=canonical),
    )
    empty = await ApplicationControlOwner.query(query_owner, "settings.current", {})
    assert empty["private_presence"]["prompts.value"] is False
    assert empty["private_presence"]["stt.custom.endpoint"] is False
    canonical = replace(
        canonical,
        intent=replace(
            canonical.intent,
            prompts=replace(canonical.intent.prompts, system_prompt_override="PRIVATE-PROMPT"),
            stt=replace(
                canonical.intent.stt,
                custom=replace(
                    canonical.intent.stt.custom, endpoint="https://PRIVATE-ENDPOINT.invalid"
                ),
            ),
        ),
    )
    query_owner.settings.committed_settings = canonical
    present = await ApplicationControlOwner.query(query_owner, "settings.current", {})
    assert present["private_presence"]["prompts.value"] is True
    assert present["private_presence"]["stt.custom.endpoint"] is True
    projected = project_settings(present)
    assert projected["private_presence"] == present["private_presence"]
    assert "PRIVATE-" not in json.dumps(projected)


def test_request_schema_rejects_private_targets(catalog):
    schema = {"anyOf": [action.schema(catalog) for action in app_actions()]}
    for request in (
        {"action": "audio.target.set", "value": r"process:exe:C:\PRIVATE-FOLDER\app.exe"},
        {"action": "secrets.delete", "name": "invented-secret"},
        {"action": "settings.edit", "field": "prompts.value"},
    ):
        with pytest.raises(ValueError):
            validate(request, schema)
