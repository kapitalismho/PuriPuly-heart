from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest

from puripuly_heart.app.ports.provider_verifier import ProviderCredentialVerificationOutcome
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.app.services.agent_chat.provider_setup import official_link_destination
from puripuly_heart.ui.app import TranslatorApp
from puripuly_heart.ui.components.agent_secure_entry import AgentSecureEntryController
from tests.app.test_agent_chat import Records, Responses, answer, tool
from tests.app.test_agent_chat_report import ReportResponses, contents, proposal


def setup_call(provider="openrouter", identity="setup"):
    return {
        "status": "completed",
        "output": [
            {
                **tool(identity),
                "name": "open_app",
                "arguments": json.dumps({"view": "provider_setup", "provider": provider}),
            }
        ],
    }


def bind_secure(owner, *, save, verify, persisted):
    adapter = AgentSecureEntryController(
        save=save,
        verify=verify,
        persist_verification=lambda *args: persisted.append(args),
    )
    owner.bind_secure_entry(adapter)
    return adapter


async def settle_notices(owner):
    if owner._notice_task is not None:
        await owner._notice_task


@pytest.mark.asyncio
async def test_fixed_official_action_activation_never_trusts_mutable_model_metadata(monkeypatch):
    transport = Responses(setup_call(), answer("Use the official provider setup."), answer())
    owner = AgentChatOwner(Records(), transport, localize=str)
    app = TranslatorApp.__new__(TranslatorApp)
    app._shutting_down = False
    opened = []
    monkeypatch.setattr(
        "puripuly_heart.ui.app.webbrowser.open", lambda url: opened.append(url) or True
    )
    owner.bind_setup_actions(app._activate_agent_setup_action)
    await owner.start()
    try:
        await owner.send("Help configure an API key")
        action = next(
            action for action in owner.snapshot.setup_actions if action.kind == "key_console"
        )
        assert action.available and not action.displayed and not action.clicked
        assert action.message_index == len(owner.snapshot.messages) - 1
        assert action.secure_request_id == owner.snapshot.secure_entry_requests[0].id
        assert owner.activate_setup_action(action.id)["status"] == "rejected"
        assert not opened
        owner.snapshot.tool_results[0].result["links"][0]["url"] = "https://untrusted.invalid"
        assert owner.mark_setup_action_displayed(action.id)["status"] == "displayed"
        assert owner.activate_setup_action(action.id)["status"] == "opened"
        assert opened == [official_link_destination(action.action_id)]
        clicked = next(item for item in owner.snapshot.setup_actions if item.id == action.id)
        assert clicked.clicked and clicked.displayed
        assert owner.activate_setup_action("https://untrusted.invalid")["status"] == "rejected"
        await owner.send("A different task")
        assert owner.activate_setup_action(action.id)["status"] == "rejected"
        assert len(opened) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_setup_metadata_stays_pending_until_response_commit():
    entered, release = asyncio.Event(), asyncio.Event()

    class Held(Responses):
        async def respond(self, **request):
            if len(self.requests) == 1:
                self.requests.append(request)
                entered.set()
                await release.wait()
                yield {"type": "completed", "response": answer("Provider setup is available.")}
                return
            async for event in super().respond(**request):
                yield event

    owner = AgentChatOwner(Records(), Held(setup_call()), localize=str)
    owner.bind_setup_actions(lambda _: {"status": "opened"})
    await owner.start()
    sending = asyncio.create_task(owner.send("Configure the provider"))
    try:
        await entered.wait()
        request = owner.snapshot.secure_entry_requests[0]
        action = next(item for item in owner.snapshot.setup_actions if item.secure_request_id)
        assert request.masked and request.status == "pending"
        assert not request.available and not request.displayed
        assert action.message_index is None and not action.available
        assert owner.activate_setup_action(action.id)["status"] == "rejected"
        assert owner.mark_secure_entry_displayed(request.id)["status"] == "rejected"
        release.set()
        await sending
        request = owner.snapshot.secure_entry_requests[0]
        assert request.status == "ready" and not request.available
        assert request.message_index is not None and not request.displayed
    finally:
        release.set()
        await owner.close()
        await asyncio.gather(sending, return_exceptions=True)


@pytest.mark.asyncio
async def test_secure_failure_then_success_resumes_with_typed_notices_without_secret_artifacts(
    caplog,
):
    transport = Responses(setup_call(), answer(), answer(), answer())
    owner = AgentChatOwner(Records(), transport, localize=str)
    saved, persisted, verified, events, snapshots = [], [], [], [], []

    async def verify(provider, credential):
        verified.append((provider, credential))
        return ProviderCredentialVerificationOutcome(
            provider=provider,
            status="failed" if len(verified) == 1 else "verified",
            error_text="private diagnostic prose must not become a notice",
        )

    adapter = bind_secure(
        owner,
        save=lambda *args: saved.append(args) or True,
        verify=verify,
        persisted=persisted,
    )
    owner.observe(snapshots.append)
    owner.observe_events(lambda event: events.append(event))
    await owner.start()
    try:
        await owner.send("Configure the provider")
        first = next(item for item in owner.snapshot.secure_entry_requests if item.available)
        assert first.masked and first.settings_alternative
        owner.mark_secure_entry_displayed(first.id)
        assert (await owner.submit_secure_entry(first.id, "synthetic-private-first"))[
            "status"
        ] == "failed"
        await settle_notices(owner)
        assert len(transport.requests) == 2
        retry = next(item for item in owner.snapshot.secure_entry_requests if item.available)
        assert retry.id == first.id and retry.status == "failed"
        assert (await owner.submit_secure_entry(retry.id, "synthetic-private-second"))[
            "status"
        ] == "verified"
        await settle_notices(owner)
        notices = [event for event in events if event["kind"] == "app_notice"]
        assert len(notices) == 1 and "result=success" in notices[0]["text"]
        assert not any("result=failed" in message.text for message in owner.snapshot.messages)
        resumed = [item.get("content", "") for item in transport.requests[-1]["input"]]
        failed = next(i for i, text in enumerate(resumed) if "result=failed" in text)
        succeeded = next(i for i, text in enumerate(resumed) if "result=success" in text)
        assert failed < succeeded
        assert len(saved) == len(verified) == len(persisted) == 2
        assert [item[2] for item in persisted] == [False, True]
        assert not adapter._value and adapter._controller is None
        assert len(transport.classifications) == 1 and len(transport.requests) == 3
        artifacts = repr(
            (
                [asdict(snapshot) for snapshot in snapshots],
                owner.report_archive,
                owner._items,
                transport.requests,
                transport.classifications,
                events,
            )
        )
        assert "synthetic-private-" not in artifacts
        assert "private diagnostic prose" not in artifacts
        assert "synthetic-private-" not in caplog.text
        assert owner.snapshot.recent_changes == () and owner.snapshot.confirmation is None
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["form_cancel", "new_input", "new_conversation"])
async def test_secure_verification_stale_results_never_persist_or_resume(interrupt):
    entered, release = asyncio.Event(), asyncio.Event()
    persisted, events = [], []
    transport = Responses(setup_call(), answer(), setup_call("deepgram", "next-setup"), answer())
    owner = AgentChatOwner(Records(), transport, localize=str)

    async def verify(provider, _credential):
        entered.set()
        await release.wait()
        return ProviderCredentialVerificationOutcome(provider=provider, status="verified")

    adapter = bind_secure(owner, save=lambda *_: True, verify=verify, persisted=persisted)
    owner.observe_events(lambda event: events.append(event))
    await owner.start()
    await owner.send("Configure the provider")
    request = next(item for item in owner.snapshot.secure_entry_requests if item.available)
    submitting = asyncio.create_task(
        owner.submit_secure_entry(request.id, "synthetic-private-held")
    )
    try:
        await entered.wait()
        assert (await owner.submit_secure_entry(request.id, "synthetic-private-duplicate")) == {
            "status": "rejected",
            "code": "verification_in_progress",
        }
        if interrupt == "form_cancel":
            assert owner.cancel_secure_entry(request.id)["status"] == "cancelled"
        elif interrupt == "new_input":
            await owner.send("Configure a different provider instead")
        else:
            owner.new_conversation()
        release.set()
        assert (await submitting)["status"] == "cancelled"
        await settle_notices(owner)
        assert not persisted
        assert not [event for event in events if event["kind"] == "app_notice"]
        assert not adapter._value
        assert (await owner.submit_secure_entry(request.id, "synthetic-private-stale"))[
            "status"
        ] == "rejected"
    finally:
        release.set()
        await owner.close()
        await asyncio.gather(submitting, return_exceptions=True)


@pytest.mark.asyncio
async def test_secure_save_failure_never_verifies_and_reports_once():
    transport = Responses(setup_call(), answer(), answer())
    owner = AgentChatOwner(Records(), transport, localize=str)
    persisted, verified, events = [], [], []
    bind_secure(
        owner,
        save=lambda *_: False,
        verify=lambda *args: verified.append(args),
        persisted=persisted,
    )
    owner.observe_events(lambda event: events.append(event))
    await owner.start()
    try:
        await owner.send("Configure the provider")
        request = next(item for item in owner.snapshot.secure_entry_requests if item.available)
        assert (await owner.submit_secure_entry(request.id, "synthetic-private-value"))[
            "status"
        ] == "failed"
        await settle_notices(owner)
        assert not persisted and not verified
        assert not [event for event in events if event["kind"] == "app_notice"]
        assert len(transport.requests) == 2
        await owner.send("Why did it fail?")
        assert len(transport.requests) == 3
        assert "result=failed" in json.dumps(transport.requests[-1]["input"])
        assert not any("result=failed" in message.text for message in owner.snapshot.messages)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_user_cancelled_secure_entry_reaches_next_message_without_resuming():
    transport = Responses(setup_call(), answer(), answer())
    owner = AgentChatOwner(Records(), transport, localize=str)
    events = []
    bind_secure(owner, save=lambda *_: True, verify=lambda *_: None, persisted=[])
    owner.observe_events(lambda event: events.append(event))
    await owner.start()
    try:
        await owner.send("Configure the provider")
        request = next(item for item in owner.snapshot.secure_entry_requests if item.available)
        messages = owner.snapshot.messages
        assert owner.cancel_secure_entry(request.id)["status"] == "cancelled"
        await settle_notices(owner)
        assert len(transport.requests) == 2 and owner.snapshot.messages == messages
        assert not [event for event in events if event["kind"] == "app_notice"]
        await owner.send("Let's do it later")
        expected = f"[app notice] secure_entry provider={request.provider} result=cancelled"
        assert expected in [item.get("content") for item in transport.requests[-1]["input"]]
        assert not any(expected in row.text for row in owner.snapshot.messages)
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_account_actions_only_open_existing_user_auth_or_fixed_public_usage(monkeypatch):
    control = Records()
    control.values["accounts.status"] = {
        "chatgpt": {"signed_in": False, "in_progress": False, "email": "private@example.invalid"},
        "managed": {"provider": None, "credential_present": False, "selected": False},
    }
    transport = Responses(
        {
            "status": "completed",
            "output": [
                {**tool(), "name": "open_app", "arguments": json.dumps({"view": "accounts"})}
            ],
        },
        answer("Account setup actions are available."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    app = TranslatorApp.__new__(TranslatorApp)
    app._shutting_down = False
    opened = []
    app.show_chatgpt_auth_dialog = lambda **options: opened.append(("chatgpt", options))
    app.show_qq_managed_auth_dialog = lambda: opened.append(("qq", {}))
    app.show_discord_managed_auth_dialog = lambda **options: opened.append(("discord", options))
    monkeypatch.setattr(
        "puripuly_heart.ui.app.webbrowser.open", lambda url: opened.append(url) or True
    )
    owner.bind_setup_actions(app._activate_agent_setup_action)
    await owner.start()
    try:
        await owner.send("Show account setup options")
        actions = [
            item for item in owner.snapshot.setup_actions if item.action_id.startswith("account.")
        ]
        assert len(actions) == 4 and not opened
        assert all(not item.displayed and not item.clicked for item in actions)
        for action in actions:
            assert owner.activate_setup_action(action.id)["status"] == "rejected"
            owner.mark_setup_action_displayed(action.id)
            assert owner.activate_setup_action(action.id)["status"] == "opened"
        assert ("chatgpt", {"enable_translation": False}) in opened
        assert ("qq", {}) in opened and ("discord", {"preview": False}) in opened
        usage = next(item for item in actions if item.kind == "public_usage")
        assert official_link_destination(usage.action_id) in opened
        assert "private@example.invalid" not in repr(
            (owner.snapshot, transport.requests, owner.report_archive)
        )
        assert not [name for name, _ in control.queries if name == "auth.status"]
        assert len(transport.requests) == 2
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_matching_settings_alternative_remains_callable_without_inline_renderer():
    transport = Responses(setup_call(), answer(), answer())
    owner = AgentChatOwner(Records(), transport, localize=str)
    opened, events = [], []
    owner.bind_verification_providers(frozenset({"openrouter"}))
    owner.bind_navigation(
        lambda section, provider: opened.append((section, provider))
        or {"opened": True, "field_visible": True, "verification_available": True}
    )
    owner.observe_events(lambda event: events.append(event))
    await owner.start()
    try:
        await owner.send("Configure the provider")
        (request,) = owner.snapshot.secure_entry_requests
        assert request.settings_alternative and not request.available
        result = await owner.open_secure_entry_settings(request.id)
        assert result["status"] == "opened"
        assert opened == [("api", request.provider)]
        owner.key_verification_completed(request.provider, True)
        owner.key_verification_completed(request.provider, True)
        await settle_notices(owner)
        assert len([event for event in events if event["kind"] == "app_notice"]) == 1
        assert len(transport.classifications) == 1 and len(transport.requests) == 3
        assert owner.snapshot.secure_entry_requests[0].status == "verified"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_support_click_bundle_excludes_secure_input_after_verification():
    transport = ReportResponses(
        setup_call(),
        answer(),
        answer(),
        {"status": "completed", "output": [proposal()]},
        answer("The diagnostic evidence can be sent at your request."),
    )
    owner = AgentChatOwner(
        Records(),
        transport,
        localize=str,
        support_log_snapshot=lambda: b"safe synthetic app log",
    )
    persisted = []
    bind_secure(
        owner,
        save=lambda *_: True,
        verify=lambda provider, _: ProviderCredentialVerificationOutcome(
            provider=provider, status="verified"
        ),
        persisted=persisted,
    )
    await owner.start()
    try:
        await owner.send("Configure the provider")
        request = next(item for item in owner.snapshot.secure_entry_requests if item.available)
        await owner.submit_secure_entry(request.id, "synthetic-private-report-key")
        await settle_notices(owner)
        await owner.send("Please send this diagnostic evidence to support")
        (offer,) = owner.snapshot.report_offers
        assert not transport.uploads and not offer.clicked
        assert (await owner.send_report(offer.id))["status"] == "sent"
        payload = contents(transport.uploads[0]["bundle"])
        assert all(b"synthetic-private-report-key" not in value for value in payload.values())
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["save", "verify"])
@pytest.mark.parametrize("external_cancel", [False, True])
async def test_secure_cancellation_stops_pending_work_without_erasing_completed_save(
    boundary, external_cancel
):
    entered, stopped = asyncio.Event(), asyncio.Event()
    saved, verified, persisted, events = [], [], [], []
    owner = AgentChatOwner(Records(), Responses(setup_call(), answer()), localize=str)

    async def hold():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def save(key, value):
        if boundary == "save":
            await hold()
        saved.append((key, value))
        return True

    async def verify(provider, value):
        verified.append(provider)
        await hold()
        return ProviderCredentialVerificationOutcome(provider=provider, status="verified")

    bind_secure(owner, save=save, verify=verify, persisted=persisted)
    owner.observe_events(lambda event: events.append(event))
    await owner.start()
    await owner.send("Configure the provider")
    request = next(item for item in owner.snapshot.secure_entry_requests if item.available)
    submitting = asyncio.create_task(
        owner.submit_secure_entry(request.id, "synthetic-private-held")
    )
    try:
        await entered.wait()
        if external_cancel:
            submitting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submitting
        else:
            owner.cancel_secure_entry(request.id)
            await asyncio.wait_for(stopped.wait(), 1)
            assert (await submitting)["status"] == "cancelled"
        assert stopped.is_set()
        assert len(saved) == (boundary == "verify")
        assert verified == (["openrouter"] if boundary == "verify" else [])
        assert not persisted
        assert not [event for event in events if event["kind"] == "app_notice"]
    finally:
        submitting.cancel()
        await asyncio.gather(submitting, return_exceptions=True)
        await owner.close()
