from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

from puripuly_heart.app.ports.agent_chat import AgentChatSnapshot, AgentMessage, AgentTurnResult
from puripuly_heart.app.ports.provider_verifier import ProviderCredentialVerificationOutcome
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.app.services.agent_chat.provider_setup import PROVIDER_SETUPS
from puripuly_heart.app.services.http_extension_registry import HttpExtensionRegistryService
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.ui.app import TranslatorApp
from puripuly_heart.ui.components.settings.api_key_field import ApiKeyField
from puripuly_heart.ui.views import agent_chat
from puripuly_heart.ui.views.agent_chat import AgentChatHooks, AgentChatView
from puripuly_heart.ui.views.settings import SettingsView
from tests.app.test_agent_chat import Records, Responses, answer

CONNECTED_PROVIDERS = frozenset(
    {
        "deepgram",
        "gemini_transcribe",
        "elevenlabs_scribe",
        "soniox",
        "google",
        "openrouter",
        "deepseek",
        "openai",
        "alibaba_beijing",
        "alibaba_singapore",
    }
)


class PresentationOwner:
    def __init__(self) -> None:
        self.snapshot = AgentChatSnapshot(
            enabled=True,
            busy=True,
            phase="working",
            messages=(
                AgentMessage("user", "old synthetic request", archived=True),
                AgentMessage("notice", "synthetic app notice"),
            ),
            tool_results=(),
        )
        self.calls = []
        self.action_started = asyncio.Event()
        self.release_action = asyncio.Event()

    def bind_verification_providers(self, providers) -> None:
        self.calls.append(("providers", providers))

    def key_verification_completed(self, *args) -> None:
        self.calls.append(("completion", args))

    def bind_navigation(self, callback) -> None:
        self.calls.append(("navigation", callback))

    def bind_setup_actions(self, callback) -> None:
        self.calls.append(("setup_actions", callback))

    def bind_secure_entry(self, controller) -> None:
        self.calls.append(("secure_entry", controller))

    async def confirm(self, *_args) -> dict:
        self.action_started.set()
        await self.release_action.wait()
        return {"status": "declined"}

    async def send(self, text) -> AgentTurnResult:
        self.calls.append(("send", text))
        return AgentTurnResult("completed", "synthetic response")

    def cancel(self) -> None:
        self.calls.append(("cancel",))
        self.snapshot = replace(self.snapshot, busy=False, phase="idle")

    def new_conversation(self) -> None:
        self.calls.append(("new",))


@pytest.fixture
def secure_settings(tmp_path, settings_events) -> SettingsView:
    view = SettingsView(
        http_extension_registry=HttpExtensionRegistryService(
            HttpExtensionRegistry(tmp_path / "synthetic-extensions")
        )
    )
    settings_events.load(view, AppSettingsVNext(), config_path=tmp_path / "settings.json")
    return view


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_real_secure_field_completion_keeps_settings_in_place_and_unbinds(
    secure_settings: SettingsView, success: bool
) -> None:
    view = secure_settings

    async def verify(_provider, _key):
        return ProviderCredentialVerificationOutcome(
            status="verified" if success else "failed", provider=_provider, error_text="ok"
        )

    view.on_verify_api_key = verify
    owner = PresentationOwner()
    app = TranslatorApp.__new__(TranslatorApp)
    app.view_settings = view
    app._agent_navigation_owner = owner
    app._shutting_down = False
    opened = []
    app._ui_application = SimpleNamespace(
        persist_provider_secret_change=lambda *_: True,
        verify_api_key=verify,
        persist_api_key_verification=lambda *_: None,
    )
    app._open_settings_tab = lambda: opened.append("settings")
    view.on_view_logs = lambda: opened.append("agent")
    before = view._provider_snapshot

    app._bind_agent_navigation()
    assert owner.calls[0] == ("providers", CONNECTED_PROVIDERS)
    assert owner.calls[1][0] == "navigation"
    navigation = owner.calls[1][1]
    result = navigation("api", "openrouter")
    assert result["opened"] is True
    assert result["completion_observable"] is True
    assert result["field_visible"] is True
    assert result["verification_available"] is True
    assert view._provider_snapshot == before

    view._openrouter_key.value = "synthetic-private-value"
    view._openrouter_key._set_status("success")
    assert not any(call[0] == "completion" for call in owner.calls)
    view._openrouter_key._handle_blur(None)
    await view._openrouter_key._run_verification()

    assert owner.calls[-1] == (
        "completion",
        ("openrouter", success, None if success else "verification_failed"),
    )
    assert opened == ["settings"]
    assert view._agent_navigation_banner.visible
    assert view._provider_snapshot.translation == before.translation
    assert "synthetic-private-value" not in repr(owner.calls)
    app._unbind_agent_navigation()
    assert owner.calls[-4:] == [
        ("providers", frozenset()),
        ("navigation", None),
        ("setup_actions", None),
        ("secure_entry", None),
    ]
    view._deepgram_key.value = "synthetic-unbound"
    view._deepgram_key._handle_blur(None)
    await view._deepgram_key._run_verification()
    assert owner.calls[-1] == ("secure_entry", None)


@pytest.mark.asyncio
async def test_stale_verification_cannot_update_settings_or_emit_completion(
    secure_settings: SettingsView,
) -> None:
    view = secure_settings
    completed = []
    before = view._provider_snapshot

    async def verify(_provider, _key):
        view._openrouter_key.value = "replacement"
        return ProviderCredentialVerificationOutcome(status="verified", provider=_provider)

    view.on_verify_api_key = verify
    view.bind_verification_completion(lambda *args: completed.append(args))
    view._openrouter_key.value = "stale"
    view._openrouter_key._handle_blur(None)
    await view._openrouter_key._run_verification()

    assert completed == []
    assert view._provider_snapshot == before
    assert view._openrouter_key.controller.last_verified_hash == ""


def test_disconnected_completion_and_unsupported_providers_are_truthful(
    secure_settings: SettingsView,
) -> None:
    view = secure_settings
    completed = []
    assert view.bind_verification_completion(lambda *args: completed.append(args)) == frozenset()
    opened = []
    result = view.open_agent_section("api", "openrouter", show_settings=lambda: opened.append(1))
    assert result["completion_observable"] is False
    assert result["verification_available"] is False
    for provider in ("local_llm", "custom", "chatgpt", "managed", "http_extension:synthetic"):
        result = view.open_agent_section("api", provider, show_settings=lambda: opened.append(1))
        assert result["opened"] is False
        assert result["completion_observable"] is False
    assert opened == [1]
    assert completed == []


@pytest.mark.asyncio
async def test_actual_setup_tool_and_navigation_expose_fields_without_provider_selection(
    secure_settings: SettingsView,
) -> None:
    view = secure_settings
    control = Records()
    owner = AgentChatOwner(control, None, localize=str)
    app = TranslatorApp.__new__(TranslatorApp)
    app.view_settings = view
    app._agent_navigation_owner = owner
    app._shutting_down = False
    opened = []
    app._current_tab = 1
    app.page = SimpleNamespace(pop_dialog=lambda: None)
    app.bottom_nav = SimpleNamespace(_selected=1, _update_visuals=lambda: None)
    app.content_area = SimpleNamespace(
        content=None, padding=None, update=lambda: opened.append("settings")
    )
    view.on_view_logs = lambda: opened.append("agent")
    mutations = []
    view.on_providers_changed = lambda: mutations.append("provider")
    view.on_request_openrouter_pkce = lambda *_: mutations.append("authorization")

    async def verify(_provider, _key):
        raise AssertionError("Navigation must not verify credentials.")

    view.on_verify_api_key = verify
    app._ui_application = SimpleNamespace(
        persist_provider_secret_change=lambda *_: True,
        verify_api_key=verify,
        persist_api_key_verification=lambda *_: None,
    )
    fields = {
        field._provider: field
        for field in view._api_keys_column.controls
        if isinstance(field, ApiKeyField)
    }
    before_visibility = {provider: field.visible for provider, field in fields.items()}
    before_pkce_visibility = view._openrouter_pkce_button_row.visible
    assert view._api_keys_card.visible is False
    before = (
        view._provider_snapshot,
        view._provider_draft,
        dict(view._provider_edits),
        view.has_provider_changes,
    )
    app._bind_agent_navigation()
    try:
        assert set(fields) == set(PROVIDER_SETUPS)
        for provider, setup in PROVIDER_SETUPS.items():
            metadata = await owner._tools.execute("get_provider_setup", {"provider": provider})
            assert metadata["secure_field"]["secret_key"] == setup.secret_key.value
            assert metadata["link_rendered"] is False
            assert metadata["link_opened"] is False
            result = await owner._tools.execute(
                "open_settings", {"section": "api", "provider": provider}
            )
            assert result["status"] == "opened"
            assert result["field_visible"] is True
            assert result["verification_available"] is True
            assert result["completion_observable"] is True
            assert view._agent_control_visible(view, fields[provider]._text_field)
            assert fields[provider]._text_field.password is True
            view._update_api_visibility()
            assert view._agent_control_visible(view, fields[provider]._text_field)
            assert view._openrouter_pkce_button_row.visible == before_pkce_visibility
            observation = await owner._tools.execute("get_provider_setup", {"provider": provider})
            assert observation["last_navigation"] == {
                "status": "opened",
                "opened": True,
                "field_visible": True,
                "verification_available": True,
                "completion_observable": True,
                "source": "last_navigation",
            }
            await owner._tools.execute("open_settings", {"section": "general"})
            assert {name: field.visible for name, field in fields.items()} == before_visibility
            assert (
                view._provider_snapshot,
                view._provider_draft,
                view._provider_edits,
                view.has_provider_changes,
            ) == before
        await owner._tools.execute("open_settings", {"section": "api", "provider": "openrouter"})
        view._return_to_agent(None)
        assert {provider: field.visible for provider, field in fields.items()} == before_visibility
        assert opened[-1] == "agent"
        assert mutations == []
        assert control.queries == []
        assert owner.snapshot.recent_changes == ()
        assert app._current_tab == app._tab_index("settings")
        assert app.content_area.content is view
        assert not getattr(app, "_settings_mutation_queue", ())
    finally:
        app._unbind_agent_navigation()
        await owner.close()


def agent_view(owner) -> tuple[AgentChatView, list]:
    tasks = []

    def run_task(factory):
        assert inspect.iscoroutinefunction(factory)
        task = asyncio.get_running_loop().create_task(factory())
        tasks.append(task)
        return task

    return AgentChatView(owner, AgentChatHooks(run_task=run_task)), tasks


@pytest.mark.asyncio
async def test_message_and_new_conversation_remain_available_during_approval() -> None:
    owner = PresentationOwner()
    card = {"id": "synthetic-card", "changes": {"stt.provider": "deepgram"}}
    owner.snapshot = replace(owner.snapshot, confirmation=card)
    view, tasks = agent_view(owner)
    view._confirm("synthetic-card", True)
    await asyncio.wait_for(owner.action_started.wait(), 2)

    assert view._action_pending
    assert view._input.disabled is False
    assert view._new_tile.disabled is False
    view._input.value = "newest request"
    await view._submit(None)
    view._new_conversation()

    assert owner.calls == [("send", "newest request"), ("new",)]
    archived, current = view._messages.controls[1], view._messages.controls[2]
    assert archived.opacity < current.opacity
    assert view._input.disabled is False
    owner.release_action.set()
    await asyncio.gather(*tasks)
    assert view._action_pending is False


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["complete", "cancel", "interrupt"])
async def test_submitted_draft_does_not_erase_newer_input(finish: str) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class DelayedResponses(Responses):
        async def prepare(self):
            entered.set()
            await release.wait()

    transport = DelayedResponses(answer())
    owner = AgentChatOwner(Records(), transport, localize=str)
    await owner.start()
    view, _ = agent_view(owner)
    view.did_mount()
    first = None
    second = None
    try:
        view._input.value = "first request"
        first = asyncio.create_task(view._submit(None))
        await asyncio.wait_for(entered.wait(), 2)
        cleared_on_submission = view._input.value == ""
        if cleared_on_submission:
            await view._submit(None)
        type_into(view, "newer draft")
        assert not view._input.disabled
        if finish == "cancel":
            owner.cancel()
        elif finish == "interrupt":
            second = asyncio.create_task(view._submit(None))
            await asyncio.wait_for(first, 2)
            type_into(view, "third draft")
        release.set()
        await asyncio.wait_for(first, 2)
        if second is not None:
            await asyncio.wait_for(second, 2)
        assert view._input.value == ("third draft" if second is not None else "newer draft")
        assert cleared_on_submission
        assert [message.text for message in owner.snapshot.messages if message.role == "user"] == (
            ["first request", "newer draft"] if second is not None else ["first request"]
        )
        assert owner.snapshot.busy is False
    finally:
        release.set()
        owner.cancel()
        await asyncio.gather(*(task for task in (first, second) if task), return_exceptions=True)
        view.will_unmount()
        await owner.close()


@pytest.mark.asyncio
async def test_first_send_and_new_conversation_preserve_rejected_input() -> None:
    transport = Responses(answer(), answer())
    owner = AgentChatOwner(Records(), transport, localize=str)
    await owner.start()
    view, _ = agent_view(owner)
    view.did_mount()
    try:
        assert not view._input.disabled
        view._input.value = "sk-syntheticSecret123456789"
        await view._submit(None)
        assert view._input.value == "sk-syntheticSecret123456789"
        assert "agent_chat.key_blocked" in chat_texts(view)
        assert "sk-syntheticSecret123456789" not in chat_texts(view)
        assert not transport.classifications
        assert not transport.requests
        view._input.value = "accepted request"
        await view._submit(None)
        assert view._input.value == ""
        assert len(transport.requests) == 1
        view._new_conversation()
        assert not view._input.disabled
        view._input.value = "new conversation request"
        await view._submit(None)
        assert view._input.value == ""
        assert len(transport.classifications) == len(transport.requests) == 2
    finally:
        view.will_unmount()
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["rejected", "unavailable"])
@pytest.mark.parametrize("newer", [None, "newer draft", ""])
async def test_delayed_rejection_restores_only_untouched_draft(status, newer) -> None:
    owner = PresentationOwner()
    entered, release = asyncio.Event(), asyncio.Event()

    async def send(text):
        entered.set()
        await release.wait()
        return AgentTurnResult(status, "recoverable rejection")

    owner.send = send
    view, _ = agent_view(owner)
    view._input.value = "submitted draft"
    pending = asyncio.create_task(view._submit(None))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if newer is not None:
            type_into(view, newer)
        release.set()
        await pending
        assert view._input.value == ("submitted draft" if newer is None else newer)
    finally:
        release.set()
        await pending


def chat_texts(view: AgentChatView) -> list[str]:
    return [text.value for item in view._messages.controls for text in agent_chat._texts(item)]


def type_into(view: AgentChatView, value: str) -> None:
    view._input.value = value
    view._input.on_change(SimpleNamespace(control=view._input))
