from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import socket
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from puripuly_heart.app.wiring import wiring_runtime_pipeline
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.application_runtime import compose_application_runtime
from puripuly_heart.composition.headless_application import HeadlessApplicationPresentation
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core import network_clients
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks
from puripuly_heart.core.vrchat_scene_service import VrchatSceneService
from tests.app.test_index_selection import index_settings


class ControlledPresentation(HeadlessApplicationPresentation):
    def __init__(self):
        super().__init__()
        self.messages = []

    def show_message(self, key, **kwargs):
        self.messages.append((key, kwargs))


class ControlledHttp:
    def __init__(self):
        self.index_status = 200
        self.index_entered = asyncio.Event()
        self.index_release = asyncio.Event()
        self.requests = []
        self.remaining = 1.0
        self.metadata_status = 200

    async def handle(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.host, request.url.path, request.headers, body))
        if request.url.host == "index-translate.bilibili.com":
            self.index_entered.set()
            await self.index_release.wait()
            if self.index_status != 200:
                return httpx.Response(self.index_status, json={"error": "synthetic outage"})
        elif request.url.host == "openrouter.ai" and request.url.path.endswith("/key"):
            if self.metadata_status != 200:
                return httpx.Response(
                    self.metadata_status, json={"error": "synthetic metadata failure"}
                )
            return httpx.Response(
                200,
                json={
                    "data": {
                        "limit": 1,
                        "limit_remaining": self.remaining,
                        "usage": 1 - self.remaining,
                    }
                },
            )
        elif request.url.host != "openrouter.ai":
            raise AssertionError(f"unexpected external request: {request.url.host}")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Synthetic translation"},
                    }
                ]
            },
        )

    def client(self, **kwargs):
        return httpx.AsyncClient(
            transport=httpx.MockTransport(self.handle),
            headers=kwargs.get("headers"),
            timeout=kwargs.get("timeout", 5),
            base_url=kwargs.get("base_url", ""),
            trust_env=False,
        )

    @property
    def index_requests(self):
        return [
            request for request in self.requests if request[0] == "index-translate.bilibili.com"
        ]


@pytest.fixture
async def composed(tmp_path, monkeypatch):
    for variable in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(variable, str(tmp_path / variable))
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "synthetic-index-test-passphrase")
    for name in tuple(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN")) or name in {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(
        wiring_runtime_pipeline,
        "_default_vrchat_scene_service",
        lambda: VrchatSceneService(
            snapshots=SimpleNamespace(snapshots=lambda: ()),
            log_directory=tmp_path / "empty-scene",
        ),
    )
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(root, "filters", list(root.filters))
    monkeypatch.setattr(root, "level", root.level)
    for handler in root.handlers:
        monkeypatch.setattr(handler, "filters", list(handler.filters))
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    settings = index_settings(AppSettingsVNext())
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            stt=replace(settings.intent.stt, provider="soniox"),
            peer_stt=replace(settings.intent.peer_stt, provider="soniox"),
            osc=replace(
                settings.intent.osc,
                connection_mode="off",
                host="127.0.0.1",
                port=udp.getsockname()[1],
                send_port=udp.getsockname()[1],
                activation_notice_enabled=False,
                chatbox_send=False,
            ),
            telemetry=replace(settings.intent.telemetry, enabled=False),
            integrated_context=replace(settings.intent.integrated_context, enabled=False),
            clipboard=replace(settings.intent.clipboard, auto_translate_enabled=False),
        ),
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    http = ControlledHttp()
    monkeypatch.setattr(network_clients, "external_async_client", http.client)

    def forbidden_external(*_args, **_kwargs):
        raise AssertionError("unexpected external transport in isolated Index control check")

    monkeypatch.setattr(network_clients, "external_client", forbidden_external)
    monkeypatch.setattr(network_clients, "external_websocket_connect", forbidden_external)
    presentations = []
    apps = []
    sinks = []

    def create():
        presentation = ControlledPresentation()
        log_file = tmp_path / f"runtime-{len(apps)}.log"
        sink = RuntimeLoggingSinks(
            logging.StreamHandler(io.StringIO()),
            logging.FileHandler(log_file, encoding="utf-8"),
            log_file,
        )
        application = compose_application_runtime(
            presentation=presentation,
            config_path=path,
            runtime_logging_sinks=sink,
        )
        presentation.application = application
        application.control().bind_instance(f"index-synthetic-{len(apps)}")
        presentations.append(presentation)
        apps.append(application)
        sinks.append(sink)
        return application, presentation

    try:
        yield path, http, create
    finally:
        for app, sink in reversed(list(zip(apps, sinks))):
            try:
                await app.stop()
            finally:
                sink.close()
                sink.stream_handler.close()
        udp.close()


async def operation(app, command, arguments, request_id):
    control = app.control()
    accepted = await control.submit(command, arguments, request_id=request_id)
    return await control.wait(accepted["operation_id"], timeout=10)


async def wait_until(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_normal_controls_share_probe_and_newer_off_fences_waiting_on(composed):
    path, http, create = composed
    app, presentation = create()
    await app.start()
    await asyncio.wait_for(http.index_entered.wait(), 5)
    control = app.control()
    on = await control.submit("translation.set", {"enabled": True}, request_id="waiting-on")
    await asyncio.sleep(0)
    assert (await control.operation(on["operation_id"]))["status"] == "running"
    off = await operation(app, "translation.set", {"enabled": False}, "newer-off")
    assert off["status"] == "applied"
    assert app.state().translation_enabled is False
    assert len(http.index_requests) == 1
    assert (await control.operation(on["operation_id"]))["terminal"] is False
    locale = await operation(
        app, "settings.apply", {"changes": {"locale": "ja"}}, "responsive-locale"
    )
    assert locale["terminal"] is True
    assert app.compatibility_settings().intent.ui.locale == "ja"
    http.index_release.set()
    on_receipt = await control.wait(on["operation_id"], timeout=5)
    assert on_receipt["status"] != "applied"
    assert app.state().translation_enabled is False
    enabled = await operation(app, "translation.set", {"enabled": True}, "reuse-success")
    assert enabled["status"] == "applied"
    assert app.state().translation_enabled is True
    assert len(http.index_requests) == 1
    assert presentation.messages == []
    assert (
        json.loads(path.read_text(encoding="utf-8"))["intent"]["translation"]["model"]
        == "index_translate_35b"
    )
    assert presentation.dashboard.translation is None
    submitted = await operation(
        app, "text.submit", {"text": "synthetic successful text"}, "successful-turn"
    )
    assert submitted["status"] == "applied"
    await wait_until(lambda: presentation.dashboard.translation == "Synthetic translation")
    assert len(http.index_requests) == 2
    assert "synthetic successful text" in http.index_requests[1][3]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_background_failure_persists_managed_and_waiting_trans_requires_normal_auth(composed):
    path, http, create = composed
    http.index_status = 503
    app, presentation = create()
    await app.start()
    await asyncio.wait_for(http.index_entered.wait(), 5)
    assert presentation.messages == []
    assert app.state().translation_enabled is False
    waiting = asyncio.create_task(
        operation(app, "translation.set", {"enabled": True}, "join-fallback")
    )
    await asyncio.sleep(0)
    http.index_release.set()
    receipt = await waiting
    assert receipt["status"] == "action_required"
    assert any(key == "discord_auth.error.action_required" for key, _ in presentation.messages)
    current = app.compatibility_settings().intent.translation
    assert (current.model, current.connection, current.openrouter_selected_source) == (
        "gemma4_26b_31b",
        "managed",
        "managed",
    )
    assert app.state().translation_enabled is False
    assert [key for key, _ in presentation.messages].count(
        "messages.index_api_fallback_managed"
    ) == 1
    assert len(http.index_requests) == 1
    assert not [
        request for request in http.requests if request[0] != "index-translate.bilibili.com"
    ]
    persisted = json.loads(path.read_text(encoding="utf-8"))["intent"]["translation"]
    assert persisted["model"] == "gemma4_26b_31b"
    assert persisted["connection"] == "managed"
    await app.stop()
    fresh, fresh_presentation = create()
    await fresh.start()
    assert fresh.compatibility_settings().intent.translation.connection == "managed"
    assert fresh.compatibility_settings().intent.translation.model == "gemma4_26b_31b"
    assert fresh.state().translation_enabled is False
    assert fresh_presentation.messages == []
    assert len(http.index_requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential_available,metadata_status", [(False, 200), (True, 200), (True, 503)]
)
async def test_settled_real_request_failure_switches_once_without_replaying_failed_turn(
    composed, credential_available, metadata_status
):
    from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
    from puripuly_heart.config.settings_vnext.facade import load_vnext_settings

    path, http, create = composed
    if credential_available:
        seed = load_vnext_settings(path).settings
        store = create_secret_store(seed.intent.secrets, config_path=path)
        store.set("openrouter_managed_api_key", "synthetic-managed-key")
    http.metadata_status = metadata_status
    http.index_release.set()
    app, presentation = create()
    await app.start()
    enabled = await operation(app, "translation.set", {"enabled": True}, "initial-enable")
    assert enabled["status"] == "applied"
    assert len(http.index_requests) == 1
    http.index_status = 503
    admitted = await operation(app, "text.submit", {"text": "synthetic user text"}, "failed-turn")
    assert admitted["status"] == "applied"
    await wait_until(
        lambda: app.compatibility_settings().intent.translation.model == "gemma4_26b_31b"
    )
    await wait_until(
        lambda: any(
            key == "messages.index_api_fallback_managed" for key, _ in presentation.messages
        )
    )
    assert len(http.index_requests) == 3
    assert not [
        request
        for request in http.requests
        if request[0] == "openrouter.ai" and request[1].endswith("/chat/completions")
    ]
    assert presentation.dashboard.translation is None
    assert [key for key, _ in presentation.messages].count(
        "messages.index_api_fallback_managed"
    ) == 1
    assert app.state().translation_enabled is credential_available
    status = await app.control().query("app.status", {})
    for channel in ("self", "peer"):
        assert status["channels"][channel]["desired_active"] is False
        assert status["channels"][channel]["effective_active"] is False
    if not credential_available:
        resumed = await operation(app, "translation.set", {"enabled": True}, "normal-later-auth")
        assert resumed["status"] == "action_required"
        assert any(key == "discord_auth.error.action_required" for key, _ in presentation.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "remaining,metadata_status,connection,credential",
    [
        (0.01, 200, "managed", "synthetic-managed-key"),
        (0.0007, 200, "openrouter", "synthetic-personal-key"),
        (0, 503, "managed", "synthetic-managed-key"),
    ],
)
async def test_committed_fallback_provider_consumes_the_selected_credential(
    composed, remaining, metadata_status, connection, credential
):
    from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
    from puripuly_heart.config.settings_vnext.facade import load_vnext_settings

    path, http, create = composed
    seed = load_vnext_settings(path).settings
    store = create_secret_store(seed.intent.secrets, config_path=path)
    store.set("openrouter_managed_api_key", "synthetic-managed-key")
    store.set("openrouter_api_key", "synthetic-personal-key")
    http.remaining = remaining
    http.metadata_status = metadata_status
    http.index_status = 503
    http.index_release.set()
    app, presentation = create()
    await app.start()
    enabled = await operation(app, "translation.set", {"enabled": True}, "fallback-authenticated")
    assert enabled["status"] == "applied"
    selected = app.compatibility_settings().intent.translation
    assert (selected.model, selected.connection) == ("gemma4_26b_31b", connection)
    assert app.state().translation_enabled is True
    admitted = await operation(app, "text.submit", {"text": "synthetic route check"}, "route-check")
    assert admitted["status"] == "applied"
    await wait_until(lambda: presentation.dashboard.translation == "Synthetic translation")
    metadata = [request for request in http.requests if request[1].endswith("/key")]
    assert metadata
    assert metadata[0][2]["authorization"] == "Bearer synthetic-managed-key"
    translations = [
        request
        for request in http.requests
        if request[0] == "openrouter.ai" and request[1].endswith("/chat/completions")
    ]
    assert translations
    assert all(request[2]["authorization"] == f"Bearer {credential}" for request in translations)
    assert all(request[3]["models"][0] == "google/gemma-4-26b-a4b-it" for request in translations)
    assert len(http.index_requests) == 1
    assert "authorization" not in http.index_requests[0][2]
    assert [key for key, _ in presentation.messages].count(
        f"messages.index_api_fallback_{connection}"
    ) == 1
    persisted = load_vnext_settings(path).settings.intent.translation
    assert (persisted.model, persisted.connection) == (selected.model, selected.connection)
    if metadata_status != 200:
        assert app._index_selection.allowance_observation.state == "unknown"
        assert (
            app._index_selection.allowance_observation.diagnostics.code
            == "index_managed_allowance_unknown"
        )


@pytest.mark.asyncio
async def test_degraded_runtime_commit_blocks_waiting_trans_until_explicit_provider_recovery(
    composed, monkeypatch
):
    from puripuly_heart.app.services.provider.provider_runtime_apply import ProviderRuntimeOwner

    path, http, create = composed
    http.index_status = 503
    http.index_release.set()
    original = ProviderRuntimeOwner.apply

    async def unavailable(self, settings, plan):
        if settings.intent.translation.model == "gemma4_26b_31b":
            raise OSError("synthetic provider application failure")
        return await original(self, settings, plan)

    app, presentation = create()
    with monkeypatch.context() as patch:
        patch.setattr(ProviderRuntimeOwner, "apply", unavailable)
        await app.start()
        receipt = await operation(app, "translation.set", {"enabled": True}, "degraded-fallback")
    assert receipt["status"] != "applied"
    assert app.state().translation_enabled is False
    assert (
        app._index_selection.transition_result.status == "settings_commit_success_runtime_degraded"
    )
    assert (
        json.loads(path.read_text(encoding="utf-8"))["intent"]["translation"]["connection"]
        == "managed"
    )
    assert [key for key, _ in presentation.messages].count(
        "messages.index_api_fallback_managed"
    ) == 1
    await app.apply_providers(force_rebuild_llm=True)
    retried = await operation(
        app, "translation.set", {"enabled": True}, "normal-auth-after-recovery"
    )
    assert retried["status"] == "action_required"
    assert app.state().translation_enabled is False


@pytest.mark.asyncio
async def test_failed_fallback_write_does_not_claim_a_switch(composed, monkeypatch):
    path, http, create = composed
    app, presentation = create()
    await app.start()
    await asyncio.wait_for(http.index_entered.wait(), 5)

    import threading

    writing = threading.Event()
    release = threading.Event()

    def fail_write(*_args):
        writing.set()
        assert release.wait(5)
        raise OSError("synthetic write failure")

    with monkeypatch.context() as patch:
        patch.setattr(type(app.control().settings.persistence), "persist", fail_write)
        waiting = asyncio.create_task(
            operation(app, "translation.set", {"enabled": True}, "failed-write")
        )
        http.index_status = 503
        http.index_release.set()
        await wait_until(writing.is_set)
        repeated = asyncio.create_task(
            operation(app, "translation.set", {"enabled": True}, "join-staged-write")
        )
        await asyncio.sleep(0)
        current = await app.control().query("settings.current", {})
        assert current["settings"]["intent"]["translation"]["model"] == "index_translate_35b"
        assert len(http.index_requests) == 1
        release.set()
        receipt = await waiting
        repeated_receipt = await repeated
    assert receipt["status"] != "applied"
    assert repeated_receipt["status"] != "applied"
    retry = await operation(app, "translation.set", {"enabled": True}, "failed-write-retry")
    assert retry["status"] != "applied"
    assert len(http.index_requests) == 1
    assert app.state().translation_enabled is False
    assert app.compatibility_settings().intent.translation.model == "index_translate_35b"
    assert (
        json.loads(path.read_text(encoding="utf-8"))["intent"]["translation"]["model"]
        == "index_translate_35b"
    )
    assert not [
        key for key, _ in presentation.messages if key.startswith("messages.index_api_fallback_")
    ]


@pytest.mark.asyncio
async def test_later_explicit_auth_completion_activates_the_persisted_managed_destination(
    composed, monkeypatch
):
    from puripuly_heart.app.wiring import wiring_managed_account
    from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
    from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
    from puripuly_heart.core.discord import discord_oauth_loopback
    from puripuly_heart.core.owned_thread import run_owned_thread_call

    path, http, create = composed
    seed = load_vnext_settings(path).settings
    seed = replace(
        seed,
        intent=replace(
            seed.intent,
            translation=replace(
                seed.intent.translation,
                openrouter_broker_base_url="https://broker.index259.invalid",
            ),
        ),
    )
    path.write_text(json.dumps(to_dict(seed)), encoding="utf-8")
    reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    reservation.bind(("127.0.0.1", 0))
    callback_port = reservation.getsockname()[1]
    reservation.close()
    monkeypatch.setattr(discord_oauth_loopback, "DISCORD_OAUTH_LOOPBACK_PORTS", (callback_port,))
    monkeypatch.setattr(
        wiring_managed_account, "get_raw_hardware_fingerprint", lambda: "synthetic-hardware"
    )
    broker_calls = []
    redirect = []
    original_handler = http.handle

    async def broker_handler(request):
        if request.url.host != "broker.index259.invalid":
            return await original_handler(request)
        body = json.loads(request.content)
        broker_calls.append(request.url.path)
        if request.url.path == "/v1/auth/discord/start":
            redirect.append(body["redirect_uri"])
            return httpx.Response(
                200,
                json={
                    "authorization_url": "https://synthetic.invalid/authorize?state=synthetic-state",
                    "redirect_uri": body["redirect_uri"],
                    "oauth_session_expires_at": "2099-01-01T00:00:00Z",
                    "issue_nonce": "synthetic-nonce",
                    "fingerprint_salt": {"version": 1, "salt": "synthetic-fingerprint-salt"},
                    "fingerprint_salt_version": 1,
                },
            )
        if request.url.path == "/v1/providers/openrouter/discord/issue":
            assert "synthetic-code" in json.dumps(body)
            assert "synthetic-state" in json.dumps(body)
            assert body["model"] == "google/gemma-4-26b-a4b-it"
            assert body["delivery_ack_supported"] is True
            return httpx.Response(
                200,
                json={
                    "openrouter_api_key": "synthetic-later-managed-key",
                    "managed_credential_ref": "synthetic-managed-reference",
                    "expires_at": "2099-01-01T00:00:00Z",
                    "openrouter_user_id": "synthetic-managed-user",
                    "delivery_ack_required": True,
                    "delivery_id": "synthetic-delivery",
                    "delivery_ack_token": "synthetic-ack-token",
                    "delivery_ack_expires_at": "2099-01-01T00:00:00Z",
                },
            )
        if request.url.path == "/v1/providers/openrouter/managed-key-delivery/ack":

            def stored_key():
                current = load_vnext_settings(path).settings
                store = create_secret_store(current.intent.secrets, config_path=path)
                return store.get("openrouter_managed_api_key")

            assert await run_owned_thread_call(stored_key) == "synthetic-later-managed-key"
            assert body["delivery_id"] == "synthetic-delivery"
            return httpx.Response(200, json={"ok": True, "status": "acknowledged"})
        raise AssertionError("unexpected synthetic Broker endpoint")

    monkeypatch.setattr(http, "handle", broker_handler)
    http.index_status = 503
    http.index_release.set()
    app, presentation = create()
    await app.start()
    first = await operation(app, "translation.set", {"enabled": True}, "needs-normal-auth")
    assert first["status"] == "action_required"
    assert broker_calls == []
    assert app.state().translation_enabled is False
    accepted = await app.control().submit(
        "auth.login",
        {"provider": "discord", "open_browser": False},
        request_id="explicit-managed-auth",
    )
    await wait_until(lambda: bool(redirect))
    assert redirect[0].startswith(f"http://127.0.0.1:{callback_port}/")
    async with httpx.AsyncClient(trust_env=False, timeout=5) as callback_client:
        callback = await callback_client.get(
            redirect[0], params={"code": "synthetic-code", "state": "synthetic-state"}
        )
    assert callback.status_code == 200
    authenticated = await app.control().wait(accepted["operation_id"], timeout=10)
    assert authenticated["status"] == "applied"
    assert broker_calls == [
        "/v1/auth/discord/start",
        "/v1/providers/openrouter/discord/issue",
        "/v1/providers/openrouter/managed-key-delivery/ack",
    ]
    assert app.state().translation_enabled is False
    resumed = await operation(app, "translation.set", {"enabled": True}, "after-normal-auth")
    assert resumed["status"] == "applied"
    submitted = await operation(
        app, "text.submit", {"text": "synthetic after auth"}, "authenticated-turn"
    )
    assert submitted["status"] == "applied"
    await wait_until(lambda: presentation.dashboard.translation == "Synthetic translation")
    requests = [
        request
        for request in http.requests
        if request[0] == "openrouter.ai" and request[1].endswith("/chat/completions")
    ]
    assert requests and requests[-1][2]["authorization"] == "Bearer synthetic-later-managed-key"
    assert len(http.index_requests) == 1
    persisted = load_vnext_settings(path).settings
    assert persisted.intent.translation.connection == "managed"
    assert (
        persisted.state.managed_connection.active_managed_credential_ref
        == "synthetic-managed-reference"
    )
    assert persisted.state.managed_connection.pending_delivery_ack_source is None


@pytest.mark.asyncio
async def test_timeout_after_atomic_write_admission_retains_commit_but_fences_runtime_and_toast(
    composed, monkeypatch
):
    import threading

    path, http, create = composed
    app, presentation = create()
    app._index_selection.total_deadline_seconds = 2.0
    await app.start()
    await asyncio.wait_for(http.index_entered.wait(), 5)
    writing = threading.Event()
    release = threading.Event()
    persistence = type(app.control().settings.persistence)
    original = persistence.persist

    def admitted_write(self, *args):
        writing.set()
        assert release.wait(8)
        return original(self, *args)

    with monkeypatch.context() as patch:
        patch.setattr(persistence, "persist", admitted_write)
        http.index_status = 503
        http.index_release.set()
        waiting = asyncio.create_task(
            operation(app, "translation.set", {"enabled": True}, "atomic-write-deadline")
        )
        try:
            await wait_until(writing.is_set)
            receipt = await asyncio.wait_for(waiting, 3)
            assert receipt["status"] != "applied"
            assert not release.is_set()
            assert app.state().translation_enabled is False
            assert app._index_selection.failure_diagnostics.code == "index_selection_deadline"
            assert (
                json.loads(path.read_text(encoding="utf-8"))["intent"]["translation"]["model"]
                == "index_translate_35b"
            )
        finally:
            release.set()
        await wait_until(lambda: app._index_selection._work_task.done())
    persisted = json.loads(path.read_text(encoding="utf-8"))["intent"]["translation"]
    assert (persisted["model"], persisted["connection"]) == ("gemma4_26b_31b", "managed")
    assert app.control().results.current.status == "settings_commit_success_runtime_interrupted"
    assert not app._index_selection.selected_runtime_applied()
    assert app.state().translation_enabled is False
    assert not [
        key for key, _ in presentation.messages if key.startswith("messages.index_api_fallback_")
    ]
    await app.apply_providers(force_rebuild_llm=True)
    recovered = await operation(
        app, "translation.set", {"enabled": True}, "explicit-after-interrupted-apply"
    )
    assert recovered["status"] == "action_required"
    assert app.state().translation_enabled is False


@pytest.mark.asyncio
async def test_real_translation_revision_is_checked_after_shared_resource_wait(composed):
    _path, http, create = composed
    http.index_release.set()
    app, _presentation = create()
    await app.start()
    assert (await operation(app, "translation.set", {"enabled": True}, "enable-before-stale"))[
        "status"
    ] == "applied"
    control = app.control()
    expected = (await control.query("settings.current", {}))["revision"]
    await control._resource_lock.acquire()
    try:
        stale = await control.submit(
            "translation.set",
            {"enabled": False},
            request_id="real-stale-off",
            expected_revision=expected,
        )
        await asyncio.sleep(0)
        assert (await control.operation(stale["operation_id"]))["terminal"] is False
        locale = await operation(
            app, "settings.apply", {"changes": {"locale": "ja"}}, "revision-during-wait"
        )
        current = (await control.query("settings.current", {}))["revision"]
        assert locale["terminal"] is True and current > expected
    finally:
        control._resource_lock.release()
    rejected = await control.wait(stale["operation_id"], timeout=5)
    assert rejected["status"] == "rejected"
    assert rejected["error"] == {"code": "revision_conflict", "current_revision": current}
    assert app.state().translation_enabled is True


@pytest.mark.asyncio
async def test_rejected_index_enable_does_not_invalidate_a_pending_gui_enable(composed):
    _path, http, create = composed
    app, _presentation = create()
    await app.start()
    await asyncio.wait_for(http.index_entered.wait(), 5)
    control = app.control()
    expected = (await control.query("settings.current", {}))["revision"]
    stale = await control.submit(
        "translation.set",
        {"enabled": True},
        request_id="stale-pending-on",
        expected_revision=expected,
    )
    await asyncio.sleep(0)
    gui = asyncio.create_task(app.set_translation_enabled(True, allow_authorization=False))
    await asyncio.sleep(0)
    changed = await operation(app, "settings.apply", {"changes": {"locale": "ja"}}, "newer-locale")
    assert changed["terminal"] is True
    current = (await control.query("settings.current", {}))["revision"]
    assert current > expected
    http.index_release.set()
    rejected = await control.wait(stale["operation_id"], timeout=5)
    assert rejected["status"] == "rejected"
    assert rejected["error"] == {"code": "revision_conflict", "current_revision": current}
    assert await asyncio.wait_for(gui, 5) is True
    assert app.state().translation_enabled is True
    assert len(http.index_requests) == 1


@pytest.mark.asyncio
async def test_failed_preflight_with_stale_revision_never_advances_pending_intent(
    composed, monkeypatch
):
    from puripuly_heart.app.ports.ui_application import TranslationEnablePreparation

    _path, http, create = composed
    app, _presentation = create()
    await app.start()
    await asyncio.wait_for(http.index_entered.wait(), 5)
    control = app.control()
    expected = (await control.query("settings.current", {}))["revision"]
    first = asyncio.Event()
    fail = asyncio.Event()
    original = app.prepare_translation_enable

    async def controlled_preparation(intent_revision=None):
        if not first.is_set():
            first.set()
            captured = app._translation_selection_intent
            await fail.wait()
            return TranslationEnablePreparation(
                intent_revision=captured,
                selection_generation=app._index_selection.generation,
                ready=False,
            )
        return await original(intent_revision)

    monkeypatch.setattr(app, "prepare_translation_enable", controlled_preparation)
    failed = await control.submit(
        "translation.set",
        {"enabled": True},
        request_id="failed-before-admission",
        expected_revision=expected,
    )
    await first.wait()
    gui = asyncio.create_task(app.set_translation_enabled(True, allow_authorization=False))
    await asyncio.sleep(0)
    pending_intent = app._translation_selection_intent
    changed = await operation(
        app, "settings.apply", {"changes": {"locale": "ja"}}, "stale-before-failure"
    )
    assert changed["terminal"] is True
    assert (await control.query("settings.current", {}))["revision"] > expected
    fail.set()
    receipt = await control.wait(failed["operation_id"], timeout=5)
    assert receipt["status"] == "degraded"
    assert receipt["reason"] == "translation_preparation_failed"
    assert app._translation_selection_intent == pending_intent
    assert not gui.done()
    assert app.state().translation_enabled is False
    http.index_release.set()
    assert await asyncio.wait_for(gui, 5) is True
    assert app.state().translation_enabled is True
    assert len(http.index_requests) == 1


@pytest.mark.asyncio
async def test_changed_selection_rejoins_current_probe_outside_final_admission(composed):
    _path, http, create = composed
    http.index_release.set()
    app, presentation = create()
    await app.start()
    assert (await operation(app, "translation.set", {"enabled": True}, "first-activation"))[
        "status"
    ] == "applied"
    assert (await operation(app, "translation.set", {"enabled": False}, "before-reselection"))[
        "status"
    ] == "applied"
    control = app.control()
    await control._resource_lock.acquire()
    try:
        away = await control.submit(
            "settings.apply",
            {
                "changes": {
                    "translation.model": "gemma4_26b_31b",
                    "translation.connection": "managed",
                }
            },
            request_id="queued-away",
        )
        back = await control.submit(
            "settings.apply",
            {
                "changes": {
                    "translation.model": "index_translate_35b",
                    "translation.connection": "public_api",
                }
            },
            request_id="queued-back",
        )
        on = await control.submit(
            "translation.set", {"enabled": True}, request_id="prepared-before-reselection"
        )
        await asyncio.sleep(0)
        assert (await control.operation(on["operation_id"]))["terminal"] is False
        http.index_release.clear()
        http.index_status = 503
    finally:
        control._resource_lock.release()
    assert (await control.wait(away["operation_id"], timeout=5))["terminal"] is True
    assert (await control.wait(back["operation_id"], timeout=5))["terminal"] is True
    await wait_until(lambda: len(http.index_requests) == 2)
    assert (await control.operation(on["operation_id"]))["terminal"] is False
    changed = await operation(
        app, "settings.apply", {"changes": {"locale": "ja"}}, "responsive-after-reselection"
    )
    assert changed["terminal"] is True
    http.index_release.set()
    result = await control.wait(on["operation_id"], timeout=5)
    assert result["status"] == "action_required"
    assert app.compatibility_settings().intent.translation.connection == "managed"
    assert app.state().translation_enabled is False
    assert len(http.index_requests) == 2
    assert [key for key, _ in presentation.messages].count(
        "messages.index_api_fallback_managed"
    ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["probe", "resource"])
async def test_freeze_fences_real_index_activation_in_preparation_or_resource_wait(composed, phase):
    _path, http, create = composed
    if phase == "resource":
        http.index_release.set()
    app, _presentation = create()
    await app.start()
    await asyncio.wait_for(http.index_entered.wait(), 5)
    control = app.control()
    if phase == "resource":
        prepared = await app.prepare_translation_enable()
        assert prepared.ready is True
        await control._resource_lock.acquire()
    try:
        on = await control.submit(
            "translation.set", {"enabled": True}, request_id=f"freeze-at-{phase}"
        )
        await asyncio.sleep(0)
        assert (await control.operation(on["operation_id"]))["terminal"] is False
        control.freeze_ingress()
        await asyncio.wait_for(control.drain_operations(), 5)
        assert (await control.operation(on["operation_id"]))["status"] == "interrupted"
        assert app.state().translation_enabled is False
    finally:
        if phase == "resource":
            control._resource_lock.release()
        http.index_release.set()
    await app._index_selection.prepare_enable()
    assert app.state().translation_enabled is False
    assert len(http.index_requests) == 1


@pytest.mark.asyncio
async def test_successful_index_preparation_still_orders_activation_after_manual_output(
    composed, monkeypatch
):
    from uuid import uuid4

    from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission

    _path, http, create = composed
    http.index_release.set()
    app, _presentation = create()
    await app.start()
    prepared = await app.prepare_translation_enable()
    assert prepared.ready is True
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held_output(_text):
        entered.set()
        await release.wait()
        return TextSubmissionAdmission("admitted", uuid4(), ())

    monkeypatch.setattr(app, "submit_text", held_output)
    control = app.control()
    manual = await control.submit(
        "text.submit", {"text": "synthetic hold"}, request_id="manual-hold"
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        on = await control.submit(
            "translation.set", {"enabled": True}, request_id="ordered-index-on"
        )
        await asyncio.sleep(0)
        assert (await control.wait(on["operation_id"], timeout=0.01))["terminal"] is False
        assert app.state().translation_enabled is False
        locale = await operation(
            app, "settings.apply", {"changes": {"locale": "ja"}}, "safe-during-manual"
        )
        assert locale["terminal"] is True
    finally:
        release.set()
    assert (await control.wait(manual["operation_id"], timeout=5))["status"] == "applied"
    assert (await control.wait(on["operation_id"], timeout=5))["status"] == "applied"
    assert app.state().translation_enabled is True
    assert len(http.index_requests) == 1
