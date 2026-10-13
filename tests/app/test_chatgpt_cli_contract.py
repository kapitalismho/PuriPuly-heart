from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from puripuly_heart.app.ports.chatgpt_account import ChatGptConnectResult, ChatGptSignOutResult
from puripuly_heart.app.wiring import create_secret_store, wiring_llm_factory
from puripuly_heart.composition.headless_application import compose_headless_application
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.chatgpt.session import ChatGptSession
from puripuly_heart.providers.llm.chatgpt_plan import ChatGptPlanLLMProvider


def _isolated_settings(path) -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(settings.intent.osc, connection_mode="off"),
        ),
    )
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")


@pytest.mark.asyncio
async def test_luna_defaults_to_chatgpt_and_reports_required_sign_in(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    _isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("chatgpt-status")
        before = (await control.query("auth.status", {}))["chatgpt"]
        assert before["signed_in"] is False
        assert before["in_progress"] is False
        assert before["sign_in_required"] is False

        submitted = await control.submit(
            "settings.apply",
            {"changes": {"translation.model": "gpt_6_luna"}},
            request_id="select-luna",
        )
        assert (await control.wait(submitted["operation_id"], timeout=10))["status"] in {
            "applied",
            "degraded",
        }
        translation = (await control.query("settings.current", {}))["settings"]["intent"][
            "translation"
        ]
        assert translation["connection"] == "chatgpt"
        after = (await control.query("auth.status", {}))["chatgpt"]
        assert after["signed_in"] is False
        assert after["in_progress"] is False
        assert after["sign_in_required"] is True
        assert "email" not in after
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_cli_chatgpt_login_and_logout_report_outcomes(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    _isolated_settings(path)
    app = compose_headless_application(path)
    calls: list[dict[str, object]] = []

    async def connect_chatgpt(*, open_browser, authorization_url_sink):
        calls.append({"open_browser": open_browser})
        authorization_url_sink("https://auth.openai.com/api/accounts/authorize?state=x")
        return ChatGptConnectResult(succeeded=True, first_sign_in=True, email="u@example.com")

    async def sign_out_chatgpt():
        return ChatGptSignOutResult(remote_revoked=False, local_credentials_removed=True)

    monkeypatch.setattr(app, "connect_chatgpt", connect_chatgpt)
    monkeypatch.setattr(app, "sign_out_chatgpt", sign_out_chatgpt)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("chatgpt-cli")

        rejected = await control.submit(
            "auth.login",
            {"provider": "chatgpt", "referral_id": "PASS-1234"},
            request_id="chatgpt-referral",
        )
        assert (await control.wait(rejected["operation_id"], timeout=5))["status"] == "rejected"
        assert calls == []

        login = await control.submit(
            "auth.login", {"provider": "chatgpt"}, request_id="chatgpt-login"
        )
        result = await control.wait(login["operation_id"], timeout=5)
        assert result["status"] == "applied"
        assert result["provider"] == "chatgpt"
        assert result["first_sign_in"] is True
        assert "email" not in result
        assert calls == [{"open_browser": False}]

        logout = await control.submit(
            "auth.logout", {"provider": "chatgpt"}, request_id="chatgpt-logout"
        )
        signed_out = await control.wait(logout["operation_id"], timeout=5)
        assert signed_out["status"] == "applied"
        assert signed_out["scope"] == "local_only"
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_cli_chatgpt_login_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    _isolated_settings(path)
    app = compose_headless_application(path)

    async def connect_chatgpt(*, open_browser, authorization_url_sink):
        return ChatGptConnectResult(succeeded=False, failure_code="access_denied")

    monkeypatch.setattr(app, "connect_chatgpt", connect_chatgpt)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("chatgpt-cli-denied")
        login = await control.submit(
            "auth.login", {"provider": "chatgpt"}, request_id="chatgpt-denied"
        )
        result = await control.wait(login["operation_id"], timeout=5)
        assert result["status"] == "rejected"
        assert result["reason"] == "access_denied"
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_selecting_chatgpt_translation_switches_to_luna_on_chatgpt(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    _isolated_settings(path)
    app = compose_headless_application(path)
    try:
        await app.start()
        control = app.control()
        control.bind_instance("chatgpt-select")
        before = (await control.query("settings.current", {}))["settings"]["intent"]["translation"]
        assert (before["model"], before["connection"]) != ("gpt_6_luna", "chatgpt")

        await app.select_chatgpt_translation()

        translation = (await control.query("settings.current", {}))["settings"]["intent"][
            "translation"
        ]
        assert translation["model"] == "gpt_6_luna"
        assert translation["connection"] == "chatgpt"
        assert translation["connection_history"]["gpt_6_luna"] == "chatgpt"
        assert (await control.query("auth.status", {}))["chatgpt"]["sign_in_required"] is True
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_signed_in_luna_startup_only_prepares_one_standby_socket(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    path = tmp_path / "settings.json"
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation, model="gpt_6_luna", connection="chatgpt"
            ),
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(settings.intent.osc, connection_mode="off"),
            overlay=replace(settings.intent.overlay, target="none"),
            telemetry=replace(settings.intent.telemetry, enabled=False),
        ),
    )
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    store = create_secret_store(settings.intent.secrets, config_path=path)
    store.set("chatgpt_refresh_token", "test-refresh-token")
    store.set("chatgpt_client_id", "test-client")
    sockets = []
    requests = []
    connected = asyncio.Event()

    class Socket:
        close_code = None

        async def send(self, request):
            requests.append(json.loads(request))

        async def recv(self):
            return json.dumps({"type": "response.completed"})

        async def close(self):
            self.close_code = 1000

    async def connect(_url, _headers):
        socket = Socket()
        sockets.append(socket)
        connected.set()
        return socket

    class LocalProvider(ChatGptPlanLLMProvider):
        def __init__(self, *args, **kwargs):
            kwargs["connector"] = connect
            super().__init__(*args, **kwargs)

    async def access_token(_session):
        return "test-token"

    monkeypatch.setattr(wiring_llm_factory, "ChatGptPlanLLMProvider", LocalProvider)
    monkeypatch.setattr(ChatGptSession, "access_token", access_token)
    app = compose_headless_application(path)
    try:
        await app.start()
        await asyncio.wait_for(connected.wait(), timeout=1)
        status = await app.control().query("app.status", {})
        assert status["translation_enabled"] is False
        assert status["channels"]["self"]["effective_active"] is False
        assert status["channels"]["peer"]["effective_active"] is False
        assert len(sockets) == 1
        assert sockets[0].close_code is None
        assert requests == []
    finally:
        await app.stop()
    assert all(socket.close_code == 1000 for socket in sockets)
