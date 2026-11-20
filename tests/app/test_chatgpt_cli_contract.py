from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from puripuly_heart.app.ports.chatgpt_account import ChatGptConnectResult, ChatGptSignOutResult
from puripuly_heart.app.wiring import (
    create_secret_store,
    wiring_llm_factory,
    wiring_runtime_pipeline,
)
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.headless_application import compose_headless_application
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.chatgpt import loopback
from puripuly_heart.core.chatgpt.oauth import (
    CHATGPT_JWKS_URL,
    CHATGPT_PLAN_SCOPE,
    CHATGPT_TOKEN_URL,
)
from puripuly_heart.core.chatgpt.session import ChatGptSession
from puripuly_heart.core.vrchat_scene_service import VrchatSceneService
from puripuly_heart.providers.llm.chatgpt_plan import ChatGptPlanLLMProvider
from tests.app.test_chatgpt_account import _id_token, _jwk


def _isolated_settings(path) -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(settings.intent.osc, connection_mode="off"),
            photo=replace(settings.intent.photo, enabled=False),
            telemetry=replace(settings.intent.telemetry, enabled=False),
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
async def test_cli_chatgpt_login_publishes_replacement_challenge_after_invalid_grant(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(loopback, "CHATGPT_OAUTH_LOOPBACK_PORTS", (0,))
    monkeypatch.setattr(
        wiring_runtime_pipeline,
        "_default_vrchat_scene_service",
        lambda: VrchatSceneService(
            snapshots=SimpleNamespace(snapshots=lambda: ()),
            log_directory=tmp_path / "empty-scene",
        ),
    )
    browser_launches = []
    monkeypatch.setattr(
        application_runtime,
        "ChatGptAccountOwner",
        partial(application_runtime.ChatGptAccountOwner, browser_opener=browser_launches.append),
    )
    path = tmp_path / "settings.json"
    _isolated_settings(path)
    issued_client = "oaiapp_test"
    exchanges = []

    def oauth_response(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CHATGPT_JWKS_URL:
            return httpx.Response(200, json={"keys": [_jwk()]})
        assert str(request.url) == CHATGPT_TOKEN_URL
        exchanges.append(parse_qs(request.content.decode()))
        if len(exchanges) == 1:
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(
            200,
            json={
                "access_token": "test-access",
                "refresh_token": "test-refresh",
                "id_token": _id_token(client_id=issued_client, nonce=second["nonce"][0]),
                "scope": f"openid email offline_access {CHATGPT_PLAN_SCOPE}",
            },
        )

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(oauth_response)) as oauth,
        httpx.AsyncClient(trust_env=False, timeout=5) as browser,
    ):
        monkeypatch.setattr(
            application_runtime,
            "ChatGptSession",
            partial(ChatGptSession, http_factory=lambda: oauth),
        )
        app = compose_headless_application(path)
        try:
            await app.start()
            control = app.control()
            control.bind_instance("chatgpt-retry")
            assert (await control.query("auth.status", {}))["chatgpt"]["signed_in"] is False
            login = await control.submit(
                "auth.login",
                {"provider": "chatgpt", "open_browser": False},
                request_id="chatgpt-retry",
            )
            operation_id = login["operation_id"]

            async def advertised_challenge(previous: str | None = None) -> str:
                try:
                    async with asyncio.timeout(5):
                        while True:
                            status = await control.query("auth.status", {})
                            challenge = status["challenge"]
                            if challenge is not None:
                                assert challenge["operation_id"] == operation_id
                                assert challenge["provider"] == "chatgpt"
                                if (url := challenge["authorization_url"]) != previous:
                                    return url
                            pending = await control.operation(operation_id)
                            assert pending["terminal"] is False, pending
                            await asyncio.sleep(0.02)
                except TimeoutError:
                    stage = "replacement" if previous is not None else "initial"
                    pytest.fail(f"auth.status did not publish the {stage} ChatGPT challenge")

            async def complete_challenge(params, code, **callback_fields):
                redirect = httpx.URL(params["redirect_uri"][0])
                assert (redirect.scheme, redirect.host) == ("http", "127.0.0.1")
                response = await browser.get(
                    redirect, params={"state": params["state"][0], "code": code, **callback_fields}
                )
                assert response.status_code == 200

            first_url = await advertised_challenge()
            first = parse_qs(urlsplit(first_url).query)
            await complete_challenge(first, "first-code", client_id=issued_client)

            second = parse_qs(urlsplit(await advertised_challenge(first_url)).query)
            assert len(exchanges) == 1
            assert second["client_id"] == [issued_client]
            assert second["state"] != first["state"]
            assert second["code_challenge"] != first["code_challenge"]
            await complete_challenge(second, "second-code")

            result = await control.wait(operation_id, timeout=5)
            assert result["status"] == "applied", result
            assert result["terminal"] is True
            assert result["operation_id"] == operation_id
            assert [form["code"] for form in exchanges] == [["first-code"], ["second-code"]]
            for params, form in zip((first, second), exchanges, strict=True):
                assert form["client_id"] == [issued_client]
                verifier = form["code_verifier"][0].encode("ascii")
                challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier).digest()).rstrip(b"=")
                assert params["code_challenge"] == [challenge.decode("ascii")]
            status = await control.query("auth.status", {})
            assert status["challenge"] is None
            assert status["chatgpt"]["signed_in"] is True
            assert not browser_launches
        finally:
            await app.stop()
        assert oauth.is_closed


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
