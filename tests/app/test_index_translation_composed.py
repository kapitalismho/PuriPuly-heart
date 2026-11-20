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


def index_settings(settings):
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model="index_translate_35b",
                connection="public_api",
                openrouter_selected_source="none",
                openrouter_selection_alias=None,
            ),
        ),
    )


class ControlledPresentation(HeadlessApplicationPresentation):
    def __init__(self):
        super().__init__()
        self.messages = []

    def show_message(self, key, **kwargs):
        self.messages.append((key, kwargs))


class ControlledHttp:
    def __init__(self):
        self.index_status = 200
        self.requests = []

    async def handle(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.host, request.url.path, request.headers, body))
        if request.url.host != "index-translate.bilibili.com":
            raise AssertionError(f"unexpected external request: {request.url.host}")
        if self.index_status != 200:
            return httpx.Response(self.index_status, json={"error": "synthetic outage"})
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
async def test_unreachable_index_keeps_selection_and_enables_without_account(composed):
    from puripuly_heart.config.settings_vnext.facade import load_vnext_settings

    path, http, create = composed
    http.index_status = 503
    app, presentation = create()
    await app.start()
    enabled = await operation(app, "translation.set", {"enabled": True}, "enable-without-account")
    assert enabled["status"] == "applied"
    assert app.state().translation_enabled is True
    assert http.requests == []
    assert presentation.messages == []
    current = app.compatibility_settings().intent.translation
    assert (current.model, current.connection) == ("index_translate_35b", "public_api")
    await app.stop()
    persisted = load_vnext_settings(path).settings.intent.translation
    assert (persisted.model, persisted.connection) == ("index_translate_35b", "public_api")
    fresh, fresh_presentation = create()
    await fresh.start()
    restarted = fresh.compatibility_settings().intent.translation
    assert (restarted.model, restarted.connection) == ("index_translate_35b", "public_api")
    assert http.requests == []
    assert fresh_presentation.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("credential_available", [False, True])
@pytest.mark.parametrize(
    "status_code,category", [(429, "rate_limit"), (503, "service_unavailable")]
)
async def test_real_request_failure_preserves_index_and_reports_error(
    composed, credential_available, status_code, category
):
    from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
    from puripuly_heart.config.settings_vnext.facade import load_vnext_settings

    path, http, create = composed
    if credential_available:
        seed = load_vnext_settings(path).settings
        store = create_secret_store(seed.intent.secrets, config_path=path)
        store.set("openrouter_managed_api_key", "synthetic-managed-key")
    app, presentation = create()
    await app.start()
    enabled = await operation(app, "translation.set", {"enabled": True}, "initial-enable")
    assert enabled["status"] == "applied"
    assert http.index_requests == []
    http.index_status = status_code
    admitted = await operation(app, "text.submit", {"text": "synthetic user text"}, "failed-turn")
    assert admitted["status"] == "applied"
    await wait_until(lambda: presentation.dashboard.issue is not None)
    history = await app.control().query("diagnostics.history", {"limit": 20})
    failures = [
        record
        for record in history["records"]
        if record["component"] == "provider.llm" and record["operation"] == "translate"
    ]
    assert len(failures) == 1
    assert failures[0]["status_code"] == status_code
    assert failures[0]["category"] == category
    assert len(http.index_requests) == 2
    assert all(request[0] == "index-translate.bilibili.com" for request in http.requests)
    assert presentation.dashboard.translation is None
    current = app.compatibility_settings().intent.translation
    persisted = load_vnext_settings(path).settings.intent.translation
    assert (current.model, current.connection) == ("index_translate_35b", "public_api")
    assert (persisted.model, persisted.connection) == (current.model, current.connection)
    assert app.state().translation_enabled is True
    http.index_status = 200
    resumed = await operation(
        app, "text.submit", {"text": "synthetic recovery text"}, "next-index-turn"
    )
    assert resumed["status"] == "applied"
    await wait_until(lambda: presentation.dashboard.translation == "Synthetic translation")
    assert len(http.index_requests) == 3
    assert "synthetic recovery text" in http.index_requests[-1][3]["messages"][1]["content"]
