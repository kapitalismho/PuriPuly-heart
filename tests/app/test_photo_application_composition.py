from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters import photo_provider
from puripuly_heart.app.services.photo_application import PhotoApplicationOwner
from puripuly_heart.app.wiring import wiring_runtime_pipeline
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.headless_application import HeadlessApplicationPresentation
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core import network_clients
from puripuly_heart.core.chatgpt.session import ChatGptSession
from puripuly_heart.core.photo.capture import PhotoCaptureWatcher
from puripuly_heart.core.photo.process import PhotoProcessRunner
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks
from puripuly_heart.core.vrchat_scene_service import VrchatSceneService
from tests.core.test_photo_native_process import ControlledConnection, png, until


@pytest.mark.skipif(
    not os.environ.get("PHOTO_NATIVE_EXECUTABLE"),
    reason="explicit native photo executable required",
)
async def test_composed_startup_starts_photo_with_captions_off_survives_pipeline_replacement_and_stops_child(
    tmp_path, monkeypatch
):
    for variable in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(variable, str(tmp_path / variable))
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "synthetic-photo-test")
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
            log_directory=tmp_path / "isolated-scene",
        ),
    )
    network_attempts = []

    def deny_transport(*args, **kwargs):
        network_attempts.append("external")
        raise AssertionError("external transport forbidden in synthetic photo composition check")

    monkeypatch.setattr(network_clients, "external_async_client", deny_transport)
    monkeypatch.setattr(network_clients, "external_client", deny_transport)
    monkeypatch.setattr(network_clients, "external_websocket_connect", deny_transport)
    monkeypatch.setattr(
        ChatGptSession,
        "status",
        lambda self: SimpleNamespace(signed_in=True, email=None, diagnostics=None),
    )
    records = {"preopened": 0, "starts": 0, "closed": 0}
    routes = []

    def connection(**kwargs):
        assert kwargs["route"] == "luna"
        assert isinstance(kwargs["chatgpt_session"], ChatGptSession)
        assert "gemini_api_key" not in kwargs
        routes.append(kwargs["route"])
        return ControlledConnection(records)

    monkeypatch.setattr(photo_provider, "make_photo_connection", connection)
    monkeypatch.setenv("PURIPULY_PHOTO_BENCHMARK_ROUTE", "personal")
    photos = tmp_path / "photos"
    photos.mkdir()
    owned = tmp_path / "owned"
    owned.mkdir()
    photo_owners = []

    def make_photo_owner(**kwargs):
        owner = PhotoApplicationOwner(
            **kwargs,
            watcher=PhotoCaptureWatcher(photos, tmp_path / "isolated-camera-log"),
            runner=PhotoProcessRunner(Path(os.environ["PHOTO_NATIVE_EXECUTABLE"])),
            backend="probe",
            temporary_root=owned,
            poll_interval=0.01,
        )
        photo_owners.append(owner)
        return owner

    monkeypatch.setattr(application_runtime, "PhotoApplicationOwner", make_photo_owner)
    base = AppSettingsVNext()
    settings = replace(
        base,
        intent=replace(
            base.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            languages=replace(base.intent.languages, target_language="ko"),
            stt=replace(base.intent.stt, provider="soniox"),
            peer_stt=replace(base.intent.peer_stt, provider="soniox"),
            osc=replace(
                base.intent.osc,
                connection_mode="off",
                activation_notice_enabled=False,
                chatbox_send=False,
            ),
            telemetry=replace(base.intent.telemetry, enabled=False),
            integrated_context=replace(base.intent.integrated_context, enabled=False),
            clipboard=replace(base.intent.clipboard, auto_translate_enabled=False),
        ),
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    log_path = tmp_path / "runtime.log"
    sinks = RuntimeLoggingSinks(
        logging.StreamHandler(io.StringIO()),
        logging.FileHandler(log_path, encoding="utf-8"),
        log_path,
    )
    access = []
    presentation = HeadlessApplicationPresentation()
    app = application_runtime.compose_application_runtime(
        presentation=presentation,
        config_path=path,
        runtime_logging_sinks=sinks,
        local_asr_evidence_sink=access.append,
    )
    presentation.application = app
    try:
        await app.start()
        owner = photo_owners[0]
        assert owner._started
        assert not app._settings.settings.overlay_enabled()
        (photos / "capture.png").write_bytes(png(1280, 720))
        await until(
            lambda: owner.lifecycle.snapshot["photo"] is not None and owner.bridge.ready.is_set()
        )
        photo_id = owner.lifecycle.snapshot["photo"]["photo_id"]
        assert records["starts"] == 0
        owner.bridge.send(
            {"type": "photo_probe_pull", "photo_id": photo_id, "event_id": "app-pull"}
        )
        await until(lambda: owner.lifecycle.snapshot["photo"]["phase"] in ("completed", "failed"))
        assert owner.lifecycle.snapshot["photo"]["phase"] == "completed"
        photo_pid = owner.supervisor.pid
        await access[0].initialize(settings)
        assert owner.supervisor.pid == photo_pid
        assert owner.lifecycle.snapshot["photo"]["phase"] == "completed"
        assert records["starts"] == 1
        assert not app._settings.settings.overlay_enabled()
        generation = owner.supervisor.generation
        await owner.supervisor.process.terminate()
        await until(
            lambda: owner.supervisor.generation > generation and owner.bridge.ready.is_set()
        )
        assert not app._settings.settings.overlay_enabled()
        assert records["starts"] == 1
        print(
            {
                "composed_startup": True,
                "caption_enabled": False,
                "photo_pid_before_pipeline_replacement": photo_pid,
                "photo_pid_after_replacement": photo_pid,
                "photo_generation_after_crash": owner.supervisor.generation,
                "starts": records["starts"],
                "controlled": "provider+credentials; no live services",
            }
        )
    finally:
        await app.stop()
        sinks.close()
        sinks.stream_handler.close()
    assert photo_owners[0].supervisor.process is None
    assert photo_owners[0].supervisor.last_shutdown_acknowledged
    assert list(owned.iterdir()) == []
    assert network_attempts == []


@pytest.mark.parametrize(
    ("signed_in", "google_key", "expected"),
    [
        (True, "synthetic-google", "luna"),
        (False, "synthetic-google", "google"),
        (None, "synthetic-google", "google"),
        (False, None, None),
        (None, None, None),
    ],
)
async def test_composed_route_priority_requires_chatgpt_session_not_openai_key(
    tmp_path, monkeypatch, signed_in, google_key, expected
):
    monkeypatch.setenv("PURIPULY_PHOTO_BENCHMARK_ROUTE", "personal")
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(ChatGptSession, "status", lambda self: SimpleNamespace(signed_in=signed_in))
    store = {
        "gemini_api_key": google_key,
        "google_api_key": "retired-key-must-not-authorize-photos",
        "openai_api_key": "must-never-authorize-luna",
    }
    monkeypatch.setattr(application_runtime, "create_secret_store", lambda *args, **kwargs: store)
    owners = []
    calls = []

    def make_owner(**kwargs):
        owner = PhotoApplicationOwner(
            **kwargs,
            watcher=PhotoCaptureWatcher(tmp_path / "photos", tmp_path / "logs"),
            temporary_root=tmp_path,
        )
        owners.append(owner)
        return owner

    monkeypatch.setattr(application_runtime, "PhotoApplicationOwner", make_owner)
    monkeypatch.setattr(
        photo_provider, "make_photo_connection", lambda **kwargs: calls.append(kwargs)
    )
    app = application_runtime.compose_application_runtime(
        presentation=HeadlessApplicationPresentation(), config_path=tmp_path / "settings.json"
    )
    app._settings.settings.canonical = AppSettingsVNext()
    try:
        route = await owners[0]._resolve_route()
        if expected is None:
            assert route is None
        else:
            route.connect(None)
            assert calls[0]["route"] == expected
            if expected == "luna":
                assert isinstance(calls[0]["chatgpt_session"], ChatGptSession)
                assert "gemini_api_key" not in calls[0]
            else:
                assert calls[0]["gemini_api_key"] == google_key
                assert "chatgpt_session" not in calls[0]
    finally:
        await app.stop()


async def test_composed_default_route_uses_relay_and_reads_the_local_wrist_calibration(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PURIPULY_PHOTO_BENCHMARK_ROUTE", raising=False)
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(ChatGptSession, "status", lambda self: SimpleNamespace(signed_in=True))
    monkeypatch.setattr(
        application_runtime,
        "create_secret_store",
        lambda *args, **kwargs: {"gemini_api_key": "synthetic-google"},
    )
    owners = []

    def make_owner(**kwargs):
        owner = PhotoApplicationOwner(
            **kwargs,
            watcher=PhotoCaptureWatcher(tmp_path / "photos", tmp_path / "logs"),
            temporary_root=tmp_path,
        )
        owners.append(owner)
        return owner

    monkeypatch.setattr(application_runtime, "PhotoApplicationOwner", make_owner)
    calibration = {
        "transform": [[1.0, 0.0, 0.0, 0.05], [0.0, 1.0, 0.0, 0.03], [0.0, 0.0, 1.0, 0.06]],
        "reference": {
            "model": "synthetic_controller_right",
            "serial": "synthetic-serial",
            "handgrip_transform": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        },
    }
    app = application_runtime.compose_application_runtime(
        presentation=HeadlessApplicationPresentation(), config_path=tmp_path / "settings.json"
    )
    canonical = AppSettingsVNext()
    app._settings.settings.canonical = replace(
        canonical,
        intent=replace(
            canonical.intent,
            languages=replace(
                canonical.intent.languages, source_language="ja", target_language="en"
            ),
        ),
    )
    try:
        owner = owners[0]
        route = await owner._resolve_route()
        assert route.identity == "managed-luna"
        assert owner.lifecycle._target_language() == "ja"
        assert owner._interaction() is None
        (tmp_path / "photo-wrist-calibration.json").write_text(
            json.dumps(calibration), encoding="utf-8"
        )
        assert owner._interaction() == {"wrist_calibration": calibration}
        (tmp_path / "photo-wrist-calibration.json").write_text("{broken", encoding="utf-8")
        assert owner._interaction() is None
    finally:
        await app.stop()
