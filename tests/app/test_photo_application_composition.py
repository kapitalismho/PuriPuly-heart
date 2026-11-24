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
from puripuly_heart.app.ports.settings_view import PhotoStatusSnapshot
from puripuly_heart.app.services.photo_application import PhotoApplicationOwner
from puripuly_heart.app.wiring import wiring_runtime_pipeline
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.headless_application import HeadlessApplicationPresentation
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, PhotoIntent, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core import network_clients
from puripuly_heart.core.chatgpt.session import ChatGptSession
from puripuly_heart.core.photo.capture import PhotoCapture, PhotoCaptureWatcher
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
        control = app.control()
        control.bind_instance("photo-capture-settings")
        disabled = await control.submit(
            "settings.apply",
            {"changes": {"photo.enabled": False}},
            request_id="photo-capture-disable",
        )
        disabled_result = await control.wait(disabled["operation_id"], timeout=10)
        assert disabled_result["status"] in {"applied", "degraded"}
        await until(lambda: owner.lifecycle.snapshot["photo"] is None)
        (photos / "while-disabled.png").write_bytes(png(64, 64))
        enabled = await control.submit(
            "settings.apply",
            {"changes": {"photo.enabled": True}},
            request_id="photo-capture-enable",
        )
        enabled_result = await control.wait(enabled["operation_id"], timeout=10)
        assert enabled_result["status"] in {"applied", "degraded"}
        assert owner.lifecycle.snapshot["photo"] is None
        month = photos / "2026-12"
        month.mkdir()
        next_photo = month / "VRChat_2026-12-01_00-00-00.001_64x64.png"
        next_photo.write_bytes(png(64, 64))
        await until(
            lambda: owner.lifecycle.snapshot["photo"] is not None
            and owner.lifecycle.snapshot["photo"]["path"] == str(next_photo.resolve())
        )
        assert records["starts"] == 1
        print(
            {
                "composed_startup": True,
                "caption_enabled": False,
                "photo_pid_before_pipeline_replacement": photo_pid,
                "photo_pid_after_replacement": photo_pid,
                "photo_generation_after_crash": owner.supervisor.generation,
                "starts": records["starts"],
                "settings_reactivation_new_month_capture": True,
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


@pytest.mark.parametrize("openai_key", ["synthetic-openai", None])
async def test_composed_openai_route_uses_only_the_official_openai_key(
    tmp_path, monkeypatch, openai_key
):
    monkeypatch.delenv("PURIPULY_PHOTO_BENCHMARK_ROUTE", raising=False)
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(ChatGptSession, "status", lambda self: SimpleNamespace(signed_in=True))
    store = {"openai_api_key": openai_key, "gemini_api_key": "synthetic-google"}
    monkeypatch.setattr(application_runtime, "create_secret_store", lambda *args, **kwargs: store)
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
    app = application_runtime.compose_application_runtime(
        presentation=HeadlessApplicationPresentation(), config_path=tmp_path / "settings.json"
    )
    canonical = AppSettingsVNext()
    app._settings.settings.canonical = replace(
        canonical, intent=replace(canonical.intent, photo=PhotoIntent(route="openai"))
    )
    try:
        route = await owners[0]._resolve_route()
        if openai_key is None:
            assert route is None
            return
        assert photo_provider.photo_route_kind(route) == "openai"
        connection = route.connect(None)
        assert isinstance(connection, photo_provider.OpenAIPhotoConnection)
        assert connection.endpoint == "https://api.openai.com/v1/responses"
        assert connection._api_key == openai_key
        with pytest.raises(photo_provider.PhotoProviderError):
            route.connect("gemini-3.5-flash-lite")
        store["openai_api_key"] = "rotated-openai"
        assert (await owners[0]._resolve_route()).identity != route.identity
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


async def test_composed_photo_settings_gate_offers_report_status_and_discard_only_when_enabled(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PURIPULY_PHOTO_BENCHMARK_ROUTE", raising=False)
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(ChatGptSession, "status", lambda self: SimpleNamespace(signed_in=False))
    monkeypatch.setattr(application_runtime, "create_secret_store", lambda *args, **kwargs: {})

    async def allowance(self):
        return 72, 100

    monkeypatch.setattr(photo_provider.PhotoProviderAdapter, "allowance", allowance)
    recycled = []
    monkeypatch.setattr(
        application_runtime,
        "recycle_captured_photo",
        lambda photo_capture: recycled.append(photo_capture) or True,
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
    app = application_runtime.compose_application_runtime(
        presentation=HeadlessApplicationPresentation(), config_path=tmp_path / "settings.json"
    )
    canonical = AppSettingsVNext()
    photo = PhotoCapture(tmp_path / "shot.png", 1, (1, 2), 0, 0.0)

    def apply_photo(**values):
        app._settings.settings.canonical = replace(
            canonical, intent=replace(canonical.intent, photo=PhotoIntent(**values))
        )

    try:
        owner = owners[0]
        apply_photo()
        assert (await owner._resolve_route()).identity == "managed-luna"
        assert await app.photo_status() == PhotoStatusSnapshot("managed", 72, 100)
        await owner.lifecycle._discard_translated(photo)
        assert recycled == []

        apply_photo(route="chatgpt")
        assert await owner._resolve_route() is None
        assert (await app.photo_status()).route is None
        apply_photo(route="gemini")
        assert await owner._resolve_route() is None
        apply_photo(route="openai")
        assert await owner._resolve_route() is None

        apply_photo(enabled=False, delete_translated=True)
        assert await owner._resolve_route() is None
        assert (await app.photo_status()).route == "managed"
        await owner.lifecycle._discard_translated(photo)
        assert recycled == [photo]
    finally:
        await app.stop()


async def test_composed_photo_status_query_names_why_photos_are_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("PURIPULY_PHOTO_BENCHMARK_ROUTE", raising=False)
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    monkeypatch.setattr(ChatGptSession, "status", lambda self: SimpleNamespace(signed_in=False))
    monkeypatch.setattr(application_runtime, "create_secret_store", lambda *args, **kwargs: {})
    fetches = []

    async def allowance(self):
        fetches.append(True)
        return 0, 50

    monkeypatch.setattr(photo_provider.PhotoProviderAdapter, "allowance", allowance)
    monkeypatch.setattr(photo_provider.PhotoProviderAdapter, "cached_allowance", lambda self: None)
    app = application_runtime.compose_application_runtime(
        presentation=HeadlessApplicationPresentation(), config_path=tmp_path / "settings.json"
    )
    canonical = AppSettingsVNext()

    def apply_photo(**values):
        current = replace(canonical, intent=replace(canonical.intent, photo=PhotoIntent(**values)))
        app._settings.settings.canonical = current
        app.control().settings.committed_settings = current

    async def status(**arguments):
        result = await app.control().query("photo.status", arguments)
        return {key: result[key] for key in result if key not in {"instance_id", "revision"}}

    try:
        apply_photo()
        assert await status() == {
            "enabled": True,
            "route": "managed",
            "availability": "allowance_unknown",
        }
        assert fetches == []
        assert await status(refresh=True) == {
            "enabled": True,
            "route": "managed",
            "availability": "allowance_exhausted",
            "remaining": 0,
            "allowance": 50,
        }
        apply_photo(route="gemini")
        assert (await status(refresh=True))["availability"] == "not_connected"
        apply_photo(enabled=False)
        assert (await status())["availability"] == "off"
        assert fetches == [True]
    finally:
        await app.stop()
