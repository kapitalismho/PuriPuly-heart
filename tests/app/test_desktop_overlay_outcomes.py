from __future__ import annotations

import io
import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.settings_view import LocaleSettingsIntent
from puripuly_heart.composition.headless_application import compose_headless_application
from puripuly_heart.config.resolved import OVERLAY_TARGET_DESKTOP
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    DesktopFletOverlayPositionIntent,
    SecretsIntent,
)
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks


@pytest.fixture(autouse=True)
async def isolated_composed_logging(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "synthetic-desktop-passphrase")
    root = logging.getLogger()
    level = root.level
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(root, "filters", list(root.filters))
    monkeypatch.setattr(root, "level", level)
    for handler in root.handlers:
        monkeypatch.setattr(handler, "filters", list(handler.filters))
    compose = compose_headless_application
    applications = []

    def tracked_compose(path):
        log_file = tmp_path / f"headless-{len(applications)}.log"
        sinks = RuntimeLoggingSinks(
            logging.StreamHandler(io.StringIO()),
            logging.FileHandler(log_file, encoding="utf-8"),
            log_file,
        )
        try:
            application = compose(path, runtime_logging_sinks=sinks)
        except BaseException:
            sinks.close()
            raise
        applications.append((application, sinks))
        return application

    monkeypatch.setitem(globals(), "compose_headless_application", tracked_compose)
    try:
        yield
    finally:
        try:
            for application, sinks in reversed(applications):
                try:
                    if application.control().pipeline.self_translation_channel is not None:
                        await application.stop()
                    else:
                        service = application._runtime_logging.installed_service
                        if service is not None:
                            service.close()
                finally:
                    sinks.close()
                    sinks.stream_handler.close()
        finally:
            root.setLevel(level)


def _application(path: Path, *, desktop: bool = False, positioned: bool = False):
    settings = AppSettingsVNext()
    position = (
        DesktopFletOverlayPositionIntent(x=140, y=85)
        if positioned
        else settings.intent.overlay.desktop_flet.position
    )
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            telemetry=replace(settings.intent.telemetry, enabled=False),
            osc=replace(settings.intent.osc, connection_mode="off"),
            overlay=replace(
                settings.intent.overlay,
                target=OVERLAY_TARGET_DESKTOP if desktop else settings.intent.overlay.target,
                desktop_flet=replace(settings.intent.overlay.desktop_flet, position=position),
            ),
        ),
    )
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    application = compose_headless_application(path)
    owner = application._settings.settings
    owner.start()
    application._settings.projection.remember_all(owner.canonical)
    return application, owner


def _persisted_desktop(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["intent"]["overlay"]["desktop_flet"]


class ControlledBridge:
    def __init__(self, *, fails_at: int | None = None):
        self.sent: list[dict] = []
        self.fails_at = fails_at

    async def broadcast_desktop_runtime_control(self, payload):
        if len(self.sent) + 1 == self.fails_at:
            raise OSError("controlled bridge unavailable")
        self.sent.append(payload)


def _connect(application, bridge: ControlledBridge | None) -> None:
    overlay = application._overlay.overlay
    overlay.active_target = OVERLAY_TARGET_DESKTOP
    overlay.state = "connected"
    overlay.runtime = SimpleNamespace(
        current_bridge_for_runtime_command=lambda: bridge,
        renderer_events_or_none=lambda: None,
    )


@pytest.mark.asyncio
async def test_lock_reports_unavailable_and_failed_bridge_without_changing_mode(tmp_path: Path):
    application, _owner = _application(tmp_path / "settings.json", desktop=True)
    assert (await application.set_desktop_overlay_captions_locked(True))[
        "status"
    ] == "action_required"
    _connect(application, None)
    assert (await application.set_desktop_overlay_captions_locked(True))[
        "reason"
    ] == "desktop_bridge_unavailable"
    failing = ControlledBridge(fails_at=1)
    _connect(application, failing)
    assert (await application.set_desktop_overlay_captions_locked(True))["status"] == "failed"
    assert not application._overlay.desktop.captions_locked
    healthy = ControlledBridge()
    _connect(application, healthy)
    assert (await application.set_desktop_overlay_captions_locked(True))["status"] == "applied"
    assert application._overlay.desktop.captions_locked
    assert healthy.sent == [{"command": "set_interaction_mode", "mode": "pass_through"}]


@pytest.mark.asyncio
async def test_size_reports_failed_save_and_idempotent_persisted_configuration(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "settings.json"
    application, owner = _application(path)
    baseline = _persisted_desktop(path)
    next_preset = next(
        preset
        for preset in application._overlay.desktop.policy.size_presets
        if preset != baseline["size_preset"]
    )
    persist = owner.persistence.persist

    def locked_file(_path, _settings):
        raise PermissionError("controlled locked settings file")

    monkeypatch.setattr(owner.persistence, "persist", locked_file)
    failed = await application.set_desktop_overlay_size_preset(next_preset)
    assert failed["status"] == "persistence_failed"
    assert owner.canonical.intent.overlay.desktop_flet.size_preset == baseline["size_preset"]
    assert _persisted_desktop(path) == baseline
    monkeypatch.setattr(owner.persistence, "persist", persist)
    applied = await application.set_desktop_overlay_size_preset(next_preset)
    assert applied["status"] == "applied"
    assert _persisted_desktop(path)["size_preset"] == next_preset
    assert (await application.set_desktop_overlay_size_preset(next_preset))["status"] == "applied"


@pytest.mark.asyncio
async def test_reset_requires_desktop_and_preserves_position_when_save_fails(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "settings.json"
    application, owner = _application(path, positioned=True)
    assert (await application.reset_desktop_overlay_position())["status"] == "action_required"
    assert _persisted_desktop(path)["position"] == {"x": 140, "y": 85}
    owner.canonical = replace(
        owner.canonical,
        intent=replace(
            owner.canonical.intent,
            overlay=replace(owner.canonical.intent.overlay, target=OVERLAY_TARGET_DESKTOP),
        ),
    )
    owner.persistence.persist(path, owner.canonical)
    application._settings.projection.remember_all(owner.canonical)

    def locked_file(_path, _settings):
        raise PermissionError("controlled locked settings file")

    monkeypatch.setattr(owner.persistence, "persist", locked_file)
    assert (await application.reset_desktop_overlay_position())["status"] == "persistence_failed"
    assert _persisted_desktop(path)["position"] == {"x": 140, "y": 85}
    assert owner.canonical.intent.overlay.desktop_flet.position.x == 140


@pytest.mark.asyncio
async def test_reset_commits_inactive_desktop_and_reports_failed_runtime_broadcast(tmp_path: Path):
    path = tmp_path / "settings.json"
    application, _owner = _application(path, desktop=True, positioned=True)
    assert (await application.reset_desktop_overlay_position())["status"] == "applied"
    assert _persisted_desktop(path)["position"] == {"x": None, "y": None}
    assert (await application.reset_desktop_overlay_position())["status"] == "applied"
    failing = ControlledBridge(fails_at=2)
    _connect(application, failing)
    result = await application.reset_desktop_overlay_position()
    assert result["status"] == "failed"
    assert result["reason"] == "desktop_broadcast_failed"
    assert [payload["command"] for payload in failing.sent] == ["set_interaction_mode"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["size", "reset"])
async def test_runtime_desktop_commit_survives_later_unrelated_focused_edit(tmp_path, action):
    path = tmp_path / "settings.json"
    application, owner = _application(path, desktop=True, positioned=True)
    before = owner.require_canonical()
    unrelated = LocaleSettingsIntent("ja")
    desktop = before.intent.overlay.desktop_flet
    if action == "size":
        size = "large" if desktop.size_preset != "large" else "tiny"
        expected = replace(desktop, size_preset=size)
        result = await application.set_desktop_overlay_size_preset(size)
    else:
        expected = replace(desktop, position=DesktopFletOverlayPositionIntent(x=None, y=None))
        result = await application.reset_desktop_overlay_position()
    assert result["status"] == "applied"
    assert owner.require_canonical().intent.overlay.desktop_flet == expected

    await application.apply_settings_intent(unrelated)

    current = owner.require_canonical()
    assert current.intent == replace(
        before.intent,
        ui=replace(before.intent.ui, locale="ja"),
        overlay=replace(before.intent.overlay, desktop_flet=expected),
    )
    loaded = SettingsVNextCanonicalPersistenceAdapter().load_active(path).canonical_settings
    assert loaded == current
