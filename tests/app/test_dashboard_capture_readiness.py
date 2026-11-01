from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.composition import application_runtime
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.domain.events import STTSessionState, UIEvent, UIEventType
from puripuly_heart.ui.i18n import get_locale, set_locale, t
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from puripuly_heart.ui.views.dashboard import DashboardView
from tests.core.runtime.test_self_capture_session import RecordingAdmission, build_owner


@pytest.fixture
async def capture_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-readiness-passphrase")
    for name in ("LOCALAPPDATA", "APPDATA", "HF_HOME"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setenv(name, str(directory))
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    locale = get_locale()
    set_locale("en")
    base = AppSettingsVNext()
    settings = replace(
        base,
        intent=replace(
            base.intent,
            stt=replace(base.intent.stt, provider="deepgram"),
            peer_stt=replace(base.intent.peer_stt, provider="deepgram"),
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(base.intent.osc, connection_mode="off"),
            telemetry=replace(base.intent.telemetry, enabled=False),
        ),
    )
    dashboard = DashboardView()
    presentation = FletUiPresentationAdapter(SimpleNamespace(view_dashboard=dashboard))
    synthetic, _, provider, sources, _, _ = build_owner()
    compose_self = application_runtime.CaptureOwnerFactory.compose_self

    def compose_synthetic_capture(factory, *args, **kwargs):
        owner = compose_self(factory, *args, **kwargs)
        for field in (
            "_admission",
            "_provider",
            "_provider_request_factory",
            "_source_factory",
            "_vad_factory",
            "_run_audio_loop",
            "_vad_sink",
        ):
            setattr(owner, field, getattr(synthetic, field))
        return owner

    monkeypatch.setattr(
        application_runtime.CaptureOwnerFactory, "compose_self", compose_synthetic_capture
    )
    access = []
    app = application_runtime.compose_application_runtime(
        presentation=presentation,
        config_path=tmp_path / "settings.json",
        local_asr_evidence_sink=access.append,
    )
    try:
        await access[0].initialize(settings)
        owner = app.control().pipeline.self_capture
        bridge = presentation.create_ui_event_bridge(
            event_queue=asyncio.Queue(), runtime_logging=None
        )
        yield app, owner, dashboard, provider, bridge
    finally:
        await app.stop()
        set_locale(locale)
    assert all(source.close_calls == 1 for source in sources)


@pytest.mark.asyncio
async def test_ready_without_recognition_and_stopped_despite_late_session_events(
    capture_dashboard,
):
    app, owner, dashboard, _, bridge = capture_dashboard
    assert dashboard.display_card._display_text.value == t("display.disconnected")

    await app.set_stt_enabled(True)

    assert owner.snapshot.effective_active
    assert app.state().stt_state is None
    assert dashboard.display_card._display_text.value == t("display.connected")
    for channel in ("self", "peer"):
        await bridge._handle_event(
            UIEvent(
                type=UIEventType.SESSION_STATE_CHANGED,
                payload=STTSessionState.DISCONNECTED,
                channel=channel,
            )
        )
    assert dashboard.display_card._display_text.value == t("display.connected")

    await app.set_stt_enabled(False)
    await bridge._handle_event(
        UIEvent(
            type=UIEventType.SESSION_STATE_CHANGED,
            payload=STTSessionState.STREAMING,
            channel="self",
        )
    )

    assert not owner.snapshot.effective_active
    assert dashboard.display_card._display_text.value == t("display.disconnected")


@pytest.mark.asyncio
async def test_pending_start_can_stop_without_publishing_ready(capture_dashboard):
    app, owner, dashboard, _, _ = capture_dashboard
    admission_gate = asyncio.Event()
    owner._admission = RecordingAdmission(gate=admission_gate)
    start = asyncio.create_task(app.set_stt_enabled(True))
    try:
        async with asyncio.timeout(1):
            while not owner._admission.calls:
                await asyncio.sleep(0)
        assert not owner.snapshot.effective_active
        assert dashboard.display_card._display_text.value == t("display.connecting")

        await app.set_stt_enabled(False)
        await asyncio.gather(start, return_exceptions=True)

        assert not owner.snapshot.desired_active
        assert dashboard.display_card._display_text.value == t("display.disconnected")
    finally:
        admission_gate.set()
        await asyncio.gather(start, return_exceptions=True)


@pytest.mark.asyncio
async def test_start_failure_is_not_ready_and_recovery_preserves_specific_error(
    capture_dashboard,
):
    app, owner, dashboard, provider, _ = capture_dashboard
    provider.start_failure = RuntimeError("synthetic provider startup failure")

    await app.set_stt_enabled(True)

    assert not owner.snapshot.effective_active
    assert dashboard.display_card._display_text.value == t("display.failed")

    provider.start_failure = None
    admission_gate = asyncio.Event()
    owner._admission = RecordingAdmission(gate=admission_gate)
    recovery = asyncio.create_task(app.set_stt_enabled(True))
    try:
        async with asyncio.timeout(1):
            while not owner._admission.calls:
                await asyncio.sleep(0)
        assert dashboard.display_card._display_text.value == t("display.connecting")
        dashboard.set_display_text("Synthetic microphone error", is_error=True)
        admission_gate.set()
        await recovery
        assert owner.snapshot.effective_active
        assert dashboard.display_card._display_text.value == "Synthetic microphone error"
        dashboard._restore_status_display()
        assert dashboard.display_card._display_text.value == t("display.connected")
    finally:
        admission_gate.set()
        await asyncio.gather(recovery, return_exceptions=True)


@pytest.mark.asyncio
async def test_capture_restart_retains_translation_and_active_notice(capture_dashboard):
    app, owner, dashboard, _, _ = capture_dashboard
    await app.set_stt_enabled(True)
    dashboard.set_display_text("Synthetic source")
    dashboard.set_display_translation_text("Synthetic translation")
    dashboard.set_managed_auth_pending(True)
    notice = dashboard.display_card._display_text.value

    await app.set_stt_enabled(False)
    await app.set_stt_enabled(True)

    assert owner.snapshot.effective_active
    assert dashboard.display_card._display_text.value == notice
    dashboard.set_managed_auth_pending(False)
    assert dashboard.display_card._display_text.value == "Synthetic translation"
    dashboard._restore_status_display()
    assert dashboard.display_card._display_text.value == t("display.connected")
