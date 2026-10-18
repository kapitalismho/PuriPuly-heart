from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.settings_view import SelfVadSettingsIntent
from puripuly_heart.app.wiring import wiring_audio_test
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_self_capture_session_config_from_vnext,
)
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.headless_application import compose_headless_application
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmission,
    PeerCaptureAdmissionStatus,
    PeerCaptureResolvedTarget,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.core.self_capture import SelfCaptureAdmission, SelfCaptureAdmissionStatus
from tests.app.test_audio_test_session_owner import CaptureControl, FeedSource, LoudnessEngine
from tests.app.test_audio_test_sources import Resolver, _route
from tests.core.runtime.test_self_capture_session import build_owner


@pytest.fixture
async def audio_application(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-audio-test-passphrase")
    for name in ("LOCALAPPDATA", "APPDATA", "HF_HOME"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setenv(name, str(directory))
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    initial = AppSettingsVNext()
    initial = replace(
        initial,
        intent=replace(
            initial.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(initial.intent.osc, connection_mode="off"),
            telemetry=replace(initial.intent.telemetry, enabled=False),
        ),
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(to_dict(initial)), encoding="utf-8")
    app = compose_headless_application(path)
    await app.start()
    control = app.control()
    control.bind_instance("audio-repair-controlled-instance")
    audio = app._audio_test.audio_test
    sources: list[FeedSource] = []

    def source(*_args, **_kwargs):
        value = FeedSource()
        value.feed(0.5, chunks=4)
        sources.append(value)
        return value

    audio.vad_engine_factory = LoudnessEngine
    audio.microphone_source_factory = source
    audio.loopback_source_factory = source
    monkeypatch.setattr(wiring_audio_test, "observe_microphone_test_route", lambda **_kw: _route())
    monkeypatch.setattr(
        wiring_audio_test,
        "determine_self_mic_capture_channels",
        lambda **_kw: SimpleNamespace(preferred_capture_channels=1),
    )
    core_target = wiring_audio_test.build_peer_capture_target_from_vnext(
        app.compatibility_settings()
    )
    audio.target_resolver = Resolver(
        PeerCaptureTargetResolution(
            status=PeerCaptureTargetStatus.RESOLVED,
            target=PeerCaptureResolvedTarget(intent=core_target),
        )
    )
    try:
        yield app, control, audio, sources, path
    finally:
        await app.stop()
        assert all(value.closed for value in sources)


@pytest.mark.asyncio
async def test_actual_self_vad_intent_prepares_paused_owner_then_restores_capture(
    audio_application,
):
    app, control, audio, sources, path = audio_application
    owner = control.pipeline.self_capture
    synthetic, *_ = build_owner()
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
    app._input_runtime.self_capture.persist_manual_fallback = lambda: True
    await app.apply_settings_intent(SelfVadSettingsIntent(speech_threshold=0.45))
    await app.set_stt_enabled(True)
    assert owner.snapshot.desired_active
    assert await app.start_audio_test()
    assert await app.wait_audio_test_ready()
    assert not owner.snapshot.desired_active
    assert audio.snapshot["restore_capture"] == ["self"]
    generation = owner.snapshot.generation
    token = audio.capture_control.state("self").token
    await app.apply_settings_intent(SelfVadSettingsIntent(speech_threshold=0.77))
    assert app.compatibility_settings().intent.stt.vad_speech_threshold == pytest.approx(0.77)
    assert json.loads(path.read_text(encoding="utf-8"))["intent"]["stt"][
        "vad_speech_threshold"
    ] == pytest.approx(0.77)
    assert owner.snapshot.generation > generation
    assert audio.capture_control.state("self").token == token
    source = audio.owner().runtime_for("self").source
    source.feed(0.0, chunks=100)
    source.feed(0.3, chunks=4)
    for _ in range(500):
        if audio.snapshot["channels"]["self"]["speech_threshold"] == 0.77:
            break
        await asyncio.sleep(0.001)
    assert audio.snapshot["channels"]["self"]["speech_threshold"] == pytest.approx(0.77)
    await app.stop_audio_test()
    assert owner.snapshot.desired_active
    assert owner.snapshot.effective_active
    assert not audio.active


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ("gui", "cli", "osc"))
@pytest.mark.parametrize("channel", ("self", "peer"))
@pytest.mark.parametrize("enabled", (False, True))
async def test_explicit_capture_entrypoints_release_restore_authority(
    audio_application, entrypoint, channel, enabled
):
    app, control, audio, _sources, _path = audio_application
    capture = CaptureControl(self_enabled=True, peer_enabled=True)
    audio.capture_control = capture
    assert await app.start_audio_test()
    assert await app.wait_audio_test_ready()
    assert audio.snapshot["restore_capture"] == ["self", "peer"]
    changes = []
    self_owner = control.pipeline.self_capture
    peer_owner = control.pipeline.current.peer_capture

    async def reject_self(_config):
        return SelfCaptureAdmission(
            SelfCaptureAdmissionStatus.REJECTED, reason="controlled_unavailable"
        )

    async def reject_peer(_config):
        return PeerCaptureAdmission(
            PeerCaptureAdmissionStatus.REJECTED, reason="controlled_unavailable"
        )

    self_owner._admission = SimpleNamespace(admit=reject_self)
    peer_owner._admission = SimpleNamespace(admit=reject_peer)

    async def self_enabled(value):
        assert not value or not audio.active
        changes.append(("self", value))
        await self_owner.apply_intent(
            build_self_capture_session_config_from_vnext(app.compatibility_settings()),
            enabled=value,
        )

    async def peer_enabled(value):
        assert not value or not audio.active
        changes.append(("peer", value))
        await peer_owner.apply_intent(
            build_peer_capture_session_config_from_vnext(app.compatibility_settings()),
            enabled=value,
        )

    app._input_runtime = SimpleNamespace(set_stt_enabled=self_enabled)
    app._peer_capture = SimpleNamespace(set_peer_translation_enabled=peer_enabled)
    if entrypoint == "gui":
        await (
            app.set_stt_enabled(enabled)
            if channel == "self"
            else app.set_peer_translation_enabled(enabled)
        )
    elif entrypoint == "cli":
        receipt = await control.submit(
            "capture.set",
            {
                "channel": channel,
                "enabled": enabled,
                **({"accept_terms": True} if channel == "peer" and enabled else {}),
            },
            request_id="capture-override",
        )
        result = await control.wait(receipt["operation_id"], timeout=3)
        assert result["terminal"]
        assert result["status"] == ("degraded" if enabled else "applied")
    else:
        osc = app._settings.application.runtime_effects._vrc_mic_sync
        osc.router.set_ingress_enabled(True)
        parameter = "PuriPuly_Talk" if channel == "self" else "PuriPuly_Listen"
        result = await osc.router.dispatch_packet("/avatar/parameters/" + parameter, enabled)
        assert result.applied is not enabled
        assert result.error == ("application_rejected" if enabled else None)
    assert changes == [(channel, enabled)]
    assert channel not in audio.snapshot["restore_capture"]
    await app.stop_audio_test()
    assert capture.enabled[channel] is False
    assert capture.enabled["peer" if channel == "self" else "self"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ("gui", "cli"))
async def test_specialized_target_entrypoints_close_test_before_cutover(
    audio_application, entrypoint
):
    app, control, audio, sources, _path = audio_application
    target = app._peer_capture.peer.target
    target.devices = SimpleNamespace(names=lambda: ("Controlled Speakers",))
    target.processes = SimpleNamespace(candidates=lambda: ())
    assert await app.start_audio_test()
    assert await app.wait_audio_test_ready()
    if entrypoint == "gui":
        await app.apply_loopback_capture_option("device:Controlled Speakers")
    else:
        receipt = await control.submit(
            "audio.target.set", {"value": "device:Controlled Speakers"}, request_id="target-cutover"
        )
        assert (await control.wait(receipt["operation_id"], timeout=3))["status"] == "applied"
    assert not audio.active
    assert all(source.closed for source in sources)
    assert target.current_value() == "device:Controlled Speakers"


class _ObservedOrderingLock(asyncio.Lock):
    def __init__(self):
        super().__init__()
        self.waiting = asyncio.Event()

    async def acquire(self):
        if self.locked():
            self.waiting.set()
        return await super().acquire()


@pytest.mark.asyncio
@pytest.mark.parametrize("lock_name", ("_resource_lock", "_lock"))
@pytest.mark.parametrize("existing_owner", (False, True))
@pytest.mark.parametrize("entrypoint", ("gui", "control"))
async def test_preowner_ordering_start_is_revoked_by_completed_stop_and_recovers(
    audio_application, lock_name, existing_owner, entrypoint
):
    from tests.ui.test_app_branches import _audio_test_app

    app, control, audio, sources, _path = audio_application
    original_owner = audio.owner() if existing_owner else None
    gate = _ObservedOrderingLock()
    setattr(control, lock_name, gate)
    await gate.acquire()
    gui = None
    starting = None
    on_receipt = None
    try:
        if entrypoint == "gui":
            gui = _audio_test_app(SimpleNamespace())
            gui._ui_application = app
            gui._on_start_audio_test()
            starting = asyncio.create_task(gui.page.tasks[0]())
        else:
            on_receipt = await control.submit(
                "audio.test", {"enabled": True}, request_id="preowner-on"
            )
        await asyncio.wait_for(gate.waiting.wait(), timeout=2)
        assert sources == []
        assert audio.owner_if_created is original_owner
        if gui is not None:
            gui._audio_test_dialog._handle_dismiss(None)
            await asyncio.wait_for(gui.page.tasks[-1](), timeout=2)
            assert not gui._audio_test_dialog.is_open
        else:
            off_receipt = await control.submit(
                "audio.test", {"enabled": False}, request_id="preowner-off"
            )
            stopped = await control.wait(off_receipt["operation_id"], timeout=2)
            assert stopped["terminal"]
            assert stopped["status"] == "applied"
            assert stopped["audio_test"]["state"] == "off"
        assert audio.snapshot["state"] == "off"
        assert audio.owner_if_created is original_owner
        gate.release()
        if starting is not None:
            await asyncio.wait_for(starting, timeout=2)
        else:
            cancelled = await control.wait(on_receipt["operation_id"], timeout=2)
            assert cancelled["terminal"]
            assert cancelled["status"] == "cancelled"
            assert cancelled["audio_test"]["state"] == "off"
        assert audio.owner_if_created is original_owner
        assert not audio.active
        assert sources == []
        if gui is not None:
            gui._on_start_audio_test()
            await asyncio.wait_for(gui.page.tasks[-1](), timeout=2)
            assert await app.wait_audio_test_ready()
            gui._audio_test_dialog._handle_dismiss(None)
            await asyncio.wait_for(gui.page.tasks[-1](), timeout=2)
        else:
            valid = await control.submit(
                "audio.test", {"enabled": True}, request_id="later-valid-on"
            )
            ready = await control.wait(valid["operation_id"], timeout=2)
            assert ready["terminal"] and ready["status"] == "applied"
            replay = await control.submit("audio.test", {"enabled": True}, request_id="preowner-on")
            assert replay["operation_id"] == on_receipt["operation_id"]
            assert replay["status"] == "cancelled"
            off = await control.submit(
                "audio.test", {"enabled": False}, request_id="later-valid-off"
            )
            stopped = await control.wait(off["operation_id"], timeout=2)
            assert stopped["terminal"] and stopped["status"] == "applied"
        assert len(sources) == 2
        assert all(source.closed for source in sources)
        assert audio.snapshot["state"] == "off"
    finally:
        if gate.locked():
            gate.release()
        if starting is not None:
            await asyncio.gather(starting, return_exceptions=True)
        await app.stop_audio_test()


@pytest.mark.asyncio
async def test_control_start_keeps_submission_authority_before_its_task_runs(audio_application):
    app, control, audio, sources, _path = audio_application
    on = await control.submit("audio.test", {"enabled": True}, request_id="unexecuted-on")
    await app.stop_audio_test()
    cancelled = await control.wait(on["operation_id"], timeout=2)
    assert cancelled["terminal"] and cancelled["status"] == "cancelled"
    assert audio.owner_if_created is None
    assert sources == []
    valid = await control.submit("audio.test", {"enabled": True}, request_id="unexecuted-recovery")
    ready = await control.wait(valid["operation_id"], timeout=2)
    assert ready["terminal"] and ready["status"] == "applied"
    await app.stop_audio_test()
    assert len(sources) == 2 and all(source.closed for source in sources)


@pytest.mark.asyncio
async def test_gui_start_waiting_in_settings_queue_is_revoked_before_dialog_open(audio_application):
    from tests.ui.test_app_branches import _audio_test_app

    app, control, audio, sources, _path = audio_application
    gui = _audio_test_app(SimpleNamespace())
    gui._ui_application = app
    entered = asyncio.Event()
    release = asyncio.Event()

    async def earlier_settings():
        entered.set()
        await release.wait()

    gui._queue_settings_mutation_task(earlier_settings)
    worker = asyncio.create_task(gui.page.tasks[0]())
    try:
        await entered.wait()
        gui._on_start_audio_test()
        off = await control.submit("audio.test", {"enabled": False}, request_id="queued-gui-off")
        stopped = await control.wait(off["operation_id"], timeout=2)
        assert stopped["terminal"] and stopped["status"] == "applied"
        release.set()
        await asyncio.wait_for(worker, timeout=2)
        assert getattr(gui, "_audio_test_dialog", None) is None
        assert audio.owner_if_created is None
        assert sources == []
        gui._on_start_audio_test()
        await asyncio.wait_for(gui.page.tasks[-1](), timeout=2)
        assert await app.wait_audio_test_ready()
        gui._audio_test_dialog._handle_dismiss(None)
        await asyncio.wait_for(gui.page.tasks[-1](), timeout=2)
        assert len(sources) == 2 and all(source.closed for source in sources)
    finally:
        release.set()
        await asyncio.gather(worker, return_exceptions=True)
        await app.stop_audio_test()
