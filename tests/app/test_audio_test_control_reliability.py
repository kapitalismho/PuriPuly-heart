from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from puripuly_heart.app.ports.audio_test import AudioDiagnosticCandidate
from puripuly_heart.app.ports.settings_view import SelfVadSettingsIntent
from puripuly_heart.app.services import audio_test as audio_test_service
from puripuly_heart.app.services.agent_chat.context import AgentActivity, collect_agent_context
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.app.services.agent_chat.tools import AgentTools
from puripuly_heart.app.services.recognition_probe import load_recognition_probe_audio
from puripuly_heart.app.wiring import wiring_audio_test
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_self_capture_session_config_from_vnext,
)
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.headless_application import compose_headless_application
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.audio import source as audio_source
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmission,
    PeerCaptureAdmissionStatus,
    PeerCaptureResolvedTarget,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.core.self_capture import SelfCaptureAdmission, SelfCaptureAdmissionStatus
from tests.app.test_agent_chat import Responses, answer, context_of, tool
from tests.app.test_audio_test_session_owner import CaptureControl, FeedSource, LoudnessEngine
from tests.app.test_audio_test_sources import Resolver, _route
from tests.core.runtime.test_self_capture_session import build_owner


@pytest.fixture
async def audio_application(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.upper().endswith("_API_KEY"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PURIPULY_AGENT_RELAY_URL", "")
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
            integrated_context=replace(initial.intent.integrated_context, enabled=False),
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


@pytest.fixture
def diagnostic_audio_application(audio_application, monkeypatch):
    monkeypatch.setattr(audio_test_service, "AUDIO_DIAGNOSE_CANDIDATE_TIMEOUT_S", 0.025)
    monkeypatch.setattr(
        wiring_audio_test,
        "enumerate_microphone_diagnostic_candidates",
        lambda **_kwargs: (
            AudioDiagnosticCandidate(
                input_host_api="Windows WASAPI",
                input_device="Desk Mic",
                device_index=7,
                max_input_channels=2,
                current_route=True,
            ),
        ),
    )
    return audio_application


@pytest.mark.asyncio
async def test_control_diagnose_is_bounded_local_and_retains_safe_measured_history(
    diagnostic_audio_application,
):
    app, control, audio, sources, path = diagnostic_audio_application
    saved = path.read_bytes()

    def reject_loopback(*_args, **_kwargs):
        pytest.fail("Microphone diagnosis must not open loopback")

    audio.loopback_source_factory = reject_loopback
    capability = control.capabilities()["commands"]["audio.diagnose"]
    assert capability["arguments"] == {} and capability["cancellable"] is True
    admitted = await control.submit("audio.diagnose", {}, request_id="diagnose-local")
    result = await control.wait(admitted["operation_id"], timeout=3)

    assert result["terminal"] and result["status"] == "applied"
    assert result["recognition"] == "not_exercised" and result["settings_changed"] is False
    assert result["diagnosis"]["cleanup_complete"] and not audio.active
    candidate = result["diagnosis"]["candidates"][0]
    assert candidate["frame_count"] == 1 and candidate["speech_detected"]
    assert candidate["failure_code"] is None
    assert len(sources) == 1 and sources[0].closed
    assert path.read_bytes() == saved
    history = await control.query("diagnostics.history", {"operation_id": result["operation_id"]})
    records = history["records"]
    observations = [item for item in records if item["code"] == "audio_diagnose_candidate"]
    assert len(observations) == 1
    measured = observations[0]
    assert measured["component"] == "audio_diagnosis"
    assert measured["operation"] == "audio.diagnose"
    assert measured["fields"]["frame_count"] == 1
    assert measured["fields"]["speech_probability_min"] == pytest.approx(0.9)
    assert measured["fields"]["speech_probability_max"] == pytest.approx(0.9)
    assert measured["fields"]["speech_threshold"] == pytest.approx(
        app.compatibility_settings().intent.stt.vad_speech_threshold
    )
    assert measured["fields"]["cleanup_complete"] is True
    assert measured["fields"]["audio_failure_code"] == "none"
    assert "observed_speech_probabilities" not in measured["fields"]
    assert "Desk Mic" not in json.dumps(records)
    assert any(item["code"] == "audio_diagnose_completed" for item in records)


@pytest.mark.asyncio
async def test_control_diagnose_rejects_arguments_without_opening_routes(
    diagnostic_audio_application,
):
    _app, control, _audio, sources, _path = diagnostic_audio_application
    admitted = await control.submit(
        "audio.diagnose", {"device": "unrequested-route"}, request_id="diagnose-arguments"
    )
    result = await control.wait(admitted["operation_id"], timeout=3)
    assert result["terminal"] and result["status"] == "rejected"
    assert result["error"]["code"] == "invalid_arguments"
    assert sources == []


@pytest.mark.asyncio
async def test_control_operation_cancel_releases_diagnostic_capture_before_terminal_receipt(
    diagnostic_audio_application, monkeypatch
):
    _app, control, audio, sources, path = diagnostic_audio_application
    monkeypatch.setattr(audio_test_service, "AUDIO_DIAGNOSE_CANDIDATE_TIMEOUT_S", 10.0)
    saved = path.read_bytes()
    admitted = await control.submit("audio.diagnose", {}, request_id="diagnose-cancel")
    for _ in range(500):
        if sources and audio.owner().runtime_for("self").source is sources[0]:
            break
        await asyncio.sleep(0.001)
    else:
        pytest.fail("Controlled diagnosis did not acquire its microphone")
    result = await control.cancel(admitted["operation_id"])
    assert result["terminal"] and result["status"] == "cancelled"
    assert result["diagnosis"]["state"] == "cancelled"
    assert result["diagnosis"]["cleanup_complete"]
    assert all(source.closed for source in sources) and not audio.active
    assert path.read_bytes() == saved


@pytest.mark.asyncio
async def test_control_cancel_before_audio_admission_returns_safe_empty_receipt(
    diagnostic_audio_application,
):
    _app, control, audio, sources, _path = diagnostic_audio_application
    await control._resource_lock.acquire()
    try:
        admitted = await control.submit("audio.diagnose", {}, request_id="diagnose-queued-cancel")
        await asyncio.sleep(0)
        result = await control.cancel(admitted["operation_id"])
    finally:
        control._resource_lock.release()
    assert result["terminal"] and result["status"] == "cancelled"
    assert result["diagnosis"]["candidates"] == []
    assert result["diagnosis"]["cleanup_complete"]
    assert result["diagnosis"]["restoration"] == {"self": "not_needed", "peer": "not_needed"}
    assert audio.owner_if_created is None and sources == []


@pytest.mark.asyncio
async def test_explicit_test_stop_cancels_diagnosis_through_shared_owner(
    diagnostic_audio_application, monkeypatch
):
    _app, control, audio, sources, _path = diagnostic_audio_application
    monkeypatch.setattr(audio_test_service, "AUDIO_DIAGNOSE_CANDIDATE_TIMEOUT_S", 10.0)
    admitted = await control.submit("audio.diagnose", {}, request_id="diagnose-explicit-stop")
    for _ in range(500):
        if sources and audio.owner().runtime_for("self").source is sources[0]:
            break
        await asyncio.sleep(0.001)
    else:
        pytest.fail("Controlled diagnosis did not acquire its microphone")
    off = await control.submit("audio.test", {"enabled": False}, request_id="stop-diagnosis")
    stopped = await control.wait(off["operation_id"], timeout=3)
    result = await control.wait(admitted["operation_id"], timeout=3)
    assert stopped["terminal"] and stopped["status"] == "applied"
    assert result["terminal"] and result["status"] == "cancelled"
    assert result["diagnosis"]["cleanup_complete"] and not audio.active
    assert all(source.closed for source in sources)


@pytest.mark.asyncio
async def test_enumeration_failure_is_terminal_sanitized_and_does_not_pause_capture(
    diagnostic_audio_application, monkeypatch
):
    _app, control, audio, sources, _path = diagnostic_audio_application
    capture = CaptureControl(self_enabled=True, peer_enabled=True)
    audio.capture_control = capture

    def fail(**_kwargs):
        raise PermissionError("private microphone path token=private-credential")

    monkeypatch.setattr(wiring_audio_test, "enumerate_microphone_diagnostic_candidates", fail)
    admitted = await control.submit("audio.diagnose", {}, request_id="diagnose-enumeration")
    result = await control.wait(admitted["operation_id"], timeout=3)
    assert result["terminal"] and result["status"] == "action_required"
    assert result["diagnosis"]["failure_code"] == "enumeration_failed"
    assert result["diagnosis"]["cleanup_complete"]
    assert sources == [] and capture.calls == []
    assert capture.enabled == {"self": True, "peer": True}
    assert "private" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
async def test_native_inventory_deadline_or_cancel_retains_readonly_cleanup_ownership(
    diagnostic_audio_application, monkeypatch, cancelled
):
    import threading

    _app, control, audio, sources, _path = diagnostic_audio_application
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    routes = wiring_audio_test.enumerate_microphone_diagnostic_candidates
    calls = []

    def enumerate_routes(**kwargs):
        calls.append(kwargs)
        loop.call_soon_threadsafe(entered.set)
        release.wait(timeout=3)
        return routes(**kwargs)

    monkeypatch.setattr(
        wiring_audio_test, "DIAGNOSTIC_ENUMERATION_TIMEOUT_S", 1.0 if cancelled else 0.025
    )
    monkeypatch.setattr(
        wiring_audio_test, "enumerate_microphone_diagnostic_candidates", enumerate_routes
    )
    try:
        admitted = await control.submit("audio.diagnose", {}, request_id="diagnose-native-query")
        await entered.wait()
        result = (
            await control.cancel(admitted["operation_id"])
            if cancelled
            else await control.wait(admitted["operation_id"], timeout=2)
        )
        assert result["terminal"]
        assert result["status"] == ("cancelled" if cancelled else "failed")
        assert result["diagnosis"]["cleanup_complete"] is False
        assert result["diagnosis"]["candidates"] == []
        assert result["diagnosis"]["restoration"] == {"self": "not_needed", "peer": "not_needed"}
        assert audio._diagnose_enumeration_task is not None
        assert not audio._diagnose_enumeration_task.done()
        retry = await control.submit("audio.diagnose", {}, request_id="diagnose-native-query-retry")
        repeated = await control.wait(retry["operation_id"], timeout=2)
        assert repeated["terminal"] and repeated["status"] == "failed"
        assert len(calls) == 1
        assert sources == []
    finally:
        release.set()
        for _ in range(500):
            if audio._diagnose_enumeration_task is None:
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("Controlled native inventory did not release its owned task")


@pytest.mark.asyncio
async def test_test_mode_diagnosis_logs_safe_stage_metadata_not_exception_prose(
    diagnostic_audio_application,
):
    _app, control, audio, sources, _path = diagnostic_audio_application
    logs = []
    errors = []
    audio.diagnostic_sink = lambda message, level, exception: logs.append(
        (message, level, exception)
    )
    audio.error_sink = errors.append

    def fail_vad():
        raise ValueError("private audio model path token=private-credential")

    audio.vad_engine_factory = fail_vad
    admitted = await control.submit("audio.diagnose", {}, request_id="diagnose-safe-vad-failure")
    result = await control.wait(admitted["operation_id"], timeout=3)
    assert result["terminal"] and result["status"] == "action_required"
    assert result["diagnosis"]["candidates"][0]["failure_code"] == "vad_unavailable"
    assert sources == [] and errors == []
    assert any("phase=begin" in message for message, _level, _exception in logs)
    assert any("exception_class=ValueError" in message for message, _level, _exception in logs)
    assert all(exception is None for _message, _level, exception in logs)
    assert "private" not in repr(logs) + json.dumps(result)


@pytest.mark.asyncio
async def test_diagnose_microphone_tool_uses_production_source_and_silero_with_synthetic_audio(
    audio_application, monkeypatch, record_property
):
    _app, control, audio, sources, path = audio_application
    saved = path.read_bytes()
    host_apis = [{"name": "Windows WASAPI", "default_input_device": 0}]
    names = (
        "Synthetic current microphone",
        "Synthetic unrelated microphone",
        "Speakers (Virtual Desktop Audio)",
        "Steam Streaming Speakers",
        "Microphone (Virtual Desktop Audio)",
        "Steam Streaming Microphone",
        "Microphone (PicoStreamingMicrophone)",
        "Headset Microphone (Oculus Virtual Audio Device)",
    )
    devices = [
        {
            "name": name,
            "hostapi": 0,
            "max_input_channels": 0 if index == 2 else 2,
            "default_samplerate": 22050.0,
        }
        for index, name in enumerate(names)
    ]
    speech = np.frombuffer(load_recognition_probe_audio(22050), dtype="<i2").astype(np.float32)
    speech /= 32768.0
    streams = []

    def query_devices(device=None, kind=None):
        if device is not None:
            return devices[device]
        return devices[0] if kind == "input" else devices

    class SyntheticInputStream:
        def __init__(self, *, device, channels, callback, **_kwargs):
            assert all(stream.closed for stream in streams)
            self.device = device
            self.channels = channels
            self.callback = callback
            self.samplerate = 22050
            self.closed = False
            streams.append(self)

        def start(self):
            if self.device == 0:
                raise OSError("controlled synthetic stream open failure")
            if self.device == 6:
                return
            assert self.device in {4, 5}
            samples = np.zeros_like(speech) if self.device == 4 else speech
            for offset in range(0, samples.size, 1024):
                block = np.repeat(samples[offset : offset + 1024, None], self.channels, axis=1)
                self.callback(block, len(block), None, False)

        def stop(self):
            pass

        def close(self):
            self.closed = True

    class RestoringCaptureControl(CaptureControl):
        async def set_enabled(self, channel, enabled):
            if enabled:
                assert streams and all(stream.closed for stream in streams)
            await super().set_enabled(channel, enabled)

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_hostapis=lambda index=None: host_apis if index is None else host_apis[index],
            query_devices=query_devices,
            default=SimpleNamespace(device=(0, -1)),
            WasapiSettings=lambda **values: SimpleNamespace(**values),
            InputStream=SyntheticInputStream,
        ),
    )
    monkeypatch.setattr(
        wiring_audio_test,
        "observe_microphone_test_route",
        audio_source.observe_microphone_test_route,
    )
    monkeypatch.setattr(
        wiring_audio_test,
        "determine_self_mic_capture_channels",
        audio_source.determine_self_mic_capture_channels,
    )
    audio.microphone_source_factory = audio_source.SoundDeviceAudioSource
    audio.vad_engine_factory = wiring_audio_test.create_audio_test_vad_engine
    capture = RestoringCaptureControl(self_enabled=True, peer_enabled=True)
    audio.capture_control = capture

    def reject_loopback(*_args, **_kwargs):
        pytest.fail("Microphone diagnosis must not open playback or loopback capture")

    audio.loopback_source_factory = reject_loopback
    tools = AgentTools(
        control,
        changed=lambda: None,
        record_action=lambda _action: None,
        emit_event=lambda _event: True,
    )
    try:
        result = await tools.execute("probe_app", {"target": "microphone"})
    finally:
        await tools.close()
    assert result["status"] == "applied"
    receipt = result["receipt"]
    assert receipt["recognition"] == "not_exercised" and receipt["settings_changed"] is False
    diagnosis = receipt["diagnosis"]
    assert diagnosis["state"] == "completed" and diagnosis["cleanup_complete"]
    assert [stream.device for stream in streams] == [0, 4, 5, 6]
    failed, silent, spoken, empty = diagnosis["candidates"]
    assert failed["failure_code"] == "source_open_failed" and not failed["opened"]
    assert silent["opened"] and silent["frame_count"] > 0 and silent["silence_observed"]
    assert silent["vad_sample_count"] > 0 and not silent["speech_detected"]
    assert silent["failure_code"] is None
    assert spoken["frame_count"] > 0 and spoken["vad_sample_count"] > 0
    assert (
        spoken["speech_detected"] and spoken["speech_probability_max"] >= spoken["speech_threshold"]
    )
    assert spoken["failure_code"] is None
    assert empty["opened"] and empty["frame_count"] == 0
    assert empty["failure_code"] == "candidate_timeout"
    assert diagnosis["selected_candidate_id"] == "input-5"
    assert diagnosis["selection_basis"] == "speech"
    assert diagnosis["restoration"] == {"self": "restored", "peer": "restored"}
    assert capture.calls == [("self", False), ("peer", False), ("self", True), ("peer", True)]
    assert capture.enabled == {"self": True, "peer": True}
    assert all(stream.closed for stream in streams) and not audio.active
    assert path.read_bytes() == saved and sources == []
    history = await control.query("diagnostics.history", {"operation_id": receipt["operation_id"]})
    assert all(name not in json.dumps(history) for name in names)
    record_property("synthetic_audio_diagnosis", json.dumps(diagnosis, sort_keys=True))


@pytest.mark.asyncio
async def test_agent_diagnosis_does_not_apply_route_until_separate_live_capture_approval(
    diagnostic_audio_application, monkeypatch
):
    app, control, audio, sources, path = diagnostic_audio_application
    monkeypatch.setattr(
        "puripuly_heart.app.services.application_audio_devices.enumerate_audio_devices",
        lambda _settings: {
            "host_apis": ["Windows WASAPI"],
            "microphones": [{"name": "Desk Mic", "host_api": "Windows WASAPI"}],
            "loopback_outputs": [],
            "errors": {},
        },
    )
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
    await app.set_stt_enabled(True)
    assert owner.snapshot.effective_active
    before = await control.query("settings.current", {})
    persisted = path.read_bytes()
    actions = []
    tools = AgentTools(
        control,
        changed=lambda: None,
        record_action=actions.append,
        emit_event=lambda _event: False,
    )
    try:
        diagnosed = await tools.execute("probe_app", {"target": "microphone"})
        receipt = diagnosed["receipt"]
        assert receipt["terminal"] and diagnosed["status"] == "applied"
        assert receipt["recognition"] == "not_exercised"
        assert receipt["settings_changed"] is False
        assert receipt["diagnosis"]["selected_candidate_id"] == "input-7"
        assert receipt["diagnosis"]["restoration"]["self"] == "restored"
        assert owner.snapshot.effective_active
        restored_source = owner.source
        restored_generation = owner.snapshot.generation
        assert len(sources) == 1 and sources[0].closed and not audio.active
        assert await control.query("settings.current", {}) == before
        assert path.read_bytes() == persisted
        context = await collect_agent_context(control, activity=AgentActivity(), topic="audio")
        tools.observe_context(context)
        selected = receipt["diagnosis"]["candidates"][0]
        changes = {
            "audio.input_host_api": selected["input_host_api"],
            "audio.input_device": selected["input_device"],
        }
        proposed = await tools.execute(
            "manage_proposal", {"action": "settings", "changes": changes}
        )
        assert proposed["status"] == "needs_confirmation"
        assert await control.query("settings.current", {}) == before
        assert path.read_bytes() == persisted
        declined = await tools.confirm(proposed["confirmation"]["id"], False)
        assert declined["status"] == "declined"
        assert owner.snapshot.effective_active
        assert owner.source is restored_source
        assert owner.snapshot.generation == restored_generation
        assert path.read_bytes() == persisted
        proposed = await tools.execute(
            "manage_proposal", {"action": "settings", "changes": changes}
        )
        applied = await tools.confirm(proposed["confirmation"]["id"], True)
        assert applied["status"] == "applied" and applied["receipt"]["terminal"]
        current = await control.query("settings.current", {})
        assert current["revision"] > before["revision"]
        assert app.compatibility_settings().intent.audio.input_device == "Desk Mic"
        assert app.compatibility_settings().intent.audio.input_host_api == "Windows WASAPI"
        saved_audio = json.loads(path.read_text(encoding="utf-8"))["intent"]["audio"]
        assert saved_audio["input_device"] == "Desk Mic"
        assert saved_audio["input_host_api"] == "Windows WASAPI"
        assert owner.snapshot.effective_active and owner.source is not None
        assert owner.source is not restored_source
        assert restored_source.close_calls == 1
        assert owner.snapshot.generation > restored_generation
        assert [action["tool"] for action in actions if action["phase"] == "completed"] == [
            "audio.diagnose",
            "settings.apply",
        ]
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_paid_route_requires_current_user_card_before_real_settings_persist(
    audio_application, monkeypatch
):
    app, control, _audio, sources, path = audio_application
    monkeypatch.setattr(
        "puripuly_heart.app.services.application_audio_devices.enumerate_audio_devices",
        lambda _settings: {
            "host_apis": [],
            "microphones": [],
            "loopback_outputs": [],
            "errors": {},
        },
    )
    before = await control.query("settings.current", {})
    persisted = path.read_bytes()
    tools = AgentTools(
        control,
        changed=lambda: None,
        record_action=lambda _action: None,
        emit_event=lambda _event: False,
    )
    try:
        context = await collect_agent_context(
            control, activity=AgentActivity(), topic="translation"
        )
        tools.observe_context(context)
        changes = {"translation.model": "gpt_6_luna", "translation.connection": "official_byok"}
        proposed = await tools.execute(
            "manage_proposal", {"action": "settings", "changes": changes}
        )
        card = proposed["confirmation"]
        assert proposed["status"] == "needs_confirmation"
        assert card["route_effects"][0]["cost"] == "provider_billed"
        assert card["route_effects"][0]["credential_readiness"] == "absent"
        assert (await tools.confirm("not-the-current-card", True))["status"] == "rejected"
        assert (
            await tools.execute(
                "manage_proposal", {"action": "settings", "changes": changes, "approved": True}
            )
        )["status"] == "rejected"
        assert await control.query("settings.current", {}) == before
        assert path.read_bytes() == persisted
        assert (await tools.confirm(card["id"], False))["status"] == "declined"
        assert (await tools.confirm(card["id"], True))["status"] == "rejected"
        assert path.read_bytes() == persisted
        proposed = await tools.execute(
            "manage_proposal", {"action": "settings", "changes": changes}
        )
        approved = await tools.confirm(proposed["confirmation"]["id"], True)
        assert approved["receipt"]["terminal"]
        assert approved["status"] == "degraded"
        assert approved["route_effects"][0]["credential_readiness"] == "absent"
        saved = json.loads(path.read_text(encoding="utf-8"))["intent"]["translation"]
        assert saved["model"] == "gpt_6_luna" and saved["connection"] == "official_byok"
        assert app.compatibility_settings().intent.translation.model == "gpt_6_luna"
        status = await control.query("app.status", {})
        assert status["translation_enabled"] is False
        assert not status["channels"]["self"]["effective_active"]
        assert not status["channels"]["peer"]["effective_active"]
        assert sources == []
    finally:
        await tools.close()


@pytest.mark.asyncio
async def test_agent_readonly_context_observes_real_encrypted_store_without_releasing_secrets(
    audio_application, monkeypatch
):
    _app, control, _audio, sources, path = audio_application
    monkeypatch.setattr(
        "puripuly_heart.app.services.application_audio_devices.enumerate_audio_devices",
        lambda _settings: {
            "host_apis": ["Windows WASAPI"],
            "microphones": [{"name": "Synthetic input", "host_api": "Windows WASAPI"}],
            "loopback_outputs": [],
            "errors": {},
        },
    )
    admitted = await control.submit(
        "secrets.set",
        {"name": "openai_api_key", "value": "PRIVATE-SYNTHETIC-KEY"},
        request_id="synthetic-readonly-context",
    )
    stored = await control.wait(admitted["operation_id"])
    assert stored["status"] == "applied"
    secrets_path = path.parent / "secrets.json"
    saved = path.read_bytes(), secrets_path.read_bytes()
    revision = (await control.query("settings.current", {}))["revision"]
    events = tuple(control.events.history)
    transport = Responses(
        {"status": "completed", "output": [tool("audio", arguments='{"topic":"audio"}')]},
        {
            "status": "completed",
            "output": [tool("translation", arguments='{"topic":"translation"}')],
        },
        {"status": "completed", "output": [tool("accounts", arguments='{"topic":"accounts"}')]},
        answer("Read-only application evidence."),
    )
    owner = AgentChatOwner(control, transport, localize=str)
    await owner.start()
    try:
        result = await owner.send("Inspect the synthetic audio and account state without changes.")
        assert result.status == "completed"
        context = context_of(transport.requests[-1])
        assert context["translation"]["keys"]["openai_api_key"] is True
        assert context["audio"]["devices"]["microphones"][0]["input_device"] == "Synthetic input"
        assert context["accounts"]["chatgpt"]["signed_in"] is not True
        assert "PRIVATE-SYNTHETIC-KEY" not in json.dumps(transport.requests)
        assert (path.read_bytes(), secrets_path.read_bytes()) == saved
        assert (await control.query("settings.current", {}))["revision"] == revision
        assert tuple(control.events.history) == events
        assert (await control.query("app.status", {}))["translation_enabled"] is False
        assert sources == []
    finally:
        await owner.close()
