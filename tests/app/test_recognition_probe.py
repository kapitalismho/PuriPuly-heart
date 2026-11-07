from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

import puripuly_heart.app.wiring.wiring_local_asr_provider_runtime as runtime_wiring
from puripuly_heart.app.services.agent_chat.tools import AgentTools
from puripuly_heart.app.services.application_control import ApplicationControlOwner
from puripuly_heart.app.services.application_control_events import ControlEvents
from puripuly_heart.app.services.recognition_probe import load_recognition_probe_audio
from puripuly_heart.app.services.settings.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import SharedSTTProviderFactory
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.local_asr.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.runtime.local_asr_provider_runtime import LocalASRProviderRuntimeOwner
from puripuly_heart.providers.stt.local_session import LocalOfflineSTTBackend
from puripuly_heart.runtime_layout import current_runtime_layout
from tests.core.runtime.test_local_asr_provider_runtime import FakeGpuRuntime, _resolved_config
from tests.providers.test_local_cpu_backends import _scoped_request


class ControlledLocalBackend(LocalOfflineSTTBackend):
    def __init__(self, root, channel, *, text="synthetic recognition", failure=None, hold=False):
        super().__init__(root, "controlled", "local_fun_asr_mlt_nano", stream_label=channel)
        self.text = text
        self.failure = failure
        self.hold = hold
        self.entered = threading.Event()
        self.release = threading.Event()
        self.decoded = []
        self.loads = 0
        self.closes = 0

    def _validate_runtime_assets(self):
        return None

    def _create_recognizer(self):
        self.loads += 1
        return object()

    async def _close_resource(self, resource):
        self.closes += 1

    def _decode_f32_sync(self, recognizer, samples):
        self.decoded.append(samples.copy())
        self.entered.set()
        if self.hold:
            assert self.release.wait(5)
        if self.failure is not None:
            raise self.failure
        return self.text


def request(channel, provider="local_fun_asr_mlt_nano"):
    config = _resolved_config(channel, provider)
    return ProviderRuntimeBuildRequest(
        config,
        provider_signature=(provider,),
        runtime_signature=(config.source_language,),
        recognition_projection="scoped",
    )


class ProbeRuntime:
    def __init__(self, root, monkeypatch, backend_factory=None):
        self.backends = []
        self.normal_events = []
        self.requests = {channel: request(channel) for channel in ("self", "peer")}
        self.construction = []

        def create(config, **options):
            self.construction.append(config)
            backend = (
                backend_factory(config)
                if backend_factory
                else ControlledLocalBackend(root, config.channel)
            )
            self.backends.append(backend)
            return backend

        monkeypatch.setattr(runtime_wiring, "create_stt_backend_from_resolved_config", create)
        self.factory = SharedSTTProviderFactory(SimpleNamespace(), SystemClock(), 1, None)
        self.runtime = LocalASRProviderRuntimeOwner(
            provider_factory=self.factory,
            gpu_runtime_factory=FakeGpuRuntime,
            provisioning=SimpleNamespace(),
            self_event_handler=self.normal_event,
            peer_event_handler=self.normal_event,
        )
        self.control = ApplicationControlOwner(
            application=SimpleNamespace(compatibility_settings=lambda: None),
            settings=SimpleNamespace(),
            pipeline=SimpleNamespace(local_asr_runtime=self.runtime),
            results=SettingsTransactionResultOwner(),
            events=ControlEvents(instance_id="recognition-test"),
            provisioning=lambda: pytest.fail("probe accessed provisioning"),
            gpu=lambda: pytest.fail("probe accessed discovery"),
            peer=lambda: pytest.fail("probe accessed capture"),
            calibration=lambda: None,
            gemma=lambda: pytest.fail("probe accessed translation"),
            sync_ui=lambda: pytest.fail("probe changed presentation"),
            locale_choices=("en",),
            localize=str,
            gemma_selection=lambda _: pytest.fail("probe accessed translation"),
            instance_id="recognition-test",
            recognition_probe_request=lambda _, channel: self.requests[channel],
        )

    async def normal_event(self, event):
        self.normal_events.append(event)

    async def attach(self, channel):
        engine = await self.factory.create(
            self.requests[channel], gpu_runtime=self.runtime._gpu_runtime
        )
        await self.runtime.replace_prebuilt_provider(channel, engine, start=False)
        return engine

    async def submit(self, channel, extra=None):
        accepted = await self.control.submit(
            "recognition.probe", {"channel": channel, **(extra or {})}, request_id=uuid4().hex
        )
        assert accepted["status"] == "accepted" and not accepted["terminal"]
        return await self.control.wait(accepted["operation_id"], timeout=3)

    async def close(self):
        for backend in self.backends:
            if hasattr(backend, "release"):
                backend.release.set()
        self.control.freeze_ingress()
        await self.control.drain_operations()
        await self.runtime.close()


def assert_isolated(result):
    assert result["terminal"]
    for field in ("capture", "translation", "output_delivery", "display", "accuracy"):
        assert result[field] == "not_exercised"
    assert result["fixture_language"] == "ko"
    assert result["language_coverage"] == "korean_fixture_only"
    assert result["fixture_duration_seconds"] == pytest.approx(1.194376417)
    assert "synthetic recognition" not in json.dumps(result)
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("attached", [False, True])
async def test_configured_recognizer_runs_without_capture_and_preserves_live_ownership(
    tmp_path, monkeypatch, channel, attached
):
    fixture = ProbeRuntime(tmp_path, monkeypatch)
    engine = await fixture.attach(channel) if attached else None
    live = None
    try:
        if engine is not None:
            live = await engine.session_factory(engine.probe_settings, "local-epoch")
        result = await fixture.submit(channel)
        assert_isolated(result)
        assert result["status"] == "applied"
        assert result["recognition_result"] == "succeeded"
        assert result["final_event"] == "authoritative_turn_final"
        assert result["configured_language"] == ("ko" if channel == "self" else "en")
        assert result["source_mode"] == "manual"
        assert result["input_samples"] == len(load_recognition_probe_audio(16000)) // 2
        assert result["sample_rate_hz"] == 16000
        assert result["captured_generation"] == result["current_generation"]
        assert result["captured_settings_revision"] == result["current_settings_revision"] == 0
        assert fixture.normal_events == []
        assert [event["topic"] for event in fixture.control.events.history] == ["operation"]
        assert len(fixture.backends) == 1
        backend = fixture.backends[0]
        assert backend.loads == 1
        assert len(backend.decoded) == 1
        assert np.count_nonzero(backend.decoded[0]) > 1000
        assert backend.closes == (0 if attached else 1)
        assert fixture.runtime.current_provider(channel) is engine
        if live is not None:
            req = _scoped_request(channel=channel)
            await live.begin_turn(req)
            await live.send_turn_audio(
                req.identity,
                b"\x00\x20" * 800,
                payload_sequence=1,
                source_ranges=(),
                context_only=False,
            )
            await live.seal_turn(
                req.identity,
                sealed_content_ranges=(),
                seal_reason="source_eof",
                observed_trailing_silence_ms=None,
            )
            terminal = await asyncio.wait_for(anext(live.turn_events()), 1)
            assert terminal.outcome == "final" and terminal.text == backend.text
            assert backend.decoded[-1].shape == (800,)
            assert backend.closes == 0
    finally:
        if live is not None:
            await live.close()
        await fixture.close()


@pytest.mark.parametrize(
    "text,failure", [(" \n", None), ("unused", RuntimeError("PRIVATE-FAILURE"))]
)
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_empty_and_failed_recognition_are_safe_terminal_failures(
    tmp_path, monkeypatch, text, failure, channel
):
    fixture = ProbeRuntime(
        tmp_path,
        monkeypatch,
        lambda config: ControlledLocalBackend(tmp_path, config.channel, text=text, failure=failure),
    )
    try:
        result = await fixture.submit(channel)
        assert_isolated(result)
        assert result["status"] == "failed"
        assert result["recognition_result"] in {"empty", "failed"}
        assert result["final_event"] is None
        assert fixture.backends[0].closes == 1
        assert fixture.normal_events == []
    finally:
        await fixture.close()


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("change", ["settings", "provider", "runtime"])
async def test_changed_authority_cannot_report_obsolete_success(
    tmp_path, monkeypatch, attached, change
):
    fixture = ProbeRuntime(
        tmp_path,
        monkeypatch,
        lambda config: ControlledLocalBackend(tmp_path, config.channel, hold=True),
    )
    if attached:
        await fixture.attach("peer")
    accepted = await fixture.control.submit(
        "recognition.probe", {"channel": "peer"}, request_id=uuid4().hex
    )
    try:
        async with asyncio.timeout(3):
            while not fixture.backends:
                await asyncio.sleep(0)
        backend = fixture.backends[0]
        assert await asyncio.to_thread(backend.entered.wait, 2)
        replacing = None
        if change == "settings":
            fixture.control._revision += 1
            fixture.requests["peer"] = replace(
                fixture.requests["peer"], runtime_signature=("changed",)
            )
        elif change == "provider":
            replacement = await fixture.factory.create(
                fixture.requests["peer"], gpu_runtime=fixture.runtime._gpu_runtime
            )
            replacing = asyncio.create_task(
                fixture.runtime.replace_prebuilt_provider("peer", replacement, start=False)
            )
            await asyncio.sleep(0)
        else:
            fixture.control.pipeline.local_asr_runtime = None
        backend.release.set()
        result = await fixture.control.wait(accepted["operation_id"], timeout=3)
        if replacing is not None:
            await replacing
        assert_isolated(result)
        assert result["status"] == "failed" and result["recognition_result"] == "stale"
        assert result["error"]["code"] == "stale_provider_completion"
        assert fixture.normal_events == []
        if change == "settings":
            assert result["captured_settings_revision"] == 0
            assert result["current_settings_revision"] == 1
    finally:
        await fixture.close()


@pytest.mark.parametrize("shutdown", [False, True])
async def test_cancelled_native_boundary_drains_before_closing_probe_resources(
    tmp_path, monkeypatch, shutdown
):
    fixture = ProbeRuntime(
        tmp_path,
        monkeypatch,
        lambda config: ControlledLocalBackend(tmp_path, config.channel, hold=True),
    )
    accepted = await fixture.control.submit(
        "recognition.probe", {"channel": "self"}, request_id=uuid4().hex
    )
    try:
        async with asyncio.timeout(3):
            while not fixture.backends:
                await asyncio.sleep(0)
        backend = fixture.backends[0]
        assert await asyncio.to_thread(backend.entered.wait, 2)
        if shutdown:
            fixture.control.freeze_ingress()
            cancelling = asyncio.create_task(fixture.control.drain_operations())
        else:
            cancelling = asyncio.create_task(fixture.control.cancel(accepted["operation_id"]))
        await asyncio.sleep(0.02)
        assert not cancelling.done()
        assert backend.closes == 0
        backend.release.set()
        await cancelling
        result = await fixture.control.operation(accepted["operation_id"])
        assert_isolated(result)
        assert result["status"] == ("interrupted" if shutdown else "cancelled")
        assert result["recognition_result"] == result["status"]
        assert backend.closes == 1
        assert fixture.normal_events == []
    finally:
        await fixture.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_agent_tools_preserve_component_receipt_and_reject_model_input(
    tmp_path, monkeypatch, channel
):
    fixture = ProbeRuntime(tmp_path, monkeypatch)
    tools = AgentTools(
        fixture.control,
        changed=lambda: pytest.fail("probe changed settings"),
        record_action=lambda _: None,
        emit_event=lambda _: pytest.fail("probe emitted a notice"),
    )
    try:
        name = f"probe_{channel}_recognition"
        for arguments in (
            {"text": "PRIVATE"},
            {"audio": "PRIVATE"},
            {"path": "PRIVATE"},
            {"channel": channel},
        ):
            rejected = await tools.execute(name, arguments)
            assert rejected["status"] == "rejected"
        assert fixture.backends == []
        result = await tools.execute(name, {})
        assert result["status"] == "applied"
        assert_isolated(result["receipt"])
        assert result["receipt"]["channel"] == channel
        assert result["receipt"]["final_event"] == "authoritative_turn_final"
        assert result["receipt"]["recognition_result"] == "succeeded"
    finally:
        await tools.close()
        await fixture.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("interruption", ["stale", "cancelled"])
async def test_agent_probe_rejects_late_success_then_recovers_without_capture(
    tmp_path, monkeypatch, channel, interruption
):
    created = []

    def backend_factory(config):
        backend = ControlledLocalBackend(
            tmp_path, config.channel, hold=not created, text="PRIVATE-SYNTHETIC-RECOGNITION"
        )
        created.append(backend)
        return backend

    fixture = ProbeRuntime(tmp_path, monkeypatch, backend_factory)
    actions = []
    tools = AgentTools(
        fixture.control,
        changed=lambda: pytest.fail("probe changed settings presentation"),
        record_action=actions.append,
        emit_event=lambda _: pytest.fail("probe requested confirmation"),
    )
    task = asyncio.create_task(tools.execute(f"probe_{channel}_recognition", {}))
    cancelling = None
    try:
        async with asyncio.timeout(3):
            while not created:
                await asyncio.sleep(0)
        backend = created[0]
        assert await asyncio.to_thread(backend.entered.wait, 2)
        operation_id = actions[-1]["receipt"]["operation_id"]
        assert actions[-1]["phase"] == "submitted" and not task.done()
        if interruption == "stale":
            fixture.control._revision += 1
            fixture.requests[channel] = replace(
                fixture.requests[channel], runtime_signature=("changed-language",)
            )
        else:
            cancelling = asyncio.create_task(fixture.control.cancel(operation_id))
            await asyncio.sleep(0)
            assert not task.done() and backend.closes == 0
        backend.release.set()
        if cancelling is not None:
            await cancelling
        result = await asyncio.wait_for(task, 3)
        assert_isolated(result["receipt"])
        assert result["status"] == ("failed" if interruption == "stale" else "cancelled")
        assert result["receipt"]["recognition_result"] == interruption
        assert result["receipt"]["final_event"] == (
            "authoritative_turn_final" if interruption == "stale" else None
        )
        assert actions[-1]["phase"] == "completed" and actions[-1]["result"] == result
        assert backend.closes == 1
        assert fixture.runtime.current_provider(channel) is None
        recovered = await tools.execute(f"probe_{channel}_recognition", {})
        assert recovered["status"] == "applied"
        assert recovered["receipt"]["recognition_result"] == "succeeded"
        assert recovered["receipt"]["capture"] == "not_exercised"
        assert recovered["receipt"]["current_settings_revision"] == fixture.control._revision
        assert len(created) == 2 and all(item.closes == 1 for item in created)
        assert fixture.runtime.current_provider(channel) is None
        assert fixture.normal_events == []
        assert "PRIVATE-" not in json.dumps(actions)
        assert tools.confirmation is None
    finally:
        for backend in created:
            backend.release.set()
        if cancelling is not None:
            await cancelling
        await asyncio.gather(task, return_exceptions=True)
        await tools.close()
        await fixture.close()


@pytest.mark.parametrize(
    "arguments", [{}, {"channel": "both"}, {"channel": []}, {"channel": "peer", "path": "PRIVATE"}]
)
async def test_command_rejects_invalid_channel_and_external_audio(tmp_path, monkeypatch, arguments):
    fixture = ProbeRuntime(tmp_path, monkeypatch)
    try:
        admitted = await fixture.control.submit(
            "recognition.probe", arguments, request_id=uuid4().hex
        )
        result = await fixture.control.wait(admitted["operation_id"], timeout=1)
        assert result["status"] == "rejected" and result["terminal"]
        assert fixture.backends == []
        assert "PRIVATE" not in json.dumps(result)
    finally:
        await fixture.close()


def test_bundled_speech_is_single_known_asset_and_resolves_independent_of_cwd(
    tmp_path, monkeypatch
):
    resource = current_runtime_layout().package_resource("data", "diagnostics", "korean-eung.wav")
    assert (
        hashlib.sha256(resource.read_bytes()).hexdigest()
        == "fcbbaa5b5a548685cc6107a0b62ea43a238e0bb5570201237baaf0e9b0a03ffc"
    )
    expected = load_recognition_probe_audio(16000)
    monkeypatch.chdir(tmp_path)
    assert load_recognition_probe_audio(16000) == expected
    native_root = tmp_path / "installed"
    installed = native_root / "puripuly_heart" / "data" / "diagnostics" / "korean-eung.wav"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(resource.read_bytes())
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", str(native_root))
    assert load_recognition_probe_audio(16000) == expected


async def test_missing_native_assets_do_not_start_a_worker_or_report_success(tmp_path, monkeypatch):
    from puripuly_heart.core.local_asr.local_stt_assets import FUN_ASR_MLT_NANO_MODEL_ID
    from puripuly_heart.providers.stt.local_cpp import LocalCppSTTBackend
    from tests.providers.test_local_cpp_backend import WorkerFactory

    workers = WorkerFactory()
    fixture = ProbeRuntime(
        tmp_path,
        monkeypatch,
        lambda config: LocalCppSTTBackend(
            model_dir=tmp_path / "absent",
            model_id=FUN_ASR_MLT_NANO_MODEL_ID,
            provider_id="local_fun_asr_mlt_nano",
            stream_label=config.channel,
            process_factory=workers,
        ),
    )
    try:
        result = await fixture.submit("peer")
        assert_isolated(result)
        assert result["status"] == "failed" and result["recognition_result"] == "failed"
        assert workers.workers == []
        assert str(tmp_path) not in json.dumps(result)
        assert fixture.runtime.current_provider("peer") is None
    finally:
        await fixture.close()


async def test_missing_packaged_speech_is_a_prerequisite_not_recognition_success(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", str(tmp_path / "missing-payload"))
    fixture = ProbeRuntime(tmp_path, monkeypatch)
    try:
        result = await fixture.submit("self")
        assert_isolated(result)
        assert result["status"] == "action_required"
        assert result["recognition_result"] == "unavailable"
        assert result["error"]["code"] == "recognition_probe_asset_unavailable"
        assert fixture.backends[0].loads == 0
        assert fixture.backends[0].decoded == []
    finally:
        await fixture.close()


@pytest.mark.parametrize(
    "cleanup_failure", [RuntimeError("PRIVATE-CLEANUP"), asyncio.CancelledError()]
)
async def test_failed_probe_session_cleanup_stays_owned_until_backend_release(
    tmp_path, monkeypatch, cleanup_failure
):
    from puripuly_heart.providers.stt.local_session import LocalOfflineSTTSession

    fixture = ProbeRuntime(tmp_path, monkeypatch)
    engine = await fixture.attach("self")
    close = LocalOfflineSTTSession.close
    failed = []

    async def failing_once(session):
        if not failed:
            failed.append(session)
            raise cleanup_failure
        await close(session)

    monkeypatch.setattr(LocalOfflineSTTSession, "close", failing_once)
    try:
        result = await fixture.submit("self")
        assert_isolated(result)
        assert result["status"] == "failed" and result["recognition_result"] == "failed"
        assert engine.cleanup_debt == 1
        assert engine.probe_sessions == failed
        assert fixture.backends[0].closes == 0
        blocked = await fixture.submit("self")
        assert blocked["status"] == "action_required"
        assert blocked["recognition_result"] == "unavailable"
        assert len(fixture.backends[0].decoded) == 1
        await fixture.runtime.release_channel("self", mode="abort")
        assert engine.cleanup_debt == 0 and engine.probe_sessions == []
        assert fixture.backends[0].closes == 1
    finally:
        await fixture.close()


async def test_cpu_auto_probe_reuses_delegate_without_joining_live_handoff(tmp_path, monkeypatch):
    import puripuly_heart.providers.stt.local_cpu as cpu
    from tests.providers.test_local_cpu_backends import _ready_snapshot

    delegates = []

    def delegate_factory(model_id, **options):
        backend = ControlledLocalBackend(tmp_path, options["stream_label"])
        delegates.append(backend)
        return backend

    monkeypatch.setattr(
        cpu, "inspect_required_cpu_model_installs", lambda *args, **kwargs: _ready_snapshot()
    )
    fixture = ProbeRuntime(
        tmp_path,
        monkeypatch,
        lambda config: cpu.LocalCPUAutoSTTBackend(
            source_language=config.source_language,
            stream_label=config.channel,
            backend_factory=delegate_factory,
        ),
    )
    fixture.requests["self"] = request("self", "local_cpu_auto")
    engine = await fixture.attach("self")
    live = await engine.session_factory(engine.probe_settings, "live")
    try:
        result = await fixture.submit("self")
        assert result["recognition_result"] == "succeeded"
        assert len(delegates) == 1
        assert delegates[0].loads == 1 and len(delegates[0].decoded) == 1
        assert delegates[0].closes == 0
    finally:
        await live.close()
        await fixture.close()


async def test_cold_normal_provider_build_waits_for_probe_owned_model_release(
    tmp_path, monkeypatch
):
    fixture = ProbeRuntime(
        tmp_path,
        monkeypatch,
        lambda config: ControlledLocalBackend(tmp_path, config.channel, hold=True),
    )
    admitted = await fixture.control.submit(
        "recognition.probe", {"channel": "peer"}, request_id=uuid4().hex
    )
    building = None
    try:
        async with asyncio.timeout(3):
            while not fixture.backends:
                await asyncio.sleep(0)
        probe_backend = fixture.backends[0]
        assert await asyncio.to_thread(probe_backend.entered.wait, 2)
        building = asyncio.create_task(
            fixture.runtime.replace_provider(fixture.requests["peer"], start=False)
        )
        await asyncio.sleep(0.02)
        assert not building.done()
        assert len(fixture.backends) == 1 and probe_backend.closes == 0
        probe_backend.release.set()
        await building
        result = await fixture.control.wait(admitted["operation_id"], timeout=3)
        assert result["terminal"]
        assert result["recognition_result"] in {"succeeded", "stale"}
        assert probe_backend.closes == 1
        assert len(fixture.backends) == 2
        assert fixture.runtime.current_provider("peer") is not None
    finally:
        if building is not None:
            fixture.backends[0].release.set()
            await building
        await fixture.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_rolling_receipt_identifies_only_the_exercised_member(tmp_path, monkeypatch, channel):
    from puripuly_heart.config.provider_values import STTProviderName
    from puripuly_heart.core.stt.rolling import RollingProviderDefinition, RollingSTTBackend
    from puripuly_heart.providers.stt import gemini_transcribe as gemini
    from tests.core.test_recognition_probe_sessions import FinalLiveSession
    from tests.providers.test_gemini_transcribe_backend import (
        _RecordingFactory,
        _StubAsyncTransport,
        _StubClient,
        _StubSyncTransport,
    )

    def prepare(api_key, language_codes, custom_vocabulary):
        return gemini._GeminiClientResources(
            client=_StubClient(),
            sync_transport=_StubSyncTransport(),
            async_transport=_StubAsyncTransport(),
            config=gemini._build_live_config_sync(language_codes, custom_vocabulary),
        )

    monkeypatch.setattr(gemini, "_prepare_gemini_resources_sync", prepare)
    transport = FinalLiveSession("isolated final member result")
    member = gemini.GeminiTranscribeSTTBackend(
        "offline-controlled",
        live_connect_factory=_RecordingFactory(transport),
        drain_timeout_s=0.01,
    )
    rolling = RollingSTTBackend(
        (
            RollingProviderDefinition(
                STTProviderName.GEMINI_TRANSCRIBE, lambda: member, lambda: True
            ),
            RollingProviderDefinition(
                STTProviderName.DEEPGRAM,
                lambda: pytest.fail("recognition probe changed rolling member"),
                lambda: True,
            ),
        )
    )
    fixture = ProbeRuntime(tmp_path, monkeypatch, lambda config: rolling)
    fixture.requests[channel] = request(channel, "rolling_free")
    before = dict(rolling._states)
    try:
        result = await fixture.submit(channel)
        assert_isolated(result)
        assert result["status"] == "applied" and result["recognition_result"] == "succeeded"
        assert result["configured_provider"] == "rolling_free"
        assert result["exercised_provider"] == "gemini_transcribe"
        assert result["recognition_scope"] == "single_eligible_rolling_member"
        assert result["final_event"] == "final_recognition_unit"
        assert rolling._states == before and rolling._last_failures == {}
        assert transport.closed
    finally:
        await fixture.close()


class ControlledTeardownBackend(ControlledLocalBackend):
    def __init__(self, root, channel, *, fail_close=False, block_close=False):
        super().__init__(root, channel)
        self.fail_close = fail_close
        self.close_entered = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_completed = asyncio.Event()
        if not block_close:
            self.close_release.set()

    async def _close_resource(self, resource):
        self.close_entered.set()
        await self.close_release.wait()
        if self.fail_close:
            raise RuntimeError("PRIVATE-BACKEND-CLEANUP")
        await super()._close_resource(resource)
        self.close_completed.set()


@pytest.mark.parametrize("recovery", ["shutdown", "build"])
async def test_cold_backend_cleanup_failure_is_failed_owned_and_recoverable(
    tmp_path, monkeypatch, recovery
):
    overlap = []

    def create(config):
        overlap.append(any(backend.is_loaded for backend in fixture.backends))
        return ControlledTeardownBackend(tmp_path, config.channel, fail_close=not fixture.backends)

    fixture = ProbeRuntime(tmp_path, monkeypatch, create)
    try:
        result = await fixture.submit("peer")
        backend = fixture.backends[0]
        assert_isolated(result)
        assert result["status"] == "failed" and result["recognition_result"] == "failed"
        assert result["final_event"] == "authoritative_turn_final"
        assert len(backend.decoded) == 1 and backend.is_loaded
        assert fixture.runtime._probe_handles
        blocked = await fixture.submit("peer")
        assert blocked["status"] == "action_required"
        assert blocked["recognition_result"] == "unavailable"
        if recovery == "shutdown":
            with pytest.raises(ExceptionGroup):
                await fixture.runtime.close()
        else:
            failed = await fixture.runtime.replace_provider(fixture.requests["peer"], start=False)
            assert failed.status == "failed"
            assert fixture.runtime.current_provider("peer") is None
        assert backend.is_loaded and fixture.runtime._probe_handles
        assert overlap == [False]
        backend.fail_close = False
        if recovery == "shutdown":
            await fixture.runtime.close()
        else:
            applied = await fixture.runtime.replace_provider(fixture.requests["peer"], start=False)
            assert applied.status == "applied"
            assert overlap == [False, False]
        assert backend.close_completed.is_set() and not backend.is_loaded
        assert not fixture.runtime._probe_handles
        assert fixture.normal_events == []
    finally:
        for backend in fixture.backends:
            backend.fail_close = False
            backend.close_release.set()
        await fixture.close()


@pytest.mark.parametrize("recovery", ["build", "cancel_then_build", "cancel_then_shutdown"])
async def test_cold_pending_backend_close_fences_receipt_and_later_resource_admission(
    tmp_path, monkeypatch, recovery
):
    overlap = []

    def create(config):
        overlap.append(any(backend.is_loaded for backend in fixture.backends))
        return ControlledTeardownBackend(tmp_path, config.channel, block_close=not fixture.backends)

    fixture = ProbeRuntime(tmp_path, monkeypatch, create)
    fixture.requests["peer"] = replace(
        fixture.requests["peer"],
        config=replace(fixture.requests["peer"].config, drain_timeout_s=0.001),
    )
    admitted = await fixture.control.submit(
        "recognition.probe", {"channel": "peer"}, request_id=uuid4().hex
    )
    reclaiming = None
    try:
        async with asyncio.timeout(3):
            while not fixture.backends:
                await asyncio.sleep(0)
        backend = fixture.backends[0]
        await asyncio.wait_for(backend.close_entered.wait(), 1)
        pending = await fixture.control.wait(admitted["operation_id"], timeout=0.03)
        assert not pending["terminal"] and pending["recognition_result"] != "succeeded"
        assert backend.is_loaded and fixture.runtime._probe_handles
        if recovery.startswith("cancel"):
            cancelled = await asyncio.wait_for(
                fixture.control.cancel(admitted["operation_id"]), timeout=1
            )
            assert cancelled["status"] == cancelled["recognition_result"] == "cancelled"
            blocked = await fixture.submit("peer")
            assert blocked["status"] == "action_required"
            assert backend.is_loaded and fixture.runtime._probe_handles
        reclaiming = asyncio.create_task(
            fixture.runtime.close()
            if recovery.endswith("shutdown")
            else fixture.runtime.replace_provider(fixture.requests["peer"], start=False)
        )
        await asyncio.sleep(0)
        assert not reclaiming.done() and backend.is_loaded
        assert overlap == [False]
        backend.close_release.set()
        await asyncio.wait_for(reclaiming, timeout=2)
        result = await fixture.control.wait(admitted["operation_id"], timeout=2)
        assert_isolated(result)
        if recovery.startswith("cancel"):
            assert result["status"] == result["recognition_result"] == "cancelled"
        elif result["recognition_result"] == "succeeded":
            assert result["captured_generation"] == result["current_generation"]
        else:
            assert result["recognition_result"] == "stale"
        assert backend.close_completed.is_set() and not backend.is_loaded
        assert not fixture.runtime._probe_handles
        assert not any(overlap) and fixture.normal_events == []
    finally:
        for backend in fixture.backends:
            backend.close_release.set()
        if reclaiming is not None:
            await reclaiming
        await fixture.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("timeout", [False, True])
async def test_custom_probe_post_connect_cancel_and_readiness_timeout_close_only_owned_socket(
    tmp_path, monkeypatch, channel, timeout
):
    from puripuly_heart.providers.stt.custom import CustomSTTBackend
    from tests.providers.test_custom_stt import _StartupWebSocket

    live_socket = _StartupWebSocket()
    opening_socket = _StartupWebSocket(block_update=True, block_close=True)
    recovered_socket = _StartupWebSocket()
    sockets = iter((live_socket, opening_socket, recovered_socket))

    async def connect(*args, **kwargs):
        return next(sockets)

    fixture = ProbeRuntime(
        tmp_path,
        monkeypatch,
        lambda config: CustomSTTBackend(
            mode="realtime",
            compatibility="openai_realtime",
            endpoint="ws://127.0.0.1:1/v1",
            model="owned-synthetic",
            websocket_connect=connect,
        ),
    )
    fixture.requests[channel] = request(channel, "custom_realtime")
    engine = await fixture.attach(channel)
    settings = engine.probe_settings
    watchdogs = engine.watchdog_resolver(settings)
    engine.watchdog_resolver = lambda _: replace(
        watchdogs, readiness_timeout_s=0.02 if timeout else 1
    )
    live = await engine.session_factory(settings, "owned-live")
    admitted = await fixture.control.submit(
        "recognition.probe", {"channel": channel}, request_id=uuid4().hex
    )
    cancelling = None
    try:
        await asyncio.wait_for(opening_socket.update_started.wait(), 1)
        if not timeout:
            cancelling = asyncio.create_task(fixture.control.cancel(admitted["operation_id"]))
        await asyncio.wait_for(opening_socket.close_started.wait(), 1)
        pending = await fixture.control.operation(admitted["operation_id"])
        assert not pending["terminal"] and not opening_socket.closed
        assert not live_socket.closed
        opening_socket.close_release.set()
        result = await fixture.control.wait(admitted["operation_id"], timeout=2)
        if cancelling is not None:
            await cancelling
        assert_isolated(result)
        assert result["status"] == ("failed" if timeout else "cancelled")
        assert result["recognition_result"] == ("failed" if timeout else "cancelled")
        assert opening_socket.closed and not live_socket.closed
        assert engine.cleanup_debt == 0
        assert fixture.runtime.current_provider(channel) is engine
        recovered = await fixture.submit(channel)
        assert recovered["recognition_result"] == "succeeded"
        assert recovered["final_event"] == "authoritative_turn_final"
        assert bytes(recovered_socket.audio) == load_recognition_probe_audio(16000)
        assert recovered_socket.closed and not live_socket.closed
        assert fixture.normal_events == []
        await fixture.runtime.close()
        assert not live_socket.closed
    finally:
        opening_socket.close_release.set()
        if cancelling is not None:
            await cancelling
        await live.close()
        await fixture.close()
