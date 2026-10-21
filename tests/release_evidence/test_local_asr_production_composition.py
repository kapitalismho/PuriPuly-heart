from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeDiagnostic

import puripuly_heart.composition.local_asr_production_evidence as composition
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.release_evidence import local_asr_production_composition as evidence


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 process handles are required")
def test_process_present_observes_live_and_exited_process() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert evidence._process_present(process.pid)
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert not evidence._process_present(process.pid)


def _terminal(*, text: str = "transcript", outcome: str = "final") -> STTProviderTurnTerminal:
    return STTProviderTurnTerminal(
        identity=STTProviderTurnIdentity(
            segment=AudioSegmentIdentity(
                activation_generation=1,
                segment_order=1,
                segment_id=uuid4(),
                capture_epoch=1,
            ),
            provider_epoch_id="epoch",
            provider_turn_id="turn",
        ),
        outcome=outcome,
        text=text,
        text_authority="authoritative" if outcome == "final" else "none",
    )


def _owner_snapshot(
    *,
    worker_pid: int | None,
    active_channels: frozenset[str],
    configured_device_id: str = "vulkan-index-0",
    model_resident: bool = True,
    self_model: str | None = "qwen3-asr-1.7b",
    peer_model: str | None = "qwen3-asr-1.7b",
) -> SimpleNamespace:
    return SimpleNamespace(
        snapshot=SimpleNamespace(
            gpu=SimpleNamespace(
                worker_pid=worker_pid,
                active_channels=active_channels,
                configured_device_id=configured_device_id,
                model_resident=model_resident,
            ),
            channels=(
                SimpleNamespace(channel="self", model_id=self_model),
                SimpleNamespace(channel="peer", model_id=peer_model),
            ),
        )
    )


def test_require_final_preserves_scoped_terminal_evidence() -> None:
    fact = evidence._require_final(
        _terminal(text="heard speech"),
        channel="peer",
        stage="peer inference",
    )

    assert fact["text"] == "heard speech"
    assert fact["is_final"] is True
    assert fact["channel"] == "peer"
    assert fact["outcome"] == "final"


def test_require_final_rejects_empty_or_non_final_terminal() -> None:
    with pytest.raises(RuntimeError):
        evidence._require_final(
            _terminal(text="", outcome="final"),
            channel="self",
            stage="self inference",
        )
    with pytest.raises(RuntimeError):
        evidence._require_final(
            _terminal(text="heard", outcome="failed"),
            channel="self",
            stage="self inference",
        )


def test_require_final_rejects_cross_channel_domain_result() -> None:
    with pytest.raises(RuntimeError, match="expected 'peer'"):
        evidence._require_final(
            SimpleNamespace(
                transcript=SimpleNamespace(
                    text="transcript",
                    is_final=True,
                    channel="self",
                    final_language_runs=(),
                )
            ),
            channel="peer",
            stage="peer inference",
        )


@pytest.mark.asyncio
async def test_wait_final_any_prefers_retired_then_current() -> None:
    retired = [_terminal(text="retired")]
    current = [_terminal(text="current")]
    preferred = await evidence._wait_final_any((retired, 0), (current, 0))
    assert preferred.text == "retired"

    only_current = await evidence._wait_final_any(([], 0), (current, 0))
    assert only_current.text == "current"


def test_console_safe_preserves_report_and_escapes_unencodable_text() -> None:
    rendered = '{"transcript": "Käse"}'

    assert evidence._console_safe(rendered, encoding="cp949") == ('{"transcript": "K\\xe4se"}')


@pytest.mark.asyncio
async def test_run_stage_reports_and_cancels_a_blocked_production_operation() -> None:
    report: dict[str, object] = {}
    never = asyncio.Event()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_operation() -> None:
        started.set()
        try:
            await never.wait()
        finally:
            cancelled.set()

    with pytest.raises(RuntimeError, match="blocked_operation.*exceeded"):
        await evidence._run_stage(
            report,
            "blocked_operation",
            blocked_operation(),
            timeout=0.01,
        )

    assert report["active_stage"] == "blocked_operation"
    assert report["stage_timeout"]["stack"]
    assert started.is_set()
    assert cancelled.is_set()


def test_recovered_activation_requires_new_resolved_physical_device() -> None:
    owner = SimpleNamespace(
        diagnostics=(
            ProviderRuntimeDiagnostic(event="activation_ready", device_id="old-device"),
            ProviderRuntimeDiagnostic(event="gpu_recovery", outcome="applied"),
            ProviderRuntimeDiagnostic(
                event="activation_ready",
                device_id="vulkan-index-0",
            ),
        )
    )

    fact = evidence._require_activation_device(
        owner,
        diagnostics_start=1,
        expected_device_id="vulkan-index-0",
        stage="recovered inference",
    )

    assert fact["device_id"] == "vulkan-index-0"


def test_require_gpu_session_rejects_worker_before_channel_open() -> None:
    owner = _owner_snapshot(worker_pid=None, active_channels=frozenset(), model_resident=False)
    with pytest.raises(RuntimeError):
        evidence._require_gpu_session(
            owner,
            channel="self",
            expected_device_id="vulkan-index-0",
            stage="production Self inference",
        )


def test_require_gpu_session_requires_shared_pid_and_both_channels() -> None:
    after_self = _owner_snapshot(worker_pid=4242, active_channels=frozenset({"self"}))
    pid = evidence._require_gpu_session(
        after_self,
        channel="self",
        expected_device_id="vulkan-index-0",
        stage="production Self inference",
    )
    assert pid == 4242

    after_peer = _owner_snapshot(worker_pid=4242, active_channels=frozenset({"self", "peer"}))
    shared = evidence._require_gpu_session(
        after_peer,
        channel="peer",
        expected_device_id="vulkan-index-0",
        expected_pid=4242,
        stage="production Peer inference",
    )
    assert shared == 4242

    with pytest.raises(RuntimeError):
        evidence._require_gpu_session(
            _owner_snapshot(worker_pid=99, active_channels=frozenset({"self", "peer"})),
            channel="peer",
            expected_device_id="vulkan-index-0",
            expected_pid=4242,
            stage="production Peer inference",
        )
    with pytest.raises(RuntimeError):
        evidence._require_gpu_session(
            _owner_snapshot(worker_pid=4242, active_channels=frozenset({"peer"})),
            channel="peer",
            expected_device_id="vulkan-index-0",
            expected_pid=4242,
            stage="production Peer inference",
        )


def test_runner_rejects_non_packaged_execution_with_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        evidence,
        "current_runtime_layout",
        lambda: SimpleNamespace(host_kind="source"),
    )
    report_path = tmp_path / "report.json"

    result = evidence.run_local_asr_production_composition(
        audio_path=tmp_path / "speech.wav",
        report_path=report_path,
        candidate="candidate-sha",
        expected_gpu_name="RX 7900 XTX",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 1
    assert report["status"] == "failed"
    assert report["failure_type"] == "RuntimeError"


def test_runner_accepts_native_runtime_layout_before_validating_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(evidence, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        evidence,
        "current_runtime_layout",
        lambda: SimpleNamespace(
            host_kind="native",
            host_executable=tmp_path / "PuriPulyHeart.exe",
        ),
    )
    monkeypatch.setattr(evidence, "local_gpu_model_path", lambda: tmp_path / "missing-model")

    result = evidence.run_local_asr_production_composition(
        audio_path=tmp_path / "missing-speech.wav",
        report_path=report_path,
        candidate="candidate-sha",
        expected_gpu_name="RX 7900 XTX",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 1
    assert report["failure_type"] == "FileNotFoundError"


def test_runner_initializes_and_closes_through_the_real_composition_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "settings.json"
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"synthetic-model-identity")
    audio_path = tmp_path / "speech.wav"
    with wave.open(str(audio_path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(16000)
        sink.writeframes(np.zeros(160, dtype="<i2").tobytes())
    calls: list[str] = []
    initialized: list[AppSettingsVNext] = []
    original_settings = AppSettingsVNext()
    settings_before_run = copy.deepcopy(original_settings)

    def load_settings() -> AppSettingsVNext:
        calls.append("load_settings")
        return original_settings

    async def initialize(settings: AppSettingsVNext) -> None:
        calls.append("initialize")
        initialized.append(settings)

    async def stop() -> None:
        calls.append("stop")

    def compose_runtime(**kwargs):
        assert kwargs["config_path"] == config_path
        calls.append("compose_runtime")
        kwargs["local_asr_evidence_sink"](
            SimpleNamespace(
                config_path=config_path,
                load_compatibility_settings=load_settings,
                initialize=initialize,
                owner=object(),
            )
        )
        return SimpleNamespace(stop=stop)

    monkeypatch.setattr(composition, "compose_application_runtime", compose_runtime)
    monkeypatch.setattr(evidence, "os", SimpleNamespace(name="nt", environ={}))
    monkeypatch.setattr(
        evidence,
        "current_runtime_layout",
        lambda: SimpleNamespace(
            host_kind="native",
            host_executable=tmp_path / "PuriPulyHeart.exe",
        ),
    )
    monkeypatch.setattr(evidence, "local_gpu_model_path", lambda: model_path)
    monkeypatch.setattr(evidence, "default_settings_path", lambda: config_path)
    report_path = tmp_path / "report.json"

    exit_code = evidence.run_local_asr_production_composition(
        audio_path=audio_path,
        report_path=report_path,
        candidate="candidate-sha",
        expected_gpu_name="synthetic-device",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert calls == ["compose_runtime", "load_settings", "initialize", "stop"]
    assert len(initialized) == 1
    settings = initialized[0]
    assert settings.intent.stt.provider == settings.intent.peer_stt.provider == "local_qwen_gpu"
    assert settings.intent.secrets.backend == "encrypted_file"
    assert settings.intent.osc.chatbox_send is False
    assert original_settings == settings_before_run
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["failure_type"] == "RuntimeError"
    assert report["failure"] == "production application did not compose the canonical owner"
    assert report["candidate"] == "candidate-sha"
    assert report["audio_samples"] == 160
    assert "shutdown_failure" not in report


class _ScopedSession:
    def __init__(self) -> None:
        self.buffer = STTProviderEventBuffer()
        self.audio = bytearray()

    async def begin_turn(self, _request: STTProviderTurnRequest) -> None:
        return None

    async def send_turn_audio(
        self,
        _identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        **_kwargs,
    ) -> None:
        self.audio.extend(pcm16le)

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        **_kwargs,
    ) -> None:
        self.buffer.put(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="final",
                text="scoped transcript",
                text_authority="authoritative",
            )
        )

    async def abort_turn(self, _identity: STTProviderTurnIdentity, **_kwargs) -> None:
        return None

    async def turn_events(self):
        async for event in self.buffer.events():
            yield event

    async def stop(self) -> None:
        return None

    async def close(self) -> None:
        self.buffer.close()


def _scoped_request() -> SimpleNamespace:
    return SimpleNamespace(
        provider_id="fake-scoped",
        provider_signature=("fake-scoped",),
        runtime_signature=("fake-scoped", "runtime"),
        session_options=None,
        config=SimpleNamespace(
            source_language="en",
            source_mode="manual",
            sample_rate_hz=16000,
            vad_speech_threshold=0.6,
            vad_hangover_ms=900,
            vad_pre_roll_ms=500,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("channel", "generation"), [("self", 3), ("peer", 7)])
async def test_production_probe_uses_each_owned_scoped_recognition_path(
    channel: str,
    generation: int,
) -> None:
    events: list[object] = []
    dispatched: list[str] = []
    request = _scoped_request()
    session = _ScopedSession()
    engine = ScopedRecognitionEngine(
        channel=channel,
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=events.append,
        accepted_settings_scope=(
            request.provider_id,
            request.provider_signature,
            request.runtime_signature,
        ),
    )

    class Runtime:
        async def handle_vad_event(self, owned: object) -> None:
            dispatched.append("self")
            await engine.handle_owned_vad_event(owned)

        async def handle_peer_owned_vad_event(self, owned: object) -> None:
            dispatched.append("peer")
            await engine.handle_owned_vad_event(owned)

    samples = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
    runtime = Runtime()
    try:
        final = await evidence._send_utterance(
            application=SimpleNamespace(self_vad=runtime, peer_vad=runtime),
            channel=channel,
            samples=samples,
            events=events,
            request=request,
            activation_generation=generation,
        )
    finally:
        await engine.close()

    assert dispatched == [channel, channel]
    assert isinstance(final, STTProviderTurnTerminal)
    assert final.outcome == "final"
    assert final.text == "scoped transcript"
    assert final.identity.segment.activation_generation == generation
    assert len(session.audio) == samples.size * 2


@pytest.mark.asyncio
async def test_recovered_ready_provider_delivers_after_channel_restart() -> None:
    events: list[object] = []
    dispatch_completed = asyncio.Event()

    async def event_handler(event: object) -> None:
        events.append(event)
        await asyncio.sleep(0)
        dispatch_completed.set()

    request = _scoped_request()
    session = _ScopedSession()
    engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        accepted_settings_scope=(
            request.provider_id,
            request.provider_signature,
            request.runtime_signature,
        ),
    )
    handle = ProviderRuntimeHandle(
        name="recovered_self",
        event_handler=event_handler,
    )
    await handle.replace_provider(engine, start=False)
    assert await handle.start_if_provider(engine)

    class SelfRuntime:
        async def handle_vad_event(self, owned: object) -> None:
            await engine.handle_owned_vad_event(owned)

    try:
        final = await evidence._send_utterance_staged(
            report={},
            stage="recovered_self",
            application=SimpleNamespace(self_vad=SelfRuntime()),
            channel="self",
            samples=np.linspace(-0.5, 0.5, 160, dtype=np.float32),
            events=events,
            request=request,
            activation_generation=4,
        )
        await asyncio.wait_for(dispatch_completed.wait(), timeout=1.0)
    finally:
        await handle.close()

    assert isinstance(final, STTProviderTurnTerminal)
    assert final.outcome == "final"
