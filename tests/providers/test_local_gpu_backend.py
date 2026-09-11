from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.app.ports.gpu_worker import (
    GpuWorkerActivation,
    GpuWorkerDevice,
    GpuWorkerTranscription,
)
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
)
from puripuly_heart.core.runtime.gpu_asr import (
    GpuASRDecodeDropped,
    GpuASRWorkDiscarded,
    GpuASRWorkExpired,
)
from puripuly_heart.core.stt.backend import (
    STTBackendTranscriptEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend

SCOPED_PROJECTION = STTSessionProjection(mode="scoped", provider_epoch_id="gpu-epoch")


pytestmark = pytest.mark.asyncio


class FakeSharedGpuRuntime:
    def __init__(self) -> None:
        self.active_channels: set[str] = set()
        self.activations: list[tuple[str, Path, str, str]] = []
        self.submissions: list[tuple[str, np.ndarray, float, str | None]] = []
        self.deactivations: list[str] = []
        self.deactivation_failures = 0
        self.detected_language: str | None = "en"
        self.submit_failures: list[BaseException] = []

    async def activate_channel(
        self,
        channel: str,
        *,
        model_path: Path,
        model_id: str,
        device_id: str,
    ) -> GpuWorkerActivation:
        self.active_channels.add(channel)
        self.activations.append((channel, model_path, model_id, device_id))
        return GpuWorkerActivation(
            device=GpuWorkerDevice(
                device_id="vk:0",
                registry_index=0,
                name="GPU",
                description="GPU",
                device_type="discrete",
                memory_total_bytes=1,
                memory_free_bytes=1,
            ),
            model_load_seconds=0.1,
            warmup_seconds=0.2,
        )

    async def submit(
        self,
        channel: str,
        samples_f32: np.ndarray,
        *,
        speech_end_at: float,
        language_hint: str | None = None,
    ) -> GpuWorkerTranscription:
        self.submissions.append((channel, samples_f32.copy(), speech_end_at, language_hint))
        if self.submit_failures:
            raise self.submit_failures.pop(0)
        return GpuWorkerTranscription(
            text="hello",
            detected_language=self.detected_language,
            audio_seconds=0.01,
            decode_seconds=0.02,
            rtf=2.0,
        )

    async def deactivate_channel(self, channel: str) -> None:
        if self.deactivation_failures > 0:
            self.deactivation_failures -= 1
            raise RuntimeError("GPU shutdown failed")
        self.active_channels.discard(channel)
        self.deactivations.append(channel)


def _scoped_request(order: int) -> STTProviderTurnRequest:
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(
            activation_generation=1,
            segment_order=order,
            segment_id=uuid4(),
            capture_epoch=1,
        ),
        provider_epoch_id="gpu-epoch",
        provider_turn_id=f"gpu-turn-{order}",
    )
    return STTProviderTurnRequest(
        identity=identity,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="local_qwen_gpu",
            provider_signature=("local_qwen_gpu",),
            runtime_signature=("local_qwen_gpu",),
            source_mode="desktop",
            source_language="auto",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.4,
            vad_hangover_ms=800,
            vad_pre_roll_ms=500,
        ),
    )


async def test_gpu_scoped_terminal_preserves_identity_and_expiry_is_not_empty(
    tmp_path: Path,
) -> None:
    runtime = FakeSharedGpuRuntime()
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="peer",
        model_path=tmp_path / "model.gguf",
        model_id="gpu-model",
        device_id="vk:0",
        source_mode="auto",
    )
    session = await backend.open_session(projection=SCOPED_PROJECTION)
    first = _scoped_request(1)
    await session.begin_turn(first)
    await session.send_turn_audio(
        first.identity,
        b"\x00\x40" * 160,
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await session.seal_turn(
        first.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert terminal.identity == first.identity
    assert terminal.outcome == "final"
    assert terminal.final_language_runs[0].language == "en"

    runtime.submit_failures.append(GpuASRWorkExpired("expired"))
    second = _scoped_request(2)
    await session.begin_turn(second)
    await session.send_turn_audio(
        second.identity,
        b"\x00\x40" * 160,
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await session.seal_turn(
        second.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.identity == second.identity
    assert terminal.outcome == "expired"
    assert terminal.text_authority == "none"
    await session.close()
    await backend.close()


async def test_gpu_scoped_empty_error_and_close_terminal_matrix(tmp_path: Path) -> None:
    runtime = FakeSharedGpuRuntime()
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="self",
        model_path=tmp_path / "model.gguf",
        model_id="gpu-model",
        device_id="vk:0",
    )
    session = await backend.open_session(projection=SCOPED_PROJECTION)
    empty = _scoped_request(1)
    await session.begin_turn(empty)
    await session.seal_turn(
        empty.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.outcome == "empty"
    await session.close()

    session = await backend.open_session(projection=SCOPED_PROJECTION)
    runtime.submit_failures.append(RuntimeError("native decode error"))
    failed = _scoped_request(2)
    await session.begin_turn(failed)
    await session.send_turn_audio(
        failed.identity,
        b"\x00\x40" * 160,
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await session.seal_turn(
        failed.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.outcome == "failed"
    assert terminal.epoch_disposition == "retire"
    await session.close()

    session = await backend.open_session(projection=SCOPED_PROJECTION)
    closed = _scoped_request(3)
    await session.begin_turn(closed)
    stream = session.turn_events()
    await session.close()
    terminal = await asyncio.wait_for(stream.__anext__(), timeout=1)
    assert terminal.outcome == "failed"
    assert terminal.failure_reason == "session_closed"
    await backend.close()


async def test_gpu_active_timeout_quarantines_resource_until_repeated_off_cleanup(
    tmp_path: Path,
) -> None:
    class BlockingRuntime(FakeSharedGpuRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def submit(
            self,
            channel: str,
            samples_f32: np.ndarray,
            *,
            speech_end_at: float,
            language_hint: str | None = None,
        ) -> GpuWorkerTranscription:
            self.submissions.append((channel, samples_f32.copy(), speech_end_at, language_hint))
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
                raise
            return GpuWorkerTranscription(
                text="late",
                detected_language=None,
                audio_seconds=0.01,
                decode_seconds=0.02,
                rtf=2.0,
            )

    runtime = BlockingRuntime()
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="self",
        model_path=tmp_path / "model.gguf",
        model_id="gpu-model",
        device_id="vk:0",
        active_decode_timeout_s=0.01,
    )
    session = await backend.open_session(projection=SCOPED_PROJECTION)
    request = _scoped_request(1)
    await session.begin_turn(request)
    await session.send_turn_audio(
        request.identity,
        b"\x00\x40" * 160,
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    with pytest.raises(RuntimeError, match="already sealed"):
        await session.seal_turn(
            request.identity,
            sealed_content_ranges=(),
            seal_reason="duplicate",
            observed_trailing_silence_ms=800,
        )
    await asyncio.wait_for(runtime.started.wait(), timeout=1)
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.outcome == "failed"
    assert terminal.failure_reason == "local_decode_timeout"
    assert terminal.epoch_disposition == "retire"
    with pytest.raises(RuntimeError, match="awaiting decode cleanup"):
        await backend.open_session(projection=SCOPED_PROJECTION)
    assert len(runtime.submissions) == 1
    assert runtime.deactivations == []

    first_off = asyncio.create_task(session.abort_for_toggle_off())
    second_off = asyncio.create_task(session.abort_for_toggle_off())
    await asyncio.sleep(0)
    assert not first_off.done()
    assert len(runtime.submissions) == 1
    assert runtime.deactivations == []
    runtime.release.set()
    await asyncio.wait_for(first_off, timeout=1)
    await asyncio.wait_for(second_off, timeout=1)

    replacement = await backend.open_session()
    assert len(runtime.activations) == 1
    await replacement.close()
    await backend.close()
    assert runtime.deactivations == ["self"]


async def test_backend_is_lazy_and_deactivates_only_its_channel(tmp_path: Path) -> None:
    runtime = FakeSharedGpuRuntime()
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="self",
        model_path=tmp_path / "model.gguf",
        model_id="gpu-model",
        device_id="vk:0",
    )

    assert runtime.activations == []

    session = await backend.open_session()
    assert runtime.activations == [("self", tmp_path / "model.gguf", "gpu-model", "vk:0")]
    second_session = await backend.open_session()

    assert runtime.activations == [
        ("self", tmp_path / "model.gguf", "gpu-model", "vk:0"),
    ]

    await session.close()
    await second_session.close()
    assert runtime.deactivations == []
    await backend.close()
    assert runtime.deactivations == ["self"]


async def test_backend_close_can_retry_after_runtime_shutdown_failure(tmp_path: Path) -> None:
    runtime = FakeSharedGpuRuntime()
    runtime.deactivation_failures = 1
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="self",
        model_path=tmp_path / "model.gguf",
        model_id="gpu-model",
        device_id="vk:0",
    )
    await backend.open_session()

    with pytest.raises(RuntimeError, match="GPU shutdown failed"):
        await backend.close()

    await backend.close()
    assert runtime.deactivations == ["self"]


async def test_session_submits_float_audio_at_speech_end_without_blocking() -> None:
    runtime = FakeSharedGpuRuntime()
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="peer",
        model_path=Path("model.gguf"),
        model_id="gpu-model",
        device_id="auto",
        speech_end_clock=lambda: 12.5,
    )
    session = await backend.open_session()

    await session.send_audio_f32(np.array([0.25, -0.25], dtype=np.float32))
    await session.on_speech_end()
    event = await asyncio.wait_for(anext(session.events()), timeout=0.5)

    assert event.text == "hello"
    assert event.is_final is True
    assert event.final_language_runs == ()
    assert len(runtime.submissions) == 1
    channel, samples, speech_end_at, language_hint = runtime.submissions[0]
    assert channel == "peer"
    assert np.array_equal(samples, np.array([0.25, -0.25], dtype=np.float32))
    assert speech_end_at == 12.5
    assert language_hint is None

    await session.close()
    await backend.close()


async def test_gpu_qwen_preserves_full_audio_on_speech_end() -> None:
    runtime = FakeSharedGpuRuntime()
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="self",
        model_path=Path("model.gguf"),
        model_id="qwen3-asr-1.7b",
        device_id="auto",
    )
    session = await backend.open_session()
    samples = np.arange(16_000, dtype=np.float32)

    await session.send_audio_f32(samples)
    await session.on_speech_end(trailing_silence_ms=400)
    event = await asyncio.wait_for(anext(session.events()), timeout=0.5)

    assert event == STTBackendTranscriptEvent(text="hello", is_final=True)
    assert len(runtime.submissions) == 1
    assert np.array_equal(runtime.submissions[0][1], samples)
    await session.close()
    await backend.close()


async def test_decode_drop_emits_empty_final_and_keeps_session_for_new_utterance() -> None:
    runtime = FakeSharedGpuRuntime()
    runtime.submit_failures.append(GpuASRDecodeDropped("decode_failure"))
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="peer",
        model_path=Path("model.gguf"),
        model_id="gpu-model",
        device_id="auto",
    )
    session = await backend.open_session()
    events = session.events()

    await session.send_audio_f32(np.zeros(120_000, dtype=np.float32))
    await session.on_speech_end()
    dropped = await asyncio.wait_for(anext(events), timeout=0.5)
    await session.send_audio_f32(np.ones(1600, dtype=np.float32))
    await session.on_speech_end()
    recovered = await asyncio.wait_for(anext(events), timeout=0.5)

    assert dropped.text == ""
    assert dropped.is_final is True
    assert recovered.text == "hello"
    assert recovered.is_final is True
    assert len(runtime.submissions) == 2
    assert runtime.submissions[0][1].size == 120_000
    assert runtime.submissions[1][1].size == 1600
    await session.close()
    await backend.close()


async def test_work_expiry_emits_empty_final_and_keeps_session_for_new_utterance() -> None:
    runtime = FakeSharedGpuRuntime()
    runtime.submit_failures.append(GpuASRWorkExpired("speech_end_ttl"))
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="peer",
        model_path=Path("model.gguf"),
        model_id="gpu-model",
        device_id="auto",
    )
    session = await backend.open_session()
    events = session.events()

    await session.send_audio_f32(np.zeros(120_000, dtype=np.float32))
    await session.on_speech_end()
    expired = await asyncio.wait_for(anext(events), timeout=0.5)
    await session.send_audio_f32(np.ones(1600, dtype=np.float32))
    await session.on_speech_end()
    recovered = await asyncio.wait_for(anext(events), timeout=0.5)

    assert expired.text == ""
    assert expired.is_final is True
    assert recovered.text == "hello"
    assert recovered.is_final is True
    assert len(runtime.submissions) == 2
    await session.close()
    await backend.close()


async def test_work_discard_still_fails_the_session() -> None:
    runtime = FakeSharedGpuRuntime()
    runtime.submit_failures.append(GpuASRWorkDiscarded("channel_disabled"))
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="peer",
        model_path=Path("model.gguf"),
        model_id="gpu-model",
        device_id="auto",
    )
    session = await backend.open_session()
    events = session.events()

    await session.send_audio_f32(np.ones(1600, dtype=np.float32))
    await session.on_speech_end()
    dropped = await asyncio.wait_for(anext(events), timeout=0.5)

    assert dropped.text == ""
    assert dropped.is_final is True
    with pytest.raises(GpuASRWorkDiscarded, match="channel_disabled"):
        await asyncio.wait_for(anext(events), timeout=0.5)

    await session.close()
    await backend.close()


async def test_peer_auto_emits_one_detected_language_run_for_whole_utterance() -> None:
    runtime = FakeSharedGpuRuntime()
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="peer",
        model_path=Path("model.gguf"),
        model_id="gpu-model",
        device_id="auto",
        source_mode="auto",
        language_hint=None,
    )
    session = await backend.open_session()

    await session.send_audio_f32(np.array([0.25], dtype=np.float32))
    await session.on_speech_end()
    event = await asyncio.wait_for(anext(session.events()), timeout=0.5)

    assert [(run.text, run.language) for run in event.final_language_runs] == [("hello", "en")]
    assert runtime.submissions[0][3] is None
    await session.close()
    await backend.close()


async def test_peer_manual_passes_hint_without_exposing_detected_language_run() -> None:
    runtime = FakeSharedGpuRuntime()
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="peer",
        model_path=Path("model.gguf"),
        model_id="gpu-model",
        device_id="auto",
        source_mode="manual",
        language_hint="ja",
    )
    session = await backend.open_session()

    await session.send_audio_f32(np.array([0.25], dtype=np.float32))
    await session.on_speech_end()
    event = await asyncio.wait_for(anext(session.events()), timeout=0.5)

    assert event.final_language_runs == ()
    assert runtime.submissions[0][3] == "ja"
    await session.close()
    await backend.close()


async def test_peer_auto_missing_detected_language_omits_run_for_manual_fallback() -> None:
    runtime = FakeSharedGpuRuntime()
    runtime.detected_language = None
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="peer",
        model_path=Path("model.gguf"),
        model_id="gpu-model",
        device_id="auto",
        source_mode="auto",
    )
    session = await backend.open_session()

    await session.send_audio_f32(np.array([0.25], dtype=np.float32))
    await session.on_speech_end()
    event = await asyncio.wait_for(anext(session.events()), timeout=0.5)

    assert event.text == "hello"
    assert event.final_language_runs == ()
    await session.close()
    await backend.close()
