from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from puripuly_heart.app.ports.gpu_worker import (
    GpuWorkerActivation,
    GpuWorkerDevice,
    GpuWorkerTranscription,
)
from puripuly_heart.core.runtime.gpu_asr import (
    GpuASRDecodeDropped,
    GpuASRWorkDiscarded,
    GpuASRWorkExpired,
)
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend

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
