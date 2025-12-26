from __future__ import annotations

import asyncio
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION") != "1", reason="set INTEGRATION=1 to run integration tests"
)


@dataclass
class MockOscSender:
    messages: list[str] = field(default_factory=list)
    typing_states: list[bool] = field(default_factory=list)

    def send_chatbox(self, text: str) -> None:
        self.messages.append(text)

    def send_typing(self, is_typing: bool) -> None:
        self.typing_states.append(is_typing)


class SimpleClock:
    def now(self) -> float:
        return time.time()


def load_audio_wav(filepath: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(filepath), "rb") as f:
        sample_rate = f.getframerate()
        n_frames = f.getnframes()
        audio_data = f.readframes(n_frames)

    samples_int16 = np.frombuffer(audio_data, dtype=np.int16)
    samples_f32 = samples_int16.astype(np.float32) / 32768.0
    return samples_f32, sample_rate


@pytest.mark.asyncio
async def test_qwen_asr_llm_pipeline_smoke() -> None:
    api_key = os.getenv("ALIBABA_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("missing env var ALIBABA_API_KEY (or DASHSCOPE_API_KEY)")

    try:
        import dashscope  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "dashscope is required for this integration test; install project dependencies."
        ) from exc

    from ajebal_daera_translator.config.prompts import load_prompt_for_provider
    from ajebal_daera_translator.core.language import get_llm_language_name
    from ajebal_daera_translator.core.llm.provider import SemaphoreLLMProvider
    from ajebal_daera_translator.core.orchestrator.hub import ClientHub
    from ajebal_daera_translator.core.osc.smart_queue import SmartOscQueue
    from ajebal_daera_translator.core.stt.controller import ManagedSTTProvider
    from ajebal_daera_translator.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
    from ajebal_daera_translator.domain.events import UIEventType
    from ajebal_daera_translator.providers.llm.qwen import QwenLLMProvider
    from ajebal_daera_translator.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend

    audio_env = os.getenv("TEST_AUDIO_PATH")
    audio_path = (
        Path(audio_env)
        if audio_env
        else Path(__file__).parent.parent.parent / ".test_audio" / "test_speech.wav"
    )
    if not audio_path.exists():
        pytest.skip(f"Audio file not found: {audio_path}")

    audio_samples, sample_rate = load_audio_wav(audio_path)
    if sample_rate not in (8000, 16000):
        pytest.skip(f"Unsupported sample rate: {sample_rate}")

    stt_backend = QwenASRRealtimeSTTBackend(
        api_key=api_key,
        model=os.getenv("QWEN_ASR_MODEL", "qwen3-asr-flash-realtime"),
        endpoint=os.getenv(
            "QWEN_ASR_ENDPOINT", "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
        ),
        language=os.getenv("QWEN_ASR_LANGUAGE", "ko"),
        sample_rate_hz=sample_rate,
    )
    stt = ManagedSTTProvider(
        backend=stt_backend,
        sample_rate_hz=sample_rate,
        reset_deadline_s=90.0,
        drain_timeout_s=5.0,
        bridging_ms=300,
    )

    llm_base = QwenLLMProvider(
        api_key=api_key,
        model=os.getenv("QWEN_LLM_MODEL", "qwen-mt-flash"),
    )
    llm = SemaphoreLLMProvider(inner=llm_base, semaphore=asyncio.Semaphore(1))

    mock_sender = MockOscSender()
    osc = SmartOscQueue(
        sender=mock_sender,
        clock=SimpleClock(),
        max_chars=144,
        cooldown_s=0.1,
        ttl_s=10.0,
    )

    source_lang = os.getenv("QWEN_LLM_SOURCE_LANGUAGE", "ko")
    target_lang = os.getenv("QWEN_LLM_TARGET_LANGUAGE", "en")
    system_prompt = load_prompt_for_provider("qwen") or "Translate ${sourceName} to ${targetName}."
    system_prompt = system_prompt.replace("${sourceName}", get_llm_language_name(source_lang))
    system_prompt = system_prompt.replace("${targetName}", get_llm_language_name(target_lang))

    hub = ClientHub(
        stt=stt,
        llm=llm,
        osc=osc,
        source_language=source_lang,
        target_language=target_lang,
        system_prompt=system_prompt,
        fallback_transcript_only=False,
        translation_enabled=True,
    )

    got_result = asyncio.Event()
    translation_text = ""
    error_message = ""

    async def track_events() -> None:
        nonlocal translation_text, error_message
        while True:
            try:
                event = await asyncio.wait_for(hub.ui_events.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if event.type == UIEventType.TRANSLATION_DONE:
                translation_text = event.payload.text
                got_result.set()
            elif event.type == UIEventType.ERROR:
                error_message = str(event.payload)
                got_result.set()

    await hub.start(auto_flush_osc=True)
    event_task = asyncio.create_task(track_events())
    await asyncio.sleep(0.5)

    try:
        utterance_id = uuid4()
        chunk_samples = sample_rate // 10
        if chunk_samples <= 0:
            pytest.skip("Invalid chunk size for sample rate")

        chunks = [
            audio_samples[i : i + chunk_samples]
            for i in range(0, len(audio_samples), chunk_samples)
        ]
        if not chunks:
            pytest.skip("Audio file is empty")

        pre_roll = np.zeros(chunk_samples, dtype=np.float32)
        await hub.handle_vad_event(SpeechStart(utterance_id, pre_roll=pre_roll, chunk=chunks[0]))
        await asyncio.sleep(0.05)

        for chunk in chunks[1:]:
            await hub.handle_vad_event(SpeechChunk(utterance_id, chunk=chunk))
            await asyncio.sleep(0.05)

        await hub.handle_vad_event(SpeechEnd(utterance_id))

        await asyncio.wait_for(got_result.wait(), timeout=30.0)
    finally:
        event_task.cancel()
        await hub.stop()
        await asyncio.gather(event_task, return_exceptions=True)

    if error_message:
        pytest.fail(f"Pipeline error: {error_message}")

    assert translation_text.strip()
