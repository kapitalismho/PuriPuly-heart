"""E2E Latency Measurement Test.

Measures end-to-end latency for the complete pipeline:
Audio → STT (Deepgram) → LLM (Gemini) → OSC

Runs 5 iterations and outputs detailed latency statistics.
"""

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
    os.getenv("INTEGRATION") != "1",
    reason="set INTEGRATION=1 to run integration tests",
)


@dataclass
class IterationMetrics:
    """Metrics for a single iteration."""

    iteration: int
    audio_send_start: float = 0.0
    audio_send_end: float = 0.0
    stt_final: float = 0.0
    llm_done: float = 0.0
    osc_sent: float = 0.0
    transcript_text: str = ""
    translation_text: str = ""

    @property
    def stt_latency_ms(self) -> float:
        """Time from audio send start to STT final result."""
        if self.stt_final and self.audio_send_start:
            return (self.stt_final - self.audio_send_start) * 1000
        return 0.0

    @property
    def llm_latency_ms(self) -> float:
        """Time from STT final to LLM translation done."""
        if self.llm_done and self.stt_final:
            return (self.llm_done - self.stt_final) * 1000
        return 0.0

    @property
    def total_latency_ms(self) -> float:
        """Total time from audio send start to OSC sent."""
        if self.osc_sent and self.audio_send_start:
            return (self.osc_sent - self.audio_send_start) * 1000
        return 0.0


@dataclass
class MockOscSender:
    """Mock OSC sender that records messages without network."""

    messages: list[str] = field(default_factory=list)
    typing_states: list[bool] = field(default_factory=list)

    def send_chatbox(self, text: str) -> None:
        self.messages.append(text)

    def send_typing(self, is_typing: bool) -> None:
        self.typing_states.append(is_typing)


class SimpleClock:
    """Simple clock for OSC queue."""

    def now(self) -> float:
        return time.time()


def load_audio_wav(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Load WAV file and return (float32_samples, sample_rate)."""
    with wave.open(str(filepath), "rb") as f:
        sample_rate = f.getframerate()
        n_frames = f.getnframes()
        audio_data = f.readframes(n_frames)

    # Convert PCM16 to float32 (-1.0 to 1.0)
    samples_int16 = np.frombuffer(audio_data, dtype=np.int16)
    samples_f32 = samples_int16.astype(np.float32) / 32768.0
    return samples_f32, sample_rate


def print_summary_table(results: list[IterationMetrics]) -> None:
    """Print detailed latency summary table."""
    print("\n" + "=" * 70)
    print("E2E LATENCY MEASUREMENT RESULTS (5 Iterations)")
    print("=" * 70)
    print(f"{'Iter':<6} {'STT (ms)':<12} {'LLM (ms)':<12} {'Total (ms)':<12} {'Transcript'}")
    print("-" * 70)

    stt_latencies = []
    llm_latencies = []
    total_latencies = []

    for m in results:
        print(
            f"{m.iteration:<6} "
            f"{m.stt_latency_ms:<12.1f} "
            f"{m.llm_latency_ms:<12.1f} "
            f"{m.total_latency_ms:<12.1f} "
            f"{m.transcript_text[:20]}..."
        )
        if m.stt_latency_ms > 0:
            stt_latencies.append(m.stt_latency_ms)
        if m.llm_latency_ms > 0:
            llm_latencies.append(m.llm_latency_ms)
        if m.total_latency_ms > 0:
            total_latencies.append(m.total_latency_ms)

    print("-" * 70)

    if stt_latencies and llm_latencies and total_latencies:
        avg_stt = sum(stt_latencies) / len(stt_latencies)
        avg_llm = sum(llm_latencies) / len(llm_latencies)
        avg_total = sum(total_latencies) / len(total_latencies)

        min_stt = min(stt_latencies)
        min_llm = min(llm_latencies)
        min_total = min(total_latencies)

        max_stt = max(stt_latencies)
        max_llm = max(llm_latencies)
        max_total = max(total_latencies)

        print(f"{'AVG':<6} {avg_stt:<12.1f} {avg_llm:<12.1f} {avg_total:<12.1f}")
        print(f"{'MIN':<6} {min_stt:<12.1f} {min_llm:<12.1f} {min_total:<12.1f}")
        print(f"{'MAX':<6} {max_stt:<12.1f} {max_llm:<12.1f} {max_total:<12.1f}")

    print("=" * 70)


@pytest.mark.asyncio
async def test_e2e_latency_5_iterations():
    """Measure E2E latency for 5 iterations using test audio.

    Pipeline: Audio → Deepgram STT → Gemini LLM → OSC

    Run with:
        set DEEPGRAM_API_KEY=your_key
        set GOOGLE_API_KEY=your_key
        set INTEGRATION=1
        set TEST_AUDIO_PATH=path\to\test_speech.wav
        python -m pytest tests/integration/test_e2e_latency_measurement.py -v -s
    """
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if not deepgram_key:
        pytest.skip("missing DEEPGRAM_API_KEY env var")
    if not google_key:
        pytest.skip("missing GOOGLE_API_KEY env var")

    from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
    from puripuly_heart.core.orchestrator.hub import ClientHub
    from puripuly_heart.core.osc.smart_queue import SmartOscQueue
    from puripuly_heart.core.stt.controller import ManagedSTTProvider
    from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
    from puripuly_heart.domain.events import UIEventType
    from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
    from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend

    # Load audio file
    audio_env = os.getenv("TEST_AUDIO_PATH")
    audio_path = (
        Path(audio_env)
        if audio_env
        else Path(__file__).parent.parent.parent / ".test_audio" / "test_speech.wav"
    )
    if not audio_path.exists():
        pytest.skip(f"Audio file not found: {audio_path}")

    audio_samples, sample_rate = load_audio_wav(audio_path)
    print(f"\nLoaded audio: {audio_path.name} ({len(audio_samples)} samples, {sample_rate} Hz)")

    # Results storage
    results: list[IterationMetrics] = []

    # Run 5 iterations
    for iteration in range(5):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/5")
        print(f"{'='*60}")

        metrics = IterationMetrics(iteration=iteration)

        # Create fresh providers for each iteration
        stt_backend = DeepgramRealtimeSTTBackend(
            api_key=deepgram_key,
            model="nova-3",
            language="ko",
            sample_rate_hz=sample_rate,
        )

        stt = ManagedSTTProvider(
            backend=stt_backend,
            sample_rate_hz=sample_rate,
            reset_deadline_s=90.0,
            drain_timeout_s=5.0,
            bridging_ms=300,
        )

        llm_base = GeminiLLMProvider(api_key=google_key)
        llm = SemaphoreLLMProvider(inner=llm_base, semaphore=asyncio.Semaphore(1))

        mock_sender = MockOscSender()
        osc = SmartOscQueue(
            sender=mock_sender,
            clock=SimpleClock(),
            max_chars=144,
            cooldown_s=0.1,
            ttl_s=10.0,
        )

        # Load default prompt
        from puripuly_heart.config.prompts import load_prompt
        from puripuly_heart.core.language import get_llm_language_name

        source_lang = "ko"
        target_lang = "en"
        system_prompt = load_prompt("default.txt") or "Translate naturally and concisely."
        # Replace language placeholders in prompt
        system_prompt = system_prompt.replace("${sourceName}", get_llm_language_name(source_lang))
        system_prompt = system_prompt.replace("${targetName}", get_llm_language_name(target_lang))

        hub = ClientHub(
            stt=stt,
            llm=llm,
            osc=osc,
            source_language=source_lang,
            target_language=target_lang,
            system_prompt=system_prompt,
            fallback_transcript_only=True,
            translation_enabled=True,
        )

        # Event tracking
        got_stt_final = asyncio.Event()
        got_llm_done = asyncio.Event()
        got_osc_sent = asyncio.Event()

        async def track_events():
            while True:
                try:
                    event = await asyncio.wait_for(hub.ui_events.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                ts = time.perf_counter()

                if event.type == UIEventType.TRANSCRIPT_FINAL:
                    # Only record the FIRST STT final (when LLM translation starts)
                    if metrics.stt_final == 0.0:
                        metrics.stt_final = ts
                        metrics.transcript_text = event.payload.text
                    print(f"  [STT Final] {event.payload.text}")
                    got_stt_final.set()

                elif event.type == UIEventType.TRANSLATION_DONE:
                    metrics.llm_done = ts
                    metrics.translation_text = event.payload.text
                    print(f"  [LLM Done] {event.payload.text}")
                    got_llm_done.set()

                elif event.type == UIEventType.OSC_SENT:
                    metrics.osc_sent = ts
                    print(f"  [OSC Sent] {event.payload.text}")
                    got_osc_sent.set()

                elif event.type == UIEventType.ERROR:
                    print(f"  [ERROR] {event.payload}")

        await hub.start(auto_flush_osc=True)
        event_task = asyncio.create_task(track_events())

        # Wait for STT session to initialize
        await asyncio.sleep(0.5)

        try:
            # Record start time
            metrics.audio_send_start = time.perf_counter()

            # Create utterance ID for this iteration
            utterance_id = uuid4()

            # Split audio into chunks (100ms each = sample_rate // 10 samples)
            chunk_samples = sample_rate // 10  # 100ms worth of samples
            chunks = [
                audio_samples[i : i + chunk_samples]
                for i in range(0, len(audio_samples), chunk_samples)
            ]

            # Send SpeechStart with first chunk (and empty pre_roll)
            if chunks:
                pre_roll = np.zeros(chunk_samples, dtype=np.float32)
                first_chunk = chunks[0]
                await hub.handle_vad_event(
                    SpeechStart(utterance_id, pre_roll=pre_roll, chunk=first_chunk)
                )
                await asyncio.sleep(0.05)

                # Send remaining chunks as SpeechChunk
                for chunk in chunks[1:]:
                    await hub.handle_vad_event(SpeechChunk(utterance_id, chunk=chunk))
                    await asyncio.sleep(0.05)

                # Signal end of speech
                await hub.handle_vad_event(SpeechEnd(utterance_id))

            metrics.audio_send_end = time.perf_counter()
            print(
                f"  Audio sent in {(metrics.audio_send_end - metrics.audio_send_start)*1000:.0f}ms"
            )

            # Wait for pipeline completion (timeout: 15s)
            try:
                await asyncio.wait_for(got_osc_sent.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                print("  [TIMEOUT] Pipeline did not complete in 15s")

            # Force OSC flush
            osc.process_due()

            # Print iteration results
            print(f"  STT Latency:   {metrics.stt_latency_ms:>8.1f} ms")
            print(f"  LLM Latency:   {metrics.llm_latency_ms:>8.1f} ms")
            print(f"  Total Latency: {metrics.total_latency_ms:>8.1f} ms")

        finally:
            event_task.cancel()
            await hub.stop()
            await asyncio.gather(event_task, return_exceptions=True)

        results.append(metrics)

        # Small delay between iterations
        await asyncio.sleep(1.0)

    # Print summary
    print_summary_table(results)

    # Assertions
    successful = [r for r in results if r.total_latency_ms > 0]
    assert len(successful) >= 3, f"At least 3 iterations should succeed, got {len(successful)}"
