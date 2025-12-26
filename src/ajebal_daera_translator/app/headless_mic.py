from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ajebal_daera_translator.app.wiring import (
    create_llm_provider,
    create_secret_store,
    create_stt_backend,
)
from ajebal_daera_translator.config.paths import default_vad_model_path
from ajebal_daera_translator.config.settings import AppSettings
from ajebal_daera_translator.core.audio.format import normalize_audio_f32
from ajebal_daera_translator.core.audio.source import (
    AudioSource,
    SoundDeviceAudioSource,
    resolve_sounddevice_input_device,
)
from ajebal_daera_translator.core.clock import SystemClock
from ajebal_daera_translator.core.orchestrator.hub import ClientHub
from ajebal_daera_translator.core.osc.smart_queue import SmartOscQueue
from ajebal_daera_translator.core.osc.udp_sender import VrchatOscUdpSender
from ajebal_daera_translator.core.stt.controller import ManagedSTTProvider
from ajebal_daera_translator.core.vad.bundled import SILERO_VAD_VERSION, ensure_silero_vad_onnx
from ajebal_daera_translator.core.vad.gating import VadGating
from ajebal_daera_translator.core.vad.silero import SileroVadOnnx

logger = logging.getLogger(__name__)

# Hardcoded STT session reset deadline (not configurable via settings)
STT_RESET_DEADLINE_S = 180.0


@dataclass(slots=True)
class HeadlessMicRunner:
    settings: AppSettings
    config_path: Path
    vad_model_path: Path
    use_llm: bool = True
    clock: SystemClock = SystemClock()

    async def run(self) -> int:
        secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
        llm = create_llm_provider(self.settings, secrets=secrets) if self.use_llm else None

        backend = create_stt_backend(self.settings, secrets=secrets)
        stt = ManagedSTTProvider(
            backend=backend,
            sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
            clock=self.clock,
            reset_deadline_s=STT_RESET_DEADLINE_S,
            drain_timeout_s=self.settings.stt.drain_timeout_s,
            bridging_ms=self.settings.audio.ring_buffer_ms,
        )

        sender = VrchatOscUdpSender(
            host=self.settings.osc.host,
            port=self.settings.osc.port,
            chatbox_address=self.settings.osc.chatbox_address,
            chatbox_send=self.settings.osc.chatbox_send,
            chatbox_clear=self.settings.osc.chatbox_clear,
        )
        osc = SmartOscQueue(
            sender=sender,
            clock=self.clock,
            max_chars=self.settings.osc.chatbox_max_chars,
            cooldown_s=self.settings.osc.cooldown_s,
            ttl_s=self.settings.osc.ttl_s,
        )

        hub = ClientHub(
            stt=stt,
            llm=llm,
            osc=osc,
            clock=self.clock,
            source_language=self.settings.languages.source_language,
            target_language=self.settings.languages.target_language,
            system_prompt=self.settings.system_prompt,
            fallback_transcript_only=not self.use_llm,
        )

        if self.vad_model_path == default_vad_model_path():
            try:
                self.vad_model_path = ensure_silero_vad_onnx(target_path=self.vad_model_path)
            except Exception as exc:
                logger.error("Failed to prepare Silero VAD model (%s): %s", SILERO_VAD_VERSION, exc)
                return 2

        if not self.vad_model_path.exists():
            logger.error("VAD model file not found: %s", self.vad_model_path)
            return 2

        vad = VadGating(
            engine=SileroVadOnnx(model_path=self.vad_model_path),
            sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
            ring_buffer_ms=self.settings.audio.ring_buffer_ms,
            speech_threshold=self.settings.stt.vad_speech_threshold,
        )

        device_idx = None
        with contextlib.suppress(Exception):
            device_idx = resolve_sounddevice_input_device(
                host_api=self.settings.audio.input_host_api,
                device=self.settings.audio.input_device,
            )

        source: AudioSource = SoundDeviceAudioSource(
            sample_rate_hz=None,  # Use device default; resampled later to internal rate
            channels=self.settings.audio.internal_channels,
            device=device_idx,
        )

        await hub.start(auto_flush_osc=True)
        try:
            await run_audio_vad_loop(
                source=source,
                vad=vad,
                hub=hub,
                target_sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
            )
        except KeyboardInterrupt:
            return 0
        finally:
            with contextlib.suppress(Exception):
                await source.close()
            await hub.stop()
            sender.close()

        return 0


async def run_audio_vad_loop(
    *,
    source: AudioSource,
    vad: VadGating,
    hub: ClientHub,
    target_sample_rate_hz: int,
) -> None:
    chunk_samples = vad.chunk_samples
    buffer = np.empty((0,), dtype=np.float32)

    async for frame in source.frames():
        normalized = normalize_audio_f32(
            frame.samples,
            input_sample_rate_hz=frame.sample_rate_hz,
            target_sample_rate_hz=target_sample_rate_hz,
        )
        buffer = np.concatenate([buffer, normalized.samples.reshape(-1)])
        while buffer.size >= chunk_samples:
            chunk = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]
            for ev in vad.process_chunk(chunk):
                await hub.handle_vad_event(ev)
