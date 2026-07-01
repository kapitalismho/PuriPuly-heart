from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from puripuly_heart.app.wiring import (
    create_llm_provider,
    create_peer_stt_backend,
    create_secret_store,
    create_stt_backend,
)
from puripuly_heart.config.audio_host_api import normalize_input_host_api
from puripuly_heart.config.paths import default_vad_model_path
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.config.vad_defaults import DEFAULT_STABLE_VAD_HANGOVER_MS
from puripuly_heart.core.audio.desktop_pipeline import DesktopPeerPipeline
from puripuly_heart.core.audio.desktop_source import DesktopLoopbackAudioSource
from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.audio.source import (
    AudioSource,
    SoundDeviceAudioSource,
    resolve_sounddevice_input_device,
)
from puripuly_heart.core.audio.streaming_resampler import MonoFirstStreamingResampler
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.osc.receiver import VrcMicState, VrcOscReceiver
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.storage.secrets import SecretStore
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.vad.bundled import SILERO_VAD_VERSION, ensure_silero_vad_onnx
from puripuly_heart.core.vad.gating import VadGating, create_peer_vad_gating
from puripuly_heart.core.vad.silero import SileroVadOnnx
from puripuly_heart.core.vad.sink import VadEventSink

logger = logging.getLogger(__name__)

# Hardcoded STT session reset deadline (not configurable via settings)
STT_RESET_DEADLINE_S = 180.0


class HeadlessMicInitializationError(Exception):
    pass


def _create_headless_llm_provider(*, settings: AppSettings, secrets: SecretStore) -> LLMProvider:
    try:
        return create_llm_provider(settings, secrets=secrets)
    except ValueError as exc:
        raise HeadlessMicInitializationError(
            f"Headless mic LLM initialization failed: {exc}"
        ) from exc


@dataclass(slots=True)
class _HubVadSink:
    hub: ClientHub
    channel: str = "self"

    async def handle_vad_event(self, event) -> None:  # noqa: ANN001
        if self.channel == "peer":
            await self.hub.handle_peer_vad_event(event)
            return
        await self.hub.handle_vad_event(event)


async def _run_peer_loop_with_isolation(coro, *, logger_label: str) -> None:  # noqa: ANN001
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("%s failed: %s", logger_label, exc)


@dataclass(slots=True)
class HeadlessMicRunner:
    settings: AppSettings
    config_path: Path
    vad_model_path: Path
    use_llm: bool = True
    clock: SystemClock = SystemClock()

    async def run(self) -> int:
        secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
        llm = (
            _create_headless_llm_provider(settings=self.settings, secrets=secrets)
            if self.use_llm
            else None
        )

        backend = create_stt_backend(self.settings, secrets=secrets)
        stt = ManagedSTTProvider(
            backend=backend,
            sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
            stt_provider_name=self.settings.provider.stt,
            clock=self.clock,
            reset_deadline_s=STT_RESET_DEADLINE_S,
            drain_timeout_s=self.settings.stt.drain_timeout_s,
            bridging_ms=self.settings.audio.ring_buffer_ms,
        )
        peer_stt = None
        if self.settings.ui.peer_translation_enabled:
            try:
                peer_backend = create_peer_stt_backend(self.settings, secrets=secrets)
                peer_stt = ManagedSTTProvider(
                    backend=peer_backend,
                    sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
                    stt_provider_name=self.settings.provider.peer_stt,
                    channel="peer",
                    clock=self.clock,
                    reset_deadline_s=STT_RESET_DEADLINE_S,
                    drain_timeout_s=self.settings.stt.drain_timeout_s,
                    bridging_ms=max(1, self.settings.desktop_audio.vad_pre_roll_ms),
                )
            except Exception as exc:
                logger.warning("Peer STT backend unavailable: %s", exc)

        sender = VrchatOscUdpSender(
            host=self.settings.osc.host,
            port=self.settings.osc.port,
            chatbox_address=self.settings.osc.chatbox_address,
            chatbox_send=self.settings.osc.chatbox_send,
            chatbox_clear=self.settings.osc.chatbox_clear,
        )
        osc = ChatboxPaginator(
            sender=sender,
            clock=self.clock,
            max_chars=self.settings.osc.chatbox_max_chars,
        )

        hub = ClientHub(
            stt=stt,
            llm=llm,
            osc=osc,
            peer_stt=peer_stt,
            clock=self.clock,
            source_language=self.settings.languages.source_language,
            target_language=self.settings.languages.target_language,
            system_prompt=self.settings.system_prompt,
            fallback_transcript_only=not self.use_llm,
            peer_translation_enabled=self.settings.ui.peer_translation_enabled
            and peer_stt is not None,
            integrated_context_enabled=self.settings.ui.integrated_context_enabled,
            low_latency_mode=self.settings.stt.low_latency_mode,
            low_latency_merge_gap_ms=self.settings.stt.low_latency_merge_gap_ms,
            low_latency_spec_retry_max=self.settings.stt.low_latency_spec_retry_max,
            hangover_s=(
                self.settings.stt.low_latency_vad_hangover_ms / 1000.0
                if self.settings.stt.low_latency_mode
                else DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
            ),
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
            hangover_ms=(
                self.settings.stt.low_latency_vad_hangover_ms
                if self.settings.stt.low_latency_mode
                else DEFAULT_STABLE_VAD_HANGOVER_MS
            ),
        )

        def _resolve_device(host_api: str, device: str) -> int | None:
            try:
                return resolve_sounddevice_input_device(host_api=host_api, device=device)
            except Exception as exc:
                logger.warning(
                    "Device resolution failed (host_api=%r, device=%r): %s", host_api, device, exc
                )
                return None

        def _open_source(
            dev_idx: int | None,
            *,
            wasapi_auto_convert: bool = False,
            wasapi_exclusive: bool = False,
        ) -> SoundDeviceAudioSource:
            return SoundDeviceAudioSource(
                sample_rate_hz=None,
                channels=self.settings.audio.internal_channels,
                device=dev_idx,
                wasapi_auto_convert=wasapi_auto_convert,
                wasapi_exclusive=wasapi_exclusive,
            )

        saved_host_api = self.settings.audio.input_host_api
        host_api_profile = normalize_input_host_api(saved_host_api)
        host_api = host_api_profile.actual_host_api
        first_open_used_wasapi_flags = (
            host_api_profile.wasapi_auto_convert or host_api_profile.wasapi_exclusive
        )
        device_name = self.settings.audio.input_device

        # 1차 시도: 설정된 Host API + 마이크
        device_idx = _resolve_device(host_api, device_name)
        source: AudioSource | None = None

        try:
            source = _open_source(
                device_idx,
                wasapi_auto_convert=host_api_profile.wasapi_auto_convert,
                wasapi_exclusive=host_api_profile.wasapi_exclusive,
            )
            logger.info(
                "Microphone opened "
                "(saved_host_api=%r, actual_host_api=%r, device=%r, device_idx=%s, "
                "wasapi_auto_convert=%s, wasapi_exclusive=%s)",
                saved_host_api,
                host_api,
                device_name,
                device_idx,
                host_api_profile.wasapi_auto_convert,
                host_api_profile.wasapi_exclusive,
            )
        except Exception as exc:
            logger.error(
                "Failed to open microphone (host_api=%r, device=%r): %s", host_api, device_name, exc
            )

        # 2차 시도: Host API 무시, 마이크 이름만
        if source is None and device_name:
            fallback_idx = _resolve_device("", device_name)
            if fallback_idx != device_idx or first_open_used_wasapi_flags:
                try:
                    source = _open_source(
                        fallback_idx,
                        wasapi_auto_convert=False,
                        wasapi_exclusive=False,
                    )
                    logger.info("Microphone opened with fallback (device_idx=%s)", fallback_idx)
                except Exception as exc:
                    logger.error("Fallback microphone failed: %s", exc)

        # 3차 시도: 시스템 기본 장치
        if source is None:
            try:
                source = _open_source(
                    None,
                    wasapi_auto_convert=False,
                    wasapi_exclusive=False,
                )
                logger.info("Microphone opened with system default")
            except Exception as exc:
                logger.error("System default microphone failed: %s", exc)

        if source is None:
            logger.error("All microphone attempts failed")
            return 2

        peer_vad = None
        peer_source = None
        if self.settings.ui.peer_translation_enabled and hub.peer_stt is not None:
            try:
                peer_vad = create_peer_vad_gating(
                    engine=SileroVadOnnx(model_path=self.vad_model_path),
                    sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
                    ring_buffer_ms=self.settings.desktop_audio.vad_pre_roll_ms,
                    speech_threshold=self.settings.desktop_audio.vad_speech_threshold,
                    hangover_ms=self.settings.desktop_audio.vad_hangover_ms,
                )
                peer_source = DesktopPeerPipeline(
                    source=DesktopLoopbackAudioSource(
                        device_name=self.settings.desktop_audio.output_device
                    ),
                    target_sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
                )
            except Exception as exc:
                logger.warning("Desktop peer loop unavailable: %s", exc)
                peer_vad = None
                peer_source = None

        vrc_mic_state = VrcMicState()
        vrc_mic_audio_gate = VrcMicAudioGate(
            state=vrc_mic_state,
            enabled=self.settings.osc.vrc_mic_intercept,
        )
        receiver: VrcOscReceiver | None = None
        if self.settings.osc.vrc_mic_intercept:
            receiver = VrcOscReceiver(state=vrc_mic_state)
            try:
                await receiver.start()
            except OSError as exc:
                logger.warning("VRChat mic sync receiver unavailable: %s", exc)
                receiver = None
            vrc_mic_audio_gate.set_receiver_active(receiver is not None)
            vrc_mic_audio_gate.reset()

        await hub.start(auto_flush_osc=True)
        try:
            loops = [
                run_audio_vad_loop(
                    source=source,
                    vad=vad,
                    sink=_HubVadSink(hub=hub),
                    target_sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
                    audio_gate=vrc_mic_audio_gate,
                )
            ]
            if peer_source is not None and peer_vad is not None:
                loops.append(
                    _run_peer_loop_with_isolation(
                        run_audio_vad_loop(
                            source=peer_source,
                            vad=peer_vad,
                            sink=_HubVadSink(hub=hub, channel="peer"),
                            target_sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
                        ),
                        logger_label="Peer desktop loop",
                    )
                )
            await asyncio.gather(*loops)
        except KeyboardInterrupt:
            return 0
        finally:
            with contextlib.suppress(Exception):
                await source.close()
            if peer_source is not None:
                with contextlib.suppress(Exception):
                    await peer_source.close()
            await hub.stop()
            if receiver is not None:
                receiver.stop()
            sender.close()

        return 0


async def run_audio_vad_loop(
    *,
    source: AudioSource,
    vad: VadGating,
    sink: VadEventSink,
    target_sample_rate_hz: int,
    audio_gate: VrcMicAudioGate | None = None,
    channel_label: str = "self",
    is_detailed_enabled: Callable[[], bool] | None = None,
    log_detailed: Callable[[str], object] | None = None,
) -> None:
    chunk_samples = vad.chunk_samples
    buffer = np.empty((0,), dtype=np.float32)
    resampler: MonoFirstStreamingResampler | None = None
    source_format: tuple[int, int] | None = None
    gate_gated_audio_ms = 0.0
    gate_passed_audio_ms = 0.0
    gate_log_accumulated_ms = 0.0
    vad_input_accumulated_audio_ms = 0.0
    vad_input_last_log_ms = 0.0
    vad_input_was_near_silent: bool | None = None

    def _diagnostics_enabled() -> bool:
        if is_detailed_enabled is None or log_detailed is None:
            return False
        with contextlib.suppress(Exception):
            return bool(is_detailed_enabled())
        return False

    def _log_detailed_best_effort(message: str) -> None:
        if log_detailed is None:
            return
        with contextlib.suppress(Exception):
            log_detailed(message)

    async def _process_buffered_chunks() -> None:
        nonlocal buffer, gate_gated_audio_ms, gate_passed_audio_ms, gate_log_accumulated_ms
        while buffer.size >= chunk_samples:
            chunk = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]
            original_chunk = chunk
            if audio_gate is not None:
                chunk = audio_gate.process_chunk(chunk)
                if _diagnostics_enabled():
                    with contextlib.suppress(Exception):
                        chunk_ms = chunk.size * 1000.0 / float(target_sample_rate_hz)
                        gate_log_accumulated_ms += chunk_ms
                        if np.any(original_chunk) and not np.any(chunk):
                            gate_gated_audio_ms += chunk_ms
                        else:
                            gate_passed_audio_ms += chunk_ms
                        if gate_log_accumulated_ms >= 1000.0 and (
                            gate_gated_audio_ms > 0.0 or gate_log_accumulated_ms >= 10000.0
                        ):
                            _log_detailed_best_effort(
                                f"[AudioDiag][Gate][{channel_label}] "
                                f"enabled={audio_gate.enabled} "
                                f"receiver_active={audio_gate.receiver_active} "
                                f"gated_audio_ms={gate_gated_audio_ms:.1f} "
                                f"passed_audio_ms={gate_passed_audio_ms:.1f}"
                            )
                            gate_log_accumulated_ms = 0.0
                            gate_gated_audio_ms = 0.0
                            gate_passed_audio_ms = 0.0
            for ev in vad.process_chunk(chunk):
                await sink.handle_vad_event(ev)

    async for frame in source.frames():
        frame_format = (frame.sample_rate_hz, frame.channels)
        if source_format is None:
            source_format = frame_format
            resampler = MonoFirstStreamingResampler(
                input_sample_rate_hz=frame.sample_rate_hz,
                output_sample_rate_hz=target_sample_rate_hz,
                input_channels=frame.channels,
            )
        elif frame_format != source_format:
            raise ValueError(
                "source audio format changed during streaming: "
                f"expected {source_format[0]}Hz/{source_format[1]}ch, "
                f"got {frame.sample_rate_hz}Hz/{frame.channels}ch"
            )

        assert resampler is not None
        normalized = resampler.resample_chunk(frame.samples)
        if normalized.size:
            if _diagnostics_enabled():
                with contextlib.suppress(Exception):
                    vad_input_frame = AudioFrameF32(
                        samples=normalized.reshape(-1),
                        sample_rate_hz=target_sample_rate_hz,
                        channels=1,
                    )
                    vad_input_metrics = compute_audio_frame_metrics(vad_input_frame)
                    vad_input_accumulated_audio_ms += vad_input_metrics.audio_ms
                    near_silent = (
                        vad_input_metrics.rms_db <= -60.0 and vad_input_metrics.zero_ratio >= 0.98
                    )
                    should_log_vad_input = vad_input_last_log_ms <= 0.0
                    if vad_input_was_near_silent is not None:
                        should_log_vad_input = should_log_vad_input or (
                            near_silent != vad_input_was_near_silent
                        )
                    should_log_vad_input = should_log_vad_input or (
                        near_silent
                        and vad_input_accumulated_audio_ms - vad_input_last_log_ms >= 3000.0
                    )
                    should_log_vad_input = should_log_vad_input or (
                        vad_input_accumulated_audio_ms - vad_input_last_log_ms >= 10000.0
                    )
                    if should_log_vad_input:
                        vad_input_last_log_ms = vad_input_accumulated_audio_ms
                        vad_input_was_near_silent = near_silent
                        _log_detailed_best_effort(
                            f"[AudioDiag][VADInput][{channel_label}] "
                            f"source_rate={frame.sample_rate_hz} "
                            f"source_channels={frame.channels} "
                            f"target_rate={target_sample_rate_hz} "
                            f"samples={vad_input_metrics.samples} "
                            f"audio_ms={vad_input_metrics.audio_ms:.1f} "
                            f"rms_db={vad_input_metrics.rms_db:.1f} "
                            f"peak_db={vad_input_metrics.peak_db:.1f} "
                            f"zero_ratio={vad_input_metrics.zero_ratio:.3f}"
                        )
            buffer = np.concatenate([buffer, normalized.reshape(-1)])
            await _process_buffered_chunks()

    if resampler is None:
        return

    tail = resampler.flush()
    if tail.size:
        buffer = np.concatenate([buffer, tail.reshape(-1)])
    await _process_buffered_chunks()
