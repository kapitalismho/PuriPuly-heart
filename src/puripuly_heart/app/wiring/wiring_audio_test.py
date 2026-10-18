from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.app.adapters.audio_test_sources import (
    LoopbackAudioTestSourceOpener,
    MicrophoneAudioTestSourceOpener,
)
from puripuly_heart.app.ports.audio_test import (
    AudioTestCaptureControlPort,
    AudioTestCaptureState,
    AudioTestChannel,
)
from puripuly_heart.app.services.audio_test import (
    AudioTestChannelPlan,
    AudioTestObservationCallback,
    AudioTestSessionOwner,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio.audio_test_vad import AudioTestVadProbe, AudioTestVadSettings
from puripuly_heart.core.audio.source import (
    SoundDeviceAudioSource,
    determine_self_mic_capture_channels,
    observe_microphone_test_route,
)
from puripuly_heart.core.peer_capture import (
    PeerCaptureResolvedTarget,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolverPort,
)
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.vad.gating import (
    PEER_VAD_START_COMMIT_CHUNKS,
    PEER_VAD_START_DEBOUNCE_CHUNKS,
)

from .wiring_composition import (
    create_audio_test_capture_adapter,
    create_audio_test_vad_engine,
    create_peer_capture_source_adapter,
    create_peer_capture_target_resolver_adapter,
)
from .wiring_stt_factory import (
    build_peer_capture_target_from_vnext,
    self_vad_hangover_ms_from_vnext,
)

logger = logging.getLogger(__name__)

MICROPHONE_FRAME_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class AudioTestAudioSettings:
    input_host_api: str
    input_device: str
    internal_channels: int
    capture_target: PeerCaptureTargetIntent
    output_device: str

    def signature(self) -> tuple[object, ...]:
        return (
            self.input_host_api,
            self.input_device,
            self.internal_channels,
            self.capture_target,
            self.output_device,
        )


def audio_test_settings_from_vnext(settings: AppSettingsVNext) -> AudioTestAudioSettings:
    return AudioTestAudioSettings(
        input_host_api=settings.intent.audio.input_host_api,
        input_device=settings.intent.audio.input_device,
        internal_channels=1,
        capture_target=build_peer_capture_target_from_vnext(settings),
        output_device=settings.intent.desktop_audio.output_device,
    )


def audio_test_vad_settings_from_vnext(
    settings: AppSettingsVNext,
    channel: AudioTestChannel,
) -> AudioTestVadSettings:
    if channel == "peer":
        desktop_audio = settings.intent.desktop_audio
        return AudioTestVadSettings(
            speech_threshold=desktop_audio.vad_speech_threshold,
            hangover_ms=desktop_audio.vad_hangover_ms,
            ring_buffer_ms=desktop_audio.vad_pre_roll_ms,
            start_debounce_chunks=PEER_VAD_START_DEBOUNCE_CHUNKS,
            start_commit_chunks=PEER_VAD_START_COMMIT_CHUNKS,
        )
    return AudioTestVadSettings(
        speech_threshold=settings.intent.stt.vad_speech_threshold,
        hangover_ms=self_vad_hangover_ms_from_vnext(settings),
        ring_buffer_ms=settings.intent.audio.ring_buffer_ms,
    )


@dataclass(slots=True)
class AudioTestCaptureControl:
    self_capture_provider: Callable[[], SelfCaptureSessionOwner | None]
    local_pending_provider: Callable[[], bool]
    set_self_enabled: Callable[[bool], Awaitable[object]]
    peer_owner_provider: Callable[[], object | None]
    peer_capture_provider: Callable[[], object | None]
    set_peer_enabled: Callable[[bool], Awaitable[object]]

    def state(self, channel: AudioTestChannel) -> AudioTestCaptureState:
        return self._peer_state() if channel == "peer" else self._self_state()

    async def set_enabled(self, channel: AudioTestChannel, enabled: bool) -> None:
        if channel == "peer":
            await self.set_peer_enabled(enabled)
        else:
            await self.set_self_enabled(enabled)

    def _self_state(self) -> AudioTestCaptureState:
        owner = self.self_capture_provider()
        snapshot = owner.snapshot if owner is not None else None
        enabled = bool(snapshot is not None and snapshot.desired_active)
        source_open = bool(
            owner is not None
            and (
                owner.loop_task is not None
                or owner.source is not None
                or owner.cleanup_source is not None
            )
        )
        return AudioTestCaptureState(
            enabled=enabled,
            busy=enabled or self.local_pending_provider() or source_open,
            source_open=source_open,
            token=(id(owner), owner),
            close_exception=owner.last_cleanup_exception if owner is not None else None,
        )

    def _peer_state(self) -> AudioTestCaptureState:
        owner = self.peer_owner_provider()
        snapshot = owner.snapshot() if owner is not None else None
        enabled = bool(snapshot is not None and snapshot.intent_enabled)
        capture = self.peer_capture_provider()
        capture_snapshot = capture.snapshot if capture is not None else None
        source_open = bool(
            capture_snapshot is not None
            and (capture_snapshot.has_source or capture_snapshot.has_loop_task)
        )
        return AudioTestCaptureState(
            enabled=enabled,
            busy=enabled or source_open,
            source_open=source_open,
            token=(id(owner), owner),
        )


def _idle_snapshot() -> dict[str, object]:
    channel = {
        "state": "off",
        "failure_reason": None,
        "failure_type": None,
        "device": {},
        "meter_level": 0.0,
        "speech_probability": 0.0,
        "speech_detected": False,
        "speech_threshold": None,
    }
    return {
        "state": "off",
        "desired_active": False,
        "effective_active": False,
        "channels": {"self": dict(channel), "peer": dict(channel)},
        "restore_capture": [],
    }


@dataclass(slots=True)
class AudioTestRuntime:
    settings_provider: Callable[[], AppSettingsVNext | None]
    capture_control: AudioTestCaptureControlPort
    target_resolver: PeerCaptureTargetResolverPort
    loopback_source_factory: Callable[[PeerCaptureResolvedTarget, str], object]
    log_sink: Callable[[str], None]
    diagnostic_sink: Callable[[str, int, BaseException | None], None]
    error_sink: Callable[[str], None]
    microphone_source_factory: Callable[..., object] = field(
        default_factory=lambda: SoundDeviceAudioSource
    )
    vad_engine_factory: Callable[[], object] = create_audio_test_vad_engine
    _owner: AudioTestSessionOwner | None = field(default=None, init=False, repr=False)

    @property
    def owner_if_created(self) -> AudioTestSessionOwner | None:
        return self._owner

    @property
    def active(self) -> bool:
        return self._owner.active if self._owner is not None else False

    @property
    def snapshot(self) -> dict[str, object]:
        if self._owner is None:
            return _idle_snapshot()
        return self._owner.snapshot

    async def wait_ready(self) -> bool:
        return await self._owner.wait_ready() if self._owner is not None else False

    def owner(self) -> AudioTestSessionOwner:
        owner = self._owner
        if owner is None:
            owner = AudioTestSessionOwner(
                capture_port=create_audio_test_capture_adapter(log_sink=self.log_sink),
                plan_factory=self.plan,
                vad_settings=self.vad_settings,
                capture_control=self.capture_control,
                log_sink=self.log_sink,
                diagnostics_sink=self.on_diagnostic,
            )
            self._owner = owner
        return owner

    @staticmethod
    def audio_signature(settings: AppSettingsVNext | None) -> tuple[object, ...] | None:
        if settings is None:
            return None
        return audio_test_settings_from_vnext(settings).signature()

    async def start(
        self,
        *,
        observation_callback: AudioTestObservationCallback | None = None,
    ) -> bool:
        if self.settings_provider() is None:
            return False
        return await self.owner().start(observation_callback=observation_callback)

    async def stop(self) -> None:
        if self._owner is not None:
            await self._owner.stop()

    async def close(self) -> None:
        if self._owner is not None:
            await self._owner.close()

    def release_capture_restore(self, channel: AudioTestChannel) -> None:
        if self._owner is not None:
            self._owner.release_capture_restore(channel)

    def vad_settings(self, channel: AudioTestChannel) -> AudioTestVadSettings:
        return audio_test_vad_settings_from_vnext(self._settings(), channel)

    def plan(self, channel: AudioTestChannel) -> AudioTestChannelPlan:
        audio = audio_test_settings_from_vnext(self._settings())

        def probe_factory(settings: AudioTestVadSettings) -> AudioTestVadProbe:
            return AudioTestVadProbe(engine=self.vad_engine_factory(), settings=settings)

        if channel == "peer":
            return AudioTestChannelPlan(
                opener=LoopbackAudioTestSourceOpener(
                    target=audio.capture_target,
                    output_device=audio.output_device,
                    target_resolver=self.target_resolver,
                    source_factory=self.loopback_source_factory,
                ),
                probe_factory=probe_factory,
                ready_on_open=True,
            )
        return AudioTestChannelPlan(
            opener=MicrophoneAudioTestSourceOpener(
                saved_host_api=audio.input_host_api,
                requested_device=audio.input_device,
                internal_channels=audio.internal_channels,
                route_observer=observe_microphone_test_route,
                channel_decision=determine_self_mic_capture_channels,
                source_factory=self.microphone_source_factory,
                log_sink=self.log_sink,
            ),
            probe_factory=probe_factory,
            frame_timeout_s=MICROPHONE_FRAME_TIMEOUT_S,
        )

    def on_diagnostic(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None,
    ) -> None:
        channel = metadata.get("channel")
        if event == "session_failed":
            self.error_sink(f"Audio test error channel={channel}: {exception}")
            return
        if event == "cleanup_retry_failed":
            self.error_sink(f"Audio test cleanup retry failed: {exception}")
            return
        if event == "observation_callback_failed":
            exc_info = (
                (type(exception), exception, exception.__traceback__)
                if exception is not None
                else None
            )
            logger.debug("Audio-test observation callback raised", exc_info=exc_info)
            return
        error_type = type(exception).__name__ if exception is not None else None
        self.diagnostic_sink(
            f"[AudioTest] owner event={event} channel={channel} error_type={error_type}",
            logging.WARNING,
            exception,
        )

    def _settings(self) -> AppSettingsVNext:
        settings = self.settings_provider()
        if settings is None:
            raise RuntimeError("Audio test requires loaded settings")
        return settings


def compose_audio_test_runtime(
    *,
    settings_provider: Callable[[], AppSettingsVNext | None],
    capture_control: AudioTestCaptureControlPort,
    log_sink: Callable[[str], None],
    log_diagnostic: Callable[[str], object],
    diagnostic_sink: Callable[[str, int, BaseException | None], None],
    error_sink: Callable[[str], None],
) -> AudioTestRuntime:
    source_adapter = create_peer_capture_source_adapter(
        log_diagnostic=log_diagnostic,
        wrap_source=lambda source: source,
    )

    def open_loopback_source(target: PeerCaptureResolvedTarget, output_device: str) -> object:
        return source_adapter.open_raw_source(target, output_device=output_device)

    return AudioTestRuntime(
        settings_provider=settings_provider,
        capture_control=capture_control,
        target_resolver=create_peer_capture_target_resolver_adapter(),
        loopback_source_factory=open_loopback_source,
        log_sink=log_sink,
        diagnostic_sink=diagnostic_sink,
        error_sink=error_sink,
    )


__all__ = [
    "AudioTestAudioSettings",
    "AudioTestCaptureControl",
    "AudioTestRuntime",
    "audio_test_settings_from_vnext",
    "audio_test_vad_settings_from_vnext",
    "compose_audio_test_runtime",
]
