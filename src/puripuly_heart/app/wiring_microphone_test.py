from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCapturePort,
    MicrophoneTestCaptureRequest,
)
from puripuly_heart.app.services.microphone_test import (
    MicrophoneTestSelfCaptureState,
    MicrophoneTestSessionOwner,
    MicrophoneTestSessionRequest,
)
from puripuly_heart.app.wiring_composition import create_microphone_test_capture_adapter
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.core.audio.source import (
    SoundDeviceAudioSource,
    determine_self_mic_capture_channels,
    observe_microphone_test_route,
)
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MicrophoneTestRuntime:
    settings_provider: Callable[[], AppSettings | None]
    self_capture_provider: Callable[[], SelfCaptureSessionOwner | None]
    local_pending_provider: Callable[[], bool]
    disable_self_capture: Callable[[], Awaitable[object]]
    clock: Clock
    log_sink: Callable[[str], None]
    detailed_sink: Callable[[str, int, BaseException | None], None]
    error_sink: Callable[[str], None]
    source_factory: Callable[..., object] = field(default_factory=lambda: SoundDeviceAudioSource)
    _owner: MicrophoneTestSessionOwner | None = field(default=None, init=False, repr=False)

    @property
    def owner_if_created(self) -> MicrophoneTestSessionOwner | None:
        return self._owner

    @property
    def active(self) -> bool:
        return self._owner.active if self._owner is not None else False

    def owner(self) -> MicrophoneTestSessionOwner:
        owner = self._owner
        if owner is None:
            owner = MicrophoneTestSessionOwner(
                capture_port=self.capture_port(),
                capture_request_factory=self.capture_request,
                self_capture_snapshot=self.self_capture_state,
                disable_self_capture=self.disable_self_capture,
                log_sink=self.log_sink,
                diagnostics_sink=self.on_diagnostic,
            )
            self._owner = owner
        return owner

    @staticmethod
    def audio_signature(
        settings: AppSettings | None,
    ) -> tuple[object, ...] | None:
        if settings is None:
            return None
        return (
            settings.audio.input_host_api,
            settings.audio.input_device,
            settings.audio.internal_sample_rate_hz,
            settings.audio.internal_channels,
        )

    def self_capture_state(self) -> MicrophoneTestSelfCaptureState:
        owner = self.self_capture_provider()
        source_open = bool(
            owner is not None
            and (
                owner.loop_task is not None
                or owner.source is not None
                or owner.cleanup_source is not None
            )
        )
        return MicrophoneTestSelfCaptureState(
            stop_required=bool(
                owner is not None
                and owner.snapshot.desired_active
                or self.local_pending_provider()
                or source_open
            ),
            source_open=source_open,
            close_exception=owner.last_cleanup_exception if owner is not None else None,
        )

    async def start(
        self,
        *,
        meter_callback: Callable[[float], object] | None,
        level_log_interval_s: float,
    ) -> bool:
        signature = self.audio_signature(self.settings_provider())
        if signature is None:
            return False
        return await self.owner().start(
            MicrophoneTestSessionRequest(
                audio_signature=signature,
                meter_callback=meter_callback,
                level_log_interval_s=level_log_interval_s,
            )
        )

    async def stop(self) -> None:
        if self._owner is not None:
            await self._owner.stop()

    async def close(self) -> None:
        if self._owner is not None:
            await self._owner.close()

    def capture_port(self) -> MicrophoneTestCapturePort:
        return create_microphone_test_capture_adapter(
            clock=self.clock,
            log_sink=self.log_sink,
            meter_sink=lambda value, callback, generation: self.owner().set_meter_level(
                value,
                callback,
                generation=generation,
            ),
            route_observer=observe_microphone_test_route,
            channel_decision=determine_self_mic_capture_channels,
            source_factory=self.source_factory,
        )

    def capture_request(
        self,
        generation: int | None,
        meter_callback: Callable[[float], object] | None,
        level_log_interval_s: float,
    ) -> MicrophoneTestCaptureRequest:
        settings = self.settings_provider()
        if settings is None:
            raise RuntimeError("Microphone test capture requires settings")
        return MicrophoneTestCaptureRequest(
            saved_host_api=settings.audio.input_host_api,
            requested_device=settings.audio.input_device,
            internal_channels=settings.audio.internal_channels,
            generation=generation,
            meter_callback=meter_callback,
            level_log_interval_s=level_log_interval_s,
        )

    def on_diagnostic(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None,
    ) -> None:
        if event == "session_failed":
            self.error_sink(f"Microphone test error: {exception}")
            return
        if event == "cleanup_retry_failed":
            self.error_sink(f"Microphone test cleanup retry failed: {exception}")
            return
        if event == "meter_callback_failed":
            exc_info = (
                (type(exception), exception, exception.__traceback__)
                if exception is not None
                else None
            )
            logger.debug("Microphone-test meter callback raised", exc_info=exc_info)
            return
        self.detailed_sink(
            f"[MicTest] owner event={event} error_type={metadata.get('error_type')}",
            logging.WARNING,
            exception,
        )


__all__ = ["MicrophoneTestRuntime"]
