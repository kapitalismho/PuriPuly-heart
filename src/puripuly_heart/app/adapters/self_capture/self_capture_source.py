from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.config.audio_host_api import InputHostApiProfile
from puripuly_heart.core.audio.source import SelfMicCaptureChannelDecision
from puripuly_heart.core.self_capture import SelfCaptureSessionConfig

SelfCaptureHostApiNormalizer = Callable[[str | None], InputHostApiProfile]
SelfCaptureDeviceResolver = Callable[..., int | None]
SelfCaptureChannelDecision = Callable[..., SelfMicCaptureChannelDecision]
SelfCaptureAudioSourceFactory = Callable[..., object]
SelfCaptureDetailedLog = Callable[..., object]
SelfCaptureSourceWrapper = Callable[[object], object]


@dataclass(frozen=True, slots=True)
class SelfCaptureSourceAdapter:
    normalize_host_api: SelfCaptureHostApiNormalizer
    resolve_device: SelfCaptureDeviceResolver
    channel_decision: SelfCaptureChannelDecision
    source_factory: SelfCaptureAudioSourceFactory
    log_detailed: SelfCaptureDetailedLog
    wrap_source: SelfCaptureSourceWrapper

    def __call__(self, config: SelfCaptureSessionConfig) -> object:
        host_api_profile = self.normalize_host_api(config.input_host_api)
        host_api = host_api_profile.actual_host_api
        first_open_used_wasapi_flags = (
            host_api_profile.wasapi_auto_convert or host_api_profile.wasapi_exclusive
        )
        device_idx = self._resolve_device(host_api, config.input_device)
        source: object | None = None
        try:
            source = self._open_source_with_mono_retry(
                config,
                device_idx,
                attempt="primary",
                host_api_for_log=host_api,
                device_for_log=config.input_device,
                wasapi_auto_convert=host_api_profile.wasapi_auto_convert,
                wasapi_exclusive=host_api_profile.wasapi_exclusive,
            )
            self.log_detailed(
                "[STT] Microphone opened: "
                f"saved_host_api={config.input_host_api!r} "
                f"actual_host_api={host_api!r} "
                f"device={config.input_device!r} "
                f"device_idx={device_idx} "
                f"wasapi_auto_convert={host_api_profile.wasapi_auto_convert} "
                f"wasapi_exclusive={host_api_profile.wasapi_exclusive}"
            )
        except Exception as exc:
            self.log_detailed(
                "[STT] Microphone open detail: "
                f"host_api={host_api!r} device={config.input_device!r} error={exc}",
                level=logging.ERROR,
            )
        if source is None and config.input_device:
            fallback_idx = self._resolve_device("", config.input_device)
            if fallback_idx != device_idx or first_open_used_wasapi_flags:
                try:
                    source = self._open_source_with_mono_retry(
                        config,
                        fallback_idx,
                        attempt="name_fallback",
                        host_api_for_log="",
                        device_for_log=config.input_device,
                    )
                    self.log_detailed(
                        f"[STT] Microphone opened with fallback: device_idx={fallback_idx}"
                    )
                except Exception as exc:
                    self.log_detailed(
                        f"[STT] Fallback microphone detail: error={exc}",
                        level=logging.ERROR,
                    )
        if source is None:
            try:
                source = self._open_source_with_mono_retry(
                    config,
                    None,
                    attempt="system_default",
                    host_api_for_log="",
                    device_for_log="",
                )
                self.log_detailed("[STT] Microphone opened with system default")
            except Exception as exc:
                self.log_detailed(
                    f"[STT] System default microphone detail: error={exc}",
                    level=logging.ERROR,
                )
        if source is None:
            raise RuntimeError("All microphone attempts failed")
        return self.wrap_source(source)

    def _resolve_device(self, host_api: str, device: str) -> int | None:
        try:
            return self.resolve_device(host_api=host_api, device=device)
        except Exception as exc:
            self.log_detailed(
                "[STT] Device resolution detail: "
                f"host_api={host_api!r} device={device!r} error={exc}",
                level=logging.WARNING,
            )
            return None

    def _open_source_with_mono_retry(
        self,
        config: SelfCaptureSessionConfig,
        device_idx: int | None,
        *,
        attempt: str,
        host_api_for_log: str,
        device_for_log: str,
        wasapi_auto_convert: bool = False,
        wasapi_exclusive: bool = False,
    ) -> object:
        decision = self.channel_decision(
            device_idx=device_idx,
            internal_channels=config.internal_channels,
        )
        try:
            return self._open_source_once(
                config,
                device_idx,
                attempt=attempt,
                requested_channels=decision.preferred_capture_channels,
                decision=decision,
                host_api_for_log=host_api_for_log,
                device_for_log=device_for_log,
                wasapi_auto_convert=wasapi_auto_convert,
                wasapi_exclusive=wasapi_exclusive,
            )
        except Exception as exc:
            if decision.preferred_capture_channels <= config.internal_channels:
                raise
            self.log_detailed(
                "[STT] Microphone open detail: "
                f"attempt={attempt!r} "
                f"host_api={host_api_for_log!r} "
                f"device={device_for_log!r} "
                f"device_idx={device_idx} "
                f"preferred_capture_channels={decision.preferred_capture_channels} "
                f"requested_channels={decision.preferred_capture_channels} "
                f"wasapi_auto_convert={wasapi_auto_convert} "
                f"wasapi_exclusive={wasapi_exclusive} "
                f"metadata_status={decision.metadata.metadata_status!r} "
                "will_retry_mono=True "
                f"error={exc}",
                level=logging.WARNING,
            )
            return self._open_source_once(
                config,
                device_idx,
                attempt=f"{attempt}_mono_retry",
                requested_channels=config.internal_channels,
                decision=decision,
                host_api_for_log=host_api_for_log,
                device_for_log=device_for_log,
                wasapi_auto_convert=wasapi_auto_convert,
                wasapi_exclusive=wasapi_exclusive,
            )

    def _open_source_once(
        self,
        config: SelfCaptureSessionConfig,
        device_idx: int | None,
        *,
        attempt: str,
        requested_channels: int,
        decision: SelfMicCaptureChannelDecision,
        host_api_for_log: str,
        device_for_log: str,
        wasapi_auto_convert: bool = False,
        wasapi_exclusive: bool = False,
    ) -> object:
        source = self.source_factory(
            sample_rate_hz=None,
            channels=requested_channels,
            device=device_idx,
            wasapi_auto_convert=wasapi_auto_convert,
            wasapi_exclusive=wasapi_exclusive,
        )
        metadata = decision.metadata
        opened_channels = self._source_int(source, "opened_channels", requested_channels)
        frame_channels = self._source_int(source, "frame_channels", opened_channels)
        actual_sample_rate_hz = self._source_int(source, "actual_sample_rate_hz", 0)
        self.log_detailed(
            "[STT] Microphone capture format: "
            f"attempt={attempt!r} "
            f"internal_channels={decision.internal_channels} "
            f"preferred_capture_channels={decision.preferred_capture_channels} "
            f"requested_channels={requested_channels} "
            f"opened_channels={opened_channels} "
            f"frame_channels={frame_channels} "
            "frame_channels_source='opened_fallback' "
            f"saved_host_api={config.input_host_api!r} "
            f"actual_host_api={host_api_for_log!r} "
            f"device={device_for_log!r} "
            f"device_idx={device_idx} "
            f"wasapi_auto_convert={wasapi_auto_convert} "
            f"wasapi_exclusive={wasapi_exclusive} "
            f"actual_sample_rate_hz={actual_sample_rate_hz or None} "
            f"metadata_device_idx={metadata.device_idx} "
            f"metadata_device_name={metadata.name!r} "
            f"device_max_input_channels={metadata.max_input_channels} "
            f"device_default_samplerate={metadata.default_samplerate} "
            f"metadata_status={metadata.metadata_status!r} "
            f"metadata_error={metadata.metadata_error!r}"
        )
        return source

    @staticmethod
    def _source_int(source: object, attr: str, fallback: int) -> int:
        try:
            return int(getattr(source, attr, fallback))
        except Exception:
            return fallback


__all__ = [
    "SelfCaptureAudioSourceFactory",
    "SelfCaptureChannelDecision",
    "SelfCaptureDetailedLog",
    "SelfCaptureDeviceResolver",
    "SelfCaptureHostApiNormalizer",
    "SelfCaptureSourceAdapter",
    "SelfCaptureSourceWrapper",
]
