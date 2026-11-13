from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from puripuly_heart.config.audio_host_api import InputHostApiProfile
from puripuly_heart.core.audio.source import AudioSourceOpenFailure, SelfMicCaptureChannelDecision
from puripuly_heart.core.diagnostics import current_diagnostic_operation, emit_failure, exception_diagnostics
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.self_capture import SelfCaptureSessionConfig

SelfCaptureHostApiNormalizer = Callable[[str | None], InputHostApiProfile]
SelfCaptureDeviceResolver = Callable[..., int | None]
SelfCaptureChannelDecision = Callable[..., SelfMicCaptureChannelDecision]
SelfCaptureAudioSourceFactory = Callable[..., object]
SelfCaptureDiagnosticLog = Callable[..., object]
SelfCaptureSourceWrapper = Callable[[object], object]


class MicrophoneOpenError(RuntimeError):
    def __init__(self, failures: list[ErrorDiagnostics]) -> None:
        super().__init__("All microphone attempts failed")
        self.failure_diagnostics = ErrorDiagnostics(
            component="capture.self",
            operation="source_open",
            code="capture.self.source_open_failed",
            category="lifecycle",
            visibility="basic",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={"attempts": len(failures)},
            causes=tuple(failures),
            channel="self",
            **current_diagnostic_operation(),
        )


@dataclass(slots=True)
class SelfCaptureSourceAdapter:
    normalize_host_api: SelfCaptureHostApiNormalizer
    resolve_device: SelfCaptureDeviceResolver
    channel_decision: SelfCaptureChannelDecision
    source_factory: SelfCaptureAudioSourceFactory
    log_diagnostic: SelfCaptureDiagnosticLog
    wrap_source: SelfCaptureSourceWrapper
    last_open_diagnostics: ErrorDiagnostics | None = field(default=None, init=False)
    last_open_recovered: bool = field(default=False, init=False)
    diagnostic_delivery_failure: ErrorDiagnostics | None = field(default=None, init=False)

    def __call__(self, config: SelfCaptureSessionConfig) -> object:
        self.last_open_diagnostics = None
        self.last_open_recovered = False
        host_api_profile = self.normalize_host_api(config.input_host_api)
        host_api = host_api_profile.actual_host_api
        first_open_used_wasapi_flags = (
            host_api_profile.wasapi_auto_convert or host_api_profile.wasapi_exclusive
        )
        failures: list[ErrorDiagnostics] = []
        device_idx = self._resolve_device(host_api, config.input_device, failures)
        source: object | None = None
        try:
            source = self._open_source_with_mono_retry(
                config,
                device_idx,
                failures=failures,
                attempt="primary",
                host_api_for_log=host_api,
                device_for_log=config.input_device,
                wasapi_auto_convert=host_api_profile.wasapi_auto_convert,
                wasapi_exclusive=host_api_profile.wasapi_exclusive,
            )
            self._log(
                "[STT] Microphone opened: "
                f"saved_host_api={config.input_host_api!r} "
                f"actual_host_api={host_api!r} "
                f"device={config.input_device!r} "
                f"device_idx={device_idx} "
                f"wasapi_auto_convert={host_api_profile.wasapi_auto_convert} "
                f"wasapi_exclusive={host_api_profile.wasapi_exclusive}"
            )
        except AudioSourceOpenFailure as exc:
            self.last_open_diagnostics = exc.failure_diagnostics
            raise
        except Exception as exc:
            self._log(
                f"[STT] Microphone open failed cause={type(exc).__name__}",
                level=logging.ERROR,
            )
        if source is None and config.input_device:
            fallback_idx = self._resolve_device("", config.input_device, failures)
            if fallback_idx != device_idx or first_open_used_wasapi_flags:
                try:
                    source = self._open_source_with_mono_retry(
                        config,
                        fallback_idx,
                        failures=failures,
                        attempt="name_fallback",
                        host_api_for_log="",
                        device_for_log=config.input_device,
                    )
                    self._log(
                        f"[STT] Microphone opened with fallback: device_idx={fallback_idx}"
                    )
                except AudioSourceOpenFailure as exc:
                    self.last_open_diagnostics = exc.failure_diagnostics
                    raise
                except Exception as exc:
                    self._log(
                        f"[STT] Fallback microphone failed cause={type(exc).__name__}",
                        level=logging.ERROR,
                    )
        if source is None:
            try:
                source = self._open_source_with_mono_retry(
                    config,
                    None,
                    failures=failures,
                    attempt="system_default",
                    host_api_for_log="",
                    device_for_log="",
                )
                self._log("[STT] Microphone opened with system default")
            except AudioSourceOpenFailure as exc:
                self.last_open_diagnostics = exc.failure_diagnostics
                raise
            except Exception as exc:
                self._log(
                    f"[STT] System default microphone failed cause={type(exc).__name__}",
                    level=logging.ERROR,
                )
        if source is None:
            failure = MicrophoneOpenError(failures)
            self.last_open_diagnostics = failure.failure_diagnostics
            raise failure
        if failures:
            self.last_open_diagnostics = MicrophoneOpenError(failures).failure_diagnostics
            self.last_open_recovered = True
            emit_failure(
                logging.getLogger(__name__),
                self.last_open_diagnostics,
                severity="warning",
                disposition="recovered",
                stage="source_open",
            )
        return self.wrap_source(source)

    def _resolve_device(
        self, host_api: str, device: str, failures: list[ErrorDiagnostics]
    ) -> int | None:
        try:
            return self.resolve_device(host_api=host_api, device=device)
        except Exception as exc:
            failures.append(self._cause(exc, attempt="device_resolution"))
            self._log(
                f"[STT] Device resolution failed cause={type(exc).__name__}",
                level=logging.WARNING,
            )
            return None

    def _open_source_with_mono_retry(
        self,
        config: SelfCaptureSessionConfig,
        device_idx: int | None,
        *,
        failures: list[ErrorDiagnostics],
        attempt: str,
        host_api_for_log: str,
        device_for_log: str,
        wasapi_auto_convert: bool = False,
        wasapi_exclusive: bool = False,
    ) -> object:
        try:
            decision = self.channel_decision(
                device_idx=device_idx,
                internal_channels=config.internal_channels,
            )
        except Exception as exc:
            failures.append(self._cause(exc, attempt=attempt))
            raise
        if decision.metadata.failure_diagnostics is not None:
            failures.append(decision.metadata.failure_diagnostics)
        try:
            return self._open_source_once(
                config,
                device_idx,
                failures=failures,
                attempt=attempt,
                requested_channels=decision.preferred_capture_channels,
                decision=decision,
                host_api_for_log=host_api_for_log,
                device_for_log=device_for_log,
                wasapi_auto_convert=wasapi_auto_convert,
                wasapi_exclusive=wasapi_exclusive,
            )
        except AudioSourceOpenFailure:
            raise
        except Exception as exc:
            if decision.preferred_capture_channels <= config.internal_channels:
                raise
            self._log(
                "[STT] Microphone channel open failed "
                f"attempt={attempt!r} will_retry_mono=True "
                f"cause={type(exc).__name__}",
                level=logging.WARNING,
            )
            return self._open_source_once(
                config,
                device_idx,
                failures=failures,
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
        failures: list[ErrorDiagnostics],
        attempt: str,
        requested_channels: int,
        decision: SelfMicCaptureChannelDecision,
        host_api_for_log: str,
        device_for_log: str,
        wasapi_auto_convert: bool = False,
        wasapi_exclusive: bool = False,
    ) -> object:
        try:
            source = self.source_factory(
                sample_rate_hz=None,
                channels=requested_channels,
                device=device_idx,
                wasapi_auto_convert=wasapi_auto_convert,
                wasapi_exclusive=wasapi_exclusive,
            )
        except AudioSourceOpenFailure as exc:
            exc.failure_diagnostics = MicrophoneOpenError(
                [*failures, self._cause(exc.failure, attempt=attempt)]
            ).failure_diagnostics
            raise
        except Exception as exc:
            failures.append(self._cause(exc, attempt=attempt))
            raise
        metadata = decision.metadata
        opened_channels = self._source_int(source, "opened_channels", requested_channels)
        frame_channels = self._source_int(source, "frame_channels", opened_channels)
        actual_sample_rate_hz = self._source_int(source, "actual_sample_rate_hz", 0)
        self._log(
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
            f"metadata_status={metadata.metadata_status!r}"
        )
        return source

    @staticmethod
    def _cause(exception: BaseException, *, attempt: str) -> ErrorDiagnostics:
        return exception_diagnostics(
            exception,
            component="capture.self",
            operation="source_open",
            code="capture.self.source_open_failed",
            fields={"capture_attempt": attempt},
        )

    def _log(self, message: str, *, level: int = logging.INFO) -> None:
        try:
            self.log_diagnostic(message, level=level)
        except Exception as exc:
            self.diagnostic_delivery_failure = exception_diagnostics(
                exc, component="capture.self", operation="observe",
                code="capture.self.diagnostic_sink_failed",
            )

    @staticmethod
    def _source_int(source: object, attr: str, fallback: int) -> int:
        try:
            return int(getattr(source, attr, fallback))
        except Exception:
            return fallback


__all__ = [
    "SelfCaptureAudioSourceFactory",
    "SelfCaptureChannelDecision",
    "SelfCaptureDiagnosticLog",
    "SelfCaptureDeviceResolver",
    "SelfCaptureHostApiNormalizer",
    "SelfCaptureSourceAdapter",
    "SelfCaptureSourceWrapper",
]
