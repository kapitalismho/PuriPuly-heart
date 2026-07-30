from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget


@dataclass(frozen=True, slots=True)
class PeerCaptureTargetResolutionService:
    def resolve(
        self,
        *,
        legacy_output_device: str,
        persisted_capture_target: ResolvedDesktopAudioCaptureTarget | None,
    ) -> ResolvedDesktopAudioCaptureTarget:
        if persisted_capture_target is not None:
            if persisted_capture_target.kind == "process":
                return persisted_capture_target
            if persisted_capture_target.device_name == legacy_output_device:
                return persisted_capture_target
            if (
                persisted_capture_target.kind == "default_output_device"
                and not legacy_output_device
            ):
                return persisted_capture_target
        if legacy_output_device:
            return ResolvedDesktopAudioCaptureTarget(
                kind="named_output_device",
                device_name=legacy_output_device,
            )
        return ResolvedDesktopAudioCaptureTarget(kind="default_output_device")


__all__ = ["PeerCaptureTargetResolutionService"]
