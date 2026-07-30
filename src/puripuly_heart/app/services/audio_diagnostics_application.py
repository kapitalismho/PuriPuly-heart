from __future__ import annotations

from dataclasses import dataclass, field

from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.services.application_runtime_logging import (
    ApplicationRuntimeLoggingOwner,
)
from puripuly_heart.app.wiring_capture_runtime import CaptureDiagnosticsAdapter
from puripuly_heart.core.audio.diagnostics import (
    EXPECTED_FAULT_SIGNATURES,
    AudioFaultProfile,
)
from puripuly_heart.core.stt.controller import FinalTranscriptSuppressedNotification

LOCAL_QWEN_HALLUCINATION_GUIDANCE_TRIGGER_COUNT = 2


@dataclass(slots=True)
class AudioDiagnosticsApplicationOwner:
    presentation: UiPresentationPort
    runtime_logging: ApplicationRuntimeLoggingOwner
    capture_fault_profile: str = field(init=False, default="none")
    stt_fault_profile: str = field(init=False, default="none")
    local_qwen_hallucination_detection_count: int = field(init=False, default=0)
    local_qwen_hallucination_modal_shown: bool = field(init=False, default=False)
    _capture_adapter: CaptureDiagnosticsAdapter | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def debug_allowed(self) -> bool:
        return bool(getattr(self.presentation, "debug_ui_preview", False))

    def detailed_enabled(self) -> bool:
        return self.runtime_logging.mode == "detailed"

    def on_final_transcript_suppressed(
        self,
        notification: FinalTranscriptSuppressedNotification,
    ) -> None:
        self.runtime_logging.emit_detailed(
            "[STT][SuppressedFinalNotification] "
            f"provider={notification.stt_provider_name.value} "
            f"channel={notification.channel} "
            f"utterance_id={str(notification.utterance_id)[:8]}"
        )
        if notification.stt_provider_name.value == "local_qwen":
            self.record_local_qwen_hallucination_guidance_detection(notification)

    def record_local_qwen_hallucination_guidance_detection(
        self,
        notification: FinalTranscriptSuppressedNotification,
    ) -> None:
        self.local_qwen_hallucination_detection_count += 1
        count = self.local_qwen_hallucination_detection_count
        self.runtime_logging.emit_detailed(
            "[STT][SuppressedFinalNotification] "
            f"local_qwen_guidance count={count} "
            f"channel={notification.channel} "
            f"modal_shown={self.local_qwen_hallucination_modal_shown}"
        )
        if count < LOCAL_QWEN_HALLUCINATION_GUIDANCE_TRIGGER_COUNT:
            return
        if self.local_qwen_hallucination_modal_shown:
            return
        if not self.presentation.show_local_qwen_hallucination_dialog():
            self.runtime_logging.emit_detailed(
                "[STT][SuppressedFinalNotification] "
                f"local_qwen_guidance count={count} guidance_modal=unavailable"
            )
            return
        self.local_qwen_hallucination_modal_shown = True

    def cycle_capture_fault_profile(self) -> str:
        if not self.debug_allowed():
            return "none"
        profiles = [
            AudioFaultProfile.NONE,
            AudioFaultProfile.CAPTURE_SILENT_FIRST_CHANNEL,
            AudioFaultProfile.CAPTURE_ATTENUATE_40DB,
            AudioFaultProfile.CAPTURE_NEAR_SILENCE_NOISE,
            AudioFaultProfile.CAPTURE_BUFFER_DROPOUTS,
        ]
        current = AudioFaultProfile(self.capture_fault_profile)
        next_profile = profiles[(profiles.index(current) + 1) % len(profiles)]
        self.capture_fault_profile = next_profile.value
        self.runtime_logging.emit_detailed(
            "[AudioDiag][DebugFault] "
            f"capture_profile={next_profile.value} "
            "expected_signature="
            f"{EXPECTED_FAULT_SIGNATURES.get(next_profile.value, 'none')}"
        )
        return self.capture_fault_profile

    def cycle_stt_fault_profile(self) -> str:
        if not self.debug_allowed():
            return "none"
        profiles = [
            AudioFaultProfile.NONE,
            AudioFaultProfile.STT_INPUT_LOW_SNR_VAD_PASS,
        ]
        current = AudioFaultProfile(self.stt_fault_profile)
        next_profile = profiles[(profiles.index(current) + 1) % len(profiles)]
        self.stt_fault_profile = next_profile.value
        self.runtime_logging.emit_detailed(
            "[AudioDiag][DebugFault] "
            f"stt_profile={next_profile.value} "
            "expected_signature="
            f"{EXPECTED_FAULT_SIGNATURES.get(next_profile.value, 'none')}"
        )
        return self.stt_fault_profile

    def clear_fault_profiles(self) -> None:
        self.capture_fault_profile = "none"
        self.stt_fault_profile = "none"
        self.runtime_logging.emit_detailed(
            "[AudioDiag][DebugFault] capture_profile=none stt_profile=none"
        )

    def capture_adapter(self) -> CaptureDiagnosticsAdapter:
        adapter = self._capture_adapter
        if adapter is None:
            adapter = CaptureDiagnosticsAdapter(
                detailed_enabled=self.detailed_enabled,
                debug_allowed=self.debug_allowed,
                capture_fault_profile=lambda: self.capture_fault_profile,
                log_detailed=self.runtime_logging.emit_detailed,
            )
            self._capture_adapter = adapter
        return adapter


__all__ = [
    "AudioDiagnosticsApplicationOwner",
    "LOCAL_QWEN_HALLUCINATION_GUIDANCE_TRIGGER_COUNT",
]
