from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from puripuly_heart.app.ports.audio_test import AudioDiagnosticCandidate
from puripuly_heart.app.services.application_audio_devices import safe_audio_device_label

if TYPE_CHECKING:
    from puripuly_heart.core.audio.audio_test_vad import AudioTestVadSample

AUDIO_DIAGNOSE_MAX_CANDIDATES = 4
AUDIO_DIAGNOSE_CANDIDATE_TIMEOUT_S = 3.0


def audio_diagnosis_receipt(
    *, input_host_api: str, input_device: str, failure_code: str | None = None
) -> dict[str, object]:
    return {
        "status": "action_required",
        "recognition": "not_exercised",
        "settings_changed": False,
        "diagnosis": {
            "state": "unavailable",
            "scope": "local_capture_vad",
            "current_route": {
                "input_host_api": safe_audio_device_label(input_host_api),
                "input_device": safe_audio_device_label(input_device),
            },
            "candidates": [],
            "selected_candidate_id": None,
            "selection_basis": None,
            "cleanup_complete": True,
            "restoration": {"self": "not_needed", "peer": "not_needed"},
            "failure_code": failure_code,
            "bounds": {
                "max_candidates": AUDIO_DIAGNOSE_MAX_CANDIDATES,
                "candidate_timeout_s": AUDIO_DIAGNOSE_CANDIDATE_TIMEOUT_S,
            },
        },
    }


@dataclass(slots=True)
class AudioDiagnosticEvidence:
    candidate: AudioDiagnosticCandidate
    opened: bool = False
    frame_count: int = 0
    vad_sample_count: int = 0
    silence_observed: bool = False
    speech_detected: bool = False
    peak_level: float = 0.0
    speech_probability_min: float | None = None
    speech_probability_max: float | None = None
    speech_threshold: float | None = None
    observed_speech_probabilities: list[float] = field(default_factory=list)
    elapsed_ms: int = 0
    failure_code: str | None = None
    cleanup_complete: bool = True

    def observe(self, samples: tuple[AudioTestVadSample, ...]) -> None:
        self.vad_sample_count += len(samples)
        for sample in samples:
            if not math.isfinite(sample.probability) or not 0.0 <= sample.probability <= 1.0:
                raise ValueError("invalid_vad_probability")
            self.silence_observed |= sample.level <= 0.001
            self.speech_detected |= sample.speech
            self.peak_level = max(self.peak_level, sample.level)
            self.speech_probability_min = min(
                (
                    self.speech_probability_min
                    if self.speech_probability_min is not None
                    else sample.probability
                ),
                sample.probability,
            )
            self.speech_probability_max = max(
                (
                    self.speech_probability_max
                    if self.speech_probability_max is not None
                    else sample.probability
                ),
                sample.probability,
            )
            self.speech_threshold = sample.threshold
            if (
                len(self.observed_speech_probabilities) < 16
                and sample.probability not in self.observed_speech_probabilities
            ):
                self.observed_speech_probabilities.append(sample.probability)

    @property
    def selection_rank(self) -> int:
        if self.failure_code is not None or not self.cleanup_complete:
            return 0
        if self.speech_detected:
            return 3
        if self.frame_count:
            return 2
        return int(self.opened)

    def snapshot(self) -> dict[str, object]:
        candidate = self.candidate
        return {
            "id": f"input-{candidate.device_index}",
            "input_host_api": safe_audio_device_label(candidate.input_host_api),
            "input_device": safe_audio_device_label(candidate.input_device),
            "device_index": candidate.device_index,
            "current_route": candidate.current_route,
            "opened": self.opened,
            "frames_received": self.frame_count > 0,
            "frame_count": self.frame_count,
            "vad_sample_count": self.vad_sample_count,
            "silence_observed": self.silence_observed,
            "speech_detected": self.speech_detected,
            "peak_level": self.peak_level,
            "speech_probability_min": self.speech_probability_min,
            "speech_probability_max": self.speech_probability_max,
            "speech_threshold": self.speech_threshold,
            "observed_speech_probabilities": list(self.observed_speech_probabilities),
            "elapsed_ms": self.elapsed_ms,
            "failure_code": self.failure_code,
            "cleanup_complete": self.cleanup_complete,
        }
