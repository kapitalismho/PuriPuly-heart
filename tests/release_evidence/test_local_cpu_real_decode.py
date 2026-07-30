from __future__ import annotations

from puripuly_heart.release_evidence.local_cpu_real_decode import (
    _attempt_payload,
    _evidence_diagnostics_enabled,
)


def test_real_decode_enables_content_safe_attempt_diagnostics() -> None:
    assert _evidence_diagnostics_enabled() is True


def test_attempt_payload_preserves_cpu_timing_without_transcript_content() -> None:
    payload = _attempt_payload(
        "[LocalASR][Attempt] channel=evidence "
        "model=qwen3-asr-0.6b-int8-sherpa backend=CPU "
        "audio_seconds=6.720 decode_seconds=0.738 rtf=0.109821 "
        "result=success queue_wait_seconds=0.000",
        "qwen3-asr-0.6b-int8-sherpa",
    )

    assert payload == {
        "channel": "evidence",
        "model": "qwen3-asr-0.6b-int8-sherpa",
        "backend": "CPU",
        "audio_seconds": 6.72,
        "decode_seconds": 0.738,
        "rtf": 0.109821,
        "result": "success",
        "queue_wait_seconds": 0.0,
    }
