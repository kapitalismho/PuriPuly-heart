from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.release_evidence import local_cpu_real_decode as evidence


def _write_wav(path: Path, samples: np.ndarray, *, sample_rate_hz: int = 16000) -> None:
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    frames = (pcm * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(sample_rate_hz)
        sink.writeframes(frames)


def test_measured_decode_records_positive_timing_without_transcript() -> None:
    payload = evidence._measured_decode(audio_seconds=6.72, decode_seconds=0.738)

    assert payload["backend"] == "CPU"
    assert payload["result"] == "success"
    assert payload["audio_seconds"] == 6.72
    assert payload["decode_seconds"] == 0.738
    assert payload["rtf"] == pytest.approx(0.738 / 6.72)
    assert "text" not in payload


def test_measured_decode_rejects_non_positive_timing() -> None:
    with pytest.raises(RuntimeError):
        evidence._measured_decode(audio_seconds=0.0, decode_seconds=0.5)
    with pytest.raises(RuntimeError):
        evidence._measured_decode(audio_seconds=1.0, decode_seconds=0.0)


def test_require_nonempty_final_rejects_empty_or_partial_results() -> None:
    evidence._require_nonempty_final(STTBackendTranscriptEvent(text="heard", is_final=True))
    with pytest.raises(RuntimeError):
        evidence._require_nonempty_final(STTBackendTranscriptEvent(text="", is_final=True))
    with pytest.raises(RuntimeError):
        evidence._require_nonempty_final(STTBackendTranscriptEvent(text="partial", is_final=False))


@pytest.mark.asyncio
async def test_decode_case_uses_backend_final_and_measured_timing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_root = tmp_path
    _write_wav(audio_root / "parakeet-v3-en.wav", np.linspace(-0.2, 0.2, 16000, dtype=np.float32))
    sent: list[np.ndarray] = []

    class Session:
        async def send_audio_f32(self, samples: np.ndarray) -> None:
            sent.append(np.asarray(samples, dtype=np.float32))

        async def on_speech_end(self, **_kwargs) -> None:
            return None

        async def events(self):
            yield STTBackendTranscriptEvent(text="hello", is_final=True)

        async def close(self) -> None:
            return None

    class Backend:
        provider_id = "local_parakeet_v3"

        async def open_session(self, **_kwargs) -> Session:
            return Session()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(evidence, "create_local_cpu_backend", lambda *_args, **_kwargs: Backend())
    result = await evidence._decode_case(
        evidence.DECODE_CASES[0],
        model_root=tmp_path / "models",
        audio_root=audio_root,
    )

    assert result["model_id"] == evidence.DECODE_CASES[0].model_id
    assert result["provider_id"] == "local_parakeet_v3"
    assert result["execution_backend"] == "CPU"
    assert result["decode_result"] == {"status": "nonempty_final", "text_length": 5}
    assert "text" not in result["decode_result"]
    assert result["attempt"]["backend"] == "CPU"
    assert result["attempt"]["result"] == "success"
    assert result["attempt"]["audio_seconds"] == pytest.approx(1.0)
    assert result["attempt"]["decode_seconds"] > 0
    assert result["attempt"]["rtf"] == pytest.approx(
        result["attempt"]["decode_seconds"] / result["attempt"]["audio_seconds"]
    )
    assert sent and sent[0].size == 16000


@pytest.mark.asyncio
async def test_decode_case_fails_when_backend_returns_empty_final(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_wav(tmp_path / "parakeet-v3-en.wav", np.ones(1600, dtype=np.float32))

    class Session:
        async def send_audio_f32(self, _samples: np.ndarray) -> None:
            return None

        async def on_speech_end(self, **_kwargs) -> None:
            return None

        async def events(self):
            yield STTBackendTranscriptEvent(text="", is_final=True)

        async def close(self) -> None:
            return None

    class Backend:
        provider_id = "local_parakeet_v3"

        async def open_session(self, **_kwargs) -> Session:
            return Session()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(evidence, "create_local_cpu_backend", lambda *_args, **_kwargs: Backend())
    with pytest.raises(RuntimeError):
        await evidence._decode_case(
            evidence.DECODE_CASES[0],
            model_root=tmp_path / "models",
            audio_root=tmp_path,
        )


@pytest.mark.asyncio
async def test_run_evidence_writes_failed_report_for_missing_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def missing_models(*_args, **_kwargs):
        raise FileNotFoundError("required CPU model root is missing")

    monkeypatch.setattr(
        evidence,
        "inspect_required_cpu_model_installs",
        missing_models,
    )
    report_path = tmp_path / "reports" / "cpu-failure.json"

    exit_code = await evidence.run_evidence(
        model_root=tmp_path / "missing-models",
        audio_root=tmp_path / "missing-audio",
        report_path=report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["failure_type"] == "FileNotFoundError"
    assert report["failure"] == "required CPU model root is missing"
