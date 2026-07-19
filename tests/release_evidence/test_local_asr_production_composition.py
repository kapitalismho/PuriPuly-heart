from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.release_evidence import local_asr_production_composition as evidence


def _final_event(*, channel: str, text: str = "transcript") -> object:
    return SimpleNamespace(
        transcript=SimpleNamespace(
            text=text,
            is_final=True,
            channel=channel,
            final_language_runs=(),
        )
    )


def test_require_final_preserves_channel_evidence() -> None:
    fact = evidence._require_final(
        _final_event(channel="peer"),
        channel="peer",
        stage="peer inference",
    )

    assert fact["text"] == "transcript"
    assert fact["is_final"] is True
    assert fact["channel"] == "peer"


def test_require_final_rejects_cross_channel_result() -> None:
    with pytest.raises(RuntimeError, match="expected 'peer'"):
        evidence._require_final(
            _final_event(channel="self"),
            channel="peer",
            stage="peer inference",
        )


def test_runner_rejects_non_packaged_execution_with_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delattr(evidence.sys, "frozen", raising=False)
    report_path = tmp_path / "report.json"

    result = evidence.run_local_asr_production_composition(
        audio_path=tmp_path / "speech.wav",
        report_path=report_path,
        candidate="candidate-sha",
        expected_gpu_name="RX 7900 XTX",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 1
    assert report["status"] == "failed"
    assert report["candidate"] == "candidate-sha"
    assert report["failure_type"] == "RuntimeError"
    assert "packaged Windows app" in report["failure"]
