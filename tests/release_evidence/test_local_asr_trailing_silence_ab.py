from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from puripuly_heart.release_evidence.local_asr_trailing_silence_ab import (
    EXPECTED_TRIMMED_MS,
    REPORTED_TRAILING_SILENCE_MS,
    SAFETY_TAIL_MS,
    _authority_identity,
    _close_resources,
    _comparison,
    _ObservedDecode,
    _samples_sha256,
    _summarize_pairs,
    _text_payload,
    _trim_payload,
    _with_fixed_trailing_silence,
    run_evidence,
)


class _RecordingClose:
    def __init__(
        self,
        events: list[str],
        name: str,
        error: BaseException | None = None,
    ) -> None:
        self._events = events
        self._name = name
        self._error = error

    async def close(self) -> None:
        self._events.append(self._name)
        if self._error is not None:
            raise self._error


def _result(
    text: str,
    *,
    submitted_audio_seconds: float,
    decode_seconds: float,
    queue_wait_seconds: float,
) -> _ObservedDecode:
    return _ObservedDecode(
        text=text,
        payload={
            "transcript": _text_payload(text),
            "attempt": {
                "decode_seconds": decode_seconds,
                "queue_wait_seconds": queue_wait_seconds,
            },
            "trim": {"submitted_audio_seconds": submitted_audio_seconds},
        },
    )


def test_fixed_input_appends_exact_reported_silence() -> None:
    samples = np.arange(16_000, dtype=np.float32)

    fixed = _with_fixed_trailing_silence(samples)

    assert REPORTED_TRAILING_SILENCE_MS == 400
    assert SAFETY_TAIL_MS == 128
    assert EXPECTED_TRIMMED_MS == 272
    assert fixed.size == 22_400
    assert np.array_equal(fixed[:16_000], samples)
    assert np.count_nonzero(fixed[16_000:]) == 0
    assert _samples_sha256(fixed) == _samples_sha256(fixed.copy())


def test_trim_payload_parses_content_safe_required_durations() -> None:
    payload = _trim_payload(
        "[LocalASR][Trim] channel=evidence model=model-id backend=CPU "
        "audio_before_seconds=1.400 reported_trailing_silence_seconds=0.400 "
        "actual_trimmed_seconds=0.272 submitted_audio_seconds=1.128",
        "model-id",
    )

    assert payload == {
        "channel": "evidence",
        "model": "model-id",
        "backend": "CPU",
        "audio_before_seconds": 1.4,
        "actual_trimmed_seconds": 0.272,
        "submitted_audio_seconds": 1.128,
        "reported_trailing_silence_seconds": 0.4,
    }


def test_comparison_and_summary_accept_exact_transcript_and_ending_preservation() -> None:
    baseline = _result(
        "same transcript.",
        submitted_audio_seconds=1.4,
        decode_seconds=0.4,
        queue_wait_seconds=0.02,
    )
    trimmed = _result(
        "same transcript.",
        submitted_audio_seconds=1.128,
        decode_seconds=0.3,
        queue_wait_seconds=0.01,
    )
    comparison = _comparison(baseline, trimmed)
    pair = {
        "repetition": 1,
        "baseline": baseline.payload,
        "trimmed": trimmed.payload,
        "comparison": comparison,
    }

    assert comparison == {
        "transcript_equal": True,
        "ending_preserved": True,
        "terminal_deletion_regression": False,
        "submitted_audio_reduction_seconds": 0.272,
    }
    summary = _summarize_pairs([pair])
    assert summary["status"] == "passed"
    assert summary["baseline_empty_transcript_rate"] == 0.0
    assert summary["trimmed_empty_transcript_rate"] == 0.0
    assert summary["submitted_reduction_matches_policy"] is True


def test_summary_rejects_terminal_deletion_and_empty_regression() -> None:
    baseline = _result(
        "ending",
        submitted_audio_seconds=1.4,
        decode_seconds=0.4,
        queue_wait_seconds=0.0,
    )
    trimmed = _result(
        "",
        submitted_audio_seconds=1.128,
        decode_seconds=0.3,
        queue_wait_seconds=0.0,
    )
    pair = {
        "repetition": 1,
        "baseline": baseline.payload,
        "trimmed": trimmed.payload,
        "comparison": _comparison(baseline, trimmed),
    }

    summary = _summarize_pairs([pair])

    assert summary["status"] == "failed"
    assert summary["all_transcripts_equal"] is False
    assert summary["all_endings_preserved"] is False
    assert summary["trimmed_empty_transcript_rate"] == 1.0


def test_authority_identity_requires_snapshot_ref_and_pin(tmp_path: Path) -> None:
    authority_ref = "https://example.invalid/issues/50"
    authority_pin = "a" * 64
    snapshot_path = tmp_path / "authority.snapshot.md"
    snapshot_path.write_text(
        "---\n"
        "document_kind: github_issue\n"
        f"document_ref: {authority_ref}\n"
        "document_node_id: issue-node\n"
        "document_updated_at: 2026-08-02T00:00:00Z\n"
        f"document_sha256: {authority_pin}\n"
        f"authority_ref: {authority_ref}\n"
        f"authority_pin: {authority_pin}\n"
        "---\n"
        "request\n",
        encoding="utf-8",
    )

    identity = _authority_identity(authority_ref, authority_pin, snapshot_path)

    assert identity["ref"] == authority_ref
    assert identity["pin_sha256"] == authority_pin
    assert identity["snapshot_sha256"]
    with pytest.raises(RuntimeError, match="pin does not match"):
        _authority_identity(authority_ref, "b" * 64, snapshot_path)


@pytest.mark.asyncio
async def test_cleanup_attempts_every_resource_and_preserves_first_failure() -> None:
    events: list[str] = []
    first_error = RuntimeError("session close failed")

    with pytest.raises(RuntimeError) as raised:
        await _close_resources(
            (
                _RecordingClose(events, "session", first_error),
                _RecordingClose(events, "backend", RuntimeError("backend close failed")),
                _RecordingClose(events, "runtime"),
            ),
            primary_error=None,
        )

    assert raised.value is first_error
    assert events == ["session", "backend", "runtime"]


@pytest.mark.asyncio
async def test_public_runner_removes_stale_report_before_failure(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text('{"status":"passed"}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        await run_evidence(
            model_root=tmp_path,
            audio_root=tmp_path,
            gpu_worker_path=tmp_path / "missing-worker.exe",
            gpu_device_id="auto",
            report_path=report_path,
            repetitions=3,
            authority_ref="https://example.invalid/issues/50",
            authority_pin="a" * 64,
            authority_snapshot_path=tmp_path / "missing-authority.snapshot.md",
            candidate_tree="b" * 40,
        )

    assert not report_path.exists()
