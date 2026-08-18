from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data import natural_acquisition
from experiments.psem_training_strategy_gate.data.natural_acquisition import (
    NaturalAcquisitionError,
    _download_to_part,
    _identity_components,
    _materialization_lock,
    _materialize_one,
    _prepare_partial,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    ADDITIONAL_AMI_MEETINGS,
    AMI_EXPANSION_COMPONENTS,
    AMI_EXPANSION_EXCLUSIONS,
    BASELINE_AMI_MEETINGS,
    EXPECTED_AMI_MEETINGS,
    wav_identity,
)


def test_expansion_inventory_is_exact_and_disjoint_from_the_baseline() -> None:
    assert len(AMI_EXPANSION_COMPONENTS) == 14
    assert len(ADDITIONAL_AMI_MEETINGS) == 48
    assert not ADDITIONAL_AMI_MEETINGS & BASELINE_AMI_MEETINGS
    assert EXPECTED_AMI_MEETINGS == (
        BASELINE_AMI_MEETINGS | ADDITIONAL_AMI_MEETINGS
    )
    assert set(ADDITIONAL_AMI_MEETINGS) == {
        meeting_id
        for component in AMI_EXPANSION_COMPONENTS.values()
        for meeting_id in component
    }
    assert AMI_EXPANSION_EXCLUSIONS == {
        "ES2010d": {
            "reason": "official_mix_headset_is_16khz_stereo_pcm16",
            "waveform_size_bytes": 62030212,
            "waveform_sha256": "0435f583b7ffa7055e7c5a884ed6eafcbfcb3b4f4fe6f04a16e34f7e04c2455e",
            "sample_rate_hz": 16000,
            "channels": 2,
            "sample_width_bytes": 2,
        },
        "ES2011c": {
            "reason": "manual_segment_has_reversed_bounds",
            "waveform_size_bytes": 51714092,
            "waveform_sha256": "8c2d75e76817ab770ccab71e48e1d41e62729992e6f928bf2a6a8a4bd381e111",
            "annotation_file": "ami/annotations/segments/ES2011c.C.segments.xml",
            "annotation_file_size_bytes": 25179,
            "annotation_file_sha256": "68c2e2cc866416b67166394e5a7cd77a752d04faf551264cdb22f37414b605b3",
            "invalid_segment_index": 0,
            "transcriber_start": "78.122",
            "transcriber_end": "78.001",
        },
    }
    assert not set(AMI_EXPANSION_EXCLUSIONS) & ADDITIONAL_AMI_MEETINGS


def test_identity_components_join_series_and_global_speakers() -> None:
    meetings = {
        "AA1001a": {"speaker_ids": ["speaker-a"]},
        "AA1001b": {"speaker_ids": ["speaker-b"]},
        "BB2001a": {"speaker_ids": ["speaker-b"]},
        "CC3001a": {"speaker_ids": ["speaker-c"]},
    }
    assert _identity_components(meetings, set(meetings)) == [
        ("AA1001a", "AA1001b", "BB2001a"),
        ("CC3001a",),
    ]


def test_existing_valid_materialization_is_reused(tmp_path: Path) -> None:
    meeting_id = "ES2005a"
    target = (
        tmp_path
        / "ami"
        / "audio"
        / meeting_id
        / f"{meeting_id}.Mix-Headset.wav"
    )
    target.parent.mkdir(parents=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)
    assert _materialize_one(
        tmp_path, meeting_id, 0.01, wav_identity(target)
    ) == "existing"


def test_oversized_or_complete_invalid_partial_is_restarted(tmp_path: Path) -> None:
    part_path = tmp_path / "audio.wav.part"
    part_path.write_bytes(b"x" * 60)
    assert _prepare_partial(part_path, 50) == 0
    assert not part_path.exists()
    part_path.write_bytes(b"x" * 50)
    assert _prepare_partial(part_path, 50) == 0
    assert not part_path.exists()


def test_full_size_partial_with_eof_error_is_restarted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part_path = tmp_path / "audio.wav.part"
    part_path.write_bytes(b"x" * 50)
    monkeypatch.setattr(
        natural_acquisition,
        "wav_identity",
        lambda *_: (_ for _ in ()).throw(EOFError()),
    )
    assert _prepare_partial(part_path, 50) == 0
    assert not part_path.exists()


def test_full_size_partial_with_wrong_identity_is_restarted(tmp_path: Path) -> None:
    part_path = tmp_path / "audio.wav.part"
    with wave.open(str(part_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)
    expected = wav_identity(part_path)
    expected["waveform_sha256"] = "f" * 64
    assert _prepare_partial(part_path, part_path.stat().st_size, expected) == 0
    assert not part_path.exists()


def test_download_rejects_payload_before_writing_past_remote_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": "5"}

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    monkeypatch.setattr(natural_acquisition, "_remote_size", lambda *_: 5)
    monkeypatch.setattr(
        natural_acquisition, "urlopen", lambda *_args, **_kwargs: Response(b"123456")
    )
    part_path = tmp_path / "audio.wav.part"
    with pytest.raises(NaturalAcquisitionError, match="exceeds remote size"):
        _download_to_part(
            "https://example.invalid/audio.wav",
            part_path,
            1.0,
            {"waveform_size_bytes": 5},
        )
    assert part_path.stat().st_size <= 5


def test_existing_waveform_must_match_the_accepted_identity(tmp_path: Path) -> None:
    meeting_id = "ES2005a"
    target = (
        tmp_path
        / "ami"
        / "audio"
        / meeting_id
        / f"{meeting_id}.Mix-Headset.wav"
    )
    target.parent.mkdir(parents=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)
    expected = wav_identity(target)
    expected["waveform_sha256"] = "f" * 64
    with pytest.raises(NaturalAcquisitionError, match="identity changed"):
        _materialize_one(tmp_path, meeting_id, 0.01, expected)


def test_materialization_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    with _materialization_lock(tmp_path):
        with pytest.raises(NaturalAcquisitionError, match="another AMI materializer"):
            with _materialization_lock(tmp_path):
                raise AssertionError
