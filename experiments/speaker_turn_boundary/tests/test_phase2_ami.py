from __future__ import annotations

import numpy as np

from experiments.speaker_turn_boundary.corpus.ami import (
    AmiWord,
    _load_words_xml,
    _speaker_from_filename,
    words_to_regions,
)
from experiments.speaker_turn_boundary.ground_truth import classify_active_speaker_transitions
from experiments.speaker_turn_boundary.tests.phase2_helpers import write_pcm16_wav


def _words_xml_fixture(tmp_path) -> str:
    content = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root nite:id="ES2003a.A.words" xmlns:nite="http://nite.sourceforge.net/">
   <w nite:id="ES2003a.A.words0" starttime="1.0" endtime="1.5">hello</w>
   <w nite:id="ES2003a.A.words1" starttime="2.0" endtime="2.4">%%</w>
   <w nite:id="ES2003a.A.words2" starttime="3.0" endtime="3.2" punc="true">,</w>
</nite:root>
"""
    path = tmp_path / "ES2003a.A.words.xml"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_load_words_xml_no_who_attribute(tmp_path):
    words = _load_words_xml(_words_xml_fixture(tmp_path))
    assert len(words) == 3
    assert words[0].start_time_s == 1.0
    assert words[0].end_time_s == 1.5
    assert words[1].ambiguous is True
    assert words[2].ambiguous is False
    assert words[2].text == ","


def test_speaker_from_filename():
    from pathlib import Path

    speaker = _speaker_from_filename(Path("ES2003a.B.words.xml"), "ES2003a")
    assert speaker == "ES2003a.ParticipantB"


def test_words_to_regions_gap_change_and_overlap():
    words = [
        AmiWord("ES2003a.ParticipantA", 0.1, 0.4, "okay", False),
        AmiWord("ES2003a.ParticipantB", 0.9, 1.3, "right", False),
        AmiWord("ES2003a.ParticipantA", 1.5, 2.0, "yes", False),
        AmiWord("ES2003a.ParticipantB", 1.7, 2.2, "but", False),
    ]
    regions, stats = words_to_regions(words, duration_samples=48000)
    assert stats["skipped_zero_length_words"] == 0
    assert sum(r.end_sample - r.start_sample for r in regions) == 48000
    changes, transitions = classify_active_speaker_transitions(regions)
    kinds = [change.kind for change in changes]
    assert kinds == ["gap_speaker_change", "gap_speaker_change", "interruption_onset"]
    assert changes[0].change_sample == int(round(0.9 * 16000))
    assert changes[1].change_sample == int(round(1.5 * 16000))
    assert changes[2].change_sample == int(round(1.7 * 16000))
    overlap_regions = [r for r in regions if len(r.speakers) == 2]
    assert overlap_regions
    assert overlap_regions[0].speakers == {
        "ES2003a.ParticipantA",
        "ES2003a.ParticipantB",
    }


def test_words_to_regions_zero_length_words_skipped():
    words = [
        AmiWord("s1", 0.1, 0.1, "punct", False),
        AmiWord("s1", 0.2, 0.4, "real", False),
    ]
    regions, stats = words_to_regions(words, duration_samples=16000)
    assert stats["skipped_zero_length_words"] == 1
    assert [region.speakers for region in regions] == [frozenset(), {"s1"}, frozenset()]


def test_words_to_regions_ambiguous_tagging():
    words = [
        AmiWord("s1", 0.1, 0.3, "clear", False),
        AmiWord("s2", 0.2, 0.4, "%%", True),
    ]
    regions, _ = words_to_regions(words, duration_samples=16000)
    assert any(region.ambiguous for region in regions)


def test_ami_meeting_end_to_end(tmp_path):

    from experiments.speaker_turn_boundary.corpus.ami import load_ami_meeting

    samples = np.zeros(16000 * 20, dtype=np.float32)
    wav_path = tmp_path / "audio" / "ES2003a.Mix-Headset.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16_wav(wav_path, samples)
    words_dir = tmp_path / "annotations" / "words"
    words_dir.mkdir(parents=True)
    for letter in ["A", "B", "C", "D"]:
        content = f"""<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root nite:id="ES2003a.{letter}.words" xmlns:nite="http://nite.sourceforge.net/">
   <w nite:id="ES2003a.{letter}.words0" starttime="1.0" endtime="1.5">{letter}word</w>
</nite:root>
"""
        (words_dir / f"ES2003a.{letter}.words.xml").write_text(content, encoding="utf-8")
    meeting = load_ami_meeting("ES2003a", "pilot", wav_path, tmp_path / "annotations")
    assert meeting.duration_samples == 320000
    assert len(meeting.words) == 4
    assert len(meeting.participants) == 4
    assert meeting.participants == [
        "ES2003a.ParticipantA",
        "ES2003a.ParticipantB",
        "ES2003a.ParticipantC",
        "ES2003a.ParticipantD",
    ]
    assert meeting.wav_sha256
