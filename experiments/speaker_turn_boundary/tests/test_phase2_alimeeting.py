from __future__ import annotations

import wave

import numpy as np

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ
from experiments.speaker_turn_boundary.corpus.alimeeting import (
    TextGridInterval,
    _unquote,
    index_alimeeting_eval,
    intervals_to_regions,
    load_wav_channel_0,
    parse_textgrid,
)
from experiments.speaker_turn_boundary.ground_truth import classify_active_speaker_transitions


def _textgrid_content() -> str:
    return """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 100.0
tiers? <exists>
size = 2
item []:
	item [1]:
		class = "IntervalTier"
		name = "N_SPK8013"
		xmin = 0
		xmax = 100.0
		intervals: size = 2
		intervals [1]:
			xmin = 1.0
			xmax = 2.5
			text = "你好"
		intervals [2]:
			xmin = 5.0
			xmax = 6.0
			text = "好的"
	item [2]:
		class = "IntervalTier"
		name = "N_SPK8014"
		xmin = 0
		xmax = 100.0
		intervals: size = 1
		intervals [1]:
			xmin = 2.0
			xmax = 3.0
			text = "嗯嗯"
"""


def test_parse_textgrid_tiers_and_intervals(tmp_path):
    path = tmp_path / "R8001_M8004.TextGrid"
    path.write_text(_textgrid_content(), encoding="utf-8")
    tiers = parse_textgrid(path)
    assert [name for name, _ in tiers] == ["N_SPK8013", "N_SPK8014"]
    assert len(tiers[0][1]) == 2
    assert tiers[0][1][0] == (1.0, 2.5, "你好")
    assert len(tiers[1][1]) == 1


def test_parse_textgrid_ignores_tier_level_xmin_xmax(tmp_path):
    path = tmp_path / "t.TextGrid"
    path.write_text(_textgrid_content(), encoding="utf-8")
    tiers = parse_textgrid(path)
    assert all(
        not (start == 0.0 and end == 100.0) for _, intervals in tiers for start, end, _ in intervals
    )


def test_unquote():
    assert _unquote('"hello"') == "hello"
    assert _unquote('""') == ""
    assert _unquote("plain") == "plain"


def test_intervals_to_regions_overlap_inference():
    intervals = [
        TextGridInterval("SPK8013", 1.0, 2.5, "你好"),
        TextGridInterval("SPK8014", 2.0, 3.0, "嗯"),
        TextGridInterval("SPK8013", 5.0, 6.0, "好的"),
    ]
    regions, stats = intervals_to_regions(intervals, duration_samples=16000 * 8)
    assert stats["skipped_zero_length_intervals"] == 0
    assert sum(r.end_sample - r.start_sample for r in regions) == 16000 * 8
    overlap = [r for r in regions if len(r.speakers) == 2]
    assert overlap
    assert overlap[0].speakers == {"SPK8013", "SPK8014"}
    changes, _ = classify_active_speaker_transitions(regions)
    assert any(change.kind == "interruption_onset" for change in changes)


def _eight_channel_wav(path, duration_samples: int) -> None:
    rng = np.random.default_rng(7)
    pcm = (rng.standard_normal((duration_samples, 8)) * 100).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(8)
        handle.setsampwidth(2)
        handle.setframerate(CANONICAL_SAMPLE_RATE_HZ)
        handle.writeframes(pcm.tobytes())


def test_load_wav_channel_0(tmp_path):
    path = tmp_path / "R8001_M8004_MS801.wav"
    _eight_channel_wav(path, 3200)
    channel0 = load_wav_channel_0(path)
    assert channel0.size == 3200


def test_index_alimeeting_eval_layout(tmp_path):
    far_audio = tmp_path / "alimeeting" / "Eval_Ali" / "Eval_Ali_far" / "audio_dir"
    far_textgrid = tmp_path / "alimeeting" / "Eval_Ali" / "Eval_Ali_far" / "textgrid_dir"
    far_audio.mkdir(parents=True)
    far_textgrid.mkdir(parents=True)
    _eight_channel_wav(far_audio / "R8001_M8004_MS801.wav", 8000)
    (far_textgrid / "R8001_M8004.TextGrid").write_text(_textgrid_content(), encoding="utf-8")
    sessions = index_alimeeting_eval(tmp_path)
    assert len(sessions) == 1
    session = sessions[0]
    assert session.session_id == "R8001_M8004"
    assert session.speakers == ["SPK8013", "SPK8014"]
    assert len(session.intervals) == 3
