from experiments.speaker_turn_boundary.ground_truth import SpeakerRegion
from experiments.speaker_turn_boundary.turn_episode.audit import rebuilt_tag


def test_rebuilt_tag_detects_stable_overlap_crossing_scored_start():
    regions = [
        SpeakerRegion(
            audio_epoch=0,
            start_sample=0,
            end_sample=3200,
            speakers=frozenset({"A", "B"}),
            ambiguous=False,
        )
    ]

    assert rebuilt_tag([], 1600, 4800, regions) == "overlap_present"
