from __future__ import annotations

import copy

import pytest

from benchmarks.asr.issue441_score import Journal, late_metadata, score_recording, to_source, turn_score


def recording():
    return {
        "id": "fixture", "language": "en", "diagnostic": True, "sha256": "frozen",
        "spans": [
            {"id": "a", "start_s": 0, "end_s": 1, "speaker": "A", "text": "no", "reference_available": True},
            {"id": "b", "start_s": 2, "end_s": 3, "speaker": "B", "text": "no", "reference_available": True},
            {"id": "c", "start_s": 4, "end_s": 5, "speaker": "A", "text": "yes", "reference_available": True},
        ],
        "anchors": [
            {"id": "a1", "time_s": 0.5, "span_id": "a", "speaker": "A", "span_end_s": 1},
            {"id": "b1", "time_s": 2.5, "span_id": "b", "speaker": "B", "span_end_s": 3},
            {"id": "c1", "time_s": 4.5, "span_id": "c", "speaker": "A", "span_end_s": 5},
        ], "overlaps": [], "changes": [], "turn_opportunities": [],
    }


def run(events=(), arm="E0", observations=()):
    return {"arm": arm, "recording_id": "fixture", "connected": True,
            "metadata": {"input_sha256": "frozen"}, "events": list(events),
            "observations": list(observations), "status": "completed"}


def event(seq, arrival, kind, **raw):
    return {"seq": seq, "arrival_s": arrival, "kind": "native", "session_id": "session",
            "raw": {"message_type": kind, **raw}}


def diarized(seq, arrival, index, start, identity, text="no"):
    return event(seq, arrival, "diarized_transcript", segment_index=index,
                 words=[{"text": text, "start": start, "end": start + 1, "speaker_id": identity}])


def test_late_merge_does_not_repair_online_and_mapping_is_fixed():
    events = [diarized(0, 1.5, 0, 0, "fragment"), diarized(1, 3.5, 1, 2, "other"),
              diarized(2, 5.5, 2, 4, "canonical", "yes"),
              event(3, 7, "speakers_merged", from_speaker_id="fragment", to_speaker_id="canonical")]
    original = copy.deepcopy(events)
    result = score_recording(recording(), run(events))
    assert result["mapping"] == {"session": {"canonical": "A", "other": "B"}}
    assert result["views"]["online"]["counts"]["correct"] == 2
    assert result["views"]["online"]["counts"]["wrong"] == 1
    assert result["views"]["online"]["anchors"][0]["unmatched_id"]
    assert result["views"]["eventual"]["counts"]["correct"] == 3
    assert events == original


def test_identical_text_correlates_only_by_segment_index():
    events = [event(0, 1, "committed_transcript", segment_index=0, text="no"),
              event(1, 3, "committed_transcript", segment_index=1, text="no"),
              diarized(2, 3.2, 1, 2, "B"), diarized(3, 7, 0, 0, "A")]
    journal = Journal(run(events))
    assert journal.commits()[0]["start_s"] == 0
    assert journal.commits()[1]["start_s"] == 2
    assert journal.snapshot(4)[0]["speaker"] is None
    assert score_recording(recording(), run(events))["views"]["online"]["anchors"][0]["outcome"] == "missing_timing"


@pytest.mark.parametrize("case,expected", [
    ([], "missing_text"),
    ([event(0, 0.8, "committed_transcript", segment_index=0, text="no")], "missing_timing"),
    ([diarized(0, 0.8, 0, 0, None)], "missing_speaker"),
])
def test_missing_evidence_never_reduces_fixed_anchor_denominator(case, expected):
    result = score_recording(recording(), run(case))
    for view in result["views"].values():
        assert view["counts"]["eligible"] == 3
        assert view["counts"]["assigned"] == 0
        assert view["anchors"][0]["outcome"] == expected


def test_unconnected_is_unrun_not_model_failure():
    disconnected = run()
    disconnected["connected"] = False
    result = score_recording(recording(), disconnected)
    assert result["status"] == "unrun"
    assert "views" not in result
    assert result["planned_anchors"] == 3


def test_empty_draft_retraction_and_diarized_revision_replace_stale_words():
    observations = [
        {"arrival_s": 0.5, "start_s": 0, "end_s": 1, "speaker": "stale", "text": "no", "settled": False, "segment_index": None},
        {"arrival_s": 1.2, "start_s": None, "end_s": None, "speaker": None, "text": "no", "settled": False, "segment_index": 0},
        {"arrival_s": 3, "start_s": 0, "end_s": 1, "speaker": "A", "text": "no", "settled": True, "segment_index": 0},
    ]
    snapshots = [
        {"arrival_s": 0.5, "revision": 0, "snapshot_key": "scribe:partial", "operation": "replace", "observation_indices": [0]},
        {"arrival_s": 1.2, "revision": 1, "snapshot_key": "scribe:partial", "operation": "retract", "observation_indices": []},
        {"arrival_s": 1.2, "revision": 1, "snapshot_key": "scribe:segment:0:committed", "operation": "replace", "observation_indices": [1]},
        {"arrival_s": 3, "revision": 2, "snapshot_key": "scribe:segment:0:diarized", "operation": "replace", "observation_indices": [2]},
    ]
    data = run(observations=observations)
    data["snapshots"] = snapshots
    journal = Journal(data)
    assert len(journal.snapshot(2)) == 1
    assert journal.snapshot(2)[0]["speaker"] is None
    assert len(journal.snapshot()) == 1
    assert journal.snapshot()[0]["speaker"] == "A"


def test_mapping_does_not_remap_each_turn_to_hide_fragmentation():
    events = [diarized(0, 1, 0, 0, "A0"), diarized(1, 3, 1, 2, "B"), diarized(2, 5, 2, 4, "A1")]
    result = score_recording(recording(), run(events))
    assert result["views"]["eventual"]["counts"]["correct"] == 2
    assert result["views"]["eventual"]["counts"]["wrong"] == 1


def test_activity_only_needs_no_text():
    data = run(arm="N", observations=[{"arrival_s": 1, "start_s": 0, "end_s": 1, "speaker": "0", "text": None, "settled": True}])
    result = score_recording(recording(), data)
    assert result["activity_only"]
    assert result["views"]["eventual"]["counts"]["correct"] == 1
    assert result["views"]["eventual"]["counts"]["eligible"] == 3


def test_late_identity_grouping_and_unsafe_cut_are_observable_outcomes():
    data = run([
        event(0, 1.3, "committed_transcript", segment_index=0, text="no two"),
        event(1, 2, "diarized_transcript", segment_index=0, words=[
            {"text": "no", "start": 0, "end": 0.6, "speaker_id": "A"},
            {"text": "two", "start": 0.6, "end": 1, "speaker_id": "B"}]),
    ])
    reference = recording()
    reference["turn_opportunities"] = [{"id": "unsafe", "time_s": 1, "start_s": 0.8, "end_s": 1.2,
                                        "kind": "unsafe", "reference_text": "two"}]
    journal = Journal(data)
    assert late_metadata(journal)["counts"] == {"regroup_retranslate_required": 1}
    assert turn_score(reference, journal)["counts"]["premature_cuts"] == 1
    assert turn_score(reference, journal)["counts"]["causes"] == {"unavailable": 1}


def test_late_timestamp_event_cannot_overwrite_indexed_diarized_identity():
    data = run([diarized(0, 1, 0, 0, "A"),
                event(1, 2, "committed_transcript_with_timestamps", segment_index=0,
                      words=[{"text": "no", "start": 0, "end": 1}])])
    assert Journal(data).snapshot()[0]["speaker"] == "A"


def test_source_mapping_does_not_bridge_discarded_audio_or_padding():
    data = run(arm="S1")
    data["sends"] = [{"status": "completed", "session_id": "session",
                     "transmitted_start_s": 0, "transmitted_end_s": 1.2,
                     "source_ranges": [{"start_s": 0, "end_s": 0.5},
                                       {"start_s": 5, "end_s": 5.5}]}]
    unit = {"start_s": 0.25, "end_s": 0.75, "session": "session", "speaker": "A"}
    assert to_source(unit, data)[0]["start_s"] is None
    unit["start_s"], unit["end_s"] = 0.75, 1.15
    assert to_source(unit, data)[0]["start_s"] is None
    unit["start_s"], unit["end_s"] = 0.6, 0.8
    assert to_source(unit, data)[0]["start_s"] == pytest.approx(5.1)


def test_ending_delay_uses_last_reference_words_not_earlier_fragment():
    data = run([
        event(0, 0.6, "committed_transcript", segment_index=0, text="first"),
        event(1, 0.7, "diarized_transcript", segment_index=0, words=[
            {"text": "first", "start": 0, "end": 0.4, "speaker_id": "A"}]),
        event(2, 1.6, "committed_transcript", segment_index=1, text="no"),
        event(3, 1.8, "diarized_transcript", segment_index=1, words=[
            {"text": "no", "start": 0.4, "end": 1, "speaker_id": "A"}]),
    ])
    reference = recording()
    reference["turn_opportunities"] = [{"id": "end", "time_s": 1, "start_s": 1, "end_s": 1,
                                        "kind": "ending", "reference_text": "first no", "span_ids": ["a"]}]
    result = turn_score(reference, Journal(data))
    assert result["opportunities"][0]["ending_to_commit_s"] == pytest.approx(0.6)
    assert result["opportunities"][0]["merged_boundary"] is None


def test_paired_continuing_pause_cut_is_found_without_exact_word_endpoint():
    data = run([
        event(0, 1.6, "committed_transcript", segment_index=0, text="no"),
        diarized(1, 1.8, 0, 0, "A"),
        event(2, 3.6, "committed_transcript", segment_index=1, text="no"),
        diarized(3, 3.8, 1, 2, "A"),
    ])
    reference = recording()
    reference["turn_opportunities"] = [{"id": "continue", "time_s": 1.1, "start_s": 1.1, "end_s": 1.9,
                                        "kind": "unsafe", "reference_text": "no | no", "span_ids": ["a", "b"]}]
    assert turn_score(reference, Journal(data))["opportunities"][0]["premature_cut"] is True


def test_caption_composition_reuses_native_map_and_obeys_output_cutoff():
    data = run(arm="S1", events=[
        {"seq": 0, "arrival_s": 1.1, "session_id": "session", "raw": {"tokens": [
            {"text": "no", "start_ms": 0, "end_ms": 1000, "speaker": "nativeA", "is_final": True}]}},
        {"seq": 1, "arrival_s": 3.1, "session_id": "session", "raw": {"tokens": [
            {"text": "no", "start_ms": 2000, "end_ms": 3000, "speaker": "nativeB", "is_final": True}]}},
    ])
    data["source_origin_mono_s"] = 100
    data["session_scopes"] = [{"session_id": "session", "speaker_session_scope": "scope"}]
    data["sends"] = [{"status": "completed", "session_id": "session", "transmitted_start_s": 0,
                     "transmitted_end_s": 5, "source_start_s": 0, "source_end_s": 5}]
    assignment = {"attribution": {"key": {"speaker_scope_id": "scope", "local_speaker_id": "nativeA"}}}
    data["application"] = {"groups": [], "events": [
        {"t": 103.5, "event": {"utterance_id": "joined", "is_final": True, "speaker_assignment": assignment}}],
        "requests": [], "captions": []}
    for start, identity in ((0, "nativeA"), (2, "nativeB")):
        data["application"]["groups"].append({
            "t": 101.2 + start, "kind": "continuation" if start else "new",
            "previous": "joined" if start else None, "accepted_caption": {"utterance_id": "joined"},
            "transcript": {"text": "no", "final_speaker_runs": [
                {"text": "no", "speaker_id": identity, "session_scope": "scope",
                 "source_start_ms": start * 1000, "source_end_ms": (start + 1) * 1000}]}})
    reference = recording()
    reference["changes"] = [{"id": "switch", "time_s": 2, "from_speaker": "A", "to_speaker": "B", "kind": "change"}]
    result = score_recording(reference, data)
    caption = result["caption_structure"]
    assert result["views"]["eventual"]["counts"]["correct"] == 2
    assert caption["views"]["online"]["anchors"][0]["outcome"] == "missing_text"
    assert caption["views"]["eventual"]["anchors"][1]["outcome"] == "wrong"
    assert caption["counts"]["harmful_cross_speaker_joins"] == 1


def test_cap_context_and_empty_commits_never_count_as_useful_output():
    data = run([
        event(0, 1.2, "committed_transcript", segment_index=0, text="no"),
        {"seq": 1, "arrival_s": 1.2, "session_id": "session", "kind": "native_commit_control_state",
         "raw": {"segment_index": 0, "cause": "unknown_native_or_manual_race",
                 "last_command": {"reason": "cap", "last_native_commit_seq": None}}},
        diarized(2, 1.3, 0, 0, "A"),
        event(3, 3.2, "committed_transcript", segment_index=1, text="no"),
        diarized(4, 3.3, 1, 2, "A"),
        event(5, 6, "committed_transcript", segment_index=2, text=""),
    ])
    reference = recording()
    reference["turn_opportunities"] = [{"id": "clause", "time_s": 1, "start_s": 1, "end_s": 2,
                                        "kind": "intermediate", "reference_text": "no | no", "span_ids": ["a", "b"]}]
    result = turn_score(reference, Journal(data))
    assert result["counts"]["useful_intermediate"] == 0
    assert result["counts"]["fragments_translation_triggers"] == 2
    assert result["counts"]["empty_commits"] == 1
    assert result["opportunities"][0]["cap_eof_context_excluded_from_useful"]


def test_text_only_partial_snapshot_is_missing_timing_not_missing_text():
    data = run([event(0, 0.8, "partial_transcript", text="no")])
    data["snapshots"] = [{"event_seq": 0, "arrival_s": 0.8, "revision": 0,
                         "snapshot_key": "scribe:partial", "operation": "replace",
                         "observation_indices": [], "session_id": "session", "settled": False}]
    result = score_recording(recording(), data)
    assert result["views"]["online"]["counts"]["eligible"] == 3
    assert result["views"]["online"]["anchors"][0]["outcome"] == "missing_timing"


def test_frozen_identity_anchor_remains_eligible_without_reference_transcript():
    reference = recording()
    reference["spans"][0].update(text="", reference_available=False)
    result = score_recording(reference, run([diarized(0, 1, 0, 0, "A")]))
    assert result["views"]["eventual"]["counts"]["eligible"] == 3
    assert result["views"]["eventual"]["anchors"][0]["outcome"] == "correct"


def test_unanchored_reference_limits_accuracy_not_native_change_observation():
    reference = recording()
    reference["anchors"] = [anchor for anchor in reference["anchors"] if anchor["speaker"] == "A"]
    reference["changes"] = [{"id": "switch", "time_s": 2, "from_speaker": "A", "to_speaker": "B",
                             "source_span_id": "a", "target_span_id": "b", "kind": "speaker_change"}]
    data = run([diarized(0, 1, 0, 0, "A"), diarized(1, 3, 1, 2, "B"),
                diarized(2, 5, 2, 4, "A")])
    result = score_recording(reference, data)
    assert result["views"]["eventual"]["counts"]["eligible"] == 2
    assert result["reference_scope"]["unanchored_speakers"] == ["B"]
    change = result["changes"][-1]
    assert change["status"] == "observed"
    assert change["missed_change"] is False
    assert change["correct_identity_pair"] is None
    assert result["timing"]["new_speaker_cases"][1]["status"] == "reference_unanchored"
