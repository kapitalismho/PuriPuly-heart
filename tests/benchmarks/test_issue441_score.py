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


def metadata_run(speakers, exported, *, punctuation=""):
    words = []
    for index, (text, identity) in enumerate(zip(("we", "do", "not"), speakers, strict=True)):
        words.append({"type": "word", "text": text, "start": index / 3, "end": (index + 1) / 3,
                      "speaker_id": identity})
        if index != 2:
            words.append({"type": "spacing", "text": " ", "speaker_id": "not_source"})
    words.insert(2, {"type": "audio_event", "text": "[noise]", "speaker_id": "not_source"})
    if punctuation:
        words.append({"type": "word", "text": punctuation, "speaker_id": None})
    data = run([event(0, 1.1, "committed_transcript", segment_index=0, text="we do not"),
                event(1, 2, "diarized_transcript", segment_index=0, words=words)])
    if exported:
        data["observations"] = [{"arrival_s": 1.1, "text": "we do not", "speaker": None,
                                 "start_s": None, "end_s": None, "segment_index": 0,
                                 "settled": True, "raw": data["events"][0]["raw"]}]
        data["observations"].extend(
            {"arrival_s": 2, "text": word["text"], "speaker": word.get("speaker_id"),
             "start_s": word.get("start"), "end_s": word.get("end"), "segment_index": 0,
             "settled": True, "raw": word} for word in words)
        data["snapshots"] = [
            {"event_seq": 0, "arrival_s": 1.1, "revision": 0, "session_id": "session",
             "snapshot_key": "scribe:segment:0:committed_transcript", "operation": "replace",
             "observation_indices": [0]},
            {"event_seq": 1, "arrival_s": 2, "revision": 1, "session_id": "session",
             "snapshot_key": "scribe:segment:0:diarized", "operation": "replace",
             "observation_indices": list(range(1, len(words) + 1))},
        ]
    return data


@pytest.mark.parametrize("exported", [False, True], ids=["raw", "exported"])
@pytest.mark.parametrize(("speakers", "consequence", "groups"), [
    (("A", None, "A"), "identity_only", 1),
    ((None, "A", "A"), "identity_only", 1),
    (("A", "A", "A"), "identity_only", 1),
    (("A", None, "B"), "regroup_retranslate_required", 2),
    (("A", "B", "A"), "regroup_retranslate_required", 3),
])
def test_late_metadata_uses_source_group_partitions_not_word_run_counts(exported, speakers, consequence, groups):
    result = late_metadata(Journal(metadata_run(speakers, exported)))
    case = result["cases"][0]
    assert result["projection"] == "counterfactual_current_composition"
    assert case["consequence"] == consequence
    assert len(case["eventual_source_groups"]) == groups
    assert "".join(group["normalized_source_text"] for group in case["eventual_source_groups"]) == "wedonot"
    assert case["text_changed"] is False


@pytest.mark.parametrize("exported", [False, True], ids=["raw", "exported"])
def test_meaningful_punctuation_revision_is_not_erased_as_identity_only(exported):
    result = late_metadata(Journal(metadata_run(("A", None, "A"), exported, punctuation="?")))
    case = result["cases"][0]
    assert case["consequence"] == "source_text_changed"
    assert case["text_changed"] is True
    assert case["grouping_changed"] is None
    assert case["eventual_source_groups"][0]["normalized_source_text"] == "wedonot?"


@pytest.mark.asyncio
@pytest.mark.parametrize(("speakers", "groups"), [
    (("A", None, "A"), 1), ((None, "A", "A"), 1), (("A", "A", "A"), 1),
    (("A", None, "B"), 2), (("A", "B", "A"), 3),
])
async def test_counterfactual_partition_matches_actual_native_consumer_requests_and_captions(speakers, groups):
    from benchmarks.asr.issue441_score import compact_source_text
    from tests.core.test_native_peer_caption import final, idle, native, setup

    harness, backend, events, _ = setup()
    try:
        await harness.start()
        await final(harness, native(tuple(zip(("we ", "do ", "not"), speakers, strict=True))))
        await idle(harness)
        expected = late_metadata(Journal(metadata_run(speakers, False)))["cases"][0]["eventual_source_groups"]
        assert [compact_source_text(call.text) for call in backend.calls] == [
            group["normalized_source_text"] for group in expected]
        assert len(backend.calls) == groups
        assert len({item.utterance_id for item in events if getattr(item, "is_final", False)}) == groups
    finally:
        await harness.stop()


@pytest.mark.parametrize("reason", ["cap", "eof"])
@pytest.mark.parametrize("receipt", [1.10, 1.151], ids=["in-flight", "just-completed"])
def test_manual_send_interval_races_never_count_as_useful_prediction(reason, receipt):
    data = run([
        event(0, receipt, "committed_transcript", segment_index=0, text="no"),
        diarized(1, 1.3, 0, 0, "A"),
        event(2, 3.2, "committed_transcript", segment_index=1, text="no"),
        diarized(3, 3.3, 1, 2, "A"),
    ])
    data["sends"] = [{"kind": f"{reason}_commit", "start_s": 1.05, "completed_s": 1.15,
                     "status": "completed", "session_id": "session",
                     "control": {"reason": reason, "last_native_commit_seq": None}}]
    reference = recording()
    reference["turn_opportunities"] = [{"id": "intermediate", "kind": "intermediate", "time_s": 1,
                                        "start_s": 1, "end_s": 2, "span_ids": ["a", "b"],
                                        "reference_text": "no | no"}]
    result = turn_score(reference, Journal(data))
    assert result["counts"]["fragments_translation_triggers"] == 2
    assert result["counts"]["useful_intermediate"] == 0
    assert result["opportunities"][0]["cap_eof_context_excluded_from_useful"]
    contexts = result["commits"][0]["possible_manual_commands"]
    assert contexts[0]["association"] == ("in_flight_send" if receipt <= 1.15 else "first_receipt_after_send")
    assert result["commits"][0]["cause"] == "unavailable"


def test_clean_native_intermediate_remains_positive_without_manual_control():
    data = run([event(0, 1.2, "committed_transcript", segment_index=0, text="no"),
                diarized(1, 1.3, 0, 0, "A"),
                event(2, 3.2, "committed_transcript", segment_index=1, text="no"),
                diarized(3, 3.3, 1, 2, "A")])
    reference = recording()
    reference["turn_opportunities"] = [{"id": "intermediate", "kind": "intermediate", "time_s": 1,
                                        "start_s": 1, "end_s": 2, "span_ids": ["a", "b"],
                                        "reference_text": "no | no"}]
    result = turn_score(reference, Journal(data))
    assert result["counts"]["useful_intermediate"] == 1
    assert result["opportunities"][0]["cap_eof_context_excluded_from_useful"] is False


@pytest.mark.parametrize("observed", ["first", "first not", ""])
def test_missing_ending_or_negation_retains_opportunity_without_fast_completion(observed):
    events = []
    if observed:
        words = [{"text": word, "start": index / 5, "end": (index + 1) / 5, "speaker_id": "A"}
                 for index, word in enumerate(observed.split())]
        events = [event(0, 1, "committed_transcript", segment_index=0, text=observed),
                  event(1, 1.2, "diarized_transcript", segment_index=0, words=words)]
    reference = recording()
    reference["spans"][0].update(end_s=10, text="first no")
    reference["turn_opportunities"] = [{"id": "end", "kind": "ending", "time_s": 10,
                                        "start_s": 10, "end_s": 10, "span_ids": ["a"],
                                        "reference_text": "first no"}]
    result = turn_score(reference, Journal(run(events)))
    assert result["counts"]["opportunities"] == 1
    assert result["counts"]["endings_latency_unavailable"] == 1
    assert result["ending_to_commit"]["n"] == 0
    assert result["opportunities"][0]["ending_to_commit_s"] is None
    assert result["opportunities"][0]["content"]["reference"] == "first no"
    assert result["opportunities"][0]["content"]["reference_contained"] is False


def test_full_reference_tail_preserves_legitimate_signed_timing():
    reference = recording()
    reference["spans"][0].update(end_s=10, text="first no")
    reference["turn_opportunities"] = [{"id": "end", "kind": "ending", "time_s": 10,
                                        "start_s": 10, "end_s": 10, "span_ids": ["a"],
                                        "reference_text": "first no"}]
    data = run([
        event(0, 4.1, "committed_transcript", segment_index=0, text="first"),
        event(1, 4.2, "diarized_transcript", segment_index=0, words=[
            {"text": "first", "start": 0, "end": 4, "speaker_id": "A"}]),
        event(2, 9.9, "committed_transcript", segment_index=1, text="no"),
        event(3, 10.2, "diarized_transcript", segment_index=1, words=[
            {"text": "no", "start": 9.5, "end": 9.8, "speaker_id": "A"}]),
    ])
    result = turn_score(reference, Journal(data))
    assert result["counts"]["endings_latency_eligible"] == 1
    assert result["opportunities"][0]["ending_to_commit_s"] == pytest.approx(-0.1)
    assert result["ending_to_commit"]["n"] == 1
