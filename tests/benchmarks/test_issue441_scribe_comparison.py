import pytest

from benchmarks.asr.issue441_scribe_capture import map_word
from benchmarks.asr.issue441_scribe_fresh import (
    actual_source_index,
    match_interval,
    request_key,
    reserve_selected,
)
from benchmarks.asr.issue441_scribe_replay import (
    classify_metadata,
    exact_coverage,
    selected_requests,
)


def wire(tx, source, *, session="first", completed=1, status="completed"):
    return {
        "session_id": session,
        "status": status,
        "completed_mono_s": completed,
        "transmitted_start_s": tx[0],
        "transmitted_end_s": tx[1],
        "source_ranges": [{"start_s": a, "end_s": b} for a, b in source],
    }


def test_contiguous_scribe_word_maps_only_at_actual_completed_receipt():
    sends = [
        wire((0.0, 0.25), [(5.0, 5.25)]),
        wire((0.25, 0.5), [(5.25, 5.5)], session="retired"),
        wire((0.25, 0.5), [(5.25, 5.5)], status="failed"),
        wire((0.25, 0.5), [(5.25, 5.5)], completed=2),
    ]
    earlier = map_word(sends, "first", 1.9, 0.2, 0.3)
    later = map_word(sends, "first", 2.0, 0.2, 0.3)
    assert earlier[:2] == (None, None)
    assert earlier[3] == "discontinuous_or_partial_source_mapping"
    assert earlier[2] == [{"start_s": 5.2, "end_s": 5.25}]
    assert later[:2] == pytest.approx((5.2, 5.3))
    assert later[3] == "mapped"


@pytest.mark.parametrize(
    "sends",
    [
        [wire((0, 1), [(2, 3)]), wire((1, 2), [(4, 5)])],
        [wire((0, 1), [(2, 3)]), wire((1, 2), [(2.5, 3.5)])],
        [wire((0, 0.9), [(2, 2.9)]), wire((0.9, 1), []), wire((1, 2), [(2.9, 3.9)])],
    ],
)
def test_scribe_gap_overlap_and_generated_padding_are_not_broad_envelopes(sends):
    a, b, pieces, state = map_word(sends, "first", 2, 0.8, 1.2)
    assert (a, b) == (None, None)
    assert len(pieces) == 2
    assert state == "discontinuous_or_partial_source_mapping"
    assert map_word(sends, "first", 2, 1.0, 1.2)[3] == "mapped"


def group(caption_id, a, b, source, speaker):
    return {
        "caption_id": caption_id,
        "source": source,
        "source_parts": [{"segment_index": 12, "commit_char_range": [a, b], "source": source}],
        "speaker_assignment": speaker,
    }


def test_same_source_bounds_and_attribution_change_cost_no_request():
    before = [{"groups": [group("existing", 0, 8, "One Two.", None)]}]
    after = [{"groups": [group("existing", 0, 8, "One Two.", "A")]}]
    assert classify_metadata(before, after, [], True) == "identity_only"
    assert exact_coverage(after, {12: "One Two."}) == {
        "missing_commit_characters": [],
        "multiply_covered_commit_characters": [],
        "group_text_mismatches": [],
    }


def test_regroup_keeps_complete_source_once_and_selects_by_provenance():
    before = [{"groups": [group("same", 0, 8, "One Two.", None)]}]
    after = [{"groups": [group("same", 0, 4, "One ", "A"), group("new", 4, 8, "Two.", "B")]}]
    assert classify_metadata(before, after, [7, 8], True) == "changed_partition_with_new_request"
    result = {
        "source_turns": [{"turn_id": "same", "segment_index": 12}],
        "requests": [
            {"request_index": 7, "lineage": {"caption_id": "same", "source_parts": []}},
            {"request_index": 8, "lineage": after[0]["groups"][1]},
            {
                "request_index": 9,
                "lineage": {
                    **group("unrelated", 4, 8, "Two.", "B"),
                    "source_parts": [{"segment_index": 13}],
                },
            },
        ],
    }
    assert selected_requests(result, {"E0": {"index": 12}}) == [7, 8]
    assert exact_coverage(after, {12: "One Two."})["missing_commit_characters"] == []
    duplicated = exact_coverage([*before, *after], {12: "One Two."})
    assert len(duplicated["multiply_covered_commit_characters"]) == 8


def fresh_lineage(session="current", *, chars=(0, 4), kind="repair"):
    return {
        "admission_ordinal": 17,
        "source_order": 2,
        "revision": 3,
        "kind": kind,
        "source_parts": [
            {
                "provider_session_id": session,
                "segment_index": 0,
                "commit_char_range": list(chars),
            }
        ],
    }


def test_fresh_mt_key_survives_new_child_id_but_not_old_scope_or_body_change():
    key = request_key({**fresh_lineage(), "caption_id": "observed-child"})
    assert key == request_key({**fresh_lineage(), "caption_id": "replay-child"})
    assert key != request_key(fresh_lineage("retired"))
    assert key != request_key(fresh_lineage(chars=(0, 6)))
    selected, used = {key: {"body": {"text": "Word", "context": "prior"}}}, set()
    assert not reserve_selected(selected, used, request_key(fresh_lineage("retired")), {})
    with pytest.raises(ValueError, match="not exactly matched"):
        reserve_selected(selected, used, key, {"text": "Word", "context": "wrong"})
    assert used == set()
    assert reserve_selected(selected, used, key, {"text": "Word", "context": "prior"})
    with pytest.raises(ValueError, match="not exactly matched"):
        reserve_selected(selected, used, key, {"text": "Word", "context": "prior"})


def test_fresh_selection_requires_actual_scoped_current_word_range():
    segment = {
        "content_ranges": [
            {"source_start_sample": 0, "source_end_sample": 32000, "source_sample_rate_hz": 16000}
        ]
    }
    commits = [
        {
            "kind": "native",
            "session_id": session,
            "seq": seq,
            "raw": {
                "message_type": "committed_transcript",
                "segment_index": 0,
                "text": "Word next",
            },
        }
        for seq, session in ((7, "current"), (8, "retired"))
    ]
    boundary = {
        "kind": "controller_commit",
        "status": "completed",
        "identity": {"provider_epoch_id": "epoch", "provider_turn_id": "turn"},
    }
    events = [
        {
            "kind": "native",
            "session_id": session,
            "seq": seq,
            "raw": {"message_type": "diarized_transcript", "segment_index": 0, "text": "Word next"},
        }
        for seq, session in ((9, "current"), (100, "retired"))
    ]
    capture = {
        "rows": [
            {
                "kind": "final",
                "event": {
                    "native_segment": {"provider_session_id": "current", "segment_index": 0},
                    "text": "Word next",
                },
                "receipt": {
                    "segment": segment,
                    "provider_epoch_id": "epoch",
                    "provider_turn_id": "turn",
                },
            }
        ],
        "sends": [{**boundary, "session_id": "retired"}, {**boundary, "session_id": "current"}],
        "events": [*commits, *events],
        "observations": [
            {
                "event_seq": seq,
                "kind": "diarized_transcript",
                "mapping_status": "mapped",
                "text": word,
                "word_index": i,
                "source_ranges": [{"start_s": a, "end_s": b}],
            }
            for seq, ranges in ((9, [(0.2, 0.4), (1.0, 1.3)]), (100, [(1.0, 1.3), (0.2, 0.4)]))
            for i, (word, (a, b)) in enumerate(zip(("Word", " next"), ranges, strict=True))
        ],
    }
    index = actual_source_index(capture)
    assert index["current", 0]["commit_native_event_seqs"] == [7]
    assert index["current", 0]["completed_commit_write_indices"] == [1]
    first = {"lineage": fresh_lineage()}
    second = {"lineage": fresh_lineage(chars=(4, 9))}
    assert match_interval(first, index, (1.01, 1.1)) == (
        "owned_range_intersects_without_localized_group",
        [],
    )
    status, evidence = match_interval(second, index, (1.01, 1.1))
    assert status == "localized_in_frozen_interval"
    assert [(item["event_seq"], item["word_index"]) for item in evidence] == [(9, 1)]
    assert match_interval({"lineage": fresh_lineage(kind="preview")}, index, (0.3, 0.4)) == (
        "not_complete_group",
        [],
    )
