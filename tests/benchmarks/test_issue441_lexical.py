from __future__ import annotations

import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from benchmarks.asr.issue441_lexical import (
    PROTOCOL,
    Journal,
    ami_units,
    canonical_hash,
    compare,
    derive_entries,
    digest,
    inventory,
    ja_units,
    ko_units,
    load,
    observe,
    prepare,
    streams,
    verify_frozen,
)


def entry(text="charger", language="en", start=0, end=2):
    from benchmarks.asr.issue441_score import normalized

    return {
        "entry_id": "anchor",
        "recording_id": "fixture",
        "language": language,
        "granularity": "phrase" if language == "ja" else "word",
        "eligible": True,
        "reference_text": text,
        "normalized_reference": normalized(text),
        "start_s": start,
        "end_s": end,
        "online_cutoff_s": 3,
        "original_anchor": {"speaker": "A"},
        "source_occurrence_id": "source:word",
        "annotated_overlap": False,
    }


def word(text, start=0.1, end=0.9, identity="native", **extra):
    return {
        "text": text,
        "start_s": start,
        "end_s": end,
        "provider_start_s": start,
        "provider_end_s": end,
        "source_ranges": [{"start_s": start, "end_s": end}] if start is not None else [],
        "mapping_status": "mapped" if start is not None else "missing_provider_timing",
        "speaker": identity,
        "raw": {"type": "word"},
        **extra,
    }


class Capture:
    def __init__(self, arm="S0"):
        self.run = {
            "arm": arm,
            "recording_id": "fixture",
            "events": [],
            "observations": [],
            "snapshots": [],
            "sends": [
                {
                    "session_id": "session",
                    "status": "completed",
                    "start_s": -2,
                    "completed_s": 0,
                    "transmitted_start_s": 0,
                    "transmitted_end_s": 2,
                    "source_ranges": [{"start_s": 0, "end_s": 2}],
                    "kind": "audio",
                }
            ],
        }

    def put(
        self,
        arrival,
        key,
        words=(),
        operation="replace",
        segment=None,
        settled=False,
        session="session",
        raw=None,
    ):
        seq = len(self.run["events"])
        self.run["events"].append(
            {
                "seq": seq,
                "arrival_s": arrival,
                "session_id": session,
                "raw": raw or {},
                "kind": "native",
            }
        )
        indices = []
        for index, unit in enumerate(words):
            indices.append(len(self.run["observations"]))
            self.run["observations"].append(
                {
                    **unit,
                    "event_seq": seq,
                    "arrival_s": arrival,
                    "snapshot_key": key,
                    "revision": seq,
                    "word_index": index,
                    "segment_index": segment,
                    "session_id": session,
                    "settled": settled,
                }
            )
        self.run["snapshots"].append(
            {
                "event_seq": seq,
                "arrival_s": arrival,
                "snapshot_key": key,
                "revision": seq,
                "operation": operation,
                "observation_indices": indices,
                "segment_index": segment,
                "session_id": session,
                "settled": settled,
            }
        )

    def observation(self, reference=None, cutoff=float("inf"), mapping=None):
        primary, _ = streams(Journal(self.run, "fixture"), cutoff)
        return observe(
            primary,
            reference or entry(),
            {"session": {"native": "A"}} if mapping is None else mapping,
        )


def final_capture(words):
    capture = Capture()
    capture.put(1, "soniox:final", words, operation="append", settled=True)
    return capture


def test_common_word_support_keeps_subpiece_gaps_without_point_coverage():
    subpieces = [word("ch", 0.1, 0.2), word("arg", 0.5, 0.6), word("er.", 0.8, 0.9)]
    result = final_capture(subpieces).observation()
    assert result["outcome"] == "correct"
    assert result["occurrence_count"] == 1
    assert [piece["text"] for piece in result["localized"][0]["pieces"]] == ["ch", "arg", "er."]
    capture = Capture("E0")
    capture.put(1, "scribe:segment:0:diarized", [word("charger.")], segment=0, settled=True)
    assert capture.observation()["outcome"] == "correct"


def test_every_piece_must_intersect_independent_bounds_but_need_not_be_contained():
    assert (
        final_capture([word("no", -0.1, 0.2)]).observation(entry("no", start=0, end=1))["outcome"]
        == "correct"
    )
    capture = final_capture([word("ch", -1, 0), word("arger", 0.2, 0.9)])
    assert capture.observation()["occurrence_count"] == 0


def test_repeated_occurrences_are_time_localized_and_identity_blind():
    capture = final_capture([word("no ", 0.1, 0.3), word("no", 0.5, 0.8, "wrong")])
    result = capture.observation(entry("no"))
    assert result["reason"] == "alignment_ambiguous"
    swapped = copy.deepcopy(capture.run)
    for piece in swapped["observations"]:
        piece["speaker"] = "wrong" if piece["speaker"] == "native" else "native"
    blocks, _ = streams(Journal(swapped, "fixture"), float("inf"))
    assert (
        observe(blocks, entry("no"), {"session": {"native": "A"}})["reason"]
        == "alignment_ambiguous"
    )
    assert capture.observation(entry("no", start=0, end=0.4))["outcome"] == "correct"


@pytest.mark.parametrize(
    "pieces,text",
    [
        (["ch", "a", "rger"], "a"),
        (["x", "a"], "a"),
        (["한", "국", "어"], "국"),
        (["1", "2", "3"], "2"),
        (["the"], "he"),
        (["a", "\u0301"], "a"),
    ],
)
def test_inside_word_or_combining_boundaries_are_not_invented(pieces, text):
    result = final_capture([word(piece) for piece in pieces]).observation(
        entry(text, "ko" if text == "국" else "en")
    )
    assert result["occurrence_count"] == 0


def test_raw_spacing_and_punctuation_boundaries_are_preserved():
    capture = final_capture([word("ch "), word("a"), word("!rger")])
    assert capture.observation(entry("a"))["outcome"] == "correct"
    capture = final_capture(
        [
            word("ch"),
            word(" ", identity="foreign", raw={"type": "spacing"}),
            word("a"),
            word(".", identity="foreign"),
        ]
    )
    assert capture.observation(entry("a"))["outcome"] == "correct"


def test_normalization_composes_across_native_piece_boundaries():
    capture = final_capture([word("e", 0.1, 0.3), word("\u0301", 0.3, 0.4)])
    assert capture.observation(entry("É"))["outcome"] == "correct"
    assert len(capture.observation(entry("É"))["localized"][0]["lexical_evidence"]) == 2
    assert final_capture([word("ＮＯ!")]).observation(entry("no"))["outcome"] == "correct"


def test_untimed_negation_is_not_filtered_to_fabricate_a_match():
    capture = final_capture([word("do ", 0.1, 0.2), word("not ", None, None), word("go", 0.8, 0.9)])
    assert capture.observation(entry("do go"))["reason"] == "exact_lexical_mismatch"
    result = capture.observation(entry("do not go"))
    assert result["reason"] == "exact_text_unlocalizable_timing_or_mapping"
    assert result["occurrence_count"] == 0
    assert len(result["unlocalized_exact"][0]["lexical_evidence"]) == 3


@pytest.mark.parametrize(
    "identities,reason,missing,mixed",
    [
        (["native", "other"], "mixed_identity", False, True),
        (["native", None], "missing_speaker", True, False),
        (["native", "other", None], "mixed_identity", True, True),
    ],
)
def test_speaker_assignment_requires_every_lexical_identity(identities, reason, missing, mixed):
    capture = final_capture(
        [word(text, identity=identity) for text, identity in zip(["ch", "arg", "er"], identities)]
    )
    result = capture.observation(entry("".join(["ch", "arg", "er"][: len(identities)])))
    assert result["reason"] == reason
    assert result["missing_speaker"] is missing
    assert result["mixed_identity"] is mixed
    assert result["outcome"] == "unassigned"


def test_unmatched_native_identity_is_assigned_wrong_not_excluded():
    result = final_capture([word("no", identity="unmatched")]).observation(entry("no"))
    assert result["outcome"] == "wrong"
    assert result["unmatched_id"]


def test_partial_replace_clear_and_native_final_before_partial_order():
    capture = Capture()
    capture.put(0.2, "soniox:partial", [word("bad")])
    capture.put(0.3, "soniox:final", [word("do ")], operation="append", settled=True)
    capture.put(0.4, "soniox:partial", [word("go", 1, 1.2)])
    assert capture.observation(entry("do go"), 0.5)["outcome"] == "correct"
    assert not capture.observation(entry("do go"), 0.5)["settled"]
    capture.put(1, "soniox:partial", [])
    assert capture.observation(entry("do go"), 1.1)["occurrence_count"] == 0
    capture.put(4, "soniox:partial", [word("go", 1, 1.2)])
    assert capture.observation(entry("do go"), 3)["occurrence_count"] == 0
    assert capture.observation(entry("do go"))["outcome"] == "correct"


def test_late_indexed_metadata_and_merge_cannot_repair_cutoff():
    capture = Capture("E0")
    capture.put(
        0.5,
        "scribe:segment:0:committed_transcript",
        [word("no", None, None)],
        segment=0,
        settled=True,
    )
    capture.put(
        4, "scribe:segment:0:diarized", [word("no", identity="fragment")], segment=0, settled=True
    )
    capture.put(
        5,
        "merge",
        operation="retract",
        raw={
            "message_type": "speakers_merged",
            "from_speaker_id": "fragment",
            "to_speaker_id": "native",
        },
    )
    assert (
        capture.observation(entry("no"), 3)["reason"]
        == "exact_text_unlocalizable_timing_or_mapping"
    )
    assert capture.observation(entry("no"), 4.5)["outcome"] == "wrong"
    assert capture.observation(entry("no"))["outcome"] == "correct"


@pytest.mark.parametrize("timestamp_first", [False, True])
def test_unindexed_timestamp_receipt_orders_never_supply_indexed_identity(timestamp_first):
    capture = Capture("E0")
    capture.put(0.2, "scribe:segment:0:committed_transcript", [word("no", None, None)], segment=0)
    operations = [
        ("scribe:unindexed:1", [word("no", 0.1, 0.8, None)], None),
        ("scribe:segment:0:diarized", [word("no", 0.2, 0.7)], 0),
    ]
    if not timestamp_first:
        operations.reverse()
    for index, (key, words, segment) in enumerate(operations):
        capture.put(0.5 + index, key, words, segment=segment, settled=True)
    journal = Journal(capture.run, "fixture")
    primary, diagnostic = streams(journal, float("inf"))
    assert observe(primary, entry("no"), {"session": {"native": "A"}})["outcome"] == "correct"
    assert len(diagnostic) == 1
    assert diagnostic[0]["units"][0]["start_s"] == 0.1
    if timestamp_first:
        primary, diagnostic = streams(journal, 1)
        assert (
            observe(primary, entry("no"), {})["reason"]
            == "exact_text_unlocalizable_timing_or_mapping"
        )
        assert observe(diagnostic, entry("no"), {})["reason"] == "missing_speaker"


def test_indexed_protocol_order_missing_segment_barrier_and_session_fence():
    capture = Capture("E0")
    capture.put(0.2, "scribe:segment:1:diarized", [word("go", 1, 1.2)], segment=1)
    capture.put(0.3, "scribe:segment:0:diarized", [word("do ")], segment=0)
    assert capture.observation(entry("do go"))["outcome"] == "correct"
    capture = Capture("E0")
    capture.put(0.2, "scribe:segment:0:diarized", [word("do ")], segment=0)
    capture.put(0.3, "scribe:segment:2:diarized", [word("go", 1, 1.2)], segment=2)
    assert capture.observation(entry("do go"))["occurrence_count"] == 0
    capture = Capture()
    capture.put(0.2, "soniox:final", [word("do ")], operation="append", session="one")
    capture.put(0.3, "soniox:partial", [word("go", 1, 1.2)], session="two")
    assert capture.observation(entry("do go"))["occurrence_count"] == 0


@pytest.mark.parametrize(
    "status",
    [
        "unmapped_padding_or_unsent",
        "discontinuous_or_partial_source_mapping",
        "missing_provider_timing",
    ],
)
def test_mapping_failures_are_preserved_without_full_ledger_repair(status):
    capture = final_capture([word("no", mapping_status=status)])
    capture.run["sends"] = [
        {
            "status": "completed",
            "source_start_s": 0,
            "source_end_s": 2,
            "transmitted_start_s": 0,
            "transmitted_end_s": 2,
        }
    ]
    result = capture.observation(entry("no"))
    assert result["occurrence_count"] == 0
    assert result["unlocalized_exact"][0]["pieces"][0]["mapping_status"] == status


def test_source_discontinuity_and_backward_replay_are_not_healed():
    capture = final_capture(
        [word("no", source_ranges=[{"start_s": 0.1, "end_s": 0.3}, {"start_s": 0.6, "end_s": 0.9}])]
    )
    assert capture.observation(entry("no"))["occurrence_count"] == 0
    capture = final_capture([word("do ", 0.6, 0.8), word("go", 0.1, 0.3)])
    assert (
        capture.observation(entry("do go"))["unlocalized_exact"][0]["reason"]
        == "backward_source_or_provider_progression"
    )
    capture = final_capture(
        [
            word("no ", 0.1, 0.3),
            word("no", 0.1, 0.3, "other", provider_start_s=1, provider_end_s=1.2),
        ]
    )
    assert capture.observation(entry("no"))["reason"] == "alignment_ambiguous"


def sent(start, end, ranges, completed=1, status="completed", session="session", **extra):
    return {
        "session_id": session,
        "status": status,
        "start_s": completed - 0.01,
        "completed_s": completed,
        "transmitted_start_s": start,
        "transmitted_end_s": end,
        "source_ranges": [
            {"start_s": source_start, "end_s": source_end} for source_start, source_end in ranges
        ],
        "kind": "audio",
        **extra,
    }


def bridge_capture(sends, right_source=(0.8, 1), right_provider=(0.8, 1)):
    capture = Capture("S1")
    capture.run["sends"] = sends
    capture.put(0.4, "soniox:final", [word("do ", 0.1, 0.3)], operation="append", settled=True)
    capture.put(
        1.2,
        "soniox:final",
        [
            word(
                "go",
                *right_source,
                provider_start_s=right_provider[0],
                provider_end_s=right_provider[1],
            )
        ],
        operation="append",
        settled=True,
    )
    return capture


def test_causal_completed_silence_bridge_preserves_captured_piece_mappings():
    sends = [
        sent(0, 0.3, [(0, 0.3)], completed=0.35),
        sent(0.3, 0.8, [(0.3, 0.8)], completed=0.9),
        sent(0.8, 1, [(0.8, 1)], completed=1.1),
    ]
    capture = bridge_capture(sends)
    before = copy.deepcopy(capture.run)
    result = capture.observation(entry("do go"), 1.3)
    assert result["outcome"] == "correct"
    assert capture.run == before
    proof = result["localized"][0]["bridge_proofs"][0]
    assert proof["cutoff_s"] == 1.2
    assert proof["supported"]
    assert [row["send"]["completed_s"] for row in proof["sends"]] == [0.35, 0.9, 1.1]
    assert [piece["source_ranges"] for piece in result["localized"][0]["pieces"]] == [
        before["observations"][0]["source_ranges"],
        before["observations"][1]["source_ranges"],
    ]


@pytest.mark.parametrize(
    "sends,right_source,right_provider,reason",
    [
        (
            [sent(0, 0.3, [(0, 0.3)], completed=0.35), sent(0.3, 0.5, [(0.8, 1)])],
            (0.8, 1),
            (0.3, 0.5),
            "omitted_or_replayed_source_in_bridge",
        ),
        (
            [
                sent(0, 0.3, [(0, 0.3)], completed=0.35),
                sent(0.3, 0.8, [], kind="padding", generated=True),
                sent(0.8, 1, [(0.3, 0.5)]),
            ],
            (0.3, 0.5),
            (0.8, 1),
            "inserted_padding_in_bridge",
        ),
        (
            [
                sent(0, 0.3, [(0, 0.3)], completed=0.35),
                sent(0.3, 0.8, [], kind="padding", generated=True),
                sent(0.8, 1, [(0.8, 1)]),
            ],
            (0.8, 1),
            (0.8, 1),
            "inserted_padding_in_bridge",
        ),
        (
            [sent(0, 0.3, [(0, 0.3)], completed=0.35), sent(0.8, 1, [(0.8, 1)])],
            (0.8, 1),
            (0.8, 1),
            "missing_or_ambiguous_transmitted_bridge",
        ),
        (
            [],
            (0.8, 1),
            (0.8, 1),
            "missing_causal_completed_bridge",
        ),
    ],
)
def test_mapping_breaks_cannot_manufacture_cross_piece_occurrence(
    sends, right_source, right_provider, reason
):
    result = bridge_capture(sends, right_source, right_provider).observation(entry("do go"))
    assert result["outcome"] == "unassigned"
    assert result["occurrence_count"] == 0
    candidate = result["unlocalized_exact"][0]
    assert candidate["reason"] == "cross_piece_source_bridge_unsupported"
    assert candidate["bridge_proofs"][0]["reason"] == reason


@pytest.mark.parametrize(
    "extra",
    [
        {"completed": 2},
        {"status": "failed"},
        {"session": "another-session"},
        {"completed": None},
    ],
)
def test_late_failed_or_unproven_bridge_cannot_repair_captured_occurrence(extra):
    middle = sent(0.3, 0.8, [(0.3, 0.8)])
    if "completed" in extra:
        middle["completed_s"] = extra["completed"]
    if "status" in extra:
        middle["status"] = extra["status"]
    if "session" in extra:
        middle["session_id"] = extra["session"]
    sends = [
        sent(0, 0.3, [(0, 0.3)], completed=0.35),
        middle,
        sent(0.8, 1, [(0.8, 1)], completed=1.1),
    ]
    capture = bridge_capture(sends)
    assert capture.observation(entry("do go"), 1.3)["occurrence_count"] == 0
    assert capture.observation(entry("do go"))["occurrence_count"] == 0


def test_future_send_cannot_validate_an_already_captured_mapped_piece():
    capture = bridge_capture(
        [
            sent(0, 0.3, [(0, 0.3)], completed=0.5),
            sent(0.3, 0.8, [(0.3, 0.8)], completed=0.9),
            sent(0.8, 1, [(0.8, 1)], completed=1.1),
        ]
    )
    proof = capture.observation(entry("do go"))["unlocalized_exact"][0]["bridge_proofs"][0]
    assert proof["reason"] == "send_not_completed_at_piece_capture"


def test_native_piece_overlap_is_allowed_with_causal_contiguous_source_proof():
    capture = final_capture([word("do ", 0.1, 0.6), word("go", 0.4, 0.9)])
    result = capture.observation(entry("do go"))
    assert result["outcome"] == "correct"
    assert result["localized"][0]["bridge_proofs"][0]["supported"]


def test_backward_provider_progression_remains_unsupported():
    capture = final_capture(
        [
            word("do ", 0.1, 0.3),
            word("go", 0.8, 1, provider_start_s=0.05, provider_end_s=0.25),
        ]
    )
    result = capture.observation(entry("do go"))
    assert result["occurrence_count"] == 0
    assert result["unlocalized_exact"][0]["reason"] == "backward_source_or_provider_progression"


def make_ami(path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "corpusResources/meetings.xml",
            '<meetings><meeting observation="ES2004d"><speaker nxt_agent="A" global_name="A"/></meeting></meetings>',
        )
        archive.writestr(
            "words/ES2004d.A.words.xml",
            '<words xmlns:nite="http://nite.sourceforge.net/"><w nite:id="w0" starttime="0.1" endtime="0.9">no</w><gap nite:id="g0"/><w nite:id="w1" starttime="1.1" endtime="1.3" trunc="true">n</w></words>',
        )
        archive.writestr(
            "segments/ES2004d.A.segments.xml",
            '<segments xmlns:nite="http://nite.sourceforge.net/"><segment nite:id="s0"><child href="words#id(w0)..id(w1)"/></segment></segments>',
        )


def source_record(path, language="en", parent_available=False):
    return {
        "id": "fixture",
        "language": language,
        "duration_s": 2,
        "source": {"annotation_path": path.name, "annotation_sha256": digest(path), "offset_s": 0},
        "anchors": [
            {
                "id": "anchor",
                "time_s": 0.5,
                "span_id": "parent",
                "speaker": "A",
                "span_end_s": 1.5,
                "reference_word": "no",
            }
        ],
        "spans": [
            {
                "id": "parent",
                "source_annotation_id": "s0",
                "start_s": 0,
                "end_s": 1.5,
                "source_start_s": 0,
                "source_end_s": 1.5,
                "speaker": "A",
                "text": "no n",
                "reference_available": parent_available,
            }
        ],
    }


def test_ami_clean_word_eligibility_is_not_parent_coarse_availability(tmp_path):
    path = tmp_path / "ami.zip"
    make_ami(path)
    assert ami_units(path)["s0"][0]["clean"]
    record = source_record(path)
    derived = derive_entries({"recordings": [record]}, tmp_path)
    assert derived[0]["eligible"]
    assert not derived[0]["original_parent"]["reference_available"]
    assert derived[0]["source_unit"]["id"] == "w0"
    assert derived[0]["online_cutoff_s"] == 2.5
    assert derived[0]["original_anchor"] == record["anchors"][0]
    assert not ami_units(path)["s0"][1]["clean"]


def test_source_overlap_repetition_and_duplicate_occurrences_remain_visible(tmp_path):
    path = tmp_path / "ami.zip"
    make_ami(path)
    record = source_record(path)
    record["spans"].append(
        {
            "id": "other",
            "speaker": "B",
            "source_start_s": 0.2,
            "source_end_s": 0.3,
        }
    )
    record["anchors"].append({**record["anchors"][0], "id": "second-anchor"})
    entries = derive_entries({"recordings": [record]}, tmp_path)
    assert len(entries) == 2
    assert all(unit["annotated_overlap"] for unit in entries)
    assert all(unit["duplicate_source_unit"] for unit in entries)
    assert all(unit["same_normalized_source_text_count"] == 1 for unit in entries)
    assert inventory(entries)["unique_source_units"] == 1
    assert inventory(entries)["eligible"] == 2


def test_ko_duplicate_named_tier_selects_item_five_hangul(tmp_path):
    path = tmp_path / "ko.TextGrid"
    path.write_text(
        'item [5]:\n name = "pWord.ortho."\n intervals [7]:\n xmin = 0.1\n xmax = 0.9\n text = "안녕"\nitem [6]:\n name = "pWord.ortho."\n intervals [7]:\n xmin = 0.1\n xmax = 0.9\n text = "annyeong"\n',
        encoding="utf-16",
    )
    units = ko_units(path)
    assert len(units) == 1
    assert units[0]["text"] == "안녕"
    assert units[0]["id"] == "item[5]/intervals[7]"
    record = source_record(path, "ko")
    assert derive_entries({"recordings": [record]}, tmp_path)[0]["reference_text"] == "안녕"


def test_ja_tagged_phrase_is_not_stripped_or_subselected(tmp_path):
    path = tmp_path / "ja.eaf"
    path.write_text(
        '<ANNOTATION_DOCUMENT><TIME_ORDER><TIME_SLOT TIME_SLOT_ID="t0" TIME_VALUE="0"/><TIME_SLOT TIME_SLOT_ID="t1" TIME_VALUE="1500"/></TIME_ORDER><TIER TIER_ID="A"><ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="s0" TIME_SLOT_REF1="t0" TIME_SLOT_REF2="t1"><ANNOTATION_VALUE>あ&lt;tag&gt;の</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION></TIER></ANNOTATION_DOCUMENT>',
        encoding="utf-8",
    )
    assert not ja_units(path)["s0"]["clean"]
    record = source_record(path, "ja")
    record["spans"][0]["text"] = "あ<tag>の"
    result = derive_entries({"recordings": [record]}, tmp_path)[0]
    assert not result["eligible"]
    assert result["reference_text"] == "あ<tag>の"
    assert result["eligibility_reason"] == "tagged_or_truncated_source_unit"


def test_source_hash_checked_before_derivation(tmp_path):
    path = tmp_path / "ami.zip"
    make_ami(path)
    record = source_record(path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source hash changed"):
        derive_entries({"recordings": [record]}, tmp_path)


def synthetic_frozen(tmp_path):
    annotation = tmp_path / "ami.zip"
    make_ami(annotation)
    manifest = {"recordings": [source_record(annotation)]}
    capture = final_capture([word("n", 0.1, 0.4), word("o", 0.6, 0.9)])
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(capture.run), encoding="utf-8")
    score = {
        "arms": {
            "S0": {
                "recordings": [
                    {
                        "recording_id": "fixture",
                        "mapping": {"session": {"native": "A"}},
                        "log_sha256": digest(run_path),
                    }
                ]
            }
        }
    }
    paths = {"manifest.json": manifest, "score.json": score}
    for name, value in paths.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    entries = derive_entries(manifest, tmp_path)
    frozen = {
        "protocol": PROTOCOL,
        "protocol_sha256": canonical_hash(PROTOCOL),
        "original_manifest": "manifest.json",
        "original_score": "score.json",
        "entries": entries,
        "inventory": inventory(entries),
        "preserved_files": [
            {"path": name, "sha256": digest(tmp_path / name)} for name in (*paths, "run.json")
        ],
        "runs": [
            {
                "arm": "S0",
                "recording_id": "fixture",
                "path": "run.json",
                "sha256": digest(run_path),
                "mapping": score["arms"]["S0"]["recordings"][0]["mapping"],
            }
        ],
    }
    return frozen


def test_compare_keeps_original_files_maps_cutoffs_and_source_denominators(tmp_path):
    frozen = synthetic_frozen(tmp_path)
    original = copy.deepcopy(frozen)
    result = compare(frozen, tmp_path)
    assert frozen == original
    assert result["invariance_before"] == result["invariance_after"]
    assert [row["cutoff_s"] for row in result["rows"]] == [2.5, None]
    assert all(row["outcome"] == "correct" for row in result["rows"])
    assert all(row["eligible"] == 1 and row["correct"] == 1 for row in result["summary"])
    frozen["runs"][0]["mapping"] = {"session": {"native": "B"}}
    with pytest.raises(ValueError, match="Fixed original mapping changed"):
        verify_frozen(frozen, tmp_path)


@pytest.mark.parametrize("filename", ["manifest.json", "score.json", "run.json"])
def test_comparison_rejects_changed_original_evidence(tmp_path, filename):
    frozen = synthetic_frozen(tmp_path)
    (tmp_path / filename).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Preserved file changed"):
        compare(frozen, tmp_path)


def test_comparison_rejects_changed_reference_cutoff(tmp_path):
    frozen = synthetic_frozen(tmp_path)
    frozen["entries"][0]["online_cutoff_s"] = 0.5
    with pytest.raises(ValueError, match="Source-only entries changed"):
        compare(frozen, tmp_path)


def test_whole_japanese_phrase_is_not_an_easier_subphrase():
    capture = final_capture([word("ない"), word("かも", 1, 1.2)])
    assert capture.observation(entry("ないかも", "ja"))["outcome"] == "correct"
    assert (
        capture.observation(entry("ないかもしれない", "ja"))["reason"] == "exact_lexical_mismatch"
    )


def test_source_ineligible_inventory_does_not_become_provider_failure(tmp_path):
    frozen = synthetic_frozen(tmp_path)
    excluded = copy.deepcopy(frozen["entries"][0])
    excluded.update(entry_id="excluded", eligible=False, eligibility_reason="no_lexical_reference")
    frozen["entries"].append(excluded)
    from benchmarks.asr.issue441_lexical import summarize

    rows = [
        {
            "arm": "S0",
            "view": "online",
            "entry_id": "anchor",
            "outcome": "unassigned",
            "reason": "missing_text",
        },
        {
            "arm": "S0",
            "view": "online",
            "entry_id": "excluded",
            "outcome": "source_ineligible",
            "reason": "no_lexical_reference",
        },
    ]
    summary = summarize(rows, frozen["entries"])
    assert all(row["eligible"] == 1 and row["unassigned"] == 1 for row in summary)


def test_cli_comparison_on_isolated_synthetic_files_is_exclusive(tmp_path):
    frozen = synthetic_frozen(tmp_path)
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    output = tmp_path / "result.json"
    script = Path(__file__).resolve().parents[2] / "benchmarks/asr/issue441_lexical.py"
    command = [
        sys.executable,
        str(script),
        "compare",
        "--frozen",
        str(frozen_path),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert load(output)["frozen_inputs_sha256"] == digest(frozen_path)
    assert len(load(output)["rows"]) == 2
    assert all(row["localized"][0]["bridge_proofs"][0]["supported"] for row in load(output)["rows"])
    repeated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr


def test_production_preparation_rejects_changed_original_hashes(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Original manifest hash changed"):
        prepare(tmp_path, path, path, path, path)
