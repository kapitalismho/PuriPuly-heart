from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from puripuly_heart.core.orchestrator.translation_turn import (
    _final_transcript_fragment,
    _final_transcript_segments,
    _native_soniox_source_groups,
)
from puripuly_heart.core.speaker_identity import native_soniox_attribution
from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun
from tests.core.test_peer_caption_lifecycle import (
    final,
    idle,
    partial,
    transcript,
    turn_loop,
)
from tests.core.test_peer_caption_lifecycle import setup as lifecycle_setup


def setup():
    harness, backend, events, clock = lifecycle_setup()
    harness.peer_owner.cloud_captions.speaker_diarization = lambda provider_id: (
        provider_id == "soniox"
    )
    return harness, backend, events, clock


def native(texts, order=1, *, session="session", states=None, languages=None, identity=None):
    item = transcript("".join(text for text, _ in texts), order, identity=identity, final=True)
    runs = tuple(
        FinalSpeakerRun(
            text,
            speaker,
            session,
            source="soniox",
            attribution_state=(states[index] if states else "identified" if speaker else "missing"),
        )
        for index, (text, speaker) in enumerate(texts)
    )
    return replace(
        item,
        final_speaker_runs=runs,
        final_language_runs=tuple(
            FinalLanguageRun(text, languages[index] if languages else "en")
            for index, (text, _) in enumerate(texts)
        ),
    )


@pytest.mark.parametrize(
    ("speakers", "expected", "representatives"),
    [
        (("A", None, "A"), ["one two three"], ["A"]),
        (("A", None, "B"), ["one two ", "three"], ["A", "B"]),
        ((None, "B", "B"), ["one two three"], ["B"]),
        ((None, None, None), ["one two three"], [None]),
        (("A", "B", "A"), ["one ", "two ", "three"], ["A", "B", "A"]),
    ],
)
def test_native_groups_conserve_original_source_and_missing_attribution(
    speakers, expected, representatives
):
    item = native(tuple(zip(("one ", "two ", "three"), speakers, strict=True)))
    groups = _native_soniox_source_groups(item)
    fragments = [
        _final_transcript_fragment(item, group, item.utterance_id, preserve_speaker_runs=True)
        for group in groups
    ]
    assert [fragment.text for fragment in fragments] == expected
    assert "".join(fragment.text for fragment in fragments) == item.text
    assert tuple(run for fragment in fragments for run in fragment.final_speaker_runs) == (
        item.final_speaker_runs
    )
    assert [
        (attribution.key.local_speaker_id if attribution.key else None)
        for fragment in fragments
        for attribution in [native_soniox_attribution(fragment.text, fragment.final_speaker_runs)]
    ] == representatives
    assert [group.source_text_start for group in groups] == [
        sum(len(text) for text in expected[:index]) for index in range(len(expected))
    ]
    assert [group.source_text_end for group in groups] == [
        sum(len(text) for text in expected[: index + 1]) for index in range(len(expected))
    ]


def test_native_groups_attach_punctuation_without_relaxing_other_segment_callers():
    item = native((("( ", None), ("one", "A"), (", ", None), ("two", "A"), ("!", "B")))
    (group,) = _native_soniox_source_groups(item)
    fragment = _final_transcript_fragment(
        item, group, item.utterance_id, preserve_speaker_runs=True
    )
    assert fragment.text == item.text
    assert fragment.final_speaker_runs == item.final_speaker_runs
    assert (
        native_soniox_attribution(fragment.text, fragment.final_speaker_runs).key.local_speaker_id
        == "A"
    )
    assert [group.text for group in _final_transcript_segments(item, split_speakers=True)] == [
        "( one, ",
        "two!",
    ]


@pytest.mark.parametrize("boundary", ["language", "session", "malformed", "mixed"])
def test_native_groups_retain_unsafe_or_incompatible_boundaries(boundary):
    item = native((("one ", "A"), ("two ", None), ("three", "A")))
    if boundary == "language":
        item = replace(
            item,
            final_language_runs=(
                FinalLanguageRun("one ", "en"),
                FinalLanguageRun("two three", "ja"),
            ),
        )
    elif boundary == "session":
        item = replace(
            item,
            final_speaker_runs=(
                item.final_speaker_runs[0],
                *(replace(run, session_scope="new") for run in item.final_speaker_runs[1:]),
            ),
        )
    else:
        item = replace(
            item,
            final_speaker_runs=(
                item.final_speaker_runs[0],
                replace(item.final_speaker_runs[1], attribution_state=boundary),
                item.final_speaker_runs[2],
            ),
        )
    groups = _native_soniox_source_groups(item)
    assert len(groups) == (2 if boundary in ("language", "session") else 3)
    assert "".join(group.text for group in groups) == item.text


def test_broken_correspondence_retains_text_and_native_fallback_evidence():
    item = native((("one two", "A"),))
    item = replace(item, final_speaker_runs=(replace(item.final_speaker_runs[0], text="wrong"),))
    (group,) = _native_soniox_source_groups(item)
    fragment = _final_transcript_fragment(
        item, group, item.utterance_id, preserve_speaker_runs=True
    )
    attribution = native_soniox_attribution(fragment.text, fragment.final_speaker_runs)
    assert fragment.text == item.text
    assert attribution.state == "malformed"
    assert attribution.source == "soniox"
    assert attribution.speaker_scope_id == "session"
    assert attribution.key is None


@pytest.mark.asyncio
@pytest.mark.parametrize("lines", [2.0, 3.0], ids=["merge", "continuation"])
async def test_known_constraint_survives_unknown_join_without_borrowing_color(monkeypatch, lines):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: lines,
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        first = native((("The first part,", "A"),))
        await final(harness, first)
        await idle(harness)
        clock.advance(0.2)
        unknown = native((("and unassigned", None),), 2)
        await partial(
            harness, replace(unknown, is_final=False, final_language_runs=(), final_speaker_runs=())
        )
        await final(harness, unknown)
        await idle(harness)
        joined = events[-1]
        if lines == 2:
            assert joined.utterance_id == first.utterance_id
            assert joined.source_text == "The first part, and unassigned"
            assert joined.speaker_assignment.attribution.key.local_speaker_id == "A"
            assert (
                joined.speaker_assignment.source_text_revision
                == hashlib.sha256(joined.source_text.encode()).hexdigest()[:16]
            )
        else:
            assert joined.utterance_id == unknown.utterance_id
            assert joined.source_text == "and unassigned"
            assert joined.speaker_assignment.attribution.state == "missing"
            assert joined.speaker_assignment.palette_index is None
            assert "## Peer continuation" in backend.calls[-1].system_prompt
        clock.advance(0.2)
        other = native((("but another speaker", "B"),), 3)
        await partial(
            harness, replace(other, is_final=False, final_language_runs=(), final_speaker_runs=())
        )
        await final(harness, other)
        await idle(harness)
        assert events[-1].utterance_id == other.utterance_id
        assert events[-1].source_text == other.text
        assert "## Peer continuation" not in backend.calls[-1].system_prompt
        assert backend.calls[-1].text == other.text
        assert events[-1].speaker_assignment.attribution.key.local_speaker_id == "B"
        assert [entry.text for entry in harness.peer_runtime.translation_history] == (
            ["The first part, and unassigned", other.text]
            if lines == 2
            else [first.text, unknown.text, other.text]
        )
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("lines", [2.0, 3.0], ids=["merge", "continuation"])
async def test_independent_unknown_resets_constraint_and_first_known_join_establishes_it(
    monkeypatch, lines
):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: lines,
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        first = native((("The first part,", "A"),))
        await final(harness, first)
        await idle(harness)
        for order, text, speaker in (
            (2, "okay", None),
            (3, "and another part", "B"),
            (4, "but original speaker", "A"),
        ):
            clock.advance(0.2)
            item = native(((text, speaker),), order)
            await partial(
                harness,
                replace(item, is_final=False, final_language_runs=(), final_speaker_runs=()),
            )
            await final(harness, item)
            await idle(harness)
        joined_source = "okay and another part" if lines == 2 else "and another part"
        assert [call.text for call in backend.calls] == [
            first.text,
            "okay",
            joined_source,
            "but original speaker",
        ]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            *(["okay"] if lines == 3 else []),
            joined_source,
            "but original speaker",
        ]
        assert ("## Peer continuation" in backend.calls[-2].system_prompt) == (lines == 3)
        assert "## Peer continuation" not in backend.calls[-1].system_prompt
        assert events[-2].speaker_assignment.attribution.key.local_speaker_id == "B"
        assert events[-2].speaker_assignment.palette_index == 1
        assert events[-1].speaker_assignment.attribution.key.local_speaker_id == "A"
        assert events[-1].speaker_assignment.palette_index == 0
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ["session-known", "session-unknown", "malformed", "mixed", "correspondence", "language"],
)
async def test_unsafe_final_or_actual_session_change_cannot_join(monkeypatch, boundary):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: 2.0,
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        first = native((("The first part,", None if boundary == "session-unknown" else "A"),))
        await final(harness, first)
        await idle(harness)
        clock.advance(0.2)
        incoming = native(
            (("and another part", "A" if boundary == "session-known" else None),),
            2,
            session="new" if boundary.startswith("session") else "session",
        )
        if boundary in ("malformed", "mixed"):
            incoming = replace(
                incoming,
                final_speaker_runs=(
                    replace(incoming.final_speaker_runs[0], attribution_state=boundary),
                ),
            )
        elif boundary == "correspondence":
            incoming = replace(
                incoming,
                final_speaker_runs=(replace(incoming.final_speaker_runs[0], text="wrong"),),
            )
        elif boundary == "language":
            incoming = replace(
                incoming, final_language_runs=(FinalLanguageRun(incoming.text, "ja"),)
            )
        await partial(
            harness,
            replace(incoming, is_final=False, final_language_runs=(), final_speaker_runs=()),
        )
        await final(harness, incoming)
        await idle(harness)
        assert [call.text for call in backend.calls] == [first.text, incoming.text]
        assert events[-1].utterance_id == incoming.utterance_id
        assert events[-1].speaker_assignment.attribution.source == "soniox"
        assert events[-1].speaker_assignment.attribution.speaker_scope_id == (
            "new" if boundary.startswith("session") else "session"
        )
        assert "## Peer continuation" not in backend.calls[-1].system_prompt
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["deepgram", "soniox"])
async def test_non_target_routes_keep_strict_source_segmentation(provider):
    harness, backend, events, _ = setup()
    try:
        await harness.start()
        item = native((("one ", "A"), ("two ", None), ("three", "A")))
        if provider == "soniox":
            item = replace(item, final_speaker_runs=())
        await final(harness, item, provider=provider)
        await idle(harness)
        assert [call.text for call in backend.calls] == (
            [item.text] if provider == "soniox" else ["one ", "two ", "three"]
        )
        if provider == "soniox":
            assert events[-1].speaker_assignment.attribution.state == "non_diarized"
            assert events[-1].speaker_assignment.attribution.source != "soniox"
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_grouped_final_replaces_preview_without_tail_barrier_and_keeps_history_order():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        item = native((("one ", "A"), ("unknown ", None), ("two ", "B"), ("three", "A")))
        await partial(
            harness, replace(item, is_final=False, final_language_runs=(), final_speaker_runs=())
        )
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert events[-1].speaker_assignment is None
        assert harness.peer_runtime.translation_history == []
        backend.block = True
        backend.failures.add("three")
        await final(harness, item)
        await turn_loop()
        assert [call.text for call in backend.calls[1:]] == ["one unknown ", "two ", "three"]
        assert [call.context for call in backend.calls[1:]] == [
            "",
            '- [peer] "one unknown "',
            '- [peer] "one unknown "\n- [peer] "two "',
        ]
        backend.completions[1].set_result(None)
        await turn_loop()
        assert [event.source_text for event in events if event.type == "translation_final"] == [
            "two "
        ]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "one unknown ",
            "two ",
            "three",
        ]
        backend.completions[0].set_result(None)
        await turn_loop()
        finals = [event for event in events if event.type == "translation_final"]
        assert [event.source_text for event in finals] == ["two ", "one unknown "]
        assert finals[1].utterance_id == item.utterance_id
        assert finals[0].utterance_id != item.utterance_id
        assert [event.speaker_assignment.palette_index for event in finals] == [1, 0]
        backend.completions[2].set_result(None)
        await idle(harness)
        finals = [event for event in events if event.type == "translation_final"]
        assert [event.source_text for event in finals] == ["two ", "one unknown ", "three"]
        assert finals[-1].text == ""
        assert finals[-1].speaker_assignment.palette_index == 0
        assert (
            "".join(entry.text for entry in harness.peer_runtime.translation_history) == item.text
        )
        assert len({event.utterance_id for event in finals}) == 3
        assert harness.osc.messages == []
    finally:
        for completion in backend.completions:
            if not completion.done():
                completion.set_result(None)
        await harness.stop()


@pytest.mark.asyncio
async def test_native_newer_repair_settles_source_and_attribution_before_stale_result(monkeypatch):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: 2.0,
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        first = native((("The first part,", None),))
        await final(harness, first)
        await idle(harness)
        assert events[-1].speaker_assignment.attribution.state == "missing"
        backend.block = True
        for order, text, speaker in ((2, "and unknown", None), (3, "and identified", "A")):
            clock.advance(0.2)
            item = native(((text, speaker),), order)
            await partial(
                harness,
                replace(item, is_final=False, final_language_runs=(), final_speaker_runs=()),
            )
            await final(harness, item)
            await turn_loop()
        backend.completions[1].set_result(None)
        await turn_loop()
        settled = events[-1]
        assert settled.utterance_id == first.utterance_id
        assert settled.peer_caption_revision == 3
        assert settled.source_text == "The first part, and unknown and identified"
        assert settled.speaker_assignment.attribution.key.local_speaker_id == "A"
        assert settled.speaker_assignment.palette_index == 0
        assert len(settled.peer_superseded_unit_ids) == 2
        backend.completions[0].set_result(None)
        await idle(harness)
        assert events[-1] is settled
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            settled.source_text
        ]
        assert backend.calls[-1].context == backend.calls[0].context
    finally:
        for completion in backend.completions:
            if not completion.done():
                completion.set_result(None)
        await harness.stop()


@pytest.mark.asyncio
async def test_ready_sibling_is_not_blocked_by_first_group_repair(monkeypatch):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: 2.0,
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        first = native((("The first part,", "A"),))
        await final(harness, first)
        await idle(harness)
        clock.advance(0.2)
        item = native((("and more ", "A"), ("other speaker", "B")), 2)
        await partial(
            harness, replace(item, is_final=False, final_language_runs=(), final_speaker_runs=())
        )
        backend.block = True
        await final(harness, item)
        await turn_loop()
        backend.completions[1].set_result(None)
        await turn_loop()
        assert [event.source_text for event in events] == [first.text, "other speaker"]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "The first part, and more ",
            "other speaker",
        ]
        backend.completions[0].set_result(None)
        await idle(harness)
        assert [event.source_text for event in events] == [
            first.text,
            "other speaker",
            "The first part, and more ",
        ]
        assert events[-1].utterance_id == first.utterance_id
        assert events[-2].utterance_id not in (first.utterance_id, item.utterance_id)
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "The first part, and more ",
            "other speaker",
        ]
        assert backend.calls[-1].context == '- [peer] "The first part, and more "'
    finally:
        for completion in backend.completions:
            if not completion.done():
                completion.set_result(None)
        await harness.stop()
