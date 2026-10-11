from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.core.speaker_identity import PeerSpeakerIdentityAllocator
from puripuly_heart.core.stt.backend import STTRecognitionUnit
from puripuly_heart.core.stt.recognition_units import recognition_transcript
from puripuly_heart.domain.models import (
    FinalSpeakerRun,
    SpeakerAssignment,
    SpeakerAttribution,
    SpeakerKey,
    Transcript,
)
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
from puripuly_heart.ui.overlay_calibration import OverlayCalibration


def _assignment(scope: str, speaker: str | None, order: int, index: int | None):
    attribution = SpeakerAttribution(
        "identified" if speaker is not None else "missing",
        SpeakerKey("soniox", scope, speaker) if speaker is not None else None,
        "soniox",
        scope,
    )
    return SpeakerAssignment(attribution, (1, order, 0), index)


def _peer_event(adapter, turn_id, text, assignment):
    return adapter.translation_final(
        utterance_id=turn_id,
        channel="peer",
        text=text,
        source_text=text,
        source_language="en",
        target_language="ko",
        applied_context_mode="integrated",
        speaker_assignment=assignment,
    )


@pytest.mark.asyncio
async def test_first_readable_color_is_immutable_across_updates_self_and_replay() -> None:
    clock = FakeClock(_now=20.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    turn, self_turn = uuid4(), uuid4()
    await presenter.emit(_peer_event(adapter, turn, "first", _assignment("scope", None, 1, None)))
    visible_since = presenter._entries[("peer", turn)].visible_since
    assert presenter.snapshot().blocks[0].speaker_style == "gray"
    await presenter.emit(_peer_event(adapter, turn, "revised", _assignment("scope", "A", 1, 0)))
    await presenter.emit(
        adapter.transcript_final(
            Transcript(self_turn, "self", True, channel="self"),
            source_language="en",
            target_language="ko",
        )
    )
    blocks = {block.id: block for block in presenter.snapshot().blocks}
    assert blocks[f"peer:{turn}"].speaker_style == "gray"
    assert blocks[f"self:{self_turn}"].speaker_style is None
    assert presenter._entries[("peer", turn)].visible_since == visible_since
    assert presenter.snapshot().to_dict()["blocks"][0]["speaker_style"] == "gray"
    await presenter.close()


@pytest.mark.asyncio
async def test_non_diarized_gold_does_not_block_identified_speaker_scope() -> None:
    clock = FakeClock(_now=25.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    off, identified, unknown = uuid4(), uuid4(), uuid4()
    off_assignment = SpeakerAssignment(SpeakerAttribution("non_diarized"), (1, 1, 0))
    await presenter.emit(_peer_event(adapter, off, "off", off_assignment))
    await presenter.emit(_peer_event(adapter, identified, "on", _assignment("new", "A", 2, 0)))
    assert [block.speaker_style for block in presenter.snapshot().blocks] == ["gold", "gold"]
    await presenter.emit(
        _peer_event(adapter, unknown, "unknown", _assignment("new", None, 3, None))
    )
    assert [block.speaker_style for block in presenter.snapshot().blocks] == ["gold", "gray"]
    await presenter.close()


@pytest.mark.asyncio
async def test_gemini_peer_without_speaker_evidence_uses_gray_across_readable_updates() -> None:
    clock = FakeClock(_now=26.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    allocator = PeerSpeakerIdentityAllocator()
    stream = RecognitionStreamIdentity("peer", 1, 1, "epoch", ("gemini",))
    unit = STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), "stream words")
    transcript = recognition_transcript(unit, created_at=clock.now(), publication_order=1)
    assignment = allocator.observe(transcript, child_sequence=0)

    await presenter.emit(
        adapter.transcript_final(
            transcript,
            source_language="en",
            target_language="ko",
            speaker_assignment=assignment,
        )
    )
    await presenter.emit(
        _peer_event(
            adapter,
            transcript.utterance_id,
            "translation",
            assignment,
        )
    )

    assert assignment.attribution.state == "not_ready"
    assert presenter.snapshot().blocks[0].speaker_style == "gray"
    await presenter.emit(_peer_event(adapter, transcript.utterance_id, "revised", assignment))
    assert presenter.snapshot().blocks[0].speaker_style == "gray"
    await presenter.close()


@pytest.mark.asyncio
async def test_speaker_identity_color_survives_non_diarized_turn() -> None:
    clock = FakeClock(_now=27.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    allocator = PeerSpeakerIdentityAllocator()
    for order, speaker, expected_style in (
        (1, "A", "gold"),
        (2, "B", "cyan"),
        (3, None, "gray"),
        (4, "B", "cyan"),
    ):
        text = f"turn {order}"
        runs = (
            (
                FinalSpeakerRun(
                    text,
                    speaker,
                    "session",
                    source="soniox",
                    attribution_state="identified",
                ),
            )
            if speaker is not None
            else ()
        )
        transcript = Transcript(
            uuid4(),
            text,
            True,
            channel="peer",
            final_speaker_runs=runs,
            publication_generation=1,
            source_order=order,
        )
        assignment = allocator.observe(transcript, child_sequence=0)
        await presenter.emit(_peer_event(adapter, transcript.utterance_id, text, assignment))
        assert presenter.snapshot().blocks[-1].speaker_style == expected_style
    await presenter.close()


@pytest.mark.asyncio
async def test_scope_handoff_keeps_old_color_and_grays_new_until_old_leaves() -> None:
    clock = FakeClock(_now=30.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    old, new, later = uuid4(), uuid4(), uuid4()
    await presenter.emit(_peer_event(adapter, old, "old readable", _assignment("old", "A", 1, 0)))
    await presenter.emit(_peer_event(adapter, new, "new readable", _assignment("new", "A", 2, 0)))
    blocks = {block.id: block for block in presenter.snapshot().blocks}
    assert blocks[f"peer:{old}"].speaker_style == "gold"
    assert blocks[f"peer:{new}"].speaker_style == "gray"
    clock.advance(1)
    await presenter.emit(_peer_event(adapter, later, "new later", _assignment("new", "A", 3, 0)))
    assert presenter.snapshot().blocks[-1].speaker_style == "gold"
    assert presenter._entries[("peer", new)].speaker_style == "gray"
    await presenter.close()


@pytest.mark.asyncio
async def test_late_old_scope_result_cannot_reclaim_new_palette() -> None:
    clock = FakeClock(_now=40.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    new, old = uuid4(), uuid4()
    await presenter.emit(_peer_event(adapter, new, "new", _assignment("new", "A", 3, 0)))
    await presenter.emit(_peer_event(adapter, old, "late old", _assignment("old", "A", 2, 0)))
    blocks = {block.id: block for block in presenter.snapshot().blocks}
    assert blocks[f"peer:{new}"].speaker_style == "gold"
    assert blocks[f"peer:{old}"].speaker_style == "gray"
    await presenter.close()


def _overflow(scope: str, speaker: str, order: int):
    return replace(_assignment(scope, speaker, order, None), palette_overflow=True)


async def _divider_after(events) -> tuple[bool, dict[str, str | None]]:
    clock = FakeClock(_now=50.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    for kind, assignment in events:
        clock.advance(1)
        turn = uuid4()
        if kind == "self":
            await presenter.emit(
                adapter.transcript_final(
                    Transcript(turn, "self words", True, channel="self"),
                    source_language="en",
                    target_language="ko",
                )
            )
        else:
            await presenter.emit(_peer_event(adapter, turn, f"peer {turn}", assignment))
    snapshot = presenter.snapshot()
    styles = {block.id: block.speaker_style for block in snapshot.blocks}
    assert snapshot.to_dict().get("speaker_divider", False) is snapshot.speaker_divider
    await presenter.close()
    return snapshot.speaker_divider, styles


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "expected"),
    [
        pytest.param(
            [("peer", _overflow("s", "E", 5)), ("peer", _overflow("s", "F", 6))],
            True,
            id="overflow-E+overflow-F",
        ),
        pytest.param(
            [("peer", _overflow("s", "F", 5)), ("peer", _overflow("s", "E", 6))],
            True,
            id="swapped-order",
        ),
        pytest.param(
            [("peer", _overflow("s", "E", 5)), ("peer", _overflow("s", "E", 6))],
            False,
            id="E+E",
        ),
        pytest.param(
            [("peer", _assignment("s", None, 5, None)), ("peer", _overflow("s", "E", 6))],
            False,
            id="unknown+E",
        ),
        pytest.param(
            [("peer", _assignment("s", None, 5, None)), ("peer", _assignment("s", None, 6, None))],
            False,
            id="unknown+unknown",
        ),
        pytest.param(
            [("peer", _overflow("old", "E", 5)), ("peer", _overflow("new", "F", 6))],
            False,
            id="old-scope-overflow+new-scope-overflow",
        ),
        pytest.param(
            [("peer", _assignment("s", "A", 5, 0)), ("peer", _overflow("s", "E", 6))],
            False,
            id="gold+E",
        ),
        pytest.param(
            [("self", None), ("peer", _overflow("s", "E", 6))],
            False,
            id="self+E",
        ),
        pytest.param([("peer", _overflow("s", "E", 5))], False, id="single"),
        pytest.param(
            [("peer", None), ("peer", _overflow("s", "E", 6))],
            False,
            id="not-ready+E",
        ),
    ],
)
async def test_speaker_divider_matrix(events, expected: bool) -> None:
    divider, styles = await _divider_after(events)
    assert divider is expected
    assert len(styles) == len(events)
    for kind, assignment in events:
        if kind == "peer" and assignment is not None and assignment.palette_overflow:
            assert "gray" in styles.values()


@pytest.mark.asyncio
async def test_handoff_gray_next_to_overflow_draws_no_divider() -> None:
    clock = FakeClock(_now=60.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    old, handoff, overflow = uuid4(), uuid4(), uuid4()
    await presenter.emit(_peer_event(adapter, old, "old", _assignment("old", "A", 1, 0)))
    await presenter.emit(_peer_event(adapter, handoff, "handoff", _assignment("new", "X", 2, 0)))
    assert presenter._entries[("peer", handoff)].speaker_gray_reason == "scope_handoff"
    clock.advance(1)
    await presenter.emit(_peer_event(adapter, overflow, "overflow", _overflow("new", "F", 3)))
    snapshot = presenter.snapshot()
    assert [block.id for block in snapshot.blocks] == [f"peer:{handoff}", f"peer:{overflow}"]
    assert presenter._entries[("peer", overflow)].speaker_gray_reason == "palette_overflow"
    assert snapshot.speaker_divider is False
    await presenter.close()


@pytest.mark.asyncio
async def test_gray_reasons_keep_overflow_distinct_from_missing_attribution() -> None:
    clock = FakeClock(_now=70.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    missing, overflow = uuid4(), uuid4()
    await presenter.emit(_peer_event(adapter, missing, "m", _assignment("s", None, 1, None)))
    await presenter.emit(_peer_event(adapter, overflow, "o", _overflow("s", "E", 2)))
    assert presenter._entries[("peer", missing)].speaker_gray_reason == "missing"
    assert presenter._entries[("peer", overflow)].speaker_gray_reason == "palette_overflow"
    assert {block.speaker_style for block in presenter.snapshot().blocks} == {"gray"}
    await presenter.close()


@pytest.mark.asyncio
async def test_divider_is_frozen_against_late_attribution_metadata() -> None:
    clock = FakeClock(_now=80.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    first, second = uuid4(), uuid4()
    await presenter.emit(_peer_event(adapter, first, "e", _overflow("s", "E", 5)))
    await presenter.emit(_peer_event(adapter, second, "f", _overflow("s", "F", 6)))
    assert presenter.snapshot().speaker_divider is True
    revision = presenter.snapshot().revision
    await presenter.emit(_peer_event(adapter, first, "e revised", _assignment("s", "E", 5, 0)))
    snapshot = presenter.snapshot()
    assert snapshot.revision > revision
    assert snapshot.speaker_divider is True
    assert {block.speaker_style for block in snapshot.blocks} == {"gray"}
    await presenter.close()

    clock = FakeClock(_now=90.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    await presenter.emit(_peer_event(adapter, first, "e", None))
    await presenter.emit(_peer_event(adapter, second, "f", _overflow("s", "F", 6)))
    await presenter.emit(_peer_event(adapter, first, "e late", _overflow("s", "E", 5)))
    assert presenter.snapshot().speaker_divider is False
    await presenter.close()


@pytest.mark.asyncio
async def test_divider_leaves_when_either_block_expires() -> None:
    clock = FakeClock(_now=100.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    first, second = uuid4(), uuid4()
    await presenter.emit(_peer_event(adapter, first, "e", _overflow("s", "E", 5)))
    clock.advance(3)
    await presenter.emit(_peer_event(adapter, second, "f", _overflow("s", "F", 6)))
    assert presenter.snapshot().speaker_divider is True
    clock.advance(30)
    await presenter.emit(_peer_event(adapter, uuid4(), "g", _assignment("s", None, 7, None)))
    snapshot = presenter.snapshot()
    assert f"peer:{first}" not in {block.id for block in snapshot.blocks}
    assert snapshot.speaker_divider is False
    await presenter.close()
