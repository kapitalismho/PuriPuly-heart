from uuid import uuid4

from puripuly_heart.core.speaker_transition import PeerSpeakerTransitionInterpreter
from puripuly_heart.domain.models import FinalSpeakerRun, Transcript


def _turn(
    speaker: str | None,
    *,
    scope: str = "socket-1",
    generation: int = 1,
    order: int = 1,
    text: str = "hello",
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> Transcript:
    return Transcript(
        utterance_id=uuid4(),
        text=text,
        is_final=True,
        channel="peer",
        final_speaker_runs=(
            FinalSpeakerRun(
                text,
                speaker,
                scope,
                source_start_ms=order * 1000 if start_ms is None else start_ms,
                source_end_ms=(order * 1000) + 900 if end_ms is None else end_ms,
                speaker_confidence=0.95,
            ),
        ),
        publication_generation=generation,
        source_order=order,
    )


def test_strict_same_context_comparisons_do_not_treat_unknown_or_reconnect_as_change() -> None:
    interpreter = PeerSpeakerTransitionInterpreter()
    first = _turn("A", order=1)
    same = _turn("A", order=2)
    changed = _turn("B", order=3)
    unknown = _turn(None, order=4)
    after_unknown = _turn("C", order=5)
    reconnect = _turn("D", scope="socket-2", order=6)

    assert interpreter.observe(first, child_sequence=0).comparison == "context_reset"
    assert interpreter.observe(same, child_sequence=0).comparison == "continuity"
    assert interpreter.observe(changed, child_sequence=0).comparison == "transition"
    assert interpreter.observe(unknown, child_sequence=0).comparison == "unavailable"
    assert interpreter.observe(after_unknown, child_sequence=0).comparison == "context_reset"
    assert interpreter.observe(reconnect, child_sequence=0).comparison == "context_reset"


def test_generation_reset_and_reordered_or_duplicate_position_withhold_transition() -> None:
    interpreter = PeerSpeakerTransitionInterpreter()
    first = _turn("A", generation=4, order=8)
    reset = _turn("B", generation=5, order=1)
    late = _turn("C", generation=5, order=1)

    assert interpreter.observe(first, child_sequence=0).comparison == "context_reset"
    assert interpreter.observe(reset, child_sequence=0).comparison == "context_reset"
    assert interpreter.observe(late, child_sequence=0).comparison == "unavailable"


def test_distinct_children_at_one_source_position_compare_in_child_order_once() -> None:
    interpreter = PeerSpeakerTransitionInterpreter()
    first = _turn("A", order=3, start_ms=3000, end_ms=3100)
    second = _turn("B", order=3, start_ms=3100, end_ms=3200)
    repeated = interpreter.observe(first, child_sequence=0)
    transition = interpreter.observe(second, child_sequence=1)

    assert repeated.comparison == "context_reset"
    assert transition.comparison == "transition"
    assert interpreter.observe(second, child_sequence=1).comparison == "unavailable"


def test_overlap_and_incomplete_timing_withhold_and_reset_reference() -> None:
    interpreter = PeerSpeakerTransitionInterpreter()

    assert (
        interpreter.observe(
            _turn("A", order=1, start_ms=100, end_ms=300), child_sequence=0
        ).comparison
        == "context_reset"
    )
    assert (
        interpreter.observe(
            _turn("B", order=2, start_ms=250, end_ms=400), child_sequence=0
        ).comparison
        == "unavailable"
    )
    incomplete = _turn("C", order=3)
    incomplete = Transcript(
        utterance_id=incomplete.utterance_id,
        text=incomplete.text,
        is_final=True,
        channel="peer",
        final_speaker_runs=(FinalSpeakerRun("hello", "C", "socket-1"),),
        publication_generation=1,
        source_order=3,
    )
    assert interpreter.observe(incomplete, child_sequence=0).comparison == "unavailable"
    assert (
        interpreter.observe(
            _turn("D", order=4, start_ms=500, end_ms=600), child_sequence=0
        ).comparison
        == "context_reset"
    )
