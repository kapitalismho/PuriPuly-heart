from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
)
from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.self_translation_channel import (
    SelfTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationOutputSubmission,
    TranslationTurnChild,
    TranslationTurnLifecycleOwner,
    TranslationTurnProcessResult,
    TranslationTurnRequest,
)
from puripuly_heart.core.translation_policy import TranslationRuntimePolicy
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import FinalLanguageRun, Transcript, Translation
from tests.helpers.translation_owners import compose_translation_test_harness


class RecordingOutput:
    def __init__(self, trace: list[tuple[object, ...]] | None = None) -> None:
        self.submissions: list[TranslationOutputSubmission] = []
        self.trace = trace

    async def submit_translation_output(self, submission: TranslationOutputSubmission) -> None:
        self.submissions.append(submission)
        if self.trace is not None:
            self.trace.append(("output", submission.sequence, submission.outcome))


def _translated_result(child: TranslationTurnChild) -> TranslationTurnProcessResult:
    return TranslationTurnProcessResult(
        "translated",
        TranslationOutputSubmission(
            parent_utterance_id=child.parent_utterance_id,
            child_utterance_id=child.utterance_id,
            sequence=child.sequence,
            channel=child.channel,
            source=child.source,
            source_text=child.transcript.text,
            source_language="en",
            target_language=child.target_language,
            outcome="translated",
            config_snapshot=child.config_snapshot,
            translation=Translation(
                utterance_id=child.utterance_id,
                text="안녕",
                source_text=child.transcript.text,
                source_language="en",
                target_language=child.target_language,
                channel=child.channel,
            ),
        ),
    )


def _request(
    *,
    parent_id: UUID,
    turn_kind: str,
    runs: tuple[FinalLanguageRun, ...] = (),
    targets: tuple[str, ...] = ("ko",),
) -> TranslationTurnRequest:
    channel = "peer" if turn_kind == "peer" else "self"
    return TranslationTurnRequest(
        transcript=Transcript(
            utterance_id=parent_id,
            text="".join(run.text for run in runs) if runs else "hello",
            is_final=True,
            channel=channel,
            final_language_runs=runs,
        ),
        source="Peer" if channel == "peer" else "You",
        turn_kind=turn_kind,
        target_languages=targets,
        config_snapshot=TranslationRuntimeConfigSnapshot(
            revision=0,
            value=TranslationRuntimeConfig(),
        ),
    )


def _owner(
    *,
    process_child=None,
    output=None,
    trace=None,
    predecessor_wait_observer=None,
) -> TranslationTurnLifecycleOwner:
    events = trace if trace is not None else []

    async def created(child: TranslationTurnChild) -> None:
        events.append(("created", child.sequence, child.utterance_id))

    async def started(child: TranslationTurnChild, _task: asyncio.Task[object]) -> None:
        events.append(("started", child.sequence, child.utterance_id))

    async def process(child: TranslationTurnChild, cancellation_requested):
        if process_child is not None:
            return await process_child(child, cancellation_requested)
        return "translated"

    async def terminal(child: TranslationTurnChild, outcome: str) -> None:
        events.append(("terminal", child.sequence, outcome))

    async def closed(parent_id: UUID) -> None:
        events.append(("closed", parent_id))

    async def rejected(parent_id: UUID) -> None:
        events.append(("rejected", parent_id))

    return TranslationTurnLifecycleOwner(
        on_child_created=created,
        on_child_started=started,
        process_child=process,
        on_child_terminal=terminal,
        on_parent_closed=closed,
        on_parent_rejected=rejected,
        predecessor_wait_observer=predecessor_wait_observer,
        output=output,
    )


def test_policy_rejects_retired_fast_translation_off_choice() -> None:
    with pytest.raises(ValueError, match="fixed enabled"):
        TranslationRuntimePolicy(fast_translation_enabled=False)


def test_channel_owners_use_one_injected_generic_translation_owner() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=object())
    peer_owner_source = inspect.getsource(inspect.getmodule(PeerTranslationChannelOwner))
    self_source = inspect.getsource(inspect.getmodule(SelfTranslationChannelOwner))
    assert harness.self_owner.translation_turns is harness.translation_turns
    assert harness.peer_owner.translation_turns is harness.translation_turns
    assert type(harness.translation_turns.output).__name__ == ("TranslationChannelOwnerCallbacks")
    assert harness.translation_turns.lifecycle_owner_snapshot()["owner"] == (
        "TranslationTurnLifecycleOwner"
    )
    assert "PeerFinalRunsLifecycleOwner" not in peer_owner_source
    assert "TranslationTurnLifecycleOwner(" not in peer_owner_source
    assert peer_owner_source.count('self.translation_turns.cancel_pending(channel="peer")') == 3
    assert self_source.count('self.translation_turns.cancel_pending(channel="self")') == 2
    assert "self.translation_turns.cancel_pending()" not in peer_owner_source

    source_root = Path(__file__).resolve().parents[2] / "src" / "puripuly_heart"
    legacy_owner_references = {
        source_file.relative_to(source_root).as_posix()
        for source_file in source_root.rglob("*.py")
        if "PeerFinalRunsLifecycleOwner" in source_file.read_text(encoding="utf-8")
    }
    assert legacy_owner_references == {"core/orchestrator/peer_final_runs.py"}
    assert not any(
        ".process_translation(" in source_file.read_text(encoding="utf-8")
        for source_file in source_root.rglob("*.py")
    )


@pytest.mark.asyncio
async def test_production_manual_self_and_peer_finals_enter_the_generic_owner() -> None:
    recorded: list[tuple[TranslationTurnRequest, bool]] = []

    class RecordingOwner:
        async def submit(self, request, *, wait_for_parent=False):
            recorded.append((request, wait_for_parent))
            return (request.transcript.utterance_id,)

    harness = compose_translation_test_harness(stt=None, llm=None, osc=object())
    harness.replace_translation_turn_owner_for_test(RecordingOwner())

    await harness.self_owner.submit_text("manual")
    self_id = uuid4()
    await harness.dispatch_stt_event(
        STTFinalEvent(
            self_id,
            Transcript(self_id, "self", is_final=True, channel="self"),
        )
    )
    peer_id = uuid4()
    await harness.dispatch_stt_event(
        STTFinalEvent(
            peer_id,
            Transcript(peer_id, "peer", is_final=True, channel="peer"),
        )
    )

    assert [request.turn_kind for request, _ in recorded] == ["manual", "self", "peer"]
    assert [request.transcript.channel for request, _ in recorded] == [
        "self",
        "self",
        "peer",
    ]
    assert all(wait_for_parent for _, wait_for_parent in recorded)


@pytest.mark.asyncio
async def test_manual_and_self_single_child_preserve_parent_identity() -> None:
    for turn_kind in ("manual", "self"):
        parent_id = uuid4()
        owner = _owner()
        try:
            child_ids = await owner.submit(_request(parent_id=parent_id, turn_kind=turn_kind))
            await owner.wait_for_idle()
        finally:
            await owner.close()
        assert child_ids == (parent_id,)


@pytest.mark.asyncio
async def test_peer_children_are_stable_unique_and_ordered_across_runs_and_targets() -> None:
    parent_id = uuid4()
    runs = (
        FinalLanguageRun("日本語", "ja"),
        FinalLanguageRun("中文", "zh"),
    )
    first_owner = _owner()
    second_owner = _owner()
    try:
        first = await first_owner.submit(
            _request(parent_id=parent_id, turn_kind="peer", runs=runs, targets=("ko", "en"))
        )
        await first_owner.wait_for_idle()
        second = await second_owner.submit(
            _request(parent_id=parent_id, turn_kind="peer", runs=runs, targets=("ko", "en"))
        )
        await second_owner.wait_for_idle()
    finally:
        await first_owner.close()
        await second_owner.close()
    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4
    assert parent_id not in first


@pytest.mark.asyncio
async def test_whole_and_missing_language_facts_reach_the_owner_without_recovery() -> None:
    observed: list[tuple[str | None, str, str]] = []

    async def process(child: TranslationTurnChild, _cancellation_requested):
        observed.append((child.detected_language, child.transcript.text, child.context_policy))
        return "source_only" if child.detected_language is None else "translated"

    owner = _owner(process_child=process)
    try:
        whole_parent = uuid4()
        missing_parent = uuid4()
        await owner.submit(
            _request(
                parent_id=whole_parent,
                turn_kind="peer",
                runs=(FinalLanguageRun("whole", "en"),),
            )
        )
        await owner.submit(
            _request(
                parent_id=missing_parent,
                turn_kind="peer",
                runs=(FinalLanguageRun("missing", ""),),
            )
        )
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert observed == [
        ("en", "whole", "integrated_preferred"),
        (None, "missing", "integrated_preferred"),
    ]
    assert owner.is_parent_closed(whole_parent)
    assert owner.is_parent_closed(missing_parent)


@pytest.mark.asyncio
async def test_parent_closes_only_after_every_child_terminalizes() -> None:
    trace: list[tuple[object, ...]] = []
    owner = _owner(trace=trace)
    parent_id = uuid4()
    try:
        await owner.submit(
            _request(
                parent_id=parent_id,
                turn_kind="peer",
                runs=(FinalLanguageRun("one", "en"), FinalLanguageRun("둘", "ko")),
            )
        )
        await owner.wait_for_idle()
    finally:
        await owner.close()
    terminal_indexes = [index for index, event in enumerate(trace) if event[0] == "terminal"]
    close_index = next(index for index, event in enumerate(trace) if event[0] == "closed")
    assert len(terminal_indexes) == 2
    assert max(terminal_indexes) < close_index


@pytest.mark.asyncio
async def test_unsupported_and_provider_failure_outcomes_are_terminal() -> None:
    outcomes = iter(("source_only", "failed"))

    async def process(_child, _cancellation_requested):
        return next(outcomes)

    trace: list[tuple[object, ...]] = []
    owner = _owner(process_child=process, trace=trace)
    try:
        await owner.submit(
            _request(
                parent_id=uuid4(),
                turn_kind="peer",
                runs=(FinalLanguageRun("?", "unsupported"), FinalLanguageRun("ok", "en")),
            )
        )
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert [event[2] for event in trace if event[0] == "terminal"] == [
        "source_only",
        "failed",
    ]


@pytest.mark.asyncio
async def test_provider_exception_becomes_failed_terminal_outcome() -> None:
    async def process(_child, _cancellation_requested):
        raise RuntimeError("provider failed")

    trace: list[tuple[object, ...]] = []
    owner = _owner(process_child=process, trace=trace)
    parent_id = uuid4()
    try:
        await owner.submit(_request(parent_id=parent_id, turn_kind="self"))
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert ("terminal", 0, "failed") in trace
    assert owner.is_parent_closed(parent_id)


@pytest.mark.asyncio
async def test_close_cancels_active_and_queued_children_and_closes_parent() -> None:
    entered = asyncio.Event()

    async def process(_child, _cancellation_requested):
        entered.set()
        await asyncio.Future()

    trace: list[tuple[object, ...]] = []
    owner = _owner(process_child=process, trace=trace)
    parent_id = uuid4()
    await owner.submit(
        _request(
            parent_id=parent_id,
            turn_kind="peer",
            runs=(FinalLanguageRun("one", "en"), FinalLanguageRun("둘", "ko")),
        )
    )
    await entered.wait()
    await owner.close()
    assert [event[2] for event in trace if event[0] == "terminal"] == [
        "cancelled",
        "cancelled",
    ]
    assert owner.is_parent_closed(parent_id)
    assert owner.has_resources is False


@pytest.mark.asyncio
async def test_peer_cancellation_keeps_queued_self_turn_running() -> None:
    entered = asyncio.Event()
    terminal_events: list[tuple[str, str, str]] = []

    async def process(child, _cancellation_requested):
        if child.channel == "peer":
            entered.set()
            await asyncio.Future()
        return "translated"

    owner = _owner(process_child=process)

    async def terminal(child, outcome) -> None:
        terminal_events.append((child.channel, child.transcript.text, outcome))

    owner.on_child_terminal = terminal
    peer_parent_id = uuid4()
    self_parent_id = uuid4()
    try:
        await owner.submit(
            _request(
                parent_id=peer_parent_id,
                turn_kind="peer",
                runs=(FinalLanguageRun("one", "en"), FinalLanguageRun("둘", "ko")),
            )
        )
        await entered.wait()
        await owner.submit(_request(parent_id=self_parent_id, turn_kind="self"))
        await asyncio.wait_for(owner.cancel_pending(channel="peer"), timeout=1)
        await asyncio.wait_for(owner.wait_for_idle(), timeout=1)
    finally:
        await owner.close()

    assert sorted(terminal_events) == sorted(
        [
            ("peer", "one", "cancelled"),
            ("peer", "둘", "cancelled"),
            ("self", "hello", "translated"),
        ]
    )
    assert owner.is_parent_closed(peer_parent_id)
    assert owner.is_parent_closed(self_parent_id)


@pytest.mark.asyncio
async def test_blocked_peer_parent_does_not_serialize_self_parent() -> None:
    peer_entered = asyncio.Event()

    async def process(child, _cancellation_requested):
        if child.channel == "peer":
            peer_entered.set()
            await asyncio.Future()
        return "translated"

    owner = _owner(process_child=process)
    peer_parent_id = uuid4()
    self_parent_id = uuid4()
    try:
        await owner.submit(_request(parent_id=peer_parent_id, turn_kind="peer"))
        await peer_entered.wait()
        await asyncio.wait_for(
            owner.submit(
                _request(parent_id=self_parent_id, turn_kind="self"),
                wait_for_parent=True,
            ),
            timeout=1,
        )
        assert owner.is_parent_closed(self_parent_id)
        assert not owner.is_parent_closed(peer_parent_id)
        await owner.cancel_pending(channel="peer")
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_parents_run_in_submission_order_within_one_channel() -> None:
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    started_texts: list[str] = []

    async def process(child, _cancellation_requested):
        started_texts.append(child.transcript.text)
        if child.transcript.text == "first":
            first_entered.set()
            await release_first.wait()
        return "translated"

    owner = _owner(process_child=process)
    try:
        await owner.submit(
            _request(
                parent_id=uuid4(),
                turn_kind="peer",
                runs=(FinalLanguageRun("first", "en"),),
            )
        )
        await owner.submit(
            _request(
                parent_id=uuid4(),
                turn_kind="peer",
                runs=(FinalLanguageRun("second", "en"),),
            )
        )
        await first_entered.wait()
        assert started_texts == ["first"]
        release_first.set()
        await owner.wait_for_idle()
        assert started_texts == ["first", "second"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_owner_submits_typed_output_before_terminal_callback() -> None:
    trace: list[tuple[object, ...]] = []
    output = RecordingOutput(trace)

    async def process(child: TranslationTurnChild, _cancellation_requested):
        translation = Translation(
            utterance_id=child.utterance_id,
            text="안녕",
            source_text=child.transcript.text,
            source_language="en",
            target_language=child.target_language,
            channel=child.channel,
        )
        submission = TranslationOutputSubmission(
            parent_utterance_id=child.parent_utterance_id,
            child_utterance_id=child.utterance_id,
            sequence=child.sequence,
            channel=child.channel,
            source=child.source,
            source_text=child.transcript.text,
            source_language="en",
            target_language=child.target_language,
            outcome="translated",
            config_snapshot=child.config_snapshot,
            translation=translation,
            applied_context_mode="local",
        )
        return TranslationTurnProcessResult("translated", submission)

    owner = _owner(process_child=process, output=output, trace=trace)
    parent_id = uuid4()
    try:
        await owner.submit(_request(parent_id=parent_id, turn_kind="manual"))
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert [submission.child_utterance_id for submission in output.submissions] == [parent_id]
    assert [event[0] for event in trace][-3:] == ["output", "terminal", "closed"]


@pytest.mark.asyncio
async def test_output_adapter_failure_terminalizes_child_and_closes_parent() -> None:
    class FailingOutput:
        async def submit_translation_output(self, _submission) -> None:
            raise RuntimeError("output failed")

    async def process(child: TranslationTurnChild, _cancellation_requested):
        translation = Translation(
            utterance_id=child.utterance_id,
            text="안녕",
            source_text=child.transcript.text,
            source_language="en",
            target_language=child.target_language,
            channel=child.channel,
        )
        return TranslationTurnProcessResult(
            "translated",
            TranslationOutputSubmission(
                parent_utterance_id=child.parent_utterance_id,
                child_utterance_id=child.utterance_id,
                sequence=child.sequence,
                channel=child.channel,
                source=child.source,
                source_text=child.transcript.text,
                source_language="en",
                target_language=child.target_language,
                outcome="translated",
                config_snapshot=child.config_snapshot,
                translation=translation,
            ),
        )

    trace: list[tuple[object, ...]] = []
    owner = _owner(process_child=process, output=FailingOutput(), trace=trace)
    parent_id = uuid4()
    try:
        await owner.submit(_request(parent_id=parent_id, turn_kind="self"))
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert ("terminal", 0, "failed") in trace
    assert owner.is_parent_closed(parent_id)


@pytest.mark.asyncio
async def test_terminal_adapter_failure_does_not_strand_parent() -> None:
    parent_id = uuid4()

    async def terminal(_child, _outcome) -> None:
        raise RuntimeError("terminal adapter failed")

    owner = _owner()
    owner.on_child_terminal = terminal
    try:
        await owner.submit(_request(parent_id=parent_id, turn_kind="manual"))
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert owner.is_parent_closed(parent_id)
    assert owner.has_resources is False


@pytest.mark.asyncio
async def test_same_channel_predecessor_wait_is_observed_without_source_text() -> None:
    release_first = asyncio.Event()
    waits: list[tuple[str, dict[str, object]]] = []

    async def process(child, _cancellation_requested):
        if child.parent_utterance_id == first_id:
            await release_first.wait()
        return "translated"

    def observe(event: str, fields) -> None:
        waits.append((event, dict(fields)))

    first_id = uuid4()
    second_id = uuid4()
    owner = _owner(process_child=process, predecessor_wait_observer=observe)
    try:
        await owner.submit(_request(parent_id=first_id, turn_kind="self"))
        await owner.submit(_request(parent_id=second_id, turn_kind="self"))
        await asyncio.sleep(0)
        assert [event for event, _fields in waits] == ["predecessor_wait_start"]
        assert waits[0][1]["parent_utterance_id"] == str(second_id)
        assert waits[0][1]["predecessor_utterance_id"] == str(first_id)
        assert "hello" not in str(waits)
        release_first.set()
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert [event for event, _fields in waits] == [
        "predecessor_wait_start",
        "predecessor_wait_end",
    ]


@pytest.mark.asyncio
async def test_blocked_overlay_does_not_delay_next_same_channel_llm() -> None:
    first_overlay_released = asyncio.Event()
    events: list[str] = []

    async def process(child: TranslationTurnChild, _cancellation_requested):
        events.append(f"llm:{child.transcript.text}")
        return _translated_result(child)

    class BlockingOutput:
        async def submit_translation_output(
            self,
            submission: TranslationOutputSubmission,
        ) -> None:
            events.append(f"overlay-start:{submission.source_text}")
            if submission.source_text == "first":
                await first_overlay_released.wait()
            events.append(f"overlay-end:{submission.source_text}")

    owner = _owner(process_child=process, output=BlockingOutput())
    try:
        await owner.submit(
            _request(
                parent_id=uuid4(),
                turn_kind="self",
                runs=(FinalLanguageRun("first", "en"),),
            )
        )
        await owner.submit(
            _request(
                parent_id=uuid4(),
                turn_kind="self",
                runs=(FinalLanguageRun("second", "en"),),
            )
        )
        for _ in range(50):
            if "llm:second" in events:
                break
            await asyncio.sleep(0)
        assert events == [
            "llm:first",
            "overlay-start:first",
            "llm:second",
        ]
        first_overlay_released.set()
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert events == [
        "llm:first",
        "overlay-start:first",
        "llm:second",
        "overlay-end:first",
        "overlay-start:second",
        "overlay-end:second",
    ]


@pytest.mark.asyncio
async def test_same_channel_output_order_follows_submission_order() -> None:
    started: list[str] = []
    output = RecordingOutput()

    async def process(child: TranslationTurnChild, _cancellation_requested):
        started.append(child.transcript.text)
        return _translated_result(child)

    owner = _owner(process_child=process, output=output)
    try:
        await owner.submit(
            _request(
                parent_id=uuid4(),
                turn_kind="self",
                runs=(FinalLanguageRun("first", "en"),),
            )
        )
        await owner.submit(
            _request(
                parent_id=uuid4(),
                turn_kind="self",
                runs=(FinalLanguageRun("second", "en"),),
            )
        )
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert started == ["first", "second"]
    assert [submission.source_text for submission in output.submissions] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
async def test_unexecuted_child_does_not_hold_semantic_gate_on_overlay() -> None:
    overlay_released = asyncio.Event()
    events: list[str] = []

    async def created(child: TranslationTurnChild) -> None:
        events.append(f"created:{child.transcript.text}")
        if child.transcript.text == "a1":
            raise RuntimeError("create failed")

    async def process(child: TranslationTurnChild, _cancellation_requested):
        events.append(f"llm:{child.transcript.text}")
        return _translated_result(child)

    class BlockingOutput:
        async def submit_translation_output(
            self,
            submission: TranslationOutputSubmission,
        ) -> None:
            events.append(f"overlay-start:{submission.source_text}")
            if submission.source_text == "a2":
                await overlay_released.wait()
            events.append(f"overlay-end:{submission.source_text}")

    owner = _owner(process_child=process, output=BlockingOutput())
    owner.on_child_created = created
    try:
        await owner.submit(
            _request(
                parent_id=uuid4(),
                turn_kind="peer",
                runs=(FinalLanguageRun("a1", "en"), FinalLanguageRun("a2", "ja")),
            )
        )
        await owner.submit(
            _request(
                parent_id=uuid4(),
                turn_kind="peer",
                runs=(FinalLanguageRun("b", "en"),),
            )
        )
        for _ in range(50):
            if "llm:b" in events:
                break
            await asyncio.sleep(0)
        assert "llm:a2" in events
        assert "llm:b" in events
        assert "overlay-start:a2" in events
        assert "overlay-end:a2" not in events
        overlay_released.set()
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert "llm:a1" not in events
    assert events[-2:] == ["overlay-start:b", "overlay-end:b"]
