from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.language import map_detected_language_for_llm
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.orchestrator.peer_final_runs import (
    PeerFinalRunChild,
    PeerFinalRunsLifecycleOwner,
)
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.vad.gating import SpeechEnd
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import FinalLanguageRun, Transcript, Translation
from puripuly_heart.providers.stt.soniox import _SonioxSession
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass(frozen=True, slots=True)
class _ControlledScenario:
    name: str
    languages: tuple[str, ...]
    final_token_end_ms: tuple[int, ...]
    token_batch_sizes: tuple[int, ...]
    expected_segment_count: int
    intentional_pause_ms: int = 0


@dataclass(frozen=True, slots=True)
class _ControlledTrace:
    language_sequence: tuple[str, ...]
    parsed_final_token_end_ms: tuple[int, ...]
    segment_count: int


@dataclass(frozen=True, slots=True)
class _SimulationRecord:
    simulated: bool
    language_sequence: tuple[str, ...]
    segment_count: int
    latency_ms: int
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SimulatedParticipant:
    participant_id: str
    language: str


@dataclass(frozen=True, slots=True)
class _SimulatedLanguageRun:
    participant_id: str
    language: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class _SimulationSchedule:
    name: str
    participants: tuple[_SimulatedParticipant, ...]
    runs: tuple[_SimulatedLanguageRun, ...]


@dataclass(slots=True)
class _RecordingOverlaySink:
    events: list[object] = field(default_factory=list)
    presenter: OverlayPresenter | None = None

    async def emit(self, event: object) -> None:
        self.events.append(event)
        if self.presenter is not None:
            await self.presenter.emit(event)


@dataclass(slots=True)
class _DeterministicLLM(LLMProvider):
    requested_source_languages: list[str] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        _ = (system_prompt, context)
        self.requested_source_languages.append(source_language)
        return Translation(
            utterance_id=utterance_id,
            text=f"translated-{len(self.requested_source_languages)}",
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            channel="peer",
        )

    async def close(self) -> None:
        return


_CONTROLLED_SCENARIOS = (
    _ControlledScenario("korean-only", ("ko", "ko"), (100, 200), (2,), 1),
    _ControlledScenario("japanese-only", ("ja", "ja"), (100, 200), (2,), 1),
    _ControlledScenario("generic-chinese-taiwan-mandarin", ("zh", "zh"), (100, 200), (2,), 1),
    _ControlledScenario("japanese-to-chinese", ("ja", "zh", "zh"), (100, 200, 300), (3,), 2),
    _ControlledScenario(
        "chinese-to-japanese-to-korean", ("zh", "ja", "ko"), (100, 200, 300), (3,), 3
    ),
    _ControlledScenario("same-language-pause", ("ko", "ko"), (100, 1100), (1, 1), 1, 1000),
)


def _controlled_session() -> _SonioxSession:
    return _SonioxSession(
        api_key="",
        model="controlled",
        endpoint="",
        sample_rate_hz=16000,
        language_hints=[],
        context_terms=[],
        keepalive_interval_s=1.0,
        trailing_silence_ms=0,
        connect_timeout_s=1.0,
        enable_language_identification=True,
    )


def _controlled_final_runs(
    languages: tuple[str, ...],
    *,
    final_token_end_ms: tuple[int, ...] | None = None,
    token_batch_sizes: tuple[int, ...] | None = None,
) -> tuple[tuple[FinalLanguageRun, ...], _ControlledTrace]:
    session = _controlled_session()
    final_token_end_ms = final_token_end_ms or tuple(
        100 * (index + 1) for index in range(len(languages))
    )
    token_batch_sizes = token_batch_sizes or (len(languages),)
    assert len(final_token_end_ms) == len(languages)
    assert sum(token_batch_sizes) == len(languages)
    assert list(final_token_end_ms) == sorted(final_token_end_ms)
    tokens = [
        {
            "text": f"token-{index}",
            "language": language,
            "is_final": True,
            "end_ms": end_ms,
        }
        for index, (language, end_ms) in enumerate(zip(languages, final_token_end_ms, strict=True))
    ]
    session._pending_finalize_requests = 1
    batch_start = 0
    for batch_index, batch_size in enumerate(token_batch_sizes):
        batch_end = batch_start + batch_size
        batch = tokens[batch_start:batch_end]
        if batch_index == len(token_batch_sizes) - 1:
            batch = [*batch, {"text": "<fin>", "is_final": True}]
        session._handle_message(json.dumps({"tokens": batch}))
        batch_start = batch_end

    event = session._events.get_nowait()
    assert [token.text for token in session._final_tokens] == [token["text"] for token in tokens]
    assert event.text == "".join(token["text"] for token in tokens)
    assert [run.language for run in event.final_language_runs] == list(dict.fromkeys(languages))
    assert "zh-CN" not in [run.language for run in event.final_language_runs]
    assert "zh-TW" not in [run.language for run in event.final_language_runs]
    parsed_final_token_end_ms = tuple(token.end_ms for token in session._final_tokens)
    assert all(end_ms is not None for end_ms in parsed_final_token_end_ms)
    return event.final_language_runs, _ControlledTrace(
        language_sequence=tuple(run.language for run in event.final_language_runs),
        parsed_final_token_end_ms=tuple(int(end_ms) for end_ms in parsed_final_token_end_ms),
        segment_count=len(event.final_language_runs),
    )


async def _run_lifecycle(
    parents: tuple[tuple[FinalLanguageRun, ...], ...],
) -> tuple[list[str], list[UUID], list[UUID], dict[UUID, tuple[str, ...]]]:
    child_languages: list[str] = []
    child_terminals: list[UUID] = []
    parent_closures: list[UUID] = []

    async def on_child_created(_child: PeerFinalRunChild) -> None:
        return

    async def on_child_started(
        _child: PeerFinalRunChild,
        _task: asyncio.Task[str],
    ) -> None:
        return

    async def process_child(child: PeerFinalRunChild, cancellation_requested) -> str:
        assert cancellation_requested() is False
        mapped = map_detected_language_for_llm(child.detected_language or "")
        assert mapped is not None
        if child.detected_language == "zh":
            assert (mapped.code, mapped.name) == ("zh", "Chinese")
        return "translated"

    parent_ids = [uuid4() for _ in parents]
    terminal_traces: dict[UUID, list[str]] = {parent_id: [] for parent_id in parent_ids}

    async def on_child_terminal(child: PeerFinalRunChild, outcome: str) -> None:
        child_languages.append(child.detected_language or "")
        child_terminals.append(child.utterance_id)
        terminal_traces[child.parent_utterance_id].append(f"child_terminal:{outcome}")

    async def on_parent_closed(parent_id: UUID) -> None:
        parent_closures.append(parent_id)
        terminal_traces[parent_id].append("parent_closed")

    async def on_parent_rejected(_parent_id: UUID) -> None:
        raise AssertionError("controlled parent must not be rejected")

    owner = PeerFinalRunsLifecycleOwner(
        on_child_created=on_child_created,
        on_child_started=on_child_started,
        process_child=process_child,
        on_child_terminal=on_child_terminal,
        on_parent_closed=on_parent_closed,
        on_parent_rejected=on_parent_rejected,
    )
    try:
        for parent_id, runs in zip(parent_ids, parents, strict=True):
            await owner.submit_parent(
                Transcript(
                    utterance_id=parent_id,
                    text="".join(run.text for run in runs),
                    is_final=True,
                    channel="peer",
                    final_language_runs=runs,
                ),
                source="controlled-simulation",
            )
        await owner.wait_for_idle()
        assert parent_closures == parent_ids
        assert len(child_terminals) == sum(len(runs) for runs in parents)
        for parent_id, runs in zip(parent_ids, parents, strict=True):
            trace = terminal_traces[parent_id]
            assert trace == [*("child_terminal:translated",) * len(runs), "parent_closed"]
            assert trace.index("parent_closed") == len(trace) - 1
        return (
            child_languages,
            child_terminals,
            parent_closures,
            {parent_id: tuple(trace) for parent_id, trace in terminal_traces.items()},
        )
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", _CONTROLLED_SCENARIOS, ids=lambda item: item.name)
async def test_controlled_soniox_readiness_scenarios_run_twice(
    scenario: _ControlledScenario,
) -> None:
    expected_languages = list(dict.fromkeys(scenario.languages))

    for _attempt in range(2):
        runs, controlled_trace = _controlled_final_runs(
            scenario.languages,
            final_token_end_ms=scenario.final_token_end_ms,
            token_batch_sizes=scenario.token_batch_sizes,
        )
        child_languages, child_terminals, parent_closures, terminal_traces = await _run_lifecycle(
            (runs,)
        )

        assert child_languages == expected_languages
        assert controlled_trace.language_sequence == tuple(expected_languages)
        assert controlled_trace.segment_count == scenario.expected_segment_count
        assert len(child_terminals) == scenario.expected_segment_count
        assert len(parent_closures) == 1
        assert tuple(terminal_traces.values()) == (
            (*("child_terminal:translated",) * scenario.expected_segment_count, "parent_closed"),
        )
        if scenario.intentional_pause_ms:
            assert (
                controlled_trace.parsed_final_token_end_ms[1]
                - controlled_trace.parsed_final_token_end_ms[0]
                == scenario.intentional_pause_ms
            )


def _schedule_latency_ms(schedule: _SimulationSchedule) -> int:
    return max(run.end_ms for run in schedule.runs) - min(run.start_ms for run in schedule.runs)


def _overlapping_run_pairs(schedule: _SimulationSchedule) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for index, first in enumerate(schedule.runs):
        for second in schedule.runs[index + 1 :]:
            if first.start_ms < second.end_ms and second.start_ms < first.end_ms:
                pairs.append((first.participant_id, second.participant_id))
    return tuple(pairs)


async def _run_simulated_schedule(schedule: _SimulationSchedule) -> _SimulationRecord:
    participants_by_id = {
        participant.participant_id: participant for participant in schedule.participants
    }
    assert len(participants_by_id) == 4
    assert all(
        participants_by_id[run.participant_id].language == run.language for run in schedule.runs
    )
    assert all(run.start_ms < run.end_ms for run in schedule.runs)

    overlay = _RecordingOverlaySink()
    osc = RecordingOscQueue()
    llm = _DeterministicLLM()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=osc,
        overlay_sink=overlay,
        peer_translation_enabled=True,
    )
    parent_ids = [uuid4() for _ in schedule.runs]
    expected_languages = [run.language for run in schedule.runs]

    try:
        for index, (parent_id, run) in enumerate(zip(parent_ids, schedule.runs, strict=True)):
            modeled_run = FinalLanguageRun(text=f"simulated-run-{index}", language=run.language)
            await harness.peer_owner.handle_peer_vad_event(SpeechEnd(parent_id))
            await harness.dispatch_stt_event(
                STTFinalEvent(
                    utterance_id=parent_id,
                    transcript=Transcript(
                        utterance_id=parent_id,
                        text=modeled_run.text,
                        is_final=True,
                        channel="peer",
                        final_language_runs=(modeled_run,),
                    ),
                )
            )
        await harness.translation_turns.wait_for_idle()

        terminal_events = [
            event
            for event in overlay.events
            if getattr(event, "type", None) in {"translation_final", "utterance_closed"}
        ]
        translations = [
            event
            for event in terminal_events
            if getattr(event, "type", None) == "translation_final"
        ]
        closures = [
            event for event in terminal_events if getattr(event, "type", None) == "utterance_closed"
        ]
        expected_terminal_types = [
            event_type
            for _ in schedule.runs
            for event_type in ("translation_final", "utterance_closed")
        ]
        failures: list[str] = []
        if llm.requested_source_languages != expected_languages:
            failures.append("language_order")
        if len(translations) != len(schedule.runs):
            failures.append("translation_count")
        if [event.utterance_id for event in closures] != [
            event.utterance_id for event in translations
        ]:
            failures.append("terminal_order")
        if [event.type for event in terminal_events] != expected_terminal_types:
            failures.append("terminal_sequence")
        if not all(
            harness.translation_turns.is_parent_closed(parent_id) for parent_id in parent_ids
        ):
            failures.append("parent_closure")
        decisions = [
            decision
            for decision in harness.output_runtime.routing_decisions
            if decision.route == "self_chatbox" and decision.publication_kind == "peer_subtitle"
        ]
        if len(decisions) != len(schedule.runs) or any(
            (decision.decision, decision.reason) != ("denied", "peer_chatbox_denied")
            for decision in decisions
        ):
            failures.append("peer_chatbox_routing")
        if osc.messages:
            failures.append("peer_chatbox_publication")

        record = _SimulationRecord(
            simulated=True,
            language_sequence=tuple(llm.requested_source_languages),
            segment_count=len(translations),
            latency_ms=_schedule_latency_ms(schedule),
            failures=tuple(failures),
        )
        assert record.language_sequence == tuple(expected_languages)
        assert record.segment_count == len(schedule.runs)
        assert record.latency_ms == _schedule_latency_ms(schedule)
        assert not record.failures
        return record
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_deterministic_four_participant_normal_and_limited_overlap_simulation() -> None:
    participants = (
        _SimulatedParticipant("participant-zh-1", "zh"),
        _SimulatedParticipant("participant-ja-1", "ja"),
        _SimulatedParticipant("participant-ko-1", "ko"),
        _SimulatedParticipant("participant-zh-2", "zh"),
    )
    normal_schedule = _SimulationSchedule(
        "normal-turns",
        participants,
        (
            _SimulatedLanguageRun("participant-zh-1", "zh", 0, 100),
            _SimulatedLanguageRun("participant-ja-1", "ja", 200, 320),
            _SimulatedLanguageRun("participant-ko-1", "ko", 400, 540),
            _SimulatedLanguageRun("participant-zh-2", "zh", 600, 760),
        ),
    )
    overlap_schedule = _SimulationSchedule(
        "limited-overlap",
        participants,
        (
            _SimulatedLanguageRun("participant-zh-1", "zh", 0, 300),
            _SimulatedLanguageRun("participant-ja-1", "ja", 200, 430),
            _SimulatedLanguageRun("participant-ko-1", "ko", 500, 620),
            _SimulatedLanguageRun("participant-zh-2", "zh", 700, 850),
        ),
    )

    assert _overlapping_run_pairs(normal_schedule) == ()
    assert _overlapping_run_pairs(overlap_schedule) == (("participant-zh-1", "participant-ja-1"),)
    normal_record = await _run_simulated_schedule(normal_schedule)
    overlap_record = await _run_simulated_schedule(overlap_schedule)

    assert normal_record.simulated is True
    assert overlap_record.simulated is True
    assert normal_record.language_sequence == ("zh", "ja", "ko", "zh")
    assert overlap_record.language_sequence == ("zh", "ja", "ko", "zh")
    assert normal_record.segment_count == len(normal_schedule.runs)
    assert overlap_record.segment_count == len(overlap_schedule.runs)
    assert normal_record.latency_ms == _schedule_latency_ms(normal_schedule)
    assert overlap_record.latency_ms == _schedule_latency_ms(overlap_schedule)


@pytest.mark.asyncio
async def test_controlled_peer_output_preserves_original_and_denies_chatbox() -> None:
    runs, _ = _controlled_final_runs(("ja", "zh", "ko"))

    for show_peer_original in (True, False):
        parent_id = uuid4()
        presenter = OverlayPresenter(
            calibration=OverlayCalibration(),
            peer_presentation_refresh_burst=False,
            show_peer_original=show_peer_original,
        )
        overlay = _RecordingOverlaySink(presenter=presenter)
        osc = RecordingOscQueue()
        llm = _DeterministicLLM()
        harness = compose_translation_test_harness(
            stt=None,
            llm=llm,
            osc=osc,
            overlay_sink=overlay,
            peer_translation_enabled=True,
        )

        try:
            await harness.peer_owner.handle_peer_vad_event(SpeechEnd(parent_id))
            await harness.dispatch_stt_event(
                STTFinalEvent(
                    utterance_id=parent_id,
                    transcript=Transcript(
                        utterance_id=parent_id,
                        text="".join(run.text for run in runs),
                        is_final=True,
                        channel="peer",
                        final_language_runs=runs,
                    ),
                )
            )
            await harness.translation_turns.wait_for_idle()

            terminal_events = [
                event
                for event in overlay.events
                if getattr(event, "type", None) in {"translation_final", "utterance_closed"}
            ]
            translations = [
                event
                for event in terminal_events
                if getattr(event, "type", None) == "translation_final"
            ]
            closures = [
                event
                for event in terminal_events
                if getattr(event, "type", None) == "utterance_closed"
            ]

            assert [event.source_text for event in translations] == [run.text for run in runs]
            assert [event.utterance_id for event in closures] == [
                event.utterance_id for event in translations
            ]
            assert [event.type for event in terminal_events] == [
                "translation_final",
                "utterance_closed",
                "translation_final",
                "utterance_closed",
                "translation_final",
                "utterance_closed",
            ]
            assert llm.requested_source_languages == ["ja", "zh", "ko"]
            assert "zh-CN" not in llm.requested_source_languages
            assert "zh-TW" not in llm.requested_source_languages
            blocks = presenter.snapshot().blocks
            assert [block.primary_text for block in blocks] == [
                event.text for event in translations[-2:]
            ]
            assert [block.secondary_text for block in blocks] == [run.text for run in runs[-2:]]
            assert [block.secondary_enabled for block in blocks] == [
                show_peer_original,
                show_peer_original,
            ]
            assert all(
                block.channel == "peer" and block.block_variant == "finalized" for block in blocks
            )
            assert all(
                "label" not in field_name and "speaker" not in field_name
                for block in blocks
                for field_name in block.to_dict()
            )
            assert all(
                "label" not in field_name and "speaker" not in field_name
                for event in translations
                for field_name in event.__dataclass_fields__
            )
            assert osc.messages == []
            decision = harness.output_runtime.routing_decisions[-1]
            assert (decision.decision, decision.reason) == ("denied", "peer_chatbox_denied")
            assert all(run.text not in repr(decision) for run in runs)
        finally:
            await harness.stop()
