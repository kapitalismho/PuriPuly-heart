from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.language import map_detected_language_for_llm
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationTurnChild,
    TranslationTurnLifecycleOwner,
    TranslationTurnRequest,
)
from puripuly_heart.core.overlay.presenter import OverlayPresenter
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
    expected_languages: tuple[str, ...]
    intentional_pause_ms: int = 0


@dataclass(frozen=True, slots=True)
class _ControlledTrace:
    language_sequence: tuple[str, ...]
    parsed_final_token_end_ms: tuple[int, ...]
    segment_count: int


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

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Translation:
        _ = (system_prompt, context, max_output_tokens)
        response = {
            "segment-1": "English translation",
            "token-0": "Japanese translation",
            "token-1": "Chinese translation",
            "token-2": "Korean translation",
            "parent-0": "First participant",
            "parent-1": "Second participant",
            "parent-2": "Third participant",
            "parent-3": "Fourth participant",
        }[text]
        return Translation(
            utterance_id=utterance_id,
            text=response,
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            channel="peer",
        )

    async def close(self) -> None:
        return


_CONTROLLED_SCENARIOS = (
    _ControlledScenario("korean-only", ("ko", "ko"), (100, 200), (2,), ("ko",)),
    _ControlledScenario("japanese-only", ("ja", "ja"), (100, 200), (2,), ("ja",)),
    _ControlledScenario("generic-chinese-taiwan-mandarin", ("zh", "zh"), (100, 200), (2,), ("zh",)),
    _ControlledScenario(
        "japanese-to-chinese", ("ja", "zh", "zh"), (100, 200, 300), (3,), ("ja", "zh")
    ),
    _ControlledScenario(
        "chinese-to-japanese-to-korean",
        ("zh", "ja", "ko"),
        (100, 200, 300),
        (3,),
        ("zh", "ja", "ko"),
    ),
    _ControlledScenario(
        "returning-language", ("ja", "zh", "ja"), (100, 200, 300), (3,), ("ja", "zh", "ja")
    ),
    _ControlledScenario("same-language-pause", ("ko", "ko"), (100, 1100), (1, 1), ("ko",), 1000),
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

    event = session._event_projection._legacy_events.get_nowait()
    assert [token.text for token in session._final_tokens] == [token["text"] for token in tokens]
    assert event.text == "".join(token["text"] for token in tokens)
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

    async def on_child_created(_child: TranslationTurnChild) -> None:
        return

    async def on_child_started(
        _child: TranslationTurnChild,
        _task: asyncio.Task[object],
    ) -> None:
        return

    async def process_child(child: TranslationTurnChild, cancellation_requested) -> str:
        assert cancellation_requested() is False
        mapped = map_detected_language_for_llm(child.detected_language or "")
        assert mapped is not None
        if child.detected_language == "zh":
            assert (mapped.code, mapped.name) == ("zh", "Chinese")
        return "translated"

    parent_ids = [uuid4() for _ in parents]
    terminal_traces: dict[UUID, list[str]] = {parent_id: [] for parent_id in parent_ids}

    async def on_child_terminal(child: TranslationTurnChild, outcome: str) -> None:
        child_languages.append(child.detected_language or "")
        child_terminals.append(child.utterance_id)
        terminal_traces[child.parent_utterance_id].append(f"child_terminal:{outcome}")

    async def on_parent_closed(parent_id: UUID) -> None:
        parent_closures.append(parent_id)
        terminal_traces[parent_id].append("parent_closed")

    async def on_parent_rejected(_parent_id: UUID) -> None:
        raise AssertionError("controlled parent must not be rejected")

    owner = TranslationTurnLifecycleOwner(
        on_child_created=on_child_created,
        on_child_started=on_child_started,
        process_child=process_child,
        on_child_terminal=on_child_terminal,
        on_parent_closed=on_parent_closed,
        on_parent_rejected=on_parent_rejected,
    )
    try:
        for parent_id, runs in zip(parent_ids, parents, strict=True):
            await owner.submit(
                TranslationTurnRequest(
                    transcript=Transcript(
                        utterance_id=parent_id,
                        text="".join(run.text for run in runs),
                        is_final=True,
                        channel="peer",
                        final_language_runs=runs,
                        publication_generation=0,
                        source_order=parent_ids.index(parent_id) + 1,
                    ),
                    source="controlled-simulation",
                    turn_kind="peer",
                    target_languages=("en",),
                    config_snapshot=TranslationRuntimeConfigSnapshot(
                        revision=0,
                        value=TranslationRuntimeConfig(),
                    ),
                )
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
async def test_soniox_tokens_preserve_ordered_language_runs_through_turn_owner(
    scenario: _ControlledScenario,
) -> None:
    runs, trace = _controlled_final_runs(
        scenario.languages,
        final_token_end_ms=scenario.final_token_end_ms,
        token_batch_sizes=scenario.token_batch_sizes,
    )
    child_languages, child_terminals, parent_closures, terminal_traces = await _run_lifecycle(
        (runs,)
    )

    assert tuple(child_languages) == scenario.expected_languages
    assert trace.language_sequence == scenario.expected_languages
    assert trace.segment_count == len(scenario.expected_languages)
    assert len(child_terminals) == len(scenario.expected_languages)
    assert len(parent_closures) == 1
    assert tuple(terminal_traces.values()) == (
        (*("child_terminal:translated",) * len(scenario.expected_languages), "parent_closed"),
    )
    assert trace.parsed_final_token_end_ms == scenario.final_token_end_ms
    if scenario.intentional_pause_ms:
        assert (
            trace.parsed_final_token_end_ms[1] - trace.parsed_final_token_end_ms[0]
            == scenario.intentional_pause_ms
        )


@pytest.mark.asyncio
async def test_four_peer_parents_preserve_language_content_and_output_order() -> None:
    languages = ("zh", "ja", "ko", "zh")
    parent_ids = [uuid4() for _ in languages]
    overlay = _RecordingOverlaySink()
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=_DeterministicLLM(),
        osc=osc,
        overlay_sink=overlay,
        peer_translation_enabled=True,
    )
    try:
        for index, (parent_id, language) in enumerate(zip(parent_ids, languages, strict=True)):
            text = f"parent-{index}"
            await harness.dispatch_stt_event(
                STTFinalEvent(
                    parent_id,
                    Transcript(
                        utterance_id=parent_id,
                        text=text,
                        is_final=True,
                        channel="peer",
                        final_language_runs=(FinalLanguageRun(text=text, language=language),),
                    ),
                )
            )
        await harness.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()

        translations = [event for event in overlay.events if event.type == "translation_final"]
        closures = [event for event in overlay.events if event.type == "utterance_closed"]
        assert [event.source_language for event in translations] == list(languages)
        assert [event.source_text for event in translations] == [
            "parent-0",
            "parent-1",
            "parent-2",
            "parent-3",
        ]
        assert [event.text for event in translations] == [
            "First participant",
            "Second participant",
            "Third participant",
            "Fourth participant",
        ]
        assert [event.utterance_id for event in closures] == [
            event.utterance_id for event in translations
        ]
        assert [
            event.type for event in overlay.events if event.type != "peer_transcript_final"
        ] == [kind for _ in languages for kind in ("translation_final", "utterance_closed")]
        assert all(
            harness.translation_turns.is_parent_closed(parent_id) for parent_id in parent_ids
        )
        assert osc.messages == []
        denied = [
            decision
            for decision in harness.output_runtime.routing_decisions
            if decision.route == "self_chatbox" and decision.publication_kind == "peer_subtitle"
        ]
        assert [(decision.decision, decision.reason) for decision in denied] == [
            ("denied", "peer_chatbox_denied")
        ] * 4
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("languages", "expected_translated_indexes"),
    [
        (("unknown", "und"), ()),
        (("unknown", "en", "unsupported"), (1,)),
    ],
)
async def test_production_peer_parent_preserves_per_segment_language_eligibility(
    languages: tuple[str, ...],
    expected_translated_indexes: tuple[int, ...],
) -> None:
    parent_id = uuid4()
    runs = tuple(
        FinalLanguageRun(text=f"segment-{index}", language=language)
        for index, language in enumerate(languages)
    )
    overlay = _RecordingOverlaySink()
    llm = _DeterministicLLM()
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        overlay_sink=overlay,
        osc=osc,
        peer_translation_enabled=True,
    )
    try:
        harness.record_peer_speech_end_for_test(parent_id)
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
        await harness.output_runtime.wait_for_peer_output_idle()

        translations = [
            event for event in overlay.events if getattr(event, "type", None) == "translation_final"
        ]
        peer_finals = [
            event
            for event in overlay.events
            if getattr(event, "type", None) == "peer_transcript_final"
        ]
        closures = [
            event for event in overlay.events if getattr(event, "type", None) == "utterance_closed"
        ]
        assert [event.source_text for event in translations] == [
            runs[index].text for index in expected_translated_indexes
        ]
        assert [event.text for event in peer_finals] == [
            run.text for index, run in enumerate(runs) if index not in expected_translated_indexes
        ]
        assert len(closures) == len(runs)
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_controlled_peer_output_preserves_original_and_denies_chatbox() -> None:
    runs, _ = _controlled_final_runs(("ja", "zh", "ko"))

    for show_peer_original in (True, False):
        parent_id = uuid4()
        presenter = OverlayPresenter(
            calibration=OverlayCalibration(),
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
            harness.record_peer_speech_end_for_test(parent_id)
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
            await harness.output_runtime.wait_for_peer_output_idle()

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
            assert [event.source_language for event in translations] == ["ja", "zh", "ko"]
            assert [event.text for event in translations] == [
                "Japanese translation",
                "Chinese translation",
                "Korean translation",
            ]
            blocks = presenter.snapshot().blocks
            assert [block.primary_text for block in blocks] == [
                "Chinese translation",
                "Korean translation",
            ]
            assert [block.secondary_text for block in blocks] == [run.text for run in runs[-2:]]
            assert [block.secondary_enabled for block in blocks] == [
                show_peer_original,
                show_peer_original,
            ]
            assert all(
                block.channel == "peer" and block.block_variant == "finalized" for block in blocks
            )
            forbidden_raw_speaker_fields = {
                "speaker_id",
                "speaker_label",
                "speaker_session_scope",
            }
            assert all(forbidden_raw_speaker_fields.isdisjoint(block.to_dict()) for block in blocks)
            assert all(
                forbidden_raw_speaker_fields.isdisjoint(event.__dataclass_fields__)
                for event in translations
            )
            assert osc.messages == []
            decision = next(
                decision
                for decision in reversed(harness.output_runtime.routing_decisions)
                if decision.route == "self_chatbox" and decision.publication_kind == "peer_subtitle"
            )
            assert (decision.decision, decision.reason) == ("denied", "peer_chatbox_denied")
            assert all(run.text not in repr(decision) for run in runs)
        finally:
            await harness.stop()
            await presenter.close()
