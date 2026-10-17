from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections import Counter
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

ISOLATED_ROOT = os.environ.get("SELF_GRACE_233_ISOLATED_ROOT")
if ISOLATED_ROOT is None or Path(__file__).resolve().parents[3] != Path(ISOLATED_ROOT).resolve():
    raise RuntimeError("Run this historical fixture through tests/fixtures/self_grace_233/run.py")

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.orchestrator.channel_runtime import _SpeculativeAttemptStatus
from puripuly_heart.core.stt.backend import (
    STTNativeProvenance,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine, STTRecognitionWatchdogs
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
from tests.core.runtime.test_self_capture_session import RecordingProvider, build_owner, config
from tests.core.test_stt_scoped_engine import ControlledScopedSession
from tests.helpers import translation_owners

BASELINE = "75c4485a6787d504f5335d1602973f63b4d0c234"
POLICIES = ("unchanged", "elapsed", "no_op", "combined")


@dataclass(frozen=True)
class Step:
    at: float
    event: str
    acoustic: str = "A"
    text: str = ""
    outcome: str = "final"
    source_at: float | None = None


@dataclass(frozen=True)
class Case:
    name: str
    steps: tuple[Step, ...]
    llm_delay: float = 0.05
    overlap: bool = True
    two_targets: bool = False
    fail_first: bool = False
    retire_after_cancel: bool = False
    awaiting_timeout: float = 3.0
    until: float = 2.5
    route: str = "modeled_local_final_only"
    expected_defect: bool = False


def speech(acoustic: str, start: float, end: float, *, confirmed: bool = True) -> list[Step]:
    steps = [Step(start, "start", acoustic)]
    if confirmed:
        steps.extend(Step(start + offset, "chunk", acoustic) for offset in (0.01, 0.02, 0.03))
    steps.append(Step(end, "end", acoustic))
    return steps


CASES = (
    Case("final_after_end", tuple(speech("A", 0.0, 0.1) + [Step(0.5, "terminal", text="hello")])),
    Case("final_short_asr", tuple(speech("A", 0.0, 0.1) + [Step(0.2, "terminal", text="hello")])),
    Case(
        "provider_terminal_pre_end",
        tuple(speech("A", 0.0, 0.3) + [Step(0.1, "terminal", text="hello")]),
    ),
    Case(
        "delayed_VAD_receipt",
        tuple(
            speech("A", 0.0, 0.3)[:-1]
            + [Step(0.3, "end", source_at=0.1), Step(0.5, "terminal", text="hello")]
        ),
    ),
    Case(
        "future_VAD_anchor_fallback",
        tuple(
            speech("A", 0.0, 0.1)[:-1]
            + [Step(0.1, "end", source_at=0.5), Step(0.2, "terminal", text="hello")]
        ),
    ),
    Case(
        "slow_translation",
        tuple(speech("A", 0.0, 0.1) + [Step(0.5, "terminal", text="hello")]),
        llm_delay=0.8,
    ),
    Case(
        "stable_before_end_terminal_early",
        tuple(
            speech("A", 0.0, 0.3)
            + [Step(0.1, "stable", text="hello"), Step(0.4, "terminal", text="hello")]
        ),
        route="modeled_stable",
    ),
    Case(
        "stable_after_end_terminal_late",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.3, "stable", text="hello"), Step(1.0, "terminal", text="hello")]
        ),
        route="modeled_stable",
    ),
    Case(
        "stable_terminal_before_timer",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.3, "stable", text="hello"), Step(0.4, "terminal", text="hello")]
        ),
        route="modeled_stable",
    ),
    Case(
        "duplicate_native_and_same_snapshot",
        tuple(
            speech("A", 0.0, 0.1)
            + [
                Step(0.2, "stable", text="hello"),
                Step(0.3, "duplicate", text="hello"),
                Step(0.35, "same_snapshot", text="hello"),
                Step(0.4, "terminal", text="hello"),
            ]
        ),
        route="modeled_stable",
    ),
    Case(
        "boundary_period",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.2, "stable", text="hello"), Step(0.4, "terminal", text="hello.")]
        ),
        route="modeled_stable",
    ),
    Case(
        "question_is_revision",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.2, "stable", text="hello"), Step(0.4, "terminal", text="hello?")]
        ),
        route="modeled_stable",
    ),
    Case(
        "number_is_revision",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.2, "stable", text="pay"), Step(0.4, "terminal", text="pay 20")]
        ),
        route="modeled_stable",
    ),
    Case(
        "negation_is_revision",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.2, "stable", text="do"), Step(0.4, "terminal", text="do not")]
        ),
        route="modeled_stable",
    ),
    Case(
        "stable_contradiction",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.2, "stable", text="pay 10"), Step(0.4, "terminal", text="pay 20")]
        ),
        route="modeled_stable",
    ),
    Case(
        "repeated_identical_speech",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.2, "terminal", text="hello")]
            + speech("B", 0.3, 0.4)
            + [Step(0.5, "terminal", "B", "hello")]
        ),
    ),
    Case(
        "delayed_A_during_B_pending",
        tuple(
            speech("P", 0.0, 0.1)
            + [Step(0.15, "terminal", "P", "prior")]
            + speech("A", 0.2, 0.3)
            + speech("B", 0.4, 1.3, confirmed=False)
            + [Step(0.5, "terminal", text="hello"), Step(1.4, "terminal", "B", "world")]
        ),
        expected_defect=True,
    ),
    Case(
        "delayed_A_during_B_confirmed",
        tuple(
            speech("P", 0.0, 0.1)
            + [Step(0.15, "terminal", "P", "prior")]
            + speech("A", 0.2, 0.3)
            + speech("B", 0.4, 1.3)
            + [Step(0.5, "terminal", text="hello"), Step(1.4, "terminal", "B", "world")]
        ),
        expected_defect=True,
    ),
    Case(
        "B_before_buffer_final_only",
        tuple(
            speech("A", 0.0, 0.1)
            + speech("B", 0.2, 1.1)
            + [Step(0.3, "terminal", text="hello"), Step(1.2, "terminal", "B", "world")]
        ),
        expected_defect=True,
    ),
    Case(
        "serial_provider_B_queued",
        tuple(
            speech("A", 0.0, 0.1)
            + speech("B", 0.2, 1.1)
            + [Step(0.3, "terminal", text="hello"), Step(1.2, "terminal", "B", "world")]
        ),
        overlap=False,
    ),
    *(
        Case(
            f"matching_B_{outcome}",
            tuple(
                speech("A", 0.0, 0.1)
                + [Step(0.15, "terminal", text="hello")]
                + speech("B", 0.2, 0.3)
                + [Step(0.4, "terminal", "B", outcome=outcome)]
            ),
        )
        for outcome in ("empty", "failed", "cancelled", "expired")
    ),
    Case(
        "matching_B_false_start",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.15, "terminal", text="hello")]
            + speech("B", 0.2, 0.3, confirmed=False)
            + [Step(0.4, "terminal", "B", outcome="empty")]
        ),
    ),
    Case(
        "matching_B_resume_timeout",
        tuple(
            speech("A", 0.0, 0.1) + [Step(0.15, "terminal", text="hello")] + speech("B", 0.2, 0.3)
        ),
        awaiting_timeout=0.3,
        until=1.2,
    ),
    Case(
        "watchdog_prefix_suffix_terminal",
        (
            Step(0.0, "start"),
            Step(0.1, "stable", text="hello"),
            Step(0.8, "stable_suffix", text=" world"),
            Step(1.0, "end"),
            Step(1.1, "terminal", text="hello world"),
        ),
        awaiting_timeout=0.2,
        route="modeled_stable",
    ),
    Case(
        "two_targets",
        tuple(speech("A", 0.0, 0.1) + [Step(0.5, "terminal", text="hello")]),
        two_targets=True,
    ),
    Case(
        "speculation_failure",
        tuple(speech("A", 0.0, 0.1) + [Step(0.5, "terminal", text="hello")]),
        fail_first=True,
    ),
    Case(
        "provider_replacement",
        tuple(
            speech("A", 0.0, 0.1) + [Step(0.2, "terminal", text="hello"), Step(0.3, "replace_llm")]
        ),
        llm_delay=0.3,
    ),
    Case(
        "translation_configuration_change",
        tuple(speech("A", 0.0, 0.1) + [Step(0.2, "terminal", text="hello"), Step(0.3, "language")]),
        llm_delay=0.3,
    ),
    Case(
        "recognition_configuration_barrier",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.2, "terminal", text="hello"), Step(0.3, "recognition_config")]
            + speech("B", 0.4, 0.5)
            + [Step(0.6, "terminal", "B", "world")]
        ),
    ),
    Case(
        "OFF_manual_survives",
        tuple(
            speech("A", 0.0, 0.1)
            + [
                Step(0.2, "terminal", text="hello"),
                Step(0.25, "manual", text="manual"),
                Step(0.3, "OFF"),
            ]
        ),
        llm_delay=0.3,
    ),
    Case(
        "shutdown_running",
        tuple(speech("A", 0.0, 0.1) + [Step(0.2, "terminal", text="hello"), Step(0.3, "shutdown")]),
        llm_delay=0.3,
    ),
    Case(
        "active_SELF_intervening_Peer",
        tuple(
            speech("A", 0.0, 0.1)
            + [Step(0.2, "terminal", text="hello"), Step(0.3, "peer", text="peer")]
        ),
        llm_delay=0.6,
    ),
    Case(
        "manual_and_recognition_unit",
        (
            Step(0.0, "manual", text="manual"),
            Step(0.1, "recognition_unit", text="independent unit"),
        ),
    ),
    *(
        Case(
            f"G2_fixed_three_{timing}",
            tuple(
                speech("A", 0.0, 0.1)
                + [Step(0.2, "terminal", text="one")]
                + speech("B", 0.8, 0.9)
                + [Step(1.0, "terminal", "B", "two")]
                + speech("C", 1.6, 1.7)
                + [Step(1.8, "terminal", "C", "three")]
            ),
            llm_delay=delay,
            retire_after_cancel=True,
            until=4.0,
        )
        for timing, delay in (("fast", 0.05), ("slow", 1.0))
    ),
    *(
        Case(
            f"G1_continuation_{onset:.2f}_{timing}",
            tuple(
                speech("A", 0.0, 0.1)
                + [Step(0.2, "terminal", text="one")]
                + speech("B", onset, onset + 0.1)
                + [Step(onset + 0.2, "terminal", "B", "two")]
            ),
            llm_delay=delay,
            until=3.0,
        )
        for onset in (0.49, 0.50, 0.55, 0.60, 0.61)
        for timing, delay in (("fast", 0.05), ("slow", 0.8))
    ),
    *(
        Case(
            f"B_unresolved_C_{outcome}",
            tuple(
                speech("P", 0.0, 0.1)
                + [Step(0.15, "terminal", "P", "prior")]
                + speech("A", 0.2, 0.3)
                + speech("B", 0.4, 0.5)
                + speech("C", 0.6, 0.7, confirmed=outcome == "final")
                + [
                    Step(0.8, "terminal", "C", "later" if outcome == "final" else "", outcome),
                    Step(0.9, "terminal", text="hello"),
                    Step(1.8, "terminal", "B", "world"),
                ]
            ),
            until=3.0,
            expected_defect=True,
        )
        for outcome in ("empty", "final")
    ),
    Case(
        "prebuffer_B_ended_pending",
        tuple(
            speech("A", 0.0, 0.1)
            + speech("B", 0.2, 0.3)
            + [Step(0.4, "terminal", text="hello"), Step(1.0, "terminal", "B", "world")]
        ),
        expected_defect=True,
    ),
    Case(
        "B_watchdog_survives_C_false_start",
        tuple(
            speech("P", 0.0, 0.1)
            + [Step(0.15, "terminal", "P", "prior")]
            + speech("A", 0.2, 0.3)
            + speech("B", 0.4, 0.5)
            + speech("C", 0.6, 0.7, confirmed=False)
            + [Step(0.8, "terminal", "C", outcome="empty"), Step(0.9, "terminal", text="hello")]
        ),
        awaiting_timeout=0.5,
        expected_defect=True,
    ),
    Case(
        "same_acoustic_suffix_during_B_pending",
        (
            Step(0.0, "start"),
            Step(0.1, "stable", text="hello"),
            Step(0.75, "end"),
            Step(0.8, "start", "B"),
            Step(0.9, "stable_suffix", text=" world"),
            Step(1.4, "terminal", text="hello world"),
            Step(1.5, "end", "B"),
            Step(1.6, "terminal", "B", outcome="empty"),
        ),
        awaiting_timeout=0.2,
        route="modeled_overlap_stable_not_real_provider_proof",
        expected_defect=True,
    ),
)


@dataclass
class ControlledClock:
    value: float = 0.0
    sleepers: list[tuple[float, asyncio.Future]] = field(default_factory=list)
    observer: object | None = None
    source_observed_at: float | None = None

    def now(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        if delay <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        self.sleepers.append((self.value + delay, future))
        await future

    async def settle(self) -> None:
        for _ in range(80):
            await asyncio.sleep(0)

    async def advance_to(self, target: float) -> None:
        assert target >= self.value
        while True:
            live = [(deadline, future) for deadline, future in self.sleepers if not future.done()]
            due = [deadline for deadline, _ in live if deadline <= target + 1e-9]
            next_at = min(due) if due else target
            if self.observer is not None:
                self.observer(next_at - self.value)
            self.value = next_at
            for deadline, future in live:
                if deadline <= next_at + 1e-9 and not future.done():
                    future.set_result(None)
            self.sleepers = [(deadline, future) for deadline, future in live if not future.done()]
            await self.settle()
            if not due:
                return


class AsyncioClockFacade:
    def __init__(self, clock: ControlledClock, *, source_loop: bool = False):
        self.clock = clock
        self.source_loop = source_loop

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def sleep(self, delay, result=None):
        await self.clock.sleep(delay)
        return result

    def get_running_loop(self):
        loop = asyncio.get_running_loop()
        if not self.source_loop:
            return loop
        clock = self.clock

        class LoopClockFacade:
            def time(self):
                return clock.now() if clock.source_observed_at is None else clock.source_observed_at

            def __getattr__(self, name):
                return getattr(loop, name)

        return LoopClockFacade()


class ScheduledLLM:
    def __init__(self, clock, trace, *, delay, fail_first=False, retiring=False):
        self.clock = clock
        self.trace = trace
        self.delay = delay
        self.fail_first = fail_first
        self.retiring = retiring
        self.calls = []
        self.active = 0
        self.retiring_active = 0

    async def translate(self, *, utterance_id, text, target_language, context="", **kwargs):
        call = {
            "text": text,
            "target": target_language,
            "invoked_at_s": self.clock.now(),
            "context": context,
        }
        self.calls.append(call)
        self.active += 1
        self.trace("llm_invoked", **call, physical_active=self.active)
        completion = asyncio.create_task(self.clock.sleep(self.delay))
        cancelled = False
        try:
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError:
                cancelled = True
                self.trace("llm_logical_cancel", text=text, physical_done=not self.retiring)
                if not self.retiring:
                    completion.cancel()
                    raise
                self.retiring_active += 1
                await asyncio.shield(completion)
            call["ready_at_s"] = self.clock.now()
            if self.fail_first and len(self.calls) == 1:
                self.trace("llm_failed", text=text)
                raise RuntimeError("controlled speculative failure")
            self.trace(
                "llm_ready", text=text, target=target_language, after_logical_cancel=cancelled
            )
            return Translation(
                utterance_id=utterance_id, text=f"translated:{target_language}:{text}"
            )
        finally:
            self.active -= 1
            if cancelled and self.retiring:
                self.retiring_active -= 1
            self.trace("llm_physical_finished", text=text, physical_active=self.active)

    async def close(self):
        self.trace("llm_closed", physical_active=self.active)


class GraceProbeMixin:
    _timeout_source = ContextVar("grace_233_timeout_source", default=None)

    def configure_experiment(self, policy, trace):
        self.policy = policy
        self.trace = trace
        self.acoustic_names = {}
        self.publication_acoustics = {}
        self.observed_ends = {}
        self.actual_terminals = {}
        self.active_acoustics = set()
        self.unresolved_acoustics = set()
        self.observed_chunks = Counter()
        self.suppress_rearm = False
        self.grace_only_wait_s = 0.0
        self.commits = []
        self.premature_commits = []
        self.metrics = Counter()
        self.deadlines = {}

    async def handle_vad_event(self, event):
        vad = event.event if hasattr(event, "segment") else event
        acoustic = vad.utterance_id
        name = self.acoustic_names.get(acoustic, "unknown")
        if isinstance(vad, SpeechStart):
            self.active_acoustics.add(acoustic)
            self.unresolved_acoustics.add(acoustic)
            self.observed_chunks[acoustic] = 0
        elif isinstance(vad, SpeechChunk):
            self.observed_chunks[acoustic] += 1
        elif isinstance(vad, SpeechEnd):
            self.active_acoustics.discard(acoustic)
            self.observed_ends[acoustic] = event.segment.sealed_at_monotonic_s
            if self.observed_chunks[acoustic] < 3:
                self.unresolved_acoustics.discard(acoustic)
        self.trace(
            "vad_owner_receipt",
            acoustic=name,
            kind=type(vad).__name__,
            sealed_at_s=getattr(getattr(event, "segment", None), "sealed_at_monotonic_s", None),
            source_ranges=[
                asdict(item)
                for item in getattr(getattr(event, "segment", None), "content_ranges", ())
            ],
        )
        await super().handle_vad_event(event)

    async def _handle_scoped_stt_event(self, event, *, low_latency_mode):
        acoustic = event.identity.segment.segment_id
        if isinstance(event, STTProviderTurnTerminal):
            self.actual_terminals[acoustic] = event.outcome
            self.unresolved_acoustics.discard(acoustic)
        self.trace(
            "recognition_owner_receipt",
            acoustic=self.acoustic_names.get(acoustic),
            kind=type(event).__name__,
            outcome=getattr(event, "outcome", None),
            contribution=(
                asdict(event.contribution)
                if isinstance(event, STTProviderTurnUpdate) and event.contribution
                else None
            ),
            included_contributions=[
                asdict(item) for item in getattr(event, "included_contributions", ())
            ],
            source_identity={
                "activation": event.identity.segment.activation_generation,
                "order": event.identity.segment.segment_order,
                "capture_epoch": event.identity.segment.capture_epoch,
            },
            stable_is_terminal=False,
        )
        await super()._handle_scoped_stt_event(event, low_latency_mode=low_latency_mode)

    async def _handle_low_latency_final(
        self,
        transcript,
        *,
        recognition_scope=None,
        source_identity=None,
    ):
        for identity, publication in self._scoped_publication_ids.items():
            self.publication_acoustics[publication] = identity.segment.segment_id
        buffer = self.merge_buffer
        self.suppress_rearm = False
        if self.policy in ("no_op", "combined") and buffer is not None:
            for index, publication in enumerate(buffer.utterance_ids):
                if publication == transcript.utterance_id:
                    self.suppress_rearm = (
                        self.output_projection.soft_reuse_mode(
                            buffer.parts[index], transcript.text.strip()
                        )
                        is not None
                        and not buffer.awaiting_vad_end
                        and not buffer.resume_pending
                        and not buffer.resume_confirmed
                    )
        before_resume = self.acoustic_names.get(buffer.resume_utterance_id) if buffer else None
        self.trace(
            "publication_contribution",
            acoustic=self.acoustic_names.get(
                self.publication_acoustics.get(transcript.utterance_id)
            ),
            publication=str(transcript.utterance_id),
            text=transcript.text,
            resume_before=before_resume,
        )
        try:
            if source_identity is None:
                await super()._handle_low_latency_final(
                    transcript, recognition_scope=recognition_scope
                )
            else:
                await super()._handle_low_latency_final(
                    transcript,
                    recognition_scope=recognition_scope,
                    source_identity=source_identity,
                )
        finally:
            self.suppress_rearm = False
        buffer = self.merge_buffer
        self.trace(
            "publication_reconciled",
            resume_after=self.acoustic_names.get(buffer.resume_utterance_id) if buffer else None,
        )

    def _credit_anchor(self, buffer):
        acoustics = {
            self.publication_acoustics.get(publication) for publication in buffer.utterance_ids
        }
        if None in acoustics or not acoustics or self.active_acoustics or self.unresolved_acoustics:
            return None
        if buffer.awaiting_vad_end or buffer.resume_pending or buffer.resume_confirmed:
            return None
        if any(
            self.actual_terminals.get(acoustic) != "final" or acoustic not in self.observed_ends
            for acoustic in acoustics
        ):
            return None
        anchors = [self.observed_ends[acoustic] for acoustic in acoustics]
        if any(
            anchor is None or not math.isfinite(anchor) or anchor > self.clock.now()
            for anchor in anchors
        ):
            return None
        return max(anchors)

    def _restart_post_end_grace(self, buffer):
        if self.suppress_rearm and buffer.finalize_wait_task is not None:
            self.trace("grace_rearm_suppressed", deadline_s=self.deadlines.get(buffer.merge_id))
            return
        anchor = self._credit_anchor(buffer) if self.policy in ("elapsed", "combined") else None
        if anchor is None:
            super()._restart_post_end_grace(buffer)
            deadline = (
                self.clock.now()
                + self.config_snapshot().value.low_latency_finalize_wait_ms / 1000.0
            )
            rule = "handler_receipt_baseline"
        else:
            self._cancel_finalize_wait(buffer)
            deadline = max(
                self.clock.now(),
                anchor + self.config_snapshot().value.low_latency_finalize_wait_ms / 1000.0,
            )
            buffer.finalize_wait_started_at = self.clock.now()
            buffer.finalize_wait_task = asyncio.create_task(
                self._finalize_wait_timeout(
                    buffer.merge_id,
                    buffer.finalize_wait_started_at,
                    max(0, round((deadline - self.clock.now()) * 1000)),
                )
            )
            rule = "sealed_VAD_end_credit"
        self.deadlines[buffer.merge_id] = deadline
        self.trace(
            "grace_armed",
            rule=rule,
            anchor_s=anchor,
            deadline_s=deadline,
            terminal_readiness=sorted(self.actual_terminals.values()),
        )

    async def _settle_scoped_terminal(self, identity):
        buffer = self.merge_buffer
        if (
            self.policy in ("elapsed", "combined")
            and buffer is not None
            and buffer.finalize_wait_task is not None
            and self._credit_anchor(buffer) is not None
        ):
            if (
                self.deadlines.get(buffer.merge_id, 0)
                > self._credit_anchor(buffer)
                + self.config_snapshot().value.low_latency_finalize_wait_ms / 1000.0
            ):
                self._restart_post_end_grace(buffer)
        await super()._settle_scoped_terminal(identity)

    async def _evaluate_speculative_next_action(self, buffer, *, reason):
        attempt = buffer.speculative_attempt
        self.trace(
            "eligibility_evaluated",
            reason=reason,
            blocker=self._continuation_blocker(buffer),
            attempt_status=attempt.status.value if attempt is not None else None,
            provider_generation_current=attempt is not None
            and attempt.provider_generation == self.translation_requests.provider_generation,
            translation_config_current=attempt is not None
            and self._translation_config_matches(
                attempt.config_snapshot.value, self.config_snapshot().value
            ),
            resume_acoustic=self.acoustic_names.get(buffer.resume_utterance_id),
            awaiting_vad_end=buffer.awaiting_vad_end,
            grace_deadline_s=self.deadlines.get(buffer.merge_id),
            active_acoustics=sorted(self.acoustic_names[item] for item in self.active_acoustics),
            unresolved_acoustics=sorted(
                self.acoustic_names[item] for item in self.unresolved_acoustics
            ),
        )
        await super()._evaluate_speculative_next_action(buffer, reason=reason)

    def account_wait(self, duration):
        buffer = self.merge_buffer
        if (
            buffer is None
            or buffer.finalize_wait_task is None
            or buffer.awaiting_vad_end
            or buffer.resume_pending
            or buffer.resume_confirmed
            or self.active_acoustics
        ):
            return
        source_acoustics = {
            self.publication_acoustics.get(publication) for publication in buffer.utterance_ids
        }
        if self.unresolved_acoustics - source_acoustics:
            return
        attempt = buffer.speculative_attempt
        if (
            attempt is not None
            and attempt.status is _SpeculativeAttemptStatus.READY
            and attempt.provider_generation == self.translation_requests.provider_generation
            and self._translation_config_matches(
                attempt.config_snapshot.value, self.config_snapshot().value
            )
        ):
            self.grace_only_wait_s += duration

    async def _resume_end_timeout(self, *arguments):
        key = arguments[-2]
        token = self._timeout_source.set(key)
        try:
            await super()._resume_end_timeout(*arguments)
        finally:
            self._timeout_source.reset(token)

    def _emit_metric(self, message, *args):
        self.metrics[message.split()[1]] += 1
        if message.split()[1] == "resume_end_timeout":
            key = self._timeout_source.get()
            if key is not None:
                acoustic = getattr(key, "segment_id", key)
                self.unresolved_acoustics.discard(acoustic)
                self.trace(
                    "continuation_watchdog_settled", acoustic=self.acoustic_names.get(acoustic)
                )

    async def _commit_merge(self, buffer, *, reason):
        parts = list(buffer.parts)
        acoustics = [
            self.acoustic_names.get(self.publication_acoustics.get(publication))
            for publication in buffer.utterance_ids
        ]
        blockers = self._continuation_blocker(buffer)
        await super()._commit_merge(buffer, reason=reason)
        if blockers is None and self.merge_buffer is not buffer:
            row = {
                "at_s": self.clock.now(),
                "parts": parts,
                "acoustics": acoustics,
                "reason": reason,
            }
            self.commits.append(row)
            source_acoustics = {
                self.publication_acoustics.get(publication) for publication in buffer.utterance_ids
            }
            if (self.active_acoustics | self.unresolved_acoustics) - source_acoustics:
                self.premature_commits.append(row)
            self.trace(
                "source_committed",
                **row,
                active_acoustics=sorted(
                    self.acoustic_names.get(item) for item in self.active_acoustics
                ),
                unresolved_acoustics=sorted(
                    self.acoustic_names.get(item) for item in self.unresolved_acoustics
                ),
            )


async def run_case(case: Case, policy: str, monkeypatch) -> dict:
    from puripuly_heart.core.orchestrator import self_translation_channel
    from puripuly_heart.core.runtime import self_capture
    from tests.core.test_self_translation_low_latency import FakeOscQueue, RecordingOverlaySink

    owner_source = os.environ["SELF_GRACE_233_OWNER_SOURCE"]
    source_module = self_translation_channel
    owner_class = type(
        "GraceExperimentOwner",
        (GraceProbeMixin, source_module.SelfTranslationChannelOwner),
        {},
    )

    clock = ControlledClock()
    timeline = []

    def trace(event, **fields):
        timeline.append({"at_s": clock.now(), "event": event, **fields})

    class TimedOverlay(RecordingOverlaySink):
        async def emit(self, event):
            trace(
                "overlay_accepted",
                kind=getattr(event, "type", None),
                primary=getattr(event, "text", ""),
                secondary=getattr(event, "secondary_text", ""),
                physical_display_observed=False,
            )
            await super().emit(event)

    session = ControlledScopedSession()
    session.allows_sealed_turn_overlap = case.overlap
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        channel="self",
        sleep=clock.sleep,
        monotonic_clock=clock.now,
        watchdog_resolver=lambda _: STTRecognitionWatchdogs(final_timeout_s=20, idle_timeout_s=60),
        session_lifetime_enabled=False,
    )
    llm = ScheduledLLM(
        clock,
        trace,
        delay=case.llm_delay,
        fail_first=case.fail_first,
        retiring=case.retire_after_cancel,
    )
    overlay = TimedOverlay()
    with monkeypatch.context() as patch:
        patch.setattr(translation_owners, "SelfTranslationChannelOwner", owner_class)
        patch.setattr(source_module, "asyncio", AsyncioClockFacade(clock))
        patch.setattr(self_capture, "asyncio", AsyncioClockFacade(clock, source_loop=True))
        harness = translation_owners.compose_translation_test_harness(
            stt=engine,
            llm=llm,
            osc=FakeOscQueue(),
            clock=clock,
            overlay_sink=overlay,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=400,
            low_latency_awaiting_vad_timeout_s=case.awaiting_timeout,
            source_language="en",
            target_language="ko",
            self_target_languages=("ko", "ja") if case.two_targets else ("ko",),
            concurrency_limit=10,
            peer_translation_enabled=True,
        )
        owner = harness.self_owner
        owner.configure_experiment(policy, trace)
        clock.observer = owner.account_wait
        capture, _, _, _, loop, _ = build_owner(provider=RecordingProvider(ready=True), sink=owner)
        session_config = replace(
            config(vad_hangover_ms=400),
            provider_id="local_qwen" if case.overlap else "deepgram",
            provider_signature=("grace233",),
            runtime_signature=("grace233",),
        )
        profile = asdict(harness.configuration.snapshot().value)
        ids = {name: uuid4() for name in ("P", "A", "B", "C")}
        owner.acoustic_names = {value: name for name, value in ids.items()}
        sequence = Counter()
        stable_events = {}
        stopped = False
        await harness.start()
        await capture.apply_intent(session_config, enabled=True)
        await loop.started.wait()
        guarded = loop.calls[0]["sink"]
        engine.accepted_settings_scope = (
            guarded.ledger.settings.provider_id,
            guarded.ledger.settings.provider_signature,
            guarded.ledger.settings.runtime_signature,
        )
        try:
            for step in sorted(case.steps, key=lambda item: item.at):
                await clock.advance_to(step.at)
                trace("scheduled_input", step=asdict(step))
                acoustic = ids[step.acoustic]
                if step.event in ("start", "chunk", "end"):
                    if step.event == "end":
                        event = SpeechEnd(acoustic, trailing_silence_ms=0)
                    else:
                        sample_start = round(step.at * 16000)
                        span = AudioCaptureSpan(
                            capture_epoch=1,
                            callback_sequence=sample_start + 1,
                            source_sample_rate_hz=16000,
                            source_start_sample=sample_start,
                            source_end_sample=sample_start + 160,
                            source_start_monotonic_s=step.at,
                            source_end_monotonic_s=step.at + 0.01,
                            normalized_sample_rate_hz=16000,
                            normalized_start_sample=sample_start,
                            normalized_end_sample=sample_start + 160,
                        )
                        data = np.full(160, 0.1, dtype=np.float32)
                        event = (
                            SpeechStart(
                                acoustic, np.empty(0, dtype=np.float32), data, chunk_capture=(span,)
                            )
                            if step.event == "start"
                            else SpeechChunk(acoustic, data, chunk_capture=(span,))
                        )
                    clock.source_observed_at = step.source_at
                    try:
                        await guarded.handle_vad_event(event)
                    finally:
                        clock.source_observed_at = None
                elif step.event in (
                    "stable",
                    "stable_suffix",
                    "same_snapshot",
                    "duplicate",
                    "terminal",
                ):
                    request = next(
                        (
                            request
                            for request in session.requests
                            if request.identity.segment.segment_id == acoustic
                        ),
                        None,
                    )
                    assert request is not None, (case.name, step, session.requests)
                    identity = request.identity
                    if step.event == "terminal":
                        session.emit(
                            STTProviderTurnTerminal(
                                identity,
                                step.outcome,
                                text=step.text,
                                text_authority=(
                                    "authoritative"
                                    if step.outcome in ("final", "empty")
                                    else "none"
                                ),
                                failure_reason=(
                                    None if step.outcome in ("final", "empty") else step.outcome
                                ),
                            )
                        )
                    elif step.event == "duplicate":
                        session.emit(stable_events[step.acoustic])
                    else:
                        sequence[step.acoustic] += 1
                        event = STTProviderTurnUpdate(
                            identity,
                            sequence[step.acoustic],
                            "stable",
                            "replace" if step.event == "same_snapshot" else "append",
                            step.text,
                            provenance=STTNativeProvenance(
                                native_event_id=f"{step.acoustic}:{sequence[step.acoustic]}"
                            ),
                        )
                        stable_events[step.acoustic] = event
                        session.emit(event)
                elif step.event == "replace_llm":
                    replacement = ScheduledLLM(clock, trace, delay=0.05)
                    await harness.replace_llm_provider(replacement)
                elif step.event == "language":
                    harness.replace_configuration(
                        target_language="de", self_target_languages=("de",)
                    )
                elif step.event == "recognition_config":
                    guarded.ledger.rebind(
                        activation_generation=capture.snapshot.generation,
                        settings=replace(
                            guarded.ledger.settings,
                            provider_signature=("changed",),
                            runtime_signature=("changed",),
                        ),
                    )
                    session = ControlledScopedSession()
                    session.allows_sealed_turn_overlap = case.overlap
                    engine.accepted_settings_scope = (
                        guarded.ledger.settings.provider_id,
                        guarded.ledger.settings.provider_signature,
                        guarded.ledger.settings.runtime_signature,
                    )
                elif step.event == "manual":
                    await owner.submit_text(step.text)
                elif step.event == "recognition_unit":
                    stream = RecognitionStreamIdentity("self", 1, 1, "independent", ())
                    await owner.handle_recognition_unit(
                        STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), step.text)
                    )
                elif step.event == "peer":
                    transcript = Transcript(uuid4(), step.text, is_final=True, channel="peer")
                    await harness.dispatch_stt_event(
                        STTFinalEvent(transcript.utterance_id, transcript)
                    )
                elif step.event == "OFF":
                    await owner.reset_provider_channel()
                elif step.event == "shutdown":
                    await harness.stop()
                    stopped = True
                else:
                    raise AssertionError(step.event)
                await clock.settle()
            await clock.advance_to(case.until)
            finals = []
            while not harness.ui_events.empty():
                event = harness.ui_events.get_nowait()
                if event.type is UIEventType.TRANSCRIPT_FINAL:
                    finals.append({"channel": event.channel, "text": event.payload.text})
            result = {
                "case": case.name,
                "policy": policy,
                "revision": BASELINE,
                "owner_source": owner_source,
                "owner_source_sha256": hashlib.sha256(
                    Path(self_translation_channel.__file__).read_bytes()
                ).hexdigest(),
                "profile": profile,
                "route": case.route,
                "provider_boundary": "ControlledScopedSession, not live ASR",
                "source_inputs": len(
                    {step.acoustic for step in case.steps if step.event == "start"}
                ),
                "scheduled_steps": [
                    asdict(step) for step in sorted(case.steps, key=lambda item: item.at)
                ],
                "llm_delay_s": case.llm_delay,
                "provider_overlap": case.overlap,
                "commits": owner.commits,
                "premature_commits": owner.premature_commits,
                "required_continuation_status": "failed" if owner.premature_commits else "passed",
                "grace_only_wait_ms": round(owner.grace_only_wait_s * 1000, 3),
                "metrics": dict(owner.metrics),
                "llm_invocations": [row for row in timeline if row["event"] == "llm_invoked"],
                "physical_active_at_observation_end": sum(
                    row["event"] == "llm_invoked" for row in timeline
                )
                - sum(row["event"] == "llm_physical_finished" for row in timeline),
                "continuation_entries_at_observation_end": (
                    len(owner._acoustic_continuations)
                    if hasattr(owner, "_acoustic_continuations")
                    else None
                ),
                "finals": finals,
                "timeline": timeline,
                "evidence_limits": [
                    "Synthetic metadata/PCM levels are not speech or semantic quality proof",
                    "Timer futures explicitly wake with compatible controlled clock; clock advance alone is never a sleep",
                    "Sink acceptance is not chatbox send or physical display",
                    "Stable contribution is not request terminal",
                ],
            }
        finally:
            await guarded.abort()
            await capture.close()
            if not stopped:
                await harness.stop()
            await clock.settle()
    return result


def record_result(result):
    report_dir = os.environ.get("SELF_GRACE_233_REPORT_DIR")
    if report_dir:
        path = Path(report_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"{result['case']}__{result['policy']}.json").open(
            "x", encoding="utf-8"
        ) as output:
            output.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pytest", "pytest-asyncio", "sherpa-onnx", "onnxruntime")
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_grace_233_controlled_comparison(case, policy, monkeypatch):
    result = await run_case(case, policy, monkeypatch)
    record_result(result)
    expected_defect = case.expected_defect if result["owner_source"] == "baseline" else False
    assert bool(result["premature_commits"]) is expected_defect
    assert result["physical_active_at_observation_end"] == 0
    if result["owner_source"] == "historical-correctness":
        assert result["continuation_entries_at_observation_end"] == 0
        if case.expected_defect:
            expected_parts = {
                "delayed_A_during_B_pending": ["prior", "hello", "world"],
                "delayed_A_during_B_confirmed": ["prior", "hello", "world"],
                "B_before_buffer_final_only": ["hello", "world"],
                "B_unresolved_C_empty": ["prior", "hello", "world"],
                "B_unresolved_C_final": ["prior", "hello", "world", "later"],
                "prebuffer_B_ended_pending": ["hello", "world"],
                "B_watchdog_survives_C_false_start": ["prior", "hello"],
                "same_acoustic_suffix_during_B_pending": ["hello", "world"],
            }
            assert [part for row in result["commits"] for part in row["parts"]] == expected_parts[
                case.name
            ]
    if case.name == "final_after_end":
        expected = 0.9 if policy in ("unchanged", "no_op") else 0.55
        assert result["commits"][0]["at_s"] == pytest.approx(expected)
        assert len(result["llm_invocations"]) == 1
    if case.name == "slow_translation":
        assert result["commits"][0]["at_s"] == pytest.approx(1.3)
        assert result["grace_only_wait_ms"] == 0
    if case.name == "duplicate_native_and_same_snapshot":
        assert result["metrics"]["final_phase"] == 1
        assert len(result["llm_invocations"]) == 1
    if case.name == "repeated_identical_speech":
        assert result["commits"][0]["parts"] == ["hello", "hello"]
    if case.name == "watchdog_prefix_suffix_terminal":
        assert [row["parts"] for row in result["commits"]] == [["hello"], ["world"]]
    if case.name == "manual_and_recognition_unit":
        assert result["commits"] == []
        assert [row["text"] for row in result["finals"]] == ["manual", "independent unit"]
    if case.name == "boundary_period":
        assert len(result["llm_invocations"]) == 1
        assert result["commits"][0]["parts"] == ["hello."]
    if case.name in ("question_is_revision", "number_is_revision", "negation_is_revision"):
        assert len(result["llm_invocations"]) == 2
        assert result["llm_invocations"][-1]["text"] == result["commits"][0]["parts"][0]
        assert not any(row["event"] == "grace_rearm_suppressed" for row in result["timeline"])
    if case.name == "stable_contradiction":
        terminals = [
            row
            for row in result["timeline"]
            if row["event"] == "recognition_owner_receipt" and row["outcome"] is not None
        ]
        assert terminals[0]["outcome"] == "failed"
        assert result["finals"] == [{"channel": "self", "text": "pay 10"}]
    if case.name == "two_targets":
        assert [row["target"] for row in result["llm_invocations"]] == ["ko", "ja"]
        assert all(row["at_s"] < result["commits"][0]["at_s"] for row in result["llm_invocations"])
    if case.name == "speculation_failure":
        assert len(result["llm_invocations"]) == 2
        assert result["finals"] == [{"channel": "self", "text": "hello"}]
    if case.name == "OFF_manual_survives":
        assert result["finals"] == [{"channel": "self", "text": "manual"}]
        assert result["commits"] == []
    if case.name == "shutdown_running":
        assert result["finals"] == []
        assert result["commits"] == []
    if case.name.startswith("matching_B_"):
        assert result["finals"] == [{"channel": "self", "text": "hello"}]
    if case.name.startswith("G2_fixed_three_"):
        assert result["source_inputs"] == 3
        assert len(result["llm_invocations"]) == 3
        if case.name.endswith("fast"):
            assert [row["parts"] for row in result["commits"]] == [["one"], ["two"], ["three"]]
        else:
            assert [row["parts"] for row in result["commits"]] == [["one", "two", "three"]]
            assert result["metrics"]["spec_cancel"] == 2
            assert any(
                row["event"] == "llm_invoked" and row["physical_active"] == 2
                for row in result["timeline"]
            )
    if case.name == "provider_terminal_pre_end":
        terminals = [
            row
            for row in result["timeline"]
            if row["event"] == "recognition_owner_receipt" and row["outcome"] is not None
        ]
        assert terminals[0]["at_s"] == pytest.approx(0.3)
        assert result["llm_invocations"][0]["at_s"] == pytest.approx(0.3)
    if case.name == "delayed_VAD_receipt":
        ends = [
            row
            for row in result["timeline"]
            if row["event"] == "vad_owner_receipt" and row["kind"] == "SpeechEnd"
        ]
        assert ends[0]["at_s"] == pytest.approx(0.3)
        assert ends[0]["sealed_at_s"] == pytest.approx(0.1)
        expected = 0.55 if policy in ("elapsed", "combined") else 0.9
        assert result["commits"][0]["at_s"] == pytest.approx(expected)
    if case.name == "future_VAD_anchor_fallback":
        assert result["commits"][0]["at_s"] == pytest.approx(0.6)
        assert all(
            row["rule"] == "handler_receipt_baseline"
            for row in result["timeline"]
            if row["event"] == "grace_armed"
        )
    if case.name == "active_SELF_intervening_Peer":
        previews = [
            row
            for row in result["timeline"]
            if row["event"] == "overlay_accepted" and row["kind"] == "self_active_update"
        ]
        assert previews[0]["primary"] == "hello"
        assert previews[0]["at_s"] < result["commits"][0]["at_s"]
        assert not any(
            row["event"] == "overlay_accepted" and row["kind"] == "self_active_clear"
            for row in result["timeline"]
            if previews[0]["at_s"] <= row["at_s"] < result["commits"][0]["at_s"]
        )
        assert any(
            row["event"] == "overlay_accepted"
            and row["kind"] == "translation_final"
            and row["primary"] == "translated:ko:hello"
            for row in result["timeline"]
        )
