from __future__ import annotations

import asyncio
import json
import sys
import threading
import types
import wave
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from uuid import UUID, uuid4

import numpy as np

from experiments.psem_r2_policy.arms import (
    apply_observe_evidence,
    control_partition,
    r1_project_diagnostic,
    r2_translation_config,
    score_arm,
    translate_assignment,
)
from experiments.psem_r2_policy.budget import (
    BudgetError,
    BudgetLedger,
    Phase,
    deepgram_reserve_usd,
    openrouter_reserve_usd,
)
from experiments.psem_r2_policy.metrics import (
    aggregate_cluster_parents,
    cluster_id_for_meeting,
    confirmatory_decision,
    latency_by_operation,
    latency_record,
    load_ami_words,
    pair_parent_guard,
    policy_delta_rows,
    primary_pool_membership,
    u8_case_report,
    write_artifact,
)
from experiments.psem_r2_policy.credentials import load_runtime_secrets
from experiments.psem_r2_policy.sortformer_live import (
    NativeSortformerProducer,
    evidence_payload,
    hypothesis_from_live_event,
    hypothesis_at_boundary,
)
from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import (
    _recognition_retention_profile,
    _recognition_watchdogs,
)
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.runtime_resolution import STT_DEFAULT_DRAIN_TIMEOUT_S
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController
from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot, PeerAudioSegmentLedger
from puripuly_heart.core.audio.pretranslation_ownership import (
    PretranslationOwnershipOwner,
    PretranslationOwnershipUnit,
)
from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.orchestrator.configuration import TranslationRuntimeConfig
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.orchestrator.translation_turn import TranslationTurnChild
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmission,
    PeerCaptureAdmissionStatus,
    PeerCaptureLanguageFacts,
    PeerCaptureProviderMutation,
    PeerCaptureProviderMutationStatus,
    PeerCaptureResolvedTarget,
    PeerCaptureSessionConfig,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal, STTSessionProjection
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart, VadGating, create_peer_vad_gating
from puripuly_heart.domain.models import FinalLanguageRun, Translation
from puripuly_heart.providers.llm.openrouter import HttpxOpenRouterClient, OpenRouterLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import (
    TranslationOwnersTestHarness,
    compose_translation_test_harness,
)

HZ = 16000
RING_BUFFER_MS = 500
PREROLL_SECONDS = RING_BUFFER_MS / 1000.0
HANGOVER_SECONDS = 0.8
# Per-connection meter gap: <=0.5 s session prefix plus whole-second rounding.
SESSION_PAD_BILLABLE_SECONDS = 2.0
PINNED_TRANSLATION = "google/gemma-4-26b-a4b-it"
LIVE_ROUTE = {
    "asr_provider": "deepgram",
    "asr_model": "nova-3",
    "backend": "DeepgramRealtimeSTTBackend",
    "constructed_by": "DeepgramRealtimeSTTBackend",
    "session": "_DeepgramSDKSession",
    "translation": PINNED_TRANSLATION,
    "direction": "en->ko",
}
AMI_AUDIO = Path(r"C:/Users/salee/AppData/Local/Temp/opencode/stb_phase2_corpora/ami/audio")
_INTERCEPT_SCRIPTS: list[tuple["InterceptScript", ...]] = []
_INTERCEPT_CURSORS: list[int] = []
_INTERCEPT_LOOPS: list[asyncio.AbstractEventLoop] = []


@dataclass(slots=True)
class InterceptWord:
    word: str
    start: float
    end: float
    punctuated_word: str | None = None
    language: str = "en"


@dataclass(slots=True)
class InterceptScript:
    transcript: str
    words: tuple[InterceptWord, ...]
    translation: str = "안녕"
    failure: str | None = None
    finalize_ack_delay_s: float = 0.0
    translation_delay_s: float = 0.0
    session_open_delay_s: float = 0.0


@dataclass(slots=True)
class OwnershipObserveReceipt:
    hypothesis_id: str
    revision: int
    disposition: str
    available_at_monotonic_s: float
    applied_at_monotonic_s: float
    capture_epoch: int = 1
    requested_transition_sample: int | None = None
    actual_applied_sample: int | None = None
    segment_id: Any = None
    producer_generation: object = None
    reference_generation: object = None


class InterceptOpenRouterClient:
    def __init__(self, translation: str = "안녕", *, delay_s: float = 0.0) -> None:
        self.translation = translation
        self.delay_s = float(delay_s)
        self.calls: list[dict[str, Any]] = []

    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "text": text,
                "system_prompt": system_prompt,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
                "scene_participant_count": scene_participant_count,
            }
        )
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        return self.translation

    async def close(self) -> None:
        return None


def _safe_translation_error(exc: BaseException) -> dict[str, Any]:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    message = f"Translation request failed ({type(exc).__name__})"
    if isinstance(status, int):
        message = f"Translation request failed (HTTP {status})"
    return {
        "type": type(exc).__name__,
        "status": status if isinstance(status, int) else None,
        "message": message,
    }


def _translation_result_text(result: object) -> str:
    if isinstance(result, Translation):
        return result.translated_text
    translated = getattr(result, "translated_text", None)
    if isinstance(translated, str):
        return translated
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return str(result)


class BudgetedOpenRouter:
    def __init__(
        self,
        inner: OpenRouterLLMProvider,
        *,
        ledger: BudgetLedger | None,
        phase: Phase,
        network: bool,
        clock: Any | None = None,
        clock_scope: str = "system_monotonic",
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._phase = phase
        self._network = network
        self._clock = clock or SystemClock().now
        self._clock_scope = clock_scope
        self.reserves: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.arm: str | None = None

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
    ) -> Translation:
        client = HttpxOpenRouterClient(
            api_key="reserve-bound",
            model=PINNED_TRANSLATION,
            max_tokens=100,
        )
        body = client._build_request_body(
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
            scene_participant_count=scene_participant_count,
        )
        serialized = json.dumps(body, ensure_ascii=False)
        amount = openrouter_reserve_usd(serialized_request=serialized, max_tokens=100)
        request_id = f"openrouter-{uuid4().hex}"
        meta = {
            "kind": "openrouter",
            "model": PINNED_TRANSLATION,
            "bytes": len(serialized.encode("utf-8")),
        }
        self.reserves.append({"id": request_id, "usd": amount, "meta": meta})
        request_record = {
            "id": request_id,
            "arm": self.arm,
            "utterance_id": str(utterance_id),
            "text": text,
            "system_prompt": system_prompt,
            "source_language": source_language,
            "target_language": target_language,
            "context": context,
            "scene_participant_count": scene_participant_count,
            "bytes": meta["bytes"],
            "usd": amount,
            "outcome": None,
            "translated_text": None,
            "error": None,
            "dispatch_monotonic_s": self._clock(),
            "completion_monotonic_s": None,
            "clock_scope": self._clock_scope,
        }
        self.requests.append(request_record)
        if self._ledger is not None:
            self._ledger.reserve(
                request_id,
                phase=self._phase,
                amount_usd=amount,
                meta=meta,
            )
        try:
            result = await self._inner.translate(
                utterance_id=utterance_id,
                text=text,
                system_prompt=system_prompt,
                source_language=source_language,
                target_language=target_language,
                context=context,
                scene_participant_count=scene_participant_count,
            )
        except BaseException as exc:
            request_record["outcome"] = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            )
            request_record["error"] = _safe_translation_error(exc)
            request_record["completion_monotonic_s"] = self._clock()
            if self._ledger is not None:
                self._ledger.settle(request_id, keep_reserve=True)
            raise
        request_record["outcome"] = "translated"
        request_record["translated_text"] = _translation_result_text(result)
        request_record["completion_monotonic_s"] = self._clock()
        if self._ledger is not None:
            if self._network:
                self._ledger.settle(request_id, keep_reserve=True)
            else:
                self._ledger.settle(request_id, billed_usd=0.0)
        return result

    async def close(self) -> None:
        await self._inner.close()


@dataclass(slots=True)
class _RecordingTranslationOutput:
    inner: Any
    records: list[dict[str, Any]]
    clock: Any
    clock_scope: str

    async def submit_translation_output(self, submission: Any) -> object | None:
        translation = submission.translation
        self.records.append(
            {
                "parent_utterance_id": str(submission.parent_utterance_id),
                "child_utterance_id": str(submission.child_utterance_id),
                "sequence": int(submission.sequence),
                "channel": submission.channel,
                "source": submission.source,
                "source_text": submission.source_text,
                "source_language": submission.source_language,
                "target_language": submission.target_language,
                "outcome": submission.outcome,
                "translated_text": (
                    None if translation is None else _translation_result_text(translation)
                ),
                "failure_code": submission.failure_code,
                "submitted_at_monotonic_s": self.clock(),
                "clock_scope": self.clock_scope,
            }
        )
        return await self.inner.submit_translation_output(submission)


def children_payload(
    children: Sequence[TranslationTurnChild],
    *,
    outputs: Sequence[Mapping[str, Any]] = (),
    requests: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    output_by_child = {
        str(row["child_utterance_id"]): row
        for row in outputs
        if row.get("child_utterance_id") is not None
    }
    request_by_child = {
        str(row["utterance_id"]): row for row in requests if row.get("utterance_id") is not None
    }
    rows: list[dict[str, Any]] = []
    for child in children:
        child_id = str(child.utterance_id)
        output = output_by_child.get(child_id)
        request = request_by_child.get(child_id)
        if request is None:
            request_status = "not_called"
        elif request.get("outcome") == "translated":
            request_status = "succeeded"
        else:
            request_status = "failed"
        rows.append(
            {
                "utterance_id": child_id,
                "parent_utterance_id": str(child.parent_utterance_id),
                "ownership_group_id": child.ownership_group_id,
                "text": child.transcript.text,
                "outcome": None if output is None else output.get("outcome"),
                "translated_text": (None if output is None else output.get("translated_text")),
                "failure_code": None if output is None else output.get("failure_code"),
                "request_status": request_status,
                "request_id": None if request is None else request.get("id"),
                "error": None if request is None else request.get("error"),
                "dispatch_monotonic_s": (
                    None if request is None else request.get("dispatch_monotonic_s")
                ),
                "completion_monotonic_s": (
                    None if request is None else request.get("completion_monotonic_s")
                ),
                "output_submission_monotonic_s": (
                    None if output is None else output.get("submitted_at_monotonic_s")
                ),
                "clock_scope": (
                    output.get("clock_scope")
                    if output is not None
                    else (None if request is None else request.get("clock_scope"))
                ),
            }
        )
    return rows


def token_span(terminal: STTProviderTurnTerminal | None) -> tuple[int | None, int | None]:
    if terminal is None:
        return None, None
    starts = [
        int(token.source_start_sample)
        for token in terminal.timed_tokens
        if token.source_start_sample is not None
    ]
    ends = [
        int(token.source_end_sample)
        for token in terminal.timed_tokens
        if token.source_end_sample is not None
    ]
    return (min(starts) if starts else None, max(ends) if ends else None)


def _synthetic_unit(
    terminal: STTProviderTurnTerminal,
    *,
    group_id: str,
    relation: str,
) -> tuple[PretranslationOwnershipUnit, ...]:
    tokens = terminal.timed_tokens
    if not tokens:
        return ()
    languages = tuple(dict.fromkeys(token.language for token in tokens if token.language))
    runs = terminal.final_language_runs or (
        FinalLanguageRun(terminal.text, languages[0] if languages else "en"),
    )
    start, end = token_span(terminal)
    return (
        PretranslationOwnershipUnit(
            group_id=group_id,
            relation=relation,
            text=terminal.text,
            language_runs=runs,
            token_indexes=tuple(range(len(tokens))),
            start_source_sample=start,
            end_source_sample=end,
        ),
    )


def r0_units(terminal: STTProviderTurnTerminal) -> tuple[PretranslationOwnershipUnit, ...]:
    return _synthetic_unit(terminal, group_id="R0-0", relation="CURRENT")


def unassigned_units(
    terminal: STTProviderTurnTerminal,
) -> tuple[PretranslationOwnershipUnit, ...]:
    return _synthetic_unit(terminal, group_id="", relation="UNKNOWN")


def effective_units(
    assignment: object | None,
    terminal: STTProviderTurnTerminal | None,
) -> tuple[PretranslationOwnershipUnit, ...]:
    if terminal is None or not terminal.text:
        return ()
    units = tuple(getattr(assignment, "units", ()) or ())
    if (
        getattr(assignment, "disposition", None) == "assigned"
        and bool(getattr(assignment, "conserved", False))
        and units
        and "".join(unit.text for unit in units) == terminal.text
    ):
        return units
    return unassigned_units(terminal)


def arm_merge(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "blocked",
            "ineligible",
            "reason",
            "unavailable",
            "zero_delay_injected",
            "assignment",
            "conserved",
            "n_units",
            "group_ids",
            "token_indexes",
            "relations",
            "child_ids",
            "child_groups",
            "child_texts",
            "reconstructed",
            "translated",
            "outcomes",
            "child_translations",
        )
        if key in summary
    }


def sanitize_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def pool_contamination(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {
        "attributable_chars": 0,
        "contaminated_chars": 0,
        "unknown_chars": 0,
        "mixed_chars": 0,
        "unaligned_chars": 0,
    }
    eligible = False
    sequential = False
    for arm in arms:
        contamination = arm.get("contamination")
        if not contamination:
            continue
        for name in totals:
            totals[name] += int(contamination.get(name) or 0)
        eligible = eligible or bool(contamination.get("eligible"))
        sequential = sequential or bool(contamination.get("sequential_target"))
    attributable = totals["attributable_chars"]
    return {
        **totals,
        "eligible": bool(eligible and attributable),
        "sequential_target": sequential,
        "proportion": ((totals["contaminated_chars"] / attributable) if attributable else None),
        "coverage": (
            "partial"
            if (totals["mixed_chars"] or totals["unaligned_chars"] or totals["unknown_chars"])
            else "full"
        ),
    }


def _arm_rows(parents: Sequence[Mapping[str, Any]], arm: str) -> list[Mapping[str, Any]]:
    return [row.get(arm) or {} for row in parents]


def _flat(rows: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    for row in rows:
        values.extend(list(row.get(key) or ()))
    return values


def _arm_view(arm: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arm.items() if key != "ledger"}


def r2_session_summary(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = _arm_rows(parents, "r2")
    return {
        "n_parents": len(parents),
        "assignment": (
            parents[0].get("assignment")
            if len({row.get("assignment") for row in parents}) == 1 and parents
            else "mixed"
        ),
        "conserved": (
            all(row.get("conserved") is not False for row in parents) if parents else False
        ),
        "n_units": sum(len(row.get("group_ids") or ()) for row in parents),
        "group_ids": _flat(parents, "group_ids"),
        "child_ids": _flat(arms, "child_ids"),
        "child_groups": _flat(arms, "child_groups"),
        "child_texts": _flat(arms, "child_texts"),
        "reconstructed": "".join(str(row.get("reconstructed") or "") for row in parents),
        "per_parent": [
            {
                "index": row.get("index"),
                "parent_id": row.get("parent_id"),
                **_arm_view(arm),
            }
            for row, arm in zip(parents, arms)
        ],
    }


def r0_session_summary(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = _arm_rows(parents, "r0")
    return {
        "n_parents": len(parents),
        "disposition": "disabled",
        "n_units": sum(int(arm.get("n_units") or 0) for arm in arms),
        "translated": bool(parents) and all(bool(arm.get("translated")) for arm in arms),
        "outcomes": _flat(arms, "outcomes"),
        "child_translations": _flat(arms, "child_translations"),
        "child_ids": _flat(arms, "child_ids"),
        "child_groups": _flat(arms, "child_groups"),
        "child_texts": _flat(arms, "child_texts"),
        "per_parent": [
            {
                "index": row.get("index"),
                "parent_id": row.get("parent_id"),
                **_arm_view(arm),
            }
            for row, arm in zip(parents, arms)
        ],
    }


def r1_session_summary(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = _arm_rows(parents, "r1")
    return {
        "n_parents": len(parents),
        "diagnostic": True,
        "translated": False,
        "seals": _flat(arms, "seals"),
        "per_parent": [
            {
                "index": row.get("index"),
                "parent_id": row.get("parent_id"),
                "seals": list(arm.get("seals") or ()),
                "history": list(arm.get("history") or ()),
            }
            for row, arm in zip(parents, arms)
        ],
    }


def control_session_summary(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = _arm_rows(parents, "control")
    reasons = [str(arm.get("reason")) for arm in arms if arm.get("blocked")]
    return {
        "n_parents": len(parents),
        "blocked": bool(arms) and all(bool(arm.get("blocked")) for arm in arms),
        "reason": reasons[0] if len(set(reasons)) == 1 and reasons else (reasons or None),
        "unavailable": _flat(arms, "unavailable"),
        "per_parent": [
            {
                "index": row.get("index"),
                "parent_id": row.get("parent_id"),
                **_arm_view(arm),
            }
            for row, arm in zip(parents, arms)
        ],
    }


def intercept_boundary(script: InterceptScript, *, offset_samples: int = 0) -> int:
    if not script.words:
        return offset_samples
    return offset_samples + int(round(float(script.words[0].end) * HZ))


def intercept_hypothesis(
    script: InterceptScript,
    *,
    offset_samples: int,
    capture_epoch: int,
    available_at_monotonic_s: float,
    producer_generation: object,
    reference_generation: object,
    hypothesis_id: str,
) -> ProspectiveSpeakerHypothesis:
    boundary = intercept_boundary(script, offset_samples=offset_samples)
    return ProspectiveSpeakerHypothesis(
        hypothesis_id=hypothesis_id,
        revision=0,
        capture_epoch=capture_epoch,
        support_start_sample=boundary - 100,
        support_end_sample=boundary + 100,
        estimated_transition_sample=boundary,
        observed_frontier_sample=boundary + 1600,
        available_at_monotonic_s=available_at_monotonic_s,
        producer_generation=producer_generation,
        reference_generation=reference_generation,
        producer_valid=True,
        reference_valid=True,
        local_slot=1,
    )


def compose_r2_harness(
    llm: BudgetedOpenRouter,
    *,
    owner: PretranslationOwnershipOwner | None = None,
    config: TranslationRuntimeConfig | None = None,
    output_records: list[dict[str, Any]] | None = None,
    clock: Any | None = None,
    clock_scope: str = "system_monotonic",
) -> TranslationOwnersTestHarness:
    configuration = config or r2_translation_config()
    llm.arm = "r2"
    harness = compose_translation_test_harness(
        osc=RecordingOscQueue(),
        llm=llm,
        peer_translation_enabled=configuration.peer_translation_enabled,
        translation_enabled=configuration.translation_enabled,
        peer_source_language=configuration.peer_source_language,
        peer_target_language=configuration.peer_target_language,
        source_language=configuration.source_language,
        target_language=configuration.target_language,
        fallback_transcript_only=configuration.fallback_transcript_only,
        system_prompt=configuration.system_prompt,
        context_time_window_s=configuration.context_time_window_s,
        context_max_entries=configuration.context_max_entries,
        integrated_context_time_window_s=configuration.integrated_context_time_window_s,
        integrated_context_max_entries=configuration.integrated_context_max_entries,
    )
    if output_records is not None:
        harness.translation_turns.output = _RecordingTranslationOutput(
            inner=harness.translation_turns.output,
            records=output_records,
            clock=clock or SystemClock().now,
            clock_scope=clock_scope,
        )
    if owner is not None:
        harness.peer_owner.pretranslation_ownership = owner
    return harness


class EnergyVadEngine:
    def __init__(self, threshold: float = 1e-3) -> None:
        self.threshold = threshold

    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float:
        _ = sample_rate_hz
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        return 1.0 if rms >= self.threshold else 0.0

    def reset(self) -> None:
        return None


def _silero_engine() -> Any:
    from puripuly_heart.core.vad.bundled import ensure_silero_vad_onnx
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    return SileroVadOnnx(ensure_silero_vad_onnx())


def make_peer_vad(
    *,
    engine: Any | None = None,
    use_silero: bool = False,
    onset_chunks: int | None = None,
) -> VadGating:
    selected = engine
    if selected is None and use_silero:
        selected = _silero_engine()
    if selected is None:
        if use_silero:
            raise FileNotFoundError("Silero VAD model is required")
        selected = EnergyVadEngine()
    if onset_chunks is None:
        return create_peer_vad_gating(
            selected,
            sample_rate_hz=HZ,
            ring_buffer_ms=RING_BUFFER_MS,
            hangover_ms=800,
        )
    return VadGating(
        selected,
        sample_rate_hz=HZ,
        ring_buffer_ms=RING_BUFFER_MS,
        hangover_ms=800,
        start_debounce_chunks=onset_chunks,
        start_commit_chunks=onset_chunks,
        external_delivery_boundaries=True,
        diagnostic_label="peer",
    )


class _IdleSource:
    async def close(self) -> None:
        return None


CAPTURE_FRAME_SECONDS = 512.0 / float(HZ)
FEED_PROGRESS_INTERVAL_S = 1.0
PEER_CAPTURE_CONFIG_TARGET = PeerCaptureTargetIntent(kind="default_output_device")


def _peer_capture_config() -> PeerCaptureSessionConfig:
    """Harness config for the production peer capture owner generation."""
    settings = _settings()
    return PeerCaptureSessionConfig(
        provider_id=settings.provider_id,
        provider_signature=settings.provider_signature,
        runtime_signature=settings.runtime_signature,
        capture_signature=(PEER_CAPTURE_CONFIG_TARGET, settings.target_sample_rate_hz),
        capture_target=PEER_CAPTURE_CONFIG_TARGET,
        language=PeerCaptureLanguageFacts(
            settings.source_mode,
            settings.source_language,
            settings.expected_languages,
        ),
        target_sample_rate_hz=settings.target_sample_rate_hz,
        vad_speech_threshold=settings.vad_speech_threshold,
        vad_hangover_ms=settings.vad_hangover_ms,
        vad_pre_roll_ms=settings.vad_pre_roll_ms,
    )


def _underlying_failure_reason(failure: BaseException) -> str:
    """Deepest concrete reason on the failure chain, never the bare class name."""
    reason = ""
    seen: set[int] = set()
    current: BaseException | None = failure
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip()
        if text:
            reason = text
        current = current.__cause__ or current.__context__
    return reason or type(failure).__name__


def _settings() -> AudioSegmentSettingsSnapshot:
    return AudioSegmentSettingsSnapshot(
        provider_id="deepgram",
        provider_signature=("deepgram", "nova-3"),
        runtime_signature=("deepgram", "nova-3", "en"),
        source_mode="manual",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=HZ,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )


def _span(
    start: int,
    end: int,
    *,
    epoch: int = 1,
    sequence: int = 1,
    start_monotonic_s: float | None = None,
    end_monotonic_s: float | None = None,
) -> AudioCaptureSpan:
    started = start / HZ if start_monotonic_s is None else start_monotonic_s
    ended = end / HZ if end_monotonic_s is None else end_monotonic_s
    return AudioCaptureSpan(
        capture_epoch=epoch,
        callback_sequence=sequence,
        source_sample_rate_hz=HZ,
        source_start_sample=start,
        source_end_sample=end,
        source_start_monotonic_s=started,
        source_end_monotonic_s=ended,
        normalized_sample_rate_hz=HZ,
        normalized_start_sample=start,
        normalized_end_sample=end,
    )


def load_wav_16k(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("WAV must be 16-bit PCM")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape((-1, channels)).mean(axis=1)
    if rate != HZ:
        src_len = int(samples.shape[0])
        dst_len = max(int(src_len * (HZ / rate)), 1)
        samples = np.interp(
            np.linspace(0.0, src_len - 1, num=dst_len),
            np.arange(src_len),
            samples,
        ).astype(np.float32)
    return samples


def ami_wav_path(meeting: str) -> Path:
    candidates = [
        AMI_AUDIO / f"{meeting}.Mix-Headset.wav",
        AMI_AUDIO / f"{meeting}.wav",
        AMI_AUDIO / meeting / f"{meeting}.Mix-Headset.wav",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"AMI wav not found for {meeting}")


def _results_event(script: InterceptScript, *, from_finalize: bool) -> SimpleNamespace:
    if from_finalize:
        alternative = SimpleNamespace(transcript="", words=())
    else:
        alternative = SimpleNamespace(
            transcript=script.transcript,
            words=tuple(
                SimpleNamespace(
                    word=item.word,
                    punctuated_word=item.punctuated_word or item.word,
                    start=item.start,
                    end=item.end,
                    language=item.language,
                )
                for item in script.words
            ),
        )
    return SimpleNamespace(
        channel=SimpleNamespace(alternatives=[alternative]),
        is_final=True,
        speech_final=False,
        from_finalize=from_finalize,
        metadata=SimpleNamespace(request_id="intercept-session", from_finalize=from_finalize),
    )


def intercept_scripts(script: object) -> tuple[InterceptScript, ...]:
    if isinstance(script, InterceptScript):
        return (script,)
    if script is None:
        return ()
    return tuple(script)


@contextmanager
def install_deepgram_intercept(
    script: InterceptScript | Sequence[InterceptScript],
) -> Iterator[tuple[InterceptScript, ...]]:
    sequence = intercept_scripts(script)
    _INTERCEPT_SCRIPTS.append(sequence)
    _INTERCEPT_CURSORS.append(0)
    _INTERCEPT_LOOPS.append(asyncio.get_running_loop())
    saved = {
        name: sys.modules.get(name)
        for name in (
            "deepgram",
            "deepgram.core",
            "deepgram.core.events",
            "deepgram.extensions",
            "deepgram.extensions.types",
            "deepgram.extensions.types.sockets",
        )
    }

    class FakeEventType:
        OPEN = "open"
        MESSAGE = "message"
        ERROR = "error"
        CLOSE = "close"

    class FakeControlMessage:
        def __init__(self, type: str) -> None:
            self.type = type

    class FakeConnection:
        def __init__(self) -> None:
            self._on_message = None
            self._on_error = None
            self._on_close = None
            self._loop = _INTERCEPT_LOOPS[-1] if _INTERCEPT_LOOPS else None
            self.sent_media: list[bytes] = []
            index = _INTERCEPT_CURSORS[-1]
            _INTERCEPT_CURSORS[-1] = index + 1
            self.script = sequence[min(index, len(sequence) - 1)]

        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def on(self, event_type, callback) -> None:
            if event_type == FakeEventType.OPEN:
                delay = float(self.script.session_open_delay_s)
                if delay > 0:
                    threading.Timer(delay, callback, args=(object(),)).start()
                else:
                    callback(object())
            elif event_type == FakeEventType.MESSAGE:
                self._on_message = callback
            elif event_type == FakeEventType.ERROR:
                self._on_error = callback
            elif event_type == FakeEventType.CLOSE:
                self._on_close = callback

        def start_listening(self) -> None:
            return None

        def send_media(self, data: bytes) -> None:
            self.sent_media.append(data)

        def send_control(self, message) -> None:
            if getattr(message, "type", None) != "Finalize":
                return
            if not sequence or self._on_message is None:
                return
            current = self.script
            if current.failure == "failed":
                if self._on_error is not None:
                    self._on_error(RuntimeError("intercept transport failure"))
                return
            self._on_message(_results_event(current, from_finalize=False))
            if current.failure == "degraded":
                if self._on_error is not None:
                    self._on_error(RuntimeError("intercept transport failure"))
                return
            if current.finalize_ack_delay_s > 0 and self._loop is not None:
                self._loop.call_later(
                    float(current.finalize_ack_delay_s),
                    self._on_message,
                    _results_event(current, from_finalize=True),
                )
                return
            self._on_message(_results_event(current, from_finalize=True))

    class FakeV1:
        def connect(self, **kwargs: Any) -> FakeConnection:
            _ = kwargs
            return FakeConnection()

    class FakeListen:
        v1 = FakeV1()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            _ = api_key
            self.listen = FakeListen()

    deepgram_pkg = types.ModuleType("deepgram")
    deepgram_pkg.DeepgramClient = FakeClient
    deepgram_core = types.ModuleType("deepgram.core")
    deepgram_events = types.ModuleType("deepgram.core.events")
    deepgram_events.EventType = FakeEventType
    deepgram_ext = types.ModuleType("deepgram.extensions")
    deepgram_ext_types = types.ModuleType("deepgram.extensions.types")
    deepgram_sockets = types.ModuleType("deepgram.extensions.types.sockets")
    deepgram_sockets.ListenV1ControlMessage = FakeControlMessage
    sys.modules["deepgram"] = deepgram_pkg
    sys.modules["deepgram.core"] = deepgram_core
    sys.modules["deepgram.core.events"] = deepgram_events
    sys.modules["deepgram.extensions"] = deepgram_ext
    sys.modules["deepgram.extensions.types"] = deepgram_ext_types
    sys.modules["deepgram.extensions.types.sockets"] = deepgram_sockets
    try:
        yield sequence
    finally:
        if _INTERCEPT_SCRIPTS and _INTERCEPT_SCRIPTS[-1] is sequence:
            _INTERCEPT_SCRIPTS.pop()
            _INTERCEPT_CURSORS.pop()
            _INTERCEPT_LOOPS.pop()
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _attach_live_source(owner: PeerCaptureSessionOwner, ledger: PeerAudioSegmentLedger) -> None:
    owner._segment_ledger = ledger
    owner._segment_ledgers.append(ledger)
    owner._activate_publication_generation(ledger.activation_generation)
    owner._desired_active = True
    owner._closed = False


def _peer_source(engine: ScopedRecognitionEngine, clock: SystemClock) -> PeerCaptureSessionOwner:
    class Admission:
        async def admit(self, _config: object) -> PeerCaptureAdmission:
            return PeerCaptureAdmission(PeerCaptureAdmissionStatus.ADMITTED)

    class Resolver:
        async def resolve(self, target: PeerCaptureTargetIntent) -> PeerCaptureTargetResolution:
            return PeerCaptureTargetResolution(
                PeerCaptureTargetStatus.RESOLVED,
                target=PeerCaptureResolvedTarget(intent=target),
            )

    class Provider:
        def is_ready(self, _config: object) -> bool:
            return True

        async def replace(self, _request: object, **_kwargs: object) -> PeerCaptureProviderMutation:
            return PeerCaptureProviderMutation(PeerCaptureProviderMutationStatus.APPLIED)

        async def start_ingress(self) -> None:
            return None

        async def release(self, *, mode: str, **_kwargs: object) -> None:
            _ = mode
            return None

    async def hold_capture(**_kwargs: object) -> None:
        await asyncio.Event().wait()

    return PeerCaptureSessionOwner(
        admission=Admission(),
        target_resolver=Resolver(),
        provider=Provider(),
        clock=clock,
        provider_request_factory=lambda *_args: "deepgram-live",
        source_factory=lambda *_args: _IdleSource(),
        vad_factory=lambda *_args: make_peer_vad(),
        run_audio_loop=hold_capture,
        vad_sink=engine,
    )


@dataclass(slots=True)
class ContinuousC5LiveRunner:
    network: bool
    ownership_enabled: bool = True
    intercept: InterceptScript | Sequence[InterceptScript] | None = None
    secrets: dict[str, str] = field(default_factory=dict)
    budget: BudgetLedger | None = None
    phase: Phase = "dev"
    context_pad_seconds: float = 0.0
    use_silero: bool = False
    vad_engine: Any | None = None
    artifact_dir: Path | None = None
    methods: list[str] = field(default_factory=list)
    open_session_calls: int = 0
    receipts: list[Any] = field(default_factory=list)
    children: list[TranslationTurnChild] = field(default_factory=list)
    marks: dict[str, float | None] = field(default_factory=dict)
    deepgram_reserve_usd: float | None = None
    translation_reserves: list[dict[str, Any]] = field(default_factory=list)
    translation_requests: list[dict[str, Any]] = field(default_factory=list)
    translation_outputs: list[dict[str, Any]] = field(default_factory=list)

    _clock: SystemClock = field(default_factory=SystemClock, init=False)
    _backend: DeepgramRealtimeSTTBackend | None = field(default=None, init=False)
    _engine: ScopedRecognitionEngine | None = field(default=None, init=False)
    _ledger: PeerAudioSegmentLedger | None = field(default=None, init=False)
    _c5: ListenDeliveryController | None = field(default=None, init=False)
    _vad: VadGating | None = field(default=None, init=False)
    _peer_source: PeerCaptureSessionOwner | None = field(default=None, init=False)
    _harness: Any = field(default=None, init=False)
    _owner: PretranslationOwnershipOwner | None = field(default=None, init=False)
    _producer: object = field(default=None, init=False)
    _reference: object = field(default=None, init=False)
    _utterance_id: Any = field(default=None, init=False)
    _cursor: int = field(default=0, init=False)
    _sequence: int = field(default=0, init=False)
    _terminal: STTProviderTurnTerminal | None = field(default=None, init=False)
    _terminal_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _admitted: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _intercept_cm: Any = field(default=None, init=False)
    _audio_seconds: float = field(default=0.0, init=False)
    _pcm_buffer: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32),
        init=False,
    )
    _speech_chunks: int = field(default=0, init=False)
    _silence_chunks: int = field(default=0, init=False)
    _seal_reasons: list[str] = field(default_factory=list, init=False)
    _hypotheses: list[ProspectiveSpeakerHypothesis] = field(default_factory=list, init=False)
    _evidence: list[dict[str, Any]] = field(default_factory=list, init=False)
    _llm: Any = field(default=None, init=False)
    _reserved_audio_seconds: float = field(default=0.0, init=False)
    _sent_audio_seconds: float = field(default=0.0, init=False)
    _reserved_sessions: int = field(default=0, init=False)
    _frontiers: list[dict[str, Any]] = field(default_factory=list, init=False)
    parent_terminals: list[STTProviderTurnTerminal] = field(default_factory=list, init=False)
    deepgram_reconciled: bool = field(default=False, init=False)
    deepgram_settled_usd: float | None = field(default=None, init=False)
    _deepgram_base_entry: str | None = field(default=None, init=False)
    _deepgram_base_amount: float = field(default=0.0, init=False)
    _deepgram_extra_entries: list[dict[str, Any]] = field(default_factory=list, init=False)
    _deepgram_pad_entries: list[dict[str, Any]] = field(default_factory=list, init=False)
    _session_segments: set[str] = field(default_factory=set, init=False)
    _segment_sent_start: dict[str, float] = field(default_factory=dict, init=False)
    parent_marks: dict[str, dict[str, float | None]] = field(default_factory=dict, init=False)
    child_terminals: dict[str, tuple[str, float | None]] = field(default_factory=dict, init=False)
    admissions: int = field(default=0, init=False)
    _admission_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _peer_config: PeerCaptureSessionConfig | None = field(default=None, init=False)
    _capture_generation: int | None = field(default=None, init=False)
    _dispatch: object | None = field(default=None, init=False)
    _submitted_segments: set[Any] = field(default_factory=set, init=False)
    _terminal_segments: set[Any] = field(default_factory=set, init=False)
    _recent_terminal_failures: list[str] = field(default_factory=list, init=False)
    _fed_samples: int = field(default=0, init=False)
    _feed_progress: list[list[float]] = field(default_factory=list, init=False)
    _feed_progress_second: int = field(default=0, init=False)
    _synthetic_hangover_samples: int = field(default=0, init=False)
    _last_span_end_s: float | None = field(default=None, init=False)
    _buffered_real_samples: int = field(default=0, init=False)
    _flush_pad_samples: int = field(default=0, init=False)
    _dropped_tail_samples: int = field(default=0, init=False)
    _capture_end_monotonic_s: float | None = field(default=None, init=False)
    _faulted: bool = field(default=False, init=False)
    _unprocessed_samples: int = field(default=0, init=False)
    provider_fault: dict[str, Any] | None = field(default=None, init=False)
    seal_lateness: list[dict[str, Any]] = field(default_factory=list, init=False)
    seal_lateness_violations: int = field(default=0, init=False)
    task_failures: list[str] = field(default_factory=list, init=False)
    _loop_exception_handler: Any = field(default=None, init=False)
    _task_handler_installed: bool = field(default=False, init=False)

    def _note(self, name: str) -> None:
        self.methods.append(name)

    def _record_task_failure(
        self,
        loop: asyncio.AbstractEventLoop,
        context: dict[str, Any],
    ) -> None:
        exception = context.get("exception")
        self.task_failures.append(
            str(exception) if exception is not None else str(context.get("message"))
        )
        handler = self._loop_exception_handler
        if handler is not None:
            handler(loop, context)

    def _install_task_failure_handler(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop_exception_handler = loop.get_exception_handler()
        loop.set_exception_handler(self._record_task_failure)
        self._task_handler_installed = True

    def _restore_task_failure_handler(self) -> None:
        if not self._task_handler_installed:
            return
        self._task_handler_installed = False
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(self._loop_exception_handler)

    async def open(self, *, audio_seconds: float = 0.2) -> None:
        self._note("open")
        self._install_task_failure_handler()
        self._audio_seconds = audio_seconds
        hangover_s = HANGOVER_SECONDS
        preroll_s = PREROLL_SECONDS
        tail_s = 512.0 / float(HZ)
        reconnect_bound = 0
        self._reserved_audio_seconds = (
            max(audio_seconds, 0.001) + self.context_pad_seconds + hangover_s + preroll_s + tail_s
        )
        self._sent_audio_seconds = 0.0
        self._reserved_sessions = 1 + reconnect_bound
        self._deepgram_extra_entries = []
        self._deepgram_pad_entries = []
        self._session_segments = set()
        self._segment_sent_start = {}
        self._deepgram_base_amount = deepgram_reserve_usd(
            max_audio_seconds=max(audio_seconds, 0.001),
            context_pad_seconds=self.context_pad_seconds,
            hangover_seconds=hangover_s,
            preroll_seconds=preroll_s,
            tail_seconds=tail_s,
            copies=1,
            reconnect_bound=reconnect_bound,
        )
        self.deepgram_reserve_usd = self._deepgram_base_amount
        if self.network and self.budget is None:
            raise BudgetError("network Deepgram requires a shared BudgetLedger")
        if self.budget is not None:
            entry = self.budget.reserve(
                f"deepgram-{uuid4().hex}",
                phase=self.phase,
                amount_usd=self.deepgram_reserve_usd,
                meta={
                    "kind": "deepgram",
                    "audio_seconds": audio_seconds,
                    "hangover_seconds": hangover_s,
                    "preroll_seconds": preroll_s,
                    "tail_seconds": tail_s,
                    "reconnect_bound": reconnect_bound,
                },
            )
            self._deepgram_base_entry = str(entry["id"])
        if self.intercept is not None:
            self._intercept_cm = install_deepgram_intercept(self.intercept)
            self._intercept_cm.__enter__()
        key = (self.secrets.get("DEEPGRAM_API_KEY") or "intercept-key").strip() or "intercept-key"
        provider_settings = SimpleNamespace(
            provider=STTProviderName.DEEPGRAM.value,
            drain_timeout_s=STT_DEFAULT_DRAIN_TIMEOUT_S,
            sample_rate_hz=HZ,
            channel="peer",
            provider_options={},
        )
        backend = DeepgramRealtimeSTTBackend(
            api_key=key,
            language="en",
            model="nova-3",
            keyterms=(),
            drain_timeout_s=provider_settings.drain_timeout_s,
        )
        self._backend = backend
        original = backend.open_session

        async def tracked_open(*, projection: STTSessionProjection = STTSessionProjection()):
            self.open_session_calls += 1
            self._reserve_scoped_session_open()
            return await original(projection=projection)

        setattr(backend, "open_session", tracked_open)
        clock = self._clock
        owner = PretranslationOwnershipOwner(enabled=self.ownership_enabled)
        self._owner = owner
        scripts = intercept_scripts(self.intercept)
        translation = scripts[0].translation if scripts else "안녕"
        inner = OpenRouterLLMProvider(
            api_key=(self.secrets.get("OPENROUTER_API_KEY") or "intercept-key"),
            model=PINNED_TRANSLATION,
            max_tokens=100,
            client=(
                None
                if self.network
                else InterceptOpenRouterClient(
                    translation,
                    delay_s=scripts[0].translation_delay_s if scripts else 0.0,
                )
            ),
        )
        llm = BudgetedOpenRouter(
            inner,
            ledger=self.budget,
            phase=self.phase,
            network=self.network,
            clock=self._clock.now,
            clock_scope="runner_monotonic",
        )
        self._llm = llm
        self.translation_reserves = llm.reserves
        self.translation_requests = llm.requests
        harness = compose_r2_harness(
            llm,
            owner=owner,
            output_records=self.translation_outputs,
            clock=self._clock.now,
            clock_scope="runner_monotonic",
        )
        await harness.start()
        self._harness = harness
        terminals: list[STTProviderTurnTerminal] = []

        async def session_factory(settings, epoch_id: str):
            _ = settings
            return await backend.open_session(
                projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch_id)
            )

        settings_scope = _settings()
        engine = ScopedRecognitionEngine(
            session_factory=session_factory,
            terminal_failure_sink=self._on_provider_terminal_failure,
            watchdog_resolver=lambda _settings: _recognition_watchdogs(provider_settings),
            accepted_settings_scope=(
                settings_scope.provider_id,
                settings_scope.provider_signature,
                settings_scope.runtime_signature,
            ),
            retention_profile_resolver=lambda settings: _recognition_retention_profile(
                provider_settings,
                settings,
            ),
            event_drain_timeout_s=provider_settings.drain_timeout_s,
        )
        self._engine = engine
        callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
        peer_source = _peer_source(engine, clock)
        callbacks.bind_self(harness.self_owner)
        callbacks.bind_peer(harness.peer_owner)
        callbacks.bind_peer_capture(peer_source)

        async def sink(event) -> None:
            if isinstance(event, STTProviderTurnTerminal):
                terminals.append(event)
                self.parent_terminals.append(event)
                self._terminal = event
                segment_id = event.identity.segment.segment_id
                self._terminal_segments.add(segment_id)
                if event.outcome not in ("final", "empty"):
                    self._recent_terminal_failures.append(str(event.failure_reason))
                    del self._recent_terminal_failures[:-4]
                now = clock.now()
                marks = self.parent_marks.setdefault(
                    str(segment_id),
                    {},
                )
                marks["recognition_terminal"] = now
                self.marks["recognition_terminal"] = now
                self._terminal_event.set()
            await callbacks.peer_event_handler(event)
            if isinstance(event, STTProviderTurnTerminal):
                now = clock.now()
                marks = self.parent_marks.setdefault(
                    str(event.identity.segment.segment_id),
                    {},
                )
                marks["translation_admission"] = now
                self.marks["translation_admission"] = now
                self.admissions += 1
                self._admitted.set()
                self._admission_event.set()

        engine.bind_event_sink(sink)
        inner_created = harness.translation_turns.on_child_created
        inner_terminal = harness.translation_turns.on_child_terminal

        async def created(child: TranslationTurnChild) -> None:
            self.children.append(child)
            parent_key = str(child.parent_utterance_id)
            self.parent_marks.setdefault(parent_key, {}).setdefault(
                "partition",
                clock.now(),
            )
            await inner_created(child)

        async def terminated(child: TranslationTurnChild, outcome: str) -> None:
            self.child_terminals[str(child.utterance_id)] = (outcome, clock.now())
            await inner_terminal(child, outcome)

        harness.translation_turns.on_child_created = created
        harness.translation_turns.on_child_terminal = terminated
        vad = make_peer_vad(
            engine=self.vad_engine,
            use_silero=self.use_silero or self.intercept is None,
            onset_chunks=1 if self.intercept is not None else None,
        )
        self._vad = vad
        self._pcm_buffer = np.empty((0,), dtype=np.float32)
        self._buffered_real_samples = 0
        self._last_span_end_s = None
        self._fed_samples = 0
        self._feed_progress.clear()
        self._feed_progress_second = 0
        self._flush_pad_samples = 0
        self._dropped_tail_samples = 0
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._seal_reasons = []

        peer_config = _peer_capture_config()
        capture = await peer_source.apply_intent(peer_config, enabled=True)
        ledger = peer_source.segment_ledger
        if ledger is None:
            raise RuntimeError("peer capture owner did not publish a segment ledger")

        async def emit_owned(owned) -> None:
            event = owned.event
            if isinstance(event, SpeechEnd):
                self._seal_reasons.append(str(event.reason))
                self._record_seal_lateness(owned)
                self._utterance_id = None
            await self._dispatch_owned(owned)

        c5 = ListenDeliveryController(
            vad=vad,
            ledger=ledger,
            emit=emit_owned,
            monotonic_clock=clock.now,
        )
        peer_source.bind_pretranslation_ownership(owner)
        self._ledger = ledger
        self._c5 = c5
        self._peer_source = peer_source
        self._peer_config = peer_config
        self._capture_generation = capture.generation
        self._dispatch = peer_source.guard_vad_sink(capture.generation)
        self._producer = object()
        self._reference = object()
        self.marks["open"] = clock.now()

    async def _dispatch_owned(self, owned: object) -> None:
        if self._faulted:
            return
        sink = self._dispatch
        if sink is None:
            raise RuntimeError("runner is not open")
        segment = getattr(owned, "segment", None)
        identity = getattr(segment, "identity", None)
        segment_id = getattr(identity, "segment_id", None)
        if isinstance(getattr(owned, "event", None), SpeechStart) and segment_id is not None:
            self._submitted_segments.add(segment_id)
        try:
            await sink.handle_owned_vad_event(owned)
        except RuntimeError as exc:
            await self._note_dispatch_fault(str(exc))

    def _record_seal_lateness(self, owned: object) -> None:
        segment = getattr(owned, "segment", None)
        opened_at = getattr(segment, "opened_at_monotonic_s", None)
        sealed_at = getattr(segment, "sealed_at_monotonic_s", None)
        if opened_at is None or sealed_at is None:
            return
        requested_deadline = opened_at + float(ListenDeliveryController.HARD_LIMIT_S)
        lateness = max(0.0, sealed_at - (requested_deadline + CAPTURE_FRAME_SECONDS))
        event = getattr(owned, "event", None)
        self.seal_lateness.append(
            {
                "segment_id": str(getattr(getattr(segment, "identity", None), "segment_id", "")),
                "seal_reason": str(getattr(event, "reason", "")),
                "opened_at_monotonic_s": opened_at,
                "requested_deadline_monotonic_s": requested_deadline,
                "sealed_at_monotonic_s": sealed_at,
                "lateness_s": lateness,
            }
        )
        if lateness > 0.0:
            self.seal_lateness_violations += 1

    async def _on_provider_terminal_failure(self, failure: Exception) -> None:
        await self._record_provider_fault(
            _underlying_failure_reason(failure),
            type(failure).__name__,
            failure,
        )

    async def _note_dispatch_fault(self, reason: str) -> None:
        await self._record_provider_fault(reason, "RuntimeError", RuntimeError(reason))

    async def _record_provider_fault(
        self,
        reason: str,
        exception_name: str,
        failure: Exception,
    ) -> None:
        if self.provider_fault is not None:
            return
        if self._recent_terminal_failures:
            reason = f"{reason} after {';'.join(self._recent_terminal_failures)}"
        self.provider_fault = {
            "reason": reason,
            "exception": exception_name,
            "at_monotonic_s": self._clock.now(),
            "open_session_calls": self.open_session_calls,
        }
        self._faulted = True
        source = self._peer_source
        if source is not None:
            await source.handle_terminal_provider_failure(failure)

    async def _finish_dispatch(self) -> None:
        sink = self._dispatch
        if sink is None:
            return
        if self._faulted:
            await sink.abort()
            return
        await sink.finish()

    async def _emit_vad(self, event: object) -> None:
        c5 = self._c5
        if c5 is None:
            raise RuntimeError("runner is not open")
        if isinstance(event, SpeechStart):
            self._utterance_id = event.utterance_id
            if self.marks.get("source_support") is None:
                self.marks["source_support"] = self._clock.now()
        elif isinstance(event, SpeechEnd):
            self._seal_reasons.append(str(event.reason))
            if event.reason == "delivery_deadline":
                self.marks["c5_deadline_violation"] = 1.0
            self._utterance_id = None
        await c5.handle_vad_event(event)

    async def _process_ready_chunks(self) -> None:
        vad = self._vad
        c5 = self._c5
        if vad is None or c5 is None:
            raise RuntimeError("runner is not open")
        chunk_samples = int(vad.chunk_samples)
        while self._pcm_buffer.size >= chunk_samples:
            chunk = self._pcm_buffer[:chunk_samples]
            self._pcm_buffer = self._pcm_buffer[chunk_samples:]
            self._buffered_real_samples -= min(self._buffered_real_samples, chunk_samples)
            self._sequence += 1
            start = self._cursor
            end = start + chunk_samples
            duration = chunk_samples / float(HZ)
            capture_end_s = self._capture_end_monotonic_s
            if capture_end_s is None:
                capture_end_s = self._clock.now()
            end_monotonic_s = capture_end_s - float(self._buffered_real_samples) / float(HZ)
            if self._last_span_end_s is not None and end_monotonic_s < self._last_span_end_s:
                end_monotonic_s = self._last_span_end_s
            self._last_span_end_s = end_monotonic_s
            span = _span(
                start,
                end,
                sequence=self._sequence,
                start_monotonic_s=end_monotonic_s - duration,
                end_monotonic_s=end_monotonic_s,
            )
            events = vad.process_owned_chunk(chunk, (span,))
            speech = bool(vad.last_observation_was_speech)
            if speech:
                self._speech_chunks += 1
            else:
                self._silence_chunks += 1
            for event in events:
                await self._emit_vad(event)
            await c5.observe_acoustic_chunk(speech_observed=speech, capture=(span,))
            self._cursor = end
            self._frontiers.append(
                {
                    "sample": end,
                    "available_at_monotonic_s": end_monotonic_s,
                }
            )

    def _ensure_audio_reserved(self, extra_seconds: float) -> None:
        if extra_seconds <= 0:
            return
        projected = self._sent_audio_seconds + extra_seconds
        if projected <= self._reserved_audio_seconds + 1e-12:
            self._sent_audio_seconds = projected
            return
        need = projected - self._reserved_audio_seconds
        amount = deepgram_reserve_usd(max_audio_seconds=max(need, 0.001))
        if self.budget is not None:
            entry = self.budget.reserve(
                f"deepgram-extra-{uuid4().hex}",
                phase=self.phase,
                amount_usd=amount,
                meta={"kind": "deepgram-extra", "audio_seconds": need},
            )
            self._deepgram_extra_entries.append(
                {
                    "id": str(entry["id"]),
                    "kind": "deepgram-extra",
                    "amount_usd": amount,
                    "sent": True,
                }
            )
        elif self.network:
            raise BudgetError("unreserved Deepgram PCM")
        self._reserved_audio_seconds += need
        self.deepgram_reserve_usd = (self.deepgram_reserve_usd or 0.0) + amount
        self._sent_audio_seconds = projected

    def _record_feed_progress(self) -> None:
        position = float(self._fed_samples) / float(HZ)
        second = int(position / FEED_PROGRESS_INTERVAL_S)
        if second <= self._feed_progress_second:
            return
        self._feed_progress_second = second
        self._feed_progress.append([self._clock.now(), position])

    async def _ingest(self, samples: np.ndarray, *, synthetic: bool = False) -> None:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        if synthetic:
            self._synthetic_hangover_samples += int(audio.size)
        else:
            self._fed_samples += int(audio.size)
            self._record_feed_progress()
        if self._faulted:
            self._unprocessed_samples += int(audio.size)
            return
        self._ensure_audio_reserved(float(audio.size) / float(HZ))
        self._capture_end_monotonic_s = self._clock.now()
        if self._pcm_buffer.size:
            self._pcm_buffer = np.concatenate([self._pcm_buffer, audio])
        else:
            self._pcm_buffer = audio.copy()
        self._buffered_real_samples += int(audio.size)
        await self._process_ready_chunks()

    async def _flush_partial(self) -> None:
        vad = self._vad
        if vad is None or self._pcm_buffer.size == 0:
            return
        pad = int(vad.chunk_samples) - int(self._pcm_buffer.size)
        if pad > 0:
            self._flush_pad_samples += pad
            self._pcm_buffer = np.concatenate(
                [self._pcm_buffer, np.zeros((pad,), dtype=np.float32)]
            )
        await self._process_ready_chunks()

    async def feed(self, samples: np.ndarray, *, context_only: bool = False) -> None:
        self._note("feed")
        _ = context_only
        await self._ingest(samples)

    async def receive(self, hypothesis: ProspectiveSpeakerHypothesis) -> OwnershipObserveReceipt:
        self._note("receive")
        owner = self._owner
        if owner is None:
            raise RuntimeError("runner is not open")
        self.marks["producer_receipt"] = self._clock.now()
        disposition = owner.observe(hypothesis)
        self._hypotheses.append(hypothesis)
        receipt = OwnershipObserveReceipt(
            hypothesis_id=hypothesis.hypothesis_id,
            revision=hypothesis.revision,
            disposition=str(disposition),
            available_at_monotonic_s=hypothesis.available_at_monotonic_s,
            applied_at_monotonic_s=self._clock.now(),
            capture_epoch=hypothesis.capture_epoch,
            requested_transition_sample=hypothesis.estimated_transition_sample,
            actual_applied_sample=hypothesis.estimated_transition_sample,
            producer_generation=hypothesis.producer_generation,
            reference_generation=hypothesis.reference_generation,
        )
        self.receipts.append(receipt)
        return receipt

    def apply_evidence(self, payload: dict[str, Any]) -> str:

        owner = self._owner
        if owner is None:
            raise RuntimeError("runner is not open")
        status = apply_observe_evidence(owner, payload)
        row = dict(payload)
        row["observe_evidence_status"] = status
        row["producer_generation_matches_active"] = (
            payload.get("producer_generation") is self._producer
        )
        row["reference_generation_matches_active"] = (
            payload.get("reference_generation") is self._reference
        )
        self._evidence.append(row)
        return status

    async def finalize(self) -> STTProviderTurnTerminal:
        self._note("finalize")
        vad = self._vad
        c5 = self._c5
        if c5 is None or vad is None:
            raise RuntimeError("runner is not open")
        await self._flush_partial()
        hangover_chunks = max(int(vad.hangover_chunks), 1)
        silence = np.zeros((int(vad.chunk_samples),), dtype=np.float32)
        for _ in range(hangover_chunks):
            if not vad.in_speech:
                break
            await self._ingest(silence, synthetic=True)
        if vad.in_speech:
            sealed = vad.seal_active(reason="source_eof")
            if sealed is not None:
                await self._emit_vad(sealed)
        await self._finish_dispatch()
        await asyncio.wait_for(self._terminal_event.wait(), timeout=8.0)
        terminal = self._terminal
        if terminal is None:
            raise RuntimeError("scoped Deepgram session did not emit a terminal")
        return terminal

    async def admit(self) -> STTProviderTurnTerminal:
        self._note("admit")
        await asyncio.wait_for(self._admitted.wait(), timeout=8.0)
        terminal = self._terminal
        if terminal is None:
            raise RuntimeError("admission missing provider terminal")
        return terminal

    async def translate(self) -> list[TranslationTurnChild]:
        self._note("translate")
        harness = self._harness
        if harness is None:
            raise RuntimeError("runner is not open")
        await harness.translation_turns.wait_for_idle()
        self.marks["translation_completion"] = self._clock.now()
        return list(self.children)

    async def close(self) -> None:
        self._restore_task_failure_handler()
        if self._peer_source is not None and self._peer_config is not None:
            await self._peer_source.apply_intent(self._peer_config, enabled=False)
        await self._finish_dispatch()
        if self._c5 is not None:
            await self._c5.close()
        if self._engine is not None:
            await self._engine.close()
        if self._harness is not None:
            await self._harness.stop()
        if self._backend is not None:
            close = getattr(self._backend, "close", None)
            if callable(close):
                result = close()
                if asyncio.iscoroutine(result):
                    await result
        if self._intercept_cm is not None:
            self._intercept_cm.__exit__(None, None, None)
            self._intercept_cm = None

    def live_hypothesis(self, boundary: int) -> ProspectiveSpeakerHypothesis:
        return hypothesis_at_boundary(
            boundary,
            capture_epoch=1,
            available_at_monotonic_s=self._clock.now(),
            producer_generation=self._producer,
            reference_generation=self._reference,
        )

    async def wait_for_admissions(self, count: int) -> None:
        while self.admissions < count:
            self._admission_event.clear()
            if self.admissions >= count:
                return
            await self._admission_event.wait()

    async def deliver_intercept_scripts(
        self,
        scripts: Sequence[InterceptScript],
        *,
        delivered: set[int],
    ) -> None:
        """Deliver prospectively-timed fixture hypotheses/evidence per open parent."""
        ledger = self._ledger
        if ledger is None or not scripts:
            return
        segment_id = ledger.current_open_segment_id
        if segment_id is None:
            return
        order = next(
            (
                snapshot.identity.segment_order
                for snapshot in ledger.snapshots
                if snapshot.identity.segment_id == segment_id
            ),
            None,
        )
        if order is None or order in delivered:
            return
        index = order - 1
        script = scripts[index % len(scripts)]
        origin = self._open_segment_origin()
        offset = 0 if origin is None else origin
        await self.receive(
            intercept_hypothesis(
                script,
                offset_samples=offset,
                capture_epoch=1,
                available_at_monotonic_s=self._clock.now(),
                producer_generation=self._producer,
                reference_generation=self._reference,
                hypothesis_id=f"intercept-{index}",
            )
        )
        for payload in intercept_covering_evidence(
            script,
            capture_epoch=1,
            producer_generation=self._producer,
            reference_generation=self._reference,
            available_at_monotonic_s=self._clock.now(),
            offset_samples=offset,
        ):
            self.apply_evidence(payload)
        delivered.add(order)

    def _scoped_turn_key(self) -> str | None:
        ledger = self._ledger
        if ledger is None:
            return None
        segment_id = ledger.current_open_segment_id
        return None if segment_id is None else str(segment_id)

    def _reserve_scoped_session_open(self) -> None:
        """Reserve the session pad and any re-sent PCM before a session opens.

        The pad covers the per-connection prefix and whole-second rounding that
        the sent-PCM bound cannot see; it stays reserved unless an exact provider
        usage audit justifies release.
        """
        self._reserve_session_pad()
        key = self._scoped_turn_key()
        if key is not None and key not in self._session_segments:
            self._session_segments.add(key)
            self._segment_sent_start[key] = self._sent_audio_seconds
            return
        if key is None and self.open_session_calls <= 1:
            return
        start = self._segment_sent_start.get(key) if key is not None else None
        turn_seconds = max(self._sent_audio_seconds - start, 0.0) if start is not None else 0.0
        bound = max(turn_seconds, PREROLL_SECONDS) + HANGOVER_SECONDS + 512.0 / float(HZ)
        amount = deepgram_reserve_usd(max_audio_seconds=bound, channels=1)
        entry_id: str | None = None
        if self.budget is None:
            if self.network:
                raise BudgetError("Deepgram retry without budget ledger")
        else:
            entry = self.budget.reserve(
                f"deepgram-retry-{uuid4().hex}",
                phase=self.phase,
                amount_usd=amount,
                meta={
                    "kind": "deepgram-retry",
                    "audio_seconds": bound,
                    "scoped_turn": key,
                },
            )
            entry_id = str(entry["id"])
            self._deepgram_extra_entries.append(
                {
                    "id": entry_id,
                    "kind": "deepgram-retry",
                    "amount_usd": amount,
                    "scoped_turn": key,
                }
            )
        self.deepgram_reserve_usd = (self.deepgram_reserve_usd or 0.0) + amount

    def _reserve_session_pad(self) -> None:
        amount = deepgram_reserve_usd(max_audio_seconds=SESSION_PAD_BILLABLE_SECONDS, channels=1)
        if self.budget is None:
            if self.network:
                raise BudgetError("Deepgram session pad without budget ledger")
            return
        entry = self.budget.reserve(
            f"deepgram-session-pad-{uuid4().hex}",
            phase=self.phase,
            amount_usd=amount,
            meta={
                "kind": "deepgram-session-pad",
                "billable_seconds": SESSION_PAD_BILLABLE_SECONDS,
                "scoped_turn": self._scoped_turn_key(),
            },
        )
        self._deepgram_pad_entries.append({"id": str(entry["id"]), "amount_usd": amount})
        self.deepgram_reserve_usd = (self.deepgram_reserve_usd or 0.0) + amount

    def _completion_auditable(self) -> bool:
        ledger = self._ledger
        if ledger is None or self.open_session_calls < 1:
            return False
        if ledger.current_open_segment_id is not None:
            return False
        receipts = ledger.terminal_receipts
        if not receipts:
            return False
        unreliable = {"failed", "expired", "cancelled"}
        return all(str(receipt.outcome) not in unreliable for receipt in receipts)

    def reconcile_deepgram_budget(self, *, completed: bool) -> None:
        """Release the unused allowance only on auditable completion.

        Per-session pads and failed in-flight sessions stay reserved.
        """
        self.deepgram_reconciled = False
        self.deepgram_settled_usd = None
        ledger = self.budget
        if ledger is None or self._deepgram_base_entry is None:
            return
        if not completed or not self._completion_auditable():
            return
        verified = deepgram_reserve_usd(
            max_audio_seconds=max(self._sent_audio_seconds, 0.001),
            channels=1,
        )
        billed = min(self._deepgram_base_amount, verified)
        ledger.settle(self._deepgram_base_entry, billed_usd=billed)
        for item in self._deepgram_extra_entries:
            if item.get("sent"):
                ledger.settle(str(item["id"]), billed_usd=float(item["amount_usd"]))
        self.deepgram_reconciled = True
        self.deepgram_settled_usd = billed

    def _open_segment_origin(self) -> int | None:
        """Source sample that a scoped session reports as session time zero.

        Fixture scripts carry session-relative word clocks, so the first owned
        sample of the open segment is shifted back by the preroll the segment
        can claim from unclaimed audio before it.
        """
        ledger = self._ledger
        if ledger is None:
            return None
        segment_id = ledger.current_open_segment_id
        if segment_id is None:
            return None
        content_start: int | None = None
        claimed_before = 0
        for snapshot in ledger.snapshots:
            ranges = snapshot.content_ranges
            if not ranges:
                continue
            if snapshot.identity.segment_id == segment_id:
                content_start = int(ranges[0].source_start_sample)
                continue
            end = int(ranges[-1].source_end_sample)
            if content_start is None or end <= content_start:
                claimed_before = max(claimed_before, end)
        if content_start is None:
            return None
        available = max(content_start - claimed_before, 0)
        preroll = min(int(round(PREROLL_SECONDS * HZ)), available)
        return content_start - preroll

    async def _seal_and_wait(
        self,
        target_admissions: int,
        *,
        silence_seconds: float,
        timeout_s: float,
    ) -> None:
        chunks = max(int(round(silence_seconds * HZ / 512.0)), 1)
        silence = np.zeros((512,), dtype=np.float32)
        for _ in range(chunks):
            if self.admissions >= target_admissions:
                return
            await self.feed(silence)
        if self.admissions >= target_admissions:
            return
        try:
            await asyncio.wait_for(
                self.wait_for_admissions(target_admissions),
                timeout=max(float(timeout_s), 0.01),
            )
        except asyncio.TimeoutError:
            return

    def _timing_failures(self, parents: Sequence[Mapping[str, Any]]) -> list[str]:
        failures: list[str] = []
        for row in parents:
            member, _reason = primary_pool_membership(row)
            if not member:
                continue
            parent_id = str(row.get("parent_id"))
            marks = row.get("marks") or {}
            receipt = row.get("receipt") or {}
            if marks.get("recognition_terminal") is None:
                failures.append(f"missing_recognition_terminal:{parent_id}")
            if marks.get("translation_admission") is None:
                failures.append(f"missing_admission_timing:{parent_id}")
            if receipt.get("terminal_at_monotonic_s") is None:
                failures.append(f"missing_receipt_timing:{parent_id}")
        return failures

    async def settle_pending_parents(self, *, timeout_s: float = 30.0) -> None:
        ledger = self._ledger
        if ledger is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(float(timeout_s), 0.0)
        while loop.time() < deadline:
            receipts = len(ledger.terminal_receipts)
            if (
                ledger.current_open_segment_id is None
                and receipts >= len(self._seal_reasons)
                and len(self.parent_terminals) >= receipts
            ):
                return
            await asyncio.sleep(0.02)

    def _requests_by_child(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in self.translation_requests:
            grouped.setdefault(str(row["utterance_id"]), []).append(row)
        return grouped

    def _children_by_parent(self) -> dict[str, list[TranslationTurnChild]]:
        grouped: dict[str, list[TranslationTurnChild]] = {}
        for child in self.children:
            grouped.setdefault(str(child.parent_utterance_id), []).append(child)
        return grouped

    async def evaluate_parents(
        self,
        *,
        meeting: str | None = None,
        native_chunks: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        ledger = self._ledger
        owner = self._owner
        clock = self._clock
        words = load_ami_words(meeting) if meeting else []
        receipt_rows = list(ledger.terminal_receipts) if ledger is not None else []
        provenance_valid = all(
            row.get("reference_valid") is not False
            and str(row.get("observe_evidence_status") or "observed") == "observed"
            for row in self._evidence
        )
        receipt_by_id = {str(item.identity.segment_id): item for item in receipt_rows}
        terminals = {str(item.identity.segment.segment_id): item for item in self.parent_terminals}
        ordered = [str(item.identity.segment.segment_id) for item in self.parent_terminals]
        for item in sorted(receipt_rows, key=lambda row: row.identity.segment_order):
            key = str(item.identity.segment_id)
            if key not in terminals and key not in ordered:
                ordered.append(key)
        requests_by_child = self._requests_by_child()
        children_by_parent = self._children_by_parent()
        parents: list[dict[str, Any]] = []
        for index, key in enumerate(ordered):
            terminal = terminals.get(key)
            receipt = receipt_by_id.get(key)
            marks = dict(self.parent_marks.get(key) or {})
            text = terminal.text if terminal is not None else ""
            tokens = tuple(terminal.timed_tokens) if terminal is not None else ()
            outcome = (
                receipt.outcome
                if receipt is not None
                else (terminal.outcome if terminal is not None else "missing")
            )
            assignment = None
            if owner is not None and terminal is not None:
                assignment = owner.committed(terminal.identity.segment.segment_id)
            units = effective_units(assignment, terminal)
            span = token_span(terminal)
            parent_hypotheses = tuple(
                item
                for item in self._hypotheses
                if span[0] is not None
                and span[1] is not None
                and span[0] <= int(item.estimated_transition_sample) <= span[1]
            )
            children = children_by_parent.get(key, [])
            child_outcomes = {
                str(child.utterance_id): self.child_terminals.get(str(child.utterance_id))
                for child in children
            }
            completions = [
                value[1]
                for value in child_outcomes.values()
                if value is not None and value[1] is not None
            ]
            if completions:
                marks.setdefault("translation_completion", max(completions))
            freeze = marks.get("recognition_terminal") or clock.now()
            deadline = marks.get("translation_admission")
            admission = float(clock.now() if deadline is None else deadline)
            requests = [
                row
                for child in children
                for row in requests_by_child.get(str(child.utterance_id), [])
            ]
            r0: dict[str, Any] = {
                "translated": False,
                "outcomes": [],
                "child_translations": [],
                "child_errors": {},
                "child_ids": [],
                "child_groups": [],
                "child_texts": [],
                "requests": [],
                "skipped_reason": None,
            }
            if terminal is not None and text:
                previous_arm = self._llm.arm
                self._llm.arm = "r0"
                try:
                    summary = await translate_assignment(
                        terminal,
                        owner=PretranslationOwnershipOwner(enabled=False),
                        events=(),
                        evidence=(),
                        llm=self._llm,
                        admitted_at_monotonic_s=admission,
                    )
                finally:
                    self._llm.arm = previous_arm
                r0.update(
                    {
                        "translated": bool(summary["translated"]),
                        "outcomes": list(summary["outcomes"]),
                        "child_translations": list(summary["child_translations"]),
                        "child_ids": list(summary["child_ids"]),
                        "child_groups": list(summary["child_groups"]),
                        "child_texts": list(summary["child_texts"]),
                    }
                )
                r0["requests"] = [
                    row
                    for child_id in summary["child_ids"]
                    for row in self.translation_requests
                    if str(row.get("utterance_id")) == str(child_id)
                ]
                r0["child_errors"] = {
                    str(row["utterance_id"]): row["error"]
                    for row in r0["requests"]
                    if row.get("error") is not None
                }
            elif terminal is None:
                r0["skipped_reason"] = "missing_terminal"
            else:
                r0["skipped_reason"] = "empty_text"
            r2_children = children_payload(
                children,
                outputs=self.translation_outputs,
                requests=requests,
            )
            r2: dict[str, Any] = {
                "translated": bool(children)
                and all(
                    (child_outcomes.get(str(child.utterance_id)) or (None, None))[0] == "translated"
                    for child in children
                ),
                "child_ids": [str(child.utterance_id) for child in children],
                "child_groups": [child.ownership_group_id for child in children],
                "child_texts": [child.transcript.text for child in children],
                "child_translations": [row.get("translated_text") for row in r2_children],
                "child_errors": {
                    str(row["utterance_id"]): row["error"]
                    for row in r2_children
                    if row.get("error") is not None
                },
                "child_outcomes": {
                    child_id: (None if value is None else value[0])
                    for child_id, value in child_outcomes.items()
                },
                "requests": requests,
            }
            if terminal is not None and text:
                control = await control_partition(
                    terminal,
                    meeting=meeting,
                    native_chunks=native_chunks,
                    admitted_at_monotonic_s=admission,
                    producer_generation=self._producer,
                    reference_generation=self._reference,
                )
            else:
                control = {"blocked": True, "reason": "missing_parent_text"}
            if terminal is not None and text:
                scoring = {
                    "words": words,
                    "receipts": self.receipts,
                    "marks": marks,
                    "meeting": meeting,
                    "seal_reasons": [str(reason) for reason in self._seal_reasons],
                    "speech_chunks": self._speech_chunks,
                    "silence_chunks": self._silence_chunks,
                }
                r0_scored = {**score_arm(terminal, r0_units(terminal), **scoring), **r0}
                r2_scored = {**score_arm(terminal, units, **scoring), **r2}
                if control.get("blocked"):
                    control_scored = {**arm_merge(control), "contamination": None}
                else:
                    control_scored = {
                        **score_arm(
                            terminal,
                            tuple(control.get("units") or ()),
                            **scoring,
                        ),
                        **arm_merge(control),
                    }
                r1 = r1_project_diagnostic(
                    terminal,
                    parent_hypotheses,
                    freeze_monotonic_s=float(freeze),
                    frontiers=list(self._frontiers),
                )
            else:
                r0_scored = dict(r0)
                r2_scored = dict(r2)
                control_scored = {**arm_merge(control), "contamination": None}
                r1 = {
                    "diagnostic": True,
                    "translated": False,
                    "assignment": "diagnostic",
                    "n_units": 0,
                    "group_ids": [],
                    "child_groups": [],
                    "seals": [],
                    "history": [],
                }
            text_authority = (
                receipt.text_authority
                if receipt is not None
                else (terminal.text_authority if terminal is not None else None)
            )
            failure_reason = (
                receipt.failure_reason
                if receipt is not None
                else (terminal.failure_reason if terminal is not None else None)
            )
            if terminal is None or receipt is None:
                status = "unsuccessful"
            elif outcome in {"failed", "expired", "cancelled"}:
                status = "unsuccessful"
            elif str(text_authority) == "degraded":
                status = "degraded"
            else:
                status = "complete"
            accounted = terminal is not None or receipt is not None
            record = {
                "index": index,
                "parent_id": key,
                "meeting": meeting,
                "cluster_id": cluster_id_for_meeting(meeting) if meeting else None,
                "outcome": outcome,
                "terminal_outcome": None if terminal is None else terminal.outcome,
                "seal_reason": None if receipt is None else receipt.segment.seal_reason,
                "status": status,
                "clean_completion": status == "complete",
                "degraded": status == "degraded",
                "unsuccessful_source_processing": not accounted,
                "accounted": accounted,
                "provenance_valid": provenance_valid,
                "text_authority": text_authority,
                "failure_reason": failure_reason,
                "incomplete": not accounted,
                "outage": False,
                "text": text,
                "n_timed": len(tokens),
                "timed_start_ms": [token.start_ms for token in tokens],
                "timed_timings": [token.timing for token in tokens],
                "span": [span[0], span[1]],
                "assignment": None if assignment is None else assignment.disposition,
                "conserved": None if assignment is None else bool(assignment.conserved),
                "unknown_reasons": list(getattr(assignment, "unknown_reasons", ()) or ()),
                "group_ids": [unit.group_id for unit in units],
                "reconstructed": "".join(unit.text for unit in units),
                "children": children_payload(
                    children,
                    outputs=self.translation_outputs,
                    requests=requests,
                ),
                "requests": requests,
                "evidence": [sanitize_evidence(row) for row in self._evidence],
                "hypotheses": [
                    {
                        "hypothesis_id": item.hypothesis_id,
                        "revision": item.revision,
                        "capture_epoch": item.capture_epoch,
                        "support_start_sample": item.support_start_sample,
                        "support_end_sample": item.support_end_sample,
                        "estimated_transition_sample": item.estimated_transition_sample,
                        "observed_frontier_sample": item.observed_frontier_sample,
                        "available_at_monotonic_s": item.available_at_monotonic_s,
                        "producer_generation_matches_active": (
                            item.producer_generation is self._producer
                        ),
                        "reference_generation_matches_active": (
                            item.reference_generation is self._reference
                        ),
                        "producer_valid": item.producer_valid,
                        "reference_valid": item.reference_valid,
                        "retracted": item.retracted,
                        "local_slot": item.local_slot,
                    }
                    for item in parent_hypotheses
                ],
                "marks": marks,
                "latency": latency_record(marks),
                "receipt": (
                    None
                    if receipt is None
                    else {
                        "segment_order": receipt.identity.segment_order,
                        "outcome": receipt.outcome,
                        "seal_reason": receipt.segment.seal_reason,
                        "terminal_at_monotonic_s": receipt.terminal_at_monotonic_s,
                        "text_authority": receipt.text_authority,
                        "failure_reason": receipt.failure_reason,
                        "content_ranges": [
                            [item.source_start_sample, item.source_end_sample]
                            for item in receipt.segment.content_ranges
                        ],
                    }
                ),
                "guard": pair_parent_guard(r0_scored.get("guard"), r2_scored.get("guard")),
                "r0": r0_scored,
                "r2": r2_scored,
                "r1": r1,
                "control": control_scored,
                "tokens": (r2_scored.get("ledger") or {}).get("tokens", []),
                "units": (r2_scored.get("ledger") or {}).get("units", []),
                "receipts": (r2_scored.get("ledger") or {}).get("receipts", []),
            }
            record["sequential_target"] = bool(
                (r0_scored.get("contamination") or {}).get("sequential_target")
                or (r2_scored.get("contamination") or {}).get("sequential_target")
                or (control_scored.get("contamination") or {}).get("sequential_target")
            )
            parents.append(record)
        aggregate = aggregate_cluster_parents(parents)
        per_cluster: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in parents:
            cluster = str(row.get("cluster_id") or row.get("meeting") or "unknown")
            slot = per_cluster.setdefault(cluster, {})
            for arm_name, arm_key in (
                ("R0", "r0"),
                ("R2", "r2"),
                ("R1", "r1"),
                ("control", "control"),
            ):
                arm = row.get(arm_key) or {}
                if arm.get("contamination") is None:
                    continue
                slot.setdefault(arm_name, []).append(arm)
        policies = {
            cluster: {
                arm_name: {"contamination": pool_contamination(arms)}
                for arm_name, arms in arms_by_name.items()
            }
            for cluster, arms_by_name in per_cluster.items()
        }
        return {
            "n_parents": len(parents),
            "parents": parents,
            "parent_texts": [row["text"] for row in parents],
            "aggregate": aggregate,
            "policy": policy_delta_rows(per_cluster=policies),
            "latency_by_operation": latency_by_operation([row["marks"] for row in parents]),
            "enabled": r2_session_summary(parents),
            "disabled": r0_session_summary(parents),
            "r0": r0_session_summary(parents),
            "r2": r2_session_summary(parents),
            "r1": r1_session_summary(parents),
            "control": control_session_summary(parents),
        }

    async def run_pcm(
        self,
        samples: np.ndarray,
        *,
        hypotheses: tuple[ProspectiveSpeakerHypothesis, ...] | None = None,
        boundary: int | None = 1600,
        covering_evidence: tuple[dict[str, Any], ...] | None = None,
        apply_intercept_evidence: bool = True,
        meeting: str | None = None,
        native_chunks: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        audio_seconds = float(np.asarray(samples).size) / float(HZ)
        await self.open(audio_seconds=audio_seconds)
        try:
            await self.feed(samples)
            if hypotheses is None and boundary is not None:
                hypotheses = (self.live_hypothesis(boundary),)
            for item in hypotheses or ():
                await self.receive(item)
            payloads = list(covering_evidence or ())
            if not payloads and apply_intercept_evidence and self.intercept is not None:
                payloads = intercept_covering_evidence(
                    self.intercept,
                    capture_epoch=1,
                    producer_generation=self._producer,
                    reference_generation=self._reference,
                    available_at_monotonic_s=self._clock.now(),
                )
            for payload in payloads:
                self.apply_evidence(payload)
            await self.finalize()
            await self.admit()
            await self.translate()
            return await self._session_payload(
                meeting=meeting,
                native_chunks=native_chunks,
            )
        finally:
            await self.close()

    async def run_bursts(
        self,
        bursts: Sequence[np.ndarray],
        *,
        hypotheses: Sequence[Sequence[ProspectiveSpeakerHypothesis]] = (),
        covering_evidence: Sequence[Sequence[Mapping[str, Any]]] = (),
        meeting: str | None = None,
        native_chunks: Sequence[Mapping[str, Any]] = (),
        seal_silence_seconds: float = 3.0,
        seal_timeout_seconds: float = 20.0,
        apply_intercept_evidence: bool = True,
    ) -> dict[str, Any]:
        scripts = intercept_scripts(self.intercept)
        intercept_delivered: set[int] = set()
        audio_seconds = sum(float(np.asarray(burst).size) for burst in bursts) / float(HZ)
        await self.open(audio_seconds=max(audio_seconds, 0.001))
        try:
            for index, burst in enumerate(bursts):
                await self.feed(burst)
                burst_hypotheses = tuple(hypotheses[index]) if index < len(hypotheses) else ()
                for item in burst_hypotheses:
                    await self.receive(item)
                payloads = list(covering_evidence[index]) if index < len(covering_evidence) else []
                for payload in payloads:
                    self.apply_evidence(payload)
                if not burst_hypotheses and not payloads and apply_intercept_evidence:
                    await self.deliver_intercept_scripts(scripts, delivered=intercept_delivered)
                await self._seal_and_wait(
                    self.admissions + 1,
                    silence_seconds=seal_silence_seconds,
                    timeout_s=seal_timeout_seconds,
                )
            await self._seal_and_wait(
                len(bursts),
                silence_seconds=seal_silence_seconds,
                timeout_s=seal_timeout_seconds,
            )
            await self.finalize()
            await self.admit()
            await self.translate()
            return await self._session_payload(
                meeting=meeting,
                native_chunks=native_chunks,
            )
        finally:
            await self.close()

    async def _session_payload(
        self,
        *,
        meeting: str | None,
        native_chunks: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        await self.settle_pending_parents()
        session = await self.evaluate_parents(
            meeting=meeting,
            native_chunks=native_chunks,
        )
        receipts_payload = [
            {
                "hypothesis_id": item.hypothesis_id,
                "disposition": item.disposition,
                "available_at_monotonic_s": item.available_at_monotonic_s,
                "applied_at_monotonic_s": item.applied_at_monotonic_s,
                "requested_transition_sample": item.requested_transition_sample,
                "capture_epoch": item.capture_epoch,
                "receipt_kind": "native_arrival",
            }
            for item in self.receipts
        ]
        conserved = all(
            bool((row.get("r2") or {}).get("conservation", {}).get("conserved_units_to_parent"))
            for row in session["parents"]
            if row["text"]
        )
        incomplete = any(row["incomplete"] for row in session["parents"])
        outage = any(row["outage"] for row in session["parents"])
        degraded = any(row["degraded"] for row in session["parents"])
        clean_completion = bool(session["parents"]) and all(
            bool(row.get("clean_completion")) for row in session["parents"]
        )
        payload = {
            "ok": False,
            "incomplete": incomplete,
            "outage": outage,
            "degraded": degraded,
            "clean_completion": clean_completion,
            "accepted_text_conserved": conserved,
            "sealed_segments": len(self._seal_reasons),
            "declared_source_samples": int(round(self._audio_seconds * HZ)),
            "timing_failures": self._timing_failures(session["parents"]),
            "task_failures": list(self.task_failures),
            "network": self.network,
            "intercept": self.intercept is not None,
            "adapter_reads_words": True,
            "adapter_records_origin": True,
            "path": (
                "c5_wav->scoped_engine->deepgram_open_session/feed/finalize"
                "->psem_receive->peer_admit->openrouter_children"
            ),
            "methods": list(self.methods),
            "open_session_calls": self.open_session_calls,
            "receipts": receipts_payload,
            "vad_speech_chunks": self._speech_chunks,
            "vad_silence_chunks": self._silence_chunks,
            "c5_seal_reasons": [str(reason) for reason in self._seal_reasons],
            "capture_timing": {
                "arrival_anchored": arrival_anchored(receipts_payload),
                **native_chunk_arrival_stats(native_chunks),
                "input_source_samples": int(round(self._audio_seconds * HZ)),
                "capture_frame_seconds": CAPTURE_FRAME_SECONDS,
                "feed_progress": {
                    "interval_s": FEED_PROGRESS_INTERVAL_S,
                    "samples": [list(item) for item in self._feed_progress],
                },
                "fed_source_samples": self._fed_samples,
                "synthetic_hangover_samples": self._synthetic_hangover_samples,
                "fed_total_source_samples": self._fed_samples + self._synthetic_hangover_samples,
                "chunked_source_samples": int(self._cursor),
                "buffered_source_samples": int(self._pcm_buffer.size),
                "flush_pad_source_samples": self._flush_pad_samples,
                "dropped_tail_source_samples": self._dropped_tail_samples,
                "unprocessed_source_samples": int(self._unprocessed_samples),
                "last_arrival_monotonic_s": self._capture_end_monotonic_s,
            },
            "seal_lateness": {
                "deadline_s": float(ListenDeliveryController.HARD_LIMIT_S),
                "quantization_s": CAPTURE_FRAME_SECONDS,
                "max_lateness_s": max(
                    (row["lateness_s"] for row in self.seal_lateness),
                    default=0.0,
                ),
                "violations": int(self.seal_lateness_violations),
                "pairs": list(self.seal_lateness),
            },
            "dispatch": {
                "submitted_segments": len(self._submitted_segments),
                "terminal_segments": len(self._terminal_segments),
                "capture_generation": self._capture_generation,
            },
            "provider_fault": self.provider_fault,
            "meeting": meeting,
            "translation_outputs": list(self.translation_outputs),
            "phase": self.phase,
            "live_route": LIVE_ROUTE,
            "deepgram_reserve_usd": self.deepgram_reserve_usd,
            "translation_requests": list(self.translation_requests),
            "children": children_payload(
                self.children,
                outputs=self.translation_outputs,
                requests=self.translation_requests,
            ),
            **session,
        }
        payload["u8"] = u8_case_report(payload)
        payload["execution_completed"] = bool(payload["u8"]["execution_completed"])
        payload["evaluation_valid"] = bool(payload["u8"]["evaluation_valid"])
        payload["operational_clean"] = bool(payload["u8"]["operational_clean"])
        payload["decision"] = confirmatory_decision(
            cluster_rows=session["aggregate"]["cluster_rows"],
            coverage=session["aggregate"]["coverage"],
            safety_failures=payload["u8"]["safety_failures"],
            evaluation=payload["u8"],
        )
        payload["conditional_support"] = bool(payload["decision"].get("pass"))
        payload["incomplete"] = not payload["execution_completed"]
        payload["outage"] = self.provider_fault is not None
        payload["ok"] = payload["execution_completed"] and payload["evaluation_valid"]
        artifact = write_artifact(
            "last_live_run.json" if self.network else "last_lab_run.json",
            {
                "methods": payload["methods"],
                "open_session_calls": payload["open_session_calls"],
                "parents": session["parents"],
                "n_parents": session["n_parents"],
                "aggregate": session["aggregate"],
                "policy": session["policy"],
                "decision": payload["decision"],
                "latency_by_operation": session["latency_by_operation"],
                "receipts": receipts_payload,
                "requests": payload["translation_requests"],
                "translation_outputs": payload["translation_outputs"],
                "children": payload["children"],
                "vad_speech_chunks": self._speech_chunks,
                "vad_silence_chunks": self._silence_chunks,
                "c5_seal_reasons": payload["c5_seal_reasons"],
                "deepgram_reserve_usd": self.deepgram_reserve_usd,
                "network": self.network,
                "live_route": LIVE_ROUTE,
                "capture_timing": payload["capture_timing"],
                "dispatch": payload["dispatch"],
                "provider_fault": self.provider_fault,
                "seal_lateness": {
                    "rows": payload["seal_lateness"]["pairs"],
                    "violations": self.seal_lateness_violations,
                },
                "meeting": meeting,
                "sealed_segments": payload["sealed_segments"],
                "declared_source_samples": payload["declared_source_samples"],
                "execution_completed": payload["execution_completed"],
                "evaluation_valid": payload["evaluation_valid"],
                "operational_clean": payload["operational_clean"],
                "conditional_support": payload["conditional_support"],
                "u8": payload["u8"],
            },
            directory=self.artifact_dir,
        )
        payload["artifact"] = artifact
        self.reconcile_deepgram_budget(completed=clean_completion and not outage)
        payload["deepgram_reconciled"] = self.deepgram_reconciled
        payload["deepgram_settled_usd"] = self.deepgram_settled_usd
        payload["deepgram_reserved_usd"] = self.deepgram_reserve_usd
        return payload


def arrival_anchored(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """True when every receipt is stamped from the live stream, never hindsight."""
    if not receipts:
        return False
    last_applied = max(float(row["applied_at_monotonic_s"]) for row in receipts)
    return all(
        float(row["available_at_monotonic_s"])
        <= float(row["applied_at_monotonic_s"])
        <= last_applied
        for row in receipts
    )


def native_chunk_arrival_stats(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Quantization of the native producer's arrival stamps, including batched dumps."""
    arrivals = [
        float(row["available_at_monotonic_s"])
        for row in chunks
        if row.get("available_at_monotonic_s") is not None
    ]
    if not arrivals:
        return {
            "native_chunk_arrival_span_s": 0.0,
            "native_chunk_arrival_distinct_stamps": 0,
            "native_chunk_arrival_max_batch": 0,
            "native_chunk_arrival_batched": False,
        }
    batches: dict[float, int] = {}
    for stamp in arrivals:
        batches[stamp] = batches.get(stamp, 0) + 1
    return {
        "native_chunk_arrival_span_s": max(arrivals) - min(arrivals),
        "native_chunk_arrival_distinct_stamps": len(batches),
        "native_chunk_arrival_max_batch": max(batches.values()),
        "native_chunk_arrival_batched": len(batches) < len(arrivals),
    }


def intercept_covering_evidence(
    script: InterceptScript,
    *,
    capture_epoch: int,
    producer_generation: object,
    reference_generation: object,
    available_at_monotonic_s: float,
    offset_samples: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, word in enumerate(script.words):
        start = offset_samples + int(round(float(word.start) * HZ))
        end = offset_samples + int(round(float(word.end) * HZ))
        rows.append(
            {
                "capture_epoch": capture_epoch,
                "start_sample": start,
                "end_sample": end,
                "available_at_monotonic_s": available_at_monotonic_s,
                "relation": "CURRENT" if index == 0 else "OTHER",
                "producer_generation": producer_generation,
                "reference_generation": reference_generation,
                "reference_valid": True,
            }
        )
    return rows


def intercept_script(
    transcript: str,
    words: Sequence[tuple[str, float, float]],
    *,
    preroll_s: float = 0.0,
    translation: str = "안녕",
    failure: str | None = None,
) -> InterceptScript:
    return InterceptScript(
        transcript=transcript,
        words=tuple(
            InterceptWord(
                word=text,
                start=preroll_s + start,
                end=preroll_s + end,
                punctuated_word=(f"{text} " if index < len(words) - 1 else text),
            )
            for index, (text, start, end) in enumerate(words)
        ),
        translation=translation,
        failure=failure,
    )


def hello_there_script(*, preroll_s: float = 0.0) -> InterceptScript:
    return intercept_script(
        "Hello there",
        (("Hello", 0.0, 0.1), ("there", 0.1, 0.2)),
        preroll_s=preroll_s,
    )


def one_two_script(
    *,
    preroll_s: float = 0.0,
    failure: str | None = None,
) -> InterceptScript:
    return intercept_script(
        "One two",
        (("One", 0.0, 0.1), ("two", 0.1, 0.2)),
        preroll_s=preroll_s,
        failure=failure,
    )


def hello_there_pcm() -> np.ndarray:
    n = 10 * 512
    t = np.arange(n, dtype=np.float32) / float(HZ)
    return (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


async def run_intercepted_live() -> dict[str, Any]:
    runner = ContinuousC5LiveRunner(
        network=False,
        ownership_enabled=True,
        intercept=intercept_script(
            "Hello there",
            (("Hello", 0.0, 0.16), ("there", 0.16, 0.32)),
        ),
        secrets=load_runtime_secrets(),
    )
    samples = hello_there_pcm()
    await runner.open(audio_seconds=float(samples.size) / float(HZ))
    try:
        await runner.feed(samples)
        native = NativeSortformerProducer("unused.wav", clock=runner._clock.now)
        arrived = native.decoder.ingest_chunk(
            0,
            [[0.9, 0.0], [0.9, 0.0], [0.0, 0.9], [0.0, 0.9]],
            available_at_monotonic_s=runner._clock.now(),
            receipt_kind="recorded_native_probe",
        )
        for event in arrived:
            await runner.receive(
                hypothesis_from_live_event(
                    event,
                    capture_epoch=1,
                    producer_generation=runner._producer,
                    reference_generation=runner._reference,
                )
            )
        for item in native.drain_evidence():
            runner.apply_evidence(
                evidence_payload(
                    item,
                    capture_epoch=1,
                    producer_generation=runner._producer,
                    reference_generation=runner._reference,
                )
            )
        await runner.finalize()
        await runner.admit()
        await runner.translate()
        return await runner._session_payload(
            meeting=None,
            native_chunks=native.chunk_payloads(),
        )
    finally:
        await runner.close()


async def run_continuous_wav(
    wav_path: str | Path,
    *,
    network: bool,
    secrets: dict[str, str] | None = None,
    budget: BudgetLedger | None = None,
    intercept: InterceptScript | Sequence[InterceptScript] | None = None,
    sortformer: bool = False,
    meeting: str | None = None,
    phase: Phase = "dev",
    pace: bool | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    samples = load_wav_16k(wav_path)
    runner = ContinuousC5LiveRunner(
        network=network,
        ownership_enabled=True,
        intercept=intercept,
        secrets=secrets or load_runtime_secrets(),
        budget=budget,
        phase=phase,
        use_silero=intercept is None,
        artifact_dir=artifact_dir,
    )
    producer = None
    if sortformer:
        producer = NativeSortformerProducer(wav_path, clock=runner._clock.now)
        if not producer.available:
            raise FileNotFoundError("native Sortformer producer binary or model is missing")
        producer.start()
    try:
        audio_seconds = float(samples.size) / float(HZ)
        scripts = intercept_scripts(intercept)
        intercept_delivered: set[int] = set()
        await runner.open(audio_seconds=audio_seconds)
        chunk = 512
        offset = 0
        paced = bool(network or sortformer) if pace is None else bool(pace)
        loop = asyncio.get_running_loop()
        started = loop.time()
        while offset < samples.size:
            end = min(offset + chunk, samples.size)
            await runner.feed(samples[offset:end])
            offset = end
            if paced:
                await asyncio.sleep(max(offset / float(HZ) - (loop.time() - started), 0.0))
            if scripts:
                await runner.deliver_intercept_scripts(
                    scripts,
                    delivered=intercept_delivered,
                )
            if producer is not None:
                for event in producer.poll():
                    await runner.receive(
                        hypothesis_at_boundary(
                            event.boundary,
                            capture_epoch=1,
                            available_at_monotonic_s=event.available_at_monotonic_s,
                            producer_generation=runner._producer,
                            reference_generation=runner._reference,
                            hypothesis_id=event.event_id,
                            local_slot=event.candidate_slot,
                        )
                    )
                for item in producer.drain_evidence():
                    runner.apply_evidence(
                        evidence_payload(
                            item,
                            capture_epoch=1,
                            producer_generation=runner._producer,
                            reference_generation=runner._reference,
                        )
                    )
        await runner.finalize()
        await runner.admit()
        await runner.translate()
        payload = await runner._session_payload(
            meeting=meeting,
            native_chunks=producer.chunk_payloads() if producer is not None else (),
        )
        return {
            "ok": payload["ok"],
            "completed": payload["execution_completed"],
            "network": network,
            "meeting": meeting,
            "phase": payload["phase"],
            "wav_path": str(wav_path),
            "n_parents": payload["n_parents"],
            "n_children": len(payload["children"]),
            "methods": payload["methods"],
            "open_session_calls": payload["open_session_calls"],
            "parents": payload["parents"],
            "aggregate": payload["aggregate"],
            "policy": payload["policy"],
            "decision": payload["decision"],
            "latency_by_operation": payload["latency_by_operation"],
            "r0": payload["r0"],
            "r2": payload["r2"],
            "r1": payload["r1"],
            "control": payload["control"],
            "native_chunks": producer.chunk_payloads() if producer is not None else [],
            "receipts": payload["receipts"],
            "c5_seal_reasons": payload["c5_seal_reasons"],
            "capture_timing": payload["capture_timing"],
            "seal_lateness": payload["seal_lateness"],
            "dispatch": payload["dispatch"],
            "provider_fault": payload["provider_fault"],
            "evidence": [sanitize_evidence(row) for row in runner._evidence],
            "translation_requests": payload["translation_requests"],
            "children": payload["children"],
            "live_route": LIVE_ROUTE,
            "deepgram_reserve_usd": runner.deepgram_reserve_usd,
            "deepgram_reconciled": payload["deepgram_reconciled"],
            "deepgram_settled_usd": payload["deepgram_settled_usd"],
            "clean_completion": payload["clean_completion"],
            "incomplete": payload["incomplete"],
            "outage": payload["outage"],
            "degraded": payload["degraded"],
            "execution_completed": payload["execution_completed"],
            "evaluation_valid": payload["evaluation_valid"],
            "operational_clean": payload["operational_clean"],
            "conditional_support": payload["conditional_support"],
            "sealed_segments": payload["sealed_segments"],
            "declared_source_samples": payload["declared_source_samples"],
            "timing_failures": payload["timing_failures"],
            "task_failures": payload["task_failures"],
            "u8": payload["u8"],
            "executor": "run_continuous_wav",
            "intercept": bool(scripts),
            "paced": paced,
            "artifact": payload["artifact"],
        }
    finally:
        if producer is not None:
            producer.close()
        await runner.close()


def write_pcm_wav(path: str | Path, samples: np.ndarray, *, hz: int = HZ) -> Path:
    target = Path(path)
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(hz)
        handle.writeframes(pcm.tobytes())
    return target
