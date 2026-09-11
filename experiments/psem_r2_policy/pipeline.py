from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.psem_r2_policy.budget import LEDGER_PATH, BudgetLedger, load_billing_bounds
from experiments.psem_r2_policy.live_runner import (
    LIVE_ROUTE,
    ami_wav_path,
    run_continuous_wav,
    run_intercepted_live,
)
from experiments.psem_r2_policy.phase import holdout_unlock_error, resolve_meetings
from experiments.psem_r2_policy.secrets import credential_presence, load_runtime_secrets
from experiments.psem_r2_policy.sortformer_live import hypothesis_at_boundary
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.audio.pretranslation_ownership import PretranslationOwnershipOwner
from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfigSnapshot,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationTurnChild,
    TranslationTurnLifecycleOwner,
    TranslationTurnProcessResult,
    TranslationTurnRequest,
)
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTTimedToken,
)
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.domain.models import FinalLanguageRun, Transcript
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend


def hypotheses_from_boundaries(
    boundaries: list[int], *, capture_epoch: int = 1
) -> tuple[ProspectiveSpeakerHypothesis, ...]:
    producer = object()
    reference = object()
    events = []
    for index, boundary in enumerate(boundaries, start=1):
        events.append(
            hypothesis_at_boundary(
                boundary,
                capture_epoch=capture_epoch,
                available_at_monotonic_s=1.0,
                producer_generation=producer,
                reference_generation=reference,
                hypothesis_id=f"h{index}",
            )
        )
    return tuple(events)


def make_terminal(tokens: tuple[STTTimedToken, ...]) -> STTProviderTurnTerminal:
    text = "".join(token.text for token in tokens)
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(
            activation_generation=1,
            segment_order=1,
            segment_id=uuid4(),
            capture_epoch=1,
        ),
        provider_epoch_id="epoch",
        provider_turn_id="turn",
    )
    raw = STTProviderTurnTerminal(
        identity=identity,
        outcome="final",
        text=text,
        final_language_runs=(FinalLanguageRun(text, "en"),),
        text_authority="authoritative",
        timed_tokens=tokens,
    )
    return STTScopedTurnNormalizer(identity).apply_terminal(raw)


async def admit_units(
    terminal: STTProviderTurnTerminal,
    *,
    enabled: bool,
    events: tuple[ProspectiveSpeakerHypothesis, ...] = (),
    evidence: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    from experiments.psem_r2_policy.arms import apply_observe_evidence, r2_translation_config

    owner = PretranslationOwnershipOwner(enabled=enabled)
    for event in events:
        owner.observe(event)
    for item in evidence:
        apply_observe_evidence(owner, item)
    created: list[TranslationTurnChild] = []

    async def process(child: TranslationTurnChild, _cancel):
        created.append(child)
        return TranslationTurnProcessResult("source_only")

    async def noop(*_args, **_kwargs):
        return None

    turns = TranslationTurnLifecycleOwner(
        on_child_created=noop,
        on_child_started=noop,
        process_child=process,
        on_child_terminal=noop,
        on_parent_closed=noop,
        on_parent_rejected=noop,
    )
    assignment = owner.assign(
        parent_utterance_id=terminal.identity.segment.segment_id,
        timed_tokens=terminal.timed_tokens,
        capture_epoch=terminal.identity.segment.capture_epoch,
        admitted_at_monotonic_s=2.0,
        parent_text=terminal.text,
    )
    units = (
        assignment.units if assignment.disposition == "assigned" and assignment.conserved else ()
    )
    text = terminal.text
    runs = terminal.final_language_runs or (FinalLanguageRun(text, "en"),)
    request = TranslationTurnRequest(
        transcript=Transcript(
            utterance_id=terminal.identity.segment.segment_id,
            text=text,
            is_final=True,
            channel="peer",
            final_language_runs=runs,
            publication_generation=1,
            source_order=1,
        ),
        source="Peer",
        turn_kind="peer",
        target_languages=("ko",),
        config_snapshot=TranslationRuntimeConfigSnapshot(revision=0, value=r2_translation_config()),
        ownership_units=units,
    )
    await turns.open_channel_ingress("peer")
    await turns.start()
    child_ids = await turns.submit(request)
    await turns.wait_for_idle()
    await turns.close()
    return {
        "parent": str(terminal.identity.segment.segment_id),
        "assignment": assignment.disposition,
        "conserved": assignment.conserved,
        "n_units": len(units),
        "group_ids": [unit.group_id for unit in units],
        "child_ids": [str(item) for item in child_ids],
        "child_groups": [child.ownership_group_id for child in created],
        "child_texts": [child.transcript.text for child in created],
        "reconstructed": "".join(unit.text for unit in units),
        "timed_timings": [token.timing for token in terminal.timed_tokens],
        "start_ms_present": sum(token.start_ms is not None for token in terminal.timed_tokens),
        "n_timed": len(terminal.timed_tokens),
    }


async def run_synthetic_path() -> dict[str, Any]:
    tokens = (
        STTTimedToken(
            text="Hello ",
            language="en",
            start_ms=0,
            end_ms=100,
            timing="interval",
            source_start_sample=0,
            source_end_sample=1600,
        ),
        STTTimedToken(
            text="there",
            language="en",
            start_ms=100,
            end_ms=200,
            timing="interval",
            source_start_sample=1600,
            source_end_sample=3200,
        ),
    )
    terminal = make_terminal(tokens)
    events = hypotheses_from_boundaries([1600])
    producer = events[0].producer_generation
    reference = events[0].reference_generation
    evidence = (
        {
            "capture_epoch": 1,
            "start_sample": 0,
            "end_sample": 1600,
            "available_at_monotonic_s": 1.0,
            "relation": "CURRENT",
            "producer_generation": producer,
            "reference_generation": reference,
            "reference_valid": True,
        },
        {
            "capture_epoch": 1,
            "start_sample": 1600,
            "end_sample": 3200,
            "available_at_monotonic_s": 1.0,
            "relation": "OTHER",
            "producer_generation": producer,
            "reference_generation": reference,
            "reference_valid": True,
        },
    )
    enabled = await admit_units(terminal, enabled=True, events=events, evidence=evidence)
    disabled = await admit_units(terminal, enabled=False, events=events, evidence=evidence)
    return {
        "enabled": enabled,
        "disabled": disabled,
        "text": terminal.text,
        "n_timed": len(terminal.timed_tokens),
        "path": "offline_replay_generic_timed_tokens->scoped_normalizer->pretranslation_owner->translation_turn_children",
        "network": False,
        "live_route": LIVE_ROUTE,
    }


async def run_intercepted_deepgram_path() -> dict[str, Any]:
    return await run_intercepted_live()


def build_paid_stt_backend(secrets: dict[str, str] | None = None) -> DeepgramRealtimeSTTBackend:
    payload = secrets if secrets is not None else load_runtime_secrets()
    return DeepgramRealtimeSTTBackend(
        api_key=payload.get("DEEPGRAM_API_KEY") or "",
        language="en",
        model="nova-3",
        keyterms=(),
    )


def paid_refusal(
    reason: str,
    *,
    keys: dict[str, bool] | None = None,
    phase: str | None = None,
    meeting: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "completed": False,
        "refused": True,
        "paid_blocked": True,
        "network": False,
        "runner_called": False,
        "reason": reason,
        "phase": phase,
        "meeting": meeting,
        "credentials_present": keys or credential_presence(),
        "paid_executor": "run_continuous_wav",
    }


def refuse_paid_if_disabled(
    *,
    phase: str | None,
    meeting: str | None,
    wav_path: str | None,
) -> dict[str, Any] | None:
    bounds = load_billing_bounds()
    keys = credential_presence()
    if not bounds.get("paid_ready"):
        return paid_refusal(
            "paid_ready is false; Director billing/protocol barrier still closed",
            keys=keys,
            phase=phase,
            meeting=meeting,
        )
    if phase not in {"dev", "holdout"} or not meeting:
        return paid_refusal(
            "--paid/--phase requires a declared --phase and --meeting",
            keys=keys,
            phase=phase,
            meeting=meeting,
        )
    try:
        resolve_meetings(phase, meeting)
    except ValueError as exc:
        return paid_refusal(str(exc), keys=keys, phase=phase, meeting=meeting)
    if phase == "holdout":
        locked = holdout_unlock_error()
        if locked is not None:
            return paid_refusal(locked, keys=keys, phase=phase, meeting=meeting)
    try:
        declared = ami_wav_path(meeting)
    except FileNotFoundError as exc:
        return paid_refusal(str(exc), keys=keys, phase=phase, meeting=meeting)
    if wav_path is not None:
        given = Path(wav_path).resolve()
        if given != declared.resolve():
            return paid_refusal(
                "arbitrary wav cannot bypass declared phase meeting audio",
                keys=keys,
                phase=phase,
                meeting=meeting,
            )
    if not (keys.get("DEEPGRAM_API_KEY") and keys.get("OPENROUTER_API_KEY")):
        return paid_refusal("required API keys are absent", keys=keys, phase=phase, meeting=meeting)
    return None


async def run_paid_live(
    wav_path: str | None = None,
    *,
    budget: Any | None = None,
    phase: str | None = None,
    meeting: str | None = None,
) -> dict[str, Any]:
    refused = refuse_paid_if_disabled(phase=phase, meeting=meeting, wav_path=wav_path)
    if refused is not None:
        return refused
    assert meeting is not None
    assert phase in {"dev", "holdout"}
    secrets = load_runtime_secrets()
    keys = credential_presence(secrets)
    backend = build_paid_stt_backend(secrets)
    declared = ami_wav_path(meeting)
    ledger = budget if budget is not None else BudgetLedger(LEDGER_PATH)
    live = await run_continuous_wav(
        declared,
        network=True,
        secrets=secrets,
        budget=ledger,
        intercept=None,
        sortformer=True,
        meeting=meeting,
    )
    live["credentials_present"] = keys
    live["backend"] = type(backend).__name__
    live["model"] = backend.model
    live["keyterms"] = list(backend.keyterms)
    live["paid_executor"] = "run_continuous_wav"
    live["executor"] = "run_continuous_wav"
    live["live_methods"] = live.get("methods")
    live["budget_defensible"] = True
    live["paid_blocked"] = False
    live["refused"] = False
    live["runner_called"] = True
    live["completed"] = bool(live.get("completed"))
    live["phase"] = phase
    live["meeting"] = meeting
    live["ledger_path"] = str(getattr(ledger, "path", LEDGER_PATH))
    return live
