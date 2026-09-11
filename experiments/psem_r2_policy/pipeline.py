from __future__ import annotations

from typing import Any
from uuid import uuid4

from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.audio.pretranslation_ownership import PretranslationOwnershipOwner
from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
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

from experiments.psem_r2_policy.budget import load_billing_bounds
from experiments.psem_r2_policy.live_runner import (
    LIVE_ROUTE,
    ContinuousC5LiveRunner,
    hello_there_pcm,
    hello_there_script,
    run_continuous_wav,
    run_intercepted_live,
)
from experiments.psem_r2_policy.secrets import credential_presence, load_runtime_secrets
from experiments.psem_r2_policy.sortformer_live import hypothesis_at_boundary


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
) -> dict[str, Any]:
    owner = PretranslationOwnershipOwner(enabled=enabled)
    for event in events:
        owner.observe(event)
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
    )
    units = assignment.units if assignment.disposition == "assigned" and assignment.conserved else ()
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
        config_snapshot=TranslationRuntimeConfigSnapshot(
            revision=0, value=TranslationRuntimeConfig()
        ),
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
    enabled = await admit_units(terminal, enabled=True, events=events)
    disabled = await admit_units(terminal, enabled=False, events=events)
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


async def run_paid_live(wav_path: str | None = None) -> dict[str, Any]:
    bounds = load_billing_bounds()
    secrets = load_runtime_secrets()
    keys = credential_presence(secrets)
    backend = build_paid_stt_backend(secrets)
    runner = ContinuousC5LiveRunner(
        network=False,
        ownership_enabled=True,
        intercept=hello_there_script(),
        secrets=secrets,
    )
    exercised = await runner.run_pcm(hello_there_pcm(), boundary=1600)
    payload = {
        "ok": False,
        "network": False,
        "paid_blocked": not bool(bounds.get("paid_ready")),
        "reason": "paid_ready is false; Director billing/protocol barrier still closed",
        "credentials_present": keys,
        "backend": type(backend).__name__,
        "model": backend.model,
        "keyterms": list(backend.keyterms),
        "would_call": "DeepgramRealtimeSTTBackend.open_session",
        "budget_defensible": bool(bounds.get("budget_defensible")),
        "live_methods": exercised.get("methods"),
        "open_session_calls": exercised.get("open_session_calls"),
        "paid_executor": "run_continuous_wav",
        "path": exercised.get("path"),
        "live_route": LIVE_ROUTE,
        "deepgram_reserve_usd": exercised.get("deepgram_reserve_usd"),
        "intercept_ok": bool(exercised.get("ok")),
    }
    if not bounds.get("paid_ready"):
        return payload
    if not keys["DEEPGRAM_API_KEY"] or not keys["OPENROUTER_API_KEY"]:
        payload["reason"] = "required API keys are absent"
        payload["ok"] = False
        return payload
    if wav_path is None:
        payload["reason"] = "paid_ready true requires an explicit Director DEV wav path"
        payload["ok"] = False
        return payload
    live = await run_continuous_wav(
        wav_path,
        network=True,
        secrets=secrets,
        intercept=None,
        sortformer=True,
    )
    live["credentials_present"] = keys
    live["backend"] = type(backend).__name__
    return live
