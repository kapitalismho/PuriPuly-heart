from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.arms import (
    evaluate_protocol_arms,
    r2_rendered_system_prompt,
    r2_translation_config,
)
from experiments.psem_r2_policy.budget import BudgetLedger, deepgram_reserve_usd
from experiments.psem_r2_policy.live_runner import (
    PINNED_TRANSLATION,
    PREROLL_SECONDS,
    BudgetedOpenRouter,
    ContinuousC5LiveRunner,
    EnergyVadEngine,
    InterceptOpenRouterClient,
    compose_r2_harness,
    hello_there_pcm,
    hello_there_script,
    install_deepgram_intercept,
    one_two_script,
    run_continuous_wav,
    run_intercepted_live,
    write_pcm_wav,
)
from experiments.psem_r2_policy.pipeline import run_paid_live
from experiments.psem_r2_policy.secrets import ORIGINAL_ENV_LOCAL, credential_presence
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    AudioSegmentSnapshot,
    AudioSegmentTerminalReceipt,
)
from puripuly_heart.core.audio.pretranslation_ownership import PretranslationOwnershipOwner
from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis
from puripuly_heart.core.orchestrator.configuration import TranslationRuntimeConfig
from puripuly_heart.core.orchestrator.translation_turn import TranslationTurnChild
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTSessionProjection,
    STTTimedToken,
)
from puripuly_heart.domain.models import FinalLanguageRun, Translation
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from tests.helpers.translation_owners import TranslationOwnersTestHarness


@pytest.mark.asyncio
async def test_intercepted_live_runner_uses_open_feed_receive_finalize_admit_translate() -> None:
    result = await run_intercepted_live()
    assert result["network"] is False
    assert result["open_session_calls"] >= 1
    assert result["methods"][:6] == [
        "open",
        "feed",
        "receive",
        "finalize",
        "admit",
        "translate",
    ]
    assert result["n_parents"] == 1
    assert result["incomplete"] is False
    parent = result["parents"][0]
    assert parent["text"] == "Hello there"
    assert parent["n_timed"] == 2
    assert parent["timed_start_ms"] == [0, 100]
    assert parent["timed_timings"] == ["interval", "interval"]
    assert parent["span"] == [0, 3200]
    assert parent["receipt"]["outcome"] == "final"
    enabled = result["enabled"]
    assert enabled["conserved"] is True
    assert enabled["group_ids"] == ["CURRENT-0", "OTHER-1"]
    assert enabled["child_groups"] == ["CURRENT-0", "OTHER-1"]
    assert enabled["child_texts"] == ["Hello ", "there"]
    assert enabled["reconstructed"] == "Hello there"
    disabled = result["disabled"]
    assert disabled["child_groups"] == [""]
    assert disabled["child_texts"] == ["Hello there"]
    assert result["vad_speech_chunks"] > 0
    assert result["c5_seal_reasons"]
    assert "always" not in "".join(result["c5_seal_reasons"])
    assert parent["tokens"]
    assert parent["r2"]["conservation"]["missing_token_ids"] == []
    receipt = result["receipts"][0]
    assert receipt["receipt_kind"] == "native_arrival"
    assert receipt["available_at_monotonic_s"] <= receipt["applied_at_monotonic_s"]


@pytest.mark.asyncio
async def test_paid_gate_rejects_before_network_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[object] = []

    async def fake_runner(*_args: object, **_kwargs: object) -> dict:
        called.append(True)
        return {"ok": True, "completed": True}

    monkeypatch.setattr("experiments.psem_r2_policy.pipeline.run_continuous_wav", fake_runner)
    payload = await run_paid_live()
    assert payload["ok"] is False
    assert payload["refused"] is True
    assert payload["paid_blocked"] is True
    assert payload["network"] is False
    assert payload["runner_called"] is False
    assert called == []
    assert "open_session_calls" not in payload
    assert "methods" not in payload
    presence = credential_presence()
    assert set(presence) == {"DEEPGRAM_API_KEY", "OPENROUTER_API_KEY"}
    assert payload["credentials_present"] == presence
    assert ORIGINAL_ENV_LOCAL.name == ".env.local"


@pytest.mark.asyncio
async def test_enabled_paid_handler_calls_real_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "ES2009a.Mix-Headset.wav"
    wav.write_bytes(b"fake")
    captured: dict[str, object] = {}

    async def fake_runner(path: object, **kwargs: object) -> dict:
        captured["path"] = path
        captured["kwargs"] = kwargs
        return {
            "ok": True,
            "completed": True,
            "network": True,
            "methods": ["open", "feed", "finalize"],
            "executor": "run_continuous_wav",
        }

    monkeypatch.setattr(
        "experiments.psem_r2_policy.pipeline.load_billing_bounds",
        lambda: {"paid_ready": True, "budget_defensible": True},
    )
    monkeypatch.setattr("experiments.psem_r2_policy.pipeline.ami_wav_path", lambda meeting: wav)
    monkeypatch.setattr("experiments.psem_r2_policy.pipeline.run_continuous_wav", fake_runner)
    payload = await run_paid_live(phase="dev", meeting="ES2009a")
    assert payload["ok"] is True
    assert payload["completed"] is True
    assert payload["runner_called"] is True
    assert captured["path"] == wav
    kwargs = captured["kwargs"]
    assert kwargs["network"] is True
    assert kwargs["intercept"] is None
    assert kwargs["sortformer"] is True
    assert kwargs["meeting"] == "ES2009a"
    assert kwargs["budget"] is not None


@pytest.mark.asyncio
async def test_network_runner_reserves_before_sdk_open_and_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[object] = []
    ledger = BudgetLedger(tmp_path / "budget.json")
    real_reserve = ledger.reserve

    def tracked_reserve(
        request_id: str,
        *,
        phase: str,
        amount_usd: float,
        meta: dict | None = None,
    ) -> dict:
        order.append(("reserve", (meta or {}).get("kind")))
        return real_reserve(request_id, phase=phase, amount_usd=amount_usd, meta=meta)

    monkeypatch.setattr(ledger, "reserve", tracked_reserve)
    original_open = DeepgramRealtimeSTTBackend.open_session

    async def wrapped(
        self: DeepgramRealtimeSTTBackend,
        *,
        projection: STTSessionProjection = STTSessionProjection(),
    ):
        order.append("open_session")
        return await original_open(self, projection=projection)

    monkeypatch.setattr(DeepgramRealtimeSTTBackend, "open_session", wrapped)

    async def fake_translate(self: object, **kwargs: object) -> Translation:
        order.append(("llm", kwargs.get("text")))
        return Translation(uuid4(), text="안녕")

    monkeypatch.setattr(OpenRouterLLMProvider, "translate", fake_translate)
    with install_deepgram_intercept(hello_there_script()):
        runner = ContinuousC5LiveRunner(
            network=True,
            intercept=None,
            budget=ledger,
            use_silero=False,
            vad_engine=EnergyVadEngine(),
            secrets={"DEEPGRAM_API_KEY": "k", "OPENROUTER_API_KEY": "k"},
        )
        await runner.run_pcm(hello_there_pcm(), boundary=1600)
    assert order[0] == ("reserve", "deepgram")
    assert "open_session" in order
    assert order.index(("reserve", "deepgram")) < order.index("open_session")
    llm_reserves = [item for item in order if item == ("reserve", "openrouter")]
    llm_calls = [item for item in order if isinstance(item, tuple) and item[0] == "llm"]
    assert llm_reserves
    assert llm_calls
    assert order.index(("reserve", "openrouter")) < min(
        i for i, item in enumerate(order) if isinstance(item, tuple) and item[0] == "llm"
    )
    snap = ledger.snapshot()
    deepgram = [entry for entry in snap.entries if entry["meta"].get("kind") == "deepgram"]
    assert len(deepgram) == 1
    assert deepgram[0]["state"] == "settled"
    assert snap.spent_usd == pytest.approx(0.0)
    assert snap.credit_usd >= deepgram[0]["settled_usd"] - 1e-12
    duplicate = deepgram_reserve_usd(
        max_audio_seconds=0.224,
        hangover_seconds=0.8,
        preroll_seconds=0.5,
        tail_seconds=512.0 / 16000.0,
        copies=1,
        reconnect_bound=1,
    )
    assert deepgram[0]["reserved_usd"] < duplicate
    assert snap.credit_usd > 0


@pytest.mark.asyncio
async def test_live_runner_disabled_arm_keeps_one_unsplit_parent() -> None:
    runner = ContinuousC5LiveRunner(
        network=False,
        ownership_enabled=False,
        intercept=hello_there_script(),
    )
    result = await runner.run_pcm(hello_there_pcm(), boundary=1600)
    assert result["enabled"]["group_ids"] == [""]
    assert result["enabled"]["conserved"] is True
    assert result["disabled"]["disposition"] == "disabled"
    assert result["disabled"]["n_units"] == 0
    assert result["disabled"]["child_groups"] == [""]
    assert len(result["disabled"]["child_ids"]) == 1
    assert result["open_session_calls"] >= 1


@pytest.mark.asyncio
async def test_continuous_wav_runs_consecutive_parents_without_state_reset(
    tmp_path: Path,
) -> None:
    first = hello_there_script()
    second = one_two_script(preroll_s=PREROLL_SECONDS)
    silence = np.zeros((80000,), dtype=np.float32)
    wav = write_pcm_wav(
        tmp_path / "two_parents.wav",
        np.concatenate([hello_there_pcm(), silence, hello_there_pcm(), silence]),
    )
    with install_deepgram_intercept((first, second)):
        result = await run_continuous_wav(
            wav,
            network=False,
            secrets={},
            intercept=(first, second),
            meeting=None,
        )
    assert result["ok"] is True
    assert result["completed"] is True
    assert result["n_parents"] == 2
    assert result["open_session_calls"] == 2
    methods = result["methods"]
    for earlier, later in (
        ("open", "feed"),
        ("feed", "receive"),
        ("receive", "finalize"),
        ("finalize", "admit"),
        ("admit", "translate"),
    ):
        assert methods.index(earlier) < methods.index(later)
    parents = result["parents"]
    assert [parent["text"] for parent in parents] == ["Hello there", "One two"]
    assert [parent["outcome"] for parent in parents] == ["final", "final"]
    assert [parent["incomplete"] for parent in parents] == [False, False]
    assert parents[1]["span"][0] > parents[0]["span"][1]
    for parent in parents:
        assert parent["conserved"] is True
        assert len(parent["group_ids"]) == 2
        assert parent["group_ids"] == parent["r2"]["child_groups"]
    assert result["receipts"][0]["requested_transition_sample"] == parents[0]["span"][1] - 1600
    assert result["receipts"][1]["requested_transition_sample"] == parents[1]["span"][1] - 1600
    assert [row["disposition"] for row in result["receipts"]] == ["assigned", "assigned"]
    assert result["r2"]["child_texts"] == ["Hello ", "there", "One ", "two"]
    assert result["r0"]["child_texts"] == ["Hello there", "One two"]
    assert result["r2"]["child_groups"] == [
        "CURRENT-0",
        "OTHER-1",
        "CURRENT-0",
        "OTHER-1",
    ]
    assert (
        result["children"][0]["parent_utterance_id"] != result["children"][2]["parent_utterance_id"]
    )
    artifact = json.loads(Path(result["artifact"]["path"]).read_text(encoding="utf-8"))
    assert artifact["n_parents"] == 2
    assert [row["text"] for row in artifact["parents"]] == ["Hello there", "One two"]


@pytest.mark.asyncio
async def test_every_fresh_session_reserves_a_pad_before_open_and_keeps_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend

    healthy = hello_there_script()
    failed = one_two_script(preroll_s=PREROLL_SECONDS, failure="failed")
    silence = np.zeros((40000,), dtype=np.float32)
    wav = write_pcm_wav(
        tmp_path / "two_sessions.wav",
        np.concatenate([hello_there_pcm(), silence, hello_there_pcm(), silence]),
    )
    ledger = BudgetLedger(tmp_path / "budget.json")
    pads_seen_at_open: list[int] = []
    real_open = DeepgramRealtimeSTTBackend.open_session

    async def spy(self: object, *, projection: object = None) -> object:
        pads_seen_at_open.append(
            sum(
                1
                for entry in ledger.snapshot().entries
                if entry["meta"].get("kind") == "deepgram-session-pad"
            )
        )
        return await real_open(self, projection=projection)

    monkeypatch.setattr(DeepgramRealtimeSTTBackend, "open_session", spy)
    with install_deepgram_intercept((healthy, failed)):
        result = await run_continuous_wav(
            wav,
            network=True,
            secrets={"DEEPGRAM_API_KEY": "k", "OPENROUTER_API_KEY": "k"},
            intercept=(healthy, failed),
            meeting=None,
            budget=ledger,
            pace=False,
        )
    pads = [
        entry
        for entry in ledger.snapshot().entries
        if entry["meta"].get("kind") == "deepgram-session-pad"
    ]
    assert result["open_session_calls"] == len(pads) >= 2
    assert pads_seen_at_open == list(range(1, len(pads) + 1))
    assert [entry["state"] for entry in pads] == ["reserved"] * len(pads)
    for entry in pads:
        assert entry["meta"]["billable_seconds"] == 2.0
        assert entry["reserved_usd"] == pytest.approx(2 / 60.0 * 0.0077)
    statuses = [parent["status"] for parent in result["parents"]]
    assert statuses[:2] == ["complete", "unsuccessful"]
    assert result["deepgram_reconciled"] is False


@pytest.mark.asyncio
async def test_unsuccessful_and_degraded_parents_leave_the_eligible_pool(
    tmp_path: Path,
) -> None:
    healthy = hello_there_script()
    failed = one_two_script(preroll_s=PREROLL_SECONDS, failure="failed")
    degraded = one_two_script(preroll_s=PREROLL_SECONDS, failure="degraded")
    silence = np.zeros((40000,), dtype=np.float32)
    wav = write_pcm_wav(
        tmp_path / "mixed.wav",
        np.concatenate(
            [
                hello_there_pcm(),
                silence,
                hello_there_pcm(),
                silence,
                hello_there_pcm(),
                silence,
            ]
        ),
    )
    with install_deepgram_intercept((healthy, failed, degraded)):
        result = await run_continuous_wav(
            wav,
            network=False,
            secrets={},
            intercept=(healthy, failed, degraded),
            meeting=None,
        )
    parents = result["parents"]
    assert [parent["status"] for parent in parents] == [
        "complete",
        "unsuccessful",
        "degraded",
    ]
    assert parents[0]["conserved"] is True
    assert parents[0]["group_ids"] == ["CURRENT-0", "OTHER-1"]
    assert len(parents[0]["children"]) == 2
    assert parents[1]["text"] == ""
    assert parents[1]["text_authority"] == "none"
    assert parents[1]["failure_reason"] == "deepgram_transport_error"
    assert parents[1]["outage"] is True
    assert parents[1]["incomplete"] is True
    assert parents[2]["text_authority"] == "degraded"
    assert parents[2]["failure_reason"] == "deepgram_transport_error"
    assert parents[2]["conserved"] is True
    assert parents[2]["incomplete"] is False
    aggregate = result["aggregate"]
    assert aggregate["n_incomplete_preserved"] == 1
    assert aggregate["n_degraded_conditional"] == 1
    assert [row["status"] for row in aggregate["unsuccessful_parents"]] == ["unsuccessful"]
    assert [row["status"] for row in aggregate["degraded_parents"]] == ["degraded"]
    assert aggregate["degraded_parents"][0]["conditional_ownership"] is True
    assert aggregate["coverage"]["coverage_integrity"] is False
    assert result["decision"]["coverage_integrity"] is False
    assert result["decision"]["pass"] is False
    assert result["clean_completion"] is False
    assert result["degraded"] is True
    assert result["incomplete"] is True
    assert result["ok"] is False


@pytest.mark.asyncio
async def test_paced_run_feeds_audio_at_source_rate(tmp_path: Path) -> None:
    first = hello_there_script()
    silence = np.zeros((16000,), dtype=np.float32)
    samples = np.concatenate([hello_there_pcm(), silence, hello_there_pcm(), silence])
    wav = write_pcm_wav(tmp_path / "paced.wav", samples)
    started = time.monotonic()
    with install_deepgram_intercept(first):
        result = await run_continuous_wav(
            wav,
            network=False,
            secrets={},
            intercept=first,
            meeting=None,
            pace=True,
        )
    elapsed = time.monotonic() - started
    audio_seconds = samples.size / 16000.0
    assert result["paced"] is True
    assert result["n_parents"] >= 1
    assert elapsed >= audio_seconds - 0.5


def _assert_capture_ledger_balanced(capture: dict) -> None:
    consumed = (
        capture["chunked_source_samples"]
        + capture["unprocessed_source_samples"]
        + capture["buffered_source_samples"]
        + capture["dropped_tail_source_samples"]
    )
    supplied = capture["fed_source_samples"] + capture["flush_pad_source_samples"]
    assert consumed == supplied


def _burst_meeting(count: int, *, silence_samples: int = 16000) -> np.ndarray:
    silence = np.zeros((silence_samples,), dtype=np.float32)
    return np.concatenate([np.concatenate([hello_there_pcm(), silence]) for _ in range(count)])


def _sustained_speech_pcm(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000.0
    return (0.25 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float32)


@pytest.mark.asyncio
async def test_paced_capture_holds_source_time_while_provider_handshakes_drag(
    tmp_path: Path,
) -> None:
    count = 4
    scripts = tuple(replace(hello_there_script(), session_open_delay_s=1.0) for _ in range(count))
    samples = _burst_meeting(count)
    wav = write_pcm_wav(tmp_path / "slow_handshake.wav", samples)
    started = time.monotonic()
    with install_deepgram_intercept(scripts):
        result = await run_continuous_wav(
            wav,
            network=False,
            secrets={},
            intercept=scripts,
            meeting=None,
            pace=True,
        )
    elapsed = time.monotonic() - started
    audio_seconds = samples.size / 16000.0
    capture = result["capture_timing"]
    _assert_capture_ledger_balanced(capture)
    assert capture["fed_source_samples"] == samples.size
    assert capture["unprocessed_source_samples"] == 0
    assert capture["last_arrival_monotonic_s"] - started <= audio_seconds + 0.5
    assert capture["last_arrival_monotonic_s"] - started >= audio_seconds - 0.5
    dispatch = result["dispatch"]
    assert dispatch["submitted_segments"] == count
    assert dispatch["terminal_segments"] == count
    assert result["provider_fault"] is None
    assert result["n_parents"] == count
    assert result["seal_lateness"]["violations"] == 0
    assert elapsed < audio_seconds + 2.0


@pytest.mark.asyncio
async def test_transport_failure_on_first_segment_leaves_capture_flowing() -> None:
    background_errors: list[str] = []

    def record_background(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        _ = loop
        exception = context.get("exception")
        background_errors.append(str(exception) if exception else str(context.get("message")))

    asyncio.get_running_loop().set_exception_handler(record_background)
    count = 4
    scripts = (replace(hello_there_script(), failure="failed"),) + (hello_there_script(),) * (
        count - 1
    )
    samples = _burst_meeting(count, silence_samples=16000)
    runner = ContinuousC5LiveRunner(
        network=False,
        ownership_enabled=True,
        intercept=scripts,
    )
    delivered: set[int] = set()
    with install_deepgram_intercept(scripts):
        await runner.open(audio_seconds=float(samples.size) / 16000.0)
        loop = asyncio.get_running_loop()
        started = loop.time()
        offset = 0
        while offset < samples.size:
            end = min(offset + 512, samples.size)
            await runner.feed(samples[offset:end])
            offset = end
            await asyncio.sleep(max(offset / 16000.0 - (loop.time() - started), 0.0))
            await runner.deliver_intercept_scripts(scripts, delivered=delivered)
        await runner.finalize()
        await runner.admit()
        await runner.translate()
        payload = await runner._session_payload(meeting=None, native_chunks=1)
        await runner.close()
    outcomes = [terminal.outcome for terminal in runner.parent_terminals]
    assert outcomes == ["failed", "final", "final", "final"]
    assert runner.parent_terminals[0].failure_reason == "deepgram_transport_error"
    assert payload["dispatch"]["submitted_segments"] == count
    assert payload["dispatch"]["terminal_segments"] == count
    assert payload["provider_fault"] is None
    _assert_capture_ledger_balanced(payload["capture_timing"])
    assert background_errors == []


@pytest.mark.asyncio
async def test_close_finishes_owned_dispatch_sink() -> None:
    script = hello_there_script()
    samples = _burst_meeting(1, silence_samples=8000)
    runner = ContinuousC5LiveRunner(
        network=False,
        ownership_enabled=True,
        intercept=script,
    )
    with install_deepgram_intercept(script):
        await runner.open(audio_seconds=float(samples.size) / 16000.0)
        await runner.feed(samples)
        await runner.close()
    pending = [
        task for task in asyncio.all_tasks() if task.get_name() == "peer-vad-dispatch"
    ]
    assert pending == []


@pytest.mark.asyncio
async def test_seal_lateness_flags_blocked_capture_loop_not_deadline_seals(tmp_path: Path) -> None:
    script = hello_there_script()
    sustained = _sustained_speech_pcm(8.0)
    runner = ContinuousC5LiveRunner(
        network=False,
        ownership_enabled=True,
        intercept=script,
    )
    with install_deepgram_intercept(script):
        await runner.open(audio_seconds=float(sustained.size) / 16000.0)
        await runner.feed(sustained[: sustained.size // 2])
        time.sleep(7.0)
        await runner.feed(sustained[sustained.size // 2 :])
        await runner.finalize()
        await runner.admit()
        await runner.translate()
        blocked_payload = await runner._session_payload(meeting=None, native_chunks=1)
        await runner.close()
    assert blocked_payload["c5_seal_reasons"][0] == "delivery_deadline"
    assert blocked_payload["seal_lateness"]["violations"] >= 1
    assert blocked_payload["seal_lateness"]["max_lateness_s"] > 0

    wav = write_pcm_wav(tmp_path / "paced_deadline.wav", sustained)
    with install_deepgram_intercept(script):
        paced = await run_continuous_wav(
            wav,
            network=False,
            secrets={},
            intercept=script,
            meeting=None,
            pace=True,
        )
    assert "delivery_deadline" in paced["c5_seal_reasons"]
    assert paced["seal_lateness"]["violations"] == 0
    assert 0.0 <= paced["seal_lateness"]["max_lateness_s"] <= paced["seal_lateness"]["quantization_s"]
    paced_pairs = paced["seal_lateness"]["pairs"]
    assert paced_pairs
    deadline_rows = [
        row for row in paced_pairs if row["seal_reason"] == "delivery_deadline"
    ]
    assert deadline_rows
    for row in deadline_rows:
        measured = max(
            0.0,
            row["sealed_at_monotonic_s"]
            - row["requested_deadline_monotonic_s"]
            - paced["seal_lateness"]["quantization_s"],
        )
        assert row["lateness_s"] == measured


@pytest.mark.asyncio
async def test_short_zero_network_run_recognizes_every_parent(tmp_path: Path) -> None:
    parents = 3
    first = hello_there_script()
    second = one_two_script(preroll_s=PREROLL_SECONDS)
    samples = _burst_meeting(parents, silence_samples=16000)
    wav = write_pcm_wav(tmp_path / "short_clean.wav", samples)
    with install_deepgram_intercept((first, second)):
        result = await run_continuous_wav(
            wav,
            network=False,
            secrets={},
            intercept=(first, second),
            meeting=None,
            pace=True,
        )
    assert result["n_parents"] == parents
    assert [parent["status"] for parent in result["parents"]] == ["complete"] * parents
    assert [parent["outcome"] for parent in result["parents"]] == ["final"] * parents
    assert result["dispatch"]["submitted_segments"] == parents
    assert result["dispatch"]["terminal_segments"] == parents
    assert result["provider_fault"] is None
    _assert_capture_ledger_balanced(result["capture_timing"])
    assert result["capture_timing"]["unprocessed_source_samples"] == 0
    assert result["seal_lateness"]["violations"] == 0


@pytest.mark.asyncio
async def test_c5_silence_hangover_seals_without_always_speech() -> None:
    runner = ContinuousC5LiveRunner(
        network=False,
        ownership_enabled=True,
        intercept=hello_there_script(),
    )
    result = await runner.run_pcm(hello_there_pcm(), boundary=None)
    assert result["vad_speech_chunks"] > 0
    assert result["vad_silence_chunks"] > 0
    assert result["c5_seal_reasons"]
    assert result["c5_seal_reasons"][0] in {"delivery_pause", "source_eof", "silence"}


def _peer_settings() -> AudioSegmentSettingsSnapshot:
    return AudioSegmentSettingsSnapshot(
        provider_id="deepgram",
        provider_signature=("deepgram", "nova-3"),
        runtime_signature=("deepgram",),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )


def _peer_receipt(segment: AudioSegmentIdentity) -> AudioSegmentTerminalReceipt:
    snapshot = AudioSegmentSnapshot(
        identity=segment,
        settings=_peer_settings(),
        content_ranges=(),
        context_ranges=(),
        failed_ranges=(),
        content_sample_count=0,
        context_sample_count=0,
        failed_normalized_sample_count=0,
        failed_source_sample_count=0,
        prefix_context_sample_count=0,
        synthetic_context_sample_count=0,
        genuine_onset=True,
        state="terminal",
        opened_at_monotonic_s=0.0,
        sealed_at_monotonic_s=1.0,
        seal_reason="silence",
    )
    return AudioSegmentTerminalReceipt(
        identity=segment,
        outcome="final",
        segment=snapshot,
        terminal_at_monotonic_s=1.0,
        text_authority="authoritative",
    )


def _peer_terminal(
    segment: AudioSegmentIdentity,
    text: str,
    first: str,
    second: str,
) -> STTProviderTurnTerminal:
    return STTProviderTurnTerminal(
        identity=STTProviderTurnIdentity(
            segment=segment,
            provider_epoch_id="epoch",
            provider_turn_id="turn",
        ),
        outcome="final",
        text=text,
        final_language_runs=(FinalLanguageRun(text, "en"),),
        text_authority="authoritative",
        timed_tokens=(
            STTTimedToken(
                text=first,
                language="en",
                start_ms=0,
                end_ms=100,
                timing="interval",
                source_start_sample=0,
                source_end_sample=1600,
            ),
            STTTimedToken(
                text=second,
                language="en",
                start_ms=100,
                end_ms=200,
                timing="interval",
                source_start_sample=1600,
                source_end_sample=3200,
            ),
        ),
    )


def _observe_cut(owner: PretranslationOwnershipOwner, hypothesis_id: str) -> None:
    owner.observe(
        ProspectiveSpeakerHypothesis(
            hypothesis_id=hypothesis_id,
            revision=0,
            capture_epoch=1,
            support_start_sample=1500,
            support_end_sample=1700,
            estimated_transition_sample=1600,
            observed_frontier_sample=3200,
            available_at_monotonic_s=0.5,
            producer_generation=1,
            reference_generation=1,
            producer_valid=True,
            reference_valid=True,
            local_slot=1,
        )
    )
    for start, end, relation in ((0, 1600, "CURRENT"), (1600, 3200, "OTHER")):
        owner.observe_evidence(
            capture_epoch=1,
            start_sample=start,
            end_sample=end,
            available_at_monotonic_s=0.5,
            relation=relation,
            producer_generation=1,
            reference_generation=1,
            reference_valid=True,
        )


async def _submit_parent(
    harness: TranslationOwnersTestHarness,
    owner: PretranslationOwnershipOwner,
    *,
    text: str,
    first: str,
    second: str,
    hypothesis_id: str,
    segment_order: int,
) -> STTProviderTurnTerminal:
    _observe_cut(owner, hypothesis_id)
    segment = AudioSegmentIdentity(
        activation_generation=1,
        segment_order=segment_order,
        segment_id=uuid4(),
        capture_epoch=1,
    )
    terminal = _peer_terminal(segment, text, first, second)
    await harness.peer_owner.handle_provider_turn_terminal(_peer_receipt(segment), terminal)
    await harness.translation_turns.wait_for_idle()
    return terminal


def _intercept_llm(intercept: InterceptOpenRouterClient) -> BudgetedOpenRouter:
    return BudgetedOpenRouter(
        OpenRouterLLMProvider(
            api_key="intercept-key",
            model=PINNED_TRANSLATION,
            max_tokens=100,
            client=intercept,
        ),
        ledger=None,
        phase="dev",
        network=False,
    )


async def _run_two_parents(
    llm: BudgetedOpenRouter,
    *,
    config: TranslationRuntimeConfig | None = None,
) -> tuple[list[STTProviderTurnTerminal], list[TranslationTurnChild]]:
    owner = PretranslationOwnershipOwner(enabled=True)
    harness = compose_r2_harness(llm, owner=owner, config=config)
    created: list[TranslationTurnChild] = []
    inner_created = harness.translation_turns.on_child_created

    async def record(child: TranslationTurnChild) -> None:
        created.append(child)
        await inner_created(child)

    harness.translation_turns.on_child_created = record
    await harness.start()
    try:
        terminals = [
            await _submit_parent(
                harness,
                owner,
                text="Hello there",
                first="Hello ",
                second="there",
                hypothesis_id="cut-a",
                segment_order=1,
            ),
            await _submit_parent(
                harness,
                owner,
                text="Nice day",
                first="Nice ",
                second="day",
                hypothesis_id="cut-b",
                segment_order=2,
            ),
        ]
        return terminals, created
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_consecutive_parents_use_fixed_empty_context_and_shared_prompt() -> None:
    intercept = InterceptOpenRouterClient("안녕")
    llm = _intercept_llm(intercept)
    terminals, created = await _run_two_parents(llm)
    parents = [str(terminal.identity.segment.segment_id) for terminal in terminals]
    assert [child.parent_utterance_id for child in created] == [
        terminals[0].identity.segment.segment_id,
        terminals[0].identity.segment.segment_id,
        terminals[1].identity.segment.segment_id,
        terminals[1].identity.segment.segment_id,
    ]
    assert [child.ownership_group_id for child in created] == [
        "CURRENT-0",
        "OTHER-1",
        "CURRENT-0",
        "OTHER-1",
    ]
    assert [str(child.utterance_id) for child in created] == [
        entry["utterance_id"] for entry in llm.requests
    ]
    assert len(dict.fromkeys(parents)) == 2
    calls = list(intercept.calls)
    assert [call["text"] for call in calls] == ["Hello ", "there", "Nice ", "day"]
    assert {call["system_prompt"] for call in calls} == {r2_rendered_system_prompt()}
    assert all(call["context"] == "" for call in calls)
    assert all(call["scene_participant_count"] is None for call in calls)
    assert {call["source_language"] for call in calls} == {"en"}
    assert {call["target_language"] for call in calls} == {"ko"}
    assert [entry["text"] for entry in llm.requests] == [call["text"] for call in calls]
    assert [entry["context"] for entry in llm.requests] == [call["context"] for call in calls]
    assert [entry["system_prompt"] for entry in llm.requests] == [
        call["system_prompt"] for call in calls
    ]
    assert all(entry["usd"] > 0 for entry in llm.requests)
    assert {entry["arm"] for entry in llm.requests} == {"r2"}
    llm.arm = "r0"
    arms = await evaluate_protocol_arms(
        terminals[1],
        r2_events=(),
        evidence=(),
        llm=llm,
        freeze_monotonic_s=2.0,
        admitted_at_monotonic_s=2.0,
        meeting=None,
        native_chunks=(),
        frontiers=(),
    )
    assert arms["r0"]["translated"] is True
    assert len(intercept.calls) == len(calls) + 1
    assert llm.requests[-1]["arm"] == "r0"
    disabled_call = intercept.calls[-1]
    assert disabled_call["text"] == "Nice day"
    assert disabled_call["context"] == ""
    assert disabled_call["scene_participant_count"] is None
    assert disabled_call["system_prompt"] == calls[0]["system_prompt"]


@pytest.mark.asyncio
async def test_default_context_windows_would_inject_history_across_parents() -> None:
    intercept = InterceptOpenRouterClient("안녕")
    llm = _intercept_llm(intercept)
    config = replace(
        r2_translation_config(),
        context_time_window_s=30.0,
        integrated_context_time_window_s=40.0,
    )
    await _run_two_parents(llm, config=config)
    calls = intercept.calls
    assert [call["text"] for call in calls] == ["Hello ", "there", "Nice ", "day"]
    assert calls[0]["context"] == ""
    history_contexts = [call["context"] for call in calls[1:]]
    assert all(context for context in history_contexts)
    assert any('[peer] "Hello"' in context for context in history_contexts)
