from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.budget import (
    BillingBoundError,
    BudgetError,
    BudgetLedger,
    deepgram_reserve_usd,
    openrouter_reserve_usd,
)
from experiments.psem_r2_policy.live_runner import (
    PREROLL_SECONDS,
    ContinuousC5LiveRunner,
    hello_there_pcm,
    hello_there_script,
    install_deepgram_intercept,
    one_two_script,
    run_continuous_wav,
    write_pcm_wav,
)
from experiments.psem_r2_policy.phase import phase_plan
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider

HZ = 16000
TAIL_SECONDS = 512.0 / HZ
BOUNDS = json.loads(
    (ROOT / "experiments" / "psem_r2_policy" / "BILLING_BOUNDS.json").read_text(encoding="utf-8")
)


def _declared_meeting_seconds(meeting: str) -> float:
    import wave

    from experiments.psem_r2_policy.live_runner import ami_wav_path

    with wave.open(str(ami_wav_path(meeting)), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _one_pass_usd(audio_seconds: float, **extra: float) -> float:
    params: dict[str, float] = {
        "max_audio_seconds": audio_seconds,
        "hangover_seconds": 0.8,
        "preroll_seconds": PREROLL_SECONDS,
        "tail_seconds": TAIL_SECONDS,
        "copies": 1,
        "reconnect_bound": 0,
    }
    params.update(extra)
    return deepgram_reserve_usd(**params)


def _meeting_wav(tmp_path: Path) -> tuple[Path, np.ndarray]:
    silence = np.zeros((40000,), dtype=np.float32)
    samples = np.concatenate([hello_there_pcm(), silence, hello_there_pcm(), silence])
    return write_pcm_wav(tmp_path / "meeting.wav", samples), samples


async def _run_meeting(
    tmp_path: Path,
    scripts: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, BudgetLedger, np.ndarray]:
    async def fake_translate(self: object, **kwargs: object) -> Translation:
        return Translation(uuid4(), text="\uc548\ub155")

    monkeypatch.setattr(OpenRouterLLMProvider, "translate", fake_translate)
    wav, samples = _meeting_wav(tmp_path)
    ledger = BudgetLedger(tmp_path / "budget.json")
    with install_deepgram_intercept(scripts):
        payload = await run_continuous_wav(
            wav,
            network=True,
            secrets={"DEEPGRAM_API_KEY": "k", "OPENROUTER_API_KEY": "k"},
            intercept=scripts,
            meeting=None,
            budget=ledger,
            pace=False,
        )
    return payload, ledger, samples


def _entries(ledger: BudgetLedger, kind: str) -> list[dict]:
    return [entry for entry in ledger.snapshot().entries if entry["meta"].get("kind") == kind]


@pytest.mark.asyncio
async def test_healthy_meeting_reserves_one_pass_and_settles_verified_pcm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = (hello_there_script(), one_two_script(preroll_s=PREROLL_SECONDS))
    payload, ledger, samples = await _run_meeting(tmp_path, scripts, monkeypatch)
    audio_seconds = float(samples.size) / HZ
    one_pass = _one_pass_usd(audio_seconds)
    duplicate_pass = _one_pass_usd(audio_seconds, reconnect_bound=1)
    assert payload["clean_completion"] is True
    assert payload["deepgram_reconciled"] is True
    assert payload["n_parents"] == 2
    deepgram_entries = _entries(ledger, "deepgram")
    assert len(deepgram_entries) == 1
    base = deepgram_entries[0]
    assert base["reserved_usd"] == pytest.approx(one_pass)
    assert base["reserved_usd"] < duplicate_pass
    assert base["state"] == "settled"
    verified = deepgram_reserve_usd(max_audio_seconds=audio_seconds, copies=1, reconnect_bound=0)
    assert verified < one_pass
    assert base["settled_usd"] == pytest.approx(min(one_pass, verified).__round__(12))
    assert base["settled_usd"] < base["reserved_usd"]
    pads = _entries(ledger, "deepgram-session-pad")
    assert len(pads) == payload["open_session_calls"]
    pad_usd = deepgram_reserve_usd(max_audio_seconds=2.0, copies=1, reconnect_bound=0)
    assert [entry["reserved_usd"] for entry in pads] == pytest.approx([pad_usd] * len(pads))
    assert [entry["state"] for entry in pads] == ["reserved"] * len(pads)
    snap = ledger.snapshot()
    assert snap.spent_usd == pytest.approx(0.0)
    assert snap.credit_usd == pytest.approx(
        base["settled_usd"] + sum(entry["reserved_usd"] for entry in pads)
    )


@pytest.mark.asyncio
async def test_unsuccessful_meeting_keeps_full_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = (
        hello_there_script(),
        one_two_script(preroll_s=PREROLL_SECONDS, failure="failed"),
    )
    payload, ledger, samples = await _run_meeting(tmp_path, scripts, monkeypatch)
    audio_seconds = float(samples.size) / HZ
    one_pass = _one_pass_usd(audio_seconds)
    assert payload["incomplete"] is True
    assert payload["clean_completion"] is False
    assert payload["deepgram_reconciled"] is False
    assert payload["deepgram_settled_usd"] is None
    base = _entries(ledger, "deepgram")[0]
    assert base["reserved_usd"] == pytest.approx(one_pass)
    assert base["state"] == "reserved"
    assert base["settled_usd"] is None
    snap = ledger.snapshot()
    assert snap.spent_usd == 0
    assert snap.credit_usd >= base["reserved_usd"] - 1e-12


def test_phase_plan_arithmetic_uses_regular_rate_and_declares_headroom() -> None:
    plan = phase_plan()
    assert plan["deepgram_rate_tier"] == "regular"
    assert plan["deepgram_rate_usd_per_minute"] == pytest.approx(0.0077)
    per_request = plan["openrouter_request_bound_usd"]
    for phase, row in plan["phases"].items():
        recomputed = sum(
            _one_pass_usd(_declared_meeting_seconds(meeting)) for meeting in row["meetings"]
        )
        assert row["deepgram_one_pass_usd"] == pytest.approx(recomputed)
        assert row["openrouter_allowance_usd"] == pytest.approx(
            row["audio_seconds"]
            * plan["parent_rate_per_second"]
            * plan["requests_per_parent"]
            * per_request
        )
        assert row["deepgram_credit_exempt"] is True
        assert row["cash_total_usd"] == pytest.approx(row["openrouter_allowance_usd"])
        assert row["fits"] == (row["cash_total_usd"] <= row["phase_cap_usd"] + 1e-9)
        assert row["requests_that_fit"] == int(
            max(row["phase_cap_usd"] - row["cash_total_usd"], 0.0) // per_request
        )
        assert row["phase_cap_usd"] == pytest.approx(BOUNDS["phase_caps_usd"][phase])
    combined = plan["combined"]
    assert plan["cash_scope"] == "openrouter"
    assert combined["fits"] is True
    assert combined["cash_total_with_contingency_usd"] == pytest.approx(
        combined["openrouter_allowance_usd"] + combined["contingency_usd"]
    )
    assert combined["cash_total_with_contingency_usd"] < combined["combined_cap_usd"]
    assert plan["phases"]["dev"]["fits"] is True
    assert plan["phases"]["holdout"]["fits"] is True


def test_deepgram_requires_positive_audio_seconds() -> None:
    with pytest.raises(BudgetError, match="max_audio_seconds"):
        deepgram_reserve_usd(max_audio_seconds=0)


def test_deepgram_reserve_uses_processed_audio_not_websocket_wall() -> None:
    one_minute = deepgram_reserve_usd(max_audio_seconds=60)
    assert one_minute == pytest.approx(0.0077)
    long_wall_short_audio = deepgram_reserve_usd(max_audio_seconds=1)
    assert long_wall_short_audio == pytest.approx(0.0077 / 60.0)
    stereo = deepgram_reserve_usd(max_audio_seconds=60, channels=2)
    assert stereo == pytest.approx(0.0154)


def test_deepgram_context_pad_is_included_and_ceiled() -> None:
    exact = deepgram_reserve_usd(max_audio_seconds=1.1, context_pad_seconds=0.2)
    assert exact == pytest.approx(2 / 60.0 * 0.0077)


def test_openrouter_utf8_bytes_exceed_korean_char_count() -> None:
    body = {
        "messages": [{"role": "user", "content": "안녕"}],
        "max_tokens": 100,
    }
    serialized = json.dumps(body, ensure_ascii=False)
    by_bytes = openrouter_reserve_usd(serialized_request=body)
    by_chars = (len(serialized) * 0.042 + 100 * 0.22) / 1_000_000.0
    assert by_bytes > by_chars
    assert len("안녕".encode("utf-8")) == 6
    assert len("안녕") == 2
    with pytest.raises(BillingBoundError):
        openrouter_reserve_usd(serialized_request=body, max_tokens=101)


def test_phase_and_total_caps_block_over_reserve(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json", cap_usd=5.0)
    ledger.reserve("dev-1", phase="dev", amount_usd=BOUNDS["phase_caps_usd"]["dev"])
    with pytest.raises(BudgetError, match="dev phase cap"):
        ledger.reserve("dev-2", phase="dev", amount_usd=0.01)
    ledger.reserve("hold-1", phase="holdout", amount_usd=BOUNDS["phase_caps_usd"]["holdout"])
    with pytest.raises(BudgetError, match="cap would be exceeded"):
        ledger.reserve(
            "cont-1", phase="contingency", amount_usd=BOUNDS["phase_caps_usd"]["contingency"] + 0.01
        )
    snap = ledger.snapshot()
    assert snap.reserved_usd == pytest.approx(4.5)
    assert snap.remaining_usd == pytest.approx(0.5)


def test_uncertain_settle_keeps_reserve(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("a", phase="dev", amount_usd=0.4)
    ledger.settle("a", billed_usd=None)
    snap = ledger.snapshot()
    assert snap.reserved_usd == pytest.approx(0.4)
    assert snap.spent_usd == 0


def test_failed_request_cannot_release_from_missing_usage_log(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("missing-log", phase="dev", amount_usd=0.2)
    ledger.settle("missing-log", billed_usd=None, keep_reserve=True)
    assert ledger.snapshot().reserved_usd == pytest.approx(0.2)


def test_reliable_settle_releases_unused_reserve(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("a", phase="dev", amount_usd=0.4)
    ledger.settle("a", billed_usd=0.1)
    snap = ledger.snapshot()
    assert snap.spent_usd == pytest.approx(0.1)
    assert snap.reserved_usd == 0


def test_restart_reloads_inflight_reserve(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    first = BudgetLedger(path)
    first.reserve("live", phase="dev", amount_usd=0.8)
    second = BudgetLedger(path)
    snap = second.snapshot()
    assert snap.reserved_usd == pytest.approx(0.8)
    with pytest.raises(BudgetError, match="already reserved"):
        second.reserve("live", phase="dev", amount_usd=0.1)


def test_concurrent_last_slot_has_one_winner(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("seed", phase="dev", amount_usd=1.6)
    results: list[str] = []

    def attempt(name: str) -> None:
        try:
            ledger.reserve(name, phase="dev", amount_usd=0.4)
            results.append(f"ok:{name}")
        except BudgetError:
            results.append(f"fail:{name}")

    threads = [
        threading.Thread(target=attempt, args=("t1",)),
        threading.Thread(target=attempt, args=("t2",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(item.startswith("ok:") for item in results) == 1
    assert sum(item.startswith("fail:") for item in results) == 1
    snap = ledger.snapshot()
    assert snap.phase_reserved["dev"] == pytest.approx(2.0)


def test_deepgram_hangover_preroll_tail_and_reconnect_are_reserved() -> None:
    base = deepgram_reserve_usd(max_audio_seconds=1.0)
    padded = deepgram_reserve_usd(
        max_audio_seconds=1.0,
        hangover_seconds=0.8,
        preroll_seconds=0.5,
        tail_seconds=512.0 / 16000.0,
        reconnect_bound=1,
    )
    assert padded > base
    assert padded == pytest.approx(6 / 60.0 * 0.0077)


@pytest.mark.asyncio
async def test_retry_session_reserves_turn_bound_before_opening(
    tmp_path: Path,
) -> None:
    from puripuly_heart.core.audio.ownership import (
        AudioSegmentSettingsSnapshot,
        PeerAudioSegmentLedger,
    )
    from puripuly_heart.core.clock import SystemClock
    from puripuly_heart.core.vad.gating import SpeechStart

    ledger = BudgetLedger(tmp_path / "budget.json")
    runner = ContinuousC5LiveRunner(
        network=True,
        intercept=None,
        budget=ledger,
        secrets={"DEEPGRAM_API_KEY": "k", "OPENROUTER_API_KEY": "k"},
        use_silero=False,
    )
    runner._clock = SystemClock()
    runner._deepgram_base_entry = "base"
    runner._audio_seconds = 120.0
    declared_pass = _one_pass_usd(120.0)
    ledger.reserve("base", phase="dev", amount_usd=declared_pass)
    segment_ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="deepgram",
            provider_signature=("deepgram", "nova-3"),
            runtime_signature=("deepgram",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=HZ,
            vad_speech_threshold=0.4,
            vad_hangover_ms=800,
            vad_pre_roll_ms=500,
        ),
    )
    runner._ledger = segment_ledger
    runner._sent_audio_seconds = 12.0
    segment_ledger.observe_vad_event(
        SpeechStart(
            utterance_id=uuid4(),
            pre_roll=np.zeros((8000,), dtype=np.float32),
            chunk=np.zeros((512,), dtype=np.float32),
        ),
        now_monotonic_s=0.0,
    )
    assert segment_ledger.current_open_segment_id is not None
    runner._reserve_scoped_session_open()
    assert _entries(ledger, "deepgram-retry") == []
    runner._reserve_scoped_session_open()
    retries = _entries(ledger, "deepgram-retry")
    assert len(retries) == 1
    expected_bound = max(0.0, PREROLL_SECONDS) + 0.8 + TAIL_SECONDS
    assert retries[0]["meta"]["audio_seconds"] == pytest.approx(expected_bound)
    assert retries[0]["state"] == "reserved"
    assert retries[0]["reserved_usd"] < declared_pass
    pads = _entries(ledger, "deepgram-session-pad")
    assert len(pads) == 2
    snap = ledger.snapshot()
    assert snap.reserved_usd == pytest.approx(declared_pass)
    assert snap.credit_usd == pytest.approx(
        retries[0]["reserved_usd"] + sum(entry["reserved_usd"] for entry in pads)
    )


def test_credit_usage_is_exempt_while_cash_caps_still_refuse(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    history = [
        ("dg-base", 0.18018, "deepgram"),
        ("dg-pad", 0.00025666667, "deepgram-session-pad"),
        ("or-1", 0.101803, "openrouter"),
    ]
    for request_id, amount, kind in history:
        ledger.reserve(request_id, phase="dev", amount_usd=amount, meta={"kind": kind})
    for index in range(6):
        ledger.reserve(
            f"dg-extra-{index}",
            phase="dev",
            amount_usd=1.0,
            meta={"kind": "deepgram-extra"},
        )
    ledger.reserve("dg-retry", phase="holdout", amount_usd=1.0, meta={"kind": "deepgram-retry"})
    snap = ledger.snapshot()
    assert snap.credit_usd == pytest.approx(0.18018 + 0.00025666667 + 7.0)
    assert snap.credit_entries == 9
    assert snap.spent_usd == pytest.approx(0.0)
    assert snap.reserved_usd == pytest.approx(0.101803)
    assert snap.remaining_usd == pytest.approx(5.0 - 0.101803)
    assert [entry["id"] for entry in snap.entries][:3] == ["dg-base", "dg-pad", "or-1"]
    assert len(snap.entries) == 10
    with pytest.raises(BudgetError, match="dev phase cap"):
        ledger.reserve("or-over", phase="dev", amount_usd=2.2, meta={"kind": "openrouter"})
    with pytest.raises(BudgetError, match="dev phase cap"):
        ledger.reserve("unknown-kind", phase="dev", amount_usd=2.2, meta={"kind": "mystery"})
    assert len(ledger.snapshot().entries) == 10
