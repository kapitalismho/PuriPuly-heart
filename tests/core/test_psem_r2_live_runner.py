from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.live_runner import (
    ContinuousC5LiveRunner,
    hello_there_pcm,
    hello_there_script,
    run_intercepted_live,
)
from experiments.psem_r2_policy.pipeline import run_paid_live
from experiments.psem_r2_policy.secrets import ORIGINAL_ENV_LOCAL, credential_presence


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
    assert result["text"] == "Hello there"
    assert result["n_timed"] == 2
    assert result["timed_start_ms"] == [0, 100]
    enabled = result["enabled"]
    assert enabled["conserved"] is True
    assert enabled["group_ids"] == ["CURRENT-0", "OTHER-1"]
    assert enabled["child_groups"] == ["CURRENT-0", "OTHER-1"]
    assert enabled["child_texts"] == ["Hello ", "there"]
    assert result["vad_speech_chunks"] > 0
    assert result["c5_seal_reasons"]
    assert "always" not in "".join(result["c5_seal_reasons"])
    assert result["ledger"]["tokens"]
    assert result["metrics"]["conservation"]["missing_token_ids"] == []
    receipt = result["receipts"][0]
    assert receipt["available_at_monotonic_s"] <= receipt["applied_at_monotonic_s"]


@pytest.mark.asyncio
async def test_paid_gate_still_exercises_live_methods_without_network() -> None:
    payload = await run_paid_live()
    assert payload["network"] is False
    assert payload["paid_blocked"] is True
    assert payload["backend"] == "DeepgramRealtimeSTTBackend"
    assert payload["open_session_calls"] >= 1
    assert payload["live_methods"][:6] == [
        "open",
        "feed",
        "receive",
        "finalize",
        "admit",
        "translate",
    ]
    assert payload["paid_executor"] == "run_continuous_wav"
    presence = credential_presence()
    assert set(presence) == {"DEEPGRAM_API_KEY", "OPENROUTER_API_KEY"}
    assert payload["credentials_present"] == presence
    assert ORIGINAL_ENV_LOCAL.name == ".env.local"


@pytest.mark.asyncio
async def test_live_runner_disabled_arm_keeps_one_unsplit_parent() -> None:
    runner = ContinuousC5LiveRunner(
        network=False,
        ownership_enabled=False,
        intercept=hello_there_script(),
    )
    result = await runner.run_pcm(hello_there_pcm(), boundary=1600)
    assert result["enabled"]["group_ids"] == []
    assert result["disabled"]["disposition"] == "disabled"
    assert result["open_session_calls"] >= 1


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
