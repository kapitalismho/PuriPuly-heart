from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.budget import BudgetLedger
from experiments.psem_r2_policy.live_runner import (
    ContinuousC5LiveRunner,
    EnergyVadEngine,
    hello_there_pcm,
    hello_there_script,
    install_deepgram_intercept,
    run_intercepted_live,
)
from experiments.psem_r2_policy.pipeline import run_continuous_wav, run_paid_live
from experiments.psem_r2_policy.secrets import ORIGINAL_ENV_LOCAL, credential_presence
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from puripuly_heart.core.stt.backend import STTSessionProjection


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
async def test_paid_gate_rejects_before_network_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[object] = []

    async def fake_runner(*_args: object, **_kwargs: object) -> dict:
        called.append(True)
        return {"ok": True, "completed": True}

    monkeypatch.setattr(
        "experiments.psem_r2_policy.pipeline.run_continuous_wav", fake_runner
    )
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
    monkeypatch.setattr(
        "experiments.psem_r2_policy.pipeline.ami_wav_path", lambda meeting: wav
    )
    monkeypatch.setattr(
        "experiments.psem_r2_policy.pipeline.run_continuous_wav", fake_runner
    )
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
        return real_reserve(
            request_id, phase=phase, amount_usd=amount_usd, meta=meta
        )

    monkeypatch.setattr(ledger, "reserve", tracked_reserve)
    original_open = DeepgramRealtimeSTTBackend.open_session

    async def wrapped(self: DeepgramRealtimeSTTBackend, *, projection: STTSessionProjection = STTSessionProjection()):
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
    assert snap.spent_usd == 0
    assert snap.reserved_usd > 0


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
