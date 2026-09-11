from __future__ import annotations

import inspect
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.arms import apply_observe_evidence
from experiments.psem_r2_policy.live_runner import (
    ContinuousC5LiveRunner,
    hello_there_pcm,
    hello_there_script,
    write_pcm_wav,
)
from experiments.psem_r2_policy.metrics import (
    aggregate_cluster_parents,
    confirmatory_decision,
    sequential_merge_contamination,
)
from experiments.psem_r2_policy.pipeline import (
    LIVE_ROUTE,
    admit_units,
    make_terminal,
    run_intercepted_deepgram_path,
    run_synthetic_path,
)
from experiments.psem_r2_policy.run import main
from experiments.psem_r2_policy.sortformer_live import LiveTransitionDecoder
from puripuly_heart.core.audio.pretranslation_ownership import PretranslationOwnershipOwner
from puripuly_heart.core.stt.backend import STTTimedToken
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend


@pytest.mark.asyncio
async def test_synthetic_adapter_path_splits_and_conserves() -> None:
    result = await run_synthetic_path()
    enabled = result["enabled"]
    assert result["network"] is False
    assert result["text"] == "Hello there"
    assert enabled["conserved"] is True
    assert enabled["group_ids"] == ["CURRENT-0", "OTHER-1"]
    assert enabled["child_groups"] == ["CURRENT-0", "OTHER-1"]
    assert enabled["child_texts"] == ["Hello ", "there"]
    assert result["disabled"]["n_units"] == 0
    assert result["disabled"]["child_groups"] == [""]
    assert result["live_route"]["asr_provider"] == "deepgram"
    assert result["live_route"]["asr_model"] == LIVE_ROUTE["asr_model"]


@pytest.mark.asyncio
async def test_intercepted_deepgram_words_split_and_conserve() -> None:
    result = await run_intercepted_deepgram_path()
    enabled = result["enabled"]
    assert result["network"] is False
    assert result["intercept"] is True
    assert result["adapter_reads_words"] is True
    assert result["n_timed"] == 2
    assert result["timed_start_ms"] == [0, 100]
    assert enabled["conserved"] is True
    assert enabled["group_ids"] == ["CURRENT-0", "OTHER-1"]
    assert enabled["child_texts"] == ["Hello ", "there"]


@pytest.mark.asyncio
async def test_intercepted_live_runner_uses_open_feed_receive_finalize() -> None:
    result = await run_intercepted_deepgram_path()
    assert "open" in result["methods"]
    assert "feed" in result["methods"]
    assert "receive" in result["methods"]
    assert "finalize" in result["methods"]
    assert "admit" in result["methods"]
    assert "translate" in result["methods"]
    assert result["open_session_calls"] >= 1
    assert result["r0"]["assignment"] == "disabled"
    assert result["r0"]["translated"] is True
    assert result["r2"]["group_ids"] == ["CURRENT-0", "OTHER-1"]
    assert result["path"].startswith("c5_wav->scoped_engine->deepgram_open_session")


@pytest.mark.asyncio
async def test_hypotheses_alone_do_not_label_current() -> None:
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
    from experiments.psem_r2_policy.pipeline import hypotheses_from_boundaries

    terminal = make_terminal(tokens)
    events = hypotheses_from_boundaries([1600])
    enabled = await admit_units(terminal, enabled=True, events=events, evidence=())
    assert enabled["assignment"] == "assigned"
    assert enabled["group_ids"] == ["UNKNOWN-0", "UNKNOWN-1"]
    assert enabled["child_texts"] == ["Hello ", "there"]


@pytest.mark.asyncio
async def test_live_intercept_without_covering_evidence_stays_unknown() -> None:
    runner = ContinuousC5LiveRunner(
        network=False,
        ownership_enabled=True,
        intercept=hello_there_script(),
        secrets={"DEEPGRAM_API_KEY": "intercept-key", "OPENROUTER_API_KEY": "intercept-key"},
    )
    result = await runner.run_pcm(
        hello_there_pcm(),
        boundary=1600,
        apply_intercept_evidence=False,
    )
    assert result["enabled"]["group_ids"] == ["UNKNOWN-0", "UNKNOWN-1"]


def test_observe_evidence_returns_source_status_and_rejects_extra_kwargs() -> None:
    owner = PretranslationOwnershipOwner(enabled=True)
    payload = {
        "capture_epoch": 1,
        "start_sample": 0,
        "end_sample": 1600,
        "available_at_monotonic_s": 1.0,
        "relation": "CURRENT",
        "producer_generation": object(),
        "reference_generation": object(),
        "reference_valid": True,
        "kind": "must-not-be-forwarded",
        "event_id": "ev.1",
    }
    assert apply_observe_evidence(owner, payload) == "observed"
    invalid = dict(payload)
    invalid["reference_valid"] = False
    assert apply_observe_evidence(owner, invalid) == "invalid"
    signature = inspect.signature(owner.observe_evidence)
    assert list(signature.parameters) == [
        "capture_epoch",
        "start_sample",
        "end_sample",
        "available_at_monotonic_s",
        "relation",
        "producer_generation",
        "reference_generation",
        "reference_valid",
    ]


def test_overlap_and_none_are_observe_evidence_not_c13_cuts() -> None:
    decoder = LiveTransitionDecoder()
    overlap = decoder.ingest_chunk(
        0, [[0.9, 0.9, 0.0, 0.0]], available_at_monotonic_s=1.0
    )
    none = decoder.ingest_chunk(
        1, [[0.0, 0.0, 0.0, 0.0]], available_at_monotonic_s=1.1
    )
    assert overlap == []
    assert none == []
    kinds = [item.kind for item in decoder.evidence]
    assert "overlap" in kinds
    assert "none" in kinds
    assert all(item.relation == "UNKNOWN" for item in decoder.evidence)


def test_same_speaker_span_is_not_primary_eligible() -> None:
    units = (
        {
            "token_indexes": [0, 1],
            "relation": "CURRENT",
            "start_source_sample": 0,
            "end_source_sample": 3200,
            "group_id": "CURRENT-0",
        },
    )
    attributed = (
        {
            "token_id": 0,
            "text": "Hello ",
            "status": "attributable",
            "roles": ["A"],
            "start_src": 0,
            "end_src": 1600,
        },
        {
            "token_id": 1,
            "text": "there",
            "status": "attributable",
            "roles": ["A"],
            "start_src": 1600,
            "end_src": 3200,
        },
    )
    words = (
        {"start_src": 0, "end_src": 1600, "role": "A"},
        {"start_src": 1600, "end_src": 3200, "role": "A"},
    )
    record = sequential_merge_contamination(
        units=units, attributed=attributed, words=words
    )
    assert record["sequential_target"] is False
    assert record["eligible"] is False


def test_empty_cluster_confirmatory_is_inconclusive_not_zero_pass() -> None:
    clustered = aggregate_cluster_parents([])
    decision = confirmatory_decision(cluster_rows=clustered["cluster_rows"])
    assert clustered["cluster_rows"] == []
    assert decision["pass"] is False
    assert decision["n_eligible_clusters"] == 0
    assert "Inconclusive" in decision["result"]


def test_live_route_uses_deepgram_backend_class() -> None:
    assert LIVE_ROUTE["asr_provider"] == "deepgram"
    assert LIVE_ROUTE["asr_model"] == "nova-3"
    assert LIVE_ROUTE["backend"] == DeepgramRealtimeSTTBackend.__name__
    assert LIVE_ROUTE["constructed_by"] == DeepgramRealtimeSTTBackend.__name__


def test_paid_cli_runs_wav_executor_while_director_gate_closed(
    tmp_path: Path,
) -> None:
    wav = write_pcm_wav(tmp_path / "hello.wav", hello_there_pcm())
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = main(["--paid", "--wav", str(wav)])
    finally:
        sys.stdout = old
    payload = json.loads(buf.getvalue())
    assert code == 0
    assert payload["executor"] == "run_continuous_wav"
    assert payload["wav_path"] == str(wav)
    assert payload["paid_blocked"] is True
    assert payload["network"] is False
    assert "open" in payload["methods"]
    assert "feed" in payload["methods"]
    assert "finalize" in payload["methods"]


def test_holdout_cli_stays_locked() -> None:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = old
    sys.stdout = buf
    try:
        code = main(["--phase", "holdout"])
    finally:
        sys.stdout = old
    payload = json.loads(buf.getvalue())
    assert code == 1
    assert payload["ok"] is False
    assert "locked" in payload["reason"]
    assert payload["confirmatory"]["pass"] is False
    assert "Inconclusive" in payload["confirmatory"]["result"]
