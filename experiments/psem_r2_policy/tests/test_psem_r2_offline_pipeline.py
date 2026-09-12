from __future__ import annotations

import hashlib
import inspect
import io
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy import metrics as psem_metrics
from experiments.psem_r2_policy.arms import apply_observe_evidence, r2_rendered_system_prompt
from experiments.psem_r2_policy.live_runner import (
    ContinuousC5LiveRunner,
    hello_there_pcm,
    hello_there_script,
    write_pcm_wav,
)
from experiments.psem_r2_policy.metrics import (
    aggregate_cluster_parents,
    cluster_id_for_meeting,
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
from experiments.psem_r2_policy.sortformer_live import (
    LiveTransitionDecoder,
    NativeSortformerProducer,
    evidence_payload,
    hypothesis_from_live_event,
)
from puripuly_heart.core.audio.pretranslation_ownership import PretranslationOwnershipOwner
from puripuly_heart.core.stt.backend import STTTimedToken
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend


@pytest.fixture(autouse=True)
def _psem_artifacts_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "psem-artifacts"
    monkeypatch.setattr(psem_metrics, "ARTIFACTS", target)
    return target


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
    assert result["n_parents"] == 1
    assert result["parents"][0]["n_timed"] == 2
    assert result["parents"][0]["timed_start_ms"] == [0, 160]
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
    assert result["r0"]["disposition"] == "disabled"
    assert result["r0"]["translated"] is True
    assert result["r0"]["outcomes"] == ["translated"]
    assert result["r0"]["child_translations"] == ["안녕"]
    requests = result["translation_requests"]
    assert [row["arm"] for row in requests] == ["r2", "r2", "r0"]
    assert [row["text"] for row in requests] == ["Hello ", "there", "Hello there"]
    assert all(row["context"] == "" for row in requests)
    assert all(row["scene_participant_count"] is None for row in requests)
    assert {row["system_prompt"] for row in requests} == {r2_rendered_system_prompt()}
    children = result["children"]
    assert [row["text"] for row in children] == ["Hello ", "there"]
    assert [row["ownership_group_id"] for row in children] == ["CURRENT-0", "OTHER-1"]
    assert {row["parent_utterance_id"] for row in children} == {result["parents"][0]["parent_id"]}
    assert [row["utterance_id"] for row in requests[:2]] == [
        row["utterance_id"] for row in children
    ]
    assert requests[2]["utterance_id"] == result["r0"]["child_ids"][0]
    assert result["r2"]["group_ids"] == ["CURRENT-0", "OTHER-1"]
    assert result["path"].startswith("c5_wav->scoped_engine->deepgram_open_session")
    parent = result["parents"][0]
    hypothesis = parent["hypotheses"][0]
    assert hypothesis["support_start_sample"] == 2559
    assert hypothesis["support_end_sample"] == 2560
    assert hypothesis["observed_frontier_sample"] == 5120
    assert hypothesis["producer_generation_matches_active"] is True
    assert hypothesis["reference_generation_matches_active"] is True
    assert hypothesis["producer_valid"] is True
    assert hypothesis["reference_valid"] is True
    assert hypothesis["retracted"] is False
    assert all(
        row["producer_generation_matches_active"]
        and row["reference_generation_matches_active"]
        for row in parent["evidence"]
    )


@pytest.mark.asyncio
async def test_hypotheses_without_reference_coverage_preserve_whole_parent() -> None:
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
    assert enabled["group_ids"] == ["UNKNOWN-0"]
    assert enabled["child_texts"] == ["Hello there"]


@pytest.mark.asyncio
async def test_live_intercept_without_covering_evidence_preserves_whole_parent() -> None:
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
    assert result["enabled"]["group_ids"] == ["UNKNOWN-0"]
    assert result["enabled"]["child_texts"] == ["Hello there"]


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
    overlap = decoder.ingest_chunk(0, [[0.9, 0.9, 0.0, 0.0]], available_at_monotonic_s=1.0)
    none = decoder.ingest_chunk(1, [[0.0, 0.0, 0.0, 0.0]], available_at_monotonic_s=1.1)
    assert overlap == []
    assert none == []
    kinds = [item.kind for item in decoder.evidence]
    assert "overlap" in kinds
    assert "none" in kinds
    assert all(item.relation == "UNKNOWN" for item in decoder.evidence)
def test_native_producer_observation_coverage_splits_only_without_source_holes() -> None:
    generation = object()
    tokens = (
        STTTimedToken(
            text="Hello ",
            language="en",
            start_ms=0,
            end_ms=160,
            timing="interval",
            source_start_sample=0,
            source_end_sample=2560,
        ),
        STTTimedToken(
            text="there",
            language="en",
            start_ms=160,
            end_ms=320,
            timing="interval",
            source_start_sample=2560,
            source_end_sample=5120,
        ),
    )

    covered = NativeSortformerProducer("unused.wav")
    events = covered.decoder.ingest_chunk(
        0,
        [[0.9, 0.0], [0.9, 0.0], [0.0, 0.9], [0.0, 0.9]],
        available_at_monotonic_s=1.0,
    )
    owner = PretranslationOwnershipOwner(enabled=True)
    for event in events:
        owner.observe(
            hypothesis_from_live_event(
                event,
                capture_epoch=1,
                producer_generation=generation,
                reference_generation=generation,
            )
        )
    for item in covered.drain_evidence():
        assert apply_observe_evidence(
            owner,
            evidence_payload(
                item,
                capture_epoch=1,
                producer_generation=generation,
                reference_generation=generation,
            ),
        ) == "observed"
    assignment = owner.assign(
        parent_utterance_id=uuid4(),
        timed_tokens=tokens,
        capture_epoch=1,
        admitted_at_monotonic_s=1.0,
        parent_text="Hello there",
    )
    assert [unit.group_id for unit in assignment.units] == ["CURRENT-0", "OTHER-1"]
    assert "".join(unit.text for unit in assignment.units) == "Hello there"

    incomplete = NativeSortformerProducer("unused.wav")
    incomplete.decoder.ingest_chunk(
        0,
        [[0.9, 0.0], [0.9, 0.0]],
        available_at_monotonic_s=1.0,
    )
    gap_events = incomplete.decoder.ingest_chunk(
        3,
        [[0.0, 0.9], [0.0, 0.9]],
        available_at_monotonic_s=1.0,
    )
    gap_owner = PretranslationOwnershipOwner(enabled=True)
    for event in gap_events:
        gap_owner.observe(
            hypothesis_from_live_event(
                event,
                capture_epoch=1,
                producer_generation=generation,
                reference_generation=generation,
            )
        )
    for item in incomplete.drain_evidence():
        apply_observe_evidence(
            gap_owner,
            evidence_payload(
                item,
                capture_epoch=1,
                producer_generation=generation,
                reference_generation=generation,
            ),
        )
    gap_tokens = (
        STTTimedToken(
            text="Hello ",
            language="en",
            start_ms=0,
            end_ms=240,
            timing="interval",
            source_start_sample=0,
            source_end_sample=3840,
        ),
        STTTimedToken(
            text="there",
            language="en",
            start_ms=240,
            end_ms=400,
            timing="interval",
            source_start_sample=3840,
            source_end_sample=6400,
        ),
    )
    abstained = gap_owner.assign(
        parent_utterance_id=uuid4(),
        timed_tokens=gap_tokens,
        capture_epoch=1,
        admitted_at_monotonic_s=1.0,
        parent_text="Hello there",
    )
    assert [unit.group_id for unit in abstained.units] == ["UNKNOWN-0"]
    assert abstained.unknown_reasons == ("insufficient_evidence_coverage",)




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
    record = sequential_merge_contamination(units=units, attributed=attributed, words=words)
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


def test_paid_cli_refuses_arbitrary_wav_and_exits_nonzero(tmp_path: Path) -> None:
    wav = write_pcm_wav(tmp_path / "hello.wav", hello_there_pcm())
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = main(["--paid", "--wav", str(wav)])
    finally:
        sys.stdout = old
    payload = json.loads(buf.getvalue())
    assert code == 1
    assert payload["refused"] is True
    assert payload["paid_blocked"] is True
    assert payload["runner_called"] is False
    assert "methods" not in payload
    assert "open_session_calls" not in payload


def test_holdout_cli_stays_locked() -> None:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = main(["--phase", "holdout"])
    finally:
        sys.stdout = old
    payload = json.loads(buf.getvalue())
    assert code == 1
    assert payload["ok"] is False
    assert payload.get("refused") is True
    assert "outputs" not in payload
    assert "locked" in payload["reason"] or "paid_ready" in payload["reason"]


def test_phase_conditional_evaluation_separates_execution_from_cleanliness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.psem_r2_policy import run as run_module

    def assessed_guard() -> dict:
        words = [
            {
                "id": "w0",
                "role": "S",
                "start": 0.0,
                "end": 0.4,
                "start_src": 0,
                "end_src": 6400,
                "text": "hello",
            }
        ]
        scored = psem_metrics.score_parent(
            parent_text="hello",
            tokens=[
                {
                    "token_id": "t0",
                    "text": "hello",
                    "source_start_sample": 0,
                    "source_end_sample": 6400,
                }
            ],
            units=[
                {
                    "group_id": "g0",
                    "relation": "CURRENT",
                    "token_indexes": ["t0"],
                    "start_source_sample": 0,
                    "end_source_sample": 6400,
                }
            ],
            words=words,
        )
        return psem_metrics.pair_parent_guard(scored["guard"], scored["guard"])

    def parent_row(meeting: str, *, outcome: str, text: str, authority: str) -> dict:
        return {
            "parent_id": f"{meeting}-0",
            "meeting": meeting,
            "cluster_id": cluster_id_for_meeting(meeting),
            "sequential_target": True,
            "guard": assessed_guard(),
            "status": "unsuccessful" if outcome != "final" else "complete",
            "degraded": authority == "degraded",
            "clean_completion": authority == "authoritative" and outcome == "final",
            "text_authority": authority,
            "failure_reason": None if outcome == "final" else "expired_before_recognition",
            "outcome": outcome,
            "seal_reason": "delivery_pause",
            "conserved": True,
            "accounted": True,
            "incomplete": False,
            "outage": False,
            "text": text,
            "marks": {"recognition_terminal": 1.0, "translation_admission": 1.1},
            "receipt": {"terminal_at_monotonic_s": 1.0},
            "r0": {
                "contamination": {
                    "proportion": 0.5,
                    "attributable_chars": 8,
                    "contaminated_chars": 4,
                    "eligible": True,
                }
            },
            "r2": {
                "contamination": {
                    "proportion": 0.0,
                    "attributable_chars": 8,
                    "contaminated_chars": 0,
                    "eligible": True,
                }
            },
        }

    scenario: dict[str, object] = {"outcome": "final", "authority": "authoritative"}

    case_capture = {
        "input_source_samples": 960000,
        "fed_source_samples": 960000,
        "chunked_source_samples": 960000,
        "unprocessed_source_samples": 0,
        "buffered_source_samples": 0,
        "dropped_tail_source_samples": 0,
        "flush_pad_source_samples": 0,
    }

    async def fake_live(_wav, *, budget=None, phase=None, meeting=None) -> dict:
        parents = [
            parent_row(
                str(meeting),
                outcome=str(scenario["outcome"]),
                text="Hello there",
                authority=str(scenario["authority"]),
            )
        ]
        return {
            "refused": False,
            "paid_blocked": False,
            "network": True,
            "meeting": meeting,
            "parents": parents,
            "capture_timing": dict(case_capture),
            "declared_source_samples": case_capture["input_source_samples"],
            "sealed_segments": len(parents),
            "provider_fault": None,
            "task_failures": [],
        }

    def run_cli(argv: list[str]) -> tuple[int, dict]:
        import io as io_module

        buffer = io_module.StringIO()
        previous = sys.stdout
        sys.stdout = buffer
        try:
            code = run_module.main(argv)
        finally:
            sys.stdout = previous
        return code, json.loads(buffer.getvalue())

    monkeypatch.setattr(run_module, "refuse_paid_if_disabled", lambda **kwargs: None)
    monkeypatch.setattr(
        run_module,
        "write_case_output",
        lambda *args, **kwargs: {"path": str(tmp_path / "case.json"), "sha256": "test"},
    )
    monkeypatch.setattr(run_module, "run_paid_live", fake_live)
    monkeypatch.setattr(run_module, "LEDGER_PATH", tmp_path / "budget.json")

    clean_code, clean_payload = run_cli(["--phase", "dev"])
    assert clean_code == 0
    assert clean_payload["execution_completed"] is True
    assert clean_payload["evaluation_valid"] is True
    assert clean_payload["clean_completion"] is True
    assert clean_payload["operational_clean"] is True
    assert clean_payload["ok"] is True
    assert clean_payload["confirmatory"]["n_eligible_clusters"] == 3
    assert clean_payload["confirmatory"]["pass"] is False
    assert "Inconclusive" in clean_payload["confirmatory"]["result"]

    scenario["authority"] = "degraded"
    degraded_code, degraded_payload = run_cli(["--phase", "dev"])
    assert degraded_code == 0
    assert degraded_payload["clean_completion"] is False
    assert degraded_payload["operational_clean"] is False
    assert degraded_payload["execution_completed"] is True
    assert degraded_payload["evaluation_valid"] is True
    assert degraded_payload["ok"] is True
    assert degraded_payload["n_degraded_conditional"] == 5
    assert degraded_payload["confirmatory"]["n_eligible_clusters"] == 3
    assert degraded_payload["operational_census"]["overall"]["counts"]["degraded_prefix"] == 5

    scenario["authority"] = "authoritative"
    scenario["outcome"] = "expired"
    expired_code, expired_payload = run_cli(["--phase", "dev"])
    assert expired_code == 0
    assert expired_payload["n_operationally_unsuccessful"] == 5
    assert expired_payload["operational_clean"] is False
    assert expired_payload["execution_completed"] is True
    assert expired_payload["confirmatory"]["pass"] is False
    assert expired_payload["operational_census"]["overall"]["counts"]["expired"] == 5

    scenario["outcome"] = "final"

    async def failing_live(_wav, *, budget=None, phase=None, meeting=None) -> dict:
        payload = await fake_live(_wav, budget=budget, phase=phase, meeting=meeting)
        payload["provider_fault"] = {"reason": "provider_recovery_exhausted"}
        return payload

    monkeypatch.setattr(run_module, "run_paid_live", failing_live)
    aborted_code, aborted_payload = run_cli(["--phase", "dev"])
    assert aborted_code == 1
    assert aborted_payload["execution_completed"] is False
    assert aborted_payload["ok"] is False
    assert any(
        "aborted_recording" in reason for reason in aborted_payload["execution_incomplete_reasons"]
    )


def test_case_output_digest_matches_preserved_utf8_bytes(tmp_path: Path) -> None:
    from experiments.psem_r2_policy.phase import write_case_output

    payload = {"text": "한글 𝛼", "lines": ["first", "second"]}
    record = write_case_output("dev", "ES2009a", payload, directory=tmp_path)
    raw = Path(record["path"]).read_bytes()
    assert record["sha256"] == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw) == payload
