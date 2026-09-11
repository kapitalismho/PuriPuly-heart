from __future__ import annotations

import sys
from pathlib import Path

import pytest
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.metrics import (
    MIN_ELIGIBLE_CLUSTERS,
    attribute_tokens,
    confirmatory_decision,
    conservation_record,
    fragmentation_record,
    paired_cluster_bootstrap,
    policy_delta_rows,
    sequential_merge_contamination,
)


def test_conservation_prefix_reports_lost_suffix() -> None:
    record = conservation_record(
        parent_text="Hello there",
        unit_texts=["Hello "],
        token_texts=["Hello ", "there"],
        token_ids=[0, 1],
        unit_token_ids=[[0]],
    )
    assert record["missing"] == "there"
    assert record["missing_text"] == "there"
    assert record["duplicate_text"] == ""
    assert record["missing_token_ids"] == [1]
    assert record["duplicate_token_ids"] == []
    assert record["order_preserved"] is False
    assert record["conserved_units_to_parent"] is False
    assert record["conserved_token_ids"] is False


def test_conservation_token_ids_independent_of_text() -> None:
    record = conservation_record(
        parent_text="ab",
        unit_texts=["ab"],
        token_texts=["a", "b"],
        token_ids=["t0", "t1"],
        unit_token_ids=[["t0", "t0"]],
    )
    assert record["conserved_units_to_parent"] is True
    assert record["missing_token_ids"] == ["t1"]
    assert record["duplicate_token_ids"] == ["t0"]
    assert record["order_preserved"] is False
    assert record["conserved_token_ids"] is False


def test_fragmentation_singleton_is_lexical_not_group_name() -> None:
    record = fragmentation_record(
        ["OTHER-1", "CURRENT-0"],
        unit_texts=["Hi", "Hello there"],
        unit_token_counts=[1, 2],
    )
    assert record["singleton_fragments"] == 1
    assert record["children_per_parent"] == 2


def test_mixed_ami_overlap_is_retained_not_forced() -> None:
    tokens = [
        {
            "token_id": 0,
            "text": "hi",
            "source_start_sample": 0,
            "source_end_sample": 1600,
        }
    ]
    words = [
        {"role": "A", "start_src": 0, "end_src": 1200, "text": "a"},
        {"role": "B", "start_src": 800, "end_src": 2000, "text": "b"},
    ]
    rows = attribute_tokens(tokens, words)
    assert rows[0]["status"] == "mixed"
    assert rows[0]["ambiguous"] is True
    assert rows[0]["roles"] == ["A", "B"]


def test_no_attributable_text_is_ineligible_not_zero() -> None:
    units = [
        {
            "group_id": "CURRENT-0",
            "relation": "CURRENT",
            "token_indexes": [0],
            "start_source_sample": 0,
            "end_source_sample": 1600,
        }
    ]
    attributed = [
        {
            "token_id": 0,
            "text": "hi",
            "status": "unaligned",
            "roles": [],
            "start_src": 0,
            "end_src": 1600,
            "ambiguous": False,
        }
    ]
    scored = sequential_merge_contamination(units=units, attributed=attributed, words=[])
    assert scored["eligible"] is False
    assert scored["proportion"] is None
    assert scored["coverage"] == "none"
    assert scored["attributable_chars"] == 0


def test_sequential_merge_counts_crossed_speaker_text() -> None:
    words = [
        {"role": "A", "start_src": 0, "end_src": 1600, "text": "aa"},
        {"role": "B", "start_src": 1600, "end_src": 3200, "text": "bb"},
    ]
    attributed = [
        {
            "token_id": 0,
            "text": "aa",
            "status": "attributable",
            "roles": ["A"],
            "start_src": 0,
            "end_src": 1600,
            "ambiguous": False,
        },
        {
            "token_id": 1,
            "text": "bb",
            "status": "attributable",
            "roles": ["B"],
            "start_src": 1600,
            "end_src": 3200,
            "ambiguous": False,
        },
    ]
    merged = sequential_merge_contamination(
        units=[
            {
                "group_id": "CURRENT-0",
                "relation": "CURRENT",
                "token_indexes": [0, 1],
                "start_source_sample": 0,
                "end_source_sample": 3200,
            }
        ],
        attributed=attributed,
        words=words,
    )
    split = sequential_merge_contamination(
        units=[
            {
                "group_id": "CURRENT-0",
                "relation": "CURRENT",
                "token_indexes": [0],
                "start_source_sample": 0,
                "end_source_sample": 1600,
            },
            {
                "group_id": "OTHER-1",
                "relation": "OTHER",
                "token_indexes": [1],
                "start_source_sample": 1600,
                "end_source_sample": 3200,
            },
        ],
        attributed=attributed,
        words=words,
    )
    assert merged["eligible"] is True
    assert merged["proportion"] == 0.5
    assert split["eligible"] is True
    assert split["proportion"] == 0.0
    rows = policy_delta_rows(
        per_cluster={
            "ES2009": {
                "R0": {"contamination": merged},
                "R2": {"contamination": split},
            }
        }
    )
    assert rows[0]["delta"] == -0.5
    assert rows[0]["eligible"] is True


def test_confirmatory_under_eight_clusters_is_inconclusive() -> None:
    rows = [
        {"cluster_id": "ES2009", "eligible": True, "delta": -0.2},
        {"cluster_id": "ES2002", "eligible": True, "delta": -0.1},
        {"cluster_id": "EN2009", "eligible": True, "delta": -0.3},
    ]
    decision = confirmatory_decision(cluster_rows=rows)
    assert decision["n_eligible_clusters"] == 3
    assert decision["n_eligible_clusters"] < MIN_ELIGIBLE_CLUSTERS
    assert decision["pass"] is False
    assert decision["result"].startswith("Inconclusive")
    assert decision["cluster_mean_delta"] == pytest_approx_mean([-0.2, -0.1, -0.3])
    boot = paired_cluster_bootstrap([-0.2, -0.1, -0.3])
    assert boot["seed"] == 156
    assert boot["resamples"] == 10000
    assert boot["n_clusters"] == 3


def pytest_approx_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_confirmatory_eight_clusters_can_support() -> None:
    rows = [{"cluster_id": str(index), "eligible": True, "delta": -0.1} for index in range(8)]
    decision = confirmatory_decision(cluster_rows=rows)
    assert decision["n_eligible_clusters"] == 8
    assert decision["pass"] is True
    assert decision["ci95"][1] < 0
    assert decision["result"].startswith("Supported")


def test_r1_seals_at_accepted_frontier_not_estimated_boundary() -> None:
    from experiments.psem_r2_policy.arms import r1_project_diagnostic
    from experiments.psem_r2_policy.pipeline import make_terminal
    from experiments.psem_r2_policy.sortformer_live import hypothesis_at_boundary
    from puripuly_heart.core.stt.backend import STTTimedToken

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
    producer = object()
    reference = object()
    event = hypothesis_at_boundary(
        1600,
        capture_epoch=1,
        available_at_monotonic_s=1.0,
        producer_generation=producer,
        reference_generation=reference,
    )
    result = r1_project_diagnostic(
        terminal,
        (event,),
        freeze_monotonic_s=2.0,
        frontiers=[{"sample": 2400, "available_at_monotonic_s": 1.0}],
    )
    assert result["translated"] is False
    assert result["diagnostic"] is True
    assert result["history"][0]["outcome"] == "applied"
    assert result["history"][0]["requestedX"] == 1600
    assert result["history"][0]["sealedZ"] == 2400


def test_control_charges_first_covering_chunk_not_transition_receipt() -> None:
    from experiments.psem_r2_policy.arms import charged_gt_events

    producer = object()
    reference = object()
    used, missing = charged_gt_events(
        gt_boundaries=[{"at_src": 1600}],
        native_chunks=(
            {
                "start_sample": 0,
                "end_sample": 1280,
                "available_at_monotonic_s": 1.0,
            },
            {
                "start_sample": 1280,
                "end_sample": 2560,
                "available_at_monotonic_s": 1.5,
            },
        ),
        capture_epoch=1,
        producer_generation=producer,
        reference_generation=reference,
    )
    assert missing == ()
    assert len(used) == 1
    assert used[0].available_at_monotonic_s == 1.5
    assert used[0].estimated_transition_sample == 1600


@pytest.mark.asyncio
async def test_missing_gt_blocks_control_and_skips_r1_control_translate() -> None:
    from experiments.psem_r2_policy.arms import evaluate_protocol_arms
    from experiments.psem_r2_policy.pipeline import make_terminal
    from puripuly_heart.core.stt.backend import STTTimedToken

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

    class CountingLLM:
        n = 0

        async def translate(
            self,
            *,
            utterance_id,
            text: str,
            system_prompt: str,
            source_language: str,
            target_language: str,
            context: str = "",
            scene_participant_count: int | None = None,
        ) -> object:
            from puripuly_heart.domain.models import Translation

            CountingLLM.n += 1
            return Translation(utterance_id, text="안녕", source_text=text, channel="peer")
    llm = CountingLLM()



    arms = await evaluate_protocol_arms(
        terminal,
        r2_events=(),
        evidence=(),
        llm=llm,
        freeze_monotonic_s=2.0,
        admitted_at_monotonic_s=2.0,
        meeting=None,
        native_chunks=(),
        frontiers=(),
    )
    assert arms["r1"]["translated"] is False
    assert arms["control"]["translated"] is False
    assert arms["control"]["blocked"] is True
    assert arms["control"]["reason"] == "no_meeting"
    assert arms["r0"]["translated"] is True
    assert arms["r0"]["outcomes"] == ["translated"]
    assert arms["r0"]["child_translations"] == ["안녕"]
    assert arms["r2"]["translated"] is False
    assert llm.n == 1


@pytest.mark.asyncio
async def test_r0_uses_openrouter_provider_signature_and_r2_prompt() -> None:
    import inspect
    from uuid import UUID

    from experiments.psem_r2_policy.arms import (
        evaluate_protocol_arms,
        r2_rendered_system_prompt,
    )
    from experiments.psem_r2_policy.live_runner import (
        BudgetedOpenRouter,
        InterceptOpenRouterClient,
    )
    from experiments.psem_r2_policy.pipeline import make_terminal
    from puripuly_heart.core.stt.backend import STTTimedToken
    from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider

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
    intercept = InterceptOpenRouterClient("안녕")
    llm = OpenRouterLLMProvider(
        api_key="intercept-key",
        model="google/gemma-4-26b-a4b-it",
        max_tokens=100,
        client=intercept,
    )
    provider_params = tuple(inspect.signature(OpenRouterLLMProvider.translate).parameters)
    assert provider_params == (
        "self",
        "utterance_id",
        "text",
        "system_prompt",
        "source_language",
        "target_language",
        "context",
        "scene_participant_count",
    )
    assert tuple(inspect.signature(BudgetedOpenRouter.translate).parameters) == provider_params
    arms = await evaluate_protocol_arms(
        terminal,
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
    assert arms["r0"]["outcomes"] == ["translated"]
    assert arms["r0"]["child_translations"] == ["안녕"]
    assert arms["r2"]["translated"] is False
    assert arms["r2"]["outcomes"] == ["source_only"]
    assert intercept.calls
    call = intercept.calls[0]
    assert set(call) == {
        "text",
        "system_prompt",
        "source_language",
        "target_language",
        "context",
        "scene_participant_count",
    }
    assert call["text"] == "Hello there"
    assert call["source_language"] == "en"
    assert call["target_language"] == "ko"
    assert call["context"] == ""
    assert call["scene_participant_count"] is None
    prompt = r2_rendered_system_prompt()
    assert prompt
    assert "${sourceName}" not in prompt
    assert call["system_prompt"] == prompt
    child_id = UUID(arms["r0"]["child_ids"][0])
    inspect.signature(OpenRouterLLMProvider.translate).bind(
        llm,
        utterance_id=child_id,
        text=call["text"],
        system_prompt=call["system_prompt"],
        source_language=call["source_language"],
        target_language=call["target_language"],
        context=call["context"],
        scene_participant_count=call["scene_participant_count"],
    )
