from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.metrics import (
    MIN_ELIGIBLE_CLUSTERS,
    aggregate_cluster_parents,
    attribute_tokens,
    confirmatory_decision,
    conservation_record,
    fragmentation_record,
    paired_cluster_bootstrap,
    policy_delta_rows,
    sequential_merge_contamination,
    u8_case_report,
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


def _arm(contaminated: int, attributable: int) -> dict:
    return {
        "contamination": {
            "contaminated_chars": contaminated,
            "attributable_chars": attributable,
            "proportion": (contaminated / attributable) if attributable else None,
            "eligible": True,
            "sequential_target": True,
        }
    }


def test_operational_failures_are_counted_while_degraded_prefixes_join_the_pool() -> None:
    parents = [
        {
            "cluster_id": "C1",
            "meeting": "ES2009a",
            "parent_id": "clean",
            "status": "complete",
            "outcome": "final",
            "text": "Hello there",
            "text_authority": "authoritative",
            "sequential_target": True,
            "r0": _arm(20, 200),
            "r2": _arm(10, 200),
        },
        {
            "cluster_id": "C1",
            "meeting": "ES2009a",
            "parent_id": "failed",
            "status": "unsuccessful",
            "outcome": "failed",
            "text_authority": "none",
            "failure_reason": "deepgram_transport_error",
            "accounted": True,
            "incomplete": False,
            "outage": False,
            "sequential_target": True,
            "text": "",
            "conserved": None,
            "r0": _arm(0, 0),
            "r2": _arm(0, 0),
        },
        {
            "cluster_id": "C1",
            "meeting": "ES2009a",
            "parent_id": "degraded",
            "status": "degraded",
            "outcome": "final",
            "text_authority": "degraded",
            "failure_reason": "deepgram_transport_error",
            "degraded": True,
            "sequential_target": True,
            "text": "accepted prefix",
            "conserved": True,
            "r0": _arm(0, 120),
            "r2": _arm(0, 120),
        },
    ]
    clustered = aggregate_cluster_parents(parents)
    assert clustered["n_operationally_unsuccessful"] == 1
    assert clustered["n_incomplete_source_parents"] == 0
    assert clustered["n_degraded_conditional"] == 1
    assert clustered["cluster_rows"][0]["n_parents"] == 2
    assert clustered["cluster_rows"][0]["r0_chars"] == 320
    assert clustered["unsuccessful_parents"][0]["failure_reason"] == "deepgram_transport_error"
    degraded_row = clustered["degraded_parents"][0]
    assert degraded_row["conditional_ownership"] is True
    assert degraded_row["status"] == "degraded"
    assert degraded_row["conserved"] is True
    assert degraded_row["r2_proportion"] == pytest.approx(0.0)
    assert clustered["coverage"]["operational_clean"] is False
    decision = confirmatory_decision(
        cluster_rows=clustered["cluster_rows"],
        coverage=clustered["coverage"],
    )
    assert decision["pass"] is False
    assert decision["operational_clean"] is False
    assert decision["n_operationally_unsuccessful"] == 1
    assert decision["n_degraded_conditional"] == 1


def test_clean_parents_only_keep_operational_clean() -> None:
    parents = [
        {
            "cluster_id": "C1",
            "meeting": "ES2009a",
            "parent_id": "clean",
            "status": "complete",
            "outcome": "final",
            "text": "Hello there",
            "text_authority": "authoritative",
            "sequential_target": True,
            "r0": _arm(20, 200),
            "r2": _arm(10, 200),
        },
    ]
    clustered = aggregate_cluster_parents(parents)
    assert clustered["coverage"]["operational_clean"] is True
    assert clustered["degraded_parents"] == []
    assert clustered["unsuccessful_parents"] == []


def _u8_parent(
    cluster: str,
    *,
    outcome: str = "final",
    text: str = "Hello there",
    authority: str = "authoritative",
    r0_prop: float = 0.5,
    r2_prop: float = 0.0,
    sequential: bool = True,
    accounted: bool = True,
    conserved: bool | None = True,
) -> dict:
    return {
        "cluster_id": cluster,
        "meeting": "ES2009a",
        "parent_id": f"{cluster}-0",
        "status": "unsuccessful" if outcome != "final" else "complete",
        "outcome": outcome,
        "text": text,
        "text_authority": authority,
        "failure_reason": None if outcome == "final" else "expired_before_recognition",
        "sequential_target": sequential,
        "accounted": accounted,
        "incomplete": not accounted,
        "outage": False,
        "conserved": conserved,
        "r0": _arm(int(round(r0_prop * 10)), 10),
        "r2": _arm(int(round(r2_prop * 10)), 10),
    }


def _u8_case(parents: list[dict], **overrides: object) -> dict:
    case = {
        "meeting": "ES2009a",
        "phase": "dev",
        "parents": parents,
        "capture_timing": {
            "input_source_samples": 960000,
            "fed_source_samples": 960000,
            "chunked_source_samples": 960000,
            "unprocessed_source_samples": 0,
            "buffered_source_samples": 0,
            "dropped_tail_source_samples": 0,
            "flush_pad_source_samples": 0,
        },
        "declared_source_samples": 960000,
        "sealed_segments": len(parents),
        "provider_fault": None,
        "task_failures": [],
    }
    case.update(overrides)
    return case


def _u8_phase(parents: list[dict], **overrides: object) -> dict:
    from experiments.psem_r2_policy.phase import aggregate_phase

    return aggregate_phase(parents, cases=[_u8_case(parents, **overrides)], phase="dev")


def test_eight_clusters_with_one_operational_expiry_can_conditionally_pass() -> None:
    clusters = [f"C{index}" for index in range(8)]
    parents = [_u8_parent(cluster) for cluster in clusters]
    parents.append(_u8_parent("C8", outcome="expired", text="", authority="none"))

    summary = _u8_phase(parents)
    decision = summary["confirmatory"]

    assert summary["execution_completed"] is True
    assert summary["evaluation_valid"] is True
    assert summary["operational_clean"] is False
    assert decision["pass"] is True
    assert decision["conditional_support"] is True
    assert decision["n_eligible_clusters"] == 8
    assert decision["cluster_mean_delta"] == pytest.approx(-0.5)
    assert decision["ci95"][1] < 0
    assert decision["improved_cluster_share"] == 1.0
    assert decision["n_operationally_unsuccessful"] == 1

    census = summary["operational_census"]["overall"]
    assert census["denominator"] == 9
    assert census["counts"]["expired"] == 1
    assert census["counts"]["final_nonempty"] == 8

    bounds = decision["sensitivity"]["formed_parent_selection_bounds"]
    assert bounds["N_total"] == 8
    assert bounds["M_total"] == 1
    expired = [row for row in bounds["per_cluster"] if row["cluster_id"] == "C8"][0]
    assert expired["N"] == 0
    assert expired["M"] == 1
    assert expired["lower"] == pytest.approx((0.0 - 1) / 1)
    assert expired["upper"] == pytest.approx((0.0 + 1) / 1)
    complete_cluster = [row for row in bounds["per_cluster"] if row["cluster_id"] == "C0"][0]
    assert complete_cluster["N"] == 1
    assert complete_cluster["M"] == 0
    assert complete_cluster["lower"] == pytest.approx(-0.5)
    assert complete_cluster["upper"] == pytest.approx(-0.5)
    assert bounds["equal_cluster_mean_lower"] == pytest.approx((-0.5 * 8 - 1.0) / 9)
    assert bounds["equal_cluster_mean_upper"] == pytest.approx((-0.5 * 8 + 1.0) / 9)
    assert bounds["fragility"] == "bounds_exclude_zero"


def test_missing_text_conservation_and_unobserved_tail_never_conditionally_pass() -> None:
    clusters = [f"C{index}" for index in range(8)]
    healthy = [_u8_parent(cluster) for cluster in clusters]

    conservation = _u8_phase([*healthy[:7], _u8_parent("C7", conserved=False)])
    assert conservation["confirmatory"]["result"] == "Safety failure"
    assert conservation["confirmatory"]["pass"] is False
    assert conservation["confirmatory"]["safety_failures"]
    assert conservation["evaluation_valid"] is False

    tail = _u8_phase(
        healthy,
        capture_timing={
            "input_source_samples": 960000,
            "fed_source_samples": 940000,
            "chunked_source_samples": 940000,
            "unprocessed_source_samples": 20000,
            "buffered_source_samples": 0,
            "dropped_tail_source_samples": 0,
            "flush_pad_source_samples": 0,
        },
    )
    assert tail["execution_completed"] is False
    assert tail["evaluation_valid"] is False
    assert tail["confirmatory"]["pass"] is False
    assert tail["confirmatory"]["execution_completed"] is False
    assert any(
        "unprocessed_source_samples" in reason for reason in tail["evaluation_invalid_reasons"]
    )

    missing_parent = _u8_phase(healthy, sealed_segments=9)
    assert missing_parent["execution_completed"] is False
    assert any(
        "missing_parents" in reason for reason in missing_parent["execution_incomplete_reasons"]
    )
    assert missing_parent["confirmatory"]["pass"] is False


def test_empty_parent_is_census_visible_but_never_an_estimator_denominator_point() -> None:
    clusters = [f"C{index}" for index in range(8)]
    parents = [_u8_parent(cluster) for cluster in clusters[:7]]
    parents.append(_u8_parent("C7", outcome="empty", text="", authority="none"))

    summary = _u8_phase(parents)
    decision = summary["confirmatory"]
    assert decision["n_eligible_clusters"] == 7
    assert decision["pass"] is False
    assert summary["operational_census"]["overall"]["counts"]["empty"] == 1
    empty_row = [
        row
        for row in summary["cluster_aggregate"]["pool_exclusions"]
        if row["operational_outcome"] == "empty"
    ]
    assert empty_row and empty_row[0]["pool_exclusion"] == "empty_text"
    bounds = decision["sensitivity"]["formed_parent_selection_bounds"]
    assert bounds["N_total"] == 7
    assert bounds["M_total"] == 1
    assert all(row["N"] <= 1 for row in bounds["per_cluster"])


def test_degraded_prefix_parent_joins_the_primary_pool_and_complete_only_drops_it() -> None:
    clusters = [f"C{index}" for index in range(8)]
    parents = [_u8_parent(cluster) for cluster in clusters[:7]]
    parents.append(_u8_parent("C7", authority="degraded", r2_prop=0.5))

    summary = _u8_phase(parents)
    decision = summary["confirmatory"]
    assert decision["n_eligible_clusters"] == 8
    assert decision["cluster_mean_delta"] == pytest.approx((-0.5 * 7 + 0.0) / 8)
    assert summary["n_degraded_conditional"] == 1
    assert summary["operational_clean"] is False
    assert summary["degraded_parents"][0]["conditional_ownership"] is True
    complete_only = decision["sensitivity"]["complete_only"]
    assert complete_only["n_clusters"] == 7
    assert complete_only["cluster_mean_delta"] == pytest.approx(-0.5)
    assert complete_only["change_from_primary"] == pytest.approx(-0.5 - (-0.4375))


def test_unknown_outcome_record_stays_visible_and_invalidates_evaluation() -> None:
    clusters = [f"C{index}" for index in range(8)]
    parents = [_u8_parent(cluster) for cluster in clusters[:7]]
    parents.append({**_u8_parent("C7"), "outcome": "provider_side_abort"})

    summary = _u8_phase(parents)
    census = summary["operational_census"]
    assert census["overall"]["n_unknown_outcome"] == 1
    assert census["unknown_outcome_rows"][0]["operational_outcome"].startswith("unknown:")
    assert summary["evaluation_valid"] is False
    assert any(
        "unknown_outcome_records" in reason for reason in summary["evaluation_invalid_reasons"]
    )
    assert summary["confirmatory"]["pass"] is False


def test_census_keeps_silent_cases_and_operational_outcomes_separate_from_source() -> None:
    clusters = [f"C{index}" for index in range(8)]
    parents = [_u8_parent(cluster) for cluster in clusters]
    parents.append(_u8_parent("C8", outcome="cancelled", text="", authority="none"))
    from experiments.psem_r2_policy.phase import aggregate_phase

    silent_case = _u8_case([], meeting="ES2002b", sealed_segments=0)
    cases = [_u8_case(parents), silent_case]
    summary = aggregate_phase(parents, cases=cases, phase="dev")

    census = summary["operational_census"]
    assert census["by_meeting"]["ES2009a"]["counts"]["cancelled"] == 1
    assert "ES2002b" in census["source_accounting"]
    assert census["source_accounting"]["ES2002b"]["formed_parents"] == 0
    assert census["source_accounting"]["ES2002b"]["fed_source_samples"] == 960000
    assert census["by_phase"]["dev"]["denominator"] == 9


def test_final_frame_pad_is_reconciled_while_lost_samples_are_not() -> None:
    parents = [_u8_parent("ES2009a")]
    padded = _u8_case(parents)
    padded["capture_timing"]["chunked_source_samples"] = 960512
    padded["capture_timing"]["flush_pad_source_samples"] = 512
    report = u8_case_report(padded)
    assert report["execution_completed"] is True
    assert report["execution_incomplete_reasons"] == []
    assert report["source_accounting"]["flush_pad_source_samples"] == 512

    unreconciled = _u8_case(parents)
    unreconciled["capture_timing"]["chunked_source_samples"] = 960000
    unreconciled["capture_timing"]["flush_pad_source_samples"] = 512
    report = u8_case_report(unreconciled)
    assert report["execution_completed"] is False
    assert any(
        reason.startswith("source_ledger_unreconciled:")
        for reason in report["execution_incomplete_reasons"]
    )

    dropped = _u8_case(parents)
    dropped["capture_timing"]["chunked_source_samples"] = 959872
    dropped["capture_timing"]["dropped_tail_source_samples"] = 128
    report = u8_case_report(dropped)
    assert report["execution_completed"] is False
    assert any(
        reason.startswith("dropped_tail_source_samples:")
        for reason in report["execution_incomplete_reasons"]
    )
