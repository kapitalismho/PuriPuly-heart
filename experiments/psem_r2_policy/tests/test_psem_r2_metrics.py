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
    pair_parent_guard,
    paired_cluster_bootstrap,
    policy_delta_rows,
    score_parent,
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
    parents = [{**_u8_parent(cluster), "guard": _assessed_guard()} for cluster in clusters]
    parents.append(_u8_parent("C8", outcome="expired", text="", authority="none"))

    summary = _u8_phase(parents)
    decision = summary["confirmatory"]

    assert summary["execution_completed"] is True
    assert summary["evaluation_valid"] is True
    assert summary["operational_clean"] is False
    assert summary["u8"]["guard"]["coverage"] == {
        "grouping_safety_assessed": True,
        "missing_guard_parents": 0,
        "unassessed_coverage_parents": 0,
        "unassessed_alignment_parents": 0,
        "no_annotation_source_parents": 0,
    }
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

    cluster_rows = summary["cluster_aggregate"]["cluster_rows"]
    scored = [row for row in cluster_rows if row["cluster_id"] == "C0"][0]
    assert scored["newly_unassigned_chars"] == 0
    assert scored["r0_unaligned_chars"] == 0
    assert scored["r2_unaligned_chars"] == 0
    assert scored["r0_mixed_chars"] == 0
    assert scored["r2_mixed_chars"] == 0
    assert scored["r2_unknown_chars"] == 0
    assert scored["worst_case_r2"]["contaminated_chars"] == 0
    assert scored["worst_case_r2"]["proportion"] == 0.0
    assert scored["benefit_explained_only_by_unassigned"] is False

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


def test_synthetic_hangover_never_masks_underfed_input() -> None:
    parents = [_u8_parent("ES2009a")]
    underfed = _u8_case(parents)
    underfed["declared_source_samples"] = 192000
    underfed["capture_timing"]["input_source_samples"] = 192000
    underfed["capture_timing"]["fed_source_samples"] = 191488
    underfed["capture_timing"]["synthetic_hangover_samples"] = 12800
    underfed["capture_timing"]["chunked_source_samples"] = 191488 + 12800
    report = u8_case_report(underfed)
    assert report["execution_completed"] is False
    assert "unconsumed_source_samples:512" in report["execution_incomplete_reasons"]

    overfed = _u8_case(parents)
    overfed["declared_source_samples"] = 192000
    overfed["capture_timing"]["input_source_samples"] = 192000
    overfed["capture_timing"]["fed_source_samples"] = 192512
    overfed["capture_timing"]["synthetic_hangover_samples"] = 12800
    overfed["capture_timing"]["chunked_source_samples"] = 192512 + 12800
    report = u8_case_report(overfed)
    assert report["execution_completed"] is False
    assert "overfed_source_samples:512" in report["execution_incomplete_reasons"]

    covered = _u8_case(parents)
    covered["declared_source_samples"] = 192000
    covered["capture_timing"]["input_source_samples"] = 192000
    covered["capture_timing"]["fed_source_samples"] = 192000
    covered["capture_timing"]["synthetic_hangover_samples"] = 12800
    covered["capture_timing"]["chunked_source_samples"] = 192000 + 12800
    report = u8_case_report(covered)
    assert report["execution_completed"] is True
    assert report["source_accounting"]["fed_source_samples"] == 192000
    assert report["source_accounting"]["synthetic_hangover_samples"] == 12800


def test_phase_census_groups_by_each_case_real_phase() -> None:
    from experiments.psem_r2_policy.phase import aggregate_phase

    dev_parents = [_u8_parent("C0"), _u8_parent("C1")]
    holdout_parents = [
        _u8_parent("C2"),
        _u8_parent("C3", outcome="expired", text="", authority="none"),
    ]
    dev_case = {**_u8_case(dev_parents), "phase": "dev"}
    holdout_case = {**_u8_case(holdout_parents), "phase": "holdout"}
    summary = aggregate_phase(
        dev_parents + holdout_parents,
        cases=[dev_case, holdout_case],
        phase=None,
    )
    census = summary["operational_census"]["by_phase"]
    assert set(census) == {"dev", "holdout"}
    assert census["dev"]["counts"]["final_nonempty"] == 2
    assert census["holdout"]["counts"]["expired"] == 1
    assert census["holdout"]["counts"]["final_nonempty"] == 1
    assert summary["operational_clean"] is False
    assert summary["confirmatory"]["n_eligible_clusters"] < MIN_ELIGIBLE_CLUSTERS
    assert summary["confirmatory"]["pass"] is False
    assert summary["confirmatory"]["conditional_support"] is False


def test_newly_unassigned_detail_is_retained_and_still_blocks() -> None:
    parents = [_u8_parent(f"C{index}") for index in range(8)]
    harmed = {**_u8_parent("C0"), "r2": _arm(0, 4)}
    parents[0] = harmed

    summary = _u8_phase(parents)
    decision = summary["confirmatory"]
    row = [
        item for item in summary["cluster_aggregate"]["cluster_rows"] if item["cluster_id"] == "C0"
    ][0]

    assert row["newly_unassigned_chars"] == 6
    assert row["worst_case_r2"]["attributable_chars"] == 10
    assert row["worst_case_r2"]["contaminated_chars"] == 6
    assert row["worst_case_r2"]["proportion"] == pytest.approx(0.6)
    assert row["benefit_explained_only_by_unassigned"] is True
    assert decision["benefit_explained_only_by_unassigned"] is True
    assert decision["pass"] is False
    assert decision["conditional_support"] is False


def _guard_words(entries: list[tuple[str, int, int, str]]) -> list[dict]:
    return [
        {
            "id": f"w{index}",
            "role": role,
            "start": start / 16000.0,
            "end": end / 16000.0,
            "start_src": start,
            "end_src": end,
            "text": text,
        }
        for index, (role, start, end, text) in enumerate(entries)
    ]


def _guard_token(token_id: str, text: str, start: int, end: int) -> dict:
    return {
        "token_id": token_id,
        "text": text,
        "source_start_sample": start,
        "source_end_sample": end,
    }


def _guard_unit(group_id: str, relation: str, token_ids: list[str], start: int, end: int) -> dict:
    return {
        "group_id": group_id,
        "relation": relation,
        "token_indexes": list(token_ids),
        "start_source_sample": start,
        "end_source_sample": end,
    }


def _arm_guard(
    *, tokens: list[dict], units: list[dict], words: list[dict], parent_text: str
) -> dict:
    scored = score_parent(parent_text=parent_text, tokens=tokens, units=units, words=words)
    return scored["guard"]


_ABA_WORDS = _guard_words(
    [
        ("A", 0, 6400, "alpha"),
        ("B", 8000, 14400, "bravo"),
        ("A", 16000, 22400, "charlie"),
    ]
)
_ABA_TOKENS = [
    _guard_token("t0", "alpha", 0, 6400),
    _guard_token("t1", "bravo", 8000, 14400),
    _guard_token("t2", "charlie", 16000, 22400),
]


def _assessed_guard() -> dict:
    clean = _arm_guard(
        tokens=[_guard_token("t0", "sure", 0, 6400)],
        units=[_guard_unit("g0", "CURRENT", ["t0"], 0, 6400)],
        words=_guard_words([("S", 0, 6400, "sure")]),
        parent_text="sure",
    )
    return pair_parent_guard(clean, clean)


def _unaligned_guard(*, annotation: bool) -> dict:
    words = _guard_words([("S", 100000, 106400, "elsewhere")]) if annotation else []
    scored = score_parent(
        parent_text="sure",
        tokens=[_guard_token("t0", "sure", 0, 6400)],
        units=[_guard_unit("g0", "CURRENT", ["t0"], 0, 6400)],
        words=words,
    )
    return pair_parent_guard(scored["guard"], scored["guard"])


_OVERLAP_WORDS = _guard_words(
    [
        ("A", 0, 6400, "alpha"),
        ("C", 0, 6400, "charlie"),
        ("B", 12800, 19200, "bravo"),
    ]
)
_OVERLAP_TOKENS = [_guard_token("t0", "alpha", 0, 6400)]
_OVERLAP_UNITS = [_guard_unit("g0", "CURRENT", ["t0"], 0, 14000)]
_UNMAPPED_WORDS = _guard_words([("A", 0, 6400, "alpha"), ("B", 12800, 19200, "bravo")])


def _scored_overlap(
    *,
    words: list[dict] | None = None,
    tokens: list[dict] | None = None,
    units: list[dict] | None = None,
    parent_text: str = "alpha",
) -> tuple[dict, dict]:
    scored = score_parent(
        parent_text=parent_text,
        tokens=_OVERLAP_TOKENS if tokens is None else tokens,
        units=_OVERLAP_UNITS if units is None else units,
        words=_OVERLAP_WORDS if words is None else words,
    )
    return scored, pair_parent_guard(scored["guard"], scored["guard"])


def _overlap_parent(cluster: str = "C0", **overrides: object) -> dict:
    scored, guard = _scored_overlap()
    parent = {
        **_u8_parent(cluster, text="alpha"),
        "parent_id": f"{cluster}-overlap",
        "span": [0, 14000],
        "guard": guard,
        "r0": scored,
        "r2": scored,
    }
    parent.update(overrides)
    return parent


def _without_annotation_count(parent: dict) -> dict:
    guard = dict(parent["guard"])
    checked = dict(guard.get("checked") or {})
    checked.pop("annotation_tokens", None)
    guard["checked"] = checked
    return {**parent, "guard": guard}


def _score_fields(rows: list[dict]) -> list[dict]:
    keys = (
        "cluster_id",
        "eligible",
        "r0_chars",
        "r2_chars",
        "r0_contaminated",
        "r2_contaminated",
        "r0_proportion",
        "r2_proportion",
        "delta",
        "newly_unassigned_chars",
    )
    return [{key: row[key] for key in keys} for row in rows]


def test_wrong_merge_oracle_flags_newly_merged_token_across_verified_boundary() -> None:
    r0_guard = _arm_guard(
        tokens=_ABA_TOKENS,
        units=[_guard_unit("g0", "CURRENT", ["t0", "t1", "t2"], 0, 22400)],
        words=_ABA_WORDS,
        parent_text="alpha bravo charlie",
    )
    r2_guard = _arm_guard(
        tokens=_ABA_TOKENS,
        units=[
            _guard_unit("g0", "CURRENT", ["t0"], 0, 6400),
            _guard_unit("g1", "CURRENT", ["t1", "t2"], 8000, 22400),
        ],
        words=_ABA_WORDS,
        parent_text="alpha bravo charlie",
    )

    assert r0_guard["wrong_token_ids"] == ["t1"]
    assert r2_guard["wrong_token_ids"] == ["t2"]
    assert r0_guard["wrong_chars"] == len("bravo")
    assert r2_guard["wrong_chars"] == len("charlie")
    paired = pair_parent_guard(r0_guard, r2_guard)
    assert paired["assessed"] is True
    assert paired["failures"] == ["wrong_merge:1"]
    assert paired["severe"] is True
    assert paired["wrong_merge"]["new_wrong_token_ids"] == ["t2"]
    assert paired["wrong_merge"]["new_wrong_tokens"] == 1
    assert paired["wrong_merge"]["new_wrong_chars"] == len("charlie")
    assert paired["wrong_merge"]["r0_wrong_tokens"] == 1
    assert paired["wrong_merge"]["r0_wrong_chars"] == len("bravo")
    assert paired["wrong_merge"]["r2_wrong_chars"] == len("charlie")
    r2_units = paired["arm_witnesses"]["r2"]["units"]
    assert r2_units[0]["crossed_verified_boundary"] is False
    assert r2_units[1]["reference_role"] == "B"
    assert r2_units[1]["wrong_token_ids"] == ["t2"]
    assert r2_units[1]["wrong_chars"] == len("charlie")
    assert r2_units[0]["wrong_chars"] == 0


def test_wrong_merge_oracle_accepts_unchanged_and_safe_split_grouping() -> None:
    r0_guard = _arm_guard(
        tokens=_ABA_TOKENS,
        units=[_guard_unit("g0", "CURRENT", ["t0", "t1", "t2"], 0, 22400)],
        words=_ABA_WORDS,
        parent_text="alpha bravo charlie",
    )
    split_guard = _arm_guard(
        tokens=_ABA_TOKENS,
        units=[
            _guard_unit("g0", "CURRENT", ["t0", "t1"], 0, 14400),
            _guard_unit("g1", "CURRENT", ["t2"], 16000, 22400),
        ],
        words=_ABA_WORDS,
        parent_text="alpha bravo charlie",
    )
    for guard in (r0_guard, split_guard):
        paired = pair_parent_guard(r0_guard, guard)
        assert paired["assessed"] is True
        assert paired["wrong_merge"]["new_wrong_token_ids"] == []
        assert paired["wrong_merge"]["new_wrong_chars"] == 0
        assert paired["failures"] == []
        assert paired["severe"] is False


def test_same_speaker_cross_owner_guard_variants() -> None:
    words = _guard_words([("S", 0, 6400, "sure"), ("S", 8000, 14400, "thing")])
    tokens = [_guard_token("t0", "sure", 0, 6400), _guard_token("t1", "thing", 8000, 14400)]
    single = _arm_guard(
        tokens=tokens,
        units=[_guard_unit("g0", "CURRENT", ["t0", "t1"], 0, 14400)],
        words=words,
        parent_text="sure thing",
    )
    cross_owner = _arm_guard(
        tokens=tokens,
        units=[
            _guard_unit("g0", "CURRENT", ["t0"], 0, 6400),
            _guard_unit("g1", "OTHER", ["t1"], 8000, 14400),
        ],
        words=words,
        parent_text="sure thing",
    )
    two_other = _arm_guard(
        tokens=tokens,
        units=[
            _guard_unit("g0", "OTHER", ["t0"], 0, 6400),
            _guard_unit("g1", "OTHER", ["t1"], 8000, 14400),
        ],
        words=words,
        parent_text="sure thing",
    )
    unknown_side = _arm_guard(
        tokens=tokens,
        units=[
            _guard_unit("g0", "CURRENT", ["t0"], 0, 6400),
            _guard_unit("g1", "UNKNOWN", ["t1"], 8000, 14400),
        ],
        words=words,
        parent_text="sure thing",
    )

    severe = pair_parent_guard(single, cross_owner)
    assert severe["same_speaker"]["stratum"] is True
    assert severe["failures"] == ["same_speaker_cross_owner:2"]
    assert severe["same_speaker"]["witnesses"] == [
        {"role": "S", "current_token_ids": ["t0"], "other_token_ids": ["t1"]}
    ]
    assert pair_parent_guard(single, two_other)["failures"] == []
    assert pair_parent_guard(single, unknown_side)["failures"] == []
    assert pair_parent_guard(cross_owner, cross_owner)["failures"] == []


def test_guard_coverage_gaps_are_inconclusive_and_only_bind_in_pool_parents() -> None:
    clusters = [f"C{index}" for index in range(8)]
    assessed = [{**_u8_parent(cluster), "guard": _assessed_guard()} for cluster in clusters]

    missing = list(assessed)
    missing[0] = _u8_parent("C0")
    summary = _u8_phase(missing)
    decision = summary["confirmatory"]
    assert summary["evaluation_valid"] is False
    assert decision["pass"] is False
    assert decision["conditional_support"] is False
    assert decision["result"].startswith("Inconclusive")
    assert decision["safety_failures"] == []
    assert "ES2009a:missing_guard:C0-0" in summary["evaluation_invalid_reasons"]
    guard = summary["u8"]["guard"]
    assert guard["severe"] is False
    assert guard["failures"] == []
    assert guard["coverage"]["grouping_safety_assessed"] is False
    assert guard["coverage"]["missing_guard_parents"] == 1

    unpaired = list(assessed)
    unpaired[0] = {**_u8_parent("C0"), "guard": pair_parent_guard(None, None)}
    unpaired_summary = _u8_phase(unpaired)
    assert unpaired_summary["evaluation_valid"] is False
    assert unpaired_summary["confirmatory"]["result"].startswith("Inconclusive")
    assert unpaired_summary["u8"]["guard"]["coverage"]["missing_guard_parents"] == 1

    primary_unaligned = list(assessed)
    primary_unaligned[0] = {**_u8_parent("C0"), "guard": _unaligned_guard(annotation=True)}
    primary_summary = _u8_phase(primary_unaligned)
    assert primary_summary["evaluation_valid"] is False
    assert primary_summary["confirmatory"]["result"].startswith("Inconclusive")
    assert primary_summary["confirmatory"]["safety_failures"] == []
    primary_guard = primary_summary["u8"]["guard"]
    assert primary_guard["severe"] is False
    assert primary_guard["wrong_merge_tokens"] == 0
    assert primary_guard["wrong_merge_chars"] == 0
    assert primary_guard["coverage"]["grouping_safety_assessed"] is False
    assert primary_guard["coverage"]["unassessed_coverage_parents"] == 1
    assert primary_guard["coverage"]["unassessed_alignment_parents"] == 1
    assert primary_guard["cases"][0]["parents"][0]["coverage_status"] == (
        "no_attributable_lexical_tokens"
    )
    assert primary_guard["cases"][0]["parents"][0]["grouping_safety_assessed"] is False

    non_pool_missing = list(assessed)
    non_pool_missing.append(_u8_parent("C9", sequential=False))
    non_pool_missing_summary = _u8_phase(non_pool_missing)
    assert non_pool_missing_summary["evaluation_valid"] is False
    assert non_pool_missing_summary["confirmatory"]["result"].startswith("Inconclusive")
    assert "ES2009a:missing_guard:C9-0" in non_pool_missing_summary["evaluation_invalid_reasons"]
    assert non_pool_missing_summary["u8"]["guard"]["coverage"]["missing_guard_parents"] == 1
    assert non_pool_missing_summary["confirmatory"]["safety_failures"] == []

    degraded_assessed = list(assessed) + [
        {**_u8_parent("C8", authority="degraded"), "guard": _assessed_guard()}
    ]
    assert _u8_phase(degraded_assessed)["evaluation_valid"] is True
    degraded_without_guard = list(assessed) + [_u8_parent("C8", authority="degraded")]
    degraded_summary = _u8_phase(degraded_without_guard)
    assert degraded_summary["evaluation_valid"] is False
    assert degraded_summary["u8"]["guard"]["coverage"]["missing_guard_parents"] == 1

    operational = list(assessed) + [_u8_parent("C8", outcome="expired", text="", authority="none")]
    assert _u8_phase(operational)["evaluation_valid"] is True

    non_pool = list(assessed)
    non_pool.append(
        {
            **_u8_parent("C9", sequential=False),
            "guard": _unaligned_guard(annotation=True),
        }
    )
    non_pool_summary = _u8_phase(non_pool)
    assert non_pool_summary["evaluation_valid"] is True
    non_pool_guard = non_pool_summary["u8"]["guard"]
    assert non_pool_guard["coverage"]["grouping_safety_assessed"] is True
    assert non_pool_guard["coverage"]["unassessed_coverage_parents"] == 0
    assert non_pool_guard["coverage"]["unassessed_alignment_parents"] == 1

    no_annotation = list(assessed)
    no_annotation[0] = {**_u8_parent("C0"), "guard": _unaligned_guard(annotation=False)}
    no_annotation_summary = _u8_phase(no_annotation)
    assert no_annotation_summary["evaluation_valid"] is False
    assert no_annotation_summary["confirmatory"]["result"].startswith("Inconclusive")
    assert no_annotation_summary["confirmatory"]["safety_failures"] == []
    no_annotation_guard = no_annotation_summary["u8"]["guard"]
    assert no_annotation_guard["coverage"]["grouping_safety_assessed"] is False
    assert no_annotation_guard["coverage"]["no_annotation_source_parents"] == 1
    assert no_annotation_guard["coverage"]["unassessed_coverage_parents"] == 1
    assert no_annotation_guard["coverage"]["unassessed_alignment_parents"] == 0
    assert no_annotation_guard["cases"][0]["parents"][0]["coverage_status"] == (
        "no_annotation_source"
    )

    non_pool_no_annotation = list(assessed)
    non_pool_no_annotation.append(
        {
            **_u8_parent("C9", sequential=False),
            "guard": _unaligned_guard(annotation=False),
        }
    )
    non_pool_no_annotation_summary = _u8_phase(non_pool_no_annotation)
    assert non_pool_no_annotation_summary["evaluation_valid"] is True
    non_pool_no_annotation_guard = non_pool_no_annotation_summary["u8"]["guard"]
    assert non_pool_no_annotation_guard["coverage"]["grouping_safety_assessed"] is True
    assert non_pool_no_annotation_guard["coverage"]["no_annotation_source_parents"] == 1
    assert non_pool_no_annotation_guard["coverage"]["unassessed_coverage_parents"] == 0

    zero_gt = [
        {**_u8_parent(cluster), "guard": _unaligned_guard(annotation=False)} for cluster in clusters
    ]
    zero_gt_summary = _u8_phase(zero_gt)
    assert zero_gt_summary["evaluation_valid"] is False
    assert zero_gt_summary["u8"]["guard"]["coverage"]["grouping_safety_assessed"] is False
    assert zero_gt_summary["u8"]["guard"]["coverage"]["unassessed_coverage_parents"] == 8
    assert zero_gt_summary["u8"]["guard"]["coverage"]["no_annotation_source_parents"] == 8
    assert zero_gt_summary["u8"]["guard"]["severe"] is False
    assert zero_gt_summary["u8"]["guard"]["wrong_merge_tokens"] == 0
    assert zero_gt_summary["u8"]["guard"]["wrong_merge_chars"] == 0
    assert zero_gt_summary["confirmatory"]["safety_failures"] == []


def test_measured_overlap_unassessable_parent_is_reported_without_changing_the_estimate() -> None:
    clusters = [f"C{index}" for index in range(8)]
    healthy = [{**_u8_parent(cluster), "guard": _assessed_guard()} for cluster in clusters]
    baseline = _u8_phase(healthy)

    overlap = _overlap_parent("C0")
    summary = _u8_phase([*healthy, overlap])
    decision = summary["confirmatory"]

    assert summary["execution_completed"] is True
    assert summary["evaluation_valid"] is True
    assert summary["evaluation_invalid_reasons"] == []
    assert decision["evaluation_valid"] is True
    assert decision["pass"] is True
    assert decision["n_eligible_clusters"] == 8

    assert summary["operational_census"]["overall"]["counts"]["final_nonempty"] == 9
    pool_row = [
        row
        for row in summary["cluster_aggregate"]["pool_exclusions"]
        if row["parent_id"] == "C0-overlap"
    ][0]
    assert pool_row["pool_exclusion"] == "overlap_unassessable"

    assert _score_fields(summary["cluster_aggregate"]["cluster_rows"]) == _score_fields(
        baseline["cluster_aggregate"]["cluster_rows"]
    )
    assert decision["cluster_mean_delta"] == pytest.approx(
        baseline["confirmatory"]["cluster_mean_delta"]
    )
    assert decision["ci95"] == baseline["confirmatory"]["ci95"]

    guard = summary["u8"]["guard"]
    assert guard["overlap_unassessable_parents"] == 1
    parent_row = [
        row for row in guard["cases"][0]["parents"] if row["parent_id"] == "C0-overlap"
    ][0]
    assert parent_row["coverage_status"] == "no_attributable_lexical_tokens"
    assert parent_row["lexical_tokens"] == 0
    assert parent_row["annotation_tokens"] == 3
    assert parent_row["excluded"]["mixed"] == 1
    assert parent_row["grouping_safety_assessed"] is False

    bounds = decision["sensitivity"]["formed_parent_selection_bounds"]
    base_bounds = baseline["confirmatory"]["sensitivity"]["formed_parent_selection_bounds"]
    assert bounds["N_total"] == base_bounds["N_total"] == 8
    assert bounds["M_total"] == base_bounds["M_total"] + 1
    assert bounds["M_unscorable_total"] == 1
    assert bounds["M_overlap_unassessable_total"] == 1
    assert bounds["overlap_unassessable_subset"] == [{"cluster_id": "C0", "count": 1}]
    cluster_row = [row for row in bounds["per_cluster"] if row["cluster_id"] == "C0"][0]
    assert cluster_row["N"] == 1
    assert cluster_row["M"] == 1
    assert cluster_row["M_overlap_unassessable"] == 1
    assert cluster_row["lower"] == pytest.approx((-0.5 - 1.0) / 2)
    assert cluster_row["upper"] == pytest.approx((-0.5 + 1.0) / 2)

    report = summary["u8"]["overlap_coverage"]
    overall = report["overall"]
    accepted = 8 * len("Hello there") + len("alpha")
    assert overall["formed_parents"] == 9
    assert overall["accepted_nonempty_parents"] == 9
    assert overall["accepted_chars"] == accepted
    assert overall["qualified_parents"] == 1
    assert overall["qualified_accepted_chars"] == len("alpha")
    assert overall["qualified_parent_rate_of_formed"] == pytest.approx(1 / 9)
    assert overall["qualified_parent_rate_of_accepted_nonempty"] == pytest.approx(1 / 9)
    assert overall["accepted_char_coverage"] == pytest.approx(len("alpha") / accepted)
    assert overall["missing_guard_parents"] == 0
    assert overall["zero_coverage_parents"] == 1
    assert "no safety claim" in report["claim_scope"]
    assert report["by_case"][0]["meeting"] == "ES2009a"
    assert report["by_cluster"]["C0"]["qualified_parents"] == 1
    assert report["by_phase"]["dev"]["qualified_parents"] == 1
    detail = report["parents"][0]
    assert detail["parent_id"] == "C0-overlap"
    assert detail["overlap_unassessable"] is True
    assert detail["qualification_reason"] is None
    assert detail["guard_computed"] is True
    assert detail["coverage_status"] == "no_attributable_lexical_tokens"
    assert detail["source_interval"] == [0, 14000]
    assert detail["accepted_chars"] == len("alpha")
    assert detail["checked_lexical_tokens"] == 0
    assert detail["annotation_tokens"] == 3
    assert detail["excluded_tokens"] == {"punctuation_only": 0, "mixed": 1, "unaligned": 0}
    assert detail["r0_mixed_chars"] == len("alpha")
    assert detail["r2_mixed_chars"] == len("alpha")
    assert detail["r0_contamination_reason"] == "no_attributable_accepted_text"
    assert detail["r2_contamination_reason"] == "no_attributable_accepted_text"


def test_overlap_unassessable_requires_measured_multi_role_overlap_evidence() -> None:
    clusters = [f"C{index}" for index in range(8)]
    assessed = [{**_u8_parent(cluster), "guard": _assessed_guard()} for cluster in clusters]

    unmapped_scored, unmapped_guard = _scored_overlap(
        words=_UNMAPPED_WORDS,
        tokens=[_guard_token("t0", "mumble", 7000, 8000)],
        parent_text="mumble",
    )
    no_annotation = {**_u8_parent("C0"), "guard": _unaligned_guard(annotation=False)}

    variants = {
        "missing_guard": {**_overlap_parent("C0"), "guard": None},
        "missing_paired_records": {**_overlap_parent("C0"), "r2": None},
        "invalid_provenance": {**_overlap_parent("C0"), "provenance_valid": False},
        "unmapped_only": {
            **_u8_parent("C0", text="mumble"),
            "parent_id": "C0-unmapped",
            "span": [0, 14000],
            "guard": unmapped_guard,
            "r0": unmapped_scored,
            "r2": unmapped_scored,
            "conserved": True,
        },
        "paired_score_present": {**_overlap_parent("C0"), "r0": _arm(0, 10), "r2": _arm(0, 10)},
        "no_annotation_source": no_annotation,
        "absent_annotation_count": _without_annotation_count(_overlap_parent("C0")),
    }

    overlap_row = unmapped_guard["checked"]["excluded"]
    assert overlap_row == {"punctuation_only": 0, "mixed": 0, "unaligned": 1}

    for name, parent in variants.items():
        summary = _u8_phase([*assessed, parent])
        assert summary["u8"]["guard"]["overlap_unassessable_parents"] == 0, name
        assert summary["u8"]["overlap_coverage"]["overall"]["qualified_parents"] == 0, name
        assert summary["evaluation_valid"] is False, name
        assert summary["confirmatory"]["pass"] is False, name
        assert summary["confirmatory"]["safety_failures"] == [], name

    assert "ES2009a:missing_guard:C0-overlap" in _u8_phase(
        [*assessed, variants["missing_guard"]]
    )["evaluation_invalid_reasons"]
    assert "ES2009a:missing_paired_score:1" in _u8_phase([*assessed, variants["unmapped_only"]])[
        "evaluation_invalid_reasons"
    ]
    unmapped_detail = _u8_phase([*assessed, variants["unmapped_only"]])["u8"]["overlap_coverage"][
        "parents"
    ][0]
    assert unmapped_detail["qualification_reason"] == "no_mixed_overlap_tokens"
    assert unmapped_detail["excluded_tokens"]["unaligned"] == 1
    assert "ES2009a:invalid_provenance:1" in _u8_phase([*assessed, variants["invalid_provenance"]])[
        "evaluation_invalid_reasons"
    ]
    present_summary = _u8_phase([*assessed, variants["paired_score_present"]])
    assert "ES2009a:grouping_safety_unassessed:C0-overlap" in present_summary[
        "evaluation_invalid_reasons"
    ]
    assert present_summary["u8"]["guard"]["coverage"]["unassessed_coverage_parents"] == 1
    stripped_summary = _u8_phase([*assessed, variants["absent_annotation_count"]])
    stripped_row = stripped_summary["u8"]["overlap_coverage"]["parents"][0]
    assert stripped_row["coverage_status"] == "no_attributable_lexical_tokens"
    assert stripped_row["annotation_tokens"] is None
    assert stripped_row["qualification_reason"] == "no_annotation_tokens"
    assert stripped_summary["evaluation_valid"] is False
    annotation_summary = _u8_phase([*assessed, variants["no_annotation_source"]])
    assert annotation_summary["u8"]["guard"]["coverage"]["no_annotation_source_parents"] == 1
    assert annotation_summary["u8"]["guard"]["coverage"]["unassessed_coverage_parents"] == 1
    assert annotation_summary["u8"]["guard"]["overlap_unassessable_parents"] == 0


def test_overlap_only_case_stays_unassessed_and_cannot_conditionally_pass() -> None:
    clusters = [f"C{index}" for index in range(8)]
    parents = [_overlap_parent(cluster) for cluster in clusters]

    summary = _u8_phase(parents)
    decision = summary["confirmatory"]

    assert summary["execution_completed"] is True
    assert summary["evaluation_valid"] is True
    assert decision["evaluation_valid"] is True
    assert decision["pass"] is False
    assert decision["conditional_support"] is False
    assert decision["result"].startswith("Inconclusive")
    assert decision["n_eligible_clusters"] == 0
    assert decision["safety_failures"] == []
    assert decision["benefit_explained_only_by_unassigned"] is False

    guard = summary["u8"]["guard"]
    assert guard["assessed_parents"] == 8
    assert guard["unassessed_parents"] == 0
    assert guard["severe"] is False
    assert guard["overlap_unassessable_parents"] == 8
    assert guard["coverage"]["grouping_safety_assessed"] is False
    assert guard["coverage"]["unassessed_alignment_parents"] == 8
    assert guard["coverage"]["unassessed_coverage_parents"] == 0
    assert all(row["grouping_safety_assessed"] is False for row in guard["cases"][0]["parents"])

    bounds = decision["sensitivity"]["formed_parent_selection_bounds"]
    assert bounds["N_total"] == 0
    assert bounds["M_total"] == 8
    assert bounds["M_unscorable_total"] == 8
    assert bounds["M_overlap_unassessable_total"] == 8
    assert bounds["equal_cluster_mean_lower"] == pytest.approx(-1.0)
    assert bounds["equal_cluster_mean_upper"] == pytest.approx(1.0)
    assert bounds["fragility"] == "bounds_cross_zero"

    report = summary["u8"]["overlap_coverage"]
    assert report["overall"]["formed_parents"] == 8
    assert report["overall"]["qualified_parents"] == 8
    assert report["overall"]["qualified_parent_rate_of_formed"] == pytest.approx(1.0)
    assert report["overall"]["qualified_parent_rate_of_accepted_nonempty"] == pytest.approx(1.0)
    assert report["overall"]["accepted_char_coverage"] == pytest.approx(1.0)
    assert all(row["overlap_unassessable"] is True for row in report["parents"])


def test_punctuation_only_tokens_are_excluded_not_severe() -> None:
    words = _guard_words([("S", 0, 6400, "sure"), ("S", 8000, 14400, "thing")])
    tokens = [
        _guard_token("t0", "sure", 0, 6400),
        _guard_token("t1", ",", 0, 300),
        _guard_token("t2", "thing", 8000, 14400),
    ]
    single = _arm_guard(
        tokens=tokens,
        units=[_guard_unit("g0", "CURRENT", ["t0", "t1", "t2"], 0, 14400)],
        words=words,
        parent_text="sure, thing",
    )
    split = _arm_guard(
        tokens=tokens,
        units=[
            _guard_unit("g0", "CURRENT", ["t0", "t2"], 0, 14400),
            _guard_unit("g1", "OTHER", ["t1"], 0, 300),
        ],
        words=words,
        parent_text="sure, thing",
    )
    paired = pair_parent_guard(single, split)
    assert paired["failures"] == []
    assert paired["same_speaker"]["stratum"] is True
    assert paired["checked"]["lexical_tokens"] == 2
    assert paired["checked"]["excluded"]["punctuation_only"] == 1
    assert split["same_speaker_stratum"] is True


def test_unknown_grouping_is_assessed_with_zero_concrete_claims() -> None:
    words = _guard_words([("S", 0, 6400, "sure")])
    tokens = [_guard_token("t0", "sure", 0, 6400)]
    single = _arm_guard(
        tokens=tokens,
        units=[_guard_unit("g0", "CURRENT", ["t0"], 0, 6400)],
        words=words,
        parent_text="sure",
    )
    unknown = _arm_guard(
        tokens=tokens,
        units=[_guard_unit("g0", "UNKNOWN", ["t0"], 0, 6400)],
        words=words,
        parent_text="sure",
    )
    paired = pair_parent_guard(single, unknown)
    assert paired["assessed"] is True
    assert paired["failures"] == []
    assert paired["checked"]["concrete_relation_claims"] == 0
    assert paired["checked"]["grouping_safety_assessed"] is True


def test_new_wrong_merge_blocks_adoption_despite_favourable_cluster_means() -> None:
    from experiments.psem_r2_policy.phase import aggregate_phase

    r0_guard = _arm_guard(
        tokens=_ABA_TOKENS,
        units=[_guard_unit("g0", "CURRENT", ["t0", "t1", "t2"], 0, 22400)],
        words=_ABA_WORDS,
        parent_text="alpha bravo charlie",
    )
    r2_guard = _arm_guard(
        tokens=_ABA_TOKENS,
        units=[
            _guard_unit("g0", "CURRENT", ["t0"], 0, 6400),
            _guard_unit("g1", "CURRENT", ["t1", "t2"], 8000, 22400),
        ],
        words=_ABA_WORDS,
        parent_text="alpha bravo charlie",
    )
    failing = pair_parent_guard(r0_guard, r2_guard)
    assert failing["severe"] is True

    parents = [_u8_parent(f"C{index}") for index in range(8)]
    parents[0] = {**_u8_parent("C0"), "parent_id": "C0-0", "guard": failing}
    case = _u8_case(parents)
    summary = aggregate_phase(parents, cases=[case], phase="dev")
    decision = summary["confirmatory"]

    assert decision["cluster_mean_delta"] is not None
    assert decision["cluster_mean_delta"] < 0
    assert decision["improved_cluster_share"] >= 0.7
    assert decision["pass"] is False
    assert decision["conditional_support"] is False
    assert decision["result"] == "Safety failure"
    assert summary["evaluation_valid"] is False
    assert any("wrong_merge:1" in item for item in decision["safety_failures"])
    guard = summary["u8"]["guard"]
    assert guard["severe"] is True
    assert guard["wrong_merge_tokens"] == 1
    assert guard["wrong_merge_chars"] == len("charlie")
    case_guard = guard["cases"][0]
    assert case_guard["wrong_merge_tokens"] == 1
    assert case_guard["wrong_merge_chars"] == len("charlie")
    assert case_guard["parents"][0]["new_wrong_chars"] == len("charlie")
    assert case_guard["parents"][0]["new_wrong_tokens"] == 1
    assert guard["assessed_parents"] == 1
    assert guard["unassessed_parents"] == 7
