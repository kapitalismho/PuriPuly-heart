"""Issue #185 assembled owner-boundary probe; synthetic audio and providers only.

Run at the repository root: uv run python experiments/issue_185/integration_probe.py
The #180 module is imported for its corrected scenarios, never its historical
revision guard, main entrypoint, or obsolete Self-translation-serial assertion.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
PRIOR_PATH = ROOT / "experiments/issue_180/probe.py"
SPEC = importlib.util.spec_from_file_location("issue180_corrected_probe", PRIOR_PATH)
assert SPEC is not None and SPEC.loader is not None
prior = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prior
SPEC.loader.exec_module(prior)

from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.llm.fallback_racing import FallbackRacingLLMProvider, LLMProviderAttempt
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.orchestrator.translation_turn import TranslationTurnRequest
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.translation_owners import compose_translation_test_harness

NAMESPACE = UUID("b069ce89-a47f-495d-89e3-e752e13c8105")
SOURCE_PATHS = (
    "src/puripuly_heart/core/orchestrator/configuration.py",
    "src/puripuly_heart/core/orchestrator/self_translation_channel.py",
    "src/puripuly_heart/core/translation_policy.py",
    "src/puripuly_heart/core/runtime/self_capture.py",
    "src/puripuly_heart/core/runtime/peer_channel.py",
    "src/puripuly_heart/core/runtime/output_batch.py",
    "src/puripuly_heart/core/stt/session_projection.py",
    "experiments/issue_180/probe.py",
    "experiments/issue_185/integration_probe.py",
    "src/puripuly_heart/core/orchestrator/translation_turn.py",
    "src/puripuly_heart/core/llm/fallback_racing.py",
    "src/puripuly_heart/core/llm/provider.py",
    "src/puripuly_heart/core/runtime/gpu_asr.py",
    "src/puripuly_heart/providers/stt/local_gpu.py",
    "src/puripuly_heart/providers/stt/local_qwen_sherpa.py",
    "src/puripuly_heart/core/stt/scoped_engine.py",
    "src/puripuly_heart/core/runtime/provider_handle.py",
    "src/puripuly_heart/core/runtime/output.py",
    "src/puripuly_heart/core/overlay/presenter.py",
    "tests/helpers/translation_owners.py",
)


def event(rows: list[dict[str, Any]], scenario: str, name: str, **fields: Any) -> dict[str, Any]:
    return prior.one_row(rows, scenario, name, **fields)


def check_reused_scenarios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios = {row["scenario"] for row in rows}
    self_intervals: dict[str, dict[str, int]] = {}
    for scenario in ("self_successive_immediate_terminal", "self_successive_delayed_terminal"):
        first_ready = event(rows, scenario, "translation_completion", source_text="turn-1")
        second_start = event(rows, scenario, "translation_start", source_text="turn-2")
        second_ready = event(rows, scenario, "translation_completion", source_text="turn-2")
        assert second_start["t_us"] < first_ready["t_us"]
        assert second_ready["t_us"] < first_ready["t_us"]
        originals = [
            row
            for row in rows
            if row["scenario"] == scenario
            and row["event"] == "application_receipt_ready"
            and row["event_type"] == "self_transcript_final"
        ]
        translated = [
            row
            for row in rows
            if row["scenario"] == scenario
            and row["event"] == "application_receipt_ready"
            and row["event_type"] == "translation_final"
        ]
        assert len(originals) == len(translated) == 2
        assert translated[0]["t_us"] <= translated[1]["t_us"]
        first_source = event(rows, scenario, "self_source_available", turn=1)["source_id"]
        second_source = event(rows, scenario, "self_source_available", turn=2)["source_id"]
        second_original = event(
            rows,
            scenario,
            "application_receipt_ready",
            occupant=second_source,
            event_type="self_transcript_final",
        )
        batch = event(
            rows, scenario, "destination_batch_insert", parent_id=second_original["publication_id"]
        )
        assert batch["scope"] == "self:original" and batch["active"]
        assert second_original["t_us"] < first_ready["t_us"]
        assert [
            (row["samples"], row["context_only"])
            for row in rows
            if row["scenario"] == scenario
            and row["event"] == "provider_write"
            and row.get("turn") == 2
        ] == [(2, True), (8, False)]
        first_release = event(
            rows,
            scenario,
            "destination_batch_release",
            parent_id=first_source,
            scope="self",
        )
        assert first_release["scope"] == "self"
        assert first_release["disposition"] == "applied"
        second_apply = event(
            rows,
            scenario,
            "application_receipt_ready",
            occupant=second_source,
            event_type="translation_final",
        )
        ordered_eligible_us = max(second_ready["t_us"], first_release["t_us"])
        assert first_release["t_us"] <= second_apply["t_us"]
        self_intervals[scenario] = {
            "b_provider_dispatch_to_ready": second_ready["t_us"] - second_start["t_us"],
            "b_computation_ready_to_ordered_application": second_apply["t_us"]
            - second_ready["t_us"],
            "b_predecessor_order_release": first_release["t_us"],
            "b_order_eligible_at": ordered_eligible_us,
            "b_eligible_to_internal_application": second_apply["t_us"] - ordered_eligible_us,
        }
    overlap = "stt_delayed_terminal_overlap"
    assert (
        event(rows, overlap, "provider_begin", turn=2)["t_ms"]
        < event(
            rows,
            overlap,
            "provider_terminal_receipt",
            turn=1,
        )["t_ms"]
    )
    assert event(rows, "stt_delayed_terminal", "provider_begin", turn=2)["t_ms"] == 350
    assert [
        row["t_ms"]
        for row in rows
        if row["scenario"] == "output_burst" and row["event"] == "application_receipt_ready"
    ] == [0, 0, 1000, 2000, 3000]
    assert (
        event(
            rows,
            "output_protection_release",
            "application_receipt_ready",
            event_type="peer_transcript_final",
        )["t_ms"]
        == 400
    )
    paced = event(rows, "output_real_clock_pacing", "real_clock_bound")
    assert 900_000 <= paced["ready_to_application_us"] <= 1_500_000
    assert 0 <= paced["eligibility_to_application_us"] <= 100_000
    return {
        "scenarios": sorted(scenarios),
        "rows": len(rows),
        "self_intervals_us": self_intervals,
        "peer_real_clock_us": {
            "ready_to_application": paced["ready_to_application_us"],
            "eligible_recheck_to_application": paced["eligibility_to_application_us"],
        },
    }


async def run_race_under_translation_owner() -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    origin = loop.time()
    rows: list[dict[str, Any]] = []
    a_id = uuid5(NAMESPACE, "a")
    b_id = uuid5(NAMESPACE, "b")
    c_id = uuid5(NAMESPACE, "c-direct-capacity-control")
    a_release = asyncio.Event()
    loser_release = asyncio.Event()
    b_winner_started = asyncio.Event()
    b_winner_release = asyncio.Event()
    b_loser_started = asyncio.Event()
    c_started = asyncio.Event()

    def record(name: str, **values: Any) -> None:
        rows.append({"event": name, "t_us": round((loop.time() - origin) * 1_000_000), **values})

    class Attempt:
        def __init__(self, index: int) -> None:
            self.index = index
            self.calls: list[tuple[str, int]] = []

        async def translate(
            self,
            *,
            utterance_id: UUID,
            text: str,
            system_prompt: str,
            source_language: str,
            target_language: str,
            context: str = "",
            scene_participant_count: int | None = None,
            max_output_tokens: int | None = None,
        ) -> Translation:
            label = "A" if utterance_id == a_id else "B" if utterance_id == b_id else "C"
            self.calls.append((label, self.index))
            record("attempt_start", text=label, index=self.index, context=context)
            if label == "B" and self.index == 0:
                b_loser_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    record("b_loser_cancelled")
                    while not loser_release.is_set():
                        try:
                            await loser_release.wait()
                        except asyncio.CancelledError:
                            continue
                    record("b_loser_finished")
                    raise
            if label == "A":
                await a_release.wait()
            elif label == "B" and self.index == 1:
                b_winner_started.set()
                await b_winner_release.wait()
            elif label == "C":
                c_started.set()
            record("attempt_ready", text=label, index=self.index)
            return Translation(
                utterance_id=utterance_id,
                text=f"translated-{text}",
                source_text=text,
                source_language=source_language,
                target_language=target_language,
                channel="self",
            )

        async def close(self) -> None:
            record("attempt_close", index=self.index)

    primary, fallback = Attempt(0), Attempt(1)
    race = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(primary),
            LLMProviderAttempt(fallback, start_after_ms=0),
        ),
        loser_grace_ms=10,
    )
    bounded = SemaphoreLLMProvider(inner=race, semaphore=asyncio.Semaphore(2))
    harness = compose_translation_test_harness(
        llm=bounded,
        osc=prior.NullChatbox(),
        clock=SystemClock(),
        source_language="en",
        target_language="ja",
        self_target_languages=("ja",),
        context_max_entries=3,
    )
    turns = harness.translation_turns
    original_output = turns.output
    original_admitted = turns.on_parent_admitted
    assert original_admitted is not None

    async def observe_admission(children: Any) -> None:
        await original_admitted(children)
        child = children[0]
        prepared = harness.self_owner._admitted_requests[child.utterance_id]
        record(
            "ordered_admission",
            text="A" if child.parent_utterance_id == a_id else "B",
            context=prepared.context,
            config_revision=child.config_snapshot.revision,
            provider_generation=harness.translation_requests.provider_generation,
        )

    turns.on_parent_admitted = observe_admission
    assert original_output is not None

    class ObserveOutput:
        async def submit_translation_output(self, submission: Any) -> Any:
            record(
                "ordered_handoff",
                parent=str(submission.parent_utterance_id),
                outcome=submission.outcome,
            )
            return await original_output.submit_translation_output(submission)

    turns.output = ObserveOutput()
    snapshot = harness.configuration.snapshot()

    def request(identity: UUID, text: str) -> TranslationTurnRequest:
        return TranslationTurnRequest(
            transcript=Transcript(identity, text, is_final=True, channel="self"),
            source="Mic",
            turn_kind="self",
            target_languages=("ja",),
            config_snapshot=snapshot,
        )

    await harness.start()
    capacity_task: asyncio.Task[Translation] | None = None
    try:
        await turns.submit(request(a_id, "synthetic first"))
        await turns.submit(request(b_id, "synthetic second"))
        await asyncio.wait_for(b_loser_started.wait(), 3)
        await asyncio.wait_for(b_winner_started.wait(), 3)
        b_winner_release.set()
        for _ in range(1000):
            if any(row["event"] == "attempt_ready" and row["text"] == "B" for row in rows):
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("B winner never became ready")
        assert not any(row["event"] == "attempt_ready" and row["text"] == "A" for row in rows)
        assert not any(row["event"] == "ordered_handoff" for row in rows)
        assert bounded.semaphore.locked()
        capacity_task = asyncio.create_task(
            bounded.translate(
                utterance_id=c_id,
                text="synthetic third",
                system_prompt="fixture",
                source_language="en",
                target_language="ja",
            )
        )
        await asyncio.sleep(0.03)
        assert not c_started.is_set() and bounded.semaphore.locked()
        assert any(row["event"] == "b_loser_cancelled" for row in rows)
        a_release.set()
        await asyncio.wait_for(c_started.wait(), 3)
        assert not loser_release.is_set() and bounded.semaphore.locked()
        await asyncio.wait_for(capacity_task, 3)
        await asyncio.wait_for(turns.wait_for_idle(), 3)
        assert [row["parent"] for row in rows if row["event"] == "ordered_handoff"] == [
            str(a_id),
            str(b_id),
        ]
        assert [
            (row["text"], row["index"])
            for row in rows
            if row["event"] == "attempt_start" and row["text"] == "B"
        ] == [("B", 0), ("B", 1)]
        assert next(
            row["t_us"] for row in rows if row["event"] == "attempt_ready" and row["text"] == "B"
        ) < next(
            row["t_us"] for row in rows if row["event"] == "attempt_ready" and row["text"] == "A"
        )
        for text in ("A", "B"):
            admission = next(
                row for row in rows if row["event"] == "ordered_admission" and row["text"] == text
            )
            start = next(
                row
                for row in rows
                if row["event"] == "attempt_start" and row["text"] == text and row["index"] == 0
            )
            assert admission["t_us"] <= start["t_us"]
            assert admission["context"] == start["context"]
            assert admission["config_revision"] == snapshot.revision
            assert (
                admission["provider_generation"] == harness.translation_requests.provider_generation
            )
        assert "synthetic first" in next(
            row["context"]
            for row in rows
            if row["event"] == "ordered_admission" and row["text"] == "B"
        )
        result = {
            "scenario": "Self A/B production translation owner + racing provider + semaphore(2), direct C capacity control",
            "rows": rows,
            "attempt_counts": {
                text: sum(
                    source_text == text
                    for provider in (primary, fallback)
                    for source_text, _ in provider.calls
                )
                for text in ("A", "B", "C")
            },
        }
    finally:
        a_release.set()
        b_winner_release.set()
        loser_release.set()
        if capacity_task is not None:
            await asyncio.gather(capacity_task, return_exceptions=True)
        await harness.stop()
    assert any(row["event"] == "b_loser_finished" for row in rows)
    assert len([row for row in rows if row["event"] == "attempt_close"]) == 2
    assert not bounded.semaphore.locked()
    return result


async def main() -> None:
    rows = [
        *(
            await prior.run_self_end_to_end(
                scenario="self_first_isolated", delayed_a_terminal=False, include_b=False
            )
        ),
        *(
            await prior.run_self_end_to_end(
                scenario="self_successive_immediate_terminal",
                delayed_a_terminal=False,
                include_b=True,
            )
        ),
        *(
            await prior.run_self_end_to_end(
                scenario="self_successive_fast_translation",
                delayed_a_terminal=False,
                include_b=True,
                a_translation_delay_s=0.0,
            )
        ),
        *(
            await prior.run_self_end_to_end(
                scenario="self_successive_delayed_terminal", delayed_a_terminal=True, include_b=True
            )
        ),
        *(await prior.run_stt_case(False)),
        *(await prior.run_stt_case(True)),
        *(await prior.run_stt_case(True, allows_sealed_turn_overlap=True)),
        *(await prior.run_output_control()),
        *(await prior.run_output_burst()),
        *(await prior.run_output_protection()),
        *(await prior.run_output_head_of_line()),
        *(await prior.run_real_clock_pacing()),
    ]
    summary = check_reused_scenarios(rows)
    interaction = await run_race_under_translation_owner()
    result = {
        "status": "passed",
        "baseline_revision": "78d90ca9722d3c88e05448bbe7c958892c5b11ec",
        "ambient_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": prior.np.__version__,
            "soxr": version("soxr"),
            "sounddevice": version("sounddevice"),
            "janus": version("janus"),
            "onnxruntime": version("onnxruntime"),
        },
        "configuration": {
            "synthetic_audio_hz": 16000,
            "self_targets": ["ja"],
            "translation_mode": "single-target",
            "low_latency_mode": False,
            "self_provider_delays_s": {"A": 0.18, "B": 0.04},
            "racing_attempts_per_operation": 2,
            "racing_loser_grace_ms": 10,
            "racing_semaphore_capacity": 2,
            "peer_replacement_policy_ms": 1000,
            "post_end_grace_policy_ms": 400,
        },
        "summary": summary,
        "owner_interaction": interaction,
        "scenario_rows": rows,
        "scope": "controlled production owners; no microphone, GPU native inference, network, renderer or HMD; monotonic-process application receipt is not physical display",
    }
    destination = Path(__file__).with_name("integration_trace.json")
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "scenarios": len(summary["scenarios"]),
                "rows": len(rows),
                "interaction_rows": len(interaction["rows"]),
                "trace": str(destination),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
