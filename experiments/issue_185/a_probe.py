from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid5

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.orchestrator.configuration import TranslationRuntimeConfig
from puripuly_heart.core.orchestrator.translation_turn import TranslationTurnRequest
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness

NAMESPACE = UUID("07c6070a-c90b-4eb4-953a-37025dd905a1")


async def run() -> list[dict[str, object]]:
    loop = asyncio.get_running_loop()
    origin = loop.time()
    rows: list[dict[str, object]] = []
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    a_id = uuid5(NAMESPACE, "speech-a")
    b_id = uuid5(NAMESPACE, "speech-b")

    def record(event: str, **fields: object) -> None:
        rows.append({"event": event, "t_us": round((loop.time() - origin) * 1_000_000), **fields})

    class GatedProvider:
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
        ) -> Translation:
            record(
                "provider_invoked",
                parent=str(utterance_id),
                text=text,
                context=context,
                target=target_language,
                prompt_sha256=hashlib.sha256(system_prompt.encode()).hexdigest(),
                provider_generation=harness.translation_requests.provider_generation,
            )
            if utterance_id == a_id:
                await first_release.wait()
            else:
                second_started.set()
            record("provider_ready", parent=str(utterance_id))
            return Translation(
                utterance_id=utterance_id,
                text=f"translated-{text}",
                source_text=text,
                source_language=source_language,
                target_language=target_language,
                channel="self",
            )

        async def close(self) -> None:
            pass

    harness = compose_translation_test_harness(
        llm=GatedProvider(),
        osc=RecordingOscQueue(),
        clock=SystemClock(),
        source_language="en",
        target_language="ja",
        self_target_languages=("ja",),
        system_prompt="${sourceName}|${targetName}",
        context_max_entries=3,
    )
    turns = harness.translation_turns
    original_admitted = turns.on_parent_admitted
    original_output = turns.output
    assert original_admitted is not None and original_output is not None

    async def admitted(children):
        await original_admitted(children)
        child = children[0]
        prepared = harness.self_owner._admitted_requests[child.utterance_id]
        record(
            "ordered_admission_complete",
            parent=str(child.parent_utterance_id),
            prepared_context=prepared.context,
            prompt_sha256=hashlib.sha256(prepared.system_prompt.encode()).hexdigest(),
            provider_generation=harness.translation_requests.provider_generation,
            configuration_revision=child.config_snapshot.revision,
        )

    class OutputObservation:
        async def submit_translation_output(self, submission):
            record(
                "ordered_output_handoff",
                parent=str(submission.parent_utterance_id),
                outcome=submission.outcome,
            )
            return await original_output.submit_translation_output(submission)

    turns.on_parent_admitted = admitted
    turns.output = OutputObservation()
    snapshot = harness.configuration.snapshot()
    assert isinstance(snapshot.value, TranslationRuntimeConfig)

    def request(parent: UUID, text: str) -> TranslationTurnRequest:
        return TranslationTurnRequest(
            transcript=Transcript(parent, text, is_final=True, channel="self"),
            source="Mic",
            turn_kind="self",
            target_languages=("ja",),
            config_snapshot=snapshot,
        )

    await harness.start()
    try:
        await turns.submit(request(a_id, "synthetic first"))
        await turns.submit(request(b_id, "synthetic second"))
        await asyncio.wait_for(second_started.wait(), timeout=3)
        assert not first_release.is_set()
        assert not any(row["event"] == "ordered_output_handoff" for row in rows)
        await asyncio.sleep(0.02)
        assert not any(row["event"] == "ordered_output_handoff" for row in rows)
        first_release.set()
        await asyncio.wait_for(turns.wait_for_idle(), timeout=3)
        assert [row["parent"] for row in rows if row["event"] == "ordered_output_handoff"] == [
            str(a_id),
            str(b_id),
        ]
        assert [row["parent"] for row in rows if row["event"] == "provider_invoked"] == [
            str(a_id),
            str(b_id),
        ]
        admissions = [row for row in rows if row["event"] == "ordered_admission_complete"]
        invocations = [row for row in rows if row["event"] == "provider_invoked"]
        assert admissions[1]["t_us"] <= invocations[1]["t_us"]
        assert admissions[0]["prepared_context"] == invocations[0]["context"]
        assert admissions[1]["prepared_context"] == invocations[1]["context"]
        assert [row["prompt_sha256"] for row in admissions] == [
            row["prompt_sha256"] for row in invocations
        ]
        assert [row["provider_generation"] for row in admissions] == [
            row["provider_generation"] for row in invocations
        ]
        assert [row["configuration_revision"] for row in admissions] == [
            snapshot.revision,
            snapshot.revision,
        ]
        assert "synthetic first" in str(invocations[1]["context"])
        assert invocations[1]["t_us"] < next(
            row["t_us"]
            for row in rows
            if row["event"] == "provider_ready" and row["parent"] == str(a_id)
        )
        return rows
    finally:
        first_release.set()
        await harness.stop()


if __name__ == "__main__":
    trace = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "profile": {
            "source_language": "en",
            "target_languages": ["ja"],
            "context_max_entries": 3,
            "system_prompt_template": "${sourceName}|${targetName}",
            "first_provider_gate_s": 0.02,
        },
        "scenario": (
            "gated Self single-target synthetic speech; real Self admission/request/output "
            "owners; no device/provider/display"
        ),
        "rows": asyncio.run(run()),
    }
    destination = Path(__file__).with_name("a_trace.json")
    destination.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(trace['rows'])} events; {destination.name}")
