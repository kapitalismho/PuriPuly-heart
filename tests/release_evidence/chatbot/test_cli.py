from __future__ import annotations

import json

from benchmarks.chatbot import __main__ as cli
from benchmarks.chatbot import runner


async def test_completed_controlled_dialogues_do_not_bypass_run_calibration(
    monkeypatch, tmp_path, relay_context_manifest
):
    class OfflineModels(runner.ControlledModels):
        attempts = []

        async def prepare(self):
            return None

        async def close(self):
            return None

    async def controlled_attempt(case, models, factory, *, identity, context_mode, relay_contexts):
        return await runner.run_attempt(
            case,
            models,
            runner.controlled_transport,
            identity=identity,
            controlled=True,
            context_mode=context_mode,
            relay_contexts=relay_contexts,
        )

    monkeypatch.setattr(cli, "make_models", lambda *args, **kwargs: OfflineModels())
    monkeypatch.setattr(cli, "run_attempt", controlled_attempt)
    manifest = tmp_path / "relay-context.json"
    output = tmp_path / "controlled-run.json"
    cli.write_json(manifest, relay_context_manifest)
    args = cli.parser().parse_args(
        [
            "run",
            "--case",
            "recognition-capture-off",
            "--subscription-only",
            "--isolated-staging",
            "--broker",
            "http://127.0.0.1:18793",
            "--relay",
            "http://127.0.0.1:18791/v1/respond",
            "--relay-build",
            relay_context_manifest["relay_build"],
            "--relay-context",
            str(manifest),
            "--output",
            str(output),
        ]
    )

    assert await cli.execute(args) == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scenario_status"] == "passed"
    assert report["status"] == "blocked_calibration"
    assert report["calibration"]["status"] == "blocked"
    assert report["live_pass"] is False
    assert report["access_attempts"] == []
    assert report["relay_contexts"] == relay_context_manifest
    assert report["summary"]["saved_runs"] == 3
    assert report["cases"][0]["status"] == "passed"
    assert all(
        attempt["controlled"] for run in report["cases"][0]["runs"] for attempt in run["attempts"]
    )
    journal = output.with_suffix(".attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(journal) == 3
    assert {json.loads(line)["id"] for line in journal} == {
        f"recognition-capture-off/run-{number}/attempt-1" for number in range(1, 4)
    }
