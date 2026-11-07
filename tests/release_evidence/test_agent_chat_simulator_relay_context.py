from __future__ import annotations

import json
from copy import deepcopy

import pytest

from puripuly_heart.app.services.agent_chat.modules import MODULES
from puripuly_heart.release_evidence.agent_chat_simulator import __main__ as cli
from puripuly_heart.release_evidence.agent_chat_simulator import runner
from puripuly_heart.release_evidence.agent_chat_simulator.app import ScenarioApp
from puripuly_heart.release_evidence.agent_chat_simulator.relay_context import (
    load_relay_context,
    upstream_context_id,
    validate_relay_context,
)
from puripuly_heart.release_evidence.agent_chat_simulator.scenarios import select_scenarios
from tests.app.test_agent_chat import context_of


def test_manifest_preserves_emitted_prefix_cache_questions_and_options(
    relay_context_manifest, tmp_path
):
    source = tmp_path / "context.json"
    source.write_text(json.dumps(relay_context_manifest), encoding="utf-8")
    loaded = load_relay_context(source, relay_build=relay_context_manifest["relay_build"])
    assert loaded == relay_context_manifest
    loaded["contexts"]["work"]["input_prefix"].clear()
    assert relay_context_manifest["contexts"]["work"]["input_prefix"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(format="guessed-current-markdown"),
        lambda value: value.pop("agent_protocol"),
        lambda value: value.update(agent_protocol=3),
        lambda value: value.update(relay_build="c" * 64),
        lambda value: value.pop("relay_build"),
        lambda value: value.update(bundle_hashes={}),
        lambda value: value["bundle_hashes"].update({"../private-state": "a" * 64}),
        lambda value: value["contexts"].pop("classify"),
        lambda value: value["contexts"]["classify"].pop("questions"),
        lambda value: value["contexts"]["classify"].update(
            questions=[{"instructions": "not a question"}]
        ),
        lambda value: value["contexts"]["work"].update(
            input_prefix=[{"role": "user", "content": "untrusted"}]
        ),
        lambda value: value["contexts"]["work"].update(tool_choice="auto"),
        lambda value: value["contexts"]["work"].update(reasoning={"effort": "unknown"}),
        lambda value: value["contexts"]["work"].update(reasoning={"effort": "medium"}),
        lambda value: value["contexts"]["work"].update(reasoning={"effort": []}),
        lambda value: value["contexts"]["work"].update(stream="true"),
        lambda value: value["contexts"]["social"].update(tools=[{"type": "function"}]),
        lambda value: value["contexts"]["out_of_scope"].pop("instructions"),
    ],
)
def test_manifest_rejects_missing_wrong_or_per_call_fields(relay_context_manifest, mutation):
    build = relay_context_manifest["relay_build"]
    mutation(relay_context_manifest)
    with pytest.raises(ValueError, match="relay context manifest"):
        validate_relay_context(relay_context_manifest, relay_build=build)


@pytest.mark.parametrize("effort", ["low", "high", "xhigh"])
def test_manifest_accepts_current_response_efforts(relay_context_manifest, effort):
    relay_context_manifest["contexts"]["work"]["reasoning"]["effort"] = effort
    assert (
        validate_relay_context(
            relay_context_manifest, relay_build=relay_context_manifest["relay_build"]
        )
        == relay_context_manifest
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manifest_state", ["missing", "wrong_build", "missing_build", "wrong_format", "missing_work"]
)
async def test_live_cli_rejects_context_before_model_or_identity_access(
    relay_context_manifest, manifest_state, monkeypatch, tmp_path
):
    def forbidden(_arguments):
        pytest.fail("Invalid live context must be rejected before constructing models")

    monkeypatch.setattr(cli, "make_models", forbidden)
    arguments = [
        "run",
        "--case",
        "recognition-capture-off",
        "--relay-build",
        relay_context_manifest["relay_build"],
        "--relay",
        "http://127.0.0.1:18791",
        "--broker",
        "http://127.0.0.1:18793",
        "--isolated-staging",
    ]
    if manifest_state != "missing":
        if manifest_state == "wrong_build":
            relay_context_manifest["relay_build"] = "c" * 64
        elif manifest_state == "wrong_format":
            relay_context_manifest["format"] = "historical-not-an-export"
        elif manifest_state == "missing_work":
            del relay_context_manifest["contexts"]["work"]
        elif manifest_state == "missing_build":
            del arguments[3:5]
            monkeypatch.delenv("PURIPULY_SIMULATOR_RELAY_BUILD", raising=False)
        manifest = tmp_path / "context.json"
        manifest.write_text(json.dumps(relay_context_manifest), encoding="utf-8")
        arguments.extend(["--relay-context", str(manifest)])
    else:
        monkeypatch.delenv("PURIPULY_SIMULATOR_RELAY_CONTEXT", raising=False)
    with pytest.raises(ValueError, match="relay.context"):
        await cli.execute(cli.parser().parse_args(arguments))


@pytest.mark.asyncio
async def test_live_runner_rejects_missing_context_before_actor_or_transport():
    def forbidden(_app):
        pytest.fail("Missing live context must be rejected before creating the transport")

    with pytest.raises(ValueError, match="Live attempts require trusted context"):
        await runner.run_attempt(
            select_scenarios(["recognition-capture-off"])[0],
            object(),
            forbidden,
            identity="missing",
        )


@pytest.mark.parametrize(
    ("method", "route", "identity"),
    [
        ("classify", None, "classify"),
        ("respond", "work", "work"),
        ("respond", "unclear", "work"),
        ("respond", "social", "social"),
        ("respond", "out_of_scope", "out_of_scope"),
    ],
)
def test_context_reference_uses_actual_route(method, route, identity):
    assert upstream_context_id(method, {"route": route}) == identity


@pytest.mark.asyncio
async def test_recorded_inputs_and_allow_tools_stay_per_call(relay_context_manifest):
    class Transport:
        def __init__(self):
            self.arguments = []

        async def respond(self, **arguments):
            self.arguments.append(deepcopy(arguments))
            yield {"type": "completed", "response": {"output": []}}

    actual = Transport()
    app = ScenarioApp(select_scenarios(["recognition-capture-off"])[0])
    recording = runner.RecordingTransport(actual, app)
    inputs = [
        {"type": "configuration_update", "reasoning": {"effort": "low"}},
        {"role": "user", "content": "Synthetic app help"},
        {"type": "configuration_update", "reasoning": {"effort": "high"}},
    ]
    for route, allow_tools in (
        ("work", False),
        ("unclear", True),
        ("social", True),
        ("out_of_scope", False),
    ):
        arguments = {
            "input": deepcopy(inputs),
            "route": route,
            "effort": "high",
            "allow_tools": allow_tools,
        }
        async for _ in recording.respond(**arguments):
            pass
        assert actual.arguments[-1] == arguments == recording.calls[-1]["request"]
        assert recording.calls[-1]["upstream_context_id"] == (
            "work" if route == "unclear" else route
        )
        assert "relay_contexts" not in actual.arguments[-1]
    evidence = runner.invocation_evidence(
        recording.calls, [], relay_contexts=relay_context_manifest
    )
    assert evidence["relay_contexts"] == relay_context_manifest
    assert [call["request"]["allow_tools"] for call in evidence["M_invocations"]] == [
        False,
        True,
        True,
        False,
    ]
    assert all(call["request"]["input"] == inputs for call in evidence["M_invocations"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context_mode", "expected_modules"),
    [
        ("selected", ["core", "recognition", "accounts"]),
        ("full", list(MODULES)),
    ],
)
async def test_smoke_evidence_matches_selected_context_and_full_comparison(
    context_mode, expected_modules, tmp_path
):
    output = tmp_path / f"{context_mode}.json"
    arguments = cli.parser().parse_args(
        [
            "smoke",
            "--case",
            "explain-recognition-options",
            "--context-mode",
            context_mode,
            "--output",
            str(output),
        ]
    )
    assert await cli.execute(arguments) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    attempt = report["cases"][0]["attempt"]
    assert report["context_mode"] == attempt["context_mode"] == context_mode
    decisions = attempt["effective_decisions"]
    assert [decision["context_relation"] for decision in decisions] == ["new", "continue"]
    assert all(decision["context_modules"] == expected_modules for decision in decisions)
    assert all(decision["context_mode"] == context_mode for decision in decisions)
    assert all({"context_source", "context_reason"} <= decision.keys() for decision in decisions)
    assert [call["decision"] for call in attempt["D_invocations"]] == decisions
    assert [event for event in attempt["events"] if event.get("kind") == "decision"] == decisions
    for call in attempt["M_invocations"]:
        context = context_of(call["request"])
        assert context["modules"] == expected_modules
        assert [module for module in MODULES if module in context] == expected_modules


@pytest.mark.asyncio
async def test_trusted_context_reaches_only_judge_not_actor_or_client(
    relay_context_manifest, monkeypatch
):
    judge_payloads = []
    original = runner.judge

    async def observe_judge(models, payload, calls):
        judge_payloads.append(deepcopy(payload))
        return await original(models, payload, calls)

    class IsolatedActor(runner.ControlledModels):
        async def complete(self, **arguments):
            if arguments["role"] == "user":
                assert "TRUSTED_CONTEXT_ONLY_NOT_FOR_TERRA" not in json.dumps(arguments)
                assert "relay_contexts" not in arguments["payload"]
            return await super().complete(**arguments)

    monkeypatch.setattr(runner, "judge", observe_judge)
    result = await runner.run_attempt(
        select_scenarios(["recognition-capture-off"])[0],
        IsolatedActor(),
        runner.controlled_transport,
        identity="controlled-isolation",
        controlled=True,
        relay_contexts=relay_context_manifest,
    )
    assert result["relay_contexts"] == relay_context_manifest
    assert judge_payloads and judge_payloads[0]["relay_contexts"] == relay_context_manifest
    assert result["D_invocations"] and result["M_invocations"]
    for call in result["D_invocations"] + result["M_invocations"]:
        assert call["upstream_context_id"] in relay_context_manifest["contexts"]
        assert "TRUSTED_CONTEXT_ONLY_NOT_FOR_TERRA" not in json.dumps(call["request"])
    report = runner.report_header(
        relay="controlled",
        build=relay_context_manifest["relay_build"],
        controlled=True,
        relay_contexts=relay_context_manifest,
    )
    assert report["controlled_smoke"] and not report["live_pass"]
    assert report["relay_contexts"] == relay_context_manifest


@pytest.mark.asyncio
async def test_blocked_grounding_evidence_is_not_an_invalid_judgment(monkeypatch):
    async def unavailable(*_arguments):
        return {"status": "blocked", "code": "upstream_context_missing", "revision": "synthetic"}

    monkeypatch.setattr(runner, "judge", unavailable)
    result = await runner.run_attempt(
        select_scenarios(["recognition-capture-off"])[0],
        runner.ControlledModels(),
        runner.controlled_transport,
        identity="controlled-missing-grounding",
        controlled=True,
    )
    assert result["status"] == "blocked"
    assert result["error"] == "upstream_context_missing"
    assert result["judgment"]["status"] == "blocked"
