from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from html.parser import HTMLParser

import pytest

from puripuly_heart.release_evidence.agent_chat_simulator.html_report import render_html


@dataclass
class Element:
    tag: str
    attrs: dict[str, str | None] = field(default_factory=dict)
    children: list = field(default_factory=list)

    def text(self):
        return "".join(
            child.text() if isinstance(child, Element) else child for child in self.children
        )

    def find(self, **attrs):
        found = []
        for child in self.children:
            if isinstance(child, Element):
                if all(
                    child.attrs.get(key.rstrip("_").replace("_", "-")) == value
                    for key, value in attrs.items()
                ):
                    found.append(child)
                found.extend(child.find(**attrs))
        return found


class Document(HTMLParser):
    def __init__(self, content):
        super().__init__(convert_charrefs=True)
        self.root = Element("root")
        self.stack = [self.root]
        self.feed(content)
        self.close()

    def handle_starttag(self, tag, attrs):
        element = Element(tag, dict(attrs))
        self.stack[-1].children.append(element)
        if tag not in {"meta", "br", "hr", "input", "img", "link", "wbr"}:
            self.stack.append(element)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def report(case_id="fixture"):
    return {
        "format": "puri-agent-simulator-1",
        "campaign_id": "one-fresh-campaign",
        "source_tree": "candidate-source",
        "relay_build": "emitted-build",
        "campaign_plan_sha256": "frozen-plan",
        "judge_revision": "rubric-revision",
        "judge_model": "gpt-6.1-sol",
        "judge_effort": "xhigh",
        "agent_protocol": 3,
        "controlled_smoke": False,
        "relay_contexts": {"contexts": {"work": {"instructions": "not-public-context"}}},
        "advisory": ["tone"],
        "limits": {"logical_runs_per_case": 3},
        "status": "incomplete",
        "live_pass": False,
        "planned_scenarios": 2,
        "planned_runs": 6,
        "saved_scenarios": 999,
        "saved_runs": 999,
        "cases": [
            {
                "id": case_id,
                "status": "invalid_judgment",
                "runs": [
                    {
                        "id": f"{case_id}/run-1",
                        "status": "invalid_judgment",
                        "attempts": [
                            {
                                "id": f"{case_id}/run-1/attempt-1",
                                "status": "invalid_judgment",
                                "error": "judge citation invalid",
                                "transcript": [
                                    {
                                        "turn": 1,
                                        "role": "user",
                                        "text": "마이크가 안 돼요.\n<문자 그대로> & 확인",
                                    },
                                    {"turn": 1, "role": "assistant", "text": "먼저 상태를 볼게."},
                                    {"turn": 2, "role": "notice", "text": "[app notice] 승인 결과"},
                                    {
                                        "turn": 2,
                                        "role": "assistant",
                                        "text": "승인 후 이어지는 첫 문장.",
                                    },
                                    {
                                        "turn": 2,
                                        "role": "assistant",
                                        "text": "같은 턴의 두 번째 이어쓰기.",
                                    },
                                ],
                                "actions": [
                                    {
                                        "turn": 2,
                                        "action": "browser_step",
                                        "target": "create_key",
                                        "text": "화면 지침을 그대로 복사한 조작 증거",
                                        "provider": "openrouter",
                                        "result": {
                                            "status": "failed",
                                            "code": "synthetic_page_blocked",
                                        },
                                    }
                                ],
                                "code": {"accepted_state": True, "prohibitions_pass": False},
                                "D_invocations": [
                                    {
                                        "id": 7,
                                        "turn": 1,
                                        "status": "completed",
                                        "source": "fallback",
                                        "response": {
                                            "answers": [{"name": "route", "choice": "work"}]
                                        },
                                        "effective_route": "unclear",
                                        "effective_effort": "medium",
                                        "fallback_code": "invalid_classification",
                                    }
                                ],
                                "judgment": {
                                    "status": "invalid",
                                    "errors": ["bad quote"],
                                    "rejudgments": 1,
                                    "criteria": [
                                        {
                                            "name": "grounded",
                                            "verdict": "fail",
                                            "turn": 2,
                                            "quote": "승인 후 이어지는 첫 문장.",
                                        }
                                    ],
                                    "routing": [
                                        {
                                            "call_id": 7,
                                            "route": "fail",
                                            "effort": "pass",
                                            "quote": "마이크가 안 돼요.",
                                        }
                                    ],
                                },
                                "model_calls": [
                                    {
                                        "role": "judge",
                                        "rejudgment": 0,
                                        "value": {
                                            "criteria": [
                                                {
                                                    "name": "grounded",
                                                    "verdict": "fail",
                                                    "quote": "invalid-first-quote",
                                                }
                                            ]
                                        },
                                    },
                                    {
                                        "role": "judge",
                                        "rejudgment": 1,
                                        "value": {
                                            "criteria": [
                                                {
                                                    "name": "grounded",
                                                    "verdict": "fail",
                                                    "quote": "invalid-second-quote",
                                                }
                                            ]
                                        },
                                    },
                                ],
                                "archive": [],
                            }
                        ],
                    },
                    {
                        "id": f"{case_id}/run-2",
                        "status": "blocked",
                        "attempts": [],
                        "missing_prerequisites": ["Sol access unavailable"],
                    },
                ],
            }
        ],
    }


def save(tmp_path, value, name="report.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def render(tmp_path, value):
    source = save(tmp_path, value)
    output = tmp_path / "latest.html"
    result = render_html([source], output)
    return Document(output.read_text(encoding="utf-8")).root, result, output


def test_dialogue_continuations_controls_verdicts_and_all_attempts_survive(tmp_path):
    value = report()
    run = value["cases"][0]["runs"][0]
    first = run["attempts"][0]
    historical = deepcopy(first)
    historical.update(
        id="fixture/run-1/earlier-attempt", status="blocked", error="earlier timeout", transcript=[]
    )
    run["attempts"].insert(0, historical)
    harness = {
        "id": "fixture/run-3",
        "status": "harness_error",
        "attempts": [
            {
                "id": "fixture/run-3/attempt-1",
                "status": "harness_error",
                "error": "invalid target",
                "harness_error": {"code": "invalid_user_action"},
            }
        ],
    }
    value["cases"][0]["runs"].append(harness)
    root, result, _ = render(tmp_path, value)
    attempts = root.find(data_kind="attempt")
    assert [(item.attrs["data-id"], item.attrs["data-status"]) for item in attempts] == [
        (historical["id"], "blocked"),
        (first["id"], "invalid_judgment"),
        (harness["attempts"][0]["id"], "harness_error"),
    ]
    dialogue = attempts[1].find(data_kind="dialogue")[0]
    messages = [item for item in dialogue.find() if "data-sequence" in item.attrs]
    assert [item.find(class_="literal")[0].text() for item in messages] == [
        item["text"] for item in first["transcript"]
    ]
    assert [item.attrs["data-role"] for item in messages] == [
        "user",
        "assistant",
        "notice",
        "assistant",
        "assistant",
    ]
    assert len(dialogue.find(data_role="control")) == 1
    assert dialogue.text().index("create_key") < dialogue.text().index("[app notice]")
    assert "earlier timeout" in attempts[0].text()
    assert "invalid target" in attempts[2].text()
    code = attempts[1].find(data_kind="code-gates")[0].text()
    rubric = attempts[1].find(data_kind="sol-rubric")[0].text()
    actual = attempts[1].find(data_kind="d-actual")[0].text()
    routing = attempts[1].find(data_kind="d-judgment")[0].text()
    assert '"accepted_state": true' in code and '"prohibitions_pass": false' in code
    assert "grounded" in rubric and "fail" in rubric and "bad quote" in rubric
    assert "invalid_classification" in actual and "unclear" in actual and "work" in actual
    assert "fail" in routing and "pass" in routing and "마이크가 안 돼요." in routing
    evidence = attempts[1].find(data_kind="action-evidence")[0]
    assert evidence.tag == "details" and "open" not in evidence.attrs
    assert "synthetic_page_blocked" in evidence.text()
    assert "invalid-first-quote" in evidence.text() and "invalid-second-quote" in evidence.text()
    assert result["coverage"] == {
        "listed_scenarios": 1,
        "saved_scenarios": 1,
        "listed_runs": 3,
        "saved_runs": 2,
        "attempts": 3,
        "planned_scenarios": 2,
        "planned_runs": 6,
    }
    assert all(item.attrs["data-status"] != "passed" for item in root.find(data_kind="run"))


def test_same_campaign_shards_have_stable_output_hashes_navigation_and_coverage(tmp_path):
    first, second = report("one"), report("two")
    second["cases"][0].update(status="passed", successes=2)
    second["cases"][0]["runs"][0]["status"] = "passed"
    second["cases"][0]["runs"][0]["attempts"][0]["status"] = "passed"
    paths = [save(tmp_path, first, "first.json"), save(tmp_path, second, "second.json")]
    originals = [path.read_bytes() for path in paths]
    a, b = tmp_path / "a.html", tmp_path / "b.html"
    result = render_html(paths, a)
    render_html(paths, b)
    assert a.read_bytes() == b.read_bytes()
    assert [path.read_bytes() for path in paths] == originals
    root = Document(a.read_text(encoding="utf-8")).root
    text = root.text()
    for path, raw in zip(paths, originals):
        assert path.name in text and hashlib.sha256(raw).hexdigest() in text
    for identity in (
        "one-fresh-campaign",
        "candidate-source",
        "emitted-build",
        "frozen-plan",
        "rubric-revision",
        "gpt-6.1-sol",
        "xhigh",
    ):
        assert identity in root.find(data_kind="campaign")[0].text()
    assert result["coverage"]["planned_scenarios"] == 4
    assert result["coverage"]["saved_scenarios"] == 2
    assert result["coverage"]["saved_runs"] == 2
    assert result["coverage"]["attempts"] == 2
    targets = {item.attrs["id"] for item in root.find() if "id" in item.attrs}
    links = [item for item in root.find() if item.tag == "a"]
    assert links and all(
        item.attrs["href"].startswith("#") and item.attrs["href"][1:] in targets for item in links
    )
    assert root.find(data_kind="case")[1].attrs["data-status"] == "passed"


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("campaign_id", "older-campaign"),
        ("source_tree", "older-source"),
        ("relay_build", "older-build"),
        ("judge_revision", "older-rubric"),
        ("judge_model", "other-model"),
        ("judge_effort", "other-effort"),
        ("campaign_plan_sha256", "older-plan"),
        ("controlled_smoke", True),
        ("relay_contexts", {"contexts": {"work": {"instructions": "different-context"}}}),
        ("M_model", "different-response-model"),
        ("D_model", "different-decision-model"),
        ("arm_id", "different-arm"),
        ("adapter_sha256", "different-adapter"),
        ("provider_identity", {"M_provider": "different-provider"}),
    ],
)
def test_cross_campaign_or_identity_mismatch_never_replaces_existing_html(
    tmp_path, field, replacement
):
    first, second = report("one"), report("two")
    second[field] = replacement
    paths = [save(tmp_path, first, "first.json"), save(tmp_path, second, "second.json")]
    output = tmp_path / "latest.html"
    output.write_bytes(b"previous export remains intact")
    with pytest.raises(ValueError, match="metadata mismatch"):
        render_html(paths, output)
    assert output.read_bytes() == b"previous export remains intact"


@pytest.mark.parametrize("kind", ["case", "run", "attempt", "same-file"])
def test_overlapping_identities_are_rejected_instead_of_merged(tmp_path, kind):
    first, second = report("one"), report("two")
    if kind == "case":
        second["cases"][0]["id"] = first["cases"][0]["id"]
    elif kind == "run":
        second["cases"][0]["runs"][0]["id"] = first["cases"][0]["runs"][0]["id"]
    elif kind == "attempt":
        second["cases"][0]["runs"][0]["attempts"][0]["id"] = first["cases"][0]["runs"][0][
            "attempts"
        ][0]["id"]
    first_path = save(tmp_path, first, "first.json")
    paths = [
        first_path,
        first_path if kind == "same-file" else save(tmp_path, second, "second.json"),
    ]
    output = tmp_path / "latest.html"
    with pytest.raises(ValueError, match="Duplicate or overlapping"):
        render_html(paths, output)
    assert not output.exists()


def test_legacy_single_report_is_readable_but_cannot_be_silently_combined(tmp_path):
    first, second = report("one"), report("two")
    first.pop("campaign_id")
    second.pop("campaign_id")
    first_path, second_path = save(tmp_path, first, "first.json"), save(
        tmp_path, second, "second.json"
    )
    output = tmp_path / "legacy.html"
    render_html([first_path], output)
    root = Document(output.read_text(encoding="utf-8")).root
    assert "캠페인 ID 미기록" in root.text()
    assert len(root.find(data_kind="attempt")) == 1
    with pytest.raises(ValueError, match="explicit matching campaign"):
        render_html([first_path, second_path], output)


def test_injection_is_literal_in_every_surface_without_executable_markup(tmp_path):
    attack = '<script>window.pwned=1</script><img src="https://bad.invalid" onerror="alert(1)"> & "quoted"'
    value = report(attack)
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    attempt["transcript"][0]["text"] = attack
    attempt["error"] = attack
    attempt["actions"][0].update(text=attack, target=attack)
    attempt["D_invocations"][0]["response"] = {"answers": attack}
    attempt["judgment"]["criteria"][0]["quote"] = attack
    attempt["archive"] = [
        {
            "type": "function_call_output",
            "call_id": "injection",
            "output": json.dumps({"detail": attack}),
        }
    ]
    root, _, output = render(tmp_path, value)
    document = output.read_text(encoding="utf-8")
    assert "&lt;script&gt;" in document and "&lt;img" in document
    assert attack in root.find(data_kind="dialogue")[0].find(data_role="user")[0].text()
    assert attack in root.find(data_kind="sol-rubric")[0].text()
    assert attack in root.find(data_kind="case")[0].attrs["data-id"]
    for element in root.find():
        assert element.tag not in {
            "script",
            "img",
            "svg",
            "iframe",
            "object",
            "embed",
            "link",
            "form",
        }
        assert not any(name.lower().startswith("on") for name in element.attrs)
        assert "src" not in element.attrs
        if "href" in element.attrs:
            assert element.attrs["href"].startswith("#")
    assert "Content-Security-Policy" in document and "default-src 'none'" in document


def test_private_model_material_is_excluded_while_tool_failure_is_retained_once(tmp_path):
    value = report()
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    private = [
        "private-reasoning",
        "encrypted-private",
        "native-body-private",
        "credential-private",
        "state-duplicate-private",
    ]
    output = {"status": "failed", "detail": "meaningful tool failure", "authorization": private[3]}
    attempt["archive"] = [
        {"type": "reasoning", "content": private[0], "encrypted_content": private[1]},
        {
            "type": "function_call",
            "call_id": "tool-1",
            "name": "probe_component",
            "arguments": '{"component":"self_capture"}',
        },
        {"type": "function_call_output", "call_id": "tool-1", "output": json.dumps(output)},
        {
            "type": "function_call_output",
            "call_id": "tool-2",
            "output": json.dumps(
                {
                    "status": "failed",
                    "reasoning": private[0],
                    "encrypted_content": private[1],
                    "request_body": private[2],
                    "api_key": private[3],
                }
            ),
        },
    ]
    attempt["actions"][0]["result"] = {
        "status": "failed",
        "tool_results": [{"call_id": "tool-1", "result": output}],
    }
    attempt["actions"][0]["visible_before"] = {"state": private[4]}
    attempt["relay_calls"] = [
        {
            "method": "respond",
            "id": 1,
            "status": "failed",
            "code": "relay_failed",
            "request": {"input": private[2]},
            "state_before": private[4],
        }
    ]
    attempt["model_calls"][0]["native_request"] = private[2]
    attempt["access_attempts"] = [
        {"status": "failed", "api_key": private[3], "request_body": private[2]}
    ]
    attempt["code"]["state"] = {"full": private[4]}
    root, _, output_path = render(tmp_path, value)
    document = output_path.read_text(encoding="utf-8")
    for secret in private:
        assert secret not in document, secret
    assert root.find(data_kind="action-evidence")[0].text().count("meaningful tool failure") == 1
    ledger = root.find(data_kind="tool-ledger")[0]
    matching = [
        item
        for item in ledger.find(data_entry_kind="model_tool")
        if "tool-1" in item.find(data_kind="tool-identifiers")[0].text()
    ]
    assert len(matching) == 1
    assert matching[0].find(data_kind="tool-result")[0].text().count("meaningful tool failure") == 1
    assert "probe_component" in root.text() and "relay_failed" in root.text()
    assert "not-public-context" not in document
    assert root.text().count('"accepted_state"') == 1
    assert '"accepted_state": true' in root.find(data_kind="code-gates")[0].text()


@pytest.mark.parametrize("snapshot", [{"full": "nested-state-private"}, ["nested-state-private"]])
def test_structured_state_snapshots_are_excluded_but_scalar_states_and_error_codes_survive(
    tmp_path, snapshot
):
    value = report()
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    attempt["code"] = "model_access_failed"
    attempt["access_failure"] = {"code": "subscription_unavailable", "state": snapshot}
    attempt["actions"][0]["result"] = {
        "status": "failed",
        "tool_results": [
            {
                "call_id": "failed-tool",
                "result": {
                    "state": "completed",
                    "code": "microphone_unavailable",
                    "detail": "meaningful failed probe",
                },
            }
        ],
    }
    root, _, output_path = render(tmp_path, value)
    assert "nested-state-private" not in output_path.read_text(encoding="utf-8")
    assert "model_access_failed" in root.text()
    assert "subscription_unavailable" in root.text()
    evidence = root.find(data_kind="action-evidence")[0].text()
    assert '"state": "completed"' in evidence
    assert "microphone_unavailable" in evidence and "meaningful failed probe" in evidence


def test_input_report_cannot_be_overwritten_and_incomplete_json_is_not_exported(tmp_path):
    path = save(tmp_path, report())
    original = path.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        render_html([path], path)
    assert path.read_bytes() == original
    broken = tmp_path / "truncated.json"
    broken.write_bytes(b'{"format":"puri-agent-simulator-1",')
    output = tmp_path / "latest.html"
    with pytest.raises(ValueError):
        render_html([broken], output)
    assert not output.exists()


@pytest.mark.asyncio
async def test_cli_render_does_not_initialize_models_accounts_login_or_relay(monkeypatch, tmp_path):
    import httpx

    from puripuly_heart.release_evidence.agent_chat_simulator import __main__ as cli
    from puripuly_heart.release_evidence.agent_chat_simulator import models

    def forbidden(*args, **kwargs):
        pytest.fail("offline HTML rendering initialized model/account/network access")

    for name in (
        "make_models",
        "load_relay_context",
        "AgentRelayTransport",
        "AgentRelayTokenOwner",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    for name in ("SimulatorModels", "ChatGptAccountOwner", "ChatGptSession", "KeyringSecretStore"):
        monkeypatch.setattr(models, name, forbidden)
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(httpx, "Client", forbidden)
    first, second = save(tmp_path, report("one"), "first.json"), save(
        tmp_path, report("two"), "second.json"
    )
    output = tmp_path / "latest.html"
    arguments = cli.parser().parse_args(
        [
            "render-html",
            "--report",
            str(first),
            "--report",
            str(second),
            "--output",
            str(output),
        ]
    )
    assert await cli.execute(arguments) == 0
    assert len(Document(output.read_text(encoding="utf-8")).root.find(data_kind="case")) == 2
    assert cli.parser().parse_args(
        [
            "render-html",
            "--report",
            str(first),
            str(second),
            "--output",
            str(output),
        ]
    ).reports == [str(first), str(second)]


def test_nonspoken_action_without_its_own_transcript_row_keeps_turn_chronology(tmp_path):
    value = report()
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    attempt["transcript"] = [
        {"turn": 1, "role": "user", "text": "먼저 할 말"},
        {"turn": 1, "role": "assistant", "text": "첫 답변"},
        {"turn": 3, "role": "user", "text": "조작 다음 발화"},
        {"turn": 3, "role": "assistant", "text": "다음 답변"},
    ]
    root, _, _ = render(tmp_path, value)
    dialogue = root.find(data_kind="dialogue")[0]
    sequence = [item.attrs["data-role"] for item in dialogue.find() if "data-role" in item.attrs]
    assert sequence == ["user", "assistant", "control", "user", "assistant"]
    assert dialogue.text().index("create_key") < dialogue.text().index("조작 다음 발화")


def test_blocked_checkpoint_without_attempts_declares_unsaved_coverage(tmp_path):
    value = report()
    value["status"] = "blocked"
    value["cases"][0]["status"] = "blocked"
    for run in value["cases"][0]["runs"]:
        run.update(status="blocked", attempts=[], missing_prerequisites=["missing broker"])
    root, result, _ = render(tmp_path, value)
    assert result["coverage"] == {
        "listed_scenarios": 1,
        "saved_scenarios": 0,
        "listed_runs": 2,
        "saved_runs": 0,
        "attempts": 0,
        "planned_scenarios": 2,
        "planned_runs": 6,
    }
    assert not root.find(data_kind="attempt")
    assert len(root.find(data_kind="run")) == 2
    assert all(item.attrs["data-status"] == "blocked" for item in root.find(data_kind="run"))
    assert "missing broker" in root.text()


@pytest.mark.parametrize(
    "payload,visible",
    [
        (
            {"action": "Start", "target": "<invalid-target>", "api_key": "invalid-action-private"},
            "<invalid-target>",
        ),
        (["invalid-list-action"], "invalid-list-action"),
        ("invalid-string-action", "invalid-string-action"),
        (None, '"invalid_user_action": null'),
    ],
)
def test_harness_error_retains_invalid_payload_without_treating_it_as_executed(
    tmp_path, payload, visible
):
    value = report()
    run = value["cases"][0]["runs"][0]
    run["status"] = "harness_error"
    attempt = run["attempts"][0]
    attempt.update(
        status="harness_error",
        error="Invalid structured user action",
        harness_error={"code": "invalid_user_action", "detail": "Invalid structured user action"},
        invalid_user_action=payload,
        actions=[],
        transcript=[],
        model_calls=[],
    )
    attempt.pop("judgment")
    root, _, output_path = render(tmp_path, value)
    attempt_element = root.find(data_kind="attempt")[0]
    assert attempt_element.attrs["data-status"] == "harness_error"
    assert '"invalid_user_action"' in attempt_element.text()
    assert visible in attempt_element.text()
    assert "invalid-action-private" not in output_path.read_text(encoding="utf-8")
    assert not root.find(data_role="control") and not root.find(data_role="user")
    assert all(item.tag != "invalid-target" for item in root.find())


def ledger_json(entry, kind):
    return [
        json.loads(item.text())
        for item in entry.find(data_kind=kind)[0].find()
        if item.tag == "pre"
    ]


def test_tool_ledger_correlates_by_call_id_in_archive_order_and_keeps_tool_only_turns(tmp_path):
    value = report()
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    first = {"status": "observed", "component": "self_capture"}
    second = {"status": "failed", "code": "microphone_unavailable"}
    attempt["actions"] = [
        {
            "turn": 1,
            "action": "speak",
            "result": {
                "status": "completed",
                "tool_results": [
                    {"call_id": "second", "tool": "probe_component", "result": second},
                    {"call_id": "first", "tool": "get_app_context", "result": first},
                ],
            },
        }
    ]
    attempt["transcript"] = attempt["transcript"][:2]
    attempt["archive"] = [
        {
            "type": "function_call",
            "call_id": "first",
            "name": "get_app_context",
            "arguments": '{"topic":"audio","after":null}',
        },
        {
            "type": "function_call",
            "call_id": "second",
            "name": "probe_component",
            "arguments": '{"component":"self_capture"}',
        },
        {"type": "function_call_output", "call_id": "second", "output": json.dumps(second)},
        {"type": "function_call_output", "call_id": "first", "output": json.dumps(first)},
        {
            "type": "function_call",
            "call_id": "notice-call",
            "name": "get_app_context",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "notice-call",
            "output": '{"status":"observed","revision":2}',
        },
    ]
    attempt["relay_calls"] = [
        {
            "id": 11,
            "turn": 3,
            "method": "respond",
            "origin": "[app notice] ready",
            "response": {"output": [{"type": "function_call", "call_id": "notice-call"}]},
        }
    ]
    root, _, _ = render(tmp_path, value)
    ledger = root.find(data_kind="tool-ledger")[0]
    assert [item.attrs["data-turn"] for item in ledger.find(data_kind="tool-turn")] == ["1", "3"]
    entries = ledger.find(data_entry_kind="model_tool")
    identifiers = [ledger_json(item, "tool-identifiers")[0] for item in entries]
    assert [item["call_id"] for item in identifiers] == ["first", "second", "notice-call"]
    assert [item.attrs["data-status"] for item in entries] == ["observed", "failed", "observed"]
    assert ledger_json(entries[0], "tool-result") == [first]
    assert ledger_json(entries[1], "tool-result") == [second]
    assert ledger_json(entries[0], "tool-arguments") == [{"topic": "audio"}]
    assert len(identifiers[0]["sources"]) == 3
    assert identifiers[2]["relay_call_ids"] == [11]
    assert identifiers[2]["origins"] == ["[app notice] ready"]
    assert all(item.attrs["data-call-recorded"] == "true" for item in entries)


def test_tool_ledger_never_invents_missing_calls_arguments_results_or_id_matches(tmp_path):
    value = report()
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    attempt["transcript"] = []
    attempt["actions"] = [
        {
            "turn": 2,
            "action": "speak",
            "result": {
                "tool_results": [
                    {
                        "call_id": "action-only",
                        "tool": "probe_component",
                        "result": {"status": "observed"},
                    }
                ]
            },
        }
    ]
    attempt["archive"] = [
        {
            "turn": 1,
            "type": "function_call",
            "call_id": "missing-output",
            "name": "probe_component",
            "status": "completed",
        },
        {
            "turn": 2,
            "type": "function_call_output",
            "call_id": "unmatched",
            "output": '{"status":"failed"}',
        },
        {"turn": 4, "type": "function_call_output", "call_id": "empty"},
        {"turn": 5, "type": "function_call", "name": "probe_component", "arguments": "{}"},
        {"turn": 5, "type": "function_call_output", "output": '{"status":"failed"}'},
        {
            "type": "function_call",
            "call_id": "unassigned",
            "name": "probe_component",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "unassigned",
            "output": '{"status":"observed"}',
        },
    ]
    root, _, _ = render(tmp_path, value)
    entries = root.find(data_kind="tool-ledger")[0].find(data_entry_kind="model_tool")
    by_id = {ledger_json(item, "tool-identifiers")[0]["call_id"]: item for item in entries}
    missing = by_id["missing-output"]
    assert missing.attrs["data-status"] == "missing_result"
    assert missing.attrs["data-result-recorded"] == "false"
    assert missing.find(data_missing="arguments") and missing.find(data_missing="result")
    for call_id in ("unmatched", "action-only", "empty"):
        assert by_id[call_id].attrs["data-call-recorded"] == "false"
        assert by_id[call_id].find(data_missing="call")
        assert by_id[call_id].find(data_missing="arguments")
    assert by_id["empty"].attrs["data-status"] == "missing_result"
    anonymous = [
        item for item in entries if ledger_json(item, "tool-identifiers")[0]["call_id"] is None
    ]
    assert len(anonymous) == 2
    assert [item.attrs["data-result-recorded"] for item in anonymous] == ["false", "true"]
    unassigned = root.find(data_kind="tool-ledger")[0].find(data_kind="tool-turn")[-1]
    assert unassigned.attrs["data-turn-recorded"] == "false"
    assert [
        ledger_json(item, "tool-identifiers")[0]["call_id"]
        for item in unassigned.find(data_entry_kind="model_tool")
    ] == ["unassigned"]


def test_tool_ledger_preserves_conflicting_saved_results_instead_of_selecting_success(tmp_path):
    value = report()
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    attempt["archive"] = [
        {
            "type": "function_call",
            "call_id": "conflict",
            "name": "change_settings",
            "status": "completed",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "conflict", "output": '{"status":"pending"}'},
    ]
    attempt["actions"] = [
        {
            "turn": 1,
            "action": "speak",
            "result": {"tool_results": [{"call_id": "conflict", "result": {"status": "applied"}}]},
        }
    ]
    root, _, _ = render(tmp_path, value)
    entry = root.find(data_entry_kind="model_tool")[0]
    assert entry.attrs["data-status"] == "unknown"
    assert ledger_json(entry, "tool-result") == [{"status": "pending"}, {"status": "applied"}]
    evidence = root.find(data_kind="action-evidence")[0].text()
    assert '"status": "pending"' in evidence and '"status": "applied"' in evidence


def test_approval_and_async_completion_do_not_rewrite_model_tool_status(tmp_path):
    value = report()
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    confirmation = {
        "status": "needs_confirmation",
        "confirmation": {"id": "card-1", "changes": {"translation.model": "managed_gemma"}},
    }
    applied = {
        "status": "applied",
        "receipt": {"operation_id": "op-1", "status": "applied", "terminal": True},
    }
    pending = {
        "status": "pending",
        "receipt": {"operation_id": "op-2", "status": "accepted", "terminal": False},
    }
    completed = {
        "status": "applied",
        "receipt": {"operation_id": "op-2", "status": "applied", "terminal": True},
    }
    notice = "[app notice] confirmation " + json.dumps({"card_id": "card-1", "result": completed})
    attempt["actions"] = [
        {
            "turn": 1,
            "action": "speak",
            "result": {"tool_results": [{"call_id": "settings", "result": confirmation}]},
        },
        {"turn": 2, "action": "approve", "result": applied},
        {
            "turn": 3,
            "action": "speak",
            "result": {"tool_results": [{"call_id": "probe", "result": pending}]},
        },
    ]
    attempt["archive"] = [
        {
            "type": "function_call",
            "call_id": "settings",
            "name": "change_settings",
            "status": "completed",
            "arguments": '{"changes":{"translation.model":"managed_gemma"}}',
        },
        {"type": "function_call_output", "call_id": "settings", "output": json.dumps(confirmation)},
        {
            "type": "app_operation",
            "tool": "change_settings",
            "action_id": "apply-1",
            "phase": "submitted",
            "receipt": {"operation_id": "op-1", "status": "accepted"},
        },
        {
            "type": "app_operation",
            "tool": "change_settings",
            "action_id": "apply-1",
            "phase": "completed",
            "result": applied,
        },
        {
            "type": "function_call",
            "call_id": "probe",
            "name": "probe_translation",
            "status": "completed",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "probe", "output": json.dumps(pending)},
        {
            "type": "app_operation",
            "tool": "probe_translation",
            "action_id": "probe-1",
            "phase": "completed",
            "result": completed,
        },
    ]
    attempt["events"] = [
        {"turn": 4, "kind": "app_notice", "text": notice},
        {"turn": 4, "kind": "confirmation", "card_id": "card-1", "result": completed},
    ]
    attempt["transcript"] = [{"turn": 4, "role": "notice", "text": notice}]
    root, _, _ = render(tmp_path, value)
    ledger = root.find(data_kind="tool-ledger")[0]
    model = ledger.find(data_entry_kind="model_tool")
    assert [item.attrs["data-status"] for item in model] == ["needs_confirmation", "pending"]
    assert [ledger_json(item, "tool-identifiers")[0]["invocation_status"] for item in model] == [
        "completed",
        "completed",
    ]
    assert ledger_json(model[0], "tool-result")[0]["confirmation"]["id"] == "card-1"
    assert ledger.find(data_entry_kind="user_action")[0].attrs["data-status"] == "applied"
    operations = ledger.find(data_entry_kind="app_operation")
    assert [item.attrs["data-status"] for item in operations] == ["accepted", "applied", "applied"]
    assert [ledger_json(item, "tool-identifiers")[0]["phase"] for item in operations] == [
        "submitted",
        "completed",
        "completed",
    ]
    notice_entries = ledger.find(data_entry_kind="app_notice")
    assert len(notice_entries) == 1
    assert notice_entries[0].attrs["data-status"] == "applied"
    assert ledger_json(notice_entries[0], "tool-identifiers")[0]["card_id"] == "card-1"
    assert ledger_json(notice_entries[0], "tool-result") == [completed]
    turn_four = ledger.find(data_kind="tool-turn", data_turn="4")[0]
    assert turn_four.find(data_entry_kind="app_operation")
    assert not turn_four.find(data_entry_kind="model_tool")


def test_microphone_summary_separates_frames_level_vad_and_unexercised_recognition(tmp_path):
    value = report()
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    result = {
        "status": "applied",
        "receipt": {
            "operation_id": "mic-1",
            "status": "applied",
            "recognition": "not_exercised",
            "settings_changed": False,
            "diagnosis": {
                "state": "completed",
                "selected_candidate_id": "speech",
                "candidates": [
                    {
                        "id": "silent",
                        "frames_received": True,
                        "frame_count": 12,
                        "peak_level": 0.0,
                        "speech_detected": False,
                    },
                    {
                        "id": "speech",
                        "frames_received": True,
                        "frame_count": 12,
                        "peak_level": 0.24,
                        "speech_detected": True,
                    },
                    {"id": "missing", "frames_received": False, "failure_code": "capture_failed"},
                ],
            },
        },
    }
    attempt["actions"] = [
        {
            "turn": 1,
            "action": "speak",
            "result": {
                "tool_results": [
                    {"call_id": "mic", "tool": "diagnose_microphone", "result": result}
                ]
            },
        }
    ]
    attempt["archive"] = [
        {
            "type": "function_call",
            "call_id": "mic",
            "name": "diagnose_microphone",
            "arguments": "{}",
        }
    ]
    attempt.update(controlled=False, measurement_origin={"kind": "fresh_dialogue"})
    root, _, _ = render(tmp_path, value)
    ledger = root.find(data_kind="tool-ledger")[0]
    assert (
        ledger.find(data_kind="fixture-provenance")[0].attrs["data-provenance"]
        == "synthetic_fixture"
    )
    summary = ledger_json(ledger.find(data_entry_kind="model_tool")[0], "tool-result")[0]["receipt"]
    assert summary["recognition"] == "not_exercised"
    assert summary["settings_changed"] is False
    silent, speech, missing = summary["diagnosis"]["candidates"]
    assert (silent["frames_received"], silent["level"], silent["speech_detected"]) == (
        True,
        "zero",
        False,
    )
    assert (speech["frames_received"], speech["level"], speech["speech_detected"]) == (
        True,
        "nonzero",
        True,
    )
    assert missing == {"id": "missing", "frames_received": False, "failure_code": "capture_failed"}


def test_ledger_and_embedded_notice_json_redact_auth_and_private_reasoning(tmp_path):
    value = report()
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    secrets = [
        "argument-auth-private",
        "output-token-private",
        "notice-reasoning-private",
        "notice-auth-private",
    ]
    arguments = {"component": "self_capture", "auth": {"credential": secrets[0]}}
    result = {
        "status": "failed",
        "code": "capture_failed",
        "detail": "retained failure",
        "nested": {"access_token": secrets[1]},
    }
    notice = "[app notice] confirmation " + json.dumps(
        {
            "card_id": "card-safe",
            "result": {
                "status": "failed",
                "code": "apply_failed",
                "reasoning": secrets[2],
                "authorization": secrets[3],
            },
        }
    )
    attempt["archive"] = [
        {
            "type": "function_call",
            "call_id": "redact",
            "name": "probe_component",
            "arguments": json.dumps(arguments),
        },
        {"type": "function_call_output", "call_id": "redact", "output": json.dumps(result)},
    ]
    attempt["actions"] = [
        {
            "turn": 1,
            "action": "speak",
            "result": {"tool_results": [{"call_id": "redact", "result": result}]},
        }
    ]
    attempt["events"] = [{"turn": 2, "kind": "app_notice", "text": notice}]
    attempt["transcript"] = [{"turn": 2, "role": "notice", "text": notice}]
    root, _, output = render(tmp_path, value)
    document = output.read_text(encoding="utf-8")
    assert all(secret not in document for secret in secrets)
    ledger = root.find(data_kind="tool-ledger")[0]
    tool = ledger.find(data_entry_kind="model_tool")[0]
    assert ledger_json(tool, "tool-arguments")[0]["component"] == "self_capture"
    assert ledger_json(tool, "tool-result")[0]["code"] == "capture_failed"
    notice_entry = ledger.find(data_entry_kind="app_notice")[0]
    assert ledger_json(notice_entry, "tool-result")[0]["code"] == "apply_failed"
    assert root.find(data_kind="action-evidence")[0].text().count("retained failure") == 1


@pytest.mark.parametrize("events", [None, []])
def test_legacy_notice_only_completion_keeps_its_own_turn_and_result(tmp_path, events):
    value = report()
    attempt = value["cases"][0]["runs"][0]["attempts"][0]
    result = {"status": "applied", "receipt": {"operation_id": "legacy-op", "status": "applied"}}
    notice = "[app notice] confirmation " + json.dumps({"card_id": "legacy-card", "result": result})
    attempt.update(
        actions=[],
        events=events,
        transcript=[{"turn": 7, "role": "notice", "text": notice}],
        archive=[
            {
                "type": "app_operation",
                "tool": "change_settings",
                "phase": "completed",
                "result": result,
            }
        ],
    )
    root, _, _ = render(tmp_path, value)
    ledger = root.find(data_kind="tool-ledger")[0]
    assert not ledger.find(data_entry_kind="model_tool")
    turns = ledger.find(data_kind="tool-turn")
    assert len(turns) == 1 and turns[0].attrs["data-turn"] == "7"
    assert ledger_json(ledger.find(data_entry_kind="app_operation")[0], "tool-result") == [result]
    assert ledger_json(ledger.find(data_entry_kind="app_notice")[0], "tool-result") == [result]
