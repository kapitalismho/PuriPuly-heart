from __future__ import annotations

import json

import pytest

from puripuly_heart.app.ports.agent_chat import AgentMessage
from puripuly_heart.app.services.agent_chat.modules import MODULES
from puripuly_heart.app.services.agent_chat.routing import (
    AgentDecision,
    classification_input,
    extract_decision,
)


def decision(route="work", effort="high"):
    return {
        "answers": [
            {"name": "effort", "type": "choice", "choice": effort},
            {"name": "route", "type": "choice", "choice": route},
        ]
    }


def selected_decision(*modules, relation="new", route="work", effort="high"):
    result = decision(route, effort)
    result["answers"].extend(
        {
            "name": f"context_{module}",
            "type": "choice",
            "choice": "include" if module in modules else "omit",
        }
        for module in MODULES[1:]
    )
    result["answers"].append({"name": "context_relation", "type": "choice", "choice": relation})
    return result


def test_independent_context_choices_allow_cross_area_tasks_without_a_two_module_cap():
    result = selected_decision("audio", "recognition", "languages", "output", relation="continue")
    extracted = extract_decision(result)
    assert extracted.route == "work" and extracted.effort == "high"
    assert extracted.context_modules == ("core", "audio", "recognition", "languages", "output")
    assert extracted.context_relation == "continue"


@pytest.mark.parametrize(
    "failure",
    ["missing", "duplicate", "invalid", "unclear", "refusal", "relation", "structured"],
)
def test_invalid_context_selection_keeps_valid_work_route_without_a_partial_subset(failure):
    result = selected_decision("translation", "accounts", effort="xhigh")
    answer = next(item for item in result["answers"] if item["name"] == "context_audio")
    if failure == "missing":
        result["answers"].remove(answer)
    elif failure == "duplicate":
        result["answers"].append(dict(answer))
    elif failure == "refusal":
        answer["type"] = "refusal"
    elif failure == "relation":
        result["answers"][-1]["choice"] = "unclear"
    else:
        answer["choice"] = {"invalid": True} if failure == "structured" else failure
    extracted = extract_decision(result)
    assert extracted.route == "work" and extracted.effort == "xhigh"
    assert extracted.context_modules is None
    assert extracted.context_reason is not None


@pytest.mark.parametrize("route", ["work", "unclear", "social", "out_of_scope"])
@pytest.mark.parametrize("effort", ["low", "high", "xhigh"])
def test_extracts_native_choices_by_name_and_forces_brief_routes_low(route, effort):
    result = decision(route, effort)
    result["answers"][0].update(confidence=0.01, probabilities=[])
    assert extract_decision(result) == AgentDecision(
        route, "low" if route in {"social", "out_of_scope"} else effort
    )


@pytest.mark.parametrize(
    "result",
    [
        None,
        {"answers": []},
        decision("unknown"),
        {"answers": [{"name": "route", "type": "choice", "choice": "work"}] * 2},
    ],
)
def test_unusable_route_falls_back_to_unclear_high(result):
    extracted = extract_decision(result)
    assert extracted == AgentDecision("unclear", "high")
    assert extracted.source == "fallback"


@pytest.mark.parametrize("route", ["work", "unclear", "social", "out_of_scope"])
@pytest.mark.parametrize("effort", [None, {"invalid": "structured"}, "medium", "max"])
def test_only_malformed_effort_keeps_the_route(route, effort):
    extracted = extract_decision(decision(route, effort))
    assert extracted == AgentDecision(
        route, "low" if route in {"social", "out_of_scope"} else "high"
    )
    assert extracted.source == "fallback"
    assert extracted.reason == "invalid_effort"


def test_missing_effort_keeps_valid_route():
    assert extract_decision(
        {"answers": [{"name": "route", "type": "choice", "choice": "work"}]}
    ) == AgentDecision("work", "high")


@pytest.mark.parametrize("name", ["route", "effort", "context_unknown", None])
@pytest.mark.parametrize("duplicate", [False, True])
def test_any_refusal_overrides_other_choices_with_isolated_brief_route(name, duplicate):
    result = decision("work", "xhigh")
    refusal = {"type": "refusal"}
    if name is not None:
        refusal["name"] = name
    if duplicate:
        result["answers"].append(refusal)
    else:
        result["answers"] = [refusal]
    extracted = extract_decision(result)
    assert extracted == AgentDecision("out_of_scope", "low")
    assert extracted.source == "fallback"
    assert extracted.reason == "classification_refused"


def test_compact_input_contains_current_message_four_recent_messages_and_short_summaries():
    messages = [AgentMessage("user", "OLD-EXCLUDED"), AgentMessage("notice", "PRIVATE-NOTICE")]
    messages += [
        AgentMessage("assistant" if i % 2 else "user", f"message-{i} " + "x" * 1000)
        for i in range(6)
    ]
    sent = classification_input(
        "그걸로 해",
        recent_messages=messages,
        active_task="Model recommendation " + "t" * 1000,
        last_puri_action="get_app_context translation observed " + "a" * 1000,
        previous_modules=("core", "translation", "accounts"),
    )
    assert len(sent) == 1
    assert sent[0]["role"] == "user"
    content = json.loads(sent[0]["content"])
    assert set(content) == {
        "current_message",
        "recent_messages",
        "active_task",
        "last_puri_action",
        "previous_modules",
    }
    assert content["previous_modules"] == ["core", "translation", "accounts"]
    assert content["current_message"] == "그걸로 해"
    assert [item["content"].split()[0] for item in content["recent_messages"]] == [
        f"message-{i}" for i in range(2, 6)
    ]
    assert all(len(item["content"]) == 600 for item in content["recent_messages"])
    assert len(content["active_task"]) == len(content["last_puri_action"]) == 240
    assert "OLD-EXCLUDED" not in sent[0]["content"]
    assert "PRIVATE-NOTICE" not in sent[0]["content"]
    assert messages[0].text == "OLD-EXCLUDED"


def test_first_message_has_no_invented_conversation_or_task():
    sent = classification_input(
        "그거 좀", recent_messages=[], active_task=None, last_puri_action=None
    )
    assert json.loads(sent[0]["content"]) == {
        "current_message": "그거 좀",
        "recent_messages": [],
        "active_task": None,
        "last_puri_action": None,
        "previous_modules": [],
    }
