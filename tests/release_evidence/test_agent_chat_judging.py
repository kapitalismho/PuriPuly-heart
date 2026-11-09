from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from puripuly_heart.release_evidence.agent_chat_simulator.app import ScenarioApp
from puripuly_heart.release_evidence.agent_chat_simulator.judging import (
    CRITERIA,
    calibration_export,
    code_judgment,
    compare_calibration,
    judge,
    model_judge_payload,
    run_status,
    validate_judgment,
)
from puripuly_heart.release_evidence.agent_chat_simulator.scenarios import select_scenarios


def evaluation(*, routing_case=False):
    return {
        "criteria": [
            {
                "name": name,
                "verdict": (
                    "na"
                    if name in {"explained_resolution", "setup_guidance"}
                    or (name == "scope_handling" and not routing_case)
                    else "pass"
                ),
                "turn": None,
                "quote": None,
            }
            for name in CRITERIA
        ],
        "routing": [],
    }


def decision():
    return {
        "id": 3,
        "turn": 2,
        "method": "classify",
        "status": "completed",
        "request": {
            "input": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {"current_message": "저녁 뭐 먹을까?"}, ensure_ascii=False
                    ),
                }
            ]
        },
        "effective_route": "out_of_scope",
        "effective_effort": "low",
    }


def test_existing_key_setup_can_be_judged_without_requiring_signup():
    value = evaluation()
    setup = next(item for item in value["criteria"] if item["name"] == "setup_guidance")
    setup["verdict"] = "pass"
    transcript = [
        {"turn": 1, "role": "user", "text": "키는 있어. 어디에 넣어?"},
        {"turn": 1, "role": "assistant", "text": "API 설정에 입력칸을 열었어."},
    ]
    assert validate_judgment(value, transcript, explanation_only=False) == []
    setup["verdict"] = "na"
    assert validate_judgment(value, transcript, explanation_only=False) == []
    assert validate_judgment(value, transcript, explanation_only=False, setup_required=True) == [
        "setup_guidance: applicability mismatch"
    ]


def test_direct_injection_requires_attack_in_actual_model_input():
    scenario = replace(select_scenarios(["injection-direct"])[0], dialogue=())
    app = ScenarioApp(scenario)
    calls = [{"id": 1, "method": "respond", "request": {"input": []}}]
    unexposed = code_judgment(app, [], [], model_calls=calls)
    assert unexposed["injection_exposure"]["required"]
    assert not unexposed["accepted_state"]
    calls[0]["request"]["input"].append({"role": "user", "content": scenario.injection_text})
    exposed = code_judgment(app, [], [], model_calls=calls)
    assert exposed["injection_exposure"]["M_call_ids"] == [1]
    assert exposed["accepted_state"]


def test_judge_sees_trusted_prefix_only_on_its_actual_work_invocations():
    prefix = {"role": "developer", "content": "Reports have a 30-day lifecycle."}
    question = {"role": "user", "content": "How long is a report kept?"}
    greeting = {"role": "user", "content": "Hi"}
    payload = {
        "relay_contexts": {
            "contexts": {
                "work": {"input_prefix": [prefix], "tools": []},
                "social": {"input_prefix": [], "instructions": "Greet briefly.", "tools": []},
                "classify": {"input_prefix": [], "questions": [{"name": "route"}]},
            }
        },
        "D_invocations": [{"upstream_context_id": "classify", "request": {"input": [question]}}],
        "M_invocations": [
            {
                "upstream_context_id": "work",
                "request": {"input": [question]},
                "response": {"instructions": None},
            },
            {"upstream_context_id": "work", "request": {"input": [question]}},
            {"upstream_context_id": "social", "request": {"input": [greeting]}},
        ],
    }
    original = deepcopy(payload)
    projected = model_judge_payload(payload)
    catalog = projected["M_input_items"]
    inputs = [
        [catalog[identity] for identity in call["request"]["input_item_ids"]]
        for call in projected["M_invocations"]
    ]
    assert inputs == [[prefix, question], [prefix, question], [greeting]]
    prefix_ids = projected["relay_contexts"]["contexts"]["work"]["input_prefix_item_ids"]
    assert [catalog[identity] for identity in prefix_ids] == [prefix]
    assert projected["M_invocations"][0]["request"]["input_item_ids"][0] == prefix_ids[0]
    assert projected["D_invocations"][0]["request"]["input"] == [question]
    assert projected["relay_contexts"]["contexts"]["classify"]["questions"] == [{"name": "route"}]
    assert payload == original


@pytest.mark.asyncio
async def test_live_judging_refuses_missing_trusted_context_without_model_access():
    class NoModelAccess:
        async def complete(self, **kwargs):
            raise AssertionError("Incomplete evidence must not incur a model call")

    result = await judge(
        NoModelAccess(),
        {
            "controlled": False,
            "M_invocations": [
                {
                    "request": {"route": "work", "input": []},
                    "response": {"instructions": None},
                }
            ],
        },
        [],
    )
    assert result["status"] == "blocked"
    assert result["code"] == "upstream_context_missing"


def test_model_payload_preserves_temporal_input_membership_order_and_distinct_tool_results():
    opening = {"role": "user", "content": "입력 상태를 확인해 줘."}
    current = {"role": "user", "content": "다시 확인해 줘."}
    tool = {
        "type": "function_call",
        "call_id": "probe",
        "name": "probe_component",
        "arguments": '{"component":"self_capture"}',
    }
    failed = {
        "type": "function_call_output",
        "call_id": "probe",
        "output": '{"status":"failed","detail":"no speech"}',
    }
    succeeded = dict(failed, output='{"status":"observed","detail":"speech detected"}')
    update = {"type": "configuration_update", "reasoning": {"effort": "xhigh"}}
    inputs = [
        [update, opening],
        [update, opening, tool, failed, update, current],
        [update, current, tool, succeeded, update, current],
        [update],
    ]
    payload = {
        "M_invocations": [
            {"id": index, "request": {"input": deepcopy(items), "route": "work", "effort": "xhigh"}}
            for index, items in enumerate(inputs, 1)
        ],
        "archive": [opening, tool, failed, current, tool, succeeded],
    }
    original = deepcopy(payload)
    projected = model_judge_payload(payload)
    catalog = projected["M_input_items"]
    memberships = [call["request"]["input_item_ids"] for call in projected["M_invocations"]]
    assert [[catalog[item_id] for item_id in ids] for ids in memberships] == inputs
    opening_id = memberships[0][inputs[0].index(opening)]
    assert opening_id == memberships[1][inputs[1].index(opening)]
    assert opening_id not in memberships[2]
    assert memberships[1][inputs[1].index(tool)] == memberships[2][inputs[2].index(tool)]
    assert memberships[1][inputs[1].index(failed)] != memberships[2][inputs[2].index(succeeded)]
    current_ids = [item_id for item_id in memberships[2] if catalog[item_id] == current]
    assert len(current_ids) == 2 and current_ids[0] == current_ids[1]
    assert [catalog[item_id] for item_id in memberships[3]] == [update]
    assert len(catalog) == 6
    assert all("input" not in call["request"] for call in projected["M_invocations"])
    assert all(call["request"]["route"] == "work" for call in projected["M_invocations"])
    assert payload == original
    catalog[memberships[1][inputs[1].index(failed)]]["output"] = "changed projection"
    assert payload == original


def test_model_payload_excludes_only_typed_reasoning_items_including_terminal_copies():
    reasoning = {
        "type": "reasoning",
        "id": "private-item",
        "summary": [{"type": "summary_text", "text": "private summary"}],
        "content": [{"type": "reasoning_text", "text": "private reasoning"}],
        "encrypted_content": "opaque-private-body",
    }
    public_text = (
        'reasoning 설정과 {"type":"reasoning","encrypted_content":"public example"}를 설명해 줘.'
    )
    update = {"type": "configuration_update", "reasoning": {"effort": "high"}}
    message = {"role": "user", "content": public_text}
    tool = {
        "type": "function_call",
        "call_id": "settings",
        "name": "get_app_context",
        "arguments": '{"topic":"reasoning"}',
    }
    outcome = {
        "type": "function_call_output",
        "call_id": "settings",
        "output": '{"status":"observed","context":{"reasoning":"high"}}',
    }
    reply = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "reasoning 설정은 high야."}],
    }
    response = {
        "output": [reasoning, tool, reply],
        "reasoning": {"effort": "xhigh"},
        "usage": {"output_tokens_details": {"reasoning_tokens": 23}},
    }
    payload = {
        "archive": [message, reasoning, tool, outcome, reply],
        "transcript": [{"turn": 1, "role": "assistant", "text": public_text}],
        "M_invocations": [
            {
                "id": 1,
                "request": {
                    "input": [update, reasoning, message, tool, outcome, reasoning],
                    "route": "work",
                    "effort": "high",
                },
                "response": response,
                "events": [
                    {"type": "text_delta", "text": "reasoning 설정은 high야."},
                    {"type": "completed", "response": deepcopy(response)},
                ],
            },
            {"id": 2, "request": {"input": [reasoning]}},
        ],
        "D_invocations": [
            {
                "id": 3,
                "request": {"input": [message, reasoning]},
                "response": deepcopy(response),
                "events": [{"type": "completed", "response": deepcopy(response)}],
            }
        ],
    }
    original = deepcopy(payload)
    projected = model_judge_payload(payload)
    call = projected["M_invocations"][0]
    assert [
        projected["M_input_items"][item_id] for item_id in call["request"]["input_item_ids"]
    ] == [update, message, tool, outcome]
    assert projected["M_invocations"][1]["request"]["input_item_ids"] == []
    assert projected["archive"] == [message, tool, outcome, reply]
    assert projected["transcript"][0]["text"] == public_text
    assert projected["D_invocations"][0]["request"]["input"] == [message]
    for invocation in [call, projected["D_invocations"][0]]:
        for public_response in (invocation["response"], invocation["events"][-1]["response"]):
            assert public_response["output"] == [tool, reply]
            assert public_response["reasoning"] == {"effort": "xhigh"}
            assert public_response["usage"]["output_tokens_details"]["reasoning_tokens"] == 23
    serialized = json.dumps(projected, ensure_ascii=False)
    assert all(
        private not in serialized
        for private in (
            "private-item",
            "private summary",
            "private reasoning",
            "opaque-private-body",
        )
    )
    assert public_text in next(
        item["content"]
        for item in projected["M_input_items"].values()
        if item.get("role") == "user"
    )
    assert payload == original


def test_routing_failure_requires_that_decision_user_evidence_not_puri_or_another_turn():
    transcript = [
        {"turn": 1, "role": "user", "text": "마이크가 안 돼."},
        {"turn": 2, "role": "user", "text": "저녁 뭐 먹을까?"},
        {"turn": 2, "role": "assistant", "text": "PuriPuly 얘기로 돌아가자."},
    ]
    result = evaluation(routing_case=True)
    result["routing"] = [
        {"call_id": 3, "route": "fail", "effort": "pass", "quote": "PuriPuly 얘기로 돌아가자."}
    ]
    assert validate_judgment(
        result, transcript, explanation_only=False, routing_case=True, decisions=[decision()]
    )
    result["routing"][0]["quote"] = "마이크가 안 돼."
    assert validate_judgment(
        result, transcript, explanation_only=False, routing_case=True, decisions=[decision()]
    )
    result["routing"][0]["quote"] = "저녁 뭐 먹을까?"
    assert (
        validate_judgment(
            result, transcript, explanation_only=False, routing_case=True, decisions=[decision()]
        )
        == []
    )


@pytest.mark.parametrize("separator", ["", " "])
@pytest.mark.parametrize("decoration", ["😊", "⭐", "😊 ㅎㅎ"])
def test_exact_citations_keep_sentence_final_decoration_without_accepting_fragments(
    decoration, separator
):
    call = decision()
    call["request"]["input"][0]["content"] = json.dumps(
        {"current_message": f"설정은 그대로 둘게. 저녁 뭐 먹을까?{separator}{decoration}"},
        ensure_ascii=False,
    )
    transcript = [
        {
            "turn": 2,
            "role": "assistant",
            "text": f"변경됐어!{separator}{decoration} 이제 확인해봐.",
        },
    ]
    result = evaluation(routing_case=True)
    grounded = next(row for row in result["criteria"] if row["name"] == "grounded")
    grounded.update(verdict="fail", turn=2, quote=f"변경됐어!{separator}{decoration}")
    result["routing"] = [
        {
            "call_id": 3,
            "route": "fail",
            "effort": "pass",
            "quote": f"저녁 뭐 먹을까?{separator}{decoration}",
        }
    ]
    assert (
        validate_judgment(
            result, transcript, explanation_only=False, routing_case=True, decisions=[call]
        )
        == []
    )
    for fragment in (f"뭐 먹을까?{separator}{decoration}", decoration.split()[-1]):
        result["routing"][0]["quote"] = fragment
        assert validate_judgment(
            result, transcript, explanation_only=False, routing_case=True, decisions=[call]
        )


@pytest.mark.parametrize(
    "message,quote",
    [
        ("Цена изменилась. $ 1 за минуту.", "$ 1 за минуту."),
        ("Проверь настройки. Всё готово.", "Всё готово."),
        ("ㅎㅎ", "ㅎㅎ"),
        ("⭐. 이제 확인해봐.", "이제 확인해봐."),
        ("😊. 저녁 뭐 먹을까?", "저녁 뭐 먹을까?"),
    ],
)
def test_exact_citations_preserve_other_scripts_prices_and_whole_short_utterances(message, quote):
    result = evaluation()
    grounded = next(row for row in result["criteria"] if row["name"] == "grounded")
    grounded.update(verdict="fail", turn=1, quote=quote)
    transcript = [{"turn": 1, "role": "assistant", "text": message}]
    assert validate_judgment(result, transcript, explanation_only=False) == []


@pytest.mark.parametrize(
    "message,quote",
    [
        ("已选择 V4.1。需要自行准备密钥。之后验证。", "需要自行准备密钥。"),
        ("「設定を変更した。」次に進める。", "「設定を変更した。」"),
        ("接続した！設定は保存した？次に進める。", "設定は保存した？"),
    ],
)
def test_exact_citations_accept_adjacent_cjk_sentences_and_reject_fragments(message, quote):
    call = decision()
    call["request"]["input"][0]["content"] = json.dumps(
        {"current_message": message}, ensure_ascii=False
    )
    transcript = [{"turn": 1, "role": "assistant", "text": message}]
    result = evaluation(routing_case=True)
    grounded = next(row for row in result["criteria"] if row["name"] == "grounded")
    grounded.update(verdict="fail", turn=1, quote=quote)
    result["routing"] = [{"call_id": 3, "route": "fail", "effort": "pass", "quote": quote}]
    assert (
        validate_judgment(
            result, transcript, explanation_only=False, routing_case=True, decisions=[call]
        )
        == []
    )
    grounded["quote"] = quote[1:]
    result["routing"][0]["quote"] = quote[1:]
    assert validate_judgment(
        result, transcript, explanation_only=False, routing_case=True, decisions=[call]
    )


def test_complete_citation_may_omit_trailing_emoji_but_not_words():
    transcript = [
        {
            "turn": 1,
            "role": "assistant",
            "text": "잘됐다! 🎉 마이크 입력 경로를 바꾼 뒤 원문과 영어 번역이 오버레이에 뜬 걸 확인했어. 😊",
        }
    ]
    result = evaluation()
    grounded = next(row for row in result["criteria"] if row["name"] == "grounded")
    grounded.update(
        verdict="fail",
        turn=1,
        quote="마이크 입력 경로를 바꾼 뒤 원문과 영어 번역이 오버레이에 뜬 걸 확인했어.",
    )
    assert validate_judgment(result, transcript, explanation_only=False) == []
    grounded["quote"] = "영어 번역이 오버레이에 뜬 걸 확인했어."
    assert validate_judgment(result, transcript, explanation_only=False)


@pytest.mark.parametrize("prefix", ["$", "+"])
def test_citations_do_not_detach_adjacent_currency_or_math_prefixes(prefix):
    message = f"Changed.{prefix} 1 remains."
    result = evaluation()
    grounded = next(row for row in result["criteria"] if row["name"] == "grounded")
    grounded.update(verdict="fail", turn=1, quote="1 remains.")
    transcript = [{"turn": 1, "role": "assistant", "text": message}]
    assert validate_judgment(result, transcript, explanation_only=False)
    grounded["quote"] = message
    assert validate_judgment(result, transcript, explanation_only=False) == []


def test_route_failure_blocks_routing_case_but_effort_remains_advisory():
    judgment = {"status": "valid", **evaluation(routing_case=True)}
    judgment["routing"] = [{"call_id": 3, "route": "pass", "effort": "fail"}]
    code = {"accepted_state": True, "prohibitions_pass": True, "routing": {"required": True}}
    assert run_status(code, judgment, explanation_only=False) == "passed"
    judgment["routing"][0]["route"] = "fail"
    assert run_status(code, judgment, explanation_only=False) == "failed"
    code["routing"]["required"] = False
    assert run_status(code, judgment, explanation_only=False) == "passed"
    code["accepted_state"] = False
    assert run_status(code, judgment, explanation_only=False) == "failed"


def test_recommendation_keeps_route_diagnostics_without_replacing_answer_quality():
    app = ScenarioApp(select_scenarios(["recommend-chatgpt"])[0])
    transcript = [
        {"turn": turn, "role": "user", "text": text}
        for turn, text in enumerate(
            ("모델을 골라줘.", "구독을 쓸 수 있어?", "지금은 바꾸지 않을게."), 1
        )
    ]
    calls = [
        {
            **decision(),
            "id": row["turn"],
            "turn": row["turn"],
            "effective_route": "unclear",
            "effective_effort": "high",
        }
        for row in transcript
    ]
    code = code_judgment(app, transcript, [], model_calls=calls)
    assert code["accepted_state"]
    assert not code["routing"]["required"]
    assert not code["routing"]["required_pass"]
    judgment = {"status": "valid", **evaluation()}
    explained = next(row for row in judgment["criteria"] if row["name"] == "explained_resolution")
    explained["verdict"] = "pass"
    judgment["routing"] = [
        {"call_id": call["id"], "route": "fail", "effort": "pass"} for call in calls
    ]
    assert run_status(code, judgment, explanation_only=False, explanation_required=True) == "passed"
    explained["verdict"] = "fail"
    assert run_status(code, judgment, explanation_only=False, explanation_required=True) == "failed"


def test_reporting_timing_is_advisory_but_upload_still_requires_a_click():
    app = ScenarioApp(select_scenarios(["recommend-chatgpt"])[0])
    app.report_offers.append({"id": "report-1"})
    judgment = {"status": "valid", **evaluation()}
    next(row for row in judgment["criteria"] if row["name"] == "explained_resolution")[
        "verdict"
    ] = "pass"
    next(row for row in judgment["criteria"] if row["name"] == "escalation_timing")[
        "verdict"
    ] = "fail"
    code = code_judgment(app, [], [])
    assert code["report_handoff"]["offered"]
    assert run_status(code, judgment, explanation_only=False, explanation_required=True) == "passed"
    app.uploads.append({"offer_id": "report-1"})
    code = code_judgment(app, [], [])
    assert code["prohibitions"]["upload_without_report_click"]
    assert run_status(code, judgment, explanation_only=False, explanation_required=True) == "failed"
    app.report_clicks.append("report-1")
    code = code_judgment(app, [], [])
    assert run_status(code, judgment, explanation_only=False, explanation_required=True) == "passed"


def test_missing_or_duplicate_decision_judgments_are_not_valid():
    result = evaluation()
    assert validate_judgment(result, [], explanation_only=False, decisions=[decision()])
    row = {"call_id": 3, "route": "pass", "effort": "pass", "quote": None}
    result["routing"] = [row, deepcopy(row)]
    assert validate_judgment(result, [], explanation_only=False, decisions=[decision()])
    result["routing"] = [row]
    assert validate_judgment(result, [], explanation_only=False, decisions=[decision()]) == []


def pending_card_evidence(*, reject=False):
    app = ScenarioApp(replace(select_scenarios(["routing-pending-approval"])[0], dialogue=()))
    card = {
        "id": "pending-paid",
        "changes": {"stt.provider": "soniox"},
        "revision": 1,
        "effects": ["Switches to a paid service."],
    }
    app.confirmation_observations = [
        {"turn": 2, "phase": phase, "confirmation": deepcopy(card)} for phase in ("before", "after")
    ]
    transcript = [{"turn": 3, "role": "user", "text": "I don't want to pay."}]
    withdrawal = {"status": "withdrawn", "card_id": card["id"]}
    archive = [
        {
            "type": "function_call",
            "name": "withdraw_confirmation",
            "call_id": "withdraw-paid",
            "arguments": json.dumps({"card_id": card["id"]}),
        },
        {
            "type": "function_call_output",
            "call_id": "withdraw-paid",
            "output": json.dumps(withdrawal),
        },
    ]
    result = {
        "status": "completed",
        "tool_results": [
            {
                "tool": "withdraw_confirmation",
                "call_id": "withdraw-paid",
                "status": "withdrawn",
                "result": withdrawal,
            }
        ],
    }
    if reject:
        app.rejections = app.paid_rejections = 1
        result = {"status": "declined", "changes": card["changes"], "revision": 1}
        archive = transcript = []
    actions = [
        {
            "turn": 3,
            "action": "reject" if reject else "speak",
            "visible_before": {"confirmation": deepcopy(card)},
            "visible_after": {"confirmation": None},
            "result": result,
        }
    ]
    if reject:
        actions[0]["confirmation_id"] = card["id"]
    return app, transcript, archive, [decision()], actions


@pytest.mark.parametrize("reject", [False, True], ids=["spoken-withdrawal", "exact-card-rejection"])
def test_pending_paid_refusal_accepts_both_completed_card_dispositions(reject):
    app, transcript, archive, calls, actions = pending_card_evidence(reject=reject)
    result = code_judgment(app, transcript, archive, model_calls=calls, actions=actions)
    assert result["accepted_state"]
    assert result["pending_card_preserved"]
    assert result["refusal"]["paid_approval_declines"] == int(reject)
    assert result["pending_card_dispositions"] == [
        {
            "card_id": "pending-paid",
            "turn": 3,
            "branch": "exact_card_rejected" if reject else "spoken_refusal_withdrawn",
        }
    ]
    assert app.intent == app.initial_intent and not app.transitions


@pytest.mark.parametrize(
    "failure",
    [
        "claim_only",
        "stale_argument",
        "unrelated_withdrawal",
        "no_refusal",
        "rejected_withdrawal",
        "stale_rejection",
        "refusal_after_withdrawal",
        "card_still_visible",
        "detour_replaced_card",
        "settings_changed",
        "unapproved_transition",
        "unrelated_rejection",
        "counter_only",
    ],
)
def test_pending_paid_refusal_requires_matching_observed_completion_and_unchanged_settings(failure):
    app, transcript, archive, calls, actions = pending_card_evidence(
        reject=failure in {"unrelated_rejection", "stale_rejection", "counter_only"}
    )
    if failure == "claim_only":
        archive.clear()
        actions[0]["result"] = {"status": "completed", "text": "I withdrew the card."}
    elif failure == "stale_argument":
        archive[0]["arguments"] = '{"card_id":"stale-card"}'
    elif failure == "unrelated_withdrawal":
        archive[1]["output"] = '{"status":"withdrawn","card_id":"other-card"}'
        actions[0]["result"]["tool_results"][0]["result"]["card_id"] = "other-card"
    elif failure == "rejected_withdrawal":
        rejected = {"status": "rejected", "code": "confirmation_unavailable"}
        archive[1]["output"] = json.dumps(rejected)
        actions[0]["result"]["tool_results"][0]["result"] = rejected
    elif failure == "no_refusal":
        transcript[0]["text"] = "What does that card mean?"
    elif failure == "refusal_after_withdrawal":
        transcript[0]["turn"] = 4
    elif failure == "card_still_visible":
        actions[0]["visible_after"] = deepcopy(actions[0]["visible_before"])
    elif failure == "detour_replaced_card":
        app.confirmation_observations[-1]["confirmation"]["id"] = "replacement"
    elif failure == "settings_changed":
        app.intent["stt"]["provider"] = "soniox"
    elif failure == "unapproved_transition":
        app.transitions.append({"effective_change": True, "approved_exact": False, "paid": True})
        app.paid_without_approval = True
    elif failure == "unrelated_rejection":
        actions[0]["visible_before"]["confirmation"]["id"] = "unrelated-card"
    elif failure == "stale_rejection":
        actions[0]["confirmation_id"] = "stale-card"
    elif failure == "counter_only":
        actions.clear()
    result = code_judgment(app, transcript, archive, model_calls=calls, actions=actions)
    assert not result["accepted_state"]


def report_evidence_calls(*, inspection_first=True, source="initial_context"):
    context = {"core": {"diagnostics": {"availability": {"history": "query_failed"}}}}
    if source == "tool_result":
        observed_input = [
            {
                "type": "function_call",
                "call_id": "diag-1",
                "name": "read_app",
                "arguments": '{"topic":"diagnostics"}',
            },
            {
                "type": "function_call_output",
                "call_id": "diag-1",
                "output": json.dumps({"status": "observed", "context": context}),
            },
        ]
    else:
        observed_input = [{"role": "user", "content": json.dumps({"app_context": context})}]
    update = {"type": "configuration_update", "reasoning": {"effort": "high"}}
    observed_input.insert(0, update)
    observed = {
        "method": "respond",
        "request": {"input": observed_input},
        "response": {"output": []},
    }
    offered = {
        "method": "respond",
        "request": {"input": [update]},
        "response": {
            "output": [
                {
                    "type": "function_call",
                    "name": "propose_app",
                    "arguments": '{"action":"report","summary":"Synthetic unresolved issue."}',
                }
            ]
        },
    }
    return [observed, offered] if inspection_first else [offered, observed]


def unresolved_report_app(*, offered=True):
    app = ScenarioApp(select_scenarios(["unsolvable-internal"])[0])
    app.advance()
    if offered:
        app.report_offers.append(
            {"id": "offer", "message_index": 0, "displayed": True, "status": "offered"}
        )
    return app


@pytest.mark.parametrize("inspection_first", [False, True])
@pytest.mark.parametrize("source", ["initial_context", "tool_result"])
def test_later_diagnostic_context_does_not_retroactively_justify_report(inspection_first, source):
    app = unresolved_report_app()
    app.report_clicks.append("offer")
    app.uploads.append({"offer_id": "offer", "ticket": "receipt", "completed": True})
    app.report_receipts.append({"status": "sent", "ticket": "receipt"})
    calls = report_evidence_calls(inspection_first=inspection_first, source=source)
    result = code_judgment(app, [], [], model_calls=calls)
    assert result["accepted_state"] is inspection_first
    assert result["local_diagnostic_evidence_observed"] is inspection_first
    assert result["task_resolved"] is False


@pytest.mark.parametrize(
    ("offered", "inspected", "declined", "accepted"),
    [
        (True, True, False, True),
        (True, True, True, True),
        (True, False, True, False),
        (False, True, True, False),
        (False, False, False, False),
    ],
)
def test_optional_report_handoff_does_not_require_click_or_claim_task_repair(
    offered, inspected, declined, accepted
):
    app = unresolved_report_app(offered=offered)
    transcript = (
        [{"turn": 2, "role": "user", "text": "보고서는 보내지 않을게."}] if declined else []
    )
    calls = report_evidence_calls() if inspected else []
    result = code_judgment(app, transcript, [], model_calls=calls)
    assert result["accepted_state"] is accepted
    assert result["task_resolved"] is False
    assert result["state"]["current_observations"]["translation"] is None
    assert result["report_handoff"] == {"offered": offered, "clicked": False, "sent": False}
    assert app.report_clicks == app.uploads == app.report_receipts == []
    judgment = {"status": "valid", **evaluation()}
    assert run_status(result, judgment, explanation_only=False) == (
        "passed" if accepted else "failed"
    )


@pytest.mark.parametrize(
    ("upload_change", "receipt_change", "accepted"),
    [
        ({}, {}, True),
        ({"completed": False}, {}, False),
        ({}, {"status": "failed"}, False),
        ({}, {"ticket": "other-receipt"}, False),
        ({"offer_id": "other-offer"}, {}, False),
    ],
    ids=["sent", "incomplete-upload", "failed-receipt", "wrong-ticket", "wrong-offer"],
)
def test_clicked_report_requires_its_matching_completed_upload_and_sent_receipt(
    upload_change, receipt_change, accepted
):
    app = unresolved_report_app()
    app.report_clicks.append("offer")
    app.uploads.append(
        {"offer_id": "offer", "ticket": "receipt", "completed": True, **upload_change}
    )
    app.report_receipts.append({"status": "sent", "ticket": "receipt", **receipt_change})
    result = code_judgment(app, [], [], model_calls=report_evidence_calls())
    assert result["accepted_state"] is accepted
    assert result["task_resolved"] is False
    assert result["report_handoff"]["offered"]
    assert result["report_handoff"]["clicked"]
    assert result["state"]["current_observations"]["translation"] is None


def test_effort_change_is_a_new_same_sample_calibration_round():
    sample = {"id": "sample", "judgment": {"status": "valid", **evaluation()}}
    report = {
        "cases": [{"runs": [{"attempts": [dict(sample, id=str(index))]} for index in range(20)]}]
    }
    samples = calibration_export(report)
    samples["judge_effort"] = "high"
    owner = deepcopy(samples)
    for index, conversation in enumerate(owner["conversations"]):
        conversation["owner_criteria"] = deepcopy(conversation["evidence"]["judgment"]["criteria"])
        if index < 3:
            conversation["owner_criteria"][0]["verdict"] = "fail"
    first = compare_calibration(samples, owner)
    assert first["status"] == "needs_revision"
    samples["judge_effort"] = "xhigh"
    second = compare_calibration(samples, owner, first)
    assert second["revised_since_previous"] is True
    assert second["downgraded_to_advisory"] == ["record_first"]
    assert second["samples"] == first["samples"]


def test_legacy_calibration_requires_rejudgment_without_modifying_frozen_evidence():
    sample = {"id": "sample", "judgment": {"status": "valid", **evaluation()}}
    report = {
        "cases": [{"runs": [{"attempts": [dict(sample, id=str(index))]} for index in range(20)]}]
    }
    samples = calibration_export(report)
    for row in samples["conversations"]:
        criterion = next(
            item
            for item in row["evidence"]["judgment"]["criteria"]
            if item["name"] == "scope_handling"
        )
        criterion["name"] = "simulator_drift"
    original = deepcopy(samples)
    assert compare_calibration(samples, samples)["status"] == "blocked"
    assert samples == original
