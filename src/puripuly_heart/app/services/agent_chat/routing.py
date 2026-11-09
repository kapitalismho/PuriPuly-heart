from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from puripuly_heart.app.ports.agent_chat import AgentEffort, AgentMessage, AgentRoute
from puripuly_heart.app.services.agent_chat.modules import MODULES

_CONTEXT_QUESTIONS = frozenset(
    ("context_relation", *(f"context_{module}" for module in MODULES[1:]))
)


@dataclass(frozen=True, slots=True)
class AgentDecision:
    route: AgentRoute = "unclear"
    effort: AgentEffort = "high"
    source: str = field(default="model", compare=False)
    reason: str | None = field(default=None, compare=False)
    context_modules: tuple[str, ...] | None = None
    context_relation: str = "unclear"
    context_reason: str | None = field(default=None, compare=False)


def _context_selection(answers: list) -> dict:
    choices = {}
    for name in _CONTEXT_QUESTIONS:
        matching = [
            answer for answer in answers if isinstance(answer, dict) and answer.get("name") == name
        ]
        if len(matching) != 1 or matching[0].get("type") != "choice":
            return {"context_reason": "invalid_context_selection"}
        choices[name] = matching[0].get("choice")
    relation = choices["context_relation"]
    if relation not in ("new", "continue"):
        return {"context_reason": "unclear_context_relation"}
    modules = ["core"]
    for module in MODULES[1:]:
        choice = choices[f"context_{module}"]
        if choice not in ("include", "omit"):
            return {
                "context_relation": relation,
                "context_reason": "unclear_context_selection",
            }
        if choice == "include":
            modules.append(module)
    return {"context_modules": tuple(modules), "context_relation": relation}


def extract_decision(result: object) -> AgentDecision:
    if not isinstance(result, dict) or not isinstance(result.get("answers"), list):
        return AgentDecision(source="fallback", reason="invalid_classification")
    answers = result["answers"]
    if any(
        isinstance(answer, dict)
        and answer.get("type") == "refusal"
        and (not isinstance(answer.get("name"), str) or answer["name"] not in _CONTEXT_QUESTIONS)
        for answer in answers
    ):
        return AgentDecision(
            "out_of_scope", "low", source="fallback", reason="classification_refused"
        )
    routes = [
        answer for answer in answers if isinstance(answer, dict) and answer.get("name") == "route"
    ]
    if len(routes) != 1 or routes[0].get("type") != "choice":
        return AgentDecision(source="fallback", reason="invalid_route")
    route = routes[0].get("choice")
    if route not in ("work", "social", "out_of_scope", "unclear"):
        return AgentDecision(source="fallback", reason="invalid_route")
    efforts = [
        answer for answer in answers if isinstance(answer, dict) and answer.get("name") == "effort"
    ]
    effort = (
        efforts[0].get("choice")
        if len(efforts) == 1 and efforts[0].get("type") == "choice"
        else None
    )
    invalid_effort = effort not in ("low", "medium", "high")
    if invalid_effort:
        effort = "high"
    context = _context_selection(answers) if route == "work" and not invalid_effort else {}
    return AgentDecision(
        route,
        "low" if route in {"social", "out_of_scope"} else effort,
        source="fallback" if invalid_effort else "model",
        reason="invalid_effort" if invalid_effort else None,
        **context,
    )


def classification_input(
    text: str,
    *,
    recent_messages: Sequence[AgentMessage],
    active_task: str | None,
    last_puri_action: str | None,
    previous_modules: Sequence[str] = (),
) -> list[dict]:
    recent = [message for message in recent_messages if message.role in {"user", "assistant"}][-4:]
    return [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "current_message": text,
                    "recent_messages": [
                        {"role": message.role, "content": message.text[:600]} for message in recent
                    ],
                    "active_task": active_task[:240] if active_task else None,
                    "last_puri_action": last_puri_action[:240] if last_puri_action else None,
                    "previous_modules": [
                        module for module in MODULES if module in previous_modules
                    ],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    ]
