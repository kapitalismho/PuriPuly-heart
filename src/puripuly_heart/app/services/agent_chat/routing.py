from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from puripuly_heart.app.ports.agent_chat import AgentEffort, AgentMessage, AgentRoute


@dataclass(frozen=True, slots=True)
class AgentDecision:
    route: AgentRoute = "unclear"
    effort: AgentEffort = "high"
    source: str = field(default="model", compare=False)
    reason: str | None = field(default=None, compare=False)


def extract_decision(result: object) -> AgentDecision:
    if not isinstance(result, dict) or not isinstance(result.get("answers"), list):
        return AgentDecision(source="fallback", reason="invalid_classification")
    answers = result["answers"]
    if any(isinstance(answer, dict) and answer.get("type") == "refusal" for answer in answers):
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
    invalid_effort = effort not in ("low", "high", "xhigh")
    if invalid_effort:
        effort = "high"
    return AgentDecision(
        route,
        "low" if route in {"social", "out_of_scope"} else effort,
        source="fallback" if invalid_effort else "model",
        reason="invalid_effort" if invalid_effort else None,
    )


def classification_input(
    text: str,
    *,
    recent_messages: Sequence[AgentMessage],
    active_task: str | None,
    last_puri_action: str | None,
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
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    ]
