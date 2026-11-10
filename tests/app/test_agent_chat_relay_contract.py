from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from puripuly_heart.app.adapters.agent_relay_transport import AgentRelayTransport
from puripuly_heart.app.ports.agent_chat import AGENT_CHAT_PROTOCOL
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from tests.app.test_agent_chat import Records, Tokens, answer, tool
from tests.app.test_agent_chat_routing import selected_decision

FIXTURES = Path(__file__).resolve().parents[2] / "agent-relay/tests/app-requests"
_VALUES = frozenset(
    {"agent_protocol", "route", "effort", "allow_tools", "type", "role", "name", "status"}
)


def _shape(value, key=None):
    if isinstance(value, dict):
        return {name: _shape(item, name) for name, item in sorted(value.items())}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return value if key in _VALUES else type(value).__name__


async def _app_requests() -> list[dict]:
    decisions = [
        selected_decision("output"),
        selected_decision(route="social", effort="low"),
        selected_decision(route="out_of_scope", effort="low"),
    ]
    responses = [
        {
            "status": "completed",
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "encrypted_content": "synthetic",
                    "summary": [],
                },
                {
                    **tool(),
                    "name": "read_app",
                    "arguments": json.dumps({"topic": "output"}),
                },
            ],
        },
        answer(),
        answer("You're welcome."),
        answer("I can only help with this app."),
    ]
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append({"path": request.url.path, "body": body})
        if request.url.path == "/v1/classify":
            return httpx.Response(200, json=decisions.pop(0))
        event = {"type": "completed", "response": responses.pop(0)}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(event)}\n\n".encode(),
        )

    transport = AgentRelayTransport("https://relay.example/v1/respond", Tokens())
    await transport._client.aclose()
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    owner = AgentChatOwner(Records(), transport, localize=str)
    await owner.start()
    try:
        for text in ("Why are captions missing?", "Thanks!", "Write me a poem."):
            assert (await owner.send(text)).status == "completed"
    finally:
        await owner.close()
    assert not decisions and not responses
    turns = {}
    for request in requests:
        turn_id = request["body"]["turn_id"]
        request["body"]["turn_id"] = turns.setdefault(turn_id, f"{len(turns) + 1:032x}")
    return requests


@pytest.mark.asyncio
async def test_current_app_requests_match_the_relay_contract_fixture():
    requests = await _app_requests()
    assert [request["path"] for request in requests] == [
        "/v1/classify",
        "/v1/respond",
        "/v1/respond",
        "/v1/classify",
        "/v1/respond",
        "/v1/classify",
        "/v1/respond",
    ]
    assert {request["body"]["agent_protocol"] for request in requests} == {AGENT_CHAT_PROTOCOL}
    fixture = FIXTURES / f"protocol-{AGENT_CHAT_PROTOCOL}.json"
    if os.environ.get("PURI_RECORD_RELAY_FIXTURES") == "1":
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_bytes(
            (json.dumps(requests, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
    recorded = json.loads(fixture.read_text(encoding="utf-8"))
    assert _shape(requests) == _shape(recorded)
