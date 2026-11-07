from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from puripuly_heart.app.adapters.agent_relay_transport import AgentRelayTransport
from puripuly_heart.app.ports.agent_chat import AgentTransportError
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.app.services.agent_chat.presentation import completed_sentences
from tests.app.test_agent_chat import Records, Responses, Tokens, answer, tool


class HeldWire(httpx.AsyncByteStream):
    def __init__(self):
        self.chunks = asyncio.Queue()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        while True:
            event = await self.chunks.get()
            if event is None:
                return
            yield ("data: " + json.dumps(event) + "\n\n").encode()

    async def aclose(self):
        self.closed.set()

    def emit(self, event):
        self.chunks.put_nowait(event)


@pytest.mark.asyncio
async def test_held_http_stream_presents_sentence_before_terminal_without_committing_output():
    wire = HeldWire()
    transport = AgentRelayTransport("https://relay.example/v1/respond", Tokens())
    await transport._client.aclose()

    def handle(request):
        assert json.loads(request.content)["agent_protocol"] == 4
        if request.url.path == "/v1/classify":
            return httpx.Response(
                200,
                json={
                    "answers": [
                        {"type": "choice", "name": "route", "choice": "work"},
                        {"type": "choice", "name": "effort", "choice": "high"},
                    ]
                },
            )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=wire)

    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    owner = AgentChatOwner(Records(), transport, localize=str)
    presented = asyncio.Event()
    snapshots = []

    def observe(snapshot):
        snapshots.append(snapshot)
        if snapshot.presentation is not None and snapshot.presentation.sentences:
            presented.set()

    owner.observe(observe)
    await owner.start()
    pending = asyncio.create_task(owner.send("Synthetic request"))
    wire.emit({"type": "text_delta", "text": "First sentence. Incomplete"})
    try:
        await asyncio.wait_for(presented.wait(), 5)
        snapshot = owner.snapshot
        assert snapshot.busy and not pending.done()
        assert snapshot.presentation.sentences == ("First sentence.",)
        assert not snapshot.presentation.completed
        assert not snapshot.tool_results
        assert all(message.role != "assistant" for message in snapshot.messages)
        assert not any(
            item.get("type") in {"reasoning", "function_call", "message"} for item in owner._items
        )
        final = answer("First sentence. Incomplete final fragment")
        final["output"].insert(
            0, {"type": "reasoning", "encrypted_content": "NEVER-PRESENT", "summary": []}
        )
        wire.emit({"type": "completed", "response": final})
        wire.chunks.put_nowait(None)
        result = await pending
        assert result.status == "completed"
        assert owner.snapshot.presentation.completed
        assert owner.snapshot.presentation.sentences == (
            "First sentence.",
            "Incomplete final fragment",
        )
        assert [
            message.text for message in owner.snapshot.messages if message.role == "assistant"
        ] == [result.text]
        assert "NEVER-PRESENT" not in repr(snapshots)
        assert "NEVER-PRESENT" not in repr(owner.report_archive)
        assert wire.closed.is_set()
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["cancel", "new_conversation", "error"])
async def test_provisional_sentences_clear_on_interruption_without_tool_dispatch(interrupt):
    entered, release = asyncio.Event(), asyncio.Event()

    class Held(Responses):
        async def respond(self, **request):
            yield {"type": "text_delta", "text": "Provisional sentence."}
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            if interrupt == "error":
                raise AgentTransportError()
            yield {"type": "completed", "response": {"status": "completed", "output": [tool()]}}

    owner = AgentChatOwner(Records(), Held(), localize=str)
    await owner.start()
    pending = asyncio.create_task(owner.send("Synthetic request"))
    try:
        await entered.wait()
        assert owner.snapshot.presentation.sentences == ("Provisional sentence.",)
        if interrupt != "error":
            getattr(owner, interrupt)()
        release.set()
        result = await pending
        assert result.status == ("failed" if interrupt == "error" else "cancelled")
        assert owner.snapshot.presentation is None
        assert not owner.snapshot.tool_results
        assert not any(item.get("type") == "function_call" for item in owner._items)
        assert all(message.text != "Provisional sentence." for message in owner.snapshot.messages)
    finally:
        release.set()
        await owner.close()


@pytest.mark.asyncio
async def test_tool_arguments_remain_unexecuted_until_authoritative_terminal():
    entered, release = asyncio.Event(), asyncio.Event()

    class Held(Responses):
        async def respond(self, **request):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield {"type": "text_delta", "text": "I will check the app."}
                entered.set()
                await release.wait()
                yield {"type": "completed", "response": {"status": "completed", "output": [tool()]}}
            else:
                yield {"type": "completed", "response": answer()}

    records = Records()
    owner = AgentChatOwner(records, Held(), localize=str)
    await owner.start()
    pending = asyncio.create_task(owner.send("Synthetic request"))
    try:
        await entered.wait()
        before = len(records.queries)
        assert not owner.snapshot.tool_results
        release.set()
        result = await pending
        assert len(records.queries) > before
        assert [item.status for item in result.tool_results] == ["observed"]
        assert len([item for item in owner._items if item.get("type") == "function_call"]) == 1
    finally:
        release.set()
        await owner.close()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Costs $3.14. Next fragment", ("Costs $3.14.",)),
        ("Price $3.", ()),
        ("Visit https://example.com/path?q=3.14", ()),
        ("Open https://example.com/docs?", ()),
        ("Open https://example.com/path!", ()),
        ("Visit www.example.com/docs?", ()),
        ("Ready?", ("Ready?",)),
        ("Done!", ("Done!",)),
        ("She said “Ready!” Next", ("She said “Ready!”",)),
        ("1. First item.\n2. Next fragment", ("1. First item.",)),
        ("Use `x.y()` safely. Next", ("Use `x.y()` safely.",)),
        ("```python\nx = 3.14\n```", ()),
        ("완료했어요。 다음", ("완료했어요。",)),
    ],
)
def test_sentence_boundaries_preserve_structured_text(text, expected):
    assert completed_sentences(text) == expected
    assert "".join(completed_sentences(text, final=True)).replace(" ", "").replace(
        "\n", ""
    ) == text.replace(" ", "").replace("\n", "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        [{"type": "text_delta", "text": "Not committed."}],
        [{"type": "completed", "response": answer()}, {"type": "completed", "response": answer()}],
        [
            {"type": "text_delta", "text": "Not committed."},
            {"type": "error", "error": {"code": "upstream_unavailable"}},
        ],
        [{"type": "reasoning_delta", "text": "NEVER-PRESENT"}],
    ],
)
async def test_transport_rejects_missing_duplicate_and_unsafe_terminal_streams(events):
    transport = AgentRelayTransport("https://relay.example/v1/respond", Tokens())
    await transport._client.aclose()
    transport._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="".join("data: " + json.dumps(event) + "\n\n" for event in events),
            )
        )
    )
    try:
        with pytest.raises(AgentTransportError):
            [
                event
                async for event in transport.respond(
                    input=[
                        {"type": "configuration_update", "reasoning": {"effort": "high"}},
                        {"role": "user", "content": "Inspect the current app state."},
                    ],
                    turn_id="0123456789abcdef0123456789abcdef",
                    effort="high",
                    route="work",
                    allow_tools=True,
                )
            ]
    finally:
        await transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("classification", "source", "reason"),
    [
        ({"answers": []}, "fallback", "invalid_route"),
        ({"answers": [{"type": "refusal"}]}, "fallback", "classification_refused"),
        (AgentTransportError("relay_request_failed"), "fallback", "classify_transport_error"),
        (
            {
                "answers": [
                    {"type": "choice", "name": "route", "choice": "out_of_scope"},
                    {"type": "choice", "name": "effort", "choice": "high"},
                ]
            },
            "model",
            None,
        ),
    ],
)
async def test_effective_decision_events_expose_safe_route_effort_and_fallback_cause(
    classification, source, reason
):
    class Classified(Responses):
        async def classify(self, **request):
            if isinstance(classification, Exception):
                raise classification
            return classification

    transport = Classified(answer())
    owner = AgentChatOwner(Records(), transport, localize=str)
    events = []
    owner.observe_events(lambda event: events.append(event))
    await owner.start()
    try:
        assert (await owner.send("PRIVATE user text")).status == "completed"
        decisions = [event for event in events if event["kind"] == "decision"]
        assert len(decisions) == 1
        event = decisions[0]
        assert event["entry"] == "message"
        assert event["source"] == source and event["reason"] == reason
        assert event["route"] == transport.requests[0]["route"]
        assert event["effort"] == transport.requests[0]["effort"]
        assert "PRIVATE" not in json.dumps(event)
    finally:
        await owner.close()
