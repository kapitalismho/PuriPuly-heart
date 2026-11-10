from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from puripuly_heart.app.adapters.agent_relay_transport import AgentRelayTransport
from puripuly_heart.app.ports.agent_chat import (
    AGENT_CHAT_PROTOCOL,
    AgentServiceError,
    AgentTransportError,
)
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.ui.i18n import available_locales, t_for_locale
from tests.app.test_agent_chat import Records, Responses, Tokens, answer, context_of, tool
from tests.app.test_agent_chat_routing import decision


class RoutedResponses(Responses):
    def __init__(self, routing, *responses):
        super().__init__(*responses)
        self.routing = routing

    async def classify(self, **request):
        self.classifications.append(deepcopy(request))
        if isinstance(self.routing, Exception):
            raise self.routing
        return self.routing


def owner_for(transport, records=None, **kwargs):
    return AgentChatOwner(
        records or Records(),
        transport,
        localize=lambda key: t_for_locale("ko", key),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "effort", "effective"),
    [
        ("work", "high", "high"),
        ("work", "medium", "medium"),
        ("work", "low", "low"),
        ("unclear", "high", "high"),
        ("social", "medium", "low"),
        ("out_of_scope", "medium", "low"),
    ],
)
async def test_route_controls_model_invocation_and_whole_turn_effort(route, effort, effective):
    responses = [{"status": "completed", "output": [tool()]}] if route == "work" else []
    transport = RoutedResponses(
        decision(route, effort), *responses, answer("확인할 수 있는 기록을 안내해요.")
    )
    owner = owner_for(transport)
    await owner.start()
    try:
        result = await owner.send("synthetic routing message")
        assert result.status == "completed"
        assert len(transport.classifications) == 1
        sent = transport.classifications[0]["input"][0]["content"]
        assert "app_context" not in sent and "PRIVATE-" not in sent
        assert len(transport.requests) == (2 if route == "work" else 1)
        assert all(request["effort"] == effective for request in transport.requests)
        assert all(request["route"] == route for request in transport.requests)
        assert transport.requests[0]["allow_tools"] is (route in {"work", "unclear"})
        if route in {"social", "out_of_scope"}:
            assert transport.requests[0]["input"] == [
                {"role": "user", "content": "synthetic routing message"}
            ]
            assert "app_context" not in json.dumps(owner._items)
        else:
            assert context_of(transport.requests[0])["core"]["revision"] == 42
        updates = [item for item in owner._items if item.get("type") == "configuration_update"]
        assert bool(updates) is (route in {"work", "unclear"})
        for request in transport.requests:
            transmitted = [
                item for item in request["input"] if item.get("type") == "configuration_update"
            ]
            if route in {"social", "out_of_scope"}:
                assert not transmitted
            else:
                assert transmitted[-1]["reasoning"]["effort"] == effective
                context_index = next(
                    index
                    for index, item in enumerate(request["input"])
                    if item.get("content", "").startswith('{"app_context":')
                )
                assert request["input"][context_index - 1] is transmitted[-1]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_effective_effort_history_survives_social_and_tool_continuations():
    def effective_inputs(items):
        effort = None
        observed = []
        for item in items:
            if item.get("type") == "configuration_update":
                effort = item["reasoning"]["effort"]
            elif item.get("role") == "user":
                content = item["content"]
                label = "app_context" if content.startswith('{"app_context":') else content
                observed.append((label, effort))
            elif item.get("type") == "function_call_output":
                observed.append((item["call_id"], effort))
        return observed

    transport = RoutedResponses(
        decision("work", "medium"),
        {"status": "completed", "output": [tool("capture")]},
        answer("Capture checked."),
        answer("You're welcome."),
        {"status": "completed", "output": [tool("output")]},
        answer("Output checked."),
        answer("Settings checked."),
    )
    owner = owner_for(transport)
    await owner.start()
    try:
        await owner.send("Inspect capture.")
        medium = [
            ("app_context", "medium"),
            ("Inspect capture.", "medium"),
            ("capture", "medium"),
        ]
        assert effective_inputs(transport.requests[0]["input"]) == medium[:-1]
        assert effective_inputs(transport.requests[1]["input"]) == medium
        transport.routing = decision("social", "high")
        await owner.send("Thanks!")
        assert transport.requests[-1]["input"] == [
            {"role": "user", "content": "Inspect capture."},
            {"role": "assistant", "content": "Capture checked."},
            {"role": "user", "content": "Thanks!"},
        ]
        transport.routing = decision("unclear", "high")
        await owner.send("Inspect output.")
        high = medium + [
            ("Thanks!", "medium"),
            ("app_context", "high"),
            ("Inspect output.", "high"),
            ("output", "high"),
        ]
        assert effective_inputs(transport.requests[3]["input"]) == high[:-1]
        assert effective_inputs(transport.requests[4]["input"]) == high
        transport.routing = decision("work", "low")
        await owner.send("Inspect settings.")
        assert effective_inputs(transport.requests[-1]["input"]) == high + [
            ("app_context", "low"),
            ("Inspect settings.", "low"),
        ]
        assert [request["effort"] for request in transport.requests] == [
            "medium",
            "medium",
            "low",
            "high",
            "high",
            "low",
        ]
        assert "configuration_update" not in json.dumps(transport.classifications)
        assert not any(item.get("type") == "configuration_update" for item in owner.report_archive)
        assert [message.text for message in owner.snapshot.messages if message.role == "user"] == [
            "Inspect capture.",
            "Thanks!",
            "Inspect output.",
            "Inspect settings.",
        ]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_reset_during_context_collection_cannot_record_stale_effort(monkeypatch):
    collected, release = asyncio.Event(), asyncio.Event()
    transport = RoutedResponses(decision("work", "high"), answer("Current context."))
    owner = owner_for(transport)
    collect_context = owner._context
    first = True

    async def held_context(**arguments):
        nonlocal first
        context = await collect_context(**arguments)
        if first:
            first = False
            collected.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
        return context

    monkeypatch.setattr(owner, "_context", held_context)
    await owner.start()
    pending = asyncio.create_task(owner.send("Stale high-effort task."))
    try:
        await collected.wait()
        assert not owner._items
        owner.new_conversation()
        release.set()
        assert (await pending).status == "cancelled"
        assert not owner._items and not transport.requests
        transport.routing = decision("work", "low")
        assert (await owner.send("Current low-effort task.")).status == "completed"
        updates = [
            item["reasoning"]["effort"]
            for item in transport.requests[-1]["input"]
            if item.get("type") == "configuration_update"
        ]
        assert updates == ["low"]
    finally:
        release.set()
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary_routing",
    [
        decision("out_of_scope", "high"),
        {"answers": [{"name": "route", "type": "refusal"}]},
        {
            "answers": [
                {"name": "route", "type": "choice", "choice": "work"},
                {"name": "effort", "type": "refusal"},
            ]
        },
    ],
)
async def test_off_topic_after_work_is_private_without_waiting_and_preserves_work_continuation(
    boundary_routing,
):
    context_started, release_context = asyncio.Event(), asyncio.Event()

    class HeldRecords(Records):
        hold = False

        async def query(self, name, arguments):
            if self.hold:
                context_started.set()
                await release_context.wait()
            return await super().query(name, arguments)

    class Router(RoutedResponses):
        async def classify(self, **request):
            if self.routing == boundary_routing:
                await context_started.wait()
            return await super().classify(**request)

    work_reasoning = {"type": "reasoning", "id": "work", "encrypted_content": "WORK-REASONING"}
    boundary_reasoning = {**work_reasoning, "encrypted_content": "BOUNDARY-REASONING"}
    transport = Router(
        decision("work", "high"),
        {"status": "completed", "output": [work_reasoning, tool()]},
        answer("Prior app evidence."),
        {
            "status": "completed",
            "output": [boundary_reasoning, *answer("Generated redirect.")["output"]],
        },
        answer("Continued app work."),
    )
    records = HeldRecords()
    owner = owner_for(transport, records)
    await owner.start()
    try:
        await owner.send("Investigate the synthetic app failure.")
        previous_input = deepcopy(owner._items)
        records.hold = True
        transport.routing = boundary_routing
        async with asyncio.timeout(2):
            result = await owner.send("Give me a pasta recipe.")
        assert result.status == "completed" and result.text == "Generated redirect."
        assert context_started.is_set() and not release_context.is_set()
        assert transport.requests[-1] == {
            "input": [{"role": "user", "content": "Give me a pasta recipe."}],
            "turn_id": transport.classifications[-1]["turn_id"],
            "effort": "low",
            "route": "out_of_scope",
            "allow_tools": False,
        }
        assert len(transport.requests) == 3
        assert owner.snapshot.messages[-1].role == "assistant"
        records.hold = False
        release_context.set()
        transport.routing = decision("work", "high")
        await owner.send("Continue the app investigation.")
        continued = transport.requests[-1]["input"]
        assert continued[: len(previous_input)] == previous_input
        assert {"role": "user", "content": "Give me a pasta recipe."} in continued
        assert {"role": "assistant", "content": "Generated redirect."} in continued
        assert "WORK-REASONING" in json.dumps(continued)
        assert "BOUNDARY-REASONING" not in json.dumps(continued)
        assert len(transport.classifications) == 3
    finally:
        release_context.set()
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["social", "out_of_scope"])
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("open_settings", '{"section":"overlay"}'),
        ("probe_translation", "{}"),
    ],
)
async def test_brief_routes_never_execute_unexpected_tool_calls(route, name, arguments):
    transport = RoutedResponses(
        decision(route, "high"),
        {
            "status": "completed",
            "output": [tool(arguments=arguments) | {"name": name}],
        },
    )
    owner = owner_for(transport)
    navigations = []
    owner.bind_navigation(lambda *args: navigations.append(args) or {"opened": True})
    await owner.start()
    try:
        result = await owner.send("Synthetic brief response.")
        assert result.status == "interrupted"
        assert len(transport.requests) == 1
        assert transport.requests[0]["allow_tools"] is False
        assert transport.requests[0]["effort"] == "low"
        assert not navigations
        if route == "out_of_scope":
            assert not any(item.get("type") == "function_call" for item in owner._items)
        else:
            assert result.tool_results[0].status == "not_executed"
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        '{"action":"translation.probe"}',
        '{"action":"translation.probe","text":"not allowed"}',
        "null",
        "[]",
    ],
)
async def test_probe_tool_requires_empty_arguments_before_owner_dispatch(monkeypatch, arguments):
    transport = RoutedResponses(
        decision("work", "high"),
        {
            "status": "completed",
            "output": [tool(arguments=arguments) | {"name": "execute_app"}],
        },
        answer("Observed the tool result."),
    )
    owner = owner_for(transport)
    dispatched = []
    receipt = {"status": "applied", "receipt": {"terminal": True, "status": "applied"}}

    async def execute(name, values, *, require_current, request_id=None):
        require_current()
        dispatched.append((name, values))
        return receipt

    monkeypatch.setattr(owner._tools, "execute", execute)
    await owner.start()
    try:
        result = await owner.send("Check translation using the available app tool.")
        assert result.status == "completed"
        assert len(transport.classifications) == 1 and len(transport.requests) == 2
        assert all(request["effort"] == "high" for request in transport.requests)
        output = json.loads(
            next(
                item["output"]
                for item in transport.requests[-1]["input"]
                if item.get("type") == "function_call_output"
            )
        )
        if arguments == '{"action":"translation.probe"}':
            assert dispatched == [("probe_translation", {})]
            assert output == receipt
        else:
            assert not dispatched
            assert output == {"status": "rejected", "code": "invalid_tool_call"}
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["cancel", "message"])
async def test_late_off_topic_response_cannot_publish_or_act_after_interruption(interrupt):
    entered, release = asyncio.Event(), asyncio.Event()

    class LateResponse(RoutedResponses):
        async def respond(self, **request):
            self.requests.append(deepcopy(request))
            if len(self.requests) == 1:
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                yield {"type": "completed", "response": answer("LATE-BOUNDARY")}
                return
            yield {"type": "completed", "response": answer("Current work.")}

    transport = LateResponse(decision("out_of_scope", "high"))
    owner = owner_for(transport)
    await owner.start()
    first = asyncio.create_task(owner.send("Synthetic unrelated request."))
    try:
        await entered.wait()
        if interrupt == "cancel":
            owner.cancel()
        else:
            transport.routing = decision("work", "high")
            assert (await owner.send("Current app request.")).text == "Current work."
        release.set()
        assert (await first).status == "cancelled"
        assert "LATE-BOUNDARY" not in json.dumps(owner._items)
        assert all(message.text != "LATE-BOUNDARY" for message in owner.snapshot.messages)
        assert not owner.snapshot.busy
    finally:
        release.set()
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (AgentTransportError(), "failed"),
        (AgentServiceError("service_authentication_failed"), "service_unavailable"),
        (AgentServiceError("service_allowance_exhausted"), "service_unavailable"),
        (
            {
                "status": "incomplete",
                "output": [
                    {"type": "reasoning", "encrypted_content": "BOUNDARY-REASONING"},
                    tool(),
                ],
            },
            "interrupted",
        ),
    ],
)
async def test_off_topic_failure_keeps_exchange_recoverable_without_retry(failure, status):
    transport = RoutedResponses(decision("out_of_scope", "high"), failure, answer("Recovered."))
    owner = owner_for(transport)
    await owner.start()
    try:
        failed = await owner.send("Synthetic unrelated request.")
        assert failed.status == status
        assert len(transport.requests) == 1
        assert "BOUNDARY-REASONING" not in json.dumps(owner._items)
        assert not any(item.get("type") == "function_call" for item in owner._items)
        transport.routing = decision("work", "high")
        assert (await owner.send("Return to my app problem.")).status == "completed"
        continued = transport.requests[-1]["input"]
        assert {"role": "user", "content": "Synthetic unrelated request."} in continued
        assert {"role": "assistant", "content": failed.text} in continued
        assert len(transport.classifications) == len(transport.requests) == 2
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_classification_and_local_records_start_together():
    context_started, classifier_started = asyncio.Event(), asyncio.Event()

    class ConcurrentRecords(Records):
        async def query(self, name, arguments):
            context_started.set()
            await classifier_started.wait()
            return await super().query(name, arguments)

    class ConcurrentRouter(RoutedResponses):
        async def classify(self, **request):
            classifier_started.set()
            await context_started.wait()
            return await super().classify(**request)

    transport = ConcurrentRouter(decision(), answer())
    owner = owner_for(transport, ConcurrentRecords())
    await owner.start()
    try:
        async with asyncio.timeout(2):
            assert (await owner.send("synthetic failure")).status == "completed"
        assert context_started.is_set() and classifier_started.is_set()
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("routing", "expected_route"),
    [
        (AgentTransportError(), "unclear"),
        (decision("invalid", "high"), "unclear"),
        (decision("work", "max"), "work"),
        (decision("work", "xhigh"), "work"),
        (decision("unclear", None), "unclear"),
    ],
)
async def test_failed_classification_falls_back_once_without_retry(routing, expected_route):
    transport = RoutedResponses(routing, answer())
    owner = owner_for(transport)
    await owner.start()
    try:
        assert (await owner.send("synthetic request")).status == "completed"
        assert len(transport.classifications) == len(transport.requests) == 1
        assert transport.requests[0]["route"] == expected_route
        assert transport.requests[0]["effort"] == "high"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_timeout_late_decision_cannot_override_model_or_next_generation():
    release = asyncio.Event()
    late_finished = asyncio.Event()

    class LateRouter(RoutedResponses):
        async def classify(self, **request):
            self.classifications.append(deepcopy(request))
            if len(self.classifications) == 1:
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                late_finished.set()
                return decision("out_of_scope", "low")
            return decision("work", "medium")

        async def respond(self, **request):
            async for event in super().respond(**request):
                if len(self.requests) == 2:
                    release.set()
                    await late_finished.wait()
                yield event

    transport = LateRouter(None, answer("First answer"), answer("Second answer"))
    owner = owner_for(transport, decision_timeout=0.01)
    await owner.start()
    try:
        first = await owner.send("first synthetic request")
        assert first.text == "First answer"
        assert transport.requests[0]["effort"] == "high"
        second = await owner.send("second synthetic request")
        assert second.text == "Second answer"
        assert [request["effort"] for request in transport.requests] == ["high", "medium"]
        assert len(transport.classifications) == 2
        assert owner.snapshot.messages[-1].text == "Second answer"
    finally:
        release.set()
        await owner.close()


@pytest.mark.asyncio
async def test_error_and_notice_entries_bypass_classifier_and_reuse_effort():
    transport = RoutedResponses(decision("out_of_scope"), answer(), answer(), answer())
    owner = owner_for(transport)
    await owner.start()
    try:
        assert (
            await owner.send("Explain this app error", diagnostic_id="synthetic-error")
        ).status == "completed"
        assert (await owner.send("Give me a pasta recipe.")).status == "completed"
        owner.bind_verification_providers(frozenset({"openai"}))
        await owner._tools.execute("open_settings", {"section": "api", "provider": "openai"})
        owner.key_verification_completed("openai", True)
        await owner._notice_task
        assert len(transport.classifications) == 1
        assert [request["effort"] for request in transport.requests] == ["high", "low", "high"]
        updates = [
            item["reasoning"]["effort"]
            for item in transport.requests[-1]["input"]
            if item.get("type") == "configuration_update"
        ]
        assert updates == ["high", "high"]
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
async def test_initial_and_reset_notice_defaults_are_high_after_a_brief_route(reset):
    transport = RoutedResponses(decision("work", "medium"), answer(), answer(), answer())
    owner = owner_for(transport)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        if reset:
            assert (await owner.send("Inspect the previous app problem.")).status == "completed"
            owner.new_conversation()
        transport.routing = decision("out_of_scope", "medium")
        assert (await owner.send("Give me a pasta recipe.")).status == "completed"
        await owner._tools.execute("open_settings", {"section": "api", "provider": "openai"})
        owner.key_verification_completed("openai", True)
        await owner._notice_task
        assert [request["effort"] for request in transport.requests] == (
            (["medium"] if reset else []) + ["low", "high"]
        )
        assert len(transport.classifications) == (2 if reset else 1)
        assert [
            item["reasoning"]["effort"]
            for item in transport.requests[-1]["input"]
            if item.get("type") == "configuration_update"
        ] == ["high"]
        assert "Inspect the previous app problem." not in json.dumps(transport.requests[-1])
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
async def test_notice_continuation_preserves_the_selected_work_effort(effort):
    transport = RoutedResponses(decision("work", effort), answer(), answer())
    owner = owner_for(transport)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        assert (await owner.send("Inspect the app setup.")).status == "completed"
        await owner._tools.execute("open_settings", {"section": "api", "provider": "openai"})
        owner.key_verification_completed("openai", True)
        await owner._notice_task
        assert len(transport.classifications) == 1
        assert [request["effort"] for request in transport.requests] == [effort, effort]
        assert [
            item["reasoning"]["effort"]
            for item in transport.requests[-1]["input"]
            if item.get("type") == "configuration_update"
        ] == [effort, effort]
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        "service_authentication_failed",
        "service_allowance_exhausted",
        "service_shutdown",
        "unsupported_protocol",
    ],
)
async def test_service_terminal_transport_outcomes_never_fall_back_to_model(code):
    seen = []
    transport = AgentRelayTransport(
        "http://127.0.0.1:8788/v1/respond", Tokens(), allow_loopback_http=True
    )
    await transport._client.aclose()

    def handle(request):
        seen.append(request)
        return httpx.Response(
            400 if code == "unsupported_protocol" else 503,
            json={
                "error": {
                    "code": "unsupported_agent_protocol" if code == "unsupported_protocol" else code
                }
            },
        )

    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    owner = owner_for(transport)
    await owner.start()
    try:
        result = await owner.send("synthetic request")
        assert result.status == "service_unavailable"
        assert result.text == t_for_locale("ko", f"agent_chat.{code}")
        assert [request.url.path for request in seen] == ["/v1/classify"]
        assert json.loads(seen[0].content)["agent_protocol"] == AGENT_CHAT_PROTOCOL
        assert seen[0].headers["authorization"] == "Bearer synthetic-installation-token"
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        "service_authentication_failed",
        "service_allowance_exhausted",
        "service_shutdown",
        "unsupported_protocol",
    ],
)
async def test_terminal_classification_retains_request_for_contextual_recovery(code):
    seen = []
    original = "Make the overlay background more opaque."
    retry = "Please try that again."
    guidance = t_for_locale("ko", f"agent_chat.{code}")
    transport = AgentRelayTransport(
        "http://127.0.0.1:8788/v1/respond", Tokens(), allow_loopback_http=True
    )
    await transport._client.aclose()

    def handle(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(
                400 if code == "unsupported_protocol" else 503,
                json={
                    "error": {
                        "code": (
                            "unsupported_agent_protocol" if code == "unsupported_protocol" else code
                        )
                    }
                },
            )
        if request.url.path == "/v1/classify":
            return httpx.Response(200, json=decision("work", "high"))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: "
            + json.dumps(
                {
                    "type": "completed",
                    "response": answer("I can continue from your overlay request."),
                }
            )
            + "\n\n",
        )

    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    owner = owner_for(transport)
    await owner.start()
    try:
        failed = await owner.send(original)
        assert failed.status == "service_unavailable"
        assert failed.text == guidance
        assert [request.url.path for request in seen] == ["/v1/classify"]
        assert "app_context" not in json.dumps(owner._items)

        recovered = await owner.send(retry)
        assert recovered.status == "completed"
        assert [request.url.path for request in seen] == [
            "/v1/classify",
            "/v1/classify",
            "/v1/respond",
        ]
        classification = json.loads(seen[1].content)
        assert original in classification["input"][0]["content"]
        assert "app_context" not in json.dumps(classification)
        response = json.loads(seen[2].content)
        conversation = [
            item
            for item in response["input"]
            if item.get("role") in {"user", "assistant"}
            and not item.get("content", "").startswith('{"app_context":')
        ]
        assert conversation == [
            {"role": "user", "content": original},
            {"role": "assistant", "content": guidance},
            {"role": "user", "content": retry},
        ]
        assert context_of(response)["core"]["revision"] == 42
        assert response["effort"] == "high"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_shutdown_during_classification_never_starts_model():
    started, release = asyncio.Event(), asyncio.Event()

    class ClosingRouter(RoutedResponses):
        async def classify(self, **request):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return decision()

    transport = ClosingRouter(None)
    owner = owner_for(transport)
    await owner.start()
    active = asyncio.create_task(owner.send("synthetic request"))
    await started.wait()
    owner.freeze_ingress()
    release.set()
    await owner.close()
    assert active.cancelled()
    assert not transport.requests
    assert transport.closed


def test_static_service_guidance_exists_for_every_app_locale():
    for locale in available_locales():
        for suffix in (
            "service_authentication_failed",
            "service_allowance_exhausted",
            "service_shutdown",
            "unsupported_protocol",
            "key_blocked",
        ):
            key = f"agent_chat.{suffix}"
            assert t_for_locale(locale, key) not in ("", key)
