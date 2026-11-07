from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from puripuly_heart.core import network_clients
from puripuly_heart.core.external_network import ProxyPolicy
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.release_evidence.agent_chat_simulator import __main__ as cli
from puripuly_heart.release_evidence.agent_chat_simulator import models
from puripuly_heart.release_evidence.agent_chat_simulator.models import (
    SimulatorModelError,
    SimulatorModels,
    SubscriptionConnectGate,
)

_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_USAGE = {
    "input_tokens": 1000,
    "output_tokens": 100,
    "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 100},
    "output_tokens_details": {"reasoning_tokens": 60},
    "total_tokens": 1100,
}
_SECRET = "sk-private-synthetic-do-not-publish"
_REAL_ACCOUNT_CONNECT = models.ChatGptAccountOwner.connect


def response(model="gpt-6.1-terra", **overrides):
    return {
        "model": model,
        "status": "completed",
        "usage": _USAGE,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": '{"ok": true}'}],
            }
        ],
        **overrides,
    }


async def complete(client, role="user"):
    return await client.complete(
        role=role, instructions="Synthetic only", payload={"visible": "test"}, schema=_SCHEMA
    )


@pytest.fixture(autouse=True)
def isolated_access(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Uncontrolled external access")

    monkeypatch.setattr(network_clients, "external_async_client", forbidden)
    monkeypatch.setattr(network_clients, "external_websocket_connect", forbidden)
    monkeypatch.setattr(models.ChatGptAccountOwner, "connect", forbidden)
    store = InMemorySecretStore()

    def benchmark_store(*, service_name):
        assert service_name == "puripuly-heart-agent-simulator"
        return store

    monkeypatch.setattr(models, "KeyringSecretStore", benchmark_store)
    return store


@pytest.fixture
def http_boundary(monkeypatch):
    original = httpx.AsyncClient
    requests, clients = [], []

    def install(handler):
        def handle(request):
            requests.append(request)
            return handler(request)

        def factory(**kwargs):
            client = original(transport=httpx.MockTransport(handle), trust_env=False, **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(network_clients, "external_async_client", factory)
        return requests, clients

    return install


@pytest.fixture
def subscription_boundary(monkeypatch):
    state = SimpleNamespace(
        events=[],
        requests=[],
        sockets=[],
        stores=[],
        connects=0,
        succeeded=True,
        code=None,
        reply=None,
        open_errors=[],
        starts=[],
        receiving=asyncio.Event(),
    )

    async def connect(owner):
        state.connects += 1
        store = owner.session.secret_store()
        assert isinstance(store, InMemorySecretStore)
        state.stores.append(store)
        if state.succeeded:
            store.set("chatgpt_refresh_token", "synthetic-refresh")
            store.set("chatgpt_client_id", "synthetic-client")
        return SimpleNamespace(
            succeeded=state.succeeded, failure_code=state.code, email="private@example.invalid"
        )

    async def access_token(session):
        return "synthetic-subscription-token"

    class Socket:
        def __init__(self):
            self.closed = False
            self.requests = []
            self.events = []

        def __await__(self):
            return self.open().__await__()

        async def open(self):
            state.starts.append(models.monotonic())
            if state.open_errors:
                raise state.open_errors.pop(0)
            return self

        async def close(self):
            self.closed = True

        async def send(self, text):
            assert not self.closed and not self.events
            request = json.loads(text)
            state.requests.append(request)
            self.requests.append(request)
            if state.reply is not None:
                self.events = list(state.reply(request))
                return
            while state.events:
                event = state.events.pop(0)
                self.events.append(event)
                if isinstance(event, (BaseException, asyncio.Future)):
                    break
                if isinstance(event, dict) and event.get("type") in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                    "error",
                }:
                    break

        async def recv(self):
            state.receiving.set()
            event = self.events.pop(0)
            if isinstance(event, asyncio.Future):
                event = await event
            if isinstance(event, BaseException):
                raise event
            return event if isinstance(event, str) else json.dumps(event)

    def websocket(url, **kwargs):
        assert url == "wss://api.openai.com/v1/responses"
        assert kwargs["additional_headers"] == {
            "Authorization": "Bearer synthetic-subscription-token"
        }
        socket = Socket()
        state.sockets.append(socket)
        return socket

    state.real_access_token = models.ChatGptSession.access_token
    monkeypatch.setattr(models.ChatGptAccountOwner, "connect", connect)
    monkeypatch.setattr(models.ChatGptSession, "access_token", access_token)
    monkeypatch.setattr(network_clients, "external_websocket_connect", websocket)
    return state


async def test_no_implicit_sign_in_or_ambient_account_access():
    client = SimulatorModels()
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == "access_unavailable"
        assert client.attempts == [
            {
                "access_path": "subscription_ws",
                "phase": "sign_in",
                "status": "not_available",
                "code": "subscription_not_attempted",
            }
        ]
    finally:
        await client.close()


async def test_persisted_login_refreshes_across_clients_without_browser(
    monkeypatch,
    isolated_access,
    subscription_boundary,
    http_boundary,
):
    store, state = isolated_access, subscription_boundary
    store.set("chatgpt_client_id", "synthetic-client")
    store.set("chatgpt_refresh_token", "previous-refresh")
    monkeypatch.setattr(models.ChatGptSession, "access_token", state.real_access_token)
    requests, clients = http_boundary(
        lambda req: httpx.Response(
            200,
            json={
                "access_token": "synthetic-subscription-token",
                "refresh_token": "rotated-refresh",
                "expires_in": 3600,
                "scope": "chatgpt.tokens.use.direct",
            },
        )
    )
    state.events = [{"type": "response.completed", "response": response()} for _ in range(2)]
    for _ in range(2):
        client = SimulatorModels(api_key="", sign_in=True)
        try:
            assert (await complete(client))["value"] == {"ok": True}
        finally:
            await client.close()
        assert store.get("chatgpt_refresh_token") == "rotated-refresh"
    assert state.connects == 0
    assert len(requests) == 2
    assert b"previous-refresh" in requests[0].content
    assert b"rotated-refresh" in requests[1].content
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("failure", ["server_error", "temporarily_unavailable"])
async def test_refresh_service_failure_allows_next_independent_call(
    monkeypatch,
    isolated_access,
    subscription_boundary,
    http_boundary,
    failure,
):
    store, state = isolated_access, subscription_boundary
    store.set("chatgpt_client_id", "synthetic-client")
    store.set("chatgpt_refresh_token", "previous-refresh")
    monkeypatch.setattr(models.ChatGptSession, "access_token", state.real_access_token)
    replies = [
        httpx.Response(503, json={"error": failure}),
        httpx.Response(
            200,
            json={
                "access_token": "synthetic-subscription-token",
                "refresh_token": "rotated-refresh",
                "expires_in": 3600,
                "scope": "chatgpt.tokens.use.direct",
            },
        ),
    ]
    requests, _ = http_boundary(lambda req: replies.pop(0))
    state.events = [{"type": "response.completed", "response": response()}]
    client = SimulatorModels(api_key="", sign_in=True)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == "http_unavailable"
        assert caught.value.evidence["auth_failure_code"] == failure
        assert len(requests) == 1 and not state.requests
        assert store.get("chatgpt_refresh_token") == "previous-refresh"
        assert (await complete(client))["value"] == {"ok": True}
        assert len(requests) == 2 and len(state.requests) == 1
    finally:
        await client.close()


async def test_rejected_persisted_login_requires_explicit_reauthentication(
    monkeypatch,
    isolated_access,
    subscription_boundary,
    http_boundary,
):
    store, state = isolated_access, subscription_boundary
    store.set("chatgpt_client_id", "synthetic-client")
    store.set("chatgpt_refresh_token", "rejected-refresh")
    monkeypatch.setattr(models.ChatGptSession, "access_token", state.real_access_token)
    requests, _ = http_boundary(lambda req: httpx.Response(400, json={"error": "invalid_grant"}))
    for _ in range(2):
        client = SimulatorModels(api_key="", sign_in=True)
        try:
            with pytest.raises(SimulatorModelError) as caught:
                await complete(client)
            assert caught.value.code == "reauthentication_required"
        finally:
            await client.close()
    assert state.connects == 0
    assert len(requests) == 1
    assert store.get("chatgpt_refresh_token") is None
    args = cli.parser().parse_args(["probe", "--subscription-only", "--reauthenticate"])
    client = cli.make_models(args)
    try:
        await client.prepare()
        assert state.connects == 1
        assert store.get("chatgpt_refresh_token") == "synthetic-refresh"
    finally:
        await client.close()


async def test_unavailable_keyring_fails_closed_even_with_api_key(monkeypatch):
    class UnavailableStore:
        def get(self, key):
            raise RuntimeError(_SECRET)

    monkeypatch.setattr(models, "KeyringSecretStore", lambda **kwargs: UnavailableStore())
    client = SimulatorModels(api_key=_SECRET, sign_in=True)
    try:
        for _ in range(2):
            with pytest.raises(SimulatorModelError) as caught:
                await client.prepare()
            assert caught.value.code == "keyring_unavailable"
            assert _SECRET not in json.dumps(caught.value.evidence)
    finally:
        await client.close()


async def test_late_keyring_read_failure_blocks_api_fallback(monkeypatch):
    class LateUnavailableStore(InMemorySecretStore):
        def get(self, key):
            if key == "chatgpt_client_id":
                raise RuntimeError(_SECRET)
            return None

    monkeypatch.setattr(models, "KeyringSecretStore", lambda **kwargs: LateUnavailableStore())
    client = SimulatorModels(api_key=_SECRET, sign_in=True)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == "keyring_unavailable"
        assert _SECRET not in json.dumps(caught.value.evidence)
    finally:
        await client.close()


@pytest.mark.parametrize("failed_key", ["chatgpt_client_id", "chatgpt_host_id", "chatgpt_account"])
async def test_account_owner_storage_failure_blocks_api_before_browser(monkeypatch, failed_key):
    class LateUnavailableStore(InMemorySecretStore):
        account_reads = 0

        def get(self, key):
            if key == "chatgpt_account":
                self.account_reads += 1
                if failed_key == key and self.account_reads > 1:
                    raise RuntimeError(_SECRET)
                return None
            if key == failed_key:
                raise RuntimeError(_SECRET)
            return {
                "chatgpt_client_id": "synthetic-client",
                "chatgpt_host_id": "synthetic-host",
            }.get(key)

    def forbidden_browser(url):
        pytest.fail("Storage failure must not launch a browser")

    real_owner = models.ChatGptAccountOwner
    monkeypatch.setattr(real_owner, "connect", _REAL_ACCOUNT_CONNECT)
    monkeypatch.setattr(
        models,
        "ChatGptAccountOwner",
        lambda **kwargs: real_owner(
            **kwargs,
            browser_opener=forbidden_browser,
            listener_factory=lambda **kwargs: SimpleNamespace(
                redirect_uri="http://127.0.0.1:1/callback",
                close=lambda: None,
            ),
        ),
    )
    monkeypatch.setattr(models, "KeyringSecretStore", lambda **kwargs: LateUnavailableStore())
    client = SimulatorModels(api_key=_SECRET, reauthenticate=True)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == "keyring_unavailable"
        assert _SECRET not in json.dumps(caught.value.evidence)
    finally:
        await client.close()


@pytest.mark.parametrize(
    "event,code",
    [
        (TimeoutError(), "timeout"),
        (OSError(), "network_error"),
        ({"type": "error", "status": 503}, "http_unavailable"),
        (models.ChatGptAuthError("refresh_network_error"), "network_error"),
    ],
)
async def test_transient_subscription_failure_allows_next_independent_call(
    subscription_boundary,
    event,
    code,
):
    state = subscription_boundary
    state.events = [event, {"type": "response.completed", "response": response()}]
    client = SimulatorModels(api_key="", sign_in=True)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == code
        assert len(state.requests) == 1
        assert len(state.sockets) == 1 and state.sockets[0].closed
        assert (await complete(client))["value"] == {"ok": True}
        assert len(state.requests) == 2
        assert len(state.sockets) == 2 and not state.sockets[1].closed
        assert client.attempts[-1]["subscription_wire"]["connection_action"] == "open"
        assert client.attempts[-1]["subscription_wire"]["connection_sequence"] == 2
    finally:
        await client.close()


async def test_stable_subscription_denial_stays_disabled_without_api_key(subscription_boundary):
    state = subscription_boundary
    state.events = [{"type": "error", "status": 403}]
    client = SimulatorModels(api_key="", sign_in=True)
    try:
        with pytest.raises(SimulatorModelError) as first:
            await complete(client)
        assert first.value.code == "access_refused"
        with pytest.raises(SimulatorModelError) as second:
            await complete(client)
        assert second.value.code == "access_unavailable"
        assert len(state.requests) == 1
    finally:
        await client.close()


async def test_cli_missing_selected_key_never_uses_ambient_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", _SECRET)
    monkeypatch.delenv("MISSING_SYNTHETIC_KEY", raising=False)
    args = cli.parser().parse_args(["probe", "--api-key-env", "MISSING_SYNTHETIC_KEY"])
    client = cli.make_models(args)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == "access_unavailable"
        assert _SECRET not in json.dumps(caught.value.evidence)
    finally:
        await client.close()


async def test_cli_subscription_only_signs_in_without_ambient_api_fallback(
    monkeypatch, subscription_boundary
):
    monkeypatch.setenv("OPENAI_API_KEY", _SECRET)
    state = subscription_boundary
    state.events = [
        {"type": "error", "error": {"code": "subscription_sharing_unsupported_capability"}},
        {"type": "error", "error": {"code": "subscription_sharing_unsupported_capability"}},
    ]
    client = cli.make_models(cli.parser().parse_args(["probe", "--subscription-only"]))
    try:
        report = await client.probe()
        assert state.connects == 1
        assert len(state.requests) == 2
        assert all(result["status"] == "failed" for result in report["roles"].values())
        assert all(
            result["code"] == "subscription_sharing_unsupported_capability"
            for result in report["roles"].values()
        )
        assert _SECRET not in json.dumps(report)
    finally:
        await client.close()


async def test_api_probe_completes_exact_model_effort_pairs_and_retains_usage(http_boundary):
    requests, clients = http_boundary(
        lambda req: httpx.Response(200, json=response(json.loads(req.content)["model"]))
    )
    client = SimulatorModels(api_key=_SECRET)
    try:
        report = await client.probe()
        assert {role: result["status"] for role, result in report["roles"].items()} == {
            "user": "completed",
            "judge": "completed",
        }
        for role, request in zip(("user", "judge"), requests):
            body = json.loads(request.content)
            result = report["roles"][role]
            assert (body["model"], body["reasoning"]["effort"]) == (
                ("gpt-6.1-terra", "medium") if role == "user" else ("gpt-6.1-sol", "xhigh")
            )
            assert str(request.url) == "https://api.openai.com/v1/responses"
            assert request.headers["Authorization"] == f"Bearer {_SECRET}"
            assert body["store"] is False and body["stream"] is False
            assert body["max_output_tokens"] == 25000
            assert body["text"]["format"]["strict"] is True
            assert body["text"]["format"]["schema"] == _SCHEMA
            timeout_s = 180 if role == "user" else 300
            assert request.extensions["timeout"] == {
                phase: timeout_s for phase in ("connect", "read", "write", "pool")
            }
            assert result["timeout_s"] == timeout_s
            assert result["transport_elapsed_s"] >= 0
            assert result["value"] == {"ok": True}
            assert result["usage"] == _USAGE
            assert result["returned_model"] == body["model"]
            assert result["cost_usd"] == pytest.approx(0.001335 * (1 if role == "user" else 2))
            assert result["cost_basis"]["kind"] == "standard_estimate_not_invoice"
        assert _SECRET not in json.dumps(report)
        snapshot = client.attempts
        snapshot.clear()
        assert len(client.attempts) == 3
    finally:
        await client.close()
    assert all(item.is_closed for item in clients)


async def test_environment_key_is_only_explicitly_named_environment_lookup(
    monkeypatch, http_boundary
):
    monkeypatch.setenv("OPENAI_API_KEY", _SECRET)
    requests, _ = http_boundary(lambda req: httpx.Response(200, json=response()))
    client = SimulatorModels()
    try:
        assert (await complete(client))["access_path"] == "api_key_http"
        assert requests[0].headers["Authorization"] == f"Bearer {_SECRET}"
    finally:
        await client.close()


async def test_subscription_success_precedes_api_and_requires_terminal_completion(
    subscription_boundary,
):
    state = subscription_boundary
    state.events = [
        {"type": "response.output_text.delta", "delta": "not the final JSON"},
        {"type": "response.completed", "response": response()},
    ]
    client = SimulatorModels(api_key=_SECRET, sign_in=True)
    try:
        result = await complete(client)
        assert result["value"] == {"ok": True}
        assert result["access_path"] == "subscription_ws"
        assert result["cost_usd"] is None
        assert result["cost_basis"] == {"kind": "subscription_billing_separate"}
        assert state.requests[0]["reasoning"] == {"effort": "medium"}
        assert state.requests[0]["text"]["format"]["schema"] == _SCHEMA
        assert state.requests[0]["store"] is False
        assert not state.sockets[0].closed
        assert state.connects == 1
        assert "private@example.invalid" not in json.dumps(client.access_evidence)
    finally:
        await client.close()
    assert state.sockets[0].closed
    assert state.stores[0].get("chatgpt_refresh_token") == "synthetic-refresh"


async def test_subscription_uses_role_specific_deadlines_and_safe_phase_timings(
    monkeypatch, subscription_boundary
):
    state = subscription_boundary
    now, deadlines = [100.0], []
    original_timeout = asyncio.timeout

    def timeout(delay):
        deadlines.append(delay)
        return original_timeout(delay)

    class AdmissionGate:
        async def wait(self):
            now[0] += 2.5

    def reply(request):
        now[0] += 3
        return [{"type": "response.completed", "response": response(request["model"])}]

    monkeypatch.setattr(models, "monotonic", lambda: now[0])
    monkeypatch.setattr(models.asyncio, "timeout", timeout)
    state.reply = reply
    client = SimulatorModels(api_key="", sign_in=True, subscription_gate=AdmissionGate())
    try:
        results = [await complete(client), await complete(client, "judge"), await complete(client)]
        assert deadlines == [180, 300, 180]
        assert [result["timeout_s"] for result in results] == deadlines
        assert [result["transport_elapsed_s"] for result in results] == [5.5, 5.5, 3]
        assert [
            result["subscription_wire"]["connection_admission_wait_s"] for result in results
        ] == [2.5, 2.5, 0]
        assert all(result["subscription_wire"]["provider_elapsed_s"] == 3 for result in results)
        assert [result["subscription_wire"]["connection_action"] for result in results] == [
            "open",
            "open",
            "reuse",
        ]
        assert [request["reasoning"]["effort"] for request in state.requests] == [
            "medium",
            "xhigh",
            "medium",
        ]
        assert _SECRET not in json.dumps(client.attempts)
    finally:
        await client.close()


@pytest.mark.parametrize("phase", ["admission", "provider"])
async def test_subscription_timeout_distinguishes_connection_admission_from_provider_without_replay(
    monkeypatch, subscription_boundary, phase
):
    state = subscription_boundary
    now = [100.0]

    class AdmissionGate:
        async def wait(self):
            now[0] += 2.5
            if phase == "admission":
                raise TimeoutError()

    def reply(request):
        now[0] += 3
        return [TimeoutError()]

    monkeypatch.setattr(models, "monotonic", lambda: now[0])
    state.reply = reply
    client = SimulatorModels(api_key="", sign_in=True, subscription_gate=AdmissionGate())
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == "timeout"
        evidence = caught.value.evidence
        assert evidence["timeout_s"] == 180
        assert evidence["transport_elapsed_s"] == (2.5 if phase == "admission" else 5.5)
        wire = evidence["subscription_wire"]
        assert wire["connection_admission_wait_s"] == 2.5
        assert wire["provider_elapsed_s"] == (None if phase == "admission" else 3)
        assert len(state.requests) == len(state.starts) == int(phase == "provider")
        assert sum(attempt.get("status") == "failed" for attempt in client.attempts) == 1
        if phase == "provider":
            assert state.sockets[0].closed
        assert _SECRET not in json.dumps(evidence)
    finally:
        await client.close()


async def test_reused_subscription_serializes_independent_inputs_and_outputs(subscription_boundary):
    state = subscription_boundary
    pending = asyncio.get_running_loop().create_future()

    def reply(request):
        visible = json.loads(request["input"][0]["content"][0]["text"])["visible"]
        if visible == "first":
            return [pending]
        terminal = response()
        terminal["output"][0]["content"][0]["text"] = '{"ok": false}'
        return [{"type": "response.completed", "response": terminal}]

    state.reply = reply
    client = SimulatorModels(api_key="", sign_in=True)
    try:
        async with asyncio.TaskGroup() as tasks:
            first = tasks.create_task(
                client.complete(
                    role="user",
                    instructions="First independent instructions",
                    payload={"visible": "first"},
                    schema=_SCHEMA,
                )
            )
            await state.receiving.wait()
            second = tasks.create_task(
                client.complete(
                    role="user",
                    instructions="Second independent instructions",
                    payload={"visible": "second"},
                    schema=_SCHEMA,
                )
            )
            await asyncio.sleep(0)
            assert len(state.requests) == 1
            pending.set_result({"type": "response.completed", "response": response()})
        assert first.result()["value"] == {"ok": True}
        assert second.result()["value"] == {"ok": False}
        assert len(state.sockets) == 1 and not state.sockets[0].closed
        for index, request in enumerate(state.requests):
            assert request["type"] == "response.create"
            assert request["model"] == "gpt-6.1-terra"
            assert request["reasoning"] == {"effort": "medium"}
            assert request["store"] is False
            assert request["previous_response_id"] is None
            assert request["max_output_tokens"] == 25000
            assert request["text"]["format"]["schema"] == _SCHEMA
            assert request["instructions"] == (
                "First independent instructions"
                if index == 0
                else "Second independent instructions"
            )
            assert request["input"] == [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps({"visible": "first" if index == 0 else "second"}),
                        }
                    ],
                }
            ]
        first_wire = first.result()["subscription_wire"]
        second_wire = second.result()["subscription_wire"]
        assert first_wire["connection_action"] == "open"
        assert first_wire["handshake_status"] == "opened"
        assert second_wire["connection_action"] == "reuse"
        assert second_wire["handshake_status"] == "not_attempted"
        assert first_wire["connection_sequence"] == second_wire["connection_sequence"] == 1
        assert second_wire["event_counts"] == {"response.completed": 1}
        assert "synthetic-subscription-token" not in json.dumps(client.attempts)
    finally:
        await client.close()
    assert state.sockets[0].closed


async def test_subscription_roles_keep_independent_socket_lifetimes(subscription_boundary):
    state = subscription_boundary
    pending = asyncio.get_running_loop().create_future()

    def reply(request):
        if request["model"] == "gpt-6.1-terra" and not pending.done():
            return [pending]
        terminal = response(request["model"])
        terminal["output"][0]["content"][0]["text"] = json.dumps(
            {"ok": request["model"] == "gpt-6.1-terra"}
        )
        return [{"type": "response.completed", "response": terminal}]

    state.reply = reply
    client = SimulatorModels(api_key="", sign_in=True, subscription_gate=SubscriptionConnectGate(0))
    try:
        async with asyncio.TaskGroup() as tasks:
            user = tasks.create_task(complete(client))
            await state.receiving.wait()
            judge = await complete(client, "judge")
            assert judge["value"] == {"ok": False}
            assert not user.done()
            pending.set_result({"type": "response.completed", "response": response()})
        assert user.result()["value"] == {"ok": True}
        user_again = await complete(client)
        judge_again = await complete(client, "judge")
        assert user_again["value"] == {"ok": True}
        assert judge_again["value"] == {"ok": False}
        assert len(state.sockets) == 2
        assert [request["model"] for request in state.sockets[0].requests] == [
            "gpt-6.1-terra",
            "gpt-6.1-terra",
        ]
        assert [request["model"] for request in state.sockets[1].requests] == [
            "gpt-6.1-sol",
            "gpt-6.1-sol",
        ]
        assert state.sockets[1].requests[0]["reasoning"] == {"effort": "xhigh"}
        assert user_again["subscription_wire"]["connection_sequence"] == 1
        assert judge_again["subscription_wire"]["connection_sequence"] == 2
        assert not any(socket.closed for socket in state.sockets)
    finally:
        await client.close()
    assert all(socket.closed for socket in state.sockets)


async def test_cancelled_subscription_drops_reused_socket_without_replay(subscription_boundary):
    state = subscription_boundary
    pending = asyncio.get_running_loop().create_future()
    terminal = response()
    terminal["output"][0]["content"][0]["text"] = '{"ok": false}'
    state.events = [
        {"type": "response.completed", "response": response()},
        pending,
        {"type": "response.completed", "response": terminal},
    ]
    client = SimulatorModels(api_key="", sign_in=True, subscription_gate=SubscriptionConnectGate(0))
    cancelled = None
    try:
        assert (await complete(client))["value"] == {"ok": True}
        state.receiving.clear()
        cancelled = asyncio.create_task(
            client.complete(
                role="user",
                instructions="Interrupted independent request",
                payload={"visible": "cancelled"},
                schema=_SCHEMA,
            )
        )
        await state.receiving.wait()
        assert len(state.requests) == 2 and len(state.sockets) == 1
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert pending.cancelled()
        assert state.sockets[0].closed
        assert len(state.requests) == 2 and len(state.sockets) == 1
        after = await client.complete(
            role="user",
            instructions="Next independent request",
            payload={"visible": "after"},
            schema=_SCHEMA,
        )
        assert after["value"] == {"ok": False}
        assert after["subscription_wire"]["connection_action"] == "open"
        assert after["subscription_wire"]["connection_sequence"] == 2
        assert len(state.sockets) == 2 and not state.sockets[1].closed
        assert [
            json.loads(request["input"][0]["content"][0]["text"])["visible"]
            for request in state.requests
        ] == ["test", "cancelled", "after"]
    finally:
        if cancelled is not None and not cancelled.done():
            cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)
        await client.close()


async def test_shared_subscription_gate_paces_new_connections_across_clients(
    monkeypatch, subscription_boundary
):
    state = subscription_boundary
    now, waits = [50.0], []
    real_sleep = asyncio.sleep

    async def sleep(delay):
        waits.append(delay)
        now[0] += delay
        await real_sleep(0)

    monkeypatch.setattr(models, "monotonic", lambda: now[0])
    monkeypatch.setattr(models.asyncio, "sleep", sleep)
    state.reply = lambda request: [
        {"type": "response.completed", "response": response(request["model"])}
    ]
    gate = SubscriptionConnectGate(min_interval_s=1.25)
    first = SimulatorModels(api_key="", sign_in=True, subscription_gate=gate)
    second = SimulatorModels(api_key="", sign_in=True, subscription_gate=gate)
    try:
        await asyncio.gather(
            complete(first), complete(second), complete(first, "judge"), complete(second, "judge")
        )
        assert state.starts == [50.0, 51.25, 52.5, 53.75]
        assert waits == [1.25, 1.25, 1.25]
        assert len(state.sockets) == 4
        results = await asyncio.gather(
            complete(second, "judge"), complete(first), complete(second), complete(first, "judge")
        )
        assert all(
            result["subscription_wire"]["connection_action"] == "reuse" for result in results
        )
        assert state.starts == [50.0, 51.25, 52.5, 53.75]
        assert waits == [1.25, 1.25, 1.25]
        assert all(len(socket.requests) == 2 for socket in state.sockets)
        await first.close()
        assert state.sockets[0].closed and state.sockets[2].closed
        assert not state.sockets[1].closed and not state.sockets[3].closed
    finally:
        await first.close()
        await second.close()
    assert all(socket.closed for socket in state.sockets)


async def test_subscription_send_error_closes_socket_without_replay(subscription_boundary):
    state = subscription_boundary

    def interrupted(request):
        raise OSError(_SECRET)

    state.reply = interrupted
    client = SimulatorModels(api_key="", sign_in=True, subscription_gate=SubscriptionConnectGate(0))
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == "network_error"
        assert len(state.requests) == 1 and len(state.sockets) == 1
        assert state.sockets[0].closed
        assert caught.value.evidence["subscription_wire"]["handshake_status"] == "opened"
        assert _SECRET not in json.dumps(caught.value.evidence)
        state.reply = lambda request: [{"type": "response.completed", "response": response()}]
        after = await complete(client)
        assert after["value"] == {"ok": True}
        assert after["subscription_wire"]["connection_sequence"] == 2
        assert len(state.requests) == 2 and len(state.sockets) == 2
    finally:
        await client.close()


@pytest.mark.parametrize(
    "body,content_kind,error_code,error_type",
    [
        (
            json.dumps(
                {
                    "error": {
                        "code": "subscription_sharing_usage_limit_exceeded",
                        "type": "permission_error",
                        "message": _SECRET,
                        "param": _SECRET,
                    },
                    "authorization": _SECRET,
                    "email": "private@example.invalid",
                }
            ).encode(),
            "json",
            "subscription_sharing_usage_limit_exceeded",
            "permission_error",
        ),
        (
            json.dumps({"error": {"code": _SECRET, "type": _SECRET, "message": _SECRET}}).encode(),
            "json",
            "unknown",
            "unknown",
        ),
        (f"<html><body>{_SECRET}</body></html>".encode(), "html", "unknown", "unknown"),
        (f"Private error: {_SECRET}".encode(), "text", "unknown", "unknown"),
        (b"\xff" + _SECRET.encode(), "binary", "unknown", "unknown"),
        (b"", "empty", "unknown", "unknown"),
        (b"x" * 65537, "unknown", "unknown", "unknown"),
    ],
    ids=["known-json", "unknown-json", "html", "text", "binary", "empty", "oversized"],
)
async def test_subscription_handshake_records_only_safe_allowlisted_diagnostics(
    subscription_boundary, body, content_kind, error_code, error_type
):
    state = subscription_boundary
    state.open_errors = [
        InvalidStatus(
            Response(
                403,
                _SECRET,
                Headers(
                    [
                        ("x-request-id", "req_synthetic-123"),
                        ("retry-after", "7.5"),
                        ("set-cookie", f"session={_SECRET}"),
                        ("authorization", f"Bearer {_SECRET}"),
                        ("x-private", _SECRET),
                        ("content-type", f"application/json; private={_SECRET}"),
                    ]
                ),
                body,
            )
        )
    ]
    client = SimulatorModels(api_key="", sign_in=True)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == "access_refused"
        assert caught.value.evidence["http_status"] == 403
        wire = caught.value.evidence["subscription_wire"]
        assert wire["connection_action"] == "open"
        assert wire["connection_sequence"] == 1
        assert wire["handshake_status"] == "rejected"
        assert wire["handshake_diagnostics"] == {
            "response_content_kind": content_kind,
            "error_code": error_code,
            "error_type": error_type,
            "request_id": "req_synthetic-123",
            "retry_after_s": 7.5,
        }
        assert wire["event_counts"] == {}
        assert not state.requests and len(state.sockets) == 1
        published = (
            str(caught.value) + json.dumps(caught.value.evidence) + json.dumps(client.attempts)
        )
        for sensitive in (_SECRET, "private@example.invalid", "synthetic-subscription-token"):
            assert sensitive not in published
        with pytest.raises(SimulatorModelError) as second:
            await complete(client)
        assert second.value.code == "access_unavailable"
        assert not state.requests and len(state.sockets) == 1
    finally:
        await client.close()


@pytest.mark.parametrize(
    "request_ids,retry_after",
    [
        ([f"Bearer {_SECRET}"], _SECRET),
        (["r" * 129], "Wed, 21 Oct 2015 07:28:00 GMT"),
        (["req_first", "req_second"], "86401"),
        (["req/unsafe"], "-1"),
    ],
)
async def test_subscription_handshake_omits_unbounded_or_ambiguous_headers(
    subscription_boundary, request_ids, retry_after
):
    state = subscription_boundary
    state.open_errors = [
        InvalidStatus(
            Response(
                403,
                "Forbidden",
                Headers(
                    [("x-request-id", value) for value in request_ids]
                    + [("retry-after", retry_after), ("cookie", _SECRET)]
                ),
                b"{}",
            )
        )
    ]
    client = SimulatorModels(api_key="", sign_in=True)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.evidence["subscription_wire"]["handshake_diagnostics"] == {
            "response_content_kind": "json",
            "error_code": "unknown",
            "error_type": "unknown",
        }
        assert _SECRET not in json.dumps(caught.value.evidence)
        assert not state.requests
    finally:
        await client.close()


async def test_close_attempts_both_role_sockets_when_one_close_fails(
    monkeypatch, subscription_boundary
):
    state = subscription_boundary
    state.reply = lambda request: [
        {"type": "response.completed", "response": response(request["model"])}
    ]
    client = SimulatorModels(api_key="", sign_in=True, subscription_gate=SubscriptionConnectGate(0))
    try:
        await complete(client)
        await complete(client, "judge")

        async def failing_close():
            state.sockets[0].closed = True
            raise OSError(_SECRET)

        monkeypatch.setattr(state.sockets[0], "close", failing_close)
        with pytest.raises(SimulatorModelError) as caught:
            await client.close()
        assert caught.value.code == "close_failed"
        assert all(socket.closed for socket in state.sockets)
        assert _SECRET not in json.dumps(caught.value.evidence)
        assert state.stores[0].get("chatgpt_refresh_token") == "synthetic-refresh"
        with pytest.raises(SimulatorModelError) as closed:
            await complete(client)
        assert closed.value.code == "closed"
    finally:
        await client.close()


async def test_subscription_probe_reads_finalized_items_when_live_terminal_output_is_empty(
    subscription_boundary,
):
    state = subscription_boundary
    for model in ("gpt-6.1-terra", "gpt-6.1-sol"):
        state.events.extend(
            [
                {"type": "response.created"},
                {"type": "response.output_text.delta", "delta": '{"ok":'},
                {"type": "response.output_text.delta", "delta": " true}"},
                {"type": "response.output_text.done", "text": '{"ok": true}'},
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": response(model)["output"][0],
                },
                {"type": "response.completed", "response": response(model, output=[])},
            ]
        )
    client = SimulatorModels(api_key="", sign_in=True)
    try:
        report = await client.probe()
        for result in report["roles"].values():
            assert result["status"] == "completed"
            assert result["value"] == {"ok": True}
            assert result["usage"] == _USAGE
            assert result["cost_usd"] is None
        assert len(state.requests) == 2
        assert len(state.sockets) == 2 and not any(socket.closed for socket in state.sockets)
    finally:
        await client.close()
    assert all(socket.closed for socket in state.sockets)


async def test_subscription_deltas_without_finalized_items_remain_failed_with_usage(
    subscription_boundary,
):
    subscription_boundary.events = [
        {"type": "response.output_text.delta", "delta": '{"ok": true}'},
        {"type": "response.completed", "response": response(output=[])},
    ]
    client = SimulatorModels(api_key=_SECRET, sign_in=True)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == "malformed_result"
        assert caught.value.evidence["usage"] == _USAGE
        assert (
            caught.value.evidence["subscription_wire"]["event_counts"]["response.output_text.delta"]
            == 1
        )
        assert len(subscription_boundary.requests) == 1
    finally:
        await client.close()


async def test_malformed_subscription_probe_fails_both_roles_without_leaking_output(
    subscription_boundary,
):
    state = subscription_boundary
    for model in ("gpt-6.1-terra", "gpt-6.1-sol"):
        terminal = response(model)
        terminal["output"][0]["id"] = _SECRET
        terminal["output"][0]["content"][0]["text"] = '{"private":"' + _SECRET + '"'
        state.events.extend(
            [
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": response(model)["output"][0],
                },
                {"type": "response.completed", "response": terminal},
            ]
        )
    client = SimulatorModels(api_key=_SECRET, sign_in=True)
    try:
        report = await client.probe()
        for role in ("user", "judge"):
            result = report["roles"][role]
            assert result["status"] == "failed"
            assert result["code"] == "malformed_result"
            assert result["evidence"]["usage"] == _USAGE
            assert result["evidence"]["cost_usd"] is None
        assert len(state.requests) == 2
        assert all(socket.closed for socket in state.sockets)
        assert _SECRET not in json.dumps(report)
    finally:
        await client.close()


@pytest.mark.parametrize(
    "event",
    [
        {
            "type": "error",
            "error": {"code": "subscription_sharing_unsupported_capability", "message": _SECRET},
        },
        {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "usage": _USAGE,
                "error": {"code": "subscription_sharing_unsupported_capability"},
            },
        },
        {"type": "error", "status": 401, "error": {"message": _SECRET}},
    ],
)
async def test_refused_access_falls_back_once_to_same_model(
    subscription_boundary, http_boundary, event
):
    state = subscription_boundary
    state.events = [event]
    requests, _ = http_boundary(lambda req: httpx.Response(200, json=response()))
    client = SimulatorModels(api_key=_SECRET, sign_in=True)
    try:
        first = await complete(client)
        second = await complete(client)
        assert first["access_path"] == second["access_path"] == "api_key_http"
        assert len(state.requests) == 1 and len(requests) == 2
        assert {json.loads(req.content)["model"] for req in requests} == {"gpt-6.1-terra"}
        refusal = client.attempts[1]
        assert refusal["code"] == (
            "authentication_failed"
            if event.get("status") == 401
            else "subscription_sharing_unsupported_capability"
        )
        if event["type"] == "response.failed":
            assert refusal["usage"] == _USAGE
        assert _SECRET not in json.dumps(client.attempts)
    finally:
        await client.close()


async def test_explicit_sign_in_failure_records_safe_code_then_api(
    subscription_boundary, http_boundary
):
    state = subscription_boundary
    state.succeeded, state.code = False, "plan_scope_missing"
    http_boundary(lambda req: httpx.Response(200, json=response()))
    client = SimulatorModels(api_key=_SECRET, sign_in=True)
    try:
        assert (await complete(client))["access_path"] == "api_key_http"
        assert client.attempts[0]["code"] == "plan_scope_missing"
        assert not state.requests
    finally:
        await client.close()


@pytest.mark.parametrize(
    "override,code",
    [
        ({"status": "incomplete"}, "response_incomplete"),
        ({"status": "failed", "error": {"message": _SECRET}}, "response_not_completed"),
        ({"model": "gpt-6.1-luna"}, "model_mismatch"),
        ({"model": None}, "model_mismatch"),
        (
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "refusal", "refusal": _SECRET}],
                    }
                ]
            },
            "response_refusal",
        ),
        (
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "NaN"}],
                    }
                ]
            },
            "malformed_result",
        ),
        ({"output": []}, "malformed_result"),
    ],
)
async def test_api_rejects_nonresults_without_retry_and_preserves_usage(
    http_boundary, override, code
):
    requests, _ = http_boundary(lambda req: httpx.Response(200, json=response(**override)))
    client = SimulatorModels(api_key=_SECRET)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == code
        assert caught.value.evidence["usage"] == _USAGE
        assert caught.value.evidence["cost_usd"] == pytest.approx(0.001335)
        assert len(requests) == 1
        assert _SECRET not in str(caught.value) + json.dumps(caught.value.evidence)
    finally:
        await client.close()


@pytest.mark.parametrize(
    "kind,status,code",
    [
        ("response.incomplete", "incomplete", "response_incomplete"),
        ("response.failed", "completed", "malformed_response"),
        ("response.completed", "completed", "response_refusal"),
    ],
)
async def test_content_or_incomplete_failure_never_triggers_api(
    subscription_boundary, kind, status, code
):
    subscription_boundary.events = [
        {
            "type": kind,
            "response": response(
                status=status,
                output=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "refusal", "refusal": _SECRET}],
                    }
                ],
            ),
        }
    ]
    client = SimulatorModels(api_key=_SECRET, sign_in=True)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == code
        assert caught.value.evidence["usage"] == _USAGE
        assert len(subscription_boundary.requests) == 1
    finally:
        await client.close()


@pytest.mark.parametrize(
    "usage,expected",
    [
        (_USAGE, 0.001335),
        ({"input_tokens": 272000, "output_tokens": 100}, 0.2725),
        ({"input_tokens": 272001, "output_tokens": 100}, 0.544752),
        (
            {
                "input_tokens": 300000,
                "output_tokens": 100,
                "input_tokens_details": {"cached_tokens": 100000, "cache_write_tokens": 50000},
            },
            0.43575,
        ),
        (None, None),
        ({}, None),
        ({"input_tokens": True, "output_tokens": 2}, None),
        (
            {"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 11}},
            None,
        ),
        (
            {
                "input_tokens": 10,
                "output_tokens": 2,
                "input_tokens_details": {"cached_tokens": "secret"},
            },
            None,
        ),
        ({"input_tokens": 10, "output_tokens": 2, "total_tokens": 11}, None),
        (
            {
                "input_tokens": 10,
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 3},
            },
            None,
        ),
    ],
)
async def test_valid_disjoint_usage_cost_and_unknown_invalid_usage(http_boundary, usage, expected):
    http_boundary(lambda req: httpx.Response(200, json=response(usage=usage)))
    client = SimulatorModels(api_key=_SECRET)
    try:
        result = await complete(client)
        if expected is None:
            assert result["cost_usd"] is None
            assert result["cost_basis"]["usage_status"] == "missing_or_invalid"
        else:
            assert result["cost_usd"] == pytest.approx(expected)
            assert result["cost_basis"]["usage_status"] == "valid"
    finally:
        await client.close()


@pytest.mark.parametrize(
    "status,body,code",
    [
        (401, {"error": {"code": "invalid_api_key", "message": _SECRET}}, "invalid_api_key"),
        (403, {"error": {"code": _SECRET, "message": _SECRET}}, "access_refused"),
        (200, ["not a response"], "malformed_response"),
    ],
)
async def test_http_errors_are_safe_and_never_retried(http_boundary, status, body, code):
    requests, _ = http_boundary(lambda req: httpx.Response(status, json=body))
    client = SimulatorModels(api_key=_SECRET)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == code
        assert _SECRET not in str(caught.value) + json.dumps(caught.value.evidence)
        assert len(requests) == 1
    finally:
        await client.close()


async def test_http_failed_response_retains_usage_and_estimated_charge(http_boundary):
    http_boundary(
        lambda req: httpx.Response(
            429,
            json=response(
                status="failed",
                error={"code": "rate_limit_exceeded", "message": _SECRET},
            ),
        )
    )
    client = SimulatorModels(api_key=_SECRET)
    try:
        with pytest.raises(SimulatorModelError) as caught:
            await complete(client)
        assert caught.value.code == "rate_limit_exceeded"
        assert caught.value.evidence["usage"] == _USAGE
        assert caught.value.evidence["cost_usd"] == pytest.approx(0.001335)
        assert _SECRET not in json.dumps(caught.value.evidence)
    finally:
        await client.close()


async def test_invalid_usage_remains_observable_without_exposing_strings(http_boundary):
    usage = {
        "input_tokens": -1,
        "output_tokens": 2,
        "input_tokens_details": {
            "cached_tokens": _SECRET,
        },
    }
    http_boundary(lambda req: httpx.Response(200, json=response(usage=usage)))
    client = SimulatorModels(api_key=_SECRET)
    try:
        result = await complete(client)
        assert result["usage"] == {
            "input_tokens": -1,
            "output_tokens": 2,
            "input_tokens_details": {"cached_tokens": "[redacted]"},
        }
        assert result["cost_usd"] is None
        assert _SECRET not in json.dumps(result)
    finally:
        await client.close()


@pytest.mark.parametrize(
    "http_status,terminal_status",
    [
        (200, "completed"),
        (200, "incomplete"),
        (429, "failed"),
    ],
)
async def test_negative_cached_usage_is_never_laundered_into_estimate(
    http_boundary,
    http_status,
    terminal_status,
):
    usage = {
        "input_tokens": 10,
        "output_tokens": 2,
        "input_tokens_details": {"cached_tokens": -1},
    }
    http_boundary(
        lambda req: httpx.Response(
            http_status,
            json=response(status=terminal_status, usage=usage),
        )
    )
    client = SimulatorModels(api_key=_SECRET)
    try:
        if terminal_status == "completed":
            evidence = await complete(client)
        else:
            with pytest.raises(SimulatorModelError) as caught:
                await complete(client)
            evidence = caught.value.evidence
        assert evidence["usage"] == usage
        assert evidence["cost_usd"] is None
        assert evidence["cost_basis"]["usage_status"] == "missing_or_invalid"
        assert client.attempts[-1]["usage"] == usage
        assert client.attempts[-1]["cost_usd"] is None
    finally:
        await client.close()


async def test_standalone_probe_through_shared_network_and_real_loopback_http(monkeypatch):
    seen = []
    connections = []

    async def serve(reader, writer):
        connections.append(writer)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            body = json.loads(await reader.readexactly(length))
            seen.append((headers, body))
            encoded = json.dumps(response(body["model"])).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                + str(len(encoded)).encode()
                + b"\r\n\r\n"
                + encoded
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    real_dispatch = httpx.AsyncHTTPTransport.handle_async_request

    async def local_dispatch(transport, request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        request.url = httpx.URL(f"http://127.0.0.1:{port}/v1/responses")
        return await real_dispatch(transport, request)

    def local_client(**kwargs):
        return _SHARED_FACTORY(policy=ProxyPolicy(environment={}, system={}), **kwargs)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", local_dispatch)
    monkeypatch.setattr(network_clients, "external_async_client", local_client)
    client = SimulatorModels(api_key="synthetic-loopback-only")
    try:
        report = await client.probe()
        assert [report["roles"][role]["status"] for role in ("user", "judge")] == [
            "completed",
            "completed",
        ]
        assert [(body["model"], body["reasoning"]["effort"]) for _, body in seen] == [
            ("gpt-6.1-terra", "medium"),
            ("gpt-6.1-sol", "xhigh"),
        ]
        assert all(headers.startswith(b"POST /v1/responses HTTP/1.1") for headers, _ in seen)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()
        for writer in connections:
            writer.close()
            await writer.wait_closed()


_SHARED_FACTORY = network_clients.external_async_client
