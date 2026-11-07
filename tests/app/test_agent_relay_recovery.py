from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from puripuly_heart.app.adapters.agent_relay_auth import AgentRelayTokenOwner
from puripuly_heart.app.adapters.agent_relay_transport import AgentRelayTransport
from puripuly_heart.app.ports.agent_chat import AgentServiceError, AgentTransportError
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.core.managed_identity import ManagedIdentityBundle, encode_base64url
from tests.app.test_agent_chat import Records, Responses, answer
from tests.app.test_agent_chat_lifecycle import notices_done, watch


def token_response(token: str = "synthetic-token") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "expires_at": (datetime.now(UTC) + timedelta(seconds=3600)).isoformat(),
        },
    )


def token_owner(handler, *, clock=lambda: 0.0) -> AgentRelayTokenOwner:
    key = Ed25519PrivateKey.generate()
    bundle = ManagedIdentityBundle(
        "01961ad7-a7c1-7000-8000-0123456789ab",
        encode_base64url(key.public_key().public_bytes_raw()),
        key,
    )

    async def identity():
        return bundle

    owner = AgentRelayTokenOwner(
        lambda: "https://broker.example", identity, app_version="synthetic", clock=clock
    )
    owner._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return owner


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [503, "timeout"])
async def test_transient_authentication_retries_once_with_fresh_proof(failure):
    proofs = []

    def issue(request):
        proofs.append(json.loads(request.content))
        if len(proofs) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("synthetic", request=request)
            return httpx.Response(failure)
        return token_response()

    owner = token_owner(issue)
    try:
        assert await owner.get_token() == "synthetic-token"
        assert len(proofs) == 2
        assert proofs[0]["nonce"] != proofs[1]["nonce"]
        assert await owner.get_token() == "synthetic-token"
        assert len(proofs) == 2
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected", "attempts"),
    [
        (401, "service_authentication_failed", 1),
        (403, "service_authentication_failed", 1),
        (503, "service_authentication_unavailable", 2),
    ],
)
async def test_authentication_rejection_and_unavailability_are_distinct(status, expected, attempts):
    requests = []

    def issue(request):
        requests.append(request)
        return httpx.Response(status)

    owner = token_owner(issue)
    try:
        with pytest.raises(AgentServiceError, match=expected):
            await owner.get_token()
        assert len(requests) == attempts
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_authentication_rate_limit_blocks_new_requests_until_server_cooldown():
    now = [0.0]
    requests = []

    def issue(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"retry_after_ms": 30_000}})
        return token_response()

    owner = token_owner(issue, clock=lambda: now[0])
    try:
        for instant in (0.0, 1.0, 29.9):
            now[0] = instant
            with pytest.raises(AgentServiceError, match="service_rate_limited"):
                await owner.get_token()
        assert len(requests) == 1
        now[0] = 30.0
        assert await owner.get_token() == "synthetic-token"
        assert len(requests) == 2
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_concurrent_401s_do_not_invalidate_a_newer_token_or_duplicate_refresh():
    issued = []
    old_requests = []
    both_old = asyncio.Event()
    refreshed = asyncio.Event()

    def issue(request):
        issued.append(request)
        if len(issued) == 2:
            refreshed.set()
        return token_response(f"synthetic-{len(issued)}")

    async def respond(request):
        if request.headers["authorization"] == "Bearer synthetic-1":
            old_requests.append(request)
            order = len(old_requests)
            if order == 2:
                both_old.set()
            await both_old.wait()
            if order == 2:
                await refreshed.wait()
                await asyncio.sleep(0)
            return httpx.Response(401)
        return httpx.Response(200, json={"answers": []})

    owner = token_owner(issue)
    relay = AgentRelayTransport("https://relay.example/v1/respond", owner)
    await relay._client.aclose()
    relay._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        await relay.prepare()
        results = await asyncio.wait_for(
            asyncio.gather(
                relay.classify(input=[], turn_id="a" * 32),
                relay.classify(input=[], turn_id="b" * 32),
            ),
            timeout=5,
        )
        assert results == [{"answers": []}, {"answers": []}]
        assert len(issued) == 2
        assert len(old_requests) == 2
        assert await owner.get_token() == "synthetic-2"
    finally:
        await relay.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["classify", "respond", "support"])
async def test_relay_401_is_retried_only_once_and_other_errors_are_not_replayed(operation):
    issued, requests = [], []
    status = [401]

    def issue(request):
        issued.append(request)
        return token_response(f"synthetic-{len(issued)}")

    def reject(request):
        requests.append(request)
        return httpx.Response(status[0], json={"error": {"code": "service_authentication_failed"}})

    owner = token_owner(issue)
    relay = AgentRelayTransport("https://relay.example/v1/respond", owner)
    await relay._client.aclose()
    relay._client = httpx.AsyncClient(transport=httpx.MockTransport(reject))

    async def invoke():
        if operation == "classify":
            await relay.classify(input=[], turn_id="a" * 32)
        elif operation == "respond":
            async for _ in relay.respond(
                input=[], turn_id="a" * 32, effort="low", route="social", allow_tools=False
            ):
                raise AssertionError("Rejected response must not emit events")
        else:
            await relay.upload_support(
                bundle=b"synthetic",
                ticket="PP-0123456789ABCDEF",
                app_version="synthetic",
                diagnostic_codes=(),
            )

    try:
        with pytest.raises(AgentTransportError):
            await invoke()
        assert len(requests) == len(issued) == 2
        status[0] = 503
        with pytest.raises(AgentTransportError):
            await invoke()
        assert len(requests) == 3
        assert len(issued) == 2
    finally:
        await relay.close()


@pytest.mark.asyncio
async def test_oversize_request_is_rejected_before_authentication_or_relay_request():
    requests = []

    def issue(request):
        requests.append(request)
        return token_response()

    relay = AgentRelayTransport("https://relay.example/v1/respond", token_owner(issue))
    try:
        with pytest.raises(AgentServiceError, match="request_too_large"):
            await relay.classify(
                input=[{"role": "user", "content": "中" * 30_000}], turn_id="a" * 32
            )
        with pytest.raises(AgentServiceError, match="request_too_large"):
            await anext(
                relay.respond(
                    input=[{"role": "user", "content": "中" * 400_000}],
                    turn_id="a" * 32,
                    effort="low",
                    route="social",
                    allow_tools=False,
                )
            )
        assert requests == []
    finally:
        await relay.close()


@pytest.mark.asyncio
async def test_prompt_budget_includes_classification_and_notices_without_extra_worker_requests():
    transport = Responses(*(answer("Synthetic observation.") for _ in range(15)))
    owner = AgentChatOwner(Records(), transport, localize=str)
    owner.bind_verification_providers(frozenset({"openai"}))
    await owner.start()
    try:
        assert (await owner.send("Inspect synthetic state.")).status == "completed"
        turn_id = transport.requests[0]["turn_id"]
        for _ in range(13):
            await watch(owner)
            owner.key_verification_completed("openai", True)
            await notices_done(owner)
        assert len(transport.classifications) == 1
        assert len(transport.requests) == 14
        assert {request["turn_id"] for request in transport.requests} == {turn_id}
        assert transport.requests[-1]["allow_tools"] is False
        await watch(owner)
        owner.key_verification_completed("openai", True)
        await notices_done(owner)
        assert len(transport.requests) == 14
        assert owner.snapshot.messages[-1].text == "agent_chat.service_turn_exhausted"
        assert (await owner.send("Continue with a new request.")).status == "completed"
        assert transport.requests[-1]["turn_id"] != turn_id
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_rejected_token_retry_shares_the_turn_http_budget_but_not_support_quota():
    issued = []
    requests = []

    def issue(request):
        issued.append(request)
        return token_response(f"token-{len(issued)}")

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(401)
        if request.url.path == "/v1/classify":
            return httpx.Response(200, json={"answers": []})
        if request.url.path == "/v1/support":
            return httpx.Response(200, json={"ticket": "PP-0123456789ABCDEF"})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: " + json.dumps({"type": "completed", "response": answer()}) + "\n\n",
        )

    relay = AgentRelayTransport("https://relay.example/v1/respond", token_owner(issue))
    await relay._client.aclose()
    relay._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    request = {
        "input": [{"role": "user", "content": "Synthetic request."}],
        "turn_id": "a" * 32,
        "effort": "low",
        "route": "social",
        "allow_tools": False,
    }
    try:
        await relay.classify(input=request["input"], turn_id=request["turn_id"])
        for _ in range(13):
            assert [event["type"] async for event in relay.respond(**request)] == ["completed"]
        assert len(requests) == 15 and len(issued) == 2
        with pytest.raises(AgentServiceError, match="service_turn_exhausted"):
            await anext(relay.respond(**request))
        assert len(requests) == 15 and len(issued) == 2
        await relay.classify(input=request["input"], turn_id="b" * 32)
        assert len(requests) == 16
        receipt = await relay.upload_support(
            bundle=b"synthetic-gzip",
            ticket="PP-0123456789ABCDEF",
            app_version="synthetic",
            diagnostic_codes=[],
        )
        assert receipt == {"ticket": "PP-0123456789ABCDEF"}
        assert len(requests) == 17 and len(issued) == 2
    finally:
        await relay.close()
