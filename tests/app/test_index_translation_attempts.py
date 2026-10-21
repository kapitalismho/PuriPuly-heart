from __future__ import annotations

import asyncio
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.app.wiring.wiring_llm_factory import create_llm_provider_from_resolved_config
from puripuly_heart.config.runtime_resolution import (
    RuntimeResolutionInput,
    TranslationRuntimeIntent,
    resolve_llm_config,
)
from puripuly_heart.core.llm.fallback_racing import FallbackRacingLLMProvider, LLMProviderRaceError
from puripuly_heart.core.llm.index_translate import (
    IndexRequestLimitError,
    IndexServiceUnavailableError,
    is_index_service_failure,
)
from puripuly_heart.providers.llm import index_translate


class ForbiddenSecrets:
    def get(self, key):
        raise AssertionError("Public Index requests must not read credentials")


class ManualDelay:
    def __init__(self):
        self.now = 0.0
        self.scheduled = asyncio.Event()
        self.waiters = []

    async def sleep(self, delay):
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((self.now + delay, future))
        self.scheduled.set()
        await future

    async def advance_to(self, now):
        self.now = now
        for deadline, future in self.waiters:
            if deadline <= now and not future.done():
                future.set_result(None)
        for _ in range(20):
            await asyncio.sleep(0)


@pytest.fixture
def build_provider(monkeypatch):
    original = index_translate.network_clients.external_async_client
    monkeypatch.setattr(index_translate, "select_tls", lambda: object())
    monkeypatch.setattr(index_translate, "ProxyPolicy", lambda: object())
    clients = []

    def build(handler):
        def external(**kwargs):
            client = original(**kwargs, transport=httpx.MockTransport(handler))
            clients.append(client)
            return client

        monkeypatch.setattr(index_translate.network_clients, "external_async_client", external)
        config = resolve_llm_config(
            RuntimeResolutionInput(
                translation=TranslationRuntimeIntent(
                    model="index_translate_35b", connection="public_api"
                )
            )
        )
        assert len(config.attempts) == 2
        provider = create_llm_provider_from_resolved_config(config, secrets=ForbiddenSecrets())
        race = provider.inner
        assert isinstance(race, FallbackRacingLLMProvider)
        delay = ManualDelay()
        race.sleeper = delay.sleep
        race.clock = lambda: delay.now
        return provider, delay, clients

    return build


def _kwargs():
    return {
        "utterance_id": uuid4(),
        "text": "Synthetic hello.",
        "system_prompt": "Unchanged rendered prompt",
        "source_language": "en",
        "target_language": "es",
    }


def _success(text="Hola."):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ]
        },
    )


@pytest.mark.parametrize("duplicate_succeeds", [True, False])
async def test_early_primary_error_waits_for_1300ms_and_never_adds_third_attempt(
    build_provider, duplicate_succeeds
):
    sent = []
    primary_finished = asyncio.Event()
    duplicate_sent = asyncio.Event()

    async def handler(request):
        sent.append(delay.now)
        if len(sent) == 1:
            primary_finished.set()
            return httpx.Response(503)
        duplicate_sent.set()
        return _success() if duplicate_succeeds else httpx.Response(503)

    provider, delay, clients = build_provider(handler)
    task = asyncio.create_task(provider.translate(**_kwargs()))
    try:
        await asyncio.wait_for(primary_finished.wait(), 1)
        await asyncio.wait_for(delay.scheduled.wait(), 1)
        await delay.advance_to(1.299)
        assert sent == [0.0]
        assert not task.done()
        await delay.advance_to(1.3)
        await asyncio.wait_for(duplicate_sent.wait(), 1)
        if duplicate_succeeds:
            assert (await asyncio.wait_for(task, 1)).text == "Hola."
        else:
            with pytest.raises(LLMProviderRaceError) as failure:
                await asyncio.wait_for(task, 1)
            assert len(failure.value.errors) == 2
            assert is_index_service_failure(failure.value)
        await delay.advance_to(100.0)
        assert sent == [0.0, 1.3]
    finally:
        await provider.close()
    assert all(client.is_closed for client in clients)


async def test_primary_success_prevents_delayed_duplicate(build_provider):
    sent = []

    async def handler(request):
        sent.append(delay.now)
        return _success()

    provider, delay, clients = build_provider(handler)
    try:
        assert (await provider.translate(**_kwargs())).text == "Hola."
        await delay.advance_to(100)
        assert sent == [0.0]
    finally:
        await provider.close()
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("winning_branch", [0, 1])
async def test_one_invalid_branch_cannot_defeat_the_other_valid_winner(
    build_provider, winning_branch
):
    started = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    count = 0

    async def handler(request):
        nonlocal count
        branch = count
        count += 1
        started[branch].set()
        await release[branch].wait()
        return _success("Valid winner") if branch == winning_branch else httpx.Response(503)

    provider, delay, clients = build_provider(handler)
    task = asyncio.create_task(provider.translate(**_kwargs()))
    try:
        await asyncio.wait_for(started[0].wait(), 1)
        await asyncio.wait_for(delay.scheduled.wait(), 1)
        await delay.advance_to(1.3)
        await asyncio.wait_for(started[1].wait(), 1)
        release[1 - winning_branch].set()
        await delay.advance_to(1.4)
        assert not task.done()
        release[winning_branch].set()
        assert (await asyncio.wait_for(task, 1)).text == "Valid winner"
        assert count == 2
    finally:
        await provider.close()
    assert all(client.is_closed for client in clients)


async def test_duplicate_winner_cancels_owned_primary_and_client_cleanup(build_provider):
    primary_started = asyncio.Event()
    primary_cancelled = asyncio.Event()
    count = 0

    async def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            primary_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                primary_cancelled.set()
                raise
        return _success()

    provider, delay, clients = build_provider(handler)
    task = asyncio.create_task(provider.translate(**_kwargs()))
    try:
        await asyncio.wait_for(primary_started.wait(), 1)
        await asyncio.wait_for(delay.scheduled.wait(), 1)
        await delay.advance_to(1.3)
        assert (await asyncio.wait_for(task, 1)).text == "Hola."
    finally:
        await provider.close()
    assert primary_cancelled.is_set()
    assert count == 2
    assert all(client.is_closed for client in clients)


async def test_cancellation_before_delay_never_sends_duplicate(build_provider):
    started = asyncio.Event()
    count = 0

    async def handler(request):
        nonlocal count
        count += 1
        started.set()
        await asyncio.Event().wait()

    provider, delay, clients = build_provider(handler)
    task = asyncio.create_task(provider.translate(**_kwargs()))
    try:
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await delay.advance_to(100)
        assert count == 1
    finally:
        await provider.close()
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize(
    "errors",
    [
        (),
        (IndexRequestLimitError("output_limit"),),
        (IndexServiceUnavailableError("timeout"), IndexRequestLimitError("input_limit")),
        (asyncio.CancelledError(),),
        (RuntimeError("unknown failure"),),
    ],
)
def test_only_all_service_failures_establish_unavailability(errors):
    assert not is_index_service_failure(LLMProviderRaceError(errors))
    assert is_index_service_failure(
        LLMProviderRaceError(
            (IndexServiceUnavailableError("timeout"), IndexServiceUnavailableError("transport"))
        )
    )
