from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.app.wiring.wiring_llm_factory import create_llm_provider_from_resolved_config
from puripuly_heart.config.runtime_resolution import (
    RuntimeResolutionInput,
    TranslationRuntimeIntent,
    resolve_llm_config,
)
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.llm.fallback_racing import FallbackRacingLLMProvider
from puripuly_heart.core.llm.index_translate import is_index_service_failure
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.vad.gating import SpeechEnd
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import Transcript
from puripuly_heart.providers.llm import index_translate
from tests.app.test_index_translation_attempts import ForbiddenSecrets, ManualDelay
from tests.helpers.fakes import FakeSender
from tests.helpers.translation_owners import compose_translation_test_harness


def success(text="Valid translation"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {"finish_reason": "stop", "message": {"role": "assistant", "content": text}}
            ]
        },
    )


@pytest.fixture
def build_owner(monkeypatch):
    original = index_translate.network_clients.external_async_client
    monkeypatch.setattr(index_translate, "select_tls", lambda: object())
    monkeypatch.setattr(index_translate, "ProxyPolicy", lambda: object())
    clients = []
    config = resolve_llm_config(
        RuntimeResolutionInput(
            translation=TranslationRuntimeIntent(
                model="index_translate_35b", connection="public_api"
            )
        )
    )
    assert [attempt.start_after_ms for attempt in config.attempts] == [0, 1300]

    def build(handler, **configuration):
        def external(**kwargs):
            client = original(**kwargs, transport=httpx.MockTransport(handler))
            clients.append(client)
            return client

        monkeypatch.setattr(index_translate.network_clients, "external_async_client", external)
        delay = ManualDelay()

        def provider_factory():
            provider = create_llm_provider_from_resolved_config(config, secrets=ForbiddenSecrets())
            assert isinstance(provider.inner, FallbackRacingLLMProvider)
            provider.inner.clock = lambda: delay.now
            provider.inner.sleeper = delay.sleep
            return provider

        provider = provider_factory()
        sender = FakeSender()
        values = dict(
            low_latency_mode=True,
            low_latency_finalize_wait_ms=0,
            low_latency_awaiting_vad_timeout_s=60,
            source_language="en",
            target_language="es",
            chatbox_include_source=False,
            system_prompt="${sourceName}|${targetName}",
        )
        values.update(configuration)
        harness = compose_translation_test_harness(
            stt=None,
            llm=provider,
            osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
            **values,
        )
        return harness, provider, provider_factory, delay, sender, clients

    return build


async def send_final(harness, utterance_id, text="Synthetic hello"):
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(utterance_id=utterance_id, text=text, is_final=True),
        )
    )


def error_events(harness):
    events = []
    while not harness.ui_events.empty():
        event = harness.ui_events.get_nowait()
        if event.type == UIEventType.ERROR:
            events.append(event)
    return events


async def target_snapshot(harness, target_index, status):
    async with asyncio.timeout(2):
        while True:
            event = await harness.ui_events.get()
            assert event.type is not UIEventType.ERROR
            if event.type is UIEventType.SELF_TURN_UPDATED:
                if event.payload.targets[target_index].status == status:
                    return event.payload


async def close_owner(harness, clients):
    await harness.self_owner.close()
    await harness.stop()
    assert clients and all(client.is_closed for client in clients)


@pytest.mark.parametrize("duplicate_outcome", ["service_failure", "winner", "request_limit"])
async def test_unchanged_speculative_result_keeps_one_index_request_budget(
    build_owner, duplicate_outcome
):
    sent = []
    dispatched = [asyncio.Event(), asyncio.Event()]
    active = 0

    async def handler(request):
        nonlocal active
        active += 1
        try:
            sent.append(delay.now)
            dispatched[min(len(sent) - 1, 1)].set()
            if len(sent) == 2 and duplicate_outcome == "winner":
                return success()
            return httpx.Response(413 if duplicate_outcome == "request_limit" else 503)
        finally:
            active -= 1

    harness, provider, _, delay, sender, clients = build_owner(handler)
    failures = []

    def settled(exception):
        assert active == 0
        assert not provider.inner._operation_tasks
        failures.append(exception)

    harness.translation_requests.settled_failure_sink = settled
    utterance_id = uuid4()
    try:
        await send_final(harness, utterance_id)
        await asyncio.wait_for(dispatched[0].wait(), 1)
        await asyncio.wait_for(delay.scheduled.wait(), 1)
        await delay.advance_to(1.299)
        assert sent == [0.0]
        await delay.advance_to(1.3)
        await target_snapshot(harness, 0, "ready" if duplicate_outcome == "winner" else "failed")
        assert sent == [0.0, 1.3]
        assert failures == []
        assert error_events(harness) == []
        assert sender.sent == (["Valid translation"] if duplicate_outcome == "winner" else [])
        await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 1)
        await delay.advance_to(100)
        assert harness.self_owner.merge_buffer is None
        assert sent == [0.0, 1.3]
        if duplicate_outcome == "winner":
            assert failures == []
            assert sender.sent == ["Valid translation"]
            assert error_events(harness) == []
        else:
            assert len(failures) == 1
            assert is_index_service_failure(failures[0]) == (duplicate_outcome == "service_failure")
            assert sender.sent == []
            assert len(error_events(harness)) == 1
        assert [entry.text for entry in harness.self_runtime.translation_history] == [
            "Synthetic hello"
        ]
    finally:
        await close_owner(harness, clients)


async def test_secondary_prestarted_failure_keeps_its_index_budget_until_final_admission(
    build_owner,
):
    sent = {"primary": [], "secondary": []}
    started = {"primary": asyncio.Event(), "secondary": asyncio.Event()}
    primary_release = asyncio.Event()
    active = {"primary": 0, "secondary": 0}

    async def handler(request):
        payload = json.loads(request.content)
        target = "secondary" if "Japanese" in payload["messages"][0]["content"] else "primary"
        sent[target].append(delay.now)
        started[target].set()
        active[target] += 1
        try:
            if target == "secondary":
                return httpx.Response(503)
            await primary_release.wait()
            return success("Primary translation")
        finally:
            active[target] -= 1

    harness, _, _, delay, sender, clients = build_owner(
        handler, self_target_languages=("es", "ja"), low_latency_finalize_wait_ms=1000
    )
    failures = []

    def settled(exception):
        assert active["secondary"] == 0
        failures.append(exception)

    harness.translation_requests.settled_failure_sink = settled
    utterance_id = uuid4()
    try:
        await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))
        await send_final(harness, utterance_id)
        await asyncio.wait_for(started["primary"].wait(), 1)
        await asyncio.wait_for(started["secondary"].wait(), 1)
        await asyncio.wait_for(delay.scheduled.wait(), 1)
        await delay.advance_to(1.299)
        assert sent == {"primary": [0.0], "secondary": [0.0]}
        await delay.advance_to(1.3)
        snapshot = await target_snapshot(harness, 1, "failed")
        assert not snapshot.source_closed
        assert failures == [] and error_events(harness) == []
        primary_release.set()
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        await delay.advance_to(100)
        assert sent == {"primary": [0.0, 1.3], "secondary": [0.0, 1.3]}
        assert len(failures) == 1 and is_index_service_failure(failures[0])
        assert len(error_events(harness)) == 1
        assert sender.sent == ["Primary translation"]
    finally:
        primary_release.set()
        await close_owner(harness, clients)


@pytest.mark.parametrize(
    "change", ["source", "soft_source", "configuration", "generation", "new_turn"]
)
async def test_changed_request_or_new_turn_gets_its_own_index_budget(build_owner, change):
    sent = []
    primary_started = asyncio.Event()

    async def handler(request):
        sent.append(delay.now)
        primary_started.set()
        return success("New request winner") if len(sent) == 4 else httpx.Response(503)

    harness, _, provider_factory, delay, sender, clients = build_owner(handler)
    failures = []
    harness.translation_requests.settled_failure_sink = failures.append
    utterance_id = uuid4()
    try:
        await send_final(harness, utterance_id)
        await asyncio.wait_for(primary_started.wait(), 1)
        await asyncio.wait_for(delay.scheduled.wait(), 1)
        await delay.advance_to(1.3)
        await target_snapshot(harness, 0, "failed")
        assert failures == []
        text = "Synthetic hello"
        if change == "source":
            text += " changed"
        elif change == "soft_source":
            text += "..."
        elif change == "configuration":
            harness.replace_configuration(system_prompt="New ${sourceName}|${targetName}")
        elif change == "generation":
            await harness.replace_llm_provider(provider_factory())
        else:
            await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 1)
            assert len(failures) == 1 and len(error_events(harness)) == 1
            utterance_id = uuid4()
        primary_started.clear()
        delay.scheduled.clear()
        await send_final(harness, utterance_id, text)
        await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))
        await asyncio.wait_for(primary_started.wait(), 1)
        await asyncio.wait_for(delay.scheduled.wait(), 1)
        await delay.advance_to(2.599)
        assert sent == [0.0, 1.3, 1.3]
        await delay.advance_to(2.6)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 1)
        await delay.advance_to(100)
        assert sent == [0.0, 1.3, 1.3, 2.6]
        assert len(failures) == (1 if change == "new_turn" else 0)
        assert error_events(harness) == []
        assert sender.sent == ["New request winner"]
    finally:
        await close_owner(harness, clients)


@pytest.mark.parametrize("settled", [False, True])
async def test_self_retirement_discards_index_failure_and_pending_duplicate(build_owner, settled):
    sent = []
    primary_started = asyncio.Event()

    async def handler(request):
        sent.append(delay.now)
        primary_started.set()
        return httpx.Response(503)

    harness, _, _, delay, sender, clients = build_owner(handler)
    failures = []
    harness.translation_requests.settled_failure_sink = failures.append
    try:
        await send_final(harness, uuid4())
        await asyncio.wait_for(primary_started.wait(), 1)
        await asyncio.wait_for(delay.scheduled.wait(), 1)
        if settled:
            await delay.advance_to(1.3)
            await target_snapshot(harness, 0, "failed")
        await harness.self_owner.reset_provider_channel()
        await delay.advance_to(100)
        assert harness.self_owner.merge_buffer is None
        assert len(sent) == (2 if settled else 1)
        assert failures == [] and error_events(harness) == [] and sender.sent == []
    finally:
        await close_owner(harness, clients)
