from __future__ import annotations

import asyncio
import json
import threading
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.config.prompts import (
    load_prompt_for_provider,
    render_translation_prompt_template,
)
from puripuly_heart.core.llm.index_translate import (
    IndexRequestLimitError,
    IndexServiceUnavailableError,
    IndexTranslateError,
    is_index_service_failure,
)
from puripuly_heart.providers.llm import index_translate
from puripuly_heart.providers.llm.messages import build_translation_user_message


def _completion(content: object = "Hola.", *, finish_reason: object = "stop") -> dict:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ]
    }


@pytest.fixture
def transport_factory(monkeypatch):
    clients = []
    options = []
    original = index_translate.network_clients.external_async_client
    tls = object()
    policy = object()
    monkeypatch.setattr(index_translate, "select_tls", lambda: tls)
    monkeypatch.setattr(index_translate, "ProxyPolicy", lambda: policy)

    def install(handler):
        def external(**kwargs):
            options.append(kwargs)
            client = original(**kwargs, transport=httpx.MockTransport(handler))
            clients.append(client)
            return client

        monkeypatch.setattr(index_translate.network_clients, "external_async_client", external)
        return clients, options, tls, policy

    return install


def _kwargs(**overrides):
    return {
        "utterance_id": uuid4(),
        "text": "Hello.",
        "system_prompt": "Rendered prompt {source_language} with literal {braces}",
        "source_language": "en",
        "target_language": "es",
        "context": "Prior synthetic exchange",
        "scene_participant_count": 4,
        **overrides,
    }


@pytest.mark.parametrize("override", [None, "Override ${source_name} -> ${target_name} {literal}"])
async def test_request_preserves_rendered_prompt_and_shared_input_preparation(
    transport_factory, override
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_completion())

    clients, options, tls, policy = transport_factory(handler)
    template = load_prompt_for_provider("index_translate") if override is None else override
    assert load_prompt_for_provider("index_translate") == load_prompt_for_provider("openrouter")
    prompt = render_translation_prompt_template(
        template, source_name="English", target_name="Spanish", input_channel="peer"
    )
    provider = index_translate.IndexTranslateProvider()
    kwargs = _kwargs(system_prompt=prompt, max_output_tokens=123)
    try:
        result = await provider.translate(**kwargs)
        assert result.utterance_id == kwargs["utterance_id"]
        assert result.text == "Hola."
        assert len(requests) == 1
        request = requests[0]
        assert str(request.url) == index_translate.INDEX_TRANSLATE_ENDPOINT
        assert request.method == "POST"
        assert request.headers["User-Agent"] == "PuriPulyHeart"
        assert "authorization" not in request.headers
        assert json.loads(request.content) == {
            "model": "Index-Translate-35B-A3B",
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": build_translation_user_message(
                        text=kwargs["text"],
                        context=kwargs["context"],
                        scene_participant_count=kwargs["scene_participant_count"],
                    ),
                },
            ],
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 123,
            "stream": False,
        }
        assert options[0]["tls"] is tls
        assert options[0]["policy"] is policy
    finally:
        await provider.close()
    assert len(clients) == 1
    assert clients[0].is_closed


async def test_default_budget_probe_and_translation_share_one_owned_client(transport_factory):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    clients, *_ = transport_factory(handler)
    provider = index_translate.IndexTranslateProvider()
    try:
        assert await provider.check_availability() is None
        await provider.translate(**_kwargs())
        assert len(bodies) == 2
        assert bodies[0]["max_tokens"] == 64
        assert "Prior synthetic exchange" not in str(bodies[0])
        assert bodies[1]["max_tokens"] == index_translate.INDEX_DEFAULT_OUTPUT_TOKENS
        assert bodies[1]["messages"][0]["content"] == _kwargs()["system_prompt"]
        assert len(clients) == 1
    finally:
        await provider.close()
        await provider.close()
    with pytest.raises(asyncio.CancelledError):
        await provider.check_availability()
    assert len(clients) == 1


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"finish_reason": "stop", "message": None}]},
        _completion(""),
        _completion(" \n "),
        _completion(None),
        _completion("<think>private reasoning</think>Hola."),
        _completion("<analysis>private</analysis>"),
        _completion("<|thinking|>private"),
        _completion([{"type": "reasoning", "text": "private"}]),
        _completion("Hola.", finish_reason=None),
        _completion("Hola.", finish_reason="tool_calls"),
        _completion("Hola.", finish_reason="content_filter"),
        {"choices": [{"finish_reason": "stop", "message": {"role": "user", "content": "x"}}]},
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "reasoning_content": "private"},
                }
            ]
        },
    ],
)
async def test_invalid_terminal_responses_never_become_subtitles(transport_factory, data):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=data)

    transport_factory(handler)
    provider = index_translate.IndexTranslateProvider()
    try:
        with pytest.raises(IndexServiceUnavailableError) as failure:
            await provider.translate(**_kwargs())
        assert failure.value.reason == "invalid_response"
        assert len(calls) == 1
        assert is_index_service_failure(failure.value)
    finally:
        await provider.close()


@pytest.mark.parametrize("content", [" Hola. ", [{"type": "text", "text": "Hola."}]])
async def test_valid_content_does_not_publish_separate_reasoning(transport_factory, content):
    data = _completion(content)
    data["choices"][0]["message"]["reasoning_content"] = "private reasoning"
    transport_factory(lambda request: httpx.Response(200, json=data))
    provider = index_translate.IndexTranslateProvider()
    try:
        assert (await provider.translate(**_kwargs())).text == "Hola."
    finally:
        await provider.close()


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
async def test_http_service_failure_is_typed_without_retry_or_response_text(
    transport_factory, status
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="private server body")

    transport_factory(handler)
    provider = index_translate.IndexTranslateProvider()
    try:
        with pytest.raises(IndexServiceUnavailableError) as failure:
            await provider.check_availability()
        assert failure.value.status_code == status
        assert "private server body" not in str(failure.value)
        assert len(calls) == 1
    finally:
        await provider.close()


@pytest.mark.parametrize(
    ("status", "data", "expected_type", "reason"),
    [
        (413, {}, IndexRequestLimitError, "input_limit"),
        (400, {}, IndexTranslateError, "invalid_request"),
        (422, {}, IndexTranslateError, "invalid_request"),
        (
            200,
            _completion("partial", finish_reason="length"),
            IndexRequestLimitError,
            "output_limit",
        ),
    ],
)
async def test_request_limits_do_not_establish_service_failure(
    transport_factory, status, data, expected_type, reason
):
    transport_factory(lambda request: httpx.Response(status, json=data))
    provider = index_translate.IndexTranslateProvider()
    try:
        with pytest.raises(expected_type) as failure:
            await provider.translate(**_kwargs())
        assert failure.value.reason == reason
        assert not is_index_service_failure(failure.value)
    finally:
        await provider.close()


async def test_malformed_json_is_classified(transport_factory):
    transport_factory(lambda request: httpx.Response(200, text="not json"))
    provider = index_translate.IndexTranslateProvider()
    try:
        with pytest.raises(IndexServiceUnavailableError, match="invalid_response"):
            await provider.translate(**_kwargs())
    finally:
        await provider.close()


@pytest.mark.parametrize("cancel", [False, True])
async def test_pending_http_keeps_loop_responsive_and_preserves_cancellation(
    transport_factory, cancel
):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(request):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    clients, *_ = transport_factory(handler)
    provider = index_translate.IndexTranslateProvider(availability_timeout=0.05)
    task = asyncio.create_task(provider.check_availability())
    await asyncio.wait_for(started.wait(), 1)
    progressed = asyncio.Event()
    asyncio.get_running_loop().call_soon(progressed.set)
    await asyncio.wait_for(progressed.wait(), 1)
    if cancel:
        task.cancel()
    try:
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(IndexServiceUnavailableError, match="timeout"):
                await task
        assert cancelled.is_set()
    finally:
        await provider.close()
    assert clients[0].is_closed


async def test_tls_policy_preparation_is_off_loop_and_settled_before_close(
    transport_factory, monkeypatch
):
    clients, *_ = transport_factory(lambda request: httpx.Response(200, json=_completion()))
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def prepare():
        entered.set()
        release.wait(2)
        completed.set()
        return object()

    monkeypatch.setattr(index_translate, "select_tls", prepare)
    provider = index_translate.IndexTranslateProvider()
    task = asyncio.create_task(provider.check_availability())
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not completed.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await provider.close()
    assert completed.is_set()
    assert clients == []


@pytest.mark.parametrize("budget", [0, -1, True])
async def test_invalid_output_budget_does_not_dispatch_or_mark_service_unavailable(
    transport_factory, budget
):
    clients, *_ = transport_factory(lambda request: pytest.fail("Unexpected HTTP dispatch"))
    provider = index_translate.IndexTranslateProvider()
    try:
        with pytest.raises(IndexRequestLimitError) as failure:
            await provider.translate(**_kwargs(max_output_tokens=budget))
        assert not is_index_service_failure(failure.value)
        assert clients == []
    finally:
        await provider.close()


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_failures_are_classified_without_adapter_retry(
    transport_factory, error_type
):
    calls = []

    def handler(request):
        calls.append(request)
        raise error_type("private transport detail", request=request)

    transport_factory(handler)
    provider = index_translate.IndexTranslateProvider()
    try:
        with pytest.raises(IndexServiceUnavailableError) as failure:
            await provider.translate(**_kwargs())
        assert failure.value.reason == (
            "timeout" if error_type is httpx.ReadTimeout else "transport"
        )
        assert "private transport detail" not in str(failure.value)
        assert len(calls) == 1
    finally:
        await provider.close()
