from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from tests.helpers.translation_owners import compose_translation_test_harness
from tests.integration.helpers import (
    WARMUP_DELAY_S,
    MockOscSender,
    SimpleClock,
    chunk_audio,
    get_qwen_asr_endpoint,
    get_qwen_base_url,
    integration_mark,
    load_audio_wav,
    next_ui_event,
    require_env,
    require_module,
    resolve_test_audio_path,
    send_vad_events,
    wait_for_event,
)

pytestmark = integration_mark()


@pytest.mark.asyncio
async def test_qwen_asr_llm_pipeline_smoke() -> None:
    api_key = require_env("ALIBABA_API_KEY")
    require_module(
        "dashscope",
        reason="dashscope is required for this integration test; install project dependencies.",
    )

    from puripuly_heart.config.prompts import load_prompt_for_provider
    from puripuly_heart.core.language import get_llm_language_name
    from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
    from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
    from puripuly_heart.core.stt.controller import ManagedSTTProvider
    from puripuly_heart.domain.events import UIEventType
    from puripuly_heart.providers.llm.qwen import QwenLLMProvider
    from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend

    audio_path = resolve_test_audio_path()
    if not audio_path.exists():
        pytest.skip(f"Audio file not found: {audio_path}")

    audio_samples, sample_rate = load_audio_wav(audio_path)
    runtime_sample_rate_hz = 16000
    if sample_rate != runtime_sample_rate_hz:
        from puripuly_heart.core.audio.format import resample_f32_linear

        audio_samples = resample_f32_linear(
            audio_samples,
            from_rate_hz=sample_rate,
            to_rate_hz=runtime_sample_rate_hz,
        )
        sample_rate = runtime_sample_rate_hz

    stt_backend = QwenASRRealtimeSTTBackend(
        api_key=api_key,
        model=os.getenv("QWEN_ASR_MODEL", "qwen3-asr-flash-realtime"),
        endpoint=get_qwen_asr_endpoint(),
        language=os.getenv("QWEN_ASR_LANGUAGE", "ko"),
        sample_rate_hz=sample_rate,
    )
    stt = ManagedSTTProvider(
        backend=stt_backend,
        sample_rate_hz=sample_rate,
        reset_deadline_s=90.0,
        drain_timeout_s=5.0,
        bridging_ms=300,
    )

    llm_base = QwenLLMProvider(
        api_key=api_key,
        base_url=get_qwen_base_url(),
        model=os.getenv("QWEN_LLM_MODEL", "qwen3.8-flash"),
    )
    llm = SemaphoreLLMProvider(inner=llm_base, semaphore=asyncio.Semaphore(1))

    mock_sender = MockOscSender()
    osc = ChatboxPaginator(
        sender=mock_sender,
        clock=SimpleClock(),
        max_chars=144,
    )

    source_lang = os.getenv("QWEN_LLM_SOURCE_LANGUAGE", "ko")
    target_lang = os.getenv("QWEN_LLM_TARGET_LANGUAGE", "en")
    system_prompt = load_prompt_for_provider("qwen") or "Translate ${sourceName} to ${targetName}."
    system_prompt = system_prompt.replace("${sourceName}", get_llm_language_name(source_lang))
    system_prompt = system_prompt.replace("${targetName}", get_llm_language_name(target_lang))

    harness = compose_translation_test_harness(
        stt=stt,
        llm=llm,
        osc=osc,
        source_language=source_lang,
        target_language=target_lang,
        system_prompt=system_prompt,
        fallback_transcript_only=False,
        translation_enabled=True,
    )

    got_result = asyncio.Event()
    translation_text = ""
    error_message = ""

    async def track_events() -> None:
        nonlocal translation_text, error_message
        while True:
            event = await next_ui_event(harness.ui_events)
            if event is None:
                continue

            if event.type == UIEventType.TRANSLATION_DONE:
                translation_text = event.payload.text
                got_result.set()
            elif event.type == UIEventType.ERROR:
                error_message = str(event.payload)
                got_result.set()

    await harness.start(auto_flush_osc=True)
    event_task = asyncio.create_task(track_events())
    await asyncio.sleep(WARMUP_DELAY_S)

    try:
        utterance_id = uuid4()
        try:
            chunks, _chunk_samples = chunk_audio(audio_samples, sample_rate_hz=sample_rate)
        except ValueError:
            pytest.skip("Invalid chunk size for sample rate")
        if not chunks:
            pytest.skip("Audio file is empty")

        await send_vad_events(harness, utterance_id, chunks)

        if not await wait_for_event(got_result):
            pytest.fail("Pipeline did not complete in time")
    finally:
        event_task.cancel()
        await harness.stop()
        await asyncio.gather(event_task, return_exceptions=True)

    if error_message:
        pytest.fail(f"Pipeline error: {error_message}")

    assert translation_text.strip()
