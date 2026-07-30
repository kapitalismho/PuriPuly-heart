from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from threading import Barrier, Event
from uuid import uuid4

import pytest

from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigCategory,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationTurnLifecycleOwner,
    TranslationTurnProcessResult,
)
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.translation_owners import compose_translation_test_harness

_CONFIG_FIELD_NAMES = {field.name for field in fields(TranslationRuntimeConfig)}


class FakeOsc:
    def __init__(self) -> None:
        self.messages = []

    def enqueue(self, message) -> None:
        self.messages.append(message)

    def send_immediate(self, _text: str) -> bool:
        return True

    def send_typing(self, _is_typing: bool) -> None:
        return None

    def set_typing_reason(self, _reason: str, _active: bool) -> None:
        return None

    def clear_typing_reasons(self) -> None:
        return None

    def process_due(self) -> None:
        return None


def test_owner_replaces_one_frozen_snapshot_with_typed_change() -> None:
    owner = TranslationRuntimeConfigurationOwner()
    before = owner.snapshot()
    next_value = replace(
        before.value,
        source_language="ja",
        system_prompt="prompt",
        translation_enabled=False,
        low_latency_finalize_wait_ms=250,
    )

    change = owner.replace(next_value)

    assert change.before is before
    assert change.after is owner.snapshot()
    assert change.after.revision == before.revision + 1
    assert change.after.value is next_value
    assert change.changed_fields == {
        "source_language",
        "system_prompt",
        "translation_enabled",
        "low_latency_finalize_wait_ms",
    }
    assert change.categories == {
        TranslationRuntimeConfigCategory.LANGUAGES,
        TranslationRuntimeConfigCategory.PROMPT,
        TranslationRuntimeConfigCategory.ENABLEMENT,
        TranslationRuntimeConfigCategory.LOW_LATENCY,
    }
    assert change.self_language_changed is True
    assert change.peer_language_changed is True


def test_change_tracks_effective_peer_language_fallback() -> None:
    initial = TranslationRuntimeConfig(
        source_language="ko",
        target_language="en",
        peer_source_language="ja",
        peer_target_language="fr",
    )
    owner = TranslationRuntimeConfigurationOwner(initial)

    source_only_change = owner.replace(replace(initial, source_language="zh-CN"))
    peer_change = owner.replace(replace(source_only_change.after.value, peer_source_language=""))

    assert source_only_change.self_language_changed is True
    assert source_only_change.peer_language_changed is False
    assert peer_change.self_language_changed is False
    assert peer_change.peer_language_changed is True


def test_owner_snapshots_never_expose_mixed_concurrent_values() -> None:
    first = TranslationRuntimeConfig(
        source_language="ko",
        target_language="en",
        system_prompt="first",
    )
    second = replace(
        first,
        source_language="ja",
        target_language="fr",
        system_prompt="second",
    )
    owner = TranslationRuntimeConfigurationOwner(first)
    done = Event()
    observed: list[tuple[str, str, str]] = []

    def writer() -> None:
        for index in range(1000):
            owner.replace(first if index % 2 == 0 else second)
        done.set()

    def reader() -> None:
        while not done.is_set():
            value = owner.snapshot().value
            observed.append((value.source_language, value.target_language, value.system_prompt))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(reader) for _ in range(3)]
        executor.submit(writer).result()
        for future in futures:
            future.result()

    assert observed
    assert set(observed) <= {("ko", "en", "first"), ("ja", "fr", "second")}


def test_peer_owner_has_no_mutable_translation_configuration_fields() -> None:
    owner_fields = {field.name for field in fields(PeerTranslationChannelOwner)}

    assert not owner_fields & _CONFIG_FIELD_NAMES
    assert "config_snapshot" in owner_fields
    assert "translation_runtime_configuration" not in owner_fields


def test_translation_fixture_compatibility_access_delegates_to_the_owner() -> None:
    owner = TranslationRuntimeConfigurationOwner()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=FakeOsc(),
        translation_runtime_configuration=owner,
    )

    harness.replace_configuration(source_language="ja")

    assert harness.configuration.snapshot().value.source_language == "ja"
    assert owner.snapshot().revision == 1
    assert owner.snapshot().value.source_language == "ja"


def test_fixture_configuration_update_is_linearizable_with_owner_updates() -> None:
    barrier = Barrier(2)

    class CoordinatedOwner(TranslationRuntimeConfigurationOwner):
        def transform(self, transformer):
            barrier.wait(timeout=1)
            return super().transform(transformer)

    owner = CoordinatedOwner()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=FakeOsc(),
        translation_runtime_configuration=owner,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        compatibility_future = executor.submit(
            harness.replace_configuration,
            source_language="ja",
        )
        owner_future = executor.submit(
            owner.transform,
            lambda value: replace(value, translation_enabled=False),
        )
        compatibility_future.result()
        owner_future.result()

    snapshot = owner.snapshot()
    assert snapshot.revision == 2
    assert snapshot.value.source_language == "ja"
    assert snapshot.value.translation_enabled is False


@pytest.mark.asyncio
async def test_translation_turn_submit_parent_captures_the_configuration_revision() -> None:
    config_owner = TranslationRuntimeConfigurationOwner()
    captured = []

    async def child_created(child) -> None:
        captured.append(child)

    async def child_started(_child, _task) -> None:
        return None

    async def process_child(_child, _cancellation_requested):
        await asyncio.sleep(0)
        return TranslationTurnProcessResult("source_only")

    async def child_terminal(_child, _outcome) -> None:
        return None

    async def parent_callback(_parent_id) -> None:
        return None

    lifecycle = TranslationTurnLifecycleOwner(
        on_child_created=child_created,
        on_child_started=child_started,
        process_child=process_child,
        on_child_terminal=child_terminal,
        on_parent_closed=parent_callback,
        on_parent_rejected=parent_callback,
        config_snapshot=config_owner.snapshot,
    )
    snapshot = config_owner.snapshot()
    transcript = Transcript(
        utterance_id=uuid4(),
        text="hello",
        is_final=True,
        created_at=1.0,
        channel="self",
    )

    await lifecycle.submit_parent(
        transcript,
        source="Mic",
        target_languages=("en",),
    )
    config_owner.replace(replace(snapshot.value, target_language="fr"))
    await lifecycle.wait_for_idle()

    assert len(captured) == 1
    assert captured[0].config_snapshot is snapshot


@pytest.mark.asyncio
async def test_translation_fixture_captures_configuration_when_the_turn_is_created() -> None:
    owner = TranslationRuntimeConfigurationOwner(TranslationRuntimeConfig(target_language="en"))
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=FakeOsc(),
        translation_runtime_configuration=owner,
    )
    captured = []

    class Turns:
        async def submit(self, request, *, wait_for_parent: bool = False):
            _ = wait_for_parent
            captured.append(request)
            return (request.transcript.utterance_id,)

    harness.replace_translation_turn_owner_for_test(Turns())
    transcript = Transcript(
        utterance_id=uuid4(),
        text="hello",
        is_final=True,
        created_at=1.0,
        channel="self",
    )

    await harness.ensure_translation(transcript)
    owner.replace(replace(owner.snapshot().value, target_language="fr"))

    assert len(captured) == 1
    assert captured[0].config_snapshot.revision == 0
    assert captured[0].config_snapshot.value.target_language == "en"
    assert captured[0].target_languages == ("en",)


@pytest.mark.asyncio
async def test_in_flight_turn_keeps_output_policy_and_later_turn_uses_replacement() -> None:
    class GatedLLM:
        def __init__(self) -> None:
            self.started: asyncio.Queue[None] = asyncio.Queue()
            self.release: asyncio.Queue[None] = asyncio.Queue()

        async def translate(
            self,
            *,
            utterance_id,
            text,
            system_prompt,
            source_language,
            target_language,
            context="",
        ):
            _ = (system_prompt, context)
            await self.started.put(None)
            await self.release.get()
            return Translation(
                utterance_id=utterance_id,
                text="translated",
                source_text=text,
                source_language=source_language,
                target_language=target_language,
                channel="self",
            )

        async def close(self) -> None:
            return None

    owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(
            source_language="en",
            target_language="ko",
            chatbox_include_source=True,
        )
    )
    llm = GatedLLM()
    osc = FakeOsc()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=osc,
        translation_runtime_configuration=owner,
    )

    async def start_turn(text: str):
        transcript = Transcript(
            utterance_id=uuid4(),
            text=text,
            is_final=True,
            channel="self",
        )
        task = asyncio.create_task(
            harness.ensure_translation(
                transcript,
                wait_for_parent=True,
            )
        )
        await asyncio.wait_for(llm.started.get(), timeout=1)
        return task

    try:
        first = await start_turn("first")
        owner.transform(lambda value: replace(value, chatbox_include_source=False))
        await llm.release.put(None)
        await first

        second = await start_turn("second")
        await llm.release.put(None)
        await second
    finally:
        await harness.translation_turns.close()

    assert [message.text for message in osc.messages] == [
        "first (translated)",
        "translated",
    ]
