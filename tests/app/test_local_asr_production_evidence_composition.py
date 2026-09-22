import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import puripuly_heart.composition.local_asr_production_evidence as composition_module
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.domain.events import STTSessionState, UIEvent, UIEventType
from puripuly_heart.ui.event_dispatch import UIEventBridge


@pytest.mark.asyncio
async def test_evidence_ui_bridge_drains_publications_before_handoff_commit(
    monkeypatch,
) -> None:
    queue: asyncio.Queue[UIEvent] = asyncio.Queue(maxsize=1)
    old_provider = _BoundaryProvider(at_boundary=False)
    new_provider = _BoundaryProvider(at_boundary=True)
    provider_handle = ProviderRuntimeHandle(name="evidence-test", provider=old_provider)
    published_statuses: list[str] = []
    bridge = UIEventBridge(
        event_queue=queue,
        dashboard_destination=SimpleNamespace(
            publish_status=published_statuses.append,
            publish_transcript=lambda *_args, **_kwargs: True,
            publish_translation=lambda *_args, **_kwargs: True,
            publish_error=lambda _text: None,
        ),
        history_destination=SimpleNamespace(
            append_entry=lambda *_args, **_kwargs: None,
        ),
    )
    bridge_task: asyncio.Task[None] | None = None

    async def start_application_events() -> None:
        nonlocal bridge_task
        bridge_task = asyncio.create_task(bridge.run())
        await bridge.wait_started()

    start_callbacks = SimpleNamespace(
        start_output=lambda _auto_flush: _record_async([], None),
        open_self_ingress=lambda: _record_async([], None),
        open_peer_ingress=lambda: _record_async([], None),
        start_translation_turns=lambda: _record_async([], None),
        start_local_asr=lambda: _record_async([], None),
    )
    access = SimpleNamespace(
        config_path=Path("settings.json"),
        start_callbacks=start_callbacks,
        start_application_events=start_application_events,
    )

    class FakeApplication:
        async def stop(self) -> None:
            bridge.close()
            if bridge_task is not None:
                bridge_task.cancel()
                await asyncio.gather(bridge_task, return_exceptions=True)

    def compose_runtime(**kwargs):
        kwargs["local_asr_evidence_sink"](access)
        return FakeApplication()

    monkeypatch.setattr(
        composition_module,
        "compose_application_runtime",
        compose_runtime,
    )
    evidence = composition_module.compose_local_asr_production_evidence(
        config_path=Path("settings.json"),
    )

    try:
        await evidence.start_runtime()

        handoff = asyncio.create_task(
            provider_handle.handoff_provider_at_boundary(new_provider, start=True)
        )
        await asyncio.sleep(0)
        await asyncio.wait_for(
            _publish_states_and_commit_handoff(queue, provider_handle),
            timeout=0.2,
        )
        assert await handoff is old_provider
        assert provider_handle.provider is new_provider
        await asyncio.wait_for(queue.join(), timeout=0.2)
    finally:
        await evidence.close()
        await provider_handle.close()

    assert published_statuses == ["connected", "stopping"]


@pytest.mark.asyncio
async def test_initialize_preserves_canonical_owner_failure_contract(
    monkeypatch,
) -> None:
    access = SimpleNamespace(
        config_path=Path("settings.json"),
        load_compatibility_settings=lambda: object(),
        initialize=lambda _value: _record_async([], None),
        owner=object(),
        retry_gpu_activation=lambda: _record_async([], None),
    )

    def compose_runtime(**kwargs):
        kwargs["local_asr_evidence_sink"](access)
        return SimpleNamespace(stop=lambda: _record_async([], None))

    monkeypatch.setattr(
        composition_module,
        "compose_application_runtime",
        compose_runtime,
    )
    evidence = composition_module.compose_local_asr_production_evidence(
        config_path=Path("settings.json"),
    )

    with pytest.raises(
        RuntimeError,
        match="production application did not compose the canonical owner",
    ):
        await evidence.initialize(object())


class _BoundaryProvider:
    def __init__(self, *, at_boundary: bool) -> None:
        self.is_at_utterance_boundary = at_boundary

    async def close(self) -> None:
        return None


async def _publish_states_and_commit_handoff(
    queue: asyncio.Queue[UIEvent],
    provider_handle: ProviderRuntimeHandle,
) -> None:
    await queue.put(
        UIEvent(
            type=UIEventType.SESSION_STATE_CHANGED,
            payload=STTSessionState.STREAMING,
            channel="peer",
        )
    )
    await queue.put(
        UIEvent(
            type=UIEventType.SESSION_STATE_CHANGED,
            payload=STTSessionState.DRAINING,
            channel="self",
        )
    )
    await provider_handle.commit_pending_handoff()


async def _record_async(events: list[object], value: object) -> None:
    events.append(value)
