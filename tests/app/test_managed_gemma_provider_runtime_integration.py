from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.app.wiring import wiring_provider_runtime
from puripuly_heart.app.wiring.wiring_provider_runtime import compose_provider_runtime
from puripuly_heart.config.prompts import render_translation_prompt_template
from puripuly_heart.config.settings import (
    AppSettings,
    TranslationConnection,
    TranslationModel,
    materialize_translation_settings,
)
from puripuly_heart.core.language import get_llm_language_name


class RecordingLlmRuntime:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.provider = object()

    async def replace_provider(self, provider: object | None, *, start: bool) -> object | None:
        assert start is False
        previous = self.provider
        self.provider = provider
        self.events.append(("replace", provider))
        return previous


def _managed_settings() -> AppSettings:
    settings = AppSettings()
    settings.translation.model = TranslationModel.MANAGED_GEMMA
    settings.translation.connection = TranslationConnection.GPU
    return materialize_translation_settings(settings)


def _components(
    *,
    settings: AppSettings,
    runtime: RecordingLlmRuntime,
    managed_gemma: object,
    events: list[object],
):
    async def no_op() -> None:
        return None

    async def no_op_bool(_value: bool) -> None:
        return None

    return compose_provider_runtime(
        config_path=Path("settings.json"),
        settings=SimpleNamespace(current=settings),
        llm_runtime_provider=lambda: runtime,
        http_extensions=SimpleNamespace(),
        local_asr_runtime_provider=lambda: None,
        translation_runtime_configuration_provider=lambda: None,
        self_capture_provider=lambda: None,
        self_capture_owner=lambda: SimpleNamespace(),
        peer=lambda: SimpleNamespace(),
        peer_desired=lambda _settings: False,
        canonical_settings=lambda _settings: SimpleNamespace(),
        clear_local_pending=lambda: None,
        sync_local_notice=lambda: None,
        managed_pending_sink=lambda _value: None,
        managed_pending_provider=lambda: False,
        dashboard_managed_pending_sink=lambda _value: None,
        sync_effective_flags=lambda _settings: None,
        refresh_overlay=lambda: None,
        refresh_peer_runtime=no_op,
        replace_self_stt=no_op_bool,
        self_state_sink=lambda _state: None,
        self_availability=lambda _state: True,
        gpu_recovery=lambda _settings, _plan: no_op(),
        managed_release=lambda: SimpleNamespace(),
        managed_delegate_ready=lambda: None,
        runtime_logging=None,
        translation_needs_key_sink=lambda value: events.append(("needs-key", value)),
        usage_refresh=no_op,
        failure_sink=lambda message: events.append(("failure", message)),
        success_sink=lambda message: events.append(("success", message)),
        additional_signature_sink=lambda _settings: None,
        managed_gemma=managed_gemma,
    )


@pytest.mark.asyncio
async def test_provider_is_not_installed_until_managed_readiness_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    entered = asyncio.Event()
    ready = asyncio.Event()
    provider = object()
    activation_runtime = object()

    async def release() -> None:
        events.append("release")

    class ManagedGemma:
        async def prepare(self, selection: ManagedGemmaTranslationSelection) -> object:
            assert selection == ManagedGemmaTranslationSelection(
                backend="gpu",
                source_language="ko",
                target_language="en",
                system_prompt=render_translation_prompt_template(
                    settings.system_prompt,
                    source_name=get_llm_language_name("ko"),
                    target_name=get_llm_language_name("en"),
                ),
            )
            events.append("prepare")
            entered.set()
            await ready.wait()
            return SimpleNamespace(runtime=activation_runtime, release=release)

    def create_backend(_settings: AppSettings, **kwargs: object) -> object:
        events.append("create")
        assert kwargs["managed_gemma_runtime"] is activation_runtime
        assert kwargs["managed_gemma_release"] is release
        return provider

    monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", create_backend)
    runtime = RecordingLlmRuntime(events)
    settings = _managed_settings()
    components = _components(
        settings=settings,
        runtime=runtime,
        managed_gemma=ManagedGemma(),
        events=events,
    )

    rebuild = asyncio.create_task(components.llm_rebuild.rebuild())
    await entered.wait()

    assert runtime.provider is None
    assert events == [("replace", None), "prepare"]

    ready.set()
    await rebuild

    assert runtime.provider is provider
    assert events[:4] == [("replace", None), "prepare", "create", ("replace", provider)]


@pytest.mark.asyncio
async def test_provider_construction_failure_releases_prepared_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    async def release() -> None:
        events.append("release")

    class ManagedGemma:
        async def prepare(self, _selection: ManagedGemmaTranslationSelection) -> object:
            return SimpleNamespace(runtime=object(), release=release)

    def fail_backend(_settings: AppSettings, **_kwargs: object) -> object:
        raise RuntimeError("provider construction failed")

    monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", fail_backend)
    runtime = RecordingLlmRuntime(events)
    components = _components(
        settings=_managed_settings(),
        runtime=runtime,
        managed_gemma=ManagedGemma(),
        events=events,
    )

    await components.llm_rebuild.rebuild()

    assert runtime.provider is None
    assert "release" in events
    assert events[-2:] == [
        ("needs-key", False),
        ("failure", "LLM provider not available"),
    ]
