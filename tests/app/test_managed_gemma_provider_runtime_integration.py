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


def _translation_config(*, enabled: bool) -> object:
    return SimpleNamespace(
        snapshot=lambda: SimpleNamespace(value=SimpleNamespace(translation_enabled=enabled))
    )


def _components(
    *,
    settings: AppSettings,
    runtime: RecordingLlmRuntime,
    managed_gemma: object,
    events: list[object],
    translation_enabled: bool | None = None,
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
        translation_runtime_configuration_provider=lambda: (
            None
            if translation_enabled is None
            else _translation_config(enabled=translation_enabled)
        ),
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
async def test_provider_is_installed_without_preparing_when_translation_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    provider = object()
    activation_runtime = object()

    class ManagedGemma:
        runtime = activation_runtime

        async def prepare(self, _selection: ManagedGemmaTranslationSelection) -> object:
            events.append("prepare")
            raise AssertionError("rebuild must not prepare managed Gemma while translation is off")

    def create_backend(_settings: AppSettings, **kwargs: object) -> object:
        events.append("create")
        assert kwargs["managed_gemma_runtime"] is activation_runtime
        assert kwargs["managed_gemma_release"] is wiring_provider_runtime.noop_managed_gemma_release
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

    await components.llm_rebuild.rebuild()

    assert runtime.provider is provider
    assert events[:3] == [("replace", None), "create", ("replace", provider)]
    assert "prepare" not in events


@pytest.mark.asyncio
async def test_provider_waits_for_readiness_when_translation_is_on(
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
        runtime = object()

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
        assert kwargs["managed_gemma_release"] is wiring_provider_runtime.noop_managed_gemma_release
        return provider

    monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", create_backend)
    runtime = RecordingLlmRuntime(events)
    settings = _managed_settings()
    components = _components(
        settings=settings,
        runtime=runtime,
        managed_gemma=ManagedGemma(),
        events=events,
        translation_enabled=True,
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
async def test_provider_prepares_when_peer_translation_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    provider = object()
    activation_runtime = object()

    class ManagedGemma:
        runtime = object()

        async def prepare(self, _selection: ManagedGemmaTranslationSelection) -> object:
            events.append("prepare")
            return SimpleNamespace(runtime=activation_runtime, release=lambda: None)

    def create_backend(_settings: AppSettings, **kwargs: object) -> object:
        events.append("create")
        assert kwargs["managed_gemma_runtime"] is activation_runtime
        assert kwargs["managed_gemma_release"] is wiring_provider_runtime.noop_managed_gemma_release
        return provider

    monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", create_backend)
    runtime = RecordingLlmRuntime(events)
    settings = _managed_settings()
    settings.ui.peer_translation_enabled = True
    components = _components(
        settings=settings,
        runtime=runtime,
        managed_gemma=ManagedGemma(),
        events=events,
        translation_enabled=False,
    )

    await components.llm_rebuild.rebuild()

    assert runtime.provider is provider
    assert events[:4] == [("replace", None), "prepare", "create", ("replace", provider)]


@pytest.mark.asyncio
async def test_provider_construction_failure_keeps_demand_owned_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    async def release() -> None:
        events.append("release")

    class ManagedGemma:
        runtime = object()

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
        translation_enabled=True,
    )

    await components.llm_rebuild.rebuild()

    assert runtime.provider is None
    assert "release" not in events
    assert events[-2:] == [
        ("needs-key", False),
        ("failure", "LLM provider not available"),
    ]
