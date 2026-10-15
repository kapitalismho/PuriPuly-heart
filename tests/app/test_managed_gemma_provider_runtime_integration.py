from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.app.wiring import wiring_provider_runtime
from puripuly_heart.app.wiring.wiring_managed_gemma import managed_gemma_selection
from puripuly_heart.app.wiring.wiring_provider_runtime import compose_provider_runtime
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
)


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


def _managed_settings(model: str = TranslationModel.MANAGED_GEMMA.value) -> AppSettingsVNext:
    settings = AppSettingsVNext()
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model=model,
                connection=TranslationConnection.GPU.value,
            ),
        ),
    )


def _settings_owner(
    settings: AppSettingsVNext,
    *,
    peer_translation_enabled: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        canonical=settings,
        peer_translation_enabled=lambda: peer_translation_enabled,
    )


def _translation_config(*, enabled: bool) -> object:
    from puripuly_heart.core.orchestrator.configuration import (
        TranslationRuntimeConfig,
        TranslationRuntimeConfigurationOwner,
    )

    return TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=enabled)
    )


def _components(
    *,
    settings: AppSettingsVNext,
    runtime: RecordingLlmRuntime,
    managed_gemma: object,
    events: list[object],
    translation_enabled: bool | None = None,
    peer_translation_enabled: bool = False,
):
    async def no_op() -> None:
        return None

    async def no_op_bool(_value: bool) -> None:
        return None

    peer = SimpleNamespace(
        effective_enabled=lambda: False,
        activation_requested=lambda **_kwargs: False,
        last_runtime_signature=None,
        last_provider_signature=None,
    )
    configuration = (
        None if translation_enabled is None else _translation_config(enabled=translation_enabled)
    )

    return compose_provider_runtime(
        config_path=Path("settings.json"),
        settings=_settings_owner(
            settings,
            peer_translation_enabled=peer_translation_enabled,
        ),
        llm_runtime_provider=lambda: runtime,
        http_extensions=SimpleNamespace(),
        local_asr_runtime_provider=lambda: None,
        translation_runtime_configuration_provider=lambda: configuration,
        self_capture_provider=lambda: None,
        self_capture_owner=lambda: SimpleNamespace(),
        peer=lambda: peer,
        peer_desired=lambda _settings: False,
        canonical_settings=lambda current: current,
        clear_local_pending=lambda: None,
        sync_local_notice=lambda: None,
        managed_pending_sink=lambda _value: None,
        managed_pending_provider=lambda: False,
        dashboard_managed_pending_sink=lambda _value: None,
        sync_effective_flags=lambda: None,
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
@pytest.mark.parametrize("model", ["managed_gemma", "managed_gemma_26b"])
async def test_provider_is_installed_without_preparing_when_translation_is_off(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    events: list[object] = []
    provider = object()
    activation_runtime = object()

    class ManagedGemma:
        runtime = activation_runtime

        async def prepare(self, _selection: ManagedGemmaTranslationSelection) -> object:
            events.append("prepare")
            raise AssertionError("rebuild must not prepare managed Gemma while translation is off")

    def create_backend(**kwargs: object) -> object:
        events.append("create")
        assert kwargs["managed_gemma_runtime"] is activation_runtime
        assert kwargs["managed_gemma_release"] is wiring_provider_runtime.noop_managed_gemma_release
        return provider

    monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", create_backend)
    runtime = RecordingLlmRuntime(events)
    settings = _managed_settings(model)
    components = _components(
        settings=settings,
        runtime=runtime,
        managed_gemma=ManagedGemma(),
        events=events,
    )

    await components.llm_rebuild.rebuild()

    assert runtime.provider is provider
    assert events[:2] == ["create", ("replace", provider)]
    assert "prepare" not in events


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["managed_gemma", "managed_gemma_26b"])
async def test_provider_waits_for_readiness_when_translation_is_on(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
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
            assert selection == managed_gemma_selection(settings)
            events.append("prepare")
            entered.set()
            await ready.wait()
            return SimpleNamespace(runtime=activation_runtime, release=release)

    def create_backend(**kwargs: object) -> object:
        events.append("create")
        assert kwargs["managed_gemma_runtime"] is activation_runtime
        assert kwargs["managed_gemma_release"] is wiring_provider_runtime.noop_managed_gemma_release
        return provider

    monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", create_backend)
    runtime = RecordingLlmRuntime(events)
    previous = runtime.provider
    settings = _managed_settings(model)
    components = _components(
        settings=settings,
        runtime=runtime,
        managed_gemma=ManagedGemma(),
        events=events,
        translation_enabled=True,
    )

    rebuild = asyncio.create_task(components.llm_rebuild.rebuild())
    await entered.wait()

    assert runtime.provider is previous
    assert events == ["prepare"]

    ready.set()
    await rebuild

    assert runtime.provider is provider
    assert events[:3] == ["prepare", "create", ("replace", provider)]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["managed_gemma", "managed_gemma_26b"])
async def test_provider_prepares_when_peer_translation_is_on(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    events: list[object] = []
    provider = object()
    activation_runtime = object()

    class ManagedGemma:
        runtime = object()

        async def prepare(self, _selection: ManagedGemmaTranslationSelection) -> object:
            events.append("prepare")
            return SimpleNamespace(runtime=activation_runtime, release=lambda: None)

    def create_backend(**kwargs: object) -> object:
        events.append("create")
        assert kwargs["managed_gemma_runtime"] is activation_runtime
        assert kwargs["managed_gemma_release"] is wiring_provider_runtime.noop_managed_gemma_release
        return provider

    monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", create_backend)
    runtime = RecordingLlmRuntime(events)
    settings = _managed_settings(model)
    components = _components(
        settings=settings,
        runtime=runtime,
        managed_gemma=ManagedGemma(),
        events=events,
        translation_enabled=False,
        peer_translation_enabled=True,
    )

    await components.llm_rebuild.rebuild()

    assert runtime.provider is provider
    assert events[:3] == ["prepare", "create", ("replace", provider)]


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

    def fail_backend(**_kwargs: object) -> object:
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


def _connection(settings, backend):
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(settings.intent.translation, connection=backend),
        ),
    )


async def _bound_request(handle, transport, text, *, reverse=False):
    from uuid import uuid4

    from puripuly_heart.core.translation_backend import TranslationBackendRequest

    identity = uuid4()
    task = asyncio.create_task(
        handle.provider.translate(
            TranslationBackendRequest(
                utterance_id=identity,
                text=text,
                system_prompt="Translate {source_language} to {target_language}",
                source_language="en" if reverse else "ko",
                target_language="ko" if reverse else "en",
            )
        )
    )
    message, slot = await asyncio.wait_for(transport.started.get(), timeout=1)
    assert text in message
    return identity, task, message, slot


async def _bound_components(tmp_path, monkeypatch):
    from puripuly_heart.app.services.managed_gemma_translation import ManagedGemmaTranslationOwner
    from puripuly_heart.app.services.provider.provider_runtime_apply import ProviderRuntimeState
    from puripuly_heart.core.local_translation import runtime_profile
    from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
    from tests.core.local_translation.test_gemma_runtime import ControlledTransport, _runtime

    monkeypatch.setattr(
        runtime_profile, "default_threads_profile_path", lambda: tmp_path / "threads.json"
    )
    monkeypatch.setattr(
        wiring_provider_runtime, "create_secret_store", lambda *_args, **_kwargs: object()
    )
    native, commands, processes, transports, _calls, _logs = _runtime(
        tmp_path, transport_builder=ControlledTransport
    )
    managed = ManagedGemmaTranslationOwner(runtime=native)
    handle = ProviderRuntimeHandle(name="llm", provider=None)
    components = _components(
        settings=_managed_settings(),
        runtime=handle,
        managed_gemma=managed,
        events=[],
        translation_enabled=True,
    )
    components.runtime.state_provider = lambda _settings: ProviderRuntimeState(
        True, handle.provider is not None, False, False, False, False
    )
    components.runtime.self_runtime_convergence = lambda _settings: True
    components.runtime.peer_runtime_convergence = lambda _settings: None

    async def controlled_stt():
        return None

    components.runtime.refresh_self_stt = controlled_stt
    await components.llm_rebuild.rebuild()
    components.sync_signatures(components.effects.settings.canonical)
    return components, managed, handle, commands, processes, transports


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinated", [False, True], ids=["channel-refresh", "gpu-recovery"])
async def test_partial_stt_failure_retains_actual_llm_binding_and_rebuilds_return_to_gpu(
    tmp_path,
    monkeypatch,
    coordinated,
):
    from puripuly_heart.app.ports.runtime_apply import RuntimeApplyRequest
    from puripuly_heart.app.services.provider.provider_runtime_apply import (
        ProviderRuntimeApplyAdapter,
    )
    from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
        build_llm_provider_signature,
    )

    components, managed, handle, commands, processes, transports = await _bound_components(
        tmp_path, monkeypatch
    )
    initial = components.effects.settings.canonical
    initial_stt = components.signatures.last_self_provider
    components.runtime.self_runtime_convergence = lambda _settings: False
    components.runtime.gpu_restart_decision = lambda _before, _after: coordinated
    try:
        cpu = _connection(initial, "cpu")
        plan = components.runtime.build_plan(cpu, force_rebuild_llm=False)
        assert plan.should_rebuild_llm
        result = await ProviderRuntimeApplyAdapter(components.runtime, cpu, plan).apply_runtime(
            RuntimeApplyRequest({}, "controlled-backend-change", None)
        )
        assert result.status == "failed"
        assert result.diagnostics.code == "stt_runtime_apply_not_converged"
        assert components.signatures.last_llm_provider == build_llm_provider_signature(cpu)
        assert components.signatures.last_self_provider == initial_stt
        assert handle.provider.provider.backend == "cpu"
        assert managed.runtime.readiness.effective_backend == "cpu"
        assert processes[0].terminated
        identity, request, message, _slot = await _bound_request(
            handle, transports[-1], "cpu result"
        )
        transports[-1].gates[message].set()
        translated = await request
        assert (translated.utterance_id, translated.text) == (identity, message)

        plan = components.runtime.build_plan(initial, force_rebuild_llm=False)
        assert plan.should_rebuild_llm
        result = await ProviderRuntimeApplyAdapter(components.runtime, initial, plan).apply_runtime(
            RuntimeApplyRequest({}, "controlled-return-to-gpu", None)
        )
        assert result.status == "failed"
        assert result.diagnostics.code == "stt_runtime_apply_not_converged"
        assert components.signatures.last_llm_provider == build_llm_provider_signature(initial)
        assert components.signatures.last_self_provider == initial_stt
        assert handle.provider.provider.backend == "gpu"
        assert managed.runtime.readiness.effective_backend == "vulkan"
        assert len(commands) == 3
        assert processes[1].terminated
        transport = transports[-1]
        first_id, first, first_message, first_slot = await _bound_request(handle, transport, "self")
        second_id, second, second_message, second_slot = await _bound_request(
            handle, transport, "peer", reverse=True
        )
        assert first_slot != second_slot
        assert transport.peak_active == 2
        transport.gates[second_message].set()
        transport.gates[first_message].set()
        first_result, second_result = await asyncio.gather(first, second)
        assert (first_result.utterance_id, first_result.text) == (first_id, first_message)
        assert (second_result.utterance_id, second_result.text) == (second_id, second_message)
        assert len(commands) == 3
        assert not processes[-1].terminated
    finally:
        await handle.replace_provider(None, start=False)
        await managed.close()


@pytest.mark.asyncio
async def test_failed_llm_replacement_is_retryable_without_committing_failed_profile(
    tmp_path,
    monkeypatch,
):
    from puripuly_heart.app.ports.runtime_apply import RuntimeApplyRequest
    from puripuly_heart.app.services.provider.provider_runtime_apply import (
        ProviderRuntimeApplyAdapter,
    )

    components, managed, handle, _commands, _processes, transports = await _bound_components(
        tmp_path, monkeypatch
    )
    initial = components.effects.settings.canonical
    cpu = _connection(initial, "cpu")
    factory = wiring_provider_runtime.create_translation_backend
    components.signatures.last_self_provider = ("previous-stt-binding",)

    def fail_factory(**_kwargs):
        raise RuntimeError("controlled LLM construction failure")

    monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", fail_factory)
    try:
        plan = components.runtime.build_plan(cpu, force_rebuild_llm=False)
        result = await ProviderRuntimeApplyAdapter(components.runtime, cpu, plan).apply_runtime(
            RuntimeApplyRequest({}, "controlled-failed-replacement", None)
        )
        assert result.status == "failed"
        assert result.diagnostics.code == "provider_runtime_apply_unavailable"
        assert handle.provider is None
        assert components.signatures.last_llm_provider == ()
        assert (
            components.signatures.last_self_provider
            == components.runtime.self_signature_builder(cpu)
        )
        assert components.signatures.last_self_provider != ("previous-stt-binding",)
        retry = components.runtime.build_plan(cpu, force_rebuild_llm=False)
        assert retry.should_rebuild_llm
        monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", factory)
        result = await ProviderRuntimeApplyAdapter(components.runtime, cpu, retry).apply_runtime(
            RuntimeApplyRequest({}, "controlled-successful-retry", None)
        )
        assert result.status == "applied"
        assert handle.provider.provider.backend == "cpu"
        assert components.signatures.last_llm_provider != ()
        assert not components.runtime.build_plan(cpu, force_rebuild_llm=False).should_rebuild_llm
        identity, request, message, _slot = await _bound_request(
            handle, transports[-1], "retry result"
        )
        transports[-1].gates[message].set()
        translated = await request
        assert (translated.utterance_id, translated.text) == (identity, message)
    finally:
        await handle.replace_provider(None, start=False)
        await managed.close()
