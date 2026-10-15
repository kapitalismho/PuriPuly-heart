from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
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
    components.sync_stt_signatures(components.effects.settings.canonical)
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
        assert result.diagnostics.code == "provider_runtime_apply_exception"
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


@asynccontextmanager
async def _composed_signature_application(
    tmp_path,
    monkeypatch,
    *,
    fail_initial=False,
    initialization="startup",
    advance_intent=False,
):
    import io
    import json
    import logging
    import socket
    import subprocess

    from puripuly_heart.app.wiring import wiring_local_asr_provider_runtime, wiring_runtime_pipeline
    from puripuly_heart.composition import application_runtime, headless_application
    from puripuly_heart.composition.local_asr_production_evidence import (
        _ApplicationLocalASRProductionEvidence,
    )
    from puripuly_heart.config.settings_vnext.schema import SecretsIntent
    from puripuly_heart.config.settings_vnext.serialization import to_dict
    from puripuly_heart.core.local_translation import runtime_profile
    from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks
    from tests.app.test_stt_provider_apply_vertical import _TransportBackend
    from tests.core.local_translation.test_gemma_runtime import _runtime

    for name in (
        "HOME",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "HF_HOME",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "TEMP",
        "TMP",
    ):
        root = tmp_path / name
        root.mkdir()
        monkeypatch.setenv(name, str(root))
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-signature-test")
    monkeypatch.setattr(
        runtime_profile, "default_threads_profile_path", lambda: tmp_path / "threads.json"
    )
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    native_root = tmp_path / "controlled-native"
    native_root.mkdir()
    native, commands, processes, transports, _calls, _logs = _runtime(native_root)
    monkeypatch.setattr(
        application_runtime, "create_managed_gemma_runtime", lambda **_kwargs: native
    )
    stt_backends = []

    def create_stt(*_args, **_kwargs):
        backend = _TransportBackend()
        stt_backends.append(backend)
        return backend

    monkeypatch.setattr(
        wiring_local_asr_provider_runtime, "create_stt_backend_from_resolved_config", create_stt
    )
    factory = wiring_provider_runtime.create_translation_backend
    construction = SimpleNamespace(fail_initial=fail_initial, fail_replacement=False, calls=[])

    def initial_backend(**kwargs):
        construction.calls.append("initial")
        if construction.fail_initial:
            raise RuntimeError("controlled initial LLM failure")
        provider = factory(**kwargs)
        if advance_intent:
            owner = app._settings.application.settings
            owner.canonical = _connection(owner.canonical, "cpu")
        return provider

    def replacement_backend(**kwargs):
        construction.calls.append("replacement")
        if construction.fail_replacement:
            raise RuntimeError("controlled replacement LLM failure")
        return factory(**kwargs)

    monkeypatch.setattr(wiring_runtime_pipeline, "create_translation_backend", initial_backend)
    monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", replacement_backend)

    def reject_process(*_args, **_kwargs):
        raise AssertionError("controlled composition must not launch external processes")

    monkeypatch.setattr(subprocess, "Popen", reject_process)
    connect = socket.socket.connect

    def local_connect(connection, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("controlled composition forbids external connections")
        return connect(connection, address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)
    sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sink.bind(("127.0.0.1", 0))
    settings = _managed_settings()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            stt=replace(settings.intent.stt, provider="soniox", custom_terms={}),
            peer_stt=replace(settings.intent.peer_stt, provider="soniox"),
            secrets=SecretsIntent(
                backend="encrypted_file", encrypted_file_path=str(tmp_path / "secrets.json")
            ),
            telemetry=replace(settings.intent.telemetry, enabled=False),
            integrated_context=replace(settings.intent.integrated_context, enabled=False),
            osc=replace(
                settings.intent.osc,
                connection_mode="off",
                host="127.0.0.1",
                port=sink.getsockname()[1],
                send_port=sink.getsockname()[1],
                activation_notice_enabled=False,
            ),
        ),
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    logger = logging.getLogger()
    monkeypatch.setattr(logger, "handlers", list(logger.handlers))
    monkeypatch.setattr(logger, "filters", list(logger.filters))
    monkeypatch.setattr(logger, "level", logger.level)
    for handler in logger.handlers:
        monkeypatch.setattr(handler, "filters", list(handler.filters))
    log_path = tmp_path / "host.log"
    sinks = RuntimeLoggingSinks(
        logging.StreamHandler(io.StringIO()),
        logging.FileHandler(log_path, encoding="utf-8"),
        log_path,
    )
    captured = []
    compose_runtime = headless_application.compose_application_runtime

    def capture_runtime(**kwargs):
        return compose_runtime(**kwargs, local_asr_evidence_sink=captured.append)

    monkeypatch.setattr(headless_application, "compose_application_runtime", capture_runtime)
    app = headless_application.compose_headless_application(path, runtime_logging_sinks=sinks)
    try:
        if initialization == "startup":
            await app.start()
        else:
            evidence = _ApplicationLocalASRProductionEvidence(captured[0], app)
            await evidence.initialize(settings)
            await evidence.start_runtime()
        control = app.control()
        control.bind_instance("isolated-signature-writer-regression")
        channel = control.pipeline.local_asr_runtime.snapshot.channel_for("self")
        assert channel.provider_id == "soniox"
        assert channel.has_resources
        assert channel.phase in {"ready", "dormant", "running"}
        capture = control.pipeline.self_capture.snapshot
        assert capture.failure_reason is None
        assert capture.provider_status.value == "ready"
        assert not capture.desired_active
        yield SimpleNamespace(
            app=app,
            control=control,
            construction=construction,
            native=native,
            commands=commands,
            processes=processes,
            transports=transports,
            stt_backends=stt_backends,
            settings_path=path,
            signature=lambda: app._provider.provider_application.runtime.signature_cache_provider()[
                2
            ],
        )
    finally:
        await app.stop()
        await native.close()
        sink.close()
        sinks.close()
        sinks.stream_handler.close()
    assert all(process.returncode == 0 for process in processes)
    assert not native._server_tasks
    assert not native._operation_tasks
    assert all(session.closed.is_set() for backend in stt_backends for session in backend.sessions)


async def _control_operation(control, command, arguments, request_id):
    admitted = await control.submit(command, arguments, request_id=request_id)
    receipt = await control.wait(admitted["operation_id"], timeout=5)
    assert receipt["terminal"]
    return receipt


async def _completed_control_translation(control, text):
    stream = control.subscribe(topics=["translation"], include_translations=True)

    async def observe():
        async for event in stream:
            return event

    observed = asyncio.create_task(observe())
    await asyncio.sleep(0)
    try:
        receipt = await _control_operation(
            control, "text.submit", {"text": text}, "signature-consumer"
        )
        assert receipt["admission"] == "admitted"
        event = await asyncio.wait_for(observed, timeout=5)
        await asyncio.wait_for(control.pipeline.translation_turns.wait_for_idle(), timeout=5)
        assert event["utterance_id"] in [receipt["turn_id"], *receipt["child_turn_ids"]]
        assert event["text"] == "translated"
        return event
    finally:
        await stream.aclose()
        if not observed.done():
            observed.cancel()
            await asyncio.gather(observed, return_exceptions=True)


@pytest.mark.asyncio
async def test_composed_failed_replacement_survives_successful_locale_and_retries_consumer(
    tmp_path,
    monkeypatch,
):
    from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
        build_llm_provider_signature,
    )

    async with _composed_signature_application(tmp_path, monkeypatch) as fixture:
        control = fixture.control
        assert fixture.signature() == build_llm_provider_signature(
            fixture.app.compatibility_settings()
        )
        assert fixture.construction.calls == ["initial"]
        assert fixture.commands == []
        fixture.construction.fail_replacement = True
        failed = await _control_operation(
            control, "settings.apply", {"changes": {"translation.connection": "cpu"}}, "fail-llm"
        )
        assert failed["status"] == "degraded"
        assert failed["transaction"]["status"] == "settings_commit_success_runtime_degraded"
        assert failed["transaction"]["diagnostics"]["code"] == "provider_runtime_apply_exception"
        assert control.pipeline.llm_runtime.provider is None
        assert fixture.signature() == ()
        calls = tuple(fixture.construction.calls)
        locale = await _control_operation(
            control, "settings.apply", {"changes": {"locale": "ja"}}, "successful-unrelated-locale"
        )
        assert locale["status"] == "applied"
        assert locale["transaction"]["status"] == "settings_commit_success_runtime_applied"
        assert fixture.app.compatibility_settings().intent.ui.locale == "ja"
        assert fixture.signature() == ()
        assert tuple(fixture.construction.calls) == calls
        fixture.app._settings.application.runtime_effects.restore_memory(
            fixture.app.compatibility_settings()
        )
        assert fixture.signature() == ()
        fixture.construction.fail_replacement = False
        recovered = await _control_operation(
            control,
            "settings.apply",
            {"changes": {"translation.connection": "cpu"}},
            "same-profile-retry",
        )
        assert recovered["status"] == "applied"
        assert fixture.construction.calls == ["initial", "replacement", "replacement"]
        assert control.pipeline.llm_runtime.provider.provider.backend == "cpu"
        assert fixture.signature() == build_llm_provider_signature(
            fixture.app.compatibility_settings()
        )
        enabled = await _control_operation(
            control, "translation.set", {"enabled": True}, "enable-recovered-consumer"
        )
        assert enabled["status"] == "applied"
        await _completed_control_translation(control, "Synthetic retry after locale success.")
        assert fixture.native.readiness.effective_backend == "cpu"
        assert len(fixture.commands) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_initial", [False, True], ids=["attached", "unavailable"])
@pytest.mark.parametrize("initialization", ["startup", "production-evidence"])
async def test_composed_initial_llm_signature_tracks_actual_attachment(
    tmp_path,
    monkeypatch,
    fail_initial,
    initialization,
):
    from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
        build_llm_provider_signature,
    )

    async with _composed_signature_application(
        tmp_path, monkeypatch, fail_initial=fail_initial, initialization=initialization
    ) as fixture:
        control = fixture.control
        assert fixture.construction.calls == ["initial"]
        assert fixture.commands == []
        if fail_initial:
            assert control.pipeline.llm_runtime.provider is None
            assert fixture.signature() == ()
        else:
            assert control.pipeline.llm_runtime.provider is not None
            assert fixture.signature() == build_llm_provider_signature(
                fixture.app.compatibility_settings()
            )
        locale = await _control_operation(
            control, "settings.apply", {"changes": {"locale": "ja"}}, "initial-unrelated-locale"
        )
        assert locale["status"] == "applied"
        if fail_initial:
            assert fixture.signature() == ()
        applied = await _control_operation(
            control,
            "settings.apply",
            {"changes": {"translation.connection": "gpu"}},
            "initial-same-profile",
        )
        assert applied["status"] == "applied"
        assert fixture.construction.calls == (
            ["initial", "replacement"] if fail_initial else ["initial"]
        )
        assert control.pipeline.llm_runtime.provider.provider.backend == "gpu"
        assert fixture.signature() == build_llm_provider_signature(
            fixture.app.compatibility_settings()
        )
        assert (
            await _control_operation(
                control, "translation.set", {"enabled": True}, "initial-consumer-enable"
            )
        )["status"] == "applied"
        await _completed_control_translation(control, "Synthetic initial binding consumer.")
        assert fixture.native.readiness.effective_backend == "vulkan"


@pytest.mark.asyncio
async def test_composed_memory_restore_does_not_stamp_unapplied_llm_identity(tmp_path, monkeypatch):
    from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
        build_llm_provider_signature,
    )

    async with _composed_signature_application(tmp_path, monkeypatch) as fixture:
        control = fixture.control
        gpu = fixture.app.compatibility_settings()
        cpu = await _control_operation(
            control, "settings.apply", {"changes": {"translation.connection": "cpu"}}, "bound-cpu"
        )
        assert cpu["status"] == "applied"
        actual_signature = fixture.signature()
        assert actual_signature != build_llm_provider_signature(gpu)
        fixture.app._settings.application.runtime_effects.restore_memory(gpu)
        assert fixture.signature() == actual_signature
        assert control.pipeline.llm_runtime.provider.provider.backend == "cpu"
        locale = await _control_operation(
            control, "settings.apply", {"changes": {"locale": "ja"}}, "restore-unrelated-locale"
        )
        assert locale["status"] == "applied"
        assert fixture.signature() == actual_signature
        assert control.pipeline.llm_runtime.provider.provider.backend == "cpu"
        assert fixture.construction.calls == ["initial", "replacement"]
        applied = await _control_operation(
            control,
            "settings.apply",
            {"changes": {"translation.connection": "gpu"}},
            "restore-actual-gpu",
        )
        assert applied["status"] == "applied"
        assert fixture.construction.calls == ["initial", "replacement", "replacement"]
        assert control.pipeline.llm_runtime.provider.provider.backend == "gpu"
        assert fixture.signature() == build_llm_provider_signature(gpu)
        assert (
            await _control_operation(
                control, "translation.set", {"enabled": True}, "restored-consumer-enable"
            )
        )["status"] == "applied"
        await _completed_control_translation(control, "Synthetic restored actual GPU binding.")


@pytest.mark.asyncio
@pytest.mark.parametrize("initialization", ["startup", "production-evidence"])
async def test_composed_initial_binding_retains_launch_snapshot_when_intent_advances(
    tmp_path, monkeypatch, initialization
):
    from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
        build_llm_provider_signature,
    )

    async with _composed_signature_application(
        tmp_path, monkeypatch, initialization=initialization, advance_intent=True
    ) as fixture:
        current = fixture.app.compatibility_settings()
        launch_settings = _connection(current, "gpu")
        assert current.intent.translation.connection == "cpu"
        assert fixture.control.pipeline.llm_runtime.provider.provider.backend == "gpu"
        assert fixture.signature() == build_llm_provider_signature(launch_settings)
        assert fixture.signature() != build_llm_provider_signature(current)
        assert fixture.commands == []
        locale = await _control_operation(
            fixture.control, "settings.apply", {"changes": {"locale": "ja"}}, "advanced-locale"
        )
        assert locale["status"] == "applied"
        assert fixture.signature() == build_llm_provider_signature(launch_settings)
        assert fixture.construction.calls == ["initial"]
        applied = await _control_operation(
            fixture.control,
            "settings.apply",
            {"changes": {"translation.connection": "cpu"}},
            "apply-advanced-intent",
        )
        assert applied["status"] == "applied"
        assert fixture.construction.calls == ["initial", "replacement"]
        assert fixture.control.pipeline.llm_runtime.provider.provider.backend == "cpu"
        assert fixture.signature() == build_llm_provider_signature(current)
        assert (
            await _control_operation(
                fixture.control, "translation.set", {"enabled": True}, "advanced-consumer-enable"
            )
        )["status"] == "applied"
        await _completed_control_translation(
            fixture.control, "Synthetic completed advanced snapshot consumer."
        )
        assert fixture.native.readiness.effective_backend == "cpu"
