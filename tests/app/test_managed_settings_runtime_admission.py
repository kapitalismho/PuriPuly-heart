from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from puripuly_heart.app.services.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)
from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEExchangeResult

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.adapters.sync_secret_store import SyncSecretStoreAdapter
from puripuly_heart.app.ports.runtime_apply import RuntimeApplyRequest
from puripuly_heart.app.ports.settings_repository import SettingsCommitRequest
from puripuly_heart.app.ports.settings_view import (
    OpenRouterPkceTarget,
    ProviderApplyIntent,
    TranslationSelectionEdit,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    CanonicalSettingsPatchRepository,
    SettingsOwner,
)
from puripuly_heart.app.services.capture.self_capture_application import (
    SelfCaptureApplicationOwner,
    SelfCaptureApplicationSettings,
)
from puripuly_heart.app.services.gpu_provider_recovery_application import (
    GpuProviderRecoveryApplicationRequest,
)
from puripuly_heart.app.services.openrouter_pkce_flow import OpenRouterPkceApplicationOwner
from puripuly_heart.app.services.peer_application import PeerApplicationOwner, PeerApplicationState
from puripuly_heart.app.services.provider.provider_runtime_apply import (
    ProviderRuntimeApplyAdapter,
    ProviderRuntimeApplyPlan,
)
from puripuly_heart.app.services.provider.provider_settings import (
    ProviderApplicationOwner,
    ProviderSettingsOwner,
    provider_verification_context,
)
from puripuly_heart.app.services.settings.settings_application import (
    settings_view_surface_snapshots,
)
from puripuly_heart.app.services.settings.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.app.wiring import (
    wiring_capture_runtime,
    wiring_local_asr_provider_runtime,
    wiring_provider_runtime,
)
from puripuly_heart.app.wiring.wiring_capture_runtime import CaptureOwnerFactory
from puripuly_heart.app.wiring.wiring_composition import (
    create_gpu_provider_recovery_application_owner,
)
from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import SharedSTTProviderFactory
from puripuly_heart.app.wiring.wiring_managed_auth_factory import (
    CanonicalManagedIdentityStateAdapter,
)
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_self_capture_session_config_from_vnext,
)
from puripuly_heart.config.provider_values import OpenRouterSelectionAlias, STTProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.peer_capture import (
    PeerCaptureResolvedTarget,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.core.runtime.local_asr_provider_runtime import LocalASRProviderRuntimeOwner
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend
from tests.core.runtime.test_local_asr_provider_runtime import (
    GPU_DEVICE,
    FakeGpuRuntime,
    FakeProvisioningPort,
)
from tests.core.runtime.test_peer_capture_session import FakeSource, FakeVadSink

REFERENCE = "admission-reference-258"
LATER_REFERENCE = "later-admission-reference-258"


async def _nothing(*args, **kwargs):
    return None


class LlmResource:
    def __init__(self, model: str, concurrency: int):
        self.model = model
        self.concurrency = concurrency
        self.closing = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.closed = False

    async def close(self):
        self.closing.set()
        await self.release.wait()
        self.closed = True


class ControlledGpu(FakeGpuRuntime):
    def __init__(self, sink, entered: asyncio.Event, release: asyncio.Event):
        super().__init__(sink)
        self.entered = entered
        self.release = release

    async def discover_devices(self):
        self.entered.set()
        await self.release.wait()
        return await super().discover_devices()

    async def activate_channel(self, channel, *, operation_id=None, request_id=None, **kwargs):
        return await super().activate_channel(channel, **kwargs)


@pytest.fixture
async def composed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    home.mkdir()
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
        monkeypatch.setenv(key, str(home))
    initial = AppSettingsVNext()
    initial = replace(
        initial,
        intent=replace(
            initial.intent,
            translation=replace(initial.intent.translation, model="local_llm", connection="ollama"),
            stt=replace(initial.intent.stt, provider=STTProviderName.ROLLING_FREE.value),
            peer_stt=replace(initial.intent.peer_stt, provider=STTProviderName.ROLLING_FREE.value),
        ),
    )
    initial = replace(
        initial,
        state=replace(
            initial.state,
            peer_translation=replace(initial.state.peer_translation, eula_accepted=True),
        ),
    )
    path = tmp_path / "settings.json"
    persistence = SettingsVNextCanonicalPersistenceAdapter()
    persistence.persist(path, initial)
    owner = SettingsOwner(path=path, persistence=persistence)
    owner.start()
    owner.remember_projection(owner.require_committed())
    secrets = InMemorySecretStore()
    old_llm = LlmResource("initial", initial.intent.translation.concurrency_limit)
    llm = ProviderRuntimeHandle(name="llm", provider=old_llm)
    created_llm = []
    backends = []
    native_instances = []
    native_entered, native_release = asyncio.Event(), asyncio.Event()
    native_release.set()
    recovery_entered, recovery_release = asyncio.Event(), asyncio.Event()
    recovery_release.set()

    def llm_factory(**kwargs):
        resource = LlmResource(kwargs["translation_model"], kwargs["concurrency_limit"])
        created_llm.append(resource)
        return resource

    original_backend_factory = (
        wiring_local_asr_provider_runtime.create_stt_backend_from_resolved_config
    )

    def backend_factory(config, **kwargs):
        backend = original_backend_factory(config, **kwargs)
        backends.append(backend)
        return backend

    def gpu_factory(sink):
        native = ControlledGpu(sink, native_entered, native_release)
        native_instances.append(native)
        return native

    monkeypatch.setattr(
        wiring_provider_runtime, "create_secret_store", lambda *args, **kwargs: secrets
    )
    monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", llm_factory)
    monkeypatch.setattr(
        wiring_local_asr_provider_runtime,
        "create_stt_backend_from_resolved_config",
        backend_factory,
    )
    runtime = LocalASRProviderRuntimeOwner(
        provider_factory=SharedSTTProviderFactory(
            secrets=secrets,
            clock=SystemClock(),
            reset_deadline_s=1,
            gpu_model_path=tmp_path / "synthetic-model",
        ),
        gpu_runtime_factory=gpu_factory,
        provisioning=FakeProvisioningPort(),
        self_event_handler=_nothing,
        peer_event_handler=_nothing,
        retired_event_handler=_nothing,
        self_exception_handler=_nothing,
        peer_exception_handler=_nothing,
    )

    async def ready(*args):
        return True

    async def resolve_target(target):
        return PeerCaptureTargetResolution(
            PeerCaptureTargetStatus.RESOLVED, PeerCaptureResolvedTarget(target)
        )

    async def audio_loop(**kwargs):
        await asyncio.Event().wait()

    sources = []

    def source_factory(*args):
        source = FakeSource()
        sources.append(source)
        return source

    monkeypatch.setattr(
        wiring_capture_runtime,
        "create_peer_capture_target_resolver_adapter",
        lambda: SimpleNamespace(resolve=resolve_target),
    )
    monkeypatch.setattr(
        wiring_capture_runtime,
        "create_peer_capture_source_adapter",
        lambda **kwargs: source_factory,
    )
    monkeypatch.setattr(
        wiring_capture_runtime, "create_peer_capture_vad_adapter", lambda: lambda config: object()
    )
    monkeypatch.setattr(
        wiring_capture_runtime,
        "create_peer_capture_audio_loop_adapter",
        lambda **kwargs: audio_loop,
    )
    monkeypatch.setattr(
        wiring_capture_runtime,
        "create_peer_capture_vad_sink_adapter",
        lambda **kwargs: FakeVadSink(),
    )
    factory = CaptureOwnerFactory(
        canonical_provider=owner.require_committed,
        self_admission=SimpleNamespace(admit=_nothing),
        ensure_peer_local_ready=ready,
        clock=SystemClock(),
        log_basic=lambda *args: None,
        log_diagnostic=lambda *args: None,
        source_wrapper=lambda source: source,
        self_state_sink=lambda snapshot: None,
        self_diagnostic_sink=lambda event: None,
        peer_state_sink=lambda snapshot: None,
        peer_diagnostic_sink=lambda event: None,
        local_asr_diagnostic_sink=lambda event: None,
    )
    channel_reset = SimpleNamespace(reset_provider_channel=_nothing)
    self_capture = factory.compose_self(None, runtime, channel_reset, None)
    peer_capture = factory.compose_peer(SimpleNamespace(), runtime, channel_reset)

    def peer_state(settings):
        return PeerApplicationState(
            settings_available=True,
            peer_intent_enabled=owner.peer_translation_enabled(),
            eula_accepted=settings.state.peer_translation.eula_accepted,
            overlay_intent_enabled=owner.overlay_enabled(),
            peer_provider_id=settings.intent.peer_stt.provider,
            runtime_available=True,
            peer_provider_available=runtime.snapshot.channel_for("peer").provider_id is not None,
            overlay_state="connected",
        )

    peer = PeerApplicationOwner(
        state_provider=lambda: peer_state(owner.require_committed()),
        config_factory=lambda: build_peer_capture_session_config_from_vnext(
            owner.require_committed()
        ),
        peer_intent_sink=owner.set_peer_translation_enabled,
        overlay_intent_sink=owner.set_overlay_enabled,
        persist_manual_fallback=lambda: True,
        ensure_local_ready=_nothing,
        clear_cpu_pending=lambda: None,
        clear_gpu_pending=lambda: None,
        clear_switched_pending=lambda: None,
        sync_local_notice=lambda: None,
        presentation_changed=lambda: None,
        begin_overlay_start=_nothing,
        effective_sink=lambda *args: None,
        disclosure_sink=lambda: None,
        superseded_sink=lambda: None,
        log_basic=lambda message: None,
        log_diagnostic=lambda message: None,
        log_failure=lambda message: None,
    )
    peer.bind_runtime(peer_capture)
    self_application = SelfCaptureApplicationOwner(
        settings_provider=lambda: SelfCaptureApplicationSettings(
            build_self_capture_session_config_from_vnext(owner.require_committed()),
            owner.require_committed().intent.stt.provider,
            None,
        ),
        runtime_available=lambda: True,
        capture_owner=lambda: self_capture,
        capture_owner_if_created=lambda: self_capture,
        persist_manual_fallback=lambda: True,
        reset_local_pending=lambda: None,
        clear_gpu_pending=lambda: None,
        overlay_state_provider=lambda: "off",
        mark_promo_eligible=lambda: None,
        dashboard_enabled_sink=lambda value: None,
        dashboard_needs_key_sink=lambda value: None,
        dashboard_needs_key=lambda available: not available,
        state_sink=lambda snapshot: None,
        sync_effective_flags=lambda: None,
        sync_local_notice=lambda: None,
        log_basic=lambda message: None,
        log_diagnostic=lambda *args: None,
    )

    async def refresh_peer(settings):
        await peer.refresh_runtime_with_config(
            build_peer_capture_session_config_from_vnext(settings), state=peer_state(settings)
        )

    async def replace_self(settings, smooth):
        await self_application.replace_provider_with_config(
            build_self_capture_session_config_from_vnext(settings), smooth_local=smooth
        )

    async def quiesce(channels):
        if "self" in channels:
            await self_capture.suspend_provider_consumer()
        if "peer" in channels:
            await peer_capture.suspend_provider_consumer()

    recovery = create_gpu_provider_recovery_application_owner(
        runtime_provider=lambda: runtime,
        pending_provider=frozenset,
        pending_clear=lambda channels: None,
        failure_sink=lambda reason: None,
        runtime_state_sink=lambda snapshot: None,
        quiesce=quiesce,
        self_owner_factory=lambda: self_capture,
        peer_owner_provider=lambda: peer_capture,
        self_state_sink=lambda snapshot: None,
        diagnostic_sink=lambda event: None,
    )

    def recovery_request(settings, plan):
        self_config = build_self_capture_session_config_from_vnext(settings)
        peer_config = build_peer_capture_session_config_from_vnext(settings)

        async def refresh_self_bound():
            recovery_entered.set()
            await recovery_release.wait()
            await components.effects.refresh_self_stt(settings)

        async def refresh_peer_bound():
            recovery_entered.set()
            await recovery_release.wait()
            await components.effects.refresh_peer(settings)

        return GpuProviderRecoveryApplicationRequest(
            device_id=settings.intent.stt.gpu_device_id,
            reason="settings_restart",
            self_gpu_selected=settings.intent.stt.provider == STTProviderName.LOCAL_QWEN_GPU.value,
            peer_gpu_selected=settings.intent.peer_stt.provider
            == STTProviderName.LOCAL_QWEN_GPU.value,
            self_desired=self_capture.snapshot.desired_active,
            peer_enabled=peer.desired_active(peer_state(settings)),
            self_config_factory=lambda: self_config,
            peer_config_factory=lambda: peer_config,
            self_request_factory=lambda: factory.self_provider_request(self_config, True),
            peer_request_factory=lambda config: factory.peer_provider_request(config, True),
            ensure_self_switch=lambda: replace_self(settings, True),
            refresh_self=refresh_self_bound,
            refresh_peer=refresh_peer_bound,
            should_refresh_self=plan.should_refresh_self_stt,
            should_refresh_peer=plan.should_refresh_peer,
        )

    async def recover_gpu(settings, plan):
        await recovery.recover(lambda: recovery_request(settings, plan))

    configuration = TranslationRuntimeConfigurationOwner(TranslationRuntimeConfig())
    components = wiring_provider_runtime.compose_provider_runtime(
        config_path=path,
        settings=owner,
        llm_runtime_provider=lambda: llm,
        http_extensions=SimpleNamespace(),
        local_asr_runtime_provider=lambda: runtime,
        translation_runtime_configuration_provider=lambda: configuration,
        self_capture_provider=lambda: self_capture,
        self_capture_owner=lambda: self_capture,
        peer=lambda: peer,
        peer_state=peer_state,
        canonical_settings=lambda settings: owner.normalize(settings),
        clear_local_pending=lambda: None,
        sync_local_notice=lambda: None,
        managed_pending_sink=lambda pending: None,
        managed_pending_provider=lambda: False,
        dashboard_managed_pending_sink=lambda pending: None,
        sync_effective_flags=lambda: None,
        refresh_overlay=lambda: None,
        refresh_peer_runtime=refresh_peer,
        replace_self_stt=replace_self,
        self_state_sink=lambda snapshot: None,
        self_availability=self_application.project_availability,
        gpu_recovery=recover_gpu,
        managed_release=lambda: SimpleNamespace(rebuild=_nothing, service=None),
        managed_delegate_ready=lambda: None,
        runtime_logging=SimpleNamespace(),
        translation_needs_key_sink=lambda value: None,
        usage_refresh=_nothing,
        failure_sink=lambda message: None,
        success_sink=lambda message: None,
        additional_signature_sink=lambda settings: None,
    )
    common, classified = [], []
    original_common = components.effects.apply_common
    original_state = components.effects.state

    def observe_common(settings):
        common.append(settings)
        original_common(settings)

    def observe_state(settings):
        classified.append(settings)
        return original_state(settings)

    components.runtime.common_effect = observe_common
    components.runtime.state_provider = observe_state
    results = SettingsTransactionResultOwner()
    application = ProviderApplicationOwner(
        settings=owner,
        runtime=components.runtime,
        merge_settings=copy.deepcopy,
        preserve_before_replace=_nothing,
        sync_ui=lambda: None,
        order24_patch_provider=lambda settings: None,
        apply_order24=_nothing,
        remember_order22=lambda settings: None,
        mutation_service_provider=lambda: None,
        results=results,
        sync_memory=lambda settings: None,
        capture_runtime_signatures=components.capture_signatures_before_canonical_mutation,
        sync_stt_signatures=components.sync_stt_signatures,
        consume_superseded_settings=components.signatures.consume_superseded,
        active_local_asr_change=lambda *args: False,
        compensate_local_asr=_nothing,
        llm_retry_pending=lambda: components.signatures.last_llm_provider == (),
        mark_llm_retry=components.signatures.mark_llm_retry,
    )
    context = SimpleNamespace(
        owner=owner,
        initial=owner.require_committed(),
        application=application,
        components=components,
        initial_canonical=owner.require_canonical(),
        operations=[],
        release_gates=[old_llm.release, native_release, recovery_release],
        runtime=runtime,
        llm=llm,
        old_llm=old_llm,
        created_llm=created_llm,
        configuration=configuration,
        self_capture=self_capture,
        peer_capture=peer_capture,
        self_application=self_application,
        peer=peer,
        peer_state=peer_state,
        factory=factory,
        secrets=secrets,
        common=common,
        classified=classified,
        backends=backends,
        native_instances=native_instances,
        sources=sources,
        native_entered=native_entered,
        native_release=native_release,
        recovery_entered=recovery_entered,
        recovery_release=recovery_release,
        recovery_request=recovery_request,
        recovery=recovery,
        results=results,
    )
    try:
        yield context
    finally:
        for gate in context.release_gates:
            gate.set()
        await asyncio.wait_for(asyncio.gather(*context.operations, return_exceptions=True), 10)
        await self_capture.close()
        await peer.close()
        await llm.close()
        await runtime.close()


def _assert_reference(context, expected):
    fresh = (
        SettingsVNextCanonicalPersistenceAdapter()
        .load_active(context.owner.path)
        .canonical_settings
    )
    for settings in (
        context.owner.require_canonical(),
        context.owner.require_committed(),
        context.owner.projection_snapshot,
        fresh,
    ):
        assert settings.state.managed_connection.active_managed_credential_ref == expected
    assert context.owner.mutation_depth == 0
    assert not context.owner.rollback_pending


async def _managed(context, reference):
    state = CanonicalManagedIdentityStateAdapter(context.owner)
    state.active_managed_credential_ref = reference
    await run_owned_thread_call(state.persist)


async def _locale(context):
    current = context.owner.require_committed()
    repository = context.owner.create_canonical_patch_repository(
        base_settings=current, committed_settings=current
    )
    result = await repository.save(
        SettingsCommitRequest(
            values={"intent.ui.locale": "ja"}, expected_revision=None, reason="later_locale"
        )
    )
    assert result.succeeded
    fresh = (
        SettingsVNextCanonicalPersistenceAdapter()
        .load_active(context.owner.path)
        .canonical_settings
    )
    assert fresh.intent.ui.locale == "ja"
    _assert_reference(context, LATER_REFERENCE)


def _gate_admission(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    original = CanonicalSettingsPatchRepository.save
    pending = True

    async def save(repository, request):
        nonlocal pending
        if pending:
            pending = False
            entered.set()
            await release.wait()
        return await original(repository, request)

    monkeypatch.setattr(CanonicalSettingsPatchRepository, "save", save)
    return entered, release


async def _pkce(context):
    owner = context.owner
    provider_settings = ProviderSettingsOwner(
        settings=owner,
        binding=ProviderVerificationBindingOwner(
            context_provider=lambda provider: provider_verification_context(
                owner.require_committed(), provider, low_latency=True
            )
        ),
        secret_store_factory=lambda settings: SyncSecretStoreAdapter(context.secrets),
        active_secret_provider=lambda settings, key: context.secrets.get(key),
    )

    async def flow():
        return OpenRouterPKCEExchangeResult(api_key="synthetic-pkce-key", user_id="synthetic-user")

    application = OpenRouterPkceApplicationOwner(
        flow=SimpleNamespace(run_flow=flow),
        verifier=SimpleNamespace(verify_api_key=lambda *args: asyncio.sleep(0, result=True)),
        settings=owner,
        provider_settings=provider_settings,
        provider_runtime=context.components.runtime,
        secret_store_factory=lambda settings: SyncSecretStoreAdapter(context.secrets),
        failure_message_sink=lambda message: None,
        failure_diagnostics_sink=lambda *args: None,
        failure_route=lambda source: None,
        results=context.results,
    )
    provider, *_ = settings_view_surface_snapshots(owner.require_committed())
    staged = replace(
        provider.translation,
        model=TranslationModel.GEMMA4_26B_31B,
        connection=TranslationConnection.OPENROUTER,
    )
    target = OpenRouterPkceTarget(
        selection_alias=OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK,
        provider_intent=ProviderApplyIntent(
            (TranslationSelectionEdit(staged, ((staged.model, staged.connection),)),)
        ),
        system_prompt="synthetic admitted prompt",
    )
    return await application.connect(target=target, launch_source="settings")


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["translation", "stt", "pkce"])
async def test_current_committed_admission_survives_actual_runtime_and_later_locale(
    composed, monkeypatch, route
):
    context = composed
    initial = context.initial
    entered, release = _gate_admission(monkeypatch)
    context.release_gates.append(release)
    if route == "translation":
        candidate = replace(
            initial,
            intent=replace(
                initial.intent,
                translation=replace(
                    initial.intent.translation,
                    concurrency_limit=initial.intent.translation.concurrency_limit + 1,
                ),
            ),
        )
        operation = asyncio.create_task(context.application._apply_translation(candidate))
    elif route == "stt":
        candidate = replace(
            initial,
            intent=replace(
                initial.intent,
                stt=replace(initial.intent.stt, provider=STTProviderName.LOCAL_QWEN_GPU.value),
            ),
        )
        operation = asyncio.create_task(context.application._apply_stt_language_audio(candidate))
    else:
        operation = asyncio.create_task(_pkce(context))
    context.operations.append(operation)
    await asyncio.wait_for(entered.wait(), 5)
    await _managed(context, REFERENCE)
    _assert_reference(context, REFERENCE)
    if route == "stt":
        context.native_release.clear()
    else:
        context.old_llm.release.clear()
    release.set()
    waiting = context.native_entered if route == "stt" else context.old_llm.closing
    await asyncio.wait_for(waiting.wait(), 5)
    assert not operation.done()
    assert context.common[-1].state.managed_connection.active_managed_credential_ref == REFERENCE
    await _managed(context, LATER_REFERENCE)
    _assert_reference(context, LATER_REFERENCE)
    context.native_release.set()
    context.old_llm.release.set()
    assert await asyncio.wait_for(operation, 5)
    assert context.results.current.status == "settings_commit_success_runtime_applied"
    assert context.classified[-1] is context.common[-1]
    _assert_reference(context, LATER_REFERENCE)
    if route == "translation":
        expected = candidate.intent.translation.concurrency_limit
        assert context.configuration.snapshot().value.concurrency_limit == expected
        assert context.llm.provider.concurrency == expected
    elif route == "stt":
        assert (
            context.runtime.snapshot.channel_for("self").provider_id
            == STTProviderName.LOCAL_QWEN_GPU.value
        )
        assert any(isinstance(backend, LocalGpuSTTBackend) for backend in context.backends)
    else:
        assert context.configuration.snapshot().value.system_prompt == "synthetic admitted prompt"
        assert context.secrets.get("openrouter_api_key") == "synthetic-pkce-key"
    await _locale(context)


@pytest.mark.asyncio
async def test_explicit_runtime_only_gpu_restart_preserves_authority_through_consumer_completion(
    composed,
):
    context = composed
    before = context.owner.path.read_bytes()
    snapshot = context.owner.require_committed()
    requested = replace(
        snapshot,
        intent=replace(
            snapshot.intent,
            stt=replace(
                snapshot.intent.stt,
                provider=STTProviderName.LOCAL_QWEN_GPU.value,
                gpu_device_id=GPU_DEVICE.device_id,
            ),
        ),
    )
    context.recovery_release.clear()
    plan = ProviderRuntimeApplyPlan(False, False, True, coordinated_gpu_restart=True)
    operation = asyncio.create_task(
        ProviderRuntimeApplyAdapter(
            context.components.runtime, lambda: requested, plan
        ).apply_runtime(RuntimeApplyRequest({}, "explicit_runtime_only", None))
    )
    context.operations.append(operation)
    await asyncio.wait_for(context.recovery_entered.wait(), 5)
    assert not operation.done()
    assert context.runtime.current_provider("self") is None
    assert context.owner.require_canonical() is context.initial_canonical
    assert context.owner.require_committed() is snapshot
    assert context.owner.path.read_bytes() == before
    context.recovery_release.set()
    result = await asyncio.wait_for(operation, 5)
    assert result.status == "applied"
    assert context.common[-1] is requested
    assert context.classified[-1] is requested
    assert (
        context.runtime.snapshot.channel_for("self").provider_id
        == STTProviderName.LOCAL_QWEN_GPU.value
    )
    backend = next(
        backend for backend in context.backends if isinstance(backend, LocalGpuSTTBackend)
    )
    assert backend.device_id == GPU_DEVICE.device_id
    await context.runtime.current_provider("self").warmup()
    assert ("self", GPU_DEVICE.device_id) in context.native_instances[-1].activation_calls
    assert context.owner.require_canonical() is context.initial_canonical
    assert context.owner.require_committed() is snapshot
    assert context.owner.path.read_bytes() == before


@pytest.mark.asyncio
async def test_peer_factory_and_owner_keep_explicit_gpu_device_without_canonical_replacement(
    composed,
):
    context = composed
    before = context.owner.path.read_bytes()
    snapshot = context.owner.require_committed()
    requested = replace(
        snapshot,
        intent=replace(
            snapshot.intent,
            stt=replace(snapshot.intent.stt, gpu_device_id=GPU_DEVICE.device_id),
            peer_stt=replace(
                snapshot.intent.peer_stt, provider=STTProviderName.LOCAL_QWEN_GPU.value
            ),
        ),
    )
    context.owner.set_peer_translation_enabled(True)
    context.owner.set_overlay_enabled(True)
    plan = ProviderRuntimeApplyPlan(False, True, False)
    result = await ProviderRuntimeApplyAdapter(
        context.components.runtime, lambda: requested, plan
    ).apply_runtime(RuntimeApplyRequest({}, "explicit_peer_override", None))
    assert result.status == "applied"
    assert (
        context.runtime.snapshot.channel_for("peer").provider_id
        == STTProviderName.LOCAL_QWEN_GPU.value
    )
    backend = next(
        backend
        for backend in context.backends
        if isinstance(backend, LocalGpuSTTBackend) and backend.channel == "peer"
    )
    assert backend.device_id == GPU_DEVICE.device_id
    assert ("peer", GPU_DEVICE.device_id) in context.native_instances[-1].activation_calls
    assert context.owner.require_canonical() is context.initial_canonical
    assert context.owner.require_committed() is snapshot
    assert context.owner.path.read_bytes() == before
    previous = context.runtime.current_provider("peer")
    previous_source = context.sources[-1]
    context.recovery_entered.clear()
    context.recovery_release.clear()
    restarting = asyncio.create_task(
        ProviderRuntimeApplyAdapter(
            context.components.runtime,
            lambda: requested,
            ProviderRuntimeApplyPlan(False, True, False, coordinated_gpu_restart=True),
        ).apply_runtime(RuntimeApplyRequest({}, "explicit_peer_restart", None))
    )
    context.operations.append(restarting)
    await asyncio.wait_for(context.recovery_entered.wait(), 5)
    assert not restarting.done()
    assert (
        context.runtime.snapshot.channel_for("peer").provider_id
        == STTProviderName.LOCAL_QWEN_GPU.value
    )
    assert not context.peer_capture.snapshot.effective_active
    assert previous_source.close_calls == 1
    assert context.owner.path.read_bytes() == before
    context.recovery_release.set()
    restarted = await asyncio.wait_for(restarting, 5)
    assert restarted.status == "applied"
    assert context.runtime.current_provider("peer") is not previous
    assert not previous.is_live
    assert context.peer_capture.snapshot.effective_active
    assert ("peer", GPU_DEVICE.device_id) in context.native_instances[-1].activation_calls
    assert context.owner.require_canonical() is context.initial_canonical
    assert context.owner.require_committed() is snapshot
    assert context.owner.path.read_bytes() == before


@pytest.mark.asyncio
async def test_failed_provider_commit_does_not_apply_runtime_and_can_retry(composed, monkeypatch):
    context = composed
    initial = context.owner.require_committed()
    candidate = replace(
        initial,
        intent=replace(
            initial.intent,
            translation=replace(
                initial.intent.translation,
                concurrency_limit=initial.intent.translation.concurrency_limit + 1,
            ),
        ),
    )
    entered, release = _gate_admission(monkeypatch)
    context.release_gates.append(release)
    operation = asyncio.create_task(context.application._apply_translation(candidate))
    context.operations.append(operation)
    await asyncio.wait_for(entered.wait(), 5)
    await _managed(context, REFERENCE)
    persistence = context.owner.persistence
    original = persistence.persist

    def fail_provider(path, settings):
        if (
            settings.intent.translation.concurrency_limit
            == candidate.intent.translation.concurrency_limit
        ):
            raise OSError("synthetic provider commit rejection")
        original(path, settings)

    monkeypatch.setattr(persistence, "persist", fail_provider)
    release.set()
    assert await asyncio.wait_for(operation, 5)
    assert context.results.current.status == "settings_commit_failed"
    assert context.common == []
    assert context.created_llm == []
    assert context.llm.provider is context.old_llm
    _assert_reference(context, REFERENCE)
    monkeypatch.setattr(persistence, "persist", original)
    current = context.owner.require_committed()
    retried = replace(
        current, intent=replace(current.intent, translation=candidate.intent.translation)
    )
    assert await context.application._apply_translation(retried)
    assert context.results.current.status == "settings_commit_success_runtime_applied"
    _assert_reference(context, REFERENCE)
    await _managed(context, LATER_REFERENCE)
    await _locale(context)
