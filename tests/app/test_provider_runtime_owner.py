from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from puripuly_heart.app.services.provider_runtime_apply import (
    LlmProviderRebuildContext,
    LlmProviderRebuildOwner,
    ProviderRuntimeApplyAdapter,
    ProviderRuntimeApplyPlan,
    ProviderRuntimeOwner,
    ProviderRuntimeState,
)

from puripuly_heart.app.ports.runtime_apply import RuntimeApplyRequest
from puripuly_heart.app.services.translation_enable import (
    ManagedTranslationPreparation,
    TranslationEnableOwner,
    TranslationEnableState,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle


def _runtime_owner(
    *,
    events: list[str],
    state: ProviderRuntimeState,
    current: object | None = None,
    cache: tuple[object | None, object | None, object | None] = (None, None, None),
    convergence=None,
    peer_convergence=None,
) -> ProviderRuntimeOwner:
    async def record(name: str) -> None:
        events.append(name)

    async def recover(_settings: object, _plan: ProviderRuntimeApplyPlan) -> None:
        events.append("gpu")

    return ProviderRuntimeOwner(
        state_provider=lambda _settings: state,
        common_effect=lambda _settings: events.append("common"),
        rebuild_llm=lambda: record("llm"),
        recover_gpu=recover,
        refresh_peer=lambda: record("peer"),
        refresh_self_stt=lambda: record("self"),
        signature_sink=lambda _settings: events.append("signatures"),
        llm_retry_sink=lambda: events.append("retry"),
        current_settings_provider=lambda: current,
        signature_cache_provider=lambda: cache,
        self_signature_builder=lambda settings: ("self", settings),
        peer_signature_builder=lambda settings, canonical: (
            "peer",
            settings,
            canonical,
        ),
        llm_signature_builder=lambda settings: ("llm", settings),
        gpu_restart_decision=lambda _current, _next: True,
        self_runtime_convergence=convergence or (lambda _settings: None),
        peer_runtime_convergence=peer_convergence or (lambda _settings: None),
    )


def test_provider_runtime_owner_builds_plan_from_owned_signature_state() -> None:
    owner = _runtime_owner(
        events=[],
        state=ProviderRuntimeState(False, False, False, False, False, False),
        current="current",
    )

    plan = owner.build_plan(
        "next",
        force_rebuild_llm=False,
        canonical_settings="canonical",
    )

    assert plan == ProviderRuntimeApplyPlan(
        should_rebuild_llm=True,
        should_refresh_peer=False,
        should_refresh_self_stt=True,
        coordinated_gpu_restart=True,
    )


def test_provider_runtime_owner_forces_self_refresh_when_live_runtime_is_stale() -> None:
    owner = _runtime_owner(
        events=[],
        state=ProviderRuntimeState(True, True, True, True, True, False),
        current="next",
        cache=(("self", "next"), ("peer",), ("llm", "next")),
        convergence=lambda _settings: False,
    )

    plan = owner.build_plan("next", force_rebuild_llm=False)

    assert plan.should_refresh_self_stt is True


def test_provider_runtime_owner_does_not_prepare_idle_peer_for_uncached_signature() -> None:
    owner = _runtime_owner(
        events=[],
        state=ProviderRuntimeState(True, True, True, False, False, False),
        current="next",
        cache=(("self", "next"), None, ("llm", "next")),
        peer_convergence=lambda _settings: False,
    )

    plan = owner.build_plan("next", force_rebuild_llm=False)

    assert plan.should_refresh_peer is False


def test_provider_runtime_owner_forces_peer_refresh_when_live_runtime_is_stale() -> None:
    owner = _runtime_owner(
        events=[],
        state=ProviderRuntimeState(True, True, True, True, False, True),
        current="next",
        cache=(("self", "next"), ("peer", "next", None), ("llm", "next")),
        peer_convergence=lambda _settings: False,
    )

    plan = owner.build_plan("next", force_rebuild_llm=False)

    assert plan.should_refresh_peer is True


@pytest.mark.asyncio
async def test_provider_runtime_owner_rejects_stale_self_runtime_before_caching_success() -> None:
    events: list[str] = []
    owner = _runtime_owner(
        events=events,
        state=ProviderRuntimeState(True, True, True, True, True, False),
        convergence=lambda _settings: False,
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        await owner.apply("settings", ProviderRuntimeApplyPlan(False, False, False))

    assert events == ["common"]


@pytest.mark.asyncio
async def test_provider_runtime_owner_rejects_stale_peer_runtime_before_caching_success() -> None:
    events: list[str] = []
    owner = _runtime_owner(
        events=events,
        state=ProviderRuntimeState(True, True, True, True, False, True),
        peer_convergence=lambda _settings: False,
    )

    with pytest.raises(RuntimeError, match="Peer STT runtime did not converge"):
        await owner.apply("settings", ProviderRuntimeApplyPlan(False, False, False))

    assert events == ["common"]


@pytest.mark.asyncio
async def test_provider_runtime_owner_caches_only_after_self_runtime_converges() -> None:
    events: list[str] = []
    owner = _runtime_owner(
        events=events,
        state=ProviderRuntimeState(True, True, True, True, True, False),
        convergence=lambda _settings: True,
    )

    await owner.apply("settings", ProviderRuntimeApplyPlan(False, False, False))

    assert events == ["common", "signatures"]


@pytest.mark.asyncio
async def test_provider_runtime_owner_applies_channel_effects_in_order_and_marks_retry() -> None:
    events: list[str] = []
    owner = _runtime_owner(
        events=events,
        state=ProviderRuntimeState(True, False, True, True, True, True),
    )

    await owner.apply(
        "settings",
        ProviderRuntimeApplyPlan(True, True, True),
    )

    assert events == ["common", "llm", "peer", "self", "signatures", "retry"]


@pytest.mark.asyncio
async def test_provider_runtime_owner_gpu_recovery_replaces_channel_refreshes() -> None:
    events: list[str] = []
    owner = _runtime_owner(
        events=events,
        state=ProviderRuntimeState(True, False, True, True, True, True),
    )

    await owner.apply(
        "settings",
        ProviderRuntimeApplyPlan(True, True, True, coordinated_gpu_restart=True),
    )

    assert events == ["common", "llm", "gpu", "signatures", "retry"]


@pytest.mark.parametrize(
    ("state", "plan", "expected_code"),
    [
        (
            ProviderRuntimeState(True, False, True, True, True, True),
            ProviderRuntimeApplyPlan(True, False, False),
            "provider_runtime_apply_unavailable",
        ),
        (
            ProviderRuntimeState(True, True, False, True, True, True),
            ProviderRuntimeApplyPlan(False, False, True),
            "stt_runtime_apply_unavailable",
        ),
        (
            ProviderRuntimeState(True, True, True, False, True, True),
            ProviderRuntimeApplyPlan(False, True, False),
            "peer_stt_runtime_apply_unavailable",
        ),
    ],
)
def test_provider_runtime_owner_reports_requested_runtime_unavailability(
    state: ProviderRuntimeState,
    plan: ProviderRuntimeApplyPlan,
    expected_code: str,
) -> None:
    owner = _runtime_owner(events=[], state=state)

    result = owner.unavailable_result(
        "settings",
        plan,
        operation="apply",
        surface="provider",
    )

    assert result is not None
    assert result.diagnostics is not None
    assert result.diagnostics.code == expected_code


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinated_gpu_restart", [False, True])
@pytest.mark.parametrize("usage_refresh_fails", [False, True])
@pytest.mark.parametrize("wrapped_factory_error", [False, True])
async def test_llm_factory_failure_preserves_requested_stt_replacement(
    coordinated_gpu_restart: bool,
    usage_refresh_fails: bool,
    wrapped_factory_error: bool,
) -> None:
    class Resource:
        closed = False

        async def close(self) -> None:
            self.closed = True

    old_stt, new_stt, old_llm = Resource(), Resource(), Resource()
    stt = ProviderRuntimeHandle(name="self_stt", provider=old_stt)
    llm = ProviderRuntimeHandle(name="llm", provider=old_llm)
    diagnostic = ErrorDiagnostics(
        component="provider.llm",
        operation="provider_construction",
        code="provider.auth",
        category="auth",
        visibility="basic",
        content_policy="metadata_only",
        status_code=401,
        retry_after_ms=None,
        fields={},
    )

    def fail_factory(_settings: object) -> object:
        error = RuntimeError("controlled provider construction failure")
        error.failure_diagnostics = diagnostic
        if wrapped_factory_error:
            raise RuntimeError("controlled provider factory wrapper") from error
        raise error

    async def replace_llm(provider: object | None) -> object | None:
        return await llm.replace_provider(provider, start=False)

    async def replace_stt() -> None:
        await stt.replace_provider(new_stt, start=False)

    async def refresh_usage() -> None:
        if usage_refresh_fails:
            raise RuntimeError("controlled usage refresh failure")

    rebuild = LlmProviderRebuildOwner(
        context_provider=lambda: LlmProviderRebuildContext(
            settings="settings",
            replace_provider=replace_llm,
            requires_secret=True,
        ),
        provider_factory=fail_factory,
        availability_sink=lambda _value: None,
        usage_refresh=refresh_usage,
        failure_sink=lambda _message: None,
        success_sink=lambda _message: None,
    )
    owner = _runtime_owner(
        events=[],
        state=ProviderRuntimeState(True, False, True, False, True, False),
    )
    owner.rebuild_llm = lambda: rebuild.rebuild(propagate_failure=True)
    owner.refresh_self_stt = replace_stt
    owner.recover_gpu = lambda _settings, _plan: replace_stt()
    try:
        result = await ProviderRuntimeApplyAdapter(
            owner=owner,
            settings="settings",
            plan=ProviderRuntimeApplyPlan(
                should_rebuild_llm=True,
                should_refresh_peer=False,
                should_refresh_self_stt=True,
                coordinated_gpu_restart=coordinated_gpu_restart,
            ),
        ).apply_runtime(
            RuntimeApplyRequest(settings_values={}, reason="combined_apply", correlation_id=None)
        )

        assert stt.provider is new_stt
        assert old_stt.closed is True
        assert new_stt.closed is False
        assert llm.provider is None
        assert old_llm.closed is True
        assert result.status == "failed"
        assert result.diagnostics is not None
        assert result.diagnostics.diagnostic_id == diagnostic.diagnostic_id
        assert result.diagnostics.code == diagnostic.code
        assert result.diagnostics.status_code == 401
    finally:
        await stt.close()
        await llm.close()


@pytest.mark.asyncio
async def test_llm_provider_rebuild_owner_delivers_success_projection_and_usage_refresh() -> None:
    events: list[object] = []
    provider = object()

    async def replace(value: object | None) -> None:
        events.append(("replace", value))

    owner = LlmProviderRebuildOwner(
        context_provider=lambda: LlmProviderRebuildContext(
            settings="settings",
            replace_provider=replace,
            requires_secret=True,
        ),
        provider_factory=lambda settings: (
            events.append(("create", settings)),
            provider,
        )[1],
        availability_sink=lambda value: events.append(("needs_key", value)),
        usage_refresh=lambda: _record_async(events, "usage"),
        failure_sink=lambda message: events.append(("failure", message)),
        success_sink=lambda message: events.append(("success", message)),
    )

    await owner.rebuild()

    assert events == [
        ("create", "settings"),
        ("replace", provider),
        ("needs_key", False),
        "usage",
        ("success", "[Settings] LLM provider rebuilt successfully"),
    ]


@pytest.mark.asyncio
async def test_llm_provider_rebuild_owner_noops_without_runtime_context() -> None:
    events: list[object] = []
    owner = LlmProviderRebuildOwner(
        context_provider=lambda: None,
        provider_factory=lambda _settings: events.append("create"),
        availability_sink=lambda value: events.append(value),
        usage_refresh=lambda: _record_async(events, "usage"),
        failure_sink=lambda message: events.append(message),
        success_sink=lambda message: events.append(message),
    )

    await owner.rebuild()

    assert events == []


@pytest.mark.asyncio
async def test_managed_provider_rebuild_enables_replacement_while_byok_llm_closes() -> None:
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    replacement_llm = object()
    runtime_values: list[bool] = []
    dashboard_values: list[bool] = []
    context_clears: list[str] = []

    class SlowClosingByokLlm:
        async def close(self) -> None:
            close_started.set()
            await release_close.wait()

    @dataclass
    class RuntimeState:
        llm: object | None
        translation_enabled: bool = False

    runtime = RuntimeState(llm=SlowClosingByokLlm())

    async def replace_provider(provider: object | None) -> None:
        previous = runtime.llm
        runtime.llm = provider
        if previous is not None:
            await previous.close()

    rebuild = LlmProviderRebuildOwner(
        context_provider=lambda: LlmProviderRebuildContext(
            settings=SimpleNamespace(selected_source="managed"),
            replace_provider=replace_provider,
            requires_secret=False,
        ),
        provider_factory=lambda _settings: replacement_llm,
        availability_sink=lambda _value: None,
        usage_refresh=lambda: _record_async([], "usage"),
        failure_sink=lambda _message: None,
        success_sink=lambda _message: None,
    )

    def translation_state() -> TranslationEnableState:
        return TranslationEnableState(
            runtime_available=True,
            translation_enabled=runtime.translation_enabled,
            llm_available=runtime.llm is not None,
            settings_available=True,
            provider_name="openrouter",
            qwen_region=None,
            managed_selected=True,
            managed_china=False,
            managed_local_key_available=False,
            managed_release_service_available=False,
            ingress_frozen=False,
        )

    def set_runtime(enabled: bool) -> None:
        runtime_values.append(enabled)
        runtime.translation_enabled = enabled

    async def prepare(_allow_authorization: bool) -> ManagedTranslationPreparation:
        raise AssertionError("managed preparation must wait for the provider switch")

    translation = TranslationEnableOwner(
        state_provider=translation_state,
        managed_prepare=prepare,
        founder_route=lambda: _false_async(),
        pending_sink=lambda _value: None,
        runtime_ensurer=lambda _mode: _false_async(),
        usage_refresh_sink=lambda: None,
        usage_refresh_now=lambda: _record_async([], "usage"),
        runtime_sink=set_runtime,
        dashboard_sink=dashboard_values.append,
        clear_context=lambda: context_clears.append("clear"),
        warmup=lambda: _record_async([], "warmup"),
        message_sink=lambda _key, _values: None,
        qq_dialog_sink=lambda: None,
        result_sink=lambda _result: None,
        log_basic=lambda _message: None,
        log_diagnostic=lambda _message: None,
        log_error=lambda _message: None,
        founder_letter_sink=lambda: None,
    )

    rebuild_task = asyncio.create_task(rebuild.rebuild())
    await close_started.wait()

    assert runtime.llm is replacement_llm
    assert await translation.set_enabled(True) is True
    assert runtime.llm is replacement_llm
    assert runtime.translation_enabled is True
    assert runtime_values == [True]
    assert dashboard_values == []
    assert context_clears == ["clear"]

    release_close.set()
    await rebuild_task

    assert runtime.llm is replacement_llm


async def _record_async(events: list[object], value: object) -> None:
    events.append(value)


async def _false_async() -> bool:
    return False
