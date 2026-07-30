from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.provider_runtime_apply import (
    LlmProviderRebuildContext,
    LlmProviderRebuildOwner,
    ProviderRuntimeApplyPlan,
    ProviderRuntimeOwner,
    ProviderRuntimeState,
)
from puripuly_heart.app.services.translation_enable import (
    ManagedTranslationPreparation,
    TranslationEnableOwner,
    TranslationEnableState,
)


def _runtime_owner(
    *,
    events: list[str],
    state: ProviderRuntimeState,
    current: object | None = None,
    cache: tuple[object | None, object | None, object | None] = (None, None, None),
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
        should_refresh_peer=True,
        should_refresh_self_stt=True,
        coordinated_gpu_restart=True,
    )


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
        ("replace", None),
        ("create", "settings"),
        ("replace", provider),
        ("needs_key", False),
        "usage",
        ("success", "[Settings] LLM provider rebuilt successfully"),
    ]


@pytest.mark.asyncio
async def test_llm_provider_rebuild_owner_contains_factory_failure() -> None:
    events: list[object] = []

    async def replace(value: object | None) -> None:
        events.append(("replace", value))

    def fail_factory(_settings: object) -> object:
        raise RuntimeError("unavailable")

    owner = LlmProviderRebuildOwner(
        context_provider=lambda: LlmProviderRebuildContext(
            settings=SimpleNamespace(),
            replace_provider=replace,
            requires_secret=True,
        ),
        provider_factory=fail_factory,
        availability_sink=lambda value: events.append(("needs_key", value)),
        usage_refresh=lambda: _record_async(events, "usage"),
        failure_sink=lambda message: events.append(("failure", message)),
        success_sink=lambda message: events.append(("success", message)),
    )

    await owner.rebuild()

    assert events == [
        ("replace", None),
        ("replace", None),
        ("needs_key", True),
        "usage",
        ("failure", "LLM provider not available"),
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
async def test_managed_provider_rebuild_blocks_concurrent_enable_from_closing_byok_llm() -> None:
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

    async def prepare() -> ManagedTranslationPreparation:
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
        log_detailed=lambda _message: None,
        log_error=lambda _message: None,
        founder_letter_sink=lambda: None,
    )

    rebuild_task = asyncio.create_task(rebuild.rebuild())
    await close_started.wait()

    assert runtime.llm is None
    assert await translation.set_enabled(True) is False
    assert runtime.llm is None
    assert runtime.translation_enabled is False
    assert runtime_values == [False]
    assert dashboard_values == [False]
    assert context_clears == []

    release_close.set()
    await rebuild_task

    assert runtime.llm is replacement_llm


async def _record_async(events: list[object], value: object) -> None:
    events.append(value)


async def _false_async() -> bool:
    return False
