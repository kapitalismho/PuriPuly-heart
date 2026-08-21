from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.app.services.translation_enable import (
    ManagedTranslationPreparation,
    TranslationEnableOwner,
    TranslationEnableState,
)


def _state(
    *,
    runtime_available: bool = True,
    translation_enabled: bool = False,
    llm_available: bool = True,
    settings_available: bool = True,
    provider_name: str | None = "gemini",
    qwen_region: str | None = None,
    managed_selected: bool = False,
    managed_china: bool = False,
    managed_local_key_available: bool = False,
    managed_release_service_available: bool = False,
    ingress_frozen: bool = False,
) -> TranslationEnableState:
    return TranslationEnableState(
        runtime_available=runtime_available,
        translation_enabled=translation_enabled,
        llm_available=llm_available,
        settings_available=settings_available,
        provider_name=provider_name,
        qwen_region=qwen_region,
        managed_selected=managed_selected,
        managed_china=managed_china,
        managed_local_key_available=managed_local_key_available,
        managed_release_service_available=managed_release_service_available,
        ingress_frozen=ingress_frozen,
    )


def _owner(
    state_box: list[TranslationEnableState],
    *,
    preparation: ManagedTranslationPreparation | None = None,
    prepare=None,
    founder_route=None,
    pending: list[bool] | None = None,
    runtime_modes: list[str] | None = None,
    usage_refreshes: list[str] | None = None,
    runtime_values: list[bool] | None = None,
    dashboard_values: list[bool] | None = None,
    messages: list[tuple[str, dict[str, object]]] | None = None,
    qq_calls: list[str] | None = None,
    logs: list[tuple[str, str]] | None = None,
    founder_calls: list[str] | None = None,
    clears: list[str] | None = None,
    warmups: list[str] | None = None,
    teardowns: list[str] | None = None,
    starting_values: list[bool] | None = None,
) -> TranslationEnableOwner:
    pending_values = pending if pending is not None else []
    runtime_mode_values = runtime_modes if runtime_modes is not None else []
    refresh_values = usage_refreshes if usage_refreshes is not None else []
    runtime_sink_values = runtime_values if runtime_values is not None else []
    dashboard_sink_values = dashboard_values if dashboard_values is not None else []
    message_values = messages if messages is not None else []
    qq_values = qq_calls if qq_calls is not None else []
    log_values = logs if logs is not None else []
    founder_values = founder_calls if founder_calls is not None else []
    clear_values = clears if clears is not None else []
    warmup_values = warmups if warmups is not None else []
    teardown_values = teardowns if teardowns is not None else []
    starting_sink_values = starting_values if starting_values is not None else []

    async def default_prepare() -> ManagedTranslationPreparation:
        return preparation or ManagedTranslationPreparation(ready=True)

    async def default_founder_route() -> bool:
        return False

    async def ensure_runtime(mode: str) -> bool:
        runtime_mode_values.append(mode)
        state_box[0] = replace(state_box[0], llm_available=True)
        return True

    async def refresh_now() -> None:
        refresh_values.append("now")

    def set_runtime(enabled: bool) -> None:
        runtime_sink_values.append(enabled)
        state_box[0] = replace(state_box[0], translation_enabled=enabled)

    async def warmup() -> None:
        warmup_values.append("warmup")

    async def teardown() -> None:
        teardown_values.append("teardown")

    return TranslationEnableOwner(
        state_provider=lambda: state_box[0],
        managed_prepare=prepare or default_prepare,
        founder_route=founder_route or default_founder_route,
        pending_sink=pending_values.append,
        runtime_ensurer=ensure_runtime,
        usage_refresh_sink=lambda: refresh_values.append("scheduled"),
        usage_refresh_now=refresh_now,
        runtime_sink=set_runtime,
        dashboard_sink=dashboard_sink_values.append,
        clear_context=lambda: clear_values.append("clear"),
        warmup=warmup,
        message_sink=lambda key, values: message_values.append((key, dict(values))),
        qq_dialog_sink=lambda: qq_values.append("show"),
        result_sink=lambda _result: None,
        log_basic=lambda message: log_values.append(("basic", message)),
        log_detailed=lambda message: log_values.append(("detailed", message)),
        log_error=lambda message: log_values.append(("error", message)),
        founder_letter_sink=lambda: founder_values.append("show"),
        teardown=teardown,
        starting_sink=starting_sink_values.append,
    )


@pytest.mark.asyncio
async def test_nonmanaged_enable_owns_runtime_context_and_warmup_sequence() -> None:
    state_box = [_state(provider_name="qwen", qwen_region="china")]
    runtime_values: list[bool] = []
    clears: list[str] = []
    warmups: list[str] = []
    logs: list[tuple[str, str]] = []
    owner = _owner(
        state_box,
        runtime_values=runtime_values,
        clears=clears,
        warmups=warmups,
        logs=logs,
    )

    assert await owner.set_enabled(True) is True

    assert runtime_values == [True]
    assert clears == ["clear"]
    assert warmups == ["warmup"]
    assert ("basic", "[Translation] Enabled with provider: qwen") in logs
    assert (
        "detailed",
        "[Translation] Provider detail: provider=qwen region=china",
    ) in logs


@pytest.mark.asyncio
async def test_disable_owns_runtime_context_and_teardown_sequence() -> None:
    state_box = [_state(translation_enabled=True)]
    runtime_values: list[bool] = []
    clears: list[str] = []
    teardowns: list[str] = []
    owner = _owner(
        state_box,
        runtime_values=runtime_values,
        clears=clears,
        teardowns=teardowns,
    )

    assert await owner.set_enabled(False) is False

    assert runtime_values == [False]
    assert clears == ["clear"]
    assert teardowns == ["teardown"]


@pytest.mark.asyncio
async def test_enable_owns_starting_flag_around_warmup() -> None:
    state_box = [_state()]
    starting_values: list[bool] = []
    warmups: list[str] = []

    async def warmup() -> None:
        assert starting_values == [True]
        warmups.append("warmup")

    owner = _owner(state_box, starting_values=starting_values, warmups=warmups)
    owner.warmup = warmup

    assert await owner.set_enabled(True) is True
    assert starting_values == [True, False]
    assert warmups == ["warmup"]


@pytest.mark.asyncio
async def test_stale_enable_does_not_clear_newer_starting_flag() -> None:
    state_box = [_state()]
    starting_values: list[bool] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def warmup() -> None:
        entered.set()
        await release.wait()

    owner = _owner(state_box, starting_values=starting_values)
    owner.warmup = warmup

    enabling = asyncio.create_task(owner.set_enabled(True))
    await entered.wait()
    assert starting_values == [True]

    await owner.set_enabled(False)
    assert starting_values[-1] is False
    release.set()
    await enabling
    assert starting_values[-1] is False


@pytest.mark.asyncio
async def test_missing_llm_disables_runtime_and_dashboard() -> None:
    state_box = [_state(llm_available=False)]
    runtime_values: list[bool] = []
    dashboard_values: list[bool] = []
    logs: list[tuple[str, str]] = []
    owner = _owner(
        state_box,
        runtime_values=runtime_values,
        dashboard_values=dashboard_values,
        logs=logs,
    )

    assert await owner.set_enabled(True) is False

    assert runtime_values == [False]
    assert dashboard_values == [False]
    assert logs[-1] == (
        "error",
        "Translation is ON but LLM provider is not configured.",
    )


@pytest.mark.asyncio
async def test_managed_ready_sequences_pending_and_runtime_rebuild() -> None:
    state_box = [
        _state(
            llm_available=False,
            managed_selected=True,
            managed_release_service_available=True,
        )
    ]
    pending: list[bool] = []
    runtime_modes: list[str] = []
    runtime_values: list[bool] = []
    owner = _owner(
        state_box,
        pending=pending,
        runtime_modes=runtime_modes,
        runtime_values=runtime_values,
    )

    assert await owner.set_enabled(True) is True

    assert pending == [True, False]
    assert runtime_modes == ["if_missing"]
    assert runtime_values == [True]


@pytest.mark.asyncio
async def test_newer_disable_fences_stale_managed_prepare_result() -> None:
    state_box = [
        _state(
            managed_selected=True,
            managed_release_service_available=True,
        )
    ]
    started = asyncio.Event()
    release = asyncio.Event()
    runtime_values: list[bool] = []

    async def prepare() -> ManagedTranslationPreparation:
        started.set()
        await release.wait()
        return ManagedTranslationPreparation(ready=True)

    owner = _owner(state_box, prepare=prepare, runtime_values=runtime_values)
    enable_task = asyncio.create_task(owner.set_enabled(True))
    await started.wait()

    assert await owner.set_enabled(False) is False
    release.set()
    assert await enable_task is False

    assert runtime_values == [False]
    assert state_box[0].translation_enabled is False


@pytest.mark.asyncio
async def test_managed_failure_refreshes_usage_and_routes_qq_without_message() -> None:
    state_box = [
        _state(
            managed_selected=True,
            managed_china=True,
            managed_release_service_available=True,
        )
    ]
    pending: list[bool] = []
    refreshes: list[str] = []
    runtime_values: list[bool] = []
    dashboard_values: list[bool] = []
    messages: list[tuple[str, dict[str, object]]] = []
    qq_calls: list[str] = []
    logs: list[tuple[str, str]] = []
    owner = _owner(
        state_box,
        preparation=ManagedTranslationPreparation(
            ready=False,
            message_key="qq_managed_auth.required",
            diagnostics_text="operation=prepare code=required",
            show_qq_dialog=True,
        ),
        pending=pending,
        usage_refreshes=refreshes,
        runtime_values=runtime_values,
        dashboard_values=dashboard_values,
        messages=messages,
        qq_calls=qq_calls,
        logs=logs,
    )

    assert await owner.set_enabled(True) is False

    assert pending == [True, False]
    assert refreshes == ["now"]
    assert runtime_values == [False]
    assert dashboard_values == [False]
    assert messages == []
    assert qq_calls == ["show"]
    assert logs[-1] == (
        "error",
        "[ManagedAuth] operation=prepare code=required",
    )


@pytest.mark.asyncio
async def test_founder_route_and_close_reject_enable_without_prepare() -> None:
    state_box = [
        _state(
            managed_selected=True,
            managed_release_service_available=True,
        )
    ]
    prepare_calls: list[str] = []

    async def prepare() -> ManagedTranslationPreparation:
        prepare_calls.append("prepare")
        return ManagedTranslationPreparation(ready=True)

    async def founder_route() -> bool:
        return True

    owner = _owner(state_box, prepare=prepare, founder_route=founder_route)

    assert await owner.set_enabled(True) is False
    assert prepare_calls == []

    await owner.close()

    assert await owner.set_enabled(True) is False


def test_managed_exhaustion_disables_runtime_and_optionally_reopens_letter() -> None:
    state_box = [_state(translation_enabled=True)]
    pending: list[bool] = []
    runtime_values: list[bool] = []
    dashboard_values: list[bool] = []
    founder_calls: list[str] = []
    owner = _owner(
        state_box,
        pending=pending,
        runtime_values=runtime_values,
        dashboard_values=dashboard_values,
        founder_calls=founder_calls,
    )

    owner.disable_for_managed_exhaustion(reopen_founder_letter=True)

    assert owner.intent_enabled is False
    assert pending == [False]
    assert runtime_values == [False]
    assert dashboard_values == [False]
    assert founder_calls == ["show"]
