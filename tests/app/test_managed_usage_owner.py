from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.app.services.managed.managed_usage import (
    ManagedUsageMetadataResult,
    ManagedUsageOwner,
    ManagedUsageState,
    ManagedUsageViewState,
)
from puripuly_heart.core.openrouter.openrouter_metadata import OpenRouterKeyMetadata


def _state(
    *,
    settings_available: bool = True,
    managed_key_visible: bool = True,
    release_settings_available: bool = True,
    installation_id: str | None = "installation",
    entitlement_ref: str | None = "entitlement",
    ingress_frozen: bool = False,
) -> ManagedUsageState:
    return ManagedUsageState(
        settings_available=settings_available,
        managed_key_visible=managed_key_visible,
        release_settings_available=release_settings_available,
        installation_id=installation_id,
        entitlement_ref=entitlement_ref,
        ingress_frozen=ingress_frozen,
    )


def _owner(
    state_box: list[ManagedUsageState],
    *,
    metadata_fetcher=None,
    pending: list[bool] | None = None,
    views: list[ManagedUsageViewState] | None = None,
    disabled: list[bool] | None = None,
    warnings: list[tuple[str, BaseException | None]] | None = None,
    auto_show: bool = False,
) -> ManagedUsageOwner:
    pending_sink = pending if pending is not None else []
    view_sink = views if views is not None else []
    disabled_sink = disabled if disabled is not None else []
    warning_sink = warnings if warnings is not None else []

    async def no_metadata() -> ManagedUsageMetadataResult:
        return ManagedUsageMetadataResult(key_available=False, metadata=None)

    return ManagedUsageOwner(
        state_provider=lambda: state_box[0],
        release_service_provider=lambda: None,
        metadata_fetcher=metadata_fetcher or no_metadata,
        pending_sink=pending_sink.append,
        view_sink=view_sink.append,
        disable_translation_sink=disabled_sink.append,
        auto_show_founder_letter_provider=lambda _metadata: auto_show,
        warning_sink=lambda message, exception: warning_sink.append((message, exception)),
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    (
        (None, None),
        (OpenRouterKeyMetadata(limit_usd=None, remaining_usd=1, usage_usd=None), None),
        (OpenRouterKeyMetadata(limit_usd=0, remaining_usd=1, usage_usd=None), None),
        (OpenRouterKeyMetadata(limit_usd=10, remaining_usd=6.04, usage_usd=None), 60),
        (OpenRouterKeyMetadata(limit_usd=10, remaining_usd=12, usage_usd=None), 100),
        (OpenRouterKeyMetadata(limit_usd=10, remaining_usd=-1, usage_usd=None), 0),
    ),
)
def test_remaining_percent_is_clamped_and_handles_unavailable_metadata(
    metadata: OpenRouterKeyMetadata | None,
    expected: int | None,
) -> None:
    assert ManagedUsageOwner.remaining_percent_for(metadata) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected_visible"),
    (
        (_state(settings_available=False), False),
        (_state(managed_key_visible=False), False),
        (_state(release_settings_available=False), True),
    ),
)
async def test_refresh_projects_unavailable_managed_states(
    state: ManagedUsageState,
    expected_visible: bool,
) -> None:
    state_box = [state]
    pending: list[bool] = []
    views: list[ManagedUsageViewState] = []
    owner = _owner(state_box, pending=pending, views=views)

    await owner.refresh()

    assert pending == [False]
    assert views[-1].visible is expected_visible
    assert views[-1].remaining_percent is None
    assert owner.usage_metadata is None


@pytest.mark.asyncio
async def test_refresh_applies_metadata_before_disabling_exhausted_translation() -> None:
    state_box = [_state()]
    events: list[str] = []
    pending: list[bool] = []
    views: list[ManagedUsageViewState] = []
    disabled: list[bool] = []
    metadata = OpenRouterKeyMetadata(limit_usd=10, remaining_usd=0, usage_usd=10)

    async def fetch() -> ManagedUsageMetadataResult:
        events.append("metadata")
        return ManagedUsageMetadataResult(key_available=True, metadata=metadata)

    owner = _owner(
        state_box,
        metadata_fetcher=fetch,
        pending=pending,
        views=views,
        disabled=disabled,
        auto_show=True,
    )

    await owner.refresh()

    assert events == ["metadata"]
    assert pending == [False]
    assert views[-1].remaining_percent == 0
    assert disabled == [True]
    await owner.close()


@pytest.mark.asyncio
async def test_refresh_discards_metadata_after_entitlement_changes() -> None:
    state_box = [_state(entitlement_ref="old")]
    entered = asyncio.Event()
    release = asyncio.Event()
    views: list[ManagedUsageViewState] = []
    disabled: list[bool] = []
    metadata = OpenRouterKeyMetadata(limit_usd=10, remaining_usd=0, usage_usd=10)

    async def fetch() -> ManagedUsageMetadataResult:
        entered.set()
        await release.wait()
        return ManagedUsageMetadataResult(key_available=True, metadata=metadata)

    owner = _owner(
        state_box,
        metadata_fetcher=fetch,
        views=views,
        disabled=disabled,
    )
    owner.usage_metadata = metadata
    owner.usage_metadata_entitlement_ref = "old"
    route = asyncio.create_task(owner.should_route_to_founder_letter())
    await entered.wait()
    state_box[0] = _state(entitlement_ref="new")
    release.set()
    routed = await route

    assert routed is False
    assert disabled == []
    assert owner.usage_metadata is None
    assert views == []
    await owner.close()


@pytest.mark.asyncio
async def test_close_cancels_scheduled_usage_refresh_and_rejects_new_work() -> None:
    state_box = [_state()]
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def fetch() -> ManagedUsageMetadataResult:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    owner = _owner(state_box, metadata_fetcher=fetch)

    assert owner.schedule_usage_refresh() is True
    await entered.wait()
    await owner.close()

    assert cancelled.is_set()
    assert owner.schedule_usage_refresh() is False


@pytest.mark.asyncio
async def test_failed_refresh_retains_cause_without_changing_cached_route():
    async def fail():
        raise PermissionError(13, "synthetic-private-path")

    disabled = []
    warnings = []
    owner = _owner([_state()], metadata_fetcher=fail, disabled=disabled, warnings=warnings)
    owner.usage_metadata = OpenRouterKeyMetadata(limit_usd=10, remaining_usd=0, usage_usd=10)
    owner.usage_metadata_entitlement_ref = "entitlement"

    assert await owner.should_route_to_founder_letter() is True
    assert disabled == [True]
    assert owner.last_result.diagnostics.fields["errno"] == 13
    assert "synthetic-private" not in str(owner.last_result)
    await owner.close()
    assert warnings == []


def test_managed_account_snapshot_preserves_failed_usage_observation():
    from types import SimpleNamespace

    from puripuly_heart.app.adapters.ui_runtime import UiManagedRuntimeAdapter
    from puripuly_heart.core.diagnostics import exception_diagnostics

    diagnostics = exception_diagnostics(
        PermissionError(13, "synthetic-private-path"),
        component="managed_usage",
        operation="managed_usage_refresh",
        code="managed_usage_refresh_failed",
    )
    usage = _owner([_state()])
    usage.last_result = ManagedUsageMetadataResult(True, None, diagnostics, scope=_state())
    auth = SimpleNamespace(
        state_provider=lambda: SimpleNamespace(
            settings_available=True,
            managed_selected=True,
            managed_china=False,
            release_service_available=True,
            ingress_frozen=False,
            local_key_available=True,
        ),
        dashboard_action=lambda: "idle",
    )
    result = UiManagedRuntimeAdapter(SimpleNamespace(auth=auth, usage=usage)).managed_account_snapshot()
    assert result["diagnostics"] is diagnostics
    assert result["credential_present"] is True
    assert result["budget"]["remaining_percent"] is None
    assert "synthetic-private" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_retired_usage_failure_is_scoped_history_not_current_account(raises):
    from types import SimpleNamespace

    from puripuly_heart.app.adapters.ui_runtime import UiManagedRuntimeAdapter
    from puripuly_heart.core.diagnostics import exception_diagnostics

    state = [_state(entitlement_ref="old")]
    entered = asyncio.Event()
    finish = asyncio.Event()
    diagnostics = exception_diagnostics(
        PermissionError(13, "synthetic-private-path"),
        component="managed_usage",
        operation="managed_usage_refresh",
        code="managed_usage_refresh_failed",
    )

    async def fetch():
        entered.set()
        await finish.wait()
        if raises:
            raise PermissionError(13, "synthetic-private-path")
        return ManagedUsageMetadataResult(True, None, diagnostics)

    disabled = []
    views = []
    owner = _owner(state, metadata_fetcher=fetch, disabled=disabled, views=views)
    work = asyncio.create_task(owner.refresh_best_effort())
    await entered.wait()
    state[0] = _state(entitlement_ref="new")
    finish.set()
    await work
    auth = SimpleNamespace(
        state_provider=lambda: SimpleNamespace(
            settings_available=True, managed_selected=True, managed_china=False,
            release_service_available=True, ingress_frozen=False, local_key_available=True,
        ),
        dashboard_action=lambda: "idle",
    )
    current = UiManagedRuntimeAdapter(SimpleNamespace(auth=auth, usage=owner)).managed_account_snapshot()
    assert current["diagnostics"] is None
    assert current["secondary_diagnostics"] == ()
    assert current["budget"]["remaining_percent"] is None
    assert owner.current_result() is None
    assert owner.superseded_result.scope.entitlement_ref == "old"
    assert owner.superseded_result.diagnostics.fields["errno"] == 13
    assert disabled == []
    assert views == []
    assert "synthetic-private" not in str(current)
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_scope", [False, True])
async def test_warning_delivery_failure_remains_with_originating_usage_scope(newer_scope):
    from types import SimpleNamespace

    from puripuly_heart.app.adapters.ui_runtime import UiManagedRuntimeAdapter

    state = [_state(entitlement_ref="old")]
    entered = asyncio.Event()
    finish = asyncio.Event()
    metadata = OpenRouterKeyMetadata(limit_usd=10, remaining_usd=6, usage_usd=4)
    warnings = []

    async def fetch():
        if state[0].entitlement_ref == "old":
            entered.set()
            await finish.wait()
            raise PermissionError(13, "synthetic-private-original")
        return ManagedUsageMetadataResult(True, metadata)

    def fail_warning(_message, exception):
        warnings.append(exception)
        raise OSError(5, "synthetic-private-notification")

    disabled = []
    views = []
    owner = _owner(state, metadata_fetcher=fetch, disabled=disabled, views=views)
    owner.warning_sink = fail_warning
    old = asyncio.create_task(owner.refresh_best_effort())
    try:
        await entered.wait()
        current_before_warning = None
        if newer_scope:
            state[0] = _state(entitlement_ref="new")
            await owner.refresh()
            current_before_warning = owner.current_result()
            assert current_before_warning.metadata is metadata
        finish.set()
        await old
        auth = SimpleNamespace(
            state_provider=lambda: SimpleNamespace(
                settings_available=True, managed_selected=True, managed_china=False,
                release_service_available=True, ingress_frozen=False, local_key_available=True,
            ),
            dashboard_action=lambda: "idle",
        )
        snapshot = UiManagedRuntimeAdapter(SimpleNamespace(auth=auth, usage=owner)).managed_account_snapshot()
        if newer_scope:
            assert owner.current_result() is current_before_warning
            assert snapshot["diagnostics"] is None
            assert snapshot["secondary_diagnostics"] == ()
            assert snapshot["budget"]["remaining_percent"] == 60
            observed = owner.superseded_result
            assert observed.scope.entitlement_ref == "old"
            assert len(views) == 1
        else:
            observed = owner.current_result()
            assert observed.scope.entitlement_ref == "old"
            assert snapshot["diagnostics"].diagnostic_id == observed.diagnostics.diagnostic_id
            assert snapshot["secondary_diagnostics"] == observed.secondary_diagnostics
            assert views == []
        assert observed.diagnostics.fields["errno"] == 13
        assert observed.secondary_diagnostics[0].code == "managed_usage_notification_failed"
        assert observed.secondary_diagnostics[0].fields["errno"] == 5
        assert observed.secondary_diagnostics[0].cause_id == observed.diagnostics.diagnostic_id
        assert len(warnings) == 1
        assert disabled == []
        assert "synthetic-private" not in str(snapshot)
        assert "synthetic-private" not in str(observed)
    finally:
        finish.set()
        await asyncio.gather(old, return_exceptions=True)
        await owner.close()
