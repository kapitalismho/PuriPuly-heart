from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.app.services.managed_usage import (
    ManagedUsageMetadataResult,
    ManagedUsageOwner,
    ManagedUsageState,
    ManagedUsageViewState,
)
from puripuly_heart.core.openrouter_metadata import OpenRouterKeyMetadata


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
