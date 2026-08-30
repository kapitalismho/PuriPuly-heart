from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.app.services.managed_usage import (
    ManagedUsageMetadataResult,
    ManagedUsageOwner,
    ManagedUsageState,
    ManagedUsageViewState,
)
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterStatusRefreshResult,
    TalkTogetherPassStatus,
)
from puripuly_heart.core.openrouter_metadata import OpenRouterKeyMetadata


def _state(
    *,
    settings_available: bool = True,
    managed_key_visible: bool = True,
    release_settings_available: bool = True,
    installation_id: str | None = "installation",
    entitlement_ref: str | None = "entitlement",
    referral_id: str | None = "7KQ9M2",
    ingress_frozen: bool = False,
) -> ManagedUsageState:
    return ManagedUsageState(
        settings_available=settings_available,
        managed_key_visible=managed_key_visible,
        release_settings_available=release_settings_available,
        installation_id=installation_id,
        entitlement_ref=entitlement_ref,
        referral_id=referral_id,
        ingress_frozen=ingress_frozen,
    )


def _owner(
    state_box: list[ManagedUsageState],
    *,
    metadata_fetcher=None,
    release_service_provider=None,
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
        release_service_provider=release_service_provider or (lambda: None),
        metadata_fetcher=metadata_fetcher or no_metadata,
        pending_sink=pending_sink.append,
        view_sink=view_sink.append,
        disable_translation_sink=disabled_sink.append,
        auto_show_founder_letter_provider=lambda _metadata: auto_show,
        normalize_referral_id=lambda value: (
            str(value).strip().upper() if value is not None and str(value).strip() else None
        ),
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


def test_pass_status_cache_is_scoped_to_current_identity() -> None:
    state_box = [_state(referral_id="7kq9m2")]
    views: list[ManagedUsageViewState] = []
    owner = _owner(state_box, views=views)
    pass_status = TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=2,
        invite_limit=3,
    )

    owner.set_view_state(
        visible=True,
        remaining_percent=60,
        referral_id="7kq9m2",
        pass_status=pass_status,
    )

    assert views[-1].pass_status == pass_status
    state_box[0] = _state(entitlement_ref="replacement", referral_id="7KQ9M2")
    assert owner.cached_pass_status_for("7KQ9M2") is None


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
async def test_refresh_applies_metadata_before_exhaustion_and_status_work() -> None:
    state_box = [_state()]
    events: list[str] = []
    pending: list[bool] = []
    views: list[ManagedUsageViewState] = []
    disabled: list[bool] = []
    metadata = OpenRouterKeyMetadata(limit_usd=10, remaining_usd=0, usage_usd=10)

    async def fetch() -> ManagedUsageMetadataResult:
        events.append("metadata")
        return ManagedUsageMetadataResult(key_available=True, metadata=metadata)

    class ReleaseService:
        async def refresh_managed_status(self) -> ManagedOpenRouterStatusRefreshResult:
            events.append("status")
            return ManagedOpenRouterStatusRefreshResult(referral_id="7KQ9M2")

    owner = _owner(
        state_box,
        metadata_fetcher=fetch,
        release_service_provider=ReleaseService,
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
async def test_status_refresh_preserves_cached_pass_on_failure() -> None:
    state_box = [_state()]
    warnings: list[tuple[str, BaseException | None]] = []
    owner = _owner(state_box, warnings=warnings)
    pass_status = TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=1,
        invite_limit=3,
    )
    owner.set_view_state(
        visible=True,
        remaining_percent=70,
        referral_id="7KQ9M2",
        pass_status=pass_status,
    )
    error = RuntimeError("status failed")

    class ReleaseService:
        async def refresh_managed_status(self):
            raise error

    result = await owner.refresh_status_best_effort(service=ReleaseService())

    assert result.succeeded is False
    assert result.pass_status == pass_status
    assert warnings == [("[ManagedAuth] Managed status refresh failed: status failed", error)]
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
