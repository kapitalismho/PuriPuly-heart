from __future__ import annotations

import pytest

from puripuly_heart.app.services.managed_auth import (
    ManagedAuthExecutionResult,
    ManagedAuthOwner,
    ManagedAuthState,
)
from puripuly_heart.core.managed_openrouter_release import TalkTogetherPassStatus
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
    TransactionResult,
)


def _state(
    *,
    settings_available: bool = True,
    managed_selected: bool = True,
    managed_china: bool = False,
    local_key_available: bool = False,
    release_service_available: bool = True,
    runtime_available: bool = True,
    ingress_frozen: bool = False,
) -> ManagedAuthState:
    return ManagedAuthState(
        settings_available=settings_available,
        managed_selected=managed_selected,
        managed_china=managed_china,
        local_key_available=local_key_available,
        release_service_available=release_service_available,
        runtime_available=runtime_available,
        ingress_frozen=ingress_frozen,
    )


def _owner(
    state_box: list[ManagedAuthState],
    *,
    qq_result: ManagedAuthExecutionResult | None = None,
    discord_result: ManagedAuthExecutionResult | None = None,
    pending: list[bool] | None = None,
    runtime_modes: list[str] | None = None,
    views: list[tuple[str | None, TalkTogetherPassStatus | None]] | None = None,
    refreshes: list[str] | None = None,
    messages: list[tuple[str, dict[str, object]]] | None = None,
    results: list[TransactionResult] | None = None,
    logs: list[str] | None = None,
    runtime_ready: bool = True,
) -> ManagedAuthOwner:
    pending_sink = pending if pending is not None else []
    runtime_sink = runtime_modes if runtime_modes is not None else []
    view_sink = views if views is not None else []
    refresh_sink = refreshes if refreshes is not None else []
    message_sink = messages if messages is not None else []
    result_sink = results if results is not None else []
    log_sink = logs if logs is not None else []

    async def execute_qq(
        _qq_identity: str,
        _credential: str,
    ) -> ManagedAuthExecutionResult:
        return qq_result or ManagedAuthExecutionResult(succeeded=True)

    async def execute_discord(
        _referral_id: str | None,
        callback,
    ) -> ManagedAuthExecutionResult:
        if callback is not None:
            callback()
        return discord_result or ManagedAuthExecutionResult(succeeded=True)

    async def ensure_runtime(mode: str) -> bool:
        runtime_sink.append(mode)
        return runtime_ready

    return ManagedAuthOwner(
        state_provider=lambda: state_box[0],
        pending_sink=pending_sink.append,
        qq_executor=execute_qq,
        discord_executor=execute_discord,
        runtime_ensurer=ensure_runtime,
        usage_view_sink=lambda referral_id, pass_status: view_sink.append(
            (referral_id, pass_status)
        ),
        usage_refresh_sink=lambda: refresh_sink.append("refresh"),
        message_sink=lambda key, values: message_sink.append((key, dict(values))),
        result_sink=result_sink.append,
        log_sink=log_sink.append,
    )


@pytest.mark.parametrize(
    ("state", "pending", "in_progress", "expected"),
    (
        (_state(managed_selected=False), False, False, "continue"),
        (_state(), True, False, "in_progress"),
        (_state(), False, True, "in_progress"),
        (_state(local_key_available=True), False, False, "continue"),
        (_state(), False, False, "prompt"),
    ),
)
def test_dashboard_action_is_owned_by_managed_auth_state(
    state: ManagedAuthState,
    pending: bool,
    in_progress: bool,
    expected: str,
) -> None:
    owner = _owner([state])
    owner.pending = pending
    owner.discord_in_progress = in_progress

    assert owner.dashboard_action() == expected


@pytest.mark.asyncio
async def test_qq_success_clears_pending_and_routes_runtime_rebuild() -> None:
    pending: list[bool] = []
    runtime_modes: list[str] = []
    owner = _owner(
        [_state(managed_china=True)],
        qq_result=ManagedAuthExecutionResult(
            succeeded=True,
            runtime_rebuild="if_missing",
        ),
        pending=pending,
        runtime_modes=runtime_modes,
    )
    owner.pending = True

    assert await owner.start_qq(qq_identity="qq", credential="credential") is True

    assert pending == [False]
    assert runtime_modes == ["if_missing"]


@pytest.mark.asyncio
async def test_qq_failure_maps_service_message_at_owner_boundary() -> None:
    owner = _owner(
        [_state(managed_china=True)],
        qq_result=ManagedAuthExecutionResult(
            succeeded=False,
            message_key="qq_managed_auth.mismatch",
            message_kwargs={"retry_after_ms": 10},
        ),
    )

    assert await owner.start_qq(
        qq_identity="qq",
        credential="credential",
    ) == (
        "qq_auth.error.credential_mismatch",
        {"retry_after_ms": 10},
    )


@pytest.mark.asyncio
async def test_discord_success_sequences_pending_result_runtime_view_and_refresh() -> None:
    pending: list[bool] = []
    runtime_modes: list[str] = []
    views: list[tuple[str | None, TalkTogetherPassStatus | None]] = []
    refreshes: list[str] = []
    results: list[TransactionResult] = []
    callbacks: list[str] = []
    pass_status = TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=2,
        invite_limit=5,
    )
    transaction = TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=None,
        diagnostics=None,
    )
    owner = _owner(
        [_state()],
        discord_result=ManagedAuthExecutionResult(
            succeeded=True,
            transaction_result=transaction,
            referral_bonus_applied=True,
            referral_id="7KQ9M2",
            pass_status=pass_status,
            runtime_rebuild="always",
        ),
        pending=pending,
        runtime_modes=runtime_modes,
        views=views,
        refreshes=refreshes,
        results=results,
    )

    assert (
        await owner.start_discord(
            on_callback_received=lambda: callbacks.append("callback"),
        )
        is True
    )

    assert callbacks == ["callback"]
    assert pending == [True, False]
    assert results == [transaction]
    assert runtime_modes == ["always"]
    assert views == [("7KQ9M2", pass_status)]
    assert refreshes == ["refresh"]
    assert owner.last_referral_bonus_applied is True
    assert owner.discord_in_progress is False
    assert owner.callback_received_hook is None


@pytest.mark.asyncio
async def test_discord_failure_contains_safe_message_and_restores_state() -> None:
    pending: list[bool] = []
    messages: list[tuple[str, dict[str, object]]] = []
    logs: list[str] = []
    owner = _owner(
        [_state()],
        discord_result=ManagedAuthExecutionResult(
            succeeded=False,
            message_key="discord_auth.error.expired",
            message_kwargs={"retry_after_ms": 50},
            error_class="auth",
        ),
        pending=pending,
        messages=messages,
        logs=logs,
    )

    assert await owner.start_discord() is False

    assert pending == [True, False]
    assert messages == [("discord_auth.error.expired", {"retry_after_ms": 50})]
    assert logs == [
        "[ManagedAuth] Discord auth failed: " "message_key=discord_auth.error.expired class=auth"
    ]
    assert owner.discord_in_progress is False
    assert owner.callback_received_hook is None


@pytest.mark.asyncio
async def test_discord_runtime_failure_does_not_publish_usage_state() -> None:
    messages: list[tuple[str, dict[str, object]]] = []
    views: list[tuple[str | None, TalkTogetherPassStatus | None]] = []
    refreshes: list[str] = []
    owner = _owner(
        [_state()],
        discord_result=ManagedAuthExecutionResult(
            succeeded=True,
            runtime_rebuild="always",
        ),
        messages=messages,
        views=views,
        refreshes=refreshes,
        runtime_ready=False,
    )

    assert await owner.start_discord() is False

    assert messages == [("discord_auth.error.retry", {})]
    assert views == []
    assert refreshes == []


@pytest.mark.asyncio
async def test_close_rejects_new_auth_and_clears_callback_and_pending() -> None:
    pending: list[bool] = []
    messages: list[tuple[str, dict[str, object]]] = []
    owner = _owner([_state(managed_china=True)], pending=pending, messages=messages)
    owner.pending = True
    owner.discord_in_progress = True
    owner.callback_received_hook = lambda: None

    await owner.close()

    assert owner.pending is False
    assert owner.discord_in_progress is False
    assert owner.callback_received_hook is None
    assert await owner.start_discord() is False
    assert await owner.start_qq(
        qq_identity="qq",
        credential="credential",
    ) == ("qq_auth.error.retry", {})
    assert messages == [("discord_auth.error.retry", {})]
