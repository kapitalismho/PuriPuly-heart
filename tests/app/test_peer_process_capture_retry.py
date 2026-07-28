import asyncio

import pytest

from puripuly_heart.app.services.peer_process_capture_retry import (
    PeerProcessCaptureRetryOwner,
)


class RetryRuntime:
    def __init__(
        self,
        events: list[object],
        *,
        result: bool = True,
        error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.result = result
        self.error = error

    async def retry_process_capture(self, *, config: object) -> bool:
        self.events.append(("retry", config))
        if self.error is not None:
            raise self.error
        return self.result


def _owner(
    events: list[object],
    *,
    settings: object | None,
    runtime: RetryRuntime | None,
    active: bool = True,
    ready: bool = True,
) -> PeerProcessCaptureRetryOwner:
    async def ensure_ready() -> bool:
        events.append("ready")
        return ready

    return PeerProcessCaptureRetryOwner(
        settings_provider=lambda: settings,
        runtime_provider=lambda: runtime,
        should_be_active=lambda current: events.append(("active", current)) or active,
        ensure_ready=ensure_ready,
        build_config=lambda current: events.append(("config", current)) or "fresh-config",
        on_retry_succeeded=lambda: events.append("success"),
        sync_effective_flags=lambda current: events.append(("sync", current)),
        refresh_consumers=lambda: events.append("refresh"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings_present", "runtime_present", "active", "expected_events"),
    [
        (False, True, True, []),
        (True, False, True, []),
        (True, True, False, [("active", "settings")]),
    ],
)
async def test_retry_short_circuits_unavailable_or_inactive_state(
    settings_present: bool,
    runtime_present: bool,
    active: bool,
    expected_events: list[object],
) -> None:
    events: list[object] = []
    settings = "settings" if settings_present else None
    runtime = RetryRuntime(events) if runtime_present else None
    owner = _owner(
        events,
        settings=settings,
        runtime=runtime,
        active=active,
    )

    assert await owner.retry() is False
    assert events == expected_events


@pytest.mark.asyncio
async def test_retry_short_circuits_when_readiness_is_retained() -> None:
    events: list[object] = []
    settings = object()
    owner = _owner(
        events,
        settings=settings,
        runtime=RetryRuntime(events),
        ready=False,
    )

    assert await owner.retry() is False
    assert events == [("active", settings), "ready"]


@pytest.mark.asyncio
async def test_missing_settings_does_not_resolve_runtime() -> None:
    owner = PeerProcessCaptureRetryOwner(
        settings_provider=lambda: None,
        runtime_provider=lambda: (_ for _ in ()).throw(RuntimeError("runtime resolved")),
        should_be_active=lambda _settings: True,
        ensure_ready=lambda: asyncio.sleep(0, result=True),
        build_config=lambda _settings: object(),
        on_retry_succeeded=lambda: None,
        sync_effective_flags=lambda _settings: None,
        refresh_consumers=lambda: None,
    )

    assert await owner.retry() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [False, True])
async def test_retry_preserves_result_and_effect_order(result: bool) -> None:
    events: list[object] = []
    settings = object()
    owner = _owner(
        events,
        settings=settings,
        runtime=RetryRuntime(events, result=result),
    )

    assert await owner.retry() is result
    assert events == [
        ("active", settings),
        "ready",
        ("config", settings),
        ("retry", "fresh-config"),
        *(["success"] if result else []),
        ("sync", settings),
        "refresh",
    ]


@pytest.mark.asyncio
async def test_retry_re_resolves_state_at_each_baseline_execution_point() -> None:
    events: list[object] = []
    initial_settings = object()
    ready_settings = object()
    post_retry_settings = object()
    settings = [initial_settings]
    initial_runtime = RetryRuntime(events)

    class ReplacingRuntime:
        async def retry_process_capture(self, *, config: object) -> bool:
            events.append(("retry", config))
            settings[0] = post_retry_settings
            return True

    ready_runtime = ReplacingRuntime()
    runtime: list[RetryRuntime | ReplacingRuntime] = [initial_runtime]

    async def ensure_ready() -> bool:
        events.append("ready")
        settings[0] = ready_settings
        runtime[0] = ready_runtime
        return True

    owner = PeerProcessCaptureRetryOwner(
        settings_provider=lambda: settings[0],
        runtime_provider=lambda: runtime[0],
        should_be_active=lambda current: events.append(("active", current)) or True,
        ensure_ready=ensure_ready,
        build_config=lambda current: events.append(("config", current))
        or ("fresh-config", current),
        on_retry_succeeded=lambda: events.append("success"),
        sync_effective_flags=lambda current: events.append(("sync", current)),
        refresh_consumers=lambda: events.append("refresh"),
    )

    assert await owner.retry() is True
    assert events == [
        ("active", initial_settings),
        "ready",
        ("config", ready_settings),
        ("retry", ("fresh-config", ready_settings)),
        "success",
        ("sync", post_retry_settings),
        "refresh",
    ]


@pytest.mark.asyncio
async def test_retry_propagates_runtime_exception_without_effects() -> None:
    events: list[object] = []
    settings = object()
    owner = _owner(
        events,
        settings=settings,
        runtime=RetryRuntime(events, error=RuntimeError("retry failed")),
    )

    with pytest.raises(RuntimeError, match="retry failed"):
        await owner.retry()

    assert events == [
        ("active", settings),
        "ready",
        ("config", settings),
        ("retry", "fresh-config"),
    ]


@pytest.mark.asyncio
async def test_retry_propagates_cancellation_without_effects() -> None:
    events: list[object] = []
    settings = object()
    owner = _owner(
        events,
        settings=settings,
        runtime=RetryRuntime(events, error=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await owner.retry()

    assert events == [
        ("active", settings),
        "ready",
        ("config", settings),
        ("retry", "fresh-config"),
    ]
