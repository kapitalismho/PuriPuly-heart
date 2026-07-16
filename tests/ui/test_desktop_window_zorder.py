from __future__ import annotations

import pytest

from puripuly_heart.ui import desktop_window_zorder


class FakeWin32WindowApi:
    def __init__(
        self,
        *,
        windows: tuple[int, ...] = (101,),
        styles: tuple[int, ...] = (desktop_window_zorder._WS_EX_TRANSPARENT,),
        set_result: tuple[bool, int | None] = (True, None),
        enum_error: int | None = None,
    ) -> None:
        self.windows = windows
        self.styles = list(styles)
        self.current_style = self.styles[-1] if self.styles else 0
        self.set_result = set_result
        self.enum_error = enum_error
        self.window_queries: list[int] = []
        self.set_calls: list[int] = []
        self.pid = 4321

    def top_level_windows_for_process(
        self, pid: int
    ) -> desktop_window_zorder.WindowEnumerationResult:
        self.window_queries.append(pid)
        return desktop_window_zorder.WindowEnumerationResult(
            windows=self.windows,
            win32_error=self.enum_error,
        )

    def is_window(self, hwnd: int) -> bool:
        return hwnd in self.windows

    def process_id(self, hwnd: int) -> int | None:
        return self.pid if hwnd in self.windows else None

    def extended_style(self, hwnd: int) -> int:
        assert hwnd in self.windows
        if self.styles:
            self.current_style = self.styles.pop(0)
        return self.current_style

    def set_topmost_no_activate(self, hwnd: int) -> tuple[bool, int | None]:
        self.set_calls.append(hwnd)
        if self.set_result[0]:
            self.current_style |= desktop_window_zorder._WS_EX_TOPMOST
        return self.set_result


@pytest.mark.asyncio
async def test_windows_zorder_port_reasserts_bound_process_window_after_click_through() -> None:
    api = FakeWin32WindowApi()
    port = desktop_window_zorder.WindowsWindowZOrderPort(api=api)
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result == desktop_window_zorder.WindowZOrderResult(
        applied=True,
        reason="applied",
        click_through_confirmed=True,
        topmost_style_present=True,
    )
    assert api.window_queries == [4321]
    assert api.set_calls == [101]


@pytest.mark.asyncio
async def test_windows_zorder_port_waits_for_click_through_before_reasserting() -> None:
    api = FakeWin32WindowApi(
        styles=(0, 0, desktop_window_zorder._WS_EX_TRANSPARENT),
    )
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    port = desktop_window_zorder.WindowsWindowZOrderPort(
        api=api,
        timeout_s=0.5,
        poll_interval_s=0.01,
        sleep=record_sleep,
    )
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is True
    assert result.click_through_confirmed is True
    assert len(sleeps) == 2
    assert api.set_calls == [101]


@pytest.mark.asyncio
async def test_windows_zorder_port_reasserts_best_effort_when_click_through_is_unconfirmed() -> (
    None
):
    api = FakeWin32WindowApi(styles=(0,))
    port = desktop_window_zorder.WindowsWindowZOrderPort(api=api, timeout_s=0)
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is True
    assert result.reason == "applied_unconfirmed"
    assert result.click_through_confirmed is False
    assert api.set_calls == [101]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("windows", "expected_reason"),
    [
        ((), "window_not_found"),
        ((101, 102), "ambiguous_window"),
    ],
)
async def test_windows_zorder_port_rejects_missing_or_ambiguous_windows(
    windows: tuple[int, ...],
    expected_reason: str,
) -> None:
    api = FakeWin32WindowApi(windows=windows)
    port = desktop_window_zorder.WindowsWindowZOrderPort(api=api, timeout_s=0)
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is False
    assert result.reason == expected_reason
    assert api.set_calls == []


@pytest.mark.asyncio
async def test_windows_zorder_port_retries_transient_ambiguity() -> None:
    class TransientAmbiguityApi(FakeWin32WindowApi):
        def __init__(self) -> None:
            super().__init__(windows=(101, 102))
            self.batches = [(101, 102), (101,)]

        def top_level_windows_for_process(
            self, pid: int
        ) -> desktop_window_zorder.WindowEnumerationResult:
            self.window_queries.append(pid)
            self.windows = self.batches.pop(0)
            return desktop_window_zorder.WindowEnumerationResult(windows=self.windows)

        def extended_style(self, hwnd: int) -> int:
            return desktop_window_zorder._WS_EX_TRANSPARENT | (
                desktop_window_zorder._WS_EX_TOPMOST if self.set_calls else 0
            )

    api = TransientAmbiguityApi()
    port = desktop_window_zorder.WindowsWindowZOrderPort(api=api)
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is True
    assert api.window_queries == [4321, 4321]
    assert api.set_calls == [101]


@pytest.mark.asyncio
async def test_windows_zorder_port_reports_window_enumeration_failure() -> None:
    api = FakeWin32WindowApi(windows=(), enum_error=5)
    port = desktop_window_zorder.WindowsWindowZOrderPort(api=api)
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is False
    assert result.reason == "enum_windows_failed"
    assert result.win32_error == 5


@pytest.mark.asyncio
async def test_windows_zorder_port_revalidates_process_before_mutation() -> None:
    class ReusedWindowApi(FakeWin32WindowApi):
        def __init__(self) -> None:
            super().__init__()
            self.process_queries = 0

        def process_id(self, hwnd: int) -> int | None:
            self.process_queries += 1
            return 4321 if self.process_queries == 1 else 9999

    api = ReusedWindowApi()
    port = desktop_window_zorder.WindowsWindowZOrderPort(api=api)
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is False
    assert result.reason == "window_changed"
    assert api.set_calls == []


@pytest.mark.asyncio
async def test_windows_zorder_port_stops_when_closed_during_polling() -> None:
    api = FakeWin32WindowApi(styles=(0,))
    port: desktop_window_zorder.WindowsWindowZOrderPort

    async def close_during_sleep(_delay: float) -> None:
        port.close()

    port = desktop_window_zorder.WindowsWindowZOrderPort(
        api=api,
        sleep=close_during_sleep,
    )
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is False
    assert result.reason == "binding_changed"
    assert api.set_calls == []


@pytest.mark.asyncio
async def test_windows_zorder_port_reports_set_window_pos_failure() -> None:
    api = FakeWin32WindowApi(set_result=(False, 5))
    port = desktop_window_zorder.WindowsWindowZOrderPort(api=api)
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is False
    assert result.reason == "set_window_pos_failed"
    assert result.win32_error == 5


@pytest.mark.asyncio
async def test_windows_zorder_port_rejects_missing_topmost_style_after_success() -> None:
    class MissingTopmostApi(FakeWin32WindowApi):
        def set_topmost_no_activate(self, hwnd: int) -> tuple[bool, int | None]:
            self.set_calls.append(hwnd)
            return True, None

    api = MissingTopmostApi()
    port = desktop_window_zorder.WindowsWindowZOrderPort(api=api)
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is False
    assert result.reason == "topmost_style_missing"
    assert api.set_calls == [101]


@pytest.mark.asyncio
async def test_windows_zorder_port_polls_async_topmost_postcondition() -> None:
    class DelayedTopmostApi(FakeWin32WindowApi):
        def set_topmost_no_activate(self, hwnd: int) -> tuple[bool, int | None]:
            self.set_calls.append(hwnd)
            return True, None

    api = DelayedTopmostApi(
        styles=(
            desktop_window_zorder._WS_EX_TRANSPARENT,
            desktop_window_zorder._WS_EX_TRANSPARENT,
            desktop_window_zorder._WS_EX_TRANSPARENT | desktop_window_zorder._WS_EX_TOPMOST,
        )
    )
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    port = desktop_window_zorder.WindowsWindowZOrderPort(api=api, sleep=record_sleep)
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is True
    assert result.topmost_style_present is True
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_windows_zorder_port_close_discards_process_binding() -> None:
    api = FakeWin32WindowApi()
    port = desktop_window_zorder.WindowsWindowZOrderPort(api=api)
    port.bind_process(4321)
    port.close()

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is False
    assert result.reason == "closed"
    assert api.window_queries == []


@pytest.mark.asyncio
async def test_noop_zorder_port_is_immediate_and_unsupported() -> None:
    port = desktop_window_zorder.NoopWindowZOrderPort()
    port.bind_process(4321)

    result = await port.reassert_topmost_after_click_through()

    assert result.applied is False
    assert result.reason == "unsupported_platform"


def test_ctypes_win32_api_sets_topmost_without_moving_sizing_or_activating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeUser32:
        def SetWindowPos(self, *args: object) -> int:
            calls.append(args)
            return 1

    api = object.__new__(desktop_window_zorder._CtypesWin32WindowApi)
    api._user32 = FakeUser32()
    monkeypatch.setattr(
        desktop_window_zorder.ctypes,
        "set_last_error",
        lambda _value: None,
        raising=False,
    )
    monkeypatch.setattr(
        desktop_window_zorder.ctypes,
        "get_last_error",
        lambda: 0,
        raising=False,
    )

    result = api.set_topmost_no_activate(101)

    assert result == (True, None)
    assert len(calls) == 1
    hwnd, insert_after, x, y, width, height, flags = calls[0]
    assert hwnd == 101
    assert (
        insert_after.value
        == desktop_window_zorder.wintypes.HWND(desktop_window_zorder._HWND_TOPMOST).value
    )
    assert (x, y, width, height) == (0, 0, 0, 0)
    assert flags == (
        desktop_window_zorder._SWP_NOMOVE
        | desktop_window_zorder._SWP_NOSIZE
        | desktop_window_zorder._SWP_NOACTIVATE
        | desktop_window_zorder._SWP_ASYNCWINDOWPOS
    )
