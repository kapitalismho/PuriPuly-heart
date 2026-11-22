"""Run the source GUI and print timing-only History receipts to stderr."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from itertools import count
from pathlib import Path
from time import perf_counter
from unittest.mock import patch


@contextmanager
def trace_history_entries():
    import flet as ft

    from puripuly_heart.ui.views.agent_chat import AgentChatView

    create_view = AgentChatView.__init__
    open_history = AgentChatView._open_history
    close_history = AgentChatView._close_history
    scroll_to = ft.ListView.scroll_to
    sequence = count(1)
    pending = {}
    started = perf_counter()

    def created(view, *args, **kwargs):
        create_view(view, *args, **kwargs)
        on_press = view._history_tile.on_tap_down

        def pressed(event):
            elapsed = (perf_counter() - started) * 1000
            print(f"HISTORY_TIMING press_at_ms={elapsed:.1f}", file=sys.stderr, flush=True)
            return on_press(event)

        view._history_tile.on_tap_down = pressed

    def opened(view):
        key = id(view._history_list)
        entry = (next(sequence), perf_counter())
        pending[key] = entry
        try:
            open_history(view)
        except BaseException:
            pending.pop(key, None)
            raise
        elapsed = (perf_counter() - entry[1]) * 1000
        print(
            f"HISTORY_TIMING click={entry[0]} attached_rows={len(view._history_list.controls)} "
            f"handler_start_at_ms={(entry[1] - started) * 1000:.1f} handler_ms={elapsed:.1f}",
            file=sys.stderr,
            flush=True,
        )

    def closed(view):
        pending.pop(id(view._history_list), None)
        return close_history(view)

    async def scrolled(listing, *args, **kwargs):
        key = id(listing)
        entry = pending.get(key)
        result = await scroll_to(listing, *args, **kwargs)
        if entry is not None and pending.get(key) is entry:
            del pending[key]
            elapsed = (perf_counter() - entry[1]) * 1000
            print(
                f"HISTORY_TIMING click={entry[0]} client_scroll_reply_ms={elapsed:.1f}",
                file=sys.stderr,
                flush=True,
            )
        return result

    with (
        patch.object(AgentChatView, "__init__", created),
        patch.object(AgentChatView, "_open_history", opened),
        patch.object(AgentChatView, "_close_history", closed),
        patch.object(ft.ListView, "scroll_to", scrolled),
    ):
        yield


if __name__ == "__main__":
    source = Path.cwd() / "src"
    if not (source / "puripuly_heart" / "main.py").is_file():
        sys.exit("Start this probe from the source checkout you want to measure.")
    sys.path.insert(0, str(source))
    from puripuly_heart.main import main

    with trace_history_entries():
        raise SystemExit(main([*sys.argv[1:], "run-gui"]))
