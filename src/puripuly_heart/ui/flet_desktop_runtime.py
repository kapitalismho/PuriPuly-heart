from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
from collections.abc import Callable, Iterator

REQUIRED_FLET_DESKTOP_HOOKS = (
    "__locate_and_unpack_flet_view",
    "open_flet_view_async",
)


class UnsupportedFletDesktopRuntimeError(RuntimeError):
    pass


def _installed_flet_desktop_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("flet-desktop")
    except PackageNotFoundError:
        return "unknown"


def require_flet_desktop_hooks(module: object) -> None:
    missing = [
        name for name in REQUIRED_FLET_DESKTOP_HOOKS if not callable(getattr(module, name, None))
    ]
    if not missing:
        return
    raise UnsupportedFletDesktopRuntimeError(
        f"flet_desktop {_installed_flet_desktop_version()} does not expose "
        f"{', '.join(missing)}, which the desktop overlay needs to launch its hidden view "
        "and own the view process. Pin flet-desktop to a supported version or update "
        "puripuly_heart.ui.flet_desktop_runtime."
    )


async def open_hidden_view(
    page_url: str,
    assets_dir: str | None,
    hidden: bool,
) -> tuple[asyncio.subprocess.Process, str | None]:
    import flet_desktop

    require_flet_desktop_hooks(flet_desktop)
    locate_view = getattr(flet_desktop, "__locate_and_unpack_flet_view")
    args, flet_env, pid_file = locate_view(page_url, assets_dir, hidden and os.name != "nt")
    kwargs: dict[str, object] = {"env": flet_env}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    return (
        await asyncio.create_subprocess_exec(args[0], *args[1:], **kwargs),
        pid_file,
    )


@contextlib.contextmanager
def patch_hidden_view_launcher(
    *,
    on_process_started: Callable[[int, str | None], None] | None = None,
) -> Iterator[None]:
    import flet_desktop

    require_flet_desktop_hooks(flet_desktop)

    async def launch(
        page_url: str,
        assets_dir: str | None,
        hidden: bool,
    ) -> tuple[asyncio.subprocess.Process, str | None]:
        process, pid_file = await open_hidden_view(page_url, assets_dir, hidden)
        if on_process_started is not None and process.pid is not None:
            on_process_started(int(process.pid), pid_file)
        return process, pid_file

    original = flet_desktop.open_flet_view_async
    flet_desktop.open_flet_view_async = launch
    try:
        yield
    finally:
        flet_desktop.open_flet_view_async = original
