from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class VrchatOscPresence:
    vrchat_running: bool
    osc_listening: bool | None

    @property
    def should_prompt_enable_osc(self) -> bool:
        return self.vrchat_running and self.osc_listening is False


_VRCHAT_PROCESS_NAMES = frozenset({"vrchat.exe"})


def probe_vrchat_osc_presence(
    *,
    port: int = 9000,
    process_iter: Callable[..., object] | None = None,
    net_connections: Callable[..., object] | None = None,
) -> VrchatOscPresence:
    if port <= 0 or port > 65535:
        return VrchatOscPresence(vrchat_running=False, osc_listening=None)
    if process_iter is None and net_connections is None and sys.platform != "win32":
        return VrchatOscPresence(vrchat_running=False, osc_listening=None)

    if process_iter is None or net_connections is None:
        try:
            psutil = _import_psutil()
        except Exception:
            return VrchatOscPresence(vrchat_running=False, osc_listening=None)
        iter_processes = process_iter or psutil.process_iter
        list_connections = net_connections or psutil.net_connections
    else:
        iter_processes = process_iter
        list_connections = net_connections

    if process_iter is None and sys.platform == "win32":
        vrchat_pids = _collect_windows_vrchat_pids()
    else:
        vrchat_pids = _collect_vrchat_pids(iter_processes)
    if not vrchat_pids:
        return VrchatOscPresence(vrchat_running=False, osc_listening=None)

    try:
        connections = list_connections(kind="udp")
    except Exception:
        return VrchatOscPresence(vrchat_running=True, osc_listening=None)

    for connection in connections:
        try:
            laddr = getattr(connection, "laddr", None)
            if laddr is None:
                continue
            connection_port = getattr(laddr, "port", None)
            if connection_port is None and isinstance(laddr, (tuple, list)) and laddr:
                connection_port = laddr[-1]
            if connection_port != port:
                continue
            pid = getattr(connection, "pid", None)
            if isinstance(pid, int) and pid in vrchat_pids:
                return VrchatOscPresence(vrchat_running=True, osc_listening=True)
            if pid is None:
                # Port is bound but pid unavailable; treat as listening when VRChat is running.
                return VrchatOscPresence(vrchat_running=True, osc_listening=True)
        except Exception:
            continue

    return VrchatOscPresence(vrchat_running=True, osc_listening=False)


def _collect_vrchat_pids(process_iter: Callable[..., object]) -> set[int]:
    pids: set[int] = set()
    try:
        iterator = process_iter(["pid", "name"])
    except TypeError:
        try:
            iterator = process_iter()
        except Exception:
            return pids
    except Exception:
        return pids

    for process in iterator:
        try:
            info = getattr(process, "info", None)
            if isinstance(info, dict):
                name = info.get("name")
                pid = info.get("pid")
            else:
                name = process.name() if callable(getattr(process, "name", None)) else None
                pid = getattr(process, "pid", None)
            if not isinstance(name, str) or not isinstance(pid, int):
                continue
            if name.casefold() in _VRCHAT_PROCESS_NAMES:
                pids.add(pid)
        except Exception:
            continue
    return pids


def _collect_windows_vrchat_pids() -> set[int]:
    from puripuly_heart.core.audio.windows_process_table import list_windows_processes

    try:
        entries = list_windows_processes()
    except OSError:
        return set()
    return {entry.pid for entry in entries if entry.image_name.casefold() in _VRCHAT_PROCESS_NAMES}


def _import_psutil():  # noqa: ANN201
    import psutil

    return psutil
