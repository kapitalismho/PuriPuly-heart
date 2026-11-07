from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from puripuly_heart.core.osc.vrchat_osc_presence import probe_vrchat_osc_presence


def test_probe_vrchat_osc_presence_not_running() -> None:
    presence = probe_vrchat_osc_presence(
        port=9000,
        process_iter=lambda *_args, **_kwargs: [],
        net_connections=lambda *_args, **_kwargs: [],
    )
    assert presence.vrchat_running is False
    assert presence.osc_listening is None
    assert presence.should_prompt_enable_osc is False


def test_probe_vrchat_osc_presence_running_without_listener() -> None:
    processes = [SimpleNamespace(info={"pid": 11, "name": "VRChat.exe"})]
    presence = probe_vrchat_osc_presence(
        port=9000,
        process_iter=lambda *_args, **_kwargs: processes,
        net_connections=lambda *_args, **_kwargs: [],
    )
    assert presence.vrchat_running is True
    assert presence.osc_listening is False
    assert presence.should_prompt_enable_osc is True


def test_probe_vrchat_osc_presence_running_with_listener() -> None:
    processes = [SimpleNamespace(info={"pid": 22, "name": "VRChat.exe"})]
    connections = [SimpleNamespace(laddr=SimpleNamespace(ip="127.0.0.1", port=9000), pid=22)]
    presence = probe_vrchat_osc_presence(
        port=9000,
        process_iter=lambda *_args, **_kwargs: processes,
        net_connections=lambda *_args, **_kwargs: connections,
    )
    assert presence.vrchat_running is True
    assert presence.osc_listening is True
    assert presence.should_prompt_enable_osc is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process table")
def test_probe_vrchat_osc_presence_uses_windows_process_names(monkeypatch) -> None:
    from puripuly_heart.core.audio import windows_process_table

    monkeypatch.setattr(
        windows_process_table,
        "list_windows_processes",
        lambda: (
            windows_process_table.WindowsProcessEntry(pid=5, parent_pid=1, image_name="game.exe"),
            windows_process_table.WindowsProcessEntry(
                pid=33, parent_pid=1, image_name="VRChat.exe"
            ),
        ),
    )
    connections = [SimpleNamespace(laddr=SimpleNamespace(ip="0.0.0.0", port=9000), pid=33)]

    presence = probe_vrchat_osc_presence(
        port=9000,
        net_connections=lambda *_args, **_kwargs: connections,
    )

    assert presence.vrchat_running is True
    assert presence.osc_listening is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process table")
def test_probe_vrchat_osc_presence_preserves_unknown_process_list_failure(
    monkeypatch,
) -> None:
    from puripuly_heart.core.audio import windows_process_table

    def fail() -> tuple[windows_process_table.WindowsProcessEntry, ...]:
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(windows_process_table, "list_windows_processes", fail)

    presence = probe_vrchat_osc_presence(
        port=9000,
        net_connections=lambda *_args, **_kwargs: [],
    )

    assert presence.vrchat_running is None
    assert presence.osc_listening is None


@pytest.mark.parametrize(
    ("binding", "pid", "expected"),
    [
        ("127.0.0.1", 22, "matched"),
        ("0.0.0.0", 22, "matched"),
        ("127.0.0.1", 33, "other_process"),
        ("127.0.0.1", None, "unknown"),
        ("127.0.0.1", 0, "unknown"),
        ("127.0.0.2", 22, "not_listening"),
        ("192.0.2.20", 22, "not_listening"),
        ("::", 22, "unknown"),
        (None, 22, "unknown"),
    ],
)
def test_listener_match_requires_destination_address_and_known_vrchat_pid(
    binding, pid, expected
) -> None:
    presence = probe_vrchat_osc_presence(
        host="127.0.0.1",
        port=19000,
        process_iter=lambda *_args: [SimpleNamespace(info={"pid": 22, "name": "VRChat.exe"})],
        net_connections=lambda **_kwargs: [
            SimpleNamespace(laddr=SimpleNamespace(ip=binding, port=19000), pid=pid)
        ],
    )
    assert presence.listener_status == expected
    assert presence.osc_listening is (
        True if expected == "matched" else None if expected == "unknown" else False
    )


@pytest.mark.parametrize("host", ["192.0.2.30", "2001:db8::1"])
def test_remote_destination_is_not_applicable_without_process_or_socket_enumeration(host) -> None:
    def forbidden(*_args, **_kwargs):
        pytest.fail("Remote destination must not probe local listeners")

    presence = probe_vrchat_osc_presence(
        host=host, port=19000, process_iter=forbidden, net_connections=forbidden
    )
    assert presence.listener_status == "not_applicable"
    assert presence.vrchat_running is None
    assert presence.osc_listening is None


@pytest.mark.parametrize("boundary", ["processes", "connections", "iteration"])
def test_presence_observation_failure_remains_unknown(boundary) -> None:
    def fail(*_args, **_kwargs):
        raise PermissionError("controlled access denial")

    def broken_iteration(*_args, **_kwargs):
        yield SimpleNamespace(info={"pid": 22, "name": "VRChat.exe"})
        raise PermissionError("controlled partial enumeration")

    presence = probe_vrchat_osc_presence(
        port=19000,
        process_iter=(
            fail
            if boundary == "processes"
            else (
                broken_iteration
                if boundary == "iteration"
                else lambda *_args: [SimpleNamespace(info={"pid": 22, "name": "VRChat.exe"})]
            )
        ),
        net_connections=fail,
    )
    assert presence.vrchat_running is (True if boundary == "connections" else None)
    assert presence.osc_listening is None
    assert presence.listener_status == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unknown", "not_applicable", "matched", "other_process"])
async def test_presence_adapter_retains_diagnostic_observation_and_notice_uncertainty(
    status,
) -> None:
    from puripuly_heart.app.adapters.vrchat_osc_presence import PsutilVrchatOscPresenceAdapter
    from puripuly_heart.core.osc.vrchat_osc_presence_contract import VrchatOscPresence

    calls = []
    listening = True if status == "matched" else False if status == "other_process" else None

    def probe(*, host, port):
        calls.append((host, port))
        return VrchatOscPresence(True, listening, status)

    adapter = PsutilVrchatOscPresenceAdapter(probe=probe)
    observation = await adapter.observe("127.0.0.2", 19000)
    assert observation.listener_status == status
    assert await adapter.should_prompt_enable_osc(port=19001) is (
        False if status == "matched" else True if status == "other_process" else None
    )
    assert calls == [("127.0.0.2", 19000), ("127.0.0.1", 19001)]
