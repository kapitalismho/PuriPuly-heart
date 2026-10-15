from __future__ import annotations

import ctypes
import errno
import json
import os
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path

import pytest

from puripuly_heart.core import control_instance
from puripuly_heart.core.control_instance import (
    InstanceAlreadyRunning,
    InstanceSecurityError,
    acquire,
    discover,
    discover_instances,
)

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows ACL and file-lock contract")


def test_aliases_duplicate_owner_and_restart_changes_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "session" / "settings.json"
    settings.parent.mkdir()
    alias = settings.parent / ".." / "session" / "settings.json"
    first = str(uuid.uuid4())
    second = str(uuid.uuid4())
    with acquire(settings) as lease:
        assert discover(alias) is None
        with pytest.raises(InstanceAlreadyRunning):
            acquire(alias)
        lease.publish(port=51234, token="t" * 43, instance_id=first)
        record = discover(alias)
        assert record is not None
        assert (record.port, record.token, record.instance_id) == (51234, "t" * 43, first)
        assert record.__dataclass_params__.frozen
        assert discover_instances() == [
            {"settings_path": str(settings.resolve()).lower(), "port": 51234, "instance_id": first}
        ]
    assert discover(settings) is None
    assert discover_instances() == []
    with acquire(alias) as lease:
        lease.publish(port=51235, token="u" * 43, instance_id=second)
        assert discover(settings).instance_id == second
        assert discover_instances()[0]["instance_id"] == second
    assert discover(settings) is None


def test_cross_thread_close_releases_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    lease = acquire(settings)
    lease.publish(port=23456, token="x" * 43, instance_id=str(uuid.uuid4()))
    failures: list[BaseException] = []

    def close_on_worker() -> None:
        try:
            lease.close()
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=close_on_worker)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert not failures
    assert discover(settings) is None
    with acquire(settings) as successor:
        successor.publish(port=23457, token="y" * 43, instance_id=str(uuid.uuid4()))
        assert discover(settings).port == 23457


def test_unsafe_lock_acl_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    with acquire(settings) as lease:
        lease.publish(port=23456, token="x" * 43, instance_id=str(uuid.uuid4()))
        lock = lease._path.with_suffix(".lock")
        try:
            subprocess.run(
                ["icacls", str(lock), "/grant", "*S-1-1-0:R"], check=True, capture_output=True
            )
            with pytest.raises(InstanceSecurityError):
                discover(settings)
        finally:
            lease._security.protect(lock)


def test_existing_parent_can_have_readable_inherited_acl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    parent = tmp_path / "puripuly-heart"
    parent.mkdir()
    subprocess.run(["icacls", str(parent), "/grant", "*S-1-1-0:R"], check=True, capture_output=True)
    settings = tmp_path / "settings.json"
    with acquire(settings) as lease:
        lease.publish(port=23456, token="x" * 43, instance_id=str(uuid.uuid4()))
        assert discover(settings).port == 23456
        assert discover_instances()[0]["port"] == 23456


def test_cross_process_duplicate_and_crash_are_not_live_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    script = (
        "import os,sys; from pathlib import Path; "
        "from puripuly_heart.core.control_instance import acquire; "
        "l=acquire(Path(sys.argv[1])); "
        "l.publish(port=23456,token='a'*43,instance_id='12345678-1234-4234-9234-123456789012'); "
        "print('ready',flush=True); sys.stdin.readline(); os._exit(0)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(settings)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        assert discover(settings).port == 23456
        with pytest.raises(InstanceAlreadyRunning):
            acquire(settings)
        child.stdin.write("exit\n")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        assert discover(settings) is None
        assert discover_instances() == []
        with acquire(settings) as lease:
            lease.publish(port=23457, token="b" * 43, instance_id=str(uuid.uuid4()))
            assert discover(settings).port == 23457
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_unsafe_acl_and_reparse_record_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    with acquire(settings) as lease:
        lease.publish(port=23456, token="c" * 43, instance_id=str(uuid.uuid4()))
        record = lease._path
        try:
            subprocess.run(
                ["icacls", str(record), "/grant", "*S-1-1-0:R"], check=True, capture_output=True
            )
            with pytest.raises(InstanceSecurityError):
                discover(settings)
            assert discover_instances() == []
        finally:
            lease._security.protect(record)
    with acquire(settings) as lease:
        lease.publish(port=23457, token="d" * 43, instance_id=str(uuid.uuid4()))
        record = lease._path
        record.unlink()
        target = tmp_path / "attacker.json"
        target.write_text(
            json.dumps({"port": 9999, "token": "z" * 43, "instance_id": str(uuid.uuid4())})
        )
        os.symlink(target, record)
        with pytest.raises(InstanceSecurityError):
            discover(settings)
        assert discover_instances() == []
        record.unlink()


def test_directory_acl_must_remain_user_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    with acquire(settings) as lease:
        lease.publish(port=23456, token="f" * 43, instance_id=str(uuid.uuid4()))
        root = lease._path.parent
        try:
            subprocess.run(
                ["icacls", str(root), "/grant", "*S-1-1-0:R"], check=True, capture_output=True
            )
            with pytest.raises(InstanceSecurityError):
                discover(settings)
            with pytest.raises(InstanceSecurityError):
                discover_instances()
        finally:
            lease._security.protect(root)


def test_anonymous_windows_token_cannot_open_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    script = (
        "import ctypes,sys; "
        "k=ctypes.WinDLL('kernel32',use_last_error=True); "
        "a=ctypes.WinDLL('advapi32',use_last_error=True); "
        "k.GetCurrentThread.restype=ctypes.c_void_p; "
        "a.ImpersonateAnonymousToken.argtypes=[ctypes.c_void_p]; "
        "assert a.ImpersonateAnonymousToken(k.GetCurrentThread()); "
        "k.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]; "
        "k.CreateFileW.restype=ctypes.c_void_p; "
        "handle=k.CreateFileW(sys.argv[1],0x80000000,1,None,3,0,None); "
        "error=ctypes.get_last_error(); "
        "assert handle == ctypes.c_void_p(-1).value and error == 5,(handle,error); "
        "assert a.RevertToSelf()"
    )
    with acquire(settings) as lease:
        lease.publish(port=23456, token="e" * 43, instance_id=str(uuid.uuid4()))
        result = subprocess.run(
            [sys.executable, "-c", script, str(lease._path)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("action", ["close", "replace"])
def test_record_reader_preserves_owner_cleanup_and_failed_republication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    first = str(uuid.uuid4())
    second = str(uuid.uuid4())
    with acquire(settings) as lease:
        lease.publish(port=23456, token="a" * 43, instance_id=first)
        with lease._security.open_record(lease._path) as reader:
            assert not os.get_inheritable(reader.fileno())
            if action == "close":
                lease.close()
                assert discover(settings) is None
            else:
                with lease._path.open("r", encoding="utf-8") as conflicting_reader:
                    with pytest.raises(PermissionError):
                        lease.publish(port=23457, token="b" * 43, instance_id=second)
                    assert json.load(conflicting_reader)["instance_id"] == first
                assert discover(settings).instance_id == first
                assert not list(lease._path.parent.glob("*.tmp"))
            assert json.load(reader)["instance_id"] == first
        assert reader.closed
        if action == "replace":
            lease.publish(port=23457, token="b" * 43, instance_id=second)
            assert discover(settings).instance_id == second
    assert discover(settings) is None
    with acquire(settings) as successor:
        successor.publish(port=23458, token="c" * 43, instance_id=str(uuid.uuid4()))
        assert discover(settings).port == 23458


@pytest.mark.parametrize(
    ("query", "pause_on_read"),
    [(discover, 1), (discover_instances, 1), (discover_instances, 2)],
)
def test_inflight_discovery_read_cannot_fail_owner_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, query, pause_on_read: int
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    lease = acquire(settings)
    lease.publish(port=23456, token="a" * 43, instance_id=str(uuid.uuid4()))
    original_open = control_instance._WindowsSecurity.open_record
    entered = threading.Event()
    release = threading.Event()
    results = []
    failures = []
    reads = 0

    @contextmanager
    def gated_open(security, path):
        nonlocal reads
        with original_open(security, path) as reader:
            if threading.current_thread() is worker:
                reads += 1
                if reads == pause_on_read:
                    entered.set()
                    assert release.wait(5)
            yield reader

    def observe():
        try:
            results.append(query(settings) if query is discover else query())
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(control_instance._WindowsSecurity, "open_record", gated_open)
    worker = threading.Thread(target=observe)
    worker.start()
    try:
        assert entered.wait(3)
        assert discover(settings).port == 23456
        lease.close()
        release.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert not failures
        assert results == ([None] if query is discover else [[]])
        assert discover(settings) is None
    finally:
        release.set()
        worker.join(timeout=5)
        lease.close()


@pytest.mark.parametrize(
    "invalid",
    [
        b"\xff",
        b"{",
        b"x" * 4097,
        b"[]",
        {"port": True},
        {"token": "short"},
        {"instance_id": "invalid"},
        {"settings_path": "relative.json"},
        {"extra": True},
        "different_settings_identity",
    ],
)
def test_shared_record_reads_preserve_validation_and_release_failed_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    with acquire(settings) as lease:
        lease.publish(port=23456, token="a" * 43, instance_id=str(uuid.uuid4()))
        original = lease._path.read_bytes()
        payload = json.loads(original)
        if isinstance(invalid, bytes):
            malformed = invalid
        else:
            if isinstance(invalid, dict):
                payload.update(invalid)
            else:
                payload["settings_path"] = os.path.normcase(
                    str((tmp_path / "different.json").resolve())
                )
            malformed = json.dumps(payload).encode()
        try:
            lease._path.write_bytes(malformed)
            with pytest.raises(InstanceSecurityError):
                discover(settings)
            assert discover_instances() == []
        finally:
            lease._path.write_bytes(original)
        assert discover(settings).port == 23456
    assert discover(settings) is None


def test_missing_record_read_preserves_file_not_found_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    with acquire(settings) as lease:
        with pytest.raises(FileNotFoundError):
            lease._security.open_record(lease._path)
        lease.publish(port=23456, token="a" * 43, instance_id=str(uuid.uuid4()))
        lease._path.unlink()
        assert discover(settings) is None


@pytest.mark.parametrize("stage", ["descriptor", "stream"])
def test_record_reader_construction_failure_releases_native_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = tmp_path / "settings.json"
    resources = []

    def fail(resource, *_args, **_kwargs):
        resources.append(resource)
        raise OSError(errno.EMFILE, "synthetic reader construction failure")

    with acquire(settings) as lease:
        lease.publish(port=23456, token="a" * 43, instance_id=str(uuid.uuid4()))
        with monkeypatch.context() as scoped:
            if stage == "descriptor":
                scoped.setattr(control_instance.msvcrt, "open_osfhandle", fail)
            else:
                scoped.setattr(control_instance.os, "fdopen", fail)
            with pytest.raises(OSError) as failure:
                lease._security.open_record(lease._path)
            assert failure.value.errno == errno.EMFILE
        assert len(resources) == 1
        if stage == "descriptor":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetHandleInformation.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(wintypes.DWORD),
            ]
            flags = wintypes.DWORD()
            assert not kernel.GetHandleInformation(resources[0], ctypes.byref(flags))
            assert ctypes.get_last_error() == 6
        else:
            with pytest.raises(OSError) as closed:
                os.fstat(resources[0])
            assert closed.value.errno == errno.EBADF
        assert discover(settings).port == 23456
