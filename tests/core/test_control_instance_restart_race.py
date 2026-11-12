from __future__ import annotations

import sys
import threading
import types
import uuid
from pathlib import Path
from typing import ClassVar

import pytest

if sys.platform != "win32":
    sys.modules.setdefault("msvcrt", types.ModuleType("msvcrt"))

from puripuly_heart.core import control_instance
from puripuly_heart.core.control_instance import InstanceRecord, acquire, discover


class FakeWindowsSecurity:
    _guard: ClassVar = threading.Lock()
    _locked_paths: ClassVar[set[str]] = set()

    def close(self) -> None:
        pass

    def protect(self, path: Path) -> None:
        pass

    def verify_path(self, path: Path) -> None:
        pass

    def open_record(self, path: Path):
        return path.open("r", encoding="utf-8")

    def lock(self, stream) -> bool:
        path = str(stream.name)
        with self._guard:
            if path in self._locked_paths:
                return False
            self._locked_paths.add(path)
            return True

    def unlock(self, stream) -> None:
        with self._guard:
            self._locked_paths.discard(str(stream.name))


@pytest.mark.parametrize("same_process", [True, False])
def test_discover_does_not_return_replaced_owner_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_process: bool
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(control_instance, "_WindowsSecurity", FakeWindowsSecurity)
    settings = tmp_path / "settings.json"
    first_id = str(uuid.uuid4())
    second_id = str(uuid.uuid4())
    first = acquire(settings)
    second = None
    first.publish(port=23456, token="a" * 43, instance_id=first_id)
    original_record = control_instance._record
    replaced = False

    def replace_owner_after_record_read(path, security, *, temporary=False):
        nonlocal second, replaced
        record = original_record(path, security, temporary=temporary)
        if not temporary and not replaced and path == first._path:
            replaced = True
            first.close()
            second = acquire(settings)
            second.publish(port=23457, token="b" * 43, instance_id=second_id)
            if not same_process:
                with control_instance._local_leases_lock:
                    control_instance._local_leases.discard(str(second._path))
        return record

    monkeypatch.setattr(control_instance, "_record", replace_owner_after_record_read)
    try:
        assert discover(settings) == InstanceRecord(23457, "b" * 43, second_id)
        assert replaced
    finally:
        first.close()
        if second is not None:
            second.close()


@pytest.mark.parametrize("same_process", [True, False])
def test_discover_returns_none_when_owner_closes_before_current_record_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_process: bool
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(control_instance, "_WindowsSecurity", FakeWindowsSecurity)
    settings = tmp_path / "settings.json"
    lease = acquire(settings)
    lease.publish(port=23456, token="a" * 43, instance_id=str(uuid.uuid4()))
    original_record = control_instance._record
    removed = False

    def remove_record_after_stale_read(path, security, *, temporary=False):
        nonlocal removed
        record = original_record(path, security, temporary=temporary)
        if not temporary and not removed and path == lease._path:
            removed = True
            lease._path.unlink()
            if not same_process:
                with control_instance._local_leases_lock:
                    control_instance._local_leases.discard(str(lease._path))
        return record

    monkeypatch.setattr(control_instance, "_record", remove_record_after_stale_read)
    try:
        assert discover(settings) is None
        assert removed
    finally:
        lease.close()


@pytest.mark.parametrize("same_process", [True, False])
def test_discover_still_rejects_invalid_current_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_process: bool
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(control_instance, "_WindowsSecurity", FakeWindowsSecurity)
    settings = tmp_path / "settings.json"
    lease = acquire(settings)
    lease.publish(port=23456, token="a" * 43, instance_id=str(uuid.uuid4()))
    original_record = control_instance._record
    initial_read = True

    def invalidate_after_initial_read(path, security, *, temporary=False):
        nonlocal initial_read
        if initial_read and not temporary:
            initial_read = False
            return original_record(path, security, temporary=temporary)
        raise control_instance.InstanceSecurityError("synthetic invalid current record")

    if not same_process:
        with control_instance._local_leases_lock:
            control_instance._local_leases.discard(str(lease._path))
    monkeypatch.setattr(control_instance, "_record", invalidate_after_initial_read)
    try:
        with pytest.raises(control_instance.InstanceSecurityError):
            discover(settings)
    finally:
        monkeypatch.setattr(control_instance, "_record", original_record)
        lease.close()
