from types import SimpleNamespace

from puripuly_heart.core.audio import process_identity


class FakeProcessError(Exception):
    pass


class FakePlatform:
    def __init__(self) -> None:
        self.calls = 0

    def ppid_map(self) -> dict[int, int]:
        self.calls += 1
        return {10: 1, 20: 10}


class FakePsutil:
    AccessDenied = FakeProcessError
    NoSuchProcess = FakeProcessError
    ZombieProcess = FakeProcessError

    def __init__(self) -> None:
        self._psplatform = FakePlatform()
        self.requested_attrs = None

    def Process(self):
        return SimpleNamespace(username=lambda: "desktop\\owner")

    def process_iter(self, attrs):
        self.requested_attrs = attrs
        return (
            SimpleNamespace(
                info={
                    "pid": 10,
                    "exe": r"C:\Apps\one.exe",
                    "username": "desktop\\owner",
                    "create_time": 1.25,
                }
            ),
            SimpleNamespace(
                info={
                    "pid": 20,
                    "exe": r"C:\Apps\two.exe",
                    "username": "desktop\\other",
                    "create_time": 2.5,
                }
            ),
        )


def test_snapshot_scan_uses_one_bulk_parent_lookup(monkeypatch) -> None:
    psutil = FakePsutil()
    monkeypatch.setattr(process_identity, "_import_psutil", lambda: psutil)

    snapshots = tuple(process_identity.PsutilCurrentUserProcessSnapshots().snapshots())

    assert psutil._psplatform.calls == 1
    assert psutil.requested_attrs == ["pid", "exe", "username", "create_time"]
    assert [(item.pid, item.parent_pid) for item in snapshots] == [(10, 1), (20, 10)]
    assert [item.is_current_user for item in snapshots] == [True, False]
    assert [item.instance_id for item in snapshots] == ["10:1.25", "20:2.5"]
