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


def test_windows_snapshots_project_process_records() -> None:
    from puripuly_heart.core.audio.windows_process_table import WindowsProcessRecord

    records = (
        WindowsProcessRecord(
            pid=10,
            parent_pid=0,
            image_name="one.exe",
            executable_path=r"C:\Apps\one.exe",
            is_current_user=True,
            create_time=1.25,
        ),
        WindowsProcessRecord(
            pid=20,
            parent_pid=10,
            image_name="two.exe",
            executable_path=r"C:\Apps\two.exe",
            is_current_user=False,
            create_time=None,
        ),
        WindowsProcessRecord(
            pid=30,
            parent_pid=10,
            image_name="Secure System",
            executable_path=None,
            is_current_user=False,
            create_time=3.5,
        ),
    )

    snapshots = tuple(
        process_identity.WindowsCurrentUserProcessSnapshots(scan=lambda: records).snapshots()
    )

    assert [(item.pid, item.parent_pid) for item in snapshots] == [
        (10, None),
        (20, 10),
        (30, 10),
    ]
    assert [item.is_current_user for item in snapshots] == [True, False, False]
    assert [item.executable_path for item in snapshots] == [
        r"C:\Apps\one.exe",
        r"C:\Apps\two.exe",
        None,
    ]
    assert [item.instance_id for item in snapshots] == ["10:1.25", None, None]


def test_default_snapshots_follow_platform(monkeypatch) -> None:
    monkeypatch.setattr(process_identity.sys, "platform", "win32")
    assert isinstance(
        process_identity.default_current_user_process_snapshots(),
        process_identity.WindowsCurrentUserProcessSnapshots,
    )

    monkeypatch.setattr(process_identity.sys, "platform", "linux")
    assert isinstance(
        process_identity.default_current_user_process_snapshots(),
        process_identity.PsutilCurrentUserProcessSnapshots,
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
