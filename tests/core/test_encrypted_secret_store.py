from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import pytest

from puripuly_heart.core.storage import secrets
from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore


def test_retained_stores_observe_new_keys_rotations_and_deletions(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    retained = EncryptedFileSecretStore(path, passphrase="synthetic")
    writer = EncryptedFileSecretStore(path, passphrase="synthetic")
    writer.set("openrouter_managed_qq_api_key", "managed-new")
    assert retained.get("openrouter_managed_qq_api_key") == "managed-new"
    writer.set("openrouter_managed_qq_api_key", "managed-rotated")
    assert retained.get("openrouter_managed_qq_api_key") == "managed-rotated"
    writer.delete("openrouter_managed_qq_api_key")
    assert retained.get("openrouter_managed_qq_api_key") is None


def test_stale_writer_and_deleter_preserve_new_managed_authority(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    older = EncryptedFileSecretStore(path, passphrase="synthetic")
    older.set("unrelated", "keep")
    older.set("obsolete", "remove")
    newer = EncryptedFileSecretStore(path, passphrase="synthetic")
    expected = {
        "openrouter_managed_qq_api_key": "managed-new",
        "openrouter_managed_qq_delivery_ack_token": "ack-new",
        "openrouter_managed_operation_resume_token": "resume-new",
        "unrelated": "keep",
        "chatgpt_refresh_token": "rotated",
    }
    for key, value in expected.items():
        if key.startswith("openrouter_"):
            newer.set(key, value)
    older.set("chatgpt_refresh_token", "rotated")
    older.delete("obsolete")
    reloaded = EncryptedFileSecretStore(path, passphrase="synthetic")
    assert {key: reloaded.get(key) for key in expected} == expected
    assert reloaded.get("obsolete") is None


def test_creation_serializes_salt_selection_across_instances(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "secrets.json"
    writing = threading.Event()
    release = threading.Event()
    attempted = threading.Event()
    original = secrets._atomic_write_json

    def gated_write(target: Path, data: object) -> None:
        if threading.current_thread().name.startswith("first"):
            writing.set()
            assert release.wait(10)
        original(target, data)

    def second_open() -> EncryptedFileSecretStore:
        attempted.set()
        return EncryptedFileSecretStore(path, passphrase="synthetic")

    monkeypatch.setattr(secrets, "_atomic_write_json", gated_write)
    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="first") as first_pool,
        ThreadPoolExecutor(max_workers=1) as second_pool,
    ):
        opening = first_pool.submit(EncryptedFileSecretStore, path, passphrase="synthetic")
        try:
            assert writing.wait(10)
            other_opening = second_pool.submit(second_open)
            assert attempted.wait(10)
            with pytest.raises(TimeoutError):
                other_opening.result(timeout=0.1)
        finally:
            release.set()
        first = opening.result(timeout=10)
        second = other_opening.result(timeout=10)
    first.set("discord", "first-secret")
    second.set("qq", "second-secret")
    fresh = EncryptedFileSecretStore(path, passphrase="synthetic")
    assert fresh.get("discord") == "first-secret"
    assert fresh.get("qq") == "second-secret"


def test_overlapping_instances_serialize_read_modify_replace(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "secrets.json"
    first = EncryptedFileSecretStore(path, passphrase="synthetic")
    second = EncryptedFileSecretStore(path, passphrase="synthetic")
    first.set("unrelated", "keep")
    writing = threading.Event()
    release = threading.Event()
    attempted = threading.Event()
    original = secrets._atomic_write_json

    def gated_write(target: Path, data: object) -> None:
        if threading.current_thread().name.startswith("first"):
            writing.set()
            assert release.wait(10)
        original(target, data)

    def second_write() -> None:
        attempted.set()
        second.set("resume", "resume-new")

    monkeypatch.setattr(secrets, "_atomic_write_json", gated_write)
    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="first") as first_pool,
        ThreadPoolExecutor(max_workers=1) as second_pool,
    ):
        pending_first = first_pool.submit(first.set, "managed", "managed-new")
        try:
            assert writing.wait(10)
            pending_second = second_pool.submit(second_write)
            assert attempted.wait(10)
            with pytest.raises(TimeoutError):
                pending_second.result(timeout=0.1)
        finally:
            release.set()
        pending_first.result(timeout=10)
        pending_second.result(timeout=10)
    reloaded = EncryptedFileSecretStore(path, passphrase="synthetic")
    assert reloaded.get("managed") == "managed-new"
    assert reloaded.get("resume") == "resume-new"
    assert reloaded.get("unrelated") == "keep"


_PROCESS_WRITER = """
import json
import sys
from pathlib import Path
from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore

path, prefix = sys.argv[1:]
print("ready", flush=True)
sys.stdin.readline()
store = EncryptedFileSecretStore(Path(path), passphrase="synthetic")
print("opened", flush=True)
sys.stdin.readline()
for index in range(12):
    store.set(f"{prefix}-{index}", f"value-{prefix}-{index}")
store.set(f"{prefix}-obsolete", "remove")
store.delete(f"{prefix}-obsolete")
print(json.dumps({"salt": json.loads(Path(path).read_text())["salt"]}), flush=True)
"""


@pytest.mark.parametrize("preexisting", [False, True])
def test_concurrent_process_creation_and_mutation_survive_fresh_process_reload(
    tmp_path: Path, preexisting: bool
) -> None:
    path = tmp_path / "secrets.json"
    if preexisting:
        initial = EncryptedFileSecretStore(path, passphrase="synthetic")
        initial.set("unrelated", "keep")
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _PROCESS_WRITER, str(path), prefix],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for prefix in ("discord", "qq", "chatgpt")
    ]
    try:
        for child in children:
            assert child.stdout.readline().strip() == "ready"
        for child in children:
            child.stdin.write("open\n")
            child.stdin.flush()
        for child in children:
            assert child.stdout.readline().strip() == "opened"
        for child in children:
            child.stdin.write("write\n")
            child.stdin.flush()
        salts = []
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stderr
            salts.append(json.loads(stdout)["salt"])
        assert len(set(salts)) == 1
        expected = {
            f"{prefix}-{index}": f"value-{prefix}-{index}"
            for prefix in ("discord", "qq", "chatgpt")
            for index in range(12)
        }
        if preexisting:
            expected["unrelated"] = "keep"
        reload_code = """
import json
import sys
from pathlib import Path
from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore
store = EncryptedFileSecretStore(Path(sys.argv[1]), passphrase="synthetic")
keys = json.loads(sys.argv[2])
print(json.dumps({key: store.get(key) for key in keys}))
"""
        keys = [*expected, *(f"{prefix}-obsolete" for prefix in ("discord", "qq", "chatgpt"))]
        reloaded = subprocess.run(
            [sys.executable, "-c", reload_code, str(path), json.dumps(keys)],
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
            check=True,
        )
        assert json.loads(reloaded.stdout) == {
            **expected,
            "discord-obsolete": None,
            "qq-obsolete": None,
            "chatgpt-obsolete": None,
        }
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=10)
            for stream in (child.stdin, child.stdout, child.stderr):
                stream.close()


@pytest.mark.parametrize("operation", ["get", "set", "delete"])
def test_retained_store_rejects_changed_salt_without_touching_replacement(
    tmp_path: Path, operation: str
) -> None:
    path = tmp_path / "secrets.json"
    retained = EncryptedFileSecretStore(path, passphrase="synthetic")
    replacement_path = tmp_path / "replacement.json"
    replacement = EncryptedFileSecretStore(replacement_path, passphrase="synthetic")
    replacement.set("managed", "replacement-secret")
    replacement_path.replace(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="salt changed"):
        if operation == "set":
            retained.set("managed", "old-instance")
        else:
            getattr(retained, operation)("managed")
    assert path.read_bytes() == before
    assert EncryptedFileSecretStore(path, passphrase="synthetic").get("managed") == (
        "replacement-secret"
    )


@pytest.mark.parametrize("operation", ["get", "set", "delete"])
@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_read_failures_are_not_missing_secrets_or_permission_to_overwrite(
    tmp_path: Path, operation: str, damage: str
) -> None:
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="synthetic")
    store.set("managed", "keep")
    if damage == "missing":
        path.unlink()
        expected_error = FileNotFoundError
    else:
        path.write_text("not-json", encoding="utf-8")
        expected_error = json.JSONDecodeError
    with pytest.raises(expected_error):
        if operation == "set":
            store.set("managed", "replacement")
        else:
            getattr(store, operation)("managed")
    if damage == "missing":
        assert not path.exists()
    else:
        assert path.read_text(encoding="utf-8") == "not-json"


def test_failed_atomic_replace_preserves_contents_and_allows_retry(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="synthetic")
    store.set("unrelated", "keep")
    before = path.read_bytes()
    original = Path.replace

    def denied(source: Path, target: Path) -> Path:
        raise PermissionError("synthetic replacement failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", denied)
        with pytest.raises(PermissionError):
            store.set("managed", "new")
    assert path.read_bytes() == before
    assert not path.with_suffix(".json.tmp").exists()
    assert Path.replace is original
    store.set("managed", "new")
    assert store.get("managed") == "new"
    assert store.get("unrelated") == "keep"
