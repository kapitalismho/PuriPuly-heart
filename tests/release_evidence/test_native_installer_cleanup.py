from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from puripuly_heart.release_evidence.native_installer_cleanup import (
    LEGACY_SCHEMA,
    NATIVE_SCHEMA,
    cleanup_plan,
    generate_include,
    normalize_windows_relative,
    obsolete_owned_files,
    read_inventory,
)

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="Windows ownership cleanup handles")


def _row(path: str, data: bytes = b"official legacy file") -> dict:
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _manifest(schema: str, *rows: dict) -> dict:
    return {"schema": schema, "inventory": list(rows)}


def _plan(*rows: dict) -> dict:
    legacy = _manifest(LEGACY_SCHEMA, *rows)
    native = _manifest(NATIVE_SCHEMA, _row("native-only.exe", b"new"))
    return cleanup_plan(json.dumps(legacy).encode(), json.dumps(native).encode())


def test_difference_preserves_shared_paths_regardless_of_hash_case_and_separator() -> None:
    legacy = _manifest(
        LEGACY_SCHEMA,
        _row("MSVCP140_1.dll"),
        _row("SOXR\\soxr.dll"),
        _row("old\\obsolete.dll"),
    )
    native = _manifest(
        NATIVE_SCHEMA,
        _row("msvcp140_1.DLL", b"repaired runtime"),
        _row("soxr/SoXr.dll", b"new soxr"),
    )

    assert [row.path for row in obsolete_owned_files(legacy, native)] == ["old/obsolete.dll"]


def test_known_packaged_user_data_is_preserved_even_if_absent_from_native_inventory() -> None:
    protected = [
        "settings.json",
        "Settings.json.bak",
        "secrets.json",
        ".env",
        ".env.local",
        "prompts/translation_prompt.md",
        "extensions/plugin.py",
        "models/model.onnx",
    ]
    legacy = _manifest(
        LEGACY_SCHEMA, *(_row(path) for path in protected), _row("runtime/obsolete.dll")
    )
    native = _manifest(NATIVE_SCHEMA, _row("app/prompts/translation_prompt.md"))

    assert [row.path for row in obsolete_owned_files(legacy, native)] == ["runtime/obsolete.dll"]
    plan = cleanup_plan(json.dumps(legacy).encode(), json.dumps(native).encode())
    assert plan["preserved_paths"] == sorted(protected, key=str.casefold)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/root.dll",
        "\\root.dll",
        "C:\\root.dll",
        "C:relative.dll",
        "//host/share/file",
        "../user",
        "dir/../user",
        "dir/./file",
        "dir//file",
        "file:stream",
        "dir/file.",
        "dir /file",
        "dir/file ",
        "NUL",
        "con.txt",
        "dir/COM1.dll",
        "LPT².dat",
        "dir/wild*.dll",
        "dir/question?.dll",
        'dir/quote".dll',
        "dir/<name>",
        "dir/line\nfile",
        "dir/tab\tfile",
        "NUL .dll",
        "COM0.dat",
        "dir/CONIN$",
        "CONOUT$.txt",
        "CLOCK$",
    ],
)
def test_inventory_rejects_windows_aliases_and_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        normalize_windows_relative(path)


@pytest.mark.parametrize(
    "rows",
    [
        [_row("folder/file"), _row("FOLDER\\FILE")],
        [_row("folder"), _row("folder/file")],
        [{**_row("file"), "bytes": True}],
        [{**_row("file"), "bytes": -1}],
        [{**_row("file"), "bytes": 2**63}],
        [{**_row("file"), "sha256": "0" * 63}],
        [{**_row("file"), "sha256": "g" * 64}],
        [{**_row("file"), "wildcard": True}],
    ],
)
def test_inventory_rejects_ambiguous_or_incomplete_ownership(rows: list[dict]) -> None:
    with pytest.raises(ValueError):
        read_inventory(_manifest(LEGACY_SCHEMA, *rows), LEGACY_SCHEMA)


@pytest.mark.parametrize(
    ("old", "new"),
    [("folder", "folder/new.dll"), ("folder/old.dll", "folder")],
)
def test_difference_rejects_file_directory_cutovers_that_cannot_copy_safely(
    old: str, new: str
) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        obsolete_owned_files(
            _manifest(LEGACY_SCHEMA, _row(old)), _manifest(NATIVE_SCHEMA, _row(new))
        )


def test_plan_uses_exact_input_digest_and_empty_directory_order() -> None:
    legacy = json.dumps(_manifest(LEGACY_SCHEMA, _row("old/a/b.dll"), _row("old/c.dll"))).encode()
    native = json.dumps(_manifest(NATIVE_SCHEMA, _row("native.exe"))).encode()
    plan = cleanup_plan(legacy, native)

    assert plan["native_manifest_sha256"] == hashlib.sha256(native).hexdigest()
    assert plan["legacy_manifest_sha256"] == hashlib.sha256(legacy).hexdigest()
    assert plan["directories"] == ["old/a", "old"]
    changed = cleanup_plan(legacy, native + b"\n")
    assert changed["native_manifest_sha256"] != plan["native_manifest_sha256"]
    assert changed["inventory"] == plan["inventory"]


def test_generator_never_overwrites_manifests_and_invalid_input_creates_no_include(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.json"
    native = tmp_path / "native.json"
    output = tmp_path / "cleanup.iss"
    legacy.write_text(json.dumps(_manifest(LEGACY_SCHEMA, _row("old.dll"))), encoding="utf-8")
    native.write_text(json.dumps(_manifest(NATIVE_SCHEMA, _row("native.dll"))), encoding="utf-8")
    before = legacy.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        generate_include(legacy, native, legacy)
    assert legacy.read_bytes() == before
    native.write_text(json.dumps(_manifest(NATIVE_SCHEMA, _row("../user"))), encoding="utf-8")
    with pytest.raises(ValueError):
        generate_include(legacy, native, output)
    assert not output.exists()


def _invoke(tmp_path: Path, root: Path, plan: dict) -> tuple[subprocess.CompletedProcess, str]:
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "result.txt"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    result_path.unlink(missing_ok=True)
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    process = subprocess.run(
        [
            str(powershell),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "installer/native-cleanup.ps1"),
            "-Root",
            str(root),
            "-Plan",
            str(plan_path),
            "-ResultFile",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    detail = result_path.read_text(encoding="utf-8") if result_path.exists() else process.stderr
    return process, detail


@WINDOWS
def test_cleanup_removes_only_exact_owned_files_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "installed"
    (root / "old/deep").mkdir(parents=True)
    (root / "old/deep/exact.dll").write_bytes(b"official legacy file")
    (root / "old/user.dll").write_bytes(b"unlisted user content")
    (root / "native-only.exe").write_bytes(b"new")
    obsolete = ["python312.dll", "base_library.zip", "scipy/special/old.pyd", "numpy.libs/old.dll"]
    for relative in obsolete:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"official legacy file")
    (root / "app/prompts").mkdir(parents=True)
    (root / "app/prompts/translation_prompt.md").write_bytes(b"new prompt asset")
    plan = _plan(_row("old/deep/exact.dll"), _row("absent.dll"), *(_row(path) for path in obsolete))

    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode == 0, detail
    assert not (root / "old/deep").exists()
    assert (root / "old/user.dll").read_bytes() == b"unlisted user content"
    assert (root / "native-only.exe").read_bytes() == b"new"
    assert all(not (root / path).exists() for path in obsolete)
    assert not (root / "scipy").exists()
    assert not (root / "numpy.libs").exists()
    assert (root / "app/prompts/translation_prompt.md").read_bytes() == b"new prompt asset"
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode == 0, detail


@WINDOWS
def test_protected_old_prompts_do_not_block_or_get_deleted(tmp_path: Path) -> None:
    root = tmp_path / "installed"
    (root / "prompts").mkdir(parents=True)
    prompt = root / "prompts/translation_prompt.md"
    prompt.write_bytes(b"customized user prompt")
    exact = root / "old.dll"
    exact.write_bytes(b"official legacy file")
    plan = _plan(_row("prompts/translation_prompt.md"), _row("old.dll"))
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode == 0, detail
    assert prompt.read_bytes() == b"customized user prompt"
    assert not exact.exists()


@WINDOWS
@pytest.mark.parametrize("replacement", [b"changed legacy file!", b"short"])
def test_changed_obsolete_file_is_preserved_while_other_owned_files_are_removed(
    tmp_path: Path, replacement: bytes
) -> None:
    root = tmp_path / "installed"
    root.mkdir()
    exact = root / "a-exact.dll"
    changed = root / "z-changed.dll"
    exact.write_bytes(b"official legacy file")
    changed.write_bytes(replacement)
    plan = _plan(_row(exact.name), _row(changed.name))

    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode == 0, detail
    assert str(changed) in detail
    assert not exact.exists()
    assert changed.read_bytes() == replacement


@WINDOWS
@pytest.mark.parametrize("position", ["root", "root-ancestor", "candidate-ancestor"])
def test_junction_is_rejected_without_traversing_target(tmp_path: Path, position: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    if position == "candidate-ancestor":
        root = tmp_path / "installed"
        root.mkdir()
        link = root / "old"
        relative = "old/old.dll"
        target_file = outside / "old.dll"
    elif position == "root-ancestor":
        link = tmp_path / "parent-link"
        root = link / "installed"
        (outside / "installed").mkdir()
        relative = "old.dll"
        target_file = outside / "installed/old.dll"
    else:
        link = tmp_path / "installed"
        root = link
        relative = "old.dll"
        target_file = outside / "old.dll"
    target_file.write_bytes(b"official legacy file")
    subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(outside)],
        capture_output=True,
        check=True,
    )
    try:
        process, detail = _invoke(tmp_path, root, _plan(_row(relative)))
        assert process.returncode == (0 if position == "candidate-ancestor" else 20)
        assert str(link) in detail
        assert target_file.read_bytes() == b"official legacy file"
    finally:
        link.rmdir()


@WINDOWS
def test_hard_linked_obsolete_file_is_not_removed(tmp_path: Path) -> None:
    root = tmp_path / "installed"
    root.mkdir()
    outside = tmp_path / "outside.dll"
    outside.write_bytes(b"official legacy file")
    path = root / "old.dll"
    os.link(outside, path)
    process, detail = _invoke(tmp_path, root, _plan(_row(path.name)))
    assert process.returncode == 0, detail
    assert str(path) in detail
    assert path.read_bytes() == outside.read_bytes() == b"official legacy file"


@WINDOWS
def test_delete_access_failure_preserves_file_and_retry_succeeds(tmp_path: Path) -> None:
    from ctypes import wintypes

    root = tmp_path / "installed"
    root.mkdir()
    path = root / "old.dll"
    path.write_bytes(b"official legacy file")
    plan = _plan(_row(path.name))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
    assert handle != wintypes.HANDLE(-1).value
    try:
        process, detail = _invoke(tmp_path, root, plan)
        assert process.returncode == 0, detail
        assert str(path) in detail
        assert path.read_bytes() == b"official legacy file"
    finally:
        assert kernel.CloseHandle(handle)
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode == 0, detail
    assert not path.exists()


@WINDOWS
@pytest.mark.parametrize(
    "unsafe", ["../outside.dll", "old.dll:stream", "old.dll.", "dir/../outside.dll"]
)
def test_helper_rejects_unsafe_candidates_before_touching_any_file(
    tmp_path: Path, unsafe: str
) -> None:
    root = tmp_path / "installed"
    root.mkdir()
    exact = root / "old.dll"
    outside = tmp_path / "outside.dll"
    exact.write_bytes(b"official legacy file")
    outside.write_bytes(b"official legacy file")
    plan = _plan(_row("old.dll"))
    plan["inventory"].append(_row(unsafe))
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode != 0
    assert detail
    assert exact.read_bytes() == outside.read_bytes() == b"official legacy file"


@WINDOWS
def test_helper_refuses_a_plan_that_attempts_protected_data_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "installed"
    (root / "prompts").mkdir(parents=True)
    prompt = root / "prompts/translation_prompt.md"
    prompt.write_bytes(b"official legacy file")
    plan = _plan(_row("old.dll"))
    plan["inventory"].append(_row("prompts/translation_prompt.md"))
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode != 0
    assert "prompts/translation_prompt.md" in detail
    assert prompt.read_bytes() == b"official legacy file"


@WINDOWS
def test_empty_cleanup_inventory_succeeds_without_creating_or_deleting_the_root(
    tmp_path: Path,
) -> None:
    row = _row("common.dll")
    plan = cleanup_plan(
        json.dumps(_manifest(LEGACY_SCHEMA, row)).encode(),
        json.dumps(_manifest(NATIVE_SCHEMA, row)).encode(),
    )
    root = tmp_path / "absent"
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode == 0, detail
    assert not root.exists()


def _native_worker_upgrade(root: Path) -> tuple[dict, dict[str, bytes], dict[str, bytes]]:
    root.mkdir()
    old_names = (
        "PuriPulyHeartGpuWorker.exe",
        "transcribe.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-vulkan.dll",
        "ggml-cpu-x64.dll",
        "ggml-cpu-retired.dll",
    )
    prior = {name: f"prior-owned-{name}".encode() for name in old_names}
    for name, payload in prior.items():
        (root / name).write_bytes(payload)
    current = {
        name: f"current-owned-{name}".encode()
        for name in ("PuriPulyHeartASRWorker.exe", *old_names[1:-1])
    }
    current["_runtime/local_qwen/onnxruntime.dll"] = b"current-japanese-runtime"
    current["_runtime/local_qwen/onnxruntime_providers_shared.dll"] = b"current-japanese-shared"
    for name, payload in current.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    inventory = {
        "schema_version": 1,
        "cpu_selection": "upstream-cpuid",
        "build_id": "a" * 32,
        "files": [
            {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in prior.items()
        ],
    }
    (root / "gpu-worker-runtime.json").write_text(json.dumps(inventory), encoding="utf-8")
    native_bytes = json.dumps(
        _manifest(NATIVE_SCHEMA, *(_row(name, payload) for name, payload in current.items()))
    ).encode()
    (root / "native-artifact-manifest.json").write_bytes(native_bytes)
    legacy_bytes = json.dumps(
        _manifest(
            LEGACY_SCHEMA,
            _row("historical-marker.dll"),
            _row("_runtime/local_qwen/onnxruntime.dll", b"old-cache-bytes"),
        )
    ).encode()
    return cleanup_plan(legacy_bytes, native_bytes), prior, current


@WINDOWS
def test_native_upgrade_retires_only_old_owned_worker_inventory_and_obsolete_dll(tmp_path):
    root = tmp_path / "installed"
    plan, prior, current = _native_worker_upgrade(root)
    protected = {
        "user.dll": b"unlisted-user-dll",
        "models/parakeet-tdt-ctc-0.6b-ja-int8-sherpa/sentinel": b"japanese-model",
        "models/qwen3-asr-1.7b-q6-k-transcribe-vulkan/sentinel": b"gpu-model",
        "settings.json": b"settings",
    }
    for name, payload in protected.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode == 0, detail
    assert not (root / "PuriPulyHeartGpuWorker.exe").exists()
    assert not (root / "ggml-cpu-retired.dll").exists()
    assert not (root / "gpu-worker-runtime.json").exists()
    assert all((root / name).read_bytes() == payload for name, payload in current.items())
    assert all((root / name).read_bytes() == payload for name, payload in protected.items())
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode == 0, detail
    assert all((root / name).read_bytes() == payload for name, payload in current.items())


@WINDOWS
@pytest.mark.parametrize("cause", ["modified", "open-handle"])
def test_native_upgrade_partial_failure_keeps_old_inventory_for_safe_retry(tmp_path, cause):
    root = tmp_path / "installed"
    plan, prior, current = _native_worker_upgrade(root)
    worker = root / "PuriPulyHeartGpuWorker.exe"
    handle = None
    try:
        if cause == "modified":
            worker.write_bytes(b"changed-unowned-worker")
        else:
            handle = worker.open("rb")
        process, detail = _invoke(tmp_path, root, plan)
        assert process.returncode == 0, detail
        assert worker.exists()
        assert (root / "gpu-worker-runtime.json").exists()
        assert not (root / "ggml-cpu-retired.dll").exists()
        assert all((root / name).read_bytes() == payload for name, payload in current.items())
    finally:
        if handle is not None:
            handle.close()
    if cause == "modified":
        worker.write_bytes(prior[worker.name])
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode == 0, detail
    assert not worker.exists()
    assert not (root / "gpu-worker-runtime.json").exists()


@WINDOWS
@pytest.mark.parametrize("defect", ["current-hash", "unknown-entry", "traversal"])
def test_native_upgrade_rejects_unproven_ownership_before_deleting_anything(tmp_path, defect):
    root = tmp_path / "installed"
    plan, prior, current = _native_worker_upgrade(root)
    if defect == "current-hash":
        path = root / "native-artifact-manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        inventory_path = root / "gpu-worker-runtime.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["files"].append(
            {
                "path": "user.dll" if defect == "unknown-entry" else "../outside.dll",
                "sha256": hashlib.sha256(b"user-content").hexdigest(),
            }
        )
        inventory_path.write_text(json.dumps(inventory))
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode != 0
    assert (root / "PuriPulyHeartGpuWorker.exe").read_bytes() == prior["PuriPulyHeartGpuWorker.exe"]
    assert (root / "ggml-cpu-retired.dll").read_bytes() == prior["ggml-cpu-retired.dll"]
    assert (root / "gpu-worker-runtime.json").exists()
    assert all((root / name).read_bytes() == payload for name, payload in current.items())


@WINDOWS
@pytest.mark.parametrize(
    "inventory_name", ["gpu-worker-runtime.json", "native-artifact-manifest.json"]
)
def test_native_upgrade_never_follows_inventory_reparse_targets(tmp_path, inventory_name):
    root = tmp_path / "installed"
    plan, prior, current = _native_worker_upgrade(root)
    path = root / inventory_name
    target = tmp_path / inventory_name
    path.rename(target)
    original = target.read_bytes()
    try:
        path.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink permission unavailable: {type(error).__name__}")
    process, detail = _invoke(tmp_path, root, plan)
    assert process.returncode != 0
    assert target.read_bytes() == original
    assert (root / "PuriPulyHeartGpuWorker.exe").read_bytes() == prior["PuriPulyHeartGpuWorker.exe"]
    assert (root / "ggml-cpu-retired.dll").exists()
