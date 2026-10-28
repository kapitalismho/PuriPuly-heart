from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path

from .local_stt_assets import default_local_stt_model_root
from .local_stt_runtime_installer import LocalSTTProvisioningLease

_RETIRED_PAYLOADS = {
    "qwen3-asr-0.6b-int8-sherpa": {
        "conv_frontend.onnx": (
            44148281,
            "d22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e",
        ),
        "decoder.int8.onnx": (
            755914231,
            "4f6885be5959ae26af3089d38ee7972c5fafbeeb1cf8d5e76eab6d8b61ca5771",
        ),
        "encoder.int8.onnx": (
            182491662,
            "60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9",
        ),
        "tokenizer/merges.txt": (
            1671853,
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        ),
        "tokenizer/tokenizer_config.json": (
            12487,
            "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
        ),
        "tokenizer/vocab.json": (
            2776833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    },
    "parakeet-tdt-0.6b-v3-int8-sherpa": {
        "decoder.int8.onnx": (
            11845275,
            "179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e",
        ),
        "encoder.int8.onnx": (
            652184281,
            "acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247",
        ),
        "joiner.int8.onnx": (
            6355277,
            "3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3",
        ),
        "tokens.txt": (93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
    },
}
RETIRED_CPU_ASR_MODEL_IDS = frozenset(_RETIRED_PAYLOADS)
_RETIRED_SETUP_STAGING_SOURCES = {
    "qwen3-asr-0.6b-int8-sherpa": frozenset({"huggingface", "modelscope"}),
    "parakeet-tdt-0.6b-v3-int8-sherpa": frozenset({"huggingface", "hf_mirror"}),
}
_INSTALLED_MANIFEST = "installed-manifest.json"
_PENDING_MANIFEST = ".retirement-pending.json"


@dataclass(frozen=True, slots=True)
class RetiredLocalASRCleanupResult:
    removed_count: int
    pending_count: int
    error_codes: tuple[str, ...]


class _ProtectedAsset(ValueError):
    pass


def _is_reparse(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _candidate_model_id(name: str) -> str | None:
    for model_id in RETIRED_CPU_ASR_MODEL_IDS:
        if name in {model_id, f"{model_id}.backup"}:
            return model_id
        if name in {
            f"{model_id}.staging-{source}" for source in _RETIRED_SETUP_STAGING_SOURCES[model_id]
        }:
            return model_id
        for suffix in (".staging-", ".deleting-"):
            if name.startswith(f"{model_id}{suffix}"):
                token = name[len(model_id) + len(suffix) :]
                if len(token) == 32 and all(char in "0123456789abcdef" for char in token):
                    return model_id
    return None


def _scan_candidate(directory: Path, model_id: str) -> tuple[list[Path], list[Path]]:
    allowed_files = {*_RETIRED_PAYLOADS[model_id], _INSTALLED_MANIFEST, _PENDING_MANIFEST}
    allowed_dirs = {
        str(parent).replace("\\", "/")
        for name in allowed_files
        for parent in Path(name).parents
        if str(parent) != "."
    }
    files: list[Path] = []
    directories: list[Path] = []
    remaining = [directory]
    while remaining:
        parent = remaining.pop()
        if _is_reparse(parent):
            raise _ProtectedAsset("protected_reparse_point")
        for child in parent.iterdir():
            if _is_reparse(child):
                raise _ProtectedAsset("protected_reparse_point")
            relative = child.relative_to(directory).as_posix()
            if child.is_dir() and relative in allowed_dirs:
                directories.append(child)
                remaining.append(child)
            elif child.is_file() and relative in allowed_files:
                files.append(child)
            else:
                raise _ProtectedAsset("unrecognized_contents")
    return files, directories


def _matches_identity(path: Path, model_id: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError, UnicodeError, ValueError:
        return False
    return isinstance(payload, dict) and all(
        payload.get(key) == value
        for key, value in {
            "manifest_version": 1,
            "model_id": model_id,
            "engine": "sherpa-onnx",
            "install_dirname": model_id,
        }.items()
    )


def _recognized(directory: Path, model_id: str, files: list[Path]) -> bool:
    for name in (_INSTALLED_MANIFEST, _PENDING_MANIFEST):
        identity_path = directory / name
        if identity_path in files and _matches_identity(identity_path, model_id):
            return True
    for path in files:
        expected = _RETIRED_PAYLOADS[model_id].get(path.relative_to(directory).as_posix())
        if expected is None or path.stat().st_size != expected[0]:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() == expected[1]:
            return True
    return False


def _remove_recognized(directory: Path, model_id: str, files: list[Path], dirs: list[Path]) -> None:
    pending = directory / _PENDING_MANIFEST
    identity = json.dumps(
        {
            "manifest_version": 1,
            "model_id": model_id,
            "engine": "sherpa-onnx",
            "install_dirname": model_id,
        }
    )
    pending.write_text(identity, encoding="utf-8")
    for path in files:
        if path != pending:
            if _is_reparse(path):
                raise _ProtectedAsset("protected_reparse_point")
            path.unlink()
    for path in sorted(dirs, key=lambda item: len(item.parts), reverse=True):
        if _is_reparse(path):
            raise _ProtectedAsset("protected_reparse_point")
        path.rmdir()
    pending.unlink()
    try:
        directory.rmdir()
    except OSError:
        pending.write_text(identity, encoding="utf-8")
        raise


def cleanup_retired_cpu_asr_assets(
    *, model_root: Path | None = None
) -> RetiredLocalASRCleanupResult:
    root = (model_root or default_local_stt_model_root()).absolute()
    try:
        for ancestor in (root, *root.parents):
            if ancestor.exists() or ancestor.is_symlink():
                if _is_reparse(ancestor):
                    return RetiredLocalASRCleanupResult(0, 1, ("protected_reparse_point",))
        if not root.exists():
            return RetiredLocalASRCleanupResult(0, 0, ())
        lease = LocalSTTProvisioningLease.acquire(model_root=root, wait=False)
    except OSError:
        return RetiredLocalASRCleanupResult(0, 1, ("root_unavailable",))
    if lease is None:
        return RetiredLocalASRCleanupResult(0, 1, ("lease_busy",))
    removed = 0
    pending = 0
    errors: set[str] = set()
    try:
        for directory in root.iterdir():
            model_id = _candidate_model_id(directory.name)
            if model_id is None:
                continue
            try:
                if _is_reparse(directory):
                    raise _ProtectedAsset("protected_reparse_point")
                if not directory.is_dir():
                    raise _ProtectedAsset("unrecognized_contents")
                files, dirs = _scan_candidate(directory, model_id)
                if not _recognized(directory, model_id, files):
                    raise _ProtectedAsset("unrecognized_contents")
                _remove_recognized(directory, model_id, files, dirs)
                removed += 1
            except _ProtectedAsset as exc:
                pending += 1
                errors.add(str(exc))
            except OSError:
                pending += 1
                errors.add("delete_failed")
    except OSError:
        pending += 1
        errors.add("root_unavailable")
    finally:
        try:
            lease.close()
        except OSError:
            pending += 1
            errors.add("lease_release_failed")
    return RetiredLocalASRCleanupResult(removed, pending, tuple(sorted(errors)))
