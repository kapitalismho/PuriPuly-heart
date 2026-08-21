from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from puripuly_heart.config.paths import default_models_dir

GEMMA_MODEL_ID = "gemma-4-e4b-it-qat-ud-q4-k-xl"
GEMMA_REPO_ID = "unsloth/gemma-4-E4B-it-qat-GGUF"
GEMMA_REVISION = "8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265"
GEMMA_MODEL_FILENAME = "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
GEMMA_DRAFT_FILENAME = "mtp-gemma-4-E4B-it.gguf"
GEMMA_INSTALL_DIRNAME = GEMMA_MODEL_ID
GEMMA_INSTALLED_MANIFEST_FILENAME = "installed-manifest.json"
GEMMA_INSTALLED_MANIFEST_VERSION = 1
GEMMA_UPSTREAM_REPO_ID = "google/gemma-4-E4B-it"
GEMMA_LICENSE = "Apache-2.0"
GEMMA_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"


class GemmaAssetError(RuntimeError):
    pass


class GemmaInstallMissingError(GemmaAssetError):
    pass


class GemmaInstallInvalidError(GemmaAssetError):
    pass


@dataclass(frozen=True, slots=True)
class GemmaAsset:
    filename: str
    size_bytes: int
    sha256: str


GEMMA_ASSETS = (
    GemmaAsset(
        filename=GEMMA_MODEL_FILENAME,
        size_bytes=4_215_695_776,
        sha256="df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3",
    ),
    GemmaAsset(
        filename=GEMMA_DRAFT_FILENAME,
        size_bytes=59_678_016,
        sha256="423074e537504b4f9ec5eafed5c639fac82c96631626efccacdd3c4039b20605",
    ),
)


@dataclass(frozen=True, slots=True)
class InstalledGemmaManifest:
    manifest_version: int
    model_id: str
    repo_id: str
    revision: str
    upstream_repo_id: str
    license: str
    license_url: str
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "model_id": self.model_id,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "upstream_repo_id": self.upstream_repo_id,
            "license": self.license,
            "license_url": self.license_url,
            "files": list(self.files),
        }

    @classmethod
    def expected(cls) -> InstalledGemmaManifest:
        return cls(
            manifest_version=GEMMA_INSTALLED_MANIFEST_VERSION,
            model_id=GEMMA_MODEL_ID,
            repo_id=GEMMA_REPO_ID,
            revision=GEMMA_REVISION,
            upstream_repo_id=GEMMA_UPSTREAM_REPO_ID,
            license=GEMMA_LICENSE,
            license_url=GEMMA_LICENSE_URL,
            files=tuple(asset.filename for asset in GEMMA_ASSETS),
        )

    @classmethod
    def from_dict(cls, value: object) -> InstalledGemmaManifest:
        if not isinstance(value, dict):
            raise GemmaInstallInvalidError("Gemma installed manifest must be an object")
        raw_files = value.get("files")
        if not isinstance(raw_files, list) or not all(isinstance(item, str) for item in raw_files):
            raise GemmaInstallInvalidError("Gemma installed manifest files are invalid")
        try:
            return cls(
                manifest_version=int(value["manifest_version"]),
                model_id=str(value["model_id"]),
                repo_id=str(value["repo_id"]),
                revision=str(value["revision"]),
                upstream_repo_id=GEMMA_UPSTREAM_REPO_ID,
                license=GEMMA_LICENSE,
                license_url=GEMMA_LICENSE_URL,
                files=tuple(raw_files),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GemmaInstallInvalidError("Gemma installed manifest is invalid") from exc


GemmaInstallStatus = Literal["ready", "missing", "invalid"]


@dataclass(frozen=True, slots=True)
class GemmaInstallState:
    status: GemmaInstallStatus
    install_dir: Path
    manifest: InstalledGemmaManifest | None = None
    error_message: str | None = None


def default_gemma_install_dir() -> Path:
    return default_models_dir() / GEMMA_INSTALL_DIRNAME


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_installed_manifest(install_dir: Path) -> InstalledGemmaManifest:
    path = install_dir / GEMMA_INSTALLED_MANIFEST_FILENAME
    if not path.is_file():
        raise GemmaInstallMissingError("Gemma installed manifest is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GemmaInstallInvalidError("Gemma installed manifest is invalid") from exc
    manifest = InstalledGemmaManifest.from_dict(value)
    if manifest != InstalledGemmaManifest.expected():
        raise GemmaInstallInvalidError("Gemma installed manifest does not match the pinned model")
    return manifest


def validate_gemma_install(
    install_dir: Path | None = None,
    *,
    verify_checksums: bool = True,
) -> InstalledGemmaManifest:
    resolved = (install_dir or default_gemma_install_dir()).resolve()
    if not resolved.exists():
        raise GemmaInstallMissingError("Gemma model directory is missing")
    if not resolved.is_dir():
        raise GemmaInstallInvalidError("Gemma model path is not a directory")
    manifest = _load_installed_manifest(resolved)
    for asset in GEMMA_ASSETS:
        path = resolved / asset.filename
        if not path.is_file():
            raise GemmaInstallInvalidError(f"missing required Gemma file: {asset.filename}")
        if path.stat().st_size != asset.size_bytes:
            raise GemmaInstallInvalidError(
                f"size mismatch for required Gemma file: {asset.filename}"
            )
        if verify_checksums and _sha256_file(path) != asset.sha256:
            raise GemmaInstallInvalidError(
                f"checksum mismatch for required Gemma file: {asset.filename}"
            )
    return manifest


def inspect_gemma_install(
    install_dir: Path | None = None,
    *,
    verify_checksums: bool = False,
) -> GemmaInstallState:
    resolved = (install_dir or default_gemma_install_dir()).resolve()
    try:
        manifest = validate_gemma_install(resolved, verify_checksums=verify_checksums)
    except GemmaInstallMissingError:
        return GemmaInstallState(status="missing", install_dir=resolved)
    except GemmaInstallInvalidError as exc:
        return GemmaInstallState(
            status="invalid",
            install_dir=resolved,
            error_message=str(exc),
        )
    return GemmaInstallState(status="ready", install_dir=resolved, manifest=manifest)


__all__ = [
    "GEMMA_ASSETS",
    "GEMMA_DRAFT_FILENAME",
    "GEMMA_INSTALL_DIRNAME",
    "GEMMA_INSTALLED_MANIFEST_FILENAME",
    "GEMMA_MODEL_FILENAME",
    "GEMMA_MODEL_ID",
    "GEMMA_REPO_ID",
    "GEMMA_REVISION",
    "GEMMA_UPSTREAM_REPO_ID",
    "GEMMA_LICENSE",
    "GEMMA_LICENSE_URL",
    "GemmaAsset",
    "GemmaAssetError",
    "GemmaInstallInvalidError",
    "GemmaInstallMissingError",
    "GemmaInstallState",
    "InstalledGemmaManifest",
    "default_gemma_install_dir",
    "inspect_gemma_install",
    "validate_gemma_install",
]
