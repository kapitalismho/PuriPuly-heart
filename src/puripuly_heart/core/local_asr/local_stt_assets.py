from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

from puripuly_heart.config import paths

LOCAL_STT_MODEL_ID = "qwen3-asr-0.6b-int8-sherpa"
LOCAL_STT_ENGINE = "sherpa-onnx"
LOCAL_STT_INSTALL_DIRNAME = "qwen3-asr-0.6b-int8-sherpa"
LOCAL_STT_INSTALLED_MANIFEST_FILENAME = "installed-manifest.json"
LOCAL_STT_MANIFEST_RELATIVE_PATH = f"data/models/{LOCAL_STT_INSTALL_DIRNAME}.manifest.json"
PARAKEET_V3_MODEL_ID = "parakeet-tdt-0.6b-v3-int8-sherpa"
PARAKEET_JAPANESE_MODEL_ID = "parakeet-tdt-ctc-0.6b-ja-int8-sherpa"
LOCAL_QWEN_GPU_MODEL_ID = "qwen3-asr-1.7b-q6-k-transcribe-vulkan"
REQUIRED_CPU_LOCAL_STT_MODEL_IDS = (
    PARAKEET_V3_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    LOCAL_STT_MODEL_ID,
)
LOCAL_STT_MANIFEST_RELATIVE_PATHS = {
    LOCAL_STT_MODEL_ID: LOCAL_STT_MANIFEST_RELATIVE_PATH,
    PARAKEET_V3_MODEL_ID: f"data/models/{PARAKEET_V3_MODEL_ID}.manifest.json",
    PARAKEET_JAPANESE_MODEL_ID: f"data/models/{PARAKEET_JAPANESE_MODEL_ID}.manifest.json",
    LOCAL_QWEN_GPU_MODEL_ID: f"data/models/{LOCAL_QWEN_GPU_MODEL_ID}.manifest.json",
}


class LocalSTTAssetError(RuntimeError):
    """Base error for local STT asset contract failures."""


class LocalSTTModelMissingError(LocalSTTAssetError):
    """Raised when the local STT model is not installed."""


class LocalSTTManifestInvalidError(LocalSTTAssetError):
    """Raised when the installed local STT manifest or files are invalid."""


class LocalQwenSherpaLoadError(RuntimeError):
    """Raised when the local sherpa recognizer cannot be initialized."""


class LocalParakeetSherpaLoadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LocalSTTAssetSource:
    name: str
    revision: str
    repo_id: str = ""
    download_url_template: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "revision": self.revision,
            "repo_id": self.repo_id,
            "download_url_template": self.download_url_template,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, object]) -> "LocalSTTAssetSource":
        return cls(
            name=str(data.get("name", name)),
            revision=str(data["revision"]),
            repo_id=str(data.get("repo_id", "")),
            download_url_template=str(data.get("download_url_template", "")),
        )


@dataclass(frozen=True, slots=True)
class LocalSTTAssetFile:
    relative_path: str
    sha256: str
    size_bytes: int | None = None
    source_path_overrides: dict[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        if self.source_path_overrides:
            payload["source_path_overrides"] = dict(self.source_path_overrides)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LocalSTTAssetFile":
        size_bytes = data.get("size_bytes")
        raw_overrides = data.get("source_path_overrides")
        if raw_overrides is None:
            source_path_overrides = None
        else:
            if not isinstance(raw_overrides, dict):
                raise LocalSTTManifestInvalidError(
                    "local STT asset manifest has invalid source_path_overrides"
                )
            source_path_overrides = {}
            for source_name, remote_path in raw_overrides.items():
                if not isinstance(source_name, str) or not isinstance(remote_path, str):
                    raise LocalSTTManifestInvalidError(
                        "local STT asset manifest has invalid source_path_overrides"
                    )
                source_path_overrides[source_name] = remote_path
        return cls(
            relative_path=str(data["relative_path"]),
            sha256=str(data["sha256"]).lower(),
            size_bytes=int(size_bytes) if size_bytes is not None else None,
            source_path_overrides=source_path_overrides,
        )

    def remote_path_for_source(self, source_name: str) -> str:
        if self.source_path_overrides and source_name in self.source_path_overrides:
            return self.source_path_overrides[source_name]
        return self.relative_path


@dataclass(frozen=True, slots=True)
class InstalledLocalSTTManifest:
    manifest_version: int
    model_id: str
    engine: str
    install_dirname: str
    selected_source: str
    selected_revision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "model_id": self.model_id,
            "engine": self.engine,
            "install_dirname": self.install_dirname,
            "selected_source": self.selected_source,
            "selected_revision": self.selected_revision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "InstalledLocalSTTManifest":
        return cls(
            manifest_version=int(data["manifest_version"]),
            model_id=str(data["model_id"]),
            engine=str(data["engine"]),
            install_dirname=str(data["install_dirname"]),
            selected_source=str(data["selected_source"]),
            selected_revision=str(data["selected_revision"]),
        )


LocalSTTInstallStatus = Literal["ready", "missing", "invalid"]


@dataclass(frozen=True, slots=True)
class LocalSTTInstallState:
    status: LocalSTTInstallStatus
    installed_manifest: InstalledLocalSTTManifest | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class LocalSTTAssetManifest:
    manifest_version: int
    installed_manifest_version: int
    model_id: str
    engine: str
    upstream_repo: str
    install_dirname: str
    sources: dict[str, LocalSTTAssetSource]
    files: tuple[LocalSTTAssetFile, ...]
    installed_manifest_filename: str = LOCAL_STT_INSTALLED_MANIFEST_FILENAME

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "installed_manifest_version": self.installed_manifest_version,
            "model_id": self.model_id,
            "engine": self.engine,
            "upstream_repo": self.upstream_repo,
            "install_dirname": self.install_dirname,
            "installed_manifest_filename": self.installed_manifest_filename,
            "sources": {key: value.to_dict() for key, value in self.sources.items()},
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LocalSTTAssetManifest":
        raw_sources = data.get("sources")
        if not isinstance(raw_sources, dict) or not raw_sources:
            raise LocalSTTManifestInvalidError("local STT asset manifest is missing sources")

        sources: dict[str, LocalSTTAssetSource] = {}
        for name, source_data in raw_sources.items():
            if not isinstance(name, str) or not isinstance(source_data, dict):
                raise LocalSTTManifestInvalidError("local STT asset manifest has invalid sources")
            sources[name] = LocalSTTAssetSource.from_dict(name, source_data)

        raw_files = data.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise LocalSTTManifestInvalidError("local STT asset manifest is missing files")

        files: list[LocalSTTAssetFile] = []
        for file_data in raw_files:
            if not isinstance(file_data, dict):
                raise LocalSTTManifestInvalidError("local STT asset manifest has invalid files")
            files.append(LocalSTTAssetFile.from_dict(file_data))

        return cls(
            manifest_version=int(data["manifest_version"]),
            installed_manifest_version=int(data["installed_manifest_version"]),
            model_id=str(data["model_id"]),
            engine=str(data["engine"]),
            upstream_repo=str(data["upstream_repo"]),
            install_dirname=str(data["install_dirname"]),
            installed_manifest_filename=str(
                data.get("installed_manifest_filename", LOCAL_STT_INSTALLED_MANIFEST_FILENAME)
            ),
            sources=sources,
            files=tuple(files),
        )


def default_local_stt_model_root() -> Path:
    return paths.default_models_dir()


def default_local_stt_model_dir(model_id: str = LOCAL_STT_MODEL_ID) -> Path:
    manifest_relative_path = LOCAL_STT_MANIFEST_RELATIVE_PATHS.get(model_id)
    if manifest_relative_path is None:
        raise LocalSTTManifestInvalidError(f"unknown local STT model_id: {model_id}")
    return default_local_stt_model_root() / Path(manifest_relative_path).name.removesuffix(
        ".manifest.json"
    )


def default_local_stt_installed_manifest_path(model_dir: Path | None = None) -> Path:
    resolved_model_dir = model_dir or default_local_stt_model_dir()
    return resolved_model_dir / LOCAL_STT_INSTALLED_MANIFEST_FILENAME


def default_local_stt_source_for_locale(locale: str | None) -> str:
    normalized = (locale or "").strip().replace("_", "-").lower()
    if normalized in {"zh-cn", "zh-hk", "zh-hant-hk"}:
        return "modelscope"
    return "huggingface"


def load_local_stt_asset_manifest(model_id: str = LOCAL_STT_MODEL_ID) -> LocalSTTAssetManifest:
    manifest_relative_path = LOCAL_STT_MANIFEST_RELATIVE_PATHS.get(model_id)
    if manifest_relative_path is None:
        raise LocalSTTManifestInvalidError(f"unknown local STT model_id: {model_id}")
    manifest_path = resources.files("puripuly_heart").joinpath(manifest_relative_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise LocalSTTManifestInvalidError("local STT asset manifest must be a JSON object")
    manifest = LocalSTTAssetManifest.from_dict(payload)
    if manifest.model_id != model_id:
        raise LocalSTTManifestInvalidError("local STT asset manifest model_id does not match")
    return manifest


def _load_installed_local_stt_manifest(
    model_dir: Path,
    *,
    manifest: LocalSTTAssetManifest,
) -> InstalledLocalSTTManifest:
    installed_manifest_path = model_dir / manifest.installed_manifest_filename
    if not installed_manifest_path.exists():
        raise LocalSTTModelMissingError("local STT installed manifest is missing")
    try:
        payload = json.loads(installed_manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise LocalSTTManifestInvalidError("local STT installed manifest must be a JSON object")
        installed = InstalledLocalSTTManifest.from_dict(payload)
    except LocalSTTManifestInvalidError:
        raise
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise LocalSTTManifestInvalidError("invalid local STT installed manifest") from exc
    if installed.manifest_version != manifest.installed_manifest_version:
        raise LocalSTTManifestInvalidError("stale installed manifest version")
    if installed.model_id != manifest.model_id:
        raise LocalSTTManifestInvalidError("installed manifest model_id does not match")
    if installed.engine != manifest.engine:
        raise LocalSTTManifestInvalidError("installed manifest engine does not match")
    if installed.install_dirname != manifest.install_dirname:
        raise LocalSTTManifestInvalidError("installed manifest install_dirname does not match")
    expected_source = manifest.sources.get(installed.selected_source)
    if expected_source is None:
        raise LocalSTTManifestInvalidError("installed manifest selected_source is unsupported")
    if installed.selected_revision != expected_source.revision:
        raise LocalSTTManifestInvalidError("stale installed manifest revision")
    return installed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_local_stt_model_dir(model_dir: Path) -> None:
    if not model_dir.exists():
        raise LocalSTTModelMissingError("local STT model directory is missing")
    if not model_dir.is_dir():
        raise LocalSTTManifestInvalidError("local STT model path is not a directory")


def _validate_required_model_files(
    model_dir: Path,
    *,
    manifest: LocalSTTAssetManifest,
    verify_checksums: bool,
) -> None:
    for asset in manifest.files:
        asset_path = model_dir / asset.relative_path
        if not asset_path.exists():
            raise LocalSTTManifestInvalidError(
                f"missing required model file: {asset.relative_path}"
            )
        if not asset_path.is_file():
            raise LocalSTTManifestInvalidError(
                f"required model path is not a file: {asset.relative_path}"
            )
        if verify_checksums and _sha256_file(asset_path) != asset.sha256:
            raise LocalSTTManifestInvalidError(
                f"checksum mismatch for required model file: {asset.relative_path}"
            )


def validate_local_stt_runtime_ready(
    model_dir: Path | None = None,
    *,
    manifest: LocalSTTAssetManifest | None = None,
) -> InstalledLocalSTTManifest:
    resolved_manifest = manifest or load_local_stt_asset_manifest()
    resolved_model_dir = (
        model_dir or default_local_stt_model_root() / resolved_manifest.install_dirname
    )

    _validate_local_stt_model_dir(resolved_model_dir)
    installed = _load_installed_local_stt_manifest(
        resolved_model_dir,
        manifest=resolved_manifest,
    )
    _validate_required_model_files(
        resolved_model_dir,
        manifest=resolved_manifest,
        verify_checksums=False,
    )
    return installed


def validate_local_stt_install(
    model_dir: Path | None = None,
    *,
    manifest: LocalSTTAssetManifest | None = None,
) -> InstalledLocalSTTManifest:
    resolved_manifest = manifest or load_local_stt_asset_manifest()
    resolved_model_dir = (
        model_dir or default_local_stt_model_root() / resolved_manifest.install_dirname
    )

    _validate_local_stt_model_dir(resolved_model_dir)
    installed = _load_installed_local_stt_manifest(
        resolved_model_dir,
        manifest=resolved_manifest,
    )
    _validate_required_model_files(
        resolved_model_dir,
        manifest=resolved_manifest,
        verify_checksums=True,
    )
    return installed


def inspect_local_stt_install_state(
    model_dir: Path | None = None,
    *,
    manifest: LocalSTTAssetManifest | None = None,
    verify_checksums: bool = False,
) -> LocalSTTInstallState:
    resolved_manifest = manifest or load_local_stt_asset_manifest()
    resolved_model_dir = (
        model_dir or default_local_stt_model_root() / resolved_manifest.install_dirname
    )
    try:
        validator = (
            validate_local_stt_install if verify_checksums else validate_local_stt_runtime_ready
        )
        installed = validator(resolved_model_dir, manifest=resolved_manifest)
    except LocalSTTModelMissingError:
        return LocalSTTInstallState(status="missing")
    except LocalSTTManifestInvalidError as exc:
        return LocalSTTInstallState(status="invalid", error_message=str(exc))
    return LocalSTTInstallState(status="ready", installed_manifest=installed)
