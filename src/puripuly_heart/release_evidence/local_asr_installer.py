from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

from puripuly_heart.config.settings_vnext.schema import VNEXT_SETTINGS_SCHEMA_VERSION
from puripuly_heart.core.local_asr.local_stt_assets import (
    LOCAL_STT_MANIFEST_RELATIVE_PATHS,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    InstalledLocalSTTManifest,
    LocalSTTAssetManifest,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.model_download_sources import HF_MIRROR_ENDPOINT
from puripuly_heart.release_evidence.native_installer_cleanup import normalize_windows_relative
from puripuly_heart.runtime_layout import current_runtime_layout


def _string(value: str) -> str:
    return "'" + value.replace("'", "''").replace("\r", "").replace("\n", "' + #10 + '") + "'"


def _source(manifest: LocalSTTAssetManifest, name: str):
    return manifest.sources.get(name, manifest.sources["huggingface"])


def _url(manifest: LocalSTTAssetManifest, asset, source_name: str) -> str:
    source = _source(manifest, source_name)
    source_key = source_name if source_name in manifest.sources else "huggingface"
    url = source.download_url_template.format(path=asset.remote_path_for_source(source_key))
    if source_name == "hf_mirror" and source_name not in manifest.sources:
        if not url.startswith("https://huggingface.co/"):
            raise ValueError("a derived mirror requires a canonical Hugging Face URL")
        url = HF_MIRROR_ENDPOINT + url.removeprefix("https://huggingface.co")
    return url


def render_model_data(manifests: Sequence[LocalSTTAssetManifest] | None = None) -> str:
    canonical = manifests is None
    models = (
        tuple(manifests)
        if manifests is not None
        else tuple(
            load_local_stt_asset_manifest(model_id) for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS
        )
    )
    if tuple(model.model_id for model in models) != REQUIRED_CPU_LOCAL_STT_MODEL_IDS:
        raise ValueError("Setup requires the exact ordered current CPU model registry")
    lines = [f"#define CanonicalSettingsVersion {VNEXT_SETTINGS_SCHEMA_VERSION}", ""]
    for model in models:
        if (
            normalize_windows_relative(model.install_dirname) != model.install_dirname
            or "/" in model.install_dirname
        ):
            raise ValueError("Setup model directory must be a safe single component")
        if not model.files or len(model.files) >= 100:
            raise ValueError("Setup requires between one and 99 files per model")
        for asset in model.files:
            normalize_windows_relative(asset.relative_path)
            if (
                type(asset.size_bytes) is not int
                or asset.size_bytes < 0
                or not re.fullmatch(r"[0-9a-f]{64}", asset.sha256)
            ):
                raise ValueError("Setup assets require complete pinned sizes and SHA256 values")
        if canonical:
            relative = LOCAL_STT_MANIFEST_RELATIVE_PATHS[model.model_id]
            digest = hashlib.sha256(
                current_runtime_layout().package_resource(relative).read_bytes()
            ).hexdigest()
            packaged = "\\app\\puripuly_heart\\" + relative.replace("/", "\\")
            lines.extend(
                [
                    f'#if GetSHA256OfFile(MyPackagedAppDir + "{packaged}") != "{digest}"',
                    "  #error Canonical ASR metadata does not match the packaged model manifest. Regenerate the include from the final payload.",
                    "#endif",
                ]
            )
    lines.extend(
        [
            "",
            "function AsrModelCount(): Integer;",
            "begin",
            f"  Result := {len(models)};",
            "end;",
            "",
        ]
    )

    def function(name, result_type, rows, *, files=False, source=False):
        arguments = "ModelIndex: Integer"
        if files:
            arguments += "; FileIndex: Integer"
        if source:
            arguments += "; SourceName: String"
        selector = "(ModelIndex * 100) + FileIndex" if files else "ModelIndex"
        lines.extend(
            [f"function {name}({arguments}): {result_type};", "begin", f"  case {selector} of"]
        )
        lines.extend(f"    {key}: Result := {value};" for key, value in rows)
        lines.extend(
            [
                "  else",
                "    RaiseException('Unknown canonical ASR model or file index.');",
                "  end;",
                "end;",
                "",
            ]
        )

    function(
        "AsrModelId", "String", [(i, _string(model.model_id)) for i, model in enumerate(models)]
    )
    function(
        "AsrInstallDirname",
        "String",
        [(i, _string(model.install_dirname)) for i, model in enumerate(models)],
    )
    function(
        "AsrFileCount", "Integer", [(i, str(len(model.files))) for i, model in enumerate(models)]
    )
    function(
        "AsrModelSize",
        "Int64",
        [
            (i, str(sum(asset.size_bytes for asset in model.files)))
            for i, model in enumerate(models)
        ],
    )
    files = [
        (i * 100 + j, model, asset)
        for i, model in enumerate(models)
        for j, asset in enumerate(model.files)
    ]
    function(
        "AsrFilePath",
        "String",
        [(key, _string(asset.relative_path.replace("/", "\\"))) for key, model, asset in files],
        files=True,
    )
    function(
        "AsrFileSha256",
        "String",
        [(key, _string(asset.sha256)) for key, model, asset in files],
        files=True,
    )
    function(
        "AsrFileSize",
        "Int64",
        [(key, str(asset.size_bytes)) for key, model, asset in files],
        files=True,
    )
    function(
        "AsrPrimaryUrl",
        "String",
        [(key, _string(_url(model, asset, "huggingface"))) for key, model, asset in files],
        files=True,
    )
    function(
        "AsrMirrorUrl",
        "String",
        [(key, _string(_url(model, asset, "hf_mirror"))) for key, model, asset in files],
        files=True,
    )
    lines.extend(
        [
            "function AsrSourceRevision(ModelIndex: Integer; SourceName: String): String;",
            "begin",
            "  if (SourceName <> 'huggingface') and (SourceName <> 'hf_mirror') then",
            "    RaiseException('Unsupported canonical ASR source.');",
            "  case ModelIndex of",
        ]
    )
    for index, model in enumerate(models):
        primary = _string(_source(model, "huggingface").revision)
        mirror = _string(_source(model, "hf_mirror").revision)
        lines.extend(
            [
                f"    {index}: if SourceName = 'hf_mirror' then Result := {mirror}",
                f"       else Result := {primary};",
            ]
        )
    lines.extend(
        [
            "  else",
            "    RaiseException('Unknown canonical ASR model index.');",
            "  end;",
            "end;",
            "",
        ]
    )
    rows = []
    for index, model in enumerate(models):
        installed = InstalledLocalSTTManifest(
            model.installed_manifest_version,
            model.model_id,
            model.engine,
            model.install_dirname,
            "__ASR_SOURCE__",
            "__ASR_REVISION__",
        )
        literal = _string(json.dumps(installed.to_dict(), indent=2) + "\n")
        literal = literal.replace("__ASR_SOURCE__", "' + SourceName + '")
        literal = literal.replace(
            "__ASR_REVISION__", "' + AsrSourceRevision(ModelIndex, SourceName) + '"
        )
        rows.append((index, literal))
    function("AsrInstalledManifest", "String", rows, source=True)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = render_model_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(data, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
