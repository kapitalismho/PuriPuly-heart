from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Sequence

from puripuly_heart.release_evidence.windows_product_metadata import PRODUCT_NAME, version_tuple

PACKAGED_LICENSE_PATHS = (
    "puripuly_heart/data/licenses/SCIPY-1.18.0-LICENSE.txt",
    "puripuly_heart/data/licenses/PYTHON-3.14.10-LICENSE.txt",
    "puripuly_heart/data/licenses/DART-BRIDGE-1.10.0-LICENSE.txt",
    "puripuly_heart/data/licenses/GRANITE-SPEECH-5.0-LICENSE.txt",
    "puripuly_heart/data/licenses/GRANITE-SPEECH-5.0-NOTICE.txt",
    "puripuly_heart/data/licenses/GIGAAM-V3-LICENSE.txt",
    "puripuly_heart/data/licenses/GIGAAM-V3-NOTICE.txt",
    "scipy/_lib/_uarray/LICENSE",
    "scipy/fft/_duccfft/LICENSE.md",
    "scipy/integrate/LICENSE_DOP",
    "scipy/spatial/qhull_src/COPYING_QHULL.txt",
    "zeroconf-0.150.0.dist-info/licenses/COPYING",
    "onnxruntime/LICENSE",
    "onnxruntime/ThirdPartyNotices.txt",
    "sherpa_onnx-1.13.8.dist-info/licenses/LICENSE",
    "sounddevice-0.5.5.dist-info/licenses/LICENSE",
    "charset_normalizer-3.4.4.dist-info/licenses/LICENSE",
    "cffi-2.0.0.dist-info/licenses/AUTHORS",
    "cffi-2.0.0.dist-info/licenses/LICENSE",
    "pyaudiowpatch-0.2.12.8.dist-info/licenses/LICENSE.txt",
    "msgpack-1.2.1.dist-info/licenses/COPYING",
    "psutil-7.2.2.dist-info/LICENSE",
    "pyyaml-6.0.3.dist-info/licenses/LICENSE",
)
SOXR_MANIFEST_PARTS = ("build", "soxr-release-inputs", "manifest.json")
SOXR_COMPLIANCE_PARTS = ("third_party", "soxr")
SOXR_LICENSE_FILENAME = "COPYING.LGPL-2.1.txt"
SOXR_BUNDLE_MANIFEST_ENTRY = "manifest.json"


def _decode_ver_bytes(raw: object) -> str:
    if isinstance(raw, str):
        return raw.rstrip("\x00")
    if not isinstance(raw, (bytes, bytearray)):
        return str(raw)
    data = bytes(raw)
    if b"\x00" in data:
        try:
            return data.decode("utf-16le").rstrip("\x00")
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8").rstrip("\x00")
    except UnicodeDecodeError:
        return data.decode("latin1").rstrip("\x00")


def read_pe_product_metadata(exe_path: Path) -> dict[str, str]:
    resolved = Path(exe_path).resolve()
    if not resolved.is_file():
        raise RuntimeError(f"PE file not found: {exe_path}")
    try:
        import pefile
    except ImportError as exc:
        raise RuntimeError("pefile is required to read Windows PE product metadata") from exc
    try:
        pe = pefile.PE(str(resolved), fast_load=True)
    except Exception as exc:
        raise RuntimeError(f"pefile probe failed for {resolved}: {exc}") from exc
    try:
        try:
            pe.parse_data_directories(
                directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]]
            )
        except Exception as exc:
            raise RuntimeError(f"pefile resource parse failed for {resolved}: {exc}") from exc
        result: dict[str, str] = {}
        for file_info in pe.FileInfo or []:
            for entry in file_info:
                if not hasattr(entry, "StringTable"):
                    continue
                for table in entry.StringTable:
                    for key, value in table.entries.items():
                        result[_decode_ver_bytes(key)] = _decode_ver_bytes(value)
        try:
            fixed = pe.VS_FIXEDFILEINFO[0]
            filevers = (
                f"{fixed.FileVersionMS >> 16}.{fixed.FileVersionMS & 0xFFFF}."
                f"{fixed.FileVersionLS >> 16}.{fixed.FileVersionLS & 0xFFFF}"
            )
            prodvers = (
                f"{fixed.ProductVersionMS >> 16}.{fixed.ProductVersionMS & 0xFFFF}."
                f"{fixed.ProductVersionLS >> 16}.{fixed.ProductVersionLS & 0xFFFF}"
            )
        except AttributeError, IndexError:
            filevers = ""
            prodvers = ""
        result["__FileVersionBinary"] = filevers
        result["__ProductVersionBinary"] = prodvers
        return result
    finally:
        pe.close()


def verify_pe_product_metadata(
    exe_path: Path,
    *,
    expected_version: str,
    expected_product: str = PRODUCT_NAME,
) -> dict[str, str]:
    metadata = read_pe_product_metadata(exe_path)
    product = metadata.get("ProductName", "").strip()
    if product != expected_product:
        raise RuntimeError(
            f"ProductName mismatch in {exe_path}: expected {expected_product!r}, found {product!r}"
        )
    numbers = version_tuple(expected_version.strip())
    expected_binary = f"{numbers[0]}.{numbers[1]}.{numbers[2]}.{numbers[3]}"
    binary_product = metadata.get("__ProductVersionBinary", "").strip()
    if binary_product:
        if binary_product != expected_binary:
            raise RuntimeError(
                f"binary ProductVersion mismatch in {exe_path}: "
                f"expected {expected_binary}, found {binary_product}"
            )
    else:
        text_product = metadata.get("ProductVersion", "").strip()
        if text_product not in {expected_version.strip(), expected_binary}:
            raise RuntimeError(
                f"ProductVersion mismatch in {exe_path}: "
                f"expected {expected_version!r} ({expected_binary}), found {text_product!r}"
            )
    return metadata


def _package_relative(package_dir: Path, relative: str) -> Path:
    parts = [part for part in relative.replace("\\", "/").split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise RuntimeError(f"packaged payload path escapes the package: {relative!r}")
    return package_dir.joinpath(*parts)


def verify_packaged_license_payloads(
    package_dir: Path,
    *,
    application_root: str = "",
    dependency_root: str = "",
    license_paths: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    root = Path(package_dir).resolve()
    if not root.is_dir():
        raise RuntimeError(f"packaged application directory not found: {package_dir}")
    if license_paths is None:
        license_paths = PACKAGED_LICENSE_PATHS
    verified: list[dict[str, object]] = []
    for relative in license_paths:
        area_root = application_root if relative.startswith("puripuly_heart/") else dependency_root
        located_relative = "/".join(part for part in (area_root, relative) if part)
        path = _package_relative(root, located_relative)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except OSError, ValueError:
            raise RuntimeError(
                f"packaged upstream license payload not found: {located_relative}"
            ) from None
        if not resolved.is_file() or path.is_symlink():
            raise RuntimeError(f"packaged upstream license payload not found: {located_relative}")
        payload = resolved.read_bytes()
        if not payload.strip():
            raise RuntimeError(f"packaged upstream license payload is empty: {located_relative}")
        verified.append(
            {
                "path": located_relative.replace("\\", "/"),
                "size": len(payload),
            }
        )
    return verified


def verify_soxr_packaging(
    package_dir: Path,
    repo_root: Path,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    root = Path(repo_root).resolve()
    package_root = Path(package_dir).resolve()
    if not package_root.is_dir():
        raise RuntimeError(f"packaged application directory not found: {package_dir}")
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root.joinpath(*SOXR_MANIFEST_PARTS)
    )
    if not manifest.is_file():
        raise RuntimeError(f"prepared soxr release inputs manifest not found: {manifest}")
    try:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"soxr release inputs manifest is not valid JSON: {manifest}") from exc
    bundle_field = (
        manifest_payload.get("third_party_source_bundle_path")
        if isinstance(manifest_payload, dict)
        else None
    )
    if not bundle_field or not str(bundle_field).strip():
        raise RuntimeError(
            "soxr release inputs manifest is missing the third-party source bundle path"
        )
    compliance_dir = package_root.joinpath(*SOXR_COMPLIANCE_PARTS)
    bundle_path = compliance_dir / Path(str(bundle_field).strip()).name
    if not bundle_path.is_file():
        raise RuntimeError(f"packaged soxr source bundle not found: {bundle_path}")
    try:
        archive = zipfile.ZipFile(bundle_path)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"soxr third-party source bundle is not a valid zip: {bundle_path}"
        ) from exc
    with archive:
        entries = archive.namelist()
        if SOXR_BUNDLE_MANIFEST_ENTRY not in entries:
            raise RuntimeError("soxr third-party source bundle is missing manifest.json")
        try:
            bundle_manifest = json.loads(archive.read(SOXR_BUNDLE_MANIFEST_ENTRY).decode("utf-8"))
        except ValueError as exc:
            raise RuntimeError("soxr third-party source bundle manifest is not valid JSON") from exc
        sources = bundle_manifest.get("sources") if isinstance(bundle_manifest, dict) else None
        if not isinstance(sources, list) or not sources:
            raise RuntimeError("soxr third-party source bundle manifest is missing source entries")
        validated_sources: list[dict[str, object]] = []
        for source in sources:
            filename = str(source.get("filename", "")).strip() if isinstance(source, dict) else ""
            if not filename:
                raise RuntimeError(
                    "soxr third-party source bundle manifest contains a blank source filename"
                )
            if filename not in entries:
                raise RuntimeError(
                    f"soxr third-party source bundle is missing source archive: {filename}"
                )
            modifications = source.get("modifications", [])
            if modifications is None:
                modifications = []
            for modification in modifications:
                name = str(modification).strip()
                if name not in entries:
                    raise RuntimeError(
                        f"soxr third-party source bundle is missing modification: {name}"
                    )
            validated_sources.append({"filename": filename})
    packaged_license = compliance_dir / SOXR_LICENSE_FILENAME
    if not packaged_license.is_file():
        raise RuntimeError(f"packaged soxr LGPL license text not found: {packaged_license}")
    return {
        "bundle": {
            "filename": bundle_path.name,
            "size": bundle_path.stat().st_size,
            "sources": validated_sources,
        },
        "compliance": {
            "license": SOXR_LICENSE_FILENAME,
            "bundle": bundle_path.name,
        },
    }
