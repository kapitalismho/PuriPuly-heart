from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Sequence

from puripuly_heart.release_evidence.windows_product_metadata import PRODUCT_NAME, version_tuple

RELEASE_IDENTITY_SCHEMA = "puripuly-heart/release-identity/v1"
SOXR_BUNDLE_FILENAME = "PuriPulyHeart-soxr-third-party-source-bundle.zip"
PLACEHOLDER = "{{INSTALLER_EXE}}"
LOCAL_ORIGIN = "local"
HOSTED_ORIGIN = "github-hosted"
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")

PACKAGED_LICENSE_PAYLOADS = (
    (
        "puripuly_heart/data/licenses/SCIPY-1.18.0-LICENSE.txt",
        "f0f5c56b298ec9795df1199edc328a987242716d14a1bde6813112e22dc15f99",
    ),
    (
        "puripuly_heart/data/licenses/PYTHON-3.12.10-LICENSE.txt",
        "e502c6b880ff58d614901495a9009c136539cd0b1e2a2abb8fc00b934c203419",
    ),
    (
        "scipy/_lib/_uarray/LICENSE",
        "e4c4acc31e8287066d7c995d26e3ef902ca1977ee187b260b11c278366503811",
    ),
    (
        "scipy/fft/_duccfft/LICENSE.md",
        "c04661685cff9d8035fe1c4c3ab357e7f7633cafc06a73e9d90c3c5fede8a86a",
    ),
    (
        "scipy/integrate/LICENSE_DOP",
        "6cc2ed6e35b8f376ff3d8f86aeef4e0c34c2309bd6f4a0565936d577839ba023",
    ),
    (
        "scipy/spatial/qhull_src/COPYING_QHULL.txt",
        "2a6ebb495e26dd5f88cbc40c593733da6edb27b64936f70409cd5a1acafb1c32",
    ),
    (
        "zeroconf-0.150.0.dist-info/licenses/COPYING",
        "9c35fa0c7991bb24076471ac2abb0ca6b8a26d393bb7a004193d18f71e5714a0",
    ),
    ("onnxruntime/LICENSE", "c250d6278f0b47a6439fb7592b08b58a55eb9f535aa49a1db63211c3f982b674"),
    (
        "onnxruntime/ThirdPartyNotices.txt",
        "fb0af774b4d7cffc5b9d046f2aaeade2f37df2f80abf8033c95dfffcc77a8866",
    ),
    (
        "aiohttp-3.13.2.dist-info/licenses/LICENSE.txt",
        "c1493e9f10d59d1fba9f9df28008e1557e33cf9fbac8ce1263aa3393982803de",
    ),
    (
        "aiohttp-3.13.2.dist-info/licenses/vendor/llhttp/LICENSE",
        "6ddfa628db76d2d87b8968bafbad60f51c4ec88100dd7bb96f97216d88af0808",
    ),
    (
        "sherpa_onnx-1.13.4.dist-info/licenses/LICENSE",
        "3ddf9be5c28fe27dad143a5dc76eea25222ad1dd68934a047064e56ed2fa40c5",
    ),
    (
        "sounddevice-0.5.5.dist-info/licenses/LICENSE",
        "b6eea21bbafbaf7a2177e492f1be6b5efc6f4e708b1980773de51cdaa52f8211",
    ),
    (
        "charset_normalizer-3.4.4.dist-info/licenses/LICENSE",
        "18577485d3704f1a479ded8e573c0976cfed315fd2fd17983fa988da4c2f70d1",
    ),
    (
        "cffi-2.0.0.dist-info/licenses/AUTHORS",
        "2a67a60bbfb33759d67d645ff131b8e6d6bc4cafc232d751fcac3eda4d314cc8",
    ),
    (
        "cffi-2.0.0.dist-info/licenses/LICENSE",
        "5ba24ddc57067f9249add644c3afc41a5d6dc37e23433ef759d95df370b0af63",
    ),
    (
        "pyaudiowpatch-0.2.12.8.dist-info/licenses/LICENSE.txt",
        "2b94fbbfe7259b16c5426692398ac816c520b8d0e9d234330df05ac75331f0a6",
    ),
    (
        "frozenlist-1.8.0.dist-info/licenses/LICENSE",
        "6fd5243e92dd7f98ec69c7ac377728e74905709ff527a5bf98d6d0263c04f5b6",
    ),
    (
        "msgpack-1.2.1.dist-info/licenses/COPYING",
        "4fbdff42eba4593c16f7a7ea7078883b2b734870a1e2b636284ce2dce62bf4f7",
    ),
    (
        "multidict-6.7.0.dist-info/licenses/LICENSE",
        "93d11a968e2f0f36373c40811ff6d20e173f58c3cab5884cd6617bbfd795492a",
    ),
    (
        "propcache-0.4.1.dist-info/licenses/LICENSE",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    (
        "propcache-0.4.1.dist-info/licenses/NOTICE",
        "56d6ac6c8105c0a51304c21db060e361af9a8ea0af9a75c239c28b5d13693838",
    ),
    (
        "psutil-7.2.2.dist-info/LICENSE",
        "c7adc4d5d1337a548b967421f1fbe258b93033a0417708fd6f4e38f8ecbceb80",
    ),
    (
        "pyyaml-6.0.3.dist-info/licenses/LICENSE",
        "8d3928f9dc4490fd635707cb88eb26bd764102a7282954307d3e5167a577e8a4",
    ),
    (
        "yarl-1.22.0.dist-info/licenses/LICENSE",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    (
        "yarl-1.22.0.dist-info/licenses/NOTICE",
        "56d6ac6c8105c0a51304c21db060e361af9a8ea0af9a75c239c28b5d13693838",
    ),
)
SOXR_LICENSE_SOURCE_PARTS = ("src", "puripuly_heart", "data", "licenses", "COPYING.LGPL-2.1.txt")
SOXR_MANIFEST_PARTS = ("build", "soxr-release-inputs", "manifest.json")
SOXR_COMPLIANCE_PARTS = ("third_party", "soxr")
SOXR_LICENSE_FILENAME = "COPYING.LGPL-2.1.txt"
SOXR_BUNDLE_MANIFEST_ENTRY = "manifest.json"


def installer_filename(version: str) -> str:
    text = version.strip()
    if not text:
        raise RuntimeError("release version is blank")
    version_tuple(text)
    return f"PuriPulyHeart-Setup-{text}.exe"


def check_tag_matches_version(tag: str, version: str) -> str:
    expected_version = version.strip()
    version_tuple(expected_version)
    expected_tag = f"v{expected_version}"
    if tag.strip() != expected_tag:
        raise RuntimeError(f"release tag mismatch: expected {expected_tag!r}, found {tag!r}")
    return expected_tag


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, int | str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"release artifact not found: {path}")
    return {"size": resolved.stat().st_size, "sha256": _sha256(resolved)}


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
        except (AttributeError, IndexError):
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


def render_release_body(template_text: str, installer_exe: str) -> str:
    return template_text.replace(PLACEHOLDER, installer_exe)


def verify_release_surface(
    *,
    version: str,
    tag: str,
    title: str,
    installer_exe: str,
    body_text: str,
    asset_names: Sequence[str],
) -> str:
    expected_version = version.strip()
    if not expected_version:
        raise RuntimeError("release version is blank")
    try:
        version_tuple(expected_version)
    except ValueError as exc:
        raise RuntimeError(f"release version is not a dotted numeric version: {version!r}") from exc
    expected_tag = f"v{expected_version}"
    if tag.strip() != expected_tag:
        raise RuntimeError(f"release tag mismatch: expected {expected_tag!r}, found {tag!r}")
    if title.strip() != tag.strip():
        raise RuntimeError(f"release title mismatch: expected {tag!r}, found {title!r}")
    expected_installer = f"PuriPulyHeart-Setup-{expected_version}.exe"
    if installer_exe.strip() != expected_installer:
        raise RuntimeError(
            f"installer filename mismatch: expected {expected_installer!r}, "
            f"found {installer_exe!r}"
        )
    if PLACEHOLDER in body_text:
        raise RuntimeError(
            "release body still contains an unrendered {{INSTALLER_EXE}} placeholder"
        )
    if expected_installer not in body_text:
        raise RuntimeError(
            f"release body does not reference the produced installer {expected_installer!r}"
        )
    basenames = sorted(Path(name).name for name in asset_names)
    if expected_installer not in basenames:
        raise RuntimeError(
            f"release assets do not include the produced installer "
            f"{expected_installer!r}: {basenames}"
        )
    return expected_installer


def _require_sha40(value: str, label: str) -> str:
    text = value.strip().lower()
    if not _SHA40.fullmatch(text):
        raise RuntimeError(f"{label} must be a 40-character lowercase hex SHA: {value!r}")
    return text


def _require_sha64(value: str, label: str) -> str:
    text = value.strip().lower()
    if not _SHA64.fullmatch(text):
        raise RuntimeError(f"{label} must be a 64-character lowercase hex SHA-256: {value!r}")
    return text


def build_provenance(
    *,
    version: str,
    tag: str,
    source_sha: str,
    build_origin: str,
    repository: str = "",
    server_url: str = "",
    run_id: str = "",
    run_attempt: str = "",
    artifacts: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    expected_version = version.strip()
    version_tuple(expected_version)
    expected_tag = check_tag_matches_version(tag, expected_version)
    sha = _require_sha40(source_sha, "source SHA")
    if build_origin not in (LOCAL_ORIGIN, HOSTED_ORIGIN):
        raise RuntimeError(f"build origin must be {LOCAL_ORIGIN!r} or {HOSTED_ORIGIN!r}")
    numbers = version_tuple(expected_version)
    expected_binary = f"{numbers[0]}.{numbers[1]}.{numbers[2]}.{numbers[3]}"
    normalized_artifacts: list[dict[str, object]] = []
    for entry in artifacts:
        filename = str(entry.get("filename", "")).strip()
        if not filename or "/" in filename or "\\" in filename:
            raise RuntimeError(f"provenance artifact filename must be a bare name: {filename!r}")
        size = entry.get("size")
        sha256 = _require_sha64(str(entry.get("sha256", "")), f"artifact {filename} SHA-256")
        if not isinstance(size, int) or size < 0:
            raise RuntimeError(f"artifact {filename} size must be a non-negative int")
        normalized: dict[str, object] = {
            "role": str(entry.get("role", "")).strip(),
            "filename": filename,
            "size": size,
            "sha256": sha256,
        }
        if not normalized["role"]:
            raise RuntimeError(f"artifact {filename} role is blank")
        for field in ("product_name", "product_version", "file_description"):
            if field in entry:
                normalized[field] = str(entry[field]).strip()
        if "product_name" in normalized and normalized["product_name"] != PRODUCT_NAME:
            raise RuntimeError(f"artifact {filename} product name is not {PRODUCT_NAME!r}")
        if "product_version" in normalized and normalized["product_version"] not in {
            expected_version,
            expected_binary,
        }:
            raise RuntimeError(
                f"artifact {filename} product version disagrees with {expected_version!r}"
            )
        normalized_artifacts.append(normalized)
    normalized_artifacts.sort(key=lambda item: str(item["filename"]).casefold())
    provenance: dict[str, object] = {
        "schema": RELEASE_IDENTITY_SCHEMA,
        "product_name": PRODUCT_NAME,
        "version": expected_version,
        "tag": expected_tag,
        "source_sha": sha,
        "build_origin": build_origin,
        "artifacts": normalized_artifacts,
    }
    if build_origin == HOSTED_ORIGIN:
        repo = repository.strip()
        server = server_url.strip()
        run = run_id.strip()
        attempt = run_attempt.strip()
        if not repo or not server or not run or not attempt:
            raise RuntimeError(
                "github-hosted provenance requires repository, server URL, run id, and run attempt"
            )
        if not run.isdigit() or not attempt.isdigit():
            raise RuntimeError("github-hosted run id and run attempt must be decimal strings")
        provenance["repository"] = repo
        provenance["workflow"] = {
            "server_url": server,
            "run_id": run,
            "run_attempt": attempt,
        }
    else:
        if run_id.strip() or run_attempt.strip():
            raise RuntimeError("local provenance must not carry a workflow run id or attempt")
        if repository.strip():
            provenance["repository"] = repository.strip()
    return provenance


def write_provenance_file(path: Path, provenance: dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def verify_assets_against_provenance(
    provenance: dict[str, object],
    asset_paths: Sequence[Path],
) -> None:
    if provenance.get("schema") != RELEASE_IDENTITY_SCHEMA:
        raise RuntimeError("provenance schema mismatch")
    entries: dict[str, dict[str, object]] = {}
    raw_artifacts = provenance.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise RuntimeError("provenance carries no artifacts")
    for entry in raw_artifacts:
        if not isinstance(entry, dict):
            raise RuntimeError("provenance artifact entry is malformed")
        entries[str(entry.get("filename", "")).strip()] = entry
    if not asset_paths:
        raise RuntimeError("no release assets supplied for provenance verification")
    for asset in asset_paths:
        resolved = Path(asset).resolve()
        if not resolved.is_file():
            raise RuntimeError(f"release asset not found: {asset}")
        entry = entries.get(resolved.name)
        if entry is None:
            raise RuntimeError(
                f"release asset {resolved.name!r} has no provenance record; "
                f"provenance is detached from the shipped bytes"
            )
        actual = file_identity(resolved)
        if (
            int(entry.get("size", -1)) != actual["size"]
            or str(entry.get("sha256", "")).lower() != str(actual["sha256"]).lower()
        ):
            raise RuntimeError(
                f"release asset {resolved.name!r} disagrees with its provenance record"
            )


def _package_relative(package_dir: Path, relative: str) -> Path:
    parts = [part for part in relative.replace("\\", "/").split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise RuntimeError(f"packaged payload path escapes the package: {relative!r}")
    return package_dir.joinpath(*parts)


def verify_packaged_license_payloads(package_dir: Path) -> list[dict[str, object]]:
    root = Path(package_dir).resolve()
    if not root.is_dir():
        raise RuntimeError(f"packaged application directory not found: {package_dir}")
    verified: list[dict[str, object]] = []
    for relative, expected in PACKAGED_LICENSE_PAYLOADS:
        path = _package_relative(root, relative)
        if not path.is_file():
            raise RuntimeError(f"packaged upstream license payload not found: {relative}")
        actual = _sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"packaged upstream license payload sha256 mismatch: {relative}")
        verified.append(
            {
                "path": relative.replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": actual,
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
    bundle_path = Path(str(bundle_field).strip())
    if not bundle_path.is_absolute():
        bundle_path = root / bundle_path
    bundle_path = bundle_path.resolve()
    if not bundle_path.is_file():
        raise RuntimeError(f"soxr third-party source bundle not found: {bundle_path}")
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
            digest = hashlib.sha256()
            with archive.open(filename) as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            matches = [
                item
                for item in sources
                if isinstance(item, dict) and str(item.get("filename", "")).strip() == filename
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"soxr third-party source bundle manifest must describe "
                    f"{filename} exactly once"
                )
            expected = str(matches[0].get("sha256", "")).strip().lower()
            if not _SHA64.fullmatch(expected):
                raise RuntimeError(
                    f"soxr third-party source bundle manifest carries a malformed "
                    f"sha256 for {filename}"
                )
            if actual.lower() != expected:
                raise RuntimeError(f"soxr third-party source bundle hash mismatch for {filename}")
            modifications = matches[0].get("modifications", [])
            if modifications is None:
                modifications = []
            for modification in modifications:
                name = str(modification).strip()
                if name not in entries:
                    raise RuntimeError(
                        f"soxr third-party source bundle is missing modification: {name}"
                    )
            validated_sources.append({"filename": filename, "sha256": actual})
    bundle_identity = file_identity(bundle_path)
    license_source = root.joinpath(*SOXR_LICENSE_SOURCE_PARTS)
    if not license_source.is_file():
        raise RuntimeError(f"soxr LGPL license text not found: {license_source}")
    compliance_dir = package_root.joinpath(*SOXR_COMPLIANCE_PARTS)
    packaged_license = compliance_dir / SOXR_LICENSE_FILENAME
    if not packaged_license.is_file():
        raise RuntimeError(f"packaged soxr LGPL license text not found: {packaged_license}")
    if _sha256(packaged_license) != _sha256(license_source):
        raise RuntimeError("packaged soxr LGPL license text hash does not match the staged source")
    packaged_bundle = compliance_dir / bundle_path.name
    if not packaged_bundle.is_file():
        raise RuntimeError(f"packaged soxr source bundle not found: {packaged_bundle}")
    if _sha256(packaged_bundle) != bundle_identity["sha256"]:
        raise RuntimeError(
            "packaged soxr source bundle hash does not match the staged source bundle"
        )
    return {
        "bundle": {
            "filename": bundle_path.name,
            "size": bundle_identity["size"],
            "sha256": bundle_identity["sha256"],
            "sources": validated_sources,
        },
        "compliance": {
            "license": SOXR_LICENSE_FILENAME,
            "bundle": bundle_path.name,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("verify-build")
    build_parser.add_argument("--version", required=True)
    build_parser.add_argument("--tag", required=True)
    build_parser.add_argument("--source-sha", required=True)
    build_parser.add_argument("--build-origin", default=LOCAL_ORIGIN)
    build_parser.add_argument("--repository", default="")
    build_parser.add_argument("--server-url", default="")
    build_parser.add_argument("--run-id", default="")
    build_parser.add_argument("--run-attempt", default="")
    build_parser.add_argument("--main-exe", type=Path, required=True)
    build_parser.add_argument("--gpu-worker-exe", type=Path, required=True)
    build_parser.add_argument("--overlay-exe", type=Path, required=True)
    build_parser.add_argument("--installer", type=Path, required=True)
    build_parser.add_argument("--soxr-bundle", type=Path, default=None)
    build_parser.add_argument("--provenance-out", type=Path, required=True)
    render_parser = subparsers.add_parser("render-body")
    render_parser.add_argument("--template", type=Path, required=True)
    render_parser.add_argument("--installer-exe", required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    publish_parser = subparsers.add_parser("verify-publish")
    publish_parser.add_argument("--version", required=True)
    publish_parser.add_argument("--tag", required=True)
    publish_parser.add_argument("--title", required=True)
    publish_parser.add_argument("--installer-exe", required=True)
    publish_parser.add_argument("--body", type=Path, required=True)
    publish_parser.add_argument("--asset", dest="assets", type=Path, action="append", required=True)
    publish_parser.add_argument("--provenance", type=Path, default=None)
    licenses_parser = subparsers.add_parser("verify-packaged-licenses")
    licenses_parser.add_argument("--package-dir", type=Path, required=True)
    licenses_parser.add_argument("--repo-root", type=Path, default=None)
    licenses_parser.add_argument("--soxr-manifest", type=Path, default=None)
    return parser


def _run_verify_build(arguments: argparse.Namespace) -> dict[str, object]:
    expected_version = arguments.version.strip()
    check_tag_matches_version(arguments.tag, expected_version)
    expected_installer = installer_filename(expected_version)
    if Path(arguments.installer).name != expected_installer:
        raise RuntimeError(
            f"installer filename mismatch: expected {expected_installer!r}, "
            f"found {Path(arguments.installer).name!r}"
        )
    if (
        arguments.soxr_bundle is not None
        and Path(arguments.soxr_bundle).name != SOXR_BUNDLE_FILENAME
    ):
        raise RuntimeError(
            f"compliance bundle filename mismatch: expected {SOXR_BUNDLE_FILENAME!r}, "
            f"found {Path(arguments.soxr_bundle).name!r}"
        )
    main_metadata = verify_pe_product_metadata(
        arguments.main_exe, expected_version=expected_version
    )
    gpu_metadata = verify_pe_product_metadata(
        arguments.gpu_worker_exe, expected_version=expected_version
    )
    overlay_metadata = verify_pe_product_metadata(
        arguments.overlay_exe, expected_version=expected_version
    )
    installer_metadata = verify_pe_product_metadata(
        arguments.installer, expected_version=expected_version
    )
    candidates: list[tuple[str, Path, dict[str, str]]] = [
        ("main-executable", Path(arguments.main_exe), main_metadata),
        ("gpu-worker-executable", Path(arguments.gpu_worker_exe), gpu_metadata),
        ("overlay-executable", Path(arguments.overlay_exe), overlay_metadata),
        ("installer", Path(arguments.installer), installer_metadata),
    ]
    artifacts: list[dict[str, object]] = []
    for role, path, metadata in candidates:
        identity = file_identity(path)
        artifacts.append(
            {
                "role": role,
                "filename": path.resolve().name,
                "size": identity["size"],
                "sha256": identity["sha256"],
                "product_name": metadata.get("ProductName", "").strip(),
                "product_version": metadata.get("ProductVersion", "").strip(),
                "file_description": metadata.get("FileDescription", "").strip(),
            }
        )
    if arguments.soxr_bundle is not None:
        bundle_identity = file_identity(Path(arguments.soxr_bundle))
        artifacts.append(
            {
                "role": "compliance-bundle",
                "filename": Path(arguments.soxr_bundle).resolve().name,
                "size": bundle_identity["size"],
                "sha256": bundle_identity["sha256"],
            }
        )
    provenance = build_provenance(
        version=expected_version,
        tag=arguments.tag,
        source_sha=arguments.source_sha,
        build_origin=arguments.build_origin,
        repository=arguments.repository,
        server_url=arguments.server_url,
        run_id=arguments.run_id,
        run_attempt=arguments.run_attempt,
        artifacts=artifacts,
    )
    write_provenance_file(Path(arguments.provenance_out), provenance)
    return provenance


def _run_render_body(arguments: argparse.Namespace) -> dict[str, object]:
    template_text = Path(arguments.template).read_text(encoding="utf-8")
    installer_exe = arguments.installer_exe.strip()
    body = render_release_body(template_text, installer_exe)
    if PLACEHOLDER in body:
        raise RuntimeError(
            "release body still contains an unrendered {{INSTALLER_EXE}} placeholder"
        )
    if installer_exe not in body:
        raise RuntimeError(f"rendered release body does not reference {installer_exe!r}")
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    return {"installer_exe": installer_exe, "body_path": str(output)}


def _run_verify_publish(arguments: argparse.Namespace) -> dict[str, object]:
    body_text = Path(arguments.body).read_text(encoding="utf-8")
    asset_paths = [Path(item) for item in arguments.assets]
    expected_installer = verify_release_surface(
        version=arguments.version,
        tag=arguments.tag,
        title=arguments.title,
        installer_exe=arguments.installer_exe,
        body_text=body_text,
        asset_names=[path.name for path in asset_paths],
    )
    summary: dict[str, object] = {
        "version": arguments.version.strip(),
        "tag": arguments.tag.strip(),
        "installer_exe": expected_installer,
        "assets": sorted(path.name for path in asset_paths),
    }
    if arguments.provenance is not None:
        provenance_path = Path(arguments.provenance).resolve()
        if not provenance_path.is_file():
            raise RuntimeError(f"provenance record not found: {arguments.provenance}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if str(provenance.get("version", "")).strip() != arguments.version.strip():
            raise RuntimeError("provenance version disagrees with the release version")
        if str(provenance.get("tag", "")).strip() != arguments.tag.strip():
            raise RuntimeError("provenance tag disagrees with the release tag")
        verify_assets_against_provenance(provenance, asset_paths)
        summary["provenance"] = str(provenance_path)
        summary["build_origin"] = provenance.get("build_origin")
    return summary


def _run_verify_packaged_licenses(arguments: argparse.Namespace) -> dict[str, object]:
    package_dir = Path(arguments.package_dir).resolve()
    if not package_dir.is_dir():
        raise RuntimeError(f"packaged application directory not found: {arguments.package_dir}")
    repo_root = (
        Path(arguments.repo_root).resolve() if arguments.repo_root is not None else Path.cwd()
    )
    payloads = verify_packaged_license_payloads(package_dir)
    soxr = verify_soxr_packaging(package_dir, repo_root, arguments.soxr_manifest)
    return {
        "package_dir": str(package_dir),
        "license_payloads": len(payloads),
        "source_bundle": soxr["bundle"],
        "compliance": soxr["compliance"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "verify-build":
        result = _run_verify_build(arguments)
    elif arguments.command == "render-body":
        result = _run_render_body(arguments)
    elif arguments.command == "verify-publish":
        result = _run_verify_publish(arguments)
    else:
        result = _run_verify_packaged_licenses(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
