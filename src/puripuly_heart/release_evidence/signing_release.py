from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse
from uuid import UUID

from puripuly_heart.release_evidence.release_identity import (
    HOSTED_ORIGIN,
    build_provenance,
    check_tag_matches_version,
    file_identity,
    installer_filename,
    verify_pe_product_metadata,
    verify_product_metadata_records,
    verify_release_surface,
    write_provenance_file,
)

COMPONENTS = (
    "PuriPulyHeart.exe",
    "PuriPulyHeartOverlay.exe",
    "PuriPulyHeartGpuWorker.exe",
    "uninstaller.exe",
)
SCHEMA = "puripuly-heart/two-stage-signing/v1"
PROJECT = "PuriPuly-heart"


def read_record(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Malformed evidence: {path}")
    return value


def exact_files(root: Path, names: Sequence[str]) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"Missing artifact directory: {root}")
    entries = list(root.iterdir())
    if sorted(p.name for p in entries) != sorted(names):
        raise RuntimeError(f"Artifact targets mismatch: {[p.name for p in entries]}")
    for entry in entries:
        if not entry.is_file() or entry.is_symlink():
            raise RuntimeError(f"Artifact is not a regular file: {entry}")
    return [root / name for name in names]


def payload_identity(path: Path) -> str:
    data = bytearray(path.read_bytes())
    if len(data) < 64 or data[:2] != b"MZ":
        raise RuntimeError(f"Not a PE signing target: {path}")
    pe = struct.unpack_from("<I", data, 60)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        raise RuntimeError(f"Invalid PE signing target: {path}")
    optional = pe + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic not in (0x10B, 0x20B):
        raise RuntimeError(f"Unsupported PE target: {path}")
    security = optional + (112 if magic == 0x20B else 96) + 32
    certificate, size = struct.unpack_from("<II", data, security)
    if certificate or size:
        if certificate % 8 or size < 8 or certificate + size != len(data):
            raise RuntimeError(f"Invalid Authenticode certificate table: {path}")
        del data[certificate:]
    data[optional + 64 : optional + 68] = b"\0" * 4
    data[security : security + 8] = b"\0" * 8
    data.extend(b"\0" * (-len(data) % 8))
    return hashlib.sha256(data).hexdigest()


def unsigned_record(path: Path, version: str) -> dict:
    metadata = verify_pe_product_metadata(path, expected_version=version)
    return {
        **file_identity(path),
        "payload_sha256": payload_identity(path),
        "metadata_resources": metadata["version_resources"],
    }


def compiler_arguments(
    script: Path, payload: Path, cleanup: Path, uninstallers: Path, output: Path
) -> list[str]:
    return [
        "/DNativeExperimental=1",
        f"/DMyPackagedAppDir={payload}",
        f"/DMyStagedOverlayDir={payload}",
        f"/DNativeCleanupInclude={cleanup}",
        f"/DSigningUninstallerDir={uninstallers}",
        f"/O{output}",
        str(script),
    ]


def prepare(
    version: str, payload: Path, root: Path, compiler: Path, script: Path, cleanup: Path
) -> dict:
    installer_filename(version)
    if root.exists():
        raise RuntimeError(f"Signing staging must be fresh: {root}")
    (root / "components").mkdir(parents=True)
    (root / "uninstallers").mkdir()
    for name in COMPONENTS[:-1]:
        unsigned_record(payload / name, version)
        shutil.copyfile(payload / name, root / "components" / name)
    result = subprocess.run(
        [
            str(compiler),
            *compiler_arguments(
                script, payload, cleanup, root / "uninstallers", root / "discarded-installer"
            ),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    (root / "prepare-compiler.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    generated = list((root / "uninstallers").iterdir())
    if (
        result.returncode == 0
        or len(generated) != 1
        or not re.fullmatch(r"uninst-[0-9.]+-[0-9a-f]+\.e(?:32|64)", generated[0].name)
    ):
        raise RuntimeError("Compiler did not stop at the generated uninstaller signing boundary")
    shutil.copyfile(generated[0], root / "components" / "uninstaller.exe")
    files = exact_files(root / "components", COMPONENTS)
    record = {
        "version": version,
        "compiler": file_identity(compiler),
        "script": file_identity(script),
        "cleanup": file_identity(cleanup),
        "uninstaller_filename": generated[0].name,
        "targets": {p.name: unsigned_record(p, version) for p in files},
        "payload": tree_identity(payload),
    }
    write_provenance_file(root / "prepared.json", record)
    return record


def tree_identity(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Payload contains a symbolic link: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_identity(path)
    return result


def verify_signature(path: Path, version: str) -> dict:
    metadata = verify_pe_product_metadata(path, expected_version=version)
    script = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "verify-authenticode.ps1"
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script), "-FilePath", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Authenticode validation failed for {path.name}: {result.stderr.strip()}"
        )
    signature = json.loads(result.stdout)
    return {
        **file_identity(path),
        "payload_sha256": payload_identity(path),
        "metadata_resources": metadata["version_resources"],
        "signature": signature,
    }


def validate_result(directory: Path, expected: dict, version: str) -> dict:
    paths = exact_files(directory, tuple(expected))
    validated = {}
    for path in paths:
        target = verify_signature(path, version)
        if target["payload_sha256"] != expected[path.name]["payload_sha256"]:
            raise RuntimeError(f"Signing changed the submitted payload: {path.name}")
        validated[path.name] = target
    return validated


def rebuild(
    version: str, payload: Path, root: Path, compiler: Path, script: Path, cleanup: Path
) -> dict:
    prepared = read_record(root / "prepared.json")
    if prepared["version"] != version:
        raise RuntimeError("Prepared version mismatch")
    for name, path in (("compiler", compiler), ("script", script), ("cleanup", cleanup)):
        if file_identity(path) != prepared[name]:
            raise RuntimeError(f"Uninstaller rebuild input mismatch: {name}")
    if tree_identity(payload) != prepared["payload"]:
        raise RuntimeError("Unsigned payload changed after component submission")
    validated = validate_result(root / "signed-components", prepared["targets"], version)
    write_provenance_file(
        root / "validated-components.json", {"version": version, "targets": validated}
    )
    handoff = root / "signed-payload"
    shutil.copytree(payload, handoff)
    for name in COMPONENTS[:-1]:
        shutil.copyfile(root / "signed-components" / name, handoff / name)
    expected_tree = dict(prepared["payload"])
    for name in COMPONENTS[:-1]:
        expected_tree[name] = {key: validated[name][key] for key in ("size", "sha256")}
    manifest_path = handoff / "native-artifact-manifest.json"
    rebuild_cleanup = cleanup
    if manifest_path.is_file():
        manifest = read_record(manifest_path)
        if manifest.get("schema") != "puripuly-heart/native-artifact-manifest/v1":
            raise RuntimeError("Native artifact manifest schema mismatch")
        for name in COMPONENTS[:-1]:
            matches = [entry for entry in manifest["inventory"] if entry.get("path") == name]
            if len(matches) != 1 or matches[0].get("sha256") != prepared["targets"][name]["sha256"]:
                raise RuntimeError(f"Native signing inventory mismatch: {name}")
            matches[0].update(bytes=validated[name]["size"], sha256=validated[name]["sha256"])
        write_provenance_file(manifest_path, manifest)
        expected_tree[manifest_path.name] = file_identity(manifest_path)
        from puripuly_heart.release_evidence.native_installer_cleanup import generate_include

        legacy = (
            Path(__file__).resolve().parents[3]
            / "native"
            / "windows_host"
            / "legacy-pyinstaller-v2.7.0.json"
        )
        rebuild_cleanup = root / "signed-native-cleanup.iss"
        generate_include(legacy, manifest_path, rebuild_cleanup)
    if tree_identity(handoff) != expected_tree:
        raise RuntimeError("Signed-byte handoff mismatch")
    uninstaller = root / "uninstallers" / prepared["uninstaller_filename"]
    exact_files(root / "uninstallers", [prepared["uninstaller_filename"]])
    shutil.copyfile(root / "signed-components" / "uninstaller.exe", uninstaller)
    output = root / "installer-candidate"
    output.mkdir()
    result = subprocess.run(
        [
            str(compiler),
            *compiler_arguments(script, handoff, rebuild_cleanup, root / "uninstallers", output),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    (root / "rebuild-compiler.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Installer rebuild from signed bytes failed")
    exact_files(root / "uninstallers", [prepared["uninstaller_filename"]])
    if file_identity(uninstaller) != {
        key: validated["uninstaller.exe"][key] for key in ("size", "sha256")
    }:
        raise RuntimeError("Compiler replaced the signed uninstaller")
    if tree_identity(handoff) != expected_tree:
        raise RuntimeError("Compiler changed the signed payload")
    installer = exact_files(output, [installer_filename(version)])[0]
    record = {
        "version": version,
        "targets": {installer.name: unsigned_record(installer, version)},
    }
    write_provenance_file(root / "installer-candidate.json", record)
    write_provenance_file(
        root / "rebuild-inputs.json",
        {
            "compiler": prepared["compiler"],
            "script": prepared["script"],
            "cleanup": file_identity(rebuild_cleanup),
            "payload": expected_tree,
            "uninstaller": file_identity(uninstaller),
        },
    )
    return record


def validate_installed(directory: Path, components: dict, version: str) -> dict:
    uninstallers = list(directory.glob("unins*.exe"))
    if len(uninstallers) != 1:
        raise RuntimeError("Installation must contain exactly one generated uninstaller")
    paths = {name: directory / name for name in COMPONENTS[:-1]}
    paths["uninstaller.exe"] = uninstallers[0]
    result = {}
    for name, path in paths.items():
        verified = verify_signature(path, version)
        if any(verified[key] != components[name][key] for key in ("size", "sha256")):
            raise RuntimeError(f"Installed signed-byte mismatch: {name}")
        result[name] = {**verified, "installed_filename": path.name}
    return result


def validate_installer(version: str, root: Path) -> dict:
    expected = read_record(root / "installer-candidate.json")
    if expected["version"] != version:
        raise RuntimeError("Installer candidate version mismatch")
    record = {
        "version": version,
        "targets": validate_result(root / "signed-installer", expected["targets"], version),
    }
    write_provenance_file(root / "validated-installer.json", record)
    return record


def verify_context(
    version: str,
    tag: str,
    source_sha: str,
    ref: str,
    workflow_ref: str,
    workflow_sha: str,
    test: bool,
) -> dict:
    check_tag_matches_version(tag, version)
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or workflow_sha != source_sha:
        raise RuntimeError("Workflow and source SHA mismatch")
    if ref != f"refs/tags/{tag}" or not workflow_ref.endswith(
        f"/.github/workflows/release-native.yml@{ref}"
    ):
        raise RuntimeError("Signing requires the verified workflow at the selected tag ref")
    if not test and os.environ.get("SIGNPATH_PRODUCTION_READY") != "true":
        raise RuntimeError(
            "Production cutover not attested: certificate, narrow origin and MFA review required"
        )
    if not os.environ.get("SIGNPATH_API_TOKEN"):
        raise RuntimeError("SIGNPATH_API_TOKEN is missing")
    if not os.environ.get("SIGNPATH_EXPECTED_PUBLISHER") or not re.fullmatch(
        r"[0-9A-Fa-f]{64}", os.environ.get("SIGNPATH_EXPECTED_ROOT_SHA256", "")
    ):
        raise RuntimeError("Pinned signing publisher/root authority is missing")
    approved_origin = os.environ.get("SIGNPATH_RELEASE_ORIGIN", "").replace("{tag}", tag)
    if not test and (not approved_origin or "*" in approved_origin):
        raise RuntimeError("Exact reviewed production tag-origin representation is required")
    return {
        "policy": "test-signing" if test else "release-signing",
        "ref": ref,
        "workflow_ref": workflow_ref,
        "workflow_sha": workflow_sha,
        "publisher": os.environ["SIGNPATH_EXPECTED_PUBLISHER"],
        "root_sha256": os.environ["SIGNPATH_EXPECTED_ROOT_SHA256"].lower(),
        "approved_origin": approved_origin if not test else None,
        "production_ready": not test,
    }


def validate_request(
    request: dict,
    phase: str,
    context: dict,
    *,
    version: str,
    repository: str,
    source_sha: str,
    run_id: str,
    run_attempt: str,
) -> dict:
    if not re.fullmatch(r"[1-9][0-9]*", str(request.get("artifact_id", ""))):
        raise RuntimeError(f"Missing uploaded artifact ID: {phase}")
    try:
        request_id = str(UUID(request.get("request_id", "")))
    except ValueError, TypeError, AttributeError:
        raise RuntimeError(f"Malformed signing request ID: {phase}") from None
    metadata = request.get("metadata", {})
    if request.get("action_outcome") != "success" or request.get("metadata_status") != "retrieved":
        raise RuntimeError(f"Signing request did not succeed: {phase}")
    expected = {
        "signingRequestId": request_id,
        "projectSlug": PROJECT,
        "artifactConfigurationSlug": phase,
        "signingPolicySlug": context["policy"],
        "status": "Completed",
        "workflowStatus": "Completed",
        "isFinalStatus": True,
        "webLink": request.get("request_url"),
    }
    for field, value in expected.items():
        if not value or metadata.get(field) != value:
            raise RuntimeError(f"Signing request metadata mismatch: {phase}/{field}")
    if metadata.get("parameters") != {"version": version}:
        raise RuntimeError(f"Signing request metadata mismatch: {phase}/parameters")
    origin = metadata.get("origin", {})
    source = origin.get("repositoryData", {})
    repo_url = str(source.get("url", "")).removesuffix(".git").rstrip("/")
    if repo_url != f"https://github.com/{repository}" or source.get("commitId") != source_sha:
        raise RuntimeError(f"Signing source provenance mismatch: {phase}")
    branch = source.get("branchName")
    if not isinstance(branch, str) or not branch or not source.get("sourceControlManagementType"):
        raise RuntimeError(f"Missing literal SignPath origin: {phase}")
    if context["policy"] == "release-signing" and branch != context.get("approved_origin"):
        raise RuntimeError("Production origin differs from the reviewed narrow origin")
    build_url = origin.get("buildData", {}).get("url", "")
    parsed = urlparse(build_url)
    expected_path = f"/{repository}/actions/runs/{run_id}"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") not in (expected_path, f"{expected_path}/attempts/{run_attempt}")
    ):
        raise RuntimeError(f"Signing workflow origin mismatch: {phase}")
    return {
        "repository_url": repo_url,
        "branch_name": branch,
        "commit_id": source_sha,
        "build_url": build_url,
    }


def validate_provenance(
    provenance: dict,
    asset: Path,
    *,
    version: str,
    tag: str,
    source_sha: str,
    repository: str,
    run_id: str,
    run_attempt: str,
) -> None:
    check_tag_matches_version(tag, version)
    if provenance.get("schema") != SCHEMA or provenance.get("status") != "validated":
        raise RuntimeError("Incomplete signing provenance")
    for field, value in (
        ("version", version),
        ("tag", tag),
        ("source_sha", source_sha),
        ("repository", repository),
    ):
        if provenance.get(field) != value:
            raise RuntimeError(f"Signing provenance mismatch: {field}")
    context = provenance.get("context", {})
    if (
        context.get("ref") != f"refs/tags/{tag}"
        or context.get("workflow_sha") != source_sha
        or context.get("workflow_ref")
        != f"{repository}/.github/workflows/release-native.yml@refs/tags/{tag}"
        or provenance.get("policy") != context.get("policy")
    ):
        raise RuntimeError("Signing source/ref provenance mismatch")
    workflow = provenance.get("workflow", {})
    if (
        provenance.get("build_origin") != HOSTED_ORIGIN
        or workflow.get("run_id") != run_id
        or workflow.get("run_attempt") != run_attempt
    ):
        raise RuntimeError("Signing workflow provenance mismatch")
    requests = provenance.get("requests", {})
    origins = [
        validate_request(
            requests.get(phase, {}),
            phase,
            context,
            version=version,
            repository=repository,
            source_sha=source_sha,
            run_id=run_id,
            run_attempt=run_attempt,
        )
        for phase in ("components", "installer")
    ]
    if (
        origins[0] != origins[1]
        or requests["components"]["request_id"] == requests["installer"]["request_id"]
        or requests["components"]["artifact_id"] == requests["installer"]["artifact_id"]
    ):
        raise RuntimeError("Two signing requests do not share a unique, consistent build origin")
    validation = provenance.get("validation", {})
    submitted = provenance.get("submitted", {})
    for group, names in (
        ("components", COMPONENTS),
        ("installed", COMPONENTS),
        ("installer", [installer_filename(version)]),
    ):
        targets = validation.get(group, {})
        if sorted(targets) != sorted(names):
            raise RuntimeError(f"Missing signing validation: {group}")
        for name, target in targets.items():
            signature = target.get("signature", {})
            if (
                signature.get("status") != "Valid"
                or not signature.get("timestamp")
                or signature.get("publisher") != context.get("publisher")
                or signature.get("root_sha256") != context.get("root_sha256")
                or not signature.get("wintrust")
            ):
                raise RuntimeError("Unsigned, unexpected or incomplete signature evidence")
            verify_product_metadata_records(
                target.get("metadata_resources"), expected_version=version
            )
            comparison = (
                validation["components"] if group == "installed" else submitted.get(group, {})
            )
            if name not in comparison or target.get("payload_sha256") != comparison[name].get(
                "payload_sha256"
            ):
                raise RuntimeError("Signing payload provenance mismatch")
            if group == "installed" and any(
                target.get(key) != comparison[name].get(key) for key in ("size", "sha256")
            ):
                raise RuntimeError("Installed byte provenance mismatch")
    if asset.name != installer_filename(version):
        raise RuntimeError("Draft asset filename mismatch")
    outer = validation["installer"][asset.name]
    if file_identity(asset) != {key: outer[key] for key in ("size", "sha256")}:
        raise RuntimeError("Draft asset differs from validated signed bytes")
    if provenance.get("uninstall") != "passed":
        raise RuntimeError("Clean uninstall validation missing")


def finalize(
    root: Path,
    evidence: Path,
    *,
    version: str,
    tag: str,
    source_sha: str,
    repository: str,
    run_id: str,
    run_attempt: str,
) -> dict:
    context = read_record(evidence / "context.json")
    result = build_provenance(
        version=version,
        tag=tag,
        source_sha=source_sha,
        build_origin=HOSTED_ORIGIN,
        repository=repository,
        server_url="https://github.com",
        run_id=run_id,
        run_attempt=run_attempt,
    )
    result.update(
        {
            "schema": SCHEMA,
            "status": "validated",
            "policy": context["policy"],
            "context": context,
            "requests": {
                phase: read_record(evidence / f"{phase}-request.json")
                for phase in ("components", "installer")
            },
            "submitted": {
                "components": read_record(root / "prepared.json")["targets"],
                "installer": read_record(root / "installer-candidate.json")["targets"],
            },
            "validation": {
                group: read_record(root / f"validated-{group}.json")["targets"]
                for group in ("components", "installer", "installed")
            },
            "uninstall": read_record(evidence / "install-result.json")["uninstall"],
            "draft_asset": (
                installer_filename(version) if context["policy"] == "release-signing" else None
            ),
        }
    )
    validate_provenance(
        result,
        root / "signed-installer" / installer_filename(version),
        version=version,
        tag=tag,
        source_sha=source_sha,
        repository=repository,
        run_id=run_id,
        run_attempt=run_attempt,
    )
    write_provenance_file(evidence / "provenance.json", result)
    return result


def verify_publish(
    provenance: dict,
    asset: Path,
    *,
    version: str,
    tag: str,
    source_sha: str,
    repository: str,
    run_id: str,
    run_attempt: str,
) -> None:
    if (
        provenance.get("schema") != SCHEMA
        or provenance.get("policy") != "release-signing"
        or provenance.get("status") != "validated"
    ):
        raise RuntimeError("Only a validated production-signed candidate can create a draft")
    if provenance.get("context", {}).get("production_ready") is not True:
        raise RuntimeError("Production cutover attestation missing")
    validate_provenance(
        provenance,
        asset,
        version=version,
        tag=tag,
        source_sha=source_sha,
        repository=repository,
        run_id=run_id,
        run_attempt=run_attempt,
    )


def verify_draft_delivery(release: dict, provenance: dict) -> dict:
    tag = provenance["tag"]
    if (
        provenance.get("policy") != "release-signing"
        or provenance.get("status") != "validated"
        or provenance.get("context", {}).get("production_ready") is not True
        or release.get("draft") is not True
        or release.get("tag_name") != tag
        or release.get("name") != tag
    ):
        raise RuntimeError(
            "Release delivery must remain a production-signed draft at the verified tag"
        )
    assets = release.get("assets", [])
    expected_name = installer_filename(provenance["version"])
    if len(assets) != 1 or assets[0].get("name") != expected_name:
        raise RuntimeError("Draft delivery contains missing or extra assets")
    asset = assets[0]
    expected = provenance["validation"]["installer"][expected_name]
    if (
        asset.get("size") != expected["size"]
        or asset.get("digest") != f"sha256:{expected['sha256']}"
        or not isinstance(asset.get("id"), int)
        or asset["id"] < 1
    ):
        raise RuntimeError("Draft release asset is detached from validated signed bytes")
    repo_url = f"https://github.com/{provenance['repository']}"
    if (
        release.get("html_url") != f"{repo_url}/releases/tag/{tag}"
        or asset.get("browser_download_url")
        != f"{repo_url}/releases/download/{tag}/{expected_name}"
        or asset.get("state") != "uploaded"
        or type(release.get("id")) is not int
        or release["id"] < 1
    ):
        raise RuntimeError("Draft asset delivery origin mismatch")
    return {
        "tag": tag,
        "source_sha": provenance["source_sha"],
        "workflow": provenance["workflow"],
        "release_url": release["html_url"],
        "release_id": release["id"],
        "asset_id": asset["id"],
        "asset_url": asset["browser_download_url"],
        "filename": expected_name,
        "sha256": expected["sha256"],
        "draft": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "rebuild"):
        command = sub.add_parser(name)
        for field in ("version", "payload", "root", "compiler", "script", "cleanup-include"):
            command.add_argument(
                f"--{field}", required=True, type=str if field == "version" else Path
            )
    validate = sub.add_parser("validate-installer")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--version", required=True)
    installed = sub.add_parser("validate-installed")
    installed.add_argument("--root", type=Path, required=True)
    installed.add_argument("--directory", type=Path, required=True)
    installed.add_argument("--version", required=True)
    request_parser = sub.add_parser("validate-request")
    request_parser.add_argument("--phase", choices=("components", "installer"), required=True)
    request_parser.add_argument("--evidence", type=Path, required=True)
    for field in ("version", "repository", "source-sha", "run-id", "run-attempt"):
        request_parser.add_argument(f"--{field}", required=True)
    context = sub.add_parser("verify-context")
    for field in ("version", "tag", "source-sha", "ref", "workflow-ref", "workflow-sha", "output"):
        context.add_argument(f"--{field}", required=True)
    context.add_argument("--test", action="store_true")
    for name in ("finalize", "verify-publish"):
        command = sub.add_parser(name)
        for field in ("version", "tag", "source-sha", "repository", "run-id", "run-attempt"):
            command.add_argument(f"--{field}", required=True)
        command.add_argument("--evidence", type=Path, required=True)
        if name == "finalize":
            command.add_argument("--root", type=Path, required=True)
        else:
            command.add_argument("--asset", type=Path, required=True)
            command.add_argument("--body", type=Path, required=True)
            command.add_argument("--candidate-id", required=True)
    draft_parser = sub.add_parser("verify-draft-delivery")
    draft_parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("prepare", "rebuild"):
        operation = prepare if args.command == "prepare" else rebuild
        result = operation(
            args.version,
            args.payload.resolve(),
            args.root.resolve(),
            args.compiler.resolve(),
            args.script.resolve(),
            args.cleanup_include.resolve(),
        )
    elif args.command == "validate-installer":
        result = validate_installer(args.version, args.root)
    elif args.command == "validate-installed":
        components = read_record(args.root / "validated-components.json")
        if components["version"] != args.version:
            raise RuntimeError("Installed version mismatch")
        result = {
            "version": args.version,
            "targets": validate_installed(args.directory, components["targets"], args.version),
        }
        write_provenance_file(args.root / "validated-installed.json", result)
    elif args.command == "verify-draft-delivery":
        result = verify_draft_delivery(
            read_record(args.evidence / "draft-delivery.json"),
            read_record(args.evidence / "provenance.json"),
        )
        write_provenance_file(args.evidence / "validated-draft-delivery.json", result)
    elif args.command == "validate-request":
        context = read_record(args.evidence / "context.json")
        result = validate_request(
            read_record(args.evidence / f"{args.phase}-request.json"),
            args.phase,
            context,
            version=args.version,
            repository=args.repository,
            source_sha=args.source_sha,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
        )
        if args.phase == "installer":
            first = validate_request(
                read_record(args.evidence / "components-request.json"),
                "components",
                context,
                version=args.version,
                repository=args.repository,
                source_sha=args.source_sha,
                run_id=args.run_id,
                run_attempt=args.run_attempt,
            )
            if result != first:
                raise RuntimeError("SignPath requests report inconsistent build origins")
    elif args.command in ("finalize", "verify-publish"):
        keywords = {
            key: getattr(args, key)
            for key in ("version", "tag", "source_sha", "repository", "run_id", "run_attempt")
        }
        if args.command == "finalize":
            result = finalize(args.root, args.evidence, **keywords)
        else:
            result = read_record(args.evidence / "provenance.json")
            delivery = read_record(args.evidence / "delivery.json")
            if delivery.get("candidate_artifact_id") != args.candidate_id or not re.fullmatch(
                r"[1-9][0-9]*", args.candidate_id
            ):
                raise RuntimeError("Validated candidate artifact identity mismatch")
            verify_release_surface(
                version=args.version,
                tag=args.tag,
                title=args.tag,
                installer_exe=args.asset.name,
                body_text=args.body.read_text(encoding="utf-8"),
                asset_names=[args.asset.name],
            )
            exact_files(args.asset.parent, [installer_filename(args.version)])
            verify_publish(result, args.asset, **keywords)
    else:
        result = verify_context(
            args.version,
            args.tag,
            args.source_sha,
            args.ref,
            args.workflow_ref,
            args.workflow_sha,
            args.test,
        )
        write_provenance_file(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
