from __future__ import annotations

import copy
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from puripuly_heart.release_evidence import signing_release as signing

VERSION = "2.8.3"
SHA = "a" * 40
CONTEXT = {
    "policy": "release-signing",
    "ref": f"refs/tags/v{VERSION}",
    "workflow_ref": f"owner/repo/.github/workflows/release-native.yml@refs/tags/v{VERSION}",
    "workflow_sha": SHA,
    "publisher": "CN=Approved publisher",
    "root_sha256": "b" * 64,
    "approved_origin": f"v{VERSION}",
    "production_ready": True,
}
KEYWORDS = {
    "version": VERSION,
    "tag": f"v{VERSION}",
    "source_sha": SHA,
    "repository": "owner/repo",
    "run_id": "123",
    "run_attempt": "2",
}


def pe_bytes(*, signed=False, content=b"payload"):
    data = bytearray(512)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 88, 0x20B)
    data[400 : 400 + len(content)] = content
    if signed:
        struct.pack_into("<I", data, 152, 321)
        struct.pack_into("<II", data, 232, 512, 16)
        data.extend(b"certificate-data")
    return bytes(data)


def metadata_resources(version):
    return [
        {
            "resource_id": 1,
            "language": 0,
            "translation": "000004b0",
            "strings": {
                "ProductName": "PuriPuly <3",
                "ProductVersion": version,
            },
            "file_version_binary": f"{version}.0",
            "product_version_binary": f"{version}.0",
        }
    ]


def verified(path, version):
    return {
        **signing.file_identity(path),
        "payload_sha256": signing.payload_identity(path),
        "metadata_resources": metadata_resources(version),
        "signature": {
            "status": "Valid",
            "publisher": CONTEXT["publisher"],
            "root_sha256": CONTEXT["root_sha256"],
            "timestamp": {"certificate_sha256": "c" * 64},
            "wintrust": "Successfully verified",
        },
    }


def request(phase):
    number = "1" if phase == "components" else "2"
    request_id = f"{number * 8}-{number * 4}-{number * 4}-{number * 4}-{number * 12}"
    url = f"https://app.signpath.io/requests/{request_id}"
    return {
        "artifact_id": "456" if phase == "components" else "789",
        "request_id": request_id,
        "request_url": url,
        "action_outcome": "success",
        "metadata_status": "retrieved",
        "metadata": {
            "signingRequestId": request_id,
            "status": "Completed",
            "workflowStatus": "Completed",
            "isFinalStatus": True,
            "projectSlug": signing.PROJECT,
            "artifactConfigurationSlug": phase,
            "signingPolicySlug": "release-signing",
            "parameters": {"version": VERSION},
            "webLink": url,
            "origin": {
                "repositoryData": {
                    "url": "https://github.com/owner/repo.git",
                    "branchName": f"v{VERSION}",
                    "commitId": SHA,
                    "sourceControlManagementType": "git",
                },
                "buildData": {"url": "https://github.com/owner/repo/actions/runs/123/attempts/2"},
            },
        },
    }


@pytest.fixture
def candidate(tmp_path):
    asset = tmp_path / signing.installer_filename(VERSION)
    asset.write_bytes(pe_bytes(signed=True))
    components = {name: verified(asset, VERSION) for name in signing.COMPONENTS}
    outer = {asset.name: verified(asset, VERSION)}
    record = {
        "schema": signing.SCHEMA,
        "status": "validated",
        "policy": "release-signing",
        "context": copy.deepcopy(CONTEXT),
        **KEYWORDS,
        "build_origin": "github-hosted",
        "workflow": {"run_id": "123", "run_attempt": "2"},
        "requests": {phase: request(phase) for phase in ("components", "installer")},
        "submitted": {"components": copy.deepcopy(components), "installer": copy.deepcopy(outer)},
        "validation": {
            "components": components,
            "installed": copy.deepcopy(components),
            "installer": outer,
        },
        "uninstall": "passed",
    }
    return asset, record


def test_only_validated_production_candidate_can_create_draft(candidate):
    asset, record = candidate
    signing.verify_publish(record, asset, **KEYWORDS)
    record["policy"] = "test-signing"
    with pytest.raises(RuntimeError, match="production-signed"):
        signing.verify_publish(record, asset, **KEYWORDS)


@pytest.mark.parametrize(
    "mutation",
    [
        "unsigned",
        "timestamp",
        "publisher",
        "root",
        "metadata",
        "missing-resource-proof",
        "hidden-resource-mismatch",
        "version",
        "installed",
        "payload",
        "bytes",
        "uninstall",
        "request",
        "rejected",
        "artifact",
        "ref",
        "source",
        "attempt",
        "extra",
        "missing",
        "cutover",
    ],
)
def test_unsigned_or_detached_candidate_never_passes_draft_gate(candidate, mutation):
    asset, record = candidate
    target = record["validation"]["installer"][asset.name]
    if mutation == "unsigned":
        target["signature"]["status"] = "NotSigned"
    elif mutation == "timestamp":
        target["signature"]["timestamp"] = None
    elif mutation == "publisher":
        target["signature"]["publisher"] = "CN=Imposter"
    elif mutation == "root":
        target["signature"]["root_sha256"] = "d" * 64
    elif mutation == "metadata":
        target["metadata_resources"][0]["strings"]["ProductName"] = "Other product"
    elif mutation == "missing-resource-proof":
        target.pop("metadata_resources")
    elif mutation == "hidden-resource-mismatch":
        hidden = copy.deepcopy(target["metadata_resources"][0])
        hidden["strings"]["ProductName"] = "Other product"
        target["metadata_resources"].insert(0, hidden)
    elif mutation == "version":
        target["metadata_resources"][0]["strings"]["ProductVersion"] = "9.0.0"
    elif mutation == "installed":
        record["validation"]["installed"][signing.COMPONENTS[0]]["sha256"] = "e" * 64
    elif mutation == "payload":
        target["payload_sha256"] = "e" * 64
    elif mutation == "bytes":
        asset.write_bytes(b"unsigned replacement")
    elif mutation == "uninstall":
        record["uninstall"] = "failed"
    elif mutation == "request":
        record["requests"]["installer"]["request_id"] = ""
    elif mutation == "rejected":
        record["requests"]["installer"]["metadata"]["status"] = "Denied"
    elif mutation == "artifact":
        record["requests"]["components"]["artifact_id"] = ""
    elif mutation == "ref":
        record["context"]["ref"] = "refs/heads/main"
    elif mutation == "source":
        record["source_sha"] = "f" * 40
    elif mutation == "attempt":
        record["workflow"]["run_attempt"] = "1"
    elif mutation == "extra":
        record["validation"]["components"]["upstream.exe"] = target
    elif mutation == "missing":
        del record["validation"]["components"]["uninstaller.exe"]
    else:
        record["context"]["production_ready"] = False
    with pytest.raises(RuntimeError):
        signing.verify_publish(record, asset, **KEYWORDS)


@pytest.mark.parametrize(
    "field,value",
    [
        ("projectSlug", "other"),
        ("artifactConfigurationSlug", "installer"),
        ("signingPolicySlug", "test-signing"),
        ("status", "Failed"),
        ("workflowStatus", "Denied"),
        ("isFinalStatus", False),
        ("signingRequestId", "3" * 36),
        ("origin", {}),
    ],
)
def test_request_metadata_must_match_uploaded_phase_and_verified_source(field, value):
    record = request("components")
    record["metadata"][field] = value
    with pytest.raises(RuntimeError):
        signing.validate_request(
            record,
            "components",
            CONTEXT,
            version=VERSION,
            repository="owner/repo",
            source_sha=SHA,
            run_id="123",
            run_attempt="2",
        )


def test_literal_tag_origin_is_recorded_and_both_requests_must_agree(candidate):
    asset, record = candidate
    record["context"]["policy"] = record["policy"] = "test-signing"
    for item in record["requests"].values():
        item["metadata"]["signingPolicySlug"] = "test-signing"
        item["metadata"]["origin"]["repositoryData"][
            "branchName"
        ] = "literal-unobserved-representation"
    signing.validate_provenance(record, asset, **KEYWORDS)
    record["requests"]["installer"]["metadata"]["origin"]["repositoryData"][
        "branchName"
    ] = "other-origin"
    with pytest.raises(RuntimeError, match="consistent build origin"):
        signing.validate_provenance(record, asset, **KEYWORDS)


@pytest.mark.parametrize("extra", ["upstream.exe", "nested", "PuriPulyHeart.EXE"])
def test_explicit_artifact_selection_rejects_extra_or_renamed_targets(tmp_path, extra):
    for name in signing.COMPONENTS:
        (tmp_path / name).write_bytes(pe_bytes())
    if extra == "PuriPulyHeart.EXE":
        (tmp_path / "PuriPulyHeart.exe").unlink()
    (tmp_path / extra).write_bytes(b"extra")
    with pytest.raises(RuntimeError, match="targets mismatch"):
        signing.exact_files(tmp_path, signing.COMPONENTS)


def test_signing_may_only_change_authenticode_fields(tmp_path, monkeypatch):
    unsigned = tmp_path / "unsigned.exe"
    unsigned.write_bytes(pe_bytes())
    signed = tmp_path / "signed"
    signed.mkdir()
    result = signed / "unsigned.exe"
    result.write_bytes(pe_bytes(signed=True))
    monkeypatch.setattr(signing, "verify_signature", verified)
    expected = {unsigned.name: {"payload_sha256": signing.payload_identity(unsigned)}}
    assert (
        signing.validate_result(signed, expected, VERSION)[unsigned.name]["sha256"]
        != signing.file_identity(unsigned)["sha256"]
    )
    result.write_bytes(pe_bytes(signed=True, content=b"changed"))
    with pytest.raises(RuntimeError, match="changed the submitted payload"):
        signing.validate_result(signed, expected, VERSION)


def test_rebuild_consumes_exact_signed_bytes_and_preserves_upstream_payload(tmp_path, monkeypatch):
    payload, root = tmp_path / "payload", tmp_path / "signing"
    payload.mkdir()
    (root / "signed-components").mkdir(parents=True)
    (root / "uninstallers").mkdir()
    compiler, script, cleanup = [
        tmp_path / name for name in ("compiler.exe", "installer.iss", "cleanup.iss")
    ]
    for path in (compiler, script, cleanup):
        path.write_bytes(b"build-input")
    for name in signing.COMPONENTS[:-1]:
        (payload / name).write_bytes(pe_bytes())
    (payload / "upstream.exe").write_bytes(b"upstream signature unchanged")
    signing.write_provenance_file(
        payload / "native-artifact-manifest.json",
        {
            "schema": "puripuly-heart/native-artifact-manifest/v1",
            "inventory": [
                {
                    "path": name,
                    "bytes": (payload / name).stat().st_size,
                    "sha256": signing.file_identity(payload / name)["sha256"],
                }
                for name in signing.COMPONENTS[:-1]
            ],
        },
    )
    for name in signing.COMPONENTS:
        (root / "signed-components" / name).write_bytes(pe_bytes(signed=True))
    (root / "uninstallers" / "uninst-generated.e32").write_bytes(pe_bytes())
    prepared = {
        "version": VERSION,
        "compiler": signing.file_identity(compiler),
        "script": signing.file_identity(script),
        "cleanup": signing.file_identity(cleanup),
        "uninstaller_filename": "uninst-generated.e32",
        "payload": signing.tree_identity(payload),
        "targets": {
            name: {
                **signing.file_identity(payload / signing.COMPONENTS[0]),
                "payload_sha256": signing.payload_identity(payload / signing.COMPONENTS[0]),
            }
            for name in signing.COMPONENTS
        },
    }
    signing.write_provenance_file(root / "prepared.json", prepared)
    monkeypatch.setattr(signing, "verify_signature", verified)
    monkeypatch.setattr(
        signing,
        "verify_pe_product_metadata",
        lambda path, expected_version: {
            **metadata_resources(expected_version)[0]["strings"],
            "version_resources": metadata_resources(expected_version),
        },
    )
    consumed = []

    def compile_installer(arguments, **kwargs):
        staged = root / "signed-payload"
        consumed.extend((staged / name).read_bytes() for name in signing.COMPONENTS[:-1])
        assert (staged / "upstream.exe").read_bytes() == (payload / "upstream.exe").read_bytes()
        assert (root / "uninstallers" / "uninst-generated.e32").read_bytes() == pe_bytes(
            signed=True
        )
        manifest = signing.read_record(staged / "native-artifact-manifest.json")
        for entry in manifest["inventory"]:
            assert entry["sha256"] == signing.file_identity(staged / entry["path"])["sha256"]
        (root / "installer-candidate" / signing.installer_filename(VERSION)).write_bytes(pe_bytes())
        return SimpleNamespace(returncode=0, stdout="compiled", stderr="")

    monkeypatch.setattr(signing.subprocess, "run", compile_installer)
    result = signing.rebuild(VERSION, payload, root, compiler, script, cleanup)
    assert consumed == [pe_bytes(signed=True)] * 3
    assert result["version"] == VERSION
    assert (payload / signing.COMPONENTS[0]).read_bytes() == pe_bytes()


def test_installed_uninstaller_and_components_must_reuse_identical_signed_bytes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(signing, "verify_signature", verified)
    for name in signing.COMPONENTS[:-1]:
        (tmp_path / name).write_bytes(pe_bytes(signed=True))
    uninstaller = tmp_path / "unins000.exe"
    uninstaller.write_bytes(pe_bytes(signed=True))
    expected = {name: verified(uninstaller, VERSION) for name in signing.COMPONENTS}
    result = signing.validate_installed(tmp_path, expected, VERSION)
    assert result["uninstaller.exe"]["installed_filename"] == "unins000.exe"
    uninstaller.write_bytes(pe_bytes())
    with pytest.raises(RuntimeError, match="signed-byte mismatch"):
        signing.validate_installed(tmp_path, expected, VERSION)


def test_context_requires_secret_exact_workflow_ref_and_explicit_production_cutover(monkeypatch):
    arguments = (VERSION, f"v{VERSION}", SHA, CONTEXT["ref"], CONTEXT["workflow_ref"], SHA)
    monkeypatch.setenv("SIGNPATH_API_TOKEN", "token-that-must-not-escape")
    monkeypatch.setenv("SIGNPATH_EXPECTED_PUBLISHER", CONTEXT["publisher"])
    monkeypatch.setenv("SIGNPATH_EXPECTED_ROOT_SHA256", CONTEXT["root_sha256"])
    monkeypatch.delenv("SIGNPATH_PRODUCTION_READY", raising=False)
    assert signing.verify_context(*arguments, True)["policy"] == "test-signing"
    with pytest.raises(RuntimeError, match="cutover"):
        signing.verify_context(*arguments, False)
    monkeypatch.setenv("SIGNPATH_PRODUCTION_READY", "true")
    monkeypatch.setenv("SIGNPATH_RELEASE_ORIGIN", f"v{VERSION}")
    assert signing.verify_context(*arguments, False)["production_ready"] is True
    monkeypatch.delenv("SIGNPATH_API_TOKEN")
    with pytest.raises(RuntimeError, match="missing"):
        signing.verify_context(*arguments, True)


def test_finalize_emits_no_success_evidence_after_validation_failure(candidate, tmp_path):
    asset, record = candidate
    evidence, root = tmp_path / "evidence", tmp_path / "root"
    evidence.mkdir()
    (root / "signed-installer").mkdir(parents=True)
    shutil.copyfile(asset, root / "signed-installer" / asset.name)
    signing.write_provenance_file(evidence / "context.json", record["context"])
    for phase, item in record["requests"].items():
        signing.write_provenance_file(evidence / f"{phase}-request.json", item)
    for group, item in record["validation"].items():
        signing.write_provenance_file(root / f"validated-{group}.json", {"targets": item})
    signing.write_provenance_file(
        root / "prepared.json", {"targets": record["submitted"]["components"]}
    )
    signing.write_provenance_file(
        root / "installer-candidate.json", {"targets": record["submitted"]["installer"]}
    )
    signing.write_provenance_file(evidence / "install-result.json", {"uninstall": "failed"})
    with pytest.raises(RuntimeError, match="uninstall"):
        signing.finalize(root, evidence, **KEYWORDS)
    assert not (evidence / "provenance.json").exists()
    signing.write_provenance_file(evidence / "install-result.json", {"uninstall": "passed"})
    completed = signing.finalize(root, evidence, **KEYWORDS)
    assert completed["requests"]["components"]["artifact_id"] == "456"
    assert completed["draft_asset"] == asset.name
    assert signing.read_record(evidence / "provenance.json")["status"] == "validated"


def test_failure_request_identity_is_durable_and_token_never_leaks(tmp_path, monkeypatch):
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    token = "secret-token-do-not-log"
    monkeypatch.setenv("SIGNPATH_API_TOKEN", token)
    monkeypatch.setenv("ARTIFACT_ID", "345")
    monkeypatch.setenv("REQUEST_ID", "malformed-request-id")
    monkeypatch.setenv("REQUEST_URL", f"https://example.invalid/{token}")
    monkeypatch.setenv("REQUEST_RESULT", "failure")
    script = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "record-signing-request.ps1"
    result = subprocess.run(
        [
            executable,
            "-NoProfile",
            "-File",
            str(script),
            "-Phase",
            "components",
            "-OutputDir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    assert result.returncode != 0
    record = (tmp_path / "components-request.json").read_text(encoding="utf-8-sig")
    assert token not in record + result.stdout + result.stderr
    parsed = json.loads(record)
    assert parsed["artifact_id"] == "345"
    assert parsed["request_id"] == "malformed-request-id"
    assert parsed["metadata_status"] == "invalid-request-id"


@pytest.mark.parametrize("name", ["extra.exe", "uninstaller-renamed.exe", None])
def test_prepare_refuses_ambiguous_or_missing_generated_uninstaller(tmp_path, monkeypatch, name):
    payload = tmp_path / "payload"
    payload.mkdir()
    for target in signing.COMPONENTS[:-1]:
        (payload / target).write_bytes(pe_bytes())
    monkeypatch.setattr(
        signing,
        "verify_pe_product_metadata",
        lambda path, expected_version: {
            **metadata_resources(expected_version)[0]["strings"],
            "version_resources": metadata_resources(expected_version),
        },
    )

    def failed_compile(arguments, **kwargs):
        if name:
            (tmp_path / "root" / "uninstallers" / name).write_bytes(pe_bytes())
        return SimpleNamespace(returncode=1, stdout="compiler failure", stderr="")

    monkeypatch.setattr(signing.subprocess, "run", failed_compile)
    with pytest.raises(RuntimeError, match="signing boundary"):
        signing.prepare(
            VERSION,
            payload,
            tmp_path / "root",
            tmp_path / "compiler",
            tmp_path / "script",
            tmp_path / "cleanup",
        )
    assert not (tmp_path / "root" / "prepared.json").exists()


@pytest.mark.parametrize(
    "mutation", [None, "published", "unsigned", "extra", "digest", "size", "origin"]
)
def test_draft_delivery_connects_only_validated_asset_bytes(candidate, mutation):
    asset, provenance = candidate
    digest = signing.file_identity(asset)
    release = {
        "id": 456,
        "draft": True,
        "tag_name": KEYWORDS["tag"],
        "name": KEYWORDS["tag"],
        "html_url": f"https://github.com/owner/repo/releases/tag/{KEYWORDS['tag']}",
        "assets": [
            {
                "id": 987,
                "name": asset.name,
                "size": digest["size"],
                "state": "uploaded",
                "digest": f"sha256:{digest['sha256']}",
                "browser_download_url": f"https://github.com/owner/repo/releases/download/{KEYWORDS['tag']}/{asset.name}",
            }
        ],
    }
    if mutation == "published":
        release["draft"] = False
    elif mutation == "unsigned":
        provenance["policy"] = "test-signing"
    elif mutation == "extra":
        release["assets"].append({"name": "unsigned.exe"})
    elif mutation == "digest":
        release["assets"][0]["digest"] = "sha256:" + "f" * 64
    elif mutation == "size":
        release["assets"][0]["size"] += 1
    elif mutation == "origin":
        release["html_url"] = "https://example.invalid/release"
    if mutation:
        with pytest.raises(RuntimeError):
            signing.verify_draft_delivery(release, provenance)
    else:
        result = signing.verify_draft_delivery(release, provenance)
        assert result["asset_id"] == 987
        assert result["source_sha"] == SHA
        assert result["sha256"] == digest["sha256"]


@pytest.mark.skipif(os.name != "nt", reason="Native build initialization requires Windows")
def test_signing_context_survives_native_build_initialization(monkeypatch):
    import yaml

    powershell = shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell 7 is unavailable")
    repo = Path(__file__).resolve().parents[2]
    workflow = yaml.load(
        (repo / ".github" / "workflows" / "release-native.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    environment = workflow["env"]
    release_root = PureWindowsPath(environment["NATIVE_BUILD_ROOT"]).parent
    scratch = Path("C:/d177/native-integration")
    scratch.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SIGNPATH_API_TOKEN", "local-lifecycle-fixture-not-a-real-token")
    monkeypatch.setenv("SIGNPATH_EXPECTED_PUBLISHER", CONTEXT["publisher"])
    monkeypatch.setenv("SIGNPATH_EXPECTED_ROOT_SHA256", CONTEXT["root_sha256"])
    with tempfile.TemporaryDirectory(prefix="issue225-lifecycle-", dir=scratch) as directory:
        release = Path(directory)
        build = release.joinpath(
            *PureWindowsPath(environment["NATIVE_BUILD_ROOT"]).relative_to(release_root).parts
        )
        evidence = release.joinpath(
            *PureWindowsPath(environment["SIGNING_EVIDENCE"]).relative_to(release_root).parts
        )
        context = signing.verify_context(
            VERSION, KEYWORDS["tag"], SHA, CONTEXT["ref"], CONTEXT["workflow_ref"], SHA, True
        )
        signing.write_provenance_file(evidence / "context.json", context)
        transient = build / "evidence" / "reset-sentinel.json"
        signing.write_provenance_file(transient, {"transient": True})
        inputs = release / "empty-inputs"
        inputs.mkdir()
        archive = inputs / "empty-python-embed.zip"
        archive.touch()
        reset = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-File",
                str(repo / "scripts" / "ci" / "build-native-experimental.ps1"),
                "-ToolPython",
                sys.executable,
                "-PinnedInputRoot",
                str(inputs),
                "-PythonEmbedArchive",
                str(archive),
                "-SoxrBuildEnvironment",
                str(inputs),
                "-BuildRoot",
                str(build),
                "-OutputDir",
                str(release / "dist"),
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            check=False,
        )
        assert reset.returncode != 0
        assert (build / "input-index").is_dir()
        assert not transient.exists()
        record = request("components")
        record["metadata"]["signingPolicySlug"] = "test-signing"
        signing.write_provenance_file(evidence / "components-request.json", record)
        assert (
            signing.main(
                [
                    "validate-request",
                    "--phase",
                    "components",
                    "--evidence",
                    str(evidence),
                    "--version",
                    VERSION,
                    "--repository",
                    "owner/repo",
                    "--source-sha",
                    SHA,
                    "--run-id",
                    KEYWORDS["run_id"],
                    "--run-attempt",
                    KEYWORDS["run_attempt"],
                ]
            )
            == 0
        )
        assert signing.read_record(evidence / "context.json") == context


@pytest.mark.parametrize("phase", ["components", "installer"])
@pytest.mark.parametrize("parameters", [{"version": VERSION}, {"version": "9.9.9"}, {}, None])
def test_request_version_matches_service_metadata_contract(phase, parameters):
    item = request(phase)
    item["metadata"]["parameters"] = parameters
    arguments = {
        "version": VERSION,
        "repository": "owner/repo",
        "source_sha": SHA,
        "run_id": "123",
        "run_attempt": "2",
    }
    if parameters == {"version": VERSION}:
        assert signing.validate_request(item, phase, CONTEXT, **arguments)["commit_id"] == SHA
    else:
        with pytest.raises(RuntimeError, match="parameters"):
            signing.validate_request(item, phase, CONTEXT, **arguments)


@pytest.mark.skipif(os.name != "nt", reason="Native smoke isolation requires Windows")
@pytest.mark.parametrize("fail_after_version", [False, True])
def test_native_smoke_owns_child_appdata_and_cleans_it_after_success_or_failure(
    tmp_path, fail_after_version
):
    from puripuly_heart import __version__

    powershell = shutil.which("pwsh")
    compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    if powershell is None or not compiler.is_file():
        pytest.skip("PowerShell 7 and the Windows .NET compiler are required")
    launcher_source = tmp_path / "launcher.cs"
    launcher = tmp_path / "launcher.exe"
    launcher_source.write_text(
        """
using System;
using System.Diagnostics;
class Launcher {
    static int Main(string[] args) {
        if (args.Length != 2 || args[0] != "--headless") return 99;
        Console.Error.WriteLine("child_localappdata=" + Environment.GetEnvironmentVariable("LOCALAPPDATA"));
        Console.Error.WriteLine("child_appdata=" + Environment.GetEnvironmentVariable("APPDATA"));
        var start = new ProcessStartInfo();
        start.FileName = Environment.GetEnvironmentVariable("PROBE_PYTHON");
        start.Arguments = "-m puripuly_heart.main " + args[1];
        start.UseShellExecute = false;
        using (var child = Process.Start(start)) {
            child.WaitForExit();
            return child.ExitCode;
        }
    }
}
""",
        encoding="utf-8",
    )
    compiled = subprocess.run(
        [str(compiler), "/nologo", f"/out:{launcher}", str(launcher_source)],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    local, roaming = tmp_path / "ambient-local", tmp_path / "ambient-roaming"
    for directory in (local, roaming):
        directory.mkdir()
        (directory / "sentinel").write_text("unrelated user state", encoding="utf-8")
    smoke = tmp_path / "smoke"
    repo = Path(__file__).resolve().parents[2]
    script = r"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($env:PROBE_SCRIPT, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$function = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Invoke-Headless' }, $true)
. ([scriptblock]::Create($function.Extent.Text))
$mainTry = @($ast.EndBlock.Statements | Where-Object { $_ -is [System.Management.Automation.Language.TryStatementAst] })[0]
$hostExe = $env:PROBE_HOST
$smokeRoot = $env:PROBE_SMOKE
$previousPythonPath = $env:PYTHONPATH
$previousDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$ambientLocal = $env:LOCALAPPDATA
$ambientRoaming = $env:APPDATA
$sandboxCreated = $false
$failure = $null
Push-Location $env:PROBE_REPO
try {
    $setup = $mainTry.Body.Statements | Where-Object { $_ -is [System.Management.Automation.Language.PipelineAst] -and $_.Extent.Text.StartsWith('New-Item ') -and $_.Extent.Text.Contains('$smokeRoot') }
    foreach ($statement in $setup) { . ([scriptblock]::Create($statement.Extent.Text)) }
    $version = Invoke-Headless -ArgumentList @('--version')
    $sandboxCreated = Test-Path -LiteralPath (Join-Path $smokeRoot 'LocalAppData/puripuly-heart')
    if ($env:PROBE_FAIL -eq 'true') { [void](Invoke-Headless -ArgumentList @('not-a-command')) }
} catch {
    $failure = $_
} finally {
    foreach ($statement in $mainTry.Finally.Statements) { . ([scriptblock]::Create($statement.Extent.Text)) }
}
@{
    version = $version
    sandbox_created = $sandboxCreated
    sandbox_removed = -not (Test-Path -LiteralPath $smokeRoot)
    ambient_local_unchanged = $env:LOCALAPPDATA -eq $ambientLocal
    ambient_roaming_unchanged = $env:APPDATA -eq $ambientRoaming
    pythonpath_restored = $env:PYTHONPATH -eq $previousPythonPath
    bytecode_restored = $env:PYTHONDONTWRITEBYTECODE -eq $previousDontWriteBytecode
    child_failed = $null -ne $failure
} | ConvertTo-Json
if ($null -ne $failure) { throw $failure }
"""
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        env={
            **os.environ,
            "PYTHONPATH": str(repo / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "LOCALAPPDATA": str(local),
            "APPDATA": str(roaming),
            "PROBE_PYTHON": sys.executable,
            "PROBE_SCRIPT": str(repo / "scripts/ci/build-native-release-artifacts.ps1"),
            "PROBE_HOST": str(launcher),
            "PROBE_SMOKE": str(smoke),
            "PROBE_REPO": str(repo),
            "PROBE_FAIL": str(fail_after_version).lower(),
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    assert (result.returncode != 0) is fail_after_version, result.stdout + result.stderr
    observed = json.loads(result.stdout[result.stdout.index("\n{") + 1 :])
    assert observed["version"] == __version__
    assert observed["child_failed"] is fail_after_version
    assert all(
        observed[key]
        for key in (
            "sandbox_created",
            "sandbox_removed",
            "ambient_local_unchanged",
            "ambient_roaming_unchanged",
            "pythonpath_restored",
            "bytecode_restored",
        )
    )
    assert f"child_localappdata={smoke / 'LocalAppData'}" in result.stdout
    assert f"child_appdata={smoke / 'RoamingAppData'}" in result.stdout
    for directory in (local, roaming):
        assert (directory / "sentinel").read_text(encoding="utf-8") == "unrelated user state"
        assert not (directory / "puripuly-heart").exists()
