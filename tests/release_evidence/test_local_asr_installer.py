from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tomllib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from puripuly_heart.config.settings_vnext.schema import VNEXT_SETTINGS_SCHEMA_VERSION
from puripuly_heart.core.local_asr.local_stt_assets import (
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    LocalSTTAssetSource,
    load_local_stt_asset_manifest,
    validate_local_stt_install,
)
from puripuly_heart.core.local_asr.local_stt_catalog import inspect_required_cpu_model_installs
from puripuly_heart.core.local_asr.local_stt_runtime_installer import LocalSTTProvisioningLease
from puripuly_heart.release_evidence.local_asr_installer import render_model_data
from puripuly_heart.release_evidence.native_installer_cleanup import (
    LEGACY_SCHEMA,
    NATIVE_SCHEMA,
    generate_include,
)

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="Executable Windows Setup behavior")


@pytest.fixture()
def compiler():
    configured = os.environ.get("PURIPULY_HEART_TEST_ISCC")
    candidates = (
        [Path(configured)]
        if configured
        else [
            Path(os.environ.get(name, "C:/Program Files")) / "Inno Setup 7/ISCC.exe"
            for name in ("ProgramFiles", "ProgramFiles(x86)")
        ]
    )
    executable = next((path.resolve() for path in candidates if path.is_file()), None)
    if executable is None:
        pytest.skip("Inno Setup >=7.1 compiler is required; set PURIPULY_HEART_TEST_ISCC")
    version = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=True
    )
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version.stdout)
    assert match and tuple(map(int, match.groups())) >= (7, 1, 0), version.stdout
    return executable


def _compile(tmp_path, compiler, manifests=None, language="english", defines=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    include = tmp_path / "models.iss"
    include.write_text(render_model_data(manifests), encoding="utf-8")
    package = tmp_path / "package"
    if manifests is None:
        destination = package / "app/puripuly_heart/data/models"
        destination.mkdir(parents=True, exist_ok=True)
        for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS:
            name = f"{model_id}.manifest.json"
            shutil.copyfile(ROOT / "src/puripuly_heart/data/models" / name, destination / name)
    data_root = tmp_path / "profile"
    data_root.mkdir(exist_ok=True)
    report = tmp_path / "report.json"
    identity = json.loads(
        (ROOT / "tests/config/public_compatibility_surfaces_snapshot.json").read_text()
    )["installer_identity"]["defines"]
    identity_json = ",".join('"' + name + '":"{#' + name + '}"' for name in identity)
    script = tmp_path / "probe.iss"
    body = r"""
#define MyPackagedAppDir "@PACKAGE@"
#define LocalASRModelsInclude "@MODELS@"
#define InstallerSmokeAppDataRoot "@PROFILE@"
#include "@IDENTITY@"
[Setup]
AppId=Issue317IsolatedAsrProbe
AppName=Issue317 isolated ASR probe
AppVersion=1.0
DefaultDirName={tmp}\unused-probe-app
CreateAppDir=no
Uninstallable=no
CreateUninstallRegKey=no
UsePreviousAppDir=no
UsePreviousGroup=no
UsePreviousTasks=no
UsePreviousLanguage=no
PrivilegesRequired=lowest
DisableWelcomePage=yes
DisableReadyPage=yes
DisableFinishedPage=yes
OutputDir=@OUTPUT@
OutputBaseFilename=probe
[Languages]
Name: "@LANGUAGE@"; MessagesFile: "compiler:Default.isl"
[Tasks]
Name: "redownloadasr"; Description: "Replace known ASR assets"; Flags: unchecked
[CustomMessages]
LocalSttDownloadFailed=ASR provisioning failed
LocalSttDiskSpaceCheckFailed=Storage check failed: %1
LocalSttDiskSpaceFailed=Storage insufficient: %1 %2 %3
[Code]
function ResolveInstallerSmokeAppDataRoot(Param: String): String;
begin
  Result := '@PROFILE@';
end;
function ResolveLocalSttAppDataRoot(): String;
begin
  Result := '@PROFILE@';
end;
#include "@ASR@"
#include "@TELEMETRY@"
function JsonString(Value: String): String;
begin
  StringChangeEx(Value, '\', '\\', True);
  StringChangeEx(Value, '"', '\"', True);
  StringChangeEx(Value, #13, '\r', True);
  StringChangeEx(Value, #10, '\n', True);
  Result := '"' + Value + '"';
end;
function JsonBoolean(Value: Boolean): String;
begin
  if Value then Result := 'true' else Result := 'false';
end;
procedure InitializeWizard();
begin
  DownloadPage := CreateDownloadPage('ASR probe', '', @DownloadLocalSttProgress);
end;
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  OperationError, Document, Model, Files: String;
  ModelIndex, FileIndex: Integer;
  Exists, Enabled, TelemetryReadable: Boolean;
begin
  OperationError := '';
  if ExpandConstant('{param:provision|0}') = '1' then
    OperationError := RunSetupLocalSttProvisioning();
  TelemetryReadable := ProbeExistingTelemetryPreference(Exists, Enabled);
  Document := '{"success":' + JsonBoolean(OperationError = '') +
    ',"telemetry_readable":' + JsonBoolean(TelemetryReadable) +
    ',"telemetry_exists":' + JsonBoolean(Exists) +
    ',"telemetry_enabled":' + JsonBoolean(Enabled) +
    ',"schema_version":{#CanonicalSettingsVersion},"identity":{@IDENTITYJSON@},"models":[';
  for ModelIndex := 0 to AsrModelCount() - 1 do begin
    if ModelIndex > 0 then Document := Document + ',';
    Model := '{"model_id":' + JsonString(AsrModelId(ModelIndex)) +
      ',"install_dirname":' + JsonString(AsrInstallDirname(ModelIndex)) +
      ',"size_bytes":' + IntToStr(AsrModelSize(ModelIndex)) +
      ',"installed_manifest":' + JsonString(AsrInstalledManifest(ModelIndex, 'hf_mirror')) + ',"files":[';
    Files := '';
    for FileIndex := 0 to AsrFileCount(ModelIndex) - 1 do begin
      if FileIndex > 0 then Files := Files + ',';
      Files := Files + '{"path":' + JsonString(AsrFilePath(ModelIndex, FileIndex)) +
        ',"sha256":' + JsonString(AsrFileSha256(ModelIndex, FileIndex)) +
        ',"size_bytes":' + IntToStr(AsrFileSize(ModelIndex, FileIndex)) +
        ',"primary":' + JsonString(AsrPrimaryUrl(ModelIndex, FileIndex)) +
        ',"mirror":' + JsonString(AsrMirrorUrl(ModelIndex, FileIndex)) + '}';
    end;
    Document := Document + Model + Files + ']}';
  end;
  Document := Document + ']}';
  if not SaveStringToFile('@REPORT@', Document, False) then RaiseException('Cannot write probe receipt');
  Result := '';
end;
"""
    replacements = {
        "PACKAGE": package,
        "MODELS": include,
        "PROFILE": data_root,
        "IDENTITY": ROOT / "installer/application-identity.iss",
        "OUTPUT": tmp_path / "output",
        "LANGUAGE": language,
        "ASR": ROOT / "installer/local-asr.iss",
        "TELEMETRY": ROOT / "installer/telemetry-preference.iss",
        "IDENTITYJSON": identity_json,
        "REPORT": report,
    }
    for key, value in replacements.items():
        body = body.replace(f"@{key}@", str(value))
    script.write_text(body, encoding="utf-8-sig")
    command = [str(compiler), "--no-ide-signtools", *defines, str(script)]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return {
        "executable": tmp_path / "output/probe.exe",
        "profile": data_root,
        "report": report,
        "script": script,
        "package": package,
        "command": command,
    }


def _run(probe, *arguments):
    probe["report"].unlink(missing_ok=True)
    environment = os.environ.copy()
    temporary = probe["profile"] / "temp"
    temporary.mkdir(exist_ok=True)
    environment.update(TEMP=str(temporary), TMP=str(temporary))
    executed = subprocess.run(
        [
            str(probe["executable"]),
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            f'/LOG={probe["profile"] / "setup.log"}',
            *arguments,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr
    return json.loads(probe["report"].read_text())


@WINDOWS
def test_compiled_identity_and_current_schema_telemetry_reader(tmp_path, compiler):
    probe = _compile(tmp_path, compiler)
    settings = probe["profile"] / "settings.json"
    for version, enabled, readable in (
        (VNEXT_SETTINGS_SCHEMA_VERSION, False, True),
        (VNEXT_SETTINGS_SCHEMA_VERSION, True, True),
        (VNEXT_SETTINGS_SCHEMA_VERSION + 1, False, False),
    ):
        settings.write_text(
            json.dumps(
                {
                    "settings_version": version,
                    "intent": {"telemetry": {"enabled": enabled}},
                    "state": {},
                }
            )
        )
        result = _run(probe)
        assert result["telemetry_readable"] is readable
        assert result["telemetry_exists"] is True
        if readable:
            assert result["telemetry_enabled"] is enabled
        assert result["schema_version"] == VNEXT_SETTINGS_SCHEMA_VERSION
    expected_identity = json.loads(
        (ROOT / "tests/config/public_compatibility_surfaces_snapshot.json").read_text()
    )["installer_identity"]["defines"]
    assert result["identity"] == expected_identity
    assert [model["model_id"] for model in result["models"]] == list(
        REQUIRED_CPU_LOCAL_STT_MODEL_IDS
    )
    for actual in result["models"]:
        manifest = load_local_stt_asset_manifest(actual["model_id"])
        assert actual["size_bytes"] == sum(asset.size_bytes for asset in manifest.files)
        assert actual["install_dirname"] == manifest.install_dirname
        installed = json.loads(actual["installed_manifest"])
        assert installed["model_id"] == manifest.model_id
        assert installed["engine"] == manifest.engine
        assert installed["selected_source"] == "hf_mirror"
        for observed, asset in zip(actual["files"], manifest.files, strict=True):
            assert observed["path"].replace("\\", "/") == asset.relative_path
            assert observed["size_bytes"] == asset.size_bytes
            assert observed["sha256"] == asset.sha256
            assert observed["primary"].startswith("https://huggingface.co/")
            assert observed["mirror"].startswith("https://hf-mirror.net/")
    first = REQUIRED_CPU_LOCAL_STT_MODEL_IDS[0]
    packaged_manifest = probe["package"] / f"app/puripuly_heart/data/models/{first}.manifest.json"
    packaged_manifest.write_bytes(packaged_manifest.read_bytes() + b"\n")
    stale = subprocess.run(probe["command"], capture_output=True, text=True, timeout=60)
    assert stale.returncode != 0


@WINDOWS
def test_compiler_rejects_alternate_app_identity_without_isolated_data_root(tmp_path, compiler):
    identity = "/DMyAppId={{12345678-1234-1234-1234-123456789ABC}"
    with pytest.raises(AssertionError):
        _compile(tmp_path / "unisolated", compiler, defines=(identity,))
    isolated = _compile(
        tmp_path / "isolated",
        compiler,
        defines=(identity, "/DMyAppDataDirName=issue317-isolated-installer"),
    )
    observed = _run(isolated)["identity"]
    assert observed["MyAppId"] == "{{12345678-1234-1234-1234-123456789ABC}"
    assert observed["MyAppDataDirName"] == "issue317-isolated-installer"


@contextmanager
def _sources(directory):
    state = {"primary": "good", "mirror": "good", "requests": []}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, *_args):
            pass

        def do_GET(self):
            source = self.path.strip("/").split("/", 1)[0]
            state["requests"].append((source, self.path))
            mode = state.get(source, "good")
            if mode == "missing":
                self.send_error(404)
            elif mode == "corrupt":
                path = Path(self.translate_path(self.path))
                payload = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(b"x" * len(payload))
            else:
                super().do_GET()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _synthetic_manifests(directory, url):
    manifests = []
    for index, model_id in enumerate(REQUIRED_CPU_LOCAL_STT_MODEL_IDS):
        base = load_local_stt_asset_manifest(model_id)
        files = []
        for file_index, asset in enumerate(base.files):
            payload = f"owned-model-{index}-file-{file_index}".encode() * 8
            for source in ("primary", "mirror"):
                target = directory / source / model_id / asset.relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            files.append(
                replace(asset, size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            )
        manifests.append(
            replace(
                base,
                files=tuple(files),
                sources={
                    "huggingface": LocalSTTAssetSource(
                        "huggingface",
                        "a" * 40,
                        download_url_template=f"{url}/primary/{model_id}/{{path}}",
                    ),
                    "hf_mirror": LocalSTTAssetSource(
                        "hf_mirror",
                        "b" * 40,
                        download_url_template=f"{url}/mirror/{model_id}/{{path}}",
                    ),
                },
            )
        )
    return tuple(manifests)


@WINDOWS
def test_real_skip_receipt_satisfies_existing_lifecycle_consumer_without_downloading(
    tmp_path, compiler
):
    consumer = tmp_path / "consume-lifecycle-receipt.ps1"
    consumer.write_text(
        r"""
param([string]$BuilderPath, [string]$LogPath)
$ErrorActionPreference = 'Stop'
$tokens = $null
$parseErrors = $null
$tree = [System.Management.Automation.Language.Parser]::ParseFile(
    $BuilderPath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw 'Cannot parse lifecycle consumer.' }
$function = $tree.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Invoke-InstallerLifecycle'
}, $true)
if ($null -eq $function) { throw 'Cannot find lifecycle consumer.' }
$guards = @($function.Body.FindAll({
    param($node)
    if ($node -isnot [System.Management.Automation.Language.IfStatementAst]) {
        return $false
    }
    return $null -ne $node.Clauses[0].Item1.Find({
        param($expression)
        $expression -is [System.Management.Automation.Language.VariableExpressionAst] -and
            $expression.VariablePath.UserPath -eq 'logText'
    }, $true)
}, $true))
if ($guards.Count -ne 1) { throw 'Cannot isolate the lifecycle receipt guard.' }
$logText = Get-Content -LiteralPath $LogPath -Raw
& ([ScriptBlock]::Create($guards[0].Extent.Text))
""",
        encoding="utf-8-sig",
    )

    def consume(log):
        return subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(consumer),
                str(ROOT / "scripts/ci/build-native-release-artifacts.ps1"),
                str(log),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    with _sources(tmp_path / "sources") as (url, state):
        manifests = _synthetic_manifests(tmp_path / "sources", url)
        skipped = _compile(
            tmp_path / "skipped",
            compiler,
            manifests,
            defines=("/DSkipLocalSttProvisioning=1",),
        )
        assert _run(skipped, "/provision=1")["success"] is True
        assert state["requests"] == []
        assert not (skipped["profile"] / "models").exists()
        skipped_log = skipped["profile"] / "setup.log"
        accepted = consume(skipped_log)
        assert accepted.returncode == 0, accepted.stdout + accepted.stderr

        normal = _compile(tmp_path / "normal", compiler, manifests)
        assert _run(normal, "/provision=1")["success"] is True
        assert state["requests"]
        assert inspect_required_cpu_model_installs(
            normal["profile"] / "models",
            manifests={manifest.model_id: manifest for manifest in manifests},
        ).cpu_auto_available
        rejected = consume(normal["profile"] / "setup.log")
        assert rejected.returncode != 0
        assert "controlled no-network provisioning boundary" in rejected.stderr

        for outcome in ("failed", "completed successfully"):
            contradicted_log = tmp_path / f"contradicted-{outcome}.log"
            contradicted_log.write_bytes(
                skipped_log.read_bytes() + f"\nLocal STT provisioning {outcome}\n".encode("utf-8")
            )
            contradicted = consume(contradicted_log)
            assert contradicted.returncode != 0
            assert "controlled no-network provisioning boundary" in contradicted.stderr


@WINDOWS
@pytest.mark.parametrize("language", ["english", "chinesesimplified"])
def test_real_setup_downloads_verifies_promotes_and_preserves_valid_install_on_hash_failure(
    tmp_path, compiler, language
):
    source = tmp_path / "sources"
    with _sources(source) as (url, state):
        manifests = _synthetic_manifests(source, url)
        probe = _compile(tmp_path / "setup", compiler, manifests, language)
        state["primary"] = "missing"
        first = _run(probe, "/provision=1")
        assert first["success"] is True
        root = probe["profile"] / "models"
        by_id = {manifest.model_id: manifest for manifest in manifests}
        ready = inspect_required_cpu_model_installs(root, manifests=by_id)
        assert ready.cpu_auto_available
        originals = {}
        for manifest in manifests:
            model = root / manifest.install_dirname
            assert (
                validate_local_stt_install(model, manifest=manifest).selected_source == "hf_mirror"
            )
            for path in model.rglob("*"):
                if path.is_file():
                    originals[path] = path.read_bytes()
        if language == "chinesesimplified":
            assert {source for source, _ in state["requests"]} == {"mirror"}
        else:
            assert {source for source, _ in state["requests"]} == {"primary", "mirror"}
        before_reuse = len(state["requests"])
        assert _run(probe, "/provision=1")["success"] is True
        assert len(state["requests"]) == before_reuse
        state["primary"] = state["mirror"] = "corrupt"
        rejected = _run(probe, "/provision=1", "/TASKS=redownloadasr")
        assert rejected["success"] is False
        assert all(path.read_bytes() == payload for path, payload in originals.items())
        assert inspect_required_cpu_model_installs(root, manifests=by_id).cpu_auto_available
        assert not list(root.glob("*.backup"))
        assert not list(root.glob("*.staging-*"))


@WINDOWS
def test_real_setup_shares_python_provisioning_lease_and_recovers_after_release(tmp_path, compiler):
    with _sources(tmp_path / "sources") as (url, state):
        manifests = _synthetic_manifests(tmp_path / "sources", url)
        probe = _compile(tmp_path / "setup", compiler, manifests)
        root = probe["profile"] / "models"
        lease = LocalSTTProvisioningLease.acquire(model_root=root, wait=False)
        assert lease is not None
        try:
            blocked = _run(probe, "/provision=1")
            assert blocked["success"] is False
            assert state["requests"] == []
            assert all(
                not (root / model_id).exists() for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS
            )
        finally:
            lease.close()
        assert _run(probe, "/provision=1")["success"] is True
        assert inspect_required_cpu_model_installs(
            root, manifests={m.model_id: m for m in manifests}
        ).cpu_auto_available


@WINDOWS
def test_real_setup_replacement_preserves_unknown_files_inside_known_model_directory(
    tmp_path, compiler
):
    with _sources(tmp_path / "sources") as (url, state):
        manifests = _synthetic_manifests(tmp_path / "sources", url)
        probe = _compile(tmp_path / "setup", compiler, manifests)
        assert _run(probe, "/provision=1")["success"] is True
        model_dir = probe["profile"] / "models" / REQUIRED_CPU_LOCAL_STT_MODEL_IDS[0]
        sentinel = model_dir / "user-file"
        sentinel.write_bytes(b"unknown-neighbor")
        before = {path: path.read_bytes() for path in model_dir.rglob("*") if path.is_file()}
        replaced = _run(probe, "/provision=1", "/TASKS=redownloadasr")
        assert replaced["success"] is False
        assert all(path.read_bytes() == data for path, data in before.items())
        assert sentinel.read_bytes() == b"unknown-neighbor"


@WINDOWS
def test_full_setup_compiles_current_model_include_and_emits_canonical_product_metadata(
    tmp_path, compiler
):
    from puripuly_heart.release_evidence.release_identity import verify_pe_product_metadata

    probe = _compile(tmp_path / "inputs", compiler)
    inventory = []
    for name in (
        "PuriPulyHeart.exe",
        "puripuly.exe",
        "PuriPulyHeartOverlay.exe",
        "PuriPulyHeartASRWorker.exe",
    ):
        payload = f"nonexecuted packaging fixture {name}".encode()
        (probe["package"] / name).write_bytes(payload)
        inventory.append(
            {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    native = probe["package"] / "native-artifact-manifest.json"
    native.write_text(json.dumps({"schema": NATIVE_SCHEMA, "inventory": inventory}))
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "schema": LEGACY_SCHEMA,
                "inventory": [
                    {"path": "old.dll", "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
                ],
            }
        )
    )
    cleanup = tmp_path / "native-cleanup.iss"
    generate_include(legacy, native, cleanup)
    output = tmp_path / "full-setup"
    compiled = subprocess.run(
        [
            str(compiler),
            "--no-ide-signtools",
            f'/DMyPackagedAppDir={probe["package"]}',
            f"/DNativeCleanupInclude={cleanup}",
            f'/DLocalASRModelsInclude={probe["script"].parent / "models.iss"}',
            f"/O{output}",
            "/Fcompiled-full-setup",
            str(ROOT / "installer.iss"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    identity = json.loads(
        (ROOT / "tests/config/public_compatibility_surfaces_snapshot.json").read_text()
    )["installer_identity"]["defines"]
    expected = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    values = verify_pe_product_metadata(
        output / "compiled-full-setup.exe",
        expected_version=expected,
        expected_product=identity["MyAppName"],
    )
    assert values["CompanyName"].strip() == identity["MyAppPublisher"]
