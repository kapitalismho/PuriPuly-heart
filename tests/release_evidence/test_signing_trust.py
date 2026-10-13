from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows certificate stores are required")

ROOT_SNAPSHOT = """
$ErrorActionPreference = 'Stop'
$result = @{}
foreach ($location in @('CurrentUser', 'LocalMachine')) {
    $store = [System.Security.Cryptography.X509Certificates.X509Store]::new('Root', $location)
    try {
        $store.Open([System.Security.Cryptography.X509Certificates.OpenFlags]::ReadOnly -bor [System.Security.Cryptography.X509Certificates.OpenFlags]::OpenExistingOnly)
        $fingerprints = @($store.Certificates | ForEach-Object {
            [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($_.RawData))
        } | Sort-Object)
        $bytes = [System.Text.Encoding]::UTF8.GetBytes(($fingerprints -join ','))
        $result[$location] = @{
            count = $fingerprints.Count
            sha256 = [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($bytes))
        }
    } finally { $store.Dispose() }
}
$result | ConvertTo-Json -Depth 3
"""


@pytest.fixture(scope="module")
def public_authority() -> dict[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Trust boundary fixture")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    encoded = certificate.public_bytes(serialization.Encoding.DER)
    return {
        "SIGNPATH_EXPECTED_PUBLISHER": "CN=Trust boundary fixture",
        "SIGNPATH_EXPECTED_ROOT_SHA256": hashlib.sha256(encoded).hexdigest(),
        "SIGNPATH_TEST_ROOT_CERTIFICATE_BASE64": base64.b64encode(encoded).decode("ascii"),
    }


@pytest.fixture(scope="module")
def powershell() -> str:
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell 7 is unavailable")
    return executable


def snapshot_roots(powershell: str) -> dict:
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", ROOT_SNAPSHOT],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=20,
        check=True,
    )
    return json.loads(result.stdout)


def invoke_trust(powershell: str, authority: dict[str, str], *, test: bool = True):
    environment = dict(os.environ, **authority)
    environment.pop("RUNNER_ENVIRONMENT", None)
    environment.pop("GITHUB_ACTIONS", None)
    script = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "prepare-signing-trust.ps1"
    arguments = [powershell, "-NoProfile", "-NonInteractive", "-File", str(script)]
    if test:
        arguments.append("-TestSigning")
    return subprocess.run(
        arguments,
        env=environment,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize(
    "mutation,expected_error",
    [
        ("malformed-pin", "Independently verified publisher/root variables are required"),
        ("mismatched-pin", "Test trust anchor differs from the pinned SHA256"),
        ("missing-certificate", "independently obtained, pinned test root certificate"),
        ("unsupported-local", "isolated GitHub-hosted Windows runner"),
    ],
)
def test_trust_cli_rejects_invalid_or_local_import_without_changing_roots(
    powershell, public_authority, mutation, expected_error
):
    authority = dict(public_authority)
    if mutation == "malformed-pin":
        authority["SIGNPATH_EXPECTED_ROOT_SHA256"] = "not-a-certificate-pin"
    elif mutation == "mismatched-pin":
        authority["SIGNPATH_EXPECTED_ROOT_SHA256"] = "0" * 64
    elif mutation == "missing-certificate":
        authority["SIGNPATH_TEST_ROOT_CERTIFICATE_BASE64"] = ""
    before = snapshot_roots(powershell)
    result = invoke_trust(powershell, authority)
    after = snapshot_roots(powershell)
    assert result.returncode != 0
    assert expected_error in result.stderr
    assert "importing pinned authority" not in result.stdout
    assert after == before


def test_production_trust_preflight_still_performs_no_import(powershell, public_authority):
    before = snapshot_roots(powershell)
    result = invoke_trust(powershell, public_authority, test=False)
    after = snapshot_roots(powershell)
    assert result.returncode == 0
    assert result.stdout.strip() == ""
    assert after == before
