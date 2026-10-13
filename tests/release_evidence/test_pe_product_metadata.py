from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from puripuly_heart.release_evidence import release_identity as identity
from puripuly_heart.release_evidence import signing_release as signing

VERSION = "2.8.3"
PRODUCT = "PuriPuly <3"


def _version_node(key, value=b"", *, units=0, kind=1, children=()):
    body = bytearray(struct.pack("<HHH", 0, units, kind) + key.encode("utf-16le") + b"\0\0")
    body.extend(b"\0" * (-len(body) % 4))
    body.extend(value)
    body.extend(b"\0" * (-len(body) % 4))
    for child in children:
        body.extend(child)
        body.extend(b"\0" * (-len(body) % 4))
    struct.pack_into("<H", body, 0, len(body))
    return bytes(body)


def _version_info(tables, binary=(2, 8, 3, 0)):
    strings = []
    translations = []
    for translation, product, version in tables:
        values = []
        for key, value in (
            ("ProductName", product),
            ("ProductVersion", version),
            ("FileDescription", "Metadata fixture"),
        ):
            encoded = value.encode("utf-16le") + b"\0\0"
            values.append(_version_node(key, encoded, units=len(encoded) // 2))
        strings.append(_version_node(translation, children=values))
        translations.append(struct.pack("<HH", int(translation[:4], 16), int(translation[4:], 16)))
    fixed = struct.pack(
        "<13I",
        0xFEEF04BD,
        0x10000,
        0x00070001,
        0,
        (binary[0] << 16) | binary[1],
        (binary[2] << 16) | binary[3],
        0x3F,
        0,
        0x40004,
        1,
        0,
        0,
        0,
    )
    translated = b"".join(translations)
    return _version_node(
        "VS_VERSION_INFO",
        fixed,
        units=len(fixed),
        kind=0,
        children=(
            _version_node("StringFileInfo", children=strings),
            _version_node(
                "VarFileInfo",
                children=(_version_node("Translation", translated, units=len(translated), kind=0),),
            ),
        ),
    )


def write_versioned_pe(path: Path, resources):
    count = len(resources)
    data_offset = 64 + count * 8
    blobs_offset = data_offset + count * 16
    resource = bytearray(blobs_offset)
    for offset, children in ((0, 1), (24, 1), (48, count)):
        struct.pack_into("<IIHHHH", resource, offset, 0, 0, 0, 0, 0, children)
    struct.pack_into("<II", resource, 16, 16, 0x80000000 | 24)
    struct.pack_into("<II", resource, 40, 1, 0x80000000 | 48)
    for index, (language, tables, binary) in enumerate(resources):
        blob = _version_info(tables, binary)
        start = len(resource)
        resource.extend(blob)
        resource.extend(b"\0" * (-len(resource) % 4))
        struct.pack_into("<II", resource, 64 + index * 8, language, data_offset + index * 16)
        struct.pack_into(
            "<IIII", resource, data_offset + index * 16, 0x1000 + start, len(blob), 0, 0
        )
    header = bytearray(512)
    header[:2] = b"MZ"
    struct.pack_into("<I", header, 60, 128)
    header[128:132] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", header, 132, 0x8664, 1, 0, 0, 0, 240, 0x22)
    optional = 152
    struct.pack_into("<H", header, optional, 0x20B)
    struct.pack_into("<Q", header, optional + 24, 0x140000000)
    struct.pack_into("<II", header, optional + 32, 0x1000, 0x200)
    struct.pack_into("<II", header, optional + 56, 0x2000, 0x200)
    struct.pack_into("<I", header, optional + 108, 16)
    struct.pack_into("<II", header, optional + 128, 0x1000, len(resource))
    raw_size = (len(resource) + 511) & ~511
    struct.pack_into(
        "<8sIIIIIIHHI",
        header,
        392,
        b".rsrc\0\0\0",
        len(resource),
        0x1000,
        raw_size,
        512,
        0,
        0,
        0,
        0,
        0x40000040,
    )
    path.write_bytes(header + resource + b"\0" * (raw_size - len(resource)))
    return path


def test_exact_metadata_preserves_every_resource_language_and_table(tmp_path):
    executable = write_versioned_pe(
        tmp_path / "target.exe",
        [
            (0, [("000004b0", PRODUCT, VERSION)], (2, 8, 3, 0)),
            (1033, [("040904b0", PRODUCT, VERSION), ("041204b0", PRODUCT, VERSION)], (2, 8, 3, 0)),
        ],
    )
    metadata = identity.verify_pe_product_metadata(executable, expected_version=VERSION)
    record = signing.unsigned_record(executable, VERSION)
    assert metadata["ProductName"] == PRODUCT
    assert [(row["language"], row["translation"]) for row in record["metadata_resources"]] == [
        (0, "000004b0"),
        (1033, "040904b0"),
        (1033, "041204b0"),
    ]
    assert all(row["strings"]["ProductName"] == PRODUCT for row in record["metadata_resources"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", PRODUCT + " " * 49),
        ("name", "PuriPuly \u2764\ufe0f "),
        ("version", VERSION + " " * 45),
        ("version", "9.9.9"),
    ],
)
def test_preparation_rejects_raw_mismatch_instead_of_stripping_or_flattening(
    tmp_path, field, value
):
    name = value if field == "name" else PRODUCT
    version = value if field == "version" else VERSION
    executable = write_versioned_pe(
        tmp_path / "target.exe",
        [
            (0, [("000004b0", name, version)], (2, 8, 3, 0)),
            (1033, [("040904b0", PRODUCT, VERSION)], (2, 8, 3, 0)),
        ],
    )
    records = identity.read_pe_product_metadata(executable)
    assert records[0]["strings"]["ProductName"] == name
    assert records[0]["strings"]["ProductVersion"] == version
    assert records[-1]["strings"]["ProductName"] == PRODUCT
    with pytest.raises(RuntimeError, match="Product(Name|Version) mismatch"):
        signing.unsigned_record(executable, VERSION)


def test_bad_first_translation_is_not_hidden_by_valid_last_translation(tmp_path):
    executable = write_versioned_pe(
        tmp_path / "target.exe",
        [
            (
                0,
                [("000004b0", "Wrong product", VERSION), ("040904b0", PRODUCT, VERSION)],
                (2, 8, 3, 0),
            ),
        ],
    )
    with pytest.raises(RuntimeError, match="ProductName mismatch"):
        identity.verify_pe_product_metadata(executable, expected_version=VERSION)


def test_binary_version_mismatch_in_any_resource_blocks_submission(tmp_path):
    executable = write_versioned_pe(
        tmp_path / "target.exe",
        [
            (0, [("000004b0", PRODUCT, VERSION)], (9, 9, 9, 0)),
            (1033, [("040904b0", PRODUCT, VERSION)], (2, 8, 3, 0)),
        ],
    )
    with pytest.raises(RuntimeError, match="binary ProductVersion mismatch"):
        signing.unsigned_record(executable, VERSION)


def test_exact_inno_profile_retains_padding_and_is_not_a_project_executable_profile(tmp_path):
    name, text_version = signing.inno_product_metadata(VERSION)
    executable = write_versioned_pe(
        tmp_path / "uninstaller.exe",
        [
            (0, [("000004b0", name, text_version)], (2, 8, 3, 0)),
            (1033, [("040904b0", name, text_version)], (2, 8, 3, 0)),
        ],
    )
    record = signing.unsigned_record(executable, VERSION, inno=True)
    assert record["product_name"] == PRODUCT.ljust(60, " ")
    assert record["product_version"] == VERSION.ljust(50, " ")
    assert len(record["metadata_resources"]) == 2
    assert all(row["strings"]["ProductName"] == name for row in record["metadata_resources"])
    with pytest.raises(RuntimeError, match="ProductName mismatch"):
        signing.unsigned_record(executable, VERSION)


@pytest.mark.parametrize(
    "mutation",
    [
        "short-name",
        "long-name",
        "tab-name",
        "emoji",
        "short-version",
        "wrong-version",
        "binary-version",
        "resource-conflict",
    ],
)
def test_inno_profile_rejects_every_deviation_from_exact_fixed_width(tmp_path, mutation):
    name, text_version = signing.inno_product_metadata(VERSION)
    binary = (2, 8, 3, 0)
    if mutation == "short-name":
        name = name[:-1]
    elif mutation == "long-name":
        name += " "
    elif mutation == "tab-name":
        name = name[:-1] + "\t"
    elif mutation == "emoji":
        name = "PuriPuly \u2764\ufe0f".ljust(60, " ")
    elif mutation == "short-version":
        text_version = text_version[:-1]
    elif mutation == "wrong-version":
        text_version = "9.9.9".ljust(50, " ")
    elif mutation == "binary-version":
        binary = (9, 9, 9, 0)
    first_name, first_version = signing.inno_product_metadata(VERSION)
    first = (0, [("000004b0", first_name, first_version)], (2, 8, 3, 0))
    if mutation == "resource-conflict":
        name = PRODUCT
    executable = write_versioned_pe(
        tmp_path / "uninstaller.exe",
        [
            first,
            (1033, [("040904b0", name, text_version)], binary),
        ],
    )
    with pytest.raises(RuntimeError, match="Product(Name|Version) mismatch"):
        signing.unsigned_record(executable, VERSION, inno=True)


def test_inno_version_contract_rejects_overlength_without_truncation():
    version = "0" * 51 + ".1.0"
    with pytest.raises(ValueError, match="50-character"):
        signing.inno_product_metadata(version)


def test_project_text_version_matches_canonical_parameter_in_every_table(tmp_path):
    executable = write_versioned_pe(
        tmp_path / "project.exe",
        [
            (
                1033,
                [
                    ("000004b0", PRODUCT, VERSION),
                    ("040904b0", PRODUCT, f"{VERSION}.0"),
                ],
                (2, 8, 3, 0),
            )
        ],
    )
    with pytest.raises(RuntimeError, match="ProductVersion mismatch"):
        signing.unsigned_record(executable, VERSION)


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell caller requires pwsh")
@pytest.mark.parametrize(
    "mode,padded_outer,padded_host,passes",
    [
        ("outer", True, False, True),
        ("outer", False, False, False),
        ("project", True, False, True),
        ("project", True, True, False),
    ],
)
def test_local_release_metadata_callers_select_exact_target_profile(
    tmp_path, mode, padded_outer, padded_host, passes
):
    root = Path(__file__).resolve().parents[2]
    inno_name, inno_version = signing.inno_product_metadata(VERSION)
    for name in signing.COMPONENTS[:-1]:
        padded = padded_host and name == "PuriPulyHeart.exe"
        write_versioned_pe(
            tmp_path / name,
            [
                (
                    1033,
                    [
                        (
                            "000004b0",
                            inno_name if padded else PRODUCT,
                            inno_version if padded else VERSION,
                        )
                    ],
                    (2, 8, 3, 0),
                )
            ],
        )
    outer = write_versioned_pe(
        tmp_path / "Setup.exe",
        [
            (
                1033,
                [
                    (
                        "000004b0",
                        inno_name if padded_outer else PRODUCT,
                        inno_version if padded_outer else VERSION,
                    )
                ],
                (2, 8, 3, 0),
            )
        ],
    )
    script = r"""
$ErrorActionPreference = 'Stop'
$ToolPython = $env:CALLER_PYTHON
$appVersion = $env:CALLER_VERSION
$installerPath = $env:CALLER_INSTALLER
$OutputDir = $env:CALLER_PAYLOAD
$hostExe = Join-Path $OutputDir 'PuriPulyHeart.exe'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($env:CALLER_SCRIPT, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$invoke = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Invoke-Checked' }, $true)
. ([scriptblock]::Create($invoke.Extent.Text))
$assignment = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -eq '$metadataCheck' }, $true)
. ([scriptblock]::Create($assignment.Extent.Text))
$variable = if ($env:CALLER_MODE -eq 'outer') { '$installerPath' } else { '$hostExe' }
$caller = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq 'Invoke-Checked' -and $n.Extent.Text.Contains($variable) }, $true)
. ([scriptblock]::Create($caller.Extent.Text))
"""
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src"),
            "CALLER_SCRIPT": str(root / "scripts/ci/build-native-release-artifacts.ps1"),
            "CALLER_PYTHON": sys.executable,
            "CALLER_VERSION": VERSION,
            "CALLER_PAYLOAD": str(tmp_path),
            "CALLER_INSTALLER": str(outer),
            "CALLER_MODE": mode,
        },
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        check=False,
    )
    assert (result.returncode == 0) is passes, result.stdout + result.stderr
    if not passes:
        assert "ProductName mismatch" in result.stderr
