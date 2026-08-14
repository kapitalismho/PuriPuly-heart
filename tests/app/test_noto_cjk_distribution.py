from __future__ import annotations

import hashlib
from pathlib import Path

from tests.helpers.paths import REPO_ROOT as ROOT

NOTO_CJK_FONT_SOURCE_RELATIVE_DIR = Path("src/puripuly_heart/data/fonts")
NOTO_CJK_PROVENANCE_RELATIVE_DIR = Path("third_party/noto-sans-cjk")
NOTO_CJK_FONT_FILENAME = "NotoSansCJK-Medium.ttc"
NOTO_CJK_FONT_SHA256 = "197d5e1e019faca33a4d55931c7d68b8056f3b97cb862049f5cb8de9efdfb8ce"
NOTO_CJK_FONT_SIZE_BYTES = 18_354_360


def _read(relative_path: str | Path) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_vendored_noto_sans_cjk_medium_ttc_matches_recorded_provenance() -> None:
    font_path = ROOT / NOTO_CJK_FONT_SOURCE_RELATIVE_DIR / NOTO_CJK_FONT_FILENAME
    sha256sums = _read(NOTO_CJK_PROVENANCE_RELATIVE_DIR / "SHA256SUMS.txt")
    provenance_readme = _read(NOTO_CJK_PROVENANCE_RELATIVE_DIR / "README.md")

    assert font_path.is_file()
    assert font_path.stat().st_size == NOTO_CJK_FONT_SIZE_BYTES
    assert _sha256(font_path) == NOTO_CJK_FONT_SHA256
    assert sha256sums == f"{NOTO_CJK_FONT_SHA256}  {NOTO_CJK_FONT_FILENAME}\n"
    assert f"- Byte length: `{NOTO_CJK_FONT_SIZE_BYTES:,}`" in provenance_readme
    assert f"`{NOTO_CJK_FONT_SHA256}`" in provenance_readme


def test_build_spec_stages_noto_cjk_font_and_distribution_provenance_files() -> None:
    spec = _read("build.spec")

    assert (
        'NOTO_CJK_SOURCE_FONT_PATH = src_path / "puripuly_heart" / "data" / "fonts" / '
        '"NotoSansCJK-Medium.ttc"' in spec
    )
    assert '(str(src_path / "puripuly_heart" / "data"), "puripuly_heart/data")' in spec
    assert 'NOTO_CJK_PACKAGED_FONT_RELATIVE_DIR = Path("fonts")' not in spec
    assert 'NOTO_CJK_PROVENANCE_DIR = Path("third_party/noto-sans-cjk").resolve()' in spec
    for provenance_file in ("OFL.txt", "README.md", "SHA256SUMS.txt"):
        assert (
            f'(str(NOTO_CJK_PROVENANCE_DIR / "{provenance_file}"), '
            "NOTO_CJK_PACKAGED_PROVENANCE_RELATIVE_DIR.as_posix())" in spec
        )


def test_release_script_checks_staged_noto_cjk_font_before_installer_creation() -> None:
    script = _read("scripts/ci/build-release-artifacts.ps1")

    assert f'$PinnedNotoCjkFontSha256 = "{NOTO_CJK_FONT_SHA256}"' in script
    assert (
        "$notoCjkFontSourcePath = Join-Path $PWD "
        '"src/puripuly_heart/data/fonts/NotoSansCJK-Medium.ttc"' in script
    )
    assert (
        "$packagedNotoCjkFontPath = Join-Path $distDir "
        '"puripuly_heart\\data\\fonts\\NotoSansCJK-Medium.ttc"' in script
    )
    assert (
        "Assert-FileSha256Equals -Path $notoCjkFontSourcePath -ExpectedSha256 "
        '$PinnedNotoCjkFontSha256 -Label "Source Noto Sans CJK Medium TTC"' in script
    )
    packaged_hash_check = (
        "Assert-FileSha256Equals -Path $packagedNotoCjkFontPath -ExpectedSha256 "
        '$PinnedNotoCjkFontSha256 -Label "Packaged Noto Sans CJK Medium TTC"'
    )
    assert packaged_hash_check in script
    assert script.index(packaged_hash_check) < script.index('Write-Host "Building installer..."')


def test_installer_configuration_places_noto_cjk_font_under_packaged_app_data_fonts() -> None:
    script = _read("installer.iss")

    assert (
        "#define NotoCjkFontRelativePath "
        '"puripuly_heart\\data\\fonts\\NotoSansCJK-Medium.ttc"' in script
    )
    assert (
        "; Bundled CJK font is staged at {#MyPackagedAppDir}\\{#NotoCjkFontRelativePath}; "
        "the recursive packaged-tree copy installs it to {app}\\{#NotoCjkFontRelativePath}."
        in script
    )
    assert (
        'Source: "{#MyPackagedAppDir}\\*"; DestDir: "{app}"; Flags: ignoreversion '
        "recursesubdirs createallsubdirs; Excludes: "
        '"{#MyAppExeName},{#MyOverlayExeName},{#MyGpuWorkerExeName}"' in script
    )
    assert "NotoSansCJK-Medium.ttc" not in script.split("Excludes:", maxsplit=1)[1]


def test_third_party_notices_cover_noto_sans_cjk_medium_distribution() -> None:
    notices = _read("src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt")

    start = notices.index("\nNoto Sans CJK Medium TTC\n")
    end = notices.index("----", start)
    section = notices[start:end]

    assert "Noto Sans CJK" in section
    assert "NotoSansCJK-Medium.ttc" in section
    assert "License: SIL Open Font License 1.1" in section
    assert "Source/provenance bundle: third_party\\noto-sans-cjk\\" in section
    assert (
        "Bundled source path: src\\puripuly_heart\\data\\fonts\\NotoSansCJK-Medium.ttc" in section
    )
    assert "Bundled runtime path: puripuly_heart\\data\\fonts\\NotoSansCJK-Medium.ttc" in section
    assert "Byte length: 18,354,360" in section
    assert f"SHA256: {NOTO_CJK_FONT_SHA256}" in section
    assert "Sans2.004" in section
