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

    assert font_path.is_file()
    assert font_path.stat().st_size == NOTO_CJK_FONT_SIZE_BYTES
    assert _sha256(font_path) == NOTO_CJK_FONT_SHA256
    assert sha256sums == f"{NOTO_CJK_FONT_SHA256}  {NOTO_CJK_FONT_FILENAME}\n"
