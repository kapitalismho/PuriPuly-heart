"""Desktop markerless speaker-transition emphasis visual check."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_speaker_modes_steamvr import main


def _main(argv: list[str] | None = None) -> int:
    return main(argv, surface="desktop")


if __name__ == "__main__":
    raise SystemExit(_main())
