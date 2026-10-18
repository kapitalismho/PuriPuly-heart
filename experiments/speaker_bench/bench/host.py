"""Development host for the speaker benchmark.

Substitutes only the Peer loopback device source and installs read-only observers,
then runs the ordinary `run-headless` composition.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parents[1]
sys.path.insert(0, str(BENCH_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bench import observe, source  # noqa: E402


def main() -> int:
    control = Path(os.environ[source.CONTROL_ENV])
    observe.install(control / "observations.jsonl")

    from puripuly_heart.core.audio import desktop_source

    desktop_source.DesktopLoopbackAudioSource = source.BenchLoopbackSource

    from puripuly_heart.main import main as app_main

    return app_main(["run-headless", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
