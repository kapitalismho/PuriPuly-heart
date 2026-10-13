from __future__ import annotations

import os
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "src"))
os.environ["PYTHONPATH"] = str(repo / "src")
os.environ["PURIPULY_HEART_TEST_PROFILE"] = "desktop-gui-smoke"
os.environ["PURIPULY_HEART_TEST_DATA_DIR"] = str(repo.parent / "desktop-gui-smoke-data")
os.environ["FLET_VIEW_PATH"] = str(repo / "build" / "flet" / "windows" / "flet")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.chdir(repo)

from puripuly_heart.main import main

main()
