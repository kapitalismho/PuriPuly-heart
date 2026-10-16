from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(autouse=True)
def _isolated_user_data_dirs(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("user-data")
    for name in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME"):
        path = root / name.lower()
        path.mkdir()
        monkeypatch.setenv(name, str(path))
