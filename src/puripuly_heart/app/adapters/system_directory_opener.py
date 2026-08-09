from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SystemDirectoryOpener:
    platform: str = sys.platform

    def open(self, directory: Path) -> None:
        if self.platform == "win32":
            subprocess.Popen(["explorer", str(directory)])
            return
        if self.platform == "darwin":
            subprocess.Popen(["open", str(directory)])
            return
        subprocess.Popen(["xdg-open", str(directory)])
