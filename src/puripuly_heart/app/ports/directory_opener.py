from pathlib import Path
from typing import Protocol


class DirectoryOpenerPort(Protocol):
    def open(self, directory: Path) -> None: ...


__all__ = ["DirectoryOpenerPort"]
