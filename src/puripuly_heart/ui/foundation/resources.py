from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


@dataclass(frozen=True, slots=True)
class FoundationResourceLocator:
    assets_root: Path

    @classmethod
    def packaged(cls) -> FoundationResourceLocator:
        return cls(Path(__file__).resolve().parents[2] / "data")

    def asset_url(self, relative_path: str) -> str:
        normalized = self._normalize(relative_path)
        return normalized.as_posix()

    def filesystem_path(self, relative_path: str) -> Path:
        normalized = self._normalize(relative_path)
        return self.assets_root.joinpath(*normalized.parts)

    def require_file(self, relative_path: str) -> Path:
        path = self.filesystem_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Required UI asset is missing: {relative_path}")
        return path

    @staticmethod
    def _normalize(relative_path: str) -> PurePosixPath:
        normalized_text = relative_path.replace("\\", "/")
        path = PurePosixPath(normalized_text)
        if (
            not normalized_text
            or path.is_absolute()
            or PureWindowsPath(relative_path).is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(f"UI asset path must be a safe relative path: {relative_path!r}")
        return path


DEFAULT_FOUNDATION_RESOURCES = FoundationResourceLocator.packaged()


__all__ = ["DEFAULT_FOUNDATION_RESOURCES", "FoundationResourceLocator"]
