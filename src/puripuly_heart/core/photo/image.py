from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.core.owned_thread import run_owned_thread_call

_SEND_WIDTH = 1280
_SEND_HEIGHT = 720


class PhotoImageError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PreparedPhoto:
    jpeg_base64: str
    source_path: Path
    source_width: int
    source_height: int
    scale: float
    offset_x: int
    offset_y: int
    preparation_ms: float

    def map_box(
        self, box: tuple[int, int, int, int], *, google: bool
    ) -> tuple[float, float, float, float]:
        if len(box) != 4 or any(type(value) is not int for value in box):
            raise PhotoImageError("photo_box_invalid")
        if google:
            y0, x0, y1, x1 = box
            if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
                raise PhotoImageError("photo_box_invalid")
            left, top, right, bottom = (
                x0 * _SEND_WIDTH / 1000,
                y0 * _SEND_HEIGHT / 1000,
                x1 * _SEND_WIDTH / 1000,
                y1 * _SEND_HEIGHT / 1000,
            )
        else:
            left, top, right, bottom = box
            if not (0 <= left < right <= _SEND_WIDTH and 0 <= top < bottom <= _SEND_HEIGHT):
                raise PhotoImageError("photo_box_invalid")
        mapped = (
            max(0.0, (left - self.offset_x) / self.scale),
            max(0.0, (top - self.offset_y) / self.scale),
            min(float(self.source_width), (right - self.offset_x) / self.scale),
            min(float(self.source_height), (bottom - self.offset_y) / self.scale),
        )
        if mapped[0] >= mapped[2] or mapped[1] >= mapped[3]:
            raise PhotoImageError("photo_box_outside_image")
        return mapped


def _prepared_path(result: Mapping[str, object], key: str, directory: Path) -> Path:
    value = result.get(key)
    if not isinstance(value, str) or not value:
        raise PhotoImageError("photo_preparation_invalid")
    path = Path(value).resolve(strict=True)
    if not path.is_relative_to(directory) or not path.is_file():
        raise PhotoImageError("photo_preparation_invalid")
    return path


def _load_prepared_photo(result: Mapping[str, object], directory: Path) -> PreparedPhoto:
    try:
        directory = directory.resolve(strict=True)
        source_path = _prepared_path(result, "source_path", directory)
        jpeg_path = _prepared_path(result, "jpeg_path", directory)
        width, height = result.get("width"), result.get("height")
        scale = result.get("scale")
        offset_x, offset_y = result.get("offset_x"), result.get("offset_y")
        preparation_ms = result.get("preparation_ms")
        if not (
            type(width) is int
            and width > 0
            and type(height) is int
            and height > 0
            and type(scale) in (int, float)
            and math.isfinite(scale)
            and scale > 0
            and type(offset_x) is int
            and 0 <= offset_x < _SEND_WIDTH
            and type(offset_y) is int
            and 0 <= offset_y < _SEND_HEIGHT
            and type(preparation_ms) in (int, float)
            and math.isfinite(preparation_ms)
            and preparation_ms >= 0
            and source_path != jpeg_path
        ):
            raise PhotoImageError("photo_preparation_invalid")
        jpeg = jpeg_path.read_bytes()
        if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
            raise PhotoImageError("photo_prepared_image_incomplete")
        return PreparedPhoto(
            jpeg_base64=base64.b64encode(jpeg).decode("ascii"),
            source_path=source_path,
            source_width=width,
            source_height=height,
            scale=float(scale),
            offset_x=offset_x,
            offset_y=offset_y,
            preparation_ms=float(preparation_ms),
        )
    except OSError:
        raise PhotoImageError("photo_prepared_image_unavailable") from None


async def load_prepared_photo(result: Mapping[str, object], *, directory: Path) -> PreparedPhoto:
    return await run_owned_thread_call(lambda: _load_prepared_photo(result, directory))
