from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.core.owned_thread import run_owned_thread_call

_SEND_LONG = 1280
_SEND_SHORT = 720


def send_size(width: int, height: int) -> tuple[int, int]:
    return (_SEND_SHORT, _SEND_LONG) if height > width else (_SEND_LONG, _SEND_SHORT)


class PhotoImageError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class MappedPhotoBox:
    bbox: tuple[float, float, float, float]
    corrected: bool


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

    @property
    def send_size(self) -> tuple[int, int]:
        return send_size(self.source_width, self.source_height)

    def map_box(self, box: tuple[int, int, int, int], *, google: bool) -> MappedPhotoBox:
        if len(box) != 4 or any(type(value) is not int for value in box):
            raise PhotoImageError("photo_box_invalid")
        send_width, send_height = self.send_size
        if google:
            y0, x0, y1, x1 = box
            try:
                left, top, right, bottom = (
                    x0 * send_width / 1000,
                    y0 * send_height / 1000,
                    x1 * send_width / 1000,
                    y1 * send_height / 1000,
                )
            except OverflowError:
                raise PhotoImageError("photo_box_invalid") from None
        else:
            left, top, right, bottom = box
        if left >= right or top >= bottom:
            raise PhotoImageError("photo_box_invalid")
        clipped = (
            max(left, self.offset_x),
            max(top, self.offset_y),
            min(right, send_width, self.offset_x + self.source_width * self.scale),
            min(bottom, send_height, self.offset_y + self.source_height * self.scale),
        )
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            raise PhotoImageError("photo_box_outside_image")
        clip_distance = max(
            clipped[0] - left, clipped[1] - top, right - clipped[2], bottom - clipped[3]
        )
        if clip_distance > 2:
            raise PhotoImageError("photo_box_clipping_excessive")
        area = (right - left) * (bottom - top)
        retained = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
        if retained < area * 0.95:
            raise PhotoImageError("photo_box_clipping_excessive")
        return MappedPhotoBox(
            (
                max(0.0, (clipped[0] - self.offset_x) / self.scale),
                max(0.0, (clipped[1] - self.offset_y) / self.scale),
                min(float(self.source_width), (clipped[2] - self.offset_x) / self.scale),
                min(float(self.source_height), (clipped[3] - self.offset_y) / self.scale),
            ),
            corrected=clipped != (left, top, right, bottom),
        )


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
            and 0 <= offset_x < send_size(width, height)[0]
            and type(offset_y) is int
            and 0 <= offset_y < send_size(width, height)[1]
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
