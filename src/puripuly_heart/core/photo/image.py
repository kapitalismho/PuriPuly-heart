from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.core.owned_thread import run_owned_thread_call

from .parsing import Quad

_SEND_LONG = 1280
_SEND_SHORT = 720


def send_size(width: int, height: int) -> tuple[int, int]:
    return (_SEND_SHORT, _SEND_LONG) if height > width else (_SEND_LONG, _SEND_SHORT)


class PhotoImageError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class MappedPhotoQuad:
    quad: Quad
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

    def map_quad(self, quad: Quad, *, google: bool) -> MappedPhotoQuad:
        try:
            if len(quad) != 4 or any(
                len(point) != 2 or any(type(value) not in (int, float) for value in point)
                for point in quad
            ):
                raise PhotoImageError("photo_quad_invalid")
            send_width, send_height = self.send_size
            points = tuple(
                (
                    float(x) * send_width / 1000 if google else float(x),
                    float(y) * send_height / 1000 if google else float(y),
                )
                for x, y in quad
            )
        except TypeError, ValueError, OverflowError:
            raise PhotoImageError("photo_quad_invalid") from None
        if not all(math.isfinite(value) for point in points for value in point):
            raise PhotoImageError("photo_quad_invalid")
        area = _quad_area(points)
        if area is None:
            raise PhotoImageError("photo_quad_invalid")
        left, top = self.offset_x, self.offset_y
        right = min(send_width, left + self.source_width * self.scale)
        bottom = min(send_height, top + self.source_height * self.scale)
        clipped = tuple((min(max(x, left), right), min(max(y, top), bottom)) for x, y in points)
        retained = _quad_area(clipped)
        if retained is None:
            raise PhotoImageError("photo_quad_outside_image")
        correction = max(
            abs(value - original)
            for point, before in zip(clipped, points, strict=True)
            for value, original in zip(point, before, strict=True)
        )
        if correction > 2 or retained < area * 0.95:
            raise PhotoImageError("photo_quad_clipping_excessive")
        a, b, c, d = (
            (
                min(float(self.source_width), max(0.0, (x - left) / self.scale)),
                min(float(self.source_height), max(0.0, (y - top) / self.scale)),
            )
            for x, y in clipped
        )
        return MappedPhotoQuad((a, b, c, d), corrected=clipped != points)


def _quad_area(points: tuple[tuple[float, float], ...]) -> float | None:
    for index in range(4):
        a, b, c = (points[(index + step) % 4] for step in range(3))
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if not math.isfinite(cross) or cross <= 0:
            return None
    a, b, c, d = points
    area = (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
        + (c[0] - a[0]) * (d[1] - a[1])
        - (c[1] - a[1]) * (d[0] - a[0])
    ) / 2
    return area if math.isfinite(area) and area > 0 else None


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
