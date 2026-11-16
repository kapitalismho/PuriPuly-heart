from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path

import pytest

from puripuly_heart.core.photo.image import PhotoImageError, PreparedPhoto, load_prepared_photo

_FIXTURES = Path(__file__).parent / "fixtures"


def _native_result(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source.png"
    jpeg = tmp_path / "send.jpg"
    source.write_bytes((_FIXTURES / "synthetic.png").read_bytes())
    jpeg.write_bytes((_FIXTURES / "synthetic.jpg").read_bytes())
    return {
        "photo_id": "synthetic-photo",
        "attempt_id": 1,
        "source_path": str(source),
        "jpeg_path": str(jpeg),
        "width": 1001,
        "height": 777,
        "scale": 720 / 777,
        "offset_x": (1280 - round(1001 * 720 / 777)) // 2,
        "offset_y": 0,
        "preparation_ms": 12.3456789,
    }


async def test_reader_transmits_exact_native_jpeg_and_keeps_native_source_metadata(tmp_path):
    result = _native_result(tmp_path)
    jpeg = Path(result["jpeg_path"]).read_bytes()
    source = Path(result["source_path"]).read_bytes()
    prepared = await load_prepared_photo(result, directory=tmp_path)
    assert base64.b64decode(prepared.jpeg_base64) == jpeg
    assert prepared.source_path.read_bytes() == source
    assert (prepared.source_width, prepared.source_height) == (1001, 777)
    assert prepared.scale == 720 / 777
    assert prepared.offset_x == result["offset_x"]
    assert prepared.offset_y == 0
    assert prepared.preparation_ms == result["preparation_ms"]
    assert set(tmp_path.iterdir()) == {Path(result["source_path"]), Path(result["jpeg_path"])}


def rectangle(left, top, right, bottom):
    return ((left, top), (right, top), (right, bottom), (left, bottom))


async def test_source_and_google_quads_map_each_axis_to_original_float_coordinates(tmp_path):
    prepared = await load_prepared_photo(_native_result(tmp_path), directory=tmp_path)
    quad = ((300.25, 100.5), (850.75, 150.25), (900.5, 600.75), (350.5, 550.25))
    expected = tuple(
        ((x - prepared.offset_x) / prepared.scale, y / prepared.scale) for x, y in quad
    )
    mapped = prepared.map_quad(quad, google=False)
    for actual, point in zip(mapped.quad, expected, strict=True):
        assert actual == pytest.approx(point)
    assert not mapped.corrected
    normalized = tuple((x / 1280 * 1000, y / 720 * 1000) for x, y in quad)
    google = prepared.map_quad(normalized, google=True)
    for actual, point in zip(google.quad, expected, strict=True):
        assert actual == pytest.approx(point)
    for coordinates, google in [
        (rectangle(0, 0, 1000, 1000), True),
        (rectangle(0, 0, 1280, 720), False),
    ]:
        with pytest.raises(PhotoImageError, match="photo_quad_clipping_excessive"):
            prepared.map_quad(coordinates, google=google)
    with pytest.raises(PhotoImageError, match="photo_quad_outside_image"):
        prepared.map_quad(rectangle(0, 0, prepared.offset_x - 1, 100), google=False)


@pytest.mark.parametrize(
    "width,height,scale,offset_x,offset_y",
    [
        (1600, 900, 0.8, 0, 0),
        (900, 1600, 0.8, 0, 0),
        (1000, 1000, 0.72, 280, 0),
        (600, 1600, 0.8, 120, 0),
        (1600, 600, 0.8, 0, 120),
    ],
)
def test_normalized_rotated_quad_preserves_text_local_start_and_letterboxing(
    width, height, scale, offset_x, offset_y
):
    prepared = PreparedPhoto("", Path("source.png"), width, height, scale, offset_x, offset_y, 0)
    original = (
        (width * 0.7, height * 0.2),
        (width * 0.8, height * 0.7),
        (width * 0.4, height * 0.8),
        (width * 0.3, height * 0.3),
    )
    send_width, send_height = prepared.send_size
    normalized = tuple(
        ((x * scale + offset_x) / send_width * 1000, (y * scale + offset_y) / send_height * 1000)
        for x, y in original
    )
    mapped = prepared.map_quad(normalized, google=True)
    assert not mapped.corrected
    for actual, expected in zip(mapped.quad, original, strict=True):
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "quad",
    [
        (),
        (0, 0, 1, 1),
        ((0, 0),) * 3,
        ((False, 0), (100, 0), (100, 100), (0, 100)),
        ((0, 0), (100, 0), (100, float("nan")), (0, 100)),
        ((0, 0), (100, 0), (100, float("inf")), (0, 100)),
        ((0, 0), (100, 0), (100, 10**400), (0, 100)),
        ((0, 0), (0, 100), (100, 100), (100, 0)),
        ((0, 0), (100, 100), (100, 0), (0, 100)),
        ((0, 0), (100, 0), (20, 20), (0, 100)),
        ((0, 0), (100, 0), (100, 0), (0, 100)),
        ((0, 0), (50, 0), (100, 0), (0, 100)),
    ],
)
@pytest.mark.parametrize("google", [False, True])
def test_invalid_shape_winding_nonfinite_or_degenerate_quads_are_rejected(quad, google):
    prepared = PreparedPhoto("", Path("source.png"), 1280, 720, 1, 0, 0, 0)
    with pytest.raises(PhotoImageError, match="photo_quad_invalid"):
        prepared.map_quad(quad, google=google)


@pytest.mark.parametrize(
    "quad,google,expected",
    [
        (rectangle(-2, 100, 100, 200), False, rectangle(0, 100, 100, 200)),
        (rectangle(0, 701, 100, 721), False, rectangle(0, 701, 100, 720)),
        (rectangle(-1, 100, 100, 200), True, rectangle(0, 72, 128, 144)),
        (rectangle(100, -2, 200, 100), True, rectangle(128, 0, 256, 72)),
        (
            ((-1, 100), (100, 120), (90, 220), (-2, 200)),
            False,
            ((0, 100), (100, 120), (90, 220), (0, 200)),
        ),
    ],
)
def test_bounded_vertex_clipping_preserves_usable_polygon_area(quad, google, expected):
    prepared = PreparedPhoto("", Path("source.png"), 1280, 720, 1, 0, 0, 0)
    mapped = prepared.map_quad(quad, google=google)
    for actual, point in zip(mapped.quad, expected, strict=True):
        assert actual == pytest.approx(point)
    assert mapped.corrected


@pytest.mark.parametrize(
    "quad,google,reason",
    [
        (rectangle(-3, 100, 100, 200), False, "photo_quad_clipping_excessive"),
        (rectangle(0, 702, 100, 721), False, "photo_quad_clipping_excessive"),
        (rectangle(51, 717, 271, 726), False, "photo_quad_clipping_excessive"),
        (rectangle(51, 722, 271, 730), False, "photo_quad_outside_image"),
        (rectangle(-2, 100, 100, 200), True, "photo_quad_clipping_excessive"),
        (((0, -2), (100, 0), (100, 4), (0, 2)), False, "photo_quad_clipping_excessive"),
        (((-2, 0), (0, -2), (100, 100), (0, 100)), False, "photo_quad_outside_image"),
    ],
)
def test_unusable_quads_are_not_relocated_squeezed_or_replaced_by_aabb(quad, google, reason):
    prepared = PreparedPhoto("", Path("source.png"), 1280, 720, 1, 0, 0, 0)
    with pytest.raises(PhotoImageError, match=reason):
        prepared.map_quad(quad, google=google)


def test_clipping_policy_uses_source_content_inside_letterboxing():
    prepared = PreparedPhoto("", Path("square.png"), 720, 720, 1, 280, 0, 0)
    mapped = prepared.map_quad(rectangle(278, 100, 378, 200), google=False)
    assert mapped.quad == rectangle(0, 100, 98, 200)
    assert mapped.corrected
    with pytest.raises(PhotoImageError, match="photo_quad_clipping_excessive"):
        prepared.map_quad(rectangle(277, 100, 378, 200), google=False)
    with pytest.raises(PhotoImageError, match="photo_quad_outside_image"):
        prepared.map_quad(rectangle(0, 100, 270, 200), google=False)


@pytest.mark.parametrize("key", ["source_path", "jpeg_path"])
async def test_native_result_cannot_read_outside_owned_directory(tmp_path, key):
    directory = tmp_path / "owned"
    directory.mkdir()
    result = _native_result(directory)
    outside = tmp_path / "personal-file"
    outside.write_bytes((_FIXTURES / "synthetic.jpg").read_bytes())
    result[key] = str(outside)
    with pytest.raises(PhotoImageError) as failure:
        await load_prepared_photo(result, directory=directory)
    assert str(failure.value) == "photo_preparation_invalid"
    assert outside.read_bytes() == (_FIXTURES / "synthetic.jpg").read_bytes()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("width", True),
        ("height", 0),
        ("scale", float("nan")),
        ("scale", 0),
        ("offset_x", -1),
        ("offset_y", 720),
        ("preparation_ms", float("inf")),
    ],
)
async def test_invalid_native_metadata_does_not_produce_payload(tmp_path, key, value):
    result = _native_result(tmp_path)
    result[key] = value
    with pytest.raises(PhotoImageError, match="photo_preparation_invalid"):
        await load_prepared_photo(result, directory=tmp_path)


async def test_missing_or_unfinished_native_jpeg_never_produces_payload(tmp_path):
    result = _native_result(tmp_path)
    jpeg = Path(result["jpeg_path"])
    jpeg.write_bytes(jpeg.read_bytes()[:-2])
    with pytest.raises(PhotoImageError, match="photo_prepared_image_incomplete"):
        await load_prepared_photo(result, directory=tmp_path)
    jpeg.unlink()
    with pytest.raises(PhotoImageError) as failure:
        await load_prepared_photo(result, directory=tmp_path)
    assert str(failure.value) == "photo_prepared_image_unavailable"
    assert str(tmp_path) not in str(failure.value)


async def test_native_jpeg_read_runs_off_loop_and_cancellation_drains_started_read(
    tmp_path, monkeypatch
):
    result = _native_result(tmp_path)
    jpeg = Path(result["jpeg_path"])
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    observed_thread = []
    original_read = Path.read_bytes

    def controlled_read(path):
        if path == jpeg:
            observed_thread.append(threading.get_ident())
            started.set()
            release.wait(2)
            try:
                return original_read(path)
            finally:
                finished.set()
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", controlled_read)
    task = asyncio.create_task(load_prepared_photo(result, directory=tmp_path))
    try:
        async with asyncio.timeout(1):
            while not started.is_set():
                await asyncio.sleep(0.002)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert not finished.is_set()
        assert observed_thread == [observed_thread[0]]
        assert observed_thread[0] != threading.get_ident()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
    assert jpeg.exists()
    assert Path(result["source_path"]).exists()
