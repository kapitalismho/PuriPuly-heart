from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path

import pytest

from puripuly_heart.core.photo.image import PhotoImageError, load_prepared_photo

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


async def test_source_and_google_boxes_map_to_original_float_coordinates(tmp_path):
    prepared = await load_prepared_photo(_native_result(tmp_path), directory=tmp_path)
    expected = (
        (300 - prepared.offset_x) / prepared.scale,
        100 / prepared.scale,
        (900 - prepared.offset_x) / prepared.scale,
        600 / prepared.scale,
    )
    assert prepared.map_box((300, 100, 900, 600), google=False) == pytest.approx(expected)
    assert prepared.map_box((0, 0, 1000, 1000), google=True) == pytest.approx((0, 0, 1001, 777))
    assert prepared.map_box((0, 0, 1280, 720), google=False) == pytest.approx((0, 0, 1001, 777))
    google = prepared.map_box((100, 200, 800, 700), google=True)
    assert google == pytest.approx(
        (
            (256 - prepared.offset_x) / prepared.scale,
            72 / prepared.scale,
            (896 - prepared.offset_x) / prepared.scale,
            576 / prepared.scale,
        )
    )
    with pytest.raises(PhotoImageError, match="photo_box_outside_image"):
        prepared.map_box((0, 0, prepared.offset_x - 1, 100), google=False)
    for box in [(10, 0, 9, 100), (0, 0, 1281, 720), (False, 0, 500, 300)]:
        with pytest.raises(PhotoImageError, match="photo_box_invalid"):
            prepared.map_box(box, google=False)
    with pytest.raises(PhotoImageError, match="photo_box_invalid"):
        prepared.map_box((0, 0, 1001, 1000), google=True)


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
