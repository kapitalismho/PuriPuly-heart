from __future__ import annotations

from puripuly_heart.core.photo.capture import PhotoCapture
from puripuly_heart.core.photo.recycle import recycle_captured_photo


def _capture(path):
    status = path.stat()
    return PhotoCapture(path, status.st_size, (status.st_dev, status.st_ino), 0, 0.0)


def test_recycles_the_captured_file_when_it_is_unchanged(tmp_path):
    path = tmp_path / "VRChat_2026-11-15.png"
    path.write_bytes(b"png")
    sent = []

    assert recycle_captured_photo(_capture(path), send=lambda value: sent.append(value) or True)
    assert sent == [str(path)]


def test_leaves_a_replaced_or_missing_file_alone(tmp_path):
    path = tmp_path / "VRChat_2026-11-15.png"
    path.write_bytes(b"png")
    captured = _capture(path)
    keeper = tmp_path / "keeper.png"
    keeper.write_bytes(b"keep")
    path.unlink()
    sent = []

    assert not recycle_captured_photo(captured, send=lambda value: sent.append(value) or True)
    keeper.replace(path)
    assert not recycle_captured_photo(captured, send=lambda value: sent.append(value) or True)
    assert sent == []
    assert path.read_bytes() == b"keep"
