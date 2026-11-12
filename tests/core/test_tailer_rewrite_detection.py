from __future__ import annotations

from pathlib import Path

import pytest

from puripuly_heart.core.vrchat_scene_tailer import (
    VrchatSceneLogTailer,
    VrchatSceneLogTruncated,
)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _emulate_stable_creation_identity(tailer: VrchatSceneLogTailer, path: Path) -> None:
    status = path.stat()
    recorded = tailer._identity
    if recorded is None:
        return
    if len(recorded) == 2:
        tailer._identity = (status.st_dev, status.st_ino)
    else:
        tailer._identity = (status.st_dev, status.st_ino, status.st_ctime)


def test_append_is_not_reported_as_truncation(tmp_path: Path) -> None:
    path = tmp_path / "output_log_2026-09-07_01-48-37.txt"
    _write(path, "first\nsecond\npartial")
    tailer = VrchatSceneLogTailer(tmp_path)

    assert tailer.begin_replay(path) == ["first", "second"]
    assert tailer.read_new_lines() == []

    with open(path, "a", encoding="utf-8") as stream:
        stream.write("-tail\nthird\n")

    assert tailer.read_new_lines() == ["partial-tail", "third"]


def test_append_across_multibyte_chunks(tmp_path: Path) -> None:
    path = tmp_path / "output_log_2026-09-07_01-48-37.txt"
    _write(path, "alpha\n")
    tailer = VrchatSceneLogTailer(tmp_path, read_chunk_bytes=3)

    assert tailer.begin_replay(path) == ["alpha"]

    with open(path, "a", encoding="utf-8") as stream:
        stream.write("héllo-wörld\n")

    assert tailer.read_new_lines() == ["héllo-wörld"]


def test_in_place_rewrite_longer_than_offset_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "output_log_2026-09-07_01-48-37.txt"
    _write(path, "line1\nline2\n")
    tailer = VrchatSceneLogTailer(tmp_path)

    assert tailer.begin_replay(path) == ["line1", "line2"]

    _write(path, "NEWa\nNEWb\nNEWc\n!\n")
    _emulate_stable_creation_identity(tailer, path)

    with pytest.raises(VrchatSceneLogTruncated):
        tailer.read_new_lines()

    assert tailer.begin_replay(path) == ["NEWa", "NEWb", "NEWc", "!"]


def test_shorter_rewrite_is_detected_and_replay_recovers(tmp_path: Path) -> None:
    path = tmp_path / "output_log_2026-09-07_01-48-37.txt"
    _write(path, "first\nsecond\n")
    tailer = VrchatSceneLogTailer(tmp_path)

    assert tailer.begin_replay(path) == ["first", "second"]

    _write(path, "replacement\n")
    with pytest.raises(VrchatSceneLogTruncated):
        tailer.read_new_lines()

    assert tailer.begin_replay(path) == ["replacement"]


def test_same_size_in_place_rewrite_is_detected(tmp_path: Path) -> None:
    import os as _os

    path = tmp_path / "output_log_2026-09-07_01-48-37.txt"
    _write(path, "aaaa\n")
    tailer = VrchatSceneLogTailer(tmp_path)

    assert tailer.begin_replay(path) == ["aaaa"]

    stamp = path.stat().st_mtime + 5
    _write(path, "bbbb\n")
    _os.utime(path, (stamp, stamp))

    with pytest.raises(VrchatSceneLogTruncated):
        tailer.read_new_lines()


@pytest.mark.parametrize("read_chunk_bytes", [3, 64, 65, 65536])
def test_same_size_and_mtime_rewrite_is_detected_and_replay_recovers(
    tmp_path: Path, read_chunk_bytes: int
) -> None:
    import os as _os

    path = tmp_path / "output_log_2026-09-07_01-48-37.txt"
    path.write_bytes(b"a" * 128 + b"\n")
    tailer = VrchatSceneLogTailer(tmp_path, read_chunk_bytes=read_chunk_bytes)
    assert tailer.begin_replay(path) == ["a" * 128]
    original_status = path.stat()

    path.write_bytes(b"a" * 96 + b"b" + b"a" * 31 + b"\n")
    _os.utime(path, ns=(original_status.st_atime_ns, original_status.st_mtime_ns))
    rewritten_status = path.stat()
    assert (rewritten_status.st_dev, rewritten_status.st_ino) == (
        original_status.st_dev,
        original_status.st_ino,
    )
    assert rewritten_status.st_size == original_status.st_size
    assert rewritten_status.st_mtime_ns == original_status.st_mtime_ns

    with pytest.raises(VrchatSceneLogTruncated):
        tailer.read_new_lines()

    assert tailer.begin_replay(path) == ["a" * 96 + "b" + "a" * 31]
    with open(path, "ab") as stream:
        stream.write(b"after-rewrite\n")
    assert tailer.read_new_lines() == ["after-rewrite"]


def test_equal_size_replacement_file_is_detected(tmp_path: Path) -> None:
    import os as _os

    path = tmp_path / "output_log_2026-09-07_01-48-37.txt"
    _write(path, "aaaa\n")
    tailer = VrchatSceneLogTailer(tmp_path)

    assert tailer.begin_replay(path) == ["aaaa"]
    assert tailer.read_new_lines() == []

    replacement = tmp_path / "output_log_2026-09-07_01-48-37.next"
    _write(replacement, "bbbb\n")
    _os.replace(replacement, path)

    with pytest.raises(VrchatSceneLogTruncated):
        tailer.read_new_lines()

    assert tailer.begin_replay(path) == ["bbbb"]
