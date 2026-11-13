from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from puripuly_heart.core.vrchat_scene_tailer import (
    VrchatSceneLogCandidate,
    VrchatSceneLogTailer,
    VrchatSceneLogTruncated,
    log_start_time,
    select_vrchat_log,
)


def test_log_start_time_parses_vrchat_names_in_local_time() -> None:
    parsed = log_start_time("output_log_2026-09-07_01-48-37.txt")

    assert parsed is not None
    local = time.localtime(parsed)
    assert (local.tm_year, local.tm_mon, local.tm_mday, local.tm_hour, local.tm_min) == (
        2026,
        9,
        7,
        1,
        48,
    )


def test_log_start_time_rejects_foreign_names() -> None:
    assert log_start_time("output_log_notes.txt") is None
    assert log_start_time("player.log") is None
    assert log_start_time("") is None
    assert log_start_time("output_log_2026-13-99_99-99-99.txt") is None


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _candidate(
    path: Path, name_time: float, created: float, modified: float
) -> VrchatSceneLogCandidate:
    return VrchatSceneLogCandidate(
        path=path, name_time=name_time, created=created, modified=modified
    )


def test_select_accepts_fresh_match_for_current_lifecycle() -> None:
    now = 1788767641.0
    create = now - 5
    current = _candidate(Path("output_log_now.txt"), now - 5, now - 5, now)

    selected = select_vrchat_log((current,), create_time=create, now=now)

    assert selected == current.path


def test_select_rejects_previous_lifecycle_log_on_immediate_restart() -> None:
    now = 1788767641.0
    create = now - 5
    previous = _candidate(Path("output_log_old.txt"), now - 600, now - 600, now - 8)

    assert select_vrchat_log((previous,), create_time=create, now=now) is None


def test_select_rejects_prior_lifecycle_log_with_recent_exit_flush() -> None:
    create = 1788767641.0
    now = create + 0.2
    flushed = _candidate(Path("output_log_prior.txt"), create - 300, create - 300, create + 0.1)

    assert select_vrchat_log((flushed,), create_time=create, now=now) is None


def test_select_rejects_prior_name_with_nearby_creation() -> None:
    create = 1788767641.0
    now = create + 0.2
    skewed = _candidate(Path("output_log_prior.txt"), create - 300, create - 30, create + 0.1)

    assert select_vrchat_log((skewed,), create_time=create, now=now) is None


def test_select_rejects_same_second_name_created_before_process_start() -> None:
    create = 1788767641.4
    now = create + 0.2
    ambiguous = _candidate(Path("output_log_same.txt"), 1788767641.0, create - 0.5, now)

    assert select_vrchat_log((ambiguous,), create_time=create, now=now) is None


def test_select_accepts_same_second_name_created_after_process_start() -> None:
    create = 1788767641.4
    now = create + 0.2
    current = _candidate(Path("output_log_same.txt"), 1788767641.0, create + 0.05, now)

    assert select_vrchat_log((current,), create_time=create, now=now) == current.path


def test_select_accepts_name_second_before_process_start() -> None:
    create = 1788875102.39
    now = create + 1694.0
    current = _candidate(Path("output_log_live.txt"), 1788875101.0, create, now - 2)

    assert select_vrchat_log((current,), create_time=create, now=now) == current.path


def test_select_rejects_name_beyond_start_tolerance() -> None:
    create = 1788767641.0
    now = create + 0.2
    stale = _candidate(Path("output_log_stale.txt"), create - 30, create - 0.05, now)

    assert select_vrchat_log((stale,), create_time=create, now=now) is None


def test_select_rejects_prestart_name_without_nearby_creation() -> None:
    create = 1788767641.0
    now = create + 0.2
    previous = _candidate(Path("output_log_previous.txt"), create - 3, create - 600, now - 2)

    assert select_vrchat_log((previous,), create_time=create, now=now) is None


def test_select_accepts_delayed_fresh_log_over_stale_previous() -> None:
    now = 1788767641.0
    create = now - 5
    previous = _candidate(Path("output_log_old.txt"), now - 600, now - 600, now - 8)
    fresh = _candidate(Path("output_log_new.txt"), now - 2, now - 2, now)

    selected = select_vrchat_log((previous, fresh), create_time=create, now=now)

    assert selected == fresh.path


def test_select_accepts_rotation_substantially_after_process_start() -> None:
    now = 1788767641.0
    create = now - 7200
    first = _candidate(Path("output_log_first.txt"), create, create, now - 60)
    rotated = _candidate(Path("output_log_second.txt"), now - 10, now - 10, now)

    selected = select_vrchat_log((first, rotated), create_time=create, now=now)

    assert selected == rotated.path


def test_select_accepts_late_start_long_running_process() -> None:
    now = 1788767641.0
    create = now - 7200
    active = _candidate(Path("output_log_active.txt"), create, create, now - 5)

    selected = select_vrchat_log((active,), create_time=create, now=now)

    assert selected == active.path


def test_select_accepts_quiet_origin_log_without_recent_writes() -> None:
    now = 1788767641.0
    create = now - 3600
    quiet = _candidate(Path("output_log_quiet.txt"), create, create, create + 10)

    selected = select_vrchat_log((quiet,), create_time=create, now=now)

    assert selected == quiet.path


def test_select_rejects_future_names() -> None:
    now = 1788767641.0
    create = now - 5
    future = _candidate(Path("output_log_future.txt"), now + 3600, now, now)

    assert select_vrchat_log((future,), create_time=create, now=now) is None


def test_select_prefers_newest_live_log() -> None:
    now = 1788767641.0
    create = now - 7200
    first = _candidate(Path("output_log_first.txt"), create, create, now)
    second = _candidate(Path("output_log_second.txt"), now - 100, now - 100, now)

    selected = select_vrchat_log((first, second), create_time=create, now=now)

    assert selected == second.path


def test_scan_candidates_skips_unparseable_names() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write(root / "output_log_2026-09-07_01-48-37.txt", "known\n")
        _write(root / "output_log_scratch.txt", "scratch\n")
        tailer = VrchatSceneLogTailer(root)

        found = tailer.scan_candidates()

        assert [item.path.name for item in found] == ["output_log_2026-09-07_01-48-37.txt"]
        assert found[0].name_time == log_start_time("output_log_2026-09-07_01-48-37.txt")


def test_replay_returns_complete_lines_and_holds_partial_tail() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        path = root / "output_log_2026-09-07_01-48-37.txt"
        _write(path, "first\nsecond\npartial")
        tailer = VrchatSceneLogTailer(root)

        assert tailer.begin_replay(path) == ["first", "second"]
        assert tailer.read_new_lines() == []

        with open(path, "a", encoding="utf-8") as stream:
            stream.write("-tail\nthird\n")

        assert tailer.read_new_lines() == ["partial-tail", "third"]


def test_incremental_read_survives_split_multibyte_characters() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        path = root / "output_log_2026-09-07_01-48-37.txt"
        _write(path, "alpha\n")
        tailer = VrchatSceneLogTailer(root, read_chunk_bytes=3)

        assert tailer.begin_replay(path) == ["alpha"]

        with open(path, "a", encoding="utf-8") as stream:
            stream.write("héllo-wörld\n")

        assert tailer.read_new_lines() == ["héllo-wörld"]


def test_truncation_raises_and_replay_recovers() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        path = root / "output_log_2026-09-07_01-48-37.txt"
        _write(path, "first\nsecond\n")
        tailer = VrchatSceneLogTailer(root)

        assert tailer.begin_replay(path) == ["first", "second"]

        _write(path, "replacement\n")
        with pytest.raises(VrchatSceneLogTruncated):
            tailer.read_new_lines()

        assert tailer.begin_replay(path) == ["replacement"]


def test_same_path_replacement_with_equal_size_is_detected() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        path = root / "output_log_2026-09-07_01-48-37.txt"
        _write(path, "aaaa\n")
        tailer = VrchatSceneLogTailer(root)

        assert tailer.begin_replay(path) == ["aaaa"]
        assert tailer.read_new_lines() == []

        replacement = root / "output_log_2026-09-07_01-48-37.next"
        _write(replacement, "bbbb\n")
        os.replace(replacement, path)

        with pytest.raises(VrchatSceneLogTruncated):
            tailer.read_new_lines()

        assert tailer.begin_replay(path) == ["bbbb"]


def test_same_size_in_place_rewrite_is_detected() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        path = root / "output_log_2026-09-07_01-48-37.txt"
        _write(path, "aaaa\n")
        tailer = VrchatSceneLogTailer(root)

        assert tailer.begin_replay(path) == ["aaaa"]

        stamp = path.stat().st_mtime + 5
        _write(path, "bbbb\n")
        os.utime(path, (stamp, stamp))

        with pytest.raises(VrchatSceneLogTruncated):
            tailer.read_new_lines()


def test_missing_file_read_raises_os_error() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        path = root / "output_log_2026-09-07_01-48-37.txt"
        _write(path, "first\n")
        tailer = VrchatSceneLogTailer(root)
        tailer.begin_replay(path)
        path.unlink()

        with pytest.raises(OSError):
            tailer.read_new_lines()


def test_scan_distinguishes_missing_directory_from_failed_observation(tmp_path, monkeypatch):
    missing = VrchatSceneLogTailer(tmp_path / "missing")
    assert missing.scan_candidates() == ()
    original = PermissionError(13, "private scene log directory")

    def unavailable(_path):
        raise original

    monkeypatch.setattr(Path, "iterdir", unavailable)
    with pytest.raises(PermissionError) as caught:
        VrchatSceneLogTailer(tmp_path).scan_candidates()
    assert caught.value is original
