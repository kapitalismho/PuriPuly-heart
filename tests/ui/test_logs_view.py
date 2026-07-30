"""Tests for LogsView batch deletion optimization."""

import asyncio
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest

from puripuly_heart.ui.views import logs as logs_module
from puripuly_heart.ui.views.logs import (
    CLEANUP_BATCH,
    MAX_LOG_ENTRIES,
    FletLogHandler,
    LiveLogViewModel,
    LogsView,
    _get_log_dir,
)


class TestLogsView:
    def test_logs_view_folder_button_uses_content_api(self):
        view = LogsView()

        assert view._folder_button.content == logs_module.t("logs.open_folder")

    def test_logs_view_exposes_mode_button_with_current_mode_label_and_icon(self):
        view = LogsView()

        assert view.runtime_logging_mode == "basic"
        assert view._mode_button.content == logs_module.t("logs.mode.basic")
        assert view._mode_button.icon == logs_module.ft.Icons.ARTICLE

    def test_logs_view_exposes_conversation_button_after_mode_button(self):
        view = LogsView()

        assert view._conversation_button.content == logs_module.t("logs.conversation.show")
        assert view._conversation_button.icon == logs_module.ft.Icons.CHAT_BUBBLE_OUTLINE

        assert view._header_button_row.controls == [
            view._folder_button,
            view._mode_button,
            view._conversation_button,
        ]

    def test_conversation_toggle_shows_empty_state_and_returns_to_system_logs(self):
        view = LogsView()

        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            view.append_log("system line")
            view._on_conversation_button_click(SimpleNamespace())

            assert view._log_text.value == logs_module.t("logs.conversation.empty")
            assert view._conversation_button.content == logs_module.t("logs.conversation.hide")

            view.set_runtime_logging_mode("detailed")
            assert view._conversation_button.content == logs_module.t("logs.conversation.hide")
            assert view._mode_button.content == logs_module.t("logs.mode.detailed")

            view._on_conversation_button_click(SimpleNamespace())

        assert view._log_text.value == "system line"
        assert view._conversation_button.content == logs_module.t("logs.conversation.show")

    def test_conversation_records_render_with_runtime_localized_source_labels(self):
        view = LogsView()

        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            with patch.object(
                logs_module,
                "_format_conversation_timestamp",
                return_value="18:06:12",
            ):
                view.append_conversation_record(
                    source="Mic",
                    channel="self",
                    source_text="ありがとう",
                    translated_text="고마워",
                    origin_wall_clock_ms=1712345678901,
                )
            with patch.object(logs_module, "source_label", return_value="Mic"):
                view._on_conversation_button_click(SimpleNamespace())
                assert view._log_text.value == "[18:06:12] Mic\nありがとう\n고마워"
            with patch.object(logs_module, "source_label", return_value="마이크"):
                view.apply_locale()

        assert view._log_text.value == "[18:06:12] 마이크\nありがとう\n고마워"
        record = view.conversation_records[0]
        assert record == logs_module.ConversationRecord(
            timestamp_label="18:06:12",
            source="Mic",
            channel="self",
            source_text="ありがとう",
            translated_text="고마워",
        )

    def test_conversation_records_returns_read_only_snapshot(self):
        view = LogsView()

        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            with patch.object(
                logs_module,
                "_format_conversation_timestamp",
                return_value="18:06:12",
            ):
                view.append_conversation_record(
                    source="Mic",
                    channel="self",
                    source_text="ありがとう",
                    translated_text="고마워",
                    origin_wall_clock_ms=1712345678901,
                )

        records = view.conversation_records
        assert isinstance(records, tuple)
        with pytest.raises(AttributeError):
            records.append(  # type: ignore[attr-defined]
                logs_module.ConversationRecord(
                    "18:06:13",
                    "Mic",
                    "self",
                    "bypass source",
                    "bypass translation",
                )
            )
        assert len(view.conversation_records) == 1

    def test_append_conversation_record_rerenders_active_conversation_view(self):
        view = LogsView()

        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            with patch.object(
                logs_module,
                "_format_conversation_timestamp",
                side_effect=["18:06:12", "18:06:13"],
            ):
                view._on_conversation_button_click(SimpleNamespace())
                assert view._log_text.value == logs_module.t("logs.conversation.empty")

                view.append_conversation_record(
                    source="Mic",
                    channel="self",
                    source_text="最初",
                    translated_text="처음",
                    origin_wall_clock_ms=1712345678901,
                )
                assert "最初" in view._log_text.value
                assert "처음" in view._log_text.value
                assert logs_module.t("logs.conversation.empty") not in view._log_text.value

                view.append_conversation_record(
                    source="Mic",
                    channel="self",
                    source_text="次",
                    translated_text="다음",
                    origin_wall_clock_ms=1712345679901,
                )

        assert "最初" in view._log_text.value
        assert "처음" in view._log_text.value
        assert "次" in view._log_text.value
        assert "다음" in view._log_text.value

    def test_append_conversation_record_skips_blank_original_or_translation(self):
        view = LogsView()

        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            with patch.object(logs_module, "_format_conversation_timestamp") as timestamp:
                view.append_conversation_record(
                    source="Mic",
                    channel="self",
                    source_text="   ",
                    translated_text="번역",
                    origin_wall_clock_ms=1712345678901,
                )
                view.append_conversation_record(
                    source="Mic",
                    channel="self",
                    source_text="원문",
                    translated_text="\n\t ",
                    origin_wall_clock_ms=1712345679901,
                )

        assert view.conversation_records == ()
        timestamp.assert_not_called()

    def test_format_conversation_timestamp_formats_origin_wall_clock_ms(self):
        seen: dict[str, float | str] = {}

        class FakeTimestamp:
            def strftime(self, fmt: str) -> str:
                seen["fmt"] = fmt
                return "18:06:12"

        class FakeDatetime:
            @staticmethod
            def fromtimestamp(timestamp_s: float) -> FakeTimestamp:
                seen["timestamp_s"] = timestamp_s
                return FakeTimestamp()

        with patch.object(logs_module, "datetime", FakeDatetime):
            formatted = logs_module._format_conversation_timestamp(1712345678901)

        assert formatted == "18:06:12"
        assert seen == {"timestamp_s": 1712345678.901, "fmt": "%H:%M:%S"}

    def test_conversation_record_channel_is_stored_without_rendering(self):
        view = LogsView()

        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            with patch.object(
                logs_module,
                "_format_conversation_timestamp",
                return_value="18:06:12",
            ):
                view.append_conversation_record(
                    source="Mic",
                    channel="future-channel",
                    source_text="こんにちは",
                    translated_text="안녕",
                    origin_wall_clock_ms=1712345678901,
                )
            view._on_conversation_button_click(SimpleNamespace())

        assert view.conversation_records[0].channel == "future-channel"
        assert "future-channel" not in view._log_text.value
        assert "こんにちは" in view._log_text.value
        assert "안녕" in view._log_text.value

    def test_system_log_flush_does_not_mix_into_active_conversation_view(self):
        view = LogsView()

        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            with patch.object(
                logs_module,
                "_format_conversation_timestamp",
                return_value="18:06:12",
            ):
                view.append_conversation_record(
                    source="Mic",
                    channel="self",
                    source_text="あああ",
                    translated_text="아아아",
                    origin_wall_clock_ms=1712345678901,
                )
            view._on_conversation_button_click(SimpleNamespace())
            conversation_text = view._log_text.value

            view.append_log("system while conversation visible")

            assert view._log_text.value == conversation_text
            view._on_conversation_button_click(SimpleNamespace())

        assert view._log_text.value == "system while conversation visible"

    def test_conversation_model_cleanup_removes_oldest_records_first(self):
        model = logs_module.ConversationViewModel(max_entries=2, cleanup_batch=1)

        model.append(
            logs_module.ConversationRecord("18:06:01", "Mic", "self", "source 1", "translated 1")
        )
        model.append(
            logs_module.ConversationRecord("18:06:02", "Mic", "self", "source 2", "translated 2")
        )
        model.append(
            logs_module.ConversationRecord("18:06:03", "Mic", "self", "source 3", "translated 3")
        )
        model.append(
            logs_module.ConversationRecord("18:06:04", "Mic", "self", "source 4", "translated 4")
        )

        assert [record.source_text for record in model.records] == [
            "source 2",
            "source 3",
            "source 4",
        ]
        assert model.cleanup_count == 1

    def test_logs_view_mode_button_toggles_mode_and_notifies_listener(self):
        view = LogsView()
        seen: list[str] = []

        view.on_mode_change = lambda mode: seen.append(mode)

        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            view._on_mode_button_click(SimpleNamespace())

        assert view.runtime_logging_mode == "detailed"
        assert view._mode_button.content == logs_module.t("logs.mode.detailed")
        assert seen == ["detailed"]

        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            view._on_mode_button_click(SimpleNamespace())

        assert view.runtime_logging_mode == "basic"
        assert view._mode_button.content == logs_module.t("logs.mode.basic")
        assert seen == ["detailed", "basic"]

    def test_logs_view_preserves_existing_lines_when_switching_back_to_basic(self):
        model = LiveLogViewModel()

        model.append("[DETAILED] line")
        model.append("basic line")

        assert model.visible_lines[-2:] == ["[DETAILED] line", "basic line"]

        view = LogsView()
        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            view.set_runtime_logging_mode("detailed")
            view.append_log("[DETAILED] before off")
            view.set_runtime_logging_mode("basic")
            view.append_log("basic after off")
            view._flush_logs()

        assert view._log_text.value == "[DETAILED] before off\nbasic after off"

    def test_append_log_adds_entry(self):
        """로그 항목이 정상적으로 추가되는지 확인"""
        view = LogsView()
        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            view.append_log("test message")
        assert len(view.log_list.controls) == 1

    def test_batch_cleanup_triggers_at_threshold(self):
        """4500개 초과 시 500개 배치 삭제 확인"""
        view = LogsView()
        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            for i in range(MAX_LOG_ENTRIES + CLEANUP_BATCH + 1):
                view.append_log(f"log {i}")
        assert len(view.log_list.controls) == MAX_LOG_ENTRIES + 1

    def test_no_cleanup_under_threshold(self):
        """4500개 이하면 삭제 안 함"""
        view = LogsView()
        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            for i in range(MAX_LOG_ENTRIES + CLEANUP_BATCH):
                view.append_log(f"log {i}")
        assert len(view.log_list.controls) == MAX_LOG_ENTRIES + CLEANUP_BATCH

    def test_oldest_entries_removed_first(self):
        """오래된 항목부터 삭제되는지 확인"""
        view = LogsView()
        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            for i in range(MAX_LOG_ENTRIES + CLEANUP_BATCH + 1):
                view.append_log(f"log {i}")
        # 첫 번째 남은 항목이 "log 500"이어야 함
        first_text = view.log_list.controls[0].value
        assert "log 500" in first_text

    def test_flush_logs_appends_only_new_lines_when_buffer_only_grows(self):
        view = LogsView()
        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            view._model.append("line 1")
            view._flush_logs()

            view._model.append("line 2")
            view._flush_logs()

        assert view._log_text.value == "line 1\nline 2"
        assert view._rendered_line_count == 2
        assert view._last_cleanup_count == 0

    def test_flush_logs_rebuilds_after_cleanup_compaction(self):
        view = LogsView()
        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            for i in range(MAX_LOG_ENTRIES):
                view._model.append(f"log {i}")
            view._flush_logs()

            for i in range(MAX_LOG_ENTRIES, MAX_LOG_ENTRIES + CLEANUP_BATCH + 1):
                view._model.append(f"log {i}")
            view._flush_logs()

        assert view._last_cleanup_count == 1
        assert view._rendered_line_count == len(view._model.visible_lines)
        assert view._log_text.value.splitlines()[0] == "log 500"

    def test_apply_locale_updates_title_and_folder_text(self):
        view = LogsView()
        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=None):
            view.set_runtime_logging_mode("detailed")
            view.apply_locale()
        assert view._title_text.value == logs_module.t("logs.title")
        assert view._folder_button.content == logs_module.t("logs.open_folder")
        assert view._mode_button.content == logs_module.t("logs.mode.detailed")
        assert view._mode_button.icon == logs_module.ft.Icons.ARTICLE

    def test_open_log_folder_uses_platform_specific_launcher(self):
        view = LogsView()
        commands = []
        log_dir = logs_module.Path("/tmp/logs")

        def fake_popen(cmd):
            commands.append(cmd)

        with (
            patch.object(type(view), "page", new_callable=PropertyMock, return_value=object()),
            patch.object(logs_module, "_get_log_dir", return_value=log_dir),
            patch.object(logs_module.subprocess, "Popen", side_effect=fake_popen),
            patch.object(logs_module.sys, "platform", "linux"),
        ):
            view._open_log_folder(None)

        assert commands == [["xdg-open", str(log_dir)]]

    def test_open_log_folder_windows_and_macos(self):
        view = LogsView()
        commands = []
        log_dir = logs_module.Path("/tmp/logs")

        def fake_popen(cmd):
            commands.append(cmd)

        with (
            patch.object(type(view), "page", new_callable=PropertyMock, return_value=object()),
            patch.object(logs_module, "_get_log_dir", return_value=log_dir),
            patch.object(logs_module.subprocess, "Popen", side_effect=fake_popen),
            patch.object(logs_module.sys, "platform", "win32"),
        ):
            view._open_log_folder(None)

        with (
            patch.object(type(view), "page", new_callable=PropertyMock, return_value=object()),
            patch.object(logs_module, "_get_log_dir", return_value=log_dir),
            patch.object(logs_module.subprocess, "Popen", side_effect=fake_popen),
            patch.object(logs_module.sys, "platform", "darwin"),
        ):
            view._open_log_folder(None)

        assert commands == [["explorer", str(log_dir)], ["open", str(log_dir)]]

    def test_get_log_dir_delegates_to_user_config_dir(self):
        fake_paths = type(
            "FakePaths", (), {"user_config_dir": staticmethod(lambda: Path("/tmp/cfg"))}
        )
        with patch.dict("sys.modules", {"puripuly_heart.config.paths": fake_paths}):
            assert _get_log_dir() == Path("/tmp/cfg")

    def test_flet_log_handler_emit_success_and_error_path(self):
        class GoodView:
            def __init__(self):
                self.lines = []

            def append_log(self, line: str) -> None:
                self.lines.append(line)

        good = GoodView()
        handler = FletLogHandler(good)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        assert good.lines and "hello" in good.lines[0]

        class BadView:
            def append_log(self, _line: str) -> None:
                raise RuntimeError("fail")

        FletLogHandler(BadView()).emit(record)

    def test_flet_log_handler_marshals_worker_thread_updates_to_page_loop(self):
        async def scenario() -> None:
            view = LogsView()
            ui_thread_id = threading.get_ident()
            append_completed = threading.Event()
            seen: dict[str, int] = {}

            class FakePage:
                def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
                    self.loop = loop

            def fake_append_log(_line: str) -> None:
                seen["thread_id"] = threading.get_ident()
                append_completed.set()

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="hello from worker",
                args=(),
                exc_info=None,
            )

            with patch.object(
                type(view),
                "page",
                new_callable=PropertyMock,
                return_value=FakePage(asyncio.get_running_loop()),
            ):
                with patch.object(view, "append_log", side_effect=fake_append_log):
                    handler = FletLogHandler(view)
                    worker = threading.Thread(target=handler.emit, args=(record,))
                    worker.start()
                    await asyncio.to_thread(append_completed.wait, 1)
                    worker.join(timeout=1)

            assert append_completed.is_set()
            assert seen["thread_id"] == ui_thread_id

        asyncio.run(scenario())

    def test_flet_log_handler_delivers_same_loop_updates_immediately(self):
        async def scenario() -> None:
            view = LogsView()

            class FakePage:
                def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
                    self.loop = loop

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="hello on ui loop",
                args=(),
                exc_info=None,
            )

            with patch.object(
                type(view),
                "page",
                new_callable=PropertyMock,
                return_value=FakePage(asyncio.get_running_loop()),
            ):
                with patch.object(view, "append_log") as append_log:
                    FletLogHandler(view).emit(record)

            append_log.assert_called_once()
            assert view._model.visible_lines == []
            assert view._pending_update is False

        asyncio.run(scenario())

    def test_append_log_threadsafe_buffers_when_page_loop_dispatch_fails(self):
        view = LogsView()

        with patch.object(
            type(view),
            "page",
            new_callable=PropertyMock,
            return_value=SimpleNamespace(loop=object()),
        ):
            with patch.object(
                logs_module.asyncio,
                "run_coroutine_threadsafe",
                side_effect=RuntimeError("loop closed"),
            ):
                with patch.object(view, "append_log") as append_log:
                    view.append_log_threadsafe("buffered line")

        append_log.assert_not_called()
        assert view._model.visible_lines == ["buffered line"]
        assert view._pending_update is True

    def test_attach_log_handler_idempotent(self):
        view = LogsView()
        added = []

        class DummyLogger:
            def addHandler(self, handler):
                added.append(handler)

        with patch.object(logs_module.logging, "getLogger", return_value=DummyLogger()):
            view.attach_log_handler()
            view.attach_log_handler()

        assert len(added) == 1

    @patch("time.time", side_effect=[0.0, 0.3, 0.4])
    def test_scroll_to_bottom_flushes_pending_and_awaits_scroll(self, _mock_time):
        view = LogsView()
        scrolled = {"called": False}

        async def fake_scroll_to(**_kwargs):
            scrolled["called"] = True

        view._log_scroll.scroll_to = fake_scroll_to
        with patch.object(type(view), "page", new_callable=PropertyMock, return_value=object()):
            with patch.object(type(view._log_text), "update", lambda self: None):
                view._pending_update = True
                view._log_buffer.append("line")
                asyncio.run(view.scroll_to_bottom())

        assert scrolled["called"] is True
