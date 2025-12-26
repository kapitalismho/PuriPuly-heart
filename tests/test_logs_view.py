"""Tests for LogsView batch deletion optimization."""

from unittest.mock import PropertyMock, patch

from puripuly_heart.ui.views.logs import CLEANUP_BATCH, MAX_LOG_ENTRIES, LogsView


class TestLogsView:
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
