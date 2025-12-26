"""Tests for RotatingFileHandler file logging."""

import logging
from logging.handlers import RotatingFileHandler


def test_rotating_file_handler_creates_log_file(tmp_path):
    """로그 파일이 생성되는지 확인"""
    log_file = tmp_path / "test.log"
    handler = RotatingFileHandler(log_file, maxBytes=1024, backupCount=0)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("test_file")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("test message")
    handler.close()

    assert log_file.exists()
    assert "test message" in log_file.read_text()


def test_rotating_handler_with_backup_count_zero(tmp_path):
    """backupCount=0일 때 로그 파일이 계속 기록되는지 확인"""
    log_file = tmp_path / "test.log"
    handler = RotatingFileHandler(log_file, maxBytes=100, backupCount=0)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("test_rotation")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # 여러 메시지 작성
    for i in range(10):
        logger.info(f"message {i:03d}")

    handler.close()

    content = log_file.read_text()
    # 최신 메시지가 파일에 있어야 함
    assert "message 009" in content
