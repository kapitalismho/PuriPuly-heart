from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

_GLOBAL_LOGGER_NAMES = ("", "puripuly_heart", "puripuly_heart.runtime.session")


@pytest.fixture(autouse=True)
def _isolated_user_data_dirs(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("user-data")
    for name in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME"):
        path = root / name.lower()
        path.mkdir()
        monkeypatch.setenv(name, str(path))


@pytest.fixture(autouse=True)
def _restore_global_logging_and_locale():
    from puripuly_heart.ui.i18n import get_locale, set_locale

    loggers = [logging.getLogger(name or None) for name in _GLOBAL_LOGGER_NAMES]
    handlers = [tuple(logger.handlers) for logger in loggers]
    levels = [logger.level for logger in loggers]
    locale = get_locale()
    yield
    set_locale(locale)
    for logger, level in zip(loggers, levels, strict=True):
        logger.setLevel(level)
    for logger, original in zip(loggers, handlers, strict=True):
        for handler in tuple(logger.handlers):
            if handler not in original:
                logger.removeHandler(handler)
