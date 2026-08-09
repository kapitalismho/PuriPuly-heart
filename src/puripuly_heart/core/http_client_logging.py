from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager

_HTTP_LOG_SUPPRESSION = contextvars.ContextVar(
    "puripuly_heart_http_log_suppression",
    default=False,
)


def _is_http_logger_name(name: str) -> bool:
    return (
        name == "httpx"
        or name.startswith("httpx.")
        or name == "httpcore"
        or name.startswith("httpcore.")
    )


class _HttpClientLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (_HTTP_LOG_SUPPRESSION.get() and _is_http_logger_name(record.name))


_HTTP_CLIENT_LOG_FILTER = _HttpClientLogFilter()


def _install_http_client_log_filters() -> None:
    logger_objects: list[logging.Logger] = [
        logging.getLogger("httpx"),
        logging.getLogger("httpcore"),
    ]
    for logger_name, logger in logging.Logger.manager.loggerDict.items():
        if _is_http_logger_name(logger_name) and isinstance(logger, logging.Logger):
            logger_objects.append(logger)
    handlers: list[logging.Handler] = list(logging.getLogger().handlers)
    seen_logger_ids: set[int] = set()
    seen_handler_ids: set[int] = {id(handler) for handler in handlers}
    for logger in logger_objects:
        if id(logger) not in seen_logger_ids:
            seen_logger_ids.add(id(logger))
            if not any(item is _HTTP_CLIENT_LOG_FILTER for item in logger.filters):
                logger.addFilter(_HTTP_CLIENT_LOG_FILTER)
        for handler in logger.handlers:
            if id(handler) not in seen_handler_ids:
                seen_handler_ids.add(id(handler))
                handlers.append(handler)
    for handler in handlers:
        if not any(item is _HTTP_CLIENT_LOG_FILTER for item in handler.filters):
            handler.addFilter(_HTTP_CLIENT_LOG_FILTER)


@contextmanager
def suppress_http_client_logs() -> Iterator[None]:
    _install_http_client_log_filters()
    token = _HTTP_LOG_SUPPRESSION.set(True)
    try:
        yield
    finally:
        _HTTP_LOG_SUPPRESSION.reset(token)
