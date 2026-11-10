from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .image import PreparedPhoto

PHOTO_FAILURE_CODES = frozenset(
    {
        "photo_authentication_failed",
        "photo_quota_exhausted",
        "photo_provider_unavailable",
        "photo_request_rejected",
        "photo_usage_limit",
        "photo_stream_incomplete",
        "photo_stream_invalid",
        "photo_content_rejected",
        "photo_timeout",
        "photo_transport_failed",
        "photo_connection_closed",
        "photo_request_in_progress",
        "photo_provider_failed",
        "photo_route_changed",
        "photo_send_contract",
        "photo_stream_contract",
        "photo_incomplete",
        "photo_parse_failed",
        "photo_no_usable_blocks",
        "photo_request_failed",
        "photo_preparation_failed",
    }
)


@dataclass(frozen=True, slots=True)
class PhotoRequestSent:
    monotonic_time: float


@dataclass(frozen=True, slots=True)
class PhotoTextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class PhotoStreamCompleted:
    usage: dict
    model: str


class PhotoProviderError(Exception):
    def __init__(self, code: str, *, retryable: bool = True, quota: bool = False) -> None:
        self.code = code if code in PHOTO_FAILURE_CODES else "photo_provider_failed"
        self.retryable = retryable
        self.quota = quota
        super().__init__(self.code)


class PhotoProviderConnection(Protocol):
    model: str
    google: bool

    connect_ms: float

    async def preopen(self) -> None: ...

    def stream(
        self, image: PreparedPhoto, target_language: str
    ) -> AsyncIterator[PhotoRequestSent | PhotoTextDelta | PhotoStreamCompleted]: ...

    async def close(self) -> None: ...


PhotoStreamEvent = PhotoRequestSent | PhotoTextDelta | PhotoStreamCompleted
