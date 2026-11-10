from __future__ import annotations

import asyncio
import contextlib
import tempfile
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from puripuly_heart.core.lifecycle import LifecycleScope
from puripuly_heart.core.owned_thread import run_owned_thread_call

from .capture import PhotoCapture
from .image import PhotoImageError, PreparedPhoto
from .parsing import PhotoBlockParser, PhotoParseError, PhotoParseEvent
from .protocol import (
    PhotoProviderConnection,
    PhotoProviderError,
    PhotoRequestSent,
    PhotoStreamCompleted,
    PhotoTextDelta,
)


@dataclass(frozen=True, slots=True)
class PhotoRoute:
    identity: str
    connect: Callable[[str | None], PhotoProviderConnection] = field(repr=False, compare=False)


@dataclass(slots=True)
class _Photo:
    capture: PhotoCapture
    route: PhotoRoute
    photo_id: str = field(default_factory=lambda: str(uuid4()))
    attempt_id: int = 0
    starts: int = 0
    phase: str = "offer"
    visible: bool = False
    failure_code: str | None = None
    image: PreparedPhoto | None = None
    preparations: dict[int, dict] = field(default_factory=dict)
    blocks: list[dict] = field(default_factory=list)
    excluded: dict[int, str] = field(default_factory=dict)
    corrected: set[int] = field(default_factory=set)
    first_block_drawn: bool = False
    accepted_at: float | None = None
    pull: asyncio.Event = field(default_factory=asyncio.Event)
    events: deque[str] = field(default_factory=lambda: deque(maxlen=64))
    task: asyncio.Task | None = None
    directory: tempfile.TemporaryDirectory | None = None


class PhotoLifecycle:
    def __init__(
        self,
        *,
        resolve_route: Callable[[], Awaitable[PhotoRoute | None]],
        target_language: Callable[[], str],
        publish: Callable[[dict], None],
        diagnostic: Callable[[str, dict], None],
        prepare: Callable[..., Awaitable[PreparedPhoto]],
        clock: Callable[[], float] = time.monotonic,
        temporary_root: Path | None = None,
    ) -> None:
        self._resolve_route = resolve_route
        self._target_language = target_language
        self._publish = publish
        self._diagnostic = diagnostic
        self._prepare = prepare
        self._clock = clock
        self._temporary_root = temporary_root
        self._scope = LifecycleScope("PhotoLifecycle")
        self._current: _Photo | None = None
        self._retired: list[_Photo] = []
        self._revision = 0
        self._frozen = False
        self._renderer_generation = 0
        self._draw_eligible: set[tuple[str, int]] = set()

    @property
    def request_starts(self) -> int:
        return self._current.starts if self._current else 0

    @property
    def snapshot(self) -> dict:
        photo = self._current
        value = None
        if photo is not None:
            capture = photo.capture
            image = photo.image
            value = {
                "photo_id": photo.photo_id,
                "path": str(capture.path),
                "expected_file_id": [str(item) for item in capture.file_id],
                "file_size": capture.file_size,
                "shutter_unix_ms": capture.shutter_unix_ms,
                "attempt_id": photo.attempt_id,
                "phase": photo.phase,
                "offer_open": photo.phase == "offer" and self._age(photo) < 10,
                "visible": photo.visible,
                "image": (
                    None
                    if image is None
                    else {
                        "path": str(image.source_path),
                        "width": image.source_width,
                        "height": image.source_height,
                    }
                ),
                "preparations": [dict(item) for item in photo.preparations.values()],
                "blocks": [dict(item) for item in photo.blocks],
                "excluded_blocks": list(photo.excluded),
                "failure_code": photo.failure_code,
                "first_block_drawn": photo.first_block_drawn,
            }
        return {"type": "photo_snapshot", "revision": self._revision, "photo": value}

    def renderer_changed(self, generation: int) -> None:
        self._renderer_generation = generation
        self._draw_eligible.clear()
        photo = self._current
        if photo is not None and not photo.blocks:
            self._draw_eligible.add((photo.photo_id, photo.attempt_id))

    def stop_ingress(self) -> None:
        self._frozen = True

    def _age(self, photo: _Photo) -> float:
        return self._clock() - photo.capture.observed_monotonic

    def _active(self, photo: _Photo) -> bool:
        return not self._frozen and self._current is photo and self._age(photo) < 600

    def _changed(self) -> None:
        self._revision += 1
        self._publish(self.snapshot)

    async def captured(self, capture: PhotoCapture) -> bool:
        if self._frozen:
            return False
        route = await self._resolve_route()
        if self._frozen:
            return False
        self._retire_current()
        if route is None or self._clock() - capture.observed_monotonic >= 10:
            self._changed()
            return False
        photo = _Photo(capture=capture, route=route)
        self._current = photo
        photo.task = self._scope.create_task(self._run(photo), name=photo.photo_id)
        self._changed()
        return True

    def _retire_current(self) -> None:
        photo = self._current
        self._current = None
        self._draw_eligible.clear()
        if photo is not None:
            if photo.task is not None and not photo.task.done():
                photo.task.cancel()
            self._retired.append(photo)

    async def refresh(self, route: PhotoRoute | None) -> None:
        photo = self._current
        if photo is not None:
            if self._age(photo) >= 600 or route is None or route.identity != photo.route.identity:
                self._retire_current()
                self._changed()
            elif photo.phase == "offer" and self._age(photo) >= 10:
                photo.phase = "expired"
                if photo.task is not None:
                    photo.task.cancel()
                self._changed()
        for retired in tuple(self._retired):
            if retired.task is None or retired.task.done():
                await self._release(retired)
                self._retired.remove(retired)

    def pull(self, photo_id: str, event_id: str) -> bool:
        photo = self._current
        if self._frozen or photo is None or photo.photo_id != photo_id or self._age(photo) >= 600:
            return False
        if not event_id or len(event_id) > 128 or event_id in photo.events:
            return False
        photo.events.append(event_id)
        if photo.phase == "expired" or (photo.attempt_id == 0 and self._age(photo) >= 10):
            return False
        photo.visible = True
        if photo.phase == "offer":
            photo.attempt_id = 1
            photo.phase = "preparing"
            photo.accepted_at = self._clock()
            self._measure(photo, "accepted", route=photo.route.identity)
            photo.pull.set()
            self._draw_eligible.add((photo.photo_id, photo.attempt_id))
        self._changed()
        return True

    def hidden(self, photo_id: str) -> None:
        photo = self._current
        if not self._frozen and photo is not None and photo.photo_id == photo_id:
            photo.visible = False
            self._changed()

    def first_block_drawn(
        self, photo_id: str, attempt_id: int, block_id: int, generation: int
    ) -> dict | None:
        photo = self._current
        if (
            photo is None
            or not self._active(photo)
            or generation != self._renderer_generation
            or photo.photo_id != photo_id
            or photo.attempt_id != attempt_id
            or not any(block["block_id"] == block_id for block in photo.blocks)
            or photo.first_block_drawn
            or (photo_id, attempt_id) not in self._draw_eligible
        ):
            return None
        photo.first_block_drawn = True
        self._draw_eligible.discard((photo_id, attempt_id))
        self._measure(photo, "first_display", block_id=block_id)
        self._changed()
        return {
            "type": "photo_haptic",
            "photo_id": photo_id,
            "attempt_id": attempt_id,
            "block_id": block_id,
        }

    def _measure(self, photo: _Photo, stage: str, **fields: object) -> None:
        self._diagnostic(
            stage,
            {
                "photo_id": photo.photo_id,
                "attempt_id": photo.attempt_id,
                **(
                    {"request_elapsed_ms": (self._clock() - photo.accepted_at) * 1000}
                    if photo.accepted_at is not None
                    else {}
                ),
                **fields,
            },
        )

    def _exclude(self, photo: _Photo, index: int, reason: str) -> None:
        photo.excluded[index] = reason
        photo.preparations.pop(index, None)
        photo.corrected.discard(index)
        self._measure(photo, "block_excluded", block_id=index, failure_code=reason)
        self._changed()

    def _discard_unfinished(self, photo: _Photo) -> None:
        completed = {block["block_id"] for block in photo.blocks}
        for index in tuple(photo.preparations):
            if index not in completed:
                self._exclude(photo, index, "photo_incomplete_block")

    def _accept_parsed(
        self, photo: _Photo, parsed: PhotoParseEvent, google: bool, elapsed: float
    ) -> None:
        if parsed.index in photo.excluded:
            return
        if parsed.kind == "bbox_ready":
            assert photo.image is not None
            try:
                mapped = photo.image.map_box(parsed.bbox, google=google)
            except PhotoImageError as error:
                self._exclude(photo, parsed.index, error.code)
                return
            photo.preparations[parsed.index] = {
                "block_id": parsed.index,
                "bbox": list(mapped.bbox),
                "source": None,
                "source_line_count": None,
            }
            if mapped.corrected:
                photo.corrected.add(parsed.index)
                self._measure(photo, "block_corrected", block_id=parsed.index)
        else:
            if not parsed.source.strip():
                self._exclude(photo, parsed.index, "photo_source_empty")
                return
            preparation = photo.preparations[parsed.index]
            if parsed.kind == "source_ready":
                preparation["source"] = parsed.source
                preparation["source_line_count"] = parsed.source_line_count
            else:
                if not parsed.translation.strip():
                    self._exclude(photo, parsed.index, "photo_translation_empty")
                    return
                photo.blocks.append(
                    {
                        "block_id": parsed.index,
                        "bbox": preparation["bbox"],
                        "source": parsed.source,
                        "translation": parsed.translation,
                    }
                )
                if len(photo.blocks) == 1:
                    self._measure(photo, "first_block", duration_ms=elapsed)
        self._measure(photo, parsed.kind, duration_ms=elapsed, block_id=parsed.index)
        self._changed()

    async def _connection(self, photo: _Photo, model: str | None) -> PhotoProviderConnection:
        route = await self._resolve_route()
        if not self._active(photo) or route is None or route.identity != photo.route.identity:
            raise PhotoProviderError("photo_route_changed", retryable=False)
        return route.connect(model)

    async def _run(self, photo: _Photo) -> None:
        connection: PhotoProviderConnection | None = None
        try:
            for setup in range(2):
                try:
                    connection = await self._connection(photo, None)
                    await asyncio.wait_for(connection.preopen(), timeout=10)
                    if not self._active(photo):
                        return
                    self._measure(photo, "connection", duration_ms=connection.connect_ms)
                    break
                except Exception as exc:
                    if connection is not None:
                        await connection.close()
                        connection = None
                    if setup or isinstance(exc, PhotoProviderError) and not exc.retryable:
                        raise
            if not photo.pull.is_set():
                await asyncio.wait_for(photo.pull.wait(), timeout=max(0, 10 - self._age(photo)))
            if not self._active(photo):
                return
            photo.directory = tempfile.TemporaryDirectory(
                prefix="puripuly-photo-", dir=self._temporary_root
            )
            photo.image = await self._prepare(
                photo.photo_id,
                photo.attempt_id,
                photo.capture.path,
                Path(photo.directory.name),
                expected_file_id=photo.capture.file_id,
            )
            if not self._active(photo):
                return
            self._measure(
                photo,
                "preparation",
                duration_ms=photo.image.preparation_ms,
                image_base64_bytes=len(photo.image.jpeg_base64),
            )
            self._changed()
            target_language = self._target_language()
            for execution in range(2):
                assert connection is not None
                parser = PhotoBlockParser(google=connection.google)
                sent: float | None = None
                completed = False
                first_text = True
                try:
                    route = await self._resolve_route()
                    if route is None or route.identity != photo.route.identity:
                        raise PhotoProviderError("photo_route_changed", retryable=False)
                    async with contextlib.aclosing(
                        connection.stream(photo.image, target_language)
                    ) as stream:
                        async for event in stream:
                            if not self._active(photo):
                                return
                            if isinstance(event, PhotoRequestSent):
                                if sent is not None or photo.starts >= 2:
                                    raise PhotoProviderError("photo_send_contract", retryable=False)
                                sent = event.monotonic_time
                                photo.starts += 1
                                photo.phase = "requesting"
                                self._measure(photo, "request_sent", model=connection.model)
                                self._changed()
                            elif isinstance(event, PhotoTextDelta):
                                if sent is None:
                                    raise PhotoProviderError("photo_send_contract", retryable=False)
                                elapsed = (self._clock() - sent) * 1000
                                if first_text:
                                    self._measure(photo, "ttft", duration_ms=elapsed)
                                    first_text = False
                                for parsed in parser.feed(event.text):
                                    self._accept_parsed(photo, parsed, connection.google, elapsed)
                            elif isinstance(event, PhotoStreamCompleted):
                                if sent is None or completed:
                                    raise PhotoProviderError("photo_send_contract", retryable=False)
                                parser.finish()
                                completed = True
                                self._measure(
                                    photo,
                                    "total",
                                    duration_ms=(self._clock() - sent) * 1000,
                                    model=event.model,
                                    output_blocks=len(photo.blocks),
                                    corrected_blocks=len(photo.corrected),
                                    excluded_blocks=len(photo.excluded),
                                    usage=event.usage,
                                )
                            else:
                                raise PhotoProviderError("photo_stream_contract", retryable=False)
                    if not completed:
                        raise PhotoProviderError("photo_incomplete")
                    if photo.excluded:
                        if not photo.blocks:
                            raise PhotoProviderError("photo_no_usable_blocks", retryable=False)
                        photo.phase = "partial"
                    elif not photo.blocks:
                        photo.phase = "no_text"
                    elif photo.corrected:
                        photo.phase = "completed_corrected"
                    else:
                        photo.phase = "completed"
                    self._measure(
                        photo,
                        "result",
                        outcome=photo.phase,
                        output_blocks=len(photo.blocks),
                        corrected_blocks=len(photo.corrected),
                        excluded_blocks=len(photo.excluded),
                    )
                    self._changed()
                    return
                except Exception as exc:
                    error = (
                        exc
                        if isinstance(exc, PhotoProviderError)
                        else PhotoProviderError(
                            (
                                "photo_parse_failed"
                                if isinstance(exc, (PhotoParseError, ValueError))
                                else "photo_request_failed"
                            ),
                            retryable=not isinstance(exc, (PhotoParseError, ValueError)),
                        )
                    )
                    self._measure(photo, "request_failed", failure_code=error.code)
                    if not self._active(photo):
                        return
                    if (
                        not error.retryable
                        or execution
                        or photo.starts >= 2
                        or sent is not None
                        and not connection.google
                    ):
                        raise error from None
                    model = (
                        "gemini-3.1-flash-lite"
                        if error.quota
                        and connection.google
                        and connection.model == "gemini-3.5-flash-lite"
                        and not photo.blocks
                        else connection.model
                    )
                    await connection.close()
                    connection = None
                    if not self._active(photo):
                        return
                    if sent is not None:
                        photo.attempt_id += 1
                    photo.preparations.clear()
                    photo.blocks.clear()
                    photo.excluded.clear()
                    photo.corrected.clear()
                    photo.first_block_drawn = False
                    photo.phase = "preparing"
                    self._draw_eligible = {(photo.photo_id, photo.attempt_id)}
                    self._changed()
                    connection = await self._connection(photo, model)
                    await asyncio.wait_for(connection.preopen(), timeout=10)
                    self._measure(photo, "connection", duration_ms=connection.connect_ms)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            if self._active(photo):
                if photo.attempt_id == 0:
                    photo.phase = "expired"
                else:
                    photo.phase = "failed"
                    photo.failure_code = "photo_timeout"
                    self._discard_unfinished(photo)
                    self._measure(photo, "failed", failure_code=photo.failure_code)
                self._changed()
        except Exception as exc:
            if self._active(photo):
                photo.phase = "failed"
                photo.failure_code = (
                    exc.code if isinstance(exc, PhotoProviderError) else "photo_preparation_failed"
                )
                self._discard_unfinished(photo)
                self._measure(photo, "failed", failure_code=photo.failure_code)
                self._changed()
        finally:
            if connection is not None:
                try:
                    await connection.close()
                except Exception:
                    self._measure(
                        photo, "cleanup_failed", failure_code="photo_connection_close_failed"
                    )

    async def _release(self, photo: _Photo) -> None:
        if photo.directory is not None:
            await run_owned_thread_call(photo.directory.cleanup)
            photo.directory = None

    async def close(self) -> None:
        self.stop_ingress()
        self._retire_current()
        self._changed()
        await self._scope.close()
        for photo in tuple(self._retired):
            await self._release(photo)
            self._retired.remove(photo)
