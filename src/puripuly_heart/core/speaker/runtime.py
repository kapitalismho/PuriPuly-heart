from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import CaptureStreamInput
from puripuly_heart.core.speaker.contracts import (
    BackendReady,
    BackendUpdate,
    RuntimeLimits,
    SourceMapPiece,
    SpeakerBackend,
    SpeakerEvidence,
    SpeakerSnapshot,
)


@dataclass(slots=True)
class _Block:
    pcm: np.ndarray
    capture: tuple[AudioCaptureSpan, ...]
    synthetic: bool = False
    pause: bool = False


class SpeakerRuntime:
    def __init__(
        self,
        backend: SpeakerBackend,
        *,
        limits: RuntimeLimits | None = None,
        onset_context_samples: int = 8000,
        short_gap_samples: int = 8000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend = backend
        self._limits = limits or RuntimeLimits()
        if not 0 <= onset_context_samples <= self._limits.max_pending_samples:
            raise ValueError("onset context exceeds PCM capacity")
        if not 0 <= short_gap_samples <= self._limits.max_backend_lag_samples:
            raise ValueError("short gap exceeds mapping capacity")
        self._onset_limit = onset_context_samples
        self._short_gap = short_gap_samples
        self._clock = clock
        self._state = "new"
        self._ready: BackendReady | None = None
        self._epoch = 0
        self._session: str | None = None
        self._sessions: set[str] = set()
        self._capture_epoch: int | None = None
        self._source_end: int | None = None
        self._source_eof = False
        self._graceful_eof = False
        self._queue: deque[_Block] = deque()
        self._ring: deque[_Block] = deque()
        self._ring_samples = 0
        self._queued_samples = 0
        self._inflight_samples = 0
        self._inflight_pieces = 0
        self._evidence_map_pieces = 0
        self._maps: deque[SourceMapPiece] = deque()
        self._evidence: deque[SpeakerEvidence] = deque()
        self._wake = asyncio.Event()
        self._startup: asyncio.Task[None] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._shutdown: asyncio.Task[None] | None = None
        self._active = False
        self._silence_samples = 0
        self._compact = 0
        self._processed = 0
        self._immutable = 0
        self._backend_epoch = -1
        self._error: str | None = None
        self._evictions = 0
        self._overloads = 0
        self._stale = 0
        self._copied = 0
        self._max_copy = 0

    async def start(self) -> None:
        if self._state == "new":
            self._state = "starting"
            self._startup = asyncio.create_task(self._load())
        if self._startup is not None:
            await asyncio.shield(self._startup)

    async def _load(self) -> None:
        try:
            ready = await self._backend.start()
            if not 0 <= ready.tail_padding_samples <= self._limits.max_tail_padding_samples:
                raise ValueError("backend tail padding exceeds capacity")
            if not 0 <= ready.max_revision_samples <= self._limits.max_backend_lag_samples:
                raise ValueError("backend pending/revision horizon exceeds capacity")
            if len(ready.diagnostics) > 64 or any(
                len(key) > 128 or len(value) > 2048 for key, value in ready.diagnostics
            ):
                raise ValueError("backend diagnostics exceed capacity")
            if self._state != "starting":
                return
            self._ready = ready
            self._state = "ready"
            self._worker = asyncio.create_task(self._run())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail("backend_start_failed", exc)

    def reset(self, reason: str = "manual_reset") -> None:
        if self._state in {"stopping", "cleanup_pending", "stopped"}:
            return
        detach = self._graceful_eof
        self._fence(reason)
        if detach:
            self._detach()

    def _fence(self, reason: str) -> None:
        discarded = tuple(self._maps) + self._unobserved(
            tuple(span for block in self._queue for span in block.capture)
            + tuple(span for block in self._ring for span in block.capture)
        )
        self._emit(reason=reason, source_map=discarded[: self._limits.max_map_pieces])
        self._epoch += 1
        self._graceful_eof = False
        self._queue.clear()
        self._ring.clear()
        self._maps.clear()
        self._queued_samples = 0
        self._ring_samples = 0
        self._compact = self._processed = self._immutable = 0
        self._active = False
        self._silence_samples = 0
        self._wake.set()

    def end_capture(self, capture_session_id: str, reason: str) -> None:
        if capture_session_id != self._session:
            self._stale += 1
            return
        if reason == "source_eof" and self._state == "ready":
            if self._graceful_eof:
                self._stale += 1
                return
            self._tail()
            self._source_eof = True
            self._graceful_eof = True
            self._wake.set()
            return
        self._fence(reason)
        self._detach()

    def _detach(self) -> None:
        self._session = None
        self._capture_epoch = None
        self._source_end = None
        self._source_eof = False
        self._graceful_eof = False

    def _finish_eof(self) -> None:
        if not self._graceful_eof or self._queue:
            return
        for block in self._ring:
            self._emit(reason="omitted_silence", source_map=self._unobserved(block.capture))
        self._ring.clear()
        self._ring_samples = 0
        if any(piece.capture is not None for piece in self._maps):
            self._emit(reason="source_eof_pending_coverage", source_map=tuple(self._maps))
        self._emit(reason="source_eof")
        self._maps.clear()
        self._detach()

    def observe_input(self, capture_session_id: str, event: CaptureStreamInput) -> None:
        if self._state != "ready":
            return
        if (
            self._graceful_eof
            and capture_session_id != self._session
            and capture_session_id not in self._sessions
        ):
            if not capture_session_id or len(capture_session_id) > 256:
                self._error = "invalid_capture_session_id"
                return
            if len(self._sessions) >= self._limits.max_capture_sessions:
                self._error = "capture_session_capacity_exhausted"
                return
            self._fence("capture_replaced")
            self._detach()
        if self._session is None:
            if capture_session_id in self._sessions:
                self._stale += 1
                return
            if len(self._sessions) >= self._limits.max_capture_sessions:
                self._error = "capture_session_capacity_exhausted"
                return
            if not capture_session_id or len(capture_session_id) > 256:
                self._error = "invalid_capture_session_id"
                return
            self._sessions.add(capture_session_id)
            self._session = capture_session_id
            self._epoch += 1
            self._compact = self._processed = self._immutable = 0
            self._wake.set()
        elif capture_session_id != self._session or self._source_eof:
            self._stale += 1
            return
        if event.chunk.size == 0:
            if event.boundary_reason == "source_discontinuity":
                self._fence("source_discontinuity")
                self._source_end = None
                self._capture_epoch = None
            if event.boundary_reason == "source_eof":
                self._tail()
                self._source_eof = True
            return
        try:
            capture = self._validate(event)
        except ValueError as exc:
            self._overload(event.capture, "invalid_original_input", str(exc))
            return
        if any(span.discontinuity_before is not None for span in capture[1:]):
            offset = 0
            for index, span in enumerate(capture):
                size = span.normalized_sample_count
                self.observe_input(
                    capture_session_id,
                    CaptureStreamInput(
                        event.chunk[offset : offset + size],
                        (span,),
                        event.boundary_reason if index == len(capture) - 1 else None,
                        event.speech_observed,
                    ),
                )
                offset += size
            return
        first, last = capture[0], capture[-1]
        start = first.normalized_start_sample
        end = last.normalized_end_sample
        assert start is not None and end is not None
        if self._capture_epoch is not None and first.capture_epoch < self._capture_epoch:
            self._stale += 1
            return
        if self._capture_epoch == first.capture_epoch and self._source_end is not None:
            if end <= self._source_end:
                self._stale += 1
                return
            if start < self._source_end:
                self._overload(capture, "overlapping_original_input")
                self._source_end = end
                return
        discontinuity = event.boundary_reason == "source_discontinuity" or any(
            span.discontinuity_before is not None for span in capture
        )
        if self._source_end is not None and (
            self._capture_epoch != first.capture_epoch or start != self._source_end or discontinuity
        ):
            self._fence("source_coverage_gap")
        elif discontinuity:
            self._fence("source_discontinuity")
        self._capture_epoch = first.capture_epoch
        self._source_end = end
        if event.speech_observed:
            if not self._active:
                while self._ring:
                    block = self._ring.popleft()
                    self._ring_samples -= block.pcm.size
                    if not self._enqueue(block):
                        self._overload(capture, "input_overload")
                        return
                self._active = True
            self._silence_samples = 0
            if not self._enqueue(self._block(event.chunk, capture)):
                self._overload(capture, "input_overload")
        else:
            offset = 0
            if self._active:
                assert self._ready is not None
                keep = max(self._short_gap, self._ready.tail_padding_samples)
                count = min(event.chunk.size, max(0, keep - self._silence_samples))
                if count:
                    spans = self._slice(capture, start, start + count)
                    if not self._enqueue(self._block(event.chunk[:count], spans)):
                        self._overload(capture, "input_overload")
                        return
                    self._silence_samples += count
                    offset = count
                if self._silence_samples >= keep:
                    self._tail()
            if offset < event.chunk.size:
                spans = self._slice(capture, start + offset, end)
                self._retain_onset(self._block(event.chunk[offset:], spans))
        if event.boundary_reason == "source_eof":
            self._tail()
            self._source_eof = True

    def _validate(self, event: CaptureStreamInput) -> tuple[AudioCaptureSpan, ...]:
        if event.chunk.ndim != 1 or event.chunk.dtype != np.float32:
            raise ValueError("original PCM must be mono float32")
        if event.chunk.size > self._limits.max_input_samples:
            raise ValueError("original chunk exceeds capacity")
        if not event.capture or len(event.capture) > self._limits.max_map_pieces:
            raise ValueError("original source mapping exceeds capacity")
        expected = event.capture[0].normalized_start_sample
        epoch = event.capture[0].capture_epoch
        total = 0
        for span in event.capture:
            if (
                span.capture_epoch != epoch
                or span.normalized_sample_rate_hz != 16_000
                or span.normalized_start_sample != expected
                or span.normalized_end_sample is None
            ):
                raise ValueError("original source mapping must be contiguous normalized 16kHz")
            total += span.normalized_sample_count
            expected = span.normalized_end_sample
        if total != event.chunk.size:
            raise ValueError("PCM and source mapping length differ")
        return event.capture

    @staticmethod
    def _slice(
        capture: tuple[AudioCaptureSpan, ...], start: int, end: int
    ) -> tuple[AudioCaptureSpan, ...]:
        result = []
        for span in capture:
            assert span.normalized_start_sample is not None
            assert span.normalized_end_sample is not None
            left = max(start, span.normalized_start_sample)
            right = min(end, span.normalized_end_sample)
            if left < right:
                result.append(span.slice_normalized(left, right))
        return tuple(result)

    def _block(self, pcm: np.ndarray, capture: tuple[AudioCaptureSpan, ...]) -> _Block:
        copied = np.array(pcm, dtype=np.float32, copy=True, order="C")
        copied.flags.writeable = False
        self._copied += copied.size
        self._max_copy = max(self._max_copy, copied.size)
        return _Block(copied, capture)

    def _pieces(self) -> int:
        return (
            len(self._maps)
            + self._inflight_pieces
            + sum(max(1, len(block.capture)) for block in self._queue)
            + sum(len(block.capture) for block in self._ring)
        )

    def _enqueue(self, block: _Block) -> bool:
        if (
            self._queued_samples + self._inflight_samples + self._ring_samples + block.pcm.size
            > self._limits.max_pending_samples
            or len(self._queue) + int(self._inflight_samples > 0) >= self._limits.max_pending_items
            or self._pieces() + max(1, len(block.capture)) > self._limits.max_map_pieces
        ):
            return False
        self._queue.append(block)
        self._queued_samples += block.pcm.size
        self._wake.set()
        return True

    def _retain_onset(self, block: _Block) -> None:
        self._ring.append(block)
        self._ring_samples += block.pcm.size
        while self._ring and (
            self._ring_samples > self._onset_limit
            or self._ring_samples + self._queued_samples + self._inflight_samples
            > self._limits.max_pending_samples
            or self._pieces() > self._limits.max_map_pieces
            or len(self._ring) + len(self._queue) >= self._limits.max_pending_items
        ):
            old = self._ring.popleft()
            excess = max(0, self._ring_samples - self._onset_limit)
            remove = min(old.pcm.size, excess) if excess else old.pcm.size
            self._ring_samples -= old.pcm.size
            first = old.capture[0].normalized_start_sample
            assert first is not None
            self._emit(
                reason="omitted_silence",
                source_map=self._unobserved(self._slice(old.capture, first, first + remove)),
            )
            if remove < old.pcm.size:
                remainder = self._block(
                    old.pcm[remove:],
                    self._slice(old.capture, first + remove, first + old.pcm.size),
                )
                self._ring.appendleft(remainder)
                self._ring_samples += remainder.pcm.size

    def _tail(self) -> None:
        if not self._active:
            return
        assert self._ready is not None
        remaining = self._ready.tail_padding_samples
        while remaining:
            size = min(remaining, self._limits.max_input_samples)
            pcm = np.zeros(size, dtype=np.float32)
            pcm.flags.writeable = False
            if not self._enqueue(_Block(pcm, (), synthetic=True)):
                self._overload((), "tail_overload")
                return
            remaining -= size
        empty = np.empty(0, dtype=np.float32)
        empty.flags.writeable = False
        if not self._enqueue(_Block(empty, (), pause=True)):
            self._overload((), "tail_overload")
            return
        self._active = False
        self._silence_samples = 0

    def _overload(
        self, capture: tuple[AudioCaptureSpan, ...], reason: str, detail: str | None = None
    ) -> None:
        self._overloads += 1
        capture = capture[: self._limits.max_map_pieces]
        lost = (
            tuple(span for block in self._queue for span in block.capture)
            + tuple(span for block in self._ring for span in block.capture)
            + capture
        )
        self._emit(reason=reason, source_map=self._unobserved(lost[: self._limits.max_map_pieces]))
        self._fence(reason)
        if detail is not None:
            self._error = detail[:2048]

    @staticmethod
    def _unobserved(capture: tuple[AudioCaptureSpan, ...]) -> tuple[SourceMapPiece, ...]:
        return tuple(SourceMapPiece(0, 0, span, "unobserved") for span in capture)

    def _emit(
        self,
        *,
        reason: str | None = None,
        source_map: tuple[SourceMapPiece, ...] = (),
        update: BackendUpdate | None = None,
        frame=None,
    ) -> None:
        if self._session is None:
            return
        if reason is not None:
            reason = reason[:2048]
        while self._evidence and (
            len(self._evidence) >= self._limits.max_evidence_items
            or self._evidence_map_pieces + len(source_map) > self._limits.max_evidence_map_pieces
        ):
            self._evidence_map_pieces -= len(self._evidence.popleft().source_map)
            self._evictions += 1
        self._evidence.append(
            SpeakerEvidence(
                self._session,
                self._epoch,
                self._clock(),
                source_map,
                self._processed if update is None else update.processed_through_sample,
                self._immutable if update is None else update.immutable_through_sample,
                frame,
                (
                    "closed"
                    if frame is None
                    or frame.end_sample
                    <= (self._immutable if update is None else update.immutable_through_sample)
                    else "open"
                ),
                reason,
            )
        )
        self._evidence_map_pieces += len(source_map)

    async def _run(self) -> None:
        try:
            while self._state == "ready":
                await self._wake.wait()
                self._wake.clear()
                if self._state != "ready":
                    break
                if self._backend_epoch != self._epoch:
                    epoch = self._epoch
                    await self._backend.reset(epoch)
                    if epoch != self._epoch:
                        self._wake.set()
                        continue
                    self._backend_epoch = epoch
                while self._queue and self._backend_epoch == self._epoch:
                    epoch = self._epoch
                    block = self._queue.popleft()
                    self._queued_samples -= block.pcm.size
                    self._inflight_samples = block.pcm.size
                    self._inflight_pieces = max(1, len(block.capture))
                    if block.pause:
                        update = await self._backend.pause()
                    else:
                        start = self._compact
                        cursor = start
                        for span in block.capture:
                            right = cursor + span.normalized_sample_count
                            self._maps.append(SourceMapPiece(cursor, right, span, "original"))
                            cursor = right
                        if block.synthetic:
                            self._maps.append(
                                SourceMapPiece(
                                    start, start + block.pcm.size, None, "synthetic_padding"
                                )
                            )
                        self._compact += block.pcm.size
                        self._inflight_pieces = 0
                        update = await self._backend.feed(block.pcm, start)
                    self._inflight_samples = 0
                    self._inflight_pieces = 0
                    if epoch != self._epoch or self._state != "ready":
                        self._wake.set()
                        break
                    self._accept(update)
                    if block.pause:
                        self._emit(reason="vad_pause")
                if self._state == "ready" and self._backend_epoch == self._epoch:
                    self._finish_eof()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail("backend_work_failed", exc)
        finally:
            self._inflight_samples = 0
            self._inflight_pieces = 0

    def _mapped(self, start: int, end: int) -> tuple[SourceMapPiece, ...]:
        pieces = []
        covered = 0
        for piece in self._maps:
            left = max(start, piece.compact_start_sample)
            right = min(end, piece.compact_end_sample)
            if left >= right:
                continue
            capture = piece.capture
            if capture is not None:
                assert capture.normalized_start_sample is not None
                offset = capture.normalized_start_sample - piece.compact_start_sample
                capture = capture.slice_normalized(left + offset, right + offset)
            pieces.append(SourceMapPiece(left, right, capture, piece.kind))
            covered += right - left
        if covered != end - start:
            raise ValueError("backend frame has no retained original/synthetic mapping")
        return tuple(pieces)

    def _accept(self, update: BackendUpdate) -> None:
        assert self._ready is not None
        if not (
            self._processed <= update.processed_through_sample <= self._compact
            and self._immutable
            <= update.immutable_through_sample
            <= update.processed_through_sample
            and len(update.frames) <= self._limits.max_frames_per_update
        ):
            raise ValueError("invalid or oversized backend frontiers/update")
        previous_end = self._immutable
        if self._compact - update.immutable_through_sample > self._ready.max_revision_samples:
            raise ValueError("backend exceeded pending/revision mapping horizon")
        for frame in update.frames:
            if not (
                previous_end
                <= frame.start_sample
                < frame.end_sample
                <= update.processed_through_sample
                and len(frame.probabilities) == 8
                and all(math.isfinite(value) and 0 <= value <= 1 for value in frame.probabilities)
                and len(frame.active_slots) <= 8
                and len(set(frame.active_slots)) == len(frame.active_slots)
                and all(0 <= slot < 8 for slot in frame.active_slots)
            ):
                raise ValueError("invalid raw eight-channel probability frame")
            previous_end = frame.end_sample
        for frame in update.frames:
            mapped = self._mapped(frame.start_sample, frame.end_sample)
            self._emit(source_map=mapped, update=update, frame=frame)
        self._processed = update.processed_through_sample
        self._immutable = update.immutable_through_sample
        while self._maps and self._maps[0].compact_end_sample <= self._immutable:
            self._maps.popleft()
        if self._maps and self._maps[0].compact_start_sample < self._immutable:
            piece = self._maps.popleft()
            mapped = self._slice_piece(piece, self._immutable)
            self._maps.appendleft(mapped)

    @staticmethod
    def _slice_piece(piece: SourceMapPiece, start: int) -> SourceMapPiece:
        capture = piece.capture
        if capture is not None:
            assert capture.normalized_start_sample is not None
            assert capture.normalized_end_sample is not None
            capture = capture.slice_normalized(
                capture.normalized_start_sample + start - piece.compact_start_sample,
                capture.normalized_end_sample,
            )
        return SourceMapPiece(start, piece.compact_end_sample, capture, piece.kind)

    def _fail(self, reason: str, exc: Exception) -> None:
        self._error = f"{reason}: {exc}"[:2048]
        self._emit(reason=reason, source_map=tuple(self._maps))
        self._fence(reason)
        self._state = "unavailable"

    def drain_evidence(self, max_items: int) -> tuple[SpeakerEvidence, ...]:
        if max_items < 0:
            raise ValueError("max_items cannot be negative")
        result = tuple(self._evidence.popleft() for _ in range(min(max_items, len(self._evidence))))
        self._evidence_map_pieces -= sum(len(item.source_map) for item in result)
        return result

    def snapshot(self) -> SpeakerSnapshot:
        return SpeakerSnapshot(
            self._state,
            self._state == "ready",
            self._epoch,
            self._session,
            not self._active,
            self._queued_samples + self._inflight_samples,
            len(self._queue) + int(self._inflight_samples > 0),
            self._pieces(),
            self._ring_samples,
            len(self._evidence),
            self._evidence_map_pieces,
            self._evictions,
            self._overloads,
            self._stale,
            self._copied,
            self._max_copy,
            sum(
                task is not None and not task.done()
                for task in (self._startup, self._worker, self._shutdown)
            ),
            self._processed,
            self._immutable,
            self._limits,
            () if self._ready is None else self._ready.diagnostics,
            self._error,
        )

    async def stop(self) -> None:
        if self._shutdown is None or (self._shutdown.done() and self._state == "cleanup_pending"):
            if self._shutdown is None:
                self._state = "stopping"
                self._fence("feature_stopped")
                self._session = None
            else:
                self._state = "stopping"
            self._shutdown = asyncio.create_task(self._close())
            self._shutdown.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
        await asyncio.shield(self._shutdown)

    async def _close(self) -> None:
        try:
            tasks = [task for task in (self._startup, self._worker) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._backend.stop()
        except asyncio.CancelledError:
            self._error = "backend_stop_cancelled"
            self._state = "cleanup_pending"
            raise
        except Exception as exc:
            self._error = f"backend_stop_failed: {exc}"[:2048]
            self._state = "cleanup_pending"
            raise
        else:
            self._state = "stopped"
            if self._error is not None and self._error.startswith(
                ("backend_stop_failed:", "backend_stop_cancelled")
            ):
                self._error = None
        finally:
            self._ready = None
