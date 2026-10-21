from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.audio_test import (
    AudioTestCaptureRequest,
    AudioTestRuntimePort,
    AudioTestSourceUnavailable,
)

AudioTestLogSink = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class AudioTestCaptureAdapter:
    log_sink: AudioTestLogSink

    async def capture(
        self,
        request: AudioTestCaptureRequest,
        *,
        runtime: AudioTestRuntimePort,
    ) -> None:
        generation = request.generation
        source: object | None = None
        pending_frame: asyncio.Task[object] | None = None
        end_exception: BaseException | None = None
        frame_count = 0
        try:
            try:
                opened = await request.opener.open(on_acquired=runtime.retain_source)
            except asyncio.CancelledError:
                raise
            except AudioTestSourceUnavailable as exc:
                self._fail(request, exc.reason, exc.cause)
                return
            except Exception as exc:
                self._fail(request, "source_open_failed", type(exc).__name__)
                return

            if not runtime.attach_source(opened.source, generation=generation):
                await runtime.close_source(opened.source)
                return
            source = opened.source
            if request.opened_callback is not None:
                request.opened_callback(generation, dict(opened.device))
            if request.ready_on_open and request.ready_callback is not None:
                request.ready_callback(generation)

            frame_iterator = source.frames()
            pending_frame = runtime.create_frame_task(
                anext(frame_iterator),
                generation=generation,
            )
            while True:
                done, _pending = await asyncio.wait(
                    {pending_frame},
                    timeout=request.frame_timeout_s,
                )
                if not done:
                    raise TimeoutError("no_audio_frames")
                try:
                    frame = pending_frame.result()
                except StopAsyncIteration:
                    pending_frame = None
                    self._fail(request, "capture_ended", None)
                    break

                frame_count += 1
                if frame_count == 1 and not request.ready_on_open:
                    if request.ready_callback is not None:
                        request.ready_callback(generation)
                if request.frame_callback is not None:
                    request.frame_callback(generation, frame)
                pending_frame = runtime.create_frame_task(
                    anext(frame_iterator),
                    generation=generation,
                )
        except asyncio.CancelledError as exc:
            end_exception = exc
            raise
        except TimeoutError as exc:
            end_exception = exc
            self._fail(request, "no_audio_frames", None)
        except Exception as exc:
            end_exception = exc
            self._fail(request, "capture_failed", type(exc).__name__)
        finally:
            cleanup_failures: list[Exception] = []
            source = source or runtime.source
            if pending_frame is not None and not pending_frame.done():
                try:
                    await runtime.cancel_frame_task(pending_frame)
                except Exception as exc:
                    cleanup_failures.append(exc)
            if source is not None:
                try:
                    await runtime.close_source(source)
                except Exception as exc:
                    cleanup_failures.append(exc)
                    if request.failure_callback is not None:
                        request.failure_callback(
                            generation,
                            "source_close_failed",
                            type(exc).__name__,
                        )
                if end_exception is None:
                    self.log_sink(
                        f"[AudioTest] channel={request.channel} completed frames={frame_count}"
                    )
            self._raise_cleanup_failures(cleanup_failures)

    def _fail(self, request: AudioTestCaptureRequest, reason: str, cause: str | None) -> None:
        self.log_sink(
            f"[AudioTest] channel={request.channel} failed reason={reason} cause={cause or 'none'}"
        )
        if request.failure_callback is not None:
            request.failure_callback(request.generation, reason, cause)

    @staticmethod
    def _raise_cleanup_failures(failures: list[Exception]) -> None:
        if not failures:
            return
        if len(failures) == 1:
            raise failures[0]
        raise ExceptionGroup("Audio test capture cleanup failed", failures)


__all__ = ["AudioTestCaptureAdapter", "AudioTestLogSink"]
