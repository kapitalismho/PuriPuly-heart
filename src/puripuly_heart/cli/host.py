from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from puripuly_heart.cli.transport import ControlServer
from puripuly_heart.core.diagnostics import (
    emit_failure,
    exception_diagnostics,
    serialize_diagnostics,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.owned_thread import run_owned_thread_call

logger = logging.getLogger(__name__)


def _record_failure(exception: BaseException, operation: str) -> ErrorDiagnostics:
    diagnostics = getattr(exception, "failure_diagnostics", None)
    if isinstance(diagnostics, ErrorDiagnostics):
        return diagnostics
    diagnostics = exception_diagnostics(
        exception, component="control.host", operation=operation, code="control_host_failed"
    )
    exception.failure_diagnostics = diagnostics
    emit_failure(
        logger, diagnostics,
        severity="info" if isinstance(exception, asyncio.CancelledError) else "error",
        disposition="cancelled" if isinstance(exception, asyncio.CancelledError) else "failed",
    )
    return diagnostics


def _retain_secondary(
    primary: BaseException, secondary: BaseException, operation: str
) -> None:
    diagnostics = getattr(primary, "failure_diagnostics", None)
    if not isinstance(diagnostics, ErrorDiagnostics):
        diagnostics = _record_failure(primary, operation)
    cleanup = getattr(secondary, "failure_diagnostics", None)
    if not isinstance(cleanup, ErrorDiagnostics):
        cleanup = exception_diagnostics(
            secondary, component="control.host", operation=operation, code="control_host_failed"
        )
    cleanup = replace(cleanup, cause_id=diagnostics.diagnostic_id)
    secondary.failure_diagnostics = cleanup
    emit_failure(logger, cleanup, severity="warning", disposition="degraded")
    primary.failure_diagnostics = replace(
        diagnostics, causes=(*diagnostics.causes, cleanup)
    )


class HostedApplication:
    def __init__(self, boundary: Any, lease: Any):
        self.boundary = boundary
        self.lease = lease
        self.instance_id = str(uuid.uuid4())
        self.token = secrets.token_urlsafe(48)
        self.control = boundary.control()
        self.control.bind_instance(self.instance_id)
        self.server = ControlServer(self.control, instance_id=self.instance_id, token=self.token)
        self._endpoint_lock = asyncio.Lock()
        self._closing = False
        from puripuly_heart.app.services.application_shutdown import application_shutdown_callback
        from puripuly_heart.core.lifecycle import (
            SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS,
            SHUTDOWN_PHASE_FREEZE_INGRESS,
        )

        boundary.register_application_shutdown_callbacks(
            (
                application_shutdown_callback(
                    phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
                    owner_name="LocalControlHost",
                    callback_name="freeze_control_ingress",
                    callback=self._freeze_ingress,
                ),
                application_shutdown_callback(
                    phase=SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS,
                    owner_name="LocalControlHost",
                    callback_name="close_control_endpoint",
                    callback=self._close_endpoint,
                ),
            )
        )

    def _freeze_ingress(self) -> None:
        self._closing = True
        self.server.freeze()

    async def _close_endpoint(self) -> None:
        self._freeze_ingress()
        async with self._endpoint_lock:
            primary: BaseException | None = None
            try:
                await self.server.close()
            except BaseException as exc:
                primary = exc
                _record_failure(exc, "endpoint_close")
                raise
            finally:
                try:
                    self.lease.close()
                except BaseException as exc:
                    if primary is None:
                        _record_failure(exc, "endpoint_close")
                        raise
                    _retain_secondary(primary, exc, "endpoint_close")

    async def start(self) -> None:
        try:
            if self._closing:
                raise asyncio.CancelledError
            await self.boundary.start()
            async with self._endpoint_lock:
                if self._closing:
                    raise asyncio.CancelledError
                port = await self.server.start()
                if self._closing:
                    raise asyncio.CancelledError
                await run_owned_thread_call(
                    lambda: self.lease.publish(
                        port=port, token=self.token, instance_id=self.instance_id
                    )
                )
                if self._closing:
                    raise asyncio.CancelledError
        except BaseException as exc:
            _record_failure(exc, "host_start")
            try:
                await self._close_endpoint()
            except BaseException as cleanup:
                _retain_secondary(exc, cleanup, "endpoint_close")
            raise

    async def wait_for_stop(self) -> None:
        await self.control.wait_for_stop_request()
        self._freeze_ingress()
        self.control.freeze_ingress()
        await self.server.wait_for_idle()

    async def close(self) -> None:
        self._freeze_ingress()
        self.control.freeze_ingress()
        primary: BaseException | None = None
        try:
            await self.boundary.stop()
        except BaseException as exc:
            primary = exc
            _record_failure(exc, "host_close")
            raise
        finally:
            try:
                await self._close_endpoint()
            except BaseException as cleanup:
                if primary is None:
                    raise
                _retain_secondary(primary, cleanup, "endpoint_close")


async def run_headless(config_path: Path, *, ready_file: Path | None = None) -> int:
    from puripuly_heart.composition.headless_application import compose_headless_application
    from puripuly_heart.core.control_instance import acquire

    if os.name == "nt":
        from puripuly_heart.core.windows_process_ownership import retain_current_process_job

        retain_current_process_job()

    lease = acquire(config_path)
    host: HostedApplication | None = None
    primary: BaseException | None = None
    try:
        boundary = compose_headless_application(config_path)
        host = HostedApplication(boundary, lease)
        await host.start()
        if ready_file is not None:
            _write_ready(ready_file, {"state": "ready", "instance_id": host.instance_id})
        await host.wait_for_stop()
        return 0
    except BaseException as exc:
        primary = exc
        diagnostics = _record_failure(exc, "host_start")
        if ready_file is not None:
            try:
                _write_ready(
                    ready_file,
                    {"state": "failed", "reason": type(exc).__name__,
                     "diagnostics": serialize_diagnostics(diagnostics)},
                )
            except BaseException as cleanup:
                _retain_secondary(exc, cleanup, "readiness")
        raise
    finally:
        try:
            if host is not None:
                await host.close()
            else:
                lease.close()
        except BaseException as cleanup:
            if primary is None:
                _record_failure(cleanup, "host_close")
                raise
            _retain_secondary(primary, cleanup, "host_close")


def _write_ready(path: Path, status: dict[str, object]) -> None:
    pending = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    pending.write_text(json.dumps(status), encoding="utf-8")
    pending.replace(path)


class GuiControlBoundary:
    def __init__(self, boundary: Any, lease: Any):
        self._boundary = boundary
        self._host = HostedApplication(boundary, lease)
        self._stop_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._boundary, name)

    async def start(self) -> None:
        await self._host.start()
        self._stop_task = asyncio.create_task(self._stop_on_request())

    async def _close_host(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._host.close())
        await asyncio.shield(self._close_task)

    async def _stop_on_request(self) -> None:
        await self._host.wait_for_stop()
        await self._close_host()
        await self._boundary.close_presentation()

    async def stop(self) -> None:
        if self._stop_task is not None and self._close_task is None:
            self._stop_task.cancel()
        await self._close_host()


def gui_control_factory(factory: Any, lease: Any | None = None) -> Any:
    pending = [] if lease is None else [lease]

    def compose(*, config_path: Path, **kwargs: Any) -> GuiControlBoundary:
        from puripuly_heart.core.control_instance import acquire

        lease = pending.pop() if pending else acquire(config_path)
        try:
            boundary = factory(config_path=config_path, **kwargs)
            return GuiControlBoundary(boundary, lease)
        except BaseException as exc:
            _record_failure(exc, "host_start")
            try:
                lease.close()
            except BaseException as cleanup:
                _retain_secondary(exc, cleanup, "endpoint_close")
            raise

    return compose
