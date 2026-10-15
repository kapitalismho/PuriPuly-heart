from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import socket
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from puripuly_heart.core.local_translation.assets import (
    GEMMA_26B_MODEL_ID,
    GemmaModelSpec,
    default_gemma_install_dir,
    e4b_gemma_spec,
)
from puripuly_heart.core.local_translation.prefix_cache import GemmaPrefixCache
from puripuly_heart.core.local_translation.provisioning import ensure_gemma_installed
from puripuly_heart.core.local_translation.runtime_profile import (
    GEMMA_RUNTIME_PROFILE_VERSION,
    LLAMA_CPP_BUILD,
    LLAMA_CPP_COMMIT,
    EffectiveGemmaBackend,
    GemmaBackend,
    GemmaRuntimePaths,
    build_gemma_server_command,
    default_gemma_runtime_paths,
)

GEMMA_PROMPT_TEMPLATE_VERSION = "translation-prefix-v1"
GEMMA_SLOT_COUNT = 3

_ServerResult = TypeVar("_ServerResult")


async def validate_gemma_runtime_version(executable: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        str(executable),
        "--version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=10.0)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    version = output.decode("utf-8", errors="replace")
    match = re.search(r"\bbuild\s+(\d+),\s+commit\s+([0-9a-f]{9,40})\b", version)
    if (
        process.returncode != 0
        or match is None
        or f"b{match[1]}" != LLAMA_CPP_BUILD
        or not LLAMA_CPP_COMMIT.startswith(match[2])
    ):
        raise ManagedGemmaRuntimeVersionError(
            f"Incompatible llama.cpp executable: {executable}. "
            f"Use {LLAMA_CPP_BUILD} ({LLAMA_CPP_COMMIT[:9]}) or remove "
            "PURIPULY_HEART_LLAMA_CPP_ROOT to use the managed runtime."
        )


class ManagedGemmaRuntimeError(RuntimeError):
    pass


class ManagedGemmaRuntimeClosedError(ManagedGemmaRuntimeError):
    pass


class ManagedGemmaRuntimeVersionError(ManagedGemmaRuntimeError):
    pass


class ManagedGemmaRuntimeProcess(Protocol):
    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


class ManagedGemmaTransport(Protocol):
    async def wait_until_ready(self, *, timeout_s: float) -> None: ...

    async def prepare_prefix(self, *, system_prompt: str, slot_id: int) -> None: ...

    async def wait_slot_idle(self, *, slot_id: int) -> None: ...

    async def restore_prefix(self, *, filename: str, slot_id: int) -> bool: ...

    async def save_prefix(self, *, filename: str, slot_id: int) -> None: ...

    async def translate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        slot_id: int,
        max_output_tokens: int | None = None,
    ) -> ManagedGemmaResponse: ...

    async def close(self) -> None: ...


ManagedGemmaProcessFactory = Callable[
    [tuple[str, ...], Path],
    Awaitable[ManagedGemmaRuntimeProcess],
]
ManagedGemmaTransportFactory = Callable[[str], ManagedGemmaTransport]
GemmaProvisioner = Callable[..., Awaitable[object]]
GemmaRuntimeLogSink = Callable[[str, int], None]


@dataclass(frozen=True, slots=True)
class ManagedGemmaMetrics:
    prompt_tokens: int
    cached_prompt_tokens: int | None
    completion_tokens: int
    prompt_ms: float
    generation_ms: float
    generation_tps: float
    drafted_tokens: int | None = None
    accepted_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ManagedGemmaResponse:
    text: str
    metrics: ManagedGemmaMetrics


@dataclass(frozen=True, slots=True)
class ManagedGemmaReadiness:
    requested_backend: GemmaBackend
    effective_backend: EffectiveGemmaBackend
    source_language: str
    target_language: str
    prefix_identity: str


def _allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


async def _default_process_factory(
    command: tuple[str, ...],
    cwd: Path,
) -> ManagedGemmaRuntimeProcess:
    return await asyncio.create_subprocess_exec(
        command[0],
        *command[1:],
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=cwd,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def build_managed_gemma_system_prompt(
    *,
    system_prompt: str,
    source_language: str,
    target_language: str,
) -> str:
    if "{source_language}" in system_prompt or "{target_language}" in system_prompt:
        return system_prompt.format(
            source_language=source_language,
            target_language=target_language,
        )
    return system_prompt


def _prefix_identity(
    *,
    spec: GemmaModelSpec,
    system_prompt: str,
    source_language: str,
    target_language: str,
    effective_backend: EffectiveGemmaBackend,
    profile_identity: str,
) -> str:
    payload = "\0".join(
        (
            spec.model_id,
            spec.revision,
            LLAMA_CPP_BUILD,
            LLAMA_CPP_COMMIT,
            effective_backend,
            profile_identity,
            GEMMA_RUNTIME_PROFILE_VERSION,
            *(f"{asset.filename}:{asset.sha256}" for asset in spec.assets),
            GEMMA_PROMPT_TEMPLATE_VERSION,
            source_language,
            target_language,
            system_prompt,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ManagedGemmaRuntimeOwner:
    resource_fields = ("process", "transport", "prefix_identity", "active_backend")
    stop_ingress = "reject readiness and translation after close"
    shutdown_policy = "close HTTP transport, terminate server, then kill after bounded grace"
    late_callback_rule = "request-local slots; drain superseded work; reject obsolete results"

    def __init__(
        self,
        *,
        install_dir: Path | None = None,
        runtime_paths: GemmaRuntimePaths | None = None,
        provisioner: GemmaProvisioner = ensure_gemma_installed,
        process_factory: ManagedGemmaProcessFactory = _default_process_factory,
        transport_factory: ManagedGemmaTransportFactory,
        port_allocator: Callable[[], int] = _allocate_loopback_port,
        log_sink: GemmaRuntimeLogSink | None = None,
        startup_timeout_s: float = 120.0,
        shutdown_timeout_s: float = 10.0,
        prefix_cache: GemmaPrefixCache | None = None,
        version_validator: Callable[[Path], Awaitable[None]] = validate_gemma_runtime_version,
    ) -> None:
        self._explicit_install_dir = None if install_dir is None else install_dir.resolve()
        self._install_dir = (self._explicit_install_dir or default_gemma_install_dir()).resolve()
        self._runtime_paths = runtime_paths or default_gemma_runtime_paths()
        self._provisioner = provisioner
        self._process_factory = process_factory
        self._transport_factory = transport_factory
        self._port_allocator = port_allocator
        self._log_sink = log_sink
        self._startup_timeout_s = max(0.1, float(startup_timeout_s))
        self._shutdown_timeout_s = max(0.1, float(shutdown_timeout_s))
        self._prefix_cache = prefix_cache
        self._version_validator = version_validator
        self._process: ManagedGemmaRuntimeProcess | None = None
        self._transport: ManagedGemmaTransport | None = None
        self._requested_backend: GemmaBackend | None = None
        self._requested_vulkan_device: str | None = None
        self._active_spec: GemmaModelSpec | None = None
        self._effective_backend: EffectiveGemmaBackend | None = None
        self._prefix_identity: str | None = None
        self._slot_identities: list[str | None] = [None] * GEMMA_SLOT_COUNT
        self._slot_lru: list[int] = list(range(GEMMA_SLOT_COUNT))
        self._busy_slots: set[int] = set()
        self._slot_changed = asyncio.Event()
        self._generation = 0
        self._unsettled = False
        self._transitioning = False
        self._readiness: ManagedGemmaReadiness | None = None
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._server_tasks: set[asyncio.Task[object]] = set()
        self._operation_tasks: set[asyncio.Task[object]] = set()
        self._releasing = False
        self._closed = False

    @property
    def readiness(self) -> ManagedGemmaReadiness | None:
        process = self._process
        if process is None or process.returncode is not None:
            return None
        return self._readiness

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": type(self).__name__,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    async def prepare(
        self,
        *,
        backend: GemmaBackend,
        source_language: str,
        target_language: str,
        system_prompt: str,
        vulkan_device: str = "Vulkan0",
        provision_kwargs: Mapping[str, object] | None = None,
        spec: GemmaModelSpec | None = None,
    ) -> ManagedGemmaReadiness:
        task = self._register_operation()
        try:
            async with self._condition:
                readiness, _slot_id = await self._prepare_locked(
                    backend=backend,
                    source_language=source_language,
                    target_language=target_language,
                    system_prompt=system_prompt,
                    vulkan_device=vulkan_device,
                    provision_kwargs=provision_kwargs,
                    spec=spec,
                )
                return readiness
        finally:
            self._operation_tasks.discard(task)

    async def translate(
        self,
        *,
        backend: GemmaBackend,
        source_language: str,
        target_language: str,
        system_prompt: str,
        user_message: str,
        vulkan_device: str = "Vulkan0",
        spec: GemmaModelSpec | None = None,
        max_output_tokens: int | None = None,
    ) -> ManagedGemmaResponse:
        task = self._register_operation()
        slot_id: int | None = None
        try:
            async with self._condition:
                readiness, slot_id = await self._prepare_locked(
                    backend=backend,
                    source_language=source_language,
                    target_language=target_language,
                    system_prompt=system_prompt,
                    vulkan_device=vulkan_device,
                    provision_kwargs=None,
                    spec=spec,
                )
                transport = self._transport
                if transport is None:
                    raise ManagedGemmaRuntimeError("managed Gemma transport is unavailable")
                generation = self._generation
                self._busy_slots.add(slot_id)
            transport_kwargs = {
                "system_prompt": build_managed_gemma_system_prompt(
                    system_prompt=system_prompt,
                    source_language=source_language,
                    target_language=target_language,
                ),
                "user_message": user_message,
                "slot_id": slot_id,
            }
            if max_output_tokens is not None:
                transport_kwargs["max_output_tokens"] = max_output_tokens
            response = await self._await_server_work(
                self._translate_settled(transport, transport_kwargs, slot_id)
            )
            self._ensure_accepting()
            if generation != self._generation:
                raise ManagedGemmaRuntimeError("managed Gemma result superseded by runtime change")
            self._ensure_process_alive()
            self._log_metrics(readiness, response.metrics)
            return response
        finally:
            if slot_id is not None:
                self._busy_slots.discard(slot_id)
                self._slot_changed.set()
            self._operation_tasks.discard(task)

    async def _wait_slot_change_locked(self) -> None:
        self._slot_changed.clear()
        self._condition.release()
        cancellation: asyncio.CancelledError | None = None
        try:
            await self._slot_changed.wait()
        finally:
            while True:
                try:
                    await self._condition.acquire()
                    break
                except asyncio.CancelledError as exc:
                    cancellation = exc
            if cancellation is not None:
                raise cancellation

    async def _drain_locked(self) -> None:
        while self._busy_slots:
            await self._wait_slot_change_locked()

    async def _await_server_work(self, work: Awaitable[_ServerResult]) -> _ServerResult:
        task = asyncio.ensure_future(work)
        self._server_tasks.add(task)
        cancellation: asyncio.CancelledError | None = None
        try:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as exc:
                    cancellation = exc
                except BaseException:
                    break
            if cancellation is not None:
                if not task.cancelled():
                    task.exception()
                raise cancellation
            return task.result()
        finally:
            self._server_tasks.discard(task)

    async def _translate_settled(
        self,
        transport: ManagedGemmaTransport,
        arguments: dict[str, object],
        slot_id: int,
    ) -> ManagedGemmaResponse:
        try:
            return await transport.translate(**arguments)  # type: ignore[arg-type]
        except Exception:
            try:
                await transport.wait_slot_idle(slot_id=slot_id)
            except Exception:
                self._unsettled = True
            self._slot_identities[slot_id] = None
            raise

    def _register_operation(self) -> asyncio.Task[object]:
        if self._closed:
            raise ManagedGemmaRuntimeClosedError("managed Gemma runtime is closed")
        if self._releasing:
            raise ManagedGemmaRuntimeError("managed Gemma runtime is releasing")
        task = asyncio.current_task()
        if task is None:
            raise ManagedGemmaRuntimeError("managed Gemma operation requires an asyncio task")
        self._operation_tasks.add(task)
        return task

    def _ensure_accepting(self) -> None:
        if self._closed:
            raise ManagedGemmaRuntimeClosedError("managed Gemma runtime is closed")
        if self._releasing:
            raise ManagedGemmaRuntimeError("managed Gemma runtime is releasing")

    async def _prepare_locked(
        self,
        *,
        backend: GemmaBackend,
        source_language: str,
        target_language: str,
        system_prompt: str,
        vulkan_device: str,
        provision_kwargs: Mapping[str, object] | None,
        spec: GemmaModelSpec | None,
    ) -> tuple[ManagedGemmaReadiness, int]:
        self._ensure_accepting()
        if backend not in {"cpu", "gpu"}:
            raise ValueError("managed Gemma backend must be cpu or gpu")
        if not source_language.strip() or not target_language.strip():
            raise ValueError("managed Gemma language pair must be non-empty")
        resolved_spec = spec or self._active_spec or e4b_gemma_spec()
        if resolved_spec.model_id == GEMMA_26B_MODEL_ID and backend != "gpu":
            raise ManagedGemmaRuntimeError(
                "Gemma 4 26B requires a working Vulkan GPU; select E4B for CPU inference"
            )
        install_dir = (
            self._explicit_install_dir or default_gemma_install_dir(resolved_spec)
        ).resolve()
        rendered_prompt = build_managed_gemma_system_prompt(
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
        )
        requested_vulkan_device = vulkan_device if backend == "gpu" else None
        while True:
            await self._condition.wait_for(lambda: not self._transitioning)
            self._ensure_accepting()
            process_alive = self._process is not None and self._process.returncode is None
            runtime_changed = (
                not process_alive
                or self._requested_backend != backend
                or self._requested_vulkan_device != requested_vulkan_device
                or self._active_spec != resolved_spec
                or self._install_dir != install_dir
                or self._unsettled
            )
            if runtime_changed:
                self._transitioning = True
                if not self._unsettled:
                    self._generation += 1
                self._readiness = None
                try:
                    await self._drain_locked()
                    self._ensure_accepting()
                    self._install_dir = install_dir
                    kwargs = dict(provision_kwargs or {})
                    kwargs.setdefault("install_dir", install_dir)
                    kwargs.setdefault("spec", resolved_spec)
                    await self._provisioner(**kwargs)
                    self._ensure_accepting()
                    await self._stop_locked()
                    try:
                        await self._start_locked(
                            backend=backend, vulkan_device=vulkan_device, spec=resolved_spec
                        )
                        self._ensure_process_alive()
                    except asyncio.CancelledError:
                        await self._stop_locked()
                        raise
                    except Exception as exc:
                        await self._stop_locked()
                        if (
                            backend != "gpu"
                            or resolved_spec.model_id == GEMMA_26B_MODEL_ID
                            or isinstance(exc, ManagedGemmaRuntimeVersionError)
                        ):
                            raise ManagedGemmaRuntimeError(
                                f"managed Gemma {backend} startup failed; "
                                f"check the runtime and GPU resources or explicitly select E4B CPU: {exc}"
                            ) from exc
                        self._emit(
                            "[ManagedGemma] backend_fallback requested=gpu effective=cpu reason=vulkan_start_failed",
                            logging.WARNING,
                        )
                        await self._start_locked(
                            backend="cpu", vulkan_device=vulkan_device, spec=resolved_spec
                        )
                        self._ensure_process_alive()
                    self._requested_backend = backend
                    self._requested_vulkan_device = requested_vulkan_device
                    self._active_spec = resolved_spec
                finally:
                    self._transitioning = False
                    self._condition.notify_all()
            capacity = 2 if self._effective_backend == "vulkan" else 1
            if len(self._busy_slots) < capacity:
                break
            await self._wait_slot_change_locked()
        effective = self._effective_backend
        if effective is None:
            raise ManagedGemmaRuntimeError("managed Gemma backend is unavailable")
        identity = self._make_prefix_identity(
            resolved_spec, rendered_prompt, source_language, target_language, effective
        )
        slot_id, needs_prefix = self._assign_slot(identity)
        if needs_prefix:
            self._readiness = None
            self._slot_identities[slot_id] = None
            transport = self._transport
            if transport is None:
                raise ManagedGemmaRuntimeError("managed Gemma transport is unavailable")
            try:
                await self._await_server_work(
                    self._prepare_prefix(transport, identity, slot_id, rendered_prompt)
                )
            except Exception as exc:
                self._transitioning = True
                try:
                    await self._drain_locked()
                    self._ensure_accepting()
                    await self._stop_locked()
                    if (
                        backend != "gpu"
                        or effective != "vulkan"
                        or resolved_spec.model_id == GEMMA_26B_MODEL_ID
                    ):
                        raise ManagedGemmaRuntimeError(
                            "managed Gemma prefix preparation failed; check backend resources "
                            "or explicitly select another supported model/backend"
                        ) from exc
                    self._emit(
                        "[ManagedGemma] backend_fallback requested=gpu effective=cpu reason=vulkan_prefill_failed",
                        logging.WARNING,
                    )
                    await self._start_locked(
                        backend="cpu", vulkan_device=vulkan_device, spec=resolved_spec
                    )
                    self._requested_backend = backend
                    self._requested_vulkan_device = requested_vulkan_device
                    self._active_spec = resolved_spec
                    effective = "cpu"
                    identity = self._make_prefix_identity(
                        resolved_spec, rendered_prompt, source_language, target_language, effective
                    )
                    slot_id, _needs_prefix = self._assign_slot(identity)
                    transport = self._transport
                    if transport is None:
                        raise ManagedGemmaRuntimeError("managed Gemma transport is unavailable")
                    try:
                        await self._await_server_work(
                            self._prepare_prefix(transport, identity, slot_id, rendered_prompt)
                        )
                    except BaseException:
                        await self._stop_locked()
                        raise
                finally:
                    self._transitioning = False
                    self._condition.notify_all()
            self._ensure_accepting()
            self._slot_identities[slot_id] = identity
        self._prefix_identity = identity
        self._ensure_accepting()
        self._ensure_process_alive()
        readiness = ManagedGemmaReadiness(
            requested_backend=backend,
            effective_backend=effective,
            source_language=source_language,
            target_language=target_language,
            prefix_identity=identity,
        )
        self._readiness = readiness
        return readiness, slot_id

    def _make_prefix_identity(
        self,
        spec: GemmaModelSpec,
        prompt: str,
        source: str,
        target: str,
        backend: EffectiveGemmaBackend,
    ) -> str:
        executable = (
            self._runtime_paths.vulkan_server
            if backend == "vulkan"
            else self._runtime_paths.cpu_server
        )
        return _prefix_identity(
            spec=spec,
            system_prompt=prompt,
            source_language=source,
            target_language=target,
            effective_backend=backend,
            profile_identity=self._profile_identity + "\0" + str(executable.resolve()),
        )

    async def _prepare_prefix(
        self,
        transport: ManagedGemmaTransport,
        identity: str,
        slot_id: int,
        prompt: str,
    ) -> None:
        try:
            restored = await self._restore_prefix_locked(transport, identity, slot_id)
            if not restored:
                await transport.prepare_prefix(system_prompt=prompt, slot_id=slot_id)
                self._ensure_process_alive()
                await self._save_prefix_locked(transport, identity, slot_id)
        except Exception:
            try:
                await transport.wait_slot_idle(slot_id=slot_id)
            except Exception:
                self._unsettled = True
            raise

    def _clear_slots(self) -> None:
        self._slot_identities = [None] * GEMMA_SLOT_COUNT
        self._slot_lru = list(range(GEMMA_SLOT_COUNT))
        self._prefix_identity = None

    def _touch_slot(self, slot_id: int) -> None:
        if slot_id in self._slot_lru:
            self._slot_lru.remove(slot_id)
        self._slot_lru.append(slot_id)

    def _assign_slot(self, identity: str) -> tuple[int, bool]:
        for slot_id, resident in enumerate(self._slot_identities):
            if resident == identity and slot_id not in self._busy_slots:
                self._touch_slot(slot_id)
                return slot_id, False
        for slot_id, resident in enumerate(self._slot_identities):
            if resident is None and slot_id not in self._busy_slots:
                self._touch_slot(slot_id)
                return slot_id, True
        slot_id = next(slot for slot in self._slot_lru if slot not in self._busy_slots)
        self._touch_slot(slot_id)
        return slot_id, True

    async def _restore_prefix_locked(
        self,
        transport: ManagedGemmaTransport,
        identity: str,
        slot_id: int,
    ) -> bool:
        cache = self._prefix_cache
        backend = self._effective_backend
        if cache is None or backend is None or not cache.has(identity, backend):
            return False
        filename = cache.filename_for(identity, backend)
        try:
            restored = await transport.restore_prefix(filename=filename, slot_id=slot_id)
        except Exception:
            await transport.wait_slot_idle(slot_id=slot_id)
            return False
        if restored:
            cache.touch(identity, backend)
            self._emit(
                f"[ManagedGemma] prefix_cache restored backend={backend} identity={identity[:12]} slot={slot_id}",
                logging.INFO,
            )
        return restored

    async def _save_prefix_locked(
        self,
        transport: ManagedGemmaTransport,
        identity: str,
        slot_id: int,
    ) -> None:
        cache = self._prefix_cache
        backend = self._effective_backend
        if cache is None or backend is None:
            return
        filename = cache.filename_for(identity, backend)
        try:
            await transport.save_prefix(filename=filename, slot_id=slot_id)
        except Exception:
            await transport.wait_slot_idle(slot_id=slot_id)
            self._emit(
                f"[ManagedGemma] prefix_cache save_failed backend={backend} identity={identity[:12]} slot={slot_id}",
                logging.WARNING,
            )
            return
        cache.remember(identity, backend)

    def _ensure_process_alive(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            raise ManagedGemmaRuntimeError("managed Gemma process exited before readiness")

    async def _wait_for_startup(
        self,
        *,
        process: ManagedGemmaRuntimeProcess,
        transport: ManagedGemmaTransport,
        backend: GemmaBackend,
    ) -> None:
        readiness = asyncio.create_task(
            transport.wait_until_ready(timeout_s=self._startup_timeout_s)
        )
        process_exit = asyncio.create_task(process.wait())
        try:
            done, _pending = await asyncio.wait(
                (readiness, process_exit),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if process_exit in done:
                exit_code = process_exit.result()
                self._emit(
                    f"[ManagedGemma] process_exit phase=startup backend={backend} exit_code={exit_code}",
                    logging.ERROR,
                )
                raise ManagedGemmaRuntimeError(
                    f"managed Gemma {backend} process exited during startup "
                    f"with exit code {exit_code}"
                )
            await readiness
        finally:
            for task in (readiness, process_exit):
                if not task.done():
                    task.cancel()
            await asyncio.gather(readiness, process_exit, return_exceptions=True)

    async def _start_locked(
        self,
        *,
        backend: GemmaBackend,
        vulkan_device: str,
        spec: GemmaModelSpec,
    ) -> None:
        executable = (
            self._runtime_paths.vulkan_server
            if backend == "gpu"
            else self._runtime_paths.cpu_server
        )
        if not executable.is_file():
            raise ManagedGemmaRuntimeError(f"managed llama.cpp server is missing: {executable}")
        await self._version_validator(executable)
        port = self._port_allocator()
        process_cwd = self._install_dir
        slot_save_path = None
        if self._prefix_cache is not None:
            self._prefix_cache.cache_dir.mkdir(parents=True, exist_ok=True)
            process_cwd = self._prefix_cache.cache_dir.resolve()
            slot_save_path = Path(".")
        command = build_gemma_server_command(
            executable=executable,
            install_dir=self._install_dir,
            backend=backend,
            port=port,
            vulkan_device=vulkan_device,
            slot_save_path=slot_save_path,
            spec=spec,
        )
        port_index = command.index("--port")
        self._profile_identity = "\0".join(command[1:port_index] + command[port_index + 2 :])
        self._unsettled = False
        process = await self._process_factory(command, process_cwd)
        self._process = process
        try:
            transport = self._transport_factory(f"http://127.0.0.1:{port}")
            self._transport = transport
            self._effective_backend = "vulkan" if backend == "gpu" else "cpu"
            await self._wait_for_startup(
                process=process,
                transport=transport,
                backend=backend,
            )
        except BaseException:
            await self._stop_locked()
            raise

    async def release(self) -> None:
        if self._closed:
            return
        self._releasing = True
        self._readiness = None
        await self._cancel_operations()
        async with self._lock:
            await self._stop_locked()
        self._releasing = False

    async def close(self) -> None:
        if (
            self._closed
            and self._process is None
            and self._transport is None
            and not self._operation_tasks
        ):
            return
        self._closed = True
        self._readiness = None
        await self._cancel_operations()
        async with self._lock:
            await self._stop_locked()

    async def _cancel_operations(self) -> None:
        current = asyncio.current_task()
        tasks = tuple(task for task in self._operation_tasks if task is not current)
        for task in tasks:
            task.cancel()
        if tasks:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self._shutdown_timeout_s,
            )
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            if pending:
                await self._stop_locked()
                for work in tuple(self._server_tasks):
                    work.cancel()
                done, pending = await asyncio.wait(pending, timeout=self._shutdown_timeout_s)
                if done:
                    await asyncio.gather(*done, return_exceptions=True)
                if pending:
                    raise ManagedGemmaRuntimeError(
                        f"managed Gemma operations did not stop: {len(pending)}"
                    )

    async def _stop_locked(self) -> None:
        transport = self._transport
        process = self._process
        self._requested_backend = None
        self._requested_vulkan_device = None
        self._effective_backend = None
        self._clear_slots()
        self._readiness = None
        failures: list[BaseException] = []
        if transport is not None:
            try:
                await asyncio.wait_for(
                    transport.close(),
                    timeout=self._shutdown_timeout_s,
                )
            except BaseException as exc:
                failures.append(exc)
            else:
                if self._transport is transport:
                    self._transport = None
        if process is not None:
            process_failures: list[BaseException] = []
            if process.returncode is None:
                try:
                    process.terminate()
                except BaseException as exc:
                    process_failures.append(exc)
                else:
                    try:
                        await asyncio.wait_for(
                            process.wait(),
                            timeout=self._shutdown_timeout_s,
                        )
                    except asyncio.TimeoutError:
                        pass
                    except BaseException as exc:
                        process_failures.append(exc)
            if process.returncode is None:
                try:
                    process.kill()
                except BaseException as exc:
                    process_failures.append(exc)
                else:
                    try:
                        await asyncio.wait_for(
                            process.wait(),
                            timeout=self._shutdown_timeout_s,
                        )
                    except BaseException as exc:
                        process_failures.append(exc)
            if process.returncode is not None and self._process is process:
                self._process = None
            else:
                failures.extend(process_failures)
                if not process_failures:
                    failures.append(
                        ManagedGemmaRuntimeError(
                            "managed Gemma process did not stop after terminate and kill"
                        )
                    )
        if failures:
            try:
                if len(failures) == 1:
                    raise failures[0]
                raise BaseExceptionGroup("managed Gemma runtime shutdown failed", failures)
            finally:
                self._emit(
                    f"[ManagedGemma] shutdown_failed failures={len(failures)}",
                    logging.ERROR,
                )

    def _log_metrics(
        self,
        readiness: ManagedGemmaReadiness,
        metrics: ManagedGemmaMetrics,
    ) -> None:
        fields = [
            f"backend={readiness.effective_backend}",
            f"language_pair={readiness.source_language}->{readiness.target_language}",
            f"prompt_tokens={metrics.prompt_tokens}",
        ]
        if metrics.cached_prompt_tokens is not None:
            fields.append(f"cached_prompt_tokens={metrics.cached_prompt_tokens}")
        fields.extend(
            (
                f"completion_tokens={metrics.completion_tokens}",
                f"prompt_ms={metrics.prompt_ms:.3f}",
                f"generation_ms={metrics.generation_ms:.3f}",
                f"generation_tps={metrics.generation_tps:.3f}",
            )
        )
        if readiness.effective_backend == "cpu":
            if metrics.drafted_tokens is not None:
                fields.append(f"drafted_tokens={metrics.drafted_tokens}")
            if metrics.accepted_tokens is not None:
                fields.append(f"accepted_tokens={metrics.accepted_tokens}")
        self._emit("[ManagedGemma][Performance] " + " ".join(fields), logging.INFO)

    def _emit(self, message: str, level: int) -> None:
        if self._log_sink is None:
            return
        result = self._log_sink(message, level)
        if inspect.isawaitable(result):
            raise TypeError("managed Gemma log sink must be synchronous")


__all__ = [
    "GEMMA_PROMPT_TEMPLATE_VERSION",
    "ManagedGemmaMetrics",
    "ManagedGemmaReadiness",
    "ManagedGemmaResponse",
    "ManagedGemmaRuntimeClosedError",
    "ManagedGemmaRuntimeError",
    "ManagedGemmaRuntimeVersionError",
    "ManagedGemmaRuntimeOwner",
    "ManagedGemmaRuntimeProcess",
    "ManagedGemmaTransport",
    "build_managed_gemma_system_prompt",
    "validate_gemma_runtime_version",
]
