from __future__ import annotations

import ipaddress
import dataclasses
import hashlib
import json
import os
import socket
import sys
import time
import psutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
ARM = Path(os.environ["PP246_ARM"])
TRACE = ARM / "trace.jsonl"


def sanitized(text):
    for path, label in ((str(ARM), "<ARM>"), (str(ROOT), "<CHECKOUT>"), (str(Path(os.environ["PP246_EXTERNAL"])), "<EXTERNAL>")):
        for form in (path, path.replace("\\", "\\\\"), path.replace("\\", "/")):
            text = text.replace(form, label)
    return text


def record(kind, **data):
    row = {"kind": kind, "monotonic": time.perf_counter(), **data}
    with TRACE.open("a", encoding="utf-8") as handle:
        handle.write(sanitized(json.dumps(row, ensure_ascii=False, default=str)) + "\n")


def loopback(host):
    if host in (None, "", "localhost", b"localhost"):
        return True
    try:
        return ipaddress.ip_address(host.decode() if isinstance(host, bytes) else host).is_loopback
    except ValueError:
        return False


original_getaddrinfo = socket.getaddrinfo
original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex


def guarded_getaddrinfo(host, *args, **kwargs):
    if not loopback(host):
        record("network_denied", operation="getaddrinfo", host=str(host))
        raise PermissionError("experiment forbids non-loopback DNS")
    return original_getaddrinfo(host, *args, **kwargs)


def guarded_connect(self, address):
    if not isinstance(address, tuple) or not loopback(address[0]):
        record("network_denied", operation="connect", host=str(address[0]) if isinstance(address, tuple) else "non_ip")
        raise PermissionError("experiment forbids non-loopback connections")
    return original_connect(self, address)


def guarded_connect_ex(self, address):
    if not isinstance(address, tuple) or not loopback(address[0]):
        record("network_denied", operation="connect_ex", host=str(address[0]) if isinstance(address, tuple) else "non_ip")
        raise PermissionError("experiment forbids non-loopback connections")
    return original_connect_ex(self, address)


socket.getaddrinfo = guarded_getaddrinfo
socket.socket.connect = guarded_connect
socket.socket.connect_ex = guarded_connect_ex


from puripuly_heart.core.audio.process_identity import PsutilCurrentUserProcessSnapshots

PsutilCurrentUserProcessSnapshots.snapshots = lambda self: ()

from puripuly_heart.core.local_translation import runtime as runtime_module

if os.environ.get("PP246_IDENTITY_BUILD"):
    runtime_module.LLAMA_CPP_BUILD = os.environ["PP246_IDENTITY_BUILD"]
    runtime_module.LLAMA_CPP_COMMIT = os.environ["PP246_IDENTITY_COMMIT"]

from puripuly_heart.providers.llm.managed_gemma import HttpxManagedGemmaTransport
from puripuly_heart.release_evidence.managed_gemma_production import _captured_process_factory
from puripuly_heart.composition import application_runtime


class TracingTransport(HttpxManagedGemmaTransport):
    def _require_client(self):
        client = super()._require_client()
        if not getattr(client, "_issue246_observed", False):
            async def response_hook(response):
                if response.url.path.startswith("/slots/"):
                    await response.aread()
                    record("slot_http", path=response.url.path, action=response.url.params.get("action"), status_code=response.status_code, payload=response.json())
            client.event_hooks["response"].append(response_hook)
            client._issue246_observed = True
        return client

    async def prepare_prefix(self, **kwargs):
        start = time.perf_counter()
        result = await super().prepare_prefix(**kwargs)
        record("prepare_prefix", slot=kwargs["slot_id"], prompt_sha256=hashlib.sha256(kwargs["system_prompt"].encode()).hexdigest(), wall_ms=(time.perf_counter() - start) * 1000)
        return result

    async def save_prefix(self, **kwargs):
        result = await super().save_prefix(**kwargs)
        record("save_prefix", **kwargs)
        return result

    async def restore_prefix(self, **kwargs):
        start = time.perf_counter()
        result = await super().restore_prefix(**kwargs)
        record("restore_prefix", result=result, wall_ms=(time.perf_counter() - start) * 1000, **kwargs)
        return result

    async def translate(self, **kwargs):
        start = time.perf_counter()
        result = await super().translate(**kwargs)
        message = kwargs["user_message"]
        record("translate_completed", slot=kwargs["slot_id"], metrics=dataclasses.asdict(result.metrics), text=result.text, wall_ms=(time.perf_counter() - start) * 1000, user_message_sha256=hashlib.sha256(message.encode()).hexdigest(), context_entries=sum(line.startswith(('- [self] "', '- [peer] "')) for line in message.splitlines()))
        observer = getattr(self, "observe_resources", None)
        if observer:
            observer()
        return result

    async def close(self):
        await super().close()
        record("transport_closed", client_cleared=self._client is None)


original_factory = application_runtime.create_managed_gemma_runtime


def observed_factory(**kwargs):
    original_log_sink = kwargs.get("log_sink")
    def sink(message, level):
        record("runtime_log", message=message, level=level)
        if original_log_sink:
            original_log_sink(message, level)
    kwargs["log_sink"] = sink
    owner = original_factory(**kwargs)
    cache = owner._prefix_cache
    original_has = cache.has
    def has(identity, backend):
        result = original_has(identity, backend)
        record("cache_lookup", identity=identity, backend=backend, eligible=result, existing_files=sorted(p.name for p in cache.cache_dir.glob("*.bin")))
        return result
    cache.has = has
    captures = []
    captured_factory = _captured_process_factory(captures)
    async def process_factory(command, cwd):
        extras = ("--lazy-mode", os.environ["PP246_LAZY"]) if os.environ.get("PP246_LAZY") else ()
        if os.environ.get("PP246_CACHE_RAM") is not None:
            extras += ("--cache-ram", os.environ["PP246_CACHE_RAM"])
        capture = await captured_factory(tuple(command) + extras, cwd)
        record("native_started", command=capture.command, pid=capture.process.pid, create_time=psutil.Process(capture.process.pid).create_time())
        return capture
    owner._process_factory = process_factory
    def observe_resources():
        for capture in captures:
            if capture.process.returncode is None:
                try:
                    info = psutil.Process(capture.process.pid).memory_info()
                    record("native_resources", pid=capture.process.pid, working_set=info.rss, private_commit=info.private, peak_working_set=info.peak_wset, peak_private_commit=info.peak_pagefile)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                    record("native_resource_observation_failed", pid=capture.process.pid, exception_type=type(exc).__name__)
    def transport_factory(base_url):
        transport = TracingTransport(base_url)
        transport.observe_resources = observe_resources
        return transport
    owner._transport_factory = transport_factory
    original_prepare = owner._prepare_locked
    async def prepare(**arguments):
        result = await original_prepare(**arguments)
        record("owner_ready", readiness=dataclasses.asdict(result), active_slot=owner._active_slot, slot_identities=owner._slot_identities, slot_lru=owner._slot_lru)
        return result
    owner._prepare_locked = prepare
    original_close = owner.close
    async def close():
        observe_resources()
        try:
            await original_close()
        except BaseException as exc:
            record("owner_close_failed", exception_type=type(exc).__name__)
            raise
        finally:
            for index, capture in enumerate(captures):
                if capture.process.returncode is not None:
                    await capture.wait()
                (ARM / f"native-{index}.txt").write_text(sanitized(capture.output), encoding="utf-8")
                record("native_exit", pid=capture.process.pid, returncode=capture.returncode)
                await capture.scope.close()
            record("owner_close_observed", process_cleared=owner._process is None, transport_cleared=owner._transport is None)
    owner.close = close
    return owner


application_runtime.create_managed_gemma_runtime = observed_factory

import puripuly_heart
from puripuly_heart.main import main

record("host_identity", python=sys.executable, package=puripuly_heart.__file__, checkout="7a8151be6114a950f062e2430bd36f899d57e4de", pid=os.getpid(), runtime_build=runtime_module.LLAMA_CPP_BUILD, runtime_commit=runtime_module.LLAMA_CPP_COMMIT, process_discovery="synthetic_empty", gui="not_run")
raise SystemExit(main(["run-headless", "--config", os.environ["PP246_CONFIG"], "--ready-file", str(ARM / "ready.json")]))
