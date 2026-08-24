from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import socket
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

from puripuly_heart.core.http_client_logging import suppress_http_client_logs
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.osc.control_schema import (
    OSC_MUTE_SELF_ADDRESS,
    OSC_PARAMETER_DEFINITIONS,
)
from puripuly_heart.core.osc.oscquery_contract import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
    OscQueryServicePort,
    OscQueryServicesChanged,
)

OSCJSON_SERVICE_TYPE = "_oscjson._tcp.local."
OSC_SERVICE_TYPE = "_osc._udp.local."
OSCQUERY_ADVERTISEMENT_TTL_SECONDS = 120
_UNREGISTER_RETRY_DELAY_SECONDS = 0.05

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CallbackOscQueryService(OscQueryServicePort):
    discover: Callable[[], Awaitable[OscQueryServiceInfo | None]]
    advertise: Callable[[OscQueryAdvertisement], Awaitable[None]]
    unadvertise: Callable[[], Awaitable[None]]
    query: Callable[[OscQueryServiceInfo], Awaitable[Mapping[str, object]]]
    start_callback: Callable[[OscQueryServicesChanged | None], Awaitable[None]] | None = None
    stop_callback: Callable[[], Awaitable[None]] | None = None

    async def start(self, services_changed: OscQueryServicesChanged | None = None) -> None:
        if self.start_callback is not None:
            await self.start_callback(services_changed)

    async def stop(self) -> None:
        if self.stop_callback is not None:
            await self.stop_callback()

    async def discover_vrchat(self) -> OscQueryServiceInfo | None:
        return await self.discover()

    async def advertise_receiver(self, advertisement: OscQueryAdvertisement) -> None:
        await self.advertise(advertisement)

    async def unadvertise_receiver(self) -> None:
        await self.unadvertise()

    async def query_avatar(self, service: OscQueryServiceInfo) -> Mapping[str, object]:
        return await self.query(service)


class ZeroconfOscQueryService(OscQueryServicePort):
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        service_name: str = "PuriPuly Heart",
    ) -> None:
        self.host = host
        self.service_name = service_name
        self.started = False
        self.advertisement: OscQueryAdvertisement | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._zeroconf: object | None = None
        self._browsers: list[object] = []
        self._services: dict[str, OscQueryServiceInfo] = {}
        self._raw_services: dict[str, dict[str, object]] = {}
        self._services_changed: OscQueryServicesChanged | None = None
        self._callback_sequence = 0
        self._callback_scope = LifecycleScope("ZeroconfOscQueryService")
        self._http_server: asyncio.Server | None = None
        self._http_tasks: set[asyncio.Task[None]] = set()
        self._query_tree: Mapping[str, object] = {}
        self._host_info: Mapping[str, object] = {}
        self._registered_infos: list[object] = []

    async def start(self, services_changed: OscQueryServicesChanged | None = None) -> None:
        if self.started:
            self._services_changed = services_changed
            return
        try:
            from zeroconf import ServiceBrowser, Zeroconf
        except ImportError as exc:
            raise RuntimeError("OSCQuery automatic mode requires the zeroconf package") from exc

        self._loop = asyncio.get_running_loop()
        self._services_changed = services_changed
        try:
            self._zeroconf = Zeroconf()
            for service_type in (OSCJSON_SERVICE_TYPE, OSC_SERVICE_TYPE):
                self._browsers.append(
                    ServiceBrowser(
                        self._zeroconf,
                        service_type,
                        handlers=[self._service_state_changed],
                    )
                )
            self.started = True
            self._notify_services_changed()
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        await self.unadvertise_receiver()
        for browser in self._browsers:
            cancel = getattr(browser, "cancel", None)
            if callable(cancel):
                cancel()
        self._browsers.clear()
        await self._callback_scope.close()
        self._callback_scope = LifecycleScope("ZeroconfOscQueryService")
        zeroconf = self._zeroconf
        self._zeroconf = None
        if zeroconf is not None:
            close = getattr(zeroconf, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        self._services.clear()
        self._raw_services.clear()
        self._services_changed = None
        self._loop = None
        self.started = False

    async def discover_vrchat(self) -> OscQueryServiceInfo | None:
        candidates = [service for service in self._services.values() if service.is_vrchat]
        if not candidates:
            candidates = [
                service
                for service in self._services.values()
                if "vrchat" in service.service_id.casefold()
            ]
        if not candidates:
            return None
        for candidate in sorted(candidates, key=lambda service: service.service_id.casefold()):
            resolved = await self._resolve_host_info(candidate)
            if resolved is not None:
                return resolved
            self._forget_service(candidate.service_id)
        return None

    async def advertise_receiver(self, advertisement: OscQueryAdvertisement) -> None:
        if self._zeroconf is None:
            raise RuntimeError("OSCQuery service is not started")
        await self.unadvertise_receiver()
        if self._http_server is None:
            self._http_server = await asyncio.start_server(
                self._handle_http_client,
                host=advertisement.host,
                port=advertisement.query_port or 0,
            )
        query_port = int(self._http_server.sockets[0].getsockname()[1])
        self.advertisement = replace(advertisement, query_port=query_port)
        self._query_tree = self._build_query_tree(self.advertisement)
        self._host_info = self._build_host_info(self.advertisement)
        try:
            from zeroconf import ServiceInfo

            address = socket.inet_aton(advertisement.host)
            name = mdns_instance_name(self.advertisement.service_name)
            await self._forget_unreachable_puripuly_services()
            self._forget_service(name)
            server = f"{_service_name(socket.gethostname())}.local."
            properties = {
                b"txtvers": b"1",
                b"NAME": self.advertisement.service_name.encode("utf-8"),
                b"OSC_IP": advertisement.host.encode("ascii"),
                b"OSC_PORT": str(advertisement.port).encode("ascii"),
                b"EXTENSIONS": b"",
            }
            infos = [
                ServiceInfo(
                    OSCJSON_SERVICE_TYPE,
                    f"{name}.{OSCJSON_SERVICE_TYPE}",
                    addresses=[address],
                    port=query_port,
                    properties=properties,
                    server=server,
                    host_ttl=OSCQUERY_ADVERTISEMENT_TTL_SECONDS,
                    other_ttl=OSCQUERY_ADVERTISEMENT_TTL_SECONDS,
                ),
                ServiceInfo(
                    OSC_SERVICE_TYPE,
                    f"{name}.{OSC_SERVICE_TYPE}",
                    addresses=[address],
                    port=advertisement.port,
                    properties=properties,
                    server=server,
                    host_ttl=OSCQUERY_ADVERTISEMENT_TTL_SECONDS,
                    other_ttl=OSCQUERY_ADVERTISEMENT_TTL_SECONDS,
                ),
            ]
        except (ImportError, OSError, ValueError) as exc:
            await self._close_http_server()
            self.advertisement = None
            raise RuntimeError("failed to create OSCQuery advertisement") from exc
        registered: list[object] = []
        try:
            for info in infos:
                await self._register_service(info)
                registered.append(info)
        except BaseException:
            self._registered_infos = registered
            await self._unregister_registered_infos()
            await self._close_http_server()
            self.advertisement = None
            raise
        self._registered_infos = registered

    async def unadvertise_receiver(self) -> None:
        await self._unregister_registered_infos()
        self.advertisement = None
        self._query_tree = {}
        self._host_info = {}
        await self._close_http_server()

    async def _register_service(self, info: object) -> None:
        zeroconf = self._zeroconf
        if zeroconf is None:
            raise RuntimeError("OSCQuery service is not started")

        def _register() -> None:
            zeroconf.register_service(info, allow_name_change=True)

        await asyncio.to_thread(_register)

    async def _unregister_registered_infos(self) -> None:
        zeroconf = self._zeroconf
        infos = list(self._registered_infos)
        if zeroconf is None or not infos:
            self._registered_infos.clear()
            return
        remaining = [info for info in infos if not await self._unregister_service(zeroconf, info)]
        if remaining:
            await asyncio.sleep(_UNREGISTER_RETRY_DELAY_SECONDS)
            remaining = [
                info for info in remaining if not await self._unregister_service(zeroconf, info)
            ]
        if remaining:
            logger.error(
                "OSCQuery unregister failed for %s service(s)",
                len(remaining),
            )
        self._registered_infos = remaining

    async def _unregister_service(self, zeroconf: object, info: object) -> bool:
        try:
            await self._await_zeroconf_unregister(zeroconf, info)
        except Exception:
            logger.warning(
                "OSCQuery unregister attempt failed",
                exc_info=True,
            )
            return False
        return True

    async def _await_zeroconf_unregister(self, zeroconf: object, info: object) -> None:
        async_unregister = getattr(zeroconf, "async_unregister_service", None)
        loop = getattr(zeroconf, "loop", None)
        if callable(async_unregister) and loop is not None:

            async def _unregister() -> None:
                result = async_unregister(info)
                if inspect.isawaitable(result):
                    result = await result
                if inspect.isawaitable(result):
                    await result

            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_unregister(), loop))
            return
        unregister = getattr(zeroconf, "unregister_service", None)
        if not callable(unregister):
            raise RuntimeError("OSCQuery service cannot unregister advertisements")
        await asyncio.to_thread(unregister, info)

    async def query_avatar(self, service: OscQueryServiceInfo) -> Mapping[str, object]:
        if service.query_port is None:
            return {}
        try:
            import httpx

            with suppress_http_client_logs():
                async with httpx.AsyncClient(timeout=1.5) as client:
                    response = await client.get(
                        f"http://{service.host}:{service.query_port}/avatar"
                    )
                    response.raise_for_status()
                    payload = response.json()
        except Exception:
            return {}
        return payload if isinstance(payload, Mapping) else {}

    async def _resolve_host_info(
        self,
        service: OscQueryServiceInfo,
    ) -> OscQueryServiceInfo | None:
        if service.query_port is None:
            return service
        try:
            import httpx

            with suppress_http_client_logs():
                async with httpx.AsyncClient(timeout=1.5) as client:
                    response = await client.get(
                        f"http://{service.host}:{service.query_port}/?HOST_INFO"
                    )
                    response.raise_for_status()
                    payload = response.json()
        except Exception:
            return None
        if not isinstance(payload, Mapping):
            return None
        host = payload.get("OSC_IP")
        resolved_host = host.strip() if isinstance(host, str) and host.strip() else service.host
        resolved_port = _mapping_port(payload, "OSC_PORT") or service.osc_send_port
        return replace(
            service,
            host=resolved_host,
            osc_send_port=resolved_port,
        )

    def _service_state_changed(
        self,
        zeroconf: object,
        service_type: str,
        name: str,
        state_change: object,
    ) -> None:
        state_name = getattr(state_change, "name", str(state_change)).casefold()
        info = None
        if state_name != "removed":
            get_info = getattr(zeroconf, "get_service_info", None)
            if callable(get_info):
                with contextlib.suppress(Exception):
                    info = get_info(service_type, name, timeout=1000)
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._queue_service_update, service_type, name, info)

    def _queue_service_update(self, service_type: str, name: str, info: object | None) -> None:
        key = _base_service_name(name, service_type)
        if info is None:
            raw = self._raw_services.get(key)
            if raw is not None:
                raw.pop(service_type, None)
                if not raw:
                    self._raw_services.pop(key, None)
                    self._services.pop(key, None)
                else:
                    self._services[key] = self._merge_service_info(key, raw)
        else:
            self._raw_services.setdefault(key, {})[service_type] = info
            self._services[key] = self._merge_service_info(key, self._raw_services[key])
        self._notify_services_changed()

    def _merge_service_info(
        self,
        service_id: str,
        raw: Mapping[str, object],
    ) -> OscQueryServiceInfo:
        query_info = raw.get(OSCJSON_SERVICE_TYPE)
        osc_info = raw.get(OSC_SERVICE_TYPE)
        preferred = query_info or osc_info
        host = _service_host(preferred) or _service_host(osc_info) or "127.0.0.1"
        properties = _service_properties(preferred)
        osc_properties = _service_properties(osc_info)
        query_port = _service_port(query_info)
        send_port = _property_int(osc_properties, "OSC_PORT", "OSC_SEND_PORT") or _service_port(
            osc_info
        )
        if send_port is None:
            send_port = _property_int(properties, "OSC_PORT", "OSC_SEND_PORT")
        receive_port = _property_int(properties, "OSC_RECEIVE_PORT")
        service_name = str(properties.get("NAME", service_id))
        is_vrchat = "vrchat" in f"{service_id} {service_name}".casefold()
        return OscQueryServiceInfo(
            service_id=service_id,
            host=host,
            query_port=query_port,
            osc_send_port=send_port,
            osc_receive_port=receive_port,
            is_vrchat=is_vrchat,
        )

    async def _forget_unreachable_puripuly_services(self) -> None:
        for service_id, service in list(self._services.items()):
            if "puripuly" not in service_id.casefold():
                continue
            if service.query_port is None:
                self._forget_service(service_id)
                continue
            if await self._resolve_host_info(service) is None:
                self._forget_service(service_id)

    def _forget_service(self, service_id: str) -> None:
        self._raw_services.pop(service_id, None)
        self._services.pop(service_id, None)

    def _notify_services_changed(self) -> None:
        callback = self._services_changed
        if callback is None or self._loop is None or self._loop.is_closed():
            return
        result = callback(tuple(self._services.values()))
        if not inspect.isawaitable(result):
            return
        try:
            self._callback_sequence += 1
            start_lifecycle_task(
                self._callback_scope,
                result,
                name=f"services-changed-{self._callback_sequence}",
            )
        except RuntimeError:
            return

    async def _handle_http_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        current_task = asyncio.current_task()
        if current_task is not None:
            self._http_tasks.add(current_task)
        try:
            request = await asyncio.wait_for(reader.readline(), timeout=1.0)
            parts = request.decode("ascii", errors="ignore").strip().split()
            if len(parts) < 2 or parts[0].upper() != "GET":
                await self._write_http_response(writer, 400, {})
                return
            status, payload = self._resolve_query_target(parts[1])
            await self._write_http_response(writer, status, payload)
        except (asyncio.CancelledError, TimeoutError, UnicodeError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            if current_task is not None:
                self._http_tasks.discard(current_task)

    async def _close_http_server(self) -> None:
        server = self._http_server
        self._http_server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        for task in tuple(self._http_tasks):
            task.cancel()
        if self._http_tasks:
            await asyncio.gather(*self._http_tasks, return_exceptions=True)
        self._http_tasks.clear()

    def _resolve_query_target(self, target: str) -> tuple[int, Mapping[str, object]]:
        parsed = urlsplit(target)
        query_attributes = [
            part.partition("=")[0].casefold() for part in parsed.query.split("&") if part
        ]
        if query_attributes:
            if query_attributes == ["host_info"]:
                return 200, self._host_info
            if len(query_attributes) != 1:
                return 400, {}
            node = self._resolve_query_node(parsed.path or "/")
            if node is None:
                return 404, {}
            attribute = query_attributes[0]
            if attribute == "access":
                return 200, {"ACCESS": node.get("ACCESS", 0)}
            if attribute == "value":
                access = node.get("ACCESS", 0)
                if isinstance(access, int) and not access & 1:
                    return 204, {}
                if "VALUE" not in node:
                    return 200, {}
                return 200, {"VALUE": node["VALUE"]}
            return 400, {}

        path = parsed.path or "/"
        node = self._resolve_query_node(path)
        return (200, node) if node is not None else (404, {})

    def _resolve_query_node(self, path: str) -> Mapping[str, object] | None:
        normalized_path = "/" if not path.strip("/") else f"/{path.strip('/')}"
        node: object = self._query_tree
        if normalized_path == "/":
            return node if isinstance(node, Mapping) else None
        for segment in normalized_path.strip("/").split("/"):
            if not isinstance(node, Mapping):
                return None
            contents = node.get("CONTENTS")
            if not isinstance(contents, Mapping):
                return None
            node = contents.get(segment)
        return node if isinstance(node, Mapping) else None

    @staticmethod
    async def _write_http_response(
        writer: asyncio.StreamWriter,
        status: int,
        payload: Mapping[str, object],
    ) -> None:
        reasons = {200: "OK", 204: "No Content", 400: "Bad Request", 404: "Not Found"}
        encoded = (
            b"" if status == 204 else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        writer.write(
            f"HTTP/1.1 {status} {reasons.get(status, 'Error')}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(encoded)}\r\nConnection: close\r\n\r\n".encode("ascii") + encoded
        )
        await writer.drain()

    @staticmethod
    def _build_query_tree(advertisement: OscQueryAdvertisement) -> Mapping[str, object]:
        parameters: dict[str, object] = {}
        for definition in OSC_PARAMETER_DEFINITIONS.values():
            parameters[definition.name] = {
                "FULL_PATH": definition.address,
                "ACCESS": 3,
                "TYPE": "T" if definition.value_type == "bool" else "i",
                "VALUE": [False] if definition.value_type == "bool" else [0],
            }
        parameters["MuteSelf"] = {
            "FULL_PATH": OSC_MUTE_SELF_ADDRESS,
            "ACCESS": 2,
            "TYPE": "T",
        }
        return {
            "FULL_PATH": "/",
            "ACCESS": 0,
            "CONTENTS": {
                "avatar": {
                    "FULL_PATH": "/avatar",
                    "ACCESS": 0,
                    "CONTENTS": {
                        "change": {
                            "FULL_PATH": "/avatar/change",
                            "ACCESS": 2,
                            "TYPE": "s",
                        },
                        "parameters": {
                            "FULL_PATH": "/avatar/parameters",
                            "ACCESS": 0,
                            "CONTENTS": parameters,
                        },
                    },
                }
            },
        }

    @staticmethod
    def _build_host_info(advertisement: OscQueryAdvertisement) -> Mapping[str, object]:
        return {
            "NAME": advertisement.service_name,
            "OSC_IP": advertisement.host,
            "OSC_PORT": advertisement.port,
            "OSC_TRANSPORT": "UDP",
            "EXTENSIONS": {
                "ACCESS": True,
                "VALUE": True,
            },
        }


def mdns_instance_name(display_name: str, *, pid: int | None = None) -> str:
    process_id = os.getpid() if pid is None else pid
    return _service_name(f"{display_name} {process_id}")


def _service_name(value: str) -> str:
    normalized = "".join(char if char.isalnum() else "-" for char in value).strip("-")
    return normalized or "PuriPuly-Heart"


def _base_service_name(name: str, service_type: str) -> str:
    suffix = f".{service_type}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _service_host(info: object | None) -> str | None:
    if info is None:
        return None
    parsed = getattr(info, "parsed_addresses", None)
    if callable(parsed):
        with contextlib.suppress(Exception):
            addresses = parsed()
            if addresses:
                return str(addresses[0])
    return None


def _service_port(info: object | None) -> int | None:
    value = getattr(info, "port", None) if info is not None else None
    return int(value) if isinstance(value, int) and 1 <= value <= 65535 else None


def _service_properties(info: object | None) -> dict[str, str]:
    properties = getattr(info, "properties", {}) if info is not None else {}
    normalized: dict[str, str] = {}
    if not isinstance(properties, Mapping):
        return normalized
    for key, value in properties.items():
        key_text = key.decode("utf-8", errors="ignore") if isinstance(key, bytes) else str(key)
        if isinstance(value, bytes):
            value_text = value.decode("utf-8", errors="ignore")
        else:
            value_text = str(value)
        normalized[key_text.upper()] = value_text
    return normalized


def _property_int(properties: Mapping[str, str], *keys: str) -> int | None:
    for key in keys:
        value = properties.get(key)
        if value is None:
            continue
        with contextlib.suppress(TypeError, ValueError):
            parsed = int(value)
            if 1 <= parsed <= 65535:
                return parsed
    return None


def _mapping_port(values: Mapping[str, object], key: str) -> int | None:
    value = values.get(key)
    if isinstance(value, bool):
        return None
    with contextlib.suppress(TypeError, ValueError):
        parsed = int(value)
        if 1 <= parsed <= 65535:
            return parsed
    return None


__all__ = [
    "CallbackOscQueryService",
    "OSCQUERY_ADVERTISEMENT_TTL_SECONDS",
    "ZeroconfOscQueryService",
    "mdns_instance_name",
]
