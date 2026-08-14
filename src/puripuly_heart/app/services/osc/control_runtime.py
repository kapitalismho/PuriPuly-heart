from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Callable, Mapping

from puripuly_heart.app.ports.osc_control import (
    OSC_PARAMETER_DEFINITIONS,
    OscConnectionMode,
    OscControlCodecError,
    decode_control_message,
)
from puripuly_heart.app.ports.oscquery import OscQueryServicePort
from puripuly_heart.app.services.vrc_mic_sync import VrcMicSyncOwner
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.runtime.oscquery import (
    VRCHAT_OSC_DEFAULT_INPUT_PORT,
    OscQueryRuntime,
)

from .control_application import SettingsBackedOscControlApplication
from .control_router import OscControlRouter
from .noop_query import NoopOscQueryService
from .state_publisher import OscCanonicalState, OscStatePublisher


class OscControlIntegrationOwner:
    def __init__(
        self,
        *,
        receiver_owner: VrcMicSyncOwner,
        settings_provider: Callable[[], object | None],
        apply_settings: Callable[[object], object],
        application_provider: Callable[[], object | None],
        sender_provider: Callable[[], object | None],
        state_provider: Callable[[], OscCanonicalState],
        language_state_provider: Callable[[], tuple[str, str, str, str]],
        translation_model_normalizer: Callable[[object], object],
        query_service: OscQueryServicePort | None = None,
        error_sink: Callable[[str], None] | None = None,
        resync_timeout_seconds: float = 1.5,
    ) -> None:
        self._receiver_owner = receiver_owner
        self._settings_provider = settings_provider
        self._application_provider = application_provider
        self._sender_provider = sender_provider
        self._state_provider = state_provider
        self._error_sink = error_sink
        self._mode: OscConnectionMode = "off"
        self._send_port = 9000
        self._receive_port = 9001
        self._host = "127.0.0.1"
        self._configured_connection: tuple[str, OscConnectionMode, int, int] | None = None
        self._accepting_ingress = True
        self._closed = False
        self._resync_timeout_seconds = float(resync_timeout_seconds)
        self._resync_generation = 0
        self._resync_deadline = 0.0
        self._resync_unsettled: set[str] = set()
        self._resync_ready_generation: int | None = None
        self._publisher: OscStatePublisher | None = None
        self._avatar_parameter_diagnostics: dict[str, object] = {}
        self._avatar_sequence = 0
        self._automatic_query_generation = 0
        self._automatic_query_task: asyncio.Task[None] | None = None
        self._scope = LifecycleScope("OscControlIntegrationOwner")
        self._apply_settings_callback = apply_settings

        application = SettingsBackedOscControlApplication(
            settings_provider=settings_provider,
            apply_settings=self._apply_settings,
            translation_model_normalizer=translation_model_normalizer,
            set_self_capture_command=self._application_command("set_stt_enabled"),
            set_peer_capture_command=self._application_command("set_peer_translation_enabled"),
            set_translation_command=self._application_command("set_translation_enabled"),
            set_captions_command=self._application_command("set_overlay_enabled"),
        )
        self.router = OscControlRouter(
            application,
            language_state_provider=language_state_provider,
            echo_suppression_provider=self._is_echo,
            canonical_state_republisher=self._publish_delta,
            canonical_state_full_republisher=self._publish_full,
            error_sink=error_sink,
        )
        self._receiver_owner.set_packet_handlers(
            control_packet_handler=self._handle_control_packet,
            avatar_change_handler=self._handle_avatar_change,
        )
        service = query_service or NoopOscQueryService()
        self.query_runtime = OscQueryRuntime(
            service=service,
            receiver_start=self._receiver_owner.ensure_receiver,
            receiver_stop=self._receiver_owner.stop_receiver,
            receiver_effective_port=lambda: self._receiver_owner.effective_port,
            sender_destination_changed=self._set_sender_destination,
            snapshot_publisher=self._publish_snapshot,
            resync_starter=self._begin_query_resync,
            resync_generation_provider=self._current_resync_generation,
            avatar_inspector=self._inspect_avatar_tree,
        )

    @property
    def last_enabled(self) -> bool | None:
        return self._receiver_owner.last_enabled

    @property
    def receiver(self) -> object | None:
        return self._receiver_owner.receiver

    @property
    def runtime(self) -> object | None:
        return self._receiver_owner.runtime

    @property
    def effective_receive_port(self) -> int:
        return self._receiver_owner.effective_port

    @property
    def connection_mode(self) -> OscConnectionMode:
        return self._mode

    @property
    def effective_send_port(self) -> int | None:
        if self._mode == "automatic":
            return self.query_runtime.effective_send_port or VRCHAT_OSC_DEFAULT_INPUT_PORT
        return self._send_port

    @property
    def accepting_ingress(self) -> bool:
        return self._accepting_ingress

    @property
    def avatar_parameter_diagnostics(self) -> Mapping[str, object]:
        return dict(self._avatar_parameter_diagnostics)

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": "OscControlIntegrationOwner",
            "resource_fields": (
                "router",
                "query_runtime",
                "_publisher",
                "_scope",
            ),
            "stop_ingress": "reject OSC packets and configuration changes",
            "shutdown_policy": "stop discovery, receiver, publisher, and command worker",
            "late_callback_rule": "receiver owner generation and closed router reject late work",
        }

    def stop_ingress(self) -> None:
        self._accepting_ingress = False
        self.router.set_ingress_enabled(False)
        self._receiver_owner.stop_ingress()

    async def configure(self, *, enabled: bool) -> None:
        if not self._accepting_ingress:
            return
        settings = self._settings_provider()
        if settings is not None:
            key = (
                settings.osc.host,
                settings.osc.connection_mode,
                int(settings.osc.send_port or settings.osc.port),
                int(settings.osc.receive_port),
            )
            if self._configured_connection != key:
                await self.configure_connection(
                    mode=key[1],
                    send_port=key[2],
                    receive_port=key[3],
                    host=key[0],
                )
            elif key[1] != "off":
                self._ensure_publisher()
        await self._receiver_owner.configure(enabled=enabled)
        if self._mode != "off" and not self._automatic_query_inflight():
            self._publish_delta()

    async def configure_connection(
        self,
        *,
        mode: OscConnectionMode,
        send_port: int,
        receive_port: int,
        host: str = "127.0.0.1",
    ) -> None:
        if not self._accepting_ingress:
            return
        if mode not in {"automatic", "manual", "off"}:
            raise ValueError(f"unsupported OSC connection mode: {mode!r}")
        if not 1 <= int(send_port) <= 65535:
            raise ValueError("OSC send port must be in 1..65535")
        if not 1 <= int(receive_port) <= 65535:
            raise ValueError("OSC receive port must be in 1..65535")

        key = (host, mode, int(send_port), int(receive_port))
        if self._configured_connection == key and mode == "off":
            await self.router.suspend_ingress()
            if self._publisher is not None:
                self._publisher.close()
            return
        if self._configured_connection == key and (
            mode == "manual" or self.query_runtime.started or self._automatic_query_inflight()
        ):
            self._ensure_publisher()
            self._publish_delta()
            return

        await self.router.suspend_ingress()
        await self._cancel_automatic_query_start()
        await self.query_runtime.stop()
        self._send_port = int(send_port)
        self._receive_port = int(receive_port)
        self._host = host
        self._mode = mode
        self._configured_connection = (host, mode, self._send_port, self._receive_port)
        resync_generation = self._begin_resync() if mode != "off" else None
        receiver_port = 0 if mode == "automatic" else self._receive_port
        await self._receiver_owner.configure_control(
            active=mode != "off",
            host=host,
            port=receiver_port,
            force_restart=True,
        )
        if mode == "off":
            if self._publisher is not None:
                self._publisher.close()
            return

        self._ensure_publisher()
        if mode == "manual":
            await self._set_sender_destination(host, self._send_port)
            self._publish_start(resync_generation)
            self.router.set_ingress_enabled(True)
            return

        self.router.set_ingress_enabled(True)
        self._schedule_automatic_query_start(resync_generation)

    def _automatic_query_inflight(self) -> bool:
        task = self._automatic_query_task
        return task is not None and not task.done()

    def _schedule_automatic_query_start(self, resync_generation: int | None) -> None:
        self._automatic_query_generation += 1
        generation = self._automatic_query_generation
        self._automatic_query_task = start_lifecycle_task(
            self._scope,
            self._run_automatic_query_start(resync_generation, generation),
            name=f"oscquery-automatic-start-{generation}",
        )

    async def _run_automatic_query_start(
        self,
        resync_generation: int | None,
        generation: int,
    ) -> None:
        try:
            await self.query_runtime.start(
                "automatic",
                snapshot_generation=resync_generation,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if (
                generation != self._automatic_query_generation
                or not self._accepting_ingress
                or self._closed
            ):
                return
            self._report_error(f"OSCQuery automatic discovery unavailable: {type(exc).__name__}")
            await self._receiver_owner.ensure_receiver()
            await self._set_sender_destination(self._host, VRCHAT_OSC_DEFAULT_INPUT_PORT)
            self._publish_start(resync_generation)
        finally:
            if self._automatic_query_task is asyncio.current_task():
                self._automatic_query_task = None

    async def _cancel_automatic_query_start(self) -> None:
        task = self._automatic_query_task
        self._automatic_query_task = None
        self._automatic_query_generation += 1
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def wait_automatic_query_start(self) -> None:
        task = self._automatic_query_task
        if task is not None:
            await task

    async def close(self) -> None:
        if self._closed and self._receiver_owner.receiver is None:
            return
        self.stop_ingress()
        self._closed = True
        await self.wait_automatic_query_start()
        await self._scope.close()
        await self.query_runtime.stop()
        await self.router.close()
        if self._publisher is not None:
            self._publisher.close()
        await self._receiver_owner.close()

    def _application_command(
        self,
        method_name: str,
    ) -> Callable[[bool], object]:
        async def command(value: bool) -> object:
            application = self._application_provider()
            method = getattr(application, method_name, None) if application is not None else None
            if not callable(method):
                raise RuntimeError(f"OSC application command is not wired: {method_name}")
            result = await method(value)
            if not self._dashboard_command_matches_canonical_state(method_name, value):
                return False
            self._publish_delta()
            return result

        return command

    async def _apply_settings(self, settings: object) -> object:
        result = self._apply_settings_callback(settings)
        if inspect.isawaitable(result):
            return await result
        return result

    def _dashboard_command_matches_canonical_state(self, method_name: str, value: bool) -> bool:
        field_by_method = {
            "set_stt_enabled": "self_capture",
            "set_peer_translation_enabled": "peer_capture",
            "set_translation_enabled": "translation",
            "set_overlay_enabled": "captions",
        }
        field_name = field_by_method.get(method_name)
        if field_name is None:
            return True
        try:
            return bool(getattr(self._state_provider(), field_name)) is bool(value)
        except Exception:
            return False

    async def _set_sender_destination(self, host: str, port: int) -> None:
        sender = self._sender_provider()
        if sender is None:
            return
        setter = getattr(sender, "set_destination", None)
        if not callable(setter):
            raise RuntimeError("VRChat OSC sender does not support destination changes")
        setter(host, int(port))

    def _ensure_publisher(self) -> OscStatePublisher | None:
        sender = self._sender_provider()
        if sender is None:
            return None
        if self._publisher is None:
            self._publisher = OscStatePublisher(sender)
        return self._publisher

    def _is_echo(self, message: object) -> bool:
        publisher = self._publisher
        if publisher is None:
            return False
        parameter = getattr(message, "name", None)
        value = getattr(message, "value", None)
        return (
            isinstance(parameter, str)
            and isinstance(value, (bool, int))
            and publisher.is_echo(
                parameter,
                value,
            )
        )

    def _begin_resync(self) -> int:
        self._resync_generation += 1
        self._resync_deadline = time.monotonic() + self._resync_timeout_seconds
        self._resync_unsettled = set(OSC_PARAMETER_DEFINITIONS)
        self._resync_ready_generation = None
        return self._resync_generation

    def _current_resync_generation(self) -> int:
        return self._resync_generation

    def _begin_query_resync(
        self,
        _reason: str,
        parent_generation: int | None,
    ) -> int | None:
        if not self._accepting_ingress or self._mode != "automatic":
            return None
        if parent_generation is not None and parent_generation != self._resync_generation:
            return None
        return self._begin_resync()

    def _expire_resync_if_needed(self) -> None:
        if self._resync_unsettled and time.monotonic() >= self._resync_deadline:
            self._resync_unsettled.clear()

    def _mark_resync_ready(self, generation: int | None = None) -> None:
        expected_generation = self._resync_generation if generation is None else generation
        if expected_generation == self._resync_generation:
            self._resync_ready_generation = expected_generation

    def _publish_start(self, generation: int | None = None) -> None:
        if self._mode == "off":
            return
        publisher = self._ensure_publisher()
        if publisher is None:
            return
        try:
            publisher.start(self._state_provider())
        except Exception:
            return
        self._mark_resync_ready(generation)

    def _publish_delta(self) -> None:
        if self._mode == "off":
            return
        publisher = self._ensure_publisher()
        if publisher is None:
            return
        with contextlib.suppress(Exception):
            publisher.publish_state(self._state_provider())

    def publish_delta(self) -> None:
        self._publish_delta()

    def _publish_full(self) -> None:
        if self._mode == "off":
            return
        publisher = self._ensure_publisher()
        if publisher is None:
            return
        try:
            publisher.publish_full(self._state_provider())
        except Exception:
            return

    async def _publish_snapshot(
        self,
        reason: str,
        generation: int | None = None,
    ) -> None:
        if not self._accepting_ingress or self._mode == "off":
            return
        if generation is not None and generation != self._resync_generation:
            return
        publisher = self._ensure_publisher()
        if publisher is None:
            return
        try:
            state = self._state_provider()
            if reason == "avatar_change":
                publisher.on_avatar_change(state)
            elif reason == "discovery":
                publisher.on_discovery(state)
            else:
                publisher.start(state)
        except Exception as exc:
            self._report_error(f"OSC state publication failed: {type(exc).__name__}")
            return
        self._mark_resync_ready(generation)

    async def _inspect_avatar_tree(self, tree: object) -> None:
        parameters = _extract_avatar_parameters(tree)
        missing = tuple(name for name in OSC_PARAMETER_DEFINITIONS if name not in parameters)
        type_mismatches = tuple(
            name
            for name, definition in OSC_PARAMETER_DEFINITIONS.items()
            if name in parameters
            and isinstance(parameters[name], Mapping)
            and parameters[name].get("TYPE") != ("T" if definition.value_type == "bool" else "i")
        )
        self._avatar_parameter_diagnostics = {
            "present": tuple(name for name in OSC_PARAMETER_DEFINITIONS if name in parameters),
            "missing": missing,
            "type_mismatches": type_mismatches,
        }

    def _handle_control_packet(self, address: str, values: tuple[object, ...]) -> bool:
        if self._mode == "off":
            return False
        self._expire_resync_if_needed()
        try:
            message = decode_control_message(address, *values)
        except OscControlCodecError:
            return self.router.handle_packet(address, *values)
        if message.name not in self._resync_unsettled:
            return self.router.handle_packet(address, *values)
        if self._resync_ready_generation != self._resync_generation:
            return False
        try:
            canonical = OscStatePublisher.value_for_state(
                self._state_provider(),
                message.name,
            )
        except Exception:
            return False
        if OscStatePublisher.values_equal(message.name, message.value, canonical):
            self._resync_unsettled.discard(message.name)
        return False

    def _handle_avatar_change(self, _values: tuple[object, ...]) -> None:
        if not self._accepting_ingress or self._mode == "off":
            return
        generation = self._begin_resync()
        try:
            self._avatar_sequence += 1
            start_lifecycle_task(
                self._scope,
                self.query_runtime.on_avatar_change(snapshot_generation=generation),
                name=f"avatar-change-{self._avatar_sequence}",
            )
        except RuntimeError:
            return

    def _report_error(self, message: str) -> None:
        if self._error_sink is not None:
            self._error_sink(message)


def _extract_avatar_parameters(tree: object) -> Mapping[str, object]:
    if not isinstance(tree, Mapping):
        return {}
    direct = tree.get("parameters")
    if isinstance(direct, Mapping):
        contents = direct.get("CONTENTS", direct)
        if isinstance(contents, Mapping):
            return contents
    contents = tree.get("CONTENTS")
    if isinstance(contents, Mapping):
        direct = contents.get("parameters")
        if isinstance(direct, Mapping):
            nested = direct.get("CONTENTS", direct)
            if isinstance(nested, Mapping):
                return nested
        for child in contents.values():
            parameters = _extract_avatar_parameters(child)
            if parameters:
                return parameters
    return {}


__all__ = ["OscControlIntegrationOwner"]
