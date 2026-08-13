from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.osc.control_router import OscControlRouter


class FakeApplication:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.gate = asyncio.Event()
        self.started = asyncio.Event()

    async def set_self_capture(self, enabled: bool) -> object:
        self.calls.append(("self_capture", enabled))
        return None

    async def set_peer_capture(self, enabled: bool) -> object:
        self.calls.append(("peer_capture", enabled))
        return None

    async def set_translation(self, enabled: bool) -> object:
        self.calls.append(("translation", enabled))
        return None

    async def set_captions(self, enabled: bool) -> object:
        self.calls.append(("captions", enabled))
        return None

    async def set_languages(self, **kwargs: str) -> object:
        self.calls.append(("languages", tuple(kwargs.values())))
        return None

    async def set_peer_auto_detect(self, enabled: bool) -> object:
        self.calls.append(("peer_auto", enabled))
        return None

    async def set_self_asr(self, provider: str) -> object:
        self.started.set()
        await self.gate.wait()
        self.calls.append(("self_asr", provider))
        return None

    async def set_peer_asr(self, provider: str) -> object:
        self.calls.append(("peer_asr", provider))
        return None

    async def set_translation_model(self, model: str) -> object:
        self.calls.append(("model", model))
        return None

    async def set_fallback(self, alias: str) -> object:
        self.calls.append(("fallback", alias))
        return None

    async def set_mute_sync(self, enabled: bool) -> object:
        self.calls.append(("mute_sync", enabled))
        return None

    async def set_chatbox_source(self, enabled: bool) -> object:
        self.calls.append(("chatbox_source", enabled))


class RejectingApplication(FakeApplication):
    def __init__(self, result: object) -> None:
        super().__init__()
        self.result = result

    async def set_translation(self, enabled: bool) -> object:
        self.calls.append(("translation", enabled))
        return self.result
        return None


@pytest.mark.asyncio
async def test_router_routes_boolean_and_language_controls() -> None:
    application = FakeApplication()
    router = OscControlRouter(application)

    assert await router.dispatch_packet("/avatar/parameters/PuriPuly_Talk", True)
    await router.dispatch_packet("/avatar/parameters/PuriPuly_SelfSrcLang", 16)

    assert application.calls == [
        ("self_capture", True),
        ("languages", ("ja", "en", "en", "ko")),
    ]
    await router.close()


@pytest.mark.asyncio
async def test_router_publishes_canonical_delta_after_successful_application() -> None:
    application = FakeApplication()
    delta_calls = 0

    def publish_delta() -> None:
        nonlocal delta_calls
        delta_calls += 1

    router = OscControlRouter(
        application,
        canonical_state_republisher=publish_delta,
    )

    result = await router.dispatch_packet("/avatar/parameters/PuriPuly_Trans", True)

    assert result.applied is True
    assert delta_calls == 1
    await router.close()


@pytest.mark.asyncio
async def test_router_routes_the_complete_public_control_matrix() -> None:
    application = FakeApplication()
    application.gate.set()
    router = OscControlRouter(application)

    packets = [
        ("PuriPuly_Talk", True),
        ("PuriPuly_Listen", True),
        ("PuriPuly_Trans", True),
        ("PuriPuly_Captions", True),
        ("PuriPuly_PeerAuto", True),
        ("PuriPuly_MuteSync", True),
        ("PuriPuly_ChatboxSource", True),
        ("PuriPuly_SelfSrcLang", 16),
        ("PuriPuly_SelfDstLang", 7),
        ("PuriPuly_PeerSrcLang", 7),
        ("PuriPuly_PeerDstLang", 17),
        ("PuriPuly_SelfASR", 7),
        ("PuriPuly_PeerASR", 1),
        ("PuriPuly_Translator", 5),
        ("PuriPuly_Fallback", 0),
    ]

    for name, value in packets:
        result = await router.dispatch_packet(f"/avatar/parameters/{name}", value)
        assert result.applied is True

    assert [name for name, _value in application.calls] == [
        "self_capture",
        "peer_capture",
        "translation",
        "captions",
        "peer_auto",
        "mute_sync",
        "chatbox_source",
        "languages",
        "languages",
        "languages",
        "languages",
        "self_asr",
        "peer_asr",
        "model",
        "fallback",
    ]
    assert application.calls[-8:] == [
        ("languages", ("ja", "en", "en", "ko")),
        ("languages", ("ja", "en", "en", "ko")),
        ("languages", ("ja", "en", "en", "ko")),
        ("languages", ("ja", "en", "en", "ko")),
        ("self_asr", "soniox"),
        ("peer_asr", "local_parakeet_v3"),
        ("model", "gemini3_flash"),
        ("fallback", "none"),
    ]
    await router.close()


@pytest.mark.asyncio
async def test_router_coalesces_superseded_expensive_controls() -> None:
    application = FakeApplication()
    router = OscControlRouter(application)

    first = asyncio.create_task(router.dispatch_packet("/avatar/parameters/PuriPuly_SelfASR", 1))
    await asyncio.sleep(0)
    second = asyncio.create_task(router.dispatch_packet("/avatar/parameters/PuriPuly_SelfASR", 7))
    await asyncio.sleep(0)
    application.gate.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.applied is True or first_result.superseded is True
    assert second_result.applied is True
    assert application.calls[-1] == ("self_asr", "soniox")
    await router.close()


@pytest.mark.asyncio
async def test_router_suppresses_values_echoed_from_its_publisher() -> None:
    application = FakeApplication()
    router = OscControlRouter(
        application,
        echo_suppression_provider=lambda message: message.name == "PuriPuly_Talk"
        and message.value is True,
    )

    result = await router.dispatch_packet("/avatar/parameters/PuriPuly_Talk", True)

    assert result.applied is False
    assert result.error == "echo_suppressed"
    assert application.calls == []
    await router.close()


@pytest.mark.asyncio
async def test_echo_suppression_does_not_drop_a_newer_boolean_reversal() -> None:
    class BlockingBooleanApplication(FakeApplication):
        async def set_self_capture(self, enabled: bool) -> object:
            self.started.set()
            await self.gate.wait()
            self.calls.append(("self_capture", enabled))
            return None

    application = BlockingBooleanApplication()
    router = OscControlRouter(
        application,
        echo_suppression_provider=lambda message: message.name == "PuriPuly_Talk"
        and message.value is False,
    )
    first = asyncio.create_task(router.dispatch_packet("/avatar/parameters/PuriPuly_Talk", True))
    await application.started.wait()
    second = asyncio.create_task(router.dispatch_packet("/avatar/parameters/PuriPuly_Talk", False))
    await asyncio.sleep(0)
    application.gate.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.applied is True
    assert second_result.applied is True
    assert application.calls == [
        ("self_capture", True),
        ("self_capture", False),
    ]
    await router.close()


@pytest.mark.asyncio
async def test_router_republishes_canonical_state_after_invalid_id() -> None:
    application = FakeApplication()
    delta_republish_calls = 0
    full_republish_calls = 0

    def republish_delta() -> None:
        nonlocal delta_republish_calls
        delta_republish_calls += 1

    def republish_full() -> None:
        nonlocal full_republish_calls
        full_republish_calls += 1

    router = OscControlRouter(
        application,
        canonical_state_republisher=republish_delta,
        canonical_state_full_republisher=republish_full,
    )

    assert router.handle_packet("/avatar/parameters/PuriPuly_SelfASR", 99) is False
    assert delta_republish_calls == 0
    assert full_republish_calls == 1
    assert application.calls == []
    await router.close()


@pytest.mark.asyncio
async def test_router_resolves_inflight_coalesced_command_when_closed() -> None:
    application = FakeApplication()
    router = OscControlRouter(application)
    dispatch_task = asyncio.create_task(
        router.dispatch_packet("/avatar/parameters/PuriPuly_SelfASR", 1)
    )
    await application.started.wait()

    await router.close()

    result = await asyncio.wait_for(dispatch_task, timeout=1)
    assert result.applied is False
    assert result.error == "router_closed"


@pytest.mark.asyncio
async def test_router_drains_started_command_after_ingress_is_disabled() -> None:
    application = FakeApplication()
    router = OscControlRouter(application)
    dispatch_task = asyncio.create_task(
        router.dispatch_packet("/avatar/parameters/PuriPuly_SelfASR", 1)
    )
    await application.started.wait()

    router.set_ingress_enabled(False)
    application.gate.set()

    result = await asyncio.wait_for(dispatch_task, timeout=1)
    assert result.applied is True
    assert result.error is None
    assert ("self_asr", "local_parakeet_v3") in application.calls
    await router.close()


@pytest.mark.asyncio
async def test_router_rechecks_generation_after_waiting_for_serial_lock() -> None:
    application = FakeApplication()
    router = OscControlRouter(application)
    first = asyncio.create_task(router.dispatch_packet("/avatar/parameters/PuriPuly_SelfASR", 1))
    await application.started.wait()

    second = asyncio.create_task(router.dispatch_packet("/avatar/parameters/PuriPuly_Talk", True))
    await asyncio.sleep(0)
    router.set_ingress_enabled(False)
    application.gate.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.applied is True
    assert first_result.error is None
    assert second_result.applied is False
    assert second_result.error == "router_disabled"
    assert ("self_asr", "local_parakeet_v3") in application.calls
    assert ("self_capture", True) not in application.calls
    await router.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "application_result",
    [False, SimpleNamespace(status="settings_commit_success_runtime_degraded")],
)
async def test_router_republishes_after_rejected_application_result(
    application_result: object,
) -> None:
    application = RejectingApplication(application_result)
    full_republish_calls = 0

    def republish_full() -> None:
        nonlocal full_republish_calls
        full_republish_calls += 1

    router = OscControlRouter(
        application,
        canonical_state_full_republisher=republish_full,
    )

    result = await router.dispatch_packet("/avatar/parameters/PuriPuly_Trans", True)

    assert result.applied is False
    assert result.error == "application_rejected"
    assert full_republish_calls == 1
    await router.close()


@pytest.mark.asyncio
async def test_router_republishes_after_application_exception() -> None:
    class RaisingApplication(FakeApplication):
        async def set_translation(self, _enabled: bool) -> object:
            raise RuntimeError("application failed")

    application = RaisingApplication()
    full_republish_calls = 0

    def republish_full() -> None:
        nonlocal full_republish_calls
        full_republish_calls += 1

    router = OscControlRouter(
        application,
        canonical_state_full_republisher=republish_full,
    )

    result = await router.dispatch_packet("/avatar/parameters/PuriPuly_Trans", True)

    assert result.applied is False
    assert result.error == "RuntimeError"
    assert full_republish_calls == 1
    await router.close()
