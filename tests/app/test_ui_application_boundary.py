from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.ui_application import UiApplicationBoundary


class RecordingBackend:
    def __init__(self) -> None:
        self.config_path = Path("settings.json")
        self.runtime_logging_mode = "detailed"
        self.settings = SimpleNamespace(
            ui=SimpleNamespace(peer_translation_eula_accepted=False),
            provider=SimpleNamespace(llm=SimpleNamespace(value="local_llm")),
            overlay=SimpleNamespace(target=SimpleNamespace(value="desktop")),
            api_key_verified=SimpleNamespace(openrouter=True),
        )
        self.hub = SimpleNamespace(
            translation_enabled=True,
            llm=object(),
            stt_session_state=lambda channel: f"{channel}-listening",
        )
        self.microphone_test_active = True
        self.desktop_overlay_captions_locked = True
        self.last_discord_managed_auth_referral_bonus_applied = True
        self.events: list[tuple[object, ...]] = []

    async def start(self) -> None:
        self.events.append(("start",))

    async def stop(self) -> None:
        self.events.append(("stop",))

    async def submit_text(self, text: str) -> None:
        self.events.append(("submit", text))

    async def set_translation_enabled(self, enabled: bool) -> bool:
        self.events.append(("translation", enabled))
        return enabled

    async def set_peer_translation_enabled(self, enabled: bool) -> bool:
        self.events.append(("peer", enabled))
        return enabled

    async def apply_settings(self, settings: object) -> None:
        self.settings = settings
        self.events.append(("settings", settings))

    async def apply_providers(self, *args, **kwargs) -> None:
        self.events.append(("providers", args, kwargs))

    async def apply_telemetry_consent(self, consent: str) -> object:
        self.events.append(("telemetry", consent))
        return self.settings

    def persist_settings(self) -> None:
        self.events.append(("persist",))

    def clear_provider_verification(self, provider: str) -> None:
        setattr(self.settings.api_key_verified, provider, False)
        self.events.append(("clear-verification", provider))


def test_state_is_a_semantic_snapshot_without_exposing_backend_objects() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    state = boundary.state()

    assert state.config_path == Path("settings.json")
    assert state.runtime_logging_mode == "detailed"
    assert state.translation_enabled is True
    assert state.stt_state == "self-listening"
    assert state.peer_translation_eula_accepted is False
    assert state.microphone_test_active is True
    assert state.provider_name == "local_llm"
    assert state.overlay_target == "desktop"
    assert state.desktop_overlay_captions_locked is True
    assert state.managed_auth_referral_bonus_applied is True
    assert not hasattr(state, "hub")
    assert not hasattr(state, "settings")


def test_compatibility_settings_is_detached_and_missing_ui_state_stays_unknown() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    detached = boundary.compatibility_settings()
    detached.ui.peer_translation_eula_accepted = True

    assert backend.settings.ui.peer_translation_eula_accepted is False
    backend.settings = None
    assert boundary.state().peer_translation_eula_accepted is None


@pytest.mark.asyncio
async def test_primary_intents_delegate_once_and_preserve_results() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    await boundary.start()
    await boundary.submit_text("hello")
    result = await boundary.set_translation_enabled(True)
    await boundary.stop()

    assert result is True
    assert backend.events == [
        ("start",),
        ("submit", "hello"),
        ("translation", True),
        ("stop",),
    ]


@pytest.mark.asyncio
async def test_eula_acceptance_is_owned_at_the_boundary_before_peer_enable() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    result = await boundary.accept_peer_translation_eula_and_enable()

    assert result is True
    assert backend.events[0][0] == "settings"
    assert backend.settings.ui.peer_translation_eula_accepted is True
    assert backend.events[1] == ("peer", True)


@pytest.mark.asyncio
async def test_provider_apply_preserves_no_argument_and_forced_rebuild_contracts() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    await boundary.apply_providers()
    await boundary.apply_providers(force_rebuild_llm=True)
    pending = object()
    await boundary.apply_providers(pending)

    assert backend.events == [
        ("providers", (), {}),
        ("providers", (), {"force_rebuild_llm": True}),
        ("providers", (pending,), {}),
    ]


@pytest.mark.asyncio
async def test_telemetry_and_verification_mutations_stay_behind_named_intents() -> None:
    backend = RecordingBackend()
    boundary = UiApplicationBoundary(backend)

    returned = await boundary.apply_telemetry_consent("allow")
    boundary.clear_provider_verification("openrouter")

    assert returned is backend.settings
    assert backend.settings.api_key_verified.openrouter is False
    assert backend.events == [
        ("telemetry", "allow"),
        ("clear-verification", "openrouter"),
    ]


@pytest.mark.asyncio
async def test_boundary_owns_managed_auth_task_cancellation_and_terminal_close() -> None:
    boundary = UiApplicationBoundary(RecordingBackend())
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def auth_task() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    handle = boundary.start_managed_auth_task(
        task_runner=lambda factory: asyncio.create_task(factory()),
        task_factory=auth_task,
        task_name="discord-managed-auth-dialog",
        generation=1,
    )
    await started.wait()

    assert boundary.managed_auth_task_names() == ("discord-managed-auth-dialog",)
    await boundary.close_managed_auth_tasks()
    await asyncio.gather(handle, return_exceptions=True)

    assert handle.cancelled() is True
    assert cancelled.is_set() is True
    assert boundary.managed_auth_tasks_open() is False


@pytest.mark.asyncio
async def test_boundary_owns_github_prompt_generation_and_cancellation() -> None:
    boundary = UiApplicationBoundary(RecordingBackend())
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def prompt(generation: int) -> bool:
        assert boundary.is_current_github_star_prompt_generation(generation)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return True

    task = boundary.start_github_star_prompt(prompt)
    await started.wait()
    boundary.stop_github_star_prompt_ingress()
    await boundary.close_github_star_prompt_runtime()

    assert await task is False
    assert cancelled.is_set() is True
