from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("flet")

from puripuly_heart.ui.components.settings.api_key_field import ApiKeyField
from puripuly_heart.ui.components.settings.api_key_verification_controller import (
    ApiKeyVerificationController,
)


@pytest.mark.asyncio
async def test_api_key_field_default_status_mode_verifies_saved_value() -> None:
    saved: list[tuple[str, str]] = []
    verified: list[tuple[str, str]] = []

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        return True, "ok"

    field = ApiKeyField(
        "settings.api_keys.openrouter",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_save=lambda key, value: saved.append((key, value)),
    )

    field._text_field.value = "provider-secret"
    field._handle_change(None)
    field._handle_blur(None)
    await field._run_verification()

    assert len(field.controls) == 2
    assert saved == [("openrouter_api_key", "provider-secret")]
    assert verified == [("openrouter", "provider-secret")]
    assert field._current_status == "success"
    assert field.controller.last_verified_hash == ApiKeyVerificationController.key_hash(
        "provider-secret"
    )


@pytest.mark.asyncio
async def test_api_key_field_default_status_mode_verifies_unchanged_loaded_value() -> None:
    saved: list[tuple[str, str]] = []
    verified: list[tuple[str, str]] = []

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        return True, "ok"

    field = ApiKeyField(
        "settings.api_keys.openrouter",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_save=lambda key, value: saved.append((key, value)),
    )

    field.value = "loaded-secret"
    field._handle_blur(None)
    await field._run_verification()

    assert saved == []
    assert verified == [("openrouter", "loaded-secret")]
    assert field._current_status == "success"
    assert field.controller.last_verified_hash == ApiKeyVerificationController.key_hash(
        "loaded-secret"
    )


@pytest.mark.asyncio
async def test_api_key_field_verifies_latest_edit_after_blur_during_inflight_verification() -> None:
    saved: list[tuple[str, str]] = []
    verified: list[tuple[str, str]] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        if key == "first-secret":
            first_started.set()
            await release_first.wait()
        return True, "ok"

    field = ApiKeyField(
        "settings.api_keys.openrouter",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_save=lambda key, value: saved.append((key, value)),
    )

    field._text_field.value = "first-secret"
    field._handle_change(None)
    field._handle_blur(None)
    verification_task = asyncio.create_task(field._run_verification())
    await first_started.wait()

    field._text_field.value = "second-secret"
    field._handle_change(None)
    field._handle_blur(None)

    release_first.set()
    await verification_task

    assert saved == [
        ("openrouter_api_key", "first-secret"),
        ("openrouter_api_key", "second-secret"),
    ]
    assert verified == [
        ("openrouter", "first-secret"),
        ("openrouter", "second-secret"),
    ]
    assert field._current_status == "success"
    assert field.controller.last_verified_hash == ApiKeyVerificationController.key_hash(
        "second-secret"
    )


def test_api_key_field_can_hide_status_and_skip_verification() -> None:
    saved: list[tuple[str, str]] = []
    verified: list[tuple[str, str]] = []

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        return True, "ok"

    field = ApiKeyField(
        "settings.local_llm.api_key",
        "local_llm_api_key",
        "local_llm",
        on_verify=verify,
        on_save=lambda key, value: saved.append((key, value)),
        show_status=False,
    )

    field._text_field.value = "local-secret"
    field._handle_change(None)
    field._handle_blur(None)

    assert len(field.controls) == 1
    assert saved == [("local_llm_api_key", "local-secret")]
    assert verified == []
    assert not field.controller.has_pending


def test_api_key_field_does_not_save_unchanged_loaded_value() -> None:
    saved: list[tuple[str, str]] = []
    field = ApiKeyField(
        "settings.local_llm.api_key",
        "local_llm_api_key",
        "local_llm",
        on_save=lambda key, value: saved.append((key, value)),
        show_status=False,
    )

    field.value = "loaded-secret"
    field._handle_blur(None)

    assert saved == []


@pytest.mark.asyncio
async def test_api_key_field_does_not_verify_when_secret_save_fails() -> None:
    verified: list[tuple[str, str]] = []

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        return True, "ok"

    field = ApiKeyField(
        "settings.api_keys.openrouter",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_save=lambda _key, _value: False,
    )
    field._text_field.value = "unsaved-secret"
    field._handle_change(None)

    field._handle_blur(None)
    await field._run_verification()

    assert verified == []
    assert field._current_status == "error"
    assert field.controller.last_verified_hash == ""
    assert not field.controller.has_pending


@pytest.mark.asyncio
async def test_api_key_field_awaits_secret_transaction_before_verification() -> None:
    events: list[str] = []
    release_save = asyncio.Event()

    async def save(_key: str, _value: str) -> bool:
        events.append("save-start")
        await release_save.wait()
        events.append("save-complete")
        return True

    async def verify(_provider: str, _key: str) -> tuple[bool, str]:
        events.append("verify")
        return True, "ok"

    field = ApiKeyField(
        "settings.api_keys.openrouter",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_save=save,
    )
    field._text_field.value = "new-secret"
    field._handle_change(None)
    field._handle_blur(None)
    task = asyncio.create_task(field._run_verification())
    await asyncio.sleep(0)

    assert events == ["save-start"]
    release_save.set()
    await task

    assert events == ["save-start", "save-complete", "verify"]


@pytest.mark.asyncio
async def test_controller_verifies_and_reports_success() -> None:
    saved: list[tuple[str, str]] = []
    verified: list[tuple[str, str]] = []
    statuses: list[str] = []
    messages: list[tuple[str, str]] = []
    current = {"value": "provider-secret"}

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        return True, "ok"

    controller = ApiKeyVerificationController(
        secret_key="openrouter_api_key",
        provider="openrouter",
        on_verify=verify,
        on_save=lambda key, value: saved.append((key, value)),
        on_status=statuses.append,
        on_message=lambda key, msg: messages.append((key, msg)),
    )
    controller.set_value_getter(lambda: current["value"])

    controller.notify_edit()
    controller.handle_blur("provider-secret")
    drain = controller.run_pending()
    assert drain is not None
    await drain

    assert saved == [("openrouter_api_key", "provider-secret")]
    assert verified == [("openrouter", "provider-secret")]
    assert controller.status == "success"
    assert controller.last_verified_hash == controller.key_hash("provider-secret")
    assert statuses[-1] == "success"
    assert messages[0][0] == "snackbar.verification_ok"


@pytest.mark.asyncio
async def test_controller_ignores_stale_result_when_value_changed_midflight() -> None:
    verified: list[tuple[str, str]] = []
    statuses: list[str] = []
    current = {"value": "stale-secret"}

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        current["value"] = "edited-secret"
        return False, "401 unauthorized"

    controller = ApiKeyVerificationController(
        secret_key="openrouter_api_key",
        provider="openrouter",
        on_verify=verify,
        on_status=statuses.append,
    )
    controller.set_value_getter(lambda: current["value"])

    controller.handle_blur("stale-secret")
    drain = controller.run_pending()
    assert drain is not None
    await drain

    assert verified == [("openrouter", "stale-secret")]
    assert controller.status == "verifying"
    assert controller.last_verified_hash == ""
