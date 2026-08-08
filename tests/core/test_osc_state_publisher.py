from __future__ import annotations

import pytest

from puripuly_heart.app.services.osc.state_publisher import (
    OscCanonicalState,
    OscStatePublisher,
    state_from_settings,
)
from puripuly_heart.config.settings import AppSettings, TranslationConnection, TranslationModel


class FakeSender:
    def __init__(self) -> None:
        self.messages: list[tuple[str, object]] = []

    def send_message(self, address: str, *values: object) -> None:
        self.messages.append((address, values[0] if len(values) == 1 else tuple(values)))

    def send_chatbox(self, text: str) -> None:
        raise AssertionError(text)

    def send_typing(self, is_typing: bool) -> None:
        raise AssertionError(is_typing)


def test_state_publisher_sends_full_snapshot_then_only_deltas() -> None:
    sender = FakeSender()
    publisher = OscStatePublisher(sender)
    state = OscCanonicalState()

    full = publisher.start(state)
    assert len(full) == 15
    assert len(sender.messages) == 15
    assert publisher.is_echo("PuriPuly_Trans", False) is True
    assert publisher.is_echo("PuriPuly_Trans", True) is False

    changed = publisher.publish_delta(
        OscCanonicalState(translation=True, mute_sync=True, fallback="none")
    )
    assert {item.parameter for item in changed} == {"PuriPuly_Trans", "PuriPuly_MuteSync"}


def test_state_publisher_full_snapshot_republishes_after_discovery() -> None:
    sender = FakeSender()
    publisher = OscStatePublisher(sender)
    state = OscCanonicalState()

    publisher.start(state)
    publisher.on_avatar_change(state)

    assert len(sender.messages) == 30


@pytest.mark.parametrize(
    ("model", "connection", "expected"),
    [
        (
            TranslationModel.DEEPSEEK_V4_FLASH,
            TranslationConnection.OFFICIAL_BYOK,
            "deepseek_v4_flash_official",
        ),
        (
            TranslationModel.DEEPSEEK_V4_FLASH,
            TranslationConnection.OPENROUTER,
            "openrouter_deepseek_v4_flash",
        ),
        (TranslationModel.GEMMA4, TranslationConnection.OPENROUTER, "openrouter_gemma4_26b_a4b"),
        (
            TranslationModel.GEMMA4_26B_31B,
            TranslationConnection.OPENROUTER,
            "openrouter_gemma4_26b_31b",
        ),
        (TranslationModel.GEMMA4_31B, TranslationConnection.OPENROUTER, "openrouter_gemma4_31b"),
        (TranslationModel.GEMMA4_26B_31B, TranslationConnection.MANAGED, "managed_gemma4_26b_31b"),
        (TranslationModel.GEMMA4_31B, TranslationConnection.MANAGED, "managed_gemma4_31b"),
        (
            TranslationModel.GEMMA4_31B_CEREBRAS,
            TranslationConnection.OFFICIAL_BYOK,
            "cerebras_gemma4_31b",
        ),
    ],
)
def test_state_from_settings_publishes_each_fallback_alias(
    model: TranslationModel,
    connection: TranslationConnection,
    expected: str,
) -> None:
    settings = AppSettings()
    settings.translation.fallback.enabled = True
    settings.translation.fallback.model = model
    settings.translation.fallback.connection = connection

    assert state_from_settings(settings).fallback == expected
