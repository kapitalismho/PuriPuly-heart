from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.app.services.osc.state_publisher import (
    OscCanonicalState,
    OscStatePublisher,
    state_from_settings,
)
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel


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
    assert (
        "/avatar/parameters/PuriPuly_SelfDstLang2",
        255,
    ) in sender.messages
    assert (
        "/avatar/parameters/PuriPuly_ChatboxSource",
        False,
    ) in sender.messages
    assert publisher.is_echo("PuriPuly_Trans", False) is True
    assert publisher.is_echo("PuriPuly_Trans", True) is False

    changed = publisher.publish_delta(OscCanonicalState(translation=True, mute_sync=True))
    assert {item.parameter for item in changed} == {"PuriPuly_Trans", "PuriPuly_MuteSync"}


def test_state_publisher_full_snapshot_republishes_after_discovery() -> None:
    sender = FakeSender()
    publisher = OscStatePublisher(sender)
    state = OscCanonicalState()

    publisher.start(state)
    publisher.on_discovery(state)

    assert len(sender.messages) == 30


@pytest.mark.parametrize(
    ("model", "connection", "expected_id"),
    [
        (TranslationModel.GEMINI_FLASH, TranslationConnection.OFFICIAL_BYOK, 5),
        (TranslationModel.GEMMA4_31B, TranslationConnection.MANAGED, 1),
        (TranslationModel.CUSTOM_HTTP, TranslationConnection.CUSTOM_HTTP, 9),
        (TranslationModel.MANAGED_GEMMA, TranslationConnection.CPU, 10),
        (TranslationModel.MANAGED_GEMMA, TranslationConnection.GPU, 11),
        (TranslationModel.DEEPSEEK_V4_FLASH, TranslationConnection.OPENROUTER, 3),
        (TranslationModel.DEEPSEEK_V4_FLASH_41, TranslationConnection.OFFICIAL_BYOK, 13),
    ],
)
def test_state_publisher_uses_translation_selection_ids(
    model: TranslationModel,
    connection: TranslationConnection,
    expected_id: int,
) -> None:
    baseline = AppSettingsVNext()
    settings = replace(
        baseline,
        intent=replace(
            baseline.intent,
            translation=replace(
                baseline.intent.translation,
                model=model.value,
                connection=connection.value,
            ),
        ),
    )
    state = state_from_settings(settings)
    sender = FakeSender()

    OscStatePublisher(sender).start(state)

    assert ("/avatar/parameters/PuriPuly_Translator", expected_id) in sender.messages


@pytest.mark.parametrize(
    ("provider", "expected_id"),
    [
        (STTProviderName.CUSTOM_OFFLINE, 8),
        (STTProviderName.CUSTOM_REALTIME, 9),
        (STTProviderName.CUSTOM, 8),
        (STTProviderName.QWEN_AUDIO, 12),
        (STTProviderName.GEMINI_TRANSCRIBE, 10),
        (STTProviderName.ELEVENLABS_SCRIBE, 11),
        (STTProviderName.ROLLING_FREE, 13),
    ],
)
def test_state_publisher_publishes_custom_asr_ids(
    provider: STTProviderName,
    expected_id: int,
) -> None:
    baseline = AppSettingsVNext()
    settings = replace(
        baseline,
        intent=replace(
            baseline.intent,
            stt=replace(baseline.intent.stt, provider=provider.value),
        ),
    )
    sender = FakeSender()

    OscStatePublisher(sender).start(state_from_settings(settings))

    assert ("/avatar/parameters/PuriPuly_SelfASR", expected_id) in sender.messages


@pytest.mark.parametrize(
    ("provider", "expected_id"),
    [
        (STTProviderName.GEMINI_TRANSCRIBE, 10),
        (STTProviderName.ELEVENLABS_SCRIBE, 11),
        (STTProviderName.QWEN_AUDIO, 12),
        (STTProviderName.ROLLING_FREE, 13),
    ],
)
def test_state_publisher_publishes_peer_asr_ids_for_new_providers(
    provider: STTProviderName,
    expected_id: int,
) -> None:
    baseline = AppSettingsVNext()
    settings = replace(
        baseline,
        intent=replace(
            baseline.intent,
            peer_stt=replace(baseline.intent.peer_stt, provider=provider.value),
        ),
    )
    sender = FakeSender()

    OscStatePublisher(sender).start(state_from_settings(settings))

    assert ("/avatar/parameters/PuriPuly_PeerASR", expected_id) in sender.messages
