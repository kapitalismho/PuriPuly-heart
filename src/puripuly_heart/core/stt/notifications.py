from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.domain.models import ChannelId


@dataclass(frozen=True, slots=True)
class FinalTranscriptSuppressedNotification:
    utterance_id: UUID
    channel: ChannelId
    stt_provider_name: STTProviderName
