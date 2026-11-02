from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol
from uuid import UUID

from puripuly_heart.core.messages import DiagnosticFieldValue


def _freeze_metadata(
    values: Mapping[str, DiagnosticFieldValue],
) -> Mapping[str, DiagnosticFieldValue]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class PeerSubtitlePublication:
    utterance_id: str
    transcript_text: str | None
    translation_text: str | None
    source_language: str | None
    target_language: str | None
    is_final: bool
    metadata: Mapping[str, DiagnosticFieldValue]
    peer_caption_revision: int = 0
    peer_superseded_unit_ids: tuple[UUID, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class SubtitleOverlayOutputPort(Protocol):
    async def publish_peer_subtitle(
        self,
        publication: PeerSubtitlePublication,
    ) -> None: ...


__all__ = [
    "PeerSubtitlePublication",
    "SubtitleOverlayOutputPort",
]
