from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol

from puripuly_heart.core.messages import DiagnosticFieldValue, ErrorDiagnostics
from puripuly_heart.core.output.chatbox import (
    SelfChatboxOutputPort,
    SelfUtterancePublication,
    SystemDisclosurePublication,
)
from puripuly_heart.core.output.subtitle import (
    PeerSubtitlePublication,
    SubtitleOverlayOutputPort,
)

OutputRoute = Literal[
    "self_chatbox",
    "subtitle_overlay",
    "dashboard",
    "conversation_feed",
    "system_disclosure_chatbox",
]
OUTPUT_ROUTE_SELF_CHATBOX: Final[OutputRoute] = "self_chatbox"
OUTPUT_ROUTE_SUBTITLE_OVERLAY: Final[OutputRoute] = "subtitle_overlay"
OUTPUT_ROUTE_DASHBOARD: Final[OutputRoute] = "dashboard"
OUTPUT_ROUTE_CONVERSATION_FEED: Final[OutputRoute] = "conversation_feed"
OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX: Final[OutputRoute] = "system_disclosure_chatbox"
OUTPUT_ROUTES: Final[tuple[OutputRoute, ...]] = (
    OUTPUT_ROUTE_SELF_CHATBOX,
    OUTPUT_ROUTE_SUBTITLE_OVERLAY,
    OUTPUT_ROUTE_DASHBOARD,
    OUTPUT_ROUTE_CONVERSATION_FEED,
    OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
)

OutputPublicationKind = Literal[
    "self_utterance",
    "peer_subtitle",
    "system_disclosure",
    "conversation_feed",
]
PUBLICATION_KIND_SELF_UTTERANCE: Final[OutputPublicationKind] = "self_utterance"
PUBLICATION_KIND_PEER_SUBTITLE: Final[OutputPublicationKind] = "peer_subtitle"
PUBLICATION_KIND_SYSTEM_DISCLOSURE: Final[OutputPublicationKind] = "system_disclosure"
PUBLICATION_KIND_CONVERSATION_FEED: Final[OutputPublicationKind] = "conversation_feed"
OUTPUT_PUBLICATION_KINDS: Final[tuple[OutputPublicationKind, ...]] = (
    PUBLICATION_KIND_SELF_UTTERANCE,
    PUBLICATION_KIND_PEER_SUBTITLE,
    PUBLICATION_KIND_SYSTEM_DISCLOSURE,
    PUBLICATION_KIND_CONVERSATION_FEED,
)

OutputRoutingDecisionStatus = Literal["published", "skipped", "denied"]
OUTPUT_ROUTING_DECISION_PUBLISHED: Final[OutputRoutingDecisionStatus] = "published"
OUTPUT_ROUTING_DECISION_SKIPPED: Final[OutputRoutingDecisionStatus] = "skipped"
OUTPUT_ROUTING_DECISION_DENIED: Final[OutputRoutingDecisionStatus] = "denied"
OUTPUT_ROUTING_DECISION_STATUSES: Final[tuple[OutputRoutingDecisionStatus, ...]] = (
    OUTPUT_ROUTING_DECISION_PUBLISHED,
    OUTPUT_ROUTING_DECISION_SKIPPED,
    OUTPUT_ROUTING_DECISION_DENIED,
)


def _freeze_metadata(
    values: Mapping[str, DiagnosticFieldValue],
) -> Mapping[str, DiagnosticFieldValue]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class ConversationFeedPublication:
    utterance_id: str
    transcript_text: str | None
    translation_text: str | None
    source_language: str | None
    target_language: str | None
    metadata: Mapping[str, DiagnosticFieldValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class OutputRoutingDecision:
    decision: OutputRoutingDecisionStatus
    route: OutputRoute
    publication_id: str
    publication_kind: OutputPublicationKind
    reason: str | None
    metadata: Mapping[str, DiagnosticFieldValue]
    diagnostics: ErrorDiagnostics | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class DashboardOutputPort(Protocol):
    async def publish_system_disclosure(
        self,
        publication: SystemDisclosurePublication,
    ) -> None: ...


class ConversationFeedPort(Protocol):
    async def publish_conversation_entry(
        self,
        publication: ConversationFeedPublication,
    ) -> None: ...


class OutputRoutingObserverPort(Protocol):
    async def observe_output_routing(
        self,
        decision: OutputRoutingDecision,
    ) -> None: ...


__all__ = [
    "ConversationFeedPort",
    "ConversationFeedPublication",
    "DashboardOutputPort",
    "OUTPUT_PUBLICATION_KINDS",
    "OUTPUT_ROUTE_CONVERSATION_FEED",
    "OUTPUT_ROUTE_DASHBOARD",
    "OUTPUT_ROUTE_SELF_CHATBOX",
    "OUTPUT_ROUTE_SUBTITLE_OVERLAY",
    "OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX",
    "OUTPUT_ROUTING_DECISION_DENIED",
    "OUTPUT_ROUTING_DECISION_PUBLISHED",
    "OUTPUT_ROUTING_DECISION_SKIPPED",
    "OUTPUT_ROUTING_DECISION_STATUSES",
    "OUTPUT_ROUTES",
    "OutputPublicationKind",
    "OutputRoute",
    "OutputRoutingDecision",
    "OutputRoutingDecisionStatus",
    "OutputRoutingObserverPort",
    "PUBLICATION_KIND_CONVERSATION_FEED",
    "PUBLICATION_KIND_PEER_SUBTITLE",
    "PUBLICATION_KIND_SELF_UTTERANCE",
    "PUBLICATION_KIND_SYSTEM_DISCLOSURE",
    "PeerSubtitlePublication",
    "SelfChatboxOutputPort",
    "SelfUtterancePublication",
    "SubtitleOverlayOutputPort",
    "SystemDisclosurePublication",
]
