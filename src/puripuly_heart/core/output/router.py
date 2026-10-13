from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from puripuly_heart.core.diagnostics import emit_failure, safe_exception_fields
from puripuly_heart.core.messages import DiagnosticFieldValue, ErrorDiagnostics
from puripuly_heart.core.output.chatbox import (
    SelfChatboxOutputPort,
    SelfUtterancePublication,
    SystemDisclosurePublication,
)
from puripuly_heart.core.output.models import (
    OUTPUT_ROUTE_CONVERSATION_FEED,
    OUTPUT_ROUTE_DASHBOARD,
    OUTPUT_ROUTE_SELF_CHATBOX,
    OUTPUT_ROUTE_SUBTITLE_OVERLAY,
    OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
    OUTPUT_ROUTING_DECISION_DENIED,
    OUTPUT_ROUTING_DECISION_PUBLISHED,
    OUTPUT_ROUTING_DECISION_SKIPPED,
    PUBLICATION_KIND_CONVERSATION_FEED,
    PUBLICATION_KIND_PEER_SUBTITLE,
    PUBLICATION_KIND_SELF_UTTERANCE,
    PUBLICATION_KIND_SYSTEM_DISCLOSURE,
    ConversationFeedPort,
    ConversationFeedPublication,
    DashboardOutputPort,
    OutputPublicationKind,
    OutputRoute,
    OutputRoutingDecision,
    OutputRoutingDecisionStatus,
    OutputRoutingObserverPort,
)
from puripuly_heart.core.output.subtitle import (
    PeerSubtitlePublication,
    SubtitleOverlayOutputPort,
)

_DecisionMetadata = Mapping[str, DiagnosticFieldValue]
_Publisher = Callable[[], Awaitable[None]]

_DESTINATION_PUBLISH_FAILED_REASON = "destination_publish_failed"
_DESTINATION_UNCONFIGURED_REASON = "destination_unconfigured"
_PEER_CHATBOX_DENIED_REASON = "peer_chatbox_denied"
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OutputRouter:
    self_chatbox: SelfChatboxOutputPort | None = None
    subtitle_overlay: SubtitleOverlayOutputPort | None = None
    dashboard: DashboardOutputPort | None = None
    conversation_feed: ConversationFeedPort | None = None
    observers: Sequence[OutputRoutingObserverPort] = field(default_factory=tuple)
    observer_failure_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.observers = tuple(self.observers)

    async def publish_self_utterance(
        self,
        publication: SelfUtterancePublication,
    ) -> tuple[OutputRoutingDecision, ...]:
        return (
            await self._publish_or_skip(
                publisher=(
                    None
                    if self.self_chatbox is None
                    else lambda: self.self_chatbox.publish_self_utterance(publication)
                ),
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=publication.utterance_id,
                publication_kind=PUBLICATION_KIND_SELF_UTTERANCE,
                metadata={"channel": "self"},
            ),
        )

    async def publish_peer_subtitle(
        self,
        publication: PeerSubtitlePublication,
    ) -> tuple[OutputRoutingDecision, ...]:
        return (
            await self._publish_or_skip(
                publisher=(
                    None
                    if self.subtitle_overlay is None
                    else lambda: self.subtitle_overlay.publish_peer_subtitle(publication)
                ),
                route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
                publication_id=publication.utterance_id,
                publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
                metadata={"channel": "peer"},
            ),
        )

    async def publish_system_disclosure(
        self,
        publication: SystemDisclosurePublication,
    ) -> tuple[OutputRoutingDecision, ...]:
        return (
            await self._publish_or_skip(
                publisher=(
                    None
                    if self.self_chatbox is None
                    else lambda: self.self_chatbox.publish_system_disclosure(publication)
                ),
                route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
                publication_id=publication.disclosure_id,
                publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
                metadata={"channel": "system"},
            ),
            await self._publish_or_skip(
                publisher=(
                    None
                    if self.dashboard is None
                    else lambda: self.dashboard.publish_system_disclosure(publication)
                ),
                route=OUTPUT_ROUTE_DASHBOARD,
                publication_id=publication.disclosure_id,
                publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
                metadata={"channel": "system"},
            ),
        )

    async def publish_conversation_entry(
        self,
        publication: ConversationFeedPublication,
    ) -> tuple[OutputRoutingDecision, ...]:
        return (
            await self._publish_or_skip(
                publisher=(
                    None
                    if self.conversation_feed is None
                    else lambda: self.conversation_feed.publish_conversation_entry(publication)
                ),
                route=OUTPUT_ROUTE_CONVERSATION_FEED,
                publication_id=publication.utterance_id,
                publication_kind=PUBLICATION_KIND_CONVERSATION_FEED,
                metadata={"channel": "self"},
            ),
        )

    async def deny_peer_chatbox_attempt(
        self,
        publication: PeerSubtitlePublication,
    ) -> tuple[OutputRoutingDecision, ...]:
        return (
            await self._record_decision(
                status=OUTPUT_ROUTING_DECISION_DENIED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=publication.utterance_id,
                publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
                reason=_PEER_CHATBOX_DENIED_REASON,
                metadata={"attempted_route": OUTPUT_ROUTE_SELF_CHATBOX, "channel": "peer"},
            ),
        )

    async def _publish_or_skip(
        self,
        *,
        publisher: _Publisher | None,
        route: OutputRoute,
        publication_id: str,
        publication_kind: OutputPublicationKind,
        metadata: _DecisionMetadata,
    ) -> OutputRoutingDecision:
        if publisher is None:
            return await self._record_decision(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=route,
                publication_id=publication_id,
                publication_kind=publication_kind,
                reason=_DESTINATION_UNCONFIGURED_REASON,
                metadata=metadata,
            )

        try:
            await publisher()
        except Exception as exc:
            return await self._record_decision(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=route,
                publication_id=publication_id,
                publication_kind=publication_kind,
                reason=_DESTINATION_PUBLISH_FAILED_REASON,
                metadata={
                    **metadata,
                    "error_type": safe_exception_fields(exc)["exception_class"],
                    **safe_exception_fields(exc),
                },
            )
        return await self._record_decision(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=route,
            publication_id=publication_id,
            publication_kind=publication_kind,
            reason=None,
            metadata=metadata,
        )

    async def _record_decision(
        self,
        *,
        status: OutputRoutingDecisionStatus,
        route: OutputRoute,
        publication_id: str,
        publication_kind: OutputPublicationKind,
        reason: str | None,
        metadata: _DecisionMetadata,
    ) -> OutputRoutingDecision:
        diagnostics = None
        if reason == _DESTINATION_PUBLISH_FAILED_REASON:
            diagnostics = ErrorDiagnostics(
                component="output_runtime",
                operation="publish",
                code=reason,
                category="lifecycle",
                visibility="basic",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                fields={
                    "destination": route,
                    "publication_id": publication_id,
                    "publication_kind": publication_kind,
                    **{
                        key: value
                        for key, value in metadata.items()
                        if key
                        in {
                            "exception_class",
                            "errno",
                            "winerror",
                            "cause_class_1",
                            "cause_class_2",
                            "cause_class_3",
                            "cause_class_4",
                        }
                    },
                },
                channel=(
                    metadata.get("channel") if metadata.get("channel") in {"self", "peer"} else None
                ),
            )
            emit_failure(logger, diagnostics, disposition="degraded", stage="destination_delivery")
        decision = OutputRoutingDecision(
            decision=status,
            route=route,
            publication_id=publication_id,
            publication_kind=publication_kind,
            reason=reason,
            metadata=metadata,
            diagnostics=diagnostics,
        )
        for observer in self.observers:
            try:
                await observer.observe_output_routing(decision)
            except Exception as exc:
                self.observer_failure_count += 1
                if self.observer_failure_count == 1:
                    emit_failure(
                        logger,
                        ErrorDiagnostics(
                            component="output_runtime",
                            operation="observe",
                            code="output_observer_failed",
                            category="lifecycle",
                            visibility="diagnostic_only",
                            content_policy="metadata_only",
                            status_code=None,
                            retry_after_ms=None,
                            fields={"destination": route, "publication_id": publication_id},
                            cause_id=diagnostics.diagnostic_id if diagnostics else None,
                        ),
                        severity="warning",
                        disposition="degraded",
                        stage="routing_observer",
                        exception=exc,
                    )
                continue
        return decision


__all__ = ["OutputRouter"]
