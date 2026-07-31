from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from puripuly_heart.core import messages
from puripuly_heart.core.output.models import (
    ConversationFeedPublication,
    OutputRoutingDecision,
    PeerSubtitlePublication,
    SelfUtterancePublication,
    SystemDisclosurePublication,
)
from tests.helpers.ast_sources import assert_no_forbidden_imports, imported_modules

FORBIDDEN_IMPORT_PREFIXES = (
    "flet",
    "puripuly_heart.app.adapters",
    "puripuly_heart.config.settings",
    "puripuly_heart.core.managed_openrouter_broker_client",
    "puripuly_heart.core.osc",
    "puripuly_heart.core.overlay",
    "puripuly_heart.core.runtime_logging",
    "puripuly_heart.providers",
    "puripuly_heart.ui",
)


def _assert_no_forbidden_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert_no_forbidden_imports(Path(module.__file__ or ""), FORBIDDEN_IMPORT_PREFIXES)


def _output_router_class() -> type[object]:
    output = importlib.import_module("puripuly_heart.core.output")
    router = getattr(output, "OutputRouter", None)
    assert router is not None
    return router


@dataclass(slots=True)
class RecordingSelfChatbox:
    self_utterances: list[SelfUtterancePublication] = field(default_factory=list)
    disclosures: list[SystemDisclosurePublication] = field(default_factory=list)

    async def publish_self_utterance(self, publication: SelfUtterancePublication) -> None:
        self.self_utterances.append(publication)

    async def publish_system_disclosure(self, publication: SystemDisclosurePublication) -> None:
        self.disclosures.append(publication)


@dataclass(slots=True)
class RecordingSubtitleOverlay:
    peer_subtitles: list[PeerSubtitlePublication] = field(default_factory=list)

    async def publish_peer_subtitle(self, publication: PeerSubtitlePublication) -> None:
        self.peer_subtitles.append(publication)


@dataclass(slots=True)
class RecordingDashboard:
    disclosures: list[SystemDisclosurePublication] = field(default_factory=list)

    async def publish_system_disclosure(self, publication: SystemDisclosurePublication) -> None:
        self.disclosures.append(publication)


@dataclass(slots=True)
class RecordingConversationFeed:
    entries: list[ConversationFeedPublication] = field(default_factory=list)

    async def publish_conversation_entry(self, publication: ConversationFeedPublication) -> None:
        self.entries.append(publication)


@dataclass(slots=True)
class RecordingObserver:
    decisions: list[OutputRoutingDecision] = field(default_factory=list)

    async def observe_output_routing(self, decision: OutputRoutingDecision) -> None:
        self.decisions.append(decision)


@dataclass(slots=True)
class FailingSystemDisclosureChatbox:
    disclosures: list[SystemDisclosurePublication] = field(default_factory=list)

    async def publish_self_utterance(self, publication: SelfUtterancePublication) -> None:
        raise AssertionError("self utterances are not part of this test")

    async def publish_system_disclosure(self, publication: SystemDisclosurePublication) -> None:
        self.disclosures.append(publication)
        raise RuntimeError("chatbox publish failed")


def test_router_facade_is_import_safe() -> None:
    _assert_no_forbidden_imports("puripuly_heart.core.output.router")


def test_chatbox_and_subtitle_contract_modules_are_canonical_owners() -> None:
    _assert_no_forbidden_imports("puripuly_heart.core.output.chatbox")
    _assert_no_forbidden_imports("puripuly_heart.core.output.subtitle")
    chatbox = importlib.import_module("puripuly_heart.core.output.chatbox")
    subtitle = importlib.import_module("puripuly_heart.core.output.subtitle")
    output_models = importlib.import_module("puripuly_heart.core.output.models")

    chatbox_owned_names = (
        "SelfUtterancePublication",
        "SystemDisclosurePublication",
        "SelfChatboxOutputPort",
    )
    subtitle_owned_names = ("PeerSubtitlePublication", "SubtitleOverlayOutputPort")

    for name in chatbox_owned_names:
        contract = getattr(chatbox, name)
        assert contract.__module__ == "puripuly_heart.core.output.chatbox"
        assert getattr(output_models, name) is contract
    for name in subtitle_owned_names:
        contract = getattr(subtitle, name)
        assert contract.__module__ == "puripuly_heart.core.output.subtitle"
        assert getattr(output_models, name) is contract


def test_router_and_adapters_import_canonical_channel_contract_modules() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    expected_imports = {
        repo_root
        / "src"
        / "puripuly_heart"
        / "core"
        / "output"
        / "router.py": {
            "puripuly_heart.core.output.chatbox",
            "puripuly_heart.core.output.subtitle",
        },
        repo_root
        / "src"
        / "puripuly_heart"
        / "core"
        / "osc"
        / "chatbox_paginator.py": {"puripuly_heart.core.output.chatbox"},
        repo_root
        / "src"
        / "puripuly_heart"
        / "core"
        / "overlay"
        / "sink.py": {"puripuly_heart.core.output.subtitle"},
    }

    for module_file, expected in expected_imports.items():
        imports = imported_modules(module_file)
        assert expected <= imports


def test_peer_chatbox_denial_reason_is_closed_safe_contract() -> None:
    OutputRouter = _output_router_class()

    signature = inspect.signature(OutputRouter.deny_peer_chatbox_attempt)

    assert "reason" not in signature.parameters


@pytest.mark.asyncio
async def test_router_publishes_self_utterance_only_to_self_chatbox_and_observer() -> None:
    OutputRouter = _output_router_class()
    chatbox = RecordingSelfChatbox()
    overlay = RecordingSubtitleOverlay()
    dashboard = RecordingDashboard()
    conversation = RecordingConversationFeed()
    observer = RecordingObserver()
    router = OutputRouter(
        self_chatbox=chatbox,
        subtitle_overlay=overlay,
        dashboard=dashboard,
        conversation_feed=conversation,
        observers=(observer,),
    )
    publication = SelfUtterancePublication(
        utterance_id="self-1",
        transcript_text="hello self",
        translation_text="bonjour self",
        source_language="en",
        target_language="fr",
        is_final=True,
        metadata={"safe": True},
    )

    decisions = await router.publish_self_utterance(publication)

    assert chatbox.self_utterances == [publication]
    assert chatbox.disclosures == []
    assert overlay.peer_subtitles == []
    assert dashboard.disclosures == []
    assert conversation.entries == []
    assert decisions == tuple(observer.decisions)
    assert [decision.decision for decision in decisions] == ["published"]
    assert decisions[0].route == "self_chatbox"
    assert decisions[0].publication_id == "self-1"
    assert decisions[0].publication_kind == "self_utterance"
    assert decisions[0].metadata == {"channel": "self"}


@pytest.mark.asyncio
async def test_router_routes_peer_subtitles_to_overlay_and_denies_chatbox_attempt_safely() -> None:
    OutputRouter = _output_router_class()
    chatbox = RecordingSelfChatbox()
    overlay = RecordingSubtitleOverlay()
    observer = RecordingObserver()
    router = OutputRouter(
        self_chatbox=chatbox,
        subtitle_overlay=overlay,
        observers=(observer,),
    )
    publication = PeerSubtitlePublication(
        utterance_id="peer-1",
        transcript_text="secret peer transcript",
        translation_text="secret peer translation",
        source_language="ja",
        target_language="en",
        is_final=True,
        metadata={"safe": True},
    )

    publish_decisions = await router.publish_peer_subtitle(publication)
    deny_decisions = await router.deny_peer_chatbox_attempt(publication)

    assert overlay.peer_subtitles == [publication]
    assert chatbox.self_utterances == []
    assert chatbox.disclosures == []
    assert [decision.decision for decision in publish_decisions] == ["published"]
    assert publish_decisions[0].route == "subtitle_overlay"
    assert [decision.decision for decision in deny_decisions] == ["denied"]
    denied = deny_decisions[0]
    assert denied.route == "self_chatbox"
    assert denied.publication_id == "peer-1"
    assert denied.publication_kind == "peer_subtitle"
    assert denied.reason == "peer_chatbox_denied"
    assert denied.metadata == {"attempted_route": "self_chatbox", "channel": "peer"}
    assert tuple(observer.decisions) == publish_decisions + deny_decisions
    assert "secret peer transcript" not in repr(denied)
    assert "secret peer translation" not in repr(denied)


@pytest.mark.asyncio
async def test_router_skips_unconfigured_destinations_with_diagnostic_safe_metadata() -> None:
    OutputRouter = _output_router_class()
    observer = RecordingObserver()
    router = OutputRouter(observers=(observer,))
    publication = PeerSubtitlePublication(
        utterance_id="peer-2",
        transcript_text="hidden transcript",
        translation_text="hidden translation",
        source_language="ko",
        target_language="en",
        is_final=True,
        metadata={"text": "do not copy"},
    )

    decisions = await router.publish_peer_subtitle(publication)

    assert decisions == tuple(observer.decisions)
    assert [decision.decision for decision in decisions] == ["skipped"]
    assert decisions[0].route == "subtitle_overlay"
    assert decisions[0].reason == "destination_unconfigured"
    assert decisions[0].metadata == {"channel": "peer"}
    assert "hidden transcript" not in repr(decisions[0])
    assert "hidden translation" not in repr(decisions[0])
    assert "do not copy" not in repr(decisions[0])


@pytest.mark.asyncio
async def test_router_publishes_system_disclosures_without_transcript_fields() -> None:
    OutputRouter = _output_router_class()
    chatbox = RecordingSelfChatbox()
    dashboard = RecordingDashboard()
    observer = RecordingObserver()
    router = OutputRouter(self_chatbox=chatbox, dashboard=dashboard, observers=(observer,))
    publication = SystemDisclosurePublication(
        disclosure_id="system-1",
        message=messages.UserMessageRef(
            key="runtime.disclosure",
            params={"provider": "openrouter"},
            severity=messages.SEVERITY_INFO,
        ),
        metadata={"safe": True},
    )

    decisions = await router.publish_system_disclosure(publication)

    assert chatbox.disclosures == [publication]
    assert dashboard.disclosures == [publication]
    assert not hasattr(publication, "transcript_text")
    assert not hasattr(publication, "translation_text")
    assert [decision.decision for decision in decisions] == ["published", "published"]
    assert [decision.route for decision in decisions] == ["system_disclosure_chatbox", "dashboard"]
    assert {decision.publication_kind for decision in decisions} == {"system_disclosure"}
    assert tuple(observer.decisions) == decisions


@pytest.mark.asyncio
async def test_router_observers_never_decide_routes() -> None:
    OutputRouter = _output_router_class()
    chatbox = RecordingSelfChatbox()

    class ReturningObserver:
        async def observe_output_routing(self, decision: OutputRoutingDecision) -> str:
            return "deny-anyway"

    router = OutputRouter(self_chatbox=chatbox, observers=(ReturningObserver(),))
    publication = SelfUtterancePublication(
        utterance_id="self-2",
        transcript_text="hello",
        translation_text=None,
        source_language="en",
        target_language=None,
        is_final=True,
        metadata={},
    )

    decisions = await router.publish_self_utterance(publication)

    assert chatbox.self_utterances == [publication]
    assert [decision.decision for decision in decisions] == ["published"]


@pytest.mark.asyncio
async def test_observer_failures_do_not_interrupt_system_disclosure_routes() -> None:
    OutputRouter = _output_router_class()
    chatbox = RecordingSelfChatbox()
    dashboard = RecordingDashboard()
    observer = RecordingObserver()

    class RaisingObserver:
        async def observe_output_routing(self, decision: OutputRoutingDecision) -> None:
            raise RuntimeError("observer unavailable")

    router = OutputRouter(
        self_chatbox=chatbox,
        dashboard=dashboard,
        observers=(RaisingObserver(), observer),
    )
    publication = SystemDisclosurePublication(
        disclosure_id="system-observer-failure",
        message=messages.UserMessageRef(
            key="runtime.disclosure",
            params={"provider": "openrouter"},
            severity=messages.SEVERITY_INFO,
        ),
        metadata={"safe": True},
    )

    decisions = await router.publish_system_disclosure(publication)

    assert chatbox.disclosures == [publication]
    assert dashboard.disclosures == [publication]
    assert [decision.decision for decision in decisions] == ["published", "published"]
    assert [decision.route for decision in decisions] == [
        "system_disclosure_chatbox",
        "dashboard",
    ]
    assert tuple(observer.decisions) == decisions


@pytest.mark.asyncio
async def test_system_disclosure_port_failure_records_safe_skip_and_continues_routes() -> None:
    OutputRouter = _output_router_class()
    chatbox = FailingSystemDisclosureChatbox()
    dashboard = RecordingDashboard()
    observer = RecordingObserver()
    router = OutputRouter(self_chatbox=chatbox, dashboard=dashboard, observers=(observer,))
    publication = SystemDisclosurePublication(
        disclosure_id="system-port-failure",
        message=messages.UserMessageRef(
            key="runtime.disclosure",
            params={"provider": "openrouter"},
            severity=messages.SEVERITY_INFO,
        ),
        metadata={"safe": True},
    )

    decisions = await router.publish_system_disclosure(publication)

    assert chatbox.disclosures == [publication]
    assert dashboard.disclosures == [publication]
    assert [decision.decision for decision in decisions] == ["skipped", "published"]
    failed = decisions[0]
    assert failed.route == "system_disclosure_chatbox"
    assert failed.reason == "destination_publish_failed"
    assert failed.metadata == {"channel": "system", "error_type": "RuntimeError"}
    assert "chatbox publish failed" not in repr(failed)
    assert tuple(observer.decisions) == decisions
