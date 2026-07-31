from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from puripuly_heart.core import messages
from tests.helpers.ast_sources import assert_no_forbidden_imports

FORBIDDEN_IMPORT_PREFIXES = (
    "flet",
    "puripuly_heart.app.adapters",
    "puripuly_heart.config.settings",
    "puripuly_heart.core.managed_openrouter_broker_client",
    "puripuly_heart.core.osc",
    "puripuly_heart.core.runtime_logging",
    "puripuly_heart.providers",
    "puripuly_heart.ui",
)


def _assert_no_forbidden_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert_no_forbidden_imports(Path(module.__file__ or ""), FORBIDDEN_IMPORT_PREFIXES)


def _assert_protocol_method_is_async(protocol: type[object], method_name: str) -> None:
    method = getattr(protocol, method_name)
    assert inspect.iscoroutinefunction(method)


def test_observability_contracts_are_import_safe_protocol_sinks() -> None:
    _assert_no_forbidden_imports("puripuly_heart.core.observability")
    observability = importlib.import_module("puripuly_heart.core.observability")

    expected_sinks = {
        "RuntimeLogSink": "emit_runtime_log",
        "DiagnosticsSink": "emit_diagnostic",
        "ProviderObservationSink": "emit_provider_observation",
        "ConversationRecordSink": "record_conversation",
        "PersistedDiagnosticStore": "persist_diagnostic",
    }
    for sink_name, method_name in expected_sinks.items():
        sink = getattr(observability, sink_name)
        assert getattr(sink, "_is_protocol", False), f"{sink_name} must stay abstract"
        _assert_protocol_method_is_async(sink, method_name)


def test_structured_observability_events_use_safe_message_contracts() -> None:
    observability = importlib.import_module("puripuly_heart.core.observability")
    diagnostic = messages.ErrorDiagnostics(
        component="runtime.apply",
        operation="apply",
        code="provider_degraded",
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_DETAILED,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"provider": "openrouter"},
    )
    event = observability.DiagnosticEvent(
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        severity=messages.SEVERITY_WARNING,
        visibility=messages.DIAGNOSTIC_VISIBILITY_DETAILED,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        correlation_id="corr-1",
        diagnostics=diagnostic,
        fields={"provider": "openrouter", "latency_ms": 20.5},
    )
    provider_observation = observability.ProviderObservationEvent(
        provider="openrouter",
        operation="verify_secret",
        outcome="failure",
        correlation_id="corr-1",
        diagnostics=diagnostic,
        fields={"status_code": 503},
    )
    conversation_record = observability.ConversationRecord(
        utterance_id="utt-1",
        speaker_channel="self",
        transcript_text="hello",
        translation_text="bonjour",
        source_language="en",
        target_language="fr",
        metadata={"is_final": True},
    )

    for dto in (event, provider_observation, conversation_record):
        assert is_dataclass(dto)
        assert not hasattr(dto, "__dict__")

    with pytest.raises(FrozenInstanceError):
        event.correlation_id = "corr-2"  # type: ignore[misc]
    with pytest.raises(TypeError):
        event.fields["provider"] = "qwen"  # type: ignore[index]
    with pytest.raises(TypeError):
        provider_observation.fields["status_code"] = 429  # type: ignore[index]
    with pytest.raises(TypeError):
        conversation_record.metadata["is_final"] = False  # type: ignore[index]

    hints = get_type_hints(observability.DiagnosticEvent)
    assert hints["category"] == messages.DiagnosticCategory
    assert hints["severity"] == messages.Severity
    assert hints["visibility"] == messages.DiagnosticVisibility
    assert hints["content_policy"] == messages.ContentPolicy
    assert hints["diagnostics"] == messages.ErrorDiagnostics | None


def test_structured_observability_event_fields_are_json_safe_scalars() -> None:
    observability = importlib.import_module("puripuly_heart.core.observability")

    with pytest.raises(TypeError):
        observability.DiagnosticEvent(
            category=messages.DIAGNOSTIC_CATEGORY_UNKNOWN,
            severity=messages.SEVERITY_WARNING,
            visibility=messages.DIAGNOSTIC_VISIBILITY_DETAILED,
            content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
            correlation_id="corr-json-safe-type",
            diagnostics=None,
            fields={"nested": {"not": "json-safe field scalar"}},
        )

    with pytest.raises(ValueError):
        observability.RuntimeLogEvent(
            category=messages.DIAGNOSTIC_CATEGORY_UNKNOWN,
            severity=messages.SEVERITY_WARNING,
            visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
            correlation_id="corr-json-safe-float",
            message=None,
            diagnostics=None,
            fields={"elapsed_ms": math.inf},
        )


def test_output_contracts_separate_self_peer_system_and_observer_payloads() -> None:
    _assert_no_forbidden_imports("puripuly_heart.core.output.models")
    output_models = importlib.import_module("puripuly_heart.core.output.models")

    message = messages.UserMessageRef(
        key="runtime.disclosure",
        params={"provider": "openrouter"},
        severity=messages.SEVERITY_INFO,
    )
    self_publication = output_models.SelfUtterancePublication(
        utterance_id="self-1",
        transcript_text="hello",
        translation_text="bonjour",
        source_language="en",
        target_language="fr",
        is_final=True,
        metadata={"route": "chatbox"},
    )
    peer_publication = output_models.PeerSubtitlePublication(
        utterance_id="peer-1",
        transcript_text="hello",
        translation_text="bonjour",
        source_language="en",
        target_language="fr",
        is_final=True,
        metadata={"route": "subtitle"},
    )
    disclosure = output_models.SystemDisclosurePublication(
        disclosure_id="system-1",
        message=message,
        metadata={"route": "system_disclosure"},
    )
    denied_decision = output_models.OutputRoutingDecision(
        decision="denied",
        route="self_chatbox",
        publication_id="peer-1",
        publication_kind="peer_subtitle",
        reason="peer_chatbox_denied",
        metadata={"attempted_route": "self_chatbox"},
    )

    for dto in (self_publication, peer_publication, disclosure, denied_decision):
        assert is_dataclass(dto)
        assert not hasattr(dto, "__dict__")

    assert {field.name for field in fields(output_models.SystemDisclosurePublication)} == {
        "disclosure_id",
        "message",
        "metadata",
    }
    assert not hasattr(denied_decision, "transcript_text")
    assert not hasattr(denied_decision, "translation_text")
    with pytest.raises(TypeError):
        denied_decision.metadata["attempted_route"] = "subtitle"  # type: ignore[index]

    assert set(get_args(output_models.OutputRoutingDecisionStatus)) == {
        "published",
        "skipped",
        "denied",
    }
    assert self_publication.translation_text == peer_publication.translation_text


def test_output_ports_are_async_protocol_only_destinations() -> None:
    output_models = importlib.import_module("puripuly_heart.core.output.models")

    expected_ports = {
        "SelfChatboxOutputPort": ("publish_self_utterance", "publish_system_disclosure"),
        "SubtitleOverlayOutputPort": ("publish_peer_subtitle",),
        "DashboardOutputPort": ("publish_system_disclosure",),
        "ConversationFeedPort": ("publish_conversation_entry",),
        "OutputRoutingObserverPort": ("observe_output_routing",),
    }
    for port_name, method_names in expected_ports.items():
        port = getattr(output_models, port_name)
        assert getattr(port, "_is_protocol", False), f"{port_name} must stay abstract"
        for method_name in method_names:
            _assert_protocol_method_is_async(port, method_name)

    hints = get_type_hints(output_models.SubtitleOverlayOutputPort.publish_peer_subtitle)
    assert hints["publication"] == output_models.PeerSubtitlePublication
    hints = get_type_hints(output_models.OutputRoutingObserverPort.observe_output_routing)
    assert hints["decision"] == output_models.OutputRoutingDecision
