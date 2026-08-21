"""Unit tests for LLM translation context memory feature."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.orchestrator.channel_runtime import ChannelRuntime, ContextEntry
from puripuly_heart.core.orchestrator.context import ContextResolver
from puripuly_heart.core.translation_backend import LlmTranslationBackend
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.translation_owners import compose_translation_test_harness

# ── Mock classes ──────────────────────────────────────────────────────────────


class FakeClock:
    """Fake clock for testing time-based logic."""

    def __init__(self, initial_time: float = 0.0):
        self._time = initial_time

    def now(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


@dataclass
class FakeLLMProvider:
    """Fake LLM provider that records calls."""

    calls: list[dict] = field(default_factory=list)
    response_text: str = "translated"

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ):
        from puripuly_heart.domain.models import Translation

        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "context": context,
            }
        )
        return Translation(utterance_id=utterance_id, text=self.response_text)

    async def close(self) -> None:
        pass


@dataclass
class FakeOscQueue:
    """Fake OSC queue that records enqueued messages."""

    messages: list = field(default_factory=list)

    def enqueue(self, msg) -> None:
        self.messages.append(msg)

    def send_typing(self, on: bool) -> None:
        pass

    def set_typing_reason(self, reason: str, active: bool) -> None:
        pass

    def process_due(self) -> None:
        pass


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestContextFiltering:
    """Test context time window and max entries filtering."""

    def test_context_filters_by_time_window(self):
        """Context entries older than time_window_s should be excluded."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=5.0,
            context_max_entries=3,
        )

        # Add entries at different times
        harness.self_runtime.translation_history = [
            ContextEntry(text="old", source_language="ko", target_language="en", timestamp=3.0),
            ContextEntry(text="recent1", source_language="ko", target_language="en", timestamp=6.0),
            ContextEntry(text="recent2", source_language="ko", target_language="en", timestamp=8.0),
        ]

        valid = harness.get_valid_context()

        assert len(valid) == 2
        assert valid[0].text == "recent1"
        assert valid[1].text == "recent2"

    def test_context_filters_by_max_entries(self):
        """Only the most recent max_entries should be considered."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=20.0,  # Default window
            context_max_entries=2,
        )

        harness.self_runtime.translation_history = [
            ContextEntry(text="first", source_language="ko", target_language="en", timestamp=7.0),
            ContextEntry(text="second", source_language="ko", target_language="en", timestamp=8.0),
            ContextEntry(text="third", source_language="ko", target_language="en", timestamp=9.0),
        ]

        valid = harness.get_valid_context()

        # Should only get last 2
        assert len(valid) == 2
        assert valid[0].text == "second"
        assert valid[1].text == "third"

    def test_context_filters_by_language_pair(self):
        """Only entries with the current language pair should be included."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=20.0,
        )

        harness.self_runtime.translation_history = [
            ContextEntry(text="wrong", source_language="ja", target_language="en", timestamp=9.0),
            ContextEntry(text="ok", source_language="ko", target_language="en", timestamp=9.5),
        ]

        valid = harness.get_valid_context()

        assert len(valid) == 1
        assert valid[0].text == "ok"

    def test_context_filters_short_entries(self):
        """Entries shorter than 2 characters should be excluded."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=20.0,
        )

        harness.self_runtime.translation_history = [
            ContextEntry(text="a", source_language="ko", target_language="en", timestamp=9.0),
            ContextEntry(text="ok", source_language="ko", target_language="en", timestamp=9.5),
        ]

        valid = harness.get_valid_context()

        assert len(valid) == 1
        assert valid[0].text == "ok"

    def test_context_cleared_on_clear_context(self):
        """clear_context() should empty the history."""
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
        )

        harness.self_runtime.translation_history = [
            ContextEntry(text="test", source_language="ko", target_language="en", timestamp=1.0),
        ]
        harness.clear_context()

        assert len(harness.self_runtime.translation_history) == 0

    def test_old_entries_removed_when_full(self):
        """When max_entries is exceeded, oldest should be removed."""
        clock = FakeClock(initial_time=10.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_max_entries=3,
        )

        # Add 3 entries (at capacity)
        harness.self_runtime.translation_history = [
            ContextEntry(text="e1", source_language="ko", target_language="en", timestamp=7.0),
            ContextEntry(text="e2", source_language="ko", target_language="en", timestamp=8.0),
            ContextEntry(text="e3", source_language="ko", target_language="en", timestamp=9.0),
        ]

        # Add a 4th entry
        harness.self_runtime.translation_history.append(
            ContextEntry(text="e4", source_language="ko", target_language="en", timestamp=10.0)
        )
        if (
            len(harness.self_runtime.translation_history)
            > harness.configuration.snapshot().value.context_max_entries
        ):
            harness.self_runtime.translation_history.pop(0)

        assert len(harness.self_runtime.translation_history) == 3
        assert harness.self_runtime.translation_history[0].text == "e2"  # e1 removed

    def test_context_resolver_tracks_updated_owner_settings_after_init(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=10.0),
            context_time_window_s=20.0,
            context_max_entries=3,
        )
        harness.self_runtime.translation_history = [
            ContextEntry(text="first", source_language="ko", target_language="en", timestamp=8.0),
            ContextEntry(text="second", source_language="ko", target_language="en", timestamp=9.0),
            ContextEntry(text="third", source_language="ko", target_language="en", timestamp=9.5),
        ]

        harness.replace_configuration(context_max_entries=1)
        harness.replace_configuration(context_time_window_s=2.0)
        harness.set_clock(FakeClock(initial_time=11.0))
        valid = harness.get_valid_context()

        assert [entry.text for entry in valid] == ["third"]


class TestContextPassedToLLM:
    """Test that context is correctly passed to LLM."""

    @pytest.mark.asyncio
    async def test_context_passed_to_llm(self):
        """LLM should receive formatted context string."""
        clock = FakeClock(initial_time=10.0)
        fake_llm = FakeLLMProvider()
        harness = compose_translation_test_harness(
            stt=None,
            llm=fake_llm,
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=5.0,
            context_max_entries=3,
        )

        # Add some context
        harness.self_runtime.translation_history = [
            ContextEntry(text="hello", source_language="ko", target_language="en", timestamp=8.0),
        ]

        await harness.self_owner.submit_text("world")
        await asyncio.gather(
            *harness.self_runtime.translation_tasks.values(), return_exceptions=True
        )

        # Verify LLM was called with context
        assert len(fake_llm.calls) == 1
        call = fake_llm.calls[0]
        assert call["context"] == '- [self] "hello"'

    @pytest.mark.asyncio
    async def test_empty_context_when_no_history(self):
        """LLM should receive empty context when no history."""
        clock = FakeClock(initial_time=10.0)
        fake_llm = FakeLLMProvider()
        harness = compose_translation_test_harness(
            stt=None,
            llm=fake_llm,
            osc=FakeOscQueue(),
            clock=clock,
        )

        harness.self_runtime.translation_history = []

        await harness.self_owner.submit_text("test")
        await asyncio.gather(
            *harness.self_runtime.translation_tasks.values(), return_exceptions=True
        )

        assert len(fake_llm.calls) == 1
        assert fake_llm.calls[0]["context"] == ""

    @pytest.mark.asyncio
    async def test_empty_context_when_all_expired(self):
        """LLM should receive empty context when all entries are expired."""
        clock = FakeClock(initial_time=100.0)  # Far in the future
        fake_llm = FakeLLMProvider()
        harness = compose_translation_test_harness(
            stt=None,
            llm=fake_llm,
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=5.0,
        )

        # All entries are very old
        harness.self_runtime.translation_history = [
            ContextEntry(text="old", source_language="ko", target_language="en", timestamp=1.0),
        ]

        await harness.self_owner.submit_text("test")
        await asyncio.gather(
            *harness.self_runtime.translation_tasks.values(), return_exceptions=True
        )

        assert len(fake_llm.calls) == 1
        assert fake_llm.calls[0]["context"] == ""


class TestContextFormatting:
    """Test context formatting for LLM."""

    def test_format_context_empty(self):
        """Empty context list should return empty string."""
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
        )

        result = harness.format_context([])
        assert result == ""

    def test_format_context_single_entry(self):
        """Single entry should be formatted correctly."""
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=20.0),
        )

        entries = [
            ContextEntry(text="안녕", source_language="ko", target_language="en", timestamp=8.0)
        ]
        result = harness.format_context(entries)

        assert result == '- [self] "안녕"'

    def test_format_context_multiple_entries(self):
        """Multiple entries should all be included."""
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=20.0),
        )

        entries = [
            ContextEntry(text="a", source_language="ko", target_language="en", timestamp=8.0),
            ContextEntry(text="b", source_language="ko", target_language="en", timestamp=9.0),
        ]
        result = harness.format_context(entries)

        assert '- [self] "a"' in result
        assert '- [self] "b"' in result


class TestContextInternalPaths:
    def test_context_resolver_formats_local_without_relative_age(self):
        runtime = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=112.0),
        ).self_runtime
        runtime.remember_context("hello there", timestamp=100.0)
        resolver = ContextResolver(clock=FakeClock(initial_time=112.0))

        context, mode = resolver.resolve_local(
            runtime=runtime,
            source_language="en",
            target_language="ko",
        )

        assert mode == "local"
        assert context == '- [self] "hello there"'

    def test_translation_fixture_uses_local_context_when_peer_translation_is_off(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=112.0),
            integrated_context_enabled=True,
            peer_translation_enabled=False,
        )
        harness.self_runtime.remember_context("self only", timestamp=100.0)

        context, mode = harness.translation_requests.context_resolver.resolve_for_request(
            runtime=harness.self_runtime,
            other_runtime=harness.peer_runtime,
            requested_mode="integrated",
            peer_translation_enabled=harness.configuration.snapshot().value.peer_translation_enabled,
            source_language="en",
            target_language="ko",
        )

        assert mode == "local"
        assert context == '- [self] "self only"'

    def test_context_resolver_formats_integrated_with_channel_prefix(self):
        self_runtime = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=112.0),
        ).self_runtime
        peer_runtime = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=112.0),
        ).peer_runtime
        self_runtime.remember_context("I am ready", timestamp=100.0)
        peer_runtime.remember_context(
            "hello from peer",
            timestamp=105.0,
        )
        resolver = ContextResolver(clock=FakeClock(initial_time=112.0))

        context, mode = resolver.resolve_for_request(
            runtime=self_runtime,
            other_runtime=peer_runtime,
            requested_mode="integrated",
            peer_translation_enabled=True,
            source_language="en",
            target_language="ko",
        )

        assert mode == "integrated"
        assert context == ('- [self] "I am ready"\n- [peer] "hello from peer"')

    def test_context_resolver_always_uses_integrated_when_peer_enabled(self):
        self_runtime = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=112.0),
        ).self_runtime
        peer_runtime = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=112.0),
        ).peer_runtime
        self_runtime.remember_context("safe local line", timestamp=100.0)
        peer_runtime.remember_context(
            "peer line",
            timestamp=105.0,
        )
        resolver = ContextResolver(clock=FakeClock(initial_time=112.0))

        context, mode = resolver.resolve_for_request(
            runtime=self_runtime,
            other_runtime=peer_runtime,
            requested_mode="integrated",
            peer_translation_enabled=True,
            source_language="en",
            target_language="ko",
        )

        assert mode == "integrated"
        assert '- [peer] "peer line"' in context

    def test_context_resolver_falls_back_to_local_when_peer_context_is_empty(self):
        self_runtime = ChannelRuntime(channel="self")
        peer_runtime = ChannelRuntime(channel="peer")
        self_runtime.remember_context(
            "safe local line",
            timestamp=100.0,
            source_language="en",
            target_language="ko",
        )
        resolver = ContextResolver(clock=FakeClock(initial_time=112.0))

        context, mode = resolver.resolve_for_request(
            runtime=self_runtime,
            other_runtime=peer_runtime,
            requested_mode="integrated",
            peer_translation_enabled=True,
            source_language="en",
            target_language="ko",
        )

        assert mode == "local"
        assert context == '- [self] "safe local line"'

    def test_integrated_context_uses_40_second_window_before_entry_budget(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=100.0),
            integrated_context_enabled=True,
            peer_translation_enabled=True,
        )
        harness.self_runtime.remember_context(
            "41 seconds old", timestamp=59.0, source_language="en", target_language="ko"
        )
        harness.self_runtime.remember_context(
            "self recent", timestamp=70.0, source_language="en", target_language="ko"
        )
        harness.peer_runtime.remember_context(
            "peer recent", timestamp=71.0, source_language="en", target_language="ko"
        )
        harness.self_runtime.remember_context(
            "self newest", timestamp=72.0, source_language="en", target_language="ko"
        )

        context, mode = harness.translation_requests.context_resolver.resolve_for_request(
            runtime=harness.self_runtime,
            other_runtime=harness.peer_runtime,
            requested_mode="integrated",
            peer_translation_enabled=True,
            source_language="en",
            target_language="ko",
            other_source_language="en",
            other_target_language="ko",
        )

        assert mode == "integrated"
        assert "41 seconds old" not in context
        assert context == (
            '- [self] "self recent"\n' '- [peer] "peer recent"\n' '- [self] "self newest"'
        )

    def test_integrated_context_uses_latest_4_combined_entries_after_timestamp_merge(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=100.0),
            integrated_context_enabled=True,
            peer_translation_enabled=True,
        )
        harness.self_runtime.remember_context(
            "self 1", timestamp=70.0, source_language="en", target_language="ko"
        )
        harness.peer_runtime.remember_context(
            "peer 1", timestamp=71.0, source_language="en", target_language="ko"
        )
        harness.self_runtime.remember_context(
            "self 2", timestamp=72.0, source_language="en", target_language="ko"
        )
        harness.peer_runtime.remember_context(
            "peer 2", timestamp=73.0, source_language="en", target_language="ko"
        )
        harness.self_runtime.remember_context(
            "self 3", timestamp=74.0, source_language="en", target_language="ko"
        )

        context, mode = harness.translation_requests.context_resolver.resolve_for_request(
            runtime=harness.self_runtime,
            other_runtime=harness.peer_runtime,
            requested_mode="integrated",
            peer_translation_enabled=True,
            source_language="en",
            target_language="ko",
            other_source_language="en",
            other_target_language="ko",
        )

        assert mode == "integrated"
        assert "self 1" not in context
        assert context == (
            '- [peer] "peer 1"\n' '- [self] "self 2"\n' '- [peer] "peer 2"\n' '- [self] "self 3"'
        )

    def test_context_resolver_default_integrated_context_uses_40_second_window_and_latest_4_entries(
        self,
    ):
        self_runtime = ChannelRuntime(channel="self")
        peer_runtime = ChannelRuntime(channel="peer")
        self_runtime.remember_context(
            "41 seconds old", timestamp=59.0, source_language="en", target_language="ko"
        )
        self_runtime.remember_context(
            "recent 1", timestamp=61.0, source_language="en", target_language="ko"
        )
        peer_runtime.remember_context(
            "recent 2", timestamp=62.0, source_language="en", target_language="ko"
        )
        self_runtime.remember_context(
            "recent 3", timestamp=63.0, source_language="en", target_language="ko"
        )
        peer_runtime.remember_context(
            "recent 4", timestamp=64.0, source_language="en", target_language="ko"
        )
        self_runtime.remember_context(
            "recent 5", timestamp=65.0, source_language="en", target_language="ko"
        )
        resolver = ContextResolver(clock=FakeClock(initial_time=100.0))

        context, mode = resolver.resolve_for_request(
            runtime=self_runtime,
            other_runtime=peer_runtime,
            requested_mode="integrated",
            peer_translation_enabled=True,
            source_language="en",
            target_language="ko",
            other_source_language="en",
            other_target_language="ko",
        )

        assert mode == "integrated"
        assert "41 seconds old" not in context
        assert "recent 1" not in context
        assert context == (
            '- [peer] "recent 2"\n'
            '- [self] "recent 3"\n'
            '- [peer] "recent 4"\n'
            '- [self] "recent 5"'
        )

    def test_prepare_llm_request_formats_prompt_and_context(self):
        clock = FakeClock(initial_time=20.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            system_prompt="Translate ${sourceName} to ${targetName}",
        )
        harness.self_runtime.translation_history = [
            ContextEntry(text="안녕", source_language="ko", target_language="en", timestamp=19.0),
        ]

        prompt, context, now = harness.prepare_translation_request("입력")

        assert "${sourceName}" not in prompt
        assert "${targetName}" not in prompt
        assert context == '- [self] "안녕"'
        assert now == 20.0


class TestContextSerializationContract:
    def test_context_serialization_is_stable_as_time_advances(self):
        entry = ContextEntry(
            text="hello",
            source_language="en",
            target_language="ko",
            timestamp=100.0,
        )

        first = ContextResolver(clock=FakeClock(initial_time=105.0)).format_local([entry])
        second = ContextResolver(clock=FakeClock(initial_time=115.0)).format_local([entry])

        assert first == '- [self] "hello"'
        assert second == '- [self] "hello"'
        assert first == second

    def test_formatted_context_contains_no_timestamp_derived_syntax(self):
        entries = [
            ContextEntry(
                text="hello",
                source_language="en",
                target_language="ko",
                timestamp=100.0,
                channel="self",
            ),
            ContextEntry(
                text="hi",
                source_language="en",
                target_language="ko",
                timestamp=101.0,
                channel="peer",
            ),
        ]

        context = ContextResolver(clock=FakeClock(initial_time=110.0)).format_local(entries)

        assert context == '- [self] "hello"\n- [peer] "hi"'
        assert "ago" not in context
        assert "seconds" not in context
        assert "100.0" not in context
        assert "101.0" not in context

    @pytest.mark.asyncio
    async def test_provider_request_context_contains_no_relative_time_syntax(self):
        clock = FakeClock(initial_time=110.0)
        fake_llm = FakeLLMProvider()
        harness = compose_translation_test_harness(
            stt=None,
            llm=fake_llm,
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=60.0,
            context_max_entries=3,
        )
        harness.self_runtime.translation_history = [
            ContextEntry(text="hello", source_language="ko", target_language="en", timestamp=100.0),
        ]

        await harness.self_owner.submit_text("world")
        await asyncio.gather(
            *harness.self_runtime.translation_tasks.values(), return_exceptions=True
        )

        assert len(fake_llm.calls) == 1
        assert fake_llm.calls[0]["context"] == '- [self] "hello"'
        assert "ago" not in fake_llm.calls[0]["context"]
        assert "seconds" not in fake_llm.calls[0]["context"]


class TestContextLogging:
    def test_prepare_llm_request_without_runtime_logging_includes_redacted_context_summary(
        self, caplog: pytest.LogCaptureFixture
    ):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=20.0),
        )

        with caplog.at_level(logging.INFO, logger="puripuly_heart.core.orchestrator.translation"):
            harness.prepare_translation_request("입력")

        assert "[Translation] Context mode: channel=self mode=local" in caplog.messages
        assert (
            "[Translation] Context apply: channel=self mode=local "
            "request_chars=2 entries=0 self_entries=0 peer_entries=0 context_chars=0"
        ) in caplog.messages

    def test_prepare_llm_request_without_runtime_logging_redacts_local_context_text(
        self, caplog: pytest.LogCaptureFixture
    ):
        clock = FakeClock(initial_time=20.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
        )
        harness.self_runtime.remember_context(
            "secret context",
            timestamp=19.0,
            source_language="ko",
            target_language="en",
        )
        expected_context = '- [self] "secret context"'

        with caplog.at_level(logging.INFO, logger="puripuly_heart.core.orchestrator.translation"):
            harness.prepare_translation_request("secret request")

        assert "[Translation] Context mode: channel=self mode=local" in caplog.messages
        assert not any("secret request" in message for message in caplog.messages)
        assert not any("secret context" in message for message in caplog.messages)
        assert (
            "[Translation] Context apply: channel=self mode=local "
            f"request_chars=14 entries=1 self_entries=1 peer_entries=0 "
            f"context_chars={len(expected_context)}"
        ) in caplog.messages

    def test_prepare_llm_request_counts_peer_local_context_as_peer_entries(
        self, caplog: pytest.LogCaptureFixture
    ):
        clock = FakeClock(initial_time=20.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
        )
        harness.peer_runtime.remember_context(
            "secret peer context",
            timestamp=19.0,
            source_language="ko",
            target_language="en",
        )
        expected_context = '- [peer] "secret peer context"'

        with caplog.at_level(logging.INFO, logger="puripuly_heart.core.orchestrator.translation"):
            _, context, _ = harness.prepare_translation_request(
                "secret request", runtime=harness.peer_runtime
            )

        assert context == expected_context
        assert "[Translation] Context mode: channel=peer mode=local" in caplog.messages
        assert not any("secret request" in message for message in caplog.messages)
        assert not any("secret peer context" in message for message in caplog.messages)
        assert (
            "[Translation] Context apply: channel=peer mode=local "
            f"request_chars=14 entries=1 self_entries=0 peer_entries=1 "
            f"context_chars={len(expected_context)}"
        ) in caplog.messages

    def test_prepare_llm_request_without_runtime_logging_redacts_integrated_context_text(
        self, caplog: pytest.LogCaptureFixture
    ):
        clock = FakeClock(initial_time=20.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            integrated_context_enabled=True,
            peer_translation_enabled=True,
        )
        harness.self_runtime.remember_context(
            "secret self text",
            timestamp=19.0,
            source_language="ko",
            target_language="en",
        )
        harness.peer_runtime.remember_context(
            "secret peer text",
            timestamp=19.5,
            source_language="ko",
            target_language="en",
        )

        with caplog.at_level(logging.INFO, logger="puripuly_heart.core.orchestrator.translation"):
            harness.prepare_translation_request("secret request")

        apply_logs = [
            message
            for message in caplog.messages
            if message.startswith("[Translation] Context apply:")
        ]
        assert len(apply_logs) == 1
        assert "secret request" not in apply_logs[0]
        assert not any("secret self text" in message for message in caplog.messages)
        assert not any("secret peer text" in message for message in caplog.messages)
        assert "channel=self" in apply_logs[0]
        assert "mode=integrated" in apply_logs[0]
        assert "request_chars=14" in apply_logs[0]
        assert "entries=2" in apply_logs[0]
        assert "self_entries=1" in apply_logs[0]
        assert "peer_entries=1" in apply_logs[0]
        assert "context_chars=" in apply_logs[0]

    def test_prepare_llm_request_logs_context_mode_only_when_changed(
        self, caplog: pytest.LogCaptureFixture
    ):
        clock = FakeClock(initial_time=20.0)
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
        )
        harness.self_runtime.remember_context(
            "안녕",
            timestamp=19.0,
            source_language="ko",
            target_language="en",
        )

        with caplog.at_level(logging.INFO, logger="puripuly_heart.core.orchestrator.translation"):
            harness.prepare_translation_request("first")
            harness.prepare_translation_request("second")
            harness.replace_configuration(integrated_context_enabled=True)
            harness.replace_configuration(peer_translation_enabled=True)
            harness.peer_runtime.remember_context(
                "peer context",
                timestamp=19.5,
                source_language="ko",
                target_language="en",
            )
            harness.prepare_translation_request("third")
            harness.prepare_translation_request("fourth")

        mode_logs = [
            message
            for message in caplog.messages
            if message.startswith("[Translation] Context mode:")
        ]
        assert mode_logs == [
            "[Translation] Context mode: channel=self mode=local",
            "[Translation] Context mode: channel=self mode=integrated",
        ]

    @pytest.mark.asyncio
    async def test_submit_text_without_llm_enqueues_transcript_only(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=None,
            osc=FakeOscQueue(),
            clock=FakeClock(),
        )

        utterance_id = await harness.self_owner.submit_text("hello")
        bundle = harness.bundle_for(utterance_id)
        assert bundle.translation is None

        events = [await harness.ui_events.get(), await harness.ui_events.get()]
        assert [event.type for event in events] == [
            UIEventType.TRANSCRIPT_FINAL,
            UIEventType.OSC_SENT,
        ]

    @pytest.mark.asyncio
    async def test_ensure_translation_deduplicates_same_utterance(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
        )
        transcript = Transcript(utterance_id=uuid4(), text="hello", is_final=True)

        await harness.ensure_translation(transcript)
        await harness.ensure_translation(transcript)

        assert len(harness.llm_runtime.provider.provider.calls) == 1
        assert harness.translation_turns.is_parent_closed(transcript.utterance_id)
        assert harness.self_runtime.translation_tasks == {}

    @pytest.mark.asyncio
    async def test_in_flight_duplicate_peer_final_preserves_original_parent_cleanup(self):
        started = asyncio.Event()
        release = asyncio.Event()
        call_count = 0
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=12.0),
            peer_translation_enabled=True,
        )

        async def blocking_translate(**kwargs):
            nonlocal call_count
            call_count += 1
            started.set()
            await release.wait()
            return Translation(utterance_id=kwargs["utterance_id"], text="translated")

        harness.llm_runtime.provider.provider.translate = blocking_translate
        parent_id = uuid4()
        harness.peer_runtime.utterance_start_times[parent_id] = 12.0
        harness.peer_runtime.speech_ended_ids.add(parent_id)
        harness.peer_owner._peer_parent_speech_end_times[parent_id] = 12.0
        transcript = Transcript(
            utterance_id=parent_id,
            text="peer hello",
            is_final=True,
            channel="peer",
        )
        event = STTFinalEvent(utterance_id=parent_id, transcript=transcript)

        await harness.dispatch_stt_event(event)
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await harness.dispatch_stt_event(event)
        assert harness.translation_turns.is_parent_active(parent_id)

        release.set()
        await harness.translation_turns.wait_for_idle()

        assert call_count == 1
        assert harness.translation_turns.is_parent_closed(parent_id)
        assert not harness.translation_turns.is_parent_active(parent_id)
        assert parent_id not in harness.peer_owner._peer_translation_parent_ids
        assert parent_id not in harness.peer_owner._peer_parent_speech_end_times
        assert parent_id not in harness.peer_owner._peer_parent_turn_ids
        assert parent_id not in harness.peer_runtime.utterance_start_times
        assert parent_id not in harness.peer_runtime.speech_ended_ids

    @pytest.mark.asyncio
    async def test_closed_translation_owner_rejection_clears_peer_parent_marker(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
            peer_translation_enabled=True,
        )
        await harness.translation_turns.close()
        parent_id = uuid4()
        transcript = Transcript(
            utterance_id=parent_id,
            text="peer hello",
            is_final=True,
            channel="peer",
        )

        await harness.ensure_translation(transcript)

        assert not harness.translation_turns.is_parent_active(parent_id)
        assert not harness.translation_turns.is_parent_closed(parent_id)
        assert parent_id not in harness.peer_owner._peer_translation_parent_ids

    @pytest.mark.asyncio
    async def test_non_accepting_translation_owner_rejection_clears_peer_parent_marker(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
            peer_translation_enabled=True,
        )
        parent_id = uuid4()
        transcript = Transcript(
            utterance_id=parent_id,
            text="peer hello",
            is_final=True,
            channel="peer",
        )
        harness.translation_turns._accepting = False
        try:
            await harness.ensure_translation(transcript)
        finally:
            harness.translation_turns._accepting = True

        assert not harness.translation_turns.is_parent_active(parent_id)
        assert not harness.translation_turns.is_parent_closed(parent_id)
        assert parent_id not in harness.peer_owner._peer_translation_parent_ids

    @pytest.mark.asyncio
    async def test_blocked_peer_rejection_clears_peer_parent_marker(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
            peer_translation_enabled=True,
        )
        parent_id = uuid4()
        transcript = Transcript(
            utterance_id=parent_id,
            text="peer hello",
            is_final=True,
            channel="peer",
        )
        harness.translation_turns._blocked_channels.add("peer")
        try:
            await harness.ensure_translation(transcript)
        finally:
            harness.translation_turns._blocked_channels.discard("peer")

        assert not harness.translation_turns.is_parent_active(parent_id)
        assert not harness.translation_turns.is_parent_closed(parent_id)
        assert parent_id not in harness.peer_owner._peer_translation_parent_ids

    @pytest.mark.asyncio
    async def test_submit_text_translation_success_updates_bundle_and_events(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(response_text="OK"),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=1.0),
        )
        utterance_id = await harness.self_owner.submit_text("hello")
        await asyncio.gather(
            *harness.self_runtime.translation_tasks.values(), return_exceptions=True
        )

        bundle = harness.bundle_for(utterance_id)
        assert bundle.translation is not None
        assert bundle.translation.text == "OK"

        events = [
            await harness.ui_events.get(),
            await harness.ui_events.get(),
            await harness.ui_events.get(),
        ]
        assert [event.type for event in events] == [
            UIEventType.TRANSCRIPT_FINAL,
            UIEventType.TRANSLATION_DONE,
            UIEventType.OSC_SENT,
        ]

    @pytest.mark.asyncio
    async def test_peer_translation_stays_off_chatbox_on_success(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(response_text="OK"),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=112.0),
            integrated_context_enabled=True,
            peer_translation_enabled=True,
        )
        harness.replace_configuration(source_language="en")
        harness.replace_configuration(target_language="ko")
        transcript = Transcript(
            utterance_id=uuid4(),
            text="peer hello",
            is_final=True,
            channel="peer",
        )

        await harness.dispatch_stt_event(
            STTFinalEvent(utterance_id=transcript.utterance_id, transcript=transcript)
        )
        await harness.translation_turns.wait_for_idle()

        events = [await harness.ui_events.get(), await harness.ui_events.get()]
        translation_event = events[1]
        bundle = harness.bundle_for(translation_event.utterance_id, channel="peer")

        assert bundle.translation is not None
        assert harness.osc.messages == []
        assert [event.type for event in events] == [
            UIEventType.TRANSCRIPT_FINAL,
            UIEventType.TRANSLATION_DONE,
        ]

    @pytest.mark.asyncio
    async def test_peer_translation_error_fallback_does_not_publish_chatbox(self):
        harness = compose_translation_test_harness(
            stt=None,
            llm=FakeLLMProvider(response_text="OK"),
            osc=FakeOscQueue(),
            clock=FakeClock(initial_time=112.0),
            fallback_transcript_only=True,
            integrated_context_enabled=True,
            peer_translation_enabled=True,
        )
        harness.llm_runtime.attach_provider_reference(
            LlmTranslationBackend(FakeLLMProvider(response_text="OK"))
        )
        harness.replace_configuration(source_language="en")
        harness.replace_configuration(target_language="ko")
        transcript = Transcript(
            utterance_id=uuid4(),
            text="peer hello",
            is_final=True,
            channel="peer",
        )

        async def failing_translate(**kwargs):  # noqa: ANN003
            raise RuntimeError("boom")

        harness.llm_runtime.provider.provider.translate = failing_translate  # type: ignore[method-assign]

        await harness.ensure_translation(transcript)
        await asyncio.gather(
            *harness.peer_runtime.translation_tasks.values(), return_exceptions=True
        )

        assert harness.osc.messages == []
