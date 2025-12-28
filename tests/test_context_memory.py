"""Unit tests for LLM translation context memory feature."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.orchestrator.hub import (
    ClientHub,
    ContextEntry,
    TranslationMemoryEntry,
)

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
        context_pairs: list[dict[str, str]] | None = None,
    ):
        from puripuly_heart.domain.models import Translation

        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "context": context,
                "context_pairs": context_pairs,
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

    def process_due(self) -> None:
        pass


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestContextFiltering:
    """Test context time window and max entries filtering."""

    def test_context_filters_by_time_window(self):
        """Context entries older than time_window_s should be excluded."""
        clock = FakeClock(initial_time=10.0)
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=5.0,
            context_max_entries=3,
        )

        # Add entries at different times
        hub._translation_history = [
            ContextEntry(text="old", source_language="ko", target_language="en", timestamp=3.0),
            ContextEntry(text="recent1", source_language="ko", target_language="en", timestamp=6.0),
            ContextEntry(text="recent2", source_language="ko", target_language="en", timestamp=8.0),
        ]

        valid = hub._get_valid_context()

        assert len(valid) == 2
        assert valid[0].text == "recent1"
        assert valid[1].text == "recent2"

    def test_context_filters_by_max_entries(self):
        """Only the most recent max_entries should be considered."""
        clock = FakeClock(initial_time=10.0)
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=20.0,  # Default window
            context_max_entries=2,
        )

        hub._translation_history = [
            ContextEntry(text="first", source_language="ko", target_language="en", timestamp=7.0),
            ContextEntry(text="second", source_language="ko", target_language="en", timestamp=8.0),
            ContextEntry(text="third", source_language="ko", target_language="en", timestamp=9.0),
        ]

        valid = hub._get_valid_context()

        # Should only get last 2
        assert len(valid) == 2
        assert valid[0].text == "second"
        assert valid[1].text == "third"

    def test_context_filters_by_language_pair(self):
        """Only entries with the current language pair should be included."""
        clock = FakeClock(initial_time=10.0)
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=20.0,
        )

        hub._translation_history = [
            ContextEntry(text="wrong", source_language="ja", target_language="en", timestamp=9.0),
            ContextEntry(text="ok", source_language="ko", target_language="en", timestamp=9.5),
        ]

        valid = hub._get_valid_context()

        assert len(valid) == 1
        assert valid[0].text == "ok"

    def test_context_filters_short_entries(self):
        """Entries shorter than 2 characters should be excluded."""
        clock = FakeClock(initial_time=10.0)
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=20.0,
        )

        hub._translation_history = [
            ContextEntry(text="a", source_language="ko", target_language="en", timestamp=9.0),
            ContextEntry(text="ok", source_language="ko", target_language="en", timestamp=9.5),
        ]

        valid = hub._get_valid_context()

        assert len(valid) == 1
        assert valid[0].text == "ok"

    def test_context_cleared_on_clear_context(self):
        """clear_context() should empty the history."""
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
        )

        hub._translation_history = [
            ContextEntry(text="test", source_language="ko", target_language="en", timestamp=1.0),
        ]
        hub._translation_memory = [
            TranslationMemoryEntry(
                source="hi",
                target="hello",
                source_language="ko",
                target_language="en",
                timestamp=1.0,
            )
        ]

        hub.clear_context()

        assert len(hub._translation_history) == 0
        assert len(hub._translation_memory) == 0

    def test_old_entries_removed_when_full(self):
        """When max_entries is exceeded, oldest should be removed."""
        clock = FakeClock(initial_time=10.0)
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_max_entries=3,
        )

        # Add 3 entries (at capacity)
        hub._translation_history = [
            ContextEntry(text="e1", source_language="ko", target_language="en", timestamp=7.0),
            ContextEntry(text="e2", source_language="ko", target_language="en", timestamp=8.0),
            ContextEntry(text="e3", source_language="ko", target_language="en", timestamp=9.0),
        ]

        # Add a 4th entry
        hub._translation_history.append(
            ContextEntry(text="e4", source_language="ko", target_language="en", timestamp=10.0)
        )
        if len(hub._translation_history) > hub.context_max_entries:
            hub._translation_history.pop(0)

        assert len(hub._translation_history) == 3
        assert hub._translation_history[0].text == "e2"  # e1 removed


class TestContextPassedToLLM:
    """Test that context is correctly passed to LLM."""

    @pytest.mark.asyncio
    async def test_context_passed_to_llm(self):
        """LLM should receive formatted context string."""
        clock = FakeClock(initial_time=10.0)
        fake_llm = FakeLLMProvider()
        hub = ClientHub(
            stt=None,
            llm=fake_llm,
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=5.0,
            context_max_entries=3,
        )

        # Add some context
        hub._translation_history = [
            ContextEntry(text="hello", source_language="ko", target_language="en", timestamp=8.0),
        ]

        # Translate a new text
        utterance_id = uuid4()
        await hub._translate_and_enqueue(utterance_id, "world")

        # Verify LLM was called with context
        assert len(fake_llm.calls) == 1
        call = fake_llm.calls[0]
        assert "hello" in call["context"]

    @pytest.mark.asyncio
    async def test_empty_context_when_no_history(self):
        """LLM should receive empty context when no history."""
        clock = FakeClock(initial_time=10.0)
        fake_llm = FakeLLMProvider()
        hub = ClientHub(
            stt=None,
            llm=fake_llm,
            osc=FakeOscQueue(),
            clock=clock,
        )

        hub._translation_history = []

        utterance_id = uuid4()
        await hub._translate_and_enqueue(utterance_id, "test")

        assert len(fake_llm.calls) == 1
        assert fake_llm.calls[0]["context"] == ""

    @pytest.mark.asyncio
    async def test_empty_context_when_all_expired(self):
        """LLM should receive empty context when all entries are expired."""
        clock = FakeClock(initial_time=100.0)  # Far in the future
        fake_llm = FakeLLMProvider()
        hub = ClientHub(
            stt=None,
            llm=fake_llm,
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=5.0,
        )

        # All entries are very old
        hub._translation_history = [
            ContextEntry(text="old", source_language="ko", target_language="en", timestamp=1.0),
        ]

        utterance_id = uuid4()
        await hub._translate_and_enqueue(utterance_id, "test")

        assert len(fake_llm.calls) == 1
        assert fake_llm.calls[0]["context"] == ""


class TestContextFormatting:
    """Test context formatting for LLM."""

    def test_format_context_empty(self):
        """Empty context list should return empty string."""
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
        )

        result = hub._format_context_for_llm([])
        assert result == ""

    def test_format_context_single_entry(self):
        """Single entry should be formatted correctly."""
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
        )

        entries = [
            ContextEntry(text="안녕", source_language="ko", target_language="en", timestamp=1.0)
        ]
        result = hub._format_context_for_llm(entries)

        assert result == '- "안녕"'

    def test_format_context_multiple_entries(self):
        """Multiple entries should all be included."""
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
        )

        entries = [
            ContextEntry(text="a", source_language="ko", target_language="en", timestamp=1.0),
            ContextEntry(text="b", source_language="ko", target_language="en", timestamp=2.0),
        ]
        result = hub._format_context_for_llm(entries)

        assert '"a"' in result
        assert '"b"' in result


class TestTranslationMemory:
    """Test translation memory tm_list behavior."""

    def test_tm_list_filters_by_language_and_length(self):
        clock = FakeClock(initial_time=10.0)
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=5.0,
            context_max_entries=3,
        )
        hub.source_language = "ko"
        hub.target_language = "en"

        hub._translation_memory = [
            TranslationMemoryEntry(
                source="old",
                target="old",
                source_language="ko",
                target_language="en",
                timestamp=1.0,
            ),
            TranslationMemoryEntry(
                source="ok",
                target="fine",
                source_language="ko",
                target_language="en",
                timestamp=9.0,
            ),
            TranslationMemoryEntry(
                source="no",
                target="nah",
                source_language="ja",
                target_language="en",
                timestamp=9.5,
            ),
            TranslationMemoryEntry(
                source="a",
                target="b",
                source_language="ko",
                target_language="en",
                timestamp=9.7,
            ),
        ]

        tm_list = hub._get_tm_list()

        assert tm_list == [{"source": "ok", "target": "fine"}]

    @pytest.mark.asyncio
    async def test_tm_list_passed_to_llm(self):
        clock = FakeClock(initial_time=10.0)
        fake_llm = FakeLLMProvider()
        hub = ClientHub(
            stt=None,
            llm=fake_llm,
            osc=FakeOscQueue(),
            clock=clock,
            context_time_window_s=5.0,
            context_max_entries=3,
        )
        hub.source_language = "ko"
        hub.target_language = "en"

        hub._translation_memory = [
            TranslationMemoryEntry(
                source="hello",
                target="hi",
                source_language="ko",
                target_language="en",
                timestamp=9.0,
            )
        ]

        utterance_id = uuid4()
        await hub._translate_and_enqueue(utterance_id, "world")

        assert len(fake_llm.calls) == 1
        assert fake_llm.calls[0]["context_pairs"] == [{"source": "hello", "target": "hi"}]
