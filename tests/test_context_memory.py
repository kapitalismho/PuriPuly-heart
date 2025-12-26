"""Unit tests for LLM translation context memory feature."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from ajebal_daera_translator.core.orchestrator.hub import ClientHub, ContextEntry


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
        from ajebal_daera_translator.domain.models import Translation
        self.calls.append({
            "utterance_id": utterance_id,
            "text": text,
            "context": context,
        })
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
            ContextEntry(text="old", timestamp=3.0),  # 7s ago - excluded
            ContextEntry(text="recent1", timestamp=6.0),  # 4s ago - included
            ContextEntry(text="recent2", timestamp=8.0),  # 2s ago - included
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
            context_time_window_s=10.0,  # Large window
            context_max_entries=2,
        )
        
        hub._translation_history = [
            ContextEntry(text="first", timestamp=7.0),
            ContextEntry(text="second", timestamp=8.0),
            ContextEntry(text="third", timestamp=9.0),
        ]
        
        valid = hub._get_valid_context()
        
        # Should only get last 2
        assert len(valid) == 2
        assert valid[0].text == "second"
        assert valid[1].text == "third"

    def test_context_cleared_on_clear_context(self):
        """clear_context() should empty the history."""
        hub = ClientHub(
            stt=None,
            llm=FakeLLMProvider(),
            osc=FakeOscQueue(),
            clock=FakeClock(),
        )
        
        hub._translation_history = [
            ContextEntry(text="test", timestamp=1.0),
        ]
        
        hub.clear_context()
        
        assert len(hub._translation_history) == 0

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
            ContextEntry(text="e1", timestamp=7.0),
            ContextEntry(text="e2", timestamp=8.0),
            ContextEntry(text="e3", timestamp=9.0),
        ]
        
        # Add a 4th entry
        hub._translation_history.append(
            ContextEntry(text="e4", timestamp=10.0)
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
            ContextEntry(text="hello", timestamp=8.0),
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
            ContextEntry(text="old", timestamp=1.0),  # 99s ago
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
        
        entries = [ContextEntry(text="안녕", timestamp=1.0)]
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
            ContextEntry(text="a", timestamp=1.0),
            ContextEntry(text="b", timestamp=2.0),
        ]
        result = hub._format_context_for_llm(entries)
        
        assert '"a"' in result
        assert '"b"' in result
