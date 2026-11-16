from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

Quad = tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]


class PhotoParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PhotoBlock:
    quad: Quad
    source: str
    translation: str


@dataclass(frozen=True, slots=True)
class QuadReady:
    index: int
    quad: Quad
    kind: Literal["quad_ready"] = field(default="quad_ready", init=False)


@dataclass(frozen=True, slots=True)
class SourceReady:
    index: int
    quad: Quad
    source: str
    source_line_count: int
    kind: Literal["source_ready"] = field(default="source_ready", init=False)


@dataclass(frozen=True, slots=True)
class BlockComplete:
    index: int
    quad: Quad
    source: str
    translation: str
    kind: Literal["block_complete"] = field(default="block_complete", init=False)


PhotoParseEvent = QuadReady | SourceReady | BlockComplete

_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_WHITESPACE = " \t\r\n"
_PUNCTUATION = {
    "root_colon": (":", "blocks_start"),
    "block_colon": (":", "field_value"),
    "root_end": ("}", "done"),
}


class PhotoBlockParser:
    def __init__(self) -> None:
        self._fields = ("quad", "source", "translation")
        self._state = "preamble"
        self._depth = 0
        self._string: list[str] | None = None
        self._escaped = False
        self._number: list[str] | None = None
        self._coordinates: list[float] = []
        self._points: list[tuple[float, float]] = []
        self._field_name: str | None = None
        self._seen_fields: set[str] = set()
        self._source_emitted = False
        self._quad: Quad | None = None
        self._source: str | None = None
        self._translation: str | None = None
        self._blocks: list[PhotoBlock] = []
        self._error: PhotoParseError | None = None
        self._finished = False
        self._offset = 0

    def feed(self, delta: str) -> list[PhotoParseEvent]:
        if self._finished:
            raise PhotoParseError("cannot feed a finished photo response")
        if not isinstance(delta, str):
            raise TypeError("photo delta must be a string")
        events: list[PhotoParseEvent] = []
        if self._error is not None:
            return events
        for char in delta:
            self._offset += 1
            try:
                self._consume(char, events)
            except PhotoParseError as error:
                self._error = error
                break
        return events

    def finish(self) -> list[PhotoBlock]:
        if self._error is not None:
            raise self._error
        if self._state != "done" or self._depth != 0 or self._string is not None:
            self._fail("incomplete photo response")
        self._finished = True
        return list(self._blocks)

    def _fail(self, message: str) -> None:
        self._error = PhotoParseError(f"{message} at character {self._offset}")
        raise self._error

    def _consume(self, char: str, events: list[PhotoParseEvent]) -> None:
        if self._string is not None:
            self._string.append(char)
            if self._escaped:
                self._escaped = False
            elif char == "\\":
                self._escaped = True
            elif char == '"':
                token = "".join(self._string)
                self._string = None
                try:
                    value = json.loads(token)
                except ValueError:
                    self._fail("invalid JSON string")
                self._accept_string(value, events)
            return
        if self._number is not None:
            if char in _WHITESPACE or char in ",]":
                token = "".join(self._number)
                self._number = None
                if not _NUMBER.fullmatch(token):
                    self._fail("quad coordinates must be JSON numbers")
                try:
                    self._coordinates.append(float(token))
                except ValueError:
                    self._fail("invalid quad number")
                self._state = "point_separator"
            else:
                self._number.append(char)
                return
        if self._state == "preamble":
            if char == "[":
                self._fail("photo response must be an object, not a top-level list")
            if char == "{":
                self._depth = 1
                self._state = "root_key"
            return
        if self._state == "done":
            return
        if char in _WHITESPACE:
            return
        if self._state in {"root_key", "block_key"}:
            if char != '"':
                self._fail("expected a JSON field name")
            self._string = ['"']
            return
        if self._state in _PUNCTUATION:
            expected, next_state = _PUNCTUATION[self._state]
            if char != expected:
                self._fail(f"expected {expected!r}")
            if char == "}":
                self._depth -= 1
            self._state = next_state
            return
        if self._state == "blocks_start":
            if char != "[":
                self._fail("blocks must be an array")
            self._depth += 1
            self._state = "block_or_end"
            return
        if self._state in {"block_or_end", "block_required"}:
            if char == "]" and self._state == "block_or_end":
                self._depth -= 1
                self._state = "root_end"
                return
            if char != "{":
                self._fail("expected a block object")
            self._depth += 1
            self._field_name = None
            self._seen_fields.clear()
            self._source_emitted = False
            self._coordinates = []
            self._points = []
            self._quad = None
            self._source = None
            self._translation = None
            self._state = "block_key"
            return
        if self._state == "field_value":
            if self._field_name == self._fields[0]:
                if char != "[":
                    self._fail("quad must be an array")
                self._depth += 1
                self._state = "point_start"
            else:
                if char != '"':
                    self._fail(f"{self._field_name} must be a string")
                self._string = ['"']
            return
        if self._state == "point_start":
            if char != "[":
                self._fail("quad must contain exactly four coordinate pairs")
            self._depth += 1
            self._coordinates = []
            self._state = "point_number"
            return
        if self._state == "point_number":
            if char not in "-0123456789":
                self._fail("quad coordinates must be JSON numbers")
            self._number = [char]
            return
        if self._state == "point_separator":
            if char == "," and len(self._coordinates) == 1:
                self._state = "point_number"
            elif char == "]" and len(self._coordinates) == 2:
                self._depth -= 1
                self._points.append((self._coordinates[0], self._coordinates[1]))
                self._state = "quad_separator"
            else:
                self._fail("quad points must contain exactly two numbers")
            return
        if self._state == "quad_separator":
            if char == "," and len(self._points) < 4:
                self._state = "point_start"
            elif char == "]" and len(self._points) == 4:
                self._depth -= 1
                a, b, c, d = self._points
                self._quad = (a, b, c, d)
                events.append(QuadReady(index=len(self._blocks), quad=self._quad))
                self._emit_source(events)
                self._state = "field_separator"
            else:
                self._fail("quad must contain exactly four coordinate pairs")
            return
        if self._state == "field_separator":
            if char == ",":
                self._state = "block_key"
                return
            if char != "}":
                self._fail("expected a comma or end of block")
            if self._quad is None or self._source is None or self._translation is None:
                self._fail("incomplete block")
            self._depth -= 1
            block = PhotoBlock(self._quad, self._source, self._translation)
            events.append(
                BlockComplete(
                    index=len(self._blocks),
                    quad=block.quad,
                    source=block.source,
                    translation=block.translation,
                )
            )
            self._blocks.append(block)
            self._state = "blocks_separator"
            return
        if self._state == "blocks_separator":
            if char == ",":
                self._state = "block_required"
            elif char == "]":
                self._depth -= 1
                self._state = "root_end"
            else:
                self._fail("expected a comma or end of blocks")
            return
        self._fail("invalid photo response")

    def _accept_string(self, value: str, events: list[PhotoParseEvent]) -> None:
        if self._state == "root_key":
            if value != "blocks":
                self._fail("photo response must contain only blocks")
            self._state = "root_colon"
        elif self._state == "block_key":
            if value not in self._fields:
                self._fail(f"unexpected block field {value!r}")
            if value in self._seen_fields:
                self._fail(f"duplicate block field {value!r}")
            self._seen_fields.add(value)
            self._field_name = value
            self._state = "block_colon"
        elif self._state == "field_value" and self._field_name == "source":
            self._source = value
            self._emit_source(events)
            self._state = "field_separator"
        elif self._state == "field_value" and self._field_name == "translation":
            self._translation = value
            self._state = "field_separator"
        else:
            self._fail("unexpected JSON string")

    def _emit_source(self, events: list[PhotoParseEvent]) -> None:
        if self._quad is not None and self._source is not None and not self._source_emitted:
            events.append(
                SourceReady(
                    index=len(self._blocks),
                    quad=self._quad,
                    source=self._source,
                    source_line_count=self._source.count("\n") + 1,
                )
            )
            self._source_emitted = True
