from __future__ import annotations

import re

from puripuly_heart.app.ports.agent_chat import AgentPresentation

_CLOSERS = "\"'”’»)]}*_"
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "e.g", "i.e", "etc", "vs"})


def completed_sentences(text: str, *, final: bool = False) -> tuple[str, ...]:
    sentences = []
    start = 0
    index = 0
    code = 0
    while index < len(text):
        char = text[index]
        if char == "`":
            end = index + 1
            while end < len(text) and text[end] == "`":
                end += 1
            width = end - index
            if not code:
                code = width
            elif code == width:
                code = 0
            index = end
            continue
        if code or char not in ".!?。！？":
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in ".!?。！？" + _CLOSERS:
            end += 1
        following = text[end : end + 1]
        if following and not following.isspace() and ord(following) < 0x3000:
            index += 1
            continue
        token_start = max(text.rfind(" ", start, index), text.rfind("\n", start, index)) + 1
        token = text[token_start:index]
        if not following and (
            "https://" in token or "http://" in token or token.startswith("www.")
        ):
            index += 1
            continue
        line_start = text.rfind("\n", start, index) + 1
        if char == "." and (
            (index and text[index - 1].isdigit() and (not following or following.isdigit()))
            or re.fullmatch(r"\s*\d+", text[line_start:index]) is not None
            or token.casefold() in _ABBREVIATIONS
            or re.fullmatch(r"[A-Za-z]", token) is not None
        ):
            index += 1
            continue
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
        index = end
    if final and text[start:].strip():
        sentences.append(text[start:].strip())
    return tuple(sentences)


class AgentPresentationBuffer:
    def __init__(self, generation: int, response_index: int) -> None:
        self._generation = generation
        self._response_index = response_index
        self._text = ""
        self.snapshot = AgentPresentation(generation, response_index, ())

    def append(self, text: str) -> bool:
        self._text += text
        sentences = completed_sentences(self._text)
        if sentences == self.snapshot.sentences:
            return False
        self.snapshot = AgentPresentation(self._generation, self._response_index, sentences)
        return True

    def complete(self, text: str) -> None:
        self._text = ""
        self.snapshot = AgentPresentation(
            self._generation, self._response_index, completed_sentences(text, final=True), True
        )
