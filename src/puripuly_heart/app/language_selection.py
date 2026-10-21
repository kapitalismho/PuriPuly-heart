from collections.abc import Sequence
from dataclasses import dataclass

RECENT_LANGUAGE_QUEUE_LIMIT = 3


def unify_recent_language_codes(*groups: Sequence[str]) -> list[str]:
    """Fold newest-first language lists into one queue of at most three codes."""
    merged: list[str] = []
    for group in groups:
        for code in group:
            if not isinstance(code, str):
                continue
            normalized = code.strip()
            if not normalized or normalized in merged:
                continue
            merged.append(normalized)
            if len(merged) >= RECENT_LANGUAGE_QUEUE_LIMIT:
                return merged
    return merged


@dataclass(frozen=True, slots=True)
class LanguageSelectionChange:
    source_code: str
    target_code: str
    peer_source_code: str
    peer_target_code: str
    peer_source_mode: str
    recent_source_codes: tuple[str, ...]
    recent_target_codes: tuple[str, ...]
    secondary_target_code: str = ""
