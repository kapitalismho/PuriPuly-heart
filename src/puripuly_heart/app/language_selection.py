from dataclasses import dataclass


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
