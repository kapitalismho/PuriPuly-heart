from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ChannelId = Literal["self", "peer"]
U64_MAX = (1 << 64) - 1
# `active_peer` remains a reserved compatibility/fallback variant. Normal
# product peer rows are primary-visible only after translation arrival.
BlockVariant = Literal["active_self", "active_peer", "finalized"]


@dataclass(frozen=True, slots=True)
class OverlayPresentationCalibration:
    anchor: str = "head_locked"
    offset_x: float = 0.0
    offset_y: float = -0.45
    distance: float = 1.1
    text_scale: float = 1.0
    background_alpha: float = 0.24

    def to_dict(self) -> dict[str, object]:
        return {
            "anchor": self.anchor,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "distance": self.distance,
            "text_scale": self.text_scale,
            "background_alpha": self.background_alpha,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OverlayPresentationCalibration":
        return cls(
            anchor=str(data.get("anchor", "head_locked")),
            offset_x=float(data.get("offset_x", 0.0)),
            offset_y=float(data.get("offset_y", -0.45)),
            distance=float(data.get("distance", 1.1)),
            text_scale=float(data.get("text_scale", 1.0)),
            background_alpha=float(data.get("background_alpha", 0.24)),
        )


@dataclass(frozen=True, slots=True)
class OverlayPresentationBlock:
    id: str
    occupant_key: str
    appearance_seq: int
    channel: ChannelId
    block_variant: BlockVariant
    primary_text: str
    secondary_text: str
    secondary_enabled: bool
    primary_language: str | None = None
    secondary_language: str | None = None
    update_id: str | None = None
    origin_wall_clock_ms: int | None = None
    session_scope: str | None = None
    source_text_hash: str | None = None
    source_text_len: int | None = None
    logical_turn_key: str | None = None
    publication_scope: str | None = None
    publication_generation: int | None = None
    publication_order: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "occupant_key": self.occupant_key,
            "appearance_seq": self.appearance_seq,
            "channel": self.channel,
            "block_variant": self.block_variant,
            "primary_text": self.primary_text,
            "secondary_text": self.secondary_text,
            "secondary_enabled": self.secondary_enabled,
        }
        if self.update_id is not None:
            payload["update_id"] = self.update_id
        if self.primary_language is not None:
            payload["primary_language"] = self.primary_language
        if self.secondary_language is not None:
            payload["secondary_language"] = self.secondary_language
        if self.origin_wall_clock_ms is not None:
            payload["origin_wall_clock_ms"] = self.origin_wall_clock_ms
        if self.session_scope is not None:
            payload["session_scope"] = self.session_scope
        if self.source_text_hash is not None:
            payload["source_text_hash"] = self.source_text_hash
        if self.source_text_len is not None:
            payload["source_text_len"] = self.source_text_len
        if self.logical_turn_key is not None:
            payload["logical_turn_key"] = self.logical_turn_key
        if self.publication_scope is not None:
            payload["publication_scope"] = self.publication_scope
        if self.publication_generation is not None:
            payload["publication_generation"] = self.publication_generation
        if self.publication_order is not None:
            payload["publication_order"] = self.publication_order
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OverlayPresentationBlock":
        if not isinstance(data, dict):
            raise ValueError("overlay presentation block must be an object")
        channel = data.get("channel")
        if channel not in ("self", "peer"):
            raise ValueError(f"invalid overlay presentation channel: {channel!r}")
        block_variant = data.get("block_variant")
        if block_variant not in ("active_self", "active_peer", "finalized"):
            raise ValueError(f"invalid overlay presentation block variant: {block_variant!r}")
        if block_variant == "active_self" and channel != "self":
            raise ValueError("active_self blocks require channel='self'")
        if block_variant == "active_peer" and channel != "peer":
            raise ValueError("active_peer blocks require channel='peer'")
        occupant_key = _require_string_field(data, "occupant_key").strip()
        if not occupant_key:
            raise ValueError("occupant_key must be a non-empty string")
        appearance_seq = _require_non_negative_int_field(data, "appearance_seq")
        return cls(
            id=_require_string_field(data, "id"),
            occupant_key=occupant_key,
            appearance_seq=appearance_seq,
            channel=channel,
            block_variant=block_variant,
            primary_text=_require_string_field(data, "primary_text"),
            secondary_text=_require_string_field(data, "secondary_text"),
            secondary_enabled=_require_bool_field(data, "secondary_enabled"),
            primary_language=_optional_string_field(data, "primary_language"),
            secondary_language=_optional_string_field(data, "secondary_language"),
            update_id=_optional_string_field(data, "update_id"),
            origin_wall_clock_ms=_optional_int_field(data, "origin_wall_clock_ms"),
            session_scope=_optional_string_field(data, "session_scope"),
            source_text_hash=_optional_string_field(data, "source_text_hash"),
            source_text_len=_optional_int_field(data, "source_text_len"),
            logical_turn_key=_optional_string_field(data, "logical_turn_key"),
            publication_scope=_optional_non_empty_string_field(data, "publication_scope"),
            publication_generation=_optional_non_negative_int_field(
                data, "publication_generation"
            ),
            publication_order=_optional_non_negative_int_field(data, "publication_order"),
        )


@dataclass(frozen=True, slots=True)
class SemanticRetirementFrontier:
    scope: str
    generation: int
    order: int

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "generation": self.generation,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SemanticRetirementFrontier":
        scope = _require_string_field(data, "scope").strip()
        if not scope:
            raise ValueError("semantic retirement frontier scope must be non-empty")
        return cls(
            scope=scope,
            generation=_require_non_negative_int_field(data, "generation"),
            order=_require_non_negative_int_field(data, "order"),
        )


@dataclass(frozen=True, slots=True)
class NativeFreshRenderGenerations:
    self: int | None = None
    peer: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_generation(self.self, "self")
        _validate_optional_generation(self.peer, "peer")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.self is not None:
            payload["self"] = self.self
        if self.peer is not None:
            payload["peer"] = self.peer
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "NativeFreshRenderGenerations":
        if not isinstance(data, dict):
            raise ValueError("native fresh render generations must be an object")
        return cls(
            self=_optional_non_negative_int_field(data, "self"),
            peer=_optional_non_negative_int_field(data, "peer"),
        )


@dataclass(frozen=True, slots=True)
class NativeFreshRenderTargets:
    self: str | None = None
    peer: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.self is not None:
            payload["self"] = self.self
        if self.peer is not None:
            payload["peer"] = self.peer
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "NativeFreshRenderTargets":
        if not isinstance(data, dict):
            raise ValueError("native fresh render targets must be an object")
        return cls(
            self=_optional_non_empty_string_field(data, "self"),
            peer=_optional_non_empty_string_field(data, "peer"),
        )


@dataclass(frozen=True, slots=True)
class NativeQuietTailEpisode:
    phase: str
    generation: int

    def __post_init__(self) -> None:
        if self.phase not in {"stream", "final"}:
            raise ValueError("native quiet tail episode phase must be stream or final")
        _validate_optional_generation(self.generation, "generation")

    def to_dict(self) -> dict[str, object]:
        return {"phase": self.phase, "generation": self.generation}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "NativeQuietTailEpisode":
        if not isinstance(data, dict):
            raise ValueError("native quiet tail episode must be an object")
        return cls(
            phase=_require_string_field(data, "phase"),
            generation=_require_non_negative_int_field(data, "generation"),
        )


@dataclass(frozen=True, slots=True)
class NativeQuietTailEpisodes:
    self: NativeQuietTailEpisode | None = None
    peer: NativeQuietTailEpisode | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.self is not None:
            payload["self"] = self.self.to_dict()
        if self.peer is not None:
            payload["peer"] = self.peer.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "NativeQuietTailEpisodes":
        if not isinstance(data, dict):
            raise ValueError("native quiet tail episodes must be an object")
        return cls(
            self=(NativeQuietTailEpisode.from_dict(data["self"]) if "self" in data else None),
            peer=(NativeQuietTailEpisode.from_dict(data["peer"]) if "peer" in data else None),
        )


@dataclass(frozen=True, slots=True)
class OverlayPresentationSnapshot:
    revision: int = 0
    calibration: OverlayPresentationCalibration = field(
        default_factory=OverlayPresentationCalibration
    )
    blocks: list[OverlayPresentationBlock] = field(default_factory=list)
    native_fresh_render_generations: NativeFreshRenderGenerations | None = None
    native_fresh_render_targets: NativeFreshRenderTargets | None = None
    native_quiet_tail_episodes: NativeQuietTailEpisodes | None = None
    semantic_retirement_frontiers: list[SemanticRetirementFrontier] = field(default_factory=list)

    def __post_init__(self) -> None:
        generations = self.native_fresh_render_generations
        if generations is not None and generations.self is None and generations.peer is None:
            object.__setattr__(self, "native_fresh_render_generations", None)
        targets = self.native_fresh_render_targets
        if targets is not None and targets.self is None and targets.peer is None:
            object.__setattr__(self, "native_fresh_render_targets", None)
        episodes = self.native_quiet_tail_episodes
        if episodes is not None and episodes.self is None and episodes.peer is None:
            object.__setattr__(self, "native_quiet_tail_episodes", None)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "revision": self.revision,
            "calibration": self.calibration.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
        }
        if self.native_fresh_render_generations is not None:
            payload["native_fresh_render_generations"] = (
                self.native_fresh_render_generations.to_dict()
            )
        if self.native_fresh_render_targets is not None:
            payload["native_fresh_render_targets"] = self.native_fresh_render_targets.to_dict()
        if self.native_quiet_tail_episodes is not None:
            payload["native_quiet_tail_episodes"] = self.native_quiet_tail_episodes.to_dict()
        if self.semantic_retirement_frontiers:
            payload["semantic_retirement_frontiers"] = [
                frontier.to_dict() for frontier in self.semantic_retirement_frontiers
            ]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OverlayPresentationSnapshot":
        calibration = data.get("calibration", {})
        if not isinstance(calibration, dict):
            raise ValueError("overlay presentation snapshot calibration must be an object")

        raw_blocks = data.get("blocks", [])
        if not isinstance(raw_blocks, list):
            raise ValueError("overlay presentation snapshot blocks must be a list")

        blocks: list[OverlayPresentationBlock] = []
        for block in raw_blocks:
            if not isinstance(block, dict):
                raise ValueError(
                    "overlay presentation snapshot blocks must contain only dict items"
                )
            blocks.append(OverlayPresentationBlock.from_dict(block))

        raw_generations = data.get("native_fresh_render_generations")
        if raw_generations is not None and not isinstance(raw_generations, dict):
            raise ValueError("native fresh render generations must be an object")
        raw_targets = data.get("native_fresh_render_targets")
        if raw_targets is not None and not isinstance(raw_targets, dict):
            raise ValueError("native fresh render targets must be an object")
        raw_episodes = data.get("native_quiet_tail_episodes")
        if raw_episodes is not None and not isinstance(raw_episodes, dict):
            raise ValueError("native quiet tail episodes must be an object")
        raw_frontiers = data.get("semantic_retirement_frontiers", [])
        if not isinstance(raw_frontiers, list) or not all(
            isinstance(frontier, dict) for frontier in raw_frontiers
        ):
            raise ValueError("semantic retirement frontiers must be a list of objects")
        return cls(
            revision=int(data.get("revision", 0)),
            calibration=OverlayPresentationCalibration.from_dict(calibration),
            blocks=blocks,
            native_fresh_render_generations=(
                NativeFreshRenderGenerations.from_dict(raw_generations)
                if raw_generations is not None
                else None
            ),
            native_fresh_render_targets=(
                NativeFreshRenderTargets.from_dict(raw_targets) if raw_targets is not None else None
            ),
            native_quiet_tail_episodes=(
                NativeQuietTailEpisodes.from_dict(raw_episodes)
                if raw_episodes is not None
                else None
            ),
            semantic_retirement_frontiers=[
                SemanticRetirementFrontier.from_dict(frontier)
                for frontier in raw_frontiers
                if isinstance(frontier, dict)
            ],
        )


def _require_string_field(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_bool_field(data: dict[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a bool")
    return value


def _require_non_negative_int_field(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an int")
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _optional_string_field(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_non_empty_string_field(data: dict[str, object], key: str) -> str | None:
    value = _optional_string_field(data, key)
    if value is not None and not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_int_field(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an int")
    return value


def _optional_non_negative_int_field(data: dict[str, object], key: str) -> int | None:
    value = _optional_int_field(data, key)
    _validate_optional_generation(value, key)
    return value


def _validate_optional_generation(value: int | None, key: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an int")
    if value < 0 or value > U64_MAX:
        raise ValueError(f"{key} must be between 0 and {U64_MAX}")
