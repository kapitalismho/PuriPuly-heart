from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from experiments.speaker_turn_boundary.adapters.eres2netv2 import cosine_similarity

from .phase4_design import ceil_grid
from .schemas import ProposalEvent


class Phase5ProposalError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if value.size != 192 or not np.isfinite(value).all() or norm == 0.0:
        raise Phase5ProposalError("invalid embedding")
    return value / norm


def state_hash(state: dict[str, Any]) -> str:
    def encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            array = np.asarray(value, dtype="<f4")
            return {
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        if isinstance(value, dict):
            return {key: encode(item) for key, item in sorted(value.items())}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, tuple):
            return [encode(item) for item in value]
        return value

    return content_sha256(encode(state))


def proposal(
    profile: dict[str, Any],
    *,
    source_session_id: str,
    audio_epoch: int,
    boundary: int,
    observed: int,
    confidence: float,
    state_provenance: dict[str, Any],
    debug_evidence: dict[str, Any],
) -> dict[str, Any]:
    provenance_id = str(state_provenance["mode"]) + ":" + content_sha256(state_provenance)[:32]
    identity = {
        "profile_id": profile["proposal_profile_id"],
        "audio_epoch": audio_epoch,
        "source_session_id": source_session_id,
        "boundary_source_sample": boundary,
        "observed_source_sample_at_emit": observed,
        "state_provenance": provenance_id,
        "debug_evidence": debug_evidence,
    }
    row = {
        "proposal_id": "proposal:" + content_sha256(identity)[:32],
        "family": "eres2netv2",
        "checkpoint": str(profile["checkpoint"]),
        "profile_id": str(profile["proposal_profile_id"]),
        "audio_epoch": audio_epoch,
        "source_session_id": source_session_id,
        "proposal_kind": "speaker_change_unknown",
        "boundary_source_sample": boundary,
        "observed_source_sample_at_emit": observed,
        "emitted_monotonic_ns": 0,
        "confidence": min(1.0, max(0.0, float(confidence))),
        "confidence_semantics_id": str(profile["confidence_semantics_id"]),
        "state_provenance": provenance_id,
        "debug_evidence": {
            **debug_evidence,
            "state_provenance_evidence": state_provenance,
        },
    }
    ProposalEvent.from_dict(row)
    return row


def embedding(
    embeddings: Mapping[tuple[int, int], np.ndarray], window: tuple[int, int]
) -> np.ndarray:
    try:
        return normalized(embeddings[window])
    except KeyError as error:
        raise Phase5ProposalError(f"embedding window missing: {window}") from error


def append_progress(
    rows: list[dict[str, Any]],
    *,
    audio_epoch: int,
    observed: int,
    safe: int,
) -> None:
    if rows and observed < int(rows[-1]["observed_source_sample"]):
        raise Phase5ProposalError("observed frontier regressed")
    if rows and safe < int(rows[-1]["safe_boundary_frontier_sample"]):
        raise Phase5ProposalError("safe frontier regressed")
    if safe > observed:
        raise Phase5ProposalError("safe frontier exceeds observation")
    rows.append(
        {
            "audio_epoch": audio_epoch,
            "observed_source_sample": observed,
            "safe_boundary_frontier_sample": safe,
        }
    )


def adjacent_trace(
    embeddings: Mapping[tuple[int, int], np.ndarray],
    profile: dict[str, Any],
    *,
    source_session_id: str,
    audio_epoch: int,
    warm_start: int,
    tail_end: int,
) -> dict[str, Any]:
    window = int(profile["window_samples"])
    step = int(profile["step_samples"])
    threshold = float(profile["proposal_threshold"]["value"])
    confirmation = profile["confirmation"]
    direct = confirmation == "direct_each_qualifying_probe"
    if not direct and int(confirmation) not in (1, 2):
        raise Phase5ProposalError("invalid adjacent confirmation")
    required = 1 if direct else int(confirmation)
    events: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    latched = False
    first = ceil_grid(warm_start + window, step)
    for boundary in range(first, tail_end - window + 1, step):
        left_window = (boundary - window, boundary)
        right_window = (boundary, boundary + window)
        left = embedding(embeddings, left_window)
        right = embedding(embeddings, right_window)
        change_score = 1.0 - cosine_similarity(left, right)
        qualifies = change_score > threshold
        before = state_hash({"pending": pending, "latched": latched})
        emit: dict[str, Any] | None = None
        if direct:
            if qualifies:
                emit = {
                    "boundary": boundary,
                    "observed": boundary + window,
                    "scores": [change_score],
                    "positions": [boundary],
                }
        elif latched:
            if not qualifies:
                latched = False
            pending = None
        elif required == 1:
            if qualifies:
                emit = {
                    "boundary": boundary,
                    "observed": boundary + window,
                    "scores": [change_score],
                    "positions": [boundary],
                }
                latched = True
            pending = None
        elif qualifies and pending is None:
            pending = {"score": change_score, "boundary": boundary}
        elif qualifies and pending is not None:
            emit = {
                "boundary": int(pending["boundary"]),
                "observed": boundary + window,
                "scores": [float(pending["score"]), change_score],
                "positions": [int(pending["boundary"]), boundary],
            }
            pending = None
            latched = True
        else:
            pending = None
        after = state_hash({"pending": pending, "latched": latched})
        if emit is not None:
            events.append(
                proposal(
                    profile,
                    source_session_id=source_session_id,
                    audio_epoch=audio_epoch,
                    boundary=int(emit["boundary"]),
                    observed=int(emit["observed"]),
                    confidence=float(sum(emit["scores"]) / len(emit["scores"])),
                    state_provenance={
                        "mode": "episode_reset",
                        "pre_state_sha256": before,
                        "post_state_sha256": after,
                    },
                    debug_evidence={
                        "change_scores": emit["scores"],
                        "confirmation_boundaries": emit["positions"],
                        "left_window": list(left_window),
                        "right_window": list(right_window),
                    },
                )
            )
        safe = int(pending["boundary"]) - 1 if pending is not None else boundary
        append_progress(
            progress,
            audio_epoch=audio_epoch,
            observed=boundary + window,
            safe=max(warm_start, safe),
        )
    append_progress(
        progress,
        audio_epoch=audio_epoch,
        observed=tail_end,
        safe=tail_end,
    )
    return {
        "proposals": events,
        "progress": progress,
        "tail_evidence": {
            "tail_closed": True,
            "pending_confirmation_suppressed": pending is not None,
            "pending_boundary_source_sample": (
                int(pending["boundary"]) if pending is not None else None
            ),
        },
    }


def ema(left: np.ndarray, right: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    return normalized((1.0 - alpha) * left + alpha * right)


def historical_anchor_step(
    state: dict[str, Any],
    profile: dict[str, Any],
    probe: np.ndarray,
    window: tuple[int, int],
) -> dict[str, Any] | None:
    threshold = float(profile["proposal_threshold"]["value"])
    confirmation = int(profile["confirmation"])
    anchor = state["anchor"]
    score = 1.0 - cosine_similarity(anchor, probe)
    qualifies = score > threshold
    pending = state.get("pending")
    emitted: dict[str, Any] | None = None
    if confirmation == 1:
        if qualifies:
            emitted = {
                "boundary": window[0],
                "scores": [score],
                "positions": [window[0]],
                "mutual_similarity": None,
            }
            state["anchor"] = probe
        elif profile.get("anchor_update") == "ema":
            state["anchor"] = ema(anchor, probe, float(profile["anchor_ema_alpha"]))
        state["pending"] = None
        return emitted
    if qualifies and pending is None:
        state["pending"] = {"score": score, "embedding": probe, "window": window}
    elif qualifies and pending is not None:
        mutual = cosine_similarity(pending["embedding"], probe)
        if mutual >= float(profile["mutual_similarity_threshold"]):
            emitted = {
                "boundary": int(pending["window"][0]),
                "scores": [float(pending["score"]), score],
                "positions": [int(pending["window"][0]), window[0]],
                "mutual_similarity": mutual,
            }
            state["anchor"] = pending["embedding"]
            state["pending"] = None
        else:
            state["pending"] = {"score": score, "embedding": probe, "window": window}
    else:
        state["pending"] = None
        if profile.get("anchor_update") == "ema":
            state["anchor"] = ema(anchor, probe, float(profile["anchor_ema_alpha"]))
    return emitted


def native_anchor_step(
    state: dict[str, Any],
    profile: dict[str, Any],
    probe: np.ndarray,
    window: tuple[int, int],
) -> dict[str, Any] | None:
    mode = str(profile["profile_class"])
    threshold = float(profile["proposal_threshold"]["value"])
    if mode == "prototype_memory_4":
        selected = max(
            state["prototypes"],
            key=lambda row: (cosine_similarity(row["embedding"], probe), -row["ordinal"]),
        )
        score = 1.0 - cosine_similarity(selected["embedding"], probe)
    else:
        selected = None
        score = 1.0 - cosine_similarity(state["anchor"], probe)
    pending = state.get("pending")
    emitted: dict[str, Any] | None = None
    if mode in ("stable_no_update", "stable_ema"):
        if score > threshold:
            emitted = {
                "boundary": window[0],
                "scores": [score],
                "positions": [window[0]],
                "mutual_similarity": None,
                "selected_prototype_ordinal": None,
            }
        if mode == "stable_ema" and score <= 0.30:
            state["anchor"] = ema(state["anchor"], probe)
        return emitted
    if mode == "confirmed_anchor":
        if score > threshold:
            if pending is not None:
                mutual = cosine_similarity(pending["embedding"], probe)
                if mutual >= 0.50:
                    emitted = {
                        "boundary": int(pending["window"][0]),
                        "scores": [float(pending["score"]), score],
                        "positions": [int(pending["window"][0]), window[0]],
                        "mutual_similarity": mutual,
                        "selected_prototype_ordinal": None,
                    }
                    state["anchor"] = pending["embedding"]
                    state["pending"] = None
                    return emitted
            state["pending"] = {"embedding": probe, "window": window, "score": score}
        else:
            state["pending"] = None
            if score <= 0.30:
                state["anchor"] = ema(state["anchor"], probe)
        return emitted
    if mode != "prototype_memory_4" or selected is None:
        raise Phase5ProposalError(f"unsupported native profile class: {mode}")
    similarity = 1.0 - score
    if similarity >= 0.70:
        selected["embedding"] = ema(selected["embedding"], probe)
        selected["window"] = window
        state["pending"] = None
    elif similarity >= 0.50:
        state["pending"] = None
    elif pending is None:
        state["pending"] = {"embedding": probe, "window": window, "score": score}
    else:
        mutual = cosine_similarity(pending["embedding"], probe)
        if mutual >= 0.50:
            emitted = {
                "boundary": int(pending["window"][0]),
                "scores": [float(pending["score"]), score],
                "positions": [int(pending["window"][0]), window[0]],
                "mutual_similarity": mutual,
                "selected_prototype_ordinal": int(selected["ordinal"]),
            }
            ordinal = int(state["next_ordinal"])
            state["next_ordinal"] = ordinal + 1
            if len(state["prototypes"]) >= 4:
                oldest = min(state["prototypes"], key=lambda row: row["ordinal"])
                state["prototypes"].remove(oldest)
            state["prototypes"].append(
                {
                    "ordinal": ordinal,
                    "embedding": pending["embedding"],
                    "window": pending["window"],
                }
            )
            state["pending"] = None
        else:
            state["pending"] = {"embedding": probe, "window": window, "score": score}
    return emitted


def anchor_trace(
    embeddings: Mapping[tuple[int, int], np.ndarray],
    profile: dict[str, Any],
    *,
    source_session_id: str,
    audio_epoch: int,
    replay_start: int,
    warm_start: int,
    tail_end: int,
) -> dict[str, Any]:
    window_samples = int(profile["window_samples"])
    step = int(profile["step_samples"])
    first = ceil_grid(replay_start + window_samples, step)
    state: dict[str, Any] = {"anchor": None, "pending": None}
    events: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    for end in range(first, tail_end + 1, step):
        probe_window = (end - window_samples, end)
        probe = embedding(embeddings, probe_window)
        if state["anchor"] is None:
            state["anchor"] = probe
            state["anchor_window"] = probe_window
            if profile["profile_class"] == "prototype_memory_4":
                state["prototypes"] = [{"ordinal": 0, "embedding": probe, "window": probe_window}]
                state["next_ordinal"] = 1
            continue
        before = state_hash(state)
        if profile["origin"] == "historical_phase3_profile":
            emitted = historical_anchor_step(state, profile, probe, probe_window)
        else:
            emitted = native_anchor_step(state, profile, probe, probe_window)
        after = state_hash(state)
        if end < warm_start:
            continue
        if emitted is not None and int(emitted["boundary"]) >= warm_start:
            events.append(
                proposal(
                    profile,
                    source_session_id=source_session_id,
                    audio_epoch=audio_epoch,
                    boundary=int(emitted["boundary"]),
                    observed=end,
                    confidence=float(sum(emitted["scores"]) / len(emitted["scores"])),
                    state_provenance={
                        "mode": "source_prefix",
                        "replay_start": replay_start,
                        "pre_state_sha256": before,
                        "post_state_sha256": after,
                    },
                    debug_evidence={
                        "change_scores": emitted["scores"],
                        "confirmation_boundaries": emitted["positions"],
                        "mutual_similarity": emitted["mutual_similarity"],
                        "probe_window": list(probe_window),
                        "selected_prototype_ordinal": emitted.get("selected_prototype_ordinal"),
                    },
                )
            )
        pending = state.get("pending")
        safe = int(pending["window"][0]) - 1 if pending is not None else probe_window[0]
        append_progress(
            progress,
            audio_epoch=audio_epoch,
            observed=end,
            safe=max(warm_start, safe),
        )
    append_progress(
        progress,
        audio_epoch=audio_epoch,
        observed=tail_end,
        safe=tail_end,
    )
    pending = state.get("pending")
    return {
        "proposals": events,
        "progress": progress,
        "final_state_sha256": state_hash(state),
        "tail_evidence": {
            "tail_closed": True,
            "pending_confirmation_suppressed": pending is not None,
            "pending_boundary_source_sample": (
                int(pending["window"][0]) if pending is not None else None
            ),
        },
    }


def source_prefix_routes(
    embeddings: Mapping[tuple[int, int], np.ndarray],
    profile: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    if not episodes or profile["scored_state_mode"] != "source_prefix":
        raise Phase5ProposalError(
            "source-prefix routes require episodes and a source-prefix profile"
        )
    source_sessions = {str(row["session_id"]) for row in episodes}
    if len(source_sessions) != 1:
        raise Phase5ProposalError("source-prefix routes must share one source session")
    episode_ids = [str(row["episode_id"]) for row in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise Phase5ProposalError("source-prefix route episode identity collision")
    window_samples = int(profile["window_samples"])
    step = int(profile["step_samples"])
    maximum_tail = max(int(row["bounds"]["tail_end"]) for row in episodes)
    first = ceil_grid(window_samples, step)
    source_session_id = next(iter(source_sessions))
    execution_identity = {
        "proposal_profile_id": str(profile["proposal_profile_id"]),
        "source_session_id": source_session_id,
        "maximum_tail_end_sample": maximum_tail,
        "window_samples": window_samples,
        "step_samples": step,
    }
    execution_id = "source-prefix:" + content_sha256(execution_identity)
    route_by_id: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        route_by_id[episode_id] = {
            "episode": episode,
            "warm_start": int(episode["bounds"]["warm_start"]),
            "tail_end": int(episode["bounds"]["tail_end"]),
            "proposals": [],
            "progress": [],
            "final_state_sha256": None,
            "tail_evidence": None,
            "finalized": False,
        }
    warm_order = sorted(
        route_by_id,
        key=lambda episode_id: (
            int(route_by_id[episode_id]["warm_start"]),
            episode_id,
        ),
    )
    tail_order = sorted(
        route_by_id,
        key=lambda episode_id: (
            int(route_by_id[episode_id]["tail_end"]),
            episode_id,
        ),
    )
    warm_index = 0
    tail_index = 0
    active: set[str] = set()
    state: dict[str, Any] = {"anchor": None, "pending": None}

    def finalize(episode_id: str) -> None:
        route = route_by_id[episode_id]
        if route["finalized"]:
            return
        episode = route["episode"]
        tail_end = int(route["tail_end"])
        append_progress(
            route["progress"],
            audio_epoch=int(episode["audio_epoch"]),
            observed=tail_end,
            safe=tail_end,
        )
        pending = state.get("pending")
        route["final_state_sha256"] = state_hash(state)
        route["tail_evidence"] = {
            "tail_closed": True,
            "pending_confirmation_suppressed": pending is not None,
            "pending_boundary_source_sample": (
                int(pending["window"][0]) if pending is not None else None
            ),
        }
        route["finalized"] = True
        active.discard(episode_id)

    for end in range(first, maximum_tail + 1, step):
        while (
            tail_index < len(tail_order)
            and int(route_by_id[tail_order[tail_index]]["tail_end"]) < end
        ):
            finalize(tail_order[tail_index])
            tail_index += 1
        while (
            warm_index < len(warm_order)
            and int(route_by_id[warm_order[warm_index]]["warm_start"]) <= end
        ):
            episode_id = warm_order[warm_index]
            if not route_by_id[episode_id]["finalized"]:
                active.add(episode_id)
            warm_index += 1
        probe_window = (end - window_samples, end)
        probe = embedding(embeddings, probe_window)
        if state["anchor"] is None:
            state["anchor"] = probe
            state["anchor_window"] = probe_window
            if profile["profile_class"] == "prototype_memory_4":
                state["prototypes"] = [{"ordinal": 0, "embedding": probe, "window": probe_window}]
                state["next_ordinal"] = 1
            while (
                tail_index < len(tail_order)
                and int(route_by_id[tail_order[tail_index]]["tail_end"]) == end
            ):
                finalize(tail_order[tail_index])
                tail_index += 1
            continue
        before = state_hash(state)
        if profile["origin"] == "historical_phase3_profile":
            emitted = historical_anchor_step(state, profile, probe, probe_window)
        else:
            emitted = native_anchor_step(state, profile, probe, probe_window)
        after = state_hash(state)
        pending = state.get("pending")
        safe = int(pending["window"][0]) - 1 if pending is not None else probe_window[0]
        for episode_id in sorted(active):
            route = route_by_id[episode_id]
            if end > int(route["tail_end"]):
                continue
            episode = route["episode"]
            warm_start = int(route["warm_start"])
            if emitted is not None and int(emitted["boundary"]) >= warm_start:
                route["proposals"].append(
                    proposal(
                        profile,
                        source_session_id=source_session_id,
                        audio_epoch=int(episode["audio_epoch"]),
                        boundary=int(emitted["boundary"]),
                        observed=end,
                        confidence=float(sum(emitted["scores"]) / len(emitted["scores"])),
                        state_provenance={
                            "mode": "source_prefix",
                            "replay_start": 0,
                            "pre_state_sha256": before,
                            "post_state_sha256": after,
                        },
                        debug_evidence={
                            "change_scores": emitted["scores"],
                            "confirmation_boundaries": emitted["positions"],
                            "mutual_similarity": emitted["mutual_similarity"],
                            "probe_window": list(probe_window),
                            "selected_prototype_ordinal": emitted.get("selected_prototype_ordinal"),
                        },
                    )
                )
            append_progress(
                route["progress"],
                audio_epoch=int(episode["audio_epoch"]),
                observed=end,
                safe=max(warm_start, safe),
            )
        while (
            tail_index < len(tail_order)
            and int(route_by_id[tail_order[tail_index]]["tail_end"]) == end
        ):
            finalize(tail_order[tail_index])
            tail_index += 1
    while tail_index < len(tail_order):
        finalize(tail_order[tail_index])
        tail_index += 1
    routes: list[dict[str, Any]] = []
    for episode_id in sorted(route_by_id):
        route = route_by_id[episode_id]
        ordered = sorted(
            route["proposals"],
            key=lambda row: (
                int(row["observed_source_sample_at_emit"]),
                int(row["boundary_source_sample"]),
                str(row["profile_id"]),
                str(row["proposal_id"]),
            ),
        )
        routes.append(
            {
                "episode_id": episode_id,
                "audio_epoch": int(route["episode"]["audio_epoch"]),
                "proposals": ordered,
                "progress": route["progress"],
                "proposal_count": len(ordered),
                "proposal_trace_sha256": content_sha256(ordered),
                "progress_trace_sha256": content_sha256(route["progress"]),
                "final_state_sha256": route["final_state_sha256"],
                "tail_evidence": route["tail_evidence"],
            }
        )
    return {
        "source_prefix_execution_id": execution_id,
        "source_session_id": source_session_id,
        "proposal_profile_id": str(profile["proposal_profile_id"]),
        "probe_step_count": len(range(first, maximum_tail + 1, step)),
        "route_count": len(routes),
        "routes": routes,
        "route_index_sha256": content_sha256(
            [
                {
                    "episode_id": row["episode_id"],
                    "audio_epoch": row["audio_epoch"],
                    "proposal_trace_sha256": row["proposal_trace_sha256"],
                    "progress_trace_sha256": row["progress_trace_sha256"],
                    "final_state_sha256": row["final_state_sha256"],
                }
                for row in routes
            ]
        ),
    }


def generate_proposal_trace(
    embeddings: Mapping[tuple[int, int], np.ndarray],
    profile: dict[str, Any],
    episode: dict[str, Any],
) -> dict[str, Any]:
    bounds = episode["bounds"]
    warm_start = int(bounds["warm_start"])
    tail_end = int(bounds["tail_end"])
    source_session_id = str(episode["session_id"])
    audio_epoch = int(episode["audio_epoch"])
    if profile["profile_class"] == "adjacent":
        trace = adjacent_trace(
            embeddings,
            profile,
            source_session_id=source_session_id,
            audio_epoch=audio_epoch,
            warm_start=warm_start,
            tail_end=tail_end,
        )
    else:
        replay_start = 0 if profile["scored_state_mode"] == "source_prefix" else warm_start
        trace = anchor_trace(
            embeddings,
            profile,
            source_session_id=source_session_id,
            audio_epoch=audio_epoch,
            replay_start=replay_start,
            warm_start=warm_start,
            tail_end=tail_end,
        )
    ordered = sorted(
        trace["proposals"],
        key=lambda row: (
            int(row["observed_source_sample_at_emit"]),
            int(row["boundary_source_sample"]),
            str(row["profile_id"]),
            str(row["proposal_id"]),
        ),
    )
    if ordered != trace["proposals"]:
        raise Phase5ProposalError("proposal output order drift")
    return {
        **trace,
        "proposal_count": len(ordered),
        "proposal_trace_sha256": content_sha256(ordered),
        "progress_trace_sha256": content_sha256(trace["progress"]),
    }
