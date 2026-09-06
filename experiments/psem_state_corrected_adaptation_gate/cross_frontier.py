from __future__ import annotations

from typing import Any, Mapping, Sequence


class CrossFrontierError(RuntimeError):
    pass


ARTIFACT_ROLE = "issue-121-cross-arm-dev-frontier"
CANONICAL_VERSION = 1

HORIZONS_MS = (100, 300, 500)
GROUPS = ("macro", "ami", "alimeeting", "pooled")
GROUP_ORDER = ("macro", "ami", "alimeeting", "pooled")
KINDS = ("raw", "calibrated")

RAW_REFERENCE_THRESHOLD = 0.5
REFERENCE_KIND = "raw_f0_at_0.5"
FRONTIER_SLICE_LIMIT = 16


def bounded_threshold_slice(grid: Sequence[Any], limit: int = FRONTIER_SLICE_LIMIT) -> list[float]:
    ordered = [float(t) for t in grid]
    count = int(limit)
    if count <= 0:
        raise CrossFrontierError("frontier slice limit is not positive")
    if len(ordered) <= count:
        return list(ordered)
    last = len(ordered) - 1
    out: list[float] = []
    for i in range(count):
        threshold = ordered[round(i * last / (count - 1))]
        if threshold not in out:
            out.append(threshold)
    return out


def project_frontier_cost(measured_seconds: float, sampled: int, total: int) -> float:
    if int(sampled) <= 0 or int(total) <= 0:
        raise CrossFrontierError("frontier slice counts are not positive")
    return float(measured_seconds) * float(total) / float(sampled)

BASELINE_ARM = {"R-T2-SC": "R-H-SC", "R-TA-SC": "R-T2-SC"}

_POINT_FIELDS = ("threshold", "false_cuts_per_hour", "contamination", "miss_rate")


def normalize_point(payload: Any, what: str) -> dict[str, float]:
    if isinstance(payload, dict):
        source: Mapping[str, Any] = payload
    else:
        get = getattr(payload, "__getattribute__", None)
        try:
            source = {field: get(field) for field in _POINT_FIELDS}
        except AttributeError as exc:
            raise CrossFrontierError(f"frontier {what} is malformed: {exc}") from exc
    try:
        point = {field: float(source[field]) for field in _POINT_FIELDS}
    except (KeyError, TypeError, ValueError) as exc:
        raise CrossFrontierError(f"frontier {what} is malformed: {exc}") from exc
    return point


def _check_reference(reference: dict[str, float], what: str) -> dict[str, float]:
    if float(reference["threshold"]) != float(RAW_REFERENCE_THRESHOLD):
        raise CrossFrontierError(
            f"frontier {what} reference threshold is {reference['threshold']}, "
            f"required {RAW_REFERENCE_THRESHOLD}"
        )
    return reference


def check_budget(budget: Any, reference: Mapping[str, float], what: str) -> float:
    try:
        value = float(budget)
    except (TypeError, ValueError) as exc:
        raise CrossFrontierError(f"frontier {what} budget is malformed: {exc}") from exc
    if value != float(reference["false_cuts_per_hour"]):
        raise CrossFrontierError(
            f"frontier {what} budget {value} differs from the raw-F0@0.5 "
            f"reference rate {float(reference['false_cuts_per_hour'])}"
        )
    return value


def select_envelopes(
    reference: Mapping[str, float], points: Sequence[Mapping[str, float]]
) -> dict[str, Any]:
    budget = float(reference["false_cuts_per_hour"])
    within = [dict(p) for p in points if float(p["false_cuts_per_hour"]) <= budget]
    if not within:
        return {"budget": budget, "c_envelope": None, "m_envelope": None, "useful": False}
    c_best = min(within, key=lambda p: (float(p["contamination"]), float(p["miss_rate"])))
    m_best = min(within, key=lambda p: (float(p["miss_rate"]), float(p["contamination"])))
    useful = any(
        float(p["contamination"]) < float(reference["contamination"])
        and float(p["miss_rate"]) < float(reference["miss_rate"])
        for p in (c_best, m_best)
    )
    return {"budget": budget, "c_envelope": c_best, "m_envelope": m_best, "useful": bool(useful)}


def build_block(
    points: Sequence[Any],
    reference: Any,
    diagnostics: Mapping[str, Any] | None = None,
    what: str = "block",
) -> dict[str, Any]:
    norm_points = [normalize_point(p, f"{what} point") for p in points]
    if not norm_points:
        raise CrossFrontierError(f"frontier {what} has no exact candidate points")
    norm_ref = _check_reference(normalize_point(reference, f"{what} reference"), what)
    envelopes = select_envelopes(norm_ref, norm_points)
    return {
        "points": norm_points,
        "reference": norm_ref,
        "reference_kind": REFERENCE_KIND,
        "budget": envelopes["budget"],
        "c_envelope": envelopes["c_envelope"],
        "m_envelope": envelopes["m_envelope"],
        "useful": envelopes["useful"],
        "diagnostics": dict(diagnostics or {}),
    }


def validate_block(block: Any, what: str) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise CrossFrontierError(f"frontier {what} is not an object")
    points_raw = block.get("points")
    if not isinstance(points_raw, list) or not points_raw:
        raise CrossFrontierError(f"frontier {what} has no exact candidate points")
    points = [normalize_point(p, f"{what} point") for p in points_raw]
    if "reference" not in block:
        raise CrossFrontierError(f"frontier {what} lacks the fixed F0 reference")
    reference = _check_reference(normalize_point(block["reference"], f"{what} reference"), what)
    if block.get("reference_kind", REFERENCE_KIND) != REFERENCE_KIND:
        raise CrossFrontierError(
            f"frontier {what} reference_kind is {block.get('reference_kind')!r}, "
            f"required {REFERENCE_KIND!r}"
        )
    if "budget" not in block:
        raise CrossFrontierError(f"frontier {what} lacks the fixed budget")
    budget = check_budget(block["budget"], reference, what)
    expected = select_envelopes(reference, points)
    for which in ("c_envelope", "m_envelope"):
        got = block.get(which)
        want = expected[which]
        if want is None:
            if got is not None:
                raise CrossFrontierError(
                    f"frontier {what} {which} should be absent (nothing within budget)"
                )
        else:
            if got is None:
                raise CrossFrontierError(
                    f"frontier {what} {which} is missing (a within-budget point exists)"
                )
            norm = normalize_point(got, f"{what} {which}")
            for field in _POINT_FIELDS:
                if norm[field] != float(want[field]):
                    raise CrossFrontierError(
                        f"frontier {what} {which}.{field} differs from the "
                        f"within-budget envelope ({norm[field]} != {float(want[field])})"
                    )
    if bool(block.get("useful")) != bool(expected["useful"]):
        raise CrossFrontierError(f"frontier {what} usefulness flag is inconsistent")
    diagnostics = block.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise CrossFrontierError(f"frontier {what} diagnostics are not an object")
    return {
        "points": points,
        "reference": reference,
        "reference_kind": REFERENCE_KIND,
        "budget": budget,
        "c_envelope": expected["c_envelope"],
        "m_envelope": expected["m_envelope"],
        "useful": bool(expected["useful"]),
        "diagnostics": dict(diagnostics),
    }


def validate_canonical(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise CrossFrontierError("canonical frontier document is not an object")
    if doc.get("artifact_role") != ARTIFACT_ROLE:
        raise CrossFrontierError(
            f"canonical frontier artifact_role is {doc.get('artifact_role')!r}, "
            f"required {ARTIFACT_ROLE!r}"
        )
    horizons_ms = doc.get("horizons_ms")
    if [int(h) for h in (horizons_ms or [])] != [int(h) for h in HORIZONS_MS]:
        raise CrossFrontierError(
            f"canonical frontier horizons_ms must be {list(HORIZONS_MS)}"
        )
    horizons = doc.get("horizons")
    if not isinstance(horizons, dict):
        raise CrossFrontierError("canonical frontier lacks horizons")
    parsed: dict[str, Any] = {}
    for horizon_ms in HORIZONS_MS:
        horizon = str(horizon_ms)
        groups = horizons.get(horizon)
        if not isinstance(groups, dict):
            raise CrossFrontierError(f"canonical frontier lacks horizon {horizon}")
        if set(groups) != set(GROUP_ORDER):
            raise CrossFrontierError(
                f"canonical frontier groups at horizon {horizon} are {sorted(groups)}, "
                f"required {list(GROUP_ORDER)}"
            )
        parsed[horizon] = {}
        for group in GROUP_ORDER:
            kinds = groups[group]
            if not isinstance(kinds, dict):
                raise CrossFrontierError(
                    f"canonical frontier lacks kinds for {group} at horizon {horizon}"
                )
            if set(kinds) != set(KINDS):
                raise CrossFrontierError(
                    f"canonical frontier kinds for {group} at horizon {horizon} "
                    f"are {sorted(kinds)}, required {list(KINDS)}"
                )
            parsed[horizon][group] = {}
            for kind in KINDS:
                parsed[horizon][group][kind] = validate_block(
                    kinds[kind], f"{group}/{kind} at horizon {horizon}"
                )
    sources = doc.get("sources", {})
    if not isinstance(sources, dict):
        raise CrossFrontierError("canonical frontier sources are not an object")
    return {
        "artifact_role": ARTIFACT_ROLE,
        "version": int(doc.get("version", CANONICAL_VERSION)),
        "arm": str(doc.get("arm", "")),
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "horizons": parsed,
        "sources": dict(sources),
    }


def load_canonical_frontier(path: Any, expected_arm: str) -> dict[str, Any]:
    from pathlib import Path as _Path

    try:
        payload = __import__("json").loads(_Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CrossFrontierError(f"canonical frontier is unreadable: {exc}") from exc
    validated = validate_canonical(payload)
    if validated["arm"] != str(expected_arm):
        raise CrossFrontierError(
            f"canonical frontier arm is {validated['arm']!r}, required {str(expected_arm)!r}"
        )
    return validated


def macro_average_points(
    ami_points: Sequence[Any], ali_points: Sequence[Any], what: str = "macro"
) -> list[dict[str, float]]:
    ami = [normalize_point(p, f"{what}/ami point") for p in ami_points]
    ali = [normalize_point(p, f"{what}/alimeeting point") for p in ali_points]
    ami_grid = [p["threshold"] for p in ami]
    ali_grid = [p["threshold"] for p in ali]
    if ami_grid != ali_grid:
        raise CrossFrontierError(
            f"{what} evaluated grids differ: ami {ami_grid} != alimeeting {ali_grid}"
        )
    if not ami_grid:
        raise CrossFrontierError(f"{what} has no evaluated thresholds")
    ami_by_t = {p["threshold"]: p for p in ami}
    ali_by_t = {p["threshold"]: p for p in ali}
    averaged = []
    for threshold in ami_grid:
        left, right = ami_by_t[threshold], ali_by_t[threshold]
        averaged.append(
            {
                "threshold": float(threshold),
                "false_cuts_per_hour": (
                    float(left["false_cuts_per_hour"]) + float(right["false_cuts_per_hour"])
                )
                / 2.0,
                "contamination": (
                    float(left["contamination"]) + float(right["contamination"])
                )
                / 2.0,
                "miss_rate": (float(left["miss_rate"]) + float(right["miss_rate"])) / 2.0,
            }
        )
    return averaged


def macro_average_reference(
    ami_reference: Any, ali_reference: Any, what: str = "macro"
) -> dict[str, float]:
    left = _check_reference(normalize_point(ami_reference, f"{what}/ami reference"), what)
    right = _check_reference(
        normalize_point(ali_reference, f"{what}/alimeeting reference"), what
    )
    return {
        "threshold": float(RAW_REFERENCE_THRESHOLD),
        "false_cuts_per_hour": (
            float(left["false_cuts_per_hour"]) + float(right["false_cuts_per_hour"])
        )
        / 2.0,
        "contamination": (float(left["contamination"]) + float(right["contamination"]))
        / 2.0,
        "miss_rate": (float(left["miss_rate"]) + float(right["miss_rate"])) / 2.0,
    }


def pooled_point_from_sums(
    total_cuts: float,
    total_seconds: float,
    total_refs: float,
    total_missed: float,
    total_contamination_seconds: float,
    threshold: float,
) -> dict[str, float]:
    if total_seconds <= 0:
        raise CrossFrontierError("pooled group has no active speech")
    if total_refs <= 0:
        raise CrossFrontierError("pooled group has no reference replacements")
    hours = float(total_seconds) / 3600.0
    return {
        "threshold": float(threshold),
        "false_cuts_per_hour": float(total_cuts) / hours,
        "contamination": float(total_contamination_seconds) / hours,
        "miss_rate": float(total_missed) / float(total_refs),
    }


def _delta_point(
    front: Mapping[str, float] | None, back: Mapping[str, float] | None
) -> dict[str, float] | None:
    if front is None or back is None:
        return None
    return {
        "threshold": float(front["threshold"]),
        "d_false_cuts_per_hour": float(front["false_cuts_per_hour"])
        - float(back["false_cuts_per_hour"]),
        "d_contamination": float(front["contamination"]) - float(back["contamination"]),
        "d_miss_rate": float(front["miss_rate"]) - float(back["miss_rate"]),
    }


def compare_candidate_to_baseline(
    candidate_points: Sequence[Any],
    baseline_points: Sequence[Any],
    f0_reference: Any,
    arm: str,
    baseline: str,
    corpora: Mapping[str, Any],
    what: str = "comparison",
) -> dict[str, Any]:
    reference = _check_reference(
        normalize_point(f0_reference, f"{what} F0 reference"), what
    )
    candidate = select_envelopes(
        reference, [normalize_point(p, f"{what} candidate point") for p in candidate_points]
    )
    base = select_envelopes(
        reference,
        [normalize_point(p, f"{what} baseline point") for p in baseline_points],
    )
    depth = {
        "c_envelope": base["c_envelope"],
        "m_envelope": base["m_envelope"],
        "useful": base["useful"],
    }
    delta = {
        "c_envelope": _delta_point(candidate["c_envelope"], base["c_envelope"]),
        "m_envelope": _delta_point(candidate["m_envelope"], base["m_envelope"]),
    }
    return {
        "arm": str(arm),
        "baseline": str(baseline),
        "budget": float(reference["false_cuts_per_hour"]),
        "budget_threshold": float(RAW_REFERENCE_THRESHOLD),
        "budget_kind": REFERENCE_KIND,
        "candidate": {
            "c_envelope": candidate["c_envelope"],
            "m_envelope": candidate["m_envelope"],
            "useful": candidate["useful"],
        },
        "c_envelope": candidate["c_envelope"],
        "m_envelope": candidate["m_envelope"],
        "useful": candidate["useful"],
        "baseline_depth": depth,
        "delta_vs_baseline": delta,
        "delta_vs_f0": {
            "c_envelope": _delta_point(candidate["c_envelope"], reference),
            "m_envelope": _delta_point(candidate["m_envelope"], reference),
        },
        "corpora": dict(corpora),
    }


EXACT_THRESHOLD_CHUNK = 128

_EXACT_CONTEXT: dict[str, Any] = {}


def _as_score_array(values: Any) -> Any:
    import numpy as _np

    if isinstance(values, _np.ndarray) and values.dtype == _np.float64:
        return values
    return _np.asarray(values, dtype=_np.float64)


def init_exact_context(entries: Mapping[str, Any]) -> None:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    arm_runtime.spawn_worker_init()
    manifest = dict(entries)
    _EXACT_CONTEXT.clear()
    if manifest and all(
        isinstance(value, dict) and "dev" not in value and "path" in value
        for value in manifest.values()
    ):
        _EXACT_CONTEXT["manifest"] = {
            str(key): str(value["path"]) for key, value in manifest.items()
        }
        _EXACT_CONTEXT["cached"] = None
        return
    _EXACT_CONTEXT["members"] = {
        str(key): {
            "dev": value["dev"],
            "scores": {
                str(kind): _as_score_array(scores)
                for kind, scores in dict(value["scores"]).items()
            },
        }
        for key, value in manifest.items()
    }
    _EXACT_CONTEXT["cached"] = None


def _ensure_exact_member(member: str) -> tuple[Any, dict[str, Any]]:
    members = _EXACT_CONTEXT.get("members")
    if isinstance(members, dict) and str(member) in members:
        entry = members[str(member)]
        return entry["dev"], entry["scores"]
    manifest = _EXACT_CONTEXT.get("manifest")
    if not isinstance(manifest, dict) or str(member) not in manifest:
        raise CrossFrontierError(f"frontier member is unknown: {member}")
    if _EXACT_CONTEXT.get("cached") != str(member):
        import pickle as _pickle

        with open(manifest[str(member)], "rb") as handle:
            entry = _pickle.load(handle)
        _EXACT_CONTEXT["dev"] = entry["dev"]
        _EXACT_CONTEXT["scores"] = {
            str(kind): _as_score_array(scores)
            for kind, scores in dict(entry["scores"]).items()
        }
        _EXACT_CONTEXT["cached"] = str(member)
    return _EXACT_CONTEXT["dev"], _EXACT_CONTEXT["scores"]


def _representative_thresholds(
    scores: Any, ordered: Sequence[float]
) -> tuple[list[float], list[int]]:
    import numpy as _np

    flat = [float(t) for t in ordered]
    if not flat:
        return [], []
    levels = _np.sort(_np.asarray(scores, dtype=_np.float64).ravel())
    counts = levels.size - _np.searchsorted(
        levels, _np.asarray(flat, dtype=_np.float64), side="left"
    )
    seen: dict[int, int] = {}
    representatives: list[float] = []
    positions: list[int] = []
    for position, count in enumerate([int(c) for c in counts]):
        slot = seen.get(count)
        if slot is None:
            slot = len(representatives)
            seen[count] = slot
            representatives.append(flat[position])
        positions.append(slot)
    return representatives, positions


def index_threshold_rows(rows: Sequence[Mapping[str, Any]]) -> dict[float, Mapping[str, Any]]:
    indexed: dict[float, Mapping[str, Any]] = {}
    for row in rows:
        threshold = float(row["threshold"])
        if threshold in indexed:
            raise CrossFrontierError(f"duplicate frontier threshold: {threshold}")
        indexed[threshold] = row
    return indexed


def exact_threshold_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
        decode_scores,
        session_metrics,
    )

    member = str(payload["member"])
    kind = str(payload["kind"])
    horizon_ms = int(payload["horizon_ms"])
    thresholds = [float(t) for t in payload["thresholds"]]
    dev, kind_scores = _ensure_exact_member(member)
    scored = kind_scores[kind]
    primitives = []
    for threshold in thresholds:
        events = decode_scores(dev, scored, threshold=float(threshold), confirmation_ms=horizon_ms)
        metrics = session_metrics(dev, events)
        primitives.append(
            {
                "threshold": float(threshold),
                "false_cut_count": int(metrics["false_cut_count"]),
                "active_speech_seconds": float(metrics["active_speech_seconds"]),
                "reference_replacement_count": int(metrics["reference_replacement_count"]),
                "missed_replacement_count": int(metrics["missed_replacement_count"]),
                "exclusive_other_contamination_seconds": float(
                    metrics["exclusive_other_contamination_seconds"]
                ),
            }
        )
    return {
        "member": member,
        "kind": kind,
        "horizon_ms": horizon_ms,
        "thresholds": thresholds,
        "primitives": primitives,
    }


def sum_primitives(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    total_cuts = 0
    total_seconds = 0.0
    total_refs = 0
    total_missed = 0
    total_contamination = 0.0
    for row in rows:
        total_cuts += int(row["false_cut_count"])
        total_seconds += float(row["active_speech_seconds"])
        total_refs += int(row["reference_replacement_count"])
        total_missed += int(row["missed_replacement_count"])
        total_contamination += float(row["exclusive_other_contamination_seconds"])
    return {
        "false_cut_count": float(total_cuts),
        "active_speech_seconds": float(total_seconds),
        "reference_replacement_count": float(total_refs),
        "missed_replacement_count": float(total_missed),
        "exclusive_other_contamination_seconds": float(total_contamination),
    }


def plan_exact_tasks(
    grids: Mapping[str, Mapping[str, Sequence[float]]],
    horizons_ms: Sequence[int],
    chunk: int = EXACT_THRESHOLD_CHUNK,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for member in sorted(grids):
        kinds = grids[member]
        for kind in sorted(kinds):
            ordered = [float(t) for t in kinds[kind]]
            for horizon_ms in [int(h) for h in horizons_ms]:
                for i in range(0, len(ordered), int(chunk)):
                    tasks.append(
                        {
                            "member": str(member),
                            "kind": kind,
                            "horizon_ms": int(horizon_ms),
                            "thresholds": ordered[i : i + int(chunk)],
                        }
                    )
    return tasks


def run_exact_wave(
    members: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    workers: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import time as _time

    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    ordered_tasks = [dict(task) for task in tasks]
    resolved = arm_runtime.resolve_workers(workers)
    receipt = arm_runtime.worker_receipt(workers, len(ordered_tasks))
    start = _time.perf_counter()
    expected: dict[str, list[float]] = {}
    for task in ordered_tasks:
        key = "/".join([str(task["member"]), str(task["kind"]), str(int(task["horizon_ms"]))])
        expected.setdefault(key, []).extend([float(t) for t in task["thresholds"]])
    normalized = {str(key): value for key, value in dict(members).items()}
    rep_want: dict[str, list[float]] = {}
    rep_slot: dict[str, list[int]] = {}
    for key, want in expected.items():
        member, kind, _horizon = key.split("/")
        scores = dict(normalized[member]["scores"])[kind]
        representatives, positions = _representative_thresholds(scores, want)
        rep_want[key] = representatives
        rep_slot[key] = positions

    def _rep_key(item: str) -> tuple[str, str, int]:
        member, kind, horizon = item.split("/")
        return (member, kind, int(horizon))

    rep_tasks: list[dict[str, Any]] = []
    for key in sorted(rep_want, key=_rep_key):
        member, kind, horizon = key.split("/")
        representatives = rep_want[key]
        for i in range(0, len(representatives), int(EXACT_THRESHOLD_CHUNK)):
            rep_tasks.append(
                {
                    "member": member,
                    "kind": kind,
                    "horizon_ms": int(horizon),
                    "thresholds": representatives[i : i + int(EXACT_THRESHOLD_CHUNK)],
                }
            )
    rep_results: dict[str, Any] = {}
    counts: dict[str, Any] = {}

    def _ingest_rep(output: Mapping[str, Any]) -> None:
        member = str(output["member"])
        kind = str(output["kind"])
        horizon_ms = int(output["horizon_ms"])
        bucket = (
            rep_results.setdefault(member, {}).setdefault(kind, {}).setdefault(horizon_ms, [])
        )
        bucket.extend(list(output["primitives"]))

    if resolved <= 1 or len(ordered_tasks) <= 1:
        try:
            _EXACT_CONTEXT.clear()
            for member in sorted({str(task["member"]) for task in rep_tasks}):
                entry = normalized[member]
                _EXACT_CONTEXT["members"] = {
                    member: {
                        "dev": entry["dev"],
                        "scores": {
                            str(kind): _as_score_array(scores)
                            for kind, scores in dict(entry["scores"]).items()
                        },
                    }
                }
                _EXACT_CONTEXT["cached"] = None
                for task in [item for item in rep_tasks if str(item["member"]) == member]:
                    _ingest_rep(exact_threshold_task(task))
        finally:
            _EXACT_CONTEXT.clear()
        pool_count = 0
    else:
        import concurrent.futures
        import multiprocessing
        import os as _os
        import pickle as _pickle
        import tempfile as _tempfile

        by_member: dict[str, list[dict[str, Any]]] = {}
        for task in rep_tasks:
            by_member.setdefault(str(task["member"]), []).append(task)
        with _tempfile.TemporaryDirectory(prefix="cross-frontier-") as tmpdir:
            manifest: dict[str, dict[str, str]] = {}
            for index, member in enumerate(sorted(by_member)):
                entry = normalized[member]
                path = _os.path.join(tmpdir, f"member-{index}.pkl")
                with open(path, "wb") as handle:
                    _pickle.dump(
                        {
                            "dev": entry["dev"],
                            "scores": {
                                str(kind): scores
                                for kind, scores in dict(entry["scores"]).items()
                            },
                        },
                        handle,
                        protocol=_pickle.HIGHEST_PROTOCOL,
                    )
                manifest[member] = {"path": path}
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(resolved),
                mp_context=context,
                initializer=init_exact_context,
                initargs=(manifest,),
            ) as pool:
                for member in sorted(by_member):
                    for output in pool.map(exact_threshold_task, by_member[member]):
                        _ingest_rep(output)
        pool_count = 1
    rep_indexed: dict[str, dict[float, Mapping[str, Any]]] = {}
    for key, representatives in rep_want.items():
        member, kind, horizon = key.split("/")
        rows = rep_results.get(member, {}).get(kind, {}).get(int(horizon), [])
        got = [float(row["threshold"]) for row in rows]
        if got != [float(t) for t in representatives]:
            raise CrossFrontierError(f"frontier thresholds incomplete for {key}")
        rep_indexed[key] = {float(row["threshold"]): row for row in rows}
    results: dict[str, Any] = {}
    for key, want in expected.items():
        member, kind, horizon = key.split("/")
        representatives = rep_want[key]
        positions = rep_slot[key]
        lookup = rep_indexed[key]
        bucket = (
            results.setdefault(member, {}).setdefault(kind, {}).setdefault(int(horizon), [])
        )
        for position, threshold in enumerate(want):
            row = dict(lookup[float(representatives[positions[position]])])
            row["threshold"] = float(threshold)
            bucket.append(row)
    for key, want in expected.items():
        member, kind, horizon = key.split("/")
        got_rows = results.get(member, {}).get(kind, {}).get(int(horizon), [])
        got = [float(row["threshold"]) for row in got_rows]
        if got != [float(t) for t in want]:
            raise CrossFrontierError(f"frontier thresholds incomplete for {key}")
        counts[key] = {"expected": len(want), "observed": len(got)}
    scored_total = sum(len(item) for item in rep_want.values())
    full_total = sum(len(item) for item in expected.values())
    receipt = {
        **receipt,
        "total_tasks": len(ordered_tasks),
        "pool_count": pool_count,
        "exact": True,
        "score_tasks": len(rep_tasks),
        "reused_primitives": int(full_total - scored_total),
        "primitive_counts": counts,
        "elapsed_seconds": _time.perf_counter() - start,
    }
    return results, receipt
