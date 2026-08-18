from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

from experiments.psem_training_strategy_gate.data.identity_components import (
    IdentityGraphError,
    validate_split_assignment,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    HISTORICAL_CONFIGS,
    canonical_sha256,
    sha256_file,
)
from experiments.psem_training_strategy_gate.data.split_feasibility import (
    AUTHORITY_PIN,
    AUTHORITY_REF,
    NEGATIVE_EXPOSURE_REQUIREMENTS,
    NO_MODEL_FIELDS,
    ROLE_REQUIREMENTS,
    SAMPLE_RATE_HZ,
    TOPOLOGY_REQUIREMENTS,
    _eval_eligible_component_ids,
    _load_json,
    _load_jsonl,
    build_split_feasibility,
)
from experiments.psem_training_strategy_gate.data.topology_census import (
    render_data_census,
)

TRAIN_ROLE = "PSEM-STRATEGY-TRAIN"
DEV_ROLE = "PSEM-STRATEGY-DEV"
EVAL_ROLE = "PSEM-STRATEGY-EVAL"
OFFICIAL_ROLES = (TRAIN_ROLE, DEV_ROLE, EVAL_ROLE)
SELECTION_ORDER = (EVAL_ROLE, DEV_ROLE, TRAIN_ROLE)
SEARCH_ALGORITHM = "integer-bound-eval-frontier-seeded-dev-prefix"
SEARCH_ALGORITHM_VERSION = "1"
SEARCH_SEED = 770076
EVAL_FRONTIER_PER_VIEW = 64
DEV_PERMUTATIONS_PER_EVAL = 24
MAX_EVAL_SUBSETS = 2_000_000
PLANNING_TARGET_HOURS = {TRAIN_ROLE: 24, DEV_ROLE: 6, EVAL_ROLE: 10}


class SplitAssignmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentStats:
    component_id: str
    source_ids: tuple[str, ...]
    eval_eligible: bool
    scored_samples: int
    stable_singleton_samples: int
    ongoing_overlap_samples: int
    topology_counts: tuple[tuple[str, int], ...]
    speaker_ids: tuple[str, ...]
    corpora: tuple[str, ...]
    acoustic_groups: tuple[str, ...]
    masked_transitions: int
    actual_transitions: int
    ambiguous_samples: int
    source_scored_samples: tuple[tuple[str, int], ...]
    speaker_association_samples: tuple[tuple[str, int], ...]
    corpus_scored_samples: tuple[tuple[str, int], ...]

    def topology(self) -> dict[str, int]:
        return dict(self.topology_counts)


@dataclass(frozen=True)
class Aggregate:
    component_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    scored_samples: int
    stable_singleton_samples: int
    ongoing_overlap_samples: int
    topology_counts: tuple[tuple[str, int], ...]
    speaker_ids: tuple[str, ...]
    corpora: tuple[str, ...]
    acoustic_groups: tuple[str, ...]
    masked_transitions: int
    actual_transitions: int
    ambiguous_samples: int
    source_scored_samples: tuple[tuple[str, int], ...]
    component_scored_samples: tuple[tuple[str, int], ...]
    speaker_association_samples: tuple[tuple[str, int], ...]
    corpus_scored_samples: tuple[tuple[str, int], ...]

    def topology(self) -> dict[str, int]:
        return dict(self.topology_counts)


def _qualified_speaker(corpus: str, speaker_id: str) -> str:
    return f"{corpus}:{speaker_id}"


def _require_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SplitAssignmentError(f"invalid nonnegative integer: {label}")
    return value


def _build_components(
    data_dir: Path,
    graph: dict[str, Any],
    overlap_rows: list[dict[str, Any]],
) -> list[ComponentStats]:
    source_rows = _load_jsonl(data_dir / "source_manifest.jsonl")
    topology_rows = _load_jsonl(data_dir / "topology_manifest.jsonl")
    source_by_id = {row.get("source_id"): row for row in source_rows}
    topology_by_id = {row.get("source_id"): row for row in topology_rows}
    if (
        len(source_by_id) != len(source_rows)
        or len(topology_by_id) != len(topology_rows)
        or set(source_by_id) != set(topology_by_id)
    ):
        raise SplitAssignmentError("split input source coverage is invalid")
    eligible_ids = _eval_eligible_component_ids(graph, overlap_rows)
    components = graph.get("components")
    if not isinstance(components, list):
        raise SplitAssignmentError("identity component inventory is invalid")
    result: list[ComponentStats] = []
    for component in components:
        component_id = component.get("component_id")
        source_ids = component.get("source_ids")
        if not isinstance(component_id, str) or not isinstance(source_ids, list) or not source_ids:
            raise SplitAssignmentError("identity component row is invalid")
        if any(not isinstance(source_id, str) or source_id not in source_by_id for source_id in source_ids):
            raise SplitAssignmentError("identity component source binding is invalid")
        sources = [source_by_id[source_id] for source_id in source_ids]
        topologies = [topology_by_id[source_id] for source_id in source_ids]
        if any(
            row.get("split_role") != "UNASSIGNED_CANDIDATE"
            or row.get("component_id") is not None
            for row in topologies
        ):
            raise SplitAssignmentError("candidate topology rows are not in the pre-split state")
        topology_counts = {
            topology: sum(
                _require_int(row.get("primary_topology_counts", {}).get(topology), topology)
                for row in topologies
            )
            for topology in TOPOLOGY_REQUIREMENTS
        }
        speakers = sorted(
            {
                _qualified_speaker(str(row["corpus"]), speaker_id)
                for row in sources
                for speaker_id in row.get("speaker_ids", [])
                if isinstance(speaker_id, str) and speaker_id
            }
        )
        if any(
            not isinstance(row.get("speaker_ids"), list)
            or not row["speaker_ids"]
            or len(row["speaker_ids"])
            != len({speaker_id for speaker_id in row["speaker_ids"] if isinstance(speaker_id, str)})
            for row in sources
        ):
            raise SplitAssignmentError("known speaker inventory is invalid")
        source_samples = {
            str(row["source_id"]): _require_int(topology["scored_samples"], "scored_samples")
            for row, topology in zip(sources, topologies, strict=True)
        }
        speaker_samples: dict[str, int] = {}
        corpus_samples: dict[str, int] = {}
        for source, topology in zip(sources, topologies, strict=True):
            samples = _require_int(topology.get("scored_samples"), "scored_samples")
            corpus = str(source["corpus"])
            corpus_samples[corpus] = corpus_samples.get(corpus, 0) + samples
            for speaker_id in source["speaker_ids"]:
                key = _qualified_speaker(corpus, speaker_id)
                speaker_samples[key] = speaker_samples.get(key, 0) + samples
        result.append(
            ComponentStats(
                component_id=component_id,
                source_ids=tuple(sorted(source_ids)),
                eval_eligible=component_id in eligible_ids,
                scored_samples=sum(source_samples.values()),
                stable_singleton_samples=sum(
                    _require_int(row.get("stable_singleton_samples"), "stable_singleton_samples")
                    for row in topologies
                ),
                ongoing_overlap_samples=sum(
                    _require_int(row.get("ongoing_overlap_samples"), "ongoing_overlap_samples")
                    for row in topologies
                ),
                topology_counts=tuple(sorted(topology_counts.items())),
                speaker_ids=tuple(speakers),
                corpora=tuple(sorted({str(row["corpus"]) for row in sources})),
                acoustic_groups=tuple(
                    sorted(
                        {
                            f"{row['corpus']}:{row['meeting_type']}:{row['recording_recipe']}"
                            for row in sources
                        }
                    )
                ),
                masked_transitions=sum(
                    _require_int(
                        row.get("mask_diagnostics", {}).get("masked_transition_count"),
                        "masked_transition_count",
                    )
                    for row in topologies
                ),
                actual_transitions=sum(
                    _require_int(
                        row.get("mask_diagnostics", {}).get("actual_transition_count"),
                        "actual_transition_count",
                    )
                    for row in topologies
                ),
                ambiguous_samples=sum(
                    _require_int(row.get("ambiguous_samples"), "ambiguous_samples")
                    for row in topologies
                ),
                source_scored_samples=tuple(sorted(source_samples.items())),
                speaker_association_samples=tuple(sorted(speaker_samples.items())),
                corpus_scored_samples=tuple(sorted(corpus_samples.items())),
            )
        )
    ordered = sorted(result, key=lambda component: component.component_id)
    covered = [source_id for component in ordered for source_id in component.source_ids]
    if len(covered) != len(set(covered)) or set(covered) != set(source_by_id):
        raise SplitAssignmentError("component inventory does not exactly cover split sources")
    return ordered


def _sum_pairs(components: Iterable[ComponentStats], field: str) -> tuple[tuple[str, int], ...]:
    values: dict[str, int] = {}
    for component in components:
        for key, value in getattr(component, field):
            values[key] = values.get(key, 0) + value
    return tuple(sorted(values.items()))


def _aggregate(components: list[ComponentStats], indices: Iterable[int]) -> Aggregate:
    selected = [components[index] for index in sorted(indices)]
    topology_counts = {
        topology: sum(component.topology()[topology] for component in selected)
        for topology in TOPOLOGY_REQUIREMENTS
    }
    return Aggregate(
        component_ids=tuple(component.component_id for component in selected),
        source_ids=tuple(sorted(source_id for component in selected for source_id in component.source_ids)),
        scored_samples=sum(component.scored_samples for component in selected),
        stable_singleton_samples=sum(component.stable_singleton_samples for component in selected),
        ongoing_overlap_samples=sum(component.ongoing_overlap_samples for component in selected),
        topology_counts=tuple(sorted(topology_counts.items())),
        speaker_ids=tuple(sorted({speaker for component in selected for speaker in component.speaker_ids})),
        corpora=tuple(sorted({corpus for component in selected for corpus in component.corpora})),
        acoustic_groups=tuple(
            sorted({group for component in selected for group in component.acoustic_groups})
        ),
        masked_transitions=sum(component.masked_transitions for component in selected),
        actual_transitions=sum(component.actual_transitions for component in selected),
        ambiguous_samples=sum(component.ambiguous_samples for component in selected),
        source_scored_samples=_sum_pairs(selected, "source_scored_samples"),
        component_scored_samples=tuple(
            sorted((component.component_id, component.scored_samples) for component in selected)
        ),
        speaker_association_samples=_sum_pairs(selected, "speaker_association_samples"),
        corpus_scored_samples=_sum_pairs(selected, "corpus_scored_samples"),
    )


def _role_minimum_passes(role: str, aggregate: Aggregate) -> bool:
    requirement = ROLE_REQUIREMENTS[role]
    return (
        aggregate.scored_samples
        >= requirement["scored_hours"] * 3600 * SAMPLE_RATE_HZ
        and len(aggregate.source_ids) >= requirement["independent_meetings"]
    )


def _eval_and_complement_pass(eval_aggregate: Aggregate, complement: Aggregate) -> bool:
    if not _role_minimum_passes(EVAL_ROLE, eval_aggregate):
        return False
    if (
        eval_aggregate.stable_singleton_samples
        < NEGATIVE_EXPOSURE_REQUIREMENTS["stable_singleton_samples"]["eval"]
        or eval_aggregate.ongoing_overlap_samples
        < NEGATIVE_EXPOSURE_REQUIREMENTS["ongoing_overlap_samples"]["eval"]
        or complement.stable_singleton_samples
        < NEGATIVE_EXPOSURE_REQUIREMENTS["stable_singleton_samples"]["train_dev"]
        or complement.ongoing_overlap_samples
        < NEGATIVE_EXPOSURE_REQUIREMENTS["ongoing_overlap_samples"]["train_dev"]
    ):
        return False
    eval_topology = eval_aggregate.topology()
    complement_topology = complement.topology()
    return all(
        eval_topology[topology] >= requirement["eval"]
        and complement_topology[topology] >= requirement["train_dev"]
        for topology, requirement in TOPOLOGY_REQUIREMENTS.items()
    )


def _topology_slack(eval_aggregate: Aggregate, complement: Aggregate) -> Fraction:
    eval_topology = eval_aggregate.topology()
    complement_topology = complement.topology()
    values = []
    for topology, requirement in TOPOLOGY_REQUIREMENTS.items():
        values.append(Fraction(eval_topology[topology] - requirement["eval"], requirement["eval"]))
        values.append(
            Fraction(
                complement_topology[topology] - requirement["train_dev"],
                requirement["train_dev"],
            )
        )
    return min(values)


def _integer_topology_upper_bounds(total: Aggregate) -> dict[str, tuple[Fraction, tuple[int, ...]]]:
    result: dict[str, tuple[Fraction, tuple[int, ...]]] = {}
    totals = total.topology()
    for topology, requirement in TOPOLOGY_REQUIREMENTS.items():
        candidates = []
        for eval_count in range(
            requirement["eval"], totals[topology] - requirement["train_dev"] + 1
        ):
            slack = min(
                Fraction(eval_count - requirement["eval"], requirement["eval"]),
                Fraction(
                    totals[topology] - eval_count - requirement["train_dev"],
                    requirement["train_dev"],
                ),
            )
            candidates.append((slack, eval_count))
        if not candidates:
            raise SplitAssignmentError(f"aggregate topology lower bound failed: {topology}")
        best = max(slack for slack, _ in candidates)
        result[topology] = (best, tuple(count for slack, count in candidates if slack == best))
    return result


def _reachable_suffix(values: list[int], maximum: int) -> list[set[int]]:
    suffix = [set() for _ in range(len(values) + 1)]
    suffix[-1].add(0)
    for index in range(len(values) - 1, -1, -1):
        value = values[index]
        suffix[index] = suffix[index + 1] | {
            candidate + value
            for candidate in suffix[index + 1]
            if candidate + value <= maximum
        }
    return suffix


def _mask_indices(mask: int, count: int) -> tuple[int, ...]:
    return tuple(index for index in range(count) if mask & (1 << index))


def _ratio(value: int, total: int) -> Fraction:
    return Fraction(value, total) if total else Fraction(0, 1)


def _eval_view_ranks(aggregate: Aggregate) -> tuple[tuple[Any, ...], ...]:
    target = PLANNING_TARGET_HOURS[EVAL_ROLE] * 3600 * SAMPLE_RATE_HZ
    distance = abs(aggregate.scored_samples - target)
    quality = -_ratio(aggregate.masked_transitions, aggregate.actual_transitions)
    dominance = -max(
        (_ratio(value, aggregate.scored_samples) for _, value in aggregate.component_scored_samples),
        default=Fraction(0, 1),
    )
    common = (
        len(aggregate.speaker_ids),
        len(aggregate.source_ids),
        len(aggregate.corpora),
        len(aggregate.acoustic_groups),
        quality,
        dominance,
        -distance,
    )
    return (
        (-distance, *common),
        (len(aggregate.speaker_ids), -distance, *common[1:]),
        (len(aggregate.source_ids), -distance, *common),
        (len(aggregate.corpora), len(aggregate.acoustic_groups), -distance, *common),
        (quality, dominance, -distance, *common),
    )


def _push_frontier(
    heaps: list[list[tuple[tuple[Any, ...], int]]],
    ranks: tuple[tuple[Any, ...], ...],
    mask: int,
) -> None:
    for heap, rank in zip(heaps, ranks, strict=True):
        item = (rank, mask)
        if len(heap) < EVAL_FRONTIER_PER_VIEW:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)


def _enumerate_eval_frontier(
    components: list[ComponentStats],
) -> tuple[set[int], dict[str, Any]]:
    all_indices = set(range(len(components)))
    eligible_global = [index for index, component in enumerate(components) if component.eval_eligible]
    eligible = [components[index] for index in eligible_global]
    total = _aggregate(components, all_indices)
    upper_bounds = _integer_topology_upper_bounds(total)
    global_upper_bound = min(bound for bound, _ in upper_bounds.values())
    limiting = sorted(
        topology for topology, (bound, _) in upper_bounds.items() if bound == global_upper_bound
    )
    limiting_topology = limiting[0]
    target_counts = upper_bounds[limiting_topology][1]
    values = [component.topology()[limiting_topology] for component in eligible]
    suffix = _reachable_suffix(values, max(target_counts))
    heaps: list[list[tuple[tuple[Any, ...], int]]] = [[] for _ in _eval_view_ranks(_aggregate([], []))]
    evaluated = 0
    feasible = 0
    achieved_upper_bound = False
    enumeration_truncated = False

    def visit(position: int, remaining: int, mask: int) -> None:
        nonlocal evaluated, feasible, achieved_upper_bound, enumeration_truncated
        if evaluated >= MAX_EVAL_SUBSETS:
            enumeration_truncated = True
            return
        if remaining not in suffix[position]:
            return
        if position == len(eligible):
            evaluated += 1
            global_mask = 0
            for local_index in _mask_indices(mask, len(eligible)):
                global_mask |= 1 << eligible_global[local_index]
            eval_indices = set(_mask_indices(global_mask, len(components)))
            eval_aggregate = _aggregate(components, eval_indices)
            complement = _aggregate(components, all_indices - eval_indices)
            if not _eval_and_complement_pass(eval_aggregate, complement):
                return
            if (
                complement.scored_samples
                < (ROLE_REQUIREMENTS[TRAIN_ROLE]["scored_hours"] + ROLE_REQUIREMENTS[DEV_ROLE]["scored_hours"])
                * 3600
                * SAMPLE_RATE_HZ
                or len(complement.source_ids)
                < ROLE_REQUIREMENTS[TRAIN_ROLE]["independent_meetings"]
                + ROLE_REQUIREMENTS[DEV_ROLE]["independent_meetings"]
            ):
                return
            slack = _topology_slack(eval_aggregate, complement)
            if slack != global_upper_bound:
                return
            feasible += 1
            achieved_upper_bound = True
            _push_frontier(heaps, _eval_view_ranks(eval_aggregate), global_mask)
            return
        visit(position + 1, remaining, mask)
        value = values[position]
        if value <= remaining:
            visit(position + 1, remaining - value, mask | (1 << position))

    for target in target_counts:
        visit(0, target, 0)
    if enumeration_truncated:
        raise SplitAssignmentError(
            "bounded EVAL search exhausted before completing the optimal frontier"
        )
    if not achieved_upper_bound:
        raise SplitAssignmentError("no EVAL assignment attains the integer topology slack bound")
    frontier = {mask for heap in heaps for _, mask in heap}
    return frontier, {
        "integer_topology_upper_bounds": {
            topology: {
                "numerator": bound.numerator,
                "denominator": bound.denominator,
                "decimal": round(float(bound), 8),
                "optimal_eval_counts": list(counts),
            }
            for topology, (bound, counts) in sorted(upper_bounds.items())
        },
        "global_upper_bound": {
            "numerator": global_upper_bound.numerator,
            "denominator": global_upper_bound.denominator,
            "decimal": round(float(global_upper_bound), 8),
        },
        "limiting_topology": limiting_topology,
        "limiting_topologies": limiting,
        "target_eval_counts": list(target_counts),
        "evaluated_exact_count_subsets": evaluated,
        "upper_bound_feasible_subsets": feasible,
        "frontier_assignment_count": len(frontier),
        "max_eval_subsets": MAX_EVAL_SUBSETS,
        "eval_enumeration_complete": True,
        "upper_bound_achieved": achieved_upper_bound,
    }


def _stable_order(indices: Iterable[int], components: list[ComponentStats], token: str) -> list[int]:
    return sorted(
        indices,
        key=lambda index: hashlib.sha256(
            f"{SEARCH_SEED}|{token}|{components[index].component_id}".encode("utf-8")
        ).digest(),
    )


def _assignment_score(
    summaries: dict[str, Aggregate],
    topology_slack: Fraction,
) -> tuple[Any, ...]:
    planning_error = sum(
        abs(
            summaries[role].scored_samples
            - PLANNING_TARGET_HOURS[role] * 3600 * SAMPLE_RATE_HZ
        )
        for role in OFFICIAL_ROLES
    )
    speaker_diversity = min(len(summaries[role].speaker_ids) for role in OFFICIAL_ROLES)
    meeting_diversity = min(len(summaries[role].source_ids) for role in OFFICIAL_ROLES)
    component_diversity = min(len(summaries[role].component_ids) for role in OFFICIAL_ROLES)
    minimum_corpus_diversity = min(
        len(summaries[role].corpora) for role in OFFICIAL_ROLES
    )
    minimum_acoustic_diversity = min(
        len(summaries[role].acoustic_groups) for role in OFFICIAL_ROLES
    )
    corpus_diversity = sum(len(summaries[role].corpora) for role in OFFICIAL_ROLES)
    acoustic_diversity = sum(len(summaries[role].acoustic_groups) for role in OFFICIAL_ROLES)
    worst_masked_fraction = max(
        _ratio(summaries[role].masked_transitions, summaries[role].actual_transitions)
        for role in OFFICIAL_ROLES
    )
    worst_ambiguous_fraction = max(
        _ratio(summaries[role].ambiguous_samples, summaries[role].scored_samples)
        for role in OFFICIAL_ROLES
    )
    worst_speaker_association = max(
        (
            _ratio(value, summaries[role].scored_samples)
            for role in OFFICIAL_ROLES
            for _, value in summaries[role].speaker_association_samples
        ),
        default=Fraction(0, 1),
    )
    worst_source_dominance = max(
        _ratio(value, summaries[role].scored_samples)
        for role in OFFICIAL_ROLES
        for _, value in summaries[role].source_scored_samples
    )
    worst_component_dominance = max(
        _ratio(value, summaries[role].scored_samples)
        for role in OFFICIAL_ROLES
        for _, value in summaries[role].component_scored_samples
    )
    worst_corpus_dominance = max(
        _ratio(value, summaries[role].scored_samples)
        for role in OFFICIAL_ROLES
        for _, value in summaries[role].corpus_scored_samples
    )
    return (
        topology_slack,
        minimum_corpus_diversity,
        minimum_acoustic_diversity,
        -planning_error,
        -worst_corpus_dominance,
        speaker_diversity,
        meeting_diversity,
        component_diversity,
        corpus_diversity,
        acoustic_diversity,
        -worst_masked_fraction,
        -worst_ambiguous_fraction,
        -worst_speaker_association,
        -worst_source_dominance,
        -worst_component_dominance,
    )


def _search_assignment(
    components: list[ComponentStats],
) -> tuple[dict[str, set[int]], dict[str, Any], tuple[Any, ...]]:
    frontier, eval_search = _enumerate_eval_frontier(components)
    all_indices = set(range(len(components)))
    best: tuple[tuple[Any, ...], str, dict[str, set[int]], dict[str, Aggregate]] | None = None
    complete_candidates = 0
    for frontier_number, eval_mask in enumerate(sorted(frontier)):
        eval_indices = set(_mask_indices(eval_mask, len(components)))
        remaining = all_indices - eval_indices
        eval_aggregate = _aggregate(components, eval_indices)
        complement = _aggregate(components, remaining)
        topology_slack = _topology_slack(eval_aggregate, complement)
        orders = [
            sorted(remaining, key=lambda index: components[index].scored_samples),
            sorted(remaining, key=lambda index: -components[index].scored_samples),
            sorted(remaining, key=lambda index: components[index].component_id),
        ]
        orders.extend(
            _stable_order(remaining, components, f"dev|{frontier_number}|{iteration}")
            for iteration in range(DEV_PERMUTATIONS_PER_EVAL)
        )
        for order in orders:
            dev_indices: set[int] = set()
            for index in order[:-1]:
                dev_indices.add(index)
                train_indices = remaining - dev_indices
                dev_aggregate = _aggregate(components, dev_indices)
                train_aggregate = _aggregate(components, train_indices)
                if not _role_minimum_passes(DEV_ROLE, dev_aggregate):
                    continue
                if not _role_minimum_passes(TRAIN_ROLE, train_aggregate):
                    break
                complete_candidates += 1
                summaries = {
                    TRAIN_ROLE: train_aggregate,
                    DEV_ROLE: dev_aggregate,
                    EVAL_ROLE: eval_aggregate,
                }
                score = _assignment_score(summaries, topology_slack)
                assignments = {
                    TRAIN_ROLE: train_indices,
                    DEV_ROLE: set(dev_indices),
                    EVAL_ROLE: eval_indices,
                }
                tie = canonical_sha256(
                    {
                        role: [components[i].component_id for i in sorted(indices)]
                        for role, indices in assignments.items()
                    }
                )
                candidate = (score, tie, assignments, summaries)
                if best is None or score > best[0] or (score == best[0] and tie < best[1]):
                    best = candidate
    if best is None:
        raise SplitAssignmentError("bounded DEV search found no valid complete assignment")
    score, tie, assignments, _ = best
    return assignments, {
        **eval_search,
        "dev_permutations_per_eval": DEV_PERMUTATIONS_PER_EVAL,
        "dev_search_exhaustive": False,
        "secondary_search_bounded": True,
        "complete_valid_candidates_evaluated": complete_candidates,
        "chosen_assignment_sha256": tie,
    }, score


def _fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": round(float(value), 8),
    }


def _role_summary(role: str, aggregate: Aggregate) -> dict[str, Any]:
    topology = aggregate.topology()
    return {
        "role": role,
        "component_count": len(aggregate.component_ids),
        "source_count": len(aggregate.source_ids),
        "independent_meetings": len(aggregate.source_ids),
        "scored_samples": aggregate.scored_samples,
        "scored_hours": round(aggregate.scored_samples / SAMPLE_RATE_HZ / 3600, 6),
        "stable_singleton_samples": aggregate.stable_singleton_samples,
        "stable_singleton_hours": round(
            aggregate.stable_singleton_samples / SAMPLE_RATE_HZ / 3600, 6
        ),
        "ongoing_overlap_samples": aggregate.ongoing_overlap_samples,
        "ongoing_overlap_hours": round(
            aggregate.ongoing_overlap_samples / SAMPLE_RATE_HZ / 3600, 6
        ),
        "primary_topology_counts": topology,
        "known_speaker_count": len(aggregate.speaker_ids),
        "corpora": list(aggregate.corpora),
        "acoustic_groups": list(aggregate.acoustic_groups),
        "masked_transition_fraction": round(
            float(_ratio(aggregate.masked_transitions, aggregate.actual_transitions)), 8
        ),
        "ambiguous_sample_fraction": round(
            float(_ratio(aggregate.ambiguous_samples, aggregate.scored_samples)), 8
        ),
        "maximum_known_speaker_source_association_share": round(
            float(
                max(
                    (
                        _ratio(value, aggregate.scored_samples)
                        for _, value in aggregate.speaker_association_samples
                    ),
                    default=Fraction(0, 1),
                )
            ),
            8,
        ),
        "maximum_source_scored_share": round(
            float(
                max(
                    (_ratio(value, aggregate.scored_samples) for _, value in aggregate.source_scored_samples),
                    default=Fraction(0, 1),
                )
            ),
            8,
        ),
        "maximum_component_scored_share": round(
            float(
                max(
                    (
                        _ratio(value, aggregate.scored_samples)
                        for _, value in aggregate.component_scored_samples
                    ),
                    default=Fraction(0, 1),
                )
            ),
            8,
        ),
        "maximum_corpus_scored_share": round(
            float(
                max(
                    (_ratio(value, aggregate.scored_samples) for _, value in aggregate.corpus_scored_samples),
                    default=Fraction(0, 1),
                )
            ),
            8,
        ),
    }


def _hard_gate_results(summaries: dict[str, Aggregate]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for role in OFFICIAL_ROLES:
        aggregate = summaries[role]
        required_samples = ROLE_REQUIREMENTS[role]["scored_hours"] * 3600 * SAMPLE_RATE_HZ
        results.extend(
            [
                {
                    "id": f"{role}.scored_samples",
                    "observed": aggregate.scored_samples,
                    "required": required_samples,
                    "passed": aggregate.scored_samples >= required_samples,
                },
                {
                    "id": f"{role}.independent_meetings",
                    "observed": len(aggregate.source_ids),
                    "required": ROLE_REQUIREMENTS[role]["independent_meetings"],
                    "passed": len(aggregate.source_ids)
                    >= ROLE_REQUIREMENTS[role]["independent_meetings"],
                },
            ]
        )
    train_dev = Aggregate(
        component_ids=tuple(
            sorted(summaries[TRAIN_ROLE].component_ids + summaries[DEV_ROLE].component_ids)
        ),
        source_ids=tuple(sorted(summaries[TRAIN_ROLE].source_ids + summaries[DEV_ROLE].source_ids)),
        scored_samples=summaries[TRAIN_ROLE].scored_samples + summaries[DEV_ROLE].scored_samples,
        stable_singleton_samples=summaries[TRAIN_ROLE].stable_singleton_samples
        + summaries[DEV_ROLE].stable_singleton_samples,
        ongoing_overlap_samples=summaries[TRAIN_ROLE].ongoing_overlap_samples
        + summaries[DEV_ROLE].ongoing_overlap_samples,
        topology_counts=tuple(
            sorted(
                (
                    topology,
                    summaries[TRAIN_ROLE].topology()[topology]
                    + summaries[DEV_ROLE].topology()[topology],
                )
                for topology in TOPOLOGY_REQUIREMENTS
            )
        ),
        speaker_ids=(),
        corpora=(),
        acoustic_groups=(),
        masked_transitions=0,
        actual_transitions=0,
        ambiguous_samples=0,
        source_scored_samples=(),
        component_scored_samples=(),
        speaker_association_samples=(),
        corpus_scored_samples=(),
    )
    for topology, requirement in TOPOLOGY_REQUIREMENTS.items():
        for label, aggregate, key in (
            ("train_dev", train_dev, "train_dev"),
            ("eval", summaries[EVAL_ROLE], "eval"),
        ):
            observed = aggregate.topology()[topology]
            required = requirement[key]
            results.append(
                {
                    "id": f"{label}.primary_topology_counts.{topology}",
                    "observed": observed,
                    "required": required,
                    "passed": observed >= required,
                }
            )
    for exposure, requirement in NEGATIVE_EXPOSURE_REQUIREMENTS.items():
        field = exposure
        for label, aggregate, key in (
            ("train_dev", train_dev, "train_dev"),
            ("eval", summaries[EVAL_ROLE], "eval"),
        ):
            observed = getattr(aggregate, field)
            required = requirement[key]
            results.append(
                {
                    "id": f"{label}.{exposure}",
                    "observed": observed,
                    "required": required,
                    "passed": observed >= required,
                }
            )
    return results


def _build_split_manifest_uncached(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
    input_fingerprint: str,
) -> dict[str, Any]:
    feasibility = build_split_feasibility(data_dir, registry_path, source_registry_path)
    if feasibility["blocking_lower_bounds"]:
        raise SplitAssignmentError("aggregate lower bounds block component assignment search")
    graph = _load_json(data_dir / "identity_components.json")
    overlap = _load_json(data_dir / "wavlm_pretraining_overlap.json")
    overlap_rows = overlap.get("sources")
    if not isinstance(overlap_rows, list):
        raise SplitAssignmentError("pretraining overlap source inventory is invalid")
    components = _build_components(data_dir, graph, overlap_rows)
    assignments, search, score = _search_assignment(components)
    summaries = {
        role: _aggregate(components, indices) for role, indices in assignments.items()
    }
    source_assignments = {
        source_id: role
        for role, indices in assignments.items()
        for index in indices
        for source_id in components[index].source_ids
    }
    try:
        validate_split_assignment(graph, source_assignments, data_dir)
    except IdentityGraphError as exc:
        raise SplitAssignmentError("chosen assignment violates the identity graph") from exc
    overlap_by_source = {row.get("source_id"): row for row in overlap_rows}
    if any(
        source_assignments.get(source_id) == EVAL_ROLE
        and overlap_by_source[source_id].get("eval_forbidden_by_pretraining_overlap") is not False
        for source_id in source_assignments
    ):
        raise SplitAssignmentError("chosen EVAL contains forbidden pretraining overlap")
    gates = _hard_gate_results(summaries)
    if not gates or any(result["passed"] is not True for result in gates):
        raise SplitAssignmentError("chosen assignment does not pass every role-specific hard gate")
    topology_slack = score[0]
    component_rows = [
        {
            "component_id": component.component_id,
            "role": role,
            "source_ids": list(component.source_ids),
            "eval_eligible": component.eval_eligible,
        }
        for role in OFFICIAL_ROLES
        for index in sorted(assignments[role], key=lambda item: components[item].component_id)
        for component in (components[index],)
    ]
    source_rows = _load_jsonl(data_dir / "source_manifest.jsonl")
    source_by_id = {row["source_id"]: row for row in source_rows}
    source_assignment_rows = [
        {
            "source_id": source_id,
            "session_id": source_by_id[source_id]["session_id"],
            "corpus": source_by_id[source_id]["corpus"],
            "component_id": next(
                component.component_id for component in components if source_id in component.source_ids
            ),
            "role": source_assignments[source_id],
            "waveform_sha256": source_by_id[source_id]["waveform_sha256"],
            "annotation_sha256": source_by_id[source_id]["annotation_sha256"],
        }
        for source_id in sorted(source_assignments)
    ]
    assignment_payload = {
        "components": component_rows,
        "sources": source_assignment_rows,
    }
    return {
        "schema_version": 1,
        "artifact_role": "psem_component_split_assignment",
        "authority_ref": AUTHORITY_REF,
        "authority_pin": AUTHORITY_PIN,
        "contract_version": _load_json(data_dir / "topology_census.json")["contract_version"],
        "natural_data_only": True,
        "official_roles": list(OFFICIAL_ROLES),
        "selection_order": list(SELECTION_ORDER),
        "input_artifacts": {
            **feasibility["input_artifacts"],
            "annotation_manifest_sha256": sha256_file(data_dir / "annotation_manifest.jsonl"),
            "normalization_manifest_sha256": sha256_file(
                data_dir / "normalization_manifest.jsonl"
            ),
            "topology_manifest_sha256": sha256_file(data_dir / "topology_manifest.jsonl"),
            "prior_exposure_manifest_sha256": sha256_file(
                data_dir / "prior_exposure_manifest.jsonl"
            ),
            "split_lower_bound_decision_sha256": feasibility["decision_basis_sha256"],
        },
        "search": {
            "algorithm": SEARCH_ALGORITHM,
            "algorithm_version": SEARCH_ALGORITHM_VERSION,
            "seed": SEARCH_SEED,
            "input_fingerprint_sha256": input_fingerprint,
            "selection_order": list(SELECTION_ORDER),
            "eval_frontier_per_view": EVAL_FRONTIER_PER_VIEW,
            "objective_order": [
                "maximize minimum normalized topology slack",
                "maximize minimum per-role corpus and acoustic-group diversity",
                "minimize distance from 24/6/10 hour operational buffer targets",
                "minimize worst per-role corpus dominance",
                "maximize minimum known-speaker diversity",
                "maximize minimum independent-meeting diversity",
                "maximize minimum connected-component diversity",
                "maximize total corpus and acoustic-group diversity",
                "minimize masked and ambiguous fractions",
                "minimize known-speaker association, source, and component dominance",
            ],
            "model_derived_quantities_allowed": False,
            **search,
        },
        "assignment_sha256": canonical_sha256(assignment_payload),
        "assignments": assignment_payload,
        "role_summaries": {
            role: _role_summary(role, summaries[role]) for role in OFFICIAL_ROLES
        },
        "hard_gate_results": gates,
        "hard_gate_status": "pass",
        "leakage_audit": {
            "exact_source_coverage": len(source_assignments) == len(source_rows),
            "component_may_span_roles": False,
            "meeting_session_may_span_roles": False,
            "waveform_may_span_roles": False,
            "known_speaker_may_span_roles": False,
            "prior_selection_exposed_component_in_eval": False,
            "exact_known_wavlm_pretraining_overlap_in_eval": False,
        },
        "objective_result": {
            "minimum_normalized_topology_slack": _fraction_record(topology_slack),
            "integer_global_upper_bound_achieved": search["upper_bound_achieved"],
            "planning_target_hours": PLANNING_TARGET_HOURS,
        },
        "summary_rationale": (
            "EVAL was selected only from freshness-eligible connected components and attains the "
            "integer global upper bound on minimum normalized topology slack. DEV was then selected "
            "by reproducibly seeded component-prefix search, and every remaining component was "
            "assigned to TRAIN. The final choice uses annotation-only diversity, quality, dominance, "
            "and operational-hour criteria after all hard gates; no model-derived quantity participates."
        ),
        "model_policy": {field: False for field in NO_MODEL_FIELDS},
    }


def _input_fingerprint(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
) -> str:
    names = (
        "operational_label_contract.json",
        "annotation_calibration.json",
        "ANNOTATION_CALIBRATION.md",
        "source_manifest.jsonl",
        "prior_exposure_manifest.jsonl",
        "annotation_manifest.jsonl",
        "normalization_manifest.jsonl",
        "topology_manifest.jsonl",
        "topology_census.json",
        "identity_components.json",
        "wavlm_pretraining_overlap.json",
    )
    return canonical_sha256(
        {
            "data": {name: sha256_file(data_dir / name) for name in names},
            "historical_prior_exposure_configs": {
                relative_path: sha256_file(data_dir.parents[2] / relative_path)
                for relative_path in HISTORICAL_CONFIGS.values()
            },
            "registry": sha256_file(registry_path),
            "source_registry": sha256_file(source_registry_path),
        }
    )


@lru_cache(maxsize=8)
def _build_split_manifest_cached(
    data_dir: str,
    registry_path: str,
    source_registry_path: str,
    input_fingerprint: str,
) -> dict[str, Any]:
    if not input_fingerprint:
        raise SplitAssignmentError("split input fingerprint is missing")
    return _build_split_manifest_uncached(
        Path(data_dir),
        Path(registry_path),
        Path(source_registry_path),
        input_fingerprint,
    )


def build_split_manifest(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
) -> dict[str, Any]:
    resolved_data_dir = data_dir.resolve()
    resolved_registry = registry_path.resolve()
    resolved_source_registry = source_registry_path.resolve()
    fingerprint = _input_fingerprint(
        resolved_data_dir, resolved_registry, resolved_source_registry
    )
    return copy.deepcopy(
        _build_split_manifest_cached(
            str(resolved_data_dir),
            str(resolved_registry),
            str(resolved_source_registry),
            fingerprint,
        )
    )


def build_resolved_split_feasibility(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = build_split_feasibility(data_dir, registry_path, source_registry_path)
    checked = manifest if manifest is not None else _load_json(data_dir / "split_manifest.json")
    rebuilt = build_split_manifest(data_dir, registry_path, source_registry_path)
    if checked != rebuilt:
        raise SplitAssignmentError("checked split manifest is not current")
    resolved = dict(result)
    resolved.update(
        {
            "search_status": "valid_component_assignment_found",
            "valid_assignment_exists": True,
            "assignment_manifest_emitted": True,
            "assignments": {
                row["component_id"]: row["role"] for row in checked["assignments"]["components"]
            },
            "split_manifest_canonical_sha256": canonical_sha256(checked),
            "role_summaries": checked["role_summaries"],
            "hard_gate_status": checked["hard_gate_status"],
            "search": checked["search"],
        }
    )
    return resolved


def validate_checked_split_package(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
    manifest_path: Path | None = None,
    feasibility_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checked_manifest = _load_json(manifest_path or data_dir / "split_manifest.json")
    rebuilt_manifest = build_split_manifest(data_dir, registry_path, source_registry_path)
    if checked_manifest != rebuilt_manifest:
        raise SplitAssignmentError("checked split manifest is not current")
    checked_feasibility = _load_json(
        feasibility_path or data_dir / "split_feasibility.json"
    )
    rebuilt_feasibility = build_resolved_split_feasibility(
        data_dir, registry_path, source_registry_path, rebuilt_manifest
    )
    if checked_feasibility != rebuilt_feasibility:
        raise SplitAssignmentError("checked split feasibility is not current")
    return checked_manifest, checked_feasibility


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_split_assignment(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
    manifest_output: Path,
    feasibility_output: Path,
    markdown_output: Path,
) -> None:
    manifest = build_split_manifest(data_dir, registry_path, source_registry_path)
    _write_json_atomic(manifest_output, manifest)
    resolved = build_resolved_split_feasibility(
        data_dir, registry_path, source_registry_path, manifest
    )
    _write_json_atomic(feasibility_output, resolved)
    census = _load_json(data_dir / "topology_census.json")
    topology_rows = _load_jsonl(data_dir / "topology_manifest.jsonl")
    markdown_output.write_text(
        render_data_census(
            census,
            topology_rows,
            manifest,
            sha256_file(manifest_output),
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--feasibility-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    write_split_assignment(
        args.data_dir.resolve(),
        args.registry.resolve(),
        args.source_registry.resolve(),
        args.manifest_output.resolve(),
        args.feasibility_output.resolve(),
        args.markdown_output.resolve(),
    )


if __name__ == "__main__":
    main()
