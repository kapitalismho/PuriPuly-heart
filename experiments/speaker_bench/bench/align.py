from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Item:
    token: str
    lo: float
    hi: float


def _compatible(ref: Item, hyp: Item, tolerance: float) -> bool:
    return hyp.lo <= ref.hi + tolerance and ref.lo - tolerance <= hyp.hi


def align(ref: list[Item], hyp: list[Item], *, tolerance: float) -> list[int | None]:
    """Minimum edit alignment; matches and substitutions require time compatibility.

    Returns, for each reference item, the aligned hypothesis index or None (deleted).
    On ties, later hypothesis tokens become insertions, so reference tokens align to the
    earliest compatible hypothesis tokens.
    """
    n, m = len(ref), len(hyp)
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = inf
            if i and j and _compatible(ref[i - 1], hyp[j - 1], tolerance):
                best = cost[i - 1][j - 1] + (0 if ref[i - 1].token == hyp[j - 1].token else 1)
            if i:
                best = min(best, cost[i - 1][j] + 1)
            if j:
                best = min(best, cost[i][j - 1] + 1)
            cost[i][j] = best
    mapping: list[int | None] = [None] * n
    i, j = n, m
    while i or j:
        here = cost[i][j]
        if j and cost[i][j - 1] + 1 == here:
            j -= 1
            continue
        if (
            i
            and j
            and _compatible(ref[i - 1], hyp[j - 1], tolerance)
            and cost[i - 1][j - 1] + (0 if ref[i - 1].token == hyp[j - 1].token else 1) == here
        ):
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
            continue
        i -= 1
    return mapping


def edit_counts(ref: list[str], hyp: list[str]) -> dict[str, int]:
    n, m = len(ref), len(hyp)
    previous = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        current = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            sub = previous[j - 1]
            substitution = (
                sub[0] + (ref[i - 1] != hyp[j - 1]),
                sub[1] + (ref[i - 1] != hyp[j - 1]),
                sub[2],
                sub[3],
            )
            dele = previous[j]
            deletion = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
            ins = current[j - 1]
            insertion = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
            current.append(min(substitution, deletion, insertion, key=lambda c: c[0]))
        previous = current
    total, substitutions, deletions, insertions = previous[m]
    return {
        "reference": n,
        "errors": total,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }
