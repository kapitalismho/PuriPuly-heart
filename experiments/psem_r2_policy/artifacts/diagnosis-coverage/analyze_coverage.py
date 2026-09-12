"""Overlap analysis for the two zero-attributable parents (read-only).

Recomputes metrics.attribute_tokens on the projected token records and prints
every GT-word overlap per token interval so the `mixed` verdict can be judged
against the frozen rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = HERE.parent.parent
sys.path.insert(0, str(POLICY))

import metrics  # noqa: E402

HZ = metrics.HZ
PROJECTION = HERE / "parents-projection.json"


def s(sample: int | None) -> str:
    return "None" if sample is None else f"{sample / HZ:.3f}"


def main() -> int:
    data = json.loads(PROJECTION.read_text(encoding="utf-8"))
    words = metrics.load_ami_words("ES2009a")
    print(f"GT words loaded: {len(words)}")
    for pid, rec in data.items():
        span = rec.get("span")
        print(f"\n===== parent {pid} idx={rec.get('index')} span={span} ({s(span[0])}-{s(span[1])}s) seal={rec.get('seal_reason')}")
        print(f"  text={rec.get('text')!r}")
        assigned = rec.get("assignment")
        print("  assignment:", json.dumps(assigned, ensure_ascii=False)[:300])
        toks = rec.get("tokens") or []
        attributed = metrics.attribute_tokens(toks, words)
        statuses = [row["status"] for row in attributed]
        print("  recomputed statuses:", statuses)
        print("  guard.checked:", json.dumps((rec.get("guard") or {}).get("checked"), ensure_ascii=False))
        print("  GT words inside parent span:")
        for w in words:
            if span and (span[0] <= w["start_src"] < span[1] or span[0] < w["end_src"] <= span[1]):
                print(f"    {w['role']} {s(w['start_src'])}-{s(w['end_src'])} {w['text']!r}")
        for row, tok in zip(attributed, toks):
            if tok.get("timing") != "interval":
                print(f"  tok{row['token_id']} {row['text']!r} timing={tok.get('timing')} -> {row['status']}")
                continue
            start, end = row["start_src"], row["end_src"]
            print(f"  tok{row['token_id']} {row['text']!r} {s(start)}-{s(end)}s ({end - start} samp) -> {row['status']} roles={row['roles']}")
            for w in words:
                if w["end_src"] > start and w["start_src"] < end:
                    ov = min(end, w["end_src"]) - max(start, w["start_src"])
                    print(
                        f"      {w['role']} {s(w['start_src'])}-{s(w['end_src'])} ov={ov / HZ:.3f}s {w['text']!r}"
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
