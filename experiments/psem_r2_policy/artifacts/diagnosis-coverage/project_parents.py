"""Project the two zero-attributable parents out of the 602 MB case payload.

Read-only: mmap + JSON-aware element scan; only the two matching parent
objects are ever decoded. Prints token-level timing fields verbatim.
"""

from __future__ import annotations

import json
import mmap
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = HERE.parent.parent  # experiments/psem_r2_policy
CASE = POLICY / "artifacts" / "dev" / "ES2009a" / "20260911T234330502940Z.json"

TARGETS = (
    "175c0667-a551-49c7-88ce-27faa4780d29",
    "5e2351e4-e74f-4054-998e-e383042ae569",
)


def elements(buf: memoryview, open_idx: int):
    """Yield (start, end) spans of each element of the array opening at open_idx."""
    n = len(buf)
    i = open_idx + 1  # skip '['
    depth = 0
    in_str = False
    start = None
    while i < n:
        b = buf[i]
        if in_str:
            if b == 0x5C:
                i += 2
                continue
            if b == 0x22:
                in_str = False
            i += 1
            continue
        if b == 0x22:
            in_str = True
        elif b in (0x7B, 0x5B):
            if depth == 0 and b == 0x7B:
                start = i
            depth += 1
        elif b in (0x7D, 0x5D):
            depth -= 1
            if depth == 0:
                if b == 0x5D and start is None:
                    return
                yield start, i + 1
                start = None
            elif depth < 0:
                return
        i += 1


def main() -> int:
    size = CASE.stat().st_size
    print(f"case={CASE.name} size={size}")
    keep: dict[str, dict] = {}
    with CASE.open("rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        buf = memoryview(mm)
        arr = mm.find(b'"parents"')
        arr = mm.find(b"[", arr)
        count = 0
        for s, e in elements(buf, arr):
            count += 1
            chunk = mm[s:e]
            for t in TARGETS:
                if t.encode() in chunk:
                    keep[t] = json.loads(chunk)
        print(f"parents scanned={count}")
        for t in TARGETS:
            rec = keep.get(t)
            if rec is None:
                print(f"!! {t} not found")
                continue
            print("\n=== parent", t)
            for key in (
                "index",
                "outcome",
                "terminal_outcome",
                "status",
                "seal_reason",
                "text_authority",
                "text",
                "n_timed",
                "timed_start_ms",
                "timed_timings",
                "span",
                "group_ids",
                "conserved",
                "unknown_reasons",
            ):
                print(f"  {key} = {json.dumps(rec.get(key), ensure_ascii=False)[:400]}")
            checked = (rec.get("guard") or {}).get("checked")
            print("  guard.checked =", json.dumps(checked, ensure_ascii=False)[:400])
            toks = rec.get("tokens") or []
            print(f"  n_tokens = {len(toks)}")
            for tok in toks:
                print("   tok:", json.dumps(tok, ensure_ascii=False)[:400])
        buf.release()
        mm.close()
    out = HERE / "parents-projection.json"
    out.write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
