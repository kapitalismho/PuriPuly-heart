"""Zero-cost stand-in command for exercising supervised_exec.py.

Emits a deterministic byte payload on stdout (with a self-describing marker as
the final line), periodic stderr lines, an optional sleep, and a chosen exit
code. It performs no network or provider activity and produces no experiment
result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from typing import Sequence

BLOCK_BYTES = 1 << 20


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--bytes", type=int, default=BLOCK_BYTES)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--stderr-lines", type=int, default=0)
    parser.add_argument("--stderr-period-s", type=float, default=0.0)
    parser.add_argument(
        "--unicode-sample",
        action="store_true",
        help="include Korean plus characters outside the cp949 locale codec in the stdout marker",
    )
    args = parser.parse_args(argv)

    block = random.Random(20260912).randbytes(BLOCK_BYTES)
    digest = hashlib.sha256()
    out = sys.stdout.buffer
    if args.unicode_sample:
        sample = "반말 존댓말 ∑ 𝄞 😀"
        repeats = max(1, int(args.bytes) // len(sample.encode("utf-8")))
        text = sample * repeats
        sys.stdout.write(text + "\n")  # text path through the stdio codec (a cp949 stdout raises here)
        sys.stdout.flush()
        marker = json.dumps(
            {"marker": "standin-text", "chars": len(text), "repeats": repeats, "unicode_sample": sample},
            ensure_ascii=False,
        )
    else:
        remaining = max(0, int(args.bytes))
        while remaining > 0:
            chunk = block[: min(len(block), remaining)]
            out.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        out.flush()
        marker = "\n" + json.dumps(
            {"marker": "standin", "payload_bytes": int(args.bytes), "sha256": digest.hexdigest()}, ensure_ascii=False
        )
    print(marker, flush=True)  # text path through the stdio codec, as the real launcher dump does

    started = time.monotonic()
    for index in range(max(0, int(args.stderr_lines))):
        print(f"standin stderr line {index + 1}/{args.stderr_lines} at t={time.monotonic() - started:.3f}s", file=sys.stderr, flush=True)
        if args.stderr_period_s > 0:
            time.sleep(args.stderr_period_s)
    if args.sleep_s > 0:
        print(f"standin sleeping {args.sleep_s}s", file=sys.stderr, flush=True)
        time.sleep(args.sleep_s)
    print(f"standin exiting {args.exit_code}", file=sys.stderr, flush=True)
    return int(args.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
