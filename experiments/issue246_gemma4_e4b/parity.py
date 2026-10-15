from __future__ import annotations

import argparse
from types import SimpleNamespace

from measure import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--off-only", action="store_true")
    args = parser.parse_args()
    for label, extra, mtp in (
        ("new-cpu-greedy-on", ["--cache-ram", "0", "--spec-draft-sampling", "greedy"], True),
        ("new-cpu-greedy-off-clean", ["--cache-ram", "0"], False),
    ):
        if args.off_only and mtp:
            continue
        run(SimpleNamespace(release="b11537", backend="cpu", label=label, phase="quality", count=12, extra=extra, temperature=0.0, mtp=mtp))


if __name__ == "__main__":
    main()
