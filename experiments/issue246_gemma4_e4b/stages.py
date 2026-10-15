from __future__ import annotations

import argparse
from types import SimpleNamespace

from measure import run

STAGES = {
    "compatibility": [
        ("b10423", "vulkan", "old-vulkan-quality", "quality", 12, []),
        ("b11537", "vulkan", "new-vulkan-quality", "quality", 12, []),
    ],
    "ple-tuning": [
        ("b11537", "cpu", "new-cpu-auto-tune", "tuning", 18, []),
        ("b11537", "cpu", "new-cpu-lazy-on-tune", "tuning", 18, ["--lazy-mode", "on"]),
        ("b11537", "cpu", "new-cpu-lazy-off-sanity", "tuning", 3, ["--lazy-mode", "off"]),
        ("b11537", "vulkan", "new-vulkan-auto-tune", "tuning", 18, []),
        ("b11537", "vulkan", "new-vulkan-lazy-on-tune", "tuning", 18, ["--lazy-mode", "on"]),
        ("b11537", "vulkan", "new-vulkan-lazy-off-sanity", "tuning", 3, ["--lazy-mode", "off"]),
    ],
    "final-update": [
        ("b10423", "cpu", "old-cpu-final-a", "final", 33, []),
        ("b11537", "cpu", "new-cpu-final-a", "final", 33, []),
        ("b11537", "cpu", "new-cpu-final-b", "final", 33, []),
        ("b10423", "cpu", "old-cpu-final-b", "final", 33, []),
        ("b10423", "vulkan", "old-vulkan-final-a", "final", 33, []),
        ("b11537", "vulkan", "new-vulkan-final-a", "final", 33, []),
        ("b11537", "vulkan", "new-vulkan-final-b", "final", 33, []),
        ("b10423", "vulkan", "old-vulkan-final-b", "final", 33, []),
    ],
    "resource-tuning": [
        ("b11537", "cpu", "new-cpu-cache0-tune", "tuning", 18, ["--cache-ram", "0"]),
        ("b11537", "cpu", "new-cpu-cache0-fa-on-tune", "tuning", 18, ["--cache-ram", "0", "--flash-attn", "on"]),
        ("b10423", "cpu", "old-cpu-cache0-tune", "tuning", 18, ["--cache-ram", "0"]),
        ("b11537", "vulkan", "new-vulkan-cache0-tune", "tuning", 18, ["--cache-ram", "0"]),
    ],
    "followon-placement": [
        ("b11630", "cpu", "followon-cpu-on-tune", "tuning", 9, ["--lazy-mode", "on"]),
        ("b11630", "vulkan", "followon-vulkan-on-tune", "tuning", 9, ["--lazy-mode", "on"]),
    ],
    "final-cache0": [
        ("b10423", "cpu", "old-cpu-cache0-final-a", "final", 33, ["--cache-ram", "0"]),
        ("b11537", "cpu", "new-cpu-cache0-final-a", "final", 33, ["--cache-ram", "0"]),
        ("b11537", "cpu", "new-cpu-cache0-final-b", "final", 33, ["--cache-ram", "0"]),
        ("b10423", "cpu", "old-cpu-cache0-final-b", "final", 33, ["--cache-ram", "0"]),
        ("b10423", "vulkan", "old-vulkan-cache0-final-a", "final", 33, ["--cache-ram", "0"]),
        ("b11537", "vulkan", "new-vulkan-cache0-final-a", "final", 33, ["--cache-ram", "0"]),
        ("b11537", "vulkan", "new-vulkan-cache0-final-b", "final", 33, ["--cache-ram", "0"]),
        ("b10423", "vulkan", "old-vulkan-cache0-final-b", "final", 33, ["--cache-ram", "0"]),
    ],
    "soak-default": [
        ("b10423", "cpu", "old-cpu-default-soak", "soak", 96, []),
        ("b11537", "cpu", "new-cpu-default-soak", "soak", 96, []),
        ("b10423", "vulkan", "old-vulkan-default-soak", "soak", 96, []),
        ("b11537", "vulkan", "new-vulkan-default-soak", "soak", 96, []),
    ],
    "soak-cache0": [
        ("b10423", "cpu", "old-cpu-cache0-soak", "soak", 200, ["--cache-ram", "0"]),
        ("b11537", "cpu", "new-cpu-cache0-soak", "soak", 200, ["--cache-ram", "0"]),
        ("b10423", "vulkan", "old-vulkan-cache0-soak", "soak", 200, ["--cache-ram", "0"]),
        ("b11537", "vulkan", "new-vulkan-cache0-soak", "soak", 200, ["--cache-ram", "0"]),
    ],
    "repack-diagnostic": [
        ("b11537", "cpu", "new-cpu-cache0-norepack-tune", "tuning", 6, ["--cache-ram", "0", "--no-repack"]),
    ],
    "literal-confirmation": [
        ("b10423", "cpu", "literal-old-cpu-default", "literal", 12, []),
        ("b11537", "cpu", "literal-new-cpu-default", "literal", 12, []),
        ("b10423", "cpu", "literal-old-cpu-cache0", "literal", 12, ["--cache-ram", "0"]),
        ("b11537", "cpu", "literal-new-cpu-cache0", "literal", 12, ["--cache-ram", "0"]),
        ("b10423", "vulkan", "literal-old-vulkan-default", "literal", 12, []),
        ("b11537", "vulkan", "literal-new-vulkan-default", "literal", 12, []),
        ("b10423", "vulkan", "literal-old-vulkan-cache0", "literal", 12, ["--cache-ram", "0"]),
        ("b11537", "vulkan", "literal-new-vulkan-cache0", "literal", 12, ["--cache-ram", "0"]),
    ],
    "cache-eviction-attribution": [
        ("b11537", "cpu", "new-cpu-cache512-soak", "soak", 80, ["--cache-ram", "512"]),
        ("b11537", "vulkan", "new-vulkan-cache512-soak", "soak", 80, ["--cache-ram", "512"]),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    for release, backend, label, phase, count, extra in STAGES[args.stage]:
        run(SimpleNamespace(release=release, backend=backend, label=label, phase=phase, count=count, extra=extra, temperature=0.6))


if __name__ == "__main__":
    main()
