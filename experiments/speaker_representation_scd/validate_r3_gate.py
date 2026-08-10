from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.speaker_representation_scd.r3_gate import validate_r3_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--skip-process-scan", action="store_true")
    args = parser.parse_args()
    result = validate_r3_gate(
        cache_root=args.cache_root,
        scan_processes=not args.skip_process_scan,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
