from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from puripuly_heart.app.adapters.nemotron.artifacts import _parse, inspect_artifact, sha256_file


def verify(source: Path, output: Path) -> dict:
    source_fields, source_tensors, _, _, source_start, _ = _parse(source)
    output_fields, output_tensors, _, _, output_start, _ = _parse(output)
    if {k: v for k, v in source_fields.items() if k != "general.file_type"} != {
        k: v for k, v in output_fields.items() if k != "general.file_type"
    }:
        raise ValueError("conversion changed nonprecision metadata")
    if len(source_tensors) != len(output_tensors):
        raise ValueError("conversion changed tensor count")
    total_values = 0
    with source.open("rb") as src, output.open("rb") as dst:
        for original, converted in zip(source_tensors, output_tensors):
            if original[:2] != converted[:2]:
                raise ValueError("conversion changed tensor identity/geometry")
            src.seek(source_start + original[3])
            dst.seek(output_start + converted[3])
            source_dtype = np.dtype("<f4" if original[2] == 0 else "<f2")
            target_dtype = np.dtype("<f4" if converted[2] == 0 else "<f2")
            remaining = original[4] // source_dtype.itemsize
            total_values += remaining
            while remaining:
                count = min(remaining, 262144)
                values = np.frombuffer(src.read(count * source_dtype.itemsize), dtype=source_dtype)
                observed = np.frombuffer(
                    dst.read(count * target_dtype.itemsize), dtype=target_dtype
                )
                if (
                    values.size != count
                    or observed.size != count
                    or not np.array_equal(values.astype(target_dtype), observed)
                ):
                    raise ValueError(f"conversion tensor mismatch: {original[0]}")
                remaining -= count
    return {
        "status": "passed",
        "tensors_checked": len(source_tensors),
        "values_checked": total_values,
        "source_sha256": sha256_file(source),
        "output": inspect_artifact(output),
        "verified": "every target value equals exact source cast under pinned official dtype selection; auxiliary F32 unchanged",
        "lineage": "source-checkpoint relationship remains unverified",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.source, args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
