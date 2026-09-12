"""Archived one-shot post-case adapter for the 2026-09-11 historical cp949 dump.

Not part of the canonical execution path and never invoked automatically. The
supervised driver now pins PYTHONIOENCODING=utf-8 for its children, so future
launcher dumps are valid UTF-8 and the read-only canonical wrapper
(`artifacts/excluded-launcher-shadowing/wrappers/case_tools.py`) consumes them
directly. This tool only exists to re-derive the ES2009a 2026-09-11 case output
whose dump was written with the locale codec before that pin: it reads the same
dump with an explicit, recorded codec cascade (utf-8 strict, then cp949 strict,
no replacement decoding) and performs the identical canonical steps.

Usage (from the target worktree root):
  .venv/Scripts/python.exe .../dev-supervised/post_case.py --meeting M --case-dir DIR [--verify-only]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

TARGET = Path(
    r"C:/Users/salee/Documents/dev/puripuly_heart/.worktrees/puripuly_heart/experiment-v2-speaker-change-turn-boundaries-ls"
)
HOME_DIR = TARGET / "experiments/psem_r2_policy/artifacts/dev-supervised"
CASE_TOOLS = TARGET / "experiments/psem_r2_policy/artifacts/excluded-launcher-shadowing/wrappers/case_tools.py"
CODECS = ("utf-8", "cp949")

sys.dont_write_bytecode = True


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_dump(path: Path) -> tuple[dict[str, Any], str, int]:
    """Return (payload, codec used, dump bytes), strict decode, never tolerant-lossy."""
    raw_bytes = path.stat().st_size
    errors: list[str] = []
    for codec in CODECS:
        try:
            with path.open("r", encoding=codec, errors="strict") as handle:
                return json.load(handle), codec, raw_bytes
        except UnicodeDecodeError as exc:
            errors.append(f"{codec}: UnicodeDecodeError at {exc.start}")
        except json.JSONDecodeError as exc:
            errors.append(f"{codec}: JSONDecodeError {exc}")
    raise SystemExit("dump is not readable under any recorded codec: " + "; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--meeting", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--label", default="post-case")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="exercise mode: decode and summarise the dump without writing a canonical case output",
    )
    args = parser.parse_args(argv)
    case_dir = Path(args.case_dir)
    stdout_path = case_dir / "stdout.json"

    tools = _load_module("psem_case_tools", CASE_TOOLS)
    import experiments.psem_r2_policy.phase as harness_phase  # noqa: PLC0415 - after path insert
    import experiments.psem_r2_policy.budget as harness_budget  # noqa: PLC0415

    assert harness_phase.ARTIFACTS == tools.CANONICAL_ARTIFACTS, harness_phase.ARTIFACTS
    assert harness_budget.LEDGER_PATH == tools.CANONICAL_ARTIFACTS / "budget_ledger.json"

    payload, codec, dump_bytes = load_dump(stdout_path)
    digest = hashlib.sha256(stdout_path.read_bytes()).hexdigest()
    if args.verify_only:
        output = {
            "canonical_write": "skipped by --verify-only (exercise mode); no artifact outside the driver home dir was written"
        }
    else:
        output = harness_phase.write_case_output("dev", args.meeting, payload)
    pointer = "post-case-verify.json" if args.verify_only else "canonical_case.json"
    (case_dir / pointer).write_text(
        json.dumps(
            {
                "meeting": args.meeting,
                "case_output": output,
                "stdout_sha256": digest,
                "stdout_bytes": dump_bytes,
                "stdout_codec": codec,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    meta: dict[str, Any] = {}
    for line in (case_dir / "stderr.log").read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                meta = json.loads(stripped)
                break
            except json.JSONDecodeError:
                continue
    summary = {
        "case_output": output,
        "stdout": {
            "path": str(stdout_path),
            "bytes": dump_bytes,
            "sha256": digest,
            "codec": codec,
            "codec_note": "dump serialised with the locale codec because stdout was redirected to a file; content verified lossless under this codec",
        },
        "launch": {
            "capsule_root": meta.get("capsule_root"),
            "fingerprint": meta.get("fingerprint"),
            "ledger_path": meta.get("ledger_path"),
            "case_output_dir": meta.get("case_output_dir"),
            "modules": meta.get("modules"),
            "prompt": meta.get("prompt"),
            "prompt_probe": meta.get("prompt_probe"),
        },
        **tools._compact(payload),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "meeting",
                    "ok",
                    "completed",
                    "execution_completed",
                    "evaluation_valid",
                    "operational_clean",
                    "counts",
                    "ledger",
                    "case_output",
                    "u8_execution_incomplete_reasons",
                    "u8_evaluation_invalid_reasons",
                    "u8_safety_failures",
                )
                if key in summary
            }
            | {"stdout_codec": codec, "stdout_bytes": dump_bytes},
            indent=1,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
