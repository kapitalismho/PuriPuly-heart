from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.live_runner import LIVE_ROUTE
from experiments.psem_r2_policy.pipeline import (
    run_intercepted_deepgram_path,
    run_paid_live,
    run_synthetic_path,
)


def load_protocol() -> dict:
    return json.loads((EXP / "PROTOCOL.json").read_text(encoding="utf-8"))


def _ok_split(result: dict) -> bool:
    enabled = result["enabled"]
    return (
        result["network"] is False
        and result["text"] == "Hello there"
        and enabled["conserved"] is True
        and enabled["group_ids"] == ["CURRENT-0", "OTHER-1"]
        and enabled["child_groups"] == ["CURRENT-0", "OTHER-1"]
    )


async def execute(*, paid: bool, offline_replay: bool) -> dict:
    protocol = load_protocol()
    if paid:
        paid_result = await run_paid_live()
        paid_result["asr"] = LIVE_ROUTE["asr_provider"]
        paid_result["asr_model"] = LIVE_ROUTE["asr_model"]
        paid_result["translation"] = LIVE_ROUTE["translation"]
        paid_result["protocol_revision"] = protocol.get("revision")
        return paid_result
    if offline_replay:
        synthetic = await run_synthetic_path()
        return {
            "ok": _ok_split(synthetic),
            "network": False,
            "mode": "offline_replay",
            "asr": LIVE_ROUTE["asr_provider"],
            "path": synthetic["path"],
            "synthetic": synthetic,
            "protocol_revision": protocol.get("revision"),
        }
    intercept = await run_intercepted_deepgram_path()
    return {
        "ok": _ok_split(intercept) and intercept["n_timed"] == 2,
        "network": False,
        "mode": "intercept",
        "asr": LIVE_ROUTE["asr_provider"],
        "asr_model": LIVE_ROUTE["asr_model"],
        "translation": LIVE_ROUTE["translation"],
        "path": intercept["path"],
        "intercept": intercept,
        "protocol_revision": protocol.get("revision"),
        "protocol_asr": protocol.get("providers", {}).get("asr"),
        "live_methods": intercept.get("methods"),
        "open_session_calls": intercept.get("open_session_calls"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--offline-replay", nargs="?", const="generic")
    args = parser.parse_args(argv)
    payload = asyncio.run(
        execute(paid=args.paid, offline_replay=bool(args.offline_replay))
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.paid:
        return 0 if payload.get("network") is False else 1
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
