from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.budget import (
    BillingBoundError,
    BudgetError,
    BudgetLedger,
    deepgram_reserve_usd,
    openrouter_reserve_usd,
)


def test_deepgram_requires_positive_audio_seconds() -> None:
    with pytest.raises(BudgetError, match="max_audio_seconds"):
        deepgram_reserve_usd(max_audio_seconds=0)


def test_deepgram_reserve_uses_processed_audio_not_websocket_wall() -> None:
    one_minute = deepgram_reserve_usd(max_audio_seconds=60)
    assert one_minute == pytest.approx(0.0077)
    long_wall_short_audio = deepgram_reserve_usd(max_audio_seconds=1)
    assert long_wall_short_audio == pytest.approx(0.0077 / 60.0)
    stereo = deepgram_reserve_usd(max_audio_seconds=60, channels=2)
    assert stereo == pytest.approx(0.0154)


def test_deepgram_context_pad_is_included_and_ceiled() -> None:
    exact = deepgram_reserve_usd(max_audio_seconds=1.1, context_pad_seconds=0.2)
    assert exact == pytest.approx(2 / 60.0 * 0.0077)


def test_openrouter_utf8_bytes_exceed_korean_char_count() -> None:
    body = {
        "messages": [{"role": "user", "content": "안녕"}],
        "max_tokens": 100,
    }
    serialized = json.dumps(body, ensure_ascii=False)
    by_bytes = openrouter_reserve_usd(serialized_request=body)
    by_chars = (
        len(serialized) * 0.042 + 100 * 0.22
    ) / 1_000_000.0
    assert by_bytes > by_chars
    assert len("안녕".encode("utf-8")) == 6
    assert len("안녕") == 2
    with pytest.raises(BillingBoundError):
        openrouter_reserve_usd(serialized_request=body, max_tokens=101)


def test_phase_and_total_caps_block_over_reserve(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json", cap_usd=5.0)
    ledger.reserve("dev-1", phase="dev", amount_usd=2.0)
    with pytest.raises(BudgetError, match="dev phase cap"):
        ledger.reserve("dev-2", phase="dev", amount_usd=0.01)
    ledger.reserve("hold-1", phase="holdout", amount_usd=2.5)
    with pytest.raises(BudgetError, match="cap would be exceeded"):
        ledger.reserve("cont-1", phase="contingency", amount_usd=0.51)
    snap = ledger.snapshot()
    assert snap.reserved_usd == pytest.approx(4.5)
    assert snap.remaining_usd == pytest.approx(0.5)


def test_uncertain_settle_keeps_reserve(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("a", phase="dev", amount_usd=0.4)
    ledger.settle("a", billed_usd=None)
    snap = ledger.snapshot()
    assert snap.reserved_usd == pytest.approx(0.4)
    assert snap.spent_usd == 0


def test_failed_request_cannot_release_from_missing_usage_log(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("missing-log", phase="dev", amount_usd=0.2)
    ledger.settle("missing-log", billed_usd=None, keep_reserve=True)
    assert ledger.snapshot().reserved_usd == pytest.approx(0.2)


def test_reliable_settle_releases_unused_reserve(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("a", phase="dev", amount_usd=0.4)
    ledger.settle("a", billed_usd=0.1)
    snap = ledger.snapshot()
    assert snap.spent_usd == pytest.approx(0.1)
    assert snap.reserved_usd == 0


def test_restart_reloads_inflight_reserve(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    first = BudgetLedger(path)
    first.reserve("live", phase="dev", amount_usd=0.8)
    second = BudgetLedger(path)
    snap = second.snapshot()
    assert snap.reserved_usd == pytest.approx(0.8)
    with pytest.raises(BudgetError, match="already reserved"):
        second.reserve("live", phase="dev", amount_usd=0.1)


def test_concurrent_last_slot_has_one_winner(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("seed", phase="dev", amount_usd=1.6)
    results: list[str] = []

    def attempt(name: str) -> None:
        try:
            ledger.reserve(name, phase="dev", amount_usd=0.4)
            results.append(f"ok:{name}")
        except BudgetError:
            results.append(f"fail:{name}")

    threads = [
        threading.Thread(target=attempt, args=("t1",)),
        threading.Thread(target=attempt, args=("t2",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(item.startswith("ok:") for item in results) == 1
    assert sum(item.startswith("fail:") for item in results) == 1
    snap = ledger.snapshot()
    assert snap.phase_reserved["dev"] == pytest.approx(2.0)


def test_deepgram_hangover_preroll_tail_and_reconnect_are_reserved() -> None:
    base = deepgram_reserve_usd(max_audio_seconds=1.0)
    padded = deepgram_reserve_usd(
        max_audio_seconds=1.0,
        hangover_seconds=0.8,
        preroll_seconds=0.5,
        tail_seconds=512.0 / 16000.0,
        reconnect_bound=1,
    )
    assert padded > base
    assert padded == pytest.approx(6 / 60.0 * 0.0077)
