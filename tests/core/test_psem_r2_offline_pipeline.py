from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.pipeline import (
    LIVE_ROUTE,
    run_intercepted_deepgram_path,
    run_synthetic_path,
)


@pytest.mark.asyncio
async def test_synthetic_adapter_path_splits_and_conserves() -> None:
    result = await run_synthetic_path()
    enabled = result["enabled"]
    assert result["network"] is False
    assert result["text"] == "Hello there"
    assert enabled["conserved"] is True
    assert enabled["group_ids"] == ["CURRENT-0", "OTHER-1"]
    assert enabled["child_groups"] == ["CURRENT-0", "OTHER-1"]
    assert enabled["child_texts"] == ["Hello ", "there"]
    assert result["disabled"]["n_units"] == 0
    assert result["disabled"]["child_groups"] == [""]
    assert result["live_route"]["asr_provider"] == "deepgram"
    assert result["live_route"]["asr_model"] == LIVE_ROUTE["asr_model"]


@pytest.mark.asyncio
async def test_intercepted_deepgram_words_split_and_conserve() -> None:
    result = await run_intercepted_deepgram_path()
    enabled = result["enabled"]
    assert result["network"] is False
    assert result["intercept"] is True
    assert result["adapter_reads_words"] is True
    assert result["n_timed"] == 2
    assert result["timed_start_ms"] == [0, 100]
    assert enabled["conserved"] is True
    assert enabled["group_ids"] == ["CURRENT-0", "OTHER-1"]
    assert enabled["child_texts"] == ["Hello ", "there"]
