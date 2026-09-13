from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Phase = Literal["dev", "holdout", "contingency"]

_PHASE_CAPS = {"dev": 3.0, "holdout": 2.25, "contingency": 0.0}
_TOTAL_CAP = 5.25
_CREDIT_KIND_PREFIX = "deepgram"
_RATES_PATH = Path(__file__).with_name("rates.json")
_BOUNDS_PATH = Path(__file__).with_name("BILLING_BOUNDS.json")
LEDGER_PATH = Path(__file__).resolve().parent / "artifacts" / "budget_ledger.json"


class BudgetError(RuntimeError):
    pass


class BillingBoundError(BudgetError):
    pass


def _is_credit_kind(kind: object) -> bool:
    """Deepgram usage is credit-funded: informational, never cash-capped.

    Unclassified kinds are cash so a missing or unknown marker fails closed.
    """
    return str(kind or "").startswith(_CREDIT_KIND_PREFIX)


def _lock_file(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_rates() -> dict[str, Any]:
    return json.loads(_RATES_PATH.read_text(encoding="utf-8"))


def load_billing_bounds() -> dict[str, Any]:
    return json.loads(_BOUNDS_PATH.read_text(encoding="utf-8"))


def openrouter_reserve_usd(
    *,
    serialized_request: str | bytes | dict[str, Any],
    max_tokens: int | None = None,
) -> float:
    rates = load_rates()["openrouter"]
    pinned = int(rates["pinned_max_tokens"])
    if max_tokens is None:
        max_tokens = pinned
    if max_tokens > pinned:
        raise BillingBoundError("openrouter max_tokens exceeds pinned adapter bound")
    if isinstance(serialized_request, dict):
        payload = json.dumps(serialized_request, ensure_ascii=False).encode("utf-8")
    elif isinstance(serialized_request, str):
        payload = serialized_request.encode("utf-8")
    else:
        payload = serialized_request
    input_tokens = len(payload)
    return (
        input_tokens * float(rates["input_usd_per_million"])
        + max_tokens * float(rates["output_usd_per_million"])
    ) / 1_000_000.0


def deepgram_reserve_usd(
    *,
    max_audio_seconds: float,
    channels: int = 1,
    context_pad_seconds: float = 0.0,
    hangover_seconds: float = 0.0,
    preroll_seconds: float = 0.0,
    tail_seconds: float = 0.0,
    copies: int = 1,
    reconnect_bound: int = 0,
) -> float:
    bounds = load_billing_bounds()["deepgram"]
    if not bounds.get("defensible"):
        raise BillingBoundError(str(bounds.get("blocker") or "deepgram billing is not defensible"))
    if max_audio_seconds <= 0:
        raise BudgetError("max_audio_seconds must be positive")
    if channels < 1:
        raise BudgetError("channels must be at least 1")
    if copies < 1:
        raise BudgetError("copies must be at least 1")
    if reconnect_bound < 0:
        raise BudgetError("reconnect_bound cannot be negative")
    for name, value in (
        ("context_pad_seconds", context_pad_seconds),
        ("hangover_seconds", hangover_seconds),
        ("preroll_seconds", preroll_seconds),
        ("tail_seconds", tail_seconds),
    ):
        if value < 0:
            raise BudgetError(f"{name} cannot be negative")
    rates = load_rates()["deepgram"]
    sent = (
        max_audio_seconds + context_pad_seconds + hangover_seconds + preroll_seconds + tail_seconds
    )
    sessions = copies * (1 + reconnect_bound)
    billed_seconds = math.ceil(sent * channels) * sessions
    if billed_seconds < 1:
        billed_seconds = 1
    return billed_seconds / 60.0 * float(rates["usd_per_minute"])


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Cash totals for the caps plus informational credit usage.

    ``spent_usd``/``reserved_usd``/``remaining_usd``/``phase_*`` cover cash
    requests only; Deepgram credit usage is reported separately.
    """

    spent_usd: float
    reserved_usd: float
    remaining_usd: float
    phase_spent: dict[str, float]
    phase_reserved: dict[str, float]
    entries: tuple[dict[str, Any], ...]
    credit_usd: float = 0.0
    credit_entries: int = 0


@dataclass(frozen=True, slots=True)
class _LedgerTotals:
    spent_usd: float
    reserved_usd: float
    phase_spent: dict[str, float]
    phase_reserved: dict[str, float]
    credit_usd: float
    credit_entries: int


class BudgetLedger:
    def __init__(self, path: str | Path, *, cap_usd: float = _TOTAL_CAP) -> None:
        self.path = Path(path)
        self.cap_usd = cap_usd
        self._thread_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_unlocked(
                {
                    "cap_usd": cap_usd,
                    "phase_caps_usd": dict(_PHASE_CAPS),
                    "entries": [],
                }
            )

    def reserve(
        self,
        request_id: str,
        *,
        phase: Phase,
        amount_usd: float,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if amount_usd < 0:
            raise BudgetError("reserve amount must be non-negative")
        if phase not in _PHASE_CAPS:
            raise BudgetError(f"unknown budget phase: {phase}")
        meta = dict(meta or {})
        credit = _is_credit_kind(meta.get("kind"))
        with self._locked_state() as state:
            existing = self._entry(state, request_id)
            if existing is not None and existing["state"] == "reserved":
                raise BudgetError(f"request already reserved: {request_id}")
            if not credit:
                totals = self._totals(state)
                phase_cap = float(state["phase_caps_usd"][phase])
                if (
                    totals.phase_spent[phase] + totals.phase_reserved[phase] + amount_usd
                    > phase_cap + 1e-12
                ):
                    raise BudgetError(f"{phase} phase cap would be exceeded")
                if (
                    totals.spent_usd + totals.reserved_usd + amount_usd
                    > float(state["cap_usd"]) + 1e-12
                ):
                    raise BudgetError("combined cap would be exceeded")
            entry = {
                "id": request_id,
                "phase": phase,
                "state": "reserved",
                "reserved_usd": amount_usd,
                "settled_usd": None,
                "meta": meta,
                "ts": time.time(),
            }
            state["entries"].append(entry)
            self._write_unlocked(state)
            return dict(entry)

    def settle(
        self,
        request_id: str,
        *,
        billed_usd: float | None = None,
        keep_reserve: bool = False,
    ) -> dict[str, Any]:
        with self._locked_state() as state:
            entry = self._entry(state, request_id)
            if entry is None or entry["state"] != "reserved":
                raise BudgetError(f"no reserved request: {request_id}")
            if billed_usd is None or keep_reserve:
                entry["state"] = "kept"
                entry["settled_usd"] = None
            else:
                if billed_usd < 0:
                    raise BudgetError("billed amount must be non-negative")
                if billed_usd > float(entry["reserved_usd"]) + 1e-12:
                    raise BudgetError("billed amount exceeds reserve")
                entry["state"] = "settled"
                entry["settled_usd"] = billed_usd
            self._write_unlocked(state)
            return dict(entry)

    def snapshot(self) -> BudgetSnapshot:
        with self._locked_state() as state:
            totals = self._totals(state)
            return BudgetSnapshot(
                spent_usd=totals.spent_usd,
                reserved_usd=totals.reserved_usd,
                remaining_usd=float(state["cap_usd"]) - totals.spent_usd - totals.reserved_usd,
                phase_spent=totals.phase_spent,
                phase_reserved=totals.phase_reserved,
                entries=tuple(dict(item) for item in state["entries"]),
                credit_usd=totals.credit_usd,
                credit_entries=totals.credit_entries,
            )

    def _entry(self, state: dict[str, Any], request_id: str) -> dict[str, Any] | None:
        for item in reversed(state["entries"]):
            if item["id"] == request_id:
                return item
        return None

    def _totals(self, state: dict[str, Any]) -> _LedgerTotals:
        spent = 0.0
        reserved = 0.0
        credit = 0.0
        credit_entries = 0
        phase_spent = {name: 0.0 for name in _PHASE_CAPS}
        phase_reserved = {name: 0.0 for name in _PHASE_CAPS}
        for item in state["entries"]:
            phase = item["phase"]
            is_credit = _is_credit_kind((item.get("meta") or {}).get("kind"))
            if item["state"] == "reserved" or item["state"] == "kept":
                amount = float(item["reserved_usd"])
                if is_credit:
                    credit += amount
                    credit_entries += 1
                else:
                    reserved += amount
                    phase_reserved[phase] += amount
            elif item["state"] == "settled":
                amount = float(item["settled_usd"])
                if is_credit:
                    credit += amount
                    credit_entries += 1
                else:
                    spent += amount
                    phase_spent[phase] += amount
        return _LedgerTotals(
            spent_usd=spent,
            reserved_usd=reserved,
            phase_spent=phase_spent,
            phase_reserved=phase_reserved,
            credit_usd=credit,
            credit_entries=credit_entries,
        )

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        payload = json.dumps(state, indent=1)
        fd, tmp = tempfile.mkstemp(prefix="budget-", suffix=".json", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _locked_state(self):
        return _LockedState(self)


class _LockedState:
    def __init__(self, ledger: BudgetLedger) -> None:
        self._ledger = ledger
        self._handle: Any = None
        self._state: dict[str, Any] | None = None

    def __enter__(self) -> dict[str, Any]:
        self._ledger._thread_lock.acquire()
        lock_path = self._ledger.path.with_suffix(".lock")
        self._handle = open(lock_path, "a+b")
        _lock_file(self._handle)
        if self._ledger.path.exists():
            self._state = json.loads(self._ledger.path.read_text(encoding="utf-8"))
        else:
            self._state = {
                "cap_usd": self._ledger.cap_usd,
                "phase_caps_usd": dict(_PHASE_CAPS),
                "entries": [],
            }
        return self._state

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._handle is not None:
                _unlock_file(self._handle)
                self._handle.close()
        finally:
            self._ledger._thread_lock.release()


__all__ = [
    "BillingBoundError",
    "BudgetError",
    "BudgetLedger",
    "BudgetSnapshot",
    "load_billing_bounds",
    "load_rates",
    "openrouter_reserve_usd",
    "deepgram_reserve_usd",
]
