from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


P0_PASS = "P0-PASS"
P0_FAIL = "P0-FAIL"
H_SCREEN = "H-SCREEN"
H_CONFIRM = "H-CONFIRM"
H_ACCEPT = "H-ACCEPT"
H_STOP = "H-STOP"
OPEN_T2 = "OPEN-T2"
T2_SCREEN = "T2-SCREEN"
T2_CONFIRM = "T2-CONFIRM"
T2_ACCEPT = "T2-ACCEPT"
T2_STOP = "T2-STOP"
OPEN_TA = "OPEN-TA"
TA_SCREEN = "TA-SCREEN"
TA_CONFIRM = "TA-CONFIRM"
TA_ACCEPT = "TA-ACCEPT"
TA_STOP = "TA-STOP"
CANDIDATE_FROZEN = "CANDIDATE-FROZEN"
EVAL_OPEN = "EVAL-OPEN"
FINAL_F0 = "FINAL-F0"
FINAL_R_H_SC = "FINAL-R-H-SC"
FINAL_R_T2_SC = "FINAL-R-T2-SC"
FINAL_R_TA_SC = "FINAL-R-TA-SC"
FINAL_STOP = "FINAL-STOP"

ALLOWED_NEXT: dict[str, tuple[str, ...]] = {
    P0_PASS: (H_SCREEN,),
    P0_FAIL: (P0_FAIL,),
    H_SCREEN: (H_CONFIRM, H_STOP, OPEN_T2),
    H_CONFIRM: (H_ACCEPT, H_STOP),
    H_ACCEPT: (CANDIDATE_FROZEN,),
    H_STOP: (FINAL_F0, FINAL_STOP),
    OPEN_T2: (T2_SCREEN,),
    T2_SCREEN: (T2_CONFIRM, T2_STOP),
    T2_CONFIRM: (T2_ACCEPT, T2_STOP),
    T2_ACCEPT: (CANDIDATE_FROZEN,),
    T2_STOP: (FINAL_F0, FINAL_STOP, FINAL_R_H_SC),
    OPEN_TA: (TA_SCREEN,),
    TA_SCREEN: (TA_CONFIRM, TA_STOP),
    TA_CONFIRM: (TA_ACCEPT, TA_STOP),
    TA_ACCEPT: (CANDIDATE_FROZEN,),
    TA_STOP: (FINAL_F0, FINAL_STOP, FINAL_R_H_SC, FINAL_R_T2_SC),
    CANDIDATE_FROZEN: (EVAL_OPEN,),
    EVAL_OPEN: (FINAL_F0, FINAL_R_H_SC, FINAL_R_T2_SC, FINAL_R_TA_SC, FINAL_STOP),
}


class GateError(RuntimeError):
    pass


@dataclass(slots=True)
class GateTracker:
    branches: dict[str, str] = field(default_factory=dict)

    def current(self, branch: str) -> str | None:
        return self.branches.get(branch)

    def advance(self, branch: str, gate: str) -> dict[str, object]:
        if gate not in ALLOWED_NEXT and gate not in (
            FINAL_F0, FINAL_R_H_SC, FINAL_R_T2_SC, FINAL_R_TA_SC, FINAL_STOP,
        ):
            raise GateError(f"unknown gate: {gate}")
        prior = self.branches.get(branch)
        if prior is None:
            if gate not in (P0_PASS, P0_FAIL):
                raise GateError("branch must open with P0-PASS or P0-FAIL")
            self.branches[branch] = gate
            return {"branch": branch, "prior": None, "gate": gate, "edge": True}
        if gate == prior:
            return {"branch": branch, "prior": prior, "gate": gate, "edge": False}
        allowed = ALLOWED_NEXT.get(prior, ())
        if gate not in allowed:
            raise GateError(f"illegal gate transition: {prior} -> {gate}")
        self.branches[branch] = gate
        return {"branch": branch, "prior": prior, "gate": gate, "edge": True}


def check_p0_receipt(
    record: Any,
    input_hash: str,
    checkpoint_hash: str,
    partition_hash: str,
) -> dict[str, object]:
    if not isinstance(record, dict):
        raise GateError("P0 receipt is missing")
    if record.get("verdict") != "PASS":
        raise GateError("P0 receipt is not PASS")
    for key, expected in (
        ("input_hash", input_hash),
        ("checkpoint_hash", checkpoint_hash),
        ("partition_hash", partition_hash),
    ):
        if record.get(key) != expected:
            raise GateError(f"P0 receipt {key} mismatch")
    return {"gate": P0_PASS, "edge": True}


def check_gate1_receipt(record: Any, input_hash: str) -> dict[str, object]:
    if not isinstance(record, dict):
        raise GateError("Gate-1 receipt is missing")
    if record.get("decision") != OPEN_T2:
        raise GateError("Gate-1 receipt does not open T2")
    if not record.get("h_candidate_hash"):
        raise GateError("Gate-1 receipt lacks H candidate hash")
    if record.get("input_hash") != input_hash:
        raise GateError("Gate-1 receipt input hash mismatch")
    return {"gate": OPEN_T2, "edge": True}


def check_gate2_receipt(record: Any, input_hash: str) -> dict[str, object]:
    if not isinstance(record, dict):
        raise GateError("Gate-2 receipt is missing")
    if record.get("decision") != OPEN_TA:
        raise GateError("Gate-2 receipt does not open TA")
    if not record.get("t2_candidate_hash"):
        raise GateError("Gate-2 receipt lacks T2 candidate hash")
    if record.get("input_hash") != input_hash:
        raise GateError("Gate-2 receipt input hash mismatch")
    return {"gate": OPEN_TA, "edge": True}


def check_confirmation_receipt(record: Any, arm: str, input_hash: str) -> dict[str, object]:
    if not isinstance(record, dict):
        raise GateError("confirmation receipt is missing")
    if record.get("arm") != arm:
        raise GateError("confirmation receipt arm mismatch")
    if record.get("seed") != 7302:
        raise GateError("confirmation receipt seed mismatch")
    if not record.get("candidate_hash"):
        raise GateError("confirmation receipt lacks candidate hash")
    if record.get("input_hash") != input_hash:
        raise GateError("confirmation receipt input hash mismatch")
    return {"gate": "CONFIRM-SEED", "edge": True}
