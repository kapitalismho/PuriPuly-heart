from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


NEMO_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
SCREEN_SEED = 7301
CONFIRM_SEED = 7302


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    body = {**payload, "payload_sha256": canonical_sha256(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return path


def experiment_manifest(
    issue: int = 121,
    seed: int = SCREEN_SEED,
    arms: tuple[str, ...] = ("F0-CONTINUOUS", "R-H-SC"),
) -> dict[str, Any]:
    return {
        "artifact_role": "issue-121-experiment-manifest",
        "issue": issue,
        "screen_seed": seed,
        "confirm_seed": CONFIRM_SEED,
        "arms": list(arms),
        "nemo_sha256": NEMO_SHA256,
        "optimizer": {"name": "AdamW", "microbatch": 1, "accumulation": 16, "effective": 16},
    }


def sampling_calibration_manifest(
    fit: list[str], calib: list[str], salt: str, target_frac: float
) -> dict[str, Any]:
    return {
        "artifact_role": "issue-121-sampling-calibration-manifest",
        "train_fit": sorted(fit),
        "train_calib": sorted(calib),
        "salt": salt,
        "target_frac": target_frac,
        "gradients": {"train_fit": True, "train_calib": False},
    }


def module_mode_receipt(backbone_eval: bool, head_train: bool, arm: str) -> dict[str, Any]:
    return {
        "artifact_role": "issue-121-module-mode-receipt",
        "arm": arm,
        "sortformer_eval": backbone_eval,
        "psem_head_train": head_train,
        "frozen_representation_ok": bool(backbone_eval and head_train),
    }


def material_vertical_slice_record(
    ami_source: str,
    alimeeting_source: str,
    checks: dict[str, bool],
    mode: str,
) -> dict[str, Any]:
    verdict = "PASS" if all(checks.values()) else "FAIL"
    return {
        "artifact_role": "issue-121-material-vertical-slice",
        "ami_source": ami_source,
        "alimeeting_source": alimeeting_source,
        "checks": dict(checks),
        "verdict": verdict,
        "mode": mode,
        "nemo_sha256": NEMO_SHA256,
    }


def p0_pass_receipt(
    input_hash: str,
    checkpoint_hash: str,
    partition_hash: str,
    ami_source: str,
    alimeeting_source: str,
) -> dict[str, Any]:
    return {
        "artifact_role": "issue-121-p0-pass-receipt",
        "verdict": "PASS",
        "input_hash": input_hash,
        "checkpoint_hash": checkpoint_hash,
        "partition_hash": partition_hash,
        "ami_source": ami_source,
        "alimeeting_source": alimeeting_source,
        "nemo_sha256": NEMO_SHA256,
    }


def confirmation_receipt(arm: str, input_hash: str, candidate_hash: str = "") -> dict[str, Any]:
    return {
        "artifact_role": "issue-121-confirmation-authorization",
        "arm": arm,
        "seed": CONFIRM_SEED,
        "input_hash": input_hash,
        "candidate_hash": candidate_hash,
    }


def gate1_receipt(h_candidate_hash: str, input_hash: str) -> dict[str, Any]:
    return {
        "artifact_role": "issue-121-gate1-authorization",
        "decision": "OPEN-T2",
        "h_candidate_hash": h_candidate_hash,
        "input_hash": input_hash,
    }


def gate2_receipt(t2_candidate_hash: str, input_hash: str) -> dict[str, Any]:
    return {
        "artifact_role": "issue-121-gate2-authorization",
        "decision": "OPEN-TA",
        "t2_candidate_hash": t2_candidate_hash,
        "input_hash": input_hash,
    }
