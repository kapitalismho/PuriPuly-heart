from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.psem_state_corrected_adaptation_gate import arm_runtime
from experiments.psem_state_corrected_adaptation_gate import calibrate as calibrate_mod
from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod
from experiments.psem_state_corrected_adaptation_gate import streaming as streaming_mod
from experiments.psem_state_corrected_adaptation_gate.arm_runtime import (
    ArmError,
    AuthorizationError,
)


class HArmError(ArmError):
    pass


H_ARM = arm_runtime.ARM_R_H_SC
HEAD_LR = 1e-3
HEAD_WD = 1e-4
HEAD_CLIP_NORM = 1.0
ANCHOR_LOSS_WEIGHT = 0.5
CHUNK_FRAMES = streaming_mod.CHUNK_FRAMES
CACHE_DIRNAME = "cache"
CACHE_MANIFEST_NAME = "cache_manifest.json"
H_TARGETS_DIRNAME = "h_targets"
PROFILE_RECEIPT_NAME = "profile_receipt.json"
MIN_PROFILE_STEPS = 8
MAX_PROFILE_STEPS = 16
EXPERIMENT_MANIFEST_NAME = "experiment_manifest.json"
SAMPLING_MANIFEST_NAME = "data_sampling_calibration_manifest.json"
MODE_RECEIPT_NAME = "parameter_module_mode_receipt.json"
TRAINING_METRICS_NAME = "training_metrics.json"
CALIBRATION_METRICS_NAME = "calibration_metrics.json"
DEV_FRONTIER_NAME = "dev_frontier.json"
GPU_EXPORT_DIRNAME = "gpu_export"
GPU_EXPORT_MANIFEST_NAME = "gpu_export_manifest.json"
GPU_EXPORT_PROGRESS_NAME = "gpu_export_progress.json"
GPU_EXPORT_ARTIFACT_ROLE = "issue-121-h-gpu-export"
MAPPING_DIAGNOSTICS_DIRNAME = "diagnostics"


def require_h_arm(config: arm_runtime.ArmRunConfig) -> arm_runtime.ArmRunConfig:
    if config.arm != H_ARM:
        raise AuthorizationError(f"h_arm serves R-H-SC only: {config.arm}")
    return config


def check_authorized(config: arm_runtime.ArmRunConfig, store: Path) -> dict[str, Any]:
    require_h_arm(config)
    return arm_runtime.check_authorization(config, Path(store))


def partition_hash_for(manifest: dict[str, Any]) -> str:
    fit, calib = fit_calib_from_bundle(manifest)
    return arm_runtime.canonical_sha256(
        {
            "fit": fit,
            "calib": calib,
            "salt": str(manifest.get("salt", "")),
            "target_frac": float(manifest.get("target_frac", 0.0)),
        }
    )


H_IDENTITY_FILES = (
    "h_arm.py",
    "run_h_arm.py",
    "arm_runtime.py",
    "head.py",
    "streaming.py",
    "calibrate.py",
    "frontier.py",
    "cross_frontier.py",
    "partition.py",
    "lifecycle.py",
    "multiplicity.py",
    "material.py",
    "stages.py",
)


def h_code_digest(root: Path | str | None = None) -> str:
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in H_IDENTITY_FILES:
        digest.update((base / name).read_bytes())
    return digest.hexdigest()

def verify_config_binding(
    config: arm_runtime.ArmRunConfig, manifest: dict[str, Any], inputs: HArmPodInputs
) -> dict[str, Any]:
    try:
        expected_code = h_code_digest()
    except OSError as exc:
        raise AuthorizationError(f"H code identity is missing: {exc}") from exc
    if config.code_hash != expected_code:
        raise AuthorizationError("H code binding mismatch")
    try:
        expected_partition = partition_hash_for(manifest)
        _, expected_weights = arm_runtime.bind_class_weights(dict(manifest.get("class_weights", {})))
    except (ArmError, KeyError, TypeError, ValueError) as exc:
        raise AuthorizationError(f"H binding is invalid: {exc}") from exc
    if config.partition_hash != expected_partition:
        raise AuthorizationError("H partition binding mismatch")
    if config.weights_hash != expected_weights:
        raise AuthorizationError("H class-weights binding mismatch")
    try:
        manifest_bytes = (Path(inputs.bundle_dir) / "stage_a_manifest.json").read_bytes()
    except OSError as exc:
        raise AuthorizationError(f"H input manifest is missing: {exc}") from exc
    if hashlib.sha256(manifest_bytes).hexdigest() != config.input_hash:
        raise AuthorizationError("H input binding mismatch")
    try:
        checkpoint_bytes = Path(inputs.checkpoint).read_bytes()
    except OSError as exc:
        raise AuthorizationError(f"H checkpoint is missing: {exc}") from exc
    if hashlib.sha256(checkpoint_bytes).hexdigest() != config.checkpoint_hash:
        raise AuthorizationError("H checkpoint binding mismatch")
    return {
        "partition_hash": expected_partition,
        "weights_hash": expected_weights,
        "input_hash": config.input_hash,
        "checkpoint_hash": config.checkpoint_hash,
    }


def forbid_eval(source_ids: list[str], roles: dict[str, str] | None = None) -> None:
    for source_id in source_ids:
        if "eval" in str(source_id).lower():
            raise HArmError(f"EVAL access is forbidden: {source_id}")
    for source_id, role in dict(roles or {}).items():
        if str(role).upper() == "EVAL":
            raise HArmError(f"EVAL access is forbidden: {source_id}")


def assemble_features(
    hidden: Any, slot_logits: Any, selected: Any, max_nonanchor: Any, delay: Any
) -> Any:
    import numpy as np

    parts = [
        np.asarray(hidden, dtype=np.float32),
        np.asarray(slot_logits, dtype=np.float32),
        np.asarray(selected, dtype=np.float32).reshape((-1, 1)),
        np.asarray(max_nonanchor, dtype=np.float32).reshape((-1, 1)),
        np.asarray(delay, dtype=np.float32).reshape((-1, 1)),
    ]
    expected = dict(arm_runtime.HEAD_INPUT_PARTS)
    shapes = [p.shape for p in parts]
    if not (
        parts[0].ndim == 2
        and parts[0].shape[1] == expected["hidden"]
        and parts[1].ndim == 2
        and parts[1].shape[1] == expected["slot_logits"]
    ):
        raise HArmError(f"evidence geometry differs: {[tuple(s) for s in shapes]}")
    frames = parts[0].shape[0]
    for part in parts[1:]:
        if part.shape[0] != frames:
            raise HArmError("evidence/target frame counts differ")
    features = np.concatenate(parts, axis=1)
    arm_runtime.check_head_input_dim(int(features.shape[1]))
    return features


FROZEN_EVIDENCE_DIRNAME = "frozen_evidence"


def frozen_evidence_dir(root: Path) -> Path:
    return Path(root) / FROZEN_EVIDENCE_DIRNAME


def frozen_evidence_identity(
    binding: dict[str, Any], payload: Mapping[str, Any], source_id: str = ""
) -> dict[str, Any]:
    return {
        "source_id": str(source_id or dict(payload).get("source_id", "")),
        "checkpoint_hash": str(dict(binding).get("checkpoint_hash", "")),
        "input_hash": str(dict(binding).get("input_hash", "")),
        "code_hash": str(dict(binding).get("code_hash", "")),
        "waveform_sha256": str(dict(payload).get("waveform_sha256", "")),
        "audio_ref": str(dict(payload).get("audio_ref", "")),
        "num_frames": int(dict(payload).get("num_frames", 0)),
    }


def frozen_evidence_key(identity: dict[str, Any]) -> str:
    return arm_runtime.canonical_sha256(dict(identity))


def write_frozen_evidence(
    root: Path,
    identity: dict[str, Any],
    hidden: Any,
    slot_logits: Any,
    slot_of: dict[str, int],
    mapping_rows: list[dict[str, Any]],
    timing: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    base = frozen_evidence_dir(root)
    base.mkdir(parents=True, exist_ok=True)
    key = frozen_evidence_key(dict(identity))
    hidden_arr = np.asarray(hidden, dtype=np.float32)
    logits_arr = np.asarray(slot_logits, dtype=np.float32)
    target = base / f"{key}.npz"
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, hidden192=hidden_arr, slot_logits4=logits_arr)
    os.replace(tmp, target)
    digest = arm_runtime.sha256_file(target)
    meta = {
        "artifact_role": "issue-121-h-frozen-evidence",
        "key": key,
        "identity": dict(identity),
        "sha256": digest,
        "frames": int(hidden_arr.shape[0]),
        "slot_of": {str(k): int(v) for k, v in dict(slot_of).items()},
        "mapping_rows": [dict(row) for row in mapping_rows],
        "timing": dict(timing),
    }
    arm_runtime.atomic_write_json(base / f"{key}.json", meta)
    return meta


def read_frozen_evidence(root: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    import numpy as np

    base = frozen_evidence_dir(root)
    key = frozen_evidence_key(dict(identity))
    meta_path = base / f"{key}.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HArmError(f"frozen evidence is unreadable: {key}") from exc
    if not isinstance(meta, dict) or meta.get("identity") != dict(identity):
        raise HArmError(f"frozen evidence identity mismatch: {key}")
    npz_path = base / f"{key}.npz"
    if not npz_path.is_file():
        raise HArmError(f"frozen evidence payload is missing: {key}")
    if arm_runtime.sha256_file(npz_path) != meta.get("sha256"):
        raise HArmError(f"frozen evidence hash mismatch: {key}")
    with open(npz_path, "rb") as handle:
        payload = np.load(handle, allow_pickle=False)
        hidden = np.array(payload["hidden192"])
        logits = np.array(payload["slot_logits4"])
    if hidden.shape[0] != int(meta.get("frames", -1)):
        raise HArmError(f"frozen evidence frame counts differ: {key}")
    return {"hidden192": hidden, "slot_logits4": logits, "meta": meta}


def frozen_evidence_paths(root: Path, identity: dict[str, Any]) -> tuple[Path, Path, str]:
    base = frozen_evidence_dir(Path(root))
    key = frozen_evidence_key(dict(identity))
    return base / f"{key}.npz", base / f"{key}.json", key


def frozen_hit_meta(root: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    npz_path, meta_path, key = frozen_evidence_paths(Path(root), dict(identity))
    if not meta_path.is_file() or not npz_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HArmError(f"frozen evidence is unreadable: {key}") from exc
    if not isinstance(meta, dict) or meta.get("identity") != dict(identity):
        raise HArmError(f"frozen evidence identity mismatch: {key}")
    if arm_runtime.sha256_file(npz_path) != meta.get("sha256"):
        raise HArmError(f"frozen evidence hash mismatch: {key}")
    return meta


def atomic_copy_file(src: Path, dst: Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def reuse_frozen_to_cache(
    run_dir: Path,
    source_id: str,
    frozen_npz: Path,
    frozen_meta: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    target = cache_npz_path(run_dir, str(source_id))
    atomic_copy_file(Path(frozen_npz), target)
    digest = arm_runtime.sha256_file(target)
    if digest != frozen_meta.get("sha256"):
        raise HArmError(f"evidence cache copy differs: {source_id}")
    meta = {
        "artifact_role": "issue-121-h-evidence-cache",
        "source_id": str(source_id),
        "file": f"{CACHE_DIRNAME}/{source_id}.npz",
        "sha256": digest,
        "frames": int(frozen_meta.get("frames", -1)),
        "hidden_dim": 192,
        "slot_dim": 4,
        "slot_of": {str(k): int(v) for k, v in dict(frozen_meta.get("slot_of", {})).items()},
        "mapping_rows": [dict(row) for row in frozen_meta.get("mapping_rows", [])],
        "timing": dict(frozen_meta.get("timing", {})),
        "binding": cache_identity(dict(binding)),
    }
    arm_runtime.atomic_write_json(cache_meta_path(run_dir, str(source_id)), meta)
    return meta


def verify_source_cache_file(
    run_dir: Path, source_id: str, expected_binding: dict[str, Any]
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    meta_path = cache_meta_path(run_dir, str(source_id))
    if not meta_path.is_file():
        raise HArmError(f"evidence cache is missing: {source_id}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HArmError(f"evidence cache is unreadable: {source_id}") from exc
    if not isinstance(meta, dict) or meta.get("source_id") != str(source_id):
        raise HArmError(f"evidence cache is invalid: {source_id}")
    if meta.get("binding") != cache_identity(dict(expected_binding)):
        raise HArmError(f"evidence cache binding mismatch: {source_id}")
    npz_path = cache_npz_path(run_dir, str(source_id))
    if not npz_path.is_file():
        raise HArmError(f"evidence cache payload is missing: {source_id}")
    if arm_runtime.sha256_file(npz_path) != meta.get("sha256"):
        raise HArmError(f"evidence cache hash mismatch: {source_id}")
    return meta


def cache_unmapped_frames(run_dir: Path, source_id: str) -> Any:
    try:
        meta = json.loads(cache_meta_path(Path(run_dir), str(source_id)).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HArmError(f"evidence cache is unreadable: {source_id}") from exc
    if not isinstance(meta, dict):
        raise HArmError(f"evidence cache is invalid: {source_id}")
    return meta.get("timing", {}).get("unmapped_frames", ())



def cache_npz_path(run_dir: Path, source_id: str) -> Path:
    return Path(run_dir) / CACHE_DIRNAME / f"{source_id}.npz"


def cache_meta_path(run_dir: Path, source_id: str) -> Path:
    return Path(run_dir) / CACHE_DIRNAME / f"{source_id}.json"


def write_source_cache(
    run_dir: Path,
    source_id: str,
    hidden: Any,
    slot_logits: Any,
    slot_of: dict[str, int],
    mapping_rows: list[dict[str, Any]],
    timing: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    run_dir = Path(run_dir)
    hidden_arr = np.asarray(hidden, dtype=np.float32)
    logits_arr = np.asarray(slot_logits, dtype=np.float32)
    if hidden_arr.ndim != 2 or hidden_arr.shape[1] != 192:
        raise HArmError(f"cached hidden geometry differs: {hidden_arr.shape}")
    if logits_arr.ndim != 2 or logits_arr.shape[1] != 4 or logits_arr.shape[0] != hidden_arr.shape[0]:
        raise HArmError(f"cached slot-logit geometry differs: {logits_arr.shape}")
    target = cache_npz_path(run_dir, source_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, hidden192=hidden_arr, slot_logits4=logits_arr)
    os.replace(tmp, target)
    digest = arm_runtime.sha256_file(target)
    meta = {
        "artifact_role": "issue-121-h-evidence-cache",
        "source_id": source_id,
        "file": f"{CACHE_DIRNAME}/{source_id}.npz",
        "sha256": digest,
        "frames": int(hidden_arr.shape[0]),
        "hidden_dim": 192,
        "slot_dim": 4,
        "slot_of": {str(k): int(v) for k, v in dict(slot_of).items()},
        "mapping_rows": [dict(row) for row in mapping_rows],
        "timing": dict(timing),
        "binding": cache_identity(dict(binding)),
    }
    arm_runtime.atomic_write_json(cache_meta_path(run_dir, source_id), meta)
    return meta


def cache_identity(binding: dict[str, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in dict(binding).items() if str(k) != "seed"}


def read_cache_table(
    run_dir: Path, source_ids: list[str], expected_binding: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for source_id in list(source_ids):
        table[str(source_id)] = read_source_cache(Path(run_dir), str(source_id), dict(expected_binding))
    return table


def read_source_cache(
    run_dir: Path, source_id: str, expected_binding: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    run_dir = Path(run_dir)
    meta_path = cache_meta_path(run_dir, source_id)
    if not meta_path.is_file():
        raise HArmError(f"evidence cache is missing: {source_id}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HArmError(f"evidence cache is unreadable: {source_id}") from exc
    if not isinstance(meta, dict) or meta.get("source_id") != source_id:
        raise HArmError(f"evidence cache is invalid: {source_id}")
    if meta.get("binding") != cache_identity(dict(expected_binding)):
        raise HArmError(f"evidence cache binding mismatch: {source_id}")
    npz_path = cache_npz_path(run_dir, source_id)
    if not npz_path.is_file():
        raise HArmError(f"evidence cache payload is missing: {source_id}")
    if arm_runtime.sha256_file(npz_path) != meta.get("sha256"):
        raise HArmError(f"evidence cache hash mismatch: {source_id}")
    with open(npz_path, "rb") as handle:
        payload = np.load(handle, allow_pickle=False)
        hidden = np.array(payload["hidden192"])
        logits = np.array(payload["slot_logits4"])
    if hidden.shape[0] != int(meta.get("frames", -1)):
        raise HArmError(f"evidence cache frame counts differ: {source_id}")
    return {"hidden192": hidden, "slot_logits4": logits, "meta": meta}


def write_cache_manifest(
    run_dir: Path, records: dict[str, dict[str, Any]], binding: dict[str, Any]
) -> Path:
    body = {
        "artifact_role": "issue-121-h-cache-manifest",
        "binding": cache_identity(dict(binding)),
        "sources": {str(k): dict(v) for k, v in dict(records).items()},
    }
    return arm_runtime.atomic_write_json(Path(run_dir) / CACHE_MANIFEST_NAME, body)


def require_cache_coverage(
    run_dir: Path, fit: list[str], calib: list[str], expected_binding: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = Path(run_dir) / CACHE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise HArmError("evidence cache manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("binding") != cache_identity(dict(expected_binding)):
        raise HArmError("evidence cache manifest binding mismatch")
    needed = sorted(set(fit) | set(calib))
    sources = manifest.get("sources", {})
    missing = [s for s in needed if s not in sources]
    if missing:
        raise HArmError(f"evidence cache omits sources: {missing}")
    for source_id in needed:
        verify_source_cache_file(run_dir, source_id, expected_binding)
    return manifest


def load_valid_cache(
    run_dir: Path, source_id: str, expected_binding: dict[str, Any], frames: int
) -> dict[str, Any] | None:
    run_dir = Path(run_dir)
    if not cache_meta_path(run_dir, source_id).is_file() or not cache_npz_path(run_dir, source_id).is_file():
        return None
    cached = read_source_cache(run_dir, source_id, dict(expected_binding))
    hidden = cached["hidden192"]
    logits = cached["slot_logits4"]
    if hidden.ndim != 2 or hidden.shape[1] != 192 or logits.ndim != 2 or logits.shape[1] != 4:
        raise HArmError(f"evidence cache geometry differs: {source_id}")
    if hidden.shape[0] != int(frames) or logits.shape[0] != int(frames):
        raise HArmError(f"evidence cache frame counts differ: {source_id}")
    return cached


def fit_calib_from_bundle(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    fit = manifest.get("fit")
    calib = manifest.get("calib")
    if not isinstance(fit, list) or not fit or not all(isinstance(s, str) for s in fit):
        raise HArmError("stage-A bundle FIT list is invalid")
    if not isinstance(calib, list) or not calib or not all(isinstance(s, str) for s in calib):
        raise HArmError("stage-A bundle CALIB list is invalid")
    return sorted(fit), sorted(calib)


def h_binding(config: arm_runtime.ArmRunConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    bound, _ = arm_runtime.bind_class_weights(dict(manifest.get("class_weights", {})))
    return {
        **dict(config.binding),
        "sampling_sha256": str(manifest.get("sampling_sha256", "")),
        "class_weights": bound,
        "fit": sorted(manifest.get("fit", [])),
        "calib": sorted(manifest.get("calib", [])),
        "salt": str(manifest.get("salt", "")),
        "target_frac": float(manifest.get("target_frac", 0.0)),
    }


def check_target_geometry(payload: dict[str, Any], source_id: str) -> dict[str, Any]:
    try:
        frames = int(payload.get("num_frames", -1))
    except (TypeError, ValueError) as exc:
        raise HArmError(f"target geometry is invalid: {source_id}") from exc
    arrays = [
        payload.get("y_anchor"),
        payload.get("y_replace"),
        payload.get("valid"),
        payload.get("multiplicity"),
        payload.get("episode_ids"),
    ]
    if frames <= 0 or any(not isinstance(values, list) for values in arrays):
        raise HArmError(f"target geometry is invalid: {source_id}")
    if any(len(values) != frames for values in arrays):
        raise HArmError(f"target geometry differs: {source_id}")
    if payload.get("source_id") != source_id:
        raise HArmError(f"target identity differs: {source_id}")
    return payload


def ensure_full_source_targets(
    bundle_dir: Path,
    manifest: dict[str, Any],
    run_dir: Path,
    builder: Callable[[str], dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    from experiments.psem_state_corrected_adaptation_gate import stages as stages_mod

    fit, calib = fit_calib_from_bundle(manifest)
    needed = sorted(set(fit) | set(calib))
    entries = manifest.get("targets", {})
    present = [s for s in needed if isinstance(entries.get(s), dict)]
    payloads: dict[str, dict[str, Any]] = {}
    if present:
        loaded = stages_mod.load_stage_targets(Path(bundle_dir), manifest, present)
        payloads.update({sid: with_authoritative_corpus(sid, item) for sid, item in loaded.items()})
    missing = [s for s in needed if s not in payloads]
    out_dir = Path(run_dir) / H_TARGETS_DIRNAME
    remaining: list[str] = []
    for source_id in missing:
        cached = out_dir / f"{source_id}.json"
        if not cached.is_file():
            remaining.append(source_id)
            continue
        try:
            loaded_json = json.loads(cached.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HArmError(f"supplemental h_target is unreadable: {source_id}") from exc
        payload = with_authoritative_corpus(source_id, check_target_geometry(dict(loaded_json), source_id))
        if str(payload.get("sampling_sha256", manifest.get("sampling_sha256"))) != str(
            manifest.get("sampling_sha256")
        ):
            raise HArmError(f"backfill sampling identity differs: {source_id}")
        payloads[source_id] = payload

    if remaining and builder is None:
        raise HArmError(f"stage-A bundle omits FIT targets: {remaining}")
    for source_id in remaining:
        payload = check_target_geometry(dict(builder(source_id)), source_id)
        if str(payload.get("sampling_sha256", manifest.get("sampling_sha256"))) != str(
            manifest.get("sampling_sha256")
        ):
            raise HArmError(f"backfill sampling identity differs: {source_id}")
        corpus = str(payload.get("corpus", ""))
        if corpus not in ("AMI", "AliMeeting"):
            raise HArmError(f"backfill corpus is unknown: {source_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        arm_runtime.atomic_write_json(out_dir / f"{source_id}.json", payload)
        payloads[source_id] = payload

    for source_id in needed:
        check_target_geometry(payloads[source_id], source_id)
    return payloads


def _to_float_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        return [float(v) for v in values.detach().to("cpu").reshape(-1).tolist()]
    return [float(v) for v in values]


def target_mult_list(payload: dict[str, Any], unmapped: Any = ()) -> list[float]:
    blocked = set(int(i) for i in (unmapped or ()))
    multiplicity = [float(m) for m in payload.get("multiplicity", [])]
    valid = [bool(v) for v in payload.get("valid", [])]
    if len(multiplicity) != len(valid):
        raise HArmError("target/multiplicity geometry differs")
    return [
        m * (1.0 if v else 0.0) * (0.0 if i in blocked else 1.0)
        for i, (m, v) in enumerate(zip(multiplicity, valid))
    ]


def loss_flags_for_source(num_frames: int, mult_weight: Any) -> list[bool]:
    weights = _to_float_list(mult_weight)
    bounds = streaming_mod.chunk_boundaries(int(num_frames), CHUNK_FRAMES)
    flags: list[bool] = []
    for start, end in bounds:
        flags.append(sum(weights[i] for i in range(start, end)) > 0)
    return flags


def loss_flags_from_mult_list(num_frames: int, mult: Sequence[float]) -> list[bool]:
    weights = [float(v) for v in mult]
    bounds = streaming_mod.chunk_boundaries(int(num_frames), CHUNK_FRAMES)
    return [sum(weights[i] for i in range(start, end)) > 0 for start, end in bounds]


def build_loss_weight_tensors(torch: Any, weights: Mapping[str, float], dtype: Any, device: Any) -> dict[str, Any]:
    return {
        "replacement_positive_weight": torch.as_tensor(
            weights["replacement_positive_weight"], dtype=dtype, device=device
        ),
        "anchor_positive_weight": torch.as_tensor(
            weights["anchor_positive_weight"], dtype=dtype, device=device
        ),
    }


def plan_fit_schedule(
    fit_sources: list[str], flags_by_source: dict[str, list[bool]]
) -> dict[str, Any]:
    return arm_runtime.plan_schedule(list(fit_sources), dict(flags_by_source))


def freeze_backbone_train_head(wrapper: Any, head_module: Any) -> dict[str, Any]:
    wrapper.eval()
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    head_module.train(True)
    wrapper_trainable = [name for name, p in wrapper.named_parameters() if p.requires_grad]
    head_trainable = [name for name, p in head_module.named_parameters() if p.requires_grad]
    if wrapper_trainable:
        raise HArmError("frozen backbone exposes trainable parameters")
    if not head_trainable:
        raise HArmError("residual head exposes no trainable parameters")
    if bool(wrapper.training):
        raise HArmError("frozen wrapper is not in eval mode")
    if not bool(head_module.training):
        raise HArmError("residual head is not in train mode")
    return {
        "sortformer_eval": True,
        "psem_head_train": True,
        "frozen_trainable_count": 0,
        "head_trainable_count": len(head_trainable),
        "frozen_representation_ok": True,
    }


def build_head_optimizer(torch: Any, head_module: Any) -> Any:
    params = [p for p in head_module.parameters() if p.requires_grad]
    if not params:
        raise HArmError("residual head exposes no trainable parameters")
    return torch.optim.AdamW(params, lr=HEAD_LR, weight_decay=HEAD_WD)


def warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (int(step) + 1) / int(warmup_steps))


def build_warmup_scheduler(torch: Any, optimizer: Any, warmup_steps: int) -> Any:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: warmup_factor(int(step), int(warmup_steps))
    )


def serialize_blob(torch: Any, obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def deserialize_blob(torch: Any, data: bytes) -> Any:
    return torch.load(io.BytesIO(bytes(data)), weights_only=False)


def collect_head_state(torch: Any, head_module: Any) -> Any:
    return {k: v.detach().clone() for k, v in head_module.state_dict().items()}


def pending_head_grads(torch: Any, head_module: Any) -> dict[str, Any]:
    grads: dict[str, Any] = {}
    for name, param in head_module.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def checkpoint_after_source(
    torch: Any,
    run_dir: Path,
    source_id: str,
    completed: list[str],
    binding: dict[str, Any],
    head_module: Any,
    optimizer: Any,
    scheduler: Any,
    pending: dict[str, Any] | None = None,
) -> Path:
    counts = dict(pending or {})
    blobs = {
        "model": serialize_blob(
            torch,
            {
                "head_state": collect_head_state(torch, head_module),
                "pending_grads": pending_head_grads(torch, head_module),
                "accum_count": int(counts.get("accum_count", 0)),
                "steps_taken": int(counts.get("steps_taken", 0)),
            },
        ),
        "optimizer": serialize_blob(torch, optimizer.state_dict()),
        "scheduler": serialize_blob(torch, scheduler.state_dict()),
        "rng": serialize_blob(torch, snapshot_rng(torch)),
    }
    return arm_runtime.save_source_checkpoint(
        Path(run_dir), source_id, list(completed), dict(binding), blobs=blobs
    )


def restore_head_state(
    torch: Any,
    run_dir: Path,
    binding: dict[str, Any],
    head_module: Any,
    optimizer: Any,
    scheduler: Any,
    source_order: list[str] | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if record is None:
        record = arm_runtime.load_source_checkpoint(Path(run_dir), dict(binding))
    if record.get("fresh"):
        return {**record, "pending": {"accum_count": 0, "steps_taken": 0}}
    completed = list(record.get("completed_sources", []))
    if not completed:
        return {**record, "pending": {"accum_count": 0, "steps_taken": 0}}
    if source_order:
        in_order = [s for s in source_order if s in set(completed)]
        latest = in_order[-1] if in_order else completed[-1]
    else:
        latest = completed[-1]
    blobs = record["blobs"][latest]
    model_blob = deserialize_blob(torch, Path(blobs["model"]).read_bytes())
    if isinstance(model_blob, dict) and "head_state" in model_blob:
        head_module.load_state_dict(model_blob["head_state"])
        pending_grads = dict(model_blob.get("pending_grads", {}))
        pending = {
            "accum_count": int(model_blob.get("accum_count", 0)),
            "steps_taken": int(model_blob.get("steps_taken", 0)),
        }
    else:
        head_module.load_state_dict(model_blob)
        pending_grads = {}
        pending = {"accum_count": 0, "steps_taken": 0}
    optimizer.load_state_dict(deserialize_blob(torch, Path(blobs["optimizer"]).read_bytes()))
    scheduler.load_state_dict(deserialize_blob(torch, Path(blobs["scheduler"]).read_bytes()))
    rng_blob = deserialize_blob(torch, Path(blobs["rng"]).read_bytes())
    if isinstance(rng_blob, dict) and "torch_cpu" in rng_blob:
        restore_rng(torch, rng_blob)
    else:
        torch.set_rng_state(rng_blob)
    for name, param in head_module.named_parameters():
        grad = pending_grads.get(name)
        param.grad = grad.detach().clone() if grad is not None else None
    return {**record, "pending": pending}


def remaining_fit_sources(fit_sources: list[str], completed: list[str]) -> list[str]:
    return arm_runtime.resume_plan(list(fit_sources), list(completed))


def chunk_loss_value(
    torch: Any,
    product: Any,
    anchor: Any,
    y_replace: Any,
    y_anchor: Any,
    mult: Any,
    weights: dict[str, float],
    cached_weights: dict[str, Any] | None = None,
) -> Any:
    denom = mult.sum().clamp_min(1.0)
    if cached_weights is None:
        cached_weights = build_loss_weight_tensors(torch, weights, product.dtype, product.device)
    replace = torch.nn.functional.binary_cross_entropy_with_logits(
        product,
        y_replace,
        pos_weight=cached_weights["replacement_positive_weight"],
        reduction="none",
    )
    anchor_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        anchor,
        y_anchor,
        pos_weight=cached_weights["anchor_positive_weight"],
        reduction="none",
    )
    return (replace * mult).sum() / denom + ANCHOR_LOSS_WEIGHT * (anchor_loss * mult).sum() / denom


def accumulate_source_loss(torch: Any, loss_terms: list[Any]) -> float:
    if not loss_terms:
        return 0.0
    return float(torch.stack(loss_terms).sum().cpu().item())


def run_fit_pass(
    torch: Any,
    wrapper: Any,
    head_module: Any,
    features_by_source: dict[str, Any],
    targets_by_source: dict[str, dict[str, Any]],
    weights: dict[str, float],
    run_dir: Path,
    binding: dict[str, Any],
    schedule: dict[str, Any],
    completed: list[str] | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    pending: dict[str, Any] | None = None,
    limit_sources: list[str] | None = None,
    feature_loader: Callable[[str], Any] | None = None,
    target_loader: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    done: list[str] = []
    for prior in (completed or []):
        if prior not in done:
            done.append(prior)
    scope = set(limit_sources) if limit_sources is not None else None
    ordered = list(schedule.get("sources", []))
    if not ordered:
        raise HArmError("fit schedule is empty")
    group_sizes: dict[int, int] = {}
    for chunk in schedule.get("chunks", []):
        if chunk.get("contributes"):
            step = int(chunk["optimizer_step"])
            group_sizes[step] = group_sizes.get(step, 0) + 1
    total_steps = int(schedule["total_steps"])
    warmup_steps = int(schedule["warmup_steps"])
    if optimizer is None:
        optimizer = build_head_optimizer(torch, head_module)
    if scheduler is None:
        scheduler = build_warmup_scheduler(torch, optimizer, warmup_steps)
    head_module.train(True)
    incoming = dict(pending or {})
    accum_count = int(incoming.get("accum_count", 0))
    steps_taken = int(incoming.get("steps_taken", 0))
    if accum_count <= 0:
        optimizer.zero_grad()
    chunk_index: dict[tuple[str, int], dict[str, Any]] = {}
    for chunk in schedule.get("chunks", []):
        chunk_index[(str(chunk["source"]), int(chunk["chunk_index"]))] = chunk
    per_source: dict[str, Any] = {}
    trainable_params = [p for p in head_module.parameters() if p.requires_grad]
    for source_id in ordered:
        if source_id in done:
            continue
        if scope is not None and source_id not in scope:
            continue
        if feature_loader is not None:
            features = feature_loader(source_id)
        else:
            features = features_by_source[source_id]
        if source_id in targets_by_source:
            target = targets_by_source[source_id]
        elif target_loader is not None:
            target = target_loader(source_id)
        else:
            raise HArmError(f"fit targets are missing: {source_id}")
        frames = int(target["num_frames"])
        bounds = streaming_mod.chunk_boundaries(frames, CHUNK_FRAMES)
        carrier = streaming_mod.StateCarrier(None, source_id)
        loss_terms: list[Any] = []
        source_chunks = 0
        source_steps = 0
        weight_cache: dict[str, Any] | None = None
        for index, (start, end) in enumerate(bounds):
            piece = features[:, start:end]
            outputs, next_state = head_module(piece, carrier.carry())
            carrier.state = next_state
            entry = chunk_index[(source_id, index)]
            if index < len(bounds) - 1:
                carrier.detach()
            if not entry.get("contributes"):
                continue
            group_size = group_sizes[int(entry["optimizer_step"])]
            product = outputs["product_logit"] if "product_logit" in outputs else (
                target["f0"][:, start:end] + outputs["z_residual"]
            )
            if weight_cache is None:
                weight_cache = build_loss_weight_tensors(torch, weights, product.dtype, product.device)
            loss = chunk_loss_value(
                torch,
                product,
                outputs["anchor_logit"],
                target["y_replace"][:, start:end],
                target["y_anchor"][:, start:end],
                target["mult_weight"][:, start:end],
                weights,
                cached_weights=weight_cache,
            ) / group_size
            loss.backward()
            loss_terms.append(loss.detach())
            source_chunks += 1
            accum_count += 1
            if entry["is_step_boundary"]:
                torch.nn.utils.clip_grad_norm_(trainable_params, HEAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps_taken += 1
                source_steps += 1
                accum_count = 0
        source_loss = accumulate_source_loss(torch, loss_terms)
        del loss_terms
        per_source[source_id] = {
            "chunks": len(bounds),
            "loss_chunks": source_chunks,
            "steps": source_steps,
            "loss_sum": source_loss,
        }
        checkpoint_after_source(
            torch, Path(run_dir), source_id, list(done), dict(binding), head_module, optimizer, scheduler,
            pending={"accum_count": accum_count, "steps_taken": steps_taken},
        )
        if source_id not in done:
            done.append(source_id)
        del features, target, piece, outputs, next_state
    frozen_moved = any(p.requires_grad for p in wrapper.parameters())
    if frozen_moved:
        raise HArmError("frozen backbone exposes trainable parameters")
    return {
        "schedule": schedule,
        "loss_chunks": int(schedule["loss_chunks"]),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "steps_taken": steps_taken,
        "accum_count": accum_count,
        "per_source": per_source,
        "completed_sources": list(done),
    }


def _cuda_peak_bytes(torch: Any, device: Any) -> int:
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated(device))
    except (RuntimeError, AttributeError, ValueError):
        pass
    return 0


def _gpu_utilization() -> float | None:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _cpu_utilization() -> float | None:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=0.1))
    except ImportError:
        pass
    return None


def run_profile(
    run_dir: Path,
    binding: dict[str, Any],
    step_fn: Callable[[], dict[str, Any]],
    dev_infer_fn: Callable[[], dict[str, Any]],
    total_steps: int,
    torch: Any | None = None,
    device: Any | None = None,
    steps: int = MIN_PROFILE_STEPS,
    hourly_cost_usd: float = 0.0,
) -> dict[str, Any]:
    if not (MIN_PROFILE_STEPS <= int(steps) <= MAX_PROFILE_STEPS):
        raise HArmError(f"profile steps must be {MIN_PROFILE_STEPS}-{MAX_PROFILE_STEPS}")
    if torch is not None and device is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
        except (RuntimeError, AttributeError):
            pass
    start = time.perf_counter()
    io_bytes = 0
    for _ in range(int(steps)):
        info = dict(step_fn() or {})
        io_bytes += int(info.get("io_bytes", 0))
    if torch is not None and device is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
        except (RuntimeError, AttributeError):
            pass
    seconds = time.perf_counter() - start
    seconds_per_step = seconds / int(steps)
    dev_info = dict(dev_infer_fn() or {})
    peak_vram = _cuda_peak_bytes(torch, device) if torch is not None else 0
    dev_slice = dict(dev_info.get("frontier_slice") or {})
    dev_seconds = float(dev_info.get("infer_seconds", 0.0))
    dev_io = int(dev_info.get("io_bytes", 0))
    projected_train = seconds_per_step * int(total_steps)
    projected_cost = projected_train / 3600.0 * float(hourly_cost_usd)

    cpu_tail = dict(dev_info.get("cpu_tail") or {})
    if not cpu_tail and dev_slice:
        cpu_tail = {
            "kind": "estimated",
            "representative_slice": dict(dev_slice),
            "seconds": float(dev_slice["seconds"]) if "seconds" in dev_slice else None,
            "projected_seconds": float(dev_slice["projected_seconds"]) if "projected_seconds" in dev_slice else None,
        }
    measured = {
        "optimizer_steps": int(steps),
        "train_seconds": float(seconds),
        "seconds_per_step": seconds_per_step,
        "representative_dev_infer_seconds": dev_seconds,
        "frozen_evidence_seconds": dev_info.get("measured_frozen_evidence_seconds"),
        "frozen_evidence_source_id": dev_info.get("measured_frozen_source_id"),
    }
    estimated = {
        "kind": "estimated",
        "frozen_evidence_seconds": dev_info.get("estimated_frozen_evidence_seconds"),
        "frozen_evidence_population": dict(dev_info.get("frozen_evidence_population") or {}),
        "all_dev_infer_seconds": dev_info.get("estimated_all_dev_infer_seconds"),
        "dev_population": dev_info.get("dev_population"),
        "cpu_tail_seconds": cpu_tail.get("projected_seconds") if cpu_tail else None,
        "cpu_tail": cpu_tail or None,
    }
    receipt = {
        "artifact_role": "issue-121-h-profile-receipt",
        "binding": dict(binding),
        "optimizer_steps": int(steps),
        "seconds_per_step": seconds_per_step,
        "peak_vram_bytes": peak_vram,
        "gpu_utilization": _gpu_utilization(),
        "cpu_utilization": _cpu_utilization(),
        "io_bytes": io_bytes + dev_io,
        "dev_infer_seconds": dev_seconds,
        "dev_frontier_slice": dict(dev_slice),
        "projected_train_seconds": projected_train,
        "projected_cost_usd": projected_cost,
        "hourly_cost_usd": float(hourly_cost_usd),
        "train": {
            "optimizer_steps": int(steps),
            "seconds_per_step": seconds_per_step,
            "total_steps": int(total_steps),
            "projected_train_seconds": projected_train,
            "projected_cost_usd": projected_cost,
            "hourly_cost_usd": float(hourly_cost_usd),
        },
        "dev_inference": {
            "seconds": dev_seconds,
            "io_bytes": dev_io,
            "kind": "measured",
        },
        "measured": measured,
        "estimated": estimated,
        "assembly_projection": {
            "train_io_bytes": io_bytes,
            "dev_io_bytes": dev_io,
            "io_bytes": io_bytes + dev_io,
            "projected_train_seconds": projected_train,
            "projected_cost_usd": projected_cost,
            "kind": "estimated",
        },
    }
    arm_runtime.atomic_write_json(Path(run_dir) / PROFILE_RECEIPT_NAME, receipt)
    return receipt


def require_profile(run_dir: Path, expected_binding: dict[str, Any]) -> dict[str, Any]:
    path = Path(run_dir) / PROFILE_RECEIPT_NAME
    if not path.is_file():
        raise HArmError("profile receipt is missing: run profile before the full pass")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HArmError("profile receipt is unreadable") from exc
    if receipt.get("binding") != dict(expected_binding):
        raise HArmError("profile receipt binding mismatch")
    steps = int(receipt.get("optimizer_steps", 0))
    if not (MIN_PROFILE_STEPS <= steps <= MAX_PROFILE_STEPS):
        raise HArmError("profile receipt step count is outside 8-16")
    if float(receipt.get("dev_infer_seconds", -1.0)) < 0:
        raise HArmError("profile receipt lacks DEV inference timing")
    return receipt


def _frontier_worker(payload: dict[str, Any]) -> dict[str, Any]:
    points = [
        frontier_mod.FrontierPoint(
            threshold=float(t),
            false_cuts_per_hour=float(m[0]),
            contamination=float(m[1]),
            miss_rate=float(m[2]),
        )
        for t, m in zip(payload["thresholds"], payload["metrics"])
    ]
    f0 = payload["f0_metric"]
    reference = frontier_mod.FrontierPoint(
        threshold=frontier_mod.RAW_REFERENCE_THRESHOLD,
        false_cuts_per_hour=float(f0[0]),
        contamination=float(f0[1]),
        miss_rate=float(f0[2]),
    )
    envelopes = frontier_mod.select_envelopes(reference, points)
    return {
        "source_id": payload["source_id"],
        "horizon_ms": payload["horizon_ms"],
        "points": [
            {
                "threshold": p.threshold,
                "false_cuts_per_hour": p.false_cuts_per_hour,
                "contamination": p.contamination,
                "miss_rate": p.miss_rate,
            }
            for p in points
        ],
        "reference": {
            "threshold": reference.threshold,
            "false_cuts_per_hour": reference.false_cuts_per_hour,
            "contamination": reference.contamination,
            "miss_rate": reference.miss_rate,
        },
        "envelopes": {
            "budget": envelopes["budget"],
            "useful": envelopes["useful"],
            "c_envelope": None if envelopes["c_envelope"] is None else {
                "threshold": envelopes["c_envelope"].threshold,
                "false_cuts_per_hour": envelopes["c_envelope"].false_cuts_per_hour,
                "contamination": envelopes["c_envelope"].contamination,
                "miss_rate": envelopes["c_envelope"].miss_rate,
            },
            "m_envelope": None if envelopes["m_envelope"] is None else {
                "threshold": envelopes["m_envelope"].threshold,
                "false_cuts_per_hour": envelopes["m_envelope"].false_cuts_per_hour,
                "contamination": envelopes["m_envelope"].contamination,
                "miss_rate": envelopes["m_envelope"].miss_rate,
            },
        },
    }


def sigmoid_list(values: Any) -> list[float]:
    return [calibrate_mod.sigmoid(float(v)) for v in (values or [])]


def mask_unmapped(values: list[float], mapped: list[bool]) -> list[float]:
    if len(values) != len(mapped):
        raise HArmError("score/mapping geometry differs")
    return [float(v) if m else float("-inf") for v, m in zip(values, mapped)]


def run_calibration_stage(
    run_dir: Path,
    binding: dict[str, Any],
    calib_raw: dict[str, dict[str, list[float]]],
    workers: int | None = None,
) -> dict[str, Any]:
    ordered = sorted(calib_raw)
    if not ordered:
        raise HArmError("TRAIN-CALIB predictions are empty")
    for source_id, entry in calib_raw.items():
        arm_runtime.save_source_predictions(
            Path(run_dir),
            source_id,
            {
                "f0_logit": [float(v) for v in entry["f0"]],
                "candidate_logit": [float(v) for v in entry["candidate"]],
                "f0_prob": sigmoid_list(entry["f0"]),
                "candidate_prob": sigmoid_list(entry["candidate"]),
                "target": [float(v) for v in entry["target"]],
            },
            dict(binding),
        )
        arm_runtime.record_mapping_diagnostics(
            Path(run_dir),
            source_id,
            float(entry.get("coverage", 1.0)),
            float(entry.get("agreement", 1.0)),
            {"role": "TRAIN-CALIB"},
        )
    f0_all: list[float] = []
    cand_all: list[float] = []
    target_all: list[float] = []
    for source_id in ordered:
        entry = calib_raw[source_id]
        f0_all.extend(float(v) for v in entry["f0"])
        cand_all.extend(float(v) for v in entry["candidate"])
        target_all.extend(float(v) for v in entry["target"])
    f0_cal = calibrate_mod.fit_affine_calibrator(f0_all, target_all, "TRAIN-CALIB")
    cand_cal = calibrate_mod.fit_affine_calibrator(cand_all, target_all, "TRAIN-CALIB")
    metrics = {
        "artifact_role": "issue-121-h-calibration-metrics",
        "binding": dict(binding),
        "sources": ordered,
        "frames": len(target_all),
        "f0": {
            **{k: v for k, v in f0_cal.items()},
            "ap": calibrate_mod.average_precision(
                [calibrate_mod.sigmoid(z) for z in f0_all], target_all
            ),
        },
        "candidate": {
            **{k: v for k, v in cand_cal.items()},
            "ap": calibrate_mod.average_precision(
                [calibrate_mod.sigmoid(z) for z in cand_all], target_all
            ),
        },
    }
    arm_runtime.atomic_write_json(Path(run_dir) / CALIBRATION_METRICS_NAME, metrics)
    return metrics

def dev_calibrator_pair(calibration: dict[str, Any], role: str) -> tuple[float, float]:
    entry = calibration.get(role, {})
    if not isinstance(entry, dict):
        raise HArmError(f"calibrator is missing: {role}")
    slope = float(entry.get("slope", 0.0))
    intercept = float(entry.get("intercept", 0.0))
    if not slope > 0:
        raise HArmError(f"calibrator slope must be positive: {role}")
    return slope, intercept

def dev_frontier_inputs(
    f0_logit: list[float], cand_logit: list[float], mapped: list[bool], calibrators: dict[str, Any]
) -> dict[str, Any]:
    f0_raw = [float(v) for v in f0_logit]
    cand_raw = [float(v) for v in cand_logit]
    flags = [bool(v) for v in mapped]
    if not (len(f0_raw) == len(cand_raw) == len(flags)) or not f0_raw:
        raise HArmError("dev score/mapping geometry differs")
    f0_slope, f0_intercept = dev_calibrator_pair(calibrators, "f0")
    cand_slope, cand_intercept = dev_calibrator_pair(calibrators, "candidate")
    f0_cal = calibrate_mod.apply_affine(f0_raw, f0_slope, f0_intercept)
    cand_cal = calibrate_mod.apply_affine(cand_raw, cand_slope, cand_intercept)
    candidate_raw_prob = sigmoid_list(cand_raw)
    candidate_raw_masked = mask_unmapped(candidate_raw_prob, flags)
    candidate_cal_masked = mask_unmapped(sigmoid_list(cand_cal), flags)
    return {
        "f0_logit": f0_raw,
        "candidate_logit": cand_raw,
        "f0_prob": sigmoid_list(f0_raw),
        "candidate_prob": sigmoid_list(cand_raw),
        "f0_cal_logit": f0_cal,
        "candidate_cal_logit": cand_cal,
        "f0_cal_prob": sigmoid_list(f0_cal),
        "candidate_cal_prob": sigmoid_list(cand_cal),
        "f0_masked": mask_unmapped(sigmoid_list(f0_raw), flags),
        "candidate_raw_masked": candidate_raw_masked,
        "candidate_cal_masked": candidate_cal_masked,
        "mapped": flags,
        "thresholds_raw": frontier_mod.unique_thresholds(candidate_raw_masked),
        "thresholds_calibrated": frontier_mod.unique_thresholds(candidate_cal_masked),
        "f0_reference": frontier_mod.RAW_REFERENCE_THRESHOLD,
    }


def run_dev_frontier(
    run_dir: Path,
    binding: dict[str, Any],
    dev_scores: dict[str, dict[str, Any]],
    metric_tables: dict[str, dict[int, dict[str, Any]]],
    corpus_of: dict[str, str],
    dev_calibrators: dict[str, Any],
    group_tables: dict[str, Any] | None = None,
    workers: int | None = None,
    phase: dict[str, Any] | None = None,
) -> dict[str, Any]:
    forbid_eval(list(dev_scores))
    prepared: dict[str, Any] = {}
    for source_id in sorted(dev_scores):
        entry = dev_scores[source_id]
        prepared[source_id] = dev_frontier_inputs(
            [float(v) for v in entry["f0"]],
            [float(v) for v in entry["candidate"]],
            [bool(v) for v in entry.get("mapped", [True] * len(entry["candidate"]))],
            dict(dev_calibrators),
        )
        prepared[source_id]["target"] = [float(v) for v in entry["target"]]
        prepared[source_id]["frames"] = len(entry["target"])
    def _kind_maps(table: dict[str, Any]) -> dict[str, Any]:
        calibrated = table.get("by_threshold_calibrated")
        raw = table.get("by_threshold_raw")
        if not isinstance(calibrated, dict) or not isinstance(raw, dict):
            raise HArmError("dev metric table lacks raw and calibrated thresholds")
        return {"raw": raw, "calibrated": calibrated}
    payloads: list[dict[str, Any]] = []
    for source_id in sorted(prepared):
        kind_thresholds = {
            "raw": prepared[source_id]["thresholds_raw"],
            "calibrated": prepared[source_id]["thresholds_calibrated"],
        }
        for horizon_ms in frontier_mod.HORIZONS_MS:
            table = metric_tables[source_id][int(horizon_ms)]
            maps = _kind_maps(table)
            for kind in ("raw", "calibrated"):
                thresholds = kind_thresholds[kind]
                by_threshold = maps[kind]
                payloads.append(
                    {
                        "source_id": f"{kind}/{source_id}",
                        "horizon_ms": int(horizon_ms),
                        "thresholds": thresholds,
                        "metrics": [list(by_threshold[t]) for t in thresholds],
                        "f0_metric": list(table["f0"]),
                    }
                )
    results = [_frontier_worker(payload) for payload in payloads]
    by_source_kinds: dict[str, Any] = {}
    for item in results:
        kind, source_id = str(item["source_id"]).split("/", 1)
        item = {**item, "source_id": source_id}
        by_source_kinds.setdefault(kind, {}).setdefault(source_id, {})[str(item["horizon_ms"])] = item
    dev_metrics: dict[str, Any] = {}
    for source_id in sorted(prepared):
        entry = prepared[source_id]
        kept = [i for i, m in enumerate(entry["mapped"]) if m]
        kept_target = [entry["target"][i] for i in kept]
        kept_raw_prob = [entry["candidate_prob"][i] for i in kept]
        kept_f0_cal = [entry["f0_cal_logit"][i] for i in kept]
        kept_cand_cal = [entry["candidate_cal_logit"][i] for i in kept]
        dev_metrics[source_id] = {
            "frames": entry["frames"],
            "kept_frames": len(kept),
            "raw_ap": calibrate_mod.average_precision(kept_raw_prob, kept_target),
            "f0_cal_nll": calibrate_mod.nll_loss(kept_f0_cal, kept_target),
            "f0_cal_brier": calibrate_mod.brier_score(kept_f0_cal, kept_target),
            "candidate_cal_nll": calibrate_mod.nll_loss(kept_cand_cal, kept_target),
            "candidate_cal_brier": calibrate_mod.brier_score(kept_cand_cal, kept_target),
        }
    if not isinstance(group_tables, dict):
        raise HArmError("group frontiers are missing")

    def _group_kind_blocks(entry: dict[str, Any], group_name: str, horizon_ms: int) -> dict[str, Any]:
        kinds = entry.get("kinds")
        if not isinstance(kinds, dict):
            raise HArmError(f"group frontier lacks kinds: {group_name}/{horizon_ms}")
        out = {}
        for kind in ("raw", "calibrated"):
            block = kinds.get(kind)
            if not isinstance(block, dict):
                raise HArmError(f"group frontier lacks {kind}: {group_name}/{horizon_ms}")
            out[kind] = block
        return out
    group_payloads: list[dict[str, Any]] = []
    for group_name in ("AMI", "AliMeeting", "pooled"):
        node = group_tables.get(group_name, {})
        for horizon_ms in frontier_mod.HORIZONS_MS:
            entry = node.get(int(horizon_ms))
            if not isinstance(entry, dict):
                raise HArmError(f"group frontier is missing: {group_name}/{horizon_ms}")
            for kind, block in _group_kind_blocks(entry, group_name, int(horizon_ms)).items():
                thresholds = [float(t) for t in block["thresholds"]]
                by_t = {float(p[0]): list(p[1:4]) for p in block["points"]}
                group_payloads.append(
                    {
                        "source_id": f"{kind}/group/{group_name}",
                        "horizon_ms": int(horizon_ms),
                        "thresholds": thresholds,
                        "metrics": [by_t[t] for t in thresholds],
                        "f0_metric": list(block["f0"]),
                    }
                )
    group_results = [_frontier_worker(payload) for payload in group_payloads]
    by_group_kinds: dict[str, Any] = {}
    for item in group_results:
        kind, _, name = str(item["source_id"]).split("/", 2)
        item = {**item, "source_id": f"group/{name}"}
        by_group_kinds.setdefault(kind, {}).setdefault(name, {})[str(item["horizon_ms"])] = item
    from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod
    canonical_horizons: dict[str, Any] = {}
    canonical_sources: dict[str, Any] = {}
    for source_id in sorted(prepared):
        canonical_sources[source_id] = {}
        for horizon_ms in frontier_mod.HORIZONS_MS:
            key = str(horizon_ms)
            canonical_sources[source_id][key] = {}
            for kind in ("raw", "calibrated"):
                node = by_source_kinds[kind][source_id][key]
                canonical_sources[source_id][key][kind] = {
                    "points": [dict(p) for p in node["points"]],
                    "reference": dict(node["reference"]),
                    "reference_kind": cross_mod.REFERENCE_KIND,
                    "budget": float(node["envelopes"]["budget"]),
                    "c_envelope": None if node["envelopes"]["c_envelope"] is None else dict(node["envelopes"]["c_envelope"]),
                    "m_envelope": None if node["envelopes"]["m_envelope"] is None else dict(node["envelopes"]["m_envelope"]),
                    "useful": bool(node["envelopes"]["useful"]),
                    "diagnostics": dict(dev_metrics[source_id]),
                }
    for horizon_ms in frontier_mod.HORIZONS_MS:
        key = str(horizon_ms)
        canonical_horizons[key] = {}
        members_by_group = {
            "ami": sorted(sid for sid, corpus in corpus_of.items() if corpus == "AMI"),
            "alimeeting": sorted(sid for sid, corpus in corpus_of.items() if corpus == "AliMeeting"),
        }
        members_by_group["pooled"] = sorted(set(members_by_group["ami"]) | set(members_by_group["alimeeting"]))
        members_by_group["macro"] = list(members_by_group["pooled"])
        for legacy_name, group in (("AMI", "ami"), ("AliMeeting", "alimeeting"), ("pooled", "pooled")):
            canonical_horizons[key][group] = {}
            for kind in ("raw", "calibrated"):
                node = by_group_kinds[kind][legacy_name][key]
                canonical_horizons[key][group][kind] = {
                    "points": [dict(p) for p in node["points"]],
                    "reference": dict(node["reference"]),
                    "reference_kind": cross_mod.REFERENCE_KIND,
                    "budget": float(node["envelopes"]["budget"]),
                    "c_envelope": None if node["envelopes"]["c_envelope"] is None else dict(node["envelopes"]["c_envelope"]),
                    "m_envelope": None if node["envelopes"]["m_envelope"] is None else dict(node["envelopes"]["m_envelope"]),
                    "useful": bool(node["envelopes"]["useful"]),
                    "diagnostics": {"members": members_by_group[group], "member_count": len(members_by_group[group])},
                }
        canonical_horizons[key]["macro"] = {}
        for kind in ("raw", "calibrated"):
            ami_node = by_group_kinds[kind]["AMI"][key]
            ali_node = by_group_kinds[kind]["AliMeeting"][key]
            try:
                macro_points = cross_mod.macro_average_points(ami_node["points"], ali_node["points"], what=f"macro/{kind}")
                macro_ref = cross_mod.macro_average_reference(ami_node["reference"], ali_node["reference"], what=f"macro/{kind}")
                macro_env = cross_mod.select_envelopes(macro_ref, macro_points)
            except cross_mod.CrossFrontierError as exc:
                raise HArmError(f"macro grids differ across corpora: {key}/{kind}: {exc}") from exc
            canonical_horizons[key]["macro"][kind] = {
                "points": macro_points,
                "reference": macro_ref,
                "reference_kind": cross_mod.REFERENCE_KIND,
                "budget": macro_env["budget"],
                "c_envelope": macro_env["c_envelope"],
                "m_envelope": macro_env["m_envelope"],
                "useful": macro_env["useful"],
                "diagnostics": {"members": members_by_group["macro"], "member_count": len(members_by_group["macro"])},
            }
    for source_id in sorted(prepared):
        entry = prepared[source_id]
        arm_runtime.save_source_predictions(
            Path(run_dir),
            f"dev_{source_id}",
            {
                "f0_logit": entry["f0_logit"],
                "candidate_logit": entry["candidate_logit"],
                "f0_prob": entry["f0_prob"],
                "candidate_prob": entry["candidate_prob"],
                "f0_cal_logit": entry["f0_cal_logit"],
                "candidate_cal_logit": entry["candidate_cal_logit"],
                "f0_cal_prob": entry["f0_cal_prob"],
                "candidate_cal_prob": entry["candidate_cal_prob"],
                "target": entry["target"],
                "mapped": entry["mapped"],
            },
            dict(binding),
        )
    frontier_doc = {
        "artifact_role": cross_mod.ARTIFACT_ROLE,
        "version": cross_mod.CANONICAL_VERSION,
        "arm": H_ARM,
        "binding": dict(binding),
        "horizons_ms": list(frontier_mod.HORIZONS_MS),
        "group_order": list(cross_mod.GROUP_ORDER),
        "horizons": canonical_horizons,
        "sources": canonical_sources,
        "phase": dict(phase or {}),
    }
    cross_mod.validate_canonical(frontier_doc)
    arm_runtime.atomic_write_json(Path(run_dir) / DEV_FRONTIER_NAME, frontier_doc)
    return frontier_doc


@dataclass(slots=True)
class HArmDeps:
    load_bundle_manifest: Callable[[], dict[str, Any]] | None = None
    bundle_dir: Path | None = None
    build_missing_targets: Callable[[str], dict[str, Any]] | None = None
    build_evidence: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
    build_features: Callable[[dict[str, Any]], Any] | None = None
    build_targets: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
    load_wrapper_head: Callable[[], tuple[Any, Any]] | None = None
    load_torch: Callable[[], Any] | None = None

    total_profile_steps: int = 0
    hourly_cost_usd: float = 0.0
    workers: int | None = None
    profile_batch: Callable[[Any, Any, Any], dict[str, Any]] | None = None
    profile_dev_sample: Callable[[Any, Any, Any], dict[str, Any]] | None = None
    export_gpu_evidence: Callable[[Any, Any, Any], dict[str, Any]] | None = None

def gpu_export_dir(run_dir: Path) -> Path:
    return Path(run_dir) / GPU_EXPORT_DIRNAME


def gpu_export_manifest_path(run_dir: Path) -> Path:
    return gpu_export_dir(run_dir) / GPU_EXPORT_MANIFEST_NAME


def gpu_export_progress_path(run_dir: Path) -> Path:
    return gpu_export_dir(run_dir) / GPU_EXPORT_PROGRESS_NAME


def export_npz_name(role: str, source_id: str) -> str:
    if role not in ("calib", "dev"):
        raise HArmError(f"export role is invalid: {role}")
    return f"{role}_{source_id}.npz"


def corpus_family(corpus: str) -> str:
    mapping = {"AMI": "ami_mix_headset", "AliMeeting": "alimeeting_far_ch0"}
    family = mapping.get(str(corpus))
    if family is None:
        raise HArmError(f"export corpus is unknown: {corpus}")
    return family


def resolve_source_corpus(source_id: str, payload: Mapping[str, Any] | None = None) -> str:
    body = dict(payload or {})
    corpus = str(body.get("corpus") or "")
    if corpus in ("AMI", "AliMeeting"):
        return corpus
    from experiments.psem_sortformer_adaptation_depth.preflight import SOURCE_MANIFEST_PATH
    from experiments.psem_state_corrected_adaptation_gate.material import load_source_rows

    row = load_source_rows(SOURCE_MANIFEST_PATH).get(str(source_id), {})

    corpus = str(row.get("corpus") or "")
    if corpus not in ("AMI", "AliMeeting"):
        raise HArmError(f"export corpus is unknown: {corpus}")
    return corpus


def with_authoritative_corpus(source_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    corpus = resolve_source_corpus(source_id, body)
    body["corpus"] = corpus
    if not str(body.get("source_family") or ""):
        body["source_family"] = corpus_family(corpus)
    return body



def trained_head_digest(torch: Any, head_module: Any) -> str:
    payload = {str(k): v.detach().to("cpu").contiguous() for k, v in head_module.state_dict().items()}
    return hashlib.sha256(serialize_blob(torch, payload)).hexdigest()


def write_aligned_export_npz(
    path: Path,
    f0_raw: Sequence[float],
    cand_raw: Sequence[float],
    target: Sequence[float],
    valid: Sequence[bool],
    mapped: Sequence[bool],
) -> str:
    import numpy as np

    f0 = np.asarray(list(f0_raw), dtype=np.float64)
    cand = np.asarray(list(cand_raw), dtype=np.float64)
    y = np.asarray(list(target), dtype=np.float64)
    valid_arr = np.asarray(list(valid), dtype=bool)
    mapped_arr = np.asarray(list(mapped), dtype=bool)
    if not (len(f0) == len(cand) == len(y) == len(valid_arr) == len(mapped_arr)) or len(f0) == 0:
        raise HArmError(f"export NPZ geometry differs: {path}")
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_name(f"{target_path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            f0_raw=f0,
            cand_raw=cand,
            target=y,
            valid=valid_arr,
            mapped=mapped_arr,
        )
    os.replace(tmp, target_path)
    load_aligned_export_npz(target_path)
    return arm_runtime.sha256_file(target_path)


def load_aligned_export_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    try:
        with np.load(Path(path), allow_pickle=False) as data:
            missing = [key for key in ("f0_raw", "cand_raw", "target", "valid", "mapped") if key not in data.files]
            if missing:
                raise HArmError(f"export NPZ lacks fields {missing}: {path}")
            f0 = np.asarray(data["f0_raw"], dtype=np.float64).reshape(-1)
            cand = np.asarray(data["cand_raw"], dtype=np.float64).reshape(-1)
            target = np.asarray(data["target"], dtype=np.float64).reshape(-1)
            valid = np.asarray(data["valid"], dtype=bool).reshape(-1)
            mapped = np.asarray(data["mapped"], dtype=bool).reshape(-1)
    except (OSError, ValueError) as exc:
        raise HArmError(f"export NPZ is unreadable: {path}") from exc
    if not (len(f0) == len(cand) == len(target) == len(valid) == len(mapped)) or len(f0) == 0:
        raise HArmError(f"export NPZ geometry differs: {path}")
    return {
        "f0_raw": f0,
        "cand_raw": cand,
        "target": target,
        "valid": valid,
        "mapped": mapped,
        "frames": int(len(f0)),
    }


def export_source_entry(
    role: str,
    source_id: str,
    frames: int,
    family: str,
    mapping_mapped: int,
    mapping_total: int,
    unmapped_frames: int,
    kept_frames: int,
    coverage: Mapping[str, Any],
    infer_seconds: float,
) -> dict[str, Any]:
    return {
        "file": export_npz_name(role, source_id),
        "frames": int(frames),
        "family": str(family),
        "mapping_mapped": int(mapping_mapped),
        "mapping_total": int(mapping_total),
        "unmapped_frames": int(unmapped_frames),
        "kept_frames": int(kept_frames),
        "coverage": {str(k): v for k, v in dict(coverage).items()},
        "infer_seconds": float(infer_seconds),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HArmError(f"export document is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise HArmError(f"export document is not an object: {path}")
    return payload


def gpu_export_identity(binding: Mapping[str, Any], head_digest: str, metrics_sha256: str) -> dict[str, Any]:
    head = str(head_digest or "")
    metrics = str(metrics_sha256 or "")
    if not head or not metrics:
        raise HArmError("GPU export identity lacks trained-head or training-metrics hash")
    return {
        "binding": dict(binding),
        "trained_head_sha256": head,
        "training_metrics_sha256": metrics,
    }


def export_progress_identity(binding: Mapping[str, Any], head_digest: str, metrics_sha256: str = "") -> dict[str, Any]:
    return gpu_export_identity(binding, head_digest, metrics_sha256)


def require_gpu_export_identity(payload: Mapping[str, Any], expected: Mapping[str, Any], what: str) -> None:
    got_head = str(payload.get("trained_head_sha256") or "")
    metrics = payload.get("training_metrics")
    if isinstance(metrics, dict):
        got_metrics = str(metrics.get("sha256") or "")
    else:
        got_metrics = str(payload.get("training_metrics_sha256") or "")
    if not got_head or not got_metrics:
        raise HArmError(f"{what} lacks trained-head or training-metrics identity")
    if dict(payload.get("binding") or {}) != dict(expected["binding"]):
        raise HArmError(f"{what} binding differs")
    if got_head != str(expected["trained_head_sha256"]):
        raise HArmError(f"{what} trained-head identity differs")
    if got_metrics != str(expected["training_metrics_sha256"]):
        raise HArmError(f"{what} training-metrics identity differs")


def load_export_progress(
    run_dir: Path, binding: Mapping[str, Any], head_digest: str, metrics_sha256: str = ""
) -> dict[str, Any]:
    expected = gpu_export_identity(binding, head_digest, metrics_sha256)
    path = gpu_export_progress_path(run_dir)
    if not path.is_file():
        return {"identity": expected, "completed": {}}
    payload = _read_json_object(path)
    identity = dict(payload.get("identity") or {})
    if not identity.get("trained_head_sha256") or not identity.get("training_metrics_sha256"):
        raise HArmError("partial GPU export lacks trained-head or training-metrics identity")
    if identity != expected:
        raise HArmError("partial GPU export binding or trained-head identity differs")
    completed = payload.get("completed")
    if not isinstance(completed, dict):
        raise HArmError("partial GPU export ledger is invalid")
    records: dict[str, dict[str, Any]] = {}
    for key, value in completed.items():
        if not isinstance(value, dict) or "sha256" not in value or "entry" not in value:
            raise HArmError("partial GPU export ledger is invalid")
        records[str(key)] = {"sha256": str(value["sha256"]), "entry": dict(value["entry"])}
    return {"identity": expected, "completed": records}




def resume_export_npz(export_dir: Path, filename: str, expected_sha256: str | None) -> str | None:
    path = Path(export_dir) / filename
    if not path.is_file():
        return None
    digest = arm_runtime.sha256_file(path)
    load_aligned_export_npz(path)
    if expected_sha256 is None:
        raise HArmError(f"GPU export file exists without a matching ledger hash: {filename}")
    if digest != str(expected_sha256):
        raise HArmError(f"GPU export file is corrupt: {filename}")
    return digest


def write_gpu_export_manifest(
    run_dir: Path,
    config: arm_runtime.ArmRunConfig,
    calib: Mapping[str, dict[str, Any]],
    dev: Mapping[str, dict[str, Any]],
    frozen_cache_inference_seconds: Mapping[str, float],
    training_metrics_path: Path,
    *,
    fit: Sequence[str],
    salt: str,
    target_frac: float,
    trained_head_sha256: str,

) -> dict[str, Any]:
    export_dir = gpu_export_dir(run_dir)
    calib_sources = sorted(calib)
    dev_sources = sorted(dev)
    fit_sources = sorted(str(s) for s in fit)
    files: dict[str, str] = {}
    for role, table in (("calib", calib), ("dev", dev)):
        for source_id in sorted(table):
            entry = table[source_id]
            filename = str(entry["file"])
            if filename != export_npz_name(role, source_id):
                raise HArmError(f"export filename differs: {filename}")
            loaded = load_aligned_export_npz(export_dir / filename)
            if int(loaded["frames"]) != int(entry["frames"]):
                raise HArmError(f"export frame counts differ: {source_id}")
            files[filename] = arm_runtime.sha256_file(export_dir / filename)
    metrics_path = Path(training_metrics_path)
    if not metrics_path.is_file():
        raise HArmError("training metrics are missing for GPU export")
    try:
        rel_metrics = str(metrics_path.relative_to(Path(run_dir)))
    except ValueError:
        rel_metrics = str(metrics_path)
    metrics_sha = arm_runtime.sha256_file(metrics_path)
    head_digest = str(trained_head_sha256 or "")
    gpu_export_identity(dict(config.binding), head_digest, metrics_sha)
    body = {
        "artifact_role": GPU_EXPORT_ARTIFACT_ROLE,
        "binding": dict(config.binding),
        "arm": H_ARM,
        "seed": int(config.seed),
        "fit": fit_sources,
        "salt": str(salt),
        "target_frac": float(target_frac),
        "calib_sources": calib_sources,
        "dev_sources": dev_sources,
        "calib": {sid: dict(calib[sid]) for sid in calib_sources},
        "dev": {sid: dict(dev[sid]) for sid in dev_sources},
        "files": files,
        "frozen_cache_inference_seconds": {
            str(k): float(v) for k, v in dict(frozen_cache_inference_seconds).items()
        },
        "trained_head_sha256": head_digest,
        "training_metrics": {
            "path": rel_metrics.replace("\\", "/"),
            "sha256": metrics_sha,
        },
    }
    arm_runtime.atomic_write_json(gpu_export_manifest_path(run_dir), body)
    progress = gpu_export_progress_path(run_dir)
    if progress.is_file():
        progress.unlink()
    return body


def join_dev_export_population(runtime: Mapping[str, Any], snapshots: Sequence[Any]) -> list[dict[str, Any]]:
    if not runtime:
        raise HArmError("frozen DEV scoring population is empty")
    snap_of = {str(s.source_id): s for s in snapshots}
    members: list[dict[str, Any]] = []
    for source_id in sorted(runtime):
        forbid_eval([source_id], {source_id: str(getattr(runtime[source_id], "role", ""))})
        snapshot = snap_of.get(source_id)
        if snapshot is None:
            raise HArmError(f"DEV snapshot is missing for scoring session: {source_id}")
        family = str(getattr(snapshot, "source_family", "") or "")
        if not family:
            raise HArmError(f"DEV family is missing: {source_id}")
        members.append(
            {
                "source_id": str(source_id),
                "snapshot": snapshot,
                "session": runtime[source_id],
                "family": family,
            }
        )
    return members


def load_dev_export_population(corpus_root: Path, reference_root: Path) -> list[dict[str, Any]]:
    from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import load_sessions
    from experiments.psem_sortformer_adaptation_depth.execution import load_scoring_sessions
    from experiments.psem_training_strategy_gate.sampling import DEV_ROLE

    runtime = load_scoring_sessions(Path(corpus_root), Path(reference_root), DEV_ROLE)
    return join_dev_export_population(runtime, load_sessions())



def run_postprocess_command(export_dir: Path, out_dir: Path, workers: int | None = None) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate.h_postprocess import run_postprocess

    count = 8 if workers is None else int(workers)
    return run_postprocess(Path(export_dir), Path(out_dir), workers=count)



def default_torch_loader() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise HArmError("torch is unavailable on this host") from exc
    arm_runtime.enforce_thread_caps()
    return torch


def seed_all_from_config(torch: Any, seed: int) -> dict[str, Any]:
    import random

    random.seed(int(seed))
    numpy_seeded = False
    try:
        import numpy as _np

        _np.random.seed(int(seed) % (2**32))
        numpy_seeded = True
    except ImportError:
        numpy_seeded = False
    torch.manual_seed(int(seed))
    cuda_seeded = False
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            cuda_seeded = True
    except (RuntimeError, AttributeError):
        cuda_seeded = False
    return {
        "seed": int(seed),
        "python": True,
        "numpy": numpy_seeded,
        "torch_cpu": True,
        "torch_cuda": cuda_seeded,
    }


def snapshot_rng(torch: Any) -> dict[str, Any]:
    import random

    cuda_state = None
    try:
        if torch.cuda.is_available():
            cuda_state = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    except (RuntimeError, AttributeError):
        cuda_state = None
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
    }


def restore_rng(torch: Any, snap: dict[str, Any]) -> None:
    import random

    random.setstate(snap["python"])
    torch.set_rng_state(snap["torch_cpu"])
    cuda_state = snap.get("torch_cuda")
    if cuda_state is None:
        return
    try:
        available = bool(torch.cuda.is_available())
    except (RuntimeError, AttributeError):
        available = False
    if not available:
        return
    states = [s.cpu() if hasattr(s, "cpu") else s for s in cuda_state]
    try:
        torch.cuda.set_rng_state_all(states)
    except (RuntimeError, AttributeError, ValueError, TypeError) as exc:
        raise HArmError(f"CUDA RNG restoration failed: {exc}") from exc


def run_profile_command(
    config: arm_runtime.ArmRunConfig,
    store: Path,
    deps: HArmDeps,
) -> dict[str, Any]:
    authorization = check_authorized(config, store)
    run_dir = config.run_dir()
    manifest = deps.load_bundle_manifest()
    fit, calib = fit_calib_from_bundle(manifest)
    forbid_eval(list(fit) + list(calib))
    binding = h_binding(config, manifest)
    if deps.profile_batch is None:
        raise HArmError("profile requires a real target/evidence batch")
    if deps.profile_dev_sample is None:
        raise HArmError("profile requires a real representative DEV inference")
    torch = (deps.load_torch or default_torch_loader)()
    seed_report = seed_all_from_config(torch, int(config.seed))
    wrapper, head_module = deps.load_wrapper_head()
    wrapper_was_training = bool(wrapper.training)
    head_was_training = bool(head_module.training)
    freeze_backbone_train_head(wrapper, head_module)
    initial_state = collect_head_state(torch, head_module)
    optimizer = build_head_optimizer(torch, head_module)
    precompute_start = time.perf_counter()
    batch = dict(deps.profile_batch(head_module, wrapper, torch))
    precompute_seconds = time.perf_counter() - precompute_start
    cache_bytes = int(batch.get("cache_bytes", batch.get("io_bytes", 0)))
    weights = dict(binding["class_weights"])
    windows = [tuple(w) for w in batch.get("windows", [])] or [(0, int(batch["features"].shape[1]))]
    carrier = streaming_mod.StateCarrier(None, str(batch.get("source_id", "profile")))
    device = next(head_module.parameters()).device

    trainable_params = [p for p in head_module.parameters() if p.requires_grad]
    profile_weight_cache: dict[str, Any] | None = None

    def step_fn() -> dict[str, Any]:
        nonlocal profile_weight_cache
        start, end = windows[step_fn.calls % len(windows)]
        step_fn.calls += 1
        optimizer.zero_grad()
        outputs, next_state = head_module(batch["features"][:, start:end], carrier.carry())
        carrier.state = next_state
        carrier.detach()
        product = outputs["product_logit"] if "product_logit" in outputs else (batch["f0"][:, start:end] + outputs["z_residual"])
        if profile_weight_cache is None:
            profile_weight_cache = build_loss_weight_tensors(torch, weights, product.dtype, product.device)
        loss = chunk_loss_value(
            torch,
            product,
            outputs["anchor_logit"],
            batch["y_replace"][:, start:end],
            batch["y_anchor"][:, start:end],
            batch["mult_weight"][:, start:end],
            weights,
            cached_weights=profile_weight_cache,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, HEAD_CLIP_NORM)
        optimizer.step()
        return {"io_bytes": int(batch.get("io_bytes", 0))}

    step_fn.calls = 0

    def dev_infer_fn() -> dict[str, Any]:
        sample = dict(deps.profile_dev_sample(head_module, wrapper, torch))
        frozen = dict(batch.get("frozen_evidence") or {})
        fit_ids = list(frozen.get("fit_sources") or fit)
        calib_ids = list(frozen.get("calib_sources") or calib)
        unbuilt = list(frozen.get("unbuilt_sources") or [])
        measured_frozen = frozen.get("measured_seconds")
        if frozen.get("hit"):
            measured_frozen = None
        estimated_frozen = None
        if measured_frozen is not None and unbuilt:
            estimated_frozen = float(measured_frozen) * int(len(unbuilt))
        dev_population = sample.get("dev_population")
        if dev_population is None and sample.get("dev_sources"):
            dev_population = len(list(sample["dev_sources"]))
        representative = float(sample.get("infer_seconds", 0.0))
        estimated_all_dev = None
        if dev_population:
            estimated_all_dev = representative * int(dev_population)
        cpu_tail = dict(sample.get("cpu_tail") or {})
        if not cpu_tail and sample.get("frontier_slice"):
            slice_info = dict(sample["frontier_slice"])
            cpu_tail = {
                "kind": "estimated",
                "representative_slice": slice_info,
                "seconds": slice_info.get("seconds"),
                "projected_seconds": slice_info.get("projected_seconds"),
            }
        return {
            **sample,
            "measured_frozen_evidence_seconds": None if measured_frozen is None else float(measured_frozen),
            "measured_frozen_source_id": frozen.get("source_id"),
            "estimated_frozen_evidence_seconds": estimated_frozen,
            "frozen_evidence_population": {
                "fit": int(len(fit_ids)),
                "calib": int(len(calib_ids)),
                "unbuilt": int(len(unbuilt)),
                "kind": "estimated" if estimated_frozen is not None else "unmeasured",
            },
            "estimated_all_dev_infer_seconds": estimated_all_dev,
            "dev_population": None if dev_population is None else int(dev_population),
            "cpu_tail": cpu_tail,
        }


    try:
        receipt = run_profile(
            run_dir,
            binding,
            step_fn,
            dev_infer_fn,
            int(deps.total_profile_steps or 0),
            torch=torch,
            device=device,
            steps=MIN_PROFILE_STEPS,
            hourly_cost_usd=float(deps.hourly_cost_usd),
        )
    finally:
        head_module.load_state_dict({k: v.clone() for k, v in initial_state.items()})
        for param in head_module.parameters():
            param.grad = None
        head_module.train(head_was_training)
        if not wrapper_was_training:
            wrapper.eval()
    receipt = {
        **receipt, "authoritative": False, "scope": "profile-only",
        "stateful": True, "windows": len(windows), "group_size": 1,
        "seed": int(config.seed), "seed_report": seed_report,
        "precompute_seconds": float(precompute_seconds),
        "cache_bytes": int(cache_bytes),
        "evidence_cache": dict(batch.get("frozen_evidence", {})),
        "workers": arm_runtime.worker_receipt(deps.workers, 1),
    }
    arm_runtime.atomic_write_json(run_dir / PROFILE_RECEIPT_NAME, receipt)
    return {"authorization": authorization, "profile": receipt}


def run_h_arm(
    config: arm_runtime.ArmRunConfig,
    store: Path,
    deps: HArmDeps,
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import receipts as receipts_mod

    authorization = check_authorized(config, store)
    run_dir = config.run_dir()
    manifest = deps.load_bundle_manifest()
    fit, calib = fit_calib_from_bundle(manifest)
    forbid_eval(list(fit) + list(calib))
    binding = h_binding(config, manifest)
    require_profile(run_dir, binding)
    payloads = ensure_full_source_targets(
        Path(deps.bundle_dir), manifest, run_dir, deps.build_missing_targets
    )
    torch = (deps.load_torch or default_torch_loader)()
    seed_report = seed_all_from_config(torch, int(config.seed))
    wrapper, head_module = deps.load_wrapper_head()
    mode_audit = freeze_backbone_train_head(wrapper, head_module)
    arm_runtime.atomic_write_json(
        run_dir / MODE_RECEIPT_NAME,
        receipts_mod.module_mode_receipt(True, True, H_ARM),
    )
    cache_records: dict[str, dict[str, Any]] = {}
    needed_cache = arm_runtime.chronological_sources(sorted(set(fit) | set(calib)))
    frozen_reused: list[str] = []
    frozen_built: list[str] = []
    cache_timings: dict[str, float] = {}

    for source_id in needed_cache:
        payload = check_target_geometry(dict(payloads[source_id]), source_id)
        cache_started = time.perf_counter()

        frames = int(payload["num_frames"])
        if cache_meta_path(run_dir, source_id).is_file() and cache_npz_path(run_dir, source_id).is_file():
            meta = verify_source_cache_file(run_dir, source_id, binding)
            if int(meta.get("frames", -1)) != frames:
                raise HArmError(f"evidence cache frame counts differ: {source_id}")
            cache_records[source_id] = {
                "file": meta["file"],
                "sha256": meta["sha256"],
                "frames": meta["frames"],
            }
            cache_timings[source_id] = time.perf_counter() - cache_started
            del meta
            continue


        identity = frozen_evidence_identity(binding, payload, source_id)
        frozen_npz, _, _ = frozen_evidence_paths(config.root, identity)
        hit = frozen_hit_meta(config.root, identity)
        if hit is not None:
            frozen_reused.append(source_id)
            meta = reuse_frozen_to_cache(run_dir, source_id, frozen_npz, hit, binding)
            del hit
        else:
            frozen_built.append(source_id)
            evidence = deps.build_evidence(source_id, payloads[source_id])
            import numpy as np

            hidden_arr = np.asarray(evidence["hidden192"], dtype=np.float32)
            logits_arr = np.asarray(evidence["slot_logits4"], dtype=np.float32)
            if hidden_arr.ndim != 2 or hidden_arr.shape[1] != 192:
                raise HArmError(f"cached hidden geometry differs: {hidden_arr.shape}")
            if logits_arr.ndim != 2 or logits_arr.shape[1] != 4 or logits_arr.shape[0] != hidden_arr.shape[0]:
                raise HArmError(f"cached slot-logit geometry differs: {logits_arr.shape}")
            frozen_meta = write_frozen_evidence(
                config.root,
                identity,
                hidden_arr,
                logits_arr,
                dict(evidence.get("slot_of", {})),
                list(evidence.get("mapping_rows", [])),
                dict(evidence.get("timing", {})),
            )
            del hidden_arr, logits_arr
            meta = reuse_frozen_to_cache(run_dir, source_id, frozen_npz, frozen_meta, binding)
            del evidence, frozen_meta
        if int(meta.get("frames", -1)) != frames:
            raise HArmError(f"evidence cache frame counts differ: {source_id}")
        cache_records[source_id] = {
            "file": meta["file"],
            "sha256": meta["sha256"],
            "frames": meta["frames"],
        }
        del meta
        cache_timings[source_id] = time.perf_counter() - cache_started

    write_cache_manifest(run_dir, cache_records, binding)
    require_cache_coverage(run_dir, fit, calib, binding)
    flags_by_source: dict[str, list[bool]] = {}
    for source_id in arm_runtime.chronological_sources(list(fit)):
        payload = check_target_geometry(dict(payloads[source_id]), source_id)
        mult = target_mult_list(dict(payload), cache_unmapped_frames(run_dir, source_id))
        flags_by_source[source_id] = loss_flags_from_mult_list(int(payload["num_frames"]), mult)
        del mult
    schedule = plan_fit_schedule(list(fit), flags_by_source)
    del flags_by_source
    checkpoint = arm_runtime.load_source_checkpoint(run_dir, binding)
    completed = list(checkpoint.get("completed_sources", [])) if not checkpoint.get("fresh") else []
    remaining = remaining_fit_sources(list(fit), completed)
    def _feature_loader(source_id: str) -> Any:
        cached = read_source_cache(Path(run_dir), str(source_id), dict(binding))
        hidden = cached["hidden192"]
        logits = cached["slot_logits4"]
        if hidden.ndim != 2 or hidden.shape[1] != 192 or logits.ndim != 2 or logits.shape[1] != 4:
            raise HArmError(f"evidence cache geometry differs: {source_id}")
        if hidden.shape[0] != logits.shape[0]:
            raise HArmError(f"evidence cache frame counts differ: {source_id}")
        out = deps.build_features(
            {
                "hidden192": hidden,
                "slot_logits4": logits,
                "target": payloads[str(source_id)],
                "slot_of": {str(k): int(v) for k, v in dict(cached["meta"].get("slot_of", {})).items()},
            }
        )
        del cached, hidden, logits
        return out
    def _target_loader(source_id: str) -> dict[str, Any]:
        return deps.build_targets(str(source_id), payloads[str(source_id)])
    if remaining:
        optimizer = build_head_optimizer(torch, head_module)
        scheduler = build_warmup_scheduler(torch, optimizer, int(schedule["warmup_steps"]))
        pending: dict[str, Any] = {"accum_count": 0, "steps_taken": 0}
        if not checkpoint.get("fresh"):
            restored = restore_head_state(
                torch, run_dir, binding, head_module, optimizer, scheduler,
                source_order=list(schedule.get("sources", [])),
                record=checkpoint,
            )
            pending = dict(restored.get("pending", pending))
        train_metrics = run_fit_pass(
            torch,
            wrapper,
            head_module,
            {},
            {},
            dict(binding["class_weights"]),
            run_dir,
            binding,
            schedule,
            completed=completed,
            optimizer=optimizer,
            scheduler=scheduler,
            pending=pending,
            feature_loader=_feature_loader,
            target_loader=_target_loader,
        )
    else:
        if completed and not checkpoint.get("fresh"):
            optimizer = build_head_optimizer(torch, head_module)
            scheduler = build_warmup_scheduler(torch, optimizer, int(schedule["warmup_steps"]))
            restore_head_state(
                torch, run_dir, binding, head_module, optimizer, scheduler,
                source_order=list(schedule.get("sources", [])),
                record=checkpoint,
            )
        train_metrics = {
            "schedule": schedule,
            "loss_chunks": int(schedule["loss_chunks"]),
            "total_steps": int(schedule["total_steps"]),
            "warmup_steps": int(schedule["warmup_steps"]),
            "steps_taken": int(schedule["total_steps"]),
            "accum_count": 0,
            "per_source": {},
            "completed_sources": completed,
        }

    train_metrics = {**train_metrics, "artifact_role": "issue-121-h-training-metrics", "mode_audit": mode_audit}
    train_metrics = {
        **train_metrics,
        "seed": int(config.seed),
        "seed_report": seed_report,
        "evidence_cache": {
            "reused": frozen_reused,
            "built": frozen_built,
            "inference_seconds": {str(k): float(v) for k, v in cache_timings.items()},
        },
        "trained_head_sha256": trained_head_digest(torch, head_module),
    }
    if deps.export_gpu_evidence is None:
        raise HArmError("run requires GPU evidence export")
    metrics_path = arm_runtime.atomic_write_json(run_dir / TRAINING_METRICS_NAME, train_metrics)
    export_doc = deps.export_gpu_evidence(head_module, wrapper, torch)
    if not isinstance(export_doc, dict) or export_doc.get("artifact_role") != GPU_EXPORT_ARTIFACT_ROLE:
        raise HArmError("GPU evidence export manifest is invalid")
    arm_runtime.atomic_write_json(run_dir / EXPERIMENT_MANIFEST_NAME, receipts_mod.experiment_manifest())
    arm_runtime.atomic_write_json(
        run_dir / SAMPLING_MANIFEST_NAME,
        receipts_mod.sampling_calibration_manifest(
            list(fit), list(calib), str(manifest.get("salt", "")), float(manifest.get("target_frac", 0.0))
        ),
    )
    artifacts = [
        run_dir / EXPERIMENT_MANIFEST_NAME,
        run_dir / SAMPLING_MANIFEST_NAME,
        run_dir / MODE_RECEIPT_NAME,
        metrics_path,
        gpu_export_manifest_path(run_dir),
        run_dir / CACHE_MANIFEST_NAME,
        run_dir / PROFILE_RECEIPT_NAME,
    ]
    for extra in sorted(gpu_export_dir(run_dir).glob("*.npz")):
        artifacts.append(extra)
    for extra in sorted((run_dir / CACHE_DIRNAME).glob("*.npz")):
        artifacts.append(extra)
    for extra in sorted((run_dir / CACHE_DIRNAME).glob("*.json")):
        artifacts.append(extra)
    checkpoint_manifest = run_dir / arm_runtime.CHECKPOINT_DIRNAME / arm_runtime.CHECKPOINT_NAME
    if checkpoint_manifest.is_file():
        artifacts.append(checkpoint_manifest)
    for extra in sorted((run_dir / arm_runtime.CHECKPOINT_DIRNAME).glob("*.pt")):
        artifacts.append(extra)
    manifest_path = arm_runtime.write_final_manifest(
        run_dir,
        {
            "artifact_role": "issue-121-h-final-manifest",
            "arm": H_ARM,
            "seed": config.seed,
            "binding": binding,
            "authorization": authorization,
            "gpu_export": GPU_EXPORT_ARTIFACT_ROLE,
        },
        artifacts,
    )
    return {"authorization": authorization, "run_dir": str(run_dir), "final_manifest": str(manifest_path)}



def pod_model_context(inputs: HArmPodInputs, seed: int) -> dict[str, Any]:
    torch = default_torch_loader()
    from experiments.psem_sortformer_adaptation_depth.nemo_adapter import load_pinned_sortformer
    from experiments.psem_state_corrected_adaptation_gate import head as head_mod

    wrapper, runtime_receipt = load_pinned_sortformer(
        Path(inputs.checkpoint), Path(inputs.nemo_checkout), Path(inputs.dependency_lock), str(inputs.device)
    )
    json.dumps(runtime_receipt)
    seed_report = seed_all_from_config(torch, int(seed))
    device = next(wrapper.parameters()).device
    head_module = head_mod.ResidualPSEMHead(arm_runtime.HEAD_INPUT_DIM)
    head_module.to(device)
    return {"torch": torch, "wrapper": wrapper, "head": head_module, "device": device, "seed": seed_report}


@dataclass(slots=True)
class HArmPodInputs:
    bundle_dir: Path
    checkpoint: Path
    nemo_checkout: Path
    dependency_lock: Path
    corpus_root: Path
    reference_root: Path
    sampling_manifest: Path
    device: str = "cuda"
    workers: int | None = None
    hourly_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_dir", Path(self.bundle_dir))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "nemo_checkout", Path(self.nemo_checkout))
        object.__setattr__(self, "dependency_lock", Path(self.dependency_lock))
        object.__setattr__(self, "corpus_root", Path(self.corpus_root))
        object.__setattr__(self, "reference_root", Path(self.reference_root))
        object.__setattr__(self, "sampling_manifest", Path(self.sampling_manifest))


def pod_stage_manifest(inputs: HArmPodInputs) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import stages as stages_mod

    stages_mod.verify_bundle_manifest(Path(inputs.bundle_dir), "stage_a_manifest.json")
    with open(Path(inputs.bundle_dir) / "stage_a_manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise HArmError("stage-A bundle manifest is invalid")
    return manifest


def pod_full_target_cache(inputs: HArmPodInputs) -> dict[str, dict[str, Any]]:
    from experiments.psem_sortformer_adaptation_depth.sampling import load_training_sessions
    from experiments.psem_state_corrected_adaptation_gate import material as material_mod
    from experiments.psem_state_corrected_adaptation_gate import stages as stages_mod

    population = material_mod.resolve_sampling_population(Path(inputs.sampling_manifest))
    sessions = load_training_sessions(Path(inputs.corpus_root), Path(inputs.reference_root))
    cache = material_mod.build_all_source_targets(
        sessions, dict(population["rows_by_source"]), material_mod.resolve_worker_count(inputs.workers)
    )
    serialized: dict[str, dict[str, Any]] = {}
    for source_id, entry in cache.items():
        payload = stages_mod.serialize_authority(source_id, entry)
        payload["audio_ref"] = str(sessions[source_id].audio_ref)
        payload["waveform_sha256"] = str(sessions[source_id].waveform_sha256)
        payload["sampling_sha256"] = str(population["sampling_sha256"])
        rows = population["rows_by_source"].get(source_id, [])
        payload["corpus"] = next(
            (row.get("corpus") for row in rows if row.get("corpus") in ("AMI", "AliMeeting")), ""
        )
        serialized[source_id] = with_authoritative_corpus(source_id, payload)

    return serialized


def pod_payload_for(inputs: HArmPodInputs, run_dir: Path, manifest: dict[str, Any], source_id: str) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import stages as stages_mod

    entries = manifest.get("targets", {})
    if isinstance(entries.get(source_id), dict):
        payload = stages_mod.load_stage_targets(Path(inputs.bundle_dir), manifest, [source_id])[source_id]
    else:
        cached = Path(run_dir) / H_TARGETS_DIRNAME / f"{source_id}.json"
        payload = json.loads(cached.read_text(encoding="utf-8"))
    return with_authoritative_corpus(source_id, payload)



def pod_ensure_targets(inputs: HArmPodInputs, run_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from experiments.psem_state_corrected_adaptation_gate import stages as stages_mod

    fit, calib = fit_calib_from_bundle(manifest)
    needed = sorted(set(fit) | set(calib))
    entries = manifest.get("targets", {})
    present = [s for s in needed if isinstance(entries.get(s), dict)]
    payloads: dict[str, dict[str, Any]] = {}
    if present:
        loaded = stages_mod.load_stage_targets(Path(inputs.bundle_dir), manifest, present)
        payloads.update({sid: with_authoritative_corpus(sid, item) for sid, item in loaded.items()})

    missing = [s for s in needed if s not in payloads]
    out_dir = Path(run_dir) / H_TARGETS_DIRNAME
    remaining: list[str] = []
    for source_id in missing:
        cached = out_dir / f"{source_id}.json"
        if not cached.is_file():
            remaining.append(source_id)
            continue
        try:
            loaded = json.loads(cached.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HArmError(f"supplemental h_target is unreadable: {source_id}") from exc
        payload = with_authoritative_corpus(source_id, check_target_geometry(dict(loaded), source_id))
        if str(payload.get("sampling_sha256")) != str(manifest.get("sampling_sha256")):
            raise HArmError(f"backfill sampling identity differs: {source_id}")
        payloads[source_id] = payload
    if remaining:
        cache = pod_full_target_cache(inputs)
        for source_id in remaining:
            payload = with_authoritative_corpus(
                source_id, check_target_geometry(dict(cache[source_id]), source_id)
            )
            if str(payload.get("sampling_sha256")) != str(manifest.get("sampling_sha256")):
                raise HArmError(f"backfill sampling identity differs: {source_id}")
            out_dir.mkdir(parents=True, exist_ok=True)
            arm_runtime.atomic_write_json(out_dir / f"{source_id}.json", payload)
            payloads[source_id] = payload

    for source_id in needed:
        check_target_geometry(payloads[source_id], source_id)
    return payloads



def pod_selected_best(logits: Any, episode_ids: list[str | None], slot_of: dict[str, int]) -> tuple[list[float], list[float]]:
    selected: list[float] = []
    best: list[float] = []
    for index in range(len(episode_ids)):
        row = [float(v) for v in logits[index]]
        episode = episode_ids[index]
        if episode is not None and episode in slot_of:
            slot = int(slot_of[episode])
            selected.append(row[slot])
            rest = [v for j, v in enumerate(row) if j != slot]
        else:
            selected.append(row[0])
            rest = row[1:]
        best.append(max(rest))
    return selected, best


def pod_f0_from_selected(selected: list[float]) -> list[float]:
    out: list[float] = []
    for value in selected:
        p = 1.0 / (1.0 + math.exp(min(max(-float(value), -80.0), 80.0)))
        anchor = min(max(1.0 - p, 1e-6), 1.0 - 1e-6)
        out.append(math.log(anchor / (1.0 - anchor)))
    return out


def pod_head_features(ctx: dict[str, Any], hidden: Any, logits: Any, payload: dict[str, Any], slot_of: dict[str, int]) -> Any:
    torch, device = ctx["torch"], ctx["device"]
    episode_ids = [None if v is None else str(v) for v in payload["episode_ids"]]
    selected, best = pod_selected_best(logits, episode_ids, dict(slot_of))
    features = assemble_features(hidden, logits, selected, best, [1.04] * len(episode_ids))
    return torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)


def pod_source_evidence(inputs: HArmPodInputs, ctx: dict[str, Any], source_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    torch, wrapper, device = ctx["torch"], ctx["wrapper"], ctx["device"]
    import torchaudio

    from experiments.psem_state_corrected_adaptation_gate import material as material_mod
    from experiments.psem_state_corrected_adaptation_gate import stages as stages_mod

    audio, sample_rate = stages_mod.load_waveform_bytes(
        torchaudio, Path(inputs.corpus_root), payload["audio_ref"], payload["waveform_sha256"], source_id
    )
    if sample_rate != material_mod.SAMPLE_RATE_HZ or audio.ndim != 2 or audio.shape[0] != 1:
        raise HArmError(f"source waveform geometry is invalid: {source_id}")
    frame_count = int(payload["num_frames"])
    usable, tail = material_mod.slice_waveform_frames(int(audio.shape[1]), frame_count, source_id)
    waveform = audio[:, :usable].to(device)
    passage = material_mod.run_adjacent_windows(torch, wrapper, waveform, material_mod.CHUNK_FRAMES, True)
    evidence = material_mod.concat_windows(torch, passage["windows"])
    material_mod.require_frame_alignment(evidence["hidden"].shape[1], frame_count, source_id)
    episode_ids = [None if v is None else str(v) for v in payload["episode_ids"]]
    anchor_active = [a == 1.0 for a in payload["y_anchor"]]
    valid = [bool(v) for v in payload["valid"]]
    probabilities = evidence["probabilities"][0].detach().cpu().tolist()
    slot_of, mapping_rows, unmapped_frames = material_mod.oracle_slot_mapping(
        episode_ids, anchor_active, valid, probabilities
    )
    hidden = evidence["hidden"][0].detach().cpu().float().numpy()
    logits = evidence["logits"][0].detach().cpu().float().numpy()
    return {
        "hidden192": np.asarray(hidden, dtype=np.float32),
        "slot_logits4": np.asarray(logits, dtype=np.float32),
        "slot_of": {str(k): int(v) for k, v in dict(slot_of).items()},
        "mapping_rows": [dict(r) for r in mapping_rows],
        "timing": {
            "tail_excluded": int(tail),
            "unmapped_frames": [int(i) for i in unmapped_frames],
            "windows": int(len(passage["windows"])),
            "boundary_steps": [int(s) for s in passage.get("boundary_steps", [])],
        },
    }


def sum_session_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise HArmError("group metrics are empty")
    active = sum(float(r.get("active_speech_seconds", 0.0)) for r in rows)
    if active <= 0:
        raise HArmError("group metrics lack active speech")
    return {
        "active_speech_seconds": active,
        "reference_replacement_count": sum(int(r.get("reference_replacement_count", 0)) for r in rows),
        "false_cut_count": sum(int(r.get("false_cut_count", 0)) for r in rows),
        "missed_replacement_count": sum(int(r.get("missed_replacement_count", 0)) for r in rows),
        "exclusive_other_contamination_seconds": sum(
            float(r.get("exclusive_other_contamination_seconds", 0.0)) for r in rows
        ),
        "exclusive_other_contamination_seconds_per_active_speech_hour": sum(
            float(r.get("exclusive_other_contamination_seconds", 0.0)) for r in rows
        )
        / (active / 3600.0),
    }


def union_probability_grid(prob_lists: Sequence[Sequence[float]]) -> list[float]:
    union: set[float] = set()
    for scores in prob_lists:
        for value in [float(v) for v in scores]:
            if value != float("-inf"):
                union.add(value)
    if not union:
        raise HArmError("DEV group has no frontier scores")
    return sorted(union, reverse=True)


def pod_dev_tables(
    inputs: HArmPodInputs, ctx: dict[str, Any], calibrators: dict[str, Any]
) -> tuple[Any, Any, Any]:
    from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import load_sessions
    from experiments.psem_sortformer_adaptation_depth.execution import load_scoring_sessions
    from experiments.psem_state_corrected_adaptation_gate import material as material_mod
    from experiments.psem_state_corrected_adaptation_gate import stages as stages_mod
    from experiments.psem_training_strategy_gate.sampling import DEV_ROLE

    torch, wrapper, head_module, device = ctx["torch"], ctx["wrapper"], ctx["head"], ctx["device"]
    f0_slope, f0_intercept = dev_calibrator_pair(calibrators, "f0")
    cand_slope, cand_intercept = dev_calibrator_pair(calibrators, "candidate")
    was_training = bool(head_module.training)
    head_module.eval()
    try:
        dev_runtime = load_scoring_sessions(Path(inputs.corpus_root), Path(inputs.reference_root), DEV_ROLE)
        dev_sessions = load_sessions()
        scores: dict[str, Any] = {}
        tables: dict[str, Any] = {}
        corpus_of: dict[str, str] = {}
        workers = arm_runtime.resolve_workers(inputs.workers)
        members: list[dict[str, Any]] = []
        with torch.no_grad():
            for family in stages_mod.DEV_FAMILIES:
                candidates = [s for s in dev_sessions if material_mod.is_dev_family_session(s, family)]
                dev = candidates[0]
                raw = material_mod.infer_dev_raw_logits(
                    torch, wrapper, head_module, dev, dev_runtime[dev.source_id], Path(inputs.corpus_root), device
                )
                f0_raw = [float(v) for v in raw["f0_raw"]]
                cand_raw = [float(v) for v in raw["cand_raw"]]
                target = [float(v) for v in raw["target"]]
                mapped = [bool(v) for v in raw["mapped_flags"]]
                f0_cal = calibrate_mod.apply_affine(f0_raw, f0_slope, f0_intercept)
                cand_cal = calibrate_mod.apply_affine(cand_raw, cand_slope, cand_intercept)
                f0_prob = mask_unmapped(sigmoid_list(f0_raw), mapped)
                cand_raw_prob = mask_unmapped(sigmoid_list(cand_raw), mapped)
                cand_cal_prob = mask_unmapped(sigmoid_list(cand_cal), mapped)
                scores[dev.source_id] = {
                    "f0": f0_raw, "candidate": cand_raw, "target": target, "mapped": mapped,
                    "frames": len(target),
                    "f0_cal_logit": f0_cal, "candidate_cal_logit": cand_cal,
                }
                corpus_of[dev.source_id] = "AMI" if "ami" in family else "AliMeeting"
                members.append({
                    "source_id": dev.source_id,
                    "corpus": corpus_of[dev.source_id],
                    "dev": dev,
                    "f0_prob": f0_prob,
                    "cand_raw_prob": cand_raw_prob,
                    "cand_cal_prob": cand_cal_prob,
                })
                tables[dev.source_id] = {}
        from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

        member_scores = {
            m["source_id"]: {
                "dev": m["dev"],
                "scores": {
                    "raw": [float(v) for v in m["cand_raw_prob"]],
                    "calibrated": [float(v) for v in m["cand_cal_prob"]],
                    "f0": [float(v) for v in m["f0_prob"]],
                },
            }
            for m in members
        }
        grids = {
            "raw": union_probability_grid([m["cand_raw_prob"] for m in members]),
            "calibrated": union_probability_grid([m["cand_cal_prob"] for m in members]),
            "f0": [0.5],
        }
        wave_grids = {
            sid: {kind: list(grids[kind]) for kind in ("raw", "calibrated", "f0")}
            for sid in member_scores
        }
        tasks = cross_mod.plan_exact_tasks(wave_grids, list(frontier_mod.HORIZONS_MS))
        primitives, wave_receipt = cross_mod.run_exact_wave(member_scores, tasks, workers)
        corpora = {
            "AMI": [m["source_id"] for m in members if m["corpus"] == "AMI"],
            "AliMeeting": [m["source_id"] for m in members if m["corpus"] == "AliMeeting"],
        }
        corpora["pooled"] = sorted(set(corpora["AMI"]) | set(corpora["AliMeeting"]))

        def _point(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
            try:
                totals = cross_mod.sum_primitives(rows)
                return cross_mod.pooled_point_from_sums(
                    totals["false_cut_count"],
                    totals["active_speech_seconds"],
                    totals["reference_replacement_count"],
                    totals["missed_replacement_count"],
                    totals["exclusive_other_contamination_seconds"],
                    float(threshold),
                )
            except cross_mod.CrossFrontierError as exc:
                raise HArmError(f"group frontier failed: {exc}") from exc

        for sid in member_scores:
            tables[sid] = {}
            for horizon_ms in frontier_mod.HORIZONS_MS:
                by_kind = {}
                for kind in ("raw", "calibrated"):
                    rows_by_threshold = {float(p["threshold"]): p for p in primitives[sid][kind][int(horizon_ms)]}
                    kind_map = {}
                    for t in grids[kind]:
                        point = _point([rows_by_threshold[float(t)]], float(t))
                        kind_map[float(t)] = (
                            float(point["false_cuts_per_hour"]),
                            float(point["contamination"]),
                            float(point["miss_rate"]),
                        )
                    by_kind[kind] = kind_map
                f0_point = _point(primitives[sid]["f0"][int(horizon_ms)], 0.5)
                tables[sid][int(horizon_ms)] = {
                    "f0": (
                        float(f0_point["false_cuts_per_hour"]),
                        float(f0_point["contamination"]),
                        float(f0_point["miss_rate"]),
                    ),
                    "by_threshold_raw": by_kind["raw"],
                    "by_threshold_calibrated": by_kind["calibrated"],
                }
        groups: dict[str, Any] = {}
        thresholds_plan: dict[str, Any] = {}
        member_index: dict[str, Any] = {}
        for sid in member_scores:
            member_index[sid] = {}
            for kind in ("raw", "calibrated", "f0"):
                member_index[sid][kind] = {}
                for horizon_ms in frontier_mod.HORIZONS_MS:
                    member_index[sid][kind][int(horizon_ms)] = cross_mod.index_threshold_rows(
                        primitives[sid][kind][int(horizon_ms)]
                    )
        for group_name in ("AMI", "AliMeeting", "pooled"):
            group_members = corpora[group_name]
            if not group_members:
                continue
            groups[group_name] = {}
            thresholds_plan[group_name] = {}
            for horizon_ms in frontier_mod.HORIZONS_MS:
                f0_point = _point(
                    [member_index[sid]["f0"][int(horizon_ms)][0.5] for sid in group_members], 0.5
                )
                kind_blocks = {}
                for kind in ("raw", "calibrated"):
                    kind_points = []
                    for threshold in grids[kind]:
                        rows = [
                            member_index[sid][kind][int(horizon_ms)][float(threshold)]
                            for sid in group_members
                        ]
                        point = _point(rows, float(threshold))
                        kind_points.append(
                            (
                                float(threshold),
                                float(point["false_cuts_per_hour"]),
                                float(point["contamination"]),
                                float(point["miss_rate"]),
                            )
                        )
                    kind_blocks[kind] = {
                        "thresholds": [float(t) for t in grids[kind]],
                        "points": kind_points,
                        "f0": (
                            float(f0_point["false_cuts_per_hour"]),
                            float(f0_point["contamination"]),
                            float(f0_point["miss_rate"]),
                        ),
                    }
                    thresholds_plan[group_name][kind] = len(grids[kind])
                groups[group_name][int(horizon_ms)] = {"kinds": kind_blocks}
        phase = {
            **wave_receipt,
            "thresholds": thresholds_plan,
        }
    finally:
        head_module.train(was_training)
    return scores, tables, corpus_of, groups, phase


_CARRIED_CACHE: dict[str, Any] | None = None


def pod_target_tensors(ctx: dict[str, Any], run_dir: Path, binding: dict[str, Any], source_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    carried = _CARRIED_CACHE
    if carried is not None:
        try:
            if (
                str(carried.get("run_dir")) == str(Path(run_dir).resolve())
                and str(carried.get("binding_hash")) == arm_runtime.canonical_sha256(dict(binding))
                and str(source_id) in dict(carried.get("table", {}))
            ):
                return pod_target_tensors_from_cache(
                    ctx, dict(carried["table"])[str(source_id)], source_id, payload
                )
        except (ValueError, TypeError, KeyError, OSError):
            pass
    cached = read_source_cache(Path(run_dir), source_id, dict(binding))
    return pod_target_tensors_from_cache(ctx, cached, source_id, payload)


def pod_target_tensors_from_cache(ctx: dict[str, Any], cached: dict[str, Any], source_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    torch, device = ctx["torch"], ctx["device"]
    slot_of = {str(k): int(v) for k, v in dict(cached["meta"]["slot_of"]).items()}
    episode_ids = [None if v is None else str(v) for v in payload["episode_ids"]]
    selected, _ = pod_selected_best(cached["slot_logits4"], episode_ids, slot_of)
    f0 = pod_f0_from_selected(selected)
    unmapped = set(int(i) for i in cached["meta"].get("timing", {}).get("unmapped_frames", []))
    mult = target_mult_list(payload, unmapped)
    return {
        "num_frames": int(payload["num_frames"]),
        "y_replace": torch.as_tensor([payload["y_replace"]], dtype=torch.float32, device=device),
        "y_anchor": torch.as_tensor([payload["y_anchor"]], dtype=torch.float32, device=device),
        "mult_weight": torch.as_tensor([mult], dtype=torch.float32, device=device),
        "f0": torch.as_tensor([f0], dtype=torch.float32, device=device),
    }


def pod_calib_raw(inputs: HArmPodInputs, ctx: dict[str, Any], run_dir: Path, binding: dict[str, Any], manifest: dict[str, Any] | None = None, cached: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, list[float]]]:
    torch, head_module, _device = ctx["torch"], ctx["head"], ctx["device"]
    resolved_manifest = dict(manifest) if manifest is not None else pod_stage_manifest(inputs)
    _, calib = fit_calib_from_bundle(resolved_manifest)
    out: dict[str, dict[str, list[float]]] = {}
    was_training = bool(head_module.training)
    head_module.eval()
    try:
        out = _pod_calib_raw_inner(torch, head_module, inputs, ctx, run_dir, binding, calib, resolved_manifest, cached=cached)
    finally:
        head_module.train(was_training)
    return out


def _pod_calib_raw_inner(
    torch: Any, head_module: Any, inputs: HArmPodInputs, ctx: dict[str, Any],
    run_dir: Path, binding: dict[str, Any], calib: list[str],
    manifest: dict[str, Any] | None = None,
    cached: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, list[float]]]:
    carried = dict(cached or {})
    out: dict[str, dict[str, list[float]]] = {}
    with torch.no_grad():
        for source_id in sorted(calib):
            started = time.perf_counter()
            cached = carried.get(source_id)
            if cached is None:
                cached = read_source_cache(Path(run_dir), source_id, dict(binding))
            payload = pod_payload_for(inputs, Path(run_dir), dict(manifest) if manifest is not None else pod_stage_manifest(inputs), source_id)
            slot_of = {str(k): int(v) for k, v in dict(cached["meta"]["slot_of"]).items()}
            features = pod_head_features(ctx, cached["hidden192"], cached["slot_logits4"], payload, slot_of)
            frames = int(payload["num_frames"])
            bounds = streaming_mod.chunk_boundaries(frames, CHUNK_FRAMES)
            gru_state: Any = None
            resid_parts: list[Any] = []
            for index, (start, end) in enumerate(bounds):
                outputs, gru_state = head_module(features[:, start:end], gru_state)
                resid_parts.append(outputs["z_residual"])
                if index < len(bounds) - 1:
                    gru_state = streaming_mod.StateCarrier(gru_state, source_id).detach()
            z = torch.cat(resid_parts, dim=1)[0].detach().cpu().tolist()
            del resid_parts
            episode_ids = [None if v is None else str(v) for v in payload["episode_ids"]]
            selected, _ = pod_selected_best(cached["slot_logits4"], episode_ids, slot_of)
            f0 = pod_f0_from_selected(selected)
            mapping_rows = [dict(r) for r in cached["meta"].get("mapping_rows", [])]
            mapped_rows = [r for r in mapping_rows if r.get("status") == "mapped"]
            unmapped = [int(i) for i in cached["meta"].get("timing", {}).get("unmapped_frames", ())]
            mapped_flags = [i not in set(unmapped) for i in range(frames)]
            valid_flags = [bool(v) for v in payload["valid"]]
            target = [float(v) for v in payload["y_replace"]]
            from experiments.psem_state_corrected_adaptation_gate.material import mask_calibration

            kept, coverage_counts = mask_calibration(target, valid_flags, mapped_flags)
            family = str(payload.get("source_family") or "") or corpus_family(
                resolve_source_corpus(source_id, payload)
            )

            out[source_id] = {
                "f0": [float(v) for v in f0],
                "candidate": [float(a + b) for a, b in zip(f0, z)],
                "target": target,
                "valid": valid_flags,
                "mapped": mapped_flags,
                "coverage": float(len(mapped_rows) / max(1, len(mapping_rows))),
                "coverage_counts": dict(coverage_counts),
                "agreement": 1.0,
                "mapping_mapped": int(len(mapped_rows)),
                "mapping_total": int(len(mapping_rows)),
                "unmapped_frames": int(len(unmapped)),
                "kept_frames": int(len(kept)),
                "infer_seconds": time.perf_counter() - started,
                "family": family,
                "frames": frames,
            }
            del features, gru_state, cached, payload, z, f0, selected

    return out
def export_trained_gpu_evidence(
    config: arm_runtime.ArmRunConfig,
    inputs: HArmPodInputs,
    ctx: dict[str, Any],
    manifest: dict[str, Any],
    run_dir: Path | None = None,
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate.material import infer_dev_raw_logits

    run_dir = Path(run_dir or config.run_dir())
    export_dir = gpu_export_dir(run_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    torch, wrapper, head_module, device = ctx["torch"], ctx["wrapper"], ctx["head"], ctx["device"]
    fit, calib = fit_calib_from_bundle(manifest)
    forbid_eval(list(fit) + list(calib))
    binding = dict(config.binding)

    head_digest = trained_head_digest(torch, head_module)
    metrics_path = Path(run_dir) / TRAINING_METRICS_NAME
    if not metrics_path.is_file():
        raise HArmError("training metrics are missing for GPU export")
    train_metrics = _read_json_object(metrics_path)
    frozen_seconds = {
        str(k): float(v)
        for k, v in dict(train_metrics.get("evidence_cache", {}).get("inference_seconds", {})).items()
    }
    metrics_sha = arm_runtime.sha256_file(metrics_path)
    expected_identity = gpu_export_identity(binding, head_digest, metrics_sha)
    members = load_dev_export_population(Path(inputs.corpus_root), Path(inputs.reference_root))
    forbid_eval([m["source_id"] for m in members])
    existing_manifest = gpu_export_manifest_path(run_dir)
    if existing_manifest.is_file():
        body = _read_json_object(existing_manifest)
        if body.get("artifact_role") != GPU_EXPORT_ARTIFACT_ROLE:
            raise HArmError("GPU export manifest binding differs")
        require_gpu_export_identity(body, expected_identity, "completed GPU export")
        if sorted(body.get("calib_sources") or []) != sorted(calib):
            raise HArmError("GPU export CALIB population differs")
        if sorted(body.get("dev_sources") or []) != [m["source_id"] for m in members]:
            raise HArmError("GPU export DEV population differs")
        files = dict(body.get("files") or {})
        for rel, digest in files.items():
            path = export_dir / str(rel)
            if arm_runtime.sha256_file(path) != str(digest):
                raise HArmError(f"GPU export file is corrupt: {rel}")
            load_aligned_export_npz(path)
        return body
    progress = load_export_progress(run_dir, binding, head_digest, metrics_sha)

    completed = dict(progress["completed"])
    existing_npz = sorted(p.name for p in export_dir.glob("*.npz"))
    if existing_npz and not completed:
        raise HArmError("GPU export files exist without a matching resume ledger")
    for filename in existing_npz:
        record = completed.get(filename)
        if not isinstance(record, dict):
            raise HArmError(f"GPU export file exists without a matching ledger hash: {filename}")
        resume_export_npz(export_dir, filename, str(record["sha256"]))

    def _commit(filename: str, digest: str, entry: dict[str, Any]) -> None:
        completed[filename] = {"sha256": digest, "entry": dict(entry)}
        arm_runtime.atomic_write_json(
            gpu_export_progress_path(run_dir),
            {"identity": progress["identity"], "completed": completed},
        )

    calib_table: dict[str, dict[str, Any]] = {}
    dev_table: dict[str, dict[str, Any]] = {}
    was_training = bool(head_module.training)
    head_module.eval()
    try:
        pending_calib = [s for s in sorted(calib) if export_npz_name("calib", s) not in completed]
        calib_raw = {}
        if pending_calib:
            calib_raw = _pod_calib_raw_inner(
                torch, head_module, inputs, ctx, run_dir, h_binding(config, manifest), pending_calib, manifest
            )
        for source_id in sorted(calib):
            filename = export_npz_name("calib", source_id)
            record = completed.get(filename)
            resumed = resume_export_npz(
                export_dir, filename, None if record is None else str(record["sha256"])
            )
            if resumed is not None and record is not None:
                calib_table[source_id] = dict(record["entry"])
                continue
            entry_raw = calib_raw[source_id]
            digest = write_aligned_export_npz(
                export_dir / filename,
                entry_raw["f0"],
                entry_raw["candidate"],
                entry_raw["target"],
                entry_raw["valid"],
                entry_raw["mapped"],
            )
            entry = export_source_entry(
                "calib",
                source_id,
                int(entry_raw["frames"]),
                str(entry_raw["family"]),
                int(entry_raw["mapping_mapped"]),
                int(entry_raw["mapping_total"]),
                int(entry_raw["unmapped_frames"]),
                int(entry_raw["kept_frames"]),
                dict(entry_raw["coverage_counts"]),
                float(entry_raw["infer_seconds"]),
            )
            _commit(filename, digest, entry)
            calib_table[source_id] = entry
        with torch.no_grad():
            for member in members:
                source_id = member["source_id"]
                filename = export_npz_name("dev", source_id)
                record = completed.get(filename)
                resumed = resume_export_npz(
                    export_dir, filename, None if record is None else str(record["sha256"])
                )
                if resumed is not None and record is not None:
                    dev_table[source_id] = dict(record["entry"])
                    continue
                raw = infer_dev_raw_logits(
                    torch,
                    wrapper,
                    head_module,
                    member["snapshot"],
                    member["session"],
                    Path(inputs.corpus_root),
                    device,
                )
                digest = write_aligned_export_npz(
                    export_dir / filename,
                    raw["f0_raw"],
                    raw["cand_raw"],
                    raw["target"],
                    raw["valid"],
                    raw["mapped_flags"],
                )
                entry = export_source_entry(
                    "dev",
                    source_id,
                    int(raw["grid_frames"]),
                    str(member["family"]),
                    int(raw["mapping_mapped"]),
                    len(raw["mapping_rows"]),
                    len(raw["unmapped_frames"]),
                    int(raw["coverage"]["kept"]),
                    dict(raw["coverage"]),
                    float(raw["infer_seconds"]),
                )
                _commit(filename, digest, entry)
                dev_table[source_id] = entry
    finally:
        head_module.train(was_training)
    return write_gpu_export_manifest(
        run_dir,
        config,
        calib_table,
        dev_table,
        frozen_seconds,
        metrics_path,
        fit=fit,
        salt=str(manifest.get("salt", "")),
        target_frac=float(manifest.get("target_frac", 0.0)),
        trained_head_sha256=head_digest,
    )







def pod_planned_train_steps(
    manifest: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    run_dir: Path | None = None,
    binding: dict[str, Any] | None = None,
) -> int:
    fit, _ = fit_calib_from_bundle(manifest)
    missing = [s for s in fit if s not in payloads]
    if missing:
        raise HArmError(f"planned steps lack exact target geometry: {missing}")
    flags: dict[str, list[bool]] = {}
    for source_id in fit:
        payload = check_target_geometry(dict(payloads[source_id]), source_id)
        unmapped: Any = ()
        if run_dir is not None and binding is not None:
            if cache_meta_path(Path(run_dir), source_id).is_file() or cache_npz_path(Path(run_dir), source_id).is_file():
                unmapped = cache_unmapped_frames(Path(run_dir), source_id)
        flags[source_id] = loss_flags_from_mult_list(
            int(payload["num_frames"]), target_mult_list(payload, unmapped)
        )
    return arm_runtime.compute_total_steps(sum(sum(1 for f in v if f) for v in flags.values()))


def frozen_root_for(run_dir: Path, root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    return Path(run_dir).resolve().parent.parent


def pod_profile_batch(inputs: HArmPodInputs, ctx: dict[str, Any], run_dir: Path, binding: dict[str, Any], manifest: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    torch, device = ctx["torch"], ctx["device"]
    frozen_root = frozen_root_for(run_dir, root)
    fit, calib = fit_calib_from_bundle(manifest)
    unbuilt: list[str] = []
    for source_id in arm_runtime.chronological_sources(sorted(set(fit) | set(calib))):
        payload = check_target_geometry(
            dict(pod_payload_for(inputs, Path(run_dir), manifest, source_id)), source_id
        )
        identity = frozen_evidence_identity(binding, payload, source_id)
        if frozen_hit_meta(frozen_root, identity) is None:
            unbuilt.append(source_id)
    for source_id in arm_runtime.chronological_sources(list(fit)):
        payload = check_target_geometry(
            dict(pod_payload_for(inputs, Path(run_dir), manifest, source_id)), source_id
        )
        identity = frozen_evidence_identity(binding, payload, source_id)
        hit_meta = frozen_hit_meta(frozen_root, identity)
        frozen_hit = hit_meta is not None
        evidence_started = time.perf_counter()
        if frozen_hit:
            loaded = read_frozen_evidence(frozen_root, identity)
            if loaded is None:
                raise HArmError(f"frozen evidence disappeared: {source_id}")
            evidence = {
                "hidden192": loaded["hidden192"],
                "slot_logits4": loaded["slot_logits4"],
                "slot_of": dict(loaded["meta"].get("slot_of", {})),
                "mapping_rows": list(loaded["meta"].get("mapping_rows", [])),
                "timing": dict(loaded["meta"].get("timing", {})),
            }
            frozen_meta = hit_meta
            measured_seconds = None
        else:
            evidence = pod_source_evidence(inputs, ctx, source_id, payload)
            frozen_meta = write_frozen_evidence(
                frozen_root,
                identity,
                evidence["hidden192"],
                evidence["slot_logits4"],
                dict(evidence.get("slot_of", {})),
                list(evidence.get("mapping_rows", [])),
                dict(evidence.get("timing", {})),
            )
            measured_seconds = time.perf_counter() - evidence_started

        slot_of = {str(k): int(v) for k, v in dict(evidence["slot_of"]).items()}
        unmapped = evidence.get("timing", {}).get("unmapped_frames", ())
        mult = target_mult_list(payload, unmapped)
        replace = [float(v) for v in payload["y_replace"]]
        frames = int(payload["num_frames"])
        frozen_info = {
            "source_id": source_id,
            "hit": bool(frozen_hit),
            "key": str(frozen_meta["key"]),
            "bytes": int(
                frozen_evidence_dir(frozen_root).joinpath(str(frozen_meta["key"]) + ".npz").stat().st_size
            ),
            "measured_seconds": measured_seconds,
            "fit_sources": list(fit),
            "calib_sources": list(calib),
            "unbuilt_sources": list(unbuilt),
        }

        for start, end in streaming_mod.chunk_boundaries(frames, CHUNK_FRAMES):
            window_mult = sum(mult[i] for i in range(start, end))
            window_replace = sum(replace[i] for i in range(start, end) if mult[i] > 0)
            if window_mult > 0 and window_replace > 0:
                features = pod_head_features(
                    ctx, evidence["hidden192"], evidence["slot_logits4"], payload, slot_of
                )
                selected, _ = pod_selected_best(
                    evidence["slot_logits4"],
                    [None if v is None else str(v) for v in payload["episode_ids"]],
                    slot_of,
                )
                f0 = pod_f0_from_selected(selected)
                try:
                    cache_bytes = int(features.numel() * 4)
                except (RuntimeError, AttributeError, TypeError):
                    cache_bytes = int(payload["num_frames"]) * 1280 * 2
                return {
                    "source_id": source_id,
                    "chunk_start": start,
                    "chunk_end": end,
                    "frozen_evidence": frozen_info,
                    "cache_bytes": cache_bytes,
                    "features": features,
                    "y_replace": torch.as_tensor([replace], dtype=torch.float32, device=device),
                    "y_anchor": torch.as_tensor(
                        [[float(v) for v in payload["y_anchor"]]], dtype=torch.float32, device=device
                    ),
                    "mult_weight": torch.as_tensor([mult], dtype=torch.float32, device=device),
                    "f0": torch.as_tensor([f0], dtype=torch.float32, device=device),
                    "windows": streaming_mod.chunk_boundaries(frames, CHUNK_FRAMES),
                    "io_bytes": int(payload["num_frames"]) * 1280 * 2,
                }
    raise HArmError("profile batch lacks valid mapped multiplicity with replacement support")

def score_frontier_slice(
    decode_scores: Any,
    session_metrics: Any,
    dev: Any,
    scores: list[float],
    horizon_ms: int,
    limit: int | None = None,
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

    prob_list = [float(s) for s in scores]
    grid = frontier_mod.unique_thresholds(prob_list)
    sliced = cross_mod.bounded_threshold_slice(
        grid, cross_mod.FRONTIER_SLICE_LIMIT if limit is None else int(limit)
    )
    import numpy as np

    start = time.perf_counter()
    points = []
    for threshold in sliced:
        events = decode_scores(
            dev,
            np.asarray(prob_list, dtype=np.float64),
            threshold=float(threshold),
            confirmation_ms=int(horizon_ms),
        )
        metrics = session_metrics(dev, events)
        active_hours = float(metrics["active_speech_seconds"]) / 3600.0
        points.append(
            {
                "threshold": float(threshold),
                "false_cuts_per_hour": float(metrics["false_cut_count"]) / active_hours,
                "contamination": float(
                    metrics["exclusive_other_contamination_seconds_per_active_speech_hour"]
                ),
                "miss_rate": float(metrics["missed_replacement_count"])
                / float(metrics["reference_replacement_count"]),
            }
        )
    seconds = time.perf_counter() - start
    return {
        "horizon_ms": int(horizon_ms),
        "sampled_thresholds": len(sliced),
        "total_thresholds": len(grid),
        "points": points,
        "seconds": float(seconds),
        "projected_seconds": cross_mod.project_frontier_cost(seconds, len(sliced), len(grid)),
    }


def pod_profile_dev_sample(inputs: HArmPodInputs, ctx: dict[str, Any]) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import material as material_mod

    torch, wrapper, head_module, device = ctx["torch"], ctx["wrapper"], ctx["head"], ctx["device"]
    members = load_dev_export_population(Path(inputs.corpus_root), Path(inputs.reference_root))
    member = members[0]
    raw = material_mod.infer_dev_raw_logits(
        torch,
        wrapper,
        head_module,
        member["snapshot"],
        member["session"],
        Path(inputs.corpus_root),
        device,
    )
    from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
        decode_scores,
        session_metrics,
    )

    mapped = [bool(v) for v in raw["mapped_flags"]]
    cand_prob = mask_unmapped(sigmoid_list([float(v) for v in raw["cand_raw"]]), mapped)
    frontier_slice = score_frontier_slice(decode_scores, session_metrics, member["snapshot"], cand_prob, 100)
    return {
        "infer_seconds": float(raw["infer_seconds"]),
        "io_bytes": int(raw["grid_frames"]) * 1280 * 2,
        "frontier_slice": frontier_slice,
        "dev_sources": [m["source_id"] for m in members],
        "dev_population": len(members),
        "cpu_tail": {
            "kind": "estimated",
            "representative_slice": dict(frontier_slice),
            "seconds": float(frontier_slice["seconds"]),
            "projected_seconds": float(frontier_slice["projected_seconds"]),
        },
    }




def pod_deps(
    config: arm_runtime.ArmRunConfig,
    inputs: HArmPodInputs,
    manifest: dict[str, Any] | None = None,
    payloads: dict[str, dict[str, Any]] | None = None,
) -> HArmDeps:
    state: dict[str, Any] = {}
    if manifest is not None:
        state["manifest"] = dict(manifest)
    if payloads is not None:
        state["payloads"] = {str(k): dict(v) for k, v in dict(payloads).items()}

    def _manifest() -> dict[str, Any]:
        if "manifest" not in state:
            state["manifest"] = pod_stage_manifest(inputs)
        return state["manifest"]

    def _payload_table() -> dict[str, dict[str, Any]]:
        if "payloads" not in state:
            state["payloads"] = pod_ensure_targets(inputs, config.run_dir(), _manifest())
        return state["payloads"]

    def _ctx() -> dict[str, Any]:
        if "ctx" not in state:
            state["ctx"] = pod_model_context(inputs, int(config.seed))
        return state["ctx"]

    def _binding() -> dict[str, Any]:
        return h_binding(config, _manifest())

    return HArmDeps(
        load_bundle_manifest=_manifest,
        bundle_dir=Path(inputs.bundle_dir),
        build_missing_targets=lambda s: dict(_payload_table()[s]),
        build_evidence=lambda s, p: pod_source_evidence(inputs, _ctx(), s, p),
        build_features=lambda c: pod_head_features(
            _ctx(), c["hidden192"], c["slot_logits4"], c["target"],
            (
                {str(k): int(v) for k, v in dict(c["slot_of"]).items()}
                if isinstance(c.get("slot_of"), dict)
                else {str(k): int(v) for k, v in dict(read_source_cache(config.run_dir(), c["target"]["source_id"], _binding())["meta"]["slot_of"]).items()}
            ),
        ),
        build_targets=lambda s, p: pod_target_tensors(_ctx(), config.run_dir(), _binding(), s, p),
        load_wrapper_head=lambda: (_ctx()["wrapper"], _ctx()["head"]),
        load_torch=lambda: _ctx()["torch"],
        workers=inputs.workers,
        hourly_cost_usd=float(inputs.hourly_cost_usd),
        profile_batch=lambda head_module, wrapper, torch: pod_profile_batch(inputs, _ctx(), config.run_dir(), _binding(), _manifest(), config.root),
        profile_dev_sample=lambda head_module, wrapper, torch: pod_profile_dev_sample(inputs, _ctx()),
        export_gpu_evidence=lambda head_module, wrapper, torch: export_trained_gpu_evidence(
            config, inputs, _ctx(), _manifest(), config.run_dir()
        ),

        total_profile_steps=pod_planned_train_steps(
            _manifest(), _payload_table(), config.run_dir(), h_binding(config, _manifest())
        ),
    )


def pod_ensure_targets_entry(inputs: HArmPodInputs, run_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return pod_ensure_targets(inputs, Path(run_dir), dict(manifest))


def run_h_arm_pod(config: arm_runtime.ArmRunConfig, store: Path, inputs: HArmPodInputs) -> dict[str, Any]:
    authorization = check_authorized(config, Path(store))
    manifest = pod_stage_manifest(inputs)
    binding_report = verify_config_binding(config, manifest, inputs)
    payloads = pod_ensure_targets(inputs, config.run_dir(), manifest)
    with arm_runtime.arm_gpu_lock(
        config.root, {"run_id": f"{config.arm}-{config.seed}", "arm": config.arm, "seed": int(config.seed), "command": "run"}
    ):
        out = run_h_arm(config, Path(store), pod_deps(config, inputs, manifest, payloads))
    out["binding_report"] = binding_report
    out["authorization"] = authorization
    return out


def run_profile_pod(config: arm_runtime.ArmRunConfig, store: Path, inputs: HArmPodInputs) -> dict[str, Any]:
    authorization = check_authorized(config, Path(store))
    manifest = pod_stage_manifest(inputs)
    binding_report = verify_config_binding(config, manifest, inputs)
    payloads = pod_ensure_targets(inputs, config.run_dir(), manifest)
    with arm_runtime.arm_gpu_lock(
        config.root, {"run_id": f"{config.arm}-{config.seed}", "arm": config.arm, "seed": int(config.seed), "command": "profile"}
    ):
        out = run_profile_command(config, Path(store), pod_deps(config, inputs, manifest, payloads))
    out["binding_report"] = binding_report
    out["authorization"] = authorization
    return out
