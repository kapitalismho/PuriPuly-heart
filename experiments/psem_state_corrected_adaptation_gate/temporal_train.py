from __future__ import annotations

import io
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

CHUNK_FRAMES = 375
CHUNK_SECONDS = 30.0
FRAME_SAMPLES = 1280
HEAD_INPUT_DIM = 199
ANCHOR_LOSS_WEIGHT = 0.5
NATIVE_LOSS_WEIGHT = 0.5
PROFILE_MIN_STEPS = 8
PROFILE_MAX_STEPS = 16
CALIB_ROLE = "TRAIN-CALIB"
EVAL_ROLE = "PSEM-STRATEGY-EVAL"
TRAIN_NATIVE_ROLES = ("PSEM-STRATEGY-TRAIN",)
ACCUMULATION_DEFAULT = 16
WAVEFORM_PREFETCH_MAX = 4
_DECODE_CACHE_KEY: str | None = None
_DECODE_CACHE_WAVEFORM: Any = None
_DECODE_CACHE_NUM_FRAMES = 0
_DECODE_CACHE_TAIL_EXCLUDED = 0
LEGACY_POLICY_ARM = {
    "R-T2-SC": "T2-TOP",
    "R-TA-SC": "TA-ALL-TEMPORAL",
}
TEMPORAL_ARMS = tuple(LEGACY_POLICY_ARM)

FROZEN_ENCODER_PREFIXES = (
    "sortformer.encoder.",
    "sortformer.frontend_encoder.",
    "encoder.",
    "frontend_encoder.",
)
FROZEN_ENCODER_SUBSTRINGS = ("acoustic", "nest")


class TemporalArmError(RuntimeError):
    pass


def require_temporal_arm(arm: str) -> str:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    if arm not in TEMPORAL_ARMS:
        raise arm_runtime.AuthorizationError(f"temporal runner rejects arm: {arm}")
    return arm


def legacy_policy_arm(arm: str) -> str:
    require_temporal_arm(arm)
    return LEGACY_POLICY_ARM[arm]


_POLICY_MODULE: Any = None


def _policy() -> Any:
    global _POLICY_MODULE
    if _POLICY_MODULE is None:
        import importlib.util

        path = (
            Path(__file__).resolve().parent.parent
            / "psem_sortformer_adaptation_depth"
            / "parameter_policy.py"
        )
        spec = importlib.util.spec_from_file_location(
            "issue121_temporal_parameter_policy", path
        )
        if spec is None or spec.loader is None:
            raise TemporalArmError("frozen #107 parameter policy is unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _POLICY_MODULE = module
    return _POLICY_MODULE


def is_frozen_encoder_param(name: str) -> bool:
    lowered = name.lower()
    if name.startswith(FROZEN_ENCODER_PREFIXES):
        return True
    return any(sub in lowered for sub in FROZEN_ENCODER_SUBSTRINGS)


def trainable_names(names: Sequence[str], arm: str) -> list[str]:
    policy = _policy()
    legacy = legacy_policy_arm(arm)
    return [name for name in names if policy.should_train(name, legacy)]


def audit_temporal_trainability(names: Sequence[str], arm: str) -> dict[str, Any]:
    policy = _policy()
    legacy = legacy_policy_arm(arm)
    inventory = policy.audit_parameter_graph(names)
    expected = inventory["trainable_by_arm"][legacy]
    violations = [n for n in expected if is_frozen_encoder_param(n)]
    if violations:
        raise TemporalArmError(f"acoustic/NEST encoder would train: {violations[:4]}")
    return {
        "arm": arm,
        "legacy_arm": legacy,
        "trainable": list(expected),
        "temporal_layers": list(inventory["temporal_layers"]),
        "acoustic_encoder_frozen": True,
    }


def apply_temporal_parameter_policy(model: Any, arm: str) -> dict[str, Any]:
    policy = _policy()
    legacy = legacy_policy_arm(arm)
    receipt = policy.apply_parameter_policy(model, legacy)
    actual = [n for n, p in model.named_parameters() if p.requires_grad]
    violations = [n for n in actual if is_frozen_encoder_param(n)]
    if violations:
        raise TemporalArmError(
            f"acoustic/NEST encoder exposes trainable parameters: {violations[:4]}"
        )
    return {**receipt, "arm": arm, "acoustic_encoder_frozen": True}


def _module_trainable_hint(path: str, arm: str) -> bool:
    policy = _policy()
    legacy = legacy_policy_arm(arm)
    probe = f"{path}.weight" if not path.endswith(".weight") else path
    if is_frozen_encoder_param(probe):
        return False
    try:
        return bool(policy.should_train(probe, legacy))
    except Exception:
        return False


def set_temporal_module_modes(model: Any, arm: str) -> dict[str, Any]:
    require_temporal_arm(arm)
    train_paths: list[str] = []
    eval_paths: list[str] = []
    for path, module in model.named_modules():
        if not path:
            continue
        if _module_trainable_hint(path, arm):
            module.train(True)
            train_paths.append(path)
        else:
            module.train(False)
            eval_paths.append(path)
    return {"arm": arm, "train_modules": train_paths, "eval_modules": eval_paths}


def audit_temporal_module_modes(model: Any, arm: str) -> dict[str, Any]:
    require_temporal_arm(arm)
    bad_train: list[str] = []
    bad_eval: list[str] = []
    for path, module in model.named_modules():
        if not path:
            continue
        training = bool(module.training)
        if _module_trainable_hint(path, arm):
            if not training:
                bad_train.append(path)
        else:
            if training:
                bad_eval.append(path)
    if bad_train or bad_eval:
        raise TemporalArmError(
            "module modes diverge from the parameter policy: "
            f"train_expected={bad_train[:4]} eval_expected={bad_eval[:4]}"
        )
    return {"arm": arm, "modes_ok": True}


def optimizer_param_groups(
    model: Any, head_module: Any, arm: str
) -> dict[str, list[str]]:
    policy = _policy()
    groups: dict[str, list[str]] = {"head": [], "activity": [], "temporal": []}
    for name, param in head_module.named_parameters():
        if param.requires_grad:
            groups["head"].append(f"psem_head.{name}")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("psem_head.") or name in groups["head"]:
            if name not in groups["head"]:
                groups["head"].append(name)
        elif policy.is_activity_head(name):
            groups["activity"].append(name)
        else:
            groups["temporal"].append(name)
    return groups


def build_optimizer(torch: Any, model: Any, head_module: Any, arm: str) -> Any:
    require_temporal_arm(arm)
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    contract = dict(arm_runtime.OPTIMIZER_CONTRACT)
    lrs = contract["group_lrs"]
    policy = _policy()
    head_params = [p for _, p in head_module.named_parameters() if p.requires_grad]
    head_params += [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("psem_head.")
    ]
    activity_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and policy.is_activity_head(n)
    ]
    temporal_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and not policy.is_activity_head(n)
        and not n.startswith("psem_head.")
    ]
    if not head_params:
        raise TemporalArmError("residual head exposes no trainable parameters")
    grouped = [
        {"params": head_params, "lr": float(lrs["head"])},
        {"params": activity_params, "lr": float(lrs["activity"])},
        {"params": temporal_params, "lr": float(lrs["temporal"])},
    ]
    grouped = [g for g in grouped if g["params"]]
    return torch.optim.AdamW(grouped, weight_decay=float(contract["weight_decay"]))


def build_scheduler(torch: Any, optimizer: Any, total_steps: int) -> Any:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    warmup = arm_runtime.compute_warmup_steps(int(total_steps))

    def _scale(step: int) -> float:
        if warmup <= 0:
            return 1.0
        if step < warmup:
            return float(step + 1) / float(warmup)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _scale)


def assert_no_eval(roles: Sequence[str]) -> None:
    for role in roles:
        if str(role) == EVAL_ROLE or "EVAL" in str(role):
            raise TemporalArmError("EVAL is prohibited in the temporal runner")


@dataclass(slots=True)
class TemporalCarry:
    sortformer_state: Any = None
    gru_state: Any = None
    source_id: str = ""
    chunks_carried: int = 0
    stale_updates: int = 0

    def detach(self, torch: Any) -> "TemporalCarry":
        self.sortformer_state = _detach(torch, self.sortformer_state)
        self.gru_state = _detach(torch, self.gru_state)
        self.chunks_carried += 1
        return self

    def reset(self, source_id: str, sortformer_state: Any = None) -> "TemporalCarry":
        self.sortformer_state = sortformer_state
        self.gru_state = None
        self.source_id = source_id
        self.chunks_carried = 0
        return self

    def mark_update(self) -> "TemporalCarry":
        self.stale_updates += 1
        return self


@dataclass(slots=True)
class SimpleEvidence:
    probabilities: Any = None


def _detach(torch: Any, value: Any) -> Any:
    from experiments.psem_state_corrected_adaptation_gate import material

    return material._detach_state(torch, value)


_ACTIVE_ARM: str | None = None


def check_single_stream(arm: str) -> None:
    global _ACTIVE_ARM
    if _ACTIVE_ARM is not None:
        raise TemporalArmError(
            f"concurrent temporal execution refused: {_ACTIVE_ARM} active"
        )
    _ACTIVE_ARM = arm


def release_stream(arm: str) -> None:
    global _ACTIVE_ARM
    if _ACTIVE_ARM == arm:
        _ACTIVE_ARM = None


def _weighted_bce_with_logits(
    torch: Any, logit: Any, target: Any, weights: Any, pos_weight: float
) -> Any:
    relu_part = logit.clamp(0.0, float("inf"))
    stable = relu_part + torch.log1p(torch.exp(-abs(logit)))
    positive = (stable - logit) * float(pos_weight)
    per_frame = torch.where(target == 1.0, positive, stable)
    return (per_frame * weights).sum()


def temporal_chunk_loss(
    torch: Any,
    product_logit: Any,
    y_replace: Any,
    anchor_logit: Any,
    y_anchor: Any,
    mult_weight: Any,
    class_weights: Mapping[str, float],
    native_loss_value: Any | None = None,
    known_support: bool | None = None,
) -> dict[str, Any]:
    if known_support is False:
        zero = (product_logit * 0).sum() * 0
        return {"loss": zero, "empty": True}
    denom = mult_weight.sum()
    if known_support is None and float(denom) == 0:
        zero = (product_logit * 0).sum() * 0
        return {"loss": zero, "empty": True}
    replacement = (
        _weighted_bce_with_logits(
            torch,
            product_logit,
            y_replace,
            mult_weight,
            float(class_weights["replacement_positive_weight"]),
        )
        / denom
    )
    anchor = (
        _weighted_bce_with_logits(
            torch,
            anchor_logit,
            y_anchor,
            mult_weight,
            float(class_weights["anchor_positive_weight"]),
        )
        / denom
    )
    total = replacement + ANCHOR_LOSS_WEIGHT * anchor
    native_part = None
    if native_loss_value is not None:
        native_part = NATIVE_LOSS_WEIGHT * native_loss_value
        total = total + native_part
    finite = torch.isfinite(total)
    ok = bool(finite.all()) if hasattr(finite, "all") else bool(finite)
    if not ok:
        raise TemporalArmError("temporal chunk loss is non-finite")
    return {
        "loss": total,
        "replacement": replacement,
        "anchor": anchor,
        "native": native_part,
        "empty": False,
    }



@dataclass(slots=True)
class PassMapping:
    source_to_mapping: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""
    arm: str = ""
    seed: int = 0

    def for_source(self, source_id: str) -> Any:
        return self.source_to_mapping[source_id]



def mapping_worker(payload: dict[str, Any]) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import material

    slot_of, rows, unmapped = material.oracle_slot_mapping(
        list(payload["episode_ids"]),
        list(payload["anchor_active"]),
        list(payload["valid"]),
        [list(row) for row in payload["probabilities"]],
    )
    return {
        "source_id": str(payload["source_id"]),
        "mapping": {"slot_of": dict(slot_of), "rows": rows, "unmapped": unmapped},
    }


def load_torch() -> Any:
    import torch

    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    arm_runtime.enforce_thread_caps()
    return torch


def load_stage_bundle(bundle_dir: Path) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import stages

    manifest = stages.verify_bundle_manifest(Path(bundle_dir), "stage_a_manifest.json")
    if manifest.get("artifact_role") != "issue-121-stage-a-bundle":
        raise TemporalArmError("bundle is not a frozen P0 Stage-A bundle")
    if int(manifest.get("version", -1)) != int(stages.STAGE_A_VERSION):
        raise TemporalArmError("bundle version differs from the frozen Stage-A")
    for key in ("sampling_sha256", "fit", "calib", "class_weights", "targets"):
        if key not in manifest:
            raise TemporalArmError(f"bundle manifest lacks {key}")
    return manifest


CODE_IDENTITY_FILES = (
    "arm_runtime.py",
    "calibrate.py",
    "frontier.py",
    "cross_frontier.py",
    "gates.py",
    "head.py",
    "lifecycle.py",
    "material.py",
    "multiplicity.py",
    "run_temporal_arm.py",
    "stages.py",
    "temporal_train.py",
    "../psem_sortformer_adaptation_depth/parameter_policy.py",
)


def _code_identity_files() -> list[tuple[str, Path]]:
    base = Path(__file__).resolve().parent
    resolved: list[tuple[str, Path]] = []
    for name in CODE_IDENTITY_FILES:
        candidate = base / name
        if not candidate.is_file():
            raise TemporalArmError(f"code identity file is missing: {name}")
        resolved.append((name, candidate))
    return resolved


def _digest_named_files(named: Sequence[tuple[str, Path]]) -> str:
    import hashlib as _hashlib

    digest = _hashlib.sha256()
    for name, path in sorted(named, key=lambda item: item[0]):
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise TemporalArmError(
                f"code identity file is unreadable: {name}"
            ) from exc
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(data)
        digest.update(b"\x00")
    return digest.hexdigest()


def _code_identity() -> str:
    return _digest_named_files(_code_identity_files())


def verify_execution_binding(
    config: Any, manifest: Mapping[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime
    from experiments.psem_state_corrected_adaptation_gate import stages

    if str(manifest.get("sampling_sha256")) != config.input_hash:
        raise TemporalArmError("input hash differs from the frozen bundle sampling hash")
    checkpoint_sha = stages.sha256_file(Path(checkpoint_path))
    if checkpoint_sha != config.checkpoint_hash:
        raise TemporalArmError("checkpoint hash differs from the authorized binding")
    if checkpoint_sha != str(manifest.get("nemo_sha256")):
        raise TemporalArmError("checkpoint differs from the frozen bundle checkpoint")
    partition_hash = arm_runtime.canonical_sha256(
        {
            "fit": sorted(manifest["fit"]),
            "calib": sorted(manifest["calib"]),
            "salt": str(manifest.get("salt", "")),
            "target_frac": float(manifest.get("target_frac", 0.0)),
        }
    )
    if partition_hash != config.partition_hash:
        raise TemporalArmError("partition hash differs from the authorized binding")
    bound, weights_hash = arm_runtime.bind_class_weights(dict(manifest["class_weights"]))
    if weights_hash != config.weights_hash:
        raise TemporalArmError("weights hash differs from the authorized binding")
    if _code_identity() != config.code_hash:
        raise TemporalArmError("code hash differs from the authorized binding")
    return {
        "fit": sorted(manifest["fit"]),
        "calib": sorted(manifest["calib"]),
        "class_weights": bound,
        "weights_hash": weights_hash,
        "checkpoint_sha256": checkpoint_sha,
    }


TARGETS_DIRNAME = "targets"
TARGETS_MANIFEST_NAME = "targets_manifest.json"


def _validated_target_payload(payload: Any, source_id: str, what: str) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import lifecycle

    if not isinstance(payload, dict):
        raise TemporalArmError(f"persisted target is invalid: {what}")
    for key in (
        "num_frames",
        "episodes",
        "y_anchor",
        "y_replace",
        "valid",
        "multiplicity",
        "episode_ids",
    ):
        if key not in payload:
            raise TemporalArmError(f"persisted target lacks {key}: {what}")
    try:
        num_frames = int(payload["num_frames"])
    except (TypeError, ValueError) as exc:
        raise TemporalArmError(f"persisted target frame count is invalid: {what}") from exc
    if num_frames <= 0:
        raise TemporalArmError(f"persisted target frame count is invalid: {what}")
    for key in ("y_anchor", "y_replace", "valid", "multiplicity", "episode_ids"):
        values = payload[key]
        if not isinstance(values, list) or len(values) != num_frames:
            raise TemporalArmError(
                f"persisted target {key} length differs from {num_frames} frames: {what}"
            )
    episodes = payload["episodes"]
    if not isinstance(episodes, list):
        raise TemporalArmError(f"persisted target episodes are invalid: {what}")
    rebuilt: list[Any] = []
    for row in episodes:
        if not isinstance(row, dict):
            raise TemporalArmError(f"persisted target episode is invalid: {what}")
        try:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TemporalArmError(
                f"persisted target episode bounds are invalid: {what}"
            ) from exc
        if not (0 <= start < end <= num_frames):
            raise TemporalArmError(
                f"persisted target episode escapes source frames: {what}"
            )
        rebuilt.append(
            lifecycle.AnchorEpisode(
                str(row.get("episode_id", "")),
                str(row.get("anchor_speaker", "")),
                start,
                end,
            )
        )
    authority = lifecycle.SourceAuthority(
        source_id=str(source_id),
        num_frames=num_frames,
        episodes=tuple(rebuilt),
        y_anchor=tuple(float(v) for v in payload["y_anchor"]),
        y_replace=tuple(float(v) for v in payload["y_replace"]),
        valid=tuple(bool(v) for v in payload["valid"]),
        ledger=dict(payload.get("ledger", {})),
    )
    return {
        "source_id": str(source_id),
        "authority": authority,
        "multiplicity": [int(v) for v in payload["multiplicity"]],
        "episode_ids": [None if v is None else str(v) for v in payload["episode_ids"]],
        "intervals": [dict(row) for row in payload.get("intervals", [])],
        "num_frames": num_frames,
        "audio_ref": str(payload.get("audio_ref", "")),
        "waveform_sha256": str(payload.get("waveform_sha256", "")),
    }


def restore_bundle_entry(bundle_dir: Path, manifest: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    targets = manifest.get("targets", {})
    if not isinstance(targets, dict) or source_id not in targets:
        raise TemporalArmError(f"bundle target is missing: {source_id}")
    entry = targets[source_id]
    if not isinstance(entry, dict):
        raise TemporalArmError(f"bundle target entry is invalid: {source_id}")
    path = Path(bundle_dir) / str(entry.get("file", ""))
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TemporalArmError(f"bundle target is unreadable: {source_id}") from exc
    expected = str(entry.get("sha256", ""))
    if expected:
        import hashlib as _hashlib

        if _hashlib.sha256(data).hexdigest() != expected:
            raise TemporalArmError(f"bundle target hash mismatch: {source_id}")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise TemporalArmError(f"bundle target is corrupt: {source_id}") from exc
    restored = _validated_target_payload(
        payload, source_id, f"bundle:{source_id}"
    )
    if restored["num_frames"] != int(entry.get("num_frames", restored["num_frames"])):
        raise TemporalArmError(f"bundle target frame count differs: {source_id}")
    restored["audio_ref"] = str(entry.get("audio_ref", restored["audio_ref"]))
    restored["waveform_sha256"] = str(
        entry.get("waveform_sha256", restored["waveform_sha256"])
    )
    return restored
def targets_manifest_path(run_dir: Path) -> Path:
    return Path(run_dir) / TARGETS_DIRNAME / TARGETS_MANIFEST_NAME


def serialize_backfill_target(
    source_id: str,
    entry: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    authority = entry["authority"]
    try:
        ledger = json.loads(json.dumps(authority.ledger))
    except (TypeError, ValueError) as exc:
        raise TemporalArmError(
            f"backfill ledger is not JSON-safe: {source_id}"
        ) from exc
    return {
        "source_id": str(source_id),
        "num_frames": int(authority.num_frames),
        "episodes": [
            {
                "episode_id": str(episode.episode_id),
                "anchor_speaker": str(episode.anchor_speaker),
                "start_frame": int(episode.start_frame),
                "end_frame": int(episode.end_frame),
            }
            for episode in authority.episodes
        ],
        "y_anchor": [float(v) for v in authority.y_anchor],
        "y_replace": [float(v) for v in authority.y_replace],
        "valid": [bool(v) for v in authority.valid],
        "ledger": ledger,
        "multiplicity": [int(v) for v in entry["multiplicity"]],
        "episode_ids": [None if v is None else str(v) for v in entry["episode_ids"]],
        "intervals": [dict(row) for row in entry.get("intervals", [])],
        "audio_ref": str(provenance["audio_ref"]),
        "waveform_sha256": str(provenance["waveform_sha256"]),
        "sampling_sha256": str(provenance["sampling_sha256"]),
        "backfilled": True,
    }


def write_backfill_target(
    run_dir: Path, source_id: str, payload: Mapping[str, Any]
) -> tuple[Path, str]:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    target = Path(run_dir) / TARGETS_DIRNAME / f"{source_id}.json"
    path = arm_runtime.atomic_write_json(target, dict(payload))
    return path, arm_runtime.sha256_file(path)


def load_backfill_target(
    run_dir: Path,
    source_id: str,
    expected: Mapping[str, Any],
    sessions: Mapping[str, Any],
    sampling_sha256: str,
) -> dict[str, Any]:
    import hashlib as _hashlib

    target_dir = (Path(run_dir) / TARGETS_DIRNAME).resolve()
    path = (target_dir / f"{source_id}.json").resolve()
    if path.parent != target_dir or not path.is_file():
        raise TemporalArmError(f"persisted target is missing: {source_id}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TemporalArmError(f"persisted target is unreadable: {source_id}") from exc
    if _hashlib.sha256(data).hexdigest() != str(expected.get("sha256", "")):
        raise TemporalArmError(f"persisted target hash mismatch: {source_id}")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise TemporalArmError(f"persisted target is corrupt: {source_id}") from exc
    entry = _validated_target_payload(payload, source_id, f"persisted:{source_id}")
    if str(payload.get("sampling_sha256", "")) != str(sampling_sha256):
        raise TemporalArmError(f"persisted target sampling mismatch: {source_id}")
    session = sessions.get(source_id)
    if session is None:
        raise TemporalArmError(f"persisted target source left the TRAIN split: {source_id}")
    if str(payload.get("waveform_sha256", "")) != str(session.waveform_sha256):
        raise TemporalArmError(f"persisted target waveform mismatch: {source_id}")
    if str(payload.get("source_id", "")) != str(source_id):
        raise TemporalArmError(f"persisted target identity mismatch: {source_id}")
    entry["backfilled"] = True
    return entry

def backfill_target_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.psem_frozen_ceiling_gate.experiment_support import (
        simulate_gt_session,
    )
    from experiments.psem_state_corrected_adaptation_gate import material

    source_id = str(payload["source_id"])
    decoded = decode_waveform_task(
        {
            "source_id": source_id,
            "audio_ref": str(payload["audio_ref"]),
            "waveform_sha256": str(payload["waveform_sha256"]),
            "corpus_root": str(payload["corpus_root"]),
        }
    )
    try:
        entry = material.build_source_targets(
            simulate_gt_session,
            source_id,
            payload["labels"],
            [dict(r) for r in payload["rows"]],
            int(decoded["num_frames"]),
        )
        document = serialize_backfill_target(
            source_id,
            {
                "authority": entry["authority"],
                "multiplicity": list(entry["multiplicity"]),
                "episode_ids": list(entry["episode_ids"]),
                "intervals": list(entry.get("intervals", [])),
            },
            {
                "audio_ref": str(payload["audio_ref"]),
                "waveform_sha256": str(payload["waveform_sha256"]),
                "sampling_sha256": str(payload["sampling_sha256"]),
            },
        )
    finally:
        del decoded
    import json as _json

    _json.dumps(document)
    return {"source_id": source_id, "document": document}


def resolve_durable_targets(
    torch: Any,
    run_dir: Path,
    bundle_dir: Path,
    manifest: Mapping[str, Any],
    sessions: Mapping[str, Any],
    rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    corpus_root: Path,
    device: Any,
    source_ids: Sequence[str],
    workers: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    from experiments.psem_frozen_ceiling_gate.experiment_support import (
        simulate_gt_session,
    )
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime
    from experiments.psem_state_corrected_adaptation_gate import material

    sampling_sha256 = str(manifest.get("sampling_sha256", ""))
    manifest_path = targets_manifest_path(run_dir)
    stored: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            stored_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TemporalArmError(f"target manifest is unreadable: {exc}") from exc
        if not isinstance(stored_doc, dict):
            raise TemporalArmError("target manifest is invalid")
        if str(stored_doc.get("sampling_sha256", "")) != sampling_sha256:
            raise TemporalArmError("target manifest sampling mismatch")
        sources = stored_doc.get("sources", {})
        if not isinstance(sources, dict):
            raise TemporalArmError("target manifest sources are invalid")
        stored = {str(sid): dict(record) for sid, record in sources.items()}
    resolved: dict[str, dict[str, Any]] = {}
    binding: dict[str, dict[str, Any]] = {}
    bundle_targets = manifest.get("targets", {})
    bundle_files = manifest.get("files", {})
    missing: list[str] = []
    for source_id in sorted(source_ids):
        if isinstance(bundle_targets, dict) and source_id in bundle_targets:
            entry = restore_bundle_entry(bundle_dir, manifest, source_id)
            rel = str(bundle_targets[source_id].get("file", ""))
            binding[source_id] = {
                "sha256": str(bundle_targets[source_id].get("sha256", bundle_files.get(rel, ""))),
                "backfilled": False,
                "waveform_sha256": str(entry.get("waveform_sha256", "")),
            }
            resolved[source_id] = entry
            continue
        record = stored.get(source_id)
        if record is not None:
            entry = load_backfill_target(
                run_dir, source_id, record, sessions, sampling_sha256
            )
            binding[source_id] = {
                "sha256": str(record.get("sha256", "")),
                "backfilled": True,
                "waveform_sha256": str(entry.get("waveform_sha256", "")),
            }
            resolved[source_id] = entry
            continue
        if source_id not in sessions:
            raise TemporalArmError(f"backfill source outside the TRAIN split: {source_id}")
        if source_id not in rows_by_source or not rows_by_source[source_id]:
            raise TemporalArmError(f"backfill source lacks sampling rows: {source_id}")
        missing.append(str(source_id))
    if missing:
        from experiments.psem_state_corrected_adaptation_gate import arm_runtime as _rt

        n_workers = _rt.resolve_workers(workers)
        if n_workers <= 1 or len(missing) <= 1:
            for source_id in missing:
                loaded = load_source_waveform(torch, sessions[source_id], corpus_root, device)
                entry = material.build_source_targets(
                    simulate_gt_session,
                    str(source_id),
                    sessions[source_id].labels,
                    [dict(r) for r in rows_by_source[source_id]],
                    int(loaded["num_frames"]),
                )
                payload = serialize_backfill_target(
                    source_id,
                    {
                        "authority": entry["authority"],
                        "multiplicity": list(entry["multiplicity"]),
                        "episode_ids": list(entry["episode_ids"]),
                        "intervals": list(entry.get("intervals", [])),
                    },
                    {
                        "audio_ref": str(sessions[source_id].audio_ref),
                        "waveform_sha256": str(sessions[source_id].waveform_sha256),
                        "sampling_sha256": sampling_sha256,
                    },
                )
                path, digest = write_backfill_target(run_dir, source_id, payload)
                resolved[source_id] = {
                    "source_id": str(source_id),
                    "authority": entry["authority"],
                    "multiplicity": list(entry["multiplicity"]),
                    "episode_ids": list(entry["episode_ids"]),
                    "intervals": list(entry.get("intervals", [])),
                    "num_frames": int(loaded["num_frames"]),
                    "audio_ref": str(sessions[source_id].audio_ref),
                    "waveform_sha256": str(sessions[source_id].waveform_sha256),
                    "backfilled": True,
                }
                binding[source_id] = {
                    "sha256": digest,
                    "backfilled": True,
                    "waveform_sha256": str(sessions[source_id].waveform_sha256),
                }
                del loaded
        else:
            payloads = [
                {
                    "source_id": source_id,
                    "labels": sessions[source_id].labels,
                    "rows": [dict(r) for r in rows_by_source[source_id]],
                    "audio_ref": str(sessions[source_id].audio_ref),
                    "waveform_sha256": str(sessions[source_id].waveform_sha256),
                    "corpus_root": str(corpus_root),
                    "sampling_sha256": sampling_sha256,
                }
                for source_id in missing
            ]
            documents = _rt.ordered_process_map(backfill_target_task, payloads, n_workers)
            by_id = {str(item["source_id"]): dict(item["document"]) for item in documents}
            for source_id in missing:
                payload = by_id[source_id]
                path, digest = write_backfill_target(run_dir, source_id, payload)
                entry = _validated_target_payload(payload, source_id, f"backfill:{source_id}")
                entry["backfilled"] = True
                resolved[source_id] = entry
                binding[source_id] = {
                    "sha256": digest,
                    "backfilled": True,
                    "waveform_sha256": str(sessions[source_id].waveform_sha256),
                }
    arm_runtime.atomic_write_json(
        manifest_path,
        {
            "artifact_role": "issue-121-target-manifest",
            "sampling_sha256": sampling_sha256,
            "sources": {
                sid: binding[sid] for sid in sorted(binding)
            },
        },
    )
    return resolved, binding


def resolve_backend(
    bundle_dir: Path,
    corpus_root: Path,
    reference_root: Path,
    sampling_manifest: Path,
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    device: str,
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import material
    from experiments.psem_sortformer_adaptation_depth.sampling import (
        load_training_sessions,
    )

    manifest = load_stage_bundle(bundle_dir)
    population = material.resolve_sampling_population(Path(sampling_manifest))
    if str(population["sampling_sha256"]) != str(manifest.get("sampling_sha256")):
        raise TemporalArmError("sampling manifest differs from the frozen bundle contract")
    if not Path(checkpoint_path).is_file():
        raise TemporalArmError(f"checkpoint missing: {checkpoint_path}")
    if not Path(nemo_checkout).is_dir():
        raise TemporalArmError(f"NeMo checkout missing: {nemo_checkout}")
    if not Path(dependency_lock).is_file():
        raise TemporalArmError(f"dependency lock missing: {dependency_lock}")
    sessions = load_training_sessions(Path(corpus_root), Path(reference_root))
    assert_no_eval([str(s.role) for s in sessions.values()])
    rows_by_source = {
        sid: rows
        for sid, rows in population["rows_by_source"].items()
        if sid in sessions
    }
    if not rows_by_source:
        raise TemporalArmError("sampling manifest has no TRAIN session overlap")
    return {
        "manifest": manifest,
        "sessions": sessions,
        "rows_by_source": rows_by_source,
    }


def open_temporal_model(
    torch: Any,
    checkpoint: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    device: str,
    arm: str,
) -> dict[str, Any]:
    from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
        load_pinned_sortformer,
    )
    import importlib as _importlib

    head_mod = _importlib.import_module(
        "experiments.psem_state_corrected_adaptation_gate.head"
    )

    require_temporal_arm(arm)
    wrapper, runtime_receipt = load_pinned_sortformer(
        Path(checkpoint), Path(nemo_checkout), Path(dependency_lock), device
    )
    device_obj = next(wrapper.parameters()).device
    policy_receipt = apply_temporal_parameter_policy(wrapper, arm)
    head_module = head_mod.ResidualPSEMHead(HEAD_INPUT_DIM)
    head_module.to(device_obj)
    head_module.train(True)
    set_temporal_module_modes(wrapper, arm)
    modes = audit_temporal_module_modes(wrapper, arm)
    return {
        "wrapper": wrapper,
        "head_module": head_module,
        "device": device_obj,
        "runtime_receipt": runtime_receipt,
        "policy_receipt": policy_receipt,
        "modes": modes,
    }


def load_source_waveform(
    torch: Any, session: Any, corpus_root: Path, device: Any
) -> dict[str, Any]:
    from experiments.psem_sortformer_adaptation_depth.execution import (
        load_source_waveform as _load,
    )

    cached = _decode_cache_lookup(_session_ref(session, "source_id"), _session_ref(session, "audio_ref"), str(corpus_root))
    if cached is not None:
        waveform = cached["waveform"]
        waveform = waveform.to(device) if hasattr(waveform, "to") else waveform
        return {
            "waveform": waveform,
            "num_frames": int(cached["num_frames"]),
            "tail_excluded": int(cached["tail_excluded"]),
        }
    waveform, duration, tail = _load(session, Path(corpus_root))
    frames = int(waveform.shape[1]) // FRAME_SAMPLES
    if frames <= 0:
        raise TemporalArmError(f"source has no complete frames: {session.source_id}")
    trimmed = waveform[:, : frames * FRAME_SAMPLES]
    _decode_cache_store(_session_ref(session, "source_id"), _session_ref(session, "audio_ref"), str(corpus_root), trimmed, frames, int(tail))
    return {
        "waveform": trimmed.to(device) if hasattr(trimmed, "to") else trimmed,
        "num_frames": frames,
        "tail_excluded": int(tail),
    }


def _dump_bytes(torch: Any, obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def snapshot_blobs(
    torch: Any,
    wrapper: Any,
    head_module: Any,
    optimizer: Any,
    scheduler: Any,
    accum: Any,
) -> dict[str, bytes]:
    rng: dict[str, Any] = {
        "torch": _dump_bytes(torch, torch.get_rng_state()),
        "python": _dump_bytes(torch, random.getstate()),
    }
    try:
        import numpy as _np

        rng["numpy"] = _dump_bytes(torch, _np.random.get_state())
    except ImportError:
        rng["numpy"] = b""
    if torch.cuda.is_available():
        rng["cuda"] = _dump_bytes(torch, torch.cuda.get_rng_state_all())
    else:
        rng["cuda"] = b""
    grads: dict[str, Any] = {}
    for prefix, module in (("model:", wrapper), ("head:", head_module)):
        for name, param in module.named_parameters():
            grad = getattr(param, "grad", None)
            if grad is not None:
                grads[prefix + name] = grad.detach().cpu()
    full_wrapper = wrapper.state_dict()
    frozen_names = sorted(n for n in full_wrapper if is_frozen_encoder_param(n))
    trainable_wrapper = {n: v for n, v in full_wrapper.items() if n not in set(frozen_names)}
    return {
        "model": _dump_bytes(
            torch,
            {
                "format": 2,
                "wrapper_trainable": trainable_wrapper,
                "wrapper_frozen_names": frozen_names,
                "head": head_module.state_dict(),
                "pending_grads": grads,
                "pending": int(accum.pending),
                "optimizer_steps": int(accum.optimizer_steps),
                "loss_chunks": int(accum.loss_chunks),
                "empty_chunks": int(accum.empty_chunks),
            },
        ),
        "optimizer": _dump_bytes(torch, optimizer.state_dict()),
        "scheduler": _dump_bytes(torch, scheduler.state_dict()),
        "rng": _dump_bytes(torch, rng),
    }


def restore_blobs(
    torch: Any,
    wrapper: Any,
    head_module: Any,
    optimizer: Any,
    scheduler: Any,
    accum: Any,
    blobs: Mapping[str, Path],
    device: Any,
) -> None:
    def _load(path: Path) -> Any:
        return torch.load(str(path), map_location=device, weights_only=False)

    model_state = _load(blobs["model"])
    if "wrapper" in model_state:
        wrapper.load_state_dict(model_state["wrapper"], strict=True)
    else:
        if model_state.get("format") != 2 or "wrapper_trainable" not in model_state:
            raise TemporalArmError("checkpoint model blob is incompatible")
        current = wrapper.state_dict()
        current_frozen = sorted(n for n in current if is_frozen_encoder_param(n))
        if current_frozen != sorted(model_state.get("wrapper_frozen_names", [])):
            raise TemporalArmError("checkpoint frozen encoder geometry is incompatible")
        stored_trainable = dict(model_state["wrapper_trainable"])
        expected_trainable = sorted(n for n in current if n not in set(current_frozen))
        if sorted(stored_trainable) != expected_trainable:
            raise TemporalArmError("checkpoint trainable arm is incompatible")
        merged = dict(current)
        merged.update(stored_trainable)
        wrapper.load_state_dict(merged, strict=True)
    head_module.load_state_dict(model_state["head"], strict=True)

    optimizer.load_state_dict(_load(blobs["optimizer"]))
    scheduler.load_state_dict(_load(blobs["scheduler"]))
    lookup: dict[str, Any] = {}
    for prefix, module in (("model:", wrapper), ("head:", head_module)):
        for name, param in module.named_parameters():
            lookup[prefix + name] = param
    for key, grad in dict(model_state.get("pending_grads", {})).items():
        if key in lookup:
            param = lookup[key]
            target = getattr(param, "device", device)
            lookup[key].grad = grad.to(device=target) if hasattr(grad, "to") else grad
    accum.pending = int(model_state.get("pending", 0))
    accum.optimizer_steps = int(model_state.get("optimizer_steps", 0))
    accum.loss_chunks = int(model_state.get("loss_chunks", 0))
    accum.empty_chunks = int(model_state.get("empty_chunks", 0))
    rng = _load(blobs["rng"])
    torch.set_rng_state(
        torch.load(io.BytesIO(rng["torch"]), map_location="cpu", weights_only=False)
    )
    random.setstate(
        torch.load(io.BytesIO(rng["python"]), map_location="cpu", weights_only=False)
    )
    if rng.get("numpy"):
        import numpy as _np

        _np.random.set_state(
            torch.load(io.BytesIO(rng["numpy"]), map_location="cpu", weights_only=False)
        )
    if rng.get("cuda"):
        torch.cuda.set_rng_state_all(
            torch.load(io.BytesIO(rng["cuda"]), map_location="cpu", weights_only=False)
        )


def chunk_spans(num_frames: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for start in range(0, int(num_frames), CHUNK_FRAMES):
        spans.append((start, min(start + CHUNK_FRAMES, int(num_frames))))
    if not spans:
        raise TemporalArmError("source has no frames")
    return spans


def frame_mult_weights(
    authority: Any,
    multiplicity: Sequence[float],
    episode_ids: Sequence[str | None],
    slot_of: Mapping[str, int],
    start: int,
    length: int,
) -> list[float]:
    valid = list(authority.valid)
    weights: list[float] = []
    for offset in range(length):
        frame = start + offset
        episode = episode_ids[frame]
        mapped = 1.0 if (episode is not None and episode in slot_of) else 0.0
        weights.append(
            float(multiplicity[frame]) * mapped * (1.0 if valid[frame] else 0.0)
        )
    return weights


def chunk_has_support(
    authority: Any,
    multiplicity: Sequence[float],
    episode_ids: Sequence[str | None],
    slot_of: Mapping[str, int],
    start: int,
    length: int,
) -> bool:
    return any(
        v > 0
        for v in frame_mult_weights(
            authority, multiplicity, episode_ids, slot_of, start, length
        )
    )


def build_source_plan(prep: Mapping[str, Any], slot_of: Mapping[str, int]) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import multiplicity as multiplicity_mod
    num_frames = int(prep["num_frames"])
    authority = prep["authority"]
    valid = authority.valid
    multiplicity = prep["multiplicity"]
    episode_ids = prep["episode_ids"]
    warmup = int(multiplicity_mod.WARMUP_FRAMES)
    full_mult = frame_mult_weights(authority, multiplicity, episode_ids, slot_of, 0, num_frames)
    chunk_support: list[bool] = []
    native_selected: list[list[int]] = []
    for start, end in chunk_spans(num_frames):
        chunk_support.append(any(v > 0 for v in full_mult[start:end]))
        selected: list[int] = []
        for offset in range(end - start):
            frame = start + offset
            episode = episode_ids[frame]
            if not valid[frame] or frame < warmup:
                continue
            if episode is None or episode not in slot_of:
                continue
            if float(multiplicity[frame]) <= 0:
                continue
            selected.extend([offset] * int(multiplicity[frame]))
        native_selected.append(selected)
    return {
        "full_mult": full_mult,
        "chunk_support": chunk_support,
        "native_selected": native_selected,
    }


def count_loss_chunks(
    entries: Mapping[str, dict[str, Any]], mapping: PassMapping, source_ids: Sequence[str]
) -> int:
    total = 0
    for source_id in source_ids:
        entry = entries[source_id]
        slot_of = mapping.for_source(source_id)["slot_of"]
        plan = build_source_plan(entry, slot_of)
        total += sum(1 for supported in plan["chunk_support"] if supported)
    return total


def build_full_source_supervision(torch: Any, entry: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import material
    from experiments.psem_state_corrected_adaptation_gate import multiplicity as multiplicity_mod

    num_frames = int(entry["num_frames"])
    active_by_frame, valid_by_frame = material.full_source_frame_labels(
        [dict(row) for row in entry.get("intervals", [])], num_frames
    )
    ranges: list[tuple[int, int, str]] = []
    for episode in entry["authority"].episodes:
        ranges.append(
            (int(episode.start_frame), int(episode.end_frame), str(episode.episode_id))
        )
    ranges.sort()
    episode_ids: list[str | None] = [None] * num_frames
    for start_frame, end_frame, episode_id in ranges:
        for frame in range(max(start_frame, 0), min(end_frame, num_frames)):
            episode_ids[frame] = episode_id
    order: list[str] = []
    for active in active_by_frame:
        for speaker in active:
            if speaker not in order:
                order.append(speaker)
    if len(order) > 4:
        raise TemporalArmError(
            f"source exceeds four arrival-order slots: {entry.get('source_id', '')}"
        )
    slot_of = {speaker: index for index, speaker in enumerate(order)}
    warmup = int(multiplicity_mod.WARMUP_FRAMES)
    full_arrival = torch.zeros((num_frames, 4), dtype=torch.float32)
    full_mask = torch.zeros((num_frames,), dtype=torch.bool)
    for frame in range(num_frames):
        if not valid_by_frame[frame] or frame < warmup:
            continue
        full_mask[frame] = True
        for speaker in active_by_frame[frame]:
            full_arrival[frame, int(slot_of[speaker])] = 1.0
    chunks: list[dict[str, Any]] = []
    for start, end in chunk_spans(num_frames):
        chunks.append(
            {
                "start": start,
                "length": end - start,
                "episode_ids": episode_ids[start:end],
                "arrival": full_arrival[start:end],
                "native_mask": full_mask[start:end],
            }
        )
    return {"num_frames": num_frames, "chunks": chunks}


def _decode_cache_key(source_id: str, audio_ref: str, corpus_root: str) -> str:
    return "\x00".join([str(source_id), str(audio_ref), str(corpus_root)])


def _decode_cache_lookup(source_id: str, audio_ref: str, corpus_root: str) -> dict[str, Any] | None:
    global _DECODE_CACHE_KEY, _DECODE_CACHE_WAVEFORM, _DECODE_CACHE_NUM_FRAMES, _DECODE_CACHE_TAIL_EXCLUDED
    if _DECODE_CACHE_KEY != _decode_cache_key(source_id, audio_ref, corpus_root):
        return None
    if _DECODE_CACHE_WAVEFORM is None:
        return None
    return {
        "source_id": str(source_id),
        "waveform": _DECODE_CACHE_WAVEFORM,
        "num_frames": int(_DECODE_CACHE_NUM_FRAMES),
        "tail_excluded": int(_DECODE_CACHE_TAIL_EXCLUDED),
    }


def _decode_cache_store(source_id: str, audio_ref: str, corpus_root: str, waveform: Any, num_frames: int, tail_excluded: int) -> None:
    global _DECODE_CACHE_KEY, _DECODE_CACHE_WAVEFORM, _DECODE_CACHE_NUM_FRAMES, _DECODE_CACHE_TAIL_EXCLUDED
    _DECODE_CACHE_KEY = _decode_cache_key(source_id, audio_ref, corpus_root)
    _DECODE_CACHE_WAVEFORM = waveform
    _DECODE_CACHE_NUM_FRAMES = int(num_frames)
    _DECODE_CACHE_TAIL_EXCLUDED = int(tail_excluded)


def _decode_cache_clear() -> None:
    global _DECODE_CACHE_KEY, _DECODE_CACHE_WAVEFORM, _DECODE_CACHE_NUM_FRAMES, _DECODE_CACHE_TAIL_EXCLUDED
    _DECODE_CACHE_KEY = None
    _DECODE_CACHE_WAVEFORM = None
    _DECODE_CACHE_NUM_FRAMES = 0
    _DECODE_CACHE_TAIL_EXCLUDED = 0


def _session_ref(session: Any, name: str) -> str:
    return str(getattr(session, name, "") or "")


def decode_waveform_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.psem_sortformer_adaptation_depth.execution import (
        load_source_waveform as _load,
    )
    from types import SimpleNamespace as _NS

    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    arm_runtime.enforce_thread_caps()

    cached = _decode_cache_lookup(
        str(payload["source_id"]), str(payload["audio_ref"]), str(payload["corpus_root"])
    )
    if cached is not None:
        return dict(cached)
    session = _NS(
        source_id=str(payload["source_id"]),
        audio_ref=str(payload["audio_ref"]),
        waveform_sha256=str(payload["waveform_sha256"]),
    )
    waveform, duration, tail = _load(session, Path(str(payload["corpus_root"])))
    frames = int(waveform.shape[1]) // FRAME_SAMPLES
    if frames <= 0:
        raise TemporalArmError(f"source has no complete frames: {session.source_id}")
    trimmed = waveform[:, : frames * FRAME_SAMPLES]
    _decode_cache_store(
        str(payload["source_id"]), str(payload["audio_ref"]), str(payload["corpus_root"]),
        trimmed, frames, int(tail),
    )
    return {
        "source_id": str(payload["source_id"]),
        "waveform": trimmed,
        "num_frames": frames,
        "tail_excluded": int(tail),
    }


def decode_payload(session: Any, corpus_root: Path) -> dict[str, Any]:
    return {
        "source_id": _session_ref(session, "source_id"),
        "audio_ref": _session_ref(session, "audio_ref"),
        "waveform_sha256": _session_ref(session, "waveform_sha256"),
        "corpus_root": str(corpus_root),
    }


def assemble_prep(
    torch: Any,
    session: Any,
    entry: Mapping[str, Any],
    decoded: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    num_frames = int(decoded["num_frames"])
    if num_frames != int(entry["num_frames"]):
        raise TemporalArmError(
            f"waveform frames differ from the frozen target: {session.source_id}"
        )
    supervision = build_full_source_supervision(torch, entry)
    waveform = decoded["waveform"]
    waveform = waveform.to(device) if hasattr(waveform, "to") else waveform
    return {
        "source_id": str(session.source_id),
        "waveform": waveform,
        "num_frames": num_frames,
        "tail_excluded": int(decoded["tail_excluded"]),
        "authority": entry["authority"],
        "multiplicity": list(entry["multiplicity"]),
        "episode_ids": list(entry["episode_ids"]),
        "chunk_sup": supervision["chunks"],
    }


def prepare_source(
    torch: Any,
    session: Any,
    entry: Mapping[str, Any],
    corpus_root: Path,
    device: Any,
) -> dict[str, Any]:
    decoded = decode_waveform_task(decode_payload(session, corpus_root))
    return assemble_prep(torch, session, entry, decoded, device)


def source_probabilities(
    torch: Any, wrapper: Any, waveform: Any
) -> list[list[float]]:
    from experiments.psem_state_corrected_adaptation_gate import material

    with torch.no_grad():
        passage = material.run_adjacent_windows(
            torch, wrapper, waveform, CHUNK_FRAMES, True
        )
        evidence = material.concat_windows(torch, passage["windows"])
    return evidence["probabilities"][0].detach().cpu().tolist()


def release_waveform(torch: Any, waveform: Any) -> None:
    del waveform
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def freeze_pass_mapping(
    torch: Any,
    arm: str,
    seed: int,
    fit_sources: Sequence[str],
    sessions: Mapping[str, Any],
    entries: Mapping[str, dict[str, Any]],
    corpus_root: Path,
    device: Any,
    wrapper: Any,
    workers: int | None,
) -> PassMapping:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    require_temporal_arm(arm)
    wrapper.eval()
    ordered = arm_runtime.chronological_sources(list(fit_sources))
    payloads: list[dict[str, Any]] = []
    for source_id in ordered:
        entry = entries[source_id]
        authority = entry["authority"]
        loaded = load_source_waveform(torch, sessions[source_id], corpus_root, device)
        try:
            probabilities = source_probabilities(torch, wrapper, loaded["waveform"])
        finally:
            release_waveform(torch, loaded["waveform"])
            del loaded
        payloads.append(
            {
                "source_id": source_id,
                "episode_ids": list(entry["episode_ids"]),
                "anchor_active": [a == 1.0 for a in authority.y_anchor],
                "valid": list(authority.valid),
                "probabilities": probabilities,
            }
        )
    results = arm_runtime.ordered_process_map(
        mapping_worker, payloads, arm_runtime.resolve_workers(workers)
    )
    frozen = {row["source_id"]: row["mapping"] for row in results}
    manifest_hash = arm_runtime.canonical_sha256(
        {sid: frozen[sid] for sid in sorted(frozen)}
    )
    set_temporal_module_modes(wrapper, arm)
    return PassMapping(
        source_to_mapping=frozen,
        manifest_hash=manifest_hash,
        arm=arm,
        seed=int(seed),
    )


def mapping_manifest_path(run_dir: Path) -> Path:
    return Path(run_dir) / "mapping_manifest.json"


def mapping_file_path(run_dir: Path) -> Path:
    return Path(run_dir) / "mapping.json"


def write_mapping_files(run_dir: Path, mapping: PassMapping, config_hash: str) -> list[Path]:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    manifest = arm_runtime.atomic_write_json(
        mapping_manifest_path(run_dir),
        {
            "arm": mapping.arm,
            "seed": mapping.seed,
            "config_hash": str(config_hash),
            "manifest_hash": mapping.manifest_hash,
            "sources": sorted(mapping.source_to_mapping),
        },
    )
    mappings = arm_runtime.atomic_write_json(
        mapping_file_path(run_dir),
        {
            "arm": mapping.arm,
            "seed": mapping.seed,
            "manifest_hash": mapping.manifest_hash,
            "sources": {sid: mapping.source_to_mapping[sid] for sid in sorted(mapping.source_to_mapping)},
        },
    )
    return [manifest, mappings]


def load_frozen_mapping(run_dir: Path, config: Any) -> PassMapping:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    manifest = dict(
        json.loads(mapping_manifest_path(run_dir).read_text(encoding="utf-8"))
    )
    payload = dict(
        json.loads(mapping_file_path(run_dir).read_text(encoding="utf-8"))
    )
    for doc in (manifest, payload):
        if doc.get("arm") != config.arm or int(doc.get("seed", -1)) != int(config.seed):
            raise TemporalArmError("stored mapping belongs to a different arm or seed")
        if doc.get("manifest_hash") != manifest.get("manifest_hash"):
            raise TemporalArmError("stored mapping files diverge")
    if manifest.get("config_hash", config.config_hash) != config.config_hash:
        raise TemporalArmError("stored mapping belongs to a different config")
    frozen = {str(sid): mapping for sid, mapping in payload["sources"].items()}
    recomputed = arm_runtime.canonical_sha256(
        {sid: frozen[sid] for sid in sorted(frozen)}
    )
    if recomputed != manifest.get("manifest_hash"):
        raise TemporalArmError("stored mapping hash does not match its content")
    return PassMapping(
        source_to_mapping=frozen,
        manifest_hash=str(manifest["manifest_hash"]),
        arm=str(manifest["arm"]),
        seed=int(manifest["seed"]),
    )


def read_mapping_hash(run_dir: Path) -> str | None:
    path = mapping_manifest_path(run_dir)
    if not path.is_file() or not mapping_file_path(run_dir).is_file():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8"))).get("manifest_hash")


def _pop_segment(
    torch: Any, buffers: dict[str, list[Any]], buffered: int, frames: int = CHUNK_FRAMES
) -> tuple[dict[str, Any], dict[str, list[Any]], int]:
    full = {
        name: torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        for name, parts in buffers.items()
    }
    segment = {name: tensor[:, :frames] for name, tensor in full.items()}
    rest = {name: tensor[:, frames:] for name, tensor in full.items()}
    kept = {
        name: [tensor] if tensor.shape[1] > 0 else [] for name, tensor in rest.items()
    }
    return segment, kept, buffered - frames


@dataclass(slots=True)
class AccumState:
    pending: int = 0
    optimizer_steps: int = 0
    loss_chunks: int = 0
    empty_chunks: int = 0


def group_scale(pending: int, accumulation: int) -> float:
    if int(pending) <= 0:
        raise TemporalArmError("optimizer group is empty")
    if int(pending) >= int(accumulation):
        return 1.0 / float(accumulation)
    return 1.0 / float(pending)


def apply_optimizer_update(
    torch: Any,
    optimizer: Any,
    scheduler: Any,
    carry: TemporalCarry,
    accum: AccumState,
    accumulation: int = ACCUMULATION_DEFAULT,
) -> None:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    scale = group_scale(accum.pending, int(accumulation))
    for group in optimizer.param_groups:
        for p in group["params"]:
            grad = getattr(p, "grad", None)
            if grad is not None:
                p.grad = grad * scale
    torch.nn.utils.clip_grad_norm_(
        [
            p
            for group in optimizer.param_groups
            for p in group["params"]
            if getattr(p, "grad", None) is not None
        ],
        float(arm_runtime.OPTIMIZER_CONTRACT["gradient_clip_norm"]),
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    carry.mark_update()
    accum.optimizer_steps += 1
    accum.pending = 0


def expand_native_loss(
    torch: Any,
    wrapper: Any,
    probabilities: Any,
    arrival: Any,
    prep: Mapping[str, Any],
    slot_of: Mapping[str, int],
    start: int,
    length: int,
    device: Any,
    preselected: Sequence[int] | None = None,
) -> Any:
    from experiments.psem_state_corrected_adaptation_gate import multiplicity as multiplicity_mod

    if preselected is None:
        warmup = int(multiplicity_mod.WARMUP_FRAMES)
        authority = prep["authority"]
        valid = authority.valid
        episode_ids = prep["episode_ids"]
        multiplicity = prep["multiplicity"]
        selected: list[int] = []
        for offset in range(length):
            frame = start + offset
            episode = episode_ids[frame]
            if not valid[frame] or frame < warmup:
                continue
            if episode is None or episode not in slot_of:
                continue
            if float(multiplicity[frame]) <= 0:
                continue
            selected.extend([offset] * int(multiplicity[frame]))
    else:
        selected = list(preselected)
    if not selected:
        return None
    expanded_probs = probabilities[:, selected, :]
    expanded_arrival = arrival[selected].unsqueeze(0)
    expanded_mask = torch.ones((1, len(selected)), dtype=torch.bool, device=device)
    native = wrapper.native_sortformer_loss(
        SimpleEvidence(probabilities=expanded_probs),
        expanded_arrival.to(device=device),
        expanded_mask,
        TRAIN_NATIVE_ROLES,
        valid_lengths=(len(selected),),
    )
    return native.value if hasattr(native, "value") else native


def build_source_device_tensors(
    torch: Any,
    prep: Mapping[str, Any],
    slot_of: Mapping[str, int],
    device: Any,
) -> dict[str, Any]:
    num_frames = int(prep["num_frames"])
    plan = build_source_plan(prep, slot_of)
    return {
        "y_replace": torch.tensor(
            [list(prep["authority"].y_replace)], dtype=torch.float32, device=device
        ),
        "y_anchor": torch.tensor(
            [list(prep["authority"].y_anchor)], dtype=torch.float32, device=device
        ),
        "mult_weight": torch.tensor([plan["full_mult"]], dtype=torch.float32, device=device),
        "chunk_support": list(plan["chunk_support"]),
        "native_selected": [list(selected) for selected in plan["native_selected"]],
    }




def seed_temporal_from_config(torch: Any, seed: int) -> dict[str, Any]:
    random.seed(int(seed))
    numpy_seeded = False
    try:
        import numpy as _np

        _np.random.seed(int(seed) % (2**32))
        numpy_seeded = True
    except ImportError:
        numpy_seeded = False
    torch_cpu = False
    try:
        manual_seed = getattr(torch, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(int(seed))
            torch_cpu = True
    except (RuntimeError, AttributeError):
        torch_cpu = False
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
        "torch_cpu": torch_cpu,
        "torch_cuda": cuda_seeded,
    }


def train_chunk(
    torch: Any,
    wrapper: Any,
    head_module: Any,
    carry: TemporalCarry,
    prep: Mapping[str, Any],
    slot_of: Mapping[str, int],
    segment: Mapping[str, Any],
    chunk_entry: Mapping[str, Any],
    class_weights: Mapping[str, float],
    device: Any,
    source_tensors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    length = int(chunk_entry["length"])
    episode_ids = chunk_entry["episode_ids"]
    hidden = segment["hidden"]
    logits = segment["logits"]
    probabilities = segment["probabilities"]
    slot_index = [
        int(slot_of[episode_id]) if episode_id is not None and episode_id in slot_of else -1
        for episode_id in episode_ids
    ]
    slot_row = torch.tensor([slot_index], dtype=torch.float32, device=device)
    one_hot = torch.zeros((1, length, 4), dtype=torch.float32, device=device)
    for _slot in range(4):
        _mask = (slot_row == float(_slot))
        _mask_f = _mask.to(torch.float32) if hasattr(_mask, "to") else _mask
        one_hot[:, :, _slot] = _mask_f
    selected = (logits * one_hot).sum(dim=-1, keepdim=True).squeeze(-1)
    anchor_mask = one_hot.bool()
    neg_inf = torch.full_like(logits, float("-inf"))
    non_anchor = torch.where(anchor_mask, neg_inf, logits)
    best_non_anchor = non_anchor.max(dim=-1, keepdim=True).values
    best_non_anchor = torch.where(
        torch.isfinite(best_non_anchor), best_non_anchor, torch.zeros_like(best_non_anchor)
    )
    delay = torch.full_like(selected.unsqueeze(-1), 1.04)
    features = torch.cat(
        [hidden, logits, selected.unsqueeze(-1), best_non_anchor, delay], dim=-1
    )
    part_widths = (
        int(hidden.shape[-1]),
        int(logits.shape[-1]),
        int(selected.unsqueeze(-1).shape[-1]),
        int(best_non_anchor.shape[-1]),
        int(delay.shape[-1]),
    )
    if part_widths != (192, 4, 1, 1, 1) or int(features.shape[-1]) != HEAD_INPUT_DIM:
        raise TemporalArmError(
            "temporal feature geometry violates 199=192+4+1+1+1: "
            f"parts={part_widths} total={int(features.shape[-1])}"
        )
    head_out, next_gru = head_module(features, carry.gru_state)
    carry.gru_state = next_gru
    anchor_logit = head_out["anchor_logit"]
    resid = head_out["z_residual"]
    f0_logit = torch.logit(
        (1.0 - torch.sigmoid(selected)).clamp(1e-6, 1.0 - 1e-6)
    )
    product_logit = f0_logit + resid
    start = int(chunk_entry["start"])
    known_support: bool | None = None
    preselected: Sequence[int] | None = None
    if source_tensors is not None:
        y_replace = source_tensors["y_replace"][:, start:start + length]
        y_anchor = source_tensors["y_anchor"][:, start:start + length]
        mult_weight = source_tensors["mult_weight"][:, start:start + length]
        support = source_tensors.get("chunk_support")
        if support is not None:
            known_support = bool(support[start // CHUNK_FRAMES])
        native_plan = source_tensors.get("native_selected")
        if native_plan is not None:
            preselected = native_plan[start // CHUNK_FRAMES]
    else:
        y_replace = torch.tensor(
            [list(prep["authority"].y_replace[start:start + length])],
            dtype=torch.float32,
            device=device,
        )
        y_anchor = torch.tensor(
            [list(prep["authority"].y_anchor[start:start + length])],
            dtype=torch.float32,
            device=device,
        )
        mult = frame_mult_weights(
            prep["authority"],
            prep["multiplicity"],
            prep["episode_ids"],
            slot_of,
            start,
            length,
        )
        mult_weight = torch.tensor([mult], dtype=torch.float32, device=device)
    native_value = expand_native_loss(
        torch,
        wrapper,
        probabilities,
        chunk_entry["arrival"],
        prep,
        slot_of,
        start,
        length,
        device,
        preselected,
    )
    return temporal_chunk_loss(
        torch,
        product_logit,
        y_replace,
        anchor_logit,
        y_anchor,
        mult_weight,
        class_weights,
        native_value,
        known_support,
    )


def train_source(
    torch: Any,
    wrapper: Any,
    head_module: Any,
    optimizer: Any,
    scheduler: Any,
    carry: TemporalCarry,
    accum: AccumState,
    prep: Mapping[str, Any],
    slot_of: Mapping[str, int],
    class_weights: Mapping[str, float],
    device: Any,
    accumulation: int = ACCUMULATION_DEFAULT,
    max_optimizer_steps: int | None = None,
    flush: bool = False,
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import material

    num_frames = int(prep["num_frames"])
    source_id = str(prep["source_id"])
    with torch.no_grad():
        counted = material.prepare_streaming(torch, wrapper, prep["waveform"])
        counting = material.init_source_state(torch, wrapper)
        counted_frames = 0
        for _, chunk, lengths, left_offset, right_offset in counted["loader"]:
            counting, counted_hidden, _, _, _ = wrapper._streaming_step(
                chunk,
                lengths,
                counting,
                left_offset=left_offset,
                right_offset=right_offset,
            )
            counted_frames += int(counted_hidden.shape[1])
        del counting
    if counted_frames != num_frames:
        raise TemporalArmError(
            f"streaming emitted {counted_frames} frames for {num_frames}-frame source {source_id}: "
            "frame count differs from authority/source length"
        )
    prepared = material.prepare_streaming(torch, wrapper, prep["waveform"])
    source_tensors = build_source_device_tensors(torch, prep, dict(slot_of), device)
    carry.reset(source_id, material.init_source_state(torch, wrapper))
    buffers: dict[str, list[Any]] = {"hidden": [], "logits": [], "probabilities": []}
    buffered = 0
    emitted = 0
    steps_before = accum.optimizer_steps
    loss_before = accum.loss_chunks
    empty_before = accum.empty_chunks
    chunks = 0
    entries = list(prep["chunk_sup"])
    stopped = False
    for _, chunk, lengths, left_offset, right_offset in prepared["loader"]:
        if stopped or emitted >= num_frames:
            break
        state, hidden, logits, probabilities, _ = wrapper._streaming_step(
            chunk,
            lengths,
            carry.sortformer_state,
            left_offset=left_offset,
            right_offset=right_offset,
        )
        carry.sortformer_state = state
        take = int(hidden.shape[1])
        if emitted + take > num_frames:
            raise TemporalArmError(
                f"streaming emitted beyond {num_frames}-frame source {source_id}: "
                "frame count differs from authority/source length"
            )
        if take <= 0:
            break
        buffers["hidden"].append(hidden)
        buffers["logits"].append(logits)
        buffers["probabilities"].append(probabilities)
        buffered += take
        emitted += take
        while buffered >= CHUNK_FRAMES and not stopped:
            segment, buffers, buffered = _pop_segment(torch, buffers, buffered)
            computed = train_chunk(
                torch,
                wrapper,
                head_module,
                carry,
                prep,
                slot_of,
                segment,
                entries[chunks],
                class_weights,
                device,
                source_tensors,
            )
            chunks += 1
            if computed["empty"]:
                accum.empty_chunks += 1
                carry.detach(torch)
                continue
            computed["loss"].backward()
            accum.pending += 1
            accum.loss_chunks += 1
            carry.detach(torch)
            if accum.pending >= int(accumulation):
                apply_optimizer_update(
                    torch, optimizer, scheduler, carry, accum, int(accumulation)
                )
                if max_optimizer_steps is not None and accum.optimizer_steps >= max_optimizer_steps:
                    stopped = True
    if buffered > 0 and not stopped:
        segment = {
            name: torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
            for name, parts in buffers.items()
        }
        computed = train_chunk(
            torch,
            wrapper,
            head_module,
            carry,
            prep,
            slot_of,
            segment,
            entries[chunks],
            class_weights,
            device,
            source_tensors,
        )
        chunks += 1
        if computed["empty"]:
            accum.empty_chunks += 1
            carry.detach(torch)
        else:
            computed["loss"].backward()
            accum.pending += 1
            accum.loss_chunks += 1
            carry.detach(torch)
            if accum.pending >= int(accumulation):
                apply_optimizer_update(
                    torch, optimizer, scheduler, carry, accum, int(accumulation)
                )
    if flush and accum.pending > 0 and not stopped:
        apply_optimizer_update(
            torch, optimizer, scheduler, carry, accum, int(accumulation)
        )
    return {
        "optimizer_steps": accum.optimizer_steps - steps_before,
        "loss_chunks": accum.loss_chunks - loss_before,
        "empty_chunks": accum.empty_chunks - empty_before,
        "chunks": chunks,
        "emitted_frames": emitted,
        "pending": accum.pending,
    }

def schedule_total_steps(total_loss_chunks: int, accumulation: int) -> int:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    if int(total_loss_chunks) <= 0:
        raise TemporalArmError("schedule has no loss-contributing chunks")
    return arm_runtime.compute_total_steps(int(total_loss_chunks), int(accumulation))


def mapping_coverage_agreement(
    prep: Mapping[str, Any], slot_of: Mapping[str, int]
) -> dict[str, float]:
    authority = prep["authority"]
    rows = slot_of_table(prep, slot_of)
    total_episodes = max(1, len(rows))
    mapped_episodes = sum(1 for r in rows if r["status"] == "mapped")
    anchor_active = [a == 1.0 for a in authority.y_anchor]
    valid = list(authority.valid)
    mapped_frames = sum(
        1
        for e, a, v in zip(prep["episode_ids"], anchor_active, valid)
        if a and v and e is not None and e in slot_of
    )
    support = sum(1 for a, v in zip(anchor_active, valid) if a and v)
    return {
        "coverage": mapped_episodes / total_episodes,
        "agreement": (mapped_frames / support) if support else 0.0,
        "mapped_episodes": float(mapped_episodes),
        "total_episodes": float(total_episodes),
        "mapped_frames": float(mapped_frames),
        "support_frames": float(support),
    }


def slot_of_table(
    prep: Mapping[str, Any], slot_of: Mapping[str, int]
) -> list[dict[str, Any]]:
    episodes = sorted({e for e in prep["episode_ids"] if e is not None})
    return [
        {"episode_id": e, "status": "mapped" if e in slot_of else "unmapped"}
        for e in episodes
    ]




def profile_receipt_path(run_dir: Path) -> Path:
    return Path(run_dir) / "profile.json"


def require_profile_receipt(
    receipt: Mapping[str, Any] | None, config: Any
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise TemporalArmError("full arm requires a profile receipt")
    steps = receipt.get("optimizer_steps")
    if not isinstance(steps, int) or not PROFILE_MIN_STEPS <= steps <= PROFILE_MAX_STEPS:
        raise TemporalArmError("profile receipt lacks 8-16 real optimizer steps")
    for key in ("seconds_per_step", "peak_vram_bytes", "dev_infer_seconds"):
        if key not in receipt:
            raise TemporalArmError(f"profile receipt lacks {key}")
    if receipt.get("arm", config.arm) != config.arm:
        raise TemporalArmError("profile receipt arm mismatch")
    if receipt.get("seed", config.seed) != config.seed:
        raise TemporalArmError("profile receipt seed mismatch")
    if receipt.get("config_hash", config.config_hash) != config.config_hash:
        raise TemporalArmError("profile receipt config mismatch")
    return dict(receipt)


def load_profile_receipt(run_dir: Path) -> dict[str, Any] | None:
    path = profile_receipt_path(run_dir)
    if not path.is_file():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def run_profile_command(config: Any, store: Path, args: Any) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    require_temporal_arm(config.arm)
    receipt = arm_runtime.check_authorization(config, Path(store))
    manifest = load_stage_bundle(args.bundle)
    binding = verify_execution_binding(config, manifest, args.checkpoint)
    torch = load_torch()
    gpu_lock = arm_runtime.arm_gpu_lock(
        config.root, {"run_id": f"{config.arm}-{config.seed}", "arm": config.arm, "seed": int(config.seed), "command": "profile"}
    )
    gpu_lock.__enter__()
    stream_started = False
    try:
        backend = resolve_backend(
            args.bundle,
            args.corpus_root,
            args.reference_root,
            args.sampling_manifest,
            args.checkpoint,
            args.nemo_checkout,
            args.dependency_lock,
            args.device,
        )
        fit_sources = arm_runtime.chronological_sources(list(binding["fit"]))
        if not fit_sources:
            raise TemporalArmError("TRAIN-FIT partition is empty")
        sessions = backend["sessions"]
        entries, target_binding = resolve_durable_targets(
            torch,
            config.run_dir(),
            args.bundle,
            manifest,
            sessions,
            backend["rows_by_source"],
            args.corpus_root,
            None,
            fit_sources,
            getattr(args, "workers", None),
        )
        weights = binding["class_weights"]
        seed_report = seed_temporal_from_config(torch, int(config.seed))
        opened = open_temporal_model(
            torch,
            args.checkpoint,
            args.nemo_checkout,
            args.dependency_lock,
            args.device,
            config.arm,
        )
        wrapper = opened["wrapper"]
        head_module = opened["head_module"]
        device = opened["device"]
        check_single_stream(config.arm)
        stream_started = True
    except BaseException:
        gpu_lock.__exit__(None, None, None)
        raise
    try:
        run_dir = config.run_dir()
        mapping_start = time.perf_counter()
        mapping = freeze_pass_mapping(
            torch,
            config.arm,
            config.seed,
            fit_sources,
            sessions,
            entries,
            args.corpus_root,
            device,
            wrapper,
            args.workers,
        )
        mapping_seconds = time.perf_counter() - mapping_start
        total_loss = count_loss_chunks(entries, mapping, fit_sources)
        total_steps = schedule_total_steps(total_loss, ACCUMULATION_DEFAULT)
        optimizer = build_optimizer(torch, wrapper, head_module, config.arm)
        scheduler = build_scheduler(torch, optimizer, total_steps)
        accum = AccumState()
        carry = TemporalCarry()
        if accum.pending == 0:
            optimizer.zero_grad()
        started = time.perf_counter()
        per_source: dict[str, Any] = {}
        for source_id in fit_sources:
            prep = prepare_source(torch, sessions[source_id], entries[source_id], args.corpus_root, device)
            result = train_source(
                torch,
                wrapper,
                head_module,
                optimizer,
                scheduler,
                carry,
                accum,
                prep,
                dict(mapping.for_source(source_id)["slot_of"]),
                weights,
                device,
                ACCUMULATION_DEFAULT,
                PROFILE_MIN_STEPS,
                False,
            )
            per_source[source_id] = result
            del prep
            if accum.optimizer_steps >= PROFILE_MIN_STEPS:
                break
        if accum.pending > 0:
            apply_optimizer_update(
                torch, optimizer, scheduler, carry, accum, ACCUMULATION_DEFAULT
            )
        if not PROFILE_MIN_STEPS <= accum.optimizer_steps <= PROFILE_MAX_STEPS:
            raise TemporalArmError("profiler did not run 8-16 real optimizer steps")
        train_seconds = time.perf_counter() - started
        dev_id, dev_seconds, dev_slice = profile_dev_inference(
            torch, wrapper, head_module, device, args
        )
        io_bytes = sum(
            int(result.get("emitted_frames", 0)) * FRAME_SAMPLES * 2
            for result in per_source.values()
        )
        from experiments.psem_state_corrected_adaptation_gate import h_arm as h_arm_mod

        gpu_utilization = h_arm_mod._gpu_utilization()
        cpu_utilization = h_arm_mod._cpu_utilization()
        peak = 0
        try:
            if torch.cuda.is_available():
                peak = int(torch.cuda.max_memory_allocated(device))
        except Exception:
            peak = 0
        seconds_per_step = train_seconds / accum.optimizer_steps
        projected_wall = seconds_per_step * total_steps
        profile = {
            "artifact_role": "issue-121-temporal-profile",
            "arm": config.arm,
            "seed": config.seed,
            "config_hash": config.config_hash,
            "mapping_hash": mapping.manifest_hash,
            "weights_hash": binding["weights_hash"],
            "optimizer_steps": int(accum.optimizer_steps),
            "schedule_total_steps": int(total_steps),
            "schedule_loss_chunks": int(total_loss),
            "seconds_per_step": float(seconds_per_step),
            "peak_vram_bytes": int(peak),
            "dev_infer_seconds": {dev_id: float(dev_seconds)},
            "dev_source": dev_id,
            "dev_frontier_slice": dict(dev_slice),
            "io_bytes": int(io_bytes),
            "gpu_utilization": gpu_utilization,
            "cpu_utilization": cpu_utilization,
            "cpu_count": int(os.cpu_count() or 1),
            "projected_wall_seconds": float(projected_wall),
            "fit_sources": fit_sources,
            "sources": sorted(per_source),
            "seed_report": seed_report,
            "mapping_seconds": float(mapping_seconds),
            "training_seconds": float(train_seconds),
            "workers": arm_runtime.worker_receipt(getattr(args, "workers", None), len(fit_sources)),
        }
        if getattr(args, "gpu_price_usd", None) is not None:
            profile["projected_cost_usd"] = (
                float(projected_wall) / 3600.0 * float(args.gpu_price_usd)
            )
        arm_runtime.atomic_write_json(profile_receipt_path(run_dir), profile)
    finally:
        _decode_cache_clear()
        if stream_started:
            release_stream(config.arm)
        gpu_lock.__exit__(None, None, None)
    return {"authorization": receipt, "profile": profile}


def profile_dev_inference(
    torch: Any, wrapper: Any, head_module: Any, device: Any, args: Any
) -> tuple[str, float]:
    from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
        load_sessions as load_dev_snapshots,
    )
    from experiments.psem_sortformer_adaptation_depth.execution import (
        load_scoring_sessions,
    )
    from experiments.psem_state_corrected_adaptation_gate import material
    from experiments.psem_state_corrected_adaptation_gate.stages import (
        DEV_FAMILIES,
        resolve_dev_session,
    )
    from experiments.psem_training_strategy_gate.sampling import DEV_ROLE

    wrapper.eval()
    head_module.eval()
    dev_runtime = load_scoring_sessions(
        Path(args.corpus_root), Path(args.reference_root), DEV_ROLE
    )
    snapshots = load_dev_snapshots()
    for family in DEV_FAMILIES:
        for snapshot in snapshots:
            if not material.is_dev_family_session(snapshot, family):
                continue
            session = resolve_dev_session(dev_runtime, snapshot.source_id)
            started = time.perf_counter()
            raw = material.infer_dev_raw_logits(
                torch,
                wrapper,
                head_module,
                snapshot,
                session,
                Path(args.corpus_root),
                device,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            dev_seconds = time.perf_counter() - started
            member = {
                "source_id": str(snapshot.source_id),
                "snapshot": snapshot,
                "f0_raw": [float(v) for v in raw["f0_raw"]],
                "cand_raw": [float(v) for v in raw["cand_raw"]],
                "unmapped": [int(v) for v in raw["unmapped_frames"]],
            }
            from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

            grid = _frontier_grid([float(v) for v in raw["cand_raw"]])
            sliced = cross_mod.bounded_threshold_slice(grid, cross_mod.FRONTIER_SLICE_LIMIT - 1)
            slice_start = time.perf_counter()
            frontier = _group_score_frontier([member], "cand_raw", sliced, 100)
            slice_seconds = time.perf_counter() - slice_start
            return (
                str(snapshot.source_id),
                dev_seconds,
                {
                    "horizon_ms": 100,
                    "sampled_thresholds": len(sliced),
                    "total_thresholds": len(grid),
                    "points": len(frontier["points"]),
                    "seconds": float(slice_seconds),
                    "projected_seconds": cross_mod.project_frontier_cost(
                        slice_seconds, len(sliced), len(grid)
                    ),
                },
            )
    raise TemporalArmError("DEV snapshot has no representative source")


def run_full_command(config: Any, store: Path, args: Any) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    require_temporal_arm(config.arm)
    receipt = arm_runtime.check_authorization(config, Path(store))
    manifest = load_stage_bundle(args.bundle)
    binding = verify_execution_binding(config, manifest, args.checkpoint)
    run_dir = config.run_dir()
    require_profile_receipt(load_profile_receipt(run_dir), config)
    torch = load_torch()
    gpu_lock = arm_runtime.arm_gpu_lock(
        config.root, {"run_id": f"{config.arm}-{config.seed}", "arm": config.arm, "seed": int(config.seed), "command": "run"}
    )
    gpu_lock.__enter__()
    stream_started = False
    try:
        backend = resolve_backend(
            args.bundle,
            args.corpus_root,
            args.reference_root,
            args.sampling_manifest,
            args.checkpoint,
            args.nemo_checkout,
            args.dependency_lock,
            args.device,
        )
        fit_sources = arm_runtime.chronological_sources(list(binding["fit"]))
        calib_sources = arm_runtime.chronological_sources(list(binding["calib"]))
        sessions = backend["sessions"]
        rows_by_source = backend["rows_by_source"]
        weights = binding["class_weights"]
        seed_report = seed_temporal_from_config(torch, int(config.seed))
        opened = open_temporal_model(
            torch,
            args.checkpoint,
            args.nemo_checkout,
            args.dependency_lock,
            args.device,
            config.arm,
        )
        wrapper = opened["wrapper"]
        head_module = opened["head_module"]
        device = opened["device"]
        check_single_stream(config.arm)
        stream_started = True
    except BaseException:
        gpu_lock.__exit__(None, None, None)
        raise
    try:
        checkpoint = arm_runtime.load_source_checkpoint(run_dir, config.binding)
        completed = list(checkpoint.get("completed_sources", []))
        entries, target_binding = resolve_durable_targets(
            torch,
            run_dir,
            args.bundle,
            manifest,
            sessions,
            rows_by_source,
            args.corpus_root,
            device,
        fit_sources + [s for s in calib_sources if s not in fit_sources],
        getattr(args, "workers", None),
    )
        if read_mapping_hash(run_dir) is None:
            mapping = freeze_pass_mapping(
                torch,
                config.arm,
                config.seed,
                fit_sources,
                sessions,
                entries,
                args.corpus_root,
                device,
                wrapper,
                args.workers,
            )
            mapping_files = write_mapping_files(run_dir, mapping, config.config_hash)
        else:
            mapping = load_frozen_mapping(run_dir, config)
            mapping_files = [mapping_manifest_path(run_dir), mapping_file_path(run_dir)]
        total_loss = count_loss_chunks(entries, mapping, fit_sources)
        total_steps = schedule_total_steps(total_loss, ACCUMULATION_DEFAULT)
        optimizer = build_optimizer(torch, wrapper, head_module, config.arm)
        scheduler = build_scheduler(torch, optimizer, total_steps)
        accum = AccumState()
        if completed:
            latest = completed[-1]
            restore_blobs(
                torch, wrapper, head_module, optimizer, scheduler,
                accum, checkpoint["blobs"][latest], device,
            )
        if accum.pending == 0:
            optimizer.zero_grad()
        pending_wave: dict[str, Any] = {}
        remaining = arm_runtime.resume_plan(fit_sources, completed)
        n_workers = arm_runtime.resolve_workers(args.workers)
        pool = None
        if n_workers > 1:
            import concurrent.futures as _futures
            import multiprocessing as _mp

            pool = _futures.ProcessPoolExecutor(
                max_workers=int(n_workers),
                mp_context=_mp.get_context("spawn"),
                initializer=arm_runtime.spawn_worker_init,
            )
        try:
            from collections import deque as _deque

            pending: Any = _deque()
            depth = arm_runtime.resolve_prefetch_depth(int(n_workers), len(remaining), WAVEFORM_PREFETCH_MAX) if pool is not None else 0
            for source_id in remaining[:depth]:
                pending.append(
                    pool.submit(
                        decode_waveform_task,
                        decode_payload(sessions[source_id], args.corpus_root),
                    )
                )
            for position, source_id in enumerate(remaining):
                if pool is not None:
                    assert len(pending) > 0
                    decoded = pending.popleft().result()
                    ahead = remaining[position + depth : position + depth + 1]
                    if ahead:
                        pending.append(
                            pool.submit(
                                decode_waveform_task,
                                decode_payload(sessions[ahead[0]], args.corpus_root),
                            )
                        )
                else:
                    decoded = decode_waveform_task(
                        decode_payload(sessions[source_id], args.corpus_root)
                    )
                prep = assemble_prep(
                    torch, sessions[source_id], entries[source_id], decoded, device
                )
                del decoded
                carry = TemporalCarry()
                result = train_source(
                    torch,
                    wrapper,
                    head_module,
                    optimizer,
                    scheduler,
                    carry,
                    accum,
                    prep,
                    dict(mapping.for_source(source_id)["slot_of"]),
                    weights,
                    device,
                    ACCUMULATION_DEFAULT,
                    None,
                    source_id == fit_sources[-1],
                )
                blobs = snapshot_blobs(
                    torch, wrapper, head_module, optimizer, scheduler, accum
                )
                arm_runtime.save_source_checkpoint(
                    run_dir, source_id, list(completed), config.binding, blobs
                )
                completed.append(source_id)
                stats = mapping_coverage_agreement(
                    prep, dict(mapping.for_source(source_id)["slot_of"])
                )
                arm_runtime.record_mapping_diagnostics(
                    run_dir,
                    source_id,
                    float(stats["coverage"]),
                    float(stats["agreement"]),
                    {
                        "mapped_episodes": stats["mapped_episodes"],
                        "total_episodes": stats["total_episodes"],
                        "mapped_frames": stats["mapped_frames"],
                        "support_frames": stats["support_frames"],
                        "emitted_frames": result["emitted_frames"],
                        "optimizer_steps": result["optimizer_steps"],
                        "pending": result["pending"],
                        "mapping_hash": mapping.manifest_hash,
                    },
                )
                pending_wave[source_id] = result
                del prep
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        calib_artifact = run_calib_command(
            torch, wrapper, head_module, device, config, run_dir,
            sessions, entries, calib_sources, args,
        )
        dev_artifact = run_dev_command(
            torch, wrapper, head_module, device, config, run_dir,
            calib_artifact["calibrators"], args,
        )
        experiment = {
            "arm": config.arm,
            "seed": config.seed,
            "config_hash": config.config_hash,
            "legacy_arm": legacy_policy_arm(config.arm),
            "weights_hash": binding["weights_hash"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "mapping_hash": mapping.manifest_hash,
            "schedule_total_steps": int(total_steps),
            "schedule_loss_chunks": int(total_loss),
        }
        data_doc = {
            "fit_sources": fit_sources,
            "calib_sources": calib_sources,
            "completed_sources": list(completed),
            "sampling_sha256": str(manifest.get("sampling_sha256")),
            "targets": {sid: dict(record) for sid, record in target_binding.items()},
        }
        module_mode = {
            "policy": opened["policy_receipt"],
            "modes": opened["modes"],
            "head_input_dim": HEAD_INPUT_DIM,
        }
        training_doc = {
            "optimizer": dict(arm_runtime.OPTIMIZER_CONTRACT),
            "accumulation": ACCUMULATION_DEFAULT,
            "pending": accum.pending,
            "optimizer_steps": accum.optimizer_steps,
            "loss_chunks": accum.loss_chunks,
            "empty_chunks": accum.empty_chunks,
            "seed_report": seed_report,
            "workers": arm_runtime.worker_receipt(getattr(args, "workers", None), len(fit_sources)),
            "sources": merge_training_sources(
                load_prior_training_sources(run_dir),
                {sid: pending_wave.get(sid, {}) for sid in fit_sources},
            ),
        }
        artifacts = write_arm_artifacts(
            run_dir,
            {
                "experiment_manifest.json": experiment,
                "data_sampling_calibration_manifest.json": data_doc,
                "parameter_module_mode_receipt.json": module_mode,
                "training_metrics.json": training_doc,
                "calibration_metrics.json": calib_artifact["document"],
                "dev_frontier.json": dev_artifact["document"],
            },
        )
        for extra in calib_artifact["files"] + dev_artifact["files"]:
            artifacts.append(Path(extra))
        artifacts.extend(mapping_files)
        artifacts.append(profile_receipt_path(run_dir))
        manifest_doc = {
            "arm": config.arm,
            "seed": config.seed,
            "config_hash": config.config_hash,
            "mapping_hash": mapping.manifest_hash,
            "target_binding": arm_runtime.canonical_sha256(
                {sid: target_binding[sid]["sha256"] for sid in sorted(target_binding)}
            ),
            "sources": sorted(completed),
        }
        final_path = arm_runtime.write_final_manifest(run_dir, manifest_doc, artifacts)
    finally:
        _decode_cache_clear()
        if stream_started:
            release_stream(config.arm)
        gpu_lock.__exit__(None, None, None)
    return {"authorization": receipt, "final_manifest": str(final_path)}


def load_prior_training_sources(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "training_metrics.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TemporalArmError(
            f"prior training metrics are unreadable: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TemporalArmError("prior training metrics are invalid")
    sources = payload.get("sources", {})
    if not isinstance(sources, dict):
        raise TemporalArmError("prior training metrics are invalid")
    return {str(sid): dict(entry) for sid, entry in sources.items()}


def merge_training_sources(
    prior: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    merged = {str(sid): dict(entry) for sid, entry in prior.items()}
    for sid, entry in current.items():
        merged[str(sid)] = dict(entry)
    return merged


def write_arm_artifacts(run_dir: Path, docs: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime

    paths: list[Path] = []
    for name in (
        "experiment_manifest.json",
        "data_sampling_calibration_manifest.json",
        "parameter_module_mode_receipt.json",
        "training_metrics.json",
        "calibration_metrics.json",
        "dev_frontier.json",
    ):
        paths.append(arm_runtime.atomic_write_json(Path(run_dir) / name, dict(docs[name])))
    return paths


def run_calib_command(
    torch: Any,
    wrapper: Any,
    head_module: Any,
    device: Any,
    config: Any,
    run_dir: Path,
    sessions: Mapping[str, Any],
    entries: Mapping[str, dict[str, Any]],
    calib_sources: Sequence[str],
    args: Any,
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime
    from experiments.psem_state_corrected_adaptation_gate import material
    from experiments.psem_state_corrected_adaptation_gate.stages import (
        fit_calibrators,
    )

    wrapper.eval()
    head_module.eval()
    buffers: dict[str, list[float]] = {"f0": [], "cand": [], "targets": []}
    files: list[str] = []
    per_source: dict[str, Any] = {}
    for source_id in calib_sources:
        entry = entries[source_id]
        prep = {
            "source_id": source_id,
            "authority": entry["authority"],
            "multiplicity": list(entry["multiplicity"]),
            "episode_ids": list(entry["episode_ids"]),
            "num_frames": int(entry["num_frames"]),
        }
        wave = load_source_waveform(torch, sessions[source_id], args.corpus_root, device)
        authority = prep["authority"]
        try:
            waveform = wave["waveform"]
            anchor_active = [a == 1.0 for a in authority.y_anchor]
            with torch.no_grad():
                out = material.infer_arm_logits(
                    torch,
                    wrapper,
                    head_module,
                    waveform,
                    [None if v is None else str(v) for v in prep["episode_ids"]],
                    anchor_active,
                    list(authority.valid),
                    int(prep["num_frames"]),
                    device,
                )
        finally:
            release_waveform(torch, wave["waveform"])
            del wave
        f0_all = out["f0_logit"].flatten().tolist()
        cand_all = (out["f0_logit"] + out["z_residual"]).flatten().tolist()
        targets_all = [float(v) for v in authority.y_replace]
        mapped = [
            i not in set(out["unmapped_frames"]) for i in range(int(prep["num_frames"]))
        ]
        kept, coverage = material.mask_calibration(targets_all, list(authority.valid), mapped)
        material.extend_calibration_buffers(buffers, f0_all, cand_all, targets_all, kept)
        path = arm_runtime.save_source_predictions(
            run_dir,
            source_id,
            {
                "f0_raw": f0_all,
                "cand_raw": cand_all,
                "targets": targets_all,
                "kept": list(kept),
                "mapping_rows": out["mapping_rows"],
            },
            config.binding,
        )
        files.append(str(path))
        per_source[source_id] = {
            "frames": int(prep["num_frames"]),
            "kept": len(kept),
            "coverage": dict(coverage),
        }
    f0_fit, cand_fit = fit_calibrators(buffers["f0"], buffers["cand"], buffers["targets"])
    document = {
        "role": CALIB_ROLE,
        "sources": per_source,
        "f0": dict(f0_fit),
        "candidate": dict(cand_fit),
    }
    return {"document": document, "files": files, "calibrators": (f0_fit, cand_fit)}


def run_dev_command(
    torch: Any,
    wrapper: Any,
    head_module: Any,
    device: Any,
    config: Any,
    run_dir: Path,
    calibrators: Any,
    args: Any,
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import arm_runtime
    from experiments.psem_state_corrected_adaptation_gate import material
    from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
        load_sessions as load_dev_snapshots,
    )
    from experiments.psem_sortformer_adaptation_depth.execution import (
        load_scoring_sessions,
    )
    from experiments.psem_state_corrected_adaptation_gate.stages import (
        DEV_FAMILIES,
        _prepare_dev_arrays,
        resolve_dev_session,
    )
    from experiments.psem_training_strategy_gate.sampling import DEV_ROLE

    cal_f0, cal_cand = calibrators
    wrapper.eval()
    head_module.eval()
    dev_runtime = load_scoring_sessions(
        Path(args.corpus_root), Path(args.reference_root), DEV_ROLE
    )
    snapshots = load_dev_snapshots()
    files: list[str] = []
    per_source: dict[str, Any] = {}
    members: list[dict[str, Any]] = []
    dev_entries: dict[str, Any] = {}
    for family in DEV_FAMILIES:
        for snapshot in snapshots:
            if not material.is_dev_family_session(snapshot, family):
                continue
            session = resolve_dev_session(dev_runtime, snapshot.source_id)
            raw = material.infer_dev_raw_logits(
                torch, wrapper, head_module, snapshot, session,
                Path(args.corpus_root), device,
            )
            path = arm_runtime.save_source_predictions(
                run_dir,
                snapshot.source_id,
                {
                    "f0_raw": raw["f0_raw"],
                    "cand_raw": raw["cand_raw"],
                    "targets": raw["target"],
                    "kept": raw["kept"],
                    "mapping_rows": raw["mapping_rows"],
                },
                config.binding,
            )
            files.append(str(path))
            prep = _prepare_dev_arrays(
                snapshot,
                [float(v) for v in raw["f0_raw"]],
                [float(v) for v in raw["cand_raw"]],
                [float(v) for v in raw["target"]],
                [bool(v) for v in raw["valid"]],
                [bool(v) for v in raw["mapped_flags"]],
                [int(v) for v in raw["unmapped_frames"]],
                dict(cal_f0),
                dict(cal_cand),
            )
            dev_entry = {
                "f0_cal": [float(v) for v in prep["f0_cal"]],
                "cand_cal": [float(v) for v in prep["cand_cal"]],
                "kept": [int(v) for v in prep["kept"]],
            }
            probs = prepare_dev_scores(
                [float(v) for v in raw["f0_raw"]],
                [float(v) for v in raw["cand_raw"]],
                [bool(v) for v in raw["mapped_flags"]],
                [int(v) for v in raw["unmapped_frames"]],
                {"f0": dict(cal_f0), "candidate": dict(cal_cand)},
            )
            members.append(
                {
                    "source_id": str(snapshot.source_id),
                    "family": str(family),
                    "snapshot": snapshot,
                    "f0_raw": [float(v) for v in raw["f0_raw"]],
                    "cand_raw": [float(v) for v in raw["cand_raw"]],
                    "f0_prob": list(probs["f0_prob"]),
                    "cand_raw_prob": list(probs["cand_raw_prob"]),
                    "cand_cal_prob": list(probs["cand_cal_prob"]),
                    "unmapped": [],
                    "f0_cal": [float(v) for v in dev_entry["f0_cal"]],
                    "cand_cal": [float(v) for v in dev_entry["cand_cal"]],
                    "kept": [int(v) for v in dev_entry["kept"]],
                    "target": [float(v) for v in raw["target"]],
                    "frames": int(raw["grid_frames"]),
                }
            )
            dev_entries[str(snapshot.source_id)] = dev_entry
    group_members: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        group_members.setdefault(_corpus_group(member["family"]), []).append(member)
    for required in ("ami", "alimeeting"):
        if not group_members.get(required):
            raise TemporalArmError(
                f"DEV aggregation requires {required} sources"
            )
    expected_baseline = "R-H-SC" if config.arm == "R-T2-SC" else "R-T2-SC"
    baseline = read_baseline_frontier(Path(args.baseline_frontier), expected_baseline)
    frontier_start = time.perf_counter()
    groups_obj, comparisons, gate_evidence, source_frontiers = build_dev_aggregates(
        config, group_members, baseline, dev_entries, getattr(args, "workers", None)
    )
    gate_evidence["frontier_slice_seconds"] = time.perf_counter() - frontier_start
    for member in members:
        sid = str(member["source_id"])
        per_source[sid] = {
            "family": str(member["family"]),
            "frames": int(member["frames"]),
            "horizons": {
                horizon: {
                    kind: {
                        "budget": float(source_frontiers[sid][horizon][kind]["f0"].false_cuts_per_hour),
                        "candidate_points": [
                            {
                                "threshold": float(p.threshold),
                                "false_cuts_per_hour": float(p.false_cuts_per_hour),
                                "contamination": float(p.contamination),
                                "miss_rate": float(p.miss_rate),
                            }
                            for p in source_frontiers[sid][horizon][kind]["points"]
                        ],
                        "reference": {
                            "threshold": float(source_frontiers[sid][horizon][kind]["f0"].threshold),
                            "false_cuts_per_hour": float(source_frontiers[sid][horizon][kind]["f0"].false_cuts_per_hour),
                            "contamination": float(source_frontiers[sid][horizon][kind]["f0"].contamination),
                            "miss_rate": float(source_frontiers[sid][horizon][kind]["f0"].miss_rate),
                        },
                    }
                    for kind in ("calibrated", "raw")
                }
                for horizon in ("100", "300", "500")
            },
        }
    document = assemble_temporal_dev_document(
        config.arm,
        groups_obj,
        per_source,
        comparisons,
        gate_evidence,
        str(args.baseline_frontier),
    )
    return {"document": document, "files": files}


def assemble_temporal_dev_document(
    arm: str,
    groups_obj: Mapping[str, Any],
    per_source: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    gate_evidence: Mapping[str, Any],
    baseline: str,
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod

    horizons: dict[str, Any] = {}
    for horizon_ms in frontier_mod.HORIZONS_MS:
        horizon = str(horizon_ms)
        horizons[horizon] = {}
        for name in ("macro", "ami", "alimeeting", "pooled"):
            horizons[horizon][name] = {}
            for kind in ("calibrated", "raw"):
                cell = groups_obj[horizon][kind][name]
                horizons[horizon][name][kind] = cross_mod.build_block(
                    _point_dicts(cell["points"]),
                    _point_dicts([cell["f0"]])[0],
                    dict(cell["metrics"]),
                    what=f"{name}/{kind} at horizon {horizon}",
                )
    document = {
        "artifact_role": cross_mod.ARTIFACT_ROLE,
        "version": cross_mod.CANONICAL_VERSION,
        "arm": str(arm),
        "horizons_ms": [int(h) for h in frontier_mod.HORIZONS_MS],
        "group_order": list(cross_mod.GROUP_ORDER),
        "horizons": horizons,
        "sources": dict(per_source),
        "families": sorted({str(entry["family"]) for entry in per_source.values()}),
        "comparisons": dict(comparisons),
        "gate_evidence": dict(gate_evidence),
        "baseline": str(baseline),
    }
    cross_mod.validate_canonical(document)
    return document


def _corpus_group(family: str) -> str:
    mapping = {"ami_mix_headset": "ami", "alimeeting_far_ch0": "alimeeting"}
    if str(family) not in mapping:
        raise TemporalArmError(f"DEV family is outside AMI/AliMeeting: {family}")
    return mapping[str(family)]


def _member_metrics_at(dev_snapshot: Any, scores: Sequence[float], threshold: float, horizon_ms: int) -> dict[str, Any]:
    import numpy as _np
    from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
        decode_scores,
        session_metrics,
    )

    events = decode_scores(
        dev_snapshot,
        _np.asarray([float(v) for v in scores], dtype=_np.float64),
        threshold=float(threshold),
        confirmation_ms=int(horizon_ms),
    )
    return dict(session_metrics(dev_snapshot, events))


def _aggregate_dev_metrics(metrics_list: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    total_cuts = sum(int(m["false_cut_count"]) for m in metrics_list)
    total_seconds = sum(float(m["active_speech_seconds"]) for m in metrics_list)
    total_refs = sum(int(m["reference_replacement_count"]) for m in metrics_list)
    total_missed = sum(int(m["missed_replacement_count"]) for m in metrics_list)
    total_contamination = sum(
        float(m["exclusive_other_contamination_seconds"]) for m in metrics_list
    )
    if total_seconds <= 0:
        raise TemporalArmError("pooled DEV group has no active speech")
    if total_refs <= 0:
        raise TemporalArmError("pooled DEV group has no reference replacements")
    hours = total_seconds / 3600.0
    return {
        "false_cut_count": float(total_cuts),
        "active_speech_hours": hours,
        "reference_replacement_count": float(total_refs),
        "missed_replacement_count": float(total_missed),
        "contamination_seconds": float(total_contamination),
    }


def _pooled_frontier_point(aggregated: Mapping[str, float], threshold: float) -> Any:
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod

    return frontier_mod.FrontierPoint(
        threshold=float(threshold),
        false_cuts_per_hour=aggregated["false_cut_count"] / aggregated["active_speech_hours"],
        contamination=aggregated["contamination_seconds"] / aggregated["active_speech_hours"],
        miss_rate=aggregated["missed_replacement_count"] / aggregated["reference_replacement_count"],
    )


def _mean_frontier_point(left: Any, right: Any) -> Any:
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod

    if float(left.threshold) != float(right.threshold):
        raise TemporalArmError("macro frontier grid mismatch")
    return frontier_mod.FrontierPoint(
        threshold=float(left.threshold),
        false_cuts_per_hour=(float(left.false_cuts_per_hour) + float(right.false_cuts_per_hour)) / 2.0,
        contamination=(float(left.contamination) + float(right.contamination)) / 2.0,
        miss_rate=(float(left.miss_rate) + float(right.miss_rate)) / 2.0,
    )


def _masked_scores(scores: Sequence[float], unmapped: Sequence[int]) -> list[float]:
    blocked = {int(i) for i in unmapped}
    return [float("-inf") if i in blocked else float(v) for i, v in enumerate(scores)]


def prepare_dev_scores(
    f0_raw: Sequence[float],
    cand_raw: Sequence[float],
    mapped_flags: Sequence[bool],
    unmapped: Sequence[int],
    calibrators: Mapping[str, Any],
) -> dict[str, list[float]]:
    from experiments.psem_state_corrected_adaptation_gate import calibrate as calibrate_mod

    flags = [bool(v) for v in mapped_flags]
    if not (len(f0_raw) == len(cand_raw) == len(flags)) or not flags:
        raise TemporalArmError("dev score/mapping geometry differs")
    cal_f0 = dict(calibrators["f0"])
    cal_cand = dict(calibrators["candidate"])
    if not float(cal_f0.get("slope", 0.0)) > 0 or not float(cal_cand.get("slope", 0.0)) > 0:
        raise TemporalArmError("dev calibrator slope must be positive")
    f0_cal = calibrate_mod.apply_affine(
        [float(v) for v in f0_raw], float(cal_f0["slope"]), float(cal_f0["intercept"])
    )
    cand_cal = calibrate_mod.apply_affine(
        [float(v) for v in cand_raw], float(cal_cand["slope"]), float(cal_cand["intercept"])
    )
    blocked = {int(i) for i in unmapped}
    for i, flag in enumerate(flags):
        if not flag:
            blocked.add(int(i))
    return {
        "f0_prob": [
            float("-inf") if i in blocked else calibrate_mod.sigmoid(float(v))
            for i, v in enumerate(f0_raw)
        ],
        "cand_raw_prob": [
            float("-inf") if i in blocked else calibrate_mod.sigmoid(float(v))
            for i, v in enumerate(cand_raw)
        ],
        "cand_cal_prob": [
            float("-inf") if i in blocked else calibrate_mod.sigmoid(float(v))
            for i, v in enumerate(cand_cal)
        ],
        "f0_cal": [float(v) for v in f0_cal],
        "cand_cal": [float(v) for v in cand_cal],
    }


def _frontier_grid(pooled_scores: Sequence[float]) -> list[float]:
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod

    try:
        return frontier_mod.unique_thresholds([float(v) for v in pooled_scores])
    except frontier_mod.FrontierError as exc:
        raise TemporalArmError(f"DEV group has no frontier scores: {exc}") from exc


def _group_score_frontier(
    group_members: Sequence[Mapping[str, Any]],
    score_key: str,
    grid: Sequence[float],
    horizon_ms: int,
    workers: int | None = 1,
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod

    members = list(group_members)
    if not members:
        raise TemporalArmError("DEV group has no member sources")
    ordered_grid = [float(t) for t in grid]
    entries: dict[str, Any] = {}
    for index, member in enumerate(members):
        blocked = {int(i) for i in member["unmapped"]}
        entries[f"member-{index:04d}"] = {
            "dev": member["snapshot"],
            "scores": {
                "cand": [
                    float("-inf") if i in blocked else float(v)
                    for i, v in enumerate(member[score_key])
                ],
                "f0": [
                    float("-inf") if i in blocked else float(v)
                    for i, v in enumerate(member["f0_raw"])
                ],
            },
        }
    wave_grids = {
        key: {"cand": list(ordered_grid), "f0": [float(frontier_mod.RAW_REFERENCE_THRESHOLD)]}
        for key in entries
    }
    tasks = cross_mod.plan_exact_tasks(wave_grids, [int(horizon_ms)])
    primitives, _ = cross_mod.run_exact_wave(entries, tasks, workers)
    indexed = {
        key: cross_mod.index_threshold_rows(primitives[key]["cand"][int(horizon_ms)])
        for key in entries
    }
    points = []
    for threshold in ordered_grid:
        rows = [indexed[key][float(threshold)] for key in entries]
        totals = cross_mod.sum_primitives(rows)
        points.append(
            frontier_mod.FrontierPoint(
                threshold=float(threshold),
                false_cuts_per_hour=totals["false_cut_count"] / (totals["active_speech_seconds"] / 3600.0),
                contamination=totals["exclusive_other_contamination_seconds"]
                / (totals["active_speech_seconds"] / 3600.0),
                miss_rate=totals["missed_replacement_count"] / totals["reference_replacement_count"],
            )
        )
    f0_totals = cross_mod.sum_primitives(
        [primitives[key]["f0"][int(horizon_ms)][0] for key in entries]
    )
    f0_point = frontier_mod.FrontierPoint(
        threshold=float(frontier_mod.RAW_REFERENCE_THRESHOLD),
        false_cuts_per_hour=f0_totals["false_cut_count"] / (f0_totals["active_speech_seconds"] / 3600.0),
        contamination=f0_totals["exclusive_other_contamination_seconds"]
        / (f0_totals["active_speech_seconds"] / 3600.0),
        miss_rate=f0_totals["missed_replacement_count"] / f0_totals["reference_replacement_count"],
    )
    return {"points": points, "f0": f0_point, "grid": list(ordered_grid)}


def _group_dev_metrics(
    group_members: Sequence[Mapping[str, Any]],
    dev_entries: Mapping[str, Any],
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import calibrate as calibrate_mod

    kept_raw: list[float] = []
    kept_target: list[float] = []
    kept_cand_cal: list[float] = []
    kept_f0_cal: list[float] = []
    frames = 0
    kept_frames = 0
    for member in group_members:
        source_id = str(member["source_id"])
        entry = dev_entries[source_id]
        kept = [int(v) for v in entry["kept"]]
        raw_cand = [float(v) for v in member["cand_raw"]]
        target = [float(v) for v in member["target"]]
        kept_raw.extend([calibrate_mod.sigmoid(raw_cand[i]) for i in kept])
        kept_target.extend([float(target[i]) for i in kept])
        kept_cand_cal.extend([float(entry["cand_cal"][i]) for i in kept])
        kept_f0_cal.extend([float(entry["f0_cal"][i]) for i in kept])
        frames += len(target)
        kept_frames += len(kept)
    if not kept_target:
        raise TemporalArmError("DEV group has no kept frames")
    return {
        "frames": int(frames),
        "kept_frames": int(kept_frames),
        "raw_ap": float(calibrate_mod.average_precision(kept_raw, kept_target)),
        "f0_nll": float(calibrate_mod.nll_loss(kept_f0_cal, kept_target)),
        "f0_brier": float(calibrate_mod.brier_score(kept_f0_cal, kept_target)),
        "candidate_nll": float(calibrate_mod.nll_loss(kept_cand_cal, kept_target)),
        "candidate_brier": float(calibrate_mod.brier_score(kept_cand_cal, kept_target)),
    }


def _mean_dev_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frames": int(left["frames"]) + int(right["frames"]),
        "kept_frames": int(left["kept_frames"]) + int(right["kept_frames"]),
        "raw_ap": (float(left["raw_ap"]) + float(right["raw_ap"])) / 2.0,
        "f0_nll": (float(left["f0_nll"]) + float(right["f0_nll"])) / 2.0,
        "f0_brier": (float(left["f0_brier"]) + float(right["f0_brier"])) / 2.0,
        "candidate_nll": (float(left["candidate_nll"]) + float(right["candidate_nll"])) / 2.0,
        "candidate_brier": (
            (float(left["candidate_brier"]) + float(right["candidate_brier"])) / 2.0
        ),
    }


def _macro_baseline_points(base: Mapping[str, Any], kind: str, horizon: str) -> list[Any]:
    macro = base.get("macro")
    if isinstance(macro, dict) and isinstance(macro.get(kind), dict):
        return list(macro[kind]["points"])
    raise TemporalArmError(f"baseline frontier lacks macro/{kind} at horizon {horizon}")


def build_dev_aggregates(
    config: Any,
    group_members: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline: Mapping[str, Any],
    dev_entries: Mapping[str, Any],
    workers: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod

    groups_obj: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    gate_evidence: dict[str, Any] = {"first": "macro", "horizons": {}}
    source_frontiers: dict[str, Any] = {}
    pooled_members_all = list(group_members.get("ami", [])) + list(
        group_members.get("alimeeting", [])
    )
    ami_metrics = _group_dev_metrics(list(group_members.get("ami", [])), dev_entries)
    ali_metrics = _group_dev_metrics(
        list(group_members.get("alimeeting", [])), dev_entries
    )
    pooled_metrics = _group_dev_metrics(pooled_members_all, dev_entries)
    macro_metrics = _mean_dev_metrics(ami_metrics, ali_metrics)
    group_metrics = {
        "macro": macro_metrics,
        "ami": ami_metrics,
        "alimeeting": ali_metrics,
        "pooled": pooled_metrics,
    }
    ami_all = list(group_members.get("ami", []))
    ali_all = list(group_members.get("alimeeting", []))
    pooled_all = ami_all + ali_all
    seen_ids: set[str] = set()
    for member in pooled_all:
        sid = str(member["source_id"])
        if sid in seen_ids:
            raise TemporalArmError(f"DEV source id is duplicated: {sid}")
        seen_ids.add(sid)
    entries: dict[str, Any] = {}
    for member in pooled_all:
        sid = str(member["source_id"])
        entries[sid] = {
            "dev": member["snapshot"],
            "scores": {
                "raw": [float(v) for v in member["cand_raw_prob"]],
                "calibrated": [float(v) for v in member["cand_cal_prob"]],
                "f0": [float(v) for v in member["f0_prob"]],
            },
        }
    grids = {}
    for kind, prob_key in (
        ("raw", "cand_raw_prob"),
        ("calibrated", "cand_cal_prob"),
    ):
        union: set[float] = set()
        for m in pooled_all:
            for v in m[prob_key]:
                if float(v) != float("-inf"):
                    union.add(float(v))
        if not union:
            raise TemporalArmError(f"DEV group has no frontier scores for {kind}")
        grids[kind] = sorted(union, reverse=True)
    grids["f0"] = [float(frontier_mod.RAW_REFERENCE_THRESHOLD)]
    wave_grids = {sid: {kind: list(grids[kind]) for kind in grids} for sid in entries}
    tasks = cross_mod.plan_exact_tasks(wave_grids, list(frontier_mod.HORIZONS_MS))
    primitives, wave_receipt = cross_mod.run_exact_wave(entries, tasks, workers)
    ami_ids = sorted(str(m["source_id"]) for m in ami_all)
    ali_ids = sorted(str(m["source_id"]) for m in ali_all)
    pooled_ids = sorted(set(ami_ids) | set(ali_ids))

    def _assembled_point(rows: list[dict[str, Any]], threshold: float) -> Any:
        try:
            totals = cross_mod.sum_primitives(rows)
            point = cross_mod.pooled_point_from_sums(
                totals["false_cut_count"],
                totals["active_speech_seconds"],
                totals["reference_replacement_count"],
                totals["missed_replacement_count"],
                totals["exclusive_other_contamination_seconds"],
                float(threshold),
            )
        except cross_mod.CrossFrontierError as exc:
            raise TemporalArmError(f"DEV group frontier failed: {exc}") from exc
        return frontier_mod.FrontierPoint(
            threshold=float(point["threshold"]),
            false_cuts_per_hour=float(point["false_cuts_per_hour"]),
            contamination=float(point["contamination"]),
            miss_rate=float(point["miss_rate"]),
        )

    member_index: dict[str, Any] = {}
    for sid in pooled_ids:
        member_index[sid] = {}
        for kind in ("raw", "calibrated", "f0"):
            member_index[sid][kind] = {}
            for horizon_ms in frontier_mod.HORIZONS_MS:
                member_index[sid][kind][int(horizon_ms)] = cross_mod.index_threshold_rows(
                    primitives[sid][kind][int(horizon_ms)]
                )

    def _group_frontier_from_wave(
        member_ids: list[str], kind: str, horizon_ms: int
    ) -> dict[str, Any]:
        points = []
        for threshold in grids[kind]:
            rows = [
                member_index[sid][kind][int(horizon_ms)][float(threshold)]
                for sid in member_ids
            ]
            points.append(_assembled_point(rows, float(threshold)))
        f0 = _assembled_point(
            [primitives[sid]["f0"][int(horizon_ms)][0] for sid in member_ids],
            float(frontier_mod.RAW_REFERENCE_THRESHOLD),
        )
        return {"points": points, "f0": f0, "grid": list(grids[kind])}

    thresholds_plan: dict[str, Any] = {}
    for horizon_ms in frontier_mod.HORIZONS_MS:
        horizon = str(horizon_ms)
        ami = ami_all
        ali = ali_all
        pooled_members = pooled_all
        horizon_groups: dict[str, Any] = {}
        horizon_compare: dict[str, Any] = {}
        for kind in ("calibrated", "raw"):
            ami_frontier = _group_frontier_from_wave(ami_ids, kind, horizon_ms)
            ali_frontier = _group_frontier_from_wave(ali_ids, kind, horizon_ms)
            pooled_frontier = _group_frontier_from_wave(pooled_ids, kind, horizon_ms)
            macro_points = cross_mod.macro_average_points(
                _point_dicts(ami_frontier["points"]),
                _point_dicts(ali_frontier["points"]),
                what=f"macro/{kind} at horizon {horizon}",
            )
            macro_ref = cross_mod.macro_average_reference(
                _point_dicts([ami_frontier["f0"]])[0],
                _point_dicts([ali_frontier["f0"]])[0],
                what=f"macro/{kind} at horizon {horizon}",
            )
            macro_f0 = frontier_mod.FrontierPoint(
                threshold=float(macro_ref["threshold"]),
                false_cuts_per_hour=float(macro_ref["false_cuts_per_hour"]),
                contamination=float(macro_ref["contamination"]),
                miss_rate=float(macro_ref["miss_rate"]),
            )
            base = baseline["horizons"][horizon]
            base_macro_budget = frontier_mod.FrontierPoint(
                threshold=float(base["ami"]["raw"]["reference"].threshold),
                false_cuts_per_hour=(
                    float(base["ami"]["raw"]["reference"].false_cuts_per_hour)
                    + float(base["alimeeting"]["raw"]["reference"].false_cuts_per_hour)
                )
                / 2.0,
                contamination=(
                    float(base["ami"]["raw"]["reference"].contamination)
                    + float(base["alimeeting"]["raw"]["reference"].contamination)
                )
                / 2.0,
                miss_rate=(
                    float(base["ami"]["raw"]["reference"].miss_rate)
                    + float(base["alimeeting"]["raw"]["reference"].miss_rate)
                )
                / 2.0,
            )
            arm_macro = {"points": macro_points, "f0": macro_f0}
            by_name = {
                "macro": arm_macro,
                "ami": ami_frontier,
                "alimeeting": ali_frontier,
                "pooled": pooled_frontier,
            }
            horizon_groups[kind] = {
                name: {
                    "points": list(frontier["points"]),
                    "f0": frontier["f0"],
                    "metrics": dict(group_metrics[name]),
                    "sources": sorted(
                        str(m["source_id"])
                        for m in (ami if name == "ami" else ali if name == "alimeeting" else pooled_members)
                    ),
                }
                for name, frontier in by_name.items()
            }
            for name in ("macro", "ami", "alimeeting", "pooled"):
                if name == "macro":
                    budget = base_macro_budget
                    baseline_h_points = _macro_baseline_points(base, kind, horizon)
                else:
                    budget = base[name]["raw"]["reference"]
                    baseline_h_points = base[name][kind]["points"]
                corpora = {
                    "horizon_ms": int(horizon_ms),
                    "kind": kind,
                    "group": name,
                    "sources": list(horizon_groups[kind][name]["sources"]),
                }
                if config.arm == "R-T2-SC":
                    compared = compare_t2_to_h_f0(
                        horizon_groups[kind][name]["points"],
                        budget,
                        corpora,
                        baseline_points=baseline_h_points,
                    )
                else:
                    compared = compare_ta_to_t2_f0(
                        horizon_groups[kind][name]["points"],
                        budget,
                        corpora,
                        baseline_points=baseline_h_points,
                    )
                horizon_compare.setdefault(name, {})[kind] = compared
            thresholds_plan.setdefault(horizon, {})[kind] = {
                name: len(grids[kind]) for name in ("macro", "ami", "alimeeting", "pooled")
            }
        for sid in pooled_ids:
            source_frontiers.setdefault(sid, {})[horizon] = {}
            for kind in ("calibrated", "raw"):
                indexed = cross_mod.index_threshold_rows(primitives[sid][kind][int(horizon_ms)])
                source_frontiers[sid][horizon][kind] = {
                    "points": [
                        _assembled_point([indexed[float(t)]], float(t)) for t in grids[kind]
                    ],
                    "f0": _assembled_point(
                        primitives[sid]["f0"][int(horizon_ms)],
                        float(frontier_mod.RAW_REFERENCE_THRESHOLD),
                    ),
                }
        groups_obj[horizon] = horizon_groups
        comparisons[horizon] = horizon_compare
        gate_evidence["horizons"][horizon] = {
            "calibrated_useful": bool(horizon_compare["macro"]["calibrated"]["useful"]),
            "raw_useful": bool(horizon_compare["macro"]["raw"]["useful"]),
            "macro_budget": float(horizon_compare["macro"]["calibrated"]["budget"]),
        }
    gate_evidence["phase"] = {**wave_receipt, "thresholds": thresholds_plan}
    return groups_obj, comparisons, gate_evidence, source_frontiers


def _frontier_point_from(payload: Any, what: str) -> Any:
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod

    if not isinstance(payload, dict):
        raise TemporalArmError(f"baseline {what} is not an object")
    try:
        point = frontier_mod.FrontierPoint(
            threshold=float(payload["threshold"]),
            false_cuts_per_hour=float(payload["false_cuts_per_hour"]),
            contamination=float(payload["contamination"]),
            miss_rate=float(payload["miss_rate"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TemporalArmError(f"baseline {what} is malformed: {exc}") from exc
    return point


def _parse_baseline_block(block: Any, name: str, kind: str, horizon: str) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod

    if not isinstance(block, dict):
        raise TemporalArmError(
            f"baseline frontier lacks {kind} block for {name} at horizon {horizon}"
        )
    points_raw = block.get("points")
    if not isinstance(points_raw, list) or not points_raw:
        raise TemporalArmError(
            f"baseline frontier has no points for {name}/{kind} at horizon {horizon}"
        )
    points = [
        _frontier_point_from(p, f"{name}/{kind} point at horizon {horizon}")
        for p in points_raw
    ]
    reference = _frontier_point_from(
        block.get("reference"),
        f"{name}/{kind} reference at horizon {horizon}",
    )
    if float(reference.threshold) != float(frontier_mod.RAW_REFERENCE_THRESHOLD):
        raise TemporalArmError(
            f"baseline reference is not the raw-0.5 point for {name}/{kind} at horizon {horizon}"
        )
    return {"points": points, "reference": reference}


def read_baseline_frontier(path: Path, expected_arm: str) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod
    from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod

    try:
        payload = dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise TemporalArmError(f"baseline frontier is unreadable: {exc}") from exc
    try:
        canonical = cross_mod.validate_canonical(payload)
    except cross_mod.CrossFrontierError as exc:
        raise TemporalArmError(f"baseline frontier is not the canonical artifact: {exc}") from exc
    if canonical["arm"] != str(expected_arm):
        raise TemporalArmError(
            f"baseline frontier arm is {canonical['arm']!r}, required {str(expected_arm)!r}"
        )
    parsed: dict[str, Any] = {}
    for horizon_ms in frontier_mod.HORIZONS_MS:
        horizon = str(horizon_ms)
        parsed[horizon] = {}
        for name in ("macro", "ami", "alimeeting", "pooled"):
            parsed[horizon][name] = {}
            for kind in ("calibrated", "raw"):
                block = canonical["horizons"][horizon][name][kind]
                parsed[horizon][name][kind] = {
                    "points": [
                        _frontier_point_from(p, f"{name}/{kind} point at horizon {horizon}")
                        for p in block["points"]
                    ],
                    "reference": _frontier_point_from(
                        block["reference"], f"{name}/{kind} reference at horizon {horizon}"
                    ),
                }
    return {"horizons": parsed, "arm": canonical["arm"]}


def _point_dicts(points: Sequence[Any]) -> list[dict[str, float]]:
    out = []
    for point in points:
        if isinstance(point, dict):
            out.append(
                {
                    "threshold": float(point["threshold"]),
                    "false_cuts_per_hour": float(point["false_cuts_per_hour"]),
                    "contamination": float(point["contamination"]),
                    "miss_rate": float(point["miss_rate"]),
                }
            )
        else:
            out.append(
                {
                    "threshold": float(point.threshold),
                    "false_cuts_per_hour": float(point.false_cuts_per_hour),
                    "contamination": float(point.contamination),
                    "miss_rate": float(point.miss_rate),
                }
            )
    return out


def _compare_with_baseline_depth(
    arm_points: Sequence[Any],
    baseline_budget: Any,
    baseline_points: Sequence[Any],
    arm: str,
    baseline: str,
    corpora: Mapping[str, Any],
) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

    if isinstance(baseline_budget, dict):
        budget_ref: Any = dict(baseline_budget)
    else:
        budget_ref = {
            "threshold": float(baseline_budget.threshold),
            "false_cuts_per_hour": float(baseline_budget.false_cuts_per_hour),
            "contamination": float(baseline_budget.contamination),
            "miss_rate": float(baseline_budget.miss_rate),
        }
    try:
        compared = cross_mod.compare_candidate_to_baseline(
            _point_dicts(list(arm_points)),
            _point_dicts(list(baseline_points)),
            budget_ref,
            arm,
            baseline,
            dict(corpora),
        )
    except cross_mod.CrossFrontierError as exc:
        raise TemporalArmError(f"comparison budget is not the raw-F0@0.5 point: {exc}") from exc
    return compared


def compare_t2_to_h_f0(
    t2_points: Sequence[Any],
    baseline_budget: Any,
    corpora: Mapping[str, Any],
    baseline_points: Sequence[Any],
) -> dict[str, Any]:
    return _compare_with_baseline_depth(
        t2_points, baseline_budget, baseline_points, "R-T2-SC", "R-H-SC+F0", corpora
    )


def compare_ta_to_t2_f0(
    ta_points: Sequence[Any],
    baseline_budget: Any,
    corpora: Mapping[str, Any],
    baseline_points: Sequence[Any],
) -> dict[str, Any]:
    return _compare_with_baseline_depth(
        ta_points, baseline_budget, baseline_points, "R-TA-SC", "R-T2-SC+F0", corpora
    )
