from __future__ import annotations

import hashlib
import io
import json
import os
import random
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import load_sessions
from experiments.psem_sortformer_adaptation_depth.authority_registry import (
    register_execution,
)
from experiments.psem_sortformer_adaptation_depth.frame_alignment import (
    action_sample_indices,
    mapping_from_action_probabilities,
    native_episode_timeline,
    native_frame_coordinates,
)
from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
    TrainableSortformerPSEM,
    load_pinned_sortformer,
)
from experiments.psem_sortformer_adaptation_depth.parameter_policy import apply_parameter_policy
from experiments.psem_sortformer_adaptation_depth.predictions import prediction_rows
from experiments.psem_sortformer_adaptation_depth.preflight import (
    REPOSITORY_ROOT,
    SOURCE_MANIFEST_PATH,
    canonical_sha256,
    require_material_execution_ready,
)
from experiments.psem_sortformer_adaptation_depth.protocol import (
    _eval_registry_marker,
    require_bound,
    validate_eval_authorization,
)
from experiments.psem_sortformer_adaptation_depth.receipts import (
    revalidate_material_training_gate,
)
from experiments.psem_sortformer_adaptation_depth.runtime_audit import (
    build_timing_receipt,
    run_gradient_update_canary,
    run_prefix_causality_audit,
)
from experiments.psem_sortformer_adaptation_depth.sampling import (
    load_sampling_rows,
    load_training_sessions,
    select_overfit_rows,
    validate_sampling_manifest,
)
from experiments.psem_sortformer_adaptation_depth.training import (
    OPTIMIZER_STEPS_PER_EPOCH,
    ClassWeights,
    _legacy_authorize_overfit_arm,
    _legacy_run_overfit_arm,
    authorize_official_training,
    evaluate_examples,
    fit_arm,
    prepare_dev_example,
    prepare_training_example,
    run_short_smoke,
)
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    normalize_reference_session,
    open_reference_checkout,
)
from experiments.psem_training_strategy_gate.preflight import canonical_sha256 as label_sha256
from experiments.psem_training_strategy_gate.sampling import (
    DEV_ROLE,
    EVAL_ROLE,
    TOPOLOGY_MANIFEST_PATH,
    TRAIN_ROLE,
    RuntimeSession,
    load_runtime_sessions,
    role_by_source,
)

FRAME_SAMPLES = 1280
SAMPLE_RATE_HZ = 16000
CHECKPOINT_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
PACKAGE_ROOT = Path(__file__).resolve().parent


class ExecutionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )
            count += 1
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": count,
    }


def seed_runtime(seed: int) -> None:
    if seed != 7301:
        raise ExecutionError("runtime seed differs from the frozen recipe")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _source_rows() -> dict[str, dict[str, Any]]:
    return {
        row["source_id"]: row
        for row in (
            json.loads(line)
            for line in SOURCE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        )
    }


def _load_eval_sessions(
    corpus_root: Path,
    reference_root: Path,
    authorization: Mapping[str, Any],
) -> dict[str, RuntimeSession]:
    validate_eval_authorization(authorization)
    auth = require_bound(authorization, "psem_sortformer_eval_open_authorization")
    if (
        auth.get("evaluation_roles") != [EVAL_ROLE]
        or auth.get("eval_open_count") != 1
        or auth.get("eval_used_for_development") is not False
    ):
        raise ExecutionError("EVAL access lacks the one-time candidate-freeze authorization")
    source_rows = _source_rows()
    topology_rows = {
        row["source_id"]: row
        for row in (
            json.loads(line)
            for line in TOPOLOGY_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        )
    }
    assignments = role_by_source()
    checkout = open_reference_checkout(reference_root)
    sessions = {}
    for source_id in sorted(source_rows):
        if assignments[source_id] != EVAL_ROLE:
            continue
        row = source_rows[source_id]
        normalized = normalize_reference_session(row, corpus_root, checkout)
        if topology_rows[source_id]["label_result_sha256"] != label_sha256(
            normalized.labels.to_dict()
        ):
            raise ExecutionError("EVAL labels differ from the frozen topology artifact")
        sessions[source_id] = RuntimeSession(
            source_id=source_id,
            role=EVAL_ROLE,
            audio_ref=str(row["audio_ref"]),
            waveform_sha256=str(row["waveform_sha256"]),
            labels=normalized.labels,
        )
    if not sessions:
        raise ExecutionError("the authorized EVAL split is empty")
    return sessions


def load_scoring_sessions(
    corpus_root: Path,
    reference_root: Path,
    role: str,
    *,
    eval_authorization: Mapping[str, Any] | None = None,
) -> dict[str, RuntimeSession]:
    if role == DEV_ROLE:
        if eval_authorization is not None:
            raise ExecutionError("DEV loading must not receive EVAL authorization")
        return load_runtime_sessions(corpus_root, reference_root, roles=(DEV_ROLE,))
    if role == EVAL_ROLE and eval_authorization is not None:
        return _load_eval_sessions(corpus_root, reference_root, eval_authorization)
    raise ExecutionError("scoring role is unauthorized or EVAL remains sealed")


def load_source_waveform(
    session: RuntimeSession,
    corpus_root: Path,
) -> tuple[torch.Tensor, int, int]:
    source_row = _source_rows().get(session.source_id)
    relative = Path(session.audio_ref)
    root = corpus_root.resolve()
    path = (root / relative).resolve()
    if (
        source_row is None
        or relative.is_absolute()
        or ".." in relative.parts
        or not path.is_relative_to(root)
        or not path.is_file()
        or sha256_file(path) != session.waveform_sha256
    ):
        raise ExecutionError(f"source waveform identity differs: {session.source_id}")
    waveform, sample_rate = torchaudio.load(path)
    duration = int(source_row["duration_samples"])
    if sample_rate != SAMPLE_RATE_HZ or waveform.shape != (1, duration):
        raise ExecutionError(f"source waveform geometry differs: {session.source_id}")
    complete_samples = duration - duration % FRAME_SAMPLES
    return waveform[:, :complete_samples], duration, duration - complete_samples


def _mapping_slots(
    activity_logits: torch.Tensor,
    session_snapshot: Any,
) -> tuple[list[str | None], list[int], list[dict[str, Any]]]:
    if activity_logits.ndim != 2 or activity_logits.shape[1] != 4 or activity_logits.shape[0] <= 0:
        raise ExecutionError("runtime native posterior geometry is invalid")
    episode_ids = list(
        native_episode_timeline(session_snapshot.reference, int(activity_logits.shape[0]))
    )
    _, native_ends = native_frame_coordinates(len(episode_ids))
    indices = action_sample_indices(native_ends, session_snapshot.ends)
    logits = activity_logits.detach().cpu().numpy().astype(np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    action_probabilities = probabilities[indices].astype(np.float32)
    action_alive = np.ones_like(action_probabilities, dtype=np.bool_)
    slots_by_episode, mapping_rows = mapping_from_action_probabilities(
        session_snapshot, action_probabilities, action_alive
    )
    slots = [slots_by_episode.get(episode_id, 0) for episode_id in episode_ids]
    return episode_ids, slots, mapping_rows


def _checkpoint_candidate(
    authorization: Mapping[str, Any], arm: str, seed: int | None
) -> Mapping[str, Any]:
    validate_eval_authorization(authorization)
    auth = require_bound(authorization, "psem_sortformer_eval_open_authorization")
    candidates = [
        row for row in auth["candidate_set"] if row.get("arm") == arm and row.get("seed") == seed
    ]
    if len(candidates) != 1:
        raise ExecutionError("requested EVAL candidate is outside the frozen set")
    return candidates[0]


def _require_frozen_checkpoint_receipt(
    candidate: Mapping[str, Any], checkpoint_receipt: Mapping[str, Any]
) -> None:
    if candidate.get("checkpoint_receipt_sha256") != checkpoint_receipt.get("payload_sha256"):
        raise ExecutionError("EVAL checkpoint receipt differs from the frozen candidate")


def load_checkpoint_state(
    model: TrainableSortformerPSEM,
    checkpoint_path: Path,
    checkpoint_receipt: Mapping[str, Any],
    arm: str,
    seed: int,
    runtime_identity: Mapping[str, Any],
) -> None:
    receipt_payload = {
        key: value for key, value in checkpoint_receipt.items() if key != "payload_sha256"
    }
    code_identity = candidate_code_identity()
    checkpoint_bytes = checkpoint_path.read_bytes() if checkpoint_path.is_file() else None
    if (
        checkpoint_receipt.get("artifact_role") != "psem_sortformer_checkpoint"
        or checkpoint_receipt.get("payload_sha256") != canonical_sha256(receipt_payload)
        or checkpoint_receipt.get("arm") != arm
        or checkpoint_receipt.get("seed") != seed
        or checkpoint_receipt.get("checkpoint_path") != str(checkpoint_path.resolve())
        or checkpoint_bytes is None
        or checkpoint_receipt.get("checkpoint_sha256")
        != hashlib.sha256(checkpoint_bytes).hexdigest()
        or checkpoint_receipt.get("checkpoint_size_bytes") != len(checkpoint_bytes)
        or checkpoint_receipt.get("runtime_identity") != runtime_identity
        or checkpoint_receipt.get("runtime_identity_sha256") != canonical_sha256(runtime_identity)
        or checkpoint_receipt.get("candidate_code_identity_sha256")
        != code_identity["payload_sha256"]
        or not isinstance(checkpoint_receipt.get("material_training_authorization"), Mapping)
        or checkpoint_receipt["material_training_authorization"].get("payload_sha256")
        != checkpoint_receipt.get("material_gate_sha256")
    ):
        raise ExecutionError("checkpoint file and receipt identity differ")
    payload = torch.load(
        io.BytesIO(checkpoint_bytes), map_location=model.sortformer.device, weights_only=True
    )
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "arm",
        "seed",
        "model_state_dict",
    }:
        raise ExecutionError("checkpoint payload schema is invalid")
    if payload["schema_version"] != 1 or payload["arm"] != arm or payload["seed"] != seed:
        raise ExecutionError("checkpoint payload arm or seed differs")
    model.load_state_dict(payload["model_state_dict"], strict=True)


@torch.no_grad()
def infer_prediction_set(
    *,
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    corpus_root: Path,
    reference_root: Path,
    output_root: Path,
    protocol_registry_root: Path,
    device: str,
    role: str,
    arm: str,
    seed: int | None,
    trained_checkpoint_path: Path | None = None,
    trained_checkpoint_receipt: Mapping[str, Any] | None = None,
    eval_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    code_identity = candidate_code_identity()
    registry_root = protocol_registry_root.resolve()
    if (
        not registry_root.is_absolute()
        or not registry_root.is_dir()
        or registry_root == REPOSITORY_ROOT
        or registry_root.is_relative_to(REPOSITORY_ROOT)
        or registry_root == output_root.resolve()
    ):
        raise ExecutionError("protocol registry root is not a distinct external directory")
    if (arm == "F0-FROZEN-FLOAT" and seed is not None) or (
        arm != "F0-FROZEN-FLOAT" and seed != 7301
    ):
        raise ExecutionError("prediction arm/seed identity differs from the protocol")
    if role == DEV_ROLE and _eval_registry_marker().exists():
        raise ExecutionError("DEV inference is sealed after the canonical EVAL open")
    if role == EVAL_ROLE:
        if eval_authorization is None:
            raise ExecutionError("EVAL inference lacks candidate-freeze authorization")
        candidate = _checkpoint_candidate(eval_authorization, arm, seed)
        authorization_payload = require_bound(
            eval_authorization, "psem_sortformer_eval_open_authorization"
        )
        if authorization_payload.get("experiment_output_root") != str(output_root.resolve()):
            raise ExecutionError("EVAL output root differs from the single-open authorization")
        if authorization_payload.get("protocol_registry_root") != str(registry_root):
            raise ExecutionError(
                "EVAL protocol registry differs from the single-open authorization"
            )
        validate_current_candidate_identity(
            {
                "schema_version": 1,
                "artifact_role": "psem_sortformer_candidate_code_identity",
                "git_head": authorization_payload["candidate_git_head"],
                "worktree_clean": True,
                "artifact_sha256s": authorization_payload["candidate_artifact_sha256s"],
                "payload_sha256": authorization_payload["candidate_code_identity_sha256"],
            }
        )
        if arm != "F0-FROZEN-FLOAT" and (
            trained_checkpoint_receipt is None
            or candidate.get("checkpoint_sha256")
            != trained_checkpoint_receipt.get("checkpoint_sha256")
        ):
            raise ExecutionError("EVAL checkpoint differs from the frozen candidate")
        if arm != "F0-FROZEN-FLOAT":
            _require_frozen_checkpoint_receipt(candidate, trained_checkpoint_receipt)
    sessions = load_scoring_sessions(
        corpus_root,
        reference_root,
        role,
        eval_authorization=eval_authorization,
    )
    model, runtime_identity = load_pinned_sortformer(
        checkpoint_path,
        nemo_checkout,
        dependency_lock,
        device,
    )
    if arm != "F0-FROZEN-FLOAT":
        if trained_checkpoint_path is None or trained_checkpoint_receipt is None:
            raise ExecutionError("trainable prediction lacks its selected checkpoint")
        material_gate = trained_checkpoint_receipt.get("material_training_authorization")
        bundle = (
            material_gate.get("validation_bundle") if isinstance(material_gate, Mapping) else None
        )
        manifest_path = (
            Path(str(bundle.get("sampling_manifest_path"))) if isinstance(bundle, Mapping) else None
        )
        if manifest_path is None:
            raise ExecutionError("trained checkpoint lacks its material validation bundle")
        if material_gate.get("authorized_protocol_registry_root") != str(registry_root):
            raise ExecutionError("trained checkpoint uses another protocol registry")
        if (
            material_gate.get("authorized_checkpoint_path") != str(checkpoint_path.resolve())
            or material_gate.get("authorized_corpus_root") != str(corpus_root.resolve())
            or material_gate.get("authorized_reference_root") != str(reference_root.resolve())
            or material_gate.get("authorized_output_root") != str(output_root.resolve())
        ):
            raise ExecutionError("EVAL runtime paths differ from the frozen material gate")
        load_checkpoint_state(
            model,
            trained_checkpoint_path,
            trained_checkpoint_receipt,
            arm,
            int(seed),
            runtime_identity,
        )
    if arm == "F0-FROZEN-FLOAT":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        parameter_policy = {
            "arm": arm,
            "all_parameters_frozen": True,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": 0,
        }
    else:
        parameter_policy = apply_parameter_policy(model, arm)
        if trained_checkpoint_receipt.get("parameter_policy") != parameter_policy:
            raise ExecutionError("checkpoint parameter policy differs from the runtime graph")
    model.eval()
    snapshots = {
        session.source_id: session
        for session in load_sessions(validate_mapping_ledger=False)
        if session.manifest["old_v2_role"] == role
    }
    if set(snapshots) != set(sessions):
        raise ExecutionError("scoring source identities differ from the #99 evaluator snapshot")
    run_root = (output_root.resolve() / "predictions" / role / arm / str(seed or "f0")).resolve()
    if not run_root.is_relative_to(output_root.resolve()):
        raise ExecutionError("prediction output path escapes the external experiment root")
    descriptors = []
    mapping_receipts = []
    for source_id in sorted(sessions):
        waveform, duration, tail = load_source_waveform(sessions[source_id], corpus_root)
        waveform = waveform.to(model.sortformer.device)
        lengths = torch.tensor([waveform.shape[1]], dtype=torch.long, device=waveform.device)
        reset = torch.zeros(
            (1, waveform.shape[1] // FRAME_SAMPLES, 1),
            dtype=torch.bool,
            device=waveform.device,
        )
        reset[:, 0, 0] = True
        evidence = model.sortformer_evidence(waveform, lengths, state_reset=reset)
        episode_ids, slots, mapping = _mapping_slots(
            evidence.activity_logits[0], snapshots[source_id]
        )
        anchor = torch.zeros_like(evidence.probabilities)
        anchor[
            0,
            torch.arange(len(slots), device=anchor.device),
            torch.tensor(slots, device=anchor.device),
        ] = 1
        if arm == "F0-FROZEN-FLOAT":
            selected = (evidence.probabilities * anchor).sum(dim=-1).clamp(1e-7, 1 - 1e-7)
            outputs = {
                "anchor_present": torch.logit(selected),
                "replacement_evidence": torch.logit(1 - selected),
            }
        else:
            outputs = model.psem_outputs(evidence, anchor)
        rows = prediction_rows(
            source_id=source_id,
            source_start_sample=0,
            evidence=evidence,
            psem_outputs=outputs,
            anchor_episode_ids=episode_ids,
            oracle_anchor_slots=slots,
            provenance={
                "split_role": role,
                "arm": arm,
                "seed": seed,
                "source_waveform_sha256": sessions[source_id].waveform_sha256,
                "base_checkpoint_sha256": CHECKPOINT_SHA256,
                "trained_checkpoint_sha256": (
                    trained_checkpoint_receipt.get("checkpoint_sha256")
                    if trained_checkpoint_receipt is not None
                    else None
                ),
                "trained_checkpoint_receipt_sha256": (
                    trained_checkpoint_receipt.get("payload_sha256")
                    if trained_checkpoint_receipt is not None
                    else None
                ),
                "runtime_identity_sha256": canonical_sha256(runtime_identity),
                "parameter_policy_sha256": canonical_sha256(parameter_policy),
                "candidate_git_head": code_identity["git_head"],
                "candidate_code_identity_sha256": code_identity["payload_sha256"],
                "experiment_output_root": str(output_root.resolve()),
            },
        )
        descriptor = write_jsonl(run_root / f"{source_id}.jsonl", rows)
        descriptors.append({"source_id": source_id, **descriptor})
        mapping_receipts.append(
            {
                "source_id": source_id,
                "source_waveform_sha256": sessions[source_id].waveform_sha256,
                "source_duration_samples": duration,
                "source_tail_samples_excluded": tail,
                "frame_count": len(rows),
                "mappings": mapping,
                "mapped_episode_count": sum(row["status"] == "mapped" for row in mapping),
                "episode_count": len(mapping),
            }
        )
        del waveform, evidence, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_prediction_set",
        "arm": arm,
        "seed": seed,
        "split_role": role,
        "sources": descriptors,
        "mapping_receipts": mapping_receipts,
        "native_frame_samples": FRAME_SAMPLES,
        "algorithmic_evidence_delay_samples": 16640,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "runtime_identity": runtime_identity,
        "parameter_policy_sha256": canonical_sha256(parameter_policy),
        "parameter_policy": parameter_policy,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": (
            0
            if arm == "F0-FROZEN-FLOAT"
            else int(trained_checkpoint_receipt["training_summary"]["trainable_parameters"])
        ),
        "trained_checkpoint_sha256": (
            trained_checkpoint_receipt.get("checkpoint_sha256")
            if trained_checkpoint_receipt is not None
            else None
        ),
        "trained_checkpoint_receipt_sha256": (
            trained_checkpoint_receipt.get("payload_sha256")
            if trained_checkpoint_receipt is not None
            else None
        ),
        "eval_authorization_sha256": (
            eval_authorization.get("payload_sha256") if eval_authorization is not None else None
        ),
        "experiment_output_root": str(output_root.resolve()),
        "protocol_registry_root": str(registry_root),
        "candidate_git_head": code_identity["git_head"],
        "candidate_code_identity_sha256": code_identity["payload_sha256"],
        "candidate_artifact_sha256s": code_identity["artifact_sha256s"],
    }
    bound = {**payload, "payload_sha256": canonical_sha256(payload)}
    return bound


def assert_clean_candidate() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if dirty:
        raise ExecutionError("material execution requires a clean committed candidate")
    return head


def candidate_code_identity() -> dict[str, Any]:
    head = assert_clean_candidate()
    paths = tuple(
        sorted(
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in PACKAGE_ROOT.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".json"}
            and "__pycache__" not in path.parts
            and "results" not in path.parts
        )
    )
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_candidate_code_identity",
        "git_head": head,
        "worktree_clean": True,
        "artifact_sha256s": {path: sha256_file(PACKAGE_ROOT / path) for path in paths},
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def candidate_git_head_is_resume_compatible(
    observed_head: object, current_head: str | None = None
) -> bool:
    current = current_head or assert_clean_candidate()
    observed = str(observed_head)
    if observed == current:
        return True
    if observed != "1334720ab9975b9a68aedf2d291eb9056baf3ff3":
        return False
    descendant = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            "f76efc468092b0bb8b9969ffb3342c1ec781af58",
            current,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if descendant.returncode != 0:
        return False
    changed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            observed,
            current,
            "--",
            str(PACKAGE_ROOT.relative_to(REPOSITORY_ROOT)),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return set(changed) == {
        "experiments/psem_sortformer_adaptation_depth/RUNPOD_DEBUG_JOURNAL.md",
        "experiments/psem_sortformer_adaptation_depth/execution.py",
        "experiments/psem_sortformer_adaptation_depth/receipts.py",
        "experiments/psem_sortformer_adaptation_depth/tests/test_training.py",
        "experiments/psem_sortformer_adaptation_depth/training.py",
    }


def validate_current_candidate_identity(identity: Mapping[str, Any]) -> None:
    current = candidate_code_identity()
    if identity == current:
        return
    payload = {key: value for key, value in identity.items() if key != "payload_sha256"}
    observed_artifacts = identity.get("artifact_sha256s")
    current_artifacts = current["artifact_sha256s"]
    compatible_paths = {
        "execution.py",
        "receipts.py",
        "training.py",
        "tests/test_training.py",
    }
    shared_paths_match = (
        isinstance(observed_artifacts, Mapping)
        and set(observed_artifacts) == set(current_artifacts)
        and all(
            observed_artifacts[path] == current_artifacts[path]
            for path in current_artifacts
            if path not in compatible_paths
        )
    )
    if (
        identity.get("schema_version") != 1
        or identity.get("artifact_role") != "psem_sortformer_candidate_code_identity"
        or identity.get("worktree_clean") is not True
        or identity.get("payload_sha256") != canonical_sha256(payload)
        or not candidate_git_head_is_resume_compatible(
            identity.get("git_head"), str(current["git_head"])
        )
        or not shared_paths_match
    ):
        raise ExecutionError("current code differs from the frozen candidate identity")


def _corpus_by_source() -> dict[str, str]:
    return {source_id: str(row["corpus"]) for source_id, row in _source_rows().items()}


def _training_batches(
    *,
    epoch: int,
    rows: Sequence[Mapping[str, Any]],
    sessions: Mapping[str, RuntimeSession],
    corpus_root: Path,
    manifest_path: Path,
    manifest_validation: Mapping[str, Any],
) -> Iterable[tuple[Any, ...]]:
    by_id = {str(row["row_id"]): row for row in rows}
    corpus_by_source = _corpus_by_source()
    for row in rows:
        if row.get("epoch") != epoch:
            continue
        source_id = str(row["source_id"])
        example = prepare_training_example(
            row,
            sessions[source_id],
            corpus_root,
            corpus_by_source[source_id],
            manifest_path=manifest_path,
            manifest_validation=manifest_validation,
            manifest_rows_by_id=by_id,
        )
        yield (example,)


def _dev_batches(
    *,
    sessions: Mapping[str, RuntimeSession],
    corpus_root: Path,
) -> Iterable[tuple[Any, ...]]:
    source_rows = _source_rows()
    root = corpus_root.resolve()
    for source_id in sorted(sessions):
        session = sessions[source_id]
        relative = Path(session.audio_ref)
        path = (root / relative).resolve()
        duration = int(source_rows[source_id]["duration_samples"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root)
            or not path.is_file()
            or sha256_file(path) != session.waveform_sha256
        ):
            raise ExecutionError("DEV waveform identity differs from the frozen source")
        complete_windows = min(duration, int(session.labels.intervals[-1].end_sample)) // 480000
        if complete_windows <= 0:
            raise ExecutionError("DEV source has no complete checkpoint-selection sequence")
        for window_index in range(complete_windows):
            start = window_index * 480000
            waveform, sample_rate = torchaudio.load(path, frame_offset=start, num_frames=480000)
            if sample_rate != SAMPLE_RATE_HZ or waveform.shape != (1, 480000):
                raise ExecutionError("DEV selection waveform geometry differs")
            yield (
                prepare_dev_example(
                    source_id=source_id,
                    corpus=str(source_rows[source_id]["corpus"]),
                    window_start_sample=start,
                    waveform=waveform[0],
                    labels=session.labels,
                ),
            )


def _save_model_checkpoint(
    model: TrainableSortformerPSEM,
    path: Path,
    arm: str,
    seed: int,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema_version": 1,
            "arm": arm,
            "seed": seed,
            "model_state_dict": model.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)
    return sha256_file(path)


def _legacy_run_training_arm(
    *,
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    corpus_root: Path,
    reference_root: Path,
    sampling_manifest: Path,
    class_weight_receipt: Mapping[str, Any],
    material_gate: Mapping[str, Any],
    output_root: Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_material_execution_ready()
    assert_clean_candidate()
    rows = load_sampling_rows(sampling_manifest)
    sessions = load_training_sessions(corpus_root, reference_root)
    manifest_validation = validate_sampling_manifest(sampling_manifest, sessions)
    revalidate_material_training_gate(
        material_gate,
        sampling_manifest_path=sampling_manifest,
        sampling_rows=rows,
        training_sessions=sessions,
        class_weight_receipt=class_weight_receipt,
        checkpoint_path=checkpoint_path,
        corpus_root=corpus_root,
        reference_root=reference_root,
        output_root=output_root,
    )
    registry_value = material_gate.get("authorized_protocol_registry_root")
    registry_root = Path(registry_value) if isinstance(registry_value, str) else None
    if registry_root is None or _eval_registry_marker().exists():
        raise ExecutionError("material training cannot resume after EVAL opened")
    authorization = authorize_official_training(material_gate, rows, class_weight_receipt)
    arm = authorization.arm
    seed = authorization.seed
    training_device = torch.device(device)
    if not torch.cuda.is_available() or training_device.type != "cuda":
        raise ExecutionError("material training requires CUDA memory accounting")
    seed_runtime(seed)
    model, runtime_identity = load_pinned_sortformer(
        checkpoint_path,
        nemo_checkout,
        dependency_lock,
        device,
    )
    parameter_policy = apply_parameter_policy(model, arm)
    dev_sessions = load_scoring_sessions(corpus_root, reference_root, DEV_ROLE)
    expected_dev_sha = canonical_sha256(sorted(dev_sessions))
    if expected_dev_sha != authorization.dev_source_ids_sha256:
        raise ExecutionError("DEV checkpoint-selection identities differ from authorization")
    checkpoint_path_out = (
        output_root.resolve() / "checkpoints" / arm / str(seed) / "selected.pt"
    ).resolve()
    if not checkpoint_path_out.is_relative_to(output_root.resolve()):
        raise ExecutionError("checkpoint output path escapes the external experiment root")
    checkpoint_state: dict[str, Any] = {}

    def checkpoint_callback(
        current_model: TrainableSortformerPSEM,
        epoch: int,
        metrics: Mapping[str, float],
    ) -> None:
        digest = _save_model_checkpoint(current_model, checkpoint_path_out, arm, seed)
        checkpoint_state.clear()
        checkpoint_state.update(
            {"epoch": epoch, "metrics": dict(metrics), "checkpoint_sha256": digest}
        )

    def dev_callback(
        current_model: TrainableSortformerPSEM,
        _epoch: int,
    ) -> Mapping[str, float | str]:
        metrics = evaluate_examples(
            current_model,
            _dev_batches(sessions=dev_sessions, corpus_root=corpus_root),
            authorization.class_weights,
        )
        return {
            "dev_total_loss": metrics["total_loss"],
            "dev_replacement_average_precision": metrics["replacement_average_precision"],
            "split_role": DEV_ROLE,
            "source_ids_sha256": expected_dev_sha,
        }

    torch.cuda.reset_peak_memory_stats(training_device)
    started = time.perf_counter()
    training_result = fit_arm(
        model,
        arm,
        authorization.class_weights,
        lambda epoch: _training_batches(
            epoch=epoch,
            rows=rows,
            sessions=sessions,
            corpus_root=corpus_root,
            manifest_path=sampling_manifest,
            manifest_validation=manifest_validation,
        ),
        OPTIMIZER_STEPS_PER_EPOCH,
        dev_callback,
        checkpoint_callback,
        authorization=authorization,
    )
    wall_clock = time.perf_counter() - started
    if (
        not checkpoint_state
        or checkpoint_state["epoch"] != training_result["selected_checkpoint"]["epoch"]
    ):
        raise ExecutionError("selected checkpoint file differs from DEV early stopping")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    peak_training_memory_bytes = int(torch.cuda.max_memory_allocated(training_device))
    if peak_training_memory_bytes <= 0:
        raise ExecutionError("material training did not produce positive CUDA memory evidence")
    training_summary = {
        "selected_epoch": checkpoint_state["epoch"],
        "selected_metrics": checkpoint_state["metrics"],
        "training_wall_clock_seconds": wall_clock,
        "peak_training_memory_bytes": peak_training_memory_bytes,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "optimizer": "AdamW",
        "maximum_epochs": 8,
        "early_stopping_patience_dev_evaluations": 2,
        "gradient_accumulation_steps": 16,
        "optimizer_steps_per_epoch": 256,
        "native_diarization_contract_passed": bool(
            material_gate.get("passed") is True
            and material_gate.get("overfit_receipt_sha256")
            and material_gate.get("gradient_receipt_sha256")
            and material_gate.get("timing_receipt_sha256")
        ),
        "native_diarization_contract_evidence_sha256": canonical_sha256(
            {
                "overfit_receipt_sha256": material_gate["overfit_receipt_sha256"],
                "gradient_receipt_sha256": material_gate["gradient_receipt_sha256"],
                "timing_receipt_sha256": material_gate["timing_receipt_sha256"],
            }
        ),
    }
    code_identity = candidate_code_identity()
    training_payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_training_result",
        **training_result,
        "seed": seed,
        "split_roles": ["PSEM-STRATEGY-TRAIN", DEV_ROLE],
        "eval_source_count": 0,
        "dev_sequence_policy": "complete_source_aligned_nonoverlapping_30_second_sequences",
        "dev_incomplete_tail_policy": "excluded_from_loss_but_included_in_product_evaluation",
        "training_wall_clock_seconds": wall_clock,
        "peak_training_memory_bytes": peak_training_memory_bytes,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "training_summary": training_summary,
        "checkpoint_path": str(checkpoint_path_out),
        "checkpoint_sha256": checkpoint_state["checkpoint_sha256"],
        "checkpoint_size_bytes": checkpoint_path_out.stat().st_size,
        "candidate_code_identity_sha256": code_identity["payload_sha256"],
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "runtime_identity": runtime_identity,
        "parameter_policy_sha256": canonical_sha256(parameter_policy),
        "parameter_policy": parameter_policy,
        "dev_source_ids_sha256": expected_dev_sha,
    }
    bound_training_result = {
        **training_payload,
        "payload_sha256": canonical_sha256(training_payload),
    }
    checkpoint_payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_checkpoint",
        "arm": arm,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path_out),
        "checkpoint_sha256": checkpoint_state["checkpoint_sha256"],
        "checkpoint_size_bytes": checkpoint_path_out.stat().st_size,
        "selected_epoch": checkpoint_state["epoch"],
        "selected_metrics": checkpoint_state["metrics"],
        "material_gate_sha256": material_gate["payload_sha256"],
        "material_training_authorization": dict(material_gate),
        "authorized_output_root": str(output_root.resolve()),
        "candidate_code_identity_sha256": code_identity["payload_sha256"],
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "runtime_identity": runtime_identity,
        "parameter_policy_sha256": canonical_sha256(parameter_policy),
        "parameter_policy": parameter_policy,
        "split_roles": ["PSEM-STRATEGY-TRAIN", DEV_ROLE],
        "eval_source_count": 0,
        "dev_source_ids_sha256": expected_dev_sha,
        "training_summary": training_summary,
        "training_result_sha256": bound_training_result["payload_sha256"],
        "training_result": bound_training_result,
    }
    checkpoint_receipt = {
        **checkpoint_payload,
        "payload_sha256": canonical_sha256(checkpoint_payload),
    }
    register_execution("training-result", bound_training_result)
    register_execution("checkpoint-receipt", checkpoint_receipt)
    return (
        bound_training_result,
        checkpoint_receipt,
    )


def run_canary_arm(
    *,
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    corpus_root: Path,
    reference_root: Path,
    sampling_manifest: Path,
    arm: str,
    device: str,
    ta_open_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require_material_execution_ready()
    if _eval_registry_marker().exists():
        raise ExecutionError("runtime canaries are sealed after EVAL opened")
    if arm not in {"H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"}:
        raise ExecutionError("runtime canary arm is invalid")
    conditional_authorization = None
    if arm == "TA-ALL-TEMPORAL":
        if not isinstance(ta_open_authorization, Mapping):
            raise ExecutionError("TA canary requires an explicit open_ta authorization")
        require_bound(ta_open_authorization, "psem_sortformer_open_ta_authorization")
        if ta_open_authorization.get("arm") != arm or ta_open_authorization.get("seed") != 7301:
            raise ExecutionError("TA canary authorization identity is invalid")
        conditional_authorization = dict(ta_open_authorization)
    elif ta_open_authorization is not None:
        raise ExecutionError("non-TA canary cannot carry a TA authorization")
    seed_runtime(7301)
    rows = load_sampling_rows(sampling_manifest)
    sessions = load_training_sessions(corpus_root, reference_root)
    validation = validate_sampling_manifest(sampling_manifest, sessions)
    first = rows[0]
    example = prepare_training_example(
        first,
        sessions[str(first["source_id"])],
        corpus_root,
        str(first["corpus"]),
        manifest_path=sampling_manifest,
        manifest_validation=validation,
        manifest_rows_by_id={str(row["row_id"]): row for row in rows},
    )
    model, runtime_identity = load_pinned_sortformer(
        checkpoint_path,
        nemo_checkout,
        dependency_lock,
        device,
    )
    apply_parameter_policy(model, arm)
    waveform = example.waveform.unsqueeze(0).to(model.sortformer.device)
    lengths = torch.tensor([480000], dtype=torch.long, device=waveform.device)
    reset = torch.zeros((1, 375, 1), dtype=torch.bool, device=waveform.device)
    reset[:, 0, 0] = True
    with torch.no_grad():
        evidence = model.sortformer_evidence(waveform, lengths, state_reset=reset)
        prefix_causality = run_prefix_causality_audit(model, waveform, lengths)
        timing = build_timing_receipt(
            lengths,
            evidence.probabilities,
            evidence.activity_logits,
            evidence.final_temporal_hidden,
            evidence.slot_alive,
            evidence.state_reset,
            evidence.evidence_delay_seconds,
            evidence.streaming_trace,
            prefix_causality,
        )
    canaries = run_gradient_update_canary(model, arm, waveform)
    result = {
        **canaries,
        "timing_receipt": timing,
        "runtime_identity": runtime_identity,
        "source_row_id": first["row_id"],
        "conditional_arm_audit_authorization": conditional_authorization,
    }
    return result


def _legacy_run_overfit_arm_implementation(
    *,
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    corpus_root: Path,
    reference_root: Path,
    sampling_manifest: Path,
    class_weight_receipt: Mapping[str, Any],
    arm: str,
    device: str,
    staged_execution_receipt: Mapping[str, Any] | None = None,
    staged_dev_results: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require_material_execution_ready()
    if _eval_registry_marker().exists():
        raise ExecutionError("overfit canaries are sealed after EVAL opened")
    from experiments.psem_sortformer_adaptation_depth.protocol import (
        _legacy_authorize_conditional_arm_audit,
    )

    conditional_authorization = _legacy_authorize_conditional_arm_audit(
        arm,
        staged_execution_receipt,
        staged_dev_results,
    )
    seed_runtime(7301)
    rows = load_sampling_rows(sampling_manifest)
    sessions = load_training_sessions(corpus_root, reference_root)
    validation = validate_sampling_manifest(sampling_manifest, sessions)
    corpus_by_source = _corpus_by_source()
    selected = list(select_overfit_rows(rows, corpus_by_source))
    authorization = _legacy_authorize_overfit_arm(
        arm,
        selected,
        rows,
        sampling_manifest,
        corpus_by_source,
        class_weight_receipt,
    )
    by_id = {str(row["row_id"]): row for row in rows}
    batches = [
        (
            prepare_training_example(
                row,
                sessions[str(row["source_id"])],
                corpus_root,
                str(row["corpus"]),
                manifest_path=sampling_manifest,
                manifest_validation=validation,
                manifest_rows_by_id=by_id,
            ),
        )
        for row in selected
    ]
    model, _ = load_pinned_sortformer(
        checkpoint_path,
        nemo_checkout,
        dependency_lock,
        device,
    )
    apply_parameter_policy(model, arm)
    raw_result = _legacy_run_overfit_arm(
        model,
        arm,
        batches,
        authorization.class_weights,
        authorization=authorization,
    )
    payload = {
        "schema_version": 1,
        "artifact_role": "overfit_arm_execution",
        **raw_result,
        "conditional_arm_audit_authorization": conditional_authorization,
    }
    result = {**payload, "payload_sha256": canonical_sha256(payload)}
    register_execution("overfit-arm", result)
    return result, selected


def run_memory_fit_preflight(
    *,
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    corpus_root: Path,
    reference_root: Path,
    sampling_manifest: Path,
    include_ta: bool,
    hourly_price_usd: float,
    hourly_price_source: str,
    required_inference_gpu_seconds: float,
    conditional_ta_inference_gpu_seconds: float,
    device: str,
) -> dict[str, Any]:
    import gc
    import platform

    from experiments.psem_sortformer_adaptation_depth.runtime_audit import (
        GRADIENT_CLIP_NORM,
        _build_memory_fit_optimizer,
    )
    from experiments.psem_sortformer_adaptation_depth.training import (
        GRADIENT_ACCUMULATION_STEPS,
        MICRO_BATCH_SIZE,
        ClassWeights,
        forward_batch,
    )

    memory_fit_started = time.perf_counter()
    hourly_price_source = hourly_price_source.strip()
    if not np.isfinite(hourly_price_usd) or hourly_price_usd <= 0:
        raise ExecutionError("resource estimate hourly price must be finite and positive")
    if not hourly_price_source:
        raise ExecutionError("resource estimate hourly price source must be non-empty")
    if not np.isfinite(required_inference_gpu_seconds) or required_inference_gpu_seconds < 0:
        raise ExecutionError("required inference GPU seconds must be finite and non-negative")
    if (
        not np.isfinite(conditional_ta_inference_gpu_seconds)
        or conditional_ta_inference_gpu_seconds < 0
    ):
        raise ExecutionError("conditional TA inference GPU seconds must be finite and non-negative")
    if not include_ta and conditional_ta_inference_gpu_seconds != 0:
        raise ExecutionError("conditional TA inference seconds require --include-ta")

    runtime_contract = json.loads(
        (PACKAGE_ROOT / "runtime_contract.json").read_text(encoding="utf-8")
    )
    optimization_contract = runtime_contract["optimization_execution"]
    memory_fit_contract = runtime_contract["memory_fit"]
    short_smoke_steps = optimization_contract["short_smoke_maximum_optimizer_steps"]
    official_steps = optimization_contract["official_maximum_optimizer_steps"]
    optimizer_steps = memory_fit_contract["optimizer_steps_per_selected_arm"]
    default_arms = tuple(memory_fit_contract["default_arms"])
    conditional_arm = memory_fit_contract["conditional_arm"]
    selected_arms = (*default_arms, conditional_arm) if include_ta else default_arms
    if (
        not isinstance(short_smoke_steps, int)
        or isinstance(short_smoke_steps, bool)
        or short_smoke_steps <= 0
        or not isinstance(official_steps, int)
        or isinstance(official_steps, bool)
        or official_steps <= 0
        or optimizer_steps != 2
        or memory_fit_contract["warmup_optimizer_steps"] != 1
        or memory_fit_contract["timed_optimizer_step_index"] != 2
        or memory_fit_contract["probe_rows_consumed"] != GRADIENT_ACCUMULATION_STEPS
    ):
        raise ExecutionError("optional resource estimator contract is invalid")
    if (
        memory_fit_contract["optional"] is not True
        or memory_fit_contract["material_training_authorization"] is not False
        or memory_fit_contract["checkpoint_persistence"] is not False
        or memory_fit_contract["timing_statistic"] != "single_second_step"
        or default_arms != ("H-HEAD", "T2-TOP")
        or conditional_arm != "TA-ALL-TEMPORAL"
    ):
        raise ExecutionError("optional resource estimator topology is invalid")

    training_device = torch.device(device)
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or training_device.type != "cuda"
    ):
        raise ExecutionError("optional resource estimate requires exactly one CUDA accelerator")

    rows = load_sampling_rows(sampling_manifest)
    sessions = load_training_sessions(corpus_root, reference_root)
    probe_rows = [row for row in rows if row.get("epoch") == 1][:GRADIENT_ACCUMULATION_STEPS]
    if len(probe_rows) != GRADIENT_ACCUMULATION_STEPS or [
        row.get("epoch_index") for row in probe_rows
    ] != list(range(GRADIENT_ACCUMULATION_STEPS)):
        raise ExecutionError("resource estimate requires 16 ordered epoch-1 probe rows")
    row_ids = [row.get("row_id") for row in probe_rows]
    if any(not isinstance(row_id, str) for row_id in row_ids) or len(set(row_ids)) != len(
        probe_rows
    ):
        raise ExecutionError("resource estimate probe row identities are invalid")
    sampling_manifest_sha256 = sha256_file(sampling_manifest)
    probe_validation = {
        "artifact_role": "optional_resource_probe_validation",
        "passed": True,
        "manifest_sha256": sampling_manifest_sha256,
        "source_manifest_row_count": len(rows),
        "consumed_row_count": len(probe_rows),
        "epoch": 1,
        "epoch_indices": list(range(GRADIENT_ACCUMULATION_STEPS)),
        "material_training_authorization": False,
    }
    class_weights = ClassWeights(replacement_positive=1.0, anchor_positive=1.0)
    rows_by_id = {str(row["row_id"]): row for row in probe_rows}
    input_started = time.perf_counter()
    examples = [
        prepare_training_example(
            row,
            sessions[str(row["source_id"])],
            corpus_root,
            str(row["corpus"]),
            manifest_path=sampling_manifest,
            manifest_validation=probe_validation,
            manifest_rows_by_id=rows_by_id,
        )
        for row in probe_rows
    ]
    input_preparation_seconds = time.perf_counter() - input_started

    properties = torch.cuda.get_device_properties(training_device)
    total_memory_bytes = int(properties.total_memory)
    arm_results: list[dict[str, Any]] = []
    runtime_identity_sha256s: set[str] = set()
    for arm in selected_arms:
        model = None
        optimizer = None
        result = None
        loss = None
        gradient_norm = None
        optimizer_step_seconds_samples: list[float] = []
        arm_started = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        seed_runtime(7301)
        try:
            model, runtime_identity = load_pinned_sortformer(
                checkpoint_path,
                nemo_checkout,
                dependency_lock,
                training_device,
            )
            parameter_policy = apply_parameter_policy(model, arm)
            optimizer = _build_memory_fit_optimizer(model, arm)
            model.train()
            torch.cuda.synchronize(training_device)
            torch.cuda.reset_peak_memory_stats(training_device)
            for _ in range(optimizer_steps):
                optimizer.zero_grad(set_to_none=True)
                step_started = time.perf_counter()
                for example in examples:
                    result = forward_batch(model, (example,), class_weights)
                    loss = result.losses["total"]
                    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                        raise ExecutionError("resource estimate forward produced a non-finite loss")
                    (loss / GRADIENT_ACCUMULATION_STEPS).backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    GRADIENT_CLIP_NORM,
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise ExecutionError(
                        "resource estimate backward produced a non-finite gradient"
                    )
                optimizer.step()
                torch.cuda.synchronize(training_device)
                optimizer_step_seconds_samples.append(time.perf_counter() - step_started)
            optimizer_step_seconds = optimizer_step_seconds_samples[1]
            peak_allocated_bytes = int(torch.cuda.max_memory_allocated(training_device))
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved(training_device))
            if peak_allocated_bytes <= 0 or peak_reserved_bytes < peak_allocated_bytes:
                raise ExecutionError("resource estimate CUDA accounting is invalid")
            runtime_identity_sha256 = canonical_sha256(runtime_identity)
            runtime_identity_sha256s.add(runtime_identity_sha256)
            arm_results.append(
                {
                    "arm": arm,
                    "fit": True,
                    "peak_allocated_memory_bytes": peak_allocated_bytes,
                    "peak_reserved_memory_bytes": peak_reserved_bytes,
                    "device_memory_headroom_bytes": total_memory_bytes - peak_reserved_bytes,
                    "device_memory_headroom_fraction": (
                        (total_memory_bytes - peak_reserved_bytes) / total_memory_bytes
                    ),
                    "warmup_optimizer_step_seconds": optimizer_step_seconds_samples[0],
                    "optimizer_step_seconds": optimizer_step_seconds,
                    "timed_optimizer_step_index": 2,
                    "timing_statistic": "single_second_step",
                    "micro_batch_seconds": optimizer_step_seconds / GRADIENT_ACCUMULATION_STEPS,
                    "arm_wall_clock_seconds": time.perf_counter() - arm_started,
                    "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    "parameter_policy_sha256": canonical_sha256(parameter_policy),
                    "runtime_identity_sha256": runtime_identity_sha256,
                    "optimizer_state_initialized": True,
                    "optimizer_steps_executed": optimizer_steps,
                    "forward_backward_micro_batches_per_step": GRADIENT_ACCUMULATION_STEPS,
                    "checkpoint_persisted": False,
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            arm_results.append(
                {
                    "arm": arm,
                    "fit": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "peak_allocated_memory_bytes": int(
                        torch.cuda.max_memory_allocated(training_device)
                    ),
                    "peak_reserved_memory_bytes": int(
                        torch.cuda.max_memory_reserved(training_device)
                    ),
                    "completed_optimizer_steps": len(optimizer_step_seconds_samples),
                    "required_optimizer_steps": optimizer_steps,
                    "arm_wall_clock_seconds": time.perf_counter() - arm_started,
                    "checkpoint_persisted": False,
                }
            )
        finally:
            gradient_norm = None
            loss = None
            optimizer = None
            result = None
            model = None
            gc.collect()
            torch.cuda.empty_cache()

    successful_steps = [
        float(row["optimizer_step_seconds"]) for row in arm_results if row["fit"] is True
    ]
    input_to_step_ratio = (
        input_preparation_seconds / min(successful_steps) if successful_steps else None
    )
    try:
        host_pages = int(os.sysconf("SC_PHYS_PAGES"))
        host_page_size = int(os.sysconf("SC_PAGE_SIZE"))
        host_memory_bytes = host_pages * host_page_size
    except (AttributeError, OSError, ValueError):
        host_memory_bytes = None
    try:
        storage = subprocess.run(
            [
                "findmnt",
                "--noheadings",
                "--output",
                "FSTYPE,SOURCE,TARGET",
                "--target",
                str(corpus_root.resolve()),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        storage = "unavailable"

    estimator_wall_clock_seconds = time.perf_counter() - memory_fit_started
    usd_per_second = hourly_price_usd / 3600.0
    successful_by_arm = {str(row["arm"]): row for row in arm_results if row["fit"] is True}
    if all(arm in successful_by_arm for arm in default_arms):
        optimizer_steps_per_arm = short_smoke_steps + official_steps
        required_training_gpu_seconds_by_arm = {
            arm: optimizer_steps_per_arm * float(successful_by_arm[arm]["optimizer_step_seconds"])
            for arm in default_arms
        }
        required_training_gpu_seconds = sum(required_training_gpu_seconds_by_arm.values())
        required_remaining_gpu_seconds = (
            required_training_gpu_seconds + required_inference_gpu_seconds
        )
        required_total_gpu_seconds = estimator_wall_clock_seconds + required_remaining_gpu_seconds
        if include_ta and conditional_arm in successful_by_arm:
            conditional_training_gpu_seconds = optimizer_steps_per_arm * float(
                successful_by_arm[conditional_arm]["optimizer_step_seconds"]
            )
            conditional_additional_gpu_seconds = (
                conditional_training_gpu_seconds + conditional_ta_inference_gpu_seconds
            )
            conditional_total_gpu_seconds = (
                required_total_gpu_seconds + conditional_additional_gpu_seconds
            )
            conditional_projection = {
                "available": True,
                "additional_arm": conditional_arm,
                "additional_optimizer_steps": optimizer_steps_per_arm,
                "additional_training_gpu_seconds": conditional_training_gpu_seconds,
                "additional_inference_gpu_seconds": conditional_ta_inference_gpu_seconds,
                "additional_gpu_seconds": conditional_additional_gpu_seconds,
                "additional_cost_usd": conditional_additional_gpu_seconds * usd_per_second,
                "projected_total_gpu_seconds": conditional_total_gpu_seconds,
                "projected_total_cost_usd": conditional_total_gpu_seconds * usd_per_second,
            }
        elif include_ta:
            conditional_projection = {
                "available": False,
                "reason": "TA resource probe did not fit",
            }
        else:
            conditional_projection = {
                "available": False,
                "reason": "TA resource probe was not requested",
            }
        cost_projection = {
            "available": True,
            "currency": "USD",
            "hourly_price_usd": hourly_price_usd,
            "hourly_price_source": hourly_price_source,
            "method": "second_optimizer_step_time_times_contract_steps_plus_operator_inference_seconds",
            "estimator_wall_clock_seconds": estimator_wall_clock_seconds,
            "estimator_cost_usd": estimator_wall_clock_seconds * usd_per_second,
            "required_scenario": {
                "arms": list(default_arms),
                "optimizer_steps_per_arm": {
                    "short_smoke": short_smoke_steps,
                    "official": official_steps,
                    "total": optimizer_steps_per_arm,
                },
                "total_optimizer_steps": optimizer_steps_per_arm * len(default_arms),
                "training_gpu_seconds_by_arm": required_training_gpu_seconds_by_arm,
                "training_gpu_seconds": required_training_gpu_seconds,
                "inference_gpu_seconds": required_inference_gpu_seconds,
                "projected_remaining_gpu_seconds": required_remaining_gpu_seconds,
                "projected_remaining_cost_usd": required_remaining_gpu_seconds * usd_per_second,
                "projected_total_gpu_seconds": required_total_gpu_seconds,
                "projected_total_cost_usd": required_total_gpu_seconds * usd_per_second,
            },
            "conditional_ta_scenario": conditional_projection,
            "excluded_gpu_time": [
                "checkpoint_io_outside_estimator",
                "container_startup",
                "retries",
                "idle_provider_billing_outside_estimator",
            ],
        }
    else:
        cost_projection = {
            "available": False,
            "currency": "USD",
            "hourly_price_usd": hourly_price_usd,
            "hourly_price_source": hourly_price_source,
            "reason": "H-HEAD and T2-TOP must fit before projection",
        }

    payload = {
        "schema_version": 3,
        "artifact_role": "optional_cloud_resource_estimate",
        "optional": True,
        "material_training_authorization": False,
        "completed": len(arm_results) == len(selected_arms),
        "all_selected_arms_fit": len(arm_results) == len(selected_arms)
        and all(row["fit"] is True for row in arm_results),
        "checkpoint_persisted": False,
        "split_roles": ["PSEM-STRATEGY-TRAIN"],
        "eval_source_count": 0,
        "selected_arms": list(selected_arms),
        "ta_included": include_ta,
        "device": {
            "count": 1,
            "name": str(properties.name),
            "total_memory_bytes": total_memory_bytes,
            "torch_cuda_version": torch.version.cuda,
        },
        "host": {
            "cpu_model": platform.processor(),
            "vcpu_count": os.cpu_count(),
            "memory_bytes": host_memory_bytes,
            "storage": storage,
        },
        "precision_mode": "float32",
        "mixed_precision": False,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "runtime_contract_sha256": canonical_sha256(runtime_contract),
        "sampling_manifest_sha256": sampling_manifest_sha256,
        "probe_validation_sha256": canonical_sha256(probe_validation),
        "estimator_class_weights": {
            "replacement_positive": 1.0,
            "anchor_positive": 1.0,
            "source": "unit_weights_for_resource_estimation_only",
        },
        "probe_row_ids": [str(row["row_id"]) for row in probe_rows],
        "probe_input_identity_sha256": canonical_sha256(
            [
                {
                    key: row[key]
                    for key in (
                        "row_id",
                        "source_id",
                        "corpus",
                        "window_start_sample",
                        "window_end_sample",
                        "target_identity_sha256",
                        "augmentation_identity_sha256",
                        "state_reset_at_window_start",
                    )
                }
                for row in probe_rows
            ]
        ),
        "input_preparation_seconds": input_preparation_seconds,
        "input_preparation_to_fastest_optimizer_step_ratio": input_to_step_ratio,
        "io_bottleneck_observation": (
            "input_preparation_slower_than_fastest_optimizer_step"
            if input_to_step_ratio is not None and input_to_step_ratio > 1.0
            else "not_observed_in_probe"
        ),
        "estimator_wall_clock_seconds": estimator_wall_clock_seconds,
        "runtime_identity_sha256s": sorted(runtime_identity_sha256s),
        "arms": arm_results,
        "maximum_peak_reserved_memory_bytes": max(
            (int(row["peak_reserved_memory_bytes"]) for row in arm_results),
            default=0,
        ),
        "cost_projection": cost_projection,
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def build_cost_receipt(
    *,
    hourly_price_usd: float,
    hourly_price_source: str,
    actual_gpu_seconds: float,
    projected_remaining_gpu_seconds: float,
    command: str,
) -> dict[str, Any]:
    values = (hourly_price_usd, actual_gpu_seconds, projected_remaining_gpu_seconds)
    if (
        not hourly_price_source.strip()
        or not command.strip()
        or not all(np.isfinite(value) for value in values)
    ):
        raise ExecutionError("cost receipt inputs are invalid")
    if hourly_price_usd <= 0 or actual_gpu_seconds < 0 or projected_remaining_gpu_seconds < 0:
        raise ExecutionError("cost receipt inputs must be non-negative with a positive price")
    actual_cost = actual_gpu_seconds * hourly_price_usd / 3600.0
    remaining_cost = projected_remaining_gpu_seconds * hourly_price_usd / 3600.0
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_cost_receipt",
        "hourly_price_usd": float(hourly_price_usd),
        "hourly_price_source": hourly_price_source.strip(),
        "actual_gpu_seconds": float(actual_gpu_seconds),
        "projected_remaining_gpu_seconds": float(projected_remaining_gpu_seconds),
        "actual_cost_usd": actual_cost,
        "projected_remaining_cost_usd": remaining_cost,
        "projected_total_gpu_seconds": float(actual_gpu_seconds + projected_remaining_gpu_seconds),
        "projected_total_cost_usd": actual_cost + remaining_cost,
        "target_total_usd": 15.0,
        "hard_stop_usd": 30.0,
        "target_is_informational": True,
        "command": command.strip(),
    }
    if payload["projected_total_cost_usd"] > 30.0:
        raise ExecutionError("projected total cost exceeds the USD-30 hard stop")
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def run_smoke_arm(
    *,
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    corpus_root: Path,
    reference_root: Path,
    sampling_manifest: Path,
    class_weight_receipt: Mapping[str, Any],
    arm: str,
    device: str,
    ta_open_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require_material_execution_ready()
    if _eval_registry_marker().exists() or arm not in {"H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"}:
        raise ExecutionError("smoke arm is unavailable or EVAL is already open")
    if arm == "TA-ALL-TEMPORAL":
        if not isinstance(ta_open_authorization, Mapping):
            raise ExecutionError("TA smoke requires an explicit open_ta authorization")
        authorization = require_bound(
            ta_open_authorization, "psem_sortformer_open_ta_authorization"
        )
        if authorization.get("arm") != arm or authorization.get("seed") != 7301:
            raise ExecutionError("TA smoke authorization identity is invalid")
    elif ta_open_authorization is not None:
        raise ExecutionError("non-TA smoke cannot carry a TA authorization")
    seed_runtime(7301)
    rows = load_sampling_rows(sampling_manifest)
    sessions = load_training_sessions(corpus_root, reference_root)
    validation = validate_sampling_manifest(sampling_manifest, sessions)
    first_rows = rows[:512]
    if len(first_rows) != 512 or [row.get("epoch_index") for row in first_rows] != list(range(512)):
        raise ExecutionError("smoke requires the first 512 ordered epoch-1 rows")
    by_id = {str(row["row_id"]): row for row in rows}
    examples = [
        (
            prepare_training_example(
                row,
                sessions[str(row["source_id"])],
                corpus_root,
                str(row["corpus"]),
                manifest_path=sampling_manifest,
                manifest_validation=validation,
                manifest_rows_by_id=by_id,
            ),
        )
        for row in first_rows
    ]
    from experiments.psem_sortformer_adaptation_depth.training import (
        build_manifest_class_weight_receipt,
    )

    expected_class_weights = build_manifest_class_weight_receipt(rows, sessions, sampling_manifest)
    if class_weight_receipt != expected_class_weights:
        raise ExecutionError("smoke class weights differ from the one-epoch manifest")
    model, runtime_identity = load_pinned_sortformer(
        checkpoint_path, nemo_checkout, dependency_lock, device
    )
    if runtime_identity.get("checkpoint_sha256") != CHECKPOINT_SHA256 or runtime_identity.get(
        "dependency_lock_sha256"
    ) != runtime_identity.get("dependency_lock", {}).get("sha256"):
        raise ExecutionError("smoke runtime identity differs from the pinned runtime")
    parameter_policy = apply_parameter_policy(model, arm)
    class_weights = ClassWeights(
        replacement_positive=float(class_weight_receipt["replacement_positive_weight"]),
        anchor_positive=float(class_weight_receipt["anchor_positive_weight"]),
    )
    result = run_short_smoke(
        model,
        arm,
        class_weights,
        examples,
        expected_row_ids=[str(row["row_id"]) for row in first_rows],
        parameter_policy=parameter_policy,
    )
    payload = {
        **result,
        "sampling_manifest_sha256": sha256_file(sampling_manifest),
        "class_weight_receipt_sha256": class_weight_receipt["payload_sha256"],
        "parameter_policy_sha256": canonical_sha256(parameter_policy),
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "runtime_identity": runtime_identity,
        "base_checkpoint_sha256": runtime_identity["checkpoint_sha256"],
        "dependency_lock_sha256": runtime_identity["dependency_lock_sha256"],
        "weights_discarded": True,
        "ta_open_authorization_sha256": (
            ta_open_authorization.get("payload_sha256")
            if ta_open_authorization is not None
            else None
        ),
    }
    receipt = {**payload, "payload_sha256": canonical_sha256(payload)}
    return receipt


def run_training_arm(
    *,
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    corpus_root: Path,
    reference_root: Path,
    sampling_manifest: Path,
    class_weight_receipt: Mapping[str, Any],
    material_gate: Mapping[str, Any],
    output_root: Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_material_execution_ready()
    assert_clean_candidate()
    rows = load_sampling_rows(sampling_manifest)
    sessions = load_training_sessions(corpus_root, reference_root)
    validation = validate_sampling_manifest(sampling_manifest, sessions)
    if _eval_registry_marker().exists():
        raise ExecutionError("official training cannot start after EVAL opened")
    authorization = authorize_official_training(material_gate, rows, class_weight_receipt)
    training_device = torch.device(device)
    if training_device.type != "cuda" or not torch.cuda.is_available():
        raise ExecutionError("official training requires one CUDA device")
    seed_runtime(authorization.seed)
    model, runtime_identity = load_pinned_sortformer(
        checkpoint_path, nemo_checkout, dependency_lock, device
    )
    parameter_policy = apply_parameter_policy(model, authorization.arm)
    output_root = output_root.resolve()
    checkpoint_path_out = output_root / "checkpoints" / authorization.arm / "7301" / "step-256.pt"
    checkpoint_state: dict[str, Any] = {}

    def save_final(
        current_model: TrainableSortformerPSEM, step: int, metrics: Mapping[str, Any]
    ) -> None:
        if step != 256 or checkpoint_state:
            raise ExecutionError("only the final step-256 checkpoint may be persisted")
        digest = _save_model_checkpoint(current_model, checkpoint_path_out, authorization.arm, 7301)
        checkpoint_state.update({"step": step, "metrics": dict(metrics), "sha256": digest})

    torch.cuda.reset_peak_memory_stats(training_device)
    started = time.perf_counter()
    training_result = fit_arm(
        model,
        authorization.arm,
        authorization.class_weights,
        lambda epoch: _training_batches(
            epoch=epoch,
            rows=rows,
            sessions=sessions,
            corpus_root=corpus_root,
            manifest_path=sampling_manifest,
            manifest_validation=validation,
        ),
        256,
        None,
        save_final,
        authorization=authorization,
    )
    wall_clock = time.perf_counter() - started
    if checkpoint_state.get("step") != 256:
        raise ExecutionError("final checkpoint step is not 256")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    peak_training_memory_bytes = int(torch.cuda.max_memory_allocated(training_device))
    if peak_training_memory_bytes <= 0:
        raise ExecutionError("official training did not produce positive CUDA memory evidence")
    summary = {
        "final_step": 256,
        "optimizer_steps": 256,
        "scheduler_total_steps": 256,
        "warmup_steps": 13,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "consumed_row_count": 4096,
        "training_wall_clock_seconds": wall_clock,
        "peak_training_memory_bytes": peak_training_memory_bytes,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "dev_callback_used": False,
        "early_stopping_used": False,
        "native_diarization_contract_passed": bool(
            material_gate.get("passed") is True
            and material_gate.get("short_smoke_receipt_sha256")
        ),
        "native_diarization_contract_evidence_sha256": canonical_sha256(
            {
                "short_smoke_receipt_sha256": material_gate["short_smoke_receipt_sha256"],
            }
        ),
    }
    code_identity = candidate_code_identity()
    training_payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_training_result",
        **training_result,
        "checkpoint_path": str(checkpoint_path_out.resolve()),
        "checkpoint_sha256": checkpoint_state["sha256"],
        "checkpoint_size_bytes": checkpoint_path_out.stat().st_size,
        "training_summary": summary,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "parameter_policy": parameter_policy,
        "parameter_policy_sha256": canonical_sha256(parameter_policy),
        "candidate_code_identity_sha256": code_identity["payload_sha256"],
        "split_roles": [TRAIN_ROLE],
        "eval_source_count": 0,
    }
    training_bound = {**training_payload, "payload_sha256": canonical_sha256(training_payload)}
    checkpoint_payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_checkpoint",
        "arm": authorization.arm,
        "seed": 7301,
        "checkpoint_path": str(checkpoint_path_out.resolve()),
        "checkpoint_sha256": checkpoint_state["sha256"],
        "checkpoint_size_bytes": checkpoint_path_out.stat().st_size,
        "final_step": 256,
        "material_gate_sha256": material_gate["payload_sha256"],
        "material_training_authorization": dict(material_gate),
        "authorized_output_root": str(output_root),
        "candidate_code_identity_sha256": code_identity["payload_sha256"],
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "parameter_policy": parameter_policy,
        "parameter_policy_sha256": canonical_sha256(parameter_policy),
        "training_summary": summary,
        "training_result_sha256": training_bound["payload_sha256"],
        "training_result": training_bound,
        "split_roles": [TRAIN_ROLE],
        "eval_source_count": 0,
    }
    checkpoint_receipt = {
        **checkpoint_payload,
        "payload_sha256": canonical_sha256(checkpoint_payload),
    }
    return training_bound, checkpoint_receipt


def run_overfit_arm_result(
    *args: Any, **kwargs: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raise ExecutionError("legacy overfit arm is not supported; use smoke-arm")
