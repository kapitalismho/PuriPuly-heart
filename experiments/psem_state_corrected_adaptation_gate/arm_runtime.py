from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
THREAD_CAP_VALUE = "1"
_ENV_CAPS_APPLIED = False
_TORCH_CAPPED_MODULE: Any | None = None


def _torch_module() -> Any:
    return __import__("sys").modules.get("torch")


def apply_thread_caps() -> dict[str, Any]:
    global _ENV_CAPS_APPLIED, _TORCH_CAPPED_MODULE
    for name in THREAD_ENV_VARS:
        os.environ[name] = THREAD_CAP_VALUE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _ENV_CAPS_APPLIED = True
    torch_mod = _torch_module()
    if torch_mod is not None and _TORCH_CAPPED_MODULE is not torch_mod:
        try:
            torch_mod.set_num_threads(1)
            try:
                torch_mod.set_num_interop_threads(1)
            except (RuntimeError, AttributeError):
                pass
            _TORCH_CAPPED_MODULE = torch_mod
        except (RuntimeError, AttributeError):
            pass
    return thread_cap_receipt()


def enforce_thread_caps() -> dict[str, Any]:
    receipt = apply_thread_caps()
    torch_mod = _torch_module()
    if torch_mod is not None:
        try:
            threads = int(torch_mod.get_num_threads())
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            raise ArmError(f"torch thread state is unreadable: {exc}") from exc
        if threads != 1:
            raise ArmError(f"torch intra-op threads are {threads}, required 1")
        try:
            interop = int(torch_mod.get_num_interop_threads())
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            raise ArmError(f"torch inter-op thread state is unreadable: {exc}") from exc
        if interop != 1:
            raise ArmError(f"torch inter-op threads are {interop}, required 1")
        receipt["torch_num_threads"] = threads
        receipt["torch_num_interop_threads"] = interop
    return receipt


def thread_cap_receipt() -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
    }
    torch_mod = _torch_module()
    if torch_mod is not None:
        try:
            receipt["torch_num_threads"] = int(torch_mod.get_num_threads())
        except (RuntimeError, AttributeError, TypeError, ValueError):
            receipt["torch_num_threads"] = None
        try:
            receipt["torch_num_interop_threads"] = int(torch_mod.get_num_interop_threads())
        except (RuntimeError, AttributeError, TypeError, ValueError):
            receipt["torch_num_interop_threads"] = None
    return receipt


def spawn_worker_init() -> None:
    apply_thread_caps()


_spawn_worker_init = spawn_worker_init


apply_thread_caps()


ARM_R_H_SC = "R-H-SC"
ARM_R_T2_SC = "R-T2-SC"
ARM_R_TA_SC = "R-TA-SC"
ALL_ARMS = (ARM_R_H_SC, ARM_R_T2_SC, ARM_R_TA_SC)

SCREEN_SEED = 7301
CONFIRM_SEED = 7302
ALLOWED_SEEDS = (SCREEN_SEED, CONFIRM_SEED)

RUN_ROOT_PREFIX = Path("/workspace/issue-121/arms")

HEAD_INPUT_DIM = 199
HEAD_INPUT_PARTS = {
    "hidden": 192,
    "slot_logits": 4,
    "selected": 1,
    "max_nonanchor": 1,
    "delay": 1,
}

OPTIMIZER_CONTRACT = {
    "name": "AdamW",
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "group_lrs": {"head": 1e-3, "activity": 1e-4, "temporal": 1e-5},
    "microbatch": 1,
    "accumulation": 16,
    "passes": 1,
    "augmentation": False,
}

ACCUMULATION = 16
DEFAULT_WORKER_CAP = 24
WAVEFORM_PREFETCH_CAP = 4

P0_RECEIPT_NAME = "p0_pass.json"
GATE1_RECEIPT_NAME = "gate1.json"
GATE2_RECEIPT_NAME = "gate2.json"
CHECKPOINT_NAME = "checkpoint.json"
CHECKPOINT_DIRNAME = "checkpoints"
CHECKPOINT_ROLES = ("model", "optimizer", "scheduler", "rng")
PREDICTIONS_DIRNAME = "predictions"
FINAL_MANIFEST_NAME = "final_manifest.json"


class ArmError(RuntimeError):
    pass


class AuthorizationError(ArmError):
    pass


class CheckpointError(ArmError):
    pass


def canonical_arm(arm: str) -> str:
    if arm not in ALL_ARMS:
        raise AuthorizationError(f"arm is not authorized: {arm}")
    return arm


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_HASH_RE = re.compile(r"[0-9a-f]{64}")


def _require_hash(value: str, name: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ArmError(f"{name} must be a lowercase 64-hex sha256")
    return value


@dataclass(slots=True, frozen=True)
class ArmRunConfig:
    arm: str
    seed: int
    root: Path
    input_hash: str
    checkpoint_hash: str
    partition_hash: str
    weights_hash: str
    code_hash: str
    class_weights: dict[str, float] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.arm not in ALL_ARMS:
            raise AuthorizationError(f"arm is not authorized: {self.arm}")
        if self.seed not in ALLOWED_SEEDS:
            raise AuthorizationError(f"seed is not authorized: {self.seed}")
        _require_hash(self.input_hash, "input_hash")
        _require_hash(self.checkpoint_hash, "checkpoint_hash")
        _require_hash(self.partition_hash, "partition_hash")
        _require_hash(self.weights_hash, "weights_hash")
        _require_hash(self.code_hash, "code_hash")
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "class_weights", dict(self.class_weights))

    @property
    def binding(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "input_hash": self.input_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "partition_hash": self.partition_hash,
            "weights_hash": self.weights_hash,
            "code_hash": self.code_hash,
            "optimizer_contract": {k: v for k, v in OPTIMIZER_CONTRACT.items()},
        }

    @property
    def config_hash(self) -> str:
        return canonical_sha256(self.binding)

    def run_dir(self) -> Path:
        return self.root / self.arm / str(self.seed)


def config_from_dict(payload: dict[str, Any], root: Path | str | None = None) -> ArmRunConfig:
    base = dict(payload)
    if root is not None:
        base["root"] = Path(root)
    return ArmRunConfig(
        arm=str(base["arm"]),
        seed=int(base["seed"]),
        root=Path(base["root"]),
        input_hash=str(base["input_hash"]),
        checkpoint_hash=str(base["checkpoint_hash"]),
        partition_hash=str(base["partition_hash"]),
        weights_hash=str(base["weights_hash"]),
        code_hash=str(base["code_hash"]),
        class_weights=dict(base.get("class_weights", {})),
    )


def load_config(path: Path) -> ArmRunConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArmError(f"config is invalid: {path}")
    return config_from_dict(payload)


def bind_class_weights(weights: dict[str, float]) -> tuple[dict[str, float], str]:
    bound = {str(k): float(v) for k, v in weights.items()}
    if set(bound) != {"replacement_positive_weight", "anchor_positive_weight"}:
        raise ArmError("class weights must carry replacement and anchor positive weights")
    for value in bound.values():
        if not math.isfinite(value) or value <= 0:
            raise ArmError("class weights must be finite and positive")
    return bound, canonical_sha256(bound)


def check_head_input_dim(dim: int) -> int:
    if int(dim) != HEAD_INPUT_DIM:
        raise ArmError(f"head input dim must be {HEAD_INPUT_DIM}")
    if sum(HEAD_INPUT_PARTS.values()) != HEAD_INPUT_DIM:
        raise ArmError("head input parts do not sum to 199")
    return HEAD_INPUT_DIM


def chronological_sources(source_ids: list[str]) -> list[str]:
    return sorted(source_ids)


def compute_total_steps(n_loss_chunks: int, accumulation: int = ACCUMULATION) -> int:
    if n_loss_chunks <= 0:
        return 0
    if accumulation <= 0:
        raise ArmError("accumulation must be positive")
    return -(-int(n_loss_chunks) // int(accumulation))


def compute_warmup_steps(total_steps: int) -> int:
    if total_steps <= 0:
        return 0
    return max(1, math.ceil(0.05 * total_steps))


def plan_schedule(
    source_ids: list[str],
    loss_flags: dict[str, list[bool]],
    accumulation: int = ACCUMULATION,
) -> dict[str, Any]:
    ordered = chronological_sources(list(source_ids))
    chunks: list[dict[str, Any]] = []
    for source_id in ordered:
        flags = [bool(v) for v in loss_flags.get(source_id, [])]
        for index, contributes in enumerate(flags):
            chunks.append(
                {"source": source_id, "chunk_index": index, "contributes": contributes}
            )
    loss_total = sum(1 for chunk in chunks if chunk["contributes"])
    total_steps = compute_total_steps(loss_total, accumulation)
    warmup_steps = compute_warmup_steps(total_steps)
    seen = 0
    for chunk in chunks:
        if not chunk["contributes"]:
            chunk["accum_position"] = None
            chunk["optimizer_step"] = None
            chunk["is_step_boundary"] = False
            continue
        seen += 1
        position = (seen - 1) % accumulation
        chunk["accum_position"] = position
        chunk["is_step_boundary"] = position == accumulation - 1 or seen == loss_total
        chunk["optimizer_step"] = (seen - 1) // accumulation
    return {
        "sources": ordered,
        "chunks": chunks,
        "loss_chunks": loss_total,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "accumulation": accumulation,
    }

def _physical_cpu_count() -> int | None:
    try:
        affinity = os.sched_affinity(0)
        if affinity:
            return int(len(affinity))
    except (AttributeError, OSError, NotImplementedError):
        pass
    try:
        import psutil as _psutil
        physical = _psutil.cpu_count(logical=False)
        if physical:
            return int(physical)
    except (ImportError, AttributeError, NotImplementedError):
        pass
    return None


def default_worker_limit(cap: int = DEFAULT_WORKER_CAP) -> int:
    logical = int(os.cpu_count() or 1)
    physical = _physical_cpu_count() or logical
    return max(1, min(int(cap), int(physical)))


def resolve_workers(requested: int | None, cap: int = DEFAULT_WORKER_CAP) -> int:
    if requested is None:
        return default_worker_limit(cap)
    logical = int(os.cpu_count() or 1)
    limit = max(1, min(int(cap), logical))
    return max(1, min(int(requested), limit))


def resolve_prefetch_depth(workers: int, num_items: int, cap: int = WAVEFORM_PREFETCH_CAP) -> int:
    return max(0, min(int(cap), int(workers), int(num_items)))


def worker_receipt(requested: int | None, num_items: int) -> dict[str, Any]:
    effective = resolve_workers(requested)
    if num_items <= 1:
        effective = 1
    return {
        "requested_workers": requested,
        "effective_workers": int(effective),
        "worker_cap": int(DEFAULT_WORKER_CAP),
        "cpu_count": int(os.cpu_count() or 1),
        "physical_cpus": _physical_cpu_count(),
        "backend": "spawn" if int(effective) > 1 and int(num_items) > 1 else "serial",
        "ordered": True,
        "thread_caps": thread_cap_receipt(),
    }


def ordered_process_map(
    worker_fn: Callable[[Any], Any], payloads: list[Any], workers: int
) -> list[Any]:
    items = list(payloads)
    apply_thread_caps()
    if workers <= 1 or len(items) <= 1:
        return [worker_fn(payload) for payload in items]
    import concurrent.futures
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=context,
        initializer=spawn_worker_init,
    ) as pool:
        return list(pool.map(worker_fn, items))


def _read_receipt(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_valid_final_manifest(
    root: Path, arm: str, seed: int, config: ArmRunConfig
) -> tuple[Path, str]:
    root_resolved = Path(root).resolve()
    candidate_dir = root_resolved / arm / str(seed)
    try:
        candidate_dir.relative_to(root_resolved)
    except ValueError as exc:
        raise AuthorizationError("prior candidate path escapes run root") from exc
    manifest_path = candidate_dir / FINAL_MANIFEST_NAME
    if not manifest_path.is_file():
        raise AuthorizationError(f"prior candidate manifest is missing: {arm}/{seed}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuthorizationError(f"prior candidate manifest is unreadable: {arm}/{seed}") from exc
    if not isinstance(manifest, dict):
        raise AuthorizationError(f"prior candidate manifest is invalid: {arm}/{seed}")
    if manifest.get("arm") != arm or manifest.get("seed") != seed:
        raise AuthorizationError(f"prior candidate identity mismatch: {arm}/{seed}")
    binding = manifest.get("binding")
    if isinstance(binding, dict):
        if binding.get("arm") != arm or binding.get("seed") != seed:
            raise AuthorizationError(f"prior candidate binding mismatch: {arm}/{seed}")
        if binding.get("input_hash") != config.input_hash:
            raise AuthorizationError(f"prior candidate input mismatch: {arm}/{seed}")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise AuthorizationError(f"prior candidate artifacts are missing: {arm}/{seed}")
    base = candidate_dir.resolve()
    for entry in entries:
        if not isinstance(entry, dict):
            raise AuthorizationError(f"prior candidate artifact is invalid: {arm}/{seed}")
        raw = entry.get("path")
        if not isinstance(raw, str) or not raw:
            raise AuthorizationError(f"prior candidate artifact is invalid: {arm}/{seed}")
        full = Path(raw) if Path(raw).is_absolute() else base / raw
        try:
            full.resolve().relative_to(base)
        except ValueError as exc:
            raise AuthorizationError(
                f"prior candidate artifact escapes run dir: {arm}/{seed}"
            ) from exc
        if not full.is_file():
            raise AuthorizationError(f"prior candidate artifact is missing: {arm}/{seed}")
        data = full.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            raise AuthorizationError(f"prior candidate artifact hash mismatch: {arm}/{seed}")
        if "size" in entry and len(data) != int(entry["size"]):
            raise AuthorizationError(f"prior candidate artifact size mismatch: {arm}/{seed}")
    return manifest_path, sha256_file(manifest_path)


def check_authorization(config: ArmRunConfig, store: Path) -> dict[str, Any]:
    from experiments.psem_state_corrected_adaptation_gate import gates as gates_mod

    canonical_arm(config.arm)
    if config.seed not in ALLOWED_SEEDS:
        raise AuthorizationError(f"seed is not authorized: {config.seed}")
    store = Path(store)
    try:
        gates_mod.check_p0_receipt(
            _read_receipt(store / P0_RECEIPT_NAME),
            config.input_hash,
            config.checkpoint_hash,
            config.partition_hash,
        )
    except gates_mod.GateError as exc:
        raise AuthorizationError(f"H gate blocked: {exc}") from exc
    if config.arm in (ARM_R_T2_SC, ARM_R_TA_SC):
        gate1_record = _read_receipt(store / GATE1_RECEIPT_NAME)
        try:
            gates_mod.check_gate1_receipt(gate1_record, config.input_hash)
        except gates_mod.GateError as exc:
            raise AuthorizationError(f"Gate-1 blocked: {exc}") from exc
        _, h_digest = _load_valid_final_manifest(
            config.root, ARM_R_H_SC, SCREEN_SEED, config
        )
        if not isinstance(gate1_record, dict) or gate1_record.get("h_candidate_hash") != h_digest:
            raise AuthorizationError("Gate-1 candidate mismatch")
    if config.arm == ARM_R_TA_SC:
        gate2_record = _read_receipt(store / GATE2_RECEIPT_NAME)
        try:
            gates_mod.check_gate2_receipt(gate2_record, config.input_hash)
        except gates_mod.GateError as exc:
            raise AuthorizationError(f"Gate-2 blocked: {exc}") from exc
        _, t2_digest = _load_valid_final_manifest(
            config.root, ARM_R_T2_SC, CONFIRM_SEED, config
        )
        if not isinstance(gate2_record, dict) or gate2_record.get("t2_candidate_hash") != t2_digest:
            raise AuthorizationError("Gate-2 candidate mismatch")
    if config.seed == CONFIRM_SEED:
        confirm_name = f"confirmation_{config.arm}.json"
        confirm_record = _read_receipt(store / confirm_name)
        try:
            gates_mod.check_confirmation_receipt(confirm_record, config.arm, config.input_hash)
        except gates_mod.GateError as exc:
            raise AuthorizationError(f"confirmation gate blocked: {exc}") from exc
        _, prior_digest = _load_valid_final_manifest(
            config.root, config.arm, SCREEN_SEED, config
        )
        if not isinstance(confirm_record, dict) or confirm_record.get("candidate_hash") != prior_digest:
            raise AuthorizationError("confirmation candidate mismatch")
    return {
        "arm": config.arm,
        "seed": config.seed,
        "config_hash": config.config_hash,
        "authorized": True,
    }


def authorize_and_run(
    config: ArmRunConfig,
    store: Path,
    executor: Callable[[ArmRunConfig], Any],
) -> Any:
    receipt = check_authorization(config, store)
    result = executor(config)
    return {"authorization": receipt, "result": result}

def _unique_tmp(target: Path) -> Path:
    return target.with_name(f"{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp(target)
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp, target)
    return target


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp(target)
    tmp.write_bytes(bytes(data))
    os.replace(tmp, target)
    return target


def _checkpoint_blob_path(checkpoint_dir: Path, source_id: str, role: str) -> Path:
    return checkpoint_dir / f"{source_id}.{role}.pt"


def _confine_blob(run_dir: Path, relpath: str) -> Path:
    if not isinstance(relpath, str) or not relpath or relpath != Path(relpath).name:
        raise CheckpointError("checkpoint blob path is not confined")
    checkpoint_dir = (Path(run_dir) / CHECKPOINT_DIRNAME).resolve()
    candidate = (checkpoint_dir / relpath).resolve()
    if candidate.parent != checkpoint_dir:
        raise CheckpointError("checkpoint blob escapes checkpoint dir")
    return candidate


def save_source_checkpoint(
    run_dir: Path,
    source_id: str,
    completed_sources: list[str],
    binding: dict[str, Any],
    blobs: dict[str, bytes] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    if not isinstance(source_id, str) or not source_id:
        raise CheckpointError("checkpoint source id is invalid")
    prior_completed = list(completed_sources)
    if any(not isinstance(v, str) for v in prior_completed):
        raise CheckpointError("checkpoint ledger is invalid")
    if len(set(prior_completed)) != len(prior_completed):
        raise CheckpointError("checkpoint ledger holds duplicates")
    if source_id in prior_completed:
        raise CheckpointError(f"checkpoint source is already completed: {source_id}")
    if set(dict(blobs or {})) != set(CHECKPOINT_ROLES):
        raise CheckpointError("checkpoint requires model/optimizer/scheduler/rng blobs")
    for role, data in dict(blobs or {}).items():
        if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
            raise CheckpointError(f"checkpoint blob is empty: {role}")
    checkpoint_dir = run_dir / CHECKPOINT_DIRNAME
    manifest_path = checkpoint_dir / CHECKPOINT_NAME
    prior: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CheckpointError(f"checkpoint is unreadable: {manifest_path}") from exc
        if not isinstance(loaded, dict) or loaded.get("binding") != dict(binding):
            raise CheckpointError("checkpoint binding mismatch")
        stored_completed = loaded.get("completed_sources")
        if (
            not isinstance(stored_completed, list)
            or any(not isinstance(v, str) for v in stored_completed)
            or list(stored_completed) != prior_completed
        ):
            raise CheckpointError("checkpoint ledger diverges from completed sources")
        sources = loaded.get("sources")
        if not isinstance(sources, dict) or set(sources) != set(stored_completed):
            raise CheckpointError("checkpoint ledger is invalid")
        prior = {str(k): v for k, v in sources.items()}
    written: dict[str, dict[str, Any]] = {}
    for role in CHECKPOINT_ROLES:
        data = bytes(dict(blobs or {})[role])
        target = _checkpoint_blob_path(checkpoint_dir, source_id, role)
        atomic_write_bytes(target, data)
        digest = hashlib.sha256(data).hexdigest()
        written[role] = {
            "path": target.name,
            "sha256": digest,
            "size": len(data),
        }
    prior[source_id] = written
    completed = [*prior_completed, source_id]
    if set(prior) != set(completed):
        raise CheckpointError("checkpoint ledger diverges from completed sources")
    payload = {
        "artifact_role": "issue-121-source-checkpoint",
        "completed_sources": completed,
        "binding": dict(binding),
        "sources": prior,
    }
    atomic_write_json(manifest_path, payload)
    referenced = {
        f"{kept_id}.{role}.pt" for kept_id in prior for role in CHECKPOINT_ROLES
    }
    try:
        children = list(checkpoint_dir.iterdir())
    except OSError:
        children = []
    for child in children:
        if not child.is_file() or not child.name.endswith(".pt"):
            continue
        if child.name in referenced:
            continue
        try:
            child.unlink()
        except OSError:
            pass
    return manifest_path





def load_source_checkpoint(run_dir: Path, expected_binding: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    path = run_dir / CHECKPOINT_DIRNAME / CHECKPOINT_NAME
    if not path.is_file():
        return {"completed_sources": [], "binding": dict(expected_binding), "blobs": {}, "fresh": True}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"checkpoint is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint is invalid: {path}")
    stored = payload.get("binding")
    if stored != dict(expected_binding):
        raise CheckpointError("checkpoint binding mismatch")
    completed = payload.get("completed_sources")
    if (
        not isinstance(completed, list)
        or any(not isinstance(v, str) for v in completed)
        or len(set(completed)) != len(completed)
    ):
        raise CheckpointError("checkpoint ledger is invalid")
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise CheckpointError("checkpoint ledger is invalid")
    if set(sources) != set(completed):
        raise CheckpointError("checkpoint ledger diverges from blob manifest")
    verified: dict[str, dict[str, Path]] = {}
    for source_id in completed:
        entry = sources.get(source_id)
        if not isinstance(entry, dict):
            raise CheckpointError(f"checkpoint blobs are incomplete: {source_id}")
        if entry.get("retained") is False:
            roles = entry.get("roles")
            if not isinstance(roles, dict) or set(roles) != set(CHECKPOINT_ROLES):
                raise CheckpointError(f"checkpoint milestone evidence is incomplete: {source_id}")
            for role in CHECKPOINT_ROLES:
                record = roles.get(role, {})
                if not str(record.get("sha256", "")) or int(record.get("size", -1)) < 0:
                    raise CheckpointError(f"checkpoint milestone evidence is invalid: {source_id}.{role}")
            continue
        if set(entry) != set(CHECKPOINT_ROLES):
            raise CheckpointError(f"checkpoint blobs are incomplete: {source_id}")
        resolved_roles: dict[str, Path] = {}
        for role in CHECKPOINT_ROLES:
            record = entry.get(role)
            if not isinstance(record, dict):
                raise CheckpointError(f"checkpoint blob record is invalid: {source_id}.{role}")
            blob_path = _confine_blob(run_dir, str(record.get("path", "")))
            if not blob_path.is_file():
                raise CheckpointError(f"checkpoint blob is missing: {source_id}.{role}")
            data = blob_path.read_bytes()
            if len(data) != int(record.get("size", -1)):
                raise CheckpointError(f"checkpoint blob size mismatch: {source_id}.{role}")
            if hashlib.sha256(data).hexdigest() != record.get("sha256"):
                raise CheckpointError(f"checkpoint blob hash mismatch: {source_id}.{role}")
            resolved_roles[role] = blob_path
        verified[source_id] = resolved_roles
    if completed and completed[-1] not in verified:
        raise CheckpointError("checkpoint latest blobs are missing")
    return {"completed_sources": list(completed), "binding": stored, "blobs": verified, "fresh": False}


def resume_plan(fit_sources: list[str], completed: list[str]) -> list[str]:
    ordered = chronological_sources(list(fit_sources))
    done = list(completed)
    if any(not isinstance(v, str) for v in done):
        raise CheckpointError("checkpoint ledger is invalid")
    if len(set(ordered)) != len(ordered):
        raise CheckpointError("TRAIN-FIT partition holds duplicates")
    if len(set(done)) != len(done):
        raise CheckpointError("checkpoint ledger holds duplicates")
    unknown = [v for v in done if v not in set(ordered)]
    if unknown:
        raise CheckpointError(f"checkpoint ledger holds unknown sources: {unknown}")
    if done != ordered[: len(done)]:
        raise CheckpointError("checkpoint ledger is not a chronological prefix")
    return ordered[len(done):]



def save_source_predictions(
    run_dir: Path,
    source_id: str,
    predictions: dict[str, Any],
    binding: dict[str, Any],
) -> Path:
    payload = {
        "artifact_role": "issue-121-source-predictions",
        "source_id": source_id,
        "predictions": dict(predictions),
        "binding": dict(binding),
    }
    target = Path(run_dir) / PREDICTIONS_DIRNAME / f"{source_id}.json"
    return atomic_write_json(target, payload)


def load_source_predictions(
    run_dir: Path, source_id: str, expected_binding: dict[str, Any]
) -> dict[str, Any]:
    path = Path(run_dir) / PREDICTIONS_DIRNAME / f"{source_id}.json"
    if not path.is_file():
        raise CheckpointError(f"predictions are missing: {source_id}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"predictions are unreadable: {source_id}") from exc
    if not isinstance(payload, dict) or payload.get("source_id") != source_id:
        raise CheckpointError(f"predictions are invalid: {source_id}")
    if payload.get("binding") != dict(expected_binding):
        raise CheckpointError("prediction binding mismatch")
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict):
        raise CheckpointError("prediction payload is invalid")
    return predictions


def record_mapping_diagnostics(
    run_dir: Path,
    source_id: str,
    coverage: float,
    agreement: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "artifact_role": "issue-121-mapping-diagnostics",
        "source_id": source_id,
        "coverage": float(coverage),
        "agreement": float(agreement),
        "extra": dict(extra or {}),
    }
    target = Path(run_dir) / "diagnostics" / f"{source_id}.mapping.json"
    return atomic_write_json(target, payload)


def write_final_manifest(
    run_dir: Path, manifest: dict[str, Any], artifacts: list[Path]
) -> Path:
    run_dir = Path(run_dir)
    entries: list[dict[str, str]] = []
    for artifact in artifacts:
        path = Path(artifact)
        if not path.is_file():
            raise ArmError(f"final artifact is missing: {path}")
        entries.append({"path": str(path), "sha256": sha256_file(path)})
    body = {**dict(manifest), "artifacts": entries}
    return atomic_write_json(run_dir / FINAL_MANIFEST_NAME, body)


ARM_GPU_LOCK_NAME = "arm_gpu.lock"


def _gpu_lock_path(scope_dir: Path) -> Path:
    return Path(scope_dir) / ARM_GPU_LOCK_NAME


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, OverflowError):
        return False
    return True


def acquire_arm_gpu_lock(scope_dir: Path, owner: dict[str, Any]) -> Path:
    path = _gpu_lock_path(scope_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "pid": int(os.getpid()),
        "time": time.time(),
        **{str(k): v for k, v in dict(owner).items()},
    }
    body = json.dumps(record, sort_keys=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, body.encode("utf-8"))
        finally:
            os.close(fd)
        return path
    except FileExistsError:
        pass
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArmError(f"GPU lock is unreadable: {exc}") from exc
    try:
        prior = json.loads(raw)
    except ValueError as exc:
        raise ArmError(f"GPU lock ownership is unknowable, refusing reclaim: {exc}") from exc
    if not isinstance(prior, dict):
        raise ArmError("GPU lock ownership is unknowable, refusing reclaim")
    try:
        pid = int(prior.get("pid", -1))
    except (TypeError, ValueError):
        raise ArmError("GPU lock owner PID is unknowable, refusing reclaim")
    if pid > 0 and _pid_alive(pid):
        raise ArmError(
            f"GPU arm already owned by live pid {pid}: {prior.get('run_id', '')}"
        )
    try:
        path.unlink()
    except OSError as exc:
        raise ArmError(f"GPU lock is not releasable: {exc}") from exc
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, body.encode("utf-8"))
    finally:
        os.close(fd)
    return path


def release_arm_gpu_lock(path: Path) -> None:
    try:
        prior = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prior = None
    if isinstance(prior, dict) and int(prior.get("pid", -1)) == int(os.getpid()):
        try:
            Path(path).unlink()
        except OSError:
            pass


class arm_gpu_lock:
    def __init__(self, scope_dir: Path, owner: dict[str, Any]) -> None:
        self.scope_dir = Path(scope_dir)
        self.owner = dict(owner)
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.path = acquire_arm_gpu_lock(self.scope_dir, self.owner)
        return self.path

    def __exit__(self, *exc: Any) -> None:
        if self.path is not None:
            release_arm_gpu_lock(self.path)
            self.path = None
