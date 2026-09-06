from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import pickle
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from _pytest.monkeypatch import MonkeyPatch

from experiments.psem_state_corrected_adaptation_gate import arm_runtime
from experiments.psem_state_corrected_adaptation_gate import material
from experiments.psem_state_corrected_adaptation_gate import temporal_train


def _h(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _bound_config(tmp: Path, manifest: dict, checkpoint: Path, arm: str, seed: int):
    partition_hash = arm_runtime.canonical_sha256(
        {
            "fit": sorted(manifest["fit"]),
            "calib": sorted(manifest["calib"]),
            "salt": str(manifest.get("salt", "")),
            "target_frac": float(manifest.get("target_frac", 0.0)),
        }
    )
    _, weights_hash = arm_runtime.bind_class_weights(dict(manifest["class_weights"]))
    return arm_runtime.config_from_dict(
        {
            "arm": arm,
            "seed": seed,
            "root": tmp / "arms",
            "input_hash": str(manifest["sampling_sha256"]),
            "checkpoint_hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "partition_hash": partition_hash,
            "weights_hash": weights_hash,
            "code_hash": temporal_train._code_identity(),
        }
    )


def _p0(store: Path, config) -> None:
    store.mkdir(parents=True, exist_ok=True)
    (store / "p0_pass.json").write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "input_hash": config.input_hash,
                "checkpoint_hash": config.checkpoint_hash,
                "partition_hash": config.partition_hash,
            }
        ),
        encoding="utf-8",
    )


def _gate1(store: Path, config, h_digest: str = "h1") -> None:
    (store / "gate1.json").write_text(
        json.dumps(
            {
                "decision": "OPEN-T2",
                "h_candidate_hash": h_digest,
                "input_hash": config.input_hash,
            }
        ),
        encoding="utf-8",
    )


def _h_manifest(root: Path, config) -> str:
    candidate_dir = Path(root) / arm_runtime.ARM_R_H_SC / str(arm_runtime.SCREEN_SEED)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    artifact = candidate_dir / "artifact.bin"
    artifact.write_bytes(b"h-candidate-artifact")
    manifest = {
        "arm": arm_runtime.ARM_R_H_SC,
        "seed": arm_runtime.SCREEN_SEED,
        "binding": {
            "arm": arm_runtime.ARM_R_H_SC,
            "seed": arm_runtime.SCREEN_SEED,
            "input_hash": config.input_hash,
        },
        "artifacts": [
            {
                "path": "artifact.bin",
                "sha256": hashlib.sha256(b"h-candidate-artifact").hexdigest(),
            }
        ],
    }
    manifest_path = candidate_dir / arm_runtime.FINAL_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return arm_runtime.sha256_file(manifest_path)


def _authorize_t2(root: Path, store: Path, config) -> None:
    _p0(store, config)
    _gate1(store, config, _h_manifest(root, config))


class FakeTensor:
    def __init__(self, data, dtype=None, device="cpu"):
        if isinstance(data, FakeTensor):
            data = data._a
        self._a = np.asarray(data, dtype=np.float64)
        self.dtype = dtype
        self.device = device
        self.grad = None

    @property
    def shape(self):
        return tuple(self._a.shape)

    def __len__(self):
        return int(self._a.shape[0])

    def __bool__(self):
        return bool(np.all(self._a))

    def __getitem__(self, idx):
        return FakeTensor(self._a[idx], self.dtype, self.device)

    def __setitem__(self, idx, value):
        self._a[idx] = value._a if isinstance(value, FakeTensor) else value

    def _bin(self, other, op):
        other = other._a if isinstance(other, FakeTensor) else other
        return FakeTensor(op(self._a, other), self.dtype, self.device)

    def __add__(self, other):
        return self._bin(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._bin(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._bin(other, lambda a, b: a - b)

    def __rsub__(self, other):
        other = other._a if isinstance(other, FakeTensor) else other
        return FakeTensor(other - self._a, self.dtype, self.device)

    def __mul__(self, other):
        return self._bin(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._bin(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._bin(other, lambda a, b: a / b)

    def __neg__(self):
        return FakeTensor(-self._a, self.dtype, self.device)

    def __gt__(self, other):
        return self._bin(other, lambda a, b: a > b)

    def __eq__(self, other):
        return self._bin(other, lambda a, b: a == b)

    def __abs__(self):
        return FakeTensor(np.abs(self._a), self.dtype, self.device)

    def exp(self):
        return FakeTensor(np.exp(self._a), self.dtype, self.device)

    def log1p(self):
        return FakeTensor(np.log1p(self._a), self.dtype, self.device)
    def __float__(self):
        return float(self._a.sum())

    def __int__(self):
        return int(self._a.sum())

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return FakeTensor(float(self._a.sum()), self.dtype, self.device)
        out = self._a.sum(axis=dim, keepdims=keepdim)
        return FakeTensor(out, self.dtype, self.device)

    def max(self, dim=None, keepdim=False):
        out = self._a.max(axis=dim, keepdims=keepdim)
        return SimpleNamespace(values=FakeTensor(out, self.dtype, self.device))

    def masked_fill(self, mask, value):
        out = self._a.copy()
        raw = mask._a if isinstance(mask, FakeTensor) else mask
        out[np.asarray(raw, dtype=bool)] = value
        return FakeTensor(out, self.dtype, self.device)

    def bool(self):
        return FakeTensor(self._a.astype(bool), self.dtype, self.device)

    def squeeze(self, dim=None):
        if dim is None:
            return FakeTensor(self._a.squeeze(), self.dtype, self.device)
        return FakeTensor(np.squeeze(self._a, axis=dim), self.dtype, self.device)

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self._a, axis=dim), self.dtype, self.device)

    def flatten(self):
        return FakeTensor(self._a.flatten(), self.dtype, self.device)

    def to(self, dtype=None, device=None):
        return FakeTensor(self._a, dtype or self.dtype, device or self.device)

    def cpu(self):
        return self

    def detach(self):
        return FakeTensor(self._a.copy(), self.dtype, self.device)

    def clone(self):
        return FakeTensor(self._a.copy(), self.dtype, self.device)

    def tolist(self):
        return self._a.tolist()

    def clamp(self, low, high):
        return FakeTensor(np.clip(self._a, low, high), self.dtype, self.device)

    def backward(self):
        return None


class FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def synchronize(device=None):
        return None

    @staticmethod
    def max_memory_allocated(device=None):
        return 0

    @staticmethod
    def get_rng_state_all():
        return b""

    @staticmethod
    def set_rng_state_all(state):
        return None


class FakeAdamW:
    def __init__(self, grouped, weight_decay=0.0):
        self.param_groups = list(grouped)
        self.steps = 0
        self.zeroed = 0
        self.seen: list[list[object]] = []

    def zero_grad(self):
        self.zeroed += 1

    def step(self):
        self.steps += 1
        self.seen.append(
            [
                (p.grad._a.copy() if p.grad is not None else None)
                for group in self.param_groups
                for p in group["params"]
            ]
        )

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = int(state["steps"])


class FakeScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = int(state["steps"])
EVIDENCE_SEEN: list[object] = []


def _make_torch():
    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.cuda = FakeCuda
    torch.float32 = "float32"
    torch.int64 = "int64"
    torch.bool = "bool"
    torch.long = "long"

    @contextlib.contextmanager
    def _no_grad():
        yield

    torch.no_grad = _no_grad
    _thread_state = {"threads": 4, "interop": 2}
    torch.get_num_threads = lambda: int(_thread_state["threads"])
    torch.set_num_threads = lambda value: _thread_state.__setitem__("threads", int(value))
    torch.get_num_interop_threads = lambda: int(_thread_state["interop"])
    torch.set_num_interop_threads = lambda value: _thread_state.__setitem__("interop", int(value))
    torch.zeros = lambda shape, dtype=None, device="cpu": FakeTensor(
        np.zeros(shape), dtype, device
    )
    torch.ones = lambda shape, dtype=None, device="cpu": FakeTensor(
        np.ones(shape), dtype, device
    )
    torch.full = lambda shape, value, dtype=None, device="cpu": FakeTensor(
        np.full(shape, value), dtype, device
    )
    torch.zeros_like = lambda t: FakeTensor(np.zeros(t.shape), t.dtype, t.device)
    torch.ones_like = lambda t, dtype=None: (
        ONES_CALLS.append(tuple(t.shape)), FakeTensor(np.ones(t.shape), dtype or t.dtype, t.device)
    )[1]
    torch.full_like = lambda t, value: FakeTensor(
        np.full(t.shape, value), t.dtype, t.device
    )
    torch.tensor = lambda data, dtype=None, device="cpu": FakeTensor(
        np.asarray(data, dtype=np.float64), dtype, device
    )
    torch.cat = lambda parts, dim=1: FakeTensor(
        np.concatenate([p._a for p in parts], axis=dim)
    )
    torch.sigmoid = lambda t: FakeTensor(1.0 / (1.0 + np.exp(-t._a)))
    torch.logit = lambda t: FakeTensor(np.log(t._a / (1.0 - t._a)))
    torch.exp = lambda t: t.exp()
    torch.log1p = lambda t: t.log1p()
    torch.isfinite = lambda t: FakeTensor(np.isfinite(t._a))
    torch.where = lambda c, a, b: FakeTensor(
        np.where(
            c._a if isinstance(c, FakeTensor) else c,
            a._a if isinstance(a, FakeTensor) else a,
            b._a if isinstance(b, FakeTensor) else b,
        )
    )
    torch.get_rng_state = lambda: b"fakerng"
    torch.set_rng_state = lambda state: None
    torch.save = lambda obj, target: pickle.dump(
        obj, open(target, "wb") if isinstance(target, (str, Path)) else target
    )
    torch.load = lambda target, map_location=None, weights_only=True: pickle.load(
        open(target, "rb") if isinstance(target, (str, Path)) else target
    )
    nn = types.ModuleType("torch.nn")
    nn.utils = SimpleNamespace(clip_grad_norm_=lambda params, max_norm: None)
    optim = types.ModuleType("torch.optim")
    optim.AdamW = FakeAdamW
    scheduler_mod = types.ModuleType("torch.optim.lr_scheduler")
    scheduler_mod.LambdaLR = lambda optimizer, fn: FakeScheduler()
    optim.lr_scheduler = scheduler_mod
    torch.nn = nn
    torch.optim = optim
    sys.modules["torch.nn"] = nn
    sys.modules["torch.optim"] = optim
    sys.modules["torch.optim.lr_scheduler"] = scheduler_mod
    return torch


class FakeParam:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requires_grad = False
        self.grad = None
        self.device = "cpu"

    def requires_grad_(self, mode: bool):
        self.requires_grad = bool(mode)
        return self


class FakeModule:
    def __init__(self, training: bool = False) -> None:
        self.training = training

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def eval(self):
        self.training = False
        return self


def _wrapper_names() -> list[str]:
    names = ["psem_head.gru.weight", "psem_head.out.weight"]
    for layer in range(18):
        base = f"sortformer.transformer_encoder.layers.{layer}"
        names += [f"{base}.attn.weight", f"{base}.norm.weight"]
    names += [
        "sortformer.sortformer_modules.first_hidden_to_hidden.weight",
        "sortformer.sortformer_modules.single_hidden_to_spks.weight",
        "sortformer.encoder.front.weight",
        "sortformer.frontend_encoder.conv.weight",
    ]
    return names


class FakeWrapper:
    def __init__(self, step_frames: int = 125) -> None:
        self._params = [(n, FakeParam(n)) for n in _wrapper_names()]
        self._modules = {n.rsplit(".", 1)[0]: FakeModule() for n in _wrapper_names()}
        self._step_frames = step_frames
        self.calls: list[str] = []

    def named_parameters(self):
        return iter(self._params)

    def parameters(self):
        return (p for _, p in self._params)

    def named_modules(self):
        yield ("", FakeModule())
        for path, module in self._modules.items():
            yield (path, module)

    def eval(self):
        return self

    def state_dict(self):
        return {n: np.zeros(2) for n, _ in self._params}

    def load_state_dict(self, state, strict=True):
        self._loaded = dict(state)

    def _streaming_step(self, chunk, lengths, state, left_offset=0, right_offset=0):
        self.calls.append("step")
        k = int(chunk.shape[1])
        hidden = FakeTensor(np.full((1, k, 192), 0.01))
        logits = FakeTensor(np.full((1, k, 4), 0.02))
        probs = FakeTensor(np.full((1, k, 4), 0.25))
        return ({"s": FakeTensor(np.zeros(2))}, hidden, logits, probs, {})

    def native_sortformer_loss(self, evidence, targets, mask, roles, valid_lengths=None):
        EVIDENCE_SEEN.append((evidence, targets, mask, valid_lengths))
        probs = evidence.probabilities
        assert probs.shape == targets.shape
        assert len(probs.shape) == 3 and probs.shape[-1] == 4
        assert mask.shape == probs.shape[:2]
        assert mask.dtype == "bool"
        if valid_lengths is None:
            lengths = (int(mask._a.sum()),)
        else:
            lengths = tuple(int(v) for v in valid_lengths)
            assert len(lengths) == probs.shape[0]
        assert all(0 < v <= probs.shape[1] for v in lengths)
        flat_probs = np.asarray(probs._a).reshape(-1, 4)
        aimed = np.asarray(targets._a).reshape(-1, 4)
        return FakeTensor(float((flat_probs * aimed).sum()))



class FakeHead:
    def __init__(self, dim: int = 199) -> None:
        self._params = [FakeParam("gru.weight"), FakeParam("out.weight")]
        for p in self._params:
            p.requires_grad_(True)

    def to(self, device):
        return self

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def named_parameters(self):
        return iter([(p.name, p) for p in self._params])

    def state_dict(self):
        return {"head": np.zeros(2)}

    def load_state_dict(self, state, strict=True):
        self._loaded = dict(state)

    def __call__(self, features, state):
        n = int(features.shape[1])
        return (
            {
                "anchor_logit": FakeTensor(np.zeros((1, n))),
                "z_residual": FakeTensor(np.zeros((1, n))),
            },
            FakeTensor(np.zeros(4)),
        )

ONES_CALLS: list[tuple] = []
EVIDENCE_SEEN: list[object] = []
DECODE_THREADS: list[int] = []


class TrackedWave:
    live: list = []

    def __init__(self, frames: int) -> None:
        self._frames = frames
        TrackedWave.live.append(weakref.ref(self))

    @property
    def shape(self):
        return (1, self._frames * 1280)

    def __getitem__(self, idx):
        return TrackedWave(self._frames)

    def to(self, device):
        return self


def _target_bundle(source_id: str, frames: int) -> dict[str, object]:
    from experiments.psem_state_corrected_adaptation_gate import lifecycle
    from experiments.psem_state_corrected_adaptation_gate import multiplicity

    half = frames // 2
    episodes = [
        lifecycle.AnchorEpisode(f"{source_id}:A00001", "a", 0, half),
        lifecycle.AnchorEpisode(f"{source_id}:A00002", "b", half, frames),
    ]
    active = [(("a",) if (f // 120) % 2 == 0 else ("b",)) for f in range(frames)]
    valid = [True] * frames
    authority = lifecycle.build_source_authority(
        source_id, frames, episodes, active, valid
    )
    mult = multiplicity.build_multiplicity(frames, [(0.0, frames * 0.08)], valid)
    intervals: list[dict[str, object]] = []
    run_start = 0
    for frame in range(1, frames + 1):
        if frame == frames or active[frame] != active[run_start]:
            intervals.append(
                {
                    "start_sample": run_start * 1280,
                    "end_sample": frame * 1280,
                    "active_speakers": list(active[run_start]),
                    "masked": False,
                }
            )
            run_start = frame
    return {
        "source_id": source_id,
        "authority": authority,
        "multiplicity": mult,
        "episode_ids": [
            f"{source_id}:A00001" if f < half else f"{source_id}:A00002"
            for f in range(frames)
        ],
        "intervals": intervals,
        "num_frames": frames,
    }


def _parity_worker(value: int) -> int:
    return value * value + 1


def _fake_decode(snapshot, scores, threshold, confirmation_ms, **kwargs):
    return (str(snapshot.source_id), float(threshold), int(confirmation_ms))


def _fake_metrics(snapshot, events):
    _, threshold, _ = events
    clipped = max(0.0, min(1.0, float(threshold)))
    return {
        "active_speech_seconds": 3600.0,
        "reference_replacement_count": 10,
        "false_cut_count": int(round(clipped * 100.0)),
        "missed_replacement_count": int(round((1.0 - clipped) * 10.0)),
        "exclusive_other_contamination_seconds": clipped * 36.0,
    }

def _loader_steps(waveform: Any) -> list[Any]:
    frames = int(waveform.shape[1]) // 1280
    steps: list[Any] = []
    remaining = frames
    while remaining > 0:
        width = min(125, remaining)
        steps.append(
            (None, FakeTensor(np.zeros((1, width, 4))), FakeTensor([width]), 0, 0)
        )
        remaining -= width
    return steps


STUB_CALLS: list[str] = []


class TemporalResumeTest(unittest.TestCase):
    def _install(self, sessions):
        monkey = MonkeyPatch()
        torch = _make_torch()
        for name in (
            "torch.nn",
            "torch.optim",
            "torch.optim.lr_scheduler",
        ):
            monkey.setitem(sys.modules, name, sys.modules[name])

        def _stub(name, **attrs):
            module = types.ModuleType(name)
            module.__path__ = []
            for key, value in attrs.items():
                setattr(module, key, value)
            monkey.setitem(sys.modules, name, module)
            parts = name.split(".")
            for depth in range(len(parts) - 1, 0, -1):
                parent_name = ".".join(parts[:depth])
                if parent_name in sys.modules:
                    monkey.setattr(
                        sys.modules[parent_name],
                        parts[depth],
                        module,
                        raising=False,
                    )
                    break
            return module

        _stub("experiments.psem_sortformer_adaptation_depth")
        _stub(
            "experiments.psem_sortformer_adaptation_depth.nemo_adapter",
            load_pinned_sortformer=lambda *a, **k: (
                STUB_CALLS.append("load_model"),
                (FakeWrapper(), {"checkpoint_sha256": "0" * 64}),
            )[1],
            SortformerEvidence=SimpleNamespace,
        )
        _stub(
            "experiments.psem_frozen_ceiling_gate.evaluate_ceiling",
            decode_scores=_fake_decode,
            session_metrics=_fake_metrics,
        )
        frames_by_id = {
            sid: int(getattr(session, "_frames", 800))
            for sid, session in dict(sessions.get("train", {})).items()
        }
        _stub(
            "experiments.psem_sortformer_adaptation_depth.execution",
            load_source_waveform=lambda session, root: (
                DECODE_THREADS.append(__import__("threading").get_ident()),
                (
                    TrackedWave(frames_by_id.get(session.source_id, 800)),
                    frames_by_id.get(session.source_id, 800) * 1280,
                    0,
                ),
            )[1],
            load_scoring_sessions=lambda *a, **k: sessions.get("dev", {}),
        )
        _stub(
            "experiments.psem_sortformer_adaptation_depth.sampling",
            load_training_sessions=lambda *a, **k: sessions.get("train", {}),
        )
        _stub(
            "experiments.psem_state_corrected_adaptation_gate.head",
            ResidualPSEMHead=FakeHead,
        )
        _stub("experiments.psem_training_strategy_gate")
        _stub(
            "experiments.psem_training_strategy_gate.sampling",
            DEV_ROLE="PSEM-STRATEGY-DEV",
        )
        _stub("experiments.psem_frozen_ceiling_gate")
        _stub(
            "experiments.psem_frozen_ceiling_gate.build_ceiling_examples",
            load_sessions=lambda: sessions.get("snapshots", []),
        )
        _stub(
            "experiments.psem_frozen_ceiling_gate.experiment_support",
            simulate_gt_session=lambda *a, **k: None,
        )
        monkey.setitem(sys.modules, "torch", torch)
        self._monkey = monkey
        return torch

    def _remove(self):
        self._monkey.undo()

    def _patch_material(self, monkey, frames: int = 49000):
        def _fake_targets(sessions, rows_by_source, workers):
            return {
                sid: _target_bundle(sid, sessions[sid]._frames) for sid in sorted(sessions)
            }

        def _fake_single(simulate, source_id, labels, rows, num_frames):
            sessions = self._sessions
            return _target_bundle(source_id, sessions[source_id]._frames)

        def _fake_resolve(checkpoint, nemo_c, lock, corpus, reference, manifest, sha, device):
            population = material.resolve_sampling_population(manifest)
            return SimpleNamespace(
                checkpoint_path=checkpoint,
                rows_by_source=dict(population["rows_by_source"]),
            )

        def _fake_prepare(torch, wrapper, waveform):
            return {
                "loader": _loader_steps(waveform),
                "device": "cpu",
            }

        def _fake_init(torch, wrapper, batch_size=1):
            return {"s": FakeTensor(np.zeros(2))}

        def _fake_windows(torch, wrapper, waveform, window_frames=375, detach_between=True):
            n = int(waveform.shape[1]) // 1280
            return {
                "windows": [
                    {
                        "hidden": FakeTensor(np.full((1, n, 192), 0.01)),
                        "logits": FakeTensor(np.full((1, n, 4), 0.02)),
                        "probabilities": FakeTensor(
                            np.tile(np.array([[[0.1, 0.2, 0.3, 0.4]]]), (1, n, 1))
                        ),
                        "emitted_frames": n,
                        "steps": 1,
                    }
                ],
                "state_out": None,
                "boundary_steps": [],
            }

        def _fake_concat(torch, windows):
            return {
                "hidden": windows[0]["hidden"],
                "logits": windows[0]["logits"],
                "probabilities": windows[0]["probabilities"],
                "emitted_frames": windows[0]["emitted_frames"],
            }

        def _fake_dev(torch, wrapper, head, snapshot, session, corpus_root, device):
            n = 400
            return {
                "f0_raw": [0.1] * n,
                "cand_raw": [0.2] * n,
                "target": [1.0 if i % 3 else 0.0 for i in range(n)],
                "valid": [True] * n,
                "mapped_flags": [True] * n,
                "kept": list(range(n)),
                "unmapped_frames": [],
                "grid_frames": n,
                "infer_seconds": 0.25,
                "mapping_rows": [],
                "mapping_mapped": 1,
                "coverage": {"frames": n, "kept": n, "positive": 1, "negative": n - 1},
            }

        monkey.setattr(material, "resolve_material_inputs", _fake_resolve)
        monkey.setattr(material, "build_all_source_targets", _fake_targets)
        monkey.setattr(material, "build_source_targets", _fake_single)
        monkey.setattr(material, "prepare_streaming", _fake_prepare)
        monkey.setattr(material, "init_source_state", _fake_init)
        monkey.setattr(material, "run_adjacent_windows", _fake_windows)
        monkey.setattr(material, "concat_windows", _fake_concat)
        monkey.setattr(material, "infer_dev_raw_logits", _fake_dev)

    def _fixture(self, tmp: Path, frames: int = 49000, small_first: int | None = None):
        from experiments.psem_state_corrected_adaptation_gate import receipts

        checkpoint = tmp / "model.nemo"
        checkpoint.write_bytes(b"fake-checkpoint-bytes")
        nemo = tmp / "nemo"
        nemo.mkdir()
        lock = tmp / "lock.json"
        lock.write_text("{}", encoding="utf-8")
        corpus = tmp / "corpus"
        corpus.mkdir()
        reference = tmp / "reference"
        reference.mkdir()
        bundle = tmp / "bundle"
        bundle.mkdir()
        sources = [f"ami-{i:02d}" for i in range(9)] + [
            f"ali-{i:02d}" for i in range(9)
        ]
        manifest_path = tmp / "sampling.jsonl"
        rows: list[str] = []
        per_source = 4096 // len(sources)
        extra = 4096 - per_source * len(sources)
        for index, sid in enumerate(sources):
            count = per_source + (1 if index < extra else 0)
            corpus_name = "AMI" if sid.startswith("ami") else "AliMeeting"
            for k in range(count):
                rows.append(
                    json.dumps(
                        {
                            "split_role": "PSEM-STRATEGY-TRAIN",
                            "source_id": sid,
                            "window_start_sample": k * 480000,
                            "window_end_sample": (k + 1) * 480000,
                            "corpus": corpus_name,
                        }
                    )
                )
        manifest_path.write_text("\n".join(rows), encoding="utf-8")
        population = material.resolve_sampling_population(manifest_path)
        sessions = {}
        for sid in sources:
            session = SimpleNamespace(
                source_id=sid,
                role="PSEM-STRATEGY-TRAIN",
                labels=None,
                audio_ref=f"{sid}.wav",
                waveform_sha256=_h(sid),
            )
            session._frames = frames
            if small_first is not None and sid == "ali-00":
                session._frames = small_first
            sessions[sid] = session
        fit = sorted(s for s in sources if not s.endswith("08"))
        calib = sorted(s for s in sources if s.endswith("08"))
        bundle_manifest = {
            "artifact_role": "issue-121-stage-a-bundle",
            "version": 1,
            "nemo_sha256": hashlib.sha256(b"fake-checkpoint-bytes").hexdigest(),
            "sampling_sha256": str(population["sampling_sha256"]),
            "fit": fit,
            "calib": calib,
            "class_weights": {
                "replacement_positive_weight": 9.0,
                "anchor_positive_weight": 1.0,
            },
            "slice_sources": fit[:2],
            "ami_source": fit[0],
            "alimeeting_source": fit[9],
            "calib_sources": calib,
            "targets": {},
            "files": {},
            "target_frac": 0.12,
            "salt": "issue-121-train-calib-v1",
        }
        receipts.write_json(bundle / "stage_a_manifest.json", bundle_manifest)
        dev_session = SimpleNamespace(source_id="dev-ami-00", role="dev")
        dev_snapshot = SimpleNamespace(
            source_id="dev-ami-00", source_family="ami_mix_headset", role="dev"
        )
        self._sessions = sessions
        return {
            "checkpoint": checkpoint,
            "nemo": nemo,
            "lock": lock,
            "corpus": corpus,
            "reference": reference,
            "bundle": bundle,
            "manifest": manifest_path,
            "sessions": sessions,
            "sources": sources,
            "fit": fit,
            "calib": calib,
            "dev": {"dev-ami-00": dev_session},
            "snapshots": [dev_snapshot],
        }

    def _cli_args(self, tmp_path, config, fix, extra=()):
        return [
            "--arm", config.arm,
            "--seed", str(config.seed),
            "--root", str(tmp_path / "arms"),
            "--store", str(tmp_path / "store"),
            "--input-hash", config.input_hash,
            "--checkpoint-hash", config.checkpoint_hash,
            "--partition-hash", config.partition_hash,
            "--weights-hash", config.weights_hash,
            "--code-hash", config.code_hash,
            "--checkpoint", str(fix["checkpoint"]),
            "--bundle", str(fix["bundle"]),
            "--nemo-checkout", str(fix["nemo"]),
            "--dependency-lock", str(fix["lock"]),
            "--corpus-root", str(fix["corpus"]),
            "--reference-root", str(fix["reference"]),
            "--sampling-manifest", str(fix["manifest"]),
            "--device", "cpu",
            "--workers", "1",
            *extra,
        ]

    def test_carry_detach_reset_staleness(self):
        torch = _make_torch()
        carry = temporal_train.TemporalCarry()
        carry.reset("src-a", sortformer_state={"fifo": FakeTensor(np.zeros(2))})
        self.assertEqual(carry.source_id, "src-a")
        self.assertEqual(carry.chunks_carried, 0)
        first = carry.sortformer_state["fifo"]
        carry.detach(torch)
        self.assertEqual(carry.chunks_carried, 1)
        self.assertIsNot(carry.sortformer_state["fifo"], first)
        carry.mark_update()
        self.assertEqual(carry.stale_updates, 1)
        carry.reset("src-b")
        self.assertEqual(carry.source_id, "src-b")
        self.assertEqual(carry.chunks_carried, 0)
        self.assertIsNone(carry.gru_state)

    def _unit_prep(self, frames: int, valid_end: int | None = None):
        authority = SimpleNamespace(
            y_replace=[1.0 if (f // 120) % 2 else 0.0 for f in range(frames)],
            y_anchor=[1.0] * frames,
            valid=[True] * frames,
            num_frames=frames,
        )
        if valid_end is not None:
            authority.valid = [f < valid_end for f in range(frames)]
        return {
            "source_id": "src-a",
            "labels": None,
            "waveform": TrackedWave(frames),
            "num_frames": frames,
            "authority": authority,
            "multiplicity": [1] * frames,
            "episode_ids": ["src-a:A00001"] * frames,
            "chunk_sup": [
                {
                    "start": start,
                    "length": end - start,
                    "episode_ids": ["src-a:A00001"] * (end - start),
                    "arrival": FakeTensor(np.zeros((end - start, 4))),
                    "native_mask": FakeTensor(np.ones(end - start)),
                }
                for start, end in temporal_train.chunk_spans(frames)
            ],
        }

    def test_tail_contributes_at_actual_length(self):
        torch = self._install({"train": {}, "dev": {}, "snapshots": []})
        monkey = MonkeyPatch()
        monkey.setattr(
            material,
            "prepare_streaming",
            lambda torch, wrapper, waveform: {
                "loader": _loader_steps(waveform),
                "device": "cpu",
            },
        )
        monkey.setattr(
            material,
            "init_source_state",
            lambda torch, wrapper, batch_size=1: {"s": FakeTensor(np.zeros(2))},
        )
        try:
            ONES_CALLS.clear()
            EVIDENCE_SEEN.clear()
            wrapper, head = FakeWrapper(), FakeHead()
            opt, sched = FakeAdamW([{"params": []}]), FakeScheduler()
            accum = temporal_train.AccumState()
            opt.zero_grad()
            result = temporal_train.train_source(
                torch,
                wrapper,
                head,
                opt,
                sched,
                temporal_train.TemporalCarry(),
                accum,
                self._unit_prep(800),
                {"src-a:A00001": 0},
                {"replacement_positive_weight": 1.0, "anchor_positive_weight": 1.0},
                "cpu",
                2,
                None,
                True,
            )
            self.assertEqual(result["chunks"], 3)
            self.assertEqual(result["loss_chunks"], 3)
            self.assertEqual(result["optimizer_steps"], 2)
            self.assertEqual(result["pending"], 0)
            self.assertEqual(result["emitted_frames"], 800)
            self.assertEqual(ONES_CALLS, [])
            for evidence, _, _, _ in EVIDENCE_SEEN:
                self.assertFalse(hasattr(evidence, "slot_alive"))
        finally:
            monkey.undo()
            self._remove()

    def test_valid_frames_excluded_from_mask(self):
        torch = self._install({"train": {}, "dev": {}, "snapshots": []})
        monkey = MonkeyPatch()
        monkey.setattr(
            material,
            "prepare_streaming",
            lambda torch, wrapper, waveform: {
                "loader": _loader_steps(waveform),
                "device": "cpu",
            },
        )
        monkey.setattr(
            material,
            "init_source_state",
            lambda torch, wrapper, batch_size=1: {"s": FakeTensor(np.zeros(2))},
        )
        try:
            wrapper, head = FakeWrapper(), FakeHead()
            opt, sched = FakeAdamW([{"params": []}]), FakeScheduler()
            accum = temporal_train.AccumState()
            opt.zero_grad()
            result = temporal_train.train_source(
                torch,
                wrapper,
                head,
                opt,
                sched,
                temporal_train.TemporalCarry(),
                accum,
                self._unit_prep(800, valid_end=0),
                {"src-a:A00001": 0},
                {"replacement_positive_weight": 1.0, "anchor_positive_weight": 1.0},
                "cpu",
                2,
            )
            self.assertEqual(result["loss_chunks"], 0)
            self.assertEqual(result["empty_chunks"], 3)
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual(accum.pending, 0)
        finally:
            monkey.undo()
            self._remove()

    def test_weighted_mean_exact_magnitudes(self):
        import math

        torch = _make_torch()
        weights = {"replacement_positive_weight": 2.0, "anchor_positive_weight": 3.0}

        def _bce(z, y, p):
            s = 1.0 / (1.0 + math.exp(-z))
            return -(p * y * math.log(s) + (1.0 - y) * math.log(1.0 - s))

        product = FakeTensor(np.array([[0.5, -1.0, 2.0, 0.0]]))
        y_replace = FakeTensor(np.array([[1.0, 0.0, 1.0, 0.0]]))
        anchor = FakeTensor(np.array([[-0.5, 0.25, 1.5, -2.0]]))
        y_anchor = FakeTensor(np.array([[0.0, 1.0, 0.0, 1.0]]))
        mult = FakeTensor(np.array([[1.0, 2.0, 0.0, 3.0]]))
        out = temporal_train.temporal_chunk_loss(
            torch, product, y_replace, anchor, y_anchor, mult, weights, None
        )
        self.assertFalse(out["empty"])
        rep = [_bce(z, y, 2.0) for z, y in zip([0.5, -1.0, 2.0, 0.0], [1.0, 0.0, 1.0, 0.0])]
        anc = [_bce(z, y, 3.0) for z, y in zip([-0.5, 0.25, 1.5, -2.0], [0.0, 1.0, 0.0, 1.0])]
        expected_rep = (rep[0] * 1.0 + rep[1] * 2.0 + rep[3] * 3.0) / 6.0
        expected_anc = (anc[0] * 1.0 + anc[1] * 2.0 + anc[3] * 3.0) / 6.0
        self.assertAlmostEqual(float(out["replacement"]), expected_rep, places=9)
        self.assertAlmostEqual(float(out["anchor"]), expected_anc, places=9)
        self.assertAlmostEqual(
            float(out["loss"]), expected_rep + 0.5 * expected_anc, places=9
        )
        doubled = FakeTensor(np.array([[2.0, 4.0, 0.0, 6.0]]))
        out2 = temporal_train.temporal_chunk_loss(
            torch, product, y_replace, anchor, y_anchor, doubled, weights, None
        )
        self.assertAlmostEqual(float(out2["replacement"]), expected_rep, places=9)
        self.assertAlmostEqual(float(out2["loss"]), float(out["loss"]), places=9)
        zero = FakeTensor(np.array([[0.0, 0.0, 0.0, 0.0]]))
        out3 = temporal_train.temporal_chunk_loss(
            torch, product, y_replace, anchor, y_anchor, zero, weights, None
        )
        self.assertTrue(out3["empty"])
        self.assertEqual(float(out3["loss"]), 0.0)
        native = FakeTensor(0.5)
        out4 = temporal_train.temporal_chunk_loss(
            torch, product, y_replace, anchor, y_anchor, mult, weights, native
        )
        self.assertAlmostEqual(
            float(out4["loss"]), expected_rep + 0.5 * expected_anc + 0.25, places=9
        )
        bad = FakeTensor(np.array([[float("nan"), 0.0, 0.0, 0.0]]))
        mult_one = FakeTensor(np.array([[1.0, 0.0, 0.0, 0.0]]))
        with self.assertRaises(temporal_train.TemporalArmError):
            temporal_train.temporal_chunk_loss(
                torch, bad, y_replace, anchor, y_anchor, mult_one, weights, None
            )

    def test_native_expansion_exact_double(self):
        torch = self._install({"train": {}, "dev": {}, "snapshots": []})
        try:
            EVIDENCE_SEEN.clear()
            wrapper, head = FakeWrapper(), FakeHead()
            prep = {
                "source_id": "src-a",
                "labels": None,
                "waveform": TrackedWave(40),
                "num_frames": 40,
                "authority": SimpleNamespace(
                    y_replace=[0.0] * 40,
                    y_anchor=[1.0] * 40,
                    valid=[True] * 40,
                    num_frames=40,
                    episodes=(),
                ),
                "multiplicity": [1] * 40,
                "episode_ids": ["e1"] * 40,
            }
            segment = {
                "hidden": FakeTensor(np.full((1, 10, 192), 0.01)),
                "logits": FakeTensor(np.full((1, 10, 4), 0.02)),
                "probabilities": FakeTensor(np.full((1, 10, 4), 0.25)),
            }
            entry = {
                "start": 30,
                "length": 10,
                "episode_ids": ["e1"] * 10,
                "arrival": FakeTensor(
                    np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (10, 1))
                ),
                "native_mask": FakeTensor(np.ones(10)),
            }
            weights = {"replacement_positive_weight": 1.0, "anchor_positive_weight": 1.0}
            carry = temporal_train.TemporalCarry()
            out1 = temporal_train.train_chunk(
                torch, wrapper, head, carry, prep, {"e1": 0}, segment, entry,
                weights, "cpu",
            )
            self.assertFalse(out1["empty"])
            first = float(out1["native"])
            (evidence1, targets1, mask1, lengths1), = EVIDENCE_SEEN[-1:]
            self.assertEqual(evidence1.probabilities.shape, (1, 10, 4))
            self.assertEqual(targets1.shape, (1, 10, 4))
            self.assertEqual(mask1.shape, (1, 10))
            self.assertEqual(tuple(lengths1), (10,))
            prep2 = dict(prep, multiplicity=[1] * 35 + [2] + [1] * 4)
            EVIDENCE_SEEN.clear()
            out2 = temporal_train.train_chunk(
                torch, wrapper, head, carry, prep2, {"e1": 0}, segment, entry,
                weights, "cpu",
            )
            (evidence2, targets2, mask2, lengths2), = EVIDENCE_SEEN[-1:]
            self.assertEqual(evidence2.probabilities.shape, (1, 11, 4))
            self.assertEqual(targets2.shape, (1, 11, 4))
            self.assertEqual(mask2.shape, (1, 11))
            self.assertEqual(tuple(lengths2), (11,))
            self.assertAlmostEqual(float(out2["native"]) - first, 0.25 / 2.0, places=9)
        finally:
            self._remove()

    def test_source_start_warmup_and_fixed_slots(self):
        torch = _make_torch()
        entry = _target_bundle("src-a", 800)
        built = temporal_train.build_full_source_supervision(torch, entry)
        self.assertEqual(len(built["chunks"]), 3)
        first_mask = built["chunks"][0]["native_mask"]
        self.assertFalse(bool(first_mask[:25]._a.any()))
        self.assertTrue(bool(first_mask[25:]._a.all()))
        for chunk in built["chunks"][1:]:
            self.assertTrue(bool(chunk["native_mask"]._a.all()))
        first_arrival = built["chunks"][0]["arrival"]
        self.assertEqual(first_arrival[0].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(first_arrival[25].tolist(), [1.0, 0.0, 0.0, 0.0])
        second_arrival = built["chunks"][1]["arrival"]
        self.assertEqual(second_arrival[0].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(second_arrival[105].tolist(), [1.0, 0.0, 0.0, 0.0])
        from experiments.psem_state_corrected_adaptation_gate import lifecycle

        speakers = ["a", "b", "c", "d", "e"]
        per_frame = [speakers[f % 5] for f in range(800)]
        many_intervals = [
            {
                "start_sample": f * 1280,
                "end_sample": (f + 1) * 1280,
                "active_speakers": [per_frame[f]],
                "masked": False,
            }
            for f in range(800)
        ]
        many_active = [(s,) for s in per_frame]
        many_authority = lifecycle.build_source_authority(
            "src-a",
            800,
            [lifecycle.AnchorEpisode("src-a:A00001", "a", 0, 800)],
            many_active,
            [True] * 800,
        )
        crowded = {
            "source_id": "src-a",
            "num_frames": 800,
            "authority": many_authority,
            "intervals": many_intervals,
        }
        with self.assertRaises(temporal_train.TemporalArmError):
            temporal_train.build_full_source_supervision(torch, crowded)

    def test_global_accumulation_bit_identity_across_resume(self):
        torch = self._install({"train": {}, "dev": {}, "snapshots": []})
        monkey = MonkeyPatch()
        monkey.setattr(
            material,
            "prepare_streaming",
            lambda torch, wrapper, waveform: {
                "loader": _loader_steps(waveform),
                "device": "cpu",
            },
        )
        monkey.setattr(
            material,
            "init_source_state",
            lambda torch, wrapper, batch_size=1: {"s": FakeTensor(np.zeros(2))},
        )
        try:
            weights = {"replacement_positive_weight": 1.0, "anchor_positive_weight": 1.0}

            def _run(sources, accum, wrapper, head, opt, sched):
                carry = temporal_train.TemporalCarry()
                totals = []
                for sid, frames in sources:
                    prep = self._unit_prep(frames)
                    prep["source_id"] = sid
                    prep["episode_ids"] = [f"{sid}:A00001"] * frames
                    for entry in prep["chunk_sup"]:
                        entry["episode_ids"] = [f"{sid}:A00001"] * entry["length"]
                    totals.append(
                        temporal_train.train_source(
                            torch, wrapper, head, opt, sched, carry, accum,
                            prep, {f"{sid}:A00001": 0}, weights, "cpu", 16,
                        )
                    )
                return totals
            wrapper_a, head_a = FakeWrapper(), FakeHead()
            opt_a, sched_a = FakeAdamW([{"params": []}]), FakeScheduler()
            accum_a = temporal_train.AccumState()
            opt_a.zero_grad()
            totals_a = _run(
                [("s-a", 8000), ("s-b", 8000)], accum_a, wrapper_a, head_a, opt_a, sched_a
            )
            after_a_steps = totals_a[0]["optimizer_steps"]
            self.assertEqual(after_a_steps, 1)
            self.assertEqual(totals_a[0]["pending"], 6)
            wrapper_b, head_b = FakeWrapper(), FakeHead()
            opt_b, sched_b = FakeAdamW([{"params": []}]), FakeScheduler()
            accum_b = temporal_train.AccumState()
            opt_b.zero_grad()
            carry_b = temporal_train.TemporalCarry()
            prep_a = self._unit_prep(8000)
            prep_a["source_id"] = "s-a"
            prep_a["episode_ids"] = ["s-a:A00001"] * 8000
            for entry in prep_a["chunk_sup"]:
                entry["episode_ids"] = ["s-a:A00001"] * entry["length"]
            temporal_train.train_source(
                torch, wrapper_b, head_b, opt_b, sched_b, carry_b, accum_b,
                prep_a, {"s-a:A00001": 0}, weights, "cpu", 16,
            )
            with tempfile.TemporaryDirectory() as tmp:
                blobs = temporal_train.snapshot_blobs(
                    torch, wrapper_b, head_b, opt_b, sched_b, accum_b
                )
                run_dir = Path(tmp) / "run"
                arm_runtime.save_source_checkpoint(
                    run_dir,
                    "s-a",
                    [],
                    {"arm": "R-T2-SC", "seed": 7301},
                    blobs,
                )
                stored = arm_runtime.load_source_checkpoint(
                    run_dir, {"arm": "R-T2-SC", "seed": 7301}
                )
                wrapper_c, head_c = FakeWrapper(), FakeHead()
                opt_c, sched_c = FakeAdamW([{"params": []}]), FakeScheduler()
                accum_c = temporal_train.AccumState()
                temporal_train.restore_blobs(
                    torch, wrapper_c, head_c, opt_c, sched_c,
                    accum_c, stored["blobs"]["s-a"], "cpu",
                )
            self.assertEqual(accum_c.pending, accum_b.pending)
            self.assertEqual(opt_c.steps, opt_b.steps)
            self.assertEqual(sched_c.steps, sched_b.steps)
            totals_c = _run(
                [("s-b", 8000)], accum_c, wrapper_c, head_c, opt_c, sched_c
            )
            self.assertEqual(accum_c.optimizer_steps, accum_a.optimizer_steps)
            self.assertEqual(opt_c.steps, opt_a.steps)
            self.assertEqual(accum_c.pending, accum_a.pending)
            self.assertEqual(accum_c.loss_chunks, accum_a.loss_chunks)
            self.assertEqual(accum_c.empty_chunks, accum_a.empty_chunks)
            self.assertEqual(
                totals_c[0]["optimizer_steps"],
                totals_a[1]["optimizer_steps"],
            )
        finally:
            monkey.undo()
            self._remove()

    def test_final_partial_grads_equal_exact_mean(self):
        torch = _make_torch()
        params = [FakeParam("a"), FakeParam("b")]
        for param in params:
            param.requires_grad_(True)
            param.grad = FakeTensor(np.full(4, 8.0))
        opt = FakeAdamW([{"params": params}])
        sched = FakeScheduler()
        carry = temporal_train.TemporalCarry()
        accum = temporal_train.AccumState()
        accum.pending = 5
        temporal_train.apply_optimizer_update(torch, opt, sched, carry, accum, 16)
        self.assertEqual(opt.seen[0][0].tolist(), [1.6, 1.6, 1.6, 1.6])
        self.assertEqual(opt.seen[0][1].tolist(), [1.6, 1.6, 1.6, 1.6])
        self.assertEqual(accum.pending, 0)
        self.assertEqual(accum.optimizer_steps, 1)
        for param in params:
            param.grad = FakeTensor(np.full(4, 8.0))
        accum.pending = 16
        temporal_train.apply_optimizer_update(torch, opt, sched, carry, accum, 16)
        self.assertEqual(opt.seen[1][0].tolist(), [0.5, 0.5, 0.5, 0.5])

    def test_e2e_flush_matches_n_item_mean(self):
        torch = self._install({"train": {}, "dev": {}, "snapshots": []})
        monkey = MonkeyPatch()
        monkey.setattr(
            material,
            "prepare_streaming",
            lambda torch, wrapper, waveform: {
                "loader": _loader_steps(waveform),
                "device": "cpu",
            },
        )
        monkey.setattr(
            material,
            "init_source_state",
            lambda torch, wrapper, batch_size=1: {"s": FakeTensor(np.zeros(2))},
        )
        try:
            wrapper, head = FakeWrapper(), FakeHead()
            for _, param in list(wrapper.named_parameters())[:1]:
                param.requires_grad_(True)
                param.grad = FakeTensor(np.full(2, 6.0))
            opt = FakeAdamW(
                [{"params": [p for _, p in list(wrapper.named_parameters())[:1]]}]
            )
            sched = FakeScheduler()
            accum = temporal_train.AccumState()
            opt.zero_grad()
            for _, param in list(wrapper.named_parameters())[:1]:
                param.grad = FakeTensor(np.full(2, 6.0))
            result = temporal_train.train_source(
                torch,
                wrapper,
                head,
                opt,
                sched,
                temporal_train.TemporalCarry(),
                accum,
                self._unit_prep(800),
                {"src-a:A00001": 0},
                {"replacement_positive_weight": 1.0, "anchor_positive_weight": 1.0},
                "cpu",
                16,
                None,
                True,
            )
            self.assertEqual(result["loss_chunks"], 3)
            self.assertEqual(result["optimizer_steps"], 1)
            stepped = [row for row in opt.seen if row[0] is not None]
            self.assertEqual(len(stepped), 1)
            self.assertEqual(stepped[0][0].tolist(), [2.0, 2.0])
        finally:
            monkey.undo()
            self._remove()

    def test_snapshot_restores_grads_and_rng(self):
        import random

        torch = _make_torch()
        random.seed(7301)
        np.random.seed(7301)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        wrapper, head = FakeWrapper(), FakeHead()
        opt, sched = FakeAdamW([{"params": []}]), FakeScheduler()
        sched.steps = 7
        accum = temporal_train.AccumState()
        accum.pending = 5
        accum.optimizer_steps = 3
        accum.loss_chunks = 11
        accum.empty_chunks = 2
        named = list(wrapper.named_parameters())
        for _, param in named[:2]:
            param.grad = FakeTensor(np.full(2, 3.0))
        named[0][1].device = "cuda"
        head_param = next(iter(head.named_parameters()))[1]
        head_param.grad = FakeTensor(np.full(2, 4.0))
        blobs = temporal_train.snapshot_blobs(torch, wrapper, head, opt, sched, accum)
        self.assertEqual(sorted(blobs), ["model", "optimizer", "rng", "scheduler"])
        random.seed(1)
        np.random.seed(1)
        wrapper2, head2 = FakeWrapper(), FakeHead()
        opt2, sched2 = FakeAdamW([{"params": []}]), FakeScheduler()
        accum2 = temporal_train.AccumState()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.pt"
            path.write_bytes(blobs["model"])
            loaded = torch.load(str(path), map_location="cpu", weights_only=False)
            self.assertEqual(loaded["pending"], 5)
            self.assertEqual(len(loaded["pending_grads"]), 3)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            arm_runtime.save_source_checkpoint(
                run_dir, "s-a", [], {"arm": "R-T2-SC", "seed": 7301}, blobs
            )
            stored = arm_runtime.load_source_checkpoint(
                run_dir, {"arm": "R-T2-SC", "seed": 7301}
            )
            list(wrapper2.named_parameters())[0][1].device = "cuda"
            temporal_train.restore_blobs(
                torch, wrapper2, head2, opt2, sched2, accum2, stored["blobs"]["s-a"], "cpu"
            )
        self.assertEqual(accum2.pending, 5)
        self.assertEqual(accum2.optimizer_steps, 3)
        self.assertEqual(accum2.loss_chunks, 11)
        self.assertEqual(accum2.empty_chunks, 2)
        restored_params = list(wrapper2.named_parameters())
        restored = [
            p.grad._a.tolist() for _, p in restored_params[:2]
        ]
        self.assertEqual(restored, [[3.0, 3.0], [3.0, 3.0]])
        self.assertEqual(restored_params[0][1].grad.device, "cuda")
        self.assertEqual(restored_params[1][1].grad.device, "cpu")
        self.assertEqual(random.getstate()[1], python_state[1])
        current = np.random.get_state()
        self.assertEqual(current[0], numpy_state[0])
        self.assertTrue(bool((current[1] == numpy_state[1]).all()))

    def test_prepare_builds_supervision_once(self):
        torch = self._install({"train": {}, "dev": {}, "snapshots": []})
        monkey = MonkeyPatch()
        monkey.setattr(
            material,
            "prepare_streaming",
            lambda torch, wrapper, waveform: {
                "loader": _loader_steps(waveform),
                "device": "cpu",
            },
        )
        monkey.setattr(
            material,
            "init_source_state",
            lambda torch, wrapper, batch_size=1: {"s": FakeTensor(np.zeros(2))},
        )
        calls: list[str] = []
        real_builder = temporal_train.build_full_source_supervision

        def _counting(torch, entry):
            calls.append(entry["source_id"])
            return real_builder(torch, entry)

        monkey.setattr(temporal_train, "build_full_source_supervision", _counting)
        try:
            session = SimpleNamespace(
                source_id="src-a", labels=None, audio_ref="x", waveform_sha256="y"
            )
            session._frames = 800
            prep = temporal_train.prepare_source(
                torch, session, _target_bundle("src-a", 800), Path("."), "cpu"
            )
            self.assertEqual(calls, ["src-a"])
            self.assertEqual(len(prep["chunk_sup"]), 3)
            spanning = prep["chunk_sup"][1]["episode_ids"]
            self.assertIn("src-a:A00001", spanning)
            self.assertIn("src-a:A00002", spanning)
            monkey.setattr(
                temporal_train,
                "build_full_source_supervision",
                lambda torch, entry: (_ for _ in ()).throw(AssertionError("rebuilt")),
            )
            wrapper, head = FakeWrapper(), FakeHead()
            opt, sched = FakeAdamW([{"params": []}]), FakeScheduler()
            result = temporal_train.train_source(
                torch,
                wrapper,
                head,
                opt,
                sched,
                temporal_train.TemporalCarry(),
                temporal_train.AccumState(),
                prep,
                {"src-a:A00001": 0, "src-a:A00002": 1},
                {"replacement_positive_weight": 1.0, "anchor_positive_weight": 1.0},
                "cpu",
                16,
            )
            self.assertEqual(result["chunks"], 3)
        finally:
            monkey.undo()
            self._remove()

    def test_merge_training_sources_prefers_current(self):
        prior = {"s-a": {"optimizer_steps": 1}, "s-b": {"optimizer_steps": 0}}
        current = {"s-b": {"optimizer_steps": 2}, "s-c": {"optimizer_steps": 0}}
        merged = temporal_train.merge_training_sources(prior, current)
        self.assertEqual(
            merged,
            {
                "s-a": {"optimizer_steps": 1},
                "s-b": {"optimizer_steps": 2},
                "s-c": {"optimizer_steps": 0},
            },
        )
        self.assertEqual(temporal_train.merge_training_sources({}, current), current)

    def test_load_prior_training_sources_missing_and_corrupt(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            self.assertEqual(temporal_train.load_prior_training_sources(run_dir), {})
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "training_metrics.json").write_text("{truncated", encoding="utf-8")
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.load_prior_training_sources(run_dir)
            (run_dir / "training_metrics.json").write_text(
                json.dumps({"sources": ["s-a"]}), encoding="utf-8"
            )
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.load_prior_training_sources(run_dir)
            (run_dir / "training_metrics.json").write_text(
                json.dumps({"sources": {"s-a": {"optimizer_steps": 1}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                temporal_train.load_prior_training_sources(run_dir),
                {"s-a": {"optimizer_steps": 1}},
            )

    def _backfill_fixture(self, tmp_path):
        session = SimpleNamespace(
            source_id="s-a",
            labels=None,
            audio_ref="s-a.wav",
            waveform_sha256=_h("wave-a"),
        )
        session._frames = 800
        manifest = {"sampling_sha256": _h("sampling"), "targets": {}, "files": {}}
        sessions = {"s-a": session}
        rows = {"s-a": [{"window_start_sample": 0, "window_end_sample": 480000}]}
        return sessions, rows, manifest

    def test_durable_backfill_persist_reuse_and_fail_closed(self):
        import tempfile

        torch = _make_torch()
        monkey = MonkeyPatch()
        monkey.setattr(
            material,
            "build_source_targets",
            lambda simulate, sid, labels, rows, num_frames: _target_bundle(sid, int(num_frames)),
        )
        monkey.setattr(
            temporal_train,
            "load_source_waveform",
            lambda torch, session, root, device: {
                "waveform": TrackedWave(800),
                "num_frames": 800,
                "tail_excluded": 0,
            },
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                run_dir = tmp_path / "run"
                sessions, rows, manifest = self._backfill_fixture(tmp_path)
                entries, binding = temporal_train.resolve_durable_targets(
                    torch, run_dir, tmp_path / "bundle", manifest, sessions, rows,
                    tmp_path, "cpu", ["s-a"],
                )
                self.assertEqual(entries["s-a"]["num_frames"], 800)
                self.assertTrue(binding["s-a"]["backfilled"])
                target_file = run_dir / "targets" / "s-a.json"
                self.assertTrue(target_file.is_file())
                manifest_file = run_dir / "targets" / "targets_manifest.json"
                self.assertTrue(manifest_file.is_file())
                first = list(entries["s-a"]["authority"].y_replace)
                monkey.setattr(
                    material,
                    "build_source_targets",
                    lambda *a, **k: (_ for _ in ()).throw(AssertionError("rebuilt")),
                )
                rerun, binding2 = temporal_train.resolve_durable_targets(
                    torch, run_dir, tmp_path / "bundle", manifest, sessions, rows,
                    tmp_path, "cpu", ["s-a"],
                )
                self.assertEqual(list(rerun["s-a"]["authority"].y_replace), list(first))
                self.assertEqual(binding2["s-a"]["sha256"], binding["s-a"]["sha256"])
                target_file.write_bytes(target_file.read_bytes()[:64])
                with self.assertRaises(temporal_train.TemporalArmError):
                    temporal_train.resolve_durable_targets(
                        torch, run_dir, tmp_path / "bundle", manifest, sessions, rows,
                        tmp_path, "cpu", ["s-a"],
                    )
        finally:
            monkey.undo()

    def test_durable_backfill_waveform_mismatch_fails_closed(self):
        import tempfile

        torch = _make_torch()
        monkey = MonkeyPatch()
        monkey.setattr(
            material,
            "build_source_targets",
            lambda simulate, sid, labels, rows, num_frames: _target_bundle(sid, int(num_frames)),
        )
        monkey.setattr(
            temporal_train,
            "load_source_waveform",
            lambda torch, session, root, device: {
                "waveform": TrackedWave(800),
                "num_frames": 800,
                "tail_excluded": 0,
            },
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                run_dir = tmp_path / "run"
                sessions, rows, manifest = self._backfill_fixture(tmp_path)
                temporal_train.resolve_durable_targets(
                    torch, run_dir, tmp_path / "bundle", manifest, sessions, rows,
                    tmp_path, "cpu", ["s-a"],
                )
                sessions["s-a"].waveform_sha256 = _h("other-wave")
                with self.assertRaises(temporal_train.TemporalArmError):
                    temporal_train.resolve_durable_targets(
                        torch, run_dir, tmp_path / "bundle", manifest, sessions, rows,
                        tmp_path, "cpu", ["s-a"],
                    )
        finally:
            monkey.undo()

    def test_feature_geometry_199_enforced(self):
        torch = _make_torch()
        wrapper, head = FakeWrapper(), FakeHead()
        prep = self._unit_prep(375)
        slot_of = {"src-a:A00001": 0}
        weights = {"replacement_positive_weight": 1.0, "anchor_positive_weight": 1.0}
        entry = dict(prep["chunk_sup"][0])
        good = {
            "hidden": FakeTensor(np.full((1, 375, 192), 0.01)),
            "logits": FakeTensor(np.full((1, 375, 4), 0.02)),
            "probabilities": FakeTensor(np.full((1, 375, 4), 0.25)),
        }
        carry = temporal_train.TemporalCarry()
        out = temporal_train.train_chunk(
            torch, wrapper, head, carry, prep, slot_of, good, entry, weights, "cpu"
        )
        self.assertFalse(out["empty"])
        bad = dict(good, hidden=FakeTensor(np.full((1, 375, 191), 0.01)))
        with self.assertRaises(temporal_train.TemporalArmError) as ctx:
            temporal_train.train_chunk(
                torch, wrapper, head, temporal_train.TemporalCarry(), prep, slot_of,
                bad, entry, weights, "cpu",
            )
        self.assertIn("199=192+4+1+1+1", str(ctx.exception))

    def test_streaming_short_aborts_before_first_backward(self):
        torch = self._install({"train": {}, "dev": {}, "snapshots": []})
        monkey = MonkeyPatch()
        monkey.setattr(
            material,
            "prepare_streaming",
            lambda torch, wrapper, waveform: {
                "loader": [
                    (None, FakeTensor(np.zeros((1, 125, 4))), FakeTensor([125]), 0, 0)
                    for _ in range(5)
                ],
                "device": "cpu",
            },
        )
        monkey.setattr(
            material,
            "init_source_state",
            lambda torch, wrapper, batch_size=1: {"s": FakeTensor(np.zeros(2))},
        )
        try:
            wrapper, head = FakeWrapper(), FakeHead()
            opt, sched = FakeAdamW([{"params": []}]), FakeScheduler()
            accum = temporal_train.AccumState()
            opt.zero_grad()
            with self.assertRaises(temporal_train.TemporalArmError) as ctx:
                temporal_train.train_source(
                    torch,
                    wrapper,
                    head,
                    opt,
                    sched,
                    temporal_train.TemporalCarry(),
                    accum,
                    self._unit_prep(800),
                    {"src-a:A00001": 0},
                    {"replacement_positive_weight": 1.0, "anchor_positive_weight": 1.0},
                    "cpu",
                    2,
                )
            self.assertIn("frame count differs", str(ctx.exception))
            self.assertEqual(opt.steps, 0)
            self.assertEqual(accum.loss_chunks, 0)
            self.assertEqual(accum.pending, 0)
        finally:
            monkey.undo()
            self._remove()

    def test_streaming_long_aborts_before_first_backward(self):
        torch = self._install({"train": {}, "dev": {}, "snapshots": []})
        monkey = MonkeyPatch()
        monkey.setattr(
            material,
            "prepare_streaming",
            lambda torch, wrapper, waveform: {
                "loader": [
                    (None, FakeTensor(np.zeros((1, 125, 4))), FakeTensor([125]), 0, 0)
                    for _ in range(8)
                ],
                "device": "cpu",
            },
        )
        monkey.setattr(
            material,
            "init_source_state",
            lambda torch, wrapper, batch_size=1: {"s": FakeTensor(np.zeros(2))},
        )
        try:
            wrapper, head = FakeWrapper(), FakeHead()
            opt, sched = FakeAdamW([{"params": []}]), FakeScheduler()
            accum = temporal_train.AccumState()
            opt.zero_grad()
            with self.assertRaises(temporal_train.TemporalArmError):
                temporal_train.train_source(
                    torch,
                    wrapper,
                    head,
                    opt,
                    sched,
                    temporal_train.TemporalCarry(),
                    accum,
                    self._unit_prep(800),
                    {"src-a:A00001": 0},
                    {"replacement_positive_weight": 1.0, "anchor_positive_weight": 1.0},
                    "cpu",
                    2,
                )
            self.assertEqual(opt.steps, 0)
            self.assertEqual(accum.loss_chunks, 0)
        finally:
            monkey.undo()
            self._remove()

    def test_mapping_freeze_bounds_waveform_residency(self):
        torch = self._install({"train": {}, "dev": {}, "snapshots": []})
        monkey = MonkeyPatch()
        TrackedWave.live.clear()

        def _windows(torch, wrapper, waveform, window_frames=375, detach_between=True):
            n = 2000
            return {
                "windows": [
                    {
                        "hidden": FakeTensor(np.full((1, n, 192), 0.01)),
                        "logits": FakeTensor(np.full((1, n, 4), 0.02)),
                        "probabilities": FakeTensor(
                            np.tile(np.array([[[0.1, 0.2, 0.3, 0.4]]]), (1, n, 1))
                        ),
                        "emitted_frames": n,
                        "steps": 1,
                    }
                ],
                "state_out": None,
                "boundary_steps": [],
            }

        def _concat(torch, windows):
            return {
                "hidden": windows[0]["hidden"],
                "logits": windows[0]["logits"],
                "probabilities": windows[0]["probabilities"],
                "emitted_frames": windows[0]["emitted_frames"],
            }

        monkey.setattr(material, "run_adjacent_windows", _windows)
        monkey.setattr(material, "concat_windows", _concat)
        try:
            sessions = {}
            entries = {}
            for sid in ("s-a", "s-b", "s-c"):
                session = SimpleNamespace(source_id=sid)
                session._frames = 2000
                sessions[sid] = session
                entries[sid] = _target_bundle(sid, 2000)
            sessions_exec = dict(sessions)
            import types as _types

            execution = sys.modules[
                "experiments.psem_sortformer_adaptation_depth.execution"
            ]
            monkey.setattr(
                execution,
                "load_source_waveform",
                lambda session, root: (TrackedWave(session._frames), session._frames * 1280, 0),
            )
            wrapper = FakeWrapper()
            mapping = temporal_train.freeze_pass_mapping(
                torch,
                arm_runtime.ARM_R_T2_SC,
                7301,
                ["s-a", "s-b", "s-c"],
                sessions_exec,
                entries,
                Path("."),
                "cpu",
                wrapper,
                1,
            )
            gc.collect()
            alive = sum(1 for ref in TrackedWave.live if ref() is not None)
            self.assertLessEqual(alive, 1)
            self.assertEqual(sorted(mapping.source_to_mapping), ["s-a", "s-b", "s-c"])
        finally:
            monkey.undo()
            self._remove()

    def test_stored_mapping_reused_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = arm_runtime.config_from_dict(
                {
                    "arm": arm_runtime.ARM_R_T2_SC,
                    "seed": 7301,
                    "root": Path(tmp) / "arms",
                    "input_hash": _h("input"),
                    "checkpoint_hash": _h("ckpt"),
                    "partition_hash": _h("part"),
                    "weights_hash": _h("w"),
                    "code_hash": _h("code"),
                }
            )
            mapping = temporal_train.PassMapping(
                source_to_mapping={"s-a": {"slot_of": {"e1": 0}, "rows": [], "unmapped": []}},
                manifest_hash="",
                arm=config.arm,
                seed=config.seed,
            )
            mapping.manifest_hash = arm_runtime.canonical_sha256(
                {sid: mapping.source_to_mapping[sid] for sid in sorted(mapping.source_to_mapping)}
            )
            run_dir = config.run_dir()
            temporal_train.write_mapping_files(run_dir, mapping, config.config_hash)
            from unittest.mock import patch

            with patch.dict(sys.modules, {"torch": None}):
                loaded = temporal_train.load_frozen_mapping(run_dir, config)
            self.assertEqual(loaded.manifest_hash, mapping.manifest_hash)

    def test_source_atomic_predictions_and_manifest_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = arm_runtime.config_from_dict(
                {
                    "arm": arm_runtime.ARM_R_T2_SC,
                    "seed": 7301,
                    "root": Path(tmp) / "arms",
                    "input_hash": _h("input"),
                    "checkpoint_hash": _h("ckpt"),
                    "partition_hash": _h("part"),
                    "weights_hash": _h("w"),
                    "code_hash": _h("code"),
                }
            )
            run_dir = config.run_dir()
            path = arm_runtime.save_source_predictions(
                run_dir, "s-a", {"logits": [0.1]}, config.binding
            )
            self.assertTrue(path.is_file())
            loaded = arm_runtime.load_source_predictions(
                run_dir, "s-a", config.binding
            )
            self.assertEqual(loaded, {"logits": [0.1]})
            with self.assertRaises(arm_runtime.CheckpointError):
                arm_runtime.load_source_predictions(run_dir, "s-b", config.binding)
            with self.assertRaises(arm_runtime.ArmError):
                arm_runtime.write_final_manifest(
                    run_dir, {"arm": config.arm}, [path, run_dir / "missing.json"]
                )
            self.assertFalse((run_dir / arm_runtime.FINAL_MANIFEST_NAME).is_file())
            final = arm_runtime.write_final_manifest(
                run_dir, {"arm": config.arm}, [path]
            )
            self.assertTrue(final.is_file())

    def test_dev_comparisons_cover_horizons_and_groups(self):
        dev = {
            "d-ami": SimpleNamespace(source_id="d-ami", role="dev"),
            "d-ali": SimpleNamespace(source_id="d-ali", role="dev"),
        }
        snapshots = [
            SimpleNamespace(source_id="d-ami", source_family="ami_mix_headset", role="dev"),
            SimpleNamespace(source_id="d-ali", source_family="alimeeting_far_ch0", role="dev"),
        ]
        torch = self._install({"train": {}, "dev": dev, "snapshots": snapshots})
        monkey = MonkeyPatch()
        monkey.setattr(
            material,
            "infer_dev_raw_logits",
            lambda torch, wrapper, head, snapshot, session, root, device: {
                "f0_raw": [0.1] * 100,
                "cand_raw": [0.2] * 100,
                "target": [1.0 if i % 3 else 0.0 for i in range(100)],
                "valid": [True] * 100,
                "mapped_flags": [True] * 100,
                "kept": list(range(100)),
                "unmapped_frames": [],
                "grid_frames": 100,
                "infer_seconds": 0.5,
                "mapping_rows": [],
                "mapping_mapped": 1,
                "coverage": {"frames": 100, "kept": 100, "positive": 1, "negative": 99},
            },
        )

        import experiments.psem_state_corrected_adaptation_gate.stages as stages_mod

        def _forbidden_score(*args, **kwargs):
            raise AssertionError("per-source exact scoring must not run")

        monkey.setattr(stages_mod, "score_dev_frontiers", _forbidden_score)

        from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

        wave_calls: list = []
        real_wave = cross_mod.run_exact_wave

        def _counting_wave(members, tasks, workers):
            wave_calls.append((dict(members), list(tasks), workers))
            return real_wave(members, tasks, workers)

        monkey.setattr(cross_mod, "run_exact_wave", _counting_wave)

        def _block():
            return cross_mod.build_block(
                [
                    {"threshold": 0.4, "false_cuts_per_hour": 10.0, "contamination": 1.0, "miss_rate": 0.5},
                    {"threshold": 0.5, "false_cuts_per_hour": 20.0, "contamination": 2.0, "miss_rate": 0.6},
                ],
                {"threshold": 0.5, "false_cuts_per_hour": 20.0, "contamination": 2.0, "miss_rate": 0.6},
            )

        baseline = {
            "artifact_role": cross_mod.ARTIFACT_ROLE,
            "version": cross_mod.CANONICAL_VERSION,
            "arm": "R-H-SC",
            "horizons_ms": [100, 300, 500],
            "group_order": list(cross_mod.GROUP_ORDER),
            "horizons": {
                str(h): {
                    group: {kind: _block() for kind in ("calibrated", "raw")}
                    for group in ("macro", "ami", "alimeeting", "pooled")
                }
                for h in (100, 300, 500)
            },
            "sources": {},
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                base_path = tmp_path / "base.json"
                base_path.write_text(json.dumps(baseline), encoding="utf-8")
                config = arm_runtime.config_from_dict(
                    {
                        "arm": arm_runtime.ARM_R_T2_SC,
                        "seed": 7301,
                        "root": tmp_path / "arms",
                        "input_hash": _h("input"),
                        "checkpoint_hash": _h("ckpt"),
                        "partition_hash": _h("part"),
                        "weights_hash": _h("w"),
                        "code_hash": _h("code"),
                    }
                )
                args = SimpleNamespace(
                    corpus_root=tmp_path,
                    reference_root=tmp_path,
                    workers=1,
                    baseline_frontier=base_path,
                )
                artifact = temporal_train.run_dev_command(
                    torch, FakeWrapper(), FakeHead(), "cpu", config,
                    config.run_dir(), ({"slope": 1.0, "intercept": 0.0}, {"slope": 1.0, "intercept": 0.0}), args,
                )
                document = artifact["document"]
                comparisons = document["comparisons"]
                self.assertEqual(sorted(comparisons), ["100", "300", "500"])
                for horizon in ("100", "300", "500"):
                    self.assertEqual(
                        comparisons[horizon]["pooled"]["calibrated"]["baseline"], "R-H-SC+F0"
                    )
                    self.assertEqual(
                        comparisons[horizon]["macro"]["calibrated"]["budget"], 20.0
                    )
                self.assertEqual(document["artifact_role"], "issue-121-cross-arm-dev-frontier")
                self.assertEqual(document["arm"], "R-T2-SC")
                self.assertEqual(len(wave_calls), 1)
                _, wave_tasks, _ = wave_calls[0]
                self.assertEqual(len(wave_tasks), 2 * 3 * 3)
                phase = document["gate_evidence"]["phase"]
                self.assertEqual(phase["pool_count"], 0)
                self.assertTrue(phase["exact"])
                self.assertEqual(phase["total_tasks"], 2 * 3 * 3)
                for horizon in ("100", "300", "500"):
                    for group in ("macro", "ami", "alimeeting", "pooled"):
                        self.assertEqual(phase["thresholds"][horizon]["calibrated"][group], 1)
                        self.assertEqual(phase["thresholds"][horizon]["raw"][group], 1)
                gate = document["gate_evidence"]
                self.assertEqual(gate["first"], "macro")
                self.assertEqual(sorted(gate["horizons"]), ["100", "300", "500"])
                horizons = document["horizons"]
                pooled_points = horizons["100"]["pooled"]["calibrated"]["points"]
                self.assertEqual(len(pooled_points), 1)
                self.assertAlmostEqual(pooled_points[0]["threshold"], 0.549834, places=5)
                self.assertEqual(pooled_points[0]["false_cuts_per_hour"], 55.0)
                for group, kept in (
                    ("ami", 100),
                    ("alimeeting", 100),
                    ("pooled", 200),
                    ("macro", 200),
                ):
                    metrics = horizons["100"][group]["calibrated"]["diagnostics"]
                    self.assertEqual(metrics["kept_frames"], kept)
        finally:
            monkey.undo()
            self._remove()

    def test_profile_has_no_source_boundary_flush(self):
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fix = self._fixture(tmp_path, small_first=2000)
            manifest = json.loads((fix["bundle"] / "stage_a_manifest.json").read_text(encoding="utf-8"))
            manifest = {k: v for k, v in manifest.items() if k != "payload_sha256"}
            config = _bound_config(
                tmp_path, manifest, fix["checkpoint"], arm_runtime.ARM_R_T2_SC, 7301
            )
            store = tmp_path / "store"
            _authorize_t2(config.root, store, config)
            STUB_CALLS.clear()
            DECODE_THREADS.clear()
            sessions = {"train": fix["sessions"], "dev": fix["dev"], "snapshots": fix["snapshots"]}
            self._install(sessions)
            monkey = MonkeyPatch()
            self._patch_material(monkey)
            seen: list[int] = []
            real_apply = temporal_train.apply_optimizer_update

            def _recording(torch, optimizer, scheduler, carry, accum, accumulation=16):
                seen.append(int(accum.pending))
                return real_apply(torch, optimizer, scheduler, carry, accum, accumulation)

            monkey.setattr(temporal_train, "apply_optimizer_update", _recording)
            try:
                from experiments.psem_state_corrected_adaptation_gate import (
                    run_temporal_arm,
                )

                code = run_temporal_arm.main(
                    self._cli_args(tmp_path, config, fix, ("--profile-only",))
                )
                self.assertEqual(code, 0)
                receipt = temporal_train.load_profile_receipt(config.run_dir())
                assert receipt is not None
                self.assertEqual(receipt["optimizer_steps"], 8)
                self.assertTrue(seen)
                for pending in seen[:-1]:
                    self.assertEqual(pending, 16)
                main_ident = threading.get_ident()
                self.assertTrue(DECODE_THREADS)
                for ident in DECODE_THREADS:
                    self.assertEqual(ident, main_ident)
            finally:
                monkey.undo()
                self._remove()

    def test_full_run_serial_single_waveform(self):
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fix = self._fixture(tmp_path, frames=2000)
            manifest = json.loads((fix["bundle"] / "stage_a_manifest.json").read_text(encoding="utf-8"))
            manifest = {k: v for k, v in manifest.items() if k != "payload_sha256"}
            config = _bound_config(
                tmp_path, manifest, fix["checkpoint"], arm_runtime.ARM_R_T2_SC, 7301
            )
            store = tmp_path / "store"
            _authorize_t2(config.root, store, config)
            from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod

            def _base_block():
                return cross_mod.build_block(
                    [
                        {"threshold": 0.4, "false_cuts_per_hour": 10.0, "contamination": 1.0, "miss_rate": 0.5},
                        {"threshold": 0.5, "false_cuts_per_hour": 20.0, "contamination": 2.0, "miss_rate": 0.6},
                    ],
                    {"threshold": 0.5, "false_cuts_per_hour": 20.0, "contamination": 2.0, "miss_rate": 0.6},
                )

            base_doc = {
                "artifact_role": cross_mod.ARTIFACT_ROLE,
                "version": cross_mod.CANONICAL_VERSION,
                "arm": "R-H-SC",
                "horizons_ms": [100, 300, 500],
                "group_order": list(cross_mod.GROUP_ORDER),
                "horizons": {
                    str(h): {
                        group: {kind: _base_block() for kind in ("calibrated", "raw")}
                        for group in ("macro", "ami", "alimeeting", "pooled")
                    }
                    for h in (100, 300, 500)
                },
                "sources": {},
            }
            base_path = tmp_path / "base.json"
            base_path.write_text(json.dumps(base_doc), encoding="utf-8")
            run_dir = config.run_dir()
            run_dir.mkdir(parents=True, exist_ok=True)
            arm_runtime.atomic_write_json(
                temporal_train.profile_receipt_path(run_dir),
                {
                    "optimizer_steps": 8,
                    "seconds_per_step": 0.01,
                    "peak_vram_bytes": 0,
                    "dev_infer_seconds": {"dev-ami-00": 0.1},
                    "arm": config.arm,
                    "seed": config.seed,
                    "config_hash": config.config_hash,
                },
            )
            dev_sessions = {
                "d-ami": SimpleNamespace(source_id="d-ami", role="dev"),
                "d-ali": SimpleNamespace(source_id="d-ali", role="dev"),
            }
            dev_snaps = [
                SimpleNamespace(source_id="d-ami", source_family="ami_mix_headset", role="dev"),
                SimpleNamespace(source_id="d-ali", source_family="alimeeting_far_ch0", role="dev"),
            ]
            sessions = {"train": fix["sessions"], "dev": dev_sessions, "snapshots": dev_snaps}
            self._install(sessions)
            monkey = MonkeyPatch()
            self._patch_material(monkey)
            from experiments.psem_state_corrected_adaptation_gate import frontier as _frontier

            def _point(threshold, cuts):
                return _frontier.FrontierPoint(
                    threshold=threshold,
                    false_cuts_per_hour=cuts,
                    contamination=1.0,
                    miss_rate=0.5,
                )

            def _score(snapshot, f0, cand, target, valid, mapped, unmapped, cal_f0, cal_cand, workers):
                horizons = {}
                for horizon in (100, 300, 500):
                    horizons[horizon] = {
                        "f0_point": _point(0.5, 20.0),
                        "candidate_points": [_point(0.4, 10.0), _point(0.5, 20.0)],
                    }
                return horizons, {
                    "f0_cal": [0.1] * 400,
                    "cand_cal": [0.2] * 400,
                    "kept": list(range(400)),
                }

            import experiments.psem_state_corrected_adaptation_gate.stages as stages_mod

            monkey.setattr(stages_mod, "score_dev_frontiers", _score)
            import concurrent.futures as _futures

            def _no_pool(*args, **kwargs):
                raise AssertionError("background pool must not be constructed")

            monkey.setattr(_futures, "ProcessPoolExecutor", _no_pool)
            TrackedWave.live.clear()
            DECODE_THREADS.clear()
            main_ident = threading.get_ident()
            try:
                from experiments.psem_state_corrected_adaptation_gate import (
                    run_temporal_arm,
                )

                code = run_temporal_arm.main(
                    self._cli_args(
                        tmp_path, config, fix,
                        ("--baseline-frontier", str(base_path)),
                    )
                )
                self.assertEqual(code, 0)
                self.assertTrue(DECODE_THREADS)
                for ident in DECODE_THREADS:
                    self.assertEqual(ident, main_ident)
                gc.collect()
                alive = sum(1 for ref in TrackedWave.live if ref() is not None)
                self.assertEqual(alive, 0)
                final = run_dir / arm_runtime.FINAL_MANIFEST_NAME
                self.assertTrue(final.is_file())
                for name in (
                    "experiment_manifest.json",
                    "data_sampling_calibration_manifest.json",
                    "parameter_module_mode_receipt.json",
                    "training_metrics.json",
                    "calibration_metrics.json",
                    "dev_frontier.json",
                ):
                    self.assertTrue((run_dir / name).is_file(), name)
                dev_doc = json.loads((run_dir / "dev_frontier.json").read_text(encoding="utf-8"))
                self.assertEqual(dev_doc["artifact_role"], "issue-121-cross-arm-dev-frontier")
                self.assertEqual(dev_doc["arm"], "R-T2-SC")
                self.assertEqual(sorted(dev_doc["horizons"]), ["100", "300", "500"])
                for horizon in ("100", "300", "500"):
                    self.assertEqual(
                        sorted(dev_doc["horizons"][horizon]),
                        ["alimeeting", "ami", "macro", "pooled"],
                    )
                    for group in ("macro", "ami", "alimeeting", "pooled"):
                        for kind in ("calibrated", "raw"):
                            block = dev_doc["horizons"][horizon][group][kind]
                            self.assertEqual(block["reference"]["threshold"], 0.5)
                            self.assertEqual(block["budget"], block["reference"]["false_cuts_per_hour"])
                self.assertEqual(sorted(dev_doc["comparisons"]), ["100", "300", "500"])
                for horizon in ("100", "300", "500"):
                    self.assertEqual(
                        sorted(dev_doc["comparisons"][horizon]),
                        ["alimeeting", "ami", "macro", "pooled"],
                    )
                    for group in ("macro", "ami", "alimeeting", "pooled"):
                        self.assertEqual(
                            sorted(dev_doc["comparisons"][horizon][group]),
                            ["calibrated", "raw"],
                        )
                self.assertEqual(dev_doc["gate_evidence"]["first"], "macro")
                data_doc = json.loads(
                    (run_dir / "data_sampling_calibration_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(data_doc["targets"])
                for sid, record in data_doc["targets"].items():
                    self.assertTrue(record["sha256"])
                final_doc = json.loads(final.read_text(encoding="utf-8"))
                self.assertEqual(
                    final_doc["target_binding"],
                    arm_runtime.canonical_sha256(
                        {sid: data_doc["targets"][sid]["sha256"] for sid in sorted(data_doc["targets"])}
                    ),
                )
            finally:
                monkey.undo()
                self._remove()

    def test_setup_failures_release_gpu_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fix = self._fixture(tmp_path, frames=2000)
            manifest = json.loads((fix["bundle"] / "stage_a_manifest.json").read_text(encoding="utf-8"))
            manifest = {k: v for k, v in manifest.items() if k != "payload_sha256"}
            config = _bound_config(
                tmp_path, manifest, fix["checkpoint"], arm_runtime.ARM_R_T2_SC, 7301
            )
            store = tmp_path / "store"
            _authorize_t2(config.root, store, config)
            sessions = {"train": fix["sessions"], "dev": {}, "snapshots": []}
            self._install(sessions)
            monkey = MonkeyPatch()
            self._patch_material(monkey)
            try:
                from experiments.psem_state_corrected_adaptation_gate import (
                    run_temporal_arm,
                )

                lock_path = Path(config.root) / arm_runtime.ARM_GPU_LOCK_NAME

                def _boom(*args, **kwargs):
                    raise temporal_train.TemporalArmError("injected setup failure")

                for dotted in (
                    "experiments.psem_state_corrected_adaptation_gate.temporal_train.resolve_backend",
                    "experiments.psem_state_corrected_adaptation_gate.temporal_train.resolve_durable_targets",
                    "experiments.psem_state_corrected_adaptation_gate.temporal_train.open_temporal_model",
                    "experiments.psem_state_corrected_adaptation_gate.temporal_train.check_single_stream",
                ):
                    probe = MonkeyPatch()
                    probe.setattr(dotted, _boom)
                    try:
                        self.assertFalse(lock_path.exists())
                        code = run_temporal_arm.main(
                            self._cli_args(tmp_path, config, fix, ("--profile-only",))
                        )
                        self.assertEqual(code, 3)
                        self.assertFalse(lock_path.exists())
                    finally:
                        probe.undo()
            finally:
                monkey.undo()
                self._remove()

    def test_cli_profile_writes_bound_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fix = self._fixture(tmp_path)
            manifest = json.loads((fix["bundle"] / "stage_a_manifest.json").read_text(encoding="utf-8"))
            manifest = {k: v for k, v in manifest.items() if k != "payload_sha256"}
            config = _bound_config(
                tmp_path, manifest, fix["checkpoint"], arm_runtime.ARM_R_T2_SC, 7301
            )
            store = tmp_path / "store"
            _authorize_t2(config.root, store, config)
            STUB_CALLS.clear()
            sessions = {"train": fix["sessions"], "dev": fix["dev"], "snapshots": fix["snapshots"]}
            self._install(sessions)
            monkey = MonkeyPatch()
            self._patch_material(monkey)
            try:
                from experiments.psem_state_corrected_adaptation_gate import (
                    run_temporal_arm,
                )

                code = run_temporal_arm.main(
                    self._cli_args(tmp_path, config, fix, ("--profile-only",))
                )
                self.assertEqual(code, 0)
                receipt = temporal_train.load_profile_receipt(config.run_dir())
                self.assertIsNotNone(receipt)
                assert receipt is not None
                self.assertEqual(receipt["optimizer_steps"], 8)
                self.assertEqual(receipt["arm"], "R-T2-SC")
                self.assertEqual(receipt["seed"], 7301)
                self.assertEqual(receipt["config_hash"], config.config_hash)
                self.assertEqual(receipt["weights_hash"], config.weights_hash)
                self.assertEqual(sorted(receipt["fit_sources"]), fix["fit"])
                self.assertEqual(receipt["dev_source"], "dev-ami-00")
                self.assertIn("seconds_per_step", receipt)
                self.assertIn("schedule_total_steps", receipt)
                self.assertTrue(STUB_CALLS)
            finally:
                monkey.undo()
                self._remove()

    def test_cli_binding_mismatch_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fix = self._fixture(tmp_path, frames=2000)
            manifest = json.loads((fix["bundle"] / "stage_a_manifest.json").read_text(encoding="utf-8"))
            manifest = {k: v for k, v in manifest.items() if k != "payload_sha256"}
            config = _bound_config(
                tmp_path, manifest, fix["checkpoint"], arm_runtime.ARM_R_T2_SC, 7301
            )
            tampered = arm_runtime.config_from_dict(
                {**{k: getattr(config, k) for k in (
                    "arm", "seed", "input_hash", "checkpoint_hash",
                    "partition_hash", "weights_hash", "code_hash")},
                 "weights_hash": "0" * 64,
                 "root": config.root}
            )
            store = tmp_path / "store"
            _authorize_t2(config.root, store, tampered)
            STUB_CALLS.clear()
            sessions = {"train": fix["sessions"], "dev": fix["dev"], "snapshots": fix["snapshots"]}
            self._install(sessions)
            monkey = MonkeyPatch()
            self._patch_material(monkey)
            try:
                from experiments.psem_state_corrected_adaptation_gate import (
                    run_temporal_arm,
                )

                code = run_temporal_arm.main(
                    self._cli_args(tmp_path, tampered, fix, ("--profile-only",))
                )
                self.assertEqual(code, 3)
                self.assertEqual(STUB_CALLS, [])
            finally:
                monkey.undo()
                self._remove()

    def test_cli_full_rejected_without_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fix = self._fixture(tmp_path, frames=2000)
            manifest = json.loads((fix["bundle"] / "stage_a_manifest.json").read_text(encoding="utf-8"))
            manifest = {k: v for k, v in manifest.items() if k != "payload_sha256"}
            config = _bound_config(
                tmp_path, manifest, fix["checkpoint"], arm_runtime.ARM_R_T2_SC, 7301
            )
            store = tmp_path / "store"
            _authorize_t2(config.root, store, config)
            STUB_CALLS.clear()
            sessions = {"train": fix["sessions"], "dev": fix["dev"], "snapshots": fix["snapshots"]}
            self._install(sessions)
            try:
                from experiments.psem_state_corrected_adaptation_gate import (
                    run_temporal_arm,
                )

                code = run_temporal_arm.main(
                    self._cli_args(
                        tmp_path, config, fix,
                        ("--baseline-frontier", str(tmp_path / "base.json")),
                    )
                )
                self.assertEqual(code, 3)
                self.assertFalse(
                    (config.run_dir() / arm_runtime.FINAL_MANIFEST_NAME).is_file()
                )
            finally:
                self._remove()

    def test_cli_auth_before_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fix = self._fixture(tmp_path, frames=2000)
            manifest = json.loads((fix["bundle"] / "stage_a_manifest.json").read_text(encoding="utf-8"))
            manifest = {k: v for k, v in manifest.items() if k != "payload_sha256"}
            config = _bound_config(
                tmp_path, manifest, fix["checkpoint"], arm_runtime.ARM_R_T2_SC, 7301
            )
            STUB_CALLS.clear()
            sessions = {"train": fix["sessions"], "dev": fix["dev"], "snapshots": fix["snapshots"]}
            self._install(sessions)
            try:
                from experiments.psem_state_corrected_adaptation_gate import (
                    run_temporal_arm,
                )

                args = self._cli_args(tmp_path, config, fix, ("--profile-only",))
                store_index = args.index("--store") + 1
                args[store_index] = str(tmp_path / "empty-store")
                code = run_temporal_arm.main(args)
                self.assertEqual(code, 3)
                self.assertEqual(STUB_CALLS, [])
            finally:
                self._remove()

    def test_cli_has_concrete_backend_options(self):
        from experiments.psem_state_corrected_adaptation_gate import run_temporal_arm

        parser = run_temporal_arm.build_parser()
        arm_action = next(
            a for a in parser._actions if "--arm" in a.option_strings
        )
        self.assertEqual(set(arm_action.choices or []), {"R-T2-SC", "R-TA-SC"})
        names = set()
        for action in parser._actions:
            names.update(action.option_strings)
        for required in (
            "--checkpoint",
            "--bundle",
            "--nemo-checkout",
            "--dependency-lock",
            "--corpus-root",
            "--reference-root",
            "--sampling-manifest",
            "--device",
            "--baseline-frontier",
        ):
            self.assertIn(required, names)
        for action in parser._actions:
            for option in action.option_strings:
                self.assertNotIn("callback", option)
                self.assertNotIn("provider", option)


if __name__ == "__main__":
    unittest.main()
