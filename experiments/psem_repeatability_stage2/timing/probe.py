"""Stage-2 timing probe (CPU-only, bounded to 3 frozen episodes).

Real local compute only: no stubs, no fabricated fallbacks. Every failure is
retained verbatim as evidence. Uses the existing frozen code path
(validate_export_manifest / npz helpers / load_pinned_sortformer /
ResidualPSEMHead) and the repo .venv (torch CPU + numpy), plus one isolated
temp interpreter for the direct upstream restore labeled non-parity.

Run from the repository root::

    .venv/Scripts/python experiments/psem_repeatability_stage2/timing/probe.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TIMING_DIR = REPO / "experiments" / "psem_repeatability_stage2" / "timing"
EXPORT_DIR = (
    REPO
    / "experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1"
    / "export/gpu_export"
)
CHECKPOINT = (
    REPO
    / ".cache/issue-107-assets/checkpoints/diar_streaming_sortformer_4spk-v2.1.nemo"
)
NEMO_CHECKOUT = REPO / ".cache/issue-107-assets/sources/NeMo"
GATE0_LOCK = REPO / ".cache/issue-121-gate0/dependency_lock.json"
PINNED_LOCK = (
    REPO
    / "experiments/psem_sortformer_adaptation_depth/results/issue-107-a40-1334720a-01"
    / "receipts/dependency-lock.json"
)
def _iso_root() -> Path:
    for candidate in (
        Path(tempfile.gettempdir()) / "psem-timing-iso",
        Path("C:/tmp/psem-timing-iso"),
    ):
        if (candidate / "Scripts" / "python.exe").is_file():
            return candidate
    return Path(tempfile.gettempdir()) / "psem-timing-iso"


ISO_ROOT = _iso_root()
ISO_PYTHON = ISO_ROOT / "Scripts" / "python.exe"
ISO_AUDIO = Path("C:/tmp/psem-timing-audio")


def _crops() -> list:
    items = []
    for case in CASES:
        meeting = str(case["source_id"]).split("_", 1)[1]
        items.append(
            {
                "case_id": case["case_id"],
                "meeting": meeting,
                "wav": str(
                    ISO_AUDIO / "ami" / "audio" / meeting / f"{meeting}.Mix-Headset.wav"
                ),
                "start": int(case["payload_sample_range"][0]),
                "end": int(case["payload_sample_range"][1]),
            }
        )
    return items
FREEZE = json.loads((TIMING_DIR / "freeze.json").read_text(encoding="utf-8"))
CASES = list(FREEZE["cases"])
COHORT = dict(FREEZE["reference_scope"]["cohort"])

SEVEN_S_FRAMES = 87
FULL_CHUNK_FRAMES = 375
HEAD_DIM = 199
PROXY_ITERS = 20
PROXY_WARMUP = 3
ISO_TIMEOUT_SECONDS = 1800

SOURCE_EVIDENCE = {
    "accelerator_gate": (
        "experiments/psem_sortformer_adaptation_depth/nemo_adapter.py:92-117 "
        "_accelerator_identity raises NeMoAdapterError unless exactly one CUDA "
        "device is exposed"
    ),
    "container_gate": (
        "experiments/psem_sortformer_adaptation_depth/nemo_adapter.py:83-89 "
        "_container_image_identity raises unless PSEM_CONTAINER_IMAGE_IDENTITY "
        "equals the pinned digest"
    ),
    "lock_validation": (
        "experiments/psem_sortformer_adaptation_depth/nemo_adapter.py:147-193 "
        "validate_dependency_lock requires schema, artifact role, nemo revision, "
        "current python version, current platform, container identity, "
        "accelerator identity, and the exact installed package inventory"
    ),
    "upstream_restore_call": (
        "experiments/psem_sortformer_adaptation_depth/nemo_adapter.py:348-353 "
        "load_pinned_sortformer restores via "
        "SortformerEncLabelModel.restore_from on verified checkpoint bytes"
    ),
    "lock_convention": (
        "experiments/psem_sortformer_adaptation_depth/issue_107_launch.py:479-481 "
        "the dependency lock is generated at runtime by write_dependency_lock "
        "into the run receipts; the canonical artifact used here is the "
        "issue-107 run receipts/dependency-lock.json, kept unmodified"
    ),
}


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_attempt(attempt_id: str, func):
    wall_begin = utcnow()
    start = time.perf_counter()
    try:
        detail = func()
        status = "ok"
        error = None
    except Exception as exc:
        detail = None
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc)[:800]}
    wall_seconds = time.perf_counter() - start
    return {
        "attempt_id": attempt_id,
        "status": status,
        "wall_seconds": wall_seconds,
        "wall_begin": wall_begin,
        "wall_end": utcnow(),
        "detail": detail,
        "error": error,
    }


def by_id(attempts: list, attempt_id: str) -> dict:
    for row in attempts:
        if row["attempt_id"] == attempt_id:
            return row
    return {}


def main() -> int:
    import torch

    import experiments.psem_state_corrected_adaptation_gate.arm_runtime as arm_runtime
    import experiments.psem_state_corrected_adaptation_gate.h_postprocess as h_post
    import experiments.psem_state_corrected_adaptation_gate.head as head_mod
    import experiments.psem_sortformer_adaptation_depth.nemo_adapter as nemo_adapter

    thread_caps = arm_runtime.enforce_thread_caps()

    attempts: list[dict] = []
    measurements: list[dict] = []

    attempts.append(
        run_attempt(
            "f0_checkpoint_verify",
            lambda: {
                "path": str(CHECKPOINT.relative_to(REPO)),
                "size_bytes": CHECKPOINT.stat().st_size,
                "sha256": sha256_file(CHECKPOINT),
                "expected_sha256": FREEZE["assets_expected"][
                    "f0_checkpoint_sha256_expected"
                ],
                "match": sha256_file(CHECKPOINT)
                == FREEZE["assets_expected"]["f0_checkpoint_sha256_expected"],
            },
        )
    )

    def _nemo_identity():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=NEMO_CHECKOUT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=NEMO_CHECKOUT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"revision": head, "clean": dirty == [], "dirty_lines": dirty[:10]}

    attempts.append(run_attempt("nemo_checkout_verify", _nemo_identity))

    def _pinned_restore(lock_path: Path):
        return (
            lambda out: {
                "lock": str(lock_path.relative_to(REPO)),
                "device_requested": "cpu",
                "restored": True,
                "runtime_receipt_keys": sorted(out[1].keys()),
            }
        )(
            nemo_adapter.load_pinned_sortformer(
                CHECKPOINT, NEMO_CHECKOUT, lock_path, "cpu"
            )
        )

    attempts.append(
        run_attempt(
            "f0_restore_pinned_gate0lock",
            lambda: _pinned_restore(GATE0_LOCK),
        )
    )
    attempts.append(
        run_attempt(
            "f0_restore_pinned_107lock",
            lambda: _pinned_restore(PINNED_LOCK),
        )
    )
    pinned_ok = (
        by_id(attempts, "f0_restore_pinned_gate0lock")["status"] == "ok"
        or by_id(attempts, "f0_restore_pinned_107lock")["status"] == "ok"
    )

    def _head_search():
        export_files = sorted(p.name for p in EXPORT_DIR.iterdir())
        gate0_files = sorted(p.name for p in (REPO / ".cache/issue-121-gate0").iterdir())
        manifest = json.loads(
            (EXPORT_DIR / "gpu_export_manifest.json").read_text(encoding="utf-8")
        )
        suspects = [
            n
            for n in export_files + gate0_files
            if n.endswith((".pt", ".safetensors", ".bin", ".ckpt"))
            or "head" in n.lower()
        ]
        return {
            "declared_trained_head_sha256": manifest.get("trained_head_sha256"),
            "export_dir_files": export_files,
            "gate0_dir_files": gate0_files,
            "weight_like_files_found": suspects,
            "local_weights_present": bool(suspects),
        }

    attempts.append(run_attempt("h7301_head_availability", _head_search))

    attempts.append(
        run_attempt(
            "export_manifest_validate",
            lambda: (
                lambda v: {
                    "arm": v["arm"],
                    "seed": v["seed"],
                    "calib_sources": v["calib_sources"],
                    "dev_sources": v["dev_sources"],
                }
            )(h_post.validate_export_manifest(EXPORT_DIR)),
        )
    )

    manifest_files = json.loads(
        (EXPORT_DIR / "gpu_export_manifest.json").read_text(encoding="utf-8")
    )["files"]
    validated = (
        h_post.validate_export_manifest(EXPORT_DIR)
        if by_id(attempts, "export_manifest_validate")["status"] == "ok"
        else None
    )
    for case in CASES:
        source_id = case["source_id"]
        cohort = COHORT[source_id]

        def _load():
            table = validated["manifest"]["dev"] if validated else None
            sc = h_post._sidecar(table, source_id, "dev")
            path = h_post._resolve_export_file(EXPORT_DIR, sc, source_id)
            digest = h_post._check_hash(path, sc, manifest_files, str(sc["file"]))
            arrays = h_post._load_npz(path)
            if int(sc.get("frames", arrays["frames"])) != arrays["frames"]:
                raise RuntimeError("sidecar frame count differs from loaded arrays")
            return {
                "file": str(sc["file"]),
                "frames": arrays["frames"],
                "sha256": digest,
                "expected_sha256": cohort["sha256"],
                "sha_match": digest == cohort["sha256"],
            }

        row = run_attempt(f"npz_load_{case['case_id']}", _load)
        attempts.append(row)
        measurements.append(
            {
                "case_id": case["case_id"],
                "component": "frozen_evidence_npz_load",
                "status": "MEASURED" if row["status"] == "ok" else "BLOCKED",
                "wall_seconds": row["wall_seconds"]
                if row["status"] == "ok"
                else None,
                "reason": None
                if row["status"] == "ok"
                else f"npz load failed: {row['error']}",
                "source_support": (
                    "full-meeting DEV npz load supported "
                    f"({cohort['frames']} frames); 7s-crop forward unsupported: "
                    "no in-scope waveform, no recurrent prefix cache"
                ),
                "model_identity": (
                    "F0/H frozen numeric export: retained arrays "
                    "(f0_raw, cand_raw, target, valid, mapped); no live model state"
                ),
                "reference_scope": (
                    f"full-meeting DEV session {source_id}, "
                    f"{cohort['frames']} native frames @80ms "
                    f"(export {cohort['export_file']}); 7s payload "
                    f"{case['payload_sample_range']} boundary "
                    f"{case['boundary_sample']} per freeze; episode prefix state unavailable"
                ),
                "causal_equivalence": (
                    "retained frozen evidence bytes with sha256 verification; "
                    "causally same bytes, not a forward recomputation"
                ),
                "limitation": (
                    "load/parse service time only; session-frame cross-check "
                    "against live sessions not performed (sessions module out of scope); "
                    "CPU time descriptive, never a GPU/WCET bound"
                ),
            }
        )

    def _head_init():
        torch.manual_seed(7301)
        module = head_mod.ResidualPSEMHead(HEAD_DIM)
        module.eval()
        params = sum(p.numel() for p in module.parameters())
        return {"params": params, "head_input_dim": HEAD_DIM}

    init_row = run_attempt("architecture_proxy_head_init", _head_init)
    attempts.append(init_row)
    torch.manual_seed(7301)
    proxy_head = (
        head_mod.ResidualPSEMHead(HEAD_DIM).eval()
        if init_row["status"] == "ok"
        else None
    )

    def _time_forward(frames: int, seed: int) -> dict:
        gen = torch.Generator().manual_seed(seed)
        batch = torch.randn(1, frames, HEAD_DIM, generator=gen, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(PROXY_WARMUP):
                proxy_head(batch)
        laps: list[float] = []
        with torch.no_grad():
            for _ in range(PROXY_ITERS):
                tick = time.perf_counter()
                proxy_head(batch)
                laps.append(time.perf_counter() - tick)
        ordered = sorted(laps)
        return {
            "frames": frames,
            "iters": PROXY_ITERS,
            "warmup_iters": PROXY_WARMUP,
            "mean_seconds": sum(laps) / len(laps),
            "min_seconds": ordered[0],
            "max_seconds": ordered[-1],
            "p50_seconds": ordered[len(ordered) // 2],
        }

    proxy_rows: dict[str, dict] = {}
    if proxy_head is not None:
        full_row = run_attempt(
            "architecture_proxy_head_forward_full_chunk",
            lambda: _time_forward(FULL_CHUNK_FRAMES, 7301),
        )
        attempts.append(full_row)
        proxy_rows["full_chunk"] = full_row
        for index, case in enumerate(CASES):
            case_row = run_attempt(
                f"architecture_proxy_head_forward_{case['case_id']}",
                lambda index=index: _time_forward(SEVEN_S_FRAMES, 7301 + index),
            )
            attempts.append(case_row)
            proxy_rows[case["case_id"]] = case_row
    else:
        for case in CASES:
            row = run_attempt(
                f"architecture_proxy_head_forward_{case['case_id']}",
                lambda: (_ for _ in ()).throw(
                    RuntimeError("proxy head construction failed; no forward timed")
                ),
            )
            attempts.append(row)
            proxy_rows[case["case_id"]] = row
    for index, case in enumerate(CASES):
        row = proxy_rows[case["case_id"]]
        ok = row["status"] == "ok"
        detail = row.get("detail") or {}
        measurements.append(
            {
                "case_id": case["case_id"],
                "component": "architecture_proxy_head_forward_7s",
                "status": "MEASURED" if ok else "BLOCKED",
                "wall_seconds": detail.get("mean_seconds") if ok else None,
                "reason": None
                if ok
                else f"proxy forward failed: {row.get('error')}",
                "source_support": (
                    f"synthetic (1,{SEVEN_S_FRAMES},{HEAD_DIM}) float32 supported "
                    "(87 complete native frames for the 112000-sample payload); "
                    "real backbone features unavailable"
                ),
                "model_identity": (
                    "ARCHITECTURE_PROXY_UNTRAINED: ResidualPSEMHead(199) "
                    f"random init torch.manual_seed(7301), synthetic inputs seed {7301 + index}; "
                    "never valid H output"
                ),
                "reference_scope": (
                    f"7s payload {case['payload_sample_range']} of "
                    f"{case['source_id']} ({case['episode']} boundary "
                    f"{case['boundary_sample']}); no recurrent prefix cache"
                ),
                "causal_equivalence": (
                    "NOT causally equivalent to any H7301 forward: untrained "
                    "weights, synthetic inputs, no backbone features, no prefix "
                    "state. Compute probe only; never summed into an H bound; "
                    "implies nothing about H bounds."
                ),
                "limitation": (
                    "CPU service time descriptive only (threads capped to 1); "
                    "repeated-mean is not a hard upper bound; sequential "
                    "backbone+head total unavailable from this probe; "
                    f"full 375-frame chunk mean "
                    f"{(proxy_rows.get('full_chunk', {}).get('detail') or {}).get('mean_seconds')}s "
                    "for scale only"
                ),
            }
        )

    def _iso_freeze():
        completed = subprocess.run(
            [str(ISO_PYTHON), "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        lines = sorted(
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip() and "==" in line
        )
        return {"iso_python": str(ISO_PYTHON), "packages": lines}

    attempts.append(run_attempt("iso_env_freeze", _iso_freeze))

    def _iso_restore():
        iso_out = Path(tempfile.gettempdir()) / "psem-iso-restore.json"
        if iso_out.is_file():
            iso_out.unlink()
        crops = _crops()
        missing = [c["wav"] for c in crops if not Path(c["wav"]).is_file()]
        if missing:
            raise RuntimeError(f"meeting audio missing in isolated cache: {missing}")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(NEMO_CHECKOUT) + os.pathsep + env.get("PYTHONPATH", "")
        wall_begin = utcnow()
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [
                    str(ISO_PYTHON),
                    str(TIMING_DIR / "iso_restore.py"),
                    str(CHECKPOINT),
                    str(iso_out),
                    json.dumps(crops),
                ],
                capture_output=True,
                text=True,
                timeout=ISO_TIMEOUT_SECONDS,
                env=env,
                cwd=str(REPO),
            )
            wall_seconds = time.perf_counter() - start
            record = json.loads(iso_out.read_text(encoding="utf-8"))
            return {
                "subprocess_wall_seconds": wall_seconds,
                "subprocess_wall_begin": wall_begin,
                "subprocess_wall_end": utcnow(),
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-1500:],
                "record": record,
            }
        except Exception as exc:
            raise RuntimeError(
                f"isolated restore subprocess failed: {type(exc).__name__}: {exc}"
            ) from exc

    iso_row = run_attempt("direct_upstream_restore_iso", _iso_restore)
    attempts.append(iso_row)
    iso_record = (iso_row.get("detail") or {}).get("record") or {}
    iso_forwards = (
        {item["case_id"]: item for item in iso_record.get("real_forwards", [])}
        if iso_row["status"] == "ok" and iso_record.get("status") == "ok"
        else {}
    )
    iso_model = iso_record.get("model") or {}
    iso_synth = iso_record.get("synthetic_supplemental") or {}

    for case in CASES:
        item = iso_forwards.get(case["case_id"]) or {}
        ok = bool(item) and bool(item.get("mean_seconds"))
        measurements.append(
            {
                "case_id": case["case_id"],
                "component": "backbone_compute_probe_real_7s",
                "status": "MEASURED" if ok else "BLOCKED",
                "wall_seconds": item.get("mean_seconds") if ok else None,
                "reason": None
                if ok
                else f"isolated real-audio forward unavailable: {iso_row.get('error') or iso_record.get('error')}",
                "source_support": (
                    "real meeting crop supported "
                    f"({item.get('crop_samples')} {item.get('payload_dtype')} samples "
                    f"@16kHz mono, payload sha {item.get('payload_sha256')}); "
                    "recurrent prefix cache and event replay unsupported"
                ),
                "model_identity": (
                    "DIRECT_UPSTREAM_RESTORE_NON_PARITY: SortformerEncLabelModel "
                    f"on cpu, {iso_model.get('params')} params, real F0 weights "
                    "sha-verified; production validator gates bypassed, therefore "
                    "never frozen parity"
                ),
                "reference_scope": (
                    f"exact frozen 7s payload {case['payload_sample_range']} of "
                    f"{case['source_id']} ({case['episode']} boundary "
                    f"{case['boundary_sample']}) read from {item.get('wav')} "
                    f"({item.get('wav_frames')} file frames); cold streaming forward "
                    "from no prefix state; frozen export logits left untouched; "
                    f"observed output shape {item.get('output_shape')}"
                ),
                "causal_equivalence": (
                    "NOT causally equivalent to frozen F0 output: isolated crop "
                    "with no recurrent prefix state, no event replay. Descriptive "
                    "compute probe only; implies nothing about H bounds; accepts "
                    "no availability."
                ),
                "limitation": (
                    "CPU service time descriptive only; import/restore/warmup cost "
                    "reported separately, not included "
                    f"(import {iso_record.get('import_seconds')}s, restore "
                    f"{iso_record.get('restore_seconds')}s, warmup "
                    f"{item.get('warmup_iters')} excluded); repeated-mean is not a "
                    "hard upper bound; GPU behavior not inferred"
                ),
            }
        )
    gate0_err = by_id(attempts, "f0_restore_pinned_gate0lock").get("error") or {}
    lock107_err = by_id(attempts, "f0_restore_pinned_107lock").get("error") or {}
    direct_ok = iso_row["status"] == "ok" and iso_record.get("status") == "ok"
    if direct_ok:
        f0_status = {
            "state": "direct_upstream_restore_non_parity_ok",
            "reason": (
                "real F0 weights restored via the exact upstream "
                "SortformerEncLabelModel.restore_from call in the isolated env "
                f"(restore {iso_record.get('restore_seconds')}s, import "
                f"{iso_record.get('import_seconds')}s); exact frozen 7s crops ran "
                "as isolated cold forwards with no prefix state; synthetic "
                f"supplemental mean {iso_synth.get('mean_seconds')}s kept outside "
                "case rows; production validator gates bypassed (lock, container, "
                "accelerator, symbol origin), so this is NOT frozen parity; pinned "
                "parity remains blocked; no availability is accepted"
            ),
        }
    else:
        f0_status = {
            "state": "live_forward_unavailable",
            "reason": (
                "pinned parity blocked "
                f"(gate0 lock: {gate0_err.get('type')}: {gate0_err.get('message')}; "
                f"107 receipts lock: {lock107_err.get('type')}: {lock107_err.get('message')}); "
                f"direct isolated restore also failed: {iso_row.get('error') or iso_record.get('error')}"
            ),
        }
    head_detail = by_id(attempts, "h7301_head_availability").get("detail") or {}
    if not head_detail.get("local_weights_present"):
        h7301_status = {
            "state": "unavailable_missing_weights",
            "reason": (
                "trained_head sha "
                f"{head_detail.get('declared_trained_head_sha256')} declared in "
                "the export manifest but no weight-like file exists in the "
                "export dir or gate0 dir; proxy measurement is untrained and non-parity"
            ),
        }
    else:
        h7301_status = {"state": "present", "reason": None}

    blockers = []
    if not pinned_ok:
        blockers.append(
            {
                "code": "F0_PINNED_PARITY_BLOCKED_AT_LOCK_IDENTITY",
                "detail": (
                    "both locks fail before import: "
                    f"gate0: {gate0_err.get('type')}: {gate0_err.get('message')}; "
                    f"107 receipts: {lock107_err.get('type')}: {lock107_err.get('message')}; "
                    "exact gates cited in source_evidence; neither lock file was modified"
                ),
            }
        )
    if not direct_ok:
        blockers.append(
            {
                "code": "F0_DIRECT_RESTORE_FAILED",
                "detail": str(
                    iso_row.get("error") or iso_record.get("error")
                ),
            }
        )
    blockers.append(
        {
            "code": "H7301_WEIGHTS_MISSING_LOCAL",
            "detail": (
                "no local file carries trained_head sha "
                f"{head_detail.get('declared_trained_head_sha256')}"
            ),
        }
    )
    blockers.append(
        {
            "code": "PREFIX_STATE_MISSING_BY_SCOPE",
            "detail": (
                "recurrent prefix state before each 7s crop start is unavailable; "
                "full source history would exceed the max-3 bound, so isolated-crop "
                "forward parity is not claimed"
            ),
        }
    )

    environment = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "thread_caps": thread_caps,
        "command": " ".join(sys.argv),
        "probe_sha256": sha256_file(Path(__file__).resolve()),
        "iso_env": {
            "root": str(ISO_ROOT),
            "python": str(ISO_PYTHON),
            "torch_version": iso_record.get("torch_version"),
            "packages": (by_id(attempts, "iso_env_freeze").get("detail") or {}).get(
                "packages"
            ),
        },
    }
    assets = {
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(REPO)),
            "sha256": (by_id(attempts, "f0_checkpoint_verify").get("detail") or {}).get(
                "sha256"
            ),
            "size_bytes": (
                by_id(attempts, "f0_checkpoint_verify").get("detail") or {}
            ).get("size_bytes"),
        },
        "nemo_checkout": {
            "path": str(NEMO_CHECKOUT.relative_to(REPO)),
            "revision": (by_id(attempts, "nemo_checkout_verify").get("detail") or {}).get(
                "revision"
            ),
            "clean": (by_id(attempts, "nemo_checkout_verify").get("detail") or {}).get(
                "clean"
            ),
        },
        "locks": {
            "gate0": str(GATE0_LOCK.relative_to(REPO)),
            "pinned_107_receipts": str(PINNED_LOCK.relative_to(REPO)),
            "note": "both kept unmodified; failures preserved as attempts",
        },
        "export_dir": str(EXPORT_DIR.relative_to(REPO)),
        "export_manifest_sha256": sha256_file(
            EXPORT_DIR / "gpu_export_manifest.json"
        ),
        "freeze_sha256": sha256_file(TIMING_DIR / "freeze.json"),
        "probe_sha256": sha256_file(Path(__file__).resolve()),
        "iso_restore_sha256": sha256_file(TIMING_DIR / "iso_restore.py"),
    }

    results = {
        "schema": "psem-repeatability-stage2-timing.v1",
        "generated_at": utcnow(),
        "freeze_cases": [c["case_id"] for c in CASES],
        "environment": environment,
        "assets": assets,
        "source_evidence": SOURCE_EVIDENCE,
        "attempts": attempts,
        "measurements": measurements,
        "full_f0_availability_status": f0_status,
        "full_h7301_availability_status": h7301_status,
        "blockers": blockers,
    }
    out_path = TIMING_DIR / "results.json"
    out_path.write_text(
        json.dumps(results, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    for attempt in attempts:
        if attempt["status"] == "ok":
            print(f"ATTEMPT ok {attempt['attempt_id']} {attempt['wall_seconds']:.6f}s")
        else:
            err = attempt["error"] or {}
            print(
                f"ATTEMPT FAILED {attempt['attempt_id']} "
                f"{attempt['wall_seconds']:.6f}s "
                f"{err.get('type')}: {err.get('message')}"
            )
    for item in measurements:
        print(
            f"MEASUREMENT {item['case_id']} {item['component']} "
            f"{item['status']} wall_seconds={item['wall_seconds']}"
        )
    print(f"WROTE {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
