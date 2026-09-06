from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.psem_state_corrected_adaptation_gate import (
    calibrate as calibrate_mod,
)
from experiments.psem_state_corrected_adaptation_gate import (
    frontier as frontier_mod,
)
from experiments.psem_state_corrected_adaptation_gate.material import (
    ClassAccumulator,
    MaterialError,
    audit_module_modes,
    build_gate0_record,
    build_horizon_result,
    candidate_frontier_points,
    is_dev_family_session,
    load_source_components,
    load_source_rows,
    mask_calibration,
    resolve_sampling_population,
    resolve_worker_count,
    select_fit_slice,
    write_gate0_artifacts,
)
from experiments.psem_state_corrected_adaptation_gate.partition import (
    SourceExposure,
    assign_train_calib,
    validate_partition_support,
)
from experiments.psem_state_corrected_adaptation_gate.receipts import (
    NEMO_SHA256,
    canonical_sha256,
    write_json,
)
STAGE_A_VERSION = 1
STAGE_B_VERSION = 1
DEV_FAMILIES = ("ami_mix_headset", "alimeeting_far_ch0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle_manifest(bundle_dir: Path, name: str) -> dict[str, Any]:
    path = bundle_dir / name
    if not path.is_file():
        raise MaterialError(f"stage bundle manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise MaterialError(f"stage bundle manifest is invalid: {path}")
    stored = manifest.get("payload_sha256")
    body = {k: v for k, v in manifest.items() if k != "payload_sha256"}
    if stored != canonical_sha256(body):
        raise MaterialError(f"stage bundle manifest hash differs: {path}")
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise MaterialError(f"stage bundle file table is invalid: {path}")
    for rel, expected in files.items():
        candidate = bundle_dir / rel
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise MaterialError(f"stage bundle file differs: {rel}")
    return manifest


def serialize_authority(source_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    authority = entry["authority"]
    return {
        "source_id": source_id,
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
        "ledger": json.loads(json.dumps(authority.ledger)),
        "multiplicity": [int(v) for v in entry["multiplicity"]],
        "episode_ids": [None if v is None else str(v) for v in entry["episode_ids"]],
        "intervals": [dict(row) for row in entry["intervals"]],
    }


def restore_authority(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(
        num_frames=int(payload["num_frames"]),
        y_anchor=list(payload["y_anchor"]),
        y_replace=list(payload["y_replace"]),
        valid=list(payload["valid"]),
    )

def prune_spooled_targets(targets_dir: Path, needed: list[str]) -> list[str]:
    keep = set(needed)
    removed: list[str] = []
    for path in sorted(targets_dir.glob("*.json")):
        if path.stem not in keep:
            path.unlink()
            removed.append(path.stem)
    return removed


def write_spooled_target(
    targets_dir: Path, source_id: str, entry: dict[str, Any], session: Any
) -> dict[str, Any]:
    payload = serialize_authority(source_id, entry)
    payload["audio_ref"] = str(session.audio_ref)
    payload["waveform_sha256"] = str(session.waveform_sha256)
    text = json.dumps(payload, sort_keys=True)
    path = targets_dir / f"{source_id}.json"
    path.write_text(text, encoding="utf-8")
    rel = f"targets/{source_id}.json"
    return {
        "file": rel,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "num_frames": payload["num_frames"],
        "audio_ref": payload["audio_ref"],
        "waveform_sha256": payload["waveform_sha256"],
    }



def run_stage_a(
    corpus_root: Path,
    reference_root: Path,
    sampling_manifest: Path,
    out_dir: Path,
    workers: int | None = None,
) -> Path:
    from experiments.psem_sortformer_adaptation_depth.preflight import (
        SOURCE_MANIFEST_PATH,
    )
    from experiments.psem_sortformer_adaptation_depth.sampling import (
        load_training_sessions,
        validate_training_waveform_paths,
    )

    worker_count = resolve_worker_count(workers)
    print(f"[gate0] phase=a-start workers={worker_count}", flush=True)
    population = resolve_sampling_population(sampling_manifest)
    rows_by_source = dict(population["rows_by_source"])
    print("[gate0] phase=a-sessions", flush=True)
    sessions = load_training_sessions(corpus_root, reference_root)
    validate_training_waveform_paths(sessions, corpus_root)
    for source_id in (str(population["ami_source"]), str(population["alimeeting_source"])):
        if source_id not in sessions:
            raise MaterialError(f"slice source outside the frozen TRAIN split: {source_id}")
    print(
        f"[gate0] phase=a-target-build sources={len(sessions)} workers={worker_count}",
        flush=True,
    )
    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    aggregates: dict[str, dict[str, int]] = {}
    spooled_items: dict[str, dict[str, Any]] = {}
    spooled_entries: dict[str, dict[str, Any]] = {}

    def _spool(item: dict[str, Any]) -> None:
        source_id = str(item["source_id"])
        spooled_entries[source_id] = write_spooled_target(
            targets_dir, source_id, item, sessions[source_id]
        )
        spooled_items[source_id] = item
        authority = item["authority"]
        aggregates[source_id] = {
            "positive_frames": int(sum(authority.y_replace)),
            "negative_frames": int(
                sum(1 for v in authority.y_replace if v == 0)
            ),
        }

    from experiments.psem_state_corrected_adaptation_gate.material import (
        _build_source_targets_worker,
        _ordered_pool_map,
        build_target_payloads,
    )

    _ordered_pool_map(
        _build_source_targets_worker,
        build_target_payloads(sessions, rows_by_source),
        worker_count,
        on_result=_spool,
    )
    print(
        f"[gate0] phase=a-target-build-done sources={len(aggregates)}", flush=True
    )
    components = load_source_components(sessions, SOURCE_MANIFEST_PATH)
    source_rows = load_source_rows(SOURCE_MANIFEST_PATH)
    corpus_of: dict[str, str] = {}
    for source_id in sessions:
        manifest_corpus = source_rows.get(source_id, {}).get("corpus")
        row_corpus = next(
            (
                row.get("corpus")
                for row in rows_by_source.get(source_id, [])
                if row.get("corpus") in ("AMI", "AliMeeting")
            ),
            None,
        )
        corpus = row_corpus or manifest_corpus
        if corpus not in ("AMI", "AliMeeting"):
            raise MaterialError(f"slice corpus is unknown: {source_id}")
        corpus_of[source_id] = corpus
    exposure_rows = [
        SourceExposure(
            source_id=source_id,
            corpus=corpus_of[source_id],
            exposure=float(len(rows_by_source.get(source_id, []))),
            positive_frames=aggregates[source_id]["positive_frames"],
            negative_frames=aggregates[source_id]["negative_frames"],
        )
        for source_id in sessions
    ]
    assignment = assign_train_calib(exposure_rows, components=components)
    by_source = {row.source_id: row for row in exposure_rows}
    validate_partition_support(assignment, by_source)
    accumulator = ClassAccumulator()
    for source_id in assignment["fit"]:
        item = spooled_items.get(source_id)
        if item is None:
            stored = json.loads((targets_dir / f"{source_id}.json").read_text(encoding="utf-8"))
            accumulator.add(
                list(stored["y_replace"]),
                list(stored["y_anchor"]),
                list(stored["multiplicity"]),
                list(stored["valid"]),
            )
        else:
            authority = item["authority"]
            accumulator.add(
                [float(v) for v in authority.y_replace],
                [float(v) for v in authority.y_anchor],
                [int(v) for v in item["multiplicity"]],
                [bool(v) for v in authority.valid],
            )
    class_weights = accumulator.weights()
    slice_sources = select_fit_slice(assignment["fit"], rows_by_source, corpus_of)
    calib_sources = [s for s in assignment["calib"] if rows_by_source.get(s)]
    if not calib_sources:
        raise MaterialError("partition has no TRAIN-CALIB source with sampling exposure")
    needed = sorted(set(slice_sources) | set(calib_sources))
    pruned = prune_spooled_targets(targets_dir, needed)
    print(f"[gate0] phase=a-prune kept={len(needed)} removed={len(pruned)}", flush=True)
    files: dict[str, str] = {}
    manifest_targets: dict[str, Any] = {}
    for source_id in needed:
        entry = spooled_entries.get(source_id)
        if entry is None:
            stored = json.loads((targets_dir / f"{source_id}.json").read_text(encoding="utf-8"))
            rel = f"targets/{source_id}.json"
            files[rel] = sha256_file(targets_dir / f"{source_id}.json")
            manifest_targets[source_id] = {
                "file": rel,
                "sha256": files[rel],
                "num_frames": int(stored["num_frames"]),
                "audio_ref": str(stored["audio_ref"]),
                "waveform_sha256": str(stored["waveform_sha256"]),
            }
        else:
            rel = str(entry["file"])
            files[rel] = str(entry["sha256"])
            manifest_targets[source_id] = {
                "file": rel,
                "sha256": str(entry["sha256"]),
                "num_frames": int(entry["num_frames"]),
                "audio_ref": str(entry["audio_ref"]),
                "waveform_sha256": str(entry["waveform_sha256"]),
            }
    manifest = {
        "artifact_role": "issue-121-stage-a-bundle",
        "version": STAGE_A_VERSION,
        "nemo_sha256": NEMO_SHA256,
        "sampling_sha256": str(population["sampling_sha256"]),
        "fit": sorted(assignment["fit"]),
        "calib": sorted(assignment["calib"]),
        "class_weights": dict(class_weights),
        "slice_sources": list(slice_sources),
        "ami_source": str(population["ami_source"]),
        "alimeeting_source": str(population["alimeeting_source"]),
        "calib_sources": sorted(calib_sources),
        "targets": manifest_targets,
        "files": files,
    }
    manifest["target_frac"] = float(assignment["target_frac"])
    manifest["salt"] = str(assignment["salt"])
    manifest_path = out_dir / "stage_a_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[gate0] phase=a-done bundle={manifest_path}", flush=True)
    return manifest_path


def load_waveform_bytes(
    torchaudio: Any, corpus_root: Path, audio_ref: str, expected_sha256: str, source_id: str
) -> Any:
    path = (corpus_root.resolve() / audio_ref).resolve()
    if not path.is_file():
        raise MaterialError(f"source waveform missing: {source_id}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise MaterialError(f"source waveform hash differs: {source_id}")
    audio, sample_rate = torchaudio.load(str(path))
    return audio, sample_rate


def load_stage_targets(
    bundle_dir: Path, manifest: dict[str, Any], source_ids: list[str]
) -> dict[str, dict[str, Any]]:
    root = bundle_dir.resolve()
    entries = manifest.get("targets", {})
    if not isinstance(entries, dict):
        raise MaterialError("stage bundle target table is invalid")
    payloads: dict[str, dict[str, Any]] = {}
    for source_id in source_ids:
        meta = entries.get(source_id)
        if not isinstance(meta, dict):
            raise MaterialError(f"stage bundle target missing: {source_id}")
        rel = str(meta.get("file", ""))
        candidate = (root / rel).resolve()
        if root not in candidate.parents and candidate != root:
            raise MaterialError(f"stage bundle target escapes bundle: {source_id}")
        if not candidate.is_file():
            raise MaterialError(f"stage bundle target missing: {source_id}")
        if sha256_file(candidate) != meta.get("sha256"):
            raise MaterialError(f"stage bundle target hash differs: {source_id}")
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise MaterialError(f"stage bundle target is invalid: {source_id}")
        for key in ("num_frames", "audio_ref", "waveform_sha256"):
            if payload.get(key) != meta.get(key):
                raise MaterialError(f"stage bundle target metadata differs: {source_id}")
        if payload.get("source_id") != source_id:
            raise MaterialError(f"stage bundle target metadata differs: {source_id}")
        try:
            frames = int(payload.get("num_frames", -1))
        except (TypeError, ValueError):
            raise MaterialError(f"stage bundle target geometry is invalid: {source_id}")
        arrays = [
            payload.get("y_anchor"),
            payload.get("y_replace"),
            payload.get("valid"),
            payload.get("multiplicity"),
            payload.get("episode_ids"),
        ]
        if frames <= 0 or any(not isinstance(values, list) for values in arrays):
            raise MaterialError(f"stage bundle target geometry is invalid: {source_id}")
        if any(len(values) != frames for values in arrays):
            raise MaterialError(f"stage bundle target geometry is invalid: {source_id}")
        payloads[source_id] = payload
    return payloads


def run_stage_b(
    bundle_dir: Path,
    checkpoint: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    corpus_root: Path,
    reference_root: Path,
    device: str,
    out_dir: Path,
    workers: int | None = None,
) -> Path:
    from experiments.psem_state_corrected_adaptation_gate.material import _require_torch

    worker_count = resolve_worker_count(workers)
    print(f"[gate0] phase=b-start workers={worker_count}", flush=True)
    manifest = verify_bundle_manifest(bundle_dir, "stage_a_manifest.json")
    slice_ids = manifest.get("slice_sources")
    calib_ids = manifest.get("calib_sources")
    if (
        not isinstance(slice_ids, list)
        or not isinstance(calib_ids, list)
        or not all(isinstance(s, str) for s in slice_ids + calib_ids)
    ):
        raise MaterialError("stage bundle source lists are invalid")
    needed = list(slice_ids) + [s for s in calib_ids if s not in slice_ids]
    payloads = load_stage_targets(bundle_dir, manifest, needed)
    torch = _require_torch()
    print("[gate0] phase=b-model-load", flush=True)
    try:
        import torchaudio

        from experiments.psem_sortformer_adaptation_depth.execution import (
            load_scoring_sessions,
        )
        from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
            load_pinned_sortformer,
        )
        from experiments.psem_state_corrected_adaptation_gate import head as head_mod
        from experiments.psem_state_corrected_adaptation_gate.material import (
            infer_arm_logits,
            infer_dev_raw_logits,
            infer_slice_source_evidence,
            mask_calibration,
            run_profiler,
            run_slice_update,
            slice_waveform_frames,
        )
        from experiments.psem_training_strategy_gate.sampling import DEV_ROLE
    except ImportError as exc:
        from experiments.psem_state_corrected_adaptation_gate.material import (
            MaterialBlockedError,
        )

        raise MaterialBlockedError(
            "material execution requires the pinned worker runtime"
        ) from exc
    wrapper, runtime_receipt = load_pinned_sortformer(
        checkpoint, nemo_checkout, dependency_lock, device
    )
    json.dumps(runtime_receipt)
    wrapper.eval()
    device_obj = next(wrapper.parameters()).device
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    head_module = head_mod.ResidualPSEMHead(199)
    head_module.to(device_obj)
    head_module.train(True)
    dropout_training = [
        bool(module.training)
        for module in wrapper.modules()
        if isinstance(module, torch.nn.Dropout)
    ]
    modes = audit_module_modes(
        bool(wrapper.training),
        dropout_training,
        bool(head_module.training),
        [name for name, p in wrapper.named_parameters() if p.requires_grad],
        [name for name, p in head_module.named_parameters() if p.requires_grad],
    )
    from experiments.psem_state_corrected_adaptation_gate.material import (
        SAMPLE_RATE_HZ,
    )

    per_source: dict[str, dict[str, Any]] = {}
    print(
        f"[gate0] phase=b-slice-inference sources={manifest['slice_sources']}",
        flush=True,
    )
    for source_id in manifest["slice_sources"]:
        entry = payloads[source_id]
        if sum(entry["multiplicity"]) == 0:
            raise MaterialError(f"slice source has no sampling exposure: {source_id}")
        audio, sample_rate = load_waveform_bytes(
            torchaudio, corpus_root, entry["audio_ref"], entry["waveform_sha256"], source_id
        )
        if sample_rate != SAMPLE_RATE_HZ or audio.ndim != 2 or audio.shape[0] != 1:
            raise MaterialError(f"source waveform geometry is invalid: {source_id}")
        authority = restore_authority(entry)
        frame_count = int(entry["num_frames"])
        usable_samples, tail_excluded = slice_waveform_frames(
            int(audio.shape[1]), frame_count, source_id
        )
        waveform = audio[:, :usable_samples].to(device_obj)
        per_source[source_id] = infer_slice_source_evidence(
            torch,
            wrapper,
            waveform,
            authority,
            list(entry["multiplicity"]),
            [None if v is None else str(v) for v in entry["episode_ids"]],
            frame_count,
            tail_excluded,
            source_id,
            device_obj,
        )
    train_source = manifest["slice_sources"][0]
    train = per_source[train_source]
    update_ctx = run_slice_update(
        torch, wrapper, head_module, train, dict(manifest["class_weights"]), device_obj, train_source
    )
    head_module.eval()
    print(
        f"[gate0] phase=b-calib-inference sources={manifest['calib_sources']}",
        flush=True,
    )
    calib_meta: dict[str, Any] = {}
    kept_total = 0
    npz_dir = out_dir / "stage_b_arrays"
    npz_dir.mkdir(parents=True, exist_ok=True)
    for source_id in manifest["calib_sources"]:
        entry = payloads[source_id]
        audio, sample_rate = load_waveform_bytes(
            torchaudio, corpus_root, entry["audio_ref"], entry["waveform_sha256"], source_id
        )
        if sample_rate != SAMPLE_RATE_HZ or audio.ndim != 2 or audio.shape[0] != 1:
            raise MaterialError(f"CALIB waveform geometry is invalid: {source_id}")
        frame_count = int(entry["num_frames"])
        usable_samples, _ = slice_waveform_frames(int(audio.shape[1]), frame_count, source_id)
        waveform = audio[:, :usable_samples].to(device_obj)
        anchor = [a == 1.0 for a in entry["y_anchor"]]
        valid = [bool(v) for v in entry["valid"]]
        with torch.no_grad():
            calib_out = infer_arm_logits(
                torch,
                wrapper,
                head_module,
                waveform,
                [None if v is None else str(v) for v in entry["episode_ids"]],
                anchor,
                valid,
                frame_count,
                device_obj,
            )
        calib_unmapped = set(calib_out["unmapped_frames"])
        mapped = [
            i not in calib_unmapped for i in range(frame_count)
        ]
        calib_targets_all = [float(v) for v in entry["y_replace"]]
        kept, coverage = mask_calibration(calib_targets_all, valid, mapped)
        kept_total += len(kept)
        f0_all = calib_out["f0_logit"].flatten().tolist()
        cand_all = (calib_out["f0_logit"] + calib_out["z_residual"]).flatten().tolist()
        _write_raw_npz(
            npz_dir / f"calib_{source_id}.npz",
            f0_all,
            cand_all,
            calib_targets_all,
            valid,
            mapped,
        )
        calib_meta[source_id] = {
            "file": f"stage_b_arrays/calib_{source_id}.npz",
            "frames": frame_count,
            "kept": len(kept),
            "coverage": dict(coverage),
            "mapping_mapped": sum(1 for r in calib_out["mapping_rows"] if r["status"] == "mapped"),
            "mapping_total": len(calib_out["mapping_rows"]),
        }
    if kept_total == 0:
        raise MaterialError("TRAIN-CALIB inference produced no frames")
    print("[gate0] phase=b-dev-inference", flush=True)
    from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import load_sessions

    dev_runtime = load_scoring_sessions(corpus_root, reference_root, DEV_ROLE)
    dev_sessions = load_sessions()
    dev_meta: dict[str, Any] = {}
    dev_timings: dict[str, float] = {}
    for family in DEV_FAMILIES:
        candidates = [s for s in dev_sessions if is_dev_family_session(s, family)]
        if not candidates:
            raise MaterialError(f"DEV snapshot has no {family} session")
        dev = candidates[0]
        if dev.source_id not in dev_runtime:
            raise MaterialError(f"DEV session outside the frozen DEV split: {dev.source_id}")
        raw = infer_dev_raw_logits(
            torch, wrapper, head_module, dev, dev_runtime[dev.source_id], corpus_root, device_obj
        )
        _write_raw_npz(
            npz_dir / f"dev_{dev.source_id}.npz",
            raw["f0_raw"],
            raw["cand_raw"],
            raw["target"],
            raw["valid"],
            raw["mapped_flags"],
        )
        dev_timings[dev.source_id] = float(raw["infer_seconds"])
        dev_meta[dev.source_id] = {
            "file": f"stage_b_arrays/dev_{dev.source_id}.npz",
            "family": family,
            "frames": int(raw["grid_frames"]),
            "mapping_mapped": int(raw["mapping_mapped"]),
            "mapping_total": len(raw["mapping_rows"]),
            "unmapped_frames": len(raw["unmapped_frames"]),
            "kept_frames": int(raw["coverage"]["kept"]),
            "coverage": {k: int(v) for k, v in raw["coverage"].items()},
            "infer_seconds": float(raw["infer_seconds"]),
        }
    profiler = run_profiler(
        torch, head_module, train, update_ctx, dict(manifest["class_weights"]), device_obj, dev_timings
    )
    bundle_manifest_path = bundle_dir / "stage_a_manifest.json"
    manifest_out = {
        "artifact_role": "issue-121-stage-b-evidence",
        "version": STAGE_B_VERSION,
        "stage_a_manifest": "stage_a_manifest.json",
        "stage_a_sha256": sha256_file(bundle_manifest_path),
        "checkpoint_sha256": str(runtime_receipt["checkpoint_sha256"]),
        "device": str(device_obj),
        "worker_count": worker_count,
        "slice_sources": list(manifest["slice_sources"]),
        "calib_sources": list(manifest["calib_sources"]),
        "dev_sources": [source_id for source_id in dev_meta],
        "equivalence": {
            source_id: {
                name: float(value)
                for name, value in per_source[source_id]["equivalence"].items()
            }
            for source_id in manifest["slice_sources"]
        },
        "mapping": {
            source_id: {
                "mapped": int(per_source[source_id]["mapping_mapped"]),
                "total": len(per_source[source_id]["mapping_rows"]),
            }
            for source_id in manifest["slice_sources"]
        },
        "identity_diff": float(update_ctx["identity_diff"]),
        "update": {"gradients_finite": True, "head_updated": True, "frozen_unchanged": True},
        "profiler": dict(profiler),
        "modes": dict(modes),
        "runtime_receipt": json.loads(json.dumps(runtime_receipt)),
        "calib": calib_meta,
        "dev": dev_meta,
        "files": {},
    }
    files: dict[str, str] = {}
    for rel in sorted(
        [calib_meta[s]["file"] for s in calib_meta]
        + [dev_meta[s]["file"] for s in dev_meta]
    ):
        files[rel] = sha256_file(out_dir / rel)
    manifest_out["files"] = files
    manifest_path = out_dir / "stage_b_manifest.json"
    write_json(manifest_path, manifest_out)
    print(f"[gate0] phase=b-done manifest={manifest_path}", flush=True)
    return manifest_path


def _write_raw_npz(
    path: Path,
    f0_raw: list[float],
    cand_raw: list[float],
    target: list[float],
    valid: list[bool],
    mapped: list[bool],
) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        f0_raw=np.asarray(f0_raw, dtype=np.float64),
        cand_raw=np.asarray(cand_raw, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
        mapped=np.asarray(mapped, dtype=bool),
    )


def fit_calibrators(
    f0_raw: list[float], cand_raw: list[float], targets: list[float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    f0_fit = calibrate_mod.fit_affine_calibrator(list(f0_raw), list(targets), "TRAIN-CALIB")
    cand_fit = calibrate_mod.fit_affine_calibrator(list(cand_raw), list(targets), "TRAIN-CALIB")
    return f0_fit, cand_fit


def _prepare_dev_arrays(
    dev: Any,
    f0_raw: list[float],
    cand_raw: list[float],
    target: list[float],
    valid: list[bool],
    mapped_flags: list[bool],
    unmapped: list[int],
    cal_f0: dict[str, Any],
    cal_cand: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from experiments.psem_state_corrected_adaptation_gate.material import (
        mask_calibration,
    )

    kept, coverage = mask_calibration(list(target), list(valid), list(mapped_flags))
    f0_cal = calibrate_mod.apply_affine(list(f0_raw), float(cal_f0["slope"]), float(cal_f0["intercept"]))
    cand_cal = calibrate_mod.apply_affine(
        list(cand_raw), float(cal_cand["slope"]), float(cal_cand["intercept"])
    )
    f0_np = np.asarray([calibrate_mod.sigmoid(z) for z in f0_cal], dtype=np.float64)
    cand_np = np.asarray([calibrate_mod.sigmoid(z) for z in cand_cal], dtype=np.float64)
    unmapped_set = set(int(i) for i in unmapped)
    f0_np[sorted(unmapped_set)] = float("-inf")
    cand_np[sorted(unmapped_set)] = float("-inf")
    kept_target = [float(target[i]) for i in kept]
    cand_sigmoid = [calibrate_mod.sigmoid(z) for z in cand_raw]
    kept_cand_raw = [cand_sigmoid[i] for i in kept]
    kept_cand_cal = [cand_cal[i] for i in kept]
    kept_f0_cal = [f0_cal[i] for i in kept]
    raw_ap = calibrate_mod.average_precision(kept_cand_raw, kept_target)
    return {
        "f0_scores": f0_np.tolist(),
        "cand_scores": cand_np.tolist(),
        "thresholds": frontier_mod.unique_thresholds(cand_np.tolist()),
        "kept_target": kept_target,
        "kept_cand_raw": kept_cand_raw,
        "kept_cand_cal": kept_cand_cal,
        "kept_f0_cal": kept_f0_cal,
        "raw_ap": raw_ap,
        "kept": list(kept),
        "coverage": dict(coverage),
        "f0_cal": list(f0_cal),
        "cand_cal": list(cand_cal),
    }


def score_dev_frontiers(
    dev: Any,
    f0_raw: list[float],
    cand_raw: list[float],
    target: list[float],
    valid: list[bool],
    mapped_flags: list[bool],
    unmapped: list[int],
    cal_f0: dict[str, Any],
    cal_cand: dict[str, Any],
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    prep = _prepare_dev_arrays(
        dev, f0_raw, cand_raw, target, valid, mapped_flags, unmapped, cal_f0, cal_cand
    )
    f0_np = np.asarray(prep["f0_scores"], dtype=np.float64)
    horizon_results: dict[int, dict[str, Any]] = {}
    print(
        f"[gate0] scorer thresholds={len(prep['thresholds'])} workers={workers}",
        flush=True,
    )
    from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
        decode_scores,
        session_metrics,
    )
    from experiments.psem_state_corrected_adaptation_gate.material import (
        REQUIRED_HORIZONS,
        _frontier_point,
        candidate_frontier_points_multi,
    )

    multi_points = candidate_frontier_points_multi(
        dev, prep["cand_scores"], prep["thresholds"], list(REQUIRED_HORIZONS), workers
    )
    for horizon_ms in REQUIRED_HORIZONS:
        f0_events = decode_scores(dev, f0_np, threshold=0.5, confirmation_ms=horizon_ms)
        f0_point = _frontier_point(session_metrics(dev, f0_events))
        candidate_points = multi_points[horizon_ms]
        envelopes = frontier_mod.select_envelopes(f0_point, candidate_points)
        horizon_results[horizon_ms] = {
            "f0_point": f0_point,
            "candidate_points": candidate_points,
            "envelopes": envelopes,
            "raw_ap": prep["raw_ap"],
            "kept_target": prep["kept_target"],
            "kept_cand_cal": prep["kept_cand_cal"],
            "kept_f0_cal": prep["kept_f0_cal"],
            "coverage": dict(prep["coverage"]),
        }
    dev_entry = {
        "f0_cal": list(prep["f0_cal"]),
        "cand_cal": list(prep["cand_cal"]),
        "kept": list(prep["kept"]),
    }
    return horizon_results, dev_entry


def resolve_dev_session(dev_sessions: Any, source_id: str) -> Any:
    session = dev_sessions.get(source_id)
    if session is None or str(getattr(session, "role", "")).lower() != "dev":
        raise MaterialError(f"DEV session outside the frozen DEV split: {source_id}")
    return session


def run_stage_c(
    bundle_dir: Path, stage_b_dir: Path, out_dir: Path, workers: int | None = None
) -> Path:
    import numpy as np

    from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import load_sessions
    from experiments.psem_state_corrected_adaptation_gate.material import (
        build_horizon_result,
        mask_calibration,
    )

    worker_count = resolve_worker_count(workers)
    print(f"[gate0] phase=c-start workers={worker_count}", flush=True)
    manifest_a = verify_bundle_manifest(bundle_dir, "stage_a_manifest.json")
    manifest_b = verify_bundle_manifest(stage_b_dir, "stage_b_manifest.json")
    if manifest_b["stage_a_sha256"] != sha256_file(bundle_dir / "stage_a_manifest.json"):
        raise MaterialError("stage B was built from a different stage A bundle")
    print("[gate0] phase=c-calibration", flush=True)
    calib_f0_raw: list[float] = []
    calib_cand_raw: list[float] = []
    calib_targets: list[float] = []
    for source_id in manifest_b["calib_sources"]:
        entry = manifest_b["calib"][source_id]
        arrays = np.load(stage_b_dir / entry["file"])
        valid = [bool(v) for v in arrays["valid"].tolist()]
        mapped = [bool(v) for v in arrays["mapped"].tolist()]
        targets = [float(v) for v in arrays["target"].tolist()]
        f0_all = [float(v) for v in arrays["f0_raw"].tolist()]
        cand_all = [float(v) for v in arrays["cand_raw"].tolist()]
        kept, _ = mask_calibration(targets, valid, mapped)
        if len(kept) != int(entry["kept"]):
            raise MaterialError(f"CALIB kept frames differ: {source_id}")
        for index in kept:
            calib_f0_raw.append(f0_all[index])
            calib_cand_raw.append(cand_all[index])
            calib_targets.append(targets[index])
    if not calib_f0_raw:
        raise MaterialError("TRAIN-CALIB inference produced no frames")
    calibration_f0, calibration_candidate = fit_calibrators(calib_f0_raw, calib_cand_raw, calib_targets)
    print("[gate0] phase=c-frontier", flush=True)
    dev_sessions = {s.source_id: s for s in load_sessions()}
    dev_points: dict[str, Any] = {}
    predictions_meta: dict[str, dict[str, str]] = {}
    from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
        decode_scores,
        session_metrics,
    )
    from experiments.psem_state_corrected_adaptation_gate.material import (
        REQUIRED_HORIZONS,
        _frontier_point,
        candidate_frontier_points_sessions,
    )

    print("[gate0] phase=c-frontier-prep", flush=True)
    preps: dict[str, Any] = {}
    sessions_by_id: dict[str, Any] = {}
    raws: dict[str, Any] = {}
    for source_id in manifest_b["dev_sources"]:
        session = resolve_dev_session(dev_sessions, source_id)
        entry = manifest_b["dev"][source_id]
        arrays = np.load(stage_b_dir / entry["file"])
        f0_raw = [float(v) for v in arrays["f0_raw"].tolist()]
        cand_raw = [float(v) for v in arrays["cand_raw"].tolist()]
        target = [float(v) for v in arrays["target"].tolist()]
        valid = [bool(v) for v in arrays["valid"].tolist()]
        mapped_flags = [bool(v) for v in arrays["mapped"].tolist()]
        unmapped = [int(i) for i in range(len(target)) if not mapped_flags[i]]
        preps[source_id] = _prepare_dev_arrays(
            session,
            f0_raw,
            cand_raw,
            target,
            valid,
            mapped_flags,
            unmapped,
            calibration_f0,
            calibration_candidate,
        )
        sessions_by_id[source_id] = session
        raws[source_id] = {"f0_raw": f0_raw, "cand_raw": cand_raw, "target": target}
    jobs = [
        {
            "key": source_id,
            "dev": sessions_by_id[source_id],
            "scores": preps[source_id]["cand_scores"],
            "thresholds": preps[source_id]["thresholds"],
            "horizons": list(REQUIRED_HORIZONS),
        }
        for source_id in manifest_b["dev_sources"]
    ]
    print(
        f"[gate0] phase=c-frontier sources={len(jobs)} workers={worker_count}",
        flush=True,
    )
    points_map = candidate_frontier_points_sessions(jobs, worker_count)
    for source_id in manifest_b["dev_sources"]:
        import numpy as _np

        session = sessions_by_id[source_id]
        prep = preps[source_id]
        entry = manifest_b["dev"][source_id]
        f0_np = _np.asarray(prep["f0_scores"], dtype=_np.float64)
        horizon_results: dict[int, dict[str, Any]] = {}
        for horizon_ms in REQUIRED_HORIZONS:
            f0_events = decode_scores(session, f0_np, threshold=0.5, confirmation_ms=horizon_ms)
            f0_point = _frontier_point(session_metrics(session, f0_events))
            candidate_points = points_map[source_id][horizon_ms]
            envelopes = frontier_mod.select_envelopes(f0_point, candidate_points)
            horizon_results[horizon_ms] = {
                "f0_point": f0_point,
                "candidate_points": candidate_points,
                "envelopes": envelopes,
                "raw_ap": prep["raw_ap"],
                "kept_target": prep["kept_target"],
                "kept_cand_cal": prep["kept_cand_cal"],
                "kept_f0_cal": prep["kept_f0_cal"],
                "coverage": dict(prep["coverage"]),
            }
        assembled: dict[int, dict[str, Any]] = {}
        for horizon_ms, result in horizon_results.items():
            assembled[horizon_ms] = build_horizon_result(
                result["f0_point"],
                result["candidate_points"],
                result["envelopes"],
                int(entry["mapping_mapped"]),
                int(entry["mapping_total"]),
                int(entry["unmapped_frames"]),
                int(entry["kept_frames"]),
                float(result["raw_ap"]),
                result["kept_cand_cal"],
                result["kept_target"],
                result["kept_f0_cal"],
            )
        dev_points[source_id] = assembled
        predictions_path = out_dir / f"raw_predictions_{source_id}.npz"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            predictions_path,
            f0_logit=np.asarray(raws[source_id]["f0_raw"], dtype=np.float64),
            candidate_logit=np.asarray(raws[source_id]["cand_raw"], dtype=np.float64),
            f0_calibrated=np.asarray(prep["f0_cal"], dtype=np.float64),
            candidate_calibrated=np.asarray(prep["cand_cal"], dtype=np.float64),
            target=np.asarray(raws[source_id]["target"], dtype=np.float64),
        )
        predictions_meta[source_id] = {
            "path": predictions_path.name,
            "sha256": sha256_file(predictions_path),
        }
    record = build_gate0_record(
        manifest_a["slice_sources"],
        dict(manifest_b["runtime_receipt"]),
        str(manifest_a["sampling_sha256"]),
        dict(manifest_b["modes"]),
        {
            "fit": list(manifest_a["fit"]),
            "calib": list(manifest_a["calib"]),
            "salt": str(manifest_a["salt"]),
            "target_frac": float(manifest_a["target_frac"]),
        },
        dict(manifest_a["class_weights"]),
        {
            source_id: {
                key: int(value)
                for key, value in manifest_b["calib"][source_id]["coverage"].items()
            }
            for source_id in manifest_b["calib_sources"]
        },
        dict(calibration_candidate),
        dict(calibration_f0),
        dev_points,
        dict(manifest_b["profiler"]),
        predictions_meta,
    )
    write_gate0_artifacts(
        out_dir,
        record,
        {
            "fit": list(manifest_a["fit"]),
            "calib": list(manifest_a["calib"]),
            "salt": str(manifest_a["salt"]),
            "target_frac": float(manifest_a["target_frac"]),
        },
        dict(manifest_a["class_weights"]),
        dict(manifest_b["modes"]),
    )
    print(f"[gate0] phase=c-done verdict={record['verdict']}", flush=True)
    return out_dir / "material_vertical_slice.json"
