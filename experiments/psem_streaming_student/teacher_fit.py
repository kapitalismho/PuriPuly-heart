from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import argparse
import csv
import ctypes
import itertools
import json
import subprocess
import sys
import tempfile
import time
import wave as wave_mod
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gt_pilot
import run_baseline
import speaker_score_diagnostic as ssd
from gt_probe import atomic_json, local_path, read_wave, sha256_file

SCHEMA = "PSEM-TEACHER-FIT-CONFIG-1"
SIDECAR_SCHEMA = "PSEM-TEACHER-FIT-SIDECAR-1"
MANIFEST_SCHEMA = "PSEM-TEACHER-FIT-MANIFEST-1"
RESULT_SCHEMA = "PSEM-TEACHER-FIT-TARGET-RESULT-1"
RELEASE_SCHEMA = "PSEM-DIRECTOR-MATERIAL-RELEASE-1"
OUTPUT_SLOTS = 4
HOP = 1280
SAMPLE_RATE = 16000
PCM_SCALE = 32768
LOGIT_CLIP = 1.0e-12
FIT_IDS = ("ami_ES2005a", "ami_ES2006a", "ami_ES2007a", "ami_ES2008a")
MATERIAL_AUTHORIZED = False


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def repo_path(relative: str) -> Path:
    require("\\" not in relative and not relative.startswith("/"), f"path must be POSIX-relative: {relative}")
    return ROOT.joinpath(*relative.split("/"))


def digest_file(path: Path) -> str:
    return sha256_file(path)


def publish_json(path: Path, value: object) -> None:
    def ready(item: object) -> Any:
        if isinstance(item, dict):
            return {str(key): ready(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [ready(child) for child in item]
        if isinstance(item, np.ndarray):
            return ready(item.tolist())
        if isinstance(item, np.generic):
            return ready(item.item())
        if isinstance(item, Path):
            return item.as_posix()
        return ssd.json_safe(item)

    atomic_json(path, ready(value))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_teacher_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    require(config.get("schema") == SCHEMA, "unexpected teacher-fit config schema")
    require(config.get("experiment_id") == "TEACHER-FIT-TARGETS-1", "unexpected experiment_id")
    require(config.get("automatic_retry") is False, "automatic retry must remain false")
    require(config.get("material_execution_requires_director_release") is True, "director release gate missing")
    source_ids = tuple(row["source_id"] for row in config["sources"])
    require(source_ids == FIT_IDS, "teacher sources must be the four FIT source-zero prefixes")
    prefixes = [int(row["prefix_samples"]) for row in config["sources"]]
    require(prefixes == [4620800, 5107200, 4620800, 4620800], "FIT prefix_samples mismatch")
    require(int(config["geometry"]["hop_samples"]) == HOP, "hop_samples must be 1280")
    require(int(config["geometry"]["output_slots"]) == OUTPUT_SLOTS, "output_slots must be 4")
    require(int(config["geometry"]["sample_rate"]) == SAMPLE_RATE, "sample_rate must be 16000")
    require(int(config["geometry"]["source_start_sample"]) == 0, "source_start_sample must be 0")
    require(config["native"]["mode"] == "unpaced_source_zero_streaming", "only unpaced streaming is authorized")
    require("TRANSCRIBE_SORTFORMER_OFFLINE_DUMP" in config["native"]["absent_environment"], "OFFLINE_DUMP must stay absent")
    require(config["native"]["required_environment"].get("TRANSCRIBE_SORTFORMER_EXPORT") == "logits", "future EXPORT must be logits")
    require("TRANSCRIBE_DUMP_HIDDEN" in config["native"]["absent_environment"], "DUMP_HIDDEN must stay absent")
    require("TRANSCRIBE_SORTFORMER_COMPRESS" in config["native"]["absent_environment"], "COMPRESS must stay absent")
    require(float(config["association"]["logit_clip"]) == LOGIT_CLIP, "association clip must remain 1e-12")
    require(config["association"]["raw_kd_probabilities_clipped"] is False, "KD probabilities must not be clipped")
    require(config["association"]["raw_kd_probabilities_softmaxed"] is False, "KD probabilities must not be softmaxed")
    require(int(config["bounds"]["optimizer_updates"]) == 0, "optimizer updates are not authorized")
    require(int(config["bounds"]["cumulative_optimizer_updates_unchanged"]) == 1583, "cumulative optimizer count must stay 1583")
    require(int(config["bounds"]["orchestrator_pytorch_model_construction"]) == 0, "orchestrator PyTorch model construction is not authorized")
    require(int(config["bounds"]["new_native_source_trajectories"]) == 4, "native teacher trajectories must remain 4")
    return config


def verify_pinned_files(config: dict[str, Any], executed_config_path: Path) -> dict[str, str]:
    executed_config_path = executed_config_path.resolve()
    hashes = {
        "contract": digest_file(repo_path(config["contract"]["path"])),
        "native_prerequisites": digest_file(repo_path(config["native_prerequisites"]["path"])),
        "original_config": digest_file(repo_path(config["inputs"]["original_config"])),
        "prepared_sources": digest_file(repo_path(config["inputs"]["prepared_sources"])),
        "speaker_score_diagnostic.py": digest_file(HERE / "speaker_score_diagnostic.py"),
        "gt_pilot.py": digest_file(HERE / "gt_pilot.py"),
        "run_baseline.py": digest_file(HERE / "run_baseline.py"),
        "gt_probe.py": digest_file(HERE / "gt_probe.py"),
        "teacher_fit.py": digest_file(HERE / "teacher_fit.py"),
        "config": digest_file(executed_config_path),
        "executed_config_path": str(executed_config_path),
    }
    require(hashes["contract"] == config["contract"]["sha256"], "teacher-fit contract hash mismatch")
    require(hashes["native_prerequisites"] == config["native_prerequisites"]["sha256"], "native prerequisites hash mismatch")
    require(hashes["original_config"] == config["inputs"]["original_config_sha256"], "original frozen config hash mismatch")
    require(hashes["prepared_sources"] == config["inputs"]["prepared_sources_sha256"], "prepared_sources.pt hash mismatch")
    for name, expected in config["reviewed_helpers"].items():
        require(hashes[name] == expected, f"reviewed helper hash mismatch for {name}")
    return hashes


def verify_native_identities(config: dict[str, Any]) -> dict[str, Any]:
    executable = local_path(config["native"]["executable"])
    model = local_path(config["native"]["model"])
    require(executable.is_file(), f"missing native executable: {executable}")
    require(model.is_file(), f"missing native model: {model}")
    exe_sha = digest_file(executable)
    model_sha = digest_file(model)
    require(exe_sha == config["native"]["executable_sha256"], "native executable hash mismatch")
    require(model_sha == config["native"]["model_sha256"], "native model hash mismatch")
    return {
        "executable": str(executable),
        "executable_sha256": exe_sha,
        "executable_bytes": executable.stat().st_size,
        "model": str(model),
        "model_sha256": model_sha,
        "model_bytes": model.stat().st_size,
        "backend": config["native"]["backend"],
    }


def load_origin_config(config: dict[str, Any]) -> dict[str, Any]:
    origin = load_json(repo_path(config["inputs"]["original_config"]))
    require(origin.get("schema") == gt_pilot.SCHEMA, "original config is not the GT-pilot frozen schema")
    return origin


def bind_fit_identities(config: dict[str, Any], origin: dict[str, Any]) -> dict[str, Any]:
    catalog = gt_pilot.catalog_sources(origin)
    identities = gt_pilot.verify_pilot_identities(origin)
    locked = set(origin["locked_identity_closure"]["source_ids"])
    rows = []
    for row in config["sources"]:
        source_id = row["source_id"]
        source = catalog[source_id]
        require(source["pilot_role"] == "FIT", f"{source_id} is not a FIT source")
        require(source["role"] == "PSEM-STRATEGY-TRAIN", f"{source_id} is not TRAIN")
        require(int(source["prefix_samples"]) == int(row["prefix_samples"]), f"{source_id} prefix mismatch")
        require(source_id not in locked, f"{source_id} is inside the locked identity closure")
        resolved = gt_pilot.resolve_frozen_source(origin, source)
        require(resolved["component"]["role"] == "PSEM-STRATEGY-TRAIN", f"{source_id} split role is not TRAIN")
        wav = local_path(source["waveform"])
        require(wav.is_file(), f"missing waveform {wav}")
        wav_sha = digest_file(wav)
        require(wav_sha == source["waveform_sha256"], f"{source_id} waveform hash mismatch")
        rows.append(
            {
                "source_id": source_id,
                "pilot_role": "FIT",
                "meeting": source["meeting"],
                "prefix_samples": int(source["prefix_samples"]),
                "nominal_frames": int(source["prefix_samples"]) // HOP,
                "waveform": str(wav),
                "waveform_sha256": wav_sha,
                "component_id": source["component_id"],
                "slots_pending_prepared": True,
            }
        )
    require([row["source_id"] for row in rows] == list(FIT_IDS), "FIT identity order mismatch")
    for role in ("CAL", "DEV"):
        for source in origin["sources"][role]:
            require(source["source_id"] not in {row["source_id"] for row in rows}, f"{role} source leaked into teacher FIT list")
    return {"sources": rows, "pilot_identities": identities, "locked_identity_closure": origin["locked_identity_closure"]}


def load_prepared(config: dict[str, Any]) -> dict[str, Any]:
    path = repo_path(config["inputs"]["prepared_sources"])
    require(digest_file(path) == config["inputs"]["prepared_sources_sha256"], "prepared_sources.pt changed during load")
    prepared = torch.load(path, map_location="cpu", weights_only=False)
    require("sources" in prepared and "fit_constant_prior" in prepared, "prepared artifact missing sources")
    return prepared


def verify_prepared_fit(config: dict[str, Any], prepared: dict[str, Any], origin: dict[str, Any]) -> dict[str, Any]:
    expected_solo = config["fit_strict_solo_counts_before_teacher_tail"]
    catalog = gt_pilot.catalog_sources(origin)
    report = {}
    for row in config["sources"]:
        source_id = row["source_id"]
        payload = prepared["sources"][source_id]
        require(payload["source"]["pilot_role"] == "FIT", f"{source_id} prepared role is not FIT")
        require(int(payload["source"]["prefix_samples"]) == int(row["prefix_samples"]), f"{source_id} prepared prefix mismatch")
        require(payload["source"]["source_id"] == source_id, f"{source_id} prepared source_id mismatch")
        require(int(payload["waveform"].numel()) == int(row["prefix_samples"]), f"{source_id} prepared waveform length")
        require(payload["waveform"].device.type == "cpu", f"{source_id} prepared waveform must stay on CPU")
        frames = int(row["prefix_samples"]) // HOP
        targets = payload["targets"]
        validity = payload["validity"]
        frontiers = payload["frontiers"]
        require(tuple(targets.shape) == (frames, OUTPUT_SLOTS), f"{source_id} target shape")
        require(tuple(validity.shape) == (frames,), f"{source_id} validity shape")
        require(tuple(frontiers.shape) == (frames,), f"{source_id} frontier shape")
        require(int(frontiers[0].item()) == HOP, f"{source_id} first frontier")
        require(int(frontiers[-1].item()) == int(row["prefix_samples"]), f"{source_id} last frontier")
        require(int((frontiers[1] - frontiers[0]).item()) == HOP, f"{source_id} hop")
        require(all(slot is not None for slot in payload["slots"]), f"{source_id} FIT slots incomplete")
        target_np = ssd.as_numpy(targets)
        valid_np = ssd.as_numpy(validity).astype(bool)
        solo = ssd.solo_partition(target_np, valid_np)
        counts = [int((solo["solo_mask"] & (solo["true_slot"] == slot)).sum()) for slot in range(OUTPUT_SLOTS)]
        require(counts == list(expected_solo[source_id]), f"{source_id} prepared strict-solo counts changed")
        require(catalog[source_id]["waveform_sha256"] == payload["source"]["waveform_sha256"], f"{source_id} prepared waveform identity")
        report[source_id] = {
            "frames": frames,
            "gt_activity_valid_bins": int(valid_np.sum()),
            "strict_solo_counts_before_teacher_tail": counts,
            "slots": list(payload["slots"]),
            "not_an_acceptance_threshold": True,
        }
    return report

def prefix_export_matches_prepared(source_wav: Path, prepared_wave: torch.Tensor, prefix_samples: int) -> dict[str, Any]:
    wave, geometry = read_wave(source_wav, prefix_samples, SAMPLE_RATE, 1)
    require(torch.equal(wave, prepared_wave.detach().cpu()), "prepared waveform is not PCM16/32768 of the source prefix")
    with tempfile.TemporaryDirectory(prefix="teacher-fit-prefix-") as temp:
        exported = Path(temp) / "prefix.wav"
        run_baseline.make_projection(source_wav, exported, prefix_samples)
        exported_wave, _exported_geometry = read_wave(exported, prefix_samples, SAMPLE_RATE, 1)
        require(torch.equal(exported_wave, wave), "exported prefix waveform differs from prepared PCM16/32768")
        with wave_mod.open(str(source_wav), "rb") as reader:
            source_pcm = reader.readframes(prefix_samples)
        with wave_mod.open(str(exported), "rb") as reader:
            require(reader.getframerate() == SAMPLE_RATE, "exported prefix rate")
            require(reader.getnchannels() == 1, "exported prefix must be mono")
            require(reader.getsampwidth() == 2, "exported prefix must be PCM16")
            exported_pcm = reader.readframes(prefix_samples)
        require(source_pcm == exported_pcm, "exported prefix PCM16 is not bit-exact")
        require(len(source_pcm) == prefix_samples * 2, "prefix PCM byte count")
    return {
        "prefix_samples": prefix_samples,
        "pcm16_bit_exact": True,
        "prepared_equals_pcm16_div_32768": True,
        "source_channels": geometry["source_channels"],
        "no_resampling_gain_or_renorm": True,
    }


def native_environment(config: dict[str, Any], dump_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("TRANSCRIBE_"):
            env.pop(key, None)
    env.update({key: str(value) for key, value in config["native"]["required_environment"].items()})
    env["TRANSCRIBE_DUMP_DIR"] = str(dump_dir)
    for key in config["native"]["absent_environment"]:
        require(key not in env, f"forbidden environment still set: {key}")
    require("TRANSCRIBE_SORTFORMER_OFFLINE_DUMP" not in env, "offline dump environment leaked")
    for key, value in config["native"]["required_environment"].items():
        require(env.get(key) == str(value), f"required environment missing: {key}")
    return env


def describe_environment(env: dict[str, str]) -> dict[str, Any]:
    transcribe = {key: env[key] for key in sorted(env) if key.startswith("TRANSCRIBE_")}
    inherited = {
        key: env[key]
        for key in sorted(env)
        if key.startswith(("CUDA_", "HIP_", "ROCR_", "HSA_", "VK_", "GGML_"))
    }
    return {
        "effective_transcribe": transcribe,
        "other_inherited_backend_flags": inherited,
        "other_inherited_backend_flags_not_exhaustive_os_environment": True,
        "absent": {key: key not in env for key in (
            "TRANSCRIBE_PSEM_PACE_16KHZ",
            "TRANSCRIBE_PSEM_PACE_FROM_SAMPLE",
            "TRANSCRIBE_PSEM_PACE_UNTIL_SAMPLE",
            "TRANSCRIBE_PSEM_EVENTS_TCP",
            "TRANSCRIBE_SORTFORMER_OFFLINE_DUMP",
            "TRANSCRIBE_DUMP_HIDDEN",
            "TRANSCRIBE_EXPORT_EMBED",
            "TRANSCRIBE_EXPORT_EMBEDDING",
            "TRANSCRIBE_SORTFORMER_COMPRESS",
        )},
    }


def relevant_environment(env: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    del config
    return describe_environment(env)


def verify_trace_profile(trace: dict[str, Any], config: dict[str, Any], prefix_samples: int) -> None:
    profile = config["native"]["profile"]
    require(int(trace["chunk_len"]) == int(profile["chunk_len"]), "trace chunk_len")
    require(int(trace["chunk_left_context"]) == int(profile["chunk_left_context"]), "trace left context")
    require(int(trace["chunk_right_context"]) == int(profile["chunk_right_context"]), "trace right context")
    require(int(trace["fifo_len"]) == int(profile["fifo_len"]), "trace fifo_len")
    require(int(trace["spkcache_len"]) == int(profile["spkcache_len"]), "trace spkcache_len")
    require(int(trace["spkcache_update_period"]) == int(profile["spkcache_update_period"]), "trace update period")
    require(int(trace["causal_frontend"]) == 1, "trace causal_frontend")
    require(int(trace["source_clock_hz"]) == SAMPLE_RATE, "trace sample rate")
    require(int(trace["frame_samples"]) == HOP, "trace frame_samples")
    require(int(trace["pcm_samples"]) == prefix_samples, "trace pcm_samples does not match bound prefix")


def assemble_teacher(
    *,
    probabilities: np.ndarray,
    chunks: list[dict[str, Any]],
    prefix_samples: int,
    used_n: int,
    n_frames: int,
) -> dict[str, Any]:
    require(probabilities.dtype == np.float32, "teacher dump must be float32")
    require(probabilities.ndim == 2 and probabilities.shape[1] == OUTPUT_SLOTS, "teacher dump shape")
    require(probabilities.shape[0] == used_n, "teacher dump rows must equal used_n")
    require(prefix_samples % HOP == 0, "prefix_samples must be a hop multiple")
    require(used_n <= n_frames, "used_n exceeds n_frames preallocation")
    nominal = prefix_samples // HOP
    require(used_n <= nominal, "used_n exceeds nominal prefix frames")
    require(bool(np.isfinite(probabilities).all()), "teacher probabilities must be finite")
    require(bool((probabilities >= 0.0).all() and (probabilities <= 1.0).all()), "teacher probabilities must lie in [0,1]")
    emitted = np.zeros(nominal, dtype=bool)
    support_valid = np.zeros(nominal, dtype=bool)
    raw_support = np.zeros(nominal, dtype=np.int64)
    raw = np.full((nominal, OUTPUT_SLOTS), np.nan, dtype=np.float32)
    cursor = 0
    incomplete_frames: list[int] = []
    for chunk in chunks:
        start = int(chunk["emit_start_frame"])
        count = int(chunk["emit_count"])
        require(start == cursor, f"non-contiguous emit_start_frame at {cursor}: {start}")
        require(count > 0, "emit_count must be positive")
        require(cursor + count <= used_n, "emit coverage exceeds used_n")
        support_end = int(chunk["raw_support_end_sample"])
        complete = support_end <= prefix_samples
        for offset in range(count):
            frame = start + offset
            require(frame < nominal, "emitted frame exceeds nominal prefix frames")
            emitted[frame] = True
            raw[frame] = probabilities[frame]
            raw_support[frame] = support_end
            support_valid[frame] = complete
            if not complete:
                incomplete_frames.append(frame)
        cursor += count
    require(cursor == used_n, "contiguous emit coverage does not equal used_n")
    missing = int((~emitted).sum())
    require(int(emitted.sum()) == used_n, "emitted mask does not match used_n")
    return {
        "nominal_frames": nominal,
        "used_n": used_n,
        "n_frames": n_frames,
        "chunk_count": len(chunks),
        "raw_probabilities": raw,
        "emitted_mask": emitted,
        "support_valid_mask": support_valid,
        "raw_support_end_sample": raw_support,
        "missing_frames": missing,
        "support_incomplete_frames": incomplete_frames,
        "support_incomplete_count": len(incomplete_frames),
        "do_not_infer_emission_from_n_frames": True,
    }


def decode_dump_dir(dump_dir: Path, prefix_samples: int, config: dict[str, Any] | None) -> dict[str, Any]:
    meta_path = dump_dir / "diar.probs.json"
    f32_path = dump_dir / "diar.probs.f32"
    trace_path = dump_dir / "diar.trace.json"
    require(meta_path.is_file() and f32_path.is_file() and trace_path.is_file(), f"incomplete dump in {dump_dir}")
    meta = load_json(meta_path)
    require(meta.get("dtype") == "f32" and meta.get("layout") == "row-major", "dump layout")
    require(isinstance(meta.get("shape"), list) and len(meta["shape"]) == 2, "dump shape metadata")
    require(int(meta["shape"][1]) == OUTPUT_SLOTS, "dump slot count")
    used_from_meta = int(meta["shape"][0])
    payload = np.fromfile(f32_path, dtype="<f4")
    require(payload.size == used_from_meta * OUTPUT_SLOTS, "dump byte/row count mismatch")
    require(f32_path.stat().st_size == used_from_meta * OUTPUT_SLOTS * 4, "dump file size mismatch")
    probabilities = payload.reshape(used_from_meta, OUTPUT_SLOTS)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if config is not None:
        verify_trace_profile(trace, config, prefix_samples)
    else:
        require(int(trace["pcm_samples"]) == prefix_samples, "trace pcm_samples does not match prefix")
        require(int(trace["frame_samples"]) == HOP, "trace frame_samples")
    used_n = int(trace["used_n"])
    n_frames = int(trace["n_frames"])
    require(used_n == used_from_meta, "trace used_n does not match dump metadata")
    require(used_n == int(trace["total_n"]), "trace used_n/total_n disagree")
    require(len(trace["chunks"]) == int(trace["chunk_count"]), "trace chunk_count")
    assembled = assemble_teacher(
        probabilities=probabilities,
        chunks=trace["chunks"],
        prefix_samples=prefix_samples,
        used_n=used_n,
        n_frames=n_frames,
    )
    assembled.update(
        {
            "dump_dir": str(dump_dir),
            "probabilities_sha256": digest_file(f32_path),
            "metadata_sha256": digest_file(meta_path),
            "trace_sha256": digest_file(trace_path),
            "dump_min": float(probabilities.min()),
            "dump_max": float(probabilities.max()),
            "mel_full_diagnostic_us": int(trace.get("mel_full_diagnostic_us", 0)),
            "initialization_us": int(trace.get("initialization_us", 0)),
            "load_us": int(trace.get("load_us", 0)),
            "trace_timing_us": {
                name: run_baseline.timing_summary([int(row[name]) for row in trace["chunks"]])
                for name in ("service_us", "frontend_us", "graph_a_us", "graph_b_us", "host_us")
            },
        }
    )
    return assembled


def clip_counts(probability: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    selected = probability[mask]
    finite = np.asarray(selected[np.isfinite(selected)], dtype=np.float64)
    return {
        "values": int(finite.size),
        "clipped_low": int((finite < LOGIT_CLIP).sum()),
        "clipped_high": int((finite > 1.0 - LOGIT_CLIP).sum()),
        "clip": LOGIT_CLIP,
    }


def class_conditioned_strict_solo_means(
    raw: np.ndarray,
    mapped: np.ndarray,
    qsolo: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for gt_slot in range(OUTPUT_SLOTS):
        class_mask = qsolo["solo_mask"] & (qsolo["true_slot"] == gt_slot)
        n = int(class_mask.sum())
        if n == 0:
            rows.append(
                {
                    "gt_slot": gt_slot,
                    "denominator": 0,
                    "mean_raw_native_probability": None,
                    "mean_mapped_native_probability": None,
                }
            )
            continue
        rows.append(
            {
                "gt_slot": gt_slot,
                "denominator": n,
                "mean_raw_native_probability": [float(value) for value in np.mean(raw[class_mask], axis=0)],
                "mean_mapped_native_probability": [float(value) for value in np.mean(mapped[class_mask], axis=0)],
            }
        )
    return {
        "grouped_by": "existing_strict_solo_gt_label_and_common_valid_and_fixed_quarter",
        "source_global_min_independent_bce_permutation_not_refit": True,
        "absent_class_is_null_not_zero": True,
        "rows": rows,
    }


def existing_association_view(association: dict[str, Any]) -> dict[str, Any]:
    skip = {"gt_ordered_probabilities", "common_mask"}
    view = {key: value for key, value in association.items() if key not in skip}
    quarters = []
    for quarter in view["chronological_quarters"]:
        quarters.append(
            {
                "quarter": quarter["quarter"],
                "frame_start": quarter["frame_start"],
                "frame_end": quarter["frame_end"],
                "common_valid_bins": quarter["common_valid_bins"],
                "strict_solo_counts": quarter["strict_solo_counts"],
                "mean_mapped_native_probability": quarter["mean_mapped_native_probability"],
                "mapping_not_refit_per_quarter": quarter["mapping_not_refit_per_quarter"],
            }
        )
    view["chronological_quarters"] = quarters
    return json.loads(json.dumps(ssd.json_safe(view)))

def permutation_costs(logits: torch.Tensor, targets: torch.Tensor, validity: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for permutation in itertools.permutations(range(OUTPUT_SLOTS)):
        permuted = logits[:, list(permutation)]
        value = float(
            torch.nn.functional.binary_cross_entropy_with_logits(permuted[validity], targets[validity])
        )
        rows.append({"permutation": list(permutation), "masked_bce": value})
    return rows


def associate_source(
    teacher: dict[str, Any],
    targets: torch.Tensor,
    validity: torch.Tensor,
    frontiers: torch.Tensor,
    slots: list[Any],
) -> dict[str, Any]:
    emitted = teacher["emitted_mask"]
    support_valid = teacher["support_valid_mask"]
    raw = teacher["raw_probabilities"]
    gt_valid = ssd.as_numpy(validity).astype(bool)
    require(gt_valid.shape == emitted.shape, "GT validity / teacher frame mismatch")
    require(tuple(frontiers.shape) == emitted.shape, "frontier / teacher frame mismatch")
    common = gt_valid & support_valid
    require(int(common.sum()) > 0, "no common-valid bins for association")
    emitted_probs = raw[emitted]
    require(bool(np.isfinite(emitted_probs).all()), "emitted probabilities must be finite")
    association_probs = np.where(np.isfinite(raw), raw, 0.5).astype(np.float64)
    clip = clip_counts(raw, common)
    logits_np = ssd.logit_from_probability(association_probs)
    logits = torch.from_numpy(logits_np)
    oracle = gt_pilot.permutation_oracle(logits, targets, torch.from_numpy(common))
    costs = permutation_costs(logits, targets, torch.from_numpy(common))
    best = oracle["best"]["permutation"]
    require(best == min((row for row in costs if row["masked_bce"] == oracle["best"]["masked_bce"]), key=lambda row: row["permutation"])["permutation"], "lexicographic tie-break failed")
    ordered = sorted(costs, key=lambda row: (row["masked_bce"], row["permutation"]))
    margin = None
    if len(ordered) > 1:
        margin = float(ordered[1]["masked_bce"] - ordered[0]["masked_bce"])
    ties = [row["permutation"] for row in costs if row["masked_bce"] == oracle["best"]["masked_bce"]]
    mapped_raw = raw[:, best]
    mapped_logits = logits_np[:, best]
    target_np = ssd.as_numpy(targets)
    solo_common = ssd.solo_partition(target_np, common)
    solo_gt = ssd.solo_partition(target_np, gt_valid)
    mapped_activity_bce = ssd.independent_bce(mapped_logits, target_np, common)
    raw_activity_bce = ssd.independent_bce(logits_np, target_np, common)
    mapped_core = ssd.conditional_core(mapped_logits, solo_common["true_slot"], solo_common["solo_mask"])
    predicted = np.where(np.isfinite(mapped_raw), mapped_raw, 0.0) >= 0.5
    counts = predicted.sum(axis=1)
    distribution = {str(active): int((counts[common] == active).sum()) for active in range(OUTPUT_SLOTS + 1)}
    n_solo = int(solo_common["solo_mask"].sum())
    exactly_one = None
    correct_single = None
    if n_solo:
        selected = predicted[solo_common["solo_mask"]]
        n_active = selected.sum(axis=1)
        exactly = n_active == 1
        pred_slot = np.full(n_solo, -1, dtype=np.int64)
        pred_slot[exactly] = selected[exactly].argmax(axis=1)
        exactly_one = float(exactly.mean())
        correct_single = float((exactly & (pred_slot == solo_common["true_slot"][solo_common["solo_mask"]])).mean())
    frames = int(emitted.shape[0])
    edges = [0, frames // 4, frames // 2, (3 * frames) // 4, frames]
    quarters = []
    for index in range(4):
        qmask = common & (np.arange(frames) >= edges[index]) & (np.arange(frames) < edges[index + 1])
        qsolo = ssd.solo_partition(target_np, qmask)
        if int(qmask.sum()) == 0:
            mean_probs = [None] * OUTPUT_SLOTS
        else:
            mean_probs = [float(value) for value in np.nanmean(mapped_raw[qmask], axis=0)]
        quarters.append(
            {
                "quarter": index,
                "frame_start": edges[index],
                "frame_end": edges[index + 1],
                "common_valid_bins": int(qmask.sum()),
                "strict_solo_counts": [
                    int((qsolo["solo_mask"] & (qsolo["true_slot"] == slot)).sum()) for slot in range(OUTPUT_SLOTS)
                ],
                "mean_mapped_native_probability": mean_probs,
                "unconditional_occupancy_population": "all_common_valid_bins_in_quarter_not_class_conditioned",
                "mapping_not_refit_per_quarter": True,
                "class_conditioned_strict_solo_means": class_conditioned_strict_solo_means(raw, mapped_raw, qsolo),
            }
        )
    gt_ordered = np.where(np.isfinite(mapped_raw), mapped_raw, np.nan).astype(np.float32)
    return {
        "permutation": best,
        "permutation_semantics": "gt_column_g_receives_native_column_permutation[g]",
        "not_human_identity_guarantee": True,
        "not_per_chunk_remapping": True,
        "oracle": oracle,
        "all_permutation_costs": costs,
        "tie_permutations": ties,
        "tie_count": len(ties),
        "second_best_margin_bce": margin,
        "logit_clip": LOGIT_CLIP,
        "logit_clip_applied_to_raw_targets": False,
        "association_clip_counts_on_common_valid": clip,
        "emitted_clip_counts": clip_counts(raw, emitted),
        "common_valid_bins": int(common.sum()),
        "gt_activity_valid_bins": int(gt_valid.sum()),
        "teacher_support_valid_bins": int(support_valid.sum()),
        "teacher_emitted_bins": int(emitted.sum()),
        "missing_or_nonemitted_bins": int((~emitted).sum()),
        "gt_strict_solo_counts": [
            int((solo_gt["solo_mask"] & (solo_gt["true_slot"] == slot)).sum()) for slot in range(OUTPUT_SLOTS)
        ],
        "common_strict_solo_counts": [
            int((solo_common["solo_mask"] & (solo_common["true_slot"] == slot)).sum()) for slot in range(OUTPUT_SLOTS)
        ],
        "raw_index_activity_bce": raw_activity_bce,
        "mapped_activity_bce": mapped_activity_bce,
        "mapped_activity_brier": ssd.independent_brier(mapped_logits, target_np, common),
        "mapped_solo": mapped_core,
        "threshold_0_5_on_raw_mapped_probabilities": {
            "threshold": 0.5,
            "predicted_active_count_distribution_on_common_valid": distribution,
            "predicted_exactly_one_active_fraction_on_strict_solo": exactly_one,
            "correct_single_active_fraction_on_strict_solo": correct_single,
            "not_physical_overlap_inference": True,
        },
        "chronological_quarters": quarters,
        "common_strict_solo_counts_uncertainty": "report_only_no_quality_threshold",
        "collapse_flag": bool(
            max(distribution.values()) == int(common.sum()) and int(common.sum()) > 0
        ),
        "slots": list(slots),
        "gt_ordered_probabilities": gt_ordered,
        "common_mask": common,
    }


def write_sidecar(
    path: Path,
    *,
    source_id: str,
    prefix_samples: int,
    generation: str,
    teacher: dict[str, Any],
    association: dict[str, Any],
    prepared_sha256: str,
    frontiers: torch.Tensor,
) -> str:
    payload = {
        "schema": SIDECAR_SCHEMA,
        "source_id": source_id,
        "pilot_role": "FIT",
        "prefix_samples": prefix_samples,
        "hop_samples": HOP,
        "output_slots": OUTPUT_SLOTS,
        "generation": generation,
        "slots": list(association["slots"]),
        "permutation": list(association["permutation"]),
        "permutation_semantics": association["permutation_semantics"],
        "not_human_identity_guarantee": True,
        "raw_probabilities": torch.from_numpy(teacher["raw_probabilities"].copy()),
        "gt_ordered_probabilities": torch.from_numpy(association["gt_ordered_probabilities"].copy()),
        "emitted_mask": torch.from_numpy(teacher["emitted_mask"].copy()),
        "support_valid_mask": torch.from_numpy(teacher["support_valid_mask"].copy()),
        "frontiers": frontiers.detach().cpu().contiguous(),
        "raw_support_end_sample": torch.from_numpy(teacher["raw_support_end_sample"].copy()),
        "prepared_sources_sha256": prepared_sha256,
        "raw_probabilities_not_clipped": True,
        "raw_probabilities_not_softmax_normalized": True,
        "teacher_support_distinct_from_gt_validity": True,
        "missing_rows_are_not_acoustic_unknown_or_silence": True,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return digest_file(path)


def synthetic_geometry_probes() -> dict[str, Any]:
    hop = HOP
    prefix = 20 * hop
    used = 18
    probs = np.full((used, OUTPUT_SLOTS), 0.25, dtype=np.float32)
    probs[:, 0] = 0.7
    chunks = []
    cursor = 0
    for index in range(3):
        count = 6
        support = prefix - hop if index < 2 else prefix + 96
        chunks.append(
            {
                "emit_start_frame": cursor,
                "emit_count": count,
                "raw_support_end_sample": support,
            }
        )
        cursor += count
    teacher = assemble_teacher(
        probabilities=probs,
        chunks=chunks,
        prefix_samples=prefix,
        used_n=used,
        n_frames=22,
    )
    require(teacher["nominal_frames"] == 20, "synthetic nominal frames")
    require(teacher["missing_frames"] == 2, "missing tail frames were not masked")
    require(not bool(teacher["emitted_mask"][-2:].any()), "missing tail marked emitted")
    require(teacher["support_incomplete_count"] == 6, "incomplete last-chunk support")
    require(not bool(teacher["support_valid_mask"][12:18].any()), "incomplete tail remained support-valid")
    require(bool(teacher["support_valid_mask"][:12].all()), "interior complete support was dropped")
    gapped = [
        {"emit_start_frame": 0, "emit_count": 6, "raw_support_end_sample": prefix},
        {"emit_start_frame": 7, "emit_count": 6, "raw_support_end_sample": prefix},
    ]
    gapped_failed = False
    try:
        assemble_teacher(
            probabilities=np.full((12, OUTPUT_SLOTS), 0.2, dtype=np.float32),
            chunks=gapped,
            prefix_samples=prefix,
            used_n=12,
            n_frames=20,
        )
    except RuntimeError as error:
        gapped_failed = "non-contiguous" in str(error)
    require(gapped_failed, "gapped emit_start_frame was accepted")
    targets = torch.zeros((20, OUTPUT_SLOTS), dtype=torch.float32)
    targets[:, 0] = 1.0
    validity = torch.ones(20, dtype=torch.bool)
    original_targets = targets.clone()
    original_validity = validity.clone()
    teacher_unavailable = teacher["support_valid_mask"]
    require(not bool(teacher_unavailable[-8:].all()), "expected unavailable tail")
    require(torch.equal(targets, original_targets) and torch.equal(validity, original_validity), "GT mutated while teacher unavailable")
    common = ssd.as_numpy(validity).astype(bool) & teacher["support_valid_mask"]
    require(int(common.sum()) == 12, "common mask did not drop unavailable teacher bins")
    require(int(original_validity.sum()) == 20, "GT validity was narrowed to teacher availability")
    swapped = np.zeros((12, OUTPUT_SLOTS), dtype=np.float32)
    swapped[:, 1] = 0.8
    swapped[:, 0] = 0.1
    swapped[:, 2] = 0.05
    swapped[:, 3] = 0.05
    swap_chunks = [
        {"emit_start_frame": 0, "emit_count": 6, "raw_support_end_sample": 12 * hop},
        {"emit_start_frame": 6, "emit_count": 6, "raw_support_end_sample": 12 * hop},
    ]
    swap_teacher = assemble_teacher(
        probabilities=swapped,
        chunks=swap_chunks,
        prefix_samples=12 * hop,
        used_n=12,
        n_frames=12,
    )
    swap_targets = torch.zeros((12, OUTPUT_SLOTS), dtype=torch.float32)
    swap_targets[:, 0] = 1.0
    swap_valid = torch.ones(12, dtype=torch.bool)
    associated = associate_source(
        swap_teacher,
        swap_targets,
        swap_valid,
        gt_pilot.occupancy_frontiers(12 * hop),
        ["A", "B", "C", "D"],
    )
    require(associated["permutation"] == [1, 0, 2, 3], "activity-BCE permutation did not recover the swap")
    tied_probs = np.full((8, OUTPUT_SLOTS), 0.25, dtype=np.float32)
    tied_teacher = assemble_teacher(
        probabilities=tied_probs,
        chunks=[{"emit_start_frame": 0, "emit_count": 8, "raw_support_end_sample": 8 * hop}],
        prefix_samples=8 * hop,
        used_n=8,
        n_frames=8,
    )
    tied_targets = torch.full((8, OUTPUT_SLOTS), 0.25, dtype=torch.float32)
    tied = associate_source(
        tied_teacher,
        tied_targets,
        torch.ones(8, dtype=torch.bool),
        gt_pilot.occupancy_frontiers(8 * hop),
        ["A", "B", "C", "D"],
    )
    require(tied["permutation"] == [0, 1, 2, 3], "tied permutations were not lexicographic")
    require(tied["tie_count"] == 24, "identical heads should tie all 24 permutations")
    clip_probs = np.full((4, OUTPUT_SLOTS), 0.5, dtype=np.float32)
    clip_probs[0, 0] = 0.0
    clip_probs[1, 1] = 1.0
    clip_teacher = assemble_teacher(
        probabilities=clip_probs,
        chunks=[{"emit_start_frame": 0, "emit_count": 4, "raw_support_end_sample": 4 * hop}],
        prefix_samples=4 * hop,
        used_n=4,
        n_frames=4,
    )
    clip_targets = torch.zeros((4, OUTPUT_SLOTS), dtype=torch.float32)
    clip_targets[:, 0] = 1.0
    clipped = associate_source(
        clip_teacher,
        clip_targets,
        torch.ones(4, dtype=torch.bool),
        gt_pilot.occupancy_frontiers(4 * hop),
        ["A", "B", "C", "D"],
    )
    require(clipped["association_clip_counts_on_common_valid"]["clipped_low"] == 1, "zero probability was not counted as clip-low")
    require(clipped["association_clip_counts_on_common_valid"]["clipped_high"] == 1, "one probability was not counted as clip-high")
    require(float(clip_teacher["raw_probabilities"][0, 0]) == 0.0, "raw sidecar probability was clipped")
    for quarter in tied["chronological_quarters"]:
        for row in quarter["class_conditioned_strict_solo_means"]["rows"]:
            require(row["mean_raw_native_probability"] is None and row["mean_mapped_native_probability"] is None, "absent strict-solo class must emit null, not zero")
            require(row["denominator"] == 0, "tied fractional GT is not strict-solo")
    swap_rows = associated["chronological_quarters"][0]["class_conditioned_strict_solo_means"]["rows"]
    require(swap_rows[0]["denominator"] > 0 and swap_rows[0]["mean_raw_native_probability"] is not None, "present GT class must publish a mean")
    require(swap_rows[1]["mean_raw_native_probability"] is None, "absent GT class leaked a zero mean")
    return {
        "missing_tail_invalid": True,
        "incomplete_support_tail_invalid": True,
        "contiguous_gap_rejected": True,
        "gt_preserved_when_teacher_unavailable": True,
        "activity_bce_permutation_recovered_swap": associated["permutation"],
        "lexicographic_tie": tied["permutation"],
        "clip_counts_disclosed": clipped["association_clip_counts_on_common_valid"],
        "raw_zero_preserved": True,
        "n_frames_not_used_as_emission": teacher["n_frames"] != teacher["used_n"],
    }


def retained_dev_parser_smoke(config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for spec in config["retained_dev_parser_smoke_read_only"]:
        dump = repo_path(spec["dump"])
        teacher = decode_dump_dir(dump, int(spec["prefix_samples"]), config)
        require(spec["role"] == "DEV_not_FIT_target", "DEV dump was classified as a FIT target")
        rows.append(
            {
                "source_id": spec["source_id"],
                "role": spec["role"],
                "used_n": teacher["used_n"],
                "n_frames": teacher["n_frames"],
                "missing_frames": teacher["missing_frames"],
                "support_incomplete_count": teacher["support_incomplete_count"],
                "chunk_count": teacher["chunk_count"],
                "dump_min": teacher["dump_min"],
                "dump_max": teacher["dump_max"],
                "probabilities_sha256": teacher["probabilities_sha256"],
                "saved_as_fit_target": False,
            }
        )
    mismatch_failed = False
    try:
        decode_dump_dir(
            repo_path(config["retained_dev_parser_smoke_read_only"][0]["dump"]),
            4620800,
            config,
        )
    except RuntimeError as error:
        mismatch_failed = "pcm_samples" in str(error)
    require(mismatch_failed, "DEV dump accepted under a FIT prefix")
    return {"sources": rows, "source_mismatch_rejected": True, "written_as_fit_targets": False}


def require_authorization() -> None:
    if not MATERIAL_AUTHORIZED:
        raise RuntimeError(
            "native teacher material is blocked until Director GO passes --authorize-material"
        )


def load_director_release(run_root: Path, config: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    path = run_root / "director_release.json"
    require(path.is_file(), "Director release is missing; native execution is not authorized")
    release = load_json(path)
    require(release.get("schema") == RELEASE_SCHEMA, "unexpected director release schema")
    require(release.get("automatic_retry") is False, "director release must forbid automatic retry")
    require(release.get("teacher_fit_sha256") == hashes["teacher_fit.py"], "director release teacher_fit.py hash mismatch")
    require(release.get("config_sha256") == hashes["config"], "director release config hash mismatch")
    require(release.get("prepared_sources_sha256") == config["inputs"]["prepared_sources_sha256"], "director release prepared hash mismatch")
    require(release.get("executable_sha256") == config["native"]["executable_sha256"], "director release executable hash mismatch")
    require(release.get("model_sha256") == config["native"]["model_sha256"], "director release model hash mismatch")
    require(release.get("contract_sha256") == config["contract"]["sha256"], "director release contract hash mismatch")
    return release


def config_release_identity_probes(executed_config_path: Path) -> dict[str, Any]:
    executed_config_path = executed_config_path.resolve()
    config = load_teacher_config(executed_config_path)
    with tempfile.TemporaryDirectory(prefix="teacher-fit-config-gate-") as temp:
        root = Path(temp)
        frozen = root / "frozen_config.json"
        frozen.write_bytes(executed_config_path.read_bytes())
        hashes = verify_pinned_files(config, frozen)
        require(hashes["config"] == digest_file(frozen), "executed config hash is not the frozen bytes")
        require(hashes["config"] != "", "executed config hash missing")
        publish_json(
            root / "director_release.json",
            {
                "schema": RELEASE_SCHEMA,
                "automatic_retry": False,
                "teacher_fit_sha256": hashes["teacher_fit.py"],
                "config_sha256": hashes["config"],
                "prepared_sources_sha256": config["inputs"]["prepared_sources_sha256"],
                "executable_sha256": config["native"]["executable_sha256"],
                "model_sha256": config["native"]["model_sha256"],
                "contract_sha256": config["contract"]["sha256"],
            },
        )
        load_director_release(root, config, hashes)
        modified = load_json(executed_config_path)
        modified["native"]["required_environment"]["TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN"] = "99"
        modified["bounds"]["whole_run_wall_seconds"] = 1
        modified_path = root / "modified_config.json"
        modified_path.write_text(json.dumps(modified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        modified_config = load_teacher_config(modified_path)
        modified_hashes = verify_pinned_files(modified_config, modified_path)
        require(modified_hashes["config"] != hashes["config"], "modified config hash collided with executed frozen hash")
        rejected = False
        try:
            load_director_release(root, modified_config, modified_hashes)
        except RuntimeError as error:
            rejected = "config hash mismatch" in str(error)
        require(rejected, "modified executed config was accepted by the release metadata gate")
        native_launched = False
    return {
        "unchanged_frozen_accepted": True,
        "modified_config_rejected_before_launch": True,
        "native_launched": native_launched,
        "executed_config_sha256": hashes["config"],
    }


def environment_isolation_probes(config: dict[str, Any]) -> dict[str, Any]:
    injected = {
        "TRANSCRIBE_SORTFORMER_PRESET": "injected-preset",
        "TRANSCRIBE_DUMP_HIDDEN": "1",
        "TRANSCRIBE_EXPORT_EMBED": "1",
        "TRANSCRIBE_SORTFORMER_OFFLINE_DUMP": "1",
        "TRANSCRIBE_PSEM_PACE_16KHZ": "1",
        "TRANSCRIBE_PSEM_EVENTS_TCP": "127.0.0.1:1",
        "TRANSCRIBE_SORTFORMER_EXPORT": "hidden",
    }
    prior = {key: os.environ.get(key) for key in injected}
    dump = Path("cpu-env-probe-dump")
    try:
        os.environ.update(injected)
        env = native_environment(config, dump)
        described = describe_environment(env)
        for key in injected:
            if key == "TRANSCRIBE_SORTFORMER_EXPORT":
                continue
            require(key not in env, f"injected {key} survived native environment construction")
        for key, value in config["native"]["required_environment"].items():
            require(env.get(key) == str(value), f"approved {key} was not set")
        require(env.get("TRANSCRIBE_DUMP_DIR") == str(dump), "DUMP_DIR was not the run-local dump")
        require(env.get("TRANSCRIBE_SORTFORMER_EXPORT") == "logits", "future EXPORT was not forced to logits")
        require(env.get("TRANSCRIBE_SORTFORMER_EXPORT") != "hidden", "injected hidden EXPORT survived")
        require("TRANSCRIBE_DUMP_HIDDEN" not in env, "injected DUMP_HIDDEN survived")
        require(described["effective_transcribe"].get("TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN") == "6", "chunk_len")
        require(described["effective_transcribe"].get("TRANSCRIBE_SORTFORMER_STREAM_LC") == "1", "left context")
        require(described["effective_transcribe"].get("TRANSCRIBE_SORTFORMER_STREAM_RC") == "7", "right context")
        require(described["effective_transcribe"].get("TRANSCRIBE_SORTFORMER_STREAM_FIFO_LEN") == "188", "fifo")
        require(described["effective_transcribe"].get("TRANSCRIBE_SORTFORMER_STREAM_SPKCACHE_LEN") == "188", "spkcache")
        require(described["effective_transcribe"].get("TRANSCRIBE_SORTFORMER_STREAM_UPDATE_PERIOD") == "144", "update period")
        require(described["effective_transcribe"].get("TRANSCRIBE_PSEM_CAUSAL_FRONTEND") == "1", "causal frontend")
        require(described["effective_transcribe"].get("TRANSCRIBE_SORTFORMER_F32_HEAD") == "1", "F32 head")
        require(described["effective_transcribe"].get("TRANSCRIBE_VK_NO_MUL_MAT_VEC") == "1", "VK_NO_MUL_MAT_VEC")
        require(described["effective_transcribe"].get("TRANSCRIBE_SORTFORMER_EXPORT") == "logits", "effective EXPORT")
        require(all(described["absent"].values()), "pace/TCP/offline/hidden extras must stay absent")
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {
        "injected_preset_hidden_export_absent": True,
        "approved_profile_present": True,
        "offline_pace_tcp_absent": True,
        "future_export_explicit_logits": True,
        "future_hidden_export_forbidden": True,
        "effective_transcribe": described["effective_transcribe"],
        "other_inherited_backend_flags_not_exhaustive_os_environment": True,
    }


def bind_frozen_config(config_path: Path, run_root: Path) -> Path:
    frozen = run_root / "frozen_config.json"
    requested = config_path.read_bytes()
    if frozen.exists():
        require(frozen.read_bytes() == requested, "requested config differs from the frozen teacher-fit config")
    else:
        temporary = frozen.with_name(f".{frozen.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(requested)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, frozen)
        finally:
            temporary.unlink(missing_ok=True)
    return frozen


def directory_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def run_native_source(
    *,
    config: dict[str, Any],
    origin_source: dict[str, Any],
    prepared_payload: dict[str, Any],
    run_root: Path,
    source_deadline: float,
    run_deadline: float,
    prepared_sha256: str,
) -> dict[str, Any]:
    source_id = origin_source["source_id"]
    source_dir = run_root / "sources" / source_id
    intent_path = source_dir / "intent.json"
    completion_path = source_dir / "completion.json"
    if completion_path.exists():
        raise RuntimeError(f"{source_id} already completed; automatic retry is forbidden")
    if intent_path.exists():
        raise RuntimeError(f"{source_id} has in-flight intent without completion; automatic retry is forbidden")
    source_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = source_dir / "dump"
    dump_dir.mkdir()
    prefix = int(origin_source["prefix_samples"])
    generation = f"{source_id}:unpaced-source-zero"
    projection = source_dir / "input.wav"
    publish_json(
        intent_path,
        {
            "source_id": source_id,
            "generation": generation,
            "prefix_samples": prefix,
            "status": "intent",
            "automatic_retry": False,
        },
    )
    export = prefix_export_matches_prepared(local_path(origin_source["waveform"]), prepared_payload["waveform"], prefix)
    run_baseline.make_projection(local_path(origin_source["waveform"]), projection, prefix)
    env = native_environment(config, dump_dir)
    command = [
        str(local_path(config["native"]["executable"])),
        "-m",
        str(local_path(config["native"]["model"])),
        "--backend",
        config["native"]["backend"],
        str(projection),
    ]
    stdout_path, stderr_path = source_dir / "native.stdout.log", source_dir / "native.stderr.log"
    query, counter, local_qpf = run_baseline.qpc_api()
    wrapper_handle = run_baseline.open_process_handle(os.getpid())
    wrapper_cpu_start = run_baseline.cpu_seconds(wrapper_handle)
    started_qpc = run_baseline.qpc_now(query, counter)
    started_wall = time.perf_counter()
    memory_samples: list[dict[str, Any]] = []
    exit_code = None
    gpu_path = source_dir / "gpu-process-memory.csv"
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            gpu = subprocess.Popen(
                [
                    "typeperf",
                    f"\\GPU Process Memory(pid_{process.pid}_*)\\Dedicated Usage",
                    "-si",
                    "1",
                    "-sc",
                    "1800",
                    "-f",
                    "CSV",
                    "-o",
                    str(gpu_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            child_handle = int(process._handle)

            def sample_memory() -> None:
                native_working, _native_peak = run_baseline.process_memory(child_handle)
                wrapper_working, _wrapper_peak = run_baseline.process_memory(wrapper_handle)
                memory_samples.append(
                    {
                        "qpc": run_baseline.qpc_now(query, counter),
                        "native_working_set_bytes": native_working,
                        "wrapper_working_set_bytes": wrapper_working,
                        "aggregate_working_set_bytes": native_working + wrapper_working,
                    }
                )

            try:
                sample_memory()
                while process.poll() is None:
                    now = time.perf_counter()
                    if now >= source_deadline or now >= run_deadline:
                        process.terminate()
                        raise TimeoutError(f"{source_id} exceeded wall cap")
                    sample_memory()
                    if memory_samples:
                        peak = max(row["aggregate_working_set_bytes"] for row in memory_samples)
                        if peak > int(config["bounds"]["max_observed_paired_host_working_set_bytes"]):
                            process.terminate()
                            raise RuntimeError(f"{source_id} exceeded paired working-set bound")
                    time.sleep(0.05)
                exit_code = process.wait(timeout=30)
                sample_memory()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                gpu.terminate()
                try:
                    gpu.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    gpu.kill()
        native_cpu = run_baseline.cpu_seconds(child_handle)
        wrapper_cpu = run_baseline.cpu_seconds(wrapper_handle) - wrapper_cpu_start
        finished_qpc = run_baseline.qpc_now(query, counter)
    finally:
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(wrapper_handle))
    if exit_code != 0:
        publish_json(
            completion_path,
            {"source_id": source_id, "status": "failed", "exit_code": exit_code, "uncertain_exposure": True},
        )
        raise RuntimeError(f"{source_id} native exit={exit_code}")
    teacher = decode_dump_dir(dump_dir, prefix, config)
    association = associate_source(
        teacher,
        prepared_payload["targets"],
        prepared_payload["validity"],
        prepared_payload["frontiers"],
        prepared_payload["slots"],
    )
    sidecar_path = source_dir / "teacher_sidecar.pt"
    sidecar_sha = write_sidecar(
        sidecar_path,
        source_id=source_id,
        prefix_samples=prefix,
        generation=generation,
        teacher=teacher,
        association=association,
        prepared_sha256=prepared_sha256,
        frontiers=prepared_payload["frontiers"],
    )
    memory_path = source_dir / "process-memory-samples.csv"
    with memory_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("qpc", "native_working_set_bytes", "wrapper_working_set_bytes", "aggregate_working_set_bytes"),
        )
        writer.writeheader()
        writer.writerows(memory_samples)
    gpu_stats = run_baseline.parse_gpu(gpu_path, process.pid)
    if gpu_stats.get("status") == "observed_process_counter":
        if int(gpu_stats["peak_bytes"]) > int(config["bounds"]["max_observed_GPU_dedicated_bytes"]):
            raise RuntimeError(f"{source_id} exceeded GPU dedicated bound")
    wall_s = (finished_qpc - started_qpc) / local_qpf
    result = {
        "schema": "PSEM-TEACHER-FIT-SOURCE-RUN-1",
        "source_id": source_id,
        "generation": generation,
        "prefix_export": export,
        "command": command,
        "environment": relevant_environment(env, config),
        "exit_code": exit_code,
        "clock": {
            "kind": "Windows QueryPerformanceCounter",
            "qpf": local_qpf,
            "process_start_qpc": started_qpc,
            "process_finish_qpc": finished_qpc,
            "process_elapsed_s": wall_s,
            "orchestrator_perf_counter_s": time.perf_counter() - started_wall,
        },
        "cost": {
            "native_process_cpu_s": native_cpu,
            "wrapper_cpu_s": wrapper_cpu,
            "memory_samples": len(memory_samples),
            "native_peak_working_set_bytes_observed": max(row["native_working_set_bytes"] for row in memory_samples) if memory_samples else 0,
            "paired_peak_working_set_bytes_observed": max(row["aggregate_working_set_bytes"] for row in memory_samples) if memory_samples else 0,
            "gpu_memory": gpu_stats,
            "native_trace_timing_us": teacher["trace_timing_us"],
            "mel_full_diagnostic_us": teacher["mel_full_diagnostic_us"],
            "unpaced_not_live_availability": True,
        },
        "teacher": {
            "used_n": teacher["used_n"],
            "n_frames": teacher["n_frames"],
            "chunk_count": teacher["chunk_count"],
            "missing_frames": teacher["missing_frames"],
            "support_incomplete_count": teacher["support_incomplete_count"],
            "probabilities_sha256": teacher["probabilities_sha256"],
            "trace_sha256": teacher["trace_sha256"],
            "sidecar_sha256": sidecar_sha,
        },
        "association": {key: value for key, value in association.items() if key not in {"gt_ordered_probabilities", "common_mask"}},
        "artifacts": {
            "projection": {"path": str(projection.relative_to(ROOT).as_posix()) if projection.is_relative_to(ROOT) else str(projection), "sha256": digest_file(projection)},
            "sidecar": {"path": str(sidecar_path.as_posix()), "sha256": sidecar_sha},
            "dump_probs": {"sha256": teacher["probabilities_sha256"]},
            "dump_trace": {"sha256": teacher["trace_sha256"]},
        },
    }
    publish_json(source_dir / "RESULT.json", result)
    publish_json(completion_path, {"source_id": source_id, "status": "completed", "exit_code": 0, "sidecar_sha256": sidecar_sha})
    return result


def smoke_stage(config_path: Path, run_root: Path) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    config = load_teacher_config(config_path)
    hashes = verify_pinned_files(config, config_path)
    native = verify_native_identities(config)
    origin = load_origin_config(config)
    identities = bind_fit_identities(config, origin)
    prepared = load_prepared(config)
    prepared_fit = verify_prepared_fit(config, prepared, origin)
    exports = {}
    for row in config["sources"]:
        source_id = row["source_id"]
        payload = prepared["sources"][source_id]
        origin_source = gt_pilot.catalog_sources(origin)[source_id]
        exports[source_id] = prefix_export_matches_prepared(
            local_path(origin_source["waveform"]), payload["waveform"], int(row["prefix_samples"])
        )
    geometry = synthetic_geometry_probes()
    dev_parser = retained_dev_parser_smoke(config)
    unauthorized = False
    try:
        require_authorization()
    except RuntimeError as error:
        unauthorized = "authorize-material" in str(error)
    require(unauthorized, "material gate failed to block unauthorized native execution")
    release_blocked = False
    try:
        load_director_release(run_root, config, hashes)
    except RuntimeError as error:
        release_blocked = "Director release is missing" in str(error) or "director release" in str(error).lower()
    require(release_blocked, "missing Director release was not treated as a native blocker")
    env = native_environment(config, run_root / "smoke-dump")
    require(all(env.get(key) == value for key, value in config["native"]["required_environment"].items()), "required unpaced environment")
    config_gate = config_release_identity_probes(config_path)
    env_gate = environment_isolation_probes(config)
    receipt = {
        "schema": "PSEM-TEACHER-FIT-SMOKE-1",
        "status": "preparation_smoke_passed",
        "orchestrator_pytorch_model_construction": 0,
        "native_model_loads": {
            "max_trajectories": 4,
            "successful_decoded_loads": 0,
            "uncertain_or_failed_launches_not_counted_as_loads": 0,
            "failed_intent_is_not_a_known_successful_load": True,
            "this_stage": "cpu_preparation",
        },
        "model_forward": False,
        "native_material_executed": False,
        "fit_targets_written": False,
        "identities": {
            "hashes": hashes,
            "native": native,
            "fit_sources": identities["sources"],
            "prepared_fit": prepared_fit,
        },
        "prefix_export": exports,
        "synthetic_geometry": geometry,
        "retained_dev_parser": dev_parser,
        "config_release_identity_probes": config_gate,
        "environment_isolation_probes": env_gate,
        "material_gate_blocks_without_authorize_and_release": True,
        "future_argv": future_argv(config_path, run_root),
    }
    publish_json(run_root / "smoke_receipt.json", receipt)
    return receipt


def bind_stage(config_path: Path, run_root: Path) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    frozen = bind_frozen_config(config_path, run_root)
    config = load_teacher_config(frozen)
    hashes = verify_pinned_files(config, frozen)
    require(hashes["config"] == digest_file(frozen), "bind config identity is not the frozen bytes")
    native = verify_native_identities(config)
    origin = load_origin_config(config)
    identities = bind_fit_identities(config, origin)
    prepared = load_prepared(config)
    prepared_fit = verify_prepared_fit(config, prepared, origin)
    receipt = {
        "schema": "PSEM-TEACHER-FIT-BIND-1",
        "status": "bound",
        "frozen_config_sha256": digest_file(frozen),
        "hashes": hashes,
        "native": native,
        "identities": identities,
        "prepared_fit": prepared_fit,
        "native_material_executed": False,
    }
    publish_json(run_root / "bind_receipt.json", receipt)
    return receipt


def future_argv(config_path: Path, run_root: Path) -> list[str]:
    return [
        sys.executable,
        str((HERE / "teacher_fit.py").as_posix()),
        "execute",
        "--config",
        str(config_path.resolve().as_posix()),
        "--run-root",
        str(run_root.resolve().as_posix()),
        "--authorize-material",
    ]


def execute_stage(config_path: Path, run_root: Path) -> dict[str, Any]:
    require_authorization()
    run_root.mkdir(parents=True, exist_ok=True)
    frozen = bind_frozen_config(config_path, run_root)
    config = load_teacher_config(frozen)
    hashes = verify_pinned_files(config, frozen)
    require(hashes["config"] == digest_file(frozen), "release/config identity is not the executed frozen bytes")
    release = load_director_release(run_root, config, hashes)
    native = verify_native_identities(config)
    origin = load_origin_config(config)
    identities = bind_fit_identities(config, origin)
    prepared = load_prepared(config)
    verify_prepared_fit(config, prepared, origin)
    catalog = gt_pilot.catalog_sources(origin)
    started = time.perf_counter()
    run_deadline = started + int(config["bounds"]["whole_run_wall_seconds"])
    source_results = []
    prepared_sha = config["inputs"]["prepared_sources_sha256"]
    try:
        for row in config["sources"]:
            source_id = row["source_id"]
            source_deadline = time.perf_counter() + int(config["bounds"]["per_source_wall_seconds"])
            result = run_native_source(
                config=config,
                origin_source=catalog[source_id],
                prepared_payload=prepared["sources"][source_id],
                run_root=run_root,
                source_deadline=min(source_deadline, run_deadline),
                run_deadline=run_deadline,
                prepared_sha256=prepared_sha,
            )
            source_results.append(result)
            used = directory_bytes(run_root)
            if used > int(config["bounds"]["max_new_artifact_bytes"]):
                raise RuntimeError("new artifact budget exceeded")
    except Exception:
        publish_json(
            run_root / "failure.json",
            {
                "status": "stopped_without_retry",
                "completed_sources": [row["source_id"] for row in source_results],
                "uncertain_source": None if len(source_results) == len(config["sources"]) else config["sources"][len(source_results)]["source_id"],
            },
        )
        raise
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "prepared_sources_sha256": prepared_sha,
        "original_config_sha256": config["inputs"]["original_config_sha256"],
        "teacher_fit_sha256": hashes["teacher_fit.py"],
        "config_sha256": hashes["config"],
        "executable_sha256": native["executable_sha256"],
        "model_sha256": native["model_sha256"],
        "sources": [
            {
                "source_id": row["source_id"],
                "sidecar_sha256": row["teacher"]["sidecar_sha256"],
                "permutation": row["association"]["permutation"],
                "used_n": row["teacher"]["used_n"],
                "support_incomplete_count": row["teacher"]["support_incomplete_count"],
            }
            for row in source_results
        ],
        "gt_remains_in_prepared_artifact": True,
        "teacher_support_distinct_from_gt_validity": True,
    }
    publish_json(run_root / "teacher_targets_manifest.json", manifest)
    public = {
        "schema": RESULT_SCHEMA,
        "status": "completed",
        "experiment_id": "TEACHER-FIT-TARGETS-1",
        "release": release,
        "identities": hashes,
        "native": native,
        "identities_bound": identities["sources"],
        "sources": source_results,
        "manifest": manifest,
        "orchestrator_wall_s": time.perf_counter() - started,
        "optimizer_updates": 0,
        "cumulative_optimizer_updates_unchanged": 1583,
        "orchestrator_pytorch_model_construction": 0,
        "native_model_loads": {
            "max_trajectories": int(config["bounds"]["new_native_source_trajectories"]),
            "successful_decoded_loads": sum(1 for row in source_results if row.get("exit_code") == 0),
            "uncertain_or_failed_launches_not_counted_as_loads": int(config["bounds"]["new_native_source_trajectories"]) - sum(1 for row in source_results if row.get("exit_code") == 0),
            "failed_intent_is_not_a_known_successful_load": True,
        },
        "findings": [
            "Teacher support validity is stored separately from GT validity.",
            "One source-global permutation was fit on common-valid independent activity BCE only.",
            "Raw sidecar probabilities are unclipped and not softmax-normalized.",
            "Association used existing 1e-12 logit clipping and disclosed clip counts.",
            "Permutation is a FIT task projection, not a human-identity guarantee.",
            "Native model-load count is successful decoded trajectories only, not launch intent.",
        ],
    }
    publish_json(repo_path(config["outputs"]["public_result"]), public)
    publish_json(run_root / "RESULT.json", public)
    return public


ORIGINAL_PUBLIC_SHA256 = "e3992cea36d3d7a74b019dcff1ffce82042de4e5e9824f9581edb297461dde28"
ORIGINAL_MANIFEST_SHA256 = "1c40adaabc8c5e63112d8d0e179bd27ca6d391c9a9a685ba56d348935b7ef968"
ORIGINAL_FROZEN_SHA256 = "6c57ecc9b8ef04feb4591b6e9fea4d813bc9739d29d5a826f71edc7cad98b0ea"
ORIGINAL_TEACHER_FIT_SHA256 = "bf08e02024093a793ad4023321b766a901dff784e2102cf77b1369871a6d1e7c"
MODEL_CPP_AFTER_SHA256 = "32ce450091abc4579b2d2354174e654c0600b69e491ddc43f813b195f7df6d60"
INCIDENTAL_DUMP_NAMES = ("diar.hidden.f32", "diar.hidden.json", "diar.logits.f32", "diar.logits.json")


def pin_path(path: Path) -> dict[str, Any]:
    relative = str(path.relative_to(ROOT).as_posix()) if path.is_relative_to(ROOT) else str(path)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": digest_file(path)}


def inventory_source_dump(dump_dir: Path) -> dict[str, Any]:
    incidental = []
    required = []
    for child in sorted(dump_dir.iterdir(), key=lambda item: item.name):
        if not child.is_file():
            continue
        item = pin_path(child)
        item["name"] = child.name
        if child.name in INCIDENTAL_DUMP_NAMES:
            incidental.append(item)
        else:
            required.append(item)
    return {
        "required_probs_trace": required,
        "incidental_hidden_and_logits": incidental,
        "incidental_bytes": sum(row["bytes"] for row in incidental),
        "hidden_bytes": sum(row["bytes"] for row in incidental if str(row["name"]).startswith("diar.hidden.")),
        "logits_bytes": sum(row["bytes"] for row in incidental if str(row["name"]).startswith("diar.logits.")),
    }


def preserve_original_public(public_path: Path, preserved: Path) -> str:
    current = digest_file(public_path)
    if current == ORIGINAL_PUBLIC_SHA256:
        preserved.write_bytes(public_path.read_bytes())
    require(preserved.is_file(), "original public result was not preserved")
    require(digest_file(preserved) == ORIGINAL_PUBLIC_SHA256, "preserved original public digest mismatch")
    return ORIGINAL_PUBLIC_SHA256


def repair_report_stage(config_path: Path, run_root: Path) -> dict[str, Any]:
    require(not MATERIAL_AUTHORIZED, "report repair must not authorize native material")
    run_root.mkdir(parents=True, exist_ok=True)
    config = load_teacher_config(config_path)
    hashes = verify_pinned_files(config, config_path)
    require(hashes["config"] != ORIGINAL_FROZEN_SHA256, "future config must not overwrite the executed frozen bytes")
    original_root = repo_path(config["outputs"]["run_root"])
    frozen = original_root / "frozen_config.json"
    manifest_path = original_root / "teacher_targets_manifest.json"
    public_path = repo_path(config["outputs"]["public_result"])
    require(digest_file(frozen) == ORIGINAL_FROZEN_SHA256, "executed frozen config was mutated")
    require(digest_file(manifest_path) == ORIGINAL_MANIFEST_SHA256, "original manifest was mutated")
    preserved = run_root / "original_TEACHER_FIT_TARGET_RESULT.json"
    preserve_original_public(public_path, preserved)
    original = load_json(preserved)
    prepared = load_prepared(config)
    origin = load_origin_config(config)
    catalog = gt_pilot.catalog_sources(origin)
    repaired_sources = []
    inventory = []
    unchanged = []
    for index, row in enumerate(config["sources"]):
        source_id = row["source_id"]
        original_source = original["sources"][index]
        require(original_source["source_id"] == source_id, "original source order mismatch")
        sidecar_path = original_root / "sources" / source_id / "teacher_sidecar.pt"
        sidecar_sha = digest_file(sidecar_path)
        require(sidecar_sha == original_source["teacher"]["sidecar_sha256"], "sidecar mutated")
        require(sidecar_sha == original["manifest"]["sources"][index]["sidecar_sha256"], "manifest sidecar mismatch")
        payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        prepared_payload = prepared["sources"][source_id]
        teacher = {
            "emitted_mask": ssd.as_numpy(payload["emitted_mask"]).astype(bool),
            "support_valid_mask": ssd.as_numpy(payload["support_valid_mask"]).astype(bool),
            "raw_probabilities": ssd.as_numpy(payload["raw_probabilities"]),
        }
        association = associate_source(
            teacher,
            prepared_payload["targets"],
            prepared_payload["validity"],
            prepared_payload["frontiers"],
            list(prepared_payload["slots"]),
        )
        original_view = existing_association_view(original_source["association"])
        repaired_view = existing_association_view(association)
        for view in (original_view, repaired_view):
            for quarter in view["chronological_quarters"]:
                quarter.pop("mean_mapped_native_probability", None)
        require(repaired_view == original_view, f"existing association metrics changed for {source_id}")
        dump_inventory = inventory_source_dump(original_root / "sources" / source_id / "dump")
        inventory.append({"source_id": source_id, "sidecar": pin_path(sidecar_path), "dump": dump_inventory})
        repaired_source = json.loads(json.dumps(original_source))
        merged_quarters = []
        for original_quarter, repaired_quarter in zip(
            original_source["association"]["chronological_quarters"],
            association["chronological_quarters"],
        ):
            merged = json.loads(json.dumps(original_quarter))
            merged["unconditional_occupancy_population"] = repaired_quarter["unconditional_occupancy_population"]
            merged["class_conditioned_strict_solo_means"] = json.loads(
                json.dumps(ssd.json_safe(repaired_quarter["class_conditioned_strict_solo_means"]))
            )
            merged["mapping_not_refit_per_quarter"] = True
            merged_quarters.append(merged)
        repaired_source["association"]["chronological_quarters"] = merged_quarters
        repaired_source["association"]["quarter_means_repaired_from_retained_raw"] = True
        repaired_sources.append(repaired_source)
        unchanged.append(
            {
                "source_id": source_id,
                "permutation": association["permutation"],
                "mapped_activity_bce": association["mapped_activity_bce"],
                "raw_index_activity_bce": association["raw_index_activity_bce"],
                "common_valid_bins": association["common_valid_bins"],
                "existing_metrics_unchanged": True,
            }
        )
    incidental_total = sum(row["dump"]["incidental_bytes"] for row in inventory)
    public = {
        "schema": RESULT_SCHEMA,
        "status": "completed_with_report_repair",
        "experiment_id": "TEACHER-FIT-TARGETS-1",
        "original_execution": {
            "public_sha256": ORIGINAL_PUBLIC_SHA256,
            "preserved_original_public": str(preserved.relative_to(ROOT).as_posix()),
            "teacher_fit_sha256": ORIGINAL_TEACHER_FIT_SHA256,
            "frozen_config_sha256": ORIGINAL_FROZEN_SHA256,
            "manifest_sha256": ORIGINAL_MANIFEST_SHA256,
            "identities": original["identities"],
            "native": original["native"],
            "release": original["release"],
            "manifest": original["manifest"],
            "orchestrator_wall_s": original["orchestrator_wall_s"],
            "optimizer_updates": original["optimizer_updates"],
            "cumulative_optimizer_updates_unchanged": original["cumulative_optimizer_updates_unchanged"],
            "orchestrator_pytorch_model_construction": original["orchestrator_pytorch_model_construction"],
            "native_model_loads": original["native_model_loads"],
            "native_material_reexecuted": False,
            "sidecars_unchanged": True,
            "manifest_unchanged": True,
            "frozen_config_unchanged": True,
        },
        "report_repair": {
            "schema": "PSEM-TEACHER-FIT-REPORT-REPAIR-1",
            "native_material_executed": False,
            "model_forward": False,
            "orchestrator_pytorch_model_construction": 0,
            "repaired_teacher_fit_sha256": hashes["teacher_fit.py"],
            "future_config_sha256": hashes["config"],
            "future_config_path": str(config_path),
            "repair_root": str(run_root.relative_to(ROOT).as_posix()),
            "helpers": {name: hashes[name] for name in ("speaker_score_diagnostic.py", "gt_pilot.py", "run_baseline.py", "gt_probe.py")},
            "existing_association_metrics_unchanged": unchanged,
            "dump_default_disclosure": {
                "model_cpp_after_sha256": MODEL_CPP_AFTER_SHA256,
                "lines": "498-512",
                "DUMP_DIR_default": "want_h=true, want_l=true",
                "EXPORT_absent_empty_unrecognized": "falls back to both hidden and logits",
                "EXPORT_logits": "disables hidden; retains required probs and logits",
                "original_no_hidden_expectation_was_not_met": True,
                "incidental_bytes": incidental_total,
                "learning_use_forbidden": True,
                "files_not_deleted": True,
                "bounded_artifact_deviation_accepted_under_AUTONOMY-2": True,
                "not_retroactive_compliance": True,
            },
            "future_export": {
                "TRANSCRIBE_SORTFORMER_EXPORT": "logits",
                "required_files": ["diar.probs.f32", "diar.probs.json", "diar.logits.f32", "diar.logits.json", "diar.trace.json"],
                "forbidden_files": ["diar.hidden.f32", "diar.hidden.json"],
                "cpu_environment_verified_not_exercised_native": True,
            },
        },
        "incidental_dump_inventory": inventory,
        "identities_bound": original["identities_bound"],
        "sources": repaired_sources,
        "findings": list(original.get("findings", [])) + [
            "Class-conditioned 4x4 raw-native and mapped probability means are grouped by existing strict-solo GT labels on the common-valid mask inside each fixed quarter; absent classes are null, not zero.",
            "Source-global min independent-activity-BCE permutation was not refit per quarter.",
            "Original DUMP_DIR default exported incidental hidden and logits; those bytes are pinned and unused for learning. Future material must set EXPORT=logits.",
        ],
    }
    publish_json(run_root / "artifact_inventory.json", inventory)
    publish_json(run_root / "metric_unchanged_proof.json", unchanged)
    publish_json(run_root / "REPAIR_RESULT.json", public)
    publish_json(public_path, public)
    return {
        "status": "completed_with_report_repair",
        "original_public_sha256": ORIGINAL_PUBLIC_SHA256,
        "incidental_bytes": incidental_total,
        "future_config_sha256": hashes["config"],
        "repaired_teacher_fit_sha256": hashes["teacher_fit.py"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "bind", "execute", "repair-report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--authorize-material", action="store_true")
    args = parser.parse_args()
    global MATERIAL_AUTHORIZED
    MATERIAL_AUTHORIZED = bool(args.authorize_material)
    config_path = args.config.resolve()
    run_root = args.run_root.resolve()
    if args.stage == "smoke":
        value = smoke_stage(config_path, run_root)
    elif args.stage == "bind":
        value = bind_stage(config_path, run_root)
    elif args.stage == "repair-report":
        value = repair_report_stage(config_path, run_root)
    else:
        value = execute_stage(config_path, run_root)
    print(json.dumps({"status": value.get("status"), "stage": args.stage}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
