from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import config as ceiling_config
from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import load_sessions
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import decode_scores, session_metrics
from experiments.psem_frozen_ceiling_gate.experiment_support import (
    aggregate_topology,
    intervals_from_manifest,
    monotonic_boundary_matches,
)
from experiments.psem_sortformer_adaptation_depth.protocol import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from experiments.psem_sortformer_adaptation_depth.receipts import paired_source_bootstrap_v1
from experiments.psem_state_corrected_adaptation_gate import arm_runtime
from experiments.psem_state_corrected_adaptation_gate import calibrate as calibrate_mod
from experiments.psem_state_corrected_adaptation_gate import cross_frontier as cross_mod
from experiments.psem_state_corrected_adaptation_gate import frontier as frontier_mod
from experiments.psem_state_corrected_adaptation_gate import frontier_sweep as sweep_mod
from experiments.psem_state_corrected_adaptation_gate.material import mask_calibration
from experiments.psem_state_corrected_adaptation_gate.partition import CALIB_SALT
from experiments.psem_state_corrected_adaptation_gate.stages import fit_calibrators, sha256_file


class PostprocessError(RuntimeError):
    pass


EXPORT_ARTIFACT_ROLE = "issue-121-h-gpu-export"
CALIBRATION_ARTIFACT_ROLE = "issue-121-h-calibration-metrics"
GATE1_DIAGNOSTICS_ROLE = "issue-121-h-gate1-diagnostics"
H_ARM = arm_runtime.ARM_R_H_SC
NPZ_KEYS = ("f0_raw", "cand_raw", "target", "valid", "mapped")
REQUIRED_CALIB = (
    "alimeeting_R1019_M1950",
    "alimeeting_R8001_M8004",
    "ami_ES2013a",
    "ami_ES2013b",
    "ami_ES2013c",
    "ami_ES2013d",
    "ami_IS1008a",
    "ami_IS1009a",
    "ami_TS3004a",
    "ami_TS3006a",
    "ami_TS3007a",
)
REQUIRED_DEV = (
    "alimeeting_R1019_M1928",
    "alimeeting_R1021_M4073",
    "alimeeting_R8009_M8019",
    "ami_EN2009d",
    "ami_ES2002b",
    "ami_ES2009a",
    "ami_ES2009b",
    "ami_ES2009c",
    "ami_ES2009d",
    "ami_ES2015d",
)
AMI_DEV = (
    "ami_EN2009d",
    "ami_ES2002b",
    "ami_ES2009a",
    "ami_ES2009b",
    "ami_ES2009c",
    "ami_ES2009d",
    "ami_ES2015d",
)
ALI_DEV = (
    "alimeeting_R1019_M1928",
    "alimeeting_R1021_M4073",
    "alimeeting_R8009_M8019",
)
FAMILY_OF = {
    **{source_id: "ami_mix_headset" for source_id in AMI_DEV},
    **{source_id: "alimeeting_far_ch0" for source_id in ALI_DEV},
}
CORPUS_OF = {
    **{source_id: "AMI" for source_id in AMI_DEV},
    **{source_id: "AliMeeting" for source_id in ALI_DEV},
}
TIMING_TOLERANCE_MS = 80.0
BOOTSTRAP_OFFSET = 0
KIND_SEED_BASE = {"raw": 0, "calibrated": 200}
HORIZON_SEED_STEP = 40
ENVELOPE_SEED_STEP = 2
BINDING_HASH_FIELDS = (
    "input_hash",
    "checkpoint_hash",
    "partition_hash",
    "weights_hash",
    "code_hash",
)
PARTITION_PAYLOAD_FIELDS = ("fit", "salt", "target_frac")

COMPACT_METRIC_KEYS = (
    "predicted_cut_count",
    "reference_replacement_count",
    "matched_replacement_count",
    "false_cut_count",
    "missed_replacement_count",
    "active_speech_seconds",
    "exclusive_other_contamination_seconds",
    "exclusive_other_contamination_seconds_per_active_speech_hour",
    "logical_episode_exclusive_other_contamination_seconds",
)




def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else int(value)
    return value



def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PostprocessError(f"manifest is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PostprocessError(f"manifest is not an object: {path}")
    return payload


def _as_bool_list(values: Any, count: int, what: str) -> list[bool]:
    listed = [bool(v) for v in values]
    if len(listed) != count:
        raise PostprocessError(f"{what} length differs")
    return listed


def _as_float_list(values: Any, count: int, what: str) -> list[float]:
    listed = [float(v) for v in values]
    if len(listed) != count:
        raise PostprocessError(f"{what} length differs")
    return listed


def corpus_of(source_id: str) -> str:
    corpus = CORPUS_OF.get(str(source_id))
    if corpus is None:
        raise PostprocessError(f"source is outside the frozen DEV set: {source_id}")
    return corpus


def union_probability_grid(prob_lists: Sequence[Sequence[float]]) -> list[float]:
    union: set[float] = set()
    for scores in prob_lists:
        for value in [float(v) for v in scores]:
            if value != float("-inf"):
                union.add(value)
    if not union:
        raise PostprocessError("DEV group has no frontier scores")
    return sorted(union, reverse=True)


def mask_unmapped(values: Sequence[float], mapped: Sequence[bool]) -> list[float]:
    if len(values) != len(mapped):
        raise PostprocessError("score/mapping geometry differs")
    return [float(v) if flag else float("-inf") for v, flag in zip(values, mapped)]


def sigmoid_list(values: Sequence[float]) -> list[float]:
    return [calibrate_mod.sigmoid(float(v)) for v in values]


def _file_sha256(path: Path) -> str:
    return sha256_file(Path(path))


def _load_npz(path: Path) -> dict[str, Any]:
    try:
        with np.load(Path(path), allow_pickle=False) as arrays:
            missing = [key for key in NPZ_KEYS if key not in arrays]
            if missing:
                raise PostprocessError(f"export NPZ lacks {missing}: {path}")
            payload = {key: np.asarray(arrays[key]) for key in NPZ_KEYS}
    except (OSError, ValueError, TypeError) as exc:
        raise PostprocessError(f"export NPZ is corrupted or truncated: {path}: {exc}") from exc
    frames = int(payload["target"].reshape(-1).shape[0])
    if frames <= 0:
        raise PostprocessError(f"export NPZ has no frames: {path}")
    for key in ("f0_raw", "cand_raw", "target", "valid", "mapped"):
        flat = np.asarray(payload[key]).reshape(-1)
        if int(flat.shape[0]) != frames:
            raise PostprocessError(f"export NPZ arrays are not aligned: {path}")
        payload[key] = flat
    if payload["f0_raw"].dtype != np.float64 or payload["cand_raw"].dtype != np.float64:
        raise PostprocessError(f"export NPZ logits must be float64: {path}")
    if payload["target"].dtype != np.float64:
        raise PostprocessError(f"export NPZ target must be float64: {path}")
    return {
        "f0_raw": _as_float_list(payload["f0_raw"], frames, "f0_raw"),
        "cand_raw": _as_float_list(payload["cand_raw"], frames, "cand_raw"),
        "target": _as_float_list(payload["target"], frames, "target"),
        "valid": _as_bool_list(payload["valid"], frames, "valid"),
        "mapped": _as_bool_list(payload["mapped"], frames, "mapped"),
        "frames": frames,
    }


def _require_sorted_list(payload: Any, expected: Sequence[str], what: str) -> list[str]:
    if not isinstance(payload, list) or [str(v) for v in payload] != list(expected):
        raise PostprocessError(f"{what} must be the frozen sorted list {list(expected)}")
    return [str(v) for v in payload]


def _sidecar(table: Any, source_id: str, what: str) -> dict[str, Any]:
    if not isinstance(table, dict) or source_id not in table or not isinstance(table[source_id], dict):
        raise PostprocessError(f"{what} sidecar is missing: {source_id}")
    return dict(table[source_id])


def _resolve_export_file(export_dir: Path, sidecar: Mapping[str, Any], source_id: str) -> Path:
    rel = sidecar.get("file")
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        raise PostprocessError(f"export file path is invalid: {source_id}")
    path = (Path(export_dir) / rel).resolve()
    root = Path(export_dir).resolve()
    if root not in path.parents and path != root:
        raise PostprocessError(f"export file escapes export dir: {source_id}")
    if not path.is_file():
        raise PostprocessError(f"export NPZ is missing: {rel}")
    return path


def _check_hash(path: Path, sidecar: Mapping[str, Any], files: Mapping[str, Any], rel: str) -> str:
    digest = _file_sha256(path)
    declared = sidecar.get("sha256")
    mapped = files.get(rel)
    if declared is not None and str(declared) != digest:
        raise PostprocessError(f"export SHA256 differs: {rel}")
    if mapped is not None and str(mapped) != digest:
        raise PostprocessError(f"export files{{}} SHA256 differs: {rel}")
    if declared is None and mapped is None:
        raise PostprocessError(f"export SHA256 is missing: {rel}")
    return digest


def partition_hash_for(fit: Sequence[str], calib: Sequence[str], salt: Any, target_frac: Any) -> str:
    return arm_runtime.canonical_sha256(
        {
            "fit": sorted(str(v) for v in fit),
            "calib": sorted(str(v) for v in calib),
            "salt": str(salt),
            "target_frac": float(target_frac),
        }
    )


def _partition_payload(manifest: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    missing: list[str] = []
    nested = manifest.get("partition")
    sources = [binding, nested if isinstance(nested, dict) else {}, manifest]
    for field in PARTITION_PAYLOAD_FIELDS:
        value = None
        for source in sources:
            if isinstance(source, dict) and field in source:
                value = source[field]
                break
        if value is None:
            missing.append(field)
        else:
            payload[field] = value
    if missing:
        raise PostprocessError(
            "export lacks partition payload to verify partition_hash: "
            + ", ".join(missing)
            + "; required fit (sorted TRAIN-FIT ids), salt, target_frac so partition_hash "
            "= canonical_sha256({fit, calib: CALIB11, salt, target_frac})"
        )
    fit = payload["fit"]
    if not isinstance(fit, list) or not fit or not all(isinstance(item, str) for item in fit):
        raise PostprocessError("export FIT list is invalid")
    if set(fit) & set(REQUIRED_CALIB):
        raise PostprocessError("TRAIN-FIT overlaps frozen CALIB11")
    try:
        frac = float(payload["target_frac"])
    except (TypeError, ValueError) as exc:
        raise PostprocessError("export target_frac is invalid") from exc
    return {"fit": sorted(fit), "salt": str(payload["salt"]), "target_frac": frac}


def verify_export_binding(manifest: Mapping[str, Any], calib_sources: Sequence[str]) -> dict[str, Any]:
    binding = manifest.get("binding")
    if not isinstance(binding, dict):
        raise PostprocessError("export binding is missing")
    required = ("arm", "seed", *BINDING_HASH_FIELDS, "optimizer_contract")
    missing = [name for name in required if name not in binding]
    if missing:
        raise PostprocessError(f"export binding lacks {missing}")
    arm = str(binding.get("arm") or "")
    if arm != H_ARM:
        raise PostprocessError(f"export binding arm is {arm!r}, required {H_ARM!r}")
    try:
        seed = int(binding["seed"])
    except (TypeError, ValueError) as exc:
        raise PostprocessError("export binding seed is invalid") from exc
    if seed not in arm_runtime.ALLOWED_SEEDS:
        raise PostprocessError(f"export binding seed is not authorized: {seed}")
    for name in BINDING_HASH_FIELDS:
        try:
            arm_runtime._require_hash(str(binding[name]), name)
        except arm_runtime.ArmError as exc:
            raise PostprocessError(str(exc)) from exc
    contract = binding.get("optimizer_contract")
    if contract != arm_runtime.OPTIMIZER_CONTRACT:
        raise PostprocessError("export optimizer_contract differs from the frozen H contract")
    if list(calib_sources) != list(REQUIRED_CALIB):
        raise PostprocessError("export CALIB identity differs from frozen CALIB11")
    payload = _partition_payload(manifest, binding)
    expected = partition_hash_for(
        payload["fit"], REQUIRED_CALIB, payload["salt"], payload["target_frac"]
    )
    if str(binding["partition_hash"]) != expected:
        raise PostprocessError(
            "export partition_hash does not match frozen CALIB11 partition payload "
            f"(got {binding['partition_hash']}, expected {expected})"
        )
    return {
        "binding": dict(binding),
        "arm": arm,
        "seed": seed,
        "partition": payload,
        "partition_hash": expected,
    }


def validate_export_manifest(export_dir: Path) -> dict[str, Any]:
    root = Path(export_dir)
    manifest_path = root / "gpu_export_manifest.json"
    if not manifest_path.is_file():
        raise PostprocessError("gpu_export_manifest.json is missing")
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_role") != EXPORT_ARTIFACT_ROLE:
        raise PostprocessError(
            f"export artifact_role is {manifest.get('artifact_role')!r}, required {EXPORT_ARTIFACT_ROLE!r}"
        )
    verified = verify_export_binding(manifest, _require_sorted_list(manifest.get("calib_sources"), REQUIRED_CALIB, "calib_sources"))
    binding = verified["binding"]
    arm = verified["arm"]
    calib_sources = list(REQUIRED_CALIB)
    dev_sources = _require_sorted_list(manifest.get("dev_sources"), REQUIRED_DEV, "dev_sources")
    listed = calib_sources + dev_sources
    if any("eval" in str(source_id).lower() for source_id in listed):
        raise PostprocessError("EVAL ids are forbidden in the H export")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise PostprocessError("export files map is missing")
    return {
        "manifest": manifest,
        "binding": dict(binding),
        "arm": arm,
        "seed": verified["seed"],
        "calib_sources": calib_sources,
        "dev_sources": dev_sources,
        "files": dict(files),
        "path": manifest_path,
    }


def load_validated_export(export_dir: Path, sessions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    validated = validate_export_manifest(export_dir)
    root = Path(export_dir)
    manifest = validated["manifest"]
    session_map = dict(sessions or {})
    if not session_map:
        session_map = {
            session.source_id: session
            for session in load_sessions()
            if str(getattr(session, "role", "")).lower() == "dev"
        }
    missing = [source_id for source_id in validated["dev_sources"] if source_id not in session_map]
    if missing:
        raise PostprocessError(f"frozen DEV sessions are missing: {missing}")
    extra_eval = [
        source_id
        for source_id in list(manifest.get("calib", {})) + list(manifest.get("dev", {}))
        if "eval" in str(source_id).lower()
    ]
    if extra_eval:
        raise PostprocessError(f"EVAL ids are forbidden in the H export: {extra_eval}")
    calib_arrays: dict[str, Any] = {}
    for source_id in validated["calib_sources"]:
        sidecar = _sidecar(manifest.get("calib"), source_id, "calib")
        path = _resolve_export_file(root, sidecar, source_id)
        digest = _check_hash(path, sidecar, validated["files"], str(sidecar["file"]))
        arrays = _load_npz(path)
        declared = sidecar.get("frames")
        if declared is not None and int(declared) != arrays["frames"]:
            raise PostprocessError(f"CALIB frame count differs: {source_id}")
        arrays["sha256"] = digest
        arrays["sidecar"] = sidecar
        calib_arrays[source_id] = arrays
    dev_arrays: dict[str, Any] = {}
    for source_id in validated["dev_sources"]:
        sidecar = _sidecar(manifest.get("dev"), source_id, "dev")
        path = _resolve_export_file(root, sidecar, source_id)
        digest = _check_hash(path, sidecar, validated["files"], str(sidecar["file"]))
        arrays = _load_npz(path)
        session = session_map[source_id]
        session_frames = int(np.asarray(session.starts).reshape(-1).shape[0])
        if arrays["frames"] != session_frames:
            raise PostprocessError(
                f"DEV NPZ frames {arrays['frames']} differ from frozen session {session_frames}: {source_id}"
            )
        declared = sidecar.get("frames")
        if declared is not None and int(declared) != arrays["frames"]:
            raise PostprocessError(f"DEV frame count differs: {source_id}")
        family = sidecar.get("family", FAMILY_OF[source_id])
        if str(family) != FAMILY_OF[source_id]:
            raise PostprocessError(f"DEV family differs: {source_id}")
        arrays["sha256"] = digest
        arrays["sidecar"] = sidecar
        arrays["session"] = session
        dev_arrays[source_id] = arrays
    return {**validated, "calib": calib_arrays, "dev": dev_arrays, "sessions": session_map}


def fit_calib_from_export(calib: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    f0_all: list[float] = []
    cand_all: list[float] = []
    target_all: list[float] = []
    coverage: dict[str, Any] = {}
    for source_id in sorted(calib):
        entry = calib[source_id]
        kept, stats = mask_calibration(entry["target"], entry["valid"], entry["mapped"])
        coverage[source_id] = dict(stats)
        for index in kept:
            f0_all.append(float(entry["f0_raw"][index]))
            cand_all.append(float(entry["cand_raw"][index]))
            target_all.append(float(entry["target"][index]))
    if not f0_all:
        raise PostprocessError("TRAIN-CALIB produced no kept frames")
    f0_fit, cand_fit = fit_calibrators(f0_all, cand_all, target_all)
    return f0_fit, cand_fit, {
        "frames": len(target_all),
        "sources": sorted(calib),
        "coverage": coverage,
        "f0_raw": f0_all,
        "cand_raw": cand_all,
        "target": target_all,
    }


def prepare_dev_member(
    source_id: str,
    arrays: Mapping[str, Any],
    cal_f0: Mapping[str, Any],
    cal_cand: Mapping[str, Any],
) -> dict[str, Any]:
    mapped = [bool(v) for v in arrays["mapped"]]
    f0_raw = [float(v) for v in arrays["f0_raw"]]
    cand_raw = [float(v) for v in arrays["cand_raw"]]
    target = [float(v) for v in arrays["target"]]
    f0_cal = calibrate_mod.apply_affine(f0_raw, float(cal_f0["slope"]), float(cal_f0["intercept"]))
    cand_cal = calibrate_mod.apply_affine(
        cand_raw, float(cal_cand["slope"]), float(cal_cand["intercept"])
    )
    kept = [i for i, flag in enumerate(mapped) if flag]
    if not kept:
        raise PostprocessError(f"DEV has no mapped frames: {source_id}")
    kept_target = [target[i] for i in kept]
    kept_raw_prob = [calibrate_mod.sigmoid(cand_raw[i]) for i in kept]
    return {
        "source_id": source_id,
        "corpus": corpus_of(source_id),
        "dev": arrays["session"],
        "frames": len(target),
        "mapped": mapped,
        "target": target,
        "f0_raw": f0_raw,
        "cand_raw": cand_raw,
        "f0_cal": f0_cal,
        "cand_cal": cand_cal,
        "f0_prob": mask_unmapped(sigmoid_list(f0_raw), mapped),
        "cand_raw_prob": mask_unmapped(sigmoid_list(cand_raw), mapped),
        "cand_cal_prob": mask_unmapped(sigmoid_list(cand_cal), mapped),
        "kept": kept,
        "raw_ap": calibrate_mod.average_precision(kept_raw_prob, kept_target),
        "f0_cal_nll": calibrate_mod.nll_loss([f0_cal[i] for i in kept], kept_target),
        "f0_cal_brier": calibrate_mod.brier_score([f0_cal[i] for i in kept], kept_target),
        "candidate_cal_nll": calibrate_mod.nll_loss([cand_cal[i] for i in kept], kept_target),
        "candidate_cal_brier": calibrate_mod.brier_score([cand_cal[i] for i in kept], kept_target),
    }


def _point_from_rows(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, float]:
    totals = cross_mod.sum_primitives(rows)
    try:
        return cross_mod.pooled_point_from_sums(
            totals["false_cut_count"],
            totals["active_speech_seconds"],
            totals["reference_replacement_count"],
            totals["missed_replacement_count"],
            totals["exclusive_other_contamination_seconds"],
            float(threshold),
        )
    except cross_mod.CrossFrontierError as exc:
        raise PostprocessError(str(exc)) from exc


def _index_kind(primitives: Mapping[str, Any], source_id: str, kind: str, horizon_ms: int) -> dict[float, Mapping[str, Any]]:
    rows = primitives[source_id][kind][int(horizon_ms)]
    return cross_mod.index_threshold_rows(rows)


def _block_from_points(
    points: Sequence[Mapping[str, float]],
    reference: Mapping[str, float],
    diagnostics: Mapping[str, Any],
    what: str,
) -> dict[str, Any]:
    try:
        return cross_mod.build_block(points, reference, diagnostics, what)
    except cross_mod.CrossFrontierError as exc:
        raise PostprocessError(str(exc)) from exc


def build_canonical_frontier(
    members: Mapping[str, Mapping[str, Any]],
    primitives: Mapping[str, Any],
    grids: Mapping[str, Sequence[float]],
    binding: Mapping[str, Any],
    phase: Mapping[str, Any],
) -> dict[str, Any]:
    ami = [source_id for source_id in REQUIRED_DEV if CORPUS_OF[source_id] == "AMI"]
    ali = [source_id for source_id in REQUIRED_DEV if CORPUS_OF[source_id] == "AliMeeting"]
    groups = {
        "ami": ami,
        "alimeeting": ali,
        "pooled": list(REQUIRED_DEV),
        "macro": list(REQUIRED_DEV),
    }
    indexed: dict[str, Any] = {}
    for source_id in REQUIRED_DEV:
        indexed[source_id] = {}
        for kind in ("raw", "calibrated", "f0"):
            indexed[source_id][kind] = {}
            for horizon_ms in frontier_mod.HORIZONS_MS:
                indexed[source_id][kind][int(horizon_ms)] = _index_kind(
                    primitives, source_id, kind, int(horizon_ms)
                )
    sources_out: dict[str, Any] = {}
    for source_id in REQUIRED_DEV:
        member = members[source_id]
        sources_out[source_id] = {}
        for horizon_ms in frontier_mod.HORIZONS_MS:
            key = str(horizon_ms)
            sources_out[source_id][key] = {}
            f0_point = _point_from_rows(
                [indexed[source_id]["f0"][int(horizon_ms)][frontier_mod.RAW_REFERENCE_THRESHOLD]],
                frontier_mod.RAW_REFERENCE_THRESHOLD,
            )
            diagnostics = {
                "frames": member["frames"],
                "kept_frames": len(member["kept"]),
                "raw_ap": member["raw_ap"],
                "f0_cal_nll": member["f0_cal_nll"],
                "f0_cal_brier": member["f0_cal_brier"],
                "candidate_cal_nll": member["candidate_cal_nll"],
                "candidate_cal_brier": member["candidate_cal_brier"],
            }
            for kind in cross_mod.KINDS:
                local_grid = union_probability_grid([member["cand_raw_prob" if kind == "raw" else "cand_cal_prob"]])
                points = [
                    _point_from_rows([indexed[source_id][kind][int(horizon_ms)][float(threshold)]], float(threshold))
                    for threshold in local_grid
                ]
                sources_out[source_id][key][kind] = _block_from_points(
                    points, f0_point, diagnostics, f"{source_id}/{kind}/{key}"
                )
    horizons_out: dict[str, Any] = {}
    for horizon_ms in frontier_mod.HORIZONS_MS:
        key = str(horizon_ms)
        horizons_out[key] = {}
        group_blocks: dict[str, Any] = {}
        for group in ("ami", "alimeeting", "pooled"):
            group_blocks[group] = {}
            members_ids = groups[group]
            f0_point = _point_from_rows(
                [
                    indexed[source_id]["f0"][int(horizon_ms)][frontier_mod.RAW_REFERENCE_THRESHOLD]
                    for source_id in members_ids
                ],
                frontier_mod.RAW_REFERENCE_THRESHOLD,
            )
            diagnostics = {"members": list(members_ids), "member_count": len(members_ids)}
            for kind in cross_mod.KINDS:
                points = [
                    _point_from_rows(
                        [indexed[source_id][kind][int(horizon_ms)][float(threshold)] for source_id in members_ids],
                        float(threshold),
                    )
                    for threshold in grids[kind]
                ]
                group_blocks[group][kind] = _block_from_points(
                    points, f0_point, diagnostics, f"{group}/{kind}/{key}"
                )
            horizons_out[key][group] = group_blocks[group]
        horizons_out[key]["macro"] = {}
        for kind in cross_mod.KINDS:
            try:
                macro_points = cross_mod.macro_average_points(
                    group_blocks["ami"][kind]["points"],
                    group_blocks["alimeeting"][kind]["points"],
                    what=f"macro/{kind}/{key}",
                )
                macro_ref = cross_mod.macro_average_reference(
                    group_blocks["ami"][kind]["reference"],
                    group_blocks["alimeeting"][kind]["reference"],
                    what=f"macro/{kind}/{key}",
                )
            except cross_mod.CrossFrontierError as exc:
                raise PostprocessError(f"macro grids differ: {key}/{kind}: {exc}") from exc
            horizons_out[key]["macro"][kind] = _block_from_points(
                macro_points,
                macro_ref,
                {"members": list(groups["macro"]), "member_count": len(groups["macro"])},
                f"macro/{kind}/{key}",
            )
    doc = {
        "artifact_role": cross_mod.ARTIFACT_ROLE,
        "version": cross_mod.CANONICAL_VERSION,
        "arm": H_ARM,
        "binding": dict(binding),
        "horizons_ms": list(frontier_mod.HORIZONS_MS),
        "group_order": list(cross_mod.GROUP_ORDER),
        "horizons": horizons_out,
        "sources": sources_out,
        "phase": dict(phase),
    }
    try:
        cross_mod.validate_canonical(doc)
    except cross_mod.CrossFrontierError as exc:
        raise PostprocessError(str(exc)) from exc
    return doc


def _compact_session_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    out = {key: metrics.get(key) for key in COMPACT_METRIC_KEYS}
    out["replacement_emit_delay_ms"] = dict(metrics.get("replacement_emit_delay_ms") or {})
    out["model_evidence_delay_ms"] = dict(metrics.get("model_evidence_delay_ms") or {})
    out["backdated_boundary_error_ms"] = dict(metrics.get("backdated_boundary_error_ms") or {})
    out["contamination_seconds_per_true_replacement"] = dict(
        metrics.get("contamination_seconds_per_true_replacement") or {}
    )
    topology = metrics.get("topology") or {}
    out["topology"] = {
        str(name): {str(k): int(v) for k, v in dict(values).items()}
        for name, values in dict(topology).items()
    }
    return _jsonable(out)


def _score_array(values: Sequence[float]) -> Any:
    return np.asarray(list(values), dtype=np.float64)


def selected_session_metrics(
    members: Mapping[str, Mapping[str, Any]],
    frontier: Mapping[str, Any],
) -> dict[str, Any]:
    cache: dict[tuple[str, str, int, float], dict[str, Any]] = {}
    details: dict[str, Any] = {}
    for horizon_ms in frontier_mod.HORIZONS_MS:
        horizon_key = str(horizon_ms)
        details[horizon_key] = {}
        for kind in cross_mod.KINDS:
            block = frontier["horizons"][horizon_key]["macro"][kind]
            wanted: list[tuple[str, float]] = [("f0", frontier_mod.RAW_REFERENCE_THRESHOLD)]
            for envelope_name in ("c_envelope", "m_envelope"):
                point = block.get(envelope_name)
                if isinstance(point, dict):
                    wanted.append((envelope_name, float(point["threshold"])))
            unique_wanted: list[tuple[str, float]] = []
            seen: set[tuple[str, float]] = set()
            for name, threshold in wanted:
                key = (name, float(threshold))
                if key in seen:
                    continue
                seen.add(key)
                unique_wanted.append((name, float(threshold)))
            kind_details: dict[str, Any] = {}
            for source_id in REQUIRED_DEV:
                member = members[source_id]
                source_row: dict[str, Any] = {}
                for name, threshold in unique_wanted:
                    score_kind = "f0_prob" if name == "f0" else (
                        "cand_raw_prob" if kind == "raw" else "cand_cal_prob"
                    )
                    cache_key = (source_id, score_kind, int(horizon_ms), float(threshold))
                    if cache_key not in cache:
                        events = decode_scores(
                            member["dev"],
                            _score_array(member[score_kind]),
                            threshold=float(threshold),
                            confirmation_ms=int(horizon_ms),
                        )
                        cache[cache_key] = _compact_session_metrics(session_metrics(member["dev"], events))
                    source_row[name] = {
                        "threshold": float(threshold),
                        "metrics": cache[cache_key],
                    }
                kind_details[source_id] = source_row
            topology_by_point: dict[str, Any] = {}
            for name, _threshold in unique_wanted:
                rows = [{"topology": kind_details[source_id][name]["metrics"]["topology"]} for source_id in REQUIRED_DEV]
                topology_by_point[name] = aggregate_topology(rows)
            details[horizon_key][kind] = {
                "points": unique_wanted,
                "sources": kind_details,
                "topology": topology_by_point,
            }
    return details


def _p90(metrics: Mapping[str, Any]) -> float | None:
    value = (metrics.get("replacement_emit_delay_ms") or {}).get("p90")
    if value is None:
        return None
    return float(value)


def _delta(front: float, back: float) -> float:
    return float(front) - float(back)


def _bootstrap_interval(deltas: Mapping[str, float], seed: int) -> dict[str, Any]:
    ordered = {str(key): float(deltas[key]) for key in sorted(deltas)}
    interval = paired_source_bootstrap_v1(ordered, seed=int(seed), resamples=int(BOOTSTRAP_RESAMPLES))
    return {
        **interval,
        "seed": int(seed),
        "resamples": int(BOOTSTRAP_RESAMPLES),
        "algorithm": "paired_source_bootstrap_v1",
        "unit": "source_or_meeting",
        "paired_source_deltas": ordered,
        "point_estimate": sum(ordered.values()) / len(ordered),
        "support_source_count": len(ordered),
        "aggregation": "source_mean_not_pooled_rate",
        "negative_is_favorable": True,
        "wholly_favorable_upper_lt_0": bool(float(interval["upper"]) < 0.0),
    }


def _seed_for(kind: str, horizon_ms: int, envelope_index: int, metric_index: int, corpus_index: int | None) -> int:
    offset = (
        int(KIND_SEED_BASE[kind])
        + list(frontier_mod.HORIZONS_MS).index(int(horizon_ms)) * HORIZON_SEED_STEP
        + int(envelope_index) * ENVELOPE_SEED_STEP
        + BOOTSTRAP_OFFSET
    )
    if corpus_index is None:
        return int(BOOTSTRAP_SEED) + offset + int(metric_index)
    return int(BOOTSTRAP_SEED) + offset + 20 + int(corpus_index) * 10 + int(metric_index)


def _meeting_rate(metrics: Mapping[str, Any], field: str) -> float:
    hours = float(metrics["active_speech_seconds"]) / 3600.0
    refs = float(metrics["reference_replacement_count"])
    if field == "contamination":
        return float(metrics["exclusive_other_contamination_seconds_per_active_speech_hour"])
    if field == "miss_rate":
        return float(metrics["missed_replacement_count"]) / refs
    if field == "false_cuts_per_hour":
        return float(metrics["false_cut_count"]) / hours
    raise PostprocessError(f"unknown meeting rate: {field}")

def _source_envelope_metrics(
    selected_kind: Mapping[str, Any], source_id: str, envelope_name: str
) -> Mapping[str, Any]:
    source_row = selected_kind.get("sources", {}).get(source_id)
    if not isinstance(source_row, dict) or envelope_name not in source_row:
        raise PostprocessError(f"selected {envelope_name} metrics are missing: {source_id}")
    entry = source_row[envelope_name]
    if not isinstance(entry, dict) or "metrics" not in entry:
        raise PostprocessError(f"selected {envelope_name} metrics are missing: {source_id}")
    return entry["metrics"]


def _envelope_topology(selected_kind: Mapping[str, Any], envelope_name: str) -> Any:
    topology = selected_kind.get("topology")
    if not isinstance(topology, dict) or envelope_name not in topology:
        return {}
    return topology[envelope_name]



def build_gate1_diagnostics(
    members: Mapping[str, Mapping[str, Any]],
    frontier: Mapping[str, Any],
    selected: Mapping[str, Any],
    calibration: Mapping[str, Any],
    export_meta: Mapping[str, Any],
) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    for horizon_ms in frontier_mod.HORIZONS_MS:
        horizon_key = str(horizon_ms)
        horizons[horizon_key] = {}
        for kind in cross_mod.KINDS:
            macro = frontier["horizons"][horizon_key]["macro"][kind]
            ami = frontier["horizons"][horizon_key]["ami"][kind]
            ali = frontier["horizons"][horizon_key]["alimeeting"][kind]
            pooled = frontier["horizons"][horizon_key]["pooled"][kind]
            selected_kind = selected[horizon_key][kind]
            envelopes: dict[str, Any] = {}
            for envelope_index, envelope_name in enumerate(("c_envelope", "m_envelope")):
                point = macro.get(envelope_name)
                if not isinstance(point, dict):
                    envelopes[envelope_name] = None
                    continue
                f0_metrics = {
                    source_id: selected_kind["sources"][source_id]["f0"]["metrics"]
                    for source_id in REQUIRED_DEV
                }
                h_metrics = {
                    source_id: _source_envelope_metrics(selected_kind, source_id, envelope_name)
                    for source_id in REQUIRED_DEV
                }
                meeting_rows = {}
                contamination_deltas = {}
                miss_deltas = {}
                for source_id in REQUIRED_DEV:
                    h_row = h_metrics[source_id]
                    f0_row = f0_metrics[source_id]
                    h_p90 = _p90(h_row)
                    f0_p90 = _p90(f0_row)
                    timing_delta = None if h_p90 is None or f0_p90 is None else h_p90 - f0_p90
                    meeting_rows[source_id] = {
                        "corpus": CORPUS_OF[source_id],
                        "h": {
                            "contamination": _meeting_rate(h_row, "contamination"),
                            "miss_rate": _meeting_rate(h_row, "miss_rate"),
                            "false_cuts_per_hour": _meeting_rate(h_row, "false_cuts_per_hour"),
                            "p50_delay_ms": (h_row.get("replacement_emit_delay_ms") or {}).get("p50"),
                            "p90_delay_ms": h_p90,
                            "predicted_cut_count": h_row.get("predicted_cut_count"),
                            "matched_replacement_count": h_row.get("matched_replacement_count"),
                            "false_cut_count": h_row.get("false_cut_count"),
                            "missed_replacement_count": h_row.get("missed_replacement_count"),
                        },
                        "f0": {
                            "contamination": _meeting_rate(f0_row, "contamination"),
                            "miss_rate": _meeting_rate(f0_row, "miss_rate"),
                            "false_cuts_per_hour": _meeting_rate(f0_row, "false_cuts_per_hour"),
                            "p50_delay_ms": (f0_row.get("replacement_emit_delay_ms") or {}).get("p50"),
                            "p90_delay_ms": f0_p90,
                        },
                        "delta_h_minus_f0": {
                            "contamination": _delta(_meeting_rate(h_row, "contamination"), _meeting_rate(f0_row, "contamination")),
                            "miss_rate": _delta(_meeting_rate(h_row, "miss_rate"), _meeting_rate(f0_row, "miss_rate")),
                            "false_cuts_per_hour": _delta(
                                _meeting_rate(h_row, "false_cuts_per_hour"),
                                _meeting_rate(f0_row, "false_cuts_per_hour"),
                            ),
                            "p90_delay_ms": timing_delta,
                        },
                        "timing_within_f0_plus_80ms": (
                            None if timing_delta is None else bool(timing_delta <= TIMING_TOLERANCE_MS)
                        ),
                    }
                    contamination_deltas[source_id] = meeting_rows[source_id]["delta_h_minus_f0"]["contamination"]
                    miss_deltas[source_id] = meeting_rows[source_id]["delta_h_minus_f0"]["miss_rate"]
                bootstrap = {
                    "contamination": {
                        "pooled_source_mean": _bootstrap_interval(
                            contamination_deltas,
                            _seed_for(kind, int(horizon_ms), envelope_index, 0, None),
                        ),
                        "ami": _bootstrap_interval(
                            {sid: contamination_deltas[sid] for sid in AMI_DEV},
                            _seed_for(kind, int(horizon_ms), envelope_index, 0, 0),
                        ),
                        "alimeeting": _bootstrap_interval(
                            {sid: contamination_deltas[sid] for sid in ALI_DEV},
                            _seed_for(kind, int(horizon_ms), envelope_index, 0, 1),
                        ),
                    },
                    "miss_rate": {
                        "pooled_source_mean": _bootstrap_interval(
                            miss_deltas,
                            _seed_for(kind, int(horizon_ms), envelope_index, 1, None),
                        ),
                        "ami": _bootstrap_interval(
                            {sid: miss_deltas[sid] for sid in AMI_DEV},
                            _seed_for(kind, int(horizon_ms), envelope_index, 1, 0),
                        ),
                        "alimeeting": _bootstrap_interval(
                            {sid: miss_deltas[sid] for sid in ALI_DEV},
                            _seed_for(kind, int(horizon_ms), envelope_index, 1, 1),
                        ),
                    },
                    "note": "pooled_source_mean is a meeting-mean CI, not a pooled-rate CI and not a macro CI",
                }
                jointly = bool(
                    bootstrap["contamination"]["pooled_source_mean"]["wholly_favorable_upper_lt_0"]
                    or bootstrap["miss_rate"]["pooled_source_mean"]["wholly_favorable_upper_lt_0"]
                )
                l1o = {}
                indexed_h = {
                    source_id: h_metrics[source_id]
                    for source_id in REQUIRED_DEV
                }
                for left_out in REQUIRED_DEV:
                    remain = [sid for sid in REQUIRED_DEV if sid != left_out]
                    remain_ami = [sid for sid in remain if sid in AMI_DEV]
                    remain_ali = [sid for sid in remain if sid in ALI_DEV]
                    def _rate_from(ids: list[str], field: str) -> float | None:
                        if not ids:
                            return None
                        hours = sum(float(indexed_h[sid]["active_speech_seconds"]) for sid in ids) / 3600.0
                        refs = sum(float(indexed_h[sid]["reference_replacement_count"]) for sid in ids)
                        contamination = sum(
                            float(indexed_h[sid]["exclusive_other_contamination_seconds"]) for sid in ids
                        )
                        missed = sum(float(indexed_h[sid]["missed_replacement_count"]) for sid in ids)
                        cuts = sum(float(indexed_h[sid]["false_cut_count"]) for sid in ids)
                        if field == "contamination":
                            return contamination / hours if hours else None
                        if field == "miss_rate":
                            return missed / refs if refs else None
                        return cuts / hours if hours else None
                    ami_c = _rate_from(remain_ami, "contamination")
                    ali_c = _rate_from(remain_ali, "contamination")
                    ami_m = _rate_from(remain_ami, "miss_rate")
                    ali_m = _rate_from(remain_ali, "miss_rate")
                    l1o[left_out] = {
                        "pooled": {
                            "contamination": _rate_from(remain, "contamination"),
                            "miss_rate": _rate_from(remain, "miss_rate"),
                            "false_cuts_per_hour": _rate_from(remain, "false_cuts_per_hour"),
                        },
                        "macro": {
                            "contamination": None if ami_c is None or ali_c is None else (ami_c + ali_c) / 2.0,
                            "miss_rate": None if ami_m is None or ali_m is None else (ami_m + ali_m) / 2.0,
                        },
                    }
                f0_p90s = [_p90(f0_metrics[sid]) for sid in REQUIRED_DEV]
                h_p90s = [_p90(h_metrics[sid]) for sid in REQUIRED_DEV]
                envelopes[envelope_name] = {
                    "threshold": float(point["threshold"]),
                    "macro_point": dict(point),
                    "useful_macro": bool(macro["useful"]),
                    "ami_point": dict(ami.get(envelope_name) or {}),
                    "alimeeting_point": dict(ali.get(envelope_name) or {}),
                    "pooled_point": dict(pooled.get(envelope_name) or {}),
                    "jointly_useful_macro": bool(
                        float(point["contamination"]) < float(macro["reference"]["contamination"])
                        and float(point["miss_rate"]) < float(macro["reference"]["miss_rate"])
                        and float(point["false_cuts_per_hour"]) <= float(macro["budget"])
                    ),
                    "meetings": meeting_rows,
                    "bootstrap": bootstrap,
                    "jointly_favorable_ci_contamination_or_miss": jointly,
                    "leave_one_meeting_out": l1o,
                    "topology": _envelope_topology(selected_kind, envelope_name),
                    "timing": {
                        "criterion_ms": TIMING_TOLERANCE_MS,
                        "meetings_within_f0_plus_80ms": {
                            source_id: meeting_rows[source_id]["timing_within_f0_plus_80ms"]
                            for source_id in REQUIRED_DEV
                        },
                        "note": "p90 delay must not exceed F0 by more than one native frame (80 ms); no winner utility is applied",
                    },
                    "p90_present": {
                        "f0": sum(v is not None for v in f0_p90s),
                        "h": sum(v is not None for v in h_p90s),
                    },
                }
            mapping = {
                source_id: {
                    "mapping_mapped": int(members[source_id]["sidecar"].get("mapping_mapped", sum(members[source_id]["mapped"]))),
                    "mapping_total": int(members[source_id]["sidecar"].get("mapping_total", members[source_id]["frames"])),
                    "unmapped_frames": int(
                        members[source_id]["sidecar"].get(
                            "unmapped_frames",
                            members[source_id]["frames"] - sum(members[source_id]["mapped"]),
                        )
                    ),
                    "kept_frames": int(members[source_id]["sidecar"].get("kept_frames", len(members[source_id]["kept"]))),
                    "coverage": members[source_id]["sidecar"].get("coverage"),
                }
                for source_id in REQUIRED_DEV
            }
            horizons[horizon_key][kind] = {
                "reference": dict(macro["reference"]),
                "budget": float(macro["budget"]),
                "useful": bool(macro["useful"]),
                "envelopes": envelopes,
                "mapping": mapping,
                "ranking": {
                    source_id: {
                        "raw_ap": members[source_id]["raw_ap"],
                        "f0_cal_nll": members[source_id]["f0_cal_nll"],
                        "f0_cal_brier": members[source_id]["f0_cal_brier"],
                        "candidate_cal_nll": members[source_id]["candidate_cal_nll"],
                        "candidate_cal_brier": members[source_id]["candidate_cal_brier"],
                    }
                    for source_id in REQUIRED_DEV
                },
            }
    return {
        "artifact_role": GATE1_DIAGNOSTICS_ROLE,
        "arm": H_ARM,
        "human_adjudication_required": True,
        "gate_receipt_emitted": False,
        "t2_opened": False,
        "eval_opened": False,
        "confirmation_seed_authorized": False,
        "binding": dict(export_meta.get("binding") or {}),
        "seed": export_meta.get("seed"),
        "calibration": {
            "f0": {k: calibration["f0"][k] for k in ("slope", "intercept", "nll", "brier", "ap", "raw_nll", "raw_brier")},
            "candidate": {
                k: calibration["candidate"][k]
                for k in ("slope", "intercept", "nll", "brier", "ap", "raw_nll", "raw_brier")
            },
            "frames": calibration["frames"],
            "sources": list(calibration["sources"]),
        },
        "bootstrap_contract": {
            "algorithm": "paired_source_bootstrap_v1",
            "seed_base": int(BOOTSTRAP_SEED),
            "resamples": int(BOOTSTRAP_RESAMPLES),
            "ci": 0.95,
            "unit": "source_or_meeting",
            "delta": "H_minus_F0",
            "negative_is_favorable": True,
            "wholly_favorable": "upper<0 on contamination OR miss_rate at selected C/M",
            "pooled_is_not_macro": True,
            "kind_seed_base": dict(KIND_SEED_BASE),
            "horizon_seed_step": HORIZON_SEED_STEP,
            "envelope_seed_step": ENVELOPE_SEED_STEP,
        },
        "timing_criterion_ms": TIMING_TOLERANCE_MS,
        "no_invented_winner_utility": True,
        "no_numeric_one_meeting_cutoff": True,
        "no_numeric_one_topology_cutoff": True,
        "horizons": horizons,
        "dev_sources": list(REQUIRED_DEV),
        "calib_sources": list(REQUIRED_CALIB),
    }


def render_decision_evidence(diagnostics: Mapping[str, Any]) -> str:
    lines = [
        "# Gate 1 decision evidence (R-H-SC vs F0)",
        "",
        "This report is evidence for human Gate 1 adjudication.",
        "It is not a gate receipt, confirmation authorization, or T2/EVAL opening.",
        "",
        f"Arm: {diagnostics['arm']}",
        f"Seed: {diagnostics.get('seed')}",
        f"Calib frames: {diagnostics['calibration']['frames']}",
        f"DEV meetings: {len(diagnostics['dev_sources'])} (AMI {len(AMI_DEV)}, AliMeeting {len(ALI_DEV)})",
        "",
        "Primary aggregation: equal-corpus macro, then AMI, AliMeeting, pooled.",
        "Bootstrap CIs are meeting-mean paired-source intervals; they are not pooled-rate or macro CIs.",
        "No winner utility is applied. Timing criterion is p90 delay <= F0 + 80 ms.",
        "Per-meeting and leave-one-meeting-out rows are listed without numeric dominance cutoffs.",
        "",
    ]
    for horizon_ms in frontier_mod.HORIZONS_MS:
        horizon_key = str(horizon_ms)
        lines.append(f"## Horizon {horizon_ms} ms")
        for kind in cross_mod.KINDS:
            node = diagnostics["horizons"][horizon_key][kind]
            lines.append(f"### {kind}")
            lines.append(
                f"- F0 raw@0.5 reference: contamination={node['reference']['contamination']:.6g} "
                f"miss={node['reference']['miss_rate']:.6g} "
                f"false_cuts/h={node['reference']['false_cuts_per_hour']:.6g}"
            )
            lines.append(f"- macro useful flag: {node['useful']}")
            for envelope_name in ("c_envelope", "m_envelope"):
                env = node["envelopes"].get(envelope_name)
                if not env:
                    lines.append(f"- {envelope_name}: none within F0 false-cut budget")
                    continue
                point = env["macro_point"]
                boot = env["bootstrap"]
                lines.append(
                    f"- {envelope_name} threshold={point['threshold']:.6g} "
                    f"contamination={point['contamination']:.6g} miss={point['miss_rate']:.6g} "
                    f"false_cuts/h={point['false_cuts_per_hour']:.6g} "
                    f"jointly_useful_macro={env['jointly_useful_macro']} "
                    f"favorable_CI={env['jointly_favorable_ci_contamination_or_miss']}"
                )
                cont = boot["contamination"]["pooled_source_mean"]
                miss = boot["miss_rate"]["pooled_source_mean"]
                lines.append(
                    f"  bootstrap meeting-mean H-F0 contamination [{cont['lower']:.6g}, {cont['upper']:.6g}] "
                    f"miss [{miss['lower']:.6g}, {miss['upper']:.6g}]"
                )
                lines.append("  per-meeting H-F0 contamination/miss/p90:")
                for source_id in REQUIRED_DEV:
                    row = env["meetings"][source_id]["delta_h_minus_f0"]
                    lines.append(
                        f"  - {source_id}: d_cont={row['contamination']:.6g} "
                        f"d_miss={row['miss_rate']:.6g} d_p90={row['p90_delay_ms']}"
                    )
            lines.append("")
    lines.append("Director must judge whether any gain is primarily one meeting or topology from the rows above.")
    lines.append("Do not treat this file as Gate 1 ACCEPT, OPEN-T2, or confirmation.")
    lines.append("")
    return "\n".join(lines)


def _calibration_document(
    binding: Mapping[str, Any],
    f0_fit: Mapping[str, Any],
    cand_fit: Mapping[str, Any],
    packed: Mapping[str, Any],
) -> dict[str, Any]:
    f0_ap = calibrate_mod.average_precision(
        [calibrate_mod.sigmoid(z) for z in packed["f0_raw"]], packed["target"]
    )
    cand_ap = calibrate_mod.average_precision(
        [calibrate_mod.sigmoid(z) for z in packed["cand_raw"]], packed["target"]
    )
    return {
        "artifact_role": CALIBRATION_ARTIFACT_ROLE,
        "binding": dict(binding),
        "sources": list(packed["sources"]),
        "frames": int(packed["frames"]),
        "coverage": dict(packed["coverage"]),
        "f0": {**dict(f0_fit), "ap": f0_ap},
        "candidate": {**dict(cand_fit), "ap": cand_ap},
    }


def _event_tuple_at_threshold(
    grid: Sequence[float],
    events: Sequence[Sequence[Any]],
    threshold: float,
    ascending: Sequence[float] | None = None,
) -> tuple[tuple[Any, ...], ...]:
    if not grid:
        raise PostprocessError("sweep produced no score thresholds")
    level = float(threshold)
    if level > float(grid[0]):
        return ()
    if level <= float(grid[-1]):
        return tuple(tuple(event) for event in events[-1])
    import bisect

    ordered = ascending
    if ordered is None:
        ordered = tuple(reversed(tuple(float(value) for value in grid)))
    position = bisect.bisect_left(ordered, level)
    return tuple(tuple(event) for event in events[len(grid) - 1 - position])


def _build_contamination_index(intervals: Sequence[Any], anchors: Sequence[str]) -> dict[str, Any]:

    ordered = tuple(
        sorted(
            intervals,
            key=lambda interval: (int(interval.start_sample), int(interval.end_sample)),
        )
    )
    starts = tuple(int(interval.start_sample) for interval in ordered)
    ends = tuple(int(interval.end_sample) for interval in ordered)
    indexes: dict[str, Any] = {}
    for anchor in sorted(set(map(str, anchors))):
        eligible = tuple(
            not interval.masked
            and str(anchor) not in interval.active_speakers
            and bool(interval.active_speakers)
            for interval in ordered
        )
        prefix = [0]
        for interval, include in zip(ordered, eligible):
            duration = max(0, int(interval.end_sample) - int(interval.start_sample))
            prefix.append(prefix[-1] + (duration if include else 0))
        indexes[str(anchor)] = {
            "starts": starts,
            "ends": ends,
            "eligible": eligible,
            "prefix": tuple(prefix),
        }
    return indexes


def _indexed_contamination_samples(
    context: Mapping[str, Any],
    anchor: str,
    start_sample: int,
    end_sample: int,
) -> int:
    import bisect
    if end_sample <= start_sample:
        return 0
    index = context["contamination_index"][str(anchor)]
    starts = index["starts"]
    ends = index["ends"]
    first = bisect.bisect_right(ends, int(start_sample))
    last = bisect.bisect_left(starts, int(end_sample))
    if first >= last:
        return 0
    total = int(index["prefix"][last] - index["prefix"][first])
    eligible = index["eligible"]
    if eligible[first]:
        total -= max(0, int(start_sample) - int(starts[first]))
    final = last - 1
    if eligible[final]:
        total -= max(0, int(ends[final]) - int(end_sample))
    return int(total)


def _primitive_context(dev: Any) -> dict[str, Any]:
    intervals = intervals_from_manifest(dev.manifest)
    references = tuple(sorted(dev.reference.events, key=lambda event: int(event.boundary_source_sample)))
    active_samples = sum(
        int(interval.end_sample) - int(interval.start_sample)
        for interval in intervals
        if interval.active_speakers
    )
    return {
        "intervals": intervals,
        "references": references,
        "contamination_index": _build_contamination_index(
            intervals,
            [str(event.anchor_id) for event in references],
        ),
        "active_speech_seconds": float(active_samples) / 16000.0,
        "scored_end_sample": int(intervals[-1].end_sample),
        "tolerance_samples": int(ceiling_config()["product_event_alignment_tolerance_ms"] * 16),
    }


def _primitive_core(context: Mapping[str, Any], events: Sequence[Sequence[Any]]) -> dict[str, Any]:
    predicted = sorted(events, key=lambda event: int(event[3]))
    references = context["references"]
    matches = monotonic_boundary_matches(
        [int(event[3]) for event in predicted],
        [int(event.boundary_source_sample) for event in references],
        int(context["tolerance_samples"]),
    )
    matched_predicted = {left for left, _ in matches}
    matched_references = {right for _, right in matches}
    predicted_by_reference = {right: predicted[left] for left, right in matches}
    contamination_values: list[float] = []
    scored_end_sample = int(context["scored_end_sample"])
    for index, reference in enumerate(references):
        next_boundary = (
            int(references[index + 1].boundary_source_sample)
            if index + 1 < len(references)
            else scored_end_sample
        )
        predicted_event = predicted_by_reference.get(index)
        stop = (
            min(int(predicted_event[5]), next_boundary)
            if predicted_event is not None
            else next_boundary
        )
        start = int(reference.boundary_source_sample)
        contamination_values.append(
            _indexed_contamination_samples(
                context,
                str(reference.anchor_id),
                start,
                max(start, stop),
            )
            / 16000.0
        )
    return {
        "false_cut_count": int(len(predicted) - len(matched_predicted)),
        "active_speech_seconds": float(context["active_speech_seconds"]),
        "reference_replacement_count": int(len(references)),
        "missed_replacement_count": int(len(references) - len(matched_references)),
        "exclusive_other_contamination_seconds": float(sum(contamination_values)),
    }


def _primitive_row(
    context: Mapping[str, Any],
    cache: dict[tuple[int, tuple[tuple[Any, ...], ...]], dict[str, Any]],
    horizon_ms: int,
    events: Sequence[Sequence[Any]],
    threshold: float,
) -> dict[str, Any]:
    event_key = tuple(tuple(event) for event in events)
    cache_key = (int(horizon_ms), event_key)
    core = cache.get(cache_key)
    if core is None:
        core = _primitive_core(context, event_key)
        cache[cache_key] = core
    return {"threshold": float(threshold), **core}


def _sweep_member_primitives(payload: Mapping[str, Any]) -> dict[str, Any]:
    arm_runtime.spawn_worker_init()
    member = str(payload["member"])
    dev = payload["dev"]
    scores = {str(kind): list(values) for kind, values in dict(payload["scores"]).items()}
    requested_grids = {
        str(kind): [float(value) for value in values]
        for kind, values in dict(payload["requested_grids"]).items()
    }
    context = _primitive_context(dev)
    cache: dict[tuple[int, tuple[tuple[Any, ...], ...]], dict[str, Any]] = {}
    primitives: dict[str, Any] = {"raw": {}, "calibrated": {}, "f0": {}}
    for horizon_ms in frontier_mod.HORIZONS_MS:
        f0_grid, f0_events = sweep_mod.sweep_threshold_events(dev, scores["f0"], int(horizon_ms))
        f0_ascending = tuple(reversed(tuple(float(value) for value in f0_grid)))
        primitives["f0"][int(horizon_ms)] = [
            _primitive_row(
                context,
                cache,
                int(horizon_ms),
                _event_tuple_at_threshold(
                    f0_grid,
                    f0_events,
                    float(threshold),
                    f0_ascending,
                ),
                float(threshold),
            )
            for threshold in requested_grids["f0"]
        ]
        for kind in ("raw", "calibrated"):
            grid, event_rows = sweep_mod.sweep_threshold_events(dev, scores[kind], int(horizon_ms))
            ascending = tuple(reversed(tuple(float(value) for value in grid)))
            primitives[kind][int(horizon_ms)] = [
                _primitive_row(
                    context,
                    cache,
                    int(horizon_ms),
                    _event_tuple_at_threshold(grid, event_rows, float(threshold), ascending),
                    float(threshold),
                )
                for threshold in requested_grids[kind]
            ]
    return {
        "member": member,
        "primitives": primitives,
        "scored_primitives": len(cache),
    }


def _run_sweep_wave(
    members: Mapping[str, Any],
    requested_grids: Mapping[str, Sequence[float]],
    workers: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import time

    grids = {
        str(kind): tuple(float(value) for value in values)
        for kind, values in requested_grids.items()
    }
    if set(grids) != {"raw", "calibrated", "f0"}:
        raise PostprocessError("CPU sweep grids must contain raw, calibrated, and f0")
    chunk = int(cross_mod.EXACT_THRESHOLD_CHUNK)
    total_tasks = len(members) * len(frontier_mod.HORIZONS_MS) * sum(
        (len(grids[kind]) + chunk - 1) // chunk for kind in sorted(grids)
    )
    resolved = arm_runtime.resolve_workers(workers)
    payloads = [
        {
            "member": str(member),
            "dev": value["dev"],
            "scores": dict(value["scores"]),
            "requested_grids": grids,
        }
        for member, value in sorted(members.items())
    ]
    start = time.perf_counter()
    results = arm_runtime.ordered_process_map(_sweep_member_primitives, payloads, resolved)
    primitives = {str(result["member"]): result["primitives"] for result in results}
    scored_total = sum(int(result["scored_primitives"]) for result in results)
    counts: dict[str, Any] = {}
    full_total = 0
    for member in sorted(members):
        for kind in sorted(grids):
            for horizon_ms in frontier_mod.HORIZONS_MS:
                key = "/".join([str(member), kind, str(int(horizon_ms))])
                rows = primitives.get(str(member), {}).get(kind, {}).get(int(horizon_ms), [])
                got = [float(row["threshold"]) for row in rows]
                want = list(grids[kind])
                if got != want:
                    raise PostprocessError(f"frontier thresholds incomplete for {key}")
                counts[key] = {"expected": len(want), "observed": len(got)}
                full_total += len(want)
    receipt = {
        **arm_runtime.worker_receipt(workers, total_tasks),
        "total_tasks": int(total_tasks),
        "pool_count": int(resolved > 1 and len(payloads) > 1),
        "exact": True,
        "score_tasks": int(scored_total),
        "reused_primitives": int(full_total - scored_total),
        "primitive_counts": counts,
        "elapsed_seconds": time.perf_counter() - start,
        "strategy": "sweep_threshold_events_exact_primitives",
    }
    return primitives, receipt


def run_postprocess(export_dir: Path, out_dir: Path, workers: int = 8) -> dict[str, Any]:
    export = load_validated_export(Path(export_dir))
    f0_fit, cand_fit, packed = fit_calib_from_export(export["calib"])
    members = {
        source_id: {
            **prepare_dev_member(source_id, export["dev"][source_id], f0_fit, cand_fit),
            "sidecar": export["dev"][source_id]["sidecar"],
        }
        for source_id in REQUIRED_DEV
    }
    grids = {
        "raw": union_probability_grid([members[source_id]["cand_raw_prob"] for source_id in REQUIRED_DEV]),
        "calibrated": union_probability_grid(
            [members[source_id]["cand_cal_prob"] for source_id in REQUIRED_DEV]
        ),
        "f0": [frontier_mod.RAW_REFERENCE_THRESHOLD],
    }
    member_scores = {
        source_id: {
            "dev": members[source_id]["dev"],
            "scores": {
                "raw": members[source_id]["cand_raw_prob"],
                "calibrated": members[source_id]["cand_cal_prob"],
                "f0": members[source_id]["f0_prob"],
            },
        }
        for source_id in REQUIRED_DEV
    }
    primitives, wave_receipt = _run_sweep_wave(member_scores, grids, int(workers))
    frontier = build_canonical_frontier(
        members, primitives, grids, export["binding"], {**wave_receipt, "thresholds": {k: len(v) for k, v in grids.items()}}
    )
    selected = selected_session_metrics(members, frontier)
    calibration = _calibration_document(export["binding"], f0_fit, cand_fit, packed)
    diagnostics = build_gate1_diagnostics(members, frontier, selected, calibration, export)
    report = render_decision_evidence(diagnostics)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    calib_path = arm_runtime.atomic_write_json(dest / "calibration_metrics.json", _jsonable(calibration))
    frontier_path = arm_runtime.atomic_write_json(dest / "dev_frontier.json", _jsonable(frontier))
    diag_path = arm_runtime.atomic_write_json(dest / "gate1_diagnostics.json", _jsonable(diagnostics))
    report_path = dest / "gate1_decision_evidence.md"
    arm_runtime.atomic_write_bytes(report_path, report.encode("utf-8"))
    return {
        "export_dir": str(Path(export_dir)),
        "out_dir": str(dest),
        "arm": H_ARM,
        "gate_receipt_emitted": False,
        "calibration_metrics": str(calib_path),
        "dev_frontier": str(frontier_path),
        "gate1_diagnostics": str(diag_path),
        "decision_evidence": str(report_path),
        "workers": arm_runtime.resolve_workers(int(workers)),
        "dev_sources": list(REQUIRED_DEV),
        "calib_sources": list(REQUIRED_CALIB),
        "wave": dict(wave_receipt),
    }
