from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.speaker_turn_boundary.config import (
    EXPERIMENT_DATA_DIR,
    EXPERIMENT_RESULTS_DIR,
    VAD_COALESCE_WINDOW_SAMPLES,
)
from experiments.speaker_turn_boundary.metadata import collect_runtime_metadata
from experiments.speaker_turn_boundary.metrics import RECALL_DEADLINES_MS
from experiments.speaker_turn_boundary.phase3_eval import (
    compact_row,
    evaluate_b0,
)
from experiments.speaker_turn_boundary.phase3_funnel import (
    build_frontier_summary,
    build_frozen_artifact,
    integer_operating_curve,
)
from experiments.speaker_turn_boundary.phase3_metrics import (
    ATTRIBUTION_HORIZON_MS,
    LOCALIZATION_TOLERANCE_MS,
)
from experiments.speaker_turn_boundary.phase3_stages import (
    ERES_ONNX_SHA256,
    LSCheckpointData,
    StageContext,
    all_eres_profiles,
    build_inputs,
    default_engine_factory,
    eres_frontend_contract,
    evaluate_eres_profile,
    evaluate_ls_profile,
    expected_model_hashes,
    file_sha256,
    load_or_capture_ls,
    ls_frontend_contract,
    ls_profiles,
    prepare_eres_embeddings,
    reconstruct_profile,
)
from experiments.speaker_turn_boundary.provenance import LS_EEND_VARIANTS
from experiments.speaker_turn_boundary.run_eres_sweep import ERES_CHECKPOINTS
from experiments.speaker_turn_boundary.schemas import canonical_json, sha256_hex

DEV_MANIFEST = "mixed_dev_pool"
HELDOUT_MANIFESTS = (
    "ls_held_out_clean",
    "ls_held_out_other",
    "ami_held_out_pilot",
    "alimeeting_eval_pilot",
)
DEV_ROWS_SCHEMA = "experiments.speaker_turn_boundary.phase3.dev_rows.v2"
DEV_SUMMARY_SCHEMA = "experiments.speaker_turn_boundary.phase3.dev_summary.v2"
HELDOUT_SUMMARY_SCHEMA = "experiments.speaker_turn_boundary.phase3.heldout_summary.v2"
DECISION_SCHEMA = "experiments.speaker_turn_boundary.phase3.decision.v2"


class Phase3RunError(RuntimeError):
    pass


def default_scratch() -> Path:
    return Path(os.environ.get("TEMP", str(Path.home() / "tmp"))) / "opencode" / "stb_phase3_v2"


def default_legacy_scratch() -> Path:
    return Path(os.environ.get("TEMP", str(Path.home() / "tmp"))) / "opencode" / "stb_phase3_run"


def context(args: argparse.Namespace) -> StageContext:
    return StageContext(
        data_dir=args.data_dir,
        corpus_root=args.corpus_root,
        hf_root=args.hf_root,
        eres_onnx_root=args.eres_onnx_root,
        scratch=args.scratch,
        legacy_scratch=args.legacy_scratch,
        engine_factory=default_engine_factory(),
    )


def results_dir(args: argparse.Namespace) -> Path:
    return args.results / "phase3"


def manifest_path(args: argparse.Namespace, manifest_id: str) -> Path:
    return args.data_dir / "manifests" / f"{manifest_id}.json"


def metric_contract() -> dict[str, Any]:
    return {
        "boundary_localization_tolerance_ms": LOCALIZATION_TOLERANCE_MS,
        "causal_recall_deadlines_ms": list(RECALL_DEADLINES_MS),
        "false_cut_attribution_horizon_ms": ATTRIBUTION_HORIZON_MS,
        "matching": "deterministic_ordered_one_to_one_max_cardinality_then_min_delay_and_localization",
        "product_matching": "lock_b0_matches_then_match_added_noncoalesced_detector_cuts",
        "coalesce_window_samples": VAD_COALESCE_WINDOW_SAMPLES,
        "rate_budget_role": "reference_slice_only_not_elimination",
        "five_minute_denominator": "source_session_time",
    }


def source_contract_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "PHASE3_DESIGN.md",
        "phase3_data.py",
        "phase3_eres.py",
        "phase3_eval.py",
        "phase3_funnel.py",
        "phase3_ls.py",
        "phase3_metrics.py",
        "phase3_stages.py",
        "run_phase3.py",
        "frontend.py",
        "reducer.py",
        "coalescing.py",
        "vad_baseline.py",
    )
    return {name: file_sha256(root / name) for name in names}


def model_file_evidence(args: argparse.Namespace) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for checkpoint, info in sorted(LS_EEND_VARIANTS.items()):
        onnx = args.hf_root / str(info["dir"]) / str(info["onnx"])
        sidecar = args.hf_root / str(info["dir"]) / str(info["sidecar"])
        actual_onnx = file_sha256(onnx)
        actual_sidecar = file_sha256(sidecar)
        passed = actual_onnx == str(info["onnx_sha256"]) and actual_sidecar == str(
            info["sidecar_sha256"]
        )
        if not passed:
            raise Phase3RunError(f"LS model preflight failed: {checkpoint}")
        evidence[f"ls_eend:{checkpoint}"] = {
            "onnx_path": str(onnx),
            "onnx_sha256": actual_onnx,
            "sidecar_path": str(sidecar),
            "sidecar_sha256": actual_sidecar,
            "passed": True,
        }
    for checkpoint, info in sorted(ERES_CHECKPOINTS.items()):
        onnx = args.eres_onnx_root / str(info["onnx"])
        actual = file_sha256(onnx)
        if actual != ERES_ONNX_SHA256[checkpoint]:
            raise Phase3RunError(f"ERes model preflight failed: {checkpoint}")
        evidence[f"eres2netv2:{checkpoint}"] = {
            "onnx_path": str(onnx),
            "onnx_sha256": actual,
            "passed": True,
        }
    return evidence


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")
    return sha256_hex(payload)


def _write_self_hashed_json(
    path: Path,
    payload: dict[str, Any],
    *,
    hash_field: str = "artifact_sha256",
) -> str:
    data = dict(payload)
    data[hash_field] = sha256_hex(data)
    _write_json(path, data)
    return str(data[hash_field])


def _verify_self_hash(payload: dict[str, Any], field: str) -> None:
    expected = payload.get(field)
    actual = sha256_hex({key: value for key, value in payload.items() if key != field})
    if expected != actual:
        raise Phase3RunError(f"artifact self hash mismatch: {actual} != {expected}")


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        profile_id = str(row["profile_id"])
        if profile_id in rows:
            raise Phase3RunError(f"duplicate row {profile_id} at line {line_number}")
        rows[profile_id] = row
    return rows


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _canonicalize_rows(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for _, row in sorted(rows.items())
    )
    path.write_text(content, encoding="utf-8", newline="\n")


def _safe_profile_id(profile_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in profile_id)


def cmd_preflight(args: argparse.Namespace) -> None:
    ctx = context(args)
    manifests = [DEV_MANIFEST]
    manifest_rows: dict[str, Any] = {}
    for manifest_id in manifests:
        manifest, inputs = build_inputs(ctx, manifest_path(args, manifest_id))
        manifest_rows[manifest_id] = {
            "manifest_sha256": manifest.hash,
            "case_count": len(inputs),
            "source_seconds": sum(case.length_samples for case in inputs) / 16000.0,
            "active_speech_seconds": sum(case.active_speech_samples for case in inputs) / 16000.0,
            "gt_change_count": sum(len(case.gt_changes) for case in inputs),
        }
    payload = {
        "schema_version": "experiments.speaker_turn_boundary.phase3.preflight.v2",
        "metric_contract": metric_contract(),
        "source_contract_hashes": source_contract_hashes(),
        "model_hashes": expected_model_hashes(),
        "model_files": model_file_evidence(args),
        "frontend_contracts": {
            "ls_eend": ls_frontend_contract(),
            "eres2netv2": eres_frontend_contract(),
        },
        "manifests": manifest_rows,
        "profile_counts": {
            "ls_eend": len(LS_EEND_VARIANTS) * len(ls_profiles()),
            "eres2netv2_adjacent": len(ERES_CHECKPOINTS) * len(all_eres_profiles()[0]),
            "eres2netv2_stable_anchor": len(ERES_CHECKPOINTS) * len(all_eres_profiles()[1]),
        },
        "runtime_metadata": collect_runtime_metadata(),
    }
    out = results_dir(args) / "preflight_v2.json"
    digest = _write_self_hashed_json(out, payload)
    print(f"wrote {out} sha256={digest}")


def cmd_dev(args: argparse.Namespace) -> None:
    ctx = context(args)
    result_root = results_dir(args)
    result_root.mkdir(parents=True, exist_ok=True)
    manifest, inputs = build_inputs(ctx, manifest_path(args, DEV_MANIFEST), case_ids=args.case)
    if args.case:
        raise Phase3RunError("development freeze cannot be created from a case subset")
    adjacent, stable = all_eres_profiles()
    run_contract: dict[str, Any] = {
        "schema_version": "experiments.speaker_turn_boundary.phase3.dev_run_contract.v2",
        "manifest_sha256": manifest.hash,
        "metric_contract": metric_contract(),
        "source_contract_hashes": source_contract_hashes(),
        "model_hashes": expected_model_hashes(),
        "frontend_contracts": {
            "ls_eend": ls_frontend_contract(),
            "eres2netv2": eres_frontend_contract(),
        },
        "profile_counts": {
            "b0": 1,
            "ls_eend": len(LS_EEND_VARIANTS) * len(ls_profiles()),
            "eres2netv2_adjacent": len(ERES_CHECKPOINTS) * len(adjacent),
            "eres2netv2_stable_anchor": len(ERES_CHECKPOINTS) * len(stable),
        },
    }
    run_contract["contract_sha256"] = sha256_hex(run_contract)
    contract_path = result_root / "dev_run_contract_v2.json"
    if contract_path.is_file():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != run_contract:
            raise Phase3RunError(
                "existing development rows belong to a different contract; use a new results directory"
            )
    else:
        _write_json(contract_path, run_contract)
    rows_path = result_root / "dev_rows_v2.jsonl"
    rows = _load_rows(rows_path)
    if "b0_vad_only" not in rows:
        b0_evaluation = evaluate_b0(inputs)
        row = compact_row(b0_evaluation)
        row["row_schema"] = DEV_ROWS_SCHEMA
        _append_row(rows_path, row)
        rows[row["profile_id"]] = row
        _write_panel_evidence(result_root / "dev_evidence", b0_evaluation)
    ls_data: dict[str, LSCheckpointData] = {}
    profiles_ls = ls_profiles()
    for checkpoint in sorted(LS_EEND_VARIANTS):
        checkpoint_data = load_or_capture_ls(ctx, manifest, inputs, checkpoint)
        ls_data[checkpoint] = checkpoint_data
        for profile in profiles_ls:
            profile_id = f"{checkpoint}:{profile.profile_id}"
            if profile_id in rows:
                continue
            evaluation = evaluate_ls_profile(inputs, checkpoint_data, profile)
            row = compact_row(evaluation)
            row["row_schema"] = DEV_ROWS_SCHEMA
            _append_row(rows_path, row)
            rows[profile_id] = row
    eres_data: dict[str, Any] = {}
    for checkpoint in sorted(ERES_CHECKPOINTS):
        checkpoint_data = prepare_eres_embeddings(
            ctx,
            manifest,
            inputs,
            checkpoint,
            adjacent_profiles=adjacent,
            stable_profiles=stable,
        )
        eres_data[checkpoint] = checkpoint_data
        for profile in [*adjacent, *stable]:
            profile_id = f"{checkpoint}:{profile.profile_id}"
            if profile_id in rows:
                continue
            evaluation = evaluate_eres_profile(inputs, checkpoint_data, profile)
            row = compact_row(evaluation)
            row["row_schema"] = DEV_ROWS_SCHEMA
            _append_row(rows_path, row)
            rows[profile_id] = row
    expected_total = (
        1
        + int(run_contract["profile_counts"]["ls_eend"])
        + int(run_contract["profile_counts"]["eres2netv2_adjacent"])
        + int(run_contract["profile_counts"]["eres2netv2_stable_anchor"])
    )
    if len(rows) != expected_total:
        raise Phase3RunError(f"expected {expected_total} development rows, got {len(rows)}")
    _canonicalize_rows(rows_path, rows)
    rows = _load_rows(rows_path)
    rows_sha256 = file_sha256(rows_path)
    ls_rows = [row for row in rows.values() if row["family"] == "ls_eend"]
    eres_rows = [row for row in rows.values() if row["family"] == "eres2netv2"]
    frontier_summaries = {
        "ls_eend": build_frontier_summary(ls_rows, family="ls_eend"),
        "eres2netv2": build_frontier_summary(eres_rows, family="eres2netv2"),
    }
    panels = {family: summary["frozen_panel"] for family, summary in frontier_summaries.items()}
    frozen = build_frozen_artifact(
        panels=panels,
        manifest_sha256=manifest.hash,
        model_hashes=expected_model_hashes(),
        frontend_contracts={
            "ls_eend": ls_frontend_contract(),
            "eres2netv2": eres_frontend_contract(),
        },
        rows_sha256=rows_sha256,
        metric_contract=metric_contract(),
    )
    frozen_path = result_root / "frozen_panel_v2.json"
    _write_json(frozen_path, frozen)
    for family, panel in panels.items():
        for selected in panel:
            profile = reconstruct_profile(selected)
            checkpoint = str(selected["checkpoint"])
            if family == "ls_eend":
                evaluation = evaluate_ls_profile(inputs, ls_data[checkpoint], profile)
            else:
                evaluation = evaluate_eres_profile(inputs, eres_data[checkpoint], profile)
            _write_panel_evidence(result_root / "dev_evidence", evaluation)
    summary: dict[str, Any] = {
        "schema_version": DEV_SUMMARY_SCHEMA,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.hash,
        "case_count": len(inputs),
        "source_seconds": sum(case.length_samples for case in inputs) / 16000.0,
        "active_speech_seconds": sum(case.active_speech_samples for case in inputs) / 16000.0,
        "gt_change_count": sum(len(case.gt_changes) for case in inputs),
        "row_count": len(rows),
        "rows_path": str(rows_path),
        "rows_sha256": rows_sha256,
        "b0": rows["b0_vad_only"],
        "frontiers": frontier_summaries,
        "frozen_panel_path": str(frozen_path),
        "frozen_sha256": frozen["frozen_sha256"],
        "validity_limits": [
            "small development exposure makes false-cut rates highly quantized",
            "examples are clustered within source sessions and synthetic recipes",
            "ERes policies are VAD-utterance scoped",
            "no authorized D4 product-domain audio is available",
        ],
    }
    out = result_root / "dev_summary_v2.json"
    digest = _write_self_hashed_json(out, summary)
    print(f"wrote {out} sha256={digest} rows={len(rows)}")


def _write_panel_evidence(root: Path, evaluation) -> str:
    path = root / f"{_safe_profile_id(evaluation.profile_id)}.json"
    return _write_self_hashed_json(path, evaluation.evidence_dict())


def cmd_refreeze(args: argparse.Namespace) -> None:
    ctx = context(args)
    result_root = results_dir(args)
    rows_path = result_root / "dev_rows_v2.jsonl"
    rows = _load_rows(rows_path)
    if len(rows) != 1369:
        raise Phase3RunError(f"refreeze requires 1369 complete development rows, got {len(rows)}")
    prior_summary_path = result_root / "dev_summary_v2.json"
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    _verify_self_hash(prior_summary, "artifact_sha256")
    rows_sha256 = file_sha256(rows_path)
    if prior_summary["rows_sha256"] != rows_sha256:
        raise Phase3RunError("development row hash differs from the completed summary")
    run_contract = json.loads(
        (result_root / "dev_run_contract_v2.json").read_text(encoding="utf-8")
    )
    manifest, inputs = build_inputs(ctx, manifest_path(args, DEV_MANIFEST))
    if manifest.hash != run_contract["manifest_sha256"]:
        raise Phase3RunError("development manifest differs from the row contract")
    ls_rows = [row for row in rows.values() if row["family"] == "ls_eend"]
    eres_rows = [row for row in rows.values() if row["family"] == "eres2netv2"]
    frontier_summaries = {
        "ls_eend": build_frontier_summary(ls_rows, family="ls_eend"),
        "eres2netv2": build_frontier_summary(eres_rows, family="eres2netv2"),
    }
    panels = {family: summary["frozen_panel"] for family, summary in frontier_summaries.items()}
    frozen = build_frozen_artifact(
        panels=panels,
        manifest_sha256=manifest.hash,
        model_hashes=expected_model_hashes(),
        frontend_contracts={
            "ls_eend": ls_frontend_contract(),
            "eres2netv2": eres_frontend_contract(),
        },
        rows_sha256=rows_sha256,
        metric_contract=metric_contract(),
    )
    frozen.pop("frozen_sha256")
    frozen["selection_provenance"] = {
        "dev_row_contract_sha256": run_contract["contract_sha256"],
        "prior_completed_summary_sha256": prior_summary["artifact_sha256"],
        "selection_source_hashes": {
            key: value
            for key, value in source_contract_hashes().items()
            if key in {"PHASE3_DESIGN.md", "phase3_funnel.py", "run_phase3.py"}
        },
        "reason": (
            "The first sensitivity panel included a maximum-recovery endpoint whose "
            "AliMeeting embedding volume was not decision-useful. Development rows and "
            "their metric contract are unchanged; only the pre-held-out representative "
            "selection rule is replaced."
        ),
    }
    frozen["frozen_sha256"] = sha256_hex(frozen)
    frozen_path = result_root / "frozen_panel_v2.json"
    _write_json(frozen_path, frozen)
    ls_data: dict[str, LSCheckpointData] = {}
    eres_data: dict[str, Any] = {}
    selected_rows = [row for family in panels.values() for row in family]
    for selected in selected_rows:
        checkpoint = str(selected["checkpoint"])
        profile = reconstruct_profile(selected)
        if selected["family"] == "ls_eend":
            if checkpoint not in ls_data:
                ls_data[checkpoint] = load_or_capture_ls(ctx, manifest, inputs, checkpoint)
            evaluation = evaluate_ls_profile(inputs, ls_data[checkpoint], profile)
        else:
            if checkpoint not in eres_data:
                adjacent = [
                    reconstruct_profile(row)
                    for row in selected_rows
                    if row["family"] == "eres2netv2"
                    and row["checkpoint"] == checkpoint
                    and row["profile_kind"] == "adjacent"
                ]
                stable = [
                    reconstruct_profile(row)
                    for row in selected_rows
                    if row["family"] == "eres2netv2"
                    and row["checkpoint"] == checkpoint
                    and row["profile_kind"] == "stable_anchor"
                ]
                eres_data[checkpoint] = prepare_eres_embeddings(
                    ctx,
                    manifest,
                    inputs,
                    checkpoint,
                    adjacent_profiles=adjacent,
                    stable_profiles=stable,
                )
            evaluation = evaluate_eres_profile(inputs, eres_data[checkpoint], profile)
        _write_panel_evidence(
            result_root / "dev_evidence" / frozen["frozen_sha256"][:12],
            evaluation,
        )
    summary = {key: value for key, value in prior_summary.items() if key != "artifact_sha256"}
    summary["frontiers"] = frontier_summaries
    summary["frozen_panel_path"] = str(frozen_path)
    summary["frozen_sha256"] = frozen["frozen_sha256"]
    summary["refreeze"] = frozen["selection_provenance"]
    digest = _write_self_hashed_json(prior_summary_path, summary)
    print(
        f"wrote {frozen_path} frozen_sha256={frozen['frozen_sha256']} " f"summary_sha256={digest}"
    )


def _load_frozen(result_root: Path) -> dict[str, Any]:
    path = result_root / "frozen_panel_v2.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    _verify_self_hash(frozen, "frozen_sha256")
    if frozen["metric_contract"] != metric_contract():
        raise Phase3RunError("frozen metric contract differs from current code")
    if frozen["model_hashes"] != expected_model_hashes():
        raise Phase3RunError("frozen model hashes differ from current registry")
    if frozen["frontend_contracts"] != {
        "ls_eend": ls_frontend_contract(),
        "eres2netv2": eres_frontend_contract(),
    }:
        raise Phase3RunError("frozen frontend contract differs from current code")
    return frozen


def cmd_heldout(args: argparse.Namespace) -> None:
    ctx = context(args)
    result_root = results_dir(args)
    frozen = _load_frozen(result_root)
    manifests = args.manifest or list(HELDOUT_MANIFESTS)
    if any(manifest == DEV_MANIFEST for manifest in manifests):
        raise Phase3RunError("development manifest cannot be passed to held-out command")
    out = result_root / "heldout_summary_v2.json"
    if out.is_file():
        previous = json.loads(out.read_text(encoding="utf-8"))
        _verify_self_hash(previous, "artifact_sha256")
        if previous["frozen_sha256"] != frozen["frozen_sha256"]:
            raise Phase3RunError("existing held-out summary belongs to a different frozen panel")
        summary = {key: value for key, value in previous.items() if key != "artifact_sha256"}
    else:
        summary = {
            "schema_version": HELDOUT_SUMMARY_SCHEMA,
            "frozen_sha256": frozen["frozen_sha256"],
            "manifests": {},
        }
    panel_rows = [row for family_rows in frozen["panels"].values() for row in family_rows]
    for manifest_id in manifests:
        manifest, inputs = build_inputs(ctx, manifest_path(args, manifest_id), case_ids=args.case)
        if args.case:
            raise Phase3RunError("held-out summary cannot be created from a case subset")
        entry: dict[str, Any] = {
            "manifest_sha256": manifest.hash,
            "case_count": len(inputs),
            "source_seconds": sum(case.length_samples for case in inputs) / 16000.0,
            "active_speech_seconds": sum(case.active_speech_samples for case in inputs) / 16000.0,
            "gt_change_count": sum(len(case.gt_changes) for case in inputs),
            "b0": compact_row(evaluate_b0(inputs)),
            "profiles": {},
        }
        ls_by_checkpoint: dict[str, LSCheckpointData] = {}
        for checkpoint in sorted(
            {str(row["checkpoint"]) for row in panel_rows if row["family"] == "ls_eend"}
        ):
            ls_by_checkpoint[checkpoint] = load_or_capture_ls(ctx, manifest, inputs, checkpoint)
        adjacent_profiles = [
            reconstruct_profile(row)
            for row in panel_rows
            if row["family"] == "eres2netv2" and row["profile_kind"] == "adjacent"
        ]
        stable_profiles = [
            reconstruct_profile(row)
            for row in panel_rows
            if row["family"] == "eres2netv2" and row["profile_kind"] == "stable_anchor"
        ]
        eres_by_checkpoint: dict[str, Any] = {}
        for checkpoint in sorted(
            {str(row["checkpoint"]) for row in panel_rows if row["family"] == "eres2netv2"}
        ):
            checkpoint_adjacent = [
                profile
                for row, profile in zip(
                    [
                        item
                        for item in panel_rows
                        if item["family"] == "eres2netv2" and item["profile_kind"] == "adjacent"
                    ],
                    adjacent_profiles,
                )
                if row["checkpoint"] == checkpoint
            ]
            checkpoint_stable = [
                profile
                for row, profile in zip(
                    [
                        item
                        for item in panel_rows
                        if item["family"] == "eres2netv2"
                        and item["profile_kind"] == "stable_anchor"
                    ],
                    stable_profiles,
                )
                if row["checkpoint"] == checkpoint
            ]
            eres_by_checkpoint[checkpoint] = prepare_eres_embeddings(
                ctx,
                manifest,
                inputs,
                checkpoint,
                adjacent_profiles=checkpoint_adjacent,
                stable_profiles=checkpoint_stable,
            )
        evidence_root = (
            result_root / "heldout_evidence" / frozen["frozen_sha256"][:12] / manifest_id
        )
        for selected in panel_rows:
            profile = reconstruct_profile(selected)
            checkpoint = str(selected["checkpoint"])
            if selected["family"] == "ls_eend":
                evaluation = evaluate_ls_profile(inputs, ls_by_checkpoint[checkpoint], profile)
            else:
                evaluation = evaluate_eres_profile(inputs, eres_by_checkpoint[checkpoint], profile)
            digest = _write_panel_evidence(evidence_root, evaluation)
            row = compact_row(evaluation)
            row["evidence_sha256"] = digest
            entry["profiles"][evaluation.profile_id] = row
        summary["manifests"][manifest_id] = entry
    digest = _write_self_hashed_json(out, summary)
    print(f"wrote {out} sha256={digest} " f"manifests_total={len(summary['manifests'])}")


def cmd_decision(args: argparse.Namespace) -> None:
    result_root = results_dir(args)
    frozen = _load_frozen(result_root)
    heldout_path = result_root / "heldout_summary_v2.json"
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    _verify_self_hash(heldout, "artifact_sha256")
    if heldout["frozen_sha256"] != frozen["frozen_sha256"]:
        raise Phase3RunError("held-out summary was not produced from the current frozen panel")
    missing_manifests = set(HELDOUT_MANIFESTS) - set(heldout["manifests"])
    if missing_manifests:
        raise Phase3RunError(f"held-out summary is incomplete; missing {sorted(missing_manifests)}")
    pooled: dict[str, dict[str, Any]] = {}
    for manifest_entry in heldout["manifests"].values():
        for profile_id, row in manifest_entry["profiles"].items():
            aggregate = pooled.setdefault(
                profile_id,
                {
                    "profile_id": profile_id,
                    "family": row["family"],
                    "checkpoint": row["checkpoint"],
                    "profile_kind": row["profile_kind"],
                    "params": row["params"],
                    "recovered_b0_misses_at_ms": {
                        str(deadline): 0 for deadline in RECALL_DEADLINES_MS
                    },
                    "extra_false_cuts": 0,
                    "added_logical_cuts": 0,
                },
            )
            for deadline in RECALL_DEADLINES_MS:
                aggregate["recovered_b0_misses_at_ms"][str(deadline)] += int(
                    row["recovered_b0_misses_at_ms"][str(deadline)]
                )
            aggregate["extra_false_cuts"] += int(row["extra_false_cuts"])
            aggregate["added_logical_cuts"] += int(row["added_logical_cuts"])
    families = {
        family: [row for row in pooled.values() if row["family"] == family]
        for family in ("ls_eend", "eres2netv2")
    }
    decision: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA,
        "frozen_sha256": frozen["frozen_sha256"],
        "heldout_summary_sha256": heldout["artifact_sha256"],
        "d4_authorized_audio_available": False,
        "status": "provisional_no_production_selection",
        "heldout_panel_integer_curves": {
            family: integer_operating_curve(rows) for family, rows in families.items()
        },
        "pooled_panel_rows": pooled,
        "selection_statement": (
            "The frozen sensitivity panel is reported at actual held-out extra-cut counts. "
            "No production family or threshold is selected because the panel is sparse, "
            "the false-cut price is not specified, and authorized D4 product-domain audio "
            "is unavailable."
        ),
        "architecture_drift": False,
    }
    out = result_root / "decision_v2.json"
    digest = _write_self_hashed_json(out, decision)
    print(f"wrote {out} sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal Phase 3 speaker-boundary experiment")
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT_DATA_DIR)
    parser.add_argument("--corpus-root", type=Path, default=None)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--eres-onnx-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=default_scratch())
    parser.add_argument("--legacy-scratch", type=Path, default=default_legacy_scratch())
    parser.add_argument("--results", type=Path, default=EXPERIMENT_RESULTS_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(func=cmd_preflight)
    dev = subparsers.add_parser("dev")
    dev.add_argument("--case", action="append", default=None)
    dev.set_defaults(func=cmd_dev)
    refreeze = subparsers.add_parser("refreeze")
    refreeze.set_defaults(func=cmd_refreeze)
    heldout = subparsers.add_parser("heldout")
    heldout.add_argument("--manifest", action="append", default=None)
    heldout.add_argument("--case", action="append", default=None)
    heldout.set_defaults(func=cmd_heldout)
    decision = subparsers.add_parser("decision")
    decision.set_defaults(func=cmd_decision)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
