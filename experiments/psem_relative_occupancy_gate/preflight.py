from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    REPOSITORY_ROOT,
    canonical_sha256,
    config,
    corpus_root,
    load_json,
    lseend_root,
    reference_root,
    research_root,
    safe_child,
    safe_output_path,
    sha256_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.provenance import (
    AUTHORITY_PIN,
    AUTHORITY_REF,
    FREEZE_FILE_SHA256,
    FREEZE_PAYLOAD_SHA256,
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    REFERENCE_TREE,
    ROLE_SOURCE_COUNTS,
    load_frozen_dataset,
    validate_corpus_waveforms,
)
from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    acquire_reference,
    validate_reference_checkout,
)


class PreflightError(RuntimeError):
    pass


SORTFORMER_PIN = {
    "family": "streaming_sortformer",
    "model_repository": "handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf",
    "model_revision": "7ef0c15dc8f9d717e9d24fac29a6e6551e9c6ddf",
    "model_filename": "diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf",
    "model_sha256": "a5dacdc650790266c7a362e54e6bf51952015487edaa606c4e11632bc32442a9",
    "source_repository": "https://github.com/handy-computer/transcribe.cpp.git",
    "source_commit": "d42c3bbdfa2f63c37e5891e27de47a612d62f221",
    "telemetry_patch_sha256": "d01eecdee26bff12af2f3ba649eebc5e7e1195871a537646c72301362c47ce44",
    "bench_relative_path": "external/r8/transcribe.cpp/build-r8-vulkan-short/bin/Release/transcribe-bench.exe",
    "bench_sha256": "e26f76e36568992a445251f197de44fa5528499f788120144947fa2aedef6c12",
    "backend": "vulkan",
    "threads": 8,
    "preset": "low_latency",
    "slot_count": 4,
    "native_frame_ms": 80,
    "chunk_audio_ms": 480,
    "algorithmic_lookahead_ms": 1040,
    "slot_validity_metadata": False,
}

LSEEND_PIN = {
    "family": "ls_eend",
    "variant": "L-AMI",
    "repository": "https://huggingface.co/GradientDescent2718/LS-EEND-ONNX",
    "revision": "cc40a1e1242c148fbbc15c132e43b8ac15056e53",
    "model_relative_path": "AMI/ls_eend_ami_step.onnx",
    "model_sha256": "5a2b813ffe41170e40d0fc08a6eb1699e579e377af30c7962d07885608a6aa77",
    "sidecar_relative_path": "AMI/ls_eend_ami_step.json",
    "sidecar_sha256": "47f29718254995ec017636d5ff31fef8b20bf47dca30d883edcb91e022dc3353",
    "backend": "CPUExecutionProvider",
    "intra_op_threads": 1,
    "inter_op_threads": 1,
    "slot_count": 4,
    "native_frame_ms": 100,
    "slot_validity_metadata": False,
}


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"id": name, "passed": bool(passed), "detail": detail}


def _file_check(name: str, path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = sha256_file(path) if path.is_file() and not path.is_symlink() else None
    return _check(
        name,
        actual == expected_sha256,
        {
            "path": str(path),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual,
        },
    )


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise PreflightError(detail)
    return result.stdout.strip()


def _git_state() -> dict[str, Any]:
    head = _git(REPOSITORY_ROOT, "rev-parse", "HEAD")
    status = _git(REPOSITORY_ROOT, "status", "--short").splitlines()
    return {"head": head, "dirty": bool(status), "dirty_paths": status}


def _normalized_repository(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def _checkout_state(
    root: Path,
    expected_repository: str,
    expected_revision: str,
    expected_dirty_paths: set[str],
) -> dict[str, Any]:
    head = _git(root, "rev-parse", "HEAD^{commit}")
    remote = _git(root, "remote", "get-url", "origin")
    status = _git(root, "status", "--short", "--untracked-files=no").splitlines()
    dirty_paths = {
        line.split(maxsplit=1)[-1].replace("\\", "/")
        for line in status
        if len(line.split(maxsplit=1)) == 2
    }
    return {
        "root": str(root),
        "head": head,
        "origin": remote,
        "tracked_dirty_paths": sorted(dirty_paths),
        "passed": head == expected_revision
        and _normalized_repository(remote) == _normalized_repository(expected_repository)
        and dirty_paths == expected_dirty_paths,
    }


def _validate_config(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = cfg.get("activity_thresholds")
    expected_activity = [round(value / 100, 2) for value in range(5, 100, 5)]
    expected_causal = {
        "active_thresholds": [0.40, 0.50, 0.60, 0.70, 0.80],
        "other_low_thresholds": [0.10, 0.20, 0.30, 0.40],
        "confirm_ms": [400, 600, 800, 1000, 1200],
        "validity_rule": "other_low_threshold < active_threshold",
        "valid_candidate_count": 95,
        "selection_order": [
            "wrong_anchor_rate_ascending",
            "fraction_enrolled_within_1500ms_descending",
            "enrollment_failure_rate_ascending",
            "enrollment_delay_p90_ascending",
            "active_threshold_descending",
            "other_low_threshold_ascending",
            "confirm_ms_ascending",
        ],
    }
    checks = [
        _check(
            "authority_pin",
            cfg.get("authority") == {"ref": AUTHORITY_REF, "sha256": AUTHORITY_PIN},
            cfg.get("authority"),
        ),
        _check(
            "experiment_namespace",
            cfg.get("experiment_id") == "psem-relative-occupancy-v0",
            cfg.get("experiment_id"),
        ),
        _check(
            "sample_rate_16k",
            cfg["dataset"]["sample_rate_hz"] == 16000,
            cfg["dataset"]["sample_rate_hz"],
        ),
        _check(
            "dataset_freeze_file_pin",
            cfg["dataset"].get("freeze_file_sha256") == FREEZE_FILE_SHA256,
            cfg["dataset"].get("freeze_file_sha256"),
        ),
        _check(
            "dataset_freeze_payload_pin",
            cfg["dataset"].get("freeze_payload_sha256") == FREEZE_PAYLOAD_SHA256,
            cfg["dataset"].get("freeze_payload_sha256"),
        ),
        _check(
            "dataset_role_counts_pin",
            cfg["dataset"].get("role_source_counts") == ROLE_SOURCE_COUNTS,
            cfg["dataset"].get("role_source_counts"),
        ),
        _check(
            "evaluation_grid_100ms",
            cfg.get("evaluation_grid_ms") == 100,
            cfg.get("evaluation_grid_ms"),
        ),
        _check(
            "gate0_enrollment_frozen",
            cfg.get("gate0_enrollment_confirm_ms") == 200,
            cfg.get("gate0_enrollment_confirm_ms"),
        ),
        _check(
            "lifecycle_proxy_frozen",
            cfg.get("lifecycle_proxy_silence_reset_ms") == 1200,
            cfg.get("lifecycle_proxy_silence_reset_ms"),
        ),
        _check(
            "diagnostic_handoff_alignment_frozen",
            cfg.get("derived_handoff_alignment_tolerance_ms") == 500,
            cfg.get("derived_handoff_alignment_tolerance_ms"),
        ),
        _check(
            "product_event_alignment_frozen",
            cfg.get("product_event_alignment_tolerance_ms") == 500,
            cfg.get("product_event_alignment_tolerance_ms"),
        ),
        _check(
            "replacement_grid_frozen",
            cfg.get("replacement_confirm_ms") == [100, 200, 300, 500],
            cfg.get("replacement_confirm_ms"),
        ),
        _check(
            "trace_schema_frozen",
            cfg.get("trace_schema_version") == "psem.relative_occupancy.trace.v1",
            cfg.get("trace_schema_version"),
        ),
        _check(
            "oracle_mapping_frozen",
            cfg.get("oracle_anchor_mapping")
            == {
                "support": "duration_weighted_mean_on_unmasked_gt_anchor_active_cells",
                "tie_break": "lowest_slot_index",
                "other_aggregation": "max_alive_non_anchor",
            },
            cfg.get("oracle_anchor_mapping"),
        ),
        _check(
            "activity_threshold_grid_frozen",
            thresholds == expected_activity,
            thresholds,
        ),
        _check(
            "causal_enrollment_grid_frozen",
            cfg.get("causal_enrollment") == expected_causal,
            cfg.get("causal_enrollment"),
        ),
        _check(
            "sortformer_contract_frozen",
            cfg.get("sortformer") == SORTFORMER_PIN,
            cfg.get("sortformer"),
        ),
        _check(
            "lseend_contract_frozen",
            cfg.get("lseend") == LSEEND_PIN,
            cfg.get("lseend"),
        ),
        _check(
            "no_model_learning",
            "training" not in cfg and "optimizer" not in cfg,
            sorted(cfg),
        ),
    ]
    dataset = load_frozen_dataset()
    contract_value = load_json(dataset.root / "operational_label_contract.json")
    checks.extend(
        [
            _check(
                "lifecycle_proxy_matches_v2",
                contract_value.get("constants_ms", {}).get("local_continuity_max_gap")
                == cfg["lifecycle_proxy_silence_reset_ms"],
                contract_value.get("constants_ms", {}).get("local_continuity_max_gap"),
            ),
            _check(
                "gate0_enrollment_matches_v2",
                contract_value.get("constants_ms", {}).get("reliable_solo_min_duration")
                == cfg["gate0_enrollment_confirm_ms"],
                contract_value.get("constants_ms", {}).get("reliable_solo_min_duration"),
            ),
        ]
    )
    return checks


def _model_paths(sr_root: Path, ls_root: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    return {
        "sortformer_model": safe_child(
            sr_root,
            Path("models/r8") / cfg["sortformer"]["model_filename"],
            "Sortformer model",
        ),
        "sortformer_bench": safe_child(
            sr_root,
            cfg["sortformer"]["bench_relative_path"],
            "Sortformer bench",
        ),
        "sortformer_checkout": safe_child(
            sr_root,
            "external/r8/transcribe.cpp",
            "Sortformer checkout",
        ),
        "sortformer_patch": safe_child(
            sr_root,
            "results/r8/streaming_sortformer_feasibility_v1/telemetry_patch.diff",
            "Sortformer telemetry patch",
        ),
        "lseend_model": safe_child(
            ls_root,
            cfg["lseend"]["model_relative_path"],
            "LS-EEND model",
        ),
        "lseend_sidecar": safe_child(
            ls_root,
            cfg["lseend"]["sidecar_relative_path"],
            "LS-EEND sidecar",
        ),
    }


def run_preflight(
    *,
    corpus: Path,
    reference: Path,
    sr_root: Path,
    ls_root: Path,
    acquire: bool,
) -> dict[str, Any]:
    corpus = corpus_root(corpus)
    reference = reference_root(reference)
    sr_root = research_root(sr_root)
    ls_root = lseend_root(ls_root)
    cfg = config()
    if acquire:
        acquire_reference(reference)
    dataset = load_frozen_dataset()
    waveform_binding = validate_corpus_waveforms(dataset, corpus)
    checks = _validate_config(cfg)
    checks.append(_check("frozen_dataset_contract", True, dataset.summary()))
    checks.append(_check("all_v2_waveforms_pinned", True, waveform_binding))
    reference_receipt: dict[str, Any]
    try:
        reference_receipt = validate_reference_checkout(reference)
        reference_ok = (
            reference_receipt.get("repository") == REFERENCE_REPOSITORY
            and reference_receipt.get("commit") == REFERENCE_COMMIT
            and reference_receipt.get("git_tree") == REFERENCE_TREE
        )
    except Exception as exc:
        reference_ok = False
        reference_receipt = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(_check("reference_checkout_pin", reference_ok, reference_receipt))
    paths = _model_paths(sr_root, ls_root, cfg)
    sortformer_checkout = _checkout_state(
        paths["sortformer_checkout"],
        cfg["sortformer"]["source_repository"],
        cfg["sortformer"]["source_commit"],
        {
            "src/arch/sortformer/model.cpp",
            "src/arch/sortformer/sortformer.h",
            "src/arch/sortformer/stream.cpp",
        },
    )
    lseend_checkout = _checkout_state(
        ls_root,
        cfg["lseend"]["repository"],
        cfg["lseend"]["revision"],
        set(),
    )
    checks.extend(
        [
            _check(
                "sortformer_source_checkout_pin",
                sortformer_checkout["passed"],
                sortformer_checkout,
            ),
            _check(
                "lseend_source_checkout_pin",
                lseend_checkout["passed"],
                lseend_checkout,
            ),
            _file_check(
                "sortformer_model_pin",
                paths["sortformer_model"],
                cfg["sortformer"]["model_sha256"],
            ),
            _file_check(
                "sortformer_bench_pin",
                paths["sortformer_bench"],
                cfg["sortformer"]["bench_sha256"],
            ),
            _file_check(
                "sortformer_telemetry_patch_pin",
                paths["sortformer_patch"],
                cfg["sortformer"]["telemetry_patch_sha256"],
            ),
            _file_check(
                "lseend_model_pin",
                paths["lseend_model"],
                cfg["lseend"]["model_sha256"],
            ),
            _file_check(
                "lseend_sidecar_pin",
                paths["lseend_sidecar"],
                cfg["lseend"]["sidecar_sha256"],
            ),
        ]
    )
    closing_dataset = load_frozen_dataset()
    checks.append(
        _check(
            "frozen_dataset_closing_identity",
            closing_dataset.summary() == dataset.summary(),
            closing_dataset.summary(),
        )
    )
    return {
        "schema_version": "psem.relative_occupancy.preflight.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authority": cfg["authority"],
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "dataset": dataset.summary(),
        "paths": {
            key: str(value)
            for key, value in {
                "corpus_root": corpus,
                "reference_root": reference,
                "research_root": sr_root,
                "lseend_root": ls_root,
                **paths,
            }.items()
        },
        "reference_receipt": reference_receipt,
        "model_source_checkouts": {
            "sortformer": sortformer_checkout,
            "lseend": lseend_checkout,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "logical_cpu_count": os.cpu_count(),
        },
        "git": {**_git_state(), "role": "informational_not_acceptance_binding"},
        "eval_status": "sealed",
        "checks": checks,
        "passed": all(row["passed"] for row in checks),
    }


def validate_preflight_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    paths = receipt.get("paths")
    if not isinstance(paths, dict):
        raise PreflightError("preflight paths are missing")
    regenerated = run_preflight(
        corpus=Path(str(paths["corpus_root"])),
        reference=Path(str(paths["reference_root"])),
        sr_root=Path(str(paths["research_root"])),
        ls_root=Path(str(paths["lseend_root"])),
        acquire=False,
    )
    stable_fields = (
        "schema_version",
        "authority",
        "config_path",
        "config_sha256",
        "dataset",
        "paths",
        "reference_receipt",
        "model_source_checkouts",
        "environment",
        "eval_status",
        "checks",
        "passed",
    )
    observed = {field: receipt.get(field) for field in stable_fields}
    expected = {field: regenerated.get(field) for field in stable_fields}
    if canonical_sha256(observed) != canonical_sha256(expected):
        raise PreflightError("preflight receipt does not match current pinned inputs")
    if regenerated["passed"] is not True or regenerated["eval_status"] != "sealed":
        raise PreflightError("preflight receipt did not pass with EVAL sealed")
    return regenerated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--lseend-root", type=Path)
    parser.add_argument("--acquire-reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_preflight(
        corpus=corpus_root(args.corpus_root),
        reference=reference_root(args.reference_root),
        sr_root=research_root(args.research_root),
        ls_root=lseend_root(args.lseend_root),
        acquire=args.acquire_reference,
    )
    write_json(safe_output_path(args.output), receipt)
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
