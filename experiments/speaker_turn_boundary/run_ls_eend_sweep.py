from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.speaker_turn_boundary.adapters.ls_eend import LSEENDOnnxDetector
from experiments.speaker_turn_boundary.config import (
    EXPERIMENT_DATA_DIR,
    EXPERIMENT_RESULTS_DIR,
    VAD_COALESCE_WINDOW_SAMPLES,
)
from experiments.speaker_turn_boundary.frontend import frontend_profile
from experiments.speaker_turn_boundary.metadata import collect_runtime_metadata
from experiments.speaker_turn_boundary.metrics import (
    aggregate_cases,
    incremental_over_b0,
)
from experiments.speaker_turn_boundary.provenance import LS_EEND_VARIANTS
from experiments.speaker_turn_boundary.reducer import ReductionProfile
from experiments.speaker_turn_boundary.sweep_common import (
    b0_aggregate_for_manifest,
    build_sweep_result,
    load_manifest,
    new_summary,
    profile_summary_dict,
    run_b0_and_detector,
)

DEFAULT_THRESHOLDS = [round(0.30 + 0.05 * index, 2) for index in range(9)]
DEFAULT_PERSISTENCE = (1, 2, 3)
DEFAULT_POLICIES = ("new_speaker_onset", "dominant_replacement")
DEFAULT_MEDIAN_WIDTHS = (1, 11)


def threshold_choices() -> list[float]:
    return DEFAULT_THRESHOLDS


def _silero_engine_factory(model_path: Path):
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    return lambda: SileroVadOnnx(model_path)


def run_sweep(
    *,
    manifest_path: Path,
    data_dir: Path,
    hf_root: Path,
    model_path: Path,
    out_dir: Path,
    checkpoints: list[str],
    thresholds: list[float],
    persistence: list[int],
    policies: list[str],
    median_widths: list[int],
    case_ids: list[str] | None,
    coalesce_window_samples: int,
    chunk_samples: int,
    smoke_dir: Path | None,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    engine_factory = _silero_engine_factory(model_path)
    smoke_wavs = sorted(smoke_dir.glob("*.wav")) if smoke_dir is not None else None
    b0_aggregate = b0_aggregate_for_manifest(
        manifest,
        data_dir,
        engine_factory=engine_factory,
        case_ids=case_ids,
        coalesce_window_samples=coalesce_window_samples,
    )
    summary = new_summary(manifest_id=manifest.manifest_id, detector_family="ls_eend")
    runtime_metadata = collect_runtime_metadata()
    for variant in checkpoints:
        info = LS_EEND_VARIANTS[variant]
        onnx_path = hf_root / info["dir"] / info["onnx"]
        sidecar_path = hf_root / info["dir"] / info["sidecar"]
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        variant_summary: dict[str, object] = {}
        for profile in _iter_profiles(thresholds, persistence, policies, median_widths):
            started = time.perf_counter()
            detector = LSEENDOnnxDetector(
                onnx_path,
                metadata,
                profile,
                checkpoint_variant=variant,
            )
            records, smoke_records = run_b0_and_detector(
                manifest,
                data_dir,
                engine_factory=engine_factory,
                detector_run=detector,
                case_ids=case_ids,
                coalesce_window_samples=coalesce_window_samples,
                smoke_wavs=smoke_wavs,
            )
            aggregate = aggregate_cases(
                [record.metrics for record in records],
                profile_id=f"{variant}:{profile.profile_id}",
                coalescing_reports=[record.coalescing.report for record in records],
                smoke_metrics=[record.metrics for record in smoke_records],
            )
            incremental = incremental_over_b0(b0_aggregate, aggregate)
            profile_id = f"{variant}:{profile.profile_id}"
            extra = {
                "frontend_profile": frontend_profile().to_dict(),
                "checkpoint_onnx_sha256": info["onnx_sha256"],
                "sidecar_sha256": info["sidecar_sha256"],
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
            build_sweep_result(
                manifest=manifest,
                detector={
                    "family": "ls_eend",
                    "checkpoint": variant,
                    "checkpoint_onnx_sha256": info["onnx_sha256"],
                    "profile": profile.to_dict(),
                    "profile_id": profile.profile_id,
                    "coalesce_window_samples": coalesce_window_samples,
                    "chunk_samples": chunk_samples,
                },
                runtime_metadata=runtime_metadata,
                epochs=records,
                aggregate=aggregate,
                b0_aggregate=b0_aggregate,
                incremental=incremental,
                smoke_epochs=smoke_records,
                extra=extra,
                out_dir=out_dir,
                out_name=f"sweep_ls_eend_{profile_id.replace(':', '_').replace('.', 'p')}.json",
            )
            variant_summary[profile.profile_id] = profile_summary_dict(
                aggregate, b0_aggregate, incremental
            )
        summary["variants"][variant] = variant_summary
    summary_path = out_dir / f"sweep_ls_eend_summary_{manifest.manifest_id}.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {summary_path}")
    return summary


def _iter_profiles(
    thresholds: list[float],
    persistence: list[int],
    policies: list[str],
    median_widths: list[int],
):
    for policy in policies:
        for median_width in median_widths:
            for persistence_value in persistence:
                for threshold in thresholds:
                    yield ReductionProfile(
                        threshold=threshold,
                        persistence=persistence_value,
                        policy=policy,
                        median_width=median_width,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 LS-EEND reducer sweep on the Phase 0 corpus"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT_DATA_DIR / "manifests" / "b0_phase0.json",
    )
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT_DATA_DIR)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=EXPERIMENT_RESULTS_DIR)
    parser.add_argument("--checkpoint", action="append", default=None)
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=None,
        help="repeatable; default 0.30..0.70 step 0.05",
    )
    parser.add_argument("--persistence", action="append", type=int, default=None)
    parser.add_argument("--policy", action="append", default=None)
    parser.add_argument("--median", action="append", type=int, default=None)
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--coalesce-window-samples", type=int, default=VAD_COALESCE_WINDOW_SAMPLES)
    parser.add_argument("--chunk-samples", type=int, default=512)
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=None,
        help="directory of 16k mono wavs run without GT (smoke inputs)",
    )
    args = parser.parse_args()

    checkpoints = args.checkpoint or sorted(LS_EEND_VARIANTS)
    thresholds = args.threshold or DEFAULT_THRESHOLDS
    persistence = args.persistence or list(DEFAULT_PERSISTENCE)
    policies = args.policy or list(DEFAULT_POLICIES)
    median_widths = args.median or list(DEFAULT_MEDIAN_WIDTHS)
    model_path = args.model
    if model_path is None:
        from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path

        model_path = Path(str(bundled_silero_vad_onnx_path()))
    run_sweep(
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        hf_root=args.hf_root,
        model_path=model_path,
        out_dir=args.out,
        checkpoints=checkpoints,
        thresholds=thresholds,
        persistence=persistence,
        policies=policies,
        median_widths=median_widths,
        case_ids=args.case,
        coalesce_window_samples=args.coalesce_window_samples,
        chunk_samples=args.chunk_samples,
        smoke_dir=args.smoke_dir,
    )


if __name__ == "__main__":
    main()
