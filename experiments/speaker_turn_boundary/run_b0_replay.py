from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from experiments.speaker_turn_boundary.coalescing import (
    CoalesceConfig,
    coalesce_vad_and_detector,
)
from experiments.speaker_turn_boundary.config import (
    B0_VAD_HANGOVER_MS,
    B0_VAD_PRE_ROLL_MS,
    BASELINE_SHA,
    EXPERIMENT_DATA_DIR,
    EXPERIMENT_RESULTS_DIR,
    RESULT_SCHEMA_VERSION,
    VAD_COALESCE_WINDOW_SAMPLES,
)
from experiments.speaker_turn_boundary.events import SpeakerBoundaryEvent
from experiments.speaker_turn_boundary.metadata import collect_runtime_metadata
from experiments.speaker_turn_boundary.schemas import (
    DatasetManifest,
    RunResult,
    validate_manifest,
)
from experiments.speaker_turn_boundary.vad_baseline import replay_wav_epoch


def _silero_engine_factory(model_path: Path):
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    return lambda: SileroVadOnnx(model_path)


def _load_detector_events(path: Path) -> list[SpeakerBoundaryEvent]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SpeakerBoundaryEvent.from_dict(item) for item in data]


def run(
    manifest: DatasetManifest,
    wav_root: Path,
    *,
    case_ids: list[str] | None,
    detector_events: list[SpeakerBoundaryEvent] | None,
    model_path: Path,
    out_dir: Path,
) -> RunResult:
    validate_manifest(manifest, wav_root)
    engine_factory = _silero_engine_factory(model_path)
    started_at = datetime.now(timezone.utc).isoformat()
    selected = [case for case in manifest.cases if case_ids is None or case.case_id in case_ids]
    epochs: list[dict[str, object]] = []
    vad_boundaries: list[SpeakerBoundaryEvent] = []
    for epoch_index, case in enumerate(selected):
        wav_path = (wav_root / case.wav_relative_path).resolve()
        result = replay_wav_epoch(
            wav_path,
            audio_epoch=epoch_index,
            engine_factory=engine_factory,
        )
        vad_boundaries.extend(result.boundaries)
        epochs.append(result.to_dict())
    outcome = coalesce_vad_and_detector(
        vad_boundaries,
        detector_events or [],
        config=CoalesceConfig(window_samples=VAD_COALESCE_WINDOW_SAMPLES),
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    result = RunResult(
        result_id=str(uuid.uuid4()),
        schema_version=RESULT_SCHEMA_VERSION,
        baseline_sha=BASELINE_SHA,
        profile_id=f"b0-{B0_VAD_HANGOVER_MS}ms-hangover-{B0_VAD_PRE_ROLL_MS}ms-preroll",
        manifest_id=manifest.manifest_id,
        manifest_sha256=manifest.hash,
        seed=int(manifest.generator.get("seed") or 0),
        runtime_metadata=collect_runtime_metadata(),
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        epochs=epochs,
        coalescing=outcome.to_dict(),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"result_{manifest.manifest_id}_{result.result_id[:8]}.json"
    hashed = result.with_self_hash()
    result.write(result_path)
    print(f"wrote result {result_path}")
    print(f"result_sha256={hashed.result_sha256}")
    print(json.dumps(outcome.report.to_dict(), indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay production VAD (B0) over Phase 0 manifest cases"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT_DATA_DIR / "manifests" / "b0_phase0.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=EXPERIMENT_DATA_DIR,
        help="experiment data root that contains wav files (default: %(default)s)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="case id to replay (repeatable; default: all cases)",
    )
    parser.add_argument(
        "--detector-events",
        type=Path,
        default=None,
        help="JSON list of SpeakerBoundaryEvent dicts to coalesce against B0",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Silero VAD ONNX path (default: bundled dev model)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EXPERIMENT_RESULTS_DIR,
        help="result output directory (default: %(default)s)",
    )
    args = parser.parse_args()
    manifest = DatasetManifest.load(args.manifest)
    model_path = args.model
    if model_path is None:
        from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path

        model_path = Path(str(bundled_silero_vad_onnx_path()))
    detector_events = _load_detector_events(args.detector_events) if args.detector_events else None
    run(
        manifest,
        args.data_dir,
        case_ids=args.case,
        detector_events=detector_events,
        model_path=model_path,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
