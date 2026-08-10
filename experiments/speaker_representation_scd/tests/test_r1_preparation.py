from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from experiments.speaker_representation_scd import acquire_r1
from experiments.speaker_representation_scd.acquire_r1 import _git_checkout
from experiments.speaker_representation_scd.acquire_r1 import (
    _write_json as write_acquisition_json,
)
from experiments.speaker_representation_scd.execution_guard import WorkerExecution
from experiments.speaker_representation_scd.extraction.common import (
    ExtractionBatch,
    l2_normalize,
    mean_pool_valid,
    trailing_window,
)
from experiments.speaker_representation_scd.extraction.fixtures import (
    EXPECTED_FIXTURE_MANIFEST_SHA256,
    EXPECTED_FIXTURE_SHA256,
    d0_fixtures,
    fixture_manifest,
    fixture_window_contract,
    mutate_future,
)
from experiments.speaker_representation_scd.provenance import (
    canonical_json_bytes,
    load_json,
    self_sha256_valid,
    sha256_bytes,
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_gate import (
    EXPECTED_ACTIONS,
    EXPECTED_EXECUTION_CODE_PATHS,
    GATE_PATH,
    SOURCE_REGISTRY_PATH,
    R1GateError,
    _validate_source_registry,
    validate_r1_gate,
)
from experiments.speaker_representation_scd.r1_smoke import (
    _expected_eres_lengths,
    _expected_ssl_length,
    _timestamp_mapping,
)
from experiments.speaker_representation_scd.r1_smoke import (
    _write_json as write_smoke_json,
)
from experiments.speaker_representation_scd.validate_r0 import DEFAULT_PATHS
from experiments.speaker_representation_scd.windows_job import MAX_JOB_MEMORY_BYTES

ROOT = Path(__file__).resolve().parents[1]


def test_r1_source_registry_is_self_hashed_and_bridges_r0_artifacts() -> None:
    source = load_json(ROOT / "models" / "source_registry.json")
    r0 = load_json(ROOT / "models" / "registry.json")
    assert self_sha256_valid(source)
    assert source["r0_registry"]["sha256"] == sha256_file(ROOT / "models" / "registry.json")
    r0_hashes = {model["model_id"]: model["artifact"]["sha256"] for model in r0["models"]}
    source_hashes = {
        model["model_id"]: next(
            row["sha256"]
            for row in model["required_files"]
            if row["path"] in {"model.safetensors", "pytorch_model.bin"}
        )
        for model in source["models"]
    }
    source_hashes[source["eres2netv2"]["model_id"]] = source["eres2netv2"]["checkpoint_file"][
        "sha256"
    ]
    assert source_hashes == r0_hashes


def test_eres_taps_are_explicit_and_fused_is_official_pool_input() -> None:
    source = load_json(ROOT / "models" / "source_registry.json")
    taps = source["eres2netv2"]["taps"]
    assert [tap["tap_id"] for tap in taps] == ["S1", "S2", "S3", "S4", "FUSED"]
    assert [tap["flattened_dimension"] for tap in taps] == [10240] * 5
    assert [tap["temporal_stride_ms"] for tap in taps] == [10, 20, 40, 80, 80]
    assert [tap["official_pool_input"] for tap in taps] == [False, False, False, False, True]


def test_r1_model_file_inventory_matches_the_gate_byte_forecast() -> None:
    source = load_json(ROOT / "models" / "source_registry.json")
    total = sum(row["size_bytes"] for model in source["models"] for row in model["required_files"])
    total += source["eres2netv2"]["checkpoint_file"]["size_bytes"]
    total += source["eres2netv2"]["checkpoint_config"]["size_bytes"]
    assert total == 1_209_138_839


def test_r1_acquisition_gate_is_valid_without_a_live_process_scan() -> None:
    result = validate_r1_gate(scan_processes=False)
    assert result.valid, result.errors
    assert result.allowed_actions == EXPECTED_ACTIONS
    gate = load_json(ROOT / GATE_PATH)
    release = load_json(ROOT / "results" / "r1" / "legacy_release.json")
    assert self_sha256_valid(gate)
    assert self_sha256_valid(release)


def test_rehashed_r1_action_expansion_is_rejected(tmp_path: Path) -> None:
    paths = set(DEFAULT_PATHS.values()) | {
        str(GATE_PATH).replace("\\", "/"),
        str(SOURCE_REGISTRY_PATH).replace("\\", "/"),
        "environment/pyproject.toml",
        "environment/uv.lock",
        "results/r1/legacy_release.json",
        *EXPECTED_EXECUTION_CODE_PATHS,
    }
    for relative in paths:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    gate_path = tmp_path / GATE_PATH
    gate = load_json(gate_path)
    gate["authorization"]["corpus_download"] = True
    gate = with_self_sha256(gate)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    result = validate_r1_gate(tmp_path, scan_processes=False)
    assert not result.valid
    assert "r1_gate.authorization: action boundary differs" in result.errors


def test_rehashed_remote_code_or_source_identity_mutation_is_rejected(tmp_path: Path) -> None:
    source = load_json(ROOT / "models" / "source_registry.json")
    source["models"][0]["trust_remote_code"] = True
    source = with_self_sha256(source)
    path = tmp_path / "source_registry.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    errors: list[str] = []
    _validate_source_registry(path, errors)
    assert any("not the reviewed R1 source identity" in error for error in errors)
    assert any("remote code must be false" in error for error in errors)


def test_d0_fixtures_have_frozen_identities_and_future_mutation_is_after_frontier() -> None:
    fixtures = d0_fixtures()
    assert len(fixtures) == 10
    assert {fixture.fixture_id: fixture.waveform_sha256 for fixture in fixtures} == (
        EXPECTED_FIXTURE_SHA256
    )
    assert sha256_bytes(canonical_json_bytes(fixture_manifest(fixtures))) == (
        EXPECTED_FIXTURE_MANIFEST_SHA256
    )
    assert {fixture.scenario_kind for fixture in fixtures} == {
        "silence",
        "one_speaker",
        "clean_a_to_b",
        "gap_a_to_b",
        "overlap_a_to_b",
        "backchannel_b_to_a",
        "gain_step_same_speaker",
        "noise_step_same_speaker",
        "timestamp_impulses",
        "channel_chirp_same_speaker",
    }
    for fixture in fixtures:
        contract = fixture_window_contract(fixture)
        assert contract["passed"], contract["errors"]
        changed = mutate_future(fixture)
        assert np.array_equal(
            fixture.waveform[: fixture.frontier_sample],
            changed[: fixture.frontier_sample],
        )
        assert not np.array_equal(
            fixture.waveform[fixture.frontier_sample :],
            changed[fixture.frontier_sample :],
        )
        assert np.array_equal(
            trailing_window(
                fixture.waveform,
                fixture.frontier_sample,
                fixture.window_samples,
            ),
            trailing_window(
                changed,
                fixture.frontier_sample,
                fixture.window_samples,
            ),
        )


def test_d0_transition_windows_contain_the_named_scenario() -> None:
    fixtures = {fixture.fixture_id: fixture for fixture in d0_fixtures()}

    def active(fixture_id: str, sample: int) -> set[str]:
        return {
            speaker
            for speaker, start, end in fixtures[fixture_id].speaker_segments
            if start <= sample < end
        }

    clean = fixtures["clean_a_to_b"]
    clean_event = clean.event_samples[0]
    assert clean.frontier_sample - clean.window_samples < clean_event < clean.frontier_sample
    assert active(clean.fixture_id, clean_event - 1) == {"A"}
    assert active(clean.fixture_id, clean_event) == {"B"}

    gap = fixtures["gap_a_to_b"]
    gap_event = gap.event_samples[0]
    gap_start = max(end for speaker, _, end in gap.speaker_segments if speaker == "A")
    assert gap.frontier_sample - gap.window_samples < gap_start < gap_event < gap.frontier_sample
    assert active(gap.fixture_id, gap_start - 1) == {"A"}
    assert active(gap.fixture_id, gap_start) == set()
    assert active(gap.fixture_id, gap_event) == {"B"}

    overlap = fixtures["overlap_a_to_b"]
    overlap_onset, overlap_exclusive = overlap.event_samples
    assert (
        overlap.frontier_sample - overlap.window_samples
        < overlap_onset
        < overlap_exclusive
        < overlap.frontier_sample
    )
    assert active(overlap.fixture_id, overlap_onset - 1) == {"A"}
    assert active(overlap.fixture_id, overlap_onset) == {"A", "B"}
    assert active(overlap.fixture_id, overlap_exclusive) == {"B"}

    backchannel = fixtures["backchannel_b_to_a"]
    backchannel_onset, backchannel_offset = backchannel.event_samples
    assert (
        backchannel.frontier_sample - backchannel.window_samples
        < backchannel_onset
        < backchannel_offset
        < backchannel.frontier_sample
    )
    assert active(backchannel.fixture_id, backchannel_onset - 1) == {"A"}
    assert active(backchannel.fixture_id, backchannel_onset) == {"A", "B"}
    assert active(backchannel.fixture_id, backchannel_offset) == {"A"}

    for fixture_id in (
        "gain_step_same_speaker",
        "noise_step_same_speaker",
        "channel_chirp_same_speaker",
    ):
        fixture = fixtures[fixture_id]
        event = fixture.event_samples[0]
        window_start = fixture.frontier_sample - fixture.window_samples
        assert window_start < event < fixture.frontier_sample
        assert fixture.nuisance_reference is not None
        assert np.array_equal(
            fixture.waveform[window_start:event],
            fixture.nuisance_reference[window_start:event],
        )
        assert not np.array_equal(
            fixture.waveform[event : fixture.frontier_sample],
            fixture.nuisance_reference[event : fixture.frontier_sample],
        )
        assert active(fixture_id, event - 1) == {"A"}
        assert active(fixture_id, event) == {"A"}


def test_valid_mean_pool_excludes_tail_and_l2_marks_zero_norm() -> None:
    values = np.asarray(
        [
            [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    pooled = mean_pool_valid(values, np.asarray([2, 3], dtype=np.int64))
    assert np.array_equal(pooled[0], np.asarray([2.0, 4.0], dtype=np.float32))
    normalized, valid = l2_normalize(pooled)
    assert valid.tolist() == [True, False]
    assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.isnan(normalized[1]).all()


def test_eres_analytic_lengths_cover_the_primary_window_grid() -> None:
    expected = {
        1600: {"S1": 8, "S2": 4, "S3": 2, "S4": 1, "FUSED": 1},
        3200: {"S1": 18, "S2": 9, "S3": 5, "S4": 3, "FUSED": 3},
        4800: {"S1": 28, "S2": 14, "S3": 7, "S4": 4, "FUSED": 4},
        8000: {"S1": 48, "S2": 24, "S3": 12, "S4": 6, "FUSED": 6},
        12000: {"S1": 73, "S2": 37, "S3": 19, "S4": 10, "FUSED": 10},
        16000: {"S1": 98, "S2": 49, "S3": 25, "S4": 13, "FUSED": 13},
    }
    assert {samples: _expected_eres_lengths(samples) for samples in expected} == expected


def test_ssl_independent_length_geometry_covers_the_primary_window_grid() -> None:
    assert {
        samples: _expected_ssl_length(samples) for samples in (1600, 3200, 4800, 8000, 12000, 16000)
    } == {1600: 4, 3200: 9, 4800: 14, 8000: 24, 12000: 37, 16000: 49}


def test_timestamp_mapping_uses_exact_window_frontier() -> None:
    fixture = d0_fixtures()[2]
    batch = ExtractionBatch(
        model_id="wavlm-base-plus",
        layers={"L6": np.zeros((1, 14, 2), dtype=np.float32)},
        valid_lengths={"L6": np.asarray([14], dtype=np.int64)},
        observed_source_samples=np.asarray([fixture.frontier_sample], dtype=np.int64),
    )
    mapping = _timestamp_mapping(batch, fixture)
    assert mapping["window_end_sample"] == fixture.frontier_sample
    assert mapping["representation_availability_source_sample"] == fixture.frontier_sample
    assert mapping["last_frame_support_samples"][1] <= fixture.frontier_sample
    assert mapping["next_hypothetical_frame_support_end_sample"] > fixture.frontier_sample
    assert mapping["passed"] is True


@pytest.mark.parametrize("writer", [write_acquisition_json, write_smoke_json])
def test_r1_evidence_writers_refuse_overwrite(tmp_path: Path, writer) -> None:
    path = tmp_path / "receipt.json"
    writer(path, {"schema_version": 1})
    with pytest.raises(R1GateError, match="refusing to overwrite"):
        writer(path, {"schema_version": 1})


def _local_git_repository(path: Path) -> tuple[str, str]:
    path.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "r1@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "R1 Test"], cwd=path, check=True)
    (path / "artifact.txt").write_text("reviewed artifact\n", encoding="utf-8")
    subprocess.run(["git", "add", "artifact.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "fixture"], cwd=path, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    return str(path.resolve()), revision


def test_git_checkout_accepts_only_the_exact_interrupted_no_checkout_state(
    tmp_path: Path,
) -> None:
    repository, revision = _local_git_repository(tmp_path / "source")
    target = tmp_path / "target"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", repository, str(target)],
        check=True,
    )
    assert not [path for path in target.iterdir() if path.name != ".git"]
    _git_checkout(
        repository,
        revision,
        target,
        cache_root=tmp_path,
        current_execution_id="2" * 32,
    )
    assert (target / "artifact.txt").read_text(encoding="utf-8") == "reviewed artifact\n"
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=target, text=True
    ).strip()


def test_git_checkout_rejects_unexpected_content_in_no_checkout_target(
    tmp_path: Path,
) -> None:
    repository, revision = _local_git_repository(tmp_path / "source")
    target = tmp_path / "target"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", repository, str(target)],
        check=True,
    )
    (target / "unexpected.txt").write_text("not acquisition state\n", encoding="utf-8")
    with pytest.raises(R1GateError, match="dirty before checkout"):
        _git_checkout(
            repository,
            revision,
            target,
            cache_root=tmp_path,
            current_execution_id="2" * 32,
        )


def _aborted_model_usage_for_lock(
    cache_root: Path, lock: Path, execution_id: str = "3" * 32
) -> dict:
    lock_time = datetime.fromtimestamp(lock.stat().st_mtime, UTC)
    usage = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r1_resource_usage",
            "execution_id": execution_id,
            "action": "models",
            "status": "aborted",
            "elapsed_seconds": 2.0,
            "failure_reason": "ExecutionGuardError: interrupted",
            "started_at_utc": (lock_time - timedelta(seconds=1)).isoformat(),
            "completed_at_utc": (lock_time + timedelta(seconds=1)).isoformat(),
            "expected_action_receipt_relative_path": "manifests/r1_model_acquisition.json",
            "action_receipt": None,
            "hard_memory_boundary": {
                "mechanism": "windows_job_object_job_memory",
                "contract_ceiling_bytes": MAX_JOB_MEMORY_BYTES,
                "enforced_job_memory_limit_bytes": MAX_JOB_MEMORY_BYTES - 1024**3,
                "reserved_headroom_bytes": 1024**3,
                "preassignment_commit_bytes": 0,
                "authoritative_peak_job_memory_bytes": 1024,
                "applied": True,
            },
        }
    )
    path = cache_root / "control" / "usage" / f"{execution_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(usage), encoding="utf-8")
    return usage


def test_git_checkout_recovers_only_a_job_attributed_zero_byte_stale_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision = _local_git_repository(tmp_path / "source")
    target = tmp_path / "target"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", repository, str(target)],
        check=True,
    )
    lock = target / ".git" / "index.lock"
    lock.write_bytes(b"")
    usage = _aborted_model_usage_for_lock(tmp_path, lock)
    monkeypatch.setattr(acquire_r1, "_active_git_processes", lambda: ())
    recoveries = _git_checkout(
        repository,
        revision,
        target,
        cache_root=tmp_path,
        current_execution_id="2" * 32,
    )
    assert not lock.exists()
    assert len(recoveries) == 1
    assert recoveries[0]["source_execution_id"] == usage["execution_id"]
    assert recoveries[0]["source_usage_self_sha256"] == usage["self_sha256"]
    recovery_path = tmp_path / recoveries[0]["relative_path"]
    recovery = load_json(recovery_path)
    assert self_sha256_valid(recovery)
    assert recovery["lock"]["size_bytes"] == 0
    assert recovery["recovery_authorized_execution_id"] == "2" * 32


def test_git_checkout_rejects_stale_lock_without_one_aborted_job_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision = _local_git_repository(tmp_path / "source")
    target = tmp_path / "target"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", repository, str(target)],
        check=True,
    )
    (target / ".git" / "index.lock").write_bytes(b"")
    monkeypatch.setattr(acquire_r1, "_active_git_processes", lambda: ())
    with pytest.raises(R1GateError, match="exactly one aborted model action"):
        _git_checkout(
            repository,
            revision,
            target,
            cache_root=tmp_path,
            current_execution_id="2" * 32,
        )


def test_git_checkout_rejects_nonempty_index_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision = _local_git_repository(tmp_path / "source")
    target = tmp_path / "target"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", repository, str(target)],
        check=True,
    )
    lock = target / ".git" / "index.lock"
    lock.write_bytes(b"owned")
    monkeypatch.setattr(acquire_r1, "_active_git_processes", lambda: ())
    with pytest.raises(R1GateError, match="expected zero-byte file"):
        _git_checkout(
            repository,
            revision,
            target,
            cache_root=tmp_path,
            current_execution_id="2" * 32,
        )
    assert lock.read_bytes() == b"owned"


def test_git_checkout_rejects_stale_lock_while_git_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision = _local_git_repository(tmp_path / "source")
    target = tmp_path / "target"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", repository, str(target)],
        check=True,
    )
    lock = target / ".git" / "index.lock"
    lock.write_bytes(b"")
    _aborted_model_usage_for_lock(tmp_path, lock)
    monkeypatch.setattr(
        acquire_r1,
        "_active_git_processes",
        lambda: ({"pid": 42, "name": "git.exe", "command": ["git"]},),
    )
    with pytest.raises(R1GateError, match="active Git processes"):
        _git_checkout(
            repository,
            revision,
            target,
            cache_root=tmp_path,
            current_execution_id="2" * 32,
        )
    assert lock.exists()
    assert not (tmp_path / "control" / "recoveries").exists()


def test_git_checkout_rejects_multiple_temporal_usage_attributions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision = _local_git_repository(tmp_path / "source")
    target = tmp_path / "target"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", repository, str(target)],
        check=True,
    )
    lock = target / ".git" / "index.lock"
    lock.write_bytes(b"")
    _aborted_model_usage_for_lock(tmp_path, lock, "3" * 32)
    _aborted_model_usage_for_lock(tmp_path, lock, "4" * 32)
    monkeypatch.setattr(acquire_r1, "_active_git_processes", lambda: ())
    with pytest.raises(R1GateError, match="exactly one aborted model action: 2"):
        _git_checkout(
            repository,
            revision,
            target,
            cache_root=tmp_path,
            current_execution_id="2" * 32,
        )
    assert lock.exists()


def test_git_checkout_reuses_crash_safe_recovery_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision = _local_git_repository(tmp_path / "source")
    target = tmp_path / "target"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", repository, str(target)],
        check=True,
    )
    lock = target / ".git" / "index.lock"
    lock.write_bytes(b"")
    _aborted_model_usage_for_lock(tmp_path, lock, "3" * 32)
    monkeypatch.setattr(acquire_r1, "_active_git_processes", lambda: ())
    first = _git_checkout(
        repository,
        revision,
        target,
        cache_root=tmp_path,
        current_execution_id="2" * 32,
    )
    clock = tmp_path / "recovery-attempt.clock"
    clock.write_bytes(b"")
    _aborted_model_usage_for_lock(tmp_path, clock, "2" * 32)
    second = _git_checkout(
        repository,
        revision,
        target,
        cache_root=tmp_path,
        current_execution_id="5" * 32,
    )
    assert second == first
    assert len(second) == 1
    assert second[0]["recovery_authorized_execution_id"] == "2" * 32


def _authoritative_predecessor_environment_receipt(
    cache_root: Path, gate: dict, *, mutate_environment: bool = False
) -> dict:
    execution_id = "1" * 32
    relative = "manifests/r1_environment_sync.json"
    environment = json.loads(json.dumps(gate["environment"]))
    if mutate_environment:
        environment["backend"] = "changed"
    predecessor = gate["receipt_compatibility"]["environment_sync_predecessors"][0]
    receipt = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r1_environment_sync_receipt",
            "supervision_binding": {
                "execution_id": execution_id,
                "expected_receipt_relative_path": relative,
                "authority": "requires_completed_usage_attestation",
            },
            "r1_gate_sha256": predecessor["r1_gate_sha256"],
            "r1_gate_self_sha256": predecessor["r1_gate_self_sha256"],
            "execution_code_manifest_sha256": predecessor[
                "execution_code_manifest_sha256"
            ],
            "environment_contract": environment,
        }
    )
    receipt_path = cache_root / relative
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    usage = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r1_resource_usage",
            "execution_id": execution_id,
            "action": "sync-environment",
            "status": "completed",
            "elapsed_seconds": 1.0,
            "failure_reason": None,
            "expected_action_receipt_relative_path": relative,
            "action_receipt": {
                "relative_path": relative,
                "sha256": sha256_file(receipt_path),
                "self_sha256": receipt["self_sha256"],
                "execution_id": execution_id,
            },
            "hard_memory_boundary": {
                "mechanism": "windows_job_object_job_memory",
                "contract_ceiling_bytes": MAX_JOB_MEMORY_BYTES,
                "enforced_job_memory_limit_bytes": MAX_JOB_MEMORY_BYTES - 1024**3,
                "reserved_headroom_bytes": 1024**3,
                "preassignment_commit_bytes": 0,
                "authoritative_peak_job_memory_bytes": 1024,
                "applied": True,
            },
        }
    )
    usage_path = cache_root / "control" / "usage" / f"{execution_id}.json"
    usage_path.parent.mkdir(parents=True)
    usage_path.write_text(json.dumps(usage), encoding="utf-8")
    return receipt


def test_acquire_models_accepts_authoritative_predecessor_environment_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = load_json(ROOT / GATE_PATH)
    sync_receipt = _authoritative_predecessor_environment_receipt(tmp_path, gate)
    requested_argv = ("r1_execute", "models")
    monkeypatch.setattr(acquire_r1, "validated_cache_root", lambda _action: tmp_path)
    monkeypatch.setattr(acquire_r1, "_runtime_versions", lambda: acquire_r1.EXPECTED_RUNTIME)
    monkeypatch.setattr(
        acquire_r1,
        "validate_worker_execution",
        lambda _root, _receipt: WorkerExecution(
            "2" * 32,
            requested_argv,
            "manifests/r1_model_acquisition.json",
        ),
    )
    monkeypatch.setattr(
        acquire_r1,
        "_acquire_huggingface",
        lambda model, _root: {"model_id": model["model_id"]},
    )
    monkeypatch.setattr(acquire_r1, "run_provenance", lambda *_args, **_kwargs: {})
    result = acquire_r1.acquire_models(
        tmp_path,
        {"mhubert-147"},
        requested_argv,
    )
    assert result["models"] == [{"model_id": "mhubert-147"}]
    assert result["environment_sync_receipt_self_sha256"] == sync_receipt["self_sha256"]


def test_acquire_models_rejects_predecessor_receipt_with_changed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = load_json(ROOT / GATE_PATH)
    _authoritative_predecessor_environment_receipt(tmp_path, gate, mutate_environment=True)
    requested_argv = ("r1_execute", "models")
    monkeypatch.setattr(acquire_r1, "validated_cache_root", lambda _action: tmp_path)
    monkeypatch.setattr(acquire_r1, "_runtime_versions", lambda: acquire_r1.EXPECTED_RUNTIME)
    monkeypatch.setattr(
        acquire_r1,
        "validate_worker_execution",
        lambda _root, _receipt: WorkerExecution(
            "2" * 32,
            requested_argv,
            "manifests/r1_model_acquisition.json",
        ),
    )
    with pytest.raises(R1GateError, match="another environment contract"):
        acquire_r1.acquire_models(tmp_path, {"mhubert-147"}, requested_argv)
