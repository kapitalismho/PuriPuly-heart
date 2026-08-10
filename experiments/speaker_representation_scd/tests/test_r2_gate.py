from __future__ import annotations

import io
import json
import tarfile
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import experiments.speaker_representation_scd.r2_gate as gate_module
import experiments.speaker_representation_scd.r2_materialize as materialize_module
from experiments.speaker_representation_scd.provenance import with_self_sha256
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT
from experiments.speaker_representation_scd.r2_gate import (
    EXPECTED_ACTIONS,
    GATE_PATH,
    validate_r2_gate,
)
from experiments.speaker_representation_scd.r2_materialize import (
    JVS_CONFIRMATORY,
    JVS_DEVELOPMENT,
    MaterializationBudget,
    R2GateError,
    _complete_range_bytes,
    _download,
    _materialize_jvs,
    _materialize_zeroth,
    _member_parts,
    _reuse_completed_zeroth,
    select_jvs_development_members,
    select_zeroth_development_members,
)


def _flac() -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.zeros(3200, dtype=np.int16), 16000, format="FLAC", subtype="PCM_16")
    return stream.getvalue()


def _wav() -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.zeros(4800, dtype=np.int16), 24000, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def _add_tar_file(handle: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    handle.addfile(member, io.BytesIO(payload))


def _add_jvs_release(handle: zipfile.ZipFile, payload: bytes) -> None:
    for index in range(1, 101):
        speaker = f"jvs{index:03d}"
        for condition in ("parallel100", "nonpara30", "whisper10", "falsetto10"):
            handle.writestr(
                f"jvs_ver1/{speaker}/{condition}/wav24kHz16bit/{condition}_001.wav",
                payload,
            )


def test_r2_gate_is_valid_without_external_execution() -> None:
    result = validate_r2_gate(scan_processes=False)
    assert result.valid, result.errors
    assert result.allowed_actions == EXPECTED_ACTIONS
    assert result.allowed_actions["development_archive_download"] is True
    assert result.allowed_actions["development_waveform_materialization"] is True
    assert result.allowed_actions["full_extraction"] is False
    assert result.allowed_actions["confirmatory_access"] is False
    assert result.allowed_actions["training"] is False


def test_rehashed_semantic_gate_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original_loader = gate_module.load_json
    gate = original_loader(EXPERIMENT_ROOT / GATE_PATH)
    mutated = deepcopy(gate)
    mutated["authorization"]["confirmatory_access"] = True
    mutated = with_self_sha256(mutated)

    def load(path: Path) -> dict:
        if path.resolve() == (EXPERIMENT_ROOT / GATE_PATH).resolve():
            return mutated
        return original_loader(path)

    monkeypatch.setattr(gate_module, "load_json", load)
    result = validate_r2_gate(scan_processes=False)
    assert not result.valid
    assert "r2_gate.authorization: differs" in result.errors


@pytest.mark.parametrize(
    "name",
    ("../audio.wav", "/audio.wav", "C:/audio.wav", "folder\\audio.wav", ""),
)
def test_archive_member_paths_fail_closed(name: str) -> None:
    with pytest.raises(R2GateError):
        _member_parts(name)


def test_zeroth_selector_uses_only_hash_selected_train_speakers(tmp_path: Path) -> None:
    archive = tmp_path / "zeroth.tar.gz"
    train = [f"speaker{i:03d}" for i in range(105)]
    tests = [f"test{i:03d}" for i in range(10)]
    payload = _flac()
    with tarfile.open(archive, "w:gz") as handle:
        for speaker in train:
            _add_tar_file(
                handle,
                f"zeroth/train_data_01/script01/{speaker}/{speaker}_001.flac",
                payload,
            )
        for speaker in tests:
            _add_tar_file(
                handle,
                f"zeroth/test_data_01/script01/{speaker}/{speaker}_001.flac",
                payload,
            )
    members, speakers = select_zeroth_development_members(archive)
    expected = tuple(
        sorted(
            train,
            key=lambda value: (
                materialize_module.hashlib.sha256(value.encode()).hexdigest(),
                value,
            ),
        )[:20]
    )
    assert speakers == expected
    assert len(members) == 20
    assert all("/train_data_01/" in f"/{member.name}" for member in members)
    assert not any("/test_data_01/" in f"/{member.name}" for member in members)


def test_zeroth_selector_rejects_train_test_speaker_overlap(tmp_path: Path) -> None:
    archive = tmp_path / "zeroth.tar.gz"
    payload = _flac()
    with tarfile.open(archive, "w:gz") as handle:
        for index in range(105):
            speaker = f"speaker{index:03d}"
            _add_tar_file(
                handle,
                f"zeroth/train_data_01/script01/{speaker}/{speaker}_001.flac",
                payload,
            )
        for index in range(10):
            speaker = "speaker000" if index == 0 else f"test{index:03d}"
            _add_tar_file(
                handle,
                f"zeroth/test_data_01/script01/{speaker}/{speaker}_001.flac",
                payload,
            )
    with pytest.raises(R2GateError, match="train/test speaker metadata"):
        select_zeroth_development_members(archive)


def test_zeroth_selector_rejects_incomplete_test_population(tmp_path: Path) -> None:
    archive = tmp_path / "zeroth.tar.gz"
    payload = _flac()
    with tarfile.open(archive, "w:gz") as handle:
        for index in range(105):
            speaker = f"speaker{index:03d}"
            _add_tar_file(
                handle,
                f"zeroth/train_data_01/script01/{speaker}/{speaker}_001.flac",
                payload,
            )
        for index in range(9):
            speaker = f"test{index:03d}"
            _add_tar_file(
                handle,
                f"zeroth/test_data_01/script01/{speaker}/{speaker}_001.flac",
                payload,
            )
    with pytest.raises(R2GateError, match="train/test speaker metadata"):
        select_zeroth_development_members(archive)


def test_jvs_selector_excludes_confirmatory_and_unused_speakers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "jvs.zip"
    payload = _wav()
    monkeypatch.setattr(
        materialize_module,
        "JVS_CONDITION_COUNTS",
        {"parallel100": 1, "nonpara30": 1, "whisper10": 1, "falsetto10": 1},
    )
    with zipfile.ZipFile(archive, "w") as handle:
        _add_jvs_release(handle, payload)
    members = select_jvs_development_members(archive)
    assert len(members) == len(JVS_DEVELOPMENT) * 4
    names = [member.filename for member in members]
    assert all(any(f"/{speaker}/" in name for speaker in JVS_DEVELOPMENT) for name in names)
    assert not any(f"/{JVS_CONFIRMATORY[0]}/" in name for name in names)
    assert not any("/jvs001/" in name for name in names)


def test_jvs_selector_rejects_case_colliding_members(tmp_path: Path) -> None:
    archive = tmp_path / "jvs.zip"
    payload = _wav()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "jvs_ver1/jvs046/parallel100/wav24kHz16bit/a.wav",
            payload,
        )
        handle.writestr(
            "jvs_ver1/jvs046/parallel100/wav24kHz16bit/A.wav",
            payload,
        )
    with pytest.raises(R2GateError, match="case-colliding"):
        select_jvs_development_members(archive)


def test_jvs_selector_rejects_missing_condition_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "jvs.zip"
    payload = _wav()
    monkeypatch.setattr(
        materialize_module,
        "JVS_CONDITION_COUNTS",
        {"parallel100": 1, "nonpara30": 1, "whisper10": 1, "falsetto10": 1},
    )
    with zipfile.ZipFile(archive, "w") as handle:
        for index in range(1, 101):
            speaker = f"jvs{index:03d}"
            for condition in ("parallel100", "nonpara30", "whisper10", "falsetto10"):
                if speaker == "jvs001" and condition == "falsetto10":
                    continue
                handle.writestr(
                    f"jvs_ver1/{speaker}/{condition}/wav24kHz16bit/{condition}_001.wav",
                    payload,
                )
    with pytest.raises(R2GateError, match="condition coverage"):
        select_jvs_development_members(archive)


def test_zeroth_materializer_never_opens_test_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "zeroth.tar.gz"
    payload = _flac()
    train = [f"speaker{i:03d}" for i in range(105)]
    with tarfile.open(archive, "w:gz") as handle:
        for speaker in train:
            _add_tar_file(
                handle,
                f"zeroth/train_data_01/script01/{speaker}/{speaker}_001.flac",
                payload,
            )
        for index in range(10):
            speaker = f"test{index:03d}"
            _add_tar_file(
                handle,
                f"zeroth/test_data_01/script01/{speaker}/{speaker}_001.flac",
                payload,
            )
    opened: list[str] = []
    original = tarfile.TarFile.extractfile

    def extractfile(handle: tarfile.TarFile, member):
        opened.append(member.name if isinstance(member, tarfile.TarInfo) else str(member))
        return original(handle, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", extractfile)
    staging = tmp_path / "staging"
    staging.mkdir()
    waveforms, metadata = _materialize_zeroth(
        tmp_path,
        archive,
        staging,
        MaterializationBudget(tmp_path),
    )
    assert len(waveforms) == 20
    assert len(metadata) == 20
    assert opened
    assert all("/train_data_01/" in f"/{name}" for name in opened)
    assert not any("/test_data_01/" in f"/{name}" for name in opened)


def test_jvs_materializer_never_opens_reserved_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "jvs.zip"
    payload = _wav()
    monkeypatch.setattr(
        materialize_module,
        "JVS_CONDITION_COUNTS",
        {"parallel100": 1, "nonpara30": 1, "whisper10": 1, "falsetto10": 1},
    )
    with zipfile.ZipFile(archive, "w") as handle:
        _add_jvs_release(handle, payload)
    opened: list[str] = []
    original = zipfile.ZipFile.read

    def read(handle: zipfile.ZipFile, name, pwd=None):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else str(name))
        return original(handle, name, pwd=pwd)

    monkeypatch.setattr(zipfile.ZipFile, "read", read)
    staging = tmp_path / "staging"
    staging.mkdir()
    waveforms, metadata = _materialize_jvs(
        tmp_path,
        archive,
        staging,
        MaterializationBudget(tmp_path),
    )
    assert len(waveforms) == len(JVS_DEVELOPMENT) * 4
    assert len(metadata) == len(JVS_DEVELOPMENT) * 4
    assert opened
    assert all(any(f"/{speaker}/" in name for speaker in JVS_DEVELOPMENT) for name in opened)
    assert not any(f"/{JVS_CONFIRMATORY[0]}/" in name for name in opened)


def test_download_refuses_preexisting_target_or_partial(tmp_path: Path) -> None:
    target = tmp_path / "archive.zip"
    partial = tmp_path / "archive.zip.part"
    target.write_bytes(b"preexisting")
    with pytest.raises(R2GateError, match="empty fixed target"):
        _download("https://example.invalid/archive", target, partial, max_bytes=100)
    target.unlink()
    partial.write_bytes(b"partial")
    with pytest.raises(R2GateError, match="empty fixed target"):
        _download("https://example.invalid/archive", target, partial, max_bytes=100)


def test_complete_range_requires_the_entire_file() -> None:
    assert _complete_range_bytes("bytes 0-99/100") == 100
    for value in ("", "bytes 1-99/100", "bytes 0-98/100", "items 0-99/100"):
        with pytest.raises(R2GateError):
            _complete_range_bytes(value)


def test_exact_completed_zeroth_recovery_is_identity_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "sources/r2/development/zeroth/archive.tar.gz"
    partial = tmp_path / "control/downloads/r2/archive.tar.gz.part"
    target.parent.mkdir(parents=True)
    with tarfile.open(target, "w:gz"):
        pass
    execution_id = "a" * 32
    usage_relative = f"control/usage/{execution_id}.json"
    usage_path = tmp_path / usage_relative
    usage_path.parent.mkdir(parents=True)
    usage = with_self_sha256(
        {
            "execution_id": execution_id,
            "action": "r2-archives",
            "status": "aborted",
            "action_receipt": None,
            "expected_action_receipt_relative_path": (
                "manifests/r2/development/development_archive_receipt.json"
            ),
        }
    )
    usage_path.write_text(json.dumps(usage), encoding="utf-8")
    monkeypatch.setattr(
        materialize_module,
        "RECOVERABLE_ZEROTH",
        {
            "execution_id": execution_id,
            "usage_relative_path": usage_relative,
            "archive_relative_path": target.relative_to(tmp_path).as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": materialize_module.sha256_file(target),
        },
    )

    transfer = _reuse_completed_zeroth(tmp_path, target, partial)
    assert transfer["reused_from_aborted_execution"] == execution_id

    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(R2GateError, match="identity differs"):
        _reuse_completed_zeroth(tmp_path, target, partial)


def test_materialization_budget_rejects_derived_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(materialize_module, "MAX_DERIVED_BYTES", 10)
    budget = MaterializationBudget(tmp_path)
    with pytest.raises(R2GateError, match="derived ceiling"):
        budget.ensure_projection(11)


def test_gate_file_has_valid_json_identity() -> None:
    gate = json.loads((EXPERIMENT_ROOT / GATE_PATH).read_text(encoding="utf-8"))
    assert gate["self_sha256"] != "pending"
