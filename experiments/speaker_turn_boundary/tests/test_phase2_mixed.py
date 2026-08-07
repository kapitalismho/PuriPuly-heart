from __future__ import annotations

import hashlib
import json

from experiments.speaker_turn_boundary.corpus.mixing import build_mixed_dev_manifest
from experiments.speaker_turn_boundary.corpus.phase2_schemas import (
    Phase2Case,
    SourceRef,
    make_phase2_manifest,
)
from experiments.speaker_turn_boundary.corpus.validation import (
    check_disjoint_global_actors,
    check_disjoint_sessions,
    check_disjoint_speakers,
    global_actor_ids,
    manifest_identity_evidence,
)
from experiments.speaker_turn_boundary.ground_truth import SpeakerRegion
from experiments.speaker_turn_boundary.schemas import canonical_json


def _synthetic_manifest(manifest_id: str, speakers: list[str], sessions: list[str]) -> object:
    cases = []
    for index, (speaker, session) in enumerate(zip(speakers, sessions)):
        cases.append(
            Phase2Case(
                case_id=f"{manifest_id}_case{index}",
                wav_relative_path="",
                duration_samples=16000,
                wav_sha256="",
                seed=1,
                regions=[SpeakerRegion(0, 0, 16000, frozenset())],
                kind="silence",
                condition={},
                sources=[
                    SourceRef(
                        role="speaker_0",
                        speaker=speaker,
                        session=session,
                        utterance=session,
                        file_sha256="",
                        original_start_sample=0,
                        original_end_sample=16000,
                        trimmed_start_sample=0,
                        trimmed_end_sample=16000,
                        cut_start_sample=0,
                        cut_end_sample=16000,
                        gain=1.0,
                    )
                ],
                active_speech_samples=0,
            )
        )
    return make_phase2_manifest(
        manifest_id=manifest_id,
        split_role="dev" if "dev" in manifest_id else "held_out",
        corpus={"name": "fake"},
        build={"script": "test"},
        disjointness_groups=[f"group_{manifest_id}"],
        generator={"script": "test"},
        cases=cases,
    )


def test_disjointness_checks_speaker_and_session(tmp_path):
    dev = _synthetic_manifest("ls_dev", ["100", "101"], ["100-1211", "101-1234"])
    held = _synthetic_manifest("ls_held", ["200", "201"], ["200-5555", "201-6666"])
    assert check_disjoint_speakers(dev, held) == []
    assert check_disjoint_sessions(dev, held) == []


def test_disjointness_checks_report_overlap(tmp_path):
    dev = _synthetic_manifest("dev_m", ["100"], ["100-1211"])
    held = _synthetic_manifest("held_m", ["100"], ["200-1211"])
    assert check_disjoint_speakers(dev, held)
    assert check_disjoint_sessions(dev, held) == []


def test_mixed_dev_manifest_merges_and_writes(tmp_path):
    dev = _synthetic_manifest("ls_dev", ["100"], ["100-1211"])
    real = _synthetic_manifest("ami_dev_pilot", ["ES2003a.ParticipantA"], ["ES2003a"])
    held = _synthetic_manifest("ls_held", ["200"], ["200-9999"])
    manifest, problems = build_mixed_dev_manifest(
        manifest_id="mixed_dev_pool",
        out_dir=tmp_path,
        source_manifests=[dev, real],
        held_out_manifests=[held],
        wav_roots=[tmp_path],
    )
    assert problems == []
    assert len(manifest.cases) == 2
    assert manifest.split_role == "dev_pool"
    assert (tmp_path / "manifests" / "mixed_dev_pool.json").is_file()
    assert sorted(manifest.disjointness_groups) == ["group_ami_dev_pilot", "group_ls_dev"]


def test_mixed_dev_manifest_reports_disjointness_violation(tmp_path):
    dev = _synthetic_manifest("ls_dev", ["100"], ["100-1211"])
    held = _synthetic_manifest("ls_held", ["100"], ["200-9999"])
    manifest, problems = build_mixed_dev_manifest(
        manifest_id="mixed_bad",
        out_dir=tmp_path,
        source_manifests=[dev],
        held_out_manifests=[held],
        wav_roots=[tmp_path],
    )
    assert problems
    assert "speaker overlap" in problems[0]


def _ami_manifest(manifest_id: str, meeting_id: str, global_names: list[str]) -> object:
    agents = {letter: name for letter, name in zip("ABCD", global_names)}
    case = Phase2Case(
        case_id=f"ami_{meeting_id}",
        wav_relative_path="",
        duration_samples=16000,
        wav_sha256="",
        seed=0,
        regions=[SpeakerRegion(0, 0, 16000, frozenset())],
        kind="real_meeting",
        condition={
            "corpus": "ami",
            "meeting_id": meeting_id,
            "partition_meta": {"agents": agents},
        },
        sources=[
            SourceRef(
                role="speaker_0",
                speaker=f"{meeting_id}.ParticipantA",
                session=meeting_id,
                utterance=meeting_id,
                file_sha256="",
                original_start_sample=0,
                original_end_sample=16000,
                trimmed_start_sample=0,
                trimmed_end_sample=16000,
                cut_start_sample=0,
                cut_end_sample=16000,
                gain=1.0,
            )
        ],
        active_speech_samples=0,
    )
    return make_phase2_manifest(
        manifest_id=manifest_id,
        split_role="pilot_dev" if "dev" in manifest_id else "pilot_held_out",
        corpus={"name": "ami"},
        build={"script": "test"},
        disjointness_groups=[f"ami_{manifest_id}"],
        generator={"script": "test"},
        cases=[case],
    )


def test_global_actor_ids_from_partition_meta():
    dev = _ami_manifest("ami_dev_pilot", "ES2003a", ["MEE011", "MEE009", "", "MEE010"])
    assert global_actor_ids(dev) == {"MEE009", "MEE010", "MEE011"}


def test_global_actor_disjointness_pass_and_overlap():
    dev = _ami_manifest("ami_dev_pilot", "ES2003a", ["MEE011", "MEE009", "MEE012", "MEE010"])
    held = _ami_manifest("ami_held_out_pilot", "ES2004a", ["FEE013", "FEE016", "MEE014", "MEO015"])
    assert check_disjoint_global_actors(dev, held) == []
    shared = _ami_manifest(
        "ami_held_out_pilot", "ES2004a", ["MEE011", "FEE016", "MEE014", "MEO015"]
    )
    problems = check_disjoint_global_actors(dev, shared)
    assert problems
    assert "global actor overlap" in problems[0]
    assert "MEE011" in problems[0]


def test_mixed_dev_manifest_rejects_shared_global_actor(tmp_path):
    dev_ls = _synthetic_manifest("ls_dev", ["100"], ["100-1211"])
    dev_ami = _ami_manifest("ami_dev_pilot", "ES2003a", ["MEE011", "MEE009", "MEE012", "MEE010"])
    held = _ami_manifest("ami_held_out_pilot", "ES2004a", ["MEE011", "FEE016", "MEE014", "MEO015"])
    manifest, problems = build_mixed_dev_manifest(
        manifest_id="mixed_bad_actors",
        out_dir=tmp_path,
        source_manifests=[dev_ls, dev_ami],
        held_out_manifests=[held],
        wav_roots=[tmp_path],
    )
    assert problems
    assert "global actor overlap" in problems[0]


def test_manifest_identity_evidence_canonical_and_tampered(tmp_path):
    manifest = _synthetic_manifest("ls_dev", ["100"], ["100-1211"])
    path = tmp_path / "ls_dev.json"
    manifest.write(path)
    evidence = manifest_identity_evidence(path)
    assert evidence["manifest_semantic_hash"] == manifest.hash
    assert evidence["manifest_canonical_bytes_ok"] is True
    assert len(evidence["manifest_canonical_file_sha256"]) == 64
    canonical_bytes = canonical_json(manifest.to_dict()).encode("utf-8")
    assert evidence["manifest_canonical_file_sha256"] == hashlib.sha256(canonical_bytes).hexdigest()
    tampered = path.read_text(encoding="utf-8") + "\n"
    path.write_text(tampered, encoding="utf-8")
    tampered_evidence = manifest_identity_evidence(path)
    assert tampered_evidence["manifest_semantic_hash"] == manifest.hash
    assert tampered_evidence["manifest_canonical_bytes_ok"] is False
    assert (
        tampered_evidence["manifest_canonical_file_sha256"]
        != evidence["manifest_canonical_file_sha256"]
    )
    assert json.loads(path.read_text(encoding="utf-8"))["manifest_id"] == "ls_dev"
