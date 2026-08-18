from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments.psem_training_strategy_gate.data.identity_components import (
    build_identity_graph,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    canonical_sha256,
    sha256_file,
)

AUTHORITY_REF = "https://github.com/kapitalismho/PuriPuly-heart/issues/77"
AUTHORITY_PIN = "5778025c8aca1ea1cb7cd8fc41645b520ca1f9f749155b5f5daada32e940b559"
MODEL_ID = "wavlm-base-plus"
MODEL_REPOSITORY = "https://huggingface.co/microsoft/wavlm-base-plus"
MODEL_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
MODEL_ARTIFACT = {
    "file_name": "pytorch_model.bin",
    "sha256": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
    "size_bytes": 377617425,
}
PRETRAINING_CORPORA = (
    {"corpus": "Libri-Light", "hours": 60000},
    {"corpus": "GigaSpeech", "hours": 10000},
    {"corpus": "VoxPopuli", "hours": 24000},
)
PROVENANCE_EVIDENCE = (
    {
        "kind": "pinned_checkpoint_model_card",
        "ref": "https://huggingface.co/microsoft/wavlm-base-plus/raw/4c66d4806a428f2e922ccfa1a962776e232d487b/README.md",
        "sha256": "d417fc16b08e2f66f0af6b371e47a5f9f1f596f6c1c593f1214d360c41d7a27e",
    },
    {
        "kind": "official_wavlm_repository",
        "ref": "https://raw.githubusercontent.com/microsoft/unilm/1ad6ea07df17b3e5c4c5d851f98721ecc43cf593/wavlm/README.md",
        "sha256": "b0a10869ea8d79c49c5d0efb026724bdef9abea3ee4a1604b8b43b2581c0e635",
    },
    {
        "kind": "official_wavlm_paper",
        "ref": "https://arxiv.org/pdf/2110.13900v5",
        "sha256": "0ca8836ebdf8236e610187738217d4c91c5ead13873472e476423f1561e9238e",
        "corpus_statement": "94k hours total",
    },
)
CLASSIFICATIONS = (
    "exact_session_overlap_known",
    "corpus_level_overlap_known",
    "no_known_overlap",
    "unknown",
)
NO_MODEL_FIELDS = (
    "model_predictions_consulted",
    "model_scores_consulted",
    "official_model_results_inspected",
    "official_model_training_performed",
)


class PretrainingOverlapError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PretrainingOverlapError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PretrainingOverlapError(f"JSON artifact must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise PretrainingOverlapError(f"invalid JSONL artifact: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise PretrainingOverlapError(f"JSONL artifact must contain objects: {path}")
    return rows


def _model_entry(registry: dict[str, Any]) -> dict[str, Any]:
    models = registry.get("models")
    if not isinstance(models, list):
        raise PretrainingOverlapError("model registry has no model inventory")
    matches = [row for row in models if row.get("model_id") == MODEL_ID]
    if len(matches) != 1:
        raise PretrainingOverlapError("pinned WavLM model entry is not unique")
    return matches[0]


def _validate_checkpoint(
    registry: dict[str, Any],
    source_registry: dict[str, Any],
    registry_sha256: str,
) -> None:
    model = _model_entry(registry)
    source_model = _model_entry(source_registry)
    artifact = model.get("artifact")
    required_files = source_model.get("required_files")
    artifact_matches = (
        isinstance(artifact, dict)
        and artifact.get("file_name") == MODEL_ARTIFACT["file_name"]
        and artifact.get("sha256") == MODEL_ARTIFACT["sha256"]
        and artifact.get("size_bytes") == MODEL_ARTIFACT["size_bytes"]
    )
    required_artifacts = (
        [
            row
            for row in required_files
            if isinstance(row, dict) and row.get("path") == MODEL_ARTIFACT["file_name"]
        ]
        if isinstance(required_files, list)
        else []
    )
    if (
        model.get("repository") != MODEL_REPOSITORY
        or model.get("revision") != MODEL_REVISION
        or not artifact_matches
        or source_model.get("repository") != MODEL_REPOSITORY
        or source_model.get("revision") != MODEL_REVISION
        or len(required_artifacts) != 1
        or required_artifacts[0].get("sha256") != MODEL_ARTIFACT["sha256"]
        or required_artifacts[0].get("size_bytes") != MODEL_ARTIFACT["size_bytes"]
        or source_registry.get("r0_registry", {}).get("sha256") != registry_sha256
    ):
        raise PretrainingOverlapError("branch-pinned WavLM checkpoint binding mismatch")


def _classify_overlap(
    corpus: str,
    session_id: str,
    *,
    exact_session_overlaps: Iterable[tuple[str, str]] = (),
    provenance_complete: bool = True,
) -> dict[str, Any]:
    exact = set(exact_session_overlaps)
    documented_corpora = {row["corpus"] for row in PRETRAINING_CORPORA}
    if (corpus, session_id) in exact:
        classification = "exact_session_overlap_known"
        basis = "exact corpus and session identity appears in checkpoint provenance"
    elif not provenance_complete:
        classification = "unknown"
        basis = "checkpoint pretraining provenance is incomplete"
    elif corpus in documented_corpora:
        classification = "corpus_level_overlap_known"
        basis = "candidate corpus appears in the documented checkpoint pretraining corpora"
    else:
        classification = "no_known_overlap"
        basis = "candidate corpus is absent from the documented checkpoint pretraining corpora"
    return {
        "classification": classification,
        "basis": basis,
        "eval_forbidden_by_pretraining_overlap": classification == "exact_session_overlap_known",
        "pretraining_clean_claimed": False,
    }


def build_pretraining_overlap_audit(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
    identity_graph_path: Path | None = None,
) -> dict[str, Any]:
    registry = _load_json(registry_path)
    source_registry = _load_json(source_registry_path)
    registry_sha256 = sha256_file(registry_path)
    _validate_checkpoint(registry, source_registry, registry_sha256)
    source_rows = _load_jsonl(data_dir / "source_manifest.jsonl")
    source_by_id = {row.get("source_id"): row for row in source_rows}
    graph_path = identity_graph_path or data_dir / "identity_components.json"
    checked_graph = _load_json(graph_path)
    rebuilt_graph = build_identity_graph(data_dir)
    if checked_graph != rebuilt_graph:
        raise PretrainingOverlapError("checked-in identity graph is not current")
    graph_nodes = checked_graph.get("nodes")
    if not isinstance(graph_nodes, list):
        raise PretrainingOverlapError("identity graph node inventory is invalid")
    node_by_source = {row.get("source_id"): row for row in graph_nodes}
    if (
        len(source_by_id) != len(source_rows)
        or any(not isinstance(source_id, str) or not source_id for source_id in source_by_id)
        or len(node_by_source) != len(graph_nodes)
        or set(node_by_source) != set(source_by_id)
    ):
        raise PretrainingOverlapError("candidate source and identity graph coverage mismatch")
    audit_rows = []
    for source_id, source in sorted(source_by_id.items()):
        corpus = source.get("corpus")
        session_id = source.get("session_id")
        waveform_sha256 = source.get("waveform_sha256")
        if not all(
            isinstance(value, str) and value for value in (corpus, session_id, waveform_sha256)
        ):
            raise PretrainingOverlapError("candidate source identity is incomplete")
        classification = _classify_overlap(corpus, session_id)
        audit_rows.append(
            {
                "source_id": source_id,
                "corpus": corpus,
                "session_id": session_id,
                "waveform_sha256": waveform_sha256,
                "identity_component_id": node_by_source[source_id]["component_id"],
                "identity_node_sha256": canonical_sha256(node_by_source[source_id]),
                **classification,
            }
        )
    counts = Counter(row["classification"] for row in audit_rows)
    return {
        "schema_version": 1,
        "artifact_role": "wavlm_pretraining_overlap_audit",
        "authority_ref": AUTHORITY_REF,
        "authority_pin": AUTHORITY_PIN,
        "audit_status": "complete_for_documented_checkpoint_provenance",
        "checkpoint": {
            "model_id": MODEL_ID,
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "artifact": MODEL_ARTIFACT,
        },
        "official_pretraining_provenance": {
            "documented_corpora": list(PRETRAINING_CORPORA),
            "documented_total_hours": sum(row["hours"] for row in PRETRAINING_CORPORA),
            "corpus_statement_scope": "exhaustive checkpoint model-card and official-repository statement",
            "exact_session_inventory_status": "not_published_in_audited_checkpoint_sources",
            "evidence": list(PROVENANCE_EVIDENCE),
        },
        "input_artifacts": {
            "model_registry_sha256": registry_sha256,
            "model_source_registry_sha256": sha256_file(source_registry_path),
            "source_manifest_sha256": sha256_file(data_dir / "source_manifest.jsonl"),
            "identity_components_sha256": sha256_file(graph_path),
            "source_ids_sha256": canonical_sha256(sorted(source_by_id)),
        },
        "classification_policy": {
            "allowed_values": list(CLASSIFICATIONS),
            "exact_session_overlap_eval_forbidden": True,
            "corpus_level_overlap_without_exact_session_proof_eval_forbidden": False,
            "unknown_reported_without_clean_claim": True,
            "checkpoint_change_permitted": False,
        },
        "summary": {
            "source_count": len(audit_rows),
            "identity_component_count": checked_graph["summary"]["component_count"],
            "classification_counts": {
                classification: counts[classification] for classification in CLASSIFICATIONS
            },
            "eval_forbidden_source_count": sum(
                row["eval_forbidden_by_pretraining_overlap"] for row in audit_rows
            ),
            "pretraining_clean_claim_count": sum(
                row["pretraining_clean_claimed"] for row in audit_rows
            ),
        },
        "model_policy": {field: False for field in NO_MODEL_FIELDS},
        "sources": audit_rows,
    }


def write_pretraining_overlap_audit(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
    output_path: Path,
) -> None:
    audit = build_pretraining_overlap_audit(data_dir, registry_path, source_registry_path)
    output_path.write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_pretraining_overlap_audit(
        args.data_dir.resolve(),
        args.registry.resolve(),
        args.source_registry.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
