from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.psem_training_strategy_gate.data.label_contract import (
    LabelContract,
    LabelContractError,
    load_contract,
)

ISSUE_77_REF = "https://github.com/kapitalismho/PuriPuly-heart/issues/77"
ISSUE_77_PIN = "5778025c8aca1ea1cb7cd8fc41645b520ca1f9f749155b5f5daada32e940b559"
ISSUE_86_REF = "https://github.com/kapitalismho/PuriPuly-heart/issues/86"
ISSUE_86_PIN = "90078d66026f1374b065a5b9022788c40fac076cd4cf307df87b5027ea3fcb63"


class DatasetContextError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetContext:
    data_dir: Path
    calibration_dir: Path
    source_contract: LabelContract
    label_contract: LabelContract
    authority_ref: str
    authority_pin: str
    freeze_id: str

    @property
    def is_v2(self) -> bool:
        return self.label_contract.contract_version == "psem-handoff-v1"


def _contract_version(path: Path, *, default: str | None = None) -> str:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetContextError(f"invalid dataset manifest: {path}") from exc
    versions = {row.get("contract_version") for row in rows if isinstance(row, dict)}
    if not rows or len(versions) != 1 or not all(isinstance(row, dict) for row in rows):
        raise DatasetContextError(f"dataset manifest contract is inconsistent: {path}")
    version = versions.pop()
    if version is None and default is not None:
        return default
    if not isinstance(version, str):
        raise DatasetContextError(f"dataset manifest contract is invalid: {path}")
    return version


def resolve_dataset_context(data_dir: Path) -> DatasetContext:
    resolved = data_dir.resolve()
    try:
        source_contract = load_contract(
            version=_contract_version(resolved / "source_manifest.jsonl", default="psem-handoff-v0")
        )
        label_contract = load_contract(
            version=_contract_version(
                resolved / "normalization_manifest.jsonl", default="psem-handoff-v0"
            )
        )
    except LabelContractError as exc:
        raise DatasetContextError("dataset contract version is unsupported") from exc
    combination = (
        source_contract.contract_version,
        label_contract.contract_version,
    )
    if combination not in {
        ("psem-handoff-v0", "psem-handoff-v0"),
        ("psem-handoff-v0", "psem-handoff-v1"),
    }:
        raise DatasetContextError(f"unsupported source/label contract combination: {combination}")
    calibration_dir = (
        resolved if (resolved / "annotation_calibration.json").is_file() else resolved.parent
    )
    if (
        not (calibration_dir / "annotation_calibration.json").is_file()
        or not (calibration_dir / "ANNOTATION_CALIBRATION.md").is_file()
    ):
        raise DatasetContextError("accepted annotation calibration is unavailable")
    if label_contract.contract_version == "psem-handoff-v1":
        return DatasetContext(
            resolved,
            calibration_dir,
            source_contract,
            label_contract,
            ISSUE_86_REF,
            ISSUE_86_PIN,
            "PSEM-STRATEGY-DATA-v2",
        )
    return DatasetContext(
        resolved,
        calibration_dir,
        source_contract,
        label_contract,
        ISSUE_77_REF,
        ISSUE_77_PIN,
        "PSEM-STRATEGY-DATA-v1",
    )
