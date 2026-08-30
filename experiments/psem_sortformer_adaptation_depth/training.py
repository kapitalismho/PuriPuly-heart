from __future__ import annotations

import hashlib
import math
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from experiments.psem_sortformer_adaptation_depth.models import composite_loss
from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
    SortformerEvidence,
    TrainableSortformerPSEM,
)
from experiments.psem_sortformer_adaptation_depth.preflight import (
    canonical_sha256,
    require_material_execution_ready,
    sha256_file,
)
from experiments.psem_sortformer_adaptation_depth.runtime_audit import (
    GRADIENT_CLIP_NORM,
    build_optimizer,
    canary_bundle_runtime_passed,
)
from experiments.psem_sortformer_adaptation_depth.sampling import (
    MAXIMUM_EPOCHS as MANIFEST_EPOCHS,
)
from experiments.psem_sortformer_adaptation_depth.sampling import (
    WINDOWS_PER_EPOCH,
    load_sampling_rows,
    load_window_waveform,
    select_overfit_rows,
    validate_sampling_manifest,
)
from experiments.psem_sortformer_adaptation_depth.supervision import (
    FRAME_COUNT,
    FrameSupervision,
    anchor_timeline,
    build_frame_supervision,
    oracle_mapping_from_frames,
)
from experiments.psem_training_strategy_gate.sampling import DEV_ROLE, TRAIN_ROLE, RuntimeSession

MAXIMUM_EPOCHS = 1
WARMUP_STEPS = 13
SMOKE_OPTIMIZER_STEPS = 32
OFFICIAL_OPTIMIZER_STEPS = 256
EARLY_STOPPING_PATIENCE = 2
OVERFIT_MAXIMUM_STEPS = 500
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
OPTIMIZER_STEPS_PER_EPOCH = OFFICIAL_OPTIMIZER_STEPS
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class TrainingContractError(RuntimeError):
    pass


class _PreparedTrainingExampleToken:
    def __reduce__(self) -> tuple[Any, tuple[()]]:
        return _prepared_training_example_token, ()


def _prepared_training_example_token() -> _PreparedTrainingExampleToken:
    return _TRAINING_EXAMPLE_TOKEN


_TRAINING_EXAMPLE_TOKEN = _PreparedTrainingExampleToken()


@dataclass(frozen=True, slots=True)
class TrainingExample:
    source_id: str
    corpus: str
    row_id: str
    waveform: torch.Tensor
    supervision: FrameSupervision
    split_role: str = TRAIN_ROLE
    epoch: int = 0
    epoch_index: int = 0
    sampling_manifest_sha256: str = ""
    target_identity_sha256: str = ""
    augmentation_identity_sha256: str = ""
    state_reset_at_start: bool = True
    window_start_sample: int = 0
    window_end_sample: int = 480000
    waveform_content_sha256: str = ""
    supervision_content_sha256: str = ""
    _factory_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ClassWeights:
    replacement_positive: float
    anchor_positive: float


@dataclass(frozen=True, slots=True)
class ForwardResult:
    losses: dict[str, Any]
    replacement_logits: torch.Tensor
    anchor_logits: torch.Tensor
    replacement_targets: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class OfficialTrainingAuthorization:
    arm: str
    seed: int
    git_head: str
    material_gate_sha256: str
    sampling_manifest_sha256: str
    class_weight_receipt_sha256: str
    dev_source_ids_sha256: str
    row_ids_by_epoch: tuple[tuple[str, ...], ...]
    input_identity_by_row: tuple[tuple[str, str, str, int, int, str, str, bool], ...]
    class_weights: ClassWeights


@dataclass(frozen=True, slots=True)
class _LegacyOverfitAuthorization:
    arm: str
    sampling_manifest_sha256: str
    selected_input_identity_sha256: str
    selected_row_ids: tuple[str, ...]
    input_identity_by_row: tuple[tuple[str, str, str, int, int, str, str, bool], ...]
    corpus_by_source: tuple[tuple[str, str], ...]
    class_weights: ClassWeights


def _tensor_content_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _supervision_content_sha256(value: FrameSupervision) -> str:
    return canonical_sha256(
        {
            "anchor_targets": _tensor_content_sha256(value.anchor_targets),
            "replacement_targets": _tensor_content_sha256(value.replacement_targets),
            "psem_mask": _tensor_content_sha256(value.psem_mask),
            "arrival_order_targets": _tensor_content_sha256(value.arrival_order_targets),
            "native_mask": _tensor_content_sha256(value.native_mask),
            "arrival_order_speakers": value.arrival_order_speakers,
            "mapping_anchor_active": _tensor_content_sha256(value.mapping_anchor_active),
            "anchor_episode_ids": value.anchor_episode_ids,
        }
    )


def _training_example_content_bound(example: TrainingExample) -> bool:
    return bool(
        example._factory_token is _TRAINING_EXAMPLE_TOKEN
        and example.waveform_content_sha256 == _tensor_content_sha256(example.waveform)
        and example.supervision_content_sha256 == _supervision_content_sha256(example.supervision)
    )


def prepare_training_example(
    row: Mapping[str, Any],
    session: RuntimeSession,
    corpus_root: Path,
    corpus: str,
    *,
    manifest_path: Path,
    manifest_validation: Mapping[str, Any],
    manifest_rows_by_id: Mapping[str, Mapping[str, Any]],
) -> TrainingExample:
    manifest_sha256 = sha256_file(manifest_path)
    row_id_value = row.get("row_id")
    if (
        manifest_validation.get("passed") is not True
        or manifest_validation.get("manifest_sha256") != manifest_sha256
        or not isinstance(row_id_value, str)
        or manifest_rows_by_id.get(row_id_value) != row
    ):
        raise TrainingContractError("training row is not bound to the validated shared manifest")
    waveform = load_window_waveform(row, session, corpus_root)
    start = row.get("window_start_sample")
    row_id = row.get("row_id")
    if (
        not isinstance(start, int)
        or not isinstance(row_id, str)
        or corpus
        not in {
            "AMI",
            "AliMeeting",
        }
        or row.get("corpus") != corpus
    ):
        raise TrainingContractError("training example metadata is invalid")
    episode_ids, anchor_speakers = anchor_timeline(session.source_id, session.labels, start)
    supervision = build_frame_supervision(
        session.labels,
        start,
        episode_ids,
        anchor_speakers,
    )
    return TrainingExample(
        source_id=session.source_id,
        corpus=corpus,
        row_id=row_id,
        waveform=waveform,
        supervision=supervision,
        split_role=TRAIN_ROLE,
        epoch=int(row["epoch"]),
        epoch_index=int(row["epoch_index"]),
        sampling_manifest_sha256=manifest_sha256,
        target_identity_sha256=str(row["target_identity_sha256"]),
        augmentation_identity_sha256=str(row["augmentation_identity_sha256"]),
        state_reset_at_start=row.get("state_reset_at_window_start") is True,
        window_start_sample=start,
        window_end_sample=int(row["window_end_sample"]),
        waveform_content_sha256=_tensor_content_sha256(waveform),
        supervision_content_sha256=_supervision_content_sha256(supervision),
        _factory_token=_TRAINING_EXAMPLE_TOKEN,
    )


def prepare_dev_example(
    *,
    source_id: str,
    corpus: str,
    window_start_sample: int,
    waveform: torch.Tensor,
    labels: Any,
) -> TrainingExample:
    if (
        not source_id
        or corpus not in {"AMI", "AliMeeting"}
        or window_start_sample < 0
        or window_start_sample % 480000
        or waveform.shape != (480000,)
        or not torch.is_floating_point(waveform)
        or not bool(torch.isfinite(waveform).all())
    ):
        raise TrainingContractError("DEV sequence geometry differs from the frozen policy")
    episode_ids, anchor_speakers = anchor_timeline(source_id, labels, window_start_sample)
    supervision = build_frame_supervision(
        labels,
        window_start_sample,
        episode_ids,
        anchor_speakers,
    )
    row_id = f"dev-{source_id}-{window_start_sample:012d}"
    return TrainingExample(
        source_id=source_id,
        corpus=corpus,
        row_id=row_id,
        waveform=waveform,
        supervision=supervision,
        split_role=DEV_ROLE,
        state_reset_at_start=True,
        window_start_sample=window_start_sample,
        window_end_sample=window_start_sample + 480000,
        waveform_content_sha256=_tensor_content_sha256(waveform),
        supervision_content_sha256=_supervision_content_sha256(supervision),
        _factory_token=_TRAINING_EXAMPLE_TOKEN,
    )


def derive_train_class_weights(examples: Sequence[TrainingExample]) -> ClassWeights:
    if not examples:
        raise TrainingContractError("TRAIN class weighting requires examples")
    counts = {
        "replacement": Counter(),
        "anchor": Counter(),
    }
    for example in examples:
        active = example.supervision.psem_mask.bool()
        counts["replacement"].update(
            int(value) for value in example.supervision.replacement_targets[active]
        )
        counts["anchor"].update(int(value) for value in example.supervision.anchor_targets[active])
    for values in counts.values():
        if values[0] <= 0 or values[1] <= 0:
            raise TrainingContractError("TRAIN class weighting requires both target classes")
    return ClassWeights(
        replacement_positive=counts["replacement"][0] / counts["replacement"][1],
        anchor_positive=counts["anchor"][0] / counts["anchor"][1],
    )


def build_manifest_class_weight_receipt(
    rows: Sequence[Mapping[str, Any]],
    sessions: Mapping[str, RuntimeSession],
    manifest_path: Path,
) -> dict[str, Any]:
    validation = validate_sampling_manifest(manifest_path, sessions)
    persisted_rows = load_sampling_rows(manifest_path)
    if not rows or any(row.get("split_role") != TRAIN_ROLE for row in rows):
        raise TrainingContractError("class weights require the shared TRAIN-only manifest")
    if list(rows) != persisted_rows or len({row.get("row_id") for row in rows}) != len(rows):
        raise TrainingContractError("class-weight rows differ from the exact shared manifest")
    cache: dict[tuple[str, int], FrameSupervision] = {}
    replacement = Counter()
    anchor = Counter()
    for row in rows:
        source_id = str(row.get("source_id"))
        start = row.get("window_start_sample")
        session = sessions.get(source_id)
        if session is None or session.role != TRAIN_ROLE or not isinstance(start, int):
            raise TrainingContractError("class-weight row is outside the bound TRAIN split")
        key = (source_id, start)
        supervision = cache.get(key)
        if supervision is None:
            episode_ids, speakers = anchor_timeline(source_id, session.labels, start)
            supervision = build_frame_supervision(session.labels, start, episode_ids, speakers)
            cache[key] = supervision
        active = supervision.psem_mask.bool()
        replacement.update(int(value) for value in supervision.replacement_targets[active])
        anchor.update(int(value) for value in supervision.anchor_targets[active])
    if any(values[0] <= 0 or values[1] <= 0 for values in (replacement, anchor)):
        raise TrainingContractError("manifest class weights require both classes")
    weights = ClassWeights(
        replacement_positive=replacement[0] / replacement[1],
        anchor_positive=anchor[0] / anchor[1],
    )
    payload = {
        "schema_version": 1,
        "artifact_role": "train_class_weight_receipt",
        "split_roles": [TRAIN_ROLE],
        "eval_source_count": 0,
        "sampling_manifest_sha256": sha256_file(manifest_path),
        "sampling_validation_sha256": canonical_sha256(validation),
        "row_count": len(rows),
        "unique_window_count": len(cache),
        "replacement_counts": dict(sorted(replacement.items())),
        "anchor_counts": dict(sorted(anchor.items())),
        "replacement_positive_weight": weights.replacement_positive,
        "anchor_positive_weight": weights.anchor_positive,
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def authorize_official_training(
    gate_receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    class_weight_receipt: Mapping[str, Any],
) -> OfficialTrainingAuthorization:
    if (
        gate_receipt.get("schema_version") != 1
        or gate_receipt.get("artifact_role") != "material_training_authorization"
        or gate_receipt.get("passed") is not True
    ):
        raise TrainingContractError("material training gate is absent or invalid")
    gate_payload = {key: value for key, value in gate_receipt.items() if key != "payload_sha256"}
    if gate_receipt.get("payload_sha256") != canonical_sha256(gate_payload):
        raise TrainingContractError("material training authorization payload is not bound")
    arm = gate_receipt.get("arm")
    seed = gate_receipt.get("seed")
    manifest_sha = gate_receipt.get("sampling_manifest_sha256")
    weight_sha = gate_receipt.get("class_weight_receipt_sha256")
    git_head = gate_receipt.get("git_head")
    weight_payload = {
        key: value for key, value in class_weight_receipt.items() if key != "payload_sha256"
    }
    if (
        arm not in {"H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"}
        or seed != 7301
        or not isinstance(manifest_sha, str)
        or not isinstance(git_head, str)
        or len(git_head) != 40
        or any(value not in "0123456789abcdef" for value in git_head)
        or class_weight_receipt.get("payload_sha256") != weight_sha
        or class_weight_receipt.get("payload_sha256") != canonical_sha256(weight_payload)
        or class_weight_receipt.get("sampling_manifest_sha256") != manifest_sha
    ):
        raise TrainingContractError(
            "official arm, seed, manifest, or class weights are unauthorized"
        )
    by_epoch: list[list[str]] = [[] for _ in range(1)]
    seen: set[str] = set()
    for row in rows:
        row_id = row.get("row_id")
        epoch = row.get("epoch")
        epoch_index = row.get("epoch_index")
        if (
            not isinstance(row_id, str)
            or row_id in seen
            or not isinstance(epoch, int)
            or not 1 <= epoch <= MANIFEST_EPOCHS
            or epoch_index != len(by_epoch[epoch - 1])
            or row.get("split_role") != TRAIN_ROLE
        ):
            raise TrainingContractError("official manifest row order or identity is invalid")
        seen.add(row_id)
        by_epoch[epoch - 1].append(row_id)
    if any(len(values) != WINDOWS_PER_EPOCH for values in by_epoch):
        raise TrainingContractError("official manifest does not contain 4096 windows per epoch")
    expected_manifest_identity = canonical_sha256(
        [
            {
                key: row.get(key)
                for key in (
                    "row_id",
                    "source_id",
                    "corpus",
                    "window_start_sample",
                    "window_end_sample",
                    "target_identity_sha256",
                    "augmentation_identity_sha256",
                    "state_reset_at_window_start",
                )
            }
            for row in rows
        ]
    )
    if gate_receipt.get("shared_input_identity_sha256") != expected_manifest_identity:
        raise TrainingContractError(
            "material gate is not bound to shared targets and augmentations"
        )
    weights = ClassWeights(
        replacement_positive=float(class_weight_receipt["replacement_positive_weight"]),
        anchor_positive=float(class_weight_receipt["anchor_positive_weight"]),
    )
    return OfficialTrainingAuthorization(
        arm=str(arm),
        seed=int(seed),
        git_head=git_head,
        material_gate_sha256=str(gate_receipt["payload_sha256"]),
        sampling_manifest_sha256=manifest_sha,
        class_weight_receipt_sha256=str(weight_sha),
        dev_source_ids_sha256=str(gate_receipt["dev_source_ids_sha256"]),
        row_ids_by_epoch=tuple(tuple(values) for values in by_epoch),
        input_identity_by_row=tuple(
            (
                str(row["row_id"]),
                str(row["source_id"]),
                str(row["corpus"]),
                int(row["window_start_sample"]),
                int(row["window_end_sample"]),
                str(row["target_identity_sha256"]),
                str(row["augmentation_identity_sha256"]),
                row["state_reset_at_window_start"] is True,
            )
            for row in rows
        ),
        class_weights=weights,
    )


def _legacy_authorize_overfit_arm(
    arm: str,
    selected_rows: Sequence[Mapping[str, Any]],
    sampling_rows: Sequence[Mapping[str, Any]],
    sampling_manifest_path: Path,
    corpus_by_source: Mapping[str, str],
    class_weight_receipt: Mapping[str, Any],
) -> _LegacyOverfitAuthorization:
    if arm not in {"H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"}:
        raise TrainingContractError("overfit arm is unauthorized")
    if list(sampling_rows) != load_sampling_rows(sampling_manifest_path):
        raise TrainingContractError("overfit sampling rows differ from the persisted manifest")
    canonical_rows = select_overfit_rows(sampling_rows, corpus_by_source)
    manifest_sha = sha256_file(sampling_manifest_path)
    payload = {key: value for key, value in class_weight_receipt.items() if key != "payload_sha256"}
    if (
        list(selected_rows) != list(canonical_rows)
        or class_weight_receipt.get("artifact_role") != "train_class_weight_receipt"
        or class_weight_receipt.get("payload_sha256") != canonical_sha256(payload)
        or class_weight_receipt.get("sampling_manifest_sha256") != manifest_sha
    ):
        raise TrainingContractError("overfit authorization is not bound to manifest class weights")
    identities = tuple(
        (
            str(row["row_id"]),
            str(row["source_id"]),
            str(row["corpus"]),
            int(row["window_start_sample"]),
            int(row["window_end_sample"]),
            str(row["target_identity_sha256"]),
            str(row["augmentation_identity_sha256"]),
            row["state_reset_at_window_start"] is True,
        )
        for row in selected_rows
    )
    if len(identities) != 60 or len({value[0] for value in identities}) != 60:
        raise TrainingContractError("overfit authorization does not contain 60 unique windows")
    return _LegacyOverfitAuthorization(
        arm=arm,
        sampling_manifest_sha256=manifest_sha,
        selected_input_identity_sha256=canonical_sha256(list(selected_rows)),
        selected_row_ids=tuple(value[0] for value in identities),
        input_identity_by_row=identities,
        corpus_by_source=tuple(
            sorted({(str(row["source_id"]), str(row["corpus"])) for row in selected_rows})
        ),
        class_weights=ClassWeights(
            replacement_positive=float(class_weight_receipt["replacement_positive_weight"]),
            anchor_positive=float(class_weight_receipt["anchor_positive_weight"]),
        ),
    )


def _batch_supervision(
    examples: Sequence[TrainingExample],
    evidence: SortformerEvidence,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if evidence.probabilities.shape[:2] != (len(examples), FRAME_COUNT):
        raise TrainingContractError("Sortformer output differs from the native 30-second grid")
    anchor_one_hot = []
    for batch_index, example in enumerate(examples):
        episode_ids = {
            value for value in example.supervision.anchor_episode_ids if value is not None
        }
        mapping = (
            oracle_mapping_from_frames(
                evidence.probabilities[batch_index].detach(),
                evidence.slot_alive[batch_index].detach(),
                example.supervision,
            )
            if episode_ids
            else {}
        )
        if not episode_ids and bool(example.supervision.psem_mask.any()):
            raise TrainingContractError("anchor-free windows must not carry PSEM supervision")
        slots = [
            mapping.get(episode_id, 0) if episode_id is not None else 0
            for episode_id in example.supervision.anchor_episode_ids
        ]
        encoded = torch.zeros(
            (FRAME_COUNT, 4),
            dtype=evidence.final_temporal_hidden.dtype,
            device=evidence.final_temporal_hidden.device,
        )
        encoded[
            torch.arange(FRAME_COUNT, device=encoded.device),
            torch.tensor(slots, device=encoded.device),
        ] = 1
        anchor_one_hot.append(encoded)
    device = evidence.probabilities.device
    return (
        torch.stack(anchor_one_hot),
        torch.stack([value.supervision.anchor_targets for value in examples]).to(device),
        torch.stack([value.supervision.replacement_targets for value in examples]).to(device),
        torch.stack([value.supervision.psem_mask for value in examples]).to(device),
        torch.stack([value.supervision.arrival_order_targets for value in examples]).to(device),
    )


def forward_batch(
    model: TrainableSortformerPSEM,
    examples: Sequence[TrainingExample],
    class_weights: ClassWeights,
) -> ForwardResult:
    if not examples:
        raise TrainingContractError("training batch must not be empty")
    waveform = torch.stack([example.waveform for example in examples]).to(model.sortformer.device)
    lengths = torch.full(
        (len(examples),),
        waveform.shape[1],
        dtype=torch.long,
        device=waveform.device,
    )
    state_reset = torch.zeros(
        (len(examples), FRAME_COUNT, 1),
        dtype=torch.bool,
        device=waveform.device,
    )
    state_reset[:, 0, 0] = torch.tensor(
        [example.state_reset_at_start for example in examples],
        dtype=torch.bool,
        device=waveform.device,
    )
    evidence = model.sortformer_evidence(waveform, lengths, state_reset=state_reset)
    anchor_one_hot, anchor_targets, replacement_targets, mask, arrival_targets = _batch_supervision(
        examples, evidence
    )
    native_mask = torch.stack([value.supervision.native_mask for value in examples]).to(
        waveform.device
    )
    outputs = model.psem_outputs(evidence, anchor_one_hot)
    roles = tuple(example.split_role for example in examples)
    if len(set(roles)) != 1 or roles[0] not in {TRAIN_ROLE, DEV_ROLE}:
        raise TrainingContractError("loss batches must contain one homogeneous TRAIN or DEV role")
    native = model.native_sortformer_loss(
        evidence,
        arrival_targets,
        native_mask,
        roles,
    )
    losses = composite_loss(
        outputs,
        replacement_targets=replacement_targets,
        anchor_targets=anchor_targets,
        mask=mask,
        replacement_positive_weight=class_weights.replacement_positive,
        anchor_positive_weight=class_weights.anchor_positive,
        sampling_roles=roles,
        native_sortformer_loss=native,
    )
    return ForwardResult(
        losses=losses,
        replacement_logits=outputs["replacement_evidence"],
        anchor_logits=outputs["anchor_present"],
        replacement_targets=replacement_targets,
        mask=mask,
    )


def duration_weighted_average_precision(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> float:
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise TrainingContractError("average-precision tensors must have identical shapes")
    active = mask.bool().flatten()
    scores = torch.sigmoid(logits.detach()).flatten()[active]
    labels = targets.detach().flatten()[active].to(torch.bool)
    positive_count = int(labels.sum())
    if positive_count <= 0:
        raise TrainingContractError("average precision requires positive replacement frames")
    order = torch.argsort(scores, descending=True, stable=True)
    ordered = labels[order].to(torch.float64)
    precision = torch.cumsum(ordered, dim=0) / torch.arange(
        1, ordered.numel() + 1, dtype=torch.float64, device=ordered.device
    )
    return float((precision * ordered).sum().cpu() / positive_count)


def warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    resolved_warmup_steps = math.ceil(total_steps * 0.05) if warmup_steps is None else warmup_steps
    if total_steps <= 0 or resolved_warmup_steps <= 0 or resolved_warmup_steps > total_steps:
        raise TrainingContractError("optimizer step budget or warmup is invalid")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / resolved_warmup_steps, 1.0),
    )


class _LegacyEarlyStopping:
    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE) -> None:
        if patience != EARLY_STOPPING_PATIENCE:
            raise TrainingContractError("early-stopping patience differs from the frozen recipe")
        self.patience = patience
        self.best_key: tuple[float, float] | None = None
        self.bad_evaluations = 0
        self.improved = False

    def update(self, total_loss: float, replacement_average_precision: float) -> bool:
        if not math.isfinite(total_loss) or not math.isfinite(replacement_average_precision):
            raise TrainingContractError("DEV checkpoint metrics must be finite")
        key = (total_loss, -replacement_average_precision)
        if self.best_key is None or key < self.best_key:
            self.best_key = key
            self.bad_evaluations = 0
            self.improved = True
            return False
        self.improved = False
        self.bad_evaluations += 1
        return self.bad_evaluations >= self.patience


def _legacy_fit_arm(
    model: TrainableSortformerPSEM,
    arm: str,
    class_weights: ClassWeights,
    train_batches: Callable[[int], Iterable[Sequence[TrainingExample]]],
    steps_per_epoch: int,
    dev_evaluate: Callable[[TrainableSortformerPSEM, int], Mapping[str, float]],
    checkpoint: Callable[[TrainableSortformerPSEM, int, Mapping[str, float]], None],
    *,
    authorization: OfficialTrainingAuthorization,
) -> dict[str, Any]:
    require_material_execution_ready()
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if (
        steps_per_epoch != OPTIMIZER_STEPS_PER_EPOCH
        or arm != authorization.arm
        or class_weights != authorization.class_weights
        or current_head != authorization.git_head
        or current_dirty
    ):
        raise TrainingContractError("official training requires optimizer steps")
    optimizer = build_optimizer(model, arm)
    scheduler = warmup_scheduler(optimizer, MAXIMUM_EPOCHS * steps_per_epoch)
    stopper = _LegacyEarlyStopping()
    history = []
    global_step = 0
    selected_epoch: int | None = None
    selected_metrics: dict[str, float] | None = None
    identity_by_row = {
        row_id: tuple(values) for row_id, *values in authorization.input_identity_by_row
    }
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        model.train()
        epoch_steps = 0
        accumulated_micro_batches = 0
        observed_row_ids: list[str] = []
        optimizer.zero_grad(set_to_none=True)
        for examples in train_batches(epoch):
            if not examples or len(examples) != MICRO_BATCH_SIZE:
                raise TrainingContractError("official training produced an empty batch")
            for example in examples:
                if (
                    example.split_role != TRAIN_ROLE
                    or example.epoch != epoch
                    or example.epoch_index != len(observed_row_ids)
                    or example.sampling_manifest_sha256 != authorization.sampling_manifest_sha256
                    or not example.target_identity_sha256
                    or not example.augmentation_identity_sha256
                    or not _training_example_content_bound(example)
                    or identity_by_row.get(example.row_id)
                    != (
                        example.source_id,
                        example.corpus,
                        example.window_start_sample,
                        example.window_end_sample,
                        example.target_identity_sha256,
                        example.augmentation_identity_sha256,
                        example.state_reset_at_start,
                    )
                ):
                    raise TrainingContractError(
                        "training batch differs from the authorized shared inputs"
                    )
                observed_row_ids.append(example.row_id)
            result = forward_batch(model, examples, class_weights)
            (result.losses["total"] / GRADIENT_ACCUMULATION_STEPS).backward()
            accumulated_micro_batches += 1
            if accumulated_micro_batches == GRADIENT_ACCUMULATION_STEPS:
                norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    GRADIENT_CLIP_NORM,
                )
                if not bool(torch.isfinite(norm)):
                    raise TrainingContractError(
                        "official training produced a non-finite gradient norm"
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_micro_batches = 0
                epoch_steps += 1
                global_step += 1
        if accumulated_micro_batches:
            raise TrainingContractError("epoch ended inside a gradient-accumulation group")
        if epoch_steps != steps_per_epoch:
            raise TrainingContractError(
                "epoch optimizer-step count differs from the shared manifest"
            )
        if observed_row_ids != list(authorization.row_ids_by_epoch[epoch - 1]):
            raise TrainingContractError("epoch window order differs from the exact shared manifest")
        model.eval()
        metrics = dict(dev_evaluate(model, epoch))
        if set(metrics) != {
            "dev_total_loss",
            "dev_replacement_average_precision",
            "split_role",
            "source_ids_sha256",
        }:
            raise TrainingContractError("DEV checkpoint selection metrics are incomplete")
        if (
            metrics["split_role"] != DEV_ROLE
            or metrics["source_ids_sha256"] != authorization.dev_source_ids_sha256
        ):
            raise TrainingContractError("checkpoint selection did not use the exact DEV split")
        metric_values = {
            "dev_total_loss": float(metrics["dev_total_loss"]),
            "dev_replacement_average_precision": float(
                metrics["dev_replacement_average_precision"]
            ),
        }
        should_stop = stopper.update(
            metric_values["dev_total_loss"],
            metric_values["dev_replacement_average_precision"],
        )
        if stopper.improved:
            checkpoint(model, epoch, metric_values)
            selected_epoch = epoch
            selected_metrics = metric_values
        history.append({"epoch": epoch, "global_step": global_step, **metric_values})
        if should_stop:
            break
    if selected_epoch is None or selected_metrics is None:
        raise TrainingContractError("no DEV-selected checkpoint was persisted")
    return {
        "arm": arm,
        "epochs_completed": len(history),
        "optimizer_steps": global_step,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "checkpoint_metric": "dev_total_loss",
        "checkpoint_tiebreak": "dev_replacement_average_precision",
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_windows_per_optimizer_step": (MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS),
        "selected_checkpoint": {
            "epoch": selected_epoch,
            **selected_metrics,
            "selection_roles": [DEV_ROLE],
            "dev_source_ids_sha256": authorization.dev_source_ids_sha256,
        },
        "authorization_sha256": authorization.material_gate_sha256,
        "history": history,
    }


@torch.no_grad()
def evaluate_examples(
    model: TrainableSortformerPSEM,
    batches: Iterable[Sequence[TrainingExample]],
    class_weights: ClassWeights,
) -> dict[str, float]:
    model.eval()
    losses = []
    totals = []
    native = []
    logits = []
    targets = []
    masks = []
    for examples in batches:
        result = forward_batch(model, examples, class_weights)
        losses.append(float(result.losses["replacement"].detach().cpu()))
        totals.append(float(result.losses["total"].detach().cpu()))
        native.append(float(result.losses["native_sortformer"].detach().cpu()))
        logits.append(result.replacement_logits.detach().cpu())
        targets.append(result.replacement_targets.detach().cpu())
        masks.append(result.mask.detach().cpu())
    if not losses or not all(math.isfinite(value) for value in (*losses, *totals, *native)):
        raise TrainingContractError("evaluation losses are absent or non-finite")
    return {
        "replacement_loss": sum(losses) / len(losses),
        "total_loss": sum(totals) / len(totals),
        "native_sortformer_loss": sum(native) / len(native),
        "replacement_average_precision": duration_weighted_average_precision(
            torch.cat(logits), torch.cat(targets), torch.cat(masks)
        ),
    }


def _legacy_run_overfit_arm(
    model: TrainableSortformerPSEM,
    arm: str,
    batches: Sequence[Sequence[TrainingExample]],
    class_weights: ClassWeights,
    *,
    maximum_steps: int = OVERFIT_MAXIMUM_STEPS,
    authorization: _LegacyOverfitAuthorization,
) -> dict[str, Any]:
    require_material_execution_ready()
    examples = [example for batch in batches for example in batch]
    sources = {(example.source_id, example.corpus) for example in examples}
    identity_by_row = {
        row_id: tuple(values) for row_id, *values in authorization.input_identity_by_row
    }
    authorized_corpus_by_source = dict(authorization.corpus_by_source)
    if (
        maximum_steps != OVERFIT_MAXIMUM_STEPS
        or arm != authorization.arm
        or class_weights != authorization.class_weights
        or len(examples) != 60
        or Counter(corpus for _, corpus in sources) != Counter({"AMI": 2, "AliMeeting": 2})
        or [example.row_id for example in examples] != list(authorization.selected_row_ids)
        or any(
            example.split_role != TRAIN_ROLE
            or authorized_corpus_by_source.get(example.source_id) != example.corpus
            or example.sampling_manifest_sha256 != authorization.sampling_manifest_sha256
            or not _training_example_content_bound(example)
            or identity_by_row.get(example.row_id)
            != (
                example.source_id,
                example.corpus,
                example.window_start_sample,
                example.window_end_sample,
                example.target_identity_sha256,
                example.augmentation_identity_sha256,
                example.state_reset_at_start,
            )
            for example in examples
        )
    ):
        raise TrainingContractError(
            "overfit subset differs from the fixed 30-minute four-source rule"
        )
    initial = evaluate_examples(model, batches, class_weights)
    optimizer = build_optimizer(model, arm)
    scheduler = warmup_scheduler(optimizer, maximum_steps)
    model.train()
    for step in range(maximum_steps):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        result = forward_batch(model, batch, class_weights)
        result.losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            GRADIENT_CLIP_NORM,
        )
        optimizer.step()
        scheduler.step()
    final = evaluate_examples(model, batches, class_weights)
    return {
        "arm": arm,
        "sampling_manifest_sha256": authorization.sampling_manifest_sha256,
        "overfit_input_identity_sha256": authorization.selected_input_identity_sha256,
        "initial_replacement_loss": initial["replacement_loss"],
        "final_replacement_loss": final["replacement_loss"],
        "duration_weighted_replacement_average_precision": final["replacement_average_precision"],
        "final_native_sortformer_loss": final["native_sortformer_loss"],
        "optimizer_steps": maximum_steps,
    }


def _legacy_build_overfit_receipt(
    arm_results: Mapping[str, Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    corpus_by_source: Mapping[str, str],
    sampling_rows: Sequence[Mapping[str, Any]],
    sampling_manifest_path: Path,
    canary_receipts: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    allowed = {"H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"}
    if not {"H-HEAD", "T2-TOP"} <= set(arm_results) <= allowed:
        raise TrainingContractError("overfit receipt arm set violates staged authorization")
    if list(sampling_rows) != load_sampling_rows(sampling_manifest_path):
        raise TrainingContractError("overfit sampling rows differ from the persisted manifest")
    canonical_rows = select_overfit_rows(sampling_rows, corpus_by_source)
    if list(selected_rows) != list(canonical_rows):
        raise TrainingContractError("overfit rows differ from the committed hash selection")
    row_ids = [row.get("row_id") for row in selected_rows]
    sources = sorted({str(row.get("source_id")) for row in selected_rows})
    if (
        len(selected_rows) != 60
        or len(set(row_ids)) != 60
        or any(not isinstance(value, str) for value in row_ids)
        or Counter(corpus_by_source.get(value) for value in sources)
        != Counter({"AMI": 2, "AliMeeting": 2})
    ):
        raise TrainingContractError("overfit receipt subset differs from the hash-selected budget")
    arms = {}
    for arm, values in arm_results.items():
        result_conditional_authorization = values.get("conditional_arm_audit_authorization")
        if (
            values.get("arm") != arm
            or values.get("optimizer_steps") != OVERFIT_MAXIMUM_STEPS
            or values.get("sampling_manifest_sha256") != sha256_file(sampling_manifest_path)
            or values.get("overfit_input_identity_sha256") != canonical_sha256(list(selected_rows))
        ):
            raise TrainingContractError(f"overfit optimizer budget differs for {arm}")
        bound = canary_receipts.get(arm, {})
        gradient = bound.get("gradient_canary_receipt")
        update = bound.get("update_canary_receipt")
        timing = bound.get("timing_receipt")
        inventory = bound.get("parameter_inventory")
        graph = bound.get("model_graph_receipt")
        canary_conditional_authorization = bound.get("conditional_arm_audit_authorization")
        if (
            not isinstance(gradient, Mapping)
            or not isinstance(update, Mapping)
            or not isinstance(timing, Mapping)
            or not isinstance(inventory, Mapping)
            or not isinstance(graph, Mapping)
            or not canary_bundle_runtime_passed(
                gradient,
                update,
                timing,
                arm,
                parameter_inventory_receipt=inventory,
                model_graph_receipt=graph,
            )
            or result_conditional_authorization != canary_conditional_authorization
            or (
                arm == "TA-ALL-TEMPORAL"
                and (
                    not isinstance(result_conditional_authorization, Mapping)
                    or result_conditional_authorization.get("artifact_role")
                    != "conditional_arm_audit_authorization"
                    or result_conditional_authorization.get("arm") != arm
                    or result_conditional_authorization.get("payload_sha256")
                    != canonical_sha256(
                        {
                            key: value
                            for key, value in result_conditional_authorization.items()
                            if key != "payload_sha256"
                        }
                    )
                )
            )
            or (arm != "TA-ALL-TEMPORAL" and result_conditional_authorization is not None)
        ):
            raise TrainingContractError(f"overfit canaries are absent or invalid for {arm}")
        arms[arm] = {
            **dict(values),
            "gradient_canary_sha256": canonical_sha256(gradient),
            "update_canary_sha256": canonical_sha256(update),
            "timing_receipt_sha256": canonical_sha256(timing),
            "parameter_inventory_sha256": canonical_sha256(inventory),
            "model_graph_receipt_sha256": canonical_sha256(graph),
        }
    payload = {
        "schema_version": 1,
        "artifact_role": "overfit_canary",
        "split_roles": [TRAIN_ROLE],
        "eval_source_count": 0,
        "selection_rule": "issue-107-overfit-source-v1+issue-107-overfit-window-v1",
        "duration_minutes": 30,
        "maximum_optimizer_steps": OVERFIT_MAXIMUM_STEPS,
        "sampling_manifest_sha256": sha256_file(sampling_manifest_path),
        "selected_row_ids": row_ids,
        "selected_rows_sha256": canonical_sha256(list(selected_rows)),
        "sources": [
            {"source_id": source_id, "corpus": corpus_by_source[source_id]} for source_id in sources
        ],
        "arms": arms,
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def fit_arm(
    model: TrainableSortformerPSEM,
    arm: str,
    class_weights: ClassWeights,
    train_batches: Callable[[int], Iterable[Sequence[TrainingExample]]],
    steps_per_epoch: int,
    dev_evaluate: Callable[[TrainableSortformerPSEM, int], Mapping[str, float]] | None,
    checkpoint: Callable[[TrainableSortformerPSEM, int, Mapping[str, Any]], None],
    *,
    authorization: OfficialTrainingAuthorization,
) -> dict[str, Any]:
    require_material_execution_ready()
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if (
        steps_per_epoch != OFFICIAL_OPTIMIZER_STEPS
        or arm != authorization.arm
        or authorization.seed != 7301
        or current_head != authorization.git_head
        or current_dirty
        or class_weights != authorization.class_weights
        or dev_evaluate is not None
        or len(authorization.row_ids_by_epoch) != 1
        or len(authorization.row_ids_by_epoch[0]) != WINDOWS_PER_EPOCH
    ):
        raise TrainingContractError("official training requires the exact 256-step lean recipe")
    optimizer = build_optimizer(model, arm)
    scheduler = warmup_scheduler(optimizer, OFFICIAL_OPTIMIZER_STEPS, WARMUP_STEPS)
    model.train()
    identity_by_row = {
        row_id: tuple(values) for row_id, *values in authorization.input_identity_by_row
    }
    observed_row_ids: list[str] = []
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    global_step = 0
    for examples in train_batches(1):
        if len(examples) != MICRO_BATCH_SIZE:
            raise TrainingContractError("official training requires microbatch size one")
        for example in examples:
            expected = identity_by_row.get(example.row_id)
            actual = (
                example.source_id,
                example.corpus,
                example.window_start_sample,
                example.window_end_sample,
                example.target_identity_sha256,
                example.augmentation_identity_sha256,
                example.state_reset_at_start,
            )
            if (
                example.split_role != TRAIN_ROLE
                or example.epoch != 1
                or example.epoch_index != len(observed_row_ids)
                or example.sampling_manifest_sha256 != authorization.sampling_manifest_sha256
                or expected != actual
                or not _training_example_content_bound(example)
            ):
                raise TrainingContractError("training batch differs from the exact shared manifest")
            observed_row_ids.append(example.row_id)
        result = forward_batch(model, examples, class_weights)
        total = result.losses["total"]
        if not bool(torch.isfinite(total)):
            raise TrainingContractError("official training produced a non-finite loss")
        losses.append(float(total.detach().cpu()))
        (total / GRADIENT_ACCUMULATION_STEPS).backward()
        accumulated += 1
        if accumulated == GRADIENT_ACCUMULATION_STEPS:
            norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                GRADIENT_CLIP_NORM,
            )
            if not bool(torch.isfinite(norm)):
                raise TrainingContractError("official training produced a non-finite gradient")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            global_step += 1
    if (
        accumulated != 0
        or global_step != OFFICIAL_OPTIMIZER_STEPS
        or observed_row_ids != list(authorization.row_ids_by_epoch[0])
    ):
        raise TrainingContractError("official training did not consume all 4096 rows in order")
    final_metrics = {
        "final_step": global_step,
        "mean_total_loss": sum(losses) / len(losses),
        "split_role": TRAIN_ROLE,
        "scheduler_total_steps": OFFICIAL_OPTIMIZER_STEPS,
        "warmup_steps": WARMUP_STEPS,
    }
    checkpoint(model, global_step, final_metrics)
    return {
        "arm": arm,
        "seed": authorization.seed,
        "optimizer_steps": global_step,
        "maximum_optimizer_steps": OFFICIAL_OPTIMIZER_STEPS,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "consumed_row_count": len(observed_row_ids),
        "row_order_sha256": canonical_sha256(observed_row_ids),
        "scheduler_total_steps": OFFICIAL_OPTIMIZER_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "dev_callback_used": False,
        "early_stopping_used": False,
        "checkpoint_step": global_step,
        "loss_summary": final_metrics,
        "authorization_sha256": authorization.material_gate_sha256,
    }


def run_short_smoke(
    model: TrainableSortformerPSEM,
    arm: str,
    class_weights: ClassWeights,
    batches: Iterable[Sequence[TrainingExample]],
    *,
    expected_row_ids: Sequence[str],
    parameter_policy: Mapping[str, Any],
) -> dict[str, Any]:
    require_material_execution_ready()
    if arm not in {"H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"}:
        raise TrainingContractError("smoke arm is unauthorized")
    if len(expected_row_ids) != 512 or len(set(expected_row_ids)) != 512:
        raise TrainingContractError("smoke requires the first 512 manifest rows")
    if parameter_policy.get("arm") != arm:
        raise TrainingContractError("smoke parameter policy identity differs from the arm")
    examples = [example for batch in batches for example in batch]
    if len(examples) != 512 or [example.row_id for example in examples] != list(expected_row_ids):
        raise TrainingContractError("smoke rows are not the first 512 manifest rows in order")
    if any(
        example.split_role != TRAIN_ROLE or not _training_example_content_bound(example)
        for example in examples
    ):
        raise TrainingContractError("smoke input identity is not content-bound")
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = build_optimizer(model, arm)
    scheduler = warmup_scheduler(optimizer, SMOKE_OPTIMIZER_STEPS, 2)
    losses: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(SMOKE_OPTIMIZER_STEPS):
        window_losses: list[float] = []
        for example in examples[
            step * GRADIENT_ACCUMULATION_STEPS : (step + 1) * GRADIENT_ACCUMULATION_STEPS
        ]:
            result = forward_batch(model, (example,), class_weights)
            total = result.losses["total"]
            if not bool(torch.isfinite(total)):
                raise TrainingContractError("smoke forward produced a non-finite loss")
            window_losses.append(float(total.detach().cpu()))
            (total / GRADIENT_ACCUMULATION_STEPS).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            GRADIENT_CLIP_NORM,
        )
        if not bool(torch.isfinite(norm)):
            raise TrainingContractError("smoke backward produced a non-finite gradient")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(sum(window_losses) / len(window_losses))
    if not all(math.isfinite(value) for value in losses):
        raise TrainingContractError("smoke losses are non-finite")
    first = sum(losses[:8]) / 8
    final = sum(losses[-8:]) / 8
    if not final < first:
        raise TrainingContractError("smoke loss did not improve over the required windows")
    changed = []
    frozen_unchanged = True
    for name, value in model.named_parameters():
        different = not torch.equal(before[name], value.detach())
        if value.requires_grad:
            changed.append((name, different))
        else:
            frozen_unchanged = frozen_unchanged and not different
    if not changed or not all(different for _, different in changed) or not frozen_unchanged:
        raise TrainingContractError("smoke parameter policy or update identity failed")
    return {
        "schema_version": 1,
        "artifact_role": "short_smoke_metrics",
        "arm": arm,
        "seed": 7301,
        "optimizer_steps": SMOKE_OPTIMIZER_STEPS,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "consumed_row_count": len(examples),
        "row_ids_sha256": canonical_sha256(list(expected_row_ids)),
        "first_eight_mean_total_loss": first,
        "last_eight_mean_total_loss": final,
        "finite_forward_backward_update": True,
        "parameter_policy": dict(parameter_policy),
        "updated_trainable_parameters": [name for name, _ in changed],
        "frozen_parameters_unchanged": frozen_unchanged,
        "weights_discarded": True,
    }
