from __future__ import annotations

import argparse
import hmac
import json
import re
import secrets
import stat
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    ParsedRttm,
    ReferenceSpan,
    build_alimeeting_speaker_map,
    build_ami_speaker_map,
    canonical_sha256,
    parse_rttm,
    resolve_reference_path,
    sha256_file,
    timestamp_to_sample,
    validate_reference_checkout,
)
from experiments.psem_training_strategy_gate.data.label_contract import (
    CanonicalInterval,
    LabelResult,
    generate_labels,
    load_contract,
    normalize_intervals,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    ProvenanceError,
    _parse_alimeeting_textgrid,
    wav_identity,
    write_jsonl,
)

SAMPLE_RATE_HZ = 16000
MASK_CLASS = "ambiguous_nonlexical_vocalization"
INVENTORY_PATH = Path(__file__).with_name("v2") / "nonlexical_risk_inventory.json"
EXPECTED_INVENTORY_SHA256 = (
    "59a4e7d289051e11ed6a0f353a3308eac33b3c7aa2238d54cf21a88ae688b66e"
)
AMI_WORD_NAME = re.compile(
    r"^(?P<session>[A-Za-z0-9]+)\.(?P<agent>[A-Za-z0-9]+)\.words\.xml$"
)
ALIMEETING_TIER_NAME = re.compile(r"^N_(SPK[0-9]+)$")
TEXTGRID_ITEM = re.compile(r"\s*item \[(\d+)\]:\s*")
TEXTGRID_INTERVAL = re.compile(r"\s*intervals \[(\d+)\]:\s*")
MARKUP_TOKEN = re.compile(
    r"\[[^\[\]\r\n]+\]|<[^<>\r\n]+>|\{[^{}\r\n]+\}|\([^()\r\n]+\)"
)
MARKUP_CHARACTERS = frozenset("[]<>{}()")
AMI_ELEMENT_ACTIONS = {
    "w": "rttm_activity_only",
    "vocalsound": "mask_by_explicit_or_point_timing",
    "disfmarker": "no_activity_no_mask",
    "gap": "no_activity_no_mask",
    "transformerror": "no_activity_no_mask",
}
ALIMEETING_CLASS_ACTIONS = {
    "empty": "no_activity_no_mask",
    "lexical_text": "rttm_activity_only",
    "human_vocal_marker_only": "full_interval_handoff_relation_mask",
    "mixed_lexical_human_vocal": (
        "rttm_activity_and_full_interval_handoff_relation_mask"
    ),
    "nonhuman_noise_marker_only": "no_activity_no_mask",
    "mixed_lexical_nonhuman_noise": "rttm_activity_only",
}
_INVENTORY_VALIDATION_TOKEN = object()
_REFERENCE_CHECKOUT_VALIDATION_KEY = secrets.token_bytes(32)


class ReferenceNormalizationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NonlexicalInventory:
    document_sha256: str
    marker_actions: Mapping[str, str]
    ami_nonzero_padding_samples: int
    ami_point_padding_samples: int
    _validation_token: object


@dataclass(frozen=True, slots=True)
class PinnedReferenceCheckout:
    root: Path
    provenance: Mapping[str, Any]
    _validation_receipt: bytes


@dataclass(frozen=True, slots=True)
class AmbiguityMask:
    start_sample: int
    end_sample: int
    mask_class: str
    speaker_id: str
    source_annotation_id: str
    annotation_class: str

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "mask_class": self.mask_class,
            "speaker_id": self.speaker_id,
            "source_annotation_id": self.source_annotation_id,
            "annotation_class": self.annotation_class,
        }


@dataclass(frozen=True, slots=True)
class ParsedNonlexical:
    masks: tuple[AmbiguityMask, ...]
    observed_class_counts: Mapping[str, int]
    observed_marker_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ReferenceMetadataFile:
    role: str
    ref: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "ref": self.ref,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ReferenceNormalizedSession:
    source_id: str
    corpus: str
    session_id: str
    scored_start_sample: int
    scored_end_sample: int
    reference_path: Path
    reference_sha256: str
    parsed_rttm: ParsedRttm
    parsed_nonlexical: ParsedNonlexical
    intervals: tuple[CanonicalInterval, ...]
    labels: LabelResult
    inventory_sha256: str
    speaker_mapping_sha256: str
    metadata_files: tuple[ReferenceMetadataFile, ...]
    source_record_sha256: str
    source_waveform_sha256: str
    source_annotation_sha256: str
    reference_checkout_provenance: Mapping[str, Any]

    def manifest_row(self) -> dict[str, Any]:
        interval_rows = [interval.to_dict() for interval in self.intervals]
        masks = [mask.to_dict() for mask in self.parsed_nonlexical.masks]
        metadata_files = [row.to_dict() for row in self.metadata_files]
        return {
            "schema_version": 1,
            "source_id": self.source_id,
            "corpus": self.corpus,
            "session_id": self.session_id,
            "contract_version": self.labels.contract_version,
            "contract_document_sha256": self.labels.contract_document_sha256,
            "nonlexical_inventory_sha256": self.inventory_sha256,
            "reference_ref": _stable_reference_ref(
                self.reference_path, self.corpus, self.session_id
            ),
            "reference_repository": self.reference_checkout_provenance["repository"],
            "reference_commit": self.reference_checkout_provenance["commit"],
            "reference_git_tree": self.reference_checkout_provenance["git_tree"],
            "reference_sha256": self.reference_sha256,
            "source_record_sha256": self.source_record_sha256,
            "source_waveform_sha256": self.source_waveform_sha256,
            "source_annotation_sha256": self.source_annotation_sha256,
            "speaker_mapping_sha256": self.speaker_mapping_sha256,
            "reference_metadata_files": metadata_files,
            "reference_metadata_sha256": canonical_sha256(metadata_files),
            "scored_start_sample": self.scored_start_sample,
            "scored_end_sample": self.scored_end_sample,
            "raw_rttm_row_count": self.parsed_rttm.raw_row_count,
            "canonical_rttm_span_count": len(self.parsed_rttm.spans),
            "clipped_tail_rttm_row_count": self.parsed_rttm.clipped_tail_row_count,
            "nonlexical_mask_count": len(masks),
            "nonlexical_mask_sha256": canonical_sha256(masks),
            "nonlexical_class_counts": dict(self.parsed_nonlexical.observed_class_counts),
            "nonlexical_marker_counts": dict(
                self.parsed_nonlexical.observed_marker_counts
            ),
            "canonical_interval_count": len(self.intervals),
            "canonical_intervals_sha256": canonical_sha256(interval_rows),
            "label_result_sha256": canonical_sha256(self.labels.to_dict()),
            "exposure": dict(self.labels.exposure),
        }


def _exact_positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReferenceNormalizationError(f"{field} must be a positive integer")
    return value


def _is_exact_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _stable_reference_ref(path: Path, corpus: str, session_id: str) -> str:
    expected_name = f"{session_id}.rttm"
    parts = path.parts
    if (
        len(parts) < 3
        or parts[-3] != corpus
        or path.name != expected_name
    ):
        raise ReferenceNormalizationError(
            "reference path has no stable corpus-relative identity"
        )
    return Path(*parts[-3:]).as_posix()


def _metadata_file(path: Path, role: str, corpus_root: Path) -> ReferenceMetadataFile:
    root = corpus_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ReferenceNormalizationError("reference metadata escapes the corpus root")
    try:
        details = path.lstat()
    except OSError as exc:
        raise ReferenceNormalizationError(f"reference metadata is unavailable: {path}") from exc
    attributes = getattr(details, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISREG(details.st_mode)
        or path.is_symlink()
        or (reparse_flag and attributes & reparse_flag)
        or not role
    ):
        raise ReferenceNormalizationError(f"reference metadata is invalid: {path}")
    return ReferenceMetadataFile(
        role=role,
        ref=resolved.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        size_bytes=details.st_size,
    )


def _bound_file_details(path: Path, corpus_root: Path) -> tuple[Path, int]:
    root = corpus_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ReferenceNormalizationError("source asset escapes the corpus root")
    try:
        details = path.lstat()
    except OSError as exc:
        raise ReferenceNormalizationError(f"source asset is unavailable: {path}") from exc
    if (
        not stat.S_ISREG(details.st_mode)
        or path.is_symlink()
        or resolved != path.absolute()
    ):
        raise ReferenceNormalizationError(f"source asset is invalid: {path}")
    return resolved, details.st_size


def _source_file_row(path: Path, corpus_root: Path) -> dict[str, Any]:
    root = corpus_root.resolve()
    resolved, size_bytes = _bound_file_details(path, root)
    return {
        "ref": resolved.relative_to(root).as_posix(),
        "size_bytes": size_bytes,
        "sha256": sha256_file(path),
    }


def load_nonlexical_inventory(path: Path | None = None) -> NonlexicalInventory:
    inventory_path = path or INVENTORY_PATH
    try:
        raw = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceNormalizationError(
            f"invalid nonlexical inventory: {inventory_path}"
        ) from exc
    document_sha256 = canonical_sha256(raw)
    if path is None and document_sha256 != EXPECTED_INVENTORY_SHA256:
        raise ReferenceNormalizationError(
            "installed nonlexical inventory does not match its pinned identity"
        )
    try:
        ami = raw["ami"]
        alimeeting = raw["alimeeting"]
        marker_actions = alimeeting["marker_tokens"]
    except (KeyError, TypeError) as exc:
        raise ReferenceNormalizationError(
            "nonlexical inventory structure is incomplete"
        ) from exc
    if (
        raw.get("schema_version") != 1
        or raw.get("inventory_version") != "psem-nonlexical-v1"
        or raw.get("contract_version") != "psem-handoff-v1"
        or raw.get("mask_class") != MASK_CLASS
        or ami.get("element_types") != AMI_ELEMENT_ACTIONS
        or ami.get("reversed_vocalsound_timing")
        != "sort_bounds_then_mask_nonzero_interval"
        or ami.get("missing_starttime")
        != "mask_between_nearest_timed_siblings_plus_nonzero_padding"
        or ami.get("missing_endtime")
        != "treat_starttime_as_point_and_apply_point_padding"
        or ami.get("unmapped_empty_speaker_file")
        != "bind_as_empty_corpus_placeholder_without_activity_or_mask"
        or alimeeting.get("text_normalization")
        != "unicode_nfkc_trim_and_collapse_whitespace"
        or alimeeting.get("markup_delimiters")
        != ["square", "angle", "curly", "round"]
        or alimeeting.get("class_actions") != ALIMEETING_CLASS_ACTIONS
        or alimeeting.get("unseen_or_malformed_markup") != "fail_preflight"
        or not isinstance(marker_actions, dict)
    ):
        raise ReferenceNormalizationError(
            "nonlexical inventory semantics do not match psem-handoff-v1"
        )
    normalized_markers: dict[str, str] = {}
    for marker, action in marker_actions.items():
        if (
            not isinstance(marker, str)
            or normalize_alimeeting_text(marker) != marker
            or MARKUP_TOKEN.fullmatch(marker) is None
            or action not in {"human_vocal", "nonhuman_noise"}
        ):
            raise ReferenceNormalizationError(
                "nonlexical marker inventory contains an invalid action"
            )
        normalized_markers[marker] = action
    return NonlexicalInventory(
        document_sha256=document_sha256,
        marker_actions=normalized_markers,
        ami_nonzero_padding_samples=_exact_positive_integer(
            ami.get("nonzero_padding_samples"), "AMI nonzero padding"
        ),
        ami_point_padding_samples=_exact_positive_integer(
            ami.get("point_padding_samples"), "AMI point padding"
        ),
        _validation_token=_INVENTORY_VALIDATION_TOKEN,
    )


def _require_validated_inventory(inventory: NonlexicalInventory) -> None:
    if inventory._validation_token is not _INVENTORY_VALIDATION_TOKEN:
        raise ReferenceNormalizationError("nonlexical inventory was not validated")


def open_reference_checkout(reference_root: Path) -> PinnedReferenceCheckout:
    root = reference_root.resolve()
    provenance = validate_reference_checkout(root)
    return PinnedReferenceCheckout(
        root=root,
        provenance=MappingProxyType(dict(provenance)),
        _validation_receipt=_reference_checkout_receipt(root, provenance),
    )


def _reference_checkout_receipt(
    root: Path, provenance: Mapping[str, Any]
) -> bytes:
    payload = (
        f"{root}\0{canonical_sha256(dict(provenance))}"
    ).encode("utf-8")
    return hmac.digest(_REFERENCE_CHECKOUT_VALIDATION_KEY, payload, "sha256")


def _require_pinned_checkout(checkout: PinnedReferenceCheckout) -> None:
    if (
        not isinstance(checkout._validation_receipt, bytes)
        or not hmac.compare_digest(
            checkout._validation_receipt,
            _reference_checkout_receipt(checkout.root, checkout.provenance),
        )
        or checkout.root != checkout.root.resolve()
        or checkout.provenance.get("repository") != REFERENCE_REPOSITORY
        or checkout.provenance.get("commit") != REFERENCE_COMMIT
        or not re.fullmatch(r"[0-9a-f]{40}", str(checkout.provenance.get("git_tree", "")))
    ):
        raise ReferenceNormalizationError("forced-alignment checkout was not validated")


def normalize_alimeeting_text(value: str) -> str:
    if not isinstance(value, str):
        raise ReferenceNormalizationError("AliMeeting annotation text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _has_lexical_content(value: str) -> bool:
    return any(unicodedata.category(character)[0] in {"L", "N"} for character in value)


def classify_alimeeting_text(
    value: str, inventory: NonlexicalInventory
) -> tuple[str, tuple[str, ...]]:
    _require_validated_inventory(inventory)
    normalized = normalize_alimeeting_text(value)
    if not normalized:
        return "empty", ()
    matches = tuple(MARKUP_TOKEN.finditer(normalized))
    if not matches:
        if any(character in MARKUP_CHARACTERS for character in normalized):
            raise ReferenceNormalizationError("unseen or malformed AliMeeting markup")
        return "lexical_text", ()
    residual_parts: list[str] = []
    cursor = 0
    markers: list[str] = []
    for match in matches:
        residual_parts.append(normalized[cursor : match.start()])
        markers.append(match.group(0))
        cursor = match.end()
    residual_parts.append(normalized[cursor:])
    residual = " ".join(residual_parts)
    if any(character in MARKUP_CHARACTERS for character in residual):
        raise ReferenceNormalizationError("unseen or malformed AliMeeting markup")
    try:
        actions = tuple(inventory.marker_actions[marker] for marker in markers)
    except KeyError as exc:
        raise ReferenceNormalizationError(
            f"unseen AliMeeting marker token: {exc.args[0]}"
        ) from exc
    lexical = _has_lexical_content(residual)
    if "human_vocal" in actions:
        annotation_class = (
            "mixed_lexical_human_vocal" if lexical else "human_vocal_marker_only"
        )
    else:
        annotation_class = (
            "mixed_lexical_nonhuman_noise"
            if lexical
            else "nonhuman_noise_marker_only"
        )
    return annotation_class, tuple(markers)


def _decimal(value: str, field: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ReferenceNormalizationError(f"invalid timestamp for {field}") from exc
    if not result.is_finite() or abs(result.adjusted()) > 18:
        raise ReferenceNormalizationError(f"invalid timestamp for {field}")
    return result


def _annotation_id(element: ET.Element, path: Path) -> str:
    value = next(
        (
            attribute_value
            for attribute_name, attribute_value in element.attrib.items()
            if attribute_name == "id" or attribute_name.endswith("}id")
        ),
        None,
    )
    if not value or value.strip() != value:
        raise ReferenceNormalizationError(f"AMI word element lacks an ID: {path}")
    return f"{path.name}#{value}"


def _local_name(element: ET.Element) -> str:
    if not isinstance(element.tag, str):
        raise ReferenceNormalizationError("AMI word XML contains an unsupported node")
    return element.tag.rsplit("}", 1)[-1]


def _neighbor_vocalsound_bounds(
    elements: Sequence[ET.Element], index: int, source_annotation_id: str
) -> tuple[Decimal, Decimal]:
    left: Decimal | None = None
    right: Decimal | None = None
    for neighbor in reversed(elements[:index]):
        start_value = neighbor.get("starttime")
        if start_value is None:
            continue
        start_time = _decimal(start_value, "AMI neighbor starttime")
        end_value = neighbor.get("endtime")
        end_time = (
            start_time
            if end_value is None
            else _decimal(end_value, "AMI neighbor endtime")
        )
        left = max(start_time, end_time)
        break
    for neighbor in elements[index + 1 :]:
        start_value = neighbor.get("starttime")
        if start_value is None:
            continue
        start_time = _decimal(start_value, "AMI neighbor starttime")
        end_value = neighbor.get("endtime")
        end_time = (
            start_time
            if end_value is None
            else _decimal(end_value, "AMI neighbor endtime")
        )
        right = min(start_time, end_time)
        break
    if left is None or right is None or left < 0 or right <= left:
        raise ReferenceNormalizationError(
            f"AMI vocalsound has no deterministic localization: {source_annotation_id}"
        )
    return left, right


def _clipped_mask(
    *,
    start_sample: int,
    end_sample: int,
    scored_start_sample: int,
    scored_end_sample: int,
    speaker_id: str,
    source_annotation_id: str,
    annotation_class: str,
) -> AmbiguityMask:
    clipped_start = max(scored_start_sample, start_sample)
    clipped_end = min(scored_end_sample, end_sample)
    if clipped_end <= clipped_start:
        raise ReferenceNormalizationError(
            f"nonlexical mask does not intersect the scored timeline: {source_annotation_id}"
        )
    return AmbiguityMask(
        start_sample=clipped_start,
        end_sample=clipped_end,
        mask_class=MASK_CLASS,
        speaker_id=speaker_id,
        source_annotation_id=source_annotation_id,
        annotation_class=annotation_class,
    )


def parse_ami_nonlexical_masks(
    session_id: str,
    word_paths: Sequence[Path],
    *,
    speaker_map: Mapping[str, str],
    scored_start_sample: int,
    scored_end_sample: int,
    inventory: NonlexicalInventory | None = None,
) -> ParsedNonlexical:
    active_inventory = inventory or load_nonlexical_inventory()
    _require_validated_inventory(active_inventory)
    masks: list[AmbiguityMask] = []
    class_counts: Counter[str] = Counter()
    marker_counts: Counter[str] = Counter()
    found_agents: set[str] = set()
    seen_ids: set[str] = set()
    for path in sorted(word_paths):
        match = AMI_WORD_NAME.fullmatch(path.name)
        if match is None or match.group("session") != session_id:
            raise ReferenceNormalizationError(f"unexpected AMI word filename: {path.name}")
        agent = match.group("agent")
        if agent in found_agents:
            raise ReferenceNormalizationError(
                f"duplicate AMI word speaker file: {session_id}.{agent}"
            )
        found_agents.add(agent)
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise ReferenceNormalizationError(f"invalid AMI word XML: {path}") from exc
        elements = list(root)
        speaker_id = speaker_map.get(agent)
        if not elements and not isinstance(speaker_id, str):
            class_counts["file:unmapped_empty_speaker_placeholder"] += 1
        if elements and (not isinstance(speaker_id, str) or not speaker_id):
            raise ReferenceNormalizationError(
                f"AMI word speaker has no official identity: {session_id}.{agent}"
            )
        for index, element in enumerate(elements):
            element_type = _local_name(element)
            if element_type not in AMI_ELEMENT_ACTIONS:
                raise ReferenceNormalizationError(
                    f"unseen AMI word element type: {element_type}"
                )
            source_annotation_id = _annotation_id(element, path)
            if source_annotation_id in seen_ids:
                raise ReferenceNormalizationError("AMI word annotation IDs are duplicated")
            seen_ids.add(source_annotation_id)
            class_counts[f"element:{element_type}"] += 1
            if element_type != "vocalsound":
                continue
            vocal_type = element.get("type")
            if not vocal_type or vocal_type.strip() != vocal_type:
                raise ReferenceNormalizationError(
                    f"AMI vocalsound type is unresolved: {source_annotation_id}"
                )
            marker_counts[vocal_type] += 1
            start_value = element.get("starttime")
            end_value = element.get("endtime")
            if start_value is None:
                start_time, end_time = _neighbor_vocalsound_bounds(
                    elements, index, source_annotation_id
                )
                inferred_bounds = True
            else:
                start_time = _decimal(start_value, "AMI vocalsound starttime")
                end_time = (
                    start_time
                    if end_value is None
                    else _decimal(end_value, "AMI vocalsound endtime")
                )
                inferred_bounds = False
            if start_time < 0 or end_time < 0:
                raise ReferenceNormalizationError(
                    f"AMI vocalsound has invalid timing: {source_annotation_id}"
                )
            reversed_bounds = end_time < start_time
            if reversed_bounds:
                start_time, end_time = end_time, start_time
            start_sample = timestamp_to_sample(start_time)
            end_sample = timestamp_to_sample(end_time)
            if inferred_bounds:
                padding = active_inventory.ami_nonzero_padding_samples
                mask_start = start_sample - padding
                mask_end = end_sample + padding
                annotation_class = "ami_unlocalized_vocalsound_neighbor_bounded"
            elif end_time == start_time:
                padding = active_inventory.ami_point_padding_samples
                mask_start = start_sample - padding
                mask_end = start_sample + padding
                annotation_class = "ami_point_or_zero_duration_vocalsound"
            elif reversed_bounds:
                padding = active_inventory.ami_nonzero_padding_samples
                mask_start = start_sample - padding
                mask_end = end_sample + padding
                annotation_class = "ami_reversed_vocalsound_bounds"
            else:
                padding = active_inventory.ami_nonzero_padding_samples
                mask_start = start_sample - padding
                mask_end = end_sample + padding
                annotation_class = "ami_nonzero_duration_vocalsound"
            class_counts[f"mask:{annotation_class}"] += 1
            masks.append(
                _clipped_mask(
                    start_sample=mask_start,
                    end_sample=mask_end,
                    scored_start_sample=scored_start_sample,
                    scored_end_sample=scored_end_sample,
                    speaker_id=speaker_id,
                    source_annotation_id=source_annotation_id,
                    annotation_class=annotation_class,
                )
            )
    if not set(speaker_map).issubset(found_agents):
        raise ReferenceNormalizationError(
            f"AMI word bundle does not match official speakers: {session_id}"
        )
    return ParsedNonlexical(
        masks=tuple(sorted(masks, key=_mask_order)),
        observed_class_counts=dict(sorted(class_counts.items())),
        observed_marker_counts=dict(sorted(marker_counts.items())),
    )


def _textgrid_value(lines: Sequence[str], key: str) -> str:
    prefix = f"{key} ="
    matches = [line.strip().split("=", 1)[1].strip() for line in lines if line.strip().startswith(prefix)]
    if len(matches) != 1:
        raise ReferenceNormalizationError(f"TextGrid field cardinality mismatch: {key}")
    value = matches[0]
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('""', '"')
    return value


def _textgrid_items(
    path: Path,
) -> list[tuple[str, str, str, list[tuple[str, str, str, str]]]]:
    try:
        _parse_alimeeting_textgrid(path)
    except (OSError, UnicodeError, ProvenanceError) as exc:
        raise ReferenceNormalizationError(
            f"invalid AliMeeting TextGrid structure: {path}"
        ) from exc
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReferenceNormalizationError(f"invalid AliMeeting TextGrid: {path}") from exc
    item_matches = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := TEXTGRID_ITEM.fullmatch(line))
    ]
    if not item_matches:
        raise ReferenceNormalizationError("AliMeeting TextGrid has no tiers")
    items = []
    for position, (start, item_match) in enumerate(item_matches):
        end = item_matches[position + 1][0] if position + 1 < len(item_matches) else len(lines)
        block = lines[start:end]
        tier_name = _textgrid_value(block, "name")
        tier_class = _textgrid_value(block, "class")
        interval_matches = [
            (index, match)
            for index, line in enumerate(block)
            if (match := TEXTGRID_INTERVAL.fullmatch(line))
        ]
        intervals = []
        for interval_position, (interval_start, interval_match) in enumerate(
            interval_matches
        ):
            interval_end = (
                interval_matches[interval_position + 1][0]
                if interval_position + 1 < len(interval_matches)
                else len(block)
            )
            interval = block[interval_start:interval_end]
            intervals.append(
                (
                    interval_match.group(1),
                    _textgrid_value(interval, "xmin"),
                    _textgrid_value(interval, "xmax"),
                    _textgrid_value(interval, "text"),
                )
            )
        items.append((item_match.group(1), tier_name, tier_class, intervals))
    return items


def _textgrid_timeline(path: Path) -> tuple[Decimal, Decimal]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReferenceNormalizationError(f"invalid AliMeeting TextGrid: {path}") from exc
    first_item = next(
        (index for index, line in enumerate(lines) if TEXTGRID_ITEM.fullmatch(line)),
        None,
    )
    if first_item is None:
        raise ReferenceNormalizationError("AliMeeting TextGrid has no tiers")
    header = lines[:first_item]
    return (
        _decimal(_textgrid_value(header, "xmin"), "AliMeeting timeline xmin"),
        _decimal(_textgrid_value(header, "xmax"), "AliMeeting timeline xmax"),
    )


def alimeeting_speaker_ids(path: Path) -> tuple[str, ...]:
    speakers = []
    for _, tier_name, tier_class, _ in _textgrid_items(path):
        match = ALIMEETING_TIER_NAME.fullmatch(tier_name)
        if tier_class != "IntervalTier" or match is None:
            raise ReferenceNormalizationError(
                f"unexpected AliMeeting TextGrid tier: {tier_name}"
            )
        speakers.append(match.group(1))
    if len(set(speakers)) != len(speakers):
        raise ReferenceNormalizationError("AliMeeting TextGrid speaker tiers are duplicated")
    return tuple(speakers)


def parse_alimeeting_nonlexical_masks(
    source_id: str,
    path: Path,
    *,
    speaker_map: Mapping[str, str],
    scored_start_sample: int,
    scored_end_sample: int,
    inventory: NonlexicalInventory | None = None,
) -> ParsedNonlexical:
    active_inventory = inventory or load_nonlexical_inventory()
    _require_validated_inventory(active_inventory)
    if source_id != f"alimeeting_{path.stem}":
        raise ReferenceNormalizationError("AliMeeting TextGrid session identity mismatch")
    timeline_start, timeline_end = _textgrid_timeline(path)
    if (
        timestamp_to_sample(timeline_start) != scored_start_sample
        or timestamp_to_sample(timeline_end) != scored_end_sample
    ):
        raise ReferenceNormalizationError(
            f"AliMeeting TextGrid timeline does not match scored range: {path}"
        )
    masks: list[AmbiguityMask] = []
    class_counts: Counter[str] = Counter()
    marker_counts: Counter[str] = Counter()
    found_speakers: set[str] = set()
    for item_id, tier_name, tier_class, intervals in _textgrid_items(path):
        match = ALIMEETING_TIER_NAME.fullmatch(tier_name)
        if tier_class != "IntervalTier" or match is None:
            raise ReferenceNormalizationError(
                f"unexpected AliMeeting TextGrid tier: {tier_name}"
            )
        source_speaker = match.group(1)
        speaker_id = speaker_map.get(source_speaker)
        if speaker_id != source_speaker:
            raise ReferenceNormalizationError(
                f"AliMeeting speaker identity is unresolved: {tier_name}"
            )
        if source_speaker in found_speakers:
            raise ReferenceNormalizationError(
                f"duplicate AliMeeting speaker tier: {tier_name}"
            )
        found_speakers.add(source_speaker)
        previous_end: Decimal | None = None
        for interval_id, start_value, end_value, text in intervals:
            annotation_class, markers = classify_alimeeting_text(
                text, active_inventory
            )
            class_counts[f"text:{annotation_class}"] += 1
            marker_counts.update(markers)
            start_time = _decimal(start_value, "AliMeeting interval xmin")
            end_time = _decimal(end_value, "AliMeeting interval xmax")
            if (
                start_time < timeline_start
                or end_time > timeline_end
                or end_time <= start_time
                or (previous_end is not None and start_time < previous_end)
            ):
                raise ReferenceNormalizationError(
                    f"AliMeeting interval exceeds scored range: {path.name}"
                )
            previous_end = end_time
            if annotation_class not in {
                "human_vocal_marker_only",
                "mixed_lexical_human_vocal",
            }:
                continue
            source_annotation_id = (
                f"{path.name}#item[{item_id}].intervals[{interval_id}]"
            )
            masks.append(
                _clipped_mask(
                    start_sample=timestamp_to_sample(start_time),
                    end_sample=timestamp_to_sample(end_time),
                    scored_start_sample=scored_start_sample,
                    scored_end_sample=scored_end_sample,
                    speaker_id=speaker_id,
                    source_annotation_id=source_annotation_id,
                    annotation_class=annotation_class,
                )
            )
    if found_speakers != set(speaker_map):
        raise ReferenceNormalizationError(
            f"AliMeeting TextGrid does not match official speakers: {source_id}"
        )
    return ParsedNonlexical(
        masks=tuple(sorted(masks, key=_mask_order)),
        observed_class_counts=dict(sorted(class_counts.items())),
        observed_marker_counts=dict(sorted(marker_counts.items())),
    )


def _mask_order(mask: AmbiguityMask) -> tuple[int, int, str, str, str]:
    return (
        mask.start_sample,
        mask.end_sample,
        mask.speaker_id,
        mask.source_annotation_id,
        mask.annotation_class,
    )


def compose_reference_timeline(
    reference_spans: Iterable[ReferenceSpan],
    ambiguity_masks: Iterable[AmbiguityMask],
    *,
    scored_start_sample: int,
    scored_end_sample: int,
) -> tuple[CanonicalInterval, ...]:
    if (
        isinstance(scored_start_sample, bool)
        or not isinstance(scored_start_sample, int)
        or isinstance(scored_end_sample, bool)
        or not isinstance(scored_end_sample, int)
        or scored_start_sample < 0
        or scored_end_sample <= scored_start_sample
    ):
        raise ReferenceNormalizationError("invalid scored reference range")
    spans = tuple(
        sorted(
            reference_spans,
            key=lambda span: (
                span.start_sample,
                span.end_sample,
                span.speaker_id,
                span.source_annotation_ids,
            ),
        )
    )
    masks = tuple(sorted(ambiguity_masks, key=_mask_order))
    if not spans:
        raise ReferenceNormalizationError("RTTM activity reference is empty")
    boundaries = {scored_start_sample, scored_end_sample}
    span_starts: dict[int, list[int]] = defaultdict(list)
    span_ends: dict[int, list[int]] = defaultdict(list)
    mask_starts: dict[int, list[int]] = defaultdict(list)
    mask_ends: dict[int, list[int]] = defaultdict(list)
    for index, span in enumerate(spans):
        if (
            span.start_sample < scored_start_sample
            or span.end_sample > scored_end_sample
            or span.end_sample <= span.start_sample
            or not span.speaker_id
            or not span.source_annotation_ids
        ):
            raise ReferenceNormalizationError("invalid canonical RTTM span")
        boundaries.update((span.start_sample, span.end_sample))
        span_starts[span.start_sample].append(index)
        span_ends[span.end_sample].append(index)
    for index, mask in enumerate(masks):
        if (
            mask.start_sample < scored_start_sample
            or mask.end_sample > scored_end_sample
            or mask.end_sample <= mask.start_sample
            or mask.mask_class != MASK_CLASS
            or not mask.speaker_id
            or not mask.source_annotation_id
        ):
            raise ReferenceNormalizationError("invalid nonlexical ambiguity mask")
        boundaries.update((mask.start_sample, mask.end_sample))
        mask_starts[mask.start_sample].append(index)
        mask_ends[mask.end_sample].append(index)
    ordered_boundaries = sorted(boundaries)
    active_spans: set[int] = set()
    active_masks: set[int] = set()
    intervals: list[CanonicalInterval] = []
    for position, boundary in enumerate(ordered_boundaries[:-1]):
        active_spans.difference_update(span_ends.get(boundary, ()))
        active_masks.difference_update(mask_ends.get(boundary, ()))
        active_spans.update(span_starts.get(boundary, ()))
        active_masks.update(mask_starts.get(boundary, ()))
        next_boundary = ordered_boundaries[position + 1]
        current_spans = [spans[index] for index in sorted(active_spans)]
        current_masks = [masks[index] for index in sorted(active_masks)]
        intervals.append(
            CanonicalInterval(
                start_sample=boundary,
                end_sample=next_boundary,
                active_speakers=tuple(
                    sorted({span.speaker_id for span in current_spans})
                ),
                source_annotation_ids=tuple(
                    sorted(
                        {
                            annotation_id
                            for span in current_spans
                            for annotation_id in span.source_annotation_ids
                        }
                    )
                ),
                handoff_relation_mask_classes=tuple(
                    sorted({mask.mask_class for mask in current_masks})
                ),
                mask_annotation_ids=tuple(
                    sorted({mask.source_annotation_id for mask in current_masks})
                ),
            )
        )
    active_spans.difference_update(span_ends.get(scored_end_sample, ()))
    active_masks.difference_update(mask_ends.get(scored_end_sample, ()))
    if active_spans or active_masks:
        raise ReferenceNormalizationError("reference sweep did not close at scored end")
    return normalize_intervals(
        intervals,
        scored_start_sample=scored_start_sample,
        scored_end_sample=scored_end_sample,
    )


def _normalize_resolved_reference_session(
    *,
    source_id: str,
    source_record_sha256: str,
    source_waveform_sha256: str,
    source_annotation_sha256: str,
    corpus: str,
    session_id: str,
    reference_path: Path,
    reference_checkout_provenance: Mapping[str, Any],
    corpus_root: Path,
    expected_speaker_ids: Sequence[str],
    scored_start_sample: int,
    scored_end_sample: int,
    meetings_path: Path | None = None,
    word_paths: Sequence[Path] = (),
    textgrid_path: Path | None = None,
    inventory: NonlexicalInventory | None = None,
) -> ReferenceNormalizedSession:
    active_inventory = inventory or load_nonlexical_inventory()
    _require_validated_inventory(active_inventory)
    if corpus == "AMI":
        if meetings_path is None or textgrid_path is not None or not word_paths:
            raise ReferenceNormalizationError("AMI reference metadata is incomplete")
        speaker_map = build_ami_speaker_map(meetings_path, session_id)
        metadata_files = (
            _metadata_file(meetings_path, "speaker_metadata", corpus_root),
            *(
                _metadata_file(path, "nonlexical_annotation", corpus_root)
                for path in sorted(word_paths)
            ),
        )
        parsed_nonlexical = parse_ami_nonlexical_masks(
            session_id,
            word_paths,
            speaker_map=speaker_map,
            scored_start_sample=scored_start_sample,
            scored_end_sample=scored_end_sample,
            inventory=active_inventory,
        )
    elif corpus == "AliMeeting":
        if textgrid_path is None or meetings_path is not None or word_paths:
            raise ReferenceNormalizationError(
                "AliMeeting reference metadata is incomplete"
            )
        if textgrid_path.stem != session_id:
            raise ReferenceNormalizationError(
                "AliMeeting TextGrid session identity mismatch"
            )
        speaker_map = build_alimeeting_speaker_map(
            alimeeting_speaker_ids(textgrid_path)
        )
        metadata_files = (
            _metadata_file(
                textgrid_path,
                "speaker_and_nonlexical_annotation",
                corpus_root,
            ),
        )
        parsed_nonlexical = parse_alimeeting_nonlexical_masks(
            f"alimeeting_{session_id}",
            textgrid_path,
            speaker_map=speaker_map,
            scored_start_sample=scored_start_sample,
            scored_end_sample=scored_end_sample,
            inventory=active_inventory,
        )
    else:
        raise ReferenceNormalizationError(f"unsupported reference corpus: {corpus}")
    if tuple(sorted(speaker_map.values())) != tuple(sorted(expected_speaker_ids)):
        raise ReferenceNormalizationError(
            f"official speaker inventory mismatch: {corpus}/{session_id}"
        )
    parsed_rttm = parse_rttm(
        reference_path,
        corpus=corpus,
        session_id=session_id,
        speaker_map=speaker_map,
        scored_start_sample=scored_start_sample,
        scored_end_sample=scored_end_sample,
    )
    intervals = compose_reference_timeline(
        parsed_rttm.spans,
        parsed_nonlexical.masks,
        scored_start_sample=scored_start_sample,
        scored_end_sample=scored_end_sample,
    )
    contract = load_contract(version="psem-handoff-v1")
    labels = generate_labels(
        intervals,
        contract=contract,
        scored_start_sample=scored_start_sample,
        scored_end_sample=scored_end_sample,
    )
    return ReferenceNormalizedSession(
        source_id=source_id,
        corpus=corpus,
        session_id=session_id,
        scored_start_sample=scored_start_sample,
        scored_end_sample=scored_end_sample,
        reference_path=reference_path,
        reference_sha256=sha256_file(reference_path),
        parsed_rttm=parsed_rttm,
        parsed_nonlexical=parsed_nonlexical,
        intervals=intervals,
        labels=labels,
        inventory_sha256=active_inventory.document_sha256,
        speaker_mapping_sha256=canonical_sha256(dict(sorted(speaker_map.items()))),
        metadata_files=metadata_files,
        source_record_sha256=source_record_sha256,
        source_waveform_sha256=source_waveform_sha256,
        source_annotation_sha256=source_annotation_sha256,
        reference_checkout_provenance=MappingProxyType(
            dict(reference_checkout_provenance)
        ),
    )


def _source_fields(
    source_row: Mapping[str, Any],
) -> tuple[str, str, str, str, str, int, tuple[str, ...], str, str]:
    source_id = source_row.get("source_id")
    corpus = source_row.get("corpus")
    session_id = source_row.get("session_id")
    waveform_sha256 = source_row.get("waveform_sha256")
    audio_ref = source_row.get("audio_ref")
    annotation_sha256 = source_row.get("annotation_sha256")
    duration_samples = source_row.get("duration_samples")
    sample_rate_hz = source_row.get("sample_rate_hz")
    speaker_ids = source_row.get("speaker_ids")
    annotation_ref = source_row.get("annotation_ref")
    source_prefix = {"AMI": "ami", "AliMeeting": "alimeeting"}.get(corpus)
    contract = load_contract()
    if (
        not isinstance(source_id, str)
        or not isinstance(session_id, str)
        or source_prefix is None
        or source_id != f"{source_prefix}_{session_id}"
        or not re.fullmatch(r"[A-Za-z0-9_]+", session_id)
        or not isinstance(waveform_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", waveform_sha256) is None
        or not isinstance(audio_ref, str)
        or not audio_ref
        or not isinstance(annotation_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", annotation_sha256) is None
        or not _is_exact_integer(duration_samples)
        or duration_samples <= 0
        or not _is_exact_integer(sample_rate_hz)
        or sample_rate_hz != SAMPLE_RATE_HZ
        or not _is_exact_integer(source_row.get("waveform_size_bytes"))
        or source_row.get("waveform_size_bytes") <= 0
        or not _is_exact_integer(source_row.get("channels"))
        or source_row.get("channels") != 1
        or not _is_exact_integer(source_row.get("sample_width_bytes"))
        or source_row.get("sample_width_bytes") != 2
        or not isinstance(speaker_ids, list)
        or not speaker_ids
        or any(
            not isinstance(speaker_id, str)
            or not speaker_id
            or speaker_id.strip() != speaker_id
            for speaker_id in speaker_ids
        )
        or len(set(speaker_ids)) != len(speaker_ids)
        or not isinstance(annotation_ref, str)
        or not annotation_ref
        or not _is_exact_integer(source_row.get("schema_version"))
        or source_row.get("schema_version") != 1
        or source_row.get("contract_version") != contract.contract_version
        or source_row.get("contract_document_sha256") != contract.document_sha256
    ):
        raise ReferenceNormalizationError("source manifest identity is invalid")
    return (
        source_id,
        corpus,
        session_id,
        waveform_sha256,
        audio_ref,
        duration_samples,
        tuple(speaker_ids),
        annotation_sha256,
        annotation_ref,
    )


def normalize_reference_session(
    source_row: Mapping[str, Any],
    corpus_root: Path,
    reference_checkout: PinnedReferenceCheckout,
) -> ReferenceNormalizedSession:
    _require_pinned_checkout(reference_checkout)
    (
        source_id,
        corpus,
        session_id,
        waveform_sha256,
        audio_ref,
        duration_samples,
        speaker_ids,
        annotation_sha256,
        annotation_ref,
    ) = _source_fields(source_row)
    root = corpus_root.resolve()
    if not root.is_dir():
        raise ReferenceNormalizationError("corpus root is unavailable")
    reference_path = resolve_reference_path(
        reference_checkout.root,
        corpus=corpus,
        session_id=session_id,
    )
    expected_audio_ref = (
        f"ami/audio/{session_id}/{session_id}.Mix-Headset.wav"
        if corpus == "AMI"
        else f"alimeeting/far_ch0/{session_id}.wav"
    )
    if audio_ref != expected_audio_ref:
        raise ReferenceNormalizationError("source waveform identity is invalid")
    audio_path = root / audio_ref
    _, audio_size_bytes = _bound_file_details(audio_path, root)
    audio_identity = wav_identity(audio_path)
    if (
        audio_identity["waveform_sha256"] != waveform_sha256
        or audio_identity["duration_samples"] != duration_samples
        or audio_identity["waveform_size_bytes"] != audio_size_bytes
        or source_row.get("waveform_size_bytes") != audio_identity["waveform_size_bytes"]
    ):
        raise ReferenceNormalizationError("source waveform receipt does not match corpus bytes")
    if corpus == "AMI":
        expected_annotation_ref = (
            f"ami/annotations/segments/{session_id}.*.segments.xml"
        )
        if annotation_ref != expected_annotation_ref:
            raise ReferenceNormalizationError("AMI annotation identity is invalid")
        scored_start_sample = 0
        scored_end_sample = duration_samples
        meetings_path = root / "ami" / "annotations" / "corpusResources" / "meetings.xml"
        segment_paths = tuple(
            sorted(
                (root / "ami" / "annotations" / "segments").glob(
                    f"{session_id}.*.segments.xml"
                )
            )
        )
        annotation_files = [
            _source_file_row(path, root) for path in (meetings_path, *segment_paths)
        ]
        word_paths = tuple(
            sorted(
                (root / "ami" / "annotations" / "words").glob(
                    f"{session_id}.*.words.xml"
                )
            )
        )
        textgrid_path = None
    else:
        annotation_path = (root / annotation_ref).resolve()
        if (
            not annotation_path.is_relative_to(root)
            or annotation_path.suffix != ".TextGrid"
            or annotation_path.stem != session_id
            or not _is_exact_integer(
                source_row.get("annotation_coverage_start_sample")
            )
            or source_row.get("annotation_coverage_start_sample") != 0
        ):
            raise ReferenceNormalizationError("AliMeeting annotation identity is invalid")
        scored_start_sample = 0
        scored_end_sample = source_row.get("annotation_coverage_end_sample")
        if (
            isinstance(scored_end_sample, bool)
            or not isinstance(scored_end_sample, int)
            or scored_end_sample <= 0
            or scored_end_sample > duration_samples
        ):
            raise ReferenceNormalizationError("AliMeeting scored range is invalid")
        meetings_path = None
        word_paths = ()
        textgrid_path = annotation_path
        annotation_files = [_source_file_row(textgrid_path, root)]
    if canonical_sha256(annotation_files) != annotation_sha256:
        raise ReferenceNormalizationError(
            "source annotation receipt does not match corpus bytes"
        )
    return _normalize_resolved_reference_session(
        source_id=source_id,
        source_record_sha256=canonical_sha256(dict(source_row)),
        source_waveform_sha256=waveform_sha256,
        source_annotation_sha256=annotation_sha256,
        corpus=corpus,
        session_id=session_id,
        reference_path=reference_path,
        reference_checkout_provenance=reference_checkout.provenance,
        corpus_root=root,
        expected_speaker_ids=speaker_ids,
        scored_start_sample=scored_start_sample,
        scored_end_sample=scored_end_sample,
        meetings_path=meetings_path,
        word_paths=word_paths,
        textgrid_path=textgrid_path,
        inventory=load_nonlexical_inventory(),
    )


def normalize_reference_inventory(
    source_manifest_path: Path,
    corpus_root: Path,
    reference_root: Path,
) -> tuple[ReferenceNormalizedSession, ...]:
    try:
        source_rows = [
            json.loads(line)
            for line in source_manifest_path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceNormalizationError("source manifest is invalid") from exc
    if not source_rows or any(not isinstance(row, dict) for row in source_rows):
        raise ReferenceNormalizationError("source manifest must contain JSON objects")
    by_source = {row.get("source_id"): row for row in source_rows}
    if len(by_source) != len(source_rows) or any(
        not isinstance(source_id, str) for source_id in by_source
    ):
        raise ReferenceNormalizationError("source manifest identities are duplicated")
    checkout = open_reference_checkout(reference_root)
    return tuple(
        normalize_reference_session(by_source[source_id], corpus_root, checkout)
        for source_id in sorted(by_source)
    )


def write_reference_normalization_manifest(
    source_manifest_path: Path,
    corpus_root: Path,
    reference_root: Path,
    output_path: Path,
) -> None:
    sessions = normalize_reference_inventory(
        source_manifest_path,
        corpus_root,
        reference_root,
    )
    write_jsonl(output_path, (session.manifest_row() for session in sessions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_reference_normalization_manifest(
        args.source_manifest.resolve(),
        args.corpus_root.resolve(),
        args.reference_root.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
