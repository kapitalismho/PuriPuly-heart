from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.psem_r2_policy.budget import (
    deepgram_reserve_usd,
    load_billing_bounds,
    load_rates,
    openrouter_reserve_usd,
)
from experiments.psem_r2_policy.metrics import (
    DEV_CLUSTERS,
    HOLDOUT_CLUSTERS,
    aggregate_cluster_parents,
    confirmatory_decision,
    latency_by_operation,
    u8_phase_report,
)

EXP = Path(__file__).resolve().parent
PROTOCOL_PATH = EXP / "PROTOCOL.json"
GATE_PATH = EXP / "HOLD_OUT_GATE.json"
PIN_PATH = EXP / "PIN_MANIFEST.json"
RUNTIME_PIN_PATH = EXP / "RUNTIME_PIN.json"
ARTIFACTS = EXP / "artifacts"
PIN_REVISION = "PSEM-R2-FREEZE-MANIFEST-1"


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def load_holdout_gate() -> dict[str, Any]:
    return json.loads(GATE_PATH.read_text(encoding="utf-8"))


def declared_meetings(phase: str) -> tuple[str, ...]:
    protocol = load_protocol()
    if phase == "dev":
        return tuple(protocol["development"]["meetings"])
    if phase == "holdout":
        return tuple(protocol["holdout"]["meetings"])
    raise ValueError(f"unknown phase: {phase}")


def resolve_meetings(phase: str, meeting: str | None) -> tuple[str, ...]:
    declared = declared_meetings(phase)
    if meeting in {None, "", "all"}:
        return declared
    if meeting not in declared:
        raise ValueError(f"meeting {meeting} is not declared for {phase}")
    return (meeting,)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_holdout_inputs() -> tuple[dict[str, str | None], dict[str, str | None], list[str]]:
    from experiments.psem_r2_policy.live_runner import ami_wav_path
    from experiments.psem_r2_policy.metrics import AMI_WORDS, ROLES

    protocol = load_protocol()
    audio: dict[str, str | None] = {}
    annotations: dict[str, str | None] = {}
    missing: list[str] = []
    for meeting in protocol["holdout"]["meetings"]:
        try:
            wav = ami_wav_path(meeting)
        except FileNotFoundError:
            audio[meeting] = None
            missing.append(f"audio:{meeting}")
        else:
            audio[meeting] = _sha256_file(wav)
        for role in ROLES:
            path = AMI_WORDS / f"{meeting}.{role}.words.xml"
            key = f"{meeting}.{role}"
            if path.is_file():
                annotations[key] = _sha256_file(path)
            else:
                annotations[key] = None
                missing.append(f"words:{key}")
    return audio, annotations, missing


def _load_runtime_pin() -> dict[str, Any]:
    if not RUNTIME_PIN_PATH.is_file():
        raise FileNotFoundError(f"required runtime pin is missing: {RUNTIME_PIN_PATH}")
    try:
        return json.loads(RUNTIME_PIN_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"runtime pin is not valid JSON: {RUNTIME_PIN_PATH}: {exc}") from exc


def _capsule_root(runtime_pin: Mapping[str, Any]) -> Path | None:
    candidate = EXP.parents[1]
    expected = candidate / str(runtime_pin["capsule"]["package_dir"])
    manifest = candidate / str(runtime_pin["capsule"]["manifest"])
    if EXP.resolve() == expected.resolve() and manifest.is_file():
        return candidate
    return None


def _pin_bindings(runtime_pin: Mapping[str, Any]) -> dict[str, Path]:
    canonical = runtime_pin["canonical"]
    capsule_root = _capsule_root(runtime_pin)
    bindings: dict[str, Path] = {}

    def bind(logical: str, canonical_path: Path) -> None:
        if logical in bindings:
            raise ValueError(f"duplicate runtime binding: {logical}")
        bindings[logical] = (capsule_root / logical) if capsule_root is not None else canonical_path

    package_dir = str(runtime_pin["capsule"]["package_dir"])
    for name in canonical["harness_files"]:
        bind(f"{package_dir}/{name}", EXP / str(name))
    for name in canonical["config_files"]:
        if str(name) != "PIN_MANIFEST.json":
            bind(f"{package_dir}/{name}", EXP / str(name))
    for logical, source in canonical.get("runtime_overrides", {}).items():
        bind(str(logical), EXP / str(source))
    for source, logical in canonical["tests"].items():
        bind(str(logical), EXP / str(source))
    return bindings


def _runtime_identity(runtime_pin: Mapping[str, Any]) -> dict[str, Any]:
    archive_pin = runtime_pin["runtime_archive"]
    prompt_pin = runtime_pin["prompt"]
    capsule_root = _capsule_root(runtime_pin)
    if capsule_root is None:
        archive = EXP / str(archive_pin["file"])
        if not archive.is_file():
            raise FileNotFoundError(f"required runtime archive is missing: {archive}")
        archive_sha = _sha256_file(archive)
        if archive_sha != str(archive_pin["sha256"]):
            raise ValueError(
                f"runtime archive sha256 mismatch: {archive_sha} != {archive_pin['sha256']}"
            )
        try:
            with tarfile.open(archive, "r:gz") as handle:
                prompt_bytes = handle.extractfile(str(prompt_pin["file"])).read()
        except (KeyError, AttributeError, tarfile.TarError) as exc:
            raise ValueError(
                f"pinned prompt is missing from runtime archive: {prompt_pin['file']}"
            ) from exc
        prompt_file_sha = hashlib.sha256(prompt_bytes).hexdigest()
        prompt_text = prompt_bytes.decode("utf-8").replace("\r\n", "\n").strip()
        prompt_text_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    else:
        capsule_manifest_path = capsule_root / str(runtime_pin["capsule"]["manifest"])
        try:
            capsule_manifest = json.loads(capsule_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"capsule manifest is not valid JSON: {capsule_manifest_path}"
            ) from exc
        carried_archive = capsule_manifest.get("runtime_archive")
        expected_archive = {
            "file": archive_pin["file"],
            "sha256": archive_pin["sha256"],
            "commit": archive_pin["commit"],
        }
        if carried_archive != expected_archive:
            raise ValueError("capsule runtime archive identity does not match RUNTIME_PIN.json")
        archive_sha = str(carried_archive["sha256"])
        prompt = capsule_root / str(prompt_pin["file"])
        if not prompt.is_file():
            raise FileNotFoundError(f"required capsule prompt is missing: {prompt}")
        prompt_file_sha = _sha256_file(prompt)
        prompt_text = prompt.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        prompt_text_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    if prompt_file_sha != str(prompt_pin["file_sha256"]):
        raise ValueError(
            f"runtime prompt file sha256 mismatch: {prompt_file_sha} != {prompt_pin['file_sha256']}"
        )
    if prompt_text_sha != str(prompt_pin["stripped_sha256"]):
        raise ValueError(
            f"runtime prompt text sha256 mismatch: {prompt_text_sha} != {prompt_pin['stripped_sha256']}"
        )
    return {
        "runtime_pin_sha256": _sha256_file(RUNTIME_PIN_PATH),
        "revision": runtime_pin["revision"],
        "archive": {
            "file": archive_pin["file"],
            "sha256": archive_sha,
            "commit": archive_pin["commit"],
        },
        "prompt": {
            "file": prompt_pin["file"],
            "file_sha256": prompt_file_sha,
            "stripped_sha256": prompt_text_sha,
        },
    }


def build_pin_manifest() -> dict[str, Any]:
    runtime_pin = _load_runtime_pin()
    bindings = _pin_bindings(runtime_pin)
    missing_files = [logical for logical, path in bindings.items() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"required frozen runtime files are missing: {', '.join(sorted(missing_files))}"
        )
    protocol = load_protocol()
    audio, annotations, missing_inputs = _hash_holdout_inputs()
    return {
        "revision": PIN_REVISION,
        "protocol_revision": protocol["revision"],
        "guard_revision": protocol["guard_oracle"]["revision"],
        "runtime": _runtime_identity(runtime_pin),
        "files": {logical: _sha256_file(path) for logical, path in sorted(bindings.items())},
        "audio": audio,
        "annotations": annotations,
        "missing_inputs": missing_inputs,
        "speaker_clusters": {
            "dev": {cluster: list(meetings) for cluster, meetings in DEV_CLUSTERS.items()},
            "holdout": {cluster: list(meetings) for cluster, meetings in HOLDOUT_CLUSTERS.items()},
        },
    }


def write_pin_manifest(path: Path | None = None) -> dict[str, Any]:
    payload = build_pin_manifest()
    if payload["missing_inputs"]:
        raise FileNotFoundError(
            f"required holdout inputs are missing: {', '.join(sorted(payload['missing_inputs']))}"
        )
    target = path or PIN_PATH
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def holdout_unlock_error() -> str | None:
    gate = load_holdout_gate()
    if not gate.get("frozen"):
        return "holdout is locked until Director freeze"
    if not PIN_PATH.is_file():
        return "holdout pin manifest is missing"
    try:
        pinned = json.loads(PIN_PATH.read_text(encoding="utf-8"))
        current = build_pin_manifest()
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return f"holdout pin integrity check failed: {exc}"
    if current["missing_inputs"]:
        return "holdout pin is missing audio or annotation files"
    if pinned != current:
        return "holdout pin manifest does not match current frozen runtime and inputs"
    return None


def case_output_path(
    phase: str,
    meeting: str,
    *,
    stamp: str | None = None,
    directory: Path | None = None,
) -> Path:
    token = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = (directory or ARTIFACTS) / phase / meeting
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{token}.json"
    if path.exists():
        raise FileExistsError(path)
    return path


def write_case_output(
    phase: str,
    meeting: str,
    payload: Mapping[str, Any],
    *,
    directory: Path | None = None,
) -> dict[str, str]:
    path = case_output_path(phase, meeting, directory=directory)
    encoded = json.dumps(payload, indent=1, ensure_ascii=False).encode("utf-8")
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    return {"path": str(path), "sha256": digest}


PARENT_RATE_PER_SECOND = 0.05
REQUESTS_PER_PARENT = 3


def _meeting_seconds(meeting: str) -> float:
    import wave

    from experiments.psem_r2_policy.live_runner import ami_wav_path

    with wave.open(str(ami_wav_path(meeting)), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def openrouter_request_bound_usd() -> float:
    """Pinned per-request allowance for the fixed R2 prompt and output bound."""
    import json

    from experiments.psem_r2_policy.arms import r2_rendered_system_prompt
    from experiments.psem_r2_policy.live_runner import PINNED_TRANSLATION
    from puripuly_heart.providers.llm.openrouter import HttpxOpenRouterClient

    client = HttpxOpenRouterClient(api_key="planning", model=PINNED_TRANSLATION, max_tokens=100)
    body = client._build_request_body(
        text="A twenty word utterance of ordinary meeting speech here to size a request.",
        system_prompt=r2_rendered_system_prompt(),
        source_language="en",
        target_language="ko",
        context="",
        scene_participant_count=None,
    )
    return openrouter_reserve_usd(
        serialized_request=json.dumps(body, ensure_ascii=False), max_tokens=100
    )


def phase_plan(
    *,
    parent_rate_per_second: float = PARENT_RATE_PER_SECOND,
    requests_per_parent: int = REQUESTS_PER_PARENT,
) -> dict[str, Any]:
    """Cash (OpenRouter) reservation arithmetic per phase.

    Deepgram usage is credit-funded per U7: it is reported as an informational
    estimate and never gates phase or combined cash fits.
    """
    bounds = load_billing_bounds()
    rates = load_rates()
    per_request = openrouter_request_bound_usd()
    deepgram_rate = float(rates["deepgram"]["usd_per_minute"])
    credit_exempt = bool(bounds["deepgram"].get("credit_exempt", False))
    phases: dict[str, dict[str, Any]] = {}
    combined_seconds = 0.0
    combined_deepgram = 0.0
    combined_openrouter = 0.0
    for phase in ("dev", "holdout"):
        declared = declared_meetings(phase)
        seconds = sum(_meeting_seconds(meeting) for meeting in declared)
        deepgram = sum(
            deepgram_reserve_usd(
                max_audio_seconds=max(_meeting_seconds(meeting), 0.001),
                hangover_seconds=0.8,
                preroll_seconds=0.5,
                tail_seconds=512.0 / 16000.0,
                copies=1,
                reconnect_bound=0,
            )
            for meeting in declared
        )
        parents = seconds * float(parent_rate_per_second)
        openrouter = parents * float(requests_per_parent) * per_request
        cap = float(bounds["phase_caps_usd"][phase])
        headroom = cap - openrouter
        phases[phase] = {
            "meetings": list(declared),
            "audio_seconds": seconds,
            "deepgram_one_pass_usd": deepgram,
            "deepgram_credit_exempt": credit_exempt,
            "openrouter_allowance_usd": openrouter,
            "cash_total_usd": openrouter,
            "phase_cap_usd": cap,
            "fits": openrouter <= cap + 1e-9,
            "openrouter_headroom_usd": headroom,
            "requests_that_fit": int(max(headroom, 0.0) // per_request),
            "nominal_parents_within_headroom": int(
                max(headroom, 0.0) // (per_request * float(requests_per_parent))
            ),
        }
        combined_seconds += seconds
        combined_deepgram += deepgram
        combined_openrouter += openrouter
    contingency = float(bounds["phase_caps_usd"]["contingency"])
    combined_cap = float(bounds["combined_hard_cap_usd"])
    combined = {
        "audio_seconds": combined_seconds,
        "deepgram_one_pass_usd": combined_deepgram,
        "deepgram_credit_exempt": credit_exempt,
        "openrouter_allowance_usd": combined_openrouter,
        "contingency_usd": contingency,
        "cash_total_with_contingency_usd": combined_openrouter + contingency,
        "combined_cap_usd": combined_cap,
        "fits": (combined_openrouter + contingency) <= combined_cap + 1e-9,
    }
    return {
        "revision": rates.get("revision"),
        "cash_scope": "openrouter",
        "deepgram_credit_exempt": credit_exempt,
        "deepgram_rate_usd_per_minute": deepgram_rate,
        "deepgram_rate_tier": str(bounds["deepgram"].get("rate_tier")),
        "openrouter_request_bound_usd": per_request,
        "parent_rate_per_second": float(parent_rate_per_second),
        "requests_per_parent": int(requests_per_parent),
        "phases": phases,
        "combined": combined,
    }


def _leave_one_cluster_out(
    cluster_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    eligible = [row for row in cluster_rows if row.get("eligible")]
    out: list[dict[str, Any]] = []
    for index, dropped in enumerate(eligible):
        rest = eligible[:index] + eligible[index + 1 :]
        deltas = [float(row["delta"]) for row in rest if row.get("delta") is not None]
        out.append(
            {
                "dropped_cluster_id": dropped.get("cluster_id"),
                "dropped_delta": dropped.get("delta"),
                "n_clusters": len(deltas),
                "mean": (sum(deltas) / len(deltas)) if deltas else None,
            }
        )
    return out


def aggregate_phase(
    parents: Sequence[Mapping[str, Any]],
    *,
    marks: Sequence[Mapping[str, float | None]] | None = None,
    cases: Sequence[Mapping[str, Any]] | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    clustered = aggregate_cluster_parents(parents)
    rows = clustered["cluster_rows"]
    deltas = [
        float(row["delta"]) for row in rows if row.get("eligible") and row.get("delta") is not None
    ]
    primary_mean = (sum(deltas) / len(deltas)) if deltas else None
    u8 = u8_phase_report(
        parents=parents,
        cases=list(cases or ()),
        phase=phase,
        primary_mean=primary_mean,
    )
    sensitivity = dict(u8["sensitivity"])
    sensitivity["leave_one_cluster_out"] = _leave_one_cluster_out(rows)
    decision = confirmatory_decision(
        cluster_rows=rows,
        safety_failures=u8["safety_failures"],
        coverage=clustered.get("coverage"),
        evaluation=u8,
        sensitivity=sensitivity,
    )
    return {
        "cluster_aggregate": clustered,
        "confirmatory": decision,
        "latency_by_operation": latency_by_operation(marks or ()),
        "u8": {**u8, "sensitivity": sensitivity},
        "n_parents": len(parents),
        "n_operationally_unsuccessful": clustered["n_operationally_unsuccessful"],
        "n_degraded_conditional": clustered["n_degraded_conditional"],
        "n_incomplete_source_parents": clustered["n_incomplete_source_parents"],
        "operational_clean": clustered["coverage"]["operational_clean"],
        "unsuccessful_parents": clustered["unsuccessful_parents"],
        "degraded_parents": clustered["degraded_parents"],
        "pool_exclusions": clustered["pool_exclusions"],
        "operational_census": u8["operational_census"],
        "execution_completed": u8["execution_completed"],
        "evaluation_valid": u8["evaluation_valid"],
        "execution_incomplete_reasons": u8["execution_incomplete_reasons"],
        "evaluation_invalid_reasons": u8["evaluation_invalid_reasons"],
        "excluded_history": [dict(item) for item in EXCLUDED_FROM_DATASET],
    }


EXCLUDED_FROM_DATASET = (
    {
        "name": "first_invalid_clock_dev",
        "artifact": "FIRST_DEV_RESULT.json",
        "reason": "invalid clock; excluded from the DEV dataset and retained as history",
    },
    {
        "name": "sixty_second_readiness_probe",
        "artifact": "C:/tmp/psem_r2_readiness_ES2009a_60s/probe-payload.json",
        "reason": "engineering readiness probe; excluded from the DEV dataset and retained as history",
    },
)
