from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import puripuly_heart.main as main_module
from puripuly_heart.core import desktop_overlay_repro_artifacts as artifacts
from puripuly_heart.ui import desktop_overlay_repro as repro


def _record(cycle: int, revision: int, disposition: str) -> dict[str, object]:
    visual = artifacts.expected_repro_visual_state(revision)
    return {
        "schema_version": 1,
        "record_type": "revision_outcome",
        "cycle": cycle,
        "wall_clock_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "monotonic_ms": revision,
        "synthetic_revision": revision,
        "expected_disposition": disposition,
        "actual_disposition": disposition,
        "render_commit_acknowledged": disposition == "committed",
        **visual,
    }


def _certifying_records(cycles: int = 100) -> list[dict[str, object]]:
    return [
        _record(((item.revision - 1) // 17) + 1, item.revision, item.expected_disposition)
        for cycle in range(1, cycles + 1)
        for batch in repro.normative_repro_schedule(cycle)
        for item in batch.revisions
    ]


def _write_certifying_artifacts(directory: Path, cycles: int = 100) -> None:
    records = _certifying_records(cycles)
    result = repro._result_record(
        outcome="completed",
        reason=None,
        cycles_requested=cycles,
        cycles_completed=cycles,
        records=records,
        renderer_shutdown_completed=True,
        bridge_shutdown_completed=True,
        backdrop_shutdown_completed=True,
    )
    artifacts.write_repro_artifacts(directory, records, result)
    (directory / artifacts.REPRO_CAPTURE_NAME).write_bytes(b"capture")


def test_normative_schedule_is_base_adjusted_fifo_with_exact_dispositions() -> None:
    first = [item for batch in repro.normative_repro_schedule(1) for item in batch.revisions]
    second = [item for batch in repro.normative_repro_schedule(2) for item in batch.revisions]

    assert [(item.revision, item.expected_disposition) for item in first] == [
        (1, "committed"),
        (2, "committed"),
        (3, "committed"),
        (4, "committed"),
        (10, "superseded"),
        (9, "stale"),
        (11, "committed"),
        (12, "superseded"),
        (13, "superseded"),
        (14, "committed"),
        (15, "committed"),
        (16, "committed"),
        (17, "committed"),
    ]
    assert [item.revision for item in second] == [item.revision + 17 for item in first]


@pytest.mark.parametrize("mutation", ["reordered", "duplicate", "missing_ack"])
def test_success_record_prevalidation_rejects_noncertifying_order_or_acknowledgement(
    mutation: str,
) -> None:
    records = _certifying_records(1)
    if mutation == "reordered":
        records[0], records[1] = records[1], records[0]
    elif mutation == "duplicate":
        records[1] = dict(records[0])
    else:
        committed = next(
            record for record in records if record["actual_disposition"] == "committed"
        )
        committed["render_commit_acknowledged"] = False

    with pytest.raises(artifacts.ReproArtifactError):
        artifacts.validate_repro_run_records(records, cycles=1)


def test_preflight_creates_absent_directory_and_rejects_nonempty_without_artifacts(
    tmp_path: Path,
) -> None:
    absent = tmp_path / "absent"
    repro.preflight_repro_arguments(repro.ReproArguments(output_dir=absent))

    assert absent.is_dir()
    assert list(absent.iterdir()) == []

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "existing.txt").write_text("x", encoding="utf-8")

    assert repro.run_desktop_overlay_repro(output_dir=blocked) == 2
    assert not (blocked / artifacts.REPRO_JSONL_NAME).exists()
    assert not (blocked / artifacts.REPRO_RESULT_NAME).exists()


@pytest.mark.parametrize(
    ("cycles", "dwell_ms"),
    [(0, 150), (1001, 150), (100, 0), (100, 10001)],
)
def test_invalid_numeric_preflight_writes_no_artifacts(
    tmp_path: Path,
    cycles: int,
    dwell_ms: int,
) -> None:
    output = tmp_path / f"invalid-{cycles}-{dwell_ms}"

    assert (
        repro.run_desktop_overlay_repro(
            cycles=cycles,
            dwell_ms=dwell_ms,
            output_dir=output,
        )
        == 2
    )
    assert not (output / artifacts.REPRO_JSONL_NAME).exists()
    assert not (output / artifacts.REPRO_RESULT_NAME).exists()


def test_unwritable_preflight_writes_no_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "unwritable"
    original_write_bytes = Path.write_bytes

    def reject_probe(path: Path, data: bytes) -> int:
        if path.name == ".desktop-overlay-repro-write-probe":
            raise OSError("unwritable")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", reject_probe)

    assert repro.run_desktop_overlay_repro(output_dir=output) == 2
    assert not (output / artifacts.REPRO_JSONL_NAME).exists()
    assert not (output / artifacts.REPRO_RESULT_NAME).exists()


def test_artifact_validator_accepts_only_complete_certifying_artifacts(tmp_path: Path) -> None:
    _write_certifying_artifacts(tmp_path)

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 0

    payload = json.loads(
        (tmp_path / artifacts.REPRO_JSONL_NAME).read_text(encoding="utf-8").splitlines()[0]
    )
    payload["unexpected"] = "unsafe"
    lines = (tmp_path / artifacts.REPRO_JSONL_NAME).read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(payload)
    (tmp_path / artifacts.REPRO_JSONL_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1


def test_artifact_validator_rejects_tampered_safe_visual_facts_and_returns_capture_metadata(
    tmp_path: Path,
) -> None:
    _write_certifying_artifacts(tmp_path)

    metadata = artifacts.validate_repro_artifacts(tmp_path)

    assert metadata == {
        "capture_name": artifacts.REPRO_CAPTURE_NAME,
        "capture_present": True,
        "capture_nonempty": True,
    }
    lines = (tmp_path / artifacts.REPRO_JSONL_NAME).read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["window_width"] = 1
    lines[0] = json.dumps(record)
    (tmp_path / artifacts.REPRO_JSONL_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1


def test_artifact_validator_rejects_short_runs_and_empty_capture(tmp_path: Path) -> None:
    _write_certifying_artifacts(tmp_path, cycles=99)

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1

    _write_certifying_artifacts(tmp_path, cycles=100)
    (tmp_path / artifacts.REPRO_CAPTURE_NAME).write_bytes(b"")

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1


def test_artifact_validator_rejects_nested_missing_reordered_and_failed_artifacts(
    tmp_path: Path,
) -> None:
    _write_certifying_artifacts(tmp_path)
    lines = (tmp_path / artifacts.REPRO_JSONL_NAME).read_text(encoding="utf-8").splitlines()
    nested = json.loads(lines[0])
    nested["slot_count"] = {"value": 0}
    lines[0] = json.dumps(nested)
    (tmp_path / artifacts.REPRO_JSONL_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1

    _write_certifying_artifacts(tmp_path)
    lines = (tmp_path / artifacts.REPRO_JSONL_NAME).read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    (tmp_path / artifacts.REPRO_JSONL_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1

    _write_certifying_artifacts(tmp_path)
    result = json.loads((tmp_path / artifacts.REPRO_RESULT_NAME).read_text(encoding="utf-8"))
    result["outcome"] = "failed"
    result["reason"] = "render_failed"
    (tmp_path / artifacts.REPRO_RESULT_NAME).write_text(json.dumps(result), encoding="utf-8")

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1

    (tmp_path / artifacts.REPRO_JSONL_NAME).unlink()

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1


def test_verifier_is_artifact_only_and_main_dispatches_both_diagnostic_routes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        main_module,
        "_run_desktop_overlay_repro",
        lambda **kwargs: calls.append(("run", kwargs)) or 0,
    )
    monkeypatch.setattr(
        main_module,
        "_verify_desktop_overlay_repro",
        lambda **kwargs: calls.append(("verify", kwargs)) or 0,
    )

    assert main_module.main(["run-desktop-overlay-repro", "--output-dir", str(tmp_path)]) == 0
    assert main_module.main(["verify-desktop-overlay-repro", "--output-dir", str(tmp_path)]) == 0
    assert calls == [
        ("run", {"cycles": 100, "dwell_ms": 150, "output_dir": tmp_path}),
        ("verify", {"output_dir": tmp_path}),
    ]


@pytest.mark.parametrize("invalid", [True, 1.0, [], {}])
def test_verifier_rejects_noninteger_or_nonscalar_schema_and_enums(
    tmp_path: Path,
    invalid: object,
) -> None:
    _write_certifying_artifacts(tmp_path)
    lines = (tmp_path / artifacts.REPRO_JSONL_NAME).read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["schema_version"] = invalid
    lines[0] = json.dumps(record)
    (tmp_path / artifacts.REPRO_JSONL_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1

    _write_certifying_artifacts(tmp_path)
    lines = (tmp_path / artifacts.REPRO_JSONL_NAME).read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["actual_disposition"] = invalid
    lines[0] = json.dumps(record)
    (tmp_path / artifacts.REPRO_JSONL_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1

    _write_certifying_artifacts(tmp_path)
    result = json.loads((tmp_path / artifacts.REPRO_RESULT_NAME).read_text(encoding="utf-8"))
    result["reason"] = invalid
    result["outcome"] = "failed"
    (tmp_path / artifacts.REPRO_RESULT_NAME).write_text(json.dumps(result), encoding="utf-8")

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1


def test_diagnostic_main_dispatch_bypasses_main_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        main_module,
        "configure_main_logging",
        lambda: pytest.fail("diagnostic commands must not initialize main logging"),
    )
    monkeypatch.setattr(main_module, "_verify_desktop_overlay_repro", lambda **_kwargs: 0)

    assert main_module.main(["verify-desktop-overlay-repro", "--output-dir", str(tmp_path)]) == 0


@pytest.mark.parametrize(
    "wall_clock",
    ["2026-02-30T00:00:00Z", "2026-01-01T24:00:00Z", "2026-01-01 00:00:00Z"],
)
def test_verifier_rejects_semantically_invalid_rfc3339_utc(tmp_path: Path, wall_clock: str) -> None:
    _write_certifying_artifacts(tmp_path)
    lines = (tmp_path / artifacts.REPRO_JSONL_NAME).read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["wall_clock_utc"] = wall_clock
    lines[0] = json.dumps(record)
    (tmp_path / artifacts.REPRO_JSONL_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert artifacts.verify_desktop_overlay_repro(output_dir=tmp_path) == 1


def test_raw_ingress_requires_current_speaker_transition_capability() -> None:
    ingress = repro.LocalAuthenticatedRawIngress(
        "token",
        repro.OverlayPresentationSnapshot(),
        "overlay-repro",
    )
    base = {
        "type": "auth",
        "session_token": "token",
        "contract_version": repro.OVERLAY_CONTRACT_VERSION,
        "overlay_instance_id": "overlay-repro",
        "runtime_generation": 1,
        "capabilities": {
            "execution_contract": repro.OVERLAY_EXECUTION_CONTRACT,
            "speaker_transition_presentation": repro.OVERLAY_SPEAKER_TRANSITION_CONTRACT,
        },
    }

    assert ingress._is_matched_desktop_auth(base)
    missing = {
        **base,
        "capabilities": {
            "execution_contract": repro.OVERLAY_EXECUTION_CONTRACT,
        },
    }
    retired = {
        **base,
        "capabilities": {
            "execution_contract": repro.OVERLAY_EXECUTION_CONTRACT,
            "speaker_transition_presentation": {
                "version": 1,
                "modes": ["A", "C", "E"],
            },
        },
    }
    assert not ingress._is_matched_desktop_auth(missing)
    assert not ingress._is_matched_desktop_auth(retired)


def test_owner_drives_raw_authenticated_fifo_gate_through_shipping_renderer(tmp_path: Path) -> None:
    from puripuly_heart.ui import desktop_overlay as desktop_module

    class FakeBackdrop:
        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class FakeWindow:
        def __init__(self) -> None:
            self.closed = asyncio.Event()
            self.state = {
                "slot_count": 0,
                "line_count": 0,
                "surface_visible": False,
                "interaction_mode": "edit",
                "window_width": 1344,
                "window_height": 336,
            }

        async def start(self, snapshot) -> None:
            await self.dispatch_snapshot(snapshot)

        async def run_until_closed(self) -> None:
            await self.closed.wait()

        async def close(self) -> None:
            self.closed.set()

        async def dispatch_snapshot(self, snapshot) -> None:
            self.state.update(
                {
                    "slot_count": min(2, len(snapshot.blocks)),
                    "line_count": min(
                        6,
                        sum(1 + int(bool(block.secondary_text)) for block in snapshot.blocks),
                    ),
                    "surface_visible": bool(snapshot.blocks),
                }
            )

        async def dispatch_runtime_control(self, payload) -> None:
            if payload.get("mode") == "pass_through":
                self.state["interaction_mode"] = "locked"

        async def advance_snapshot_history(self, snapshot) -> None:
            _ = snapshot

        def renderer_visual_state(self) -> dict[str, object]:
            return dict(self.state)

        def renderer_visual_state_for_snapshot(self, snapshot) -> dict[str, object]:
            state = dict(self.state)
            state["slot_count"] = min(2, len(snapshot.blocks))
            state["line_count"] = min(
                6,
                sum(1 + int(bool(block.secondary_text)) for block in snapshot.blocks),
            )
            state["surface_visible"] = bool(snapshot.blocks)
            return state

    async def run() -> tuple[int, repro.DesktopOverlayReproOwner]:
        window = FakeWindow()

        def factory(manifest, **kwargs):
            return desktop_module.DesktopOverlayRenderer(manifest, window=window, **kwargs)

        arguments = repro.ReproArguments(cycles=1, dwell_ms=1, output_dir=tmp_path)
        repro.preflight_repro_arguments(arguments)
        owner = repro.DesktopOverlayReproOwner(
            arguments,
            backdrop_factory=FakeBackdrop,
            renderer_factory=factory,
        )
        return await owner.run(), owner

    exit_code, owner = asyncio.run(run())

    assert exit_code == 0
    assert [
        (record["synthetic_revision"], record["actual_disposition"]) for record in owner._records
    ][4:7] == [
        (10, "superseded"),
        (9, "stale"),
        (11, "committed"),
    ]
    assert len(owner._records) == 13
    assert all(
        record["render_commit_acknowledged"] is (record["actual_disposition"] == "committed")
        for record in owner._records
    )


@pytest.mark.parametrize("failure", ["ingress_start", "backdrop_start"])
def test_owner_persists_classified_startup_failure_artifacts(tmp_path: Path, failure: str) -> None:
    class FailingIngress:
        async def start(self) -> str:
            raise OSError("ingress")

        async def close(self) -> None:
            return None

    class FailingBackdrop:
        async def start(self) -> None:
            if failure == "backdrop_start":
                raise OSError("backdrop")

        async def close(self) -> None:
            return None

    async def run() -> int:
        arguments = repro.ReproArguments(cycles=1, dwell_ms=1, output_dir=tmp_path)
        repro.preflight_repro_arguments(arguments)
        owner = repro.DesktopOverlayReproOwner(
            arguments,
            backdrop_factory=FailingBackdrop,
            ingress_factory=lambda *_args: (
                FailingIngress()
                if failure == "ingress_start"
                else repro.LocalAuthenticatedRawIngress(*_args)
            ),
        )
        return await owner.run()

    assert asyncio.run(run()) == 1
    result = json.loads((tmp_path / artifacts.REPRO_RESULT_NAME).read_text(encoding="utf-8"))
    assert result["outcome"] == "failed"
    assert result["reason"] == ("bridge_failed" if failure == "ingress_start" else "startup_failed")
