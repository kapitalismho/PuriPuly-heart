from __future__ import annotations

import argparse
import asyncio
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

from issue264_validation_runtime import ROOT, ObservedDesktop, local, prior, save, source_manifest

from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
from puripuly_heart.core.speaker_identity import PeerSpeakerIdentityAllocator
from tests.core.test_issue264_shared_speaker import admitted, frames


async def await_snapshot(presenter, predicate, *, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(presenter.snapshot()):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Expected caption transition not observed before bounded timeout")


def frozen_blocks(snapshot):
    return [block.to_dict() for block in snapshot.blocks]


def unchanged_except_style(before, after):
    ignored = {"speaker_style", "card_revision"}
    return len(before) == len(after) and all(
        {key: value for key, value in old.items() if key not in ignored}
        == {key: value for key, value in new.items() if key not in ignored}
        for old, new in zip(before, after)
    )


async def case_run(desktop, kind, order):
    record = {
        "case": kind,
        "status": "running",
        "clock": "actual SystemClock monotonic",
        "boundary": "Controlled synthetic metadata and selected terminal disposition; no acoustic/model/translation-service inference",
    }
    clock = prior.SystemClock()
    view = SpeakerEvidenceView()
    identities = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    presenter = local.OverlayPresenter(
        calibration=local.OverlayCalibration(),
        bridge=desktop,
        clock=clock,
        translation_enabled=False,
        show_peer_original=kind != "hidden_reveal",
    )
    output = local.OutputRuntime(
        chatbox=local.ChatboxGuard(),
        clock=clock,
        overlay_sink=presenter,
        activation_notice_enabled=False,
    )
    output.activate_peer_generation(1)
    output.bind_speaker_identities(identities)
    config = prior.TranslationRuntimeConfigSnapshot(
        0, prior.TranslationRuntimeConfig(peer_translation_enabled=False)
    )
    diagnostics = local.TranslationLatencyDiagnosticsOwner(
        clock=clock, config_snapshot=lambda: config
    )
    projection = local.TranslationOutputProjectionOwner(
        output_runtime=output,
        ui_messages=local.TranslationUiMessageQueue(asyncio.Queue(), output),
        diagnostics=diagnostics,
        clock=clock,
    )
    source, transcript, token = admitted(view, identities, order, 0, 1600)
    transcript = replace(
        transcript, created_at=clock.now(), publication_generation=1, source_order=order
    )
    starts = (len(desktop.snapshots), len(desktop.readbacks))

    def authority(reason):
        record.setdefault("assignment_observations", []).append(
            {
                "at": clock.now(),
                "reason": reason,
                "records": prior.serial(identities._sources),
                "assignment_revision": identities._revision,
                "evidence_revision": view.revision,
                "proofs": len(view._proofs),
            }
        )

    desktop.assignment_observer = authority
    delivery = None
    if kind in ("source_only_ttl", "translation_error_disposition", "hidden_reveal"):

        async def deliver_after_readable():
            deadline = clock.now() + 6.0
            while identities._sources[source.source_id].first_readable is None:
                if clock.now() >= deadline:
                    raise TimeoutError("Controlled source never became readable")
                await asyncio.sleep(0.005)
            first_readable = identities._sources[source.source_id].first_readable
            await asyncio.sleep(max(0, first_readable + 0.30 - clock.now()))
            record["proof_delivered_monotonic"] = clock.now()
            view.observe(frames(0, 1600, 0))

        delivery = asyncio.create_task(deliver_after_readable())
    try:
        if kind == "established_nonrecolor":
            view.observe(frames(0, 1600, 0))
        started = clock.now()
        record["source_output_started"] = started
        await projection.project_peer_source_only(
            transcript=transcript,
            source_language="en",
            target_language="ko",
            close_is_final=kind != "translation_error_disposition",
            finalize_latency=True,
            speaker_assignment=identities.assignment_for(transcript.utterance_id),
        )
        record["source_output_completed"] = clock.now()
        if kind == "hidden_reveal":
            assert presenter.snapshot().blocks == []
            assert identities.next_deadline() is None
            await asyncio.sleep(1.1)
            record["hidden_before_reveal"] = frozen_blocks(presenter.snapshot())
            await presenter.update_display_preferences(
                show_translation=False, show_peer_original=True
            )
        await await_snapshot(presenter, lambda snapshot: bool(snapshot.blocks))
        first_request = next(
            snapshot
            for snapshot in desktop.snapshots[starts[0] :]
            if snapshot["snapshot"]["blocks"]
        )
        first = first_request["snapshot"]["blocks"]
        record["first_snapshot"] = first
        record["first_readable_monotonic"] = identities._sources[source.source_id].first_readable
        entry = next(iter(presenter._entries.values()))
        expires = presenter._entry_expiration_deadline(entry)
        record["original_expiration"] = expires
        identities.retire(transcript.utterance_id)
        if kind not in ("established_nonrecolor",):
            assert first[0]["speaker_style"] == "gray"
            if delivery is None or not delivery.done():
                assert view.has_proof(token)
        if kind in ("deadline", "reset", "retirement"):
            if kind == "deadline":
                await asyncio.sleep(max(0, record["first_readable_monotonic"] + 1.1 - clock.now()))
            elif kind == "reset":
                identities.reset()
            else:
                identities.retire_source(source.source_id)
            view.observe(frames(0, 1600, 0))
            await asyncio.sleep(0.15)
            assert presenter.snapshot().blocks[0].speaker_style == "gray"
        elif kind == "established_nonrecolor":
            assert first[0]["speaker_style"] == "gold"
            view.observe(frames(1600, 3200, 1))
            await asyncio.sleep(0.15)
            assert frozen_blocks(presenter.snapshot()) == first
        else:
            await delivery
            await await_snapshot(
                presenter,
                lambda snapshot: bool(snapshot.blocks)
                and snapshot.blocks[0].speaker_style == "gold",
                timeout=1.5,
            )
            record["style_observed_monotonic"] = clock.now()
            style_request = next(
                snapshot
                for snapshot in desktop.snapshots[starts[0] :]
                if any(
                    block.get("speaker_style") == "gold" for block in snapshot["snapshot"]["blocks"]
                )
            )
            record["style_application_monotonic"] = style_request["submitted_at"]
            assert record["style_application_monotonic"] <= record["first_readable_monotonic"] + 1.0
            final = frozen_blocks(presenter.snapshot())
            assert unchanged_except_style(first, final)
            assert presenter._entry_expiration_deadline(entry) == expires
            assert not view.has_proof(token)
        record["final_snapshot"] = frozen_blocks(presenter.snapshot())
        record["expiration_after_metadata"] = presenter._entry_expiration_deadline(entry)
        record["ttl_conserved"] = record["expiration_after_metadata"] == expires
        assert record["ttl_conserved"]
        record["screenshot"] = await desktop.picture(f"controlled_{kind}_{order}")
        if kind == "source_only_ttl":
            record["ttl_wait_started_monotonic"] = clock.now()
            await asyncio.sleep(max(0, expires + 0.15 - clock.now()))
            await await_snapshot(presenter, lambda snapshot: not snapshot.blocks)
            record["expired_snapshot"] = frozen_blocks(presenter.snapshot())
            record["expiry_observed_monotonic"] = clock.now()
            view.observe(frames(1600, 3200, 1))
            await asyncio.sleep(0.05)
            assert presenter.snapshot().blocks == []
        authority("case_completed")
        record["status"] = "executed_assertions_passed"
    except Exception as error:
        record["status"] = "failed_observation"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
        authority("failed_observation")
    finally:
        if delivery is not None:
            if not delivery.done():
                delivery.cancel()
            await asyncio.gather(delivery, return_exceptions=True)
        await output.close()
        await presenter.close()
        if desktop.readback_tasks:
            await asyncio.gather(*desktop.readback_tasks)
        record["snapshots"] = desktop.snapshots[starts[0] :]
        record["readbacks"] = desktop.readbacks[starts[1] :]
        desktop.assignment_observer = None
        await desktop.replace_snapshot(local.OverlayPresentationSnapshot(revision=0, blocks=[]))
    return record


async def run(args):
    if os.environ.get("ISSUE264_VALIDATION_READY_SHA") != args.candidate:
        raise RuntimeError("Use isolated launcher only after Director stable-candidate readiness")
    desktop = ObservedDesktop(args.output)
    report = {
        "schema": "issue264-controlled-actual-caption-1",
        "candidate": args.candidate,
        "sources_sha256": source_manifest(),
        "scope": "Actual Flet source-only/error-disposition/deadline/nonrecolor/TTL/reset/retirement/hidden-reveal presentation with controlled synthetic metadata; not actual acoustics or translation execution",
        "cases": [],
        "cloud_calls": 0,
        "model_calls": 0,
        "translation_calls": 0,
    }
    try:
        await desktop.start()
        report["owned_renderer_pid"] = desktop.pid
        report["owned_backdrop_pid"] = desktop.backdrop_pid
        for index, kind in enumerate(
            (
                "source_only_ttl",
                "translation_error_disposition",
                "deadline",
                "established_nonrecolor",
                "reset",
                "retirement",
                "hidden_reveal",
            ),
            101,
        ):
            record = await case_run(desktop, kind, index)
            report["cases"].append(record)
            save(args.output / "caption.json", report)
            if record["status"] != "executed_assertions_passed":
                break
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    finally:
        await desktop.close()
        report["renderer_errors"] = desktop.errors
        report["status"] = (
            "executed"
            if len(report["cases"]) == 7
            and all(case["status"] == "executed_assertions_passed" for case in report["cases"])
            else "failed_observation"
        )
        save(args.output / "caption.json", report)
    return 0 if report["status"] == "executed" else 1


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.parent != ROOT / "build" or not args.output.name.startswith(
        "issue264-validation-"
    ):
        raise ValueError("Only assigned build resources are permitted")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(arguments())))
