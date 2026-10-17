from __future__ import annotations

import argparse
import asyncio
import atexit
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import flet as ft

from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks, SessionRuntimeLoggingService
from puripuly_heart.domain.events import UIEventType
from puripuly_heart.ui.i18n import set_locale, t
from puripuly_heart.ui.theme import COLOR_BACKGROUND, COLOR_ON_BACKGROUND
from puripuly_heart.ui.views.dashboard import DashboardView
from puripuly_heart.ui.views.logs import LogsView
from tests.fixtures.self_grace_233.gui.owner_presentation import (
    ControlledSourceOverlay,
    OwnerPresentation,
    eventually,
    independent_recognition,
)

TITLE = "SELF-STREAM-2 owner presentation repair proof"
STAGES = (
    "progressive-source",
    "progressive-preview-one",
    "progressive-extended-source",
    "progressive-preview-two",
    "progressive-final",
    "older-speech-held",
    "manual-complete",
    "late-speech-final",
    "successor-speech",
    "dual-provisional",
    "dual-secondary-first",
    "dual-final-after-snapshots",
    "dual-final-before-snapshots",
    "partial-primary-preview",
    "partial-target-failure",
    "older-manual-held",
    "newer-speech-complete",
    "late-manual-final",
    "future-manual-complete",
    "queued-legacy-source-final",
    "pending-manual-source",
    "late-manual-while-source-pending",
    "pending-manual-admitted",
    "segmented-old-first-held",
    "segmented-new-managed",
    "segmented-old-released",
    "segmented-future-independent",
    "segmented-future-managed",
    "pressure-old-speech",
    "manual-source-timeout",
    "manual-result-after-source-timeout",
    "pressure-successor-managed",
)
MODULES = (
    "puripuly_heart.domain.events",
    "puripuly_heart.ui.event_mapping",
    "puripuly_heart.ui.event_projection",
    "puripuly_heart.ui.event_dispatch",
    "puripuly_heart.ui.presentation_adapter",
    "puripuly_heart.ui.components.display_card",
    "puripuly_heart.ui.views.dashboard",
    "puripuly_heart.ui.views.logs",
    "puripuly_heart.ui.logs.contract",
    "puripuly_heart.core.runtime_logging",
    "puripuly_heart.core.orchestrator.self_translation_channel",
    "puripuly_heart.core.orchestrator.translation_turn",
    "puripuly_heart.core.orchestrator.translation_request",
    "puripuly_heart.core.orchestrator.translation_output_projection",
    "puripuly_heart.core.runtime.output",
    "puripuly_heart.core.stt.backend",
    "puripuly_heart.core.stt.recognition_units",
    "puripuly_heart.domain.recognition",
    "tests.helpers.translation_owners",
    "tests.fixtures.self_grace_233.gui.owner_presentation",
)


async def main(page, *, report_dir, native):
    set_locale("en")
    page.title = TITLE
    page.bgcolor = COLOR_BACKGROUND
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    if native:
        page.window.width = 1580
        page.window.height = 1000
        page.window.left = 0
        page.window.top = 0
    paths = {name: Path(sys.modules[name].__file__).resolve() for name in MODULES}
    paths["task_probe.run_owner_presentation"] = Path(__file__).resolve()

    def fingerprints():
        return {
            name: {
                "path": path.relative_to(Path.cwd()).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        }

    def product_manifest():
        paths = sorted((*Path("src").rglob("*.py"), *Path("src").rglob("*.json")))
        files = [
            {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ]
        digest = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "file_count": len(files),
            "python_file_count": sum(path.suffix == ".py" for path in paths),
            "json_file_count": sum(path.suffix == ".json" for path in paths),
            "sha256": digest,
            "files": files,
        }

    entry = fingerprints()
    entry_manifest = product_manifest()
    (report_dir / "source_manifest.json").write_text(
        json.dumps(entry_manifest, indent=2), encoding="utf-8"
    )
    (report_dir / "instance.json").write_text(
        json.dumps(
            {
                "python_pid": os.getpid(),
                "python": Path(sys.executable).resolve().relative_to(Path.cwd()).as_posix(),
                "native": native,
                "candidate_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "source_fingerprints": entry,
                "product_source_manifest_sha256": entry_manifest["sha256"],
                "product_python_files": entry_manifest["python_file_count"],
                "product_json_files": entry_manifest["json_file_count"],
                "product_source_files": entry_manifest["file_count"],
                "boundary": "Real Self/turn/request/output owners, bounded UI bridge, native production views and accepted runtime logger; synthetic STT events and controlled local provider/chatbox boundaries; no live audio/provider/profile/discovery/OSC sink",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def record_exit():
        exit_fingerprints = fingerprints()
        exit_manifest = product_manifest()
        (report_dir / "source_manifest_exit.json").write_text(
            json.dumps(exit_manifest, indent=2), encoding="utf-8"
        )
        (report_dir / "instance_exit.json").write_text(
            json.dumps(
                {
                    "source_fingerprints": exit_fingerprints,
                    "entry_exit_unchanged": (
                        entry == exit_fingerprints and entry_manifest == exit_manifest
                    ),
                    "product_source_manifest_sha256": exit_manifest["sha256"],
                    "product_source_unchanged": entry_manifest == exit_manifest,
                    "candidate_head": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    atexit.register(record_exit)
    dashboard_slot = ft.Container(width=900, height=650)
    logs_slot = ft.Container(width=600, height=650)
    surface = ft.Screenshot(
        content=ft.Row([dashboard_slot, logs_slot], vertical_alignment=ft.CrossAxisAlignment.START),
        expand=True,
    )
    status = ft.Text(
        "Ready · real owner/logger composition · isolated synthetic state",
        size=18,
        color=COLOR_ON_BACKGROUND,
    )
    owner = None
    runtime = None
    handler = None
    stage_index = -1
    observations = []
    facts = {}
    case = "progressive"
    current_source = None
    pending_submission = None

    async def settle_pending_submission():
        nonlocal pending_submission
        if pending_submission is not None:
            owner.source_overlay.release.set()
            await asyncio.gather(pending_submission, return_exceptions=True)
            pending_submission = None

    async def new_case(name, *, targets=("en",), finalize_wait_ms=0, source_overlay=None):
        nonlocal owner, runtime, handler, case
        await settle_pending_submission()
        showing_conversation = owner.logs._showing_conversation if owner is not None else False
        if owner is not None:
            await owner.close()
            runtime.close()
            handler.close()
        dashboard = DashboardView()
        dashboard.set_languages_from_codes(
            "fr",
            targets[0],
            "fr",
            targets[0],
            "manual",
            targets[1] if len(targets) > 1 else "",
        )
        logs = LogsView()
        if showing_conversation:
            logs._showing_conversation = True
            logs._conversation_button.content = logs._conversation_button_label()
            logs._render_conversation_text()
        dashboard_slot.content = dashboard
        logs_slot.content = logs
        handler = logging.FileHandler(
            report_dir / f"accepted-{name}.log", mode="w", encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        runtime = SessionRuntimeLoggingService(
            root_logger=logging.Logger(f"owner-gui-root-{name}"),
            session_logger=logging.Logger(f"owner-gui-session-{name}"),
            sinks=RuntimeLoggingSinks(
                stream_handler=logging.StreamHandler(io.StringIO()),
                file_handler=handler,
                log_file=report_dir / f"accepted-{name}.log",
            ),
        )
        owner = OwnerPresentation(
            dashboard,
            logs,
            runtime,
            targets=targets,
            finalize_wait_ms=finalize_wait_ms,
            source_overlay=source_overlay,
        )
        case = name
        await owner.start()

    await new_case("progressive")

    async def final_rows(count, disposition="complete"):
        await eventually(
            lambda: len(owner.logs.conversation_records) == count
            and all(record.turn_status == disposition for record in owner.logs.conversation_records)
        )

    async def advance(_event):
        nonlocal stage_index, current_source, pending_submission
        button.disabled = True
        page.update()
        stage_index += 1
        stage = STAGES[stage_index]
        if stage == "progressive-source":
            owner.provider.holds[("blue lantern", "en")] = asyncio.Event()
            current_source = await owner.source("blue lantern", close=False)
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "blue lantern"
            )
            facts["progressive_parent"] = str(next(iter(owner.snapshots)))
        elif stage == "progressive-preview-one":
            owner.provider.holds[("blue lantern", "en")].set()
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:blue lantern"
            )
        elif stage == "progressive-extended-source":
            owner.provider.holds[("blue lantern and bright", "en")] = asyncio.Event()
            await owner.revise_source(current_source, "blue lantern and bright")
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value
                == "blue lantern and bright"
            )
            assert len(owner.logs.conversation_records) == 1
            assert set(map(str, owner.snapshots)) == {facts["progressive_parent"]}
        elif stage == "progressive-preview-two":
            owner.provider.holds[("blue lantern and bright", "en")].set()
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value
                == "en:blue lantern and bright"
            )
            snapshot = next(iter(owner.snapshots.values()))
            facts["progressive_preview_update_id"] = snapshot.targets[0].translation.update_id
        elif stage == "progressive-final":
            await owner.close_source(current_source)
            await owner.idle()
            await final_rows(1)
            snapshot = next(iter(owner.snapshots.values()))
            assert (
                snapshot.targets[0].translation.update_id == facts["progressive_preview_update_id"]
            )
            assert owner.dashboard.display_card._display_text.value == "en:blue lantern and bright"
            assert len(owner.provider.calls) == 2
            assert len(runtime._conversation_record_keys) == 2
        elif stage == "older-speech-held":
            await new_case("mixed")
            owner.provider.holds[("older speech", "en")] = asyncio.Event()
            await owner.source("older speech")
            await eventually(lambda: any(value.source_closed for value in owner.snapshots.values()))
        elif stage == "manual-complete":
            await owner.harness.self_owner.submit_text("manual", source="You")
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:manual"
            )
        elif stage == "late-speech-final":
            owner.provider.holds[("older speech", "en")].set()
            await owner.idle()
            await eventually(
                lambda: any(
                    record.source_text == "older speech" and record.turn_status == "complete"
                    for record in owner.logs.conversation_records
                )
            )
            assert owner.dashboard.display_card._display_text.value == "en:manual"
        elif stage == "successor-speech":
            await owner.source("successor speech")
            await owner.idle()
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:successor speech"
            )
        elif stage == "dual-provisional":
            await new_case("dual-after", targets=("en", "ja"))
            owner.provider.holds[("dual source", "en")] = asyncio.Event()
            owner.provider.holds[("dual source", "ja")] = asyncio.Event()
            current_source = await owner.source("dual source", close=False)
            await eventually(lambda: len(owner.logs.conversation_records) == 2)
            assert not runtime._conversation_record_keys
            rows = owner.logs._log_text.value.split("\n\n")
            assert "Mic → English" in rows[0].splitlines()[0]
            assert "Mic → Japanese" in rows[1].splitlines()[0]
        elif stage == "dual-secondary-first":
            await owner.close_source(current_source)
            await eventually(lambda: len(owner.provider.calls) == 2)
            owner.provider.holds[("dual source", "ja")].set()
            await eventually(
                lambda: "ja:dual source" in owner.dashboard.display_card._display_text.value
            )
            facts["secondary_first_dashboard"] = owner.dashboard.display_card._display_text.value
            rows = owner.logs._log_text.value.split("\n\n")
            assert "Mic → English" in rows[0].splitlines()[0]
            assert t("logs.conversation.target.pending") in rows[0]
            assert "Mic → Japanese" in rows[1].splitlines()[0]
            assert "ja:dual source" in rows[1]
        elif stage == "dual-final-after-snapshots":
            owner.provider.holds[("dual source", "en")].set()
            await owner.idle()
            await final_rows(2)
            assert len(runtime._conversation_record_keys) == 3
        elif stage == "dual-final-before-snapshots":
            await new_case("dual-before", targets=("en", "ja"))
            owner.delivery_gate.clear()
            owner.provider.holds[("dual source", "en")] = asyncio.Event()
            owner.provider.holds[("dual source", "ja")] = asyncio.Event()
            current_source = await owner.source("dual source", close=False)
            await owner.close_source(current_source)
            await eventually(lambda: len(owner.provider.calls) == 2)
            for gate in owner.provider.holds.values():
                gate.set()
            await owner.idle()
            assert not owner.snapshots
            assert len(runtime._conversation_record_keys) == 3
            facts["accepted_before_any_snapshot"] = True
            owner.delivery_gate.set()
            await final_rows(2)
        elif stage == "partial-primary-preview":
            await new_case("partial", targets=("en", "ja"), finalize_wait_ms=50)
            owner.provider.failures.add(("partial source", "ja"))
            current_source = await owner.source("partial source", close=False)
            await eventually(
                lambda: "en:partial source" in owner.dashboard.display_card._display_text.value
            )
            assert t("display.turn.complete") not in owner.logs._log_text.value
        elif stage == "partial-target-failure":
            await owner.close_source(current_source)
            await owner.idle()
            await final_rows(2, "failed")
            assert "en:partial source" in owner.dashboard.display_card._display_text.value
            assert sum(call["target_language"] == "ja" for call in owner.provider.calls) == 2
            rows = owner.logs._log_text.value.split("\n\n")
            assert "Mic → English" in rows[0].splitlines()[0]
            assert t("display.turn.complete") in rows[0].splitlines()[0]
            assert t("display.turn.failed") not in rows[0].splitlines()[0]
            assert "Mic → Japanese" in rows[1].splitlines()[0]
            assert t("display.turn.failed") in rows[1].splitlines()[0]
            assert owner.dashboard.display_card._turn_state_text.value == t("display.turn.failed")
        elif stage == "older-manual-held":
            await new_case("reverse")
            owner.provider.holds[("older manual", "en")] = asyncio.Event()
            admission = await owner.harness.self_owner.submit_text("older manual", source="You")
            facts["reverse_manual_parent"] = str(admission.turn_id)
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "older manual"
            )
        elif stage == "newer-speech-complete":
            await owner.source("newer speech")
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:newer speech"
                and any(value.status == "complete" for value in owner.snapshots.values())
                and any("en:newer speech" in message.text for message in owner.chatbox.messages)
            )
        elif stage == "late-manual-final":
            owner.provider.holds[("older manual", "en")].set()
            await owner.idle()
            await owner.harness.ui_events.join()
            assert owner.dashboard.display_card._display_text.value == "en:newer speech"
            assert owner.dashboard.display_card._turn_state_text.visible
            assert owner.dashboard.display_card._turn_state_text.value == t("display.turn.complete")
            assert owner.logs._log_text.value.count("en:older manual") == 1
            assert len(owner.logs.conversation_records) == 3
            assert len(runtime._conversation_record_keys) == 4
            speech_index = next(
                index
                for index, message in enumerate(owner.chatbox.messages)
                if "en:newer speech" in message.text
            )
            manual_index = next(
                index
                for index, message in enumerate(owner.chatbox.messages)
                if "en:older manual" in message.text
            )
            assert manual_index > speech_index
            assert not owner.chatbox.messages[manual_index].self_speech
            facts["late_manual_chatbox_bypass_after_newer_speech"] = True
        elif stage == "future-manual-complete":
            await owner.harness.self_owner.submit_text("future manual", source="You")
            await owner.idle()
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:future manual"
            )
            assert len(runtime._conversation_record_keys) == 6
        elif stage == "queued-legacy-source-final":
            await new_case("queued-legacy")
            owner.delivery_gate.clear()
            await owner.harness.self_owner.submit_text("queued old manual", source="You")
            await owner.idle()
            await owner.source("queued newer speech")
            await owner.idle()
            assert not owner.snapshots
            assert len(runtime._conversation_record_keys) == 4
            owner.delivery_gate.set()
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:queued newer speech"
            )
            await owner.harness.ui_events.join()
            assert len(owner.logs.conversation_records) == 3
            assert owner.logs._log_text.value.count("en:queued old manual") == 1
            facts["queued_old_source_and_final_consumed_after_new_speech"] = True
        elif stage == "pending-manual-source":
            await new_case(
                "pending-source", source_overlay=ControlledSourceOverlay("pending manual")
            )
            owner.provider.holds[("older manual", "en")] = asyncio.Event()
            await owner.harness.self_owner.submit_text("older manual", source="You")
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "older manual"
            )
            pending_submission = asyncio.create_task(
                owner.harness.self_owner.submit_text("pending manual", source="You")
            )
            await asyncio.wait_for(owner.source_overlay.started.wait(), timeout=3)
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "pending manual"
            )
            assert not pending_submission.done()
            assert not any(call["text"] == "pending manual" for call in owner.provider.calls)
            facts["pending_source_observed_before_admission"] = True
        elif stage == "late-manual-while-source-pending":
            owner.provider.holds[("older manual", "en")].set()
            await owner.idle()
            await owner.harness.ui_events.join()
            assert not pending_submission.done()
            assert owner.dashboard.display_card._display_text.value == "pending manual"
            assert owner.logs._log_text.value.count("en:older manual") == 1
            assert any(
                "en:older manual" in message.text and not message.self_speech
                for message in owner.chatbox.messages
            )
            assert not any(call["text"] == "pending manual" for call in owner.provider.calls)
        elif stage == "pending-manual-admitted":
            owner.source_overlay.release.set()
            admission = await pending_submission
            pending_submission = None
            facts["pending_manual_parent"] = str(admission.turn_id)
            await owner.idle()
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:pending manual"
            )
            assert len(runtime._conversation_record_keys) == 4
            assert len(owner.logs.conversation_records) == 4
        elif stage == "segmented-old-first-held":
            await new_case("segmented", source_overlay=ControlledSourceOverlay("first "))
            unit, parent = independent_recognition((("first ", "en"), ("second", "ja")))
            facts["segmented_parent"] = str(parent)
            pending_submission = asyncio.create_task(
                owner.harness.self_owner.handle_recognition_unit(unit)
            )
            await asyncio.wait_for(owner.source_overlay.started.wait(), timeout=3)
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value.strip() == "first"
            )
            assert not pending_submission.done()
            assert owner.harness.self_owner.merge_buffer is None
            facts["ordinary_first_child_visible_before_parent_callback"] = True
        elif stage == "segmented-new-managed":
            await owner.source("newer managed")
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:newer managed"
                and any(value.status == "complete" for value in owner.snapshots.values())
            )
        elif stage == "segmented-old-released":
            owner.source_overlay.release.set()
            await pending_submission
            pending_submission = None
            await owner.idle()
            await owner.harness.ui_events.join()
            assert owner.dashboard.display_card._display_text.value == "en:newer managed"
            assert owner.dashboard.display_card._turn_state_text.visible
            assert owner.dashboard.display_card._turn_state_text.value == t("display.turn.complete")
            records = [
                record
                for record in owner.logs.conversation_records
                if (record.parent_utterance_id or record.utterance_id) == facts["segmented_parent"]
            ]
            assert len(records) == 4
            assert all(record.accepted for record in records)
            translations = [record for record in records if record.translated_text is not None]
            assert len({record.utterance_id for record in translations}) == 2
            assert all(record.utterance_id != facts["segmented_parent"] for record in translations)
            assert owner.logs._log_text.value.count("en:first") == 1
            assert owner.logs._log_text.value.count("en:second") == 1
            assert len(runtime._conversation_record_keys) == 6
            for text in ("first", "second", "newer managed"):
                assert sum(call["text"].strip() == text for call in owner.provider.calls) == 1
        elif stage == "segmented-future-independent":
            unit, parent = independent_recognition(
                (("future first ", "en"), ("future second", "ja"))
            )
            facts["future_independent_parent"] = str(parent)
            await owner.harness.self_owner.handle_recognition_unit(unit)
            await owner.idle()
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:future second"
            )
            assert parent not in owner.snapshots
            assert owner.logs._log_text.value.count("en:future first") == 1
            assert owner.logs._log_text.value.count("en:future second") == 1
            assert len(runtime._conversation_record_keys) == 10
        elif stage == "segmented-future-managed":
            await owner.source("future managed")
            await owner.idle()
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:future managed"
            )
            assert len(runtime._conversation_record_keys) == 12
        elif stage == "pressure-old-speech":
            await new_case("source-timeout")
            owner.provider.holds[("old speech", "en")] = asyncio.Event()
            await owner.source("old speech")
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "old speech"
                and owner.dashboard.display_card._turn_state_text.visible
                and owner.dashboard.display_card._turn_state_text.value
                == t("display.turn.translating")
            )
        elif stage == "manual-source-timeout":
            previous_timeout = await owner.fill_ui_queue_for_source_timeout()
            facts["controlled_ui_write_timeout_s"] = (
                owner.harness.output_projection.ui_messages.write_timeout_s
            )
            owner.provider.holds[("manual after dropped source", "en")] = asyncio.Event()
            admission = await owner.harness.self_owner.submit_text(
                "manual after dropped source", source="You"
            )
            facts["source_timeout_manual_parent"] = str(admission.turn_id)
            writer = owner.harness.output_projection.ui_messages
            await asyncio.wait_for(writer.wait_for_idle(), timeout=3)
            writer.write_timeout_s = previous_timeout
            receipts = [
                decision
                for decision in owner.harness.output_runtime.routing_decisions
                if decision.metadata.get("parent_utterance_id") == str(admission.turn_id)
                and decision.metadata.get("event_type") == UIEventType.TRANSCRIPT_FINAL.value
            ]
            timeouts = [
                decision for decision in receipts if decision.reason == "destination_write_timeout"
            ]
            assert len(timeouts) == 1
            assert timeouts[0].metadata["accepted_handoff"] is False
            assert timeouts[0].metadata["ui_queue_submitted"] is False
            assert not any(decision.metadata["ui_queue_submitted"] for decision in receipts)
            owner.delivery_gate.set()
            await owner.harness.ui_events.join()
            assert owner.dashboard.display_card._display_text.value == "old speech"
            assert owner.dashboard.display_card._turn_state_text.visible
        elif stage == "manual-result-after-source-timeout":
            owner.provider.holds[("manual after dropped source", "en")].set()
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value
                == "en:manual after dropped source"
                and len(runtime._conversation_record_keys) == 2
                and any(
                    "en:manual after dropped source" in message.text
                    for message in owner.chatbox.messages
                )
            )
            assert not owner.provider.holds[("old speech", "en")].is_set()
            assert owner.logs._log_text.value.count("en:manual after dropped source") == 1
            messages = [
                message
                for message in owner.chatbox.messages
                if "en:manual after dropped source" in message.text
            ]
            assert len(messages) == 1
            assert not messages[0].self_speech
            assert (
                sum(call["text"] == "manual after dropped source" for call in owner.provider.calls)
                == 1
            )
        elif stage == "pressure-successor-managed":
            await owner.source("pressure successor")
            await eventually(
                lambda: owner.dashboard.display_card._display_text.value == "en:pressure successor"
                and owner.dashboard.display_card._turn_state_text.visible
                and owner.dashboard.display_card._turn_state_text.value
                == t("display.turn.complete")
            )
            owner.provider.holds[("old speech", "en")].set()
            await owner.idle()
            await owner.harness.ui_events.join()
            assert owner.dashboard.display_card._display_text.value == "en:pressure successor"
            assert owner.dashboard.display_card._turn_state_text.visible
            assert owner.dashboard.display_card._turn_state_text.value == t("display.turn.complete")
            assert len(runtime._conversation_record_keys) == 6
            assert (
                sum(
                    "en:manual after dropped source" in message.text
                    for message in owner.chatbox.messages
                )
                == 1
            )
            assert not any(
                decision.metadata.get("parent_utterance_id")
                == facts["source_timeout_manual_parent"]
                and decision.metadata.get("event_type") == UIEventType.TRANSCRIPT_FINAL.value
                and decision.metadata["ui_queue_submitted"]
                for decision in owner.harness.output_runtime.routing_decisions
            )
        if stage in {
            "manual-complete",
            "late-speech-final",
            "future-manual-complete",
            "pending-manual-source",
            "late-manual-while-source-pending",
            "pending-manual-admitted",
            "segmented-old-first-held",
            "segmented-future-independent",
            "manual-result-after-source-timeout",
        }:
            assert owner.dashboard.display_card._turn_status is None
            assert owner.dashboard.display_card._turn_state_text.value == ""
            assert not owner.dashboard.display_card._turn_state_text.visible
        status.value = f"{stage_index + 1}/{len(STAGES)} · {stage} · retained rows {len(owner.logs.conversation_records)}"
        button.disabled = stage_index + 1 == len(STAGES)
        page.update()
        configuration = owner.harness.self_owner.config_snapshot().value
        observations.append(
            {
                "stage": stage,
                "case": case,
                "profile": {
                    "low_latency_mode": configuration.low_latency_mode,
                    "low_latency_finalize_wait_ms": configuration.low_latency_finalize_wait_ms,
                    "low_latency_awaiting_vad_timeout_s": configuration.low_latency_awaiting_vad_timeout_s,
                    "source_language": configuration.source_language,
                    "self_target_languages": configuration.self_target_languages,
                    "ui_write_timeout_s": owner.harness.output_projection.ui_messages.write_timeout_s,
                },
                "dashboard": owner.dashboard.display_card._display_text.value,
                "dashboard_status": owner.dashboard.display_card._turn_state_text.value,
                "dashboard_status_visible": bool(
                    owner.dashboard.display_card._turn_state_text.visible
                ),
                "dashboard_visible_status": (
                    owner.dashboard.display_card._turn_state_text.value
                    if owner.dashboard.display_card._turn_state_text.visible
                    else ""
                ),
                "conversation": owner.logs._log_text.value,
                "conversation_visible": owner.logs._showing_conversation,
                "rows": [asdict(record) for record in owner.logs.conversation_records],
                "accepted_record_keys": len(runtime._conversation_record_keys),
                "accepted_keys": [
                    list(key) for key in sorted(runtime._conversation_record_keys, key=str)
                ],
                "provider_calls": [
                    {**call, "utterance_id": str(call["utterance_id"])}
                    for call in owner.provider.calls
                ],
                "chatbox_messages": [
                    {
                        "utterance_id": str(message.utterance_id),
                        "text": message.text,
                        "turn_generation": message.turn_generation,
                        "turn_order": message.turn_order,
                        "self_speech": message.self_speech,
                    }
                    for message in owner.chatbox.messages
                ],
                "snapshots": [
                    {
                        "parent_id": str(value.utterance_id),
                        "status": value.status,
                        "targets": [
                            {
                                "language": target.target_language,
                                "status": target.status,
                                "child_id": (
                                    str(target.translation.utterance_id)
                                    if target.translation
                                    else None
                                ),
                            }
                            for target in value.targets
                        ],
                    }
                    for value in owner.snapshots.values()
                ],
                "ui_routing_decisions": [
                    {
                        "decision": decision.decision,
                        "reason": decision.reason,
                        "publication_id": decision.publication_id,
                        "metadata": dict(decision.metadata),
                    }
                    for decision in owner.harness.output_runtime.routing_decisions
                    if "event_type" in decision.metadata
                ],
                "facts": dict(facts),
            }
        )
        (report_dir / "observations.json").write_text(
            json.dumps({"stages": observations}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def capture(_event):
        name = STAGES[stage_index] if stage_index >= 0 else "ready"
        data = await surface.capture(pixel_ratio=1.0)
        (report_dir / f"{stage_index + 1:02d}-{name}.png").write_bytes(data)

    async def close(_event):
        await settle_pending_submission()
        if owner is not None:
            await owner.close()
            runtime.close()
            handler.close()

    button = ft.TextButton("Next snapshot", on_click=advance)
    page.on_disconnect = close
    page.add(
        ft.Text("SELF-STREAM-2 · real owner/logger UI repair proof", size=24),
        ft.Row([button, ft.TextButton("Capture surface", on_click=capture)]),
        status,
        surface,
    )
    if native:
        page.window.width = 1580
        page.window.height = 1000
        page.window.left = 0
        page.window.top = 0
        page.update()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8174)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.native and not os.environ.get("FLET_VIEW_PATH"):
        parser.error("--native requires an explicit cached FLET_VIEW_PATH")

    async def app(page):
        await main(page, report_dir=args.report_dir, native=args.native)

    os.environ["FLET_FORCE_WEB_SERVER"] = "false" if args.native else "true"
    ft.run(
        app,
        host="127.0.0.1",
        port=args.port,
        view=ft.AppView.FLET_APP if args.native else ft.AppView.WEB_BROWSER,
        no_cdn=True,
        assets_dir=None,
    )


if __name__ == "__main__":
    run()
