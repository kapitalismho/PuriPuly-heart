from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import flet as ft

import puripuly_heart.ui.event_dispatch as event_dispatch_module
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks, SessionRuntimeLoggingService
from puripuly_heart.domain.events import (
    SelfTurnPresentation,
    SelfTurnTargetPresentation,
    UIEvent,
    UIEventType,
)
from puripuly_heart.domain.models import Translation
from puripuly_heart.ui.i18n import set_locale
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from puripuly_heart.ui.theme import COLOR_BACKGROUND, COLOR_ON_BACKGROUND
from puripuly_heart.ui.views.dashboard import DashboardView
from puripuly_heart.ui.views.logs import LogsView


def draft(order, targets=("en",)):
    return SelfTurnPresentation(
        utterance_id=UUID(int=order),
        turn_generation=1,
        turn_order=order,
        revision=1,
        source_revision=1,
        source_text="The lantern is blue",
        source_language="en",
        source_closed=False,
        status="active",
        targets=tuple(SelfTurnTargetPresentation(language, 1, "pending") for language in targets),
        created_at=0.0,
    )


def ready(snapshot, text, index=0):
    targets = list(snapshot.targets)
    target = targets[index]
    targets[index] = replace(
        target,
        source_revision=snapshot.source_revision,
        status="ready",
        translation=Translation(
            snapshot.utterance_id,
            text,
            source_text=snapshot.source_text,
            source_language=snapshot.source_language,
            target_language=target.target_language,
            update_id=f"synthetic-{snapshot.turn_order}-{snapshot.source_revision}-{index}",
        ),
    )
    return replace(snapshot, revision=snapshot.revision + 1, targets=tuple(targets))


def scenario():
    source = draft(1, ("fr",))
    first = ready(source, "La lanterne est bleue")
    extended = replace(
        first, revision=3, source_revision=2, source_text="The lantern is blue and bright"
    )
    second = ready(extended, "La lanterne est bleue et brillante")
    closed = replace(second, revision=5, source_closed=True)
    final = replace(closed, revision=6, status="complete")
    old = ready(draft(2, ("fr",)), "Ancien aperçu")
    successor = ready(replace(draft(3, ("fr",)), source_text="Successor source"), "Nouvel aperçu")
    delayed = replace(old, revision=3, source_closed=True, status="complete")
    cancelled = replace(
        successor,
        revision=3,
        status="cancelled",
        targets=(replace(successor.targets[0], status="cancelled"),),
    )
    secondary = ready(draft(4, ("fr", "de")), "Die Laterne ist blau", 1)
    dual = ready(secondary, "La lanterne est bleue")
    dual_final = replace(dual, revision=4, source_closed=True, status="complete")
    blocked = draft(5, ("fr",))
    blocked_final = replace(
        ready(blocked, "Livraison débloquée"), revision=3, source_closed=True, status="complete"
    )
    return [
        ("source", source),
        ("preview-one", first),
        ("extended-source", extended),
        ("preview-two", second),
        ("source-closed", closed),
        ("identical-final", final),
        ("earlier-pending", old),
        ("successor-preview", successor),
        ("delayed-old-final", delayed),
        ("cancelled", cancelled),
        ("secondary-first", secondary),
        ("dual-preview", dual),
        ("dual-final", dual_final),
        ("blocked-ui", blocked),
        ("released-ui", blocked_final),
    ]


async def main(page, *, report_dir, native=False):
    set_locale("en")
    page.title = "SELF-STREAM-2 isolated presentation proof"
    page.bgcolor = COLOR_BACKGROUND
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    if native:
        page.window.width = 1580
        page.window.height = 1000
        page.window.left = 0
        page.window.top = 0
    dashboard = DashboardView()
    dashboard.set_languages_from_codes("en", "fr", "en", "fr", "manual")
    logs = LogsView()
    handler = logging.FileHandler(report_dir / "accepted.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    runtime = SessionRuntimeLoggingService(
        root_logger=logging.Logger("synthetic-gui-root"),
        session_logger=logging.Logger("synthetic-gui-session"),
        sinks=RuntimeLoggingSinks(
            stream_handler=logging.StreamHandler(io.StringIO()),
            file_handler=handler,
            log_file=report_dir / "accepted.log",
        ),
    )
    host = SimpleNamespace(view_dashboard=dashboard, view_logs=logs)
    adapter = FletUiPresentationAdapter(host)
    adapter.attach_runtime_log_sink(runtime)
    queue = asyncio.Queue(maxsize=4)
    bridge = adapter.create_ui_event_bridge(event_queue=queue, runtime_logging=runtime)
    admission = asyncio.Event()
    admission.set()
    entered = asyncio.Event()
    original = bridge._handle_event

    async def controlled_delivery(event):
        entered.set()
        await admission.wait()
        await original(event)

    bridge._handle_event = controlled_delivery
    consumer = asyncio.create_task(bridge.run())
    stages = scenario()
    stage_index = -1
    observations = []
    status = ft.Text(
        "Ready · overlay disabled · synthetic local presentation only",
        size=18,
        color=COLOR_ON_BACKGROUND,
    )
    surface = ft.Column(
        [
            ft.Row(
                [
                    ft.Container(content=dashboard, width=900, height=650),
                    ft.Container(content=logs, width=600, height=650),
                ],
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
        ]
    )
    screenshot = ft.Screenshot(content=surface, expand=True)
    candidate = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    module_names = (
        "puripuly_heart.domain.events",
        "puripuly_heart.ui.event_mapping",
        "puripuly_heart.ui.event_projection",
        "puripuly_heart.ui.event_dispatch",
        "puripuly_heart.ui.presentation_adapter",
        "puripuly_heart.ui.components.display_card",
        "puripuly_heart.ui.views.dashboard",
        "puripuly_heart.ui.views.logs",
    )
    fingerprints = {}
    for name in module_names:
        path = Path(sys.modules[name].__file__).resolve()
        fingerprints[name] = {
            "path": path.relative_to(Path.cwd()).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (report_dir / "instance.json").write_text(
        json.dumps(
            {
                "python_pid": os.getpid(),
                "python": Path(sys.executable).resolve().relative_to(Path.cwd()).as_posix(),
                "native": native,
                "candidate_head": candidate,
                "source_fingerprints": fingerprints,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def accepted(snapshot):
        if snapshot.status != "complete":
            return
        for index, target in enumerate(snapshot.targets):
            for _ in range(2):
                runtime.record_conversation_observation(
                    utterance_id=str(snapshot.utterance_id),
                    speaker_channel="self",
                    transcript_text=snapshot.source_text,
                    translation_text=target.translation.text if target.translation else None,
                    source_language=snapshot.source_language,
                    target_language=target.target_language,
                    metadata={"source": "Mic", "target_index": index, "disposition": "translated"},
                )

    async def advance(_event):
        nonlocal stage_index
        next_button.disabled = True
        page.update()
        stage_index += 1
        name, snapshot = stages[stage_index]
        adapter.set_dashboard_languages(
            source_language=snapshot.source_language,
            target_language=snapshot.targets[0].target_language,
            secondary_target_language=(
                snapshot.targets[1].target_language if len(snapshot.targets) > 1 else ""
            ),
            peer_source_language="en",
            peer_target_language="fr",
            peer_source_mode="manual",
            recent_source_languages=[],
            recent_target_languages=[],
            peer_auto_detect_available=False,
        )
        event = UIEvent(
            UIEventType.SELF_TURN_UPDATED,
            utterance_id=snapshot.utterance_id,
            payload=snapshot,
            source="Mic",
        )
        if name == "blocked-ui":
            admission.clear()
            entered.clear()
            await queue.put(event)
            await entered.wait()
        else:
            accepted(snapshot)
            await queue.put(event)
            admission.set()
            await queue.join()
        status.value = f"{stage_index + 1}/{len(stages)} · {name} · retained rows {len(logs.conversation_records)}"
        next_button.disabled = stage_index + 1 >= len(stages)
        page.update()
        observations.append(
            {
                "stage": name,
                "dashboard": dashboard.display_card._display_text.value,
                "dashboard_status": dashboard.display_card._turn_state_text.value,
                "conversation_visible": logs._showing_conversation,
                "conversation": logs._log_text.value,
                "rows": [asdict(record) for record in logs.conversation_records],
                "accepted_record_keys": len(runtime._conversation_record_keys),
                "queue_size": queue.qsize(),
            }
        )
        (report_dir / "observations.json").write_text(
            json.dumps(
                {
                    "candidate_head": candidate,
                    "candidate_working_tree": True,
                    "ui_import": Path(event_dispatch_module.__file__)
                    .resolve()
                    .relative_to(Path.cwd())
                    .as_posix(),
                    "boundary": "Production Flet views, presentation adapter, UIEventBridge queue consumer and runtime accepted logger; source/provider/output producers controlled; no overlay, audio, discovery, credentials or remote sinks constructed",
                    "stages": observations,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    async def capture(_event):
        name = stages[stage_index][0] if stage_index >= 0 else "ready"
        data = await screenshot.capture(pixel_ratio=1.0)
        (report_dir / f"{stage_index + 1:02d}-{name}.png").write_bytes(data)

    async def close(_event):
        bridge.close()
        admission.set()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        logs.will_unmount()
        runtime.close()
        handler.close()

    next_button = ft.TextButton("Next snapshot", on_click=advance)
    page.on_disconnect = close
    page.add(
        ft.Text("SELF-STREAM-2 · controlled production UI", size=24),
        ft.Row([next_button, ft.TextButton("Capture surface", on_click=capture)]),
        status,
        screenshot,
    )
    await bridge.wait_started()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8173)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    async def app(page):
        await main(page, report_dir=args.report_dir, native=args.native)

    if args.native and not os.environ.get("FLET_VIEW_PATH"):
        parser.error("--native requires an explicit cached FLET_VIEW_PATH")
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
