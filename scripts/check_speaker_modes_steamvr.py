"""SteamVR speaker-transition visual check for modes A, C, E."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import math
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.process import (
    DefaultOverlayProcessRunner,
    DesktopFletOverlayRunner,
    OverlayProcessManager,
)
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle
from puripuly_heart.core.speaker_transition import PeerSpeakerTransitionInterpreter
from puripuly_heart.domain.models import FinalSpeakerRun, Transcript

MODES = ("A", "C", "E")
SEQUENCE = (
    {
        "kind": "peer",
        "key": "p1",
        "speaker": "speaker-a",
        "ko": "첫 번째 이야기가 시작됩니다",
        "en": "The first story begins",
    },
    {
        "kind": "peer",
        "key": "p2",
        "speaker": "speaker-b",
        "ko": "다른 목소리로 이어집니다",
        "en": "Continued in another voice",
    },
    {"kind": "self", "key": "s1", "ko": "내 목소리 확인합니다", "en": "My voice check"},
    {
        "kind": "peer",
        "key": "p3",
        "speaker": "speaker-c",
        "ko": "다시 바뀌어 이어집니다",
        "en": "Changed again and continued",
    },
)
MODE_GUIDE = {
    "A": "Peer는 항상 Gold, 전환된 블록에만 금색 상단 마커가 붙습니다.",
    "C": "마커 없음, 전환마다 Gold와 Sky가 교대로 바뀝니다.",
    "E": "전환 직후 본문이 Sky로 강조되고 금색 마커가 붙고, 다음 턴에 본문은 Gold로 돌아오고 마커는 남습니다.",
}
STEP_CUE = {
    ("A", "p1"): "Gold, 마커 없음",
    ("A", "p2"): "Gold + 금색 상단 마커",
    ("A", "s1"): "SELF 흰색, 직전 Peer Gold+마커 유지",
    ("A", "p3"): "Gold + 금색 상단 마커",
    ("C", "p1"): "Gold, 마커 없음",
    ("C", "p2"): "Sky로 전환, 마커 없음",
    ("C", "s1"): "SELF 흰색, 직전 Peer Sky 유지",
    ("C", "p3"): "Gold로 토글백, 마커 없음",
    ("E", "p1"): "Gold, 마커 없음",
    ("E", "p2"): "Sky + 금색 마커",
    ("E", "s1"): "SELF 흰색, 직전 Peer 본문 Gold로, 마커 유지",
    ("E", "p3"): "Sky + 금색 마커",
}


def _step_delay(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("step-delay는 숫자여야 합니다") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("step-delay는 0보다 큰 유한한 숫자여야 합니다")
    return parsed


def build_parser(surface: str) -> argparse.ArgumentParser:
    target = "SteamVR" if surface == "steamvr" else "데스크탑"
    return argparse.ArgumentParser(
        description=f"{target} A/C/E 화자 전환 확인. 실제 presenter와 {target} 자막 경로로 같은 자막 순서를 A→C→E로 보여줍니다.",
    )


def _parser(surface: str) -> argparse.ArgumentParser:
    parser = build_parser(surface)
    parser.add_argument(
        "--step-delay", type=_step_delay, default=2.0, help="자막 단계 사이 대기 초 (기본 2.0)"
    )
    parser.add_argument("--once", action="store_true", help="A/C/E 한 바퀴만 재생하고 종료")
    return parser


def _process_names() -> set[str]:
    if os.name != "nt":
        raise RuntimeError("현재 OS에서는 실행 중인 오버레이 충돌을 확인할 수 없습니다")
    try:
        completed = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"실행 중인 프로세스 확인 실패: {exc}") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError("실행 중인 프로세스 확인 실패")
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    return {row[0].strip().lower() for row in rows if row and row[0].strip()}


def _desktop_controls() -> list[dict[str, object]]:
    return [
        {"command": "apply_window_bounds", "x": 100, "y": 100, "width": 1344, "height": 336},
        {
            "command": "apply_visual_config",
            "text_scale": 1.0,
            "background_alpha": 0.6,
            "outline_width": None,
            "swap_caption_languages": False,
        },
        {"command": "set_interaction_mode", "mode": "edit"},
    ]


async def _emit_peer(
    presenter: OverlayPresenter,
    adapter: OverlayEventAdapter,
    interpreter: PeerSpeakerTransitionInterpreter,
    counters: dict[str, int],
    mode: str,
    step: dict[str, str],
    session_scope: str,
) -> None:
    uid = uuid4()
    counters["order"] += 1
    counters["child"] += 1
    start_ms = counters["order"] * 100
    ko = f"[{mode}] {step['ko']}"
    en = f"[{mode}] {step['en']}"
    transcript = Transcript(
        uid,
        en,
        True,
        channel="peer",
        final_speaker_runs=(
            FinalSpeakerRun(
                en,
                step["speaker"],
                session_scope,
                source_start_ms=start_ms,
                source_end_ms=start_ms + 90,
                speaker_confidence=0.95,
            ),
        ),
        publication_generation=1,
        source_order=counters["order"],
        created_at=time.monotonic(),
    )
    claim = interpreter.observe(transcript, child_sequence=counters["child"])
    event = adapter.translation_final(
        utterance_id=uid,
        channel="peer",
        text=ko,
        source_text=en,
        source_language="en",
        target_language="ko",
        applied_context_mode="integrated",
        logical_turn_key=f"peer:{uid}",
        speaker_transition=claim.comparison,
        speaker_transition_claim_id=claim.claim_id,
    )
    receipt = await presenter.emit(event)
    if receipt.outcome != "applied":
        raise RuntimeError(f"{step['key']} 적용 실패: {receipt.outcome} {receipt.cause}")


async def _emit_self(
    presenter: OverlayPresenter,
    adapter: OverlayEventAdapter,
    mode: str,
    step: dict[str, str],
) -> None:
    uid = uuid4()
    ko = f"[{mode}] {step['ko']}"
    en = f"[{mode}] {step['en']}"
    receipt = await presenter.emit(
        adapter.transcript_final(
            Transcript(uid, ko, True, channel="self"),
            source_language="ko",
            target_language="en",
            logical_turn_key=f"self:{uid}",
        )
    )
    if receipt.outcome != "applied":
        raise RuntimeError(f"{step['key']} 적용 실패: {receipt.outcome} {receipt.cause}")
    receipt = await presenter.emit(
        adapter.translation_final(
            utterance_id=uid,
            channel="self",
            text=en,
            source_text=ko,
            source_language="ko",
            target_language="en",
            applied_context_mode="integrated",
            logical_turn_key=f"self:{uid}",
        )
    )
    if receipt.outcome != "applied":
        raise RuntimeError(f"{step['key']} 적용 실패: {receipt.outcome} {receipt.cause}")


async def _run_surface(surface: str, step_delay: float, once: bool) -> int:
    if surface == "steamvr":
        try:
            names = _process_names()
        except RuntimeError as exc:
            print(str(exc), flush=True)
            return 2
        if "puripulyheartoverlay.exe" in names:
            print("이미 실행 중인 PuriPuly 오버레이가 있어 시작하지 않습니다.", flush=True)
            return 2
    runtime = OverlayRuntimeHandle(
        overlay_instance_id=f"overlay-check-{uuid4().hex[:8]}",
        shutdown_grace_s=3.0,
    )
    exit_code = 2
    try:
        desktop = surface == "desktop"
        presenter = OverlayPresenter(
            calibration=OverlayCalibration(),
            show_translation=True,
            show_peer_original=True,
            translation_enabled=True,
            native_retry_enabled=not desktop,
            task_factory=runtime.create_child_task,
        )
        runtime.adopt_presenter(presenter)
        if not desktop:
            await presenter.begin_native_retry_epoch(enabled=True)
        bridge = OverlayBridge(
            session_token=secrets.token_urlsafe(16),
            initial_snapshot=presenter.snapshot(),
            overlay_instance_id=runtime.overlay_instance_id,
            runtime_generation=1,
            desktop_runtime_controls_enabled=desktop,
            task_factory=runtime.create_child_task,
        )
        if desktop:
            bridge.set_initial_desktop_runtime_controls(_desktop_controls())
        runtime.attach_bridge(bridge)
        await bridge.start()
        presenter.attach_bridge(bridge)
        manager = OverlayProcessManager(
            process_runner=DesktopFletOverlayRunner() if desktop else DefaultOverlayProcessRunner(),
            bridge_url=bridge.url,
            bridge_messages=bridge.messages,
            bridge_messages_authenticated=desktop,
            session_token=bridge.session_token,
            locale="ko",
            log_dir="logs",
            startup_timeout_ms=15000,
            overlay_instance_id=runtime.overlay_instance_id,
            task_factory=runtime.create_child_task,
            selected_target=surface,
            geometry_authority="flet" if desktop else "native",
            graceful_shutdown_request=bridge.broadcast_shutdown,
        )
        runtime.attach_process_manager(manager)
        await manager.start()
        if manager.state != "connected":
            if desktop:
                print(f"데스크탑 창을 시작하지 못했습니다: {manager.failure_reason}", flush=True)
            else:
                print(
                    f"SteamVR 오버레이를 시작하지 못했습니다: {manager.failure_reason}", flush=True
                )
                print("SteamVR를 직접 시작하고 HMD를 착용한 뒤 다시 실행하십시오.", flush=True)
            exit_code = 2
        else:
            print("연결됨. 화면에서 직접 확인하십시오.", flush=True)
            adapter = OverlayEventAdapter()
            counters = {"order": 0, "child": 0}
            session_scope = f"check-{uuid4().hex[:8]}"
            stop = False
            round_index = 0
            exit_code = 0
            while not stop:
                round_index += 1
                print(f"라운드 {round_index}: A→C→E", flush=True)
                for mode in MODES:
                    if stop:
                        break
                    await presenter.update_speaker_transition_mode(mode)
                    print(f"모드 {mode}: {MODE_GUIDE[mode]}", flush=True)
                    interpreter = PeerSpeakerTransitionInterpreter()
                    for step in SEQUENCE:
                        if manager.state != "connected":
                            print(f"자식이 닫혔습니다: {manager.failure_reason}", flush=True)
                            exit_code = 2
                            stop = True
                            break
                        if step["kind"] == "peer":
                            await _emit_peer(
                                presenter, adapter, interpreter, counters, mode, step, session_scope
                            )
                        else:
                            await _emit_self(presenter, adapter, mode, step)
                        print(
                            f"[{mode}] {step['key']}: {STEP_CUE[(mode, step['key'])]}", flush=True
                        )
                        await asyncio.sleep(step_delay)
                        if manager.state != "connected":
                            print(f"자식이 닫혔습니다: {manager.failure_reason}", flush=True)
                            exit_code = 2
                            stop = True
                            break
                if stop:
                    break
                if once:
                    break
                print("반복 중. 종료는 Ctrl+C.", flush=True)
    except KeyboardInterrupt:
        print("중단됨.", flush=True)
        exit_code = 130
    except asyncio.CancelledError:
        print("중단됨.", flush=True)
        exit_code = 130
    except Exception as exc:
        print(f"실패: {type(exc).__name__}: {exc}", flush=True)
        exit_code = 2
    finally:
        try:
            await runtime.close(preserve_presenter_state=False)
        except Exception as exc:
            print(f"정리 실패: {type(exc).__name__}: {exc}", flush=True)
            exit_code = 2
    return exit_code


def main(argv: list[str] | None = None, surface: str = "steamvr") -> int:
    args = _parser(surface).parse_args(argv)
    return asyncio.run(_run_surface(surface, args.step_delay, args.once))


if __name__ == "__main__":
    raise SystemExit(main())
