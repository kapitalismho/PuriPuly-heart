"""SteamVR markerless speaker-transition emphasis visual check."""

from __future__ import annotations

import argparse
import asyncio
import math
import secrets
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

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
    {
        "kind": "revision",
        "key": "p2r",
        "ko": "다른 목소리로 계속 이어집니다",
        "en": "Continued in another voice with revised text",
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
STEP_CUE = {
    "p1": "Peer Gold, no marker",
    "p2": "incoming Peer whole turn Sky, no marker",
    "p2r": "same logical Peer revision stays Sky, no marker",
    "s1": "SELF White; previous Peer returns to Gold",
    "p3": "incoming Peer whole turn Sky, no marker",
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
        description=f"{target} markerless temporary speaker emphasis check using the production presenter.",
    )


def _parser(surface: str) -> argparse.ArgumentParser:
    parser = build_parser(surface)
    parser.add_argument(
        "--step-delay", type=_step_delay, default=2.0, help="자막 단계 사이 대기 초 (기본 2.0)"
    )
    parser.add_argument("--once", action="store_true", help="한 번 재생하고 종료")
    return parser


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
    step: dict[str, str],
    session_scope: str,
) -> tuple[UUID, str, str]:
    uid = uuid4()
    counters["order"] += 1
    counters["child"] += 1
    start_ms = counters["order"] * 100
    ko = step["ko"]
    en = step["en"]
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
    return uid, claim.comparison, claim.claim_id


async def _emit_peer_revision(
    presenter: OverlayPresenter,
    adapter: OverlayEventAdapter,
    step: dict[str, str],
    peer_identity: tuple[UUID, str, str],
) -> None:
    uid, comparison, claim_id = peer_identity
    receipt = await presenter.emit(
        adapter.translation_final(
            utterance_id=uid,
            channel="peer",
            text=step["ko"],
            source_text=step["en"],
            source_language="en",
            target_language="ko",
            applied_context_mode="integrated",
            logical_turn_key=f"peer:{uid}",
            speaker_transition=comparison,
            speaker_transition_claim_id=claim_id,
        )
    )
    if receipt.outcome != "applied":
        raise RuntimeError(f"{step['key']} 적용 실패: {receipt.outcome} {receipt.cause}")


async def _emit_self(
    presenter: OverlayPresenter,
    adapter: OverlayEventAdapter,
    step: dict[str, str],
) -> None:
    uid = uuid4()
    ko = step["ko"]
    en = step["en"]
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
                print(f"라운드 {round_index}: markerless temporary emphasis", flush=True)
                interpreter = PeerSpeakerTransitionInterpreter()
                peer_identity: tuple[UUID, str, str] | None = None
                for step in SEQUENCE:
                    if manager.state != "connected":
                        print(f"자식이 닫혔습니다: {manager.failure_reason}", flush=True)
                        exit_code = 2
                        stop = True
                        break
                    if step["kind"] == "peer":
                        peer_identity = await _emit_peer(
                            presenter, adapter, interpreter, counters, step, session_scope
                        )
                    elif step["kind"] == "revision":
                        if peer_identity is None:
                            raise RuntimeError("revision 대상 Peer 턴이 없습니다")
                        await _emit_peer_revision(presenter, adapter, step, peer_identity)
                    else:
                        await _emit_self(presenter, adapter, step)
                    print(f"{step['key']}: {STEP_CUE[step['key']]}", flush=True)
                    await asyncio.sleep(step_delay)
                    if manager.state != "connected":
                        print(f"자식이 닫혔습니다: {manager.failure_reason}", flush=True)
                        exit_code = 2
                        stop = True
                        break
                if stop or once:
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
