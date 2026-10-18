from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import json
import os
import time
import traceback
from ctypes import wintypes
from pathlib import Path

import psutil
from issue264_validation_runtime import ROOT, ObservedDesktop, local, save, source_manifest


def select_replay_case(source, args):
    if source["candidate"] != args.source_candidate:
        raise ValueError("Recorded snapshot candidate does not match the authorized source")
    selected = [
        case
        for case in source["cases"]
        if case["case"] == args.source_case and case["profile"] == args.source_profile
    ]
    if len(selected) != 1:
        raise ValueError("Replay requires exactly one selected source case/profile")
    case = selected[0]
    requests = case["renderer_snapshots"]
    if len(requests) < 2 or requests[-1]["snapshot"]["blocks"]:
        raise ValueError("Replay must include recorded nonempty output and final clear")
    if any(not request["snapshot"]["blocks"] for request in requests[:-1]):
        raise ValueError("Unexpected intermediate empty snapshot")
    for previous, current in zip(requests, requests[1:]):
        if (
            current["submitted_at"] < previous["submitted_at"]
            or current["snapshot"]["revision"] <= previous["snapshot"]["revision"]
        ):
            raise ValueError("Recorded snapshot order is not monotonic")
    return case


def replay_payload_matches(original, observed, index):
    return observed == {**original, "revision": index + 1}


def install_outbound_observer(desktop, report, requests):
    connection = getattr(desktop.renderer.window._page.session, "_Session__conn")
    writer = getattr(connection, "_FletSocketServer__writer")
    if writer is None:
        raise RuntimeError("The verified Flet socket connection has no active writer")
    report["outbound_transport"] = {
        "connection_type": f"{type(connection).__module__}.{type(connection).__name__}",
        "socket_name": writer.get_extra_info("sockname"),
        "peer_name": writer.get_extra_info("peername"),
        "boundary": "Owned Flet writer.write return and writer.drain return, not native application/paint ACK",
        "events": [],
        "observer_errors": [],
    }
    observation = report["outbound_transport"]
    texts = {
        text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for request in requests
        for block in request["snapshot"]["blocks"]
        for field in ("primary_text", "secondary_text")
        if (text := block.get(field))
    }
    encoded = [(text.encode("utf-8"), digest) for text, digest in texts.items()]
    observation["public_caption_text_sha256"] = sorted(set(texts.values()))
    missing = object()
    previous = {name: vars(writer).get(name, missing) for name in ("write", "drain")}
    original_write, original_drain = writer.write, writer.drain

    def record(stage, data=None, error=None):
        try:
            value = {
                "event": stage,
                "at": time.monotonic(),
                "current_host_revision": desktop.revision,
            }
            if data is not None:
                payload = bytes(data)
                value.update(
                    bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    public_text_matches=[digest for text, digest in encoded if text in payload],
                )
            if error is not None:
                value["exception_class"] = type(error).__name__
                value["message"] = str(error)
            observation["events"].append(value)
        except Exception as error:
            observation["observer_errors"].append(
                {"exception_class": type(error).__name__, "message": str(error)}
            )

    def observed_write(data):
        try:
            result = original_write(data)
        except BaseException as error:
            record("write_failed", error=error)
            raise
        record("write_returned", data=data)
        return result

    async def observed_drain():
        try:
            result = await original_drain()
        except BaseException as error:
            record("drain_failed", error=error)
            raise
        record("drain_returned")
        return result

    writer.write, writer.drain = observed_write, observed_drain

    def restore():
        for name, value in previous.items():
            if value is missing:
                delattr(writer, name)
            else:
                setattr(writer, name, value)
        observation["instance_methods_restored"] = True

    return restore


def owned_window_facts(desktop):
    capture = desktop.capture
    overlay = capture.owned_window(desktop.pid)
    backdrop = capture.owned_window(desktop.backdrop_pid, desktop.title)
    stack = capture.windows()
    getter = capture.u.GetWindowLongPtrW
    getter.argtypes = [wintypes.HWND, ctypes.c_int]
    getter.restype = ctypes.c_ssize_t
    layered = capture.u.GetLayeredWindowAttributes
    layered.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.BYTE),
        ctypes.POINTER(wintypes.DWORD),
    ]
    layered.restype = wintypes.BOOL
    result = {"at": time.monotonic(), "current_host_revision": desktop.revision}
    for label, hwnd, pid in (
        ("overlay", overlay, desktop.pid),
        ("backdrop", backdrop, desktop.backdrop_pid),
    ):
        state = capture.window_state(hwnd, pid)
        ctypes.set_last_error(0)
        style = getter(hwnd, -20)
        error = ctypes.get_last_error()
        if style == 0 and error:
            raise ctypes.WinError(error)
        state.update(
            client_rect=capture.rect(hwnd, client=True),
            window_rect=capture.rect(hwnd),
            z_order_index=stack.index(hwnd),
            extended_style=style,
            layered=bool(style & 0x80000),
            transparent=bool(style & 0x20),
            no_redirection_bitmap=bool(style & 0x200000),
        )
        if state["layered"]:
            key, alpha, flags = wintypes.DWORD(), wintypes.BYTE(), wintypes.DWORD()
            available = bool(
                layered(hwnd, ctypes.byref(key), ctypes.byref(alpha), ctypes.byref(flags))
            )
            state["layered_attributes"] = {
                "available": available,
                "color_key": key.value if available else None,
                "alpha": alpha.value if available else None,
                "flags": flags.value if available else None,
                "error": None if available else ctypes.get_last_error(),
                "limit": "Window layered attributes, not a native caption paint acknowledgement",
            }
        result[label] = state
    result["overlay_above_backdrop"] = stack.index(overlay) < stack.index(backdrop)
    return result


async def run(args):
    if os.environ.get("ISSUE264_VALIDATION_READY_SHA") != args.candidate:
        raise RuntimeError("Use the authorized isolated launcher")
    source_path = args.source_trace.resolve()
    if source_path.parent.parent != ROOT / "build" or not source_path.parent.name.startswith(
        "issue264-validation-"
    ):
        raise ValueError("Only an assigned task-local original trace can be replayed")
    original = source_path.read_bytes()
    if hashlib.sha256(original).hexdigest() != args.source_trace_sha256:
        raise ValueError("Original trace hash does not match the authorized source")
    source = json.loads(original)
    case = select_replay_case(source, args)
    requests = case["renderer_snapshots"]
    offsets = [request["submitted_at"] - requests[0]["submitted_at"] for request in requests]
    desktop = ObservedDesktop(args.output)
    report = {
        "schema": "issue264-presentation-only-replay-2",
        "candidate": args.candidate,
        "source_candidate": source["candidate"],
        "source_trace": source_path.relative_to(ROOT).as_posix(),
        "source_trace_sha256": args.source_trace_sha256,
        "source_case": args.source_case,
        "source_profile": args.source_profile,
        "source_original_readbacks": [
            {
                key: record.get(key)
                for key in ("revision", "readback_valid", "invalid_readback_reason")
            }
            for record in case["renderer_readbacks"]
        ],
        "scope": "One authorized no-model replay of all selected authoritative snapshots at original relative intervals. Only transport revisions may rebase; all other snapshot fields are conserved. No ASR, speaker, translation, source admission, history or TTL inference is rerun. Original invalid images and original coupled latency remain invalid/unrecovered. Host and socket-drain ACKs are not native application/paint ACKs.",
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(
            (ROOT / "native/nemotron/issue264_validation_launch.py").read_bytes()
        ).hexdigest(),
        "sources_sha256": source_manifest(),
        "requested_relative_intervals_s": offsets,
        "owned_window_facts": [],
        "diagnostic_errors": [],
        "model_calls": 0,
        "translation_service_calls": 0,
        "cloud_calls": 0,
    }
    restore = None

    def window_observation():
        try:
            report["owned_window_facts"].append(owned_window_facts(desktop))
        except Exception as error:
            report["diagnostic_errors"].append(
                {
                    "stage": "owned_window_facts",
                    "exception_class": type(error).__name__,
                    "message": str(error),
                }
            )

    try:
        await desktop.start()
        report["owned_renderer_pid"] = desktop.pid
        report["owned_backdrop_pid"] = desktop.backdrop_pid
        restore = install_outbound_observer(desktop, report, requests)
        window_observation()
        started = time.monotonic()
        report["replay_started_at"] = started
        for index, request in enumerate(requests):
            await asyncio.sleep(max(0, started + offsets[index] - time.monotonic()))
            snapshot = local.OverlayPresentationSnapshot.from_dict(request["snapshot"])
            if snapshot.to_dict() != request["snapshot"]:
                raise ValueError("Recorded snapshot decoder round-trip changed its payload")
            await desktop.replace_snapshot(
                snapshot, block_expirations=request.get("block_expirations")
            )
            if not replay_payload_matches(
                request["snapshot"], desktop.snapshots[-1]["snapshot"], index
            ):
                raise ValueError("Replay changed fields other than the transport revision")
            window_observation()
        await asyncio.gather(*desktop.readback_tasks)
        report["observed_owned_children"] = [
            {"pid": child.pid, "name": child.name()}
            for child in psutil.Process().children(recursive=True)
        ]
        if any(
            "llama" in child["name"].lower() or "gpuworker" in child["name"].lower()
            for child in report["observed_owned_children"]
        ):
            raise RuntimeError("Unexpected model child in presentation-only replay")
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != args.source_trace_sha256:
            raise RuntimeError("Original trace changed during replay")
        qualified = sum(bool(record["readback_valid"]) for record in desktop.readbacks)
        report["pixel_qualification"] = {
            "qualified": qualified,
            "invalid": len(desktop.readbacks) - qualified,
            "expected": len(requests) - 1,
        }
        report["status"] = (
            "presentation_replay_completed_requires_image_inspection"
            if qualified == len(requests) - 1
            else "failed_pixel_qualification"
        )
    except Exception as error:
        report["status"] = "failed_observation"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        try:
            await desktop.close()
        except Exception as error:
            report["status"] = "failed_observation"
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
        finally:
            if restore is not None:
                try:
                    restore()
                except Exception as error:
                    report["status"] = "failed_observation"
                    report["observer_restore_error"] = f"{type(error).__name__}: {error}"
            report["snapshots"] = desktop.snapshots
            report["readbacks"] = desktop.readbacks
            report["renderer_events"] = desktop.events
            report["renderer_errors"] = desktop.errors
            report["diagnostics_valid"] = not (
                report["diagnostic_errors"]
                or report.get("outbound_transport", {}).get("observer_errors")
            )
            report["observed_relative_intervals_s"] = [
                request["submitted_at"] - report["replay_started_at"]
                for request in desktop.snapshots
            ]
            save(args.output / "presentation_replay.json", report)
    return 0 if report["status"] == "presentation_replay_completed_requires_image_inspection" else 1


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--source-trace-sha256", required=True)
    parser.add_argument("--source-candidate", required=True)
    parser.add_argument("--source-case", required=True)
    parser.add_argument("--source-profile", choices=("1.04", "0.64"), required=True)
    return parser.parse_args()
