from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import secrets
import sys
import time
import unicodedata
from ctypes import wintypes
from pathlib import Path
from uuid import uuid4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def compact(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text) if not c.isspace())


def source_groups(text, groups):
    positions = []
    normalized = ""
    for index, char in enumerate(text):
        value = compact(char)
        normalized += value
        positions.extend([index] * len(value))
    assert normalized == "".join(g["normalized_source_text"] for g in groups)
    result, offset, start = [], 0, 0
    for group in groups:
        offset += len(group["normalized_source_text"])
        end = positions[offset] if offset < len(positions) else len(text)
        result.append({**group, "text": text[start:end], "source_character_range": [start, end]})
        start = end
    assert "".join(g["text"] for g in result) == text
    return result


def bound_prompt(source_language, target_language):
    from puripuly_heart.config.prompts import get_default_prompt, render_translation_prompt_template
    from puripuly_heart.core.language import get_llm_language_name

    return render_translation_prompt_template(
        get_default_prompt(),
        source_name=get_llm_language_name(source_language),
        target_name=get_llm_language_name(target_language),
        input_channel="peer",
        peer_request_kind="normal",
    )


def assert_bound_body(body, source_language, target_language, provider_options):
    from puripuly_heart.core.language import get_llm_language_name

    prompt = body["messages"][0]["content"]
    assert "${" not in json.dumps(body), "unresolved prompt template variable"
    assert f"Source: {get_llm_language_name(source_language)}" in prompt.splitlines()
    assert [line for line in prompt.splitlines() if line.startswith("Target: ")] == [
        f"Target: {get_llm_language_name(target_language)}"
    ], "wrong target language in effective prompt"
    assert "Channel: peer" in prompt.splitlines()
    assert body["model"] == "google/gemma-4-26b-a4b-it"
    assert body["temperature"] == 0.6
    assert body["reasoning"] == {"effort": "none"}
    assert body["max_tokens"] == 100
    assert body["provider"] == provider_options


def prepare_correction(result, directory):
    from issue373 import decode, encode

    from puripuly_heart.providers.llm.openrouter import HttpxOpenRouterClient

    prior = json.loads((directory / "invalid-template" / "result.json").read_text(encoding="utf-8"))
    assert len(prior["calls"]) == 49
    s1_hashes = {u["request_sha256"] for u in prior["units"] if u["arm"] == "S1"}
    result["calls"] = [c for c in prior["calls"] if c["request_sha256"] in s1_hashes]
    assert len(result["calls"]) == 25
    provider_options = result["calls"][0]["body"]["provider"]
    client = HttpxOpenRouterClient(
        api_key="", model="google/gemma-4-26b-a4b-it", models=("google/gemma-4-26b-a4b-it",)
    )
    prepared = []
    for unit in result["units"]:
        if unit["arm"] != "E0":
            continue
        request = decode(unit["request"])
        request = dataclasses.replace(
            request, system_prompt=bound_prompt(request.source_language, request.target_language)
        )
        unit["request"] = encode(request)
        for name in ("request_sha256", "cache_hit", "mt_status", "translation"):
            unit.pop(name, None)
        kwargs = dataclasses.asdict(request)
        kwargs.pop("utterance_id")
        body = client._build_request_body(**kwargs)
        assert_bound_body(body, request.source_language, request.target_language, provider_options)
        prepared.append({"unit_id": unit["id"], "body": body})
    assert len(prepared) == 24
    for invalid_target in ("${targetName}", "English"):
        body = json.loads(json.dumps(prepared[0]["body"]))
        body["messages"][0]["content"] = body["messages"][0]["content"].replace(
            "Target: Korean", f"Target: {invalid_target}"
        )
        try:
            assert_bound_body(body, "en", "ko", provider_options)
        except AssertionError:
            pass
        else:
            raise AssertionError("literal/wrong-target offline regression was not rejected")
    result["correction_offline_check"] = {
        "valid_complete_E0_bodies": 24,
        "negative_literal_and_wrong_target_checks": 2,
        "prepared_bodies": prepared,
        "adapter_sha256": digest(__file__),
    }
    result["attempt_accounting"] = {
        "invalid_template_actual_calls": 24,
        "reused_valid_s1_actual_calls": 25,
        "corrected_e0_new_actual_calls_expected": 24,
        "total_actual_calls_expected": 73,
        "invalid_evidence_directory": str(directory / "invalid-template"),
        "reason": "Director-authorized owned prompt-binding defect correction, not quality retry",
    }
    result["reused_s1_display"] = [
        {
            **record,
            "screenshot": str(directory / "invalid-template" / Path(record["screenshot"]).name),
        }
        for record in prior["display"]
        if "-S1-" in record["unit_id"]
    ]
    result["display"] = []
    result["display_attempts"] = []
    result["display_arm_filter"] = "E0"
    result["cleanup"] = {}
    result["correction_adapter_sha256"] = digest(__file__)
    return result


def prepare(selection, corpus, output):
    from issue373 import decode, encode

    from puripuly_heart.core.translation_backend import TranslationBackendRequest

    units, cases, provenance = [], [], {}
    for item in selection["selection"]:
        case = {
            k: item[k] for k in ("id", "recording_id", "language", "source_interval_s", "reason")
        }
        case.update(units=[], consequence=item["E0"]["consequence"])
        recording = item["recording_id"]
        paths = {arm: corpus / arm / f"{recording}.json" for arm in ("S1", "E0")}
        data = {arm: json.loads(path.read_text(encoding="utf-8")) for arm, path in paths.items()}
        provenance.update({str(path): digest(path) for path in paths.values()})
        e0 = item["E0"]
        commit = next(e for e in data["E0"]["events"] if e["seq"] == e0["commit_seq"])
        diarized = next(e for e in data["E0"]["events"] if e["seq"] == e0["diarized_event_seq"])
        assert commit["raw"]["message_type"] == "committed_transcript"
        assert commit["raw"]["segment_index"] == diarized["raw"]["segment_index"] == e0["index"]
        assert commit["raw"]["text"] == e0["text"] == diarized["raw"]["text"]
        assert e0["original_commit_text_unchanged"]
        case["native_metadata"] = {
            "segment_index": e0["index"],
            "commit_event_seq": commit["seq"],
            "commit_arrival_s": commit["arrival_s"],
            "diarized_event_seq": diarized["seq"],
            "diarized_arrival_s": diarized["arrival_s"],
            "metadata_delay_s": diarized["arrival_s"] - commit["arrival_s"],
            "word_speakers": sorted(
                {w["speaker_id"] for w in diarized["raw"]["words"] if w.get("speaker_id")}
            ),
            "whole_commit_settled_state": "mixed" if len(e0["groups"]) > 1 else "identified",
            "late_generic_repair": "not supported/tested; original whole commit remains gray",
        }
        palettes = {}
        for event in data["E0"]["events"]:
            for word in event.get("raw", {}).get("words") or []:
                speaker = word.get("speaker_id")
                if speaker and speaker not in palettes:
                    palettes[speaker] = len(palettes)
        case["native_palette"] = palettes
        for match in item["S1"]["request_matches"]:
            index = match["index"]
            request = decode(data["S1"]["application"]["requests"][index]["request"])
            assert request.text == match["text"]
            candidates = [
                m
                for m in item["S1"]["event_matches"]
                if m["target"] == str(request.utterance_id) and m["text"] == request.text
            ]
            event_index = min(candidates, key=lambda m: abs(m["arrival_s"] - match["arrival_s"]))[
                "index"
            ]
            event = decode(data["S1"]["application"]["events"][event_index]["event"])
            unit = {
                "id": f"{item['id']}-S1-{index}",
                "case_id": item["id"],
                "arm": "S1",
                "kind": "captured_request",
                "request_index": index,
                "event_index": event_index,
                "original_arrival_s": match["arrival_s"],
                "request": encode(request),
                "event": encode(event),
                "selected_eventual": index in item["S1"]["selected_request_indices"],
            }
            units.append(unit)
            case["units"].append(unit["id"])
        target_language = "en" if item["language"] == "ko" else "ko"
        request = TranslationBackendRequest(
            utterance_id=uuid4(),
            text=e0["text"],
            system_prompt=bound_prompt(item["language"], target_language),
            source_language=item["language"],
            target_language=target_language,
            context="",
            scene_participant_count=None,
            max_output_tokens=None,
        )
        unit = {
            "id": f"{item['id']}-E0-commit",
            "case_id": item["id"],
            "arm": "E0",
            "kind": "original_complete_commit",
            "request": encode(request),
            "original_arrival_s": commit["arrival_s"],
            "assignment": None,
        }
        units.append(unit)
        case["units"].append(unit["id"])
        projected = source_groups(e0["text"], e0["groups"])
        case["projected_groups"] = projected
        if len(projected) > 1:
            for index, group in enumerate(projected):
                grouped_request = dataclasses.replace(
                    request, utterance_id=uuid4(), text=group["text"]
                )
                unit = {
                    "id": f"{item['id']}-E0-group-{index}",
                    "case_id": item["id"],
                    "arm": "E0",
                    "kind": "counterfactual_current_composition_new_source_unit",
                    "request": encode(grouped_request),
                    "original_arrival_s": diarized["arrival_s"],
                    "group": group,
                    "palette_index": palettes.get(group["representative"]),
                }
                units.append(unit)
                case["units"].append(unit["id"])
        cases.append(case)
    result = {
        "schema_version": 1,
        "selection_sha256": digest(output["selection_path"]),
        "manifest_sha256": selection["manifest_sha256"],
        "source_hashes": provenance,
        "runtime": {
            "executable": sys.executable,
            "python": sys.version,
            "packages": {
                n: importlib.metadata.version(n)
                for n in ("Pillow", "flet", "flet-desktop", "websockets", "httpx", "keyring")
            },
        },
        "executed_adapter_sha256": digest(__file__),
        "profile": {
            "model": "google/gemma-4-26b-a4b-it",
            "models": ["google/gemma-4-26b-a4b-it"],
            "routing": "LATENCY",
            "provider_routing": "DEFAULT",
            "temperature": 0.6,
            "reasoning_effort": "none",
            "max_tokens_default": 100,
            "timeout_s": 30,
            "setting_provenance": "Director-declared shared experiment configuration, not observed private user settings",
        },
        "context_policy": {
            "S1": "Unchanged captured request, including prompt/context/languages/max_output_tokens; captured context contains prior source turns, not newly generated MT.",
            "E0": "Current default prompt bound through production render_translation_prompt_template, recording source language, target ko except ko->en, peer/normal; empty context because sparse intervals do not replay prior MT history; no reference text. New complete projected source units translated independently, not slices of target strings.",
        },
        "limits": [
            "Settled MT then controlled display replay, not ASR->MT->GUI latency.",
            "E0 raw indexed commits are not production Scribe integration.",
            "Projected source-group translations are explicit extra work, not free recoloring or actual late-repair integration.",
            "Single-speaker settled presentation uses experiment-supplied native metadata only; original commits remain gray. Mixed whole commits never borrow reference color.",
        ],
        "cases": cases,
        "units": units,
        "calls": [],
        "display": [],
        "cleanup": {},
    }
    return result


async def translate(result, output):
    from issue373 import decode

    from puripuly_heart.core.openrouter_routing import (
        OpenRouterProviderRouting,
        OpenRouterRoutingMode,
    )
    from puripuly_heart.core.translation_backend import LlmTranslationBackend
    from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider

    key = os.environ.get("OPENROUTER_API_KEY")
    key_source = "environment" if key else "keyring:puripuly-heart/openrouter_api_key"
    if not key:
        import keyring

        key = keyring.get_password("puripuly-heart", "openrouter_api_key")
    if not key:
        raise RuntimeError("read_only_openrouter_credential_absent")
    provider = OpenRouterLLMProvider(
        api_key=key,
        model=result["profile"]["model"],
        models=tuple(result["profile"]["models"]),
        routing_mode=OpenRouterRoutingMode.LATENCY,
        provider_routing=OpenRouterProviderRouting.DEFAULT,
        max_tokens=100,
        timeout=30.0,
    )
    backend = LlmTranslationBackend(provider)
    client = provider._get_client()
    receipts = []

    async def response_receipt(response):
        await response.aread()
        payload = response.json()
        receipts.append(
            {
                "http_status": response.status_code,
                "response_received_mono_s": time.monotonic(),
                **{k: payload[k] for k in ("id", "model", "provider", "usage") if k in payload},
                "finish_reasons": [c.get("finish_reason") for c in payload.get("choices", [])],
            }
        )

    http = await client._get_http_client()
    http.event_hooks["response"].append(response_receipt)
    result["credential_lookup"] = {"source": key_source, "read_only": True, "value_recorded": False}
    cache = {call["request_sha256"]: call for call in result["calls"]}
    try:
        for unit in result["units"]:
            if unit.get("mt_status") in {"completed", "failed"}:
                unit["reused_saved_response"] = True
                continue
            request = decode(unit["request"])
            kwargs = dataclasses.asdict(request)
            kwargs.pop("utterance_id")
            body = client._build_request_body(**kwargs)
            if unit["arm"] == "E0" and "correction_offline_check" in result:
                prepared = next(
                    p
                    for p in result["correction_offline_check"]["prepared_bodies"]
                    if p["unit_id"] == unit["id"]
                )
                assert body == prepared["body"], "prepared body changed before paid execution"
                assert_bound_body(
                    body,
                    request.source_language,
                    request.target_language,
                    result["calls"][0]["body"]["provider"],
                )
            sha = hashlib.sha256(
                json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            unit["request_sha256"] = sha
            unit["cache_hit"] = sha in cache
            if sha not in cache:
                response_start = len(receipts)
                call = {
                    "request_sha256": sha,
                    "body": body,
                    "started_mono_s": time.monotonic(),
                    "unit_id": unit["id"],
                }
                try:
                    value = await backend.translate(request)
                    call.update(status="completed", translation=value.text)
                except Exception as exc:
                    call.update(
                        status="failed",
                        error=type(exc).__name__,
                        status_code=getattr(exc, "status_code", None),
                    )
                call["completed_mono_s"] = time.monotonic()
                call["backend_elapsed_s"] = call["completed_mono_s"] - call["started_mono_s"]
                call["responses"] = receipts[response_start:]
                result["calls"].append(call)
                cache[sha] = call
            unit["mt_status"] = cache[sha]["status"]
            unit["translation"] = cache[sha].get("translation")
            save(output, result)
            print(
                json.dumps(
                    {
                        "unit": unit["id"],
                        "status": unit["mt_status"],
                        "cache_hit": unit["cache_hit"],
                    }
                ),
                flush=True,
            )
    finally:
        await backend.close()
        result["cleanup"]["backend_closed"] = True
        save(output, result)


class NativeRegion:
    def __init__(self):
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        self.user.CreateWindowExW.restype = wintypes.HWND
        self.user.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        self.user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user.IsWindow.argtypes = [wintypes.HWND]
        self.user.DestroyWindow.argtypes = [wintypes.HWND]
        self.user.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        self.backdrop = None

    def start(self):
        self.backdrop = self.user.CreateWindowExW(
            0x00000008 | 0x00000080,
            "STATIC",
            "issue441-owned-neutral-backdrop",
            0x80000000 | 0x10000000 | 0x00000005,
            0,
            0,
            1344,
            336,
            None,
            None,
            None,
            None,
        )
        if not self.backdrop:
            raise ctypes.WinError(ctypes.get_last_error())
        self.user.SetWindowPos(self.backdrop, wintypes.HWND(-1), 0, 0, 1344, 336, 0x0040)
        assert self.bounds(self.backdrop) == [0, 0, 1344, 336]

    def pump(self):
        message = wintypes.MSG()
        while self.user.PeekMessageW(ctypes.byref(message), None, 0, 0, 1):
            self.user.TranslateMessage(ctypes.byref(message))
            self.user.DispatchMessageW(ctypes.byref(message))

    def bounds(self, hwnd):
        rect = wintypes.RECT()
        if not self.user.GetWindowRect(hwnd, ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())
        return [rect.left, rect.top, rect.right, rect.bottom]

    def windows(self, pid):
        found = []

        def callback(hwnd, _):
            owner = wintypes.DWORD()
            self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid and self.user.IsWindowVisible(hwnd):
                found.append({"hwnd": int(hwnd), "pid": pid, "bounds": self.bounds(hwnd)})
            return True

        self.user.EnumWindows(self.callback_type(callback), 0)
        return found

    def secure_region(self, hwnd):
        assert self.bounds(hwnd) == [0, 0, 1344, 336]
        self.user.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 1344, 336, 0x0040)
        self.user.SetWindowPos(self.backdrop, hwnd, 0, 0, 1344, 336, 0x0040 | 0x0010)
        self.pump()

    def process_handle(self, pid):
        handle = self.kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def close(self):
        if self.backdrop:
            self.user.DestroyWindow(self.backdrop)
            self.pump()
            assert not self.user.IsWindow(self.backdrop)


async def display(result, output, directory):
    from issue373 import decode, encode
    from PIL import ImageGrab

    from puripuly_heart.config.overlay_calibration import OverlayCalibration
    from puripuly_heart.core.clock import SystemClock
    from puripuly_heart.core.overlay.manifest import OVERLAY_CONTRACT_VERSION, OverlayLaunchManifest
    from puripuly_heart.core.overlay.presenter import OverlayPresenter
    from puripuly_heart.core.overlay.protocol import OverlayPresentationSnapshot
    from puripuly_heart.core.overlay.sink import TranslationFinal
    from puripuly_heart.domain.models import SpeakerAssignment, SpeakerAttribution, SpeakerKey
    from puripuly_heart.ui.desktop_overlay import (
        DesktopOverlayRenderer,
        DiagnosticLocalRendererPort,
    )
    from puripuly_heart.ui.desktop_overlay_repro import LocalAuthenticatedRawIngress

    removed_environment = {}
    for name in list(os.environ):
        if any(
            suffix in name.upper() for suffix in ("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "SECRET")
        ):
            removed_environment[name] = os.environ.pop(name)
    result["isolation"] = {
        "normal_application_startup": False,
        "settings_loaded": False,
        "capture_discovery_osc_telemetry": False,
        "renderer_child_credential_environment_removed": sorted(removed_environment),
        "token_manifest": "fresh in-memory only; never written",
        "python_import_root": str(Path(sys.modules["puripuly_heart"].__file__).resolve()),
        "renderer": "production DesktopOverlayRenderer + default FletDesktopRendererWindow",
        "backdrop": "owned Win32 STATIC gray rectangle covers full native screenshot region",
    }
    region = NativeRegion()
    token = secrets.token_urlsafe(32)
    instance_id = "issue441-mt-" + uuid4().hex
    ingress = LocalAuthenticatedRawIngress(
        token, OverlayPresentationSnapshot(revision=0, blocks=[]), instance_id
    )
    port = DiagnosticLocalRendererPort(acknowledgement_timeout_s=10.0)
    renderer = None
    renderer_task = consumer = None
    presenters = []
    ready = asyncio.Event()
    receipts, lifecycle, completions = [], [], {}
    process_handle = None
    result["renderer_receipts"] = receipts
    result["renderer_lifecycle"] = lifecycle

    async def lifecycle_sink(event):
        lifecycle.append({"mono_s": time.monotonic(), **event})
        if event.get("type") == "overlay_ready":
            ready.set()

    async def consume():
        while True:
            try:
                envelope = await port.next_event()
            except Exception as exc:
                if type(exc).__name__ == "RendererDiagnosticPortClosed":
                    return
                raise
            record = dict(envelope.record)
            receipts.append({"mono_s": time.monotonic(), **record})
            revision = record.get("renderer_revision")
            kind = record.get("event_type")
            if kind == "render_commit":
                port.acknowledge_render_commit(revision)
            if kind in ("render_commit_acknowledgement", "failed") or (
                kind == "receipt" and record.get("actual_disposition") != "committed"
            ):
                completion = completions.get(revision)
                if completion and not completion.done():
                    completion.set_result(record)

    try:
        region.start()
        bridge_url = await ingress.start()
        manifest = OverlayLaunchManifest(
            contract_version=OVERLAY_CONTRACT_VERSION,
            app_version="issue441-bounded-mt-display",
            overlay_instance_id=instance_id,
            bridge_url=bridge_url,
            session_token=token,
            parent_pid=os.getpid(),
            startup_deadline_ms=10000,
            log_dir=str(directory),
            log_level="WARNING",
            locale="en",
        )
        renderer = DesktopOverlayRenderer(
            manifest,
            lifecycle_sink=type("Lifecycle", (), {"emit": staticmethod(lifecycle_sink)})(),
            diagnostic_port=port,
        )
        consumer = asyncio.create_task(consume())
        renderer_task = asyncio.create_task(renderer.run())
        await asyncio.wait_for(ready.wait(), 30)
        await ingress.wait_connected()
        await ingress.send_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        pid = renderer.window._flet_process_pid
        if pid is None:
            pid = renderer.window._owner_process_pid()
        windows = region.windows(pid)
        exact = [w for w in windows if w["bounds"] == [0, 0, 1344, 336]]
        assert len(exact) == 1, windows
        owned = exact[0]
        process_handle = region.process_handle(pid)
        result["owned_window"] = {
            **owned,
            "backdrop_hwnd": int(region.backdrop),
            "backdrop_pid": os.getpid(),
            "all_visible_owned_pid_windows": windows,
        }
        region.secure_region(owned["hwnd"])
        revision = 0
        sequence = 0
        by_id = {u["id"]: u for u in result["units"]}

        async def show(presenter, unit, event, stage):
            nonlocal revision, sequence
            sequence += 1
            event = dataclasses.replace(
                event,
                event_id=f"{instance_id}:{sequence}",
                seq=sequence,
                created_at=presenter.clock.now(),
            )
            accepted = await presenter.emit(event)
            await asyncio.sleep(1.1)
            region.pump()
            revision += 1
            snapshot = dataclasses.replace(presenter.snapshot(), revision=revision)
            completion = asyncio.get_running_loop().create_future()
            completions[revision] = completion
            started = time.monotonic()
            await ingress.write_batch([snapshot])
            outcome = await asyncio.wait_for(completion, 15)
            if outcome.get("actual_disposition") != "committed":
                raise RuntimeError("renderer_revision_not_committed:" + str(outcome))
            await asyncio.sleep(0.35)
            region.secure_region(owned["hwnd"])
            region.pump()
            filename = f"{revision:03d}-{unit['id']}-{stage}.png"
            path = directory / filename
            capture_started = time.monotonic()
            image = ImageGrab.grab(
                bbox=tuple(owned["bounds"]), include_layered_windows=True, all_screens=True
            )
            image.save(path)
            record = {
                "unit_id": unit["id"],
                "case_id": unit["case_id"],
                "stage": stage,
                "revision": revision,
                "presenter_receipt": encode(accepted),
                "snapshot": snapshot.to_dict(),
                "renderer_receipt": outcome,
                "snapshot_send_mono_s": started,
                "capture_started_mono_s": capture_started,
                "capture_completed_mono_s": time.monotonic(),
                "screenshot": str(path),
                "screenshot_sha256": digest(path),
                "image_size": list(image.size),
                "native_window": owned,
                "event": encode(event),
            }
            result["display"].append(record)
            save(output, result)
            print(
                json.dumps({"display": unit["id"], "stage": stage, "revision": revision}),
                flush=True,
            )

        def new_presenter():
            presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=SystemClock())
            presenters.append(presenter)
            return presenter

        def generic_event(unit, assignment=None):
            request = decode(unit["request"])
            return TranslationFinal(
                event_id=uuid4().hex,
                seq=0,
                utterance_id=request.utterance_id,
                channel="peer",
                created_at=time.monotonic(),
                text=unit["translation"],
                source_text=request.text,
                source_language=request.source_language,
                target_language=request.target_language,
                speaker_assignment=assignment,
            )

        for case in result["cases"]:
            s1 = new_presenter()
            for unit_id in case["units"]:
                unit = by_id[unit_id]
                if (
                    result.get("display_arm_filter") == "E0"
                    or unit["arm"] != "S1"
                    or unit["mt_status"] != "completed"
                ):
                    continue
                event = dataclasses.replace(decode(unit["event"]), text=unit["translation"])
                await show(s1, unit, event, "captured-revision")
            await s1.close()
            original = by_id[f"{case['id']}-E0-commit"]
            if original["mt_status"] == "completed":
                presenter = new_presenter()
                await show(presenter, original, generic_event(original), "original-whole-gray")
                await presenter.close()
            settled = new_presenter()
            grouped = [
                by_id[unit_id]
                for unit_id in case["units"]
                if by_id[unit_id]["kind"] == "counterfactual_current_composition_new_source_unit"
            ]
            if not grouped:
                grouped = [original]
            for unit in grouped:
                if unit["mt_status"] != "completed":
                    continue
                group = unit.get("group", case["projected_groups"][0])
                speaker = group["representative"]
                assignment = SpeakerAssignment(
                    attribution=SpeakerAttribution(
                        state="identified" if speaker else "missing",
                        key=(
                            SpeakerKey(
                                source="scribe",
                                speaker_scope_id=group["scope"],
                                local_speaker_id=speaker,
                            )
                            if speaker
                            else None
                        ),
                        source="scribe",
                        speaker_scope_id=group["scope"],
                    ),
                    palette_index=case["native_palette"].get(speaker),
                    scope_order=(1, 0, 0),
                )
                await show(
                    settled,
                    unit,
                    generic_event(unit, assignment),
                    "settled-native-metadata-new-unit",
                )
            await settled.close()
    except Exception as exc:
        result["display_failure"] = {"error": type(exc).__name__, "message": str(exc)}
        import traceback

        result["display_failure"]["traceback"] = traceback.format_exc()
        raise
    finally:
        for presenter in presenters:
            with contextlib.suppress(Exception):
                await presenter.close()
        result["cleanup"]["presenters_closed"] = all(p._closed for p in presenters)
        if renderer is not None:
            await renderer.shutdown()
            result["cleanup"]["renderer_shutdown"] = renderer.is_shutdown
            result["cleanup"]["renderer_delivery"] = renderer.diagnostic_delivery_receipt()
        await ingress.close()
        result["cleanup"]["ingress_closed"] = True
        if renderer_task:
            try:
                result["cleanup"]["renderer_exit_code"] = await asyncio.wait_for(renderer_task, 15)
            except Exception as exc:
                result["cleanup"]["renderer_exit_error"] = type(exc).__name__
        if consumer:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        if process_handle:
            result["cleanup"]["owned_flet_process_exited"] = (
                region.kernel.WaitForSingleObject(process_handle, 10000) == 0
            )
            region.kernel.CloseHandle(process_handle)
        region.close()
        result["cleanup"]["backdrop_destroyed"] = True
        os.environ.update(removed_environment)
        save(output, result)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--corpus", default=".tmp/issue441/selected-runs")
    parser.add_argument("--output-dir", default=".tmp/issue441/mt-display")
    parser.add_argument(
        "--phase", choices=("mt", "display", "prepare-correction", "mt-correction"), required=True
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path[:0] = [str(root / "src"), str(root)]
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "result.json"
    selection_path = Path(args.selection)
    if args.phase == "mt":
        if output.exists():
            raise RuntimeError("MT output already exists; no automatic live retries")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        assert len(selection["selection"]) == 7
        result = prepare(selection, Path(args.corpus), {"selection_path": selection_path})
        save(output, result)
        await translate(result, output)
    elif args.phase == "prepare-correction":
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["selection_sha256"] == digest(selection_path)
        result = prepare_correction(result, directory)
        save(output, result)
        print(
            json.dumps({"offline_prepared_valid": 24, "negative_regressions_rejected": 2}),
            flush=True,
        )
    elif args.phase == "mt-correction":
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["selection_sha256"] == digest(selection_path)
        assert (
            len(result["calls"]) == 25
            and result["correction_offline_check"]["valid_complete_E0_bodies"] == 24
        )
        await translate(result, output)
    else:
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["selection_sha256"] == digest(selection_path)
        result.setdefault("display_attempts", []).append(
            {
                "started_mono_s": time.monotonic(),
                "adapter_sha256": digest(__file__),
                "runtime_executable": sys.executable,
            }
        )
        await display(result, output, directory)
    print(
        json.dumps(
            {
                "calls": len(result["calls"]),
                "failures": sum(c["status"] == "failed" for c in result["calls"]),
                "cache_hits": sum(u.get("cache_hit", False) for u in result["units"]),
                "captures": len(result["display"]),
                "cleanup": result["cleanup"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
