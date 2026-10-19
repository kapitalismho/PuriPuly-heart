"""Run benchmark cases through the actual development host.

Example:
  python -m bench.run --root C:\\tmp\\diarization_bench\\run1 --routes local_qwen_gpu --arms baseline,p104,p064
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from bench import env, sources

ARMS = {"baseline": None, "p104": "1.04", "p064": "0.64"}
ROUTES = {
    "local_qwen_gpu": {"languages": {"en", "ko", "ja"}, "secrets": {}},
    "local_parakeet_v3": {"languages": {"en"}, "secrets": {}},
    "soniox": {"languages": {"en", "ko", "ja"}, "secrets": {"soniox_api_key": "SONIOX_API_KEY"}},
    "deepgram": {
        "languages": {"en", "ko", "ja"},
        "secrets": {"deepgram_api_key": "DEEPGRAM_API_KEY"},
    },
    "gemini_transcribe": {
        "languages": {"en", "ko", "ja"},
        "secrets": {"gemini_transcribe_api_key": "GOOGLE_API_KEY2"},
    },
    "qwen_audio": {
        "languages": {"en", "ko", "ja"},
        "secrets": {"alibaba_api_key_beijing": "ALIBABA_API_KEY_BEIJING"},
        "host_env": "ALIBABA_API_HOST_BEIJING",
    },
}
TARGET = {"en": "ko", "ko": "en", "ja": "en"}
NEMOTRON_ARTIFACT = (
    Path.home() / "AppData" / "Local" / "Temp" / "puripuly-nemotron-bonenk" / "model.gguf"
)
LEAD_IN_S = 1.0
QUIET_S = 4.0
MIN_TAIL_S = 6.0
MAX_TAIL_S = 30.0


def log(*parts: object) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def _save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=1, default=str, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=env.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def _sanitize(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if any(
                word in k.lower()
                for word in (
                    "secret",
                    "token",
                    "api_host",
                    "credential",
                    "fingerprint",
                    "anonymous_id",
                    "installation_id",
                )
            ):
                out[k] = "<omitted>"
            else:
                out[k] = _sanitize(v)
        return out
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


class Runner:
    def __init__(self, args) -> None:
        from bench.app import BenchApp

        self.args = args
        self.root = Path(args.root).resolve()
        self.app = BenchApp(self.root)
        self.credentials = env.provider_credentials()
        self.manifest_cases = {
            c["case_id"]: c
            for c in json.loads(
                (env.BENCH_ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8")
            )["cases"]
        }
        self.language = None
        self.diagnostic_after = 0

    async def start(self) -> None:
        await self.app.start()
        app = self.app
        await app.command(
            "capture.set", {"channel": "self", "enabled": False}, accept=("applied", "degraded")
        )
        await app.command("translation.set", {"enabled": False}, accept=("applied", "degraded"))
        await app.settings_apply(
            {"chatbox.activation_notice.enabled": False, "telemetry.enabled": False},
            accept=("applied", "degraded"),
        )
        await app.command("gpu.discover", {}, accept=("applied", "degraded"))
        status = await app.query("speaker.status")
        if not (status["speaker"].get("assets") or {}).get("ready"):
            await app.command("speaker.install", {"source_path": str(NEMOTRON_ARTIFACT)})
        await app.command("overlay.set", {"enabled": True}, accept=("applied", "degraded"))

    async def stop(self) -> None:
        await self.app.stop()

    async def configure_route(self, route: str) -> dict:
        spec = ROUTES[route]
        for name, env_name in spec["secrets"].items():
            value = self.credentials.get(env_name)
            if not value:
                raise RuntimeError(f"missing credential {env_name} for {route}")
            await self.app.command(
                "secrets.set", {"name": name, "value": value}, accept=("applied", "degraded")
            )
        host_env = spec.get("host_env")
        if host_env and self.credentials.get(host_env):
            await self.app.settings_apply(
                {
                    "translation.qwen.region": "beijing",
                    "translation.qwen.beijing.api_host": self.credentials[host_env],
                },
                accept=("applied", "degraded"),
            )
        result = await self.app.command(
            "provider.apply", {"channel": "both", "provider": route}, accept=("applied", "degraded")
        )
        self.language = None
        return _sanitize(result)

    async def configure_arm(self, arm: str) -> dict:
        profile = ARMS[arm]
        if profile is None:
            result = await self.app.command(
                "speaker.configure", {"enabled": False}, accept=("applied", "degraded")
            )
        else:
            result = await self.app.command(
                "speaker.configure",
                {"enabled": True, "profile": profile, "vulkan_device_id": self.args.vulkan_device},
                accept=("applied", "degraded"),
            )
            if result["speaker"]["state"] != "ready":
                raise RuntimeError(f"speaker runtime not ready: {result['speaker'].get('reason')}")
        return _sanitize(result)

    async def set_language(self, language: str) -> None:
        if language == self.language:
            return
        await self.app.settings_apply(
            {
                "languages": {
                    "peer_source": language,
                    "peer_target": TARGET[language],
                    "peer_source_mode": "manual",
                }
            },
            accept=("applied", "degraded"),
        )
        self.language = language

    async def new_diagnostics(self) -> list:
        try:
            history = await self.app.query("diagnostics.history", {"after": self.diagnostic_after})
        except Exception:
            history = await self.app.query("diagnostics.history", {})
        records = [
            r
            for r in history.get("records", [])
            if r.get("diagnostic_sequence", 0) > self.diagnostic_after
        ]
        if history.get("latest_sequence"):
            self.diagnostic_after = max(self.diagnostic_after, history["latest_sequence"])
        return records

    async def run_case(self, case_id: str, run_dir: Path, arm: str, attempt: int = 1) -> dict:
        from bench import observe

        case = self.manifest_cases[case_id]
        tag = case_id if attempt == 1 else f"{case_id}#a{attempt}"
        wav = sources.CACHE / "cases" / f"{case_id}.wav"
        record_path = (
            run_dir
            / "sources"
            / (f"{case_id}.json" if attempt == 1 else f"{case_id}.a{attempt}.json")
        )
        (self.app.control_dir / "next_case.json").write_text(
            json.dumps(
                {
                    "case_id": tag,
                    "wav": str(wav),
                    "record_path": str(record_path),
                    "lead_in_s": LEAD_IN_S,
                }
            ),
            encoding="utf-8",
        )
        await self.new_diagnostics()
        started = time.monotonic()
        on = await self.app.command(
            "capture.set",
            {"channel": "peer", "enabled": True},
            accept=("applied", "degraded", "failed"),
        )
        channel = on.get("channel", {})
        if on["status"] != "applied" or not channel.get("effective_active"):
            await self.app.command(
                "capture.set",
                {"channel": "peer", "enabled": False},
                accept=("applied", "degraded", "failed"),
            )
            return {
                "status": "blocked",
                "reason": "peer_capture_not_active",
                "receipt": _sanitize(on),
            }
        audio_end = started + LEAD_IN_S + float(case["duration_s"])
        await asyncio.sleep(max(0.0, audio_end - time.monotonic()))
        timed_out = False
        while True:
            now = time.monotonic()
            if now - audio_end >= MIN_TAIL_S and now - observe.activity["last"] >= QUIET_S:
                break
            if now - audio_end >= MAX_TAIL_S:
                timed_out = True
                break
            await asyncio.sleep(0.25)
        status = await self.app.query("app.status")
        peer = status["channels"]["peer"]
        await self.app.command(
            "capture.set",
            {"channel": "peer", "enabled": False},
            accept=("applied", "degraded", "failed"),
        )
        await asyncio.sleep(0.5)
        if ARMS[arm] is not None:
            await self.app.command("speaker.reset", {}, accept=("applied", "degraded"))
        diagnostics = await self.new_diagnostics()
        errors = [d for d in diagnostics if d.get("severity") == "error"]
        incidents = [d for d in errors if str(d.get("component", "")).startswith("overlay")]
        failed = peer.get("failure_reason") or len(errors) > len(incidents)
        return {
            "status": "failed" if failed else "completed",
            "observation_tag": tag,
            "attempt": attempt,
            "incidents": [
                {k: d.get(k) for k in ("component", "stage", "code", "occurred_at")}
                for d in incidents
            ],
            "drain_timed_out": timed_out,
            "elapsed_s": round(time.monotonic() - started, 2),
            "effective_vad_hangover_ms": peer.get("capture", {}).get("effective_vad_hangover_ms"),
            "peer_failure_reason": peer.get("failure_reason"),
            "diagnostics": _sanitize(diagnostics),
        }

    async def run(self) -> int:
        args = self.args
        cases = [c for c in self.manifest_cases if not args.cases or c in args.cases]
        cases.sort(
            key=lambda c: (
                self.manifest_cases[c]["language"] != "en",
                self.manifest_cases[c]["language"],
                c,
            )
        )
        run_manifest = {
            "schema": "diarization-bench/run/v1",
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "python": sys.version,
            "platform": platform.platform(),
            "args": vars(args),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "parameters": {
                "lead_in_s": LEAD_IN_S,
                "quiet_s": QUIET_S,
                "min_tail_s": MIN_TAIL_S,
                "max_tail_s": MAX_TAIL_S,
            },
        }
        await self.start()
        try:
            run_manifest["gpu"] = _sanitize(await self.app.query("gpu.status"))
            for index, route in enumerate(args.routes):
                route_dir = self.root / "runs" / route
                try:
                    run_manifest.setdefault("routes", {})[route] = await self.configure_route(route)
                except Exception as exc:
                    log("route blocked", route, repr(exc))
                    _save(route_dir / "blocked.json", {"route": route, "error": repr(exc)})
                    continue
                arms = list(args.arms)
                if args.rotate_arms:
                    arms = arms[index % len(arms) :] + arms[: index % len(arms)]
                for arm in arms:
                    try:
                        arm_receipt = await self.configure_arm(arm)
                    except Exception as exc:
                        log("arm blocked", route, arm, repr(exc))
                        _save(
                            route_dir / arm / "blocked.json",
                            {"route": route, "arm": arm, "error": repr(exc)},
                        )
                        continue
                    for rep in range(args.repeat):
                        run_dir = route_dir / arm / f"r{rep + 1}"
                        _save(
                            run_dir / "arm.json",
                            {
                                "route": route,
                                "arm": arm,
                                "repetition": rep + 1,
                                "speaker": arm_receipt,
                                "settings": _sanitize(await self.app.query("settings.current")),
                            },
                        )
                        from bench import observe

                        observe.set_path(run_dir / "observations.jsonl")
                        for case_id in cases:
                            out = run_dir / "cases" / f"{case_id}.json"
                            previous = (
                                json.loads(out.read_text(encoding="utf-8"))
                                if out.is_file()
                                else None
                            )
                            if previous and previous.get("status") in (
                                "completed",
                                "not_applicable",
                            ):
                                continue
                            prior_attempts = (
                                [] if previous is None else previous.get("prior_attempts", [])
                            )
                            if previous is not None:
                                prior_attempts = prior_attempts + [
                                    {
                                        k: previous.get(k)
                                        for k in ("status", "reason", "attempt", "diagnostics")
                                    }
                                ]
                            attempt = len(prior_attempts) + 1
                            language = self.manifest_cases[case_id]["language"]
                            if language not in ROUTES[route]["languages"]:
                                _save(
                                    out,
                                    {
                                        "status": "not_applicable",
                                        "reason": f"{route} does not support {language}",
                                    },
                                )
                                continue
                            try:
                                await self.set_language(language)
                                result = await self.run_case(case_id, run_dir, arm, attempt)
                            except Exception as exc:
                                result = {
                                    "status": "failed",
                                    "reason": repr(exc),
                                    "traceback": traceback.format_exc(),
                                }
                                try:
                                    await self.app.command(
                                        "capture.set",
                                        {"channel": "peer", "enabled": False},
                                        accept=("applied", "degraded", "failed"),
                                    )
                                except Exception:
                                    pass
                            _save(
                                out,
                                {
                                    "case_id": case_id,
                                    "route": route,
                                    "arm": arm,
                                    "repetition": rep + 1,
                                    "prior_attempts": prior_attempts,
                                    **result,
                                },
                            )
                            log(
                                route,
                                arm,
                                f"r{rep + 1}",
                                case_id,
                                result["status"],
                                result.get("reason", ""),
                            )
        finally:
            run_manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _save(self.root / f"run-manifest-{int(time.time())}.json", run_manifest)
            await self.stop()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--routes", type=lambda s: s.split(","), default=["local_qwen_gpu"])
    parser.add_argument("--arms", type=lambda s: s.split(","), default=list(ARMS))
    parser.add_argument("--cases", type=lambda s: s.split(","), default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--vulkan-device", default="0")
    parser.add_argument(
        "--rotate-arms",
        action="store_true",
        help="rotate arm order per route; off by default because disabled-speaker arms "
        "after enabled arms inherit stale evidence (audit D10)",
    )
    args = parser.parse_args()
    unknown = [r for r in args.routes if r not in ROUTES] + [a for a in args.arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown routes/arms: {unknown}")
    return asyncio.run(Runner(args).run())


if __name__ == "__main__":
    raise SystemExit(main())
