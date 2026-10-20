# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "websockets==16.1.1",
#   "google-genai>=2.21.0,<3.0.0",
#   "httpx>=0.27",
#   "numpy>=2.0",
#   "onnxruntime==1.28.0",
#   "rapidfuzz>=3.9",
# ]
# ///
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import gzip
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import analyze
import fleurs
import providers

LANGS = ("ko_kr", "ja_jp", "en_us")
KEY_CANDIDATES = {
    "soniox": ("SONIOX_API_KEY",),
    "deepgram": ("DEEPGRAM_API_KEY",),
    "qwen": (
        "ALIBABA_API_KEY_BEIJING",
        "ALIBABA_API_KEY",
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY",
        "ALIBABA_CLOUD_API_KEY",
    ),
    "gemini": ("GOOGLE_API_KEY_2", "GOOGLE_API_KEY2", "GEMINI_API_KEY_2", "GOOGLE2_API_KEY"),
}


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[name] = value
    return values


def _resolve_keys(env_files: list[Path], overrides: dict[str, str]) -> dict[str, tuple[str, str]]:
    merged: dict[str, str] = {}
    for path in env_files:
        merged.update(_read_env_file(path))
    merged.update({k: v for k, v in os.environ.items() if k not in merged})
    resolved: dict[str, tuple[str, str]] = {}
    for provider, candidates in KEY_CANDIDATES.items():
        names = (overrides[provider],) if provider in overrides else candidates
        for name in names:
            if merged.get(name):
                resolved[provider] = (name, merged[name])
                break
    return resolved


async def _run_one(provider, variant, lang, data_dir, key, options, runs_dir) -> None:
    reference, pcm = fleurs.load(lang, data_dir)
    rec = providers.Recorder()
    schedule = providers.finalize_schedule(variant, reference)
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    error = None
    config: dict = {}
    try:
        config = await asyncio.wait_for(
            providers.RUNNERS[provider](
                rec,
                pcm,
                providers.LANGUAGE_CODES[lang][provider],
                variant,
                schedule,
                key,
                options,
            ),
            reference["duration_s"] + 60,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:400].replace(key, "<key>")
        rec.notes.append(traceback.format_exc(limit=4)[-1200:].replace(key, "<key>"))
    payload = {
        "provider": provider,
        "variant": variant,
        "lang": lang,
        "started_utc": started,
        "error": error,
        "provider_config": config,
        "chunk_ms": providers.CHUNK_MS,
        "finalize_schedule_audio_s": schedule,
        **rec.export(),
    }
    text = json.dumps(payload, ensure_ascii=False)
    if key and key in text:
        raise RuntimeError("refusing to write a result that contains an API key")
    with gzip.open(runs_dir / f"{provider}_{variant}_{lang}.json.gz", "wt", encoding="utf-8") as f:
        f.write(text)
    status = "ok" if error is None else f"error: {error}"
    print(
        f"[run] {provider:8} {variant:13} {lang}  events={len(rec.events):5}  {status}", flush=True
    )


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("websockets", "google-genai", "httpx", "numpy", "onnxruntime", "rapidfuzz"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


async def _main(args: argparse.Namespace) -> Path:
    overrides = dict(item.split("=", 1) for item in args.key)
    env_files = [REPO_ROOT / name for name in (".env", ".env.local")]
    if args.env_file:
        env_files = [Path(path) for path in args.env_file]
    keys = _resolve_keys(env_files, overrides)
    selected = [p for p in args.providers if p in keys]
    for provider in args.providers:
        if provider in keys:
            print(f"[keys] {provider}: using {keys[provider][0]}")
        else:
            names = overrides.get(provider) or ", ".join(KEY_CANDIDATES[provider])
            print(f"[keys] {provider}: skipped, none of {names} is set")
    if not selected:
        raise SystemExit("no provider keys found")

    data_dir = HERE / "data"
    fleurs.ensure(list(args.langs), data_dir)

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path(args.out) if args.out else HERE / "results" / stamp
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "reference").mkdir(exist_ok=True)
    for lang in args.langs:
        shutil.copy(data_dir / f"{lang}.json", results_dir / "reference" / f"{lang}.json")

    options = {"qwen_endpoint": args.qwen_endpoint}
    meta = {
        "started_utc": stamp,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "packages": _package_versions(),
        "langs": list(args.langs),
        "providers": selected,
        "variants": {p: [v for v in providers.VARIANTS[p] if v in args.variants] for p in selected},
        "key_variables": {p: keys[p][0] for p in selected},
        "qwen_endpoint": args.qwen_endpoint or providers.QWEN_URL,
        "chunk_ms": providers.CHUNK_MS,
        "gap_finalize_offset_s": providers.GAP_FINALIZE_OFFSET_S,
        "mid_finalize_every_s": providers.MID_FINALIZE_EVERY_S,
    }
    (results_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    for lang in args.langs:
        print(f"[lang] {lang}: streaming in real time", flush=True)
        started = time.monotonic()
        await asyncio.gather(
            *(
                _run_one(provider, variant, lang, data_dir, keys[provider][1], options, runs_dir)
                for provider in selected
                for variant in providers.VARIANTS[provider]
                if variant in args.variants
            )
        )
        print(f"[lang] {lang}: done in {time.monotonic() - started:.0f} s", flush=True)

    analyze.summarize(results_dir, args.decider_rtt)
    return results_dir


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    all_variants = sorted({v for vs in providers.VARIANTS.values() for v in vs})
    parser = argparse.ArgumentParser(description="Measure ASR partial/final timing on FLEURS.")
    parser.add_argument("--langs", nargs="+", default=list(LANGS), choices=LANGS)
    parser.add_argument(
        "--providers", nargs="+", default=list(providers.RUNNERS), choices=list(providers.RUNNERS)
    )
    parser.add_argument("--variants", nargs="+", default=all_variants, choices=all_variants)
    parser.add_argument("--env-file", action="append", help="defaults to .env and .env.local")
    parser.add_argument("--key", action="append", default=[], metavar="PROVIDER=VARIABLE")
    parser.add_argument("--qwen-endpoint", default=None)
    parser.add_argument("--decider-rtt", type=float, default=analyze.DEFAULT_DECIDER_RTT_S)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results_dir = asyncio.run(_main(args))
    relative = (
        results_dir.relative_to(REPO_ROOT) if results_dir.is_relative_to(REPO_ROOT) else results_dir
    )
    print(f"\nsummary: {results_dir / 'summary.md'}")
    print(f"to share: git add {relative.as_posix()} && git commit && git push")


if __name__ == "__main__":
    main()
