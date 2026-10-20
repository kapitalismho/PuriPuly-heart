# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.27",
#   "numpy>=2.0",
#   "onnxruntime==1.28.0",
#   "rapidfuzz>=3.9",
# ]
# ///
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze
import decide
import fleurs
import segmenter
import yodas

DATA = ("ko_kr", "ja_jp", "en_us", "ko_conv", "ja_conv", "en_conv")
THRESHOLDS = (0.0, 0.3, 0.5, 0.75, 0.9)
COMMAS = ",、，"
LABEL_BEFORE_S = 0.6
LABEL_AFTER_S = 0.2
COVER_BEFORE_S = 0.2
COVER_AFTER_S = 0.6


def word_labels(run: dict) -> list[tuple[float, str, str]]:
    tokens = [w for w in run["final_words"] if w["w"].strip()]
    words = []
    for i, token in enumerate(tokens):
        text = token["w"].strip()
        if not analyze.normalize(text):
            continue
        marks = text[len(text.rstrip(analyze.TERMINAL + COMMAS + analyze.CLOSERS)) :]
        j = i + 1
        while j < len(tokens) and not analyze.normalize(tokens[j]["w"]):
            marks += tokens[j]["w"].strip()
            j += 1
        label = "mid"
        if any(ch in analyze.TERMINAL for ch in marks):
            label = "sentence"
        elif any(ch in COMMAS for ch in marks):
            label = "clause"
        words.append((float(token["end_s"]), label, text))
    return words


def label_pause(words, pause: segmenter.Pause) -> tuple[str, str, float | None]:
    window = [
        w for w in words if pause.start_s - LABEL_BEFORE_S <= w[0] <= pause.start_s + LABEL_AFTER_S
    ]
    if not window:
        return "none", "", None
    end_s, label, text = window[-1]
    return label, text, end_s


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    a, b = np.array(pos), np.array(neg)
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def evaluate(name: str, pcm: bytes, words, smart) -> dict:
    analysis = segmenter.analyze_audio(pcm, smart)
    minutes = analysis.duration_s / 60
    horizon = analysis.duration_s - decide.END_EXCLUSION_S
    pauses = []
    for pause in analysis.pauses:
        if pause.score is None or pause.start_s > horizon:
            continue
        label, text, word_end = label_pause(words, pause)
        pauses.append(
            {
                "start_s": round(pause.start_s, 3),
                "duration_s": round(pause.duration_s, 3),
                "score": round(pause.score, 4),
                "label": label,
                "word": text,
                "word_end_s": word_end,
            }
        )
    sentence_ends = [w for w in words if w[1] == "sentence" and w[0] <= horizon]
    covered = [
        w
        for w in sentence_ends
        if any(
            w[0] - COVER_BEFORE_S <= p.start_s <= w[0] + COVER_AFTER_S and p.duration_s >= 0.2
            for p in analysis.pauses
        )
    ]
    thresholds = {}
    for theta in THRESHOLDS:
        cuts = [p for p in pauses if p["score"] >= theta]
        labeled = [p for p in cuts if p["label"] != "none"]
        ends = {w[0] for w in sentence_ends}
        hits = {p["word_end_s"] for p in cuts if p["label"] == "sentence"} & ends
        thresholds[str(theta)] = {
            "cuts": len(cuts),
            "cuts_per_min": round(len(cuts) / minutes, 2),
            "sentence": sum(p["label"] == "sentence" for p in labeled),
            "clause": sum(p["label"] == "clause" for p in labeled),
            "mid": sum(p["label"] == "mid" for p in labeled),
            "unlabeled": len(cuts) - len(labeled),
            "sentence_recall": round(len(hits) / len(sentence_ends), 3) if sentence_ends else None,
        }
    scores = {
        label: [p["score"] for p in pauses if p["label"] == label]
        for label in ("sentence", "clause", "mid")
    }
    preview = {}
    for policy in segmenter.POLICIES:
        cuts = segmenter.cut_schedule(analysis, policy)
        segments = segmenter.segments(analysis, cuts)
        kinds: dict[str, int] = {}
        for cut in cuts:
            if cut.kind == "speech":
                key = "in speech"
            else:
                pause = analysis.pause_at(cut.t - segmenter.FRAME_S / 2)
                key = "pause " + (label_pause(words, pause)[0] if pause else "none")
            kinds[key] = kinds.get(key, 0) + 1
        lengths = [s.end_s - s.start_s for s in segments]
        preview[policy] = {
            "segments": len(segments),
            "segments_per_min": round(len(segments) / minutes, 2),
            "mean_s": round(float(np.mean(lengths)), 2) if lengths else None,
            "max_s": round(float(np.max(lengths)), 2) if lengths else None,
            "sent_s": round(float(np.sum(lengths)), 1),
            "cuts": kinds,
        }
    return {
        "data": name,
        "minutes": round(minutes, 3),
        "voiced_share": round(float(analysis.voiced.mean()), 3),
        "pauses_total": len(analysis.pauses),
        "pauses_200ms": len(pauses),
        "sentence_ends": len(sentence_ends),
        "sentence_ends_with_200ms_pause": len(covered),
        "thresholds": thresholds,
        "auc_sentence_vs_mid": _auc(scores["sentence"], scores["mid"]),
        "mean_score": {k: round(float(np.mean(v)), 3) if v else None for k, v in scores.items()},
        "preview": preview,
        "pauses": pauses,
    }


def _pool(rows: list[dict]) -> dict:
    minutes = sum(r["minutes"] for r in rows)
    pooled = {
        "minutes": round(minutes, 2),
        "pauses_200ms": sum(r["pauses_200ms"] for r in rows),
        "sentence_ends": sum(r["sentence_ends"] for r in rows),
        "sentence_ends_with_200ms_pause": sum(r["sentence_ends_with_200ms_pause"] for r in rows),
        "thresholds": {},
    }
    for theta in THRESHOLDS:
        key = str(theta)
        entries = [r["thresholds"][key] for r in rows]
        total = {
            k: sum(e[k] for e in entries)
            for k in ("cuts", "sentence", "clause", "mid", "unlabeled")
        }
        recalls = [
            (e["sentence_recall"], r["sentence_ends"])
            for e, r in zip(entries, rows)
            if e["sentence_recall"] is not None
        ]
        weight = sum(n for _, n in recalls)
        total["cuts_per_min"] = round(total["cuts"] / minutes, 2)
        total["sentence_recall"] = (
            round(sum(v * n for v, n in recalls) / weight, 3) if weight else None
        )
        pooled["thresholds"][key] = total
    scores = {
        label: [p["score"] for r in rows for p in r["pauses"] if p["label"] == label]
        for label in ("sentence", "clause", "mid")
    }
    pooled["auc_sentence_vs_mid"] = _auc(scores["sentence"], scores["mid"])
    pooled["mean_score"] = {
        k: round(float(np.mean(v)), 3) if v else None for k, v in scores.items()
    }
    return pooled


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def render(summary: dict) -> str:
    lines = [
        "# Smart Turn at 200 ms pauses",
        "",
        "Silero VAD finds pauses in the probe audio. Smart Turn v3.2 (the bundled application "
        "model) scores each pause of at least 200 ms with up to 8 s of audio ending 200 ms into "
        "the pause; the context starts at the speech onset after the last pause of at least "
        f"{segmenter.ONSET_RESET_S:.1f} s. Each pause is labeled with the Soniox final token "
        f"that ended within {segmenter.PROBE_S:.1f} s after or {LABEL_BEFORE_S:.1f} s before the "
        "pause start: `sentence` when sentence-final punctuation follows it, `clause` for a "
        "comma, `mid` otherwise, `none` when no token ended there.",
        "",
        "- `cut at p >= x`: the pauses a cut rule would use. `x = 0` means every 200 ms pause.",
        "- `sentence recall`: share of Soniox sentence ends with such a cut.",
        "- `200 ms pause at sentence ends`: share of Soniox sentence ends followed by any pause "
        "of at least 200 ms (the ceiling for any 200 ms pause rule).",
        "",
    ]
    header = "| data | p >= | cuts/min | sentence | clause | mid | none | sentence recall |"
    sep = "|---|---|---|---|---|---|---|---|"
    for group, pooled in summary["pooled"].items():
        lines += [
            f"## {group}",
            "",
            f"- 200 ms pauses: {pooled['pauses_200ms']} in {pooled['minutes']} min.",
            f"- 200 ms pause at sentence ends: "
            f"{_pct(pooled['sentence_ends_with_200ms_pause'], pooled['sentence_ends'])} of "
            f"{pooled['sentence_ends']}.",
            (
                f"- AUC sentence vs mid: {pooled['auc_sentence_vs_mid']:.3f}"
                if pooled["auc_sentence_vs_mid"] is not None
                else "- AUC sentence vs mid: -"
            ),
            f"- Mean score: {pooled['mean_score']}",
            "",
            header,
            sep,
        ]
        rows = [r for r in summary["data"] if r["group"] == group]
        for theta in THRESHOLDS:
            t = pooled["thresholds"][str(theta)]
            labeled = t["sentence"] + t["clause"] + t["mid"]
            lines.append(
                f"| all | {theta} | {t['cuts_per_min']} | {_pct(t['sentence'], labeled)} | "
                f"{_pct(t['clause'], labeled)} | {_pct(t['mid'], labeled)} | {t['unlabeled']} | "
                f"{t['sentence_recall']} |"
            )
        for r in rows:
            for theta in (0.0, 0.5, 0.75):
                t = r["thresholds"][str(theta)]
                labeled = t["sentence"] + t["clause"] + t["mid"]
                lines.append(
                    f"| {r['data']} | {theta} | {t['cuts_per_min']} | "
                    f"{_pct(t['sentence'], labeled)} | {_pct(t['clause'], labeled)} | "
                    f"{_pct(t['mid'], labeled)} | {t['unlabeled']} | {t['sentence_recall']} |"
                )
        lines.append("")
    lines += [
        "## Cut policy preview",
        "",
        "The same cut schedules the live Qwen Audio and Gemini variants use "
        "(`segmenter.py`). `app` approximates the current Peer rules (Smart Turn complete at "
        "0.75 then 512 ms, otherwise 800 ms; 4 s/192 ms; 5 s/128 ms; 6 s hard cut with 0.3 s "
        "replay). `pause` cuts at a 200 ms pause after 2 s. `smart` cuts at a 200 ms pause when "
        f"Smart Turn is at least {segmenter.SMART_THRESHOLD}. `fixed3` cuts every 3 s. All but "
        "`app` end a turn at an 800 ms pause, cut at an 80 ms pause after 6 s and cut in speech "
        "after 10 s.",
        "",
        "| data | policy | segments/min | mean s | max s | cuts |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["data"]:
        for policy, p in r["preview"].items():
            cuts = ", ".join(f"{k} {v}" for k, v in sorted(p["cuts"].items()))
            lines.append(
                f"| {r['data']} | {policy} | {p['segments_per_min']} | {p['mean_s']} | "
                f"{p['max_s']} | {cuts} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Smart Turn at 200 ms pauses.")
    parser.add_argument("--data", nargs="*", default=list(DATA))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data_dir = HERE / "data"
    fleurs.ensure([d for d in args.data if d not in yodas.SOURCES], data_dir)
    yodas.ensure([d for d in args.data if d in yodas.SOURCES], data_dir)
    smart = segmenter.SmartTurn()
    rows, sources = [], {}
    for name in args.data:
        results_dir = decide._latest_results(name)
        if results_dir is None:
            print(f"[skip] {name}: no Soniox natural run for labels")
            continue
        run = analyze.load_run(results_dir / "runs" / f"soniox_natural_{name}.json.gz")
        _, pcm = fleurs.load(name, data_dir)
        row = evaluate(name, pcm, word_labels(run), smart)
        row["group"] = "conversation" if name in yodas.SOURCES else "read speech"
        rows.append(row)
        sources[name] = results_dir.name
        print(f"[smartturn] {name:8} pauses>=200ms={row['pauses_200ms']:3}", flush=True)
    groups = {}
    for group in ("read speech", "conversation"):
        selected = [r for r in rows if r["group"] == group]
        if selected:
            groups[group] = _pool(selected)
    summary = {"pooled": groups, "data": rows}
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or HERE / "results" / f"smartturn-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "started_utc": stamp,
        "label_sources": sources,
        "probe_s": segmenter.PROBE_S,
        "onset_reset_s": segmenter.ONSET_RESET_S,
        "window_s": segmenter.WINDOW_S,
        "smart_turn_model": segmenter.SMART_TURN_MODEL.name,
        "vad_threshold": fleurs.VAD_THRESHOLD,
        "thresholds": THRESHOLDS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    with gzip.open(out / "pauses.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            for pause in row["pauses"]:
                handle.write(json.dumps({"data": row["data"], **pause}, ensure_ascii=False) + "\n")
    compact = {
        "pooled": groups,
        "data": [{k: v for k, v in r.items() if k != "pauses"} for r in rows],
    }
    (out / "summary.json").write_text(json.dumps(compact, ensure_ascii=False, indent=1), "utf-8")
    (out / "summary.md").write_text(render(summary), encoding="utf-8")
    print(f"[smartturn] wrote {out}")


if __name__ == "__main__":
    main()
