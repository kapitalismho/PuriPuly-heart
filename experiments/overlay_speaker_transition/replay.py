from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
SCENARIO_PATH = ROOT / "scenario.json"
EXPECTED_PATH = ROOT / "expected_results.json"

MODES = ("baseline", "A1", "A2", "C1", "C2", "E1", "E2")
CYAN_HEX = {"C1": "#33D6FF", "C2": "#2DE1A8", "E1": "#33D6FF", "E2": "#2DE1A8"}
MARKER_STYLE = {"A1": "dash", "A2": "rule", "E1": "rule", "E2": "dash"}
MARKER_HEX = {"A1": "#FFD700", "A2": "#FFD700", "E1": "#FFD700", "E2": "#2DE1A8"}
WINDOW_BLOCKS = 2


def hex_for(mode: str, role: str) -> str:
    table = {"white": "#FFFFFF", "gold": "#FFD700", "cyan": CYAN_HEX.get(mode, "#33D6FF")}
    return table[role]


def fresh_state() -> dict:
    return {"run": "gold", "emphasis": None, "window": []}

def render_entry(mode: str, event: dict, role: str, marker: str) -> dict:
    block = event["block"]
    return {
        "event": event["id"],
        "block": block["id"],
        "key": block["logical_turn_key"],
        "channel": block["channel"],
        "color_role": role,
        "hex": hex_for(mode, role),
        "marker": marker,
        "marker_hex": MARKER_HEX.get(mode) if marker != "none" else None,
        "primary": block["primary_text"],
        "secondary": block["secondary_text"] if block["secondary_enabled"] else "",
    }


def step(mode: str, state: dict, event: dict) -> dict:
    kind = event["evidence"]["kind"]
    key = event["block"]["logical_turn_key"]
    channel = event["block"]["channel"]
    window = state["window"]
    keys = [entry["key"] for entry in window]
    if kind == "late_expired" or (kind == "revision" and key not in keys):
        return {
            "action": "withheld",
            "reason": "late arrival for content outside the visible window",
            "window": [dict(entry) for entry in window],
            "run": state["run"],
            "emphasis": state["emphasis"],
        }
    if kind == "revision":
        for entry in window:
            if entry["key"] == key:
                entry["primary"] = event["block"]["primary_text"]
                entry["secondary"] = event["block"]["secondary_text"] if event["block"]["secondary_enabled"] else ""
        return {
            "action": "revised-in-place",
            "reason": "same logical turn keeps identity, color and marker",
            "window": [dict(entry) for entry in window],
            "run": state["run"],
            "emphasis": state["emphasis"],
        }
    if kind == "reset_fresh":
        state["window"] = []
        state["run"] = "gold"
        state["emphasis"] = None
        window = state["window"]
    elif mode in ("E1", "E2") and state["emphasis"] is not None and key != state["emphasis"]:
        for entry in window:
            if entry["key"] == state["emphasis"] and entry["color_role"] == "cyan":
                entry["color_role"] = "gold"
                entry["hex"] = hex_for(mode, "gold")
        state["emphasis"] = None
    if channel == "self":
        role, marker, action = "white", "none", "self-white"
    elif kind == "confirmed_transition":
        if mode in ("C1", "C2"):
            state["run"] = "cyan" if state["run"] == "gold" else "gold"
            role, marker, action = state["run"], "none", "toggle-run"
        elif mode in ("E1", "E2"):
            state["emphasis"] = key
            role, marker, action = "cyan", MARKER_STYLE[mode], "emphasis-with-marker"
        else:
            role, marker, action = "gold", MARKER_STYLE.get(mode, "none"), "marker-only"
    else:
        if mode in ("C1", "C2") and channel == "peer":
            role = state["run"]
        elif channel == "peer":
            role = "gold"
        else:
            role = "white"
        marker, action = "none", "carry"
    window.append(render_entry(mode, event, role, marker))
    del window[:-WINDOW_BLOCKS]
    return {
        "action": action,
        "reason": event["evidence"]["note"],
        "window": [dict(entry) for entry in window],
        "run": state["run"],
        "emphasis": state["emphasis"],
    }


def replay(events: list) -> dict:
    out = {}
    for mode in MODES:
        state = fresh_state()
        out[mode] = [dict(step(mode, state, event), event=event["id"]) for event in events]
    return out


def window_of(results: dict, mode: str, event_id: str) -> list:
    for snap in results[mode]:
        if snap["event"] == event_id:
            return snap["window"]
    raise KeyError((mode, event_id))


def snap_of(results: dict, mode: str, event_id: str) -> dict:
    for snap in results[mode]:
        if snap["event"] == event_id:
            return snap
    raise KeyError((mode, event_id))


def check_revision_keeps_emphasis() -> bool:
    base = {
        "id": "syn-rev", "speaker_ref": "A",
        "evidence": {"kind": "revision", "reference": "x", "note": "synthetic"},
        "block": {
            "id": "bXr", "occupant_key": "slot-1", "appearance_seq": 99,
            "block_variant": "finalized", "channel": "peer",
            "logical_turn_key": "peer-k1", "publication_scope": "s",
            "publication_generation": 2, "publication_order": 2,
            "primary_language": "en", "primary_text": "reword",
            "secondary_enabled": True, "secondary_language": "en",
            "secondary_text": "reword",
        },
    }
    state = fresh_state()
    state["emphasis"] = "peer-k1"
    state["window"] = [{
        "event": "syn-arrive", "block": "bX", "key": "peer-k1",
        "channel": "peer", "color_role": "cyan", "hex": "#33D6FF",
        "marker": "rule", "primary": "orig", "secondary": "orig",
    }]
    out = step("E1", state, base)
    return (
        out["action"] == "revised-in-place"
        and state["emphasis"] == "peer-k1"
        and state["window"][0]["color_role"] == "cyan"
    )


def run_invariants(results: dict) -> list:
    checks = []

    def add(check_id: str, desc: str, ok: bool) -> None:
        checks.append({"id": check_id, "desc": desc, "result": "PASS" if ok else "FAIL"})

    w = window_of(results, "C1", "t02")
    add("segmentation-no-toggle", "t02 delivery cut keeps Gold run, no marker in any mode",
        all(entry["color_role"] == "gold" and entry["marker"] == "none" for entry in w)
        and all(entry["marker"] == "none" for mode in ("A1", "A2", "E1", "E2") for entry in window_of(results, mode, "t02")))
    w = window_of(results, "C1", "t03")
    add("first-transition-once", "t03 toggles C run to Cyan exactly once, A/E attach one marker",
        [entry["color_role"] for entry in w] == ["gold", "cyan"]
        and window_of(results, "A1", "t03")[-1]["marker"] == "dash"
        and window_of(results, "A2", "t03")[-1]["marker"] == "rule"
        and window_of(results, "E1", "t03")[-1] == {**window_of(results, "E1", "t03")[-1], "color_role": "cyan", "marker": "rule"})
    w = window_of(results, "C1", "t04")
    add("continuation-no-duplicate", "t04 keeps Cyan run, no second toggle or marker",
        [entry["color_role"] for entry in w] == ["cyan", "cyan"]
        and snap_of(results, "C1", "t04")["action"] == "carry")
    s = snap_of(results, "C1", "t05")
    e = snap_of(results, "E1", "t05")
    add("self-interposition", "t05 Self is White, C run stays Cyan, E emphasis expires on the readable Self turn",
        s["window"][-1]["color_role"] == "white" and s["run"] == "cyan"
        and e["emphasis"] is None and e["window"][-1]["color_role"] == "white")
    add("self-across-no-boundary", "t06 same speaker across Self adds no marker and keeps the Cyan run",
        window_of(results, "A2", "t06")[-1]["marker"] == "none"
        and window_of(results, "C1", "t06")[-1]["color_role"] == "cyan")
    add("consecutive-changes", "t07 and t08 each toggle once and each keeps its own marker",
        [entry["color_role"] for entry in window_of(results, "C1", "t07")] == ["cyan", "gold"]
        and [entry["color_role"] for entry in window_of(results, "C1", "t08")] == ["gold", "cyan"]
        and window_of(results, "E1", "t08")[0]["marker"] == "rule"
        and window_of(results, "E1", "t08")[0]["color_role"] == "gold"
        and window_of(results, "E1", "t08")[-1]["color_role"] == "cyan")
    add("no-person-lookup", "t08 color comes only from toggle parity, never from speaker A history",
        snap_of(results, "C1", "t08")["run"] == "cyan"
        and window_of(results, "C2", "t08")[-1]["hex"] == "#2DE1A8")
    add("unknown-withheld", "t09 mixed evidence adds no marker, no toggle, C retains its hue",
        all(window_of(results, mode, "t09")[-1]["marker"] == "none" for mode in ("A1", "A2", "E1", "E2"))
        and snap_of(results, "C1", "t09")["run"] == "cyan"
        and window_of(results, "C1", "t09")[-1]["color_role"] == "cyan")
    s = snap_of(results, "C1", "t10")
    add("reset-fresh", "t10 clears the window and restarts Gold without implying a transition",
        [entry["block"] for entry in s["window"]] == ["b10"]
        and s["window"][0]["color_role"] == "gold"
        and s["window"][0]["marker"] == "none" and s["run"] == "gold")
    add("late-withheld", "t11 revision of expired content and t12 late evidence change nothing",
        snap_of(results, "E1", "t11")["action"] == "withheld"
        and snap_of(results, "E1", "t12")["action"] == "withheld"
        and snap_of(results, "E1", "t12")["window"] == snap_of(results, "E1", "t10")["window"])
    w = window_of(results, "E1", "t13")
    add("multilingual-whole-turn", "t13 wrapping turn takes one arrival color across primary and secondary",
        w[-1]["color_role"] == "cyan" and w[-1]["marker"] == "rule"
        and window_of(results, "C2", "t13")[-1]["hex"] == "#2DE1A8")
    w = window_of(results, "E1", "t14")
    add("short-reply-expiry-keeps-marker", "t14 expires the t13 emphasis to Gold while its rule marker persists",
        [entry["color_role"] for entry in w] == ["gold", "gold"]
        and w[0]["marker"] == "rule" and w[-1]["marker"] == "none"
        and snap_of(results, "C1", "t14")["run"] == snap_of(results, "C1", "t13")["run"])
    add("revision-keeps-emphasis-synthetic", "a same-key revision never expires E emphasis",
        check_revision_keeps_emphasis())
    return checks


def main() -> None:
    raw = SCENARIO_PATH.read_bytes()
    scenario = json.loads(raw)
    events = scenario["events"]
    results = replay(events)
    checks = run_invariants(results)
    payload = {
        "meta": {
            "issue": "https://github.com/kapitalismho/PuriPuly-heart/issues/178",
            "baseline": scenario["baseline"],
            "scenario_sha256": hashlib.sha256(raw).hexdigest(),
            "scenario_version": scenario["meta"]["version"],
            "window_blocks": WINDOW_BLOCKS,
        },
        "per_mode": results,
        "invariants": checks,
    }
    EXPECTED_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [item for item in checks if item["result"] != "PASS"]
    for item in checks:
        print(f'{item["result"]} {item["id"]}: {item["desc"]}')
    print(f"events={len(events)} modes={len(MODES)} failed={len(failed)}")
    print(f"wrote {EXPECTED_PATH.name} scenario_sha256={payload['meta']['scenario_sha256'][:12]}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
