"""Authoring script that wrote spec.json. Rerun only to change the frozen case definitions."""

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.sources import PUBLIC_CORPUS  # noqa: E402

m = json.load(open(PUBLIC_CORPUS / "manifest.json", encoding="utf-8"))
cv = collections.defaultdict(list)
for x in m["traces"]:
    s = x["source"]
    if (
        s["dataset"] == "common_voice"
        and x.get("transformation") is None
        and s["relative_audio_path"].endswith(".mp3")
    ):
        cv[(s["locale"], s["speaker_id"])].append(
            (s["duration_ms"], s["relative_audio_path"], s["source_text"])
        )
for k in cv:
    cv[k] = sorted(set(cv[k]))


SPEAKERS = {
    loc: sorted((k[1] for k in cv if k[0] == loc), key=lambda i: (-len(cv[(loc, i)]), i))
    for loc in ("ko", "ja")
}


def cvclip(loc, rank, i, label, gap):
    key = (loc, SPEAKERS[loc][rank])
    d, p, t = cv[key][i]
    return {"source": "common_voice", "path": p, "text": t, "speaker": label, "gap_s": gap}


def ls(uid, label, gap=0.0, **kw):
    return {"source": "librispeech", "id": uid, "speaker": label, "gap_s": gap, **kw}


def tts(text, voice, label, gap=0.15, **kw):
    return {"source": "tts", "text": text, "voice": voice, "speaker": label, "gap_s": gap, **kw}


def alt(pairs, gaps):
    return [ls(u, s, g) for (u, s), g in zip(pairs, gaps)]


L = "1089-134686"
F = "1995-1826"
DAVID = "Microsoft David Desktop"
ZIRA = "Microsoft Zira Desktop"
cases = []


def add(cid, cat, desc, events, lang="en"):
    cases.append(
        {
            "case_id": cid,
            "category": cat,
            "language": lang,
            "description": desc,
            "lead_silence_s": 0.5,
            "tail_silence_s": 1.0,
            "events": events,
        }
    )


add(
    "ls_transfer_gap0_mf",
    "continuous_transfer",
    "Male/female alternation with 0 ms speech gaps.",
    alt(
        [
            (f"{L}-0001", "A"),
            (f"{F}-0003", "B"),
            (f"{L}-0007", "A"),
            (f"{F}-0015", "B"),
            (f"{L}-0010", "A"),
            (f"{F}-0002", "B"),
        ],
        [0] * 6,
    ),
)
add(
    "ls_transfer_gap150_ff",
    "continuous_transfer",
    "Female/female alternation with 150 ms gaps.",
    alt(
        [
            ("121-121726-0004", "A"),
            ("4446-2271-0006", "B"),
            ("121-121726-0006", "A"),
            ("4446-2271-0000", "B"),
            ("121-121726-0011", "A"),
            ("4446-2271-0005", "B"),
        ],
        [0] + [0.15] * 5,
    ),
)
add(
    "ls_transfer_gap300_mm",
    "continuous_transfer",
    "Male/male alternation with 300 ms gaps.",
    alt(
        [
            ("260-123286-0004", "A"),
            ("7021-79740-0003", "B"),
            ("260-123286-0008", "A"),
            ("7021-79740-0001", "B"),
            ("260-123286-0011", "A"),
            ("7021-79730-0002", "B"),
        ],
        [0] + [0.3] * 5,
    ),
)
add(
    "ls_transfer_mixed_gaps_mf",
    "continuous_transfer",
    "Male/female alternation with 0/150/300 ms gaps.",
    alt(
        [
            ("1188-133604-0014", "A"),
            ("5142-33396-0014", "B"),
            ("1188-133604-0010", "A"),
            ("5142-33396-0003", "B"),
            ("1188-133604-0013", "A"),
            ("5142-33396-0010", "B"),
        ],
        [0, 0, 0.15, 0.3, 0, 0.15],
    ),
)
add(
    "ls_short_turns_mf",
    "short_turns",
    "Quick exchange of 1.6-2.0 s turns with 150 ms gaps.",
    alt(
        [
            (f"{L}-0014", "A"),
            (f"{F}-0010", "B"),
            (f"{L}-0003", "A"),
            (f"{F}-0014", "B"),
            ("1188-133604-0006", "C"),
            (f"{F}-0006", "B"),
        ],
        [0] + [0.15] * 5,
    ),
)
add(
    "ls_long_rollover",
    "rollover_transfer",
    "Turns longer than six seconds handing over without a pause.",
    alt(
        [
            ("1188-133604-0005", "A"),
            ("5142-33396-0001", "B"),
            ("1188-133604-0001", "A"),
            ("5142-33396-0006", "B"),
        ],
        [0, 0, 0.15, 0],
    ),
)
add(
    "ls_paused_control",
    "paused_transfer",
    "Alternation separated by 800/1500 ms pauses (natural-end control).",
    alt(
        [
            (f"{L}-0007", "A"),
            (f"{F}-0004", "B"),
            (f"{L}-0010", "A"),
            (f"{F}-0008", "B"),
            (f"{L}-0001", "A"),
        ],
        [0, 0.8, 1.5, 0.8, 1.5],
    ),
)
add(
    "syn_short_corrections",
    "short_interjection",
    "Female speaker interrupted by short synthetic male corrections (No. / No, three. / Okay.).",
    [
        ls("121-121726-0012", "A"),
        tts("No.", DAVID, "B"),
        ls("121-121726-0002", "A", 0.15),
        tts("No, three.", DAVID, "B"),
        ls("121-121726-0008", "A", 0.15),
        tts("Okay.", DAVID, "B"),
        ls("121-121726-0001", "A", 0.15),
    ],
)
add(
    "ls_short_b_between_a",
    "short_interjection",
    "Short real female turns (about 1.6 s) between longer male turns.",
    alt(
        [
            (f"{L}-0002", "A"),
            ("5142-33396-0000", "B"),
            (f"{L}-0008", "A"),
            ("5142-33396-0013", "B"),
            (f"{L}-0004", "A"),
        ],
        [0] + [0.15] * 4,
    ),
)
add(
    "ls_overlap_transfer",
    "overlap_transfer",
    "Successor starts 300-500 ms before the predecessor ends.",
    [
        ls("260-123286-0005", "A", 0, gain=0.8),
        ls("4446-2271-0003", "B", -0.4, gain=0.8),
        ls("260-123286-0007", "A", -0.3, gain=0.8),
        ls("4446-2271-0012", "B", -0.5, gain=0.8),
        ls("260-123286-0015", "A", -0.4, gain=0.8),
    ],
)
add(
    "syn_backchannel",
    "backchannel",
    "Long male turns with short synthetic female backchannels inside them.",
    [
        ls("1188-133604-0004", "A"),
        tts("Yeah.", ZIRA, "B", within_previous_s=3.0, gain=0.8),
        tts("Mm-hmm.", ZIRA, "B", within_previous_s=6.5, gain=0.8),
        ls("1188-133604-0000", "A", 0.3),
        tts("Right.", ZIRA, "B", within_previous_s=4.0, gain=0.8),
    ],
)
add(
    "ls_same_speaker_long",
    "same_speaker",
    "One speaker for about 17 s with 200 ms internal pauses (false-split and rollover control).",
    alt(
        [
            ("2300-131720-0014", "A"),
            ("2300-131720-0006", "A"),
            ("2300-131720-0000", "A"),
            ("2300-131720-0005", "A"),
        ],
        [0, 0.2, 0.2, 0.2],
    ),
)
add(
    "ls_three_round_robin",
    "multi_speaker",
    "Three speakers in round robin with 150 ms gaps.",
    alt(
        [
            ("908-157963-0003", "A"),
            ("1221-135766-0013", "B"),
            ("672-122797-0000", "C"),
            ("908-157963-0013", "A"),
            ("1221-135766-0014", "B"),
            ("672-122797-0003", "C"),
            ("908-157963-0014", "A"),
            ("1221-135766-0002", "B"),
            ("672-122797-0010", "C"),
        ],
        [0] + [0.15] * 8,
    ),
)
add(
    "ls_return_after_gap",
    "returning_speaker",
    "Speaker A returns after about 37 s of two other speakers.",
    alt(
        [
            ("237-126133-0009", "A"),
            ("1221-135766-0004", "B"),
            ("908-157963-0010", "C"),
            ("1221-135766-0007", "B"),
            ("908-157963-0005", "C"),
            ("1221-135766-0009", "B"),
            ("237-126133-0008", "A"),
            ("237-126133-0012", "A"),
        ],
        [0] + [0.3] * 7,
    ),
)
add(
    "ls_five_speakers",
    "palette_overflow",
    "Five speakers (one more than the four colors), then two return.",
    alt(
        [
            (f"{L}-0001", "A"),
            (f"{F}-0003", "B"),
            ("260-123286-0004", "C"),
            ("4446-2271-0006", "D"),
            ("1188-133604-0014", "E"),
            (f"{L}-0007", "A"),
            ("4446-2271-0000", "D"),
        ],
        [0] + [0.2] * 6,
    ),
)
add(
    "ls_nine_speakers",
    "slot_overflow",
    "Nine speakers (one more than eight diarizer slots), then two return.",
    alt(
        [
            (f"{L}-0001", "A"),
            (f"{F}-0003", "B"),
            ("260-123286-0014", "C"),
            ("4446-2271-0002", "D"),
            ("1188-133604-0013", "E"),
            ("5142-33396-0014", "F"),
            ("908-157963-0001", "G"),
            ("1221-135766-0015", "H"),
            ("672-122797-0005", "I"),
            (f"{L}-0003", "A"),
            (f"{F}-0004", "B"),
        ],
        [0] + [0.2] * 10,
    ),
)
add(
    "cv_ko_transfer",
    "continuous_transfer",
    "Korean two-speaker alternation with 150 ms gaps.",
    [
        cvclip("ko", 4, 1, "A", 0),
        cvclip("ko", 2, 0, "B", 0.15),
        cvclip("ko", 4, 2, "A", 0.15),
        cvclip("ko", 2, 1, "B", 0.15),
        cvclip("ko", 4, 3, "A", 0.15),
        cvclip("ko", 2, 2, "B", 0.15),
    ],
    "ko",
)
add(
    "cv_ko_three",
    "multi_speaker",
    "Korean three speakers with 0/150/300 ms gaps and a returning speaker.",
    [
        cvclip("ko", 0, 0, "A", 0),
        cvclip("ko", 1, 1, "B", 0),
        cvclip("ko", 3, 0, "C", 0.15),
        cvclip("ko", 0, 1, "A", 0.3),
        cvclip("ko", 1, 2, "B", 0),
        cvclip("ko", 3, 1, "C", 0.15),
        cvclip("ko", 0, 2, "A", 0.15),
    ],
    "ko",
)
add(
    "cv_ja_pair_repeated",
    "continuous_transfer",
    "Japanese two-speaker alternation; each speaker repeats one original clip.",
    [
        cvclip("ja", 34, 0, "A", 0),
        cvclip("ja", 36, 0, "B", 0.15),
        cvclip("ja", 34, 0, "A", 0.15),
        cvclip("ja", 36, 0, "B", 0.15),
    ],
    "ja",
)
add(
    "cv_ja_multi",
    "multi_speaker",
    "Japanese three speakers in sequence, then the first returns.",
    [
        cvclip("ja", 34, 0, "A", 0),
        cvclip("ja", 36, 0, "B", 0.15),
        cvclip("ja", 35, 0, "C", 0.15),
        cvclip("ja", 34, 0, "A", 0.3),
    ],
    "ja",
)
ami = [
    ("EN2002a", 860.0),
    ("EN2002b", 475.0),
    ("EN2002c", 2410.0),
    ("EN2002d", 765.0),
    ("ES2004b", 1475.0),
    ("IS1009b", 1130.0),
]
for meeting, start in ami:
    cases.append(
        {
            "case_id": f"ami_{meeting.lower()}",
            "kind": "ami",
            "category": "natural_meeting",
            "language": "en",
            "description": f"AMI {meeting} Mix-Headset {start:.0f}-{start + 75:.0f} s; window selected for the most continuous transfers.",
            "meeting": meeting,
            "start_s": start,
            "end_s": start + 75.0,
        }
    )
spec = {
    "schema": "diarization-bench/spec/v1",
    "notes": "Frozen case definitions. Gaps are speech-to-speech seconds after energy trimming; negative gaps overlap. AMI windows come from `bench.corpus select-ami` with 75 s length and G=0.5 s.",
    "cases": cases,
}
out = Path(__file__).resolve().parent / "spec.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(len(cases))
