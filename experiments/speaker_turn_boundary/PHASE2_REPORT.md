# Phase 2 report — Benchmark data (LibriSpeech synthetic D1, AMI/AliMeeting D2, mixed dev pool D3, PuriPuly-like D4)

GitHub issue #51, Phase 2 only. Executed on top of the committed Phase 1
commit `2e909383` in the `experiment-v2-speaker-change-turn-boundaries-ls`
worktree. Phase 3 (thresholds/selection), detector sweeps, provider policy
work, production wiring, and commits were **not** begun.

**Status: Phase 2 is scientifically complete for D1/D2/D3 as an executable,
deterministic, validated benchmark-data pipeline, with real audio downloaded
and verified for all three corpora. D4 has no authorized audio and is
recorded provisional with an import schema. Everything regenerates
byte-for-byte; nothing was fabricated.**

## 1. Environment and baseline

- Experiment baseline SHA (pinned in `config.py`): `adf8cde2b5b166beb95c50a39e8941d2fee3601e`; worktree HEAD `2e909383` at run time.
- Python 3.12.10 (uv), numpy 2.5.1, scipy 1.18.0, Windows x86-64.
- External tools: ffmpeg 8.0.1 (FLAC decode, libopus encode/decode). All
  audio processing is deterministic numpy/wave code except FLAC decode and
  Opus encode/decode, which are pinned ffmpeg/libopus invocations.
- External corpus root (default): `%TEMP%\opencode\stb_phase2_corpora`
  (override: `STB_PHASE2_CORPORA_ROOT`). Default Phase 2 build output dir for
  all three build scripts: `%TEMP%\opencode\stb_phase2_corpora\phase2_build`
  (resolved on the run machine to
  `C:\Users\salee\AppData\Local\Temp\opencode\stb_phase2_corpora\phase2_build`;
  per-run override: `--out`). By default no generated wavs are written into
  the Git worktree. Total on-disk after this run:
  **11.2 GB / 14,402 files** (archives + extracted corpora + derived Phase 2
  build output). Nothing under this root is deleted or overwritten except
  our own uniquely named extraction temp dirs.

## 2. Exact commands

```powershell
# D1: acquire (resumable, md5-verified) + index + build + validate
# (default output: <corpus root>\phase2_build; --out overrides)
uv run python -m experiments.speaker_turn_boundary.build_phase2_cases --validate

# D2: AMI annotations + pilot meetings + AliMeeting Eval; builds + validates
uv run python -m experiments.speaker_turn_boundary.build_phase2_real

# D3: mixed dev pool + disjointness enforcement
uv run python -m experiments.speaker_turn_boundary.build_phase2_mixed

# validate the committed repo manifests against the external roots and
# regenerate data/results/phase2_d{1,2,3}_validation.json
uv run python -m experiments.speaker_turn_boundary.validate_phase2

# tests / lint / format (from repo root)
uv run pytest experiments/speaker_turn_boundary/tests -q
uv run ruff check experiments/speaker_turn_boundary
uv run --extra dev black experiments/speaker_turn_boundary
```

Builds write manifests and generated wavs to the external build dir by
default; the committed `data/manifests/*.json` were produced by the same
code and are byte-reproducible from any output dir for a fixed corpus root
(they store relative wav paths; the only exception is `mixed_dev_pool`,
which embeds its absolute build root by design, section 8). The committed
validation reports are regenerated with `validate_phase2` above, which
validates the repo manifests against the repo data dir plus the external
roots.

Re-running any command reproduces byte-identical manifests (verified by a
second full build into a fresh directory: all hashes equal, section 8).

## 3. Public corpora acquired, sizes, hashes

| Archive | Size (bytes) | Verification |
| --- | --- | --- |
| LibriSpeech `dev-clean.tar.gz` | 337,926,286 | MD5 `42e2234ba48799c1f50f24a7926300a1` (openslr `md5sum.txt`) |
| LibriSpeech `test-clean.tar.gz` | 346,663,984 | MD5 `32fa31d27d2e1cad72775fee3f4849a9` |
| LibriSpeech `test-other.tar.gz` | 328,757,843 | MD5 `fb5a50374b501bb3bac4815ee91d3135` |
| AMI manual annotations v1.6.2 zip | 22 MB | downloaded from groups.inf.ed.ac.uk (public CC BY 4.0) |
| AMI `Mix-Headset` wavs (4 meetings) | 126.9 MB total | SHA-256 per file in manifests; e.g. `ES2003a.Mix-Headset.wav` `41cf861a...` |
| AliMeeting `Eval_Ali.tar.gz` | 3,673,718,355 | Content-Length-matched download, extract verified; eval has 8 meetings |

Licenses: LibriSpeech CC BY 4.0 (openslr), AMI CC BY 4.0 (all signals and
transcription publicly released), AliMeeting CC BY-SA 4.0 (openslr 119).
No public corpus audio or archives are committed to Git; only scripts,
manifests, hashes, and 13 tiny (38-160 KB) representative D1 sample fixtures
are commit candidates (CC BY 4.0-derivative, regenerable).

## 4. D1 — Controlled synthetic (LibriSpeech) — complete

### Pipeline (`corpus/librispeech.py`)

- Acquisition: resumable download + official MD5 verification, extraction to
  the external root.
- Index: per-utterance speaker/chapter/session/transcript/duration; session =
  `{speaker}-{chapter}`.
- Fixed synthesis-only trim (never production VAD): 40 ms sliding-window RMS,
  speech window iff `rms >= max(0.01 * peak_rms, 1e-3)`; trim = first..last
  such window (documented `trim_method` in manifest `build`).
- Cut rule: A = trailing `target` window of the trimmed region (A ends at the
  speech end); B = leading `target` window (B starts at the speech start).
- Zero-gap guard (build-time, deterministic, documented): for every
  `gap=0` case both 40 ms junction windows must have RMS >= 2.5e-3 (margin
  over the 1e-3 final threshold); sources are re-picked deterministically
  (bounded) until satisfied, and after transforms the final-wav junction
  windows must still be >= 1e-3 or the build fails loudly.
- Transforms (documented, deterministic, recorded per case):
  `opus` (ffmpeg libopus 16 kHz mono 32 kbps), `gain` (whole-case factor
  0.5-1.5, seeded), `noise` (white, 15 dB SNR, seeded), `bandlimit` (one-pole
  IIR lowpass 6 kHz).
- Splice math: `b_onset = a_end + gap_samples` (gap) /
  `b_onset = a_end - overlap_samples` (overlap); overlap region mixed at
  equal gain (A+B)/2; regions tile the wav exactly; `active_speech_samples`
  per case from region union.

### Built counts (per split; identical grid on all three)

| Split | Manifest | Cases | Audio (min) | Active speech (min) |
| --- | --- | --- | --- | --- |
| dev-clean | `ls_dev` | 202 | 8.5 | 6.8 |
| test-clean | `ls_held_out_clean` | 202 | 8.5 | 6.7 |
| test-other | `ls_held_out_other` | 202 | 8.5 | 6.7 |

Per-split composition (identical across splits):

- Different-speaker gaps 800/300/100/0 ms: 101 cases
  (bucket counts 24/24/26/27 including 5 gap stress cases);
  different-speaker gap=0 cases: 27 per split; counting the same-speaker
  and gain-variation negatives, each split has **36 total gap=0 cases,
  all with stored acoustic zero-gap evidence** (breakdown below).
- Different-speaker overlaps 100/300/500 ms: 63 cases. Rule: overlap must be
  strictly smaller than the active duration, so the 0.50 s bucket carries
  100/300 ms and the 0.30-0.50 s stress bucket only 100 ms
  (documented `_valid_overlaps`).
- Active-duration targets: 2.0 (42), 1.5 (28), 1.0 (38), 0.75 (28), 0.5 (38),
  0.30-0.50 seeded stress bucket (20, e.g. 0.35/0.32/0.33 s per split).
- Negatives: same-speaker A1->A2 at all four gaps (24), same-speaker gain
  variation (6, B gain 0.5-1.5 at gaps 300/0), silence (4: 0.5/2/5/10 s),
  noise-only (4).
- Codec/noise stress (8): opus, gain, noise, opus+noise, bandlimit applied
  to gap-0/gap-100/overlap-300 cases at 2.0/0.5 s.
- Speakers/sessions: 40/88 (dev), 40/80 (test-clean), 33/84 (test-other),
  disjoint across splits by construction and machine-checked.

### Zero-gap acoustic validation (36 gap=0 cases per split: 27 different-speaker + 6 same-speaker + 3 gain-variation; 3 of the different-speaker cases carry opus / opus+noise / bandlimit stress transforms)

- Equation check: `b_onset_sample == a_end_sample` and
  `b_onset == a_end + gap` or `b_onset == a_end - overlap` — 0 violations.
- Final-wav junction check: every `gap_samples == 0` case must have
  pre-junction and post-junction 40 ms RMS >= 1e-3 on the final waveform.
  Minima observed: dev (0.00100 / 0.00101), test-clean (0.00100 / 0.00104),
  test-other (0.00100 / 0.00100) — all above threshold, all with stored
  `zero_gap_evidence` matching validation recomputation (tolerance 1e-6).
- Source-side: every gap-0 case stores original utterance SHA-256, trimmed
  and cut sample bounds; validation enforces `trim_start <= cut_start <
  cut_end <= trim_end <= original_end`.

## 5. D2 — Real conversational sets — complete as pilots

### AMI (`corpus/ami.py`)

- Acquisition: annotations zip (public) + `Mix-Headset` wavs via the
  official estimate-CGI mirror URL pattern (public, no credentials).
- One consistent single-channel recipe for every candidate:
  **`Mix-Headset`, 16 kHz mono** (verified per file by the loader).
- Annotation conversion: per-participant `words.xml` files
  (`words/{meeting}.{letter}.words.xml`); `<w>` elements carry no `who`
  attribute, so the speaker is **derived from the filename** as
  `{meeting}.Participant{letter}` (recorded as the documented
  `speaker_rule`). Word spans become sample-exact active-speaker regions;
  gaps between words become empty regions (so `{A}->{}->{B}` yields
  `gap_speaker_change` at B onset); overlapping words produce
  `{A,B}` regions (`interruption_onset`); words containing `%` tag their
  covering region `ambiguous`.
- Pilot (scenario-only partition, verified from `meetings.xml`):
  - dev: `ES2003a`, `IS1008a` (visibility=seen, seen_type=development;
    k10 5/9, k5 3/5) — 34.7 min, 4 participants each, 228/338 regions,
    37/64 overlap regions, 24 min active speech.
  - held-out: `ES2004a`, `IS1009a` (visibility=unseen) — 31.5 min,
    519/390 regions, 187/130 overlap regions, 23 min active speech.

### AliMeeting (`corpus/alimeeting.py`)

- Acquisition: `Eval_Ali.tar.gz` (3.42 GB) downloaded and extracted; 8 eval
  meetings.
- Recipe: **far-field array channel 0**, materialized once as canonical
  16 kHz mono PCM16 per session (`alimeeting/far_ch0/{Rxxxx_Mxxxx}.wav`,
  external root) and hash-pinned in the manifest.
- Annotation conversion: `TextGrid` IntervalTier per participant
  (`N_SPKxxxx`); session key = `Rxxxx_Mxxxx` (full R+M key) shared by wav and
  TextGrid names; non-empty-text intervals are speech spans; **overlaps are
  inferred from simultaneous tier intervals** producing multi-speaker
  regions (no clean-separation claim, per issue rule 4).
- All 8 eval sessions: 4.2 h audio, 4.0 h active speech, 4-2 speakers each
  (R8009_* are 2-speaker), 160-1,646 overlap regions per session; word-level
  Chinese annotations preserved (non-English truth), terse per-session
  interval stats in `condition`.

## 6. D3 — Mixed development pool — complete

- `mixed_dev_pool.json`: 204 cases = `ls_dev` (202) + `ami_dev_pilot` (2),
  43 min audio / 30 min active speech. No thresholds frozen.
- Machine-checked disjointness against every held-out manifest
  (`ls_held_out_clean`, `ls_held_out_other`, `ami_held_out_pilot`,
  `alimeeting_eval_pilot`): 0 speaker overlaps, 0 session overlaps,
  0 AMI global-actor overlaps
  (validation report `data/results/phase2_d3_validation.json`).
- AMI speaker disjointness is enforced at the person level, not merely the
  string level: AMI source speakers are meeting-qualified
  (`ES2003a.ParticipantA` vs `ES2004a.ParticipantA`), so string comparison
  alone is insufficient. Validation therefore also compares the official
  `partition_meta.agents` `global_name` actor IDs from `meetings.xml`
  (stored in every AMI case) between development and held-out manifests.
  The chosen pilot has **8 dev actor IDs** (MEE009, MEE010, MEE011, MEE012,
  FIE038, FIE073, MIE085, MIO086) and **8 held-out actor IDs** (FEE013,
  FEE016, MEE014, MEO015, FIE088, FIO084, FIO087, FIO089) with **zero
  overlap**; the full ID lists and the check result are recorded in
  `data/results/phase2_d2_validation.json` and
  `data/results/phase2_d3_validation.json`. Session disjointness (the
  official partition unit) is strict as well.
- AliMeeting eval is a held-out corpus (no AliMeeting dev portion in the
  pool; the full Train set is 73 GB and out of scope for this run).

## 7. D4 — PuriPuly-like acceptance set — provisional (no authorized audio)

- Read-only scan of explicitly authorized roots found **0 wavs**
  (`puripuly_like_provisional.json`, availability record). No private paths
  were searched; nothing private was read or committed.
- Provided for later authorized audio:
  - import schema `experiments.speaker_turn_boundary.puripuly_import.v1`
    (`corpus/puripuly_like.py`) with template
    `data/puripuly_import_template.json` (case_id, absolute wav path, sample
    rate, language, condition, sample-exact regions, hashes);
  - `check_authorized_inputs` verifies the 20-30-minute bar and 16 kHz mono
    canonical format on import.
- Consequence: any final detector/domain conclusion later in the experiment
  must remain **provisional for domain generalization** until this set is
  filled, per issue #51 D4.

## 8. Determinism and independent validation evidence

- Two fully independent builds (fresh output dirs) produced **identical
  semantic manifest hashes** (SHA-256 over the canonical JSON of the parsed
  manifest, `Phase2Manifest.hash`) for all LibriSpeech/AMI/AliMeeting
  manifests:
  `ls_dev e468d607...`, `ls_held_out_clean 51e327dc...`,
  `ls_held_out_other 7f73c032...`, `ami_dev_pilot d2e93ef4...`,
  `ami_held_out_pilot ac39518d...`, `alimeeting_eval_pilot 23fa1519...`.
  (`mixed_dev_pool` embeds its absolute wav roots by design, so its hash
  varies with the root; content identical.) All hashes quoted in this report
  are semantic manifest hashes, not raw file hashes.
- All 7 committed manifests with cases validate from the repo data dir
  against the external roots (`uv run python -m
  experiments.speaker_turn_boundary.validate_phase2`; wav roots = repo data
  dir, `<corpus root>/phase2_build`, corpus root): schema, 16 kHz mono
  PCM16, wav SHA-256, durations, region tiling, GT transition kinds
  (clean_handoff/gap_speaker_change/interruption_onset at the exact splice
  samples, none for negatives), splice equations, source-cut bounds,
  zero-gap acoustic evidence, duplicate case ids, and speaker/session/
  AMI-global-actor disjointness — 0 problems
  (`data/results/phase2_d{1,2,3}_validation.json`). The provisional D4
  record has 0 cases and carries no audio to validate.
- Durable manifest identity evidence: each validation report entry records
  the semantic manifest hash, **`manifest_canonical_file_sha256` — the
  SHA-256 of the canonical LF manifest file bytes** (the file bytes after
  CRLF→LF normalization, exactly the staged Git blob under normal Git text
  normalization; e.g. `ls_dev.json` `e468d607...`), and a canonical-bytes
  check (file bytes equal the canonical re-serialization, modulo newline
  translation). It is deliberately not a hash of raw platform-specific
  worktree bytes, which differ between CRLF and LF checkouts. The previous
  parse-and-rehash "self-hash" check was a tautology and was removed. The
  full generated wav set (606 cases, 46.5 MB) lives in the external
  `phase2_build/` root and is regenerable.
- Git hygiene: only scripts, manifests, hashes, validation reports, the
  import template, and 13 tiny sample fixtures are commit candidates; no
  public corpus audio, private recordings, or large blobs enter Git.

## 9. Tests

235 experiment tests pass (`235 passed`), including new Phase 2 coverage:
`test_phase2_schemas.py` (schema round-trip, self-describing fields, change
kind mapping, validation), `test_phase2_librispeech.py` (trim, splice region
tiling for gap/zero-gap/overlap, overlap-vs-duration rejection, zero-gap
evidence, stress duration bucket, valid overlaps, Opus round trip,
deterministic tiny-manifest build with zero-gap evidence and splice
equations), `test_phase2_ami.py` (no-`who` words parsing, filename speaker
derivation, gap/overlap/ambiguous region conversion, end-to-end meeting
load), `test_phase2_alimeeting.py` (TextGrid parser incl. tier-header
phantom-interval trap, channel-0 wav loading, layout indexing, overlap
inference), `test_phase2_mixed.py` (speaker/session disjointness checks,
AMI global-actor extraction and disjointness pass/overlap/violation cases,
manifest identity evidence incl. tampered-file detection, merge, violation
reporting). `ruff` clean, `black` clean.

## 10. Limitations and blockers

- D4 has no authorized PuriPuly-like audio: domain generalization is
  provisional (section 7).
- AliMeeting dev (Train) portion for the D3 pool was not downloaded
  (73.24 GB, single tarball); the eval set (3.42 GB) was downloaded, so the
  D2 held-out empirical portion is real, while the D3 real-meeting dev
  portion is AMI-only (extendable with more AMI SB sessions).
- AMI pilot covers 2 dev + 2 held-out sessions (~66 min); the full SB/SC
  sets (40 sessions) can be added by extending `AMI_*_PILOT_SESSIONS` — the
  pipeline is identical. The chosen pilot partition has verified zero
  global-actor overlap between dev and held-out (section 6); other AMI
  session pairs were not checked in this run, so extending the pilot
  requires re-running the global-actor disjointness validation.
- LibriSpeech dev-clean/test-clean/test-other are English-only; non-English
  coverage enters via AliMeeting (Mandarin).
- Per-speaker prosody variation beyond gain is not synthesized (no
  resynthesis tooling); "prosody" is represented by same-speaker
  cross-utterance A1->A2 pairs.
- The full generated WAV set is kept in the external build root; the
  committed manifests are validated against it (documented command in
  `data/README.md`).

## 11. What remains before Phase 3

1. No Phase 3 work was started: no thresholds/windows frozen, no detector
   sweeps, no selection, no provider policy work, no production wiring.
2. Phase 3 can consume `mixed_dev_pool` (dev) and
   `ls_held_out_clean` / `ls_held_out_other` / `ami_held_out_pilot` /
   `alimeeting_eval_pilot` (held-out) directly; the phase2 schema carries
   v1-compatible per-case fields (`case_id`, `wav_relative_path`,
   `duration_samples`, `wav_sha256`, `seed`, `regions`) plus condition/
   transforms/disjointness metadata.
3. If desired before Phase 3: extend AMI pilot sessions, download
   AliMeeting Train for a real-meeting dev pool, and fill the D4 import
   template once authorized PuriPuly-like audio exists (then the domain
   conclusion no longer needs the provisional label).
