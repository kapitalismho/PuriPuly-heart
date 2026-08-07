# Experiment data

No public corpora, private recordings, or large binaries are committed here.
Per issue #51 Phase 2 rules, download/build scripts, manifests, hashes, and
small regenerable artifacts are committed instead.

## Phase 0/1 artifacts

- `manifests/b0_phase0.json` — deterministic dataset manifest for the Phase 0
  golden cases: baseline SHA, case metadata, wav SHA-256 hashes, and GT
  active-speaker regions. Regenerate it (byte-for-byte) with:

  ```powershell
  uv run python -m experiments.speaker_turn_boundary.build_synthetic_cases
  ```

- `manifests/phase1_dev.json` — Phase 1 dev cases (zero-gap handoff, overlap).

- `generated/*.wav` — deterministic 16 kHz mono PCM16 clips synthesized by
  `experiments/speaker_turn_boundary/synthetic.py` (numpy only, fixed seed).
  The manifest hashes pin their exact bytes; tests regenerate them and
  validate against the manifest.

## Phase 2 artifacts (issue #51, D1-D4)

### Committed here (small, regenerable)

- `manifests/ls_dev.json`, `manifests/ls_held_out_clean.json`,
  `manifests/ls_held_out_other.json` — D1 LibriSpeech-based deterministic
  synthetic manifests (schema
  `experiments.speaker_turn_boundary.manifest.phase2.v1`), 202 cases each:
  different-speaker gaps (800/300/100/0 ms), overlaps (100/300/500 ms, only
  where the active duration allows), durations 2.0/1.5/1.0/0.75/0.50 s and a
  seeded 0.30-0.50 s stress bucket, plus same-speaker, gain-variation,
  silence, noise-only, and codec/noise/bandlimit stress negatives.
- `manifests/ami_dev_pilot.json`, `manifests/ami_held_out_pilot.json` — D2
  AMI pilot (scenario-only partition: SB dev = ES2003a, IS1008a; SC held-out
  = ES2004a, IS1009a). Single-channel recipe: `Mix-Headset` 16 kHz mono.
  GT from per-participant `words.xml` (speaker derived from the filename as
  `{meeting}.Participant{letter}`).
- `manifests/alimeeting_eval_pilot.json` — D2 AliMeeting M2MeT eval pilot,
  8 sessions (R8001_M8004 ... R8009_M8020). Recipe: far-field array channel 0
  materialized as canonical 16 kHz mono PCM16 (`alimeeting/far_ch0/` in the
  external root). GT from `TextGrid` IntervalTier per participant
  (N_SPKxxxx); overlaps inferred from simultaneous tier intervals.
- `manifests/mixed_dev_pool.json` — D3 deterministic mixed development pool
  (LibriSpeech dev synthetic + AMI dev pilot), machine-checked session/
  speaker-disjoint from every held-out manifest.
- `manifests/puripuly_like_provisional.json` — D4 provisional record: no
  authorized PuriPuly-like audio was available; import schema template in
  `data/puripuly_import_template.json`.
- `data/results/phase2_d1_validation.json`, `data/results/phase2_d2_validation.json`,
  `data/results/phase2_d3_validation.json` — machine-readable validation reports
  (per-manifest semantic hash + `manifest_canonical_file_sha256` (SHA-256
  of the canonical LF manifest file bytes, i.e. the staged Git blob after
  normal Git text normalization — not raw platform-specific worktree bytes)
  + canonical-bytes check, wav SHA-256, durations, source-cut bounds,
  zero-gap acoustic evidence, GT transition kinds, speaker/session
  disjointness, and AMI global-actor disjointness with the compared
  actor-ID lists). Regenerate them with
  `uv run python -m experiments.speaker_turn_boundary.validate_phase2`.
- `generated/*.wav` for 13 representative D1 sample cases (zero-gap, overlap,
  same-speaker, gain, stress transforms, silence, noise). These are tiny
  CC BY 4.0 fixtures; the full 606-case wav set is regenerable and lives in
  the external phase-2 build root (below), never in Git.

### External roots (never committed)

- Public corpora: `%TEMP%\opencode\stb_phase2_corpora` by default
  (override with `STB_PHASE2_CORPORA_ROOT`):
  - `archives/` — `dev-clean.tar.gz`, `test-clean.tar.gz`,
    `test-other.tar.gz` (official MD5s in `corpus/librispeech.py`),
    `ami_public_manual_1.6.2.zip`, `alimeeting_eval.tar.gz`.
  - `LibriSpeech/`, `ami/audio/`, `ami/annotations/`,
    `alimeeting/Eval_Ali/`.
  - `phase2_build/` — default Phase 2 build output dir for all three build
    scripts (`<corpus root>/phase2_build`, resolved on this machine to
    `C:\Users\salee\AppData\Local\Temp\opencode\stb_phase2_corpora\phase2_build`;
    override per run with `--out`). Holds the full deterministic build
    output (all generated wavs under `generated/` + manifests under
    `manifests/` + build validation reports under `results/`). By default
    no generated wavs are written into the Git worktree.
- External corpus archives were downloaded and hash-verified during the
  Phase 2 run (sizes: dev-clean 337,926,286 B; test-clean 346,663,984 B;
  test-other 328,757,843 B; AMI manual annotations 22 MB zip; AliMeeting
  Eval 3,673,718,355 B).

### Rebuild and validate (byte-identical output)

```powershell
# Builds default to the external <corpus root>/phase2_build output dir;
# use --out to redirect (do not point --out at this data dir unless you
# also want 606 regenerated wavs in the worktree).
uv run python -m experiments.speaker_turn_boundary.build_phase2_cases --validate
uv run python -m experiments.speaker_turn_boundary.build_phase2_real
uv run python -m experiments.speaker_turn_boundary.build_phase2_mixed

# Validate the committed manifests against the repo data dir plus the
# external roots (<corpus root>/phase2_build and <corpus root>) and
# regenerate results/phase2_d{1,2,3}_validation.json:
uv run python -m experiments.speaker_turn_boundary.validate_phase2
```

The committed D1/D2 manifests are byte-reproducible from any output dir for
a fixed corpus root (they store relative wav paths); `mixed_dev_pool`
embeds its absolute build root by design, so its file hash varies with the
root while its content is identical.

Phase 2 details: `PHASE2_REPORT.md`.
