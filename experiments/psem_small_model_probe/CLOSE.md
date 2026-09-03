# PSEM small-model probe — close-out (issue #117)

> VERIFIED existing outputs only; no new claims. V2 EVAL sessions reused
> dev-only per program approval — no generalization claim; V3 holdout
> required for any selection claim.

## Gate table 0–6

| Gate | Verdict | Key numbers |
|---|---|---|
| 0 manifest freeze | FROZEN / PASS | 84 rows: CAL12 12 (5 sess), MAIN48 48 (6), EXT24 24 (4), ONTOLOGY16 16 + CONTROL24 24 (MAIN48 subsets). `file_sha256 e5956ab0…6582`, `freeze_sha256 91efb276…6d82`. Session-disjoint (CAL∩MAIN={}, EXT∩(CAL∪MAIN)={}); `episode_id` unique 84/84; `causal_bindable` 84/84. |
| 1 adapter + decoder | PASS | smoke 6/6; `verify_vendor` 16/16 contracts (+20/20 live on synthetic PCM per LOCALENV). ECAPA receipt sha `0575cb64…e3d0126a`. CPU-only. Corpus: 15/15 sessions resolved, mono 16 kHz, frame-count match (LOCALCORPUS). |
| 2 CAL12 taus | FROZEN | tau=0.05 all 6 cells. firered O: 5/8 false, 1/2 missed; C: 4/8 false, 1/2 missed. neovad O+C: 0/8 false, 2/2 missed. contam 731.7 s/h. |
| 3 MAIN48 native O | firered SUPPORTED / neovad COLLAPSED | firered O: 12/32 false, 5/8 missed, CUT 3/8, src_err p50/p90 -820/1068 ms, dec 500/500 ms. neovad O: 0/32 false, 8/8 missed, 0 sens. contam 877.7 s/h. ≥1 formulation supported → proceeded to Gate 4. |
| 4 MAIN48 causal C + CPU | O→C FLAT; CPU PASS | firered C: 13/32 false, 5/8 missed, CUT 3/8, src_err -670/1098 ms, dec 500/500 ms. neovad C collapsed (0/8 CUT, 0 sens). CPU: all 4 cells `rtf_le_025` + `p99_lt_chunk` (firered C RTF 0.0508, O 0.0525; this machine). |
| 5 VAD replay | INTEGRATION CLEAN | GT-gate vs prod-VAD (firered C, tau 0.05): missed 5/8→1/8, false 13→21/32; retention 3/3; hit-count ratio 2.333. Gate strictly wider: agreement 0.6522, dropped 138/14136 GT-speech frames, added 8210 gate-on frames, 0/48 zero-coverage. |
| 6 ontology | REOPEN ownership (8/16) | T-better 8/16 (rule: ≥4 reopen); loss-risk increase 0/16; both-poor 8/16. Blind X/Y seed 117. GT-proxy substitution (no real ASR/translation). |

Topology note: MAIN48 carries zero `A->A+B->A` rows, so the mandatory KEEP
cell reads 0/0 for both models x regimes; KEEP rests on A / overlap_return /
A+A+B (n=32). ONTOLOGY16 session concentration: all 16 eps from ES2009a +
R1021_M1940 only.

## Gate 7: EXT24 / CONTROL24

- 10%-boundary: NOT triggered — no headline sits near a pass boundary:
  FireRed false cuts 12–13/32 blow the x1.10 budget at every tau (far above
  passing), NeoVAD missed 8/8 both regimes (collapsed, opposite end).
- corpus-opposite: NOT triggered — balanced 4+4/stratum design and no
  AMI↔AliMeeting reversal behind any gate verdict; pooled headlines decide.
- overlap-conflict: NOT triggered — overlap_return sens-vs-primary divergence
  (firered sens 346/340 vs primary 6/12 false cuts) recorded diagnostic-only
  with no threshold/frontier action.
- CONTROL24: SKIPPED per minimum-validation — ECAPA adapter exists (Gate 1
  live) but FireRed is unpromotable, so the comparison would not change action.

## Engineering conclusion (#117 close-condition letters)

- NeoVAD = E: off-the-shelf formulation unsupported as anchor tracker
  (0/8 CUT both regimes, 0 sens hits).
- FireRed = weak partial signal, NOT promotable: 12–13/32 false cuts blow the
  x1.10 budget at every tau; 3/8 CUT both regimes; O≈C flat so binding is NOT
  the bottleneck.
- VAD integration clean (Gate 5: retention 3/3; wider gate trades misses 5→1
  for false cuts 13→21 — over-triggering, no under-triggering).
- Ownership = proxy-REOPEN with GT-substitution + session-concentration
  (ES2009a + R1021_M1940 only) + both-poor caveats → follow-up issue, not
  this probe.

## Known limitations

- MAIN48 zero `A->A+B->A` rows (mandatory KEEP cell 0/0).
- ONTOLOGY16 GT-proxy, not real ASR/translation (`sherpa_onnx` weights
  uncached, HF hub cache empty, translation needs an LLM owner).
- EVAL dev-only (DEV alone too few sessions for 3 disjoint splits); V3 fresh
  holdout needed before any model-selection claim.
- CPU numbers this-machine only (Win11, Ryzen 7 9800X3D, 16 CPUs,
  torch intra/interop 8).

## Follow-up candidates (open separately, not implemented)

- FireRed operating-point / policy work.
- NeoVAD foreground-transfer-evidence-only reuse.
- Ownership/transfer primitive (needs real-ASR confirmation).
- Extraction / conditioned-ASR (both-poor 8/16).
