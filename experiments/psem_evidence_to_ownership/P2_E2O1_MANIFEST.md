# P2 E2O-1 Manifest (Outcome #142 under revised #132)

State: IMPLEMENTATION_READY. No commits permitted; no reviewed claim.

## Authority

- Parent/governing: #132 revised PSEM program. Executer: #142 Child P2/E2O-1.
- Prior audit: #133 completed (Branch F terminal, historical scope only).
- Comparator: #98 Simple Anchor (historical minimum comparator; not native
  lifecycle proof). Deployment: #121 STOP/inconclusive; H7301 pinned
  diagnostic baseline; F0 retained.
- Branch baseline `experiment-v2-speaker-change-turn-boundaries-ls` HEAD
  `83aaed984b8b245082f3ffe7bb15d71f3242361f` (upstream same, clean).
- Freeze `psem.e2o1.freeze.v1` in `FREEZE.json` (sha256
  `f96dbdf14546ad106d9ac22d9240468bfa255947b8c35392c6190af35fc9ce90`),
  recorded BEFORE any replay/results. Capability profile frozen before
  outcomes (`P2_RESEARCH_CAPABILITY_PROFILE.md/.json`).

## Scope (12 predeclared cases; mechanism examples, not prevalence)

Positives/disagreement: P1-G02 (ES2009b f829-832), P2-G03 (ES2002b
f2888-2901), P3-G04 (ES2009a f5016-5022), P4-SCORED (EN2009d A00343,
boundary 52156984). Negatives: N1-G10, N2-G13, N3-S1, N4-S2. Guards: R1
(A->A+B->A), R2 (A->A+B->A, AB subspan zero-frame invalid support), T1
(A->A+B->B), G4th (ref A / observed B+C; resolve or explicit
unsupported + synthetic SYN-G, never empirical). Full intervals, episodes,
and policies in `FREEZE.json`. G03/G04 annotations may not establish
clean positive lexical ownership; the replay reports the actual finding.

## Method

- Evidence: `load_validated_export` + `frontier_sweep.simulate_episode`
  (F0 tau 0.5; H100-C tau 0.5887844788775033; 1600-sample confirmation;
  stored lookahead in frontier). No per-case winners.
- Text: same continuous Soniox stream per timed-text case across arms
  (repaired 53-token/165-char acceptance for P4; at most two new captures
  P2/P3 per freeze). Word rule frozen: `end <= boundary` left else right,
  straddlers intact+uncertain, whole-word grouping, ±1280 sensitivity.
- Receiver: fake Audio research-only; receipts applied / already_separated
  / too_late / unsupported / invalid_scope; requested vs applied
  preserved; ownership errors supported-only else null; conservation
  exact and separate from accuracy; unknown exposure + guard harms +
  reference cascade tracked.
- Comparators per case: no-PSEM, F0, H7301, correct-transition control
  (annotation boundary +100 ms, lag UNKNOWN), Simple Anchor (historical,
  non-causal), zero-delay oracle shown only as unreachable headroom.

## Executable replay

```
./.venv/Scripts/python.exe experiments/psem_evidence_to_ownership/replay.py --run
./.venv/Scripts/python.exe experiments/psem_evidence_to_ownership/replay.py --smoke
```

`--run` writes `results/P2_EVENT_OWNERSHIP_LEDGER.json`,
`results/contract_smoke.json`, `results/evidence_bounds.json`, and
`results/replay_receipt.json` (hash-bound candidate receipt). `--smoke`
runs the synthetic receiver contract only. Capture helper:
`python experiments/psem_evidence_to_ownership/capture.py --case P3-G04`
(resp. `P2-G03`); budget max two attempts, failures recorded.

## Required outputs

- `P2_E2O1_MANIFEST.md` (this file)
- `P2_RESEARCH_CAPABILITY_PROFILE.md` (+ `.json` machine equivalent)
- `results/P2_EVENT_OWNERSHIP_LEDGER.json` (+ byte-identical top-level copy `P2_EVENT_OWNERSHIP_LEDGER.json`)
- executable replay (`replay.py --run`)
- `results/P2_NEXT_DECISION.md` (+ byte-identical top-level copy `P2_NEXT_DECISION.md`)
- supporting: `FREEZE.json`, `captures/`, `results/evidence_bounds.json`,
  `results/replay_receipt.json`, `results/contract_smoke.json`

## Constraints / non-goals (all #142 bindings)

No production edits, code comments, git mutations, publish, training, new
inference, sweeps, new dataset, provider discovery/replacement,
translation/LLM calls, or reference lifecycle changes. No combined
pipeline claims across #133's separate experiments. Unknown never
converted to PASS/zero-cost. Exactly ONE named next branch only if
justified; else gate BLOCKED with the missing measurement (never a false
stop or unknown-as-failure).

Repair layer: FREEZE_ADDENDUM.json (original freeze retained) + REVIEW_RECORD.md. Known-good runtime: ./.venv/Scripts/python.exe. No captures remain; no commits.
