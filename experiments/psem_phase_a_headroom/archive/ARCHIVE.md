# Archive (superseded history, not active)

- `causal_ownership_ledger.pending.json` — pre-barrier placeholder (OBS missing, exit 2). Superseded at barrier.
- `causal_ownership_ledger.synthetic.json` — pre-barrier synthetic path proof (labeled never-real-quality). Superseded at barrier.
- First live candidate (post-barrier, pre-checkpoint) is not preserved as a file: it was overwritten in place by repair runs before the checkpoint. Its defects are recorded here instead, never reconstructed: fixed-cohort violation (matched 7/5/4 instead of frozen 8, 6/1/1, 8 from ±32000 search regions), single-schedule violation (F0 virtual-only missing flush ~0.8 s optimism; control missing containing-chunk service ~0.13-0.2 s), R2 anchor misreference (hardcoded role B against authoritative meeting mapping FEE083 role A), BC1 enrollment attempted where anchor absent. All repaired in the live ledger plus FREEZE v1.2 clarification.
