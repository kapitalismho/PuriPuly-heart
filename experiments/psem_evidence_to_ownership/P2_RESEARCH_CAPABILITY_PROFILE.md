# P2 E2O-1 Research Capability Profile (frozen before outcomes)

Freeze: `psem.e2o1.freeze.v1` (`FREEZE.json`,
sha256 `f96dbdf14546ad106d9ac22d9240468bfa255947b8c35392c6190af35fc9ce90`).
Authority: #132 revised (parent) / #142 (this child). #133 is a completed
scoped audit; #121 STOP preserved; #98 Simple Anchor is a historical
comparator only. Research instrument only: grants no production mutation
right, does not amend `AUDIO-LISTEN-1` (#134) or #135.

## Audio input to PSEM

Source clock is file-sample time at 16 kHz. The replayed audio is
genuinely new per capture session (single continuous Soniox session per
case, chunk 512, realtime 32 ms pacing, trailing 100 ms, one finalize);
no hold, preroll, or split. Valid / missing / discontinuous spans come
from the stored session join (starts/ends/valid/masked/speech/frontiers);
the R2 `AB` subspan `[680352,685472]` joins to ZERO session frames and is
predeclared invalid support. Capture generation is `ami_mix_headset`
frozen DEV sessions. Receipt/frontier metadata is the stored per-frame
`frontiers[]` lookahead plus the measured session clock
(`session_open_s`, `arrival_wall`, `final_wall`).

## PSEM observation / request

Event id/revision: frozen F0 (`sigmoid(f0_raw) >= 0.5`) and H7301
(`sigmoid(cand_raw) >= 0.5887844788775033`, H100-C pinned #133 policy),
first-event per episode via `frontier_sweep.simulate_episode` with 1600
samples (100 ms) confirmation. Capture scope and reference generation are
the stored GT-derived / oracle-mapped single scope (never re-enrolled).
Source support is the joined valid+speech+unmasked span; estimated
transition is the returned boundary; observed frontier is the returned
frontier (includes stored lookahead); actual availability is
`send_wall(frontier) + L`, `L >= 0` UNKNOWN (never zero); validity and
uncertainty flag straddling words intact; semantic hypothesis is
sequential speaker transition at the boundary.

## Audio capability profile (research-only)

Supported operation: pre-translation pending boundary on ACCEPTED TEXT
only. Source clock as above; preserved timing is provider token `end_ms`
mapped to estimated source sample `p0 + round(ms*16)`. Mutable unit is
the accepted-text word group; expiry/commit is the text-ownership seal,
atomic at accepted-final arrival (`final_wall`) on the same session
clock. Sealed-PCM rule: PCM is never mutated. Provisional output is out
of scope; visible commit is never rewritten. Latency allowance is the
measured `max_permissible_lag = final_wall - send_wall(frontier)` per
case/arm; requests may be pending to partition subsequently accepted
tokens through the seal. All arms share deadlines, provider, text, and
policy. Word rule: whole-word `end <= boundary` stays left, else right;
straddlers intact + uncertain; token pieces grouped to whole words;
predetermined ±1280-sample sensitivity, never optimized.

```
PCM ownership seal
!= ASR final
!= text-ownership seal
!= translation dispatch
!= visible output commit
```

## Audio receipt

`applied` / `already_separated` / `too_late` / `unsupported` /
`invalid_scope`, with actual affected source/text units, applied
boundary, receipt/application time, and reason. Correct-transition
control uses the annotation-supported boundary +100 ms confirmation
support; actual compute lag stays UNKNOWN. No causal PASS is taken on a
lower bound; unmeasured lag fails acceptance rather than inventing one.
