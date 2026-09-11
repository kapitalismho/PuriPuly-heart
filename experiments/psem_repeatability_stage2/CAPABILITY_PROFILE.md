# Stage2 Research Capability Profile

Freeze `psem.repeatability.stage2.freeze.v1`. This profile is frozen BEFORE
outcomes and bounds what the probe can and cannot claim.

## Can measure

- GT word windows from the frozen public annotation members (lexical GT with
  start/end times; speaker identity via `meetings.xml` role mapping).
- NPZ evidence events (F0/H7301 first-event per full episode under the pinned
  policy) with full original masks and reference scopes.
- Per-chunk socket flush completion walls on the session clock (measured
  send side), open offset, raw provider arrivals, consumer accepted final,
  and the seal (accepted-final arrival, same clock).
- Ownership partition of GT-matched consumer words per arm boundary, with
  matched/unmatched/mixed denominators and ambiguity enumeration.
- Exact text/token-ID conservation across arms sharing one stream
  (DROP/DUP controls prove the check detects violations).
- Proxy-projected false segmentation on pure-A singleton word spans.

## Cannot measure (stays UNKNOWN, never zero-filled)

- Model/event compute lag (needs the sibling full-path timing profile;
  status PENDING at preliminary delivery).
- Out-of-payload and out-of-episode-frame support frontiers (no
  extrapolation of schedule as actual).
- ASR harms from word proxies (proxies are semantic-only, labeled NOT ASR).
- Ref-A-absent / B+C-observed safety (guard UNFULFILLED-real; no synthetic
  filler admitted).
- Any causal PASS on flush bounds alone (causal gate is separate).

## Fixed limits

- 3 provider sessions total, <=7s each, no post-connect retries.
- 0 translation / LLM calls.
- 5 arms exactly (none / F0 / H7301 / control / anchor); no per-case winners.
- Sensitivity +-1280 samples, predetermined.
- Machine equivalent: `CAPABILITY_PROFILE.json`.
