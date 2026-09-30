# #204: finite S200-only-off ablation

## Explicit authorization and frozen scope

The user explicitly approved exactly two additional requests: `brief_interjections`, then `long_return_gap`, each `D_no_S200`; at most277.760 transmitted audio seconds and540 WebSocket-envelope seconds, no retries. With completed A–D baseline16 requests/4,070.168 audio/5,205.829 connection-envelope seconds, the authorized cumulative bound is18 requests/4,347.928 audio/5,745.829 envelope seconds, within the original24/6,000/7,200 caps. Existing CC BY4.0 public AMI only. This configuration/report precedes all ablation calls. No timestamp-reservation call, new candidate sweep, production change or additional recording is authorized.

The original A–D results, source traces, experiment configuration, plans and history remain immutable. The new `config.json`, `run_plan.json`, per-call quota reservations, dry/live logs, source maps and report live only here. Production baseline remains4dc46fdac8c771d3bb58e46bd44304b86fde9c47; later integration commits do not rewrite pre-call revisions or sender hashes.

## One intervention, honest pacing

Start from each frozen original-D nominal800 packet/control sequence, verified against its actual immutable write log. Remove only packets explicitly marked `S200`. Every source packet's exact bytes/hash, packet boundaries, replay/source ranges and source availability are unchanged; no source repacketization or deduplication. Preserve every manual-finalize control, its source availability and terminal flag. Shift provider-input sample positions, source-piece maps and cuts by the exact cumulative deleted sample count. Save both original and shifted provider positions explicitly.

Brief removes19 S200 pieces/3.8 seconds, sends121.504 seconds with25 finalize controls. Return-gap removes22 pieces/4.4 seconds, sends156.256 seconds with36 controls. Source-availability/cut positions remain those of the original full120/180-second source. Genuine terminal completion stays at original source end, not compressed source time.

Reuse the existing sender and concurrent receive/terminal completion. Original D is the isolated manual-finalize diagnostic with **no intermediate `<fin>` ACK gate**, not an assertion of production adapter/archived segmented parity. Removing paced S200 necessarily removes its transmission waiting and may advance actual control/successor writes; that timing change is inseparable from S200-off here. No invented replacement delay makes wall clocks appear identical. Actual writer stalls, source backlog, control/receipt/successor timing and pending completions remain logged. Never transmit future source audio. This is a recorded-file provider comparison, not first-readable caption-readiness evidence.

## Projection and independent diagnosis

Retain raw responses, native token/counter timing and all source maps. Reuse existing Soniox SDK2.3.0 speaker grouping in provider time before source intersection, with literal native clocks and no per-finalize subtraction. Score each whole clip with existing `score_case`, pyannote.metrics4.0.0, collar0, overlap included, full UEM and one entire-case optimal speaker mapping. The independent lexical owner applies exactly the same frozen alignment/global-mapping rules and explicit unmatched/ambiguous/interjection/overlap coverage to old D and new S200-off.

## Commands

From repository root, use the discovered Windows Python3.12 executable for dry/live/projection:

```
python -B experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/run.py
python -B experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/run.py --live
python -B experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/run.py --project
wsl.exe --cd <worktree-WSL-path> experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/factorial/ablation/run.py --score
```

Default is offline/dry. Live only follows the exact frozen two-case plan, requires explicit authorization plus configured credential presence, persists cumulative reservations before connect and stops on any failed/uncertain request with no retry. Calls stop after these two cases. Dry/projection/scoring/native/coverage/resource evidence is pending at this pre-call checkpoint. No adoption or production-readiness conclusion is presumed.

## Final pre-call dry checkpoint

Actual default offline runner passed2/2 streams: brief121.504 seconds/25 barriers; return156.256 seconds/36 barriers. `dry_verification.json` verifies every retained source packet byte/hash/length/source range/replay position/availability against the actual original-D log (2,533 brief packets;3,258 return packets), unchanged control availability/terminal sequence, exact explicit provider-position remapping, no future source writes and completed terminal receipts. No S200 packet remains. The finite plan pins baseline spend/config/ledger, new authorization, traces, original-D results/write logs, sender sources, PCM/event identities and shifted maps. Frozen executed sources are `frozen/shared_sender.py` and `frozen/ablation_sender.py`; focused Ruff passed before freeze.

Two calls reserve277.760 audio seconds/540 connection-envelope seconds; cumulative authorized bound is18/4,347.928/5,745.829. The original16-call quota ledger stays immutable; the separate ablation ledger carries only these two new reservations and references the frozen original spend. No credential value is printed or retained in configuration/artifacts.

## Completed provider, projection and resource evidence

Both approved requests completed on one connection each, with25+36=61/61 finalization receipts and finished responses; no retries or further calls. `--project` returned `projected_ablation=2 sdk=@soniox/node@2.3.0`; WSL `--score` completed both whole-case projections with pyannote.metrics4.0.0. Original A–D outputs/config/history remain untouched.

| Case | Original D DER % | S200-off DER % | Difference pp | Original /off predicted speakers | Off miss /false alarm /confusion speaker-s |
| --- | ---: | ---: | ---: | --- | --- |
| brief_interjections | 83.26 | 69.81 | −13.45 | 3 /3 | 39.351 /18.710 /24.034 |
| long_return_gap | 43.66 | 44.03 | +0.37 | 2 /2 | 7.519 /36.014 /11.187 |

These are literal-time, speaker-grouped SDK DER, not native word-identity percentages. In brief, the component differences are miss−1.757/false alarm−0.953/confusion−13.110 speaker-seconds; return changes are +0.072/−0.144/+0.528. Word identity/coverage remains a separate independent diagnostic before any quality recommendation.

Actual new resources:2 calls/277.760 audio seconds/380.203 conservative connection-envelope seconds, versus the approved540-second maximum. Connected-event-to-closed-event time is378.766 seconds. Cumulative resources are18 calls/4,347.928 audio/5,586.032 connection-envelope seconds, within both the explicitly amended18/4,347.928/5,745.829 limits and original24/6,000/7,200 caps. Cumulative finalization receipts are487/487 (426 baseline +61 ablation); zero retries and no timestamp-calibration spending. `quota_ledger.json` here records only the two ablation requests with the original frozen spend, leaving the original ledger immutable.

`comparison.json` verifies **actual** retained source packet byte hashes/lengths/source ranges/availability match original D exactly, controls retain their availability/terminal sequence with shifted provider positions, every actual write occurs after source availability, all transmitted bytes are accounted, and every requested barrier completes. It retains actual receive/control/successor timestamps and resource bounds.

## Inseparable observed pacing differences

| Case | Source-packet write difference off−D median /min /max s | Finalize-write difference median /min /max s | Connection-envelope D /off s |
| --- | --- | --- | --- |
| brief_interjections | −2.500 /−4.781 /+0.125 | −2.750 /−4.750 /−0.125 | 163.797 /159.156 |
| long_return_gap | +0.079 /−1.578 /+3.579 | −0.046 /−1.546 /+3.345 | 219.750 /221.047 |

Availability signatures and retained packetization are identical, but actual wall-time scheduling is not. Brief writes generally advance as padding transmission disappears; return’s no-padding run actually has a longer connection envelope. Host/control/receive timing can dominate small padding savings. Never infer a guaranteed latency or billing saving from8.2 removed PCM seconds, and do not attribute every DER change exclusively to voice recognition. This intervention changes synthetic samples and their paced transmission together; no artificial compensating wait was inserted.

The actual quota guard was separately exercised offline with cumulative request/audio/WS limits below the frozen plan and with new-call audio/WS caps below277.760/540; all five rejected before connect. These are safety mechanics, not accuracy checks. The same native/coverage/readiness diagnostic owner completed both immutable new cases; no additional provider call is scheduled.

## Matched native identity, text coverage and provider readiness

The independent owner completed [the same frozen lexical/global-mapping diagnostic](../../ablation_identity_report.md), with full witnesses in [ablation_identity_evidence.json](../../ablation_identity_evidence.json). It independently verified actual packets, cut/control remaps, source/replay maps, provider configuration and finalized counts. Native timing widths do not provide identity support; ambiguous/unmatched/overlap/short words remain explicit. There is one new run per case against earlier immutable D, no repeat/significance or holdout claim.

| Case | Metric | Original D | S200-off |
| --- | --- | ---: | ---: |
| brief_interjections | Wrong % of known anchored words | 36.23 | 22.96 |
| brief_interjections | Correct all-reference-word coverage % | 38.68 | 45.71 |
| brief_interjections | Recognized words | 369 | 353 |
| brief_interjections | Unmatched reference words | 139 | 145 |
| brief_interjections | Correct short-interjection words /17 | 2 | 1 |
| brief_interjections | Correct words available at actual finalize writes % | 4.84 | 3.30 |
| brief_interjections | Median complete-final-word receipt after actual input write s | 3.938 | 3.641 |
| long_return_gap | Wrong % of known anchored words | 9.45 | 9.50 |
| long_return_gap | Correct all-reference-word coverage % | 80.71 | 80.27 |
| long_return_gap | Recognized words | 433 | 432 |
| long_return_gap | Unmatched reference words | 46 | 47 |
| long_return_gap | Wrong anchored initial/returning-speaker words | 38 | 38 |
| long_return_gap | Correct words available at actual finalize writes % | 5.99 | 4.21 |
| long_return_gap | Median complete-final-word receipt after actual input write s | 3.726 | 3.585 |

Brief final attribution improves in overlap and nonoverlap strata: correct coverage30.25→37.39% and47.93→54.84%, respectively. That does **not** erase the short-interjection regression: correct short words2→1, unmatched short words11→14. Its conditional short-word wrong rate33.33→0% must not be advertised as a quality win while correctly attributed coverage halves.

The returning-speaker failure is unchanged: the same23 anchored words in the initial published turn get nativeID2, globally mapped to a different reference speaker; the same15 in the return get nativeID1, globally mapped to another speaker. Both original and S200-off retain38 wrong words and only two predicted native IDs. S200 removal does not recover that identity merely by shortening provider-input support.

Post-input medians improve modestly, but correct final-word availability at actual control writes declines in both cases as controls/input may advance. Exposure at frozen source-cut time is0% in both arms under this paced recorded-file replay; substantial sender backlog is separate from provider completion. These retrospective whole-case/reference and actual-write associations combine provider/wire/finalization effects, are not independently calibrated acoustic latency, and do not exercise the application’s first-readable publication/color ownership.

Actually received final processed counters differ by−3,960 ms (brief) and−4,080 ms (return) while controls remain unchanged and removed PCM is3,800/4,400 ms. Retain those literal values; neither source-map shortening nor these discrepancies authorize a guessed timestamp subtraction. R4 remains blocked by missing independently verified acoustic anchors with quantified uncertainty. No calibration request is spent.

## Final disposition and execution boundary

Director accepted the completed **18-call** boundary and selected **defer production adoption**. This two-case intervention supports a bounded brief-window native-attribution/coverage improvement, but **S200-off is not a blanket quality/text/readiness improvement**: short-interjection/text losses remain, the returning-speaker failure persists, and correct evidence at actual finalize writes decreases. Retain current product behavior. Fewer finalizations still present the measured quality/arrival tradeoff from A–D; neither removing S200 nor fewer controls is a validated publication/ownership operating point.

Seven-case expansion, repeats and operating-point trials are not run, not empirically rejected or declared unnecessary. The independent acoustic-clock prerequisite remains exactly missing. No extra candidate, sweep, observer, translation pass, private recording, production change, fixed color wait or physical microphone/VRChat/HMD certification is performed.

Passed:2/2 actual dry streams; exact original-D retained-packet/replay/availability/manual-control proof; shifted source/provider maps; no future actual writes;2/2 actual requests and61/61 barriers/finished receipts; pinned SDK/scorer2/2; identical frozen lexical/global-mapping/coverage/arrival diagnostic; cumulative request/audio/WS quota-boundary safety probes; focused sender/runner Ruff. No provider failure or retry occurred. Bounds and credentials are handled exactly as frozen, with no key output.

All surfaces are frozen at completion: original A–D config/results/history/ledger remain immutable, executed ablation/shared source snapshots match the pre-call plan, and the new comparison/report/receipts reference those artifacts. The parent [factorial report](../../factorial_report.md) and cumulative completion record incorporate this approved ablation without rewriting original execution revisions. Director owns integration, maintained-doc updates, research commit and independent whole-checkpoint review. **No further calls.**

