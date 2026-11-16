# #441: native diarization and turn-boundary experiment

## Decision

**Defer a production/default switch to Scribe and keep the current controller.** Native Scribe metadata warrants a bounded integration-design experiment, but this pilot does not establish superior diarization accuracy, a useful `turn_prediction` advantage over matched VAD, or safe late caption regrouping.

- Fix the proposed integration boundary first: join delayed metadata by `segment_index`, preserve coherent source text and context, and distinguish identity-only updates from source-unit changes that require new MT. Actual Japanese source-fragment translations changed meaning. This is not a demonstrated defect in the existing Soniox product path.
- Do not implement a Soniox composition repair from envelope-only probes: corroboration found no confirmed harmful join among the six flagged cases, with two still unresolved.
- Nemotron's historical 1.04 s profile worked on the existing AMD GPU. This establishes standalone activity/identity feasibility, not text alignment or an integrated alternative; it does not reopen the cancelled [#218](https://github.com/kapitalismho/PuriPuly-heart/issues/218) work.
- The next narrow question is whether an indexed Scribe adapter can preserve complete source units when metadata arrives, avoiding word-fragment MT while exposing useful attribution. Waiting for metadata versus translating immediately and regrouping must account for actual latency and additional requests, not assume free recoloring.

Issue: [#441](https://github.com/kapitalismho/PuriPuly-heart/issues/441). Product baseline: `9f7474b94b885a7a9a4777b43a3a05026e3e248e`. Production sources, default routing, settings and protected documentation were not changed. No PR, remote publication, issue closure or Project-state update is claimed.

## Authority, frozen input and reference limits

The owner's subsequent instruction authorized automatic execution without new human listening and permitted replacement audio. The [readiness record](issue441-readiness.md#owner-amendment-automatic-execution) preserves that amendment. References are independently authored corpus annotations plus automatic voice/energy checks, **not newly listened or acoustically adjudicated truth**. Candidate output did not select or retime the anchors.

The [frozen manifest](issue441-inputs.json) contains six 120 s recordings: 720 s total, mono PCM16/16 kHz; 120 anchors, 24 change/re-entry opportunities and 24 diagnostic turn opportunities. Frozen at `2026-11-16T10:42:56.903751Z`; SHA-256 `dc9b9e08ddc55c2f1e501de007ceaf8d7c1b24cd529c9c5e0c2f0dcc81acf294`.

| Recording | Source / original offset | Speakers | License | Prepared WAV SHA-256 |
|---|---|---:|---|---|
| EN | AMI ES2004d / 32.368 s | 3 | CC-BY-4.0 | `ee19cafe7382e7f26249b273c0f27ce080963788e35c2ca67034f234d903b6d6` |
| JA | IngCrowd dialogue01 / 86.04 s | 2 | CC-BY-NC-4.0 | `8cdbe6868df6e141f80a14ce7e49b501cd371e185e2899a2b1a659dc22d06a2b` |
| KO | Seoul s13m26m1 / 66.588125 s | 2 | CC-BY-NC-2.0 | `cb3ec34dfa8ef68757a24fd02ece51e5dcb02b6c18bd964d4b1f8c40c433a088` |
| ZH | AliMeeting R8002_M8002_MS802 / 0 s | 4 | CC-BY-SA-4.0 | `90ebee6603cc45d83f095dfa2e918150e34aab38edaea5688a5d8756fb982775` |
| EN diagnostic | AMI ES2004d / 259 s | 4 | CC-BY-4.0 | `44e4699d5898b40a312ebc6ef82aa92739a52aade3ee0e1e2d228d4aac706fb6` |
| JA diagnostic | IngCrowd dialogue01 / 312 s | 2 | CC-BY-NC-4.0 | `e1637913e7fcc30cf92c9bc70674b827fde3d55ed8fcbc491070443dcc94ed6a` |

ZH uses fixed channel 0 of the original eight-channel OpenSLR 119 audio, not an undocumented mirror downmix. Source URLs/revisions and annotation provenance are in the manifest. Original audio and model weights are not published with this report.

Important limits:

- All 20 KO anchors belong to the interviewee. Interviewer spans failed the frozen duration/voice/energy eligibility rule; this was diagnosed from source-only evidence before inspecting candidate output, not corrected by selecting easier output. KO interviewer lexical truth is unavailable, and the single annotation tier cannot establish absence of overlap.
- EN diagnostic anchors cover 19 MEE014 and one MEO015; its other two reference speakers have no anchor. Session mapping and transition claims retain this limitation.
- ZH has speaker/time truth but no lexical reference. Genuine code switching was not established; loanwords/acronyms are not substituted for that requirement.
- Diagnostic inventory includes short turns, overlap, A→B→A returns, near-gapless changes, negation/number mentions, continuing pauses and long speech. Semantic end/intermediate/unsafe labels are transcript-derived hypotheses. Four EN intermediate points have zero-width annotation gaps: report offsets only, not an invented binary success tolerance.
- Frozen opportunities remain in denominators when unavailable. This small, unevenly represented sample is not DER, WER/CER, a population accuracy estimate, or proof of speaker-count generalization.

## Executed profiles

| Arm | Input / segmentation | Native configuration and observation boundary |
|---|---|---|
| N | Six continuous original recordings; 1.04 s streaming profile | Standalone Vulkan Nemotron, eight activity channels, immutable 10 ms outputs, activity threshold ≥0.5. No transcript, translation or display score. |
| S0 | Continuous original audio, silence preserved, 100 ms PCM sends; no intermediate finalize | Soniox `stt-rt-v6`, diarization on, endpointing off, recording-language hint, non-strict/default hints, language identification off, no context terms. |
| S1 | Actual production VAD/controller/context replay and padding | Same native Soniox configuration; native attribution and production caption composition scored separately. Structural replay uses immediate deterministic MT and is not actual MT or GUI timing. |
| E0 / T0 | Continuous original audio, 32 ms sends; actual current controller supplies manual commits | Scribe `scribe_v3_live`, diarize/timestamps on, background mode off, `multilang_mode=true`; language and secondary-language query fields omitted, no keyterms/editing. T0 reuses E0 diagnostics. |
| T1 | Two diagnostic recordings; continuous 32 ms sends | `commit_strategy=vad`, silence 1.5 s, threshold 0.4, minimum speech/silence 100 ms; conservative client 10 s source-age cap. No controller 4/5 s opportunity cuts. |
| T2 | Identical diagnostic input, send cadence, VAD settings and cap | `commit_strategy=turn_prediction`, nominal threshold 0.9. Requested policy and observed behavior are separate: server echoes did not independently confirm a distinct prediction policy. |

S1 uses actual Silero threshold 0.4, 900 ms hangover, 500 ms pre-roll and bundled SmartTurn/current controller. The controller checks the 10 s cap before 4/5 s opportunities; opportunity/ordinary/timely-incomplete pauses are 192/512/800 ms. E0 uses these boundaries without discarding VAD-unselected source audio.

T1/T2 were repaired to match E0's 512-sample/32 ms cadence **before either T condition ran**. Their cap age uses the latest completed manual frontier or received timed-word end; it is not the unobservable exact server-consumed frontier. Trailing silence can make a cap earlier than 10 s after a server boundary, and untimed commits do not reset it. Cap/EOF races are excluded from useful prediction evidence rather than attributed causally.

The Scribe effective-config echo included `vad_commit_strategy=true`, 1.5 s/0.4/100/100, v3, timestamps and background-off, but did not echo the distinct prediction, multilingual or diarization flags. Actual diarized responses and numerical turn probabilities were observed. This does not prove acceptance of every requested policy parameter.

### Nemotron runtime

Historical profile: chunk/right/left/FIFO/cache/update = **9/4/0/264/264/222**. Restored patched NeMo-Speech.cpp `b809bbb467fb2da2be0042fff5ef1f503828c9cf`, llama.cpp `bd4f514db14d87fded667787a7a963bfbaa98e89`, vcpkg `9e593bb18ea69cc5095e012465dcd675a822ed0d`; Bonenk model revision `25cf0d6b0c91b679376df0f6ef3087b4fd96af1f`. Model SHA-256 `9b277e1dcec60b2c771a1dcb45dc8e36f7285ff7c1619d880d3926a373036b38`; historical patch SHA-256 `e81a179f2a71b76b8869a4a2710983c23f219717b344cf5c37af86574f90b748`; runtime manifest SHA-256 `94732932e03718bbff4c00a9e12cb9bc3d56c8789488288b6b7ec93f51142294`.

Actual device: AMD Radeon RX 7900 XTX, Vulkan, no neural CPU fallback. A Windows long-path build failure was preserved; a temporary owned drive mapping allowed the same build recipe to complete and was removed. A 3 s silence smoke passed. Warm-up maximum 611.904 ms was retained, not subtracted. Six real sessions consumed all 11,520,000 source samples; explicit EOF padding was 17,152 samples per session, with scored output clipped to 120 s. Summed connections: 721.058 s. Nonempty update medians were 22.8–23.7 ms, p95 approximately 28 ms; maximum 121.982 ms on initial EN and 30.8–33.4 ms elsewhere. Maximum source-pacing lateness was 23.088 ms. Processing time is separate from the 1.04 s input buffer.

Only initial EN overlapped cloud speech execution. Later N runs occurred while cloud cohort schedulers were paused; whole-host CPU/memory observations are not GPU-contention qualification for general concurrent use.

## Consumption and preserved failures

| Selected cloud arm | Sessions | Audio sent (s) | Connection duration (s) | Terminal status |
|---|---:|---:|---:|---|
| S0 | 6 | 720.000 | 721.701 | Completed; JA uses the one authorized execution-anomaly repeat |
| S1 | 6 | 696.932 | 729.309 | Completed; selected input/context/padding differs intentionally |
| E0 | 6 | 720.000 | 748.752 | Three completed, three post-drain close-error partials |
| T1 | 2 | 240.000 | 246.410 | One completed, one post-drain close-error partial |
| T2 | 2 | 240.000 | 246.395 | One completed, one post-drain close-error partial |
| Total | 22 | 2,616.932 | 2,692.567 | All selected source input delivered |

There were **23 actual paid ASR connections**, including the failed original S0/JA attempt: 2,693.632 s sent and 2,769.443 s connected. No quality retry or key-validation call occurred. These are measured durations, not an independently reconciled ASR invoice.

- Original S0/JA failed at 76.7/120 s on an atomic JSON `Path.replace` `WinError 5`. A concurrent observer collision was suspected, not proven. Its checkpoint, `.writing` file, journal and process failure remain intact. After stopping that observer, `ja.retry.json` completed all 120 s and native finish; it alone is selected. Missing original input was 43.3 s plus EOF.
- E0 JA, ZH and JA diagnostic; T1 EN diagnostic; T2 EN diagnostic reported `ConnectionClosedError: sent 1000 (OK); no close frame received` **after** source completion, fresh EOF commit and all indexed diarization receipts. Respective receipt counts were 19/19, 17/17, 23/23, 13/13 and 13/13. These remain partial statuses, not clean-shutdown successes; no missing-source or timed-out-drain claim is inferred.
- A source-reference hold and Windows launcher/actual-parent suspension mismatch affected cohort orchestration. Actual schedulers were resumed with progress receipts. S0/EN and E0/EN supervisor exits became unknown after parent timeout although their provider/session evidence completed; neither EN session was repeated. Inter-session orchestration delay is not ASR latency.
- Old diagnostic supervisors were intentionally retired before any T session launched; a fresh bounded supervisor executed only the four unrun T sessions. Its nonzero cohort result retains the close errors.

## Frozen-anchor attribution

Online evidence is limited to receipt by **reference span end + 1 s**. Eventual evidence uses completed retained observations. Each session has one fixed one-to-one native/reference mapping; there is no per-anchor remapping. C/W/U = correct / wrong / unassigned. Coverage = (C+W)/eligible; conditional accuracy = C/(C+W); correct coverage = C/eligible. All denominators include unavailable anchors.

| Arm | Eligible | Online C/W/U | Eventual C/W/U | Coverage online / eventual | Conditional accuracy online / eventual | Correct coverage online / eventual |
|---|---:|---|---|---|---|---|
| N | 120 | 119 / 0 / 1 | 119 / 0 / 1 | 99.17% / 99.17% | 100% / 100% | 99.17% / 99.17% |
| S0 native | 120 | 49 / 2 / 69 | 50 / 3 / 67 | 42.50% / 44.17% | 96.08% / 94.34% | 40.83% / 41.67% |
| S1 native | 120 | 46 / 4 / 70 | 46 / 4 / 70 | 41.67% / 41.67% | 92.00% / 92.00% | 38.33% / 38.33% |
| E0 native | 120 | 86 / 1 / 33 | 98 / 0 / 22 | 72.50% / 81.67% | 98.85% / 100% | 71.67% / 81.67% |
| T1 diagnostic | 40 | 29 / 0 / 11 | 34 / 0 / 6 | 72.50% / 85.00% | 100% / 100% | 72.50% / 85.00% |
| T2 diagnostic | 40 | 29 / 0 / 11 | 34 / 0 / 6 | 72.50% / 85.00% | 100% / 100% | 72.50% / 85.00% |

| Arm | EN: online → eventual | JA: online → eventual | KO: online → eventual | ZH: online → eventual |
|---|---|---|---|---|
| N | 40/0/0 → same | 40/0/0 → same | 20/0/0 → same | 19/0/1 → same |
| S0 | 14/0/26 → same | 20/0/20 → same | 9/0/11 → 10/0/10 | 6/2/12 → 6/3/11 |
| S1 native | 14/0/26 → same | 20/0/20 → same | 7/0/13 → same | 5/4/11 → same |
| E0 | 32/1/7 → 35/0/5 | 31/0/9 → 34/0/6 | 12/0/8 → 15/0/5 | 11/0/9 → 14/0/6 |
| T1 diagnostic | 17/0/3 → 18/0/2 | 12/0/8 → 16/0/4 | Not run | Not run |
| T2 diagnostic | 17/0/3 → 18/0/2 | 12/0/8 → 16/0/4 | Not run | Not run |

For N/S0/S1/E0, EN and JA each have 40 anchors including diagnostics, and KO/ZH each have 20. T1/T2 have 20 EN and 20 JA diagnostic anchors. Assigned online provisional/settled counts: N 0/119, S0 46/5, S1 25/25, E0 33/54. Eventual assignments are settled. Settled means the retained output representation is finalized/immutable, not certainty that a speaker identity is correct.

**Case 1 — timestamp coverage is not ASR recall or a twofold speaker-classifier improvement.** Soniox often supplies short subword timestamps with gaps inside an otherwise recognized word; Scribe supplies a covering word interval. The anchor remains fixed inside that representation gap. For EN anchor 02 at 6.887 s, Soniox recognized `ch` at 6.72–6.78, `arg` at 6.96–7.02 and `er.` at 7.14–7.20; E0's eventual `charger.` spans 6.696–7.076. Similar KO and JA gaps were corroborated. No tolerance, snapping or interpolation was added after observing output.

Corrected unassigned reasons distinguish: no lexical text received; explicit anchor-covering evidence without usable timing; covering timed text without usable identity; and **`no_timed_text_coverage`**, where text exists but no timed unit covers the point and a more local cause is unsupported. An unrelated untimed token cannot establish a missing-timing cause. Eventual S0's 67 and S1's 70 unassigned points are in the last category; E0 has 16 such gaps plus six missing-speaker points. This does not establish that the corresponding speech was absent from ASR.

**Case 2 — source-input omission is not caption postprocessing.** JA anchor 10 at 30.08 s has no completed S1 source-input range covering it: the previous range ends at 29.984 s and the next begins at 30.732 s. S0 assigns the same anchor correctly. This is one observed selection difference, not an estimate of all VAD loss or an established product defect. S1 caption envelopes are a separate application representation, not extra recovered native words.

## Identity timing, changes and caption composition

The following first-appearance measurements use the first annotated span for each session/reference speaker: 17 opportunities across the six recordings, six in the diagnostic pair. First usable means any timed native identity overlapping that span; it need not be correct. Correct-identity delay uses the fixed map. The missing-correct and unanchored columns partition the remaining correct-identity denominator; they are **not** counts of missing first-usable output. These samples include startup and are not steady-state service estimates or visible-caption latency.

| Arm | First usable n / opportunities | First usable median / p95 (s) | Correct identity n | Correct median / p95 (s) | Correct unavailable / reference unanchored |
|---|---|---|---:|---|---|
| N | 16/17 | 0.737 / 1.555 | 12 | 0.724 / 1.059 | 2 / 3 |
| S0 | 15/17 | 1.304 / 2.483 | 9 | 1.424 / 1.789 | 5 / 3 |
| S1 native | 14/17 | 1.364 / 3.595 | 7 | 1.376 / 2.212 | 7 / 3 |
| E0 | 14/17 | 0.995 / 2.675 | 10 | 0.995 / 2.220 | 4 / 3 |
| T1 diagnostic | 5/6 | 1.032 / 2.916 | 3 | 1.032 / 1.041 | 1 / 2 |
| T2 diagnostic | 5/6 | 1.011 / 2.919 | 3 | 1.011 / 1.055 | 1 / 2 |

Slower correct-identity examples, not maxima of every timing measure: N JA diagnostic onset 0.05 s → receipt 1.131 s; S0 JA 0.56 → 2.444 s; S1 EN 0.336 → 2.829 s; E0 ZH 9.35 → 11.817 s. The first three explicitly expose startup; the last is a later new-speaker appearance.

Startup and EOF are separately retained below as per-arm minimum–maximum observations, not subtracted from the identity timings. Source-relative EOF/drain clocks start at the source origin. Startup definitions differ and must not be compared as pure network latency.

| Arm | Recorded pre-source setup interval (s) | Source EOF/completion clock (s) | Drain end clock (s) |
|---|---|---|---|
| N | Model creation 0.116–0.139 | Local completion 120.032–120.037 | No separate drain interval logged; explicit EOF padding retained |
| S0 | Connection-start → source 0.564–0.728 | 120.002–120.016 | 120.249–120.333; native finish in all six |
| S1 | Transport-ready → source 0.056–0.062 | 120.002–120.030 | 121.241–121.570 |
| E0 | Connection-start → source 1.645–1.968 | 120.002–120.012 | 123.018–123.032 |
| T1 | Connection-start → source 0.291–0.327 | 120.009–120.024 | 123.014–123.029 |
| T2 | Connection-start → source 0.280–0.319 | 120.002–120.005 | 123.011–123.018 |

S1 did not record connection-start, so its ready interval is not network/model startup. Scribe drain observed a new EOF commit and all indexed diarization without timeout, but the protocol has no explicit EOF acknowledgement; retained close errors still apply. N model creation precedes source origin. No inferred steady-state or drain-completion latency replaces these actual boundaries.

| Scribe arm | Commit → first indexed timed-word identity, available / commits | Median / p95 (ms) |
|---|---:|---|
| E0 | 126/128 | 60.9 / 277.2 |
| T1 | 25/25 | 54.5 / 187.5 |
| T2 | 26/26 | 56.3 / 247.9 |

The two unavailable E0 commits are empty. This latency establishes one usable indexed identity, not correctness/completeness of every word. Slower E0 cases include ZH segments 7/11 at 321.4/344.4 ms and EN diagnostic segments 7/21 at 303.0/324.2 ms. N has no transcript commit; Soniox does not have the same indexed Scribe commit boundary, so no comparable number is invented.

**Case 3 — a late correction cannot erase an online error or imply free recoloring.** E0 EN anchor 07 at 60.087 s is provisionally `speaker_2`, wrong for MEE014, at its 61.288 s cutoff. The correct `speaker_1` arrives at 67.439 s and changes only the eventual result. Separately, EN segment 16's delayed metadata changes a whole committed source text into two projected source groups; the actual MT case below demonstrates the resulting new translation work.

| Delayed-metadata projection | Identity-only | Regroup/retranslate if original whole unit already translated | Empty/no change |
|---|---:|---:|---:|
| E0 | 77 | 49 | 2 |
| T1 | 10 | 15 | 0 |
| T2 | 11 | 15 | 0 |

This applies current composition rules counterfactually; it is not implemented Scribe late repair. No live `speakers_merged` receipt occurred, so merge behavior has offline contract coverage only.

### Overlap and return probes

There are 134 annotation-derived overlapping reference-pair intervals. Both/one/neither means mapped identities observed within the interval, not proof that simultaneous lexical content was recovered. Pair intervals can overlap each other; these are not independent trials or a duration-weighted overlap error rate.

| Arm | Online both / one / neither | Eventual both / one / neither |
|---|---|---|
| N | 90 / 34 / 10 | 90 / 34 / 10 |
| S0 | 26 / 61 / 47 | 32 / 70 / 32 |
| S1 native | 24 / 71 / 39 | 26 / 77 / 31 |
| E0 | 35 / 63 / 36 | 39 / 67 / 28 |

The frozen 24 source-edge opportunities are 14 speaker-change and ten overlap-onset probes. Exact interval-edge selection can pick the same adjacent envelope for two reference sides, so these counts are not adjudicated missed/false floor changes. Seven reference pairs involve unanchored speakers.

| Arm | Edge pair available online → eventual, of 24 | Same-native-ID edge probes online → eventual | Correct identity-pair probes online → eventual |
|---|---|---|---|
| N | 18 → 18 | 10 → 10 | 4 → 4 |
| S0 | 19 → 21 | 9 → 11 | 3 → 3 |
| S1 native | 19 → 19 | 7 → 8 | 4 → 4 |
| E0 | 17 → 17 | 8 → 9 | 4 → 5 |

The corrected receive-ordered edge selector prefers later indexed/settled metadata at the same exact endpoint, retains simultaneous conflicts as unavailable and respects the cutoff. The original score's zero eventual E0 edge availability was a measurement defect, not provider absence.

Exploratory A→B→A triples were derived from the unchanged frozen anchor order, not newly chosen provider-success points. All three identities are correct with a consistent anonymous return in N 22/22 → 22/22, S0 3/22 → 3/22, S1 5/22 → 5/22 and E0 7/22 → 9/22, online → eventual; every other triple is unavailable. Triples overlap and are not 22 new recording trials. Anchor-level fragmentation probes found one S0 ZH reference using two native IDs online and two eventually; S1 had two such ZH references in each view. E0's one EN provisional fragmentation resolved eventually. No independent same-speaker floor-opportunity denominator was frozen, so a false-change rate is **unavailable**, not zero.

### S1 caption envelopes and corroboration

| S1 application-envelope view | C/W/U, of 120 | Coverage | Conditional accuracy | Correct coverage |
|---|---|---:|---:|---:|
| Online | 54 / 6 / 60 | 50% | 90% | 45% |
| Eventual | 96 / 12 / 12 | 90% | 88.89% | 80% |

Broad source-run envelopes fill native timestamp gaps by representation; increased caption-envelope coverage is not recovered ASR or directly comparable native attribution precision. The application replay uses production owners, unknown/mixed handling, grouping and context, with deterministic immediate MT only.

**Case 4 — an apparent harmful join is an envelope false alarm.** At JA diagnostic change 00, actual source groups/requests 0 and 1 have distinct caption IDs and native speakers 1 and 2. The next group's envelope begins at 4.14 s while the first reference ends at 4.19 s; the edge probe therefore selects the next group on both sides. Checking all six flagged candidates against actual source groups, requests and caption IDs yields **zero confirmed harms, four false alarms and two unresolved**. The unresolved JA change 11 lacks the overlapping `うん`; EN diagnostic change 20 has punctuation-only/missing `Okay` evidence. This is not proof of universal zero composition errors and does not justify a Soniox fix.

## Matched diagnostic turn policies

The 24 opportunities comprise nine endings, five intermediate hypotheses and ten unsafe-cut points. T0 is the existing E0 diagnostic run, not another paid pass. Content support for ending latency requires the complete normalized reference ending in the **original committed text available at that receipt**. Later corrections cannot backdate availability. Strict lexical mismatch, annotation uncertainty and zero-width gaps stay unavailable; these are not deletion/duplication verdicts.

| Measure | T0: current controller | T1: 1.5 s VAD | T2: prediction 0.9 + same VAD |
|---|---:|---:|---:|
| Native commit receipts | 49 | 25 | 26 |
| Nonempty source fragments / potential MT triggers | 48 | 25 | 26 |
| Empty commits | 1 | 0 | 0 |
| Premature cuts at unsafe points | 7/10 | 0/10 | 0/10 |
| Merged ending pairs | 3/9 | 7/9 | 6/9 |
| Eligible ending-to-commit cases | 2/9 | 3/9 | 3/9 |
| Ending-to-commit median / p95, eligible only (s) | 0.550 / 0.567 | 9.509 / 9.827 | 9.509 / 9.810 |
| Useful intermediate output after cap/EOF exclusions | 1 | 0 | 0 |
| Current-controller commands | 47 | 0 | 0 |
| Client cap commands | 0 as separate cap-command type | 22 | 22 |
| EOF commands | 2 | 2 | 2 |

The controller itself still has its normal 10 s safety cap; T0's command type does not mean that safety behavior was disabled. The latency rows have different small eligible populations and are not a matched paired effect estimate. The only eligible positive-width useful intermediate case belongs to T0; four EN points are offset-only and cap-associated T outcomes cannot be credited as prediction success. Do not turn `0` into a claim of failure on five evaluable opportunities.

| Diagnostic policy | EN / JA partial events | EN / JA maximum turn probability | EN / JA events ≥0.9 |
|---|---|---|---|
| T0 | 724 / 824 | 0.84 / 0.89 | 0 / 0 |
| T1 | 729 / 809 | 0.87 / 0.91 | 0 / 2 |
| T2 | 736 / 826 | 0.87 / 0.92 | 0 / 2 |

All partial events carried numeric probabilities. High probability does not identify the cause of a commit. Each T recording sent 11 cap commands plus EOF and no 4/5 s controller commands. The observed behavior is cap-dominated: fewer cuts inside the marked unsafe intervals also merge more ending pairs and yield long eligible commit delays. T2 did not demonstrate an actionable incremental benefit over T1 under these requested settings and echo limitations.

**Case 5 — avoiding a marked unsafe gap does not preserve the whole phrase.** At EN diagnostic 59.36–60.88 s (`infrared` → `um sender`), T0 cuts inside the marked opportunity. T1/T2 do not, but their source boundary at 61.204 s still ends a committed segment with `And the infrared, um`; the next begins `The sender would be...`. T1 receives those commits at 62.508/72.606 s, T2 at 62.543/72.657 s. The modifier/head-noun split remains outside the scored interval. Thus `0/10` is not a whole-phrase preservation score or a global absence of unsafe semantic cuts.

| Strictly supported ending case | T0 delay (s) | T1 delay (s) | T2 delay (s) |
|---|---:|---:|---:|
| EN diagnostic turn 00 | 0.569 | 1.755 | 1.740 |
| EN diagnostic turn 04 | Unavailable | 9.509 | 9.509 |
| JA diagnostic turn 06 | 0.531 | 9.862 | 9.843 |

The remaining ending opportunities lack strict original-text support, not necessarily their semantic content. In particular, normalization mismatch from disfluency, annotation markup or an inserted article must not be called ASR deletion. No automatic missing/duplicated-content total is asserted from substring mismatch.

| Zero-width EN intermediate point (s) | T0 nearest source-boundary offset (s) | T1 offset (s) | T2 offset (s) |
|---|---:|---:|---:|
| 25.76 | −0.868 | −3.980 | −3.980 |
| 32.74 | −0.444 | −0.584 | −0.584 |
| 45.62 | +0.748 | −3.448 | −3.448 |
| 57.57 | +0.650 | −5.602 | −5.482 |

These offsets are descriptive only; no binary useful-output verdict or tolerance was invented for the zero-width points.

## Actual MT and actual Desktop pixels

Seven frozen intervals were conditionally selected from the completed evidence, then held fixed. Selection SHA-256: `a252da053f364791f799500f39054448b3f8f196d35b6c086a2284b548b89710`. Source hashes, native event/segment indexes and captured S1 request indexes are retained. This is not a live Scribe production integration or an ASR→MT→GUI latency measurement.

Production `OpenRouterLLMProvider -> LlmTranslationBackend` used the declared shared experiment profile: only `google/gemma-4-26b-a4b-it`, LATENCY routing, DEFAULT provider routing, temperature 0.6, reasoning none, 100 max output tokens and 30 s timeout. No selected request overrode the output limit. All responses reported model `google/gemma-4-26b-a4b-it`, provider `NextBit`. This is explicit experiment configuration, not inspected private user settings.

S1 reused complete captured rendered prompts, source text, language pairs, contexts and participant counts. E0 used the current production-rendered default prompt, empty context and Korean target except KO→English. S1 context contains prior source turns. These differing inputs/contexts prevent a matched-context S1/E0 MT-quality ranking. Within E0, whole versus independently translated source partitions exposes the proposed regrouping consequence; translated target strings were never sliced by source offsets.

| Case / E0 source interval | S1 captured requests | E0 whole + additional requests | Metadata after commit | Observed content/presentation |
|---|---:|---:|---:|---|
| EN 0.440–1.880 | 1 | 1 + 0 | 32.8 ms | Startup `Okay, we're good to go. Okay.` readable gray, then separately settled yellow using the same MT. Identity-only, no extra request. |
| EN 74.588–78.488 | 2 | 1 + 2 | 208.7 ms | Mixed question/answer stays gray; independently translated source groups form cyan/yellow cards. Regrouping changes wording and consumes work. |
| JA 76.812–82.013 | 3 | 1 + 2 | 24.6 ms | Negation `ないかも` → `없을지도`; backchannel `あー。` → `아ー.` retained in separate visible cards. |
| KO 86.380–91.880 | 2 | 1 + 2 | 48.9 ms | Question/answer and four-year younger-sibling detail remain in English with Korean source rows. Interviewer attribution retains the anchor-coverage limit. |
| ZH 23.672–28.793 | 6 | 1 + 5 | 35.6 ms | Weekend, `Uh/no` correction, weekday, Friday/Saturday remain visible. Five new translations, not free recoloring. No lexical corpus oracle is claimed. |
| EN diagnostic 37.908–42.188 | 2 | 1 + 0 | 173.2 ms | Two PCBs/hinge text readable; identity-only reuses MT. S1 repair adds cabling text outside this E0 unit, so the texts are not equivalent. |
| JA diagnostic 52.644–62.585 | 9 | 1 + 6 | 122.8 ms | Whole `楽しみ方` conveys an enjoyment method; isolated `楽しみ` becomes `기대돼`, and `なん` becomes `뭐`. Actual colored fragments damage composition/meaning. Long whole-source row is visibly ellipsized. |

These are 49 valid selected MT calls: 25 S1, seven E0 whole commits and 17 extra E0 source-unit requests. The 17 are the cost of the measured translate-first/regroup-later projection, not an unavoidable cost of every possible integration; a metadata-waiting design has a different latency tradeoff. Native fragmentation/color is not speaker ground truth.

The first E0 attempt accidentally passed an unbound `${targetName}` prompt template. Its 24 successful responses and 26 E0 screenshots are preserved as **invalid configuration evidence**, not MT quality. After an authorized preparation repair using production prompt-rendering APIs, 24 positive effective-body checks and two negative literal/wrong-target assertions passed before exactly 24 corrected E0 calls. All 25 valid S1 responses/pixels were reused. No quality retry, request-body cache hit or HTTP/backend failure occurred. Total actual MT calls: **73**. Provider-reported valid/invalid-extra/combined cost was USD **0.00348193 / 0.00135943 / 0.00484136**, not an independently reconciled invoice. Valid backend elapsed min/median/max was **0.389 / 0.683 / 1.944 s**, separate from ASR and GUI timing.

The actual production `OverlayPresenter -> DesktopOverlayRenderer -> FletDesktopRendererWindow` ran under the intended shared Python 3.14.10 with Flet/flet-desktop 1.0.5 and Pillow 13.0.0. Fresh authenticated loopback ingress, in-memory launch tokens and owned native windows were used. A neutral owned backdrop covered the complete 1344×336 capture region; PIL captured actual screen pixels, not a renderer projection or whole private desktop.

There are 51 valid captures: 25 reused S1 and 26 corrected E0. The worker visually inspected 26 individual valid PNGs across all seven cases; the Director additionally inspected the whole Japanese caption and the `楽しみ` fragment. Target/source rows, native palette colors, rounded cards and visible inter-card gaps were observed; no obvious target glyph cutoff was seen in inspected states. This is not a claim that every long source row is fully visible. The original E0 captions remained gray; colored E0 units were **new experimental presentations using settled native metadata**, not supported same-utterance Scribe late repair. All 26 corrected revisions also had separate renderer commit/ack receipts.

Initial ingress-close-first cleanup lost two shutdown lifecycle deliveries. That owned adapter failure is preserved; corrected shutdown awaited the renderer while ingress remained open and had zero delivery failures. Both owned Flet children exited zero with exact-process handle evidence; backdrops were destroyed, backend/presenter/renderer/ingress closed, no unrelated process killed and no launch-token files retained. Credentials were read only in memory and not printed or saved; the GUI phase did not look them up or inherit credential-named environment variables.

## Reproducibility and verification

Task-local entry points: [input preparation](issue441_inputs.py), [Nemotron runner](issue441_nemotron.py), [cloud runner](issue441_cloud.py), [scorer](issue441_score.py), [MT/display adapter](issue441_mt_display.py). They do not add product providers or a general benchmark platform. Runners importing production code select the intended checkout through `--root`; the scorer uses explicit `--inputs`, `--runs` and `--output`. Provider reruns consume paid calls and are not needed to inspect this result.

Selected evidence is pinned to 28 completed/partial record artifacts: six each N/S0/S1/E0 and two each T1/T2. The explicit selection index uses the completed S0 JA repeat, never silently overwrites or auto-selects the failed attempt. Raw events, process journals, model probabilities, receive timestamps, command attempts/completions, original/updated words and input-to-source mappings remain local under `.tmp/issue441/` without publishing original audio or secrets.

The original score is retained at `scores/score.json`; the reviewed reason/probe correction produced **one** new score, `scores/score-reasons-repaired.json`, in 29.65 s. An explicit invariance proof checked 1,120 native anchor-views and 240 caption-envelope anchor-views: per-anchor C/W/U, eligible denominators, fixed session mappings, reference/times/cutoffs, assigned IDs/evidence/settled labels, per-language rates, caption targets and all 28 selected log hashes/statuses were unchanged. Only supported reason classifications and qualified change/envelope observations changed. The original score's missing-cause and harmful-join labels are superseded, not silently rewritten.

Original/corrected score SHA-256: `07cc4333869bee13b2265d6d95902b112ae16d6d36392d807a5452ebd5c07c1f` / `5ec5f84b42eb23c65be277defb8648c1173126679ee8433fdb1df5e03731b78c`.

Local evidence index, relative to `.tmp/issue441/`:

- `selected-runs/selection-index.json`: exact selected artifact paths and hashes.
- `runs/S0/cloud-execution-report.json`: effective/requested configuration, source/call accounting and all preserved cloud outcomes.
- `nemotron-runtime/restoration-evidence.json`, `nemotron-runtime/execution-evidence.json`, `nemotron-runtime/host-orchestration-evidence.json`: runtime/device/build/profiling and concurrency qualification.
- `scores/final-summary.json`, `scores/score-invariance.json`, `scores/startup-eof-by-recording.json`: complete tables, unchanged-quality proof and individual startup/EOF records.
- `scores/representative-cases.json`, `scores/native-timing-gap-diagnosis.json`, `scores/envelope-probe-corroboration.json`, `scores/anchor-return-probes.json`: the five diagnostic cases and probe corroboration.
- `mt-display/result.json`, `mt-display/summary.md`, `mt-display/source-equivalence.json`, `mt-display/invalid-template/`: effective MT bodies/receipts, valid/invalid pixels, exact owned cleanup and execution-to-final applicability.

Offline scoring can be reproduced without paid calls from the retained selection. Use a new output path to preserve both recorded scores:

```text
../../../.venv/Scripts/python.exe benchmarks/asr/issue441_score.py --inputs benchmarks/asr/issue441-inputs.json --runs .tmp/issue441/selected-runs --output .tmp/issue441/scores/replay.json
```

Checks performed:

- Actual N silence smoke and six source sessions; all selected cloud conditions and preserved failures described above.
- Eleven isolated controller/cadence/source-ledger checks passed, including actual VAD/SmartTurn/controller branches with the WebSocket replaced for the offline boundary check. These are not extra provider-quality sessions.
- Focused scorer suite: **59 passed** using `../../../.venv/Scripts/python.exe -m pytest tests/benchmarks/test_issue441_score.py -o addopts= -q`. It retains the earlier 53 cases and adds local-gap/receive-ordered change-probe behavior. Tests cover late metadata, cap/EOF races and receipt-time content availability rather than source-text wiring.
- Scoped Ruff and Black checks passed for the experiment Python files and scorer tests after owned style repairs; formatting applicability was separately checked against executed behavior. No broad product suite, build or type-check claim is made.
- Actual MT preparation/body guards, calls, real Desktop pixels and exact owned cleanup, as described above. No normal product startup, private capture or live production configuration mutation was needed.

Independent checkpoint reviews rejected and then verified repairs for mismatched T send cadence, late-metadata regroup classification, cap/EOF race credit and backdated ending content. A later bounded review accepted the local-gap reason and receive-ordered edge/probe correction before the single corrected score. Failures and unavailable observations remain in this report rather than being converted to clean successes.

Checkpoint applicability: initial harness findings were repaired through `c2b17d25852091f88aa310f4a60478590868c7fb`; AST-preserving formatting was reviewed at `af28b95c5ff25541793fd7338c126e35f0130764`; the measurement correction was reviewed at `97941ec220c22a862c00a2d1743660965101075c`; actual MT/display code and evidence were independently reviewed at `57fce7b607746a3deebb65ff8b3bf25d7416dece`, with no remaining material code findings. The last review independently checked all 49 valid HTTP receipts, 51 capture hashes, source-unit correspondence and four actual PNGs. Corrected MT/display/final-adapter full-module ASTs were equal; the initial invalid adapter was not.

**User-visible product changes: none.** The deliverables are reproducible experiment adapters, frozen non-audio references, regression coverage and this decision record. No drift in the protected product documentation was caused by this experiment.
