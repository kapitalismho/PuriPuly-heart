# #212 timed-cloud clarification: real native-wire observations

Revision **TIMED-CLOUD-WIRE-1**, 2026-10-06. Task-local report under [#212](https://github.com/kapitalismho/PuriPuly-heart/issues/212) and [#211 / N3-SPEAKER-1](https://github.com/kapitalismho/PuriPuly-heart/issues/211). Baseline supplied by Director: `0f0ba5173da51250bb5926e4a2eead5e68adff81`. No product enablement, profile selection, maintained documentation, product code, tests, settings, shared helper or shared WAV changes. No billing metering or pricing/account requests.

## Finite plan persisted before real calls

Based on research record **TIMED-3PROVIDER-PROBE-1**. `documented` means official contract, `observed` a bounded actual response, `counterexample` a bounded falsification, `inconclusive` insufficient discrimination, `not-run` an unexecuted branch. Native protocol responses are not observations of unchanged packaged adapters. All calls use locked `websockets==16.1.1` directly to preserve full upstream fields; installed Deepgram `5.3.4` and ElevenLabs `2.65.0` SDKs are not invoked. Native Scribe payloads follow the installed SDK's send/commit implementation.

No call is spent proving settled guarantees. [Soniox](https://soniox.com/docs/stt/rt/real-time-transcription.mdx) documents immutable once-only final tokens; [Deepgram](https://developers.deepgram.com/docs/finalize.md) explicitly does not guarantee empty/little-buffer Finalize responses; [ElevenLabs](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/event-reference) documents replacing partials, final committed text and timestamp metadata following plain committed text. These are **documented**, not inferred from successful traces.

Each provider initially gets at most two connections, sequential within provider, 100 ms real-time-paced chunks, 45 s wall deadline per session, zero retries. Normal session: three exact 3 s audio slots, each 500 ms explicit synthetic silence + local SAPI phrase + silence through slot end. Phrase A is repeated byte-identically twice, then distinct B. Finalize/commit at cumulative 3 and 6 s, only continue after the provider's observed completion. Third slot ends Soniox via empty TEXT frame, Deepgram via CloseStream, Scribe via commit then close. Paused completion observation <=4 s; bounded absence never means universal no response.

Short session: 100 ms explicit silence + energy-trimmed local `No.` (329.625 ms) + 200 ms silence = **629.625 ms / 10074 samples**. Threshold `abs(PCM16)>=200`, first-to-last active samples with 20 ms context either side; no shared file altered. Active speech itself is 289.625 ms. Energy bounds are approximate waveform activity, not word/forced alignment. If completed, Deepgram/Scribe send one empty control at least 3 s later. If Scribe does not complete, add only synthetic diagnostic silence to cumulative 2.4 s WITHOUT another commit, observe 3 s, then at most one further commit. These diagnostic silences are explicitly no-source, not approved product padding/ownership or later-speaker input. Stop provider branch on error/quota/auth/throttle; no automatic retry.

Soniox short session additionally permits ONE no-added-postspeech-silence finalize: after >=3 s control spacing send A/B trimmed to threshold-active bounds back-to-back, finalize at actual sent cursor, immediately send trimmed `No, three.` as successor, then stream-end drain. This deliberately contrasts recommended post-speech silence; no cadence sweep or quality guarantee. Threshold trimming can cut quiet edges, so any missing recognition is not necessarily a finalize defect.

| ID | Alternatives and discriminating input | Decision consequence | Pre-call status |
| --- | --- | --- | --- |
| S-O1 | Stream-global vs finalize-relative token/progress ms; repeated A slots differ by exactly 3 s | Cumulative sent map vs unresolved/reset origin | not-run |
| S-O2 | Repeated A produces distinct final receipts and separate completion markers vs loss/replay/conflict | Preserve true repeats; no text/time dedup | not-run |
| S-O3 | Final progress reaches sent silence frontier vs lexical end/omitted progress | Preserve coverage separately from lexical end | not-run |
| S-O4 | Pending B tail before finished/CLOSE vs missing/censored drain | Preserve stream-end tail and retirement ordering | not-run |
| S-O5 | 629.625 ms fresh `No.` recognized/finalized vs empty/no completion/error | No universal 1 s source floor; don't discard short speech | not-run |
| S-O6 | One no-added-postsilence finalize plus immediate successor preserves A/B/tail vs loss/crosscursor ambiguity | Passive timed join remains preferred; no rapid-cadence certification | not-run |
| D-O1 | Actual Nova-3 Results expose words/start/duration/channel/model vs omitted data | Preserve native evidence; exact original transcript, no guessed joins | not-run |
| D-O2 | Stream-global vs finalize-relative seconds; repeated A + B slots | Cumulative input map vs unresolved origin | not-run |
| D-O3 | Distinct A finals and observed from_finalize vs no ACK/repetition loss | Reuse only with completion evidence; retain absent-ACK retirement fallback | not-run |
| D-O4 | B final tail / Metadata / CLOSE drain vs no tail/censored close | Irreversible CloseStream retirement | not-run |
| D-O5 | Fresh short No produces final/ACK/CloseStream tail vs empty/failure | Preserve short source independently of control result | not-run |
| D-O6 | Empty control gives result vs no result within 4 s | Either outcome leaves documented no-ACK guarantee unchanged | not-run |
| E-O1 | Global vs commit-relative vs speech-relative word seconds; 500 ms slot leads and 3 s repeated-A offset | Known origin required before mapped timed join | not-run |
| E-O2 | Ci/Ti adjacency/order vs interleaving/missing metadata under continuation immediately after Ci; identical A text cannot pick owner | Pair only unambiguous commit generation; missing/ambiguous metadata stays gray | not-run |
| E-O3 | Exact words/spacing text equals plain/timed text vs mismatch/null/missing timing | Exact text offsets only; no proportional reconstruction | not-run |
| E-O4 | Two genuine A commits and B continuation vs carryover/loss/unsolicited commit | Distinct final contributions without pre-roll replay | not-run |
| E-O5 | Mandatory >=1 s commit gate vs completed <1 s input | Successful short commit falsifies gate for this sample/config only | not-run |
| E-O6 | Sub2s commit completes immediately vs completes after silence with no second control vs only second control/never | Diagnostic startup behavior; no product manufactured source extension | not-run |
| E-O7 | Empty committed text/metadata vs error/no response/missing text | Explicit empty is not absence; late prior metadata retains old owner | not-run |
| X-O1 | Per-result stable identity vs connection-only/no ID; passive response inspection | Local receipt identity does not prove vendor replay dedup | not-run |
| X-U1 | Universal duplication/reordering/metadata-adjacency/timestamp monotonicity | Cannot prove with finite traces; retain official guarantee limits | inconclusive by design |
| X-U2 | Universal safe cadence/short accuracy/latency/max lifetime/account cost | No stress/lifetime/pricing/billing experiments; remain unknown unless documented | not-run |
| X-OFF1 | Product mapping/ledger/overlap/Nemotron-only identity invariants | Prior fixtures remain separate evidence; not a live server proposition | intentionally unchanged |

## Input/environment reproducibility

Preparation artifact `timed-cloud-results-preparation.json` records exact WAV/PCM/trim hashes, samples/durations, waveform activity bounds, configured default route/model checks, source SHA-256 and Python/platform/package versions. Configured routes matched matrix defaults; no full settings dump. Audio is generated locally by Director via Microsoft Zira Desktop SAPI, not microphone/private audio or external TTS. A: `The red boat is ready.` 1.945 s; B: `The blue train is late.` 2.025 s; No: 1.135 s including silence; correction: `No, three.` 1.92 s including silence. No audio file is authored or altered by this worker.

Commands run from repository root with the locked parent virtual environment (this block initially records preparation only; execution observations added below):

```powershell
../../../.venv/Scripts/python.exe docs/experiments/issue-212-clarification/timed_cloud_probe.py --prepare --output docs/experiments/issue-212-clarification/timed-cloud-results-preparation.json
```

Preparation exited 0 and performed no connections. Responses persist full sanitized JSON fields, receive ordinals/elapsed monotonic time, and sent-sample/control cursors. Input chunks record sample count/SHA-256, not base64 audio. Session/request/event/item/connection IDs receive stable per-session aliases; private account/project/user/organization fields and fetched secrets are redacted; models/model UUIDs are retained. No headers or credential-bearing URLs are persisted.

## Actual outcomes

### Executed bounded traces

The real calls below were executed; these are not constructed parser fixtures. All commands exited 0, which means the harness persisted outcomes, **not** that every provider branch succeeded. Two connections/provider were used initially. Soniox sent 12.98825 s, Deepgram 9.629625 s, ElevenLabs 9.629625 s. No auth/quota/model error, retry, alternate endpoint, SDK installation or billing instrumentation occurred.

```powershell
../../../.venv/Scripts/python.exe docs/experiments/issue-212-clarification/timed_cloud_probe.py --provider soniox --output docs/experiments/issue-212-clarification/timed-cloud-results-soniox.json
../../../.venv/Scripts/python.exe docs/experiments/issue-212-clarification/timed_cloud_probe.py --provider deepgram --output docs/experiments/issue-212-clarification/timed-cloud-results-deepgram.json
../../../.venv/Scripts/python.exe docs/experiments/issue-212-clarification/timed_cloud_probe.py --provider elevenlabs --output docs/experiments/issue-212-clarification/timed-cloud-results-elevenlabs.json
../../../.venv/Scripts/python.exe docs/experiments/issue-212-clarification/timed_cloud_probe.py --provider elevenlabs --sessions short --output docs/experiments/issue-212-clarification/timed-cloud-results-elevenlabs-short.json
```

Tool wall times: Soniox 21.04 s; Deepgram 19.78 s; ElevenLabs normal 12.92 s; ElevenLabs short 4.67 s. Initial provider invocations ran independently in parallel, never multiple simultaneous connections to one provider.

**Harness history, retained rather than overwritten:** ElevenLabs normal supplied all three plain/timed pairs, then client `close()` produced `ConnectionClosedError: sent 1000 (OK); no close frame received` and local observed code 1006. Initial harness classified this requested-close exception as a provider branch error and therefore skipped the already-planned short session. Corrected only the harness to distinguish client-requested teardown from a runtime/provider error, retaining the receive-error event verbatim. Ran the planned short session separately, not normal again. Original traces record script SHA-256 `9833a0ca16ba2bff7af4a94c4b0d2e6601164ab077cc68b4a6c7447ba8dd9d89`; short trace records the corrected script hash. Original normal JSON's `provider_branch_stopped` is this harness stopping condition, not auth/quota failure. The close handshake remains an observed abnormal teardown, not silently labeled clean. The short session ended on a genuine server `commit_throttled`, immediately stopping that provider. No response was fabricated or replaced.

### Soniox

Native diarization and language identification were disabled; independent token timings/progress still arrived. Requested `stt-rt-v5`; no returned deployed model/build ID in observed responses.

| Trace / event ordinals | Exact text / control / timing | Interpretation |
| --- | --- | --- |
| normal `44` | Cursor 48000 (3 s), finalize at elapsed 3.776290 s; final `The red boat is ready.` then final `<fin>` at 3.996870 s; lexical range 480–1740 ms; final/total progress **3120 ms** | Final text and marker observed; progress is 120 ms beyond sent PCM, not identity-mapped source coverage |
| normal `86` | Cursor 96000 (6 s), finalize 7.024634 s; final ` The red boat is ready.` then `<fin>` at 7.240745 s; lexical range 3600–4800 ms; progress **6240 ms** | True repeat retained with stream-scale times, not a reset to zero; progress 240 ms beyond sent PCM |
| normal `129–131` | Empty TEXT end at 144000 (9 s), elapsed 10.268838 s; final ` The blue train is late.` at 10.482882 s, lexical 6660–7980 ms; progress **9240 ms**; `finished:true` next receipt at 10.482924 s; CLOSE 1000 at 11.695887 s | Tail before finished/CLOSE observed; exact complete text `The red boat is ready. The red boat is ready. The blue train is late.` |
| short `12` | Finalize at 10074 (629.625 ms), elapsed 1.210308 s; `No.` plus `<fin>` at 1.383264 s; token 180–240 ms; progress **720 ms** | Subsecond input recognized; final progress 90.375 ms beyond actual PCM |
| short `52` | Single no-added-postsilence finalize at 45943 (2871.4375 ms), elapsed 6.482884 s; successor audio sent immediately; at receipt 6.658533 s cursor already 49143; final ` The red boat is ready. The blue train is late.<fin>`; progress **3000 ms** | Pre-control A/B text and marker preserved even while new PCM arrives; marker must bind control cursor 45943, not receipt cursor 49143 |
| short `66–68` | End at 63812 (3988.25 ms), elapsed 7.601019 s; final tail ` No. 3.` at 7.777203 s (3060–3720 ms, punctuation point 3720); progress **4200 ms**; finished next, CLOSE 1000 | Successor correction conserved as actual output `No. The red boat is ready. The blue train is late. No. 3.`; no claim of verbatim `No, three.` quality |

**Counterexample to a naive identity sample map:** processed ms exceed supplied audio, accumulating across controls. The two repeated-A token starts differ by 3120 ms, not exact 3000 ms. Progress excess alone does not prove lexical clock inflation; the subsequently approved matched passive discriminant below compares both. These traces do not establish a universal padding/rounding formula, nor authorize subtracting an assumed 120 ms per finalize. The stream-scale origin does not reset, but **a direct ms→sent-sample projection across finalize is not qualified**. Keep out-of-map ranges unknown and retain raw progress/control history; `<fin>` proves its serialized control completed, not that returned progress is a capture frontier.

No final token replay or duplicate marker appeared in these bounded traces; the official once-only final-token contract, not absence in this sample, remains the authority. Control/tail timing arrived with punctuation/subword granularity and zero-length marker/punctuation intervals. No native speaker metadata participated in any decision.

### Deepgram

Requested `nova-3`; actual returned model: `general-nova-3`, version **2025-04-17.21547**, arch `nova-3`, model UUID **40bd3654-e622-47c4-a111-63a61b23bfe8**. Raw wire retained words and metadata; this is not proof the current SDK adapter preserves them.

| Trace / event ordinals | Exact observation | Interpretation |
| --- | --- | --- |
| normal `34` | Finalize cursor 48000 at elapsed 3.854475 s; Results 4.113831 s, `start=0`, `duration=3`, `is_final=true`, `from_finalize=true`, transcript `The red boat is ready.`; words from 0.39999998 to 1.68 s | Nonempty flush/word evidence observed |
| normal `68` | Finalize cursor 96000 at 7.132393 s; Results 7.384560 s, `start=3`, `duration=3`, same genuine transcript, `is_final=true`, `from_finalize=true`; words 3.4–4.68 s | Word/result clock cumulative across finalize; second real repeat is a distinct source interval |
| normal `102–104` | CloseStream cursor 144000 at 10.409079 s; Results at 10.652428 s, `start=6`, `duration=3`, `is_final=true`, `from_finalize=false`, `The blue train is late.`, words 6.4–7.76 s; Metadata 10.652506 s; CLOSE1000 10.656356 s | Final B tail precedes summary and connection termination. Transport is not reusable after this control |
| short `10` | Finalize cursor 10074 at 1.521902 s; Results 1.824231 s, `start=0`, `duration=.629625`, `No.`, word 0–.39999998 s, final/from_finalize true | Short input accepted/recognized; no universal 1 s source floor |
| short `12–16` | Empty Finalize at same cursor elapsed 4.530885 s; **no Results during 4 s bounded observation**; CloseStream8.532677 s; Metadata8.725643 s; CLOSE1000 8.732874 s | Bounded no-ACK observed, consistent with documented nonguarantee. Quiet socket is not authoritative empty or reusable successor ownership |

All observed Results were `is_final=true`; `speech_final` was absent, not required. The connection request ID remained the same for all its results, so it is not a result delivery identity. Native punctuated words can reconstruct these transcripts with explicit spaces for this sample; preserve original exact transcript and offsets rather than generalizing that reconstruction to all outputs. In particular, concatenating the two identical transcript strings verbatim without a text-boundary policy yields `ready.The`, not an automatically inserted space. No transport-level whitespace is silently invented in this report.

### ElevenLabs

Requested and `session_started.config` echoed `scribe_v2_realtime`, PCM16000, manual commit, English, `include_timestamps=true`, language detection false, background filter false, no_verbatim false, entity detection null. Server returned no deployed model revision. Session ID is connection-scoped; plain/timed events had **no event/segment/commit ID**.

| Trace / event ordinals | Exact observation | Interpretation |
| --- | --- | --- |
| normal `38 / 41` | Commit cursor48000 at elapsed3.513224 s; plain `The red boat is ready.`3.791134 s; timed3.806365 s after successor audio already advanced cursor49600; words/spacing .64–1.68 s | Actual committed event names, late metadata ownership; concatenate `words.text` exactly equals committed text |
| normal `76 / 79` | Commit cursor96000 at6.822056 s; same genuine plain text7.086510 s; timed7.101053 s at cursor97600; words/spacing3.64–4.68 s | Every matching A word/spacing time shifts **exactly3.0 s**, ruling out commit-relative/speech-relative resets for this trace |
| normal `114 / 116` | Commit cursor144000 at10.114212 s; `The blue train is late.` plain10.339489 s; timed10.353409 s; words/spacing6.64–7.74 s | Third origin confirms stream-global clock; exact word+spacing text coverage |
| short `11 / 13` | Commit10074 at.923103 s; plain `No.`1.132716 s; timed1.149935 s, word .2–.36 s, punctuation character point .36 | **629.625 ms fresh input finalized without additional audio**; counterexample to mandatory1 s or2 s manual-commit startup gate for this configuration/sample |
| short `14 / 15 / 19` | Empty commit same cursor3.932326 s;4.110288 s actual `commit_throttled`: `Commit request ignored: only 0.00s of uncommitted audio. You need at least 0.3s of uncommitted audio before committing.`; CLOSE1000 reason `commit_throttled`4.289569 s | Explicit empty-input rejection and server-stated **0.3 s uncommitted-audio** threshold; not merely a wall-clock commit-spacing requirement. No empty committed event/metadata was produced before error/close |

Plain/timed receive order was C1,T1,C2,T2,C3,T3. Outgoing successor audio was deliberately allowed immediately after each plain C, so T1/T2 arrived under an advanced sent cursor while still enriching the old commit. No text-equality pairing is warranted: first two C texts are identical genuine speech. The trace has unambiguous single-pending metadata observations; it does not prove universal adjacency/FIFO, absence of vendor replay, or safe pairing when metadata is missing/interleaved. Word/spacing data and additionally present character times were retained raw, including null speaker/channel/language fields. **Missing/null words or genuinely reordered timestamp metadata were not observed**; they remain fallback cases, not tested server behaviors.

The conditional 2.4 s startup-extension branch was **not run** because short C/T already arrived. Do not manufacture that branch just to fulfill a checklist: the successful short commit already discriminates its decision-relevant startup alternative. The server-stated 0.3 s minimum was not threshold-swept; acceptance at precisely .3 s or with .1–.3 s speech-only buffers remains unqualified. This is an observed server error contract for the selected route/date, not an official universal SLA.

**Director's selected enhanced intake (policy only, not implemented):** use actual `committed_transcript_with_timestamps` (**CWT**) as the sole normal final-text + word contribution. Its required own `text` is authoritative even with optional/null `words`; missing usable words means that CWT text stays gray. Plain `committed_transcript` C is non-admitting completion/fallback evidence, not a separately published source requiring later string matching. This is supported by the [official committed event schema/semantics](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/event-reference), and the present trace establishes stream-global CWT times for the selected route/configuration. Cross-event C/T text pairing is therefore **not an inherent normal-admission gate**. Remaining gates: mapped CWT coordinates, unambiguous source/control generation, exactly one winner between normal CWT and any plain fallback, and late CWT never re-admitting a fallback-owned contribution or leaking into a successor. Missing CWT permits plain fallback only with proven owner; unresolved provenance is not merely unknown speaker color. No fixed speaker wait/input pause is selected. These rules can support a conditional T implementation without promising universal timestamp delivery/order/replay or claiming this reducer exists in the packaged adapter.

### Proposition disposition and policy proposals

| IDs | Final label / result | Director action |
| --- | --- | --- |
| S-O1 | observed stream-scale continuation **with counterexample to direct sent-coordinate identity** | Amend S1 time/progress row: control-associated time inflation is observed; defer mapping across finalize until justified origin/offset behavior, never hardcode subtraction |
| S-O2,S-O4,S-O5 | observed repeated finals, normal terminal tail/finished/CLOSE, short recognition | Preserve tokens/progress independent of diarization; keep final-tail intake and subsecond speech; no product control enablement |
| S-O3 | counterexample: final progress exceeds actual PCM frontier | Progress is provider processing evidence, not a certified direct capture enclosure; map only covered intervals, preserve uncertainty |
| S-O6 | observed one no-added-postsilence flush plus successor tail | Keep recommendation/cadence limitations; one useful trace does not establish arbitrary rapid finalize safety |
| D-O1,D-O2,D-O3,D-O4,D-O5 | observed native evidence/model, cumulative time, distinct repeats, final tail/retirement, short text | Preservation/T candidate supported for this selected wire route; packaged adapter mapping/tail work still required |
| D-O6 | observed no result within4s; universal absence inconclusive | Keep documented no-ACK fallback and CloseStream irreversible retirement; do not infer safe reuse from timeout |
| E-O1,E-O2,E-O3,E-O4 | observed global word seconds, plain-before-timed pairs under continuation, exact text coverage, distinct repeats | Replace unconditional unknown origin for this pinned probe configuration with observed stream-global semantics; follow Director's CWT-only normal admission/one-winner fallback policy, not cross-event string pairing; retain generation/mapping and fail-closed ambiguity gates |
| E-O5 | counterexample to1s minimum:629.625ms completed `No.` | Reject invented1s floor; preserve short speech |
| E-O6 | immediate short completion observed; conditional extension not-run | Do not impose universal2s manual commit startup floor; no diagnostic padding adopted into product |
| E-O7 | observed explicit empty rejection + server-stated0.3s uncommitted minimum | Add actual threshold/error provenance. Avoid empty/subthreshold enhanced commits; do not discard source or fabricate speaker-owned silence. Exact threshold qualification remains separate |
| X-O1 | Soniox no stable result ID; Deepgram request connection-only; Scribe session connection-only/no per-commit ID | Receiver receipt/contribution identity protects internal replay only; do not dedup text or claim upstream IDs |
| X-U1,X-U2 | inconclusive/not-run universal order/replay/cadence/accuracy/latency/lifetime/cost | Keep documented guarantee limits/account unknowns. No rate stress, long sessions, pricing requests or meters |
| X-OFF1 | intentionally unchanged | Prior offline fixture observations remain separate; no fake-parser rerun substitutes for these calls |

The maintained policy's previous statements that no live paid/synthetic-audio calls were authorized/run, no live settings were inspected, and timestamp-origin qualification was entirely absent need scoped historical wording rather than deletion of the prior evidence. User authorized these minimal real API requests here. Only default model/endpoint fields and approved ASR keys were read; no selected profile/route, account subscription or product setting was changed. Director should link these traces and distinguish the unchanged adapter losses from newly observed upstream behavior. Enhanced Soniox/Deepgram/ElevenLabs production timed join still needs actual piecewise maps, immutable contribution/tail reconciliation, optional metadata fail-closed logic and provider-specific controlled enablement; this experiment does not deliver or certify product integration.

No repository tests/builds/linters/formatters were run. Executed verification is only source/environment preparation, bounded real native-wire experiments and deterministic examination of their recorded events/text/times. Do not automatically rerun API commands during handoff: that creates new requests. Director can inspect existing JSON and script hashes without API calls; a reproduction requires explicit scheduling and the same authorized account/model/audio envelope.

## Approved Soniox passive discriminant (persisted before third call)

Director approved exactly one third 9 s A/A/B session, same PCM slots/sample counts but **no finalize**, only empty TEXT end at144000 samples. Added finite **S-O1b/S-O3b**: compare lexical token intervals AND final-progress overrun against the 3/6/9 s controlled trace. Alternatives: passive clock already drifts/quantizes similarly; manual flush changes progress alone; manual flush also shifts lexical clock; insufficient token alignment to decide. Fresh source PCM baseline: each A slot48000 samples, B slot48000, total144000; cumulative thresholds48000/96000/144000. Final excess = reported progress−9000ms, recorded without clamping. Matching lexical comparisons retain per-token boundary uncertainty; progress excess alone does not prove lexical inflation. No fourth connection without a distinct policy-changing ambiguity. Total permitted Soniox after this call:3connections/21.98825s sent PCM. If implicit padding cannot be reliably mapped, future policy must stay conservative rather than inventing a correction model.

### Third-session outcome and matched clock comparison

Executed exactly the approved third connection; no fourth was used. Command exited0, tool wall11.44s, session wall11.031634s:

```powershell
../../../.venv/Scripts/python.exe docs/experiments/issue-212-clarification/timed_cloud_probe.py --provider soniox --sessions passive --output docs/experiments/issue-212-clarification/timed-cloud-results-soniox-passive.json
```

`timed-cloud-results-soniox-comparison.json` is deterministic recorded-data analysis, not another call. All nine input-map segments/PCM hashes are exactly equal to the controlled normal run; both sent144000 samples/9000ms. Passive session has **zero finalize controls**, one empty TEXT end at elapsed9.675798s/cursor144000. Passive lexical text/token sequence is exactly equal to controlled run: `The red boat is ready. The red boat is ready. The blue train is late.`

| Passive received event | Cursor / progress / exact observation |
| --- | --- |
| `85`, elapsed6.938126s | Sent100800samples; final progress2040ms, total6120ms; first A final tokens through `ad` |
| `112`, elapsed8.950422s | Sent132800samples; final progress4080ms, total8160ms; final first A `y.` and second A ` The` |
| `123`, elapsed9.858248s | Sent144000samples; final/total progress**9000ms**, final remainder second A+B |
| `124`, elapsed9.858297s | `finished:true`, final/total progress9000ms |
| `125`, elapsed11.031546s | Transport CLOSE1000 |

Final progress excess: controlled +240ms (9240−9000), passive **0ms** (9000−9000). Intermediate progress is provider processing availability, not receipt's sent frontier (e.g. passive2040ms final while6300ms has been sent).

**Lexical token comparison, controlled minus passive at equivalent PCM source windows** (complete per-token values retained in comparison JSON): first A's eight token intervals are **all identical**. Second A's first seven token intervals are each +120ms; last `y.` is +60ms. B's ` The`, ` tra`, `e.` are +180ms; ` bl`, `ue`, `in`, ` is`, ` lat` are +240ms. Example: second-A ` The` controlled3600–3660ms vs passive3480–3540ms; B ` bl` controlled6900–6960ms vs passive6660–6720ms. This supplies actual lexical evidence beyond progress overrun: the matched flush-containing trace exhibits downstream displacement while the first source window is identical. Native timing quantization/ASR variability still exists; this experiment establishes neither the underlying mechanism nor an exact correction law.

**S-O1b/S-O3b: observed control-associated progress AND lexical displacement; passive time map consistent for this trace.** Current policy must not claim that provider token/progress ms always equal direct sent PCM coordinates across manual finalize. No guessed implicit-padding map, clamp, assumed origin reset, new model or subtraction was implemented. Conditional passive continuous timed intake remains feasible after ordinary mapping/adapter qualification, but any manual-finalize-crossing timed attribution stays gray/conservative until explicit reliable provider control-clock mapping is available. Gray must preserve text and source-order provenance; it is not permission to guess successor ownership. Existing normal legal finalize/text completion remains available; Soniox final-text immutability and short-source preservation are unchanged. Ask provider contract/support for the control-induced coordinate behavior rather than adding further calls that cannot prove a universal formula.

Final actual envelope: **Soniox3connections/21.98825s PCM; Deepgram2connections/9.629625s; ElevenLabs2connections/9.629625s**. Total7real connections. Final reproducible script SHA-256: `3bdb77673b8c4cec18a4f0e1f2099311b5944b252e081eacc03247dd43ad710e`, recorded by the passive artifact; earlier actual traces intentionally retain their original script hashes. No additional live probe is required merely to prove guaranteed Scribe metadata delivery; no such guarantee was inferred.
