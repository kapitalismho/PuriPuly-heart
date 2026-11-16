# #441 pilot readiness and execution gaps

## Owner amendment: automatic execution

After this readiness checkpoint, the owner directed: “아 그런거 없이 자동으로 해줘. 필요하면 오디오 소스를 바꾸고” — proceed automatically without the manual prerequisite, changing audio sources if necessary.

This supersedes the requirement for new human/audio listening before the pilot. Use independently authored corpus references plus automatic signal/voice checks; freeze the inputs and scoring opportunities before inspecting candidate outputs. Mark these references as annotation-derived, not listened or acoustically adjudicated. Keep uncertain semantic boundaries and overlap limitations visible. Audio replacement is permitted subject to source conditions. All other matched-input, provider/controller, scoring, timing, production-preservation and reporting requirements remain in force.

Completed automatic execution and the decision are recorded in [the #441 results report](issue441-results.md). The sections below are the historical pre-amendment readiness record, not the resumed execution status. The listening blocker and its proposed human handoff are superseded; the original unrun conditions and setup gaps must not be mistaken for the final result.

## Disposition

**Blocked before the required input/label freeze. No pilot condition has run.** This is a prerequisite investigation, not a completed comparison or a supported rejection of Scribe. Keep [#441](https://github.com/kapitalismho/PuriPuly-heart/issues/441) open. Do not choose a provider or diagnose a Soniox defect from these observations.

Inspected checkout: `9f7474b94b885a7a9a4777b43a3a05026e3e248e`, matching the issue's planning pin and remote `dev` at investigation start. The worktree was clean. The existing PuriPuly Project item was Backlog. No GitHub, Project, product, settings, credential, or production-routing changes were made. This report is local until explicitly published.

The decisive missing prerequisite is **audio-listened reference adjudication**. The exposed assistant tools provide no auditory input. Installed playback software does not let this assistant hear a recording. The [old reference file](issue373-references.json) explicitly declares text-only annotation and no new human/audio adjudication. Neither its labels nor candidate ASR output can substitute for the required listening. A human or audio-capable adjudicator must freeze the selected spans before candidate output inspection.

## What was actually checked

Read-only inspection covered the current issue and #373/#356 amendments, #218 cancellation and historical runtime records, production owners, existing benchmark scripts, public provider contracts, and bounded known local artifact locations. Local Python probes checked WAV structure/hashes, dependency metadata and credential **presence only**. There were no provider connections, model inference, microphone/loopback capture, audio playback, actual MT calls, GUI runs, or product tests. No credentials or original audio are included here.

The reusable interpreter reported Python 3.14.10; installed metadata included NumPy 2.5.1, ONNX Runtime 1.28.0 and websockets 17.4. This is not an import-success or inference-performance claim. The benchmark must explicitly import the intended checkout through its existing `--root` convention; the worktree has no own virtual environment.

### Reusable audio

Both existing WAVs were opened with Python `wave` and hashed from their bytes. Each contains 1,920,000 frames of 16 kHz mono PCM16: 120 seconds. The hashes match the [#373 report](issue373-results.md).

| Input | Original interval | Manifest reference inventory | SHA-256 |
|---|---|---|---|
| AMI ES2004d, English | 32.368–152.368 s | 33 turns; 21 reference-available; three manifest speakers | `ee19cafe7382e7f26249b273c0f27ce080963788e35c2ca67034f234d903b6d6` |
| IngCrowd dialogue01, Japanese | 86.04–206.04 s | 49 turns; 47 reference-available; two manifest speakers | `8cdbe6868df6e141f80a14ce7e49b501cd371e185e2899a2b1a659dc22d06a2b` |

The inventory is not newly verified acoustic ground truth. Retain [English](inputs/replay-en.json), [Japanese](inputs/replay-ja.json), and [source/license pins](inputs/sources.json). AMI is CC-BY-4.0; IngCrowd is CC-BY-NC-4.0, used for noncommercial verification. Do not redistribute the audio. Original AMI and IngCrowd dialogue01 files and annotations remain locally available; other cached IngCrowd conversations may supply diagnostic material after listening.

Korean/Chinese prepared audio was absent from the checked benchmark data directories and historical preparation paths. Local Git retains `docs/experiments/issue-308/manifests/replay-natural-ko.json`, `replay-natural-zh.json`, and `scripts/bench_asr308_data.py` at `6fa20d64eeb080158fa60c48b1bbb0597ac55eb1`:

- Korean Seoul `s13m26m1`, 66.5881179–186.5881179 s, is a candidate two-speaker interview. The historical manifest has 38 turns, 34 reference-available. Interviewer `<IVER>` spans lack lexical text, and a single annotation tier does not establish absence of overlap. The source record declares CC-BY-NC-2.0. It is not a selected/listened #441 input.
- Chinese AISHELL-4 `L_R003S01C02`, 145.3515625–265.3515625 s, has **five** manifest speaker IDs. Do not reuse this window unchanged: #441 requires 2–4 speakers. Its source record declares CC-BY-SA-4.0 and fixed channel 0 of eight-channel audio. Select another eligible intact window/source before scoring.
- No approximately four-minute diagnostic subset, 100–160 listened single-speaker anchors, approximately 20 change/re-entry opportunities, overlap labels, or approximately 24 ending/intermediate/unsafe-cut opportunities has been frozen.

Bounded public-source probes found alternatives but did not freeze any input:

- **Chinese:** [AliMeeting/OpenSLR 119](https://www.openslr.org/119) provides natural 2–4-person meetings under CC-BY-SA-4.0. The [Argmax mirror](https://huggingface.co/datasets/argmaxinc/ali-meetings/tree/67b7506f0db35d211aca5d0fae7ff946439f1bea), test row 0 (`R8002_M8002_MS802`), has four reference speakers. A 4,096-byte HTTP range request returned 206; its WAV header reports PCM16/16 kHz/mono and 2,066.52 seconds. The mirror's original-channel conversion recipe is not established. Its expiring asset URL must be resolved again. No interval was listened to or selected.
- **Korean:** [MagicHub ASR-KCSC](https://magichub.com/datasets/korean-conversational-speech-corpus/) has an accessible [pinned publisher mirror](https://huggingface.co/datasets/MagicHub/korean-conversational-speech-corpus/tree/364fb908b14b9e8383ef6bf9f7ebf5088ffddaf3). The `A0051_S0001_0_G0101/G0102` pair has speaker/time/transcript references; both 4,096-byte range requests returned 206 and headers report PCM16/16 kHz/mono, 906.6901875 seconds. These are separate speaker tracks, not a verified single conversation mix. The publisher page links CC-BY-NC-ND-4.0, the card says academic-only, and bundled text restricts reproduction. Cloud-upload permission and track synchronization are unresolved; do not silently adopt this source. The historical Seoul candidate remains an alternative.

Only small remote headers/ranges and public metadata were inspected; no complete corpus audio was downloaded, played or submitted to ASR. These leads do not replace the listening prerequisite or establish source/provider retention permissions.

### Provider and local-runtime readiness

Direct read-only Windows keyring lookups returned present for both `soniox_api_key` and `elevenlabs_scribe_api_key`; values were not printed. Presence does **not** establish account validity, billing, model entitlement or current service availability. The historical #218 Scribe suspension is not evidence of today's account state. No live entitlement check was attempted.

Nemotron's model and runtime were absent at the known prior temporary model path, application model directory and original checkout's `build/nemotron` directory. This is bounded location evidence, not a machine-wide absence claim. The existing GPU inventory included AMD Radeon RX 7900 XTX; no Vulkan/model execution was observed.

Recoverable standalone references, without reviving the cancelled product integration:

- Local Git `c7e7ae0d5bfb8a5ab8637c0dbc1e40eb0f84e6f9`: `native/nemotron/prepare.py`, `patches.json`, `acquire_preconverted.py`, `issue218_report.json`, and historical runtime/model manifests.
- Model: `Bonenk/Nemotron-3-Diarization-GGUF`, revision `25cf0d6b0c91b679376df0f6ef3087b4fd96af1f`, `Nemotron-3-Diarization.f16.gguf`, 201,832,928 bytes, SHA-256 `9b277e1dcec60b2c771a1dcb45dc8e36f7285ff7c1619d880d3926a373036b38`.
- Runtime source: `NVIDIA/NeMo-Speech.cpp` at `b809bbb467fb2da2be0042fff5ef1f503828c9cf`, with historical pinned patches/dependencies. A model file alone is not a working runtime bundle.
- Only the existing 1.04 s profile is in scope: chunk/right/left/FIFO/speaker-cache/update-period = `9/4/0/264/264/222`. Its buffer label is not measured processing delay. Historical runtime recovery has not been attempted.

The old `.tmp/issue373` traces and `build/issue-373-display` evidence were absent in the checked current/original checkouts. Their published report remains historical evidence, not a new run.

## Conditions and missing observations

The following are required configurations, **not executed configurations**. No condition produced an event log or consumed provider audio in this investigation. Correct/wrong/unassigned counts, denominators, overlap outcomes, timing distributions and boundary counts are **not measured**, not zero accuracy or zero failures.

| Condition | Required operating profile | Present execution gap |
|---|---|---|
| N | Continuous Nemotron, one session/recording, existing 1.04 s profile | Frozen inputs/labels and restored working runtime/artifact; no activity or processing-delay observations |
| S0 | `stt-rt-v6`; continuous silence-preserving input; diarization on; endpoint detection off; no intermediate finalize | Frozen inputs/labels and continuous capture adapter; account/model access untested |
| S1 | Current production VAD/input/finalize, native Soniox attribution and caption composition | Frozen inputs/labels and strengthened same-run native/caption observations; access untested |
| E0 / T0 | `scribe_v3_live`; continuous audio; actual current controller's manual boundaries | Frozen inputs/labels, raw v3 adapter and boundary-only controller bridge; access untested |
| T1 | Same Scribe input/features; `commit_strategy=vad`; 1.5 s silence fallback; 10 s cap | Same prerequisites; separate real session required |
| T2 | Same as T1 except `commit_strategy=turn_prediction`; documented 0.9 threshold | Same prerequisites; separate real session required, not shadow-signal replay |

All Scribe arms require `diarize=true`, timestamps on, background filtering off, declared matched language/multilingual settings, and no initial keyterms/transcript editing. T0 must retain shadow `turn_probability`. T1/T2 must omit production 4/5 s opportunity cuts and mark cap/EOF commits separately. Exact language settings remain an execution choice to freeze with the selected corpus.

No five representative provider/application successes or failures can be reported without the new logs. No promising candidate or supported rejection has been established. The conditional 6–8 interval actual-MT/display check remains pending evaluation, not waived or passed.

## Reuse and smallest missing experiment mechanics

Reuse [issue373.py](issue373.py) and production owners, not its old CLI/scorer unchanged. Keep all additions experiment-local; changing production Scribe or speaker contracts is outside this issue.

1. **Continuous audio and ledger.** Feed every native-arm sample once, preserving silence. Record attempted/completed writes, source/transmitted ranges, monotonic arrival/send clocks, session identity, startup/EOF, actual sent duration and connection duration. S1 needs its real VAD omissions, replay/context and padding mapped separately. The old capture hook records before an awaited write, and the old scorer reconstructs padding from policy constants; neither is a completed-send ledger.
2. **Real controller for E0.** Send continuous PCM independently; use actual production VAD/SmartTurn/controller decisions only for manual commit boundaries. Do not forward VAD-selected/replayed/padded PCM to E0. Preserve precedence: 10 s cap, cloud 4/5 s opportunities with 192 ms pauses, then ordinary SmartTurn 512 ms or timely-incomplete 800 ms. Record actual SmartTurn readiness and branch outcomes, not just intended defaults.
3. **Raw Scribe events.** The production adapter defaults to v2, manual commits and text-only projection. Merely overriding its model name does not capture v3 diarization. Retain partials, committed text, delayed diarized events and merges, joined by `segment_index`. The official guide and generated event schema differ; preserve raw extra fields and report missing required fields rather than synthesizing them.
4. **Independent online snapshots.** Score the same frozen anchors at reference-span end +1 s and eventually. Keep missing text/timing/speaker reasons separate, use one session-level one-to-one speaker map, and preserve receive-ordered provisional/settled states. Later diarization/merges cannot repair the earlier snapshot. Replace the old recognized-token denominator; it cannot measure the issue's missing-content anchors.
5. **Application observations.** Separate S0 versus S1-native input/context/finalization effects from S1-native versus S1-captions composition effects. The old replay uses immediate deterministic MT; it can establish structure, not real in-flight MT/GUI timing. Current native-speaker caption authority is Soniox-specific. Do not disguise Scribe as Soniox or split translated strings by source offsets. Measure identity-only versus regrouping/retranslation consequences separately.
6. **Matched boundary runs.** Use one cap-only supervisor for T1/T2, the same 1.5 s fallback and no 4/5 s commands. Preserve commit/cap races, missing tails and drain timeouts. Bound a recording by its duration plus drain rather than the old fixed 160 s capture timeout. Persist partial evidence on failure and isolate condition subprocesses.

Official references inspected: [Scribe commit strategies](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/transcripts-and-commit-strategies), [diarization/merges](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/diarization), [API parameters](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime), and [Soniox's finalization tradeoff](https://docs.soniox.com/stt/concepts/speaker-diarization).

## Resume boundary

1. Arrange a human/audio-capable adjudicator. Acquire/select eligible Korean/Chinese conversations and diagnostic intervals; freeze licenses, source offsets/hashes and listened scoring opportunities before candidate output inspection. Keep uncertain/overlap spans explicit. Do not silently weaken this requirement to text-only labels.
2. Restore only the standalone Nemotron runtime/artifact at the recorded profile. Validate its inventory and actual execution; do not resurrect #218 integration or qualification.
3. Add the bounded missing adapters/observations. Check source conservation, failed sends, delayed/interleaved indexed events, merges after cutoff, zero-output sessions, actual controller branches and cap/EOF races offline.
4. Run one real paced pass per required condition. Record actual model/config acceptance and service failures; repeat only bounded anomalies. Score online/eventual views and boundary outcomes, then decide adopt/defer/fix-first from those results. Perform actual MT/display intervals only if a candidate is promising.
5. Preserve the comparison, roughly five explanatory cases and evidence limits in a linked shared record. This local readiness report does not fulfill that publication or issue-completion criterion.

**Recommendation now: defer the integration decision pending the required pilot, not reject Scribe.** The next unresolved question is whether timely v3 speaker metadata improves attribution under the actual controller on the frozen listened inputs, and whether prediction adds useful boundaries beyond the matched 1.5 s VAD baseline. No production fix is justified by this readiness inspection. User-visible product changes: none. Protected documentation drift caused by this investigation: none.
