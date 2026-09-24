# Issue 185 outcome G — remote ASR sealed-turn overlap

Investigation only. No production overlap flag, protocol relaxation, product change, or paid/live provider call.

## Execution record

| Fact | Value |
| --- | --- |
| Revision | `78d90ca9722d3c88e05448bbe7c958892c5b11ec` |
| Requested baseline | same commit |
| Environment | Python 3.14.7, configured project environment |
| Reproduce | `PYTHONPATH=src python experiments/issue_185/g_remote_overlap_probe.py` |
| Installed SDKs used only as import/version context | deepgram-sdk 5.3.4, elevenlabs 2.65.0, google-genai 2.21.0, dashscope 1.26.4, websockets 16.1.1, httpx 0.28.1 |
| Repo pins | `deepgram-sdk>=5.3.4,<6`, `elevenlabs>=2.60.0,<3`, `google-genai>=2.21.0,<3`, `dashscope==1.26.4`, `websockets==16.1.1` |
| Probe | `experiments/issue_185/g_remote_overlap_probe.py` |
| Sanitized results | `experiments/issue_185/g_results.json` |
| Mock cases | 38 passed, 0 failed, 0 probe errors |
| Live wire evidence | **not run** — no spending authorization |

`ScopedRecognitionEngine` still waits for a remote terminal unless the session exposes `allows_sealed_turn_overlap`. None of the current remote sessions set that flag. The probe observed `overlap_flag: false` on every production session it constructed. Local CPU/GPU overlap from #183 is unchanged and is not a remote result.

Architecture (`docs/architecture.md`) keeps recognition ownership in `ScopedRecognitionEngine` and per-turn receipts in `STTSessionEventProjection`. Remote adapters own native requests, barriers, and protocol errors. #134 C7 still allows one unresolved submitted provider turn per connection. #134 C9 remains the per-route completion table. This investigation does not amend either contract.

## Current remote adapters

Current selectors in `STTProviderName`: `deepgram`, `gemini_transcribe`, `elevenlabs_scribe`, `qwen_audio`, `soniox`, `custom` / `custom_offline` / `custom_realtime`, and `rolling_free`.

`qwen_asr` is not a current adapter. Settings migration rewrites that provider to `rolling_free` or `qwen_audio` and drops the `qwen_asr` key. There is no Qwen ASR Realtime session class on this revision. The #134 C9 row for Qwen ASR Realtime is historical here, not a live overlap candidate.

Aliases resolve to one of the sessions below. Rolling free is a member composition, not a fourth protocol.

## Evidence classes

- **Documented:** primary vendor pages read on 2026-09-25, listed per adapter. Documentation is not conformance.
- **Mocked:** production session methods and production message handlers, with fake sockets/SDK objects and no network. 38/38 cases passed.
- **Live:** not run.

## Matrix

| Adapter | Current barrier | Native correlation available to the adapter | A-final / B-input | Empty / error identity | Disposition |
| --- | --- | --- | --- | --- | --- |
| Deepgram | One projection turn. Seal sends `Finalize`. Turn closes only on `from_finalize`, or on the existing CloseStream drain if that ack is absent. `speech_final` / `is_final` are not the barrier. | Connection `request_id` and optional channel index. No turn id. `from_finalize` is a boolean, and Deepgram says it is not guaranteed when little audio is buffered. | B `begin_turn` raises `STT session already has an unresolved turn` while A is sealed. After A is released, a `from_finalize` scheduled against the then-active identity is attributed to B. | Empty `from_finalize` is authoritative empty for A. A later ack with no active turn ends the epoch as `deepgram_idle_result`. Missing ack fails A as `deepgram_finalize_ack_missing` and retires the epoch. | **no-overlap** |
| Soniox | One projection turn and `_pending_finalize_requests == 1` after seal. Barrier is `<fin>` for that one manual finalize. `<end>` is ignored. | Message `request_id` is session-scoped. No finalize id. | B cannot begin until `<fin>`. Docs allow another `finalize` and more audio after `<fin>`, not a second outstanding finalize. A final token that arrives after B has begun is appended to B. | Empty `<fin>` is authoritative empty for A. Unsealed or duplicate `<fin>` is `soniox_protocol_ambiguity` and retires the epoch. No adapter-local final timer; unresolved `<fin>` simply holds the turn. | **no-overlap** |
| Gemini Transcribe Live | One scoped turn, plus both authoritative `input_transcription` and ActivityEnd. ACK alone does not promote interim. | `ActivityStart` / `ActivityEnd` have no fields. Transcription messages have no turn id and no guaranteed order against other server messages. Handler is first-pending FIFO. | B `begin_turn` raises `Gemini allows one unresolved scoped turn` after seal and before both barriers. A second authoritative message is not applied to another turn. | Empty authoritative text is empty for A. Unsolicited ActivityEnd retires the idle epoch. Timeout keeps interim only as degraded A and retires the epoch. | **no-overlap** |
| ElevenLabs Scribe | One projection turn. Manual `commit()` then `committed_transcript`. Partials and timestamp transcripts do not resolve the turn. | Adapter provenance ignores the payload. Published `CommittedTranscript` required fields are `message_type` and `text`. `session_id` is session-level. No commit id is read. | B cannot begin while A is unresolved. After B is sealed, a later `committed_transcript` is attributed to B. | Empty committed text is authoritative empty. Unsolicited committed text with no active turn retires the epoch. Two committed events during the commit write fail the epoch. Commit failure fails A. No adapter final-wait timer. | **no-overlap** |
| Qwen Audio streaming | One client `task_id`. Seal sends `finish-task`. Barrier is `task-finished` for that id. `sentence_end` is not the turn barrier. | `task_id` is a client UUID for one task. `sentence_id` dedupes sentences inside that task. Docs say `task-finished` ends that task; the connection may then be closed or reused. `task-failed` is not a reusable success. | B cannot begin while finish is outstanding (`Qwen Audio task is not active`). Stale `task-a` results after the next task starts are ignored. | Empty `task-finished` is authoritative empty for A. Missing `task-finished` fails A and retires the epoch. | **no-overlap** |
| Custom offline | One projection turn, one `_scoped_task`, and the transcribe lock. Completion of that HTTP call resolves that request. | No shared stream. `POST /audio/transcriptions` returns a transcription object for the file in that call. The adapter snapshots PCM into the task before the call. | B `begin_turn` and a second seal are rejected while A's task is unresolved. A's response stays on A. | Whitespace-only text is empty for A. HTTP 500 is failed for A, not empty text, and retires the epoch. The 50 s total-timeout branch was not waited out. | **no-overlap** |
| Custom realtime | One pending commit. `turn_detection` must stay null. Completion must match the one scoped `item_id` captured from `input_audio_buffer.committed`. Unkeyed completion after reuse retires the epoch. | Server `item_id` can separate completions after commit. The client commit message has no caller-supplied turn id. Audio appended before `committed` is not proven to be excluded from the open buffer. | B cannot begin while A's commit barrier is unresolved. A completion before `committed` retires the epoch. A late `item-a` completion does not become B's text after B has `item-b`. | Empty keyed completion is authoritative empty. Unkeyed duplicate retires the epoch. Final timeout fails A and retires the epoch. | **no-overlap** |
| Rolling free | Forwards scoped calls to the selected member. No overlap flag and no extra id. | Member switch is an epoch boundary. C9 forbids replaying ambiguously submitted audio into another member. | Same as the active member. | Same as the active member. | **no-overlap** |

No adapter is **safe-overlap**. No adapter is left **unresolved** as a production recommendation: every current remote route stays serial. Live traces remain unrun, which blocks a conformance claim, not this no-overlap conclusion.

## Adverse cases executed

Each row is a production-adapter mock, not a guessed trace. Details are in `g_results.json`.

| Case | What the production code did |
| --- | --- |
| Delayed A, B ready | Every streaming and HTTP session rejected B while A was sealed and unresolved. The non-barrier token (`speech_final`, `<end>`, authoritative text alone, partial/timestamp transcript, `sentence_end`) did not release A. |
| Empty A | Empty Deepgram ack, Soniox `<fin>`, Gemini authoritative text, Scribe committed text, Qwen `task-finished`, custom HTTP whitespace, and custom realtime keyed completion were authoritative empty for A, not a new turn. |
| Duplicate / unsolicited | Deepgram idle ack, Soniox extra `<fin>`, Gemini extra ActivityEnd, Scribe committed-with-no-turn, and custom realtime unkeyed completion retired the epoch instead of inventing B. Scribe duplicate committed events during the commit write failed the epoch and did not publish the second text. |
| Failure / timeout | Deepgram missing ack, Gemini finalize timeout, Qwen missing `task-finished`, Scribe commit failure, custom HTTP 500, and custom realtime final timeout all terminalized A and retired the epoch. Soniox has no adapter-local timer; an unresolved `<fin>` keeps B blocked. The engine's 20 s horizon was not executed here. |
| Late after reconnect / config change | Abort with `config_changed` retired the old epoch. A replacement session with a new epoch accepted B. Late messages on the old session did not appear in the new session. |
| Simultaneous work | A second `begin_turn` failed on every adapter while the first turn was open. Rolling forwarded that rejection and added no correlation. |

One mock result is sharper than "B is rejected":

- Deepgram: after A has been released and B sealed, a `from_finalize` captured against the then-active identity becomes B's text. A result still carrying A's retired identity is ignored.
- Soniox: a final token arriving after B has begun is appended to B. The only id on the message is the session `request_id`.
- Scribe: a `committed_transcript` arriving after B is sealed becomes B's result. The event has no commit id.

That is late A text becoming B's result under serial reuse, not under an enabled overlap flag. It is why a guessed FIFO or a blanket flag is unsafe. It is also narrower than C3's rule that an ambiguous unkeyed completion must retire the epoch rather than move A's late text onto B. Whether vendors emit that extra message is **live-unknown**. This investigation does not patch it. A correction, if the maintainer treats the mock as a C3 gap, is a separate implementation assignment. It is not authorization to overlap turns.

Qwen and custom realtime did not show that mis-attribution: stale `task_id` and mismatched `item_id` were ignored. Those keys still do not make overlap safe. Qwen documents one task, then `task-finished`, then reuse. Custom realtime stores one item id, rejects a second begin, and cannot prove that audio appended before `input_audio_buffer.committed` stays out of A's buffer.

Custom offline is the only protocol whose response is naturally bound to one request. The current session still refuses a second turn and a second seal, and C7/C9 still say one unresolved submitted turn and one sealed PCM request. Request-scoped HTTP is not permission to run two in-flight transcriptions.

## Vendor pages read

- Deepgram Finalize: https://developers.deepgram.com/docs/finalize — `Finalize` flushes the stream; `from_finalize: true` marks that flush and is not guaranteed when little audio is buffered. Optional `channel` is not a turn id.
- Soniox manual finalization: https://soniox.com/docs/stt/rt/manual-finalization — `{"type":"finalize"}` finalizes audio received up to that point and returns `<fin>`. Further audio is described after that completion, with no finalize id.
- Gemini live transcription: https://ai.google.dev/gemini-api/docs/live-api/live-transcribe — `input_transcription` is the finalized transcript; interim text is separate.
- Gemini Live WebSocket reference: https://ai.google.dev/api/live — `activityStart` and `activityEnd` have no fields. `inputTranscription` has no guaranteed ordering against other server messages.
- ElevenLabs realtime: https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime — `committed_transcript` is the stable segment result. Its required schema fields are `message_type` and `text`.
- Qwen/Fun-ASR server events: https://www.alibabacloud.com/help/en/model-studio/fun-asr-server-events — `task_id` is the client task UUID; `task-finished` ends that task and allows close or reuse; `sentence_end` is a sentence result, not the task barrier.
- OpenAI transcriptions create: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create — one `POST /audio/transcriptions` returns the transcription object for that file.
- OpenAI realtime reference: https://developers.openai.com/api/reference/resources/realtime — conversation items have ids. The adapter's own commit payload is only `{"type":"input_audio_buffer.commit"}`.

## Decision

**No production overlap extension is recommended.** No narrow contract amendment is requested. The maintainer remains the decision owner if a later change is proposed.

What would be required before any one adapter could overlap, and which this investigation does not grant:

- an explicit amendment to the affected #134 C7/C9 row and the shared one-outstanding-turn rule;
- vendor conformance, including an authorized sanitized live trace, that successor audio cannot join A's unresolved buffer and that A's late text cannot resolve B;
- adapter work that keeps a real per-turn key through admission, seal, terminal, duplicate, timeout, reconnect, and cancellation;
- no extra session, model, or paid request.

Missing credentials block only the live claim. They do not leave the serial decision open.

## Status and gaps

| Item | Status |
| --- | --- |
| Adapter matrix and mock adverse cases | passed |
| Production overlap flag | not enabled |
| Live vendor conformance | not run |
| Soniox/Scribe engine 20 s horizon and custom offline 50 s total timeout | not run in this probe; adapter hold/failure paths that were reachable without those waits were run |
| Late unkeyed result after the next sealed turn on Deepgram, Soniox, and Scribe | pre-existing mock observation; live frequency unknown; no product repair authorized |
| Qwen ASR Realtime | not a current adapter |
