# Broker/Relay efficiency: fresh #389 measurements

Measured on 2026-11-10 for [#389](https://github.com/kapitalismho/PuriPuly-heart/issues/389). All numerical evidence was collected anew; no #383–#386 performance results were reused. [Structured observations](broker-relay-efficiency.json) retain every measured sample, exclusions, source identities, platform meters, and cleanup results. Repeated per-block diagnostics are summarized by count and first/last event.

**Initial decision:** retain two small optimizations and recommend Workers Paid. Neither a general latency improvement nor complete #389 closure is established. Actual first display, visual usability, and the prerequisite comparison gaps below remain unverified. No production deployment, subscription change, push, merge, or issue closure was performed. A separately approved Broker maintenance follow-up is recorded at the end; its new measurements do not relabel this initial baseline.

## Pinned implementation and boundaries

- Integrated app/Broker/Relay baseline: `b55cb1ea1f97a84adc19a42c7dc7bfe6928e7693`.
- Local implementation commit: `670693dbea4ec97cbab5efa9912046db23d57f6f`.
- Changed production files: [Relay stream](../agent-relay/src/stream.ts) and [Broker scheduler](../broker/src/scheduled.ts). The Python/native application, authentication, image preparation, model contracts, and migrations are unchanged.
- Existing #383/#384 tool/context/recommendation changes, #385 enrollment/admission protections, and #386 `/v1/photo` implementation are present in the baseline. Their savings are not attributed to this change. Closed issues and owner-waived verification do not certify #389's prerequisite acceptance: #383–#385 retained waiver limits, and #386 was closed as not planned with remaining display/route-comparison evidence waived there.
- Current agent protocol is 6; its existing retirement decisions are not reversed. The emitted context exporter captured nine cases per arm. Classification, work, social, out-of-scope contexts, tools, and migration manifests were identical. Context SHA-256: `4a09f3c1ec3c82f500a81191bfc75254a7b34460e96e488fb4eae115d403422d`.
- Chat model remains `gpt-6-luna`; work reasoning is `high`, social/out-of-scope reasoning `low`, output cap 25,000. Photo remains `gpt-6-luna`, reasoning `none`, output cap 8,192, Korean target, identical canonical prompt/schema. Prompt/source hashes are in the JSON.
- Frozen emitted Relay bundle SHA-256: baseline `7d3be076c43f0279c612adf6e3d550058f1abd2ec2117149f323a8b26976b82f`; candidate `07e865e5489d7c23e065c713c85a4463b9c6475ffed4853534d82a7e3bdf998c`.
- Isolated Relay Workers used compatibility date `2026-04-08`, `enable_request_signal`, existing six migrations, chat/photo enabled, support uploads disabled, no cron, and synthetic JWT subjects. Real Relay admission/settlement ran. Synthetic issuance replaced Broker enrollment only; credential-setup latency is not included in photo timings.
- Isolated Broker used all 33 frozen migrations and both handlers in one Worker/version. The fetch probe invoked the actual scheduled handler/report path; this is not a native cron-event CPU measurement. Production was queried only for account aggregates, database sizes, and two read-only report-history SELECTs.

## Changes selected from fresh measurements

| Candidate | Decision and preservation boundary |
| --- | --- |
| Report progress history scan | Adopt. Replace materializing all eligible history with indexed earliest-pending/latest-delivered scalar lookups. Existing covering index suffices. Pending-first behavior, missing-date catchup, future-date exclusion, report history, leases, cleanup frequency, and writes are retained. No migration. |
| Terminal instruction/tool echo | Adopt. After successful required settlement, remove only echoed `instructions` and `tools` from the terminal response. Preserve authoritative output, final-only text, tool calls, encrypted reasoning, status, IDs, model, usage, and other fields. Supported consumers do not use those two echoes; emitted-Worker context evidence remains available. |
| Input/image envelope changes | Reject here. No measured benefit justifies changing image content, validation, preparation, resolution, prompt, or output limits. Existing incremental body reading is already baseline behavior. |
| Demand-driven pump/backpressure rewrite | Reject here. It can put slow readers on the upstream deadline and alter cancellation/settlement. Keep the existing eager pump; remove duplication without delaying deltas. |
| Token/HMAC caching or admission write consolidation | Reject here. Existing token reuse/singleflight already works. Fresh token probes showed bounded work; caching or merging failure boundaries adds isolation, revocation, or durability risk without a demonstrated material gain. |
| Additional photo pooling | No change. Actual cold/warm observations show existing connection reuse. Do not count that reuse as new savings or infer Worker-to-provider pooling. |
| Delete replay/history rows or run cleanup less often | Initially deferred because cadence changes affect physical retention or recovery timing. The owner subsequently approved hourly garbage collection only; see the follow-up below. Earlier deletion, weaker replay checks, and slower active recovery remain excluded. |
| More indexes, support streaming, placement, broad accounting rewrite | Defer. No isolated evidence establishes an equivalent worthwhile change. Support traffic is included only in account aggregates; detailed attribution and peak concurrency remain unknown. These are not claimed optimized. |

User-visible behavior: no intended UI, capability, image-quality, allowance, or response-timing change. Wire consumers no longer receive the two unused terminal echo fields. This is not a claim that headset appearance was observed.

## Relay: controlled resource isolation

Fresh local replay: six workloads, seven samples per workload per arm (84 total, first repetition retained as warmup). Exact handler, prompt bundle, and SQL ran with `node:sqlite`; these are local logical operations, not D1 meters or deployed CPU. All 42 matched pairs retained identical upstream request bytes, output hashes, accounting, and invocation counts.

Fresh deployed replay: six workloads × three repetitions × two arms = 36 completed requests. An additional same-Worker/version crossover measured four workloads × three repetitions × two arms = 24 completed requests, alternating arm order. Synthetic upstream was a service binding; it generated no paid model calls.

| Controlled workload | Deployed response bytes before | After | Bytes removed per response |
| --- | ---: | ---: | ---: |
| Short social | 4,751 | 502 | 4,249 |
| Tool work | 8,223 | 502 | 7,721 |
| Long work | 73,616 | 65,895 | 7,721 |
| Ordinary photo | 3,566 | 2,028 | 1,538 |
| Dense photo | 14,495 | 12,957 | 1,538 |
| Large-body photo | 28,132 | 26,594 | 1,538 |

The synthetic photo request bodies were 98,745 / 473,685 / 4,000,109 bytes. They isolate payload transport, not real image quality or a newly certified photo limit. Local replay used slightly different terminal fixtures but found the exact same byte savings. Large synthetic chat input repeated history 30,000 times and emitted 22,000 characters; no history/output was truncated by the optimization.

Separate Worker CPU measurements were confounded by isolate differences. The follow-on crossover placed both frozen handlers in the same Worker/version (`f1f1fac5-2586-42b1-b62f-200833ea13c4`) with matched synthetic histories:

| Workload | Baseline CPU samples, ms | Candidate CPU samples, ms |
| --- | --- | --- |
| Tool work | 8, 3, 3 | 7, 2, 3 |
| Long work | 21, 21, 14 | 19, 19, 13 |
| Dense photo | 13, 9, 6 | 9, 9, 8 |
| Large-body photo | 52, 51, 47 | 42, 42, 42 |

These small, millisecond-quantized samples do not establish a universal CPU/latency percentage or a tail percentile. The retained result is deterministic wire reduction without a consistent same-runtime CPU regression. D1 metadata proxy/logging, response handling, and synthetic transport instrumentation are included, not calibrated away.

Each controlled model response used one Relay invocation and one upstream request, with zero Broker invocations under a valid cached token. Local chat used five statements/two round trips; photo two statements/two round trips. Deployed D1 writes were 12 for the first daily chat, 8 for later chats, and 6 per photo, unchanged between arms. Metered reads varied with retained state and were identical for matched histories. Statements/logical changes are not billed row counts.

A completed chat task is not necessarily one response: classification plus an answer, and each tool continuation, require their own upstream work. Real chat-task/session mix, rejected/cancelled task costs, and mixed-concurrency capacity were not measured as a new live campaign. Existing focused tests cover affected error/cancellation/admission behavior; they are not a platform load test.

A slow-reader probe finished upstream and settlement before reading downstream: 501 events, 15,000 visible characters, one settlement, release observed, and final-only output preserved. No queue bound or peak isolate memory was measured. The existing eager queue can still retain a slow client's output; smaller terminal payloads do not establish bounded buffering.

## Broker: actual D1 meters and background work

Fresh local scheduler probes: 12 baseline + 12 candidate runs across histories of 1, 30, 365, and 3,650 days. Progress rows returned per tick fell from 2/60/730/7,300 to 2. Read calls (21), attempted mutations (10), and webhooks (0) were unchanged. Twelve synthetic signed token requests, including enrollment/renewal and HMAC rotation, returned HTTP 200 with three read calls, two returned rows, six mutation statements, and seven or twelve HMAC operations. Authentication was not weakened.

Fresh isolated cloud Broker version: `132cf3bb-47c8-4952-a70e-2b2bf0c89508`, APAC/ICN, 09:06–09:07 UTC. All 54 probe HTTP requests succeeded: 18 full ticks, 18 exact queries, six report behavior checks, six query plans, three history setups, and three snapshots. All 18 scheduler CPU traces were captured; 43/54 total traces were captured. Non-scheduler missing traces are not treated as zero CPU.

| Eligible report history | Lookup rows read before → after | Full quiet tick rows read before → after | Median lookup SQL ms before → after | Database bytes |
| --- | --- | --- | --- | ---: |
| 30 | 30 → 2 | 93 → 37 | 0.1466 → 0.1508 | 442,368 |
| 365 | 365 → 2 | 763 → 37 | 0.3444 → 0.1716 | 524,288 |
| 3,650 | 3,650 → 2 | 7,333 → 37 | 3.8627 → 0.1216 | 1,449,984 |

Each full tick retained 28 statements, ten attempted mutations, zero actual rows written, and zero outbound calls in the quiet fixture. Both arms produced identical report packets and retained identical database snapshots. Existing behavior tests also exercise pending/recovery/concurrent report paths; the quiet fixture is not a complete model of production cleanup.

Platform CPU samples for histories 30 / 365 / 3,650 were respectively `[27,15,15] / [15,16,15] / [20,23,22]` ms before and `[20,14,16] / [14,18,15] / [15,13,17]` ms after. These include fetch authentication/body handling, metadata capture, serialization, and logging around the scheduled handler. They must not be presented as exact native production cron CPU or as SQL duration.

Fresh database setup, separate from steady state: 33 migrations, 495 statements, 34,108 rows read, 1,248 rows written, 216.5733 ms summed SQL duration, 434,176 initial bytes. History seeding/reset meters are retained separately in JSON. The optimization itself requires no new schema/index, migration, retention change, or cleanup operation; steady-state index write costs were unchanged.

A read-only query against current production history found 79 eligible rows: baseline query read 79, candidate query read 2; both wrote zero and left size 2,883,584 bytes unchanged. Single-query SQL durations were 0.6635 and 1.1148 ms respectively: no small-history latency improvement is claimed. If that lookup runs twice per minute as in the pinned scheduler, it removes 221,760 reads/day. That is a conditional estimate, not a post-deployment production observation.

## Actual application/provider photo comparison

18 new paid calls: three fresh images × direct API/baseline Relay/candidate Relay × cold/warm. Actual `PhotoApplicationOwner`, lifecycle, provider adapter, native preparation, ONNX, and offscreen composition ran. No native source was newer than the reused binary. Binary/model hashes, fixture/prepared-JPEG hashes, request settings, request IDs, and per-sample data are in JSON.

Images were 1280×720 with six numbered labels, 1920×1080 with 24 labels, and 3840×2160 with twelve labels. Each fixture had exactly one prepared-JPEG hash across all routes. All routes used the same provider credential/project. Real API credentials were provisioned only into the two temporary Workers through secret stdin.

Direct pairs ran first while temporary Workers were prepared. Relay arm order was baseline→candidate for ordinary, candidate→baseline for dense, baseline→candidate for large. Cold/warm mean app-client TCP state, not model-cache state. This is not a fully randomized three-route latency experiment.

| Arm | First delta range, ms | First valid block range, ms | Schema completion range, ms | Completed / partial |
| --- | ---: | ---: | ---: | --- |
| Direct OpenAI | 1,285–5,622 | 1,532–5,891 | 3,831–9,595 | 6 / 0 |
| Baseline Relay | 1,518–2,949 | 1,903–3,097 | 4,005–8,540 | 5 / 1 |
| Candidate Relay | 1,494–3,230 | 1,716–3,452 | 3,535–9,147 | 5 / 1 |

Relative to baseline, candidate mean first-delta difference was +150.5 ms cold (sample SD 337.7) and +443.4 ms warm (SD 535.3); completion difference was −207.8 ms cold (SD 1,256.1) and +133.2 ms warm (SD 880.9). Each state has three heterogeneous matched images, not repeated measurements of one image. No stable latency improvement, meaningful causal regression, p95, or whole-response byte-saving percentage is inferred from these stochastic outputs. The controlled replay isolates the terminal change.

- All 18 reached schema completion, but only 16 reached application `completed`; two remained `partial`. Candidate dense/warm excluded one invalid geometry region and covered 23/24 labels. Baseline large/cold likewise covered 11/12. Both omissions remain failures of complete coverage, not successful complete translations. No paid retry was issued.
- **Actual first display was observed in 0/18 samples.** Offscreen composition/export and parser callbacks are not display. Accepted block counts appeared in native offscreen exports; label coverage does not prove semantic translation completeness or headset usability.
- Nine cold TCP connections and nine warm reuses were observed at the app client. Worker-to-provider connection reuse was not measured. No previous image/history was attached to the next request.
- Accepted-to-delta/block/schema times use the app clock. Python-process CPU excludes native/Worker/provider CPU. Probe wall time additionally includes orchestration, offscreen rendering/export and a 250 ms fade wait; none is substituted for platform CPU.
- Platform traces captured all twelve real Relay requests, with `ok` outcomes. Baseline CPU: `[132,192,299,272,115,77]` ms. Candidate: `[76,60,283,430,175,135]` ms in ordinary/dense/large, cold/warm order. Output grouping/chunking varied, so these are not an isolated CPU comparison. Every request wrote six D1 rows; reads ranged 5–11 with retained synthetic state.
- Provider-version pins: baseline `f96be739-de22-4d37-9608-1ef7db0d5f30`; candidate `4a6125ef-9f93-42f8-b2e4-a68008b3565b`. Instrumentation remains included in CPU.
- Configured-rate model cost estimate: **$0.006042**, against the approved $1 cap. This is not an invoice. Personal ChatGPT-login and managed OpenRouter comparisons were not rerun; their #386 waiver does not satisfy #389's stated comparison prerequisite.

## Fresh account context and capacity recommendation

A new read-only Analytics API query retrieved the last complete UTC day, 2026-11-09. This is the currently deployed account, not the pinned optimized candidate; its deployed source commit and task mix were not established.

| Observed account metric | Value |
| --- | ---: |
| Broker invocations | 2,278 |
| Relay successful invocations | 58 |
| Relay client-disconnected invocations | 16 |
| Total invocations, including disconnects | 2,352 |
| Reported Worker errors | 0 |
| D1 rows read / written | 3,489,910 / 1,992 |
| D1 read / write queries | 43,627 / 934 |
| Existing Broker + Relay database bytes | 3,018,752 |

Fresh GraphQL schema introspection confirmed CPU quantiles are microseconds: successful Broker median 17.741 ms, successful Relay median 14.317 ms, disconnected Relay median 7.953 ms. Medians are not means or monthly CPU totals. Disconnects are not assumed retried; retry multiplicity, peak concurrency, user/task mix, support attribution, and long-term storage growth remain unknown. Account totals include shared/background work rather than only successful photos.

Conditional optimized sensitivity, not measured user capacity: assume 1,440 unchanged scheduled invocations/day; scale other observed invocations. Replace only the measured report-progress lookup's estimated 227,520 reads/day with 5,760. Scale all remaining reads (including unattributed background traffic) and writes at 1×/2×/5×. Other background reads may actually remain fixed, so these are conservative activity sensitivities, not traffic forecasts. Production execution of the pinned lookup is an explicit assumption.

| Scenario | Invocations/day | D1 reads/day | D1 writes/day | Invocations/30 days | Mean CPU fitting Paid's included allowance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Optimized 1× assumption | 2,352 | 3,268,150 | 1,992 | 70,560 | 425.2 ms |
| 2× activity assumption | 3,264 | 6,530,540 | 3,984 | 97,920 | 306.4 ms |
| 5× activity assumption | 6,000 | 16,317,710 | 9,960 | 180,000 | 166.7 ms |

Official limits/prices checked on 2026-11-10:

- [Workers limits](https://developers.cloudflare.com/workers/platform/limits/): Free 100,000 requests/day, 10 ms CPU/request, 128 MB/isolate. Paid has a 30-second default CPU limit, configurable to five minutes; **memory remains 128 MB**.
- [Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/): Paid minimum $5/month includes 10 million requests and 30 million CPU-ms; overage $0.30/million requests and $0.02/million CPU-ms. Logging has separate allowances; no retained-log volume forecast was measured.
- [D1 pricing](https://developers.cloudflare.com/d1/platform/pricing/): Free five million rows read/day, 100,000 written/day, 5 GB total storage. Paid includes 25 billion reads/month, 50 million writes/month, 5 GB; overage $0.001/million reads, $1/million writes, $0.75/GB-month. [Database limits](https://developers.cloudflare.com/d1/platform/limits/): 500 MB/database Free, 10 GB/database Paid.
- [D1 result metadata](https://developers.cloudflare.com/d1/worker-api/return-object/) supplies actual row meters and SQL duration, not network time. [Worker timing fields](https://developers.cloudflare.com/changelog/post/2025-04-09-workers-timing/) supply the deployed CPU/wall distinction used here. Analytics sources: [Workers](https://developers.cloudflare.com/analytics/graphql-api/tutorials/querying-workers-metrics/) and [D1](https://developers.cloudflare.com/d1/observability/metrics-analytics/).

**Recommend Paid, not service reduction.** Even after optimization, the controlled long/large cases and all real candidate photo CPU samples exceed Free's published 10 ms/request allowance. No limit failure occurred in these isolated samples; successful probes are not proof of guaranteed Free headroom, and their instrumentation overhead was not independently subtracted. Current D1 reads already use about 70% of the Free daily allowance; the conditional 2× case exceeds it. Request/write totals and current database size are not the immediate constraint.

Expected starting cost is the $5/month minimum, conditional on monthly CPU/logging/storage remaining included. All three modeled request/D1 totals fit Paid's included amounts. As an explicit CPU sensitivity, 180,000 monthly invocations at a 430 ms mean would cost approximately **$5.95/month**; this is not a measured average, forecast, or upper bound. Provider/model costs are separate. No plan change was made.

Reconsider at sustained CPU approaching the configured Paid limit, 128 MB memory pressure, monthly included-usage boundaries, substantial retention growth, or changed workload/concurrency. Paid does not establish OCR completeness, first display, peak-memory safety, or prerequisite verification. Those evidence gaps remain regardless of quota headroom.

## Focused verification, failures, and cleanup

Verification was deliberately limited to changed contracts and their consumers; runtime input/authentication/admission/accounting protections were not relaxed.

- Passed: `pnpm exec vitest run agent-relay/tests/stream.spec.ts agent-relay/tests/respond.spec.ts agent-relay/tests/photo.spec.ts broker/tests/daily-report.spec.ts` — four files, 58 tests.
- Passed: `uv run --no-sync pytest tests/app/test_agent_chat_streaming.py tests/app/test_agent_chat_relay_contract.py tests/release_evidence/test_agent_chat_simulator_relay_context.py tests/providers/test_photo.py -q`. Captured quiet output did not provide a numeric test count.
- Passed: `pnpm typecheck`, emitted-context parity, focused replay/slow-reader checks, isolated deployed measurements, and the actual-app requests reported above. The latter retain their two partial outcomes.
- Extended existing behavior checks for removal of terminal echoes while preserving authoritative output, and interrupted report recovery with unordered pending/future dates and missing-day catchup. No permanent benchmark framework or incidental implementation-text test was added.
- Not run: full Broker/Relay suites, broad simulator campaigns, native rebuild, unrelated GUI checks, headset display/legibility, fresh personal ChatGPT/OpenRouter comparisons, or a mixed-concurrency/peak-memory campaign. These are not implicitly passed or assigned as routine owner testing. #389 closure remains incomplete where its acceptance requires the missing evidence.
- Probe failures retained: one Windows JSON encoding error; two Node quoting errors before the slow-reader probe executed; one proxy-path 403; one unsupported same-account public-Worker fetch 502 corrected with a synthetic-only service binding; initial PTY trace readiness failures corrected with piped tails; two unauthenticated cleanup-client attempts corrected by attaching headers; and one shell-quoted cleanup scan corrected using direct process arguments. No failed attempt was silently counted as a successful sample.
- The direct photo probe initially left a Pillow handle open, causing cleanup failure after measurements. Process exit released it and the owned runtime was deleted. Its initial native shutdown acknowledgement is unknown; eight other runtime groups recorded acknowledgement. No native probe process remains.
- All four temporary Workers and three temporary D1 databases were deleted. Authenticated follow-up reads returned Worker 404/code 10007 and D1 404/code 7404. Trace collectors stopped, no owned trace process remained, and temporary app/runtime/secret state was removed. Broker and Relay cleanup evidence is in JSON. Only aggregate read-only queries touched existing resources.
- Protected documentation was intentionally unchanged. Pre-existing drift remains in `docs/architecture.md` (personal-Luna-only photo description), `agent-relay/README.md` (missing hosted photo endpoint), and `PRIVACY.md` (missing hosted-image path); these arose from the integrated photo work, not this optimization. No new architecture/privacy behavior was introduced.

## Approved maintenance follow-up — 2026-11-10

The owner approved separating housekeeping from urgent recovery: retain minute-resolution key recovery/revocation, permit approximately one extra hour before physical deletion under a normal schedule, avoid report aggregation without a delivery claim, and skip completed identity-transition scans. This amends the initial cadence deferral, not #286's active-recovery contract.

Implementation: `a55e1b44075bb844ba85217cfcd0e592d58dabc8`. New baseline: `99e3a19ed2958277d9655bc6ac75389c0454603a`, already containing the initial #389 improvements. All follow-up measurements were collected anew. [Follow-up structured evidence](broker-maintenance-efficiency.json) contains source/bundle hashes, every deployed sample, setup meters, simulated-day results, and cleanup proof. Unrelated existing edits to `agent-relay/src/usage.ts` and `docs/photo-managed-comparison.md` were left untouched.

### Implemented behavior

- `handleScheduled` still runs every minute. Stale key-delivery reconciliation and managed-operation recovery retain their budgets, ownership, ACK/auth expiry, six-hour unattended retry eligibility, and sixteen-minute abandoned claims. No cron, quota, concurrency, or request-lifetime setting changed.
- `applyScheduledRetention` now runs only when the **scheduled UTC minute is zero**. It retains all current deletion predicates and pending-report preservation. This covers abuse/request/audit retention, app-active days, abandoned preflight installations, proof nonces, token-rate state, and enrollment records. Logical expiry and request-time window enforcement remain independent of physical deletion. Replay tombstones can remain longer, not become valid again.
- A missed/failed housekeeping run waits for a later hourly opportunity. The approximately one-hour extra residence bound assumes normal successful scheduling; it is not an outage guarantee. No new persistent scheduler/cache or schema was introduced.
- `runDailyReport` acquires the existing report-date lease before aggregation. Without a claim it returns `{ ok: true, sent: false, payload: null }`. No production consumer needs a skipped report preview. Aggregation and webhook failures release the owned lease for the next eligible minute; successful delivery still requires durable completion. Pending-date catchup and overlapping claims remain protected.
- `runNetworkIdentityBackfill` reads authoritative mode first. Confirmed keyed-only mode returns completed/no-work without legacy hook/event scans. Ongoing legacy/dual transition and finalization barriers remain. Raw hooks manually introduced after a completed transition are no longer automatically converted; current operator hooks must use the supported keyed representation.

### New complete-day local execution

The existing SQLite D1 fixture and frozen actual handlers ran all 1,440 scheduled minutes of one synthetic UTC day per arm. Seed: 20 active subjects across 40 dates (800 rows), 300 key-issuance events, 80 delivered report dates, current 33-migration schema, completed identity transition. The next daily report was initially due, not pre-skipped.

| Simulated day metric | Before | After |
| --- | ---: | ---: |
| Scheduled invocations | 1,440 | 1,440 |
| Executed SQL statements | 44,611 | 13,219 |
| Report aggregate SELECTs | 5,740 | 4 |
| SQLite logical row changes | 112 | 112 |
| Successful synthetic report sends | 1 | 1 |

Both arms delivered the identical payload at 00:05 UTC and finished with identical business state: 700 active-day rows, 290 issuance events, and 81 delivered dates. The four retained aggregates are the required once-daily key count and DAU/WAU/MAU. These statement/change counts are **not D1 billing meters**, and the simulated clock is not a 24-hour production soak.

### New deployed D1 and CPU measurements

One temporary Worker/version (`a4004ada-ae03-4511-9f2d-1a818b6345d2`) contained both frozen handlers and used one isolated D1 database, compatibility date `2026-04-08`. Fixture: 600 active-day rows, 300 issuance events, 80 delivered dates, completed identity transition. Ordinary/hourly samples alternated arm order; each arm/case ran three times. Two additional due-report calls, one per arm, restored eligibility before each call.

| Scheduled case | D1 rows read before → after | D1 rows written before → after | Samples per arm |
| --- | --- | --- | ---: |
| Ordinary minute, report already delivered | 1,394 → 20 | 0 → 0 | 3 |
| Hour boundary, report already delivered | 1,394 → 330 | 0 → 0 | 3 |
| Report actually due | 1,394 → 1,080 | 5 → 5 | 1 |

All 14 invocations returned HTTP 200 with platform outcome `ok`; all 14 CPU traces were captured. Quiet-period database snapshots were unchanged. Both due-report calls performed four aggregates, produced the same webhook payload, and persisted `delivered`. External/provider requests were blocked; webhook transport was synthetic, with zero paid model calls.

| Scheduled case | Baseline CPU samples, ms | Candidate CPU samples, ms |
| --- | --- | --- |
| Ordinary minute | 37, 24, 29 | 12, 10, 8 |
| Hour boundary | 27, 24, 32 | 18, 20, 16 |
| Report actually due | 39 | 15 |

CPU is actual platform CPU for an authenticated HTTP wrapper invoking the scheduled handler, including D1 metadata capture, serialization/logging, and synthetic webhook handling. It is **not exact native production cron CPU**. Small samples and shared-runtime warmup do not establish tail confidence or a general latency percentage. SQL duration, network wall time, CPU, and row billing remain separate.

Setup was excluded from workload comparisons: 33 migrations/495 statements read 34,108 rows and wrote 1,248; three synthetic seed statements read 1,076 and wrote 3,841. Eligibility resets and snapshot reads are recorded separately. The product change requires no new migration/index or data deletion beyond the existing housekeeping predicates.

These fixture savings do not establish the same percentage reduction in the previously observed account-wide reads. Production has not received this change, and its full query mix was not newly attributed. Minute Worker invocation count remains unchanged. The Broker-only change does not resolve the earlier OCR CPU, actual-display, or route-comparison gaps, nor independently establish that Free is sufficient.

### Focused verification and operational record

- Passed once after integration: `pnpm exec vitest run broker/tests/daily-report.spec.ts broker/tests/managed-key-delivery.spec.ts broker/tests/puri-token.spec.ts broker/tests/managed-issuance-durability.spec.ts` — **four files, 121 tests**.
- Passed: scoped TypeScript API check using root compiler options with only Broker source/tests included — 112 files, zero diagnostics. Unrelated Relay edits were not included in the check.
- Covered hour/off-hour deletion with fresh state retained, ordinary-minute key recovery, logical expiry/replay boundaries, unavailable aggregation for delivered/in-flight reports, aggregation failure and next-minute retry, pending report preservation, and ongoing/completed identity transition behavior. Removed incidental SQL-count/source-inventory assertions rather than preserving implementation-pinning tests.
- Existing Vite native-config-loader compatibility and Node experimental-SQLite warnings remain. No invoked follow-up check or measurement failed. Full repository/app/native/GUI suites and a new concurrency/peak-memory campaign were not run.
- No production deployment, existing-database mutation, provider invocation, billing change, push, merge, or issue closure. The temporary Worker and D1 database were deleted; authenticated follow-up reads returned 404/code 10007 and 404/code 7404. Trace collector/process and temporary bundle/config/probe state were removed.
- User-visible change is maintenance timing, not UI or photo processing: expired records can remain until the next hourly cleanup, while active recovery remains minutely. Protected documents were not edited. `broker/README.md` still groups scheduled work without describing the new hourly housekeeping distinction; this record supplies that operational detail. `PRIVACY.md` retains the same cutoff rules and “remain until cleanup” wording; physical deletion cadence is clarified here. Existing photo-documentation drift above remains separate.

