# Chatbot benchmark

Development-only synthetic Agent Chat measurements. Run from the repository root in the locked project's prepared Python environment; the benchmark is not an installed application command or a production dependency. Application owners still define context and tool execution; [Agent Relay](../../services/agent-relay/README.md) owns emitted prompts and model requests.

Updating a populated checkout may leave ignored `__pycache__` files in the retired simulator directory. Inert cache/namespace residue is not the old command or a compatibility shim; the maintained entrypoint is `benchmarks.chatbot`. Do not retain runnable source there or delete unrelated state to satisfy a directory-existence check. Preserve private reports separately from disposable bytecode.

## Controlled offline smoke

Set `PRIVATE_RESULTS` to an existing private directory outside every checkout, using an absolute path. Always pass explicit output paths: CLI defaults are relative to the working directory. Keep reports, conversation archives, calibration samples, owner judgments, and rendered HTML out of tracked/public artifacts.

```sh
uv run --no-sync python -m benchmarks.chatbot --help
uv run --no-sync python -m benchmarks.chatbot smoke --output "$PRIVATE_RESULTS/smoke.json"
```

Without `--relay`, smoke uses controlled responses and HTTPX transport without network, a Worker, real credentials, private settings, hardware, or a GUI. Actual application owners, context, tools, adapters, and bundle creation execute against synthetic state. `controlled_smoke_completed` with `live_pass=false` is wiring evidence, not live-model quality or calibrated success. Use `smoke --help` for selecting cases or the separately configured emitted-Worker wiring mode.

## Trusted Relay context

Live evaluation requires a trusted-context manifest from the exact measured emitted Relay build, not a hand-written manifest or source-file concatenation. The [offline exporter](../../services/agent-relay/operations/export-context.mjs) observes controlled outbound requests from that bundle, records module/migration hashes, and rejects external requests. It uses temporary synthetic local D1, never production D1. Follow [Relay development prerequisites](../../services/agent-relay/README.md#local-development) for the Linux-native Node workspace.

From the repository root, bundle the intended configuration with `pnpm --filter @puripuly-heart/agent-relay exec wrangler deploy --dry-run --config wrangler.jsonc --outdir "$PRIVATE_RESULTS/relay-bundle"`. This does not deploy. Then export:

```sh
node services/agent-relay/operations/export-context.mjs --emitted "$PRIVATE_RESULTS/relay-bundle/index.js" --output "$PRIVATE_RESULTS/relay-context.json" --compatibility-date 2026-04-08 --compatibility-flag enable_request_signal
```

Supply each actual compatibility flag with `--compatibility-flag`; read the measured configuration rather than inventing a date or flags. The exporter refuses an existing output file. Its `relay_build` is the emitted entry's SHA-256, not a Git commit. Pass that same identity as `--relay-build` together with `--relay-context` to the simulator. A local export does not prove a remote staging Worker is running those bytes; establish the measured deployment's identity separately. Never substitute a production endpoint or deploy as part of directory-maintenance verification.

## Live-model evaluation

Live access is separate from smoke. Use benchmark-only ChatGPT sign-in and pass `--subscription-only` to every live `probe`, `run`, and `calibration-rejudge`. Stop if subscription access is unavailable; do not fall back to API-key billing. Credentials belong in the existing benchmark sign-in/keyring path, never in reports or command arguments.

`run` additionally requires an explicit isolated Broker origin, Relay endpoint/build, matching trusted-context manifest, and `--isolated-staging`. That confirmation covers synthetic installation/token authority, D1/R2, and support destination, not just a different URL. See each subcommand's help for its current arguments, model policy, case selection, and concurrency controls. Do not use private app state as synthetic input.

```sh
uv run --no-sync python -m benchmarks.chatbot probe --subscription-only --output "$PRIVATE_RESULTS/access.json"
uv run --no-sync python -m benchmarks.chatbot run --subscription-only --isolated-staging --broker <isolated-broker-origin> --relay <isolated-relay-endpoint> --relay-build <emitted-entry-sha256> --relay-context "$PRIVATE_RESULTS/relay-context.json" --output "$PRIVATE_RESULTS/report.json" --calibration-samples "$PRIVATE_RESULTS/samples.json"
```

Neither successful access nor completed conversations establish a calibrated pass. Preserve blocked results, including `blocked_calibration`, missing prerequisites, failed attempts, and their evidence limits. The [runner](runner.py), [context validation](relay_context.py), and [judging contract](judging.py) own current semantics; this relocation introduces no calibration policy.

## Calibration and saved reports

Freeze the first 20 eligible conversations once, retain their IDs/digests, and obtain actual owner judgments. `--calibration-samples` above performs the initial export; use the standalone export only if that frozen selection does not already exist. Never invent owner scores or overwrite the frozen selection to improve agreement.

```sh
uv run --no-sync python -m benchmarks.chatbot calibration-export --report "$PRIVATE_RESULTS/report.json" --output "$PRIVATE_RESULTS/samples.json"
uv run --no-sync python -m benchmarks.chatbot calibration-compare --samples "$PRIVATE_RESULTS/samples.json" --owner "$PRIVATE_RESULTS/owner.json" --output "$PRIVATE_RESULTS/comparison.json" --report "$PRIVATE_RESULTS/report.json"
uv run --no-sync python -m benchmarks.chatbot calibration-rejudge --subscription-only --samples "$PRIVATE_RESULTS/samples.json" --output "$PRIVATE_RESULTS/rejudged.json"
uv run --no-sync python -m benchmarks.chatbot render-html --report "$PRIVATE_RESULTS/report.json" --output "$PRIVATE_RESULTS/report.html"
```

Comparison imports owner verdicts and checks that samples belong to the original report before attaching calibration. Rejudging preserves the frozen conversation IDs/digests and writes updated judge verdicts to a separate file; it never replaces the original selection. For a revised comparison, use that output as `--samples`, not the stale original judgments:

```sh
uv run --no-sync python -m benchmarks.chatbot calibration-compare --samples "$PRIVATE_RESULTS/rejudged.json" --owner "$PRIVATE_RESULTS/owner.json" --previous "$PRIVATE_RESULTS/comparison.json" --output "$PRIVATE_RESULTS/recomparison.json" --report "$PRIVATE_RESULTS/report.json"
```

Actual owner verdicts must cover the current criteria; rejudging resets outdated owner templates to unanswered entries rather than inferring scores. Rendering consumes saved results offline and does not upgrade their status or evidence scope.

For actual application effects, approval/cancellation, or GUI presentation, use the [chatbot product-area verification map](../../docs/verification.md#chatbot) and follow the same result into the affected application owner/surface. Neither simulator mode proves those effects.
