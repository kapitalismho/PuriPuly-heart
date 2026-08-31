# Issue 107 detached execution control plane

The scientific commands remain owned by `experiments.psem_sortformer_adaptation_depth.run`. The detached runner only sequences argv arrays, verifies exact inputs and generated outputs, writes durable state, and stops at operator decision boundaries. It never contains `RUNPOD_API_KEY`. The `run`, `decide`, and `status` commands reject an inherited key before reading or mutating durable run state; only the isolated `self-test` command temporarily controls that variable to verify the refusal paths.

## Persistent layout

The mounted volume must provide at least 30,000,000,000 total bytes at `/workspace`. Assets must use this layout:

```text
/workspace/issue-107/assets/checkpoints/...
/workspace/issue-107/assets/corpus
/workspace/issue-107/assets/reference
```

For `persistent_root=/workspace` and `run_id=<id>`, the runner creates:

```text
/workspace/issue-107/runs/<id>/
  control/run_config.json
  control/state.json
  control/heartbeat.json
  control/events.jsonl
  control/phase-complete/*.json
  control/decision.json
  control/decisions/*.json
  receipts/
  logs/
  output/
  protocol-registry/
```

`state.json` uses `STARTING`, `RUNNING`, `WAITING_FOR_DECISION`, `COMPLETED`, or `ERROR`. `heartbeat.json` is atomically replaced immediately and every 15 seconds while the runner owns the run. Every completed phase records command, input, output, stdout, and stderr hashes. A pre-existing required output or phase log fails closed instead of being reused. If a heartbeat write fails while a scientific child is active, the runner terminates the child's process group, records `ERROR`, and exits instead of allowing unobservable GPU work to continue.

## Run configuration

Generate the immutable schema-v1 config inside the exact derived image that will run the Pod. This is required because the generated phase argv serializes `sys.executable`; it must be the Pod interpreter, not a host or substitute environment interpreter. The generator requires all runtime paths and billing controls, fixes the container identity, rejects a projected deadline cost above USD 30, and requires the timezone-aware billing start to be no later than the current time with its maximum-runtime deadline still active. It validates the result with `detached_phase_runner.validate_config`, writes canonical JSON atomically, and refuses to overwrite an existing output. `/opt/nemo` is the expected checkout location in the prepared image and is passed explicitly; path validation itself accepts any absolute checkout path.

Phase execution requires `persistent_root=/workspace`. Every phase fails closed unless `/workspace` reports at least 30,000,000,000 total bytes and the current free bytes meet the configured reserve. `--minimum-free-bytes` cannot be lower than 8 GiB (`8589934592` bytes), but 8 GiB is only the code floor. Before a live launch, retain the mounted-volume capacity receipt and choose a conservative operator reserve at or above that floor based on the observed capacity and expected outputs; a higher value is accepted.

The timestamps below are a concrete matching example: billing starts at `2026-08-31T12:00:00+00:00`, the authorized runtime is 20 hours, and the watchdog deadline is therefore exactly `2026-09-01T08:00:00+00:00`.

```bash
python -m experiments.psem_sortformer_adaptation_depth.issue_107_launch write-config \
  --run-id issue-107-a40-20260831-01 \
  --persistent-root /workspace \
  --repository-root /workspace/repo \
  --checkpoint /workspace/issue-107/assets/checkpoints/diar_streaming_sortformer_4spk-v2.1.nemo \
  --corpus-root /workspace/issue-107/assets/corpus \
  --reference-root /workspace/issue-107/assets/reference \
  --nemo-checkout /opt/nemo \
  --image-identity sha256:14acbef50fa15281bded1d3fbbcd8029091aeba0692d5647255aa5b90eff8ca7 \
  --hourly-price-usd 0.44 \
  --hourly-price-source "RunPod deployment price recorded by the operator" \
  --billing-started-at "2026-08-31T12:00:00+00:00" \
  --max-runtime-hours 20 \
  --minimum-free-bytes 8589934592 \
  --output /workspace/issue-107/run-config.json
```

The generated graph is exact:

```text
bootstrap-f0
  command: <sys.executable> -m experiments.psem_sortformer_adaptation_depth.issue_107_launch bootstrap-f0
  required output: receipts/bootstrap-f0-summary.json
  next: h-head-material-and-dev

h-head-material-and-dev
  command: <sys.executable> -m experiments.psem_sortformer_adaptation_depth.issue_107_launch run-arm --arm H-HEAD
  required output: receipts/h-head-material-and-dev-summary.json
  next: t2-top-material-and-dev

t2-top-material-and-dev
  command: <sys.executable> -m experiments.psem_sortformer_adaptation_depth.issue_107_launch run-arm --arm T2-TOP
  required output: receipts/t2-top-material-and-dev-summary.json
  gate after-h-t2-dev:
    open_ta -> ta-all-temporal-material-and-dev
    select_candidate -> terminal
    stop -> terminal

ta-all-temporal-material-and-dev
  command: <sys.executable> -m experiments.psem_sortformer_adaptation_depth.issue_107_launch run-ta
  required output: receipts/ta-all-temporal-material-and-dev-summary.json
  gate after-ta-dev:
    select_candidate -> terminal
    stop -> terminal
```

The generated topology and phase-specific command prefixes are exact. Each phase command then appends this exact common argv suffix, in this order, with `<phase-id>` replaced by that phase's identifier:

```text
--run-id <run-id>
--persistent-root <absolute-persistent-root>
--repository-root <absolute-repository-root>
--checkpoint <absolute-checkpoint>
--corpus-root <absolute-corpus-root>
--reference-root <absolute-reference-root>
--nemo-checkout <absolute-nemo-checkout>
--image-identity sha256:14acbef50fa15281bded1d3fbbcd8029091aeba0692d5647255aa5b90eff8ca7
--hourly-price-usd <positive-usd-per-hour>
--hourly-price-source <nonempty-price-source>
--billing-started-at <timezone-aware-ISO-timestamp>
--max-runtime-hours <positive-hours>
--minimum-free-bytes <bytes-at-or-above-8589934592>
--phase-summary {run_root}/receipts/<phase-id>-summary.json
```

Each generated phase has an empty `required_inputs` array because scientific receipt hashes do not exist when the immutable config is created. The launcher invokes only supported `experiments.psem_sortformer_adaptation_depth.run` commands, which revalidate each dynamic material, checkpoint, DEV-result, staged-state, cost, and TA-authorization handoff. The scientific command order is unchanged.

Every phase summary binds the immutable config hash, exact subprocess argv and argv hashes, consumed artifact hashes, and every scientific file newly generated or changed by that phase under the run root or Git-common authority registry. Its `storage_evidence` is a list of labeled UTC snapshots. Bootstrap records `phase_start`, `before_build_lineage`, and `before_f0_inference`; every arm records `phase_start`, `before_canary`, `before_smoke`, `before_material_validation_and_training`, and `before_dev_inference`. Capacity and free reserve are checked at every snapshot, so reserve is rechecked between expensive commands.

Before phase work, the launcher snapshots existing authority records. Existing exact content-addressed records are allowed: `register_execution` accepts an identical record at its digest path, and `require_registered_execution` validates the exact receipt. Only authority records newly added or changed relative to the pre-phase snapshot appear in that phase summary. A digest collision or mismatched receipt still fails closed. The dynamic material-input JSON and phase summary are canonical, atomic, and content-bound. Generator, material-input, and summary outputs remain on persistent storage outside the worktree; the scientific CLI retains ownership of its required Git-common authority records.

Every phase receives the fixed environment `PSEM_CONTAINER_IMAGE_IDENTITY=sha256:14acbef50fa15281bded1d3fbbcd8029091aeba0692d5647255aa5b90eff8ca7`, `PSEM_SORTFORMER_NEMO_PATH`, `PSEM_CORPUS_ROOT`, `PSEM_REFERENCE_ROOT`, `PSEM_ADAPTATION_OUTPUT_ROOT={run_root}/output`, `PSEM_PROTOCOL_REGISTRY_ROOT={run_root}/protocol-registry`, and `CUDA_VISIBLE_DEVICES=0`. Neither `RUNPOD_API_KEY` nor `PSEM_ALLOW_EVAL` is permitted in the Pod environment.

## Detached runner

Start from the repository root in the Pod with the same image interpreter serialized into the config:

```bash
nohup python -m experiments.psem_sortformer_adaptation_depth.detached_phase_runner run \
  --config /workspace/issue-107/run-config.json \
  > /workspace/issue-107/runner-launch.log 2>&1 < /dev/null &
```

The runner exits normally at each decision gate. The Windows watchdog sees `WAITING_FOR_DECISION` and stops the Pod. Inspect `state.json`, phase completion receipts, logs, and scientific receipts before deciding.

Record a decision while the volume is mounted in a later Pod or SSH session:

```bash
python -m experiments.psem_sortformer_adaptation_depth.detached_phase_runner decide \
  --run-root /workspace/issue-107/runs/issue-107-a40-20260831-01 \
  --gate after-h-t2-dev \
  --action open_ta \
  --rationale "H and T2 DEV receipts reviewed; TA cost remains within the hard stop"
```

Restart `run` with the same immutable config to consume the decision. `open_ta` here only selects the configured TA phase; the scientific phase must still include and validate the explicit `run.py open-ta` authorization receipt. `select_candidate` and `stop` terminate this detached sequence without opening TA.

An interrupted `RUNNING` phase becomes `ERROR` when recovery is attempted. Discard that run-local partial phase and restart the affected arm from step zero under a new run ID. `--replace-stale-lock` only removes a crash-left lock after the operator has confirmed no runner remains; it does not make interrupted scientific work resumable. A nonempty Git-common authority registry is not itself a restart blocker: exact pre-existing records may be reused through the content-addressed registration checks, and newly added or changed records are captured by the new phase summary. Run-local receipts, outputs, protocol records, logs, and phase summaries remain no-overwrite boundaries and must not be replayed.

## External Windows watchdog

Keep `RUNPOD_API_KEY` only in the local Windows environment. A live watchdog refuses to arm without it. If the key disappears before stop confirmation, the watchdog records `stop_failed` with a sanitized failure code. The watchdog uses short `scp` connections to fetch two JSON files; it does not hold a continuous SSH session and does not transmit the API key to the Pod.

For live operation, pass the exact UTC sum of the generated config's `billing-started-at` and `max-runtime-hours`. In this matching example, `2026-08-31T12:00:00+00:00 + 20 hours = 2026-09-01T08:00:00+00:00`:

```powershell
$env:RUNPOD_API_KEY = "<local secret>"
& scripts\experiments\watch-issue-107-a40.ps1 `
  -PodId "<pod-id>" `
  -RunPodCliPath ".cache\runpodctl-windows-amd64.exe" `
  -ReceiptPath ".cache\issue-107-watchdog-receipt.json" `
  -SshTarget "root@<pod-host>" `
  -SshPort <tcp-port> `
  -IdentityFile "$HOME\.ssh\id_ed25519" `
  -RemoteRunRoot "/workspace/issue-107/runs/issue-107-a40-20260831-01" `
  -AbsoluteDeadlineUtc "2026-09-01T08:00:00+00:00"
```

`-AbsoluteDeadlineUtc` must include `Z` or an explicit offset, is normalized to UTC, is the only deadline when supplied, and must be in the future before live arming. Its receipts use `deadline_source=absolute_utc`. If it is absent, the backward-compatible deadline is armed time plus `-MaxRuntimeHours`, recorded as `deadline_source=armed_at_plus_max_runtime`; that mode is retained for local fixtures and older invocations.

The watchdog triggers a stop immediately for `WAITING_FOR_DECISION`, `ERROR`, or `COMPLETED`; after the 180-second initial grace it also triggers for a heartbeat older than 90 seconds or unreachable control files. The deadline always triggers a stop. A live `runpodctl pod stop` exit code of zero is only a request, not completion. The watchdog remains in `Stop-WatchedPod`, repeatedly issuing sanitized stop requests and querying `pod get` after `PollSeconds` delays across transient nonzero responses. It writes `stop_retrying` while the runtime is nonterminal and emits `stop_confirmed` only after `pod get` reports `stopped`, `exited`, or `terminated`, case-insensitively. The confirmation receipt includes a timestamp and attempt count. Receipts contain only selected status fields and exit codes, never CLI output or API credentials.

Validate behavior without contacting RunPod by pointing `-LocalControlDirectory` at fixture control files and using `-DryRun -Once`. Dry-run returns `would_stop` and remains finite. Fixtures may set `-InitialGraceSeconds 0` or `-MaxRuntimeHours 0` to exercise unreachable, stale-heartbeat, and fallback-deadline branches immediately; a past timezone-aware `-AbsoluteDeadlineUtc` can exercise the absolute-deadline branch in dry-run. Live mode still enforces an initial grace of at least 30 seconds and, when no absolute deadline is supplied, a maximum runtime of at least 0.1 hours.

Run the Python control-plane smoke check with:

```powershell
python experiments/psem_sortformer_adaptation_depth/detached_phase_runner.py self-test
```

The smoke check covers the decision resume path, configured and inherited API-key rejection, stale output replay rejection, interrupted `RUNNING` rejection, and injected heartbeat-write failure with child termination and durable `ERROR` state.
