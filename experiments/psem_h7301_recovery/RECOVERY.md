# H7301 exact-artifact recovery — coherent outcome (H7301exactartifactrecovery)

Documentation-only owner outcome. No code, source, prior-artifact, or Git mutation was performed to produce this report.

- Baseline: HEAD `83aaed984b8b245082f3ffe7bb15d71f3242361f`, branch `experiment-v2-speaker-change-turn-boundaries-ls`, upstream `origin/experiment-v2-speaker-change-turn-boundaries-ls`, ahead/behind `0/0` (`git rev-list --left-right --count HEAD...@{u}`).
- Worktree preserved as-found: 5 modified (`psem_sortformer_adaptation_depth/frame_alignment.py` + its test, `psem_state_corrected_adaptation_gate/material.py` + its test, `psem_state_corrected_adaptation_gate/README.md`) + 3 untracked experiment dirs (`psem_decision_sufficiency/`, `psem_evidence_to_ownership/`, `psem_repeatability_stage2/`). This report adds only the new dir `experiments/psem_h7301_recovery/` (`RECOVERY.md` + `receipt.json`).
- FIX NOTE (actual report revision time `2026-09-09T10:57:51+00:00`, not pre-frozen): this revision FIXES 6 evidence errors from the prior draft (wrong §1 global heading; release rows 30/all-v*; bundle 35 files; 32-runs/workflows mix-up; artifact-404 recorded as listing success; failed `.gitattributes` command used as scope proof). All 6 are documentation/provenance defects; the finite-blocker conclusion is unchanged. No further pointers — reviewer-confirmed complete.
- User next step authorized: recover exact H + corrected paired regen. No retraining authorized. Material scope only, per user.

## 1. Proven: weights not included in persisted bundle/export

- Results bundle (exact path): `experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/bundle_manifest.json` (+ `README.md`, `training_metrics.json`, `export/gpu_export/gpu_export_manifest.json`).
- Bundle payload = canonical postprocess outputs + deterministic gzip frontier + immutable export manifest + training metrics + numeric NPZ only: **21 NPZ (11 CALIB + 10 DEV)**, verified by bundle `files` map and export-dir listing. Payload contains **no checkpoints, no `.pt`, no `.nemo`**.
- Bundle README states explicitly the export dir "contains no audio, transcripts, PII, `.nemo` files, checkpoints, `.env` files, SSH keys, or process logs."
- Observed frozen provenance in bundle: CPU postprocess `29m18s`, 53 FIT sources, 142 training steps, 11 CALIB + 10 DEV sources.
- Scope guard: the claim is about THIS persisted bundle/export only — not a global "weights were never exported anywhere" claim.

## 2. Recovery target (exact identity)

- GPU last-checkpoint model blob: `/workspace/issue-121/arms/R-H-SC/7301/checkpoints/ami_TS3012c.model.pt` — run_dir `RUN_ROOT_PREFIX (/workspace/issue-121/arms) / R-H-SC / 7301` (`arm_runtime.py:111,229-230`), blob name `{source_id}.{role}.pt` (`arm_runtime.py:553-554`), last of 53 completed FIT sources `ami_TS3012c` after `steps_taken=142` (`training_metrics.json:4-58,27538-27539`; `bundle_manifest.json:21-22`).
- Recovery target head_state digest: `bb5029da54b01a84763d0513a544cdbcd99bb1592f73b9eef71a1916e59aea3f`, recorded in `bundle_manifest.json:21`, `training_metrics.json:27540`, `gpu_export_manifest.json:trained_head_sha256`.
- Digest semantics (existing serializer, under original runtime — NOT a container file hash): `trained_head_digest` builds `{k: v.detach().to("cpu").contiguous()}` from `head_module.state_dict()` and hashes `serialize_blob` output (`h_arm.py:1578-1580`).
- Checkpoint container semantics (`h_arm.py:714-742` `checkpoint_after_source`; `arm_runtime.py:567-613`): `model` blob = `{head_state, pending_grads, accum_count, steps_taken}`; `optimizer`, `scheduler`, `rng` are **separate** blobs. Do not mix optimizer/sched/rng bytes into the head identity check.
- Frozen-base proof (head-alone + base is weights-sufficient; backbone was NEVER adapted): `freeze_backbone_train_head` sets wrapper eval + `requires_grad_(False)` and raises unless zero wrapper-trainable and non-zero head-trainable (`h_arm.py:649-677`); `training_metrics.json:196-202` mode_audit `frozen_trainable_count=0, head_trainable_count=10, frozen_representation_ok=true, sortformer_eval=true, psem_head_train=true`. Native `hidden192/logits4` features derive from the FROZEN base; the trained artifact is head-only residual.
- Final-flush question resolved NEGATIVE: `training_metrics.json` records `accum_count=0`, `steps_taken=142` (= total); reviewer source check of the train loop/finalize found no post-loop `optimizer.step` before the digest. No pending-accum flush exists after the last per-source checkpoint; no other checkpoint is invented. Recovery target `ami_TS3012c.model.pt` → `bb5029…` stands.
- Base checkpoint present: sha `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8` (`bundle_manifest.json:12`, canonical/gpu-export bindings `checkpoint_hash`).
- Cache status: 53-source FIT evidence cache is optional inference memo, **not** DEV-comparable and **cannot** substitute for or repair the head. `hidden192/logits4` native features must be recomputed from the FROZEN base + exact head at correct source extent (see §5), never inferred from old truncated/sparse-eval frames.

## 3. Lineage search — actual NEW read-only commands/results (no repeat)

Prerequisite lanes DONE via `history://restore-artifact-lineage` and `history://restore-regeneration-contract`; the commands below are that lane's NEW searches, recorded here — not re-run. Prior local / known-archive / auth-pod / volume searches (`[]`) were accepted as prior truth and NOT re-run. Commands are quoted exactly as executed per the lineage session record; where the transcript truncates a compound tail (`…`), only the preserved prefix is quoted and the gap is marked — never reconstructed.

1. `git rev-parse HEAD && git branch --show-current && git status --porcelain=v1 && git remote -v` → baseline HEAD/branch/dirty paths (§1 baseline).
2. `gh release list --limit 30 2>&1; echo "---RELEASE_VIEW---"; gh release view --help 2>&1 | head -20` → returned **29 rows** (NOT 30), including the non-`v*` prerelease `ASR Model: Qwen3 ASR 0.6B INT8 Sherpa`. **ONLY latest-release asset contents inspected**: `v2.6.1` with two assets (`PuriPulyHeart-Setup-2.6.1.exe` 172344005 B + soxr 268189 B). No head pointer. The other 28 releases' asset contents were NOT inspected — no claim about them.
3. `git log --oneline --all -- experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/ 2>&1 | head -30; echo "---"; git log --oneline -5 2>&1` + `git show --stat --oneline becb359a 2>&1 | head -60; echo "===ANCESTOR_CHECK==="; git merge-base --is-ancestor becb359a HEAD && echo "becb359a_is_ancestor_of_HEAD" || echo "not_ancestor"` → bundle path history is exactly 1 commit (`becb359a`, ancestor of HEAD). Bundle tree = **33 tracked files at becb359a AND at HEAD** (`git ls-tree -r --name-only becb359a -- <bundle> | wc -l` = 33; `git ls-files <bundle> | wc -l` = 33), **zero `.pt`** in both; `bundle_manifest.json` `files` map = **32 entries** (manifest excludes self); whole commit `becb359a` = **77 files changed** (NOT "bundle 35"). `.pt` files DO exist in unrelated git history (e.g. commit `37f24174` experiment #117 probe) — no global "no `.pt` history" claim; scoped fact is zero `.pt` in bundle path history.
4. `gh issue view 121 --comments 2>&1 | tail -100` (+ full issue #121/#132/#142 reads) → links/discussion contain **no head-state pointer**.
5. `gh run list --limit 30 2>&1; echo "===WORKFLOWS==="; gh workflow list 2>&1` → **30 run rows** (PR CI / Release / Deploy / Broker Direct / Maintenance / Dependency Graph activity), NOT 32; no H7301 persistence lane. The "32" figure is the Actions **ARTIFACT-name** listing (item 6), whose names include `release-artifacts` and `puripuly-heart-broker-pre-migration-*` — artifact names, NOT workflow names and NOT 32 runs. Artifact contents were never fetched.
6. `gh api repos/kapitalismho/PuriPuly-heart/actions/artifacts --paginate -q ".artifacts[] | [.name, .size_in_bytes, .expired, .created_at] | @tsv" 2>&1 | head -50` (compound tail after `===TOTAL===` not preserved in transcript) → 32 artifact-name rows, contents not downloaded/inspected. Follow-up `gh api repos/kapitalismho/PuriPuly-heart/actions/artifacts/172168681 --jq "{name:.name,size:.size_in_bytes,expired:.expired,workflow:.workflow_run}" 2>&1` → **HTTP 404** (`{"message":"Not Found","documentation_url":"https://docs.github.com/rest/actions/artifacts#get-an-artifact","status":"404"}` + `gh: Not Found (HTTP 404)`): that single artifact ID is absent via REST — reason unknown, expiry NOT claimed. Separately, the latest-release asset view (`echo "===RELEASE_ASSETS==="; gh release view …`, tail not preserved) showed the v2.6.1 two assets in item 2.
7. `git show becb359a:experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/.gitattributes 2>&1` → **FAILED**: `fatal: path 'experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/.gitattributes' does not exist in 'becb359a'`. Logged explicitly and OMITTED as scope proof. Actual `.gitattributes` lives at the experiment root (`experiments/psem_state_corrected_adaptation_gate/.gitattributes`). Omission proof instead: manifest 32-entry `files` map + the zero-`.pt` `ls-tree`/`ls-files` counts in item 3.
8. `git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>&1; git status -sb 2>&1 | head -5; git rev-list --left-right --count HEAD...@{u} 2>&1` → upstream `origin/experiment-v2-speaker-change-turn-boundaries-ls`, `## branch...origin/branch`, `0/0`.
9. PodID / storageURI: **not present in reachable lineage** (code manifest paths are `/workspace/...` run-root paths, not a Pod ID or volume URI). No value invented.

**Scoped claim (exact): `no recovery pointer found in checked locations`. This is NOT a global-unrecoverable verdict and NOT a claim that all release assets / Actions artifact contents are absent — uninspected contents remain uninspected.**

## 4. Blocker

- Missing actual binary blocks both downstream stages: the exact head-state blob (`ami_TS3012c.model.pt` → digest `bb5029…`) is unavailable in-repo and unpointed-to in checked lineage locations.
- No hiding behind authorization: the block is the missing binary, not approval paperwork. No arbitrary retraining is requested or authorized; material scope only, per user.

## 5. Frozen future contract — conditional, NOT scaffold/implementation

From `history://restore-regeneration-contract` (sibling lane; no overlapping search). Applies **only IF** the exact head in §2 is recovered. Nothing below is built now:

- Validate restored head by identity: recompute `trained_head_digest` under the pinned runtime and require exact `bb5029…`. Bind pinned NeMo `1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6` + base `8abd…` + source code **NEW** binding (preserve old bindings `a3d9003a…` / `7ff72366…` as history, never reuse as identity).
- Regenerate from source 0 with native features from the FROZEN base + exact head: full-source `hidden192/logits4` per episode, whole-episode oracle refmap, GRU over full DEV prefix (not clipped); corrected native-coverage checks read source END, not sparse eval frame counts.
- Minimal source set preserving original F0/H comparability: the **ten DEV** sources in full original extent; no FIT/CALIB retrain.
- Existing `run_h_arm --command run` retrains, so it is **forbidden** for this path; the frozen head-infer driver is built **only AFTER** the artifact is available — no unused stub now.
- Runtime parity classification: original CUDA image `sha256:20f44b72f748cdd755b0ff0dcb74de40fc6ee996e9a9ecd263c41598fdd746b8` (A40, dependency lock) is the numeric-identity lane. CPU can mint a NEW numerical binding but **cannot** claim original-runtime parity. Hardware runtime is not currently available; the execution lane must be resolved when the artifact is recovered (paid Pod requires explicit approval — no demand now).
- Full corrected-F0 alone is possible work but is **not** a substitute for H and does not change this decision.
- Correctness gates may reuse the unchanged CALIB binding only with state-exact verification; never auto-reuse old DEV scores.
- Remaining two stages stay **BLOCKED** on the actual head. No regeneration success is claimed. Current report outcome = complete investigation + blocker proof.

## 6. Review-candidate pointer

- Latest review receipt (by `run_utc`): `experiments/psem_decision_sufficiency/receipt.json`, `run_utc=2026-09-09T10:39:44+00:00`, `stable_hash=304f2ada44a423258e06328b98c8a4ec5469c0fcfe8fac95c2eaf32ccfafac55` (vs `experiments/psem_repeatability_stage2/receipt.json` `run_utc=2026-09-09T09:25:44+00:00`, `stable_hash=75407280dd1ffb1d3d40750086a70f49613c189e692b3a7aeed24e4797f03f1d`).
- Worker state after writing `RECOVERY.md` + `receipt.json`: idle. Do not publish/commit.
