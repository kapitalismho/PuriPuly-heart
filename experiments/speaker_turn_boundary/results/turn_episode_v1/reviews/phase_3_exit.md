# Phase 3 exit review — provider-neutral logical-action oracle

Status: **accepted**.

| Item | Value |
| --- | --- |
| Candidate | `a6403172451b02944e569a9bd94097387aa3adc0` |
| Reviewer | `/root/phase3_exit_review` |
| Verdict date | 2026-08-09 |
| Authority SHA-256 | `ecd16c765072504f9970cb17fd5cba4f9967715954adead6ebc386fffb8d4f8c` |
| Accepted pre-execution bundle SHA-256 | `8dbcd4333297fa1dbc8b26a3ff4d9f0c708a0811588517b91184cabf20d17d36` |
| Integration target | `origin/main` at `848aa0b9f1b35388ded5a250d51a687223eac1c5` |

## Verdict

`accepted` with no material findings.

Fresh independent verification completed with zero mismatches over 82,026 detail rows,
124,803 oracle actions, and 168,903 lifecycle finalizations. The repaired stateful
`A -> B -> A` contamination sentinels match exactly: delay-250 shard line 8 has 42,784
baseline contaminated samples, and `d250:o-500:h0` has 1,168,512 baseline contaminated
samples. All fully recoverable actions match ideal source-span ownership, every exact
safe-frontier row validates, and the B0/B1 no-detector seed passes 186/186 episodes.

The public artifact verifier rejects all five required mutations. All seven deterministic
gzip shards remain below 20 MiB with exact counts and byte hashes. The 2,277,473-byte main
artifact self-validates, Black and Ruff pass, all 14 Phase 3 tests pass, and the complete
experiment test suite passes.

| Artifact identity | SHA-256 |
| --- | --- |
| Main JSON bytes | `be44a6a7764cff4c01064bc506c1d29ab6b4f35dbb48797409e68a610fea82db` |
| Main JSON canonical content | `3bdf923af366a70bbffc4e8f7ef92eb992e086cfa055c12f2bb883452a47eb19` |
| Verification JSON bytes | `83d7ad3f31777a907c5f1b810259e9e7994f8388f5dc2434c8e26f5477bd31e5` |
| Verification canonical content | `cab04fade1a8b1f33a9ac3c372807679e361b514d409b7f8d667636658e6f1cf` |
| Recomputed grid aggregate | `8f1f2c7482c2c317d9573c30081d446deac2587abdd371073629240957d157cf` |

The accepted range changes only Phase 3 experiment code, tests, and artifacts. It does
not modify production or public entrypoints and does not access confirmatory held-out or
provider paths.

Acceptance closes only the Phase 3 exit gate. It authorizes Phase 4 preparation and its
fresh pre-execution review, not Phase 4 execution, held-out access, provider calls,
merge, push, deployment, or product-contract changes.
