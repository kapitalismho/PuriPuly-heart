# Phase 4 exit review — bounded signal diagnostics

Status: **accepted**.

| Item | Value |
| --- | --- |
| Coherent review base | `ad9797ca2acffda819ffc9adec319555b72cf2a0` |
| Candidate | `5edfa67f7bb73c352b15459fdde018b196b5b5ac` |
| Scientific result commit | `c9597420a325983cc19066bdb53c6b9191724b4f` |
| Reviewer | `/root/phase4_exit_review` |
| Verdict date | 2026-08-10 |
| Authority SHA-256 | `ecd16c765072504f9970cb17fd5cba4f9967715954adead6ebc386fffb8d4f8c` |
| Accepted pre-execution bundle SHA-256 | `a6afa3dc946815c162ee18d09b1c7ad3ad08e252f7286c110b37a685fe2b1759` |
| Integration target | `origin/main` at `848aa0b9f1b35388ded5a250d51a687223eac1c5` |

## Verdict

`accepted` with no material findings.

The repaired range is coherent and limited to 24 Phase 4 result files. The execution
receipt's self-hash and committed blob match. Its three authorized suspension intervals
independently recompute to 54,651.0544049 wall seconds, 20,194.8380007 suspended seconds,
and 34,456.2164042 active seconds. The run completed normally in one attempt with zero
recovery, inside the accepted 16-hour active-time ceiling.

Independent verification reports zero mismatches and rejects all eight required coherent
mutations. All 23 completion receipts match their exact byte hashes and sizes, all 14 gzip
shards reopen to exactly 218,700 signal rows, the largest aggregate is 4,826,455 bytes, and
the largest detail shard is 2,194,903 bytes. The cache occupies 1,814,144,284 bytes, below
the 8 GiB ceiling; maximum recorded RSS is 5,936,222,208 bytes, below the 16 GiB ceiling.

The accepted mechanical disposition is:

- ERes2NetV2: `signal_go`, with 9 `eligible_go`, 5 `eligible_uncertain`, and 60
  `not_estimable` primary extractors. Phase 5 may evaluate the full predeclared policy grid.
- LS-EEND: `signal_stop`, with all 12 primary extractors `eligible_stop`. Phase 5 is limited
  to B0/B1, the raw diagnostic, and the no-neural control.

| Artifact identity | SHA-256 |
| --- | --- |
| Completion JSON bytes | `368a5c23a30e10f1884fd3797166b23ee93df0a1d0f84fc7006010b17fdec565` |
| Completion canonical content | `db75772938fc4a59f21784e9fbc279ad3003bffc72b32594d7844fec8a28f14c` |
| Verification JSON bytes | `dda1f1c1d9f51e9eec919e31f31635f07d22e0e84369cf87d6a755186ab12740` |
| Verification canonical content | `f8ba0e6498d2bc6d87854b6bdaefb5f7f15a7263ea9f98c399cd8b56d8bab51c` |
| Execution receipt JSON bytes | `f1e1472c7a02afe9a7fc05dbe4ae609a21a1522ce59487e1c3daf87086644c93` |
| Execution receipt canonical content | `58e5c229d618832b729bac5597686db20d2373ed80150520c400561fb03d229a` |
| Signal disposition JSON bytes | `f9e799bd1f78aa45b25cb913928584ba0d63d07cfbf6978495d387da2ff5a9aa` |
| Signal disposition canonical content | `669f6d4200832816b7beee03161cac2b97ec2594af00bb78df878766081bc5bf` |

Ruff and Black pass, all 45 focused Phase 4 tests pass, and all 340 tests across the 28
speaker-turn-boundary test files pass. The accepted range contains no production,
public-entrypoint, provider, credential, network, held-out, or Phase 5 path.

Acceptance closes only the Phase 4 exit gate. It authorizes Phase 5 preparation and its
fresh pre-execution review, not Phase 5 execution before that review, held-out access,
merge, push, deployment, production wiring, credential access, or provider calls.
