# Next Decision (ontology comparison, frozen diagnostic + wait gate)

Diagnostic complete. No adoption, no production change, no H training, no waiting-policy change.

## Decided

- NP (3 independent cases, per-word states read directly from ledger `paired_actualF0.rows`, not inferred triplets): SIMPLE == RICH exactly (NP1 7c0w1u: gross 3 fixes A1118/1119/1120 wrong->correct + 1 wrong withheld A1117, net +3; NP2 4c0w3u1m: gross 1 fix C1480 wrong->correct + 1 correct withheld B1653 + 1 wrong withheld C1481, NET correct +0 not gross 0; NP3 6c0w2u: gross 2 fixes B323/324 + 2 wrong withheld B322/325, net +2; both arms). No ontology win on NP under shared validity. No wait advantage observed within the available pre-terminal record (gate: +100/+300 ms adds zero applicable events both ontologies; extended +20000 adds zero distinct; margins 2.017/2.097/2.440 s); not a generic claim about waiting ever.
- Guards (where overlap bounds exist): SIMPLE trades UNKNOWN for CURRENT and strictly adds other-wrong vs RICH (R1: RICH 7c0w3u = 0 fixes, B32 wrong->UNKNOWN plus A492/A493 correct->UNKNOWN withheld, not a B32 fix; SIMPLE 9c1w0u retains 1 wrong; T1 13c5w3u vs 7c0w14u = +6 correct +5 wrong -11 unknown; COMBINED 16c7w5u vs 10c2w16u same tradeoff, not summed). R2 guard and singles identical; BC1 guarded no-event.
- Outcome is both-fail tradeoff, not a winner: RICH withholds correct (NP2 B1653, R1 A492/A493, 7 on T1 A128/132/134/135/136/137/138, 8 on COMBINED) to reach wrong-zero; SIMPLE recovers correct but reintroduces strict other-wrong (up to 5 on T1/COMBINED; R1 retains B32 wrong). Neither closes PSEM. No forced winner, no final ontology selected; no frozen named reference-free candidate identified in reviewed authorities (not selected/excluded permanently).
- Withholding explicit per case (no universal 'every wrong-zero has a correct-withheld' rule — false for NP1/NP3, which withheld wrong-only): NP1 wrong-only withheld (A1117); NP2 1 correct withheld (B1653) + 1 wrong withheld (C1481); NP3 wrong-only withheld (B322/325); R1 1 wrong + 2 correct withheld; T1 7 wrong + 7 correct withheld; COMBINED 8 wrong + 8 correct withheld plus A126 correct->wrong. Every unknown reduction under SIMPLE comes with new other-wrong. Wrong-zero alone must not promote; unknown-zero alone must not promote. Prior '2 wrong both withheld' NP2 shorthand and prior '3 correct withheld' T1 net phrasing superseded by the gross accounting above; accepted receiver docs keep their wording (upstream, owned separately, notified; not edited here).
- R1 projection stays history-only reference (accepted COMBINED 2 seals observed); this outcome recomputed R0+R2 only, matching accepted RICH exactly (verify all_match true).

## Not allowed next from this outcome

- No training or tuning in this outcome (tau/confirmation/schedule frozen; #98B validity-diff excluded). No claim that the semantic tradeoff could never be addressed by other means.
- No wait-policy adoption (gate condition NOT MET; user flexible wait preserved but no evidence for a wait arm here).
- No production/C56s claim; guards never PASS causal/safety.

## Final disposition (Director ACCEPT; conditional plan complete)

- HOLD ADOPTION. The two tested overlap-handling policies (RICH UNRESOLVED vs SIMPLE CURRENT) are not qualified for adoption: numeric outcomes are complete and no further actionable branch is justified under the current plan.
- Stop this tested overlap-policy application branch for now — NOT the research whole. Pretranslation receiver-capability result is preserved as research headroom (NP1 7c0w1u with 3 gross fixes, NP3 6c0w2u with 2 gross fixes, both wrong-zero), but neither policy candidate proceeds.
- No production adoption. No named missing observation yet, no H training, no new ontology invented now.
- Reopen only with an independently specified candidate/evidence addressing correct-word withholding vs false ownership; not just more thresholds/wait.
- Waiting: user wait policy stays flexible; no waiting experiment was performed in this outcome. The gate is a read-only count diagnostic and reads false within the available pre-terminal record (+0/+100/+300 ms add zero applicable events both ontologies; extended +20000 adds zero distinct); no future coverage claimed.

## Record control

- This outcome owns only `experiments/psem_pretranslation_ontology/` (FREEZE bede639b, probe f2703f5c, ledger 501f2228, RESULTS/WAIT_GATE/NEXT/MANIFEST as hashed). Accepted runtime files verified unchanged (`psem_pretranslation_receiver` FREEZE 07527fdd / replay 10494780 / ledger 97b2587c match frozen inputs); accepted docs RESULTS/NEXT have upstream doc-only revisions (b3afed1c/294c8f8f at this writing, subject to further doc updates) recorded as provenance in MANIFEST, not as rerun triggers. No commits (snapshot exception, Git writer unavailable). Zero paid API, zero new captures.
- Durable contribution: overlap-state semantics (UNRESOLVED vs CURRENT) move guard counts as measured above with zero NP difference and zero observed wait headroom, so any reopened candidate must beat BOTH arms' tradeoff, not one.

## 최종 요약 (Korean final summary)

- 3개 NP(서로 다른 동일 GT 8 ID씩, 총 24 ID) 합산 실측: R0 correct 12 / wrong 10 / unknown 1 / missing 1 → R2-RICH correct 17 / wrong 0 / unknown 6 / missing 1. 합계 24 일치 확인 (가드 겹침분 미합산).
- 총 gross fix 6건 = NP1 3 + NP2 1 + NP3 2. Correct withheld 1건(NP2 B1653). Wrong withheld 4건 = 1+1+2(NP1 A1117, NP2 C1481, NP3 B322/325). 기존 unknown 1 + unmatched 1 유지.
- 'wrong-zero마다 correct-withheld' 일반화는 거짓: NP1/NP3은 wrong만 withheld. 일반론 대신 케이스별 실측 기록을 따른다.
- 최종 처분: ADOPTION 보류(HOLD). 두 overlap 정책 모두 미적격, 해당 application branch 중단(연구 전체 중단 아님). NP1/NP3 headroom은 연구 자산으로 보존. 생산 반영 없음. 재개는 correct-word withholding 대 false ownership을 다루는 독립 지정 후보/증거가 있을 때만 (임계값/대기 추가 아님).
