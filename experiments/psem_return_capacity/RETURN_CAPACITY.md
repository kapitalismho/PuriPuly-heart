# Return Capacity Corrected Result (freeze v2, 31/31 PASS)
Freeze v2 `psem.return_capacity.freeze.v2` actual UTC before rerun. v1 stamp 21:15 postdated ledger 21:08:55; mtimes ~21:07-21:08 unreliable disclosed, prior hashes preserved in FREEZE revision_record, not backdated. Prereq 702720/680320 cited design record not blind. Scoring corrected per accepted review (uniform straddle-only, init A ALL, guards raw emit, full4 uniform); fairness not retuning, prior labels superseded. Smoke 18/18 edges, run 31/31 deterministic. No training/prod/Git/new inference.
## Signals full EN2009d epoch
single [672000]/684896; rearm [672000,733440]; scalar [672000,702720,733440] emit [684896,715616,746336]; full4 identical [672000,702720,733440] (strict subset, same valid, cannot earlier, no valid-frame filtering difference); oracle [672000,680320 GT Yeah start,733440] emit [684896,700256,746336]. Old 374400 invalid never fresh. Delay mixed NOT mask-only: gap f530-537 invalid plus posterior low f538 0.030 through f548 0.174 vs valid high f549 0.994; source wait 6720 frontier delay 18336 reported raw only. Guards deadline ASR UNKNOWN; raw frontier vs assumed cap only (oracle 700256 within R2 cap 701312 raw, scalar 715616 past cap raw), no usable-pending/service safety claim. Positives full service unchanged.
## Strict uniform scoring (straddle-only, init A ALL, worst bounds + all paired harms)
- R2 9w none [6,6] orig/rearm/scalar-CF/full4-CF [3,5]; oracle-CF [4,7] bounds [672000,680320]: vs orig fixed 2 pure-A (Yeah/WeAre) but new_wrong 3 nonref (B44/45/46) new_uncertain_right 1, lower 3->4 upper 5->7 worsened. Prior [1,7] hid 4B new-uncertain via exemption; corrected proves partial tradeoff with no SAFE improvement: R2 bounds worsened with new harms, aggregate benefit when present is COMBINED only.
- T1 21w none [11,11] orig None rearm/current [6,7]; scalar-CF/oracle-CF/full4-CF all [6,7] (init A, 702720 redundant, 680320 out-of-scope). Prior scalar [0,8] with new A128 straddle undeclared superseded by equal-init contract. No CF benefit T1.
- COMBINED 28w none [16,16] current [12,14]; scalar-CF/full4-CF [7,12] (5 new wrong, 9 fixed); oracle-CF [9,13] (8 new wrong, 11 fixed). Lower improves, upper 14->12/13, but 5/8 new nonref wrong persist; aggregate gain exists, no SAFE improvement.
- R1 [1,1] BC1 [4,4] SINGLEs [0,0] no-fire all arms. NP1 [0,0] NP2 [0,0] NP3 [0,1] preserved, full4 same as scalar, conservation exact, QA null. Overlap diagnostic reported uniformly all arms (scalar/oracle/full4 spans + right-overlap counts) separate from scoring.
## 확정 해석(현재 결과, v2 기준, 기록일 2026-09-10 UTC 시계 확인)
본 섹션은 측정값 수정이 아니라 해석 고정이다. 수치는 위 v2 그대로 유지한다. P3T 맥락: NP1 wrong 4->0 전환, NP2/NP3 보존, ORIG 대비 증가 guard harm 없음. 단 R2 절대 harm은 ORIG·rearm 공히 잔존한다.
R2 strict: current(orig/rearm/scalar-CF/full4-CF) [3,5] 대비 oracle-CF [4,7]. oracle은 pure-A 2개(Yeah words126, We are words128) 고정 대신 nonref 신규 wrong 3개(B44 throwing, B45 anything, B46 away) 발생, new_uncertain_right 1 추가. 하한 3->4 상한 5->7 악화로 SAFE 개선 없음.
T1 21w: none [11,11] 대비 orig None, rearm/current/scalar-CF/oracle-CF/full4-CF 전부 [6,7]. CF 이득 없음.
COMBINED 28w: current [12,14] 대비 scalar-CF/full4-CF [7,12](orig 대비 fixed 9, new wrong 5), oracle-CF [9,13](fixed 11, new wrong 8). 하한 개선은 있으나 신규 wrong 잔존으로 aggregate benefit만 있고 SAFE 개선은 아니다. 과거 zero ideal headroom 표현은 오기재로 철회한다.
full4 동일성은 이번 한정이며 일반화하지 않는다(full4 executed, scalar의 strict subset, same valid, earlier 없음, valid-frame filtering 차이 없음).
## 한계점(주요 정정 포함)
소유권 상태는 A vs OTHER UNKNOWN만 존재한다. CURRENT는 첫 split만 가능하고 다중 변경은 CF에서만 가능하다. full4(anchor GE 0.5 AND max others LT 0.5, conf 1600)는 return EVENT 필터일 뿐 지속 OVERLAP 상태/제어가 아니다.
return oracle은 GT return onset(680320) + 고정 F0 이탈(672000, 733440)만이며 전체 정답 A-only/other-only/overlap/neither 궤적이 아니다. 따라서 이번 실패는 PSEM/overlap-aware ownership이 논리적으로 불가능하다거나 H가 무용하다는 것을 증명할 수 없다.
multi-active posterior는 overlap 표현이 가능하다. 지연은 mask-only가 아니라 raw invalid support gaps(f530-537 invalid) + valid low p anchor(f538 0.030부터 f548 0.174, f549 0.994) 혼합형이다.
전 guard는 GT 단어 proxy이며 실제 ASR deadline/safety가 아니다(BC1 oracle slot 3 nonbinding 포함). 실제 NP capture 3건은 별개 증거이다.
bounds는 straddle 모호성 구간이지 통계 CI가 아니다. uniform no-overlap-exemption이므로 신규 wrong을 uncertain으로 지울 수 없다.
production/code/architecture/Git 변경 없음. v1의 return overlap exemption 오류와 T1 init 상이는 v2에서 정정되었으며 broad safe 주장은 하지 않는다.
후속 문서 UTC 표기는 신규 기록 시점이며 기존 측정 stamp(freeze v2 2026-09-09T21:20:01.247159Z, ledger 21:20:12Z) 재표기가 아니다.
## 다음 제안(미실행 기록, 결과 주장 없음)
binary return rule adoption HOLD. 제안(동결·실행 전): 동일 동결 데이터에 4상태 A-only/other-only/OVERLAP/uncertain로 current vs GT full state vs 실제 posterior를 같은 단어집합·개수로 비교하고 new wrong/new withheld/unresolved/conservation/인과 timing을 센다. overlap 상태가 각 단어를 배정 보장하지 않는다. 본 실험은 미동결·미실행이며 신규 결과 주장이 없다. Phase B는 evidence vs receiver 구분 대기이며 global no-H가 아니다. 이번 요청에서는 추가 실험을 실행하지 않음(기록만 수행).
## Verdict
HOLD(이진 복귀 규칙 운영 채택만 보류, 연구 전체 중단 아님): R2 oracle worse [3,5]->[4,7], T1 same [6,7], COMBINED aggregate gain [12,14]->[7,12]/[9,13] with 5/8 new wrong; no SAFE improvement across frozen controls. Full4 equals scalar here, not general. No H justified THIS BRANCH CURRENTLY, not globally.
