# Next Decision (corrected v2, 해석 정정)

현재 이진 복귀 규칙의 운영 채택을 보류한다. 연구 전체 중단이나 신규 데이터가 있어야만 재검토 가능하다는 결론은 철회한다. 기존 데이터의 지속적 겹침 상태 활용은 아직 검증하지 않았다.

## 확정 해석(측정값 변경 없음, 기록일 2026-09-10 UTC 시계 확인)

현 결과는 R2 current [3,5] 대비 oracle-CF [4,7](pure-A 2 고정 / nonref 신규 wrong 3: throwing, anything, away, new_uncertain_right 1 추가), T1 전부 [6,7], COMBINED current [12,14] 대비 scalar-CF [7,12](fixed 9 / new wrong 5) 및 oracle-CF [9,13](fixed 11 / new wrong 8)이다. R2는 악화이며 aggregate benefit은 COMBINED에만 있고 신규 wrong 잔존으로 SAFE 개선은 없다. full4는 이번 한정 scalar와 동일하며 일반화하지 않는다.

한계: 소유권은 A vs OTHER UNKNOWN만, CURRENT는 첫 split만, CF에서만 다중 변경이 가능하다. full4(anchor GE 0.5 AND max others LT 0.5, conf 1600)는 return EVENT 필터일 뿐 지속 OVERLAP 상태/제어가 아니다. return oracle은 GT return onset + 고정 F0 이탈뿐 전체 정답 A-only/other-only/overlap/neither 궤적이 아니므로, 이번 실패로 PSEM/overlap-aware ownership의 논리적 불가능이나 H 무용을 증명할 수 없다. multi-active posterior는 overlap 표현이 가능하며, 지연은 raw invalid support gaps + valid low p anchor 혼합형이다. guard는 GT 단어 proxy이며 실제 ASR deadline/safety가 아니다(BC1 oracle nonbinding 포함). 실제 NP capture 3건은 별개 증거이다. bounds는 straddle 모호성 구간이지 통계 CI가 아니며, uniform no-overlap-exemption으로 신규 wrong을 uncertain으로 지울 수 없다. v1 overlap exemption 오류와 T1 init 상이는 v2에서 정정되었으며 broad safe 주장은 하지 않는다. production/code/architecture/Git 변경 없음.

## 다음 제안(미실행 기록, 결과 주장 없음)

제안(동결·실행 전): 동일 동결 데이터에 4상태 A-only/other-only/OVERLAP/uncertain로 current vs GT full state vs 실제 posterior를 같은 단어집합·개수로 비교하고 new wrong/new withheld/unresolved/conservation/인과 timing을 센다. overlap 상태가 각 단어 배정을 보장하지 않는다. 본 실험은 미동결·미실행이며 신규 결과 주장이 없다. Phase B는 evidence vs receiver 구분 대기이며 global no-H가 아니다. 이번 요청에서는 추가 실험을 실행하지 않음(기록만 수행). 기존 측정 stamp(freeze v2 2026-09-09T21:20:01.247159Z, ledger 21:20:12Z) 재표기가 아니다.
