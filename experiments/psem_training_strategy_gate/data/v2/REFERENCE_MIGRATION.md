# PSEM v1 to v2 reference migration

This is a deterministic diagnostic migration report, not a boundary-quality gate. No model predictions or scores participate.

PSEM-STRATEGY-DATA-v2 adopts the commit-pinned forced-aligned AMI/AliMeeting references released by Horiguchi et al. (ASRU 2025) as the common temporal activity reference. This project does not perform additional manual boundary adjudication or independently estimate their acoustic boundary accuracy.

## Scope

- Sessions: 93
- AMI: 68
- AliMeeting: 25
- Reference integrity: `pass`

## Exposure

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 41.458659 | 7.113042 | 7.783836 | 33.590311 |
| v2 | 33.832962 | 14.738739 | 3.848209 | 29.633316 |

### Exposure by corpus

| Corpus | Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | --- | ---: | ---: | ---: | ---: |
| AMI | v1 | 29.555679 | 6.042440 | 5.355619 | 24.147000 |
| AMI | v2 | 23.052688 | 12.545431 | 2.137753 | 20.672817 |
| AliMeeting | v1 | 11.902981 | 1.070602 | 2.428217 | 9.443311 |
| AliMeeting | v2 | 10.780274 | 2.193309 | 1.710456 | 8.960499 |

## Boundary and source-span migration

- Deterministic old-span correspondences: 44279
- Derived internal-pause removal: 7.812956 h
- Derived outer-padding removal: 3.531382 h
- Absolute boundary displacement p50/p90/p99 samples: 688 / 5216 / 24960

## Topology migration

- v1/v2 exclusive episodes: 22966 / 47992
- Unchanged episodes: 3496
- Timing-only changes: 6993
- Topology-changing matches/additions/removals: 48766
- Handoff additions/removals: 9050 / 4583
- Short-backchannel net change: 2071
- Overlap takeover/return net changes: -1547 / -3367

| Retention collar | Retained v1 identity and event time | Proportion |
| --- | ---: | ---: |
| 50 ms | 6109 | 0.266002 |
| 100 ms | 8116 | 0.353392 |
| 200 ms | 10480 | 0.456327 |
| 500 ms | 11703 | 0.509579 |

### Direct/gap/overlap confusion

| v1 → v2 topology family | Matched episodes |
| --- | ---: |
| direct → direct | 200 |
| direct → gap | 418 |
| direct → overlap | 0 |
| gap → direct | 0 |
| gap → gap | 5037 |
| gap → overlap | 0 |
| overlap → direct | 231 |
| overlap → gap | 565 |
| overlap → overlap | 4583 |

### Topology by corpus

| Corpus | v1/v2 episodes | Timing-only | Topology-changing | Retained ≤50/500 ms | Handoff +/− | Masked v1/v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMI | 14502/32420 | 3009 | 36158 | 0.200041/0.399807 | 6213/3743 | 8797/17474 |
| AliMeeting | 8464/15572 | 3984 | 12608 | 0.379017/0.697661 | 2837/840 | 3022/5089 |

## Masks and integrity

- v1/v2 masked transitions: 11819 / 22563
- v2 nonlexical masks: 9687 (1.874686 h)
- RTTM terminal-tail clips: 2 total, 0 selected Train

## Per-meeting comparison

| Source | v1/v2 speech h | v1/v2 overlap h | v1/v2 reliable-solo h | Retained ≤50/500 ms | Timing-only/topology-changing | Handoff +/− | Nonlexical masks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

| `alimeeting_R0004_M0012` | 0.449217/0.400706 | 0.143403/0.084681 | 0.302908/0.305306 | 0.533605/0.598778 | 135/658 | 295/67 | 0 |
| `alimeeting_R0005_M0035` | 0.499431/0.441147 | 0.101347/0.063397 | 0.396139/0.368744 | 0.455128/0.572650 | 117/676 | 171/83 | 0 |
| `alimeeting_R0015_M0138` | 0.463964/0.435217 | 0.199044/0.149633 | 0.261733/0.277619 | 0.574675/0.685065 | 118/443 | 230/33 | 0 |
| `alimeeting_R0015_M0139` | 0.387419/0.371803 | 0.215586/0.172342 | 0.168408/0.191414 | 0.653465/0.717822 | 70/289 | 150/20 | 0 |
| `alimeeting_R0020_M0168` | 0.494967/0.455783 | 0.170847/0.126247 | 0.321056/0.320736 | 0.602247/0.642697 | 103/553 | 228/45 | 0 |
| `alimeeting_R1019_M1928` | 0.418689/0.375267 | 0.047733/0.029942 | 0.370522/0.343878 | 0.058219/0.722603 | 182/449 | 48/29 | 0 |
| `alimeeting_R1019_M1946` | 0.391667/0.351364 | 0.039592/0.022353 | 0.351311/0.327928 | 0.116438/0.815068 | 174/354 | 41/24 | 0 |
| `alimeeting_R1019_M1950` | 0.467233/0.426956 | 0.030539/0.021183 | 0.436611/0.405039 | 0.138728/0.768786 | 106/429 | 23/19 | 0 |
| `alimeeting_R1019_M1960` | 0.453097/0.405964 | 0.016608/0.008875 | 0.436281/0.396883 | 0.089796/0.828571 | 172/433 | 33/12 | 0 |
| `alimeeting_R1021_M1940` | 0.445697/0.390119 | 0.053717/0.026664 | 0.391536/0.361411 | 0.088496/0.701327 | 251/558 | 86/28 | 0 |
| `alimeeting_R1021_M1944` | 0.413508/0.378028 | 0.037708/0.024997 | 0.375303/0.351883 | 0.438753/0.875278 | 298/257 | 15/16 | 0 |
| `alimeeting_R1021_M4073` | 0.453417/0.396983 | 0.006711/0.004022 | 0.446689/0.392614 | 0.067708/0.875000 | 154/508 | 5/15 | 0 |
| `alimeeting_R1021_M4080` | 0.479806/0.430364 | 0.035269/0.020317 | 0.444219/0.407656 | 0.032258/0.648746 | 155/515 | 27/21 | 0 |
| `alimeeting_R2001_M2205` | 0.628789/0.581263 | 0.186636/0.135572 | 0.440725/0.438721 | 0.515564/0.680934 | 258/656 | 224/63 | 0 |
| `alimeeting_R2001_M2206` | 0.625753/0.577306 | 0.232411/0.151269 | 0.391469/0.417342 | 0.336820/0.583682 | 197/766 | 290/88 | 0 |
| `alimeeting_R2105_M3318` | 0.476964/0.412481 | 0.060600/0.034561 | 0.415856/0.375058 | 0.344203/0.478261 | 96/735 | 166/23 | 0 |
| `alimeeting_R2108_M3206` | 0.472011/0.427575 | 0.012036/0.008536 | 0.459761/0.418428 | 0.688406/0.782609 | 55/445 | 30/6 | 0 |
| `alimeeting_R8001_M8004` | 0.414317/0.380589 | 0.126761/0.094442 | 0.286606/0.280833 | 0.570370/0.659259 | 125/429 | 94/34 | 0 |
| `alimeeting_R8003_M8001` | 0.521439/0.471900 | 0.092639/0.065747 | 0.427058/0.401128 | 0.581871/0.669591 | 105/547 | 118/32 | 0 |
| `alimeeting_R8007_M8010` | 0.506139/0.483281 | 0.290761/0.231872 | 0.212544/0.241625 | 0.575107/0.648069 | 93/388 | 172/23 | 0 |
| `alimeeting_R8007_M8011` | 0.500539/0.466711 | 0.122994/0.098769 | 0.375908/0.362164 | 0.581169/0.652597 | 96/464 | 105/32 | 0 |
| `alimeeting_R8008_M8013` | 0.561314/0.494889 | 0.086122/0.058872 | 0.473672/0.430953 | 0.560465/0.653488 | 143/643 | 162/48 | 0 |
| `alimeeting_R8009_M8018` | 0.415167/0.381736 | 0.032306/0.021031 | 0.382342/0.359158 | 0.579387/0.768802 | 171/397 | 50/33 | 0 |
| `alimeeting_R8009_M8019` | 0.477983/0.414292 | 0.052039/0.034539 | 0.425464/0.377836 | 0.077830/0.799528 | 307/537 | 50/26 | 0 |
| `alimeeting_R8009_M8020` | 0.484456/0.428553 | 0.034806/0.020592 | 0.449189/0.406142 | 0.089109/0.824257 | 303/479 | 24/20 | 0 |
| `ami_EN2001d` | 0.809321/0.657309 | 0.128426/0.044254 | 0.680154/0.608204 | 0.162500/0.460000 | 119/961 | 200/95 | 90 |
| `ami_EN2002c` | 0.735979/0.592486 | 0.206377/0.080631 | 0.529136/0.503251 | 0.151099/0.401099 | 86/962 | 260/74 | 286 |
| `ami_EN2003a` | 0.525522/0.412256 | 0.062453/0.034437 | 0.462429/0.374796 | 0.245487/0.458484 | 59/611 | 94/54 | 112 |
| `ami_EN2004a` | 0.877426/0.716331 | 0.221488/0.092614 | 0.654872/0.616013 | 0.153659/0.370732 | 100/1000 | 204/86 | 565 |
| `ami_EN2006a` | 0.699914/0.540584 | 0.166181/0.085712 | 0.531204/0.445856 | 0.262264/0.430189 | 105/842 | 182/179 | 260 |
| `ami_EN2009d` | 1.307494/1.091608 | 0.282785/0.143163 | 1.022897/0.936277 | 0.179487/0.447964 | 183/1682 | 314/126 | 274 |
| `ami_ES2002b` | 0.549803/0.445043 | 0.093044/0.034686 | 0.456508/0.405564 | 0.138211/0.353659 | 51/767 | 176/35 | 84 |
| `ami_ES2003a` | 0.197112/0.140270 | 0.011128/0.003414 | 0.185740/0.135601 | 0.163636/0.418182 | 12/226 | 14/15 | 39 |
| `ami_ES2004a` | 0.229950/0.178031 | 0.051000/0.017582 | 0.178322/0.158286 | 0.110000/0.360000 | 26/295 | 81/27 | 60 |
| `ami_ES2005a` | 0.075109/0.053086 | 0.015393/0.004848 | 0.059591/0.047344 | 0.095238/0.190476 | 6/108 | 23/15 | 23 |
| `ami_ES2005b` | 0.589564/0.450649 | 0.109043/0.035566 | 0.479319/0.409659 | 0.158273/0.287770 | 40/758 | 130/79 | 200 |
| `ami_ES2005c` | 0.584676/0.460985 | 0.137888/0.054705 | 0.445269/0.400612 | 0.220447/0.364217 | 57/737 | 174/93 | 237 |
| `ami_ES2005d` | 0.378289/0.285508 | 0.096179/0.028929 | 0.281201/0.253305 | 0.172249/0.291866 | 25/479 | 113/67 | 204 |
| `ami_ES2006a` | 0.296366/0.215624 | 0.050417/0.011659 | 0.245333/0.202174 | 0.100000/0.254545 | 12/332 | 37/37 | 149 |
| `ami_ES2006b` | 0.552673/0.445561 | 0.086054/0.038979 | 0.466027/0.402268 | 0.110000/0.355000 | 49/669 | 111/48 | 177 |
| `ami_ES2006c` | 0.547303/0.446966 | 0.107387/0.048238 | 0.439200/0.393318 | 0.139344/0.299180 | 38/632 | 155/59 | 208 |
| `ami_ES2006d` | 0.482317/0.391345 | 0.155202/0.069769 | 0.325505/0.315918 | 0.173228/0.370079 | 49/533 | 139/83 | 270 |
| `ami_ES2007a` | 0.245608/0.174473 | 0.038961/0.011867 | 0.205896/0.160846 | 0.190840/0.351145 | 26/285 | 58/45 | 93 |
| `ami_ES2007b` | 0.390025/0.289722 | 0.060176/0.020796 | 0.329464/0.266089 | 0.178010/0.350785 | 39/474 | 63/55 | 120 |
| `ami_ES2007c` | 0.552488/0.415732 | 0.091277/0.030097 | 0.460431/0.382512 | 0.208494/0.382239 | 53/634 | 98/82 | 194 |
| `ami_ES2007d` | 0.299908/0.213886 | 0.064282/0.019904 | 0.234953/0.191079 | 0.197802/0.346154 | 30/375 | 55/58 | 122 |
| `ami_ES2008a` | 0.226031/0.176317 | 0.016605/0.005522 | 0.209032/0.169726 | 0.340206/0.453608 | 11/268 | 23/14 | 62 |
| `ami_ES2008b` | 0.498456/0.419669 | 0.045717/0.019708 | 0.451949/0.397057 | 0.376682/0.533632 | 39/597 | 85/46 | 55 |
| `ami_ES2008c` | 0.499506/0.408814 | 0.084939/0.031946 | 0.413914/0.373745 | 0.230769/0.485577 | 48/556 | 98/39 | 124 |
| `ami_ES2008d` | 0.616118/0.507671 | 0.103384/0.046209 | 0.511191/0.455782 | 0.371728/0.476440 | 54/739 | 126/105 | 200 |
| `ami_ES2009a` | 0.352307/0.283431 | 0.061611/0.024784 | 0.290111/0.255835 | 0.261628/0.453488 | 40/375 | 79/40 | 160 |
| `ami_ES2009b` | 0.354941/0.287144 | 0.041690/0.016072 | 0.312682/0.269538 | 0.264463/0.520661 | 33/359 | 40/25 | 94 |
| `ami_ES2009c` | 0.504784/0.404249 | 0.066015/0.024774 | 0.438171/0.376845 | 0.251101/0.444934 | 46/605 | 87/50 | 171 |
| `ami_ES2009d` | 0.531810/0.425721 | 0.127321/0.043383 | 0.403553/0.377844 | 0.200000/0.382456 | 59/598 | 124/87 | 274 |
| `ami_ES2010a` | 0.139552/0.102637 | 0.020326/0.007057 | 0.119084/0.094724 | 0.212121/0.348485 | 9/178 | 30/21 | 45 |
| `ami_ES2010b` | 0.399571/0.314204 | 0.047412/0.020399 | 0.351093/0.290283 | 0.223464/0.441341 | 34/504 | 61/47 | 60 |
| `ami_ES2010c` | 0.437906/0.360543 | 0.068646/0.028614 | 0.368551/0.328130 | 0.267943/0.440191 | 39/536 | 83/35 | 65 |
| `ami_ES2011a` | 0.241812/0.170458 | 0.047869/0.016882 | 0.193711/0.151712 | 0.179775/0.370787 | 17/261 | 40/22 | 120 |
| `ami_ES2011b` | 0.365124/0.296551 | 0.062330/0.031147 | 0.302333/0.262715 | 0.226667/0.400000 | 33/419 | 69/34 | 82 |
| `ami_ES2011d` | 0.422304/0.311184 | 0.096373/0.027965 | 0.325361/0.280526 | 0.086957/0.387352 | 50/525 | 74/78 | 196 |
| `ami_ES2012a` | 0.275108/0.196097 | 0.020914/0.004197 | 0.253967/0.190571 | 0.161765/0.338235 | 9/332 | 21/10 | 61 |
| `ami_ES2012b` | 0.516020/0.384586 | 0.056714/0.018331 | 0.458702/0.363729 | 0.178947/0.368421 | 38/633 | 48/44 | 133 |
| `ami_ES2012c` | 0.524626/0.407176 | 0.089979/0.034520 | 0.433944/0.368452 | 0.126214/0.364078 | 44/637 | 88/50 | 150 |
| `ami_ES2012d` | 0.224015/0.171303 | 0.049577/0.019562 | 0.174057/0.149685 | 0.170732/0.378049 | 20/261 | 44/24 | 76 |
| `ami_ES2013a` | 0.162536/0.118731 | 0.022579/0.006426 | 0.139815/0.111216 | 0.254237/0.440678 | 12/178 | 18/13 | 53 |
| `ami_ES2013b` | 0.495438/0.380780 | 0.052304/0.018907 | 0.442686/0.358704 | 0.185185/0.444444 | 55/620 | 67/38 | 92 |
| `ami_ES2013c` | 0.562190/0.429514 | 0.075300/0.020681 | 0.486404/0.406081 | 0.199029/0.383495 | 39/664 | 60/54 | 197 |
| `ami_ES2013d` | 0.412974/0.318311 | 0.048819/0.022160 | 0.363675/0.293826 | 0.224599/0.401070 | 48/530 | 63/49 | 76 |
| `ami_ES2014a` | 0.250000/0.169627 | 0.035734/0.006162 | 0.214143/0.162271 | 0.166667/0.285714 | 8/269 | 28/21 | 113 |
| `ami_ES2015d` | 0.464261/0.368991 | 0.173056/0.063888 | 0.289481/0.298415 | 0.098712/0.420601 | 54/540 | 173/61 | 216 |
| `ami_ES2016a` | 0.296694/0.205491 | 0.035689/0.011416 | 0.260559/0.191937 | 0.194175/0.417476 | 25/379 | 37/16 | 81 |
| `ami_IS1007a` | 0.205751/0.153325 | 0.044324/0.019194 | 0.160811/0.132336 | 0.251969/0.330709 | 8/245 | 43/44 | 86 |
| `ami_IS1007b` | 0.302402/0.237596 | 0.077539/0.028994 | 0.224261/0.205954 | 0.262857/0.400000 | 26/361 | 57/44 | 163 |
| `ami_IS1007c` | 0.501134/0.405527 | 0.093723/0.043239 | 0.406901/0.358525 | 0.230435/0.356522 | 32/624 | 116/47 | 122 |
| `ami_IS1007d` | 0.501865/0.398464 | 0.144800/0.045151 | 0.355776/0.349372 | 0.269076/0.393574 | 36/553 | 114/53 | 251 |
| `ami_IS1008a` | 0.221423/0.181939 | 0.015600/0.005978 | 0.205747/0.175030 | 0.222222/0.404040 | 22/237 | 31/23 | 32 |
| `ami_IS1009a` | 0.174614/0.146678 | 0.033903/0.020354 | 0.140374/0.124589 | 0.242424/0.373737 | 14/191 | 38/29 | 70 |
| `ami_TS3003b` | 0.509934/0.369193 | 0.041863/0.012265 | 0.467258/0.353350 | 0.067308/0.399038 | 58/734 | 38/49 | 51 |
| `ami_TS3004a` | 0.263960/0.199849 | 0.051405/0.022584 | 0.211681/0.174910 | 0.185897/0.416667 | 35/316 | 70/56 | 98 |
| `ami_TS3005b` | 0.594043/0.481788 | 0.119343/0.059891 | 0.472859/0.415252 | 0.243017/0.365922 | 54/817 | 166/112 | 185 |
| `ami_TS3006a` | 0.277568/0.199950 | 0.070404/0.022988 | 0.206477/0.173671 | 0.109948/0.408377 | 45/346 | 86/53 | 155 |
| `ami_TS3007a` | 0.333141/0.250479 | 0.058693/0.014665 | 0.273266/0.233993 | 0.151659/0.322275 | 40/448 | 76/80 | 183 |
| `ami_TS3008b` | 0.573487/0.473247 | 0.095355/0.051753 | 0.476956/0.417524 | 0.233227/0.444089 | 85/726 | 124/66 | 172 |
| `ami_TS3009b` | 0.607650/0.493920 | 0.160048/0.076416 | 0.445582/0.409753 | 0.184211/0.404605 | 69/747 | 143/92 | 279 |
| `ami_TS3010a` | 0.145913/0.082081 | 0.016944/0.002081 | 0.128457/0.078734 | 0.102941/0.235294 | 8/176 | 12/25 | 80 |
| `ami_TS3010b` | 0.402438/0.279288 | 0.021559/0.004990 | 0.380472/0.272426 | 0.235714/0.392857 | 24/497 | 32/41 | 60 |
| `ami_TS3010c` | 0.416169/0.289142 | 0.058616/0.014604 | 0.356813/0.271584 | 0.169082/0.405797 | 42/483 | 53/72 | 168 |
| `ami_TS3010d` | 0.349520/0.226328 | 0.039449/0.009722 | 0.309541/0.213420 | 0.189320/0.330097 | 24/431 | 58/87 | 122 |
| `ami_TS3011a` | 0.342554/0.261068 | 0.037448/0.014821 | 0.304680/0.244111 | 0.167883/0.357664 | 28/444 | 67/27 | 98 |
| `ami_TS3011b` | 0.541317/0.438025 | 0.084057/0.041765 | 0.456073/0.391239 | 0.249057/0.445283 | 62/748 | 129/57 | 119 |
| `ami_TS3011c` | 0.564518/0.441676 | 0.084502/0.035459 | 0.479078/0.401402 | 0.242236/0.413043 | 73/770 | 160/79 | 137 |
| `ami_TS3011d` | 0.445035/0.336857 | 0.089062/0.033036 | 0.354916/0.299166 | 0.192440/0.429553 | 70/580 | 102/74 | 210 |
| `ami_TS3012c` | 0.580312/0.469613 | 0.130939/0.075243 | 0.447402/0.388155 | 0.183333/0.476190 | 125/759 | 151/98 | 123 |

## Corpus diagnostic detail

### AMI

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 29.555679 | 6.042440 | 5.355619 | 24.147000 |
| v2 | 23.052688 | 12.545431 | 2.137753 | 20.672817 |

- Speech segments v1/v2: 31877 / 64990
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 28139 / 0 / 3738 / 0
- Removed internal pause: 361543536 samples (6.276797 h)
- Removed outer padding: 174764416 samples (3.034104 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 640 | 2720 | 5120 | 8000 | 22499 | 449536 |
| End samples | -268848 | -4448 | -1904 | 0 | 0 | 0 | 0 | 171 |
| Absolute samples | 0 | 0 | 1280 | 3520 | 6976 | 11520 | 31235 | 449536 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 96 | 250 | 0 |
| Gap | 0 | 2158 | 0 |
| Overlap | 134 | 448 | 2444 |

- Topology episodes v1/v2/matched/added/removed: 14502 / 32420 / 5798 / 26622 / 8704
- Unchanged/timing-only/topology-changing: 1957 / 3009 / 36158
- Overlap takeover/return changes: -1382 / -2932
- Short-backchannel change: 1379
- Handoffs v1/v2/matched/added/removed: 8738 / 11208 / 4995 / 6213 / 3743

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 2901 | 0.200041 |
| 100 ms | 3649 | 0.251620 |
| 200 ms | 4771 | 0.328989 |
| 500 ms | 5798 | 0.399807 |

- Masked transitions v1/v2/change: 8797 / 17474 / 8677
- Mask reasons v1: `{"complex_overlap_transition":2954,"continuity_unknown":3949,"mixed_unresolved_transition":1894}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":6821,"complex_overlap_transition":614,"continuity_unknown":4571,"initial_start":15,"mixed_unresolved_transition":5453}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":6821,"complex_overlap_transition":-2340,"continuity_unknown":622,"initial_start":15,"mixed_unresolved_transition":3559}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":2,"unpaired_v1_speech_segments":3738,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 9687 / 107981899 samples / 1.874686 h

### AliMeeting

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 11.902981 | 1.070602 | 2.428217 | 9.443311 |
| v2 | 10.780274 | 2.193309 | 1.710456 | 8.960499 |

- Speech segments v1/v2: 16154 / 29228
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 16140 / 6 / 8 / 0
- Removed internal pause: 88482720 samples (1.536158 h)
- Removed outer padding: 28643200 samples (0.497278 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 1280 | 1760 | 1920 | 3680 | 30720 |
| End samples | -24960 | -1760 | -640 | -480 | 0 | 0 | 0 | 160 |
| Absolute samples | 0 | 0 | 480 | 1440 | 2240 | 2720 | 4960 | 30720 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 104 | 168 | 0 |
| Gap | 0 | 2879 | 0 |
| Overlap | 97 | 117 | 2139 |

- Topology episodes v1/v2/matched/added/removed: 8464 / 15572 / 5905 / 9667 / 2559
- Unchanged/timing-only/topology-changing: 1539 / 3984 / 12608
- Overlap takeover/return changes: -165 / -435
- Short-backchannel change: 692
- Handoffs v1/v2/matched/added/removed: 5091 / 7088 / 4251 / 2837 / 840

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 3208 | 0.379017 |
| 100 ms | 4467 | 0.527765 |
| 200 ms | 5709 | 0.674504 |
| 500 ms | 5905 | 0.697661 |

- Masked transitions v1/v2/change: 3022 / 5089 / 2067
- Mask reasons v1: `{"complex_overlap_transition":1341,"continuity_unknown":618,"mixed_unresolved_transition":1063}`
- Mask reasons v2: `{"complex_overlap_transition":1030,"continuity_unknown":757,"mixed_unresolved_transition":3302}`
- Mask reason changes: `{"complex_overlap_transition":-311,"continuity_unknown":139,"mixed_unresolved_transition":2239}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":6,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":8,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

## Meeting diagnostic detail

### `alimeeting_R0004_M0012`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.449217 | 0.088319 | 0.143403 | 0.302908 |
| v2 | 0.400706 | 0.136831 | 0.084681 | 0.305306 |

- Speech segments v1/v2: 1071 / 1826
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 1071 / 0 / 0 / 0
- Removed internal pause: 5865760 samples (0.101836 h)
- Removed outer padding: 1008160 samples (0.017503 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 480 | 480 | 800 | 1600 | 9440 |
| End samples | -17600 | -800 | -480 | -480 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 480 | 480 | 960 | 1440 | 3614 | 17600 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 14 | 2 | 0 |
| Gap | 0 | 129 | 0 |
| Overlap | 2 | 6 | 91 |

- Topology episodes v1/v2/matched/added/removed: 491 / 745 / 294 / 451 / 197
- Unchanged/timing-only/topology-changing: 149 / 135 / 658
- Overlap takeover/return changes: 13 / -31
- Short-backchannel change: 87
- Handoffs v1/v2/matched/added/removed: 389 / 617 / 322 / 295 / 67

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 262 | 0.533605 |
| 100 ms | 281 | 0.572301 |
| 200 ms | 290 | 0.590631 |
| 500 ms | 294 | 0.598778 |

- Masked transitions v1/v2/change: 238 / 386 / 148
- Mask reasons v1: `{"complex_overlap_transition":94,"continuity_unknown":52,"mixed_unresolved_transition":92}`
- Mask reasons v2: `{"complex_overlap_transition":47,"continuity_unknown":53,"mixed_unresolved_transition":286}`
- Mask reason changes: `{"complex_overlap_transition":-47,"continuity_unknown":1,"mixed_unresolved_transition":194}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R0005_M0035`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.499431 | 0.079181 | 0.101347 | 0.396139 |
| v2 | 0.441147 | 0.137464 | 0.063397 | 0.368744 |

- Speech segments v1/v2: 875 / 1546
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 872 / 0 / 3 / 0
- Removed internal pause: 4760960 samples (0.082656 h)
- Removed outer padding: 924000 samples (0.016042 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 480 | 480 | 1120 | 2926 | 4320 |
| End samples | -19040 | -800 | -480 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 480 | 1600 | 2400 | 4320 | 19040 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 11 | 5 | 0 |
| Gap | 0 | 113 | 0 |
| Overlap | 7 | 10 | 97 |

- Topology episodes v1/v2/matched/added/removed: 468 / 722 / 268 / 454 / 200
- Unchanged/timing-only/topology-changing: 129 / 117 / 676
- Overlap takeover/return changes: -39 / -50
- Short-backchannel change: 39
- Handoffs v1/v2/matched/added/removed: 340 / 428 / 257 / 171 / 83

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 213 | 0.455128 |
| 100 ms | 228 | 0.487179 |
| 200 ms | 258 | 0.551282 |
| 500 ms | 268 | 0.572650 |

- Masked transitions v1/v2/change: 178 / 323 / 145
- Mask reasons v1: `{"complex_overlap_transition":62,"continuity_unknown":47,"mixed_unresolved_transition":69}`
- Mask reasons v2: `{"complex_overlap_transition":33,"continuity_unknown":49,"mixed_unresolved_transition":241}`
- Mask reason changes: `{"complex_overlap_transition":-29,"continuity_unknown":2,"mixed_unresolved_transition":172}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":3,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R0015_M0138`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.463964 | 0.019585 | 0.199044 | 0.261733 |
| v2 | 0.435217 | 0.048332 | 0.149633 | 0.277619 |

- Speech segments v1/v2: 1053 / 1689
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 1053 / 0 / 0 / 0
- Removed internal pause: 5136320 samples (0.089172 h)
- Removed outer padding: 890560 samples (0.015461 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 0 | 480 | 480 | 1836 | 10560 |
| End samples | -8320 | -640 | -480 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 480 | 960 | 1760 | 3840 | 10560 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 17 | 2 | 0 |
| Gap | 0 | 67 | 0 |
| Overlap | 2 | 1 | 113 |

- Topology episodes v1/v2/matched/added/removed: 308 / 552 / 211 / 341 / 97
- Unchanged/timing-only/topology-changing: 88 / 118 / 443
- Overlap takeover/return changes: 41 / -7
- Short-backchannel change: 49
- Handoffs v1/v2/matched/added/removed: 234 / 431 / 201 / 230 / 33

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 177 | 0.574675 |
| 100 ms | 190 | 0.616883 |
| 200 ms | 204 | 0.662338 |
| 500 ms | 211 | 0.685065 |

- Masked transitions v1/v2/change: 258 / 373 / 115
- Mask reasons v1: `{"complex_overlap_transition":153,"continuity_unknown":7,"mixed_unresolved_transition":98}`
- Mask reasons v2: `{"complex_overlap_transition":126,"continuity_unknown":8,"mixed_unresolved_transition":239}`
- Mask reason changes: `{"complex_overlap_transition":-27,"continuity_unknown":1,"mixed_unresolved_transition":141}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R0015_M0139`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.387419 | 0.013392 | 0.215586 | 0.168408 |
| v2 | 0.371803 | 0.029009 | 0.172342 | 0.191414 |

- Speech segments v1/v2: 1094 / 1694
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 1094 / 0 / 0 / 0
- Removed internal pause: 4752320 samples (0.082506 h)
- Removed outer padding: 713600 samples (0.012389 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 0 | 480 | 480 | 1120 | 5440 |
| End samples | -15040 | -480 | -480 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 480 | 640 | 1120 | 3060 | 15040 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 8 | 2 | 0 |
| Gap | 0 | 39 | 0 |
| Overlap | 1 | 3 | 89 |

- Topology episodes v1/v2/matched/added/removed: 202 / 371 / 145 / 226 / 57
- Unchanged/timing-only/topology-changing: 69 / 70 / 289
- Overlap takeover/return changes: 44 / 16
- Short-backchannel change: 18
- Handoffs v1/v2/matched/added/removed: 154 / 284 / 134 / 150 / 20

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 132 | 0.653465 |
| 100 ms | 139 | 0.688119 |
| 200 ms | 142 | 0.702970 |
| 500 ms | 145 | 0.717822 |

- Masked transitions v1/v2/change: 280 / 424 / 144
- Mask reasons v1: `{"complex_overlap_transition":170,"continuity_unknown":4,"mixed_unresolved_transition":106}`
- Mask reasons v2: `{"complex_overlap_transition":179,"continuity_unknown":4,"mixed_unresolved_transition":241}`
- Mask reason changes: `{"complex_overlap_transition":9,"continuity_unknown":0,"mixed_unresolved_transition":135}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R0020_M0168`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.494967 | 0.041322 | 0.170847 | 0.321056 |
| v2 | 0.455783 | 0.080506 | 0.126247 | 0.320736 |

- Speech segments v1/v2: 1015 / 1659
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 1014 / 0 / 1 / 0
- Removed internal pause: 4955040 samples (0.086025 h)
- Removed outer padding: 623360 samples (0.010822 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 0 | 480 | 480 | 960 | 8160 |
| End samples | -18560 | -480 | -480 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 480 | 480 | 960 | 3636 | 18560 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 15 | 1 | 0 |
| Gap | 0 | 122 | 0 |
| Overlap | 1 | 2 | 131 |

- Topology episodes v1/v2/matched/added/removed: 445 / 676 / 286 / 390 / 159
- Unchanged/timing-only/topology-changing: 179 / 103 / 553
- Overlap takeover/return changes: 10 / -36
- Short-backchannel change: 64
- Handoffs v1/v2/matched/added/removed: 307 / 490 / 262 / 228 / 45

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 268 | 0.602247 |
| 100 ms | 278 | 0.624719 |
| 200 ms | 282 | 0.633708 |
| 500 ms | 286 | 0.642697 |

- Masked transitions v1/v2/change: 222 / 340 / 118
- Mask reasons v1: `{"complex_overlap_transition":109,"continuity_unknown":8,"mixed_unresolved_transition":105}`
- Mask reasons v2: `{"complex_overlap_transition":87,"continuity_unknown":11,"mixed_unresolved_transition":242}`
- Mask reason changes: `{"complex_overlap_transition":-22,"continuity_unknown":3,"mixed_unresolved_transition":137}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":1,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R1019_M1928`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.418689 | 0.080561 | 0.047733 | 0.370522 |
| v2 | 0.375267 | 0.123983 | 0.029942 | 0.343878 |

- Speech segments v1/v2: 424 / 772
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 424 / 0 / 0 / 0
- Removed internal pause: 1849440 samples (0.032108 h)
- Removed outer padding: 1676480 samples (0.029106 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1440 | 1600 | 1760 | 1920 | 2240 | 4123 | 7680 |
| End samples | -12320 | -2560 | -2080 | -1760 | -1440 | -960 | 0 | 0 |
| Absolute samples | 0 | 1600 | 1760 | 2240 | 2880 | 3784 | 5760 | 12320 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 8 | 0 |
| Gap | 0 | 104 | 0 |
| Overlap | 9 | 5 | 75 |

- Topology episodes v1/v2/matched/added/removed: 292 / 557 / 211 / 346 / 81
- Unchanged/timing-only/topology-changing: 7 / 182 / 449
- Overlap takeover/return changes: -28 / -11
- Short-backchannel change: 11
- Handoffs v1/v2/matched/added/removed: 183 / 202 / 154 / 48 / 29

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 17 | 0.058219 |
| 100 ms | 101 | 0.345890 |
| 200 ms | 194 | 0.664384 |
| 500 ms | 211 | 0.722603 |

- Masked transitions v1/v2/change: 83 / 125 / 42
- Mask reasons v1: `{"continuity_unknown":68,"mixed_unresolved_transition":15}`
- Mask reasons v2: `{"continuity_unknown":75,"mixed_unresolved_transition":50}`
- Mask reason changes: `{"continuity_unknown":7,"mixed_unresolved_transition":35}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R1019_M1946`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.391667 | 0.081817 | 0.039592 | 0.351311 |
| v2 | 0.351364 | 0.122119 | 0.022353 | 0.327928 |

- Speech segments v1/v2: 481 / 732
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 481 / 0 / 0 / 0
- Removed internal pause: 1401440 samples (0.024331 h)
- Removed outer padding: 1974080 samples (0.034272 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1440 | 1600 | 1760 | 1920 | 2080 | 2400 | 4480 |
| End samples | -12960 | -2720 | -2240 | -1920 | -1280 | -640 | 0 | 0 |
| Absolute samples | 0 | 1440 | 1760 | 2240 | 3040 | 4480 | 10196 | 12960 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 20 | 0 |
| Gap | 0 | 126 | 0 |
| Overlap | 8 | 11 | 57 |

- Topology episodes v1/v2/matched/added/removed: 292 / 499 / 238 / 261 / 54
- Unchanged/timing-only/topology-changing: 25 / 174 / 354
- Overlap takeover/return changes: -17 / 3
- Short-backchannel change: 6
- Handoffs v1/v2/matched/added/removed: 239 / 256 / 215 / 41 / 24

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 34 | 0.116438 |
| 100 ms | 111 | 0.380137 |
| 200 ms | 222 | 0.760274 |
| 500 ms | 238 | 0.815068 |

- Masked transitions v1/v2/change: 110 / 136 / 26
- Mask reasons v1: `{"complex_overlap_transition":10,"continuity_unknown":76,"mixed_unresolved_transition":24}`
- Mask reasons v2: `{"complex_overlap_transition":3,"continuity_unknown":90,"mixed_unresolved_transition":43}`
- Mask reason changes: `{"complex_overlap_transition":-7,"continuity_unknown":14,"mixed_unresolved_transition":19}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R1019_M1950`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.467233 | 0.032383 | 0.030539 | 0.436611 |
| v2 | 0.426956 | 0.072661 | 0.021183 | 0.405039 |

- Speech segments v1/v2: 218 / 601
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 218 / 0 / 0 / 0
- Removed internal pause: 2013280 samples (0.034953 h)
- Removed outer padding: 845600 samples (0.014681 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1440 | 1600 | 1760 | 2080 | 2240 | 6550 | 9120 |
| End samples | -17440 | -2240 | -2080 | -1920 | -1600 | -1416 | 0 | 0 |
| Absolute samples | 0 | 1600 | 1760 | 2080 | 2560 | 3040 | 7144 | 17440 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 3 | 0 |
| Gap | 0 | 71 | 0 |
| Overlap | 4 | 9 | 43 |

- Topology episodes v1/v2/matched/added/removed: 173 / 506 / 133 / 373 / 40
- Unchanged/timing-only/topology-changing: 11 / 106 / 429
- Overlap takeover/return changes: -28 / -2
- Short-backchannel change: 7
- Handoffs v1/v2/matched/added/removed: 110 / 114 / 91 / 23 / 19

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 24 | 0.138728 |
| 100 ms | 58 | 0.335260 |
| 200 ms | 128 | 0.739884 |
| 500 ms | 133 | 0.768786 |

- Masked transitions v1/v2/change: 31 / 55 / 24
- Mask reasons v1: `{"continuity_unknown":26,"mixed_unresolved_transition":5}`
- Mask reasons v2: `{"continuity_unknown":32,"mixed_unresolved_transition":23}`
- Mask reason changes: `{"continuity_unknown":6,"mixed_unresolved_transition":18}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R1019_M1960`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.453097 | 0.042336 | 0.016608 | 0.436281 |
| v2 | 0.405964 | 0.089469 | 0.008875 | 0.396883 |

- Speech segments v1/v2: 309 / 672
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 308 / 0 / 1 / 0
- Removed internal pause: 2018400 samples (0.035042 h)
- Removed outer padding: 1118560 samples (0.019419 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1440 | 1600 | 1760 | 1920 | 1920 | 3028 | 4480 |
| End samples | -5600 | -2400 | -2080 | -1920 | -1600 | -696 | 0 | 0 |
| Absolute samples | 0 | 1600 | 1760 | 2240 | 2560 | 2720 | 3976 | 5600 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 11 | 0 |
| Gap | 0 | 144 | 0 |
| Overlap | 6 | 4 | 26 |

- Topology episodes v1/v2/matched/added/removed: 245 / 573 / 203 / 370 / 42
- Unchanged/timing-only/topology-changing: 10 / 172 / 433
- Overlap takeover/return changes: -19 / -9
- Short-backchannel change: 16
- Handoffs v1/v2/matched/added/removed: 165 / 186 / 153 / 33 / 12

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 22 | 0.089796 |
| 100 ms | 118 | 0.481633 |
| 200 ms | 200 | 0.816327 |
| 500 ms | 203 | 0.828571 |

- Masked transitions v1/v2/change: 36 / 51 / 15
- Mask reasons v1: `{"continuity_unknown":32,"mixed_unresolved_transition":4}`
- Mask reasons v2: `{"continuity_unknown":43,"mixed_unresolved_transition":8}`
- Mask reason changes: `{"continuity_unknown":11,"mixed_unresolved_transition":4}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":1,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R1021_M1940`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.445697 | 0.045586 | 0.053717 | 0.391536 |
| v2 | 0.390119 | 0.101164 | 0.026664 | 0.361411 |

- Speech segments v1/v2: 583 / 993
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 583 / 0 / 0 / 0
- Removed internal pause: 2438720 samples (0.042339 h)
- Removed outer padding: 2320800 samples (0.040292 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1440 | 1600 | 1760 | 1920 | 2240 | 3737 | 5920 |
| End samples | -16000 | -2560 | -2080 | -1760 | -1280 | -480 | 0 | 0 |
| Absolute samples | 0 | 1600 | 1760 | 2240 | 2880 | 3520 | 6880 | 16000 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 16 | 0 |
| Gap | 0 | 162 | 0 |
| Overlap | 13 | 17 | 64 |

- Topology episodes v1/v2/matched/added/removed: 452 / 694 / 317 / 377 / 135
- Unchanged/timing-only/topology-changing: 20 / 251 / 558
- Overlap takeover/return changes: -33 / -55
- Short-backchannel change: 25
- Handoffs v1/v2/matched/added/removed: 304 / 362 / 276 / 86 / 28

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 40 | 0.088496 |
| 100 ms | 148 | 0.327434 |
| 200 ms | 294 | 0.650442 |
| 500 ms | 317 | 0.701327 |

- Masked transitions v1/v2/change: 41 / 109 / 68
- Mask reasons v1: `{"continuity_unknown":24,"mixed_unresolved_transition":17}`
- Mask reasons v2: `{"continuity_unknown":32,"mixed_unresolved_transition":77}`
- Mask reason changes: `{"continuity_unknown":8,"mixed_unresolved_transition":60}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R1021_M1944`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.413508 | 0.074033 | 0.037708 | 0.375303 |
| v2 | 0.378028 | 0.109514 | 0.024997 | 0.351883 |

- Speech segments v1/v2: 571 / 769
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 571 / 0 / 0 / 0
- Removed internal pause: 1299040 samples (0.022553 h)
- Removed outer padding: 1476800 samples (0.025639 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 800 | 1600 | 1760 | 1920 | 3344 | 30720 |
| End samples | -17120 | -2240 | -1280 | -480 | -480 | 0 | 0 | 0 |
| Absolute samples | 0 | 480 | 960 | 1760 | 2560 | 3360 | 5374 | 30720 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 10 | 0 |
| Gap | 0 | 286 | 0 |
| Overlap | 1 | 1 | 67 |

- Topology episodes v1/v2/matched/added/removed: 449 / 582 / 393 / 189 / 56
- Unchanged/timing-only/topology-changing: 83 / 298 / 257
- Overlap takeover/return changes: -10 / -13
- Short-backchannel change: 1
- Handoffs v1/v2/matched/added/removed: 214 / 213 / 198 / 15 / 16

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 197 | 0.438753 |
| 100 ms | 309 | 0.688196 |
| 200 ms | 384 | 0.855234 |
| 500 ms | 393 | 0.875278 |

- Masked transitions v1/v2/change: 58 / 97 / 39
- Mask reasons v1: `{"continuity_unknown":41,"mixed_unresolved_transition":17}`
- Mask reasons v2: `{"continuity_unknown":56,"mixed_unresolved_transition":41}`
- Mask reason changes: `{"continuity_unknown":15,"mixed_unresolved_transition":24}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R1021_M4073`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.453417 | 0.039333 | 0.006711 | 0.446689 |
| v2 | 0.396983 | 0.095767 | 0.004022 | 0.392614 |

- Speech segments v1/v2: 227 / 708
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 227 / 0 / 0 / 0
- Removed internal pause: 2588480 samples (0.044939 h)
- Removed outer padding: 816960 samples (0.014183 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1440 | 1600 | 1760 | 1920 | 2032 | 2796 | 7360 |
| End samples | -4000 | -2240 | -2080 | -1760 | -1600 | -1328 | -480 | 0 |
| Absolute samples | 0 | 1600 | 1760 | 2080 | 2400 | 2560 | 3755 | 7360 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 3 | 0 |
| Gap | 0 | 145 | 0 |
| Overlap | 2 | 3 | 13 |

- Topology episodes v1/v2/matched/added/removed: 192 / 644 / 168 / 476 / 24
- Unchanged/timing-only/topology-changing: 6 / 154 / 508
- Overlap takeover/return changes: -7 / 0
- Short-backchannel change: -1
- Handoffs v1/v2/matched/added/removed: 72 / 62 / 57 / 5 / 15

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 13 | 0.067708 |
| 100 ms | 112 | 0.583333 |
| 200 ms | 167 | 0.869792 |
| 500 ms | 168 | 0.875000 |

- Masked transitions v1/v2/change: 26 / 48 / 22
- Mask reasons v1: `{"continuity_unknown":24,"mixed_unresolved_transition":2}`
- Mask reasons v2: `{"continuity_unknown":38,"mixed_unresolved_transition":10}`
- Mask reason changes: `{"continuity_unknown":14,"mixed_unresolved_transition":8}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R1021_M4080`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.479806 | 0.012803 | 0.035269 | 0.444219 |
| v2 | 0.430364 | 0.062244 | 0.020317 | 0.407656 |

- Speech segments v1/v2: 339 / 809
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 339 / 0 / 0 / 0
- Removed internal pause: 2427200 samples (0.042139 h)
- Removed outer padding: 1281920 samples (0.022256 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1440 | 1600 | 1760 | 1920 | 2080 | 2979 | 4960 |
| End samples | -6240 | -2240 | -2080 | -1920 | -1600 | -1088 | -183 | 0 |
| Absolute samples | 0 | 1600 | 1920 | 2080 | 2400 | 2560 | 4320 | 6240 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 18 | 0 |
| Gap | 0 | 84 | 0 |
| Overlap | 3 | 3 | 51 |

- Topology episodes v1/v2/matched/added/removed: 279 / 574 / 181 / 393 / 98
- Unchanged/timing-only/topology-changing: 2 / 155 / 515
- Overlap takeover/return changes: -17 / -62
- Short-backchannel change: 4
- Handoffs v1/v2/matched/added/removed: 160 / 166 / 139 / 27 / 21

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 9 | 0.032258 |
| 100 ms | 76 | 0.272401 |
| 200 ms | 179 | 0.641577 |
| 500 ms | 181 | 0.648746 |

- Masked transitions v1/v2/change: 14 / 100 / 86
- Mask reasons v1: `{"continuity_unknown":1,"mixed_unresolved_transition":13}`
- Mask reasons v2: `{"continuity_unknown":4,"mixed_unresolved_transition":96}`
- Mask reason changes: `{"continuity_unknown":3,"mixed_unresolved_transition":83}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R2001_M2205`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.628789 | 0.028004 | 0.186636 | 0.440725 |
| v2 | 0.581263 | 0.075531 | 0.135572 | 0.438721 |

- Speech segments v1/v2: 932 / 1633
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 932 / 0 / 0 / 0
- Removed internal pause: 5033600 samples (0.087389 h)
- Removed outer padding: 1224000 samples (0.021250 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 480 | 800 | 1120 | 2190 | 16800 |
| End samples | -22720 | -960 | -640 | -480 | 0 | 0 | 0 | 80 |
| Absolute samples | 0 | 0 | 480 | 800 | 1280 | 1760 | 4160 | 22720 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 8 | 7 | 0 |
| Gap | 0 | 105 | 0 |
| Overlap | 10 | 4 | 207 |

- Topology episodes v1/v2/matched/added/removed: 514 / 821 / 350 / 471 / 164
- Unchanged/timing-only/topology-changing: 71 / 258 / 656
- Overlap takeover/return changes: -25 / 4
- Short-backchannel change: 59
- Handoffs v1/v2/matched/added/removed: 345 / 506 / 282 / 224 / 63

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 265 | 0.515564 |
| 100 ms | 327 | 0.636187 |
| 200 ms | 343 | 0.667315 |
| 500 ms | 350 | 0.680934 |

- Masked transitions v1/v2/change: 176 / 295 / 119
- Mask reasons v1: `{"complex_overlap_transition":116,"continuity_unknown":9,"mixed_unresolved_transition":51}`
- Mask reasons v2: `{"complex_overlap_transition":69,"continuity_unknown":9,"mixed_unresolved_transition":217}`
- Mask reason changes: `{"complex_overlap_transition":-47,"continuity_unknown":0,"mixed_unresolved_transition":166}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R2001_M2206`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.625753 | 0.019436 | 0.232411 | 0.391469 |
| v2 | 0.577306 | 0.067883 | 0.151269 | 0.417342 |

- Speech segments v1/v2: 991 / 1762
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 984 / 6 / 1 / 0
- Removed internal pause: 5787360 samples (0.100475 h)
- Removed outer padding: 2680640 samples (0.046539 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2400 | 3472 | 4000 | 5627 | 10880 |
| End samples | -18560 | -2400 | -800 | -480 | 0 | 0 | 0 | 160 |
| Absolute samples | 0 | 480 | 640 | 2400 | 3520 | 4000 | 6292 | 18560 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 17 | 0 |
| Gap | 0 | 66 | 0 |
| Overlap | 9 | 13 | 157 |

- Topology episodes v1/v2/matched/added/removed: 478 / 807 / 279 / 528 / 199
- Unchanged/timing-only/topology-changing: 43 / 197 / 766
- Overlap takeover/return changes: -4 / 35
- Short-backchannel change: 53
- Handoffs v1/v2/matched/added/removed: 321 / 523 / 233 / 290 / 88

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 161 | 0.336820 |
| 100 ms | 199 | 0.416318 |
| 200 ms | 251 | 0.525105 |
| 500 ms | 279 | 0.583682 |

- Masked transitions v1/v2/change: 216 / 367 / 151
- Mask reasons v1: `{"complex_overlap_transition":140,"continuity_unknown":8,"mixed_unresolved_transition":68}`
- Mask reasons v2: `{"complex_overlap_transition":92,"continuity_unknown":18,"mixed_unresolved_transition":257}`
- Mask reason changes: `{"complex_overlap_transition":-48,"continuity_unknown":10,"mixed_unresolved_transition":189}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":6,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":1,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R2105_M3318`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.476964 | 0.036470 | 0.060600 | 0.415856 |
| v2 | 0.412481 | 0.100953 | 0.034561 | 0.375058 |

- Speech segments v1/v2: 413 / 1083
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 413 / 0 / 0 / 0
- Removed internal pause: 4657760 samples (0.080864 h)
- Removed outer padding: 630240 samples (0.010942 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 640 | 960 | 1280 | 2841 | 4480 |
| End samples | -13600 | -1120 | -800 | -480 | -480 | 0 | 0 | 0 |
| Absolute samples | 0 | 480 | 480 | 960 | 1440 | 2240 | 3960 | 13600 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 6 | 0 |
| Gap | 0 | 43 | 0 |
| Overlap | 1 | 4 | 60 |

- Topology episodes v1/v2/matched/added/removed: 276 / 712 / 132 / 580 / 144
- Unchanged/timing-only/topology-changing: 25 / 96 / 735
- Overlap takeover/return changes: -9 / -83
- Short-backchannel change: 57
- Handoffs v1/v2/matched/added/removed: 141 / 284 / 118 / 166 / 23

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 95 | 0.344203 |
| 100 ms | 122 | 0.442029 |
| 200 ms | 131 | 0.474638 |
| 500 ms | 132 | 0.478261 |

- Masked transitions v1/v2/change: 67 / 146 / 79
- Mask reasons v1: `{"complex_overlap_transition":16,"continuity_unknown":35,"mixed_unresolved_transition":16}`
- Mask reasons v2: `{"complex_overlap_transition":10,"continuity_unknown":36,"mixed_unresolved_transition":100}`
- Mask reason changes: `{"complex_overlap_transition":-6,"continuity_unknown":1,"mixed_unresolved_transition":84}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R2108_M3206`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.472011 | 0.025495 | 0.012036 | 0.459761 |
| v2 | 0.427575 | 0.069931 | 0.008536 | 0.418428 |

- Speech segments v1/v2: 180 / 605
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 180 / 0 / 0 / 0
- Removed internal pause: 2572640 samples (0.044664 h)
- Removed outer padding: 196480 samples (0.003411 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 480 | 640 | 960 | 2921 | 4960 |
| End samples | -4800 | -960 | -480 | -480 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 480 | 640 | 1136 | 1760 | 3171 | 4960 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 1 | 0 |
| Gap | 0 | 79 | 0 |
| Overlap | 0 | 1 | 23 |

- Topology episodes v1/v2/matched/added/removed: 138 / 521 / 108 / 413 / 30
- Unchanged/timing-only/topology-changing: 51 / 55 / 445
- Overlap takeover/return changes: -4 / -17
- Short-backchannel change: 12
- Handoffs v1/v2/matched/added/removed: 91 / 115 / 85 / 30 / 6

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 95 | 0.688406 |
| 100 ms | 103 | 0.746377 |
| 200 ms | 107 | 0.775362 |
| 500 ms | 108 | 0.782609 |

- Masked transitions v1/v2/change: 26 / 42 / 16
- Mask reasons v1: `{"complex_overlap_transition":2,"continuity_unknown":16,"mixed_unresolved_transition":8}`
- Mask reasons v2: `{"complex_overlap_transition":1,"continuity_unknown":19,"mixed_unresolved_transition":22}`
- Mask reason changes: `{"complex_overlap_transition":-1,"continuity_unknown":3,"mixed_unresolved_transition":14}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R8001_M8004`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.414317 | 0.016504 | 0.126761 | 0.286606 |
| v2 | 0.380589 | 0.050231 | 0.094442 | 0.280833 |

- Speech segments v1/v2: 587 / 1139
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 587 / 0 / 0 / 0
- Removed internal pause: 3935680 samples (0.068328 h)
- Removed outer padding: 588480 samples (0.010217 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 480 | 640 | 800 | 2080 | 8800 |
| End samples | -17920 | -800 | -480 | -480 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 480 | 640 | 960 | 1440 | 3200 | 17920 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 0 | 0 |
| Gap | 0 | 28 | 0 |
| Overlap | 1 | 2 | 142 |

- Topology episodes v1/v2/matched/added/removed: 270 / 512 / 178 / 334 / 92
- Unchanged/timing-only/topology-changing: 50 / 125 / 429
- Overlap takeover/return changes: -10 / 24
- Short-backchannel change: 19
- Handoffs v1/v2/matched/added/removed: 93 / 153 / 59 / 94 / 34

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 154 | 0.570370 |
| 100 ms | 168 | 0.622222 |
| 200 ms | 175 | 0.648148 |
| 500 ms | 178 | 0.659259 |

- Masked transitions v1/v2/change: 116 / 218 / 102
- Mask reasons v1: `{"complex_overlap_transition":73,"continuity_unknown":6,"mixed_unresolved_transition":37}`
- Mask reasons v2: `{"complex_overlap_transition":58,"continuity_unknown":6,"mixed_unresolved_transition":154}`
- Mask reason changes: `{"complex_overlap_transition":-15,"continuity_unknown":0,"mixed_unresolved_transition":117}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R8003_M8001`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.521439 | 0.052925 | 0.092639 | 0.427058 |
| v2 | 0.471900 | 0.102464 | 0.065747 | 0.401128 |

- Speech segments v1/v2: 670 / 1238
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 670 / 0 / 0 / 0
- Removed internal pause: 4139520 samples (0.071867 h)
- Removed outer padding: 684640 samples (0.011886 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 480 | 480 | 960 | 2449 | 10720 |
| End samples | -20640 | -640 | -480 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 480 | 480 | 1120 | 2080 | 5315 | 20640 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 7 | 0 | 0 |
| Gap | 0 | 97 | 0 |
| Overlap | 2 | 0 | 112 |

- Topology episodes v1/v2/matched/added/removed: 342 / 661 / 229 / 432 / 113
- Unchanged/timing-only/topology-changing: 122 / 105 / 547
- Overlap takeover/return changes: -1 / -34
- Short-backchannel change: 32
- Handoffs v1/v2/matched/added/removed: 147 / 233 / 115 / 118 / 32

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 199 | 0.581871 |
| 100 ms | 212 | 0.619883 |
| 200 ms | 224 | 0.654971 |
| 500 ms | 229 | 0.669591 |

- Masked transitions v1/v2/change: 139 / 219 / 80
- Mask reasons v1: `{"complex_overlap_transition":60,"continuity_unknown":27,"mixed_unresolved_transition":52}`
- Mask reasons v2: `{"complex_overlap_transition":36,"continuity_unknown":26,"mixed_unresolved_transition":157}`
- Mask reason changes: `{"complex_overlap_transition":-24,"continuity_unknown":-1,"mixed_unresolved_transition":105}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R8007_M8010`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.506139 | 0.008836 | 0.290761 | 0.212544 |
| v2 | 0.483281 | 0.031695 | 0.231872 | 0.241625 |

- Speech segments v1/v2: 1178 / 1998
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 1177 / 0 / 1 / 0
- Removed internal pause: 6019360 samples (0.104503 h)
- Removed outer padding: 1159360 samples (0.020128 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 480 | 640 | 1280 | 4236 | 16320 |
| End samples | -24960 | -480 | -480 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 480 | 800 | 1920 | 6475 | 24960 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 1 | 0 |
| Gap | 0 | 28 | 0 |
| Overlap | 1 | 0 | 113 |

- Topology episodes v1/v2/matched/added/removed: 233 / 455 / 151 / 304 / 82
- Unchanged/timing-only/topology-changing: 56 / 93 / 388
- Overlap takeover/return changes: 43 / 26
- Short-backchannel change: 30
- Handoffs v1/v2/matched/added/removed: 92 / 241 / 69 / 172 / 23

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 134 | 0.575107 |
| 100 ms | 143 | 0.613734 |
| 200 ms | 149 | 0.639485 |
| 500 ms | 151 | 0.648069 |

- Masked transitions v1/v2/change: 281 / 469 / 188
- Mask reasons v1: `{"complex_overlap_transition":195,"mixed_unresolved_transition":86}`
- Mask reasons v2: `{"complex_overlap_transition":196,"mixed_unresolved_transition":273}`
- Mask reason changes: `{"complex_overlap_transition":1,"mixed_unresolved_transition":187}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":1,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R8007_M8011`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.500539 | 0.015848 | 0.122994 | 0.375908 |
| v2 | 0.466711 | 0.049676 | 0.098769 | 0.362164 |

- Speech segments v1/v2: 743 / 1260
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 742 / 0 / 1 / 0
- Removed internal pause: 3186560 samples (0.055322 h)
- Removed outer padding: 587360 samples (0.010197 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 0 | 480 | 480 | 1600 | 8480 |
| End samples | -16480 | -640 | -480 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 480 | 800 | 1600 | 4027 | 16480 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 7 | 2 | 0 |
| Gap | 0 | 58 | 0 |
| Overlap | 1 | 2 | 119 |

- Topology episodes v1/v2/matched/added/removed: 308 / 553 / 201 / 352 / 107
- Unchanged/timing-only/topology-changing: 100 / 96 / 464
- Overlap takeover/return changes: 6 / -22
- Short-backchannel change: 24
- Handoffs v1/v2/matched/added/removed: 184 / 257 / 152 / 105 / 32

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 179 | 0.581169 |
| 100 ms | 189 | 0.613636 |
| 200 ms | 197 | 0.639610 |
| 500 ms | 201 | 0.652597 |

- Masked transitions v1/v2/change: 163 / 245 / 82
- Mask reasons v1: `{"complex_overlap_transition":102,"continuity_unknown":3,"mixed_unresolved_transition":58}`
- Mask reasons v2: `{"complex_overlap_transition":75,"continuity_unknown":5,"mixed_unresolved_transition":165}`
- Mask reason changes: `{"complex_overlap_transition":-27,"continuity_unknown":2,"mixed_unresolved_transition":107}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":1,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R8008_M8013`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.561314 | 0.059658 | 0.086122 | 0.473672 |
| v2 | 0.494889 | 0.126083 | 0.058872 | 0.430953 |

- Speech segments v1/v2: 724 / 1321
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 724 / 0 / 0 / 0
- Removed internal pause: 4727840 samples (0.082081 h)
- Removed outer padding: 846720 samples (0.014700 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 480 | 960 | 1600 | 3840 | 12160 |
| End samples | -17920 | -960 | -480 | -480 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 480 | 640 | 1280 | 2080 | 4000 | 17920 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 7 | 7 | 0 |
| Gap | 0 | 144 | 0 |
| Overlap | 7 | 6 | 86 |

- Topology episodes v1/v2/matched/added/removed: 430 / 755 / 281 / 474 / 149
- Unchanged/timing-only/topology-changing: 118 / 143 / 643
- Overlap takeover/return changes: -32 / -54
- Short-backchannel change: 52
- Handoffs v1/v2/matched/added/removed: 307 / 421 / 259 / 162 / 48

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 241 | 0.560465 |
| 100 ms | 265 | 0.616279 |
| 200 ms | 273 | 0.634884 |
| 500 ms | 281 | 0.653488 |

- Masked transitions v1/v2/change: 141 / 232 / 91
- Mask reasons v1: `{"complex_overlap_transition":39,"continuity_unknown":42,"mixed_unresolved_transition":60}`
- Mask reasons v2: `{"complex_overlap_transition":18,"continuity_unknown":47,"mixed_unresolved_transition":167}`
- Mask reason changes: `{"complex_overlap_transition":-21,"continuity_unknown":5,"mixed_unresolved_transition":107}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R8009_M8018`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.415167 | 0.043579 | 0.032306 | 0.382342 |
| v2 | 0.381736 | 0.077010 | 0.021031 | 0.359158 |

- Speech segments v1/v2: 457 / 783
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 457 / 0 / 0 / 0
- Removed internal pause: 1868960 samples (0.032447 h)
- Removed outer padding: 706080 samples (0.012258 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 640 | 960 | 1152 | 1760 | 4320 |
| End samples | -11360 | -1600 | -640 | -480 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 480 | 960 | 1920 | 2720 | 4918 | 11360 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 7 | 0 |
| Gap | 0 | 161 | 0 |
| Overlap | 2 | 1 | 78 |

- Topology episodes v1/v2/matched/added/removed: 359 / 580 / 276 / 304 / 83
- Unchanged/timing-only/topology-changing: 95 / 171 / 397
- Overlap takeover/return changes: -13 / -23
- Short-backchannel change: 11
- Handoffs v1/v2/matched/added/removed: 220 / 237 / 187 / 50 / 33

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 208 | 0.579387 |
| 100 ms | 246 | 0.685237 |
| 200 ms | 264 | 0.735376 |
| 500 ms | 276 | 0.768802 |

- Masked transitions v1/v2/change: 33 / 81 / 48
- Mask reasons v1: `{"continuity_unknown":16,"mixed_unresolved_transition":17}`
- Mask reasons v2: `{"continuity_unknown":19,"mixed_unresolved_transition":62}`
- Mask reason changes: `{"continuity_unknown":3,"mixed_unresolved_transition":45}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R8009_M8019`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.477983 | 0.067758 | 0.052039 | 0.425464 |
| v2 | 0.414292 | 0.131450 | 0.034539 | 0.377836 |

- Speech segments v1/v2: 523 / 1013
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 523 / 0 / 0 / 0
- Removed internal pause: 2774720 samples (0.048172 h)
- Removed outer padding: 1901920 samples (0.033019 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1440 | 1600 | 1760 | 1920 | 2240 | 3324 | 9280 |
| End samples | -8160 | -2400 | -1920 | -1600 | -1280 | -496 | 0 | 0 |
| Absolute samples | 0 | 1600 | 1760 | 2080 | 2560 | 3040 | 4088 | 9280 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 8 | 0 |
| Gap | 0 | 262 | 0 |
| Overlap | 2 | 4 | 49 |

- Topology episodes v1/v2/matched/added/removed: 424 / 777 / 339 / 438 / 85
- Unchanged/timing-only/topology-changing: 18 / 307 / 537
- Overlap takeover/return changes: -16 / -6
- Short-backchannel change: 16
- Handoffs v1/v2/matched/added/removed: 134 / 158 / 108 / 50 / 26

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 33 | 0.077830 |
| 100 ms | 168 | 0.396226 |
| 200 ms | 329 | 0.775943 |
| 500 ms | 339 | 0.799528 |

- Masked transitions v1/v2/change: 56 / 119 / 63
- Mask reasons v1: `{"continuity_unknown":37,"mixed_unresolved_transition":19}`
- Mask reasons v2: `{"continuity_unknown":59,"mixed_unresolved_transition":60}`
- Mask reason changes: `{"continuity_unknown":22,"mixed_unresolved_transition":41}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `alimeeting_R8009_M8020`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.484456 | 0.045436 | 0.034806 | 0.449189 |
| v2 | 0.428553 | 0.101339 | 0.020592 | 0.406142 |

- Speech segments v1/v2: 496 / 923
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 496 / 0 / 0 / 0
- Removed internal pause: 2272320 samples (0.039450 h)
- Removed outer padding: 1766400 samples (0.030667 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1440 | 1600 | 1760 | 2080 | 2240 | 3368 | 4480 |
| End samples | -6400 | -2240 | -1920 | -1600 | -1120 | -800 | 0 | 0 |
| Absolute samples | 0 | 1440 | 1760 | 2080 | 2400 | 2880 | 4014 | 6400 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 11 | 0 |
| Gap | 0 | 216 | 0 |
| Overlap | 2 | 5 | 76 |

- Topology episodes v1/v2/matched/added/removed: 404 / 723 / 333 / 390 / 71
- Unchanged/timing-only/topology-changing: 12 / 303 / 479
- Overlap takeover/return changes: -10 / -28
- Short-backchannel change: 1
- Handoffs v1/v2/matched/added/removed: 145 / 149 / 125 / 24 / 20

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 36 | 0.089109 |
| 100 ms | 176 | 0.435644 |
| 200 ms | 322 | 0.797030 |
| 500 ms | 333 | 0.824257 |

- Masked transitions v1/v2/change: 33 / 89 / 56
- Mask reasons v1: `{"continuity_unknown":9,"mixed_unresolved_transition":24}`
- Mask reasons v2: `{"continuity_unknown":18,"mixed_unresolved_transition":71}`
- Mask reason changes: `{"continuity_unknown":9,"mixed_unresolved_transition":47}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":0,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 0 / 0 samples / 0.000000 h

### `ami_EN2001d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.809321 | 0.175784 | 0.128426 | 0.680154 |
| v2 | 0.657309 | 0.327796 | 0.044254 | 0.608204 |

- Speech segments v1/v2: 686 / 1546
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 641 / 0 / 45 / 0
- Removed internal pause: 10693168 samples (0.185645 h)
- Removed outer padding: 2590928 samples (0.044981 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 960 | 2080 | 3520 | 4960 | 12032 | 49120 |
| End samples | -66032 | -2512 | -1728 | -1056 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 100 | 1440 | 2320 | 3708 | 5694 | 16454 | 66032 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 5 | 0 |
| Gap | 0 | 77 | 0 |
| Overlap | 8 | 14 | 75 |

- Topology episodes v1/v2/matched/added/removed: 400 / 902 / 184 / 718 / 216
- Unchanged/timing-only/topology-changing: 38 / 119 / 961
- Overlap takeover/return changes: -49 / -70
- Short-backchannel change: 53
- Handoffs v1/v2/matched/added/removed: 277 / 382 / 182 / 200 / 95

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 65 | 0.162500 |
| 100 ms | 106 | 0.265000 |
| 200 ms | 159 | 0.397500 |
| 500 ms | 184 | 0.460000 |

- Masked transitions v1/v2/change: 172 / 368 / 196
- Mask reasons v1: `{"complex_overlap_transition":31,"continuity_unknown":108,"mixed_unresolved_transition":33}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":66,"complex_overlap_transition":9,"continuity_unknown":147,"mixed_unresolved_transition":146}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":66,"complex_overlap_transition":-22,"continuity_unknown":39,"mixed_unresolved_transition":113}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":45,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 90 / 1109456 samples / 0.019261 h

### `ami_EN2002c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.735979 | 0.089648 | 0.206377 | 0.529136 |
| v2 | 0.592486 | 0.233141 | 0.080631 | 0.503251 |

- Speech segments v1/v2: 677 / 1891
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 634 / 0 / 43 / 0
- Removed internal pause: 13487040 samples (0.234150 h)
- Removed outer padding: 3472544 samples (0.060287 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 640 | 1720 | 3520 | 5440 | 21339 | 46080 |
| End samples | -90080 | -2780 | -1760 | -1076 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1312 | 2240 | 4480 | 9773 | 36207 | 90080 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 9 | 0 |
| Gap | 0 | 46 | 0 |
| Overlap | 3 | 14 | 71 |

- Topology episodes v1/v2/matched/added/removed: 364 / 864 / 146 / 718 / 218
- Unchanged/timing-only/topology-changing: 34 / 86 / 962
- Overlap takeover/return changes: -34 / -32
- Short-backchannel change: 68
- Handoffs v1/v2/matched/added/removed: 201 / 387 / 127 / 260 / 74

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 55 | 0.151099 |
| 100 ms | 91 | 0.250000 |
| 200 ms | 129 | 0.354396 |
| 500 ms | 146 | 0.401099 |

- Masked transitions v1/v2/change: 163 / 463 / 300
- Mask reasons v1: `{"complex_overlap_transition":87,"continuity_unknown":56,"mixed_unresolved_transition":20}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":190,"complex_overlap_transition":21,"continuity_unknown":65,"mixed_unresolved_transition":187}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":190,"complex_overlap_transition":-66,"continuity_unknown":9,"mixed_unresolved_transition":167}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":43,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 286 / 3663392 samples / 0.063601 h

### `ami_EN2003a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.525522 | 0.096778 | 0.062453 | 0.462429 |
| v2 | 0.412256 | 0.210043 | 0.034437 | 0.374796 |

- Speech segments v1/v2: 487 / 1059
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 455 / 0 / 32 / 0
- Removed internal pause: 5909920 samples (0.102603 h)
- Removed outer padding: 1800704 samples (0.031262 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 1760 | 4096 | 6240 | 14313 | 26560 |
| End samples | -59712 | -2680 | -1472 | 0 | 0 | 0 | 0 | 80 |
| Absolute samples | 0 | 0 | 960 | 2304 | 4643 | 7825 | 17050 | 59712 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 3 | 0 |
| Gap | 0 | 46 | 0 |
| Overlap | 4 | 7 | 61 |

- Topology episodes v1/v2/matched/added/removed: 277 / 574 / 127 / 447 / 150
- Unchanged/timing-only/topology-changing: 54 / 59 / 611
- Overlap takeover/return changes: -19 / -76
- Short-backchannel change: 27
- Handoffs v1/v2/matched/added/removed: 141 / 181 / 87 / 94 / 54

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 68 | 0.245487 |
| 100 ms | 83 | 0.299639 |
| 200 ms | 115 | 0.415162 |
| 500 ms | 127 | 0.458484 |

- Masked transitions v1/v2/change: 139 / 297 / 158
- Mask reasons v1: `{"complex_overlap_transition":24,"continuity_unknown":97,"mixed_unresolved_transition":18}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":94,"complex_overlap_transition":9,"continuity_unknown":109,"initial_start":1,"mixed_unresolved_transition":84}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":94,"complex_overlap_transition":-15,"continuity_unknown":12,"initial_start":1,"mixed_unresolved_transition":66}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":32,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 112 / 975872 samples / 0.016942 h

### `ami_EN2004a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.877426 | 0.079706 | 0.221488 | 0.654872 |
| v2 | 0.716331 | 0.240801 | 0.092614 | 0.616013 |

- Speech segments v1/v2: 1002 / 1988
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 805 / 0 / 197 / 0
- Removed internal pause: 11332880 samples (0.196751 h)
- Removed outer padding: 5308384 samples (0.092159 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 1600 | 3520 | 5600 | 24454 | 48800 |
| End samples | -199936 | -2992 | -1648 | -448 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1120 | 2252 | 5873 | 16647 | 43286 | 199936 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 5 | 0 |
| Gap | 0 | 27 | 0 |
| Overlap | 3 | 6 | 103 |

- Topology episodes v1/v2/matched/added/removed: 410 / 880 / 152 / 728 / 258
- Unchanged/timing-only/topology-changing: 38 / 100 / 1000
- Overlap takeover/return changes: -39 / -95
- Short-backchannel change: 55
- Handoffs v1/v2/matched/added/removed: 190 / 308 / 104 / 204 / 86

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 63 | 0.153659 |
| 100 ms | 100 | 0.243902 |
| 200 ms | 142 | 0.346341 |
| 500 ms | 152 | 0.370732 |

- Masked transitions v1/v2/change: 251 / 566 / 315
- Mask reasons v1: `{"complex_overlap_transition":149,"continuity_unknown":57,"mixed_unresolved_transition":45}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":340,"complex_overlap_transition":27,"continuity_unknown":60,"mixed_unresolved_transition":139}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":340,"complex_overlap_transition":-122,"continuity_unknown":3,"mixed_unresolved_transition":94}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":197,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 565 / 5014944 samples / 0.087065 h

### `ami_EN2006a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.699914 | 0.279390 | 0.166181 | 0.531204 |
| v2 | 0.540584 | 0.438720 | 0.085712 | 0.445856 |

- Speech segments v1/v2: 1249 / 1912
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 1113 / 0 / 136 / 0
- Removed internal pause: 7956960 samples (0.138142 h)
- Removed outer padding: 4916224 samples (0.085351 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 1600 | 2880 | 4480 | 14624 | 45600 |
| End samples | -50624 | -3936 | -1968 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1120 | 2624 | 5280 | 7808 | 21236 | 50624 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 12 | 0 |
| Gap | 0 | 103 | 0 |
| Overlap | 6 | 15 | 84 |

- Topology episodes v1/v2/matched/added/removed: 530 / 735 / 228 / 507 / 302
- Unchanged/timing-only/topology-changing: 90 / 105 / 842
- Overlap takeover/return changes: -66 / -65
- Short-backchannel change: 32
- Handoffs v1/v2/matched/added/removed: 402 / 405 / 223 / 182 / 179

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 139 | 0.262264 |
| 100 ms | 171 | 0.322642 |
| 200 ms | 208 | 0.392453 |
| 500 ms | 228 | 0.430189 |

- Masked transitions v1/v2/change: 434 / 653 / 219
- Mask reasons v1: `{"complex_overlap_transition":87,"continuity_unknown":265,"mixed_unresolved_transition":82}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":189,"complex_overlap_transition":21,"continuity_unknown":229,"mixed_unresolved_transition":214}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":189,"complex_overlap_transition":-66,"continuity_unknown":-36,"mixed_unresolved_transition":132}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":136,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 260 / 2592400 samples / 0.045007 h

### `ami_EN2009d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 1.307494 | 0.171499 | 0.282785 | 1.022897 |
| v2 | 1.091608 | 0.387385 | 0.143163 | 0.936277 |

- Speech segments v1/v2: 1245 / 2981
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 1129 / 0 / 116 / 0
- Removed internal pause: 16412192 samples (0.284934 h)
- Removed outer padding: 4186528 samples (0.072683 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 960 | 1600 | 2880 | 4640 | 13670 | 36000 |
| End samples | -87312 | -2336 | -1488 | -400 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1152 | 2048 | 3428 | 5234 | 16505 | 87312 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 5 | 0 |
| Gap | 0 | 56 | 0 |
| Overlap | 11 | 17 | 202 |

- Topology episodes v1/v2/matched/added/removed: 663 / 1580 / 297 / 1283 / 366
- Unchanged/timing-only/topology-changing: 81 / 183 / 1682
- Overlap takeover/return changes: -56 / -101
- Short-backchannel change: 77
- Handoffs v1/v2/matched/added/removed: 313 / 501 / 187 / 314 / 126

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 119 | 0.179487 |
| 100 ms | 194 | 0.292609 |
| 200 ms | 265 | 0.399698 |
| 500 ms | 297 | 0.447964 |

- Masked transitions v1/v2/change: 287 / 663 / 376
- Mask reasons v1: `{"complex_overlap_transition":128,"continuity_unknown":89,"mixed_unresolved_transition":70}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":214,"complex_overlap_transition":47,"continuity_unknown":92,"mixed_unresolved_transition":310}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":214,"complex_overlap_transition":-81,"continuity_unknown":3,"mixed_unresolved_transition":240}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":116,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 274 / 2498928 samples / 0.043384 h

### `ami_ES2002b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.549803 | 0.083462 | 0.093044 | 0.456508 |
| v2 | 0.445043 | 0.188222 | 0.034686 | 0.405564 |

- Speech segments v1/v2: 439 / 1206
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 413 / 0 / 26 / 0
- Removed internal pause: 6911856 samples (0.119998 h)
- Removed outer padding: 2714768 samples (0.047131 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 960 | 2720 | 4480 | 7360 | 10080 | 14764 | 25120 |
| End samples | -51664 | -3936 | -1984 | -672 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 712 | 2384 | 4320 | 7296 | 10080 | 21992 | 51664 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 7 | 0 |
| Gap | 0 | 6 | 0 |
| Overlap | 0 | 9 | 65 |

- Topology episodes v1/v2/matched/added/removed: 246 / 679 / 87 / 592 / 159
- Unchanged/timing-only/topology-changing: 20 / 51 / 767
- Overlap takeover/return changes: -12 / -88
- Short-backchannel change: 61
- Handoffs v1/v2/matched/added/removed: 79 / 220 / 44 / 176 / 35

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 34 | 0.138211 |
| 100 ms | 46 | 0.186992 |
| 200 ms | 67 | 0.272358 |
| 500 ms | 87 | 0.353659 |

- Masked transitions v1/v2/change: 98 / 242 / 144
- Mask reasons v1: `{"complex_overlap_transition":60,"continuity_unknown":27,"mixed_unresolved_transition":11}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":77,"complex_overlap_transition":11,"continuity_unknown":32,"mixed_unresolved_transition":122}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":77,"complex_overlap_transition":-49,"continuity_unknown":5,"mixed_unresolved_transition":111}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":26,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 84 / 748880 samples / 0.013001 h

### `ami_ES2003a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.197112 | 0.119490 | 0.011128 | 0.185740 |
| v2 | 0.140270 | 0.176332 | 0.003414 | 0.135601 |

- Speech segments v1/v2: 134 / 361
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 114 / 0 / 20 / 0
- Removed internal pause: 2500640 samples (0.043414 h)
- Removed outer padding: 815040 samples (0.014150 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 480 | 1760 | 3760 | 7472 | 12456 | 29656 | 31840 |
| End samples | -25184 | -4864 | -2272 | -584 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 480 | 2032 | 4160 | 9536 | 13067 | 27940 | 31840 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 0 | 0 |
| Gap | 0 | 8 | 0 |
| Overlap | 0 | 7 | 6 |

- Topology episodes v1/v2/matched/added/removed: 55 / 210 / 23 / 187 / 32
- Unchanged/timing-only/topology-changing: 4 / 12 / 226
- Overlap takeover/return changes: -7 / -17
- Short-backchannel change: 3
- Handoffs v1/v2/matched/added/removed: 40 / 39 / 25 / 14 / 15

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 9 | 0.163636 |
| 100 ms | 13 | 0.236364 |
| 200 ms | 18 | 0.327273 |
| 500 ms | 23 | 0.418182 |

- Masked transitions v1/v2/change: 55 / 100 / 45
- Mask reasons v1: `{"complex_overlap_transition":4,"continuity_unknown":45,"mixed_unresolved_transition":6}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":32,"continuity_unknown":42,"mixed_unresolved_transition":26}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":32,"complex_overlap_transition":-4,"continuity_unknown":-3,"mixed_unresolved_transition":20}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":20,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 39 / 529632 samples / 0.009195 h

### `ami_ES2004a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.229950 | 0.061538 | 0.051000 | 0.178322 |
| v2 | 0.178031 | 0.113457 | 0.017582 | 0.158286 |

- Speech segments v1/v2: 283 / 552
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 260 / 0 / 23 / 0
- Removed internal pause: 2906720 samples (0.050464 h)
- Removed outer padding: 2080128 samples (0.036113 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1440 | 4320 | 6608 | 10096 | 22408 | 31520 |
| End samples | -38496 | -6064 | -3080 | -540 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 2400 | 5280 | 9464 | 15292 | 30455 | 38496 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 1 | 0 |
| Gap | 0 | 5 | 0 |
| Overlap | 1 | 3 | 24 |

- Topology episodes v1/v2/matched/added/removed: 100 / 262 / 36 / 226 / 64
- Unchanged/timing-only/topology-changing: 5 / 26 / 295
- Overlap takeover/return changes: -1 / -21
- Short-backchannel change: 21
- Handoffs v1/v2/matched/added/removed: 52 / 106 / 25 / 81 / 27

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 11 | 0.110000 |
| 100 ms | 15 | 0.150000 |
| 200 ms | 24 | 0.240000 |
| 500 ms | 36 | 0.360000 |

- Masked transitions v1/v2/change: 98 / 148 / 50
- Mask reasons v1: `{"complex_overlap_transition":34,"continuity_unknown":43,"mixed_unresolved_transition":21}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":45,"complex_overlap_transition":6,"continuity_unknown":43,"initial_start":1,"mixed_unresolved_transition":53}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":45,"complex_overlap_transition":-28,"continuity_unknown":0,"initial_start":1,"mixed_unresolved_transition":32}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":23,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 60 / 657888 samples / 0.011422 h

### `ami_ES2005a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.075109 | 0.057634 | 0.015393 | 0.059591 |
| v2 | 0.053086 | 0.079658 | 0.004848 | 0.047344 |

- Speech segments v1/v2: 90 / 177
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 75 / 0 / 15 / 0
- Removed internal pause: 1080400 samples (0.018757 h)
- Removed outer padding: 541456 samples (0.009400 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2800 | 7552 | 14336 | 20115 | 22720 |
| End samples | -52256 | -6080 | -1616 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1104 | 3796 | 8945 | 15463 | 28809 | 52256 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 0 | 0 |
| Gap | 0 | 4 | 0 |
| Overlap | 0 | 0 | 3 |

- Topology episodes v1/v2/matched/added/removed: 42 / 82 / 8 / 74 / 34
- Unchanged/timing-only/topology-changing: 2 / 6 / 108
- Overlap takeover/return changes: -2 / -15
- Short-backchannel change: 5
- Handoffs v1/v2/matched/added/removed: 22 / 30 / 7 / 23 / 15

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 4 | 0.095238 |
| 100 ms | 5 | 0.119048 |
| 200 ms | 5 | 0.119048 |
| 500 ms | 8 | 0.190476 |

- Masked transitions v1/v2/change: 25 / 46 / 21
- Mask reasons v1: `{"complex_overlap_transition":13,"continuity_unknown":9,"mixed_unresolved_transition":3}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":16,"complex_overlap_transition":1,"continuity_unknown":6,"mixed_unresolved_transition":23}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":16,"complex_overlap_transition":-12,"continuity_unknown":-3,"mixed_unresolved_transition":20}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":15,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 23 / 373728 samples / 0.006488 h

### `ami_ES2005b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.589564 | 0.053008 | 0.109043 | 0.479319 |
| v2 | 0.450649 | 0.191923 | 0.035566 | 0.409659 |

- Speech segments v1/v2: 590 / 1274
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 496 / 0 / 94 / 0
- Removed internal pause: 7322080 samples (0.127119 h)
- Removed outer padding: 3608384 samples (0.062646 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 3400 | 7200 | 12320 | 26416 | 50240 |
| End samples | -116128 | -4896 | -1056 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 656 | 4160 | 9400 | 15556 | 33787 | 116128 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 3 | 0 |
| Gap | 0 | 28 | 0 |
| Overlap | 1 | 10 | 30 |

- Topology episodes v1/v2/matched/added/removed: 278 / 626 / 80 / 546 / 198
- Unchanged/timing-only/topology-changing: 26 / 40 / 758
- Overlap takeover/return changes: -21 / -97
- Short-backchannel change: 23
- Handoffs v1/v2/matched/added/removed: 161 / 212 / 82 / 130 / 79

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 44 | 0.158273 |
| 100 ms | 49 | 0.176259 |
| 200 ms | 57 | 0.205036 |
| 500 ms | 80 | 0.287770 |

- Masked transitions v1/v2/change: 144 / 348 / 204
- Mask reasons v1: `{"complex_overlap_transition":55,"continuity_unknown":50,"mixed_unresolved_transition":39}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":153,"complex_overlap_transition":13,"continuity_unknown":55,"initial_start":1,"mixed_unresolved_transition":126}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":153,"complex_overlap_transition":-42,"continuity_unknown":5,"initial_start":1,"mixed_unresolved_transition":87}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":94,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 200 / 2527040 samples / 0.043872 h

### `ami_ES2005c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.584676 | 0.053075 | 0.137888 | 0.445269 |
| v2 | 0.460985 | 0.176766 | 0.054705 | 0.400612 |

- Speech segments v1/v2: 690 / 1370
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 610 / 0 / 80 / 0
- Removed internal pause: 7088704 samples (0.123068 h)
- Removed outer padding: 4385120 samples (0.076131 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 2240 | 4960 | 9688 | 23987 | 43680 |
| End samples | -76640 | -4736 | -1520 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 528 | 3348 | 9046 | 19056 | 43229 | 76640 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 3 | 0 |
| Gap | 0 | 44 | 0 |
| Overlap | 3 | 9 | 48 |

- Topology episodes v1/v2/matched/added/removed: 313 / 637 / 114 / 523 / 199
- Unchanged/timing-only/topology-changing: 42 / 57 / 737
- Overlap takeover/return changes: -25 / -61
- Short-backchannel change: 40
- Handoffs v1/v2/matched/added/removed: 200 / 281 / 107 / 174 / 93

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 69 | 0.220447 |
| 100 ms | 76 | 0.242812 |
| 200 ms | 96 | 0.306709 |
| 500 ms | 114 | 0.364217 |

- Masked transitions v1/v2/change: 163 / 356 / 193
- Mask reasons v1: `{"complex_overlap_transition":75,"continuity_unknown":40,"mixed_unresolved_transition":48}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":185,"complex_overlap_transition":13,"continuity_unknown":46,"mixed_unresolved_transition":112}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":185,"complex_overlap_transition":-62,"continuity_unknown":6,"mixed_unresolved_transition":64}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":80,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 237 / 2674336 samples / 0.046429 h

### `ami_ES2005d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.378289 | 0.102594 | 0.096179 | 0.281201 |
| v2 | 0.285508 | 0.195375 | 0.028929 | 0.253305 |

- Speech segments v1/v2: 483 / 876
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 392 / 0 / 91 / 0
- Removed internal pause: 5012208 samples (0.087017 h)
- Removed outer padding: 3170592 samples (0.055045 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2880 | 6544 | 10952 | 34574 | 68160 |
| End samples | -74816 | -6688 | -2448 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1120 | 4640 | 10192 | 15577 | 44972 | 74816 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 1 | 0 |
| Gap | 0 | 26 | 0 |
| Overlap | 0 | 7 | 22 |

- Topology episodes v1/v2/matched/added/removed: 209 / 384 / 61 / 323 / 148
- Unchanged/timing-only/topology-changing: 28 / 25 / 479
- Overlap takeover/return changes: -21 / -56
- Short-backchannel change: 25
- Handoffs v1/v2/matched/added/removed: 130 / 176 / 63 / 113 / 67

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 36 | 0.172249 |
| 100 ms | 38 | 0.181818 |
| 200 ms | 43 | 0.205742 |
| 500 ms | 61 | 0.291866 |

- Masked transitions v1/v2/change: 120 / 255 / 135
- Mask reasons v1: `{"complex_overlap_transition":49,"continuity_unknown":42,"mixed_unresolved_transition":29}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":140,"complex_overlap_transition":4,"continuity_unknown":51,"mixed_unresolved_transition":60}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":140,"complex_overlap_transition":-45,"continuity_unknown":9,"mixed_unresolved_transition":31}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":91,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 204 / 2527936 samples / 0.043888 h

### `ami_ES2006a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.296366 | 0.060396 | 0.050417 | 0.245333 |
| v2 | 0.215624 | 0.141137 | 0.011659 | 0.202174 |

- Speech segments v1/v2: 287 / 565
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 214 / 0 / 73 / 0
- Removed internal pause: 3590240 samples (0.062331 h)
- Removed outer padding: 2519456 samples (0.043741 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1840 | 3840 | 8768 | 16512 | 24139 | 448032 |
| End samples | -57376 | -6984 | -3504 | -480 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 2560 | 5940 | 12521 | 20118 | 46284 | 448032 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 4 | 0 |
| Gap | 0 | 6 | 0 |
| Overlap | 0 | 4 | 13 |

- Topology episodes v1/v2/matched/added/removed: 110 / 270 / 28 / 242 / 82
- Unchanged/timing-only/topology-changing: 8 / 12 / 332
- Overlap takeover/return changes: -13 / -39
- Short-backchannel change: 4
- Handoffs v1/v2/matched/added/removed: 59 / 59 / 22 / 37 / 37

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 11 | 0.100000 |
| 100 ms | 13 | 0.118182 |
| 200 ms | 20 | 0.181818 |
| 500 ms | 28 | 0.254545 |

- Masked transitions v1/v2/change: 88 / 177 / 89
- Mask reasons v1: `{"complex_overlap_transition":28,"continuity_unknown":41,"mixed_unresolved_transition":19}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":89,"complex_overlap_transition":3,"continuity_unknown":53,"mixed_unresolved_transition":32}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":89,"complex_overlap_transition":-25,"continuity_unknown":12,"mixed_unresolved_transition":13}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":73,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 149 / 1661184 samples / 0.028840 h

### `ami_ES2006b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.552673 | 0.053751 | 0.086054 | 0.466027 |
| v2 | 0.445561 | 0.160862 | 0.038979 | 0.402268 |

- Speech segments v1/v2: 446 / 1131
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 406 / 0 / 40 / 0
- Removed internal pause: 6317264 samples (0.109675 h)
- Removed outer padding: 2509184 samples (0.043562 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 640 | 3040 | 6240 | 9600 | 20560 | 37920 |
| End samples | -52480 | -4604 | -2240 | -544 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1616 | 3760 | 7624 | 11751 | 23757 | 52480 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 0 | 0 |
| Gap | 0 | 16 | 0 |
| Overlap | 2 | 6 | 46 |

- Topology episodes v1/v2/matched/added/removed: 200 / 603 / 71 / 532 / 129
- Unchanged/timing-only/topology-changing: 14 / 49 / 669
- Overlap takeover/return changes: -9 / -44
- Short-backchannel change: 25
- Handoffs v1/v2/matched/added/removed: 103 / 166 / 55 / 111 / 48

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 22 | 0.110000 |
| 100 ms | 29 | 0.145000 |
| 200 ms | 44 | 0.220000 |
| 500 ms | 71 | 0.355000 |

- Masked transitions v1/v2/change: 126 / 289 / 163
- Mask reasons v1: `{"complex_overlap_transition":60,"continuity_unknown":45,"mixed_unresolved_transition":21}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":142,"complex_overlap_transition":17,"continuity_unknown":37,"mixed_unresolved_transition":93}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":142,"complex_overlap_transition":-43,"continuity_unknown":-8,"mixed_unresolved_transition":72}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":40,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 177 / 1616096 samples / 0.028057 h

### `ami_ES2006c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.547303 | 0.058720 | 0.107387 | 0.439200 |
| v2 | 0.446966 | 0.159057 | 0.048238 | 0.393318 |

- Speech segments v1/v2: 527 / 1204
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 492 / 0 / 35 / 0
- Removed internal pause: 6454912 samples (0.112064 h)
- Removed outer padding: 3018640 samples (0.052407 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 960 | 3520 | 6208 | 8712 | 16960 | 39200 |
| End samples | -74608 | -5352 | -2120 | 0 | 0 | 0 | 0 | 80 |
| Absolute samples | 0 | 0 | 1472 | 4160 | 8257 | 11852 | 18434 | 74608 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 3 | 0 |
| Gap | 0 | 16 | 0 |
| Overlap | 1 | 6 | 40 |

- Topology episodes v1/v2/matched/added/removed: 244 / 524 / 73 / 451 / 171
- Unchanged/timing-only/topology-changing: 25 / 38 / 632
- Overlap takeover/return changes: -21 / -86
- Short-backchannel change: 40
- Handoffs v1/v2/matched/added/removed: 104 / 200 / 45 / 155 / 59

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 34 | 0.139344 |
| 100 ms | 41 | 0.168033 |
| 200 ms | 52 | 0.213115 |
| 500 ms | 73 | 0.299180 |

- Masked transitions v1/v2/change: 140 / 360 / 220
- Mask reasons v1: `{"complex_overlap_transition":67,"continuity_unknown":44,"mixed_unresolved_transition":29}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":172,"complex_overlap_transition":10,"continuity_unknown":55,"mixed_unresolved_transition":123}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":172,"complex_overlap_transition":-57,"continuity_unknown":11,"mixed_unresolved_transition":94}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":35,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 208 / 1796176 samples / 0.031184 h

### `ami_ES2006d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.482317 | 0.064196 | 0.155202 | 0.325505 |
| v2 | 0.391345 | 0.155168 | 0.069769 | 0.315918 |

- Speech segments v1/v2: 764 / 1206
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 653 / 0 / 111 / 0
- Removed internal pause: 5320960 samples (0.092378 h)
- Removed outer padding: 4769392 samples (0.082802 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 2880 | 5888 | 9024 | 18336 | 40480 |
| End samples | -94368 | -6080 | -2960 | -576 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1528 | 4480 | 8960 | 13080 | 36570 | 94368 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 3 | 0 |
| Gap | 0 | 32 | 0 |
| Overlap | 1 | 9 | 46 |

- Topology episodes v1/v2/matched/added/removed: 254 / 454 / 94 / 360 / 160
- Unchanged/timing-only/topology-changing: 32 / 49 / 533
- Overlap takeover/return changes: -8 / -43
- Short-backchannel change: 24
- Handoffs v1/v2/matched/added/removed: 170 / 226 / 87 / 139 / 83

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 44 | 0.173228 |
| 100 ms | 52 | 0.204724 |
| 200 ms | 71 | 0.279528 |
| 500 ms | 94 | 0.370079 |

- Masked transitions v1/v2/change: 207 / 364 / 157
- Mask reasons v1: `{"complex_overlap_transition":106,"continuity_unknown":42,"mixed_unresolved_transition":59}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":168,"complex_overlap_transition":32,"continuity_unknown":48,"initial_start":1,"mixed_unresolved_transition":115}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":168,"complex_overlap_transition":-74,"continuity_unknown":6,"initial_start":1,"mixed_unresolved_transition":56}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":111,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 270 / 2969264 samples / 0.051550 h

### `ami_ES2007a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.245608 | 0.089477 | 0.038961 | 0.205896 |
| v2 | 0.174473 | 0.160611 | 0.011867 | 0.160846 |

- Speech segments v1/v2: 346 / 561
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 304 / 0 / 42 / 0
- Removed internal pause: 3088000 samples (0.053611 h)
- Removed outer padding: 2104944 samples (0.036544 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 560 | 3040 | 5072 | 7312 | 33443 | 38720 |
| End samples | -61536 | -4776 | -2200 | -512 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1344 | 4000 | 7270 | 13600 | 38533 | 61536 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 2 | 0 |
| Gap | 0 | 22 | 0 |
| Overlap | 2 | 3 | 10 |

- Topology episodes v1/v2/matched/added/removed: 131 / 239 / 46 / 193 / 85
- Unchanged/timing-only/topology-changing: 13 / 26 / 285
- Overlap takeover/return changes: -18 / -23
- Short-backchannel change: 9
- Handoffs v1/v2/matched/added/removed: 111 / 124 / 66 / 58 / 45

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 25 | 0.190840 |
| 100 ms | 28 | 0.213740 |
| 200 ms | 36 | 0.274809 |
| 500 ms | 46 | 0.351145 |

- Masked transitions v1/v2/change: 117 / 189 / 72
- Mask reasons v1: `{"complex_overlap_transition":18,"continuity_unknown":71,"mixed_unresolved_transition":28}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":71,"complex_overlap_transition":1,"continuity_unknown":77,"mixed_unresolved_transition":40}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":71,"complex_overlap_transition":-17,"continuity_unknown":6,"mixed_unresolved_transition":12}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":42,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 93 / 993712 samples / 0.017252 h

### `ami_ES2007b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.390025 | 0.078126 | 0.060176 | 0.329464 |
| v2 | 0.289722 | 0.178429 | 0.020796 | 0.266089 |

- Speech segments v1/v2: 363 / 792
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 321 / 0 / 42 / 0
- Removed internal pause: 5012000 samples (0.087014 h)
- Removed outer padding: 2738912 samples (0.047551 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 800 | 3200 | 5600 | 8160 | 22784 | 40320 |
| End samples | -102944 | -5712 | -2720 | -544 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1760 | 4420 | 8696 | 13660 | 48949 | 102944 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 2 | 0 |
| Gap | 0 | 32 | 0 |
| Overlap | 0 | 5 | 25 |

- Topology episodes v1/v2/matched/added/removed: 191 / 410 / 67 / 343 / 124
- Unchanged/timing-only/topology-changing: 21 / 39 / 474
- Overlap takeover/return changes: -27 / -48
- Short-backchannel change: 10
- Handoffs v1/v2/matched/added/removed: 106 / 114 / 51 / 63 / 55

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 34 | 0.178010 |
| 100 ms | 43 | 0.225131 |
| 200 ms | 57 | 0.298429 |
| 500 ms | 67 | 0.350785 |

- Masked transitions v1/v2/change: 95 / 244 / 149
- Mask reasons v1: `{"complex_overlap_transition":26,"continuity_unknown":53,"mixed_unresolved_transition":16}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":99,"complex_overlap_transition":3,"continuity_unknown":77,"mixed_unresolved_transition":65}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":99,"complex_overlap_transition":-23,"continuity_unknown":24,"mixed_unresolved_transition":49}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":42,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 120 / 1235152 samples / 0.021444 h

### `ami_ES2007c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.552488 | 0.107903 | 0.091277 | 0.460431 |
| v2 | 0.415732 | 0.244659 | 0.030097 | 0.382512 |

- Speech segments v1/v2: 583 / 1111
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 502 / 0 / 81 / 0
- Removed internal pause: 6608000 samples (0.114722 h)
- Removed outer padding: 3773296 samples (0.065509 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2880 | 5120 | 8136 | 22558 | 54240 |
| End samples | -83856 | -4588 | -1968 | -480 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1352 | 3812 | 8300 | 16187 | 44572 | 83856 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 2 | 0 |
| Gap | 0 | 42 | 0 |
| Overlap | 3 | 6 | 35 |

- Topology episodes v1/v2/matched/added/removed: 259 / 562 / 99 / 463 / 160
- Unchanged/timing-only/topology-changing: 35 / 53 / 634
- Overlap takeover/return changes: -19 / -55
- Short-backchannel change: 17
- Handoffs v1/v2/matched/added/removed: 171 / 187 / 89 / 98 / 82

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 54 | 0.208494 |
| 100 ms | 65 | 0.250965 |
| 200 ms | 78 | 0.301158 |
| 500 ms | 99 | 0.382239 |

- Masked transitions v1/v2/change: 172 / 334 / 162
- Mask reasons v1: `{"complex_overlap_transition":48,"continuity_unknown":90,"mixed_unresolved_transition":34}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":148,"complex_overlap_transition":9,"continuity_unknown":105,"mixed_unresolved_transition":72}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":148,"complex_overlap_transition":-39,"continuity_unknown":15,"mixed_unresolved_transition":38}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":81,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 194 / 2554464 samples / 0.044348 h

### `ami_ES2007d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.299908 | 0.047179 | 0.064282 | 0.234953 |
| v2 | 0.213886 | 0.133201 | 0.019904 | 0.191079 |

- Speech segments v1/v2: 406 / 704
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 349 / 0 / 57 / 0
- Removed internal pause: 4139200 samples (0.071861 h)
- Removed outer padding: 2739856 samples (0.047567 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 3040 | 6272 | 8800 | 35449 | 49120 |
| End samples | -57696 | -6080 | -2288 | -512 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1384 | 4632 | 8732 | 17811 | 38894 | 57696 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 5 | 0 |
| Gap | 0 | 26 | 0 |
| Overlap | 1 | 4 | 20 |

- Topology episodes v1/v2/matched/added/removed: 182 / 309 / 63 / 246 / 119
- Unchanged/timing-only/topology-changing: 23 / 30 / 375
- Overlap takeover/return changes: -21 / -34
- Short-backchannel change: 6
- Handoffs v1/v2/matched/added/removed: 133 / 130 / 75 / 55 / 58

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 36 | 0.197802 |
| 100 ms | 45 | 0.247253 |
| 200 ms | 55 | 0.302198 |
| 500 ms | 63 | 0.346154 |

- Masked transitions v1/v2/change: 106 / 228 / 122
- Mask reasons v1: `{"complex_overlap_transition":44,"continuity_unknown":39,"mixed_unresolved_transition":23}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":92,"complex_overlap_transition":8,"continuity_unknown":66,"mixed_unresolved_transition":62}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":92,"complex_overlap_transition":-36,"continuity_unknown":27,"mixed_unresolved_transition":39}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":57,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 122 / 1472688 samples / 0.025567 h

### `ami_ES2008a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.226031 | 0.063792 | 0.016605 | 0.209032 |
| v2 | 0.176317 | 0.113506 | 0.005522 | 0.169726 |

- Speech segments v1/v2: 193 / 430
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 167 / 0 / 26 / 0
- Removed internal pause: 2458400 samples (0.042681 h)
- Removed outer padding: 765472 samples (0.013289 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 1760 | 5600 | 8752 | 32003 | 48800 |
| End samples | -29520 | -3448 | -576 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 2560 | 6139 | 9640 | 27335 | 48800 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 3 | 0 |
| Gap | 0 | 26 | 0 |
| Overlap | 0 | 1 | 13 |

- Topology episodes v1/v2/matched/added/removed: 97 / 255 / 44 / 211 / 53
- Unchanged/timing-only/topology-changing: 29 / 11 / 268
- Overlap takeover/return changes: -4 / -31
- Short-backchannel change: 3
- Handoffs v1/v2/matched/added/removed: 41 / 50 / 27 / 23 / 14

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 33 | 0.340206 |
| 100 ms | 37 | 0.381443 |
| 200 ms | 39 | 0.402062 |
| 500 ms | 44 | 0.453608 |

- Masked transitions v1/v2/change: 65 / 114 / 49
- Mask reasons v1: `{"complex_overlap_transition":6,"continuity_unknown":48,"mixed_unresolved_transition":11}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":45,"complex_overlap_transition":2,"continuity_unknown":46,"initial_start":1,"mixed_unresolved_transition":20}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":45,"complex_overlap_transition":-4,"continuity_unknown":-2,"initial_start":1,"mixed_unresolved_transition":9}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":26,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 62 / 633472 samples / 0.010998 h

### `ami_ES2008b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.498456 | 0.121449 | 0.045717 | 0.451949 |
| v2 | 0.419669 | 0.200237 | 0.019708 | 0.397057 |

- Speech segments v1/v2: 470 / 986
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 435 / 0 / 35 / 0
- Removed internal pause: 4387104 samples (0.076165 h)
- Removed outer padding: 1155712 samples (0.020064 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 1280 | 3040 | 3984 | 6931 | 18240 |
| End samples | -30080 | -2136 | -512 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 1760 | 4049 | 5905 | 11408 | 30080 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 2 | 0 |
| Gap | 0 | 67 | 0 |
| Overlap | 4 | 6 | 34 |

- Topology episodes v1/v2/matched/added/removed: 223 / 600 / 119 / 481 / 104
- Unchanged/timing-only/topology-changing: 68 / 39 / 597
- Overlap takeover/return changes: -20 / -39
- Short-backchannel change: 17
- Handoffs v1/v2/matched/added/removed: 139 / 178 / 93 / 85 / 46

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 84 | 0.376682 |
| 100 ms | 89 | 0.399103 |
| 200 ms | 105 | 0.470852 |
| 500 ms | 119 | 0.533632 |

- Masked transitions v1/v2/change: 138 / 214 / 76
- Mask reasons v1: `{"complex_overlap_transition":24,"continuity_unknown":79,"mixed_unresolved_transition":35}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":40,"complex_overlap_transition":11,"continuity_unknown":84,"initial_start":1,"mixed_unresolved_transition":78}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":40,"complex_overlap_transition":-13,"continuity_unknown":5,"initial_start":1,"mixed_unresolved_transition":43}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":35,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 55 / 714080 samples / 0.012397 h

### `ami_ES2008c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.499506 | 0.084555 | 0.084939 | 0.413914 |
| v2 | 0.408814 | 0.175247 | 0.031946 | 0.373745 |

- Speech segments v1/v2: 422 / 974
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 388 / 0 / 34 / 0
- Removed internal pause: 5398112 samples (0.093717 h)
- Removed outer padding: 2606416 samples (0.045250 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1600 | 3840 | 6288 | 8480 | 15614 | 37600 |
| End samples | -67808 | -5312 | -2656 | -476 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 2128 | 4568 | 7288 | 9604 | 22452 | 67808 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 4 | 0 |
| Gap | 0 | 44 | 0 |
| Overlap | 3 | 10 | 35 |

- Topology episodes v1/v2/matched/added/removed: 208 / 533 / 101 / 432 / 107
- Unchanged/timing-only/topology-changing: 36 / 48 / 556
- Overlap takeover/return changes: -21 / -27
- Short-backchannel change: 26
- Handoffs v1/v2/matched/added/removed: 108 / 167 / 69 / 98 / 39

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 48 | 0.230769 |
| 100 ms | 56 | 0.269231 |
| 200 ms | 77 | 0.370192 |
| 500 ms | 101 | 0.485577 |

- Masked transitions v1/v2/change: 110 / 221 / 111
- Mask reasons v1: `{"complex_overlap_transition":40,"continuity_unknown":47,"mixed_unresolved_transition":23}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":90,"complex_overlap_transition":13,"continuity_unknown":52,"mixed_unresolved_transition":66}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":90,"complex_overlap_transition":-27,"continuity_unknown":5,"mixed_unresolved_transition":43}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":34,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 124 / 1614192 samples / 0.028024 h

### `ami_ES2008d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.616118 | 0.113277 | 0.103384 | 0.511191 |
| v2 | 0.507671 | 0.221724 | 0.046209 | 0.455782 |

- Speech segments v1/v2: 836 / 1404
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 735 / 0 / 101 / 0
- Removed internal pause: 5960928 samples (0.103488 h)
- Removed outer padding: 2636992 samples (0.045781 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 1600 | 3840 | 6080 | 18819 | 44960 |
| End samples | -50848 | -2280 | 0 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 1920 | 4800 | 7698 | 22781 | 50848 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 2 | 0 |
| Gap | 0 | 99 | 0 |
| Overlap | 1 | 7 | 60 |

- Topology episodes v1/v2/matched/added/removed: 382 / 711 / 182 / 529 / 200
- Unchanged/timing-only/topology-changing: 118 / 54 / 739
- Overlap takeover/return changes: -23 / -65
- Short-backchannel change: 20
- Handoffs v1/v2/matched/added/removed: 260 / 281 / 155 / 126 / 105

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 142 | 0.371728 |
| 100 ms | 149 | 0.390052 |
| 200 ms | 164 | 0.429319 |
| 500 ms | 182 | 0.476440 |

- Masked transitions v1/v2/change: 216 / 362 / 146
- Mask reasons v1: `{"complex_overlap_transition":63,"continuity_unknown":92,"mixed_unresolved_transition":61}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":124,"complex_overlap_transition":3,"continuity_unknown":108,"mixed_unresolved_transition":127}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":124,"complex_overlap_transition":-60,"continuity_unknown":16,"mixed_unresolved_transition":66}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":101,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 200 / 2347104 samples / 0.040748 h

### `ami_ES2009a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.352307 | 0.037195 | 0.061611 | 0.290111 |
| v2 | 0.283431 | 0.106072 | 0.024784 | 0.255835 |

- Speech segments v1/v2: 384 / 742
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 326 / 0 / 58 / 0
- Removed internal pause: 3911456 samples (0.067907 h)
- Removed outer padding: 1909872 samples (0.033157 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2680 | 5840 | 10600 | 22200 | 48480 |
| End samples | -67232 | -3852 | -576 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 488 | 3328 | 7459 | 13072 | 36697 | 67232 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 5 | 3 | 0 |
| Gap | 0 | 25 | 0 |
| Overlap | 1 | 5 | 35 |

- Topology episodes v1/v2/matched/added/removed: 172 / 350 / 78 / 272 / 94
- Unchanged/timing-only/topology-changing: 29 / 40 / 375
- Overlap takeover/return changes: -5 / -36
- Short-backchannel change: 23
- Handoffs v1/v2/matched/added/removed: 103 / 142 / 63 / 79 / 40

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 45 | 0.261628 |
| 100 ms | 51 | 0.296512 |
| 200 ms | 65 | 0.377907 |
| 500 ms | 78 | 0.453488 |

- Masked transitions v1/v2/change: 94 / 211 / 117
- Mask reasons v1: `{"complex_overlap_transition":47,"continuity_unknown":27,"mixed_unresolved_transition":20}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":112,"complex_overlap_transition":10,"continuity_unknown":31,"initial_start":1,"mixed_unresolved_transition":57}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":112,"complex_overlap_transition":-37,"continuity_unknown":4,"initial_start":1,"mixed_unresolved_transition":37}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":58,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 160 / 1547568 samples / 0.026867 h

### `ami_ES2009b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.354941 | 0.043761 | 0.041690 | 0.312682 |
| v2 | 0.287144 | 0.111558 | 0.016072 | 0.269538 |

- Speech segments v1/v2: 276 / 624
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 248 / 0 / 28 / 0
- Removed internal pause: 3509072 samples (0.060921 h)
- Removed outer padding: 1691904 samples (0.029373 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1200 | 2920 | 6976 | 11136 | 21600 | 27040 |
| End samples | -66128 | -4760 | -1768 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1440 | 4012 | 8320 | 12664 | 37204 | 66128 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 3 | 0 |
| Gap | 0 | 17 | 0 |
| Overlap | 1 | 2 | 38 |

- Topology episodes v1/v2/matched/added/removed: 121 / 358 / 63 / 295 / 58
- Unchanged/timing-only/topology-changing: 24 / 33 / 359
- Overlap takeover/return changes: -12 / -19
- Short-backchannel change: 11
- Handoffs v1/v2/matched/added/removed: 64 / 79 / 39 / 40 / 25

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 32 | 0.264463 |
| 100 ms | 35 | 0.289256 |
| 200 ms | 45 | 0.371901 |
| 500 ms | 63 | 0.520661 |

- Masked transitions v1/v2/change: 79 / 170 / 91
- Mask reasons v1: `{"complex_overlap_transition":27,"continuity_unknown":36,"mixed_unresolved_transition":16}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":81,"complex_overlap_transition":7,"continuity_unknown":41,"mixed_unresolved_transition":41}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":81,"complex_overlap_transition":-20,"continuity_unknown":5,"mixed_unresolved_transition":25}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":28,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 94 / 1028096 samples / 0.017849 h

### `ami_ES2009c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.504784 | 0.038807 | 0.066015 | 0.438171 |
| v2 | 0.404249 | 0.139342 | 0.024774 | 0.376845 |

- Speech segments v1/v2: 452 / 961
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 378 / 0 / 74 / 0
- Removed internal pause: 5327200 samples (0.092486 h)
- Removed outer padding: 2024784 samples (0.035153 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2880 | 7408 | 10264 | 24713 | 39360 |
| End samples | -59040 | -4028 | -1056 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 800 | 3276 | 7520 | 11040 | 23921 | 59040 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 4 | 8 | 0 |
| Gap | 0 | 37 | 0 |
| Overlap | 0 | 4 | 47 |

- Topology episodes v1/v2/matched/added/removed: 227 / 568 / 101 / 467 / 126
- Unchanged/timing-only/topology-changing: 43 / 46 / 605
- Overlap takeover/return changes: -14 / -60
- Short-backchannel change: 29
- Handoffs v1/v2/matched/added/removed: 123 / 160 / 73 / 87 / 50

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 57 | 0.251101 |
| 100 ms | 66 | 0.290749 |
| 200 ms | 87 | 0.383260 |
| 500 ms | 101 | 0.444934 |

- Masked transitions v1/v2/change: 95 / 211 / 116
- Mask reasons v1: `{"complex_overlap_transition":47,"continuity_unknown":28,"mixed_unresolved_transition":20}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":101,"complex_overlap_transition":7,"continuity_unknown":44,"initial_start":1,"mixed_unresolved_transition":58}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":101,"complex_overlap_transition":-40,"continuity_unknown":16,"initial_start":1,"mixed_unresolved_transition":38}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":74,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 171 / 1832944 samples / 0.031822 h

### `ami_ES2009d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.531810 | 0.055675 | 0.127321 | 0.403553 |
| v2 | 0.425721 | 0.161764 | 0.043383 | 0.377844 |

- Speech segments v1/v2: 694 / 1176
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 564 / 0 / 130 / 0
- Removed internal pause: 5773120 samples (0.100228 h)
- Removed outer padding: 3887680 samples (0.067494 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 640 | 3040 | 6832 | 10032 | 25195 | 33920 |
| End samples | -94048 | -4096 | -1504 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1152 | 3680 | 7872 | 15505 | 33704 | 94048 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 4 | 8 | 0 |
| Gap | 0 | 37 | 0 |
| Overlap | 5 | 3 | 50 |

- Topology episodes v1/v2/matched/added/removed: 285 / 515 / 109 / 406 / 176
- Unchanged/timing-only/topology-changing: 34 / 59 / 598
- Overlap takeover/return changes: -25 / -72
- Short-backchannel change: 30
- Handoffs v1/v2/matched/added/removed: 175 / 212 / 88 / 124 / 87

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 57 | 0.200000 |
| 100 ms | 71 | 0.249123 |
| 200 ms | 90 | 0.315789 |
| 500 ms | 109 | 0.382456 |

- Masked transitions v1/v2/change: 171 / 326 / 155
- Mask reasons v1: `{"complex_overlap_transition":86,"continuity_unknown":52,"mixed_unresolved_transition":33}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":164,"complex_overlap_transition":8,"continuity_unknown":54,"initial_start":1,"mixed_unresolved_transition":99}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":164,"complex_overlap_transition":-78,"continuity_unknown":2,"initial_start":1,"mixed_unresolved_transition":66}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":130,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 274 / 3992976 samples / 0.069322 h

### `ami_ES2010a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.139552 | 0.039417 | 0.020326 | 0.119084 |
| v2 | 0.102637 | 0.076332 | 0.007057 | 0.094724 |

- Speech segments v1/v2: 136 / 297
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 116 / 0 / 20 / 0
- Removed internal pause: 1816000 samples (0.031528 h)
- Removed outer padding: 838592 samples (0.014559 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 960 | 2600 | 5040 | 8880 | 20008 | 29600 |
| End samples | -40624 | -6108 | -2208 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1440 | 3896 | 10822 | 17870 | 27918 | 40624 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 4 | 0 |
| Gap | 0 | 9 | 0 |
| Overlap | 0 | 1 | 7 |

- Topology episodes v1/v2/matched/added/removed: 66 / 153 / 23 / 130 / 43
- Unchanged/timing-only/topology-changing: 9 / 9 / 178
- Overlap takeover/return changes: -9 / -18
- Short-backchannel change: 8
- Handoffs v1/v2/matched/added/removed: 45 / 54 / 24 / 30 / 21

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 14 | 0.212121 |
| 100 ms | 17 | 0.257576 |
| 200 ms | 21 | 0.318182 |
| 500 ms | 23 | 0.348485 |

- Masked transitions v1/v2/change: 38 / 78 / 40
- Mask reasons v1: `{"complex_overlap_transition":10,"continuity_unknown":20,"mixed_unresolved_transition":8}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":30,"complex_overlap_transition":1,"continuity_unknown":26,"mixed_unresolved_transition":21}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":30,"complex_overlap_transition":-9,"continuity_unknown":6,"mixed_unresolved_transition":13}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":20,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 45 / 547536 samples / 0.009506 h

### `ami_ES2010b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.399571 | 0.087090 | 0.047412 | 0.351093 |
| v2 | 0.314204 | 0.172456 | 0.020399 | 0.290283 |

- Speech segments v1/v2: 392 / 849
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 356 / 0 / 36 / 0
- Removed internal pause: 4537008 samples (0.078768 h)
- Removed outer padding: 1602048 samples (0.027813 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2080 | 3520 | 5680 | 13952 | 26240 |
| End samples | -30944 | -3712 | -944 | 0 | 0 | 0 | 0 | 171 |
| Absolute samples | 0 | 0 | 544 | 2720 | 5920 | 9051 | 20184 | 30944 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 8 | 0 |
| Gap | 0 | 25 | 0 |
| Overlap | 2 | 13 | 26 |

- Topology episodes v1/v2/matched/added/removed: 179 / 460 / 79 / 381 / 100
- Unchanged/timing-only/topology-changing: 22 / 34 / 504
- Overlap takeover/return changes: -23 / -40
- Short-backchannel change: 14
- Handoffs v1/v2/matched/added/removed: 134 / 148 / 87 / 61 / 47

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 40 | 0.223464 |
| 100 ms | 50 | 0.279330 |
| 200 ms | 64 | 0.357542 |
| 500 ms | 79 | 0.441341 |

- Masked transitions v1/v2/change: 107 / 204 / 97
- Mask reasons v1: `{"complex_overlap_transition":20,"continuity_unknown":50,"mixed_unresolved_transition":37}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":55,"complex_overlap_transition":4,"continuity_unknown":62,"mixed_unresolved_transition":83}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":55,"complex_overlap_transition":-16,"continuity_unknown":12,"mixed_unresolved_transition":46}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":1,"unpaired_v1_speech_segments":36,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 60 / 610800 samples / 0.010604 h

### `ami_ES2010c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.437906 | 0.072582 | 0.068646 | 0.368551 |
| v2 | 0.360543 | 0.149946 | 0.028614 | 0.328130 |

- Speech segments v1/v2: 430 / 927
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 404 / 0 / 26 / 0
- Removed internal pause: 4629936 samples (0.080381 h)
- Removed outer padding: 2068928 samples (0.035919 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2880 | 5232 | 7976 | 22633 | 28800 |
| End samples | -30672 | -4516 | -1840 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1272 | 3680 | 6564 | 9327 | 19759 | 30672 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 6 | 0 |
| Gap | 0 | 39 | 0 |
| Overlap | 4 | 9 | 29 |

- Topology episodes v1/v2/matched/added/removed: 209 / 492 / 92 / 400 / 117
- Unchanged/timing-only/topology-changing: 34 / 39 / 536
- Overlap takeover/return changes: -20 / -53
- Short-backchannel change: 20
- Handoffs v1/v2/matched/added/removed: 119 / 167 / 84 / 83 / 35

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 56 | 0.267943 |
| 100 ms | 69 | 0.330144 |
| 200 ms | 80 | 0.382775 |
| 500 ms | 92 | 0.440191 |

- Masked transitions v1/v2/change: 108 / 215 / 107
- Mask reasons v1: `{"complex_overlap_transition":41,"continuity_unknown":43,"mixed_unresolved_transition":24}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":55,"complex_overlap_transition":10,"continuity_unknown":53,"mixed_unresolved_transition":97}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":55,"complex_overlap_transition":-31,"continuity_unknown":10,"mixed_unresolved_transition":73}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":26,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 65 / 651344 samples / 0.011308 h

### `ami_ES2011a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.241812 | 0.067590 | 0.047869 | 0.193711 |
| v2 | 0.170458 | 0.138943 | 0.016882 | 0.151712 |

- Speech segments v1/v2: 261 / 538
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 215 / 0 / 46 / 0
- Removed internal pause: 3531200 samples (0.061306 h)
- Removed outer padding: 2081520 samples (0.036138 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2080 | 5856 | 10768 | 24512 | 449536 |
| End samples | -57280 | -5608 | -2064 | -528 | 0 | 0 | 0 | 80 |
| Absolute samples | 0 | 0 | 1120 | 3832 | 10881 | 19348 | 41206 | 449536 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 3 | 0 |
| Gap | 0 | 13 | 0 |
| Overlap | 1 | 2 | 12 |

- Topology episodes v1/v2/matched/added/removed: 89 / 232 / 33 / 199 / 56
- Unchanged/timing-only/topology-changing: 10 / 17 / 261
- Overlap takeover/return changes: -11 / -19
- Short-backchannel change: 10
- Handoffs v1/v2/matched/added/removed: 55 / 73 / 33 / 40 / 22

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 16 | 0.179775 |
| 100 ms | 22 | 0.247191 |
| 200 ms | 26 | 0.292135 |
| 500 ms | 33 | 0.370787 |

- Masked transitions v1/v2/change: 90 / 168 / 78
- Mask reasons v1: `{"complex_overlap_transition":35,"continuity_unknown":43,"mixed_unresolved_transition":12}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":78,"complex_overlap_transition":7,"continuity_unknown":45,"mixed_unresolved_transition":38}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":78,"complex_overlap_transition":-28,"continuity_unknown":2,"mixed_unresolved_transition":26}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":46,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 120 / 1783376 samples / 0.030961 h

### `ami_ES2011b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.365124 | 0.074117 | 0.062330 | 0.302333 |
| v2 | 0.296551 | 0.142690 | 0.031147 | 0.262715 |

- Speech segments v1/v2: 354 / 764
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 331 / 0 / 23 / 0
- Removed internal pause: 4259200 samples (0.073944 h)
- Removed outer padding: 1650016 samples (0.028646 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2320 | 4000 | 5280 | 14496 | 33280 |
| End samples | -114400 | -4008 | -1568 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 744 | 3200 | 5579 | 9092 | 25095 | 114400 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 1 | 0 |
| Gap | 0 | 21 | 0 |
| Overlap | 1 | 3 | 33 |

- Topology episodes v1/v2/matched/added/removed: 150 / 384 / 60 / 324 / 90
- Unchanged/timing-only/topology-changing: 22 / 33 / 419
- Overlap takeover/return changes: -10 / -24
- Short-backchannel change: 15
- Handoffs v1/v2/matched/added/removed: 82 / 117 / 48 / 69 / 34

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 34 | 0.226667 |
| 100 ms | 41 | 0.273333 |
| 200 ms | 45 | 0.300000 |
| 500 ms | 60 | 0.400000 |

- Masked transitions v1/v2/change: 104 / 182 / 78
- Mask reasons v1: `{"complex_overlap_transition":43,"continuity_unknown":46,"mixed_unresolved_transition":15}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":53,"complex_overlap_transition":13,"continuity_unknown":45,"mixed_unresolved_transition":71}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":53,"complex_overlap_transition":-30,"continuity_unknown":-1,"mixed_unresolved_transition":56}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":23,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 82 / 1116672 samples / 0.019387 h

### `ami_ES2011d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.422304 | 0.128342 | 0.096373 | 0.325361 |
| v2 | 0.311184 | 0.239461 | 0.027965 | 0.280526 |

- Speech segments v1/v2: 563 / 930
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 477 / 0 / 86 / 0
- Removed internal pause: 4917440 samples (0.085372 h)
- Removed outer padding: 5039808 samples (0.087497 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 2080 | 3200 | 4160 | 5824 | 7360 | 21388 | 36320 |
| End samples | -124208 | -7984 | -5152 | -2928 | -660 | 0 | 0 | 0 |
| Absolute samples | 0 | 2400 | 3800 | 6000 | 9206 | 13133 | 37494 | 124208 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 11 | 0 |
| Gap | 0 | 21 | 0 |
| Overlap | 4 | 26 | 30 |

- Topology episodes v1/v2/matched/added/removed: 253 / 427 / 98 / 329 / 155
- Unchanged/timing-only/topology-changing: 7 / 50 / 525
- Overlap takeover/return changes: -44 / -31
- Short-backchannel change: 5
- Handoffs v1/v2/matched/added/removed: 201 / 197 / 123 / 74 / 78

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 22 | 0.086957 |
| 100 ms | 33 | 0.130435 |
| 200 ms | 61 | 0.241107 |
| 500 ms | 98 | 0.387352 |

- Masked transitions v1/v2/change: 147 / 296 / 149
- Mask reasons v1: `{"complex_overlap_transition":58,"continuity_unknown":61,"mixed_unresolved_transition":28}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":117,"complex_overlap_transition":4,"continuity_unknown":103,"mixed_unresolved_transition":72}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":117,"complex_overlap_transition":-54,"continuity_unknown":42,"mixed_unresolved_transition":44}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":86,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 196 / 2308784 samples / 0.040083 h

### `ami_ES2012a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.275108 | 0.031707 | 0.020914 | 0.253967 |
| v2 | 0.196097 | 0.110718 | 0.004197 | 0.190571 |

- Speech segments v1/v2: 154 / 510
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 134 / 0 / 20 / 0
- Removed internal pause: 4191392 samples (0.072767 h)
- Removed outer padding: 1150640 samples (0.019976 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1120 | 3920 | 11104 | 15744 | 23048 | 103200 |
| End samples | -54896 | -5424 | -2728 | -560 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1920 | 4592 | 10484 | 16784 | 31107 | 103200 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 2 | 0 |
| Gap | 0 | 7 | 0 |
| Overlap | 1 | 3 | 9 |

- Topology episodes v1/v2/matched/added/removed: 68 / 304 / 23 / 281 / 45
- Unchanged/timing-only/topology-changing: 8 / 9 / 332
- Overlap takeover/return changes: -8 / -27
- Short-backchannel change: 4
- Handoffs v1/v2/matched/added/removed: 29 / 40 / 19 / 21 / 10

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 11 | 0.161765 |
| 100 ms | 12 | 0.176471 |
| 200 ms | 17 | 0.250000 |
| 500 ms | 23 | 0.338235 |

- Masked transitions v1/v2/change: 53 / 139 / 86
- Mask reasons v1: `{"complex_overlap_transition":12,"continuity_unknown":33,"mixed_unresolved_transition":8}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":45,"complex_overlap_transition":3,"continuity_unknown":58,"mixed_unresolved_transition":33}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":45,"complex_overlap_transition":-9,"continuity_unknown":25,"mixed_unresolved_transition":25}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":20,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 61 / 691648 samples / 0.012008 h

### `ami_ES2012b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.516020 | 0.106110 | 0.056714 | 0.458702 |
| v2 | 0.384586 | 0.237543 | 0.018331 | 0.363729 |

- Speech segments v1/v2: 396 / 990
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 342 / 0 / 54 / 0
- Removed internal pause: 6605280 samples (0.114675 h)
- Removed outer padding: 2746736 samples (0.047686 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1760 | 4280 | 8128 | 10536 | 16574 | 18080 |
| End samples | -64528 | -6256 | -3400 | -1036 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 2560 | 5120 | 9280 | 13503 | 26682 | 64528 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 5 | 0 |
| Gap | 0 | 32 | 0 |
| Overlap | 1 | 5 | 23 |

- Topology episodes v1/v2/matched/added/removed: 190 / 572 / 70 / 502 / 120
- Unchanged/timing-only/topology-changing: 21 / 38 / 633
- Overlap takeover/return changes: -27 / -31
- Short-backchannel change: 8
- Handoffs v1/v2/matched/added/removed: 85 / 89 / 41 / 48 / 44

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 34 | 0.178947 |
| 100 ms | 46 | 0.242105 |
| 200 ms | 55 | 0.289474 |
| 500 ms | 70 | 0.368421 |

- Masked transitions v1/v2/change: 120 / 265 / 145
- Mask reasons v1: `{"complex_overlap_transition":24,"continuity_unknown":75,"mixed_unresolved_transition":21}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":106,"complex_overlap_transition":8,"continuity_unknown":91,"initial_start":1,"mixed_unresolved_transition":59}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":106,"complex_overlap_transition":-16,"continuity_unknown":16,"initial_start":1,"mixed_unresolved_transition":38}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":54,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 133 / 1463520 samples / 0.025408 h

### `ami_ES2012c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.524626 | 0.088900 | 0.089979 | 0.433944 |
| v2 | 0.407176 | 0.206350 | 0.034520 | 0.368452 |

- Speech segments v1/v2: 478 / 1105
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 425 / 0 / 53 / 0
- Removed internal pause: 6637600 samples (0.115236 h)
- Removed outer padding: 3074032 samples (0.053369 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1920 | 3840 | 6496 | 8928 | 15692 | 41120 |
| End samples | -41648 | -5600 | -3456 | -1376 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 480 | 2584 | 4800 | 8259 | 11741 | 21827 | 41648 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 2 | 0 |
| Gap | 0 | 24 | 0 |
| Overlap | 2 | 9 | 38 |

- Topology episodes v1/v2/matched/added/removed: 206 / 568 / 75 / 493 / 131
- Unchanged/timing-only/topology-changing: 18 / 44 / 637
- Overlap takeover/return changes: -26 / -22
- Short-backchannel change: 19
- Handoffs v1/v2/matched/added/removed: 95 / 133 / 45 / 88 / 50

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 26 | 0.126214 |
| 100 ms | 35 | 0.169903 |
| 200 ms | 50 | 0.242718 |
| 500 ms | 75 | 0.364078 |

- Masked transitions v1/v2/change: 126 / 283 / 157
- Mask reasons v1: `{"complex_overlap_transition":63,"continuity_unknown":38,"mixed_unresolved_transition":25}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":123,"complex_overlap_transition":15,"continuity_unknown":57,"mixed_unresolved_transition":88}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":123,"complex_overlap_transition":-48,"continuity_unknown":19,"mixed_unresolved_transition":63}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":53,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 150 / 1630624 samples / 0.028309 h

### `ami_ES2012d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.224015 | 0.039778 | 0.049577 | 0.174057 |
| v2 | 0.171303 | 0.092490 | 0.019562 | 0.149685 |

- Speech segments v1/v2: 268 / 512
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 231 / 0 / 37 / 0
- Removed internal pause: 2834240 samples (0.049206 h)
- Removed outer padding: 1853568 samples (0.032180 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1280 | 3760 | 7040 | 9520 | 27472 | 35200 |
| End samples | -85696 | -6128 | -3424 | -1344 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 432 | 2552 | 5068 | 8860 | 11726 | 32857 | 85696 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 0 | 0 |
| Gap | 0 | 10 | 0 |
| Overlap | 0 | 5 | 14 |

- Topology episodes v1/v2/matched/added/removed: 82 / 236 / 31 / 205 / 51
- Unchanged/timing-only/topology-changing: 6 / 20 / 261
- Overlap takeover/return changes: -5 / 1
- Short-backchannel change: 6
- Handoffs v1/v2/matched/added/removed: 46 / 66 / 22 / 44 / 24

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 14 | 0.170732 |
| 100 ms | 16 | 0.195122 |
| 200 ms | 21 | 0.256098 |
| 500 ms | 31 | 0.378049 |

- Masked transitions v1/v2/change: 84 / 147 / 63
- Mask reasons v1: `{"complex_overlap_transition":38,"continuity_unknown":33,"mixed_unresolved_transition":13}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":53,"complex_overlap_transition":12,"continuity_unknown":40,"initial_start":1,"mixed_unresolved_transition":41}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":53,"complex_overlap_transition":-26,"continuity_unknown":7,"initial_start":1,"mixed_unresolved_transition":28}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":37,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 76 / 905408 samples / 0.015719 h

### `ami_ES2013a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.162536 | 0.066679 | 0.022579 | 0.139815 |
| v2 | 0.118731 | 0.110484 | 0.006426 | 0.111216 |

- Speech segments v1/v2: 156 / 320
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 135 / 0 / 21 / 0
- Removed internal pause: 1971520 samples (0.034228 h)
- Removed outer padding: 1050672 samples (0.018241 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1120 | 3600 | 7840 | 11504 | 18064 | 20800 |
| End samples | -76832 | -4256 | -1760 | -448 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1576 | 4096 | 8857 | 14260 | 44019 | 76832 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 2 | 0 |
| Gap | 0 | 17 | 0 |
| Overlap | 1 | 0 | 4 |

- Topology episodes v1/v2/matched/added/removed: 59 / 168 / 26 / 142 / 33
- Unchanged/timing-only/topology-changing: 11 / 12 / 178
- Overlap takeover/return changes: 1 / -20
- Short-backchannel change: 0
- Handoffs v1/v2/matched/added/removed: 41 / 46 / 28 / 18 / 13

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 15 | 0.254237 |
| 100 ms | 18 | 0.305085 |
| 200 ms | 24 | 0.406780 |
| 500 ms | 26 | 0.440678 |

- Masked transitions v1/v2/change: 65 / 93 / 28
- Mask reasons v1: `{"complex_overlap_transition":11,"continuity_unknown":44,"mixed_unresolved_transition":10}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":34,"continuity_unknown":39,"mixed_unresolved_transition":20}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":34,"complex_overlap_transition":-11,"continuity_unknown":-5,"mixed_unresolved_transition":10}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":21,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 53 / 745856 samples / 0.012949 h

### `ami_ES2013b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.495438 | 0.095848 | 0.052304 | 0.442686 |
| v2 | 0.380780 | 0.210506 | 0.018907 | 0.358704 |

- Speech segments v1/v2: 395 / 1007
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 381 / 0 / 14 / 0
- Removed internal pause: 6165600 samples (0.107042 h)
- Removed outer padding: 2571792 samples (0.044649 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1120 | 3040 | 5920 | 9600 | 21248 | 50400 |
| End samples | -79456 | -4720 | -2368 | -544 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 456 | 1744 | 3840 | 8156 | 12380 | 29671 | 79456 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 5 | 0 |
| Gap | 0 | 36 | 0 |
| Overlap | 2 | 4 | 29 |

- Topology episodes v1/v2/matched/added/removed: 189 / 588 / 84 / 504 / 105
- Unchanged/timing-only/topology-changing: 18 / 55 / 620
- Overlap takeover/return changes: -9 / -41
- Short-backchannel change: 14
- Handoffs v1/v2/matched/added/removed: 102 / 131 / 64 / 67 / 38

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 35 | 0.185185 |
| 100 ms | 46 | 0.243386 |
| 200 ms | 66 | 0.349206 |
| 500 ms | 84 | 0.444444 |

- Masked transitions v1/v2/change: 131 / 256 / 125
- Mask reasons v1: `{"complex_overlap_transition":27,"continuity_unknown":85,"mixed_unresolved_transition":19}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":84,"complex_overlap_transition":2,"continuity_unknown":100,"mixed_unresolved_transition":70}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":84,"complex_overlap_transition":-25,"continuity_unknown":15,"mixed_unresolved_transition":51}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":14,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 92 / 854464 samples / 0.014834 h

### `ami_ES2013c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.562190 | 0.092844 | 0.075300 | 0.486404 |
| v2 | 0.429514 | 0.225520 | 0.020681 | 0.406081 |

- Speech segments v1/v2: 433 / 1063
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 372 / 0 / 61 / 0
- Removed internal pause: 7024832 samples (0.121959 h)
- Removed outer padding: 2497104 samples (0.043353 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 640 | 3360 | 7504 | 12696 | 26110 | 45920 |
| End samples | -98208 | -4708 | -1872 | -440 | 0 | 0 | 0 | 171 |
| Absolute samples | 0 | 0 | 1120 | 4148 | 7660 | 12843 | 29448 | 98208 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 7 | 0 |
| Gap | 0 | 31 | 0 |
| Overlap | 1 | 9 | 28 |

- Topology episodes v1/v2/matched/added/removed: 206 / 599 / 79 / 520 / 127
- Unchanged/timing-only/topology-changing: 23 / 39 / 664
- Overlap takeover/return changes: -21 / -45
- Short-backchannel change: 5
- Handoffs v1/v2/matched/added/removed: 116 / 122 / 62 / 60 / 54

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 41 | 0.199029 |
| 100 ms | 47 | 0.228155 |
| 200 ms | 58 | 0.281553 |
| 500 ms | 79 | 0.383495 |

- Masked transitions v1/v2/change: 129 / 292 / 163
- Mask reasons v1: `{"complex_overlap_transition":37,"continuity_unknown":71,"mixed_unresolved_transition":21}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":147,"complex_overlap_transition":4,"continuity_unknown":86,"mixed_unresolved_transition":55}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":147,"complex_overlap_transition":-33,"continuity_unknown":15,"mixed_unresolved_transition":34}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":1,"unpaired_v1_speech_segments":61,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 197 / 2653851 samples / 0.046074 h

### `ami_ES2013d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.412974 | 0.114404 | 0.048819 | 0.363675 |
| v2 | 0.318311 | 0.209067 | 0.022160 | 0.293826 |

- Speech segments v1/v2: 415 / 888
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 393 / 0 / 22 / 0
- Removed internal pause: 4724368 samples (0.082020 h)
- Removed outer padding: 2073664 samples (0.036001 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2560 | 4608 | 6240 | 16505 | 32960 |
| End samples | -39136 | -3936 | -1664 | -480 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1120 | 3116 | 6080 | 9584 | 26956 | 39136 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 4 | 0 |
| Gap | 0 | 36 | 0 |
| Overlap | 2 | 3 | 24 |

- Topology episodes v1/v2/matched/added/removed: 187 / 484 / 75 / 409 / 112
- Unchanged/timing-only/topology-changing: 18 / 48 / 530
- Overlap takeover/return changes: -13 / -35
- Short-backchannel change: 16
- Handoffs v1/v2/matched/added/removed: 123 / 137 / 74 / 63 / 49

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 42 | 0.224599 |
| 100 ms | 49 | 0.262032 |
| 200 ms | 60 | 0.320856 |
| 500 ms | 75 | 0.401070 |

- Masked transitions v1/v2/change: 138 / 242 / 104
- Mask reasons v1: `{"complex_overlap_transition":22,"continuity_unknown":93,"mixed_unresolved_transition":23}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":68,"complex_overlap_transition":9,"continuity_unknown":107,"initial_start":1,"mixed_unresolved_transition":57}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":68,"complex_overlap_transition":-13,"continuity_unknown":14,"initial_start":1,"mixed_unresolved_transition":34}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":22,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 76 / 875248 samples / 0.015195 h

### `ami_ES2014a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.250000 | 0.069170 | 0.035734 | 0.214143 |
| v2 | 0.169627 | 0.149543 | 0.006162 | 0.162271 |

- Speech segments v1/v2: 221 / 450
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 164 / 0 / 57 / 0
- Removed internal pause: 3375360 samples (0.058600 h)
- Removed outer padding: 1976240 samples (0.034310 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1040 | 4480 | 9072 | 18056 | 34676 | 42080 |
| End samples | -59408 | -10440 | -4136 | 0 | 0 | 0 | 0 | 80 |
| Absolute samples | 0 | 0 | 2016 | 7496 | 17033 | 26056 | 44676 | 59408 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 0 | 0 |
| Gap | 0 | 13 | 0 |
| Overlap | 1 | 4 | 5 |

- Topology episodes v1/v2/matched/added/removed: 84 / 228 / 24 / 204 / 60
- Unchanged/timing-only/topology-changing: 11 / 8 / 269
- Overlap takeover/return changes: -11 / -28
- Short-backchannel change: 7
- Handoffs v1/v2/matched/added/removed: 46 / 53 / 25 / 28 / 21

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 14 | 0.166667 |
| 100 ms | 15 | 0.178571 |
| 200 ms | 18 | 0.214286 |
| 500 ms | 24 | 0.285714 |

- Masked transitions v1/v2/change: 77 / 151 / 74
- Mask reasons v1: `{"complex_overlap_transition":28,"continuity_unknown":45,"mixed_unresolved_transition":4}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":64,"complex_overlap_transition":1,"continuity_unknown":61,"mixed_unresolved_transition":25}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":64,"complex_overlap_transition":-27,"continuity_unknown":16,"mixed_unresolved_transition":21}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":57,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 113 / 1515104 samples / 0.026304 h

### `ami_ES2015d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.464261 | 0.072284 | 0.173056 | 0.289481 |
| v2 | 0.368991 | 0.167554 | 0.063888 | 0.298415 |

- Speech segments v1/v2: 779 / 1261
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 706 / 0 / 73 / 0
- Removed internal pause: 5522880 samples (0.095883 h)
- Removed outer padding: 8068480 samples (0.140078 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1440 | 3200 | 4640 | 6720 | 8920 | 27768 | 52000 |
| End samples | -268848 | -7840 | -4992 | -3136 | -664 | 0 | 0 | 0 |
| Absolute samples | 0 | 2112 | 3928 | 6240 | 9648 | 14307 | 41044 | 268848 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 9 | 0 |
| Gap | 0 | 25 | 0 |
| Overlap | 5 | 14 | 42 |

- Topology episodes v1/v2/matched/added/removed: 233 / 475 / 98 / 377 / 135
- Unchanged/timing-only/topology-changing: 16 / 54 / 540
- Overlap takeover/return changes: -12 / -29
- Short-backchannel change: 24
- Handoffs v1/v2/matched/added/removed: 150 / 262 / 89 / 173 / 61

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 23 | 0.098712 |
| 100 ms | 30 | 0.128755 |
| 200 ms | 51 | 0.218884 |
| 500 ms | 98 | 0.420601 |

- Masked transitions v1/v2/change: 204 / 340 / 136
- Mask reasons v1: `{"complex_overlap_transition":112,"continuity_unknown":40,"mixed_unresolved_transition":52}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":134,"complex_overlap_transition":26,"continuity_unknown":51,"mixed_unresolved_transition":129}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":134,"complex_overlap_transition":-86,"continuity_unknown":11,"mixed_unresolved_transition":77}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":73,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 216 / 2437888 samples / 0.042324 h

### `ami_ES2016a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.296694 | 0.087803 | 0.035689 | 0.260559 |
| v2 | 0.205491 | 0.179006 | 0.011416 | 0.191937 |

- Speech segments v1/v2: 266 / 669
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 238 / 0 / 28 / 0
- Removed internal pause: 4723680 samples (0.082008 h)
- Removed outer padding: 1651248 samples (0.028667 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1280 | 3160 | 6560 | 9904 | 18027 | 39040 |
| End samples | -48496 | -5996 | -3024 | -936 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 2080 | 4640 | 8584 | 11768 | 19216 | 48496 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 2 | 0 |
| Gap | 0 | 15 | 0 |
| Overlap | 1 | 6 | 15 |

- Topology episodes v1/v2/matched/added/removed: 103 / 353 / 43 / 310 / 60
- Unchanged/timing-only/topology-changing: 9 / 25 / 379
- Overlap takeover/return changes: -6 / -28
- Short-backchannel change: 12
- Handoffs v1/v2/matched/added/removed: 54 / 75 / 38 / 37 / 16

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 20 | 0.194175 |
| 100 ms | 24 | 0.233010 |
| 200 ms | 34 | 0.330097 |
| 500 ms | 43 | 0.417476 |

- Masked transitions v1/v2/change: 107 / 198 / 91
- Mask reasons v1: `{"complex_overlap_transition":15,"continuity_unknown":70,"mixed_unresolved_transition":22}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":57,"complex_overlap_transition":1,"continuity_unknown":90,"mixed_unresolved_transition":50}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":57,"complex_overlap_transition":-14,"continuity_unknown":20,"mixed_unresolved_transition":28}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":28,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 81 / 915200 samples / 0.015889 h

### `ami_IS1007a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.205751 | 0.062582 | 0.044324 | 0.160811 |
| v2 | 0.153325 | 0.115009 | 0.019194 | 0.132336 |

- Speech segments v1/v2: 273 / 471
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 242 / 0 / 31 / 0
- Removed internal pause: 2441920 samples (0.042394 h)
- Removed outer padding: 1662368 samples (0.028861 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 2200 | 8112 | 12952 | 24924 | 42400 |
| End samples | -74608 | -4064 | -432 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 2624 | 9987 | 16781 | 41085 | 74608 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 1 | 0 |
| Gap | 0 | 22 | 0 |
| Overlap | 0 | 3 | 11 |

- Topology episodes v1/v2/matched/added/removed: 127 / 198 / 42 / 156 / 85
- Unchanged/timing-only/topology-changing: 30 / 8 / 245
- Overlap takeover/return changes: -19 / -28
- Short-backchannel change: 5
- Handoffs v1/v2/matched/added/removed: 86 / 85 / 42 / 43 / 44

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 32 | 0.251969 |
| 100 ms | 34 | 0.267717 |
| 200 ms | 38 | 0.299213 |
| 500 ms | 42 | 0.330709 |

- Masked transitions v1/v2/change: 70 / 140 / 70
- Mask reasons v1: `{"complex_overlap_transition":23,"continuity_unknown":29,"mixed_unresolved_transition":18}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":54,"complex_overlap_transition":4,"continuity_unknown":34,"mixed_unresolved_transition":48}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":54,"complex_overlap_transition":-19,"continuity_unknown":5,"mixed_unresolved_transition":30}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":31,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 86 / 1070400 samples / 0.018583 h

### `ami_IS1007b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.302402 | 0.059450 | 0.077539 | 0.224261 |
| v2 | 0.237596 | 0.124256 | 0.028994 | 0.205954 |

- Speech segments v1/v2: 372 / 677
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 309 / 0 / 63 / 0
- Removed internal pause: 3376320 samples (0.058617 h)
- Removed outer padding: 2546736 samples (0.044214 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 2560 | 5280 | 7776 | 14054 | 31040 |
| End samples | -108864 | -5584 | -768 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 3800 | 11097 | 21424 | 47661 | 108864 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 1 | 0 |
| Gap | 0 | 26 | 0 |
| Overlap | 1 | 6 | 28 |

- Topology episodes v1/v2/matched/added/removed: 175 / 318 / 70 / 248 / 105
- Unchanged/timing-only/topology-changing: 36 / 26 / 361
- Overlap takeover/return changes: -20 / -44
- Short-backchannel change: 7
- Handoffs v1/v2/matched/added/removed: 97 / 110 / 53 / 57 / 44

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 46 | 0.262857 |
| 100 ms | 50 | 0.285714 |
| 200 ms | 58 | 0.331429 |
| 500 ms | 70 | 0.400000 |

- Masked transitions v1/v2/change: 88 / 194 / 106
- Mask reasons v1: `{"complex_overlap_transition":41,"continuity_unknown":30,"mixed_unresolved_transition":17}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":106,"complex_overlap_transition":6,"continuity_unknown":31,"mixed_unresolved_transition":51}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":106,"complex_overlap_transition":-35,"continuity_unknown":1,"mixed_unresolved_transition":34}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":63,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 163 / 2110368 samples / 0.036638 h

### `ami_IS1007c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.501134 | 0.085857 | 0.093723 | 0.406901 |
| v2 | 0.405527 | 0.181464 | 0.043239 | 0.358525 |

- Speech segments v1/v2: 465 / 1063
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 432 / 0 / 33 / 0
- Removed internal pause: 6003200 samples (0.104222 h)
- Removed outer padding: 2318720 samples (0.040256 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 1440 | 4144 | 6400 | 19062 | 39200 |
| End samples | -99568 | -3424 | -672 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 2576 | 5931 | 10708 | 37153 | 99568 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 2 | 0 |
| Gap | 0 | 34 | 0 |
| Overlap | 0 | 5 | 38 |

- Topology episodes v1/v2/matched/added/removed: 230 / 551 / 82 / 469 / 148
- Unchanged/timing-only/topology-changing: 43 / 32 / 624
- Overlap takeover/return changes: -31 / -50
- Short-backchannel change: 29
- Handoffs v1/v2/matched/added/removed: 103 / 172 / 56 / 116 / 47

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 53 | 0.230435 |
| 100 ms | 65 | 0.282609 |
| 200 ms | 71 | 0.308696 |
| 500 ms | 82 | 0.356522 |

- Masked transitions v1/v2/change: 110 / 250 / 140
- Mask reasons v1: `{"complex_overlap_transition":44,"continuity_unknown":42,"mixed_unresolved_transition":24}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":96,"complex_overlap_transition":10,"continuity_unknown":49,"mixed_unresolved_transition":95}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":96,"complex_overlap_transition":-34,"continuity_unknown":7,"mixed_unresolved_transition":71}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":33,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 122 / 1380144 samples / 0.023961 h

### `ami_IS1007d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.501865 | 0.061762 | 0.144800 | 0.355776 |
| v2 | 0.398464 | 0.165162 | 0.045151 | 0.349372 |

- Speech segments v1/v2: 580 / 1062
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 482 / 0 / 98 / 0
- Removed internal pause: 6237648 samples (0.108292 h)
- Removed outer padding: 4299200 samples (0.074639 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 2080 | 4800 | 12592 | 54649 | 149760 |
| End samples | -197760 | -3492 | -448 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 2560 | 8606 | 21123 | 77123 | 197760 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 3 | 0 |
| Gap | 0 | 43 | 0 |
| Overlap | 1 | 7 | 32 |

- Topology episodes v1/v2/matched/added/removed: 249 / 489 / 98 / 391 / 151
- Unchanged/timing-only/topology-changing: 51 / 36 / 553
- Overlap takeover/return changes: -22 / -68
- Short-backchannel change: 25
- Handoffs v1/v2/matched/added/removed: 142 / 203 / 89 / 114 / 53

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 67 | 0.269076 |
| 100 ms | 73 | 0.293173 |
| 200 ms | 87 | 0.349398 |
| 500 ms | 98 | 0.393574 |

- Masked transitions v1/v2/change: 140 / 291 / 151
- Mask reasons v1: `{"complex_overlap_transition":61,"continuity_unknown":29,"mixed_unresolved_transition":50}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":172,"complex_overlap_transition":8,"continuity_unknown":31,"initial_start":1,"mixed_unresolved_transition":79}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":172,"complex_overlap_transition":-53,"continuity_unknown":2,"initial_start":1,"mixed_unresolved_transition":29}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":98,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 251 / 3968288 samples / 0.068894 h

### `ami_IS1008a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.221423 | 0.040753 | 0.015600 | 0.205747 |
| v2 | 0.181939 | 0.080237 | 0.005978 | 0.175030 |

- Speech segments v1/v2: 144 / 362
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 137 / 0 / 7 / 0
- Removed internal pause: 1964448 samples (0.034105 h)
- Removed outer padding: 762816 samples (0.013243 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2080 | 5984 | 18112 | 27628 | 51360 |
| End samples | -41312 | -2704 | -640 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 480 | 2544 | 7320 | 13400 | 29408 | 51360 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 2 | 0 |
| Gap | 0 | 14 | 0 |
| Overlap | 2 | 0 | 16 |

- Topology episodes v1/v2/matched/added/removed: 99 / 214 / 40 / 174 / 59
- Unchanged/timing-only/topology-changing: 14 / 22 / 237
- Overlap takeover/return changes: -10 / -30
- Short-backchannel change: 10
- Handoffs v1/v2/matched/added/removed: 65 / 73 / 42 / 31 / 23

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 22 | 0.222222 |
| 100 ms | 27 | 0.272727 |
| 200 ms | 33 | 0.333333 |
| 500 ms | 40 | 0.404040 |

- Masked transitions v1/v2/change: 27 / 88 / 61
- Mask reasons v1: `{"complex_overlap_transition":3,"continuity_unknown":21,"mixed_unresolved_transition":3}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":31,"continuity_unknown":26,"mixed_unresolved_transition":31}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":31,"complex_overlap_transition":-3,"continuity_unknown":5,"mixed_unresolved_transition":28}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":7,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 32 / 316592 samples / 0.005496 h

### `ami_IS1009a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.174614 | 0.058395 | 0.033903 | 0.140374 |
| v2 | 0.146678 | 0.086331 | 0.020354 | 0.124589 |

- Speech segments v1/v2: 220 / 363
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 193 / 0 / 27 / 0
- Removed internal pause: 1428176 samples (0.024795 h)
- Removed outer padding: 760832 samples (0.013209 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 1760 | 4128 | 5984 | 18739 | 36960 |
| End samples | -39744 | -1552 | -416 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 0 | 1760 | 4952 | 8704 | 28614 | 39744 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 0 | 0 |
| Gap | 0 | 10 | 0 |
| Overlap | 0 | 1 | 23 |

- Topology episodes v1/v2/matched/added/removed: 99 / 165 / 37 / 128 / 62
- Unchanged/timing-only/topology-changing: 22 / 14 / 191
- Overlap takeover/return changes: -10 / -25
- Short-backchannel change: 11
- Handoffs v1/v2/matched/added/removed: 64 / 73 / 35 / 38 / 29

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 24 | 0.242424 |
| 100 ms | 28 | 0.282828 |
| 200 ms | 32 | 0.323232 |
| 500 ms | 37 | 0.373737 |

- Masked transitions v1/v2/change: 55 / 98 / 43
- Mask reasons v1: `{"complex_overlap_transition":16,"continuity_unknown":25,"mixed_unresolved_transition":14}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":39,"complex_overlap_transition":5,"continuity_unknown":17,"mixed_unresolved_transition":37}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":39,"complex_overlap_transition":-11,"continuity_unknown":-8,"mixed_unresolved_transition":23}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":27,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 70 / 613344 samples / 0.010648 h

### `ami_TS3003b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.509934 | 0.104039 | 0.041863 | 0.467258 |
| v2 | 0.369193 | 0.244781 | 0.012265 | 0.353350 |

- Speech segments v1/v2: 425 / 1077
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 390 / 0 / 35 / 0
- Removed internal pause: 6890928 samples (0.119634 h)
- Removed outer padding: 2658128 samples (0.046148 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 1760 | 2560 | 3320 | 3840 | 4800 | 11305 | 19040 |
| End samples | -27392 | -5376 | -4128 | -2900 | -1418 | 0 | 0 | 0 |
| Absolute samples | 0 | 2080 | 3160 | 4464 | 5888 | 6951 | 11401 | 27392 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 4 | 0 |
| Gap | 0 | 50 | 0 |
| Overlap | 2 | 12 | 14 |

- Topology episodes v1/v2/matched/added/removed: 208 / 674 / 83 / 591 / 125
- Unchanged/timing-only/topology-changing: 7 / 58 / 734
- Overlap takeover/return changes: -20 / -25
- Short-backchannel change: -4
- Handoffs v1/v2/matched/added/removed: 108 / 97 / 59 / 38 / 49

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 14 | 0.067308 |
| 100 ms | 27 | 0.129808 |
| 200 ms | 61 | 0.293269 |
| 500 ms | 83 | 0.399038 |

- Masked transitions v1/v2/change: 124 / 254 / 130
- Mask reasons v1: `{"complex_overlap_transition":33,"continuity_unknown":66,"mixed_unresolved_transition":25}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":37,"complex_overlap_transition":6,"continuity_unknown":124,"mixed_unresolved_transition":87}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":37,"complex_overlap_transition":-27,"continuity_unknown":58,"mixed_unresolved_transition":62}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":35,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 51 / 637584 samples / 0.011069 h

### `ami_TS3004a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.263960 | 0.109741 | 0.051405 | 0.211681 |
| v2 | 0.199849 | 0.173852 | 0.022584 | 0.174910 |

- Speech segments v1/v2: 403 / 630
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 345 / 0 / 58 / 0
- Removed internal pause: 2830944 samples (0.049148 h)
- Removed outer padding: 1897984 samples (0.032951 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 640 | 2240 | 4480 | 6528 | 27475 | 47840 |
| End samples | -69024 | -3360 | -1040 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 944 | 2776 | 5792 | 12244 | 29480 | 69024 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 4 | 0 |
| Gap | 0 | 29 | 0 |
| Overlap | 1 | 5 | 18 |

- Topology episodes v1/v2/matched/added/removed: 156 / 280 / 65 / 215 / 91
- Unchanged/timing-only/topology-changing: 20 / 35 / 316
- Overlap takeover/return changes: -15 / -26
- Short-backchannel change: 17
- Handoffs v1/v2/matched/added/removed: 120 / 134 / 64 / 70 / 56

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 29 | 0.185897 |
| 100 ms | 42 | 0.269231 |
| 200 ms | 54 | 0.346154 |
| 500 ms | 65 | 0.416667 |

- Masked transitions v1/v2/change: 136 / 187 / 51
- Mask reasons v1: `{"complex_overlap_transition":33,"continuity_unknown":75,"mixed_unresolved_transition":28}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":53,"complex_overlap_transition":7,"continuity_unknown":71,"initial_start":1,"mixed_unresolved_transition":55}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":53,"complex_overlap_transition":-26,"continuity_unknown":-4,"initial_start":1,"mixed_unresolved_transition":27}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":58,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 98 / 1141184 samples / 0.019812 h

### `ami_TS3005b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.594043 | 0.083711 | 0.119343 | 0.472859 |
| v2 | 0.481788 | 0.195966 | 0.059891 | 0.415252 |

- Speech segments v1/v2: 748 / 1442
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 691 / 0 / 57 / 0
- Removed internal pause: 6476336 samples (0.112436 h)
- Removed outer padding: 3261504 samples (0.056623 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 2080 | 4800 | 7120 | 16064 | 28480 |
| End samples | -36016 | -4808 | -672 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 200 | 3096 | 7483 | 10717 | 20640 | 36016 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 5 | 2 | 0 |
| Gap | 0 | 57 | 0 |
| Overlap | 0 | 5 | 54 |

- Topology episodes v1/v2/matched/added/removed: 358 / 714 / 131 / 583 / 227
- Unchanged/timing-only/topology-changing: 70 / 54 / 817
- Overlap takeover/return changes: -33 / -52
- Short-backchannel change: 41
- Handoffs v1/v2/matched/added/removed: 249 / 303 / 137 / 166 / 112

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 87 | 0.243017 |
| 100 ms | 97 | 0.270950 |
| 200 ms | 106 | 0.296089 |
| 500 ms | 131 | 0.365922 |

- Masked transitions v1/v2/change: 173 / 334 / 161
- Mask reasons v1: `{"complex_overlap_transition":69,"continuity_unknown":51,"mixed_unresolved_transition":53}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":139,"complex_overlap_transition":15,"continuity_unknown":58,"mixed_unresolved_transition":122}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":139,"complex_overlap_transition":-54,"continuity_unknown":7,"mixed_unresolved_transition":69}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":57,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 185 / 1550640 samples / 0.026921 h

### `ami_TS3006a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.277568 | 0.070450 | 0.070404 | 0.206477 |
| v2 | 0.199950 | 0.148068 | 0.022988 | 0.173671 |

- Speech segments v1/v2: 438 / 735
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 394 / 0 / 44 / 0
- Removed internal pause: 3853184 samples (0.066896 h)
- Removed outer padding: 3114128 samples (0.054065 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 1760 | 3040 | 5232 | 12168 | 26211 | 38240 |
| End samples | -70112 | -5096 | -3208 | -1828 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 1120 | 2416 | 4160 | 7548 | 15060 | 34732 | 70112 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 4 | 0 |
| Gap | 0 | 25 | 0 |
| Overlap | 3 | 12 | 29 |

- Topology episodes v1/v2/matched/added/removed: 191 / 292 / 78 / 214 / 113
- Unchanged/timing-only/topology-changing: 14 / 45 / 346
- Overlap takeover/return changes: -24 / -51
- Short-backchannel change: 18
- Handoffs v1/v2/matched/added/removed: 121 / 154 / 68 / 86 / 53

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 21 | 0.109948 |
| 100 ms | 38 | 0.198953 |
| 200 ms | 59 | 0.308901 |
| 500 ms | 78 | 0.408377 |

- Masked transitions v1/v2/change: 117 / 226 / 109
- Mask reasons v1: `{"complex_overlap_transition":46,"continuity_unknown":44,"mixed_unresolved_transition":27}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":105,"complex_overlap_transition":10,"continuity_unknown":42,"initial_start":1,"mixed_unresolved_transition":68}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":105,"complex_overlap_transition":-36,"continuity_unknown":-2,"initial_start":1,"mixed_unresolved_transition":41}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":44,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 155 / 1576816 samples / 0.027375 h

### `ami_TS3007a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.333141 | 0.113899 | 0.058693 | 0.273266 |
| v2 | 0.250479 | 0.196561 | 0.014665 | 0.233993 |

- Speech segments v1/v2: 462 / 732
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 378 / 0 / 84 / 0
- Removed internal pause: 3426080 samples (0.059481 h)
- Removed outer padding: 2653264 samples (0.046064 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2400 | 5536 | 8960 | 20020 | 59840 |
| End samples | -48416 | -5528 | -3392 | -832 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 1760 | 4320 | 7520 | 13712 | 31038 | 59840 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 2 | 2 | 0 |
| Gap | 0 | 31 | 0 |
| Overlap | 2 | 2 | 22 |

- Topology episodes v1/v2/matched/added/removed: 211 / 367 / 68 / 299 / 143
- Unchanged/timing-only/topology-changing: 22 / 40 / 448
- Overlap takeover/return changes: -28 / -30
- Short-backchannel change: 14
- Handoffs v1/v2/matched/added/removed: 165 / 161 / 85 / 76 / 80

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 32 | 0.151659 |
| 100 ms | 41 | 0.194313 |
| 200 ms | 54 | 0.255924 |
| 500 ms | 68 | 0.322275 |

- Masked transitions v1/v2/change: 123 / 206 / 83
- Mask reasons v1: `{"complex_overlap_transition":25,"continuity_unknown":59,"mixed_unresolved_transition":39}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":113,"continuity_unknown":53,"mixed_unresolved_transition":40}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":113,"complex_overlap_transition":-25,"continuity_unknown":-6,"mixed_unresolved_transition":1}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":84,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 183 / 2135696 samples / 0.037078 h

### `ami_TS3008b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.573487 | 0.071224 | 0.095355 | 0.476956 |
| v2 | 0.473247 | 0.171464 | 0.051753 | 0.417524 |

- Speech segments v1/v2: 596 / 1250
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 550 / 0 / 46 / 0
- Removed internal pause: 6158352 samples (0.106916 h)
- Removed outer padding: 1982192 samples (0.034413 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 1600 | 3536 | 6168 | 17465 | 35984 |
| End samples | -33728 | -2244 | -872 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 664 | 1920 | 3937 | 6769 | 20331 | 35984 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 1 | 0 |
| Gap | 0 | 53 | 0 |
| Overlap | 2 | 3 | 68 |

- Topology episodes v1/v2/matched/added/removed: 313 / 685 / 139 / 546 / 174
- Unchanged/timing-only/topology-changing: 48 / 85 / 726
- Overlap takeover/return changes: -26 / -51
- Short-backchannel change: 21
- Handoffs v1/v2/matched/added/removed: 172 / 230 / 106 / 124 / 66

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 73 | 0.233227 |
| 100 ms | 98 | 0.313099 |
| 200 ms | 127 | 0.405751 |
| 500 ms | 139 | 0.444089 |

- Masked transitions v1/v2/change: 140 / 292 / 152
- Mask reasons v1: `{"complex_overlap_transition":47,"continuity_unknown":53,"mixed_unresolved_transition":40}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":133,"complex_overlap_transition":14,"continuity_unknown":61,"mixed_unresolved_transition":84}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":133,"complex_overlap_transition":-33,"continuity_unknown":8,"mixed_unresolved_transition":44}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":46,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 172 / 1497008 samples / 0.025990 h

### `ami_TS3009b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.607650 | 0.075698 | 0.160048 | 0.445582 |
| v2 | 0.493920 | 0.189428 | 0.076416 | 0.409753 |

- Speech segments v1/v2: 802 / 1583
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 727 / 0 / 75 / 0
- Removed internal pause: 7491072 samples (0.130053 h)
- Removed outer padding: 4310528 samples (0.074836 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 800 | 2560 | 3680 | 4320 | 19360 | 62880 |
| End samples | -73968 | -4568 | -2896 | -416 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 2040 | 3544 | 5664 | 9571 | 26791 | 73968 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 6 | 0 |
| Gap | 0 | 46 | 0 |
| Overlap | 5 | 5 | 52 |

- Topology episodes v1/v2/matched/added/removed: 304 / 673 / 123 / 550 / 181
- Unchanged/timing-only/topology-changing: 38 / 69 / 747
- Overlap takeover/return changes: -20 / -45
- Short-backchannel change: 19
- Handoffs v1/v2/matched/added/removed: 205 / 256 / 113 / 143 / 92

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 56 | 0.184211 |
| 100 ms | 65 | 0.213816 |
| 200 ms | 91 | 0.299342 |
| 500 ms | 123 | 0.404605 |

- Masked transitions v1/v2/change: 213 / 452 / 239
- Mask reasons v1: `{"complex_overlap_transition":104,"continuity_unknown":46,"mixed_unresolved_transition":63}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":219,"complex_overlap_transition":17,"continuity_unknown":62,"mixed_unresolved_transition":154}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":219,"complex_overlap_transition":-87,"continuity_unknown":16,"mixed_unresolved_transition":91}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":75,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 279 / 2554976 samples / 0.044357 h

### `ami_TS3010a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.145913 | 0.143261 | 0.016944 | 0.128457 |
| v2 | 0.082081 | 0.207093 | 0.002081 | 0.078734 |

- Speech segments v1/v2: 181 / 309
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 127 / 0 / 54 / 0
- Removed internal pause: 2422112 samples (0.042051 h)
- Removed outer padding: 1347072 samples (0.023387 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 2080 | 3440 | 6656 | 14928 | 29740 | 45920 |
| End samples | -77856 | -6712 | -4560 | -2464 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 532 | 3040 | 5580 | 9936 | 18864 | 48765 | 77856 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 2 | 0 |
| Gap | 0 | 10 | 0 |
| Overlap | 0 | 3 | 1 |

- Topology episodes v1/v2/matched/added/removed: 68 / 135 / 16 / 119 / 52
- Unchanged/timing-only/topology-changing: 3 / 8 / 176
- Overlap takeover/return changes: -12 / -19
- Short-backchannel change: 5
- Handoffs v1/v2/matched/added/removed: 44 / 31 / 19 / 12 / 25

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 7 | 0.102941 |
| 100 ms | 8 | 0.117647 |
| 200 ms | 10 | 0.147059 |
| 500 ms | 16 | 0.235294 |

- Masked transitions v1/v2/change: 81 / 117 / 36
- Mask reasons v1: `{"complex_overlap_transition":7,"continuity_unknown":57,"mixed_unresolved_transition":17}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":45,"continuity_unknown":51,"mixed_unresolved_transition":21}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":45,"complex_overlap_transition":-7,"continuity_unknown":-6,"mixed_unresolved_transition":4}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":54,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 80 / 971200 samples / 0.016861 h

### `ami_TS3010b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.402438 | 0.176862 | 0.021559 | 0.380472 |
| v2 | 0.279288 | 0.300013 | 0.004990 | 0.272426 |

- Speech segments v1/v2: 324 / 789
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 292 / 0 / 32 / 0
- Removed internal pause: 5834592 samples (0.101295 h)
- Removed outer padding: 1840752 samples (0.031957 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 880 | 2560 | 4160 | 5280 | 20974 | 35840 |
| End samples | -91968 | -5204 | -3656 | -2104 | -480 | 0 | 0 | 0 |
| Absolute samples | 0 | 480 | 2400 | 4224 | 6113 | 8025 | 21310 | 91968 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 0 | 0 |
| Gap | 0 | 30 | 0 |
| Overlap | 3 | 8 | 9 |

- Topology episodes v1/v2/matched/added/removed: 140 / 456 / 55 / 401 / 85
- Unchanged/timing-only/topology-changing: 20 / 24 / 497
- Overlap takeover/return changes: -22 / -21
- Short-backchannel change: 6
- Handoffs v1/v2/matched/added/removed: 92 / 83 / 51 / 32 / 41

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 33 | 0.235714 |
| 100 ms | 37 | 0.264286 |
| 200 ms | 52 | 0.371429 |
| 500 ms | 55 | 0.392857 |

- Masked transitions v1/v2/change: 132 / 239 / 107
- Mask reasons v1: `{"complex_overlap_transition":7,"continuity_unknown":108,"mixed_unresolved_transition":17}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":44,"continuity_unknown":151,"mixed_unresolved_transition":44}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":44,"complex_overlap_transition":-7,"continuity_unknown":43,"mixed_unresolved_transition":27}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":32,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 60 / 682864 samples / 0.011855 h

### `ami_TS3010c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.416169 | 0.178831 | 0.058616 | 0.356813 |
| v2 | 0.289142 | 0.305858 | 0.014604 | 0.271584 |

- Speech segments v1/v2: 493 / 914
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 412 / 0 / 81 / 0
- Removed internal pause: 5729136 samples (0.099464 h)
- Removed outer padding: 3037696 samples (0.052738 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 960 | 2560 | 4320 | 8344 | 26099 | 93280 |
| End samples | -105696 | -4940 | -3224 | -1728 | -42 | 0 | 0 | 0 |
| Absolute samples | 0 | 480 | 2128 | 3936 | 6262 | 11176 | 40412 | 105696 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 2 | 0 |
| Gap | 0 | 43 | 0 |
| Overlap | 0 | 13 | 16 |

- Topology episodes v1/v2/matched/added/removed: 207 / 429 / 84 / 345 / 123
- Unchanged/timing-only/topology-changing: 27 / 42 / 483
- Overlap takeover/return changes: -26 / -37
- Short-backchannel change: 11
- Handoffs v1/v2/matched/added/removed: 163 / 144 / 91 / 53 / 72

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 35 | 0.169082 |
| 100 ms | 46 | 0.222222 |
| 200 ms | 72 | 0.347826 |
| 500 ms | 84 | 0.405797 |

- Masked transitions v1/v2/change: 167 / 280 / 113
- Mask reasons v1: `{"complex_overlap_transition":26,"continuity_unknown":112,"mixed_unresolved_transition":29}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":93,"continuity_unknown":132,"mixed_unresolved_transition":55}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":93,"complex_overlap_transition":-26,"continuity_unknown":20,"mixed_unresolved_transition":26}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":81,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 168 / 2127488 samples / 0.036936 h

### `ami_TS3010d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.349520 | 0.184008 | 0.039449 | 0.309541 |
| v2 | 0.226328 | 0.307200 | 0.009722 | 0.213420 |

- Speech segments v1/v2: 470 / 825
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 404 / 0 / 66 / 0
- Removed internal pause: 5314224 samples (0.092261 h)
- Removed outer padding: 2662304 samples (0.046221 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 640 | 2400 | 4160 | 7016 | 22900 | 38880 |
| End samples | -42720 | -5556 | -3752 | -1920 | -496 | 0 | 0 | 0 |
| Absolute samples | 0 | 480 | 2240 | 4240 | 6736 | 9760 | 23988 | 42720 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 0 | 6 | 0 |
| Gap | 0 | 34 | 0 |
| Overlap | 4 | 10 | 13 |

- Topology episodes v1/v2/matched/added/removed: 206 / 341 / 68 / 273 / 138
- Unchanged/timing-only/topology-changing: 24 / 24 / 431
- Overlap takeover/return changes: -28 / -35
- Short-backchannel change: 8
- Handoffs v1/v2/matched/added/removed: 160 / 131 / 73 / 58 / 87

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 39 | 0.189320 |
| 100 ms | 45 | 0.218447 |
| 200 ms | 54 | 0.262136 |
| 500 ms | 68 | 0.330097 |

- Masked transitions v1/v2/change: 187 / 322 / 135
- Mask reasons v1: `{"complex_overlap_transition":26,"continuity_unknown":141,"mixed_unresolved_transition":20}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":84,"complex_overlap_transition":2,"continuity_unknown":169,"mixed_unresolved_transition":67}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":84,"complex_overlap_transition":-24,"continuity_unknown":28,"mixed_unresolved_transition":47}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":66,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 122 / 1329104 samples / 0.023075 h

### `ami_TS3011a`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.342554 | 0.076823 | 0.037448 | 0.304680 |
| v2 | 0.261068 | 0.158309 | 0.014821 | 0.244111 |

- Speech segments v1/v2: 271 / 708
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 236 / 0 / 35 / 0
- Removed internal pause: 4447600 samples (0.077215 h)
- Removed outer padding: 1220496 samples (0.021189 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 2240 | 4000 | 5640 | 16088 | 70240 |
| End samples | -77408 | -2772 | -1248 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 640 | 2516 | 5118 | 9983 | 35575 | 77408 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 2 | 0 |
| Gap | 0 | 16 | 0 |
| Overlap | 3 | 2 | 22 |

- Topology episodes v1/v2/matched/added/removed: 137 / 398 / 49 / 349 / 88
- Unchanged/timing-only/topology-changing: 14 / 28 / 444
- Overlap takeover/return changes: -18 / -49
- Short-backchannel change: 27
- Handoffs v1/v2/matched/added/removed: 69 / 109 / 42 / 67 / 27

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 23 | 0.167883 |
| 100 ms | 32 | 0.233577 |
| 200 ms | 43 | 0.313869 |
| 500 ms | 49 | 0.357664 |

- Masked transitions v1/v2/change: 78 / 176 / 98
- Mask reasons v1: `{"complex_overlap_transition":23,"continuity_unknown":43,"mixed_unresolved_transition":12}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":75,"complex_overlap_transition":3,"continuity_unknown":50,"mixed_unresolved_transition":48}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":75,"complex_overlap_transition":-20,"continuity_unknown":7,"mixed_unresolved_transition":36}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":35,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 98 / 995008 samples / 0.017274 h

### `ami_TS3011b`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.541317 | 0.073741 | 0.084057 | 0.456073 |
| v2 | 0.438025 | 0.177033 | 0.041765 | 0.391239 |

- Speech segments v1/v2: 539 / 1252
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 508 / 0 / 31 / 0
- Removed internal pause: 6525360 samples (0.113287 h)
- Removed outer padding: 1912976 samples (0.033211 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 0 | 1440 | 3200 | 4480 | 16424 | 44160 |
| End samples | -69104 | -2240 | -896 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 512 | 1908 | 3968 | 6552 | 27814 | 69104 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 5 | 0 |
| Gap | 0 | 37 | 0 |
| Overlap | 2 | 8 | 57 |

- Topology episodes v1/v2/matched/added/removed: 265 / 704 / 118 / 586 / 147
- Unchanged/timing-only/topology-changing: 41 / 62 / 748
- Overlap takeover/return changes: -25 / -39
- Short-backchannel change: 33
- Handoffs v1/v2/matched/added/removed: 160 / 232 / 103 / 129 / 57

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 66 | 0.249057 |
| 100 ms | 81 | 0.305660 |
| 200 ms | 104 | 0.392453 |
| 500 ms | 118 | 0.445283 |

- Masked transitions v1/v2/change: 134 / 261 / 127
- Mask reasons v1: `{"complex_overlap_transition":34,"continuity_unknown":50,"mixed_unresolved_transition":50}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":86,"complex_overlap_transition":10,"continuity_unknown":46,"mixed_unresolved_transition":119}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":86,"complex_overlap_transition":-24,"continuity_unknown":-4,"mixed_unresolved_transition":69}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":31,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 119 / 986496 samples / 0.017127 h

### `ami_TS3011c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.564518 | 0.100760 | 0.084502 | 0.479078 |
| v2 | 0.441676 | 0.223602 | 0.035459 | 0.401402 |

- Speech segments v1/v2: 558 / 1266
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 502 / 0 / 56 / 0
- Removed internal pause: 7310992 samples (0.126927 h)
- Removed outer padding: 1960032 samples (0.034028 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 1920 | 3824 | 5752 | 19667 | 41920 |
| End samples | -52944 | -2324 | -1064 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 640 | 2096 | 4160 | 6842 | 22831 | 52944 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 1 | 3 | 0 |
| Gap | 0 | 54 | 0 |
| Overlap | 3 | 8 | 56 |

- Topology episodes v1/v2/matched/added/removed: 322 / 700 / 133 / 567 / 189
- Unchanged/timing-only/topology-changing: 46 / 73 / 770
- Overlap takeover/return changes: -34 / -82
- Short-backchannel change: 48
- Handoffs v1/v2/matched/added/removed: 192 / 273 / 113 / 160 / 79

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 78 | 0.242236 |
| 100 ms | 93 | 0.288820 |
| 200 ms | 120 | 0.372671 |
| 500 ms | 133 | 0.413043 |

- Masked transitions v1/v2/change: 121 / 295 / 174
- Mask reasons v1: `{"complex_overlap_transition":30,"continuity_unknown":61,"mixed_unresolved_transition":30}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":96,"complex_overlap_transition":7,"continuity_unknown":69,"mixed_unresolved_transition":123}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":96,"complex_overlap_transition":-23,"continuity_unknown":8,"mixed_unresolved_transition":93}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":56,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 137 / 1422560 samples / 0.024697 h

### `ami_TS3011d`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.445035 | 0.154247 | 0.089062 | 0.354916 |
| v2 | 0.336857 | 0.262425 | 0.033036 | 0.299166 |

- Speech segments v1/v2: 626 / 1109
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 540 / 0 / 86 / 0
- Removed internal pause: 5424928 samples (0.094183 h)
- Removed outer padding: 3210784 samples (0.055743 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 1760 | 4160 | 6400 | 27848 | 44160 |
| End samples | -80784 | -2820 | -1184 | 0 | 0 | 0 | 0 | 0 |
| Absolute samples | 0 | 0 | 792 | 2272 | 5120 | 13225 | 44033 | 80784 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 3 | 4 | 0 |
| Gap | 0 | 55 | 0 |
| Overlap | 3 | 9 | 37 |

- Topology episodes v1/v2/matched/added/removed: 291 / 523 / 125 / 398 / 166
- Unchanged/timing-only/topology-changing: 39 / 70 / 580
- Overlap takeover/return changes: -36 / -56
- Short-backchannel change: 27
- Handoffs v1/v2/matched/added/removed: 209 / 237 / 135 / 102 / 74

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 56 | 0.192440 |
| 100 ms | 82 | 0.281787 |
| 200 ms | 109 | 0.374570 |
| 500 ms | 125 | 0.429553 |

- Masked transitions v1/v2/change: 158 / 304 / 146
- Mask reasons v1: `{"complex_overlap_transition":46,"continuity_unknown":73,"mixed_unresolved_transition":39}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":114,"complex_overlap_transition":5,"continuity_unknown":80,"mixed_unresolved_transition":105}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":114,"complex_overlap_transition":-41,"continuity_unknown":7,"mixed_unresolved_transition":66}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":86,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 210 / 2256432 samples / 0.039174 h

### `ami_TS3012c`

| Reference | Speech h | Silence h | Overlap h | Reliable-solo h |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.580312 | 0.079818 | 0.130939 | 0.447402 |
| v2 | 0.469613 | 0.190517 | 0.075243 | 0.388155 |

- Speech segments v1/v2: 886 / 1527
- Deterministic/ambiguous/unpaired-v1/unpaired-v2 correspondences: 809 / 0 / 77 / 0
- Removed internal pause: 6617712 samples (0.114891 h)
- Removed outer padding: 2905504 samples (0.050443 h)

| Boundary displacement | p00 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start samples | 0 | 0 | 480 | 1600 | 3200 | 4960 | 9894 | 22400 |
| End samples | -48480 | -2864 | -1824 | -528 | 0 | 0 | 0 | 80 |
| Absolute samples | 0 | 0 | 1280 | 2400 | 3936 | 5865 | 11652 | 48480 |

| v1 → v2 family | Direct | Gap | Overlap |
| --- | ---: | ---: | ---: |
| Direct | 5 | 15 | 0 |
| Gap | 0 | 45 | 0 |
| Overlap | 3 | 6 | 119 |

- Topology episodes v1/v2/matched/added/removed: 420 / 715 / 200 / 515 / 220
- Unchanged/timing-only/topology-changing: 51 / 125 / 759
- Overlap takeover/return changes: -37 / -64
- Short-backchannel change: 30
- Handoffs v1/v2/matched/added/removed: 251 / 304 / 153 / 151 / 98

| Retention collar | Count | Proportion of v1 |
| --- | ---: | ---: |
| 50 ms | 77 | 0.183333 |
| 100 ms | 113 | 0.269048 |
| 200 ms | 172 | 0.409524 |
| 500 ms | 200 | 0.476190 |

- Masked transitions v1/v2/change: 227 / 400 / 173
- Mask reasons v1: `{"complex_overlap_transition":81,"continuity_unknown":82,"mixed_unresolved_transition":64}`
- Mask reasons v2: `{"ambiguous_nonlexical_vocalization_crossing":105,"complex_overlap_transition":39,"continuity_unknown":95,"mixed_unresolved_transition":161}`
- Mask reason changes: `{"ambiguous_nonlexical_vocalization_crossing":105,"complex_overlap_transition":-42,"continuity_unknown":13,"mixed_unresolved_transition":97}`
- Alignment risks: `{"ambiguous_v1_speech_correspondences":0,"clipped_tail_rttm_rows":0,"unpaired_v1_speech_segments":77,"unpaired_v2_speech_segments":0,"v1_source_clipped_speech_spans":0}`
- Nonlexical masks: 123 / 1046784 samples / 0.018173 h
