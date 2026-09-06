# Gate 1 decision evidence (R-H-SC vs F0)

This report is evidence for human Gate 1 adjudication.
It is not a gate receipt, confirmation authorization, or T2/EVAL opening.

Arm: R-H-SC
Seed: 7301
Calib frames: 169951
DEV meetings: 10 (AMI 7, AliMeeting 3)

Primary aggregation: equal-corpus macro, then AMI, AliMeeting, pooled.
Bootstrap CIs are meeting-mean paired-source intervals; they are not pooled-rate or macro CIs.
No winner utility is applied. Timing criterion is p90 delay <= F0 + 80 ms.
Per-meeting and leave-one-meeting-out rows are listed without numeric dominance cutoffs.

## Horizon 100 ms
### raw
- F0 raw@0.5 reference: contamination=2161.68 miss=0.775487 false_cuts/h=228.734
- macro useful flag: True
- c_envelope threshold=0.588784 contamination=1561.58 miss=0.583874 false_cuts/h=108.642 jointly_useful_macro=True favorable_CI=True
  bootstrap meeting-mean H-F0 contamination [-723.7, -119.918] miss [-0.237296, -0.0810168]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=-684.54 d_miss=-0.262195 d_p90=157.0
  - alimeeting_R1021_M4073: d_cont=-1539.91 d_miss=-0.454545 d_p90=-61.99999999999977
  - alimeeting_R8009_M8019: d_cont=-834.973 d_miss=-0.25641 d_p90=302.0
  - ami_EN2009d: d_cont=-336.991 d_miss=-0.131222 d_p90=70.59999999999991
  - ami_ES2002b: d_cont=-55.7171 d_miss=-0.0740741 d_p90=12.0
  - ami_ES2009a: d_cont=172.351 d_miss=0 d_p90=-2.5
  - ami_ES2009b: d_cont=-152.749 d_miss=-0.101449 d_p90=-11.400000000000091
  - ami_ES2009c: d_cont=-118.099 d_miss=-0.101695 d_p90=-1.400000000000091
  - ami_ES2009d: d_cont=-274.471 d_miss=-0.10101 d_p90=7.800000000000182
  - ami_ES2015d: d_cont=-88.7637 d_miss=-0.0469484 d_p90=8.0
- m_envelope threshold=0.363104 contamination=1570.01 miss=0.57104 false_cuts/h=151.139 jointly_useful_macro=True favorable_CI=True
  bootstrap meeting-mean H-F0 contamination [-717.566, -83.6309] miss [-0.263764, -0.0509118]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=-703.553 d_miss=-0.29878 d_p90=41.0
  - alimeeting_R1021_M4073: d_cont=-1636.14 d_miss=-0.527273 d_p90=71.0
  - alimeeting_R8009_M8019: d_cont=-897.417 d_miss=-0.358974 d_p90=199.0
  - ami_EN2009d: d_cont=-211.173 d_miss=-0.0859729 d_p90=14.899999999999977
  - ami_ES2002b: d_cont=-46.1394 d_miss=-0.0555556 d_p90=3.7999999999999545
  - ami_ES2009a: d_cont=109.658 d_miss=0.00925926 d_p90=0.5
  - ami_ES2009b: d_cont=-64.4205 d_miss=-0.0724638 d_p90=-8.0
  - ami_ES2009c: d_cont=-116.25 d_miss=-0.0762712 d_p90=-22.399999999999977
  - ami_ES2009d: d_cont=-164.005 d_miss=-0.0707071 d_p90=1.2000000000000455
  - ami_ES2015d: d_cont=66.2686 d_miss=0.028169 d_p90=1.599999999999909

### calibrated
- F0 raw@0.5 reference: contamination=2161.68 miss=0.775487 false_cuts/h=228.734
- macro useful flag: True
- c_envelope threshold=0.0611806 contamination=1561.58 miss=0.583874 false_cuts/h=108.642 jointly_useful_macro=True favorable_CI=True
  bootstrap meeting-mean H-F0 contamination [-723.162, -139.979] miss [-0.248356, -0.0858385]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=-684.54 d_miss=-0.262195 d_p90=157.0
  - alimeeting_R1021_M4073: d_cont=-1539.91 d_miss=-0.454545 d_p90=-61.99999999999977
  - alimeeting_R8009_M8019: d_cont=-834.973 d_miss=-0.25641 d_p90=302.0
  - ami_EN2009d: d_cont=-336.991 d_miss=-0.131222 d_p90=70.59999999999991
  - ami_ES2002b: d_cont=-55.7171 d_miss=-0.0740741 d_p90=12.0
  - ami_ES2009a: d_cont=172.351 d_miss=0 d_p90=-2.5
  - ami_ES2009b: d_cont=-152.749 d_miss=-0.101449 d_p90=-11.400000000000091
  - ami_ES2009c: d_cont=-118.099 d_miss=-0.101695 d_p90=-1.400000000000091
  - ami_ES2009d: d_cont=-274.471 d_miss=-0.10101 d_p90=7.800000000000182
  - ami_ES2015d: d_cont=-88.7637 d_miss=-0.0469484 d_p90=8.0
- m_envelope threshold=0.0316321 contamination=1570.01 miss=0.57104 false_cuts/h=151.139 jointly_useful_macro=True favorable_CI=True
  bootstrap meeting-mean H-F0 contamination [-707.277, -88.9497] miss [-0.270812, -0.0538994]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=-703.553 d_miss=-0.29878 d_p90=41.0
  - alimeeting_R1021_M4073: d_cont=-1636.14 d_miss=-0.527273 d_p90=71.0
  - alimeeting_R8009_M8019: d_cont=-897.417 d_miss=-0.358974 d_p90=199.0
  - ami_EN2009d: d_cont=-211.173 d_miss=-0.0859729 d_p90=14.899999999999977
  - ami_ES2002b: d_cont=-46.1394 d_miss=-0.0555556 d_p90=3.7999999999999545
  - ami_ES2009a: d_cont=109.658 d_miss=0.00925926 d_p90=0.5
  - ami_ES2009b: d_cont=-64.4205 d_miss=-0.0724638 d_p90=-8.0
  - ami_ES2009c: d_cont=-116.25 d_miss=-0.0762712 d_p90=-22.399999999999977
  - ami_ES2009d: d_cont=-164.005 d_miss=-0.0707071 d_p90=1.2000000000000455
  - ami_ES2015d: d_cont=66.2686 d_miss=0.028169 d_p90=1.599999999999909

## Horizon 300 ms
### raw
- F0 raw@0.5 reference: contamination=1555.19 miss=0.537089 false_cuts/h=92.2821
- macro useful flag: False
- c_envelope threshold=0.391021 contamination=1566.37 miss=0.576841 false_cuts/h=89.4454 jointly_useful_macro=False favorable_CI=False
  bootstrap meeting-mean H-F0 contamination [-39.0729, 120.87] miss [0.00294464, 0.0651966]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=-46.6135 d_miss=-0.00609756 d_p90=169.0
  - alimeeting_R1021_M4073: d_cont=-194.838 d_miss=-0.0363636 d_p90=10.0
  - alimeeting_R8009_M8019: d_cont=-8.14643 d_miss=0.128205 d_p90=56.0
  - ami_EN2009d: d_cont=161.539 d_miss=0.0588235 d_p90=-58.40000000000009
  - ami_ES2002b: d_cont=27.3109 d_miss=0 d_p90=15.600000000000136
  - ami_ES2009a: d_cont=243.474 d_miss=0.0833333 d_p90=-6.2999999999999545
  - ami_ES2009b: d_cont=-53.1179 d_miss=-0.0144928 d_p90=14.400000000000091
  - ami_ES2009c: d_cont=99.3397 d_miss=0.059322 d_p90=-36.200000000000045
  - ami_ES2009d: d_cont=-56.7085 d_miss=-0.00505051 d_p90=37.0
  - ami_ES2015d: d_cont=242.132 d_miss=0.0704225 d_p90=-34.799999999999955
- m_envelope threshold=0.391021 contamination=1566.37 miss=0.576841 false_cuts/h=89.4454 jointly_useful_macro=False favorable_CI=False
  bootstrap meeting-mean H-F0 contamination [-43.73, 125.851] miss [0.00347568, 0.0657093]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=-46.6135 d_miss=-0.00609756 d_p90=169.0
  - alimeeting_R1021_M4073: d_cont=-194.838 d_miss=-0.0363636 d_p90=10.0
  - alimeeting_R8009_M8019: d_cont=-8.14643 d_miss=0.128205 d_p90=56.0
  - ami_EN2009d: d_cont=161.539 d_miss=0.0588235 d_p90=-58.40000000000009
  - ami_ES2002b: d_cont=27.3109 d_miss=0 d_p90=15.600000000000136
  - ami_ES2009a: d_cont=243.474 d_miss=0.0833333 d_p90=-6.2999999999999545
  - ami_ES2009b: d_cont=-53.1179 d_miss=-0.0144928 d_p90=14.400000000000091
  - ami_ES2009c: d_cont=99.3397 d_miss=0.059322 d_p90=-36.200000000000045
  - ami_ES2009d: d_cont=-56.7085 d_miss=-0.00505051 d_p90=37.0
  - ami_ES2015d: d_cont=242.132 d_miss=0.0704225 d_p90=-34.799999999999955

### calibrated
- F0 raw@0.5 reference: contamination=1555.19 miss=0.537089 false_cuts/h=92.2821
- macro useful flag: False
- c_envelope threshold=0.0344807 contamination=1566.37 miss=0.576841 false_cuts/h=89.4454 jointly_useful_macro=False favorable_CI=False
  bootstrap meeting-mean H-F0 contamination [-36.0196, 124.895] miss [0.00342607, 0.0662045]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=-46.6135 d_miss=-0.00609756 d_p90=169.0
  - alimeeting_R1021_M4073: d_cont=-194.838 d_miss=-0.0363636 d_p90=10.0
  - alimeeting_R8009_M8019: d_cont=-8.14643 d_miss=0.128205 d_p90=56.0
  - ami_EN2009d: d_cont=161.539 d_miss=0.0588235 d_p90=-58.40000000000009
  - ami_ES2002b: d_cont=27.3109 d_miss=0 d_p90=15.600000000000136
  - ami_ES2009a: d_cont=243.474 d_miss=0.0833333 d_p90=-6.2999999999999545
  - ami_ES2009b: d_cont=-53.1179 d_miss=-0.0144928 d_p90=14.400000000000091
  - ami_ES2009c: d_cont=99.3397 d_miss=0.059322 d_p90=-36.200000000000045
  - ami_ES2009d: d_cont=-56.7085 d_miss=-0.00505051 d_p90=37.0
  - ami_ES2015d: d_cont=242.132 d_miss=0.0704225 d_p90=-34.799999999999955
- m_envelope threshold=0.0344807 contamination=1566.37 miss=0.576841 false_cuts/h=89.4454 jointly_useful_macro=False favorable_CI=False
  bootstrap meeting-mean H-F0 contamination [-42.5144, 128.29] miss [0.00475585, 0.0662651]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=-46.6135 d_miss=-0.00609756 d_p90=169.0
  - alimeeting_R1021_M4073: d_cont=-194.838 d_miss=-0.0363636 d_p90=10.0
  - alimeeting_R8009_M8019: d_cont=-8.14643 d_miss=0.128205 d_p90=56.0
  - ami_EN2009d: d_cont=161.539 d_miss=0.0588235 d_p90=-58.40000000000009
  - ami_ES2002b: d_cont=27.3109 d_miss=0 d_p90=15.600000000000136
  - ami_ES2009a: d_cont=243.474 d_miss=0.0833333 d_p90=-6.2999999999999545
  - ami_ES2009b: d_cont=-53.1179 d_miss=-0.0144928 d_p90=14.400000000000091
  - ami_ES2009c: d_cont=99.3397 d_miss=0.059322 d_p90=-36.200000000000045
  - ami_ES2009d: d_cont=-56.7085 d_miss=-0.00505051 d_p90=37.0
  - ami_ES2015d: d_cont=242.132 d_miss=0.0704225 d_p90=-34.799999999999955

## Horizon 500 ms
### raw
- F0 raw@0.5 reference: contamination=1978.65 miss=0.687291 false_cuts/h=30.4871
- macro useful flag: False
- c_envelope threshold=0.723464 contamination=2232.24 miss=0.817277 false_cuts/h=30.4519 jointly_useful_macro=False favorable_CI=False
  bootstrap meeting-mean H-F0 contamination [70.8331, 355.377] miss [0.0255927, 0.153624]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=562.393 d_miss=0.304878 d_p90=6.0
  - alimeeting_R1021_M4073: d_cont=94.7269 d_miss=0.0545455 d_p90=3.0
  - alimeeting_R8009_M8019: d_cont=640.479 d_miss=0.247863 d_p90=32.0
  - ami_EN2009d: d_cont=-81.3868 d_miss=-0.0248869 d_p90=4.599999999999909
  - ami_ES2002b: d_cont=95.5796 d_miss=0.00925926 d_p90=-10.0
  - ami_ES2009a: d_cont=-26.3857 d_miss=0.0462963 d_p90=-9.0
  - ami_ES2009b: d_cont=13.8571 d_miss=0.0434783 d_p90=3.2999999999999545
  - ami_ES2009c: d_cont=268.751 d_miss=0.0423729 d_p90=-1.3999999999998636
  - ami_ES2009d: d_cont=262.581 d_miss=0.0555556 d_p90=-7.7999999999999545
  - ami_ES2015d: d_cont=200.538 d_miss=0.028169 d_p90=46.200000000000045
- m_envelope threshold=0.723464 contamination=2232.24 miss=0.817277 false_cuts/h=30.4519 jointly_useful_macro=False favorable_CI=False
  bootstrap meeting-mean H-F0 contamination [67.8023, 345.655] miss [0.025623, 0.151012]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=562.393 d_miss=0.304878 d_p90=6.0
  - alimeeting_R1021_M4073: d_cont=94.7269 d_miss=0.0545455 d_p90=3.0
  - alimeeting_R8009_M8019: d_cont=640.479 d_miss=0.247863 d_p90=32.0
  - ami_EN2009d: d_cont=-81.3868 d_miss=-0.0248869 d_p90=4.599999999999909
  - ami_ES2002b: d_cont=95.5796 d_miss=0.00925926 d_p90=-10.0
  - ami_ES2009a: d_cont=-26.3857 d_miss=0.0462963 d_p90=-9.0
  - ami_ES2009b: d_cont=13.8571 d_miss=0.0434783 d_p90=3.2999999999999545
  - ami_ES2009c: d_cont=268.751 d_miss=0.0423729 d_p90=-1.3999999999998636
  - ami_ES2009d: d_cont=262.581 d_miss=0.0555556 d_p90=-7.7999999999999545
  - ami_ES2015d: d_cont=200.538 d_miss=0.028169 d_p90=46.200000000000045

### calibrated
- F0 raw@0.5 reference: contamination=1978.65 miss=0.687291 false_cuts/h=30.4871
- macro useful flag: False
- c_envelope threshold=0.0929009 contamination=2232.24 miss=0.817277 false_cuts/h=30.4519 jointly_useful_macro=False favorable_CI=False
  bootstrap meeting-mean H-F0 contamination [72.4296, 356.825] miss [0.0257239, 0.150599]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=562.393 d_miss=0.304878 d_p90=6.0
  - alimeeting_R1021_M4073: d_cont=94.7269 d_miss=0.0545455 d_p90=3.0
  - alimeeting_R8009_M8019: d_cont=640.479 d_miss=0.247863 d_p90=32.0
  - ami_EN2009d: d_cont=-81.3868 d_miss=-0.0248869 d_p90=4.599999999999909
  - ami_ES2002b: d_cont=95.5796 d_miss=0.00925926 d_p90=-10.0
  - ami_ES2009a: d_cont=-26.3857 d_miss=0.0462963 d_p90=-9.0
  - ami_ES2009b: d_cont=13.8571 d_miss=0.0434783 d_p90=3.2999999999999545
  - ami_ES2009c: d_cont=268.751 d_miss=0.0423729 d_p90=-1.3999999999998636
  - ami_ES2009d: d_cont=262.581 d_miss=0.0555556 d_p90=-7.7999999999999545
  - ami_ES2015d: d_cont=200.538 d_miss=0.028169 d_p90=46.200000000000045
- m_envelope threshold=0.0929009 contamination=2232.24 miss=0.817277 false_cuts/h=30.4519 jointly_useful_macro=False favorable_CI=False
  bootstrap meeting-mean H-F0 contamination [74.6996, 347.867] miss [0.0250222, 0.145931]
  per-meeting H-F0 contamination/miss/p90:
  - alimeeting_R1019_M1928: d_cont=562.393 d_miss=0.304878 d_p90=6.0
  - alimeeting_R1021_M4073: d_cont=94.7269 d_miss=0.0545455 d_p90=3.0
  - alimeeting_R8009_M8019: d_cont=640.479 d_miss=0.247863 d_p90=32.0
  - ami_EN2009d: d_cont=-81.3868 d_miss=-0.0248869 d_p90=4.599999999999909
  - ami_ES2002b: d_cont=95.5796 d_miss=0.00925926 d_p90=-10.0
  - ami_ES2009a: d_cont=-26.3857 d_miss=0.0462963 d_p90=-9.0
  - ami_ES2009b: d_cont=13.8571 d_miss=0.0434783 d_p90=3.2999999999999545
  - ami_ES2009c: d_cont=268.751 d_miss=0.0423729 d_p90=-1.3999999999998636
  - ami_ES2009d: d_cont=262.581 d_miss=0.0555556 d_p90=-7.7999999999999545
  - ami_ES2015d: d_cont=200.538 d_miss=0.028169 d_p90=46.200000000000045

Director must judge whether any gain is primarily one meeting or topology from the rows above.
Do not treat this file as Gate 1 ACCEPT, OPEN-T2, or confirmation.
