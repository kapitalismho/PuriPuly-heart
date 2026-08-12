# R6 Online Speaker Memory and Handoff

`EXPERIMENT_PLAN.en.md` is the scientific authority. `config.json` freezes the R6-0 data roles,
representation matrix, timing conventions, policy views, input identities, and external result path.

R6 uses the locked CPU research environment and the existing external cache. Outputs are written to:

```text
%SRSCD_CACHE_ROOT%/results/r6/online_speaker_memory_handoff_v1
```

The development and evaluation meetings are disjoint. Thresholds and operating points are selected
only from the five development meetings. The five frozen R4 natural meetings are evaluated only
after selection.

The 86 positive R4 anchors are retained in the protocol inventory as a compatibility count. They are
not a complete segmentation of the continuous meetings. R6-A1 therefore creates chronological
first-handoff units from the complete AMI and AliMeeting speaker annotations; otherwise genuine
speaker activity outside the 86 anchors would be miscounted as false events.

PowerShell entrypoints:

```powershell
$env:SRSCD_CACHE_ROOT = 'C:\Users\salee\AppData\Local\puripuly-heart-research\speaker_representation_scd_v1'
$python = 'experiments\speaker_representation_scd\environment\.venv\Scripts\python.exe'
& $python -m experiments.online_speaker_memory_handoff.protocol prepare
& $python -m experiments.online_speaker_memory_handoff.a1 smoke --representation m-l1
& $python -m experiments.online_speaker_memory_handoff.a1 run --representation m-l1
& $python -m experiments.online_speaker_memory_handoff.a1 run --representation e-s3
& $python -m experiments.online_speaker_memory_handoff.a1 run --representation e-final
& $python -m experiments.online_speaker_memory_handoff.a1 report
& $python -m experiments.online_speaker_memory_handoff.decision
```

`m-l1` and `e-s3` reuse the frozen R4 evaluation feature caches and extract only the disjoint
development material. `e-final` materializes only the query and enrollment windows required by the
chronological A1 units. A2 and B are entered only for representations promoted by the A1 gate.
The decision entrypoint requires schema-v2 metrics for all three representations, records the
Sortformer environment smoke and existing-baseline compatibility boundary, and writes the final R6
decision under the external result root.
