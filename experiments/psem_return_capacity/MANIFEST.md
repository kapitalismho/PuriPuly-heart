# Manifest (freeze v2 COMPLETE, docs-consolidated)
Freeze v2 `psem.return_capacity.freeze.v2` frozen_at `2026-09-09T21:20:01.247159Z` preserved (never rewritten); docs-only consolidation actual UTC in FREEZE revision_record, no algorithm change, no rerun needed (ledger binds same freeze_id/frozen_at). Smoke 18/18 edges, run 31/31 deterministic (7 guards + 3 positives, 9 arms: none/f0_orig/f0_rearm/scalar-CF/oracle-CF/full4-CF + scalar/oracle/full4 current). No training/paid/captures/inference/production/publication/Git mutations. Prior dirs read-only.
## Owned final (sha256)
- FREEZE.json 9a7ab4c9a8b836b4f3076f40b61c53acfe8a43858cad351bda37092fa3a52faf
- probe.py dccb2274e81815aa504199158c26dce21f06323a5bcf5d2d076bbeb8ef44b610 (zero hash-sign lines; per-symbol reuse; return_spans plumbing retained unused, no runtime edit)
- ledger.json d01f07ac4e493749c0cf99ae86532691da2740eb503d4bef4d28e80b6124cab0 (v2 21:20:12Z, 31 checks)
- RETURN_CAPACITY.md 42fa777ec4863282f384e5db152927548213efebaf2480c312084176f0e3215b
- NEXT_DECISION.md 182a81622f1a21d3b23e286c08064418956b666313b390d34997c6c5d6130560
## Prior candidate preserved (revision record in FREEZE)
- v1 FREEZE 6c397dde5e6871b44bf44d85b1564d879d3223defe1fa94db44641cc18e5d63a; v1 ledgers 427b0ea9e800012c4a29c477abbe9643fa46571c9f95a2e70ad6b7882d736aaf and dd1b45c3441cc2cf0ebfe054aa12b06d4edf941ec27df2412506fa3cad1036e8
## Frozen inputs (sha256)
OBS 3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa; old_grid 8f9474eecbd13c235062bbe8616f93de5b8b6b6316695d9864724b87dbb5814a; live_headroom e9d1137d32e0ef99acf321341df5388db8fb8ddad0b64ddb5b2a65c303f57974; replay 2e08a9ece97c0b8f80730450f5f05124e7b3f97d6268c308af2603720700753d; PhaseA freeze 543eb0c6cfa36e651375cad5c23b7337cc0f5996ace56e74220b5af0bcb5689b; decision freeze d131883413aa969c0aa6ed5eb32ac134f411ed511af372baa3bf6e5d88d5ac67; NP1 a4ed5e84f9fd866263646660edde1c5400f1269de1f58a98befa044b19da5005 NP2 a67b4a9953a1a08a55225b221f62af6208c767b67bedff6cdd72feef1dd76177 NP3 d92884ae50dcbc4e841c5dbc13c805665130dcec9fddc43a4745a32ed329b22d; P3T freeze 12e0cbeff1c38c239ff45d99032e1e6c66acb8674d01ef51c225a5103233592a ledger 65b3cff0f9be72e7fe86d6e3a205fa8b3c309a28c7f1a6f93f1cceb695c7527a probe bf351addc2e1e0c8ba50b9f5b63cc1a44436759efb4e43a3a82f43ad46d3c5ec; ARCH 8971048f129d49b48a3abf6b9d546906e155eee5ed3b1e845d60e281e062d068
## Reproduce
python experiments/psem_return_capacity/probe.py smoke
python experiments/psem_return_capacity/probe.py run
