# PSEM training-strategy gate results

Evidence status: **not run**.

No official issue #76 arm has been trained and EVAL has not been opened by this experiment.
This file must not name a training-strategy winner until both fixed seeds for all three arms,
their ensembles, the complete unique-score curves, topology and corpus slices, timing and
compute measurements, and meeting-bootstrap uncertainty have passed the terminal evidence
gate.

The eventual report must answer:

1. Did real WavLM encoder fine-tuning move the full frontier relative to frozen WavLM?
2. Did the 5–10M scratch PSEM match or exceed either pretrained path?
3. Where did the three curves cross, and which topology caused the crossings?
4. Were differences larger than seed and meeting uncertainty?
5. Which training strategy should own the next PSEM research stage?
6. What remains unknown and must not be inferred from this experiment?
