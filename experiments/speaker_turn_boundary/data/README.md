# Experiment data

No public corpora, private recordings, or large binaries are committed here.
Per issue #51 Phase 2 rules, download/build scripts, manifests, hashes, and
small regenerable artifacts are committed instead.

- `manifests/b0_phase0.json` — deterministic dataset manifest for the Phase 0
  golden cases: baseline SHA, case metadata, wav SHA-256 hashes, and GT
  active-speaker regions. Regenerate it (byte-for-byte) with:

  ```powershell
  uv run python -m experiments.speaker_turn_boundary.build_synthetic_cases
  ```

- `generated/*.wav` — deterministic 16 kHz mono PCM16 clips synthesized by
  `experiments/speaker_turn_boundary/synthetic.py` (numpy only, fixed seed).
  The manifest hashes pin their exact bytes; tests regenerate them and
  validate against the manifest.

The Phase 0 corpus is "speech-like" synthetic audio, not real speech; the
production Silero VAD detects at most one utterance per clip. Phase 1 will
add LibriSpeech-based synthetic cases (D1) and real-corpus manifests (D2/D3)
without committing audio blobs.
