# Run artifacts

`run_b0_replay.py` writes one self-describing JSON artifact per run, named
`result_<manifest_id>_<result_id-prefix>.json`, containing:

- baseline SHA and manifest id/hash;
- runtime metadata (Python, Windows, CPU, RAM, ORT version and thread
  configuration, VAD profile);
- start/end UTC times;
- per-epoch B0 boundary events and `DetectorProgress` traces;
- the coalescing report and logical cuts (see `coalescing.py`);
- `result_sha256`, a canonical hash of the artifact content.

`result_sha256` is computed over the canonical JSON of the artifact excluding
the `result_sha256` field itself and is verifiable via
`RunResult.verify_self_hash`.

Phase 0 B0 evidence: on the synthetic golden corpus the peer-profile VAD
produces no cross-utterance boundaries (at most one utterance per clip), so
`vad_cut_count = 0` here. Deterministic two-utterance B0 boundary behavior is
covered by the fake-engine tests; Phase 1 replaces the corpus with real
speech synthesis.
