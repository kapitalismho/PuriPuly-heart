# Speaker attribution and early-boundary benchmark (#266)

This harness measures the shipped Peer speaker policy per ASR route and Nemotron profile. It scores admitted source units, their presented speaker identity and their timing against case-local references. It does not score raw diarizer output.

Expected behavior comes from the issue contracts (#211 R1/R8/C2, #215, #216, #264 R2/R3), not from current code. See #266 for the metric definitions.

## Layout

| Path | Content |
| --- | --- |
| `corpus/spec.json` | Frozen case definitions (26 cases, about 15 minutes of audio) |
| `corpus/make_spec.py` | Authoring script that wrote `spec.json` |
| `corpus/references/*.json` | Case-local references: speaker turns, text and word timing where available |
| `corpus/manifest.json` | Case durations, speaker counts and PCM hashes |
| `bench/corpus.py` | Builds case WAVs (not committed) and references from public sources |
| `bench/run.py` | Runs cases through the actual development host |
| `bench/report.py` | Scores runs and writes `summary.json`, `report.md` and per-case scores |
| `bench/score.py` | Deterministic scorer; contract tests live in `tests/` |

## Sources

- LibriSpeech test-clean (CC BY 4.0), fetched per utterance from a pinned Hugging Face mirror revision.
- AMI Meeting Corpus Mix-Headset audio and manual word annotations v1.6.2 (CC BY 4.0), pinned mirror revision. Six 75 s windows were selected by `python -m bench.corpus select-ami` for the most continuous speaker transfers.
- Common Voice 17.0 Korean/Japanese clips (CC0) from the existing local public corpus `sim-v02-public-corpus-600-v1`.
- Short interjections and backchannels use Windows SAPI voices. Their references mark `synthetic_voice: true`.

Assembled cases are synthetic and do not certify VRChat acoustics. Dataset speaker identifiers are used only to build case-local labels.

## Run

From this directory, with the worktree `.venv`:

```powershell
..\..\.venv\Scripts\python.exe -m bench.corpus build
..\..\.venv\Scripts\python.exe -m bench.run --root C:\tmp\speaker_bench\run1 --routes local_qwen_gpu,local_parakeet_v3 --arms baseline,p104,p064
..\..\.venv\Scripts\python.exe -m bench.report --root C:\tmp\speaker_bench\run1
..\..\.venv\Scripts\python.exe -m pytest tests -p no:cacheprovider --rootdir . -c NUL
```

The runner composes the ordinary headless application in-process. It changes only two things:

- The Peer loopback device source is replaced by a real-time WAV source. The WAV source reports sample-exact capture spans.
- Read-only observers record source admission, first readable publication and speaker settlement.

The runner configures the host only through the application control catalog (`docs/cli.md`). It uses isolated settings, user data and an encrypted secret store under `--root`. Local ASR models are linked read-only. The Nemotron artifact is imported with `speaker.install`. Cloud credentials are read from `.env.local` and are never written to reports.

Each case runs in its own Peer capture session, and the speaker epoch is reset between cases. The desktop caption overlay must run, because first readable publication happens in its presenter.

Prerequisites:

- Windows with a Vulkan GPU.
- `build/nemotron`, plus `build/gpu_worker` for the GPU Qwen route.
- `ffmpeg` on `PATH`.
- Installed local ASR models.

Do not run the benchmark while VRChat or SteamVR is running. #218 measures non-VR conditions, and the run would compete with the session for the GPU.
