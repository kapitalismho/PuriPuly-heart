# Smart Turn experiment

The endpoint experiment is disabled by default. It can be enabled for a local run without changing persisted settings.

Set these environment variables before starting the application:

```text
PURIPULY_SMART_TURN_STAGE=shadow|active
PURIPULY_SMART_TURN_MODEL_PATH=C:\path\to\smart-turn-v3.2-cpu.onnx
PURIPULY_SMART_TURN_THRESHOLD=0.5
```

`shadow` keeps the existing VAD boundary and records probe scores. `active` changes the silence boundary to the 800 ms hard limit, probes at 224, 416, and 608 ms, and preserves the existing self speculative-translation recovery path.

The model input is the complete current turn at 16 kHz mono. Inputs longer than eight seconds keep the newest eight seconds; shorter inputs are left-padded with zeroes.

The official CPU model is `smart-turn-v3.2-cpu.onnx` from [`pipecat-ai/smart-turn-v3`](https://huggingface.co/pipecat-ai/smart-turn-v3). Its SHA-256 is `2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f`.

To measure local CPU inference against `.npy` files containing one-dimensional float audio at 16 kHz:

```text
python scripts/experiments/benchmark_smart_turn.py C:\path\to\smart-turn-v3.2-cpu.onnx C:\path\to\turn.npy --repeats 5
```

Keep shadow and active logs with the corresponding capture channel, probe time, score, threshold, acceptance, and inference latency. Evaluate the first accepted probe across the complete policy, including the hard boundary, rather than reporting each probe in isolation.

For public prediction artifacts, use one JSON object per record with this shape:

```json
{
  "id": "sample-id",
  "endpoint_bool": true,
  "scores": {"224": 0.61, "416": 0.42, "608": 0.39},
  "inference_ms": {"224": 78.4, "416": 79.1, "608": 78.7}
}
```

The policy summary reports first-crossing counts, early false-complete rate, hard-boundary fallbacks, model inference latency, end-to-end decision latency, and the delta from the current effective 512 ms VAD boundary. Missing per-probe latency is reported as unavailable rather than inferred:

```text
python scripts/experiments/evaluate_smart_turn_policy.py path\to\prediction-artifact.json
```

The public Smart Turn v3.2 test dataset is [`pipecat-ai/smart-turn-data-v3.2-test`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test). Its audio must be scored into the artifact schema before policy evaluation. The checked-in [CPU benchmark artifact](experiments/smart_turn_cpu_benchmark_2026-08-01.json) is a runtime smoke measurement using synthetic input, not an accuracy result.

A 16-record public-data smoke result is recorded in [this policy artifact](experiments/smart_turn_policy_public_sample_2026-08-01.json). It validates the measurement path and remains insufficient for changing the recovery policy.
