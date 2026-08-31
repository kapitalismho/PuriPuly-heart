# Issue 107 derived runtime image

This directory defines a thin, reproducible image on top of the exact NVIDIA NGC PyTorch base already named by the issue-107 runtime contract:

```text
nvcr.io/nvidia/pytorch@sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb
```

The build repairs the malformed `nvidia-nvimgcodec-cu12` wheel tag, preserves the base NGC torch/CUDA distribution and binary inventory, installs the validated exact compatibility pins, builds torchaudio from commit `d8831425203385077a03c1d92cfbbe3bf2106008`, installs NeMo from commit `1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6`, isolates duplicate system and setuptools-vendor metadata, runs `pip check`, and performs a CPU-side import and source-origin validation. `runtime-constraints.txt` locks all 81 distributions introduced or changed relative to the immutable base inventory. The validator requires the constraints map to equal the observed base-to-final delta, rejects any duplicate final distribution name even when versions match, rejects missing, unexpected, or mismatched distributions, and records every final distribution metadata path and version with a canonical SHA-256. The base snapshot separately records every discovered metadata distribution before isolation.

Do not run NeMo's `docker/common/install_dep.sh --library all`. It is not part of this image and can replace the protected NGC torch/CUDA stack.

## Local WSL build

From Windows PowerShell at the repository root:

```powershell
& experiments\psem_sortformer_adaptation_depth\environment\build-image.ps1
```

The wrapper uses Docker Engine in the WSL `Ubuntu` distribution and builds `linux/amd64`. Override `-Distribution` or `-Tag` when needed. It does not log in, push, publish, create a Pod, or start training.

Equivalent WSL command:

```bash
docker build --platform linux/amd64 \
  --file experiments/psem_sortformer_adaptation_depth/environment/Dockerfile \
  --tag puripuly-heart/issue-107-runtime:local \
  experiments/psem_sortformer_adaptation_depth/environment
```

## Validation boundary

The Docker build runs:

```bash
python /opt/psem/validate-runtime.py \
  --mode build \
  --receipt /opt/psem/build-validation.json
```

Build mode validates package identities, protected NGC inventory, metadata isolation, NeMo origins, torchaudio imports, the soundfile backend, and `pip check`. It cannot validate A40 CUDA access, the host driver, or the eventual registry manifest digest.

After the approved registry push, the activated runnable image is:

```text
kapitalismho/puripuly-heart@sha256:14acbef50fa15281bded1d3fbbcd8029091aeba0692d5647255aa5b90eff8ca7
```

The runtime contracts bind that immutable derived manifest digest while preserving the NGC digest as base-image provenance. On a one-GPU Pod, validate the activated digest with:

```bash
export PSEM_CONTAINER_IMAGE_IDENTITY=sha256:<derived-manifest-digest>
python /opt/psem/validate-runtime.py \
  --mode runtime \
  --expected-image-identity "$PSEM_CONTAINER_IMAGE_IDENTITY" \
  --receipt /workspace/issue-107/runtime-image-validation.json
```

The derived image deliberately does not set `PSEM_CONTAINER_IMAGE_IDENTITY`; the launch control plane supplies the activated manifest digest. Build validation does not substitute for the one-GPU runtime validation, clean-candidate preflight, or material gate.
