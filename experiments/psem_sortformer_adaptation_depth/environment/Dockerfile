FROM nvcr.io/nvidia/pytorch@sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb AS torchaudio-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_CONFIG_FILE=/dev/null \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL= \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY prepare-runtime.py /opt/psem/prepare-runtime.py

RUN python /opt/psem/prepare-runtime.py repair-nvimgcodec \
    && git clone --branch v2.6.0 --depth 1 https://github.com/pytorch/audio.git /tmp/audio \
    && test "$(git -C /tmp/audio rev-parse HEAD)" = "d8831425203385077a03c1d92cfbbe3bf2106008" \
    && SOURCE_DATE_EPOCH=1738113115 \
       PYTORCH_VERSION="$(python -c 'import importlib.metadata; print(importlib.metadata.version("torch"))')" \
       BUILD_SOX=0 BUILD_KALDI=0 BUILD_RNNT=0 USE_CUDA=0 USE_FFMPEG=0 \
       python -m pip wheel --no-deps --no-build-isolation --wheel-dir /opt/wheels /tmp/audio

FROM nvcr.io/nvidia/pytorch@sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb

LABEL org.opencontainers.image.base.name="nvcr.io/nvidia/pytorch@sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb" \
      io.puripuly.issue="107" \
      io.puripuly.nemo.revision="1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6" \
      io.puripuly.torchaudio.revision="d8831425203385077a03c1d92cfbbe3bf2106008"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_CONFIG_FILE=/dev/null \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL= \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

COPY repair-constraints.txt /opt/psem/repair-constraints.txt
COPY runtime-constraints.txt /opt/psem/runtime-constraints.txt
COPY prepare-runtime.py /opt/psem/prepare-runtime.py
COPY validate-runtime.py /opt/psem/validate-runtime.py
COPY --from=torchaudio-builder /opt/wheels/ /opt/psem/wheels/

RUN python /opt/psem/prepare-runtime.py repair-nvimgcodec \
    && python /opt/psem/prepare-runtime.py snapshot \
       --output /opt/psem/base-protected-inventory.json \
       --constraint-output /opt/psem/base-protected-constraints.txt \
    && python -m pip uninstall -y torchaudio \
    && python -m pip install --no-deps \
       --constraint /opt/psem/base-protected-constraints.txt \
       --constraint /opt/psem/repair-constraints.txt \
       --constraint /opt/psem/runtime-constraints.txt \
       /opt/psem/wheels/torchaudio-2.6.0a0+d883142-*.whl \
    && python -m pip install \
       --constraint /opt/psem/base-protected-constraints.txt \
       --constraint /opt/psem/repair-constraints.txt \
       --constraint /opt/psem/runtime-constraints.txt \
       --requirement /opt/psem/repair-constraints.txt \
    && git clone https://github.com/NVIDIA-NeMo/NeMo.git /opt/nemo \
    && git -C /opt/nemo checkout --detach 1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6 \
    && test "$(git -C /opt/nemo rev-parse HEAD)" = "1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6" \
    && python -m pip install \
       --constraint /opt/psem/base-protected-constraints.txt \
       --constraint /opt/psem/repair-constraints.txt \
       --constraint /opt/psem/runtime-constraints.txt \
       --editable "/opt/nemo[asr]" \
    && python -m pip install \
       --constraint /opt/psem/base-protected-constraints.txt \
       --constraint /opt/psem/repair-constraints.txt \
       --constraint /opt/psem/runtime-constraints.txt \
       --requirement /opt/psem/repair-constraints.txt \
    && python /opt/psem/prepare-runtime.py isolate-metadata \
       --receipt /opt/psem/metadata-isolation.json \
    && test -z "$(git -C /opt/nemo status --porcelain --untracked-files=all)" \
    && python -m pip check \
    && python /opt/psem/validate-runtime.py \
       --mode build \
       --receipt /opt/psem/build-validation.json \
    && rm -rf /root/.cache/pip /tmp/*

WORKDIR /workspace
