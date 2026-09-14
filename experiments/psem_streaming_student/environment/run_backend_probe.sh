#!/usr/bin/env bash
set -euo pipefail

readonly VENV=/opt/psem-streaming-student/rocm-7.2.1-pytorch-2.9.1
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export HSA_ENABLE_DXG_DETECTION=1
export ROCM_PATH=/opt/rocm-7.2.1
export PATH="$ROCM_PATH/bin:$VENV/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$VENV/bin/python" "$SCRIPT_DIR/gpu_backend_probe.py" "$@"
