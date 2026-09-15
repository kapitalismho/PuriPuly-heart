#!/usr/bin/env bash
set -euo pipefail

readonly VENV=/opt/psem-streaming-student/rocm-7.2.1-pytorch-2.9.1
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
readonly STAGE="${1:?stage is required}"
readonly CONFIG="${2:?config path is required}"
RUN_ROOT="${3:?run root is required}"
if [[ "$RUN_ROOT" != /* || "$RUN_ROOT" == "/" ]]; then
    printf 'run root must be a non-root absolute WSL path: %q\n' "$RUN_ROOT" >&2
    exit 2
fi
readonly RUN_ROOT

export HSA_ENABLE_DXG_DETECTION=1
export ROCM_PATH=/opt/rocm-7.2.1
export PATH="$ROCM_PATH/bin:$VENV/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT"

exec "$VENV/bin/python" "$SCRIPT_DIR/gt_probe.py" "$STAGE" --config "$CONFIG" --run-root "$RUN_ROOT"
