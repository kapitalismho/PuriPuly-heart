#!/usr/bin/env python3
import argparse
import json
import platform
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Issue 164 isolated WSL PyTorch GPU environment backend check")
    parser.add_argument(
        "--backward",
        action="store_true",
        help="explicitly opt in to one bounded tensor autograd check; never creates a model or optimizer",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch did not expose the WSL ROCm GPU")

    started = time.perf_counter()
    device = torch.device("cuda:0")
    with torch.no_grad():
        left = torch.arange(4096, dtype=torch.float32, device=device)
        right = torch.ones(4096, dtype=torch.float32, device=device)
        result = left + right
        torch.cuda.synchronize(device)
        checksum = float(result.sum().cpu())

    report = {
        "check": "environment_backend_check",
        "mode": "backward_opt_in" if args.backward else "default_no_autograd",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "cuda_api_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "tensor_elements": result.numel(),
        "tensor_checksum": checksum,
        "elapsed_seconds": time.perf_counter() - started,
        "model_created": False,
        "optimizer_created": False,
    }

    if args.backward:
        source = torch.arange(256, dtype=torch.float32, device=device, requires_grad=True)
        loss = (source * source).sum()
        loss.backward()
        torch.cuda.synchronize(device)
        report["backward_called"] = True
        report["gradient_finite"] = bool(torch.isfinite(source.grad).all().cpu())
    else:
        report["backward_called"] = False

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
