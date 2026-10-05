import os

_runtime_root = os.environ.get("PURIPULY_HEART_NATIVE_RUNTIME_ROOT")
if _runtime_root is not None:
    try:
        from _puripuly_native_runtime import install

        install(_runtime_root)
    except Exception as exc:
        raise SystemExit(f"native Python runtime activation failed: {exc}") from exc
