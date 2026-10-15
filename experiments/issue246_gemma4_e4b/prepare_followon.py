from __future__ import annotations

import subprocess
import urllib.request
import zipfile

from prepare import EXTERNAL, HERE, RELEASES, digest, sanitize, save


def main():
    release = "b11630"
    record = {"release": release, "commit": RELEASES[release]["commit"], "reason": "bounded foreign mmap range / pinned-slot follow-on; not a floating-latest search", "runtimes": {}}
    for backend in ("cpu", "vulkan"):
        size, expected = RELEASES[release][backend]
        filename = f"llama-{release}-bin-win-{backend}-x64.zip"
        archive = EXTERNAL / filename
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/{filename}"
        if not archive.exists():
            urllib.request.urlretrieve(url, archive)
        actual = digest(archive)
        assert archive.stat().st_size == size and actual == expected
        runtime = EXTERNAL / release / backend
        runtime.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(runtime)
        binary = next(runtime.rglob("llama-server.exe"))
        for option, suffix in (("--help", "help"), ("--version", "version"), ("--list-devices", "devices")):
            result = subprocess.run([str(binary), option], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
            (HERE / "results" / f"{release}-{backend}-{suffix}.txt").write_text(sanitize(result.stdout + result.stderr), encoding="utf-8")
        record["runtimes"][backend] = {"url": url, "archive_size": size, "archive_sha256": actual, "binary_sha256": digest(binary), "files": [{"name": str(p.relative_to(runtime)), "size": p.stat().st_size, "sha256": digest(p)} for p in sorted(runtime.rglob("*")) if p.is_file()]}
    save("followon-identity.json", record)


if __name__ == "__main__":
    main()
