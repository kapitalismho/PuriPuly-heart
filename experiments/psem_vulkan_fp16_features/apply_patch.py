import pathlib
import shutil
import subprocess
import sys
import zipfile
SRC_ZIP = pathlib.Path("C:/tmp/psem-vulkan-fp16-source/d42c3bb.zip")
HERE = pathlib.Path(__file__).resolve().parent
PATCHES = [HERE / "patch" / "sortformer_h.patch", HERE / "patch" / "model_cpp.patch", HERE / "patch" / "ggml_vulkan.patch"]
def sh(cmd, cwd=None):
    print(" ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=str(cwd) if cwd else None)
    print(r.stdout[-3000:])
    print(r.stderr[-3000:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    return r
def main():
    dest = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("C:/tmp/psem-vulkan-fp16-repro")
    build = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else dest / "build"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(SRC_ZIP) as z:
        z.extractall(dest)
    roots = list(dest.glob("transcribe.cpp-*"))
    assert len(roots) == 1
    src = roots[0]
    for p in PATCHES:
        sh(["patch", "-p1", "-d", str(src), "-i", str(p)])
    sh(["cmake", "-G", "Visual Studio 17 2022", "-A", "x64", "-DTRANSCRIBE_VULKAN=ON", "-DCMAKE_CXX_FLAGS=/utf-8", "-DCMAKE_C_FLAGS=/utf-8", "-S", str(src), "-B", str(build)])
    sh(["cmake", "--build", str(build), "--config", "Release", "-j", "8"])
    print(str(src))
    print(str(build))
if __name__ == "__main__":
    main()
