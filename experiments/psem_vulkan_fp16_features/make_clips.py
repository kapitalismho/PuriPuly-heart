import wave
import hashlib
import pathlib
SRC = pathlib.Path("C:/Users/salee/.psem-corpus/ami/audio/ES2009d/ES2009d.Mix-Headset.wav")
OUT = pathlib.Path("experiments/psem_vulkan_fp16_features/clips")
CLIPS = {
    "clip_7s": (33405760, 33517760),
    "clip_40s": (32877760, 33517760),
    "clip_6s": (33405760, 33501760),
    "clip_8s": (33405760, 33533760),
    "clip_20s": (33405760, 33725760),
    "prefix_5p04s": (33405760, 33486400),
    "prefix_4s": (33405760, 33469760),
    "prefix_1040ms": (33405760, 33422400),
}
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = wave.open(str(SRC), "rb")
    assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000
    n = w.getnframes()
    print(f"src frames={n} dur_s={n/16000}")
    print(f"src sha256={sha256_file(SRC)}")
    for name, (s, e) in CLIPS.items():
        assert 0 <= s < e <= n, (name, s, e, n)
        w.setpos(s)
        frames = w.readframes(e - s)
        assert len(frames) == (e - s) * 2
        pcm = frames
        h = hashlib.sha256(pcm).hexdigest()
        out_wav = OUT / f"{name}.wav"
        ow = wave.open(str(out_wav), "wb")
        ow.setnchannels(1)
        ow.setsampwidth(2)
        ow.setframerate(16000)
        ow.writeframes(pcm)
        ow.close()
        print(f"{name} [{s},{e}) len={e-s} pcm_sha256={h} wav_sha256={sha256_file(out_wav)} file={out_wav}")
    w.close()
if __name__ == "__main__":
    main()
