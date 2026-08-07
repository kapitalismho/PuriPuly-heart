from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LS_EEND_ONNX_REPO = "https://huggingface.co/GradientDescent2718/LS-EEND-ONNX"
LS_EEND_ONNX_REVISION = "cc40a1e1242c148fbbc15c132e43b8ac15056e53"
LS_EEND_ONNX_LICENSE = "mit"

FS_EEND_REPO = "https://github.com/Audio-WestlakeU/FS-EEND"
FS_EEND_REVISION = "adcdde1327bc731cc4e718aa009b8d78317388e5"
FS_EEND_LICENSE = "MIT"

THIRD_PARTY_SPEAKER_REPO = "https://github.com/modelscope/3D-Speaker"
THIRD_PARTY_SPEAKER_REVISION = "065629c313eaf1a01c65c640c46d77e61e9607b4"

ERES_STANDARD_MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
ERES_STANDARD_REVISION = "1cf80d41fb3435bd3d8df185b5c423333b2db42a"
ERES_STANDARD_FILE = "pretrained_eres2netv2.ckpt"
ERES_STANDARD_SHA256 = "0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c"
ERES_STANDARD_SIZE = 71768231
ERES_STANDARD_LICENSE = "Apache-2.0"

ERES_W24_MODEL_ID = "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
ERES_W24_REVISION = "d41a54156a6216b4c7611447be0548e4b0afb1ba"
ERES_W24_FILE = "pretrained_eres2netv2w24s4ep4.ckpt"
ERES_W24_SHA256 = "740bb6584a99ee4cf910101536acba38c15a8017ea6a3a2813ec668fb62981f1"
ERES_W24_SIZE = 214688240
ERES_W24_LICENSE = "Apache-2.0"


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    artifact_id: str
    kind: str
    source_url: str
    revision: str
    license: str
    file_name: str
    sha256: str
    size_bytes: int
    sidecar: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "source_url": self.source_url,
            "revision": self.revision,
            "license": self.license,
            "file_name": self.file_name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if self.sidecar is not None:
            data["sidecar"] = self.sidecar
        return data


LS_EEND_VARIANTS = {
    "L-AMI": {
        "dir": "AMI",
        "onnx": "ls_eend_ami_step.onnx",
        "sidecar": "ls_eend_ami_step.json",
        "onnx_sha256": "5a2b813ffe41170e40d0fc08a6eb1699e579e377af30c7962d07885608a6aa77",
        "sidecar_sha256": "47f29718254995ec017636d5ff31fef8b20bf47dca30d883edcb91e022dc3353",
        "onnx_size": 44480761,
        "sidecar_size": 1329,
    },
    "L-CALLHOME": {
        "dir": "CALLHOME",
        "onnx": "ls_eend_callhome_step.onnx",
        "sidecar": "ls_eend_callhome_step.json",
        "onnx_sha256": "b79b1b1cb2a070bfb92543d90af5530681af0e45da8bf5771e515e9c644b6604",
        "sidecar_sha256": "049084141fb3d7694e4bbdc257024761d6f6e64b8782e47d7426e3f35009dffa",
        "onnx_size": 44483833,
        "sidecar_size": 1329,
    },
    "L-DIHARD-II": {
        "dir": "DIHARD II",
        "onnx": "ls_eend_dih2_step.onnx",
        "sidecar": "ls_eend_dih2_step.json",
        "onnx_sha256": "5df89a22ba87989a01217e51d674cc547877ce5b7100dce920ab63adc3258302",
        "sidecar_sha256": "ecb42ca07888a297b3e0ae277b19e1a1412e1fa68309233f9e6196b06128f0a9",
        "onnx_size": 44486905,
        "sidecar_size": 1335,
    },
    "L-DIHARD-III": {
        "dir": "DIHARD III",
        "onnx": "ls_eend_dih3_step.onnx",
        "sidecar": "ls_eend_dih3_step.json",
        "onnx_sha256": "587ad263b46aaa5d4fc7fb9e0524d1990455f7286c3a47b2371d08df8b5671c8",
        "sidecar_sha256": "ecb42ca07888a297b3e0ae277b19e1a1412e1fa68309233f9e6196b06128f0a9",
        "onnx_size": 44486905,
        "sidecar_size": 1335,
    },
}


def fs_eend_checkpoint_artifacts() -> list[ModelArtifact]:
    base = "https://drive.google.com/file/d/"
    drive_ids = {
        "ami.ckpt": "1Zbc-8fXr_9kydjYS5SAeIaYDr6O1Ik74",
        "ch.ckpt": "1W8nYAB6YoEKMM5KZX-apVADvHaYc2Fre",
        "dih2.ckpt": "1vu7VSTnrNsooz5DzaodmctjdwblfB3wv",
        "dih3.ckpt": "115iaEG1OZwXa9tSyScXGtWeOk9JLfpER",
    }
    hashes = {
        "ami.ckpt": "5b1df8f050faabda0432a650567c0e4db9826e54ba7d709677bfe3e62db5a73e",
        "ch.ckpt": "eab0b7183075044e665dcdbd88d82c3e49094cd87c5ac862715fe6ca880e6afd",
        "dih2.ckpt": "2d6de53d69d99a3ee3737425bff4aa7490b7981f527994ca552fb6b2adf25b0f",
        "dih3.ckpt": "da62f7e16f422f1c6518bd781e2e54c318b16881fc168f60a20db7870372818c",
    }
    sizes = {
        "ami.ckpt": 49977074,
        "ch.ckpt": 49974764,
        "dih2.ckpt": 49980603,
        "dih3.ckpt": 49980603,
    }
    artifacts = []
    for file_name, drive_id in drive_ids.items():
        artifacts.append(
            ModelArtifact(
                artifact_id=f"FS-EEND:{file_name}",
                kind="pytorch_checkpoint",
                source_url=base + drive_id,
                revision=FS_EEND_REVISION,
                license=FS_EEND_LICENSE,
                file_name=file_name,
                sha256=hashes[file_name],
                size_bytes=sizes[file_name],
            )
        )
    return artifacts


def ls_eend_onnx_artifacts() -> list[ModelArtifact]:
    artifacts = []
    for variant, info in LS_EEND_VARIANTS.items():
        onnx_name = info["onnx"]
        sidecar_name = info["sidecar"]
        artifacts.append(
            ModelArtifact(
                artifact_id=f"{variant}:{onnx_name}",
                kind="onnx_step_model",
                source_url=f"{LS_EEND_ONNX_REPO}/tree/main/{info['dir']}",
                revision=LS_EEND_ONNX_REVISION,
                license=LS_EEND_ONNX_LICENSE,
                file_name=onnx_name,
                sha256=info["onnx_sha256"],
                size_bytes=info["onnx_size"],
            )
        )
        artifacts.append(
            ModelArtifact(
                artifact_id=f"{variant}:{sidecar_name}",
                kind="onnx_sidecar",
                source_url=f"{LS_EEND_ONNX_REPO}/tree/main/{info['dir']}",
                revision=LS_EEND_ONNX_REVISION,
                license=LS_EEND_ONNX_LICENSE,
                file_name=sidecar_name,
                sha256=info["sidecar_sha256"],
                size_bytes=info["sidecar_size"],
            )
        )
    return artifacts


def eres_artifacts() -> list[ModelArtifact]:
    standard_sidecar = {
        "model_id": ERES_STANDARD_MODEL_ID,
        "revision": ERES_STANDARD_REVISION,
        "embedding_size": 192,
        "baseWidth": 26,
        "scale": 2,
        "expansion": 2,
        "feat_dim": 80,
    }
    w24_sidecar = {
        "model_id": ERES_W24_MODEL_ID,
        "revision": ERES_W24_REVISION,
        "embedding_size": 192,
        "baseWidth": 24,
        "scale": 4,
        "expansion": 4,
        "feat_dim": 80,
    }
    return [
        ModelArtifact(
            artifact_id=f"E-standard:{ERES_STANDARD_FILE}",
            kind="pytorch_checkpoint",
            source_url=f"https://modelscope.cn/models/{ERES_STANDARD_MODEL_ID}",
            revision=ERES_STANDARD_REVISION,
            license=ERES_STANDARD_LICENSE,
            file_name=ERES_STANDARD_FILE,
            sha256=ERES_STANDARD_SHA256,
            size_bytes=ERES_STANDARD_SIZE,
            sidecar=standard_sidecar,
        ),
        ModelArtifact(
            artifact_id=f"E-w24s4ep4:{ERES_W24_FILE}",
            kind="pytorch_checkpoint",
            source_url=f"https://modelscope.cn/models/{ERES_W24_MODEL_ID}",
            revision=ERES_W24_REVISION,
            license=ERES_W24_LICENSE,
            file_name=ERES_W24_FILE,
            sha256=ERES_W24_SHA256,
            size_bytes=ERES_W24_SIZE,
            sidecar=w24_sidecar,
        ),
    ]


def all_artifacts() -> list[ModelArtifact]:
    return ls_eend_onnx_artifacts() + fs_eend_checkpoint_artifacts() + eres_artifacts()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact_file(artifact: ModelArtifact, path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != artifact.size_bytes:
        return False, f"size mismatch ({actual_size} != {artifact.size_bytes})"
    actual_hash = sha256_file(path)
    if actual_hash.lower() != artifact.sha256.lower():
        return False, f"sha256 mismatch ({actual_hash} != {artifact.sha256})"
    return True, "ok"
