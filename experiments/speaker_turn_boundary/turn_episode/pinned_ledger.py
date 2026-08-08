"""Frozen Section 1.1 input/code/config ledger (bundle ``phase_2_review_bundle.md``
Section 1.1, finding P2-013).

The ledger is the frozen Phase-2-review-time file list; every artifact records the
live SHA-256 of each entry plus the delta against the frozen review-time values, so
any change to a pinned input is visible in the artifact provenance instead of
silently invalidating Phase 2 outputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _EXPERIMENT_DIR / "results" / "turn_episode_v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]

# entry name -> (relative path, frozen review-time sha256 from bundle Section 1.1)
LEDGER_ENTRIES: tuple[tuple[str, Path, str], ...] = (
    (
        "turn_episode/contracts.py",
        _EXPERIMENT_DIR / "turn_episode" / "contracts.py",
        "b207d3f8b9720df5dd228aa8bd8b479c54622abb905a9ca04f580820a6fc3c03",
    ),
    (
        "turn_episode/schemas.py",
        _EXPERIMENT_DIR / "turn_episode" / "schemas.py",
        "8c449b2ed07fba11bb1e45f01cad6b22fe1c98eb8006a2600bee90170f45f2f9",
    ),
    (
        "turn_episode/build_coverage_inventory.py",
        _EXPERIMENT_DIR / "turn_episode" / "build_coverage_inventory.py",
        "dd360c9e60a5838feaea17e4b335d1fc93cdbd6df4f426077a2eedbd30e1a1e7",
    ),
    (
        "turn_episode/materialize_ami_additions.py",
        _EXPERIMENT_DIR / "turn_episode" / "materialize_ami_additions.py",
        "bf431bb5b22ec79032ee6fbe876d5ab330893521536481e590f9b054932ccdc7",
    ),
    (
        "vad_baseline.py",
        _EXPERIMENT_DIR / "vad_baseline.py",
        "7a3965fdb01eb7391dde985e5c498162d80b4e5ab565205626d684a66d8ff627",
    ),
    (
        "events.py",
        _EXPERIMENT_DIR / "events.py",
        "2193bda0f06ff9e3d4171402c9ce2296ed273f10994de35332ca070d212b347a",
    ),
    (
        "config.py",
        _EXPERIMENT_DIR / "config.py",
        "f4eb24e6c81ebcb0bdd71b6c0c9098595ae4bdddf53e05df6bd8eea925d146a6",
    ),
    (
        "ground_truth.py",
        _EXPERIMENT_DIR / "ground_truth.py",
        "34d2236595c4fb3e105b1aa5da8b4fa05e513f33979ca63c8c6903299d0f820d",
    ),
    (
        "corpus/phase2_schemas.py",
        _EXPERIMENT_DIR / "corpus" / "phase2_schemas.py",
        "7a6b4b0c9033b5ebdc97db552943c522a5218f5166e039db4b37f6744861dcf2",
    ),
    (
        "src/puripuly_heart/core/vad/silero.py",
        _REPO_ROOT / "src" / "puripuly_heart" / "core" / "vad" / "silero.py",
        "43079df5bc36ecb924b1aec7991cff2a16c04ab126bb54907c4b2a570e2cd109",
    ),
    (
        "corpus/ami.py",
        _EXPERIMENT_DIR / "corpus" / "ami.py",
        "e77171299cb7d358f2a3f029e78386d91492e94e49ca683bac90ac5173c4d0dc",
    ),
    (
        "corpus/alimeeting.py",
        _EXPERIMENT_DIR / "corpus" / "alimeeting.py",
        "f69b733c703e0060245a2d9229258caddc602238c18c7c242570713670ea929a",
    ),
    (
        "corpus/librispeech.py",
        _EXPERIMENT_DIR / "corpus" / "librispeech.py",
        "05b77dc735f572da37da0c9ebe595415f5acf4b5c40bd22e3f792dd31402d1f4",
    ),
    (
        "corpus/external.py",
        _EXPERIMENT_DIR / "corpus" / "external.py",
        "262d40233371905e6a6fac2efee2da1412c6d6df44d91ea4eb8fd2220ba21b56",
    ),
    (
        "corpus/mixing.py",
        _EXPERIMENT_DIR / "corpus" / "mixing.py",
        "ffd707787003051f5e95e9dabb70afb4eb675b970461e75cd6c062d7f121abc6",
    ),
    (
        "corpus/puripuly_like.py",
        _EXPERIMENT_DIR / "corpus" / "puripuly_like.py",
        "5f5df4ce0879bd5f7ffa35d571e700852a490803bd966c9f7a90c0f4ee848642",
    ),
    (
        "corpus/validation.py",
        _EXPERIMENT_DIR / "corpus" / "validation.py",
        "66d863a0ada96fcc765097dd574ea4b2edfc5eed6ed873aee13df34d65ff964d",
    ),
    (
        "data/manifests/ls_dev.json",
        _EXPERIMENT_DIR / "data" / "manifests" / "ls_dev.json",
        "14347cdbdb2eff4cc73489f1b59d6755723d9098089dad66ae222984e90370dd",
    ),
    (
        "data/manifests/ls_held_out_clean.json",
        _EXPERIMENT_DIR / "data" / "manifests" / "ls_held_out_clean.json",
        "c0aabc5ad8c3f00ec53d45f3b372b8ebca7ca9237720a1bb7a70b8de7dda2581",
    ),
    (
        "data/manifests/ls_held_out_other.json",
        _EXPERIMENT_DIR / "data" / "manifests" / "ls_held_out_other.json",
        "f0d169394a9fdee9e708bc9cad46c0547946bf967799fa4e2e1a398ddb984079",
    ),
    (
        "data/manifests/mixed_dev_pool.json",
        _EXPERIMENT_DIR / "data" / "manifests" / "mixed_dev_pool.json",
        "1221176c92f50a2b096e4cd64d5da0168527918e3fba539273c614eabf07a398",
    ),
    (
        "proposal_contract.json",
        _RESULTS_DIR / "proposal_contract.json",
        "0448edd933fd1d9d0a0b4d5f9f2631cb0f630c892fc4d46e1a3ec9740e80b7fb",
    ),
    (
        "fusion_contract.json",
        _RESULTS_DIR / "fusion_contract.json",
        "bfda0c3c0ea7b6613ded79e9639692a33449dcf34202b1f2a5e7ec14c45f9873",
    ),
    (
        "coverage_inventory.json",
        _RESULTS_DIR / "coverage_inventory.json",
        "02a6a118fc90c0d747e9548f07003177b3fc703f33d408d5338427cb6163dd46",
    ),
    (
        "coverage_inventory_details.jsonl",
        _RESULTS_DIR / "coverage_inventory_details.jsonl",
        "15b2e4f0efa270985c3bbc6d848ee9ed25496089268e561bff921c5c1be3ef8c",
    ),
    (
        "ami_materialization_manifest.json",
        _RESULTS_DIR / "ami_materialization_manifest.json",
        "06fe15fff87bb78218df2c086bd711590378f8741164909b59704f56841ab6c9",
    ),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _silero_model_path() -> Path:
    from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path

    return Path(str(bundled_silero_vad_onnx_path()))


def pinned_ledger() -> dict[str, str]:
    """Live SHA-256 over the frozen Section 1.1 file list."""
    ledger: dict[str, str] = {}
    for name, path, _frozen in LEDGER_ENTRIES:
        ledger[name] = _sha256_bytes(path.read_bytes()) if path.is_file() else "missing"
    model_path = _silero_model_path()
    ledger["silero_vad.onnx"] = (
        _sha256_bytes(model_path.read_bytes()) if model_path.is_file() else "missing"
    )
    return ledger


def pinned_ledger_delta() -> dict[str, dict[str, str]]:
    """Entries whose live hash differs from the frozen review-time value."""
    live = pinned_ledger()
    delta: dict[str, dict[str, str]] = {}
    for name, _path, frozen in LEDGER_ENTRIES:
        if live[name] != frozen:
            delta[name] = {"frozen": frozen, "live": live[name]}
    if "silero_vad.onnx" in live and live["silero_vad.onnx"] != (
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
    ):
        delta["silero_vad.onnx"] = {
            "frozen": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
            "live": live["silero_vad.onnx"],
        }
    return delta


def ledger_verification() -> dict[str, Any]:
    return {
        "pinned_ledger": pinned_ledger(),
        "pinned_ledger_delta": pinned_ledger_delta(),
    }
