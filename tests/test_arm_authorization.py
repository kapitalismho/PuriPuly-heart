from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.psem_state_corrected_adaptation_gate import arm_runtime
from experiments.psem_state_corrected_adaptation_gate.arm_runtime import (
    ARM_R_H_SC,
    ARM_R_T2_SC,
    ARM_R_TA_SC,
    ArmRunConfig,
    AuthorizationError,
)


def _h(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _config(tmp, arm=ARM_R_H_SC, seed=7301):
    return ArmRunConfig(
        arm=arm,
        seed=seed,
        root=Path(tmp) / "arms",
        input_hash=_h("input"),
        checkpoint_hash=_h("checkpoint"),
        partition_hash=_h("partition"),
        weights_hash=_h("weights"),
        code_hash=_h("code"),
    )


def _p0(store, config, verdict="PASS"):
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    (store / "p0_pass.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "input_hash": config.input_hash,
                "checkpoint_hash": config.checkpoint_hash,
                "partition_hash": config.partition_hash,
            }
        ),
        encoding="utf-8",
    )


def _write_prior(root, arm, seed, binding=None, claim=None, artifact_bytes=b"metrics"):
    run_dir = Path(root) / arm / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "training_metrics.json"
    artifact.write_bytes(artifact_bytes)
    manifest = {"arm": arm, "seed": seed}
    if binding is not None:
        manifest["binding"] = dict(binding)
    if claim is not None:
        manifest.update(dict(claim))
    out = arm_runtime.write_final_manifest(run_dir, manifest, [artifact])
    return out, arm_runtime.sha256_file(out)


def _gate1(store, config, digest):
    (Path(store) / "gate1.json").write_text(
        json.dumps(
            {"decision": "OPEN-T2", "h_candidate_hash": digest, "input_hash": config.input_hash}
        ),
        encoding="utf-8",
    )


def _gate2(store, config, digest):
    (Path(store) / "gate2.json").write_text(
        json.dumps(
            {"decision": "OPEN-TA", "t2_candidate_hash": digest, "input_hash": config.input_hash}
        ),
        encoding="utf-8",
    )


def _confirm(store, config, digest):
    (Path(store) / f"confirmation_{config.arm}.json").write_text(
        json.dumps(
            {
                "arm": config.arm,
                "seed": 7302,
                "candidate_hash": digest,
                "input_hash": config.input_hash,
            }
        ),
        encoding="utf-8",
    )


class AuthorizationTest(unittest.TestCase):
    def test_h_screen_passes_with_p0(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            _p0(Path(tmp) / "store", config)
            calls: list = []
            out = arm_runtime.authorize_and_run(
                config, Path(tmp) / "store", lambda cfg: calls.append(cfg) or "ok"
            )
            self.assertEqual(calls, [config])
            self.assertTrue(out["authorization"]["authorized"])

    def test_executor_never_runs_when_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp, arm=ARM_R_T2_SC)
            _p0(Path(tmp) / "store", config)
            calls: list = []
            with self.assertRaises(AuthorizationError):
                arm_runtime.authorize_and_run(
                    config, Path(tmp) / "store", lambda cfg: calls.append(cfg)
                )
            self.assertEqual(calls, [])

    def test_missing_and_mismatched_receipts_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(config, Path(tmp) / "store")
            _p0(Path(tmp) / "store", config, verdict="FAIL")
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(config, Path(tmp) / "store")
            other = _config(tmp)
            _p0(Path(tmp) / "store", other, verdict="PASS")
            tampered = ArmRunConfig(
                arm=other.arm,
                seed=other.seed,
                root=other.root,
                input_hash=_h("different-input"),
                checkpoint_hash=other.checkpoint_hash,
                partition_hash=other.partition_hash,
                weights_hash=other.weights_hash,
                code_hash=other.code_hash,
            )
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(tampered, Path(tmp) / "store")

    def test_t2_requires_gate1_and_ta_requires_gate2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arms"
            h_screen = _config(tmp, arm=ARM_R_H_SC, seed=7301)
            _, h_digest = _write_prior(root, ARM_R_H_SC, 7301, binding=h_screen.binding)
            _, t2_digest = _write_prior(root, ARM_R_T2_SC, 7302)
            t2 = _config(tmp, arm=ARM_R_T2_SC)
            _p0(Path(tmp) / "store", t2)
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(t2, Path(tmp) / "store")
            _gate1(Path(tmp) / "store", t2, h_digest)
            arm_runtime.check_authorization(t2, Path(tmp) / "store")
            ta = _config(tmp, arm=ARM_R_TA_SC)
            _p0(Path(tmp) / "store", ta)
            _gate1(Path(tmp) / "store", ta, h_digest)
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(ta, Path(tmp) / "store")
            _gate2(Path(tmp) / "store", ta, t2_digest)
            arm_runtime.check_authorization(ta, Path(tmp) / "store")

    def test_seed_7302_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arms"
            screen = _config(tmp, seed=7301)
            _, digest = _write_prior(root, ARM_R_H_SC, 7301, binding=screen.binding)
            config = _config(tmp, seed=7302)
            _p0(Path(tmp) / "store", config)
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(config, Path(tmp) / "store")
            _confirm(Path(tmp) / "store", config, digest)
            arm_runtime.check_authorization(config, Path(tmp) / "store")

    def test_eval_impossible(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AuthorizationError):
                ArmRunConfig(
                    arm="EVAL",
                    seed=7301,
                    root=Path(tmp),
                    input_hash=_h("input"),
                    checkpoint_hash=_h("checkpoint"),
                    partition_hash=_h("partition"),
                    weights_hash=_h("weights"),
                    code_hash=_h("code"),
                )
            with self.assertRaises(AuthorizationError):
                arm_runtime.canonical_arm("EVAL")
            with self.assertRaises(AuthorizationError):
                ArmRunConfig(
                    arm=ARM_R_H_SC,
                    seed=9999,
                    root=Path(tmp),
                    input_hash=_h("input"),
                    checkpoint_hash=_h("checkpoint"),
                    partition_hash=_h("partition"),
                    weights_hash=_h("weights"),
                    code_hash=_h("code"),
                )

    def test_config_load_and_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "arm": ARM_R_H_SC,
                "seed": 7301,
                "root": str(Path(tmp) / "arms"),
                "input_hash": _h("input"),
                "checkpoint_hash": _h("checkpoint"),
                "partition_hash": _h("partition"),
                "weights_hash": _h("weights"),
                "code_hash": _h("code"),
            }
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = arm_runtime.load_config(path)
            self.assertEqual(config.run_dir(), Path(tmp) / "arms" / ARM_R_H_SC / "7301")


class PriorArtifactTest(unittest.TestCase):
    def _chain(self, tmp):
        root = Path(tmp) / "arms"
        h_screen = _config(tmp, arm=ARM_R_H_SC, seed=7301)
        _, h_digest = _write_prior(root, ARM_R_H_SC, 7301, binding=h_screen.binding)
        t2_screen = _config(tmp, arm=ARM_R_T2_SC, seed=7301)
        _, t2_screen_digest = _write_prior(root, ARM_R_T2_SC, 7301)
        t2_confirm = _config(tmp, arm=ARM_R_T2_SC, seed=7302)
        _, t2_digest = _write_prior(root, ARM_R_T2_SC, 7302)
        ta_screen = _config(tmp, arm=ARM_R_TA_SC, seed=7301)
        _, ta_digest = _write_prior(root, ARM_R_TA_SC, 7301, binding=ta_screen.binding)
        return {
            "h": h_digest,
            "t2_screen": t2_screen_digest,
            "t2": t2_digest,
            "ta": ta_digest,
        }

    def test_valid_exact_chains_authorize(self):
        with tempfile.TemporaryDirectory() as tmp:
            digests = self._chain(tmp)
            store = Path(tmp) / "store"
            t2 = _config(tmp, arm=ARM_R_T2_SC)
            _p0(store, t2)
            _gate1(store, t2, digests["h"])
            h_confirm = _config(tmp, arm=ARM_R_H_SC, seed=7302)
            _p0(store, h_confirm)
            _confirm(store, h_confirm, digests["h"])
            self.assertTrue(arm_runtime.check_authorization(h_confirm, store)["authorized"])
            t2_confirm = _config(tmp, arm=ARM_R_T2_SC, seed=7302)
            _p0(store, t2_confirm)
            _gate1(store, t2_confirm, digests["h"])
            _confirm(store, t2_confirm, digests["t2_screen"])
            calls: list = []
            out = arm_runtime.authorize_and_run(
                t2_confirm, store, lambda cfg: calls.append(cfg) or "ok"
            )
            self.assertEqual(calls, [t2_confirm])
            self.assertEqual(out["result"], "ok")
            ta = _config(tmp, arm=ARM_R_TA_SC)
            _p0(store, ta)
            _gate1(store, ta, digests["h"])
            _gate2(store, ta, digests["t2"])
            self.assertTrue(arm_runtime.check_authorization(ta, store)["authorized"])
            ta_confirm = _config(tmp, arm=ARM_R_TA_SC, seed=7302)
            _p0(store, ta_confirm)
            _gate1(store, ta_confirm, digests["h"])
            _gate2(store, ta_confirm, digests["t2"])
            _confirm(store, ta_confirm, digests["ta"])
            self.assertTrue(arm_runtime.check_authorization(ta_confirm, store)["authorized"])

    def test_missing_prior_manifest_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            t2 = _config(tmp, arm=ARM_R_T2_SC)
            _p0(Path(tmp) / "store", t2)
            _gate1(Path(tmp) / "store", t2, _h("absent-candidate"))
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(t2, Path(tmp) / "store")

    def test_candidate_hash_mismatch_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arms"
            screen = _config(tmp, arm=ARM_R_H_SC, seed=7301)
            _write_prior(root, ARM_R_H_SC, 7301, binding=screen.binding)
            t2 = _config(tmp, arm=ARM_R_T2_SC)
            _p0(Path(tmp) / "store", t2)
            _gate1(Path(tmp) / "store", t2, _h("other-candidate"))
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(t2, Path(tmp) / "store")

    def test_wrong_arm_seed_input_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arms"
            screen = _config(tmp, arm=ARM_R_H_SC, seed=7301)
            _, digest = _write_prior(
                root, ARM_R_H_SC, 7301, binding=screen.binding, claim={"arm": ARM_R_T2_SC}
            )
            t2 = _config(tmp, arm=ARM_R_T2_SC)
            _p0(Path(tmp) / "store", t2)
            _gate1(Path(tmp) / "store", t2, digest)
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(t2, Path(tmp) / "store")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arms"
            screen = _config(tmp, arm=ARM_R_H_SC, seed=7301)
            _, digest = _write_prior(
                root, ARM_R_H_SC, 7301, binding=screen.binding, claim={"seed": 7302}
            )
            h_confirm = _config(tmp, arm=ARM_R_H_SC, seed=7302)
            _p0(Path(tmp) / "store", h_confirm)
            _confirm(Path(tmp) / "store", h_confirm, digest)
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(h_confirm, Path(tmp) / "store")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arms"
            screen = _config(tmp, arm=ARM_R_H_SC, seed=7301)
            foreign = dict(screen.binding)
            foreign["input_hash"] = _h("foreign-input")
            _, digest = _write_prior(root, ARM_R_H_SC, 7301, binding=foreign)
            h_confirm = _config(tmp, arm=ARM_R_H_SC, seed=7302)
            _p0(Path(tmp) / "store", h_confirm)
            _confirm(Path(tmp) / "store", h_confirm, digest)
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(h_confirm, Path(tmp) / "store")

    def test_tampered_prior_artifact_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arms"
            screen = _config(tmp, arm=ARM_R_H_SC, seed=7301)
            out, _ = _write_prior(root, ARM_R_H_SC, 7301, binding=screen.binding)
            digest = arm_runtime.sha256_file(out)
            artifact = root / ARM_R_H_SC / "7301" / "training_metrics.json"
            artifact.write_bytes(b"tampered-metrics")
            t2 = _config(tmp, arm=ARM_R_T2_SC)
            _p0(Path(tmp) / "store", t2)
            _gate1(Path(tmp) / "store", t2, digest)
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(t2, Path(tmp) / "store")

    def test_path_escape_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arms"
            run_dir = root / ARM_R_H_SC / "7301"
            run_dir.mkdir(parents=True, exist_ok=True)
            outside = Path(tmp) / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            manifest = {
                "arm": ARM_R_H_SC,
                "seed": 7301,
                "binding": _config(tmp, arm=ARM_R_H_SC, seed=7301).binding,
            }
            out = arm_runtime.write_final_manifest(run_dir, manifest, [outside])
            digest = arm_runtime.sha256_file(out)
            t2 = _config(tmp, arm=ARM_R_T2_SC)
            _p0(Path(tmp) / "store", t2)
            _gate1(Path(tmp) / "store", t2, digest)
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(t2, Path(tmp) / "store")

    def test_gate2_requires_confirmed_t2_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arms"
            h_screen = _config(tmp, arm=ARM_R_H_SC, seed=7301)
            _, h_digest = _write_prior(root, ARM_R_H_SC, 7301, binding=h_screen.binding)
            _, t2_screen_digest = _write_prior(root, ARM_R_T2_SC, 7301)
            ta = _config(tmp, arm=ARM_R_TA_SC)
            _p0(Path(tmp) / "store", ta)
            _gate1(Path(tmp) / "store", ta, h_digest)
            _gate2(Path(tmp) / "store", ta, t2_screen_digest)
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(ta, Path(tmp) / "store")

    def test_confirmation_without_candidate_hash_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arms"
            screen = _config(tmp, arm=ARM_R_H_SC, seed=7301)
            _write_prior(root, ARM_R_H_SC, 7301, binding=screen.binding)
            config = _config(tmp, arm=ARM_R_H_SC, seed=7302)
            store = Path(tmp) / "store"
            _p0(store, config)
            (store / f"confirmation_{config.arm}.json").write_text(
                json.dumps(
                    {"arm": config.arm, "seed": 7302, "input_hash": config.input_hash}
                ),
                encoding="utf-8",
            )
            with self.assertRaises(AuthorizationError):
                arm_runtime.check_authorization(config, store)


if __name__ == "__main__":
    unittest.main()
