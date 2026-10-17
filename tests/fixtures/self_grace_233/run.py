from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

BASELINE = "75c4485a6787d504f5335d1602973f63b4d0c234"
BASELINE_OWNER_SHA256 = "e29fb972bd8174d7f8fbe8d91a42b4da519f2912e5b58e6c97183ae90430a007"
CORRECTNESS_OWNER_SHA256 = "ec77ef5e91ddde932de59ed8a036e101da72373042b000b516ce831159e58e2f"
CORRECTNESS_ORIGIN = "ed92da54c32bcac4add53be4f53a119308fea361"
OWNER_PATH = "src/puripuly_heart/core/orchestrator/self_translation_channel.py"
FIXTURE_PATH = "tests/fixtures/self_grace_233"
ARCHIVE_PATHS = ("src", "tests", "pyproject.toml", "prompts")
WRAPPER = (
    "from tests.fixtures.self_grace_233.experiment import test_grace_233_controlled_comparison\n"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def portable_command(arguments, checkout: Path) -> list[str]:
    report_dir = arguments.report_dir.resolve()
    destination = (
        report_dir.relative_to(checkout).as_posix()
        if report_dir.is_relative_to(checkout)
        else "<fresh-report-directory>"
    )
    return [
        "<checkout>/.venv/Scripts/python.exe",
        "-B",
        f"{FIXTURE_PATH}/run.py",
        "--owner-source",
        arguments.owner_source,
        "--case",
        arguments.case,
        "--report-dir",
        destination,
    ]


def execute_isolated(arguments) -> int:
    root = arguments._isolated_root.resolve()
    sys.path[:0] = [str(root / "src"), str(root)]
    os.environ["SELF_GRACE_233_ISOLATED_ROOT"] = str(root)
    os.environ["SELF_GRACE_233_REPORT_DIR"] = str(arguments.report_dir)
    os.environ["SELF_GRACE_233_OWNER_SOURCE"] = arguments.owner_source
    import pytest

    from tests.fixtures.self_grace_233.experiment import CASES, environment

    class Receipt:
        collected = 0
        outcomes = Counter()

        def pytest_collection_finish(self, session):
            self.collected = len(session.items)

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.failed or report.skipped:
                self.outcomes[report.outcome] += 1

    receipt = Receipt()
    exit_code = pytest.main(
        [
            str(root / FIXTURE_PATH / "test_historical_self_grace_233.py"),
            "-k",
            arguments.case,
            "--tb=short",
            "-q",
            "--color=no",
            "-p",
            "no:cacheprovider",
            "-p",
            "pytest_asyncio.plugin",
        ],
        plugins=[receipt],
    )
    imported_sources = {}
    for name, module in sorted(sys.modules.items()):
        if (
            name == "puripuly_heart"
            or name.startswith(("puripuly_heart.", "tests."))
            or name == "tests"
        ):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(root):
                    raise RuntimeError(
                        f"Historical import escaped the archived source tree: {name}"
                    )
                imported_sources[name] = path.relative_to(root).as_posix()
    paths = sorted(arguments.report_dir.glob("*__*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    manifest = json.loads(
        (arguments.report_dir / "source_manifest.json").read_text(encoding="utf-8")
    )
    expected_hash = manifest["effective_owner_sha256"]
    if sha256(root / OWNER_PATH) != expected_hash or any(
        row["owner_source_sha256"] != expected_hash for row in rows
    ):
        raise RuntimeError("Historical owner fingerprint changed during execution")
    summary = {
        "revision": BASELINE,
        "owner_source": arguments.owner_source,
        "owner_source_sha256": expected_hash,
        "source_manifest": "source_manifest.json",
        "production_policy_mutated_by_runner": False,
        "current_worktree_product_or_harness_imported": False,
        "environment": environment(),
        "command": manifest["command"],
        "pytest_exit_code": int(exit_code),
        "collected_case_policy_samples": receipt.collected,
        "test_outcomes": dict(receipt.outcomes),
        "available_schedules": len(CASES),
        "case_policy_samples": len(rows),
        "source_input_episodes_per_policy": {
            policy: sum(row["source_inputs"] for row in rows if row["policy"] == policy)
            for policy in sorted({row["policy"] for row in rows})
        },
        "imported_sources": imported_sources,
        "rows": [
            {
                key: row[key]
                for key in (
                    "case",
                    "policy",
                    "source_inputs",
                    "commits",
                    "grace_only_wait_ms",
                    "metrics",
                    "required_continuation_status",
                    "physical_active_at_observation_end",
                )
            }
            | {"provider_invocations": len(row["llm_invocations"])}
            for row in rows
        ],
        "case_reports": [path.name for path in paths],
    }
    write_json(arguments.report_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "pytest_exit_code",
                    "collected_case_policy_samples",
                    "test_outcomes",
                    "case_policy_samples",
                    "source_input_episodes_per_policy",
                    "owner_source_sha256",
                    "environment",
                )
            }
        )
    )
    return int(exit_code)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay historical SELF-GRACE-1 evidence, not the current SELF-STREAM-2 product"
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="A new directory; existing evidence is never overwritten",
    )
    parser.add_argument(
        "--case",
        default="grace_233_controlled_comparison",
        help="pytest -k selector; default runs all 53 historical schedules",
    )
    parser.add_argument(
        "--owner-source", choices=("baseline", "historical-correctness"), default="baseline"
    )
    parser.add_argument("--_isolated-root", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    arguments.report_dir = arguments.report_dir.resolve()
    if arguments._isolated_root is not None:
        return execute_isolated(arguments)
    if arguments.report_dir.exists():
        parser.error(
            "--report-dir must not exist; choose a fresh directory to preserve original evidence"
        )
    checkout = Path(__file__).resolve().parents[3]
    fixture = checkout / FIXTURE_PATH
    archive = subprocess.run(
        ["git", "archive", "--format=tar", BASELINE, *ARCHIVE_PATHS],
        cwd=checkout,
        check=True,
        capture_output=True,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="self-grace-233-") as temporary:
        root = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
            source.extractall(root, filter="data")
        if sha256(root / OWNER_PATH) != BASELINE_OWNER_SHA256:
            raise RuntimeError("Pinned historical baseline owner fingerprint does not match")
        pinned_hashes = {
            path: sha256(root / path)
            for path in (
                OWNER_PATH,
                "src/puripuly_heart/core/orchestrator/channel_runtime.py",
                "tests/conftest.py",
                "tests/helpers/translation_owners.py",
                "tests/core/test_self_translation_low_latency.py",
                "tests/core/runtime/test_self_capture_session.py",
                "tests/core/test_stt_scoped_engine.py",
                "pyproject.toml",
            )
        }
        if arguments.owner_source == "historical-correctness":
            candidate = fixture / "correctness_results/source/self_translation_channel.py"
            if sha256(candidate) != CORRECTNESS_OWNER_SHA256:
                raise RuntimeError(
                    "Preserved historical correctness owner fingerprint does not match"
                )
            (root / OWNER_PATH).write_bytes(candidate.read_bytes())
        isolated_fixture = root / FIXTURE_PATH
        isolated_fixture.mkdir(parents=True, exist_ok=True)
        (isolated_fixture / "experiment.py").write_bytes((fixture / "experiment.py").read_bytes())
        (isolated_fixture / "test_historical_self_grace_233.py").write_text(
            WRAPPER, encoding="utf-8", newline="\n"
        )
        arguments.report_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "base_revision": BASELINE,
            "archive_paths": list(ARCHIVE_PATHS),
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "pinned_file_sha256": pinned_hashes,
            "owner_source": arguments.owner_source,
            "effective_owner_sha256": sha256(root / OWNER_PATH),
            "historical_correctness_origin": (
                CORRECTNESS_ORIGIN if arguments.owner_source == "historical-correctness" else None
            ),
            "owner_override": (
                "correctness_results/source/self_translation_channel.py"
                if arguments.owner_source == "historical-correctness"
                else None
            ),
            "experimental_fixture_sha256": sha256(isolated_fixture / "experiment.py"),
            "runner_sha256": sha256(Path(__file__)),
            "collection_adapter_sha256": sha256(
                isolated_fixture / "test_historical_self_grace_233.py"
            ),
            "command": portable_command(arguments, checkout),
            "dependency_installation": "Existing interpreter environment; no installation or downloads",
        }
        write_json(arguments.report_dir / "source_manifest.json", manifest)
        child_env = os.environ.copy()
        child_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        child_env.pop("PYTEST_ADDOPTS", None)
        child_env.pop("PYTEST_PLUGINS", None)
        child = subprocess.run(
            [
                sys.executable,
                "-B",
                "-I",
                str(Path(__file__).resolve()),
                "--_isolated-root",
                str(root),
                "--report-dir",
                str(arguments.report_dir),
                "--owner-source",
                arguments.owner_source,
                "--case",
                arguments.case,
            ],
            cwd=root,
            env=child_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        output = (
            (child.stdout + child.stderr)
            .replace(str(root), "<historical-sandbox>")
            .replace(str(checkout), "<checkout>")
        )
        with (arguments.report_dir / "pytest_output.txt").open("x", encoding="utf-8") as log:
            log.write(output)
        print(output, end="")
    write_json(
        arguments.report_dir / "execution.json",
        {
            "process_exit_code": child.returncode,
            "temporary_source_tree_removed": not root.exists(),
            "original_evidence_overwritten": False,
        },
    )
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
