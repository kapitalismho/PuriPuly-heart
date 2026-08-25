from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.psem_relative_occupancy_gate.eval_access import (
    EvalAccessError,
    recovery_finalization_receipt_path,
    validate_opened_eval_manifest,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    lseend_root,
    research_root,
    safe_output_path,
    sha256_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.run_lseend_trace import (
    LSEENDTraceError,
)
from experiments.psem_relative_occupancy_gate.run_lseend_trace import (
    run_traces as run_lseend_traces,
)
from experiments.psem_relative_occupancy_gate.run_sortformer_trace import (
    SortformerTraceError,
)
from experiments.psem_relative_occupancy_gate.run_sortformer_trace import (
    run_traces as run_sortformer_traces,
)
from experiments.psem_relative_occupancy_gate.trace_runtime import TraceRuntimeError


class EvalTraceError(RuntimeError):
    pass


def _opened_receipt(
    receipt: dict[str, Any],
    *,
    selection_sha256: str,
    access_path: Path,
) -> dict[str, Any]:
    result = dict(receipt)
    result["eval_status"] = "opened_once"
    result["eval_selection_sha256"] = selection_sha256
    result["eval_access_receipt_sha256"] = sha256_file(access_path)
    return result


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    access_path = Path(args.access_receipt).resolve()
    selection_path = Path(args.selection).resolve()
    authorization_path = Path(args.eval_authorization).resolve()
    _, selection = validate_opened_eval_manifest(
        manifest_path=manifest_path,
        access_path=access_path,
        selection_path=selection_path,
        authorization_path=authorization_path,
    )
    if recovery_finalization_receipt_path(authorization_path).exists():
        raise EvalTraceError("EVAL model traces are finalized and cannot be regenerated")
    research = research_root(Path(args.research_root) if args.research_root else None)
    ls_root = lseend_root(Path(args.lseend_root) if args.lseend_root else None)
    sortformer_output = safe_output_path(Path(args.sortformer_output))
    lseend_output = safe_output_path(Path(args.lseend_output))
    if (
        sortformer_output != manifest_path.parent / "sortformer_model_receipt.json"
        or lseend_output != manifest_path.parent / "lseend_model_receipt.json"
    ):
        raise EvalTraceError("EVAL model receipt outputs are not canonical")
    selection_sha256 = str(selection["selection_sha256"])
    sortformer = run_sortformer_traces(
        manifest=manifest_path,
        role="PSEM-STRATEGY-EVAL",
        research=research,
        trace_root=None,
        output=sortformer_output,
        source_ids=None,
        smoke_samples=None,
        resume=bool(args.resume),
        reuse_r8_cache=False,
    )
    write_json(
        sortformer_output,
        _opened_receipt(
            sortformer,
            selection_sha256=selection_sha256,
            access_path=access_path,
        ),
    )
    lseend = run_lseend_traces(
        manifest=manifest_path,
        role="PSEM-STRATEGY-EVAL",
        research=research,
        ls_root=ls_root,
        trace_root=None,
        output=lseend_output,
        source_ids=None,
        smoke_samples=None,
        resume=bool(args.resume),
    )
    write_json(
        lseend_output,
        _opened_receipt(
            lseend,
            selection_sha256=selection_sha256,
            access_path=access_path,
        ),
    )
    print(
        json.dumps(
            {
                "sortformer_output": str(sortformer_output),
                "lseend_output": str(lseend_output),
                "source_count": len(lseend["source_ids"]),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--access-receipt", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--eval-authorization", required=True)
    parser.add_argument("--research-root")
    parser.add_argument("--lseend-root")
    parser.add_argument("--sortformer-output", required=True)
    parser.add_argument("--lseend-output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (
        EvalAccessError,
        EvalTraceError,
        LSEENDTraceError,
        SortformerTraceError,
        TraceRuntimeError,
    ) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
