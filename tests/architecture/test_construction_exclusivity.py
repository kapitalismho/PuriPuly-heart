from __future__ import annotations

import pytest

from tests.helpers.ast_sources import find_constructions
from tests.helpers.paths import SOURCE_ROOT

CASES = [
    {
        "id": "cpu_repair",
        "constructor": "LocalASRCpuRepairOwner",
        "allowed_paths": ("app/wiring/wiring_composition.py",),
    },
    {
        "id": "diagnostics",
        "constructor": "LocalASRDiagnosticsOwner",
        "allowed_paths": ("app/wiring/wiring_composition.py",),
    },
    {
        "id": "gpu_provisioning",
        "constructor": "LocalASRGpuProvisioningOwner",
        "allowed_paths": ("app/services/gpu_runtime_interaction.py",),
    },
    {
        "id": "provisioning",
        "constructor": "LocalASRProvisioningOwner",
        "allowed_paths": ("app/wiring/root.py",),
    },
    {
        "id": "output_runtime",
        "constructor": "OutputRuntime",
        "allowed_paths": ("app/wiring/wiring_runtime_pipeline.py",),
    },
    {
        "id": "output_projection",
        "constructor": "TranslationOutputProjectionOwner",
        "allowed_paths": ("app/wiring/wiring_runtime_pipeline.py",),
    },
    {
        "id": "output_router",
        "constructor": "OutputRouter",
        "allowed_paths": (),
    },
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_owner_construction_is_exclusive(case) -> None:
    constructions = find_constructions(case["constructor"], SOURCE_ROOT)
    assert constructions == list(case["allowed_paths"])
