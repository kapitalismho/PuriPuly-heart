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
    {
        "id": "translation_request",
        "constructor": "TranslationRequestOwner",
        "allowed_paths": ("app/wiring/wiring_runtime_pipeline.py",),
    },
    {
        "id": "self_translation",
        "constructor": "SelfTranslationChannelOwner",
        "allowed_paths": ("app/wiring/wiring_runtime_pipeline.py",),
    },
    {
        "id": "peer_translation",
        "constructor": "PeerTranslationChannelOwner",
        "allowed_paths": ("app/wiring/wiring_runtime_pipeline.py",),
    },
    {
        "id": "translation_configuration",
        "constructor": "TranslationRuntimeConfigurationOwner",
        "allowed_paths": (
            "app/wiring/wiring_runtime_pipeline.py",
            "core/orchestrator/context.py",
        ),
    },
]


@pytest.fixture(scope="module")
def constructions() -> dict[str, list[str]]:
    return find_constructions(tuple(case["constructor"] for case in CASES), SOURCE_ROOT)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_owner_construction_is_exclusive(case, constructions) -> None:
    assert constructions[case["constructor"]] == list(case["allowed_paths"])


def test_construction_census_ignores_text_and_preserves_sites_and_multiplicity(tmp_path) -> None:
    wiring = tmp_path / "wiring.py"
    wiring.write_text(
        'description = "TranslationRequestOwner()"\n'
        "# TranslationRequestOwner()\n"
        "request = TranslationRequestOwner (\n"
        "    presentation = projection,\n"
        ")\n",
        encoding="utf-8",
    )
    names = ("TranslationRequestOwner", "OutputRuntime")
    assert find_constructions(names, tmp_path) == {
        "TranslationRequestOwner": ["wiring.py"],
        "OutputRuntime": [],
    }
    (tmp_path / "channel.py").write_text(
        "first = TranslationRequestOwner()\nsecond = TranslationRequestOwner()\n",
        encoding="utf-8",
    )
    assert find_constructions(names, tmp_path) == {
        "TranslationRequestOwner": ["channel.py", "channel.py", "wiring.py"],
        "OutputRuntime": [],
    }
