from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
OWNER_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "provider_credential_verification.py"
)
PORT_PATH = ROOT / "src" / "puripuly_heart" / "app" / "ports" / "provider_verifier.py"


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def test_controller_interactive_verification_is_an_owner_delegate() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    method = _method_source(CONTROLLER_PATH, "GuiController", "verify_api_key")
    interaction = _method_source(
        OWNER_PATH,
        "ProviderCredentialVerificationInteractionOwner",
        "verify",
    )

    assert "def _verify_qwen_llm_api_key(" not in source
    assert "ProviderCredentialVerificationRequest(" not in method
    assert "ProviderCredentialVerificationRequest(" in interaction
    assert "_get_provider_credential_verification_owner().verify(provider, key)" in method
    assert "verifier.verify_api_key(" not in method
    assert "verifier.verify_qwen_llm_api_key(" not in method


def test_provider_credential_owner_has_no_ui_dependency() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "ProviderVerifierPort" in source
    assert "ProviderCredentialVerificationOutcome" in source


def test_provider_verifier_port_declares_every_used_verification_operation() -> None:
    source = PORT_PATH.read_text(encoding="utf-8")

    assert "async def verify_api_key(" in source
    assert "async def verify_qwen_llm_api_key(" in source
    assert "async def fetch_openrouter_key_metadata(" in source
    assert "async def verify_provider_secret(" in source
