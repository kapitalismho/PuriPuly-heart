from __future__ import annotations

from tests.helpers.ast_sources import method_source as _method_source
from tests.helpers.paths import REPO_ROOT as ROOT

UI_RUNTIME_PATH = ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "ui_runtime.py"
OWNER_PATH = (
    ROOT
    / "src"
    / "puripuly_heart"
    / "app"
    / "services"
    / "provider"
    / "provider_credential_verification.py"
)
PORT_PATH = ROOT / "src" / "puripuly_heart" / "app" / "ports" / "provider_verifier.py"


def test_ui_interactive_verification_is_an_owner_delegate() -> None:
    source = UI_RUNTIME_PATH.read_text(encoding="utf-8")
    method = _method_source(
        UI_RUNTIME_PATH,
        "UiProviderRuntimeAdapter",
        "verify_api_key",
    )
    interaction = _method_source(
        OWNER_PATH,
        "ProviderCredentialVerificationInteractionOwner",
        "verify",
    )

    assert "def _verify_qwen_llm_api_key(" not in source
    assert "ProviderCredentialVerificationRequest(" not in method
    assert "ProviderCredentialVerificationRequest(" in interaction
    assert "self.credential_verification.verify(provider, key)" in method
    assert "verifier.verify_api_key(" not in method
    assert "verifier.verify_qwen_llm_api_key(" not in method


def test_provider_verifier_port_declares_every_used_verification_operation() -> None:
    source = PORT_PATH.read_text(encoding="utf-8")

    assert "async def verify_api_key(" in source
    assert "async def verify_qwen_llm_api_key(" in source
    assert "async def fetch_openrouter_key_metadata(" in source
    assert "async def verify_provider_secret(" in source
