from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"


def test_application_presence_owner_is_only_factory_composition() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert source.count("create_vrchat_osc_presence_probe_owner(") == 1
    assert "presence_provider=lambda: vrchat_osc_presence" in source
    assert "port_provider=vrchat_probe_port" in source
    assert "publish_notice=presentation.set_dashboard_vrchat_osc_notice" in source
