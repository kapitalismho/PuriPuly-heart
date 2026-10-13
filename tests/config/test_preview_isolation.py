from puripuly_heart.app.wiring.wiring_secrets_factory import _get_secret, create_secret_store
from puripuly_heart.config import paths
from puripuly_heart.config.settings_vnext.defaults import new_settings_for_first_run
from puripuly_heart.config.settings_vnext.schema import SecretsIntent
from puripuly_heart.core.storage.secrets import InMemorySecretStore


def test_preview_isolates_every_default_data_path_and_keyring(monkeypatch, tmp_path):
    monkeypatch.setenv("PURIPULY_HEART_TEST_PROFILE", "desktop-test")
    monkeypatch.setenv("PURIPULY_HEART_TEST_DATA_DIR", str(tmp_path))
    assert paths.default_settings_path() == tmp_path / "settings.json"
    assert paths.default_models_dir() == tmp_path / "models"
    store = create_secret_store(SecretsIntent(), config_path=paths.default_settings_path())
    assert store.service_name == "puripuly-heart.test.desktop-test"
    assert not new_settings_for_first_run("zh-CN").intent.telemetry.enabled
    monkeypatch.setenv("OPENROUTER_API_KEY", "production-secret")
    assert (
        _get_secret(InMemorySecretStore(), key="openrouter", env_var="OPENROUTER_API_KEY") is None
    )


def test_default_secret_namespace_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("PURIPULY_HEART_TEST_PROFILE", raising=False)
    store = create_secret_store(SecretsIntent(), config_path=tmp_path / "settings.json")
    assert store.service_name == "puripuly-heart"


def test_beta_profile_is_separate_from_production_and_developer_preview(monkeypatch, tmp_path):
    monkeypatch.setenv("PURIPULY_HEART_TEST_PROFILE", "desktop-beta")
    monkeypatch.setenv("PURIPULY_HEART_TEST_DATA_DIR", str(tmp_path / "PuriPulyHeartBeta"))
    store = create_secret_store(SecretsIntent(), config_path=paths.default_settings_path())
    assert store.service_name == "puripuly-heart.test.desktop-beta"
    assert paths.default_models_dir() == tmp_path / "PuriPulyHeartBeta" / "models"
    monkeypatch.setenv("OPENROUTER_API_KEY", "production-secret")
    assert (
        _get_secret(InMemorySecretStore(), key="openrouter", env_var="OPENROUTER_API_KEY") is None
    )


import pytest


def test_preview_rejects_external_configuration_and_secret_file(monkeypatch, tmp_path):
    isolated = tmp_path / "preview"
    isolated.mkdir()
    monkeypatch.setenv("PURIPULY_HEART_TEST_PROFILE", "desktop-test")
    monkeypatch.setenv("PURIPULY_HEART_TEST_DATA_DIR", str(isolated))
    with pytest.raises(ValueError, match="configuration"):
        create_secret_store(SecretsIntent(), config_path=tmp_path / "production.json")
    settings = SecretsIntent(
        backend="encrypted_file", encrypted_file_path=str(tmp_path / "production.secrets")
    )
    with pytest.raises(ValueError, match="secret storage"):
        create_secret_store(
            settings, config_path=isolated / "settings.json", passphrase="test-only"
        )
    assert not (tmp_path / "production.secrets").exists()
