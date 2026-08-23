from __future__ import annotations

from puripuly_heart.app.ports.settings_secrets import (
    SettingsSecretKey,
    SettingsSecretMutation,
)
from puripuly_heart.app.services.settings_secrets import SettingsSecretsOwner


class RecordingSecretStore:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.set_calls: list[tuple[str, str]] = []
        self.delete_calls: list[str] = []
        self.get_calls: list[str] = []
        self.get_failure_key: str | None = None
        self.failure: Exception | None = None

    def get(self, key: str) -> str | None:
        self.get_calls.append(key)
        if key == self.get_failure_key and self.failure is not None:
            raise self.failure
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        if self.failure is not None:
            raise self.failure
        self.values[key] = value
        self.set_calls.append((key, value))

    def delete(self, key: str) -> None:
        if self.failure is not None:
            raise self.failure
        self.values.pop(key, None)
        self.delete_calls.append(key)


def test_settings_secret_snapshot_projects_named_values_and_legacy_fallbacks() -> None:
    store = RecordingSecretStore(
        {
            "google_api_key": "google-secret",
            "openrouter_api_key": "openrouter-secret",
            "deepseek_api_key": "deepseek-secret",
            "cerebras_api_key": "cerebras-secret",
            "deepgram_api_key": "deepgram-secret",
            "soniox_api_key": "soniox-secret",
            "local_llm_api_key": "local-secret",
            "custom_stt_api_key": "custom-secret",
            "alibaba_api_key_beijing": "beijing-secret",
            "alibaba_api_key": "legacy-alibaba-secret",
        }
    )
    owner = SettingsSecretsOwner(secret_store_factory=lambda: store)

    result = owner.load()

    assert result.error_message is None
    assert result.snapshot is not None
    assert result.snapshot.google_api_key == "google-secret"
    assert result.snapshot.openrouter_api_key == "openrouter-secret"
    assert result.snapshot.deepseek_api_key == "deepseek-secret"
    assert result.snapshot.cerebras_api_key == "cerebras-secret"
    assert "openrouter-secret" not in repr(result)
    assert result.snapshot.deepgram_api_key == "deepgram-secret"
    assert result.snapshot.soniox_api_key == "soniox-secret"
    assert result.snapshot.local_llm_api_key == "local-secret"
    assert result.snapshot.custom_stt_api_key == "custom-secret"
    assert result.snapshot.alibaba_api_key_beijing == "beijing-secret"
    assert result.snapshot.alibaba_api_key_singapore == "legacy-alibaba-secret"
    assert store.set_calls == [
        ("alibaba_api_key_singapore", "legacy-alibaba-secret"),
    ]
    assert "google-secret" not in repr(result)
    assert "legacy-alibaba-secret" not in repr(result)


def test_settings_secret_load_reports_backend_construction_failure() -> None:
    def fail() -> RecordingSecretStore:
        raise RuntimeError("boom")

    result = SettingsSecretsOwner(secret_store_factory=fail).load()

    assert result.snapshot is None
    assert result.error_message == "Failed to load secrets: boom"


def test_openrouter_pkce_load_reads_only_its_previous_secret_surface() -> None:
    store = RecordingSecretStore(
        {
            "openrouter_api_key": "openrouter-secret",
            "deepseek_api_key": "deepseek-secret",
            "cerebras_api_key": "cerebras-secret",
            "alibaba_api_key": "legacy-alibaba-secret",
        }
    )

    result = SettingsSecretsOwner(
        secret_store_factory=lambda: store
    ).load_openrouter_pkce()

    assert result.read_error is None
    assert result.snapshot is not None
    assert result.snapshot.openrouter_api_key == "openrouter-secret"
    assert result.snapshot.deepseek_api_key == "deepseek-secret"
    assert result.snapshot.cerebras_api_key == "cerebras-secret"
    assert "openrouter-secret" not in repr(result)
    assert store.get_calls == [
        "openrouter_api_key",
        "deepseek_api_key",
        "cerebras_api_key",
    ]
    assert store.set_calls == []


def test_settings_secret_load_returns_the_sequential_prefix_on_read_failure() -> None:
    store = RecordingSecretStore(
        {
            "google_api_key": "google-secret",
            "openrouter_api_key": "openrouter-secret",
            "deepseek_api_key": "deepseek-secret",
            "cerebras_api_key": "cerebras-secret",
        }
    )
    failure = OSError("unavailable")
    store.get_failure_key = "deepgram_api_key"
    store.failure = failure

    result = SettingsSecretsOwner(secret_store_factory=lambda: store).load()

    assert result.snapshot is not None
    assert result.snapshot.google_api_key == "google-secret"
    assert result.snapshot.openrouter_api_key == "openrouter-secret"
    assert result.snapshot.deepseek_api_key == "deepseek-secret"
    assert result.snapshot.cerebras_api_key == "cerebras-secret"
    assert result.snapshot.deepgram_api_key is None
    assert result.snapshot.soniox_api_key is None
    assert result.read_error is failure


def test_openrouter_pkce_load_returns_its_sequential_prefix_on_read_failure() -> None:
    store = RecordingSecretStore({"openrouter_api_key": "openrouter-secret"})
    failure = OSError("unavailable")
    store.get_failure_key = "deepseek_api_key"
    store.failure = failure

    result = SettingsSecretsOwner(
        secret_store_factory=lambda: store
    ).load_openrouter_pkce()

    assert result.snapshot is not None
    assert result.snapshot.openrouter_api_key == "openrouter-secret"
    assert result.snapshot.deepseek_api_key is None
    assert result.snapshot.cerebras_api_key is None
    assert result.read_error is failure


def test_settings_secret_mutation_owns_set_delete_and_failure_results() -> None:
    store = RecordingSecretStore({"google_api_key": "secret"})
    owner = SettingsSecretsOwner(secret_store_factory=lambda: store)

    mutation = SettingsSecretMutation(
        key=SettingsSecretKey.OPENROUTER_API_KEY,
        value="value",
    )
    written = owner.mutate(mutation)
    deleted = owner.mutate(
        SettingsSecretMutation(key=SettingsSecretKey.GOOGLE_API_KEY, value="")
    )

    assert written.succeeded is True
    assert written.error_type is None
    assert deleted.succeeded is True
    assert store.set_calls == [("openrouter_api_key", "value")]
    assert store.delete_calls == ["google_api_key"]
    assert "value" not in repr(mutation)

    store.failure = OSError("unavailable")
    failed = owner.mutate(
        SettingsSecretMutation(
            key=SettingsSecretKey.OPENROUTER_API_KEY,
            value="replacement",
        )
    )

    assert failed.succeeded is False
    assert failed.error_type == "OSError"
