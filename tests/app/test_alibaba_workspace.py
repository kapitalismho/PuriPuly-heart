from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import httpx
import pytest
from puripuly_heart.app.services.provider_runtime_apply import (
    ProviderRuntimeOwner,
    ProviderRuntimeState,
)

from puripuly_heart.app.ports.secret_store import SecretSnapshot
from puripuly_heart.app.services.provider.alibaba_workspace import AlibabaWorkspaceOwner
from puripuly_heart.app.wiring.wiring_provider_runtime_policy import build_llm_provider_signature
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_stt_provider_signature_from_vnext,
    build_self_stt_provider_signature_from_vnext,
)
from puripuly_heart.config.alibaba_connection import AlibabaRegionalSettings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


@dataclass
class Secrets:
    values: dict[str, str] = field(default_factory=dict)

    async def snapshot_secret(self, key: str) -> SecretSnapshot:
        value = self.values.get(key)
        return SecretSnapshot(key, value, None, value is not None)


@dataclass
class Verifier:
    requests: list[tuple[str, str, str]] = field(default_factory=list)
    pending: asyncio.Future[bool] | None = None
    asr_success: bool = True
    translation_success: bool = True

    async def verify_qwen_audio_api_key(self, key: str, *, endpoint: str, model: str) -> bool:
        self.requests.append(("asr", endpoint, model))
        if self.pending is not None:
            return await self.pending
        return self.asr_success

    async def probe_qwen_llm_api_key(self, key: str, *, base_url: str, model: str) -> bool:
        self.requests.append(("translation", base_url, model))
        return self.translation_success


def owner_with_secret(key: str | None = "test-key"):
    settings = SimpleNamespace(canonical=AppSettingsVNext())
    secret = Secrets({"alibaba_api_key_beijing": key} if key else {})
    verifier = Verifier()
    owner = AlibabaWorkspaceOwner(
        settings,
        SimpleNamespace(secret_store_factory=lambda _current: secret),
        verifier,
    )
    return owner, settings, secret, verifier


@pytest.mark.asyncio
async def test_dedicated_draft_verifies_capabilities_independently_without_active_mutation() -> (
    None
):
    owner, settings, _, verifier = owner_with_secret()
    initial = settings.canonical
    draft = await owner.begin()
    assert draft.asr.state == "unverified" and draft.translation.state == "unverified"
    draft = await owner.edit(
        token=draft.token,
        endpoint_mode="workspace_dedicated",
        api_host="https://work-123.cn-beijing.maas.aliyuncs.com/api/v1",
    )
    verifier.asr_success = False
    asr = await owner.verify(token=draft.token, capability="asr")
    assert asr.asr.state == "failed" and asr.translation.state == "unverified"
    assert verifier.requests == [
        (
            "asr",
            "wss://work-123.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
            "qwen-audio-3.1-asr-flash-streaming",
        )
    ]
    translation = await owner.verify(token=draft.token, capability="translation")
    assert translation.translation.state == "verified" and translation.asr.state == "failed"
    assert translation.translation.credential_revision and translation.translation.credential_saved
    assert verifier.requests[-1] == (
        "translation",
        "https://work-123.cn-beijing.maas.aliyuncs.com/api/v1",
        initial.intent.translation.qwen.llm_model,
    )
    assert settings.canonical is initial
    owner.cancel(token=draft.token)
    assert settings.canonical is initial


@pytest.mark.asyncio
async def test_incomplete_or_unsafe_dedicated_draft_never_probes_or_applies() -> None:
    owner, settings, _, verifier = owner_with_secret()
    draft = await owner.begin()
    draft = await owner.edit(
        token=draft.token,
        endpoint_mode="workspace_dedicated",
        api_host="https://evil.test/?key=private",
    )
    assert draft.connection is None and draft.asr.state == "incomplete"
    assert (
        await owner.verify(token=draft.token, capability="both")
    ).translation.state == "incomplete"
    with pytest.raises(ValueError) as error:
        await owner.apply(token=draft.token, apply_settings=lambda _settings: None)
    assert "private" not in str(error.value)
    assert verifier.requests == []
    assert settings.canonical.intent.translation.qwen.beijing.endpoint_mode == "legacy_shared"


@pytest.mark.asyncio
async def test_late_response_and_key_or_model_changes_invalidate_draft_evidence() -> None:
    owner, settings, secrets, verifier = owner_with_secret()
    draft = await owner.begin()
    verifier.pending = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(owner.verify(token=draft.token, capability="asr"))
    while not verifier.requests:
        await asyncio.sleep(0)
    newer = await owner.edit(
        token=draft.token,
        endpoint_mode="workspace_dedicated",
        api_host="work-123.cn-beijing.maas.aliyuncs.com",
    )
    verifier.pending.set_result(True)
    old_result = await task
    assert old_result.asr.state == "invalidated"
    assert (await owner.read()).asr.state == "unverified"
    secrets.values["alibaba_api_key_beijing"] = "replaced-key"
    assert (await owner.read()).asr.state == "invalidated"
    with pytest.raises(ValueError):
        await owner.verify(token=newer.token, capability="translation")
    fresh = await owner.begin()
    qwen = replace(settings.canonical.intent.translation.qwen, llm_model="qwen3.5-flash")
    settings.canonical = replace(
        settings.canonical,
        intent=replace(
            settings.canonical.intent,
            translation=replace(settings.canonical.intent.translation, qwen=qwen),
        ),
    )
    assert (await owner.read()).translation.state == "invalidated"
    assert (await owner.read()).asr.state == "unverified"
    assert fresh.token != newer.token


@pytest.mark.asyncio
async def test_apply_switches_regions_and_can_explicitly_return_to_shared() -> None:
    owner, settings, secrets, _ = owner_with_secret()
    secrets.values["alibaba_api_key_singapore"] = "singapore-key"
    draft = await owner.begin()
    draft = await owner.edit(
        token=draft.token,
        endpoint_mode="workspace_dedicated",
        api_host="work-123.cn-beijing.maas.aliyuncs.com",
    )
    draft = await owner.edit(
        token=draft.token,
        region="singapore",
        endpoint_mode="workspace_dedicated",
        api_host="work-456.ap-southeast-1.maas.aliyuncs.com",
    )
    assert draft.region == "singapore"
    applied = []

    async def apply(candidate: AppSettingsVNext) -> bool:
        applied.append(candidate)
        settings.canonical = candidate
        return True

    assert await owner.apply(token=draft.token, apply_settings=apply)
    qwen = settings.canonical.intent.translation.qwen
    assert qwen.region == "singapore"
    assert qwen.beijing == AlibabaRegionalSettings(
        "workspace_dedicated", "work-123.cn-beijing.maas.aliyuncs.com", 1
    )
    assert qwen.singapore == AlibabaRegionalSettings(
        "workspace_dedicated", "work-456.ap-southeast-1.maas.aliyuncs.com", 1
    )
    draft = await owner.begin()
    draft = await owner.edit(token=draft.token, endpoint_mode="legacy_shared")
    assert await owner.apply(token=draft.token, apply_settings=apply)
    assert settings.canonical.intent.translation.qwen.singapore.endpoint_mode == "legacy_shared"
    assert (
        settings.canonical.intent.translation.qwen.singapore.api_host
        == "work-456.ap-southeast-1.maas.aliyuncs.com"
    )
    assert settings.canonical.intent.translation.qwen.singapore.revision == 2
    assert len(applied) == 2



@pytest.mark.asyncio
async def test_runtime_plan_replaces_only_affected_qwen_clients() -> None:
    initial = AppSettingsVNext()
    previous = replace(
        initial,
        intent=replace(
            initial.intent,
            translation=replace(
                initial.intent.translation, model="qwen38_flash", connection="official_byok"
            ),
            stt=replace(initial.intent.stt, provider="qwen_audio"),
            peer_stt=replace(initial.intent.peer_stt, provider="qwen_audio"),
        ),
    )
    qwen = previous.intent.translation.qwen
    regional = AlibabaRegionalSettings(
        "workspace_dedicated", "work-123.cn-beijing.maas.aliyuncs.com", 1
    )
    next_settings = replace(
        previous,
        intent=replace(
            previous.intent,
            translation=replace(
                previous.intent.translation,
                qwen=replace(qwen, beijing=regional),
            ),
        ),
    )
    events = []

    async def record(value):
        events.append(value)

    self_signature = build_self_stt_provider_signature_from_vnext
    peer_signature = build_peer_stt_provider_signature_from_vnext
    llm_signature = build_llm_provider_signature
    owner = ProviderRuntimeOwner(
        state_provider=lambda _: ProviderRuntimeState(True, True, True, True, True, True),
        common_effect=lambda _: events.append("settings"),
        rebuild_llm=lambda: record("llm"),
        recover_gpu=lambda _settings, _plan: record("gpu"),
        refresh_peer=lambda: record("peer"),
        refresh_self_stt=lambda: record("self"),
        signature_sink=lambda _: events.append("signatures"),
        llm_retry_sink=lambda: events.append("retry"),
        current_settings_provider=lambda: previous,
        signature_cache_provider=lambda: (
            self_signature(previous),
            peer_signature(previous),
            llm_signature(previous),
        ),
        self_signature_builder=self_signature,
        peer_signature_builder=lambda settings, _: peer_signature(settings),
        llm_signature_builder=llm_signature,
        gpu_restart_decision=lambda _previous, _next: False,
    )
    plan = owner.build_plan(next_settings, force_rebuild_llm=False)
    assert plan.should_rebuild_llm and plan.should_refresh_self_stt and plan.should_refresh_peer
    await owner.apply(next_settings, plan)
    assert events == ["settings", "llm", "peer", "self", "signatures"]

    unaffected = replace(
        previous,
        intent=replace(
            previous.intent,
            translation=initial.intent.translation,
            stt=initial.intent.stt,
            peer_stt=initial.intent.peer_stt,
        ),
    )
    changed_unselected = replace(
        unaffected,
        intent=replace(
            unaffected.intent,
            translation=replace(
                unaffected.intent.translation, qwen=replace(qwen, beijing=regional)
            ),
        ),
    )
    owner.current_settings_provider = lambda: unaffected
    owner.signature_cache_provider = lambda: (
        self_signature(unaffected),
        peer_signature(unaffected),
        llm_signature(unaffected),
    )
    idle_plan = owner.build_plan(changed_unselected, force_rebuild_llm=False)
    assert not idle_plan.should_rebuild_llm
    assert not idle_plan.should_refresh_self_stt
    assert not idle_plan.should_refresh_peer


@pytest.mark.asyncio
async def test_active_evidence_tracks_capability_model_and_credential_revisions() -> None:
    owner, settings, secrets, _ = owner_with_secret()
    draft = await owner.begin()
    draft = await owner.verify(token=draft.token, capability="both")

    async def apply(candidate: AppSettingsVNext) -> bool:
        settings.canonical = candidate
        return True

    assert await owner.apply(token=draft.token, apply_settings=apply)
    active = await owner.active()
    assert active.scope == "active"
    assert active.asr.state == active.translation.state == "verified"
    previous = settings.canonical
    qwen = replace(previous.intent.translation.qwen, llm_model="qwen3.5-flash")
    settings.canonical = replace(
        previous,
        intent=replace(
            previous.intent, translation=replace(previous.intent.translation, qwen=qwen)
        ),
    )
    model_changed = await owner.active()
    assert model_changed.asr.state == "verified"
    assert model_changed.translation.state == "invalidated"
    secrets.values["alibaba_api_key_beijing"] = "new-key"
    key_changed = await owner.active()
    assert key_changed.asr.state == key_changed.translation.state == "invalidated"


@pytest.mark.asyncio
async def test_apply_preserves_newer_untouched_region_and_rejects_stale_edited_region() -> None:
    owner, settings, secrets, _ = owner_with_secret()
    secrets.values["alibaba_api_key_singapore"] = "singapore-key"
    draft = await owner.begin()
    draft = await owner.edit(
        token=draft.token,
        endpoint_mode="workspace_dedicated",
        api_host="work-123.cn-beijing.maas.aliyuncs.com",
    )
    previous = settings.canonical
    qwen = previous.intent.translation.qwen
    singapore = AlibabaRegionalSettings(
        "workspace_dedicated", "work-456.ap-southeast-1.maas.aliyuncs.com", 1
    )
    settings.canonical = replace(
        previous,
        intent=replace(
            previous.intent,
            translation=replace(
                previous.intent.translation, qwen=replace(qwen, singapore=singapore)
            ),
        ),
    )

    async def apply(candidate: AppSettingsVNext) -> bool:
        settings.canonical = candidate
        return True

    assert await owner.apply(token=draft.token, apply_settings=apply)
    assert settings.canonical.intent.translation.qwen.singapore == singapore
    draft = await owner.begin()
    draft = await owner.edit(
        token=draft.token,
        region="singapore",
        api_host="work-789.ap-southeast-1.maas.aliyuncs.com",
    )
    draft = await owner.edit(token=draft.token, region="beijing")
    previous = settings.canonical
    newer = replace(singapore, api_host="work-999.ap-southeast-1.maas.aliyuncs.com", revision=2)
    qwen = replace(previous.intent.translation.qwen, singapore=newer)
    settings.canonical = replace(
        previous,
        intent=replace(
            previous.intent, translation=replace(previous.intent.translation, qwen=qwen)
        ),
    )
    with pytest.raises(ValueError, match="draft changed"):
        await owner.apply(token=draft.token, apply_settings=apply)
    assert settings.canonical.intent.translation.qwen.singapore == newer


@pytest.mark.asyncio
async def test_reselecting_shared_recovers_invalid_dedicated_host_without_key_change() -> None:
    owner, settings, secrets, _ = owner_with_secret()
    draft = await owner.begin()
    draft = await owner.edit(
        token=draft.token,
        endpoint_mode="workspace_dedicated",
        api_host="https://bad.example/?token=private",
    )
    assert draft.connection is None
    draft = await owner.edit(token=draft.token, endpoint_mode="legacy_shared")
    assert draft.connection is not None
    assert draft.api_host == ""

    async def apply(candidate: AppSettingsVNext) -> bool:
        settings.canonical = candidate
        return True

    assert await owner.apply(token=draft.token, apply_settings=apply)
    assert settings.canonical.intent.translation.qwen.beijing == AlibabaRegionalSettings()
    assert secrets.values == {"alibaba_api_key_beijing": "test-key"}


@pytest.mark.asyncio
async def test_effective_legacy_and_environment_keys_drive_presence_and_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _, secrets, verifier = owner_with_secret(None)
    monkeypatch.setenv("ALIBABA_API_KEY_BEIJING", "regional-env")
    monkeypatch.setenv("ALIBABA_API_KEY", "generic-env")
    secrets.values["alibaba_api_key"] = "legacy-key"
    draft = await owner.begin()
    assert draft.key_present
    verified = await owner.verify(token=draft.token, capability="translation")
    assert verified.translation.credential_saved
    assert verified.translation.credential_revision
    assert verifier.requests[-1][0] == "translation"
    secrets.values["alibaba_api_key_beijing"] = "new-regional-key"
    assert (await owner.read()).translation.state == "invalidated"
    secrets.values.clear()
    draft = await owner.begin()
    assert draft.key_present
    verified = await owner.verify(token=draft.token, capability="asr")
    assert verified.asr.state == "verified"
    assert not verified.asr.credential_saved
    monkeypatch.setenv("ALIBABA_API_KEY_BEIJING", "rotated-env")
    assert (await owner.read()).asr.state == "invalidated"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"code": "InvalidApiKey", "message": "private"}, "authentication_or_access"),
        ({"error": {"code": "model_not_found", "message": "private"}}, "model_unavailable"),
        ({"code": "Throttling", "message": "private"}, "rate_limited"),
        ({"code": "Unrecognized", "message": "private"}, "ambiguous"),
    ],
)
async def test_provider_error_codes_classify_without_disclosing_payload(
    payload: dict[str, object], expected: str
) -> None:
    owner, _, _, verifier = owner_with_secret()
    request = httpx.Request("POST", "https://dashscope.aliyuncs.com/api/v1")
    response = httpx.Response(400, json=payload, request=request)

    async def fail(_key: str, *, base_url: str, model: str) -> bool:
        raise httpx.HTTPStatusError("private", request=request, response=response)

    verifier.probe_qwen_llm_api_key = fail
    draft = await owner.begin()
    outcome = await owner.verify(token=draft.token, capability="translation")
    assert outcome.translation.state == "failed"
    assert outcome.translation.failure_kind == expected
    assert "private" not in repr(outcome)
