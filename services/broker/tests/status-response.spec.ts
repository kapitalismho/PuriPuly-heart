import { afterEach, describe, expect, it, vi } from 'vitest';

import { createDeviceKeyPair, signCanonicalStatusRequest } from './test-support/ed25519';
import { createTestBrokerEnv, insertEntitlement } from './test-support/sqlite-d1';
import { getTrialStatus, issueChallenge } from './test-support/trial-api';


function collectResponseKeys(value: unknown): string[] {
  if (!value || typeof value !== 'object') {
    return [];
  }

  if (Array.isArray(value)) {
    return value.flatMap((entry) => collectResponseKeys(entry));
  }

  return Object.entries(value).flatMap(([key, nested]) => [
    key,
    ...collectResponseKeys(nested),
  ]);
}



describe('GET /v1/trial/status response contract', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });


  it('returns normalized active managed state with broker-side eligibility metadata', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-active',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertEntitlement(env, {
      installation_id: 'install-status-active',
      status: 'active',
      budget_usd: 0.04,
      managed_credential_ref: 'internal-active-ref',
      issued_at: '2026-04-01T00:00:00Z',
      expires_at: '2026-10-01T00:00:00Z',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-active',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-active',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      managed_state: {
        lifecycle: 'active',
        managed_availability: true,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.04,
        issued_at: '2026-04-01T00:00:00Z',
        expires_at: '2026-10-01T00:00:00Z',
      },
      onboarding_eligibility: {
        eligible: false,
        reason: 'active',
        requires_discord_oauth: false,
      },
    });
  });


  it('returns Discord-required eligibility without silent browser launch fields when no entitlement exists', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-discord-required',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-discord-required',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-discord-required',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toEqual({
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
      onboarding_eligibility: {
        eligible: true,
        reason: 'discord_required',
        requires_discord_oauth: true,
      },
    });
    expect(payload).not.toHaveProperty('authorization_url');
    expect(payload).not.toHaveProperty('redirect_uri');
  });

  it('keeps status responses free of challenge, release-session, and credential storage fields', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-hidden-fields',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    env.__db
      .prepare(
        `INSERT INTO discord_identities (
            discord_user_ref,
            entitlement_installation_id,
            status,
            ref_secret_version,
            created_at,
            updated_at
          ) VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'raw-discord-user-123456789012345678:user@example.test',
        'install-status-hidden-fields',
        'issuing',
        1,
        '2026-04-08T06:00:00Z',
        '2026-04-08T06:00:00Z',
      );
    insertEntitlement(env, {
      installation_id: 'install-status-hidden-fields',
      status: 'pending_release',
      budget_usd: 0.07,
      managed_credential_ref: 'internal-pending-ref',
      release_session_ref: 'release-session',
      release_token_hash: 'release-token-hash',
      release_token_expires_at: '2026-04-08T06:15:00Z',
      discord_user_ref: 'raw-discord-user-123456789012345678:user@example.test',
      discord_issue_status: 'issuing',
      discord_issue_reserved_at: '2026-04-08T06:00:00Z',
      discord_issue_delivered_at: '2026-04-08T06:01:00Z',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-hidden-fields',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-hidden-fields',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });
    expect(response.status).toBe(200);

    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).not.toHaveProperty('challenge');
    expect(payload).not.toHaveProperty('challenge_expires_at');
    expect(payload).not.toHaveProperty('fingerprint_salt');
    expect(payload).not.toHaveProperty('release_token');
    expect(payload).not.toHaveProperty('release_session_ref');
    expect(payload).not.toHaveProperty('release_token_hash');
    expect(payload).not.toHaveProperty('managed_credential_ref');
    expect(payload).not.toHaveProperty('authorization_url');

    const responseKeys = collectResponseKeys(payload);
    const forbiddenResponseKeys = [
      'discord_user_ref',
      'discord_user_id',
      'discord_id',
      'discord_email',
      'discord_email_verified',
      'discord_account_created_at',
      'discord_issue_status',
      'discord_issue_reserved_at',
      'discord_issue_delivered_at',
      'state_hash',
      'redirect_uri',
      'pkce_code_verifier',
      'issue_nonce_hash',
      'authorization_url',
    ];
    const leakedResponseKeys = forbiddenResponseKeys.filter((key) =>
      responseKeys.includes(key),
    );
    expect(leakedResponseKeys).toEqual([]);
    const serializedPayload = JSON.stringify(payload);
    expect(serializedPayload).not.toContain('raw-discord-user-123456789012345678');
    expect(serializedPayload).not.toContain('user@example.test');
  });

  it('treats pending_release as broker-side onboarding continuation rather than a terminal state', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-pending-release',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertEntitlement(env, {
      installation_id: 'install-status-pending-release',
      status: 'pending_release',
      budget_usd: 0.07,
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-pending-release',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-pending-release',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({
      managed_state: {
        lifecycle: 'pending_release',
        managed_availability: true,
      },
      onboarding_eligibility: {
        eligible: false,
        reason: 'pending_release',
        requires_discord_oauth: false,
      },
    });
  });
});
