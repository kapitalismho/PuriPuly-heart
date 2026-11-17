import { afterEach, describe, expect, it, vi } from 'vitest';

import { createDeviceKeyPair, signCanonicalVerifyRequest } from './test-support/ed25519';
import {
  activatePendingReleaseSession,
  createPendingReleaseSession,
} from './test-support/openrouter-issue';
import { createTestBrokerEnv, insertEntitlement } from './test-support/sqlite-d1';
import { issueChallenge, postVerify } from './test-support/trial-api';

describe('broker duplicate hardware suppression', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns trial_not_eligible when verify sees a hardware hash already bound to an active entitlement on a different installation', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const duplicateHardwareHash = 'hardware-hash-duplicate-active';

    const active = await activatePendingReleaseSession({
      env,
      installationId: 'install-duplicate-source',
      appVersion: '1.2.3',
      hardwareHash: duplicateHardwareHash,
    });
    expect(active.response.status).toBe(200);

    const duplicateKeyPair = await createDeviceKeyPair();
    const duplicateChallenge = await issueChallenge({
      env,
      installationId: 'install-duplicate-target',
      devicePublicKey: duplicateKeyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const duplicateVerify = await signCanonicalVerifyRequest(duplicateKeyPair.privateKey, {
      installation_id: 'install-duplicate-target',
      device_public_key: duplicateKeyPair.devicePublicKey,
      challenge: duplicateChallenge.challenge,
      challenge_expires_at: duplicateChallenge.challenge_expires_at,
      hardware_hash: duplicateHardwareHash,
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const response = await postVerify(env, duplicateVerify);

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual({
      error: {
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_duplicate',
        retry_after_ms: null,
        message: 'hardware_hash is already reserved by another entitlement',
      },
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
    });

    const duplicateInstallation = env.__db
      .prepare(
        `SELECT challenge, challenge_expires_at, hardware_hash
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-duplicate-target') as Record<string, unknown>;
    expect(duplicateInstallation).toEqual({
      challenge: duplicateChallenge.challenge,
      challenge_expires_at: duplicateChallenge.challenge_expires_at,
      hardware_hash: null,
    });

    const duplicateEntitlementCount = env.__db
      .prepare(
        'SELECT COUNT(*) AS count FROM openrouter_entitlements WHERE installation_id = ?',
      )
      .get('install-duplicate-target') as { count: number };
    expect(duplicateEntitlementCount.count).toBe(0);
  });

  it('rejects duplicate hardware for a legacy active entitlement without verified snapshot fields', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const duplicateHardwareHash = 'hardware-hash-duplicate-legacy-active';

    env.__db
      .prepare(
        `INSERT INTO installations (
            installation_id,
            device_public_key,
            hardware_hash,
            hardware_hash_salt_version,
            app_version,
            created_at,
            last_seen_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'install-legacy-active-source',
        'bGVnYWN5LWFjdGl2ZS1kZXZpY2UtcHVibGljLWtleS0wMDE',
        duplicateHardwareHash,
        7,
        '1.2.3',
        '2026-04-08T05:59:00.000Z',
        '2026-04-08T05:59:00.000Z',
      );
    insertEntitlement(env, {
      installation_id: 'install-legacy-active-source',
      status: 'active',
      budget_usd: 0.07,
      managed_credential_ref: 'managed-legacy-active-ref',
      issued_at: '2026-04-08T05:59:10.000Z',
      expires_at: '2026-10-08T05:59:10.000Z',
    });

    const duplicateKeyPair = await createDeviceKeyPair();
    const duplicateChallenge = await issueChallenge({
      env,
      installationId: 'install-legacy-active-target',
      devicePublicKey: duplicateKeyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const duplicateVerify = await signCanonicalVerifyRequest(duplicateKeyPair.privateKey, {
      installation_id: 'install-legacy-active-target',
      device_public_key: duplicateKeyPair.devicePublicKey,
      challenge: duplicateChallenge.challenge,
      challenge_expires_at: duplicateChallenge.challenge_expires_at,
      hardware_hash: duplicateHardwareHash,
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const response = await postVerify(env, duplicateVerify);

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual({
      error: {
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_duplicate',
        retry_after_ms: null,
        message: 'hardware_hash is already reserved by another entitlement',
      },
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
    });
  });

  it('rejects duplicate hardware for a legacy pending_release entitlement without verified snapshot fields', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const duplicateHardwareHash = 'hardware-hash-duplicate-legacy-pending';

    env.__db
      .prepare(
        `INSERT INTO installations (
            installation_id,
            device_public_key,
            hardware_hash,
            hardware_hash_salt_version,
            app_version,
            created_at,
            last_seen_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'install-legacy-pending-source',
        'bGVnYWN5LXBlbmRpbmctZGV2aWNlLXB1YmxpYy1rZXktMDE',
        duplicateHardwareHash,
        7,
        '1.2.3',
        '2026-04-08T05:59:00.000Z',
        '2026-04-08T05:59:00.000Z',
      );
    insertEntitlement(env, {
      installation_id: 'install-legacy-pending-source',
      status: 'pending_release',
      budget_usd: 0.07,
      release_session_ref: 'legacy-pending-session',
      release_token_hash: 'legacy-pending-token-hash',
      release_token_expires_at: '2026-04-08T06:15:00.000Z',
    });

    const duplicateKeyPair = await createDeviceKeyPair();
    const duplicateChallenge = await issueChallenge({
      env,
      installationId: 'install-legacy-pending-target',
      devicePublicKey: duplicateKeyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const duplicateVerify = await signCanonicalVerifyRequest(duplicateKeyPair.privateKey, {
      installation_id: 'install-legacy-pending-target',
      device_public_key: duplicateKeyPair.devicePublicKey,
      challenge: duplicateChallenge.challenge,
      challenge_expires_at: duplicateChallenge.challenge_expires_at,
      hardware_hash: duplicateHardwareHash,
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const response = await postVerify(env, duplicateVerify);

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual({
      error: {
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_duplicate',
        retry_after_ms: null,
        message: 'hardware_hash is already reserved by another entitlement',
      },
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
    });
  });

  it('rejects duplicate hardware when another installation is still pending_release', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const duplicateHardwareHash = 'hardware-hash-duplicate-pending';

    await createPendingReleaseSession({
      env,
      installationId: 'install-pending-source',
      appVersion: '2.0.0',
      hardwareHash: duplicateHardwareHash,
    });

    const duplicateKeyPair = await createDeviceKeyPair();
    const duplicateChallenge = await issueChallenge({
      env,
      installationId: 'install-pending-target',
      devicePublicKey: duplicateKeyPair.devicePublicKey,
      appVersion: '2.0.0',
    });
    const duplicateVerify = await signCanonicalVerifyRequest(duplicateKeyPair.privateKey, {
      installation_id: 'install-pending-target',
      device_public_key: duplicateKeyPair.devicePublicKey,
      challenge: duplicateChallenge.challenge,
      challenge_expires_at: duplicateChallenge.challenge_expires_at,
      hardware_hash: duplicateHardwareHash,
      app_version: '2.0.0',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const response = await postVerify(env, duplicateVerify);

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toMatchObject({
      error: {
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_duplicate',
        message: 'hardware_hash is already reserved by another entitlement',
      },
    });

    const duplicateInstallation = env.__db
      .prepare(
        `SELECT challenge, challenge_expires_at, hardware_hash
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-pending-target') as Record<string, unknown>;
    expect(duplicateInstallation).toEqual({
      challenge: duplicateChallenge.challenge,
      challenge_expires_at: duplicateChallenge.challenge_expires_at,
      hardware_hash: null,
    });

    const duplicateEntitlementCount = env.__db
      .prepare(
        'SELECT COUNT(*) AS count FROM openrouter_entitlements WHERE installation_id = ?',
      )
      .get('install-pending-target') as { count: number };
    expect(duplicateEntitlementCount.count).toBe(0);
  });
});
