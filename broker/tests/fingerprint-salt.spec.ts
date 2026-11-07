import { DatabaseSync } from 'node:sqlite';

import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import { createDeviceKeyPair, signCanonicalVerifyRequest } from './test-support/ed25519';
import { applyBrokerMigrations } from './test-support/migrations';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { postVerify } from './test-support/trial-api';

describe('broker fingerprint salt persistence', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('creates versioned fingerprint storage and the bootstrap salt in the first migration', () => {
    const db = new DatabaseSync(':memory:');
    try {
      applyBrokerMigrations(db, { through: '0000_define_broker_persistent_state.sql' });
      const row = db.prepare("SELECT value FROM broker_config WHERE key = 'fingerprint_salt'").get() as { value: string };
      expect(JSON.parse(row.value)).toEqual({
        current: { version: 1, salt: '__BOOTSTRAP_REQUIRED__' },
        previous: null,
        rotated_at: null,
      });
      expect(db.prepare("SELECT name FROM pragma_table_info('installations') WHERE name IN ('hardware_hash_salt_version', 'challenge_salt_version') ORDER BY name").all()).toEqual([
        { name: 'challenge_salt_version' },
        { name: 'hardware_hash_salt_version' },
      ]);
    } finally {
      db.close();
    }
  });

  it('shares the stored current salt across installations and permits an in-flight challenge after rotation', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));
    const env = createTestBrokerEnv();
    const firstKey = await createDeviceKeyPair();
    const secondKey = await createDeviceKeyPair();
    const challenges = [];
    for (const [installationId, keyPair] of [
      ['salt-first', firstKey],
      ['salt-second', secondKey],
    ] as const) {
      const response = await app.request('http://broker.test/v1/trial/challenge', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          installation_id: installationId,
          device_public_key: keyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      }, env);
      expect(response.status).toBe(200);
      const payload = await response.json() as {
        challenge: string;
        challenge_expires_at: string;
        fingerprint_salt: { version: number; salt: string };
      };
      expect(payload.fingerprint_salt).toEqual({ version: 7, salt: 'shared-server-fingerprint-salt' });
      challenges.push(payload);
    }
    env.__db.prepare("UPDATE broker_config SET value = ? WHERE key = 'fingerprint_salt'").run(JSON.stringify({
      current: { version: 8, salt: 'rotated-shared-salt' },
      previous: { version: 7, salt: 'shared-server-fingerprint-salt', valid_until: '2026-04-08T06:05:00.000Z' },
      rotated_at: '2026-04-08T06:01:00.000Z',
    }));
    vi.setSystemTime(new Date('2026-04-08T06:01:00Z'));

    const freshKey = await createDeviceKeyPair();
    const freshResponse = await app.request('http://broker.test/v1/trial/challenge', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        installation_id: 'salt-fresh',
        device_public_key: freshKey.devicePublicKey,
        app_version: '1.2.3',
      }),
    }, env);
    expect(freshResponse.status).toBe(200);
    await expect(freshResponse.json()).resolves.toMatchObject({
      fingerprint_salt: { version: 8, salt: 'rotated-shared-salt' },
    });
    expect(env.__db.prepare('SELECT installation_id, challenge_salt_version FROM installations ORDER BY installation_id').all()).toEqual([
      { installation_id: 'salt-first', challenge_salt_version: 7 },
      { installation_id: 'salt-fresh', challenge_salt_version: 8 },
      { installation_id: 'salt-second', challenge_salt_version: 7 },
    ]);

    const firstChallenge = challenges[0]!;
    const verification = await postVerify(env, await signCanonicalVerifyRequest(firstKey.privateKey, {
      installation_id: 'salt-first',
      device_public_key: firstKey.devicePublicKey,
      challenge: firstChallenge.challenge,
      challenge_expires_at: firstChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-before-salt-rotation',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:01:00.000Z',
    }));
    expect(verification.status).toBe(200);
    await expect(verification.json()).resolves.toMatchObject({
      release_token: expect.any(String),
      managed_state: { lifecycle: 'pending_release', managed_availability: true },
    });
    expect(env.__db.prepare("SELECT hardware_hash_salt_version, challenge FROM installations WHERE installation_id = 'salt-first'").get()).toEqual({
      hardware_hash_salt_version: 7,
      challenge: null,
    });
  });
});
