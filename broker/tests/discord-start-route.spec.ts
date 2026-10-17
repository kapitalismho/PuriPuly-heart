import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import { createDeviceKeyPair } from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { updateAbuseControls } from './test-support/abuse-controls';

const REGISTERED_REDIRECT_URIS = [
  'http://127.0.0.1:62187/discord/callback',
  'http://127.0.0.1:62188/discord/callback',
  'http://127.0.0.1:62189/discord/callback',
];

describe('Discord OAuth start route', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('starts a pending Discord OAuth session', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-30T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const response = await postDiscordAuthStart({
      env,
      installationId: 'install-discord-start',
      devicePublicKey: keyPair.devicePublicKey,
      redirectUri: REGISTERED_REDIRECT_URIS[0],
      appVersion: '1.2.3',
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as {
      authorization_url: string;
      redirect_uri: string;
      oauth_session_expires_at: string;
      issue_nonce: string;
      fingerprint_salt: { version: number; salt: string };
      fingerprint_salt_version: number;
    };

    expect(payload.redirect_uri).toBe(REGISTERED_REDIRECT_URIS[0]);
    expect(payload.oauth_session_expires_at).toBe('2026-04-30T06:05:00.000Z');
    expect(payload.issue_nonce).toEqual(expect.any(String));
    expect(payload.fingerprint_salt).toEqual({
      version: 7,
      salt: 'shared-server-fingerprint-salt',
    });
    expect(payload.fingerprint_salt_version).toBe(7);

    const authorizationUrl = new URL(payload.authorization_url);
    expect(authorizationUrl.origin).toBe('https://discord.com');
    expect(authorizationUrl.pathname).toBe('/oauth2/authorize');
    expect(authorizationUrl.searchParams.get('client_id')).toBe(
      'test-discord-client-id',
    );
    expect(authorizationUrl.searchParams.get('redirect_uri')).toBe(
      REGISTERED_REDIRECT_URIS[0],
    );
    expect(authorizationUrl.searchParams.get('scope')).toBe('identify email');
    expect(authorizationUrl.searchParams.get('response_type')).toBe('code');
    expect(authorizationUrl.searchParams.get('code_challenge_method')).toBe('S256');
    expect(authorizationUrl.searchParams.get('state')).toEqual(expect.any(String));

    const rows = env.__db
      .prepare('SELECT * FROM discord_oauth_sessions')
      .all() as Array<{
        installation_id: string;
        device_public_key: string;
        redirect_uri: string;
        pkce_code_verifier: string;
        fingerprint_salt_version: number;
        status: string;
        created_at: string;
        expires_at: string;
      }>;
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      installation_id: 'install-discord-start',
      device_public_key: keyPair.devicePublicKey,
      redirect_uri: REGISTERED_REDIRECT_URIS[0],
      fingerprint_salt_version: 7,
      status: 'pending',
      created_at: '2026-04-30T06:00:00.000Z',
      expires_at: '2026-04-30T06:05:00.000Z',
    });
    expect(rows[0]?.pkce_code_verifier).toHaveLength(86);
  });



  it('rejects a localhost redirect URI', async () => {
    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();

    const response = await postDiscordAuthStart({
      env,
      installationId: 'install-discord-localhost',
      devicePublicKey: keyPair.devicePublicKey,
      redirectUri: 'http://localhost:62187/discord/callback',
      appVersion: '1.2.3',
    });

    expect(response.status).toBe(400);
  });

  it('rejects an installation_id already bound to a different device_public_key', async () => {
    const env = createTestBrokerEnv();
    const existingKeyPair = await createDeviceKeyPair();
    const submittedKeyPair = await createDeviceKeyPair();
    insertInstallation(env, {
      installationId: 'install-discord-start-binding-mismatch',
      devicePublicKey: existingKeyPair.devicePublicKey,
    });

    const response = await postDiscordAuthStart({
      env,
      installationId: 'install-discord-start-binding-mismatch',
      devicePublicKey: submittedKeyPair.devicePublicKey,
      redirectUri: REGISTERED_REDIRECT_URIS[0],
      appVersion: '1.2.3',
    });

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'security_fail',
        subcode: 'installation_binding_mismatch',
        message: 'installation_id is already bound to a different device_public_key',
      }),
    );
    expect(countPendingDiscordSessions(env)).toBe(0);
  });

  it('rejects a device_public_key already registered to another installation_id', async () => {
    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    insertInstallation(env, {
      installationId: 'install-discord-start-registered-other',
      devicePublicKey: keyPair.devicePublicKey,
    });

    const response = await postDiscordAuthStart({
      env,
      installationId: 'install-discord-start-registered-new',
      devicePublicKey: keyPair.devicePublicKey,
      redirectUri: REGISTERED_REDIRECT_URIS[0],
      appVersion: '1.2.3',
    });

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'security_fail',
        subcode: 'device_public_key_registered',
        message: 'device_public_key is already registered to a different installation_id',
      }),
    );
    expect(countPendingDiscordSessions(env)).toBe(0);
  });

  it.each(REGISTERED_REDIRECT_URIS)(
    'accepts exact registered redirect URI %s',
    async (redirectUri) => {
      const env = createTestBrokerEnv();
      const keyPair = await createDeviceKeyPair();

      const response = await postDiscordAuthStart({
        env,
        installationId: `install-discord-${new URL(redirectUri).port}`,
        devicePublicKey: keyPair.devicePublicKey,
        redirectUri,
        appVersion: '1.2.3',
      });

      expect(response.status).toBe(200);
      await expect(response.json()).resolves.toMatchObject({
        redirect_uri: redirectUri,
      });
    },
  );

  it.each([
    'http://127.0.0.1:62190/discord/callback',
    'http://127.0.0.1:62187/discord/callback/',
    'http://localhost:62187/discord/callback',
  ])('rejects near-miss redirect URI %s', async (redirectUri) => {
    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();

    const response = await postDiscordAuthStart({
      env,
      installationId: 'install-discord-near-miss',
      devicePublicKey: keyPair.devicePublicKey,
      redirectUri,
      appVersion: '1.2.3',
    });

    expect(response.status).toBe(400);
  });

  it('atomically enforces max pending sessions per installation under concurrent starts', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-30T06:00:00Z'));

    const pendingCountBarrier = createBarrier(2);
    const env = createTestBrokerEnv({
      beforeFirst: async ({ sql }) => {
        if (
          sql.includes('FROM discord_oauth_sessions') &&
          sql.includes("status = 'pending'")
        ) {
          await pendingCountBarrier.wait();
        }
      },
    });
    updateAbuseControls(env, (controls) => {
      controls.pendingDiscordOAuthSessions.maxPerInstallation = 1;
    });

    const keyPair = await createDeviceKeyPair();
    const responses = await Promise.all([
      postDiscordAuthStart({
        env,
        installationId: 'install-discord-concurrent-limit',
        devicePublicKey: keyPair.devicePublicKey,
        redirectUri: REGISTERED_REDIRECT_URIS[0],
        appVersion: '1.2.3',
      }),
      postDiscordAuthStart({
        env,
        installationId: 'install-discord-concurrent-limit',
        devicePublicKey: keyPair.devicePublicKey,
        redirectUri: REGISTERED_REDIRECT_URIS[0],
        appVersion: '1.2.3',
      }),
    ]);

    expect(responses.map((response) => response.status).sort()).toEqual([200, 429]);
    const row = env.__db
      .prepare(
        `SELECT COUNT(*) AS count
           FROM discord_oauth_sessions
          WHERE installation_id = ?
            AND status = 'pending'`,
      )
      .get('install-discord-concurrent-limit') as { count: number };
    expect(row.count).toBe(1);
  });
});

function createBarrier(participantCount: number): { wait: () => Promise<void> } {
  let waiting = 0;
  let release: (() => void) | null = null;
  const promise = new Promise<void>((resolve) => {
    release = resolve;
  });

  return {
    async wait(): Promise<void> {
      waiting += 1;
      if (waiting >= participantCount) {
        release?.();
      }
      await promise;
    },
  };
}

async function postDiscordAuthStart(options: {
  env: ReturnType<typeof createTestBrokerEnv>;
  installationId: string;
  devicePublicKey: string;
  redirectUri: string;
  appVersion: string;
}): Promise<Response> {
  const body: Record<string, unknown> = {
    installation_id: options.installationId,
    device_public_key: options.devicePublicKey,
    redirect_uri: options.redirectUri,
    app_version: options.appVersion,
  };

  return app.request(
    'http://broker.test/v1/auth/discord/start',
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'cf-connecting-ip': '203.0.113.20',
      },
      body: JSON.stringify(body),
    },
    options.env,
  );
}


function insertInstallation(
  env: ReturnType<typeof createTestBrokerEnv>,
  input: {
    installationId: string;
    devicePublicKey: string;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO installations (
          installation_id,
          device_public_key,
          hardware_hash,
          hardware_hash_salt_version,
          app_version,
          challenge,
          challenge_expires_at,
          challenge_salt_version,
          created_at,
          last_seen_at
        ) VALUES (?, ?, NULL, NULL, ?, NULL, NULL, NULL, ?, ?)`,
    )
    .run(
      input.installationId,
      input.devicePublicKey,
      '1.2.3',
      '2026-04-30T06:00:00.000Z',
      '2026-04-30T06:00:00.000Z',
    );
}

function countPendingDiscordSessions(env: ReturnType<typeof createTestBrokerEnv>): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM discord_oauth_sessions')
    .get() as { count: number };
  return Number(row.count);
}
