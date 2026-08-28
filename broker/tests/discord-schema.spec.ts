import { describe, expect, it } from 'vitest';

import {
  DEFAULT_BROKER_ABUSE_CONTROLS,
  REQUIRED_BINDINGS,
} from '../src/contract';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

describe('Discord OAuth managed OpenRouter schema contract', () => {
  it('requires the Discord OAuth worker secrets at the broker boundary', () => {
    expect(REQUIRED_BINDINGS.secrets).toEqual(
      expect.arrayContaining([
        'DISCORD_CLIENT_ID',
        'DISCORD_CLIENT_SECRET',
        'DISCORD_REDIRECT_URI_ALLOWLIST',
        'DISCORD_USER_REF_SECRET',
      ]),
    );
  });

  it('requires the QQ auth assertion HMAC worker secret at the broker boundary', () => {
    expect(REQUIRED_BINDINGS.secrets).toEqual(
      expect.arrayContaining(['QQ_AUTH_HMAC_PSK']),
    );
  });

  it('requires the telemetry subject HMAC worker secret at the broker boundary', () => {
    expect(REQUIRED_BINDINGS.secrets).toEqual(
      expect.arrayContaining(['TELEMETRY_SUBJECT_HMAC_SECRET']),
    );
  });

  it('caps new Discord-gated OpenRouter entitlements to 500 per UTC day by default', () => {
    expect(DEFAULT_BROKER_ABUSE_CONTROLS.newActiveEntitlementsPerDay).toEqual({
      endpoint: 'POST /v1/providers/openrouter/discord/issue',
      scope: 'global',
      maxCount: 500,
      windowDays: 1,
    });
  });

  it('rate limits QQ auth assertion attempts by IP by default', () => {
    expect(DEFAULT_BROKER_ABUSE_CONTROLS.qqAuthAssertIp).toEqual({
      endpoint: 'POST /v1/auth/qq/assert',
      scope: 'ip',
      maxRequests: 20,
      windowMinutes: 15,
    });
  });

  it('migrates Discord OAuth session, identity, and entitlement issuance columns', () => {
    const env = createTestBrokerEnv();

    const tables = env.__db
      .prepare("SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name")
      .all() as Array<{ name: string }>;
    expect(tables.map((table) => table.name)).toEqual(
      expect.arrayContaining(['discord_oauth_sessions', 'discord_identities']),
    );

    const entitlementColumns = env.__db
      .prepare("SELECT name FROM pragma_table_info('openrouter_entitlements') ORDER BY cid")
      .all() as Array<{ name: string }>;
    expect(entitlementColumns.map((column) => column.name)).toEqual(
      expect.arrayContaining([
        'discord_user_ref',
        'discord_issue_status',
        'discord_issue_reserved_at',
        'discord_issue_delivered_at',
      ]),
    );
  });

  it('migrates the QQ auth assertion table contract', () => {
    const env = createTestBrokerEnv();

    const tables = env.__db
      .prepare("SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name")
      .all() as Array<{ name: string }>;
    expect(tables.map((table) => table.name)).toEqual(
      expect.arrayContaining(['qq_auth_assertions']),
    );

    const columns = env.__db
      .prepare("SELECT name FROM pragma_table_info('qq_auth_assertions') ORDER BY cid")
      .all() as Array<{ name: string }>;
    expect(columns.map((column) => column.name)).toEqual([
      'qq_subject_ref',
      'credential_hash',
      'asserted_at',
      'received_at',
      'status',
    ]);
  });
});
