import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { BROKER_PERSISTENCE_MODEL } from '../src/contract';
import { readBrokerMigrationSql } from './test-support/migrations';
import { createTestBrokerEnv, insertEntitlement } from './test-support/sqlite-d1';

const OPENROUTER_ENTITLEMENT_COLUMNS = [
  'installation_id',
  'status',
  'budget_usd',
  'managed_credential_ref',
  'issued_at',
  'expires_at',
  'release_session_ref',
  'release_token_hash',
  'release_token_expires_at',
  'verified_hardware_hash',
  'verified_hardware_hash_salt_version',
  'discord_user_ref',
  'discord_issue_status',
  'discord_issue_reserved_at',
  'discord_issue_delivered_at',
];

const DISCORD_ENTITLEMENT_COLUMNS = [
  'discord_user_ref',
  'discord_issue_status',
  'discord_issue_reserved_at',
  'discord_issue_delivered_at',
];

describe('openrouter entitlement schema', () => {
  it('documents verified hardware snapshot columns in the persistence contract', () => {
    expect(BROKER_PERSISTENCE_MODEL.tables.openrouterEntitlements.columns).toEqual(
      OPENROUTER_ENTITLEMENT_COLUMNS,
    );
  });

  it('ships verified hardware snapshot columns in a forward entitlement migration', () => {
    expect(
      readBrokerMigrationSql('0000_define_broker_persistent_state.sql'),
    ).not.toContain('verified_hardware_hash TEXT');
    expect(
      readBrokerMigrationSql('0001_harden_installation_public_inputs.sql'),
    ).not.toContain('verified_hardware_hash TEXT');

    const migration = readBrokerMigrationSql(
      '0002_add_entitlement_verified_hardware_snapshot.sql',
    );

    expect(migration).toContain('ALTER TABLE openrouter_entitlements');
    expect(migration).toContain('verified_hardware_hash TEXT');
    expect(migration).toContain('verified_hardware_hash_salt_version INTEGER');
  });

  it('applies the verified hardware snapshot columns to migrated test databases', () => {
    const env = createTestBrokerEnv();
    const columns = env.__db
      .prepare("SELECT name FROM pragma_table_info('openrouter_entitlements') ORDER BY cid")
      .all() as Array<{ name: string }>;

    expect(columns.map((column) => column.name)).toEqual(OPENROUTER_ENTITLEMENT_COLUMNS);
  });

  it('selects Discord issue columns for full OpenRouter entitlement records', () => {
    for (const sourceFileName of ['trial-handshake.ts', 'openrouter-issue.ts']) {
      const source = readFileSync(
        new URL(`../src/${sourceFileName}`, import.meta.url),
        'utf8',
      );
      const fullRecordSelect = source.match(
        /SELECT installation_id, status, budget_usd, managed_credential_ref, issued_at,[\s\S]*?FROM openrouter_entitlements/u,
      )?.[0];

      expect(fullRecordSelect, `${sourceFileName} full entitlement SELECT`).toBeDefined();
      for (const column of DISCORD_ENTITLEMENT_COLUMNS) {
        expect(fullRecordSelect).toContain(column);
      }
    }
  });

  it('seeds the exact managed OpenRouter and Discord secret bindings in the sqlite D1 test env', () => {
    const env = createTestBrokerEnv() as Record<string, unknown>;

    expect({
      OPENROUTER_MANAGED_API_KEY: env.OPENROUTER_MANAGED_API_KEY,
      OPENROUTER_MANAGEMENT_API_KEY: env.OPENROUTER_MANAGEMENT_API_KEY,
      OPENROUTER_MANAGED_GUARDRAIL_ID: env.OPENROUTER_MANAGED_GUARDRAIL_ID,
      OPENROUTER_MANAGED_USER_HMAC_SECRET: env.OPENROUTER_MANAGED_USER_HMAC_SECRET,
      DISCORD_CLIENT_ID: env.DISCORD_CLIENT_ID,
      DISCORD_CLIENT_SECRET: env.DISCORD_CLIENT_SECRET,
      DISCORD_REDIRECT_URI_ALLOWLIST: env.DISCORD_REDIRECT_URI_ALLOWLIST,
      DISCORD_USER_REF_SECRET: env.DISCORD_USER_REF_SECRET,
    }).toEqual({
      OPENROUTER_MANAGED_API_KEY: 'test-managed-api-key',
      OPENROUTER_MANAGEMENT_API_KEY: 'test-management-api-key',
      OPENROUTER_MANAGED_GUARDRAIL_ID: 'test-managed-guardrail-id',
      OPENROUTER_MANAGED_USER_HMAC_SECRET: 'test-managed-user-hmac-secret',
      DISCORD_CLIENT_ID: 'test-discord-client-id',
      DISCORD_CLIENT_SECRET: 'test-discord-client-secret',
      DISCORD_REDIRECT_URI_ALLOWLIST:
        'http://127.0.0.1:62187/discord/callback,http://127.0.0.1:62188/discord/callback,http://127.0.0.1:62189/discord/callback',
      DISCORD_USER_REF_SECRET: 'test-discord-user-ref-secret',
    });
  });

  it('lets test helpers persist verified hardware snapshots on entitlement rows', () => {
    const env = createTestBrokerEnv();
    env.__db
      .prepare(
        `INSERT INTO installations (installation_id, device_public_key, app_version)
         VALUES (?, ?, ?)`,
      )
      .run('install-snapshot', 'device-public-key-snapshot', '1.0.0');

    insertEntitlement(env, {
      installation_id: 'install-snapshot',
      status: 'active',
      budget_usd: 0.05,
      managed_credential_ref: 'managed-credential-snapshot',
      verified_hardware_hash: 'verified-hardware-hash',
      verified_hardware_hash_salt_version: 7,
    });

    const row = env.__db
      .prepare(
        `SELECT verified_hardware_hash, verified_hardware_hash_salt_version
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-snapshot') as {
      verified_hardware_hash: string | null;
      verified_hardware_hash_salt_version: number | null;
    };

    expect(row).toEqual({
      verified_hardware_hash: 'verified-hardware-hash',
      verified_hardware_hash_salt_version: 7,
    });
  });
});

const QQ_MANAGED_ENTITLEMENT_COLUMNS = [
  'qq_subject_ref',
  'status',
  'issue_ref',
  'managed_credential_ref',
  'budget_usd',
  'reserved_at',
  'issued_at',
  'expires_at',
  'delivered_at',
  'created_at',
  'updated_at',
];

const MANAGED_KEY_DELIVERY_COLUMNS = [
  'delivery_id',
  'issue_source',
  'subject_ref',
  'installation_id',
  'managed_credential_ref',
  'ack_token_hash',
  'status',
  'created_at',
  'expires_at',
  'acknowledged_at',
  'failed_at',
  'failure_reason',
];

describe('QQ managed entitlement schema', () => {
  it('documents the lifecycle source of truth and PSK rotation guardrails in the persistence contract', () => {
    const contract = BROKER_PERSISTENCE_MODEL.tables as Record<string, unknown>;

    expect(contract.qqAuthAssertions).toMatchObject({
      purpose: expect.stringContaining('assertion evidence'),
      duplicateHandling: 'preserve original row; duplicate assertions are idempotent',
    });
    expect(contract.qqManagedEntitlements).toEqual({
      name: 'qq_managed_entitlements',
      purpose:
        'durable QQ Managed production issuance lifecycle keyed by stable subject reference',
      primaryKey: 'qq_subject_ref',
      lifecycleDecisionSource: 'qq_managed_entitlements, not qq_auth_assertions',
      rowCardinality: 'zero-or-one-row-per-qq_subject_ref',
      absenceRepresents: 'no production issuance has been reserved or used',
      storedStatuses: ['issuing', 'delivery_pending', 'active', 'cleanup_required', 'revoked'],
      automaticReissueBlockedStatuses: [
        'delivery_pending',
        'active',
        'cleanup_required',
        'revoked',
      ],
      columns: QQ_MANAGED_ENTITLEMENT_COLUMNS,
      unique: ['issue_ref'],
      partialUniqueIndexes: [
        {
          name: 'idx_qq_managed_entitlements_managed_credential_ref',
          columns: ['managed_credential_ref'],
          predicate: 'managed_credential_ref IS NOT NULL',
        },
      ],
      indexed: ['status + updated_at', 'expires_at', 'issue_ref'],
      stateInvariants: {
        active:
          'requires managed_credential_ref, issued_at, expires_at, and delivered_at',
        delivery_pending:
          'requires managed_credential_ref, issued_at, and expires_at; delivered_at remains NULL until client ACK',
        cleanup_required: 'requires managed_credential_ref',
        issuing:
          'may be stale-reclaimed only when managed_credential_ref is NULL; issuing with a credential ref requires cleanup/remediation',
        revoked: 'blocks automatic reissue',
      },
      staleIssuingPolicy: {
        ttlMinutes: 15,
        withoutManagedCredentialRef:
          'eligible for same-subject release/reclaim by a later valid request after TTL',
        withManagedCredentialRef:
          'cleanup/remediation candidate; must not be silently overwritten',
      },
      subjectRefPolicy: {
        prefix: 'ph-qq-subject-v1_',
        hmacSecretBinding: 'QQ_AUTH_HMAC_PSK',
        rotationGuardrail:
          'production QQ_AUTH_HMAC_PSK replacement requires a versioned subject-ref rotation plan with dual lookup/backfill semantics; simple secret replacement is not allowed',
      },
      rawIdentityStorage: false,
      rawCredentialStorage: false,
      rawOpenRouterKeyStorage: false,
    });
  });

  it('migrates the QQ managed entitlement table with required indexes', () => {
    const env = createTestBrokerEnv();

    const columns = env.__db
      .prepare("SELECT name FROM pragma_table_info('qq_managed_entitlements') ORDER BY cid")
      .all() as Array<{ name: string }>;
    expect(columns.map((column) => column.name)).toEqual(QQ_MANAGED_ENTITLEMENT_COLUMNS);

    const indexRows = env.__db
      .prepare("SELECT name FROM pragma_index_list('qq_managed_entitlements') ORDER BY name")
      .all() as Array<{ name: string }>;
    expect(indexRows.map((index) => index.name)).toEqual(
      expect.arrayContaining([
        'idx_qq_managed_entitlements_expires_at',
        'idx_qq_managed_entitlements_issue_ref',
        'idx_qq_managed_entitlements_managed_credential_ref',
        'idx_qq_managed_entitlements_status_updated_at',
      ]),
    );
  });

  it('enforces QQ managed entitlement state invariants and unique operational refs', () => {
    const env = createTestBrokerEnv();
    const insertEntitlement = env.__db.prepare(
      `INSERT INTO qq_managed_entitlements (
        qq_subject_ref,
        status,
        issue_ref,
        managed_credential_ref,
        budget_usd,
        reserved_at,
        issued_at,
        expires_at,
        delivered_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    );

    expect(() =>
      insertEntitlement.run(
        'ph-qq-subject-v1_active-missing-delivery',
        'active',
        'qq-issue-active-missing-delivery',
        'managed-credential-active-missing-delivery',
        0.07,
        '2026-06-10T10:00:00.000Z',
        '2026-06-10T10:01:00.000Z',
        '2026-09-10T10:01:00.000Z',
        null,
      ),
    ).toThrow(/constraint/i);

    expect(() =>
      insertEntitlement.run(
        'ph-qq-subject-v1_cleanup-missing-key',
        'cleanup_required',
        'qq-issue-cleanup-missing-key',
        null,
        0.07,
        '2026-06-10T10:00:00.000Z',
        null,
        null,
        null,
      ),
    ).toThrow(/constraint/i);

    insertEntitlement.run(
      'ph-qq-subject-v1_issuing-no-key',
      'issuing',
      'qq-issue-issuing-no-key',
      null,
      0.07,
      '2026-06-10T10:00:00.000Z',
      null,
      null,
      null,
    );
    insertEntitlement.run(
      'ph-qq-subject-v1_issuing-with-key-remediation',
      'issuing',
      'qq-issue-issuing-with-key-remediation',
      'managed-credential-remediation-candidate',
      0.07,
      '2026-06-10T10:00:00.000Z',
      null,
      null,
      null,
    );

    expect(() =>
      insertEntitlement.run(
        'ph-qq-subject-v1_duplicate-issue-ref',
        'issuing',
        'qq-issue-issuing-no-key',
        null,
        0.07,
        '2026-06-10T10:00:00.000Z',
        null,
        null,
        null,
      ),
    ).toThrow(/unique|constraint/i);

    expect(() =>
      insertEntitlement.run(
        'ph-qq-subject-v1_duplicate-credential-ref',
        'active',
        'qq-issue-duplicate-credential-ref',
        'managed-credential-remediation-candidate',
        0.07,
        '2026-06-10T10:00:00.000Z',
        '2026-06-10T10:01:00.000Z',
        '2026-09-10T10:01:00.000Z',
        '2026-06-10T10:02:00.000Z',
      ),
    ).toThrow(/unique|constraint/i);

    expect(() =>
      insertEntitlement.run(
        'ph-qq-subject-v1xmissing-literal-separator',
        'issuing',
        'qq-issue-invalid-subject-prefix',
        null,
        0.07,
        '2026-06-10T10:00:00.000Z',
        null,
        null,
        null,
      ),
    ).toThrow(/constraint/i);
  });
});

describe('managed key delivery ACK schema', () => {
  it('documents ACK delivery storage without plaintext ACK tokens', () => {
    const contract = BROKER_PERSISTENCE_MODEL.tables as Record<string, unknown>;

    expect(contract.managedKeyDeliveries).toMatchObject({
      name: 'managed_key_deliveries',
      primaryKey: 'delivery_id',
      issueSources: ['discord', 'qq'],
      storedStatuses: ['pending', 'acknowledged', 'expired', 'cleanup_required'],
      columns: MANAGED_KEY_DELIVERY_COLUMNS,
      rawOpenRouterKeyStorage: false,
      rawAckTokenStorage: false,
    });
  });

  it('migrates ACK delivery table with required indexes', () => {
    const env = createTestBrokerEnv();

    const columns = env.__db
      .prepare("SELECT name FROM pragma_table_info('managed_key_deliveries') ORDER BY cid")
      .all() as Array<{ name: string }>;
    expect(columns.map((column) => column.name)).toEqual(MANAGED_KEY_DELIVERY_COLUMNS);

    const indexRows = env.__db
      .prepare("SELECT name FROM pragma_index_list('managed_key_deliveries') ORDER BY name")
      .all() as Array<{ name: string }>;
    expect(indexRows.map((index) => index.name)).toEqual(
      expect.arrayContaining([
        'idx_managed_key_deliveries_issue_source_created_at',
        'idx_managed_key_deliveries_managed_credential_ref',
        'idx_managed_key_deliveries_status_expires_at',
      ]),
    );
  });
});
