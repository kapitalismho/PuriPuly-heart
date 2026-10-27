import { describe, expect, it } from 'vitest';

import { createTestBrokerEnv } from './test-support/sqlite-d1';

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

describe('openrouter entitlement schema', () => {
  it('applies the verified hardware snapshot columns to migrated test databases', () => {
    const env = createTestBrokerEnv();
    const columns = env.__db
      .prepare("SELECT name FROM pragma_table_info('openrouter_entitlements') ORDER BY cid")
      .all() as Array<{ name: string }>;

    expect(columns.map((column) => column.name)).toEqual(OPENROUTER_ENTITLEMENT_COLUMNS);
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
  'child_key_creation_started_at',
];

describe('QQ managed entitlement schema', () => {
  it('migrates the QQ managed entitlement table columns', () => {
    const env = createTestBrokerEnv();

    const columns = env.__db
      .prepare("SELECT name FROM pragma_table_info('qq_managed_entitlements') ORDER BY cid")
      .all() as Array<{ name: string }>;
    expect(columns.map((column) => column.name)).toEqual(QQ_MANAGED_ENTITLEMENT_COLUMNS);
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
