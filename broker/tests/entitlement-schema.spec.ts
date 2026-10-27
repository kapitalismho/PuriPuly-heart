import { describe, expect, it } from 'vitest';

import { BROKER_PERSISTENCE_MODEL } from '../src/contract';
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
      automaticReissueBlockedStatuses: ['active', 'cleanup_required', 'revoked'],
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
          'requires managed_credential_ref, issued_at, and expires_at; delivered_at remains null until ACK succeeds',
        cleanup_required: 'requires managed_credential_ref',
        issuing:
          'may be stale-reclaimed only when managed_credential_ref and child_key_creation_started_at are NULL; any started child-key creation requires manual remediation or cleanup',
        revoked: 'blocks automatic reissue',
      },
      staleIssuingPolicy: {
        ttlMinutes: 15,
        withoutManagedCredentialRef:
          'eligible for same-subject release/reclaim by a later valid request after TTL only when child-key creation never started',
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

  it('documents shared managed key delivery ACK persistence', () => {
    expect(BROKER_PERSISTENCE_MODEL.tables.managedKeyDeliveries).toEqual({
      name: 'managed_key_deliveries',
      purpose:
        'shared pending delivery ACK ledger for Discord and QQ managed key issuance',
      primaryKey: 'delivery_id',
      issueSources: ['discord', 'qq'],
      storedStatuses: ['pending', 'acknowledged', 'expired', 'cleanup_required'],
      columns: [
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
        'operation_id',
        'attempt_index',
      ],
      indexed: [
        'status + expires_at',
        'managed_credential_ref',
        'issue_source + created_at',
        'operation_id when bound',
      ],
      operationBinding:
        'deliveries link to their managed operation and attempt for stale-safe cleanup',
      rawAckTokenStorage: false,
      rawOpenRouterKeyStorage: false,
      stalePendingCleanup:
        'expired rows are claimed exclusively; abandoned claims recover only after the scheduled invocation limit, and terminal owner/ledger transitions are atomic',
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
