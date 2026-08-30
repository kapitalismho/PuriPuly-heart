import { existsSync, readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';

import { describe, expect, it } from 'vitest';

import app from '../src/index';
import {
  TEST_DEFAULT_ABUSE_CONTROLS,
  TEST_DEFAULT_ABUSE_RUNTIME_STATE,
} from './test-support/abuse-controls';
import {
  BROKER_MIGRATION_FILENAMES,
  FIRST_BROKER_MIGRATION,
  LATEST_BROKER_MIGRATION,
  applyBrokerMigrations,
  readBrokerMigrationSql,
} from './test-support/migrations';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

const DAILY_SUMMARY_V2_FINALIZER = new URL(
  '../deploy/finalize-daily-summary-v2.sql',
  import.meta.url,
);
const APP_ACTIVE_DAY_FINALIZER = new URL(
  '../deploy/finalize-app-active-day.sql',
  import.meta.url,
);

describe('broker persistent state model', () => {
  it('defines the D1 table contract, runtime config keys, and minimal release-session state', async () => {
    const contract = await import('../src/contract');

    expect(contract).toHaveProperty('BROKER_RUNTIME_CONFIG_KEYS', {
      fingerprintSalt: 'fingerprint_salt',
      abuseControls: 'abuse_controls',
      abuseRuntimeState: 'abuse_runtime_state',
      qqTalkTogetherPass: 'qq_talk_together_pass',
    });
    expect(contract).toHaveProperty('BROKER_RUNTIME_CONFIG_SCHEMA', {
      fingerprint_salt: ['current', 'previous', 'rotated_at'],
      abuse_controls: TEST_DEFAULT_ABUSE_CONTROLS,
      abuse_runtime_state: TEST_DEFAULT_ABUSE_RUNTIME_STATE,
      qq_talk_together_pass: {
        enabled: false,
        rewards_enabled: false,
        daily_warning_count: 30,
        daily_max_count: 50,
      },
    });
    expect(contract).toHaveProperty('BROKER_PUBLIC_INPUT_BOUNDS', {
      installation_id: {
        minLength: 1,
        maxLength: 128,
        rejectWhitespaceOnly: true,
        rejectControlCharacters: true,
        rejectNewlines: true,
      },
      app_version: {
        minLength: 1,
        maxLength: 64,
        rejectWhitespaceOnly: true,
        rejectControlCharacters: true,
        rejectNewlines: true,
      },
      hardware_hash: {
        minLength: 1,
        maxLength: 128,
        nullable: true,
        rejectWhitespaceOnly: true,
        rejectControlCharacters: true,
        rejectNewlines: true,
      },
    });
    expect(contract).toHaveProperty('BROKER_PERSISTENCE_MODEL', {
      database: 'Cloudflare D1',
      tables: {
        brokerConfig: {
          name: 'broker_config',
          primaryKey: 'key',
          columns: ['key', 'value', 'updated_at'],
          valueEncoding: 'JSON',
          supportedKeys: [
            'fingerprint_salt',
            'abuse_controls',
            'abuse_runtime_state',
            'qq_talk_together_pass',
          ],
          constraints: {
            key: 'supported-keys-only',
            value: 'valid-json',
          },
          seedRows: [
            'fingerprint_salt',
            'abuse_controls',
            'abuse_runtime_state',
            'qq_talk_together_pass',
          ],
        },
        installations: {
          name: 'installations',
          primaryKey: 'installation_id',
          columns: [
            'installation_id',
            'device_public_key',
            'hardware_hash',
            'hardware_hash_salt_version',
            'app_version',
            'challenge',
            'challenge_expires_at',
            'challenge_salt_version',
            'created_at',
            'last_seen_at',
          ],
          unique: ['device_public_key'],
          indexed: [
            'hardware_hash',
            'hardware_hash_salt_version',
            'challenge_expires_at',
            'last_seen_at',
          ],
          textBounds: {
            installation_id: {
              minLength: 1,
              maxLength: 128,
              rejectWhitespaceOnly: true,
              rejectControlCharacters: true,
              rejectNewlines: true,
            },
            app_version: {
              minLength: 1,
              maxLength: 64,
              rejectWhitespaceOnly: true,
              rejectControlCharacters: true,
              rejectNewlines: true,
            },
            hardware_hash: {
              minLength: 1,
              maxLength: 128,
              nullable: true,
              rejectWhitespaceOnly: true,
              rejectControlCharacters: true,
              rejectNewlines: true,
            },
          },
          updateRules: {
            onChallenge: [
              'overwrite challenge',
              'overwrite challenge_expires_at',
              'overwrite challenge_salt_version',
              'overwrite app_version',
              'clear hardware_hash and hardware_hash_salt_version only when lifecycle is none or pending_release',
              'preserve hardware_hash state for active, expired, and revoked lifecycles',
              'touch last_seen_at',
            ],
            onVerify: [
              'clear challenge',
              'clear challenge_expires_at',
              'clear challenge_salt_version',
              'persist hardware_hash only after successful verify',
              'persist hardware_hash_salt_version with hardware_hash',
            ],
            beforeVerify: ['hardware_hash stays null until verify'],
          },
        },
        openrouterEntitlements: {
          name: 'openrouter_entitlements',
          provider: 'OpenRouter',
          rowCardinality: 'zero-or-one-row-per-installation',
          primaryKey: 'installation_id',
          absenceRepresents: 'none',
          storedStatuses: ['pending_release', 'active', 'expired', 'revoked'],
          discordIssueStatuses: [
            'issuing',
            'delivery_pending',
            'active',
            'failed',
            'cleanup_required',
          ],
          columns: [
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
          ],
          unique: ['managed_credential_ref', 'discord_user_ref'],
          indexed: ['status', 'expires_at', 'discord_issue_reserved_at'],
          partialUniqueIndexes: [
            {
              name: 'idx_openrouter_entitlements_release_token_hash',
              columns: ['release_token_hash'],
              predicate: 'release_token_hash IS NOT NULL',
            },
            {
              name: 'idx_openrouter_entitlements_discord_user_ref',
              columns: ['discord_user_ref'],
              predicate: 'discord_user_ref IS NOT NULL',
            },
          ],
          updateStrategy: 'in-place',
          liveRemainingBudgetSource: 'OpenRouter metadata',
          releaseSessionState: {
            storage: 'ephemeral-columns-on-openrouter_entitlements',
            fields: [
              'release_session_ref',
              'release_token_hash',
              'release_token_expires_at',
            ],
            releaseToken: {
              binding: 'installation-bound',
              oneTimeUse: true,
              ttlMinutes: 15,
              issuanceIdempotencyKey: 'installation_identity + release_session_ref',
              verifyBehavior: 'rotate for existing pending_release row',
            },
          },
        },
        discordOAuthSessions: {
          name: 'discord_oauth_sessions',
          purpose:
            'bounded OAuth PKCE/session state for Discord-gated managed OpenRouter issuance',
          primaryKey: 'state_hash',
          columns: [
            'state_hash',
            'installation_id',
            'device_public_key',
            'redirect_uri',
            'pkce_code_verifier',
            'issue_nonce_hash',
            'fingerprint_salt_version',
            'discord_user_ref',
            'discord_email_verified',
            'discord_account_created_at',
            'eligibility_checked_at',
            'status',
            'created_at',
            'expires_at',
            'processing_started_at',
            'consumed_at',
            'referral_id',
          ],
          storedStatuses: [
            'pending',
            'processing',
            'consumed',
            'canceled',
            'failed',
            'expired',
          ],
          retention:
            'expires_at cleanup only; durable entitlement and identity evidence is separate',
          indexed: ['installation_id + status + created_at', 'expires_at', 'referral_id'],
        },
        referralCodes: {
          name: 'referral_codes',
          purpose: 'stable owned global Referral ID per managed source subject',
          primaryKey: 'referral_id',
          columns: [
            'referral_id',
            'owner_source',
            'owner_subject_ref',
            'owner_installation_id',
            'status',
            'created_at',
            'updated_at',
            'disabled_reason',
            'disabled_by',
            'disabled_at',
          ],
          referralIdFormat:
            'six uppercase approved-alphabet characters excluding 0/O/1/I/L',
          storedStatuses: ['active', 'disabled'],
          ownerSources: ['discord', 'qq'],
          unique: ['owner_source + owner_subject_ref'],
          indexed: [
            'owner_source + owner_subject_ref',
            'owner_installation_id',
            'status + referral_id',
          ],
          deletionBehavior:
            'installation aging must not cascade-delete referral code history',
        },
        referralRewards: {
          name: 'referral_rewards',
          purpose: 'global append-only source-aware referral attempt and reward ledger',
          primaryKey: 'id',
          columns: [
            'id',
            'referral_id',
            'referrer_source',
            'referrer_subject_ref',
            'referrer_installation_id',
            'referred_source',
            'referred_subject_ref',
            'referred_installation_id',
            'referred_hardware_hash',
            'referred_hardware_hash_salt_version',
            'referred_bonus_status',
            'referrer_bonus_status',
            'skip_reason',
            'failure_reason',
            'referred_managed_credential_ref',
            'referrer_managed_credential_ref',
            'created_at',
            'updated_at',
            'credited_at',
            'attempt_ip_hash',
          ],
          referralIdFormat:
            'six uppercase approved-alphabet characters excluding 0/O/1/I/L',
          subjectSources: ['discord', 'qq'],
          referredBonusStatuses: ['reserved', 'credited', 'skipped', 'failed'],
          referrerBonusStatuses: ['pending', 'applying', 'credited', 'skipped', 'failed'],
          reasonBounds: {
            skip_reason: '1-64 chars when present',
            failure_reason: '1-64 chars when present',
          },
          indexed: [
            'referral_id',
            'referrer_source + referrer_subject_ref + referred_bonus_status',
            'referred_source + referred_subject_ref + created_at',
            'referred_installation_id + created_at',
            'attempt_ip_hash + created_at',
            'referral_id + created_at',
            'referrer_source + referrer_subject_ref + created_at',
          ],
          partialUniqueIndexes: [
            {
              name: 'idx_referral_rewards_counted_referred_subject',
              columns: ['referred_source', 'referred_subject_ref'],
              predicate: "referred_bonus_status IN ('reserved', 'credited')",
            },
            {
              name: 'idx_referral_rewards_counted_referred_installation',
              columns: ['referred_installation_id'],
              predicate:
                "referred_installation_id IS NOT NULL AND referred_bonus_status IN ('reserved', 'credited')",
            },
          ],
          sourceShape:
            'Discord referred rows require installation and hardware evidence; QQ referred rows prohibit Discord hardware fields',
          deletionBehavior:
            'installation aging must not cascade-delete referral reward ledger history',
        },
        discordIdentities: {
          name: 'discord_identities',
          purpose: 'durable HMAC Discord user reference uniqueness for managed issuance',
          primaryKey: 'discord_user_ref',
          columns: [
            'discord_user_ref',
            'entitlement_installation_id',
            'status',
            'ref_secret_version',
            'created_at',
            'updated_at',
          ],
          storedStatuses: ['issuing', 'active', 'failed', 'cleanup_required'],
          foreignKeys: ['entitlement_installation_id -> installations.installation_id'],
        },
        qqAuthAssertions: {
          name: 'qq_auth_assertions',
          purpose:
            'durable anonymized QQ Bot HMAC assertion evidence for verification-only compatibility and production issuance eligibility',
          primaryKey: 'qq_subject_ref',
          columns: [
            'qq_subject_ref',
            'credential_hash',
            'asserted_at',
            'received_at',
            'status',
          ],
          storedStatuses: ['verified'],
          rawIdentityStorage: false,
          duplicateHandling: 'preserve original row; duplicate assertions are idempotent',
        },
        qqManagedEntitlements: {
          name: 'qq_managed_entitlements',
          purpose:
            'durable QQ Managed production issuance lifecycle keyed by stable subject reference',
          primaryKey: 'qq_subject_ref',
          lifecycleDecisionSource: 'qq_managed_entitlements, not qq_auth_assertions',
          rowCardinality: 'zero-or-one-row-per-qq_subject_ref',
          absenceRepresents: 'no production issuance has been reserved or used',
          storedStatuses: ['issuing', 'delivery_pending', 'active', 'cleanup_required', 'revoked'],
          automaticReissueBlockedStatuses: ['active', 'cleanup_required', 'revoked'],
          columns: [
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
          ],
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
        },
        managedKeyDeliveries: {
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
          ],
          indexed: [
            'status + expires_at',
            'managed_credential_ref',
            'issue_source + created_at',
          ],
          rawAckTokenStorage: false,
          rawOpenRouterKeyStorage: false,
          stalePendingCleanup:
            'expired rows are claimed exclusively; abandoned claims recover only after the scheduled invocation limit, and terminal owner/ledger transitions are atomic',
        },
        qqPassSettlementJobs: {
          name: 'qq_pass_settlement_jobs',
          purpose:
            'durable fenced QQ invitee/referrer reward settlement work keyed by referral reward and acknowledged delivery',
          primaryKey: 'id',
          columns: [
            'id',
            'referral_reward_id',
            'delivery_id',
            'phase',
            'attempt_count',
            'last_attempt_at',
            'next_attempt_at',
            'fencing_token',
            'lease_expires_at',
            'last_error_code',
            'created_at',
            'updated_at',
            'completed_at',
          ],
          phases: ['invitee_pending', 'referrer_pending', 'completed'],
          unique: [
            'referral_reward_id',
            'delivery_id',
            'fencing_token when claimed',
          ],
          indexed: ['phase + next_attempt_at + lease_expires_at'],
          noRetention: true,
          noCascade: true,
          fencing:
            'every claim, transition, release, and completion mutation requires the exact fencing_token',
        },
        brokerRequestEvents: {
          name: 'broker_request_events',
          purpose: ['per-endpoint rate limits', 'cross-endpoint velocity hooks'],
          columns: ['id', 'endpoint', 'ip', 'installation_id', 'observed_at'],
          appendOnly: true,
          indexed: [
            'endpoint + ip + observed_at',
            'endpoint + installation_id + observed_at',
            'ip + observed_at',
            'installation_id + observed_at',
          ],
        },
        brokerIssueSuccessEvents: {
          name: 'broker_issue_success_events',
          purpose: ['issuance spike detection', 'daily reporting'],
          issueSources: ['discord', 'qq'],
          sourceAwareSubjectModel: {
            discord: {
              issue_source: 'discord',
              installation_id: 'required existing installation identity',
              subject_ref: 'same value as installation_id',
            },
            qq: {
              issue_source: 'qq',
              installation_id: null,
              subject_ref: 'qq_subject_ref',
            },
            fakeInstallationRowsAllowed: false,
          },
          columns: [
            'id',
            'issue_source',
            'installation_id',
            'subject_ref',
            'managed_credential_ref',
            'ip_hash',
            'ip_prefix_hash',
            'asn',
            'country',
            'http_protocol',
            'tls_version',
            'tls_cipher',
            'risk_label',
            'observed_at',
          ],
          appendOnly: true,
          indexed: [
            'installation_id + observed_at',
            'issue_source + subject_ref + observed_at',
            'managed_credential_ref + observed_at',
            'ip_hash + observed_at',
            'ip_prefix_hash + observed_at',
            'asn + observed_at',
            'observed_at',
          ],
        },
        telemetrySubjects: {
          name: 'telemetry_subjects',
          purpose:
            'legacy translation-success subject bounds preserved but unused by app usage aggregation',
          primaryKey: 'subject_ref',
          columns: ['subject_ref', 'first_active_date_utc', 'last_active_date_utc'],
          indexed: ['last_active_date_utc'],
          rawTelemetryIdentifierStorage: false,
          joinedToManagedIdentity: false,
        },
        telemetryActiveDays: {
          name: 'telemetry_active_days',
          purpose:
            'legacy translation-success dates preserved but unused by app usage aggregation',
          primaryKey: ['subject_ref', 'active_date_utc'],
          columns: [
            'subject_ref',
            'active_date_utc',
            'first_received_at',
            'last_received_at',
          ],
          indexed: ['active_date_utc', 'last_received_at'],
          rawTelemetryIdentifierStorage: false,
          joinedToManagedIdentity: false,
        },
        appActiveDays: {
          name: 'app_active_days',
          purpose: 'retained anonymous app-launch dates for completed-day usage aggregation',
          primaryKey: ['subject_ref', 'active_date_utc'],
          columns: ['subject_ref', 'active_date_utc'],
          indexed: ['active_date_utc'],
          rawTelemetryIdentifierStorage: false,
          joinedToManagedIdentity: false,
        },
        brokerDailySummaryDeliveries: {
          name: 'broker_daily_summary_deliveries',
          purpose: 'v2 completed-day delivery leases and durable delivery outcomes',
          primaryKey: 'report_date_utc',
          columns: [
            'report_date_utc',
            'status',
            'lease_token',
            'lease_expires_at',
            'attempted_at',
            'delivered_at',
          ],
          indexed: ['status + report_date_utc + lease_expires_at'],
        },
        brokerAbuseRuntimeAudit: {
          name: 'broker_abuse_runtime_audit',
          purpose:
            'append-only audit trail for runtime-state changes and abuse-monitoring decisions',
          columns: ['id', 'event_kind', 'reason', 'payload_json', 'created_at'],
          appendOnly: true,
          indexed: ['event_kind + created_at', 'created_at'],
        },
        brokerVelocityCapHooks: {
          name: 'broker_velocity_cap_hooks',
          purpose: 'manual cross-endpoint velocity controls with observable outcomes',
          columns: [
            'id',
            'subject_type',
            'subject_value',
            'max_requests',
            'window_minutes',
            'outcome_code',
            'outcome_class',
            'outcome_subcode',
            'reason',
            'active',
            'created_at',
            'expires_at',
          ],
          supportedSubjects: ['ip', 'installation_id'],
          indexed: ['subject_type + subject_value + active + expires_at'],
        },
        brokerAbuseSubjectHooks: {
          name: 'broker_abuse_subject_hooks',
          purpose:
            'denylist, reputation, and fast-revocation controls with observable outcomes',
          columns: [
            'id',
            'hook_kind',
            'subject_type',
            'subject_value',
            'outcome_code',
            'outcome_class',
            'outcome_subcode',
            'reason',
            'active',
            'created_at',
            'expires_at',
          ],
          hookKinds: ['denylist', 'reputation', 'revocation'],
          supportedSubjects: ['ip', 'installation_id', 'hardware_hash'],
          indexed: ['subject_type + subject_value + hook_kind + active + expires_at'],
        },
      },
    });
  });

  it('keeps persistence details out of the public foundation response', async () => {
    const response = await app.request('http://broker.test/v1/foundation');
    expect(response.status).toBe(200);

    const payload = (await response.json()) as Record<string, unknown>;

    expect(payload).not.toHaveProperty('persistence');
    expect(payload).not.toHaveProperty('brokerPersistenceModel');
    expect(payload).not.toHaveProperty('runtimeConfig');
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

  it('ships a first D1 migration that creates the documented tables and indexes', () => {
    expect(BROKER_MIGRATION_FILENAMES).toEqual([
      '0000_define_broker_persistent_state.sql',
      '0001_add_abuse_hook_state.sql',
      '0001_harden_installation_public_inputs.sql',
      '0002_add_entitlement_verified_hardware_snapshot.sql',
      '0003_add_abuse_runtime_state_and_issue_success_events.sql',
      '0004_add_discord_oauth_managed_issue.sql',
      '0005_add_referral_persistence_foundation.sql',
      '0006_harden_referral_reward_operations.sql',
      '0007_simplify_referral_id_checks.sql',
      '0008_add_qq_auth_assertions.sql',
      '0009_add_qq_managed_entitlements.sql',
      '0010_source_aware_issue_success_events.sql',
      '0011_add_telemetry_active_days.sql',
      '0012_add_managed_key_delivery_ack.sql',
      '0013_add_telemetry_subjects_and_daily_summary_v2.sql',
      '0014_simplify_abuse_incidents.sql',
      '0015_add_app_active_days.sql',
      '0016_make_referrals_source_aware.sql',
      '0017_add_qq_pass_settlement_jobs.sql',
    ]);
    expect(existsSync(FIRST_BROKER_MIGRATION)).toBe(true);
    expect(existsSync(LATEST_BROKER_MIGRATION)).toBe(true);
    if (!existsSync(FIRST_BROKER_MIGRATION) || !existsSync(LATEST_BROKER_MIGRATION)) {
      return;
    }

    const migration = readFileSync(FIRST_BROKER_MIGRATION, 'utf8');
    const abuseHooksMigration = readBrokerMigrationSql(
      '0001_add_abuse_hook_state.sql',
    );
    const hardeningMigration = readBrokerMigrationSql(
      '0001_harden_installation_public_inputs.sql',
    );
    const verifiedSnapshotMigration = readBrokerMigrationSql(
      '0002_add_entitlement_verified_hardware_snapshot.sql',
    );
    const abuseRuntimeMigration = readBrokerMigrationSql(
      '0003_add_abuse_runtime_state_and_issue_success_events.sql',
    );
    const discordManagedIssueMigration = readBrokerMigrationSql(
      '0004_add_discord_oauth_managed_issue.sql',
    );
    const referralPersistenceMigration = readBrokerMigrationSql(
      '0005_add_referral_persistence_foundation.sql',
    );
    const referralOperationsMigration = readBrokerMigrationSql(
      '0006_harden_referral_reward_operations.sql',
    );
    const referralCheckRepairMigration = readBrokerMigrationSql(
      '0007_simplify_referral_id_checks.sql',
    );
    const qqAuthAssertionsMigration = readBrokerMigrationSql(
      '0008_add_qq_auth_assertions.sql',
    );
    const qqManagedEntitlementsMigration = readBrokerMigrationSql(
      '0009_add_qq_managed_entitlements.sql',
    );
    const sourceAwareIssueSuccessMigration = readBrokerMigrationSql(
      '0010_source_aware_issue_success_events.sql',
    );
    const telemetryActiveDaysMigration = readBrokerMigrationSql(
      '0011_add_telemetry_active_days.sql',
    );
    const managedKeyDeliveryAckMigration = readBrokerMigrationSql(
      '0012_add_managed_key_delivery_ack.sql',
    );
    const dailySummaryV2Migration = readBrokerMigrationSql(
      '0013_add_telemetry_subjects_and_daily_summary_v2.sql',
    );
    const simplifiedAbuseIncidentsMigration = readBrokerMigrationSql(
      '0014_simplify_abuse_incidents.sql',
    );
    const appActiveDaysMigration = readBrokerMigrationSql(
      '0015_add_app_active_days.sql',
    );
    const dailySummaryV2Finalizer = readFileSync(
      DAILY_SUMMARY_V2_FINALIZER,
      'utf8',
    );
    const appActiveDayFinalizer = readFileSync(APP_ACTIVE_DAY_FINALIZER, 'utf8');

    expect(migration).toContain('CREATE TABLE broker_config');
    expect(migration).toContain('CREATE TABLE installations');
    expect(migration).toContain('CREATE TABLE openrouter_entitlements');
    expect(migration).toContain('device_public_key TEXT NOT NULL UNIQUE');
    expect(migration).toContain('hardware_hash TEXT');
    expect(migration).toContain('hardware_hash_salt_version INTEGER');
    expect(migration).toContain('challenge TEXT');
    expect(migration).toContain('challenge_expires_at TEXT');
    expect(migration).toContain('challenge_salt_version INTEGER');
    expect(migration).toContain('CHECK (length(installation_id) BETWEEN 1 AND 128)');
    expect(migration).toContain('CHECK (length(app_version) BETWEEN 1 AND 64)');
    expect(migration).toContain(
      'CHECK (hardware_hash IS NULL OR length(hardware_hash) BETWEEN 1 AND 128)',
    );
    expect(migration).toContain("INSERT INTO broker_config (key, value)");
    expect(migration).toContain("'abuse_controls'");
    expect(migration).toContain("CHECK(status IN ('pending_release', 'active', 'expired', 'revoked'))");
    expect(migration).toContain('managed_credential_ref TEXT UNIQUE');
    expect(migration).toContain('release_session_ref TEXT');
    expect(migration).toContain('release_token_hash TEXT');
    expect(migration).toContain('release_token_expires_at TEXT');
    expect(migration).not.toContain('verified_hardware_hash TEXT');
    expect(migration).not.toContain('verified_hardware_hash_salt_version INTEGER');
    expect(migration).toContain('CREATE INDEX idx_installations_hardware_hash');
    expect(migration).toContain('CREATE INDEX idx_installations_hardware_hash_salt_version');
    expect(migration).toContain('CREATE INDEX idx_installations_challenge_expires_at');
    expect(migration).toContain('CREATE INDEX idx_installations_last_seen_at');
    expect(migration).toContain('CREATE INDEX idx_openrouter_entitlements_status');
    expect(migration).toContain('CREATE INDEX idx_openrouter_entitlements_expires_at');
    expect(abuseHooksMigration).toContain('CREATE TABLE broker_request_events');
    expect(abuseHooksMigration).toContain('CREATE TABLE broker_velocity_cap_hooks');
    expect(abuseHooksMigration).toContain('CREATE TABLE broker_abuse_subject_hooks');
    expect(hardeningMigration).toContain('PRAGMA defer_foreign_keys = on');
    expect(hardeningMigration).toContain('CREATE TABLE installations_hardened');
    expect(hardeningMigration).toContain('CREATE TABLE openrouter_entitlements_hardened');
    expect(hardeningMigration).toContain('INSERT INTO installations_hardened');
    expect(hardeningMigration).toContain('INSERT INTO openrouter_entitlements_hardened');
    expect(hardeningMigration).toContain('DROP TABLE openrouter_entitlements;');
    expect(hardeningMigration).toContain('ALTER TABLE installations_hardened RENAME TO installations');
    expect(hardeningMigration).toContain('PRAGMA foreign_key_check');
    expect(verifiedSnapshotMigration).toContain('ALTER TABLE openrouter_entitlements');
    expect(verifiedSnapshotMigration).toContain('verified_hardware_hash TEXT');
    expect(verifiedSnapshotMigration).toContain(
      'verified_hardware_hash_salt_version INTEGER',
    );
    expect(abuseRuntimeMigration).toContain('CREATE TABLE broker_config_v2');
    expect(abuseRuntimeMigration).toContain('abuse_runtime_state');
    expect(abuseRuntimeMigration).toContain('CREATE TABLE broker_issue_success_events');
    expect(abuseRuntimeMigration).toContain('managed_credential_ref TEXT');
    expect(abuseRuntimeMigration).toContain('ip_hash TEXT');
    expect(abuseRuntimeMigration).toContain('ip_prefix_hash TEXT');
    expect(abuseRuntimeMigration).toContain('country TEXT');
    expect(abuseRuntimeMigration).toContain('http_protocol TEXT');
    expect(abuseRuntimeMigration).toContain('tls_version TEXT');
    expect(abuseRuntimeMigration).toContain('tls_cipher TEXT');
    expect(abuseRuntimeMigration).toContain('risk_label TEXT');
    expect(abuseRuntimeMigration).toContain('CREATE TABLE broker_abuse_runtime_audit');
    expect(abuseRuntimeMigration).toContain('payload_json TEXT NOT NULL CHECK (json_valid(payload_json))');
    expect(abuseRuntimeMigration).toContain('created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP');
    expect(discordManagedIssueMigration).toContain('CREATE TABLE discord_oauth_sessions');
    expect(discordManagedIssueMigration).toContain('CREATE TABLE discord_identities');
    expect(discordManagedIssueMigration).toContain('discord_user_ref TEXT');
    expect(discordManagedIssueMigration).toContain('discord_issue_status TEXT');
    expect(discordManagedIssueMigration).toContain(
      'CREATE UNIQUE INDEX idx_openrouter_entitlements_discord_user_ref',
    );
    expect(discordManagedIssueMigration).toContain(
      'POST /v1/providers/openrouter/discord/issue',
    );
    expect(discordManagedIssueMigration).not.toContain('legacy_installation_id_mapping');
    expect(discordManagedIssueMigration).not.toContain('legacy-invalid-app-version');
    expect(referralPersistenceMigration).toContain('CREATE TABLE referral_codes');
    expect(referralPersistenceMigration).toContain('CREATE TABLE referral_rewards');
    expect(referralPersistenceMigration).toContain('ADD COLUMN referral_id TEXT');
    expect(referralPersistenceMigration).toContain(
      'CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_discord_user',
    );
    expect(referralPersistenceMigration).toContain(
      'CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_installation',
    );
    expect(referralPersistenceMigration).not.toContain('ON DELETE CASCADE');
    expect(referralOperationsMigration).toContain('ADD COLUMN disabled_reason TEXT');
    expect(referralOperationsMigration).toContain('ADD COLUMN disabled_by TEXT');
    expect(referralOperationsMigration).toContain('ADD COLUMN disabled_at TEXT');
    expect(referralOperationsMigration).toContain('ADD COLUMN attempt_ip_hash TEXT');
    expect(referralOperationsMigration).toContain(
      'CREATE INDEX idx_referral_rewards_attempt_installation_time',
    );
    expect(referralOperationsMigration).toContain(
      'CREATE INDEX idx_referral_rewards_attempt_ip_hash_time',
    );
    expect(referralOperationsMigration).toContain(
      'CREATE INDEX idx_referral_rewards_referral_velocity',
    );
    expect(referralOperationsMigration).toContain(
      'CREATE INDEX idx_referral_rewards_referrer_velocity',
    );
    expect(referralOperationsMigration).toContain('$.retention.referralSkippedDays');
    expect(referralOperationsMigration).toContain('$.retention.referralFailedDays');
    expect(referralOperationsMigration).toContain('$.referralAttempts');
    expect(referralCheckRepairMigration).toContain('PRAGMA defer_foreign_keys = on');
    expect(referralCheckRepairMigration).toContain(
      'CREATE TABLE discord_oauth_sessions_referral_id_checks_v2',
    );
    expect(referralCheckRepairMigration).toContain(
      'CREATE TABLE referral_codes_referral_id_checks_v2',
    );
    expect(referralCheckRepairMigration).toContain(
      'CREATE TABLE referral_rewards_referral_id_checks_v2',
    );
    expect(referralCheckRepairMigration).toContain(
      "AND referral_id NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'",
    );
    expect(referralCheckRepairMigration).toContain('PRAGMA foreign_key_check');
    expect(referralCheckRepairMigration).not.toContain('PRAGMA foreign_keys = OFF');
    expect(referralCheckRepairMigration).not.toContain('PRAGMA foreign_keys = ON');
    expect(qqAuthAssertionsMigration).toContain('CREATE TABLE qq_auth_assertions');
    expect(qqAuthAssertionsMigration).toContain('qq_subject_ref TEXT PRIMARY KEY');
    expect(qqAuthAssertionsMigration).toContain('credential_hash TEXT NOT NULL');
    expect(qqAuthAssertionsMigration).toContain('asserted_at TEXT NOT NULL');
    expect(qqAuthAssertionsMigration).toContain(
      'received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP',
    );
    expect(qqAuthAssertionsMigration).toContain(
      "status TEXT NOT NULL CHECK(status IN ('verified'))",
    );
    expect(qqAuthAssertionsMigration).toContain('json_insert');
    expect(qqAuthAssertionsMigration).toContain('$.qqAuthAssertIp');
    expect(qqAuthAssertionsMigration).toContain('POST /v1/auth/qq/assert');
    expect(qqAuthAssertionsMigration).not.toContain('json_set');
    expect(qqManagedEntitlementsMigration).toContain(
      'CREATE TABLE qq_managed_entitlements',
    );
    expect(qqManagedEntitlementsMigration).toContain('qq_subject_ref TEXT PRIMARY KEY');
    expect(qqManagedEntitlementsMigration).toContain(
      "qq_subject_ref GLOB 'ph-qq-subject-v1_*'",
    );
    expect(qqManagedEntitlementsMigration).toContain(
      "status TEXT NOT NULL CHECK(status IN ('issuing', 'active', 'cleanup_required', 'revoked'))",
    );
    expect(qqManagedEntitlementsMigration).toContain('issue_ref TEXT NOT NULL');
    expect(qqManagedEntitlementsMigration).toContain('managed_credential_ref TEXT');
    expect(qqManagedEntitlementsMigration).toContain('budget_usd REAL NOT NULL CHECK (budget_usd >= 0)');
    expect(qqManagedEntitlementsMigration).toContain('reserved_at TEXT NOT NULL');
    expect(qqManagedEntitlementsMigration).toContain('delivered_at TEXT');
    expect(qqManagedEntitlementsMigration).toContain(
      'CREATE UNIQUE INDEX idx_qq_managed_entitlements_issue_ref',
    );
    expect(qqManagedEntitlementsMigration).toContain(
      'CREATE UNIQUE INDEX idx_qq_managed_entitlements_managed_credential_ref',
    );
    expect(qqManagedEntitlementsMigration).toContain(
      'CREATE INDEX idx_qq_managed_entitlements_status_updated_at',
    );
    expect(qqManagedEntitlementsMigration).toContain(
      'CREATE INDEX idx_qq_managed_entitlements_expires_at',
    );
    expect(qqManagedEntitlementsMigration).not.toContain('qq_identity');
    expect(qqManagedEntitlementsMigration).not.toContain('credential TEXT');
    expect(qqManagedEntitlementsMigration).not.toContain('openrouter_api_key');
    expect(sourceAwareIssueSuccessMigration).toContain(
      'CREATE TABLE broker_issue_success_events_source_v2',
    );
    expect(sourceAwareIssueSuccessMigration).toContain('issue_source TEXT NOT NULL');
    expect(sourceAwareIssueSuccessMigration).toContain('installation_id TEXT REFERENCES installations(installation_id) ON DELETE CASCADE');
    expect(sourceAwareIssueSuccessMigration).toContain('subject_ref TEXT NOT NULL');
    expect(sourceAwareIssueSuccessMigration).toContain(
      'INSERT INTO broker_issue_success_events_source_v2',
    );
    expect(sourceAwareIssueSuccessMigration).toContain(
      'CREATE INDEX idx_broker_issue_success_events_source_subject_time',
    );
    expect(sourceAwareIssueSuccessMigration).not.toContain('qq_identity');
    expect(sourceAwareIssueSuccessMigration).not.toContain('credential TEXT');
    expect(sourceAwareIssueSuccessMigration).not.toContain('openrouter_api_key');
    expect(telemetryActiveDaysMigration).toContain('CREATE TABLE telemetry_active_days');
    expect(telemetryActiveDaysMigration).toContain('subject_ref TEXT NOT NULL');
    expect(telemetryActiveDaysMigration).toContain('active_date_utc TEXT NOT NULL');
    expect(telemetryActiveDaysMigration).toContain('PRIMARY KEY (subject_ref, active_date_utc)');
    expect(telemetryActiveDaysMigration).toContain('$.telemetryTranslationSuccessDayIp');
    expect(telemetryActiveDaysMigration).toContain('POST /v1/telemetry/translation-success-day');
    expect(telemetryActiveDaysMigration).not.toContain('telemetry_identifier');
    expect(telemetryActiveDaysMigration).not.toContain('translation_text');
    expect(appActiveDaysMigration).toContain('CREATE TABLE app_active_days');
    expect(appActiveDaysMigration).toContain('subject_ref TEXT NOT NULL');
    expect(appActiveDaysMigration).toContain('active_date_utc TEXT NOT NULL');
    expect(appActiveDaysMigration).toContain('PRIMARY KEY (subject_ref, active_date_utc)');
    expect(appActiveDaysMigration).toContain("'ph-app-subject-v1_'");
    expect(appActiveDaysMigration).not.toContain('telemetryTranslationSuccessDayIp');
    expect(appActiveDayFinalizer).toContain(
      "json_remove(value, '$.telemetryTranslationSuccessDayIp')",
    );
    expect(appActiveDaysMigration).not.toContain('anonymous_id');
    expect(appActiveDaysMigration).not.toContain('received_at');
    expect(appActiveDaysMigration).not.toContain('ip TEXT');
    expect(appActiveDaysMigration).not.toContain('metadata');
    expect(dailySummaryV2Migration).toContain('CREATE TABLE telemetry_subjects');
    expect(dailySummaryV2Migration).toContain('MIN(active_date_utc)');
    expect(dailySummaryV2Migration).toContain('MAX(active_date_utc)');
    expect(dailySummaryV2Migration).toContain(
      'CREATE TRIGGER telemetry_active_days_sync_subject_after_insert',
    );
    expect(dailySummaryV2Migration).toContain(
      'CREATE TABLE broker_daily_summary_deliveries',
    );
    expect(dailySummaryV2Migration).toContain("'$.dailyReport.hourUtc'");
    expect(dailySummaryV2Migration).toContain("'$.dailyReport.minuteUtc'");
    expect(dailySummaryV2Migration).not.toContain(
      "'$.dailyReport.includeZeroActivity'",
    );
    expect(dailySummaryV2Finalizer).toContain(
      "'$.dailyReport.includeZeroActivity'",
    );
    expect(dailySummaryV2Migration).toContain("'$.retention.issueSuccessDays'");
    expect(dailySummaryV2Migration).not.toContain('telemetry_identifier');
    expect(simplifiedAbuseIncidentsMigration).toContain(
      "'$.immediateAlerts.warning'",
    );
    expect(simplifiedAbuseIncidentsMigration).toContain(
      "'$.retention.requestEventSafetyMarginDays'",
    );
    expect(managedKeyDeliveryAckMigration).toContain('CREATE TABLE managed_key_deliveries');
    expect(managedKeyDeliveryAckMigration).toContain(
      "discord_issue_status TEXT CHECK(discord_issue_status IS NULL OR discord_issue_status IN ('issuing', 'delivery_pending', 'active', 'failed', 'cleanup_required'))",
    );
    expect(managedKeyDeliveryAckMigration).toContain(
      "status TEXT NOT NULL CHECK(status IN ('issuing', 'delivery_pending', 'active', 'cleanup_required', 'revoked'))",
    );
    expect(managedKeyDeliveryAckMigration).toContain('ack_token_hash TEXT NOT NULL UNIQUE');
    expect(managedKeyDeliveryAckMigration).toContain(
      'CREATE INDEX idx_managed_key_deliveries_status_expires_at',
    );
    expect(managedKeyDeliveryAckMigration).toContain(
      'CREATE INDEX idx_managed_key_deliveries_managed_credential_ref',
    );
    expect(managedKeyDeliveryAckMigration).toContain(
      'CREATE INDEX idx_managed_key_deliveries_issue_source_created_at',
    );
    expect(managedKeyDeliveryAckMigration).not.toContain('openrouter_api_key');
    expect(managedKeyDeliveryAckMigration).not.toContain('delivery_ack_token');
  });

  it('inserts the QQ auth assertion abuse-control default without replacing tuned JSON', () => {
    const db = new DatabaseSync(':memory:');
    try {
      applyBrokerMigrations(db, { through: '0007_simplify_referral_id_checks.sql' });

      const beforeRow = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      const before = JSON.parse(beforeRow.value) as {
        trialChallenge: { maxRequests: number };
        discordAuthStartIp: { maxRequests: number };
      } & Record<string, unknown>;
      before.trialChallenge.maxRequests = 7;
      before.discordAuthStartIp.maxRequests = 17;
      db.prepare('UPDATE broker_config SET value = ? WHERE key = ?').run(
        JSON.stringify(before),
        'abuse_controls',
      );

      applyBrokerMigrations(db, { after: '0007_simplify_referral_id_checks.sql' });

      const afterRow = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      const after = JSON.parse(afterRow.value) as {
        trialChallenge: { maxRequests: number };
        discordAuthStartIp: { maxRequests: number };
        qqAuthAssertIp?: unknown;
      };

      expect(after.trialChallenge.maxRequests).toBe(7);
      expect(after.discordAuthStartIp.maxRequests).toBe(17);
      expect(after.qqAuthAssertIp).toEqual({
        endpoint: 'POST /v1/auth/qq/assert',
        scope: 'ip',
        maxRequests: 20,
        windowMinutes: 15,
      });
    } finally {
      db.close();
    }
  });
});
