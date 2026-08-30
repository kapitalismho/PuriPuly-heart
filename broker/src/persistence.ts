export const BROKER_RUNTIME_CONFIG_KEYS = {
  fingerprintSalt: 'fingerprint_salt',
  abuseControls: 'abuse_controls',
  abuseRuntimeState: 'abuse_runtime_state',
  qqTalkTogetherPass: 'qq_talk_together_pass',
} as const;

export interface BrokerEndpointRateLimitConfig {
  endpoint: string;
  scope: 'ip' | 'installation_id';
  maxRequests: number;
  windowMinutes: number;
}

export type BrokerDailyIssuanceCapEndpoint =
  | 'POST /v1/providers/openrouter/issue'
  | 'POST /v1/providers/openrouter/discord/issue';

export interface BrokerDailyIssuanceCapConfig {
  endpoint: BrokerDailyIssuanceCapEndpoint;
  scope: 'global';
  maxCount: number | null;
  windowDays: number;
}

export interface BrokerPendingDiscordOAuthSessionsConfig {
  maxPerInstallation: number;
  maxPerIp: number;
  windowMinutes: number;
}

export interface BrokerImmediateAlertsConfig {
  warning: number;
  brake: number;
}

export interface BrokerAbuseRetentionConfig {
  requestEventSafetyMarginDays: number;
  issueSuccessDays: number;
  runtimeAuditDays: number;
  referralSkippedDays: number;
  referralFailedDays: number;
}

export interface BrokerReferralAttemptControlsConfig {
  validShaped: {
    maxPerInstallation: number;
    maxPerIp: number;
    windowMinutes: number;
  };
  unknown: {
    maxPerInstallation: number;
    maxPerIp: number;
    windowMinutes: number;
  };
  perReferralIdVelocity: {
    maxAttempts: number;
    windowMinutes: number;
  };
  perReferrerRewardVelocity: {
    maxRewards: number;
    windowMinutes: number;
  };
}

export interface BrokerDailyReportConfig {
  enabled: boolean;
  hourUtc: number;
  minuteUtc: number;
}

export interface BrokerAbuseControlsConfigValue {
  trialChallenge: BrokerEndpointRateLimitConfig;
  trialChallengeVerify: BrokerEndpointRateLimitConfig;
  openrouterIssue: BrokerEndpointRateLimitConfig;
  trialStatus: BrokerEndpointRateLimitConfig;
  discordAuthStartIp: BrokerEndpointRateLimitConfig;
  discordAuthStartInstallation: BrokerEndpointRateLimitConfig;
  discordOpenrouterIssueIp: BrokerEndpointRateLimitConfig;
  discordOpenrouterIssueInstallation: BrokerEndpointRateLimitConfig;
  qqAuthAssertIp: BrokerEndpointRateLimitConfig;
  qqAuthStatusIp: BrokerEndpointRateLimitConfig;
  pendingDiscordOAuthSessions: BrokerPendingDiscordOAuthSessionsConfig;
  newActiveEntitlementsPerDay: BrokerDailyIssuanceCapConfig;
  immediateAlerts: BrokerImmediateAlertsConfig;
  retention: BrokerAbuseRetentionConfig;
  referralAttempts: BrokerReferralAttemptControlsConfig;
  dailyReport: BrokerDailyReportConfig;
}

export const DEFAULT_BROKER_ABUSE_CONTROLS: BrokerAbuseControlsConfigValue = {
  trialChallenge: {
    endpoint: 'POST /v1/trial/challenge',
    scope: 'ip',
    maxRequests: 10,
    windowMinutes: 15,
  },
  trialChallengeVerify: {
    endpoint: 'POST /v1/trial/challenge/verify',
    scope: 'installation_id',
    maxRequests: 5,
    windowMinutes: 15,
  },
  openrouterIssue: {
    endpoint: 'POST /v1/providers/openrouter/issue',
    scope: 'installation_id',
    maxRequests: 3,
    windowMinutes: 15,
  },
  trialStatus: {
    endpoint: 'GET /v1/trial/status',
    scope: 'installation_id',
    maxRequests: 30,
    windowMinutes: 15,
  },
  discordAuthStartIp: {
    endpoint: 'POST /v1/auth/discord/start',
    scope: 'ip',
    maxRequests: 20,
    windowMinutes: 15,
  },
  discordAuthStartInstallation: {
    endpoint: 'POST /v1/auth/discord/start',
    scope: 'installation_id',
    maxRequests: 5,
    windowMinutes: 15,
  },
  discordOpenrouterIssueIp: {
    endpoint: 'POST /v1/providers/openrouter/discord/issue',
    scope: 'ip',
    maxRequests: 10,
    windowMinutes: 15,
  },
  discordOpenrouterIssueInstallation: {
    endpoint: 'POST /v1/providers/openrouter/discord/issue',
    scope: 'installation_id',
    maxRequests: 3,
    windowMinutes: 15,
  },
  qqAuthAssertIp: {
    endpoint: 'POST /v1/auth/qq/assert',
    scope: 'ip',
    maxRequests: 20,
    windowMinutes: 15,
  },
  qqAuthStatusIp: {
    endpoint: 'POST /v1/auth/qq/status',
    scope: 'ip',
    maxRequests: 30,
    windowMinutes: 15,
  },
  pendingDiscordOAuthSessions: {
    maxPerInstallation: 2,
    maxPerIp: 20,
    windowMinutes: 15,
  },
  newActiveEntitlementsPerDay: {
    endpoint: 'POST /v1/providers/openrouter/discord/issue',
    scope: 'global',
    maxCount: 500,
    windowDays: 1,
  },
  immediateAlerts: {
    warning: 10,
    brake: 70,
  },
  retention: {
    requestEventSafetyMarginDays: 1,
    issueSuccessDays: 30,
    runtimeAuditDays: 90,
    referralSkippedDays: 7,
    referralFailedDays: 30,
  },
  referralAttempts: {
    validShaped: {
      maxPerInstallation: 8,
      maxPerIp: 30,
      windowMinutes: 15,
    },
    unknown: {
      maxPerInstallation: 3,
      maxPerIp: 10,
      windowMinutes: 15,
    },
    perReferralIdVelocity: {
      maxAttempts: 25,
      windowMinutes: 60,
    },
    perReferrerRewardVelocity: {
      maxRewards: 5,
      windowMinutes: 1440,
    },
  },
  dailyReport: {
    enabled: true,
    hourUtc: 0,
    minuteUtc: 5,
  },
};

export interface BrokerAbuseRuntimeBrakeState {
  active: boolean;
  reason: 'global_threshold' | 'asn_fast_path' | 'manual' | null;
  changedAt: string | null;
  changedBy: 'system' | 'operator' | null;
}

export interface BrokerAbuseRuntimeAlertLatches {
  warning: boolean;
  warningObservedAt: string | null;
}

export interface BrokerAbuseRuntimeDailyReportState {
  lastDeliveredAt: string | null;
  lastDeliveredDateUtc: string | null;
}

export interface BrokerAbuseRuntimeStateValue {
  brake: BrokerAbuseRuntimeBrakeState;
  alertLatches: BrokerAbuseRuntimeAlertLatches;
  dailyReport: BrokerAbuseRuntimeDailyReportState;
}

export const DEFAULT_BROKER_ABUSE_RUNTIME_STATE: BrokerAbuseRuntimeStateValue = {
  brake: {
    active: false,
    reason: null,
    changedAt: null,
    changedBy: null,
  },
  alertLatches: {
    warning: false,
    warningObservedAt: null,
  },
  dailyReport: {
    lastDeliveredAt: null,
    lastDeliveredDateUtc: null,
  },
};

export interface BrokerQqTalkTogetherPassConfigValue {
  enabled: boolean;
  rewards_enabled: boolean;
  daily_warning_count: number;
  daily_max_count: number;
}

export const DEFAULT_QQ_TALK_TOGETHER_PASS_CONFIG: BrokerQqTalkTogetherPassConfigValue = {
  enabled: false,
  rewards_enabled: false,
  daily_warning_count: 30,
  daily_max_count: 50,
};

export const BROKER_RUNTIME_CONFIG_SCHEMA = {
  [BROKER_RUNTIME_CONFIG_KEYS.fingerprintSalt]: ['current', 'previous', 'rotated_at'],
  [BROKER_RUNTIME_CONFIG_KEYS.abuseControls]: DEFAULT_BROKER_ABUSE_CONTROLS,
  [BROKER_RUNTIME_CONFIG_KEYS.abuseRuntimeState]: DEFAULT_BROKER_ABUSE_RUNTIME_STATE,
  [BROKER_RUNTIME_CONFIG_KEYS.qqTalkTogetherPass]:
    DEFAULT_QQ_TALK_TOGETHER_PASS_CONFIG,
} as const;

export type BrokerRuntimeConfigKey =
  (typeof BROKER_RUNTIME_CONFIG_KEYS)[keyof typeof BROKER_RUNTIME_CONFIG_KEYS];

export interface BrokerConfigRow {
  key: BrokerRuntimeConfigKey;
  value: string;
  updated_at: string;
}

export interface FingerprintSaltVersion {
  version: number;
  salt: string;
  valid_until: string | null;
}

export interface FingerprintSaltConfigValue {
  current: {
    version: number;
    salt: string;
  };
  previous: FingerprintSaltVersion | null;
  rotated_at: string | null;
}

export interface InstallationRecord {
  installation_id: string;
  device_public_key: string;
  hardware_hash: string | null;
  hardware_hash_salt_version: number | null;
  app_version: string;
  challenge: string | null;
  challenge_expires_at: string | null;
  challenge_salt_version: number | null;
  created_at: string;
  last_seen_at: string;
}

export const BROKER_PUBLIC_INPUT_BOUNDS = {
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
} as const;

export const OPENROUTER_ENTITLEMENT_STATUS_VALUES = [
  'pending_release',
  'active',
  'expired',
  'revoked',
] as const;

export const QQ_MANAGED_ENTITLEMENT_STATUS_VALUES = [
  'issuing',
  'delivery_pending',
  'active',
  'cleanup_required',
  'revoked',
] as const;

export const QQ_MANAGED_ENTITLEMENT_AUTOMATIC_REISSUE_BLOCKING_STATUS_VALUES = [
  'active',
  'cleanup_required',
  'revoked',
] as const;

export const MANAGED_KEY_DELIVERY_STATUS_VALUES = [
  'pending',
  'acknowledged',
  'expired',
  'cleanup_required',
] as const;

export const QQ_MANAGED_ENTITLEMENT_STALE_ISSUING_POLICY = {
  ttlMinutes: 15,
  withoutManagedCredentialRef:
    'eligible for same-subject release/reclaim by a later valid request after TTL only when child-key creation never started',
  withManagedCredentialRef:
    'cleanup/remediation candidate; must not be silently overwritten',
} as const;

export const QQ_SUBJECT_REF_POLICY = {
  prefix: 'ph-qq-subject-v1_',
  hmacSecretBinding: 'QQ_AUTH_HMAC_PSK',
  rotationGuardrail:
    'production QQ_AUTH_HMAC_PSK replacement requires a versioned subject-ref rotation plan with dual lookup/backfill semantics; simple secret replacement is not allowed',
} as const;

export const DISCORD_OAUTH_SESSION_STATUS_VALUES = [
  'pending',
  'processing',
  'consumed',
  'canceled',
  'failed',
  'expired',
] as const;

export const REFERRAL_ID_FORMAT_DESCRIPTION =
  'six uppercase approved-alphabet characters excluding 0/O/1/I/L';

export const REFERRAL_CODE_STATUS_VALUES = ['active', 'disabled'] as const;

export const REFERRAL_REFERRED_BONUS_STATUS_VALUES = [
  'reserved',
  'credited',
  'skipped',
  'failed',
] as const;

export const REFERRAL_REFERRER_BONUS_STATUS_VALUES = [
  'pending',
  'applying',
  'credited',
  'skipped',
  'failed',
] as const;

export const QQ_PASS_SETTLEMENT_PHASE_VALUES = [
  'invitee_pending',
  'referrer_pending',
  'completed',
] as const;

export type OpenRouterEntitlementStatus =
  (typeof OPENROUTER_ENTITLEMENT_STATUS_VALUES)[number];

export type BrokerIssueSuccessSource = 'discord' | 'qq';

export type QqManagedEntitlementStatus =
  (typeof QQ_MANAGED_ENTITLEMENT_STATUS_VALUES)[number];

export type ManagedKeyDeliveryStatus =
  (typeof MANAGED_KEY_DELIVERY_STATUS_VALUES)[number];

export type DiscordOAuthSessionStatus =
  (typeof DISCORD_OAUTH_SESSION_STATUS_VALUES)[number];

export type ReferralCodeStatus = (typeof REFERRAL_CODE_STATUS_VALUES)[number];

export type ReferralReferredBonusStatus =
  (typeof REFERRAL_REFERRED_BONUS_STATUS_VALUES)[number];

export type ReferralReferrerBonusStatus =
  (typeof REFERRAL_REFERRER_BONUS_STATUS_VALUES)[number];

export type QqPassSettlementPhase =
  (typeof QQ_PASS_SETTLEMENT_PHASE_VALUES)[number];

export interface DiscordOAuthSessionRecord {
  state_hash: string;
  installation_id: string;
  device_public_key: string;
  redirect_uri: string;
  pkce_code_verifier: string | null;
  issue_nonce_hash: string;
  fingerprint_salt_version: number;
  discord_user_ref: string | null;
  discord_email_verified: 0 | 1 | null;
  discord_account_created_at: string | null;
  eligibility_checked_at: string | null;
  status: DiscordOAuthSessionStatus;
  created_at: string;
  expires_at: string;
  processing_started_at: string | null;
  consumed_at: string | null;
  referral_id: string | null;
}

export type ReferralSource = BrokerIssueSuccessSource;

export interface ReferralCodeRecord {
  referral_id: string;
  owner_source: ReferralSource;
  owner_subject_ref: string;
  owner_installation_id: string | null;
  status: ReferralCodeStatus;
  disabled_reason?: string | null;
  disabled_by?: string | null;
  disabled_at?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ReferralRewardRecord {
  id: number;
  referral_id: string;
  referrer_source: ReferralSource | null;
  referrer_subject_ref: string | null;
  referrer_installation_id: string | null;
  referred_source: ReferralSource;
  referred_subject_ref: string;
  referred_installation_id: string | null;
  referred_hardware_hash: string | null;
  referred_hardware_hash_salt_version: number | null;
  referred_bonus_status: ReferralReferredBonusStatus;
  referrer_bonus_status: ReferralReferrerBonusStatus;
  skip_reason: string | null;
  failure_reason: string | null;
  referred_managed_credential_ref: string | null;
  referrer_managed_credential_ref: string | null;
  attempt_ip_hash?: string | null;
  created_at: string;
  updated_at: string;
  credited_at: string | null;
}

export interface OpenRouterEntitlementRecord {
  installation_id: string;
  status: OpenRouterEntitlementStatus;
  budget_usd: number;
  managed_credential_ref: string | null;
  issued_at: string | null;
  expires_at: string | null;
  release_session_ref: string | null;
  release_token_hash: string | null;
  release_token_expires_at: string | null;
  verified_hardware_hash: string | null;
  verified_hardware_hash_salt_version: number | null;
  discord_user_ref: string | null;
  discord_issue_status:
    | 'issuing'
    | 'delivery_pending'
    | 'active'
    | 'failed'
    | 'cleanup_required'
    | null;
  discord_issue_reserved_at: string | null;
  discord_issue_delivered_at: string | null;
}

export interface BrokerRequestEventRecord {
  id: number;
  endpoint: string;
  ip: string | null;
  installation_id: string | null;
  observed_at: string;
}

export interface TelemetryActiveDayRecord {
  subject_ref: string;
  active_date_utc: string;
  first_received_at: string;
  last_received_at: string;
}

export interface TelemetrySubjectRecord {
  subject_ref: string;
  first_active_date_utc: string;
  last_active_date_utc: string;
}

export interface AppActiveDayRecord {
  subject_ref: string;
  active_date_utc: string;
}

export interface BrokerDailySummaryDeliveryRecord {
  report_date_utc: string;
  status: 'pending' | 'delivered';
  lease_token: string;
  lease_expires_at: string;
  attempted_at: string;
  delivered_at: string | null;
}

export interface QqAuthAssertionRecord {
  qq_subject_ref: string;
  credential_hash: string;
  asserted_at: string;
  received_at: string;
  status: 'verified';
}

export interface QqManagedEntitlementRecord {
  qq_subject_ref: string;
  status: QqManagedEntitlementStatus;
  issue_ref: string;
  managed_credential_ref: string | null;
  budget_usd: number;
  reserved_at: string;
  issued_at: string | null;
  expires_at: string | null;
  delivered_at: string | null;
  child_key_creation_started_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface QqPassSettlementJobRecord {
  id: number;
  referral_reward_id: number;
  delivery_id: string;
  phase: QqPassSettlementPhase;
  attempt_count: number;
  last_attempt_at: string | null;
  next_attempt_at: string;
  fencing_token: string | null;
  lease_expires_at: string | null;
  last_error_code: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface ManagedKeyDeliveryRecord {
  delivery_id: string;
  issue_source: BrokerIssueSuccessSource;
  subject_ref: string | null;
  installation_id: string | null;
  managed_credential_ref: string;
  ack_token_hash: string;
  status: ManagedKeyDeliveryStatus;
  created_at: string;
  expires_at: string;
  acknowledged_at: string | null;
  failed_at: string | null;
  failure_reason: string | null;
}

export interface BrokerIssueSuccessEventRecord {
  id: number;
  issue_source: BrokerIssueSuccessSource;
  installation_id: string | null;
  subject_ref: string;
  managed_credential_ref: string | null;
  ip_hash: string | null;
  ip_prefix_hash: string | null;
  asn: number | null;
  country: string | null;
  http_protocol: string | null;
  tls_version: string | null;
  tls_cipher: string | null;
  risk_label: string | null;
  observed_at: string;
}

export interface BrokerAbuseRuntimeAuditRecord {
  id: number;
  event_kind: string;
  reason: string | null;
  payload_json: string;
  created_at: string;
}

export interface BrokerVelocityCapHookRecord {
  id: number;
  subject_type: 'ip' | 'installation_id';
  subject_value: string;
  max_requests: number;
  window_minutes: number;
  outcome_code:
    | 'rate_limited'
    | 'issuance_suspended'
    | 'trial_unavailable'
    | 'trial_not_eligible';
  outcome_class: 'retryable' | 'terminal' | 'security_fail';
  outcome_subcode: string | null;
  reason: string | null;
  active: 0 | 1;
  created_at: string;
  expires_at: string | null;
}

export interface BrokerAbuseSubjectHookRecord {
  id: number;
  hook_kind: 'denylist' | 'reputation' | 'revocation';
  subject_type: 'ip' | 'installation_id' | 'hardware_hash';
  subject_value: string;
  outcome_code:
    | 'issuance_suspended'
    | 'trial_unavailable'
    | 'trial_not_eligible';
  outcome_class: 'retryable' | 'terminal' | 'security_fail';
  outcome_subcode: string | null;
  reason: string | null;
  active: 0 | 1;
  created_at: string;
  expires_at: string | null;
}

export const BROKER_PERSISTENCE_MODEL = {
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
      textBounds: BROKER_PUBLIC_INPUT_BOUNDS,
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
      storedStatuses: OPENROUTER_ENTITLEMENT_STATUS_VALUES,
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
      storedStatuses: DISCORD_OAUTH_SESSION_STATUS_VALUES,
      retention: 'expires_at cleanup only; durable entitlement and identity evidence is separate',
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
      referralIdFormat: REFERRAL_ID_FORMAT_DESCRIPTION,
      storedStatuses: REFERRAL_CODE_STATUS_VALUES,
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
      referralIdFormat: REFERRAL_ID_FORMAT_DESCRIPTION,
      subjectSources: ['discord', 'qq'],
      referredBonusStatuses: REFERRAL_REFERRED_BONUS_STATUS_VALUES,
      referrerBonusStatuses: REFERRAL_REFERRER_BONUS_STATUS_VALUES,
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
      storedStatuses: QQ_MANAGED_ENTITLEMENT_STATUS_VALUES,
      automaticReissueBlockedStatuses:
        QQ_MANAGED_ENTITLEMENT_AUTOMATIC_REISSUE_BLOCKING_STATUS_VALUES,
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
      staleIssuingPolicy: QQ_MANAGED_ENTITLEMENT_STALE_ISSUING_POLICY,
      subjectRefPolicy: QQ_SUBJECT_REF_POLICY,
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
      storedStatuses: MANAGED_KEY_DELIVERY_STATUS_VALUES,
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
      phases: QQ_PASS_SETTLEMENT_PHASE_VALUES,
      unique: ['referral_reward_id', 'delivery_id', 'fencing_token when claimed'],
      indexed: ['phase + next_attempt_at + lease_expires_at'],
      noRetention: true,
      noCascade: true,
      fencing: 'every claim, transition, release, and completion mutation requires the exact fencing_token',
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
} as const;

export const BROKER_RETENTION_POLICY = {
  challengePreflight: {
    statuses: ['none'],
    entitlementRow: 'absent',
    challengeState: 'present',
    inactiveDays: 1,
    reference: 'max(installations.last_seen_at, installations.challenge_expires_at)',
    deleteFrom: 'installations',
    cascadesTo: [],
  },
  pendingRelease: {
    statuses: ['pending_release'],
    inactiveDays: 30,
    reference: 'installations.last_seen_at',
    deleteFrom: 'installations',
    cascadesTo: ['openrouter_entitlements'],
  },
  terminal: {
    statuses: ['expired', 'revoked'],
    inactiveDays: 90,
    reference: 'max(installations.last_seen_at, openrouter_entitlements.expires_at)',
    deleteFrom: 'installations',
    cascadesTo: ['openrouter_entitlements'],
  },
} as const;

export const FINGERPRINT_SALT_POLICY = {
  configKey: 'fingerprint_salt',
  managedBy: 'broker',
  sharedAcrossClients: true,
  duplicateDetectionScope: 'cross-installation',
  storageModel: 'bounded-current-plus-previous',
  valueShape: {
    current: ['version', 'salt'],
    previous: ['version', 'salt', 'valid_until'],
    rotated_at: 'timestamp-or-null',
  },
  installationTracking: {
    challengeSaltVersionField: 'challenge_salt_version',
    hardwareHashSaltVersionField: 'hardware_hash_salt_version',
  },
  duplicateMatching: {
    hashField: 'hardware_hash',
    currentVersionOnly: true,
  },
  rotation: {
    newChallengesUse: 'current salt only',
    inFlightChallenges: 'accept previous salt version until challenge_expires_at',
    staleHardwareHash:
      'exclude non-current hardware_hash from duplicate matching until refreshed or cleared',
    migrationPath:
      'overwrite hardware_hash in place on next verify with current salt, otherwise clear on challenge reissue only for none or pending_release lifecycles',
  },
} as const;
