export const BROKER_RUNTIME_CONFIG_KEYS = {
  fingerprintSalt: 'fingerprint_salt',
  abuseControls: 'abuse_controls',
  abuseRuntimeState: 'abuse_runtime_state',
  networkIdentityMigration: 'network_identity_migration',
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
  managedOperationStatusIp: BrokerEndpointRateLimitConfig;
  managedOperationStatusInstallation: BrokerEndpointRateLimitConfig;
  managedOperationResumeIp: BrokerEndpointRateLimitConfig;
  managedOperationResumeInstallation: BrokerEndpointRateLimitConfig;
  managedKeyDeliveryAckIp: BrokerEndpointRateLimitConfig;
  pendingDiscordOAuthSessions: BrokerPendingDiscordOAuthSessionsConfig;
  newActiveEntitlementsPerDay: BrokerDailyIssuanceCapConfig;
  immediateAlerts: BrokerImmediateAlertsConfig;
  retention: BrokerAbuseRetentionConfig;
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
  managedOperationStatusIp: {
    endpoint: 'POST /v1/providers/openrouter/managed-operation/status',
    scope: 'ip',
    maxRequests: 30,
    windowMinutes: 15,
  },
  managedOperationStatusInstallation: {
    endpoint: 'POST /v1/providers/openrouter/managed-operation/status',
    scope: 'installation_id',
    maxRequests: 30,
    windowMinutes: 15,
  },
  managedOperationResumeIp: {
    endpoint: 'POST /v1/providers/openrouter/managed-operation/resume',
    scope: 'ip',
    maxRequests: 20,
    windowMinutes: 15,
  },
  managedOperationResumeInstallation: {
    endpoint: 'POST /v1/providers/openrouter/managed-operation/resume',
    scope: 'installation_id',
    maxRequests: 10,
    windowMinutes: 15,
  },
  managedKeyDeliveryAckIp: {
    endpoint: 'POST /v1/providers/openrouter/managed-key-delivery/ack',
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

export const DISCORD_OAUTH_SESSION_STATUS_VALUES = [
  'pending',
  'processing',
  'consumed',
  'canceled',
  'failed',
  'expired',
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
  operation_id?: string | null;
  attempt_index?: number | null;
}

