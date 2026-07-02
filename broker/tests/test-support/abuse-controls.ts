import type { TestBrokerEnv } from './sqlite-d1';

export interface StoredAbuseControls {
  trialChallenge: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  trialChallengeVerify: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  openrouterIssue: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  trialStatus: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  discordAuthStartIp: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  discordAuthStartInstallation: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  discordOpenrouterIssueIp: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  discordOpenrouterIssueInstallation: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  qqAuthAssertIp: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  telemetryTranslationSuccessDayIp: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  pendingDiscordOAuthSessions: {
    maxPerInstallation: number;
    maxPerIp: number;
    windowMinutes: number;
  };
  newActiveEntitlementsPerDay: {
    endpoint: string;
    scope: 'global';
    maxCount: number | null;
    windowDays: number;
  };
  immediateAlerts: {
    warn1: number;
    warn2: number;
    warn3: number;
    critical: number;
  };
  asnFastPath: {
    enabled: boolean;
    minIssueSuccess1h: number;
    minTopAsnSharePct: number;
  };
  asnClassifications: Array<{
    asn: number;
    kind: 'cloud_or_vps';
    displayName?: string;
  }>;
  retention: {
    requestEventsDays: number;
    issueSuccessDays: number;
    runtimeAuditDays: number;
    referralSkippedDays: number;
    referralFailedDays: number;
  };
  referralAttempts: {
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
  };
  dailyReport: {
    enabled: boolean;
    hourUtc: number;
    minuteUtc: number;
    includeZeroActivity: boolean;
  };
}

export interface StoredAbuseRuntimeState {
  brake: {
    active: boolean;
    reason: 'global_threshold' | 'asn_fast_path' | 'manual' | null;
    changedAt: string | null;
    changedBy: 'system' | 'operator' | null;
  };
  alertLatches: {
    warn1: boolean;
    warn2: boolean;
    warn3: boolean;
    critical: boolean;
  };
  dailyReport: {
    lastDeliveredAt: string | null;
    lastDeliveredDateUtc: string | null;
  };
}

export const TEST_DEFAULT_ABUSE_CONTROLS: StoredAbuseControls = {
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
  telemetryTranslationSuccessDayIp: {
    endpoint: 'POST /v1/telemetry/translation-success-day',
    scope: 'ip',
    maxRequests: 60,
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
    warn1: 10,
    warn2: 25,
    warn3: 50,
    critical: 70,
  },
  asnFastPath: {
    enabled: true,
    minIssueSuccess1h: 20,
    minTopAsnSharePct: 70,
  },
  asnClassifications: [],
  retention: {
    requestEventsDays: 30,
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
    hourUtc: 13,
    minuteUtc: 0,
    includeZeroActivity: false,
  },
};

export const TEST_DEFAULT_ABUSE_RUNTIME_STATE: StoredAbuseRuntimeState = {
  brake: {
    active: false,
    reason: null,
    changedAt: null,
    changedBy: null,
  },
  alertLatches: {
    warn1: false,
    warn2: false,
    warn3: false,
    critical: false,
  },
  dailyReport: {
    lastDeliveredAt: null,
    lastDeliveredDateUtc: null,
  },
};

export function readAbuseControls(env: TestBrokerEnv): StoredAbuseControls {
  const row = env.__db
    .prepare('SELECT value FROM broker_config WHERE key = ?')
    .get('abuse_controls') as { value: string } | undefined;

  if (!row) {
    throw new Error('missing broker_config row: abuse_controls');
  }

  return normalizeAbuseControls(JSON.parse(row.value) as unknown);
}

export function updateAbuseControls(
  env: TestBrokerEnv,
  mutate: (controls: StoredAbuseControls) => void,
): void {
  const controls = readAbuseControls(env);
  mutate(controls);
  env.__db
    .prepare('UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?')
    .run(JSON.stringify(controls), new Date().toISOString(), 'abuse_controls');
}

export function replaceAbuseControlsValue(env: TestBrokerEnv, value: unknown): void {
  env.__db
    .prepare('UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?')
    .run(JSON.stringify(value), new Date().toISOString(), 'abuse_controls');
}

export function readAbuseRuntimeState(env: TestBrokerEnv): StoredAbuseRuntimeState {
  const row = env.__db
    .prepare('SELECT value FROM broker_config WHERE key = ?')
    .get('abuse_runtime_state') as { value: string } | undefined;

  if (!row) {
    throw new Error('missing broker_config row: abuse_runtime_state');
  }

  return normalizeAbuseRuntimeState(JSON.parse(row.value) as unknown);
}

export function updateAbuseRuntimeState(
  env: TestBrokerEnv,
  mutate: (state: StoredAbuseRuntimeState) => void,
): void {
  const state = readAbuseRuntimeState(env);
  mutate(state);
  env.__db
    .prepare('UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?')
    .run(JSON.stringify(state), new Date().toISOString(), 'abuse_runtime_state');
}

export function replaceAbuseRuntimeStateValue(env: TestBrokerEnv, value: unknown): void {
  env.__db
    .prepare('UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?')
    .run(JSON.stringify(value), new Date().toISOString(), 'abuse_runtime_state');
}

export function insertVelocityCapHook(
  env: TestBrokerEnv,
  input: {
    subject_type: 'ip' | 'installation_id';
    subject_value: string;
    max_requests: number;
    window_minutes: number;
    outcome_code: string;
    outcome_class: string;
    outcome_subcode?: string | null;
    reason?: string | null;
    expires_at?: string | null;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO broker_velocity_cap_hooks (
          subject_type,
          subject_value,
          max_requests,
          window_minutes,
          outcome_code,
          outcome_class,
          outcome_subcode,
          reason,
          expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.subject_type,
      input.subject_value,
      input.max_requests,
      input.window_minutes,
      input.outcome_code,
      input.outcome_class,
      input.outcome_subcode ?? null,
      input.reason ?? null,
      input.expires_at ?? null,
    );
}

export function insertSubjectHook(
  env: TestBrokerEnv,
  input: {
    hook_kind: 'denylist' | 'reputation' | 'revocation';
    subject_type: 'ip' | 'installation_id' | 'hardware_hash';
    subject_value: string;
    outcome_code: string;
    outcome_class: string;
    outcome_subcode?: string | null;
    reason?: string | null;
    expires_at?: string | null;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO broker_abuse_subject_hooks (
          hook_kind,
          subject_type,
          subject_value,
          outcome_code,
          outcome_class,
          outcome_subcode,
          reason,
          expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.hook_kind,
      input.subject_type,
      input.subject_value,
      input.outcome_code,
      input.outcome_class,
      input.outcome_subcode ?? null,
      input.reason ?? null,
      input.expires_at ?? null,
    );
}

function normalizeAbuseControls(value: unknown): StoredAbuseControls {
  const normalized = cloneJson(TEST_DEFAULT_ABUSE_CONTROLS);
  if (!isRecord(value)) {
    return normalized;
  }

  assignRecord(normalized, value);
  assignRecord(normalized.trialChallenge, value.trialChallenge);
  assignRecord(normalized.trialChallengeVerify, value.trialChallengeVerify);
  assignRecord(normalized.openrouterIssue, value.openrouterIssue);
  assignRecord(normalized.trialStatus, value.trialStatus);
  assignRecord(normalized.discordAuthStartIp, value.discordAuthStartIp);
  assignRecord(
    normalized.discordAuthStartInstallation,
    value.discordAuthStartInstallation,
  );
  assignRecord(normalized.discordOpenrouterIssueIp, value.discordOpenrouterIssueIp);
  assignRecord(
    normalized.discordOpenrouterIssueInstallation,
    value.discordOpenrouterIssueInstallation,
  );
  assignRecord(normalized.qqAuthAssertIp, value.qqAuthAssertIp);
  assignRecord(normalized.pendingDiscordOAuthSessions, value.pendingDiscordOAuthSessions);
  assignRecord(normalized.newActiveEntitlementsPerDay, value.newActiveEntitlementsPerDay);
  assignRecord(normalized.immediateAlerts, value.immediateAlerts);
  assignRecord(normalized.asnFastPath, value.asnFastPath);
  assignRecord(normalized.retention, value.retention);
  assignRecord(normalized.referralAttempts, value.referralAttempts);
  if (isRecord(value.referralAttempts)) {
    assignRecord(normalized.referralAttempts.validShaped, value.referralAttempts.validShaped);
    assignRecord(normalized.referralAttempts.unknown, value.referralAttempts.unknown);
    assignRecord(
      normalized.referralAttempts.perReferralIdVelocity,
      value.referralAttempts.perReferralIdVelocity,
    );
    assignRecord(
      normalized.referralAttempts.perReferrerRewardVelocity,
      value.referralAttempts.perReferrerRewardVelocity,
    );
  }
  assignRecord(normalized.dailyReport, value.dailyReport);

  if (Array.isArray(value.asnClassifications)) {
    normalized.asnClassifications = value.asnClassifications
      .map(normalizeAsnClassification)
      .filter(
        (
          entry,
        ): entry is {
          asn: number;
          kind: 'cloud_or_vps';
          displayName?: string;
        } => entry !== null,
      );
  }

  return normalized;
}

function normalizeAbuseRuntimeState(value: unknown): StoredAbuseRuntimeState {
  const normalized = cloneJson(TEST_DEFAULT_ABUSE_RUNTIME_STATE);
  if (!isRecord(value)) {
    return normalized;
  }

  assignRecord(normalized, value);
  assignRecord(normalized.brake, value.brake);
  assignRecord(normalized.alertLatches, value.alertLatches);
  assignRecord(normalized.dailyReport, value.dailyReport);
  return normalized;
}

function normalizeAsnClassification(
  value: unknown,
): { asn: number; kind: 'cloud_or_vps'; displayName?: string } | null {
  if (!isRecord(value) || !Number.isInteger(value.asn) || value.kind !== 'cloud_or_vps') {
    return null;
  }

  return typeof value.displayName === 'string'
    ? {
        asn: Number(value.asn),
        kind: 'cloud_or_vps',
        displayName: value.displayName,
      }
    : {
        asn: Number(value.asn),
        kind: 'cloud_or_vps',
      };
}

function assignRecord<T extends object>(target: T, source: unknown): void {
  if (!isRecord(source)) {
    return;
  }

  Object.assign(target, source);
}

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
