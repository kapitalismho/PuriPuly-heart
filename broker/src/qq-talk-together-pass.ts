import {
  BROKER_RUNTIME_CONFIG_KEYS,
  DEFAULT_QQ_TALK_TOGETHER_PASS_CONFIG,
  type BrokerQqTalkTogetherPassConfigValue,
} from './persistence';

export async function getQqTalkTogetherPassConfig(
  db: D1Database,
): Promise<BrokerQqTalkTogetherPassConfigValue> {
  const row = await db
    .prepare('SELECT value FROM broker_config WHERE key = ?')
    .bind(BROKER_RUNTIME_CONFIG_KEYS.qqTalkTogetherPass)
    .first<{ value: string }>();
  if (!row) {
    return DEFAULT_QQ_TALK_TOGETHER_PASS_CONFIG;
  }
  try {
    const value = JSON.parse(row.value) as unknown;
    return parseQqTalkTogetherPassConfig(value) ?? DEFAULT_QQ_TALK_TOGETHER_PASS_CONFIG;
  } catch {
    return DEFAULT_QQ_TALK_TOGETHER_PASS_CONFIG;
  }
}

export function qqReferralUtcDayStartIso(now: Date): string {
  return new Date(
    Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()),
  ).toISOString();
}

function parseQqTalkTogetherPassConfig(
  value: unknown,
): BrokerQqTalkTogetherPassConfigValue | null {
  if (!isRecord(value)) {
    return null;
  }
  const enabled = value.enabled;
  const rewardsEnabled = value.rewards_enabled;
  const dailyWarningCount = value.daily_warning_count;
  const dailyMaxCount = value.daily_max_count;
  if (
    typeof enabled !== 'boolean' ||
    typeof rewardsEnabled !== 'boolean' ||
    !isNonNegativeInteger(dailyWarningCount) ||
    !isPositiveInteger(dailyMaxCount) ||
    dailyWarningCount > dailyMaxCount
  ) {
    return null;
  }
  return {
    enabled,
    rewards_enabled: rewardsEnabled,
    daily_warning_count: dailyWarningCount,
    daily_max_count: dailyMaxCount,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}
