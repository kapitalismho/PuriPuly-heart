import {
  BROKER_RUNTIME_CONFIG_KEYS,
  type FingerprintSaltConfigValue,
} from './persistence';

export async function getFingerprintSaltConfig(
  db: D1Database,
): Promise<FingerprintSaltConfigValue> {
  const row = await db
    .prepare('SELECT value FROM broker_config WHERE key = ?')
    .bind(BROKER_RUNTIME_CONFIG_KEYS.fingerprintSalt)
    .first<{ value: string }>();

  if (!row) {
    throw new Error('missing fingerprint_salt config');
  }

  return JSON.parse(row.value) as FingerprintSaltConfigValue;
}
