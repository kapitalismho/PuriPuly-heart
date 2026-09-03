const NETWORK_IDENTITY_DOMAIN = 'puripuly-heart:network-identity:v1';
export const NETWORK_IDENTITY_KEY_VERSION = 1;
export const NETWORK_IDENTITY_EPOCH_MS = 24 * 60 * 60_000;

export interface NetworkIdentityDigest {
  digest: string;
  keyVersion: number;
  epoch: string;
}

export function normalizeNetworkIdentityIp(ip: string): string | null {
  const trimmed = ip.trim();
  if (trimmed.length === 0 || trimmed.length > 256) {
    return null;
  }
  if (/[\s"'\\]/.test(trimmed)) {
    return null;
  }
  let value = trimmed;
  if (value.includes(':')) {
    value = value.toLowerCase();
    if (value.startsWith('[') && value.endsWith(']')) {
      value = value.slice(1, -1);
    }
    const zoneIndex = value.indexOf('%');
    if (zoneIndex >= 0) {
      value = value.slice(0, zoneIndex);
    }
  }
  return value.length === 0 ? null : value;
}

export function normalizeNetworkIdentityPrefix(normalizedIp: string): string {
  if (normalizedIp.includes(':')) {
    return normalizedIp.split(':').filter((part) => part.length > 0).slice(0, 4).join(':');
  }
  return normalizedIp.split('.').slice(0, 3).join('.');
}

export function resolveNetworkIdentityEpoch(now: Date): string {
  const day = new Date(Math.floor(now.getTime() / NETWORK_IDENTITY_EPOCH_MS) * NETWORK_IDENTITY_EPOCH_MS);
  return day.toISOString().slice(0, 10);
}

export function resolveNetworkIdentityEpochs(now: Date): string[] {
  const current = resolveNetworkIdentityEpoch(now);
  const previous = resolveNetworkIdentityEpoch(new Date(now.getTime() - NETWORK_IDENTITY_EPOCH_MS));
  return previous === current ? [current] : [current, previous];
}

async function hmacHex(secret: string, message: string): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey('raw', encoder.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const signature = await crypto.subtle.sign('HMAC', key, encoder.encode(message));
  return Array.from(new Uint8Array(signature), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export interface NetworkIdentitySecrets {
  current: string;
  previous: string | null;
}

export function resolveNetworkIdentitySecrets(env: Record<string, unknown>): NetworkIdentitySecrets | null {
  const current = typeof env.NETWORK_IDENTITY_HMAC_SECRET === 'string' ? env.NETWORK_IDENTITY_HMAC_SECRET.trim() : '';
  if (!current) {
    return null;
  }
  const previousRaw = typeof env.NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS === 'string' ? env.NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS.trim() : '';
  return { current, previous: previousRaw ? previousRaw : null };
}

export async function deriveNetworkIdentityDigest(
  secrets: NetworkIdentitySecrets,
  normalizedIp: string,
  now: Date,
  scope: 'ip' | 'prefix',
): Promise<NetworkIdentityDigest> {
  const epoch = resolveNetworkIdentityEpoch(now);
  const message = `${NETWORK_IDENTITY_DOMAIN}\n${NETWORK_IDENTITY_KEY_VERSION}\n${epoch}\n${scope}\n${normalizedIp}`;
  return { digest: await hmacHex(secrets.current, message), keyVersion: NETWORK_IDENTITY_KEY_VERSION, epoch };
}

export async function deriveNetworkIdentityDigestsForWindow(
  secrets: NetworkIdentitySecrets,
  normalizedIp: string,
  now: Date,
  scope: 'ip' | 'prefix',
  windowStart: Date,
): Promise<Array<{ digest: string; keyVersion: number; epoch: string }>> {
  const epochs = new Set<string>([resolveNetworkIdentityEpoch(now), resolveNetworkIdentityEpoch(windowStart)]);
  const out: Array<{ digest: string; keyVersion: number; epoch: string }> = [];
  for (const epoch of epochs) {
    const message = `${NETWORK_IDENTITY_DOMAIN}\n${NETWORK_IDENTITY_KEY_VERSION}\n${epoch}\n${scope}\n${normalizedIp}`;
    out.push({ digest: await hmacHex(secrets.current, message), keyVersion: NETWORK_IDENTITY_KEY_VERSION, epoch });
    if (secrets.previous) {
      out.push({ digest: await hmacHex(secrets.previous, message), keyVersion: NETWORK_IDENTITY_KEY_VERSION, epoch });
    }
  }
  return out;
}

export async function timingSafeEqualHex(left: string, right: string): Promise<boolean> {
  if (left.length !== right.length) {
    return false;
  }
  let diff = 0;
  for (let index = 0; index < left.length; index += 1) {
    diff |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return diff === 0;
}

export async function deriveStableNetworkIdentityDigest(
  secrets: NetworkIdentitySecrets,
  normalizedIp: string,
  scope: 'ip' | 'prefix',
): Promise<Array<{ digest: string; keyVersion: number }>> {
  const out: Array<{ digest: string; keyVersion: number }> = [
    { digest: await hmacHex(secrets.current, `${NETWORK_IDENTITY_DOMAIN}\n${NETWORK_IDENTITY_KEY_VERSION}\nstable\n${scope}\n${normalizedIp}`), keyVersion: NETWORK_IDENTITY_KEY_VERSION },
  ];
  if (secrets.previous) {
    out.push({ digest: await hmacHex(secrets.previous, `${NETWORK_IDENTITY_DOMAIN}\n${NETWORK_IDENTITY_KEY_VERSION}\nstable\n${scope}\n${normalizedIp}`), keyVersion: NETWORK_IDENTITY_KEY_VERSION });
  }
  return out;
}

export interface RequestNetworkIdentity {
  digest: string;
  prefixDigest: string;
  keyVersion: number;
  epoch: string;
}

export async function resolveRequestNetworkIdentity(
  ip: string | null,
  secrets: NetworkIdentitySecrets | null,
  now: Date,
): Promise<RequestNetworkIdentity | null> {
  if (!ip || !secrets) {
    return null;
  }
  const normalized = normalizeNetworkIdentityIp(ip);
  if (!normalized) {
    return null;
  }
  const epoch = resolveNetworkIdentityEpoch(now);
  const digest = await hmacHex(secrets.current, `${NETWORK_IDENTITY_DOMAIN}\n${NETWORK_IDENTITY_KEY_VERSION}\n${epoch}\nip\n${normalized}`);
  const prefixDigest = await hmacHex(secrets.current, `${NETWORK_IDENTITY_DOMAIN}\n${NETWORK_IDENTITY_KEY_VERSION}\n${epoch}\nprefix\n${normalizeNetworkIdentityPrefix(normalized)}`);
  return { digest, prefixDigest, keyVersion: NETWORK_IDENTITY_KEY_VERSION, epoch };
}

export async function resolveRequestNetworkIdentityCandidates(
  ip: string | null,
  secrets: NetworkIdentitySecrets | null,
  now: Date,
  windowStart: Date,
): Promise<Array<{ digest: string; keyVersion: number; epoch: string }>> {
  if (!ip || !secrets) {
    return [];
  }
  const normalized = normalizeNetworkIdentityIp(ip);
  if (!normalized) {
    return [];
  }
  return deriveNetworkIdentityDigestsForWindow(secrets, normalized, now, 'ip', windowStart);
}
export type NetworkIdentityWriteMode = 'legacy' | 'dual' | 'keyed';

export interface NetworkIdentityMigrationState {
  phase: 'dual_write' | 'keyed_only';
  purgeAfter: string | null;
}

export async function getNetworkIdentityMigrationState(
  db: D1Database,
): Promise<NetworkIdentityMigrationState | null> {
  let row: { value: string } | null = null;
  try {
    row = await db
      .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
      .first<{ value: string }>();
  } catch {
    return null;
  }
  if (!row) {
    return null;
  }
  try {
    const parsed = JSON.parse(row.value) as { phase?: unknown; purgeAfter?: unknown; purge_after?: unknown };
    return {
      phase: parsed.phase === 'keyed_only' ? 'keyed_only' : 'dual_write',
      purgeAfter:
        typeof parsed.purgeAfter === 'string'
          ? parsed.purgeAfter
          : typeof parsed.purge_after === 'string'
            ? parsed.purge_after
            : null,
    };
  } catch {
    return { phase: 'dual_write', purgeAfter: null };
  }
}

export async function resolveNetworkIdentityWriteMode(
  db: D1Database,
): Promise<NetworkIdentityWriteMode> {
  const state = await getNetworkIdentityMigrationState(db);
  if (!state) {
    return 'legacy';
  }
  return state.phase === 'keyed_only' ? 'keyed' : 'dual';
}

export interface ReferralAttemptIdentity {
  digest: { digest: string; keyVersion: number; epoch: string } | null;
  legacyHash: string | null;
}

export async function resolveReferralAttemptIdentity(
  ip: string | null,
  secrets: NetworkIdentitySecrets | null,
  now: Date,
): Promise<ReferralAttemptIdentity> {
  const normalized = ip ? normalizeNetworkIdentityIp(ip) : null;
  if (!normalized) {
    return { digest: null, legacyHash: null };
  }
  const digest = await crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(`puripuly-heart:referral-attempt-ip:v1\n${normalized}`),
  );
  const legacyHash = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('');
  const keyed = await resolveRequestNetworkIdentity(ip, secrets, now);
  return {
    digest: keyed ? { digest: keyed.digest, keyVersion: keyed.keyVersion, epoch: keyed.epoch } : null,
    legacyHash,
  };
}
