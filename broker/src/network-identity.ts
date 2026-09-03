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
  if (trimmed.includes(':')) {
    if (trimmed.includes('.')) {
      return canonicalizeEmbeddedIpv4Address(trimmed);
    }
    return canonicalizeIpv6Address(trimmed);
  }
  if (trimmed.includes('.')) {
    return canonicalizeIpv4Address(trimmed);
  }
  return null;
}

function canonicalizeIpv4Address(value: string): string | null {
  const parts = value.split('.');
  if (parts.length !== 4) {
    return null;
  }
  const canonical: string[] = [];
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part)) {
      return null;
    }
    const octet = Number(part);
    if (!Number.isInteger(octet) || octet < 0 || octet > 255) {
      return null;
    }
    canonical.push(String(octet));
  }
  return canonical.join('.');
}

function canonicalizeEmbeddedIpv4Address(value: string): string | null {
  const separator = value.lastIndexOf(':');
  if (separator < 0) {
    return null;
  }
  const ipv4 = canonicalizeIpv4Address(value.slice(separator + 1));
  if (!ipv4) {
    return null;
  }
  const head = value.slice(0, separator).toLowerCase();
  if (head === '' || head === ':' || head === '::' || head === '::ffff' || head === '0:0:0:0:0:ffff') {
    return ipv4;
  }
  const octets = ipv4.split('.').map(Number);
  const high = ((octets[0]! * 256 + octets[1]!).toString(16));
  const low = ((octets[2]! * 256 + octets[3]!).toString(16));
  return canonicalizeIpv6Address(`${value.slice(0, separator)}:${high}:${low}`);
}

function canonicalizeIpv6Address(value: string): string | null {
  let rest = value.toLowerCase();
  if (rest.startsWith('[') && rest.endsWith(']')) {
    rest = rest.slice(1, -1);
  }
  const zoneIndex = rest.indexOf('%');
  if (zoneIndex >= 0) {
    rest = rest.slice(0, zoneIndex);
  }
  if (rest.length === 0 || /[^0-9a-f:]/.test(rest)) {
    return null;
  }
  const halves = rest.split('::');
  if (halves.length > 2) {
    return null;
  }
  const head = halves[0]!.length > 0 ? halves[0]!.split(':') : [];
  const tail = halves.length === 2 && halves[1]!.length > 0 ? halves[1]!.split(':') : [];
  const groups = [...head, ...tail];
  for (const group of groups) {
    if (!/^[0-9a-f]{1,4}$/.test(group)) {
      return null;
    }
  }
  if (halves.length === 1 && groups.length !== 8) {
    return null;
  }
  if (groups.length > 8) {
    return null;
  }
  const padding = new Array<string>(8 - groups.length).fill('0');
  return [...head.map(stripIpv6LeadingZeros), ...padding, ...tail.map(stripIpv6LeadingZeros)].join(':');
}

function stripIpv6LeadingZeros(group: string): string {
  const stripped = group.replace(/^0+/, '');
  return stripped.length > 0 ? stripped : '0';
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
  currentVersion: number;
}

export function resolveNetworkIdentitySecrets(env: Record<string, unknown>): NetworkIdentitySecrets | null {
  const current = typeof env.NETWORK_IDENTITY_HMAC_SECRET === 'string' ? env.NETWORK_IDENTITY_HMAC_SECRET.trim() : '';
  if (!current) {
    return null;
  }
  const previousRaw = typeof env.NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS === 'string' ? env.NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS.trim() : '';
  const versionRaw = env.NETWORK_IDENTITY_HMAC_KEY_VERSION;
  const parsedVersion = typeof versionRaw === 'string' && versionRaw.trim().length > 0
    ? Number(versionRaw)
    : typeof versionRaw === 'number'
      ? versionRaw
      : Number.NaN;
  const currentVersion = Number.isInteger(parsedVersion) && parsedVersion >= 1 ? parsedVersion : 1;
  return { current, previous: previousRaw ? previousRaw : null, currentVersion };
}

export async function deriveNetworkIdentityDigestsForWindow(
  secrets: NetworkIdentitySecrets,
  normalizedIp: string,
  now: Date,
  scope: 'ip' | 'prefix',
  windowStart: Date,
): Promise<Array<{ digest: string; keyVersion: number; epoch: string }>> {
  const epochs = enumerateNetworkIdentityEpochs(windowStart, now);
  const out: Array<{ digest: string; keyVersion: number; epoch: string }> = [];
  for (const epoch of epochs) {
    out.push({
      digest: await hmacHex(secrets.current, `${NETWORK_IDENTITY_DOMAIN}\n${secrets.currentVersion}\n${epoch}\n${scope}\n${normalizedIp}`),
      keyVersion: secrets.currentVersion,
      epoch,
    });
    if (secrets.previous) {
      out.push({
        digest: await hmacHex(secrets.previous, `${NETWORK_IDENTITY_DOMAIN}\n${secrets.currentVersion - 1}\n${epoch}\n${scope}\n${normalizedIp}`),
        keyVersion: secrets.currentVersion - 1,
        epoch,
      });
    }
  }
  return out;
}

export function enumerateNetworkIdentityEpochs(windowStart: Date, now: Date): string[] {
  const startDay = Math.floor(windowStart.getTime() / NETWORK_IDENTITY_EPOCH_MS);
  const endDay = Math.floor(now.getTime() / NETWORK_IDENTITY_EPOCH_MS);
  const epochs: string[] = [];
  for (let day = startDay; day <= endDay; day += 1) {
    epochs.push(new Date(day * NETWORK_IDENTITY_EPOCH_MS).toISOString().slice(0, 10));
  }
  return epochs;
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
    { digest: await hmacHex(secrets.current, `${NETWORK_IDENTITY_DOMAIN}\n${secrets.currentVersion}\nstable\n${scope}\n${normalizedIp}`), keyVersion: secrets.currentVersion },
  ];
  if (secrets.previous) {
    out.push({ digest: await hmacHex(secrets.previous, `${NETWORK_IDENTITY_DOMAIN}\n${secrets.currentVersion - 1}\nstable\n${scope}\n${normalizedIp}`), keyVersion: secrets.currentVersion - 1 });
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
  const digest = await hmacHex(secrets.current, `${NETWORK_IDENTITY_DOMAIN}\n${secrets.currentVersion}\n${epoch}\nip\n${normalized}`);
  const prefixDigest = await hmacHex(secrets.current, `${NETWORK_IDENTITY_DOMAIN}\n${secrets.currentVersion}\n${epoch}\nprefix\n${normalizeNetworkIdentityPrefix(normalized)}`);
  return { digest, prefixDigest, keyVersion: secrets.currentVersion, epoch };
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
