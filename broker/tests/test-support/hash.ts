import { resolveReferralAttemptIdentity, resolveRequestNetworkIdentity } from '../../src/network-identity';

export async function sha256Base64Url(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(value),
  );

  return encodeBase64Url(new Uint8Array(digest));
}

function encodeBase64Url(bytes: Uint8Array): string {
  const binary = Array.from(bytes, (value) => String.fromCharCode(value)).join('');
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/u, '');
}

export async function attemptIpDigestFor(
  env: { NETWORK_IDENTITY_HMAC_SECRET: string },
  ip: string,
  nowIso: string,
): Promise<{ digest: string; keyVersion: number; epoch: string }> {
  const identity = await resolveRequestNetworkIdentity(
    ip,
    { current: env.NETWORK_IDENTITY_HMAC_SECRET, previous: null },
    new Date(nowIso),
  );
  if (!identity) {
    throw new Error('test IP digest derivation failed');
  }
  return { digest: identity.digest, keyVersion: identity.keyVersion, epoch: identity.epoch };
}

export async function attemptIpLegacyHashFor(ip: string): Promise<string> {
  const identity = await resolveReferralAttemptIdentity(ip, null, new Date());
  if (!identity.legacyHash) {
    throw new Error('test IP legacy hash derivation failed');
  }
  return identity.legacyHash;
}

export async function attemptIpInputFor(
  env: { NETWORK_IDENTITY_HMAC_SECRET: string },
  ip: string,
  nowIso: string,
): Promise<{
  attemptIpDigest: { digest: string; keyVersion: number; epoch: string };
  attemptIpLegacyHash: string;
}> {
  return {
    attemptIpDigest: await attemptIpDigestFor(env, ip, nowIso),
    attemptIpLegacyHash: await attemptIpLegacyHashFor(ip),
  };
}
