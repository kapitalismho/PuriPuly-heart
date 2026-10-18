import { resolveRequestNetworkIdentity } from '../../src/network-identity';

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
    {
      current: env.NETWORK_IDENTITY_HMAC_SECRET,
      previous: null,
      previousVersion: null,
      currentVersion: 1,
    },
    new Date(nowIso),
  );
  if (!identity) {
    throw new Error('test IP digest derivation failed');
  }
  return { digest: identity.digest, keyVersion: identity.keyVersion, epoch: identity.epoch };
}

