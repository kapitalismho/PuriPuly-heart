import { isRecord } from './type-guards';

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true });
let verificationKey: { secret: string; key: Promise<CryptoKey> } | undefined;

function decode(segment: string): Uint8Array<ArrayBuffer> {
  if (!/^[A-Za-z0-9_-]+$/.test(segment)) throw new Error('invalid_token');
  const binary = atob(segment.replace(/-/g, '+').replace(/_/g, '/'));
  const canonical = btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  if (canonical !== segment) throw new Error('invalid_token');
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

export function hasTokenSecret(secret: unknown): secret is string {
  return typeof secret === 'string' && encoder.encode(secret).byteLength >= 32;
}

export async function authenticate(request: Request, secret: string, now = Math.floor(Date.now() / 1000)): Promise<string | null> {
  const authorization = request.headers.get('authorization');
  if (!authorization || !/^Bearer [A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(authorization)) return null;
  try {
    const [headerPart, claimsPart, signaturePart] = authorization.slice(7).split('.');
    const header: unknown = JSON.parse(decoder.decode(decode(headerPart)));
    if (!isRecord(header) || Object.keys(header).length !== 2 || header.alg !== 'HS256' || header.typ !== 'JWT') return null;
    const signature = decode(signaturePart);
    if (signature.byteLength !== 32) return null;
    let current = verificationKey;
    if (!current || current.secret !== secret) {
      const preparing = { secret, key: crypto.subtle.importKey('raw', encoder.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['verify']) };
      verificationKey = current = preparing;
      void preparing.key.catch(() => {
        if (verificationKey === preparing) verificationKey = undefined;
      });
    }
    const key = await current.key;
    if (!await crypto.subtle.verify('HMAC', key, signature, encoder.encode(`${headerPart}.${claimsPart}`))) return null;
    const claims: unknown = JSON.parse(decoder.decode(decode(claimsPart)));
    if (!isRecord(claims)
      || claims.iss !== 'puripuly-heart-broker'
      || claims.aud !== 'puripuly-agent-relay'
      || typeof claims.sub !== 'string' || !/^[a-f0-9]{64}$/.test(claims.sub)
      || !Number.isSafeInteger(claims.iat) || !Number.isSafeInteger(claims.exp)) return null;
    const issued = claims.iat as number;
    const expires = claims.exp as number;
    if (issued < 0 || issued > now || expires <= now || expires <= issued || expires - issued > 3600) return null;
    return claims.sub;
  } catch {
    return null;
  }
}
