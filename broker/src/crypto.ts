export function isBase64Url(value: string, byteLength?: number): boolean {
  if (!/^[A-Za-z0-9_-]+$/u.test(value)) {
    return false;
  }

  try {
    const decoded = decodeBase64Url(value);
    return byteLength === undefined || decoded.length === byteLength;
  } catch {
    return false;
  }
}

function decodeBase64Url(value: string): Uint8Array {
  const padding = (4 - (value.length % 4 || 4)) % 4;
  const normalized = value.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat(padding);
  const binary = atob(normalized);
  const bytes = new Uint8Array(binary.length);

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return bytes;
}

export function encodeBase64Url(bytes: Uint8Array): string {
  const binary = Array.from(bytes, (value) => String.fromCharCode(value)).join('');
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/u, '');
}

export async function verifyEd25519Signature(input: {
  devicePublicKey: string;
  signature: string;
  payload: Uint8Array;
}): Promise<boolean> {
  try {
    const publicKey = await crypto.subtle.importKey(
      'raw',
      toArrayBuffer(decodeBase64Url(input.devicePublicKey)),
      { name: 'Ed25519' },
      false,
      ['verify'],
    );

    return crypto.subtle.verify(
      { name: 'Ed25519' },
      publicKey,
      toArrayBuffer(decodeBase64Url(input.signature)),
      toArrayBuffer(input.payload),
    );
  } catch {
    return false;
  }
}

export async function sha256Base64Url(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    toArrayBuffer(new TextEncoder().encode(value)),
  );

  return encodeBase64Url(new Uint8Array(digest));
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
}

export function randomBase64Url(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return encodeBase64Url(bytes);
}
