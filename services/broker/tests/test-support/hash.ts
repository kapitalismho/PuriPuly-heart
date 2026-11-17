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

