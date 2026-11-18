import { sha256Base64Url } from './hash';

const encoder = new TextEncoder();

export interface DeviceKeyPair {
  privateKey: CryptoKey;
  devicePublicKey: string;
}

export interface SignedVerifyRequestInput {
  installation_id: string;
  device_public_key: string;
  challenge: string;
  challenge_expires_at: string;
  hardware_hash: string;
  app_version: string;
  signed_at: string;
}

export interface SignedStatusRequestInput {
  installation_id: string;
  timestamp: string;
}

export interface SignedIssueRequestInput {
  installation_id: string;
  device_public_key: string;
  release_token: string;
  hardware_hash: string;
  reason: string;
  budget_usd: number;
  model: string;
  signed_at: string;
}

export interface SignedDiscordIssueRequestInput {
  installation_id: string;
  device_public_key: string;
  state: string;
  code: string;
  redirect_uri: string;
  hardware_hash: string;
  hardware_hash_salt_version: number;
  app_version: string;
  reason: string;
  budget_usd: number;
  model: string;
  issue_nonce: string;
  signed_at: string;
}

export async function createDeviceKeyPair(): Promise<DeviceKeyPair> {
  const keyPair = await crypto.subtle.generateKey('Ed25519', true, [
    'sign',
    'verify',
  ]);
  const rawPublicKey = await crypto.subtle.exportKey('raw', keyPair.publicKey);

  return {
    privateKey: keyPair.privateKey,
    devicePublicKey: encodeBase64Url(new Uint8Array(rawPublicKey)),
  };
}

export async function signCanonicalVerifyRequest(
  privateKey: CryptoKey,
  input: SignedVerifyRequestInput,
): Promise<SignedVerifyRequestInput & { signature: string }> {
  return {
    ...input,
    signature: await signPayload(privateKey, canonicalVerifyPayload(input)),
  };
}

export async function signNonCanonicalVerifyRequest(
  privateKey: CryptoKey,
  input: SignedVerifyRequestInput,
): Promise<SignedVerifyRequestInput & { signature: string }> {
  return {
    ...input,
    signature: await signPayload(privateKey, nonCanonicalVerifyPayload(input)),
  };
}

export async function signCanonicalStatusRequest(
  privateKey: CryptoKey,
  input: SignedStatusRequestInput,
): Promise<SignedStatusRequestInput & { signature: string }> {
  return {
    ...input,
    signature: await signPayload(privateKey, canonicalStatusPayload(input)),
  };
}

export async function signNonCanonicalStatusRequest(
  privateKey: CryptoKey,
  input: SignedStatusRequestInput,
): Promise<SignedStatusRequestInput & { signature: string }> {
  return {
    ...input,
    signature: await signPayload(privateKey, nonCanonicalStatusPayload(input)),
  };
}

export async function signCanonicalIssueRequest(
  privateKey: CryptoKey,
  input: SignedIssueRequestInput,
): Promise<SignedIssueRequestInput & { signature: string }> {
  return {
    ...input,
    signature: await signPayload(privateKey, canonicalIssuePayload(input)),
  };
}

export async function signCanonicalDiscordIssueRequest(
  privateKey: CryptoKey,
  input: SignedDiscordIssueRequestInput,
): Promise<
  SignedDiscordIssueRequestInput & {
    signature_alg: 'ed25519';
    signature: string;
  }
> {
  return {
    ...input,
    signature_alg: 'ed25519',
    signature: await signPayload(
      privateKey,
      await canonicalDiscordIssuePayload(input),
    ),
  };
}

export async function signNonCanonicalIssueRequest(
  privateKey: CryptoKey,
  input: SignedIssueRequestInput,
): Promise<SignedIssueRequestInput & { signature: string }> {
  return {
    ...input,
    signature: await signPayload(privateKey, nonCanonicalIssuePayload(input)),
  };
}

function canonicalVerifyPayload(input: SignedVerifyRequestInput): Uint8Array {
  return encoder.encode(
    [
      input.installation_id,
      input.device_public_key,
      input.challenge,
      input.challenge_expires_at,
      input.hardware_hash,
      input.app_version,
      input.signed_at,
    ].join('\n'),
  );
}

function nonCanonicalVerifyPayload(input: SignedVerifyRequestInput): Uint8Array {
  return encoder.encode(
    [
      input.challenge,
      input.installation_id,
      input.device_public_key,
      input.challenge_expires_at,
      input.hardware_hash,
      input.app_version,
      input.signed_at,
    ].join('\n'),
  );
}

function canonicalStatusPayload(input: SignedStatusRequestInput): Uint8Array {
  return encoder.encode([input.installation_id, input.timestamp].join('\n'));
}

function nonCanonicalStatusPayload(input: SignedStatusRequestInput): Uint8Array {
  return encoder.encode([input.timestamp, input.installation_id].join('\n'));
}

function canonicalIssuePayload(input: SignedIssueRequestInput): Uint8Array {
  return encoder.encode(
    [
      input.installation_id,
      input.device_public_key,
      input.release_token,
      input.hardware_hash,
      input.reason,
      String(input.budget_usd),
      input.model,
      input.signed_at,
    ].join('\n'),
  );
}

function nonCanonicalIssuePayload(input: SignedIssueRequestInput): Uint8Array {
  return encoder.encode(
    [
      input.release_token,
      input.installation_id,
      input.device_public_key,
      input.hardware_hash,
      input.reason,
      String(input.budget_usd),
      input.model,
      input.signed_at,
    ].join('\n'),
  );
}

async function canonicalDiscordIssuePayload(
  input: SignedDiscordIssueRequestInput,
): Promise<Uint8Array> {
  return encoder.encode(
    [
      'POST',
      '/v1/providers/openrouter/discord/issue',
      input.installation_id,
      input.device_public_key,
      input.state,
      await sha256Base64Url(input.code),
      input.redirect_uri,
      input.hardware_hash,
      String(input.hardware_hash_salt_version),
      input.app_version,
      input.reason,
      String(input.budget_usd),
      input.model,
      input.issue_nonce,
      input.signed_at,
    ].join('\n'),
  );
}

async function signPayload(
  privateKey: CryptoKey,
  payload: Uint8Array,
): Promise<string> {
  const signature = await crypto.subtle.sign(
    'Ed25519',
    privateKey,
    toArrayBuffer(payload),
  );
  return encodeBase64Url(new Uint8Array(signature));
}

export function encodeBase64Url(bytes: Uint8Array): string {
  const binary = Array.from(bytes, (value) => String.fromCharCode(value)).join('');
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/u, '');
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
}
