import type { ManagedIssueSource } from './managed-issuance';

const OPENROUTER_USER_ID_VERSION = 'v1';
const OPENROUTER_USER_ID_PREFIX = `ph-or-user-${OPENROUTER_USER_ID_VERSION}_`;
const OPENROUTER_USER_ID_PAYLOAD_PREFIX =
  `puripuly-heart:openrouter-user:${OPENROUTER_USER_ID_VERSION}`;

interface LegacyManagedOpenRouterUserIdInput {
  installationId: string;
  secret: string;
  issueSource?: never;
  subjectRef?: never;
}

interface SourceAwareManagedOpenRouterUserIdInput {
  issueSource: ManagedIssueSource;
  subjectRef: string;
  secret: string;
  installationId?: never;
}

export type DeriveManagedOpenRouterUserIdInput =
  | LegacyManagedOpenRouterUserIdInput
  | SourceAwareManagedOpenRouterUserIdInput;

export async function deriveManagedOpenRouterUserId(
  input: DeriveManagedOpenRouterUserIdInput,
): Promise<string | null> {
  const normalizedSubject = normalizeOpenRouterUserSubject(input);
  const normalizedSecret = input.secret.trim();

  if (!normalizedSubject || !normalizedSecret) {
    return null;
  }

  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(normalizedSecret),
    {
      name: 'HMAC',
      hash: 'SHA-256',
    },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign(
    'HMAC',
    key,
    encoder.encode(`${OPENROUTER_USER_ID_PAYLOAD_PREFIX}\n${normalizedSubject}`),
  );

  return `${OPENROUTER_USER_ID_PREFIX}${toBase64Url(signature)}`;
}

function normalizeOpenRouterUserSubject(
  input: DeriveManagedOpenRouterUserIdInput,
): string | null {
  if (isLegacyManagedOpenRouterUserIdInput(input)) {
    const normalizedInstallationId = input.installationId.trim();
    return normalizedInstallationId || null;
  }

  if (input.issueSource === 'qq') {
    const normalizedSubjectRef = input.subjectRef.trim();
    return normalizedSubjectRef ? `qq\n${normalizedSubjectRef}` : null;
  }

  const normalizedSubjectRef = input.subjectRef.trim();
  return normalizedSubjectRef || null;
}

function isLegacyManagedOpenRouterUserIdInput(
  input: DeriveManagedOpenRouterUserIdInput,
): input is LegacyManagedOpenRouterUserIdInput {
  return 'installationId' in input && typeof input.installationId === 'string';
}

function toBase64Url(value: ArrayBuffer): string {
  let binary = '';

  for (const byte of new Uint8Array(value)) {
    binary += String.fromCharCode(byte);
  }

  return btoa(binary).replace(/\+/gu, '-').replace(/\//gu, '_').replace(/=+$/u, '');
}
