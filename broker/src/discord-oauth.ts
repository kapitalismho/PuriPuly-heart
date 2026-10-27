import { encodeBase64Url } from './crypto';

export const DISCORD_AUTHORIZE_URL = 'https://discord.com/oauth2/authorize';
export const DISCORD_TOKEN_URL = 'https://discord.com/api/oauth2/token';
export const DISCORD_USER_URL = 'https://discord.com/api/users/@me';
export const DISCORD_EPOCH_MS = 1420070400000n;

const DISCORD_REDIRECT_HOST = '127.0.0.1';
const DISCORD_REDIRECT_PATH = '/discord/callback';
const DISCORD_SCOPE = 'identify email';
const DISCORD_SNOWFLAKE_SHIFT_BITS = 22n;
const MAX_FUTURE_SNOWFLAKE_SKEW_MS = 5 * 60 * 1000;

const textEncoder = new TextEncoder();

export interface PkcePair {
  codeVerifier: string;
  codeChallenge: string;
  codeChallengeMethod: 'S256';
}

export interface DiscordTokenResponse {
  access_token: string;
  token_type: string;
  expires_in?: number;
  refresh_token?: string;
  scope?: string;
}

export interface DiscordUserResponse {
  id: string;
  username?: string;
  discriminator?: string;
  global_name?: string | null;
  avatar?: string | null;
  email?: string | null;
  verified?: boolean;
}

export function parseDiscordRedirectAllowlist(raw: string): string[] {
  const values = raw
    .split(',')
    .map((value) => value.trim())
    .filter((value) => value.length > 0);

  if (values.length === 0) {
    throw new Error('Discord redirect URI allowlist must include at least one URI');
  }

  for (const value of values) {
    let parsed: URL;
    try {
      parsed = new URL(value);
    } catch {
      throw new Error('Discord redirect URI must be a valid URL');
    }

    if (parsed.protocol !== 'http:' || parsed.hostname !== DISCORD_REDIRECT_HOST) {
      throw new Error('Discord redirect URI must use http://127.0.0.1');
    }

    if (parsed.pathname !== DISCORD_REDIRECT_PATH) {
      throw new Error('Discord redirect URI must use /discord/callback path');
    }

    if (parsed.username || parsed.password || parsed.search || parsed.hash) {
      throw new Error('Discord redirect URI must be an exact loopback callback URI');
    }
  }

  return values;
}

export function assertRedirectAllowed(
  redirectUri: string,
  allowlist: readonly string[],
): void {
  if (!allowlist.includes(redirectUri)) {
    throw new Error('Discord redirect URI is not registered');
  }
}

export async function generatePkcePair(): Promise<PkcePair> {
  const verifierBytes = new Uint8Array(64);
  crypto.getRandomValues(verifierBytes);
  const codeVerifier = encodeBase64Url(verifierBytes);
  const challengeBytes = await crypto.subtle.digest(
    'SHA-256',
    textEncoder.encode(codeVerifier),
  );

  return {
    codeVerifier,
    codeChallenge: encodeBase64Url(new Uint8Array(challengeBytes)),
    codeChallengeMethod: 'S256',
  };
}

export function buildDiscordAuthorizationUrl(input: {
  clientId: string;
  redirectUri: string;
  state: string;
  codeChallenge: string;
}): string {
  const url = new URL(DISCORD_AUTHORIZE_URL);
  url.searchParams.set('client_id', input.clientId);
  url.searchParams.set('response_type', 'code');
  url.searchParams.set('redirect_uri', input.redirectUri);
  url.searchParams.set('scope', DISCORD_SCOPE);
  url.searchParams.set('state', input.state);
  url.searchParams.set('code_challenge', input.codeChallenge);
  url.searchParams.set('code_challenge_method', 'S256');
  return url.toString();
}

export async function exchangeDiscordCode(input: {
  clientId: string;
  clientSecret: string;
  code: string;
  redirectUri: string;
  codeVerifier: string;
  fetcher?: typeof fetch;
}): Promise<DiscordTokenResponse> {
  const fetcher = input.fetcher ?? fetch;
  const body = new URLSearchParams({
    grant_type: 'authorization_code',
    code: input.code,
    redirect_uri: input.redirectUri,
    client_id: input.clientId,
    client_secret: input.clientSecret,
    code_verifier: input.codeVerifier,
  });
  const response = await fetcher(DISCORD_TOKEN_URL, {
    method: 'POST',
    headers: {
      'content-type': 'application/x-www-form-urlencoded',
    },
    body: body.toString(),
  });

  if (!response.ok) {
    throw new Error('Discord token exchange failed');
  }

  return parseDiscordTokenResponse(
    await readDiscordJsonResponse(response, 'malformed Discord token response'),
  );
}

export async function fetchDiscordUser(input: {
  accessToken: string;
  fetcher?: typeof fetch;
}): Promise<DiscordUserResponse> {
  const fetcher = input.fetcher ?? fetch;
  const response = await fetcher(DISCORD_USER_URL, {
    method: 'GET',
    headers: {
      authorization: `Bearer ${input.accessToken}`,
    },
  });

  if (!response.ok) {
    throw new Error('Discord user fetch failed');
  }

  return parseDiscordUserResponse(
    await readDiscordJsonResponse(response, 'malformed Discord user response'),
  );
}

async function readDiscordJsonResponse(
  response: Response,
  malformedMessage: string,
): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    throw new Error(malformedMessage);
  }
}

function parseDiscordTokenResponse(value: unknown): DiscordTokenResponse {
  if (!isRecord(value)) {
    throw new Error('malformed Discord token response');
  }

  if (typeof value.access_token !== 'string' || typeof value.token_type !== 'string') {
    throw new Error('malformed Discord token response');
  }

  if (value.expires_in !== undefined && !isFiniteNumber(value.expires_in)) {
    throw new Error('malformed Discord token response');
  }

  if (value.refresh_token !== undefined && typeof value.refresh_token !== 'string') {
    throw new Error('malformed Discord token response');
  }

  if (value.scope !== undefined && typeof value.scope !== 'string') {
    throw new Error('malformed Discord token response');
  }

  return {
    access_token: value.access_token,
    token_type: value.token_type,
    ...(value.expires_in !== undefined ? { expires_in: value.expires_in } : {}),
    ...(value.refresh_token !== undefined ? { refresh_token: value.refresh_token } : {}),
    ...(value.scope !== undefined ? { scope: value.scope } : {}),
  };
}

function parseDiscordUserResponse(value: unknown): DiscordUserResponse {
  if (!isRecord(value) || typeof value.id !== 'string') {
    throw new Error('malformed Discord user response');
  }

  if (value.email !== undefined && value.email !== null && typeof value.email !== 'string') {
    throw new Error('malformed Discord user response');
  }

  if (value.verified !== undefined && typeof value.verified !== 'boolean') {
    throw new Error('malformed Discord user response');
  }

  return {
    id: value.id,
    ...(typeof value.username === 'string' ? { username: value.username } : {}),
    ...(typeof value.discriminator === 'string'
      ? { discriminator: value.discriminator }
      : {}),
    ...(typeof value.global_name === 'string' || value.global_name === null
      ? { global_name: value.global_name }
      : {}),
    ...(typeof value.avatar === 'string' || value.avatar === null
      ? { avatar: value.avatar }
      : {}),
    ...(value.email !== undefined ? { email: value.email } : {}),
    ...(value.verified !== undefined ? { verified: value.verified } : {}),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function deriveDiscordAccountCreatedAt(discordUserId: string): string {
  if (!/^[1-9]\d*$/u.test(discordUserId)) {
    throw new Error('invalid Discord snowflake');
  }

  const snowflake = BigInt(discordUserId);
  const timestampMs = (snowflake >> DISCORD_SNOWFLAKE_SHIFT_BITS) + DISCORD_EPOCH_MS;
  const maxSafeTimestamp = BigInt(Number.MAX_SAFE_INTEGER);
  const maxPlausibleTimestamp = BigInt(Date.now() + MAX_FUTURE_SNOWFLAKE_SKEW_MS);

  if (
    timestampMs < DISCORD_EPOCH_MS ||
    timestampMs > maxSafeTimestamp ||
    timestampMs > maxPlausibleTimestamp
  ) {
    throw new Error('invalid Discord snowflake');
  }

  const createdAt = new Date(Number(timestampMs));
  if (Number.isNaN(createdAt.getTime())) {
    throw new Error('invalid Discord snowflake');
  }

  return createdAt.toISOString();
}
