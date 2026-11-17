import { describe, expect, it } from 'vitest';

import {
  buildDiscordAuthorizationUrl,
  deriveDiscordAccountCreatedAt,
  exchangeDiscordCode,
  fetchDiscordUser,
  generatePkcePair,
  parseDiscordRedirectAllowlist,
} from '../src/discord-oauth';

const REGISTERED_REDIRECT_URIS = [
  'http://127.0.0.1:62187/discord/callback',
  'http://127.0.0.1:62188/discord/callback',
  'http://127.0.0.1:62189/discord/callback',
];

describe('Discord OAuth helpers', () => {
  it('parses exact comma-separated loopback callback redirect URIs', () => {
    expect(
      parseDiscordRedirectAllowlist(
        `${REGISTERED_REDIRECT_URIS[0]}, ${REGISTERED_REDIRECT_URIS[1]},${REGISTERED_REDIRECT_URIS[2]}`,
      ),
    ).toEqual(REGISTERED_REDIRECT_URIS);
  });

  it('rejects localhost redirect URIs', () => {
    expect(() =>
      parseDiscordRedirectAllowlist('http://localhost:62187/discord/callback'),
    ).toThrow(/Discord redirect URI must use http:\/\/127\.0\.0\.1/u);
  });

  it('generates an S256 PKCE pair with unpadded base64url values', async () => {
    const pair = await generatePkcePair();

    expect(pair.codeVerifier.length).toBeGreaterThanOrEqual(43);
    expect(pair.codeVerifier.length).toBeLessThanOrEqual(128);
    expect(pair.codeVerifier).not.toContain('=');
    expect(pair.codeChallenge).not.toContain('=');
    expect(pair.codeChallengeMethod).toBe('S256');
  });

  it('builds a Discord authorize URL with the required OAuth parameters', () => {
    const authorizationUrl = buildDiscordAuthorizationUrl({
      clientId: 'test-discord-client-id',
      redirectUri: REGISTERED_REDIRECT_URIS[0],
      state: 'state-value',
      codeChallenge: 'code-challenge-value',
    });

    const parsed = new URL(authorizationUrl);

    expect(parsed.origin).toBe('https://discord.com');
    expect(parsed.pathname).toBe('/oauth2/authorize');
    expect(parsed.searchParams.get('client_id')).toBe('test-discord-client-id');
    expect(parsed.searchParams.get('response_type')).toBe('code');
    expect(parsed.searchParams.get('redirect_uri')).toBe(REGISTERED_REDIRECT_URIS[0]);
    expect(parsed.searchParams.get('scope')).toBe('identify email');
    expect(parsed.searchParams.get('state')).toBe('state-value');
    expect(parsed.searchParams.get('code_challenge')).toBe('code-challenge-value');
    expect(parsed.searchParams.get('code_challenge_method')).toBe('S256');
  });

  it('derives Discord account creation time from a snowflake', () => {
    expect(deriveDiscordAccountCreatedAt('175928847299117063')).toBe(
      '2016-04-30T11:18:25.796Z',
    );
  });

  it('rejects invalid Discord snowflakes', () => {
    expect(() => deriveDiscordAccountCreatedAt('not-a-snowflake')).toThrow(
      'invalid Discord snowflake',
    );
  });

  it('rejects malformed Discord token responses', async () => {
    await expect(
      exchangeDiscordCode({
        clientId: 'test-discord-client-id',
        clientSecret: 'test-discord-client-secret',
        code: 'oauth-code',
        redirectUri: REGISTERED_REDIRECT_URIS[0],
        codeVerifier: 'pkce-code-verifier',
        fetcher: jsonFetcher({
          access_token: 'discord-access-token',
          token_type: 'Bearer',
          expires_in: '3600',
        }),
      }),
    ).rejects.toThrow('malformed Discord token response');
  });

  it('rejects OK Discord token responses with invalid JSON using malformed response errors', async () => {
    await expect(
      exchangeDiscordCode({
        clientId: 'test-discord-client-id',
        clientSecret: 'test-discord-client-secret',
        code: 'oauth-code',
        redirectUri: REGISTERED_REDIRECT_URIS[0],
        codeVerifier: 'pkce-code-verifier',
        fetcher: invalidJsonFetcher(),
      }),
    ).rejects.toThrow('malformed Discord token response');
  });

  it('rejects malformed Discord user responses', async () => {
    await expect(
      fetchDiscordUser({
        accessToken: 'discord-access-token',
        fetcher: jsonFetcher({
          id: '175928847299117063',
          email: 123,
          verified: true,
        }),
      }),
    ).rejects.toThrow('malformed Discord user response');
  });

  it('rejects OK Discord user responses with invalid JSON using malformed response errors', async () => {
    await expect(
      fetchDiscordUser({
        accessToken: 'discord-access-token',
        fetcher: invalidJsonFetcher(),
      }),
    ).rejects.toThrow('malformed Discord user response');
  });
});

function jsonFetcher(payload: unknown): typeof fetch {
  return (async () =>
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: {
        'content-type': 'application/json',
      },
    })) as typeof fetch;
}

function invalidJsonFetcher(): typeof fetch {
  return (async () =>
    new Response('not-json', {
      status: 200,
      headers: {
        'content-type': 'application/json',
      },
    })) as typeof fetch;
}
