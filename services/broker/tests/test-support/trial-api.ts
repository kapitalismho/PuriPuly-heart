import app from '../../src/index';

import type { TestBrokerEnv } from './sqlite-d1';

export interface ChallengeResponse {
  challenge: string;
  challenge_expires_at: string;
}

export interface TestExecutionContext {
  props: unknown;
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}

export async function issueChallenge(options: {
  env: TestBrokerEnv;
  installationId: string;
  devicePublicKey: string;
  appVersion: string;
}): Promise<ChallengeResponse> {
  const response = await app.request(
    'http://broker.test/v1/trial/challenge',
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        installation_id: options.installationId,
        device_public_key: options.devicePublicKey,
        app_version: options.appVersion,
      }),
    },
    options.env,
  );

  if (response.status !== 200) {
    throw new Error(`challenge request failed with status ${response.status}`);
  }

  return (await response.json()) as ChallengeResponse;
}

export async function postVerify(
  env: TestBrokerEnv,
  body: object | string,
): Promise<Response> {
  return app.request(
    'http://broker.test/v1/trial/challenge/verify',
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: typeof body === 'string' ? body : JSON.stringify(body),
    },
    env,
  );
}

export async function getTrialStatus(options: {
  env: TestBrokerEnv;
  installationId?: string;
  headers?: Record<string, string>;
}): Promise<Response> {
  const url = new URL('http://broker.test/v1/trial/status');
  if (options.installationId !== undefined) {
    url.searchParams.set('installation_id', options.installationId);
  }

  return app.request(
    url.toString(),
    {
      method: 'GET',
      headers: options.headers,
    },
    options.env,
  );
}

export async function postIssue(
  env: TestBrokerEnv,
  body: object | string,
): Promise<Response> {
  return app.request(
    'http://broker.test/v1/providers/openrouter/issue',
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: typeof body === 'string' ? body : JSON.stringify(body),
    },
    env,
  );
}

export async function postIssueWithExecutionContext(
  env: TestBrokerEnv,
  body: object | string,
  executionCtx: TestExecutionContext,
): Promise<Response> {
  return app.fetch(
    new Request('http://broker.test/v1/providers/openrouter/issue', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: typeof body === 'string' ? body : JSON.stringify(body),
    }),
    env,
    executionCtx,
  );
}

export async function postDiscordStart(
  env: TestBrokerEnv,
  body: object | string,
): Promise<Response> {
  return app.request(
    'http://broker.test/v1/auth/discord/start',
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'cf-connecting-ip': '203.0.113.20',
      },
      body: typeof body === 'string' ? body : JSON.stringify(body),
    },
    env,
  );
}

export async function postDiscordIssue(
  env: TestBrokerEnv,
  body: object | string,
  options: { headers?: Record<string, string> } = {},
): Promise<Response> {
  return app.request(
    'http://broker.test/v1/providers/openrouter/discord/issue',
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...options.headers,
      },
      body: typeof body === 'string' ? body : JSON.stringify(body),
    },
    env,
  );
}
