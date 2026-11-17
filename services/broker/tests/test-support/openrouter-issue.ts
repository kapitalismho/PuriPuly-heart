import {
  createDeviceKeyPair,
  signCanonicalIssueRequest,
  signCanonicalVerifyRequest,
  type DeviceKeyPair,
} from './ed25519';
import type { TestBrokerEnv } from './sqlite-d1';
import { issueChallenge, postIssue, postVerify } from './trial-api';
import { vi } from 'vitest';

let managedChildKeySequence = 0;

export interface MockOpenRouterManagementApiOptions {
  mode?: 'success' | 'cleanup_failure' | 'malformed_create';
  rawKey?: string;
  keyHash?: string;
}

export interface MockOpenRouterManagementApiHandle {
  childKey: {
    rawKey: string;
    hash: string;
  };
  fetchMock: ReturnType<typeof vi.fn>;
}

export async function createPendingReleaseSession(options: {
  env: TestBrokerEnv;
  installationId: string;
  appVersion: string;
  hardwareHash: string;
  verifySignedAt?: string;
}): Promise<{
  keyPair: DeviceKeyPair;
  releaseToken: string;
  releaseTokenExpiresAt: string;
  hardwareHash: string;
}> {
  const keyPair = await createDeviceKeyPair();
  const challenge = await issueChallenge({
    env: options.env,
    installationId: options.installationId,
    devicePublicKey: keyPair.devicePublicKey,
    appVersion: options.appVersion,
  });
  const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
    installation_id: options.installationId,
    device_public_key: keyPair.devicePublicKey,
    challenge: challenge.challenge,
    challenge_expires_at: challenge.challenge_expires_at,
    hardware_hash: options.hardwareHash,
    app_version: options.appVersion,
    signed_at: options.verifySignedAt ?? '2026-04-08T06:00:30.000Z',
  });
  const response = await postVerify(options.env, requestBody);

  if (response.status !== 200) {
    throw new Error(`verify request failed with status ${response.status}`);
  }

  const payload = (await response.json()) as {
    release_token: string;
    release_token_expires_at: string;
  };

  return {
    keyPair,
    releaseToken: payload.release_token,
    releaseTokenExpiresAt: payload.release_token_expires_at,
    hardwareHash: options.hardwareHash,
  };
}

export async function activatePendingReleaseSession(options: {
  env: TestBrokerEnv;
  installationId: string;
  appVersion: string;
  hardwareHash: string;
  verifySignedAt?: string;
  issueSignedAt?: string;
}): Promise<{
  keyPair: DeviceKeyPair;
  releaseToken: string;
  releaseTokenExpiresAt: string;
  hardwareHash: string;
  response: Response;
}> {
  mockOpenRouterManagementApi();
  const pendingRelease = await createPendingReleaseSession(options);
  const requestBody = await signCanonicalIssueRequest(pendingRelease.keyPair.privateKey, {
    installation_id: options.installationId,
    device_public_key: pendingRelease.keyPair.devicePublicKey,
    release_token: pendingRelease.releaseToken,
    hardware_hash: options.hardwareHash,
    reason: 'llm_start',
    budget_usd: 0.07,
    model: 'google/gemma-4-26b-a4b-it',
    signed_at: options.issueSignedAt ?? '2026-04-08T06:00:45.000Z',
  });
  const response = await postIssue(options.env, requestBody);

  return {
    ...pendingRelease,
    response,
  };
}

export function mockOpenRouterManagementApi(
  options: MockOpenRouterManagementApiOptions = {},
): MockOpenRouterManagementApiHandle {
  managedChildKeySequence += 1;
  const sequence = managedChildKeySequence;
  const childKey = {
    rawKey: options.rawKey ?? `or-managed-child-key-test-${sequence}`,
    hash: options.keyHash ?? `hash_managed_child_test_${sequence}`,
  };
  const mode = options.mode ?? 'success';
  let createCount = 0;

  const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';

    if (url === 'https://openrouter.ai/api/v1/keys' && method === 'POST') {
      createCount += 1;
      const createChildKey =
        createCount === 1
          ? childKey
          : {
              rawKey: `or-managed-child-key-test-${sequence}-${createCount}`,
              hash: `hash_managed_child_test_${sequence}_${createCount}`,
            };

      if (mode === 'malformed_create') {
        return jsonResponse(
          {
            key: createChildKey.rawKey,
          },
          201,
        );
      }

      return jsonResponse(
        {
          key: createChildKey.rawKey,
          data: {
            hash: createChildKey.hash,
          },
        },
        201,
      );
    }

    if (
      url === 'https://openrouter.ai/api/v1/guardrails/test-managed-guardrail-id/assignments/keys' &&
      method === 'POST'
    ) {
      return jsonResponse({ assigned_count: 1 });
    }

    if (url === `https://openrouter.ai/api/v1/keys/${childKey.hash}` && method === 'PATCH') {
      return jsonResponse({ data: { hash: childKey.hash, disabled: true } });
    }

    if (url === `https://openrouter.ai/api/v1/keys/${childKey.hash}` && method === 'DELETE') {
      if (mode === 'cleanup_failure') {
        return jsonResponse(
          {
            error: {
              code: 500,
              message: 'delete failed',
            },
          },
          500,
        );
      }

      return new Response(null, { status: 204 });
    }

    throw new Error(`unexpected OpenRouter management request: ${method} ${url}`);
  });

  vi.stubGlobal('fetch', fetchMock as typeof fetch);

  return {
    childKey,
    fetchMock,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json',
    },
  });
}
