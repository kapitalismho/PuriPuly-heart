import { describe, expect, it } from 'vitest';

import { BROKER_SERVICE_NAME } from '../../src/contract';
import { MANAGED_TRIAL_ALLOWED_MODELS } from '../../src/trial-policy';
import {
  TRIAL_STATUS_SIGNATURE_HEADER,
  TRIAL_STATUS_TIMESTAMP_HEADER,
} from '../../src/trial-handshake';
import {
  createDeviceKeyPair,
  signCanonicalIssueRequest,
  signCanonicalStatusRequest,
  signCanonicalVerifyRequest,
} from '../test-support/ed25519';

const CANONICAL_WORKER_NAME = 'puripuly-heart-broker';
const ISSUE_REASON = 'llm_start';
const ISSUE_BUDGET_USD = 0.07;
const MANAGED_ALLOWLIST_MODELS = [...MANAGED_TRIAL_ALLOWED_MODELS] as const;
const ISSUE_MODEL = MANAGED_ALLOWLIST_MODELS[0];
const POSITIVE_ROUTING_PROBE_MODELS = MANAGED_ALLOWLIST_MODELS.filter(
  (model) => model !== ISSUE_MODEL,
);
const EMPTY_CONTENT_ALLOWED_POSITIVE_ROUTING_MODELS = new Set([
  'deepseek/deepseek-v4-flash-0731',
  'deepseek/deepseek-v4-flash',
]);
const BOOTSTRAP_PLACEHOLDER = '__BOOTSTRAP_REQUIRED__';
const OPENROUTER_API_BASE_URL = new URL('https://openrouter.ai');
const smokeRunEnabled = process.env.BROKER_DEPLOY_SMOKE_RUN === 'true';
const smokeBaseUrl = process.env.BROKER_DEPLOY_SMOKE_BASE_URL?.trim();
const smokeQqAuthHmacPsk = process.env.BROKER_DEPLOY_SMOKE_QQ_AUTH_HMAC_PSK;
const smokeDisallowedModel = process.env.BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL;
const MANAGED_OPENROUTER_USER_ID_PATTERN = /^ph-or-user-v\d+_[A-Za-z0-9_-]+$/u;
const QQ_SUBJECT_REF_PATTERN = /^ph-qq-subject-v1_[A-Za-z0-9_-]+$/u;
const textEncoder = new TextEncoder();
type JsonRequestOptions = {
  method: string;
  url: URL;
  body?: unknown;
  headers?: HeadersInit;
};

const describeDeploySmoke = smokeRunEnabled ? describe : describe.skip;

describe('broker deploy smoke helpers', () => {
  it('reads issued child-key metadata from the OpenRouter current-key payload', () => {
    expect(
      readOpenRouterCurrentKeyMetadata({
        data: {
          limit: ISSUE_BUDGET_USD,
          expires_at: '2026-07-08T06:00:00.000Z',
        },
      }),
    ).toEqual({
      limit: ISSUE_BUDGET_USD,
      expiresAt: '2026-07-08T06:00:00.000Z',
    });
  });

  it('recognizes model-routing failures as guardrail enforcement for a disallowed model probe', () => {
    expect(
      isDisallowedModelGuardrailFailure(503, {
        error: {
          code: 503,
          message: 'No allowed model/provider is available for this request.',
        },
      }),
    ).toBe(true);
    expect(
      isDisallowedModelGuardrailFailure(401, {
        error: {
          code: 401,
          message: 'Invalid credentials',
        },
      }),
    ).toBe(false);
  });

  it('accepts successful OpenRouter chat completion response shapes for managed model probes', () => {
    expect(() =>
      assertSuccessfulChatCompletionResponse(
        {
          status: 200,
          body: {
            id: 'chatcmpl-123',
            choices: [
              {
                message: {
                  role: 'assistant',
                  content: 'routed',
                },
              },
            ],
          },
        },
        'qwen/qwen3.5-flash-02-23',
      ),
    ).not.toThrow();
  });

  it('accepts empty assistant content for the DeepSeek positive routing probe', () => {
    expect(() =>
      assertSuccessfulChatCompletionResponse(
        {
          status: 200,
          body: {
            id: 'chatcmpl-deepseek-empty-content',
            choices: [
              {
                message: {
                  role: 'assistant',
                  content: '',
                },
              },
            ],
          },
        },
        'deepseek/deepseek-v4-flash-0731',
      ),
    ).not.toThrow();
  });

  it('requires a distinct disallowed model probe when live smoke is enabled', () => {
    expect(normalizeDisallowedModel(undefined, MANAGED_ALLOWLIST_MODELS, false)).toBeUndefined();
    expect(
      normalizeDisallowedModel('openai/gpt-4o-mini', MANAGED_ALLOWLIST_MODELS, true),
    ).toBe('openai/gpt-4o-mini');
    expect(() =>
      normalizeDisallowedModel(ISSUE_MODEL, MANAGED_ALLOWLIST_MODELS, true),
    ).toThrow(/must differ from the managed allowlisted models/i);
    expect(() =>
      normalizeDisallowedModel(
        'qwen/qwen3.5-flash-02-23',
        MANAGED_ALLOWLIST_MODELS,
        true,
      ),
    ).toThrow(/must differ from the managed allowlisted models/i);
    expect(() =>
      normalizeDisallowedModel(
        'google/gemini-2.5-flash-lite',
        MANAGED_ALLOWLIST_MODELS,
        true,
      ),
    ).toThrow(/must differ from the managed allowlisted models/i);
    expect(() =>
      normalizeDisallowedModel(
        'deepseek/deepseek-v4-flash-0731',
        MANAGED_ALLOWLIST_MODELS,
        true,
      ),
    ).toThrow(/must differ from the managed allowlisted models/i);
    expect(() =>
      normalizeDisallowedModel(
        'deepseek/deepseek-v4-flash',
        MANAGED_ALLOWLIST_MODELS,
        true,
      ),
    ).toThrow(/must differ from the managed allowlisted models/i);
  });

  it('keeps the positive routing probes pinned to the managed secondary models', () => {
    expect(POSITIVE_ROUTING_PROBE_MODELS).toEqual([
      'qwen/qwen3.5-flash-02-23',
      'deepseek/deepseek-v4-flash-0731',
      'deepseek/deepseek-v4-flash',
      'google/gemini-2.5-flash-lite',
    ]);
    expect(MANAGED_ALLOWLIST_MODELS).toEqual(MANAGED_TRIAL_ALLOWED_MODELS);
  });

  it('computes QQ assertion credentials as lowercase HMAC-SHA256 hex', async () => {
    await expect(
      computeHmacSha256Hex('key', 'The quick brown fox jumps over the lazy dog'),
    ).resolves.toBe('f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8');
  });

  it('does not activate live deploy smoke unless explicitly opted in', () => {
    expect(shouldRunDeploySmoke(undefined)).toBe(false);
    expect(shouldRunDeploySmoke('')).toBe(false);
    expect(shouldRunDeploySmoke('false')).toBe(false);
    expect(shouldRunDeploySmoke('TRUE')).toBe(false);
    expect(shouldRunDeploySmoke('true')).toBe(true);
  });

  it('validates complete live smoke inputs before the live flow can start', () => {
    const validInputs = readLiveDeploySmokeInputs({
      baseUrl: 'https://puripuly-heart-broker.example.workers.dev',
      canonicalWorkerName: CANONICAL_WORKER_NAME,
      disallowedModel: 'openai/gpt-4o-mini',
      managedAllowlistedModels: MANAGED_ALLOWLIST_MODELS,
      qqAuthHmacPsk: 'deploy-smoke-psk',
    });

    expect(validInputs.baseUrl.toString()).toBe(
      'https://puripuly-heart-broker.example.workers.dev/',
    );
    expect(validInputs.disallowedModel).toBe('openai/gpt-4o-mini');
    expect(validInputs.qqAuthHmacPsk).toBe('deploy-smoke-psk');

    expect(() =>
      readLiveDeploySmokeInputs({
        baseUrl: 'https://puripuly-heart-broker.example.workers.dev',
        canonicalWorkerName: CANONICAL_WORKER_NAME,
        disallowedModel: undefined,
        managedAllowlistedModels: MANAGED_ALLOWLIST_MODELS,
        qqAuthHmacPsk: undefined,
      }),
    ).toThrow(
      /BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL.*BROKER_DEPLOY_SMOKE_QQ_AUTH_HMAC_PSK/is,
    );
    expect(() =>
      readLiveDeploySmokeInputs({
        baseUrl: undefined,
        canonicalWorkerName: CANONICAL_WORKER_NAME,
        disallowedModel: 'openai/gpt-4o-mini',
        managedAllowlistedModels: MANAGED_ALLOWLIST_MODELS,
        qqAuthHmacPsk: 'deploy-smoke-psk',
      }),
    ).toThrow(/BROKER_DEPLOY_SMOKE_BASE_URL/i);
  });

  it('rejects malformed managed OpenRouter user ids without echoing the value', () => {
    const malformedDerivedUserId = 'ph-or-user-v1_invalid value';

    expect(() => assertManagedOpenRouterUserId(malformedDerivedUserId)).toThrow(
      'issue success payload must include a valid openrouter_user_id',
    );

    try {
      assertManagedOpenRouterUserId(malformedDerivedUserId);
    } catch (error) {
      expect((error as Error).message).not.toContain(malformedDerivedUserId);
    }
  });

  it('redacts deploy-smoke sensitive values from failure text', () => {
    const syntheticKey = 'sk-or-v1-deploy-smoke-sensitive-key';
    const syntheticCredential = 'a'.repeat(64);
    const syntheticIdentity = 'deploy-smoke-qq-sensitive-identity';
    const syntheticSubjectRef = 'ph-qq-subject-v1_sensitiveSubject';

    const redacted = redactIssueBody(
      JSON.stringify({
        openrouter_api_key: syntheticKey,
        qq_identity: syntheticIdentity,
        credential: syntheticCredential,
        qq_subject_ref: syntheticSubjectRef,
      }) +
        ` Authorization: Bearer ${syntheticKey} qq_identity=${syntheticIdentity} credential=${syntheticCredential}`,
    );

    expect(redacted).not.toContain(syntheticKey);
    expect(redacted).not.toContain(syntheticCredential);
    expect(redacted).not.toContain(syntheticIdentity);
    expect(redacted).not.toContain(syntheticSubjectRef);
    expect(redacted).toContain('[REDACTED]');
  });

  it('does not include raw OpenRouter chat completion response bodies in failure messages', () => {
    const sensitivePayload = buildSensitiveFailurePayloadText();
    let failureMessage = '';

    try {
      assertSuccessfulChatCompletionResponse(
        {
          status: 502,
          body: {
            error: {
              message: sensitivePayload,
            },
          },
        },
        'qwen/qwen3.5-flash-02-23',
      );
    } catch (error) {
      failureMessage = readErrorMessage(error);
    }

    expect(failureMessage).toContain('qwen/qwen3.5-flash-02-23');
    expect(failureMessage).toContain('502');
    expect(failureMessage).toContain('response body redacted');
    expectFailureMessageExcludesSensitiveSentinels(failureMessage);
  });

  it('does not include raw duplicate QQ assertion response fields in failure messages', () => {
    const subcodeMismatchMessage = captureDuplicateQqAssertionFailureMessage({
      error: SENSITIVE_FAILURE_SENTINELS.rawBrokerMessageText,
      credential: SENSITIVE_FAILURE_SENTINELS.credential,
      qq_identity: SENSITIVE_FAILURE_SENTINELS.qqIdentity,
      qq_subject_ref: SENSITIVE_FAILURE_SENTINELS.subjectRef,
      raw_payload_text: SENSITIVE_FAILURE_SENTINELS.rawProviderPayloadText,
    });
    const keyFieldPresentMessage = captureDuplicateQqAssertionFailureMessage({
      error: {
        subcode: 'qq_lifetime_used',
      },
      openrouter_api_key: SENSITIVE_FAILURE_SENTINELS.keyLikeValue,
      qq_identity: SENSITIVE_FAILURE_SENTINELS.qqIdentity,
      credential: SENSITIVE_FAILURE_SENTINELS.credential,
      qq_subject_ref: SENSITIVE_FAILURE_SENTINELS.subjectRef,
      raw_payload_text: SENSITIVE_FAILURE_SENTINELS.rawProviderPayloadText,
    });

    const failureMessages = [subcodeMismatchMessage, keyFieldPresentMessage];

    expect(
      failureMessages.filter(countFailureMessageSensitiveSentinels).length,
      'duplicate QQ assertion failure messages must not include sentinel raw payload fields',
    ).toBe(0);
    for (const failureMessage of failureMessages) {
      expect(failureMessage).toContain('duplicate QQ assertion');
      expect(failureMessage).toContain('qq_lifetime_used');
      expect(failureMessage).toContain('response body redacted');
      expect(failureMessage).not.toContain('openrouter_api_key');
    }
  });

  it('does not include raw response bodies in requestJson non-ok failure messages', async () => {
    const message = await captureRequestJsonFailureMessage({
      responseText: buildSensitiveFailurePayloadText(),
      status: 502,
      targetPath: '/v1/auth/qq/assert',
    });

    expect(message).toContain('POST /v1/auth/qq/assert');
    expect(message).toContain('502');
    expect(message).toContain('failed');
    expect(message).toContain('response body redacted');
    expectFailureMessageExcludesSensitiveSentinels(message);
  });

  it('does not include raw response bodies in requestJson non-JSON failure messages', async () => {
    const message = await captureRequestJsonFailureMessage({
      responseText: buildSensitiveFailurePayloadText(),
      status: 200,
      targetPath: '/healthz',
    });

    expect(message).toContain('POST /healthz');
    expect(message).toContain('200');
    expect(message).toContain('non-JSON');
    expect(message).toContain('response body redacted');
    expectFailureMessageExcludesSensitiveSentinels(message);
  });
});

describeDeploySmoke('broker direct deploy smoke', () => {
  it('passes the canonical workers.dev trial flow', async () => {
    const liveInputs = readLiveDeploySmokeInputs({
      baseUrl: smokeBaseUrl,
      canonicalWorkerName: CANONICAL_WORKER_NAME,
      disallowedModel: smokeDisallowedModel,
      managedAllowlistedModels: MANAGED_ALLOWLIST_MODELS,
      qqAuthHmacPsk: smokeQqAuthHmacPsk,
    });
    const { baseUrl, disallowedModel, qqAuthHmacPsk } = liveInputs;

    const keyPair = await createDeviceKeyPair();
    const installationId = `deploy-smoke-${crypto.randomUUID().replace(/-/gu, '')}`.slice(
      0,
      64,
    );
    const appVersion = 'deploy-smoke-1.0.0';
    const hardwareHash = `deploy-smoke-hardware-${crypto.randomUUID()}`.slice(0, 96);

    const healthz = await requestJson({
      method: 'GET',
      url: new URL('/healthz', baseUrl),
    });
    expect(healthz.status).toBe(200);
    expect(healthz.body.ok).toBe(true);
    expect(healthz.body.service).toBe(BROKER_SERVICE_NAME);

    const foundation = await requestJson({
      method: 'GET',
      url: new URL('/v1/foundation', baseUrl),
    });
    expect(foundation.status).toBe(200);
    expect(foundation.body.service).toBe(BROKER_SERVICE_NAME);
    expect(foundation.body.trialProviderPolicy?.managedFreeTrial?.provider).toBe(
      'OpenRouter',
    );
    expect(foundation.body.trialProviderPolicy?.managedFreeTrial?.models).toEqual(
      expect.arrayContaining([...MANAGED_ALLOWLIST_MODELS]),
    );

    const qqIdentity = `deploy-smoke-qq-${crypto.randomUUID()}`;
    const qqCredential = await computeHmacSha256Hex(
      qqAuthHmacPsk,
      qqIdentity,
    );
    const qqAssertion = await requestJson({
      method: 'POST',
      url: new URL('/v1/auth/qq/assert', baseUrl),
      body: {
        qq_identity: qqIdentity,
        credential: qqCredential,
        asserted_at: new Date().toISOString(),
      },
    });
    expect(qqAssertion.status).toBe(200);
    expect(qqAssertion.body.ok).toBe(true);
    expect(qqAssertion.body.status).toBe('issued');
    const qqIssuedKey = assertQqIssuedResponse(qqAssertion.body);

    const duplicateQqAssertion = await requestJsonAllowFailure({
      method: 'POST',
      url: new URL('/v1/auth/qq/assert', baseUrl),
      body: {
        qq_identity: qqIdentity,
        credential: qqCredential,
        asserted_at: new Date().toISOString(),
      },
    });
    expect(duplicateQqAssertion.status).toBe(409);
    assertDuplicateQqLifetimeUsedResponse(duplicateQqAssertion.body);

    const challenge = await requestJson({
      method: 'POST',
      url: new URL('/v1/trial/challenge', baseUrl),
      body: {
        installation_id: installationId,
        device_public_key: keyPair.devicePublicKey,
        app_version: appVersion,
      },
    });
    expect(challenge.status).toBe(200);
    expect(typeof challenge.body.challenge).toBe('string');
    expect(typeof challenge.body.challenge_expires_at).toBe('string');
    expect(challenge.body.managed_state?.lifecycle).toBe('none');
    expect(challenge.body.fingerprint_salt?.current?.salt).not.toBe(
      BOOTSTRAP_PLACEHOLDER,
    );

    const verifySignedAt = timestampFromHeaders(challenge.headers);
    const verifyRequest = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: installationId,
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.body.challenge,
      challenge_expires_at: challenge.body.challenge_expires_at,
      hardware_hash: hardwareHash,
      app_version: appVersion,
      signed_at: verifySignedAt,
    });
    const verify = await requestJson({
      method: 'POST',
      url: new URL('/v1/trial/challenge/verify', baseUrl),
      body: verifyRequest,
    });
    expect(verify.status).toBe(200);
    expect(typeof verify.body.release_token).toBe('string');
    expect(typeof verify.body.release_token_expires_at).toBe('string');
    expect(verify.body.managed_state?.lifecycle).toBe('pending_release');
    expect(verify.body.managed_state?.managed_availability).toBe(true);

    const statusTimestamp = timestampFromHeaders(verify.headers);
    const statusRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: installationId,
      timestamp: statusTimestamp,
    });
    const statusUrl = new URL('/v1/trial/status', baseUrl);
    statusUrl.searchParams.set('installation_id', installationId);
    const status = await requestJson({
      method: 'GET',
      url: statusUrl,
      headers: {
        [TRIAL_STATUS_TIMESTAMP_HEADER]: statusRequest.timestamp,
        [TRIAL_STATUS_SIGNATURE_HEADER]: statusRequest.signature,
      },
    });
    expect(status.status).toBe(200);
    expect(status.body.managed_state?.lifecycle).toBe('pending_release');
    expect(status.body.current_entitlement?.provider).toBe('OpenRouter');

    const issueSignedAt = timestampFromHeaders(status.headers);
    const issueRequest = await signCanonicalIssueRequest(keyPair.privateKey, {
      installation_id: installationId,
      device_public_key: keyPair.devicePublicKey,
      release_token: verify.body.release_token,
      hardware_hash: hardwareHash,
      reason: ISSUE_REASON,
      budget_usd: ISSUE_BUDGET_USD,
      model: ISSUE_MODEL,
      signed_at: issueSignedAt,
    });
    const issue = await requestJson({
      method: 'POST',
      url: new URL('/v1/providers/openrouter/issue', baseUrl),
      body: issueRequest,
    });
    expect(issue.status).toBe(200);
    expect(issue.body.managed_state?.lifecycle).toBe('active');
    expect(issue.body.managed_state?.managed_availability).toBe(true);
    expect(issue.body.budget_usd).toBe(ISSUE_BUDGET_USD);
    expect(issue.body.model).toBe(ISSUE_MODEL);
    expect(typeof issue.body.openrouter_api_key).toBe('string');
    expect(issue.body.openrouter_api_key.length).toBeGreaterThan(0);
    expect(typeof issue.body.managed_credential_ref).toBe('string');
    expect(issue.body.managed_credential_ref.length).toBeGreaterThan(0);
    expect(typeof issue.body.expires_at).toBe('string');
    assertManagedOpenRouterUserId(issue.body.openrouter_user_id);

    const issuedKeyMetadata = readOpenRouterCurrentKeyMetadata(
      (
        await requestJson({
          method: 'GET',
          url: new URL('/api/v1/key', OPENROUTER_API_BASE_URL),
          headers: {
            authorization: `Bearer ${qqIssuedKey.openrouterApiKey}`,
          },
        })
      ).body,
    );
    expect(issuedKeyMetadata.limit).toBe(ISSUE_BUDGET_USD);
    expect(Date.parse(issuedKeyMetadata.expiresAt)).toBe(Date.parse(qqIssuedKey.expiresAt));

    for (const managedModel of POSITIVE_ROUTING_PROBE_MODELS) {
      const managedModelProbe = await requestOpenRouterChatCompletion(
        qqIssuedKey.openrouterApiKey,
        managedModel,
        'Reply with the single word routed.',
      );

      assertSuccessfulChatCompletionResponse(managedModelProbe, managedModel);
    }

    const guardrailProbe = await requestOpenRouterChatCompletion(
      qqIssuedKey.openrouterApiKey,
      disallowedModel,
      'Reply with the single word blocked.',
    );
    expect(guardrailProbe.status).toBeGreaterThanOrEqual(400);
    expect(
      isDisallowedModelGuardrailFailure(guardrailProbe.status, guardrailProbe.body),
    ).toBe(true);
  }, 180_000);
});

type LiveDeploySmokeInputOptions = {
  baseUrl: string | undefined;
  canonicalWorkerName: string;
  disallowedModel: string | undefined;
  managedAllowlistedModels: readonly string[];
  qqAuthHmacPsk: string | undefined;
};

type LiveDeploySmokeInputs = {
  baseUrl: URL;
  disallowedModel: string;
  qqAuthHmacPsk: string;
};

type QqIssuedKey = {
  openrouterApiKey: string;
  managedCredentialRef: string;
  expiresAt: string;
};

const SENSITIVE_FAILURE_SENTINELS = {
  rawProviderPayloadText: 'SENTINEL_RAW_PROVIDER_PAYLOAD_DO_NOT_LEAK',
  rawBrokerMessageText: 'SENTINEL_RAW_BROKER_MESSAGE_DO_NOT_LEAK',
  keyLikeValue: 'sk-or-v1-deploy-smoke-raw-leak-sentinel',
  qqIdentity: 'deploy-smoke-qq-raw-identity-sentinel',
  credential: 'b'.repeat(64),
  subjectRef: 'ph-qq-subject-v1_rawSubjectSentinel',
} as const;

function buildSensitiveFailurePayloadText(): string {
  const {
    credential,
    keyLikeValue,
    qqIdentity,
    rawBrokerMessageText,
    rawProviderPayloadText,
    subjectRef,
  } = SENSITIVE_FAILURE_SENTINELS;

  return `${JSON.stringify({
    error: {
      details: {
        credential,
        openrouter_api_key: keyLikeValue,
        qq_identity: qqIdentity,
        qq_subject_ref: subjectRef,
        raw_broker_message: rawBrokerMessageText,
      },
      message: rawProviderPayloadText,
    },
  })} Authorization: Bearer ${keyLikeValue} qq_identity=${qqIdentity} credential=${credential}`;
}

function expectFailureMessageExcludesSensitiveSentinels(message: string): void {
  for (const sentinel of Object.values(SENSITIVE_FAILURE_SENTINELS)) {
    expect(message).not.toContain(sentinel);
  }
}

function countFailureMessageSensitiveSentinels(message: string): number {
  const sensitiveIndicators = new Set(
    Object.values(SENSITIVE_FAILURE_SENTINELS).flatMap((sentinel) => [
      sentinel,
      sentinel.slice(0, Math.min(24, sentinel.length)),
    ]),
  );

  return [...sensitiveIndicators].filter((indicator) => message.includes(indicator))
    .length;
}

function captureDuplicateQqAssertionFailureMessage(payload: unknown): string {
  try {
    assertDuplicateQqLifetimeUsedResponse(payload);
  } catch (error) {
    return readErrorMessage(error);
  }

  throw new Error('duplicate QQ assertion fixture should have failed');
}

async function captureRequestJsonFailureMessage({
  responseText,
  status,
  targetPath,
}: {
  responseText: string;
  status: number;
  targetPath: string;
}): Promise<string> {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(responseText, {
      status,
      headers: {
        'content-type': 'application/json',
      },
    })) as typeof fetch;

  try {
    await requestJson({
      method: 'POST',
      url: new URL(targetPath, 'https://puripuly-heart-broker.example.workers.dev'),
    });
  } catch (error) {
    return readErrorMessage(error);
  } finally {
    globalThis.fetch = originalFetch;
  }

  throw new Error('requestJson should have thrown for the deploy-smoke failure fixture');
}

function shouldRunDeploySmoke(rawValue: string | undefined): boolean {
  return rawValue === 'true';
}

function readLiveDeploySmokeInputs({
  baseUrl: rawBaseUrl,
  canonicalWorkerName,
  disallowedModel: rawDisallowedModel,
  managedAllowlistedModels,
  qqAuthHmacPsk: rawQqAuthHmacPsk,
}: LiveDeploySmokeInputOptions): LiveDeploySmokeInputs {
  const validationErrors: string[] = [];
  let baseUrl: URL | undefined;
  let disallowedModel: string | undefined;
  let qqAuthHmacPsk: string | undefined;

  try {
    baseUrl = normalizeSmokeBaseUrl(rawBaseUrl);
    validateCanonicalWorkersDevTarget(baseUrl, canonicalWorkerName);
  } catch (error) {
    validationErrors.push(readErrorMessage(error));
  }

  try {
    disallowedModel = normalizeDisallowedModel(
      rawDisallowedModel,
      managedAllowlistedModels,
      true,
    );
  } catch (error) {
    validationErrors.push(readErrorMessage(error));
  }

  try {
    qqAuthHmacPsk = requireQqAuthHmacPsk(rawQqAuthHmacPsk);
  } catch (error) {
    validationErrors.push(readErrorMessage(error));
  }

  if (validationErrors.length > 0) {
    throw new Error(
      `Deploy smoke live inputs are incomplete: ${validationErrors.join('; ')}`,
    );
  }

  return {
    baseUrl: baseUrl as URL,
    disallowedModel: disallowedModel as string,
    qqAuthHmacPsk: qqAuthHmacPsk as string,
  };
}

function readErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function assertManagedOpenRouterUserId(value: unknown): asserts value is string {
  if (
    typeof value !== 'string' ||
    value.trim().length === 0 ||
    !MANAGED_OPENROUTER_USER_ID_PATTERN.test(value)
  ) {
    throw new Error('issue success payload must include a valid openrouter_user_id');
  }
}

function assertQqSubjectRef(payload: unknown): void {
  const body = readRecord(payload, 'QQ assertion response');
  const subjectRef = body.qq_subject_ref;

  if (typeof subjectRef !== 'string' || !QQ_SUBJECT_REF_PATTERN.test(subjectRef)) {
    throw new Error('QQ assertion response must include a valid qq_subject_ref');
  }
}

function assertQqIssuedResponse(payload: unknown): QqIssuedKey {
  const body = readRecord(payload, 'QQ issued response');

  if (body.ok !== true || body.status !== 'issued') {
    throw new Error('QQ issued response must have ok true and status issued');
  }

  assertQqSubjectRef(body);
  assertManagedOpenRouterUserId(body.openrouter_user_id);

  const openrouterApiKey = body.openrouter_api_key;
  const managedCredentialRef = body.managed_credential_ref;
  const expiresAt = body.expires_at;

  if (typeof openrouterApiKey !== 'string' || openrouterApiKey.length === 0) {
    throw new Error('QQ issued response must include a one-time openrouter_api_key');
  }

  if (typeof managedCredentialRef !== 'string' || managedCredentialRef.length === 0) {
    throw new Error('QQ issued response must include managed_credential_ref');
  }

  if (typeof expiresAt !== 'string' || Number.isNaN(Date.parse(expiresAt))) {
    throw new Error('QQ issued response must include a valid expires_at timestamp');
  }

  return {
    openrouterApiKey,
    managedCredentialRef,
    expiresAt,
  };
}

function assertDuplicateQqLifetimeUsedResponse(payload: unknown): void {
  const body = readRecord(payload, 'duplicate QQ assertion response');
  const errorSubcode = readPublicErrorSubcode(body.error);
  const includesOneTimeKey = Object.prototype.hasOwnProperty.call(
    body,
    'openrouter_api_key',
  );

  if (errorSubcode !== 'qq_lifetime_used' || includesOneTimeKey) {
    throw new Error(
      'duplicate QQ assertion response must return qq_lifetime_used without a one-time key; response body redacted',
    );
  }
}

function readPublicErrorSubcode(error: unknown): string | undefined {
  if (!isRecord(error)) {
    return undefined;
  }

  return typeof error.subcode === 'string' ? error.subcode : undefined;
}

function requireQqAuthHmacPsk(value: string | undefined): string {
  if (value === undefined || value.trim().length === 0) {
    throw new Error('BROKER_DEPLOY_SMOKE_QQ_AUTH_HMAC_PSK is required for deploy smoke');
  }

  return value;
}

function normalizeDisallowedModel(
  rawValue: string | undefined,
  managedAllowlistedModels: readonly string[],
  isRequired: boolean,
): string | undefined {
  const normalized = rawValue?.trim();

  if (!normalized) {
    if (isRequired) {
      throw new Error(
        'BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL is required for deploy smoke',
      );
    }

    return undefined;
  }

  if (managedAllowlistedModels.includes(normalized)) {
    throw new Error(
      'BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL must differ from the managed allowlisted models',
    );
  }

  return normalized;
}

function requireDisallowedModel(model: string | undefined): string {
  if (!model) {
    throw new Error('BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL is required for deploy smoke');
  }

  return model;
}

function normalizeSmokeBaseUrl(baseUrl: string | undefined): URL {
  if (!baseUrl) {
    throw new Error('BROKER_DEPLOY_SMOKE_BASE_URL is required for deploy smoke');
  }

  return new URL(baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`);
}

function validateCanonicalWorkersDevTarget(baseUrl: URL, canonicalWorkerName: string): void {
  if (baseUrl.protocol !== 'https:') {
    throw new Error('deploy smoke must target an https workers.dev URL');
  }

  if (!baseUrl.hostname.endsWith('.workers.dev')) {
    throw new Error('deploy smoke must target the canonical workers.dev hostname');
  }

  if (!baseUrl.hostname.startsWith(`${canonicalWorkerName}.`)) {
    throw new Error(
      `deploy smoke must target the canonical worker ${canonicalWorkerName}`,
    );
  }
}

function timestampFromHeaders(headers: Headers): string {
  const headerValue = headers.get('date');

  if (headerValue) {
    const parsed = Date.parse(headerValue);

    if (!Number.isNaN(parsed)) {
      return new Date(parsed).toISOString();
    }
  }

  return new Date().toISOString();
}

async function computeHmacSha256Hex(secret: string, value: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    'raw',
    textEncoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign('HMAC', key, textEncoder.encode(value));

  return Array.from(new Uint8Array(signature), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('');
}

function readOpenRouterCurrentKeyMetadata(payload: unknown): {
  limit: number;
  expiresAt: string;
} {
  const data = readObjectField(payload, 'data', 'OpenRouter current-key response');
  const { limit } = data;
  const expiresAt = data.expires_at;

  if (typeof limit !== 'number' || !Number.isFinite(limit)) {
    throw new Error('OpenRouter current-key response must include a numeric data.limit');
  }

  if (typeof expiresAt !== 'string' || Number.isNaN(Date.parse(expiresAt))) {
    throw new Error(
      'OpenRouter current-key response must include a valid ISO timestamp in data.expires_at',
    );
  }

  return {
    limit,
    expiresAt,
  };
}

function assertSuccessfulChatCompletionResponse(
  response: { status: number; body: unknown },
  requestedModel: string,
): void {
  if (response.status !== 200) {
    throw new Error(
      `Expected successful chat completion for ${requestedModel}, got ${response.status}; response body redacted`,
    );
  }

  const payload = readRecord(
    response.body,
    `OpenRouter chat completion response for ${requestedModel}`,
  );
  const choices = readArrayField(
    payload,
    'choices',
    `OpenRouter chat completion response for ${requestedModel}`,
  );

  if (typeof payload.id !== 'string' || payload.id.length === 0) {
    throw new Error(
      `OpenRouter chat completion response for ${requestedModel} must include a non-empty id`,
    );
  }

  if (choices.length === 0) {
    throw new Error(
      `OpenRouter chat completion response for ${requestedModel} must include at least one choice`,
    );
  }

  const firstChoice = readRecord(
    choices[0],
    `OpenRouter first chat completion choice for ${requestedModel}`,
  );
  const message = readObjectField(
    firstChoice,
    'message',
    `OpenRouter first chat completion choice for ${requestedModel}`,
  );

  if (message.role !== 'assistant') {
    throw new Error(
      `OpenRouter chat completion response for ${requestedModel} must include an assistant message`,
    );
  }

  if (
    !hasNonEmptyChatCompletionContent(message.content) &&
    !EMPTY_CONTENT_ALLOWED_POSITIVE_ROUTING_MODELS.has(requestedModel)
  ) {
    throw new Error(
      `OpenRouter chat completion response for ${requestedModel} must include non-empty assistant content`,
    );
  }
}

function isDisallowedModelGuardrailFailure(status: number, body: unknown): boolean {
  if (status < 400 || status === 401) {
    return false;
  }

  return /allowed model|disallowed model|model\/provider|model[^\n]*available|provider[^\n]*available|guardrail|route/iu.test(
    stringifyForPatternMatch(body),
  );
}

function readRecord(value: unknown, context: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`${context} must be a JSON object`);
  }

  return value;
}

function readArrayField(
  value: unknown,
  fieldName: string,
  context: string,
): unknown[] {
  if (!isRecord(value) || !Array.isArray(value[fieldName])) {
    throw new Error(`${context} must include an array ${fieldName}`);
  }

  return value[fieldName] as unknown[];
}

function readObjectField(
  value: unknown,
  fieldName: string,
  context: string,
): Record<string, unknown> {
  if (!isRecord(value) || !isRecord(value[fieldName])) {
    throw new Error(`${context} must include an object ${fieldName}`);
  }

  return value[fieldName] as Record<string, unknown>;
}

function hasNonEmptyChatCompletionContent(content: unknown): boolean {
  if (typeof content === 'string') {
    return content.trim().length > 0;
  }

  return Array.isArray(content) && content.length > 0;
}

function stringifyForPatternMatch(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function requestJson({ method, url, body, headers = {} }: JsonRequestOptions) {
  const response = await fetch(url, {
    method,
    headers: {
      ...(body !== undefined ? { 'content-type': 'application/json' } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const rawText = await response.text();

  if (!response.ok) {
    throw new Error(
      formatRequestJsonFailureMessage(method, url.pathname, response.status, 'failed'),
    );
  }

  try {
    return {
      status: response.status,
      headers: response.headers,
      body: JSON.parse(rawText),
    };
  } catch {
    throw new Error(
      formatRequestJsonFailureMessage(
        method,
        url.pathname,
        response.status,
        'returned non-JSON',
      ),
    );
  }
}

function formatRequestJsonFailureMessage(
  method: string,
  path: string,
  status: number,
  context: 'failed' | 'returned non-JSON',
): string {
  return `${method} ${path} ${context} with ${status}; response body redacted`;
}

async function requestJsonAllowFailure({
  method,
  url,
  body,
  headers = {},
}: JsonRequestOptions) {
  const response = await fetch(url, {
    method,
    headers: {
      ...(body !== undefined ? { 'content-type': 'application/json' } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const rawText = await response.text();

  try {
    return {
      status: response.status,
      headers: response.headers,
      body: JSON.parse(rawText),
    };
  } catch {
    return {
      status: response.status,
      headers: response.headers,
      body: redactIssueBody(rawText),
    };
  }
}

async function requestOpenRouterChatCompletion(
  apiKey: string,
  model: string,
  prompt: string,
) {
  return requestJsonAllowFailure({
    method: 'POST',
    url: new URL('/api/v1/chat/completions', OPENROUTER_API_BASE_URL),
    headers: {
      authorization: `Bearer ${apiKey}`,
    },
    body: {
      model,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
      max_tokens: 8,
    },
  });
}

function redactIssueBody(rawText: string): string {
  return rawText
    .replace(
      /"openrouter_api_key"\s*:\s*"[^"]+"/gu,
      '"openrouter_api_key":"[REDACTED]"',
    )
    .replace(/"qq_identity"\s*:\s*"[^"]+"/gu, '"qq_identity":"[REDACTED]"')
    .replace(/"credential"\s*:\s*"[^"]+"/gu, '"credential":"[REDACTED]"')
    .replace(
      /"qq_subject_ref"\s*:\s*"[^"]+"/gu,
      '"qq_subject_ref":"[REDACTED]"',
    )
    .replace(/Bearer\s+sk-or-[A-Za-z0-9._~-]+/giu, 'Bearer [REDACTED]')
    .replace(/sk-or-v1-[A-Za-z0-9._~-]+/gu, '[REDACTED]')
    .replace(/\bqq_identity=[^\s&]+/gu, 'qq_identity=[REDACTED]')
    .replace(/\bcredential=[0-9a-f]{64}\b/giu, 'credential=[REDACTED]')
    .replace(/ph-qq-subject-v1_[A-Za-z0-9_-]+/gu, '[REDACTED]');
}
