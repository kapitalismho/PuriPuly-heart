import { describe, expect, it, vi } from 'vitest';

import {
  assignManagedGuardrail,
  cleanupManagedChildKey,
  createManagedChildKey,
} from '../src/openrouter-management';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('OpenRouter management client', () => {
  it('creates a child key and returns the raw key plus hash', async () => {
    const fetchMock = vi.fn<typeof fetch>(async () =>
      jsonResponse(
        {
          key: 'or-child-key-123',
          data: {
            hash: 'hash_123',
            expires_at: '2026-10-10T06:00:00.000Z',
            limit: 0.07,
            limit_reset: null,
          },
        },
        201,
      ),
    );

    const result = await createManagedChildKey({
      managementApiKey: 'mgmt-key',
      installationId: 'install-123',
      releaseSessionRef: 'session-123',
      expiresAt: '2026-10-10T06:00:00.000Z',
      fetchImpl: fetchMock,
    });

    expect(result).toEqual({ rawKey: 'or-child-key-123', hash: 'hash_123' });

    expect(fetchMock).toHaveBeenCalledWith(
      'https://openrouter.ai/api/v1/keys',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: 'Bearer mgmt-key',
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify({
          name: 'puripuly-heart:install-123:session-123',
          limit: 0.07,
          limit_reset: null,
          include_byok_in_limit: false,
          expires_at: '2026-10-10T06:00:00.000Z',
        }),
      }),
    );
  });

  it('creates a QQ child key with source-aware metadata without installation identity', async () => {
    const fetchMock = vi.fn<typeof fetch>(async () =>
      jsonResponse(
        {
          key: 'or-qq-managed-child-key-123',
          data: {
            hash: 'hash_qq_managed_child_123',
            expires_at: '2026-10-10T06:00:00.000Z',
            limit: 0.07,
            limit_reset: null,
          },
        },
        201,
      ),
    );
    const qqSubjectRef = 'ph-qq-subject-v1_synthetic-subject-ref';
    const issueRef = 'qq-issue-ref-20260610T060000Z';

    const result = await createManagedChildKey({
      managementApiKey: 'mgmt-key',
      issueSource: 'qq',
      subjectRef: qqSubjectRef,
      issueRef,
      expiresAt: '2026-10-10T06:00:00.000Z',
      fetchImpl: fetchMock,
    });

    expect(result).toEqual({
      rawKey: 'or-qq-managed-child-key-123',
      hash: 'hash_qq_managed_child_123',
    });

    const requestBody = JSON.parse(
      String((fetchMock.mock.calls[0]?.[1] as RequestInit | undefined)?.body),
    ) as { name?: string; limit?: number };
    expect(requestBody).toEqual(
      expect.objectContaining({
        name: `puripuly-heart:qq:${issueRef}`,
        limit: 0.07,
      }),
    );
    expect(requestBody.name).not.toContain(qqSubjectRef);
    expect(requestBody.name).not.toContain('installation');
  });

  it('creates a child key with an explicit absolute limit and verifies the effective limit', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(
        {
          key: 'or-child-key-123',
          data: {
            hash: 'hash_123',
            expires_at: '2026-10-10T06:00:00.000Z',
            limit: 0.09,
            limit_reset: null,
          },
        },
        201,
      ),
    );

    const result = await createManagedChildKey({
      managementApiKey: 'mgmt-key',
      installationId: 'install-123',
      releaseSessionRef: 'session-123',
      expiresAt: '2026-10-10T06:00:00.000Z',
      limitUsd: 0.09,
      requireEffectiveLimitVerification: true,
      fetchImpl: fetchMock,
    });

    expect(result).toEqual({ rawKey: 'or-child-key-123', hash: 'hash_123' });
    expect(fetchMock).toHaveBeenCalledWith(
      'https://openrouter.ai/api/v1/keys',
      expect.objectContaining({
        body: JSON.stringify({
          name: 'puripuly-heart:install-123:session-123',
          limit: 0.09,
          limit_reset: null,
          include_byok_in_limit: false,
          expires_at: '2026-10-10T06:00:00.000Z',
        }),
      }),
    );
  });

  it('rejects explicit-limit child keys when the effective limit is below the requested limit', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(
        {
          key: 'or-child-key-123',
          data: {
            hash: 'hash_123',
            limit: 0.07,
          },
        },
        201,
      ),
    );

    await expect(
      createManagedChildKey({
        managementApiKey: 'mgmt-key',
        installationId: 'install-123',
        releaseSessionRef: 'session-123',
        expiresAt: '2026-10-10T06:00:00.000Z',
        limitUsd: 0.09,
        requireEffectiveLimitVerification: true,
        fetchImpl: fetchMock,
      }),
    ).rejects.toMatchObject({
      name: 'OpenRouterManagementError',
      operation: 'create_key',
      code: 'malformed_upstream',
      status: 201,
      message: expect.stringContaining('effective limit'),
    });
  });

  it('assigns the managed guardrail to the created key hash', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ assigned_count: 1 }));

    await assignManagedGuardrail({
      managementApiKey: 'mgmt-key',
      guardrailId: 'guardrail-123',
      keyHash: 'hash_123',
      fetchImpl: fetchMock,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      'https://openrouter.ai/api/v1/guardrails/guardrail-123/assignments/keys',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: 'Bearer mgmt-key',
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify({
          key_hashes: ['hash_123'],
        }),
      }),
    );
  });

  it('reports malformed upstream create responses clearly', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(
        {
          data: {
            hash: 'hash_123',
          },
        },
        201,
      ),
    );

    await expect(
      createManagedChildKey({
        managementApiKey: 'mgmt-key',
        installationId: 'install-123',
        releaseSessionRef: 'session-123',
        expiresAt: '2026-10-10T06:00:00.000Z',
        fetchImpl: fetchMock,
      }),
    ).rejects.toMatchObject({
      name: 'OpenRouterManagementError',
      operation: 'create_key',
      code: 'malformed_upstream',
      status: 201,
      message: expect.stringContaining('key'),
    });
  });

  it('disables and deletes the child key during cleanup', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ data: { hash: 'hash_123', disabled: true } }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));

    await expect(
      cleanupManagedChildKey({
        managementApiKey: 'mgmt-key',
        keyHash: 'hash_123',
        fetchImpl: fetchMock,
      }),
    ).resolves.toEqual({ ok: true });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'https://openrouter.ai/api/v1/keys/hash_123',
      expect.objectContaining({
        method: 'PATCH',
        headers: expect.objectContaining({
          Authorization: 'Bearer mgmt-key',
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify({ disabled: true }),
      }),
    );

    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'https://openrouter.ai/api/v1/keys/hash_123',
      expect.objectContaining({
        method: 'DELETE',
        headers: expect.objectContaining({
          Authorization: 'Bearer mgmt-key',
        }),
      }),
    );
  });

  it('surfaces malformed disable success responses instead of reporting cleanup success', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ data: { hash: 'hash_123' } }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));

    await expect(
      cleanupManagedChildKey({
        managementApiKey: 'mgmt-key',
        keyHash: 'hash_123',
        fetchImpl: fetchMock,
      }),
    ).resolves.toEqual({
      ok: false,
      reason: {
        disable: {
          ok: false,
          error: {
            operation: 'disable_key',
            code: 'malformed_upstream',
            status: 200,
            upstreamCode: null,
            message: expect.stringContaining('disabled'),
          },
        },
        delete: { ok: true },
      },
    });
  });

  it('surfaces malformed delete success responses instead of reporting cleanup success', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ data: { hash: 'hash_123', disabled: true } }))
      .mockResolvedValueOnce(jsonResponse({ deleted: false }));

    await expect(
      cleanupManagedChildKey({
        managementApiKey: 'mgmt-key',
        keyHash: 'hash_123',
        fetchImpl: fetchMock,
      }),
    ).resolves.toEqual({
      ok: false,
      reason: {
        disable: { ok: true },
        delete: {
          ok: false,
          error: {
            operation: 'delete_key',
            code: 'malformed_upstream',
            status: 200,
            upstreamCode: null,
            message: expect.stringContaining('deleted'),
          },
        },
      },
    });
  });

  it('surfaces cleanup failure details for orphan-audit handling', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ data: { hash: 'hash_123', disabled: true } }))
      .mockResolvedValueOnce(
        jsonResponse(
          {
            error: {
              code: 500,
              message: 'delete failed',
            },
          },
          500,
        ),
      );

    await expect(
      cleanupManagedChildKey({
        managementApiKey: 'mgmt-key',
        keyHash: 'hash_123',
        fetchImpl: fetchMock,
      }),
    ).resolves.toEqual({
      ok: false,
      reason: {
        disable: { ok: true },
        delete: {
          ok: false,
          error: {
            operation: 'delete_key',
            code: 'upstream_http_error',
            status: 500,
            upstreamCode: 500,
            message: 'delete failed',
          },
        },
      },
    });
  });
});
