import { describe, expect, it, vi } from 'vitest';

import { app } from '../src/app';
import {
  buildManagedOperationId,
  buildManagedOperationResumeToken,
  createManagedOperation,
  hashManagedOperationResumeToken,
} from '../src/managed-operation';
import { handleScheduled } from '../src/scheduled';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

describe('managed operation smoke', () => {
  it('serves the operation lifecycle over HTTP and converges the scheduled worker', async () => {
  const env = createTestBrokerEnv();
  const operationId = buildManagedOperationId();
  const resumeToken = buildManagedOperationResumeToken();
  await createManagedOperation(env.BROKER_DB, {
    operationId,
    resumeTokenHash: await hashManagedOperationResumeToken(resumeToken),
    issueSource: 'qq',
    subjectRef: 'ph-qq-subject-v1_smoke_operation_subject',
    installationId: null,
    devicePublicKey: null,
    now: new Date(),
  });

  const post = (path: string, body: unknown) =>
    app.request(`http://broker.test${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }, env);

  const status = await post('/v1/providers/openrouter/managed-operation/status', {
    operation_id: operationId,
    resume_token: resumeToken,
    installation_id: 'ignored-when-unbound',
  });
  expect(status.status).toBe(200);
  const statusBody = (await status.json()) as Record<string, unknown>;
  expect(statusBody).toMatchObject({ ok: true, state: 'AUTHENTICATED', client_action: 'wait' });

  const resume = await post('/v1/providers/openrouter/managed-operation/resume', {
    operation_id: operationId,
    resume_token: resumeToken,
    installation_id: 'ignored-when-unbound',
  });
  expect(resume.status).toBe(200);

  const unknown = await post('/v1/providers/openrouter/managed-operation/status', {
    operation_id: buildManagedOperationId(),
    resume_token: resumeToken,
    installation_id: 'x',
  });
  expect(unknown.status).toBe(404);

  const { updateAbuseControls } = await import('./test-support/abuse-controls');
  updateAbuseControls(env, (controls) => {
    controls.dailyReport.enabled = false;
  });
  const consoleSpy = vi.spyOn(console, 'info').mockImplementation(() => undefined);
  try {
    await handleScheduled({ scheduledTime: Date.parse('2026-09-01T12:00:00.000Z') }, env, {});
  } finally {
    consoleSpy.mockRestore();
  }

  const health = await app.request('http://broker.test/healthz', {}, env);
  expect(health.status).toBe(200);
});
});
