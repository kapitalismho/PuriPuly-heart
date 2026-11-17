import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, describe, expect, it, vi } from 'vitest';

import relay from '../../agent-relay/src/index';
import { authHeaders, relayEvents, responseEvents, tokenSecret, turnId } from '../../agent-relay/tests/fixtures';
import { runDailyReport } from '../src/scheduled';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

const networkFetch = globalThis.fetch;
const reportTime = new Date('2026-11-18T00:05:00.000Z');

type WebhookBody = {
  content: string;
  embeds: Array<{ title: string; fields: Array<{ name: string; value: string }>; color: number }>;
};

async function webhook(run: (url: string, received: WebhookBody[]) => Promise<void>) {
  const received: WebhookBody[] = [];
  const server = createServer(async (request, response) => {
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(Buffer.from(chunk));
    received.push(JSON.parse(Buffer.concat(chunks).toString('utf8')));
    response.writeHead(204).end();
  });
  const ready = Promise.withResolvers<void>();
  server.once('error', ready.reject);
  server.listen(0, '127.0.0.1', ready.resolve);
  await ready.promise;
  try {
    await run(`http://127.0.0.1:${(server.address() as AddressInfo).port}/webhook`, received);
  } finally {
    const closed = Promise.withResolvers<void>();
    server.close((error) => error ? closed.reject(error) : closed.resolve());
    await closed.promise;
  }
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('daily model usage delivery', () => {
  it('delivers completed Relay calls and separate feature costs through the webhook without identities or secrets', async () => {
    const env = createTestBrokerEnv();
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-11-17T23:59:00.000Z'));
    env.__usageDb.prepare('INSERT INTO terra_daily (utc_date, tokens) VALUES (?, ?)').run('2026-11-17', 10_000_000);
    const modelEnv = {
      AGENT_USAGE_DB: env.AGENT_USAGE_DB,
      PURI_TOKEN_SECRET: tokenSecret,
      OPENAI_API_KEY: 'synthetic-chat-key',
      PHOTO_OPENAI_API_KEY: 'synthetic-photo-key',
      AGENT_SERVICE_ENABLED: 'true',
      PHOTO_SERVICE_ENABLED: 'true',
      AGENT_USAGE_RATES: JSON.stringify({ as_of: '2026-11-17', input: 1, cache_read: 0.1,
        cache_write: 1.25, responses_output: 4, decisions_output: 0 }),
    };
    const input = [{ role: 'user', content: JSON.stringify({ current_message: 'synthetic request',
      recent_messages: [], active_task: null, previous_modules: [], last_puri_action: null }) }];
    const invoke = (path: string, body: unknown) => relay.fetch(new Request(`https://relay.test${path}`, {
      method: 'POST', headers: authHeaders(), body: JSON.stringify(body),
    }), modelEnv);
    try {
      await webhook(async (url, received) => {
        env.DISCORD_DAILY_REPORT_WEBHOOK_URL = url;
        vi.spyOn(globalThis, 'fetch').mockImplementation(async (target, init) => {
          if (String(target) === url) return networkFetch(target, init);
          expect(String(target)).toMatch(/^https:\/\/api\.openai\.com\/v1\/(decisions|responses)$/);
          const photo = new Headers(init?.headers).get('authorization') === 'Bearer synthetic-photo-key';
          if (String(target).endsWith('/decisions')) {
            return Response.json({ answers: [], usage: { input_tokens: 1000, output_tokens: 0 } });
          }
          return responseEvents({ object: 'response', status: 'completed', output: [],
            usage: { input_tokens: photo ? 3000 : 2000, output_tokens: photo ? 1000 : 500 } });
        });
        expect((await invoke('/v1/classify', { agent_protocol: 6, turn_id: turnId, input })).status).toBe(200);
        for (const [path, body] of [
          ['/v1/respond', { agent_protocol: 6, turn_id: turnId,
            input: [{ type: 'configuration_update', reasoning: { effort: 'high' } }, ...input],
            route: 'work', effort: 'high', allow_tools: true }],
          ['/v1/photo', { photo_protocol: 1, request_id: '1'.repeat(32), image_base64: '/9j/AAAA', target_language: 'ko' }],
        ] as const) {
          const response = await invoke(path, body);
          expect(response.status).toBe(200);
          expect((await relayEvents(response)).at(-1)?.type).toBe('completed');
        }
        const result = await runDailyReport(env, reportTime);
        expect(result.sent).toBe(true);
        expect(result.payload?.summary.model_usage).toEqual({
          chatbot: { classification_calls: 1, response_calls: 1, missing_usage_calls: 0, cost_usd: 0.005 },
          photo: { requests: 1, unsettled_requests: 0, missing_usage_requests: 0, estimated_date_requests: 0, cost_usd: 0.007 },
          total_cost_usd: 0.012,
        });
        expect(received).toHaveLength(1);
        const message = received[0];
        expect(JSON.parse(message.content.slice('```json\n'.length, -'\n```'.length))).toEqual(result.payload);
        expect(message.content.length).toBeLessThanOrEqual(2000);
        expect(message.embeds[0].fields).toEqual(expect.arrayContaining([
          expect.objectContaining({ name: 'Chatbot', value: 'recorded_calls=2\nresponses=1\nclassifications=1\ncost_usd=0.005000' }),
          expect.objectContaining({ name: 'Photo translation', value: 'requests=1\ncost_usd=0.007000' }),
          expect.objectContaining({ name: 'Total cost (USD)', value: '0.012000' }),
        ]));
        for (const secret of [modelEnv.OPENAI_API_KEY, modelEnv.PHOTO_OPENAI_API_KEY, tokenSecret, 'installation_hash', 'request_id', '/9j/AAAA']) {
          expect(JSON.stringify(message)).not.toContain(secret);
        }
        expect(env.__db.prepare('SELECT status FROM broker_daily_summary_deliveries').get()?.status).toBe('delivered');
        expect((await runDailyReport(env, reportTime)).sent).toBe(false);
        expect(received).toHaveLength(1);
      });
    } finally {
      env.__usageDb.close();
      env.__db.close();
    }
  });

  it('retains a failed usage aggregation for retry instead of delivering zero usage', async () => {
    const env = createTestBrokerEnv();
    const outgoing = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(null, { status: 204 }));
    env.__usageDb.prepare(`INSERT INTO usage_daily
      (installation_hash, utc_date, responses_calls, cost_usd) VALUES (?, ?, ?, ?)`)
      .run('private-installation', '2026-11-17', 3, 0.25);
    env.__usageDb.exec('ALTER TABLE photo_admissions RENAME TO temporarily_unavailable');
    try {
      await expect(runDailyReport(env, reportTime)).rejects.toThrow();
      expect(outgoing).not.toHaveBeenCalled();
      expect(env.__db.prepare('SELECT status, delivered_at FROM broker_daily_summary_deliveries').get())
        .toMatchObject({ status: 'pending', delivered_at: null });
      env.__usageDb.exec('ALTER TABLE temporarily_unavailable RENAME TO photo_admissions');
      const retried = await runDailyReport(env, new Date('2026-11-18T00:06:00.000Z'));
      expect(retried.payload?.summary.model_usage.chatbot).toMatchObject({ response_calls: 3, cost_usd: 0.25 });
      expect(outgoing).toHaveBeenCalledTimes(1);
      expect(env.__db.prepare('SELECT status FROM broker_daily_summary_deliveries').get()?.status).toBe('delivered');
    } finally {
      env.__usageDb.close();
      env.__db.close();
    }
  });

  it('shows missing usage, unsettled requests and historical date uncertainty alongside the known cost', async () => {
    const env = createTestBrokerEnv();
    const outgoing = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(null, { status: 204 }));
    env.__usageDb.prepare(`INSERT INTO usage_daily
      (installation_hash, utc_date, responses_calls, missing_usage_count, cost_usd) VALUES (?, ?, ?, ?, ?)`)
      .run('private-installation', '2026-11-17', 2, 1, 0.125);
    const day = Date.parse('2026-11-17T00:00:00.000Z');
    env.__usageDb.prepare(`INSERT INTO photo_admissions
      (installation_hash, request_id, period_start, owner, expires_at, settled, missing_usage, requested_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)`).run('private-installation', 'legacy', day, 'legacy-owner', day + 1000, 1, 1, null);
    env.__usageDb.prepare(`INSERT INTO photo_admissions
      (installation_hash, request_id, period_start, owner, expires_at, settled, missing_usage, requested_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)`).run('private-installation', 'pending', day, 'pending-owner', day + 2000, 0, 1, day + 1000);
    try {
      const sent = await runDailyReport(env, reportTime);
      expect(sent.payload?.summary.model_usage.total_cost_usd).toBe(0.125);
      const body = JSON.parse(String(outgoing.mock.calls[0][1]?.body)) as WebhookBody;
      expect(body.embeds[0].fields).toContainEqual({
        name: 'Accounting gaps',
        value: 'chatbot_missing_usage=1\nphoto_missing_usage=1\nphoto_unsettled=1\nphoto_estimated_date=1',
        inline: false,
      });
      expect(body.embeds[0].color).toBe(0xfee75c);
    } finally {
      env.__usageDb.close();
      env.__db.close();
    }
  });
});
