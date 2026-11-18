import { describe, expect, it, vi } from 'vitest';

import { sendDiscordEmbed } from '../src/discord-alerts';

describe('broker discord alert delivery', () => {
  it('inlines compact JSON code blocks into the embed description when they fit', async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));

    await sendDiscordEmbed(
      'https://discord.test/webhook',
      {
        title: 'Broker immediate abuse alert',
        color: 0xfee75c,
        description: 'Immediate broker abuse threshold crossing.',
        jsonCodeBlock: {
          attachmentFilename: 'broker-immediate-abuse-alert.json',
          payload: {
            schema_version: 'broker_abuse_interpretation_packet.v1',
            alert_id: 'alert-inline',
          },
        },
        fields: [],
      },
      fetchMock as unknown as typeof fetch,
    );

    expect(fetchMock).toHaveBeenCalledOnce();
    const request = (
      fetchMock.mock.calls as unknown as Array<[
        string | URL,
        RequestInit | undefined,
      ]>
    )[0]?.[1];

    if (!request) {
      throw new Error('expected fetch request init');
    }

    const body = JSON.parse(String(request.body)) as {
      content?: string;
      embeds: Array<{ description?: string }>;
    };

    expect(request.headers).toEqual({
      'content-type': 'application/json',
    });
    expect(body.content).toBeUndefined();
    expect(body.embeds[0]?.description).toContain(
      'Immediate broker abuse threshold crossing.',
    );
    expect(body.embeds[0]?.description).toContain('```json');
    expect(body.embeds[0]?.description).toContain(
      '"schema_version":"broker_abuse_interpretation_packet.v1"',
    );
  });

  it('falls back to a multipart JSON attachment when the compact code block would overflow Discord embed limits', async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    const oversizedPayload = {
      schema_version: 'broker_abuse_interpretation_packet.v1',
      alert_id: 'alert-attached',
      huge: 'x'.repeat(5000),
    };

    await sendDiscordEmbed(
      'https://discord.test/webhook',
      {
        title: 'Broker immediate abuse alert',
        color: 0xfee75c,
        description: 'Immediate broker abuse threshold crossing.',
        jsonCodeBlock: {
          attachmentFilename: 'broker-immediate-abuse-alert.json',
          payload: oversizedPayload,
        },
        fields: [],
      },
      fetchMock as unknown as typeof fetch,
    );

    expect(fetchMock).toHaveBeenCalledOnce();
    const request = (
      fetchMock.mock.calls as unknown as Array<[
        string | URL,
        RequestInit | undefined,
      ]>
    )[0]?.[1];

    if (!request) {
      throw new Error('expected fetch request init');
    }

    expect(request.body).toBeInstanceOf(FormData);

    const formData = request.body as FormData;
    const payloadJson = formData.get('payload_json');
    const payload = JSON.parse(String(payloadJson)) as {
      embeds: Array<{ description?: string }>;
    };
    const file = formData.get('files[0]');

    expect(payload.embeds[0]?.description).toContain('```json');
    expect(payload.embeds[0]?.description).toContain('"delivery":"attached_json_file"');
    expect(payload.embeds[0]?.description).toContain(
      '"file":"broker-immediate-abuse-alert.json"',
    );
    expect(file).toBeInstanceOf(File);
    expect((file as File).name).toBe('broker-immediate-abuse-alert.json');
    await expect((file as File).text()).resolves.toBe(JSON.stringify(oversizedPayload));
  });
});
