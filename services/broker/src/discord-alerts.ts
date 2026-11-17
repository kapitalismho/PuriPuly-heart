import type { AppUsageDailyMetrics } from './telemetry';
import type { RelayDailyUsage } from './relay-usage';

const DAILY_HEARTBEAT_COLOR_OK = 0x5865f2;
const DAILY_HEARTBEAT_COLOR_ACTIVE = 0xfee75c;
const DISCORD_EMBED_DESCRIPTION_LIMIT = 4096;

interface DiscordJsonCodeBlockInput {
  attachmentFilename: string;
  payload: unknown;
}

export interface DiscordEmbedField {
  name: string;
  value: string;
  inline?: boolean;
}

export interface DiscordEmbedInput {
  title: string;
  color: number;
  description?: string;
  content?: string;
  jsonCodeBlock?: DiscordJsonCodeBlockInput;
  fields: DiscordEmbedField[];
}

export interface DailyReportPayload {
  schema_version: 'puripuly_daily_summary.v2';
  report_date_utc: string;
  window_start: string;
  window_end: string;
  summary: AppUsageDailyMetrics & {
    keys_delivered_total: number;
    keys_delivered_discord: number;
    keys_delivered_qq: number;
    model_usage: RelayDailyUsage;
  };
}

type DiscordWebhookJsonBody = {
  content?: string;
  embeds: Array<{
    title: string;
    color: number;
    description?: string;
    fields: DiscordEmbedField[];
  }>;
};

type DiscordWebhookPayload =
  | {
      kind: 'json';
      body: DiscordWebhookJsonBody;
    }
  | {
      kind: 'multipart';
      body: FormData;
    };

export async function sendDiscordEmbed(
  webhookUrl: string,
  input: DiscordEmbedInput,
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  const payload = buildDiscordWebhookPayload(input);
  const response = await fetchImpl(
    webhookUrl,
    payload.kind === 'json'
      ? {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
          },
          body: JSON.stringify(payload.body),
        }
      : {
          method: 'POST',
          body: payload.body,
        },
  );

  if (!response.ok) {
    throw new Error(`discord webhook failed: ${response.status}`);
  }
}

export async function sendDailyReport(
  webhookUrl: string,
  packet: DailyReportPayload,
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  const summary = packet.summary;

  const usage = summary.model_usage;
  const gaps = accountingGaps(usage);
  await sendDiscordEmbed(
    webhookUrl,
    {
      title: `PuriPuly daily summary — ${packet.report_date_utc} UTC`,
      color: resolveDailyHeartbeatColor(packet, gaps.length > 0),
      description: `${packet.window_start} ≤ observed_at < ${packet.window_end}`,
      content: ['```json', JSON.stringify(packet), '```'].join('\n'),
      fields: [
        {
          name: 'Managed key issuance',
          value: [
            `keys_delivered=${summary.keys_delivered_total}`,
            `discord=${summary.keys_delivered_discord}`,
            `qq=${summary.keys_delivered_qq}`,
          ].join('\n'),
          inline: true,
        },
        {
          name: 'App usage',
          value: [
            `dau=${summary.app_dau}`,
            `wau=${summary.app_wau}`,
            `mau=${summary.app_mau}`,
          ].join('\n'),
          inline: true,
        },
        {
          name: 'Chatbot',
          value: [
            `recorded_calls=${usage.chatbot.classification_calls + usage.chatbot.response_calls}`,
            `responses=${usage.chatbot.response_calls}`,
            `classifications=${usage.chatbot.classification_calls}`,
            `cost_usd=${usage.chatbot.cost_usd.toFixed(6)}`,
          ].join('\n'),
          inline: true,
        },
        {
          name: 'Photo translation',
          value: [
            `requests=${usage.photo.requests}`,
            `cost_usd=${usage.photo.cost_usd.toFixed(6)}`,
          ].join('\n'),
          inline: true,
        },
        {
          name: 'Total cost (USD)',
          value: usage.total_cost_usd.toFixed(6),
          inline: true,
        },
        ...(gaps.length ? [{
          name: 'Accounting gaps',
          value: gaps.join('\n'),
          inline: false,
        }] : []),
      ],
    },
    fetchImpl,
  );
}

function buildDiscordWebhookPayload(input: DiscordEmbedInput): DiscordWebhookPayload {
  const compactJson = input.jsonCodeBlock
    ? JSON.stringify(input.jsonCodeBlock.payload)
    : null;
  const codeBlock = compactJson ? wrapJsonCodeBlock(compactJson) : null;

  if (
    codeBlock &&
    buildCombinedDescription(input.description, codeBlock).length >
      DISCORD_EMBED_DESCRIPTION_LIMIT
  ) {
    const deliveryCodeBlock = wrapJsonCodeBlock(
      JSON.stringify({
        delivery: 'attached_json_file',
        file: input.jsonCodeBlock?.attachmentFilename,
      }),
    );
    const body = buildJsonWebhookBody(
      input,
      buildCombinedDescription(input.description, deliveryCodeBlock),
    );
    const formData = new FormData();

    formData.set('payload_json', JSON.stringify(body));
    formData.set(
      'files[0]',
      new Blob([compactJson ?? ''], { type: 'application/json' }),
      input.jsonCodeBlock?.attachmentFilename ?? 'discord-payload.json',
    );

    return {
      kind: 'multipart',
      body: formData,
    };
  }

  return {
    kind: 'json',
    body: buildJsonWebhookBody(
      input,
      codeBlock ? buildCombinedDescription(input.description, codeBlock) : input.description,
    ),
  };
}

function buildJsonWebhookBody(
  input: DiscordEmbedInput,
  description: string | undefined,
): DiscordWebhookJsonBody {
  return {
    ...(input.content ? { content: input.content } : {}),
    embeds: [
      {
        title: input.title,
        color: input.color,
        ...(description ? { description } : {}),
        fields: input.fields,
      },
    ],
  };
}

function buildCombinedDescription(
  description: string | undefined,
  codeBlock: string,
): string {
  return description ? `${description}\n\n${codeBlock}` : codeBlock;
}

function wrapJsonCodeBlock(value: string): string {
  return ['```json', value, '```'].join('\n');
}

function accountingGaps(usage: RelayDailyUsage): string[] {
  const gaps: string[] = [];
  if (usage.chatbot.missing_usage_calls) gaps.push(`chatbot_missing_usage=${usage.chatbot.missing_usage_calls}`);
  if (usage.photo.missing_usage_requests) gaps.push(`photo_missing_usage=${usage.photo.missing_usage_requests}`);
  if (usage.photo.unsettled_requests) gaps.push(`photo_unsettled=${usage.photo.unsettled_requests}`);
  if (usage.photo.estimated_date_requests) gaps.push(`photo_estimated_date=${usage.photo.estimated_date_requests}`);
  return gaps;
}

function resolveDailyHeartbeatColor(packet: DailyReportPayload, hasAccountingGaps: boolean): number {
  if (packet.summary.keys_delivered_total > 0 || hasAccountingGaps) {
    return DAILY_HEARTBEAT_COLOR_ACTIVE;
  }

  return DAILY_HEARTBEAT_COLOR_OK;
}
