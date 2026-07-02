import type { BrokerAsnKind } from './abuse-controls';
import type { TelemetryUsageDailyMetrics } from './telemetry';

const DAILY_HEARTBEAT_COLOR_OK = 0x5865f2;
const DAILY_HEARTBEAT_COLOR_ACTIVE = 0xfee75c;
const DAILY_HEARTBEAT_COLOR_BRAKED = 0xed4245;
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
  schema_version: 'broker_daily_heartbeat.v1';
  generated_at: string;
  window_start_24h: string;
  window_end_24h: string;
  summary: {
    challenge_24h: number;
    verify_24h: number;
    issue_success_24h: number;
    highest_alert_level_24h: 'warn1' | 'warn2' | 'warn3' | 'critical' | null;
    brake_triggered_24h: boolean;
    top_asns: Array<{
      asn: number;
      count: number;
      share: number;
      kind: BrokerAsnKind;
      display_name: string | null;
    }>;
    cloud_asn_share_24h: number;
    manual_revocations_24h: number;
    translation_usage: TelemetryUsageDailyMetrics;
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
  const topAsnSummary =
    packet.summary.top_asns.length === 0
      ? 'none observed'
      : packet.summary.top_asns
          .map((entry) => {
            const displayName = entry.display_name
              ? ` (${entry.display_name})`
              : '';
            return `AS${entry.asn}${displayName}: ${entry.count} (${entry.share}%)`;
          })
          .join('\n');
  const translationUsage = packet.summary.translation_usage;
  const retentionSummary = [
    `D1=${formatRetentionSignal(translationUsage.retention.d1)}`,
    `D7=${formatRetentionSignal(translationUsage.retention.d7)}`,
    `D30=${formatRetentionSignal(translationUsage.retention.d30)}`,
  ].join('\n');

  await sendDiscordEmbed(
    webhookUrl,
    {
      title: 'Broker daily heartbeat',
      color: resolveDailyHeartbeatColor(packet),
      description: 'Daily operator heartbeat for broker abuse monitoring.',
      content: ['```json', JSON.stringify(packet), '```'].join('\n'),
      fields: [
        {
          name: '24h request counts',
          value: [
            `challenge=${packet.summary.challenge_24h}`,
            `verify=${packet.summary.verify_24h}`,
            `issue_success=${packet.summary.issue_success_24h}`,
          ].join('\n'),
          inline: true,
        },
        {
          name: 'Alert + brake',
          value: [
            `highest_alert=${packet.summary.highest_alert_level_24h ?? 'none'}`,
            `brake_triggered=${packet.summary.brake_triggered_24h}`,
            `manual_revocations=${packet.summary.manual_revocations_24h}`,
          ].join('\n'),
          inline: true,
        },
        {
          name: 'ASN concentration',
          value: [
            `cloud_asn_share_24h=${packet.summary.cloud_asn_share_24h}%`,
            topAsnSummary,
          ].join('\n'),
        },
        {
          name: 'Translation usage',
          value: [
            `active_24h=${translationUsage.active_users_24h}`,
            `active_7d=${translationUsage.active_users_7d}`,
            `active_30d=${translationUsage.active_users_30d}`,
            `dau_mau_stickiness=${formatNullablePct(translationUsage.dau_mau_stickiness_pct)}`,
            `first_active_24h=${translationUsage.first_active_users_24h}`,
            `returning_active_24h=${translationUsage.returning_active_users_24h}`,
            retentionSummary,
          ].join('\n'),
        },
      ],
    },
    fetchImpl,
  );
}

function formatRetentionSignal(
  signal: TelemetryUsageDailyMetrics['retention']['d1'],
): string {
  return `${signal.retained_users}/${signal.eligible_users} (${formatNullablePct(signal.retention_pct)}) cohort=${signal.cohort_date_utc}`;
}

function formatNullablePct(value: number | null): string {
  return value === null ? 'n/a' : `${value}%`;
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

function resolveDailyHeartbeatColor(packet: DailyReportPayload): number {
  if (packet.summary.brake_triggered_24h) {
    return DAILY_HEARTBEAT_COLOR_BRAKED;
  }

  if (packet.summary.issue_success_24h > 0) {
    return DAILY_HEARTBEAT_COLOR_ACTIVE;
  }

  return DAILY_HEARTBEAT_COLOR_OK;
}
