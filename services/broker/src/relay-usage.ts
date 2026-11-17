export type RelayDailyUsage = {
  chatbot: {
    classification_calls: number;
    response_calls: number;
    missing_usage_calls: number;
    cost_usd: number;
  };
  photo: {
    requests: number;
    unsettled_requests: number;
    missing_usage_requests: number;
    estimated_date_requests: number;
    cost_usd: number;
  };
  total_cost_usd: number;
};

export async function getRelayDailyUsage(db: D1Database, reportDateUtc: string): Promise<RelayDailyUsage> {
  const start = Date.parse(`${reportDateUtc}T00:00:00.000Z`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(reportDateUtc) || !Number.isFinite(start)
    || new Date(start).toISOString().slice(0, 10) !== reportDateUtc) {
    throw new Error('invalid_report_date_utc');
  }
  const result = await db.prepare(`
    SELECT chatbot.classification_calls, chatbot.response_calls, chatbot.missing_usage_calls,
      chatbot.chatbot_cost_usd, photo.requests, photo.unsettled_requests,
      photo.missing_usage_requests, photo.estimated_date_requests, photo.photo_cost_usd
    FROM (
      SELECT COALESCE(SUM(decisions_calls), 0) AS classification_calls,
        COALESCE(SUM(responses_calls), 0) AS response_calls,
        COALESCE(SUM(missing_usage_count), 0) AS missing_usage_calls,
        COALESCE(SUM(cost_usd), 0) AS chatbot_cost_usd
      FROM usage_daily WHERE utc_date = ?
    ) AS chatbot
    CROSS JOIN (
      SELECT COUNT(*) AS requests,
        COALESCE(SUM(CASE WHEN settled = 0 THEN 1 ELSE 0 END), 0) AS unsettled_requests,
        COALESCE(SUM(CASE WHEN settled = 1 AND missing_usage != 0 THEN 1 ELSE 0 END), 0) AS missing_usage_requests,
        COALESCE(SUM(CASE WHEN requested_at IS NULL THEN 1 ELSE 0 END), 0) AS estimated_date_requests,
        COALESCE(SUM(CASE WHEN settled = 1 THEN cost_usd ELSE 0 END), 0) AS photo_cost_usd
      FROM photo_admissions
      WHERE COALESCE(requested_at, period_start) >= ? AND COALESCE(requested_at, period_start) < ?
    ) AS photo
  `).bind(reportDateUtc, start, start + 86_400_000).all<Record<string, number>>();
  if (!result || result.success !== true || !Array.isArray(result.results) || result.results.length !== 1) {
    throw new Error('relay_usage_unavailable');
  }
  const row = result.results[0];
  if (!row || typeof row !== 'object' || Array.isArray(row)) throw new Error('relay_usage_unavailable');
  for (const key of ['classification_calls', 'response_calls', 'missing_usage_calls', 'requests',
    'unsettled_requests', 'missing_usage_requests', 'estimated_date_requests']) {
    if (!Number.isSafeInteger(row[key]) || row[key] < 0) throw new Error('relay_usage_unavailable');
  }
  for (const key of ['chatbot_cost_usd', 'photo_cost_usd']) {
    if (typeof row[key] !== 'number' || !Number.isFinite(row[key]) || row[key] < 0) {
      throw new Error('relay_usage_unavailable');
    }
  }
  const total = row.chatbot_cost_usd + row.photo_cost_usd;
  if (!Number.isFinite(total)) throw new Error('relay_usage_unavailable');
  return {
    chatbot: {
      classification_calls: row.classification_calls,
      response_calls: row.response_calls,
      missing_usage_calls: row.missing_usage_calls,
      cost_usd: row.chatbot_cost_usd,
    },
    photo: {
      requests: row.requests,
      unsettled_requests: row.unsettled_requests,
      missing_usage_requests: row.missing_usage_requests,
      estimated_date_requests: row.estimated_date_requests,
      cost_usd: row.photo_cost_usd,
    },
    total_cost_usd: total,
  };
}
