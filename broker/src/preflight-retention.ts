const CHALLENGE_PREFLIGHT_RETENTION_MS = 24 * 60 * 60 * 1000;

export async function deleteExpiredChallengePreflightInstallations(
  db: D1Database,
  input: {
    installationId: string;
    now: Date;
    devicePublicKey?: string;
  },
): Promise<void> {
  const cutoffIso = new Date(
    input.now.getTime() - CHALLENGE_PREFLIGHT_RETENTION_MS,
  ).toISOString();

  await db
    .prepare(
      `DELETE FROM installations
         WHERE (installation_id = ? OR (? IS NOT NULL AND device_public_key = ?))
           AND hardware_hash IS NULL
           AND hardware_hash_salt_version IS NULL
           AND challenge IS NOT NULL
           AND challenge_expires_at IS NOT NULL
           AND challenge_salt_version IS NOT NULL
           AND NOT EXISTS (
             SELECT 1
               FROM openrouter_entitlements
              WHERE openrouter_entitlements.installation_id = installations.installation_id
           )
           AND max(julianday(last_seen_at), julianday(challenge_expires_at)) < julianday(?)`,
    )
    .bind(
      input.installationId,
      input.devicePublicKey ?? null,
      input.devicePublicKey ?? null,
      cutoffIso,
    )
    .run();
}
