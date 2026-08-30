import { execFile } from 'node:child_process';
import { randomBytes, createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const BROKER_DATABASE_NAME = 'puripuly-heart-broker';
const DEFAULT_MAX_LEGACY = 20;
const REFERRAL_ID_LENGTH = 6;
const REFERRAL_ID_ALPHABET = '23456789ABCDEFGHJKMNPQRSTUVWXYZ';
const REFERRAL_RANDOM_REJECTION_THRESHOLD =
  Math.floor(256 / REFERRAL_ID_ALPHABET.length) * REFERRAL_ID_ALPHABET.length;
const REFERRAL_ID_MAX_RANDOM_DRAWS = 64;
const REFERRAL_ID_MAX_COLLISION_ATTEMPTS = 4096;
const REFERRAL_ID_PATTERN = new RegExp(
  `^[${REFERRAL_ID_ALPHABET}]{${REFERRAL_ID_LENGTH}}$`,
  'u',
);
const SYNTHETIC_DISCORD_USER_REF_PREFIX = 'ph-discord-user-v1_legacy_';

const EXISTING_REFERRAL_IDS_QUERY = `SELECT referral_id
  FROM referral_codes
 ORDER BY referral_id`;

const REAL_REF_CANDIDATE_QUERY = `SELECT e.installation_id,
       e.discord_user_ref
  FROM openrouter_entitlements e
  JOIN discord_identities identity
    ON identity.discord_user_ref = e.discord_user_ref
   AND identity.status = 'active'
   AND identity.entitlement_installation_id = e.installation_id
  LEFT JOIN referral_codes rc
    ON rc.owner_source = 'discord'
   AND rc.owner_subject_ref = e.discord_user_ref
 WHERE e.status = 'active'
   AND e.managed_credential_ref IS NOT NULL
   AND length(trim(e.managed_credential_ref)) > 0
   AND e.discord_user_ref IS NOT NULL
   AND length(trim(e.discord_user_ref)) > 0
   AND e.expires_at IS NOT NULL
   AND length(trim(e.expires_at)) > 0
   AND datetime(e.expires_at) >= datetime('now')
   AND e.discord_issue_status = 'active'
   AND e.discord_issue_delivered_at IS NOT NULL
   AND length(trim(e.discord_issue_delivered_at)) > 0
   AND rc.referral_id IS NULL
 ORDER BY e.installation_id`;

const SYNTHETIC_LEGACY_CANDIDATE_QUERY = `SELECT e.installation_id,
       e.issued_at,
       e.expires_at,
       e.verified_hardware_hash,
       e.verified_hardware_hash_salt_version
  FROM openrouter_entitlements e
  LEFT JOIN referral_codes rc
    ON rc.owner_installation_id = e.installation_id
  LEFT JOIN discord_identities existing_active_identity
    ON existing_active_identity.entitlement_installation_id = e.installation_id
   AND existing_active_identity.status = 'active'
 WHERE e.status = 'active'
   AND e.managed_credential_ref IS NOT NULL
   AND length(trim(e.managed_credential_ref)) > 0
   AND (e.discord_user_ref IS NULL OR length(trim(e.discord_user_ref)) = 0)
   AND e.issued_at IS NOT NULL
   AND length(trim(e.issued_at)) > 0
   AND e.expires_at IS NOT NULL
   AND length(trim(e.expires_at)) > 0
   AND datetime(e.expires_at) >= datetime('now')
   AND e.verified_hardware_hash IS NOT NULL
   AND length(trim(e.verified_hardware_hash)) > 0
   AND e.verified_hardware_hash_salt_version IS NOT NULL
   AND rc.referral_id IS NULL
   AND existing_active_identity.discord_user_ref IS NULL
 ORDER BY e.installation_id`;

export const TALK_TOGETHER_PASS_BACKFILL_QUERIES = Object.freeze({
  existingReferralIds: EXISTING_REFERRAL_IDS_QUERY,
  realRefCandidates: REAL_REF_CANDIDATE_QUERY,
  syntheticLegacyCandidates: SYNTHETIC_LEGACY_CANDIDATE_QUERY,
});

const REFERRAL_CODE_INSERT_GUARDRAILS = [
  'INSERT INTO referral_codes',
  'SELECT',
  'JOIN discord_identities identity',
  "identity.status = 'active'",
  'identity.entitlement_installation_id = e.installation_id',
  "e.status = 'active'",
  'e.managed_credential_ref IS NOT NULL',
  'length(trim(e.managed_credential_ref)) > 0',
  'e.expires_at IS NOT NULL',
  'length(trim(e.expires_at)) > 0',
  "datetime(e.expires_at) >= datetime('now')",
  "e.discord_issue_status = 'active'",
  'e.discord_issue_delivered_at IS NOT NULL',
  'length(trim(e.discord_issue_delivered_at)) > 0',
  'NOT EXISTS',
  "existing.owner_source = 'discord'",
  'existing.owner_subject_ref = e.discord_user_ref',
  'existing.referral_id =',
];

const SYNTHETIC_IDENTITY_INSERT_GUARDRAILS = [
  'INSERT INTO discord_identities',
  'SELECT',
  'FROM openrouter_entitlements e',
  'LEFT JOIN referral_codes rc',
  'LEFT JOIN discord_identities existing_active_identity',
  'existing_active_identity.entitlement_installation_id = e.installation_id',
  "existing_active_identity.status = 'active'",
  'e.installation_id =',
  "e.status = 'active'",
  'e.managed_credential_ref IS NOT NULL',
  'length(trim(e.managed_credential_ref)) > 0',
  '(e.discord_user_ref IS NULL OR length(trim(e.discord_user_ref)) = 0)',
  'e.issued_at IS NOT NULL',
  'length(trim(e.issued_at)) > 0',
  'e.expires_at IS NOT NULL',
  'length(trim(e.expires_at)) > 0',
  "datetime(e.expires_at) >= datetime('now')",
  'e.verified_hardware_hash IS NOT NULL',
  'length(trim(e.verified_hardware_hash)) > 0',
  'e.verified_hardware_hash_salt_version IS NOT NULL',
  'rc.referral_id IS NULL',
  'existing_active_identity.discord_user_ref IS NULL',
  'NOT EXISTS',
  'existing_referral_id.referral_id =',
  'existing_identity.discord_user_ref =',
];

const SYNTHETIC_ENTITLEMENT_UPDATE_GUARDRAILS = [
  'UPDATE openrouter_entitlements',
  'discord_user_ref =',
  "status = 'active'",
  "discord_issue_status = 'active'",
  'discord_issue_reserved_at = COALESCE(discord_issue_reserved_at',
  'discord_issue_delivered_at = COALESCE(discord_issue_delivered_at',
  'installation_id =',
  'managed_credential_ref IS NOT NULL',
  'length(trim(managed_credential_ref)) > 0',
  '(discord_user_ref IS NULL OR length(trim(discord_user_ref)) = 0)',
  'issued_at IS NOT NULL',
  'length(trim(issued_at)) > 0',
  'expires_at IS NOT NULL',
  'length(trim(expires_at)) > 0',
  "datetime(expires_at) >= datetime('now')",
  'verified_hardware_hash IS NOT NULL',
  'length(trim(verified_hardware_hash)) > 0',
  'verified_hardware_hash_salt_version IS NOT NULL',
  'EXISTS',
  'identity.discord_user_ref =',
  "identity.status = 'active'",
  'identity.entitlement_installation_id = openrouter_entitlements.installation_id',
  'NOT EXISTS',
  'existing_active_identity.entitlement_installation_id = openrouter_entitlements.installation_id',
  "existing_active_identity.status = 'active'",
  'existing_active_identity.discord_user_ref <>',
  'existing_referral_id.referral_id =',
  'existing_code.owner_subject_ref =',
];

export function allocateReferralId(existingCodes, randomBytesFn = randomBytes) {
  if (!(existingCodes instanceof Set)) {
    throw new TypeError('existingCodes must be a Set');
  }

  for (let attempt = 0; attempt < REFERRAL_ID_MAX_COLLISION_ATTEMPTS; attempt += 1) {
    const referralId = generateReferralIdCandidate(randomBytesFn);
    if (existingCodes.has(referralId)) {
      continue;
    }

    existingCodes.add(referralId);
    return referralId;
  }

  throw new Error('unable to allocate a collision-free Referral ID');
}

export function syntheticDiscordUserRef(installationId) {
  const normalizedInstallationId = normalizeRequiredString(
    installationId,
    'installation_id',
  );
  const suffix = createHash('sha256')
    .update(normalizedInstallationId, 'utf8')
    .digest('base64url');

  return `${SYNTHETIC_DISCORD_USER_REF_PREFIX}${suffix}`;
}

export function buildBackfillSql(input) {
  const nowIso = normalizeRequiredString(input?.nowIso, 'nowIso');
  const realRefRows = normalizeRows(input?.realRefRows, 'realRefRows');
  const legacyRows = normalizeRows(input?.legacyRows, 'legacyRows');
  const maxLegacy = normalizeMaxLegacy(input?.maxLegacy);

  if (legacyRows.length > maxLegacy) {
    throw new Error(
      `legacy candidate count ${legacyRows.length} exceeds --max-legacy ${maxLegacy}`,
    );
  }

  const existingReferralIds = new Set(
    normalizeRows(input?.existingReferralIds ?? [], 'existingReferralIds').map((value) =>
      normalizeReferralId(value),
    ),
  );
  const randomBytesFn = input?.randomBytes ?? randomBytes;
  const statements = [];

  for (const row of realRefRows) {
    const referralId = allocateReferralId(existingReferralIds, randomBytesFn);
    statements.push(
      buildRealRefReferralCodeInsert({
        referralId,
        nowIso,
        installationId: normalizeRequiredString(row.installation_id, 'installation_id'),
        discordUserRef: normalizeRequiredString(row.discord_user_ref, 'discord_user_ref'),
      }),
    );
  }

  for (const row of legacyRows) {
    const installationId = normalizeRequiredString(row.installation_id, 'installation_id');
    const syntheticRef = syntheticDiscordUserRef(installationId);
    const referralId = allocateReferralId(existingReferralIds, randomBytesFn);

    statements.push(
      buildSyntheticIdentityInsert({ referralId, nowIso, installationId, syntheticRef }),
      buildSyntheticEntitlementUpdate({ referralId, nowIso, installationId, syntheticRef }),
      buildSyntheticReferralCodeInsert({
        referralId,
        nowIso,
        installationId,
        syntheticRef,
      }),
    );
  }

  if (statements.length === 0) {
    statements.push("SELECT 'talk_together_pass_backfill_noop' AS status");
  }

  const sql = `${statements.join(';\n\n')};\n`;
  validateGeneratedSql(sql);

  return {
    sql,
    summary: {
      realRefRows: realRefRows.length,
      legacyRows: legacyRows.length,
      generatedReferralCodes: realRefRows.length + legacyRows.length,
      syntheticIdentityRows: legacyRows.length,
      existingReferralIds: existingReferralIds.size,
    },
  };
}

export function validateGeneratedSql(sql) {
  if (typeof sql !== 'string') {
    throw new TypeError('sql must be a string');
  }

  const searchableSql = stripSqlComments(sql);
  const forbiddenDmlPatterns = [
    sqliteInsertIntoTablePattern('referral_rewards'),
    sqliteUpdateTablePattern('referral_rewards'),
    sqliteDeleteFromTablePattern('referral_rewards'),
    sqliteInsertIntoTablePattern('discord_oauth_sessions'),
    sqliteUpdateTablePattern('discord_oauth_sessions'),
    sqliteDeleteFromTablePattern('discord_oauth_sessions'),
  ];
  const referralCodeInsertPattern = sqliteInsertIntoTablePattern('referral_codes');
  const syntheticIdentityInsertPattern = sqliteInsertIntoTablePattern('discord_identities');
  const syntheticEntitlementUpdatePattern = sqliteUpdateTablePattern('openrouter_entitlements');
  const statements = splitSqlStatements(searchableSql).map((statement) => ({
    statement,
    maskedStatement: maskSqlStringLiterals(statement),
  }));

  if (statements.length === 0) {
    throw new Error('unexpected SQL statement outside backfill whitelist');
  }

  for (const { maskedStatement } of statements) {
    if (/^(?:BEGIN|COMMIT|ROLLBACK|END|SAVEPOINT|RELEASE)\b/iu.test(maskedStatement)) {
      throw new Error('forbidden transaction-control statement in generated SQL');
    }

    for (const pattern of forbiddenDmlPatterns) {
      if (pattern.test(maskedStatement)) {
        throw new Error('forbidden DML detected in generated SQL');
      }
    }
  }

  for (const { statement, maskedStatement } of statements) {
    if (isNoopStatement(statement)) {
      continue;
    }

    if (referralCodeInsertPattern.test(maskedStatement)) {
      validateReferralCodeInsertStatement(statement, maskedStatement);
      continue;
    }

    if (syntheticIdentityInsertPattern.test(maskedStatement)) {
      validateSyntheticIdentityInsertStatement(statement, maskedStatement);
      continue;
    }

    if (syntheticEntitlementUpdatePattern.test(maskedStatement)) {
      validateSyntheticEntitlementUpdateStatement(statement, maskedStatement);
      continue;
    }

    throw new Error('unexpected SQL statement outside backfill whitelist');
  }
}

function isNoopStatement(statement) {
  return statement === "SELECT 'talk_together_pass_backfill_noop' AS status";
}

function validateReferralCodeInsertStatement(statement, maskedStatement) {
  if (hasUnsafeOrTautology(maskedStatement)) {
    throw new Error('generated SQL guardrail failed: unsafe OR tautology detected');
  }

  const insertPrefix = sqliteInsertIntoTablePrefixPattern('referral_codes');

  if (
    new RegExp(`${insertPrefix}\\s*\\([\\s\\S]*?\\)\\s*values\\b`, 'iu').test(maskedStatement)
  ) {
    throw new Error('generated SQL guardrail failed: referral_codes inserts must use SELECT');
  }

  if (
    !new RegExp(`${insertPrefix}\\s*\\([\\s\\S]*?\\)\\s*select\\s+'`, 'iu').test(maskedStatement)
  ) {
    throw new Error('generated SQL guardrail failed: referral_codes inserts must use SELECT');
  }

  assertStatementGuardrails(statement, REFERRAL_CODE_INSERT_GUARDRAILS);

  const referralIdMatch = statement.match(
    /SELECT\s+'((?:[^']|'')*)'\s*,\s*'discord'\s*,\s*e\.discord_user_ref\s*,\s*e\.installation_id\s*,\s*'active'/iu,
  );

  if (!referralIdMatch) {
    throw new Error('generated SQL guardrail failed: referral_id literal not found');
  }

  const referralId = referralIdMatch[1].replace(/''/gu, "'");
  if (!REFERRAL_ID_PATTERN.test(referralId)) {
    throw new Error('generated SQL guardrail failed: invalid referral_id literal');
  }
}

function validateSyntheticIdentityInsertStatement(statement, maskedStatement) {
  if (hasUnsafeOrTautology(maskedStatement)) {
    throw new Error('generated SQL guardrail failed: unsafe OR tautology detected');
  }

  const insertPrefix = sqliteInsertIntoTablePrefixPattern('discord_identities');

  if (
    new RegExp(`${insertPrefix}\\s*\\([\\s\\S]*?\\)\\s*values\\b`, 'iu').test(maskedStatement)
  ) {
    throw new Error('generated SQL guardrail failed: discord_identities inserts must use SELECT');
  }

  if (
    !new RegExp(`${insertPrefix}\\s*\\([\\s\\S]*?\\)\\s*select\\s+'`, 'iu').test(
      maskedStatement,
    )
  ) {
    throw new Error('generated SQL guardrail failed: discord_identities inserts must use SELECT');
  }

  assertStatementGuardrails(statement, SYNTHETIC_IDENTITY_INSERT_GUARDRAILS);
}

function validateSyntheticEntitlementUpdateStatement(statement, maskedStatement) {
  if (hasUnsafeOrTautology(maskedStatement)) {
    throw new Error('generated SQL guardrail failed: unsafe OR tautology detected');
  }

  assertStatementGuardrails(statement, SYNTHETIC_ENTITLEMENT_UPDATE_GUARDRAILS);
}

function assertStatementGuardrails(statement, guardrails) {
  const guardrailSearchText = buildSqlGuardrailSearchText(statement);
  for (const guardrail of guardrails) {
    if (!guardrailSearchText.includes(buildSqlGuardrailSearchText(guardrail))) {
      throw new Error(`generated SQL guardrail missing: ${guardrail}`);
    }
  }
}

function hasUnsafeOrTautology(maskedSql) {
  return [
    /\bor\s*(?:\(\s*)?true\s*(?:\))?(?=\s|$)/iu,
    /\bor\s*(?:\(\s*)?(\d+)\s*=\s*\1\s*(?:\))?/iu,
    /\bor\s*(?:\(\s*)?1\s*(?:\))?(?=\s|$)/iu,
  ].some((pattern) => pattern.test(maskedSql));
}

function sqliteInsertIntoTablePattern(tableName) {
  return new RegExp(`${sqliteInsertIntoTablePrefixPattern(tableName)}(?=\\s|\\(|$)`, 'iu');
}

function sqliteInsertIntoTablePrefixPattern(tableName) {
  return `\\b(?:insert\\s+(?:or\\s+(?:rollback|abort|replace|fail|ignore)\\s+)?into|replace\\s+into)\\s+${sqliteQualifiedTablePattern(tableName)}`;
}

function sqliteUpdateTablePattern(tableName) {
  return new RegExp(
    `\\bupdate\\s+(?:or\\s+(?:rollback|abort|replace|fail|ignore)\\s+)?${sqliteQualifiedTablePattern(tableName)}(?=\\s|$)`,
    'iu',
  );
}

function sqliteDeleteFromTablePattern(tableName) {
  return new RegExp(
    `\\bdelete\\s+from\\s+${sqliteQualifiedTablePattern(tableName)}(?=\\s|$)`,
    'iu',
  );
}

function sqliteQualifiedTablePattern(tableName) {
  return `(?:${SQLITE_ANY_IDENTIFIER_PATTERN}\\s*\\.\\s*)?${sqliteExactIdentifierPattern(tableName)}`;
}

function sqliteExactIdentifierPattern(identifier) {
  const escapedIdentifier = escapeRegex(identifier);
  return `(?:${escapedIdentifier}|"${escapedIdentifier}"|\`${escapedIdentifier}\`|\\[${escapedIdentifier}\\])`;
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
}

const SQLITE_ANY_IDENTIFIER_PATTERN =
  '(?:[A-Za-z_][A-Za-z0-9_]*|"(?:""|[^"])+"|`(?:``|[^`])+`|\\[[^\\]]+\\])';

if (isDirectExecution()) {
  await main();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.validate) {
    const sqlPath = resolve(args.validate);
    const sql = await readFile(sqlPath, 'utf8');
    validateGeneratedSql(sql);
    process.stdout.write(`validated ${sqlPath}\n`);
    return;
  }

  const configPath = resolve(requiredArg(args, 'config'));
  const outputPath = resolve(requiredArg(args, 'out'));
  const maxLegacy = normalizeMaxLegacy(args['max-legacy'] ?? DEFAULT_MAX_LEGACY);
  const nowIso = new Date().toISOString();

  const [existingReferralRows, realRefRows, legacyRows] = await Promise.all([
    queryRemoteD1(configPath, EXISTING_REFERRAL_IDS_QUERY),
    queryRemoteD1(configPath, REAL_REF_CANDIDATE_QUERY),
    queryRemoteD1(configPath, SYNTHETIC_LEGACY_CANDIDATE_QUERY),
  ]);

  const { sql, summary } = buildBackfillSql({
    nowIso,
    existingReferralIds: existingReferralRows.map((row) => row.referral_id),
    realRefRows,
    legacyRows,
    maxLegacy,
  });

  await writeFile(outputPath, sql, 'utf8');

  process.stdout.write(
    `${JSON.stringify(
      {
        realRefRows: summary.realRefRows,
        legacyRows: summary.legacyRows,
        generatedReferralCodes: summary.generatedReferralCodes,
        out: outputPath,
      },
      null,
      2,
    )}\n`,
  );
}

function buildRealRefReferralCodeInsert(input) {
  const referralId = sqlString(input.referralId);
  const nowIso = sqlString(input.nowIso);
  const installationId = sqlString(input.installationId);
  const discordUserRef = sqlString(input.discordUserRef);

  return `INSERT INTO referral_codes (
  referral_id, owner_source, owner_subject_ref, owner_installation_id, status, created_at, updated_at
)
SELECT ${referralId}, 'discord', e.discord_user_ref, e.installation_id, 'active', ${nowIso}, ${nowIso}
  FROM openrouter_entitlements e
  JOIN discord_identities identity
    ON identity.discord_user_ref = e.discord_user_ref
   AND identity.status = 'active'
   AND identity.entitlement_installation_id = e.installation_id
 WHERE e.installation_id = ${installationId}
   AND e.discord_user_ref = ${discordUserRef}
   AND e.status = 'active'
   AND e.managed_credential_ref IS NOT NULL
   AND length(trim(e.managed_credential_ref)) > 0
   AND e.discord_user_ref IS NOT NULL
   AND length(trim(e.discord_user_ref)) > 0
   AND e.expires_at IS NOT NULL
   AND length(trim(e.expires_at)) > 0
   AND datetime(e.expires_at) >= datetime('now')
   AND e.discord_issue_status = 'active'
   AND e.discord_issue_delivered_at IS NOT NULL
   AND length(trim(e.discord_issue_delivered_at)) > 0
   AND NOT EXISTS (
     SELECT 1 FROM referral_codes existing
      WHERE existing.owner_source = 'discord'
        AND (
          existing.owner_subject_ref = e.discord_user_ref
          OR existing.referral_id = ${referralId}
        )
   )`;
}

function buildSyntheticIdentityInsert(input) {
  const referralId = sqlString(input.referralId);
  const nowIso = sqlString(input.nowIso);
  const installationId = sqlString(input.installationId);
  const syntheticRef = sqlString(input.syntheticRef);

  return `INSERT INTO discord_identities (
  discord_user_ref, entitlement_installation_id, status, ref_secret_version, created_at, updated_at
)
SELECT ${syntheticRef}, e.installation_id, 'active', 1, ${nowIso}, ${nowIso}
  FROM openrouter_entitlements e
  LEFT JOIN referral_codes rc
    ON rc.owner_installation_id = e.installation_id
  LEFT JOIN discord_identities existing_active_identity
    ON existing_active_identity.entitlement_installation_id = e.installation_id
   AND existing_active_identity.status = 'active'
 WHERE e.installation_id = ${installationId}
   AND e.status = 'active'
   AND e.managed_credential_ref IS NOT NULL
   AND length(trim(e.managed_credential_ref)) > 0
   AND (e.discord_user_ref IS NULL OR length(trim(e.discord_user_ref)) = 0)
   AND e.issued_at IS NOT NULL
   AND length(trim(e.issued_at)) > 0
   AND e.expires_at IS NOT NULL
   AND length(trim(e.expires_at)) > 0
   AND datetime(e.expires_at) >= datetime('now')
   AND e.verified_hardware_hash IS NOT NULL
   AND length(trim(e.verified_hardware_hash)) > 0
   AND e.verified_hardware_hash_salt_version IS NOT NULL
   AND rc.referral_id IS NULL
   AND existing_active_identity.discord_user_ref IS NULL
   AND NOT EXISTS (
     SELECT 1 FROM referral_codes existing_referral_id
      WHERE existing_referral_id.referral_id = ${referralId}
   )
   AND NOT EXISTS (
     SELECT 1 FROM discord_identities existing_identity
      WHERE existing_identity.discord_user_ref = ${syntheticRef}
   )`;
}

function buildSyntheticEntitlementUpdate(input) {
  const referralId = sqlString(input.referralId);
  const nowIso = sqlString(input.nowIso);
  const installationId = sqlString(input.installationId);
  const syntheticRef = sqlString(input.syntheticRef);

  return `UPDATE openrouter_entitlements
   SET discord_user_ref = ${syntheticRef},
       discord_issue_status = 'active',
       discord_issue_reserved_at = COALESCE(discord_issue_reserved_at, ${nowIso}),
       discord_issue_delivered_at = COALESCE(discord_issue_delivered_at, ${nowIso})
 WHERE installation_id = ${installationId}
   AND status = 'active'
   AND managed_credential_ref IS NOT NULL
   AND length(trim(managed_credential_ref)) > 0
   AND (discord_user_ref IS NULL OR length(trim(discord_user_ref)) = 0)
   AND issued_at IS NOT NULL
   AND length(trim(issued_at)) > 0
   AND expires_at IS NOT NULL
   AND length(trim(expires_at)) > 0
   AND datetime(expires_at) >= datetime('now')
   AND verified_hardware_hash IS NOT NULL
   AND length(trim(verified_hardware_hash)) > 0
   AND verified_hardware_hash_salt_version IS NOT NULL
   AND EXISTS (
     SELECT 1 FROM discord_identities identity
      WHERE identity.discord_user_ref = ${syntheticRef}
        AND identity.status = 'active'
        AND identity.entitlement_installation_id = openrouter_entitlements.installation_id
   )
   AND NOT EXISTS (
     SELECT 1 FROM discord_identities existing_active_identity
      WHERE existing_active_identity.entitlement_installation_id = openrouter_entitlements.installation_id
        AND existing_active_identity.status = 'active'
        AND existing_active_identity.discord_user_ref <> ${syntheticRef}
   )
   AND NOT EXISTS (
     SELECT 1 FROM referral_codes existing_referral_id
      WHERE existing_referral_id.referral_id = ${referralId}
   )
   AND NOT EXISTS (
     SELECT 1 FROM referral_codes existing_code
      WHERE existing_code.owner_subject_ref = ${syntheticRef}
   )`;
}

function buildSyntheticReferralCodeInsert(input) {
  const referralId = sqlString(input.referralId);
  const nowIso = sqlString(input.nowIso);
  const installationId = sqlString(input.installationId);
  const syntheticRef = sqlString(input.syntheticRef);

  return `INSERT INTO referral_codes (
  referral_id, owner_source, owner_subject_ref, owner_installation_id, status, created_at, updated_at
)
SELECT ${referralId}, 'discord', e.discord_user_ref, e.installation_id, 'active', ${nowIso}, ${nowIso}
  FROM openrouter_entitlements e
  JOIN discord_identities identity
    ON identity.discord_user_ref = e.discord_user_ref
   AND identity.status = 'active'
   AND identity.entitlement_installation_id = e.installation_id
 WHERE e.installation_id = ${installationId}
   AND e.discord_user_ref = ${syntheticRef}
   AND e.status = 'active'
   AND e.managed_credential_ref IS NOT NULL
   AND length(trim(e.managed_credential_ref)) > 0
   AND e.discord_user_ref IS NOT NULL
   AND length(trim(e.discord_user_ref)) > 0
   AND e.expires_at IS NOT NULL
   AND length(trim(e.expires_at)) > 0
   AND datetime(e.expires_at) >= datetime('now')
   AND e.discord_issue_status = 'active'
   AND e.discord_issue_delivered_at IS NOT NULL
   AND length(trim(e.discord_issue_delivered_at)) > 0
   AND NOT EXISTS (
     SELECT 1 FROM discord_identities existing_active_identity
      WHERE existing_active_identity.entitlement_installation_id = e.installation_id
        AND existing_active_identity.status = 'active'
        AND existing_active_identity.discord_user_ref <> e.discord_user_ref
   )
   AND NOT EXISTS (
     SELECT 1 FROM referral_codes existing
      WHERE existing.owner_source = 'discord'
        AND (
          existing.owner_subject_ref = e.discord_user_ref
          OR existing.referral_id = ${referralId}
        )
   )`;
}

function generateReferralIdCandidate(randomBytesFn) {
  let referralId = '';
  let drawCount = 0;

  while (referralId.length < REFERRAL_ID_LENGTH) {
    drawCount += 1;
    if (drawCount > REFERRAL_ID_MAX_RANDOM_DRAWS) {
      throw new Error('unable to generate Referral ID from random source');
    }

    const bytes = randomBytesFn(REFERRAL_ID_LENGTH - referralId.length);
    if (!bytes || bytes.length === 0) {
      throw new Error('Referral ID random source returned no bytes');
    }

    for (const byte of bytes) {
      if (byte >= REFERRAL_RANDOM_REJECTION_THRESHOLD) {
        continue;
      }

      referralId += REFERRAL_ID_ALPHABET[byte % REFERRAL_ID_ALPHABET.length];
      if (referralId.length === REFERRAL_ID_LENGTH) {
        break;
      }
    }
  }

  return referralId;
}

async function queryRemoteD1(configPath, query) {
  let stdout;
  try {
    ({ stdout } = await execFileAsync(
      'pnpm',
      [
        'exec',
        'wrangler',
        'd1',
        'execute',
        BROKER_DATABASE_NAME,
        '--remote',
        '--config',
        configPath,
        '--json',
        '--command',
        query,
      ],
      { maxBuffer: 10 * 1024 * 1024 },
    ));
  } catch {
    throw new Error('wrangler D1 query failed');
  }

  try {
    return extractD1Rows(JSON.parse(stdout));
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error('wrangler D1 query returned invalid JSON');
    }

    throw error;
  }
}

export function extractD1Rows(value) {
  if (Array.isArray(value)) {
    if (value.some(isFailedD1Wrapper)) {
      throw new Error('wrangler D1 query failed');
    }

    if (value.every((item) => isPlainObject(item) && !isD1Wrapper(item))) {
      return value;
    }

    return value.flatMap((item) => extractD1Rows(item));
  }

  if (isFailedD1Wrapper(value)) {
    throw new Error('wrangler D1 query failed');
  }

  if (isPlainObject(value) && Array.isArray(value.results)) {
    return value.results;
  }

  if (isPlainObject(value) && Array.isArray(value.result)) {
    return value.result.flatMap((item) => extractD1Rows(item));
  }

  throw new Error('wrangler D1 JSON output did not contain query result rows');
}

function normalizeRows(value, name) {
  if (!Array.isArray(value)) {
    throw new TypeError(`${name} must be an array`);
  }

  return value;
}

function normalizeRequiredString(value, name) {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`${name} must be a non-empty string`);
  }

  return value;
}

function normalizeReferralId(value) {
  const normalized = normalizeRequiredString(value, 'referral_id').trim().toUpperCase();
  if (!REFERRAL_ID_PATTERN.test(normalized)) {
    throw new Error('existing referral_id is not a valid Pass ID');
  }

  return normalized;
}

function normalizeMaxLegacy(value) {
  let normalized;

  if (typeof value === 'number') {
    normalized = value;
  } else if (typeof value === 'string') {
    const trimmedValue = value.trim();
    if (!/^\d+$/u.test(trimmedValue)) {
      throw new Error('--max-legacy must be a non-negative integer');
    }
    normalized = Number(trimmedValue);
  } else {
    throw new Error('--max-legacy must be a non-negative integer');
  }

  if (!Number.isInteger(normalized) || normalized < 0) {
    throw new Error('--max-legacy must be a non-negative integer');
  }

  return normalized;
}

function sqlString(value) {
  return `'${String(value).replace(/'/gu, "''")}'`;
}

function stripSqlComments(sql) {
  let output = '';
  let inStringLiteral = false;

  for (let index = 0; index < sql.length; index += 1) {
    const character = sql[index];
    const nextCharacter = sql[index + 1];

    if (inStringLiteral) {
      output += character;
      if (character === "'") {
        if (nextCharacter === "'") {
          output += nextCharacter;
          index += 1;
        } else {
          inStringLiteral = false;
        }
      }
      continue;
    }

    if (character === "'") {
      inStringLiteral = true;
      output += character;
      continue;
    }

    if (character === '-' && nextCharacter === '-') {
      output += ' ';
      index += 2;
      while (index < sql.length && sql[index] !== '\n' && sql[index] !== '\r') {
        index += 1;
      }
      index -= 1;
      continue;
    }

    if (character === '/' && nextCharacter === '*') {
      output += ' ';
      index += 2;
      while (
        index < sql.length &&
        !(sql[index] === '*' && sql[index + 1] === '/')
      ) {
        index += 1;
      }
      if (index < sql.length) {
        index += 1;
      }
      continue;
    }

    output += character;
  }

  return output;
}

function maskSqlStringLiterals(sql) {
  let output = '';
  let inStringLiteral = false;

  for (let index = 0; index < sql.length; index += 1) {
    const character = sql[index];
    const nextCharacter = sql[index + 1];

    if (inStringLiteral) {
      if (character === "'") {
        if (nextCharacter === "'") {
          output += '  ';
          index += 1;
        } else {
          output += character;
          inStringLiteral = false;
        }
      } else {
        output += character === '\n' || character === '\r' ? character : ' ';
      }
      continue;
    }

    output += character;
    if (character === "'") {
      inStringLiteral = true;
    }
  }

  return output;
}

function buildSqlGuardrailSearchText(sql) {
  let output = '';
  let literalValue = '';
  let inStringLiteral = false;

  for (let index = 0; index < sql.length; index += 1) {
    const character = sql[index];
    const nextCharacter = sql[index + 1];

    if (inStringLiteral) {
      if (character === "'") {
        if (nextCharacter === "'") {
          literalValue += "'";
          index += 1;
        } else {
          output += encodedSqlLiteralMarker(literalValue);
          literalValue = '';
          inStringLiteral = false;
        }
      } else {
        literalValue += character;
      }
      continue;
    }

    if (character === "'") {
      inStringLiteral = true;
      literalValue = '';
      continue;
    }

    output += character;
  }

  if (inStringLiteral) {
    output += encodedSqlLiteralMarker(literalValue);
  }

  return output;
}

function encodedSqlLiteralMarker(value) {
  return `\uE000${Buffer.from(value, 'utf8').toString('hex')}\uE001`;
}

function splitSqlStatements(sql) {
  const statements = [];
  let current = '';
  let inString = false;

  for (let index = 0; index < sql.length; index += 1) {
    const character = sql[index];

    if (character === "'") {
      current += character;
      if (inString && sql[index + 1] === "'") {
        current += sql[index + 1];
        index += 1;
      } else {
        inString = !inString;
      }
      continue;
    }

    if (character === ';' && !inString) {
      const statement = current.trim();
      if (statement) {
        statements.push(statement);
      }
      current = '';
      continue;
    }

    current += character;
  }

  const finalStatement = current.trim();
  if (finalStatement) {
    statements.push(finalStatement);
  }

  return statements;
}

function parseArgs(argv) {
  const args = {};

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (!token?.startsWith('--')) {
      throw new Error(`unexpected argument: ${token ?? '<missing>'}`);
    }

    const key = token.slice(2);
    const value = argv[index + 1];

    if (!value || value.startsWith('--')) {
      throw new Error(`missing value for --${key}`);
    }

    args[key] = value;
    index += 1;
  }

  return args;
}

function requiredArg(args, key) {
  const value = args[key];

  if (!value) {
    throw new Error(`missing required --${key} argument`);
  }

  return value;
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isD1Wrapper(value) {
  return (
    isPlainObject(value) &&
    ('success' in value ||
      'results' in value ||
      'result' in value ||
      'error' in value ||
      'errors' in value)
  );
}

function isFailedD1Wrapper(value) {
  return (
    isPlainObject(value) &&
    (value.success === false ||
      'error' in value ||
      (Array.isArray(value.errors) && value.errors.length > 0))
  );
}

function isDirectExecution() {
  return Boolean(process.argv[1]) && import.meta.url === pathToFileURL(resolve(process.argv[1])).href;
}
