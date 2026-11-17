CREATE TABLE qq_pass_settlement_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  referral_reward_id INTEGER NOT NULL UNIQUE,
  delivery_id TEXT NOT NULL UNIQUE,
  phase TEXT NOT NULL CHECK (
    phase IN ('invitee_pending', 'referrer_pending', 'completed')
  ),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  last_attempt_at TEXT,
  next_attempt_at TEXT NOT NULL,
  fencing_token TEXT CHECK (
    fencing_token IS NULL OR length(fencing_token) BETWEEN 1 AND 64
  ),
  lease_expires_at TEXT,
  last_error_code TEXT CHECK (
    last_error_code IS NULL OR length(last_error_code) BETWEEN 1 AND 64
  ),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT,
  CHECK (
    (fencing_token IS NULL AND lease_expires_at IS NULL)
    OR (fencing_token IS NOT NULL AND lease_expires_at IS NOT NULL)
  ),
  CHECK (
    (phase = 'completed' AND completed_at IS NOT NULL)
    OR (phase <> 'completed' AND completed_at IS NULL)
  )
) STRICT;

CREATE INDEX idx_qq_pass_settlement_jobs_due
  ON qq_pass_settlement_jobs(phase, next_attempt_at, lease_expires_at)
  WHERE phase IN ('invitee_pending', 'referrer_pending');

CREATE UNIQUE INDEX idx_qq_pass_settlement_jobs_fencing_token
  ON qq_pass_settlement_jobs(fencing_token)
  WHERE fencing_token IS NOT NULL;

INSERT OR IGNORE INTO qq_pass_settlement_jobs (
  referral_reward_id,
  delivery_id,
  phase,
  attempt_count,
  last_attempt_at,
  next_attempt_at,
  fencing_token,
  lease_expires_at,
  last_error_code,
  created_at,
  updated_at,
  completed_at
)
SELECT reward.id,
       delivery.delivery_id,
       CASE
         WHEN reward.referred_bonus_status = 'credited'
           THEN 'referrer_pending'
         ELSE 'invitee_pending'
       END,
       0,
       NULL,
       COALESCE(delivery.acknowledged_at, entitlement.delivered_at, reward.updated_at),
       NULL,
       NULL,
       NULL,
       COALESCE(delivery.acknowledged_at, entitlement.delivered_at, reward.created_at),
       COALESCE(delivery.acknowledged_at, entitlement.delivered_at, reward.updated_at),
       NULL
  FROM referral_rewards reward
  JOIN qq_managed_entitlements entitlement
    ON entitlement.qq_subject_ref = reward.referred_subject_ref
   AND entitlement.status = 'active'
   AND entitlement.managed_credential_ref IS NOT NULL
   AND length(trim(entitlement.managed_credential_ref)) > 0
   AND entitlement.delivered_at IS NOT NULL
  JOIN managed_key_deliveries delivery
    ON delivery.delivery_id = (
         SELECT matching_delivery.delivery_id
           FROM managed_key_deliveries matching_delivery
          WHERE matching_delivery.issue_source = 'qq'
            AND matching_delivery.subject_ref = reward.referred_subject_ref
            AND matching_delivery.installation_id IS reward.referred_installation_id
            AND matching_delivery.managed_credential_ref = entitlement.managed_credential_ref
            AND matching_delivery.status = 'acknowledged'
          ORDER BY matching_delivery.acknowledged_at DESC,
                   matching_delivery.created_at DESC,
                   matching_delivery.delivery_id DESC
          LIMIT 1
       )
 WHERE reward.referred_source = 'qq'
   AND reward.referrer_source IS NOT NULL
   AND reward.referrer_subject_ref IS NOT NULL
   AND (
     reward.referred_bonus_status = 'reserved'
     OR (
       reward.referred_bonus_status = 'credited'
       AND reward.referrer_bonus_status IN ('pending', 'applying', 'failed')
     )
   );
