ALTER TABLE referral_codes
  ADD COLUMN disabled_reason TEXT CHECK (
    disabled_reason IS NULL OR length(disabled_reason) BETWEEN 1 AND 64
  );

ALTER TABLE referral_codes
  ADD COLUMN disabled_by TEXT CHECK (
    disabled_by IS NULL OR length(disabled_by) BETWEEN 1 AND 64
  );

ALTER TABLE referral_codes
  ADD COLUMN disabled_at TEXT;

ALTER TABLE referral_rewards
  ADD COLUMN attempt_ip_hash TEXT CHECK (
    attempt_ip_hash IS NULL OR length(attempt_ip_hash) = 64
  );

CREATE INDEX idx_referral_rewards_attempt_installation_time
  ON referral_rewards(referred_installation_id, created_at);

CREATE INDEX idx_referral_rewards_attempt_ip_hash_time
  ON referral_rewards(attempt_ip_hash, created_at)
  WHERE attempt_ip_hash IS NOT NULL;

CREATE INDEX idx_referral_rewards_referral_velocity
  ON referral_rewards(referral_id, created_at);

CREATE INDEX idx_referral_rewards_referrer_velocity
  ON referral_rewards(referrer_discord_user_ref, created_at)
  WHERE referrer_discord_user_ref IS NOT NULL;

UPDATE broker_config
   SET value = json_insert(
         value,
         '$.retention.referralSkippedDays', 7,
         '$.retention.referralFailedDays', 30,
         '$.referralAttempts', json('{"validShaped":{"maxPerInstallation":8,"maxPerIp":30,"windowMinutes":15},"unknown":{"maxPerInstallation":3,"maxPerIp":10,"windowMinutes":15},"perReferralIdVelocity":{"maxAttempts":25,"windowMinutes":60},"perReferrerRewardVelocity":{"maxRewards":5,"windowMinutes":1440}}')
       ),
       updated_at = CURRENT_TIMESTAMP
 WHERE key = 'abuse_controls';
