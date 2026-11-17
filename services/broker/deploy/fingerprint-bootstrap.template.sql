-- Generated CI bootstrap SQL must only replace the migration placeholder value.
-- Post-bootstrap validation should run as a separate query in the caller.

UPDATE broker_config
SET value = json_set(value, '$.current.salt', '__BOOTSTRAP_REQUIRED__'),
    updated_at = CURRENT_TIMESTAMP
WHERE key = 'fingerprint_salt'
  AND json_extract(value, '$.current.salt') = '__BOOTSTRAP' || '_REQUIRED__';
