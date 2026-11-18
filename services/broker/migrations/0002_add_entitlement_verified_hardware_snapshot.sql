-- Forward-only schema extension for managed child-key issuance metadata.
-- Existing databases may already have applied 0001, so the verified hardware
-- snapshot fields must land through ALTER TABLE rather than by mutating older
-- migration semantics in place.
ALTER TABLE openrouter_entitlements
  ADD COLUMN verified_hardware_hash TEXT;

ALTER TABLE openrouter_entitlements
  ADD COLUMN verified_hardware_hash_salt_version INTEGER CHECK (
    (verified_hardware_hash IS NULL AND verified_hardware_hash_salt_version IS NULL)
    OR (
      verified_hardware_hash IS NOT NULL
      AND verified_hardware_hash_salt_version IS NOT NULL
    )
  );
