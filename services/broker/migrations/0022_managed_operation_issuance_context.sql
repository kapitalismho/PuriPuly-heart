PRAGMA defer_foreign_keys = on;

ALTER TABLE managed_operations
  ADD COLUMN hardware_hash TEXT CHECK (
    hardware_hash IS NULL OR length(hardware_hash) BETWEEN 1 AND 128
  );
ALTER TABLE managed_operations
  ADD COLUMN hardware_hash_salt_version INTEGER CHECK (
    hardware_hash_salt_version IS NULL OR hardware_hash_salt_version >= 0
  );
ALTER TABLE managed_operations
  ADD COLUMN app_version TEXT CHECK (
    app_version IS NULL OR (length(app_version) BETWEEN 1 AND 64)
  );

PRAGMA foreign_key_check;
