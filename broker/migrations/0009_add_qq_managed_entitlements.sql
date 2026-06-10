CREATE TABLE qq_managed_entitlements (
  qq_subject_ref TEXT PRIMARY KEY
    CHECK (length(qq_subject_ref) > length('ph-qq-subject-v1_'))
    CHECK (qq_subject_ref GLOB 'ph-qq-subject-v1_*'),
  status TEXT NOT NULL CHECK(status IN ('issuing', 'active', 'cleanup_required', 'revoked')),
  issue_ref TEXT NOT NULL CHECK (length(issue_ref) > 0),
  managed_credential_ref TEXT,
  budget_usd REAL NOT NULL CHECK (budget_usd >= 0),
  reserved_at TEXT NOT NULL CHECK (length(reserved_at) > 0),
  issued_at TEXT,
  expires_at TEXT,
  delivered_at TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK (
    status <> 'active'
    OR (
      managed_credential_ref IS NOT NULL
      AND issued_at IS NOT NULL
      AND expires_at IS NOT NULL
      AND delivered_at IS NOT NULL
    )
  ),
  CHECK (
    status <> 'cleanup_required'
    OR managed_credential_ref IS NOT NULL
  ),
  CHECK (
    status <> 'issuing'
    OR (
      issued_at IS NULL
      AND expires_at IS NULL
      AND delivered_at IS NULL
    )
  )
) STRICT;

CREATE UNIQUE INDEX idx_qq_managed_entitlements_issue_ref
  ON qq_managed_entitlements(issue_ref);
CREATE UNIQUE INDEX idx_qq_managed_entitlements_managed_credential_ref
  ON qq_managed_entitlements(managed_credential_ref)
  WHERE managed_credential_ref IS NOT NULL;
CREATE INDEX idx_qq_managed_entitlements_status_updated_at
  ON qq_managed_entitlements(status, updated_at);
CREATE INDEX idx_qq_managed_entitlements_expires_at
  ON qq_managed_entitlements(expires_at);
