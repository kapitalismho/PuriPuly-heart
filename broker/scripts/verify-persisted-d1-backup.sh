#!/usr/bin/env bash
set -euo pipefail

expected_path="${RUNNER_TEMP:?}/puripuly-heart-broker-pre-migration-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}.sql.age"
if [[ "${D1_BACKUP_UPLOAD_OUTCOME:-}" != success || ! "${D1_BACKUP_ARTIFACT_ID:-}" =~ ^[1-9][0-9]*$ || ! "${D1_BACKUP_ARTIFACT_DIGEST:-}" =~ ^[0-9a-f]{64}$ ]]; then
  echo 'Protected D1 backup artifact upload did not succeed with persistence evidence.' >&2
  exit 1
fi
if [[ "${D1_BACKUP_PATH:-}" != "$expected_path" || ! -f "$expected_path" || -L "$expected_path" || ! -s "$expected_path" || ! "${D1_BACKUP_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
  echo 'Protected D1 backup identity is missing or invalid.' >&2
  exit 1
fi
actual_sha256="$(sha256sum "$expected_path" | cut -d ' ' -f 1)"
if [[ "$actual_sha256" != "$D1_BACKUP_SHA256" ]]; then
  echo 'Protected D1 backup integrity check failed.' >&2
  exit 1
fi
