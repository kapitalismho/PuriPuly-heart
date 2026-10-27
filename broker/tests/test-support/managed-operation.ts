import { randomBytes } from 'node:crypto';

export function buildManagedOperationId(): string {
  return `ph-mop-v1_${randomBytes(24).toString('base64url')}`;
}

export function buildManagedOperationResumeToken(): string {
  return randomBytes(32).toString('base64url');
}
