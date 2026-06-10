import { describe, expect, it } from 'vitest';

import { deriveManagedOpenRouterUserId } from '../src/openrouter-user-id';

describe('managed OpenRouter user ID derivation', () => {
  it('returns null when installation or secret input is blank', async () => {
    await expect(
      deriveManagedOpenRouterUserId({
        installationId: '   ',
        secret: 'test-managed-user-hmac-secret',
      }),
    ).resolves.toBeNull();

    await expect(
      deriveManagedOpenRouterUserId({
        installationId: 'install-123',
        secret: '   ',
      }),
    ).resolves.toBeNull();
  });

  it('derives a deterministic versioned user ID from the installation and secret', async () => {
    await expect(
      deriveManagedOpenRouterUserId({
        installationId: 'install-123',
        secret: 'test-managed-user-hmac-secret',
      }),
    ).resolves.toBe(
      'ph-or-user-v1_zmPdMJAwX46j84JJDxHRglZWMLeAe8uqXVciPw3sNm8',
    );
  });

  it('normalizes leading and trailing whitespace before deriving the user ID', async () => {
    await expect(
      Promise.all([
        deriveManagedOpenRouterUserId({
          installationId: 'install-123',
          secret: 'test-managed-user-hmac-secret',
        }),
        deriveManagedOpenRouterUserId({
          installationId: '  install-123  ',
          secret: '  test-managed-user-hmac-secret  ',
        }),
      ]),
    ).resolves.toEqual([
      'ph-or-user-v1_zmPdMJAwX46j84JJDxHRglZWMLeAe8uqXVciPw3sNm8',
      'ph-or-user-v1_zmPdMJAwX46j84JJDxHRglZWMLeAe8uqXVciPw3sNm8',
    ]);
  });

  it('changes the derived user ID when the installation changes', async () => {
    await expect(
      Promise.all([
        deriveManagedOpenRouterUserId({
          installationId: 'install-123',
          secret: 'test-managed-user-hmac-secret',
        }),
        deriveManagedOpenRouterUserId({
          installationId: 'install-456',
          secret: 'test-managed-user-hmac-secret',
        }),
      ]),
    ).resolves.toEqual([
      'ph-or-user-v1_zmPdMJAwX46j84JJDxHRglZWMLeAe8uqXVciPw3sNm8',
      'ph-or-user-v1_go9aombjO-0-BaEdvbVm07vdsObUvd9hYu_Vr-hCqu8',
    ]);
  });

  it('derives a source-aware QQ user ID from qq_subject_ref without treating it as an installation', async () => {
    const qqSubjectRef = 'ph-qq-subject-v1_synthetic-subject-ref';
    const secret = 'test-managed-user-hmac-secret';

    const [qqUserId, discordStyleUserId] = await Promise.all([
      deriveManagedOpenRouterUserId({
        issueSource: 'qq',
        subjectRef: qqSubjectRef,
        secret,
      }),
      deriveManagedOpenRouterUserId({
        installationId: qqSubjectRef,
        secret,
      }),
    ]);

    expect(qqUserId).toMatch(/^ph-or-user-v1_[A-Za-z0-9_-]+$/u);
    expect(qqUserId).not.toBe(discordStyleUserId);
  });
});
