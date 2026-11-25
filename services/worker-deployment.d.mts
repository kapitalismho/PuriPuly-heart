type Deployment = {
  workerName: string;
  legacyVariable: string;
  origin: string;
  legacyOrigin: string;
  route: { pattern: string; zone_name: string; custom_domain: true };
};

export const brokerDeployment: Deployment;
export const relayDeployment: Deployment;
export function isExactOrigin(value: string | undefined, origin: string): boolean;
export function requireLegacyOrigin(env: Record<string, string | undefined>, deployment: Deployment): string;
export function requireAccountId(value: string | undefined): string;
export function verifyWorkerTarget(env: Record<string, string | undefined>, deployment: Deployment, fetcher?: typeof fetch): Promise<void>;
