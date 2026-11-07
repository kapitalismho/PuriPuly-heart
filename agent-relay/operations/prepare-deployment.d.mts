export const workerName: string;
export const confirmation: string;
export function deploymentOrigin(value: string | undefined): string;
export function guardDeployment(env: Record<string, string | undefined>, head: string): void;
export function prepareDeployment(options: {
  env: Record<string, string | undefined>;
  directory: string;
  fetcher?: typeof fetch;
}): Promise<void>;
export function verifyDeploymentSchema(options: {
  payload: unknown;
  initializedOnly?: boolean;
}): Promise<void>;
