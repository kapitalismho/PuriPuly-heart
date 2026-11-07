export function smokeDeployment(options: {
  origin: string | undefined;
  secret: string | undefined;
  modelEnabled: boolean;
  supportEnabled: boolean;
  allowLoopback?: boolean;
  fetcher?: typeof fetch;
}): Promise<{
  status: 'rejection_smoke_passed';
  probes: number;
  model_requests: 0;
  support_uploads: 0;
}>;
