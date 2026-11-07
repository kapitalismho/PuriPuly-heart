export function configureSupportStorage(options: {
  mode: string;
  bucket: string;
  account: string;
  token: string;
  enabledConfig?: string;
  fetcher?: (url: string, options: RequestInit) => Promise<Response>;
}): Promise<{
  bucket: string;
  private: true;
  retention_days: 30;
  prefix: 'support/';
  enabled_config_created: boolean;
}>;
