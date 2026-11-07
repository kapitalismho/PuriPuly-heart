export function extractContext(upstream: unknown, input: Record<string, unknown>[]): Record<string, unknown>;

export function exportRelayContext(options: {
  emitted: string;
  relayBuild?: string;
  migrations?: string;
  compatibilityDate: string;
  compatibilityFlags?: string[];
}): Promise<{
  format: 'puri-agent-relay-context-1';
  relay_build: string;
  bundle_hashes: Record<string, string>;
  contexts: Record<string, Record<string, unknown>>;
  export_provenance: {
    mode: 'controlled_no_network';
    diagnostic_only: true;
    external_model_calls: 0;
    source: 'observed_emitted_worker_outbound_requests';
    entry_module: string;
    compatibility_date: string;
    compatibility_flags: string[];
    migrations: Record<string, string>;
    captures: Record<string, unknown>[];
    tool_choice: string;
  };
}>;
