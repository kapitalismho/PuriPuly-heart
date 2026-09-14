# Changelog

## Unreleased

- Added LISTEN-only Soniox speaker-aware segmentation with connection-scoped speaker identities, independent language and speaker boundaries, whole-parent LLM translation batching, bounded batch-aware output tokens across supported providers, and provider-neutral one-second Peer caption replacement pacing. Batch response instructions now remain in the system request boundary instead of source JSON; unsupported segments retain explicit source-only outcomes and all-unsupported parents skip the provider.
- Fixed desktop overlay startup disconnects when the renderer reports its first
  visible window through the authenticated bridge.
