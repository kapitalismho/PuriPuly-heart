# Privacy Policy

This policy covers the PuriPuly app and project-operated servers. The policies of any external services you choose also apply.

## 1. Usage statistics and your choice

To estimate user numbers, we send a random identifier and the date the app was used (UTC). The statistics do not include conversations, audio, translations, or API keys. The server transforms the identifier and stores it with activity dates, without linking it to account authentication information.

For new installations, agreement is selected by default and statistics are on. Choosing not to agree on the Privacy Policy page prevents statistics from being sent from the first launch, and you can still continue installation. You can also change this in Settings afterward. Upgrades preserve your existing choice unless you change it.

Turning statistics off stops subsequent statistics transmission and removes the device’s statistics identifier. Turning them back on creates a new identifier. Activity already stored on the server is not deleted immediately. Periodic cleanup deletes activity dated more than 35 days before today in UTC, so records for 36 dates, including today, may remain after cleanup.

## 2. Audio, translation, and integrations

Local models process data on your device. External speech recognition sends audio to the selected service; external translation sends source text and the necessary context. API keys authenticate with the service. Custom servers also receive the audio or text and authentication information configured for them.

When Qwen Audio is selected, each channel sends audio selected by local speech detection, including its leading context and trailing audio. These portions can contain pauses or background sound, and context can overlap previously sent audio. A preceding recognition task can finish while the next task receives audio on another connection, with at most two connections per channel and four when both channels use Qwen. Handover does not duplicate the whole stream, add another context replay or send fabricated keepalive audio. Unlike the previous continuous Peer path, local segment boundaries end their assigned tasks rather than leaving an all-capture stream open between segments. Capture permission, source selection and activation/mute controls continue to apply; external usage and charges can still vary.

Managed translation is processed by OpenRouter and the actual inference provider and includes a user reference. The project’s authentication server does not ordinarily relay translation content. External retention and training use depend on the service’s policies and settings; we do not guarantee zero retention or no training use for every request. To avoid external processing, choose local models or turn the relevant feature off.

VRChat integration may include participant counts in translation context, but does not send participant names or raw logs. OSC sends subtitles and control information to the configured destination; automatic connection discovers and advertises services on the local network. The default destination is on your device, and you can change or disable the connection mode in Settings.

## 3. Managed accounts and security

Managed authentication uses account services such as Discord or QQ. Managed services process account authentication information, installation and device identifiers, hashes of hardware information, and related data for authentication, entitlement checks, key delivery, and abuse prevention. The server stores account references, entitlements, key issuance and reward records, and IP-based security records, among other related records. These are separate from usage statistics, and not all records share one retention period. Aggregates such as activity and key issuance counts are sent to an operational Discord channel.

## 4. Network activity even with statistics off

At startup, the app checks GitHub for updates. Required models are downloaded from services such as Hugging Face or ModelScope. Authentication and external features you choose also communicate separately. Using local models or turning statistics off does not stop all of this network activity.

Download and update requests do not include the statistics identifier, conversations, or audio, but the destination service processes IP addresses and ordinary request information. Processing by the project’s Cloudflare infrastructure is also separate from the two fields in the statistics body. Deleting server records does not mean all backups or records from earlier versions are immediately deleted too.

## 5. Local information and deletion

Settings, models, and diagnostic logs are stored mainly in `%LOCALAPPDATA%\puripuly-heart`. The existing local runtime file includes accepted speech text and translations so that the local Conversation view and support logs can reconstruct a turn; a source is recorded once per semantic turn and each target translation is recorded separately. These records are bounded in length, and secret-shaped values such as API keys, bearer tokens, private keys, and credentials embedded in URLs are redacted. Basic-audience records appear live and in the local file; selected technical diagnostics are file-only. Automated redaction cannot guarantee that every sensitive phrase in ordinary conversation will be recognized.

Failure diagnostics are a separate metadata-only projection within the existing logging service, not a copy of the conversation-bearing file. The authenticated local CLI can query up to 256 current-run diagnostic records within a 512 KiB memory bound, with at most 4 KiB per record and 100 records per query. Records include stable failure codes, time, safe component/stage and platform/provider classifications, and applicable operation/turn/generation identities. They exclude conversation text, audio, screenshots, participant names, raw VRChat logs, credentials, authorization URLs, provider response bodies, arbitrary child stderr/JSON, personal paths, exception messages, and tracebacks. Optional rejected or truncated detail is disclosed. History is evicted at its bounds and resets on restart; it is not a durable audit. Diagnostic queries do not enable transcript or translation access, raw-file reading, or additional mutation privileges. Existing explicit content subscriptions remain separate.

The UTF-8 runtime file rotates at 20 MiB and retains one backup. Writes are buffered and batched, so forced termination, power loss, stalled or failed storage, queue pressure, or an individual record larger than the file cap can prevent persistence. A live diagnostic-health query reports bounded history, rejection, subscriber, queue, writer, and shutdown observations independently of successful file writing. Enqueue does not establish a write, a write does not guarantee durability, and aggregate health does not prove that an individual diagnostic reached disk. These changes add no remote upload or separate always-on diagnostic store.

Secrets used by the app are stored in Windows Credential Manager or encrypted files. Review and redact the conversation-bearing local log before sharing it. Removing the app data folder during uninstallation does not delete all information in Windows Credential Manager or external services.

## 6. Contact

The project operator is kapitalismho on GitHub. Contact routes are available in the project repository: https://github.com/kapitalismho/PuriPuly-heart
