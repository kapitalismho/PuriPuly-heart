# Privacy Policy

This policy covers the PuriPuly app and project-operated servers. External services you use have their own policies.

## 1. Usage statistics and your choice

- To estimate user numbers, the app sends a random identifier and the UTC activity date. The server stores a derived identifier with those dates, separately from account authentication. Statistics contain no conversations, audio, translations, or API keys.
- Statistics are on by default for new installations. Declining on the installer's Privacy Policy page disables them before first launch without blocking installation. Settings lets you change this later; upgrades preserve your choice.
- Turning statistics off stops transmission and removes the local identifier. Turning them back on creates a new one. Existing server records remain until cleanup, which removes activity more than 35 UTC days old; records for 36 dates including today may remain.

## 2. Audio, translation, and integrations

- Local models process data on your device. External recognition sends audio; external translation sends text and necessary context. Selected providers and custom servers receive the authentication information configured for them. Their retention, training, and charging policies apply.
- New installations select Bilibili's no-key Index-Translate API at `index-translate.bilibili.com`. Startup and selection send a synthetic availability request even when translation is off. Ordinary translations may send the same request twice. A failed availability check switches to Gemma 4 Managed, or Gemma 4 OpenRouter if the Managed allowance is confirmed exhausted, using that route's normal authentication and processing. Free availability is not guaranteed.
- Qwen Audio sends locally detected speech with leading and trailing audio, which can include pauses, background sound, and overlapping context. Capture selection and mute controls still apply.
- Managed translation goes to OpenRouter and its inference provider with a user reference, not through the project's authentication server. Choose local models or disable the relevant feature to avoid external audio/text processing.
- VRChat context can include participant counts, not names or raw logs. OSC sends subtitles and controls to the configured destination and can discover and advertise services on the local network. Its default destination is local; Settings lets you change or disable it.

## 3. Managed accounts and security

Discord, QQ, and managed services process authentication data, account references, installation/device identifiers, and hardware hashes for access, entitlements, credential delivery, and abuse prevention. Server records include entitlements, key issuance, historical rewards, and IP-based security information. These are separate from optional statistics and have different retention periods. Activity and key-issuance aggregates are sent to an operational Discord channel.

## 4. Network activity even with statistics off

The app checks GitHub for updates at startup and downloads models from services such as Hugging Face or ModelScope. Authentication and external features communicate independently of usage statistics. Local models and disabled statistics do not make the app fully offline.

Update/download requests contain no statistics identifier, conversations, or audio, but recipients and Cloudflare infrastructure process IP addresses and ordinary request metadata. Deleting current server records does not immediately remove backups or records retained from earlier versions.

## 5. Local information and deletion

Settings, models, and logs are stored mainly in `%LOCALAPPDATA%\puripuly-heart`. The runtime log includes recognized text and translations, rotates at 20 MiB, and retains one backup. It is not automatically uploaded and may be incomplete. Separate bounded, metadata-only diagnostics stay in memory and reset on restart; authenticated local diagnostic queries do not expose conversation text or raw log files.

Credentials are stored in Windows Credential Manager or encrypted files. Known secret patterns are redacted from logs, but ordinary sensitive prose may remain. Review logs before sharing. Removing the app data folder does not remove all credentials or external-service records.

## 6. Puri assistant and voluntary support reports

### Assistant requests

Puri starts when you send a message, without a separate chat-start consent dialog. Messages, necessary conversation/tool history, and allowlisted app-state and technical-diagnostic summaries pass through the project's relay to OpenAI. Voice transcripts, translations, audio, raw logs, complete settings, and API keys are not automatically included. Diagnostic tools may send synthetic input to your configured recognition or translation service under its normal policies and charges.

Use the app's secure credential fields, never chat, for API keys. Known-pattern blocking cannot detect every secret. Keys and their hashes are not tool inputs. Puri history and encrypted reasoning stay in app memory. Reset removes active model context but can retain the app-run report archive and recent setting-change facts; restarting clears them.

Broker authenticates the installation. Relay and OpenAI receive a derived installation identifier, not the raw ID; this is pseudonymous, not anonymous. Required usage/cost records, random prompt-turn identifiers, request limits, and abuse controls remain active with statistics off. Broker uses derived IP identifiers rather than adding raw-IP storage. Ordinary relay databases and logs retain no conversation content, setting values, or raw diagnostics.

OpenAI `store: false` does **not** mean Zero Data Retention. According to [OpenAI's data controls](https://developers.openai.com/api/docs/guides/your-data), checked 2026-10-30:

- API data is not used for training unless the customer opts in.
- Abuse-monitoring content can remain for up to 30 days, or longer for legal or harm-prevention reasons.
- Encrypted prompt-cache tensors can remain for up to 24 hours.
- Special retention controls require provider approval and have limitations; this project does not claim they are enabled.

### Voluntary support reports

Project servers retain conversation content only in reports you explicitly send with the in-chat button. A suggestion alone creates or uploads nothing. A report contains:

- This app run's entire Puri conversation, including before resets, tool calls/results, and notices; reasoning blobs are excluded.
- The current runtime log, including recognized text and translations; rotated backups are excluded.
- A random ticket, app/OS/protocol details, Puri's summary, and diagnostic codes.

Secret masking cannot remove all sensitive prose. Reports go to a private, developer-access-only Cloudflare R2 bucket with a 30-day lifecycle; deletion normally follows within about 24 hours after expiry. Discord notifications contain only the ticket, app version, compressed size, and diagnostic codes, never conversation or log text.

Reports are limited to 25 MB compressed and three per installation per UTC day, separately from model usage. Uploading or receiving a receipt makes no model request. Uncertain delivery requires an explicit retry; authentication rejection may trigger one token refresh for the same authorized upload.

## 7. Contact

The project operator is kapitalismho on GitHub. Contact routes: https://github.com/kapitalismho/PuriPuly-heart
