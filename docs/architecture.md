# Architecture

System map for PuriPuly Heart: who owns state, how work crosses boundaries, and where to read the implementation. For commands and result interpretation, see the [CLI guide](cli.md).

Keep this map about ownership, dependency direction, and cross-component invariants. Update the relevant section when a boundary changes; do not append fix histories, policy values, field catalogs, or test acceptance criteria. Those belong in the owning code and behavior tests.

## System Model

- One Python application runtime serves GUI and headless hosts. Presentation is an adapter, not a second runtime.
- Runtime owners hold state and resources; ports define their contracts; adapters connect UI, audio, providers, storage, and child processes.
- Composition constructs and connects owners. Core code must not depend on UI or composition code.
- The application owns capture, recognition, translation, and output. Native workers own native execution/rendering, not application policy.
- Broker provides managed identity, entitlement, and credential release. It is a control-plane dependency, not part of the normal utterance path.

[Application composition](../src/puripuly_heart/composition/application_runtime.py) loads settings and secrets, selects adapters, connects services, and installs startup/shutdown. Long-lived resources must transfer to explicit owners. The [runtime pipeline launcher](../src/puripuly_heart/app/wiring/wiring_runtime_pipeline.py) installs replaceable capture/provider/output components; callers must not retain them across replacement unless the API permits it.

## Runtime Ownership

Paths below are navigation entry points, not an exhaustive class inventory.

| Area | Responsibility | Entry points |
| --- | --- | --- |
| Application boundary | UI intents; typed control operations, queries, and mutation ordering | [UI boundary](../src/puripuly_heart/app/services/ui_application.py), [control owner](../src/puripuly_heart/app/services/application_control.py) |
| Local control host | Authenticated endpoint and instance lifetime | [Host](../src/puripuly_heart/cli/host.py), [instance ownership](../src/puripuly_heart/core/control_instance.py) |
| Settings | Canonical intent, persistence, and runtime application coordination | [Persistence](../src/puripuly_heart/app/services/canonical_settings_persistence.py), [settings application](../src/puripuly_heart/app/services/settings/settings_application.py) |
| Capture | Source authority, audio ordering, segmentation, and capture lifecycle | [Self](../src/puripuly_heart/core/runtime/self_capture.py), [Peer](../src/puripuly_heart/core/runtime/peer_channel.py) |
| Recognition | Channel-scoped STT; shared local ASR resources and backend transitions | [Scoped engine](../src/puripuly_heart/core/stt/scoped_engine.py), [local ASR runtime](../src/puripuly_heart/core/runtime/local_asr_provider_runtime.py) |
| Translation | Self source progression; turn admission, cancellation, and publication authority | [Self channel](../src/puripuly_heart/core/orchestrator/self_translation_channel.py), [turn lifecycle](../src/puripuly_heart/core/orchestrator/translation_turn.py) |
| Provider requests | Request preparation and provider authority; provider replacement | [Request owner](../src/puripuly_heart/core/orchestrator/translation_request.py), [provider rebuild](../src/puripuly_heart/core/runtime/provider_rebuild.py) |
| Local models | Asset operations; managed translation server and model lifecycle | [Model assets](../src/puripuly_heart/app/services/model_assets.py), [Gemma runtime](../src/puripuly_heart/core/local_translation/runtime.py) |
| Output | Routing, independent destination delivery, and delivery history | [Output runtime](../src/puripuly_heart/core/runtime/output.py), [translation projection](../src/puripuly_heart/core/orchestrator/translation_output_projection.py) |
| Overlays | Target selection, logical captions, child lifecycle, and native rendering | [Application owner](../src/puripuly_heart/app/services/overlay/overlay_application.py), [Python runtime](../src/puripuly_heart/core/runtime/overlay.py), [native runtime](../native/overlay/src/runtime.rs) |
| Accounts | Managed authentication/credentials; separate ChatGPT session | [Managed wiring](../src/puripuly_heart/app/wiring/wiring_managed_account.py), [ChatGPT owner](../src/puripuly_heart/app/services/chatgpt_account.py) |
| VRChat integration | OSC control/state; independent scene-context observation | [OSC runtime](../src/puripuly_heart/app/services/osc/control_runtime.py), [scene owner](../src/puripuly_heart/core/vrchat_scene_service.py) |
| Diagnostics | Console/file/Logs delivery and bounded current-instance evidence | [Logging service](../src/puripuly_heart/app/services/application_runtime_logging.py) |
| Shutdown | Ordered teardown across owners | [Shutdown adapter](../src/puripuly_heart/app/adapters/application_runtime_shutdown.py) |

An owner may span several processing stages. GUI, CLI, and OSC mutations share application ordering and resource-conflict boundaries; operations belong to the application, not the requesting connection. Queries distinguish saved intent from effective runtime state. Presentation and notifications are never state authorities.

## Data Handoffs

```mermaid
flowchart LR
    Mic[Microphone] --> Self[Self capture / VAD]
    Self --> SelfSTT[Self recognition]
    SelfSTT --> SelfTurn[Self translation turns]
    Text[Manual text] --> SelfTurn
    Audio[Loopback / process audio] --> Peer[Peer capture / segmentation]
    Peer --> PeerSTT[Peer recognition]
    PeerSTT --> PeerTurn[Ordered Peer translation admission]
    SelfTurn --> Output[Output runtime]
    PeerTurn --> Output
    Output --> UI[UI / overlays]
    Output -->|Self only| Chatbox[VRChat chatbox]
```

### Capture and recognition

- Capture owns source order, timing, and audio scope. Audio loss is explicit, not silence; segmentation does not depend on provider readiness.
- Self and Peer share audio contracts but retain separate endpoint policies and recognition lifecycles. Local resources may be shared without sharing channel authority.
- Provider adapters translate native events into shared STT contracts. Local audio completion and transcript delivery are distinct; provider recovery stays inside the recognition boundary.
- Source completion drains admitted input. Stop, discontinuity, and replacement revoke affected authority so late results cannot enter current work.
- VAD-only audio tests exclude normal capture. The application owner handles admission, input release, and restoration of still-authorized capture.

Contracts: [audio ownership](../src/puripuly_heart/core/audio/ownership.py), [audio delivery](../src/puripuly_heart/core/audio/listen_delivery.py), [audio-test owner](../src/puripuly_heart/app/services/audio_test.py). Behavior: [scoped recognition tests](../tests/core/test_stt_scoped_engine.py).

### Translation

- Provider adapters own provider-specific authentication, requests, streaming, and errors behind the common provider boundary. Turn owners own admission, cancellation, stale-result rejection, and publication.
- Self source closure and translation completion are independent. Speculative results are not accepted history; source/target revisions and turn identity govern presentation.
- Self and Peer have separate admission while sharing provider capacity. Peer execution may be concurrent, but context preparation and publication preserve admitted order.
- Manual text bypasses capture/STT and uses Self publication. Admission, translation completion, and destination delivery are separate outcomes.
- Managed local translation owns its server, model, and capacity behind the same provider boundary. Requests retain native resources until settlement; model/backend replacement drains work before admitting a replacement.

Behavior: [turn lifecycle](../tests/core/test_translation_turn_owner.py), [request ownership](../tests/core/test_translation_request_owner.py), [managed local runtime](../tests/core/local_translation/test_gemma_runtime.py). Model assets and execution profiles live in [assets.py](../src/puripuly_heart/core/local_translation/assets.py) and [runtime_profile.py](../src/puripuly_heart/core/local_translation/runtime_profile.py), not this map.

### Output and context

- Destinations have independent bounded delivery. One destination's failure must not block another or replay recognition/translation. Peer speech never reaches the chatbox; system disclosures require an explicit route.
- Output preserves turn/publication identity. Revisions may update retained turns without refocusing the dashboard or reviving retired captions. Preview presentation is separate from final accepted history.
- Python owns logical caption state and lifetime; native overlays own rendering. Presentation does not gate translation, and delivery receipts do not prove physical display or remote UDP receipt.
- The VRChat scene owner survives pipeline rebuilds and has no audio, VAD, or OSC dependency. Only trusted, sanitized population context reaches translation; names/raw logs stay local, and custom HTTP extensions receive no scene data.

Behavior: [output runtime](../tests/core/runtime/test_output_runtime.py), [caption policy](../tests/core/test_self_caption_policy.py), [overlay presenter](../tests/core/test_overlay_presenter.py). Protocols: [native overlays](../native/overlay/README.md), [VRChat OSC](vrchat-osc.md), [HTTP extensions](http-extensions.md).

## Settings and Credentials

| State | Authority |
| --- | --- |
| Persisted intent | Settings owner and canonical persistence |
| Resolved configuration | [Resolution](../src/puripuly_heart/config/resolved.py) and [runtime selection](../src/puripuly_heart/config/runtime_resolution.py) |
| Effective runtime state | The owner of the active resource or operation |

- UI surfaces consume frozen snapshots and emit focused typed intents. The settings owner rebases changes onto current canonical state and serializes commit/rollback; observers publish afterward in commit order.
- Persistence and runtime application are distinct. Runtime effects cannot assign canonical settings; failed application must remain degraded/retryable rather than appear applied or silently undo durable intent.
- Transactional application uses the current committed snapshot. Explicit runtime-only overrides remain ephemeral and retain their admitted configuration throughout resource construction/recovery.
- Secret-store instances observe one backend, not independent credential snapshots. Mutations preserve unrelated entries across instances/processes; unreadable storage is unknown state, not proof of absent credentials.
- Managed identity authority belongs to the settings owner. Credential storage, remote acknowledgment, authentication completion, and provider activation are separate transitions. Modern credentials require matching durable delivery proof and canonical authority; recovery resumes the same operation rather than issuing a replacement on uncertainty.
- ChatGPT authentication is separate from Broker and survives provider replacement. Refresh credentials use the secret store; access tokens remain in memory.

Entry points: [provider application](../src/puripuly_heart/app/services/provider/provider_runtime_apply.py), [secret storage](../src/puripuly_heart/core/storage/secrets.py), [managed authentication](../src/puripuly_heart/app/services/managed/), [credential proof](../src/puripuly_heart/core/managed_credential.py). Behavior: [control contract](../tests/app/test_application_control_contract.py), [provider application](../tests/app/test_stt_provider_apply_vertical.py), [secret persistence](../tests/core/test_encrypted_secret_store.py).

## Lifecycle and Concurrency

- Every task, source, process, and provider session has an owner before work starts. That owner stops ingress, cancels or drains work, rejects late callbacks, and releases resources.
- Generations, attachment tokens, and request identities fence retired work. Replacement revokes old authority; retired resources remain owned until cleanup completes. Caller cancellation is not proof that native/thread/process work settled.
- Application state and async clients stay on the Python event loop. Blocking device/model/storage/startup work uses owned threads, executors, or child processes; [owned-thread cancellation](../src/puripuly_heart/core/owned_thread.py) drains started synchronous work before resource release.
- Events cross ports, callbacks, or owned queues. Receivers mutate their own state. Ordering is local to the owner/channel, not global across UI, Self, Peer, and providers.
- Matching stops may preempt pending starts, while cleanup fences older queued starts and later replacements. Removing one channel's demand does not revoke another channel's shared resource.
- Shutdown freezes ingress and startup, drains/cancels owned work, closes external resources, then clears runtime references. Primary operational failures and secondary cleanup failures remain distinct; failed cleanup stays owned and visible.

[Dependency checks](../tests/architecture/test_dependency_boundaries.py) enforce import boundaries and record the narrow existing localization exception. [Task-ownership checks](../tests/architecture/test_lifecycle_task_guard.py) record reviewed creation sites and remaining lifetime debt. These structural checks do not prove runtime draining or cleanup; owner behavior tests do.

## Diagnostics and Runtime Layout

- Owners produce typed causes and known operation/request/turn identities before reducing them to user messages. [Diagnostics](../src/puripuly_heart/core/diagnostics.py) and [runtime logging](../src/puripuly_heart/core/runtime_logging.py) validate and project that evidence; arbitrary exception prose is not a classification source.
- Technical diagnostics exclude credentials, conversation content, audio, arbitrary child output, and personal paths. Accepted conversation records have a separate protected path; CLI content requires explicit opt-in.
- History, subscribers, and file delivery are bounded. Loss and cleanup failure remain observable independently of a failed sink. Enqueued, written, durable, and physically displayed are different claims; current-instance history is not a durable audit. See [CLI diagnostics](cli.md#retained-diagnostics-and-delivery-health).
- [Runtime layout](../src/puripuly_heart/runtime_layout.py) separates interpreter/host paths, immutable resources, and writable user data. Features use this boundary rather than the working directory or launcher flags. Packaged code changes require rebuilding the payload; startup never writes into installed directories.
- Model registries own pinned sources and integrity contracts. Provisioning validates owned staging before atomic replacement. Explicit asset operations have their own cancellation lifetime, separate from runtime demand. See [ASR installation](../src/puripuly_heart/core/local_asr/local_stt_runtime_installer.py) and [translation provisioning](../src/puripuly_heart/core/local_translation/provisioning.py).

Behavior: [runtime logging](../tests/core/test_runtime_logging.py), [output diagnostics](../tests/core/test_output_logging.py), [packaged layout](../tests/release_evidence/test_native_runtime_layout.py), [model provisioning](../tests/core/local_translation/test_gemma_provisioning.py).
