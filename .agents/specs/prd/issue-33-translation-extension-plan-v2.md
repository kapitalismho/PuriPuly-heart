# [Feature] Add Custom HTTP translation extensions behind a provider-neutral `TranslationBackend`

## Working basis

- Original request: #33 — **Support non-LLM translator API endpoint**.
- Planning baseline: `dev@6f74a8e23106bd626148439d6d4a997656afb447` (`fix: stabilize desktop overlay startup`).
- Implementation target: the current `dev` architecture. If `dev` moves before implementation begins, re-read the affected production paths and reconcile this plan with the new head rather than porting file-by-file assumptions blindly.
- This plan keeps the previously selected **B architecture**: introduce a provider-neutral `TranslationBackend` boundary above the existing `LLMProvider` implementations.
- The Custom HTTP product UX is intentionally modeled after the VRHandsFrame extension approach: HTTP request/response behavior is declared in files rather than edited through a large HTTP-builder UI.
- This document is an implementation plan only. It does not create or update a GitHub issue and does not authorize a branch, commit, push, merge, or release.

---

## Summary

Add `Custom HTTP API` as one first-class option in the existing translation-engine selector alongside the current LLM-backed choices, including `Local LLM`.

When the user selects `Custom HTTP API`, the normal translation connection/model detail surface is replaced by a dedicated Custom HTTP detail card. The card does **not** expose a Postman-like request editor. It lets the user:

- choose one installed translation extension;
- enter only the credentials declared by that extension;
- see the extension description;
- open the user extension directory; and
- reload extension files after editing them.

Each extension is one `.json` file. The application scans:

```text
<user_config_dir>/translation_extensions/*.json
```

where `<user_config_dir>` is the existing platform-specific PuriPuly configuration directory.

A real service example is committed as:

```text
examples/translation_extensions/libretranslate.json
```

using the managed LibreTranslate API shape. It is an example file only and is not automatically copied into the active user extension directory.

The runtime architecture remains provider-neutral:

```text
TranslationRequestOwner
        |
        v
TranslationBackend
     /       \
    /         \
LlmTranslationBackend     HttpExtensionTranslationBackend
        |                         |
        v                         v
 existing LLMProvider       TranslationExtension
                                + httpx
```

The HTTP translator must not pretend to be an `LLMProvider`.

---

## Background

Issue #33 asks for a configurable non-LLM translation endpoint so PuriPuly does not need to directly maintain separate Google Translate, DeepL, Papago, and similar provider integrations.

The issue explicitly references VRHandsFrame's extension-object approach: service-specific request and response knowledge should be represented as data that users can supply, rather than requiring every service to become a hard-coded application provider.

The current `dev` translation request path is still LLM-shaped at the orchestration boundary:

```text
TranslationRequestOwner
  -> captures the active runtime generation
  -> treats the active runtime object as LLMProvider
  -> prepares prompt/context/languages
  -> calls LLMProvider.translate(...)
  -> rejects stale completions after provider replacement
```

The runtime handle itself is already more generic than that request boundary. `ProviderRuntimeHandle` owns an object, a generation, replacement, and awaited close semantics. The implementation should therefore generalize the translation boundary while preserving the existing runtime ownership model.

The Settings product model already presents translation choices through the existing translation model/provider selector, and `Local LLM` already uses a conditional detail surface. `Custom HTTP API` should use the same UX pattern rather than introducing a second top-level backend selector.

---

# Product requirement

> A user can select `Custom HTTP API` as the translation engine, choose an installed JSON translation extension, enter the credentials required by that extension, and use the extension as the active translation backend without changing existing LLM behavior or losing the user's previous LLM configuration.

The feature is an extension mechanism, not a built-in provider catalog.

---

# Product UX

## Translation-engine selection

Expose `Custom HTTP API` in the same product-level selector that already contains the current translation engines, including `Local LLM`.

Conceptually:

```text
Translation engine

[ Gemma / OpenRouter ... ]
[ Gemini ...             ]
[ Local LLM              ]
[ Custom HTTP API        ]
```

Do not add a second `LLM | Custom HTTP` toggle above the existing engine selector.

At the persisted product-model level, add one canonical translation model/engine alias:

```text
custom_http
```

The exact display string remains localized UI text:

```text
Custom HTTP API
```

## Custom HTTP detail card

When `translation.model == "custom_http"`, replace/hide LLM-specific connection controls with one dedicated detail card.

Conceptual layout:

```text
Custom HTTP API

Extension
[ LibreTranslate                         v ]

LibreTranslate managed translation API

Credentials
API Key
[ **************************************** ]

Extension directory
C:\Users\<user>\AppData\Local\puripuly-heart\translation_extensions
                                         [ Open folder ]

                                              [ Reload ]
```

If an extension declares no credentials:

```text
Credentials
No credentials required.
```

If no valid extensions are installed:

```text
Extension
[ No extensions found ]

Add a .json translation extension to:
<resolved extension directory>

[ Open folder ]     [ Reload ]
```

## No trust-confirmation dialog

Do **not** add an endpoint trust-confirmation dialog or first-use warning.

Selecting and applying a Custom HTTP extension should follow the same explicit Settings apply semantics as selecting another translation engine.

## No HTTP editor

The Settings UI must not expose:

- method editing;
- URL editing;
- arbitrary header rows;
- query editors;
- JSON/form body editors;
- response selectors;
- JSON Pointer editors;
- raw request/response previews; or
- arbitrary template editing.

Those belong to the extension JSON file.

## Reload behavior

`Reload` re-scans the extension directory and re-validates all `*.json` files.

Requirements:

- no application restart required after editing an extension file;
- one invalid extension must not prevent other valid extensions from loading;
- if the selected extension definition changes, rebuild the active translation backend generation after successful Settings/runtime apply semantics;
- if the selected extension disappears or becomes invalid, do not silently switch to another extension;
- surface a concise invalid-extension count/error through the existing Settings notification pattern rather than a new diagnostics screen.

No file watcher is required in #33. Loading occurs on application/settings initialization and explicit `Reload` only.

---

# Extension storage and discovery

## User extension directory

Add to `src/puripuly_heart/config/paths.py` conceptually:

```python
TRANSLATION_EXTENSIONS_DIRNAME = "translation_extensions"


def default_translation_extensions_dir() -> Path:
    return user_config_dir() / TRANSLATION_EXTENSIONS_DIRNAME
```

The current `user_config_dir()` behavior therefore resolves to approximately:

```text
Windows
%LOCALAPPDATA%\puripuly-heart\translation_extensions\

macOS
~/Library/Application Support/puripuly-heart/translation_extensions/

Linux
$XDG_CONFIG_HOME/puripuly-heart/translation_extensions/

Linux fallback
~/.config/puripuly-heart/translation_extensions/
```

The directory is created when the Custom HTTP extension subsystem is initialized or when the user presses `Open folder`.

## File model

Use:

```text
1 extension = 1 JSON file
```

Examples:

```text
translation_extensions/
  libretranslate.json
  papago-custom.json
  company-translator.json
```

Do **not** use JSONL.

Reasons:

- individual extensions can be copied/shared independently;
- one malformed file cannot make a multi-entry JSONL document unusable;
- filenames are useful in diagnostics;
- file-level editing and version control remain simple;
- an extension can be removed by deleting one file.

## Discovery rules

At load/reload:

1. ensure the extension directory exists;
2. enumerate only direct-child `*.json` files;
3. parse each file independently as UTF-8 JSON;
4. validate each document against Extension Schema v1;
5. build a registry keyed by extension `id`;
6. keep valid extensions even when other files fail;
7. report invalid files without aborting the entire registry;
8. reject duplicate IDs deterministically rather than choosing based on directory order.

Do not recurse into subdirectories in v1.

Do not execute or import Python, JavaScript, shell, Jinja, or other code from the extension directory.

## Identity

Extension identity comes from JSON `id`, not the filename.

This must remain stable across filename changes:

```json
{
  "id": "libretranslate"
}
```

Recommended ID validation:

```regex
[a-z0-9][a-z0-9._-]{0,63}
```

IDs are case-sensitive only if the implementation can prove cross-platform filesystem/UI behavior is still unambiguous. Prefer normalizing/requiring lowercase in v1.

---

# Bundled real-service example

Commit one real service example:

```text
examples/translation_extensions/libretranslate.json
```

Use the managed LibreTranslate endpoint documented at:

```text
https://libretranslate.com/translate
```

LibreTranslate's documented simple translation request uses `POST`, accepts `q`, `source`, `target`, and `api_key`, and returns the translated string in `translatedText`.

The sample must not contain a real credential.

Recommended example:

```json
{
  "schema_version": 1,
  "id": "libretranslate",
  "name": "LibreTranslate",
  "description": "LibreTranslate managed translation API",
  "url": "https://libretranslate.com/translate",
  "request": {
    "query": {},
    "body": {
      "type": "json",
      "value": {
        "q": "{{text}}",
        "source": "{{source_language}}",
        "target": "{{target_language}}",
        "format": "text",
        "api_key": "{{secret:api_key}}"
      }
    }
  },
  "response": {
    "type": "json",
    "pointer": "/translatedText"
  },
  "secrets": [
    {
      "id": "api_key",
      "label": "API Key"
    }
  ]
}
```

This example is a reference/configuration template, not an automatically installed active extension.

Requirements:

- do not copy it into the user's `translation_extensions` directory automatically;
- do not make network traffic from the example during application startup or migration;
- do not bundle a LibreTranslate API key;
- document that the managed `libretranslate.com` service requires an API key, while other/self-hosted LibreTranslate instances may have different credential requirements;
- tests for the example must use a local fake HTTP server, not the live LibreTranslate service.

The sample should be covered by the same parser/validator tests as user files so documentation cannot drift away from the implemented schema.

---

# Translation Extension Schema v1

## Top-level shape

```text
TranslationExtensionV1
|
|-- schema_version: 1
|-- id: string
|-- name: string
|-- description?: string
|
|-- url: string
|-- headers?: map<string, string>
|
|-- request
|   |-- query?: map<string, string>
|   `-- body
|       |-- type: "json" | "form" | "none"
|       `-- value?: JSON-compatible object / string map
|
|-- response
|   |-- type: "json" | "text"
|   `-- pointer?: string
|
|-- secrets?: array
|   `-- { id, label }
|
`-- language_map?
    |-- source?: map<string, string>
    `-- target?: map<string, string>
```

## Required fields

Require:

```text
schema_version
id
name
url
request.body.type
response.type
```

`request.body.value` is required for `json` and `form` and forbidden/ignored for `none`.

`response.pointer` is required for `response.type == "json"` and not used for `text`.

## HTTP method

Extension Schema v1 is **POST-only**.

Do not expose or persist a `method` field in v1.

This deliberately narrows the request matrix and matches the reference extension use case without turning the feature into a general HTTP automation framework.

A future issue may add GET only if a concrete translation service requires it and the security/validation model remains bounded.

## URL

`url` must:

- be an absolute URL;
- use only `https` or `http`;
- contain no extension placeholder in the scheme or authority/host component;
- not include embedded user-info credentials;
- be validated before backend activation.

Do not add a trust dialog.

## Headers

`headers` is an optional string-to-string map.

Placeholders may appear in header values.

Block transport-managed or dangerous/confusing header names such as:

```text
Host
Content-Length
Transfer-Encoding
Connection
```

Do not let an extension override runtime transport ownership through these fields.

## Query

`request.query` is an optional string-to-string map appended as URL query parameters.

Placeholders may appear in query values.

Do not string-concatenate the final query URL manually. Use `httpx` parameter handling.

## Body types

### `json`

Persist a JSON-compatible value and recursively substitute placeholders in string leaves.

Send through:

```python
client.post(..., json=rendered_value)
```

Do not construct JSON by concatenating source speech/text into a JSON string.

### `form`

Require a string-keyed map whose rendered values become form data.

Send through:

```python
client.post(..., data=rendered_mapping)
```

### `none`

Send no request body.

## No raw body in v1

Do not support arbitrary raw request content in Schema v1.

If a future translation service proves this is necessary, add it deliberately with explicit encoding/content-type rules rather than accepting arbitrary bytes now.

---

# Placeholder language

Support exactly these placeholders in v1:

```text
{{text}}
{{source_language}}
{{target_language}}
{{secret:<id>}}
```

Examples:

```text
{{secret:api_key}}
{{secret:client_id}}
{{secret:client_secret}}
```

No Jinja or executable expressions.

Explicitly unsupported:

```text
{{ text | urlencode }}
{{ if ... }}
{{ env.API_KEY }}
{{ file(...) }}
{{ expression(...) }}
{{ python(...) }}
```

Unknown placeholders are validation errors, not pass-through strings.

## Placement

Allow placeholders in:

- header values;
- query values;
- JSON string leaves; and
- form values.

Do not allow placeholders to change the URL scheme or host.

The static URL path may remain fully declared by the extension file. Path placeholders are not required in v1.

---

# Secrets

## Declaration

Extensions explicitly declare the credentials they require:

```json
"secrets": [
  {
    "id": "client_id",
    "label": "Client ID"
  },
  {
    "id": "client_secret",
    "label": "Client Secret"
  }
]
```

Recommended secret ID validation:

```regex
[a-z0-9][a-z0-9._-]{0,63}
```

## Storage

Never store secret values in the extension JSON or canonical `settings.json`.

Use the existing SecretStore with namespaced dynamic keys:

```text
translation_extension.<extension_id>.<secret_id>
```

Examples:

```text
translation_extension.libretranslate.api_key
translation_extension.papago.client_id
translation_extension.papago.client_secret
```

## Validation

At extension-load time:

- every `{{secret:<id>}}` reference must point to a declared secret;
- duplicate secret declarations are invalid;
- unused declared secrets may either be allowed with a warning or rejected; prefer rejection in v1 to keep the contract precise.

At backend activation/request time:

- required secret values must exist before network I/O;
- missing values must produce a user-actionable configuration error;
- secret values must never appear in logs, diagnostics, runtime signatures, exceptions surfaced to telemetry, or committed test evidence.

## Secret changes

Use the existing Settings secret transaction semantics.

When a credential for the active extension changes successfully, update/rebuild the active HTTP backend as needed by the chosen secret-resolution implementation. Do not put the secret value itself into provider/backend signatures.

---

# Language mapping

PuriPuly keeps canonical product language codes internally.

The extension may optionally translate those codes to service-specific codes at the final HTTP boundary:

```json
"language_map": {
  "source": {
    "zh-CN": "zh",
    "zh-TW": "zt"
  },
  "target": {
    "zh-CN": "zh",
    "zh-TW": "zt"
  }
}
```

Flow:

```text
detected/product language
        |
        v
canonical PuriPuly code
        |
        v
TranslationBackendRequest
        |
        v
extension language_map
        |
        v
rendered request
```

Rules:

- source and target maps are independent;
- if a code is absent from the relevant map, pass the canonical code through unchanged;
- mapping must not mutate persisted canonical language state;
- mapping is not used to decide which languages PuriPuly exposes globally.

Generalize LLM-named language helper terminology only where the new translation boundary directly crosses it. Avoid unrelated repository-wide renames.

---

# Response extraction

Support exactly two modes.

## Plain text

```json
"response": {
  "type": "text"
}
```

The entire successful response body, after normal text decoding, is the translated string.

Reject an empty translated result.

## JSON + JSON Pointer

```json
"response": {
  "type": "json",
  "pointer": "/data/translations/0/text"
}
```

Use RFC 6901 JSON Pointer semantics.

Requirements:

- parse JSON exactly once;
- support object keys and array indices;
- support `~0` and `~1` escaping;
- allow the empty pointer `""` to refer to the root when appropriate;
- the extracted value must be a string;
- missing path, malformed pointer, invalid JSON, or non-string result is an extension response error.

Do not implement:

- JSONPath;
- jq;
- regex scraping;
- XPath;
- arbitrary expressions; or
- executable response transforms.

---

# Extension registry

Introduce a narrow registry/service responsible only for filesystem-backed extension definitions.

Conceptually:

```python
@dataclass(frozen=True, slots=True)
class LoadedTranslationExtension:
    definition: TranslationExtension
    source_path: Path
    fingerprint: str


@dataclass(frozen=True, slots=True)
class TranslationExtensionLoadError:
    source_path: Path
    message: str


class TranslationExtensionRegistry:
    def reload(self) -> TranslationExtensionRegistrySnapshot: ...
    def get(self, extension_id: str) -> LoadedTranslationExtension | None: ...
```

The fingerprint represents the validated non-secret extension definition and may be used to detect active-definition changes after `Reload`.

Do not include secret values in the fingerprint.

## Failure isolation

Example directory:

```text
libretranslate.json  valid
papago.json          valid
broken.json          invalid response.pointer
```

The registry still exposes the two valid extensions.

`broken.json` is reported separately.

Duplicate `id` definitions should invalidate every conflicting definition for that ID rather than picking whichever file happens to enumerate first.

## Reload semantics

If the currently selected extension:

- remains valid and unchanged: no unnecessary runtime replacement;
- remains valid but fingerprint changes: rebuild the active backend generation;
- disappears: selected settings remain explicit but runtime becomes unavailable/degraded until corrected;
- becomes invalid: do not fall back to a different extension or LLM automatically.

---

# Provider-neutral translation boundary

Add a translation-specific contract above `LLMProvider`.

Conceptually:

```python
@dataclass(frozen=True, slots=True)
class TranslationBackendRequest:
    utterance_id: UUID
    text: str
    source_language: str
    target_language: str
    instruction: str = ""
    context: str = ""


class TranslationBackend(Protocol):
    async def translate(self, request: TranslationBackendRequest) -> Translation: ...
    async def close(self) -> None: ...
```

The exact request shape may be adapted to current code organization, but the orchestration boundary must no longer depend on `LLMProvider`.

## LLM adapter

Add a thin adapter:

```text
LlmTranslationBackend
```

It maps the provider-neutral request to the existing `LLMProvider.translate(...)` call:

```text
instruction -> system_prompt
text -> text
source_language -> source_language
target_language -> target_language
context -> context
```

The adapter must preserve current LLM behavior exactly before the HTTP backend is considered complete.

Do not rewrite the existing provider implementations.

## HTTP adapter/backend

Add:

```text
HttpExtensionTranslationBackend
```

It owns:

- one validated `TranslationExtension` definition;
- the resolved declared credentials or a safe injected secret resolver;
- one owned `httpx.AsyncClient` per active runtime generation;
- request rendering;
- HTTP execution;
- response extraction; and
- close semantics.

The HTTP backend does not consume LLM prompt/context unless a future extension placeholder explicitly defines such a product requirement. Schema v1 intentionally only exposes text and source/target language placeholders.

---

# Translation request orchestration

Update `TranslationRequestOwner` so it captures/calls `TranslationBackend`, not `LLMProvider`.

Required invariant:

```text
TranslationRequestOwner must not contain:
if model == "custom_http": ...
```

Backend-specific behavior belongs behind `TranslationBackend` and the runtime factory.

Preserve:

- active-generation capture;
- stale-completion rejection;
- cancellation propagation;
- utterance ownership;
- output normalization/history behavior; and
- existing translation-request lifecycle semantics.

Generalize misleading LLM-only diagnostics at this crossed boundary where practical, for example:

```text
llm_request_start      -> translation_request_start
llm_done               -> translation_done
llm_available          -> translation_available
"LLM is not configured" -> "Translation backend is not configured"
```

Compatibility aliases may remain temporarily if broad renaming would create unrelated churn.

---

# Runtime factory and lifecycle

## Factory

Introduce a translation-backend factory conceptually:

```python
def create_translation_backend(settings, *, extension_registry, secret_store):
    if settings.translation.model == "custom_http":
        extension = extension_registry.require(settings.translation.extension_id)
        return HttpExtensionTranslationBackend(...)

    llm_provider = create_llm_provider(...)
    return LlmTranslationBackend(llm_provider)
```

When `custom_http` is selected:

- do not create the LLM provider;
- do not acquire/prepare managed OpenRouter runtime state that is not required;
- do not initialize unused LLM-specific transport/resources.

## Runtime owner

Reuse the current `ProviderRuntimeHandle` generation/replacement/close lifecycle as the one owner of the active translation backend.

Do not create a second long-lived runtime handle for HTTP.

Required behavior:

```text
LLM -> Custom HTTP
  replace generation
  await retirement/close as current runtime rules require
  late LLM result cannot publish

Custom HTTP -> LLM
  replace generation
  close AsyncClient
  late HTTP result cannot publish

Extension reload with changed active definition
  replace generation
  close old AsyncClient
  old completion cannot publish
```

## Runtime signature

Generalize the translation runtime signature only as far as needed.

For `custom_http`, include non-secret state such as:

```text
translation.model == custom_http
selected extension_id
validated extension fingerprint
concurrency limit if runtime-owned
```

Never include secret values.

If credential changes use an explicit rebuild signal rather than encoding secret material in the signature.

Avoid repository-wide `llm_*` renames unrelated to the active translation runtime boundary.

---

# HTTP transport policy

Use the existing `httpx` dependency already present in the repository.

Do not add another HTTP client library.

## Client ownership

Own one `httpx.AsyncClient` per `HttpExtensionTranslationBackend` runtime generation.

Close it when:

- switching away from Custom HTTP;
- switching to another extension;
- reloading a changed active extension;
- rebuilding after required credential/runtime changes; or
- shutting down.

## Method

`POST` only in Extension Schema v1.

## Timeout

Use an application-owned bounded timeout, recommended default:

```text
10 seconds
```

Do not make timeout user-configurable through the extension schema in v1.

A future issue may expose it after concrete service requirements justify doing so.

## Redirects

Disable automatic redirects by default.

## TLS

Keep normal certificate verification enabled.

Do not expose `verify=false` or custom certificate bypasses through Extension Schema v1.

## Status handling

Accept only HTTP `2xx` as successful translation responses.

Categorize failures without leaking request/response contents:

```text
timeout
connect error
TLS error
HTTP status error
invalid response JSON
missing response path
non-string translation
empty translation
configuration error
missing secret
```

## Retry

Do not add automatic HTTP retries in #33.

Retries can duplicate paid requests or hide rate-limit semantics and should only be introduced as an explicit provider-neutral policy later.

## Concurrency

Apply the existing translation concurrency limit to the active HTTP backend.

Avoid double-semaphoring the LLM path if existing `SemaphoreLLMProvider` behavior already owns the LLM-side limit.

The provider-neutral boundary should have one clearly defined effective concurrency policy for each active backend.

## Cancellation

Cancellation from the translation request must propagate into the `httpx` request.

A cancelled request must not publish a late result.

---

# Privacy and diagnostics

Never log or include in telemetry:

- source speech/transcription text;
- rendered request JSON/form body;
- rendered header values;
- query parameter values;
- secret values;
- full response body; or
- extracted response data beyond existing safe product output policy.

Safe diagnostics may include:

```text
backend kind = custom_http
extension id
HTTP method = POST
sanitized endpoint origin/path without query
HTTP status
failure category
timeout/elapsed duration
response type
JSON Pointer string
extension filename for local validation errors
```

Do not include raw authorization headers or URL query values.

---

# Canonical settings and migration

Current planning baseline uses settings schema v32. Introduce the next schema version as required by the repository's normal migration process.

## Persisted translation state

Keep the existing `TranslationIntent` LLM fields intact.

Add only the minimum Custom HTTP product state, conceptually:

```python
TranslationIntent(
    model = ... | "custom_http",
    extension_id: str | None = None,
    # existing connection/history/fallback/provider configuration remains
)
```

Do **not** persist the HTTP extension definition itself in `settings.json`.

Do not persist:

```text
url
headers
query
body
response pointer
language map
secret values
```

Those remain in the extension file or SecretStore.

## Switching behavior

When the user switches:

```text
LLM -> Custom HTTP -> LLM
```

preserve all inactive LLM state:

- previous translation model;
- connection selection/history;
- fallback selection;
- provider-specific settings; and
- existing LLM credentials.

The product-level selector still needs a recoverable previous/default LLM choice if `translation.model` itself becomes `custom_http`. Implement this using the repository's existing model-history/default-resolution conventions rather than discarding the prior LLM choice.

If the current schema cannot preserve the previous LLM model when `model` changes to `custom_http`, add the smallest explicit compatibility/history field necessary rather than duplicating all LLM settings.

## Migration defaults

Existing users migrate with no visible behavior change:

```text
existing translation model remains unchanged
extension_id = null
```

Migration must:

- be idempotent;
- create no network traffic;
- not create or overwrite extension files;
- preserve every current LLM setting; and
- preserve compatibility projections used by older/current settings surfaces.

---

# Fallback behavior

When an LLM model is active:

- existing translation fallback/hedging behavior remains unchanged.

When `custom_http` is active:

- execute only the selected HTTP extension;
- do not automatically fall back to an LLM;
- hide/disable LLM fallback controls as active configuration;
- preserve the stored fallback selection so it returns when the user switches back to an LLM engine.

Cross-backend fallback (`HTTP -> LLM` or `LLM -> HTTP`) is explicitly outside #33.

---

# Settings UI integration

Use the existing Settings contract/renderer pattern rather than mutating Flet controls directly from runtime services.

The current Settings surface already separates translation provider/model controls, translation connection controls, fallback controls, and Local LLM details. Extend that composition pattern.

## Required product states

### LLM engine selected

Existing behavior remains:

```text
Translation engine
[ existing model ]

Connection / provider-specific details
Fallback
Local LLM details when relevant
```

### Custom HTTP selected

Show:

```text
Translation engine
[ Custom HTTP API ]

Custom HTTP API detail card
  Extension dropdown
  Description
  Dynamic credential fields
  Extension directory path
  Open folder
  Reload
```

Hide/disable as appropriate:

- LLM connection selector;
- Local LLM connection/GPU details;
- managed-key controls that are not relevant;
- LLM fallback as active configuration.

Preserve their persisted values.

## Extension selector

The dropdown is populated from the current valid registry snapshot.

Display `name`; persist `id`.

Do not persist the filename as selection identity.

If two extensions somehow reach the UI with the same display name, IDs still distinguish them; duplicate IDs must already have been rejected by registry validation.

## Credential controls

Build credential inputs from `extension.secrets`.

For:

```json
"secrets": [
  {"id": "client_id", "label": "Client ID"},
  {"id": "client_secret", "label": "Client Secret"}
]
```

render:

```text
Client ID
[ ************ ]

Client Secret
[ ************ ]
```

Follow current masked-secret semantics and apply/cancel/clear behavior.

Do not echo stored secret values back into normal UI state.

## Open folder

`Open folder` opens the resolved `default_translation_extensions_dir()` using the platform-appropriate existing shell/open abstraction or the smallest new platform adapter if none exists.

If the directory does not exist, create it first.

Do not open arbitrary paths from extension content.

## Reload

`Reload` is explicit and local-only.

It must not:

- save unrelated settings;
- call the selected translation endpoint;
- test credentials;
- produce translation history/output; or
- replace the active backend unless the currently active validated definition actually changed or became invalid under normal apply/runtime reconciliation.

---

# Suggested code ownership

Exact file names may shift to current `dev` conventions, but keep responsibilities approximately separated as follows:

```text
src/puripuly_heart/core/translation/
  backend.py
  extension_schema.py
  response_pointer.py

src/puripuly_heart/providers/translation/
  llm_backend.py
  http_extension_backend.py
  http_extension_renderer.py

src/puripuly_heart/app/services/
  translation_extension_registry.py

src/puripuly_heart/app/wiring/
  wiring_translation_backend_factory.py
  wiring_provider_runtime.py              # bounded generalization only
  wiring_provider_runtime_policy.py       # backend signature/fingerprint

src/puripuly_heart/config/
  paths.py

src/puripuly_heart/config/settings_vnext/
  schema.py
  migrations/...                          # according to current repo layout

src/puripuly_heart/ui/settings/
  contract.py
  renderer.py
  ... existing provider/model surface implementation

examples/translation_extensions/
  libretranslate.json
```

Avoid creating a generic plugin framework. This is specifically a translation-extension registry and HTTP translation backend.

---

# Implementation sequence

The following is an execution sequence, **not a review-stop sequence**. Review grouping is defined separately below.

## 1. Characterize and extract the translation backend boundary

- [ ] Lock current LLM translation behavior with characterization tests around `TranslationRequestOwner`.
- [ ] Add `TranslationBackendRequest` and `TranslationBackend`.
- [ ] Add `LlmTranslationBackend` as a thin adapter over existing `LLMProvider`.
- [ ] Change the orchestrator to depend on `TranslationBackend`, not `LLMProvider`.
- [ ] Preserve provider generation/stale-completion behavior.
- [ ] Run current translation regression tests before adding HTTP behavior.

Required invariant before continuing:

> With only the LLM adapter active, existing LLM translation behavior has no intended product change.

## 2. Add Extension Schema v1 and deterministic parsing

- [ ] Add immutable typed extension definition objects.
- [ ] Add JSON parser and manual/type validation without a new general JSON-schema dependency unless already present and clearly appropriate.
- [ ] Add fixed placeholder validation/substitution.
- [ ] Add JSON/form/none body rendering.
- [ ] Add response text/JSON Pointer extraction.
- [ ] Add language mapping.
- [ ] Add secret declaration/reference validation.
- [ ] Reject duplicate IDs and invalid transport-managed headers.

Keep this layer independent of Flet and independent of real network calls so most behavior is pure/unit-testable.

## 3. Add extension paths, registry, and LibreTranslate example

- [ ] Add `default_translation_extensions_dir()`.
- [ ] Add non-recursive `*.json` registry loading.
- [ ] Preserve valid files when another file is invalid.
- [ ] Add stable non-secret definition fingerprints.
- [ ] Commit `examples/translation_extensions/libretranslate.json`.
- [ ] Validate the committed example in automated tests.
- [ ] Do not auto-copy the example into user config.

## 4. Add HTTP runtime backend and lifecycle

- [ ] Add `HttpExtensionTranslationBackend` using existing `httpx`.
- [ ] Apply POST-only, timeout, TLS, redirect, status, cancellation, and close policy.
- [ ] Integrate declared secret resolution without putting secrets into signatures/logs.
- [ ] Apply the effective translation concurrency policy.
- [ ] Generalize the translation runtime factory.
- [ ] Reuse `ProviderRuntimeHandle` for LLM and HTTP generations.
- [ ] Rebuild on active extension fingerprint changes.
- [ ] Prove retired HTTP backends cannot publish late results.

## 5. Add settings schema/migration and product selection

- [ ] Add canonical `custom_http` translation-model/engine alias.
- [ ] Add `translation.extension_id`.
- [ ] Preserve all inactive LLM configuration when switching.
- [ ] Preserve fallback while making it inactive under Custom HTTP.
- [ ] Add migration from the current schema with behavior-compatible defaults.
- [ ] Update runtime signature resolution for Custom HTTP.

## 6. Add Settings UI detail card

- [ ] Add `Custom HTTP API` to the existing translation engine selector.
- [ ] Add the conditional detail card.
- [ ] Populate extension dropdown from registry snapshot.
- [ ] Render dynamic credential inputs from declared secrets.
- [ ] Add extension directory display.
- [ ] Add `Open folder`.
- [ ] Add explicit `Reload`.
- [ ] Hide/disable irrelevant LLM connection/fallback/local controls while preserving state.
- [ ] Add localization following existing Settings conventions.
- [ ] Do not add trust confirmation.
- [ ] Do not add an HTTP request editor.

## 7. Integrate, harden, and package

- [ ] Run complete existing translation/provider/settings test suites.
- [ ] Add local fake-server integration coverage for HTTP behavior.
- [ ] Verify Windows user-config path and folder opening.
- [ ] Verify packaged build can discover user extension files.
- [ ] Ensure the LibreTranslate example remains available in repository/distribution documentation as intended.
- [ ] Verify clean shutdown with active HTTP requests/client lifecycle.
- [ ] Verify no secret/request/response leakage in normal diagnostics.

---

# Review plan

Review must **not** stop after every implementation section above.

Group review around coherent architectural/product concerns so a reviewer can reason about complete behavior instead of isolated scaffolding.

## Review unit A — Translation boundary and compatibility

Review together:

- `TranslationBackend` contract;
- `LlmTranslationBackend` adapter;
- `TranslationRequestOwner` changes;
- bounded runtime naming/signature generalization; and
- LLM characterization/regression evidence.

Primary review questions:

- Is the new boundary genuinely translation-neutral?
- Did any LLM behavior change accidentally?
- Is runtime generation/stale-result ownership preserved?
- Did the implementation avoid a broad unrelated `LLM -> translation` rename?

This unit may include code from implementation steps 1, 4, and 5 if that is what produces a coherent boundary. Do not force review to follow implementation-number boundaries.

## Review unit B — Extension contract and deterministic data path

Review together:

- Extension Schema v1;
- parser/validation;
- placeholder rendering;
- language mapping;
- secret declarations;
- JSON Pointer extraction;
- extension registry;
- fingerprint semantics; and
- the committed LibreTranslate example.

Primary review questions:

- Is the file contract small and deterministic?
- Is arbitrary execution impossible by construction?
- Are malformed/duplicate extensions isolated safely?
- Can the example be understood and copied by an extension author?
- Does the contract avoid unnecessary HTTP-framework features?

This is primarily pure logic and should have dense unit-test coverage.

## Review unit C — HTTP runtime, lifecycle, and privacy

Review together:

- `HttpExtensionTranslationBackend`;
- `httpx.AsyncClient` ownership;
- timeout/status/redirect/TLS policy;
- cancellation;
- concurrency;
- SecretStore integration;
- runtime replacement on extension changes; and
- privacy-safe diagnostics.

Primary review questions:

- Can a retired backend leak tasks/connections or publish late results?
- Are credentials kept out of settings, signatures, logs, and errors?
- Are HTTP semantics bounded enough for #33?
- Are failure categories actionable without logging private translation data?

## Review unit D — Settings product surface and release behavior

Review together:

- settings schema/migration;
- `custom_http` engine selection;
- conditional detail card;
- extension selector;
- dynamic credential inputs;
- `Open folder` and `Reload`;
- fallback/LLM-state preservation;
- localization;
- Windows/package evidence; and
- end-to-end fake-server verification.

Primary review questions:

- Does Custom HTTP feel like one translation engine option, analogous to Local LLM?
- Is the UI smaller than an HTTP editor and understandable without exposing implementation internals?
- Does switching away and back preserve inactive LLM settings?
- Does reload behave predictably when files change/disappear/break?
- Does the packaged Windows application use the expected user config directory?

## Review cadence rule

Implementation may continue across the numbered sequence until one of the review units above becomes coherent enough to review.

Do not create an artificial review pause after each implementation step.

Prefer a small number of reviewable batches with clear ownership and tests over many tiny gate-shaped reviews. Keep commits logically separable enough to debug or revert, but optimize review around complete reasoning units.

---

# Required automated tests

## Translation abstraction / LLM parity

Cover:

- existing LLM request mapping through `LlmTranslationBackend`;
- prompt/instruction mapping;
- context forwarding;
- source/target language forwarding;
- existing translation normalization;
- cancellation;
- stale-generation result rejection;
- existing fallback/hedging parity when LLM is active.

## Extension parser and validation

Cover:

- minimal valid extension;
- full valid extension;
- unsupported `schema_version`;
- invalid/missing `id`;
- duplicate extension IDs;
- missing required fields;
- invalid URL scheme;
- user-info embedded in URL;
- blocked transport headers;
- invalid body types;
- invalid response types;
- JSON response without pointer;
- duplicate/invalid secret IDs;
- undeclared secret placeholders;
- unknown placeholders;
- invalid language-map shapes.

## Renderer

Cover text containing:

- Korean;
- Japanese;
- Chinese;
- quotes;
- backslashes;
- newlines;
- emoji;
- braces that are not valid placeholders.

Cover placeholders in:

- headers;
- query;
- nested JSON string leaves;
- form values.

Prove JSON mode uses structured serialization rather than string concatenation.

## JSON Pointer

Cover:

- root pointer;
- nested objects;
- array indices;
- `~0` escaping;
- `~1` escaping;
- missing path;
- invalid array index;
- malformed pointer;
- non-string extracted value;
- invalid JSON response.

## Registry

Cover:

- empty directory;
- directory creation;
- one valid file;
- multiple valid files;
- one valid + one invalid;
- duplicate IDs;
- filename rename with same ID;
- definition fingerprint changes;
- unchanged reload;
- deletion of selected extension;
- selected extension becoming invalid;
- non-JSON files ignored;
- nested directories ignored.

## LibreTranslate example

Parse and validate the committed sample with production code.

Use a fake HTTP server to verify that the example renders conceptually as:

```json
{
  "q": "Hello",
  "source": "en",
  "target": "es",
  "format": "text",
  "api_key": "<secret>"
}
```

and extracts:

```json
{
  "translatedText": "Hola"
}
```

without contacting `libretranslate.com` in automated tests.

## HTTP backend

Cover:

- POST JSON;
- POST form;
- POST with no body;
- query parameters;
- headers;
- 2xx success;
- 400/401/403/429/500 status handling;
- timeout;
- connection failure;
- TLS failure where testable without disabling verification;
- redirect is not silently followed;
- empty response translation;
- invalid JSON;
- missing response pointer;
- cancellation;
- clean `AsyncClient` close;
- no retry behavior.

## Secret/privacy

Cover:

- missing secret fails before network I/O;
- SecretStore key names are namespaced by extension ID;
- secret value absent from settings serialization;
- secret value absent from runtime signatures;
- secret value absent from validation/runtime error strings;
- secret value absent from diagnostics/log capture;
- rendered source text/body/header values absent from diagnostics/log capture.

## Runtime replacement

Cover:

```text
LLM -> HTTP
HTTP -> LLM
HTTP extension A -> HTTP extension B
HTTP definition reload -> new HTTP generation
HTTP active extension deleted -> unavailable/degraded state
```

For every replacement:

- old backend closes;
- old completion cannot publish;
- no client/task leak remains;
- new generation owns subsequent requests.

## Settings/migration

Cover:

- current schema -> new schema migration;
- existing LLM model unchanged by migration;
- `extension_id` default;
- Custom HTTP selection round-trip;
- switching to Custom HTTP preserves previous LLM configuration;
- switching back restores previous LLM configuration;
- fallback value preserved but inactive under HTTP;
- compatibility projection/round-trip expectations;
- migration idempotence.

## Settings UI

Cover:

- `Custom HTTP API` visible in translation engine options;
- selecting it shows Custom HTTP card;
- selecting it hides/disables relevant LLM detail controls;
- switching back restores normal LLM detail controls;
- valid registry entries populate dropdown by name/id;
- dynamic credential rows match extension declarations;
- no-credential extension state;
- no-extension state;
- invalid-extension reload notification;
- Open folder resolves the canonical directory;
- Reload refreshes registry without network call;
- no trust-confirmation surface;
- no HTTP request editor surface;
- localization.

---

# Required local / Windows evidence

At one exact candidate SHA:

1. Start from an existing LLM configuration and record the selected model/connection/fallback.
2. Verify existing LLM translation still works.
3. Open Settings and confirm `Custom HTTP API` appears beside the other translation-engine choices.
4. Select Custom HTTP and confirm the dedicated detail card appears while LLM-specific detail controls become inactive/hidden as designed.
5. Use `Open folder` and confirm it opens the resolved Windows path:

   ```text
   %LOCALAPPDATA%\puripuly-heart\translation_extensions\
   ```

6. Copy a test extension JSON into the directory and press `Reload`.
7. Confirm the extension appears without restarting PuriPuly.
8. Confirm required credential fields are generated from the extension definition.
9. Use a local loopback fake translation server to perform a real PuriPuly translation through the HTTP backend.
10. Edit the active extension, press `Reload`, and confirm the backend generation is replaced cleanly.
11. Break a second extension file and confirm valid extensions still load.
12. Delete or invalidate the selected extension and confirm PuriPuly does not silently select another backend.
13. Switch back to the previous LLM engine and confirm the prior LLM model/connection/fallback state is preserved.
14. Close PuriPuly with the HTTP backend active and confirm no owned HTTP client/task remains.
15. Inspect logs/evidence and confirm no credential or private translated/source text was captured.

Do not use paid/public translation requests as required release evidence. The included LibreTranslate file is a real-service example, while acceptance testing remains deterministic through a local fake server.

---

# Acceptance criteria

- [ ] `Custom HTTP API` is a first-class option in the existing translation engine selector, alongside `Local LLM` and current LLM choices.
- [ ] Selecting Custom HTTP reveals a dedicated detail card rather than an HTTP request editor.
- [ ] The detail card supports extension selection, dynamic credentials, resolved directory display, `Open folder`, and `Reload`.
- [ ] No endpoint trust-confirmation dialog is added.
- [ ] User extensions are loaded from `<user_config_dir>/translation_extensions/*.json`.
- [ ] One extension is one `.json` file; JSONL is not used.
- [ ] Registry loading is non-recursive and isolates invalid files.
- [ ] Extension IDs, not filenames, are persisted as identity.
- [ ] A real LibreTranslate example is committed under `examples/translation_extensions/libretranslate.json`.
- [ ] The LibreTranslate example is not auto-copied to user configuration and never triggers startup/migration network traffic.
- [ ] Extension Schema v1 is POST-only.
- [ ] Schema v1 supports headers, query, JSON/form/no-body requests, fixed placeholders, optional language maps, declared secrets, and text/JSON-Pointer responses.
- [ ] Schema v1 does not support arbitrary scripts, Jinja, JSONPath, regex response scraping, raw-body scripting, environment access, or file access.
- [ ] Secrets are stored only through SecretStore under extension-scoped keys.
- [ ] Secret values never enter settings, runtime signatures, logs, diagnostics, or telemetry payloads.
- [ ] `TranslationRequestOwner` no longer imports/casts directly to `LLMProvider`.
- [ ] The HTTP translator is not implemented as an `LLMProvider` compatibility trick.
- [ ] Existing LLM providers continue through `LlmTranslationBackend` with no intended behavior change.
- [ ] The existing provider runtime generation/replacement owner is reused for both LLM and HTTP translation backends.
- [ ] Switching/reloading closes retired HTTP clients and stale results cannot publish.
- [ ] Custom HTTP does not create unused LLM/OpenRouter runtime resources.
- [ ] LLM fallback/hedging remains unchanged when an LLM engine is active.
- [ ] No automatic HTTP-to-LLM fallback exists in #33.
- [ ] Switching to Custom HTTP and back preserves the user's inactive LLM configuration and fallback selection.
- [ ] Migration from the current settings schema is idempotent and behavior-compatible for existing users.
- [ ] Existing `httpx` is reused; no additional HTTP client dependency is introduced.
- [ ] Automated HTTP integration tests use a local fake server and do not require external service availability.
- [ ] Windows packaged behavior uses the canonical user config directory and clean runtime shutdown.

---

# Non-goals

- No built-in hard-coded Google Translate provider.
- No built-in hard-coded DeepL provider.
- No built-in hard-coded Papago provider.
- No provider marketplace or remote extension registry.
- No automatic extension download/update mechanism.
- No automatic copy/install of the LibreTranslate example into user config.
- No file watcher/hot-reload daemon.
- No arbitrary method selection in Extension Schema v1.
- No GET support in v1.
- No raw arbitrary request body in v1.
- No custom timeout/retry/proxy/TLS-bypass configuration in extension files.
- No cookies/session scripting.
- No environment-variable or local-file interpolation.
- No Jinja/JavaScript/Python/shell execution.
- No JSONPath/jq/regex response transforms.
- No multiple simultaneous HTTP translation backends.
- No HTTP-to-LLM or LLM-to-HTTP cross-backend fallback.
- No general-purpose PuriPuly plugin framework.
- No Settings HTTP request builder/test console in #33.
- No endpoint trust-confirmation prompt.

---

# Risks and tradeoffs

## Extension files move complexity out of Settings UI

The file-based approach removes a large amount of UI state, validation, persistence, and draft-editing complexity. In exchange, it introduces a registry/parser contract and filesystem error handling.

This is an intentional trade: registry/parsing behavior is substantially easier to unit test and version than a large dynamic HTTP editor.

## The extension schema becomes a compatibility surface

Once users share extension JSON files, field names and semantics become externally observable.

Keep Schema v1 deliberately small. Add new capabilities through explicit schema evolution instead of silently expanding placeholder/expression behavior.

## POST-only excludes some possible services

POST-only meaningfully reduces the request matrix. If a real translation API later requires GET, add it based on evidence rather than preemptively broadening v1.

## Fixed placeholders reduce flexibility

Not every imaginable API can be represented, but fixed placeholders prevent the feature from becoming an executable templating engine and make security, logging, and testing tractable.

## File edits are advanced-user UX

The normal PuriPuly UI remains simple, but authoring a new extension requires editing JSON. This is consistent with the referenced VRHandsFrame extension philosophy and the maintenance goal of issue #33.

The committed LibreTranslate file provides a concrete starting point.

## Real-service example can drift

LibreTranslate may evolve independently of PuriPuly. Keep the sample small, cite/document the upstream service, and validate only schema-level behavior in tests. Do not require public-service availability for CI or release acceptance.

---

# Stop / narrow conditions

Stop and narrow/escalate before merging if any of the following becomes necessary:

- credentials would need to be written into canonical `settings.json`;
- Custom HTTP branches spread into translation orchestration/output/history instead of staying behind `TranslationBackend`;
- the HTTP backend cannot reuse current runtime generation/replacement ownership;
- LLM behavior changes before HTTP functionality is enabled;
- implementing the extension contract requires arbitrary script execution;
- a required service can only work by exposing environment variables or local-file contents to extension templates;
- the feature requires unapproved automatic HTTP-to-LLM fallback;
- the feature requires multiple active HTTP profiles simultaneously rather than one selected extension;
- an old HTTP backend can publish after replacement/reload;
- reloading extensions requires implicit settings persistence or a network request;
- adding `custom_http` to the existing translation selector would require unrelated redesign of STT/provider ownership;
- packaged builds cannot resolve the same canonical user extension directory as development builds;
- supporting a real-world required service would materially exceed the bounded Schema v1 contract.

In those cases, open/plan a follow-up design decision instead of silently broadening #33.

---

# Definition of Done

The issue is ready to close when:

1. the provider-neutral translation boundary is active for existing LLM translation;
2. LLM behavior remains regression-compatible;
3. `Custom HTTP API` is selectable as a normal translation engine;
4. the dedicated extension detail card works with the canonical user extension directory;
5. Extension Schema v1 is implemented, documented, deterministic, and non-executable;
6. the real LibreTranslate example is committed and validated;
7. SecretStore-backed dynamic credentials work without persistence/log leakage;
8. the HTTP backend executes through owned `httpx.AsyncClient` lifecycle with cancellation and stale-result safety;
9. extension reload/replacement behavior is deterministic;
10. switching between HTTP and LLM preserves inactive LLM configuration;
11. settings migration is compatible and idempotent;
12. required unit/integration/UI tests pass;
13. Windows/package smoke evidence is clean; and
14. review has completed across the coherent review units above rather than requiring a separate review stop after every implementation phase.
