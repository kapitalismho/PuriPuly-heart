# HTTP extensions

PuriPuly can translate through any HTTP translation API using a declarative JSON extension. Extensions are loaded from the user's `http_extensions` directory, so you can add a new API without a code change.

- Windows: `%LOCALAPPDATA%\puripuly-heart\http_extensions`
- A JSON Schema for editor validation lives next to this file: `http-extension.schema.json`
- A working reference example ships in the repository under `examples/http_extensions/mymemory.json`

Drop a `.json` file into the directory, open the settings UI's HTTP extension card, and press **Reload**. Each file must contain exactly one extension; the file name does not matter, but the `id` must be unique among loaded extensions.

## Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `schema_version` | `integer` | yes | Must be `1`. |
| `id` | `string` | yes | Lowercase identifier: `[a-z0-9][a-z0-9._-]{0,63}`. Unique within the directory; used to persist the selected extension and its credentials. |
| `name` | `string` | yes | Display name in the settings UI. |
| `description` | `string` | no | Description shown in the settings UI. |
| `url` | `string` | yes | Absolute HTTP(S) endpoint. Always called with **POST**. Must not contain credentials or placeholders. |
| `headers` | `object` | no | Static request headers (string to string). Transport-managed headers are forbidden (case-insensitive): `connection`, `content-length`, `host`, `transfer-encoding`. |
| `request` | `object` | yes | See [Request](#request). |
| `response` | `object` | yes | See [Response](#response). |
| `secrets` | `array` | no | See [Secrets](#secrets). |
| `language_map` | `object` | no | See [Language map](#language-map). |

Unknown fields are rejected.

## Request

`request` contains `query` (optional) and `body` (required).

`query` is an object of static query parameters sent with every request.

`body.type` is one of:

- `json` — the request body is `Content-Type: application/json`; `body.value` may be any JSON value.
- `form` — the request body is `Content-Type: application/x-www-form-urlencoded`; `body.value` must be a string-to-string object.
- `none` — no request body; `body.value` must be omitted.

`body.value` is a template. Strings may contain placeholders:

| Placeholder | Meaning |
| --- | --- |
| `{{text}}` | The text to translate. |
| `{{source_language}}` | Source language code after language-map remapping. |
| `{{target_language}}` | Target language code after language-map remapping. |
| `{{secret:<id>}}` | Value of the declared secret `<id>`. |

Placeholders may appear in `headers`, `request.query`, and `request.body.value`. Every declared secret must be referenced somewhere.

## Response

`response.type` is one of:

- `json` — parse the response body as JSON and select the translated string with `response.pointer`, an RFC 6901 JSON Pointer. `~0` escapes `~` and `~1` escapes `/`; array indices (and `-`-style patterns aside) are numeric, and empty strings or non-string values at the pointer are errors.
- `text` — the raw response body (stripped) is the translation; `response.pointer` must be omitted.

The translated text must be non-empty.

## Secrets

`secrets` is an array of `{ "id", "label" }`. Each entry declares a credential the user fills in the settings UI; values are stored through the system credential store, not in the extension file. `id` must match the identifier pattern and be unique; `label` is the input label shown in the UI.

An extension without credentials declares an empty array (or omits the field):

```json
"secrets": []
```

## Language map

`language_map` maps PuriPuly language codes to API language codes so an API with a different code scheme can be used without changing the app. Both `source` and `target` are optional string-to-string objects; codes without an entry pass through unchanged.

```json
"language_map": {
  "source": { "ko": "kor", "en": "eng" },
  "target": { "ja": "jpn", "en": "eng" }
}
```

## Example

```json
{
  "schema_version": 1,
  "id": "mymemory",
  "name": "MyMemory",
  "description": "MyMemory free translation API",
  "url": "https://api.mymemory.translated.net/get",
  "request": {
    "query": {},
    "body": {
      "type": "form",
      "value": {
        "q": "{{text}}",
        "langpair": "{{source_language}}|{{target_language}}",
        "mt": "1"
      }
    }
  },
  "response": {
    "type": "json",
    "pointer": "/responseData/translatedText"
  },
  "secrets": []
}
```

## Behavior and limits

- Every request is a POST with a 10 second timeout and at most 5 concurrent in-flight requests.
- Non-2xx responses, transport errors, malformed response bodies, pointer misses, and empty results surface as translation errors instead of silent fallbacks.
- Test your extension against a local mock server or a free API; PuriPuly never sends sample requests to validate an extension.
