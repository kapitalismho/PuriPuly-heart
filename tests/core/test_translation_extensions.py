from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from puripuly_heart.core.translation_backend import TranslationBackendRequest
from puripuly_heart.core.translation_extensions import (
    TranslationExtensionConfigurationError,
    TranslationExtensionRegistry,
    TranslationExtensionResponseError,
    TranslationExtensionValidationError,
    extract_translation_text,
    parse_translation_extension,
    render_translation_request,
    resolve_json_pointer,
    translation_extension_secret_key,
    translation_extension_secret_key_prefix,
)


def extension_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": 1,
        "id": "sample",
        "name": "Sample",
        "url": "https://example.test/translate",
        "request": {
            "query": {},
            "body": {"type": "json", "value": {"text": "{{text}}"}},
        },
        "response": {"type": "text"},
    }
    data.update(overrides)
    return data


def backend_request(**overrides: object) -> TranslationBackendRequest:
    values: dict[str, object] = {
        "utterance_id": uuid4(),
        "text": "안녕 {name}",
        "system_prompt": "unused",
        "source_language": "ko",
        "target_language": "en",
        "context": "unused",
    }
    values.update(overrides)
    return TranslationBackendRequest(**values)


def test_parse_minimal_extension() -> None:
    extension = parse_translation_extension(extension_data())

    assert extension.id == "sample"
    assert extension.request.body.type == "json"
    assert extension.response.type == "text"
    assert extension.fingerprint == extension.fingerprint


def test_parse_full_extension_and_render_nested_json() -> None:
    extension = parse_translation_extension(
        extension_data(
            description="description",
            headers={"X-Key": "{{secret:key}}"},
            request={
                "query": {"target": "{{target_language}}"},
                "body": {
                    "type": "json",
                    "value": {
                        "q": "{{text}}",
                        "source": "{{source_language}}",
                        "nested": ["{{secret:key}}", 1],
                    },
                },
            },
            response={"type": "json", "pointer": "/data/0/text"},
            secrets=[{"id": "key", "label": "API Key"}],
            language_map={"source": {"ko": "kr"}, "target": {"en": "eng"}},
        )
    )

    url, headers, query, body = render_translation_request(
        extension,
        backend_request(),
        secrets={"key": "secret-value"},
    )

    assert url == "https://example.test/translate"
    assert headers == {"X-Key": "secret-value"}
    assert query == {"target": "eng"}
    assert body == {
        "q": "안녕 {name}",
        "source": "kr",
        "nested": ["secret-value", 1],
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.update(schema_version=2), "schema_version"),
        (lambda data: data.update(method="GET"), "unsupported extension field"),
        (lambda data: data.update(url="ftp://example.test/translate"), "HTTP"),
        (
            lambda data: data.update(url="https://user:pass@example.test/translate"),
            "credentials",
        ),
        (
            lambda data: data.update(headers={"Host": "example.test"}),
            "transport-managed",
        ),
        (
            lambda data: data.update(
                request={"body": {"type": "json", "value": {"x": "{{unknown}}"}}}
            ),
            "unsupported placeholder",
        ),
        (
            lambda data: data.update(
                request={"body": {"type": "json", "value": {"x": "{{secret:key}}"}}}
            ),
            "not declared",
        ),
        (
            lambda data: data.update(
                request={"body": {"type": "json", "value": {"x": "{{text}}"}}},
                secrets=[{"id": "key", "label": "Key"}],
            ),
            "unused",
        ),
    ],
)
def test_invalid_extension_is_rejected(change, message: str) -> None:
    data = extension_data()
    change(data)

    with pytest.raises(TranslationExtensionValidationError, match=message):
        parse_translation_extension(data)


@pytest.mark.parametrize(
    ("field", "raw_value"),
    [("body_type", []), ("body_type", {}), ("response_type", []), ("response_type", {})],
)
def test_unhashable_schema_discriminators_are_validation_errors(
    field: str,
    raw_value: object,
) -> None:
    data = extension_data()
    if field == "body_type":
        data["request"] = {"body": {"type": raw_value, "value": {"text": "{{text}}"}}}
    else:
        data["response"] = {"type": raw_value}

    with pytest.raises(TranslationExtensionValidationError):
        parse_translation_extension(data)


def test_invalid_placeholder_does_not_echo_template_value() -> None:
    with pytest.raises(
        TranslationExtensionValidationError, match="unsupported placeholder"
    ) as error:
        parse_translation_extension(
            extension_data(
                request={
                    "body": {
                        "type": "json",
                        "value": {"key": "{{sk-sensitive-value}}"},
                    }
                }
            )
        )

    assert "sk-sensitive-value" not in str(error.value)


@pytest.mark.parametrize("body_text", ["{{text}}}", "{{text", "text}}"])
def test_unbalanced_placeholder_delimiters_are_rejected(body_text: str) -> None:
    with pytest.raises(TranslationExtensionValidationError, match="placeholder syntax"):
        parse_translation_extension(
            extension_data(request={"body": {"type": "json", "value": {"text": body_text}}})
        )


def test_form_and_no_body_extensions() -> None:
    form = parse_translation_extension(
        extension_data(request={"body": {"type": "form", "value": {"q": "{{text}}"}}})
    )
    empty = parse_translation_extension(
        extension_data(request={"body": {"type": "none", "value": {"ignored": True}}})
    )

    assert form.request.body.value == {"q": "{{text}}"}
    assert empty.request.body.value is None


def test_missing_secret_fails_without_exposing_secret_value() -> None:
    extension = parse_translation_extension(
        extension_data(
            request={"body": {"type": "json", "value": {"key": "{{secret:key}}"}}},
            secrets=[{"id": "key", "label": "API Key"}],
        )
    )

    with pytest.raises(TranslationExtensionConfigurationError, match="missing required credential"):
        render_translation_request(extension, backend_request(), secrets={})


def test_secret_store_keys_escape_dotted_segments_without_collisions() -> None:
    assert translation_extension_secret_key("libretranslate", "api_key") == (
        "translation_extension.libretranslate.api_key"
    )
    first = translation_extension_secret_key("a.b", "c")
    second = translation_extension_secret_key("a", "b.c")

    assert first == "translation_extension.a%2Eb.c"
    assert second == "translation_extension.a.b%2Ec"
    assert first != second
    assert translation_extension_secret_key_prefix("a.b") == "translation_extension.a%2Eb."


def test_json_pointer_supports_root_arrays_and_escaping() -> None:
    assert resolve_json_pointer({"value": "ok"}, "/value") == "ok"
    assert resolve_json_pointer([{"value": "ok"}], "/0/value") == "ok"
    assert resolve_json_pointer({"a/b~c": "ok"}, "/a~1b~0c") == "ok"
    assert resolve_json_pointer("root", "") == "root"


@pytest.mark.parametrize("pointer", ["value", "/bad~2escape"])
def test_invalid_json_pointer_is_rejected(pointer: str) -> None:
    with pytest.raises(TranslationExtensionValidationError):
        parse_translation_extension(extension_data(response={"type": "json", "pointer": pointer}))


def test_response_extraction_rejects_invalid_results() -> None:
    text_extension = parse_translation_extension(extension_data())
    json_extension = parse_translation_extension(
        extension_data(response={"type": "json", "pointer": "/text"})
    )

    assert extract_translation_text(text_extension, " Hola ") == "Hola"
    assert extract_translation_text(json_extension, '{"text":"Hola"}') == "Hola"
    with pytest.raises(TranslationExtensionResponseError, match="empty"):
        extract_translation_text(text_extension, "  ")
    with pytest.raises(TranslationExtensionResponseError, match="invalid response JSON"):
        extract_translation_text(json_extension, "broken")
    with pytest.raises(TranslationExtensionResponseError, match="missing"):
        extract_translation_text(json_extension, "{}")
    with pytest.raises(TranslationExtensionResponseError, match="string"):
        extract_translation_text(json_extension, '{"text":1}')


def test_invalid_response_json_does_not_chain_private_response_data() -> None:
    extension = parse_translation_extension(
        extension_data(response={"type": "json", "pointer": "/text"})
    )
    private_response = '{"text":"private-response",'

    with pytest.raises(TranslationExtensionResponseError) as error:
        extract_translation_text(extension, private_response)

    assert str(error.value) == "invalid response JSON"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    assert private_response not in repr(error.value)


def test_registry_isolates_invalid_files_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    valid = extension_data()
    other = extension_data(id="other", name="Other")
    (tmp_path / "valid.json").write_text(json.dumps(valid), encoding="utf-8")
    (tmp_path / "other.json").write_text(json.dumps(other), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "duplicate.json").write_text(json.dumps(valid), encoding="utf-8")
    (tmp_path / "unhashable-body.json").write_text(
        json.dumps(
            extension_data(
                id="unhashable-body",
                request={"body": {"type": [], "value": {"text": "{{text}}"}}},
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "unhashable-response.json").write_text(
        json.dumps(extension_data(id="unhashable-response", response={"type": {}})),
        encoding="utf-8",
    )
    (tmp_path / "ignored.txt").write_text("not json", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "nested.json").write_text(
        json.dumps(extension_data(id="nested")), encoding="utf-8"
    )

    snapshot = TranslationExtensionRegistry(tmp_path).reload()

    assert [item.definition.id for item in snapshot.extensions] == ["other"]
    assert {item.source_path.name for item in snapshot.errors} == {
        "broken.json",
        "duplicate.json",
        "valid.json",
        "unhashable-body.json",
        "unhashable-response.json",
    }


def test_registry_identity_and_fingerprint_survive_filename_changes(tmp_path: Path) -> None:
    source = tmp_path / "first.json"
    source.write_text(json.dumps(extension_data()), encoding="utf-8")
    registry = TranslationExtensionRegistry(tmp_path)

    first = registry.reload().get("sample")
    assert first is not None
    first_fingerprint = first.fingerprint

    renamed = tmp_path / "renamed.json"
    source.rename(renamed)
    unchanged = registry.reload().get("sample")
    assert unchanged is not None
    assert unchanged.source_path == renamed
    assert unchanged.fingerprint == first_fingerprint

    changed = extension_data(description="changed")
    renamed.write_text(json.dumps(changed), encoding="utf-8")
    updated = registry.reload().get("sample")
    assert updated is not None
    assert updated.fingerprint != first_fingerprint

    renamed.unlink()
    assert registry.reload().get("sample") is None


def test_example_is_valid() -> None:
    path = Path("examples/translation_extensions/libretranslate.json")

    extension = parse_translation_extension(json.loads(path.read_text(encoding="utf-8")))

    assert extension.id == "libretranslate"
    assert extension.response.pointer == "/translatedText"
