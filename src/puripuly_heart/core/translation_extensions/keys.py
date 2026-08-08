_TRANSLATION_EXTENSION_SECRET_PREFIX = "translation_extension."


def translation_extension_secret_key(extension_id: str, secret_id: str) -> str:
    return f"{translation_extension_secret_key_prefix(extension_id)}{_encode_segment(secret_id)}"


def translation_extension_secret_key_prefix(extension_id: str) -> str:
    return f"{_TRANSLATION_EXTENSION_SECRET_PREFIX}{_encode_segment(extension_id)}."


def _encode_segment(value: str) -> str:
    return value.replace("%", "%25").replace(".", "%2E")


__all__ = [
    "translation_extension_secret_key",
    "translation_extension_secret_key_prefix",
]
