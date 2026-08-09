_HTTP_EXTENSION_SECRET_PREFIX = "http_extension."


def http_extension_secret_key(http_extension_id: str, secret_id: str) -> str:
    return f"{http_extension_secret_key_prefix(http_extension_id)}{_encode_segment(secret_id)}"


def http_extension_secret_key_prefix(http_extension_id: str) -> str:
    return f"{_HTTP_EXTENSION_SECRET_PREFIX}{_encode_segment(http_extension_id)}."


def _encode_segment(value: str) -> str:
    return value.replace("%", "%25").replace(".", "%2E")


__all__ = [
    "http_extension_secret_key",
    "http_extension_secret_key_prefix",
]
