from __future__ import annotations

import re

import httpx

_CF_RAY = re.compile(r"[0-9a-fA-F]{16}(?:-[A-Z]{3})?")
_HTML_START = re.compile(rb"(?:<!doctype\s+html\b|<(?:html|head|body|title)(?=[\s>/]))", re.I)


def is_cf_ray(value: object) -> bool:
    return type(value) is str and _CF_RAY.fullmatch(value) is not None


def response_media_type(response: httpx.Response) -> str:
    return response.headers.get("content-type", "").partition(";")[0].strip().lower()


def response_evidence(response: httpx.Response, *, body: bytes = b"") -> dict[str, str]:
    fields = {}
    ray = response.headers.get("cf-ray", "")
    if is_cf_ray(ray):
        fields["cf_ray"] = ray
    if response.headers.get("cf-mitigated", "").strip().lower() == "challenge":
        fields["response_shape"] = "edge_challenge"
    elif response_media_type(response) in {"text/html", "application/xhtml+xml"} or (
        _HTML_START.match(body[:512].removeprefix(b"\xef\xbb\xbf").lstrip())
    ):
        fields["response_shape"] = "html"
    return fields
